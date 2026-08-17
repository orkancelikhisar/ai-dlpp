import { Worker } from "node:worker_threads";
import { hasValidator, type EntityType, type Rule } from "@sih/core";
import { SHADOW_PREFIX } from "./predicates.js";

/**
 * Stage 3: prove the artifacts the model authored are safe to execute.
 *
 * This is the highest-risk stage in the compiler. The model writes the regexes,
 * and a catastrophic-backtracking pattern is not a wrong answer — it is a hang.
 * Tier-0 detection runs on the send path of every message inside a browser
 * extension, where the user has no way to interrupt a synchronous regex, so a
 * ReDoS pattern that reaches the IR takes the tab with it. Everything here fails
 * closed: a rule that cannot be proven safe is rejected, not shipped with a note.
 */

/** A rule as the extract stage produces it; `sourceQuote` is carried, not read. */
export type RuleInput = Rule & { readonly sourceQuote?: string | undefined };

/** An entityType as the extract stage produces it, or as `mintShadowEntityTypes` mints one. */
export type EntityTypeInput = EntityType & { readonly sourceQuote?: string | undefined };

// -- regex safety -----------------------------------------------------------

/**
 * Runs the pattern against each adversarial input and reports back.
 *
 * A plain JS source string evaluated with `eval: true`, deliberately: the worker
 * needs no TS loader, no build step, and no path that could resolve differently
 * under vitest than under the CLI. It uses `require` rather than an import so
 * that Node classifies it as CommonJS regardless of the package's `type` field.
 *
 * `re.lastIndex = 0` between inputs because the regex is compiled with `g`,
 * matching how `runTier0` uses it — a sticky lastIndex would skip the start of
 * the next input and hide the very backtracking this exists to provoke.
 */
const WORKER_SOURCE = `
  const { parentPort, workerData } = require("node:worker_threads");
  const re = new RegExp(workerData.source, "g");
  for (const input of workerData.inputs) { re.lastIndex = 0; re.test(input); }
  parentPort.postMessage("ok");
`;

/**
 * Long enough that an exponential pattern blows past any sane budget, short
 * enough that a linear one finishes in single-digit milliseconds. 20k is also
 * the right order of magnitude for the real threat: a pasted log or code block.
 */
const ADVERSARIAL_LENGTH = 20_000;

/**
 * Appended to every adversarial input so the match must FAIL at the end. This is
 * the whole point: a catastrophic pattern is fast when it succeeds and
 * exponential when it is forced to exhaust every partition of the input, which
 * only happens on a failing match.
 */
const NON_MATCHING_SUFFIX = "!";

/** Characters that are regex syntax rather than something the pattern matches. */
const REGEX_SYNTAX = new Set([..."^$.*+?()[]{}|/"]);

/**
 * First character of `source` the pattern could plausibly consume.
 *
 * A heuristic, not a parse: it skips escapes and metacharacters and returns the
 * first survivor, so `^(a+)+$` yields "a" and `[A-Z]{5}` yields "A". Character
 * class contents are deliberately NOT skipped — a character inside a class is
 * exactly a character the pattern matches, which is what we want to feed it. It
 * can also return a quantifier digit (`\d{2,}` yields "2"), which is harmless:
 * a wrong guess costs one extra linear scan, and the fixed inputs below cover
 * the classic shapes regardless.
 */
function firstLiteralChar(source: string): string | undefined {
  for (let i = 0; i < source.length; i += 1) {
    const char = source[i]!;
    if (char === "\\") {
      i += 1; // skip the escape and whatever it escapes; \d and \w are not literals
      continue;
    }
    if (REGEX_SYNTAX.has(char)) continue;
    return char;
  }
  return undefined;
}

/**
 * The inputs a pattern is raced against: a long run of one character, and a long
 * run of an alternating pair, each ending in a character that breaks the match.
 * Between them these cover the shapes that make nested quantifiers explode.
 */
function adversarialInputs(source: string): string[] {
  const first = firstLiteralChar(source) ?? "a";
  const partner = first === "b" ? "a" : "b";
  const inputs = [
    "a".repeat(ADVERSARIAL_LENGTH),
    first.repeat(ADVERSARIAL_LENGTH),
    (first + partner).repeat(ADVERSARIAL_LENGTH / 2),
  ].map((body) => body + NON_MATCHING_SUFFIX);
  // Dedupe: for a pattern whose first literal is already "a" the first two are
  // identical, and a redundant 20k scan buys nothing.
  return [...new Set(inputs)];
}

/**
 * Runs the pattern in a worker, bounded by a wall clock.
 *
 * A worker rather than a timing heuristic because a synchronous regex CANNOT be
 * interrupted on the main thread: by the time you could measure how long it took,
 * it has already taken that long. `terminate()` on a separate thread is the only
 * mechanism in Node that actually stops a running match, which is why this stage
 * owns a worker at all.
 */
function runBounded(source: string, inputs: readonly string[], timeoutMs: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const worker = new Worker(WORKER_SOURCE, { eval: true, workerData: { source, inputs } });
    let timer: ReturnType<typeof setTimeout> | undefined;
    let settled = false;

    const finish = (error?: Error): void => {
      if (settled) return;
      settled = true;
      if (timer !== undefined) clearTimeout(timer);
      // Terminated on EVERY path, the success path included. A worker left
      // running keeps the whole process alive: in a test run that is a suite
      // that passes and then never exits, and in the CLI it is a hung command.
      void worker.terminate();
      if (error === undefined) resolve();
      else reject(error);
    };

    timer = setTimeout(() => {
      finish(
        new Error(
          `regex ${JSON.stringify(source)} timed out after ${timeoutMs}ms against adversarial input; ` +
            `it backtracks catastrophically and would hang the extension on the send path`,
        ),
      );
    }, timeoutMs);

    worker.on("message", () => finish());
    // The pattern already compiled on this thread, so the worker cannot fail on
    // construction; an error here is the match exhausting memory or stack, which
    // is the same finding as the timeout arriving by a different road.
    worker.on("error", (e) =>
      finish(
        new Error(
          `regex ${JSON.stringify(source)} exhausted resources while backtracking against ` +
            `adversarial input: ${(e as Error).message}`,
        ),
      ),
    );
    // Fails closed. `finish` is idempotent, so the exit that follows a normal
    // message or a terminate is ignored; an exit with no message at all means the
    // check never ran, and an unproven pattern must not pass.
    worker.on("exit", (code) =>
      finish(
        new Error(
          `bounded regex check for ${JSON.stringify(source)} exited (code ${code}) without a result`,
        ),
      ),
    );
  });
}

/**
 * Rejects a pattern that is invalid, matches nothing, or backtracks
 * catastrophically. Resolves — with no value — only when all three hold.
 *
 * The regex source IS echoed in these messages. That is not a convention breach:
 * a pattern is policy the model wrote, not a confidential value from a document,
 * and an unsafe-regex error the author cannot see the pattern in is unactionable.
 */
export async function checkRegexSafety(source: string, timeoutMs = 1000): Promise<void> {
  try {
    // "g" and not "u": tier-0 compiles the same way, and "u" rejects IR escapes
    // like \- that plain compilation accepts. Checking a stricter dialect than
    // the runtime uses would pass patterns the runtime then chokes on.
    new RegExp(source, "g");
  } catch (e) {
    throw new Error(`invalid regex ${JSON.stringify(source)}: ${(e as Error).message}`);
  }

  // Same gate `loadPolicyIr` applies, caught here where the compiler can say what
  // to do about it. A /g/ exec of a nullable pattern returns a zero-width match
  // without advancing lastIndex, so detection must skip the rule entirely: the
  // pattern loads, runs on every message, and matches nothing. An author who
  // believes that entity class is covered is exactly the fail-open this project
  // exists to prevent. Non-global on purpose — test() on a /g/ regex mutates
  // lastIndex.
  if (new RegExp(source).test("")) {
    throw new Error(
      `regex ${JSON.stringify(source)} can match the empty string, so it can never drive a scan ` +
        `and would silently detect nothing; require at least one character (e.g. "+" instead of "*")`,
    );
  }

  await runBounded(source, adversarialInputs(source), timeoutMs);
}

// -- rules ------------------------------------------------------------------

/**
 * Structural checks mirroring `RuleSchema`, run here so a malformed rule is
 * reported against the model's output rather than surfacing much later as a
 * loader rejection of the whole IR.
 *
 * The schema's numeric entropy ceiling is deliberately NOT duplicated here: the
 * constant lives in core, is not exported, and a second copy would drift silently
 * into either a false rejection or a false pass. `compilePolicy` round-trips its
 * own output through `loadPolicyIr` (Task 8), which is where that bound is owned.
 */
function checkVariant(rule: RuleInput): void {
  const isRegexRule = rule.regex !== undefined;
  const isEntropyRule = rule.entropyThreshold !== undefined;

  if (isRegexRule && isEntropyRule) {
    throw new Error(`rule "${rule.id}" cannot have both regex and entropyThreshold`);
  }
  if (!isRegexRule && !isEntropyRule) {
    throw new Error(`rule "${rule.id}" must have regex or entropyThreshold`);
  }
  if (!isRegexRule) {
    if (rule.validator !== undefined) {
      throw new Error(`rule "${rule.id}": validator is only valid on regex rules`);
    }
    if (rule.contextBoost !== undefined) {
      throw new Error(`rule "${rule.id}": contextBoost is only valid on regex rules`);
    }
  }
  if (!isEntropyRule && rule.minLength !== undefined) {
    throw new Error(`rule "${rule.id}": minLength is only valid on entropy rules`);
  }
}

/**
 * Validates every rule and returns them unchanged, so the stage composes as a
 * link in the pipeline rather than as a side effect.
 *
 * Throws on the first bad rule rather than collecting: the three failures here
 * (a hanging pattern, a pattern that matches nothing, an invented validator) are
 * each unshippable on their own, so there is no useful "compiled with warnings"
 * outcome to aggregate toward.
 */
export async function validateRules(rules: readonly RuleInput[]): Promise<readonly RuleInput[]> {
  for (const rule of rules) {
    // Cheap invariants first: a rule that fails one of these should not pay for
    // a worker spawn, and the diagnosis is clearer before the regex is raced.
    checkVariant(rule);

    // The fixed-validator-library invariant, enforced rather than trusted. The
    // extraction SYSTEM prompt names the four legal validators and forbids
    // inventing others, but a prompt is a request; this is the check. A rule
    // naming a validator core does not export would load and then throw at
    // detection time, mid-send.
    if (rule.validator !== undefined && !hasValidator(rule.validator)) {
      throw new Error(`rule "${rule.id}" names unknown validator "${rule.validator}"`);
    }

    if (rule.regex !== undefined) {
      try {
        await checkRegexSafety(rule.regex);
      } catch (e) {
        throw new Error(`rule "${rule.id}": ${(e as Error).message}`);
      }
    }
  }
  return rules;
}

// -- id hygiene -------------------------------------------------------------

/**
 * Tokens shorter than this are ignored. "the", "id", "no" overlap everything, and
 * a check that fires on every entityType is a check nobody reads.
 */
export const MIN_HYGIENE_TOKEN_CHARS = 4;

/**
 * Lowercase alphanumeric runs. Splitting on everything else covers kebab ids,
 * prose, and the `:` in a shadow id alike. Non-ASCII letters act as separators,
 * which is a known limit of the heuristic and not a correctness issue: the
 * extraction prompt constrains ids to lowercase kebab ASCII.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

/**
 * Warns when an entityType id looks derived from the confidential thing it
 * describes.
 *
 * Why this matters at all: entityType ids are OUTBOUND-VISIBLE. They ship to the
 * provider inside `[REDACTED:<id>]`, so an id of `project-titan` redacts the
 * string "Project Titan" and then transmits the codename anyway, in the marker.
 * The redaction succeeds and the leak happens.
 *
 * A warning and not an error, deliberately: the compiler has no way to know which
 * nouns a firm considers confidential, so this flags for the human audit that the
 * report demands rather than failing a compile on a guess.
 *
 * Call this with authored AND shadow entityTypes.
 *
 * DEVIATION from the plan, which scores every id against that entityType's
 * `examples[]`: a shadow minted by `mintShadowEntityTypes` has `examples: []` by
 * construction, so the examples heuristic is VACUOUS on shadows — it cannot warn
 * on one however the stage is wired. Shadow ids are outbound-visible exactly like
 * authored ones, and the `<predicateId>` half comes from the extraction model, so
 * they need a real check rather than a check that structurally cannot fire. For a
 * shadow the id is scored against `nlDefinition` (the predicate text) instead,
 * which is the only text a shadow carries. `SHADOW_PREFIX` is stripped before
 * tokenizing so "pred" is not itself scored: it is 4 characters, exactly at the
 * threshold, and would otherwise hit on any predicate whose text says "predicts".
 */
export function validateIdHygiene(entityTypes: readonly EntityTypeInput[]): string[] {
  const warnings: string[] = [];

  for (const entity of entityTypes) {
    const isShadow = entity.id.startsWith(SHADOW_PREFIX);
    const scoredId = isShadow ? entity.id.slice(SHADOW_PREFIX.length) : entity.id;
    const idTokens = new Set(
      tokenize(scoredId).filter((token) => token.length >= MIN_HYGIENE_TOKEN_CHARS),
    );
    if (idTokens.size === 0) continue;

    const corpus = isShadow ? [entity.nlDefinition] : entity.examples;
    const corpusLabel = isShadow ? "own predicate definition" : "own examples";

    const hits = new Set<string>();
    for (const text of corpus) {
      for (const token of tokenize(text)) {
        if (idTokens.has(token)) hits.add(token);
      }
    }
    if (hits.size === 0) continue;

    // The id and the overlapping tokens are named; the example that matched is
    // NOT. A token here is by construction a token OF the id, and the id is
    // already outbound-visible and already printed, so naming it discloses
    // nothing new — whereas an `examples[]` entry is a confidential value lifted
    // from the policy document and must never reach a log.
    const shared = [...hits].sort();
    warnings.push(
      `entityType id "${entity.id}" shares ${shared.length === 1 ? "token" : "tokens"} ` +
        `${shared.map((t) => `"${t}"`).join(", ")} with its ${corpusLabel}; ids ship outbound inside ` +
        `[REDACTED:<id>], so an id derived from a confidential value defeats the redaction — ` +
        `rename it to a generic class name`,
    );
  }

  return warnings;
}
