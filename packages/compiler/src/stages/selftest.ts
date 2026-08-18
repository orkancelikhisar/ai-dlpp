import { detect, type EntityType, type PolicyIr, type Tier } from "@sih/core";
import { z } from "zod";
import type { LlmClient } from "../llm/client.js";

/**
 * Stage 4: measure what the compiled policy actually catches.
 *
 * The whole point of this stage is in one line below — `detect(...)` from
 * `@sih/core`. The coverage numbers come from EXECUTING the real runtime over
 * generated cases, not from asking a model whether it thinks its own rules
 * work. A model grading its own regex is a model marking its own homework, and
 * the number it produces is unfalsifiable. The compiler depends on core
 * precisely so this stage can be honest: the rules are run by the same
 * `runTier0` / merge / action-resolution path that will run them on the send
 * path of every message in the extension.
 *
 * Failing this stage is NOT a compile failure. A weak entity produces a
 * warning naming the entity and its numbers, and compilation "succeeds with
 * warnings" — the report tells a human exactly what is weak, which is more
 * useful than refusing to emit an IR that is 90% right.
 */

// -- constants --------------------------------------------------------------

/**
 * Stamped on every returned case. Plan 7's contamination filter excludes the
 * self-test corpus from the evaluation corpus by n-gram overlap; a tag lets it
 * exclude by provenance as well, which catches the case n-gram overlap misses —
 * a self-test case that happens to share no long n-gram with anything, yet was
 * still generated from the very policy under evaluation. Versioned so a later
 * regeneration with different prompts is distinguishable from this one.
 */
export const CORPUS_TAG = "selftest-v1";

/** Below this fraction of positives detected, the entity is reported as weak. */
export const RECALL_THRESHOLD = 0.8;

/** Above this fraction of hard negatives detected, the entity is over-firing. */
export const MAX_FP_RATE = 0.1;

/** Cases requested per kind, per entityType. Part of the prompt, so part of the fixture key. */
const CASES_PER_KIND = 20;

/** Roughly 40 short lines plus JSON overhead, with headroom. */
const MAX_TOKENS = 4000;

/**
 * Names the response schema in the fixture key (see `requestHash`), so a change
 * to `SelfTestCasesSchema` misses every committed fixture rather than replaying
 * responses recorded against the old shape.
 */
const SCHEMA_NAME = "SelfTestCases";

/**
 * The provider detection is resolved against. Not a real adapter id: this stage
 * measures the DEFAULT action path, and naming a real provider would silently
 * score every entity under that provider's overrides. An unknown id falls back
 * to `actions.default` in `resolveAction`, which is exactly what is wanted.
 */
const SELFTEST_PROVIDER = "selftest";

// -- model contract ---------------------------------------------------------

/**
 * `min(1)` on each case: an empty string segments to nothing and matches
 * nothing, so it would count as a missed positive and drag recall down while
 * measuring the runtime not at all.
 */
export const SelfTestCasesSchema = z.object({
  positives: z.array(z.string().min(1)),
  negatives: z.array(z.string().min(1)),
});

const SYSTEM = [
  "You generate a self-test corpus for ONE entity class from a compiled data-leak policy.",
  "",
  "Return two lists:",
  "- positives: short realistic texts, each containing exactly one value of this class,",
  "  written the way it would really appear — a chat message, a config line, a code snippet,",
  "  a pasted log. Vary the surrounding context across the list.",
  "- negatives: HARD negatives. A hard negative is text that LOOKS like this class and is",
  "  not it: the same shape with a broken checksum, the neighbouring identifier format from",
  "  the same country or vendor, the class's own keyword sitting next to a value of some",
  "  other class, a value one character too short, a documented placeholder. Text with",
  "  nothing to do with the class discriminates nothing and is wasted output.",
  "",
  "Rules you must follow:",
  "1. Never reuse text from the policy document, and never copy the illustrative examples or",
  "   counter-examples you are shown. Invent every value fresh. These cases are executed",
  "   against the compiled rules and then committed as an evaluation corpus, so text lifted",
  "   from the policy would measure whether the compiler memorised its own input.",
  "2. Every value is invented. Never emit a real person's data and never emit a live",
  "   credential — a working key in a committed test corpus is a leak this system caused.",
  "3. A positive is a positive under the class's DEFINITION. You are deliberately not shown",
  "   the detection rules: writing cases to a pattern you guessed would report the coverage",
  "   of your guess rather than of the policy, which is the one number this corpus exists",
  "   to produce.",
  "4. One case per string, single line, no numbering, no commentary, no markdown fences.",
].join("\n");

/**
 * DETERMINISM CONTRACT — read before editing anything below.
 *
 * The fixture key hashes the full `user` prompt, so a single byte that varies
 * between runs means every committed fixture misses and the suite fails with a
 * hash nobody can reproduce. Therefore:
 *
 * - Every field is written out by NAME in a fixed order. Nothing here iterates
 *   an object's keys, and nothing serializes an entityType with
 *   `JSON.stringify`: the parsed IR's key order is an artifact of zod's shape
 *   and of the JSON it came from, neither of which is a stable contract.
 * - Arrays render in IR order, which is array order. They are not sorted —
 *   sorting is also deterministic but would discard the authored ordering, and
 *   determinism is already had without paying for that.
 * - No clock, no randomness, no counters, no locale-sensitive formatting.
 *   `tier` and `severity` interpolate as plain values.
 *
 * The rules are deliberately NOT in this prompt (see rule 3 in SYSTEM), and
 * neither are `surrogateKind` / `neverPseudonymize`: those decide what happens
 * to a detected value, not what the class is, so including them would change
 * fixture keys whenever the action side of the policy changed.
 */
function renderList(values: readonly string[]): string {
  return values.length === 0 ? "  (none given)" : values.map((v) => `  - ${v}`).join("\n");
}

export function userPromptFor(entity: EntityType): string {
  return [
    "Entity class:",
    `  id: ${entity.id}`,
    `  tier: ${entity.tier}`,
    `  severity: ${entity.severity}`,
    `  definition: ${entity.nlDefinition}`,
    "  illustrative examples:",
    renderList(entity.examples),
    "  illustrative counter-examples:",
    renderList(entity.counterExamples),
    "",
    `Return ${CASES_PER_KIND} positives and ${CASES_PER_KIND} hard negatives.`,
  ].join("\n");
}

// -- report shapes ----------------------------------------------------------

export interface SelfTestCase {
  readonly entityType: string;
  readonly kind: "positive" | "negative";
  /** Model-generated, never lifted from the policy — see SYSTEM rule 1. */
  readonly text: string;
  /** True when a finding under THIS entityType survived resolution. */
  readonly detected: boolean;
  /**
   * Set on a positive that no finding of its own entityType covered, but which
   * some OTHER entityType did catch — the value is detected and acted on, under
   * a different label. Absent means nothing caught it at all.
   */
  readonly shadowedBy?: string;
  /** Always CORPUS_TAG. Carried per case so a case stays excludable once split from its report. */
  readonly corpusTag: string;
}

export interface SelfTestEntityReport {
  readonly entityType: string;
  readonly tier: Tier;
  /** Cases executed. Zero when the entity was skipped. */
  readonly positives: number;
  readonly negatives: number;
  /** Positives detected under this entityType's own label. */
  readonly caught: number;
  /**
   * Positives caught by a DIFFERENT entityType instead. Overlap resolution
   * keeps one finding per span by severity, so a value matched by two entities
   * surfaces under the stricter one and this entity's label never appears.
   */
  readonly shadowed: number;
  readonly falsePositives: number;
  /**
   * LEAK-PREVENTION recall: `(caught + shadowed) / positives`. Shadowed
   * positives count as caught because they ARE caught — the runtime found the
   * value and resolved an action for it — and this is the number that
   * corresponds to the project's headline metric.
   *
   * `undefined` when there is no denominator — skipped, or the model returned
   * no positives. Undefined means UNMEASURED and is deliberately not collapsed
   * to 0: a reader charting a skipped tier-1 entity at 0% recall would be
   * reading a false alarm, which is the exact failure this stage's skip rule
   * exists to prevent.
   */
  readonly recall?: number;
  /**
   * `caught / positives` — how often this entity's own label survives. Below
   * `recall` exactly when something shadowed it. Reported separately because
   * the two answer different questions: `recall` asks whether the value leaks,
   * `labelRecall` asks whether the policy author's rule for THIS class is the
   * one that fires, which decides which surrogate and which clause citation the
   * user sees.
   */
  readonly labelRecall?: number;
  readonly fpRate?: number;
  readonly skipped: boolean;
  readonly skipReason?: string;
}

export interface SelfTestReport {
  /** One per IR entityType, in IR order, skipped ones included. */
  readonly entities: SelfTestEntityReport[];
  /**
   * Every generated case with its outcome. RETURNED, not discarded: `emit`
   * (Task 8) writes them beside the IR, so the corpus a coverage claim rests on
   * ships with the claim and can be re-executed by anyone auditing it.
   */
  readonly cases: SelfTestCase[];
  readonly warnings: string[];
  readonly corpusTag: string;
}

// -- scoring ----------------------------------------------------------------

/**
 * Why tier 1 and 2 are skipped rather than scored: their engines land in Plans
 * 4-5. Running them through tier-0-only detection today would report 0% recall
 * for every semantic and span-model entity on every compile — a warning that
 * fires unconditionally, says nothing about the policy, and trains its readers
 * to ignore the warning list that the genuinely weak entities also appear in.
 */
function skipReasonFor(tier: Tier): string | undefined {
  if (tier === 0) return undefined;
  return (
    `entityType is tier ${tier}; only tier 0 executes today, so this compile cannot ` +
    `measure it — the tier ${tier} engine arrives in Plan ${tier === 1 ? 4 : 5}`
  );
}

/** Two decimals, `toFixed` rather than any locale-aware formatter. */
function fmt(fraction: number): string {
  return fraction.toFixed(2);
}

/**
 * One case through the real pipeline.
 *
 * `some` over the RESOLVED findings, so a case counts only if a finding
 * survived overlap resolution — the same set the rewriter acts on. Reading the
 * raw pre-merge findings instead would credit the policy with catches the
 * runtime then discards.
 *
 * Returns the shadowing entityType when this entity's own label is absent but
 * something else fired. Collapsing that to a plain miss would report "caught
 * nothing" for a value the runtime caught and blocked — see `shadowed`.
 */
async function classify(
  ir: PolicyIr,
  entityTypeId: string,
  text: string,
): Promise<{ detected: boolean; shadowedBy?: string }> {
  const { findings } = await detect({
    ir,
    provider: SELFTEST_PROVIDER,
    text,
    config: { tier0: true, tier1: false, tier2: false },
  });
  if (findings.some((f) => f.entityType === entityTypeId)) return { detected: true };
  // First in resolved order: findings come back sorted by start offset, so this
  // is the leftmost survivor rather than an arbitrary one.
  const other = findings[0];
  return other === undefined ? { detected: false } : { detected: false, shadowedBy: other.entityType };
}

/**
 * Generate, execute, and score one tier-0 entityType.
 *
 * Cases are executed SEQUENTIALLY. Detection is microseconds of synchronous
 * regex work, so there is nothing to win by racing them, and sequential
 * execution keeps `cases` in a stable order (positives then negatives, each in
 * the order the model returned them) for the corpus `emit` commits.
 */
async function scoreEntity(
  client: LlmClient,
  ir: PolicyIr,
  entity: EntityType,
): Promise<{ report: SelfTestEntityReport; cases: SelfTestCase[]; warnings: string[] }> {
  const generated = await client.complete(
    { system: SYSTEM, user: userPromptFor(entity), schemaName: SCHEMA_NAME, maxTokens: MAX_TOKENS },
    SelfTestCasesSchema,
  );

  const cases: SelfTestCase[] = [];
  let caught = 0;
  let shadowed = 0;
  let falsePositives = 0;

  for (const text of generated.positives) {
    const { detected, shadowedBy } = await classify(ir, entity.id, text);
    if (detected) caught += 1;
    else if (shadowedBy !== undefined) shadowed += 1;
    cases.push({
      entityType: entity.id,
      kind: "positive",
      text,
      detected,
      ...(shadowedBy === undefined ? {} : { shadowedBy }),
      corpusTag: CORPUS_TAG,
    });
  }
  for (const text of generated.negatives) {
    // A hard negative that fires some OTHER entity is not this entity's false
    // positive; per-entity fpRate would otherwise inherit every neighbour's
    // over-firing and no longer point at the rule a reader has to fix.
    const { detected } = await classify(ir, entity.id, text);
    if (detected) falsePositives += 1;
    cases.push({ entityType: entity.id, kind: "negative", text, detected, corpusTag: CORPUS_TAG });
  }

  const positives = generated.positives.length;
  const negatives = generated.negatives.length;
  const recall = positives === 0 ? undefined : (caught + shadowed) / positives;
  const labelRecall = positives === 0 ? undefined : caught / positives;
  const fpRate = negatives === 0 ? undefined : falsePositives / negatives;

  /**
   * Warnings name the entityType id and the counts, never a case. An id is
   * outbound-visible by design and a count is a count, but a generated positive
   * is a sensitive-SHAPED string — an identifier, a key, a credential — and the
   * convention that error and warning text never carries one does not get an
   * exemption just because this particular string was invented by a model.
   */
  const warnings: string[] = [];
  if (recall === undefined) {
    warnings.push(
      `self-test: entityType "${entity.id}" produced no positives, so its coverage is ` +
        `unmeasured — treat it as unverified rather than passing`,
    );
  } else if (recall < RECALL_THRESHOLD) {
    warnings.push(
      `self-test: entityType "${entity.id}" recall ${fmt(recall)} (${caught + shadowed}/${positives} ` +
        `generated positives caught by any entityType) is below threshold ${RECALL_THRESHOLD} — ` +
        `these values reach the provider, so no rule in the policy matches what this ` +
        `entity's own definition describes`,
    );
  }
  if (shadowed > 0) {
    /**
     * Deliberately NOT worded as a threshold breach, and deliberately not
     * gated on one. Shadowing does not mean the policy leaks — the value was
     * found and an action was resolved for it — so calling it a failure would
     * be the same cry-wolf this stage avoids for tier 1/2. It is still worth
     * saying: the surviving label decides which surrogate the user sees and
     * which clause the UI cites, so an author whose entity is shadowed half the
     * time has written a rule that mostly does not speak.
     */
    warnings.push(
      `self-test: entityType "${entity.id}" was shadowed on ${shadowed}/${positives} ` +
        `generated positives — another entityType won the span under overlap resolution, so ` +
        `the value is still caught but this entity's label, surrogate, and cited clause do ` +
        `not appear`,
    );
  }
  if (fpRate === undefined) {
    warnings.push(
      `self-test: entityType "${entity.id}" produced no hard negatives, so its false-positive ` +
        `rate is unmeasured`,
    );
  } else if (fpRate > MAX_FP_RATE) {
    warnings.push(
      `self-test: entityType "${entity.id}" false-positive rate ${fmt(fpRate)} ` +
        `(${falsePositives}/${negatives} hard negatives detected) exceeds the maximum ` +
        `${MAX_FP_RATE} — its rules fire on text the policy does not cover`,
    );
  }

  return {
    report: {
      entityType: entity.id,
      tier: entity.tier,
      positives,
      negatives,
      caught,
      shadowed,
      falsePositives,
      ...(recall === undefined ? {} : { recall }),
      ...(labelRecall === undefined ? {} : { labelRecall }),
      ...(fpRate === undefined ? {} : { fpRate }),
      skipped: false,
    },
    cases,
    warnings,
  };
}

/**
 * Generate cases per tier-0 entityType, execute them against the real core
 * runtime, and report per-entity coverage.
 *
 * Entities are processed in `ir.entityTypes` order and one at a time: the LLM
 * calls are the expensive part, and a serial loop makes a fixture miss name the
 * first entity that lacks one rather than burying it among concurrent
 * rejections while hand-authoring the corpus.
 */
export async function runSelfTest(client: LlmClient, ir: PolicyIr): Promise<SelfTestReport> {
  const entities: SelfTestEntityReport[] = [];
  const cases: SelfTestCase[] = [];
  const warnings: string[] = [];

  for (const entity of ir.entityTypes) {
    const skipReason = skipReasonFor(entity.tier);
    if (skipReason !== undefined) {
      // No model call and no cases: an unmeasurable entity should not cost a
      // token, and a corpus of cases nothing ever executed is a corpus that
      // will be mistaken for evidence later.
      entities.push({
        entityType: entity.id,
        tier: entity.tier,
        positives: 0,
        negatives: 0,
        caught: 0,
        shadowed: 0,
        falsePositives: 0,
        skipped: true,
        skipReason,
      });
      continue;
    }

    const scored = await scoreEntity(client, ir, entity);
    entities.push(scored.report);
    cases.push(...scored.cases);
    warnings.push(...scored.warnings);
  }

  return { entities, cases, warnings, corpusTag: CORPUS_TAG };
}
