/**
 * The capability-ceiling arm's run. `ceiling-main.ts` is the two-line entry
 * point; everything the run decides lives here, exported, so a test can drive
 * it.
 *
 * ## Why this file exists at all
 *
 * MEASURED on 2026-09-08 against `887317f`, before this extraction: with the
 * whole run inlined in a top-level `main()` that exported nothing, reverting the
 * ledger call site from `spendLedgerFileFor(runId, passStart)` to
 * `` `ceiling-${runId}.spend.json` `` -- byte-for-byte the defect that destroyed
 * a 768-call spend ledger -- left `vitest run` reporting
 * `Tests 791 passed (791)` at exit 0. Nothing in the workspace imported the
 * script, so no test could observe any of it.
 *
 * `fd2087a` had already extracted `spendLedgerFileFor` in order to make that
 * mutation killable. The helper became testable; the CALL SITE did not, so the
 * boundary moved up one layer instead of closing. The rule this file is built
 * around: **a decision is only covered when a test observes the value that
 * decision hands to the next stage.** So `resolveRunPlan` is pure and returns
 * the ledger filename, and `runCeiling` takes `writeSpend` as an injected
 * dependency, which is how a test sees the filename each write actually
 * receives rather than the filename a helper would have produced.
 *
 * ## Environment
 *
 *   OPENROUTER_API_KEY   required
 *   SIH_CEILING_RUN_ID   run id; default "ceiling-01"
 *   SIH_CEILING_PASSES   how many passes to attempt; default 1
 *   SIH_CEILING_PROBE    "1" to run the 3-item probe and stop
 *   SIH_CEILING_LIMIT    cap items per arm (probe/debug); default the whole corpus
 *   SIH_CEILING_MODELS   comma-separated OpenRouter ids to restrict the slate to
 *   SIH_CEILING_THINKING "off" (default, the experiment's condition) or "on".
 *                        "on" exists only for z-ai/glm-5.3-flash, which refuses
 *                        every thinking-off request on all eleven of its
 *                        providers. Rows from an "on" run carry
 *                        thinkingRequested: "on" and are a DIFFERENT CONDITION
 *                        -- never pool them with the off arms.
 *   SIH_CEILING_PIN      re-pin models, e.g. "z-ai/glm-5.3-flash=NextBit".
 *                        Changes what is ASKED; the response still decides what
 *                        `provider` the record carries.
 *   SIH_CEILING_PASS_START  1-based number of the FIRST pass to write, so a
 *                        later pass can be run on its own without overwriting
 *                        an earlier one's files. See passRunIdFor.
 *
 * The key is read from the environment and nowhere else. It is never printed,
 * never written to a run artifact, never placed on a record and never included
 * in an error message -- `callChat` puts it in one header and that header is the
 * only place it exists in this process. It is deliberately NOT a field of
 * `RunPlan`: the plan is logged and is the natural thing for a test to dump, so
 * the key must not be reachable from it.
 *
 * The order of operations is chosen so that the expensive thing never happens
 * before the cheap check that would have stopped it: read the key's balance,
 * then probe three items per model per family, then run the corpus
 * cheapest-model-first. The spend guard is consulted before every call.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPolicyIr, type PolicyIr } from "@sih/core";
import {
  CEILING_MODELS,
  applyPinOverrides,
  REPRESENTATIVE_COMPLETION_TOKENS,
  REPRESENTATIVE_PROMPT_TOKENS,
  SPEND_HARD_STOP_USD,
  SpendGuard,
  armName,
  estimateCostUsd,
  mapWithConcurrency,
  passRunIdFor,
  spendLedgerFileFor,
  runCeilingItem,
  toJsonl,
  type ThinkingRequest,
  type CeilingFamily,
  type CeilingModel,
  type CeilingRecord,
  type ClientDeps,
  type ItemInput,
  type RunItemOptions,
  type SpendEntry,
} from "./ceiling.js";
import { loadCorpus } from "./corpus.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");

/** Concurrency 3, as briefed: enough to make the run finish, few enough to stay under rate limits. */
export const CONCURRENCY = 3;
/** Items per family per model in the probe. */
export const PROBE_ITEMS = 3;
/**
 * The completion ceiling per call.
 *
 * ## This is NOT the local arms' value, and the comment that said it was, lied
 *
 * An earlier version of this docblock read "The local arms run at 600
 * (`DEFAULT_TIER2_CONFIG`). Kept the same here so a truncation means the same
 * thing." That was false and nobody had read the constant: MEASURED at
 * `packages/tier2/src/manifest.ts:116`, `DEFAULT_TIER2_CONFIG.maxTokens` is
 * **512**, and its own docblock at :81 says it is "UNCHANGED at 512 through the
 * two-span schema change". So the hosted arms ran with **88 more completion
 * tokens** than the browser arms.
 *
 * ## Which way it cuts, and where it is inert
 *
 * It FAVOURS the ceiling arms: more room to finish an answer. That is the
 * conservative direction for this experiment's headline -- the ceiling arms had
 * the advantage and still did not beat the trivial floor -- and the flattering
 * direction for any local-vs-ceiling gap, which must be read with that in mind.
 *
 * It is also MEASURABLY INERT for every thinking-off arm, which is a stronger
 * statement than arguing about the direction. Counted over every call written
 * so far: the judge family's completions top out at 103 tokens (p95 61) and
 * Approach B's at 455 (p95 263). **Not one thinking-off call reached 512**, so
 * running at 512 would have produced byte-identical output. The 88 tokens only
 * ever mattered for the thinking-ON arm, where 55 of 216 calls exceeded 512 and
 * 48 hit 600 exactly. `ceiling-score.ts` prints the per-arm `>=512` and
 * `finish=length` counts so this stays checkable rather than asserted.
 *
 * NOT changed to 512 mid-experiment: the paid run was already in flight, and a
 * slate half-measured at each value would be worse than one measured at a
 * documented 600. Every row records the value it ran at.
 *
 * A thinking-ON phase cannot be measured at either cap -- see the doc -- and
 * needs a uniformly larger ceiling (8,192) applied to every model.
 */
export const MAX_TOKENS = 600;

/** `DEFAULT_TIER2_CONFIG.maxTokens`, restated so the asymmetry above is checkable here. */
export const LOCAL_ARM_MAX_TOKENS = 512;

/**
 * The key spend above which passes 2+ are abandoned, in USD.
 *
 * Named rather than inline so the gate and the sentence that reports it cannot
 * drift apart: the note below formats this same number.
 */
export const LATER_PASS_GATE_USD = 2.5;

const RUNS = join(REPO, "runs");

/**
 * Reads one variable out of an env BAG, not out of `process.env`.
 *
 * Empty string is treated as absent, which is what makes `SIH_CEILING_MODELS=`
 * mean "the whole slate" rather than "a slate matching no model" -- the latter
 * throws.
 */
function readEnv(bag: Record<string, string | undefined>, name: string): string | undefined {
  const v = bag[name];
  return v === undefined || v === "" ? undefined : v;
}

export interface KeyState {
  usage: number;
  limit: number | null;
}

/** The shape of `execFileSync` this module uses; narrowed so a test can supply one. */
export type ExecFileLike = (
  file: string,
  args: readonly string[],
  options: { cwd: string; encoding: "utf8" },
) => string;

/**
 * The commit the rows are stamped with, and whether the tree was dirty AT
 * LAUNCH.
 *
 * Called once per process, so a row from pass 3 records the state at startup and
 * not the state when that row was written -- see the run record's Sec 11.4.
 */
export function gitProvenance(
  exec: ExecFileLike = execFileSync as unknown as ExecFileLike,
  repo: string = REPO,
): { gitSha: string; gitDirty: boolean } {
  const gitSha = exec("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim();
  const status = exec("git", ["status", "--porcelain"], { cwd: repo, encoding: "utf8" });
  return { gitSha, gitDirty: status.trim().length > 0 };
}

export async function readKeyState(apiKey: string): Promise<KeyState> {
  const response = await fetch("https://openrouter.ai/api/v1/auth/key", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!response.ok) {
    // The status, never the body: an auth error body can echo the key prefix.
    throw new Error(`/auth/key returned ${response.status}`);
  }
  const body = (await response.json()) as { data?: { usage?: unknown; limit?: unknown } };
  const usage = typeof body.data?.usage === "number" ? body.data.usage : 0;
  const limit = typeof body.data?.limit === "number" ? body.data.limit : null;
  return { usage, limit };
}

export function loadIr(repo: string = REPO): { ir: PolicyIr; irHash: string } {
  const path = join(repo, "policies/compiled/p-fin.ir.json");
  const bytes = readFileSync(path);
  // Hashed from the ARTIFACT's own bytes, exactly as the page does it, so
  // `shasum -a 256` on the file reproduces what the record claims.
  const irHash = createHash("sha256").update(bytes).digest("hex");
  // The repo's own loader, not a bare schema parse: it is what narrows
  // `irVersion` and what every other IR reader here goes through, so a
  // divergence in validation between arms is impossible.
  const ir = loadPolicyIr(bytes.toString("utf8"));
  return { ir, irHash };
}

export function loadItems(repo: string = REPO): ItemInput[] {
  const jsonl = readFileSync(join(repo, "corpora/generated/injection-p-fin-v2.labelled.jsonl"), "utf8");
  return loadCorpus(jsonl).map((item) => ({
    id: item.id,
    text: item.text,
    policy: item.policy,
    gold: item.gold.map((g) => ({
      start: g.start,
      end: g.end,
      text: g.text,
      entityType: g.entityType,
      action: g.action,
    })),
  }));
}

/**
 * The ledger filename is keyed by the FIRST pass this process writes, not by the
 * base `runId`.
 *
 * MEASURED, and the reason `spendLedgerFileFor` takes `passStart`: a relaunch at
 * `passStart=2` names its arm files `ceiling-02.*` correctly (see
 * `passRunIdFor`) but every `writeSpend` call used the base id, so its ledger
 * overwrote `ceiling-ceiling-01.spend.json` -- the pass-1 window-2 ledger, 768
 * calls and $0.19269, replaced mid-run by pass 2's running total. The arm-path
 * fix and the ledger-path fix are the same footgun one layer apart;
 * `passRunIdFor` closed only the first.
 *
 * One ledger per PROCESS is correct -- the guard accumulates across the passes
 * it runs -- so the name carries the first pass rather than every pass.
 */
export function writeSpend(ledgerFile: string, guard: SpendGuard, extra: Record<string, unknown>): void {
  mkdirSync(RUNS, { recursive: true });
  writeFileSync(
    join(RUNS, ledgerFile),
    `${JSON.stringify({ ...guard.snapshot(), ...extra }, null, 2)}\n`,
  );
}

/** The per-arm JSONL file one pass writes. */
export function writeArm(fileName: string, jsonl: string): void {
  mkdirSync(RUNS, { recursive: true });
  writeFileSync(join(RUNS, fileName), jsonl);
}

// ---------------------------------------------------------------------------
// The decisions, as pure functions
// ---------------------------------------------------------------------------

export interface RunPlan {
  readonly runId: string;
  readonly passes: number;
  readonly probeOnly: boolean;
  readonly limit: number | undefined;
  readonly only: string[] | undefined;
  readonly thinking: ThinkingRequest;
  readonly passStart: number;
  readonly slate: CeilingModel[];
  /** What every `writeSpend` in this process must be handed. */
  readonly ledgerFile: string;
  /** One id per pass this process will write, in order. */
  readonly passRunIds: string[];
  /**
   * The completion ceiling every call in this run is made under.
   *
   * Overridable because a thinking-ON phase CANNOT be measured at the
   * thinking-off value: GLM-5.3-flash truncated 22% of its answers at 600, and
   * every one of its eight missed gold positives was on a truncated call. The
   * doc's Sec 7.2 asks for a uniformly larger ceiling (8,192) applied to every
   * model. Every row records the value it actually ran at.
   */
  readonly maxTokens: number;
}

/**
 * Everything the environment decides, resolved once.
 *
 * Takes the env BAG rather than reading `process.env`, so the resolution is
 * drivable from a test. `apiKey` is deliberately absent -- see the module
 * docblock.
 */
export function resolveRunPlan(bag: Record<string, string | undefined>): RunPlan {
  const runId = readEnv(bag, "SIH_CEILING_RUN_ID") ?? "ceiling-01";
  const passes = Number(readEnv(bag, "SIH_CEILING_PASSES") ?? "1");
  const probeOnly = readEnv(bag, "SIH_CEILING_PROBE") === "1";
  const rawLimit = readEnv(bag, "SIH_CEILING_LIMIT");
  const limit = rawLimit === undefined ? undefined : Number(rawLimit);
  const only = readEnv(bag, "SIH_CEILING_MODELS")?.split(",").map((s) => s.trim());
  const thinking: ThinkingRequest = readEnv(bag, "SIH_CEILING_THINKING") === "on" ? "on" : "off";
  const passStart = Number(readEnv(bag, "SIH_CEILING_PASS_START") ?? "1");
  const rawMaxTokens = readEnv(bag, "SIH_CEILING_MAX_TOKENS");
  const maxTokens = rawMaxTokens === undefined ? MAX_TOKENS : Number(rawMaxTokens);
  // Rejected rather than coerced: a typo silently becoming NaN would be sent as
  // `max_tokens: null` and every arm would run at the provider's own default,
  // which differs per provider and is not recorded anywhere.
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error(`SIH_CEILING_MAX_TOKENS must be a positive integer; got ${JSON.stringify(rawMaxTokens)}`);
  }
  const repinned = applyPinOverrides(CEILING_MODELS, readEnv(bag, "SIH_CEILING_PIN"));
  const slate = only === undefined ? repinned : repinned.filter((m) => only.includes(m.id));
  if (slate.length === 0) throw new Error(`SIH_CEILING_MODELS matched no model on the slate`);
  return {
    runId,
    passes,
    probeOnly,
    limit,
    only,
    thinking,
    passStart,
    slate,
    maxTokens,
    ledgerFile: spendLedgerFileFor(runId, passStart),
    passRunIds: Array.from({ length: passes }, (_, i) => passRunIdFor(runId, i + 1, passStart)),
  };
}

/**
 * The arm file's name.
 *
 * Keyed by the PASS run id, never the base `runId`: pass 2 writing
 * `ceiling-01.ceiling-judge-*.jsonl` would overwrite pass 1's rows, which is the
 * same footgun as the ledger one above.
 */
export function armFileName(passRunId: string, family: CeilingFamily, model: CeilingModel): string {
  return `${passRunId}.${armName(family, model)}.jsonl`;
}

/**
 * Whether to abandon passes 2+ after pass 1.
 *
 * Passes 2 and 3 run only when pass 1 left room, so a repeat can never be the
 * thing that exhausts the key. The gate is on the KEY's spend, not this run's,
 * because the key carries whatever earlier runs already charged it.
 */
export function laterPassGate(
  pass: number,
  passes: number,
  keySpendUsd: number,
): { stop: boolean; note?: string } {
  if (pass === 1 && passes > 1 && keySpendUsd >= LATER_PASS_GATE_USD) {
    return {
      stop: true,
      note:
        `passes 2+ SKIPPED: pass 1 spent $${keySpendUsd.toFixed(4)}, ` +
        `over the $${LATER_PASS_GATE_USD.toFixed(2)} gate`,
    };
  }
  return { stop: false };
}

/**
 * The `onCall` callback every item run is handed.
 *
 * The order inside it is load-bearing and is asserted: `guard.record` FIRST, so
 * the ledger the next line writes already contains the call it is reporting. The
 * checkpoint read is fire-and-forget -- the guard trips on the summed costs
 * meanwhile, and awaiting a key read inside a call callback would serialize the
 * run.
 */
export function makeCallHook(options: {
  guard: SpendGuard;
  ledgerFile: string;
  apiKey: string;
  extra: Record<string, unknown>;
  writeSpend: (ledgerFile: string, guard: SpendGuard, extra: Record<string, unknown>) => void;
  readKeyState: (apiKey: string) => Promise<KeyState>;
}): (entry: SpendEntry) => void {
  return (entry) => {
    options.guard.record(entry);
    if (options.guard.dueForCheckpoint()) {
      void options
        .readKeyState(options.apiKey)
        .then((s) => options.guard.noteCheckpoint(s.usage))
        .catch(() => undefined);
    }
    options.writeSpend(options.ledgerFile, options.guard, options.extra);
  };
}

export interface ProbeResult {
  readonly model: string;
  readonly family: CeilingFamily;
  readonly attempted: number;
  readonly parsed: number;
  readonly providersSeen: string[];
  readonly providerMatchesPin: boolean;
  readonly reasoningTokens: (number | null)[];
  readonly reasoningAllZero: boolean;
  readonly ttftMs: (number | null)[];
  readonly wallMs: number[];
  readonly errors: string[];
  readonly verdict: "pass" | "skip";
}

export function summarizeProbe(
  model: CeilingModel,
  family: CeilingFamily,
  records: readonly CeilingRecord[],
): ProbeResult {
  const calls = records.flatMap((r) => r.calls);
  const providersSeen = [...new Set(calls.map((c) => c.provider).filter((p): p is string => p !== null))];
  const reasoning = calls.map((c) => c.reasoningTokens);
  const errors = records.map((r) => r.error).filter((e): e is string => e !== null);
  const parsed = records.filter((r) => r.error === null && r.calls.some((c) => c.parse === "ok")).length;
  return {
    model: model.id,
    family,
    attempted: records.length,
    parsed,
    providersSeen,
    // A pin is a request; this compares it with the fact. A mismatch does not
    // stop the arm -- the row records both -- but it is reported loudly,
    // because the latency columns would then be about routing.
    providerMatchesPin: providersSeen.length > 0 && providersSeen.every((p) => p === model.provider),
    reasoningTokens: reasoning,
    // "All zero" only counts MEASURED zeros. A null is "the provider reported
    // no usage", which is not evidence that thinking was off.
    reasoningAllZero: reasoning.length > 0 && reasoning.every((r) => r === 0),
    ttftMs: calls.map((c) => c.ttftMs),
    wallMs: calls.map((c) => c.wallMs),
    errors,
    // A model whose probe parses NOTHING is skipped, not retried into the
    // budget. One or two failures out of three is a model that is answering,
    // and the full run's own counters are where that gets measured.
    verdict: parsed > 0 ? "pass" : "skip",
  };
}

// ---------------------------------------------------------------------------
// The orchestration
// ---------------------------------------------------------------------------

/**
 * Everything `runCeiling` touches outside itself.
 *
 * Injected as one bag rather than read from module scope, because the wiring --
 * WHICH filename each write receives, WHETHER the guard is fed at all -- is the
 * part that has broken in production and the part a test can only see from
 * here.
 */
export interface CeilingRunDeps {
  readonly env: Record<string, string | undefined>;
  readonly gitProvenance: () => { gitSha: string; gitDirty: boolean };
  readonly loadIr: () => { ir: PolicyIr; irHash: string };
  readonly loadPolicyText: () => string;
  readonly loadItems: () => ItemInput[];
  readonly readKeyState: (apiKey: string) => Promise<KeyState>;
  readonly runItem: (options: RunItemOptions) => Promise<CeilingRecord>;
  readonly writeSpend: (ledgerFile: string, guard: SpendGuard, extra: Record<string, unknown>) => void;
  readonly writeArm: (fileName: string, jsonl: string) => void;
  readonly clientDeps: (apiKey: string) => ClientDeps;
  readonly log: (line: string) => void;
  readonly now: () => number;
}

export function defaultCeilingRunDeps(): CeilingRunDeps {
  return {
    env: process.env,
    gitProvenance: () => gitProvenance(),
    loadIr: () => loadIr(),
    loadPolicyText: () => readFileSync(join(REPO, "policies/p-fin.md"), "utf8"),
    loadItems: () => loadItems(),
    readKeyState,
    runItem: runCeilingItem,
    writeSpend,
    writeArm,
    clientDeps: (apiKey) => ({
      fetch: globalThis.fetch,
      now: () => performance.now(),
      sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
      apiKey,
    }),
    log: (line) => console.log(line),
    now: () => performance.now(),
  };
}

export async function runCeiling(overrides: Partial<CeilingRunDeps> = {}): Promise<void> {
  const deps: CeilingRunDeps = { ...defaultCeilingRunDeps(), ...overrides };
  const apiKey = readEnv(deps.env, "OPENROUTER_API_KEY");
  if (apiKey === undefined) {
    throw new Error("OPENROUTER_API_KEY is not set; the ceiling arm reads its key from the environment only");
  }
  const plan = resolveRunPlan(deps.env);
  const { runId, passes, probeOnly, limit, thinking, passStart, slate, ledgerFile, maxTokens } = plan;

  const { gitSha, gitDirty } = deps.gitProvenance();
  const { ir, irHash } = deps.loadIr();
  const policyText = deps.loadPolicyText();
  const allItems = deps.loadItems();
  const items = limit === undefined ? allItems : allItems.slice(0, limit);

  const keyAtStart = await deps.readKeyState(apiKey);
  const guard = new SpendGuard({
    // The run's OWN ceiling, never the key's: a key with a $100 limit must not
    // license a $100 run. The two numbers are both reported in the snapshot.
    hardStopUsd: SPEND_HARD_STOP_USD,
    keyUsageAtStart: keyAtStart.usage,
    keyLimit: keyAtStart.limit,
    gitSha,
    gitDirty,
  });

  const clientDeps = deps.clientDeps(apiKey);

  deps.log(
    `[ceiling] runId=${runId} passes=${passes} models=${slate.length} items=${items.length} ` +
      `thinking=${thinking} ` +
      `git=${gitSha.slice(0, 12)}${gitDirty ? "+dirty" : ""}`,
  );
  deps.log(
    `[ceiling] max_tokens=${maxTokens} (local arms run at ${LOCAL_ARM_MAX_TOKENS}; ` +
      `+${maxTokens - LOCAL_ARM_MAX_TOKENS} favouring this arm)\n` +
      `[ceiling] key usage=$${keyAtStart.usage.toFixed(4)} limit=${keyAtStart.limit === null ? "none" : `$${keyAtStart.limit}`} ` +
      `hard stop=$${SPEND_HARD_STOP_USD}`,
  );
  deps.log(
    `[ceiling] slate order (cheapest first at ${REPRESENTATIVE_PROMPT_TOKENS}/${REPRESENTATIVE_COMPLETION_TOKENS} tokens): ` +
      slate
        .map(
          (m) =>
            `${m.id}@${m.provider} ~$${(
              estimateCostUsd(m, REPRESENTATIVE_PROMPT_TOKENS, REPRESENTATIVE_COMPLETION_TOKENS) * items.length
            ).toFixed(3)}`,
        )
        .join(", "),
  );

  deps.log(`[ceiling] will write passes: ${plan.passRunIds.join(", ")}`);
  deps.log(`[ceiling] ledger: runs/${ledgerFile}`);

  const families: CeilingFamily[] = ["judge", "b"];
  const probes: ProbeResult[] = [];
  const stopNotes: string[] = [];
  const ledgerExtra: Record<string, unknown> = { runId, passes, probes, stopNotes };

  const onCall = makeCallHook({
    guard,
    ledgerFile,
    apiKey,
    extra: ledgerExtra,
    writeSpend: deps.writeSpend,
    readKeyState: deps.readKeyState,
  });

  const runOne = (
    model: CeilingModel,
    family: CeilingFamily,
    item: ItemInput,
    thisRunId: string,
  ): Promise<CeilingRecord> =>
    deps.runItem({
      deps: clientDeps,
      model,
      family,
      ir,
      irHash,
      policyText,
      item,
      runId: thisRunId,
      maxTokens,
      localArmMaxTokens: LOCAL_ARM_MAX_TOKENS,
      // The bake-off's own default destination provider, so an action resolved
      // here matches an action resolved in the browser arms.
      destinationProvider: "claude",
      thinking,
      gitSha,
      gitDirty,
      onCall,
    });

  // ---- PROBE ------------------------------------------------------------
  const probeItems = items.slice(0, PROBE_ITEMS);
  for (const model of slate) {
    for (const family of families) {
      if (guard.tripped()) {
        stopNotes.push(`spend guard tripped during probe, before ${model.id}/${family}`);
        break;
      }
      const { results } = await mapWithConcurrency(probeItems, CONCURRENCY, () => guard.tripped(), (item) =>
        runOne(model, family, item, `${runId}-probe`),
      );
      const probe = summarizeProbe(model, family, results.filter((r): r is CeilingRecord => r !== undefined));
      probes.push(probe);
      deps.log(
        `[probe] ${model.id} ${family}: parsed ${probe.parsed}/${probe.attempted} ` +
          `provider=${probe.providersSeen.join("|") || "none"} ${probe.providerMatchesPin ? "(pinned)" : "(NOT THE PIN)"} ` +
          `reasoning=${probe.reasoningTokens.join(",")} ` +
          `ttft=${probe.ttftMs.map((t) => (t === null ? "null" : Math.round(t))).join(",")}ms ` +
          `wall=${probe.wallMs.map((t) => Math.round(t)).join(",")}ms ` +
          `verdict=${probe.verdict}` +
          (probe.errors.length > 0 ? ` errors=${probe.errors.slice(0, 2).join(" | ")}` : ""),
      );
    }
  }
  deps.writeSpend(ledgerFile, guard, ledgerExtra);
  if (probeOnly) {
    deps.log(`[ceiling] probe only; spent $${guard.totalUsd.toFixed(5)} so far`);
    return;
  }

  // ---- PASSES -----------------------------------------------------------
  for (let pass = 1; pass <= passes; pass++) {
    const passRunId = passRunIdFor(runId, pass, passStart);
    for (const model of slate) {
      for (const family of families) {
        const probe = probes.find((p) => p.model === model.id && p.family === family);
        if (probe?.verdict === "skip") {
          const note = `${armName(family, model)} SKIPPED: probe parsed 0 of ${probe.attempted}`;
          stopNotes.push(note);
          deps.log(`[skip] ${note}`);
          continue;
        }
        if (guard.tripped()) {
          const note =
            `SPEND GUARD TRIPPED before ${armName(family, model)} on pass ${pass}: ` +
            `$${guard.keySpendUsd.toFixed(4)} of $${SPEND_HARD_STOP_USD}`;
          stopNotes.push(note);
          deps.log(`[stop] ${note}`);
          break;
        }
        const started = deps.now();
        const { results, completed, stoppedAtIndex } = await mapWithConcurrency(
          items,
          CONCURRENCY,
          () => guard.tripped(),
          (item) => runOne(model, family, item, passRunId),
        );
        const records = results.filter((r): r is CeilingRecord => r !== undefined);
        if (stoppedAtIndex !== null) {
          const note =
            `SPEND GUARD TRIPPED mid-arm ${armName(family, model)} pass ${pass}: ` +
            `stopped at item index ${stoppedAtIndex} (${items[stoppedAtIndex]?.id ?? "?"}), ` +
            `${completed} of ${items.length} items written`;
          stopNotes.push(note);
          deps.log(`[stop] ${note}`);
        }
        if (records.length > 0) {
          deps.writeArm(armFileName(passRunId, family, model), toJsonl(records));
        }
        deps.log(
          `[arm] ${passRunId} ${armName(family, model)}: ${records.length} rows in ` +
            `${Math.round(deps.now() - started)}ms, spend so far $${guard.totalUsd.toFixed(5)}`,
        );
        deps.writeSpend(ledgerFile, guard, ledgerExtra);
      }
    }
    const gate = laterPassGate(pass, passes, guard.keySpendUsd);
    if (gate.stop) {
      stopNotes.push(gate.note!);
      deps.log(`[stop] ${gate.note!}`);
      break;
    }
  }

  const keyAtEnd = await deps.readKeyState(apiKey);
  guard.noteCheckpoint(keyAtEnd.usage);
  deps.writeSpend(ledgerFile, guard, { ...ledgerExtra, keyAtEnd });
  deps.log(
    `[ceiling] done. calls=${guard.calls} summed=$${guard.totalUsd.toFixed(5)} ` +
      `estimate=$${guard.estimatedUsd.toFixed(5)} key usage=$${keyAtEnd.usage.toFixed(5)} ` +
      `of ${keyAtEnd.limit === null ? "no limit" : `$${keyAtEnd.limit}`}`,
  );
}
