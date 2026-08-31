import { z } from "zod";
import type { DegradedNotice, DegradedReason, Tier } from "@sih/core";
import { TIER1_BACKENDS, TIER1_LABEL_FORMS } from "@sih/tier1";
import type { JudgeCallRecord, JudgeStats, Tier2Config } from "@sih/tier2";
import { GoldSpanSchema } from "./corpus.js";

/**
 * Bump when a field is removed or its meaning changes. Plan 8's Python reads
 * this first and refuses a version it does not know, so a silently reshaped
 * record cannot be scored as if it were the old one.
 *
 * STILL 1 despite this shape having changed twice since the constant was
 * written -- `text` was added, and the single `policyHash` became `irHash` plus
 * a verbatim `policyHash`. That looks like the rule above being broken in the
 * same file that states it, and is not, for one reason worth writing down
 * rather than leaving to be re-derived: NO RECORD OF ANY EARLIER SHAPE HAS EVER
 * BEEN WRITTEN. This schema landed before its only producer (`runArm` in
 * driver/run.ts), and nothing in the repo calls `toJsonl` outside this module's
 * own tests, so there is no file anywhere for a version 2 to be distinguished
 * FROM. Bumping would advertise a compatibility boundary that does not exist
 * and oblige Plan 8 to carry a reader for a shape nothing ever emitted.
 *
 * Version 1 therefore describes the only record shape that has ever existed.
 * From the first run that writes a file, the rule above applies literally.
 *
 * STILL 1 after two further additions -- `tier1Stats` and
 * `abandonedWorkInFlight` -- under exactly the reasoning above and no other.
 * Both are ADDITIONS, which the rule does not cover in the first place, and the
 * only files any producer here has ever written are the mkdtemp directories
 * test/matrix.spec.ts creates and abandons. Nothing durable exists for a
 * version 2 to be distinguished from. If that changes -- if a run is ever kept
 * -- this constant moves before the next field does.
 *
 * STILL 1 after the tier-2 evidence -- `degraded`, `tier2Stats` and
 * `config.uncertainBelow` -- and the reasoning is the same one a third time,
 * with one thing worth saying out loud because it is the closest call so far.
 * All three are additions, and no existing field changed meaning. Two of them
 * are REQUIRED under a condition, which makes them a tightening rather than a
 * pure widening: a record written by an older producer would now be refused.
 * That is a compatibility boundary in the direction that matters -- old files
 * failing a new reader -- and it is still not a version bump, for the same
 * reason as before and no other: no such file exists. Nothing durable has ever
 * been written by any producer here.
 *
 * STILL 1 after `tier2Config`, which is a fourth addition and a fifth
 * conditional requirement, on the same reasoning and no other. It is worth
 * naming why it was added rather than only that it was: the record carried
 * `config.t2Model` and nothing else about tier 2, so two arms differing only in
 * their context window or their per-call budget emitted rows byte-identical in
 * every field a scorer can group by -- the disease `tier1Config` cured one tier
 * down. If a run is ever kept, this constant moves before the next field does.
 */
export const RECORD_SCHEMA_VERSION = 1;

/**
 * Core's `Tier`, as one definition.
 *
 * Two things on a record carry a tier -- a finding and a degradation notice --
 * and a second copy of the union would be free to drift from the first.
 *
 * The coupling to core is the `satisfies` on `TIER_MEMBERS` below rather than an
 * import of a runtime value, because there is none: `Tier` is a type. That
 * literal is checked in BOTH directions, which matters asymmetrically. A member
 * this file has that core does not is harmless -- nothing would ever produce it.
 * A member CORE has that this file is missing is the real hazard: a schema that
 * compiles clean and rejects a real value at run time, which on this path means
 * a whole arm's file refused after the GPU time has been spent.
 */
const TierSchema = z.union([z.literal(0), z.literal(1), z.literal(2)]);
const TIER_MEMBERS: Record<Tier, true> = { 0: true, 1: true, 2: true } satisfies Record<
  z.infer<typeof TierSchema>,
  true
>;

/**
 * A finding as core produced it, validated against core's ACTUAL unions rather
 * than widened to strings.
 *
 * The widening is worth naming because it was the original shape here and it is
 * the tempting one: `findings` crosses a process boundary as JSON, so typing it
 * loosely feels like tolerance. It is not. Plan 8 compares `findings[].action`
 * against `gold[].action`, and gold's action is a validated enum -- so a widened
 * findings side means one comparison with a checked vocabulary on the left and
 * an unchecked one on the right, where `"blocked"` silently never matches
 * `"block"` and reads as a miss rather than a bug. MEASURED on the widened
 * version: a finding with severity "banana", tier -7, confidence 42, action
 * "obliterate" and empty entityType/source validated cleanly.
 *
 * Unions kept in step with packages/core/src/policy/types.ts (Severity, Tier,
 * Action) and detect/types.ts (Finding). If core adds an action, this fails
 * loudly on the next run, which is the intended coupling.
 */
export const RecordFindingSchema = z.object({
  /** UTF-16 code-unit offset into the record's `text`. See GoldSpanSchema in corpus.ts. */
  start: z.number().int().nonnegative(),
  /** Exclusive UTF-16 code-unit offset. See GoldSpanSchema in corpus.ts. */
  end: z.number().int().positive(),
  /**
   * Exactly `text.slice(start, end)` on this record's own `text`. Enforced by
   * the refine at the bottom of RunRecordSchema, so it is a real cross-check a
   * non-JS reader can run rather than a claim.
   *
   * `.min(1)` matches GoldSpanSchema.text and is load-bearing rather than
   * symmetry for its own sake: without it a degenerate span walks straight
   * through that refine, because `slice()` returns "" for any inverted or
   * zero-width range and "" compares equal to a claimed text of "". MEASURED on
   * the version without it, all three ACCEPTED as findings while gold rejected
   * every one: {start:999,end:1}, {start:5,end:5}, {start:8,end:3}.
   */
  text: z.string().min(1),
  /** An entityType id from the IR that ran -- same namespace as a gold span's. */
  entityType: z.string().min(1),
  severity: z.enum(["low", "medium", "high", "critical"]),
  /** Core's Tier: 0 rules, 1 span tagger, 2 semantic judge. */
  tier: TierSchema,
  /** Rule id (tier 0) or model identifier (tiers 1-2); never empty. */
  source: z.string().min(1),
  /** 0..1, per Finding.confidence. */
  confidence: z.number().min(0).max(1),
  /**
   * Core's Action union exactly. Note this is NARROWER than a gold span's,
   * which also admits "none": a resolved finding always carries a real action,
   * whereas gold can label a span the policy deliberately permits.
   */
  action: z.enum(["allow", "pseudonymize", "redact", "block"]),
});

/**
 * The reason words `DetectionResult.degraded` can carry, as VALUES.
 *
 * Built from a `satisfies Record<DegradedReason, true>` literal, which is the
 * idiom core itself uses for `PREDICATE_SCOPES` and `ENGINE_DEGRADED_REASONS`,
 * and for the same reason: the literal is checked in both directions at once.
 * A word core adds is a compile error here rather than a value this schema
 * silently starts rejecting at run time -- and rejecting one would mean an
 * arm's whole file refused after its GPU time was spent.
 *
 * The cast is what `Object.keys` costs: it is typed `string[]`, and `z.enum`
 * needs the union to infer anything narrower than `string`. It states what the
 * `satisfies` above has just established, not something new.
 */
const DEGRADED_REASONS = Object.keys({
  "failed-closed": true,
  "call-budget-exhausted": true,
  "budget-exhausted": true,
  absent: true,
  "scope-unjudged": true,
} satisfies Record<DegradedReason, true>) as [DegradedReason, ...DegradedReason[]];

/**
 * One entry of `DetectionResult.degraded`, carried onto the record.
 *
 * WHY THE RECORD CARRIES IT. `runArm` projected `DetectionResult` field by
 * field -- `findings` and `timings` -- so this array was DROPPED, silently.
 * MEASURED against the commit before this one: `run.ts` never named `degraded`
 * at all and `tsc --noEmit` was clean, even though the field is REQUIRED on
 * `DetectionResult`, because a projection that names its fields cannot be told
 * it has missed one -- and nothing downstream noticed, because this schema had
 * no field for it to be missing from. Its own doc states what that costs, and
 * it is the bake-off's central comparison: "a tier-2 arm that failed closed on
 * 40% of its messages is distinguishable from one that found nothing". Not from
 * the file, it was not.
 *
 * Counters are not a substitute, and the reason is not a matter of taste. Three
 * of the five reason words have NO counter anywhere: `absent` and
 * `scope-unjudged` are the orchestrator's own facts, which no judge is in a
 * position to count, and the budget-spent-before-start form of
 * `budget-exhausted` is filed on a path that makes no engine call at all, so
 * there is nothing for `tier2Stats` to have counted. A fourth,
 * `budget-exhausted` after a run, is taken from the orchestrator's TIMER and is
 * filed even for a judge that ignored the abort and answered in full.
 *
 * READ IT PER ENTRY. `degraded.length > 0` is not a cleanliness test: an
 * `absent` entry is filed for every tier the TierConfig switched off, so every
 * tier-0 row in the matrix carries two of them and nothing went wrong on any of
 * them.
 *
 * The `satisfies` pins the FIELD SET to core's own interface, so a field added
 * to `DegradedNotice` fails to compile here instead of going missing from the
 * file.
 */
const DegradedNoticeSchema = z.object({
  /** Which tier contributed less than a full run would have. */
  tier: TierSchema,
  reason: z.enum(DEGRADED_REASONS),
  /**
   * Required and non-empty, matching `stampEngineNotice`, which throws on an
   * empty one calling it "the entire human-readable payload of a notice". Never
   * message text, a finding's text, or model output -- see EngineDegradedNotice.
   */
  detail: z.string().min(1),
} satisfies Record<keyof DegradedNotice, z.ZodType>);

/**
 * `ChatCompletionFinishReason` as values, reached through `JudgeCallRecord`.
 *
 * apps/eval does not depend on @mlc-ai/web-llm and the union is a type with no
 * runtime array behind it, so this is as close to the source as the compiler
 * can get: `JudgeCallRecord.finishReason` is `ChatCompletionFinishReason |
 * undefined`, and the `satisfies` below is exact in both directions. READ from
 * the installed 0.2.84 declarations -- `lib/openai_api_protocols/
 * chat_completion.d.ts` declares `ChatCompletionFinishReason` as exactly these
 * four words -- rather than assumed from the OpenAI protocol they imitate,
 * which also has `content_filter` and `function_call`.
 */
type Tier2FinishReason = NonNullable<JudgeCallRecord["finishReason"]>;
const TIER2_FINISH_REASONS = Object.keys({
  stop: true,
  length: true,
  tool_calls: true,
  abort: true,
} satisfies Record<Tier2FinishReason, true>) as [Tier2FinishReason, ...Tier2FinishReason[]];

/**
 * ONE ENGINE CALL, not one message.
 *
 * A message makes one call per selected segment, plus the pinned recipe's one
 * repair retry, so all five of `finishReason`, `promptTokens`,
 * `completionTokens`, `ttftMs` and `decodeTokPerSec` are per-CALL quantities.
 * The record carries the ROWS rather than an aggregate, and the choice is
 * deliberate on three grounds:
 *
 * 1. `finishReason` does not aggregate at all. A message with one call that
 *    stopped cleanly and one that hit the token ceiling produced a PARTIAL
 *    judgement, and any single word for it -- the last call's, the first's,
 *    the "worst" -- is a fact about one call presented as a fact about the
 *    message. Task 6 kept rows for exactly this reason.
 * 2. The bake-off's latency gate is a p95 of TTFT across CALLS at a stated
 *    prompt size. Reducing each message to one number first would make that
 *    quantity uncomputable from the file: a p95 over per-message means is a
 *    different statistic, and it is not the one the gate names.
 * 3. Rows still permit every aggregate. Token totals are a sum over `calls`,
 *    which a scorer can take; the reverse -- recovering a distribution from a
 *    sum -- is not available.
 *
 * The cost is size: `calls` is one small object per segment per message, and it
 * is bounded. Task 9 measured the segments-per-message distribution over the
 * only corpus in this repository under TWO conditions, and the one that applies
 * to a tier-2 bake-off here is the no-priors one, pinned in
 * test/segments.test.ts as `perItem: {p50: 1, p95: 2, max: 2, min: 1}`. So with
 * the pinned recipe's one repair retry that is at most 4 rows on the worst
 * message and at most 2 on the median one.
 *
 * The OTHER condition -- `{p50: 1, p95: 3, max: 3}` "under tier-0 priors" --
 * is the one an earlier version of this comment quoted, and it belongs to a
 * different policy: Task 9 measured it against `minimal-ir.json`, whose
 * `entropy-rule` fires on this corpus's code fence at confidence 0.7 and
 * re-admits that fence to escalation. `semantic-ir.json` -- the only IR a
 * tier-2 arm can run here, and `planBakeoff` throws on any other -- declares
 * `rules: []`, so tier 0 finds nothing, no segment is uncertain, and a tier-0
 * arm's distribution is identical to a tier-2-only arm's. `bakeoff.ts` says the
 * same thing beside `PlannedArm.segments`.
 *
 * All five fields are OPTIONAL because `JudgeCallRecord` types all five
 * `| undefined`, so a row that carried a fabricated 0 would be worse than a row
 * that says nothing. `usage` is optional on a completion, which is where four
 * of them come from. `finishReason` needs its own reason, and it is NOT that
 * `finish_reason` can be null: READ from the installed 0.2.84 declarations, the
 * non-streaming `ChatCompletion.Choice.finish_reason` is required and
 * non-nullable, and the nullable declaration is on
 * `ChatCompletionChunk.Choice`, which the pinned recipe never requests ("No
 * `stream`", engine.ts). The real reason is that the DECLARED type is narrower
 * than the bundle's behaviour: `engine.ts` traced the field to
 * `LLMChatPipeline.getFinishReason()`, declared `ChatCompletionFinishReason |
 * undefined`, so undefined reaches a caller through a field TypeScript says is
 * always present. (A null could not be recorded here in any case: MEASURED with
 * zod 4.4.3, `.optional()` REJECTS null.)
 *
 * `.optional()` and not `.nullable()` because JSON.stringify DROPS an
 * undefined-valued key, so an absent key is what the file actually holds.
 */
const Tier2CallSchema = z.object({
  /** `choices[0].finish_reason` for THIS call, never the message's. */
  finishReason: z.enum(TIER2_FINISH_REASONS).optional(),
  /** `usage.prompt_tokens`, verbatim. */
  promptTokens: z.number().int().nonnegative().optional(),
  /** `usage.completion_tokens`, verbatim. */
  completionTokens: z.number().int().nonnegative().optional(),
  /**
   * `usage.extra.time_to_first_token_s` in milliseconds -- or `null` when the
   * engine reported one that is not a finite number.
   *
   * The null is not decoration and this is the one field that needs it.
   * `JudgeCallRecord` says the conversion is deliberately unguarded, because a
   * NaN there is a fact about the call rather than a number to invent a
   * replacement for. MEASURED with zod 4.4.3: `z.number()` rejects NaN and
   * Infinity, and `JSON.stringify(NaN)` is the string "null". So a NaN copied
   * straight onto a record produces a FILE ITS OWN READER REFUSES -- written as
   * null, rejected on the way back in, after the run. `runArm` maps it here
   * instead, where the value can be named: null means the engine reported a
   * time-to-first-token that is not a number, and absent means it reported
   * none.
   *
   * The three fields above are NOT nullable, deliberately: they come from
   * `usage`, which is either present with real counts or absent altogether, so
   * a null there would be a third state nothing produces.
   */
  ttftMs: z.number().nullable().optional(),
  /**
   * `usage.extra.decode_tokens_per_s` -- or `null` when the engine reported one
   * that is not a finite number.
   *
   * THE ONLY SOURCE for the bake-off's `minDecodeTokPerSec` gate. A record
   * carries `timings.tier2Ms` for the whole message and no per-call elapsed
   * time, so the only rate derivable from the rest of this file is
   * `completionTokens / (tier2Ms - ttft)`, which charges the judge's prompt
   * assembly, JSON parse and span ladder to the model. That gate is a FLOOR, so
   * that error kills capable arms; `bakeoff.ts` reads this field instead.
   *
   * VERIFIED in the installed 0.2.84 bundle, not assumed: the library assigns
   * `decode_tokens_per_s: completion_tokens / decode_time` with `decode_time =
   * pipeline.getCurRoundDecodingTotalTime()` -- an accumulation over this
   * round's decode steps only, so no prefill and no harness time is inside it.
   *
   * Nullable for exactly the reason `ttftMs` is, and it is MORE reachable here:
   * that division has no zero guard, and a call interrupted before its first
   * token has `completion_tokens` 0 and `decode_time` 0, so 0/0 is NaN. Null is
   * that fact spelled so it survives `JSON.stringify`; absent means the engine
   * reported no usage at all. `.nonnegative()` and NOT `.int()`: it is a rate,
   * and a division of two non-negative quantities is never negative.
   */
  decodeTokPerSec: z.number().nonnegative().nullable().optional(),
} satisfies Record<keyof JudgeCallRecord, z.ZodType>);

/**
 * The FULLY RESOLVED tier-2 settings the engine in the page was loaded with,
 * plus the per-call budget the judge over it was constructed with. Present
 * exactly when tier 2 ran.
 *
 * The tier-1 twin of this field exists because `TierConfig` has room for three
 * of tier 1's six ladder dimensions, and "two arms differing only in
 * `threshold`, `maxWidth` or `labelForm` used to emit records that were
 * byte-identical in every field a scorer can group by". Tier 2 had the same
 * disease and no field for it: `TierConfig` carries `t2Model` and nothing else,
 * so an arm run at a 4,096-token window and one run at 8,192 -- which is
 * exactly the fallback Plan 5 names for a model that cannot take 8,192, "run
 * that arm at 4,096 and report the asymmetry" -- were indistinguishable in the
 * output, and so were two arms at different per-call budgets.
 *
 * `callBudgetMs` is here rather than only on the load report because it is the
 * number `deadlineExpiries` on this row is counted AGAINST: a
 * `call-budget-exhausted` notice means the call did not answer within this
 * many milliseconds, and a row that does not say how many describes an
 * expiry nobody can size.
 *
 * What lands here is what the PAGE reported back after resolving -- the object
 * `createWebLlmEngine` was handed, after `resolveTier2Config` filled in and
 * validated every field -- and never the partial object an arm asked for. Same
 * distinction `backend` is a warning about: intent is not evidence of what ran.
 * It is still a REQUEST in one respect the record cannot fix, and the load
 * report says so: 0.2.84 exposes no accessor for the window an engine is
 * enforcing, so `contextWindowSize` is what the engine was asked for.
 * `probeContextWindow` in the page is the only channel that measures it.
 */
const Tier2RunConfigSchema = z.object({
  modelId: z.string().min(1),
  /** Restates resolveTier2Config's own bound: a positive integer. */
  contextWindowSize: z.number().int().positive(),
  /** 0 on every arm this repo can run -- resolveTier2Config refuses any other. */
  temperature: z.number().min(0),
  maxTokens: z.number().int().positive(),
  /**
   * The judge's per-CALL budget, not `ir.latencyBudgetMs`. Bounded above by
   * `MAX_BUDGET_MS`, which is what `WebLlmJudge`'s constructor enforces: a
   * larger number becomes a ~1 ms deadline in `setTimeout` rather than a longer
   * one.
   */
  callBudgetMs: z.number().positive().max(2_147_483_647),
} satisfies Record<keyof Tier2Config | "callBudgetMs", z.ZodType>);

/** Every counter is a non-negative integer; deltas over counters that only rise. */
const JUDGE_COUNTER = z.number().int().nonnegative();

/**
 * The tier-2 judge's own numbers for THIS item, as a DELTA over one `detect`.
 *
 * A DELTA, and that is the trap this field is most likely to be broken by.
 * `WebLlmJudge.stats` is CUMULATIVE across every `judge()` call the judge has
 * made -- its own docblock says so -- so a record populated from it directly
 * carries the arm's running totals, every row after the first is inflated, and
 * the suite stays green because nothing else knows what the numbers should be.
 * The page already solves this the way tier 1 does: `judgeDelta` in
 * apps/eval/src/page/main.ts subtracts the snapshot taken before the call, and
 * `tier2Status().lastDetect` is what `runArm` reads.
 *
 * WHY THE RECORD CARRIES IT, in one line borrowed from `tier1Stats`: findings
 * alone cannot separate "the model ran and found nothing" from "the model never
 * ran". A timing narrows that and does not settle it. `orchestrator.ts` sets
 * `timings.tier2Ms` only on the branch that CALLS the judge -- so an absent one
 * really does mean no judge ran -- but a present one says the JUDGE ran, not
 * that the ENGINE was asked anything: `WebLlmJudge.judge` returns an empty
 * verdict before touching the engine when the IR declares no
 * `semanticPredicates`, and a first call the caller had already aborted is
 * counted and filed with no call row. `calls` is what separates either from a
 * model that answered.
 *
 * The field set is `JudgeStats` WHOLE and the `satisfies` enforces it, so there
 * is no judgement here about which counter matters: a counter added upstream
 * fails to compile rather than going missing. That is worth stating because the
 * plan's own snippet named eight counters, and `JudgeStats` carries fifteen
 * fields -- Task 6 added the truncated/aborted response counts and the per-call
 * rows, Task 7 added the segment accounting and split the two caller-abort
 * counters, which `cancel.ts` refuses to merge because one message for both
 * "would state a falsehood in two of them".
 *
 * Read `segmentsJudged` before reading any of the loss counters. Zero means no
 * segment's answer was ever collected on this item -- an escalation that
 * selected nothing does that legitimately -- in which case the loss counters
 * are zero because there was nothing to lose, not because nothing was lost.
 */
const Tier2StatsSchema = z.object({
  /** Findings whose quote resolved uniquely in FOLDED space; the strong case. */
  rung1: JUDGE_COUNTER,
  /** Findings whose quote only matched after the ladder peeled its tail. Weaker. */
  rung2: JUDGE_COUNTER,
  /** Quotes the ladder refused, for any of its four reasons. */
  unresolvedQuotes: JUDGE_COUNTER,
  /** Findings naming a predicate the IR does not declare. Models invent ids. */
  unknownPredicates: JUDGE_COUNTER,
  /** Findings resolving to a span this run had already emitted. */
  duplicatesDropped: JUDGE_COUNTER,
  /** Segments that got a second call because the first answer would not parse. */
  repairAttempts: JUDGE_COUNTER,
  /** Segments the engine ANSWERED that still yielded no judgement. */
  failedClosed: JUDGE_COUNTER,
  /** Completions the engine reported cut off, by finishReason "length". */
  truncatedResponses: JUDGE_COUNTER,
  /** Completions the engine reported as interrupted. */
  abortedResponses: JUDGE_COUNTER,
  /** Segments whose answer parsed and was collected -- the denominator for recall. */
  segmentsJudged: JUDGE_COUNTER,
  /** Segments a stop ended the run before reaching, plus the one whose call raised it. */
  segmentsSkipped: JUDGE_COUNTER,
  /** Runs stopped by a budget expiry; at most 1 per judge() call by construction. */
  deadlineExpiries: JUDGE_COUNTER,
  /** Caller aborts that really interrupted a generation. */
  callerAbortsMidGeneration: JUDGE_COUNTER,
  /** Caller aborts while the call was still queued: nothing ran, nothing was interrupted. */
  callerAbortsWhileQueued: JUDGE_COUNTER,
  /** One row per engine call that ANSWERED, in the order they were made. */
  calls: z.array(Tier2CallSchema),
} satisfies Record<keyof JudgeStats, z.ZodType>);

export const RunRecordSchema = z
  .object({
    schemaVersion: z.literal(RECORD_SCHEMA_VERSION),
    runId: z.string().min(1),
    itemId: z.string().min(1),
    /** The corpus item's `policy` name, carried through. See CorpusItemSchema. */
    policy: z.string().min(1),
    /**
     * sha256 of the IR ARTIFACT that ran, lowercase hex: the bytes of the JSON
     * the page loaded, hashed by the page itself (apps/eval/src/page/main.ts).
     *
     * Named `irHash`, not `policyHash`, because `PolicyIr.policyHash` already
     * exists and means something else -- it is the field directly below. Two
     * different values sharing one name across one codebase would leave Plan 8
     * joining a record to an IR, finding the two disagree, and having no way to
     * tell an intended difference from corruption.
     *
     * Required, not optional: an arm's numbers are meaningless without knowing
     * exactly which compiled IR produced them, and "which IR was that?" is
     * unanswerable after the fact. Hashing the artifact's own bytes is also what
     * makes the answer checkable from outside the browser -- `shasum -a 256` on
     * the IR file reproduces it, so a reader can confirm the provenance instead
     * of taking the record's word for it.
     */
    irHash: z.string().regex(/^[0-9a-f]{64}$/),
    /**
     * `PolicyIr.policyHash` carried verbatim -- the COMPILER's sha256 of the
     * policy DOCUMENT (packages/compiler/src/stages/emit.ts). It restores the
     * link `irHash` cannot: `irHash` says which IR ran, this says which prose
     * that IR was compiled from, and scoring wants the whole
     * document -> IR -> numbers chain.
     *
     * It cannot stand in for `irHash`, and the reason is stronger than compiler
     * version drift: compilation is MODEL-DRIVEN, so the same document compiled
     * twice by the same compiler can yield two different IRs both carrying this
     * same value. It identifies the input, never the artifact.
     *
     * Deliberately NOT constrained to /^[0-9a-f]{64}$/ the way `irHash` is, even
     * though a compiled IR's really is 64 hex. This value is copied out of
     * whatever IR the page loaded, and a hand-written fixture legitimately
     * carries a placeholder -- apps/eval/fixtures/minimal-ir.json says
     * "test-hash". Tightening this would force that fixture to state the hash of
     * a policy document that does not exist: the harness lying about its own
     * provenance so its own schema would pass. `min(1)` is exactly what core's
     * PolicyIrSchema requires of the field (packages/core/src/policy/schema.ts),
     * so this is as strict as the value's own source and no stricter.
     */
    policyHash: z.string().min(1),
    arm: z.string().min(1),
    backend: z.enum(["wasm", "webgpu"]),
    provider: z.string().min(1),
    /**
     * The message the offsets below index into, copied verbatim from the corpus
     * item. This is what makes "scores standalone, without a join" true rather
     * than aspirational: without it a reader holds spans and no text to index,
     * so the UTF-16 cross-check every offset comment points at is impossible
     * from a record alone.
     *
     * MEASURED cost at projected Plan 7/8 volume (1,500 items x 8 arms x 3
     * policies = 36,000 records): the baseline record is ~456 B, so the run is
     * ~16.4 MB without text. Carrying text adds ~3.7 MB at this corpus's
     * ~93 B mean message (20.1 MB total, +23%) and ~18.4 MB at a 500 B mean
     * (34.8 MB, +112%). Each message is repeated 24 times, once per
     * arm x policy. Tens of megabytes for a file Python reads once is a price
     * worth paying to make every record self-verifying.
     */
    text: z.string().min(1),
    /**
     * The `TierConfig` handed to `detect` for this item, recorded because
     * otherwise the record asserts an arm NAME and nothing else. `arm` is a free
     * string the caller chooses; without the config beside it, a run with every
     * tier switched off is byte-for-byte identical to a detector that legitimately
     * found nothing, and both read as "this arm scored zero". This is the same
     * disease as `backend` -- intent recorded in place of fact -- except that here
     * the fact is available, because this object is the exact one `detect`
     * received.
     *
     * Kept in step with TierConfig in packages/core/src/detect/types.ts. The three
     * booleans are required there and so are required here; the model names and
     * `backend` are optional there and stay optional, so a tier-0 arm does not
     * have to invent values for tiers it never ran.
     */
    config: z.object({
      tier0: z.boolean(),
      tier1: z.boolean(),
      tier2: z.boolean(),
      t1Model: z.string().optional(),
      t2Model: z.string().optional(),
      backend: z.enum(["wasm", "webgpu"]).optional(),
      /**
       * Spec 4.1's escalation threshold: the confidence below which a tier-0/1
       * finding sends its segment to tier 2.
       *
       * REQUIRED on a tier-2 record; on any other kind of arm this schema does
       * not care either way, and the refine at the bottom says why that
       * asymmetry is deliberate rather than an oversight. `TierConfig` calls
       * this an EXPERIMENT variable rather than
       * policy, and says the bake-off's whole job is varying it per arm; two
       * tier-2 arms differing only in this value would otherwise emit records
       * identical in every field a scorer can group by. Same disease as the
       * three tier-1 ladder dimensions `TierConfig` has no room for, arriving
       * through the one dimension it does.
       *
       * The value here is RESOLVED, never the partial one an arm asked for:
       * `runArm` fills in `UNCERTAIN_BELOW` when the caller omits it and hands
       * the same object to `detect`, so this is the number `escalate.ts`
       * compared against and not a claim about a default.
       *
       * Bounds restate `uncertainSegmentStarts`' own: a finite number in
       * [0, 1], which it throws outside of. Both ends are legal there -- 0
       * switches the uncertainty branch off for an arm that escalates on
       * predicates alone, 1 escalates everything short of total certainty. NaN
       * is the value the bound is really for: `confidence < NaN` is false for
       * every finding, so an unvalidated NaN turns the branch off silently and
       * every message reports as having nothing uncertain. MEASURED with zod
       * 4.4.3, `z.number()` rejects NaN and Infinity on its own.
       */
      uncertainBelow: z.number().min(0).max(1).optional(),
    }),
    /**
     * The FULLY RESOLVED `Tier1Config` the tagger in the page was constructed
     * with. Present exactly when tier 1 ran, absent otherwise.
     *
     * `config` above is core's `TierConfig`: three booleans, two model names and
     * a backend. The tier-1 ladder has six dimensions -- modelId, precision,
     * backend, threshold, maxWidth, labelForm -- and three of them appear
     * nowhere in `TierConfig`. So two arms differing only in `threshold`,
     * `maxWidth` or `labelForm` used to emit records that were byte-identical in
     * every field a scorer can group by: a matrix that ran four arms and handed
     * Plan 8 four indistinguishable ones.
     *
     * What lands here is the config off `loadTier1`'s report -- the object the
     * GlinerSpanTagger was constructed with, after `resolveTier1Config` filled in
     * and validated every field -- and never the partial object an arm asked
     * for. Same distinction `backend` above is a warning about: intent is not
     * evidence of what ran.
     *
     * The vocabularies are IMPORTED from @sih/tier1 rather than restated, unlike
     * core's unions higher up in this file. Those are types, with no runtime
     * value to import; `TIER1_BACKENDS` and `TIER1_LABEL_FORMS` are exported
     * arrays, so importing them makes drift impossible instead of merely
     * discouraged.
     */
    tier1Config: z
      .object({
        modelId: z.string().min(1),
        backend: z.enum(TIER1_BACKENDS),
        /** Restates resolveTier1Config's own bound, (0, 1]. */
        threshold: z.number().gt(0).lte(1),
        maxWidth: z.number().int().positive(),
        labelForm: z.enum(TIER1_LABEL_FORMS),
      })
      .optional(),
    /**
     * The tier-1 tagger's own counters for THIS item, as a delta over the one
     * `detect` call. Present exactly when tier 1 ran and returned; see the
     * refine at the bottom for the coupling and why `error` is part of it.
     *
     * WHY THE RECORD CARRIES THEM. `Tier1TaggerStats` in
     * packages/tier1/src/tagger.ts says what these are for in as many words: "a
     * silently dropped span is indistinguishable from a model that found
     * nothing. An eval arm reporting recall has to be able to tell those
     * apart." Until this field existed the arm could not: `runArm` built every
     * record from `findings` and `timings` alone, so an item whose tail
     * `maxLen` cut off, or whose span the offset mapper refused, emitted a row
     * byte-identical to one where the model ran over the whole message and
     * found nothing. That is the exact recall miss the counters were introduced
     * to expose, and it was being thrown away one layer above them.
     *
     * The field set is `Tier1TaggerStats` WHOLE, not a chosen subset, so there
     * is no judgement here about which counter matters -- a counter added there
     * fails typecheck here rather than going missing. `gpuSubmits` is the one
     * thing on the page's `Tier1DetectStats` that is deliberately absent: it
     * counts the PAGE's `GPUQueue.submit` calls, not the tagger's work, and
     * `loadTier1` already refuses to finish when it disagrees with `backend`.
     *
     * Read `inferences` before reading any of the others. Zero means the graph
     * never ran on this item -- a message with no prose segment does that
     * legitimately -- in which case the four loss counters are zero because
     * there was nothing to lose, not because nothing was lost.
     */
    tier1Stats: z
      .object({
        /** `session.run` calls. Zero means the graph never ran for this item. */
        inferences: z.number().int().nonnegative(),
        /** Words that tokenised to nothing and lost their seat on the model's axis. */
        droppedWords: z.number().int().nonnegative(),
        /** Trailing words `maxLen` cut off before they reached the graph. */
        truncatedWords: z.number().int().nonnegative(),
        /** Decoded spans wider than `tier1Config.maxWidth`. markerV0 only; see tagger.ts. */
        overWideSpans: z.number().int().nonnegative(),
        /** Decoded spans the offset mapper refused to turn into a character range. */
        unmappableSpans: z.number().int().nonnegative(),
        /** Decoded spans whose score was not a finite number in 0..1. */
        nonFiniteScores: z.number().int().nonnegative(),
      })
      .optional(),
    /**
     * What the tier-2 judge did on THIS item, as a delta; see Tier2StatsSchema
     * for the field set, why it is a delta, and why counters alone are not
     * enough. Present exactly when tier 2 ran and returned -- the same coupling
     * `tier1Stats` uses, and for the same reason `error` is part of it.
     */
    tier2Stats: Tier2StatsSchema.optional(),
    /**
     * The resolved tier-2 settings this arm ran under; see
     * Tier2RunConfigSchema. Present exactly when tier 2 ran, and coupled to
     * `config.tier2` alone rather than also to `error` -- the same asymmetry
     * `tier1Config` has, and for the same reason: this describes the engine the
     * page HOLDS, which is known before the item runs and stays true whatever
     * the item does, while `tier2Stats` describes what the judge DID on it.
     */
    tier2Config: Tier2RunConfigSchema.optional(),
    findings: z.array(RecordFindingSchema),
    /**
     * Copied from the corpus item so a record scores standalone, without a join.
     * Same UTF-16 code-unit offsets as it had in the corpus -- nothing rebases
     * it -- and, like gold in the corpus, NOT necessarily sorted or disjoint,
     * unlike `findings`.
     */
    gold: z.array(GoldSpanSchema),
    timings: z.object({
      tier0Ms: z.number(),
      tier1Ms: z.number().optional(),
      tier2Ms: z.number().optional(),
    }),
    /**
     * `DetectionResult.degraded` carried whole; see DegradedNoticeSchema for
     * what each entry means and why no counter substitutes for it.
     *
     * Present exactly when `error` is null, and the asymmetry with core is
     * deliberate. There the field is REQUIRED and its doc explains why -- "[]
     * is a positive claim; undefined would be silence" -- and that reading is
     * exactly what makes it wrong on a thrown item: `detect` throws whole, so
     * there is no result to read an array off, and an empty one would be the
     * positive claim that nothing was skipped on an item that may never have
     * finished. Absent says the honest thing.
     */
    degraded: z.array(DegradedNoticeSchema).optional(),
    /**
     * Set when detection THREW for this item. The record is still written: an arm
     * that crashes on 5% of the corpus and one that scores 0 on it are different
     * results, and dropping the row makes them look identical.
     */
    error: z.string().nullable(),
    /**
     * True when an EARLIER item in this arm blew its deadline, so this row's
     * `timings` were taken while that item's work was still running.
     *
     * `page.evaluate` accepts no timeout and offers no cancellation channel, so
     * a deadline expiry bounds the DRIVER's wait and nothing else -- the
     * abandoned detection keeps executing in the browser. Every item after it
     * is therefore measured under contention with work that belongs to a
     * different row, and, because the timed-out row is the only one that gets
     * an `error`, all the contaminated rows have `error === null`. A latency
     * aggregate over `error === null` rows silently includes them. This flag is
     * what lets that aggregate exclude them.
     *
     * WHAT ELSE A FLAGGED ROW IS SHORT OF, and the earlier version of this
     * paragraph had it backwards. It claimed the findings on a flagged row are
     * still good because "`detect` is deterministic given the message, the IR
     * and the config". On a tier-0 or tier-1 arm that holds. On a TIER-2 arm
     * neither half does, and both were MEASURED against the real orchestrator:
     *
     *   1. THE FINDINGS MOVE. The orchestrator arms one wall-clock deadline
     *      over the whole `judge()` call from `ir.latencyBudgetMs`, so the same
     *      message, IR and config yield FEWER findings when the clock is
     *      slower. Measured with an engine whose only difference was 5 ms
     *      against 100 ms per call: two tier-2 findings became one, and the
     *      result gained a `budget-exhausted` notice and two skipped segments.
     *      Contention is exactly a slower clock.
     *   2. THE COUNTERS MAY NOT BE THIS ROW'S. `tier2Stats` and `tier1Stats`
     *      are DELTAS taken in the page around a `detect` on a SHARED judge and
     *      a shared tagger. An abandoned item is still running on that judge,
     *      so its calls and counters land inside the next item's delta window.
     *      Measured: an item that makes 3 calls of its own reported 6.
     *
     * So a flagged row is not merely a row with a bad timing. Read `degraded`
     * on it before scoring it: a `budget-exhausted` or `call-budget-exhausted`
     * notice says the judgement was cut short, and `tier2Stats` on the row
     * after an abandoned one may hold two items' work. `bakeoff.ts` counts
     * flagged rows in `itemsAbandonedWorkInFlight` and does NOT exclude them
     * from its ladder or its latency sample -- that module filters nothing --
     * so a nonzero count there is a reason to distrust the whole arm's tier-2
     * aggregates, not just its latencies.
     *
     * A FLAG RATHER THAN ABORTING THE ARM, and the choice is not obvious.
     * Aborting is defensible -- a wedged item means the arm's latencies are not
     * measuring the arm -- and it is what `runArm` already does for a dead
     * browser. It is rejected here for one reason: a deadline expiry is a
     * per-item event that says nothing about the items already measured, and
     * aborting throws all of them away AND, through `runMatrix`, takes the rest
     * of the matrix with it. One slow item at position 1,400 of 1,500 would
     * destroy hours of GPU time that had already produced good rows. The flag
     * keeps every row and hands the decision to the side that is doing the
     * scoring, which is where spec 2.2 puts every other judgement in this file.
     *
     * `false` on every row of an arm that never timed out, which is the normal
     * case; a run where this is true anywhere should be treated as a
     * misconfigured deadline to fix, not a tolerable outcome.
     */
    abandonedWorkInFlight: z.boolean(),
  })
  .refine(
    (r) =>
      [...r.findings, ...r.gold].every(
        (s) =>
          // `start < end` is checked explicitly, not left to the slice
          // comparison. `end` being positive was meant to bar degenerate spans
          // and an inverted `start` walks around it: slice() clamps an inverted
          // range to "" rather than throwing, so {start:999,end:1} agrees with
          // a claimed text of "" and an out-of-range start never surfaces.
          s.start < s.end &&
          s.end <= r.text.length &&
          r.text.slice(s.start, s.end) === s.text,
      ),
    { message: "a span's offsets do not hold the text it names" },
  )
  // The four checks below tie the record's LABELS to the configuration that
  // produced them. Each one is a way a complete, schema-valid, perfectly
  // scoreable file can describe a run that did not happen, and each was
  // reachable before Task 12: nothing validates a record on the producing path,
  // so the writer runs this schema and these are the checks it runs.
  .refine((r) => r.config.backend === undefined || r.config.backend === r.backend, {
    message: "config.backend contradicts the record's own backend",
  })
  .refine((r) => (r.tier1Stats !== undefined) === (r.config.tier1 && r.error === null), {
    // Coupled to `error` as well as to `config.tier1`, unlike `tier1Config`
    // below, and the asymmetry is the honest one. `tier1Config` describes the
    // tagger the page HOLDS, which is known before the item runs and stays true
    // whatever the item does. `tier1Stats` describes what the tagger DID on
    // this item, and `detect` throws whole -- so on a thrown or timed-out item
    // the page's `lastDetect` still holds the previous item's delta and there
    // is no per-item answer to give. Absent says that; a row of zeros would
    // assert that nothing was truncated, dropped or unmappable on an item where
    // the model may never have finished.
    message: "tier1Stats must be present exactly when config.tier1 is true and error is null",
  })
  .refine((r) => r.config.tier1 === (r.tier1Config !== undefined), {
    // Both directions. Tier 1 on with no config is an arm whose six-dimensional
    // rung went unrecorded; a config with tier 1 off is a tier-0 run wearing a
    // tier-1 label, which is the exact substitution runArm refuses to navigate
    // in order to prevent.
    message: "tier1Config must be present exactly when config.tier1 is true",
  })
  .refine((r) => r.tier1Config === undefined || r.tier1Config.backend === r.backend, {
    message: "tier1Config.backend contradicts the record's own backend",
  })
  .refine(
    (r) =>
      r.tier1Config === undefined ||
      r.config.t1Model === undefined ||
      r.config.t1Model === r.tier1Config.modelId,
    { message: "config.t1Model names a different rung than tier1Config.modelId" },
  )
  .refine((r) => (r.tier2Stats !== undefined) === (r.config.tier2 && r.error === null), {
    // The tier-1 coupling verbatim, one tier up, including the `error` half:
    // `tier2Status().lastDetect` is a DELTA over the judge's cumulative
    // counters, so on a thrown or timed-out item that delta belongs to the
    // PREVIOUS item and there is no per-item answer to give. A row of zeros
    // would assert that nothing failed closed, nothing was truncated and no
    // quote went unresolved on an item whose judge may never have returned.
    message: "tier2Stats must be present exactly when config.tier2 is true and error is null",
  })
  .refine((r) => r.config.tier2 === (r.tier2Config !== undefined), {
    // Both directions, exactly as `tier1Config`. Tier 2 on with no config is an
    // arm whose window, token ceiling and per-call budget went unrecorded --
    // and a run that deliberately puts one arm at a 4,096-token window would be
    // indistinguishable from a symmetric one. A config with tier 2 off is a
    // lower-tier run wearing a tier-2 label.
    message: "tier2Config must be present exactly when config.tier2 is true",
  })
  .refine(
    (r) =>
      r.tier2Config === undefined ||
      r.config.t2Model === undefined ||
      r.config.t2Model === r.tier2Config.modelId,
    { message: "config.t2Model names a different model than tier2Config.modelId" },
  )
  .refine((r) => (r.degraded !== undefined) === (r.error === null), {
    // Not coupled to any tier switch, unlike the two stats fields: every result
    // has a degradation account, and a tier-0 arm's is the two `absent`
    // entries that say which tiers its empty `findings` is silent about.
    message: "degraded must be present exactly when error is null",
  })
  .refine((r) => !r.config.tier2 || r.config.uncertainBelow !== undefined, {
    // One direction only. A tier-2 row without it cannot be compared with the
    // row beside it, which is the failure being closed. The other direction is
    // deliberately left open: a caller may hand `detect` a threshold on an arm
    // that never reaches escalation, and the record's job is to state what
    // detect received rather than to tidy it away.
    message: "a tier-2 record must carry the escalation threshold that produced it",
  });

export type RunRecord = z.infer<typeof RunRecordSchema>;
/** The resolved tier-2 settings a record states, as `runArm` receives them. */
export type Tier2RunConfig = z.infer<typeof Tier2RunConfigSchema>;
export type RecordFinding = z.infer<typeof RecordFindingSchema>;

/**
 * Serializes records as JSONL, one per line, with a terminating newline.
 *
 * U+2028 and U+2029 are escaped explicitly because `JSON.stringify` does NOT
 * escape them -- they are valid unescaped JSON string content -- while Python
 * treats both as line terminators. MEASURED: a record whose text contains a raw
 * U+2028 is emitted as one line by this function, and Python's
 * `data.splitlines()` then yields TWO fragments, both of which fail
 * `json.loads` with "Unterminated string". That is a loud failure rather than a
 * silent one, so it ranks well below the offset hazard -- but it costs one
 * replace to remove, and scraped chat is exactly where a stray U+2028 comes
 * from. `\\u2028` is valid JSON escape syntax, so the output stays parseable by
 * any conformant reader.
 */
export function toJsonl(records: readonly RunRecord[]): string {
  return (
    records
      .map((r) => JSON.stringify(r).replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029"))
      .join("\n") + "\n"
  );
}
