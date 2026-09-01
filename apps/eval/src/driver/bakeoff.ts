import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import type { Page } from "@playwright/test";
import {
  UNCERTAIN_BELOW,
  loadPolicyIr,
  runTier0,
  segmentText,
  type DegradedReason,
  type Finding,
  type PolicyIr,
  type Tier,
  type TierConfig,
} from "@sih/core";
import { TIER2_MODELS, resolveTier2Config } from "@sih/tier2";
import { z } from "zod";
import { loadCorpus, type CorpusItem } from "./corpus.js";
import { assertFileSafe } from "./main.js";
import { RunRecordSchema, toJsonl, type RunRecord } from "./record.js";
import { runArm } from "./run.js";
import {
  percentile,
  segmentSizeDistribution,
  sizeStats,
  type SegmentSizeDistribution,
  type SizeStats,
} from "./segments.js";

/**
 * The bake-off: every arm over one corpus, one JSONL file each, and a gate
 * verdict beside every file.
 *
 * ## The rule this module is built around
 *
 * **Gates are COMPUTED, never enforced by dropping data.** Every arm writes its
 * file whatever its numbers say; the verdict is a FIELD on a report that is
 * written alongside. An arm that fails a gate is a RESULT, and omitting it would
 * make the bake-off look like it had fewer contenders than it did. What to do
 * with a killed arm is Plan 8's decision and it needs the rows to make it.
 *
 * ## The kill rule that is deliberately absent
 *
 * There is no gate on the message latency budget, and there must not be. Task 9
 * measured this corpus's escalation and Plan 5 measured the per-call cost: at
 * 4.6 s per call on the CHEAPEST pinned arm, a message that escalates two
 * segments costs 9.2 s against `ir.latencyBudgetMs` -- and the compiler's
 * `DEFAULT_LATENCY_BUDGET_MS` is 5,000, which is what `minimal-ir.json` and
 * `multiclass-ir.json` carry. (`semantic-ir.json` raises it to 120,000 and its
 * own comment says why: at 5,000 the deadline fires during the first call of
 * every tier-2 spec, so nothing would ever exercise a completed judgement. That
 * is a lifecycle fixture's number, not a claim that tier 2 fits 5 s.) So tier 2
 * expires
 * mid-run on most messages that escalate more than one segment WHATEVER arm
 * wins. A rule keyed on that would kill all four arms for a reason that is not
 * about capability -- which is exactly why the spec's original gates were
 * replaced. `budget-exhausted` and `scope-unjudged` are COUNTED per arm and
 * reported beside the verdict instead; see `ArmGateReport.degradedNotices`.
 *
 * ## What this module does not compute
 *
 * Any accuracy number. Per spec 2.2 the JSONL file is the whole boundary and
 * scoring is Plan 8's Python. The gates here are all properties of the RUN --
 * latency, decode rate, how often the span ladder could place a quote, how often
 * an answer was a restatement of one already given -- and every one of them is
 * readable off a record without a gold label. That is deliberate: the smoke
 * corpus's gold is labelled `policy: "minimal-fixture"` and no gold exists for
 * the semantic predicate at all, so an accuracy gate here would be scoring
 * against labels written for a different policy.
 *
 * ## The two things that absence is NOT allowed to look like
 *
 * Both were live defects until the commit that added `ArmScoringBoundary`, and
 * both produced a complete, well-formed, internally consistent file.
 *
 *   1. **A silent zero-precision arm.** The corpus has no tier-2 gold and the
 *      corpus TEXT satisfies the tier-2 predicate, so a scorer joining a
 *      record's `findings` to its `gold` counts every CORRECT tier-2 finding as
 *      a false positive and ranks the arm that found nothing first. Every gates
 *      row now names the tiers this corpus cannot score, and why, in
 *      `scoring.tiersThisCorpusCannotScore` and `scoring.cannotScore`. The
 *      remedy is a named result rather than a refusal to run, and rather than
 *      new gold; `ArmScoringBoundary` argues both.
 *   2. **A verdict that reads as a selection.** `killed` was renamed
 *      `killedOnRunGates` and `scoring.verdictMeans` states the boundary on the
 *      row, because a reader of a gates file has no other way to learn that
 *      spec 4.2's primary criterion -- task accuracy -- was never applied.
 */

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

export const GATES = {
  /**
   * p95 time-to-first-token at the MEASURED tier-2 segment size.
   *
   * Task 9 measured that size over `corpora/fixtures/smoke.jsonl`, the only
   * corpus in this repository, keeping the segments the escalation policy
   * selects: p50 62 characters and max 153.
   *
   * RE-MEASURED HERE against the shipped `semantic-ir.json` under BOTH of the
   * conditions this driver runs, because Task 9's figure was one of them: the
   * tier-2-only families select 17 of the corpus's 19 segments and the tier-0
   * families 18, and the character distribution is the same in both -- p50 62,
   * p95 153, max 153, min 22 -- because the segment tier 0 re-admits (the
   * 66-character code fence) is larger than nine of the seventeen and smaller
   * than the largest, so it moves neither end. The WORD median does move, 9
   * without priors and 8 with, which is why the number above is quoted in
   * characters. What the extra segment does move is the per-MESSAGE count, and
   * that is `segmentsPerItem`'s business rather than this threshold's.
   *
   * MEASURED HERE for the whole prompt, by driving the real `WebLlmJudge` over
   * those two segments with a capturing engine and adding up what
   * `buildMessages` produced: against `apps/eval/fixtures/semantic-ir.json` the
   * WHOLE PROMPT is **1,031 characters at the median segment and 1,122 at the
   * largest**, of which 776 is the fixed system turn. So a TTFT measured
   * against a prompt materially bigger than ~1.1 kB is not measuring this gate,
   * and an arm must not be killed on a number taken at a different prompt size.
   *
   * Plan 5's figures for the same two segments are 1,105 and 1,196, and the
   * difference is not drift: Plan 5 states them "with the one semantic
   * predicate the repo's compiled extraction fixture carries", whose id and
   * `nlPredicate` are 74 characters longer than `semantic-ir.json`'s. The same
   * measurement reproduces both numbers exactly, one predicate each. This
   * driver is hard-required to run `semantic-ir.json` -- `planBakeoff` throws on
   * any IR with no `semanticPredicates` and its own error names that file as
   * the only one here that has one -- so 1,105/1,196 describe work no arm of
   * this bake-off can perform.
   *
   * MEASURED HERE too, through `test/bakeoff.spec.ts` on Qwen3.5-2B over a
   * two-item slice of that corpus: the engine reported 245-256 prompt TOKENS for
   * segments of 28-48 characters. At ~4.5 characters per token that is a prompt
   * of roughly 1.1 kB, which agrees with the character count above -- and it
   * shows where the size comes from: the prompt is dominated by the fixed
   * instructions and the predicate, not by the segment, so a 48-character
   * segment and a 153-character one differ by far less than 3x.
   *
   * This module cannot enforce that comparison and does not pretend to: the
   * prompt is assembled inside the browser and a record carries no copy of it.
   * What it does instead is put the size BESIDE the number, twice over --
   * `ArmGateReport.promptTokens` is the engine's own prompt-token count over
   * exactly the calls the p95 was taken over, and `ArmGateReport.segmentChars`
   * is the character-size distribution escalation selects for the arm, which is
   * the unit the 1.1 kB above is quoted in (and is PLANNED rather than observed
   * -- see that field). A reader who finds either one out of line with the
   * derivation knows the gate was applied to different work.
   */
  maxP95TtftMs: 1500,
  /**
   * Sustained decode rate. Latency here is dominated by output length.
   *
   * DERIVED from the hardware, unlike the two below: `manifest.ts` records a
   * measured decode rate per arm on this machine, and 25 is the third of the
   * four -- Qwen3.5-2B 40, Ministral-3-3B 30, Qwen3-4B 25 (exactly on the
   * line), Phi-4-mini 24. So this floor is a statement about what the slowest
   * measured arm does, and it is worth knowing that it kills Phi-4-mini on the
   * manifest's own number before the arm runs.
   */
  minDecodeTokPerSec: 25,
  /**
   * Semantic correctness: fraction of the quotes a model produced that the span
   * ladder could place, at rung 1 or rung 2.
   *
   * A CHOSEN floor, not a measured one, and this comment says so rather than
   * inventing a derivation. Spec 4.2 asks for "spans resolvable at rung <= 2"
   * and gives no number; no run in this repository has produced a distribution
   * of resolvable rates to set one from, because the bake-off that would is the
   * thing this file is for. 0.8 is a defensible starting point and nothing more.
   *
   * What to do with that: the raw counts are in `ArmGateReport.ladder`
   * (`rung1`, `rung2`, `duplicatesDropped`, `unresolvedQuotes`), so a reader who
   * disagrees with the number can recompute the rate. Revisit this constant
   * against the first real four-arm run rather than treating a kill on it as
   * settled.
   */
  minResolvableRate: 0.8,
  /**
   * Ceiling on the share of resolved findings that were a restatement of a span
   * this run had already emitted.
   *
   * Spec 4.2's words are "no duplicate-ONLY output", which is not a rate, and
   * taken literally as a rate of 1.0 it is unreachable: the first occurrence of
   * a span always counts as `rung1`/`rung2` and only later ones count as
   * duplicates, so `duplicatesDropped` is positive only when the numerator's
   * partner is too. A number is therefore a judgement call, and this one is
   * bracketed by the two duplicate behaviours Plan 5 measured -- THEIR
   * measurements, not ones taken here:
   *
   *   - Qwen3.5-2B, the recommended primary arm, "found only the AWS key,
   *     three times over": 1 distinct span and 2 restatements, a rate of 0.667.
   *     Plan 5 tells the bake-off to EXPECT that ("expect duplicates, expect
   *     misses, and do not tune the corpus to hide either"), so a ceiling under
   *     it kills the primary arm for the behaviour the plan predicted.
   *   - Phi-4-mini's duplicate LOOP, `"quote": "Halcyon"` about nineteen times
   *     until the token budget ran out mid-string: 1 distinct span and ~18
   *     restatements, a rate of about 0.95. That is the pathology worth killing
   *     an arm over -- it spends the whole answer saying one thing again.
   *
   * 0.9 sits between them. It was 0.5 before, which is under the first of those
   * and would have killed the arm Plan 5 recommends. Two points on one message
   * each are not a distribution, and this number should move when a real run
   * produces one; `ArmGateReport.ladder` carries the counts either way.
   */
  maxDuplicateRate: 0.9,
  /** A killed arm is a result and stays in the output. */
  recordKilledArms: true,
  /**
   * After ANY deadline expiry, the next call must return a non-empty body.
   * Task 3 measured the engine latching its interrupt flag on the pinned
   * non-streaming path, after which every later call returns instantly and
   * empty -- which a judge reads as "no findings" forever. `clearInterrupt`
   * fixes it by writing a field TypeScript marks private, so an upstream
   * rename would silently restore the poisoning with a green unit suite.
   * This assertion is the only guard that would notice.
   */
  assertNonEmptyAfterExpiry: true,
} as const;

/**
 * How far past its budget an interrupted call has been measured to settle.
 *
 * Task 3's number, not mine: "the drained call settled 9-18 ms past its 1500 ms
 * budget across two runs" (`packages/tier2/src/cancel.ts`). Rounded up to 20 and
 * used only as headroom on a ceiling that is then doubled, so the rounding
 * direction cannot matter. It is NOT a claim that every drain is this fast on
 * every arm -- the measurement is one model on one machine -- which is another
 * reason the ceiling is doubled rather than taken literally.
 */
export const INTERRUPT_DRAIN_OVERSHOOT_MS = 20;

/**
 * Headroom for the deterministic tiers, which no deadline in the code bounds.
 *
 * MEASURED HERE, in Node, over all 13 items of `corpora/fixtures/smoke.jsonl`,
 * one warm-up pass then 50 timed repetitions each, timing
 * `runTier0(ir, text, segmentText(text))` whole: against `minimal-ir.json`
 * (which has entropy and regex rules and finds 6 spans) the worst single item
 * was 0.033 ms and the mean 0.0023 ms; against `semantic-ir.json` (no rules)
 * the mean was 0.0010 ms with one 0.457 ms outlier whose cause was not
 * investigated. A browser is a different JIT and this is not a browser
 * measurement, but the largest number there is three orders of magnitude below
 * this default, so no plausible browser factor reaches it.
 *
 * TIER 1 is in the name of the field this feeds and is off on every bake-off
 * arm, so nothing above is a tier-1 measurement and none is needed. What the
 * rest of the allowance covers is the `page.evaluate` round trip, which is not
 * free either and which the per-item deadline also bounds.
 */
export const DEFAULT_LOWER_TIER_ALLOWANCE_MS = 1_000;

/**
 * The largest delay `setTimeout` holds without reinterpreting it.
 *
 * A LOCAL copy on purpose, and this is the third one in the repository rather
 * than a shared constant, because each sits beside the timer it guards.
 * `orchestrator.ts` has one for core's message deadline and CLAMPS to it;
 * `cancel.ts` has one for the per-call deadline and REFUSES above it, and its
 * own comment says it is kept off @sih/tier2's index so a consumer cannot hold
 * the bound without holding the guard that enforces it. This one guards
 * `run.ts`'s `withDeadline`, which is a third timer that neither clamps nor
 * refuses.
 *
 * MEASURED HERE on Node v26.0.0 rather than taken from either of them:
 * `setTimeout(fn, 6_000_002_040)` prints `TimeoutOverflowWarning: 6000002040
 * does not fit into a 32-bit signed integer. Timeout duration was set to 1` and
 * fired after 5 ms. So an over-large ceiling is not a long deadline, it is an
 * immediate one -- every item errored and every later row stamped
 * `abandonedWorkInFlight`, which is the exact failure the ceiling exists to
 * avoid.
 *
 * What the summands do downstream, so the reader knows which of them this bound
 * is really for. `ir.latencyBudgetMs` reaches core, which CLAMPS it in
 * `remainingBudgetMs` -- so core is safe from an over-large one on its own.
 * `callBudgetMs` reaches `runWithDeadline`, which refuses one -- but inside the
 * browser, on the first engine call, i.e. after the model load this module
 * exists to plan before. Neither protects the number DERIVED from them here,
 * which is the one that reaches `run.ts`.
 */
const MAX_TIMER_DELAY_MS = 2_147_483_647;

// ---------------------------------------------------------------------------
// Arm families
// ---------------------------------------------------------------------------

/**
 * Which of the two methods an arm belongs to, and which tiers it runs.
 *
 * The four names are TWO PAIRS, and the pairing is the point. Comparing
 * Approach B alone against a tier-0-plus-tier-2 compiled pipeline credits tier
 * 0's deterministic findings to the compiler, which is the confusion
 * `createBaselineBPlusTier0` exists to remove -- "compiling the policy helps"
 * and "having deterministic patterns helps" are different claims. So each
 * baseline family has a compiled family that ran the same deterministic half:
 *
 *   compiled            (tier 0 + compiled judge)  <->  baseline-b-tier0
 *   compiled-tier2-only (compiled judge alone)     <->  baseline-b
 */
export type ArmFamily = "compiled" | "compiled-tier2-only" | "baseline-b" | "baseline-b-tier0";

export interface FamilyShape {
  readonly family: ArmFamily;
  /** Whether core's tier 0 runs in front of the model. */
  readonly runsTier0: boolean;
  /** False for the Approach-B arms, which are their own `Detector`. */
  readonly runsCompiledJudge: boolean;
  /**
   * What ONE engine call covers.
   *
   * The compiled judge makes one call per SELECTED SEGMENT (escalation decides
   * which); Approach B makes one call per MESSAGE, because the whole message is
   * what its model is shown. This is intrinsic to the two methods and is the
   * one place the families legitimately differ in cost -- which is why the
   * shared wall-clock ceiling is sized from the larger of the two rather than
   * from an average that would squeeze the compiled family.
   */
  readonly judgedUnit: "segment" | "message";
  /**
   * Engine calls per judged unit, worst case.
   *
   * TWO on both families and for the same reason: the pinned recipe makes one
   * call, and adds exactly ONE repair retry, taken only when the answer fails
   * to parse. VERIFIED by reading both loops -- `WebLlmJudge.judge` and
   * `createArm` in `baselineB.ts` -- rather than assumed: each sets `repaired`
   * on the first parse failure and fails closed on the second, and any STOP
   * (a budget expiry or a caller abort) ends the whole run rather than the unit,
   * so no third call is reachable.
   */
  readonly callsPerJudgedUnit: 2;
}

const FAMILY_SHAPES: Readonly<Record<ArmFamily, FamilyShape>> = {
  compiled: {
    family: "compiled",
    runsTier0: true,
    runsCompiledJudge: true,
    judgedUnit: "segment",
    callsPerJudgedUnit: 2,
  },
  "compiled-tier2-only": {
    family: "compiled-tier2-only",
    runsTier0: false,
    runsCompiledJudge: true,
    judgedUnit: "segment",
    callsPerJudgedUnit: 2,
  },
  "baseline-b": {
    family: "baseline-b",
    runsTier0: false,
    runsCompiledJudge: false,
    judgedUnit: "message",
    callsPerJudgedUnit: 2,
  },
  "baseline-b-tier0": {
    family: "baseline-b-tier0",
    runsTier0: true,
    runsCompiledJudge: false,
    judgedUnit: "message",
    callsPerJudgedUnit: 2,
  },
};

export function familyShape(family: ArmFamily): FamilyShape {
  const shape = FAMILY_SHAPES[family];
  // Unreachable while `family` is typed, and kept because the options object
  // arrives from a CLI in the end and a string is a string at runtime.
  if (shape === undefined) {
    throw new Error(
      `unknown arm family "${String(family)}"; the slate is ${Object.keys(FAMILY_SHAPES).join(", ")}`,
    );
  }
  return shape;
}

/** The half of an arm's file name that is not the run id. Kept file-safe. */
const FAMILY_SLUGS: Readonly<Record<ArmFamily, string>> = {
  compiled: "tier2",
  "compiled-tier2-only": "tier2only",
  "baseline-b": "baselineB",
  "baseline-b-tier0": "baselineB+tier0",
};

/**
 * `<familySlug>-<modelId>`, which is both the arm's label on every record and
 * half of its file name.
 *
 * The model id is IN the name rather than only in `config.t2Model`, because a
 * directory of files is what leaves this machine and a file whose name does not
 * say which model produced it is unattributable once it does.
 */
export function armName(family: ArmFamily, modelId: string): string {
  return `${FAMILY_SLUGS[family]}-${modelId}`;
}

// ---------------------------------------------------------------------------
// The per-item wall-clock ceiling
// ---------------------------------------------------------------------------

export interface ItemDeadlineInput {
  /** The loaded IR's `latencyBudgetMs`; the orchestrator arms its deadline from it. */
  readonly latencyBudgetMs: number;
  /** The judge's (or B's) per-ENGINE-CALL budget, fixed when the arm was built. */
  readonly callBudgetMs: number;
  /** Worst-case engine calls for one item on this arm. */
  readonly maxCallsPerItem: number;
  /** Headroom for tier 0 and tier 1, which no deadline in the code bounds. */
  readonly lowerTierAllowanceMs: number;
}

export interface ItemDeadlineBound {
  /** Bound (a): the orchestrator's one deadline over the whole `judge()` call. */
  readonly messageBudgetBoundMs: number;
  /** Bound (b): the per-call budget, spent on every call the arm can make. */
  readonly callBudgetBoundMs: number;
  /**
   * Which of the two is the smaller, and therefore the one that stops the run.
   * A tie reads as `message-budget`, because on a tie the message deadline is
   * the one that fires first: it is armed once, at the start, while the call
   * bound is only reached after the last call has also used all of its budget.
   */
  readonly bindingBound: "message-budget" | "call-budget";
  /** The lower tiers plus the binding bound: the longest a GOOD item can take. */
  readonly boundMs: number;
  readonly input: ItemDeadlineInput;
}

/**
 * How long one item can legitimately take, from the two stops the code arms.
 *
 * ## Why this is derived rather than written down
 *
 * `run.ts` is explicit that `itemTimeoutMs` "exists to catch a wedge, not to
 * enforce a latency target", and that "too small is not a safe direction to err
 * in ... it converts slow items into errored records, which scores as a worse
 * arm rather than as a misconfiguration". The cost is worse than one row: the
 * expired row gets an `error`, and every row AFTER it in the arm is stamped
 * `abandonedWorkInFlight`, because `page.evaluate` has no cancellation channel
 * and the abandoned detection keeps running. One badly-sized ceiling therefore
 * invalidates the latency column of a whole arm.
 *
 * ## The two bounds, read off the code rather than guessed
 *
 * The plan derives this number as "3 selected segments x 2 calls per segment x
 * ~10 s per call = ~60 s, doubled", and that multiplies three quantities
 * without reference to either stop the code actually arms. Re-derived here:
 *
 * (a) THE MESSAGE DEADLINE. `detect` starts a clock before tier 0, computes
 *     `remainingBudgetMs(ir.latencyBudgetMs, elapsed)` when tier 2's turn comes,
 *     arms ONE `setTimeout` for that long, and passes its `AbortSignal` into
 *     `judge()`. Any stop the judge sees ends the whole run rather than the
 *     segment, so the judge returns at the first call boundary after the abort.
 *     The overshoot is the interrupt-and-drain, MEASURED by Task 3 at 9-18 ms.
 *     So tier 2 costs at most `latencyBudgetMs + drain`.
 *
 * (b) THE PER-CALL BUDGET. The judge is constructed with a fixed `budgetMs` and
 *     spends it on each call; `judge()` reads `request.budgetMs` for nothing at
 *     all (its own docblock says so). So an arm cannot exceed
 *     `maxCallsPerItem x (callBudgetMs + drain)` no matter how large the message
 *     budget is.
 *
 * The item is stopped by whichever is SMALLER. That is the correction the plan's
 * derivation needs: multiplying calls by a per-call cost, as it does, is bound
 * (b) with a guessed per-call number in place of the budget the code enforces --
 * and on the IR this bake-off must run it is not even the binding one.
 *
 * (c) THE LOWER TIERS, added on top rather than assumed to be inside (a).
 *     `ir.latencyBudgetMs` is checked at TIER 2's turn and nowhere else, so a
 *     slow tier-0 or tier-1 pass overruns it with nothing to stop it. Adding the
 *     allowance double-counts against (a) by design; the direction is safe.
 *
 * ## What the number comes out as here, and why the plan's literal is wrong
 *
 * `apps/eval/fixtures/semantic-ir.json` is the only IR in this repository a
 * tier-2 arm can run, and it carries `latencyBudgetMs: 120_000`. The page's
 * `DEFAULT_TIER2_CALL_BUDGET_MS` is 60,000 and the smoke corpus selects at most
 * 3 segments per message under that IR on the default `compiled` family --
 * MEASURED here, and it is 3 rather than 2 because that family runs tier 0,
 * whose entropy rule re-admits the corpus's one code fence -- so (b) is
 * 6 x 60,020 = 360,120 and (a) is 120,020: the MESSAGE budget binds, and the
 * bound is 121,020 with the default allowance. (A tier-2-only arm selects at
 * most 2, so its (b) is 240,080 and the same bound binds.) The plan's `itemTimeoutMs: 120_000` is BELOW that -- it is
 * the message budget itself, unrounded -- so an item that legitimately spends
 * its whole budget races the driver's own deadline. `assertItemTimeoutMs`
 * refuses it rather than letting the race decide.
 */
export function itemDeadlineBound(input: ItemDeadlineInput): ItemDeadlineBound {
  const { latencyBudgetMs, callBudgetMs, maxCallsPerItem, lowerTierAllowanceMs } = input;
  // Each of these is a summand of a ceiling that reaches `setTimeout` in
  // `run.ts`, so all five dangerous spellings have to be refused: `Infinity` --
  // the natural spelling of "no budget" -- and NaN, negative, 0 and anything
  // above 2^31-1 each produce a ~1 ms timeout rather than no timeout. The
  // previous check here was `Number.isFinite(value) && value >= 0`, which
  // caught two of the five and accepted the other three while a comment above
  // it named all five. `cancel.ts` refuses the same set with
  // `Number.isFinite(x) && x > 0 && x <= MAX_BUDGET_MS`; see
  // MAX_TIMER_DELAY_MS above for why that bound is restated here rather than
  // imported, and for what each summand does downstream on its own.
  //
  // The DERIVED ceiling is the one that matters most, because nothing else
  // guards it: an `ir.latencyBudgetMs` above 2^31-1 is reachable (core's schema
  // is `z.number().int().positive()` with no upper bound) and core itself
  // clamps it, but the item ceiling built from it here does not -- it would
  // fire in ~1 ms on every item, turning the whole arm into errored rows and
  // stamping every later row `abandonedWorkInFlight`, which is the exact
  // failure this ceiling exists to avoid. `assertItemTimeoutMs` bounds the
  // result for that reason; these three bound the inputs so the refusal names
  // the number the caller got wrong.
  //
  // The two BUDGETS must be positive and the ALLOWANCE may be 0, and the split
  // is deliberate rather than sloppy. A 0 latency or per-call budget is a
  // deadline that fires immediately -- `callBudgetMs: 0` used to pass planning
  // here and be refused only by `WebLlmJudge`'s constructor inside the browser,
  // i.e. after the model load. A 0 lower-tier allowance is a caller saying
  // "give the deterministic tiers no headroom", which is a choice and reaches
  // no timer of its own.
  for (const [what, value, floor] of [
    ["latencyBudgetMs", latencyBudgetMs, 1],
    ["callBudgetMs", callBudgetMs, 1],
    ["lowerTierAllowanceMs", lowerTierAllowanceMs, 0],
  ] as const) {
    if (!(Number.isFinite(value) && value >= floor && value <= MAX_TIMER_DELAY_MS)) {
      throw new Error(
        `${what} must be a finite number of milliseconds in [${floor}, ${MAX_TIMER_DELAY_MS}], got ` +
          `${String(value)} (${typeof value}); every spelling outside that range -- Infinity, ` +
          `NaN, 0, negative, or above 2^31-1 -- becomes a ~1 ms deadline rather than no deadline`,
      );
    }
  }
  if (!(Number.isInteger(maxCallsPerItem) && maxCallsPerItem >= 0)) {
    throw new Error(`maxCallsPerItem must be a non-negative integer, got ${String(maxCallsPerItem)}`);
  }
  const messageBudgetBoundMs = latencyBudgetMs + INTERRUPT_DRAIN_OVERSHOOT_MS;
  const callBudgetBoundMs = maxCallsPerItem * (callBudgetMs + INTERRUPT_DRAIN_OVERSHOOT_MS);
  const binding = Math.min(messageBudgetBoundMs, callBudgetBoundMs);
  return {
    messageBudgetBoundMs,
    callBudgetBoundMs,
    bindingBound: callBudgetBoundMs < messageBudgetBoundMs ? "call-budget" : "message-budget",
    boundMs: lowerTierAllowanceMs + binding,
    input,
  };
}

/**
 * The bound, doubled.
 *
 * The doubling is the WEDGE MARGIN and nothing more principled than that: this
 * ceiling exists to notice a `page.evaluate` that will never return, and the
 * two inputs it is built from are themselves a per-machine measurement (the
 * drain) and a configuration choice (the budgets). A ceiling sitting exactly on
 * the bound would fire on any item that used its whole budget, which is the
 * normal case for tier 2 on this corpus, not the pathological one.
 */
export function deriveItemTimeoutMs(input: ItemDeadlineInput): number {
  return 2 * itemDeadlineBound(input).boundMs;
}

/** Refuses a ceiling that cannot sit above the longest legitimate item. */
export function assertItemTimeoutMs(itemTimeoutMs: number, bound: ItemDeadlineBound): void {
  if (!(Number.isFinite(itemTimeoutMs) && itemTimeoutMs > 0 && itemTimeoutMs <= MAX_TIMER_DELAY_MS)) {
    throw new Error(
      `itemTimeoutMs must be a finite number of milliseconds in (0, ${MAX_TIMER_DELAY_MS}], got ` +
        `${String(itemTimeoutMs)} (${typeof itemTimeoutMs}); this number reaches setTimeout in ` +
        `run.ts, and one above 2^31-1 fires in ~1 ms rather than never -- which errors every ` +
        `item of the arm instead of catching a wedge`,
    );
  }
  if (itemTimeoutMs <= bound.boundMs) {
    throw new Error(
      `itemTimeoutMs ${itemTimeoutMs} is not above the ${bound.boundMs}ms a legitimate item can ` +
        `take on this bake-off: the ${bound.bindingBound} bound is ` +
        `${bound.bindingBound === "message-budget" ? bound.messageBudgetBoundMs : bound.callBudgetBoundMs}ms ` +
        `(ir.latencyBudgetMs ${bound.input.latencyBudgetMs}, per-call budget ` +
        `${bound.input.callBudgetMs} x ${bound.input.maxCallsPerItem} calls) plus ` +
        `${bound.input.lowerTierAllowanceMs}ms for the tiers below. A ceiling at or under that ` +
        `fires on items that were merely slow: the row becomes an error AND every later row of ` +
        `the arm is stamped abandonedWorkInFlight, which invalidates the arm's latencies. ` +
        `deriveItemTimeoutMs() returns ${2 * bound.boundMs}.`,
    );
  }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

export interface BakeoffOptions {
  readonly runId: string;
  readonly outDir: string;
  /** Path to a JSONL corpus; read once and shared by every arm. */
  readonly corpus: string;
  readonly provider: string;
  /**
   * Tier-2 model ids. Reordered into `TIER2_MODELS` order -- cheapest first by
   * VRAM, the only per-model size the library reports offline -- so the
   * expensive arms are paid for last whatever order a caller listed.
   */
  readonly models: readonly string[];
  /** Defaults to `["compiled"]`, which is the plan's four-model slate. */
  readonly families?: readonly ArmFamily[];
  /** The page IR registry name, and the file it must match. See `runBakeoff`. */
  readonly irName?: string;
  readonly irPath?: string;
  readonly contextWindowSize?: number;
  readonly callBudgetMs?: number;
  /**
   * The per-item ceiling, REQUIRED rather than derived, exactly as `ArmSpec`
   * requires it -- and then checked against `itemDeadlineBound`. A caller that
   * has not thought about the number should be made to, and one that has
   * thought about it wrongly should be told.
   */
  readonly itemTimeoutMs: number;
  /** Path to the policy DOCUMENT the baseline arms would be shown. */
  readonly policyPath?: string;
  readonly lowerTierAllowanceMs?: number;
  readonly uncertainBelow?: number;
}

export interface PlannedArm {
  readonly arm: string;
  readonly family: ArmFamily;
  readonly modelId: string;
  readonly path: string;
  /** The exact object `detect` will receive, minus the backend runArm reconciles. */
  readonly config: TierConfig;
  readonly contextWindowSize: number;
  readonly callBudgetMs: number;
  /**
   * The escalation this arm's own tier-0 setting produces, MEASURED over the
   * corpus with the IR the arm runs.
   *
   * Per arm and never shared, which is what makes the two segments-per-message
   * conditions impossible to splice here. Task 9's figures have two of them --
   * with tier-0 priors (p50 1, p95 3, max 3) and without (p50 1, p95 2, max 2)
   * -- and a previous version of the plan spliced them into one distribution
   * with p95 2 and max 3, which no single sample can have: at n = 13 the
   * nearest-rank p95 IS the maximum. This field carries one sample per arm, so
   * its p95 and its max come from the same 13 numbers.
   *
   * WHICH condition an arm lands in is a property of ITS IR as well as its tier
   * flags, and that is worth stating because it was wrong for one commit.
   * Task 9's priors condition was measured with the findings `minimal-ir.json`'s
   * `entropy-rule` produces on this corpus's code fence at confidence 0.7, which
   * re-admit that fence to escalation. `semantic-ir.json` -- the only IR a
   * tier-2 arm can run, since `planBakeoff` throws on any IR with no
   * `semanticPredicates` -- shipped `rules: []` for one commit, so `runTier0`
   * found nothing, no segment was uncertain, and a tier-0 arm's distribution was
   * IDENTICAL to a tier-2-only arm's. It now carries the same three rules, and
   * MEASURED HERE against it the two conditions are what Task 9 recorded: with
   * priors 18 segments at p50 1, p95 3, max 3 per message; without, 17 at
   * p50 1, p95 2, max 2. `bakeoff.test.ts` asserts both, and asserts the rules
   * are still there, so stripping them fails loudly rather than in a results
   * table where two arms agree for a reason that is not about tier 0.
   */
  readonly segments: SegmentSizeDistribution;
  readonly maxCallsPerItem: number;
  readonly bound: ItemDeadlineBound;
}

export interface BakeoffPlan {
  readonly arms: readonly PlannedArm[];
  /** ONE ceiling for every arm, sized from the family that needs the most. */
  readonly itemTimeoutMs: number;
  readonly bound: ItemDeadlineBound;
  readonly gatesPath: string;
  readonly ir: {
    readonly policyHash: string;
    readonly latencyBudgetMs: number;
    readonly semanticPredicates: number;
    readonly entityTypes: number;
    readonly rules: number;
  };
}

export interface PlanInput {
  readonly options: BakeoffOptions;
  /** The IR the PAGE will run, already parsed. `runBakeoff` verifies that claim. */
  readonly ir: PolicyIr;
  readonly items: readonly CorpusItem[];
  /** The policy document, verbatim. Required exactly when a baseline family is asked for. */
  readonly policyText?: string;
}

const DEFAULT_FAMILIES: readonly ArmFamily[] = ["compiled"];

function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/**
 * Everything that can be decided before a model loads, decided.
 *
 * A four-model bake-off is hours of GPU time and 7.49 GB of weights. That is
 * `test/tier2-profile.ts`'s measurement rather than one taken here -- it loaded
 * all four pinned arms at this origin and read `navigator.storage.estimate()`,
 * 1.08 GB for Qwen3.5-2B and 2 to 2.3 GB for each of the other three -- and it
 * replaces the "roughly 12 GB" this comment carried, which was nobody's
 * measurement. Learning on the last arm
 * that its file name collides, that its policy is not the one its IR came from,
 * or that the per-item ceiling was set below the message budget wastes all of
 * it. Nothing in here touches the page or the disk.
 */
export function planBakeoff(input: PlanInput): BakeoffPlan {
  const { options, ir, items } = input;
  const families = options.families ?? DEFAULT_FAMILIES;
  const uncertainBelow = options.uncertainBelow ?? UNCERTAIN_BELOW;
  const lowerTierAllowanceMs = options.lowerTierAllowanceMs ?? DEFAULT_LOWER_TIER_ALLOWANCE_MS;

  assertFileSafe("runId", options.runId);
  if (items.length === 0) {
    throw new Error(
      `corpus ${options.corpus} holds no items; every arm would write an empty file, which is ` +
        `indistinguishable from a run that scored nothing`,
    );
  }
  if (options.models.length === 0) throw new Error("a bake-off with no models measures nothing");
  if (families.length === 0) throw new Error("a bake-off with no arm families measures nothing");

  // The one IR property that decides whether ANY of this spends a model call.
  // `WebLlmJudge.judge` returns `{findings: [], scopesJudged: []}` before it
  // touches the engine when the IR declares no `semanticPredicates`, so a
  // bake-off run against such an IR produces four complete, schema-valid files
  // of zero model calls -- and every gate below comes back "not measured" while
  // the run looks like it happened.
  if (ir.semanticPredicates.length === 0) {
    throw new Error(
      `the IR this bake-off would run declares no semanticPredicates, so WebLlmJudge returns ` +
        `an empty verdict before it touches the engine and every arm would write a complete, ` +
        `schema-valid file of zero model calls. Point --ir at a policy with a semantic clause ` +
        `(apps/eval/fixtures/semantic-ir.json is the only one here that has one).`,
    );
  }

  // Resolved through the package's own validator, so an unknown model, a
  // non-zero temperature or a non-integer window throws here naming what was
  // available rather than 400 seconds later inside the page.
  const resolvedModels = options.models.map((modelId) =>
    resolveTier2Config({ modelId, contextWindowSize: options.contextWindowSize }),
  );
  const callBudgetMs = options.callBudgetMs ?? DEFAULT_TIER2_CALL_BUDGET_MS;

  // Cheapest-first by the manifest's order, whatever order the caller listed --
  // the bake-off should pay for the expensive arms last. No -1 can reach the
  // subtraction: `resolveTier2Config` above refuses an id that is not in
  // `TIER2_MODELS`, so every `findIndex` here hits.
  const ordered = [...resolvedModels].sort(
    (a, b) =>
      TIER2_MODELS.findIndex((m) => m.id === a.modelId) -
      TIER2_MODELS.findIndex((m) => m.id === b.modelId),
  );

  const needsPolicy = families.some((f) => !familyShape(f).runsCompiledJudge);
  const policyText = input.policyText;
  if (needsPolicy) {
    if (policyText === undefined || policyText === "") {
      throw new Error(
        `families [${families.join(", ")}] include an Approach-B arm, which is the WHOLE policy ` +
          `document in one prompt with no compiler. There is no such document here: pass ` +
          `policyPath, pointing at the document the IR was compiled from.`,
      );
    }
    // `PolicyIr.policyHash` is the compiler's sha256 of the SOURCE DOCUMENT --
    // `policyHash(document)` in packages/compiler/src/stages/emit.ts, verified
    // by reading it. So this equality is checkable, exact, and the only thing
    // standing between a head-to-head and a comparison of two different
    // policies. B's entire premise is being shown the document the compiled
    // arm's IR came from; shown a different one it loses (or wins) for a reason
    // invisible in every number the run produces.
    const documentHash = sha256(policyText);
    if (documentHash !== ir.policyHash) {
      throw new Error(
        `the policy document is not the one this IR was compiled from: its sha256 is ` +
          `${documentHash} and the IR's policyHash is "${ir.policyHash}". Approach B is shown ` +
          `the document and the compiled arm is shown the IR, so running them against different ` +
          `policies compares two policies rather than two methods -- and every record would look ` +
          `correct. If the IR's policyHash is not a sha256 at all, it is a hand-written fixture ` +
          `and no document in this repository can satisfy it; compile a policy first.`,
      );
    }
  }

  // Per FAMILY and never shared, so the two escalation conditions cannot be
  // spliced: a family that runs tier 0 gets priors from the IR it runs, and one
  // that does not gets none. Tier 1 is off on every bake-off arm, so tier 0's
  // findings are the whole of what `detect` would hand `uncertainSegmentStarts`.
  const segmentsByFamily = new Map<ArmFamily, SegmentSizeDistribution>();
  for (const family of families) {
    if (segmentsByFamily.has(family)) continue;
    const shape = familyShape(family);
    segmentsByFamily.set(
      family,
      segmentSizeDistribution(items, {
        hasPredicates: ir.semanticPredicates.length > 0,
        uncertainBelow,
        // The key is OMITTED rather than set to `() => []` for a family that
        // does not run tier 0, so `escalation.hasPriors` records which of the
        // two conditions this distribution is. A supplied callback that returns
        // nothing and no callback at all produce the same numbers, and only one
        // of them is an arm that ran tier 0 -- which is the distinction
        // `gateReport` checks a report's distribution against its family with.
        ...(shape.runsTier0 ? { priorFindings: (item: CorpusItem) => tier0Priors(ir, item) } : {}),
      }),
    );
  }

  const arms: PlannedArm[] = [];
  const owners = new Map<string, string>();
  for (const model of ordered) {
    for (const family of families) {
      const shape = familyShape(family);
      const segments = segmentsByFamily.get(family)!;
      const arm = armName(family, model.modelId);
      assertFileSafe("arm name", arm);
      const path = join(options.outDir, `${options.runId}.${arm}.jsonl`);
      const owner = owners.get(path);
      if (owner !== undefined) {
        throw new Error(
          `arms "${owner}" and "${arm}" would both write ${basename(path)}: a file is named by ` +
            `runId, family and model, so two arms agreeing on those leave one silently ` +
            `overwritten and the bake-off reporting more arms than it produced`,
        );
      }
      owners.set(path, arm);
      // One call per judged unit plus the single repair retry. On the compiled
      // family the unit count is the MAXIMUM of this arm's own per-item sample,
      // never the p95: at n = 13 those are the same rank anyway, and a ceiling
      // built on a percentile would be a ceiling the worst message exceeds by
      // construction.
      const maxUnits = shape.judgedUnit === "message" ? 1 : (segments.perItem?.max ?? 0);
      const maxCallsPerItem = shape.callsPerJudgedUnit * maxUnits;
      if (maxCallsPerItem === 0) {
        throw new Error(
          `arm "${arm}" would make no engine call on any of the ${items.length} corpus items: ` +
            `escalation selected no segment anywhere (${segments.segmentsTotal} segment(s) in ` +
            `total). Its file would be a complete, schema-valid transcript of a model that never ran.`,
        );
      }
      arms.push({
        arm,
        family,
        modelId: model.modelId,
        path,
        // Built from the family rather than taken from the caller, so an arm
        // cannot be recorded as something it is not. `baselineB.ts` refuses a
        // TierConfig that disagrees with the arm it was built as, for the same
        // reason and with the same argument.
        config: {
          tier0: shape.runsTier0,
          tier1: false,
          tier2: true,
          t2Model: model.modelId,
          uncertainBelow,
        },
        contextWindowSize: model.contextWindowSize,
        callBudgetMs,
        segments,
        maxCallsPerItem,
        bound: itemDeadlineBound({
          latencyBudgetMs: ir.latencyBudgetMs,
          callBudgetMs,
          maxCallsPerItem,
          lowerTierAllowanceMs,
        }),
      });
    }
  }

  // ONE ceiling for every arm, and it is the LARGEST of the per-arm bounds.
  // Same timeout is a fairness requirement -- a bake-off in which one arm had
  // longer to answer than another is measuring the deadline -- and taking the
  // largest is what stops the family that makes more calls from being squeezed
  // by a number sized for the family that makes fewer.
  const bound = arms.reduce((a, b) => (b.bound.boundMs > a.boundMs ? b.bound : a), arms[0]!.bound);
  assertItemTimeoutMs(options.itemTimeoutMs, bound);

  return {
    arms,
    itemTimeoutMs: options.itemTimeoutMs,
    bound,
    gatesPath: join(options.outDir, `${options.runId}.gates.jsonl`),
    ir: {
      policyHash: ir.policyHash,
      latencyBudgetMs: ir.latencyBudgetMs,
      semanticPredicates: ir.semanticPredicates.length,
      entityTypes: ir.entityTypes.length,
      rules: ir.rules.length,
    },
  };
}

/**
 * What tier 0 finds on one item, which is what decides that item's uncertain
 * segments.
 *
 * Core's own `runTier0` over core's own segmentation -- never a reimplementation
 * -- for the reason `segments.ts` gives about the escalation policy: a second
 * copy of half the rule is free to drift from the one `detect` runs, and the
 * drift silently sizes the ceiling for work the arm never does.
 */
function tier0Priors(ir: PolicyIr, item: CorpusItem): readonly Finding[] {
  return runTier0(ir, item.text, segmentText(item.text));
}

/**
 * The page's default per-CALL budget, duplicated here as a DEFAULT ONLY.
 *
 * It is declared in `apps/eval/src/page/main.ts` and not exported -- the page is
 * a Vite entry point, not a module this driver imports -- so the number cannot
 * be shared by reference. Two things keep the two copies honest, and neither is
 * this comment:
 *
 *   - `runBakeoff` passes this value TO `loadTier2` explicitly and then records
 *     the value the page reports back, refusing an arm whose page resolved a
 *     different one. So a drift is a refusal, not a record naming a budget the
 *     arm did not run under.
 *   - `bakeoff.test.ts` reads the page module's own literal out of its SOURCE
 *     and compares it with this one, because the refusal above only fires on a
 *     machine with a GPU and this file is planned on every machine. Exported
 *     for that test alone: nothing else here imports it.
 */
export const DEFAULT_TIER2_CALL_BUDGET_MS = 60_000;

/**
 * The message budget the COMPILER emits when a policy does not name one, kept
 * here only so a gates row can say how far from it the run was.
 *
 * It is `DEFAULT_LATENCY_BUDGET_MS` in packages/compiler/src/stages/emit.ts and
 * it is copied rather than imported, for the same reason the page's per-call
 * budget above is: `@sih/compiler` is not a dependency of `@sih/eval` and should
 * not become one -- it is Node-only, reads files and talks to an SDK, while this
 * driver's sibling half is a browser bundle. `bakeoff.test.ts` reads the
 * compiler's own literal out of its SOURCE and compares it with this one, which
 * is what keeps the copy honest; this comment is not.
 *
 * WHY A GATES ROW NEEDS IT. Every latency this bake-off can produce is taken
 * under `semantic-ir.json`'s `latencyBudgetMs` of 120,000 -- 24x this number --
 * and that budget is not a detail of the fixture, it is the deadline the
 * orchestrator arms over the WHOLE `judge()` call. So under a compiled policy's
 * default the same arm would judge fewer segments per message and file more
 * `budget-exhausted` notices, and this run measures none of that. See the module
 * header for the arithmetic (Plan 5's measured 4.6 s per call on the cheapest
 * arm, against a 5,000 ms message budget) and `ArmRunContext.latencyBudgetMs`
 * for what it would take to measure the degradation instead of caveating it.
 */
export const COMPILER_DEFAULT_LATENCY_BUDGET_MS = 5_000;

// ---------------------------------------------------------------------------
// Gates, computed
// ---------------------------------------------------------------------------

export type GateName =
  | "p95-ttft"
  | "decode-rate"
  | "resolvable-rate"
  | "duplicate-rate"
  | "non-empty-after-stop";

export interface GateOutcome {
  readonly gate: GateName;
  /** The number from `GATES`, or `undefined` for the one gate that is a predicate. */
  readonly threshold: number | undefined;
  readonly observed: number | undefined;
  /** How many observations `observed` was computed from. */
  readonly sample: number;
  /**
   * `not-measured` is a THIRD answer and not a quiet pass.
   *
   * An arm whose every message ran out of budget before its first call has no
   * latency to take a percentile of and no finding to score a ladder on.
   * Calling that a failure kills it for the budget overrun by the back door,
   * which is the one rule this module must not have; calling it a pass hides
   * that the arm was never measured. It is neither, and it does not set
   * `killedOnRunGates`.
   */
  readonly verdict: "pass" | "fail" | "not-measured";
  readonly detail: string;
}

/**
 * The settings an arm's numbers were produced under, so a gates file can be
 * joined to the run that made it.
 *
 * WHY THIS EXISTS. Without it neither output file records its own
 * configuration: `ArmGateReport` had no runId, no IR, no corpus and no budgets,
 * and `RunRecordSchema` carried `config.t2Model` and nothing else about tier 2.
 * Two runs at different context windows or per-call budgets were byte-identical
 * in every field a scorer can group by, the gates file could not be joined to
 * the IR it ran against at all, and Plan 5's own fallback for a model that
 * cannot take 8,192 -- "run that arm at 4,096 and report the asymmetry" -- was
 * unexpressible in the output. That is the same disease `tier1Config` was added
 * to `RunRecordSchema` to cure, one tier up.
 *
 * Every field here is READ BACK OFF THE RECORDS the gate was computed from,
 * except the three that no record carries (`corpus`, `itemTimeoutMs`,
 * `latencyBudgetMs`), which the caller supplies. That is deliberate: a report
 * that copied the PLAN's intentions would state what the bake-off meant to run,
 * and this file's whole subject is what it did run. `gateReport` refuses a set
 * of records that disagree with each other on any of them.
 */
export interface ArmRunContext {
  readonly runId: string;
  /** The corpus file's basename; the path is machine-specific and not evidence. */
  readonly corpus: string;
  /** sha256 of the IR artifact the PAGE loaded, as every record carries it. */
  readonly irHash: string;
  /** The IR's own `policyHash` field: the compiler's hash of the source document. */
  readonly policyHash: string;
  readonly recordSchemaVersion: number;
  /** The per-item wall-clock ceiling `runArm` was given. */
  readonly itemTimeoutMs: number;
  /**
   * `ir.latencyBudgetMs`: the message deadline the orchestrator armed, and the
   * number every latency on this report has to be read against.
   *
   * THE CAVEAT, and it applies to every arm this driver can run. `planBakeoff`
   * refuses any IR with no `semanticPredicates` and names `semantic-ir.json` as
   * the only one in this repository that has one, and that fixture carries
   * 120,000 -- 24x `COMPILER_DEFAULT_LATENCY_BUDGET_MS`, which is what the
   * compiler emits for a policy that does not name a budget and what
   * `minimal-ir.json` and `multiclass-ir.json` carry. So a bake-off run here is
   * run at a budget no compiled policy would ship with, and it is not a
   * mistake in the fixture: at 5,000 the deadline fires during the first call
   * of every tier-2 spec, so nothing would exercise a completed judgement (see
   * the module header, and Plan 5's measured 4.6 s per call on the CHEAPEST
   * arm against a 5,000 ms whole-message budget).
   *
   * What that costs, stated rather than hidden: the DEGRADATION a shipped
   * policy's budget would cause is not measured anywhere in this run. The
   * per-call numbers (`ttftMs`, `decodeTokPerSec`) are per-call and a shorter
   * message budget does not slow a call, but it changes WHICH calls happen --
   * the orchestrator's one deadline cuts the message off mid-judgement, so a
   * run at 5,000 would have a smaller sample, more calls ended by an interrupt,
   * far fewer `ladder.segmentsJudged` and far more
   * `degradedNotices["budget-exhausted"]`. None of those differences is
   * observable from this file.
   *
   * WHAT IT WOULD TAKE to measure it instead, since it is cheap and this
   * comment should not be the end of it. One second fixture -- `semantic-ir.json`
   * with `latencyBudgetMs` at 5,000 and nothing else changed -- plus one line in
   * `IR_FIXTURES` in `apps/eval/src/page/main.ts`, and then a second
   * `runBakeoff` under a different `runId` and `irName`, with an `itemTimeoutMs`
   * matching the much smaller bound `planBakeoff` derives at that budget. The
   * degradation rate is then the ratio of `ladder.segmentsJudged` and of
   * `degradedNotices["budget-exhausted"]` between the two gates rows. It is
   * deliberately NOT a per-arm dimension of one run: `armName` is
   * `<familySlug>-<modelId>` and a file is named by runId, family and model, so
   * two budgets inside one run would collide on a file name -- `planBakeoff`
   * throws on that, so it fails loudly, but making it work means putting the
   * budget in the arm's name and in `PlannedArm`, which is a bigger change than
   * the measurement is worth until someone wants it.
   */
  readonly latencyBudgetMs: number;
  /** `COMPILER_DEFAULT_LATENCY_BUDGET_MS`, so the row is self-contained. */
  readonly compilerDefaultLatencyBudgetMs: number;
  /**
   * `latencyBudgetMs / compilerDefaultLatencyBudgetMs`, to two decimals.
   *
   * A DERIVED number on the row rather than a division left to the reader,
   * because the thing it is evidence for -- "these latencies do not transfer to
   * a shipped configuration" -- is invisible while the two numbers sit in
   * different files. 24 on every run this driver can currently perform.
   */
  readonly latencyBudgetTimesCompilerDefault: number;
  /**
   * Spec 4.1's escalation threshold, resolved, as every record carries it.
   *
   * `undefined` only on a set of records that carry none, which the schema
   * makes impossible for a tier-2 arm -- and left undefined rather than filled
   * with a placeholder, because `JSON.stringify` drops the key and an absent
   * one is the honest answer where a NaN would serialise as `null` and read as
   * a threshold of zero.
   */
  readonly uncertainBelow: number | undefined;
  /** The engine settings the page reported after resolving them. */
  readonly tier2Config: RunRecord["tier2Config"];
}

/** One tier's answer to "did this arm run it, and is there anything to score it with". */
export interface TierGoldCoverage {
  readonly tier: Tier;
  /**
   * Whether the arm ran this tier, read off `config` on its own ROWS.
   *
   * `config` is the exact object `detect` received, so this is what the arm was
   * asked to run rather than what its family says it runs -- the two can differ
   * and only one of them is a fact about the work. It is NOT an attestation
   * that the tier produced anything: a tier switched on that found nothing is
   * `true` here, and `degradedNotices` is where an absent tier shows up.
   */
  readonly ran: boolean;
  /** Gold spans on those rows whose entityType the IR declares at this tier. */
  readonly goldSpans: number;
  /** The distinct entityType ids behind `goldSpans`, sorted. */
  readonly goldEntityTypes: readonly string[];
}

/**
 * What the verdicts on this report are, and -- the reason the field exists --
 * what they are not.
 *
 * ## The failure this was added for
 *
 * `corpora/fixtures/smoke.jsonl` is the only corpus in this repository. It
 * carries seven gold spans: five at tier 0 and two at tier 1, and NONE at tier
 * 2. Every arm of this bake-off runs tier 2, and three of the corpus's thirteen
 * items say things like "draft a contract renewal email for Tamarind Grocers"
 * and "the Halcyon Logistics account needs a renewal quote" -- which is the
 * shipped predicate ("a customer contract, renewal, or negotiation that has not
 * been publicly announced") almost word for word, on segments escalation
 * selects under both conditions. So an arm that answers the predicate CORRECTLY
 * emits findings that no gold span can match, a scorer joining the two counts
 * every one of them a false positive, and the arm that found nothing scores as
 * the most precise. The ranking is inverted, and every file involved is
 * complete, schema-valid and internally consistent.
 *
 * ## Why this is a field and not a refusal
 *
 * Refusing to run was the alternative and it is the wrong one, for two reasons
 * that are specific to this repository rather than general.
 *
 * First, it would throw away the measurement to prevent a misreading. Every
 * gate on this report is a property of the RUN -- latency, decode rate, whether
 * the span ladder could place a quote, whether an answer restated one already
 * given, whether the engine latched after a stop -- and every one of them is
 * computable without a single gold label. None of them is affected by the gap.
 * A refusal buys a scorer's correctness at the price of the only numbers the
 * bake-off exists to take.
 *
 * Second, and this is the decisive one: a refusal keyed on "the corpus carries
 * no gold for a tier this arm runs" can only ever be satisfied by ADDING GOLD
 * TO THE CORPUS. Labelling this corpus for the semantic predicate is Plan 7's
 * job, and doing it here -- under the pressure of a build error that will not
 * clear until it is done -- is precisely the move this project has already had
 * to revert once: editing the data until the harness is happy. A rule whose
 * only remedy is to edit the data is a machine for producing tuned data. So the
 * gap is REPORTED, in the artifact, under a name that says what it is.
 *
 * The corpus is not touched. `smoke.jsonl` is unchanged by the commit that
 * added this type.
 */
export interface ArmScoringBoundary {
  /**
   * FALSE on every report this module can produce, and it is a field rather
   * than a comment so a reader of the gates file learns it from the file.
   *
   * Spec 4.2 says "task accuracy must be the primary gate". No accuracy gate
   * exists here and none should: per spec 2.2 the JSONL file is the whole
   * boundary and scoring is Plan 8's Python. Every gate on this report is a
   * property of the run.
   */
  readonly accuracyGated: false;
  /** Per tier: whether this arm ran it, and what gold there is to score it with. */
  readonly tiers: readonly TierGoldCoverage[];
  /** The tiers of `tiers` whose `ran` is true, ascending. */
  readonly tiersRun: readonly Tier[];
  /** `goldSpans` by tier, so a reader does not have to walk `tiers`. */
  readonly goldSpansByTier: Readonly<Record<Tier, number>>;
  /**
   * Tiers this arm RAN that its rows carry no gold for.
   *
   * NON-EMPTY MEANS THIS ARM'S FINDINGS AT THOSE TIERS CANNOT BE SCORED against
   * this corpus, in either direction: no recall, because there is nothing to
   * recall, and no precision, because every finding is unmatched. `[2]` on
   * every arm of a bake-off run against the shipped corpus today.
   */
  readonly tiersThisCorpusCannotScore: readonly Tier[];
  /**
   * Gold entityTypes the IR that ran declares no tier for, sorted.
   *
   * A different way the same join comes back empty, and one this corpus is a
   * standing candidate for: its items are labelled `policy: "minimal-fixture"`
   * while the bake-off's only runnable IR is `semantic-ir.json`. It happens
   * that every id in the shipped gold IS declared there, so this is empty
   * today -- but a relabelled corpus or a recompiled policy changes that
   * silently, and a span in no tier bucket would otherwise vanish from the
   * counts above rather than show up as unmatchable.
   */
  readonly goldEntityTypesNotInIr: readonly string[];
  /** The distinct `policy` names the rows' gold was labelled under, sorted. */
  readonly goldPolicies: readonly string[];
  /**
   * One sentence per reason this arm cannot be scored against this corpus.
   *
   * An empty array is the POSITIVE claim that every tier the arm ran has gold
   * behind it, which is why it is an array of reasons rather than a boolean
   * with the reasons in a comment.
   */
  readonly cannotScore: readonly string[];
  /** What `ArmGateReport.killedOnRunGates` means, and what it must not be read as. */
  readonly verdictMeans: string;
}

export interface ArmGateReport {
  readonly arm: string;
  readonly family: ArmFamily;
  readonly modelId: string;
  /** What this arm ran under; see ArmRunContext for why it is on the gates file. */
  readonly run: ArmRunContext;
  /**
   * What this report's verdicts are and are not, and which tiers this corpus
   * can score at all. Read it before reading `killedOnRunGates`.
   */
  readonly scoring: ArmScoringBoundary;
  readonly items: number;
  /** Items whose detection THREW. Their counters are absent, not zero. */
  readonly itemsErrored: number;
  /**
   * Items measured while an earlier item's abandoned work was still running.
   *
   * COUNTED, never excluded: nothing in this function filters a record out. A
   * nonzero here invalidates more than the arm's latencies, and `record.ts`
   * carries the measurements -- the orchestrator arms a wall-clock deadline
   * from `ir.latencyBudgetMs`, so contention truncates the FINDINGS too, and
   * `tier2Stats` is a delta taken around a shared judge, so an abandoned item's
   * calls land inside the next item's window. So `ladder`, `ttftMs`,
   * `promptTokens` and every gate computed from them include rows whose
   * counters may belong to two items. Treat a nonzero here as a reason to
   * distrust the arm's tier-2 aggregates, not just its clock.
   */
  readonly itemsAbandonedWorkInFlight: number;
  /** Engine calls that ANSWERED, across the arm. */
  readonly answeredCalls: number;
  /** Calls in the latency sample: those whose engine reported a finite TTFT. */
  readonly ttftCalls: number;
  readonly ttftMs: SizeStats | undefined;
  /**
   * Prompt size over exactly the calls in the latency sample.
   *
   * The evidence for `GATES.maxP95TtftMs`'s comparability clause. The engine's
   * own `usage.prompt_tokens`, so it is a token count and not the ~1.1 kB
   * CHARACTER figure the threshold was derived at -- see `segmentChars` for a
   * number in that unit. Restricted to the latency sample deliberately: a
   * prompt column taken over a different set of calls cannot be checked against
   * the latency it stands beside.
   */
  readonly promptTokens: SizeStats | undefined;
  /** Completion size over the same calls as `promptTokens`, and for the same reason. */
  readonly completionTokens: SizeStats | undefined;
  /** Per-call decode rates, over the calls that reported a usable one. */
  readonly decodeTokPerSec: SizeStats | undefined;
  /**
   * Tokens decoded divided by seconds spent decoding, over the whole arm.
   *
   * Token-WEIGHTED and not a mean of the per-call rates, because the gate says
   * "sustained": a 5-token call that decoded slowly must not outvote a
   * 500-token call that decoded fast. Both halves come from the engine
   * (`completion_tokens` and `decode_tokens_per_s`), so no harness time is in
   * the denominator.
   */
  readonly sustainedDecodeTokPerSec: number | undefined;
  readonly ladder: {
    readonly rung1: number;
    readonly rung2: number;
    readonly unresolvedQuotes: number;
    readonly duplicatesDropped: number;
    readonly unknownPredicates: number;
    readonly failedClosed: number;
    readonly truncatedResponses: number;
    readonly abortedResponses: number;
    readonly repairAttempts: number;
    readonly segmentsJudged: number;
    readonly segmentsSkipped: number;
  };
  /**
   * Degradation notices by reason word, summed over the arm.
   *
   * REPORTED, never gated on. `budget-exhausted` and `scope-unjudged` are the
   * two that matter most here and are the two no counter substitutes for: the
   * budget-spent-before-start path makes no judge call at all, so nothing is
   * there to count, and `scope-unjudged` is the orchestrator's own fact about a
   * policy's scopes. See the module header for why neither may become a kill
   * rule.
   */
  readonly degradedNotices: Readonly<Record<DegradedReason, number>>;
  /** Items carrying at least one notice of that reason. One item can carry two. */
  readonly degradedItems: Readonly<Record<DegradedReason, number>>;
  /**
   * The character-size distribution of the segments escalation SELECTS for this
   * arm over this corpus.
   *
   * The evidence for `GATES.maxP95TtftMs` in the unit its ~1.1 kB derivation is
   * quoted in -- `promptTokens` is the same calls in the engine's tokens, and
   * neither converts into the other without the model's own tokenizer.
   *
   * PLANNED, not observed, and the name is only as true as that: it is computed
   * in Node by `planBakeoff` before the first model loads, from core's own
   * segmenter and the escalation policy under THIS arm's tier-0 setting -- so
   * it is the set of segments a healthy run of this arm would judge, not a
   * count taken off the rows. No record carries the segments the page actually
   * handed the judge, so this is the closest population the file has. The three
   * ways a real run diverges from it are all on this same report and should be
   * read beside it: `itemsErrored` (an item that threw judged none of its
   * segments), `degradedNotices["budget-exhausted"]` (a message that stopped
   * before its later segments) and `ladder.segmentsJudged` (what the judge
   * actually saw).
   *
   * What is checked rather than trusted: `gateReport` refuses a distribution
   * whose escalation condition or item count disagrees with the rows, so this
   * cannot be the OTHER family's distribution -- the two differ on this corpus
   * -- or one measured over a different corpus.
   */
  readonly segmentChars: SizeStats | undefined;
  readonly segmentsPerItem: SizeStats | undefined;
  readonly escalation: SegmentSizeDistribution["escalation"];
  readonly gates: readonly GateOutcome[];
  /**
   * True when any gate on this report FAILED. `not-measured` never sets it.
   *
   * NAMED FOR ITS SCOPE, and the rename from `killed` is the whole point of the
   * name. `killed` reads as the bake-off's answer to "which model won", and it
   * is not one: the gates it sums are latency, decode rate, span-ladder
   * resolvability, restatement rate and engine poisoning, and not one of them
   * looks at whether the arm was RIGHT. Spec 4.2's stated primary criterion is
   * task accuracy and no accuracy metric exists in this repository -- spec 2.2
   * makes the JSONL file the whole boundary and puts scoring in Plan 8. So an
   * arm can pass every gate here and be the worst model on the slate, and an
   * arm can fail one for being slow while being the only one that answers the
   * predicate. `scoring.verdictMeans` says this on the row itself.
   *
   * `bakeoff.test.ts` pins it as a behaviour rather than a promise: two arms
   * with byte-identical throughput, one whose findings match its gold and one
   * whose identical findings match nothing, get the same value here.
   */
  readonly killedOnRunGates: boolean;
}

export interface GateReportInput {
  readonly arm: string;
  readonly family: ArmFamily;
  readonly modelId: string;
  readonly records: readonly RunRecord[];
  readonly segments: SegmentSizeDistribution;
  /**
   * The three settings no record carries. Everything else on `ArmRunContext` is
   * read back off the records themselves, so these are the only numbers a
   * caller can get wrong here -- and each of them is a number this driver
   * chose rather than one the page reported.
   */
  readonly corpus: string;
  readonly itemTimeoutMs: number;
  readonly latencyBudgetMs: number;
  /**
   * The IR's `entityTypes`, which is the only thing that knows what TIER a gold
   * span belongs to.
   *
   * The fourth thing no record carries, and it is supplied by the caller for
   * the same reason the three above are -- with one honest limitation worth
   * stating rather than leaving to be discovered. Nothing here can check that
   * this array came from the IR the arm ran: a record carries `irHash`, not the
   * entityTypes behind it, so a caller passing another policy's types would
   * produce a well-formed `scoring` block describing the wrong tiers.
   * `runBakeoff` passes the IR it has already proved is the page's, by
   * comparing its own sha256 of the file with the digest the page reported, so
   * on the real path the tie is as strong as `irHash` itself.
   */
  readonly entityTypes: PolicyIr["entityTypes"];
}

/**
 * One value every record in an arm must agree on, or the report describes no
 * single run.
 *
 * The same refusal `gateReport` makes for a foreign `arm`, applied to the
 * settings: a gates row summarising rows written under two different IRs, run
 * ids or context windows would be a confident number describing neither, and
 * every field of it would look well-formed.
 */
function uniform<T>(
  records: readonly RunRecord[],
  what: string,
  read: (r: RunRecord) => T,
): T {
  const first = read(records[0]!);
  const key = (value: T): string => JSON.stringify(value ?? null);
  const odd = records.find((r) => key(read(r)) !== key(first));
  if (odd !== undefined) {
    throw new Error(
      `gateReport was given records that disagree on ${what}: item "${records[0]!.itemId}" says ` +
        `${key(first)} and item "${odd.itemId}" says ${key(read(odd))}; a gate computed over ` +
        `rows from two configurations describes neither`,
    );
  }
  return first;
}

const DEGRADED_REASONS: readonly DegradedReason[] = [
  "absent",
  "budget-exhausted",
  "call-budget-exhausted",
  "failed-closed",
  "scope-unjudged",
];

const zeroReasons = (): Record<DegradedReason, number> =>
  Object.fromEntries(DEGRADED_REASONS.map((r) => [r, 0])) as Record<DegradedReason, number>;

const TIERS: readonly Tier[] = [0, 1, 2];

/**
 * What `killedOnRunGates` means, on the row rather than in a plan nobody ships
 * with the data.
 *
 * A constant string and not a template: nothing about it varies per arm, and a
 * sentence assembled per report is a sentence that can differ between two rows
 * of the same file.
 */
const VERDICT_MEANS =
  "killedOnRunGates is a THROUGHPUT AND HYGIENE verdict and is not a selection between models. " +
  "It is the disjunction of the gates on this report, and every one of them is a property of the " +
  "run: time-to-first-token, decode rate, whether the span ladder could place the quotes the model " +
  "produced, whether an answer restated a span already emitted, and whether the engine latched " +
  "after a deadline expiry. None of them reads a gold label, so none of them can say whether this " +
  "arm was RIGHT. Spec 4.2 makes task accuracy the primary criterion for this slate; no accuracy " +
  "metric is computed anywhere in this repository, because spec 2.2 makes the JSONL file the whole " +
  "boundary and puts scoring in Plan 8. An arm can pass every gate here and be the worst model on " +
  "the slate.";

/**
 * Which tiers this arm ran, and whether its rows carry gold to score them with.
 *
 * Everything here comes off the ROWS except the entityType-to-tier map, which
 * only an IR knows; see `GateReportInput.entityTypes` for what that costs. In
 * particular `ran` is read from `config`, the object `detect` received, and not
 * from `familyShape` -- a report is a statement about work done, and the family
 * is a statement about work planned.
 */
function scoringBoundary(
  records: readonly RunRecord[],
  entityTypes: PolicyIr["entityTypes"],
): ArmScoringBoundary {
  // One tier set for the whole arm, refused if the rows disagree: a report
  // summarising rows that ran different tiers describes no single arm, exactly
  // as `uniform` already argues for the IR and the engine settings.
  const ranTier = uniform(records, "config's tier switches", (r) => ({
    0: r.config.tier0,
    1: r.config.tier1,
    2: r.config.tier2,
  }));

  const tierOf = new Map<string, Tier>(entityTypes.map((e) => [e.id, e.tier]));
  const byTier = new Map<Tier, Set<string>>(TIERS.map((t) => [t, new Set<string>()]));
  const counts: Record<Tier, number> = { 0: 0, 1: 0, 2: 0 };
  const notInIr = new Set<string>();
  const policies = new Set<string>();
  for (const record of records) {
    policies.add(record.policy);
    for (const span of record.gold) {
      const tier = tierOf.get(span.entityType);
      if (tier === undefined) {
        notInIr.add(span.entityType);
        continue;
      }
      counts[tier] += 1;
      byTier.get(tier)!.add(span.entityType);
    }
  }

  const tiers: TierGoldCoverage[] = TIERS.map((tier) => ({
    tier,
    ran: ranTier[tier],
    goldSpans: counts[tier],
    goldEntityTypes: [...byTier.get(tier)!].sort(),
  }));
  const tiersRun = tiers.filter((t) => t.ran).map((t) => t.tier);
  const unscorable = tiers.filter((t) => t.ran && t.goldSpans === 0).map((t) => t.tier);
  const goldEntityTypesNotInIr = [...notInIr].sort();
  const goldPolicies = [...policies].sort();

  // One sentence per problem, each naming the consequence rather than the
  // condition: "no gold at tier 2" is a fact a reader has to translate, and
  // "every correct finding scores as a false positive" is the translation.
  const cannotScore: string[] = [];
  for (const tier of unscorable) {
    const declared = entityTypes.filter((e) => e.tier === tier).map((e) => e.id);
    cannotScore.push(
      `this arm RAN tier ${tier} and the ${records.length} row(s) here carry no gold span at that ` +
        `tier, so a scorer joining findings to gold has nothing for a tier-${tier} finding to ` +
        `match: every one of them counts as a false positive INCLUDING every correct one, and an ` +
        `arm that found nothing at tier ${tier} scores as the most precise. The IR declares ` +
        `${declared.length} entityType(s) at tier ${tier}` +
        (declared.length === 0 ? "" : ` (${declared.join(", ")})`) +
        `; the gold here was labelled under policy ${goldPolicies.join(", ")}. Labelling the ` +
        `corpus is not this driver's job -- do not add gold to make this line go away.`,
    );
  }
  if (goldEntityTypesNotInIr.length > 0) {
    cannotScore.push(
      `gold entityType(s) ${goldEntityTypesNotInIr.join(", ")} are not declared by the IR this arm ` +
        `ran, so they belong to no tier and no finding of any tier can match them; the gold here ` +
        `was labelled under policy ${goldPolicies.join(", ")}, which is a different document than ` +
        `the one that produced this run's IR`,
    );
  }

  return {
    accuracyGated: false,
    tiers,
    tiersRun,
    goldSpansByTier: counts,
    tiersThisCorpusCannotScore: unscorable,
    goldEntityTypesNotInIr,
    goldPolicies,
    cannotScore,
    verdictMeans: VERDICT_MEANS,
  };
}

/**
 * One arm's numbers and one verdict per gate.
 *
 * PURE, and takes records rather than a path, because this is the part of the
 * bake-off whose correctness can be settled without a GPU. Nothing here filters
 * a record out: an errored item is counted as errored and a contaminated one is
 * counted as contaminated, and both stay in the file `runBakeoff` writes.
 */
export function gateReport(input: GateReportInput): ArmGateReport {
  const { arm, family, modelId, records, segments } = input;
  if (records.length === 0) {
    // Not a defensive flourish: every field of `ArmRunContext` is read off the
    // rows, so a zero-row report would have to invent them -- and `runBakeoff`
    // already refuses an arm that measured nothing, one layer up.
    throw new Error(`gateReport was given no records for arm "${arm}"; there is nothing to report`);
  }
  const foreign = records.find((r) => r.arm !== arm);
  if (foreign !== undefined) {
    // A report summing two arms' rows would be a confident number describing no
    // run at all, and every field of it would look well-formed.
    throw new Error(
      `gateReport was given a record from arm "${foreign.arm}" while summarising "${arm}"; ` +
        `a gate computed over two arms' rows describes neither`,
    );
  }
  // Read off the ROWS, never off the plan: the subject of this file is what the
  // arm did, and the plan states what it meant to do. `uniform` refuses a set
  // that disagrees rather than reporting the first row's value for all of them.
  const run: ArmRunContext = {
    runId: uniform(records, "runId", (r) => r.runId),
    corpus: basename(input.corpus),
    irHash: uniform(records, "irHash", (r) => r.irHash),
    policyHash: uniform(records, "policyHash", (r) => r.policyHash),
    recordSchemaVersion: uniform(records, "schemaVersion", (r) => r.schemaVersion),
    itemTimeoutMs: input.itemTimeoutMs,
    latencyBudgetMs: input.latencyBudgetMs,
    compilerDefaultLatencyBudgetMs: COMPILER_DEFAULT_LATENCY_BUDGET_MS,
    // Two decimals rather than an integer: the only budget this driver can
    // currently run divides exactly, and rounding a future 7,500 to 2 would put
    // a wrong number where a reader reads a caveat.
    latencyBudgetTimesCompilerDefault:
      Math.round((input.latencyBudgetMs / COMPILER_DEFAULT_LATENCY_BUDGET_MS) * 100) / 100,
    uncertainBelow: uniform(records, "config.uncertainBelow", (r) => r.config.uncertainBelow),
    tier2Config: uniform(records, "tier2Config", (r) => r.tier2Config),
  };

  // `segmentChars`, `segmentsPerItem` and `escalation` are the only fields on
  // this report that do not come off the rows: they are the PLAN's distribution,
  // measured in Node before the arm ran. That is the closest population there
  // is -- no record carries the segments the page judged -- and it is only
  // honest while the distribution describes THIS arm's work. Two ways it can
  // stop doing so, both of which produce a perfectly well-formed report:
  //
  //   - the other family's condition. On `smoke.jsonl` the tier-0 families
  //     select 18 segments and the tier-2-only families 17, with a per-message
  //     maximum of 3 against 2, so a report carrying the wrong one states a
  //     different arm's escalation under this arm's name. What that costs is a
  //     reader's cross-check rather than the run: `planBakeoff` sized
  //     `maxCallsPerItem` and the item ceiling from the RIGHT distribution
  //     before the arm ran, and this report is where those numbers are checked
  //     against the work -- so a wrong copy here breaks the check without
  //     breaking the run, which is the worse of the two.
  //   - a different `uncertainBelow`. The records carry the resolved threshold
  //     `escalate.ts` compared against and the distribution carries the one it
  //     selected under; they are the same experiment variable, so a
  //     disagreement means the plan and the run escalated differently.
  //
  // NOT checked, and the decision is deliberate rather than an omission: that
  // the distribution was measured over the same CORPUS as the rows. Neither
  // side carries a corpus identity a comparison could use -- `run.corpus` is a
  // basename this caller supplies -- and the obvious proxy, comparing
  // `segments.items` with `records.length`, would be a check on the fixture
  // rather than on the run: `runBakeoff` builds both from one `items` array, so
  // it can only fire for a direct caller, and it would force every gate
  // fixture in the tests to carry one record per corpus item. The coupling is
  // pinned where it is real instead, in `bakeoff.test.ts`'s end-to-end
  // `runBakeoff` case, which asserts each report's distribution IS its own
  // arm's.
  const shape = familyShape(family);
  if (segments.escalation.hasPriors !== shape.runsTier0) {
    throw new Error(
      `arm "${arm}" is family "${family}", which ${shape.runsTier0 ? "runs" : "does not run"} ` +
        `tier 0, but its segment distribution was measured ` +
        `${segments.escalation.hasPriors ? "WITH" : "WITHOUT"} tier-0 priors; segmentChars, ` +
        `segmentsPerItem and escalation would describe the other family's work under this arm's ` +
        `name, and the p95 gate's stated prompt size with them`,
    );
  }
  if (run.uncertainBelow !== undefined && segments.escalation.uncertainBelow !== run.uncertainBelow) {
    throw new Error(
      `arm "${arm}" ran at uncertainBelow ${run.uncertainBelow} and its segment distribution was ` +
        `measured at ${segments.escalation.uncertainBelow}; the two escalated on different ` +
        `thresholds, so the distribution is not this arm's`,
    );
  }

  const ttft: number[] = [];
  const promptTokens: number[] = [];
  const completionTokens: number[] = [];
  const decodeRates: number[] = [];
  let decodedTokens = 0;
  let decodeSeconds = 0;
  let answeredCalls = 0;
  let itemsErrored = 0;
  let itemsAbandoned = 0;
  const ladder = {
    rung1: 0,
    rung2: 0,
    unresolvedQuotes: 0,
    duplicatesDropped: 0,
    unknownPredicates: 0,
    failedClosed: 0,
    truncatedResponses: 0,
    abortedResponses: 0,
    repairAttempts: 0,
    segmentsJudged: 0,
    segmentsSkipped: 0,
  };
  const degradedNotices = zeroReasons();
  const degradedItems = zeroReasons();

  // The engine-poisoning walk, in ARM order. `records` is in corpus order and
  // `calls` is in call order within an item, so this is the order the engine saw.
  let stoppedAt: string | undefined;
  let firstCallAfterStop: string | undefined;
  let poisoned: { itemId: string; finishReason: string | undefined; completionTokens: number | undefined } | undefined;

  for (const record of records) {
    if (record.error !== null) itemsErrored += 1;
    if (record.abandonedWorkInFlight) itemsAbandoned += 1;
    for (const reason of new Set((record.degraded ?? []).map((n) => n.reason))) {
      degradedItems[reason] += 1;
    }
    for (const notice of record.degraded ?? []) degradedNotices[notice.reason] += 1;

    const stats = record.tier2Stats;
    if (stats === undefined) continue;
    ladder.rung1 += stats.rung1;
    ladder.rung2 += stats.rung2;
    ladder.unresolvedQuotes += stats.unresolvedQuotes;
    ladder.duplicatesDropped += stats.duplicatesDropped;
    ladder.unknownPredicates += stats.unknownPredicates;
    ladder.failedClosed += stats.failedClosed;
    ladder.truncatedResponses += stats.truncatedResponses;
    ladder.abortedResponses += stats.abortedResponses;
    ladder.repairAttempts += stats.repairAttempts;
    ladder.segmentsJudged += stats.segmentsJudged;
    ladder.segmentsSkipped += stats.segmentsSkipped;

    for (const call of stats.calls) {
      answeredCalls += 1;
      if (stoppedAt !== undefined && firstCallAfterStop === undefined) {
        firstCallAfterStop = record.itemId;
        // A latched engine answers instantly with an empty body and
        // `finish_reason` "abort". `completionTokens === 0` is the same event
        // seen through the other field; `undefined` there means the engine
        // reported no usage at all, which is a different fact and not evidence
        // of poisoning.
        if (call.finishReason === "abort" || call.completionTokens === 0) {
          poisoned = {
            itemId: record.itemId,
            finishReason: call.finishReason,
            completionTokens: call.completionTokens,
          };
        }
      }
      if (typeof call.ttftMs === "number") {
        ttft.push(call.ttftMs);
        if (call.promptTokens !== undefined) promptTokens.push(call.promptTokens);
        if (call.completionTokens !== undefined) completionTokens.push(call.completionTokens);
      }
      // `decodeTokPerSec` is `completion_tokens / decode_time` with no zero
      // guard, so `null` (a NaN or an infinity, mapped by runArm so the file
      // round-trips) and 0 both mean "this call decoded nothing measurable".
      // Treating either as a rate would drag the weighted figure toward zero on
      // a FLOOR gate.
      if (
        typeof call.decodeTokPerSec === "number" &&
        call.decodeTokPerSec > 0 &&
        call.completionTokens !== undefined &&
        call.completionTokens > 0
      ) {
        decodeRates.push(call.decodeTokPerSec);
        decodedTokens += call.completionTokens;
        decodeSeconds += call.completionTokens / call.decodeTokPerSec;
      }
    }

    // AFTER the calls of this item, because a stop ends the run: any call rows
    // on this item were made BEFORE the stop, not after it. VERIFIED by reading
    // both loops -- `WebLlmJudge.judge` and `baselineB.ts`'s `createArm` return
    // on every `DeadlineExpired`, so no further call is made on the item that
    // stopped.
    if (
      stoppedAt === undefined &&
      stats.deadlineExpiries + stats.callerAbortsMidGeneration + stats.callerAbortsWhileQueued > 0
    ) {
      stoppedAt = record.itemId;
    }
  }

  const sustainedDecodeTokPerSec = decodeSeconds > 0 ? decodedTokens / decodeSeconds : undefined;
  // EVERY quote the ladder was handed, which is the population both of these
  // rates are named over -- and `duplicatesDropped` belongs in it. READ from
  // `WebLlmJudge.#collect`, which runs `resolveQuote` BEFORE the duplicate
  // check: a quote that did not resolve `continue`s at the unresolved counter,
  // a duplicate `continue`s after it, and only the survivors reach `rung1` or
  // `rung2`. So a duplicate is a quote that DID resolve, and leaving it out of
  // the resolvable rate omitted it from both halves.
  //
  // What that cost, by arithmetic on the old expression: an arm on the shape
  // Plan 5 says to expect -- one distinct span, seven restatements of it, two
  // quotes the ladder refused, so 8 of 10 quotes placed -- reported
  // `(1 + 0) / (1 + 0 + 2)`, a rate of 0.333, and was killed. An arm whose span
  // ladder works but whose model restates itself was being reported as an arm
  // whose quotes do not resolve. Those are different diagnoses: a model that
  // restates itself is what `duplicate-rate` is for, and a span-recovery
  // failure is what this gate is for.
  const quotesResolved = ladder.rung1 + ladder.rung2 + ladder.duplicatesDropped;
  const resolvableDenominator = quotesResolved + ladder.unresolvedQuotes;
  const duplicateDenominator = quotesResolved;

  // AUDIT-10, put where a reader of the NUMBER is, not only in a field beside
  // it. Both latency gates carry it because both were taken under the same
  // deadline; see `ArmRunContext.latencyBudgetMs` for what the gap costs and
  // what measuring it instead would take.
  const budgetTaken =
    `taken under a message budget of ${run.latencyBudgetMs}ms, which is ` +
    `${run.latencyBudgetTimesCompilerDefault}x the compiler's ` +
    `${COMPILER_DEFAULT_LATENCY_BUDGET_MS}ms default`;

  const gates: GateOutcome[] = [
    numericGate({
      gate: "p95-ttft",
      threshold: GATES.maxP95TtftMs,
      sample: ttft.length,
      observed: ttft.length === 0 ? undefined : percentile(ttft, 95),
      passes: (observed) => observed <= GATES.maxP95TtftMs,
      notMeasured:
        "no engine call in this arm reported a time-to-first-token, so there is no latency to " +
        "take a percentile of; that is not a slow arm, it is an unmeasured one",
      measured: (observed) =>
        `p95 time-to-first-token over ${ttft.length} answered call(s) was ${observed.toFixed(0)}ms ` +
        `against a ${GATES.maxP95TtftMs}ms ceiling; read it beside promptTokens and segmentChars, ` +
        `because the ceiling was derived at a ~1.1 kB prompt, and beside ` +
        `run.latencyBudgetTimesCompilerDefault, because these calls were ${budgetTaken} -- so the ` +
        `SET of calls in this sample is not the set a compiled policy's budget would produce, and ` +
        `the degradation that difference causes is measured nowhere in this run`,
    }),
    numericGate({
      gate: "decode-rate",
      threshold: GATES.minDecodeTokPerSec,
      sample: decodeRates.length,
      observed: sustainedDecodeTokPerSec,
      passes: (observed) => observed >= GATES.minDecodeTokPerSec,
      notMeasured:
        "no engine call reported a usable decode rate: the library computes it as " +
        "completion_tokens / decode_time with no zero guard, so a call that decoded nothing " +
        "leaves 0/0",
      measured: (observed) =>
        `${decodedTokens} token(s) decoded in ${decodeSeconds.toFixed(2)}s over ` +
        `${decodeRates.length} call(s) is ${observed.toFixed(1)} tok/s against a floor of ` +
        `${GATES.minDecodeTokPerSec}; ${budgetTaken}, so a call cut short by a shipped policy's ` +
        `deadline is not in this sample`,
    }),
    numericGate({
      gate: "resolvable-rate",
      threshold: GATES.minResolvableRate,
      sample: resolvableDenominator,
      observed: resolvableDenominator === 0 ? undefined : quotesResolved / resolvableDenominator,
      passes: (observed) => observed >= GATES.minResolvableRate,
      notMeasured: "this arm produced no quote for the span ladder to place, resolvable or not",
      measured: (observed) =>
        `${quotesResolved} of ${resolvableDenominator} quote(s) resolved to a span ` +
        `(rung 1: ${ladder.rung1}, rung 2: ${ladder.rung2}, dropped as a duplicate of a span ` +
        `already emitted: ${ladder.duplicatesDropped}), a rate of ${observed.toFixed(3)} ` +
        `against a floor of ${GATES.minResolvableRate}`,
    }),
    numericGate({
      gate: "duplicate-rate",
      threshold: GATES.maxDuplicateRate,
      sample: duplicateDenominator,
      observed: duplicateDenominator === 0 ? undefined : ladder.duplicatesDropped / duplicateDenominator,
      passes: (observed) => observed <= GATES.maxDuplicateRate,
      notMeasured: "this arm emitted no finding that resolved to a span, duplicate or otherwise",
      measured: (observed) =>
        `${ladder.duplicatesDropped} of ${duplicateDenominator} resolved finding(s) were a span ` +
        `this run had already emitted, a rate of ${observed.toFixed(3)} against a ceiling of ` +
        `${GATES.maxDuplicateRate}`,
    }),
    stopGate(stoppedAt, firstCallAfterStop, poisoned),
  ];

  return {
    arm,
    family,
    modelId,
    run,
    scoring: scoringBoundary(records, input.entityTypes),
    items: records.length,
    itemsErrored,
    itemsAbandonedWorkInFlight: itemsAbandoned,
    answeredCalls,
    ttftCalls: ttft.length,
    ttftMs: sizeStats(ttft),
    promptTokens: sizeStats(promptTokens),
    completionTokens: sizeStats(completionTokens),
    decodeTokPerSec: sizeStats(decodeRates),
    sustainedDecodeTokPerSec,
    ladder,
    degradedNotices,
    degradedItems,
    segmentChars: segments.chars,
    segmentsPerItem: segments.perItem,
    escalation: segments.escalation,
    gates,
    // FAIL only. A `not-measured` gate must never kill: see `GateOutcome.verdict`.
    killedOnRunGates: gates.some((g) => g.verdict === "fail"),
  };
}

function numericGate(spec: {
  gate: GateName;
  threshold: number;
  sample: number;
  observed: number | undefined;
  passes: (observed: number) => boolean;
  notMeasured: string;
  measured: (observed: number) => string;
}): GateOutcome {
  if (spec.observed === undefined) {
    return {
      gate: spec.gate,
      threshold: spec.threshold,
      observed: undefined,
      sample: spec.sample,
      verdict: "not-measured",
      detail: spec.notMeasured,
    };
  }
  return {
    gate: spec.gate,
    threshold: spec.threshold,
    observed: spec.observed,
    sample: spec.sample,
    verdict: spec.passes(spec.observed) ? "pass" : "fail",
    detail: spec.measured(spec.observed),
  };
}

/**
 * `GATES.assertNonEmptyAfterExpiry`, computed from the file.
 *
 * The evidence is the FIRST call after the FIRST stop and nothing else. Reading
 * the whole tail would report one poisoning as several, and an "abort" late in
 * an arm belongs to whatever stopped immediately before it rather than to the
 * expiry at the top.
 */
function stopGate(
  stoppedAt: string | undefined,
  nextCallItem: string | undefined,
  poisoned: { itemId: string; finishReason: string | undefined; completionTokens: number | undefined } | undefined,
): GateOutcome {
  if (stoppedAt === undefined) {
    return {
      gate: "non-empty-after-stop",
      threshold: undefined,
      observed: undefined,
      sample: 0,
      verdict: "not-measured",
      detail: "no item in this arm was stopped by a budget expiry or a caller abort, so the " +
        "engine was never asked to survive one",
    };
  }
  if (nextCallItem === undefined) {
    return {
      gate: "non-empty-after-stop",
      threshold: undefined,
      observed: undefined,
      sample: 0,
      verdict: "not-measured",
      detail:
        `item "${stoppedAt}" was stopped but no later item made an engine call, so nothing ` +
        `exercised the engine afterwards`,
    };
  }
  if (poisoned !== undefined) {
    return {
      gate: "non-empty-after-stop",
      threshold: undefined,
      observed: undefined,
      sample: 1,
      verdict: "fail",
      detail:
        `the first engine call after the stop on item "${stoppedAt}" came back with ` +
        `finishReason "${String(poisoned.finishReason)}" and ` +
        `${String(poisoned.completionTokens)} completion token(s) on item "${poisoned.itemId}": ` +
        `the engine is latched, so every later call returns instantly and empty and the judge ` +
        `reads that as "no findings" for the rest of the arm`,
    };
  }
  return {
    gate: "non-empty-after-stop",
    threshold: undefined,
    observed: undefined,
    sample: 1,
    verdict: "pass",
    detail:
      `the first engine call after the stop on item "${stoppedAt}" answered normally (on item ` +
      `"${nextCallItem}"), so the interrupt was cleared`,
  };
}

// ---------------------------------------------------------------------------
// Running it
// ---------------------------------------------------------------------------

/**
 * Which of the planned arms this harness can actually execute today.
 *
 * SEPARATE from `planBakeoff` because the two answer different questions.
 * Planning asks whether a slate is a valid experiment; this asks whether the
 * browser half has a door for it. An arm can be a perfectly fair experiment and
 * still have nowhere to run.
 *
 * The Approach-B families have nowhere to run, and the reason is worth stating
 * exactly rather than as "unsupported". `apps/eval/src/page/main.ts` exposes
 * `detect`, which is core's orchestrator with a `WebLlmJudge` behind it;
 * `createBaselineB` is a `Detector` in its own right and the page imports
 * neither it nor a policy document, so there is no `window.__sih` call that
 * reaches it. That door is a small addition -- and it is deliberately NOT made
 * here, because two things in the corpus/policy and record layers would make a
 * B arm meaningless the moment it ran:
 *
 *   1. No IR in this repository was compiled from any policy document in it.
 *      `semantic-ir.json` carries `policyHash: "test-hash"`, which is not a
 *      sha256 of anything, so `planBakeoff` refuses every B arm here already.
 *   2. `RunRecordSchema` has `tier2Stats` and no baseline equivalent. B's
 *      counters are `BaselineStats`, which renames two events (`messagesJudged`
 *      for `segmentsJudged`, `unknownEntityTypes` for `unknownPredicates`) and
 *      adds `messageBudgetExpiries`, which no judge counter reports. Writing
 *      them into `tier2Stats` would be a record stating one event under another
 *      event's name.
 *
 * A THIRD reason stood here and no longer does, which is worth recording rather
 * than deleting: `semantic-ir.json` carried `rules: []`, so tier 0 found nothing
 * on any item and `baseline-b-tier0` versus `baseline-b` -- the pair that
 * separates "compiling helps" from "patterns help" -- would have been two runs
 * of the same thing. The fixture now carries three tier-0 rules and that pair
 * would differ (MEASURED: 18 selected segments against 17, per-message max 3
 * against 2), so the objection is gone and only the two above remain.
 *
 * So this refuses, naming them, rather than running an arm whose numbers would
 * be confident and meaningless.
 */
export function assertPageCanRun(plan: BakeoffPlan): void {
  const baseline = plan.arms.filter((a) => !familyShape(a.family).runsCompiledJudge);
  if (baseline.length === 0) return;
  throw new Error(
    `this harness cannot execute the Approach-B arm(s) [${baseline.map((a) => a.arm).join(", ")}]: ` +
      `apps/eval/src/page/main.ts publishes only core's orchestrator (window.__sih.detect) and ` +
      `never constructs createBaselineB, so no call reaches it. Adding that door is small; what ` +
      `is not is that (1) no IR here was compiled from any policy document here, so B cannot be ` +
      `shown the document the compiled arm's IR came from, and (2) RunRecordSchema has no field ` +
      `for BaselineStats, whose messagesJudged, unknownEntityTypes and messageBudgetExpiries are ` +
      `different events from the judge's. Run the compiled families here and land the baseline ` +
      `families with the compiled policy.`,
  );
}

export interface BakeoffResult {
  /** One path per arm, in the order the arms ran. Killed arms included. */
  readonly written: readonly string[];
  /** One JSON line per arm: the gate verdicts and the numbers behind them. */
  readonly gatesPath: string;
  readonly reports: readonly ArmGateReport[];
  readonly plan: BakeoffPlan;
}

/**
 * Run the slate, cheapest model first, one file per arm plus one gates file.
 *
 * Arms run SEQUENTIALLY, each starting from a freshly navigated page, for the
 * two reasons `runMatrix` gives and one more that is specific to tier 2: two
 * models loading at once contend for one GPU and corrupt every latency in both
 * files; a judge left over from the previous arm would be measured under the
 * next arm's label; and an engine LATCHED by the previous arm's interrupt
 * answers instantly and emptily forever, which the next arm would record as a
 * model that finds nothing. A fresh page costs one cache read per arm and buys
 * all three.
 *
 * It computes NO accuracy metric. Per spec 2.2 the JSONL file is the whole
 * boundary; the gates here are properties of the run, not of a gold label.
 */
export async function runBakeoff(page: Page, options: BakeoffOptions): Promise<BakeoffResult> {
  const items = loadCorpus(readFileSync(options.corpus, "utf8"));
  const irName = options.irName ?? "semantic";
  const irPath =
    options.irPath ?? join(import.meta.dirname, "..", "..", "fixtures", `${irName}-ir.json`);
  const irJson = readFileSync(irPath, "utf8");
  const ir = loadPolicyIr(irJson);

  await openHarness(page);
  // Select the IR and CHECK that the file this process parsed is the artifact
  // the page will run. `irHash()` is the page's sha256 of its own `?raw`
  // fixture text, so this equality ties the two together without either side
  // taking the other's word for it -- and it is the same digest that lands on
  // every record, so a reader can reproduce it with `shasum -a 256` on the file.
  const pageIrHash = await page.evaluate((name) => window.__sih!.useIr(name), irName);
  const fileHash = sha256(irJson);
  if (pageIrHash !== fileHash) {
    throw new Error(
      `the page's IR "${irName}" hashes to ${pageIrHash} but ${irPath} hashes to ${fileHash}; ` +
        `this driver would plan the bake-off against one policy and the page would run another, ` +
        `and every record would carry the page's hash while the ceiling and the escalation came ` +
        `from the file's`,
    );
  }

  const policyText =
    options.policyPath === undefined ? undefined : readFileSync(options.policyPath, "utf8");
  const plan = planBakeoff({ options, ir, items, policyText });
  assertPageCanRun(plan);

  mkdirSync(options.outDir, { recursive: true });
  for (const { path } of plan.arms) {
    // Before the first arm runs, not at write time, so a repeated runId costs no
    // GPU hours. The exclusive flag below is what makes it a guarantee.
    if (existsSync(path)) {
      throw new Error(
        `${path} already exists: a run under this runId has already written this arm, and ` +
          `overwriting it would silently replace one measurement with another`,
      );
    }
  }
  if (existsSync(plan.gatesPath)) {
    throw new Error(`${plan.gatesPath} already exists; use a different runId`);
  }

  const written: string[] = [];
  const reports: ArmGateReport[] = [];
  for (const arm of plan.arms) {
    await openHarness(page);
    const armIrHash = await page.evaluate((name) => window.__sih!.useIr(name), irName);
    if (armIrHash !== fileHash) {
      throw new Error(`arm "${arm.arm}": the page returned a different IR hash after navigating`);
    }
    if (!(await page.evaluate(() => window.__sih!.webgpuAvailable()))) {
      throw new Error(
        `arm "${arm.arm}": WebGPU is not available in this browser, so tier 2 is ABSENT here ` +
          `rather than degraded. Every arm would write a complete file of zero model calls.`,
      );
    }

    const load = await page.evaluate(
      (loadOptions) => window.__sih!.loadTier2(loadOptions),
      {
        modelId: arm.modelId,
        contextWindowSize: arm.contextWindowSize,
        callBudgetMs: arm.callBudgetMs,
      },
    );
    // Three checks on the load report, and they are NOT of equal strength --
    // worth saying because reading them as one block is how an echo gets
    // mistaken for an observation.
    //
    // `servedModelId` is the strong one: `loadTier2` takes it from a real
    // completion, so an engine that compiled its shaders and then died cannot
    // produce it, and it follows a `reload()` behind our back. It is still not
    // an attestation of which WEIGHTS ran -- `engine.ts` traced the field
    // through the bundle to the key of `loadedModelIdToPipeline`, i.e. the id we
    // handed to `CreateMLCEngine`.
    //
    // `contextWindowSize` and `callBudgetMs` are ECHOES: `loadTier2` resolves
    // both from what this call passed and reports them back, so agreement is
    // expected and disagreement means the page CHANGED them -- a clamp, a
    // default applied over an explicit value, a signature drift. That is worth
    // catching (the per-item ceiling was derived from these two numbers, and an
    // arm running under different ones is not bounded by it) and it is all they
    // catch. Neither says what window the engine is enforcing; 0.2.84 exposes no
    // accessor for that, and `probeContextWindow` in the page is the only
    // channel there is.
    if (load.servedModelId !== arm.modelId) {
      throw new Error(
        `arm "${arm.arm}" asked for ${arm.modelId} and the engine answered as ` +
          `"${load.servedModelId}"; every finding would be recorded against a model that did not run`,
      );
    }
    if (load.config.contextWindowSize !== arm.contextWindowSize) {
      throw new Error(
        `arm "${arm.arm}" asked for a ${arm.contextWindowSize}-token window and the page ` +
          `resolved ${load.config.contextWindowSize}; two arms at different windows measure ` +
          `context rather than method`,
      );
    }
    if (load.callBudgetMs !== arm.callBudgetMs) {
      throw new Error(
        `arm "${arm.arm}" asked for a ${arm.callBudgetMs}ms per-call budget and the page ` +
          `resolved ${load.callBudgetMs}; the per-item ceiling was derived from the number this ` +
          `driver asked for, and an arm running under a different one is not bounded by it`,
      );
    }

    const records = await runArm(page, {
      runId: options.runId,
      arm: arm.arm,
      // Tier 2 is WebGPU or nothing: `loadTier2` goes through web-llm, which
      // throws on a null adapter, and `webgpuAvailable()` above asked the
      // adapter's limits rather than guessing from a user agent. So unlike a
      // tier-0 arm's, this label is backed by the load having succeeded.
      backend: "webgpu",
      provider: options.provider,
      config: arm.config,
      // Taken from the PAGE's load report, not from `arm`, and the three checks
      // above are what make that safe: the model id came off a real completion
      // and the window and budget were compared with what this driver asked
      // for. So every row states the settings the engine was actually built
      // with rather than the settings this process intended -- which is the
      // whole difference `Tier2RunConfigSchema` exists for.
      tier2Config: {
        modelId: load.config.modelId,
        contextWindowSize: load.config.contextWindowSize,
        temperature: load.config.temperature,
        maxTokens: load.config.maxTokens,
        callBudgetMs: load.callBudgetMs,
      },
      itemTimeoutMs: plan.itemTimeoutMs,
      items,
    });

    // An arm every one of whose items failed is not a bad result, it is not a
    // result: the file is a complete, schema-valid, perfectly scoreable
    // transcript of nothing having been measured, and Plan 8 would read it as
    // an arm with no recall. This is `runMatrix`'s refusal, for the same reason.
    const failures = records.filter((r) => r.error !== null);
    if (failures.length === records.length) {
      throw new Error(
        `arm "${arm.arm}" failed on all ${records.length} of its items, so it measured nothing; ` +
          `refusing to write a file that would score as an arm with zero recall. The first ` +
          `failure was: ${String(failures[0]?.error)}`,
      );
    }
    for (const [row, record] of records.entries()) {
      const parsed = RunRecordSchema.safeParse(record);
      if (!parsed.success) {
        throw new Error(
          `arm "${arm.arm}" produced an invalid record at row ${row + 1} (item ` +
            `"${record.itemId}"): ${z.prettifyError(parsed.error)}`,
        );
      }
    }

    // The file is written BEFORE the gates are computed, and that ordering is
    // the mechanism rather than a coincidence of layout. THE central rule here
    // is that an arm which fails a gate is a RESULT: dropping it would make the
    // bake-off look like it had fewer contenders than it did, and Plan 8 needs
    // the rows to decide what to do with a killed arm. Nothing at this point
    // knows the verdict, so nothing can act on it -- there is no
    // `killedOnRunGates` in scope to branch on, and the set of arms was fixed by
    // `planBakeoff` before the first model loaded.
    //
    // Stated because it is NOT demonstrated end to end: producing a genuinely
    // killed arm needs hardware that fails a gate, and the cheapest arm on this
    // machine passes both measured ones (p95 TTFT 496 ms against 1,500, decode
    // 50 tok/s against 25). `bakeoff.test.ts` proves `gateReport` marks such an
    // arm killed; that it is still written rests on this ordering.
    try {
      writeFileSync(arm.path, toJsonl(records), { flag: "wx" });
    } catch (cause) {
      if ((cause as NodeJS.ErrnoException).code === "EEXIST") {
        throw new Error(`${arm.path} was created while this bake-off was running`, { cause });
      }
      throw cause;
    }
    written.push(arm.path);
    reports.push(
      gateReport({
        arm: arm.arm,
        family: arm.family,
        modelId: arm.modelId,
        records,
        segments: arm.segments,
        corpus: options.corpus,
        itemTimeoutMs: plan.itemTimeoutMs,
        latencyBudgetMs: plan.ir.latencyBudgetMs,
        // The parsed IR this function already proved is the page's, by
        // comparing its own sha256 of `irPath` with the digest `useIr`
        // returned. So the tier each gold entityType is scored at comes from
        // the same artifact every record's `irHash` names.
        entityTypes: ir.entityTypes,
      }),
    );
  }

  // After every arm, and losing it costs nothing: `gateReport` is pure and takes
  // records, so a run interrupted partway leaves the JSONL files -- which spec
  // 2.2 makes the whole boundary -- and every verdict in this file is
  // recomputable from them. It is written last rather than incrementally for
  // that reason: a half-written gates file beside a complete set of records
  // would be the more confusing artifact.
  writeFileSync(plan.gatesPath, reports.map((r) => JSON.stringify(r)).join("\n") + "\n", {
    flag: "wx",
  });
  return { written, gatesPath: plan.gatesPath, reports, plan };
}

/**
 * Navigate and wait for the page to publish its API, reporting a `pageerror` by
 * message.
 *
 * The same translation `tier2-profile.ts` and `tier1.spec.ts` perform, and for
 * the same failure: the IR fixtures are parsed at module scope, so a malformed
 * one leaves `__sih` unpublished and the wait fails as a bare timeout naming
 * nothing.
 */
async function openHarness(page: Page): Promise<void> {
  const pageErrors: Error[] = [];
  const listener = (error: Error): void => {
    pageErrors.push(error);
  };
  page.on("pageerror", listener);
  try {
    await page.goto("/");
    await page.waitForFunction(() => window.__sih !== undefined, undefined, { timeout: 30_000 });
  } catch (cause) {
    if (pageErrors.length > 0) {
      throw new Error(
        `harness never became ready; the page threw: ${pageErrors.map((e) => e.message).join("; ")}`,
        { cause },
      );
    }
    throw cause;
  } finally {
    page.off("pageerror", listener);
  }
  await assertServedByThisCheckout(page);
}

/**
 * The page this bake-off measures is THIS tree's page.
 *
 * `playwright.config.ts` sets `reuseExistingServer: !process.env["CI"]` and the
 * base URL is a fixed port, so a dev server another worktree left running
 * answers every `page.goto("/")` here. The IR check below covers the IR
 * ARTIFACT and only it -- an identical fixture in a tree with a different
 * `main.ts` or a different `@sih/tier2` passes it -- and no field of a record or
 * a gates row names the code that produced it. The failure is uniform across
 * arms, so it never surfaces as one arm disagreeing with another; it surfaces
 * as a whole bake-off attributed to the wrong tree.
 *
 * `harnessDir()` is a `define` compiled into the page by `vite.config.ts`, so it
 * is the SERVER's directory rather than anything this process could compute for
 * it. What it cannot catch is a second checkout at the same path, which cannot
 * exist on one machine.
 */
async function assertServedByThisCheckout(page: Page): Promise<void> {
  const served = await page.evaluate(() => window.__sih!.harnessDir());
  // `apps/eval`, from `<repo>/apps/eval/src/driver/bakeoff.ts`.
  const here = join(import.meta.dirname, "..", "..");
  if (served !== here) {
    throw new Error(
      `the page this driver navigated to was built by ${served}, but this driver is running from ` +
        `${here}. Playwright reuses an existing dev server outside CI, so a server left running ` +
        `by another worktree would serve every arm of this bake-off and every number would be ` +
        `attributed to the wrong tree. Stop that server (lsof -ti :5178) and re-run.`,
    );
  }
}
