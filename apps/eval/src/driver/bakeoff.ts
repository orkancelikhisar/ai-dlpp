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
  type JudgedUnit,
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
 * `DEFAULT_LATENCY_BUDGET_MS` is 5,000, which is what `minimal-ir.json`,
 * `multiclass-ir.json` and the repository's one COMPILED policy IR,
 * `policies/compiled/p-fin.ir.json`, all carry. (`semantic-ir.json` raises it
 * to 120,000 and its own comment says why: at 5,000 the deadline fires during
 * the first call of every tier-2 spec, so nothing would ever exercise a
 * completed judgement. That is a lifecycle fixture's number, not a claim that
 * tier 2 fits 5 s.) So tier 2 expires
 * mid-run on most messages that escalate more than one segment WHATEVER arm
 * wins. A rule keyed on that would kill all four arms for a reason that is not
 * about capability -- which is exactly why the spec's original gates were
 * replaced. `budget-exhausted` and `scope-unjudged` are COUNTED per arm and
 * reported beside the verdict instead; see `ArmGateReport.degradedNotices`.
 *
 * ## What this module does not compute
 *
 * Any accuracy number. Spec 2.2 makes the TS/Python boundary a JSONL file and
 * puts scoring in `analysis/`'s Python; the spec never mentions plan numbering,
 * and it is this repository's plan sequence, not the spec, that defers writing
 * that Python to Plan 8. Neither the directory nor the plan exists yet. The
 * gates here are all properties of the RUN --
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
 *      row now names the tiers ITS OWN ROWS cannot score, and why, in
 *      `scoring.tiersTheseRowsCannotScore` and `scoring.cannotScore` -- and the
 *      mirror case, gold at a tier this arm did NOT run, in
 *      `scoring.tiersWithGoldThisArmDidNotRun`. The remedy is a named result
 *      rather than a refusal to run, and rather than new gold;
 *      `ArmScoringBoundary` argues both.
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
   * that is `judgedUnitsPerItem`'s business rather than this threshold's.
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
   * measurement reproduces both numbers exactly, one predicate each. The
   * fixture Plan 5 measured is `policies/compiled/p-fin.ir.json`, whose one
   * predicate `pred:client-relationship-disclosure` is 194 characters of id
   * plus `nlPredicate` against `semantic-ir.json`'s 120 -- MEASURED, and 74 is
   * the difference.
   *
   * CORRECTED. This paragraph used to end "so 1,105/1,196 describe work no arm
   * of this bake-off can perform", on the premise that `semantic-ir.json` is
   * the only IR here with a semantic predicate. It is not. `p-fin.ir.json`
   * declares one too, it is registered in the page's `IR_FIXTURES`, and
   * `test/baseline.spec.ts` drives `runBakeoff` against it. So both pairs are
   * live: 1,031/1,122 on a semantic-ir arm and 1,105/1,196 on a p-fin arm, and
   * which pair this gate's ceiling should be read against is decided by
   * `run.irHash` on the row.
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
   * exactly the calls the p95 was taken over, and `ArmGateReport.judgedUnitChars`
   * is the character-size distribution of what the arm's model is shown per
   * call, which is the unit the 1.1 kB above is quoted in (and is PLANNED
   * rather than observed -- see that field). On an Approach-B arm the
   * derivation does not transfer at all: B's prompt carries the whole policy
   * document, so `judgedUnit` on the report is what says the threshold was set
   * for different work. It says the same, more mildly, on a COMPILED arm whose
   * policy declares its predicates message-scoped: the ceiling was derived at a
   * prompt built from ONE SEGMENT and such an arm is shown whole messages,
   * which is what `policies/compiled/p-fin.ir.json` makes both compiled arms of
   * the head-to-head. A reader who finds either one out of line with the
   * derivation knows the gate was applied to different work.
   *
   * ## AT THIS CORPUS SIZE THIS IS A MAXIMUM, NOT A p95
   *
   * `percentile` in `segments.ts` is nearest-rank: `rank = ceil(0.95 * n)`, and
   * COMPUTED over every n, that rank IS n for all n <= 19 -- n = 20, which is
   * `P95_EQUALS_MAX_BELOW` below, is the first sample size at which a p95 can
   * fall under the maximum. So on any sample of 19 calls or fewer this threshold
   * is applied to the arm's SLOWEST call, and one slow first call kills the arm.
   *
   * The samples this driver can currently produce are all under that. MEASURED
   * over `corpora/fixtures/smoke.jsonl` with core's own segmenter and escalation
   * policy: 13 items, 19 segments, of which the tier-0 families select 18 and
   * the tier-2-only families 17 -- so at most 18 engine calls per compiled arm
   * and 13 per Approach-B arm, before the repair retry. Every one of those is a
   * sample size where p95 and max are the same number.
   *
   * This is stated rather than fixed, and both halves are deliberate. It is not
   * fixed by switching to an interpolating percentile, which would invent a
   * number between two observations on a 13-point sample; and it is not fixed by
   * relaxing the gate, because the threshold's derivation (the ~1.1 kB prompt
   * above) is unaffected by how the sample is summarised. What a reader needs is
   * the sample size beside the verdict, and `GateOutcome.sample` carries it on
   * the row: read `p95-ttft` with that number, and below 20 read the observed
   * value as "the slowest call this arm made". The `p95-ttft` gate's own
   * `detail` says so in words on any row where it is true, so a reader who never
   * opens this file is told too.
   *
   * Nor is it fixed by `GateOutcome.minSample`, which this gate carries at 1.
   * That mechanism withholds a verdict where one contrary observation crosses
   * the threshold ALONE and the threshold says it should not -- a floor of 0.8
   * failing on the first bad quote out of two. This ceiling has no such slack
   * to appeal to: 1,500 ms tolerates no call over 1,500 ms, so the maximum is
   * the verdict this gate is written to give and a minimum sample would silence
   * it on every sample this corpus can produce rather than correct it. See
   * `GateOutcome.minSample` for the argument in full.
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
   * (`rung1`, `rung2`, `duplicatesDropped`, `unresolvedQuotes`,
   * `unresolvedMentions`), so a reader who disagrees with the number can
   * recompute the rate.
   *
   * WHAT THE RATE IS OVER CHANGED when the arms started returning two spans per
   * finding, and the threshold did not move with it -- deliberately, because
   * moving a threshold to keep a number looking the same is tuning. The
   * denominator now also carries findings whose clause placed and whose mention
   * did not, so an arm that quotes well and points badly is measured here where
   * before it could not be. Every rate in `runs/slate-p-fin-01.gates.jsonl` was
   * taken under the one-span convention and is not comparable to one taken
   * under this one.
   *
   * THE FIRST REAL SLATE HAS NOW RUN and it did not settle the number, which is
   * worth saying rather than leaving the paragraph above to read as still
   * pending. `runs/slate-p-fin-01.gates.jsonl`, 16 arms over 13 items: ten arms
   * answered a call, eight of them produced a rate of exactly 1.000, two
   * produced 0.500, and the six Approach-B arms on the other three models
   * answered nothing at all. Two values, one of which is the ceiling, are not a
   * distribution to set a floor from, so 0.8 remains a chosen number.
   *
   * What the run DID settle is the sample size, and that is `GateOutcome.minSample`'s
   * business: both 0.500 arms were killed on a sample of TWO quotes, where the
   * only reachable rates are 0, 0.5 and 1 and this floor fails two of the three.
   * The kill was withdrawn by giving the gate a minimum, not by moving this
   * number -- moving it would have been tuning the threshold to a result.
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
   *
   * THE FIRST REAL SLATE PRODUCED NO SUCH RUN. In all 16 arms of
   * `runs/slate-p-fin-01.gates.jsonl`, `duplicatesDropped` is 0 -- not one
   * restatement on any arm of any model -- so every measured rate was 0.000 and
   * the two Plan 5 anecdotes above are still the only evidence there is for
   * this ceiling. Four of the arms recorded that 0.000 as a PASS on a sample of
   * one finding, which is what `GateOutcome.minSample` now withholds: "0 of 1
   * finding was a restatement" describes how much the arm found, not whether it
   * repeats itself.
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
 * The sample size at which a nearest-rank p95 first differs from the maximum.
 *
 * COMPUTED, not chosen: `percentile` in `segments.ts` takes
 * `rank = ceil(percent * n / 100)`, and `ceil(0.95 * n) === n` for every n from
 * 1 to 19, first falling to n - 1 at n = 20. So on any sample smaller than this,
 * "p95" and "max" are the same observation and `GATES.maxP95TtftMs` is a ceiling
 * on the arm's slowest call.
 *
 * It lives here rather than inline because two places need it and they must not
 * disagree: `GATES.maxP95TtftMs`'s docblock, where a reader checks what the
 * threshold means, and the `p95-ttft` gate's `detail`, where a reader meets the
 * number itself.
 */
const P95_EQUALS_MAX_BELOW = 20;

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
    callsPerJudgedUnit: 2,
  },
  "compiled-tier2-only": {
    family: "compiled-tier2-only",
    runsTier0: false,
    runsCompiledJudge: true,
    callsPerJudgedUnit: 2,
  },
  "baseline-b": {
    family: "baseline-b",
    runsTier0: false,
    runsCompiledJudge: false,
    callsPerJudgedUnit: 2,
  },
  "baseline-b-tier0": {
    family: "baseline-b-tier0",
    runsTier0: true,
    runsCompiledJudge: false,
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

/**
 * What ONE engine call covers on an arm: the family AND the policy, together.
 *
 * It cannot be either alone, and this function exists because a field on
 * `FamilyShape` was the other alone. Approach B makes one call per MESSAGE
 * because the whole message is what its model is shown, and that is intrinsic to
 * "simply prompting" -- so B's answer here does not consult the predicates at
 * all, and must not: a B unit that moved with the policy would be the control
 * doing something the method does not do. The compiled judge's unit is NOT
 * intrinsic to the family: `WebLlmJudge` asks each predicate in the scope the
 * policy DECLARED it in, so a segment-scoped clause costs one call per SELECTED
 * segment (escalation decides which) and a message-scoped clause costs one call
 * about the whole message.
 *
 * The three compiled answers, and what each is:
 *
 *   - all predicates segment-scoped -> `"segment"`. `apps/eval/fixtures/semantic-ir.json`.
 *   - all message-scoped            -> `"message"`. `policies/compiled/p-fin.ir.json`,
 *     the only compiled policy in this repository, on which all four arms make
 *     one call per message and the two families do not differ in cost at all
 *     (MEASURED, `test/baseline.spec.ts`, 3 answered calls on each of four arms
 *     over three items).
 *   - both                          -> `"segment+message"`. NO policy here.
 *
 * ## The both-scopes decision, and what it rejected
 *
 * On a policy declaring both, one arm judges segments AND the message and no
 * single one-unit answer is true. Three rules were available: widen the union so
 * a row can say so; keep the distribution segment-based and carry the message
 * call as a separate column; or refuse to plan such an arm. The union was
 * chosen. The second reproduces, one call smaller, the exact defect this
 * function closes -- `judgedUnitChars` would describe some of the arm's prompts
 * while reading as all of them, which is the column a reader consults to check
 * the p95 TTFT gate was applied to comparable work. The third would refuse an
 * arm the planner can already size correctly (`maxCallsPerItem` has counted the
 * message call off the declared scopes since the round before this one), so it
 * would trade a reporting gap for a capability loss.
 *
 * @throws when a compiled family is asked about an IR with no semantic
 *   predicate. `WebLlmJudge.judge` returns an empty verdict before it touches
 *   the engine in that case, so the arm judges no unit of any kind and every
 *   answer here would be invented. `planBakeoff` refuses such an IR outright,
 *   one layer up and with a longer message; this is the same refusal for a
 *   direct caller of `gateReport`.
 */
export function judgedUnitFor(
  family: ArmFamily,
  semanticPredicates: PolicyIr["semanticPredicates"],
): JudgedUnit {
  if (!familyShape(family).runsCompiledJudge) return "message";
  const message = semanticPredicates.some((p) => p.scope === "message");
  const segment = semanticPredicates.some((p) => p.scope === "segment");
  if (message && segment) return "segment+message";
  if (message) return "message";
  if (segment) return "segment";
  throw new Error(
    `family "${family}" runs the compiled judge and the IR declares no semanticPredicates, so ` +
      `WebLlmJudge returns an empty verdict before it touches the engine: the arm judges no ` +
      `unit of any kind and there is none to name`,
  );
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
 * `apps/eval/fixtures/semantic-ir.json` carries `latencyBudgetMs: 120_000`,
 * and the arithmetic below is that IR's. It is NOT the only IR a tier-2 arm can
 * run: `policies/compiled/p-fin.ir.json` declares a semantic predicate too, at
 * the compiler's own 5,000 ms default, and this function is called with
 * whichever one the run loaded -- at 5,000 (a) is 5,020, and that IR's one
 * predicate is message-scoped, so a compiled arm judges ONE unit per item and
 * (b) is 2 x 60,020 = 120,040. The message budget binds far harder either way
 * and the bound is 6,020 with the default allowance. The page's
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
  /**
   * The file holding that IR. Optional, and `resolveIrPath` says what it
   * resolves to when omitted -- a registry name is not a path, and the two are
   * related by a convention that holds for three of the four names the page
   * serves.
   */
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
  /**
   * Path to the policy DOCUMENT the baseline arms are shown. Required exactly
   * when a baseline family is asked for.
   *
   * TWO things are checked against it and they are different checks.
   * `planBakeoff` requires its sha256 to equal the IR's `policyHash`, which is
   * what makes B and the compiled arm the same experiment. `runBakeoff` then
   * requires the PAGE's own digest of the document it bundled to equal this
   * file's -- the same two-sided check `irHash` gets, and the one that catches
   * a dev server left running by another worktree.
   */
  readonly policyPath?: string;
  /**
   * The page's policy-registry name for that document. Defaults to `irName`,
   * and to `DEFAULT_IR_NAME` when that is unset too.
   *
   * CORRECTED: this said it defaults to "`policyName`'s basename minus `.md`",
   * which is neither what the code does nor a sentence that parses -- a field
   * cannot default from itself, and `policyPath` is the only path here. The
   * real default is `irName`, because the page's two registries are keyed by
   * the SAME names for the two halves of one compile: `useIr("p-fin")` and the
   * policy fixture "p-fin". `planBakeoff`'s own inline comment said so while
   * this one said otherwise, and nothing detected the disagreement because no
   * caller in this repository sets this field -- `bakeoff.test.ts` pins the
   * three-level fallback explicitly for that reason.
   *
   * A NAME rather than the text, for the reason `irName` is one: a page that
   * accepted arbitrary policy text would make half a run's provenance
   * unreproducible from this repository.
   */
  readonly policyName?: string;
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
   * re-admit that fence to escalation. `semantic-ir.json` -- one of the two IRs
   * a tier-2 arm can run, since `planBakeoff` throws on any IR with no
   * `semanticPredicates`, and `policies/compiled/p-fin.ir.json` is the other
   * one here that has one -- shipped `rules: []` for one commit, so `runTier0`
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
  /**
   * The policy DOCUMENT the Approach-B arms are shown, when there are any.
   *
   * `undefined` exactly when no baseline family was asked for -- `planBakeoff`
   * refuses a baseline family without one, and refuses one whose sha256 is not
   * the IR's `policyHash`. So a present value has already been proven to be the
   * document this run's IR was compiled from, which is the whole premise of the
   * comparison.
   *
   * `chars` is here rather than derivable later because it is the number that
   * says how much bigger B's prompt is than the compiled judge's: the p95 TTFT
   * ceiling was derived at a ~1.1 kB whole prompt, and this document alone is
   * several times that.
   */
  readonly policy:
    | { readonly name: string; readonly sha256: string; readonly chars: number }
    | undefined;
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

/**
 * The page IR registry name a run selects unless told otherwise.
 *
 * `"semantic"` and not `"p-fin"`, deliberately, and the difference is a real
 * experiment variable rather than inertia. `apps/eval/fixtures/semantic-ir.json`
 * carries `latencyBudgetMs: 120000` -- 24x the compiler's default -- so a
 * compiled-family smoke run against it exercises completed judgements;
 * `policies/compiled/p-fin.ir.json` carries the compiler's own 5,000 and at
 * Plan 5's measured 4.6 s per call an arm run against it degrades on the budget.
 * Both are true measurements of different questions, and the head-to-head needs
 * the compiled one because only it can be paired with a document.
 */
const DEFAULT_IR_NAME = "semantic";

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
        `schema-valid file of zero model calls. Point --ir at a policy with a semantic clause; ` +
        `two here have one -- apps/eval/fixtures/semantic-ir.json, a hand-written fixture at a ` +
        `120,000ms message budget, and policies/compiled/p-fin.ir.json, real compiler output at ` +
        `the compiler's 5,000ms default and the only one that can be paired with a document.`,
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
  let policy: BakeoffPlan["policy"];
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
    policy = {
      // Defaulted from the IR name rather than required, because the page's two
      // registries are keyed by the SAME names for the two halves of one
      // compile -- `useIr("p-fin")` and the policy fixture "p-fin". A caller
      // pairing them differently has to say so.
      name: options.policyName ?? options.irName ?? DEFAULT_IR_NAME,
      sha256: documentHash,
      chars: policyText.length,
    };
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
        // The unit ONE engine call covers on this arm -- the family AND this
        // policy's declared scopes -- so the distribution describes the work the
        // arm does rather than the work its paired family does, or the work a
        // differently-scoped policy would have made it do. An Approach-B arm is
        // shown whole messages; a segment distribution under its name would put
        // a p50 of 62 characters beside a p95 TTFT taken on a ~5.9 kB prompt.
        // And on `p-fin`, whose one predicate is message-scoped, a COMPILED arm
        // is shown whole messages too -- which is the reading this argument was
        // hardcoded per family and could not produce.
        unit: judgedUnitFor(family, ir.semanticPredicates),
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
      // One call per judged unit plus the single repair retry.
      //
      // ONE expression over every arm, because `segments` above is now measured
      // over the unit `judgedUnitFor` derives from the family AND the policy's
      // declared scopes -- so `perItem` already counts the whole-message call
      // where the policy declares a message-scoped predicate, and counts no
      // segment call where it declares no segment-scoped one. That is the same
      // arithmetic the previous two-term expression did, off the same
      // `ir.semanticPredicates`, one layer up: an Approach-B arm judges one unit
      // per item whatever the message contains, a compiled arm judges its
      // selected segments plus one message where both scopes are declared.
      //
      // The two failures it must keep avoiding, both measured before this was
      // read off the scopes at all. Counting segments alone undercounts a
      // both-scopes policy by `callsPerJudgedUnit` calls an item -- and
      // `deriveItemTimeoutMs` turns an undercounted ceiling into a deadline a
      // legitimate item exceeds, which errors the row and stamps every later row
      // of the arm `abandonedWorkInFlight`. On a message-only policy it also
      // undercounts to ZERO whenever escalation selects nothing, which is the
      // refusal below firing on an arm that in fact makes one whole-message call
      // per item.
      //
      // The MAXIMUM of this arm's own per-item sample, never the p95: at n = 13
      // those are the same rank anyway, and a ceiling built on a percentile
      // would be a ceiling the worst message exceeds by construction.
      const maxCallsPerItem = shape.callsPerJudgedUnit * (segments.perItem?.max ?? 0);
      if (maxCallsPerItem === 0) {
        throw new Error(
          `arm "${arm}" would make no engine call on any of the ${items.length} corpus items: ` +
            `its judged unit is "${segments.unit}" because none of the IR's ` +
            `${ir.semanticPredicates.length} semantic predicate(s) is message-scoped, so this ` +
            `arm's only calls would be segment calls, and escalation selected no segment ` +
            `anywhere (${segments.segmentsTotal} segment(s) in total). Its file would be a ` +
            `complete, schema-valid transcript of a model that never ran.`,
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
        //
        // `uncertainBelow` is on the COMPILED families only. It is the threshold
        // `escalate.ts` compares a prior tier's confidence against to decide
        // which SEGMENTS reach the judge; Approach B judges the whole message in
        // one call and never escalates, so on a B arm it would be a knob that
        // turned nothing -- and `gateReport` compares this number against the
        // one the planned distribution was measured at, which would make the two
        // agree about work no B arm performs. `RunRecordSchema` refuses a B row
        // that carries one.
        config: {
          tier0: shape.runsTier0,
          tier1: false,
          tier2: true,
          t2Model: model.modelId,
          ...(shape.runsCompiledJudge ? { uncertainBelow } : {}),
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
    policy,
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
 * WHY A GATES ROW NEEDS IT. `ir.latencyBudgetMs` is not a detail of a fixture,
 * it is the deadline the orchestrator arms over the WHOLE `judge()` call -- and
 * the two IRs a bake-off can run here sit 24x apart on it.
 * `apps/eval/fixtures/semantic-ir.json` carries 120,000;
 * `policies/compiled/p-fin.ir.json` carries this number, because the compiler
 * emits it for a policy that names no budget. So the multiple belongs on the
 * row rather than in a reader's head: `run.latencyBudgetTimesCompilerDefault`
 * is 24 on a semantic-ir run, whose latencies therefore do NOT transfer to a
 * shipped configuration, and 1 on a p-fin run, whose latencies do. See the
 * module header for the arithmetic (Plan 5's measured 4.6 s per call on the
 * cheapest arm, against a 5,000 ms message budget) and
 * `ArmRunContext.latencyBudgetMs` for what each of the two conditions measures
 * and what it leaves unmeasured.
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
   * The smallest `sample` this gate will return a pass or a fail on.
   *
   * ## Why a gate needs one
   *
   * MEASURED, in `runs/slate-p-fin-01.gates.jsonl`: two arms
   * (`tier2-Ministral-3-3B-...` and its `tier2only-` twin) were killed on
   * `resolvable-rate` reading "1 of 2 quote(s) resolved ... a rate of 0.500
   * against a floor of 0.8". At a sample of two the only rates reachable are 0,
   * 0.5 and 1, and two of those three are under the floor -- so what the gate
   * actually tested was "did this arm produce one unplaceable quote", while
   * `killedOnRunGates`, the field a reader takes as the bake-off's judgement of
   * an arm, said the arm had failed a rate gate. A floor of 0.8 is a statement
   * that some unplaceable quotes are tolerated; a gate that fails on the first
   * one contradicts its own threshold.
   *
   * ## How the number is arrived at, per gate
   *
   * For the two gates whose observation is a RATE over a denominator the arm
   * itself chooses, it is `minSampleForOneContrary` on that gate's own
   * threshold: the smallest sample at which one contrary observation does not
   * by itself cross the line. `resolvable-rate` at a floor of 0.8 gives 5,
   * `duplicate-rate` at a ceiling of 0.9 gives 2, and the two differ because
   * the thresholds do -- a single shared minimum would be a number chosen
   * rather than derived, and wrong for one of them whichever it was.
   *
   * For the rest it is 1, which is a decision rather than an omission.
   * `p95-ttft` and `decode-rate` take one observation per answered CALL and the
   * corpus fixes how many calls an arm makes (12 or 13 for every arm in the
   * slate that answered at all), so neither is quantised by a denominator the
   * arm controls; and neither threshold has slack for the derivation to find --
   * a 1,500 ms ceiling tolerates no call over 1,500 ms, which
   * `GATES.maxP95TtftMs` argues at length. `non-empty-after-stop` is a
   * predicate over one specific event, so one observation is the whole
   * population there is.
   */
  readonly minSample: number;
  /**
   * `not-measured` is a THIRD answer and not a quiet pass.
   *
   * TWO situations produce it and `observed` is what tells them apart.
   *
   * NOTHING TO MEASURE (`observed` undefined). An arm whose every message ran
   * out of budget before its first call has no latency to take a percentile of
   * and no finding to score a ladder on. Calling that a failure kills it for
   * the budget overrun by the back door, which is the one rule this module must
   * not have; calling it a pass hides that the arm was never measured.
   *
   * TOO LITTLE TO RULE ON (`observed` present, `sample` under `minSample`).
   * There is a number and it is reported; what is withheld is the verdict,
   * because at that sample a single contrary observation crosses the threshold
   * on its own and a pass or a fail would be about that one observation rather
   * than about the arm.
   *
   * Neither sets `killedOnRunGates`. They are one verdict value and not two
   * because that field must treat them identically, and a fourth enum member
   * would be a second spelling of "this row carries no verdict" for every
   * reader to handle; `observed`, `sample` and `minSample` already separate
   * them on the row.
   */
  readonly verdict: "pass" | "fail" | "not-measured";
  readonly detail: string;
}

/**
 * The sample-size search bound in `minSampleForOneContrary`.
 *
 * Far above anything this driver can produce -- MEASURED, the 16-arm slate in
 * `runs/slate-p-fin-01.gates.jsonl` produced rate samples of 0 to 8 -- so it is
 * not a limit anything reaches. It exists so that a threshold with NO slack in
 * it (a floor of 1.0, a ceiling of 0) is a loud refusal rather than a gate that
 * quietly never rules.
 */
const MIN_SAMPLE_SEARCH_LIMIT = 1_000;

/**
 * The smallest sample at which ONE contrary observation does not, by itself,
 * decide a rate gate's verdict.
 *
 * COMPUTED from the gate's own threshold, not chosen: it searches for the first
 * `n` at which the gate's own `passes` accepts the rate that a sample of `n`
 * with exactly one contrary observation produces. Solving the two inequalities
 * by hand gives `n >= 1/(1 - floor)` and `n >= 1/ceiling`, so 0.8 gives 5 and
 * 0.9-as-a-ceiling gives 2; the search is used instead of the closed form
 * because `1 / (1 - 0.8)` in doubles is 5.000000000000001, and because the
 * search uses the gate's ACTUAL comparator, so the derivation and the verdict
 * cannot disagree about the boundary case.
 *
 * WHAT IT IS NOT. It is not a confidence interval and makes no distributional
 * claim; there is no significance test anywhere in this file and this does not
 * add one. It answers one narrow question -- below what size is this gate's
 * verdict a statement about a single observation rather than about the arm --
 * and that question has an exact answer given a threshold.
 *
 * `oneContraryAt` is the rate a sample of `n` with one contrary observation
 * produces: `(n - 1) / n` for a floor gate counting successes, `1 / n` for a
 * ceiling gate counting failures.
 */
export function minSampleForOneContrary(spec: {
  passes: (observed: number) => boolean;
  oneContraryAt: (n: number) => number;
}): number {
  for (let n = 1; n <= MIN_SAMPLE_SEARCH_LIMIT; n += 1) {
    if (spec.passes(spec.oneContraryAt(n))) return n;
  }
  throw new Error(
    `this gate tolerates no contrary observation at any sample up to ` +
      `${MIN_SAMPLE_SEARCH_LIMIT}: at every size, one contrary observation alone crosses its ` +
      `threshold. No minimum sample can separate "this arm is over the line" from "this arm ` +
      `produced one bad observation" here, so returning a number would be a gate that never ` +
      `rules wearing a derivation. Give the threshold slack, or give the gate a kind of verdict ` +
      `a single observation can honestly support.`,
  );
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
   * THE CAVEAT, AND WHICH RUNS IT APPLIES TO. `planBakeoff` refuses any IR with
   * no `semanticPredicates`, and TWO IRs here have one:
   * `apps/eval/fixtures/semantic-ir.json` at 120,000 -- 24x
   * `COMPILER_DEFAULT_LATENCY_BUDGET_MS` -- and
   * `policies/compiled/p-fin.ir.json` at 5,000, which is what the compiler
   * emits for a policy that does not name a budget and what `minimal-ir.json`
   * and `multiclass-ir.json` carry too.
   * `run.latencyBudgetTimesCompilerDefault` on this row says which of the two
   * produced it, and the caveat below is the 24x one's alone.
   *
   * The fixture's 120,000 is not a mistake: at 5,000 the deadline fires during
   * the first call of every tier-2 spec, so nothing would exercise a completed
   * judgement (see the module header, and Plan 5's measured 4.6 s per call on
   * the CHEAPEST arm against a 5,000 ms whole-message budget). And a p-fin run
   * is the other side of the same coin -- its latencies ARE taken at a shipped
   * policy's budget, and what it under-samples is completed judgements.
   * `test/baseline.spec.ts` runs that condition over three items and asserts
   * the multiple is 1.
   *
   * What a 24x run costs, stated rather than hidden: the DEGRADATION a shipped
   * policy's budget would cause is not measured anywhere in it. The
   * per-call numbers (`ttftMs`, `decodeTokPerSec`) are per-call and a shorter
   * message budget does not slow a call, but it changes WHICH calls happen --
   * the orchestrator's one deadline cuts the message off mid-judgement, so a
   * run at 5,000 would have a smaller sample, more calls ended by an interrupt,
   * far fewer `ladder.unitsJudged` and far more
   * `degradedNotices["budget-exhausted"]`. None of those differences is
   * observable from this file.
   *
   * WHAT IT WOULD TAKE to measure it instead, since it is cheap and this
   * comment should not be the end of it. Running p-fin is NOT it: p-fin is at
   * 5,000 but carries a different predicate and a different entityType
   * vocabulary, so the difference between its rows and a semantic-ir run's is
   * two variables at once. The single-variable version is one second fixture --
   * `semantic-ir.json` with `latencyBudgetMs` at 5,000 and nothing else changed
   * -- plus one line in `IR_FIXTURES` in `apps/eval/src/page/main.ts`, and then
   * a second `runBakeoff` under a different `runId` and `irName`, with an
   * `itemTimeoutMs` matching the much smaller bound `planBakeoff` derives at
   * that budget. The degradation rate is then the ratio of
   * `ladder.unitsJudged` and of `degradedNotices["budget-exhausted"]` between
   * the two gates rows. It is
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
   * because the thing it is evidence for -- WHETHER these latencies transfer to
   * a shipped configuration -- is invisible while the two numbers sit in
   * different files. Both values occur on runs this driver performs today: 24
   * against `apps/eval/fixtures/semantic-ir.json`, and 1 against
   * `policies/compiled/p-fin.ir.json`, whose budget IS the compiler's default
   * because `p-fin.md` names none. `test/baseline.spec.ts` asserts the 1 on
   * every row of the four-family head-to-head; `bakeoff.test.ts` asserts the 24.
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

/**
 * The HARDWARE-COST half of the research question, taken off the load report.
 *
 * WHY THIS EXISTS. The project's stated deliverable is an answer to "how well
 * can policy-conditioned, fully-local models prevent confidential-data leakage
 * in LLM prompts, and at what latency AND HARDWARE COST?" -- and until this
 * type, the hardware half was produced and thrown away. `Tier2LoadReport`
 * carries all three of these numbers, `runBakeoff` already holds one per arm
 * and asserts three other fields of it, and then nothing wrote them anywhere: no
 * field of a record and no field of a gates row could answer "what did this arm
 * cost to stand up". Every latency this bake-off reports is a per-CALL number
 * measured after the model is already resident, so none of them contains the
 * cost of getting it there.
 *
 * Three numbers and not four: `Tier2LoadReport.storageQuotaBytes` is a property
 * of the browser profile and the machine's free disk (Chrome derives it from
 * free space and it moves), not a cost this arm paid, so it is not here.
 */
export interface ArmLoadCost {
  /**
   * `Tier2LoadReport.loadMs`: wall clock for `CreateMLCEngine` alone.
   *
   * NOT NECESSARILY A COLD LOAD, and the name says `load` rather than
   * `coldLoad` for that reason. web-llm caches weights in the Cache API under
   * the page's origin, so this is a download plus a shader compile on a profile
   * that has never held the model and a cache read plus a shader compile on one
   * that has -- and the tier-2 specs run against a persistent profile precisely
   * so the second is the usual case. Nothing in the load report distinguishes
   * them; `originStorageBytes` below is the only signal in this block that the
   * cache was already populated, and it is a weak one. Read a bake-off's
   * numbers as the arm's WARM stand-up cost unless the profile was fresh.
   */
  readonly engineLoadMs: number;
  /**
   * `Tier2LoadReport.warmupMs`: wall clock for the one throwaway completion
   * `loadTier2` makes before it returns.
   *
   * Separate from `engineLoadMs` because it is a different cost with a different
   * cause -- the first generation on a freshly compiled pipeline -- and folding
   * the two into one "startup" number would hide which half a slow arm paid.
   */
  readonly engineWarmupMs: number;
  /**
   * `Tier2LoadReport.storageUsageBytes`: `navigator.storage.estimate().usage`
   * after this arm's load, in bytes.
   *
   * CUMULATIVE OVER THE ORIGIN, and therefore NOT this model's footprint. Every
   * model the profile has ever cached is in this number, so across a slate run
   * cheapest-first it climbs monotonically and the arm that ran last reports the
   * whole slate. It is a footprint measurement of the PROFILE at the moment this
   * arm loaded, which is the thing that actually runs out (`tier2-profile.ts`
   * measured an ordinary Playwright context reporting a 3,221 MB quota against
   * 7.49 GB of weights for the four pinned arms) -- and it is the only footprint
   * number the browser exposes at all.
   *
   * For a per-MODEL size, read `TIER2_MODELS[].vramRequiredMb`, which is
   * `prebuiltAppConfig`'s `vram_required_MB`. That is not copied here on
   * purpose: it is a constant from the library, not an observation of this run,
   * and `manifest.ts` records that it is 2.1x to 3.4x away from the download
   * size on the two models where both were measured.
   */
  readonly originStorageBytes: number;
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
   * exists here and none should: spec 2.2 makes the TS/Python boundary a JSONL
   * file and puts scoring in `analysis/`'s Python, which this repository has
   * not written (the spec says nothing about plan numbering; deferring it to
   * Plan 8 is ours). Every gate on this report is a property of the run.
   */
  readonly accuracyGated: false;
  /** Per tier: whether this arm ran it, and what gold there is to score it with. */
  readonly tiers: readonly TierGoldCoverage[];
  /** The tiers of `tiers` whose `ran` is true, ascending. */
  readonly tiersRun: readonly Tier[];
  /** `goldSpans` by tier, so a reader does not have to walk `tiers`. */
  readonly goldSpansByTier: Readonly<Record<Tier, number>>;
  /**
   * Tiers this arm RAN that ITS ROWS carry no gold for.
   *
   * NON-EMPTY MEANS THIS ARM'S FINDINGS AT THOSE TIERS CANNOT BE SCORED against
   * these rows, in either direction: no recall, because there is nothing to
   * recall, and no precision, because every finding is unmatched. `[2]` on
   * every arm of a bake-off run against the shipped corpus today.
   *
   * NAMED FOR THE ROWS AND NOT FOR THE CORPUS, which is what it used to be
   * called. `scoringBoundary` counts over `records` -- this arm's rows -- and
   * on the real path those are 1:1 with the corpus items (an errored row still
   * carries its `gold`, see run.ts). They are NOT 1:1 when a caller runs a
   * SLICE, which `test/bakeoff.spec.ts` and `test/baseline.spec.ts` both do:
   * three items of a thirteen-item corpus whose other ten carry most of the
   * gold. A field named for the corpus would then assert that a corpus with
   * five tier-0 gold spans cannot score tier 0.
   */
  readonly tiersTheseRowsCannotScore: readonly Tier[];
  /**
   * Tiers this arm did NOT run that its rows DO carry gold for, ascending.
   *
   * THE MIRROR of the field above, and it was missing: the join comes back
   * empty in two directions and only one of them had a name. Gold at a tier no
   * arm ran is not a precision problem -- there are no findings to be wrong --
   * it is a RECALL DENOMINATOR problem, and it is silent, because every count
   * a scorer needs is present and nothing says the arm could not have filled it.
   *
   * `[1]` on every arm of a bake-off run against the shipped corpus today:
   * `smoke.jsonl` carries two `client-name` spans, `client-name` is tier 1 in
   * both runnable IRs, and `planBakeoff` sets `tier1: false` on every arm it
   * plans. So a recall computed over `record.gold` is capped at 5/7 on the
   * whole corpus for every arm here, for a reason that is about the run's
   * configuration and not about the model. `cannotScore` carries the sentence.
   */
  readonly tiersWithGoldThisArmDidNotRun: readonly Tier[];
  /**
   * Gold entityTypes the IR that ran declares no tier for, sorted.
   *
   * A different way the same join comes back empty, and on this corpus it is
   * not a standing candidate, it is already the case. `smoke.jsonl`'s items are
   * labelled `policy: "minimal-fixture"` and their gold ids are `in-pan`,
   * `aws-key`, `generic-secret` and `client-name`. `semantic-ir.json` declares
   * all four, so a run against that fixture leaves this empty --
   * `policies/compiled/p-fin.ir.json` declares neither `aws-key` nor
   * `generic-secret`, so the compiled head-to-head `test/baseline.spec.ts`
   * runs populates it. MEASURED on that run's three-item slice: it is
   * `["aws-key", "generic-secret"]` on all four rows. A span in no tier bucket
   * would otherwise vanish from the counts above rather than show up as
   * unmatchable.
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
   * What this report's verdicts are and are not, and which tiers THIS ARM'S
   * ROWS can be scored at. Read it before reading `killedOnRunGates`.
   */
  readonly scoring: ArmScoringBoundary;
  /**
   * Which axes of spec 6.3's experiment matrix THIS RUN moves along, and which
   * it holds fixed at one point.
   *
   * On the row for the same reason `scoring.verdictMeans` is: a gates file is
   * read by someone who does not have the spec open, and the single most
   * available misreading of it is that it answers the research question. The
   * model and method points are read off the PLAN rather than asserted, so the
   * default one-family slate does not claim a four-way head-to-head; see
   * `experimentScope`.
   */
  readonly experimentScope: string;
  /**
   * What this arm cost to stand up, off the page's own load report.
   *
   * The three fields below are the hardware half of "at what latency and
   * hardware cost?", and every other number on this report is measured after
   * the model is already resident. See `ArmLoadCost` for what each one is and,
   * in two cases, what it is not.
   */
  readonly engineLoadMs: number;
  readonly engineWarmupMs: number;
  readonly originStorageBytes: number;
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
   * CHARACTER figure the threshold was derived at -- see `judgedUnitChars` for a
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
  /**
   * The span ladder and the stop accounting, summed over the arm.
   *
   * ONE shape for both methods, and the two renames are deliberate rather than
   * a translation layer's accident. `BaselineStats` calls two of these events
   * something else because Approach B judges a MESSAGE and names an ENTITY
   * CLASS where the compiled judge judges a segment and names a predicate; the
   * events are the same events, and a bake-off putting the two arms in one
   * table needs one column per event. So:
   *
   *   - `unitsJudged` is `segmentsJudged + messageScopeJudged` on a compiled arm
   *     and `messagesJudged` on a B arm: every call whose answer was collected,
   *     in whatever unit(s) `judgedUnit` on this report names. It is deliberately
   *     NOT named for either unit -- a column named `segmentsJudged` holding a
   *     count of messages is the defect this rename exists to prevent -- and it
   *     is a SUM rather than a rename on the compiled side because that judge
   *     has two kinds of judged unit where B has one. It is the population
   *     `judgedUnitsPerItem` predicts: a healthy compiled arm on a message-only
   *     policy plans one unit an item and judges one an item, and while this
   *     read `segmentsJudged` alone it planned one and reported zero.
   *   - `unknownLabels` is `unknownPredicates` on a compiled arm and
   *     `unknownEntityTypes` on a B arm -- a model inventing a label, in two
   *     vocabularies.
   *
   * Two fields have no counterpart on the other side and are `undefined` rather
   * than 0 there, because 0 would be the positive claim that the event happened
   * zero times:
   *
   *   - `unitsSkipped` (the judge's `segmentsSkipped`) is meaningless for an arm
   *     that makes one call per message: there is no second unit to skip.
   *   - `messageBudgetExpiries` is B's alone. `detect` arms the message deadline
   *     for the compiled path and files a notice instead of counting, so the
   *     compiled arms report this event only through
   *     `degradedNotices["budget-exhausted"]`.
   */
  readonly ladder: {
    readonly rung1: number;
    readonly rung2: number;
    readonly unresolvedQuotes: number;
    /**
     * Findings whose EVIDENCE clause placed and whose MENTION did not, so no
     * action span existed and the finding was dropped. Counted apart from
     * `unresolvedQuotes` because the two say different things about the model:
     * one is a quote that is not in the text, the other is a model that quoted
     * correctly and then pointed outside its own quote or at something the
     * clause says twice. Both are in `resolvable-rate`'s denominator.
     */
    readonly unresolvedMentions: number;
    /**
     * Findings whose mention resolved to the whole clause -- the model
     * answering that no smaller span will do. NOT a loss, and deliberately not
     * gated: it is legitimate for a predicate about a clause with no
     * extractable entity. It is reported because an arm at
     * `wholeClauseMentions === rung1 + rung2` narrowed nothing at all, and
     * every span it emitted is a clause `applyActions` would rewrite whole --
     * which is invisible in every other number on this row.
     */
    readonly wholeClauseMentions: number;
    readonly duplicatesDropped: number;
    readonly unknownLabels: number;
    readonly failedClosed: number;
    readonly truncatedResponses: number;
    readonly abortedResponses: number;
    readonly repairAttempts: number;
    readonly unitsJudged: number;
    /** The judge's `segmentsSkipped`. `undefined` on a message-judged arm. */
    readonly unitsSkipped: number | undefined;
    /** Approach B's own message-budget stops. `undefined` on a compiled arm. */
    readonly messageBudgetExpiries: number | undefined;
    /**
     * The compiled judge's whole-message calls, `undefined` on a B arm.
     *
     * `undefined` rather than 0 there for the reason the two fields above are:
     * Approach B judges the whole message in its ONLY call, so it has no
     * separate message-scope call to count and a 0 would be the positive claim
     * that it made none.
     *
     * `messageScopeJudged` is the message HALF of `unitsJudged` on a compiled
     * arm, not a column beside it: the two are added there. On a policy whose
     * predicates are all message-scoped -- `policies/compiled/p-fin.ir.json` is
     * one -- the judge makes no segment call at all, so this half is the whole
     * of the arm's coverage and `judgedUnit` on this row reads "message" to say
     * so. The two message-scope counters are NOT the same population and the
     * difference is not the failure count: READ from `WebLlmJudge.#judgePassage`,
     * `onCallIssued` fires inside the repair loop, so `messageScopeCalls` counts
     * ENGINE CALLS including a repair retry, while `messageScopeJudged` counts
     * message PASSAGES whose answer was collected. On a message-only policy the
     * second is at most the item count and the first exceeds it by one per
     * message that needed a repair.
     */
    readonly messageScopeCalls: number | undefined;
    readonly messageScopeJudged: number | undefined;
    readonly messageScopeFailedClosed: number | undefined;
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
   * What one engine call covers on this arm: the whole MESSAGE on the
   * Approach-B families, and on a compiled family whatever the POLICY's
   * declared scopes make it -- a selected SEGMENT, the whole message, or both.
   *
   * On the report because the two fields below are counted over it and a
   * character p50 does not say which. The value emitted here is the one
   * `judgedUnitFor` DERIVES from the family and the IR's `semanticPredicates`,
   * not the one the caller's distribution carries; `gateReport` throws when the
   * two disagree, so the check is what makes the derived answer also the
   * distribution's rather than the other way round.
   */
  readonly judgedUnit: JudgedUnit;
  /**
   * The character-size distribution of the units this arm's model is shown, one
   * sample per engine call the arm would make.
   *
   * The evidence for `GATES.maxP95TtftMs` in the unit its ~1.1 kB derivation is
   * quoted in -- `promptTokens` is the same calls in the engine's tokens, and
   * neither converts into the other without the model's own tokenizer.
   *
   * READ IT WITH `judgedUnit`, and on any arm whose unit is not `"segment"` read
   * the p95 TTFT gate's derivation as not applying: that ceiling was derived at
   * a ~1.1 kB WHOLE PROMPT, of which 776 characters are the judge's fixed system
   * turn and the rest is ONE SEGMENT. Approach B's prompt carries the whole
   * policy document on every call -- `policies/p-fin.md` alone is 5,272
   * characters -- so a B arm's prompt is several times the size the threshold
   * was set at, and its TTFT is not measuring the same thing. A COMPILED arm on
   * a message-only policy is a milder case of the same mismatch and it is the
   * one the head-to-head actually runs: no policy document in the prompt, but
   * the whole message rather than one segment of it (MEASURED over `p-fin` and
   * three items: 297 prompt tokens against B's 1,433, on message passages whose
   * median is 110 characters where the selected segments' was 45). The gate is
   * still computed and still reported, because suppressing it would hide the
   * number; what must not happen is reading such a verdict as a comparable one.
   * The evidence for saying so is on the row: this field and `promptTokens`.
   *
   * PLANNED, not observed, and the name is only as true as that: it is computed
   * in Node by `planBakeoff` before the first model loads, from core's own
   * segmenter and the escalation policy under THIS arm's tier-0 setting -- so
   * it is the set of units a healthy run of this arm would judge, not a count
   * taken off the rows. No record carries what the page actually handed the
   * model, so this is the closest population the file has. The three ways a
   * real run diverges from it are all on this same report and should be read
   * beside it: `itemsErrored` (an item that threw judged nothing),
   * `degradedNotices["budget-exhausted"]` (a message that stopped early) and
   * `ladder.unitsJudged` (what the model actually saw).
   *
   * What is checked rather than trusted: `gateReport` refuses a distribution
   * whose JUDGED UNIT or ESCALATION CONDITION disagrees with the rows, so this
   * cannot be the OTHER family's distribution -- the two differ on this corpus.
   * There are three such guards and no fourth. What is NOT checked is that the
   * distribution was measured over the same CORPUS SLICE as the rows: neither
   * side carries a corpus identity, and the obvious proxy -- `segments.items`
   * against `records.length` -- would be a check on the fixture, for the reason
   * `gateReport`'s own comment gives at length where the guards are. That
   * coupling is pinned in `bakeoff.test.ts`'s end-to-end `runBakeoff` case
   * instead, which asserts each report's distribution IS its own arm's.
   */
  readonly judgedUnitChars: SizeStats | undefined;
  readonly judgedUnitsPerItem: SizeStats | undefined;
  readonly escalation: SegmentSizeDistribution["escalation"];
  readonly gates: readonly GateOutcome[];
  /**
   * True when any gate on this report FAILED. `not-measured` never sets it,
   * in EITHER of the two situations that produce it: a gate with nothing to
   * measure, and a gate whose sample is under its own `minSample`. The second
   * one is why the two Ministral arms of `runs/slate-p-fin-01.gates.jsonl`
   * would not carry `killedOnRunGates: true` if that run were repeated -- both
   * were killed on `resolvable-rate` over a sample of two quotes.
   *
   * NAMED FOR ITS SCOPE, and the rename from `killed` is the whole point of the
   * name. `killed` reads as the bake-off's answer to "which model won", and it
   * is not one: the gates it sums are latency, decode rate, span-ladder
   * resolvability, restatement rate and engine poisoning, and not one of them
   * looks at whether the arm was RIGHT. Spec 4.2's stated primary criterion is
   * task accuracy and no accuracy metric exists in this repository -- spec 2.2
   * makes the JSONL file the whole TS/Python boundary and puts scoring in
   * `analysis/`'s Python, which nobody has written here. So an
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
  /**
   * The IR's `semanticPredicates`, which is the only thing that knows what UNIT
   * this arm's model was shown.
   *
   * The fifth thing no record carries, and it is here for the reason
   * `entityTypes` above is: it decides a field of the report and the rows cannot
   * supply it. `judgedUnitFor` reads the declared scopes off this array, and the
   * check below refuses a distribution measured over a different unit -- so a
   * compiled arm on a message-only policy can no longer be handed the segment
   * distribution its family used to name. Only the SCOPES are read; nothing here
   * looks at a predicate's id or text.
   *
   * It carries the same honest limitation `entityTypes` does: nothing here can
   * check the array came from the IR the arm ran. `runBakeoff` passes the IR it
   * has already proved is the page's.
   */
  readonly semanticPredicates: PolicyIr["semanticPredicates"];
  /**
   * What the arm cost to stand up, as the PAGE reported it.
   *
   * The sixth thing no record carries, and the one that breaks the pattern of
   * the five above: those are numbers this driver chose, and these three are
   * observations the page made. `runBakeoff` holds the `Tier2LoadReport` they
   * come from -- it already reads `servedModelId`, `config` and `callBudgetMs`
   * off the same object and refuses the arm on any of the three -- so on the
   * real path they are the load that produced the rows below them. Nothing here
   * can check that, exactly as nothing can check `entityTypes`.
   */
  readonly load: ArmLoadCost;
  /**
   * The axes the WHOLE run crosses, which no single arm's rows can show.
   *
   * The seventh thing no record carries, and the reason it is here rather than
   * inferred: a gates row names one arm, and "which methods did this run
   * compare" is a question about the other rows. `experimentScope` used to
   * answer it from a constant that named four methods and four models whatever
   * ran, which on the default slate -- `DEFAULT_FAMILIES` is `["compiled"]` --
   * was every row of a one-method run claiming a compiled-versus-Approach-B
   * head-to-head that did not happen.
   */
  readonly slate: SlateAxes;
}

/**
 * What one `runBakeoff` crosses, straight off `BakeoffPlan.arms`.
 *
 * DISTINCT VALUES in plan order, not the arm list: four arms of one family are
 * one point on the method axis, and the sentence this feeds is about points.
 */
export interface SlateAxes {
  readonly models: readonly string[];
  readonly families: readonly ArmFamily[];
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
  "TS/Python boundary and puts scoring in analysis/'s Python -- a directory this repository does " +
  "not have. (Spec 2.2 is titled 'Repository layout' and says nothing about plan numbering: " +
  "deferring that Python to Plan 8 is this project's own sequencing, not the spec's.) What IS " +
  "computed here and written to this file is five run gates -- p95-ttft, decode-rate, " +
  "resolvable-rate, duplicate-rate, non-empty-after-stop -- each with a threshold, an observation " +
  "and a verdict. An arm can pass every one of them and be the worst model on the slate. A gate " +
  "whose sample is under its own minSample reports not-measured and rules on nothing: below that " +
  "size one contrary observation crosses the threshold by itself, so the observed value is " +
  "reported and the pass/fail is withheld. Read every verdict beside the sample it was taken on.";

/** The METHOD axis's four points, in prose, so a scope sentence can name the real ones. */
const FAMILY_PROSE: Readonly<Record<ArmFamily, string>> = {
  compiled: "tier 0 + compiled judge",
  "compiled-tier2-only": "compiled judge",
  "baseline-b": "Approach B",
  "baseline-b-tier0": "B + tier 0",
};

/**
 * Which axes of spec 6.3's experiment matrix THIS run moves along.
 *
 * COMPUTED PER RUN and not a constant, which is the correction. It used to be a
 * constant, on the argument that nothing in it varies per arm and a per-report
 * sentence is one two rows of a file can disagree on. The second half of that
 * is right and the first half was false three times over, and each falsification
 * was reachable from the shipped command:
 *
 *   - it claimed the run crosses the METHOD axis at four points. `slateBakeoffOptions`
 *     leaves `families` unset, `DEFAULT_FAMILIES` is `["compiled"]`, so
 *     `SIH_BAKEOFF=1 ... pnpm -C apps/eval bakeoff` crosses ONE method and every
 *     row of its gates file claimed four -- overstating the project's central
 *     claim (does compiling help?) in the field that exists to stop a reader
 *     concluding more than the run supports;
 *   - it claimed "the four pinned arms" regardless of `SIH_BAKEOFF_MODELS`;
 *   - it ended "the corpus carries no gold for the tier every arm here runs",
 *     a property of `options.corpus`, which `SIH_BAKEOFF_CORPUS` makes free and
 *     which Plan 7 exists to change. On a labelled corpus the same ROW would
 *     have said `scoring.tiersTheseRowsCannotScore: []` and, in prose, that
 *     there is no gold -- two contradictory statements in one record.
 *
 * So the varying parts are read off the plan and off this row's own `scoring`,
 * and the fixed parts stay fixed. Two rows of ONE file still cannot disagree on
 * the model and method axes: both come from the plan, which is the same object
 * for every arm. They CAN differ on the scoring clause, and should -- two
 * families run different tier sets, so what their rows can score differs.
 *
 * What stays a constant is what no run can change:
 *
 *   - the backend axis has ONE point because this driver refuses to run at all
 *     without WebGPU (see `runBakeoff`), and web-llm has no second backend;
 *   - the policy axis has one usable point because `planBakeoff` throws on any
 *     IR with no `semanticPredicates` and `policies/compiled/` holds one policy
 *     of the three, `p-fin`, `scripts/compile-policies.ts` reporting on every run
 *     that `p-med` and `p-corp` have no fixture answering their prompts. (Two
 *     IRs are runnable -- `apps/eval/fixtures/semantic-ir.json` is the other --
 *     but only one of them is a compiled POLICY, and only a compiled policy can
 *     be paired with the document Approach B is shown.)
 *
 * The machine sentence is the one thing here that no code can check, and it is
 * the reason the paragraph exists: a gates file names no hardware anywhere, and
 * every latency and every load time in it is one GPU's.
 */
function experimentScope(slate: SlateAxes, scoring: ArmScoringBoundary): string {
  const models = [...slate.models];
  const families = [...slate.families];
  return (
    `SCOPE OF THIS RUN, against spec 6.3's matrix. This run crosses the MODEL axis at ` +
    `${models.length} point(s) (${models.join(", ")}) and the METHOD axis at ` +
    `${families.length} point(s) (${families.map((f) => FAMILY_PROSE[f]).join(", ")}). ` +
    "It holds three axes at a single point and cannot cross them: the BACKEND axis, " +
    "because tier 2 is WebGPU or absent and this driver refuses to run without it, so no WASM " +
    "comparison exists at this tier; the POLICY axis, because it refuses an IR with no semantic " +
    "predicate and one of the three policy documents has a compiled artifact; and the HARDWARE " +
    "axis, because a run is one machine and one GPU and no field of this file names either. " +
    "Against the project's research question -- 'how well can policy-conditioned, fully-local " +
    "models prevent confidential-data leakage in LLM prompts, and at what latency/hardware cost?' " +
    "-- a run answers this much: the models are policy-conditioned (at one policy) and fully local " +
    "(every arm runs in the page and no prompt leaves the browser); the latency cost is per-call " +
    "TTFT and decode rate on one machine, at the message budget in run.latencyBudgetMs, which " +
    "run.latencyBudgetTimesCompilerDefault compares with the budget the compiler emits for a " +
    "policy that names none; and the hardware cost is engineLoadMs, engineWarmupMs and " +
    "originStorageBytes on that one GPU. It answers NOTHING about 'how well ... prevent leakage': " +
    "no leak-prevention rate, over-blocking rate, span P/R/F1, policy-adaptivity delta or utility " +
    "number is computed anywhere in this repository, and none can be from this run, because there " +
    "is one policy and because " +
    (scoring.cannotScore.length === 0
      ? "scoring.cannotScore on this row is empty, so the only thing missing is the scorer"
      : `scoring.cannotScore on this row names ${String(scoring.cannotScore.length)} way(s) ` +
        `these rows cannot be joined to gold`) +
    "."
  );
}

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
  // The mirror, and it is a SEPARATE filter rather than a negation of the one
  // above because the two describe different damage: `unscorable` is findings
  // with nothing to match, this is gold with nothing that could have matched
  // it. Both can be non-empty on the same arm -- they are on every arm of a
  // bake-off run against the shipped corpus, [2] and [1] respectively.
  const goldNotRun = tiers.filter((t) => !t.ran && t.goldSpans > 0).map((t) => t.tier);
  const goldSpansTotal = TIERS.reduce<number>((sum, t) => sum + counts[t], 0);
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
  for (const tier of goldNotRun) {
    const coverage = tiers[tier]!;
    cannotScore.push(
      `the ${records.length} row(s) here carry ${coverage.goldSpans} gold span(s) at tier ` +
        `${tier} (${coverage.goldEntityTypes.join(", ")}) and this arm did NOT run tier ${tier}, ` +
        `so nothing it produced can match them. That is a RECALL DENOMINATOR this arm is ` +
        `structurally incapable of filling: a recall taken over record.gold is bounded above by ` +
        `${goldSpansTotal - coverage.goldSpans}/${goldSpansTotal} here whatever the model does, ` +
        `and an otherwise identical run with tier ${tier} switched on is not comparable with ` +
        `this one. Score per tier, or restrict the denominator to tiersRun.`,
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
    tiersTheseRowsCannotScore: unscorable,
    tiersWithGoldThisArmDidNotRun: goldNotRun,
    goldEntityTypesNotInIr,
    goldPolicies,
    cannotScore,
    verdictMeans: VERDICT_MEANS,
  };
}

/**
 * One row's model-call counters, in ONE shape whichever method produced them.
 *
 * The translation layer `assertPageCanRun` used to refuse to write, now written
 * in the one place a translation belongs: at the boundary, once, with the
 * mapping spelled out. Every gate below is then a single expression over both
 * methods rather than two expressions that could drift on a threshold.
 *
 * The mapping, and it is a mapping and not a rename of convenience:
 *
 *   segmentsJudged
 *     + messageScopeJudged <-> messagesJudged        (the unit ONE call covers)
 *   unknownPredicates      <-> unknownEntityTypes    (a model inventing a label)
 *   segmentsSkipped        <-> (none)                (no second unit to skip)
 *   (none)                 <-> messageBudgetExpiries (only B owns its own message clock)
 *
 * The first row is a SUM on the compiled side and not a rename, because the
 * compiled judge has two kinds of judged unit and B has one. See `unitsJudged`
 * below.
 *
 * `stops` folds the three stop counters, which is what the engine-poisoning walk
 * needs and all it needs: the walk asks WHETHER this item ended in a stop, and
 * the three are already reported separately on `ladder`. Folding them here is
 * what keeps that walk from having to know which method it is walking.
 *
 * Returns `undefined` for a row with no counters at all -- an errored item,
 * whose delta belongs to the previous item and whose absence `RunRecordSchema`
 * requires.
 */
function normalizeArmStats(record: RunRecord):
  | {
      readonly rung1: number;
      readonly rung2: number;
      readonly unresolvedQuotes: number;
      readonly unresolvedMentions: number;
      readonly wholeClauseMentions: number;
      readonly duplicatesDropped: number;
      readonly unknownLabels: number;
      readonly failedClosed: number;
      readonly truncatedResponses: number;
      readonly abortedResponses: number;
      readonly repairAttempts: number;
      readonly unitsJudged: number;
      readonly unitsSkipped: number | undefined;
      readonly messageBudgetExpiries: number | undefined;
      readonly messageScopeCalls: number | undefined;
      readonly messageScopeJudged: number | undefined;
      readonly messageScopeFailedClosed: number | undefined;
      readonly stops: number;
      readonly calls: NonNullable<RunRecord["tier2Stats"]>["calls"];
    }
  | undefined {
  const judge = record.tier2Stats;
  if (judge !== undefined) {
    return {
      rung1: judge.rung1,
      rung2: judge.rung2,
      unresolvedQuotes: judge.unresolvedQuotes,
      unresolvedMentions: judge.unresolvedMentions,
      wholeClauseMentions: judge.wholeClauseMentions,
      duplicatesDropped: judge.duplicatesDropped,
      unknownLabels: judge.unknownPredicates,
      failedClosed: judge.failedClosed,
      truncatedResponses: judge.truncatedResponses,
      abortedResponses: judge.abortedResponses,
      repairAttempts: judge.repairAttempts,
      // BOTH scopes' collected calls, because both are judged units on this
      // arm and `judgedUnit` says which. `segmentsJudged` alone was the whole
      // count while the unit was a per-family constant, and on a message-only
      // policy it reads 0 for an arm that judged every message -- a row saying
      // `judgedUnit: "message"` beside `unitsJudged: 0` while its planned
      // distribution held one unit per item. The message half stays visible on
      // its own as `messageScopeJudged`; this is the total, and the two
      // counters are parallel (each counts a call whose answer was collected,
      // and neither counts a failed-closed one).
      unitsJudged: judge.segmentsJudged + judge.messageScopeJudged,
      unitsSkipped: judge.segmentsSkipped,
      messageBudgetExpiries: undefined,
      messageScopeCalls: judge.messageScopeCalls,
      messageScopeJudged: judge.messageScopeJudged,
      messageScopeFailedClosed: judge.messageScopeFailedClosed,
      stops:
        judge.deadlineExpiries +
        judge.callerAbortsMidGeneration +
        judge.callerAbortsWhileQueued,
      calls: judge.calls,
    };
  }
  const baseline = record.baselineStats;
  if (baseline === undefined) return undefined;
  return {
    rung1: baseline.rung1,
    rung2: baseline.rung2,
    unresolvedQuotes: baseline.unresolvedQuotes,
    unresolvedMentions: baseline.unresolvedMentions,
    wholeClauseMentions: baseline.wholeClauseMentions,
    duplicatesDropped: baseline.duplicatesDropped,
    unknownLabels: baseline.unknownEntityTypes,
    failedClosed: baseline.failedClosed,
    truncatedResponses: baseline.truncatedResponses,
    abortedResponses: baseline.abortedResponses,
    repairAttempts: baseline.repairAttempts,
    unitsJudged: baseline.messagesJudged,
    unitsSkipped: undefined,
    messageBudgetExpiries: baseline.messageBudgetExpiries,
    // B's single call IS its message call; a separate message-scope column for
    // it would double-count the same call under two names.
    messageScopeCalls: undefined,
    messageScopeJudged: undefined,
    messageScopeFailedClosed: undefined,
    // `messageBudgetExpiries` is IN the fold and `deadlineExpiries` is too, and
    // both belong: each is a stop that ended the run for that message, and the
    // poisoning walk's question is whether the engine was interrupted before
    // the next call. `baselineB.ts`'s loop `break`s on either.
    stops:
      baseline.deadlineExpiries +
      baseline.messageBudgetExpiries +
      baseline.callerAbortsMidGeneration +
      baseline.callerAbortsWhileQueued,
    calls: baseline.calls,
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
  // Computed BEFORE `run` because `experimentScope` reads it: the scope
  // sentence's accuracy clause points at this row's own `cannotScore` rather
  // than restating a corpus property, which is how the two stopped being able
  // to contradict each other on one row.
  const scoring = scoringBoundary(records, input.entityTypes);
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

  // `judgedUnitChars`, `judgedUnitsPerItem` and `escalation` are the only fields on
  // this report that do not come off the rows: they are the PLAN's distribution,
  // measured in Node before the arm ran. That is the closest population there
  // is -- no record carries the segments the page judged -- and it is only
  // honest while the distribution describes THIS arm's work. Three ways it can
  // stop doing so, all of which produce a perfectly well-formed report:
  //
  //   - the wrong UNIT for this policy. A compiled family's unit is not the
  //     family's alone: on `policies/compiled/p-fin.ir.json`, whose one
  //     predicate is message-scoped, a compiled arm is shown whole messages and
  //     makes no segment call at all. A segment distribution there reports a
  //     p50 of 45 characters and up to 3 units an item for an arm whose calls
  //     were 110-character messages, one an item -- and it is the column a
  //     reader consults to check the p95 TTFT ceiling was applied to comparable
  //     work.
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
  // The family the caller NAMED against what the rows say ran. `detector` is a
  // record field the page's own routing produced -- `loadBaseline` is the only
  // thing that makes `detect` answer from Approach B -- so this is a check on
  // fact, not on intent, and it is the one that stops a B arm's numbers being
  // filed under a compiled family. `uniform` refuses a set of rows that
  // disagree with each other, which on this field would be an arm that changed
  // method partway through.
  const detector = uniform(records, "detector", (r) => r.detector);
  if ((detector === "core-orchestrator") !== shape.runsCompiledJudge) {
    throw new Error(
      `arm "${arm}" is family "${family}", which runs ` +
        `${shape.runsCompiledJudge ? "core's orchestrator with the compiled judge" : "Approach B"}, ` +
        `but its rows say the detector was "${detector}". The gates below are the same numbers ` +
        `either way and would look perfectly well-formed; what would be wrong is which method ` +
        `they are attributed to, which is the only question this bake-off exists to answer`,
    );
  }
  // The unit the FAMILY and the POLICY together make this arm's, against the
  // unit its distribution was measured over. Both halves matter and each has
  // caught a different swap: the family half stops a B arm's row carrying the
  // compiled families' segment distribution, and the policy half stops a
  // compiled arm on a message-only policy carrying one -- which is what every
  // compiled row of the head-to-head did while the unit was a per-family
  // constant, reporting a segment p50 of 45 characters for calls made on whole
  // messages of 110.
  const judgedUnit = judgedUnitFor(family, input.semanticPredicates);
  if (segments.unit !== judgedUnit) {
    throw new Error(
      `arm "${arm}" is family "${family}" and, under the ` +
        `${input.semanticPredicates.length} semantic predicate(s) this policy declares, its ` +
        `model is shown one ${judgedUnit} per engine call, but its size distribution was ` +
        `measured over ${segments.unit}s; judgedUnitChars and judgedUnitsPerItem would describe ` +
        `prompts other than this arm's, and the p95 TTFT gate's stated prompt size with them`,
    );
  }
  if (segments.escalation.hasPriors !== shape.runsTier0) {
    throw new Error(
      `arm "${arm}" is family "${family}", which ${shape.runsTier0 ? "runs" : "does not run"} ` +
        `tier 0, but its segment distribution was measured ` +
        `${segments.escalation.hasPriors ? "WITH" : "WITHOUT"} tier-0 priors; judgedUnitChars, ` +
        `judgedUnitsPerItem and escalation would describe the other family's work under this ` +
        `arm's ` +
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
    unresolvedMentions: 0,
    wholeClauseMentions: 0,
    duplicatesDropped: 0,
    unknownLabels: 0,
    failedClosed: 0,
    truncatedResponses: 0,
    abortedResponses: 0,
    repairAttempts: 0,
    unitsJudged: 0,
    // Accumulated as numbers and reported as `undefined` on the family that has
    // no such counter -- see `ArmGateReport.ladder`. A 0 there would be the
    // positive claim that the event happened zero times on an arm that cannot
    // produce it.
    unitsSkipped: 0,
    messageBudgetExpiries: 0,
    messageScopeCalls: 0,
    messageScopeJudged: 0,
    messageScopeFailedClosed: 0,
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

    // Exactly one of the two is populated on a returned row, and which one is
    // decided by `detector` -- `RunRecordSchema` refuses a row carrying both or
    // neither. Normalised into ONE shape here rather than summed into two
    // parallel sets of counters, so every gate below is computed by one
    // expression over both methods instead of two that could drift.
    const stats = normalizeArmStats(record);
    if (stats === undefined) continue;
    ladder.rung1 += stats.rung1;
    ladder.rung2 += stats.rung2;
    ladder.unresolvedQuotes += stats.unresolvedQuotes;
    ladder.unresolvedMentions += stats.unresolvedMentions;
    ladder.wholeClauseMentions += stats.wholeClauseMentions;
    ladder.duplicatesDropped += stats.duplicatesDropped;
    ladder.unknownLabels += stats.unknownLabels;
    ladder.failedClosed += stats.failedClosed;
    ladder.truncatedResponses += stats.truncatedResponses;
    ladder.abortedResponses += stats.abortedResponses;
    ladder.repairAttempts += stats.repairAttempts;
    ladder.unitsJudged += stats.unitsJudged;
    ladder.unitsSkipped += stats.unitsSkipped ?? 0;
    ladder.messageBudgetExpiries += stats.messageBudgetExpiries ?? 0;
    ladder.messageScopeCalls += stats.messageScopeCalls ?? 0;
    ladder.messageScopeJudged += stats.messageScopeJudged ?? 0;
    ladder.messageScopeFailedClosed += stats.messageScopeFailedClosed ?? 0;

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
    if (stoppedAt === undefined && stats.stops > 0) stoppedAt = record.itemId;
  }

  const sustainedDecodeTokPerSec = decodeSeconds > 0 ? decodedTokens / decodeSeconds : undefined;
  // EVERY quote the ladder was handed, which is the population both of these
  // rates are named over -- and `duplicatesDropped` belongs in it. READ from
  // `WebLlmJudge.#collect`, which runs `locateFinding` BEFORE the duplicate
  // check: a finding that did not place `continue`s at one of the two unplaced
  // counters, a duplicate `continue`s after it, and only the survivors reach
  // `rung1` or `rung2`. So a duplicate is a finding that DID place, and leaving
  // it out of the resolvable rate omitted it from both halves.
  //
  // What that cost, by arithmetic on the old expression: an arm on the shape
  // Plan 5 says to expect -- one distinct span, seven restatements of it, two
  // quotes the ladder refused, so 8 of 10 quotes placed -- reported
  // `(1 + 0) / (1 + 0 + 2)`, a rate of 0.333, and was killed. An arm whose span
  // ladder works but whose model restates itself was being reported as an arm
  // whose quotes do not resolve. Those are different diagnoses: a model that
  // restates itself is what `duplicate-rate` is for, and a span-recovery
  // failure is what this gate is for.
  //
  // `unresolvedMentions` belongs in the denominator for the same reason and is
  // NOT in the numerator. Since the arms started returning two spans per
  // finding, a finding can be lost at either placement: the clause is not in
  // the text (`unresolvedQuotes`), or the clause placed and the mention did not
  // (`unresolvedMentions`). Both are findings the model produced that reached
  // no span, which is exactly what this rate is over. Leaving the second out
  // would let an arm whose every mention fails report a rate of 1.000 while
  // emitting nothing -- the gate reading as a clean bill of health for the arm
  // it exists to catch.
  const quotesResolved = ladder.rung1 + ladder.rung2 + ladder.duplicatesDropped;
  const findingsUnplaced = ladder.unresolvedQuotes + ladder.unresolvedMentions;
  const resolvableDenominator = quotesResolved + findingsUnplaced;
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
        `against a ${GATES.maxP95TtftMs}ms ceiling. ` +
        // AT THIS SAMPLE SIZE THE p95 IS THE MAXIMUM, said where the number is
        // read rather than only in `GATES.maxP95TtftMs`'s docblock. `percentile`
        // is nearest-rank, `ceil(0.95 * n)` equals n for every n <= 19, and this
        // corpus produces at most 18 calls per compiled arm and 13 per B arm --
        // so on it this gate is applied to the arm's slowest call and one slow
        // call kills the arm. The condition is on the SAMPLE and not on the
        // corpus, so a run big enough for the two to differ stops saying it.
        (ttft.length < P95_EQUALS_MAX_BELOW
          ? `Over ${ttft.length} call(s) the nearest-rank p95 IS THE MAXIMUM -- ` +
            `ceil(0.95*n) = n for every n < ${P95_EQUALS_MAX_BELOW} -- so this is the slowest ` +
            `call this arm made, not a number with its tail trimmed. `
          : "") +
        `Read it beside promptTokens and judgedUnitChars, because the ceiling was derived at a ` +
        `~1.1 kB prompt built from ONE SEGMENT` +
        // CONDITIONAL on the arm's own unit, which is the family AND the
        // policy. THREE cases and not two, because the caveat is only wholly
        // true for one of them. On `"message"` -- a B arm, or a compiled arm on
        // a message-only policy -- none of the sample is the prompt shape the
        // ceiling was taken at. On `"segment+message"` HALF of it is: that arm
        // makes one whole-message call and one call per selected segment, and
        // the segment calls are exactly the system turn plus one segment the
        // ~1.1 kB was measured on. Saying "that is not the prompt size" there
        // would send a reader past the half of the column this ceiling can
        // legitimately be read against.
        //
        // Each branch spells its unit out rather than interpolating
        // `segments.unit`, because an interpolation is only ever exercised at
        // the one value that reaches it and reads as coverage of all three.
        (segments.unit === "segment"
          ? ``
          : segments.unit === "message"
            ? `, and this arm is judged per MESSAGE, so that is not the prompt size this ` +
              `number was taken at`
            : `, and this arm is judged per SEGMENT+MESSAGE, so half of these calls ARE that ` +
              `prompt shape and half are a whole message: this ceiling is comparable for the ` +
              `segment half of the sample and not for the rest`) +
        `; and beside run.latencyBudgetTimesCompilerDefault, because these calls were ` +
        `${budgetTaken}` +
        // CONDITIONAL, because the clause is only true above 1x. A run against
        // `policies/compiled/p-fin.ir.json` is AT the compiler's default, so
        // its sample IS the set a compiled policy's budget produces and the
        // caveat would be a false qualification on the more transferable of the
        // two conditions. `test/baseline.spec.ts` runs exactly that.
        (run.latencyBudgetTimesCompilerDefault === 1
          ? ` -- that IS a compiled policy's own default budget, so this sample is the set of ` +
            `calls such a policy produces`
          : ` -- so the SET of calls in this sample is not the set a compiled policy's budget ` +
            `would produce, and the degradation that difference causes is measured nowhere in ` +
            `this run`),
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
      // A floor over successes: one unplaceable quote in a sample of n is a
      // rate of (n - 1) / n. At the shipped 0.8 that makes the minimum 5, which
      // is the sample size the two Ministral arms fell four short of.
      oneContraryAt: (n) => (n - 1) / n,
      notMeasured: "this arm produced no quote for the span ladder to place, resolvable or not",
      measured: (observed) =>
        `${quotesResolved} of ${resolvableDenominator} finding(s) resolved to an action span ` +
        `(rung 1: ${ladder.rung1}, rung 2: ${ladder.rung2}, dropped as a duplicate of a span ` +
        `already emitted: ${ladder.duplicatesDropped}), a rate of ${observed.toFixed(3)} ` +
        `against a floor of ${GATES.minResolvableRate}. ` +
        // WHICH placement failed, because the two call for different responses:
        // an unplaceable clause is a model quoting text that is not there, and
        // an unplaceable mention is a model quoting correctly and then pointing
        // outside its own quote or at a name its clause repeats.
        `Of the ${findingsUnplaced} that did not, ${ladder.unresolvedQuotes} had no placeable ` +
        `clause and ${ladder.unresolvedMentions} had a clause but no placeable mention. ` +
        `${ladder.wholeClauseMentions} of the resolved ones cover their whole clause`,
    }),
    numericGate({
      gate: "duplicate-rate",
      threshold: GATES.maxDuplicateRate,
      sample: duplicateDenominator,
      observed: duplicateDenominator === 0 ? undefined : ladder.duplicatesDropped / duplicateDenominator,
      passes: (observed) => observed <= GATES.maxDuplicateRate,
      // A ceiling over failures, so the contrary observation is the duplicate
      // itself: one restatement in a sample of n is a rate of 1 / n. At the
      // shipped 0.9 that makes the minimum 2 -- a far weaker minimum than
      // resolvable-rate's, and honestly so, because a ceiling that permissive
      // is crossed by one observation only at a sample of one.
      oneContraryAt: (n) => 1 / n,
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
    scoring,
    experimentScope: experimentScope(input.slate, scoring),
    engineLoadMs: input.load.engineLoadMs,
    engineWarmupMs: input.load.engineWarmupMs,
    originStorageBytes: input.load.originStorageBytes,
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
    ladder: {
      ...ladder,
      // Reported as `undefined` on the family that cannot produce the event.
      // See `ArmGateReport.ladder`: a 0 is a measurement and these two are not
      // measurable on the other method.
      // Off the UNIT and not off the family: the counter is the judge's
      // `segmentsSkipped`, and an arm with no segment loop -- Approach B, or a
      // compiled arm on a policy declaring only message-scoped predicates --
      // has no second unit for a stop to skip past. `WebLlmJudge` computes
      // `segmentCallsOwed` as 0 when no predicate is segment-scoped, so a 0
      // here would be the positive claim that such an arm skipped none.
      unitsSkipped: segments.unit === "message" ? undefined : ladder.unitsSkipped,
      messageBudgetExpiries: shape.runsCompiledJudge ? undefined : ladder.messageBudgetExpiries,
      messageScopeCalls: shape.runsCompiledJudge ? ladder.messageScopeCalls : undefined,
      messageScopeJudged: shape.runsCompiledJudge ? ladder.messageScopeJudged : undefined,
      messageScopeFailedClosed: shape.runsCompiledJudge
        ? ladder.messageScopeFailedClosed
        : undefined,
    },
    degradedNotices,
    degradedItems,
    // The DERIVED unit, not `segments.unit`. The guard above proves the two
    // equal on every path that reaches here, so this is not a behaviour change;
    // it is the row being populated from the thing its docblock names, so that
    // weakening or reordering that guard cannot silently republish whatever
    // unit a caller's distribution happened to carry.
    judgedUnit,
    judgedUnitChars: segments.chars,
    judgedUnitsPerItem: segments.perItem,
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
  /**
   * The rate a sample of `n` with exactly ONE contrary observation produces, on
   * a gate whose denominator the arm itself chooses. Supplying it is what makes
   * `minSample` derived from this gate's threshold; omitting it leaves the
   * minimum at 1, which `GateOutcome.minSample` argues for the two per-call
   * gates.
   */
  oneContraryAt?: (n: number) => number;
  notMeasured: string;
  measured: (observed: number) => string;
}): GateOutcome {
  // Derived from THIS spec's own `passes`, so the boundary the minimum is
  // computed at and the boundary the verdict is decided at are one comparator.
  // A second copy taking the threshold constant would be free to disagree with
  // it about `>=` versus `>`, which is the whole of the difference at n = 5.
  const minSample =
    spec.oneContraryAt === undefined
      ? 1
      : minSampleForOneContrary({ passes: spec.passes, oneContraryAt: spec.oneContraryAt });
  if (spec.observed === undefined) {
    return {
      gate: spec.gate,
      threshold: spec.threshold,
      observed: undefined,
      sample: spec.sample,
      minSample,
      verdict: "not-measured",
      detail: spec.notMeasured,
    };
  }
  if (spec.sample < minSample) {
    return {
      gate: spec.gate,
      threshold: spec.threshold,
      // KEPT, unlike the branch above: there is a real observation here and
      // withholding it as well would throw away the only evidence a reader has.
      observed: spec.observed,
      sample: spec.sample,
      minSample,
      verdict: "not-measured",
      detail:
        `${spec.measured(spec.observed)} -- AND THIS GATE HAS NOT RULED ON IT. The sample is ` +
        `${spec.sample} and this gate rules only at ${minSample} or more, because below ` +
        `${minSample} a single contrary observation crosses this threshold on its own: a verdict ` +
        `here would be about that one observation and not about this arm, while the threshold ` +
        `itself has slack in it and is not "no bad observations". The value above is the ` +
        `measurement; the pass/fail is withheld, and killedOnRunGates does not see this gate.`,
    };
  }
  return {
    gate: spec.gate,
    threshold: spec.threshold,
    observed: spec.observed,
    sample: spec.sample,
    minSample,
    verdict: spec.passes(spec.observed) ? "pass" : "fail",
    detail: spec.measured(spec.observed),
  };
}

/**
 * `non-empty-after-stop`'s minimum sample, which is 1 for a reason the two rate
 * gates do not share.
 *
 * This gate is a PREDICATE over one specific event -- the first engine call
 * after the first stop -- and there is no second observation to be had: a
 * latched engine answers instantly and emptily forever, so the first call after
 * the stop either shows it or the engine was not latched. One is the whole
 * population, not a small sample of a larger one, and
 * `minSampleForOneContrary` has no rate to solve against here.
 */
const STOP_GATE_MIN_SAMPLE = 1;

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
      minSample: STOP_GATE_MIN_SAMPLE,
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
      minSample: STOP_GATE_MIN_SAMPLE,
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
      minSample: STOP_GATE_MIN_SAMPLE,
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
    minSample: STOP_GATE_MIN_SAMPLE,
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
 * What the harness needs from a plan before it will run one, and what it USED to
 * refuse.
 *
 * This function refused every Approach-B arm outright for three reasons, and
 * the record of them is worth keeping because each was true and each is now
 * closed by something checkable rather than by an assurance:
 *
 *   1. NO PAGE DOOR. `apps/eval/src/page/main.ts` published only
 *      `window.__sih.detect`, core's orchestrator with a `WebLlmJudge` behind
 *      it, and never constructed `createBaselineB` -- so no call could reach an
 *      Approach-B arm at all. Closed by `loadBaseline`, which builds the arm on
 *      the SAME engine the compiled judge runs on and makes `detect` answer
 *      from it.
 *   2. NO RECORD FIELD. `RunRecordSchema` had `tier2Stats` and no baseline
 *      equivalent, and B's counters are `BaselineStats`: two events renamed
 *      (`messagesJudged`, `unknownEntityTypes`) and one the judge does not have
 *      (`messageBudgetExpiries`). Writing them into `tier2Stats` would have been
 *      a record stating one event under another event's name. Closed by
 *      `baselineStats` plus the `detector` field that says which of the two a
 *      row carries.
 *   3. NO COMPILED IR. Every IR here was hand-written with
 *      `policyHash: "test-hash"`, which is not a sha256 of anything, so
 *      `planBakeoff`'s pairing check could not be satisfied by any document in
 *      the repository and every B arm was refused one layer up. Closed by
 *      `policies/compiled/p-fin.ir.json`, real compiler output whose
 *      `policyHash` is `shasum -a 256 policies/p-fin.md`, produced offline by
 *      `scripts/compile-policies.ts` replaying the committed LLM fixtures and
 *      re-derived on every suite run by `packages/compiler/test/compiled.test.ts`.
 *
 * A FOURTH reason stood here before those and no longer does, kept for the same
 * reason: `semantic-ir.json` carried `rules: []`, so tier 0 found nothing on any
 * item and `baseline-b-tier0` versus `baseline-b` -- the pair that separates
 * "compiling helps" from "patterns help" -- would have been two runs of the same
 * thing. That fixture now carries three tier-0 rules (MEASURED: 18 selected
 * segments against 17, per-message max 3 against 2), and the compiled p-fin IR
 * carries ten.
 *
 * What is left to check is the ONE thing a plan can still get wrong here: a
 * baseline arm needs a policy document to show its model, and `planBakeoff`
 * only requires one when a baseline family is present. This makes the pairing a
 * property of the plan rather than of the caller's memory.
 */
export function assertPageCanRun(plan: BakeoffPlan): void {
  const baseline = plan.arms.filter((a) => !familyShape(a.family).runsCompiledJudge);
  if (baseline.length === 0) return;
  if (plan.policy === undefined) {
    throw new Error(
      `this plan holds Approach-B arm(s) [${baseline.map((a) => a.arm).join(", ")}] and no policy ` +
        `document. B is the whole policy in one prompt with no compiler, so without the document ` +
        `there is nothing for it to be shown`,
    );
  }
}

// ---------------------------------------------------------------------------
// The entry point
// ---------------------------------------------------------------------------

/** `<repo>`, from `<repo>/apps/eval/src/driver/bakeoff.ts`. */
function repoRoot(): string {
  return join(import.meta.dirname, "..", "..", "..", "..");
}

/** `<repo>/apps/eval/fixtures`, where three of the page's four IRs live. */
function fixturesDir(): string {
  return join(import.meta.dirname, "..", "..", "fixtures");
}

/**
 * IR registry names whose file is NOT at `fixtures/<name>-ir.json`, and where
 * it is instead.
 *
 * ONE entry, and the reason there is a table at all rather than a second copy
 * of the page's registry: the page's `IR_FIXTURES` maps a name to BYTES through
 * a `?raw` import, and Vite inlines those at build time, so nothing this driver
 * can ask the page returns a path. Three of the four names happen to follow the
 * fixtures convention and `p-fin` -- real compiler output, which belongs beside
 * the document it was compiled from -- does not. Without this line the ONE
 * invocation the page's registry advertises, `SIH_BAKEOFF_IR=p-fin`, resolved
 * `apps/eval/fixtures/p-fin-ir.json` and died four layers down inside
 * `readFileSync` with a bare ENOENT naming a path this repository has never had.
 *
 * What keeps it from being a stale second definition is not care, it is the
 * check that already runs: `runBakeoff` takes its own sha256 of whatever this
 * resolves and refuses the run unless the page's `useIr(name)` returns the same
 * digest. So a table entry pointing at the wrong bytes is a refusal naming both
 * paths and both hashes, before a model loads -- and an entry pointing at
 * nothing at all is the refusal in `resolveIrPath` below.
 */
const IR_PATHS_OFF_CONVENTION: Readonly<Record<string, readonly string[]>> = {
  "p-fin": ["policies", "compiled", "p-fin.ir.json"],
};

/**
 * The file this driver will read for an IR, or a refusal that names the cause.
 *
 * ## Why this is a function and why it can refuse
 *
 * `irName` is the PAGE's registry key and `irPath` is a path on THIS process's
 * filesystem, and the two are related only by a convention that holds for three
 * of the four names the page serves. A name the page accepts is therefore not
 * evidence that this driver can find the bytes, and until this function existed
 * the gap was a `readFileSync` ENOENT raised from inside `runBakeoff` about a
 * filename the caller had never typed.
 *
 * The order is: an explicit `irPath` (a caller who named a file is not asking
 * to be second-guessed, and it is how `test/baseline.spec.ts` selects p-fin),
 * then the off-convention table, then `fixtures/<name>-ir.json`. Every
 * candidate is checked for existence, so the refusal below is reached by a
 * stale table entry as well as by an unknown name.
 *
 * An explicit `irPath` is NOT existence-checked here on purpose: a caller who
 * passed a path gets an ENOENT naming the path they passed, which already names
 * its own cause. It is the resolved-by-convention case that produced a filename
 * out of nowhere.
 */
export function resolveIrPath(options: Pick<BakeoffOptions, "irName" | "irPath">): string {
  if (options.irPath !== undefined && options.irPath !== "") return options.irPath;
  const irName = options.irName ?? DEFAULT_IR_NAME;
  const offConvention = IR_PATHS_OFF_CONVENTION[irName];
  const candidates =
    offConvention === undefined
      ? [join(fixturesDir(), `${irName}-ir.json`)]
      : [join(repoRoot(), ...offConvention), join(fixturesDir(), `${irName}-ir.json`)];
  for (const candidate of candidates) if (existsSync(candidate)) return candidate;
  throw new Error(
    `no IR file for "${irName}": this driver looked at ${candidates.join(" and ")} and found ` +
      `no file. The page's IR registry and this driver's filesystem are two different ` +
      `things -- the page serves an IR by name out of bytes Vite inlined, so useIr("${irName}") ` +
      `can succeed on a name whose file this process cannot find -- and only the driver reads a ` +
      `path. Pass irPath (SIH_BAKEOFF_IR_PATH, from \`pnpm -C apps/eval bakeoff\`) naming the ` +
      `file. An Approach-B family additionally needs policyPath (SIH_BAKEOFF_POLICY_PATH) ` +
      `naming the document the IR was compiled from; planBakeoff refuses that one separately.`,
  );
}

/**
 * The options a SLATE run uses, built from the environment.
 *
 * ## Why this exists, and it is not a convenience
 *
 * Until it did, `runBakeoff` had no caller outside the test suite and no command
 * ran it: the four-model bake-off this whole module is for had never been
 * executed, and the two things that HAVE run -- `bakeoff.spec.ts` (one model,
 * two items) and `baseline.spec.ts` (one model, four families, three items) --
 * are pipe-integrity checks that a reader can easily mistake for it. An
 * apparatus with no way to invoke it is an apparatus nobody can tell apart from
 * a finished experiment.
 *
 * ## The defaults, and what each is
 *
 * `models` defaults to every id in `TIER2_MODELS` and `families` is left unset,
 * so `planBakeoff` applies its own `["compiled"]` -- which together is exactly
 * the slate spec 4.2 amended to four arms. Everything else defaults to the only
 * artifact in the repository that fits: the 13-item smoke corpus and the
 * `semantic` IR.
 *
 * CORRECTED. This said `policies/compiled/p-fin.ir.json` "needs
 * `SIH_BAKEOFF_IR_PATH` and `SIH_BAKEOFF_POLICY_PATH` together", which was a
 * requirement stated here and enforced nowhere: `SIH_BAKEOFF_IR=p-fin` alone
 * was accepted by this function and then died inside `runBakeoff`'s
 * `readFileSync` on `apps/eval/fixtures/p-fin-ir.json`, a path that has never
 * existed. `resolveIrPath` now resolves that name, so `SIH_BAKEOFF_IR=p-fin` on
 * its own runs the compiled families against the compiled policy.
 * `SIH_BAKEOFF_POLICY_PATH` is needed only when a family is Approach B, which
 * is shown the DOCUMENT rather than the IR -- `planBakeoff` refuses that case
 * naming the document, and `assertPageCanRun` refuses the plan.
 *
 * `runId` is REQUIRED and has no default, deliberately. It names a measurement
 * and it is half of every output file's name; a generated one (a timestamp, say)
 * would make two runs of the same slate look like two different experiments, and
 * `runBakeoff` refuses to overwrite an existing file precisely so that a repeat
 * is a decision rather than an accident.
 *
 * `itemTimeoutMs` defaults to 250,000, which is above the 121,020 ms bound
 * `semantic-ir.json`'s 120,000 ms message budget and the page's 60,000 ms
 * per-call budget produce -- `assertItemTimeoutMs` re-derives that from the plan
 * and throws if a caller's number is under it, so this default is checked rather
 * than trusted. It exists to catch a wedge and is not a latency target.
 *
 * ## What it does NOT do
 *
 * It validates nothing itself, and that is the point: `planBakeoff` already
 * refuses an unknown model id (through `resolveTier2Config`), an unknown family
 * (`familyShape`), an unpaired policy document, an IR with no semantic predicate
 * and an `itemTimeoutMs` that is not a usable timer value -- all before a model
 * loads. A second copy of any of those checks here would be a second definition
 * free to drift from the one the run actually obeys.
 */
export function slateBakeoffOptions(env: Record<string, string | undefined>): BakeoffOptions {
  const runId = env["SIH_BAKEOFF_RUN_ID"];
  if (runId === undefined || runId === "") {
    throw new Error(
      "SIH_BAKEOFF_RUN_ID is required and names the measurement: it is half of every output " +
        "file's name, and runBakeoff refuses to overwrite a file an earlier run wrote, so a " +
        "generated id would turn a repeat of one experiment into two that look unrelated",
    );
  }
  const list = (name: string): readonly string[] | undefined => {
    const raw = env[name];
    if (raw === undefined || raw.trim() === "") return undefined;
    return raw.split(",").map((part) => part.trim()).filter((part) => part !== "");
  };
  const number = (name: string, fallback: number): number => {
    const raw = env[name];
    if (raw === undefined || raw.trim() === "") return fallback;
    // `Number` and not `parseInt`: `parseInt("250s")` is 250, and a truncated
    // timeout that still runs is worse than one that refuses. NaN falls through
    // to `assertItemTimeoutMs`, which names the field and the reason.
    return Number(raw);
  };
  const irPath = env["SIH_BAKEOFF_IR_PATH"];
  const policyPath = env["SIH_BAKEOFF_POLICY_PATH"];
  return {
    runId,
    outDir: env["SIH_BAKEOFF_OUT_DIR"] ?? join(repoRoot(), "runs"),
    corpus: env["SIH_BAKEOFF_CORPUS"] ?? join(repoRoot(), "corpora", "fixtures", "smoke.jsonl"),
    provider: env["SIH_BAKEOFF_PROVIDER"] ?? "claude",
    models: list("SIH_BAKEOFF_MODELS") ?? TIER2_MODELS.map((m) => m.id),
    // Omitted rather than defaulted, so `DEFAULT_FAMILIES` stays the one
    // definition of what a slate runs.
    ...(list("SIH_BAKEOFF_FAMILIES") === undefined
      ? {}
      : { families: list("SIH_BAKEOFF_FAMILIES") as readonly ArmFamily[] }),
    irName: env["SIH_BAKEOFF_IR"] ?? DEFAULT_IR_NAME,
    ...(irPath === undefined || irPath === "" ? {} : { irPath }),
    ...(policyPath === undefined || policyPath === "" ? {} : { policyPath }),
    itemTimeoutMs: number("SIH_BAKEOFF_ITEM_TIMEOUT_MS", 250_000),
  };
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
  const irName = options.irName ?? DEFAULT_IR_NAME;
  // Through `resolveIrPath` rather than inline, so the one invocation the
  // page's registry advertises and this driver could not satisfy --
  // `SIH_BAKEOFF_IR=p-fin` with no path -- resolves, and so a name with no file
  // anywhere refuses here naming the name, the paths tried and the option that
  // supplies the real one, instead of reaching the `readFileSync` below.
  const irPath = resolveIrPath(options);
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
  // The policy document's two-sided check, the twin of the IR one above and for
  // the same failure: this process reads a file off disk and the PAGE bundles
  // its own copy, and nothing else compares them. `playwright.config.ts` sets
  // `reuseExistingServer: !CI`, so a dev server from another worktree would
  // show Approach B a different document while every record looked correct --
  // and unlike the IR, no record field carries a digest of what B was shown.
  // (`policyHash` on a record is the IR's field, which is the hash of the
  // document the IR was COMPILED from, not of the text B was handed.) So this
  // is the only place the two can be tied together, and it happens before a
  // model loads.
  if (plan.policy !== undefined) {
    const pagePolicyHash = await page.evaluate(
      (name) => window.__sih!.policyDocHash(name),
      plan.policy.name,
    );
    if (pagePolicyHash !== plan.policy.sha256) {
      throw new Error(
        `the page's policy "${plan.policy.name}" hashes to ${pagePolicyHash} but ` +
          `${String(options.policyPath)} hashes to ${plan.policy.sha256}; Approach B would be ` +
          `shown a document this driver never read, and no field of any record would say so`,
      );
    }
  }

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

    // The Approach-B door, taken exactly when the arm's FAMILY says this arm is
    // not the compiled judge. Everything the arm runs under is already fixed by
    // the `loadTier2` above -- the engine, the window, the temperature, the
    // token ceiling and the per-call budget -- and `loadBaseline` reads all of
    // them off that load rather than taking any of them from here. What this
    // call adds is the policy DOCUMENT and which of the two B constructors to
    // use, and both come off the plan.
    const shape = familyShape(arm.family);
    if (!shape.runsCompiledJudge) {
      // Unreachable: `assertPageCanRun` refused this plan before any model
      // loaded. Kept because the alternative is passing `undefined` into the
      // page and getting an "unknown policy" from the far side of an evaluate.
      if (plan.policy === undefined) {
        throw new Error(`arm "${arm.arm}" is an Approach-B arm and this plan carries no policy`);
      }
      const baselineLoad = await page.evaluate(
        (loadOptions) => window.__sih!.loadBaseline(loadOptions),
        { family: shape.runsTier0 ? ("baseline-b-tier0" as const) : ("baseline-b" as const), policy: plan.policy.name },
      );
      // Three checks on the report, and like the tier-2 ones they are NOT of
      // equal strength. `irPolicyHash` is the strong one: the page compares its
      // own digest of the document it will put in the prompt against the IR it
      // has loaded, and refuses -- so this is reading back a refusal that has
      // already run. `policyDocSha256` is the echo that ties that refusal to
      // the file THIS process read. `servedModelId` is the same observed id the
      // tier-2 load reported, re-read here because B runs on that engine and a
      // reload between the two calls would otherwise go unnoticed.
      if (baselineLoad.policyDocSha256 !== plan.policy.sha256) {
        throw new Error(
          `arm "${arm.arm}": the page built its Approach-B arm on a document hashing ` +
            `${baselineLoad.policyDocSha256}, not the ${plan.policy.sha256} this driver planned ` +
            `from`,
        );
      }
      if (baselineLoad.irPolicyHash !== plan.ir.policyHash) {
        throw new Error(
          `arm "${arm.arm}": the page paired its Approach-B document with an IR whose policyHash ` +
            `is "${baselineLoad.irPolicyHash}", not this plan's "${plan.ir.policyHash}"`,
        );
      }
      if (baselineLoad.servedModelId !== arm.modelId) {
        throw new Error(
          `arm "${arm.arm}": the Approach-B arm was built on an engine answering as ` +
            `"${baselineLoad.servedModelId}" rather than ${arm.modelId}`,
        );
      }
      if (baselineLoad.callBudgetMs !== arm.callBudgetMs) {
        throw new Error(
          `arm "${arm.arm}": the Approach-B arm holds a ${baselineLoad.callBudgetMs}ms per-call ` +
            `budget and this driver asked for ${arm.callBudgetMs}; two arms at different per-call ` +
            `budgets measure the deadline rather than the method`,
        );
      }
    }

    const records = await runArm(page, {
      runId: options.runId,
      arm: arm.arm,
      // From the FAMILY, and it is the same value that decided whether
      // `loadBaseline` was called five lines up -- so the label on every row
      // and the code the page will run come from one expression rather than
      // two that could disagree.
      detector: shape.runsCompiledJudge ? "core-orchestrator" : "approach-b",
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
    const report = gateReport({
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
      // And the same artifact's declared scopes, which is what decides the unit
      // this arm's model was shown. Read from the IR rather than from
      // `arm.segments`, so the check below is against an independent source
      // rather than the distribution agreeing with itself.
      semanticPredicates: ir.semanticPredicates,
      // Off the PLAN and not off `options`: `options.families` is optional and
      // `DEFAULT_FAMILIES` fills it in, so reading the options would let the
      // default slate's row claim a method it never planned. `plan.arms` is
      // what was actually built.
      slate: {
        models: [...new Set(plan.arms.map((a) => a.modelId))],
        families: [...new Set(plan.arms.map((a) => a.family))],
      },
      // The hardware half of the research question, off the SAME load report
      // whose `servedModelId`, `contextWindowSize` and `callBudgetMs` were
      // checked above -- so these three numbers describe the engine that
      // produced the rows beside them, not a later or an earlier one. Before
      // this the page measured all three and the driver dropped them, and
      // neither output file could answer "at what hardware cost".
      load: {
        engineLoadMs: load.loadMs,
        engineWarmupMs: load.warmupMs,
        originStorageBytes: load.storageUsageBytes,
      },
    });
    reports.push(report);
    // APPENDED HERE, as the arm finishes, and this is a correction of a real
    // hazard rather than a tidier place to put it. The gates file used to be
    // written after the LAST arm, on the argument that "a run interrupted
    // partway leaves the JSONL files -- which spec 2.2 makes the whole boundary
    // -- and every verdict in this file is recomputable from them". That
    // argument was false and this file said so three times over:
    // `GateReportInput` documents `entityTypes` as "the fourth thing no record
    // carries", `itemTimeoutMs` and `latencyBudgetMs` as three of the first
    // three, and the load costs as the fifth. Without them `scoring.tiers`,
    // `scoring.goldSpansByTier` and `scoring.tiersTheseRowsCannotScore` cannot
    // be recomputed AT ALL.
    //
    // So a run that died on arm 2 left arm 1's complete, schema-valid, fully
    // scoreable JSONL on disk with no `scoring`, no `cannotScore` and no
    // `experimentScope` anywhere -- the silent zero-precision arm this module's
    // header calls the first of the two things absence must not look like, in
    // the one artifact that survives a partial run. Reachable from every throw
    // between here and the end: the next arm's four load-report refusals, its
    // four Approach-B refusals, its all-items-failed refusal, its invalid-record
    // refusal, `gateReport`'s own `uniform` refusals, and any `page.evaluate`
    // failure. Now every JSONL on disk has its row beside it.
    //
    // "ax" on the FIRST row and "a" after it, which is the race guard the
    // single write had: the pre-flight `existsSync` above catches a repeated
    // runId, and this catches a file created between that check and now -- an
    // hour later, on a machine where another process may have started the same
    // run. Plain "a" would silently append this run's rows to that one's.
    try {
      writeFileSync(plan.gatesPath, JSON.stringify(report) + "\n", {
        flag: reports.length === 1 ? "ax" : "a",
      });
    } catch (cause) {
      if ((cause as NodeJS.ErrnoException).code === "EEXIST") {
        throw new Error(`${plan.gatesPath} was created while this bake-off was running`, { cause });
      }
      throw cause;
    }

    // The arm is finished, so its engine is released HERE rather than left to
    // the next `page.goto`. `loadTier2` already unloads before replacing an arm
    // and its docblock gives the arithmetic -- two live engines hold two copies
    // of the weights and the two largest pinned arms are 3,432 and 3,438 MB --
    // but this loop never replaces an arm, it navigates past it, and no
    // measurement here says when Chrome releases a `GPUDevice` whose page has
    // gone away. What IS measured is the cost of assuming: a full suite run in
    // which a four-arm bake-off preceded the other tier-2 specs lost two of
    // them to `Execution context was destroyed, most likely because of a
    // navigation` inside `loadTier2`, and both passed when run on their own.
    // That is consistent with GPU memory pressure and is not proof of it, so
    // this is an explicit release and not a fix with a mechanism attached.
    //
    // AFTER the file is written and the report computed, so a failure to
    // release cannot cost an arm that has already been measured.
    await page.evaluate(() => window.__sih!.unloadTier2());
  }

  // Nothing to write here any more; every row landed with its arm. A run with
  // zero arms cannot reach this point -- `planBakeoff` refuses an empty model
  // list and an empty family list -- so the file always exists by now.
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
