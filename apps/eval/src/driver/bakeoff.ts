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
 */

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

export const GATES = {
  /**
   * p95 time-to-first-token at the MEASURED tier-2 segment size.
   *
   * Task 9 measured that size over `corpora/fixtures/smoke.jsonl`, the only
   * corpus in this repository, keeping the 17 of 19 segments the escalation
   * policy selects: p50 62 characters, max 153, and 9 words at the median. Plan
   * 5 then reports -- ITS measurement, not one taken here, and not reproducible
   * from this package because `buildMessages` is private to `judge.ts` -- that
   * assembling those through the judge's prompt gives a WHOLE PROMPT of 1,105
   * characters at the median segment and 1,196 at the largest. So a TTFT
   * measured against a prompt materially bigger than ~1.2 kB is not measuring
   * this gate, and an arm must not be killed on a number taken at a different
   * prompt size.
   *
   * MEASURED HERE, through `test/bakeoff.spec.ts` on Qwen3.5-2B over a two-item
   * slice of that corpus: the engine reported 245-256 prompt TOKENS for segments
   * of 28-48 characters. At ~4.5 characters per token that is a prompt of
   * roughly 1.1 kB, which is consistent with Plan 5's figure -- and it shows
   * where the size comes from: the prompt is dominated by the fixed instructions
   * and the predicate, not by the segment, so a 48-character segment and a
   * 153-character one differ by far less than 3x.
   *
   * This module cannot enforce that comparison and does not pretend to: the
   * prompt is assembled inside the browser and a record carries no copy of it.
   * What it does instead is put the size BESIDE the number, twice over --
   * `ArmGateReport.promptTokens` is the engine's own prompt-token count over
   * exactly the calls the p95 was taken over, and `ArmGateReport.segmentChars`
   * is the character-size distribution of the segments the arm ran, which is the
   * unit the 1.2 kB above is quoted in. A reader who finds either one out of
   * line with the derivation knows the gate was applied to different work.
   */
  maxP95TtftMs: 1500,
  /** Sustained decode rate. Latency here is dominated by output length. */
  minDecodeTokPerSec: 25,
  /** Semantic correctness: fraction of findings resolving at rung <= 2. */
  minResolvableRate: 0.8,
  /** An arm returning only duplicates has found nothing. */
  maxDuplicateRate: 0.5,
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
 * 2 segments per message under that IR, so (b) is 4 x 60,020 = 240,080 and (a)
 * is 120,020: the MESSAGE budget binds, and the bound is 121,020 with the
 * default allowance. The plan's `itemTimeoutMs: 120_000` is BELOW that -- it is
 * the message budget itself, unrounded -- so an item that legitimately spends
 * its whole budget races the driver's own deadline. `assertItemTimeoutMs`
 * refuses it rather than letting the race decide.
 */
export function itemDeadlineBound(input: ItemDeadlineInput): ItemDeadlineBound {
  const { latencyBudgetMs, callBudgetMs, maxCallsPerItem, lowerTierAllowanceMs } = input;
  for (const [what, value] of [
    ["latencyBudgetMs", latencyBudgetMs],
    ["callBudgetMs", callBudgetMs],
    ["lowerTierAllowanceMs", lowerTierAllowanceMs],
  ] as const) {
    // Every one of these reaches a `setTimeout` somewhere downstream, directly
    // or as a summand of the ceiling that does. `manifest.ts` and `cancel.ts`
    // both validate their timer numbers for the measured reason: `Infinity` --
    // the natural spelling of "no budget" -- produces a ~1 ms timeout rather
    // than an infinite one, and NaN, 0, negative and > 2^31 all do the same.
    if (!(Number.isFinite(value) && value >= 0)) {
      throw new Error(
        `${what} must be a finite non-negative number of milliseconds, got ${String(value)} ` +
          `(${typeof value}); a non-finite one becomes a ~1 ms deadline rather than no deadline`,
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
  if (!(Number.isFinite(itemTimeoutMs) && itemTimeoutMs > 0)) {
    throw new Error(
      `itemTimeoutMs must be a finite positive number of milliseconds, got ` +
        `${String(itemTimeoutMs)} (${typeof itemTimeoutMs})`,
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
   * flags, and that matters here: Task 9's priors condition was measured against
   * `minimal-ir.json`, whose `entropy-rule` fires on this corpus's code fence at
   * confidence 0.7 and re-admits it. `semantic-ir.json` -- the only IR a tier-2
   * arm can run today -- declares `rules: []`, so `runTier0` finds nothing, no
   * segment is uncertain, and a tier-0 arm's distribution here is IDENTICAL to a
   * tier-2-only arm's at p95 2, max 2. Anything quoting 3 for this bake-off is
   * quoting a different policy's number.
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
 * A four-model bake-off is hours of GPU time and roughly 12 GB of weights.
 * Learning on the last arm that its file name collides, that its policy is not
 * the one its IR came from, or that the per-item ceiling was set below the
 * message budget wastes all of it. Nothing in here touches the page or the disk.
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
        priorFindings: shape.runsTier0 ? (item) => tier0Priors(ir, item) : () => [],
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
 * be shared by reference. What keeps the two honest is that `runBakeoff` passes
 * this value TO `loadTier2` explicitly and then records the value the page
 * reports back, so a drift shows up as a refusal rather than as a record naming
 * a budget the arm did not run under.
 */
const DEFAULT_TIER2_CALL_BUDGET_MS = 60_000;

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
   * `killed`.
   */
  readonly verdict: "pass" | "fail" | "not-measured";
  readonly detail: string;
}

export interface ArmGateReport {
  readonly arm: string;
  readonly family: ArmFamily;
  readonly modelId: string;
  readonly items: number;
  /** Items whose detection THREW. Their counters are absent, not zero. */
  readonly itemsErrored: number;
  /** Items measured while an earlier item's abandoned work was still running. */
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
   * own `usage.prompt_tokens`, so it is a token count and not the ~1.2 kB
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
  /** The character-size distribution of the segments this arm ran over. */
  readonly segmentChars: SizeStats | undefined;
  readonly segmentsPerItem: SizeStats | undefined;
  readonly escalation: SegmentSizeDistribution["escalation"];
  readonly gates: readonly GateOutcome[];
  /** True when any gate FAILED. `not-measured` never sets it. */
  readonly killed: boolean;
}

export interface GateReportInput {
  readonly arm: string;
  readonly family: ArmFamily;
  readonly modelId: string;
  readonly records: readonly RunRecord[];
  readonly segments: SegmentSizeDistribution;
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
  const foreign = records.find((r) => r.arm !== arm);
  if (foreign !== undefined) {
    // A report summing two arms' rows would be a confident number describing no
    // run at all, and every field of it would look well-formed.
    throw new Error(
      `gateReport was given a record from arm "${foreign.arm}" while summarising "${arm}"; ` +
        `a gate computed over two arms' rows describes neither`,
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
  const resolvableDenominator = ladder.rung1 + ladder.rung2 + ladder.unresolvedQuotes;
  const duplicateDenominator = ladder.rung1 + ladder.rung2 + ladder.duplicatesDropped;

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
        `because the ceiling was derived at a ~1.2 kB prompt`,
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
        `${GATES.minDecodeTokPerSec}`,
    }),
    numericGate({
      gate: "resolvable-rate",
      threshold: GATES.minResolvableRate,
      sample: resolvableDenominator,
      observed:
        resolvableDenominator === 0
          ? undefined
          : (ladder.rung1 + ladder.rung2) / resolvableDenominator,
      passes: (observed) => observed >= GATES.minResolvableRate,
      notMeasured: "this arm produced no quote for the span ladder to place, resolvable or not",
      measured: (observed) =>
        `${ladder.rung1 + ladder.rung2} of ${resolvableDenominator} quote(s) resolved to a span ` +
        `(rung 1: ${ladder.rung1}, rung 2: ${ladder.rung2}), a rate of ${observed.toFixed(3)} ` +
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
    killed: gates.some((g) => g.verdict === "fail"),
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
 * here, because three things in the corpus/policy layer would make a B arm
 * meaningless the moment it ran:
 *
 *   1. No IR in this repository was compiled from any policy document in it.
 *      `semantic-ir.json` carries `policyHash: "test-hash"`, which is not a
 *      sha256 of anything, so `planBakeoff` refuses every B arm here already.
 *   2. That same IR carries `rules: []`, so core's tier 0 finds nothing on any
 *      item -- and `baseline-b-tier0` versus `baseline-b`, the pair that
 *      separates "compiling helps" from "patterns help", would be two runs of
 *      the same thing.
 *   3. `RunRecordSchema` has `tier2Stats` and no baseline equivalent. B's
 *      counters are `BaselineStats`, which renames two events (`messagesJudged`
 *      for `segmentsJudged`, `unknownEntityTypes` for `unknownPredicates`) and
 *      adds `messageBudgetExpiries`, which no judge counter reports. Writing
 *      them into `tier2Stats` would be a record stating one event under another
 *      event's name.
 *
 * So this refuses, naming all of it, rather than running an arm whose numbers
 * would be confident and meaningless.
 */
export function assertPageCanRun(plan: BakeoffPlan): void {
  const baseline = plan.arms.filter((a) => !familyShape(a.family).runsCompiledJudge);
  if (baseline.length === 0) return;
  throw new Error(
    `this harness cannot execute the Approach-B arm(s) [${baseline.map((a) => a.arm).join(", ")}]: ` +
      `apps/eval/src/page/main.ts publishes only core's orchestrator (window.__sih.detect) and ` +
      `never constructs createBaselineB, so no call reaches it. Adding that door is small; what ` +
      `is not is that (1) no IR here was compiled from any policy document here, so B cannot be ` +
      `shown the document the compiled arm's IR came from, (2) semantic-ir.json declares no ` +
      `rules, so tier 0 finds nothing and the B/B+tier0 pair would be two runs of the same arm, ` +
      `and (3) RunRecordSchema has no field for BaselineStats, whose messagesJudged, ` +
      `unknownEntityTypes and messageBudgetExpiries are different events from the judge's. Run ` +
      `the compiled families here and land the baseline families with the compiled policy.`,
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
    // knows the verdict, so nothing can act on it -- there is no `killed` in
    // scope to branch on, and the set of arms was fixed by `planBakeoff` before
    // the first model loaded.
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
}
