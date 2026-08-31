import type { Action, PolicyIr, PredicateScope, Severity, Tier } from "../policy/types.js";
import type { Segment } from "../segment/segment.js";

export interface Finding {
  /** Absolute character offsets into the original message text. */
  start: number;
  end: number;
  /** Exact matched text: always === message.slice(start, end). */
  text: string;
  entityType: string;
  severity: Severity;
  tier: Tier;
  /** Rule id (tier 0) or model identifier (tiers 1-2). */
  source: string;
  /** 0..1 */
  confidence: number;
}

export interface ResolvedFinding extends Finding {
  action: Action;
}

export interface TierConfig {
  tier0: boolean;
  tier1: boolean;
  tier2: boolean;
  t1Model?: string;
  t2Model?: string;
  backend?: "wasm" | "webgpu";
}

/**
 * Seam for tier 1 (Plan 4). Labels come from ir.entityTypes at inference.
 *
 * Implementers MUST re-derive `text` as message.slice(start, end) from the
 * offsets they report, and never pass through model-produced span text:
 * tokenizer offset drift makes the two disagree, and everything downstream
 * relies on Finding.text matching the offsets exactly.
 *
 * `signal` is a real cancellation channel where a caller supplies one, and an
 * implementer should abort in-flight work when it fires -- a timeout raced
 * against a model call leaves the inference running, which then serializes
 * behind the engine and delays the NEXT message.
 *
 * What it is not, stated because the wording here used to imply otherwise:
 * `detect` does not pass a signal to `tag`. `ir.latencyBudgetMs` is enforced
 * against tier 2 only (see `JudgeRequest`), and tier 1's time is charged
 * against that budget rather than bounded by it -- so a slow tier 1 shortens
 * tier 2's share and can spend the whole budget itself. Bounding tier 1 is a
 * change to what Plan 4 measured and has no owner yet.
 */
export interface SpanTagger {
  tag(segments: Segment[], ir: PolicyIr, signal?: AbortSignal): Promise<Finding[]>;
}

/**
 * Why a run of a tier produced less than a complete run of it would have.
 *
 * TWO unions, not one, because the two producers know different things. An
 * engine can only report what happened inside its own run; whether a tier ran
 * AT ALL, and whether a predicate scope went unevaluated, are facts about the
 * orchestrator's own calls and are not an engine's to assert. Keeping the
 * engine union narrow is what lets `EngineDegradedNotice` carry no `tier` --
 * see there.
 *
 * - `failed-closed`: the tier ran and refused to answer for part of the
 *   message. Spec section 7: a tier-2 body still invalid after one repair retry
 *   is flagged for user review and never passed through as a judgement.
 * - `budget-exhausted`: `ir.latencyBudgetMs` for this MESSAGE ran out. Emitted
 *   whether the tier was cut short mid-run or never started because the earlier
 *   tiers had already spent it.
 * - `absent`: the tier did not run. Not a failure and deliberately not spelled
 *   as one -- a machine with no WebGPU runs tier 0 and tier 1 exactly as
 *   designed -- but a caller reading an empty `findings` still has to know.
 * - `scope-unjudged`: the policy declares predicates in a scope the judge did
 *   not evaluate. See `JudgeVerdict.scopesJudged`.
 */
export type EngineDegradedReason = "failed-closed" | "budget-exhausted";
export type DegradedReason = EngineDegradedReason | "absent" | "scope-unjudged";

/**
 * A degradation an ENGINE reports about its own run.
 *
 * It carries no `tier`, on purpose. The orchestrator knows which tier it just
 * called and stamps that; an engine that could name a tier could name the wrong
 * one, and a `DetectionResult` saying tier 0 failed closed when tier 2 did is
 * exactly the records-state-intent defect `normalizeFindings` exists to prevent
 * for findings.
 */
export interface EngineDegradedNotice {
  readonly reason: EngineDegradedReason;
  /**
   * One sentence a human can act on, stating what happened rather than what was
   * intended.
   *
   * MUST NOT quote message text, a finding's text, or a model's output. A
   * notice is a diagnostic and diagnostics get logged; the sensitive string is
   * the thing this system exists to keep out of logs. Counts, ids, offsets and
   * fixed reason words locate the event without it -- the same rule
   * `normalizeFindings`' exception messages follow.
   */
  readonly detail: string;
}

/**
 * One entry in `DetectionResult.degraded`: which tier, and why it contributed
 * less than a full run would have.
 *
 * Not an `extends EngineDegradedNotice`, and the compiler is what says so:
 * `DegradedReason` WIDENS the engine union, and a widened property is not a
 * legal override of the one it widens. The two shapes are deliberately
 * separate declarations for that reason.
 */
export interface DegradedNotice {
  readonly tier: Tier;
  readonly reason: DegradedReason;
  /** Never message text, a finding's text, or model output; see EngineDegradedNotice. */
  readonly detail: string;
}

/**
 * Everything a semantic judge is given about one message.
 *
 * An object rather than positional arguments because two of these fields exist
 * only to be reachable, and a positional signature makes each new one a
 * breaking change for every implementor: `text` is what makes `scope: "message"`
 * expressible at all, and `budgetMs` is what makes the per-message budget
 * visible to a judge that spends it per segment.
 */
export interface JudgeRequest {
  /**
   * The WHOLE message, verbatim, and the string every `segments` offset indexes
   * into.
   *
   * Here because a judge cannot reconstruct it: `segments` may be a FILTERED
   * list (tier 1 is already handed prose and kv only, and tier-2 escalation
   * will filter too), so joining what it was handed would silently drop the
   * text in between and shift nothing but the meaning. A message-scoped
   * predicate whose evidence spans two segments is invisible to a judge that
   * only sees segments -- that is the whole reason this field exists.
   */
  readonly text: string;
  /** The segments this judge is asked to look at, with absolute offsets into `text`. */
  readonly segments: Segment[];
  readonly ir: PolicyIr;
  /** A snapshot of what the earlier tiers found; see the call site in `detect`. */
  readonly priorFindings: Finding[];
  /**
   * Milliseconds of `ir.latencyBudgetMs` still unspent when this request was
   * built -- the message's remaining budget, not one call's. Always positive:
   * a spent budget means the judge is not called at all.
   *
   * INFORMATIONAL. `signal` is the enforcement, and both are computed from the
   * same clock read, so a judge that ignores this number is still stopped. It
   * is here so a judge that makes N calls per message can size them against
   * what is left rather than against a per-call constant that knows nothing
   * about the other segments.
   */
  readonly budgetMs: number;
  /**
   * Fires when the message's latency budget expires. A judge that ignores it
   * runs to completion and the result is still reported as over budget -- what
   * the signal buys is that a slow model stops costing the user time, not that
   * the record is honest, which it is either way.
   */
  readonly signal?: AbortSignal;
}

/**
 * What a judge produces: its findings, plus the two things only it can say
 * about its own run.
 *
 * `findings` alone cannot distinguish "this message is clean" from "this model
 * would not answer", and that ambiguity is the failure this type exists to
 * remove -- spec section 7 requires a tier-2 body that will not parse to be
 * flagged, never silently passed through.
 */
export interface JudgeVerdict {
  readonly findings: Finding[];
  /**
   * Which predicate scopes this judge EVALUATED -- a fact about the run, not a
   * declaration of capability.
   *
   * Required, and required is the point. `SemanticPredicate.scope` went
   * unhonoured through Plans 1-5 with nothing but a docblock admitting it,
   * because nothing in the type system asked. A judge cannot compile without
   * answering now, and the orchestrator turns "the policy declares a scope you
   * did not evaluate" into a `scope-unjudged` notice on the result -- so a
   * bake-off that under-measures message-scoped predicates says so in every
   * row instead of in a write-up nobody reads.
   *
   * Naming a scope does NOT claim every predicate in it was reached: a run cut
   * short reports that separately, in `degraded`.
   */
  readonly scopesJudged: readonly PredicateScope[];
  /**
   * Degradations inside this run. Omitted (or empty) means the judge answered
   * for everything it was asked about.
   */
  readonly degraded?: readonly EngineDegradedNotice[];
}

export interface SemanticJudge {
  judge(request: JudgeRequest): Promise<JudgeVerdict>;
}

export interface DetectorEngines {
  tier1?: SpanTagger;
  tier2?: SemanticJudge;
}

export interface DetectionResult {
  /**
   * Pairwise DISJOINT and sorted by start offset -- overlaps were already
   * resolved when this array was built. A rewriter can therefore consume the
   * array in one walk without checking for collisions or re-sorting.
   *
   * Plan 2's rewriter (`pseudo/apply.ts`, `applyActions`) walks it FORWARD and
   * assembles a new string -- copy the gap since the last span, append the
   * replacement, advance the cursor -- rather than splicing the original in
   * place. Assembly is why offset shift never has to be reasoned about: the
   * replacement's new offsets simply fall out of the output length as it is
   * built, so a replacement of a different length costs nothing. (Reverse-walk
   * splicing is the alternative that keeps in-place edits valid, and it is what
   * this comment used to recommend; it survives only if every consumer edits
   * the original buffer, and it cannot report new offsets without a second
   * pass.)
   */
  findings: ResolvedFinding[];
  /**
   * Wall-clock per tier, measured around the tier's own work only. A tier that
   * did not run has no entry -- except `tier0Ms`, which is required by this type
   * and reads 0 when `config.tier0` was false. Zero there means "did not run",
   * not "ran instantly", so read it against the TierConfig that produced it
   * before charting it as a latency.
   */
  timings: { tier0Ms: number; tier1Ms?: number; tier2Ms?: number };
  /**
   * Every way this result is weaker than a complete run of the configured
   * pipeline, in tier order. EMPTY means every configured tier ran and answered
   * for the whole message -- so an empty `findings` with an empty `degraded` is
   * the one thing that entitles a caller to say the message is clean.
   *
   * REQUIRED, not optional, and that is the field's whole value. An optional
   * one lets a producer that never learned to report degradation look exactly
   * like a clean run, which is the ambiguity being closed here: today an arm
   * that fails closed on 40% of its messages and an arm that finds nothing
   * produce identical results, and the bake-off's central comparison is between
   * exactly those two. `[]` is a positive claim; `undefined` would be silence.
   *
   * It also settles a question `timings` could not: `tier0Ms` reads 0 both for
   * "ran instantly" and for "was not enabled", which its own doc warns about.
   * An `absent` entry says which.
   *
   * NOT a substitute for `ir.failMode`. Detection still THROWS on an engine
   * crash rather than filing a notice -- see `detect` -- so nothing here means
   * "something went wrong and was swallowed".
   */
  degraded: DegradedNotice[];
}
