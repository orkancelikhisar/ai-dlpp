import type { Action, PolicyIr, PredicateScope, Tier } from "../policy/types.js";
import { resolveAction } from "../policy/resolve.js";
import { segmentText } from "../segment/segment.js";
import { selectSegments, uncertainSegmentStarts } from "./escalate.js";
import { clusterOverlapping, mergeFindings } from "./merge.js";
import { runTier0 } from "./tier0.js";
import type {
  DegradedNotice,
  DetectionResult,
  DetectorEngines,
  EngineDegradedNotice,
  EngineDegradedReason,
  Finding,
  JudgeVerdict,
  ResolvedFinding,
  TierConfig,
} from "./types.js";

/**
 * Detection pipeline -- spec 4.3. Segment once, run the enabled tiers over that
 * one segmentation, normalize what they produced against the IR, resolve
 * overlaps, and attach a provider-resolved action to every survivor.
 *
 * Only tier 0 has an implementation in this plan; tiers 1-2 arrive as engines
 * through `DetectInput.engines` (Plans 4-5) and are already wired here so the
 * guards, the normalization step and the cluster-strictest action resolution
 * below are exercised by stub engines today rather than written blind later.
 *
 * The orchestrator owns everything the tiers should not have to know: what a
 * segment is worth looking at, what a label is allowed to be, what a severity
 * really is, and which action wins. Engines stay dumb -- they see the segments
 * they are handed and report spans.
 */

/**
 * Strictness order for cluster resolution. Not derived from the Action union's
 * declaration order: that is a type, and reordering it must not silently reorder
 * policy. `allow` is the identity, which is why an empty fold would return it --
 * see strictestAction, which never folds over an empty cluster.
 */
/**
 * Action strictness. Exported because the compiler resolves the same ordering
 * when it extends a provider clause to a shadow entityType, and a second copy
 * of this table would let the compiler emit a policy the runtime resolves
 * differently.
 */
export const ACTION_RANK: Record<Action, number> = {
  allow: 0,
  pseudonymize: 1,
  redact: 2,
  block: 3,
};

/**
 * Bring every raw finding under the IR's authority before anything downstream
 * reads it. Three things happen, all of them refusals to trust a producer:
 *
 * 0. The span must be in range, non-empty, and must actually hold the text the
 *    producer reported. `types.ts` states `text === message.slice(start, end)`
 *    as a contract and `SpanTagger` tells implementers to re-derive it, but
 *    nothing enforced it, and a tokenizer-backed model breaks it routinely.
 *    Everything downstream is offsets-first -- `applyActions` rewrites BY SPAN,
 *    so a drifted finding sends a neighbouring word to the vault and leaves the
 *    real value sitting in the message, and merge/cluster order by offsets that
 *    describe nothing. Zero-width is rejected with the out-of-range cases: tier
 *    0 never emits one and it is meaningless downstream (an empty real value the
 *    vault refuses to mint, or a redaction marker spliced in where nothing was
 *    detected). Tier 0 cannot trip any of this -- it slices the message itself.
 *
 *    Neither message quotes the finding's text or the message's. A finding's
 *    text is precisely the sensitive string this system exists to keep out of
 *    places it may be logged, and an exception message is such a place; the
 *    source, entityType, span and the two lengths locate the bug without it.
 *
 * 1. An entityType that is not declared in the IR is a designed rejection. Tier
 *    0 cannot produce one (the schema validates rule.entityType at load), but a
 *    tier-1/2 model absolutely can hallucinate a label, and the failure mode
 *    without this guard is a resolveAction throw deep in the merge with no
 *    mention of which engine invented the label. The message names both.
 * 2. Severity is RE-DERIVED from ir.entityTypes, matching tier0.ts: the entity
 *    table is the single source of truth for what an entity is worth, and merge
 *    resolution is severity-first, so a model that inflates its own severity
 *    would otherwise get to overrule the policy about which finding survives.
 *
 * Pure: producers' objects are never mutated, and one is copied only when its
 * severity actually disagrees with the IR.
 */
function normalizeFindings(ir: PolicyIr, text: string, findings: Finding[]): Finding[] {
  return findings.map((f) => {
    // Integrality first, and not folded into the range check below: the range
    // check is NaN-safe (comparisons against NaN are false, so the positive
    // predicate fails) but NOT null-safe, because `null` coerces to 0 -- so
    // `null >= 0` is true, `slice(null, end)` is `slice(0, end)`, and a null
    // start used to clear the text-fidelity check as well and reach the merge
    // sorting as 0. An offset must be a real integer before any ordering
    // question about it means anything.
    if (!(Number.isInteger(f.start) && Number.isInteger(f.end))) {
      throw new Error(
        `finding from source "${f.source}" (${f.entityType}) has non-integer offsets: ` +
          `[${f.start}, ${f.end})`,
      );
    }
    if (!(f.start >= 0 && f.start < f.end && f.end <= text.length)) {
      throw new Error(
        `finding from source "${f.source}" (${f.entityType}) has an out-of-range span ` +
          `[${f.start}, ${f.end}) over a ${text.length}-character message`,
      );
    }
    if (f.text !== text.slice(f.start, f.end)) {
      throw new Error(
        `finding from source "${f.source}" (${f.entityType}) reports text that does not match its span ` +
          `[${f.start}, ${f.end}): ${f.text.length} characters reported, ${f.end - f.start} in the span ` +
          `(producers must re-derive text as message.slice(start, end))`,
      );
    }
    const entity = ir.entityTypes.find((e) => e.id === f.entityType);
    if (entity === undefined) {
      throw new Error(`finding from source "${f.source}" names unknown entityType "${f.entityType}"`);
    }
    return f.severity === entity.severity ? f : { ...f, severity: entity.severity };
  });
}

/**
 * The strictest action over a whole overlap CLUSTER, which is the action every
 * winner from that cluster carries (except where `winnerAction` escalates it
 * per winner -- a neverPseudonymize winner handed `pseudonymize` becomes
 * `redact`; this function's answer is the cluster's, not the last word).
 *
 * Why not simply resolveAction(winner) -- what the plan's snippet did: merge
 * resolution is severity-first, so a critical `generic-secret` entropy run whose
 * action is `redact` beats a high-severity `in-pan` inside it whose action is
 * `block`, and the merge then throws the PAN away. Reading only the winner's
 * action downgrades a blocked message to a redacted one; the losers are gone by
 * then, so the cluster is the only place the information still exists.
 *
 * Semantics, decided deliberately: strictest is taken cluster-WIDE, not over the
 * winner's direct overlaps. Clusters are transitive, so in a chain A-B-C, C can
 * escalate the action of a winner that overlaps only A. That is the strict
 * direction of the error, and strict is the safe direction for a data-loss
 * filter: the alternative silently under-protects text that a policy did call
 * sensitive. Chains long enough for this to feel surprising are themselves a
 * signal the message is dense with entities.
 *
 * Semantic predicates (ir.semanticPredicates) have no entityType of their own,
 * and this function is keyed entirely by entityType. DECIDED by Plan 3, closing
 * the choice parked here since Plan 1: the compiler mints a SHADOW entityType
 * per predicate -- `pred:<predicateId>`, tier 2, redact, neverPseudonymize --
 * carried in ir.entityTypes and ir.actions.default like any other. The rejected
 * alternative (a `Finding.predicateId` field consulted by resolution) and the
 * reasoning for both live in packages/compiler/src/stages/predicates.ts.
 *
 * So nothing in this file changes: a predicate finding names its shadow id in
 * `entityType`, resolveAction resolves it -- provider overrides included -- and
 * winnerAction reads its neverPseudonymize like any other entityType's.
 *
 * The remaining obligation is the PRODUCER's: tier-2 engines (Plan 5) must emit
 * `entityType: "pred:<id>"`, not the bare predicate id. Getting that wrong is
 * loud rather than silent -- normalizeFindings throws on an entityType the IR
 * does not contain -- which is what this note originally existed to guarantee.
 */
function strictestAction(ir: PolicyIr, provider: string, cluster: Finding[]): Action {
  // clusterOverlapping never emits an empty cluster, so this seed is always
  // overwritten by a real resolveAction result unless the policy itself says
  // `allow` for every member.
  let strictest: Action = "allow";
  for (const f of cluster) {
    const action = resolveAction(ir, f.entityType, provider);
    if (ACTION_RANK[action] > ACTION_RANK[strictest]) strictest = action;
  }
  return strictest;
}

/**
 * The action one WINNER carries, given its cluster's strictest. Almost always
 * the cluster's answer unchanged; the exception is the one combination that
 * cluster-strictest can produce and no downstream stage can honour.
 *
 * `pseudonymize` on a neverPseudonymize entityType is rejected by the schema, so
 * it is unreachable as *policy* -- but it is reachable as a STAMP. A credential
 * class whose own action is `allow` (legal: the schema rejects only
 * pseudonymize for these) can overlap a `pseudonymize` neighbour, and allow(0) <
 * pseudonymize(1) hands the cluster's action to a winner the vault refuses to
 * mint for. The result would be a guaranteed message-level exception on a
 * perfectly valid IR, at apply time, after detection reported success.
 *
 * Resolved by escalating that winner to `redact`: the cluster did say this span
 * deserves rewriting, and redaction rewrites it without minting a format-valid
 * fake credential -- the strict direction, which is the direction this file
 * already commits to for cluster-wide escalation. Downgrading to `allow`
 * instead would forward text the policy wanted rewritten.
 *
 * PER WINNER, not per cluster: a non-credential winner in the same cluster keeps
 * `pseudonymize`, which is legal for it and preserves the answer utility that
 * pseudonymization exists for.
 */
function winnerAction(ir: PolicyIr, winner: Finding, clusterAction: Action): Action {
  if (clusterAction !== "pseudonymize") return clusterAction;
  // normalizeFindings already rejected any finding whose entityType is not in
  // the IR, so a miss here is unreachable rather than a silent pass-through.
  const entity = ir.entityTypes.find((e) => e.id === winner.entityType);
  return entity?.neverPseudonymize === true ? "redact" : clusterAction;
}

/**
 * The largest delay `setTimeout` holds without reinterpreting it.
 *
 * MEASURED HERE, on Node v26.0.0 and on Chrome 148, with the same probe in
 * both: a delay of 2147483647 did not fire within 400 ms, while 2147483648,
 * 4294967296, Infinity, 0, -1 and NaN each fired within about 1 ms. Node also
 * prints a TimeoutOverflowWarning for the first three; Chrome prints nothing at
 * all. So an over-large budget does not become a long deadline, it becomes an
 * IMMEDIATE one -- tier 2 aborted before it starts, on every message, silently
 * in the browser. `ir.latencyBudgetMs` is schema-checked as a positive integer
 * and nothing bounds it above, so this is reachable from a legal policy.
 */
const MAX_TIMER_DELAY_MS = 2_147_483_647;

/**
 * What is left of one MESSAGE's `ir.latencyBudgetMs`, or `undefined` when it is
 * spent.
 *
 * Exported for two reasons. The first is testing: the boundary that matters
 * here -- elapsed exactly equal to the budget -- cannot be steered onto with a
 * wall clock, and it is the one an off-by-one turns into a 1 ms deadline
 * (`> 0` handed on as a number is the difference between a judge that gets its
 * remaining budget and a judge that is interrupted before its first token).
 *
 * The second arrived with the Approach-B baseline and is why this is now on the
 * package index as well. `detect` is no longer the only orchestrator: B
 * implements `Detector` itself, so nothing else on that arm enforces the
 * per-MESSAGE budget spec 5.3 writes down, and B has to arm the same deadline
 * from the same arithmetic. A second copy would drift on exactly the two things
 * this function exists for -- the `> 0` boundary and the MAX_TIMER_DELAY_MS
 * clamp, without which a legal `ir.latencyBudgetMs` above 2^31-1 becomes an
 * IMMEDIATE deadline rather than a long one -- and the arm that drifted would
 * be compared on its budget rather than on its method.
 *
 * `!(remaining > 0)` rather than `remaining <= 0` so a NaN budget lands in
 * "spent" instead of passing through: `loadPolicyIr` is the boundary that keeps
 * NaN out, but if one ever arrives the safe direction is not running tier 2,
 * not running it against a 1 ms timer.
 */
export function remainingBudgetMs(latencyBudgetMs: number, elapsedMs: number): number | undefined {
  const remaining = latencyBudgetMs - elapsedMs;
  if (!(remaining > 0)) return undefined;
  return Math.min(remaining, MAX_TIMER_DELAY_MS);
}

/**
 * Every `PredicateScope`, as values to iterate.
 *
 * Built from a `satisfies Record<PredicateScope, true>` literal rather than
 * written as an array, so adding a third scope to the union is a compile error
 * here instead of a scope the result silently never reports on.
 */
const PREDICATE_SCOPES = Object.keys({
  segment: true,
  message: true,
} satisfies Record<PredicateScope, true>) as PredicateScope[];

/**
 * Scopes the policy declares a predicate in that nothing evaluated, with how
 * many predicates each one holds.
 *
 * Shared by the two paths that can leave a scope unevaluated -- a judge that
 * ran and reported evaluating fewer scopes than the policy declares, and an
 * escalation that selected no segment so the judge was never called -- because
 * the ENUMERATION must not differ between them. Each caller writes its own
 * `detail`, since the two facts are different sentences and a shared one would
 * have to be vague enough to cover both.
 */
function unjudgedScopes(
  ir: PolicyIr,
  judged: readonly PredicateScope[],
): Array<{ scope: PredicateScope; declared: number }> {
  const out: Array<{ scope: PredicateScope; declared: number }> = [];
  for (const scope of PREDICATE_SCOPES) {
    const declared = ir.semanticPredicates.filter((p) => p.scope === scope).length;
    if (declared > 0 && !judged.includes(scope)) out.push({ scope, declared });
  }
  return out;
}

/**
 * A tier that did not run, recorded as the fact it is.
 *
 * The detail says the tier was not enabled and stops there. WHY a caller
 * disabled it -- no WebGPU, an arm of an experiment, a model that failed to
 * load an hour ago -- is knowledge this function does not have, and inventing
 * the WebGPU reason here would be a record stating a cause nobody measured.
 */
function absentNotice(tier: Tier): DegradedNotice {
  return {
    tier,
    reason: "absent",
    detail: `tier ${tier} was not enabled in this TierConfig, so nothing it detects was looked for`,
  };
}

/**
 * The reason words an ENGINE is entitled to say, as values to test against.
 *
 * Built from a `satisfies Record<EngineDegradedReason, true>` literal for the
 * same reason `PREDICATE_SCOPES` is: widening the engine union has to be a
 * compile error here, not a word this guard silently starts rejecting at
 * runtime.
 */
const ENGINE_DEGRADED_REASONS = Object.keys({
  "failed-closed": true,
  "call-budget-exhausted": true,
} satisfies Record<EngineDegradedReason, true>);

/**
 * One engine notice, turned into a result entry field by field -- never a
 * spread -- with both fields defended.
 *
 * The two fields need DIFFERENT defences because the orchestrator knows
 * different amounts about them. `tier` it knows outright: this call was made as
 * tier N, so an engine's claim is overwritten rather than consulted. `reason` it
 * cannot know -- what went wrong inside a model is precisely what the engine is
 * being asked -- so there is no correct value to substitute, and the choices are
 * to refuse the notice or to file a word that means something else. It refuses,
 * exactly as `normalizeFindings` refuses an entityType the IR does not declare,
 * and for the same reason: a confidently wrong record is worse than a loud one.
 *
 * `absent` is the case that makes this matter rather than tidy. It is an
 * orchestrator-only word meaning "this tier did not run", and an engine that
 * reached this line demonstrably ran -- so a `{tier: 2, reason: "absent"}` entry
 * would say the judge never ran on a row that carries the judge's own findings.
 * That is the same inverted record the `tier` defence exists to prevent,
 * arriving through the other field. `budget-exhausted` and `scope-unjudged` are
 * refused on the same ground: both are claims about `ir.latencyBudgetMs` and the
 * policy's scopes, which only `detect` measures.
 *
 * The threat model is untyped JS -- which the eval harness's page boundary is --
 * so the guard is written against values the types already forbid, and `detail`
 * is checked for being a non-empty string on the same grounds: it is the whole
 * human-readable payload of a notice, and `DegradedNotice` types it `string`.
 * A spread would additionally copy whatever else such a producer attached (a
 * `{reason, detail, offendingBody}` shape, say) into a diagnostic that gets
 * logged, which `EngineDegradedNotice`'s docblock forbids.
 *
 * The offending word is named in the message. It is a fixed-vocabulary field
 * rather than a place a message's text lives, and naming it is what makes the
 * error diagnosable -- the same trade `normalizeFindings` makes with `source`
 * and `entityType`.
 */
function stampEngineNotice(tier: Tier, notice: EngineDegradedNotice): DegradedNotice {
  if (!ENGINE_DEGRADED_REASONS.includes(notice.reason)) {
    throw new Error(
      `tier-${tier} engine reported a degradation reason it may not name: "${String(notice.reason)}" ` +
        `(an engine may report ${ENGINE_DEGRADED_REASONS.join(" or ")}; "absent", ` +
        `"budget-exhausted" and "scope-unjudged" are facts about the orchestrator's own calls)`,
    );
  }
  if (typeof notice.detail !== "string" || notice.detail.length === 0) {
    throw new Error(
      `tier-${tier} engine reported a "${notice.reason}" degradation with no detail; ` +
        `detail is required and is the entire human-readable payload of a notice`,
    );
  }
  return { tier, reason: notice.reason, detail: notice.detail };
}

export interface DetectInput {
  ir: PolicyIr;
  provider: string;
  text: string;
  config: TierConfig;
  engines?: DetectorEngines;
}

/**
 * Run detection over one message and return resolved findings plus per-tier
 * timings.
 *
 * THROWS, and never silently degrades. Engine crashes, invalid findings and
 * configuration-guard violations all surface as exceptions, and mapping them to
 * `ir.failMode` (spec 5.3: `open` forwards the message, `closed` blocks it) is
 * the CALLER's responsibility -- the caller is the only layer that knows whether
 * it is a browser extension with a user to warn or a batch harness.
 *
 * What it DOES return short of a full run, it names: `result.degraded` carries
 * one entry per tier that did not run, was cut short by the message's latency
 * budget, or reported failing closed. That is the promise spec section 7 makes
 * -- a tier-2 body still invalid after one repair is "flagged for user review",
 * never a silent pass-through -- and it is a channel, not an exception, because
 * the surviving findings are still good and the caller still has a message to
 * decide about. An exception loses both.
 */
export async function detect(input: DetectInput): Promise<DetectionResult> {
  const { ir, provider, text, config, engines } = input;

  // Enabling a tier without supplying its engine is a caller bug, not a reason
  // to quietly run fewer tiers: the whole point of TierConfig is measuring
  // arms against each other, and an arm that silently degrades to T0 reports
  // T0's latency and T0's recall under T0+T1's name. Note that tier 2 without
  // tier 1 is NOT an error -- tier 2 judges semantic predicates over segments
  // and does not consume tier-1 spans, so predicates-only escalation is valid.
  if (config.tier1 && engines?.tier1 === undefined) {
    throw new Error("tier1 enabled but no tier-1 engine provided");
  }
  if (config.tier2 && engines?.tier2 === undefined) {
    throw new Error("tier2 enabled but no tier-2 engine provided");
  }
  // The tier blocks below re-test `engines?.tierN !== undefined`. That is TYPE
  // NARROWING, not a second decision about whether to run: the guards above have
  // already made a missing engine unreachable there. Deleting the guards because
  // those checks "look like they handle it" is exactly the silent skip.

  // The message's own clock, started before segmentation because segmentation is
  // work the user waits through. `ir.latencyBudgetMs` is a per-MESSAGE number
  // (spec 5.3) and every per-call budget below is measured against what is left
  // of it, not against the whole of it.
  const messageStarted = performance.now();

  // Segmented once and shared by every tier: segmentation is deterministic, and
  // re-running it per tier would let two tiers disagree about where a code fence
  // ends while both report absolute offsets into the same message.
  const segments = segmentText(text);
  const raw: Finding[] = [];
  const timings: DetectionResult["timings"] = { tier0Ms: 0 };
  // Appended to in tier order, so a caller can read the array top to bottom and
  // find the earliest thing that weakened the result first.
  const degraded: DegradedNotice[] = [];

  // Each tier is timed around the tier call ALONE -- normalization is
  // orchestrator overhead and charging it to a tier corrupts the comparison the
  // timings exist for. Normalizing per tier (rather than once at the end) also
  // fails fast: a tier-1 hallucination throws before tier 2 spends its budget.
  if (!config.tier0) {
    degraded.push(absentNotice(0));
  } else {
    const started = performance.now();
    const found = runTier0(ir, text, segments);
    timings.tier0Ms = performance.now() - started;
    raw.push(...normalizeFindings(ir, text, found));
  }

  if (!config.tier1) {
    degraded.push(absentNotice(1));
  } else if (engines?.tier1 !== undefined) {
    // Prose and kv only: code segments are tier 0's ground (entropy scans them)
    // and are mostly identifiers and syntax, which a span tagger reads as a wall
    // of false positives while burning the latency budget. The FILTER LIVES
    // HERE, not in the engine -- engines receive segments and tag them, so which
    // segments deserve a model is a policy-shaped decision the orchestrator owns
    // and can change for every engine at once.
    const taggable = segments.filter((s) => s.kind !== "code");
    const started = performance.now();
    const found = await engines.tier1.tag(taggable, ir);
    timings.tier1Ms = performance.now() - started;
    raw.push(...normalizeFindings(ir, text, found));
  }

  if (!config.tier2) {
    degraded.push(absentNotice(2));
  } else if (engines?.tier2 !== undefined) {
    // Escalation -- spec 4.1: "tier2 iff (uncertain OR semanticPredicates
    // present) AND TierConfig.tier2". The config half is the branch above; the
    // other half is `escalate.ts`, and it is applied HERE for the same reason
    // tier 1's kind filter is: engines answer about the segments they are
    // handed, so which segments deserve a model has to be changeable for every
    // engine at once -- including the Approach-B arm, which is not a judge.
    //
    // `raw` is the right input to the uncertainty half and the only moment it is
    // available: it holds exactly what tiers 0 and 1 found on this message, and
    // spec 4.1 defines uncertainty as what THEY left behind.
    const escalated = selectSegments(segments, {
      hasPredicates: ir.semanticPredicates.length > 0,
      uncertain: uncertainSegmentStarts(segments, raw, config.uncertainBelow),
    });

    // The per-MESSAGE budget, enforced here because nothing else can: a judge
    // makes one engine call per segment, each one compliant with its own
    // per-call budget, and twelve compliant calls are twelve times the number
    // spec 5.3 wrote down. The orchestrator is the only layer that knows both
    // `ir.latencyBudgetMs` and how much of it the earlier tiers already spent.
    const remaining = remainingBudgetMs(ir.latencyBudgetMs, performance.now() - messageStarted);

    // Tested BEFORE the budget, because the budget's notice makes a causal
    // claim ("already spent ... so the judge was not called") that would be
    // wrong here: with nothing escalated, no budget at all would have produced
    // a call, and an operator who raised `ir.latencyBudgetMs` in response would
    // see no change.
    //
    // Whether this is a DEGRADATION depends on what the policy asked for, and
    // the two cases genuinely differ:
    //
    // - No predicates and nothing uncertain: NOT a degradation, and nothing is
    //   filed. `DetectionResult.degraded` means "weaker than a full THREE-TIER
    //   run", and this is not weaker than one. `WebLlmJudge.judge` returns
    //   `{findings: [], scopesJudged: []}` on an IR with no `semanticPredicates`
    //   before it touches the engine, so calling it would have produced the same
    //   `findings` and the same `degraded` -- filing a notice here would report
    //   a degradation the pipeline does not report when the judge really runs.
    //   The one thing that DOES differ is `timings.tier2Ms`, set on the call
    //   path and unset on this one, and that is precisely how a caller asking
    //   "did tier 2 run?" tells them apart.
    //
    // - Predicates declared but no segment selected (a message that is entirely
    //   a code fence, with nothing uncertain in it): weaker, and reported. The
    //   policy declares clauses that went unevaluated, which is what
    //   `scope-unjudged` already means -- no new reason word is needed, and
    //   inventing one would split the bake-off's count of unevaluated
    //   predicates across two words that mean the same thing to a reader.
    //
    // No `timings.tier2Ms` on either path: a 0 there reads as a tier that ran
    // instantly, which is the confusion `timings`' own doc warns about.
    if (escalated.length === 0) {
      // No scope was judged, because no judge ran -- hence the empty second
      // argument. The enumeration is shared with the post-verdict loop below so
      // the two paths cannot disagree about which scopes a policy declares.
      for (const { scope, declared } of unjudgedScopes(ir, [])) {
        degraded.push({
          tier: 2,
          reason: "scope-unjudged",
          detail:
            `the policy declares ${declared} semantic predicate(s) with scope "${scope}", and ` +
            `escalation selected none of this message's ${segments.length} segment(s), so the ` +
            `tier-2 judge was not called and those predicates were not judged`,
        });
      }
    } else if (remaining === undefined) {
      // Not called at all, rather than called with a budget of zero -- and the
      // refusal has to be HERE, because no judge performs it. The shipped one
      // does not: `WebLlmJudge`'s constructor validates the per-call budget it
      // is BUILT with (`WebLlmJudgeOptions.budgetMs`) and `judge()` reads
      // `request.budgetMs` for nothing at all, which its own docblock states in
      // as many words. Handed a 0 it would ignore the 0 and run a full call.
      //
      // What a spent budget actually buys is the deadline below armed at 0, and
      // a setTimeout(0) fires in about 1 ms -- see MAX_TIMER_DELAY_MS for that
      // measurement -- so the judge would be started and aborted before its
      // first token: one model call spent to produce nothing.
      //
      // `timings.tier2Ms` stays unset for the same reason it does for any tier
      // that did not run: a 0 there would read as a tier that ran instantly.
      degraded.push({
        tier: 2,
        reason: "budget-exhausted",
        detail:
          `the ${ir.latencyBudgetMs}ms message latency budget was already spent when tier 2's ` +
          `turn came, so the judge was not called`,
      });
    } else {
      // A DEADLINE, not a race. `SemanticJudge` takes a signal precisely so an
      // over-budget run can be stopped rather than abandoned still running --
      // Plan 5 measured that racing a timeout against a WebLLM call wedges the
      // engine permanently, so the signal is the only stop that leaves the next
      // message a working engine.
      const controller = new AbortController();
      let expired = false;
      const deadline = setTimeout(() => {
        expired = true;
        controller.abort();
      }, remaining);
      const started = performance.now();
      let verdict: JudgeVerdict;
      try {
        verdict = await engines.tier2.judge({
          // The WHOLE message alongside a FILTERED segment list, which is
          // exactly the pairing `JudgeRequest.text` exists for: since
          // escalation landed, `escalated.map(s => s.text).join("")` is no
          // longer the message, so a judge that reconstructed the text from its
          // segments would silently lose whatever escalation dropped -- and a
          // `scope: "message"` predicate would be judged against a message with
          // holes in it.
          text,
          segments: escalated,
          ir,
          // A snapshot, not the live accumulator: `raw` is pushed into again the
          // moment the judge returns, so handing over the array itself would let
          // an engine that holds onto it observe findings it never saw -- or
          // splice the list detection is about to resolve.
          //
          // Shallow on purpose. The Finding objects inside ARE shared with the
          // pipeline, which is safe only because findings are treated as
          // immutable everywhere (nothing here or downstream mutates one;
          // normalizeFindings copies rather than rewrites). An engine that
          // mutates a prior in place violates that convention and corrupts the
          // merge. Deep-cloning every finding on every tier-2 call would buy
          // protection against a contract breach at a per-message allocation
          // cost, and is not worth it.
          //
          // The WHOLE message's findings, not the escalated segments'. A judge
          // weighs a segment against what the pipeline already knows, and a hit
          // one segment away is context escalation had no reason to drop.
          priorFindings: [...raw],
          budgetMs: remaining,
          signal: controller.signal,
        });
      } finally {
        // In `finally` because the throw path is the one that leaks.
        //
        // What a leaked timer costs, MEASURED here rather than assumed, because
        // this comment used to claim the larger of the two and the larger one
        // cannot happen: it does NOT abort a later message. `controller` is
        // built inside this branch, so it is per-call; a probe that neutered
        // `clearTimeout` for a call whose judge threw, then ran a second detect
        // with a 10s budget and a judge watching its signal, saw call 1's own
        // signal abort (control) and call 2's signal stay unaborted through the
        // whole run. A leaked timer aborts a controller nobody is listening to.
        //
        // What it does cost: `setTimeout` returns a REF'd handle -- measured on
        // Node v26.0.0, `hasRef()` is true, and a process whose only remaining
        // work was an uncleared 800 ms timer exited at 803 ms instead of at 0.
        // So a batch harness lingers for the rest of every abandoned budget and
        // vitest reports the run as holding a handle. Cheap to prevent, so it is
        // prevented.
        //
        // The cross-message abort is worth keeping named because it becomes REAL
        // the moment someone hoists `controller` to `detect` scope, or shares one
        // across messages: then a leaked timer aborts whatever is in flight, and
        // it reads as a slow model on a message that was never slow.
        clearTimeout(deadline);
      }
      timings.tier2Ms = performance.now() - started;

      // Recorded from the TIMER, not from what the judge returned. A judge that
      // ignores the signal and answers in full still ran past the budget, and
      // that is a fact about this message's latency either way.
      if (expired) {
        degraded.push({
          tier: 2,
          reason: "budget-exhausted",
          detail:
            `the ${ir.latencyBudgetMs}ms message latency budget ran out during the tier-2 run: ` +
            `the judge was given the ${remaining.toFixed(0)}ms that remained of it and was then ` +
            `sent an abort`,
        });
      }

      // The tier this call was made as, stamped; the engine's own words,
      // checked. See `stampEngineNotice` for why the two fields get different
      // treatment and why an off-vocabulary reason throws rather than passing.
      for (const notice of verdict.degraded ?? []) {
        degraded.push(stampEngineNotice(2, notice));
      }

      // The scope the policy ASKED about against the scopes the judge says it
      // evaluated. This is the whole of `SemanticPredicate.scope`'s enforcement
      // today: core cannot make a judge read a message, but it can refuse to let
      // a bake-off count an unevaluated predicate as a clean one.
      for (const { scope, declared } of unjudgedScopes(ir, verdict.scopesJudged)) {
        degraded.push({
          tier: 2,
          reason: "scope-unjudged",
          detail:
            `the policy declares ${declared} semantic predicate(s) with scope "${scope}" and the ` +
            `tier-2 judge reported evaluating [${verdict.scopesJudged.join(", ")}], so those ` +
            `predicates were not judged in the scope they were written for`,
        });
      }

      raw.push(...normalizeFindings(ir, text, verdict.findings));
    }
  }

  return { findings: resolveFindings(ir, provider, text, raw), timings, degraded };
}

/**
 * Everything between "the tiers have spoken" and "the caller has a
 * DetectionResult.findings": validate against the IR, resolve overlaps, and
 * attach the action each survivor carries.
 *
 * Extracted from `detect` and EXPORTED because `detect` is no longer the only
 * producer of a `DetectionResult`. The Approach-B baseline
 * (`packages/tier2/src/baselineB.ts`) implements `Detector` directly -- no
 * compiler, no tiers -- and the head-to-head between the two is the whole
 * result Plan 5 exists to produce. A second copy of this composition in that
 * arm is the way that result becomes an artifact of the harness: the recipe is
 * cluster-wide strictest action with a per-winner escalation, three decisions
 * whose reasoning lives on `strictestAction` and `winnerAction`, and an arm
 * that got any one of them differently would be compared on ACTION RESOLUTION
 * while the write-up said it was compared on method. Callers do not get to
 * hold a partly-normalized result.
 *
 * `detect` still normalizes PER TIER before calling this, and that is not
 * redundant: it fails fast, so a tier-1 hallucination throws before tier 2
 * spends its budget. Normalization is idempotent -- it validates spans and
 * re-derives severity from the same entity table -- so the second pass over an
 * already-normalized finding returns that same object.
 *
 * The per-cluster composition recipe is documented in merge.ts: cluster once,
 * merge within each cluster, and give that cluster's winners the strictest
 * action any member of the cluster resolved to. Merging a cluster is the same
 * as merging everything and filtering (a cluster re-clusters to itself), so
 * this loses nothing the whole-input call would have found.
 *
 * The result is already globally ordered and deliberately not re-sorted:
 * clusters come back ordered by start and are pairwise disjoint, and each
 * cluster's winners come back in merge.ts's canonical order, so concatenation
 * preserves both. A re-sort here on `start` alone would be a weaker order than
 * the one the clusters already carry.
 *
 * @param text the message the offsets index into; required because the first
 *   thing this does is refuse a finding whose reported text is not the slice.
 * @throws whatever `normalizeFindings` throws -- an out-of-range span, a span
 *   whose text disagrees with it, an entityType the IR does not declare.
 */
export function resolveFindings(
  ir: PolicyIr,
  provider: string,
  text: string,
  raw: Finding[],
): ResolvedFinding[] {
  const normalized = normalizeFindings(ir, text, raw);
  const findings: ResolvedFinding[] = [];
  for (const cluster of clusterOverlapping(normalized)) {
    const action = strictestAction(ir, provider, cluster);
    for (const winner of mergeFindings(cluster)) {
      findings.push({ ...winner, action: winnerAction(ir, winner, action) });
    }
  }
  return findings;
}

/**
 * The detector interface promised by spec 4.3, and the shape the Approach-B
 * baseline is measured through: `detect` with its input bound to nothing, so an
 * alternative implementation (a single-model detector, a remote one) can stand
 * in wherever a Detector is expected.
 */
export type Detector = (input: DetectInput) => Promise<DetectionResult>;
