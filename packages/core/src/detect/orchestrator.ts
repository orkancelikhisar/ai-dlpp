import type { Action, PolicyIr } from "../policy/types.js";
import { resolveAction } from "../policy/resolve.js";
import { segmentText } from "../segment/segment.js";
import { clusterOverlapping, mergeFindings } from "./merge.js";
import { runTier0 } from "./tier0.js";
import type { DetectionResult, DetectorEngines, Finding, ResolvedFinding, TierConfig } from "./types.js";

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
const ACTION_RANK: Record<Action, number> = { allow: 0, pseudonymize: 1, redact: 2, block: 3 };

/**
 * Bring every raw finding under the IR's authority before anything downstream
 * reads it. Two things happen, both of them refusals to trust a producer:
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
function normalizeFindings(ir: PolicyIr, findings: Finding[]): Finding[] {
  return findings.map((f) => {
    const entity = ir.entityTypes.find((e) => e.id === f.entityType);
    if (entity === undefined) {
      throw new Error(`finding from source "${f.source}" names unknown entityType "${f.entityType}"`);
    }
    return f.severity === entity.severity ? f : { ...f, severity: entity.severity };
  });
}

/**
 * The strictest action over a whole overlap CLUSTER, which is the action every
 * winner from that cluster carries.
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
 * Semantic predicates (ir.semanticPredicates) have no entityType, so they have
 * no entityType -> action path and cannot be resolved here at all. The open
 * decision is whether the compiler mints shadow entityTypes for them (Plan 3) or
 * Finding grows a `predicateId` that resolution consults (Plan 5); either way it
 * lands in this function. Recorded here so tier 2 does not quietly acquire an
 * action of `allow` by default when it starts emitting predicate findings.
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
 * it is a browser extension with a user to warn or a batch harness. A partial
 * result with a `degraded`/`warnings` field on DetectionResult is expected in
 * Plan 5, when tier-2 latency-budget degradation gives it a second producer.
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

  // Segmented once and shared by every tier: segmentation is deterministic, and
  // re-running it per tier would let two tiers disagree about where a code fence
  // ends while both report absolute offsets into the same message.
  const segments = segmentText(text);
  const raw: Finding[] = [];
  const timings: DetectionResult["timings"] = { tier0Ms: 0 };

  // Each tier is timed around the tier call ALONE -- normalization is
  // orchestrator overhead and charging it to a tier corrupts the comparison the
  // timings exist for. Normalizing per tier (rather than once at the end) also
  // fails fast: a tier-1 hallucination throws before tier 2 spends its budget.
  if (config.tier0) {
    const started = performance.now();
    const found = runTier0(ir, text, segments);
    timings.tier0Ms = performance.now() - started;
    raw.push(...normalizeFindings(ir, found));
  }

  if (config.tier1 && engines?.tier1 !== undefined) {
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
    raw.push(...normalizeFindings(ir, found));
  }

  if (config.tier2 && engines?.tier2 !== undefined) {
    // Every segment, plus everything found so far. Escalation -- deciding which
    // segments are worth a judge at all, and enforcing latencyBudgetMs with the
    // AbortSignal that SemanticJudge already accepts -- is Plan 5's; running the
    // judge over the whole message is the conservative placeholder.
    //
    // Priors are a snapshot, not the live accumulator: `raw` is pushed into
    // again the moment the judge returns, so handing over the array itself
    // would let an engine that holds onto it observe findings it never saw --
    // or splice the list detection is about to resolve.
    //
    // Shallow on purpose. The Finding objects inside ARE shared with the
    // pipeline, which is safe only because findings are treated as immutable
    // everywhere (nothing here or downstream mutates one; normalizeFindings
    // copies rather than rewrites). An engine that mutates a prior in place
    // violates that convention and corrupts the merge. Deep-cloning every
    // finding on every tier-2 call would buy protection against a contract
    // breach at a per-message allocation cost, and is not worth it.
    const started = performance.now();
    const found = await engines.tier2.judge(segments, ir, [...raw]);
    timings.tier2Ms = performance.now() - started;
    raw.push(...normalizeFindings(ir, found));
  }

  // The per-cluster composition recipe documented in merge.ts: cluster once,
  // merge within each cluster, and give that cluster's winners the strictest
  // action any member of the cluster resolved to. Merging a cluster is the same
  // as merging everything and filtering (a cluster re-clusters to itself), so
  // this loses nothing the whole-input call would have found.
  const findings: ResolvedFinding[] = [];
  for (const cluster of clusterOverlapping(raw)) {
    const action = strictestAction(ir, provider, cluster);
    for (const winner of mergeFindings(cluster)) findings.push({ ...winner, action });
  }
  // Already globally ordered, and deliberately not re-sorted: clusters come back
  // ordered by start and are pairwise disjoint, and each cluster's winners come
  // back in merge.ts's canonical order, so concatenation preserves both. A
  // re-sort here on `start` alone would be a weaker order than the one the
  // clusters already carry.
  return { findings, timings };
}

/**
 * The detector interface promised by spec 4.3, and the shape the Approach-B
 * baseline is measured through: `detect` with its input bound to nothing, so an
 * alternative implementation (a single-model detector, a remote one) can stand
 * in wherever a Detector is expected.
 */
export type Detector = (input: DetectInput) => Promise<DetectionResult>;
