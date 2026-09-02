import type { Severity } from "../policy/types.js";
import type { Finding } from "./types.js";

/**
 * Overlap resolution -- spec 4.1. Overlapping findings are resolved by highest
 * severity, then higher confidence, then wider span; non-overlapping findings
 * pass through, sorted by start.
 *
 * PURE over Finding: no IR, no message text, no actions. Severity was already
 * re-derived from ir.entityTypes by the producers (tier0.ts), so re-reading the
 * IR here could only disagree with them; actions are resolved downstream in
 * Task 12 over the clusters this module exposes.
 *
 * Both exports return the CALLER'S OWN finding objects, never copies, so the two
 * can be joined by reference. Composing them for action resolution goes one of
 * two ways, and either is fine:
 *
 *   per-cluster merge:  for (const cluster of clusterOverlapping(raw)) {
 *                         const winners = mergeFindings(cluster);
 *                         // action = strictest over `cluster`, applied to `winners`
                         // (orchestrator.ts then adjusts PER winner: see winnerAction)
 *                       }
 *   identity-map join:  const byMember = new Map(clusterOverlapping(raw)
 *                         .flatMap((c) => c.map((m) => [m, c] as const)));
 *                       for (const w of mergeFindings(raw)) byMember.get(w); // its cluster
 *
 * The reference-identity tests exist for the second form: a defensive clone
 * anywhere in here turns every Map lookup into a miss, silently. If that glue
 * grows past roughly five lines in Task 12, fold a combined export back into
 * this module rather than letting the two-call dance spread to more callers.
 *
 * Both comparators assume `confidence` is a finite number. A NaN makes every
 * comparison false, which makes Array#sort's ordering inconsistent and quietly
 * destroys the determinism everything below depends on -- so tier adapters must
 * validate confidence at the boundary, as types.ts already requires them to do
 * for span text.
 */

const SEVERITY_RANK: Record<Severity, number> = { low: 0, medium: 1, high: 2, critical: 3 };

/**
 * Spans are half-open, so touching is NOT overlapping: [0,5) and [5,10) name
 * disjoint characters and both survive.
 *
 * Tier 0 never emits a zero-width span, but this predicate is the only
 * definition of overlap in the module and everything else must agree with the
 * answer it gives for one: [5,5) overlaps a span that strictly contains offset
 * 5, and overlaps nothing else -- not another zero-width span, not a span it
 * merely abuts. Do not "simplify" a degenerate span into its own cluster: that
 * splits a genuine overlap across two clusters and breaks the partition.
 */
function overlaps(a: Finding, b: Finding): boolean {
  return a.start < b.end && b.start < a.end;
}

/**
 * A total order over findings that never consults input position.
 *
 * tier0 documents that its order among equal-start findings is an artifact of
 * rule declaration order and explicitly NOT a contract, and tiers 1-2 arrive in
 * whatever order a model emits. So every tiebreak here reads the findings' own
 * fields: two findings that compare 0 are equal in every field that survives
 * into the output (`text` is a function of the span), which makes the merge
 * result identical for any permutation of the same input set.
 */
function compareCanonical(a: Finding, b: Finding): number {
  if (a.start !== b.start) return a.start - b.start;
  if (a.end !== b.end) return a.end - b.end;
  if (a.source !== b.source) return a.source < b.source ? -1 : 1;
  if (a.entityType !== b.entityType) return a.entityType < b.entityType ? -1 : 1;
  if (a.tier !== b.tier) return a.tier - b.tier;
  const sev = SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity];
  if (sev !== 0) return sev;
  return b.confidence - a.confidence;
}

/** Strongest first: severity, then confidence, then width, then canonical order. */
function comparePriority(a: Finding, b: Finding): number {
  const sev = SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity];
  if (sev !== 0) return sev;
  if (a.confidence !== b.confidence) return b.confidence - a.confidence;
  const width = b.end - b.start - (a.end - a.start);
  if (width !== 0) return width;
  return compareCanonical(a, b);
}

/**
 * Group findings into transitive overlap clusters: A overlaps B and B overlaps
 * C puts all three in one cluster even when A and C are disjoint. Clusters
 * partition the input (every finding lands in exactly one), are ordered by
 * start, and their members are in canonical order.
 *
 * Exported for Task 12, which must resolve ACTIONS per cluster rather than per
 * merge winner: a critical entropy finding whose action is `redact` can win the
 * merge over a high-severity PAN inside it whose action is `block`, and taking
 * the winner's action alone would silently downgrade the message. The strictest
 * action among a cluster is the only safe answer, and only the cluster still
 * has the losers to look at.
 */
export function clusterOverlapping(findings: Finding[]): Finding[][] {
  const sorted = [...findings].sort(compareCanonical);
  const clusters: Finding[][] = [];
  let current: Finding[] | undefined;
  // The cluster's rightmost edge, which is not necessarily its last member's:
  // sorting by start means a wide span can be followed by one that ends sooner.
  let maxEnd = Number.NEGATIVE_INFINITY;
  for (const f of sorted) {
    // Sorted by start, so every member so far starts at or before f: if f
    // starts at or after the cluster's rightmost edge it can overlap none of
    // them, and if it starts before, it overlaps the member that set that edge.
    // (Zero-width spans included: canonical order puts [10,10) ahead of [10,20)
    // on the end tiebreak, which is exactly when the two do not overlap.)
    if (current === undefined || f.start >= maxEnd) {
      current = [f];
      clusters.push(current);
      maxEnd = f.end;
    } else {
      current.push(f);
      if (f.end > maxEnd) maxEnd = f.end;
    }
  }
  return clusters;
}

/**
 * Resolve overlaps, returning a non-overlapping set sorted by start. Pure: the
 * input array and its findings are never mutated, and every returned finding is
 * an input finding unchanged.
 *
 * Resolution is per CLUSTER, not pairwise down a chain. Within a cluster the
 * strongest remaining finding is kept, only the findings overlapping *it* are
 * discarded, and the rest are reconsidered. Pairwise chaining against the
 * last-kept finding instead loses coverage transitively: an entropy run [0,45)
 * that beats a weak hit at [5,15) and is then itself replaced by a stronger hit
 * at [30,40) leaves [5,15) covered by nothing, even though the finding that
 * justified dropping it is gone. Here the weak hit simply survives, because it
 * never overlapped the winner that displaced the run.
 *
 * The winner keeps its OWN span; spans are never widened to the loser's bounds.
 * For the containing-entropy case ("SECRET_TOKEN=<secret>" scored as one run
 * versus a regex matching just the secret) the excluded remainder is the key
 * NAME, which is not sensitive and does not want redacting.
 *
 * That sentence used to continue "if some other rule does consider that
 * remainder sensitive, it emits its own finding for it and that finding is
 * resolved on its own merits", and it is WRONG as a guarantee. It holds only
 * for a finding DISJOINT from the winner. A rule that covers the remainder AND
 * the winner produces a finding that overlaps the winner, and this loop
 * discards it -- so the remainder is left uncovered by the very finding that
 * was supposed to rescue it.
 *
 * MEASURED, on the shipped `policies/compiled/p-fin.ir.json` with no IR edit,
 * through the real `resolveFindings` and `applyActions`. Message: "Please
 * summarise the renewal terms we agreed with Tamarind Grocers Pvt Ltd
 * yesterday." A tier-1 `client-name` finding at [50,74) "Tamarind Grocers Pvt
 * Ltd" at confidence 0.78 meets a tier-2 `pred:client-relationship-disclosure`
 * finding at 0.9. Both are severity `high`, so confidence decides, and a model
 * confidence routinely beats a GLiNER score. The tier-1 finding is discarded in
 * every row; what survives is the tier-2 span alone:
 *
 *   [35,84)  "Please summarise the renewal terms [REDACTED:pred:...]."
 *   [50,66)  "...we agreed with [REDACTED:pred:...] Pvt Ltd yesterday."
 *   [50,58)  "...we agreed with [REDACTED:pred:...] Grocers Pvt Ltd yesterday."
 *
 * `blocked` is false in all three. The first is the whole evidence clause,
 * which is what tier 2 emitted before `packages/tier2/src/spans.ts` split the
 * clause that LOCATES a finding from the span an action REWRITES: it destroys
 * the message and covers the name. The other two are what tier 2 emits now,
 * and each ships part of the client name the discarded tier-1 finding covered
 * in full. So the split reversed this trade's failure direction, from
 * over-covering to under-covering, and moved the residual from "a key name" to
 * "the rest of the entity the loser was about". The pricing above no longer
 * covers the reachable case.
 *
 * NOT changed here, deliberately. Every fix re-prices tier-0 and tier-1
 * behaviour that the two-span change is not about, and two of the three are
 * worse than the leak: clipping the loser to its uncovered remainder, or
 * widening the winner to the union, both produce a finding whose entityType is
 * a claim about a span its detector never asserted -- and on a `pseudonymize`
 * entityType `applyActions` then mints that partial value into the vault, the
 * defect its own span-fidelity guard exists to prevent. Preferring the
 * containing span over the contained one is the third, and it is the deliberate
 * decision `keeps the narrow higher-confidence finding when a wider span
 * strictly contains it` pins. `applyActions` records the same residual as a
 * parked non-feature and names its own revisit trigger; this measurement is
 * that trigger, and the choice belongs to whoever takes it, not to a fix round
 * on tier 2.
 */
export function mergeFindings(findings: Finding[]): Finding[] {
  const kept: Finding[] = [];
  for (const cluster of clusterOverlapping(findings)) {
    // Singletons are the common case (most findings overlap nothing); skip the
    // sort and the scan for them.
    if (cluster.length === 1) {
      kept.push(cluster[0]!);
      continue;
    }
    // Strongest first, keeping whatever does not collide with an already-kept
    // winner: equivalent to "take the best, discard only what overlaps IT,
    // repeat", without rescanning the cluster on every round.
    const winners: Finding[] = [];
    for (const candidate of [...cluster].sort(comparePriority)) {
      if (!winners.some((w) => overlaps(w, candidate))) winners.push(candidate);
    }
    kept.push(...winners);
  }
  return kept.sort(compareCanonical);
}
