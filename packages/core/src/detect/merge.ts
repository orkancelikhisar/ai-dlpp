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
 * NAME, which is not sensitive and does not want redacting. If some other rule
 * does consider that remainder sensitive, it emits its own finding for it and
 * that finding is resolved on its own merits.
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
