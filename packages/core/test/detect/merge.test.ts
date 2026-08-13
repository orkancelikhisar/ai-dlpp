import { describe, expect, it } from "vitest";
import { clusterOverlapping, mergeFindings } from "../../src/detect/merge.js";
import type { Finding } from "../../src/detect/types.js";

const f = (over: Partial<Finding>): Finding => ({
  start: 0, end: 10, text: "0123456789", entityType: "x", severity: "low",
  tier: 0, source: "t", confidence: 0.9, ...over,
});

/** A finding whose `text` is consistent with its span, as producers guarantee. */
const span = (start: number, end: number, over: Partial<Finding> = {}): Finding =>
  f({ start, end, text: "x".repeat(end - start), ...over });

describe("mergeFindings", () => {
  it("keeps non-overlapping findings, sorted by start", () => {
    const out = mergeFindings([f({ start: 20, end: 25, text: "aaaaa" }), f({ start: 0, end: 5, text: "bbbbb" })]);
    expect(out.map((x) => x.start)).toEqual([0, 20]);
  });

  it("keeps the higher-severity finding on overlap", () => {
    const out = mergeFindings([
      f({ start: 0, end: 10, severity: "low" }),
      f({ start: 5, end: 15, severity: "critical", text: "abcdefghij" }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0]!.severity).toBe("critical");
  });

  it("breaks severity ties by confidence", () => {
    const out = mergeFindings([
      f({ start: 0, end: 10, confidence: 0.7 }),
      f({ start: 5, end: 15, confidence: 0.95, text: "abcdefghij" }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0]!.confidence).toBe(0.95);
  });

  it("touching spans (end === start) do not overlap", () => {
    const out = mergeFindings([f({ start: 0, end: 5, text: "aaaaa" }), f({ start: 5, end: 10, text: "bbbbb" })]);
    expect(out).toHaveLength(2);
  });

  it("returns an empty array for no findings", () => {
    expect(mergeFindings([])).toEqual([]);
  });

  it("does not mutate or alias its input", () => {
    const input = [span(5, 15, { confidence: 0.95 }), span(0, 10)];
    const snapshot = structuredClone(input);
    mergeFindings(input);
    expect(input).toEqual(snapshot);
  });

  it("breaks confidence ties by the wider span", () => {
    const out = mergeFindings([span(0, 20), span(5, 15)]);
    expect(out).toHaveLength(1);
    expect([out[0]!.start, out[0]!.end]).toEqual([0, 20]);
  });

  // Severity dominates outright: it beats confidence AND width at once, which
  // no other test pins (they hold the weaker keys equal).
  it("prefers a narrow low-confidence critical over a wide high-confidence low", () => {
    const critical = span(10, 14, { severity: "critical", confidence: 0.6, source: "aadhaar-rule" });
    const low = span(0, 40, { severity: "low", confidence: 0.99, source: "entropy-rule" });
    const out = mergeFindings([critical, low]);
    expect(out.map((x) => x.source)).toEqual(["aadhaar-rule"]);
  });

  // Winners are the SAME OBJECTS as the inputs, never copies: Task 12 joins
  // merge winners back to their clusters through an identity Map, and a defensive
  // clone here would silently turn every lookup into a miss.
  it("returns input findings by reference, not copies", () => {
    const wide = span(0, 45, { source: "entropy-rule", confidence: 0.7 });
    const narrow = span(5, 15, { source: "pan-rule", confidence: 0.9 });
    const far = span(60, 70, { source: "jwt-rule" });
    const out = mergeFindings([wide, narrow, far]);
    expect(out[0]).toBe(narrow);
    expect(out[1]).toBe(far);
  });

  // Degenerate spans tier 0 never emits. The predicate says two zero-width spans
  // at the same offset do not overlap, so merge keeps both; pinned so that any
  // future change here is a deliberate one.
  it("keeps both of two identical zero-width findings at the same offset", () => {
    const a = f({ start: 7, end: 7, text: "", source: "rule-a" });
    const b = f({ start: 7, end: 7, text: "", source: "rule-a" });
    expect(mergeFindings([a, b])).toHaveLength(2);
  });

  // (b) containment: entropy fires on the whole "SECRET_TOKEN=<secret>" run at
  // 0.7 while a regex rule matches just the secret inside it at 0.9. Equal
  // severity, so confidence decides -- and the winner keeps its OWN narrow span.
  it("keeps the narrow higher-confidence finding when a wider span strictly contains it", () => {
    const entropy = span(0, 45, { source: "entropy-rule", confidence: 0.7, entityType: "secret" });
    const regex = span(13, 41, { source: "token-rule", confidence: 0.9, entityType: "api-key" });
    const out = mergeFindings([entropy, regex]);
    expect(out).toHaveLength(1);
    expect(out[0]!.source).toBe("token-rule");
    // Span stays NARROW: no widening to the loser's bounds.
    expect([out[0]!.start, out[0]!.end]).toEqual([13, 41]);
  });

  // (c) cluster semantics: a wide finding that loses to one narrow finding must
  // not drag a second, disjoint narrow finding out of the result with it.
  it("keeps both narrow findings when one wide finding contains two disjoint ones", () => {
    const entropy = span(0, 45, { source: "entropy-rule", confidence: 0.7 });
    const a = span(5, 15, { source: "pan-rule", confidence: 0.9 });
    const b = span(30, 40, { source: "aws-rule", confidence: 0.9 });
    const out = mergeFindings([entropy, a, b]);
    expect(out.map((x) => [x.start, x.end])).toEqual([
      [5, 15],
      [30, 40],
    ]);
  });

  // The same requirement where pairwise chaining actually breaks: the wide
  // finding beats the first narrow one and is then itself displaced by a
  // stronger later one. Chaining against the last-kept finding leaves [5,15)
  // covered by nothing; per-cluster resolution keeps it, since it never
  // overlapped the finding that displaced the run.
  it("does not chain-drop a finding whose only stronger rival was itself discarded", () => {
    const entropy = span(0, 45, { source: "entropy-rule", confidence: 0.7 });
    const weak = span(5, 15, { source: "weak-rule", confidence: 0.6 });
    const strong = span(30, 40, { source: "aws-rule", confidence: 0.9 });
    const out = mergeFindings([entropy, weak, strong]);
    expect(out.map((x) => x.source)).toEqual(["weak-rule", "aws-rule"]);
  });

  // (a) tier-0 explicitly does NOT promise an order among equal-start findings,
  // so resolution may not depend on input order at all.
  it("resolves fully tied findings deterministically, independent of input order", () => {
    const a = f({ start: 0, end: 10, source: "rule-a", entityType: "aaa" });
    const b = f({ start: 0, end: 10, source: "rule-b", entityType: "bbb" });
    const forward = mergeFindings([a, b]);
    const reversed = mergeFindings([b, a]);
    expect(forward).toHaveLength(1);
    expect(forward).toEqual(reversed);
    expect(forward[0]!.source).toBe("rule-a");
  });

  it("produces the same result for any permutation of a mixed input", () => {
    const set = [
      span(0, 45, { source: "entropy-rule", confidence: 0.7 }),
      span(5, 15, { source: "pan-rule", confidence: 0.9 }),
      span(12, 20, { source: "aadhaar-rule", confidence: 0.9, severity: "critical" }),
      span(30, 40, { source: "aws-rule", confidence: 0.9 }),
      span(60, 70, { source: "jwt-rule", confidence: 0.8 }),
    ];
    const base = mergeFindings(set);
    expect(mergeFindings([...set].reverse())).toEqual(base);
    expect(mergeFindings([set[3]!, set[0]!, set[4]!, set[2]!, set[1]!])).toEqual(base);
  });
});

describe("clusterOverlapping", () => {
  it("returns no clusters for no findings", () => {
    expect(clusterOverlapping([])).toEqual([]);
  });

  it("groups transitively overlapping findings into one cluster", () => {
    // A overlaps B, B overlaps C, but A and C are disjoint.
    const out = clusterOverlapping([span(0, 10), span(8, 20), span(15, 25)]);
    expect(out).toHaveLength(1);
    expect(out[0]!.map((x) => x.start)).toEqual([0, 8, 15]);
  });

  it("splits disjoint and merely touching findings into separate clusters", () => {
    const out = clusterOverlapping([span(0, 5), span(5, 10), span(20, 25)]);
    expect(out.map((c) => c.map((x) => x.start))).toEqual([[0], [5], [20]]);
  });

  it("partitions the input regardless of order", () => {
    const set = [span(0, 45), span(5, 15), span(30, 40), span(60, 70), span(70, 75)];
    const base = clusterOverlapping(set);
    expect(clusterOverlapping([...set].reverse())).toEqual(base);
    expect(base.flat()).toHaveLength(set.length);
    expect(base.map((c) => c.length)).toEqual([3, 1, 1]);
  });

  // (d) Task 12 resolves ACTIONS over a whole cluster: the merge winner alone
  // cannot express "redact the entropy hit but block the PAN inside it".
  it("keeps every member of a cluster the merge winner would have discarded", () => {
    const entropy = span(0, 45, { source: "entropy-rule", severity: "critical", confidence: 0.7 });
    const pan = span(5, 15, { source: "pan-rule", severity: "high", confidence: 0.95 });
    const clusters = clusterOverlapping([entropy, pan]);
    expect(clusters).toHaveLength(1);
    expect(clusters[0]!.map((x) => x.source)).toEqual(["entropy-rule", "pan-rule"]);
    // By reference: Task 12 keys a Map on cluster members to find the merge
    // winner's cluster, so cluster members must be the caller's own objects.
    expect(clusters[0]![0]).toBe(entropy);
    expect(clusters[0]![1]).toBe(pan);
    // Merge keeps the critical entropy finding only; the PAN survives in the cluster.
    const merged = mergeFindings([entropy, pan]);
    expect(merged.map((x) => x.source)).toEqual(["entropy-rule"]);
  });

  it("does not mutate or alias its input", () => {
    const input = [span(5, 15), span(0, 10)];
    const snapshot = structuredClone(input);
    clusterOverlapping(input);
    expect(input).toEqual(snapshot);
  });

  // Degenerate input tier 0 never emits, pinned because clustering must agree
  // with the overlap predicate whatever it answers: a.start < b.end &&
  // b.start < a.end makes [10,10) overlap a span strictly containing offset 10,
  // and overlap nothing it merely abuts.
  it("clusters a zero-width span exactly as the overlap predicate says", () => {
    const inside = clusterOverlapping([span(0, 20, { source: "wide" }), f({ start: 10, end: 10, text: "" })]);
    expect(inside).toHaveLength(1);
    expect(inside[0]).toHaveLength(2);

    const abutting = clusterOverlapping([span(0, 10, { source: "wide" }), f({ start: 10, end: 10, text: "" })]);
    expect(abutting.map((c) => c.length)).toEqual([1, 1]);
  });
});
