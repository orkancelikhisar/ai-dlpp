import { describe, expect, it } from "vitest";
import {
  UNCERTAIN_BELOW,
  selectSegments,
  uncertainSegmentStarts,
} from "../../src/detect/escalate.js";
import { runTier0 } from "../../src/detect/tier0.js";
import { loadPolicyIr } from "../../src/policy/load.js";
import { segmentText, type Segment, type SegmentKind } from "../../src/segment/segment.js";
import type { Finding } from "../../src/detect/types.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

/**
 * Segments are built by hand here rather than by `segmentText`, so a test says
 * what shape it needs instead of depending on the segmenter's classification
 * rules. The tiling invariant (contiguous, gap-free) is `segmentText`'s to keep
 * and `selectSegments` never assumes it -- but `start` being UNIQUE per segment
 * is what `uncertain` addresses a segment by, so these lay out end-to-end.
 */
const tile = (...spec: Array<[SegmentKind, string]>): Segment[] => {
  const out: Segment[] = [];
  let at = 0;
  for (const [kind, text] of spec) {
    out.push({ kind, text, start: at, end: at + text.length });
    at += text.length;
  }
  return out;
};

const kinds = (segments: Segment[]) => segments.map((s) => s.kind);
const starts = (segments: Segment[]) => segments.map((s) => s.start);

const NOTHING = { hasPredicates: false, uncertain: [] } as const;

describe("selectSegments", () => {
  it("selects every non-code segment when the policy has predicates", () => {
    // A semantic predicate exists because no pattern can express it, so the
    // natural-language segments are the ones worth a judge.
    const segs = tile(["prose", "Acme is our client. "], ["kv", "owner: acme\n"], ["prose", "Ship it."]);
    expect(selectSegments(segs, { hasPredicates: true, uncertain: [] })).toEqual(segs);
  });

  it("skips code segments on the predicate branch, and keeps kv", () => {
    // BOTH halves are load-bearing. With only prose and code in the fixture,
    // `kind === "prose"` and `kind !== "code"` select the same thing, so the kv
    // segment is what makes this test able to tell them apart -- and kv is the
    // half the spec's wording ("prose segments") does not obviously cover. See
    // escalate.ts for why kv is in.
    const segs = tile(["code", "const k = 1;\n"], ["prose", "Acme is our client. "], ["kv", "owner: acme\n"]);
    expect(kinds(selectSegments(segs, { hasPredicates: true, uncertain: [] }))).toEqual(["prose", "kv"]);
  });

  it("selects nothing when there are no predicates and nothing is uncertain", () => {
    // The expensive default: without it every message pays a judge call per
    // segment to ask a model about a policy with no semantic clauses.
    const segs = tile(["prose", "Acme is our client. "], ["kv", "owner: acme\n"], ["code", "x=1;"]);
    expect(selectSegments(segs, NOTHING)).toEqual([]);
  });

  it("selects a segment flagged uncertain even with no predicates", () => {
    const segs = tile(["prose", "Acme is our client. "], ["prose", "Second paragraph."]);
    expect(starts(selectSegments(segs, { hasPredicates: false, uncertain: [segs[1]!.start] }))).toEqual([
      segs[1]!.start,
    ]);
  });

  it("selects a CODE segment that is uncertain, which the predicate branch would have skipped", () => {
    // The two branches are a union, not a filter chain, and the code exclusion
    // belongs to the predicate branch alone. Spec 4.1 attaches "prose" to the
    // predicate half only ("those always require tier 2 on prose segments") and
    // leaves the uncertainty half unqualified; the exclusion's own rationale is
    // about scanning EVERY code segment, which an uncertainty flag is not.
    // Without this, tier 0's entropy findings -- which only ever land in code
    // and kv -- could never escalate anything from a fenced block.
    const segs = tile(["prose", "Acme is our client. "], ["code", "const k = 'zzz';\n"]);
    expect(kinds(selectSegments(segs, { hasPredicates: true, uncertain: [segs[1]!.start] }))).toEqual([
      "prose",
      "code",
    ]);
    expect(kinds(selectSegments(segs, { hasPredicates: false, uncertain: [segs[1]!.start] }))).toEqual(["code"]);
  });

  it("never returns the same segment twice", () => {
    // A segment both uncertain and prose-with-predicates must be judged once.
    // Twice doubles the cost and produces duplicate findings that read as model
    // behaviour rather than as an escalation bug.
    const segs = tile(["prose", "Acme is our client."]);
    expect(selectSegments(segs, { hasPredicates: true, uncertain: [0] })).toHaveLength(1);
  });

  it("returns segments in input order however the two branches interleave", () => {
    // Findings carry absolute offsets, so order is not needed for correctness --
    // it is needed because a judge spends its budget in the order it is handed,
    // and a run cut short by the message deadline keeps a PREFIX. A shuffled
    // list makes which segments survive a truncated run unpredictable.
    const segs = tile(
      ["code", "aa\n"],
      ["prose", "bbbb "],
      ["code", "cc\n"],
      ["kv", "d: 1\n"],
      ["code", "ee\n"],
    );
    const picked = selectSegments(segs, {
      hasPredicates: true,
      uncertain: [segs[4]!.start, segs[0]!.start],
    });
    expect(starts(picked)).toEqual([segs[0]!.start, segs[1]!.start, segs[3]!.start, segs[4]!.start]);
  });

  it("hands back the caller's own Segment objects, not rebuilt ones", () => {
    // Offsets and text must stay exactly what the segmenter produced: every
    // finding a judge returns is rebased by `segment.start`, so a segment
    // reconstructed here is a chance for the two to disagree about where the
    // segment is.
    const segs = tile(["prose", "Acme is our client."]);
    expect(selectSegments(segs, { hasPredicates: true, uncertain: [] })[0]).toBe(segs[0]);
  });

  it("does not mutate the segments it was given", () => {
    const segs = tile(["code", "aa\n"], ["prose", "bbbb "]);
    const before = structuredClone(segs);
    selectSegments(segs, { hasPredicates: true, uncertain: [segs[0]!.start] });
    expect(segs).toEqual(before);
  });

  it("throws when an uncertain offset addresses no segment", () => {
    // `uncertain` is a list of segment STARTS, and the mistake it invites is
    // passing finding starts instead. That mistake selects nothing and looks
    // exactly like a message with no uncertainty, so it has to be loud.
    const segs = tile(["prose", "Acme is our client."]);
    expect(() => selectSegments(segs, { hasPredicates: false, uncertain: [7] })).toThrow(/no segment/i);
    expect(() => selectSegments(segs, { hasPredicates: false, uncertain: [7] })).toThrow("7");
  });

  it("selects nothing from an empty segment list", () => {
    expect(selectSegments([], { hasPredicates: true, uncertain: [] })).toEqual([]);
  });
});

const finding = (over: Partial<Finding> & Pick<Finding, "start" | "end" | "confidence">): Finding => ({
  text: "",
  entityType: "client-name",
  severity: "high",
  tier: 1,
  source: "stub",
  ...over,
});

describe("uncertainSegmentStarts", () => {
  const segs = tile(["prose", "0123456789"], ["kv", "k: v\n"], ["code", "code here"]);

  it("flags the segment holding a finding below the threshold", () => {
    const out = uncertainSegmentStarts(segs, [finding({ start: 2, end: 5, confidence: 0.5 })]);
    expect(out).toEqual([segs[0]!.start]);
  });

  it("does not flag a segment whose findings are all at or above the threshold", () => {
    const out = uncertainSegmentStarts(segs, [finding({ start: 2, end: 5, confidence: 0.95 })]);
    expect(out).toEqual([]);
  });

  it("is exclusive at the threshold", () => {
    // `< below`, not `<= below`. Stated as its own test because the boundary is
    // where a tier's fixed confidence sits: tier 0's regex floor is exactly 0.9
    // and its entropy findings are exactly 0.7, so an off-by-one-comparison
    // moves a whole class of findings across the line at once.
    expect(uncertainSegmentStarts(segs, [finding({ start: 0, end: 3, confidence: 0.6 })], 0.6)).toEqual([]);
    expect(uncertainSegmentStarts(segs, [finding({ start: 0, end: 3, confidence: 0.6 })], 0.61)).toEqual([
      segs[0]!.start,
    ]);
  });

  it("reads the threshold it is given rather than the default", () => {
    // The same finding, two thresholds, opposite answers. A hardcoded 0.8
    // survives every other test in this block, which is the failure this
    // project has shipped twice.
    const priors = [finding({ start: 0, end: 3, confidence: 0.7 })];
    expect(uncertainSegmentStarts(segs, priors, 0.9)).toEqual([segs[0]!.start]);
    expect(uncertainSegmentStarts(segs, priors, 0.5)).toEqual([]);
  });

  it("flags EVERY segment a finding overlaps", () => {
    // Tier 0's regex rules scan the whole message, not a segment, so a match
    // can straddle a boundary. Attributing it to one segment by `start` alone
    // leaves the other half unjudged.
    const straddles = finding({ start: 8, end: 12, confidence: 0.4 });
    expect(uncertainSegmentStarts(segs, [straddles])).toEqual([segs[0]!.start, segs[1]!.start]);
  });

  it("does not flag a segment a finding ends exactly at the START of", () => {
    // The RIGHT edge of the half-open overlap test, which was the unasserted
    // one: the mirror mutation on the left (`f.start < segment.end` relaxed to
    // `<=`) was already killed, so this was an asymmetry rather than a
    // deliberate omission. A finding covering [2, 10) of a segment running
    // [0, 10) reaches the next segment's first offset without covering any of
    // it, and `f.end > segment.start` relaxed to `>=` would escalate that
    // segment too.
    //
    // Not a hypothetical shape: tier 0's regex rules scan the whole message
    // rather than a segment and segment boundaries are newlines, so a finding
    // that stops on a boundary is ordinary. At the 4.6 s per tier-2 call this
    // plan measured, one unneeded segment spends 92% of the fixtures' 5000 ms
    // `latencyBudgetMs`.
    const ends_on_boundary = finding({ start: 2, end: segs[1]!.start, confidence: 0.5 });
    expect(uncertainSegmentStarts(segs, [ends_on_boundary])).toEqual([segs[0]!.start]);
  });

  it("does not flag a segment a finding starts exactly at the END of", () => {
    // The left edge, stated beside the right one so the two are read together
    // rather than one of them being covered by accident.
    const starts_on_boundary = finding({ start: segs[0]!.end, end: segs[1]!.end, confidence: 0.5 });
    expect(uncertainSegmentStarts(segs, [starts_on_boundary])).toEqual([segs[1]!.start]);
  });

  it("names each segment once however many uncertain findings it holds", () => {
    const priors = [
      finding({ start: 0, end: 2, confidence: 0.4 }),
      finding({ start: 4, end: 6, confidence: 0.3 }),
    ];
    expect(uncertainSegmentStarts(segs, priors)).toEqual([segs[0]!.start]);
  });

  it("returns starts in segment order, not in finding order", () => {
    const priors = [
      finding({ start: segs[2]!.start, end: segs[2]!.start + 2, confidence: 0.4 }),
      finding({ start: 0, end: 2, confidence: 0.4 }),
    ];
    expect(uncertainSegmentStarts(segs, priors)).toEqual([segs[0]!.start, segs[2]!.start]);
  });

  it("returns nothing when there are no prior findings", () => {
    expect(uncertainSegmentStarts(segs, [])).toEqual([]);
  });

  it("refuses a threshold that is not a real number in 0..1", () => {
    // NaN is the dangerous one: `confidence < NaN` is false for every finding,
    // so an unvalidated NaN disables the uncertainty branch silently and the
    // arm reports "nothing was uncertain" for every message it ever saw.
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -0.1, 1.1]) {
      expect(() => uncertainSegmentStarts(segs, [], bad)).toThrow(/threshold/i);
    }
    // Both ends of the legal range are usable: 0 disables the branch on
    // purpose, 1 makes every finding short of certainty escalate.
    expect(() => uncertainSegmentStarts(segs, [], 0)).not.toThrow();
    expect(() => uncertainSegmentStarts(segs, [], 1)).not.toThrow();
  });
});

describe("UNCERTAIN_BELOW against the confidences this pipeline really emits", () => {
  /**
   * The default threshold is only meaningful against real values, and every
   * number below is READ OFF `runTier0`, never typed into the test. A default
   * moved to 0.6 (under tier 0's entropy confidence) or to 0.95 (over its regex
   * floor) passes every threshold test above and fails here.
   */
  const ir = loadPolicyIr(JSON.stringify(minimalIr()));
  const MESSAGE = [
    "Hey, my PAN is ABCPD1234E for the tax form.",
    "```",
    "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
    "session = x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ",
    "```",
  ].join("\n");
  const segments = segmentText(MESSAGE);
  const tier0 = runTier0(ir, MESSAGE, segments);
  const byType = new Map(tier0.map((f) => [f.entityType, f]));

  it("premise: this message really does carry an entropy hit and a regex hit", () => {
    expect(byType.get("generic-secret")).toBeDefined();
    expect(byType.get("in-pan")).toBeDefined();
    expect(byType.get("aws-key")).toBeDefined();
  });

  it("treats tier 0's entropy findings as uncertain", () => {
    // tier0.ts fixes entropy at a confidence below every regex rule's floor and
    // says why in as many words: entropy knows a string LOOKS random, never
    // that it is a secret, and "raising the confidence is tier 1-2's job".
    // That is the finding class this branch exists to escalate.
    const entropy = byType.get("generic-secret")!;
    expect(entropy.confidence).toBeLessThan(UNCERTAIN_BELOW);
    expect(uncertainSegmentStarts(segments, [entropy])).not.toEqual([]);
  });

  it("treats tier 0's validated regex findings as certain", () => {
    // A PAN that cleared its structural validator and an AWS key id are not
    // things a 4.6-second judge call can improve on.
    for (const id of ["in-pan", "aws-key"]) {
      expect(byType.get(id)!.confidence).toBeGreaterThanOrEqual(UNCERTAIN_BELOW);
    }
    expect(uncertainSegmentStarts(segments, [byType.get("in-pan")!, byType.get("aws-key")!])).toEqual([]);
  });

  it("sits clear of both tier-0 confidences rather than on either boundary", () => {
    // Neither of tier 0's two values is within 0.05 of the threshold, so a
    // later tweak to CONTEXT_BONUS or ENTROPY_CONFIDENCE does not silently move
    // a class of findings across the line.
    for (const f of tier0) {
      expect(Math.abs(f.confidence - UNCERTAIN_BELOW)).toBeGreaterThan(0.05);
    }
  });
});
