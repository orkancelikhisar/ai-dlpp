import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Finding } from "@sih/core";
import { loadCorpus, type CorpusItem } from "../src/driver/corpus.js";
import { percentile, segmentSizeDistribution } from "../src/driver/segments.js";

const FIXTURES = join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures");

/** Prose (14) + fenced code (32) + prose (9) = 55 characters; the fence starts at 14. */
const FENCED = "Look at this:\n```\nsecret = hunter2hunter2\n```\nAny idea?";

function item(id: string, text: string): CorpusItem {
  return { id, text, policy: "segments-test", gold: [] };
}

/** `k` single-character words: `2k - 1` characters, `k` words. */
function words(k: number): string {
  return Array.from({ length: k }, () => "x").join(" ");
}

/** A tier-0 entropy finding inside FENCED's code block, at entropy's fixed 0.7. */
const UNCERTAIN_IN_FENCE: Finding = {
  start: 18,
  end: 40,
  text: "secret = hunter2hunter2",
  entityType: "generic-secret",
  severity: "critical",
  tier: 0,
  source: "entropy-rule",
  confidence: 0.7,
};

describe("percentile", () => {
  // Every expectation below is hand-computed from the nearest-rank definition
  // (rank = ceil(percent * n / 100); the answer is the rank-th smallest value)
  // and written out as a literal. None was read off this module's output.
  it("returns an observed value at the nearest rank, not an interpolated one", () => {
    // 1..100, rotated so the input is not already sorted. The three candidate
    // answers are all distinct here, which is why this sample was chosen:
    //   nearest-rank  ceil(50 * 100 / 100) = 50th smallest  -> 50   (correct)
    //   linear interp between the 50th and the 51st         -> 50.5
    //   floor-rank    floor(0.5 * 100) read 0-based         -> 51
    const oneToHundred = Array.from({ length: 100 }, (_, i) => i + 1);
    const rotated = [...oneToHundred.slice(37), ...oneToHundred.slice(0, 37)];
    expect(percentile(rotated, 50)).toBe(50);
    expect(percentile(rotated, 95)).toBe(95);
    expect(percentile(rotated, 100)).toBe(100);
    expect(percentile(rotated, 1)).toBe(1);
  });

  it("orders numerically, not the way Array.prototype.sort does by default", () => {
    // The default comparator sorts by string, so [2, 10, 9] becomes [10, 2, 9]
    // and rank 2 reads 2 rather than 9. This is the cheapest way for percentile
    // code to return a plausible wrong number, and character counts are exactly
    // the multi-digit range where it bites.
    expect(percentile([2, 10, 9], 50)).toBe(9);
    expect(percentile([2, 10, 9], 100)).toBe(10);
  });

  it("returns the value at the rank, not the rank", () => {
    // Values deliberately not equal to their ranks. n = 20:
    //   p50 -> ceil(50 * 20 / 100) = rank 10 -> 100
    //   p95 -> ceil(95 * 20 / 100) = rank 19 -> 190
    // A floor-rank off-by-one gives 110 and 200; returning the index instead of
    // the value gives 9 and 18.
    const tens = Array.from({ length: 20 }, (_, i) => (i + 1) * 10);
    expect(percentile(tens, 50)).toBe(100);
    expect(percentile(tens, 95)).toBe(190);
  });

  it("saturates at the maximum once n is small enough that no rank sits above it", () => {
    // n = 17 is the smoke corpus's selected-segment count, and this is how that
    // number has to be read: ceil(95 * 17 / 100) = ceil(16.15) = 17, the last
    // rank, so p95 IS the maximum and says nothing the maximum does not. That
    // holds for every sample below n = 20, not just for this one.
    const seventeen = [5, 3, 9, 1, 8, 2, 7, 4, 6, 15, 11, 13, 10, 14, 12, 17, 16];
    expect(seventeen).toHaveLength(17);
    expect(percentile(seventeen, 95)).toBe(17);
    expect(percentile(seventeen, 95)).toBe(Math.max(...seventeen));
    // p50 is still a genuine middle: ceil(50 * 17 / 100) = rank 9 -> 9.
    expect(percentile(seventeen, 50)).toBe(9);
  });

  it("leaves the caller's array in the order it was given", () => {
    // The docstring promises it ("the array is not mutated") and the inline
    // comment gives the mechanism ("`[...]` because sorting the caller's array
    // in place would reorder `samples.chars` under whoever holds it"), and
    // nothing tested either: dropping the copy survived the suite, because
    // every test that read a sample either sorted a copy first or compared an
    // already-sorted array. This is exported API Task 12 will call on its own
    // arrays, and an in-place sort there is a silent side effect on data the
    // caller still holds.
    const caller = [30, 10, 20];
    expect(percentile(caller, 50)).toBe(20);
    expect(caller).toEqual([30, 10, 20]);
  });

  it("refuses a sample or a percent it cannot answer for", () => {
    expect(() => percentile([], 50)).toThrow(/empty/);
    // Rank 0 does not exist under nearest-rank, and `min` is its own field.
    expect(() => percentile([1, 2], 0)).toThrow(/\(0, 100]/);
    expect(() => percentile([1, 2], 101)).toThrow(/\(0, 100]/);
    // NaN is refused by the integer clause, not the range one: `NaN > 0` is
    // false and `NaN <= 100` is false, so a range check alone would reject it
    // too -- but with a message naming the wrong problem. Both clauses matter
    // and this pins which one speaks.
    expect(() => percentile([1, 2], Number.NaN)).toThrow(/integer/);
    // A fractional percent is refused rather than rounded: the integer form is
    // what makes `percent * n / 100` land exactly on a rank boundary when the
    // true quotient is an integer, which is the whole reason there is no
    // epsilon in this arithmetic.
    expect(() => percentile([1, 2], 97.5)).toThrow(/integer/);
  });
});

describe("segmentSizeDistribution", () => {
  it("reports percentiles over a hand-built corpus whose segments are known", () => {
    // Four items, one prose segment each, sized by construction:
    //   chars  9, 19, 29, 39     words  5, 10, 15, 20
    // Hand-computed at n = 4:
    //   p50 -> ceil(50 * 4 / 100) = rank 2 -> 19 chars / 10 words
    //   p95 -> ceil(95 * 4 / 100) = rank 4 -> 39 chars / 20 words
    const d = segmentSizeDistribution([
      item("a", words(15)),
      item("b", words(5)),
      item("c", words(20)),
      item("d", words(10)),
    ]);
    expect(d.count).toBe(4);
    expect(d.items).toBe(4);
    expect(d.chars).toEqual({ p50: 19, p95: 39, max: 39, min: 9 });
    expect(d.words).toEqual({ p50: 10, p95: 20, max: 20, min: 5 });
    // One segment per item, so every rank reads 1.
    expect(d.perItem).toEqual({ p50: 1, p95: 1, max: 1, min: 1 });
    // All three samples, because they exist so that a percentile quoted
    // somewhere else can be re-derived. Emptying `samples.words` survived the
    // first mutation round -- nothing read it -- which is exactly how a
    // provenance field stops carrying provenance without anything going red.
    expect([...d.samples.chars].sort((a, b) => a - b)).toEqual([9, 19, 29, 39]);
    expect([...d.samples.words].sort((a, b) => a - b)).toEqual([5, 10, 15, 20]);
    expect(d.samples.perItem).toEqual([1, 1, 1, 1]);
  });

  it("counts UTF-16 code units, the unit every offset in this pipeline uses", () => {
    // "ab" is 2 code units and the rocket is a surrogate pair worth 2 more. A
    // code-POINT count would report 3. corpus.ts documents why this repo is
    // UTF-16 throughout; a segment's own start/end are in that unit, so a size
    // reported in any other could not be compared against them.
    const d = segmentSizeDistribution([item("astral", "ab\u{1F680}")]);
    expect(d.chars).toEqual({ p50: 4, p95: 4, max: 4, min: 4 });
  });

  it("counts UTF-8 bytes too, because bytes and not code units ceiling a token count", () => {
    // The same string in the other unit: "a" and "b" are one byte each and the
    // rocket is FOUR, so 4 code units are 6 bytes. That gap is the whole reason
    // this field exists -- a byte-level BPE bottoms out at one token per byte,
    // so this string can become up to 6 tokens, more than the "character" count
    // an earlier version of the module header called a ceiling. Asserted on a
    // non-ASCII string on purpose: on ASCII the two fields are equal and a
    // `bytes` that just copied `chars` would pass.
    const d = segmentSizeDistribution([item("astral", "ab\u{1F680}")]);
    expect(d.bytes).toEqual({ p50: 6, p95: 6, max: 6, min: 6 });
    expect(d.samples.bytes).toEqual([6]);
  });

  it("excludes code, which the escalation policy never selects on predicates alone", () => {
    // Only the two prose segments are a judge's business, so the largest thing
    // measured is the 14-character opener and not the 32-character fence.
    // Including the fence would size the budget for a call tier 2 never makes.
    const d = segmentSizeDistribution([item("fenced", FENCED)]);
    expect(d.count).toBe(2);
    expect(d.segmentsTotal).toBe(3);
    expect(d.chars).toEqual({ p50: 9, p95: 14, max: 14, min: 9 });
  });

  it("includes a code segment an earlier tier was uncertain about", () => {
    // The other half of the escalation union. selectSegments throws on an
    // uncertain offset that starts no segment, so a miscount of the fence's
    // offset fails loudly here rather than quietly selecting less.
    const d = segmentSizeDistribution([item("fenced", FENCED)], {
      priorFindings: () => [UNCERTAIN_IN_FENCE],
    });
    expect(d.count).toBe(3);
    expect(d.chars!.max).toBe(32);
  });

  it("forwards the uncertainty threshold instead of pinning core's default", () => {
    // The SAME 0.7 finding under a threshold of 0.6: no longer uncertain, so
    // the code segment drops back out. Asserted against a non-default value on
    // purpose -- a test that only ever exercised the 0.8 default could not tell
    // a forwarded threshold from a hardcoded one.
    const at08 = segmentSizeDistribution([item("fenced", FENCED)], {
      priorFindings: () => [UNCERTAIN_IN_FENCE],
      uncertainBelow: 0.8,
    });
    const at06 = segmentSizeDistribution([item("fenced", FENCED)], {
      priorFindings: () => [UNCERTAIN_IN_FENCE],
      uncertainBelow: 0.6,
    });
    expect(at08.count).toBe(3);
    expect(at06.count).toBe(2);
    expect(at06.escalation.uncertainBelow).toBe(0.6);
  });

  it("selects nothing for a policy with no predicates and no uncertainty", () => {
    // The second config for `hasPredicates`, and not a hypothetical one: both
    // IRs under apps/eval/fixtures declare `semanticPredicates: []`, and
    // WebLlmJudge returns an empty verdict without an engine call for such a
    // policy. A budget sized under this input would be sized for zero calls.
    const d = segmentSizeDistribution([item("a", words(15)), item("b", words(5))], {
      hasPredicates: false,
    });
    expect(d.count).toBe(0);
    expect(d.items).toBe(2);
    // No segment was measured, so there is no median segment. Reporting 0 would
    // let a caller size a budget off a number nothing produced.
    expect(d.chars).toBeUndefined();
    expect(d.words).toBeUndefined();
    // `perItem` still has one sample per item, and every one of them is 0.
    expect(d.perItem).toEqual({ p50: 0, p95: 0, max: 0, min: 0 });
    // And the result says so, rather than reporting the permissive default it
    // did not run under. Stamping `hasPredicates: true` here survived the first
    // mutation round: the only test reading this field used the default.
    expect(d.escalation).toEqual({ hasPredicates: false, uncertainBelow: 0.8, hasPriors: false });
  });

  it("records the escalation input it actually ran under", () => {
    // The distribution is only readable against the policy shape that produced
    // it, and the default is the permissive one. Stating it on the result is
    // what stops a number measured under one input being quoted under another.
    expect(segmentSizeDistribution([]).escalation).toEqual({
      hasPredicates: true,
      uncertainBelow: 0.8,
      hasPriors: false,
    });
  });

  it("records whether a priors source was SUPPLIED, not whether it found anything", () => {
    // The third escalation input, and the one that separates the two conditions
    // a bake-off must never splice. It has to be about the CALLER rather than
    // about the findings: a callback returning nothing on every item is still
    // the tier-0 condition -- an IR whose rules match nothing on one corpus is a
    // measurement, not a misconfiguration -- and it produces numbers identical
    // to no callback at all. `gateReport` refuses a report whose family
    // disagrees with this flag, so getting it from the findings would refuse a
    // legitimate arm.
    const items = [item("a", words(15)), item("b", words(5))];
    const silent = segmentSizeDistribution(items, { priorFindings: () => [] });
    const absent = segmentSizeDistribution(items);
    expect(silent.escalation.hasPriors).toBe(true);
    expect(absent.escalation.hasPriors).toBe(false);
    // Identical in every measured field, which is exactly why the flag cannot be
    // derived from them.
    expect({ ...silent, escalation: undefined }).toEqual({ ...absent, escalation: undefined });
  });

  it("reads a real p95 once the sample is big enough for one, not the maximum", () => {
    // Every other sample in this file, and in the corpus block below, has
    // n <= 18 -- and under nearest-rank every n <= 20 puts rank(95) at the LAST
    // rank, so p95 and max are the same number. `percentile(values, 95)`
    // replaced by `percentile(values, 100)` or by 96 survived the whole suite
    // for exactly that reason, and so did p50 moved to 49.
    //
    // n = 100 rather than the first size at which p95 clears the maximum, and
    // that is the whole subtlety: `ceil(percent * n / 100)` collapses NEIGHBOURING
    // percents onto one rank for most n. At n = 25, p95 and p96 are both rank 24
    // and p49 and p50 are both rank 13 -- measured, both of those mutants
    // survived a 25-item version of this test. Only at n = 100 does every
    // percent get its own rank.
    //
    // 100 items, one prose segment each, k = 1..100 single-character words, so a
    // segment of k words is 2k - 1 characters. Hand-computed at n = 100:
    //   p50 -> ceil(50 * 100 / 100) = rank 50  -> k 50  ->  99 chars /  50 words
    //   p95 -> ceil(95 * 100 / 100) = rank 95  -> k 95  -> 189 chars /  95 words
    //   max ->                        rank 100 -> k 100 -> 199 chars / 100 words
    // p95 is neither the maximum (199) nor p96's answer (191), and p50 is
    // neither p49's (97) nor p51's (101).
    const d = segmentSizeDistribution(
      Array.from({ length: 100 }, (_, i) => item(`k${i + 1}`, words(i + 1))),
    );
    expect(d.count).toBe(100);
    expect(d.chars).toEqual({ p50: 99, p95: 189, max: 199, min: 1 });
    expect(d.words).toEqual({ p50: 50, p95: 95, max: 100, min: 1 });
  });

  it("hands back the samples in encounter order rather than sorted", () => {
    // `samples` exists so a percentile quoted somewhere else can be re-derived,
    // which needs the raw values. Pre-sorting them survived the suite from one
    // side and sorting the caller's array inside `percentile` survived from the
    // other -- both are the same defect, and both are visible only if some test
    // reads a sample without sorting it first. This is that test.
    const d = segmentSizeDistribution([
      item("a", words(15)),
      item("b", words(5)),
      item("c", words(20)),
    ]);
    expect(d.samples.chars).toEqual([29, 9, 39]);
    expect(d.samples.words).toEqual([15, 5, 20]);
    expect(d.samples.bytes).toEqual([29, 9, 39]);
  });

  it("reports 0 words for an all-whitespace segment core's segmenter really produces", () => {
    // The guard `countWords` documents, exercised. Deleting `trimmed === \"\" ? 0 :`
    // survived the whole suite because no test ever fed it such a segment.
    //
    // Reachable, but not by the route the guard's comment used to name: a blank
    // line between two PROSE runs is merged into the surrounding run, so
    // \"a\\n\\nb\" is one segment. What stands alone is a blank run between two
    // runs of DIFFERENT kinds. MEASURED against core's segmenter, the text below
    // is code / prose \"\\n\\n\" / kv.
    const d = segmentSizeDistribution([
      item("blank-run", "```\nlet x = 1\n```\n\n\nclient: Northwind Traders"),
    ]);
    expect(d.segmentsTotal).toBe(3);
    // The fence is dropped by the predicate branch; the blank prose run and the
    // kv run are both selected.
    expect(d.count).toBe(2);
    expect(d.samples.chars).toEqual([2, 25]);
    // 0, not 1 and not 2: `\"\\n\\n\".split(/\\s+/)` is `[\"\", \"\"]`, so without the
    // guard this segment reports two words it does not have and drags
    // `words.min` off the floor.
    expect(d.samples.words).toEqual([0, 3]);
    expect(d.words).toEqual({ p50: 0, p95: 3, max: 3, min: 0 });
  });

  it("is empty-safe rather than dividing by zero", () => {
    const d = segmentSizeDistribution([]);
    expect(d.count).toBe(0);
    expect(d.items).toBe(0);
    expect(d.chars).toBeUndefined();
    expect(d.words).toBeUndefined();
    expect(d.perItem).toBeUndefined();
  });
});

describe("the measured distribution over corpora/fixtures/smoke.jsonl", () => {
  // This is the measurement Plan 5's Task 12 budget is sized from, pinned here
  // so that editing the corpus without re-sizing the budget fails a test rather
  // than leaving the plan quoting a number nothing produces any more.
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));

  it("selects 17 segments from 13 items, and here is every one of their sizes", () => {
    const d = segmentSizeDistribution(items);
    expect(items).toHaveLength(13);
    expect(d.count).toBe(17);
    expect(d.segmentsTotal).toBe(19);
    // The raw sample, ascending. Two entries were re-derived by hand from the
    // corpus text as a check on the segmenter rather than a restatement of it:
    // `pos-aws-key-code-fence` is 133 characters and splits 39 prose + 66 code
    // + 28 prose, and `pos-secret-key-value` is 110 and splits 45 prose + 65
    // kv. Both sum back to their item's length, and both contribute their
    // non-code parts below.
    expect([...d.samples.chars].sort((a, b) => a - b)).toEqual([
      22, 28, 36, 39, 39, 45, 48, 49, 62, 65, 70, 71, 72, 74, 78, 130, 153,
    ]);
  });

  it("puts the median segment at 62 characters and the largest at 153", () => {
    const d = segmentSizeDistribution(items);
    // Hand-computed from the sample above, n = 17:
    //   p50 -> ceil(50 * 17 / 100) = rank 9  -> 62
    //   p95 -> ceil(95 * 17 / 100) = rank 17 -> 153, which is also the maximum
    // The p95/max collision is a property of n, not of this corpus: any sample
    // below n = 20 has its 95th percentile at the last rank. Read the p95 as
    // "the largest thing in a 17-segment sample", never as a tail.
    expect(d.chars).toEqual({ p50: 62, p95: 153, max: 153, min: 22 });
    expect(d.words).toEqual({ p50: 9, p95: 25, max: 25, min: 3 });
  });

  it("is longer in BYTES than in code units on two of its segments, and pins both", () => {
    // This corpus already violates the rule the module header used to state.
    // Two selected segments carry non-ASCII -- `pos-emoji-before-pan` has
    // U+1F389 U+1F389 U+2714 and `neg-emoji-clean` has U+1F680 U+1F680 U+2014 --
    // so the byte sample is the character sample above with 62 -> 68 and
    // 72 -> 78 and nothing else moved. Derived by hand from those two segments
    // rather than read off the module.
    const d = segmentSizeDistribution(items);
    expect([...d.samples.bytes].sort((a, b) => a - b)).toEqual([
      22, 28, 36, 39, 39, 45, 48, 49, 65, 68, 70, 71, 74, 78, 78, 130, 153,
    ]);
    // The median moves 62 -> 65 and the maximum does not move at all, because
    // the largest segment here happens to be ASCII. That is why the budget
    // arithmetic Plan 5 quotes off THIS corpus survives the correction -- and
    // why the same arithmetic on a corpus whose longest segments are not ASCII
    // would not.
    expect(d.bytes).toEqual({ p50: 65, p95: 153, max: 153, min: 22 });
  });

  it("puts at most two judged segments in one message", () => {
    // The wall-clock budget is per MESSAGE and a judge makes one engine call
    // per selected segment, so this is the multiplier on any per-call latency.
    // It belongs beside the sizes rather than being inferred from them.
    const d = segmentSizeDistribution(items);
    expect(d.perItem).toEqual({ p50: 1, p95: 2, max: 2, min: 1 });
  });

  it("gains exactly one segment when tier 0's own findings supply the uncertainty", () => {
    // Not a hypothetical input. Running `runTier0` over this corpus with
    // apps/eval/fixtures/minimal-ir.json leaves TWO segments uncertain, both at
    // entropy's fixed 0.7: the kv block of `pos-secret-key-value`, which the
    // predicate branch already selected, and the code fence of
    // `pos-aws-key-code-fence`, which is the only way a code segment ever
    // reaches a judge. The finding below is that run's, verbatim.
    const priors = new Map<string, readonly Finding[]>([
      [
        "pos-aws-key-code-fence",
        [
          {
            start: 52,
            end: 90,
            text: "AWS_ACCESS_KEY_ID=AKIAZZ7EXAMPLE4XQ2LN",
            entityType: "generic-secret",
            severity: "critical",
            tier: 0,
            source: "entropy-rule",
            confidence: 0.7,
          },
        ],
      ],
    ]);
    const d = segmentSizeDistribution(items, {
      priorFindings: (i) => priors.get(i.id) ?? [],
    });
    expect(d.count).toBe(18);
    // The fence is 66 characters -- larger than nine of the seventeen segments
    // the predicate branch selects, and smaller than the largest, so it moves
    // neither the median nor the maximum. It does move the per-message count.
    expect(d.chars).toEqual({ p50: 62, p95: 153, max: 153, min: 22 });
    expect(d.perItem).toEqual({ p50: 1, p95: 3, max: 3, min: 1 });
  });
});
