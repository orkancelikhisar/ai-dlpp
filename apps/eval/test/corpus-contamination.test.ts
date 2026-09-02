import { describe, expect, it } from "vitest";
import {
  NGRAM_N,
  OVERLAP_DROP_THRESHOLD,
  checkContamination,
  containment,
  ngrams,
  overlapScore,
  selfTestExamplesFromCompiledCorpus,
  selfTestExamplesFromLlmFixture,
  tokenize,
  type SelfTestExample,
} from "../src/corpus/contamination.js";
import { loadSelfTestExamples } from "../src/corpus/build.js";

/** A real self-test case, copied from policies/compiled/p-fin.selftest.json. */
const EXAMPLE = "Onboarding record shows AWEFT3518Q as the permanent account number on file.";
const EXAMPLES: SelfTestExample[] = [{ sourceId: "fake", index: 0, text: EXAMPLE, corpusTag: "selftest-v1" }];

describe("tokenize / ngrams", () => {
  it("lowercases and strips punctuation so a requote does not evade the check", () => {
    expect(tokenize("The build FAILED, twice.")).toEqual(["the", "build", "failed", "twice"]);
  });

  it("produces length - n + 1 grams and none below n tokens", () => {
    const tokens = tokenize("a b c d e f g h i");
    expect(tokens).toHaveLength(9);
    expect(ngrams(tokens).size).toBe(9 - NGRAM_N + 1);
    expect(ngrams(tokenize("a b c")).size).toBe(0);
  });
});

describe("containment", () => {
  it("has no value for an empty left side, and says so rather than scoring 0", () => {
    // 0/0 is not 0. A 0 here would flow into the max as "no overlap" and turn a
    // pair that CANNOT be compared into a pair that was compared and passed.
    expect(containment(new Set(), new Set(["a b c"]))).toBeUndefined();
    expect(containment(new Set(["a b c"]), new Set())).toBe(0);
    expect(overlapScore(new Set(), new Set(["a b c"]))).toBeUndefined();
  });
});

describe("the direction of the overlap ratio", () => {
  // This is the measurement the module header rests on. It is asserted here
  // rather than only described, because the spec's own wording points at the
  // ratio that does NOT fire, and a later reader "simplifying" the max away
  // would silently turn the check off.
  const embedded = `i was asked about this yesterday. ${EXAMPLE} what should i tell them about the timeline for the correction?`;
  const a = ngrams(tokenize(embedded));
  const b = ngrams(tokenize(EXAMPLE));

  it("the item embeds the example verbatim", () => {
    expect(embedded).toContain(EXAMPLE);
    expect(b.size).toBeGreaterThan(0);
  });

  it("containment measured in the ITEM stays far below the threshold", () => {
    const itemSide = containment(a, b)!;
    expect(itemSide).toBeLessThan(0.3);
    expect(itemSide).toBeLessThan(OVERLAP_DROP_THRESHOLD);
  });

  it("containment measured in the EXAMPLE is total", () => {
    expect(containment(b, a)).toBe(1);
  });

  it("overlapScore takes the maximum, so the verbatim copy is caught", () => {
    expect(overlapScore(a, b)).toBe(1);
    expect(overlapScore(a, b)).toBe(overlapScore(b, a));
  });
});

describe("checkContamination", () => {
  it("drops an item that embeds a self-test example, and says which one", () => {
    const report = checkContamination(
      [
        { id: "clean", text: "nothing here resembles the generated corpus at all, not even a little bit of it" },
        { id: "copied", text: `preamble words go here. ${EXAMPLE} and a trailing question after it?` },
      ],
      EXAMPLES,
    );
    expect(report.dropped).toHaveLength(1);
    expect(report.dropped[0]).toMatchObject({
      itemId: "copied",
      score: 1,
      direction: "example-in-item",
      sourceId: "fake",
      exampleIndex: 0,
      example: EXAMPLE,
    });
  });

  it("reports a short item as unscoreable rather than scoring it clean", () => {
    // 0/0 is not 0. An item with no 8-gram has no evidence either way and the
    // report must not launder that into a pass.
    const report = checkContamination([{ id: "tiny", text: "too short" }], EXAMPLES);
    expect(report.dropped).toEqual([]);
    expect(report.unscoreable).toEqual([{ itemId: "tiny", tokens: 2 }]);
    expect(report.examplesUnscoreable).toBe(0);
  });

  it("reports the provenance tags of its sources", () => {
    const report = checkContamination([{ id: "x", text: "one two three four five six seven eight nine" }], EXAMPLES);
    expect(report.taggedSources).toEqual([{ corpusTag: "selftest-v1", examples: 1 }]);
    expect(report.n).toBe(NGRAM_N);
    expect(report.threshold).toBe(OVERLAP_DROP_THRESHOLD);
  });
});

describe("loading this repository's self-test examples", () => {
  it("reads the committed self-test corpus with its tag", () => {
    const parsed = selfTestExamplesFromCompiledCorpus(
      JSON.stringify([{ entityType: "in-pan", text: EXAMPLE, corpusTag: "selftest-v1" }]),
      "s",
    );
    expect(parsed).toEqual([{ sourceId: "s", index: 0, text: EXAMPLE, corpusTag: "selftest-v1" }]);
  });

  it("reads both lists of an llm fixture, because a hard negative is generated text too", () => {
    const parsed = selfTestExamplesFromLlmFixture(JSON.stringify({ positives: ["p"], negatives: ["n"] }), "f");
    expect(parsed.map((e) => [e.sourceId, e.text])).toEqual([
      ["f#positives", "p"],
      ["f#negatives", "n"],
    ]);
  });

  it("refuses a fixture missing a list rather than silently checking against half of it", () => {
    expect(() => selfTestExamplesFromLlmFixture(JSON.stringify({ positives: ["p"] }), "f")).toThrow(/negatives/);
  });

  it("reports how many self-test examples are too short for the check to see", () => {
    // 31% of the committed self-test corpus is a one-line config fragment under
    // 8 tokens. The check is blind to those; the number is reported rather than
    // absorbed into a clean-looking zero.
    const report = checkContamination([{ id: "x", text: "one two three four five six seven eight nine" }], [
      { sourceId: "s", index: 0, text: "pan_number: MKLPD7264V" },
      { sourceId: "s", index: 1, text: EXAMPLE },
    ]);
    expect(report.examplesUnscoreable).toBe(1);
  });

  it("loads every source this repository actually has", () => {
    const examples = loadSelfTestExamples();
    const compiled = examples.filter((e) => e.sourceId === "policies/compiled/p-fin.selftest.json");
    expect(compiled).toHaveLength(280);
    expect(compiled.every((e) => e.corpusTag === "selftest-v1")).toBe(true);
    const fixtures = new Set(
      examples.filter((e) => e.sourceId.startsWith("packages/compiler/")).map((e) => e.sourceId.split("#")[0]),
    );
    expect(fixtures.size).toBe(12);
    expect(examples.length).toBeGreaterThan(280);
  });
});
