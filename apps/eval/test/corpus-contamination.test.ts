import { describe, expect, it } from "vitest";
import {
  MIN_EXAMPLE_TOKENS,
  NGRAM_N,
  OVERLAP_DROP_THRESHOLD,
  buildRunIndex,
  checkContamination,
  containment,
  exampleNgrams,
  longestSharedRun,
  ngrams,
  overlapScore,
  selfTestExamplesFromCompiledCorpus,
  selfTestExamplesFromLlmFixture,
  tokenize,
  type SelfTestExample,
} from "../src/corpus/contamination.js";
import { loadSelfTestExamples } from "../src/corpus/build.js";
import { buildV2Artifacts } from "../src/corpus/build-v2.js";

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

/**
 * A REAL self-test case that a fixed n = 8 could not see, copied verbatim from
 * `policies/compiled/p-fin.selftest.json`. Six word tokens under this module's
 * tokenizer, so it carries no 8-gram at all; 263 of this repository's 723
 * examples are in that position.
 */
const SHORT_EXAMPLE = "Onboarding record shows AWEFT3518Q as PAN.";

describe("an example shorter than n, which used to be invisible", () => {
  it("has no 8-gram, so the old fixed-n rule could not match it at any score", () => {
    const tokens = tokenize(SHORT_EXAMPLE);
    expect(tokens.length).toBeLessThan(NGRAM_N);
    expect(ngrams(tokens).size).toBe(0);
  });

  it("is matched at its own length instead", () => {
    const tokens = tokenize(SHORT_EXAMPLE);
    expect(exampleNgrams(tokens).size).toBe(1);
    expect([...exampleNgrams(tokens)][0]).toBe(tokens.join(" "));
  });

  it("drops an item that copies it verbatim, which the fixed-n rule kept", () => {
    const item = {
      id: "copied-short",
      text: `hi, quick question about a record we were sent this morning. ${SHORT_EXAMPLE} is that ok to paste here or should i strip it first?`,
    };
    // The old behaviour, reconstructed from the primitives rather than asserted
    // from memory: at a fixed n = 8 the example contributes no gram, so the pair
    // cannot be compared at all and the item survives.
    expect(overlapScore(ngrams(tokenize(item.text)), ngrams(tokenize(SHORT_EXAMPLE)))).toBeUndefined();

    const report = checkContamination([item], [{ sourceId: "s", index: 7, text: SHORT_EXAMPLE }]);
    expect(report.dropped).toHaveLength(1);
    expect(report.dropped[0]).toMatchObject({ itemId: "copied-short", score: 1, direction: "example-in-item" });
    expect(report.examplesUnscoreable).toBe(0);
    expect(report.examplesScoredAtOwnLength).toBe(1);
  });

  it("still refuses to match on an example below the floor, which is a value and not phrasing", () => {
    // "encoding utf-8" is a real two-token self-test case. Matching on it would
    // drop any honest item that mentions an encoding.
    const below = "encoding: UTF-8";
    expect(tokenize(below).length).toBeLessThan(MIN_EXAMPLE_TOKENS);
    const report = checkContamination(
      [{ id: "mentions", text: "the export job writes csv with encoding: UTF-8 and a header row, which the importer then rejects for some reason" }],
      [{ sourceId: "s", index: 0, text: below }],
    );
    expect(report.dropped).toEqual([]);
    expect(report.examplesUnscoreable).toBe(1);
    // And the item is reported as unscoreable rather than kept: with the only
    // example below the floor, no comparison was made, and "kept" would claim
    // one was.
    expect(report.itemsKept).toBe(0);
    expect(report.unscoreable).toEqual([{ itemId: "mentions", tokens: 20 }]);
  });
});

describe("phrase overlap: the copying the ratio cannot see", () => {
  const stem = "The servicing console shows CIF 30045512 for this relationship.";

  it("finds the longest shared run and names the example it came from", () => {
    const index = buildRunIndex([tokenize(stem)]);
    const run = longestSharedRun(index, tokenize("the servicing console shows CIF 88112200 for that account, which is a different customer entirely"));
    expect(run).toMatchObject({ tokens: 5, phrase: "the servicing console shows cif", exampleIndex: 0 });
  });

  it("reports the run on an item the ratio scores at zero", () => {
    const item = {
      id: "shares-a-stem",
      text: "the servicing console shows CIF 88112200 for that account, and i cannot tell whether the branch code beside it matters for what i am asking",
    };
    const report = checkContamination([item], [{ sourceId: "s", index: 3, text: stem }]);
    expect(report.dropped).toEqual([]);
    expect(report.maxScoreKept).toBe(0);
    expect(report.phraseOverlap.maxRunTokens).toBe(5);
    expect(report.phraseOverlap.worst).toEqual([
      { itemId: "shares-a-stem", tokens: 5, phrase: "the servicing console shows cif", sourceId: "s", exampleIndex: 3 },
    ]);
  });

  it("distinguishes a zero measured over kept items from a zero measured over none", () => {
    // certificationSummary shipped the same defect: an empty input is not a
    // clean result. maxScoreKept is 0 in both cases; itemsKept is what tells
    // them apart.
    const empty = checkContamination([], [{ sourceId: "s", index: 0, text: stem }]);
    expect([empty.itemsKept, empty.maxScoreKept, empty.phraseOverlap.maxRunTokens]).toEqual([0, 0, 0]);
  });
});

describe("what the shipped corpus actually shares with the compiler self-test", () => {
  // The published measurement. It is not a round number and it is not zero: the
  // >0.7 8-gram ratio scores every item at 0.000, and five items share a
  // five-token sentence stem with policies/compiled/p-fin.selftest.json.
  const c = buildV2Artifacts().manifest.contamination;

  it("drops nothing, over a pool that was actually scored", () => {
    expect([c.itemsChecked, c.itemsKept, c.dropped.length, c.unscoreable.length]).toEqual([189, 189, 0, 0]);
    expect(c.maxScoreKept).toBe(0);
  });

  it("can now see the 148 examples a fixed n = 8 could not", () => {
    expect(c.examplesScoredAtOwnLength).toBe(148);
    expect(c.examplesUnscoreable).toBe(115);
    // 263 = what the old rule reported as unscoreable; 115 of those are values
    // below the floor and the other 148 are now matched at their own length.
    expect(c.examplesScoredAtOwnLength + c.examplesUnscoreable).toBe(263);
  });

  it("says why a score of 0 is a weaker statement than it looks", () => {
    // Of the examples long enough to carry a full 8-gram, most have none free
    // of a digit-bearing token, so matching one requires reproducing its
    // identifier -- which leakage.ts separately proves the generator never does.
    expect(c.examplesMatchableOnlyThroughTheirOwnIdentifier).toBe(300);
  });

  it("reports the five-token overlap that is really there", () => {
    expect(c.phraseOverlap.maxRunTokens).toBe(5);
    expect(c.phraseOverlap.histogram).toEqual([
      { tokens: 1, items: 1 },
      { tokens: 2, items: 77 },
      { tokens: 3, items: 70 },
      { tokens: 4, items: 36 },
      { tokens: 5, items: 5 },
    ]);
    expect(c.phraseOverlap.worst.map((w) => [w.itemId, w.phrase, w.sourceId])).toEqual([
      ["inj-hn03-1", "the permanent account number on", "policies/compiled/p-fin.selftest.json"],
      ["inj-hn08-0", "the servicing console shows cif", "policies/compiled/p-fin.selftest.json"],
      ["inj-o02-3", "the servicing console shows cif", "policies/compiled/p-fin.selftest.json"],
      ["inj-o03-0", "the permanent account number on", "policies/compiled/p-fin.selftest.json"],
      ["inj-o12-3", "the servicing console shows cif", "policies/compiled/p-fin.selftest.json"],
    ]);
  });
});
