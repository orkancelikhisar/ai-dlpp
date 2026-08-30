import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { MODEL_MANIFEST } from "../src/config.js";
import {
  EDGE_SLOT_END,
  EDGE_SLOT_INSIDE,
  EDGE_SLOT_START,
  EDGE_SLOTS,
  decodeBaseSpans,
  decodeEdgeSpans,
  type BaseLogitsDims,
  type EdgeLogitsDims,
} from "../src/decode.js";

/**
 * The inverse of the decoders' sigmoid, so a fixture below can be written in
 * the units the measurements were read in.
 *
 * Every probability in this file is a number this task read off a real run of
 * `packages/tier1/models/gliner-pii-edge/onnx/model.onnx` under
 * onnxruntime-node 1.21.0 -- see the per-test comments for the sentence and the
 * word each one came from.
 */
const logit = (p: number): number => Math.log(p / (1 - p));

/** Low enough that sigmoid is ~2e-9, i.e. "this cell says nothing". */
const SILENT = -20;

/** A base logits grid, `[words, widths, classes]` row-major, with one hot cell. */
function hot(
  dims: BaseLogitsDims,
  spot: { word: number; width: number; cls: number; score: number },
): Float32Array {
  const out = new Float32Array(dims.words * dims.widths * dims.classes).fill(SILENT);
  out[(spot.word * dims.widths + spot.width) * dims.classes + spot.cls] = spot.score;
  return out;
}

/** One word's `[start, end, inside]` probabilities for one class. */
type SlotTriple = readonly [start: number, end: number, inside: number];

/**
 * An edge logits grid, `[words, classes, 3]` row-major, from probabilities.
 *
 * `rows[word][class]` is that word's slot triple; a word/class with no entry
 * is silent.
 */
function edgeLogits(
  dims: EdgeLogitsDims,
  rows: Readonly<Record<number, Readonly<Record<number, SlotTriple>>>>,
): Float32Array {
  const out = new Float32Array(dims.words * dims.classes * dims.slots).fill(SILENT);
  for (const [wordKey, byClass] of Object.entries(rows)) {
    const word = Number(wordKey);
    for (const [classKey, triple] of Object.entries(byClass)) {
      const cls = Number(classKey);
      triple.forEach((p, slot) => {
        out[(word * dims.classes + cls) * dims.slots + slot] = logit(p);
      });
    }
  }
  return out;
}

describe("decodeBaseSpans", () => {
  it("reads axis 2 as an inclusive width offset", () => {
    // [batch, words, 12, classes], row-major. A hit at word 1, offset 1
    // means words 1..2 -- the span the model actually reported for a
    // two-word person name.
    const dims = { words: 4, widths: 12, classes: 2 };
    const logits = hot(dims, { word: 1, width: 1, cls: 0, score: 5 });
    expect(decodeBaseSpans(logits, dims, 0.5)).toEqual([
      { firstWord: 1, lastWord: 2, classIndex: 0, score: expect.any(Number) },
    ]);
  });

  it("never proposes a span running past the last real word", () => {
    // The [word x width] grid is rectangular, so its far corner always
    // describes spans past the end. Always present in the tensor, never valid.
    const dims = { words: 4, widths: 12, classes: 2 };
    expect(decodeBaseSpans(hot(dims, { word: 3, width: 5, cls: 0, score: 5 }), dims, 0.5)).toEqual(
      [],
    );
  });

  it("discards the corner even when the model scores it above the threshold", () => {
    // Not hypothetical. MEASURED on gliner-pii-base with 8 words and the
    // labels person/email: every out-of-range cell of the grid carries one
    // per-class CONSTANT -- 0.032 for person, 0.042 for email on that feed --
    // rather than a floor of zero. Those sat under a 0.5 threshold, but the
    // threshold is an experiment variable (Tier1Config.threshold), and a run
    // at 0.02 would otherwise turn the whole corner into `words * widths`
    // spans that run off the end of the text. The bound is what makes the
    // constant unreachable at any threshold.
    const dims = { words: 4, widths: 12, classes: 2 };
    const logits = new Float32Array(dims.words * dims.widths * dims.classes).fill(logit(0.032));
    expect(decodeBaseSpans(logits, dims, 0.02).every((s) => s.lastWord < dims.words)).toBe(true);
    // 4 words x widths 0..3 that stay in range, x 2 classes.
    expect(decodeBaseSpans(logits, dims, 0.02)).toHaveLength((4 + 3 + 2 + 1) * 2);
  });

  it("applies sigmoid per class, not softmax across classes", () => {
    // Scoring is per-class binary -- a message may hold a client name and a
    // project codename at once. Softmax would make the labels compete and
    // silently suppress the weaker of two genuine findings.
    const dims = { words: 1, widths: 12, classes: 2 };
    const logits = new Float32Array(dims.words * dims.widths * dims.classes).fill(-10);
    logits[0] = 2;
    logits[1] = 2;
    const spans = decodeBaseSpans(logits, dims, 0.5);
    expect(spans).toHaveLength(2);
    expect(spans[0]!.score).toBeCloseTo(spans[1]!.score, 6);
    // The value, not just the agreement: softmax over two equal logits is 0.5
    // exactly, which would pass the line above and still be the wrong number.
    expect(spans[0]!.score).toBeCloseTo(1 / (1 + Math.exp(-2)), 6);
  });

  it("keeps every width above the threshold at one word", () => {
    // Overlap resolution belongs to core's merge, which weighs all tiers
    // together severity-first and then by confidence. Widths 0, 1 and 2 at the
    // same word are three overlapping candidates; a decoder that returned only
    // the best of them would hide the two the merge might have preferred.
    const dims = { words: 4, widths: 12, classes: 1 };
    const logits = new Float32Array(dims.words * dims.widths * dims.classes).fill(SILENT);
    for (const width of [0, 1, 2]) {
      logits[(1 * dims.widths + width) * dims.classes] = logit(0.9 - width / 10);
    }
    expect(decodeBaseSpans(logits, dims, 0.5).map((s) => s.lastWord)).toEqual([1, 2, 3]);
  });
});

describe("decodeEdgeSpans", () => {
  it("pairs a start slot with a later end slot of the same class", () => {
    // [batch, words, classes, 3]: slot 0 = start, slot 1 = end, slot 2 =
    // inside. A name at words 1..2 appears as a start peak at 1, an end peak
    // at 2, and inside high on both -- these six probabilities are what
    // gliner-pii-edge returned for the person class on "Email Priya Sharma
    // and Rahul Mehta before Friday" at words 1 and 2.
    const dims = { words: 4, classes: 2, slots: 3 };
    const logits = edgeLogits(dims, {
      1: { 0: [0.859, 0.052, 0.859] },
      2: { 0: [0.014, 0.799, 0.797] },
    });
    expect(decodeEdgeSpans(logits, dims, 0.5, 12)).toEqual([
      { firstWord: 1, lastWord: 2, classIndex: 0, score: expect.any(Number) },
    ]);
  });

  it("does not pair a start with an end of a different class", () => {
    const dims = { words: 4, classes: 2, slots: 3 };
    const logits = edgeLogits(dims, {
      1: { 0: [0.9, 0.05, 0.9], 1: [0.05, 0.05, 0.9] },
      2: { 0: [0.05, 0.05, 0.9], 1: [0.05, 0.9, 0.9] },
    });
    expect(decodeEdgeSpans(logits, dims, 0.5, 12)).toEqual([]);
  });

  it("does not pair an end that precedes its start", () => {
    const dims = { words: 4, classes: 1, slots: 3 };
    const logits = edgeLogits(dims, {
      1: { 0: [0.05, 0.9, 0.9] },
      2: { 0: [0.05, 0.05, 0.9] },
      3: { 0: [0.9, 0.05, 0.9] },
    });
    expect(decodeEdgeSpans(logits, dims, 0.5, 12)).toEqual([]);
  });

  it("reads slot 2 as INSIDE, so a gap between two entities cannot be spanned", () => {
    // The whole reason the third slot is decoded rather than ignored, and the
    // sentence that establishes what it is. MEASURED on gliner-pii-edge,
    // person class, "Email Priya Sharma and Rahul Mehta before Friday": every
    // number below is that run.
    //
    // start peaks on "Priya" (1) and "Rahul" (4); end peaks on "Sharma" (2)
    // and "Mehta" (5). Read on start and end alone, that is THREE pairs with
    // start <= end, and the third -- 1..5 -- swallows "and" and both names
    // into one span scoring 0.782. What rejects it is slot 2: it is high on
    // all four name words (0.859, 0.797, 0.794, 0.769) and collapses to 0.091
    // on "and" (w3). Neither start (0.010) nor end (0.021) distinguishes that
    // word from the middle of a longer name, so only slot 2 can carry it.
    //
    // Two further runs say the same. A three-word name -- "Contact Priya
    // Anjali Sharma at priya@acme.io today" -- puts start 0.822 on "Priya",
    // end 0.846 on "Sharma", and only 0.107/0.196 on the middle word "Anjali",
    // whose slot 2 is 0.883, the highest of the three. And the uint8 rung
    // reproduces the shape: 0.316 on "and" against 0.626-0.787 on the names.
    const dims = { words: 8, classes: 1, slots: 3 };
    const logits = edgeLogits(dims, {
      0: { 0: [0.037, 0.028, 0.04] },
      1: { 0: [0.859, 0.052, 0.859] },
      2: { 0: [0.014, 0.799, 0.797] },
      3: { 0: [0.01, 0.021, 0.091] },
      4: { 0: [0.788, 0.059, 0.794] },
      5: { 0: [0.033, 0.782, 0.769] },
      6: { 0: [0.01, 0.01, 0.018] },
      7: { 0: [0.053, 0.07, 0.063] },
    });
    expect(decodeEdgeSpans(logits, dims, 0.5, 12)).toEqual([
      { firstWord: 1, lastWord: 2, classIndex: 0, score: expect.closeTo(0.797, 3) },
      { firstWord: 4, lastWord: 5, classIndex: 0, score: expect.closeTo(0.769, 3) },
    ]);
  });

  it("aggregates inside by MIN across the span, not by mean", () => {
    // Same measured row as above, and the reason the aggregator is named.
    // Averaging slot 2 over words 1..5 gives 0.662, which clears 0.5 and lets
    // the straddling 1..5 span through; a single dip is exactly what a mean
    // absorbs and what a min is for.
    const dims = { words: 6, classes: 1, slots: 3 };
    const inside = [0.859, 0.797, 0.091, 0.794, 0.769];
    expect(inside.reduce((a, b) => a + b, 0) / inside.length).toBeGreaterThan(0.5);
    const logits = edgeLogits(dims, {
      0: { 0: [0.859, 0.052, 0.859] },
      1: { 0: [0.014, 0.799, 0.797] },
      2: { 0: [0.01, 0.021, 0.091] },
      3: { 0: [0.788, 0.059, 0.794] },
      4: { 0: [0.033, 0.782, 0.769] },
      5: { 0: [0.01, 0.01, 0.018] },
    });
    expect(decodeEdgeSpans(logits, dims, 0.5, 12).map((s) => [s.firstWord, s.lastWord])).toEqual([
      [0, 1],
      [3, 4],
    ]);
  });

  it("holds a one-word span to the inside slot on its own word", () => {
    // Slot 2 is not an interior-only signal. MEASURED on "Ask Priya about it":
    // the one-word person at word 1 comes back start 0.723, end 0.728, inside
    // 0.690 -- all three fire on the same single word, so a width-1 span has
    // an inside score of its own to clear and is treated no differently from a
    // longer one.
    const dims = { words: 3, classes: 1, slots: 3 };
    expect(decodeEdgeSpans(edgeLogits(dims, { 1: { 0: [0.723, 0.728, 0.69] } }), dims, 0.5, 12)).toEqual(
      [{ firstWord: 1, lastWord: 1, classIndex: 0, score: expect.closeTo(0.69, 3) }],
    );
    expect(decodeEdgeSpans(edgeLogits(dims, { 1: { 0: [0.723, 0.728, 0.2] } }), dims, 0.5, 12)).toEqual(
      [],
    );
  });

  it("pairs one start with EVERY qualifying end, not just the nearest", () => {
    // The rule rests on a structural argument, not on a score: every (i, j)
    // pair sharing a start word overlaps every other, so picking one of them
    // IS overlap resolution. Core's merge owns that, across all tiers
    // together, severity-first and then by confidence; a tier that resolved
    // its own overlaps first would hand the merge a choice already made.
    //
    // An earlier version of this comment cited "3 false positives against 10
    // over seven probe sentences". That experiment has no artifact in this
    // repo -- test/fixtures/model-signature.json commits three sentences, not
    // seven -- so the figure is struck rather than repeated. MEASURED over the
    // three that ARE committed: nearest-end and every-pair emit the identical
    // set on all of them, because the inside run always breaks before a second
    // qualifying end appears. The fixture cannot separate the two rules, which
    // is why THIS test builds an input where they visibly differ.
    const dims = { words: 4, classes: 1, slots: 3 };
    const logits = edgeLogits(dims, {
      0: { 0: [0.9, 0.05, 0.9] },
      1: { 0: [0.05, 0.8, 0.9] },
      2: { 0: [0.05, 0.7, 0.9] },
      3: { 0: [0.05, 0.6, 0.9] },
    });
    expect(decodeEdgeSpans(logits, dims, 0.5, 12).map((s) => s.lastWord)).toEqual([1, 2, 3]);
  });

  it("scores a span as the weakest of its start, end and inside evidence", () => {
    // DECISION: min, not product and not mean. Tier1Config.threshold is one
    // number shared by both rungs, so edge's score has to live on the same
    // 0..1 scale as base's single sigmoid or the same config value means two
    // different things -- three 0.8s multiply to 0.512, so a product decoder
    // silently demands ~0.79 per component where base demands 0.5.
    //
    // MEASURED over the three sentences test/fixtures/model-signature.json
    // commits under `slotSemantics` (the whole of the evidence this repo
    // holds; the "seven probe sentences" an earlier comment cited are not in
    // it): at threshold 0.5 on gliner-pii-edge the minimum keeps 4 spans and
    // the product keeps 2, losing "Rahul Mehta" (0.789 x 0.782 x 0.769 =
    // 0.474) and "Priya" (0.723 x 0.728 x 0.690 = 0.363). On
    // gliner-pii-edge-uint8 the product keeps 0 of the same 4. Every span the
    // product drops there is a TRUE one.
    const dims = { words: 3, classes: 1, slots: 3 };
    const logits = edgeLogits(dims, {
      0: { 0: [0.9, 0.05, 0.75] },
      1: { 0: [0.05, 0.8, 0.7] },
      2: { 0: [0.05, 0.05, 0.9] },
    });
    const [span] = decodeEdgeSpans(logits, dims, 0.5, 12);
    expect(span!.score).toBeCloseTo(0.7, 6);
    // The three combiners this task rejected. Each returns a different number
    // on this same input, so an assertion that only checked "a span came back"
    // would not tell them apart.
    expect(span!.score).not.toBeCloseTo(0.9 * 0.8 * 0.75 * 0.7, 6);
    expect(span!.score).not.toBeCloseTo(Math.sqrt(0.9 * 0.8), 6);
    expect(span!.score).not.toBeCloseTo((0.9 + 0.8 + 0.75 + 0.7) / 4, 6);
  });

  it("stops the walk at maxWidth instead of enumerating to the end of the text", () => {
    // The bound exists for cost, so what it must NOT do is change the answer.
    // Every end here clears the threshold with the inside run wide open, so an
    // unbounded walk would propose 0..1 through 0..5 and a bounded one only
    // 0..1 and 0..2. The tagger discards the rest at exactly this width, which
    // is why cutting them here is free -- see the width-bound note in
    // src/decode.ts.
    const dims = { words: 6, classes: 1, slots: 3 };
    const logits = edgeLogits(dims, {
      0: { 0: [0.9, 0.05, 0.9] },
      1: { 0: [0.05, 0.9, 0.9] },
      2: { 0: [0.05, 0.9, 0.9] },
      3: { 0: [0.05, 0.9, 0.9] },
      4: { 0: [0.05, 0.9, 0.9] },
      5: { 0: [0.05, 0.9, 0.9] },
    });
    expect(decodeEdgeSpans(logits, dims, 0.5, 12).map((s) => s.lastWord)).toEqual([1, 2, 3, 4, 5]);
    expect(decodeEdgeSpans(logits, dims, 0.5, 3).map((s) => s.lastWord)).toEqual([1, 2]);
    // Inclusive at both ends, the same reading tagger.ts filters by: width 1
    // is the one-word span at the start word, and here that word's own end
    // slot is silent, so nothing comes back rather than a self-pair.
    expect(decodeEdgeSpans(logits, dims, 0.5, 1)).toEqual([]);
  });

  it("rejects a maxWidth that is not a positive integer rather than returning nothing", () => {
    // A zero or fractional width makes `Math.min(words, firstWord + maxWidth)`
    // cut the walk before its first iteration, so the decoder returns an empty
    // array -- indistinguishable from a model that found nothing, which is the
    // one confusion this whole file is arranged to prevent.
    const dims = { words: 4, classes: 1, slots: 3 };
    const logits = edgeLogits(dims, { 1: { 0: [0.9, 0.9, 0.9] } });
    expect(decodeEdgeSpans(logits, dims, 0.5, 1)).toEqual([
      { firstWord: 1, lastWord: 1, classIndex: 0, score: expect.any(Number) },
    ]);
    for (const bad of [0, -1, 1.5, Number.NaN]) {
      expect(() => decodeEdgeSpans(logits, dims, 0.5, bad)).toThrow(/maxWidth/);
    }
  });

  it("rejects a slot axis that is not the measured 3", () => {
    // The slot indices are hard-coded constants, so a rung whose trailing axis
    // is a different size is not something to decode on a best guess.
    const dims = { words: 4, classes: 2, slots: 4 };
    expect(() => decodeEdgeSpans(new Float32Array(4 * 2 * 4), dims, 0.5, 12)).toThrow(/slot/i);
  });
});

describe("both decoders", () => {
  it("reject a logits array whose length disagrees with the dimensions", () => {
    // A silent reshape misreads every score at a shifted stride and produces
    // plausible-looking garbage.
    expect(() =>
      decodeBaseSpans(new Float32Array(5), { words: 4, widths: 12, classes: 2 }, 0.5),
    ).toThrow(/length/i);
    expect(() =>
      decodeEdgeSpans(new Float32Array(5), { words: 4, classes: 2, slots: 3 }, 0.5, 12),
    ).toThrow(/length/i);
  });

  it("return spans sorted by descending score", () => {
    const baseDims = { words: 4, widths: 12, classes: 1 };
    const baseLogits = new Float32Array(
      baseDims.words * baseDims.widths * baseDims.classes,
    ).fill(SILENT);
    baseLogits[(0 * baseDims.widths + 0) * baseDims.classes] = logit(0.6);
    baseLogits[(1 * baseDims.widths + 0) * baseDims.classes] = logit(0.9);
    baseLogits[(2 * baseDims.widths + 0) * baseDims.classes] = logit(0.7);
    expect(decodeBaseSpans(baseLogits, baseDims, 0.5).map((s) => s.firstWord)).toEqual([1, 2, 0]);

    const edgeDims = { words: 4, classes: 1, slots: 3 };
    const edgeSpans = decodeEdgeSpans(
      edgeLogits(edgeDims, {
        0: { 0: [0.6, 0.6, 0.95] },
        1: { 0: [0.9, 0.9, 0.95] },
        2: { 0: [0.7, 0.7, 0.95] },
        3: { 0: [0.05, 0.05, 0.05] },
      }),
      edgeDims,
      0.5,
      12,
    );
    const scores = edgeSpans.map((s) => s.score);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
    expect(scores[0]).toBeCloseTo(0.9, 6);
  });

  it("order ties by position, across the class axis, so a run is reproducible", () => {
    // Task 9's harness diffs run against run, so two spans on the same score
    // must not swap places between two decodes.
    //
    // Asserted on the edge decoder specifically, and this is the point of the
    // test: `Array.prototype.sort` is stable, so a tiebreak is only observable
    // where emission order and position order DISAGREE. decodeBaseSpans walks
    // word-major and already emits in position order, which makes the tiebreak
    // a no-op there -- MEASURED: deleting it from the comparator left a
    // base-only version of this test passing. decodeEdgeSpans walks
    // class-major, so it emits 0:0, 2:0, 0:1, 2:1 for these four equal scores
    // and only the tiebreak turns that into position order.
    const dims = { words: 3, classes: 2, slots: 3 };
    const tie: SlotTriple = [0.8, 0.8, 0.9];
    const spans = decodeEdgeSpans(
      edgeLogits(dims, { 0: { 0: tie, 1: tie }, 2: { 0: tie, 1: tie } }),
      dims,
      0.5,
      12,
    );
    expect(spans.map((s) => s.score)).toEqual([0.8, 0.8, 0.8, 0.8].map(() => spans[0]!.score));
    expect(spans.map((s) => `${s.firstWord}:${s.classIndex}`)).toEqual([
      "0:0",
      "0:1",
      "2:0",
      "2:1",
    ]);

    // The same four scores through the base decoder, whose emission order the
    // tiebreak agrees with, so this states the shared order rather than tests
    // it.
    const baseDims = { words: 3, widths: 12, classes: 2 };
    const baseLogits = new Float32Array(
      baseDims.words * baseDims.widths * baseDims.classes,
    ).fill(SILENT);
    for (const [word, cls] of [
      [2, 1],
      [0, 1],
      [2, 0],
      [0, 0],
    ] as const) {
      baseLogits[(word * baseDims.widths + 0) * baseDims.classes + cls] = logit(0.8);
    }
    expect(
      decodeBaseSpans(baseLogits, baseDims, 0.5).map((s) => `${s.firstWord}:${s.classIndex}`),
    ).toEqual(["0:0", "0:1", "2:0", "2:1"]);
  });

  it("returns nothing for a word axis of zero length", () => {
    expect(decodeBaseSpans(new Float32Array(0), { words: 0, widths: 12, classes: 2 }, 0.5)).toEqual(
      [],
    );
    expect(decodeEdgeSpans(new Float32Array(0), { words: 0, classes: 2, slots: 3 }, 0.5, 12)).toEqual(
      [],
    );
  });
});

// ---------------------------------------------------------------------------
// The decoders' correctness is entirely a function of the committed probe of
// the pinned weights. These tests exist so a re-pin that moves an axis breaks
// here, loudly, instead of being silently mis-decoded.
// ---------------------------------------------------------------------------

interface FixtureAxis {
  readonly index: number;
  readonly extent: number;
  readonly role: "batch" | "words" | "classes" | "width" | "fixed";
}
interface FixtureModel {
  readonly modelId: string;
  readonly spanMode: string;
  readonly axes?: readonly FixtureAxis[];
  readonly wordAxisSizedBy?: string;
  readonly probes?: Readonly<
    Record<string, { readonly logits?: { readonly dims: readonly number[] } }>
  >;
}

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/model-signature.json", import.meta.url), "utf8"),
) as { readonly models: readonly FixtureModel[] };

function probed(modelId: string): Required<Pick<FixtureModel, "axes" | "probes">> & FixtureModel {
  const model = fixture.models.find((m) => m.modelId === modelId);
  if (model?.axes === undefined || model.probes === undefined) {
    throw new Error(`fixture has no probed axes for ${modelId}`);
  }
  return model as Required<Pick<FixtureModel, "axes" | "probes">> & FixtureModel;
}

/** Row-major strides for the fixture's OWN axis order. */
function fixtureStrides(dims: readonly number[]): number[] {
  const out = new Array<number>(dims.length).fill(1);
  for (let i = dims.length - 2; i >= 0; i -= 1) out[i] = out[i + 1]! * dims[i + 1]!;
  return out;
}

/**
 * The offset of one cell, addressed by axis ROLE through the fixture.
 *
 * This is the whole point of these tests: the coordinates go in named, the
 * fixture decides where they land, and the decoder -- which hard-codes an
 * order -- has to agree. Swap two roles in the fixture and the value is
 * written somewhere the decoder does not look.
 */
function cellOffset(
  axes: readonly FixtureAxis[],
  dims: readonly number[],
  coords: Readonly<Record<string, number>>,
): number {
  const strides = fixtureStrides(dims);
  let offset = 0;
  for (const axis of axes) {
    if (axis.role === "batch") continue;
    if (!Object.hasOwn(coords, axis.role)) throw new Error(`no coordinate for role ${axis.role}`);
    offset += coords[axis.role]! * strides[axis.index]!;
  }
  return offset;
}

const TOKEN_LEVEL = ["gliner-pii-edge", "gliner-pii-edge-uint8"] as const;
const SPAN_LEVEL = ["gliner-pii-base", "gliner-pii-base-uint8"] as const;

describe("the layout the decoders are written against", () => {
  it.each(TOKEN_LEVEL)("%s: decodeEdgeSpans reads the axes the probe measured", (modelId) => {
    const model = probed(modelId);
    expect(model.spanMode).toBe("token_level");
    // Declared: [position, batch_size, sequence_length, num_classes]. Measured:
    // batch, words, classes, and a fixed trailing 3. decodeEdgeSpans indexes
    // (word * classes + class) * 3 + slot, which is only right for this order.
    expect(model.axes.map((a) => a.role)).toEqual(["batch", "words", "classes", "fixed"]);
    expect(model.axes[3]!.extent).toBe(EDGE_SLOTS);

    // And now against the decoder rather than against a list of names: place a
    // start, an end and two insides through the fixture's own axis order, at
    // the extents the baseline probe returned, and check the decoder finds the
    // span at the words they were written to.
    const dims = model.probes["baseline"]!.logits!.dims;
    const [, words, classes] = dims as [number, number, number, number];
    const logits = new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(SILENT);
    const put = (word: number, cls: number, slot: number, p: number): void => {
      logits[cellOffset(model.axes, dims, { words: word, classes: cls, fixed: slot })] = logit(p);
    };
    put(1, 0, EDGE_SLOT_START, 0.822);
    put(2, 0, EDGE_SLOT_END, 0.846);
    put(1, 0, EDGE_SLOT_INSIDE, 0.804);
    put(2, 0, EDGE_SLOT_INSIDE, 0.883);
    expect(decodeEdgeSpans(logits, { words, classes, slots: EDGE_SLOTS }, 0.5, 12)).toEqual([
      { firstWord: 1, lastWord: 2, classIndex: 0, score: expect.closeTo(0.804, 3) },
    ]);
  });

  it.each(SPAN_LEVEL)("%s: decodeBaseSpans reads the axes the probe measured", (modelId) => {
    const model = probed(modelId);
    expect(model.spanMode).toBe("markerV0");
    // Declared: [batch_size, sequence_length, num_spans, num_classes]. Measured:
    // batch, words, a fixed 12, classes. decodeBaseSpans indexes
    // (word * widths + width) * classes + class.
    expect(model.axes.map((a) => a.role)).toEqual(["batch", "words", "fixed", "classes"]);
    // The width axis is the export's max_width, which Task 4 pinned. maxWidth
    // in Tier1Config can only ever filter AFTER this decode, and never above 12
    // -- the graph rejects a narrower span_idx enumeration outright.
    expect(model.axes[2]!.extent).toBe(MODEL_MANIFEST[modelId]!.maxWidth);

    const dims = model.probes["baseline"]!.logits!.dims;
    const [, words, widths, classes] = dims as [number, number, number, number];
    const logits = new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(SILENT);
    // Word 1, offset 1 is what the probe sentence's two-word person name came
    // back on, so this is the model's own coordinate for words 1..2.
    logits[cellOffset(model.axes, dims, { words: 1, fixed: 1, classes: 0 })] = logit(0.554);
    expect(decodeBaseSpans(logits, { words, widths, classes }, 0.5)).toEqual([
      { firstWord: 1, lastWord: 2, classIndex: 0, score: expect.closeTo(0.554, 3) },
    ]);
  });

  it.each([...TOKEN_LEVEL, ...SPAN_LEVEL])("%s sizes its word axis from text_lengths", (modelId) => {
    // Load-bearing for what these decoders do NOT do. They treat every slot on
    // the word axis as a real word, because the tensor carries no marker for a
    // padded one -- MEASURED, edge accepts text_lengths 9 against 6 words and
    // returns a 9-long word axis. Keeping text_lengths equal to the real word
    // count is therefore the caller's invariant (Task 9's), not something
    // decode can check.
    expect(probed(modelId).wordAxisSizedBy).toBe("text_lengths");
  });
});
