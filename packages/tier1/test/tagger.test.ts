import { describe, expect, it } from "vitest";
import { detect, loadPolicyIr } from "@sih/core";
import { DEFAULT_TIER1_CONFIG, MODEL_MANIFEST, type Tier1Config } from "../src/config.js";
import { GlinerSpanTagger } from "../src/tagger.js";
import { FAKE_CLS_ID, FAKE_SEP_ID, fakeSession, fakeTokenizer, tier1Ir } from "./helpers.js";
import type { OnnxTensor } from "../src/session.js";

const EDGE = "gliner-pii-edge";
const BASE = "gliner-pii-base";

const cfg = (over: Partial<Tier1Config> = {}): Tier1Config => ({
  ...DEFAULT_TIER1_CONFIG,
  ...over,
});

const nums = (tensor: OnnxTensor | undefined): number[] =>
  Array.from(tensor?.data as BigInt64Array, Number);

describe("GlinerSpanTagger", () => {
  it("emits absolute offsets when the segment does not start at zero", () => {
    // THE bug this class exists to avoid. splitWords reports offsets into the
    // SEGMENT; normalizeFindings checks them against the MESSAGE and throws.
    // Word indices below are hand-counted over the segment text -- "call"(0)
    // "Acme"(1) "Corp"(2) "today"(3) -- so nothing here is derived from the
    // code under test.
    const message = "intro line\ncall Acme Corp today";
    const start = message.indexOf("call");
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2] }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    const findings = tagger.tag(
      [{ kind: "prose", start, end: message.length, text: message.slice(start) }],
      tier1Ir(),
    );
    return findings.then((f) => {
      expect(f).toHaveLength(1);
      expect(f[0]!.text).toBe("Acme Corp");
      expect(message.slice(f[0]!.start, f[0]!.end)).toBe("Acme Corp");
      expect(f[0]!.confidence).toBeCloseTo(0.9, 6);
      expect(f[0]!.tier).toBe(1);
      // The rung, not a constant: six rungs of one experiment matrix all emit
      // tier-1 findings and the harness has to tell their output apart.
      expect(f[0]!.source).toBe(EDGE);
    });
  });

  it("survives normalizeFindings inside a real detect() call, past an emoji", async () => {
    // core is the judge of whether these offsets are right, and it throws
    // rather than warning. The fence makes the prose segment start at 21, and
    // the emoji is a SURROGATE PAIR -- two UTF-16 units for one code point --
    // so a code-point-indexed span would land one unit short of "Acme".
    // Segment words: emoji(0) please(1) email(2) Acme(3) Corp(4) about(5)
    // the(6) renewal(7).
    const message = "```\nconst x = 1;\n```\n\u{1F642} please email Acme Corp about the renewal";
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(tier1Ir())),
      provider: "claude",
      text: message,
      config: { tier0: false, tier1: true, tier2: false },
      engines: {
        tier1: new GlinerSpanTagger(
          fakeSession({ modelId: EDGE, hit: [3, 4] }),
          fakeTokenizer(),
          cfg({ modelId: EDGE }),
        ),
      },
    });
    expect(result.findings.map((f) => f.text)).toEqual(["Acme Corp"]);
    expect(message.slice(result.findings[0]!.start, result.findings[0]!.end)).toBe("Acme Corp");
    // Re-derived by core from ir.entityTypes, not taken from this tagger.
    expect(result.findings[0]!.severity).toBe("high");
  });

  it("indexes spans by the SURVIVING words when one word tokenises to nothing", async () => {
    // Task 9's lockstep invariant, reached from a message. The BOM is a word to
    // the splitter and zero subwords to the tokenizer, so the model's word axis
    // is one shorter than the split. Feeding the pre-drop word list to
    // spanFromTokens yields "Acme" here instead of "Acme Corp" -- a neighbour
    // to the vault with the real value left behind.
    const message = "intro\ncall \u{FEFF}Acme Corp today";
    const start = message.indexOf("call");
    const feeds: Array<Readonly<Record<string, OnnxTensor>>> = [];
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2], onRun: (f) => void feeds.push(f) }),
      fakeTokenizer({ zeroSubwordWords: ["\u{FEFF}"] }),
      cfg({ modelId: EDGE }),
    );
    const found = await tagger.tag(
      [{ kind: "prose", start, end: message.length, text: message.slice(start) }],
      tier1Ir(),
    );
    expect(found.map((f) => f.text)).toEqual(["Acme Corp"]);
    expect(message.slice(found[0]!.start, found[0]!.end)).toBe("Acme Corp");
    // Five words split, four seated, and text_lengths agrees with the highest
    // slot -- which is what stops the graph from padding a phantom word.
    const wordsMask = nums(feeds[0]!["words_mask"]);
    expect(nums(feeds[0]!["text_lengths"])).toEqual([4]);
    expect(Math.max(...wordsMask)).toBe(4);
    expect(tagger.stats.droppedWords).toBe(1);
  });

  it("wraps the sequence and gives every prompt position slot 0", async () => {
    const feeds: Array<Readonly<Record<string, OnnxTensor>>> = [];
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2], onRun: (f) => void feeds.push(f) }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    await tagger.tag([{ kind: "prose", start: 0, end: 20, text: "call Acme Corp today" }], tier1Ir());
    const ids = nums(feeds[0]!["input_ids"]);
    const wordsMask = nums(feeds[0]!["words_mask"]);
    expect(ids[0]).toBe(FAKE_CLS_ID);
    expect(ids.at(-1)).toBe(FAKE_SEP_ID);
    expect(nums(feeds[0]!["attention_mask"])).toEqual(ids.map(() => 1));
    // One class here, so the prompt is [CLS] <<ENT>> "client name" <<SEP>>.
    expect(wordsMask.slice(0, 14)).toEqual(Array.from({ length: 14 }, () => 0));
    expect(wordsMask.filter((slot) => slot !== 0)).toEqual([1, 2, 3, 4]);
    // Feed set is exactly the graph's; token_level takes no span inputs.
    expect(Object.keys(feeds[0]!).sort()).toEqual([...MODEL_MANIFEST[EDGE]!.inputNames].sort());
  });

  it("emits nothing and never calls the model when the policy has no tier-1 entities", async () => {
    let runs = 0;
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [0, 0], onRun: () => void (runs += 1) }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    const findings = await tagger.tag(
      [{ kind: "prose", start: 0, end: 5, text: "hello" }],
      tier1Ir({ tier: 0 }),
    );
    expect(findings).toEqual([]);
    expect(runs).toBe(0);
  });

  it("never calls the model for a segment the splitter finds no words in", async () => {
    let runs = 0;
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [0, 0], onRun: () => void (runs += 1) }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    expect(await tagger.tag([{ kind: "prose", start: 0, end: 3, text: "   " }], tier1Ir())).toEqual(
      [],
    );
    expect(runs).toBe(0);
  });

  it("drops a span that maps to no character range, and does not throw", async () => {
    // Reachable, not synthetic: splitWords emits a zero-width space as a word
    // of its own, and spanFromTokens trims it off both ends and refuses the
    // empty remainder. One finding lost, message intact.
    const text = "note \u{200B} here";
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 1] }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    expect(
      await tagger.tag([{ kind: "prose", start: 0, end: text.length, text }], tier1Ir()),
    ).toEqual([]);
    expect(tagger.stats.unmappableSpans).toBe(1);
  });

  it("labels each finding with the entityType its class index maps to", async () => {
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2], classIndex: 1 }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    const findings = await tagger.tag(
      [{ kind: "prose", start: 0, end: 29, text: "call Acme Corp today about it" }],
      tier1Ir({ extraTier1Ids: ["project-codename"] }),
    );
    expect(findings.map((f) => f.entityType)).toEqual(["project-codename"]);
  });

  it("does not start a segment once the signal has fired", async () => {
    const controller = new AbortController();
    controller.abort();
    let runs = 0;
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2], onRun: () => void (runs += 1) }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    await expect(
      tagger.tag(
        [{ kind: "prose", start: 0, end: 20, text: "call Acme Corp today" }],
        tier1Ir(),
        controller.signal,
      ),
    ).resolves.toEqual([]);
    expect(runs).toBe(0);
  });

  it("keeps the findings it already had when the signal fires mid-run", async () => {
    // Spec 5.3's degradation: what a tier has already produced still counts,
    // and the abort only stops it from spending more of the budget.
    const message = "call Acme Corp today\ncall Acme Corp today";
    const controller = new AbortController();
    let runs = 0;
    const tagger = new GlinerSpanTagger(
      fakeSession({
        modelId: EDGE,
        hit: [1, 2],
        onRun: () => {
          runs += 1;
          controller.abort();
        },
      }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    const findings = await tagger.tag(
      [
        { kind: "prose", start: 0, end: 20, text: message.slice(0, 20) },
        { kind: "prose", start: 21, end: message.length, text: message.slice(21) },
      ],
      tier1Ir(),
      controller.signal,
    );
    expect(runs).toBe(1);
    expect(findings.map((f) => f.start)).toEqual([5]);
  });

  it("enumerates markerV0 spans from the graph's own width, not the configured one", async () => {
    // Task 7 measured that span_idx enumerated to width 4 instead of 12 fails
    // inside span_rep_layer's Reshape: the width is baked into the export.
    // Tier1Config.maxWidth is therefore a POST-DECODE filter and must not reach
    // the enumeration -- so this runs with maxWidth 1 and still feeds width 12.
    const feeds: Array<Readonly<Record<string, OnnxTensor>>> = [];
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: BASE, hit: [1, 2], onRun: (f) => void feeds.push(f) }),
      fakeTokenizer(),
      cfg({ modelId: BASE, maxWidth: 1 }),
    );
    const findings = await tagger.tag(
      [{ kind: "prose", start: 0, end: 20, text: "call Acme Corp today" }],
      tier1Ir(),
    );
    const spanIdx = nums(feeds[0]!["span_idx"]);
    const spanMask = Array.from(feeds[0]!["span_mask"]?.data as Uint8Array, Number);
    expect(feeds[0]!["span_idx"]?.dims).toEqual([1, 48, 2]);
    expect(spanIdx.slice(0, 4)).toEqual([0, 0, 0, 1]);
    // Last enumerated span is word 3 at width offset 11, i.e. 3..14, which runs
    // off a four-word text and is switched off rather than clamped.
    expect(spanIdx.slice(-2)).toEqual([3, 14]);
    expect(spanMask.at(-1)).toBe(0);
    expect(spanMask.filter((on) => on === 1)).toHaveLength(10);
    // ...and the width-2 span the model reported is dropped by the filter.
    expect(findings).toEqual([]);
    expect(tagger.stats.overWideSpans).toBe(1);
  });

  it("decodes a markerV0 span to the same characters the token_level path does", async () => {
    const message = "intro line\ncall Acme Corp today";
    const start = message.indexOf("call");
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: BASE, hit: [1, 2] }),
      fakeTokenizer(),
      cfg({ modelId: BASE }),
    );
    const findings = await tagger.tag(
      [{ kind: "prose", start, end: message.length, text: message.slice(start) }],
      tier1Ir(),
    );
    expect(findings.map((f) => f.text)).toEqual(["Acme Corp"]);
    expect(message.slice(findings[0]!.start, findings[0]!.end)).toBe("Acme Corp");
  });

  it("rejects a logits tensor whose axes are not the layout the fixture pinned", async () => {
    // A re-pin that moves an axis must fail loudly. Nothing downstream would
    // notice: a longer word axis decodes at the right strides and emits spans
    // naming words the message does not have.
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2], wordsOverride: 6 }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    await expect(
      tagger.tag([{ kind: "prose", start: 0, end: 20, text: "call Acme Corp today" }], tier1Ir()),
    ).rejects.toThrow(/logits axes/);
  });

  it("refuses a policy with more tier-1 classes than the model was exported for", async () => {
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2] }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    const ir = tier1Ir({
      extraTier1Ids: Array.from({ length: 100 }, (_, i) => `extra-${String(i)}`),
    });
    await expect(
      tagger.tag([{ kind: "prose", start: 0, end: 20, text: "call Acme Corp today" }], ir),
    ).rejects.toThrow(/101/);
  });

  it("refuses a session whose graph is not the one the config names", () => {
    // The realistic six-rung mix-up: the six weight files carry only three
    // distinct names between them -- onnx/model.onnx, onnx/model_fp16.onnx and
    // onnx/model_quint8.onnx, each shared by one edge rung and one base rung --
    // and a base feed run on an edge graph returns confident, wrongly shaped
    // logits.
    expect(
      () =>
        new GlinerSpanTagger(fakeSession({ modelId: EDGE }), fakeTokenizer(), cfg({ modelId: BASE })),
    ).toThrow(/span_idx/);
  });

  it("drops a span whose score is not a finite number, and counts the drop", async () => {
    // The boundary packages/core/src/detect/merge.ts asks tier adapters to
    // hold: "Both comparators assume `confidence` is a finite number. A NaN
    // makes every comparison false, which makes Array#sort's ordering
    // inconsistent and quietly destroys the determinism everything below
    // depends on." A NaN reaches here without help -- decode.ts guards a SHORT
    // logits array, but a correctly sized one carrying a NaN passes
    // requireLength, `sigmoid(NaN)` is NaN, and `NaN < threshold` is false, so
    // the cell is emitted as a span rather than filtered.
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE, hit: [1, 2], rawLogitOverride: Number.NaN }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    const findings = await tagger.tag(
      [{ kind: "prose", start: 0, end: 20, text: "call Acme Corp today" }],
      tier1Ir(),
    );
    expect(findings).toEqual([]);
    // Counted, not merely absent -- which is the whole point. Without this the
    // record is byte-identical to a model that found nothing, and offsets.ts
    // asks in as many words that whoever wires this to a model count the drops.
    expect(tagger.stats.nonFiniteScores).toBe(1);
    // And nothing else claims the drop: a NaN is not an unmappable span or an
    // over-wide one, so a reader of the counters can tell which happened.
    expect(tagger.stats.unmappableSpans).toBe(0);
    expect(tagger.stats.overWideSpans).toBe(0);
  });

  it("leaves the counter at zero when the model genuinely found nothing", async () => {
    // The control the test above needs: same tagger, same text, no hit. Both
    // runs return no findings, and only the counter separates them.
    const tagger = new GlinerSpanTagger(
      fakeSession({ modelId: EDGE }),
      fakeTokenizer(),
      cfg({ modelId: EDGE }),
    );
    const findings = await tagger.tag(
      [{ kind: "prose", start: 0, end: 20, text: "call Acme Corp today" }],
      tier1Ir(),
    );
    expect(findings).toEqual([]);
    expect(tagger.stats.nonFiniteScores).toBe(0);
    expect(tagger.stats.inferences).toBe(1);
  });

  it("never lets a NaN score reach core's merge through detect()", async () => {
    // The end of the chain, asserted through core rather than at this class's
    // edge: merge sorts by confidence, and one NaN is enough to make that sort
    // order depend on the input permutation.
    const message = "call Acme Corp today";
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(tier1Ir())),
      provider: "claude",
      text: message,
      config: { tier0: false, tier1: true, tier2: false },
      engines: {
        tier1: new GlinerSpanTagger(
          fakeSession({ modelId: EDGE, hit: [1, 2], rawLogitOverride: Number.NaN }),
          fakeTokenizer(),
          cfg({ modelId: EDGE }),
        ),
      },
    });
    expect(result.findings).toEqual([]);
    for (const f of result.findings) expect(Number.isFinite(f.confidence)).toBe(true);
  });
});
