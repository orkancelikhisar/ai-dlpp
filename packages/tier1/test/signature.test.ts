import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { MODEL_MANIFEST } from "../src/config.js";

/**
 * What `scripts/probe-model.ts` measured from the pinned weights.
 *
 * Read and parsed rather than `import ... with { type: "json" }` so the shape
 * below is stated once, explicitly. The fixture's `models` array is
 * heterogeneous -- a rung that fails to load carries `loadError` and none of
 * the probe fields -- and an inferred JSON type collapses that into a union
 * that cannot be narrowed on a discriminant the file does not have.
 */
interface DeclaredValueInfo {
  readonly name: string;
  readonly type: string;
  readonly dims: readonly (string | number | null)[];
}

interface Axis {
  readonly index: number;
  readonly extent: number;
  readonly role: "batch" | "words" | "classes" | "width" | "fixed";
  readonly declaredName: string | number | null;
}

interface Probe {
  readonly why: string;
  readonly feed: {
    readonly batch: number;
    readonly seqLen: number;
    readonly numWords: number;
    readonly numClasses: number;
    readonly maxWordsMask: number;
    readonly textLengths: number;
    readonly numSpans: number | null;
  };
  readonly logits?: { readonly type: string; readonly dims: readonly number[] };
  readonly error?: string;
}

interface Peak {
  readonly classIndex: number;
  readonly word: number;
  /** Index along the one axis that is neither batch, words, nor classes. */
  readonly otherAxisIndex: number;
  readonly score: number;
}

interface ProbedModel {
  readonly modelId: string;
  readonly repo: string;
  readonly revision: string;
  readonly precision: string;
  readonly spanMode: string;
  readonly weightsPath: string;
  readonly weightsSha256: string;
  readonly tokenizerSha256: string;
  readonly declared: {
    readonly irVersion: number | null;
    readonly producer: string;
    readonly nodeCount: number;
    readonly inputs: readonly DeclaredValueInfo[];
    readonly outputs: readonly DeclaredValueInfo[];
  };
  readonly loadError?: string;
  readonly onnxruntimeWebLoadError?: string | null;
  readonly runtime?: {
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
    readonly logitsType: string;
  };
  readonly probes?: Readonly<Record<string, Probe>>;
  readonly axes?: readonly Axis[];
  readonly wordAxisSizedBy?: string;
  readonly peaks?: {
    readonly words: readonly string[];
    readonly labels: readonly string[];
    readonly values: readonly Peak[];
  };
}

const signature = JSON.parse(
  readFileSync(new URL("./fixtures/model-signature.json", import.meta.url), "utf8"),
) as { readonly onnxruntimeNode: string; readonly models: readonly ProbedModel[] };

const byId = new Map(signature.models.map((model) => [model.modelId, model]));

/** The rungs that loaded, i.e. the ones with probe results to assert against. */
interface RanModel extends ProbedModel {
  readonly runtime: NonNullable<ProbedModel["runtime"]>;
  readonly probes: NonNullable<ProbedModel["probes"]>;
  readonly axes: NonNullable<ProbedModel["axes"]>;
  readonly peaks: NonNullable<ProbedModel["peaks"]>;
  readonly wordAxisSizedBy: string;
}

function ran(modelId: string): RanModel {
  const model = byId.get(modelId);
  if (model === undefined) throw new Error(`fixture has no entry for ${modelId}`);
  if (model.runtime === undefined) {
    throw new Error(`fixture says ${modelId} did not load: ${model.loadError ?? "no reason given"}`);
  }
  return model as RanModel;
}

const roles = (model: RanModel): string[] => model.axes.map((axis) => axis.role);
const probe = (model: RanModel, name: string): Probe => {
  const found = model.probes[name];
  if (found === undefined) throw new Error(`${model.modelId} has no probe "${name}"`);
  return found;
};
const wordAxisOf = (model: RanModel): Axis => {
  const axis = model.axes.find((a) => a.role === "words");
  if (axis === undefined) throw new Error(`${model.modelId} has no word axis`);
  return axis;
};

const TOKEN_LEVEL = ["gliner-pii-edge", "gliner-pii-edge-uint8"] as const;
const SPAN_LEVEL = ["gliner-pii-base", "gliner-pii-base-uint8"] as const;
const LOADABLE = [...TOKEN_LEVEL, ...SPAN_LEVEL];

describe("model signature: what the fixture is pinned to", () => {
  it("covers every model the manifest names, so a new entry cannot go unprobed", () => {
    expect([...byId.keys()].sort()).toEqual(Object.keys(MODEL_MANIFEST).sort());
  });

  it("is pinned to the weights and the tokenizer the manifest pins", () => {
    // A signature measured against other bytes describes another model. Both
    // hashes matter: every probe below feeds token ids produced by
    // tokenizer.json, so a tokenizer swap moves the measurements as surely as
    // a weight swap does.
    for (const [modelId, model] of byId) {
      const entry = MODEL_MANIFEST[modelId]!;
      expect(model.weightsSha256, modelId).toBe(entry.files[entry.weightsPath]!.sha256);
      expect(model.tokenizerSha256, modelId).toBe(entry.files["tokenizer.json"]!.sha256);
      expect(model.weightsPath, modelId).toBe(entry.weightsPath);
      expect(model.revision, modelId).toBe(entry.revision);
    }
  });

  it("agrees with the manifest's independently parsed input names", () => {
    // MODEL_MANIFEST.inputNames was written in Task 4 by a separate parse of
    // the same ModelProtos. This fixture re-parsed them. The two agreeing is a
    // cross-check; the two disagreeing means one of the parsers is wrong.
    for (const [modelId, model] of byId) {
      expect(model.declared.inputs.map((i) => i.name), modelId).toEqual(
        MODEL_MANIFEST[modelId]!.inputNames,
      );
    }
  });

  it("finds exactly one output, named logits, of rank 4, in every rung", () => {
    for (const [modelId, model] of byId) {
      expect(model.declared.outputs, modelId).toHaveLength(1);
      expect(model.declared.outputs[0]!.name, modelId).toBe("logits");
      expect(model.declared.outputs[0]!.dims, modelId).toHaveLength(4);
    }
  });
});

describe("model signature: the declared axis names are not the layout", () => {
  it("declares names for edge's logits that the measured extents contradict", () => {
    // THE reason scripts/probe-model.ts runs the graph instead of only reading
    // it. torch.onnx.export writes whatever `dynamic_axes` said, and for edge
    // it said [position, batch_size, sequence_length, num_classes]. Measured,
    // the same four axes are [batch, words, classes, 3]: every name is wrong,
    // and three of the four name an axis that exists one position away. A
    // decoder indexing by these names reads classes out of the word axis.
    const model = ran("gliner-pii-edge");
    expect(model.axes.map((a) => a.declaredName)).toEqual([
      "position",
      "batch_size",
      "sequence_length",
      "num_classes",
    ]);
    expect(roles(model)).toEqual(["batch", "words", "classes", "fixed"]);
  });

  it("declares names for base's logits that the measured extents contradict", () => {
    // Base's export said [batch_size, sequence_length, num_spans,
    // num_classes]. Only the first and last are right. Axis 1 is words, not
    // subword sequence positions, and axis 2 is the span WIDTH -- 12 of them --
    // not the 72 spans the feed enumerated.
    const model = ran("gliner-pii-base");
    expect(model.axes.map((a) => a.declaredName)).toEqual([
      "batch_size",
      "sequence_length",
      "num_spans",
      "num_classes",
    ]);
    expect(roles(model)).toEqual(["batch", "words", "fixed", "classes"]);
    expect(probe(model, "baseline").feed.numSpans).toBe(72);
    expect(model.axes[2]!.extent).toBe(12);
  });
});

describe("model signature: the logits layout, per span mode", () => {
  it.each(TOKEN_LEVEL)("%s emits [batch, words, classes, 3]", (modelId) => {
    const model = ran(modelId);
    expect(model.spanMode).toBe("token_level");
    expect(roles(model)).toEqual(["batch", "words", "classes", "fixed"]);
    // Each of these came from one probe that varied one feed dimension against
    // the baseline; see `probes[*].why` in the fixture.
    expect(probe(model, "baseline").logits!.dims).toEqual([1, 6, 2, 3]);
    expect(probe(model, "more-words").logits!.dims).toEqual([1, 9, 2, 3]);
    expect(probe(model, "more-classes").logits!.dims).toEqual([1, 6, 5, 3]);
    expect(probe(model, "batch-of-two").logits!.dims).toEqual([2, 6, 2, 3]);
    // The trailing 3 is not a class count and no feed dimension moves it. The
    // fp16 export of this same graph, which onnxruntime refuses to load,
    // declares it as the literal 3 rather than as a dynamic axis -- see
    // `declared.outputs[0].dims` on gliner-pii-edge-fp16.
    expect(model.axes[3]!.extent).toBe(3);
  });

  it.each(SPAN_LEVEL)("%s emits [batch, words, 12, classes]", (modelId) => {
    const model = ran(modelId);
    expect(model.spanMode).toBe("markerV0");
    expect(roles(model)).toEqual(["batch", "words", "fixed", "classes"]);
    expect(probe(model, "baseline").logits!.dims).toEqual([1, 6, 12, 2]);
    expect(probe(model, "more-words").logits!.dims).toEqual([1, 9, 12, 2]);
    expect(probe(model, "more-classes").logits!.dims).toEqual([1, 6, 12, 5]);
    expect(probe(model, "batch-of-two").logits!.dims).toEqual([2, 6, 12, 2]);
    // 12 is gliner_config.json's max_width, which Task 4 pinned into the
    // manifest. It is a constant OF THE GRAPH here, not of the feed -- see the
    // narrower-max-width probe below.
    expect(model.axes[2]!.extent).toBe(MODEL_MANIFEST[modelId]!.maxWidth);
  });

  it.each(LOADABLE)("%s returns float32 logits whatever the weight precision is", (modelId) => {
    // The uint8 rungs dequantise internally: they load, and hand back float32.
    // A decoder therefore does not need a per-precision reader for the rungs
    // that run -- which is not the same as saying every rung runs, see below.
    expect(ran(modelId).runtime.logitsType).toBe("float32");
  });
});

describe("model signature: the pooling contract", () => {
  it.each(LOADABLE)("%s sizes the word axis from text_lengths, not max(words_mask)", (modelId) => {
    // The load-bearing question of the whole word-level design, and the answer
    // is NOT the one the word-level story suggests. The two are equal in
    // ordinary use, so only a feed where they disagree can tell them apart.
    const model = ran(modelId);
    expect(model.wordAxisSizedBy).toBe("text_lengths");
  });

  it.each(LOADABLE)("%s rejects a words_mask value above text_lengths", (modelId) => {
    // text_lengths 4 against six words carrying words_mask 1..6. Both rungs
    // fail inside the same ScatterND that pools subwords into word slots,
    // naming the offending index. So text_lengths is not a hint: it allocates
    // the word slots, and a words_mask value with no slot is a hard error
    // rather than a dropped word.
    const failing = probe(ran(modelId), "text-lengths-below-words-mask");
    expect(failing.feed.textLengths).toBe(4);
    expect(failing.feed.maxWordsMask).toBe(6);
    expect(failing.logits).toBeUndefined();
    expect(failing.error).toContain("ScatterND");
    expect(failing.error).toContain("invalid indice found, indice = 4");
  });

  it.each(TOKEN_LEVEL)("%s pads the word axis out to text_lengths", (modelId) => {
    // Six words, text_lengths 9, and the word axis comes back 9 rather than 6.
    // No words_mask value points at the three extra slots; what the graph puts
    // in them was not measured, only that they exist. So on token level a
    // caller MAY over-declare text_lengths, provided it then ignores the tail
    // -- the axis length alone does not tell a decoder how many words are
    // real.
    const model = ran(modelId);
    const padded = probe(model, "text-lengths-above-words-mask");
    expect(padded.feed.numWords).toBe(6);
    expect(padded.feed.maxWordsMask).toBe(6);
    expect(padded.feed.textLengths).toBe(9);
    expect(padded.logits!.dims[wordAxisOf(model).index]).toBe(9);
  });
});

describe("model signature: how span-mode's span_idx and span_mask must be populated", () => {
  it.each(SPAN_LEVEL)("%s ties the span count to text_lengths x 12, exactly", (modelId) => {
    // Same padded feed that token level accepts, with span_idx still
    // enumerated over the 6 real words: base rejects it in the span_rep_layer
    // Reshape, and the error states the shape it wanted. So span_idx is not
    // free-form span proposals -- it is a rectangular enumeration the graph
    // reshapes, and its length is fixed by text_lengths and by the 12 the
    // export baked in.
    const model = ran(modelId);
    const mismatched = probe(model, "text-lengths-above-words-mask");
    expect(mismatched.feed.numSpans).toBe(72);
    expect(mismatched.feed.textLengths).toBe(9);
    expect(mismatched.error).toContain("Reshape");
    expect(mismatched.error).toContain("Input shape:{1,72,768}, requested shape:{1,9,12,768}");

    // Re-enumerated over all 9 declared word slots -- 108 spans -- the same
    // feed runs and returns a 9-word axis.
    const matched = probe(model, "span-enumeration-matching-padded-text-lengths");
    expect(matched.feed.numSpans).toBe(108);
    expect(matched.logits!.dims).toEqual([1, 9, 12, 2]);
  });

  it.each(SPAN_LEVEL)("%s does not let the caller narrow the span width", (modelId) => {
    // Widths 0..3 over 6 words, i.e. 24 spans. The graph still demands
    // {1,6,12,768}. Tier1Config.maxWidth can therefore only ever be a filter
    // applied AFTER decoding on this rung; it cannot make the model cheaper,
    // and it cannot be raised above 12 at all.
    const narrower = probe(ran(modelId), "narrower-max-width");
    expect(narrower.feed.numSpans).toBe(24);
    expect(narrower.error).toContain("Input shape:{1,24,768}, requested shape:{1,6,12,768}");
  });

  it.each(TOKEN_LEVEL)("%s takes no span inputs at all", (modelId) => {
    const model = ran(modelId);
    expect(model.runtime.inputNames).not.toContain("span_idx");
    expect(model.runtime.inputNames).not.toContain("span_mask");
    expect(model.probes["narrower-max-width"]).toBeUndefined();
  });
});

describe("model signature: the word axis is indexed by WORD", () => {
  // The dimensional evidence above only shows that an axis TRACKS the word
  // count. These assertions are what show it is indexed BY word: the probe
  // sentence puts a two-word person name at words 1-2 and an email at word 4,
  // and the scores peak on those indices. The two coordinate systems really do
  // disagree on this feed -- it is 6 words inside a 21-position sequence on
  // edge and a 19-position one on base -- so a subword-indexed axis could not
  // put a peak at 1, 2 and 4 for this sentence.
  it.each(LOADABLE)("%s was probed on the sentence these indices refer to", (modelId) => {
    const model = ran(modelId);
    expect(probe(model, "baseline").feed.numWords).toBe(6);
    expect(probe(model, "baseline").feed.seqLen).toBeGreaterThan(6);
    expect(model.peaks.words).toEqual([
      "Contact",
      "Priya",
      "Sharma",
      "at",
      "priya@acme.io",
      "today",
    ]);
    expect(model.peaks.labels).toEqual(["person", "email"]);
  });

  it.each(TOKEN_LEVEL)("%s peaks person at the name's first and last word", (modelId) => {
    const model = ran(modelId);
    const person = model.peaks.labels.indexOf("person");
    const at = (classIndex: number, otherAxisIndex: number): Peak => {
      const found = model.peaks.values.find(
        (p) => p.classIndex === classIndex && p.otherAxisIndex === otherAxisIndex,
      );
      if (found === undefined) throw new Error(`no peak at class ${classIndex}/${otherAxisIndex}`);
      return found;
    };
    // MEASURED, and the only reading of the trailing 3 that these two peaks
    // support: slot 0 scores a word as a span START and slot 1 as a span END.
    // "Priya" is word 1 and "Sharma" is word 2, and the two slots pick out one
    // word each, in that order. Slot 2 is not asserted -- it peaks on a word
    // inside the same name on both rungs, but on word 1 for one and word 2 for
    // the other, so this sentence does not distinguish what it means.
    expect(at(person, 0).word).toBe(model.peaks.words.indexOf("Priya"));
    expect(at(person, 1).word).toBe(model.peaks.words.indexOf("Sharma"));
    expect(at(person, 0).score).toBeGreaterThan(0.5);
    expect(at(person, 1).score).toBeGreaterThan(0.5);

    const email = model.peaks.labels.indexOf("email");
    expect(at(email, 0).word).toBe(model.peaks.words.indexOf("priya@acme.io"));
    expect(at(email, 0).score).toBeGreaterThan(0.5);
  });

  it.each(SPAN_LEVEL)("%s peaks its person class on the two-word span at the name", (modelId) => {
    const model = ran(modelId);
    const best = (classIndex: number): Peak => {
      const candidates = model.peaks.values.filter((p) => p.classIndex === classIndex);
      return candidates.reduce((a, b) => (b.score > a.score ? b : a));
    };
    // This is what identifies axis 2 as the span WIDTH, and identifies width
    // as an OFFSET from the start word rather than a length. The model's
    // highest person score is at word 1, index 1 -- and words 1..2 are exactly
    // "Priya Sharma". A length reading would make index 1 a one-word span and
    // put the name at index 2. Two further facts point the same way: the axis
    // is 12 wide, which is gliner_config.json's max_width, and the Reshape
    // error above names {1, words, 12, 768}.
    const person = best(model.peaks.labels.indexOf("person"));
    expect(person.word).toBe(model.peaks.words.indexOf("Priya"));
    expect(person.otherAxisIndex).toBe(1);
    expect(person.score).toBeGreaterThan(0.5);

    // One word, so offset 0.
    const email = best(model.peaks.labels.indexOf("email"));
    expect(email.word).toBe(model.peaks.words.indexOf("priya@acme.io"));
    expect(email.otherAxisIndex).toBe(0);
    expect(email.score).toBeGreaterThan(0.5);
  });
});

describe("model signature: the fp16 rungs of the ladder do not load", () => {
  const FP16 = ["gliner-pii-edge-fp16", "gliner-pii-base-fp16"];
  it.each(FP16)("%s is refused by both runtimes", (modelId) => {
    // Not a probing artefact and not something to route around. Both published
    // fp16 exports carry a Cast whose declared output type contradicts its
    // consumer, and onnxruntime rejects the graph at load. It is recorded here
    // per runtime because the two runtimes are different builds: the failure
    // reproduces under onnxruntime-web, which is the package the browser path
    // actually uses, so the defect is in the exported graph.
    const model = byId.get(modelId)!;
    expect(model.runtime).toBeUndefined();
    expect(model.loadError).toContain("Type (tensor(float16))");
    expect(model.loadError).toContain("does not match expected type (tensor(float))");
    expect(model.onnxruntimeWebLoadError).toContain("does not match expected type (tensor(float))");
    // The weights themselves are intact -- they hash to the pinned value, and
    // the graph parses far enough to read its own declared signature.
    expect(model.declared.outputs[0]!.type).toBe("float16");
  });

  it("does not leave a working rung at that precision on either repo", () => {
    // Stated so the experiment matrix cannot quietly carry 6 rungs while only 4
    // can run. If a re-pin ever fixes these exports, this fails and says so.
    const loaded = signature.models.filter((m) => m.runtime !== undefined).map((m) => m.modelId);
    expect(loaded.sort()).toEqual([...LOADABLE].sort());
  });
});
