/**
 * probe-model.ts -- measures what a pinned tier-1 ONNX graph actually does with
 * a feed, and prints the result as the fixture in test/fixtures/.
 *
 * ## Running it
 *
 *   pnpm -C packages/tier1 exec vite-node scripts/probe-model.ts > test/fixtures/model-signature.json
 *
 * (Same runner as scripts/fetch-models.ts and for the same reason: this tree
 * has no tsx, and Node's own type stripping resolves `./x.js` literally while
 * every source file here imports that way.)
 *
 * Needs the weights on disk -- run scripts/fetch-models.ts first. Refuses to
 * print anything for a model whose weights or tokenizer.json do not hash to
 * the value MODEL_MANIFEST pins, because a signature measured against other
 * bytes describes another model.
 *
 * ## Why it runs the graph instead of only reading it
 *
 * MEASURED, and the reason this script exists: the DECLARED axis names in both
 * graphs are wrong. torch.onnx.export writes whatever `dynamic_axes` said, and
 * these two exports say `logits: [position, batch_size, sequence_length,
 * num_classes]` (edge) and `[batch_size, sequence_length, num_spans,
 * num_classes]` (base). Neither matches the tensor the graph returns -- see
 * `probes` in the fixture, where varying one feed dimension at a time shows
 * which axis it moves. A decoder written against the declared names would index
 * the wrong axis on both rungs.
 *
 * `onnxruntime-node` is a devDependency used only here. The measured runtime
 * path is onnxruntime-web in a browser; nothing this script learns may be baked
 * into a Node-only code path, which is why the output is a fixture.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AutoTokenizer, env, type PreTrainedTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime-node";
import { MODEL_MANIFEST, type ModelEntry } from "../src/manifest.js";

/** See the use site for why this is a variable and not a literal specifier. */
const WEB_RUNTIME: string = "onnxruntime-web";

const HERE = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = join(HERE, "..", "models");
const REPO_ROOT = join(HERE, "..", "..", "..");

/**
 * Strips this checkout's location out of a message before it is recorded.
 *
 * onnxruntime puts the model path it was given into its load errors, and the
 * fixture is committed, so leaving it absolute would pin the file to one
 * machine's home directory and make it read as changed on every other.
 */
const relativise = (message: string): string => message.split(`${REPO_ROOT}/`).join("");

env.allowRemoteModels = false;
env.localModelPath = MODELS_DIR;

// ---------------------------------------------------------------------------
// A just-enough ONNX ModelProto reader.
//
// Only the graph's input and output ValueInfoProtos are wanted; every other
// field, including the initializers that are ~all of the file, is skipped by
// length without being materialised. Written out rather than taken from a
// library because the alternative is `onnx`/protobufjs as a dependency to read
// six numbers.
// ---------------------------------------------------------------------------

/** onnx.TensorProto.DataType. Only the types these two graphs can carry. */
const ELEM_TYPE: Readonly<Record<number, string>> = {
  1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32",
  7: "int64", 8: "string", 9: "bool", 10: "float16", 11: "float64", 12: "uint32",
  13: "uint64", 16: "bfloat16",
};

class ProtoReader {
  private p: number;
  constructor(private readonly b: Buffer, start = 0, private readonly e = b.length) {
    this.p = start;
  }
  get done(): boolean {
    return this.p >= this.e;
  }
  varint(): bigint {
    let value = 0n;
    let shift = 0n;
    for (;;) {
      const byte = this.b[this.p] ?? 0;
      this.p += 1;
      value |= BigInt(byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) return value;
      shift += 7n;
    }
  }
  /** Returns [fieldNumber, wireType]. */
  tag(): readonly [number, number] {
    const key = Number(this.varint());
    return [key >>> 3, key & 7];
  }
  /**
   * Advances past one field of the given wire type.
   *
   * The length is read into a local before the cursor moves. `this.p +=
   * Number(this.varint())` reads the OLD `this.p` for the addition while the
   * varint call advances it, so the cursor lands one byte short per skipped
   * field -- measured while writing this: it desynced the top-level loop on the
   * third field of edge's model.onnx.
   */
  skip(wireType: number): void {
    if (wireType === 0) {
      this.varint();
      return;
    }
    if (wireType === 1) {
      this.p += 8;
      return;
    }
    if (wireType === 2) {
      const length = Number(this.varint());
      this.p += length;
      return;
    }
    if (wireType === 5) {
      this.p += 4;
      return;
    }
    throw new Error(`unsupported protobuf wire type ${wireType} at byte ${this.p}`);
  }
  /** A reader scoped to one length-delimited field, cursor advanced past it. */
  sub(): ProtoReader {
    const length = Number(this.varint());
    const start = this.p;
    this.p = start + length;
    return new ProtoReader(this.b, start, start + length);
  }
  str(): string {
    const length = Number(this.varint());
    const start = this.p;
    this.p = start + length;
    return this.b.toString("utf8", start, start + length);
  }
}

/** A declared axis: a fixed extent, or the export's name for a dynamic one. */
type DeclaredDim = number | string | null;

export interface DeclaredValueInfo {
  readonly name: string;
  readonly type: string;
  readonly dims: readonly DeclaredDim[];
}

function readDimension(r: ProtoReader): DeclaredDim {
  let out: DeclaredDim = null;
  while (!r.done) {
    const [field, wire] = r.tag();
    if (field === 1 && wire === 0) out = Number(r.varint());
    else if (field === 2 && wire === 2) out = r.str();
    else r.skip(wire);
  }
  return out;
}

function readShape(r: ProtoReader): DeclaredDim[] {
  const dims: DeclaredDim[] = [];
  while (!r.done) {
    const [field, wire] = r.tag();
    if (field === 1 && wire === 2) dims.push(readDimension(r.sub()));
    else r.skip(wire);
  }
  return dims;
}

function readValueInfo(r: ProtoReader): DeclaredValueInfo {
  let name = "";
  let elemType = 0;
  let dims: DeclaredDim[] = [];
  while (!r.done) {
    const [field, wire] = r.tag();
    if (field === 1 && wire === 2) {
      name = r.str();
    } else if (field === 2 && wire === 2) {
      // TypeProto -> field 1 tensor_type -> {1: elem_type, 2: shape}
      const typeProto = r.sub();
      while (!typeProto.done) {
        const [tf, tw] = typeProto.tag();
        if (tf === 1 && tw === 2) {
          const tensor = typeProto.sub();
          while (!tensor.done) {
            const [ef, ew] = tensor.tag();
            if (ef === 1 && ew === 0) elemType = Number(tensor.varint());
            else if (ef === 2 && ew === 2) dims = readShape(tensor.sub());
            else tensor.skip(ew);
          }
        } else {
          typeProto.skip(tw);
        }
      }
    } else {
      r.skip(wire);
    }
  }
  return { name, type: ELEM_TYPE[elemType] ?? `elem_type_${elemType}`, dims };
}

export interface DeclaredGraph {
  readonly irVersion: number | null;
  readonly producer: string;
  readonly opset: readonly { readonly domain: string; readonly version: number | null }[];
  readonly nodeCount: number;
  readonly inputs: readonly DeclaredValueInfo[];
  readonly outputs: readonly DeclaredValueInfo[];
}

/** Parses ModelProto {1: ir_version, 2/3: producer, 7: graph, 8: opset_import}. */
function readModelProto(bytes: Buffer): DeclaredGraph {
  const top = new ProtoReader(bytes);
  let irVersion: number | null = null;
  let producerName = "";
  let producerVersion = "";
  const opset: { domain: string; version: number | null }[] = [];
  let graph: ProtoReader | null = null;
  while (!top.done) {
    const [field, wire] = top.tag();
    if (field === 1 && wire === 0) irVersion = Number(top.varint());
    else if (field === 2 && wire === 2) producerName = top.str();
    else if (field === 3 && wire === 2) producerVersion = top.str();
    else if (field === 7 && wire === 2) graph = top.sub();
    else if (field === 8 && wire === 2) {
      const entry = top.sub();
      let domain = "";
      let version: number | null = null;
      while (!entry.done) {
        const [ef, ew] = entry.tag();
        if (ef === 1 && ew === 2) domain = entry.str();
        else if (ef === 2 && ew === 0) version = Number(entry.varint());
        else entry.skip(ew);
      }
      opset.push({ domain: domain === "" ? "ai.onnx" : domain, version });
    } else top.skip(wire);
  }
  if (graph === null) throw new Error("ModelProto carries no graph");
  // GraphProto {1: node, 11: input, 12: output}
  const inputs: DeclaredValueInfo[] = [];
  const outputs: DeclaredValueInfo[] = [];
  let nodeCount = 0;
  while (!graph.done) {
    const [field, wire] = graph.tag();
    if (field === 11 && wire === 2) inputs.push(readValueInfo(graph.sub()));
    else if (field === 12 && wire === 2) outputs.push(readValueInfo(graph.sub()));
    else {
      if (field === 1 && wire === 2) nodeCount += 1;
      graph.skip(wire);
    }
  }
  return {
    irVersion,
    producer: `${producerName} ${producerVersion}`.trim(),
    opset,
    nodeCount,
    inputs,
    outputs,
  };
}

// ---------------------------------------------------------------------------
// Feed construction.
//
// The sequence is [CLS] <<ENT>> label ... <<SEP>> word ... [SEP], with
// words_mask carrying a 1-based word slot on each text word's first subword and
// 0 everywhere else. Nothing here is taken on authority: what says this shape
// is the right one is that the graph accepts it and then puts its highest
// person score on the two words that are a person's name and its highest email
// score on the word that is an email address -- see `peaks` in the fixture.
//
// Reproduced only far enough to make the graph answer questions about its own
// axes. The real encoder is Task 9's, and this deliberately takes a pre-split
// word list rather than splitting text, so that nothing measured here depends
// on a word splitter that does not exist yet.
// ---------------------------------------------------------------------------

interface FeedSpec {
  readonly labels: readonly string[];
  readonly words: readonly string[];
  /** Overrides text_lengths, which otherwise equals `words.length`. */
  readonly textLengths?: number;
  /** Present only for a graph that declares span_idx/span_mask. */
  readonly maxWidth?: number;
  /** Enumerates span_idx over this many words instead of `words.length`. */
  readonly spanEnumerationWords?: number;
}

interface BuiltFeed {
  readonly feed: Record<string, ort.Tensor>;
  readonly seqLen: number;
  readonly numWords: number;
  readonly maxWordsMask: number;
  readonly textLengths: number;
  readonly numSpans: number | null;
}

interface Vocabulary {
  readonly cls: number;
  readonly sep: number;
  readonly entToken: string;
  readonly sepToken: string;
}

function buildFeed(
  tokenizer: PreTrainedTokenizer,
  vocab: Vocabulary,
  spec: FeedSpec,
  /** The manifest's max_width, used unless a probe deliberately varies it. */
  defaultMaxWidth: number,
  takesSpans: boolean,
): BuiltFeed {
  const encode = (text: string): number[] =>
    Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data as BigInt64Array, Number);

  const promptWords: string[] = [];
  for (const label of spec.labels) promptWords.push(vocab.entToken, label);
  promptWords.push(vocab.sepToken);
  const allWords = [...promptWords, ...spec.words];

  const inputIds: number[] = [vocab.cls];
  const wordsMask: number[] = [0];
  allWords.forEach((word, wordIndex) => {
    // 1-based, and only on a word's first subword. All six pinned
    // gliner_config.json files say subtoken_pooling "first", and MEASURED,
    // that is load-bearing rather than decorative: marking the LAST subword
    // instead runs without error and returns the same shapes, but drops the
    // baseline person score from 0.75 to 0.40 and the email score from 0.82 to
    // 0.05. 0 means "pool nothing from this position", which is every prompt
    // token and every continuation subword.
    const slot = wordIndex < promptWords.length ? 0 : wordIndex - promptWords.length + 1;
    encode(word).forEach((id, k) => {
      inputIds.push(id);
      wordsMask.push(k === 0 ? slot : 0);
    });
  });
  inputIds.push(vocab.sep);
  wordsMask.push(0);

  const numWords = spec.words.length;
  const textLengths = spec.textLengths ?? numWords;
  const int64 = (values: readonly number[], dims: number[]): ort.Tensor =>
    new ort.Tensor("int64", BigInt64Array.from(values, BigInt), dims);

  const feed: Record<string, ort.Tensor> = {
    input_ids: int64(inputIds, [1, inputIds.length]),
    attention_mask: int64(inputIds.map(() => 1), [1, inputIds.length]),
    words_mask: int64(wordsMask, [1, wordsMask.length]),
    text_lengths: int64([textLengths], [1, 1]),
  };

  let numSpans: number | null = null;
  if (takesSpans) {
    const maxWidth = spec.maxWidth ?? defaultMaxWidth;
    const spanWords = spec.spanEnumerationWords ?? numWords;
    const spans: number[] = [];
    const mask: number[] = [];
    for (let start = 0; start < spanWords; start += 1) {
      for (let width = 0; width < maxWidth; width += 1) {
        // Inclusive end, so width 0 is a one-word span. That reading is what
        // the base peaks bear out: the person score is highest at start word 1
        // width 1, and words 1-2 are the name. A span running off the end of
        // the text is still enumerated -- the enumeration has to be
        // rectangular to be a tensor -- and switched off by span_mask.
        spans.push(start, start + width);
        mask.push(start + width < numWords ? 1 : 0);
      }
    }
    numSpans = spanWords * maxWidth;
    feed["span_idx"] = int64(spans, [1, numSpans, 2]);
    feed["span_mask"] = new ort.Tensor("bool", Uint8Array.from(mask), [1, numSpans]);
  }

  return {
    feed,
    seqLen: inputIds.length,
    numWords,
    maxWordsMask: Math.max(...wordsMask),
    textLengths,
    numSpans,
  };
}

// ---------------------------------------------------------------------------
// The probe matrix.
// ---------------------------------------------------------------------------

const LABELS_2 = ["person", "email"] as const;
const LABELS_5 = ["person", "email", "phone", "address", "organisation"] as const;
/** Six words, with a two-word person name at indices 1-2 and an email at 4. */
const WORDS_6 = ["Contact", "Priya", "Sharma", "at", "priya@acme.io", "today"] as const;
const WORDS_9 = [...WORDS_6, "or", "call", "later"] as const;

interface ProbeSpec {
  readonly name: string;
  readonly why: string;
  readonly spec: FeedSpec;
}

const PROBES: readonly ProbeSpec[] = [
  {
    name: "baseline",
    why: "6 words, 2 classes. Every other probe is read as a difference from this.",
    spec: { labels: LABELS_2, words: WORDS_6 },
  },
  {
    name: "more-words",
    why: "9 words, 2 classes. Whichever axis grows by 3 is the word axis.",
    spec: { labels: LABELS_2, words: WORDS_9 },
  },
  {
    name: "more-classes",
    why: "6 words, 5 classes. Whichever axis grows by 3 is the class axis.",
    spec: { labels: LABELS_5, words: WORDS_6 },
  },
  {
    name: "text-lengths-above-words-mask",
    why:
      "6 words but text_lengths 9. If the word axis is 9 the axis is sized by " +
      "text_lengths; if it is 6 it is sized by max(words_mask).",
    spec: { labels: LABELS_2, words: WORDS_6, textLengths: 9 },
  },
  {
    name: "text-lengths-below-words-mask",
    why:
      "6 words but text_lengths 4, i.e. a words_mask value with no slot to " +
      "pool into. Records whether the graph rejects that or silently drops it.",
    spec: { labels: LABELS_2, words: WORDS_6, textLengths: 4 },
  },
  {
    name: "span-enumeration-matching-padded-text-lengths",
    why:
      "span-mode only: the probe above, but with span_idx enumerated over all " +
      "9 padded word slots rather than the 6 real words. Asks whether the " +
      "span enumeration has to be sized from text_lengths or from the words " +
      "actually present.",
    spec: { labels: LABELS_2, words: WORDS_6, textLengths: 9, spanEnumerationWords: 9 },
  },
  {
    name: "narrower-max-width",
    why:
      "span-mode only: same words and classes, span_idx enumerated to width 4 " +
      "instead of 12. Asks whether the span width is a runtime knob at all.",
    spec: { labels: LABELS_2, words: WORDS_6, maxWidth: 4 },
  },
];

// ---------------------------------------------------------------------------
// Reading the result back.
// ---------------------------------------------------------------------------

/**
 * Logits as plain numbers.
 *
 * float32 only, deliberately. MEASURED: every rung that loads returns float32
 * -- the uint8 rungs dequantise inside the graph, and the fp16 rungs do not
 * load at all -- so no float16 tensor has ever reached this function. Writing
 * a float16 reader here would be writing a decoder nobody has run against a
 * tensor nobody has seen; refusing loudly is the honest alternative, and it
 * turns a re-pin that makes an fp16 rung loadable into a visible task rather
 * than silently reinterpreted bits.
 */
function toNumbers(tensor: ort.Tensor): Float64Array {
  if (tensor.type !== "float32") {
    throw new Error(`logits came back as ${tensor.type}; this reader only handles float32`);
  }
  const numeric = tensor.data as unknown as ArrayLike<number>;
  const out = new Float64Array(tensor.size);
  for (let i = 0; i < out.length; i += 1) out[i] = Number(numeric[i]);
  return out;
}

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const round4 = (x: number): number => Math.round(x * 1e4) / 1e4;

/** Row-major strides for `dims`. */
function strides(dims: readonly number[]): number[] {
  const out = new Array<number>(dims.length).fill(1);
  for (let i = dims.length - 2; i >= 0; i -= 1) out[i] = out[i + 1]! * dims[i + 1]!;
  return out;
}

/**
 * `width` is a hypothesis the probes put to the graph and the graph rejected:
 * every attempt to vary the span width failed to run, so no axis has been
 * named `width` by any rung measured so far. It stays in the union because the
 * narrower-max-width probe is what tests for it, and a re-pin that makes the
 * width dynamic should be able to say so rather than be squeezed into `fixed`.
 */
type AxisRole = "batch" | "words" | "classes" | "width" | "fixed";

interface Axis {
  readonly index: number;
  readonly extent: number;
  readonly role: AxisRole;
  /** The name torch.onnx.export declared for this axis, right or wrong. */
  readonly declaredName: DeclaredDim;
}

/**
 * Names each logits axis from what moved it, never from the declared name.
 *
 * Each probe varies exactly one feed dimension against `baseline`, so an axis
 * whose extent changed under that probe is the axis that dimension addresses.
 * An axis no probe moved is reported `fixed`, which is a result rather than an
 * absence of one: it says no feed dimension this script can vary reaches that
 * axis. Base's axis 2 is the case in point -- it is the span width, and the
 * export baked it in at 12, so nothing a caller sends can move it.
 */
function nameAxes(
  baseline: readonly number[],
  moved: ReadonlyMap<AxisRole, readonly number[] | undefined>,
  declared: readonly DeclaredDim[],
): Axis[] {
  return baseline.map((extent, index) => {
    let role: AxisRole = "fixed";
    for (const [candidate, dims] of moved) {
      if (dims !== undefined && dims[index] !== extent) {
        if (role !== "fixed") {
          throw new Error(`logits axis ${index} moved under more than one probe`);
        }
        role = candidate;
      }
    }
    return { index, extent, role, declaredName: declared[index] ?? null };
  });
}

interface Peak {
  readonly classIndex: number;
  /** Index along the axis `nameAxes` called `words`. */
  readonly word: number;
  /**
   * Index along the one axis that is neither batch, words nor classes.
   *
   * Deliberately not named after what it means, because it means something
   * different on each rung and neither meaning is visible in a shape: see
   * `axes` for the role the deriver could give it, and the peaks themselves
   * for what it turned out to be.
   */
  readonly otherAxisIndex: number;
  readonly score: number;
}

/**
 * The highest-scoring position per class, indexed through the measured axis
 * roles rather than through an assumed layout.
 *
 * The sigmoid is here to put the reported number on a readable 0..1 scale, and
 * nothing more. It is NOT a claim that sigmoid is the activation these models
 * were trained under -- this script measured shapes and argmaxes, not a
 * training objective, and which activation a decoder should apply is Task 8's
 * to settle. The indices are unaffected either way: an argmax survives any
 * monotone transform, so `word` and `otherAxisIndex` are the same numbers on
 * the raw logits.
 */
function peaks(tensor: ort.Tensor, axes: readonly Axis[]): Peak[] {
  const dims = tensor.dims as number[];
  const stride = strides(dims);
  const values = toNumbers(tensor);
  const axisOf = (role: AxisRole): Axis | undefined => axes.find((a) => a.role === role);
  const wordAxis = axisOf("words");
  const classAxis = axisOf("classes");
  if (wordAxis === undefined || classAxis === undefined) return [];
  const otherAxis = axes.find(
    (a) => a.role !== "batch" && a.role !== "words" && a.role !== "classes",
  );

  const out: Peak[] = [];
  for (let c = 0; c < classAxis.extent; c += 1) {
    for (let f = 0; f < (otherAxis?.extent ?? 1); f += 1) {
      let best = -Infinity;
      let bestWord = -1;
      for (let w = 0; w < wordAxis.extent; w += 1) {
        const offset =
          w * stride[wordAxis.index]! +
          c * stride[classAxis.index]! +
          (otherAxis === undefined ? 0 : f * stride[otherAxis.index]!);
        const value = values[offset]!;
        if (value > best) {
          best = value;
          bestWord = w;
        }
      }
      out.push({ classIndex: c, word: bestWord, otherAxisIndex: f, score: round4(sigmoid(best)) });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

const sha256 = (bytes: Buffer): string => createHash("sha256").update(bytes).digest("hex");

/** Repeats every tensor along dim 0, to ask whether axis 0 is really batch. */
function tileBatch(feed: Record<string, ort.Tensor>, copies: number): Record<string, ort.Tensor> {
  const out: Record<string, ort.Tensor> = {};
  for (const [name, tensor] of Object.entries(feed)) {
    const dims = [tensor.dims[0]! * copies, ...tensor.dims.slice(1)];
    const source = tensor.data as unknown as {
      length: number;
      slice(a: number, b: number): unknown;
    };
    const parts: unknown[] = [];
    for (let i = 0; i < copies; i += 1) parts.push(source.slice(0, source.length));
    const flat =
      tensor.type === "bool"
        ? Uint8Array.from(parts.flatMap((p) => Array.from(p as Uint8Array)))
        : BigInt64Array.from(parts.flatMap((p) => Array.from(p as BigInt64Array)));
    out[name] = new ort.Tensor(tensor.type as "int64" | "bool", flat as never, dims);
  }
  return out;
}

async function probeModel(modelId: string, entry: ModelEntry): Promise<unknown> {
  const weightsFile = entry.files[entry.weightsPath];
  const tokenizerFile = entry.files["tokenizer.json"];
  if (weightsFile === undefined || tokenizerFile === undefined) {
    throw new Error(`${modelId}: manifest has no entry for ${entry.weightsPath} or tokenizer.json`);
  }
  const weightsPath = join(MODELS_DIR, modelId, entry.weightsPath);

  const bytes = readFileSync(weightsPath);
  const weightsSha256 = sha256(bytes);
  if (weightsSha256 !== weightsFile.sha256) {
    throw new Error(
      `${modelId}: ${entry.weightsPath} hashes to ${weightsSha256}, manifest pins ${weightsFile.sha256}`,
    );
  }
  const declared = readModelProto(bytes);

  const tokenizerSha256 = sha256(readFileSync(join(MODELS_DIR, modelId, "tokenizer.json")));
  if (tokenizerSha256 !== tokenizerFile.sha256) {
    throw new Error(
      `${modelId}: tokenizer.json hashes to ${tokenizerSha256}, manifest pins ${tokenizerFile.sha256}`,
    );
  }

  const glinerConfig = JSON.parse(
    readFileSync(join(MODELS_DIR, modelId, "gliner_config.json"), "utf8"),
  ) as Record<string, unknown>;

  const header = {
    modelId,
    repo: entry.repo,
    revision: entry.revision,
    precision: entry.precision,
    spanMode: entry.spanMode,
    weightsPath: entry.weightsPath,
    weightsSha256,
    tokenizerSha256,
    declared,
  };

  let session: ort.InferenceSession;
  try {
    session = await ort.InferenceSession.create(weightsPath);
  } catch (error) {
    // A graph that will not load is a result about this rung of the ladder, so
    // it is recorded rather than thrown. The second load is what makes it a
    // statement about the GRAPH: onnxruntime-web is the package the browser
    // path actually runs, and if it refuses the same bytes for the same reason
    // then the refusal is not an onnxruntime-node artefact.

    // The specifier goes through a `string`-typed variable, and the shape is
    // written out here, because MEASURED: on a literal specifier `tsc` reports
    // "There are types at .../onnxruntime-web/types.d.ts, but this result
    // could not be resolved when respecting package.json exports" for this
    // pinned build. There is no type to import, only `create` is used, and a
    // structural type states what is used more precisely than `any` would.
    const ortWeb = (await import(WEB_RUNTIME)) as unknown as {
      readonly InferenceSession: {
        create(path: string, options: { executionProviders: string[] }): Promise<unknown>;
      };
    };
    let webLoadError: string | null = null;
    try {
      await ortWeb.InferenceSession.create(weightsPath, { executionProviders: ["wasm"] });
    } catch (webError) {
      webLoadError = String((webError as Error).message ?? webError);
    }
    return {
      ...header,
      loadError: relativise(String((error as Error).message ?? error)),
      onnxruntimeWebLoadError: webLoadError === null ? null : relativise(webLoadError),
    };
  }
  const takesSpans = session.inputNames.includes("span_idx");
  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  // The wrapper the pinned post_processor applies, read off an empty encode
  // rather than hardcoded, so it stays right if the tokenizer is ever repinned.
  const wrapper = Array.from(
    tokenizer("", { add_special_tokens: true }).input_ids.data as BigInt64Array,
    Number,
  );
  if (wrapper.length !== 2) {
    throw new Error(`${modelId}: expected a 2-token [CLS]/[SEP] wrapper, got ${wrapper.length}`);
  }
  const vocab: Vocabulary = {
    cls: wrapper[0]!,
    sep: wrapper[1]!,
    entToken: String(glinerConfig["ent_token"]),
    sepToken: String(glinerConfig["sep_token"]),
  };

  const results: Record<string, unknown> = {};
  const dimsByProbe = new Map<string, number[] | undefined>();

  for (const probe of PROBES) {
    if ((probe.spec.maxWidth ?? probe.spec.spanEnumerationWords) !== undefined && !takesSpans) {
      continue;
    }
    const built = buildFeed(tokenizer, vocab, probe.spec, entry.maxWidth, takesSpans);
    const shared = {
      why: probe.why,
      feed: {
        batch: 1,
        seqLen: built.seqLen,
        numWords: built.numWords,
        numClasses: probe.spec.labels.length,
        maxWordsMask: built.maxWordsMask,
        textLengths: built.textLengths,
        numSpans: built.numSpans,
      },
    };
    try {
      const out = await session.run(built.feed);
      const logits = out["logits"]!;
      dimsByProbe.set(probe.name, logits.dims as number[]);
      results[probe.name] = { ...shared, logits: { type: logits.type, dims: logits.dims } };
    } catch (error) {
      dimsByProbe.set(probe.name, undefined);
      results[probe.name] = {
        ...shared,
        error: relativise(String((error as Error).message ?? error)),
      };
    }
  }

  // Batch is its own probe because no FeedSpec field addresses it.
  {
    const built = buildFeed(
      tokenizer,
      vocab,
      { labels: LABELS_2, words: WORDS_6 },
      entry.maxWidth,
      takesSpans,
    );
    const shared = {
      why: "baseline tiled to batch 2. Whichever axis doubles is the batch axis.",
      feed: {
        batch: 2,
        seqLen: built.seqLen,
        numWords: built.numWords,
        numClasses: LABELS_2.length,
        maxWordsMask: built.maxWordsMask,
        textLengths: built.textLengths,
        numSpans: built.numSpans,
      },
    };
    try {
      const out = await session.run(tileBatch(built.feed, 2));
      const logits = out["logits"]!;
      dimsByProbe.set("batch-of-two", logits.dims as number[]);
      results["batch-of-two"] = { ...shared, logits: { type: logits.type, dims: logits.dims } };
    } catch (error) {
      dimsByProbe.set("batch-of-two", undefined);
      results["batch-of-two"] = {
        ...shared,
        error: relativise(String((error as Error).message ?? error)),
      };
    }
  }

  const baselineDims = dimsByProbe.get("baseline");
  if (baselineDims === undefined) throw new Error(`${modelId}: the baseline probe did not run`);
  const axes = nameAxes(
    baselineDims,
    new Map<AxisRole, readonly number[] | undefined>([
      ["batch", dimsByProbe.get("batch-of-two")],
      ["words", dimsByProbe.get("more-words")],
      ["classes", dimsByProbe.get("more-classes")],
      ["width", dimsByProbe.get("narrower-max-width")],
    ]),
    declared.outputs[0]?.dims ?? [],
  );

  // Mechanical, from the one probe where text_lengths and max(words_mask)
  // disagree: whichever of the two the word axis came out equal to is the one
  // that sizes it. Recorded as a derivation, not as a claim about the graph's
  // internals, which this script cannot see.
  const wordAxis = axes.find((a) => a.role === "words");
  const paddedName = takesSpans
    ? "span-enumeration-matching-padded-text-lengths"
    : "text-lengths-above-words-mask";
  const paddedDims = dimsByProbe.get(paddedName);
  const paddedExtent = paddedDims === undefined || wordAxis === undefined
    ? null
    : paddedDims[wordAxis.index] ?? null;
  const paddedFeed = (
    results[paddedName] as { feed?: { textLengths: number; maxWordsMask: number } } | undefined
  )?.feed;
  const wordAxisSizedBy =
    paddedExtent === null || paddedFeed === undefined
      ? "unmeasured"
      : paddedExtent === paddedFeed.textLengths
        ? "text_lengths"
        : paddedExtent === paddedFeed.maxWordsMask
          ? "max(words_mask)"
          : `neither: ${paddedExtent}`;

  const baselineFeed = buildFeed(
    tokenizer,
    vocab,
    { labels: LABELS_2, words: WORDS_6 },
    entry.maxWidth,
    takesSpans,
  );
  const baselineLogits = (await session.run(baselineFeed.feed))["logits"]!;
  const runtime = {
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    logitsType: baselineLogits.type,
  };
  await session.release();

  return {
    ...header,
    runtime,
    probes: results,
    axes,
    wordAxisSizedBy,
    peaks: {
      words: WORDS_6,
      labels: LABELS_2,
      why:
        "For each class and each index of the axis that is neither batch, " +
        "words nor classes, the sigmoid of the highest score along the word " +
        "axis and the word it fell on. Words 1-2 are a person name and word 4 " +
        "an email address, so a peak landing there is what shows the word " +
        "axis is indexed by WORD and not by subword token -- this sentence " +
        "tokenises to 19-21 subwords for its 6 words, so the two disagree.",
      values: peaks(baselineLogits, axes),
    },
  };
}

/** The probing runtime, not the measured one -- see this file's header. */
const ORT_NODE_VERSION = String(
  (
    JSON.parse(
      readFileSync(join(HERE, "..", "node_modules", "onnxruntime-node", "package.json"), "utf8"),
    ) as { version?: unknown }
  ).version,
);

const requested = process.argv.slice(2);
const ids = requested.length > 0 ? requested : Object.keys(MODEL_MANIFEST);
const models: unknown[] = [];
for (const modelId of ids) {
  if (!Object.hasOwn(MODEL_MANIFEST, modelId)) {
    throw new Error(`unknown tier-1 model "${modelId}"`);
  }
  models.push(await probeModel(modelId, MODEL_MANIFEST[modelId]!));
}
console.log(
  JSON.stringify(
    {
      probedBy: "packages/tier1/scripts/probe-model.ts",
      onnxruntimeNode: ORT_NODE_VERSION,
      models,
    },
    null,
    2,
  ),
);
