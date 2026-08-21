/**
 * Fakes for the tier-1 tagger's two collaborators, so `test/tagger.test.ts`
 * runs in Node without the 1.4 GB of weights on disk.
 *
 * Both fakes are deliberately DERIVED FROM THE FEED rather than from the test's
 * intent: `fakeSession` reads its word extent out of the `text_lengths` tensor
 * and its class extent out of the `<<ENT>>` markers in `input_ids`, so a tagger
 * that fed either of those wrongly produces a logits tensor of the wrong size
 * and the real decoders reject it. A fake that took those numbers as
 * constructor arguments would agree with a broken tagger.
 */
import type { EntityType, PolicyIr, Tier } from "@sih/core";
import { MODEL_MANIFEST } from "../src/config.js";
import { EDGE_SLOTS } from "../src/decode.js";
import type { SubwordTokenizer } from "../src/encode.js";
import type { OnnxSession, OnnxTensor } from "../src/session.js";

/**
 * Verbatim from packages/compiler/test/fixtures/llm, as in test/labels.test.ts:
 * what the compiler writes is a sentence aimed at a frontier model.
 */
const CLIENT_NAME_DEFINITION =
  "The name of a client organisation — a company, fund or institution the firm does business with — as it appears in running prose.";

export interface Tier1IrOptions {
  /** Tier of `client-name`. 0 makes the policy hold no tier-1 entity at all. */
  readonly tier?: Tier;
  /** Ids of further tier-1 entityTypes, appended after `client-name`. */
  readonly extraTier1Ids?: readonly string[];
}

/** A policy IR that `loadPolicyIr` accepts, with `client-name` at tier 1. */
export function tier1Ir(options: Tier1IrOptions = {}): PolicyIr {
  const entityTypes: EntityType[] = [
    {
      id: "client-name",
      tier: options.tier ?? 1,
      nlDefinition: CLIENT_NAME_DEFINITION,
      examples: [],
      counterExamples: [],
      severity: "high",
      surrogateKind: "org-name",
    },
    ...(options.extraTier1Ids ?? []).map((id) => ({
      id,
      tier: 1 as Tier,
      nlDefinition: `Anything that is a ${id}.`,
      examples: [],
      counterExamples: [],
      severity: "medium" as const,
    })),
  ];
  return {
    irVersion: "1",
    policyHash: "task10-test",
    entityTypes,
    rules: [],
    semanticPredicates: [],
    actions: {
      default: Object.fromEntries(entityTypes.map((e) => [e.id, "pseudonymize" as const])),
    },
    failMode: "closed",
    latencyBudgetMs: 800,
    provenance: {},
  };
}

/** Distinct, arbitrary, and asserted on: the wrapper the encoder must emit. */
export const FAKE_CLS_ID = 101;
export const FAKE_SEP_ID = 102;
/** One id each, as a real added-special-token would be. */
export const FAKE_ENT_ID = 1001;
export const FAKE_LABEL_SEP_ID = 1002;

export interface FakeTokenizerOptions {
  /**
   * Words this tokenizer returns ZERO subwords for. Not invented: Task 9
   * measured U+FEFF encoding to zero subwords on the pinned gliner-pii-base
   * tokenizer, which is what makes the lockstep drop reachable from a message.
   */
  readonly zeroSubwordWords?: readonly string[];
}

export function fakeTokenizer(options: FakeTokenizerOptions = {}): SubwordTokenizer {
  const zero = new Set(options.zeroSubwordWords ?? []);
  return {
    clsId: FAKE_CLS_ID,
    sepId: FAKE_SEP_ID,
    // One id per CODE POINT, so a word is usually several subwords and
    // words_mask has continuation positions to get wrong.
    encodeWord: (word) => {
      if (word === "<<ENT>>") return [FAKE_ENT_ID];
      if (word === "<<SEP>>") return [FAKE_LABEL_SEP_ID];
      if (zero.has(word)) return [];
      return Array.from(word, (ch) => 2000 + (ch.codePointAt(0) ?? 0));
    },
  };
}

/** Inverse of the decoders' sigmoid, so `score` below is read as a probability. */
const logit = (p: number): number => Math.log(p / (1 - p));
/** sigmoid(-20) is ~2e-9: "this cell says nothing". */
const SILENT = -20;

export interface FakeSessionOptions {
  /** Which rung, and therefore which feeds and which logits layout. */
  readonly modelId: string;
  /** `[firstWord, lastWord]`, inclusive, in SURVIVING word indices. */
  readonly hit?: readonly [number, number];
  readonly classIndex?: number;
  /** Probability of the winning span. Default 0.9. */
  readonly score?: number;
  /**
   * Sizes the logits word axis as if the graph had returned this many words
   * instead of `text_lengths`. Only for the re-pin guard test.
   */
  readonly wordsOverride?: number;
  readonly onRun?: (feeds: Readonly<Record<string, OnnxTensor>>) => void;
}

export function fakeSession(options: FakeSessionOptions): OnnxSession {
  const entry = MODEL_MANIFEST[options.modelId];
  if (entry === undefined) throw new Error(`no such model in the manifest: ${options.modelId}`);
  const score = logit(options.score ?? 0.9);
  const classIndex = options.classIndex ?? 0;

  return {
    inputNames: entry.inputNames,
    outputNames: ["logits"],
    release: () => Promise.resolve(),
    run: (feeds) => {
      options.onRun?.(feeds);
      const ids = Array.from(feeds["input_ids"]?.data as BigInt64Array, Number);
      const classes = ids.filter((id) => id === FAKE_ENT_ID).length;
      const fed = Number((feeds["text_lengths"]?.data as BigInt64Array)[0]);
      const words = options.wordsOverride ?? fed;
      const hit = options.hit;

      let tensor: OnnxTensor;
      if (entry.spanMode === "token_level") {
        const data = new Float32Array(words * classes * EDGE_SLOTS).fill(SILENT);
        const at = (word: number, slot: number): number =>
          (word * classes + classIndex) * EDGE_SLOTS + slot;
        if (hit !== undefined && hit[1] < words) {
          data[at(hit[0], 0)] = score;
          data[at(hit[1], 1)] = score;
          for (let w = hit[0]; w <= hit[1]; w += 1) data[at(w, 2)] = score;
        }
        tensor = { dims: [1, words, classes, EDGE_SLOTS], type: "float32", data };
      } else {
        const widths = entry.maxWidth;
        const data = new Float32Array(words * widths * classes).fill(SILENT);
        if (hit !== undefined && hit[1] < words && hit[1] - hit[0] < widths) {
          data[(hit[0] * widths + (hit[1] - hit[0])) * classes + classIndex] = score;
        }
        tensor = { dims: [1, words, widths, classes], type: "float32", data };
      }
      return Promise.resolve({ logits: tensor });
    },
  };
}
