/**
 * The tier-1 span tagger: core's `SpanTagger`, implemented over one pinned
 * GLiNER-class ONNX graph.
 *
 * This file is assembly, not algorithm. Every decision it composes was measured
 * somewhere else -- the word splitter in words.ts, the lockstep encoder in
 * encode.ts, the two decoders in decode.ts, the character mapping in offsets.ts
 * -- and what is left here is the wiring between them, plus the one arithmetic
 * step that belongs to nobody else.
 *
 * ## That one step
 *
 * The tagger is handed `Segment[]`, each carrying its own offsets into the
 * message, and `splitWords` reports offsets relative to the SEGMENT. Adding
 * `segment.start` is the whole of the conversion, and it is the most likely
 * place for this package to be wrong: core's `normalizeFindings` THROWS on
 * `text !== message.slice(start, end)` rather than warning, so one wrong
 * character fails the entire message, and `applyActions` rewrites by span, so a
 * drifted finding that did slip through would send a neighbouring word to the
 * pseudonymization vault and leave the real confidential value in place.
 *
 * `text` comes from `spanFromTokens`, which slices the SEGMENT, while core
 * checks it against a slice of the MESSAGE. Those two agree exactly when the
 * absolute offsets are right, which is what makes the fidelity check a real
 * test of this addition rather than a tautology.
 *
 * ## Two rungs, two paths
 *
 * `ModelEntry.spanMode` selects both the feed set and the decoder: `markerV0`
 * additionally takes `span_idx`/`span_mask`, which `token_level` has no input
 * for. Everything from the word split to the character mapping is shared, and
 * `test/tagger.test.ts` runs the same sentence down both paths and expects the
 * same characters out.
 */
import type { Finding, PolicyIr, Segment, SpanTagger } from "@sih/core";
import { MODEL_MANIFEST, type ModelEntry, type Tier1Config } from "./config.js";
import { EDGE_SLOTS, decodeBaseSpans, decodeEdgeSpans, type DecodedSpan } from "./decode.js";
import { encodeWords, enumerateSpans, type EncodedWords, type SubwordTokenizer } from "./encode.js";
import { buildLabels, type Tier1Label } from "./labels.js";
import { spanFromTokens } from "./offsets.js";
import {
  LOGITS_OUTPUT,
  assertSignature,
  type OnnxSession,
  type OnnxTensor,
} from "./session.js";
import { splitWords } from "./words.js";

/**
 * The two prompt markers, read out of both pinned `gliner_config.json` files on
 * disk for this task: `ent_token` is `<<ENT>>` and `sep_token` is `<<SEP>>` in
 * `models/gliner-pii-edge/gliner_config.json` and
 * `models/gliner-pii-base/gliner_config.json` alike.
 *
 * Constants here rather than fields of `ModelEntry` because they do not differ
 * across the six rungs. If a re-pin ever makes them differ, they belong in the
 * manifest and this is the code that has to move.
 */
const ENT_TOKEN = "<<ENT>>";
const SEP_TOKEN = "<<SEP>>";

/**
 * What the tagger threw away, accumulated across every `tag` call it serves.
 *
 * Not decoration, and not a metric of the model: offsets.ts asks in as many
 * words that whoever wires it to a model COUNT the drops rather than let them
 * vanish, because a silently dropped span is indistinguishable from a model
 * that found nothing. An eval arm reporting recall has to be able to tell those
 * apart.
 */
export interface Tier1TaggerStats {
  /** Segments actually handed to the graph. */
  inferences: number;
  /** Words that tokenised to nothing and lost their seat (encode.ts's lockstep drop). */
  droppedWords: number;
  /** Trailing words `maxLen` cut off before they reached the graph. */
  truncatedWords: number;
  /**
   * Decoded spans wider than `Tier1Config.maxWidth`.
   *
   * markerV0 ONLY, and not because markerV0 is special: `decodeEdgeSpans` is
   * now handed the same `maxWidth` and stops enumerating at it, so a
   * token_level rung can never reach the filter below. On markerV0 the width
   * axis is baked into the export at 12 and the decoder returns every width the
   * tensor carries, so this counts what a `maxWidth` under 12 rejected. Read a
   * zero here on an edge rung as "not measurable", not as "the model proposed
   * nothing too wide".
   */
  overWideSpans: number;
  /** Decoded spans `spanFromTokens` refused to turn into a character range. */
  unmappableSpans: number;
  /**
   * Decoded spans whose score was not a finite number in 0..1, dropped here.
   *
   * The boundary this tier owes core. `packages/core/src/detect/merge.ts` says
   * it in as many words: both of its comparators assume `confidence` is finite,
   * a NaN makes every comparison false, that makes Array#sort's ordering
   * inconsistent, and the determinism everything below depends on is quietly
   * gone -- "so tier adapters must validate confidence at the boundary". This
   * is that validation, and this counter is what stops the drop from looking
   * like a model that found less.
   *
   * REACHABLE, not defensive. `sigmoid` returns NaN for a NaN logit and nothing
   * upstream rejects one: decode.ts's `requireLength` catches a short array (the
   * other NaN source) but a correctly sized tensor CARRYING a NaN passes
   * straight through, and `NaN < threshold` is false, so the cell is emitted as
   * a span rather than filtered. Task 11 measured this runtime returning
   * finite-but-wrong logits from three of four WebGPU rungs, which is one
   * upstream defect away from returning a NaN instead.
   */
  nonFiniteScores: number;
}

const int64 = (values: readonly number[], dims: readonly number[]): OnnxTensor => ({
  dims,
  type: "int64",
  data: BigInt64Array.from(values, BigInt),
});

export class GlinerSpanTagger implements SpanTagger {
  private readonly entry: ModelEntry;

  /** Live counters; see `Tier1TaggerStats`. Read after a run, never reset here. */
  readonly stats: Tier1TaggerStats = {
    inferences: 0,
    droppedWords: 0,
    truncatedWords: 0,
    overWideSpans: 0,
    unmappableSpans: 0,
    nonFiniteScores: 0,
  };

  /**
   * `config` is the WHOLE `Tier1Config`, not a `{ threshold, maxWidth }` subset:
   * `buildLabels` needs `labelForm`, `modelId` picks the graph this session is
   * supposed to be, and an options bag narrower than the config the arm was
   * recorded under is how an arm silently runs labels it did not configure.
   *
   * The signature check is here rather than left to a loader because there is
   * no loader: this constructor is the only place where a session and the
   * config naming its graph meet. The six pinned weight files carry only THREE
   * distinct names between them -- `onnx/model.onnx`, `onnx/model_fp16.onnx`
   * and `onnx/model_quint8.onnx`, each used once by edge and once by base (read
   * off `weightsPath` in manifest.ts) -- so pairing an edge session with a base
   * config is a realistic mistake, and it does not fail cleanly: an edge feed
   * run on base is missing two required inputs, while a base feed run on edge
   * carries two the graph ignores and returns confident, wrongly shaped logits
   * (session.ts, where assertSignature records the reasoning).
   */
  constructor(
    private readonly session: OnnxSession,
    private readonly tokenizer: SubwordTokenizer,
    private readonly config: Tier1Config,
  ) {
    if (!Object.hasOwn(MODEL_MANIFEST, config.modelId)) {
      throw new Error(
        `unknown tier-1 model "${config.modelId}"; available: ${Object.keys(MODEL_MANIFEST).join(", ")}`,
      );
    }
    this.entry = MODEL_MANIFEST[config.modelId] as ModelEntry;
    assertSignature(session, this.entry, config.modelId);
  }

  /**
   * `signal` is honoured BETWEEN segments and nowhere finer, which is the
   * honest granularity available here. `OnnxSession.run` takes feeds and
   * nothing else; onnxruntime-common 1.22.0-dev does declare a
   * `RunOptions.terminate` ("Terminate all incomplete OrtRun calls as soon as
   * possible if true", WebAssembly backend only), but it is session-wide rather
   * than per-run and this seam passes no options, so a fired signal stops the
   * tagger from starting more work rather than unwinding what is already in the
   * graph.
   *
   * An abort keeps the findings already collected instead of discarding them:
   * spec 5.3's degradation is "fall back to the lower tiers' findings", and
   * what this tier already produced is not less trustworthy for having a
   * successor that never ran.
   */
  async tag(segments: Segment[], ir: PolicyIr, signal?: AbortSignal): Promise<Finding[]> {
    const labels = buildLabels(ir, this.config);
    // An empty label set is a legitimate policy, not an error -- a policy may
    // declare no tier-1 entityTypes at all -- and a zero-class model would
    // spend the whole tier-1 budget to return nothing.
    if (labels.length === 0) return [];
    if (labels.length > this.entry.maxTypes) {
      // Counts only, never the label text: labels are policy-derived and the
      // ids are not secret, but an error message is a place things get logged
      // and there is nothing here a count does not already locate.
      throw new Error(
        `${this.config.modelId}: gliner_config.json pins max_types at ` +
          `${String(this.entry.maxTypes)}; this policy has ${String(labels.length)} ` +
          `tier-1 entityTypes`,
      );
    }

    const prompt = {
      labels: labels.map((label) => label.prompt),
      entToken: ENT_TOKEN,
      sepToken: SEP_TOKEN,
    };

    const findings: Finding[] = [];
    for (const segment of segments) {
      if (signal?.aborted === true) break;

      const words = splitWords(segment.text);
      if (words.length === 0) continue;

      const encoded = encodeWords(this.tokenizer, words, { prompt, maxLen: this.entry.maxLen });
      this.stats.droppedWords += encoded.droppedWords;
      this.stats.truncatedWords += encoded.truncatedWords;
      // Everything the segment held tokenised to nothing. `text_lengths` 0
      // would allocate a zero-word axis; there is no span it could describe.
      if (encoded.textLengths === 0) continue;

      this.stats.inferences += 1;
      const outputs = await this.session.run(this.buildFeeds(encoded));
      const spans = this.decode(outputs, encoded.textLengths, labels.length);

      for (const span of spans) {
        // BEFORE anything else is done with the span, because everything after
        // this point either records the score or carries it further. See
        // `Tier1TaggerStats.nonFiniteScores` for why a NaN gets this far and
        // what it costs core's merge if it leaves here.
        if (!(Number.isFinite(span.score) && span.score >= 0 && span.score <= 1)) {
          this.stats.nonFiniteScores += 1;
          continue;
        }
        // markerV0's narrowing, and only markerV0's. Its enumeration width is
        // baked into the export (Task 7: enumerating to 4 instead of 12 fails
        // inside span_rep_layer's Reshape), so the graph always returns twelve
        // widths per word and the only place to narrow is here. token_level has
        // no width axis at all and its decoder is handed `maxWidth` directly,
        // so nothing this branch could catch survives to reach it -- see
        // `Tier1TaggerStats.overWideSpans`.
        if (span.lastWord - span.firstWord + 1 > this.config.maxWidth) {
          this.stats.overWideSpans += 1;
          continue;
        }
        const label = labels[span.classIndex] as Tier1Label | undefined;
        // Unreachable while the class axis is sized from `labels.length`, which
        // `decode` asserts; kept because the alternative to a check here is
        // `undefined.entityType` reaching normalizeFindings.
        if (label === undefined) continue;

        // WORD offsets, not subword offsets: `encoded.words` is the surviving
        // word list, in the same coordinate system the model's word axis uses
        // after the lockstep drop. Handing over the pre-drop `words` array
        // instead is the bug the whole encoder exists to prevent, and it slices
        // cleanly while naming the neighbouring word.
        const local = spanFromTokens(segment.text, encoded.words, span.firstWord, span.lastWord);
        if (local === undefined) {
          // Dropped, never repaired: spanFromTokens already refused to guess,
          // and a span invented here reaches applyActions, which rewrites BY
          // SPAN. One finding lost beats one value leaked.
          this.stats.unmappableSpans += 1;
          continue;
        }

        findings.push({
          // Segment-local to absolute. See this file's header.
          start: segment.start + local.start,
          end: segment.start + local.end,
          text: local.text,
          entityType: label.entityType,
          // normalizeFindings re-derives severity from ir.entityTypes and
          // replaces whatever is here, so inside detect() this value is never
          // read. A caller using this tagger on its own gets exactly this
          // placeholder, and should read severity off the IR itself.
          severity: "low",
          tier: 1,
          // The rung, not a family name: six rungs of one experiment matrix all
          // emit tier-1 findings and the harness has to tell them apart.
          source: this.config.modelId,
          confidence: span.score,
        });
      }
    }
    return findings;
  }

  /** The feed set the pinned graph declares, and nothing beyond it. */
  private buildFeeds(encoded: EncodedWords): Record<string, OnnxTensor> {
    const seqLen = encoded.inputIds.length;
    const feeds: Record<string, OnnxTensor> = {
      input_ids: int64(encoded.inputIds, [1, seqLen]),
      attention_mask: int64(encoded.attentionMask, [1, seqLen]),
      words_mask: int64(encoded.wordsMask, [1, seqLen]),
      // Sized from the SURVIVING words. Task 7 measured this input allocating
      // the word-axis slots: a slot number with no seat fails inside ScatterND,
      // and a seat with no word is padded and silently decoded as a word.
      text_lengths: int64([encoded.textLengths], [1, 1]),
    };
    if (this.entry.spanMode === "markerV0") {
      const { spanIdx, spanMask } = enumerateSpans(encoded.textLengths, this.entry.maxWidth);
      const flat: number[] = [];
      for (const [start, end] of spanIdx) flat.push(start, end);
      feeds["span_idx"] = int64(flat, [1, spanIdx.length, 2]);
      feeds["span_mask"] = {
        dims: [1, spanMask.length],
        type: "bool",
        data: Uint8Array.from(spanMask, (on) => (on ? 1 : 0)),
      };
    }
    return feeds;
  }

  /**
   * Unwraps the logits and hands them to this rung's decoder.
   *
   * The axis check is the guard that a re-pin cannot get past. Nothing
   * downstream would notice a moved axis: `decodeEdgeSpans` and
   * `decodeBaseSpans` read at whatever strides they are told and only reject a
   * total-length mismatch, so a tensor of the right size with `words` and
   * `classes` transposed decodes into confident spans naming words the message
   * does not have. The expected extents are the axis ROLES that
   * `test/fixtures/model-signature.json` recorded by varying one feed dimension
   * at a time -- `[batch, words, classes, 3]` on token_level and `[batch,
   * words, 12, classes]` on markerV0 -- neither of which is what either graph
   * DECLARES its axes to be called.
   */
  private decode(
    outputs: Record<string, OnnxTensor>,
    words: number,
    classes: number,
  ): DecodedSpan[] {
    if (!Object.hasOwn(outputs, LOGITS_OUTPUT)) {
      throw new Error(
        `${this.config.modelId}: no ${LOGITS_OUTPUT} in the run output, got ` +
          `${Object.keys(outputs).join(", ") || "nothing"}`,
      );
    }
    const tensor = outputs[LOGITS_OUTPUT] as OnnxTensor;
    const expected =
      this.entry.spanMode === "token_level"
        ? [1, words, classes, EDGE_SLOTS]
        : [1, words, this.entry.maxWidth, classes];
    if (
      tensor.dims.length !== expected.length ||
      tensor.dims.some((extent, axis) => extent !== expected[axis])
    ) {
      throw new Error(
        `${this.config.modelId}: logits axes [${tensor.dims.join(", ")}] are not the pinned ` +
          `[${expected.join(", ")}] layout for ${this.entry.spanMode}`,
      );
    }
    // Task 7 measured float32 logits on every rung that loads at all, the uint8
    // ones included -- they dequantise inside the graph. A different array type
    // would be read as garbage by both decoders.
    const data = tensor.data;
    if (!(data instanceof Float32Array)) {
      throw new Error(
        `${this.config.modelId}: logits must arrive as float32, got ${tensor.type}`,
      );
    }
    // Batch is 1 -- asserted by the axis check above -- so the whole array is
    // the single row both decoders expect.
    return this.entry.spanMode === "token_level"
      ? decodeEdgeSpans(
          data,
          { words, classes, slots: EDGE_SLOTS },
          this.config.threshold,
          // The same width the post-decode filter below applies, handed to the
          // decoder so it stops enumerating candidates that filter would throw
          // away. The emitted set is identical; see the width-bound note in
          // decode.ts for what changes and what does not.
          this.config.maxWidth,
        )
      : decodeBaseSpans(
          data,
          { words, widths: this.entry.maxWidth, classes },
          this.config.threshold,
        );
  }
}
