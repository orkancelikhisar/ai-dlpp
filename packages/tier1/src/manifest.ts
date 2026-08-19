/**
 * The pinned model file set.
 *
 * Deliberately free of both DOM and Node types: scripts/fetch-models.ts imports
 * it from the repo root, where packages/compiler's tsconfig pulls it into a
 * DOM-free program, while packages/tier1 itself is browser-only. Keeping the
 * manifest here leaves config.ts free to grow DOM-typed session options without
 * breaking that script's typecheck.
 */

export interface ModelFile {
  readonly sha256: string;
  readonly bytes: number;
}

/**
 * How the model proposes spans. MEASURED from each repo's gliner_config.json.
 * Not cosmetic: it goes with a different ONNX graph signature, see `inputNames`.
 */
export type SpanMode = "token_level" | "markerV0";

export interface ModelEntry {
  readonly repo: string;
  /** Immutable commit sha. Every file below is resolved at this one revision. */
  readonly revision: string;
  /** Key into `files` naming the weights, as opposed to the metadata alongside them. */
  readonly weightsPath: string;
  readonly spanMode: SpanMode;
  /**
   * Graph input names in declaration order, MEASURED by parsing the ONNX
   * ModelProto of the exact pinned blob. The two models differ here -- markerV0
   * takes `span_idx`/`span_mask` that token_level does not -- so a loader must
   * branch on this rather than assume one signature.
   */
  readonly inputNames: readonly string[];
  /** gliner_config.json `max_len`: the truncation boundary, so it moves span offsets. */
  readonly maxLen: number;
  /** gliner_config.json `max_width`, the width the model was trained against. */
  readonly maxWidth: number;
  /** Every file needed to reproduce a load, keyed by repo-relative path. */
  readonly files: Readonly<Record<string, ModelFile>>;
}

/** Written into an entry until scripts/fetch-models.ts establishes the real digest. */
export const MODEL_FILE_PLACEHOLDER = "<fill from fetch-models.ts>";

export function modelFileUrl(entry: ModelEntry, path: string): string {
  return `https://huggingface.co/${entry.repo}/resolve/${entry.revision}/${path}`;
}

/**
 * Every tier-1 model the experiment matrix can select, pinned by content hash
 * at one immutable revision.
 *
 * The pin covers the whole file set, not just the weights. MEASURED: between
 * two revisions of BOTH repos, gliner_config.json changed `max_len` 1024 ->
 * 2048 under a stable filename. max_len is the truncation boundary, so it
 * decides which spans exist and at what offsets -- pinning only model.onnx
 * would reproduce byte-identical weights while the effective model moved.
 *
 * Fill each `sha256`/`bytes` by running scripts/fetch-models.ts, which verifies
 * a freshly downloaded file against the Hub tree API before printing anything.
 */
export const MODEL_MANIFEST: Readonly<Record<string, ModelEntry>> = {
  "gliner-pii-edge": {
    repo: "knowledgator/gliner-pii-edge-v1.0",
    revision: "9b7f39b0a2da971a5beea78d35f1539d4009c891",
    weightsPath: "onnx/model.onnx",
    spanMode: "token_level",
    inputNames: ["input_ids", "attention_mask", "words_mask", "text_lengths"],
    maxLen: 2048,
    maxWidth: 12,
    files: {
      "onnx/model.onnx": { sha256: "4ca588722e6d79447ad4c9c230eeba3d9d472c672a9598184a34e9f77fc35836", bytes: 181078966 },
      "gliner_config.json": { sha256: "77e6b57335c4bfd461e9041682196dd6c373a0b09bbd9269ef9e95b807915340", bytes: 4316 },
      "tokenizer.json": { sha256: "84b3a9b18f04a0ccd03b72d9f871b7e0bec40fd7021ef50bc30a7c3693c11205", bytes: 3583593 },
      "tokenizer_config.json": { sha256: "3398f6d1ad4b4c4f9874d390d060a75c58cad5e5ce9b22841b3e40643b4ada27", bytes: 21214 },
      "special_tokens_map.json": { sha256: "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049", bytes: 694 },
    },
  },
  "gliner-pii-base": {
    repo: "knowledgator/gliner-pii-base-v1.0",
    revision: "61726e0ad791dcab3e29339bbec3ad42ded65641",
    weightsPath: "onnx/model.onnx",
    spanMode: "markerV0",
    inputNames: [
      "input_ids",
      "attention_mask",
      "words_mask",
      "text_lengths",
      "span_idx",
      "span_mask",
    ],
    maxLen: 2048,
    maxWidth: 12,
    files: {
      "onnx/model.onnx": { sha256: "c6ccec44625d46bfe3191152e41d6564b69bc9d4313b7f3e419e8372679e9fed", bytes: 664764803 },
      "gliner_config.json": { sha256: "e33d3da38e0d369fa7574668d3798ca6c7d2b23cba7d628507112eeb426aaccb", bytes: 3902 },
      "tokenizer.json": { sha256: "ee028763434d18611c1c36356ea1d050e90a9fa94ede57fac48b39f85f818ad1", bytes: 8649232 },
      "tokenizer_config.json": { sha256: "3ec8a90d8758fbc56d50831990c3a3a65660f020c5b06534adf43b04091ffa9e", bytes: 1691 },
      "special_tokens_map.json": { sha256: "b2f1b2f15f29a6b6d9d6ea4eca1675d2c231a71477f151d48f79cc83a625ba21", bytes: 970 },
      "added_tokens.json": { sha256: "c358eb74586ab438484d8acf4534f67283b33041bc5ffee6b20a4a075cdc3cd6", bytes: 65 },
      "spm.model": { sha256: "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd", bytes: 2464616 },
    },
  },
};
