export interface ModelEntry {
  readonly url: string;
  readonly sha256: string;
  readonly bytes: number;
  /**
   * Hub repo id the tokenizer files are loaded from, kept separate from `url`
   * because the two resolve by different mechanisms: `url` names one blob
   * directly, while the tokenizer is a repo the loader reads several files out
   * of (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` --
   * all three verified present in both repos via the Hub model API).
   *
   * Note this id carries no revision, so the tokenizer is NOT pinned the way
   * the weights are. Pinning it belongs to the task that first loads it, which
   * is the first point at which the pin can be verified by running it.
   */
  readonly tokenizer: string;
}

/**
 * Every tier-1 model the experiment matrix can select, pinned by content hash.
 *
 * Fill `sha256` and `bytes` by running scripts/fetch-models.ts, which prints
 * both for a downloaded file. They are deliberately NOT optional -- an unpinned
 * entry would let a silently-updated upstream file invalidate measurements that
 * were already taken and reported.
 *
 * Each `url` names an immutable commit sha rather than `main`. MEASURED: both
 * forms resolve to the same CDN blob today, but a `main` url only lets the hash
 * check *detect* an upstream change, whereas the commit form still *serves* the
 * exact bytes these hashes were taken from, so an already-reported measurement
 * stays reproducible instead of merely being flagged as unreproducible.
 */
export const MODEL_MANIFEST: Readonly<Record<string, ModelEntry>> = {
  "gliner-pii-edge": {
    url: "https://huggingface.co/knowledgator/gliner-pii-edge-v1.0/resolve/9b7f39b0a2da971a5beea78d35f1539d4009c891/onnx/model.onnx",
    sha256: "4ca588722e6d79447ad4c9c230eeba3d9d472c672a9598184a34e9f77fc35836",
    bytes: 181078966,
    tokenizer: "knowledgator/gliner-pii-edge-v1.0",
  },
  "gliner-pii-base": {
    url: "https://huggingface.co/knowledgator/gliner-pii-base-v1.0/resolve/61726e0ad791dcab3e29339bbec3ad42ded65641/onnx/model.onnx",
    sha256: "c6ccec44625d46bfe3191152e41d6564b69bc9d4313b7f3e419e8372679e9fed",
    bytes: 664764803,
    tokenizer: "knowledgator/gliner-pii-base-v1.0",
  },
};

export interface Tier1Config {
  readonly modelId: string;
  readonly backend: "wasm" | "webgpu";
  /** Span score below this is discarded. An experiment variable, not a constant. */
  readonly threshold: number;
  /** Widest span in WORDS the model may propose. GLiNER span mode enumerates all widths up to this. */
  readonly maxWidth: number;
}

export const DEFAULT_TIER1_CONFIG: Tier1Config = {
  modelId: "gliner-pii-edge",
  backend: "wasm",
  threshold: 0.5,
  maxWidth: 12,
};

export function resolveTier1Config(overrides: Partial<Tier1Config>): Tier1Config {
  const config = { ...DEFAULT_TIER1_CONFIG, ...overrides };
  if (!Object.hasOwn(MODEL_MANIFEST, config.modelId)) {
    throw new Error(
      `unknown tier-1 model "${config.modelId}"; available: ${Object.keys(MODEL_MANIFEST).join(", ")}`,
    );
  }
  if (!(config.threshold > 0 && config.threshold <= 1)) {
    throw new Error(`tier-1 threshold must be in (0, 1], got ${config.threshold}`);
  }
  if (!(Number.isInteger(config.maxWidth) && config.maxWidth > 0)) {
    throw new Error(`tier-1 maxWidth must be a positive integer, got ${config.maxWidth}`);
  }
  return config;
}
