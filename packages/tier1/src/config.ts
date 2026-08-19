import { MODEL_MANIFEST } from "./manifest.js";

export {
  MODEL_FILE_PLACEHOLDER,
  MODEL_MANIFEST,
  modelFileUrl,
  type ModelEntry,
  type ModelFile,
  type SpanMode,
} from "./manifest.js";

export const TIER1_BACKENDS = ["wasm", "webgpu"] as const;
export type Tier1Backend = (typeof TIER1_BACKENDS)[number];

export interface Tier1Config {
  readonly modelId: string;
  readonly backend: Tier1Backend;
  /** Span score below this is discarded. An experiment variable, not a constant. */
  readonly threshold: number;
  /**
   * Widest span in WORDS the run will accept. Defaults to 12 because MEASURED:
   * that is `max_width` in both models' gliner_config.json, i.e. the width they
   * were trained against. How a model applies it differs by span mode, so this
   * is a ceiling the run imposes rather than a description of the graph -- see
   * `ModelEntry.spanMode` and the loader that consumes it.
   */
  readonly maxWidth: number;
}

export const DEFAULT_TIER1_CONFIG: Tier1Config = {
  modelId: "gliner-pii-edge",
  backend: "wasm",
  threshold: 0.5,
  maxWidth: 12,
};

/**
 * Drops keys whose value is undefined so they do not shadow a default.
 *
 * A bare `{ ...DEFAULT, ...overrides }` copies an own key holding undefined
 * over the default. The matrix and CLI both build overrides positionally, so
 * "option not specified" arrives as exactly that shape and would otherwise
 * throw on a missing option instead of taking the default.
 */
function definedOnly(overrides: Partial<Tier1Config>): Partial<Tier1Config> {
  return Object.fromEntries(
    Object.entries(overrides).filter(([, value]) => value !== undefined),
  ) as Partial<Tier1Config>;
}

export function resolveTier1Config(overrides: Partial<Tier1Config>): Tier1Config {
  const config = { ...DEFAULT_TIER1_CONFIG, ...definedOnly(overrides) };
  if (!Object.hasOwn(MODEL_MANIFEST, config.modelId)) {
    throw new Error(
      `unknown tier-1 model "${config.modelId}"; available: ${Object.keys(MODEL_MANIFEST).join(", ")}`,
    );
  }
  if (!TIER1_BACKENDS.includes(config.backend)) {
    throw new Error(
      `tier-1 backend must be one of ${TIER1_BACKENDS.join(", ")}, got ${config.backend}`,
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
