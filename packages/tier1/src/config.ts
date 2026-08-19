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

export const TIER1_LABEL_FORMS = ["id", "definition", "id-and-definition"] as const;
/**
 * What text each tier-1 class is prompted with, built from one entityType:
 *
 * - `id` -- the entityType id with hyphens spaced out, e.g. `"client name"`.
 * - `definition` -- the entityType's `nlDefinition`, verbatim.
 * - `id-and-definition` -- both, joined. Spec §4.1 says the injected labels are
 *   `entityTypes[].{id, nlDefinition}`, so this is that sentence read literally.
 *
 * DECISION: `id` is the default, which is a DEVIATION from spec §4.1. It is
 * taken on cost grounds, not on accuracy grounds.
 *
 * The cost is measured. Tokenising with each pinned `tokenizer.json`, the `id`
 * form of `client-name` is 2 tokens, while the three `nlDefinition` values the
 * compiler actually produced for that same entityType (in
 * packages/compiler/test/fixtures/llm) are 26-37 tokens on edge and 26-35 on
 * base. The manifest's `inputNames` -- measured by parsing the ONNX graphs --
 * carry exactly one token-id input, `input_ids`, and no label-side input, so
 * label text has nowhere to go except the same sequence as the message text,
 * against the same `maxLen`. That per-class cost multiplies by the number of
 * tier-1 classes in the policy.
 *
 * UNVERIFIED: which form scores better. No inference runs in this package yet,
 * so nothing here has been scored, and no claim about the label encoder's
 * training distribution is being made -- note that both pinned
 * `gliner_config.json` files have `labels_encoder: null`, so there is no
 * separate label tower to reason about in the first place. What would settle
 * it: one corpus and one policy under arms differing only in this field,
 * compared on span F1 (spec §6.4 metric 2). Until that runs, `id` is a default,
 * not a finding.
 */
export type Tier1LabelForm = (typeof TIER1_LABEL_FORMS)[number];

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
  /** How `buildLabels` renders each class's prompt. See `Tier1LabelForm`. */
  readonly labelForm: Tier1LabelForm;
}

export const DEFAULT_TIER1_CONFIG: Tier1Config = {
  modelId: "gliner-pii-edge",
  backend: "wasm",
  threshold: 0.5,
  maxWidth: 12,
  labelForm: "id",
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
  if (!TIER1_LABEL_FORMS.includes(config.labelForm)) {
    throw new Error(
      `tier-1 labelForm must be one of ${TIER1_LABEL_FORMS.join(", ")}, got ${config.labelForm}`,
    );
  }
  return config;
}
