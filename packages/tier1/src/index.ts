export {
  DEFAULT_TIER1_CONFIG,
  MODEL_FILE_PLACEHOLDER,
  MODEL_MANIFEST,
  modelFileUrl,
  resolveTier1Config,
  TIER1_BACKENDS,
  TIER1_LABEL_FORMS,
  type ModelEntry,
  type ModelFile,
  type SpanMode,
  type Tier1Backend,
  type Tier1Config,
  type Tier1LabelForm,
} from "./config.js";
export { buildLabels, type Tier1Label } from "./labels.js";
export { spanFromTokens, type CharSpan, type TokenOffset } from "./offsets.js";
