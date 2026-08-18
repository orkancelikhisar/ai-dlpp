export const CORE_VERSION = "0.0.1";

export type * from "./policy/types.js";
export { PolicyIrSchema } from "./policy/schema.js";
export { loadPolicyIr, PolicyLoadError, PolicyVersionError, SUPPORTED_IR_VERSION } from "./policy/load.js";
export { resolveAction } from "./policy/resolve.js";
export { segmentText, type Segment, type SegmentKind } from "./segment/segment.js";
export { getValidator, hasValidator, shannonEntropy, type Validator } from "./detect/validators.js";
export { runTier0 } from "./detect/tier0.js";
export { mergeFindings, clusterOverlapping } from "./detect/merge.js";
export { ACTION_RANK, detect, type DetectInput, type Detector } from "./detect/orchestrator.js";
export type * from "./detect/types.js";

// -- pseudonymization layer (spec §5.4) ------------------------------------
export { fnv1a64, mulberry32, seededRng } from "./pseudo/seed.js";
export { generateSurrogate } from "./pseudo/generators.js";
export { MemoryVaultStore, Vault, type VaultEntry, type VaultRecord, type VaultStore } from "./pseudo/vault.js";
export {
  decryptString,
  encryptString,
  exportVaultKey,
  generateVaultKey,
  importVaultKey,
  // The alias, not just the functions: this package compiles under `lib:
  // ES2022`, where `CryptoKey` is not a global type name at all, so a consumer
  // storing a vault key in a typed field (Plan 6's IndexedDB store, the session
  // key in chrome.storage.session) has no way to name what these return.
  type CryptoKey,
} from "./pseudo/crypto.js";
export {
  applyActions,
  type AppliedReplacement,
  type ApplyResult,
  type SkippedSpan,
} from "./pseudo/apply.js";
// `maxSurrogateLength` is here for adapters sizing the streaming holdback, and
// `surrogatePattern` is deliberately absent -- the reasoning for both lives on
// `maxSurrogateLength` itself, where hover and declaration emit will show it
// (JSDoc on a brace-list specifier reaches neither).
export { createRehydrateTransform, maxSurrogateLength, rehydrateText } from "./pseudo/rehydrate.js";
