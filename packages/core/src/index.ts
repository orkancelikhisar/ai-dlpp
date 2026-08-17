export const CORE_VERSION = "0.0.1";

export type * from "./policy/types.js";
export { PolicyIrSchema } from "./policy/schema.js";
export { loadPolicyIr, PolicyLoadError, PolicyVersionError, SUPPORTED_IR_VERSION } from "./policy/load.js";
export { resolveAction } from "./policy/resolve.js";
export { segmentText, type Segment, type SegmentKind } from "./segment/segment.js";
export { getValidator, hasValidator, shannonEntropy, type Validator } from "./detect/validators.js";
export { runTier0 } from "./detect/tier0.js";
export { mergeFindings, clusterOverlapping } from "./detect/merge.js";
export { detect, type DetectInput, type Detector } from "./detect/orchestrator.js";
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
export {
  createRehydrateTransform,
  /**
   * Exported for adapters, which need it to reason about the streaming
   * transform's cost before they wire one up: `createRehydrateTransform` holds
   * back up to this many characters at every chunk boundary so a surrogate
   * split across two chunks is still matched, and that holdback is the latency
   * a token appears to gain on its way to the user. An adapter deciding whether
   * to stream at all, or sizing its own buffers around one, is reading this
   * number. `surrogatePattern` is deliberately NOT exported alongside it: the
   * compiled alternation encodes this layer's matching rules (longest-first
   * ordering, the digit boundary on both edges), all of which belong to
   * `rehydrateText`/`createRehydrateTransform` rather than to their callers.
   */
  maxSurrogateLength,
  rehydrateText,
} from "./pseudo/rehydrate.js";
