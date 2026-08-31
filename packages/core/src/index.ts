export const CORE_VERSION = "0.0.1";

export type * from "./policy/types.js";
export { PolicyIrSchema } from "./policy/schema.js";
// The naming contract between the compiler that mints a shadow entityType per
// semantic predicate and the browser-side tier-2 judge that must name one. Here
// and not in `@sih/compiler` because that package is Node-only; see the module.
export { SHADOW_PREFIX, shadowIdFor } from "./policy/predicates.js";
export { loadPolicyIr, PolicyLoadError, PolicyVersionError, SUPPORTED_IR_VERSION } from "./policy/load.js";
export { resolveAction } from "./policy/resolve.js";
export { segmentText, type Segment, type SegmentKind } from "./segment/segment.js";
export { getValidator, hasValidator, shannonEntropy, type Validator } from "./detect/validators.js";
export { runTier0 } from "./detect/tier0.js";
export { mergeFindings, clusterOverlapping } from "./detect/merge.js";
// `resolveFindings` and `remainingBudgetMs` are here for the Approach-B
// baseline, which implements `Detector` directly and is therefore its own
// orchestrator: it has to validate findings against the IR, resolve overlaps
// and arm the message budget exactly as `detect` does, and a second copy of
// either would make the head-to-head measure the harness. `normalizeFindings`
// stays private -- `resolveFindings` is the door, and it normalizes first, so
// no caller can hold a partly-checked result.
export {
  ACTION_RANK,
  detect,
  remainingBudgetMs,
  resolveFindings,
  type DetectInput,
  type Detector,
} from "./detect/orchestrator.js";
// Spec 4.1's escalation policy. Here rather than in `@sih/tier2` -- where Plan 5
// named it -- because `detect` is its caller and core cannot depend on the tier
// it gates; see the module. `@sih/tier2` re-exports it, as `@sih/compiler` does
// for `shadowIdFor`.
export {
  UNCERTAIN_BELOW,
  selectSegments,
  uncertainSegmentStarts,
  type EscalationInput,
} from "./detect/escalate.js";
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
