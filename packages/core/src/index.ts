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
