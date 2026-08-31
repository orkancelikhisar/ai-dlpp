export {
  DeadlineExpired,
  mlcInterruptible,
  runWithDeadline,
  type Interruptible,
} from "./cancel.js";
// `buildCallParams` and `CreateEngineFn` are exported from `engine.ts` for its
// own tests and stop there. A caller holding `buildCallParams`' output can hand
// it straight to `engine.chat.completions.create`, which skips the deadline,
// the drain, the interrupt-clear and the per-engine serialization -- the four
// measured hazards this package exists to hold -- and `CreateEngineFn` is a
// test seam, not API. `WebLlmEngine` is a TYPE here and not a value: its
// constructor is private, so `createWebLlmEngine` is the only way to build one
// and the only place that can refuse a modelId disagreeing with its config.
// Same reasoning as `schema.ts` keeping `classifyJsonPrefix` private.
// Spec 4.1's escalation policy, re-exported from `@sih/core`; see escalate.ts
// for why it cannot be defined in this package.
export {
  UNCERTAIN_BELOW,
  selectSegments,
  uncertainSegmentStarts,
  type EscalationInput,
} from "./escalate.js";
// The Approach-B baseline: core's `Detector` implemented directly, with no
// compiler and no tiers, so the eval harness runs it as just another arm.
// `completionCallRecord` and `priorFindingsLine` are shared between this arm
// and the judge inside the package and stop at their own modules -- they are
// how the two arms cannot drift, not API.
export {
  createBaselineB,
  createBaselineBPlusTier0,
  type BaselineB,
  type BaselineBOptions,
  type BaselineStats,
} from "./baselineB.js";
export {
  createWebLlmEngine,
  type CompleteOptions,
  type Tier2CallParams,
  type Tier2Completion,
  type Tier2Engine,
  type WebLlmEngine,
} from "./engine.js";
export {
  WebLlmJudge,
  type JudgeCallRecord,
  type JudgeStats,
  type WebLlmJudgeOptions,
} from "./judge.js";
export {
  DEFAULT_TIER2_CONFIG,
  TIER2_MODELS,
  resolveTier2Config,
  type Tier2Config,
  type Tier2Model,
} from "./manifest.js";
export {
  BASELINE_B_SCHEMA,
  BaselineResponseSchema,
  JUDGE_SCHEMA,
  JudgeResponseSchema,
  parseBaselineResponse,
  parseJudgeResponse,
  type BaselineResponse,
  type JudgeResponse,
  type ParseResult,
} from "./schema.js";
export {
  MINIMUM_CANDIDATE_WORDS,
  resolveQuote,
  type ResolvedQuote,
} from "./spans.js";
