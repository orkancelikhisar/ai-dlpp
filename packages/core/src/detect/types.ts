import type { Action, PolicyIr, Severity, Tier } from "../policy/types.js";
import type { Segment } from "../segment/segment.js";

export interface Finding {
  /** Absolute character offsets into the original message text. */
  start: number;
  end: number;
  /** Exact matched text: always === message.slice(start, end). */
  text: string;
  entityType: string;
  severity: Severity;
  tier: Tier;
  /** Rule id (tier 0) or model identifier (tiers 1-2). */
  source: string;
  /** 0..1 */
  confidence: number;
}

export interface ResolvedFinding extends Finding {
  action: Action;
}

export interface TierConfig {
  tier0: boolean;
  tier1: boolean;
  tier2: boolean;
  t1Model?: string;
  t2Model?: string;
  backend?: "wasm" | "webgpu";
}

/**
 * Seam for tiers 1-2 (Plans 4-5). Labels come from ir.entityTypes at inference.
 *
 * Implementers MUST re-derive `text` as message.slice(start, end) from the
 * offsets they report, and never pass through model-produced span text:
 * tokenizer offset drift makes the two disagree, and everything downstream
 * relies on Finding.text matching the offsets exactly.
 *
 * `signal` is a real cancellation channel, not decoration. Per spec 5.3 a
 * tier-2 run that exceeds latencyBudgetMs degrades to the tier 0/1 findings;
 * racing a timeout alone would leak the still-running inference, which then
 * serializes behind the engine and delays the *next* message's tier-2.
 * Implementers should abort in-flight work when the signal fires.
 */
export interface SpanTagger {
  tag(segments: Segment[], ir: PolicyIr, signal?: AbortSignal): Promise<Finding[]>;
}
export interface SemanticJudge {
  judge(
    segments: Segment[],
    ir: PolicyIr,
    priorFindings: Finding[],
    signal?: AbortSignal,
  ): Promise<Finding[]>;
}

export interface DetectorEngines {
  tier1?: SpanTagger;
  tier2?: SemanticJudge;
}

export interface DetectionResult {
  /**
   * Pairwise DISJOINT and sorted by start offset -- overlaps were already
   * resolved when this array was built. A rewriter (Plan 2's redaction and
   * pseudonymization pass) can therefore splice every span in one walk without
   * checking for collisions or re-sorting; walking it in reverse is what keeps
   * the remaining offsets valid, since a replacement of a different length
   * shifts everything after it.
   */
  findings: ResolvedFinding[];
  /**
   * Wall-clock per tier, measured around the tier's own work only. A tier that
   * did not run has no entry -- except `tier0Ms`, which is required by this type
   * and reads 0 when `config.tier0` was false. Zero there means "did not run",
   * not "ran instantly", so read it against the TierConfig that produced it
   * before charting it as a latency.
   */
  timings: { tier0Ms: number; tier1Ms?: number; tier2Ms?: number };
}
