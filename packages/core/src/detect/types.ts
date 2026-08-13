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
  findings: ResolvedFinding[];
  timings: { tier0Ms: number; tier1Ms?: number; tier2Ms?: number };
}
