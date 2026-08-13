import type { PolicyIr, Rule } from "../policy/types.js";
import type { Segment } from "../segment/segment.js";
import type { Finding } from "./types.js";
import { getValidator } from "./validators.js";

/**
 * Tier 0 -- spec 4.1. Deterministic regex rules over the FULL message text.
 *
 * Scanning the whole message rather than each segment is what makes spans
 * absolute for free (a match index IS the offset into the message), and it
 * keeps entities that straddle a segment boundary detectable. Segments are
 * passed in for the entropy rules (Task 10), which need per-segment windows.
 */

const BASE_CONFIDENCE = 0.9;
const CONTEXT_BONUS = 0.05;
/** Characters of context inspected on each side of a match for boost keywords. */
const CONTEXT_WINDOW = 40;
const MAX_CONFIDENCE = 0.99;

/**
 * Severity is RE-DERIVED from ir.entityTypes for every finding and never carried
 * along from anywhere else: the entityType table is the single source of truth,
 * so a rule (or a later tier) cannot inflate or downgrade what a policy says an
 * entity is worth. The throw is a can't-happen guard -- the schema already
 * rejects a rule referencing an undeclared entityType -- kept as defense in
 * depth, since silently defaulting a severity would be a quiet policy downgrade.
 * Called once per rule (the lookup is match-independent), so a rule that matches
 * nothing still trips the guard: fail loudly on a malformed IR either way.
 */
function severityOf(ir: PolicyIr, entityTypeId: string) {
  const e = ir.entityTypes.find((et) => et.id === entityTypeId);
  if (!e) throw new Error(`rule references unknown entityType "${entityTypeId}"`);
  return e.severity;
}

/**
 * Substring hit for any of `lowerKeywords` (already lowercased by the caller)
 * within +/-CONTEXT_WINDOW of the match. The keyword must fall ENTIRELY inside
 * the window: one straddling the boundary does not count, so the effective reach
 * is CONTEXT_WINDOW minus the keyword's own length. Boost is a nudge, not a
 * decision, so the fuzziness is acceptable -- but do not read the constant as
 * "distance to the keyword".
 */
function hasNearbyKeyword(text: string, start: number, end: number, lowerKeywords: string[]): boolean {
  const window = text
    .slice(Math.max(0, start - CONTEXT_WINDOW), Math.min(text.length, end + CONTEXT_WINDOW))
    .toLowerCase();
  return lowerKeywords.some((k) => window.includes(k));
}

function runRegexRule(ir: PolicyIr, rule: Rule, text: string): Finding[] {
  const findings: Finding[] = [];
  // The IR carries regex SOURCE only; flags are ours. "g" is added here so one
  // compiled regex can walk the whole message, and compiling per call keeps
  // lastIndex private to this scan (a cached /g/ regex is stateful and would
  // make results depend on the previous message).
  const re = new RegExp(rule.regex!, "g");
  const validator = rule.validator ? getValidator(rule.validator) : undefined;
  // Loop-invariant per rule: the severity lookup and the keyword lowering do not
  // depend on the match, so they happen once rather than once per match.
  const severity = severityOf(ir, rule.entityType);
  const boostKeywords = rule.contextBoost?.length ? rule.contextBoost.map((k) => k.toLowerCase()) : undefined;
  for (let m = re.exec(text); m !== null; m = re.exec(text)) {
    // Suspenders to the loader's nullable-regex check, which only probes offset 0:
    // a lookbehind can still match empty mid-scan, and exec does not advance past a
    // zero-width match. Step over it and keep scanning -- abandoning the rule here
    // would silently drop its real matches.
    if (m[0].length === 0) {
      re.lastIndex++;
      continue;
    }
    // The validator is the false-positive gate: regex shape alone accepts
    // things like ABCXD1234E, which is not a structurally valid PAN.
    if (validator && !validator(m[0])) continue;
    const boosted = boostKeywords ? hasNearbyKeyword(text, m.index, m.index + m[0].length, boostKeywords) : false;
    findings.push({
      start: m.index,
      end: m.index + m[0].length,
      // m[0] IS text.slice(m.index, m.index + m[0].length) by construction, which
      // is the span-fidelity invariant Finding.text documents. Never substitute a
      // normalized or re-cased string here.
      text: m[0],
      entityType: rule.entityType,
      severity,
      tier: 0,
      source: rule.id,
      confidence: Math.min(MAX_CONFIDENCE, BASE_CONFIDENCE + (boosted ? CONTEXT_BONUS : 0)),
    });
  }
  return findings;
}

/**
 * Run every tier-0 rule over `text`, returning findings sorted by start offset.
 *
 * Overlapping and duplicate findings are all reported -- two rules may cover the
 * same span, and deciding which one wins is mergeFindings' job in Task 11, not
 * this function's. Equal starts keep insertion order (rule declaration order,
 * then match order within a rule) because Array#sort is stable per spec; that is
 * a described consequence, not a contract downstream should lean on.
 */
export function runTier0(ir: PolicyIr, text: string, segments: Segment[]): Finding[] {
  const findings: Finding[] = [];
  for (const rule of ir.rules) {
    if (rule.regex !== undefined) {
      findings.push(...runRegexRule(ir, rule, text));
    }
    // SEAM: entropy rules (rule.entropyThreshold !== undefined) land here in
    // Task 10 and consume `segments` -- they score sliding windows per segment
    // so that code blocks and kv values can be treated differently from prose.
  }
  void segments;
  return findings.sort((a, b) => a.start - b.start);
}
