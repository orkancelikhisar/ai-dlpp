import type { PolicyIr, Rule } from "../policy/types.js";
import type { Segment } from "../segment/segment.js";
import type { Finding } from "./types.js";
import { getValidator, shannonEntropy } from "./validators.js";

/**
 * Tier 0 -- spec 4.1. Deterministic rules: regex over the FULL message text,
 * entropy per segment.
 *
 * Scanning the whole message rather than each segment is what makes regex spans
 * absolute for free (a match index IS the offset into the message), and it
 * keeps entities that straddle a segment boundary detectable. Entropy rules
 * cannot work that way -- they are scoped to code and kv segments, so they walk
 * `segments` and rebase each match onto the segment's absolute start.
 */

const BASE_CONFIDENCE = 0.9;
const CONTEXT_BONUS = 0.05;
/**
 * Entropy findings are FIXED at 0.7 -- below any regex rule's floor. Entropy is
 * a heuristic over shape alone: it knows a string looks random, never that it is
 * a secret. Raising the confidence is tier 1-2's job (or the user's), so nothing
 * here varies it.
 */
const ENTROPY_CONFIDENCE = 0.7;
/**
 * Default candidate length when a rule omits `minLength`.
 *
 * How threshold and length interact, since the two are easy to set blindly:
 * Shannon entropy is bounded by h <= log2(distinct chars) <= log2(length), so
 * length caps what a threshold can ever see. At the default pairing (threshold
 * 4.0, minLength 20) a candidate needs at least 17 DISTINCT characters to clear
 * the bar -- 16 distinct over 20 characters maxes out at 3.92 bits/char. Two
 * consequences worth knowing before writing a policy:
 *
 * - A threshold above log2(67) = 6.07 (the secret alphabet's size) can never
 *   fire at any length. The schema rejects those outright.
 * - Hex-only secrets have 16 symbols, so they cap at EXACTLY log2(16) = 4.0 and
 *   only at perfect uniformity. A 4.0 threshold therefore makes sha/md5-style
 *   hex tokens effectively undetectable; a policy meant to cover them wants
 *   roughly 3.5.
 */
const DEFAULT_MIN_LENGTH = 20;
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
 * Entropy rules -- spec 4.1. Unlike regex rules these scan PER SEGMENT, and only
 * code and kv segments: a high-entropy run in prose is far more often a hash, an
 * id, or a URL slug than a credential, and scanning prose drowns the user in
 * false positives. The cost is that a secret pasted mid-sentence is missed here;
 * tiers 1-2 are what cover that.
 *
 * Candidates are maximal runs of the secret alphabet, so a run is scored as the
 * user wrote it rather than as an arbitrary window. Note the alphabet contains
 * "=" and "_", so `KEY=<secret>` in a kv line is ONE run: the span reported is
 * wider than the secret itself. Accepted for now -- an over-inclusive span still
 * redacts the secret, and over-inclusion is the safe direction.
 *
 * Takes no `text` parameter on purpose: segments already carry their own slice
 * and their absolute `start`, so `seg.start + m.index` is the message offset.
 *
 * Scanning per segment cannot truncate a run only because every boundary the
 * segmenter produces is flanked by "\n", "\r", or a backtick -- none of which
 * are in the secret alphabet, so no candidate straddles one. A future segmenter
 * that splits mid-line would break that and silently cut runs in half; it would
 * have to scan across the boundary or rejoin adjacent scannable segments.
 */
function runEntropyRule(ir: PolicyIr, rule: Rule, segments: Segment[]): Finding[] {
  const findings: Finding[] = [];
  const minLength = rule.minLength ?? DEFAULT_MIN_LENGTH;
  // Loop-invariant per rule, as in runRegexRule.
  const severity = severityOf(ir, rule.entityType);
  // Compiled per call for the same reason as runRegexRule's: a module-level /g/
  // regex keeps lastIndex between calls, so an exception mid-scan would leave
  // the NEXT message's scan starting at a stale offset. No explicit lastIndex
  // reset between segments: exec() zeroes it when it returns null, which is the
  // only way the inner loop exits, and an exception aborts the whole call along
  // with this call-local regex.
  //
  // Alphabet excludes "." deliberately. Including it would glue filenames,
  // version strings, and dotted config paths into one long run and score the
  // noise; the cost is that a JWT fragments into its three parts. Detecting a
  // JWT as one entity is the jwt-shape REGEX rule's job, not entropy's.
  const secretRun = /[A-Za-z0-9+/=_-]+/g;
  for (const seg of segments) {
    if (seg.kind === "prose") continue;
    for (let m = secretRun.exec(seg.text); m !== null; m = secretRun.exec(seg.text)) {
      // "+" quantifier: a match is never zero-width, so no skip-and-advance
      // guard is needed here (unlike runRegexRule, whose pattern is policy-supplied).
      if (m[0].length < minLength) continue;
      if (shannonEntropy(m[0]) < rule.entropyThreshold!) continue;
      const start = seg.start + m.index;
      findings.push({
        start,
        end: start + m[0].length,
        text: m[0],
        entityType: rule.entityType,
        severity,
        tier: 0,
        source: rule.id,
        confidence: ENTROPY_CONFIDENCE,
      });
    }
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
    // Exhaustive by construction: the schema enforces regex XOR entropyThreshold
    // on every rule, so a rule that is not a regex rule IS an entropy rule. No
    // else branch to drop a rule silently.
    if (rule.regex !== undefined) {
      findings.push(...runRegexRule(ir, rule, text));
    } else if (rule.entropyThreshold !== undefined) {
      findings.push(...runEntropyRule(ir, rule, segments));
    }
  }
  return findings.sort((a, b) => a.start - b.start);
}
