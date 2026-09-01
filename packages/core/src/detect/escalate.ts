import type { Segment } from "../segment/segment.js";
import type { Finding } from "./types.js";

/**
 * Escalation -- spec 4.1: which segments are worth a tier-2 judge.
 *
 * "tier2 iff (uncertain OR semanticPredicates present) AND TierConfig.tier2".
 * `TierConfig.tier2` is `detect`'s to check; this module answers the other two
 * and turns them into a segment list.
 *
 * ## Why this is in @sih/core and not in @sih/tier2
 *
 * Plan 5 names `packages/tier2/src/escalate.ts` as the home, and it cannot be:
 * `detect` is the caller, `@sih/core` is what declares it, and core's
 * package.json has no dependency on `@sih/tier2` while `@sih/tier2` depends on
 * core. Importing the policy from the tier it gates would be a cycle.
 *
 * `shadowIdFor` took the same shape for a DIFFERENT reason (the compiler that
 * needed it is Node-only, not circular), and the shape is what carries over:
 * define once in core, re-export from the other package. So
 * `packages/tier2/src/escalate.ts` exists, as a re-export of this module rather
 * than a second copy -- two copies are free to drift, and when they do the
 * bake-off measures one escalation policy while `detect` runs another.
 *
 * The alternative was leaving the filter inside `WebLlmJudge`. It is the wrong
 * place for the reason the orchestrator already gives for tier 1's identical
 * filter: engines receive segments and answer about them, so which segments
 * deserve a model is a policy-shaped decision, and holding it here is what lets
 * it change for every tier-2 JUDGE at once rather than once per judge. Two
 * judge arms that each filtered their own input would make the bake-off measure
 * a different escalation policy per arm while attributing the difference to
 * their models.
 *
 * ## What this module does NOT govern
 *
 * The Approach-B arm, stated because an earlier version of the paragraph above
 * claimed it did and that is false: `createBaselineB` returns a `Detector`, not
 * a `SemanticJudge`. It never reaches `detect`'s `engines.tier2`, is never
 * handed an escalated segment list, and imports nothing from this file --
 * VERIFIED by grep over `packages/tier2/src/baselineB.ts`, which names neither
 * `selectSegments` nor `uncertainSegmentStarts`. B makes one call per MESSAGE
 * by design, which its own docblock lists as intrinsic to "simply prompting",
 * so B is shown the whole message where the compiled arm is shown a selection
 * of it. That difference belongs to the method under test; it is not something
 * this module can equalise, and a write-up comparing the two arms has to say so
 * rather than treating both as governed by one escalation policy.
 *
 * ## What escalation costs and what it buys
 *
 * NOT measured by me, and quoted rather than claimed: Plan 5's own probe results
 * record 4.6 s per tier-2-shaped call on the cheapest model it pins, and no
 * model reaching p95 <= 3 s. `ir.latencyBudgetMs` is a required positive
 * integer with no schema default, and the value the test fixtures carry is
 * 5000. So on those numbers ONE unnecessary segment spends 92% of the whole
 * message budget and a second one blows it, which is why the cheapest decision
 * available -- do not call at all -- is worth making carefully.
 */

/**
 * A prior finding at or above this confidence is treated as settled; below it,
 * its segment is escalated.
 *
 * 0.8 is chosen against the values this pipeline actually emits. The two TIER-0
 * clusters below are read off a real `runTier0` run in
 * `test/detect/escalate.test.ts` rather than quoted here. The tier-1 bullet is
 * NOT, and cannot be: `runTier0` cannot emit a tier-1 confidence, no tier-1
 * model runs in core's suite, and core cannot import `@sih/tier1` at all
 * (tier1 depends on core, so the edge only goes one way). Its provenance is
 * given on its own bullet.
 *
 * - tier 0 regex rules sit at 0.9, rising to 0.95 with a context boost and
 *   capped at 0.99 (`tier0.ts`). They have cleared a structural validator where
 *   one exists, so a judge has nothing to add.
 * - tier 0 entropy rules are FIXED at 0.7, and `tier0.ts` says why in as many
 *   words: entropy knows a string looks random, never that it is a secret, and
 *   "raising the confidence is tier 1-2's job (or the user's)". That is exactly
 *   this branch's population.
 * - tier 1 passes the model's own sigmoid score through (`tier1/src/tagger.ts`
 *   sets `confidence: span.score`; `decode.ts` sigmoids the logit and skips a
 *   span scoring below the threshold), filtered at `Tier1Config.threshold`,
 *   whose shipped default is 0.5 -- so its live range is [0.5, 1) and much of
 *   it falls on the uncertain side of this constant. READ OFF TIER 1'S SOURCE,
 *   not off any run this package can perform. The coupling is asserted from the
 *   side that CAN run it: `packages/tier1/test/tagger.test.ts` drives a real
 *   `GlinerSpanTagger` at the default threshold, emits a sub-0.8 finding, and
 *   escalates a segment through this function -- so a default raised above 0.8,
 *   which would put tier 1's whole live range on the certain side and switch
 *   tier-1-driven escalation off, fails there rather than nowhere.
 *
 * 0.8 is the midpoint of tier 0's two clusters, which is the point: neither of
 * them is near the boundary, so a later tweak to `CONTEXT_BONUS` or
 * `ENTROPY_CONFIDENCE` does not move a whole class of findings across the line.
 * A threshold of exactly 0.9 would -- every unboosted regex finding sits on it,
 * and `<` versus `<=` would flip all of them at once.
 *
 * NOT calibrated against a corpus. No run of this repository has measured how
 * often a judge changes the outcome for a finding at any given confidence, so
 * this is a defensible starting point and not a tuned one. `uncertainBelow` on
 * `TierConfig` exists so the bake-off can vary it per arm rather than editing
 * this constant.
 */
export const UNCERTAIN_BELOW = 0.8;

export interface EscalationInput {
  /** `ir.semanticPredicates.length > 0`. */
  readonly hasPredicates: boolean;
  /**
   * The `start` offset of each segment an earlier tier left uncertain about --
   * segment starts, NOT finding starts. `uncertainSegmentStarts` is what
   * produces these; passing anything else throws rather than selecting nothing.
   */
  readonly uncertain: readonly number[];
}

/**
 * The segments worth spending a judge on, in input order and without repeats.
 *
 * The two branches are a UNION, and each excludes different things:
 *
 * - **Predicates present** selects every non-code segment. A semantic predicate
 *   exists because no pattern could express it, so natural language is its
 *   ground. Code is excluded on the argument the orchestrator already makes for
 *   tier 1's identical filter -- a fenced block is mostly identifiers and
 *   syntax, which a model reads as a wall of false positives while burning
 *   seconds per segment -- and tier 0's entropy rules are what cover code.
 *
 *   kv IS included, and spec 4.1's wording is narrower: it says predicates
 *   "always require tier 2 on prose segments". Included anyway, deliberately.
 *   A kv line is natural-language content as often as it is machine data
 *   (`client: Northwind Traders`, `project_codename: Titan`), tier 1 is already
 *   handed prose AND kv by the same orchestrator, and excluding it here would
 *   leave a message that is entirely a config paste with no tier-2 coverage at
 *   all while tier 1 read every line of it. Under-inclusion is the expensive
 *   direction in a DLP, which is the argument `segment.ts` makes for its own
 *   generous kv matcher.
 *
 * - **Uncertain** selects the named segments whatever their kind, code
 *   included. Spec 4.1 attaches the word "prose" to the predicate half only and
 *   leaves this half unqualified, and the code exclusion's rationale is about
 *   scanning EVERY code segment rather than the one a lower tier hedged on.
 *   Excluding code here would also make the branch nearly unreachable: tier 0's
 *   entropy findings are the archetypal uncertain finding and they are emitted
 *   for code and kv segments only.
 *
 * What the uncertainty branch does NOT buy today, stated because the spec's
 * wording invites the opposite reading: tier 2 cannot resolve a tier-0/1
 * finding's uncertainty. `WebLlmJudge` emits findings whose entityType is a
 * `pred:` shadow (`judge.ts`), and no channel exists by which one tier revises
 * another tier's confidence -- so escalating a segment because tier 0 hedged
 * about a secret gets that segment's SEGMENT-SCOPED predicates judged, not that
 * secret re-scored. And when the policy declares no predicates at all,
 * `WebLlmJudge` returns an empty verdict without an engine call, so an
 * uncertainty-only escalation currently costs nothing and yields nothing.
 *
 * The same is true when the policy declares predicates but none of them
 * `scope: "segment"`. `WebLlmJudge` partitions by scope and never enters its
 * segment loop without a segment-scoped clause to carry, so on such a policy --
 * `policies/compiled/p-fin.ir.json` is one -- this function still selects every
 * non-code segment and none of them costs a call. The selection is then a list
 * the judge is handed and ignores, which is harmless here and is NOT harmless
 * in `apps/eval`: its per-arm `judgedUnitChars` and `judgedUnitsPerItem` are
 * built from this selection and would describe passages no model was shown.
 * Recorded in the README's carried risks. The branch is wired to
 * a real source anyway (see `uncertainSegmentStarts`) because the alternative
 * is a parameter nobody populates, which reads as a working feature in every
 * test and never fires.
 */
export function selectSegments(segments: readonly Segment[], input: EscalationInput): Segment[] {
  // Built once rather than per uncertain offset: both the membership check
  // below and the caller-bug guard read it, and `uncertain` is a list, not a
  // set, because it crosses a package boundary as data.
  const bySegment = new Set(segments.map((s) => s.start));
  const uncertain = new Set(input.uncertain);
  for (const offset of uncertain) {
    if (!bySegment.has(offset)) {
      // Loud, because the failure it catches is silent. `uncertain` holds
      // segment starts and the natural mistake is handing over finding starts;
      // that selects nothing and is indistinguishable from a message where
      // nothing was uncertain. Offsets only -- never a segment's text, which is
      // the string this system exists to keep out of logs.
      throw new Error(
        `escalation was given uncertain offset ${offset}, which starts no segment ` +
          `(uncertain holds SEGMENT starts, not finding starts; this message has ` +
          `${segments.length} segment(s))`,
      );
    }
  }
  return segments.filter(
    (segment) => uncertain.has(segment.start) || (input.hasPredicates && segment.kind !== "code"),
  );
}

/**
 * The segments an earlier tier left uncertain about, as segment starts in
 * segment order.
 *
 * This is what makes `EscalationInput.uncertain` a populated seam rather than a
 * parameter with no producer. It is called with tier 0's and tier 1's findings
 * for the message -- the two tiers spec 4.1 names -- and reads the one thing
 * both of them really emit: `Finding.confidence`.
 *
 * A finding marks EVERY segment it overlaps, not the one its `start` lands in.
 * Tier 0's regex rules scan the whole message rather than a segment (`tier0.ts`
 * explains why) so a match can straddle a boundary, and attributing a straddler
 * to one side would leave the other unjudged. Overlap is half-open on both
 * sides, so a finding ending exactly where a segment begins does not mark it.
 *
 * `below` is validated rather than trusted: it reaches a `<` comparison, and
 * `confidence < NaN` is false for every finding -- so an unvalidated NaN turns
 * the whole uncertainty branch off and every message reports as having nothing
 * uncertain, which is a wrong answer that looks like a right one. Same reasoning
 * as `manifest.ts` validating the numbers that reach `setTimeout`. 0 and 1 are
 * both legal: 0 disables the branch on purpose (an arm that escalates on
 * predicates alone) and 1 escalates everything short of total certainty.
 */
export function uncertainSegmentStarts(
  segments: readonly Segment[],
  priorFindings: readonly Finding[],
  below: number = UNCERTAIN_BELOW,
): number[] {
  if (!(Number.isFinite(below) && below >= 0 && below <= 1)) {
    throw new Error(
      `escalation uncertainty threshold must be a finite number in [0, 1], got ${String(below)}`,
    );
  }
  const uncertain = priorFindings.filter((f) => f.confidence < below);
  if (uncertain.length === 0) return [];
  return segments
    .filter((segment) => uncertain.some((f) => f.start < segment.end && f.end > segment.start))
    .map((segment) => segment.start);
}
