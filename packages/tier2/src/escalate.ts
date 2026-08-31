/**
 * Escalation -- spec 4.1: which segments are worth spending a tier-2 judge on.
 *
 * RE-EXPORTED, not defined here. Plan 5 (Task 7) names this file as the home
 * for the policy and it cannot be, for one mechanical reason: `detect` in
 * `@sih/core` is the caller, and core's package.json declares no dependency on
 * `@sih/tier2` while `@sih/tier2` depends on core. Importing the policy from
 * the tier it gates is a cycle. The definition therefore lives in
 * `packages/core/src/detect/escalate.ts`, which is where the reasoning for
 * every rule it applies lives too.
 *
 * This file exists so a caller that assembles a tier-2 call OUTSIDE `detect` --
 * the eval harness driving one arm, an extension building its own loop -- can
 * reach the escalation policy from the same package as the judge it feeds,
 * instead of learning that this one piece of tier-2's contract lives elsewhere.
 *
 * A second implementation in this package was the alternative, and it is the
 * same wrong fix a duplicated `SHADOW_PREFIX` would have been: two copies are
 * free to drift, and when they do, the bake-off measures one escalation policy
 * while `detect` runs another and nothing fails. `test/escalate.test.ts` asserts
 * these are the SAME function objects, so a copy cannot be introduced quietly.
 */
export {
  UNCERTAIN_BELOW,
  selectSegments,
  uncertainSegmentStarts,
  type EscalationInput,
} from "@sih/core";
