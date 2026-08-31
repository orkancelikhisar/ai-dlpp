import { describe, expect, it } from "vitest";
import * as core from "@sih/core";
import * as tier2 from "../src/index.js";
import { selectSegments, uncertainSegmentStarts, UNCERTAIN_BELOW } from "../src/escalate.js";
import type { Segment } from "@sih/core";

/**
 * The escalation policy is DEFINED in `@sih/core` and re-exported here, because
 * `detect` is its caller and core cannot depend on the tier it gates.
 *
 * These tests exist to make a second copy impossible to introduce quietly:
 * identity, not behaviour, is what they assert. A copy-paste into this package
 * would keep every behavioural test in `packages/core/test/detect/escalate.test.ts`
 * green while `detect` and the bake-off ran different escalation policies, and
 * nothing would fail -- the exact failure mode a duplicated `SHADOW_PREFIX`
 * would have had.
 */
describe("tier-2's escalation exports", () => {
  it("are the very functions core defines, not copies of them", () => {
    expect(selectSegments).toBe(core.selectSegments);
    expect(uncertainSegmentStarts).toBe(core.uncertainSegmentStarts);
    expect(UNCERTAIN_BELOW).toBe(core.UNCERTAIN_BELOW);
  });

  it("reach a caller through this package's entry point", () => {
    // The reason the re-export exists: a harness assembling a tier-2 call
    // outside `detect` imports the judge and the policy that feeds it from one
    // place.
    expect(tier2.selectSegments).toBe(core.selectSegments);
    expect(tier2.uncertainSegmentStarts).toBe(core.uncertainSegmentStarts);
    expect(tier2.UNCERTAIN_BELOW).toBe(core.UNCERTAIN_BELOW);
  });

  it("work through the re-export", () => {
    // A smoke call, so an export that resolved to `undefined` is caught here
    // rather than at a page load.
    const segments: Segment[] = [
      { start: 0, end: 5, kind: "prose", text: "Acme " },
      { start: 5, end: 12, kind: "code", text: "x = 1;\n" },
    ];
    expect(selectSegments(segments, { hasPredicates: true, uncertain: [] })).toEqual([segments[0]]);
  });
});
