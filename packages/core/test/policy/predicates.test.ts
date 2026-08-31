import { describe, expect, it } from "vitest";
import * as barrel from "../../src/index.js";
import { SHADOW_PREFIX, shadowIdFor } from "../../src/policy/predicates.js";

/**
 * Core owns the shadow-naming contract, and until now had no test of its own
 * for it. Drift was caught only by two INDEPENDENT oracles living elsewhere --
 * `packages/compiler/test/stages/predicates.test.ts` and
 * `packages/tier2/test/helpers.ts`, both of which spell `pred:` out by hand --
 * so the definition could be edited here and the failure would surface in two
 * other packages, or in neither if someone updated them together.
 *
 * Every expectation below is a LITERAL. Building one from `SHADOW_PREFIX` would
 * prove only that the module agrees with itself, which is the defect this
 * project has shipped repeatedly.
 */
describe("shadow entityType naming", () => {
  it("mints `pred:<id>`, spelled out rather than derived", () => {
    expect(shadowIdFor("unreleased-financials")).toBe("pred:unreleased-financials");
    expect(SHADOW_PREFIX).toBe("pred:");
  });

  it("uses a character an authored entityType id cannot contain", () => {
    // The reason the prefix ends in a colon rather than a hyphen: the
    // extraction prompt constrains authored ids to lowercase kebab-case, so the
    // two id spaces cannot overlap even before the compiler's explicit
    // collision check. A prefix of "pred-" would be a legal authored id.
    expect(SHADOW_PREFIX).toContain(":");
    expect(SHADOW_PREFIX).not.toMatch(/^[a-z0-9-]+$/);
  });

  it("is total, and does not round-trip through itself", () => {
    // Documented rather than guarded: no caller has a reason to double-apply
    // it, and a guard here would be a third validation rule agreeing with
    // neither of the two callers'. Pinned so the behaviour is a decision rather
    // than an accident.
    expect(shadowIdFor("")).toBe("pred:");
    expect(shadowIdFor(shadowIdFor("x"))).toBe("pred:pred:x");
  });

  it("reaches the browser-side judge through core's barrel", () => {
    // `@sih/tier2` imports both symbols from "@sih/core", not from a deep path.
    // Dropping either from the barrel breaks tier 2 at build time in another
    // package; here it is one failing assertion in the package that owns them.
    expect(barrel.SHADOW_PREFIX).toBe("pred:");
    expect(barrel.shadowIdFor("client-relationship")).toBe("pred:client-relationship");
  });
});
