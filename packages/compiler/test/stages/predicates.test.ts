import { describe, expect, it } from "vitest";
import { SHADOW_PREFIX, mintShadowEntityTypes, shadowIdFor } from "../../src/stages/predicates.js";

const predicates = [
  { id: "unreleased-financials", nlPredicate: "Discusses revenue not yet public", scope: "segment" as const, severity: "high" as const, sourceQuote: "q" },
];

describe("mintShadowEntityTypes", () => {
  it("mints one tier-2 entityType per predicate", () => {
    const { entityTypes } = mintShadowEntityTypes(predicates);
    expect(entityTypes).toHaveLength(1);
    expect(entityTypes[0]!.id).toBe(shadowIdFor("unreleased-financials"));
    expect(entityTypes[0]!.tier).toBe(2);
  });

  it("carries the predicate text as the entityType definition", () => {
    const { entityTypes } = mintShadowEntityTypes(predicates);
    expect(entityTypes[0]!.nlDefinition).toContain("revenue not yet public");
  });

  it("inherits the predicate's severity", () => {
    expect(mintShadowEntityTypes(predicates).entityTypes[0]!.severity).toBe("high");
  });

  it("defaults shadow entityTypes to redact, never pseudonymize", () => {
    // A semantic finding has no stable surface form, so a surrogate could not be
    // rehydrated coherently; redaction is the only honest treatment.
    const { defaultActions } = mintShadowEntityTypes(predicates);
    expect(defaultActions[shadowIdFor("unreleased-financials")]).toBe("redact");
    expect(mintShadowEntityTypes(predicates).entityTypes[0]!.neverPseudonymize).toBe(true);
  });

  it("shadow ids are namespaced so they cannot collide with authored ids", () => {
    expect(shadowIdFor("x")).toMatch(new RegExp(`^${SHADOW_PREFIX}`));
  });

  it("rejects a predicate whose id would collide with an existing entityType", () => {
    expect(() => mintShadowEntityTypes(predicates, new Set([shadowIdFor("unreleased-financials")]))).toThrow(
      /collides/i,
    );
  });

  // Beyond the plan's six. The namespacing assertion above builds its regex from
  // SHADOW_PREFIX itself, so it is self-consistent under ANY prefix -- including
  // "", which passes all six tests while defeating the namespace outright
  // (mutation-verified). The prefix is pinned here independently of itself, and
  // the `:` specifically: the extraction prompt constrains authored ids to
  // lowercase kebab-case, so a colon is what makes the two id spaces disjoint.
  it("namespaces with a character an authored kebab-case id cannot contain", () => {
    expect(SHADOW_PREFIX).toBe("pred:");
    expect(shadowIdFor("x")).toBe("pred:x");
  });

  // Two predicates sharing an id would mint one entityType twice, which
  // PolicyIrSchema rejects in Task 8 far from its cause.
  it("rejects duplicate predicate ids within one batch", () => {
    expect(() => mintShadowEntityTypes([predicates[0]!, predicates[0]!])).toThrow(/collides/i);
  });
});
