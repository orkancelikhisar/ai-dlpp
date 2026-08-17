import { describe, expect, it } from "vitest";
import { checkRegexSafety, validateIdHygiene, validateRules } from "../../src/stages/validate.js";
import { mintShadowEntityTypes } from "../../src/stages/predicates.js";

describe("checkRegexSafety", () => {
  it("accepts a well-behaved pattern", async () => {
    await expect(checkRegexSafety("\\b[A-Z]{5}[0-9]{4}[A-Z]\\b")).resolves.toBeUndefined();
  });

  it("rejects a catastrophic-backtracking pattern within the time budget", async () => {
    // (a+)+$ against a long non-matching input is the classic ReDoS shape.
    await expect(checkRegexSafety("^(a+)+$")).rejects.toThrow(/timed out|backtrack/i);
  }, 15_000);

  it("rejects a syntactically invalid pattern", async () => {
    await expect(checkRegexSafety("([unclosed")).rejects.toThrow(/invalid/i);
  });

  it("rejects a pattern that can match the empty string", async () => {
    await expect(checkRegexSafety("X*")).rejects.toThrow(/empty string/i);
  });
});

describe("validateRules", () => {
  it("rejects a rule naming a validator outside the fixed library", async () => {
    await expect(
      validateRules([{ id: "r", entityType: "e", regex: "a", validator: "invented", sourceQuote: "q" }]),
    ).rejects.toThrow(/unknown validator "invented"/i);
  });

  it("accepts every validator the core library actually exports", async () => {
    for (const v of ["luhn", "verhoeff", "pan-structure", "jwt-shape"]) {
      await expect(
        validateRules([{ id: `r-${v}`, entityType: "e", regex: "\\d{2,}", validator: v, sourceQuote: "q" }]),
      ).resolves.toBeDefined();
    }
  });
});

describe("validateIdHygiene", () => {
  it("warns when an entityType id shares a token with its own examples", () => {
    // Ids ship outbound in [REDACTED:<id>]; an id derived from the confidential
    // value defeats the redaction.
    const warnings = validateIdHygiene([
      { id: "project-titan", examples: ["Project Titan"], nlDefinition: "d", tier: 1, counterExamples: [], severity: "high", sourceQuote: "q" },
    ]);
    expect(warnings[0]).toMatch(/project-titan/);
  });

  it("stays silent on a generic id", () => {
    expect(
      validateIdHygiene([
        { id: "internal-codename", examples: ["Project Titan"], nlDefinition: "d", tier: 1, counterExamples: [], severity: "high", sourceQuote: "q" },
      ]),
    ).toEqual([]);
  });

  it("warns on a shadow id sharing a token with its predicate definition", () => {
    // Shadows are minted with `examples: []` by construction, so the examples-based
    // heuristic is VACUOUS on them however it is wired -- yet a `pred:` id ships
    // outbound inside [REDACTED:<id>] exactly like an authored one, and its
    // predicate-id half comes from the extraction model. For shadows the id is
    // therefore scored against `nlDefinition` (the predicate text) instead.
    const { entityTypes } = mintShadowEntityTypes([
      {
        id: "titan-disclosure",
        nlPredicate: "Discusses Project Titan before its public announcement.",
        scope: "segment",
        severity: "high",
      },
      {
        id: "market-sensitive",
        nlPredicate: "Discusses revenue figures that have not yet been published.",
        scope: "segment",
        severity: "high",
      },
    ]);
    const warnings = validateIdHygiene(entityTypes);
    // Exactly one: the generic shadow must stay silent, or the check is noise.
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/pred:titan-disclosure/);
    expect(warnings[0]).toMatch(/titan/);
  });
});
