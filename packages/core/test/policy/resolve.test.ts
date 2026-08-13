import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { resolveAction } from "../../src/policy/resolve.js";
import { minimalIr } from "../fixtures/minimal-ir.js";
import type { EntityType } from "../../src/policy/types.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));

/**
 * IR whose entityType ids collide with names on Object.prototype / Function.prototype.
 * Nothing forbids a policy author from naming an entity "name" or "constructor", and a
 * bare index read would answer such a lookup from the prototype chain.
 */
const hazardIr = (() => {
  const raw = minimalIr();
  const probe = (id: string): EntityType => ({
    id,
    tier: 1,
    nlDefinition: `entity named ${id}`,
    examples: [],
    counterExamples: [],
    severity: "low",
  });
  for (const id of ["toString", "constructor", "name"]) {
    raw.entityTypes.push(probe(id));
    raw.actions.default[id] = "allow";
  }
  return loadPolicyIr(JSON.stringify(raw));
})();

describe("resolveAction", () => {
  it("returns the default action when no override exists", () => {
    expect(resolveAction(ir, "client-name", "chatgpt")).toBe("pseudonymize");
  });

  it("applies a provider override", () => {
    expect(resolveAction(ir, "client-name", "deepseek")).toBe("redact");
  });

  it("does not leak overrides to other entityTypes on the same provider", () => {
    expect(resolveAction(ir, "in-pan", "deepseek")).toBe("block");
  });

  it("falls back to default for unknown providers", () => {
    expect(resolveAction(ir, "client-name", "some-new-provider")).toBe("pseudonymize");
  });

  it("resolves defaults when the IR declares no providerOverrides at all", () => {
    const raw = minimalIr();
    delete raw.actions.providerOverrides;
    const bare = loadPolicyIr(JSON.stringify(raw));
    // deepseek's client-name override is gone with the field, so the default applies.
    expect(resolveAction(bare, "client-name", "deepseek")).toBe("pseudonymize");
    expect(resolveAction(bare, "in-pan", "deepseek")).toBe("block");
  });

  it("throws on an unknown entityType (programmer error)", () => {
    expect(() => resolveAction(ir, "nope", "chatgpt")).toThrow(/unknown entityType/i);
  });

  it("covers every entityType × provider without throwing", () => {
    for (const e of ir.entityTypes) {
      for (const p of ["chatgpt", "claude", "gemini", "deepseek", "unknown"]) {
        expect(["allow", "pseudonymize", "redact", "block"]).toContain(resolveAction(ir, e.id, p));
      }
    }
  });
});

/**
 * The resolver reads two plain JSON-parsed objects by key. Both inherit from
 * Object.prototype, so every lookup here would otherwise be answered by an inherited
 * member instead of the policy — returning a function or a stray string where an Action
 * belongs, which downstream reads as "not a block".
 */
describe("resolveAction: prototype-chain hazards", () => {
  it("resolves an entityType named toString from the policy, not Function.prototype", () => {
    expect(resolveAction(hazardIr, "toString", "chatgpt")).toBe("allow");
    // deepseek HAS an override object (for client-name) but no "toString" key in it.
    expect(resolveAction(hazardIr, "toString", "deepseek")).toBe("allow");
  });

  it("resolves an entityType named constructor from the policy, not Object.prototype", () => {
    expect(resolveAction(hazardIr, "constructor", "chatgpt")).toBe("allow");
    expect(resolveAction(hazardIr, "constructor", "deepseek")).toBe("allow");
  });

  it("does not answer an override lookup with a prototype member of a prototype member", () => {
    // providerOverrides["toString"] would be Function.prototype.toString, whose ".name"
    // is the string "toString" — a non-Action that no downstream check would flag.
    expect(resolveAction(hazardIr, "name", "toString")).toBe("allow");
    expect(resolveAction(hazardIr, "constructor", "toString")).toBe("allow");
  });

  it("falls back to the default for providers named after prototype members", () => {
    for (const provider of ["toString", "constructor", "valueOf", "hasOwnProperty", "__proto__"]) {
      expect(resolveAction(ir, "client-name", provider)).toBe("pseudonymize");
      expect(resolveAction(ir, "in-pan", provider)).toBe("block");
    }
  });

  it("throws for undeclared entityTypes that name prototype members", () => {
    for (const entityType of ["toString", "constructor", "valueOf", "__proto__"]) {
      expect(() => resolveAction(ir, entityType, "chatgpt")).toThrow(/unknown entityType/i);
    }
  });

  it("covers every hazardous entityType × provider and only ever yields an Action", () => {
    for (const e of hazardIr.entityTypes) {
      for (const p of ["chatgpt", "deepseek", "toString", "constructor", "__proto__", "unknown"]) {
        expect(["allow", "pseudonymize", "redact", "block"]).toContain(resolveAction(hazardIr, e.id, p));
      }
    }
  });
});
