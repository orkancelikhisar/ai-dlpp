import { describe, expect, it } from "vitest";
import { PolicyIrSchema } from "../../src/policy/schema.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

describe("PolicyIrSchema", () => {
  it("accepts a valid IR", () => {
    expect(() => PolicyIrSchema.parse(minimalIr())).not.toThrow();
  });

  it("rejects an entityType with no action mapping", () => {
    const ir = minimalIr();
    ir.entityTypes.push({
      id: "orphan", tier: 1, nlDefinition: "x", examples: [], counterExamples: [], severity: "low",
    });
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/orphan.*no action/i);
  });

  it("rejects a rule referencing an unknown entityType", () => {
    const ir = minimalIr();
    ir.rules[0]!.entityType = "nope";
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/unknown entityType/i);
  });

  it("rejects a rule with neither regex nor entropyThreshold", () => {
    const ir = minimalIr();
    delete (ir.rules[0] as unknown as Record<string, unknown>)["regex"];
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/regex or entropyThreshold/i);
  });
});
