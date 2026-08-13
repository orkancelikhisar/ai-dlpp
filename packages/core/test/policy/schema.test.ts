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

  it("rejects a rule with both regex and entropyThreshold", () => {
    const ir = minimalIr();
    ir.rules[0]!.entropyThreshold = 4.0;
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/cannot have both regex and entropyThreshold/i);
  });

  it("rejects validator on an entropy rule", () => {
    const ir = minimalIr();
    ir.rules[2]!.validator = "pan-structure";
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/validator is only valid on regex rules/i);
  });

  it("rejects contextBoost on an entropy rule", () => {
    const ir = minimalIr();
    ir.rules[2]!.contextBoost = ["secret"];
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/contextBoost is only valid on regex rules/i);
  });

  it("rejects minLength on a regex rule", () => {
    const ir = minimalIr();
    ir.rules[0]!.minLength = 20;
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/minLength is only valid on entropy rules/i);
  });

  it("rejects a duplicate entityType id", () => {
    const ir = minimalIr();
    ir.entityTypes.push({
      id: "in-pan", tier: 0, nlDefinition: "dupe", examples: [], counterExamples: [], severity: "high",
    });
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/duplicate entityType id.*in-pan/i);
  });

  it("rejects a duplicate rule id", () => {
    const ir = minimalIr();
    ir.rules.push({ id: "pan-rule", entityType: "aws-key", regex: "\\bAKIA[0-9A-Z]{16}\\b" });
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/duplicate rule id.*pan-rule/i);
  });

  it("rejects a duplicate semanticPredicate id", () => {
    const ir = minimalIr();
    ir.semanticPredicates.push(
      { id: "dupe-pred", nlPredicate: "a", scope: "segment" },
      { id: "dupe-pred", nlPredicate: "b", scope: "message" },
    );
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/duplicate semanticPredicate id.*dupe-pred/i);
  });

  it("rejects a providerOverrides key that is not a declared entityType", () => {
    const ir = minimalIr();
    ir.actions.providerOverrides!["deepseek"]!["client-nmae"] = "redact";
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/providerOverrides.*unknown entityType.*client-nmae/i);
  });

  it("rejects an actions.default key that is not a declared entityType", () => {
    const ir = minimalIr();
    ir.actions.default["no-such-type"] = "block";
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/actions\.default.*unknown entityType.*no-such-type/i);
  });

  it("rejects a provenance key that references nothing declared", () => {
    const ir = minimalIr();
    ir.provenance["ghost-rule"] = { clause: "§9.9", quote: "nothing declares this." };
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/provenance key.*ghost-rule/i);
  });

  it("accepts provenance keyed by an entityType or semanticPredicate id", () => {
    const ir = minimalIr();
    ir.semanticPredicates.push({ id: "pred-1", nlPredicate: "mentions a client", scope: "segment" });
    ir.provenance["client-name"] = { clause: "§4.1", quote: "Client names are confidential." };
    ir.provenance["pred-1"] = { clause: "§4.2", quote: "Client mentions are confidential." };
    expect(() => PolicyIrSchema.parse(ir)).not.toThrow();
  });

  it("rejects an unknown top-level key", () => {
    const ir = minimalIr();
    (ir as unknown as Record<string, unknown>)["rogueKey"] = "tampered";
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/unrecognized key/i);
  });
});
