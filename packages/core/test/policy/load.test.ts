import { describe, expect, it } from "vitest";
import { loadPolicyIr, PolicyLoadError, PolicyVersionError } from "../../src/policy/load.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

describe("loadPolicyIr", () => {
  it("loads a valid IR from JSON text", () => {
    const ir = loadPolicyIr(JSON.stringify(minimalIr()));
    expect(ir.entityTypes).toHaveLength(4);
  });

  it("refuses an unknown irVersion (spec §7: never best-effort)", () => {
    const raw = { ...minimalIr(), irVersion: "2" };
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(PolicyVersionError);
  });

  it("throws PolicyLoadError on malformed JSON", () => {
    expect(() => loadPolicyIr("{nope")).toThrow(PolicyLoadError);
  });

  it("throws PolicyLoadError on an invalid regex", () => {
    const raw = minimalIr();
    raw.rules[0]!.regex = "([unclosed";
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/invalid regex/i);
  });

  it("throws PolicyLoadError when a named validator does not exist", () => {
    const raw = minimalIr();
    raw.rules[0]!.validator = "not-a-real-validator";
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/unknown validator/i);
  });
});
