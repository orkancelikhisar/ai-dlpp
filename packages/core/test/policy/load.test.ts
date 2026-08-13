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

  // A version error means "recompile against a newer runtime"; input that is not a
  // policy IR at all carries no such advice, so it must not be classified as one.
  it.each([
    ["null", "null"],
    ["a number", "42"],
    ["an array", "[]"],
    ["an object with no irVersion", "{}"],
  ])("throws plain PolicyLoadError (not PolicyVersionError) for %s", (_label, json) => {
    expect(() => loadPolicyIr(json)).toThrow(PolicyLoadError);
    expect(() => loadPolicyIr(json)).not.toThrow(PolicyVersionError);
  });

  // Pins gate-before-schema ordering: this object would also fail schema validation,
  // but the version refusal must win so the caller learns the actionable cause.
  it("reports a present-but-unsupported version as PolicyVersionError before schema validation", () => {
    expect(() => loadPolicyIr(JSON.stringify({ irVersion: "2" }))).toThrow(PolicyVersionError);
  });

  it("wraps zod failures in PolicyLoadError rather than letting a ZodError escape", () => {
    const raw: Record<string, unknown> = { ...minimalIr() };
    delete raw["policyHash"];
    let thrown: unknown;
    try {
      loadPolicyIr(JSON.stringify(raw));
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(PolicyLoadError);
    expect((thrown as Error).message).toMatch(/failed validation/i);
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
