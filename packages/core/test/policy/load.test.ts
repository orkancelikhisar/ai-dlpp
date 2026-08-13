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

  // A nullable regex (one that can match "") cannot drive a scan: /g/ exec returns an
  // empty match without advancing lastIndex. Detection has to skip those matches, so a
  // rule built on one is malformed — reject it at compile-load time, not at scan time.
  it("throws PolicyLoadError on a regex that can match the empty string", () => {
    const raw = minimalIr();
    raw.rules[0]!.regex = "X*";
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(PolicyLoadError);
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/empty string/i);
    // ...and the check does not fire on the ordinary fixture rules.
    expect(() => loadPolicyIr(JSON.stringify(minimalIr()))).not.toThrow();
  });

  // Pins the known LIMIT of the check above: `test("")` only probes offset 0, where a
  // lookbehind cannot succeed, so this regex loads and then matches empty mid-string.
  // That gap is exactly why runTier0 keeps its own zero-width guard.
  it("accepts a lookbehind regex that matches empty only mid-string", () => {
    const raw = minimalIr();
    raw.rules[0]!.regex = "(?<=:)\\w*";
    expect(() => loadPolicyIr(JSON.stringify(raw))).not.toThrow();
  });

  it("throws PolicyLoadError when a named validator does not exist", () => {
    const raw = minimalIr();
    raw.rules[0]!.validator = "not-a-real-validator";
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/unknown validator/i);
  });

  // An object literal cannot express these: `__proto__:` in a literal sets the
  // prototype. JSON.parse is the only way to get a real own "__proto__" key — which is
  // also exactly how one arrives from a hand-edited or hostile IR.
  describe("__proto__ keys", () => {
    it("rejects a __proto__ key in providerOverrides instead of silently dropping it", () => {
      const raw = minimalIr();
      raw.actions.providerOverrides = JSON.parse('{"__proto__":{"in-pan":"allow"}}');
      expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(PolicyLoadError);
      expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/forbidden key/i);
    });

    it("rejects a __proto__ key nested inside an array element", () => {
      const raw = minimalIr();
      raw.rules[0] = JSON.parse('{"id":"pan-rule","entityType":"in-pan","regex":"x","__proto__":{}}');
      expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/forbidden key/i);
    });

    it("accepts __proto__ as a string value, which is not a key", () => {
      const raw = minimalIr();
      raw.entityTypes[0]!.examples.push("__proto__");
      expect(loadPolicyIr(JSON.stringify(raw)).entityTypes).toHaveLength(4);
    });
  });
});
