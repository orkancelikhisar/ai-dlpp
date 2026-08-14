import { describe, expect, it } from "vitest";
import { generateSurrogate } from "../../src/pseudo/generators.js";
import { getValidator } from "../../src/detect/validators.js";

const KEY = "conv1\u0000client-name\u0000Globex";

describe("generateSurrogate", () => {
  it("is deterministic per seed key", () => {
    expect(generateSurrogate("org-name", "Globex", KEY)).toBe(generateSurrogate("org-name", "Globex", KEY));
  });

  it("person-name yields a first/last pair, never the real value", () => {
    const s = generateSurrogate("person-name", "Priya Sharma", "k1");
    expect(s).toMatch(/^[A-Z][a-z]+ [A-Z][a-z]+$/);
    expect(s.toLowerCase()).not.toBe("priya sharma");
  });

  it("org-name re-rolls when the pick collides with the real value", () => {
    // Deterministic construction: whatever org key "kX" yields, minting THAT org
    // under the same key must yield something else (salted retry).
    const first = generateSurrogate("org-name", "zzz-no-collision", "kX");
    const rerolled = generateSurrogate("org-name", first, "kX");
    expect(rerolled).not.toBe(first);
  });

  it("id-number preserves PAN shape and holder type, passes the validator, differs from real", () => {
    const s = generateSurrogate("id-number", "ABCPD1234E", "k2");
    expect(s).toMatch(/^[A-Z]{5}[0-9]{4}[A-Z]$/);
    expect(s[3]).toBe("P");
    expect(getValidator("pan-structure")(s)).toBe(true);
    expect(s).not.toBe("ABCPD1234E");
  });

  it("id-number falls back to class-preserving scramble for non-PAN shapes", () => {
    const s = generateSurrogate("id-number", "AC-42-9917", "k3");
    expect(s).toMatch(/^[A-Z]{2}-[0-9]{2}-[0-9]{4}$/);
    expect(s).not.toBe("AC-42-9917");
  });

  it("opaque preserves length and character classes", () => {
    const s = generateSurrogate("opaque", "x9K2mQ8vL4jR7nT3wY6z", "k4");
    expect(s).toHaveLength(20);
    expect(s).toMatch(/^[a-z][0-9][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z]$/);
    expect(s).not.toBe("x9K2mQ8vL4jR7nT3wY6z");
  });

  it("different seed keys give different surrogates (spot check)", () => {
    expect(generateSurrogate("opaque", "abcdefgh12345678", "kA")).not.toBe(
      generateSurrogate("opaque", "abcdefgh12345678", "kB"),
    );
  });
});
