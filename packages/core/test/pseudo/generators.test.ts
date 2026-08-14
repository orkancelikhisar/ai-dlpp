import { describe, expect, it } from "vitest";
import { generateSurrogate } from "../../src/pseudo/generators.js";
import { getValidator } from "../../src/detect/validators.js";

const KEY = "conv1\u0000client-name\u0000Globex";

/** Case-insensitive tokens present in both strings, split on non-alphanumerics. */
function sharedTokens(a: string, b: string): string[] {
  const bt = new Set(splitTokens(b));
  return splitTokens(a).filter((t) => bt.has(t));
}

function splitTokens(s: string): string[] {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 1);
}

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

  it("person-name re-rolls when the pick shares a token with the real", () => {
    // Same deterministic trick as the org case: a real sharing no token with the
    // pools returns the UNSALTED pick for this key. Feed one of that pick's own
    // tokens back in as the real's first name -- keeping it would leak the real
    // first name verbatim even though the whole strings differ, so the retry
    // has to move off it.
    const unsalted = generateSurrogate("person-name", "zzz nomatch", "kT");
    const leakedFirst = unsalted.split(" ")[0]!;
    const real = `${leakedFirst} Sharma`;
    expect(sharedTokens(generateSurrogate("person-name", real, "kT"), real)).toEqual([]);
  });

  it("id-number substitutes holder type P when the real's 4th char is not one", () => {
    // "Z" is A-Z (so the real is PAN-shaped) but is not a holder type; copying it
    // through would emit a structurally invalid PAN its own validator rejects.
    const s = generateSurrogate("id-number", "ABCZD1234E", "k6");
    expect(s[3]).toBe("P");
    expect(getValidator("pan-structure")(s)).toBe(true);
  });

  it("throws rather than looping forever when the real has nothing to scramble", () => {
    // Every scramble candidate for these IS the real, so the salted retry could
    // never terminate -- a hang, not a wrong answer. Task 4's vault maps the
    // throw onto the entity's failMode.
    expect(() => generateSurrogate("opaque", "", "k7")).toThrow(/no scrambleable characters/);
    expect(() => generateSurrogate("opaque", "@@@", "k7")).toThrow(/no scrambleable characters/);
    expect(() => generateSurrogate("id-number", "----", "k7")).toThrow(/no scrambleable characters/);
  });

  it("exempts single-character tokens, which would otherwise be unsatisfiable", () => {
    // A class-preserving scramble of 26 single-letter tokens can only ever draw
    // letters the real already contains, so counting them as leaks would turn a
    // generatable value into a thrown error. Initials are not names.
    const alphabet = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
    const s = generateSurrogate("opaque", alphabet, "kY");
    expect(s).toMatch(/^[a-z]( [a-z]){25}$/);
    expect(s).not.toBe(alphabet);
  });

  it("still generates when the real has even one scrambleable character", () => {
    const s = generateSurrogate("opaque", "-a-", "k7");
    expect(s).toMatch(/^-[a-z]-$/);
    expect(s).not.toBe("-a-");
  });

  it("re-rolls when a candidate reuses a punctuation-joined token of the real", () => {
    // Same deterministic trick as above, but the real joins its tokens with a
    // hyphen rather than a space. Splitting on whitespace alone made
    // "Rohan-Mehta" a single token, so a candidate named Rohan sailed through
    // the leak check and handed back the real first name verbatim.
    const key = "kHyphen";
    const unsalted = generateSurrogate("person-name", "zzz nomatch", key);
    const real = `${unsalted.split(" ")[0]!}-Mehta`;
    const s = generateSurrogate("person-name", real, key);
    expect(sharedTokens(s, real)).toEqual([]);
  });

  it("honours an injected salt cap and reports exhaustion without echoing the real", () => {
    // The cap is injectable purely so this path is testable: at the 64 default,
    // constructing a real whose every candidate collides is impractical. Cap 0
    // means "no salt is allowed to succeed", so exhaustion fires immediately.
    const real = "Priyanka Deshpande";
    expect(() => generateSurrogate("person-name", real, "kCap", 0)).toThrow(/exhausted/i);
    let message = "";
    try {
      generateSurrogate("person-name", real, "kCap", 0);
    } catch (error) {
      message = (error as Error).message;
    }
    // The no-echo contract: this path can fire on a genuinely sensitive value,
    // so neither the real nor any token of it may reach a log or a UI string.
    expect(message).not.toContain(real);
    for (const token of splitTokens(real)) expect(message.toLowerCase()).not.toContain(token);
  });

  it("leaves the default salt budget unchanged when a cap is not injected", () => {
    expect(generateSurrogate("org-name", "Globex", KEY, 64)).toBe(generateSurrogate("org-name", "Globex", KEY));
  });

  it("different seed keys give different surrogates (spot check)", () => {
    expect(generateSurrogate("opaque", "abcdefgh12345678", "kA")).not.toBe(
      generateSurrogate("opaque", "abcdefgh12345678", "kB"),
    );
  });
});
