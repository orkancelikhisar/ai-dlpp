import { describe, expect, it } from "vitest";
import { maxSurrogateLength, rehydrateText, surrogatePattern } from "../../src/pseudo/rehydrate.js";

describe("rehydrateText", () => {
  it("replaces every known surrogate", () => {
    const map = new Map([["Vantor", "Globex"], ["Anjali Verma", "Priya Sharma"]]);
    expect(rehydrateText("Vantor hired Anjali Verma. Vantor won.", map)).toBe(
      "Globex hired Priya Sharma. Globex won.",
    );
  });

  it("prefers the longest surrogate on containment", () => {
    const map = new Map([["AB", "x"], ["ABC", "y"]]);
    expect(rehydrateText("ABC AB", map)).toBe("y x");
  });

  it("escapes regex metacharacters in surrogates", () => {
    const map = new Map([["Vellum & Gray (Ltd)", "RealCo"]]);
    expect(rehydrateText("per Vellum & Gray (Ltd) filing", map)).toBe("per RealCo filing");
  });

  it("is the identity on an empty map", () => {
    expect(rehydrateText("nothing here", new Map())).toBe("nothing here");
  });
});

describe("helpers", () => {
  it("maxSurrogateLength over the map's keys", () => {
    expect(maxSurrogateLength(new Map([["ab", "1"], ["abcd", "2"]]))).toBe(4);
    expect(maxSurrogateLength(new Map())).toBe(0);
  });

  it("surrogatePattern is undefined for an empty map", () => {
    expect(surrogatePattern(new Map())).toBeUndefined();
  });
});

/**
 * The vault's pool-exhaustion ladder mints digit-suffixed surrogates ("Vantor",
 * then "Vantor 2") as a matter of routine, so digit-adjacent collisions are not
 * exotic: without a boundary, "Vantor 2024" matches the LONGER key "Vantor 2"
 * and rehydrates to the wrong entity plus a mangled year. The lookahead is
 * digits-only on purpose -- see the pluralization pin below.
 */
describe("digit-boundary on surrogate matches", () => {
  const suffixed = new Map([["Vantor", "Aurora Labs"], ["Vantor 2", "Borealis Ltd"]]);

  it("does not match a suffixed surrogate inside a longer number", () => {
    // "Vantor 2" fails at "2|024"; the alternation retries "Vantor", whose next
    // character is a space. Base name wins, the year survives intact.
    expect(rehydrateText("Vantor 2024 report", suffixed)).toBe("Aurora Labs 2024 report");
  });

  it("prefers the base name when the digit run continues", () => {
    // "Vantor 22" is genuinely ambiguous ("Vantor 2" + "2" vs "Vantor" + " 22").
    // The boundary resolves it toward the base name; documented, not accidental.
    expect(rehydrateText("Vantor 22", suffixed)).toBe("Aurora Labs 22");
  });

  it("still rehydrates the stem of a pluralized surrogate", () => {
    // The accepted trade: a letter boundary (?!\w) would leave "Vantors"
    // untouched -- the real value never comes back for that mention. Models
    // pluralize surrogates, so erring toward stem replacement keeps the
    // rehydration and costs one trailing letter.
    expect(rehydrateText("two Vantors merged", new Map([["Vantor", "Globex"]]))).toBe(
      "two Globexs merged",
    );
  });

  it("keeps matching a surrogate that itself ends in a digit", () => {
    // The lookahead must not break surrogates whose own last character is a
    // digit when the text does not continue the run.
    expect(rehydrateText("filed by Vantor 2 today", suffixed)).toBe("filed by Borealis Ltd today");
  });
});
