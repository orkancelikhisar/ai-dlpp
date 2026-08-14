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

  // The right boundary alone is half a fix. `id-number` surrogates outside the
  // PAN shape scramble to all-digit strings, so a surrogate is routinely a bare
  // number that can sit inside any longer figure in the response.
  it("does not match a numeric surrogate inside a longer digit run", () => {
    const numeric = new Map([["8842", "1234"]]);
    // Unboundaried on the left, "8842" matches the tail of "1998842" and splices
    // the REAL id's digits into an unrelated figure: "invoice 1991234 total".
    // Silent, and it corrupts a number the user will read as authoritative.
    expect(rehydrateText("invoice 1998842 total", numeric)).toBe("invoice 1998842 total");
  });

  it("still rehydrates a numeric surrogate standing on its own", () => {
    // Control for the pin above: the left boundary must not cost the normal case.
    expect(rehydrateText("ref 8842 closed", new Map([["8842", "1234"]]))).toBe("ref 1234 closed");
  });
});

/**
 * Two mentions the boundaries knowingly decline to rehydrate. Both leave a
 * surrogate in front of the user rather than risk a wrong value; both are here
 * so a future reader sees a decision instead of a bug.
 */
describe("declined rehydrations (documented trades)", () => {
  const map = new Map([["Vantor", "Globex"]]);

  it("leaves a digit-glued mention alone", () => {
    // "Vantor2024" matches nothing: the right boundary rejects it and there is
    // no shorter alternative. The fake name reaches the user, which is the
    // deliberate side to fail on -- the alternative is matching at a digit seam,
    // exactly what produces misattribution and mangled numbers.
    expect(rehydrateText("Vantor2024 filing", map)).toBe("Vantor2024 filing");
  });

  it("is case-sensitive", () => {
    // A model that lowercases the surrogate ("the vantor deal") loses the
    // rehydration. Case-insensitive matching is not the fix: it would collapse
    // distinct surrogates and rehydrate ordinary words that happen to collide.
    expect(rehydrateText("the vantor deal", map)).toBe("the vantor deal");
  });

  it("still rehydrates a letter-glued mention", () => {
    // The counterpart to the pluralization pin, on the left edge: letters are
    // deliberately unboundaried in both directions, so a surrogate fused to a
    // word still comes back. Desirable, not tolerated.
    expect(rehydrateText("ClientVantor ticket", map)).toBe("ClientGlobex ticket");
  });
});
