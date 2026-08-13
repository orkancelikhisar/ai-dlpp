import { describe, expect, it } from "vitest";
import { shannonEntropy } from "../../src/detect/validators.js";

// jwt-shape lives in validators.test.ts with the rest of the registry; this
// file covers only the entropy scorer.
describe("shannonEntropy", () => {
  it("is 0 for a single repeated character", () => {
    expect(shannonEntropy("aaaaaaaa")).toBe(0);
  });
  it("is 1 bit/char for a two-symbol alternation", () => {
    expect(shannonEntropy("abababab")).toBeCloseTo(1.0, 5);
  });
  it("is high for a random-looking secret", () => {
    expect(shannonEntropy("x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ")).toBeCloseTo(5.0, 5);
    expect(shannonEntropy("x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ")).toBeGreaterThan(4.0);
  });
  it("is exactly log2(n) for n distinct characters", () => {
    expect(shannonEntropy("abcdefgh")).toBe(3); // 8 distinct -> 3 bits/char
    expect(shannonEntropy("abcd")).toBe(2);
  });

  // Counting is per code point (`for...of`), so the denominator must be the
  // code-point count too. Dividing by s.length (UTF-16 code units) makes the
  // probabilities sum to 0.5 for astral input, understating entropy.
  it("counts astral characters as one symbol, not two surrogate halves", () => {
    expect(shannonEntropy("\u{1F600}\u{1F600}")).toBe(0); // one repeated symbol
    expect(shannonEntropy("\u{1F600}\u{1F601}")).toBe(1); // two distinct symbols
  });
});
