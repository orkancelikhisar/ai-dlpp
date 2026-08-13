import { describe, expect, it } from "vitest";
import { getValidator, hasValidator } from "../../src/detect/validators.js";

describe("luhn", () => {
  const luhn = getValidator("luhn");
  it("accepts known-valid numbers", () => {
    expect(luhn("79927398713")).toBe(true);
    expect(luhn("4539148803436467")).toBe(true); // Visa test number
    expect(luhn("4539 1488 0343 6467")).toBe(true); // with spaces
  });
  it("rejects invalid numbers and junk", () => {
    expect(luhn("79927398714")).toBe(false);
    expect(luhn("")).toBe(false);
    expect(luhn("abcd")).toBe(false);
  });

  // Pinned decisions, not incidental behavior -- change these only deliberately.
  it("accepts all-zeros, which IS Luhn-valid", () => {
    // Filtering placeholders like 0000... is the regex rule's job, not the
    // checksum's; a validator that lied about the math would hide real numbers.
    expect(luhn("00")).toBe(true);
  });
  it("rejects a single digit via the min-length gate", () => {
    expect(luhn("0")).toBe(false); // >=2 digits required, so "0" is not a "number"
  });
});

describe("verhoeff", () => {
  const verhoeff = getValidator("verhoeff");
  it("accepts the canonical example 2363 (check digit of 236 is 3)", () => {
    expect(verhoeff("2363")).toBe(true);
  });
  it("exactly one check digit makes any base valid", () => {
    const base = "23629958402";
    const valid = [..."0123456789"].filter((d) => verhoeff(base + d));
    expect(valid).toHaveLength(1);
  });
  it("tolerates the separators real numbers are written with", () => {
    expect(verhoeff("2363 ")).toBe(true);
    expect(verhoeff("23-63")).toBe(true);
  });
  it("rejects empty input and non-digits", () => {
    expect(verhoeff("")).toBe(false);
    expect(verhoeff("abcd")).toBe(false);
  });
  // Verhoeff catches adjacent transpositions, so this only demonstrates
  // anything because the base's first two digits ("2","3") differ -- swapping
  // equal digits is a no-op and would leave the number valid.
  it("a single transposed digit invalidates", () => {
    const base = "23629958402";
    const check = [..."0123456789"].find((d) => verhoeff(base + d))!;
    const full = base + check;
    const swapped = full[1]! + full[0]! + full.slice(2);
    expect(verhoeff(swapped)).toBe(false);
  });
});

describe("pan-structure", () => {
  const pan = getValidator("pan-structure");
  it("accepts a well-formed PAN with a valid 4th character", () => {
    expect(pan("ABCPD1234E")).toBe(true);
  });
  it("rejects an invalid holder-type character", () => {
    expect(pan("ABCXD1234E")).toBe(false); // X is not a holder type
  });
  it("rejects wrong shapes", () => {
    expect(pan("ABCP1234E")).toBe(false);
    expect(pan("abcpd1234e")).toBe(false);
  });
});

describe("registry", () => {
  it("resolves every registered validator", () => {
    for (const name of ["luhn", "verhoeff", "pan-structure", "jwt-shape"]) {
      expect(hasValidator(name)).toBe(true);
      expect(typeof getValidator(name)).toBe("function");
    }
  });

  it("throws on an unregistered name", () => {
    expect(hasValidator("nope")).toBe(false);
    expect(() => getValidator("nope")).toThrow(/unknown validator "nope"/);
  });

  // Inherited Object.prototype members must not resolve as validators. A bare
  // REGISTRY[name] lookup walks the prototype chain and hands back e.g.
  // Object.prototype.toString, which returns a truthy "[object ...]" string for
  // every candidate -- a validator that fails OPEN, approving every match.
  // hasValidator and getValidator must agree on exactly the same key set.
  it("never resolves prototype-chain members", () => {
    for (const name of ["toString", "valueOf", "constructor", "hasOwnProperty", "__proto__"]) {
      expect(hasValidator(name)).toBe(false);
      expect(() => getValidator(name)).toThrow(/unknown validator/);
    }
  });
});
