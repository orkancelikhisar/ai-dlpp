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
    for (const name of ["luhn", "verhoeff", "pan-structure"]) {
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
