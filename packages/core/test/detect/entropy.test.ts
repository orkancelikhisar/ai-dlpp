import { describe, expect, it } from "vitest";
import { getValidator, shannonEntropy } from "../../src/detect/validators.js";

describe("jwt-shape", () => {
  const jwt = getValidator("jwt-shape");
  const b64url = (s: string) => Buffer.from(s).toString("base64url");

  it("accepts a structurally valid JWT", () => {
    const token = `${b64url('{"alg":"HS256","typ":"JWT"}')}.${b64url('{"sub":"1"}')}.${b64url("sig")}`;
    expect(jwt(token)).toBe(true);
  });
  it("rejects three dot-separated non-JWT parts", () => {
    expect(jwt("aaa.bbb.ccc")).toBe(false); // header decodes but has no alg
    expect(jwt("not a token")).toBe(false);
  });
});

describe("shannonEntropy", () => {
  it("is 0 for a single repeated character", () => {
    expect(shannonEntropy("aaaaaaaa")).toBe(0);
  });
  it("is 1 bit/char for a two-symbol alternation", () => {
    expect(shannonEntropy("abababab")).toBeCloseTo(1.0, 5);
  });
  it("is high for a random-looking secret", () => {
    expect(shannonEntropy("x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ")).toBeGreaterThan(4.0);
  });
});
