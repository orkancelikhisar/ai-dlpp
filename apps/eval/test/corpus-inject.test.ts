import { describe, expect, it } from "vitest";
import { applyInjections, assertInjectionInvariant, type Injection, type WrittenInjection } from "../src/corpus/inject.js";

const DIMS = { surface: "prose" } as const;

function inj(over: Partial<Injection> & Pick<Injection, "at" | "value">): Injection {
  return { prefix: "", suffix: "", type: "in-pan", family: "f", dimensions: DIMS, ...over };
}

describe("applyInjections", () => {
  it("puts the span on the value and not on the glue", () => {
    // Offsets computed by hand from the strings, not from the implementation:
    // "one two" is 7 characters, the prefix " pan " is 5, so the value starts
    // at 12 and the 10-character PAN ends at 22.
    const { text, injections } = applyInjections("one two", [
      inj({ at: 7, prefix: " pan ", value: "AAAPZ1234C", suffix: " ok" }),
    ]);
    expect(text).toBe("one two pan AAAPZ1234C ok");
    expect(injections[0]!.span).toEqual({ start: 12, end: 22, text: "AAAPZ1234C" });
    expect(text.slice(12, 22)).toBe("AAAPZ1234C");
  });

  it("rebases later spans past everything written before them", () => {
    // Two injections, given OUT of positional order. If the implementation
    // spliced in argument order the second span would be short by the length of
    // the first injection (10 characters), and the recorded offsets would still
    // look plausible.
    const carrier = "aaaa bbbb cccc";
    const { text, injections } = applyInjections(carrier, [
      inj({ at: 14, prefix: " [", value: "SECOND", suffix: "]" }),
      inj({ at: 4, prefix: " <", value: "FIRST", suffix: ">" }),
    ]);
    expect(text).toBe("aaaa <FIRST> bbbb cccc [SECOND]");
    // Returned in INPUT order: the caller's parallel arrays depend on it.
    expect(injections.map((i) => i.value)).toEqual(["SECOND", "FIRST"]);
    expect(injections[1]!.span).toEqual({ start: 6, end: 11, text: "FIRST" });
    expect(injections[0]!.span).toEqual({ start: 24, end: 30, text: "SECOND" });
    for (const i of injections) expect(text.slice(i.span.start, i.span.end)).toBe(i.value);
  });

  it("handles a multi-line value", () => {
    const pem = "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----";
    const { text, injections } = applyInjections("hello", [
      inj({ at: 5, prefix: "\n\n```\n", value: pem, suffix: "\n```\n" }),
    ]);
    expect(text.slice(injections[0]!.span.start, injections[0]!.span.end)).toBe(pem);
  });

  it("refuses a carrier that already contains the value", () => {
    // Without this the corpus would carry two occurrences and one gold span, so
    // a detector finding the other occurrence would be scored a false positive
    // for being right.
    expect(() => applyInjections("the code AAAPZ1234C is on file", [inj({ at: 29, value: "AAAPZ1234C" })])).toThrow(
      /already contains the value/,
    );
  });

  it("refuses glue that reproduces the value", () => {
    expect(() => applyInjections("hello", [inj({ at: 5, prefix: " ", value: "ZZZ", suffix: " again ZZZ" })])).toThrow(
      /occurs 2 time\(s\)/,
    );
  });

  it("allows a value injected twice, and gives it two spans", () => {
    const { text, injections } = applyInjections("aaa bbb", [
      inj({ at: 3, prefix: " ", value: "ORG", suffix: "" }),
      inj({ at: 7, prefix: " ", value: "ORG", suffix: "" }),
    ]);
    expect(text).toBe("aaa ORG bbb ORG");
    expect(injections.map((i) => i.span.start)).toEqual([4, 12]);
  });

  it("refuses an injection point outside the carrier", () => {
    expect(() => applyInjections("abc", [inj({ at: 9, value: "X" })])).toThrow(/outside the carrier/);
  });

  it("refuses an empty value", () => {
    expect(() => applyInjections("abc", [inj({ at: 1, value: "" })])).toThrow(/non-empty/);
  });
});

describe("assertInjectionInvariant", () => {
  const written = (start: number, end: number, value: string): WrittenInjection => ({
    ...inj({ at: 0, value }),
    span: { start, end, text: value },
  });

  it("catches a span whose offsets do not hold the value", () => {
    expect(() => assertInjectionInvariant("abcdefgh", [written(0, 3, "def")])).toThrow(/text.slice\(0, 3\)/);
  });

  it("catches a recorded span text that is not the injected value", () => {
    const bad: WrittenInjection = { ...inj({ at: 0, value: "def" }), span: { start: 3, end: 6, text: "xyz" } };
    expect(() => assertInjectionInvariant("abcdefgh", [bad])).toThrow(/recorded span text/);
  });

  it("catches overlapping gold spans", () => {
    expect(() => assertInjectionInvariant("abcdef", [written(0, 3, "abc"), written(2, 5, "cde")])).toThrow(/overlap/);
  });

  it("accepts adjacent, non-overlapping spans", () => {
    expect(() => assertInjectionInvariant("abcdef", [written(0, 3, "abc"), written(3, 6, "def")])).not.toThrow();
  });
});
