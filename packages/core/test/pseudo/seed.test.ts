import { describe, expect, it } from "vitest";
import { fnv1a64, mulberry32, seededRng } from "../../src/pseudo/seed.js";

describe("fnv1a64", () => {
  it("matches published FNV-1a 64 test vectors", () => {
    expect(fnv1a64("")).toBe(0xcbf29ce484222325n);
    expect(fnv1a64("a")).toBe(0xaf63dc4c8601ec8cn);
    expect(fnv1a64("foobar")).toBe(0x85944171f73967e8n);
  });

  it("is deterministic and input-sensitive", () => {
    expect(fnv1a64("conv1 client-name Globex")).toBe(fnv1a64("conv1 client-name Globex"));
    expect(fnv1a64("conv1")).not.toBe(fnv1a64("conv2"));
  });
});

describe("mulberry32 / seededRng", () => {
  it("same seed → same sequence; different seed → different sequence", () => {
    const a1 = mulberry32(42); const a2 = mulberry32(42); const b = mulberry32(43);
    const seq = (r: () => number) => [r(), r(), r()];
    expect(seq(a1)).toEqual(seq(a2));
    expect(seq(mulberry32(42))).not.toEqual(seq(b));
  });

  it("seededRng derives from the full 64-bit hash and stays in [0, 1)", () => {
    const r = seededRng("some-key");
    for (let i = 0; i < 100; i++) {
      const x = r();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
    expect(seededRng("k1")()).not.toBe(seededRng("k2")());
  });
});
