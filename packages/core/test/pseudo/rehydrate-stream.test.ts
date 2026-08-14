import { describe, expect, it } from "vitest";
import { createRehydrateTransform, rehydrateText } from "../../src/pseudo/rehydrate.js";
import { mulberry32 } from "../../src/pseudo/seed.js";

/** Push chunks through the transform, collect the full output string. */
async function pump(chunks: string[], map: Map<string, string>): Promise<string> {
  const transform = createRehydrateTransform(map);
  const writer = transform.writable.getWriter();
  const reader = transform.readable.getReader();
  const outParts: string[] = [];
  const readAll = (async () => {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      outParts.push(value);
    }
  })();
  for (const c of chunks) await writer.write(c);
  await writer.close();
  await readAll;
  return outParts.join("");
}

const MAP = new Map([["Vantor", "Globex"], ["Anjali Verma", "Priya Sharma"]]);
const SAMPLE = "Regarding Vantor: Anjali Verma of Vantor signed. Thanks, Anjali Verma.";

describe("createRehydrateTransform", () => {
  it("matches rehydrateText for every single split point of the sample", async () => {
    const expected = rehydrateText(SAMPLE, MAP);
    for (let i = 1; i < SAMPLE.length; i++) {
      const got = await pump([SAMPLE.slice(0, i), SAMPLE.slice(i)], MAP);
      expect(got, `split at ${i}`).toBe(expected);
    }
  });

  it("handles many tiny chunks (seeded random chunking, 50 rounds)", async () => {
    const expected = rehydrateText(SAMPLE, MAP);
    const rng = mulberry32(0xc0ffee);
    for (let round = 0; round < 50; round++) {
      const chunks: string[] = [];
      let i = 0;
      while (i < SAMPLE.length) {
        const step = 1 + Math.floor(rng() * 7);
        chunks.push(SAMPLE.slice(i, i + step));
        i += step;
      }
      expect(await pump(chunks, MAP)).toBe(expected);
    }
  });

  it("passes text through untouched on an empty map", async () => {
    expect(await pump(["hello ", "world"], new Map())).toBe("hello world");
  });

  it("flushes a trailing partial that never completes", async () => {
    // "Vanto" is a prefix of "Vantor" but the stream ends — it must be emitted as-is.
    expect(await pump(["ends with Vanto"], MAP)).toBe("ends with Vanto");
  });

  it("rehydrates a surrogate that is the entire final chunk", async () => {
    expect(await pump(["deal with ", "Vantor"], MAP)).toBe("deal with Globex");
  });
});

/**
 * The three contracts below are what separate this transform from the "replace
 * the buffer, then hold back a tail" shape the plan sketched. Each pins a case
 * where a decision taken on a partial buffer is NOT yet final, and each fails
 * against that shape — see the Task 8 deviations log for the fail-first output.
 *
 * The unifying rule the implementation enforces: a match may only be replaced
 * once BOTH edges of the decision are settled — its right-boundary character
 * has actually arrived (`e < buffer.length`), and every longer alternative that
 * could start where it starts was fully visible (`s <= buffer.length - maxLen`).
 * Everything else stays in the raw held-back tail and is reconsidered verbatim.
 */
describe("chunk-boundary finality", () => {
  // The vault's pool-exhaustion ladder mints "Vantor" then "Vantor 2", so a map
  // holding a surrogate and its own prefix is the routine case, not a contrived one.
  const suffixed = new Map([["Vantor", "Aurora Labs"], ["Vantor 2", "Borealis Ltd"]]);

  it("does not commit to a prefix surrogate whose longer form is still arriving", async () => {
    // "met Vantor" alone looks like a complete "Vantor" match. Replacing it there
    // is unrecoverable: the " 2" in the next chunk can no longer form "Vantor 2",
    // and the reader gets the FIRST entity's real value for the second entity's
    // mention. Deferring costs one chunk of latency and gets it right.
    expect(await pump(["met Vantor", " 2 today"], suffixed)).toBe("met Borealis Ltd today");
  });

  it("defers a complete inner match that is not at the buffer edge either", async () => {
    // Same hazard, one character further on, and this split is the one that
    // isolates it: here "Vantor" ends BEFORE the buffer edge, so its right
    // boundary is already settled and only the "was every longer alternative
    // visible?" rule can still hold it back. The split above is caught by the
    // right-edge rule as well, so on its own it would not prove this one exists.
    expect(await pump(["met Vantor ", "2 today"], suffixed)).toBe("met Borealis Ltd today");
  });

  it("agrees with rehydrateText at every split point of a digit-dense text", async () => {
    // Both surrogates, adjacent to digits and to each other, plus an all-digit
    // surrogate (the id-number shape) so both lookarounds are load-bearing at
    // seams. Pure equivalence: whatever the whole-string pass decides, the
    // stream must decide identically no matter where the chunks land.
    const map = new Map([
      ["Vantor", "Aurora Labs"],
      ["Vantor 2", "Borealis Ltd"],
      ["8842", "1234"],
    ]);
    const text = "Vantor 2 and Vantor 2024: ref 8842, not 78842 or 88421. Vantor 2Vantor 22.";
    const expected = rehydrateText(text, map);
    for (let i = 1; i < text.length; i++) {
      expect(await pump([text.slice(0, i), text.slice(i)], map), `split at ${i}`).toBe(expected);
    }
  });
});

/**
 * The right-hand digit boundary `(?![0-9])` passes VACUOUSLY at the end of a
 * string, so a match ending exactly at the buffer edge is a decision the next
 * chunk still gets a vote on. Held back until the vetoing character is real.
 */
describe("right digit boundary across a chunk seam", () => {
  const ids = new Map([["8842", "1234"]]);

  it("lets the next chunk's leading digit veto a match at the buffer edge", async () => {
    // Whole-string semantics: "88421" is one digit run, no rehydration. A
    // transform that finalized "8842" when chunk 1 ended would emit the real
    // id's digits spliced into an unrelated number.
    expect(await pump(["total 8842", "1 units"], ids)).toBe("total 88421 units");
  });

  it("still rehydrates when the next chunk clears the boundary", async () => {
    expect(await pump(["total 8842", " units"], ids)).toBe("total 1234 units");
  });
});

/**
 * The mirror hazard, and the reason the transform keeps one character of
 * already-emitted context at the head of its buffer (the "carry"): `(?<![0-9])`
 * also passes vacuously at position 0. Once the digit that should veto a match
 * has been emitted, the tail alone can no longer see it.
 */
describe("left digit boundary across a chunk seam (carry)", () => {
  const ids = new Map([["8842", "1234"]]);

  it("vetoes a match at the head of the tail using the last emitted character", async () => {
    // The split that actually exercises the carry: chunk 1 ends with the
    // surrogate itself, so the holdback keeps "8842" and emits through the "7".
    // At the next chunk the "7" survives only as the carry.
    expect(await pump(["ref 78842", " closed"], ids)).toBe("ref 78842 closed");
  });

  it("vetoes across the seam when the digit run is split before the surrogate", async () => {
    expect(await pump(["ref 7", "8842 closed"], ids)).toBe("ref 78842 closed");
  });

  it("applies the carry in flush too, when the stream ends mid digit run", async () => {
    // Chunk 1 holds "8842" back; flush sees only that tail, and must still be
    // told that a "7" preceded it.
    expect(await pump(["ref 78842"], ids)).toBe("ref 78842");
  });

  it("still rehydrates when the preceding emitted character is not a digit", async () => {
    expect(await pump(["ref ", "8842 closed"], ids)).toBe("ref 1234 closed");
  });
});

/**
 * The named tests above pin the cases we reasoned about; this one guards the
 * cases we did not. The transform's whole contract is "same answer as
 * rehydrateText, whatever the chunking", so it can be tested as a differential
 * property against the reference implementation over adversarial inputs — maps
 * whose surrogates prefix each other, are pure digits, or contain metacharacters,
 * against random text drawn from an alphabet of their own fragments. Every
 * defect found in review reproduces here, each within a few hundred rounds.
 *
 * Text is assembled from whole TOKENS — surrogates, their prefixes, digits,
 * separators — not from single characters. That is load-bearing: drawing
 * characters uniformly makes a multi-character surrogate like "ABAB" a
 * one-in-84000 event per position, and the mutation that drops the
 * longest-alternative rule then survives the whole run. With tokens, it dies in
 * the first few hundred rounds. Verified by mutation, all three rules.
 */
describe("differential property vs rehydrateText", () => {
  const maps = [
    new Map([["A", "1"], ["AB", "22"]]),
    new Map([["Vantor", "Aurora"], ["Vantor 2", "Borealis"], ["8", "9"]]),
    new Map([["12", "X"], ["128", "YY"], ["8", "Z"]]),
    new Map([["x(", "PAREN"], ["x", "EX"], ["1", "ONE"]]),
    new Map([["AB", "A"], ["ABAB", "B"], ["B", "AB"]]), // replacements that re-spell keys
  ];
  const tokens = [
    "A", "B", "AB", "ABAB", "ABA", "Vantor", "Vantor 2", "Vanto", "x(", "x",
    "8", "12", "128", "1", "2", "0", " ", " ", "no", ".",
  ];

  it("agrees on 2000 seeded random texts, maps and chunkings", async () => {
    const rng = mulberry32(0xfa11);
    for (let round = 0; round < 2000; round++) {
      const map = maps[Math.floor(rng() * maps.length)]!;
      const parts = 1 + Math.floor(rng() * 10);
      let text = "";
      for (let i = 0; i < parts; i++) text += tokens[Math.floor(rng() * tokens.length)]!;
      const chunks: string[] = [];
      for (let i = 0; i < text.length; ) {
        const step = 1 + Math.floor(rng() * 5);
        chunks.push(text.slice(i, i + step));
        i += step;
      }
      const detail = `text=${JSON.stringify(text)} chunks=${JSON.stringify(chunks)}`;
      expect(await pump(chunks, map), detail).toBe(rehydrateText(text, map));
    }
  });
});

/**
 * A streaming layer that quietly accumulates is a bug even when its output is
 * right: this runs inside a browser extension, on responses of unbounded length.
 * The holdback is bounded by the longest surrogate, so all but a fixed window
 * must already be emitted at any point in the stream.
 */
describe("bounded buffering", () => {
  it("holds back at most maxSurrogateLength on a long non-matching stream", async () => {
    const map = new Map([["Anjali Verma", "Priya Sharma"]]); // maxLen 12
    const transform = createRehydrateTransform(map);
    const writer = transform.writable.getWriter();
    const reader = transform.readable.getReader();
    let emitted = 0;
    const readAll = (async () => {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        emitted += value.length;
      }
    })();
    // Each chunk ends in a long partial surrogate, the worst case for holdback.
    const chunk = "Anjali Verm-no match here. ";
    let written = 0;
    for (let i = 0; i < 500; i++) {
      await writer.write(chunk);
      written += chunk.length;
      expect(written - emitted, `after chunk ${i}`).toBeLessThanOrEqual(12 + chunk.length);
    }
    await writer.close();
    await readAll;
    expect(emitted).toBe(written);
  });
});

describe("multiple surrogates through the stream", () => {
  const map = new Map([
    ["Vantor", "Globex"],
    ["Anjali Verma", "Priya Sharma"],
    ["Vellum & Gray (Ltd)", "Meridian Trust"],
    ["8842", "1234"],
  ]);
  const text = "Anjali Verma of Vantor filed 8842 with Vellum & Gray (Ltd) for Vantor.";

  it("rehydrates every surrogate, metacharacters included, one character at a time", async () => {
    expect(await pump([...text], map)).toBe(rehydrateText(text, map));
  });

  it("agrees with rehydrateText at every split point", async () => {
    const expected = rehydrateText(text, map);
    for (let i = 1; i < text.length; i++) {
      expect(await pump([text.slice(0, i), text.slice(i)], map), `split at ${i}`).toBe(expected);
    }
  });
});
