import { describe, expect, it } from "vitest";
import { segmentText } from "../../src/segment/segment.js";

const SAMPLE = [
  "Please review my setup.",
  "```",
  "API_KEY=abc123",
  "```",
  "username: john",
  "password: hunter2",
  "Thanks for the help!",
].join("\n");

describe("segmentText", () => {
  it("splits into prose / code / kv / prose", () => {
    const segs = segmentText(SAMPLE);
    expect(segs.map((s) => s.kind)).toEqual(["prose", "code", "kv", "prose"]);
  });

  it("keeps absolute offsets: text === original.slice(start, end)", () => {
    for (const s of segmentText(SAMPLE)) {
      expect(s.text).toBe(SAMPLE.slice(s.start, s.end));
    }
  });

  it("covers the whole input with no gaps between segment bounds", () => {
    const segs = segmentText(SAMPLE);
    expect(segs[0]!.start).toBe(0);
    expect(segs[segs.length - 1]!.end).toBe(SAMPLE.length);
    for (let i = 1; i < segs.length; i++) {
      expect(segs[i]!.start).toBe(segs[i - 1]!.end);
    }
  });

  it("treats an unclosed fence as code to the end", () => {
    const segs = segmentText("hello\n```\nSECRET=x");
    expect(segs.map((s) => s.kind)).toEqual(["prose", "code"]);
  });

  it("returns a single prose segment for plain text", () => {
    expect(segmentText("just words here")).toEqual([
      { start: 0, end: 15, kind: "prose", text: "just words here" },
    ]);
  });

  it("handles empty input", () => {
    expect(segmentText("")).toEqual([]);
  });
});

describe("segmentText documented quirks", () => {
  // Pinned, not accidental: the closing-fence match consumes "\n" but not
  // "\r\n", so a CRLF fence leaves the line ending at the head of the next
  // segment. See the Pass 1 comment in src/segment/segment.ts.
  it("leaves a stray CRLF prose segment after a CRLF-closed fence", () => {
    const input = "```\r\nk=v\r\n```\r\nafter";
    const segs = segmentText(input);
    expect(segs).toEqual([
      { start: 0, end: 13, kind: "code", text: "```\r\nk=v\r\n```" },
      { start: 13, end: 20, kind: "prose", text: "\r\nafter" },
    ]);
    // The quirk must never cost us the invariants.
    expect(segs[0]!.start).toBe(0);
    expect(segs[segs.length - 1]!.end).toBe(input.length);
    for (const s of segs) expect(s.text).toBe(input.slice(s.start, s.end));
  });

  it("emits the CRLF remainder as its own segment when the fence ends the input", () => {
    const input = "```\r\nx\r\n```\r\n";
    const segs = segmentText(input);
    expect(segs).toEqual([
      { start: 0, end: 11, kind: "code", text: "```\r\nx\r\n```" },
      { start: 11, end: 13, kind: "prose", text: "\r\n" },
    ]);
  });

  it("splits a kv run on a blank line into kv / prose / kv", () => {
    expect(segmentText("k=v\n\nk2=v2")).toEqual([
      { start: 0, end: 4, kind: "kv", text: "k=v\n" },
      { start: 4, end: 5, kind: "prose", text: "\n" },
      { start: 5, end: 10, kind: "kv", text: "k2=v2" },
    ]);
  });
});

describe("segmentText invariants (property test)", () => {
  /** Deterministic xorshift PRNG - CI must never flake on a random seed. */
  function rng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
      s ^= s << 13;
      s >>>= 0;
      s ^= s >>> 17;
      s ^= s << 5;
      s >>>= 0;
      return s / 0x100000000;
    };
  }

  const LINES = [
    "Please review my setup.", // prose
    "here is some text", // prose
    "no colon or equals sign here", // prose
    "# heading", // prose
    "API_KEY=abc123", // kv
    "username: john", // kv
    "  db.host = localhost", // kv
    "my-var=1", // kv
    "```", // fence (may open or close)
    "```ts", // fence with info string
    "", // blank line
    "   ", // whitespace-only line
    "text ```inline``` text", // fence mid-line
    "key =", // separator with no value
  ];

  function generate(rand: () => number): string {
    const count = Math.floor(rand() * 14);
    const picked: string[] = [];
    for (let i = 0; i < count; i++) {
      picked.push(LINES[Math.floor(rand() * LINES.length)]!);
    }
    const eol = rand() < 0.25 ? "\r\n" : "\n";
    let text = picked.join(eol);
    const tail = rand();
    if (tail < 0.25 && text.length > 0) text += eol; // trailing newline
    else if (tail < 0.35) text += "```"; // unterminated fence at EOF
    return text; // otherwise: no trailing newline
  }

  it("tiles the input exactly for 300 generated documents", () => {
    const rand = rng(0xc0ffee);
    for (let iter = 0; iter < 300; iter++) {
      const input = generate(rand);
      const segs = segmentText(input);
      const label = `iteration ${iter}, input ${JSON.stringify(input)}`;

      if (input.length === 0) {
        expect(segs, label).toEqual([]);
        continue;
      }

      expect(segs.length, label).toBeGreaterThan(0);
      expect(segs[0]!.start, label).toBe(0);
      expect(segs[segs.length - 1]!.end, label).toBe(input.length);

      for (let i = 1; i < segs.length; i++) {
        expect(segs[i]!.start, `${label}, boundary ${i}`).toBe(segs[i - 1]!.end);
      }

      for (const s of segs) {
        expect(s.text, label).toBe(input.slice(s.start, s.end));
        expect(["prose", "code", "kv"], label).toContain(s.kind);
        expect(s.end, label).toBeGreaterThan(s.start);
      }
    }
  });
});
