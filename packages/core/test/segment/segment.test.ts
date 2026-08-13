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
