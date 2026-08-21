import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { spanFromTokens } from "../src/offsets.js";
import { splitWords } from "../src/words.js";

/**
 * Spelled as escapes rather than pasted raw, so the source stays readable and
 * an editor cannot silently normalise one of them away.
 */
const BOM = "\u{FEFF}";
const FILE_SEPARATOR = "\x1C";
const NEL = "\x85";
const ZWSP = "\u{200B}";
const NBSP = "\xA0";
const SOH = "\x01";

interface OracleCase {
  readonly text: string;
  readonly words: readonly (readonly [string, number, number])[];
}

interface Oracle {
  readonly oracle: {
    readonly source: string;
    readonly pattern: string;
    readonly python: string;
    readonly unicodeVersion: string;
  };
  readonly cases: readonly OracleCase[];
}

const HERE = dirname(fileURLToPath(import.meta.url));
const ORACLE = JSON.parse(
  readFileSync(join(HERE, "fixtures", "word-split.json"), "utf8"),
) as Oracle;

describe("splitWords", () => {
  it("returns offsets that slice back to the word, always", () => {
    // The invariant the whole tier rests on. Includes astral characters,
    // because UTF-16 is where this breaks.
    for (const text of [
      "call Acme Corp today",
      "\u{1F389} Acme",
      "caf\u{E9} Ltd",
      "\u{5317}\u{4EAC} Corp",
      "ab",
      "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} family",
    ]) {
      for (const w of splitWords(text)) {
        expect(text.slice(w.start, w.end)).toBe(w.text);
      }
    }
  });

  it("keeps hyphenated and underscored words whole", () => {
    expect(splitWords("well-known co_op").map((w) => w.text)).toEqual(["well-known", "co_op"]);
  });

  it("treats every non-space non-word character as its own word", () => {
    expect(splitWords("a, b!").map((w) => w.text)).toEqual(["a", ",", "b", "!"]);
  });

  it("uses Unicode word characters, not ASCII", () => {
    // GLiNER.js shipped regex fails this: JS \w is ASCII-only, so an accented
    // word splits in two and every downstream word index shifts.
    expect(splitWords("caf\u{E9} Ltd").map((w) => w.text)).toEqual(["caf\u{E9}", "Ltd"]);
    expect(splitWords("\u{5317}\u{4EAC} Corp").map((w) => w.text)).toEqual([
      "\u{5317}\u{4EAC}",
      "Corp",
    ]);
  });

  it("treats BOM as a word, not whitespace, matching Python", () => {
    // JS \s includes U+FEFF; Python does not. Getting this wrong shifts every
    // subsequent word index on any text containing a BOM.
    expect(splitWords("a" + BOM + "b").map((w) => w.text)).toEqual(["a", BOM, "b"]);
  });

  it("treats C0 separators and NEL as whitespace, matching Python", () => {
    // Python \s includes U+001C-U+001F and U+0085; JS does not.
    expect(splitWords("a" + FILE_SEPARATOR + "b").map((w) => w.text)).toEqual(["a", "b"]);
    expect(splitWords("a" + NEL + "b").map((w) => w.text)).toEqual(["a", "b"]);
  });

  it("treats the zero-width format characters as words, matching Python", () => {
    // ZWSP is not Python whitespace, so it is a word of its own. It is also not
    // a word CHARACTER, so it never joins the letters either side of it.
    expect(splitWords("a" + ZWSP + "b").map((w) => w.text)).toEqual(["a", ZWSP, "b"]);
  });

  it("treats NBSP as whitespace, matching both", () => {
    expect(splitWords("a" + NBSP + "b").map((w) => w.text)).toEqual(["a", "b"]);
  });

  it("measures offsets in UTF-16 code units, not code points", () => {
    // An astral character is two code units. Python's own indices are code
    // POINTS and would put `priya` at 2, which slices to the wrong place in a
    // JS string; the reference offsets below, converted to UTF-16, put it at 3.
    // The email splits at its punctuation because `@` and `.` are not word
    // characters -- that is the reference's segmentation, not a shortcoming.
    const words = splitWords("\u{1F389} priya@acme.io");
    expect(words.map((w) => [w.text, w.start, w.end])).toEqual([
      ["\u{1F389}", 0, 2],
      ["priya", 3, 8],
      ["@", 8, 9],
      ["acme", 9, 13],
      [".", 13, 14],
      ["io", 14, 16],
    ]);
  });

  it("never emits an empty word, and never overlaps or reorders", () => {
    for (const c of ORACLE.cases) {
      let previousEnd = 0;
      for (const w of splitWords(c.text)) {
        expect(w.end).toBeGreaterThan(w.start);
        expect(w.start).toBeGreaterThanOrEqual(previousEnd);
        previousEnd = w.end;
      }
      expect(previousEnd).toBeLessThanOrEqual(c.text.length);
    }
  });

  it("is not stateful across calls", () => {
    // A `g`-flagged module-level regex driven with .exec carries lastIndex from
    // one call into the next; the second call would then start mid-string.
    const once = splitWords("call Acme Corp today");
    const twice = splitWords("call Acme Corp today");
    expect(twice).toEqual(once);
  });

  it("returns nothing for an empty or all-separator text", () => {
    expect(splitWords("")).toEqual([]);
    expect(splitWords("  \t\n  ")).toEqual([]);
  });

  it("feeds spanFromTokens directly, in the same coordinate system", () => {
    // Task 6 consumes {start, end}; this is the seam between the two, and a
    // WordSpan has to be usable as a TokenOffset without conversion.
    const text = "Contact Priya Sharma at priya@acme.io today";
    const words = splitWords(text);
    expect(spanFromTokens(text, words, 1, 2)).toEqual({
      start: 8,
      end: 20,
      text: "Priya Sharma",
    });
  });

  describe("against the GLiNER Python reference", () => {
    it("reproduces the oracle token-for-token, offsets included", () => {
      let tokens = 0;
      const failures: string[] = [];
      for (const c of ORACLE.cases) {
        const got = splitWords(c.text).map((w) => [w.text, w.start, w.end]);
        const want = c.words.map((w) => [w[0], w[1], w[2]]);
        tokens += want.length;
        if (JSON.stringify(got) !== JSON.stringify(want)) {
          failures.push(
            `${JSON.stringify(c.text)}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`,
          );
        }
      }
      expect(failures).toEqual([]);
      // Guards the fixture itself: an empty or truncated oracle would make the
      // loop above pass while proving nothing.
      expect(ORACLE.cases.length).toBeGreaterThan(300);
      expect(tokens).toBeGreaterThan(2000);
      expect(ORACLE.oracle.pattern).toBe("\\w+(?:[-_]\\w+)*|\\S");
    });
  });

  it("makes a control character its own word, which is what desyncs the reference", () => {
    // U+0001 is not whitespace in Python, so it IS a word -- and on the pinned
    // base tokenizer it encodes to zero subwords. encode.ts is where that is
    // handled; here the point is only that the splitter does emit it.
    expect(splitWords("call " + SOH + " Acme Corp").map((w) => w.text)).toEqual([
      "call",
      SOH,
      "Acme",
      "Corp",
    ]);
  });
});
