import { describe, expect, it } from "vitest";
import { spanFromTokens, type TokenOffset } from "../src/offsets.js";
import { splitWords } from "../src/words.js";

/**
 * Builds an offset array in the shape a fast tokenizer's `offset_mapping` has.
 * The VALUES here are JS string indices because that is what `spanFromTokens`
 * requires; a real fast tokenizer does not hand you those -- see the measured
 * note below, and the emoji tests, for what it hands you instead.
 */
const toks = (...pairs: [number, number][]): TokenOffset[] => pairs.map(([start, end]) => ({ start, end }));

/**
 * Token arrays below marked "measured" were produced by running the real
 * tokenizer this plan pinned -- packages/tier1/models/gliner-pii-base
 * (DebertaV2Tokenizer, Unigram + Metaspace, TemplateProcessing [CLS]/[SEP]) --
 * through Python `tokenizers` 0.22.2 with `Tokenizer.from_file`. Python offsets
 * are CODE POINT indices, so any array over non-BMP text is transcribed here
 * into UTF-16 code units, which is the only coordinate system
 * `String.prototype.slice` speaks. The transcription is stated per test.
 */

describe("spanFromTokens", () => {
  it("spans from the first token's start to the last token's end", () => {
    const text = "call Acme Corp today";
    const span = spanFromTokens(text, toks([0, 4], [5, 9], [10, 14], [15, 20]), 1, 2);
    expect(span).toEqual({ start: 5, end: 14, text: "Acme Corp" });
  });

  it("re-derives text from the string, never from concatenated token pieces", () => {
    // Subword pieces carry markers ("##Corp", "_Acme") and joining them
    // reconstructs something that is NOT a substring of the message. The
    // contract is text === message.slice(start, end); slicing is the only way
    // to satisfy it.
    const text = "Acme Corporation";
    expect(spanFromTokens(text, toks([0, 4], [5, 16]), 0, 1)!.text).toBe("Acme Corporation");
  });

  it("survives an emoji before the span", () => {
    // A smiling face is ONE code point and TWO UTF-16 units. Offsets are JS
    // string indices throughout this system because String.prototype.slice is,
    // and core's contract is expressed in slice terms.
    const text = "🙂 Acme Corp";
    const start = text.indexOf("Acme");
    expect(start).toBe(3);
    const span = spanFromTokens(text, toks([0, 2], [3, 7], [8, 12]), 1, 2);
    expect(span).toEqual({ start: 3, end: 12, text: "Acme Corp" });
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
  });

  it("survives a combining mark inside the span", () => {
    // PRECOMPOSED U+00E9, spelled as an escape because which form a literal
    // lands in depends on the editor that wrote the file. The slice assertion
    // below is true of anything that slices, so an exact expectation is added
    // to give it something only correct offsets can satisfy.
    const text = "caf\u00e9 Ltd";
    const span = spanFromTokens(text, toks([0, 4], [5, 8]), 0, 1);
    expect(span).toEqual({ start: 0, end: 8, text: "caf\u00e9 Ltd" });
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
  });

  it("returns undefined rather than a bad span when indices are out of range", () => {
    // A model may propose a span index past the token array on a truncated
    // input. Returning undefined drops the finding; returning a clamped span
    // sends a wrong offset into normalizeFindings, which throws and kills the
    // whole message.
    expect(spanFromTokens("short", toks([0, 5]), 0, 3)).toBeUndefined();
    expect(spanFromTokens("short", toks([0, 5]), 2, 2)).toBeUndefined();
  });

  it("returns undefined for an inverted or zero-width span", () => {
    expect(spanFromTokens("hello", toks([0, 0], [0, 5]), 0, 0)).toBeUndefined();
  });

  it("offsets a segment-local span into absolute message coordinates", () => {
    const message = "intro line\ncall Acme Corp today";
    const segmentStart = message.indexOf("call");
    const local = spanFromTokens(message.slice(segmentStart), toks([0, 4], [5, 9], [10, 14], [15, 20]), 1, 2)!;
    const absolute = { start: local.start + segmentStart, end: local.end + segmentStart, text: local.text };
    expect(message.slice(absolute.start, absolute.end)).toBe("Acme Corp");
  });
});

describe("spanFromTokens: the span comes from the offsets, not from a search", () => {
  it("resolves the LATER of two identical mentions", () => {
    // MEASURED array. Without this test the whole file passes under an
    // implementation that ignores `tokens` entirely and does
    // `text.indexOf(<the value>)`, because no other message here repeats its
    // entity. Text-search is precisely the drift normalizeFindings exists to
    // catch: it returns offset 0 for the first mention no matter which one the
    // model actually tagged, so applyActions rewrites the wrong occurrence and
    // leaves the tagged one in the message.
    const text = "Acme Corp emailed Acme Corp";
    const measured = toks([0, 4], [4, 9], [9, 17], [17, 22], [22, 27]);
    expect(text.indexOf("Acme Corp")).toBe(0);
    expect(spanFromTokens(text, measured, 3, 4)).toEqual({ start: 18, end: 27, text: "Acme Corp" });
    // ...and the earlier mention still resolves to the earlier offsets.
    expect(spanFromTokens(text, measured, 0, 1)).toEqual({ start: 0, end: 9, text: "Acme Corp" });
  });
});

describe("spanFromTokens: the pinned tokenizer's actual offset convention", () => {
  it("drops the leading whitespace a Metaspace token carries into its offsets", () => {
    // MEASURED, and it contradicts the whitespace-EXCLUDING arrays used above:
    // the pinned tokenizer emits ('_Acme', 4, 9) over "call Acme Corp today",
    // and text.slice(4, 9) is " Acme" -- the space before the word is inside
    // the token. Passing that through unchanged puts a leading space on every
    // span that does not start at index 0, which is most of them.
    //
    // That span still satisfies normalizeFindings, because the text is sliced;
    // it fails later and quieter. The pseudonym vault keys on Finding.text, so
    // " Acme Corp" and "Acme Corp" -- the same organisation named twice in one
    // message, once mid-sentence and once at the start -- mint two different
    // surrogates. Trimming is the one narrowing that cannot lose content: the
    // characters removed are whitespace, and no confidential value begins or
    // ends with whitespace.
    const text = "call Acme Corp today";
    const measured = toks([0, 4], [4, 9], [9, 14], [14, 20]);
    expect(text.slice(4, 9)).toBe(" Acme");
    expect(spanFromTokens(text, measured, 1, 2)).toEqual({ start: 5, end: 14, text: "Acme Corp" });
  });

  it("trims tabs and newlines, not just spaces", () => {
    // MEASURED on "Acme\tCorp\nLtd": the pinned tokenizer's tokens are
    // (0,4) 'Acme', (4,9) '\tCorp', (9,13) '\nLtd'. Metaspace swallows any
    // whitespace run, not only U+0020.
    const text = "Acme\tCorp\nLtd";
    const measured = toks([0, 4], [4, 9], [9, 13]);
    expect(text.slice(4, 9)).toBe("\tCorp");
    expect(spanFromTokens(text, measured, 1, 2)).toEqual({ start: 5, end: 13, text: "Corp\nLtd" });
  });

  it("trims a trailing whitespace token off the end of the range", () => {
    // MEASURED on the same string as below: the bare "_" token at (6, 7) covers
    // one space. As the LAST token of a range it puts the space inside the
    // span, which is the mirror of the leading-space case and just as bad for
    // the vault key.
    const text = "family 👨‍👩‍👧 photo";
    expect(spanFromTokens(text, toks([0, 6], [6, 7], [7, 9]), 0, 1)).toEqual({
      start: 0,
      end: 6,
      text: "family",
    });
  });

  it("trims a whitespace boundary that sits after an astral character", () => {
    // The MEASURED convention applied to the emoji string, transcribed to
    // UTF-16: the smiling face is (0, 2), then "_Acme" is (2, 7) -- space
    // included -- and "_Corp" is (7, 12). Trimming has to walk CODE UNITS: the
    // space is code unit 2 but code POINT 1, so a trim that indexed by code
    // point would look at "A", find no whitespace, and hand back " Acme Corp".
    const text = "🙂 Acme Corp";
    expect(text.slice(2, 7)).toBe(" Acme");
    expect(spanFromTokens(text, toks([0, 2], [2, 7], [7, 12]), 1, 2)).toEqual({
      start: 3,
      end: 12,
      text: "Acme Corp",
    });
  });

  it("keeps offsets relative to the ORIGINAL text, not a trimmed copy", () => {
    // MEASURED on "  Acme Corp  ": the tokenizer's own normalizer strips the
    // outer whitespace, yet the offsets it reports still index the original --
    // (2, 6) and (6, 11). Anything here that trimmed the text before slicing
    // would shift every offset left by two and report "me Corp".
    const text = "  Acme Corp  ";
    expect(text).toHaveLength(13);
    expect(spanFromTokens(text, toks([2, 6], [6, 11]), 0, 1)).toEqual({
      start: 2,
      end: 11,
      text: "Acme Corp",
    });
  });

  it("trims the zero-width separators JS \\s does not cover", () => {
    // MEASURED, and the reason the trim class is derived from the tokenizer's
    // fold behaviour rather than from JS `\s`. Encoding "call" + C + "Acme Corp"
    // collapses C into the FOLLOWING token's offsets, (4, 9), identically for a
    // space and for each of U+200B, U+200C and U+200D -- but `\s` contains
    // none of the three. A `\s`-only trim returns "<ZWSP>Acme Corp" here while
    // the visually identical U+FEFF case returns "Acme Corp": two vault keys
    // for one name, reachable by anyone who can type an invisible character.
    for (const separator of ["\u200b", "\u200c", "\u200d", "\ufeff"]) {
      const text = `call${separator}Acme Corp`;
      expect(spanFromTokens(text, toks([0, 4], [4, 9], [9, 14]), 1, 2)).toEqual({
        start: 5,
        end: 14,
        text: "Acme Corp",
      });
    }
  });

  it("trims the non-ASCII separators the tokenizer folds", () => {
    // MEASURED on the same shape: NBSP and IDEOGRAPHIC SPACE fold exactly as
    // U+0020 does, so an ASCII-only trim class leaves them on the span.
    for (const separator of ["\u00a0", "\u3000", "\u2009", "\u202f"]) {
      const text = `call${separator}Acme Corp`;
      expect(spanFromTokens(text, toks([0, 4], [4, 9], [9, 14]), 1, 2)).toEqual({
        start: 5,
        end: 14,
        text: "Acme Corp",
      });
    }
  });

  it("leaves an INTERIOR separator alone", () => {
    // MEASURED: "Acme" + NBSP + "Corp Ltd" tokenizes to (0,4) (4,9) (9,13).
    // Both loops walk inward from an end and stop at the first character
    // outside the class, so a separator inside the value is never reached --
    // which is what makes trimming U+200D safe, since a ZWJ only carries
    // meaning between two glyphs and a span's interior is never touched.
    const text = "Acme\u00a0Corp Ltd";
    expect(spanFromTokens(text, toks([0, 4], [4, 9], [9, 13]), 0, 1)).toEqual({
      start: 0,
      end: 9,
      text: "Acme\u00a0Corp",
    });
  });

  it("trims a run of separators, not just one", () => {
    // CONSTRUCTED, not measured: the pinned tokenizer folds only the LAST
    // separator of a run into the token (measured -- "call" + NBSP + ZWSP +
    // "Acme" gives (5, 10), not (4, 10)). This is the array a tokenizer that
    // folded the whole run would give, and it is the only thing in the file
    // that drives either trim loop past a single iteration.
    const text = "call \u200bAcme";
    expect(spanFromTokens(text, toks([0, 4], [4, 10]), 1, 1)).toEqual({
      start: 6,
      end: 10,
      text: "Acme",
    });
  });

  it("rejects a span anchored on a zero-width special token", () => {
    // MEASURED: TemplateProcessing wraps every encoding in [CLS] .. [SEP], and
    // both come back with offsets (0, 0). A span whose first index lands on
    // [CLS] would otherwise report start = 0 -- an offset with no relationship
    // to where the entity is, yet perfectly slice-consistent, so
    // normalizeFindings waves it through and applyActions rewrites from the
    // top of the message.
    const text = "call Acme Corp today";
    const withSpecials = toks([0, 0], [0, 4], [4, 9], [9, 14], [14, 20], [0, 0]);
    expect(spanFromTokens(text, withSpecials, 0, 3)).toBeUndefined();
    expect(spanFromTokens(text, withSpecials, 2, 5)).toBeUndefined();
    // The same array with both indices over real tokens is fine; the rejection
    // is about the anchors, not about specials being present in the array.
    expect(spanFromTokens(text, withSpecials, 2, 3)).toEqual({ start: 5, end: 14, text: "Acme Corp" });
  });

  it("rejects a zero-width token as an anchor even when it is not a special", () => {
    expect(spanFromTokens("call Acme", toks([0, 4], [4, 4], [4, 9]), 1, 2)).toBeUndefined();
    expect(spanFromTokens("call Acme", toks([0, 4], [4, 9], [9, 9]), 0, 2)).toBeUndefined();
  });

  it("returns undefined when the whole token range is whitespace", () => {
    // MEASURED on "family <ZWJ-joined family emoji> photo": the pinned
    // tokenizer emits a bare '_' token whose offsets cover the single space at
    // index 6. Trimming empties it, and an empty span is not a finding.
    const text = "family 👨‍👩‍👧 photo";
    expect(text.slice(6, 7)).toBe(" ");
    expect(spanFromTokens(text, toks([0, 6], [6, 7], [7, 9]), 1, 1)).toBeUndefined();
  });

  it("carries a Devanagari span with its matras intact", () => {
    // MEASURED: the pinned tokenizer splits Mumbai in Devanagari into a base
    // token that swallows the preceding space, (15,17), plus one token per
    // following matra or letter -- (17,18), (18,19), (19,20), (20,21). Matras
    // are combining marks, so a rule that refused to start or end a span on one
    // would drop most Indic names outright; this repo's own policy compiler
    // emits Indian entity types, so that is not a hypothetical corpus.
    const text = "Priya Sharma at मुंबई office";
    const measured = toks([0, 5], [5, 12], [12, 15], [15, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 28]);
    expect(text.length).toBe(28);
    const span = spanFromTokens(text, measured, 3, 7);
    expect(span).toEqual({ start: 16, end: 21, text: "मुंबई" });
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
  });

  it("returns the text in the message's own normalisation form", () => {
    // NOT a measured array: the pinned tokenizer composes NFKC and reports
    // (0, 4) here (see the next test). This is the array a NON-normalising
    // tokenizer gives for the same string, and it is the case that catches a
    // "helpful" normalize() on the way out. normalizeFindings compares the
    // STRINGS -- `f.text !== text.slice(f.start, f.end)` -- and throws; the
    // lengths appear only in the message it throws, so a same-length
    // substitution would fail just as hard. The contract is the message's own
    // code units, not a canonical form of them.
    const text = "cafe\u0301 Ltd";
    const span = spanFromTokens(text, toks([0, 5], [5, 9]), 0, 0);
    expect(span!.text).toHaveLength(5);
    expect(span!.end - span!.start).toBe(span!.text.length);
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
    expect(span!.text.normalize("NFC")).toHaveLength(4);
  });

  it("takes in a trailing combining mark the tokenizer left outside", () => {
    // MEASURED, and the ugliest thing the pinned tokenizer does: its normalizer
    // composes NFKC, so on DECOMPOSED "cafe" + U+0301 the token for the word
    // comes back as (0, 4) -- text.slice(0, 4) is "cafe" and the acute at index
    // 4 belongs to no token at all.
    //
    // Reporting (0, 4) unchanged is NOT the cosmetic outcome it looks like.
    // applyActions calls `vault.mint(conversationId, f.text, ...)`, so the
    // value stored against the surrogate would be "cafe" -- accent stripped --
    // and rehydration hands "cafe" back. A wrong value round-trips through the
    // pseudonymization layer, which is worse than the orphaned mark left in the
    // rewritten message.
    //
    // Rejecting is still refused: that drops every finding on decomposed
    // accented text, and a dropped finding leaks the value it was meant to
    // catch. Widening is the safe direction because a Mark modifies the
    // character BEFORE it, which the span already contains, so it cannot reach
    // a base character, and therefore cannot reach a neighbouring word in any
    // sense that changes a value.
    //
    // The offsets in THIS test are the subword tokenizer's, which is a path
    // nothing feeds any more -- see the word-path test below for what the live
    // caller actually hands in, which is not the same shape.
    const text = "cafe\u0301 Ltd";
    expect(text.length).toBe(9);
    expect(text.charCodeAt(4)).toBe(0x0301);
    const span = spanFromTokens(text, toks([0, 4], [5, 9]), 0, 0);
    expect(span).toEqual({ start: 0, end: 5, text: "cafe\u0301" });
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
    expect(span!.text.normalize("NFC")).toBe("caf\u00e9");
  });

  it("absorbs the mark WORD splitWords emits, which is not the same case as above", () => {
    // The live path, and the thing the header's adjustment 2 is about. On the
    // subword path the acute belonged to no token; on the WORD path it is a
    // word of its own, with its own slot on the model's axis and its own
    // scores, because a combining mark is neither \p{L} nor \p{N} and so falls
    // to the splitter's single-code-point branch. The widening therefore
    // reaches PAST the words the caller named -- deliberately, and this test is
    // what stops that from being rediscovered as a surprise.
    const text = "call Andre\u0301 Corp today";
    const words = splitWords(text);
    // Derived from the splitter, not restated: word 2 IS the bare acute.
    expect(words.map((w) => w.text)).toEqual(["call", "Andre", "\u0301", "Corp", "today"]);
    expect(words[2]).toEqual({ text: "\u0301", start: 10, end: 11 });

    // The model said ONE word wide, word 1, ending at index 10.
    const span = spanFromTokens(text, words, 1, 1);
    expect(words[1]!.end).toBe(10);
    expect(span).toEqual({ start: 5, end: 11, text: "Andre\u0301" });
    // Still exactly a slice of the message, which is core's whole contract.
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
    // And it stops at the end of the grapheme cluster: "Corp" is untouched.
    expect(span!.end).toBeLessThan(words[3]!.start);
    expect(span!.text.normalize("NFC")).toBe("Andr\u00e9");
  });

  it("widens over a RUN of stacked combining marks", () => {
    // MEASURED, and the case that makes this a name problem rather than a
    // curiosity: decomposed Vietnamese stacks two marks on one vowel, and the
    // pinned tokenizer reports (0,1) (1,4) (4,5) (7,11) for "Nguye" + U+0302 +
    // U+0303 + " Ltd" -- BOTH marks at indices 5 and 6 belong to no token. A
    // widen that ran once would mint the vault entry under "Nguye" + U+0302,
    // which is not a spelling of anything.
    const text = "Nguye\u0302\u0303 Ltd";
    expect(text).toHaveLength(11);
    const span = spanFromTokens(text, toks([0, 1], [1, 4], [4, 5], [7, 11]), 0, 2);
    expect(span).toEqual({ start: 0, end: 7, text: "Nguye\u0302\u0303" });
    expect(span!.text.normalize("NFC")).toBe("Nguy\u1ec5");
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
  });

  it("widens by whole code points over an ASTRAL combining mark", () => {
    // U+1D165 (MUSICAL SYMBOL COMBINING STEM) is category Mn and TWO code
    // units. A widen that advanced by code units would test `\p{M}` against a
    // lone surrogate, never match, and stop -- leaving the boundary inside the
    // pair for the surrogate check to reject, turning a widen into a drop.
    const text = `Acme\u{1D165} Ltd`;
    expect(text).toHaveLength(10);
    expect(spanFromTokens(text, toks([0, 4], [7, 10]), 0, 0)).toEqual({
      start: 0,
      end: 6,
      text: `Acme\u{1D165}`,
    });
  });

  it("does not widen at the START, where the base character is outside", () => {
    // The asymmetry, pinned. A mark at `start` belongs to a base character the
    // span does not contain, so pulling it in would mean pulling in a base
    // character the model did not select -- which can change the value, unlike
    // widening at the end. Measured that this is reachable: the pinned
    // tokenizer emits Devanagari matras as tokens of their own, (17,18) and
    // (18,19) in "Priya Sharma at <Mumbai> office".
    const text = "cafe\u0301 Ltd";
    expect(spanFromTokens(text, toks([4, 5], [5, 9]), 0, 0)).toEqual({
      start: 4,
      end: 5,
      text: "\u0301",
    });
  });
});

describe("spanFromTokens: UTF-16 boundaries", () => {
  it("rejects a boundary that falls between the halves of a surrogate pair", () => {
    // The reachable version of this: Python `tokenizers` reports CODE POINT
    // offsets, and the pinned tokenizer MEASURED on this string gives the first
    // emoji (7, 8). Handed to JS unconverted, slice(7, 8) is a lone high
    // surrogate -- a well-formed JS string that isWellFormed() rejects, that
    // encodes to U+FFFD on the wire, and that normalizeFindings accepts without
    // complaint because the text was sliced. Only this check stops it.
    const text = "family 👨‍👩‍👧 photo";
    expect(text.length).toBe(21);
    // 0xd83d is the HIGH half of the first emoji's surrogate pair, alone.
    expect(text.slice(7, 8)).toBe("\ud83d");
    expect(spanFromTokens(text, toks([0, 6], [6, 7], [7, 8], [8, 9]), 2, 2)).toBeUndefined();
    // Split at the start boundary rather than the end: same rejection.
    expect(spanFromTokens(text, toks([0, 6], [6, 7], [7, 8], [8, 10]), 3, 3)).toBeUndefined();
  });

  it("accepts the same span once the offsets are transcribed to UTF-16", () => {
    // The MEASURED code-point array (0,6) (6,7) (7,8) (8,9) (9,10) (10,11)
    // (11,12) (12,18) transcribed to code units: each of the three emoji costs
    // one extra unit.
    const text = "family 👨‍👩‍👧 photo";
    const utf16 = toks([0, 6], [6, 7], [7, 9], [9, 10], [10, 12], [12, 13], [13, 15], [15, 21]);
    const span = spanFromTokens(text, utf16, 1, 6);
    expect(span).toEqual({ start: 7, end: 15, text: "👨‍👩‍👧" });
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
    // Five code points in eight code units: three pairs and two joiners, whole.
    expect([...span!.text]).toHaveLength(5);
  });

  it("still spans an entity that a stray lone surrogate sits against", () => {
    // The boundary at index 1 has a HIGH surrogate behind it and "A" in front,
    // so it is not a split pair -- it is a message that was already malformed
    // before detection ran. Refusing the span here would drop the finding and
    // ship the value, which is the failure the rejection exists to prevent.
    const text = "\ud800Acme Corp";
    expect(text.charCodeAt(0)).toBe(0xd800);
    expect(spanFromTokens(text, toks([0, 1], [1, 5], [5, 10]), 1, 2)).toEqual({
      start: 1,
      end: 10,
      text: "Acme Corp",
    });
  });

  it("rejects a split at both extremes of the surrogate ranges", () => {
    // The four boundary constants, pinned. U+10000 is the LOWEST astral code
    // point -- high half 0xD800, low half 0xDC00 -- and U+10FFFF the highest,
    // 0xDBFF and 0xDFFF. Between them they witness every edge of both ranges:
    // narrowing any one constant by a single value lets one of these through.
    for (const codePoint of [0x10000, 0x10ffff]) {
      const text = `x${String.fromCodePoint(codePoint)}y`;
      expect(text).toHaveLength(4);
      // A boundary at index 2 falls between the two halves.
      expect(spanFromTokens(text, toks([0, 2], [2, 4]), 0, 0)).toBeUndefined();
      expect(spanFromTokens(text, toks([0, 2], [2, 4]), 1, 1)).toBeUndefined();
      // The whole pair, spanned properly, is fine.
      expect(spanFromTokens(text, toks([0, 1], [1, 3], [3, 4]), 1, 1)).toEqual({
        start: 1,
        end: 3,
        text: String.fromCodePoint(codePoint),
      });
    }
  });

  it("keeps a lone surrogate that was already in the message intact", () => {
    // A pre-existing unpaired surrogate is not a split pair, and dropping the
    // finding over it would let the surrounding value through.
    const text = "\ud800 Acme";
    expect(text.charCodeAt(0)).toBe(0xd800);
    expect(spanFromTokens(text, toks([0, 1], [1, 6]), 1, 1)).toEqual({ start: 2, end: 6, text: "Acme" });
  });
});

describe("spanFromTokens: malformed input", () => {
  it("returns undefined for non-integer, NaN or negative token indices", () => {
    const tokens = toks([0, 4], [5, 9]);
    expect(spanFromTokens("call Acme", tokens, Number.NaN, 1)).toBeUndefined();
    expect(spanFromTokens("call Acme", tokens, 0, Number.NaN)).toBeUndefined();
    expect(spanFromTokens("call Acme", tokens, 0.5, 1)).toBeUndefined();
    expect(spanFromTokens("call Acme", tokens, 0, 1.5)).toBeUndefined();
    expect(spanFromTokens("call Acme", tokens, -1, 1)).toBeUndefined();
    expect(spanFromTokens("call Acme", tokens, 0, -1)).toBeUndefined();
    expect(spanFromTokens("call Acme", tokens, Number.POSITIVE_INFINITY, 1)).toBeUndefined();
  });

  it("returns undefined when firstToken is past lastToken", () => {
    expect(spanFromTokens("call Acme", toks([0, 4], [5, 9]), 1, 0)).toBeUndefined();
  });

  it("returns undefined when reversed indices would still run forwards", () => {
    // The one case where `lastToken < firstToken` is the only thing standing in
    // the way: with an out-of-order array, swapping the indices yields
    // start = 0 and end = 9, which runs forwards and slices cleanly, and the
    // containment walk never executes because its bounds are already crossed.
    expect(spanFromTokens("call Acme Corp", toks([5, 9], [0, 4]), 1, 0)).toBeUndefined();
  });

  it("returns undefined for a hole in the token array", () => {
    const sparse: TokenOffset[] = [];
    sparse[2] = { start: 0, end: 4 };
    expect(sparse).toHaveLength(3);
    expect(spanFromTokens("call Acme", sparse, 0, 2)).toBeUndefined();
  });

  it("returns undefined for a hole strictly inside the range", () => {
    const sparse: TokenOffset[] = [];
    sparse[0] = { start: 0, end: 4 };
    sparse[2] = { start: 5, end: 9 };
    expect(spanFromTokens("call Acme", sparse, 0, 2)).toBeUndefined();
  });

  it("returns undefined for a non-integer offset on an INTERIOR token", () => {
    expect(spanFromTokens("call Acme Corp", toks([0, 4], [4.5, 9], [9, 14]), 0, 2)).toBeUndefined();
    expect(spanFromTokens("call Acme Corp", toks([0, 4], [4, Number.NaN], [9, 14]), 0, 2)).toBeUndefined();
  });

  it("returns undefined for an empty token array", () => {
    expect(spanFromTokens("call Acme", [], 0, 0)).toBeUndefined();
  });

  it("returns undefined for non-integer or NaN token offsets", () => {
    expect(spanFromTokens("call Acme", toks([0, 4], [Number.NaN, 9]), 1, 1)).toBeUndefined();
    expect(spanFromTokens("call Acme", toks([0, 4], [5, Number.NaN]), 1, 1)).toBeUndefined();
    expect(spanFromTokens("call Acme", toks([0, 4], [5.5, 9]), 1, 1)).toBeUndefined();
    expect(spanFromTokens("call Acme", toks([0, 4], [5, 9.5]), 1, 1)).toBeUndefined();
    expect(spanFromTokens("call Acme", toks([-1, 4], [5, 9]), 0, 1)).toBeUndefined();
  });

  it("returns undefined when the span runs past the end of the text", () => {
    // A segment-local offset array handed the wrong segment: every index is a
    // plausible integer and only the length disagrees.
    expect(spanFromTokens("call Acme", toks([0, 4], [5, 40]), 0, 1)).toBeUndefined();
  });

  it("returns undefined when a token inside the range lies outside the span", () => {
    // Out-of-order offsets: (10,14) cannot be part of a span that ends at 9.
    // Reading only the endpoints would return a span that slices cleanly and
    // describes a token range that does not exist.
    expect(spanFromTokens("call Acme Corp today", toks([0, 4], [10, 14], [5, 9]), 0, 2)).toBeUndefined();
    // Endpoint inversion: this one never reaches the walk -- reversing the two
    // tokens makes the span itself run backwards, 15 to 9.
    expect(spanFromTokens("call Acme Corp today", toks([15, 20], [5, 9]), 0, 1)).toBeUndefined();
  });

  it("returns undefined for a zero-width token in the MIDDLE of the range", () => {
    // Containment alone cannot see this one: (0, 0) sits inside any span that
    // starts at index 0. A [SEP] between two sequences is where it comes from,
    // and a span that crosses one is not a span over anything.
    expect(spanFromTokens("call Acme Corp", toks([0, 4], [0, 0], [4, 9], [9, 14]), 0, 3)).toBeUndefined();
  });

  it("returns undefined when a token in the range starts BEFORE the span", () => {
    // The other half of the containment check, which the out-of-order test
    // above does not reach: here the span is [5, 14) and token 1 sits at
    // [0, 4), entirely before it. Without `token.start >= start` this returns
    // {5, 14, "Acme Corp"} -- a clean-slicing span over a token range that is
    // not contiguous with it.
    expect(spanFromTokens("call Acme Corp today", toks([5, 9], [0, 4], [10, 14]), 0, 2)).toBeUndefined();
  });

  it("returns undefined for a token whose own offsets are inverted", () => {
    expect(spanFromTokens("call Acme Corp", toks([0, 4], [9, 5], [10, 14]), 0, 2)).toBeUndefined();
  });
});
