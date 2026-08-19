export interface TokenOffset {
  /** JS string index -- a UTF-16 code unit position -- inclusive. */
  readonly start: number;
  /** JS string index -- a UTF-16 code unit position -- exclusive. */
  readonly end: number;
}

export interface CharSpan {
  readonly start: number;
  readonly end: number;
  readonly text: string;
}

/**
 * The separator class trimmed off a span's two ENDS, derived from what the
 * pinned tokenizer's normalizer folds rather than from JS `\s`.
 *
 * Measured, by encoding `"call" + C + "Acme Corp"` for each candidate C: the
 * normalizer collapses the separator into the FOLLOWING token's offsets --
 * `(4, 9)` covering `C + "Acme"` -- identically for U+0020, U+0009, U+000A,
 * U+000C, U+000D, U+00A0, U+1680, U+2000, U+2009, U+2028, U+2029, U+202F,
 * U+205F, U+3000, U+FEFF, and also for U+200B, U+200C and U+200D. Only U+000B
 * behaved differently (it split into its own tokens).
 *
 * JS `\s` covers the first fifteen and NOT the three zero-width format
 * characters, so `\s` alone leaves exactly the separators an adversary would
 * reach for: measured, `"call " + U+200B + "Acme Corp"` puts `U+200B + "Acme"`
 * at (5, 10), and a `\s`-only trim hands back `"<ZWSP>Acme Corp"` while the
 * visually identical U+FEFF case trims clean. Two spellings of one name, two
 * vault surrogates.
 *
 * Trimming U+200D (ZWJ) is the one deliberate call here. A ZWJ is load-bearing
 * INSIDE an emoji sequence, and nothing in this file touches a span's interior
 * -- both loops below walk inward from an end and stop at the first character
 * outside this class. A ZWJ *at a boundary* is a joiner whose partner glyph is
 * on the other side of the boundary, so it joins nothing; the same holds for
 * U+200C between two words. Interior occurrences, which are the ones that carry
 * meaning in Persian and Devanagari orthography, are never reached.
 */
const TRIMMABLE = /[\s\u200b\u200c\u200d]/;

/**
 * Unicode Mark (Mn, Mc, Me) -- a character that has no standing of its own and
 * modifies the base character before it.
 */
const COMBINING_MARK = /\p{M}/u;

const isHighSurrogate = (code: number): boolean => code >= 0xd800 && code <= 0xdbff;
const isLowSurrogate = (code: number): boolean => code >= 0xdc00 && code <= 0xdfff;

/**
 * True when `index` falls BETWEEN the two halves of one surrogate pair.
 *
 * Deliberately not "is `index` a surrogate": a message can already contain an
 * unpaired surrogate, and refusing to span it would drop a finding over text
 * that was malformed before we touched it.
 */
function splitsSurrogatePair(text: string, index: number): boolean {
  if (index <= 0 || index >= text.length) return false;
  return isHighSurrogate(text.charCodeAt(index - 1)) && isLowSurrogate(text.charCodeAt(index));
}

/**
 * Turn a token-index span into a character span over `text`.
 *
 * `tokens` and the two indices MUST be in the same coordinate system. If the
 * array came from a tokenizer that adds special tokens, the indices have to be
 * over that same array, specials included -- there is no way for this function
 * to detect a caller that mixed the two, except at index 0, and a one-off shift
 * produces a span that slices cleanly and names the wrong word.
 *
 * Every offset here is a JS string index -- a UTF-16 code unit position --
 * because `String.prototype.slice` is, and core's contract (detect/types.ts) is
 * `text === message.slice(start, end)`. Code-point indices are the natural
 * output of a Python tokenizer and are wrong here: measured on the pinned
 * gliner-pii-base tokenizer via Python `tokenizers` 0.22.2, the first emoji of
 * a ZWJ family sequence comes back as (7, 8), and `slice(7, 8)` on the JS
 * string is a lone high surrogate. Convert before calling.
 *
 * Returns `undefined` on anything malformed rather than repairing it. A
 * repaired span is a wrong span that reaches `applyActions`, which rewrites BY
 * SPAN -- so a drifted finding sends a neighbouring word to the vault and
 * leaves the real value in the message. Dropping one uncertain finding is
 * strictly better than that. (There is no non-test caller yet; whoever wires
 * this to the model should count the drops rather than let them vanish.)
 *
 * Two boundary adjustments are made, both of which move a boundary to the edge
 * of the thing it was already inside:
 *
 * 1. TRIMMABLE separators come off both ends. Needed because of what the pinned
 *    tokenizer emits: measured on "call Acme Corp today", its Metaspace
 *    pre-tokenizer reports the token for "Acme" as (4, 9), which slices to
 *    " Acme" -- the preceding separator is inside the token. Untrimmed, every
 *    span that does not start at index 0 carries one. That span still passes
 *    `normalizeFindings`, because the text is sliced; it fails quietly further
 *    down, where `applyActions` calls `vault.mint(conversationId, f.text, ...)`
 *    and one organisation named twice mints two surrogates. What trimming
 *    removes is exactly what the normalizer added: the class is derived from
 *    the fold behaviour above, and only ever runs at a boundary.
 * 2. Trailing COMBINING_MARKs are taken IN. Measured: the pinned tokenizer's
 *    normalizer composes NFKC, so on decomposed "cafe" + U+0301 it reports
 *    (0, 4) and the acute at index 4 belongs to no token. Widening is safe in a
 *    way that widening over a base character would not be -- a Mark modifies
 *    the character before it, which is already inside the span, so this cannot
 *    reach a neighbouring word, let alone a neighbouring entity.
 *
 * The asymmetry is deliberate: a mark at `start` belongs to a base character
 * OUTSIDE the span, so pulling it in would mean pulling in a base character the
 * model did not select, which can change the value. That case is left alone.
 */
export function spanFromTokens(
  text: string,
  tokens: readonly TokenOffset[],
  firstToken: number,
  lastToken: number,
): CharSpan | undefined {
  // Integrality before ordering, as in normalizeFindings. Three of these four
  // conditions state the contract rather than defend it. Measured, by fuzzing
  // each weakened variant against this function over 400k random inputs: drop
  // the integrality check, the `firstToken < 0` check, or the `>=` in the
  // length bound, and the output does not change on a single input -- because
  // `tokens[NaN]`, `tokens[0.5]`, `tokens[-1]` and `tokens[tokens.length]` are
  // all `undefined`, and the guard below rejects that. The fuzz proves those
  // three mutants equivalent; it says nothing about mutants nobody wrote.
  //
  // `lastToken < firstToken` is the one that is load-bearing on its own: with
  // an out-of-order offset array, reversed indices can produce a span that runs
  // FORWARDS -- tokens [(5,9),(0,4)] read as first=1, last=0 gives [0, 9) --
  // and the walk below is skipped entirely because its bounds are crossed.
  if (!Number.isInteger(firstToken) || !Number.isInteger(lastToken)) return undefined;
  if (firstToken < 0 || lastToken < firstToken) return undefined;
  if (lastToken >= tokens.length) return undefined;

  const first = tokens[firstToken];
  const last = tokens[lastToken];
  if (first === undefined || last === undefined) return undefined;

  let start = first.start;
  let end = last.end;
  if (!(start >= 0 && start < end && end <= text.length)) return undefined;

  // Every token in the range -- the two ANCHORS included, which is why there is
  // no separate anchor check -- must be a real, non-empty interval that fits
  // inside the span its endpoints describe.
  //
  // Zero-width rejection is what handles special tokens. Measured on the pinned
  // tokenizer: it wraps every encoding in [CLS] .. [SEP] and reports both at
  // (0, 0), and encoding a PAIR puts a third [SEP], also (0, 0), between the
  // two sequences -- whose offsets then restart at 0. A span anchored on [CLS]
  // would otherwise start at 0, an offset unrelated to where the entity is and
  // slice-consistent enough that normalizeFindings waves it through; a span
  // crossing the middle [SEP] would take its start from one sequence and its
  // end from the other. Containment cannot cover either on its own, because
  // (0, 0) fits inside any span that starts at index 0.
  //
  // Containment is checked at BOTH ends. `token.end <= end` catches a token
  // reaching past the span; `token.start >= start` catches one reaching before
  // it, which is reachable independently -- tokens [(5,9),(0,4),(10,14)] read
  // as 0..2 spans [5, 14) while token 1 sits at [0, 4), entirely outside it.
  //
  // Neither catches a wholesale coordinate mismatch between `tokens` and the
  // indices, in the sense named at the top of this file. Nothing here can.
  for (let i = firstToken; i <= lastToken; i += 1) {
    const token = tokens[i];
    if (token === undefined) return undefined;
    if (!(Number.isInteger(token.start) && Number.isInteger(token.end))) return undefined;
    if (!(token.start < token.end)) return undefined;
    if (!(token.start >= start && token.end <= end)) return undefined;
  }

  while (start < end && TRIMMABLE.test(text.charAt(start))) start += 1;
  while (end > start && TRIMMABLE.test(text.charAt(end - 1))) end -= 1;
  // A range that was nothing but separators is not a finding. Measured: the
  // pinned tokenizer does emit bare U+2581 tokens covering a single space.
  if (start >= end) return undefined;

  // Advanced by whole CODE POINTS, not code units: an astral combining mark
  // (U+1D165 and friends) is two units, and `charAt` would hand `\p{M}` a lone
  // surrogate that never matches. This loop is the one place in the file that
  // has to look at a code point, and it converts back immediately.
  while (end < text.length) {
    const codePoint = text.codePointAt(end);
    if (codePoint === undefined) break;
    const mark = String.fromCodePoint(codePoint);
    if (!COMBINING_MARK.test(mark)) break;
    end += mark.length;
  }

  // Checked on the FINAL boundaries, since those are the ones that get sliced.
  // A boundary inside a surrogate pair yields a string JS is happy to build and
  // normalizeFindings is happy to accept -- it equals the slice -- but it is
  // half a character. Measured: TextEncoder turns a lone high surrogate into
  // ef bf bd, U+FFFD, so the value is already corrupt by the time it is sent,
  // and applyActions would splice a pseudonym between the halves of one glyph.
  if (splitsSurrogatePair(text, start) || splitsSurrogatePair(text, end)) return undefined;

  // Sliced, never assembled from token pieces: subword markers -- "##" for
  // WordPiece, U+2581 for the Metaspace pre-tokenizer this plan pinned -- do
  // not appear in the message, so a joined string fails the fidelity check even
  // when the offsets are perfect. This function never sees the token strings,
  // which is the structural version of the same guarantee.
  return { start, end, text: text.slice(start, end) };
}
