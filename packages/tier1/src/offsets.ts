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
 * WHAT IT DOES ON THE PATH THAT NOW EXISTS. Task 9 established that the model
 * is indexed by WORDS, not by subword tokens, and that the offsets come from
 * the regex splitter in words.ts rather than from the tokenizer -- which has
 * no offsets API and never had one. A regex word cannot begin or end with a
 * separator PART-WAY, so the trim can no longer bite into a word. Measured
 * over 272,842 words from splitWords: 11,674 of them have a TRIMMABLE first or
 * last character and every one of those is a ONE-code-point word -- U+200B,
 * U+200C, U+200D or U+FEFF, which the Python splitter treats as words in their
 * own right. So on the word path this loop only ever discards a whole
 * zero-width format word sitting at a span's end, and a span that was nothing
 * but such words is rejected below. The derivation that follows is the
 * measurement the class came from, and it is what makes that safe.
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
 * `tokens` and the two indices MUST be in the same coordinate system. The one
 * caller this plan builds is Task 9's: `splitWords` produces the array and the
 * model's logits are indexed by the same WORDS, after `encodeWords` has
 * dropped the ones that tokenise to nothing from the array and the slot
 * numbering together. That drop is the coordinate system; get it wrong and a
 * one-off shift produces a span that slices cleanly and names the wrong word,
 * which nothing in this function can detect.
 *
 * (The array no longer carries special tokens at all -- `encodeWords` returns
 * only real words -- so the `[CLS]`/`[SEP]` reasoning below describes a hazard
 * this path does not reach. The guard it justifies stays, because it is also
 * what rejects a malformed offset array.)
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
 * 1. TRIMMABLE separators come off both ends. On the word path this is what
 *    keeps a zero-width format word out of a span's ends: measured, `splitWords`
 *    emits U+200B, U+200C, U+200D and U+FEFF as words of their own, because
 *    Python's `\s` -- which the model's splitter used -- holds none of them.
 *    An untrimmed span starting on one would hand `applyActions` a value that
 *    differs invisibly from the same name written without it, and
 *    `vault.mint(conversationId, f.text, ...)` would issue two surrogates for
 *    one organisation. It was originally needed for a different reason, which
 *    still holds where it applies: measured on "call Acme Corp today", the
 *    pinned tokenizer's Metaspace pre-tokenizer reports the token for "Acme" as
 *    (4, 9), which slices to " Acme" -- the preceding separator is inside the
 *    token. Nothing feeds this function token offsets today.
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
  // Zero-width rejection. `splitWords` cannot produce a zero-width word -- both
  // branches of its pattern need at least one character -- so on the word path
  // this rejects a corrupted array rather than a legitimate one. It was written
  // for the special tokens of the abandoned subword path, and the measurement
  // stands: the pinned tokenizer wraps every encoding in [CLS] .. [SEP] and
  // reports both at (0, 0), and a PAIR encode puts a third [SEP], also (0, 0),
  // between two sequences whose offsets restart at 0. Containment cannot cover
  // that on its own, because (0, 0) fits inside any span starting at index 0.
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
