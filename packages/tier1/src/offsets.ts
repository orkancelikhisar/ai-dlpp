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
 * JS `\s`. Measured against `String.prototype.trim` on this machine's Node over
 * U+0020 U+0009 U+000A U+000B U+000C U+000D U+00A0 U+1680 U+2000 U+2028 U+2029
 * U+202F U+205F U+3000 U+FEFF: the two agree on every one of them. Also
 * measured: U+200B (zero-width space) and U+200D (zero-width joiner) are NOT in
 * this class, so neither is trimmed -- correct, because both can sit inside a
 * name and removing them would change the value the vault keys on.
 */
const WHITESPACE = /\s/;

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
 * strictly better than that, and the caller counts drops.
 *
 * The single exception is trimming whitespace off the two ends, which is a
 * narrowing rather than a repair, and is needed because of what the pinned
 * tokenizer actually emits. Measured on "call Acme Corp today": its Metaspace
 * pre-tokenizer reports the token for "Acme" as (4, 9), which slices to
 * " Acme" -- the preceding space is inside the token. Untrimmed, every span
 * that does not start at index 0 carries a leading space. That span passes
 * `normalizeFindings`, because the text is sliced; it fails quietly further
 * down, where the pseudonym vault keys on `Finding.text` and mints two
 * different surrogates for one organisation named twice in one message.
 * Trimming can only ever remove whitespace, and no confidential value begins or
 * ends with whitespace, so it cannot leave part of one behind.
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
  // all `undefined`, and the guard below rejects that.
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
  // Containment handles an offset array that is not in offset order: a
  // re-sorted or spliced array, where reading only the endpoints yields a span
  // that slices cleanly and describes a token range that does not exist.
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

  while (start < end && WHITESPACE.test(text.charAt(start))) start += 1;
  while (end > start && WHITESPACE.test(text.charAt(end - 1))) end -= 1;
  // A range that was nothing but whitespace is not a finding. Measured: the
  // pinned tokenizer does emit bare "_" tokens covering a single space.
  if (start >= end) return undefined;

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
