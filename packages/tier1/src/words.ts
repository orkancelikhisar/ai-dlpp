/**
 * The word splitter the tier-1 models were trained behind.
 *
 * GLiNER span indices are WORD indices, not subword-token indices: the
 * reference pre-splits text with a regex, feeds the WORD LIST to the subword
 * tokenizer, and passes a `words_mask` so the graph pools each word's subwords
 * into one vector. Character offsets come from this splitter and only from
 * here -- the tokenizer is never asked where anything is, which is just as well
 * because @huggingface/transformers has no offsets API to ask.
 *
 * Both pinned models say so themselves: `words_splitter_type` is `"whitespace"`
 * in gliner_config.json for gliner-pii-edge and gliner-pii-base alike, which
 * names `WhitespaceTokenSplitter` in gliner/data_processing/tokenizer.py, whose
 * pattern is `\w+(?:[-_]\w+)*|\S` under CPython's `re`.
 *
 * Deliberately free of any tokenizer import: this file is pure, and the tests
 * for it run without the 1.4 GB of weights on disk.
 */

export interface WordSpan {
  /** Exactly `text.slice(start, end)` of the message this came from. */
  readonly text: string;
  /** JS string index -- a UTF-16 code unit position -- inclusive. */
  readonly start: number;
  /** JS string index -- a UTF-16 code unit position -- exclusive. */
  readonly end: number;
}

/**
 * The Python reference pattern, transliterated so that `\w` and `\S` mean what
 * CPython means by them rather than what JavaScript does.
 *
 * Both differences are load-bearing, and I measured both over all 1,114,112
 * code points against CPython 3.9.6 (Unicode 13.0):
 *
 * 1. `\w`. JS `\w` is ASCII-only -- 63 code points against Python's 133,023 --
 *    so `cafe`-with-an-acute would split in two, and every later word index would
 *    shift. `[\p{L}\p{N}_]` under this host's ICU holds 147,597 code points and
 *    is a strict SUPERSET of Python's: zero Python members are missing, and all
 *    14,574 extras are unassigned (`Cn`) at the oracle's Unicode 13, i.e. pure
 *    version skew rather than a disagreement about any real character.
 * 2. `\S`. JS `\s` and Python `\s` are different sets. Python has U+001C-U+001F
 *    and U+0085, which JS lacks; JS has U+FEFF, which Python lacks. Spelled as
 *    below, the negated class matched Python's `\S` on every one of the
 *    1,114,112 code points, in both directions, and consumed whole code points
 *    rather than half a surrogate pair.
 *
 * The `u` flag is what makes both of those true, and it is not optional: I
 * measured the same source without it against the same fuzz set at 43,906 of
 * 174,705 tokens, and 2,815 of 20,000 texts, exact. GLiNER.js's own
 * `/\w+(?:[-_]\w+)*|\S/g` scored 44,855 tokens and 2,890 texts on that set;
 * keeping the `u` flag but writing the separator as JS `\s` scored 122,633 and
 * 11,025. This spelling scored all 174,705 and all 20,000.
 *
 * Alternation order is the reference's, and it matters -- the word branch is
 * greedy and tried first, so `well-known` is one word and not three.
 */
const WORD_PATTERN = /[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\s\x1C-\x1F\x85]|\u{FEFF}/gu;

/**
 * Splits `text` into the words the model is indexed by.
 *
 * `text === message.slice(start, end)` holds BY CONSTRUCTION rather than by
 * repair: `match.index` and the match length are UTF-16 code unit positions in
 * this very string, which is the coordinate system core's finding contract is
 * written in. The old plan for this seam went through the tokenizer's offsets
 * and had to convert from code points; nothing here does.
 *
 * Verified against the reference by running CPython's `re` over the same
 * corpus: 156/156 curated texts (527 tokens) and 20,000/20,000 fuzz texts
 * (174,705 tokens) token-exact, offsets included, with zero slice-fidelity
 * violations. The committed test/fixtures/word-split.json holds 356 of those
 * cases -- every curated one and 200 of the fuzz ones.
 *
 * The returned shape is exactly Task 6's `TokenOffset` plus the text, so the
 * result feeds `spanFromTokens` with no conversion.
 *
 * ONE KNOWN GAP, measured and not reproducible across clients. `\p{L}` and
 * `\p{N}` resolve against the HOST's ICU, and this machine's Node 26 ships
 * Unicode 17.0 against the oracle's 13.0. Fuzzing an alphabet deliberately
 * saturated with the 14,574 code points that differ, parity falls to 1,718 of
 * 20,000 texts -- and every single failure is on a text containing one of
 * them: restricted to the 488 that are not, it is 488/488. The drift can move
 * a word BOUNDARY, which changes segmentation; it cannot corrupt an offset,
 * because the offsets come from this string either way. Slice fidelity held on
 * all 40,156 texts and 404,303 tokens, skewed set included.
 */
export function splitWords(text: string): WordSpan[] {
  const words: WordSpan[] = [];
  // `matchAll` rather than an `exec` loop, because it iterates a CLONE and so
  // cannot leave a lastIndex behind on this module-level literal for the next
  // call to resume from. It is not a cure for one that is already there --
  // measured, a regex parked at lastIndex 4 makes matchAll and a fresh exec
  // loop alike skip the first word -- which is why WORD_PATTERN is
  // module-private and this is its only use. (An exec loop that runs to
  // exhaustion resets lastIndex to 0 on the failing match; one that breaks
  // early does not, and that is the bug this shape removes.)
  for (const match of text.matchAll(WORD_PATTERN)) {
    const start = match.index;
    words.push({ text: match[0], start, end: start + match[0].length });
  }
  return words;
}
