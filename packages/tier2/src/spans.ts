/**
 * Span recovery: the model returns a QUOTE, core demands OFFSETS.
 *
 * Core throws when `text !== message.slice(start, end)`
 * (`packages/core/src/detect/orchestrator.ts:98`). That check is satisfied here
 * by construction -- `text` IS the slice -- so it catches an incoherent finding
 * and cannot catch a MIS-LOCATED one. A span pointing at the wrong words slices
 * cleanly, passes core, and reaches `applyActions`, which rewrites by offset:
 * the neighbouring word goes to the vault and the real confidential value stays
 * in the message. Everything below exists to make mis-location impossible
 * rather than merely unlikely, and to REPORT how strong each resolution was.
 */

export interface ResolvedQuote {
  readonly start: number;
  readonly end: number;
  /** Always `message.slice(start, end)`. Sliced, never assembled. */
  readonly text: string;
  /** Which rung resolved it. REPORTED per finding -- see the ladder note. */
  readonly rung: 1 | 2;
}

/**
 * Characters models rewrite freely when quoting. Each entry is one code unit in
 * and one code unit out; see `foldUnit` for why that matters.
 */
const PUNCTUATION_FOLD: ReadonlyMap<string, string> = new Map([
  ["‘", "'"], ["’", "'"],
  ["“", '"'], ["”", '"'],
  ["–", "-"], ["—", "-"],
]);

/**
 * Fold one code unit to exactly one code unit.
 *
 * The length constraint is the whole point, and it is not free. Measured here
 * by enumerating all 65,536 BMP code units: exactly one has a `toLowerCase()`
 * that is not a single code unit -- U+0130 LATIN CAPITAL LETTER I WITH DOT
 * ABOVE, which lowercases to "i" + U+0307 COMBINING DOT ABOVE. Emitting those
 * two units while recording one map entry desynchronizes `folded` from `map`
 * for the entire remainder of the string: every span resolved past it lands
 * short by one character for each such letter before it. That is a silent
 * mis-location -- it still slices cleanly -- so U+0130 is left uncased. The cost is a missed match if a model rewrites "İ" as
 * "i"; the cost of the alternative is a wrong span nothing downstream detects.
 *
 * Astral characters arrive here as two separate surrogate halves, since the
 * caller walks UTF-16 code units. `toLowerCase()` on a lone surrogate returns
 * it unchanged, so a pair survives intact -- but it also means an astral
 * character is never casefolded. No model quote in the Plan 5 probe corpus
 * contained one.
 */
function foldUnit(unit: string): string {
  const punctuation = PUNCTUATION_FOLD.get(unit);
  if (punctuation !== undefined) return punctuation;
  const lowered = unit.toLowerCase();
  return lowered.length === 1 ? lowered : unit;
}

/**
 * A fold of `text` that is easier to match against, plus a map from every
 * folded index back to the ORIGINAL index it came from.
 *
 * The map is the whole point. Matching happens in folded space, where a model's
 * reflowed whitespace and altered case still line up; the span is then sliced
 * from the UNTOUCHED original, so the returned text is exactly what the message
 * says. Normalizing the message and slicing THAT would return text the user
 * never wrote.
 *
 * `map.length === folded.length` always, because every iteration pushes at most
 * one unit to each and never one without the other.
 */
export function buildFoldMap(text: string): { folded: string; map: number[] } {
  const out: string[] = [];
  const map: number[] = [];
  let inWhitespace = false;
  for (let i = 0; i < text.length; i += 1) {
    const unit = text[i]!;
    if (/\s/.test(unit)) {
      // A whitespace RUN folds to one space, mapped to the run's first index.
      if (!inWhitespace) {
        out.push(" ");
        map.push(i);
        inWhitespace = true;
      }
      continue;
    }
    inWhitespace = false;
    out.push(foldUnit(unit));
    map.push(i);
  }
  return { folded: out.join(""), map };
}

/**
 * Fold a quote the same way, without needing a map back.
 *
 * The `trim` is load-bearing beyond tidiness. A folded space maps to its RUN's
 * FIRST index, so a candidate ending on one would produce an `end` inside the
 * run instead of after it -- off by the run's length. Trimming here, plus
 * `trimToWordCharacter` on every rung-2 candidate, is what keeps a candidate
 * from ever ending on whitespace.
 */
function foldQuote(quote: string): string {
  return buildFoldMap(quote).folded.trim();
}

const isHighSurrogate = (code: number): boolean => code >= 0xd800 && code <= 0xdbff;
const isLowSurrogate = (code: number): boolean => code >= 0xdc00 && code <= 0xdfff;

/**
 * Drop trailing characters that are neither letters nor digits.
 *
 * Models append punctuation the message does not have. Observed in the Plan 5
 * probe corpus (`webllm-probe/out-e5.json`, `phi["approachB-shaped"]`, three of
 * three calls): Phi-4-mini returned "...out of the staging config?" for a
 * message reading "...out of the staging config before we send anything?" --
 * four trailing words dropped and the sentence's question mark pulled up onto
 * "config". That it came from the APPROACH B arm is the point: B and the
 * compiled pipeline share this ladder, so a ladder that fumbles here makes B
 * lose on offset arithmetic rather than on judgement.
 *
 * Shedding a whole WORD to shed one such character is not a safe substitute.
 * For a quote shaped "<label> <secret>." the last word IS the secret, so the
 * recovered span would name the credential without covering it, and
 * `applyActions` would vault the label and leave the credential in place.
 *
 * Trailing whitespace is removed too, since a space is neither a letter nor a
 * digit -- which is what keeps `end` off a whitespace run.
 *
 * Steps by code POINT, so an astral character is judged whole and a surrogate
 * pair is never half-removed.
 */
function trimToWordCharacter(candidate: string): string {
  let end = candidate.length;
  while (end > 0) {
    const width =
      end >= 2 &&
      isLowSurrogate(candidate.charCodeAt(end - 1)) &&
      isHighSurrogate(candidate.charCodeAt(end - 2))
        ? 2
        : 1;
    if (/^[\p{L}\p{N}]$/u.test(candidate.slice(end - width, end))) break;
    end -= width;
  }
  return candidate.slice(0, end);
}

/** End offsets, within `folded`, of each whitespace-delimited word. */
function wordEndOffsets(folded: string): number[] {
  const ends: number[] = [];
  let inWord = false;
  for (let i = 0; i < folded.length; i += 1) {
    if (folded[i] !== " ") inWord = true;
    else if (inWord) {
      ends.push(i);
      inWord = false;
    }
  }
  if (inWord) ends.push(folded.length);
  return ends;
}

/**
 * Rung 2 will not resolve a candidate shorter than this many words.
 *
 * Measured during Plan 5 feasibility research (not by this module): at a
 * 1,000-character window, one-word capitalized quotes are non-unique 22-51% of
 * the time, two-word 7-38%, three-word about 0%. A two-word candidate that
 * happens to be unique in one message will not be in the next, so accepting it
 * buys a resolution rate that does not survive a change of corpus.
 */
const MINIMUM_CANDIDATE_WORDS = 3;

/**
 * Turn a model-supplied quote into character offsets into `message`.
 *
 * The ladder, and why each rung exists:
 *
 *   rung 1 -- the folded quote occurs EXACTLY ONCE. The strong case.
 *   rung 2 -- it does not occur, but a PREFIX of it does, exactly once, after
 *             trailing punctuation is dropped. Models append punctuation and
 *             drop trailing words; both were observed in the probe corpus.
 *
 * Every rung-2 candidate is a literal prefix of the folded quote, so a
 * resolution is always a contiguous head of what the model actually said.
 * Candidates are tried longest-first, and each is a prefix of the previous one,
 * so the first unique hit is the LONGEST unique prefix.
 *
 * Ambiguity is refused, never resolved by picking. A quote occurring twice
 * gives no evidence about which the model meant, and a guessed span slices
 * cleanly -- so core accepts it and nothing downstream can object.
 *
 * There is deliberately no rung that searches for the quote's individual words,
 * and none that searches anywhere but the head of the quote. That is the rung
 * that would let a model score by accident.
 */
export function resolveQuote(message: string, quote: string): ResolvedQuote | undefined {
  const { folded, map } = buildFoldMap(message);
  const needle = foldQuote(quote);
  if (needle.length === 0) return undefined;

  const at = (rung: 1 | 2, foldedIndex: number, foldedLength: number): ResolvedQuote | undefined => {
    const start = map[foldedIndex];
    const lastIndex = map[foldedIndex + foldedLength - 1];
    if (start === undefined || lastIndex === undefined) return undefined;
    // A folded space maps to its run's FIRST index, so `lastIndex + 1` would
    // land inside the run. Unreachable today -- foldQuote and
    // trimToWordCharacter between them forbid a candidate ending on a space, and
    // removing this changed no result over 544k fuzzed inputs -- it is here so a
    // rung added later fails closed instead of mis-locating.
    if (folded[foldedIndex + foldedLength - 1] === " ") return undefined;
    // Every non-space folded unit came from exactly one original code unit
    // (see foldUnit), so the original character ends one unit past its start.
    const end = lastIndex + 1;
    if (!(start >= 0 && start < end && end <= message.length)) return undefined;
    // Offsets are UTF-16 code unit indices, so a boundary can fall between the
    // halves of an astral character. Half a pair still satisfies core's
    // `text === slice(start, end)` check, so it would be stored as-is.
    const splitsPair = (index: number): boolean =>
      index > 0 &&
      index < message.length &&
      isHighSurrogate(message.charCodeAt(index - 1)) &&
      isLowSurrogate(message.charCodeAt(index));
    if (splitsPair(start) || splitsPair(end)) return undefined;
    return { start, end, text: message.slice(start, end), rung };
  };

  const occurrences = (candidate: string): number[] => {
    const hits: number[] = [];
    for (let i = folded.indexOf(candidate); i !== -1; i = folded.indexOf(candidate, i + 1)) {
      hits.push(i);
    }
    return hits;
  };

  const exact = occurrences(needle);
  if (exact.length === 1) return at(1, exact[0]!, needle.length);
  if (exact.length > 1) return undefined; // ambiguous: refuse

  // Rung 2: longest unique prefix, cut at a word boundary so a candidate never
  // ends mid-word, then stripped of trailing punctuation the message may not
  // have. Both cuts only ever shorten, so a candidate stays a prefix of the
  // folded quote and of every longer candidate.
  const ends = wordEndOffsets(needle);
  let previous = "";
  for (let words = ends.length; words >= MINIMUM_CANDIDATE_WORDS; words -= 1) {
    const candidate = trimToWordCharacter(needle.slice(0, ends[words - 1]!));
    if (candidate === previous) continue;
    previous = candidate;
    if (wordEndOffsets(candidate).length < MINIMUM_CANDIDATE_WORDS) break;
    const hits = occurrences(candidate);
    if (hits.length === 1) return at(2, hits[0]!, candidate.length);
    // Candidates only shrink, and a shorter prefix occurs everywhere a longer
    // one does, so once one is ambiguous no remaining one can be unique. This
    // only stops early: swapping it for `continue` changed no result over
    // 244,160 fuzzed inputs that reach this branch.
    if (hits.length > 1) break;
  }
  return undefined;
}
