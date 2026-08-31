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
 * Smart punctuation folded to its ASCII form. DEFENSIVE, not corpus-driven.
 *
 * The honest provenance: this table anticipates a model rewriting punctuation
 * while quoting, and no model in the Plan 5 probe corpus ever did. MEASURED by
 * scanning all 13 `out-*.json` probe files for the six characters below: of the
 * 124 model-emitted `quote` fields they contain, 0 carry one, and the corpus
 * holds exactly 2 such characters in total -- two em-dashes inside a paragraph
 * of Ministral's free-form prose, not a quote-back. The probe message itself
 * has no smart punctuation, so the corpus never gave a model the chance.
 *
 * It is kept because folding costs one map lookup and the failure it prevents
 * is a whole finding lost, but nothing here should be read as an observation.
 * Each entry is one code unit in and one code unit out; see `foldUnit` for why
 * that constraint is the load-bearing part.
 */
const PUNCTUATION_FOLD: ReadonlyMap<string, string> = new Map([
  ["‘", "'"], ["’", "'"],
  ["“", '"'], ["”", '"'],
  ["–", "-"], ["—", "-"],
]);

/**
 * Fold one code unit to exactly one code unit.
 *
 * The length constraint is the whole point, and it is not free. MEASURED on
 * this machine (Node 26) by enumerating all 65,536 BMP code units: exactly one
 * has a `toLowerCase()` that is not a single code unit -- U+0130 LATIN CAPITAL
 * LETTER I WITH DOT ABOVE, which lowercases to "i" + U+0307 COMBINING DOT
 * ABOVE. Emitting those two units while recording one map entry desynchronizes
 * `folded` from `map` for the entire remainder of the string: every span
 * resolved past it lands short by one character for each such letter before it.
 * That is a silent mis-location -- it still slices cleanly -- so U+0130 is left
 * uncased. The cost is a missed match if a model rewrites "İ" as "i"; the cost
 * of the alternative is a wrong span nothing downstream detects.
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
 * The `trim` is load-bearing beyond tidiness: it is half of what keeps a
 * candidate from ever ending on a folded space, which `at` would turn into an
 * `end` one character inside the following whitespace run. The other half is
 * the space trim the peel applies after every step. See the guard in `at` for
 * the measured size of that error.
 */
function foldQuote(quote: string): string {
  return buildFoldMap(quote).folded.trim();
}

const isHighSurrogate = (code: number): boolean => code >= 0xd800 && code <= 0xdbff;
const isLowSurrogate = (code: number): boolean => code >= 0xdc00 && code <= 0xdfff;

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
 * Rung 2 will not resolve a candidate spanning fewer than this many words.
 *
 * Measured during Plan 5 feasibility research (not by this module): at a
 * 1,000-character window, one-word capitalized quotes are non-unique 22-51% of
 * the time, two-word 7-38%, three-word about 0%. A two-word candidate that
 * happens to be unique in one message will not be in the next, so accepting it
 * buys a resolution rate that does not survive a change of corpus.
 *
 * Exported because Task 6's judge prompt has to ask the model for a quote of at
 * least this many words, and a second copy of the number in the prompt would be
 * free to drift away from the number the ladder enforces. Same reason
 * `ACTION_RANK` is exported from `packages/core/src/detect/orchestrator.ts`.
 *
 * One honest caveat about what the floor now means. The peel below stops on
 * code points, not word boundaries, so the last of the three words can be a
 * single character: "the deploy pin" and "the deploy p" both pass. The
 * measurement above was taken over whole words, so it bounds the two-word case
 * this floor refuses, not the truncated-third-word case it admits. What the
 * floor still guarantees exactly is that no accepted candidate is as short as a
 * two-word prefix of the same quote.
 */
export const MINIMUM_CANDIDATE_WORDS = 3;

/**
 * Turn a model-supplied quote into character offsets into `message`.
 *
 * The ladder, and why each rung exists:
 *
 *   rung 1 -- the folded quote occurs EXACTLY ONCE. The strong case.
 *   rung 2 -- it does not occur, but a PREFIX of it does, exactly once, after
 *             some number of trailing code points are peeled off. Models append
 *             punctuation, drop trailing words, and perturb the last token;
 *             the first two were observed in the probe corpus.
 *
 * Every rung-2 candidate is a literal prefix of the folded quote, so a
 * resolution is always a contiguous head of what the model actually said.
 * Candidates are tried longest-first and each is a strict prefix of the one
 * before it, so the first unique hit is the longest unique CANDIDATE -- not
 * necessarily the longest unique prefix of the quote, since the peel skips the
 * half-steps inside an astral character.
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
    const lastFolded = foldedIndex + foldedLength - 1;
    // `map.length === folded.length`, and every caller passes a range that
    // `folded.indexOf` returned, so both lookups are in bounds. `!` is the
    // idiom this file uses for that elsewhere (`text[i]!`, `hits[0]!`).
    const start = map[foldedIndex]!;
    const lastIndex = map[lastFolded]!;
    // A folded space maps to its run's FIRST index, so an `end` derived from
    // one is `runStart + 1` -- one character INTO the run, whatever the run's
    // length. MEASURED at run lengths 1, 2, 5 and 20: the error is +1 every
    // time, and the correct `end` is `runStart`, before the run.
    //
    // This never fires. Both callers pass a candidate with no trailing space
    // (foldQuote trims once; the peel re-trims after every step), and it fired
    // 0 times over 200,000 fuzzed inputs while the surrogate check below fired
    // 1,780. Removing it alone changes no test. It is kept because the
    // invariant it depends on is enforced two functions away, and because it
    // is the only check here whose absence shows up as a silent MIS-location
    // rather than a crash: removed together with either trim, a quote resolves
    // to runStart + 1 with its text and offsets in perfect agreement, which is
    // precisely the finding core cannot reject.
    if (folded[lastFolded] === " ") return undefined;
    // Every non-space folded unit came from exactly one original code unit
    // (see foldUnit), so the original character ends one unit past its start.
    const end = lastIndex + 1;
    // Offsets are UTF-16 code unit indices, so a boundary can fall between the
    // halves of an astral character. Half a pair still satisfies core's
    // `text === slice(start, end)` check, so it would be stored as-is. This is
    // the one check here that fires: 1,780 times over 200,000 fuzzed inputs.
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
  if (exact.length > 1) return undefined; // ambiguous: refuse
  if (exact.length === 1) {
    const resolved = at(1, exact[0]!, needle.length);
    if (resolved !== undefined) return resolved;
    // `at` refused. The only thing it refuses in practice is a boundary that
    // splits a surrogate pair, which happens when the quote ends on a lone
    // surrogate matching one HALF of an astral character in the message. That
    // is a reason to keep descending, not to end the ladder -- peeling that
    // half off resolves the quote. Found by differential fuzzing, not by
    // inspection: with `return at(...)` here, such a quote resolved to nothing.
  }

  // Rung 2: peel ONE code point off the tail at a time and take the first
  // candidate that occurs exactly once.
  //
  // Peeling a character rather than a WORD is the security argument. A quote
  // shaped "<label> <secret>" carries the secret in its last word, so a
  // word-level descent sheds the secret in order to shed one appended full
  // stop: the finding then names the credential without covering it, and
  // `applyActions` vaults the label while the key stays in the message.
  //
  // Only trailing SPACES come off, never trailing punctuation, for the same
  // reason -- secrets end in exactly the characters a punctuation strip eats.
  // MEASURED over the six shapes pinned in `rung 2 must not shed the secret's
  // tail` by restoring a greedy trailing non-word strip: it left the secret's
  // last character in the message in four of them ("=", "!", "_", ")"), turned
  // a fifth from a resolution into a refusal, and got only the digit-terminated
  // one right -- which is the shape a strip cannot damage.
  //
  // The peel SHRINKS that hazard, it does not remove it. When the model
  // perturbs a character INSIDE the last word rather than appending one, the
  // longest matching prefix still stops short: for "...AKIAIOSFODNN7EXAMPLF"
  // against a message reading "...AKIAIOSFODNN7EXAMPLE" it recovers 19 of the
  // credential's 20 characters and leaves the last one in the message. That is
  // an improvement on the 0 of 20 a word-level descent recovers, not a cure,
  // and it is pinned as such by a test.
  //
  // Each candidate is a strict prefix of the one before it, and every
  // occurrence of a string is an occurrence of its prefixes, so occurrence
  // counts are monotonically non-decreasing as candidates shorten. That is what
  // makes the ambiguity break below sound.
  let end = needle.length;
  let previous = "";
  while (end > 0) {
    // Step by code POINT so an astral character is peeled whole. MEASURED:
    // forcing the width to 1 changed no result over 300,000 fuzzed inputs whose
    // messages are well formed, because the surrogate check in `at` refuses the
    // half-step and the loop continues past it. It matters when the MESSAGE
    // itself carries a lone surrogate, which nothing refuses: over 200,000
    // inputs drawn from such messages, a width of 1 returned a span one code
    // unit longer 787 times, ending on the message's lone surrogate -- half of
    // the model's astral character matched against half of nothing.
    const width =
      end >= 2 &&
      isLowSurrogate(needle.charCodeAt(end - 1)) &&
      isHighSurrogate(needle.charCodeAt(end - 2))
        ? 2
        : 1;
    end -= width;
    const candidate = needle.slice(0, end).replace(/ +$/u, "");
    if (candidate === previous) continue;
    previous = candidate;
    if (wordEndOffsets(candidate).length < MINIMUM_CANDIDATE_WORDS) break;
    const hits = occurrences(candidate);
    if (hits.length === 1) {
      const resolved = at(2, hits[0]!, candidate.length);
      if (resolved !== undefined) return resolved;
      // Refused, for the same surrogate reason as rung 1. Keep peeling. This
      // fired 0 times over 200,000 fuzzed inputs; the shape that does reach it,
      // found by construction rather than by fuzzing, is a quote ending in two
      // consecutive lone high surrogates. Kept because the cost of being wrong
      // about that is a resolvable quote silently refused.
      continue;
    }
    // Once a candidate is ambiguous no shorter one can be unique, by the
    // monotonicity noted above. MEASURED: swapping this for `continue` changed
    // no result over 300,000 fuzzed inputs, 127,646 of which reach this loop.
    if (hits.length > 1) break;
  }
  return undefined;
}
