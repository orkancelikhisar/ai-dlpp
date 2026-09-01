/**
 * Span recovery: the model returns TEXT, core demands OFFSETS.
 *
 * Core throws when `text !== message.slice(start, end)`
 * (`packages/core/src/detect/orchestrator.ts:98`). That check is satisfied here
 * by construction -- `text` IS the slice -- so it catches an incoherent finding
 * and cannot catch a MIS-LOCATED one. A span pointing at the wrong words slices
 * cleanly, passes core, and reaches `applyActions`, which rewrites by offset:
 * the neighbouring word goes to the vault and the real confidential value stays
 * in the message. Everything below exists to make mis-location impossible
 * rather than merely unlikely, and to REPORT how strong each resolution was.
 *
 * ## Two spans, because locating and acting want opposite things
 *
 * This module used to recover ONE span from ONE quote, and that span was both
 * the evidence for a finding and the range `applyActions` rewrites. Those two
 * jobs pull in opposite directions and the conflict was measured, not argued:
 *
 * - LOCATING wants a long quote. Uniqueness is the only thing rung 1 tests, and
 *   Plan 5's feasibility research put one-word capitalized quotes non-unique
 *   22-51% of the time at a 1,000-character window, two-word 7-38%, three-word
 *   about 0%. That measurement is why `MINIMUM_CANDIDATE_WORDS` is 3.
 * - ACTING wants a short span. COUNTED HERE over every `pred:` finding in
 *   `runs/slate-p-fin-02.*.jsonl` (208 records, 16 arms), which was taken under
 *   the one-span ask: 13 such findings, and FIVE of them span the entire
 *   message -- `pos-client-name-prose` [0,74) of 74 on both
 *   `tier2-Phi-4-mini-instruct-q4f16_1-MLC` and its `tier2only-` twin, and
 *   `pos-pan-prose` [0,48) of 48 on three more arms. The adjudicated gold span
 *   for the first is [43,59), "Tamarind Grocers", 16 characters of 74.
 *
 *   That is not only a scoring problem. The action on
 *   `pred:client-relationship-disclosure` is `redact`, so `applyActions` on
 *   [0,74) leaves the message as "[REDACTED:pred:client-relationship-disclosure]"
 *   and on [43,59) leaves "Can you draft a contract renewal email for
 *   [REDACTED:...] before Friday?". The first destroys the message.
 *
 * Neither side can be given up, so the two spans are separated. The model
 * returns both: the enclosing CLAUSE, which the ladder below places in the
 * passage, and the MENTION inside it, which the action covers. The mention is
 * then resolved INSIDE the already-placed clause rather than in the passage --
 * see `resolveMention` for what that buys and, more importantly, for what it
 * does not.
 */

/** A span and the text it slices, in whatever string it was searched for. */
export interface LocatedSpan {
  readonly start: number;
  readonly end: number;
  /** Always `searched.slice(start, end)`. Sliced, never assembled. */
  readonly text: string;
}

export interface ResolvedQuote extends LocatedSpan {
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
 *
 * It gates the EVIDENCE ladder and nothing else. `resolveMention` searches a
 * clause this ladder has already placed, not the passage, and applies no word
 * floor at all -- a mention is one or two words by design, which is precisely
 * the length this floor refuses. Why that is not the same admission twice is
 * argued where the exemption is taken, on `resolveMention`; the short version
 * is that the mention's uniqueness is still TESTED, over a much smaller string,
 * and a mention that fails the test is refused rather than admitted.
 */
export const MINIMUM_CANDIDATE_WORDS = 3;

/**
 * Turn a model-supplied EVIDENCE quote into character offsets into `message`.
 *
 * This places the enclosing clause and nothing else. The span an action
 * REWRITES is `locateFinding`'s answer, not this one -- see the module
 * docblock. A caller that emits this span as `Finding.start/end` is emitting a
 * whole clause for `applyActions` to rewrite, which is the defect the two-span
 * split exists to close.
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
  const search = foldedSearch(message);
  const needle = foldQuote(quote);
  if (needle.length === 0) return undefined;

  const at = (rung: 1 | 2, foldedIndex: number, foldedLength: number): ResolvedQuote | undefined => {
    const span = search.at(foldedIndex, foldedLength);
    return span === undefined ? undefined : { ...span, rung };
  };
  const occurrences = search.occurrences;

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

/**
 * The folded haystack, plus the two operations every resolution here performs
 * over it: count a candidate's occurrences, and turn one occurrence back into
 * ORIGINAL offsets.
 *
 * Extracted so `resolveQuote` and `resolveMention` cannot drift. They descend
 * differently -- one peels, one does not -- but the fold, the space guard and
 * the surrogate guard are the parts that decide whether a span is MIS-LOCATED,
 * and two copies of those would be two chances to get one of them wrong in one
 * place only. The rung is not in here because only the evidence ladder has
 * rungs.
 */
function foldedSearch(haystack: string): {
  occurrences: (candidate: string) => number[];
  at: (foldedIndex: number, foldedLength: number) => LocatedSpan | undefined;
} {
  const { folded, map } = buildFoldMap(haystack);

  const at = (foldedIndex: number, foldedLength: number): LocatedSpan | undefined => {
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
      index < haystack.length &&
      isHighSurrogate(haystack.charCodeAt(index - 1)) &&
      isLowSurrogate(haystack.charCodeAt(index));
    if (splitsPair(start) || splitsPair(end)) return undefined;
    return { start, end, text: haystack.slice(start, end) };
  };

  const occurrences = (candidate: string): number[] => {
    const hits: number[] = [];
    for (let i = folded.indexOf(candidate); i !== -1; i = folded.indexOf(candidate, i + 1)) {
      hits.push(i);
    }
    return hits;
  };

  return { occurrences, at };
}

/**
 * Turn a model-supplied MENTION into offsets INSIDE one already-placed clause.
 *
 * ## Why this is not `resolveQuote` with a smaller haystack
 *
 * Two rules differ, and both differences are deliberate.
 *
 * **No word floor.** `MINIMUM_CANDIDATE_WORDS` exists because Plan 5 measured
 * one- and two-word quotes non-unique 22-51% and 7-38% of the time at a
 * 1,000-character window. A mention is one or two words BY DESIGN -- it is the
 * name, the identifier or the value the policy is about -- so applying that
 * floor here would refuse essentially every mention there is.
 *
 * The brief that proposed this split argued the floor's premise does not carry
 * over, because the haystack is a clause rather than a message. CHECKED HERE
 * against the 13 messages of `corpora/fixtures/smoke.jsonl`, over every
 * whitespace-delimited candidate in them: at one word, 24 of 196 candidates
 * (12.2%) are non-unique in their whole message and 11 of 196 (5.6%) are still
 * non-unique inside a 50-character window centred on them; at two and three
 * words, 0 of 183 and 0 of 170 are non-unique under either. So narrowing the
 * haystack does cut one-word ambiguity, and it does NOT eliminate it.
 *
 * That measurement is also weaker than it sounds and must not be read as a
 * reproduction of Plan 5's: those messages are 48-153 characters long (mean
 * 91), so "the whole message" here is already smaller than the 1,000-character
 * window the 22-51% figure was taken at. This corpus cannot reach that
 * condition at all.
 *
 * **So the floor is not what makes dropping the floor safe.** What makes it
 * safe is that uniqueness is still TESTED -- over the clause instead of the
 * passage -- and that a mention failing the test is REFUSED. The 5.6% residue
 * above lands in that refusal, not in a guess.
 *
 * **No peel.** Rung 2 exists so a long quote survives a perturbed tail. A
 * mention's tail is the value itself, so the same descent shortens the action
 * span into the thing the action has to cover. MEASURED against this module, on
 * the clause "rotate the staging key AKIAIOSFODNN7EXAMPLE now" and the
 * mistyped mention "the staging key AKIAIOSFODNN7EXAMPLF": `resolveQuote`
 * places it at rung 2 as "the staging key AKIAIOSFODNN7EXAMPL", leaving "E now"
 * after the span -- a redaction covering 19 of the credential's 20 characters,
 * with the finding's text and offsets in perfect agreement, so neither core nor
 * `applyActions` can object. `resolveQuote`'s own rung-2 note records that
 * hazard for the evidence span, where the tail is a stray full stop and the
 * secret sits mid-quote; here the two are the same characters.
 *
 * The floor does NOT already prevent this, which is the part worth stating
 * because it is what made the rule look untested: rung 2 stops at
 * `MINIMUM_CANDIDATE_WORDS`, so it never descends on a one- or two-word
 * mention, and every short example agrees with a no-peel rule by accident. The
 * example above is four words, which is where the two rules first disagree --
 * FOUND BY MUTATION, by swapping this function for `resolveQuote` and watching
 * the whole suite stay green.
 *
 * So a mention that does not match exactly (in folded space) is refused, and
 * the cost is stated rather than hidden: a model that appends a full stop to
 * its mention loses that finding, and the arms count it
 * (`unresolvedMentions`).
 *
 * What is unchanged, because it is what stops a MIS-location: the fold, the
 * trailing-space guard and the surrogate-pair guard, all of them the same code
 * `resolveQuote` runs (`foldedSearch`).
 *
 * @param clause the text of the already-placed evidence span, NOT the passage.
 * @returns offsets relative to `clause`, or `undefined` when the mention is
 *   absent from it, occurs more than once in it, or would land on a boundary
 *   splitting a surrogate pair.
 */
export function resolveMention(clause: string, mention: string): LocatedSpan | undefined {
  const search = foldedSearch(clause);
  const needle = foldQuote(mention);
  if (needle.length === 0) return undefined;
  const hits = search.occurrences(needle);
  // Ambiguity is refused, never resolved by picking -- the doctrine
  // `resolveQuote` already applies, and it is not weaker here for being over a
  // shorter string. "Acme ... Acme" inside one clause gives no evidence about
  // which occurrence the model meant, and neither choice protects the message:
  // rewriting one leaves the other standing, in full, verbatim.
  if (hits.length !== 1) return undefined;
  return search.at(hits[0]!, needle.length);
}

/** Both spans of one model finding, plus how strongly each was placed. */
export interface LocatedFinding {
  /**
   * The clause that LOCATED the finding, in the searched passage. Evidence
   * only: no action is applied to it, and nothing downstream stores it.
   */
  readonly evidence: LocatedSpan;
  /**
   * The span an action REWRITES, in the same passage. Always inside `evidence`.
   * This is what a `Finding`'s `start`/`end` carry.
   */
  readonly action: LocatedSpan;
  /**
   * Which rung placed the EVIDENCE. Says nothing about the mention, which has
   * no rungs -- see `resolveMention`.
   */
  readonly rung: 1 | 2;
  /**
   * The model answered that the whole clause IS the operative span, so the
   * action covers the clause.
   *
   * REPORTED rather than silent, and that is the point of the field. Some
   * predicates are genuinely about a whole clause with no extractable entity,
   * and refusing those would drop a real detection; but a model that answers
   * this way for EVERY finding has quietly restored the behaviour this split
   * exists to end, and an arm's counters are the only place that difference can
   * show up. Both arms count it (`wholeClauseMentions`).
   */
  readonly actionIsWholeEvidence: boolean;
}

/**
 * What one model finding refused on, when it did.
 *
 * `"evidence"` and `"mention"` are separate because they mean different things
 * about the model: an unplaceable clause is a quote that is not in the passage,
 * while an unplaceable mention is a model that quoted a clause correctly and
 * then pointed outside it, or at something the clause says twice.
 */
export type SpanRefusal = "evidence" | "mention";

export type LocateResult =
  | { readonly ok: true; readonly at: LocatedFinding }
  | { readonly ok: false; readonly refused: SpanRefusal };

/**
 * Place one model finding's two spans: the clause in the passage, then the
 * mention inside the clause.
 *
 * ONE implementation for both arms, in this module rather than in either of
 * them, for the reason `resolveQuote` is shared: the compiled judge and
 * Approach B must place spans under identical rules or the bake-off is
 * comparing the placement.
 *
 * ## The four rules, and what each one refuses to do
 *
 * 1. **The clause is placed first, in the passage.** If it will not place, the
 *    mention is never consulted: a mention searched for in a passage whose
 *    clause is not there would be a one- or two-word search over the whole
 *    passage, which is exactly the 22-51% ambiguity the evidence ladder's word
 *    floor exists to refuse. Refusing here is what keeps the mention's search
 *    space small in every case rather than in the common one.
 *
 * 2. **`mention === quote` means "no smaller mention exists".** Taken as the
 *    model's explicit answer and short-circuited: the action span IS the
 *    evidence span. A predicate about a whole clause -- `p-fin`'s
 *    `client-relationship-disclosure` is about a name, but a predicate about,
 *    say, an unannounced decision has no extractable entity -- must stay
 *    detectable, and dropping it would lose a real finding for having no noun
 *    to point at. The short-circuit also means the answer survives a rung-2
 *    evidence resolution, where a re-search would fail by construction: the
 *    quote the model wrote is longer than the peeled clause it resolved to, so
 *    it cannot occur inside it.
 *
 *    This is the one rule that could restore the old behaviour, so it is
 *    counted rather than trusted: see `actionIsWholeEvidence`.
 *
 * 3. **A mention that is not inside its own quote is REFUSED, never resolved
 *    against the passage.** Resolving independently is the tempting repair and
 *    it is the wrong one twice over. It reopens the ambiguity window rule 1
 *    closes -- a one-word search over the whole passage -- and it acts on a
 *    self-contradictory answer: the model was asked for the smallest part OF
 *    THE CLAUSE IT JUST QUOTED, and pointed somewhere else. Fail closed; the
 *    cost is one finding, and the arms count it.
 *
 * 4. **A mention that occurs more than once inside the clause is REFUSED.**
 *    `resolveMention` owns that rule; see it for why picking one is not merely
 *    a guess but a guess that cannot help.
 */
export function locateFinding(passage: string, quote: string, mention: string): LocateResult {
  const evidence = resolveQuote(passage, quote);
  if (evidence === undefined) return { ok: false, refused: "evidence" };
  const evidenceSpan: LocatedSpan = {
    start: evidence.start,
    end: evidence.end,
    text: evidence.text,
  };
  if (mention === quote) {
    return {
      ok: true,
      at: {
        evidence: evidenceSpan,
        action: evidenceSpan,
        rung: evidence.rung,
        actionIsWholeEvidence: true,
      },
    };
  }
  const inner = resolveMention(evidence.text, mention);
  if (inner === undefined) return { ok: false, refused: "mention" };
  const start = evidence.start + inner.start;
  const end = evidence.start + inner.end;
  return {
    ok: true,
    at: {
      evidence: evidenceSpan,
      // Sliced from the PASSAGE, not carried over from the clause's own slice.
      // The two are equal by construction -- `evidence.text` is a slice of
      // `passage` -- and slicing the string the offsets index is the discipline
      // that makes core's `text === slice(start, end)` check true for the
      // reason it is meant to be true, rather than by a chain of equalities.
      action: { start, end, text: passage.slice(start, end) },
      rung: evidence.rung,
      // By SPAN, not by string. A mention differing from the quote only in case
      // or in whitespace is not `=== quote` and still resolves to the whole
      // clause, and calling that anything but a whole-clause answer would put a
      // false 0 in the counter that watches for a model which never narrows.
      actionIsWholeEvidence: start === evidence.start && end === evidence.end,
    },
  };
}
