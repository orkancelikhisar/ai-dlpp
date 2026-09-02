/**
 * Spec 6.2's contamination check: "test items with >0.7 8-gram overlap vs any
 * compiler self-test example are dropped."
 *
 * ## What "overlap" had to be decided to mean, and why the literal reading fails
 *
 * The spec names a number and a unit and leaves the ratio open. There are three
 * candidates and they are not close to each other, because the two sides differ
 * in length by about five to one in tokens and twenty to one in 8-grams.
 * MEASURED: the 280 cases in `policies/compiled/p-fin.selftest.json` have a
 * median of 9 word tokens and 2 8-grams; the 154 items this generator emits
 * have a median of 49 tokens and 42 8-grams.
 *
 * Take a concrete pair -- an 11-token self-test case ("Onboarding record shows
 * AWEFT3518Q as the permanent account number on file.", 4 8-grams) embedded
 * verbatim in an item that adds 17 tokens of its own (28 tokens, 21 8-grams).
 * All four of its 8-grams survive the embedding, so the intersection is 4, and
 * the three ratios MEASURED on that pair are:
 *
 * - Jaccard, |A n B| / |A u B| = 4/21 = 0.190. Never fires at 0.7.
 * - Containment in the item, |A n B| / |A| = 4/21 = 0.190. Also never fires.
 *   This is the reading closest to the spec's wording ("the item's overlap vs
 *   an example") and it is the one that would have made the check decorative.
 * - Containment in the EXAMPLE, |A n B| / |B| = 4/4 = 1.000. Dropped, which is
 *   the behaviour the check exists for.
 *
 * So `overlapScore` is the MAXIMUM of the two directional containments. The
 * maximum rather than the example-side alone because contamination is
 * symmetric in principle -- a short corpus item wholly inside a long example
 * would be caught by the item side -- and taking the max costs nothing.
 *
 * This is a deviation from the spec's most literal reading, and it is reported
 * as one. It is not a loosening: max(a, b) >= a, so nothing the literal reading
 * would drop survives here.
 *
 * ## Items too short to score
 *
 * An item with fewer than 8 tokens has no 8-grams, and an empty set has no
 * containment -- 0/0. Such an item is reported in `unscoreable`, NOT scored as
 * 0 and quietly kept. Scoring it 0 would be an assertion that it is
 * uncontaminated, which is precisely what an empty-set division cannot support.
 *
 * ## EXAMPLES too short to score, and why a fixed n was the defect
 *
 * The same arithmetic runs the other way and it is where this check was found
 * broken. A self-test example shorter than 8 tokens has no 8-gram either, so an
 * item that copied one VERBATIM used to pass. MEASURED over this repository's
 * 723 examples: 263 (36.4%) are under 8 word tokens, so more than a third of
 * the corpus the check exists to compare against was invisible to it.
 *
 * The fix is `n = min(NGRAM_N, exampleTokens)`: an example is matched at its own
 * length when it is shorter than 8. It can only ADD drops, and for a precise
 * reason rather than an intuition: for an example of 8 tokens or more nothing
 * changes at all, and for a shorter one the old rule made NO comparison -- an
 * empty gram set has no containment -- so anything the new rule finds is a pair
 * that previously went unjudged. It costs the check nothing on the three
 * committed corpora (MEASURED: zero additional drops on each).
 *
 * `MIN_EXAMPLE_TOKENS` is the floor, and it is not tuned. MEASURED: every
 * example under 3 tokens in this repository is a value rather than phrasing --
 * 114 are two-token `key value` config fragments whose second token is the
 * identifier ("pan_number mklpd7264v", "encoding utf-8"), and the single
 * one-token case is a bare `redis://...` connection string. Matching on one
 * would drop an item for containing one word. Value-level contamination is a different
 * check and it already runs: `leakage.ts` searches the compiler self-test text
 * for every minted span value, and `corpus-v2.test.ts` asserts it finds none.
 * 115 of 723 examples sit under the floor and are reported, not scored.
 *
 * ## Why a zero here is not evidence of no copying, and what is reported instead
 *
 * All three committed corpora score 0.000 -- no item shares a single 8-gram with
 * any example. That number is true and it is nearly uninformative, for a reason
 * that is arithmetic rather than empirical: a short example's identifier sits in
 * the middle of it, so EVERY 8-gram window of that example contains the
 * identifier, and an 8-gram match therefore requires the item to reproduce a
 * self-test identifier verbatim -- which `leakage.ts` separately proves it never
 * does. MEASURED: of the 460 examples that have at least one 8-gram, 300 (65.2%)
 * have no 8-gram free of a digit-bearing token.
 *
 * So the copying that IS present is invisible to the ratio. MEASURED
 * independently of this module, by longest common contiguous token run over
 * every (item, example) pair: `corpora/generated/injection-p-fin-v2.jsonl`
 * shares a 5-token run with `policies/compiled/p-fin.selftest.json` on 5 of its
 * 189 items, over exactly two phrases -- "the servicing console shows cif" and
 * "the permanent account number on". Both are sentence stems the corpus
 * generator and the compiler's self-test generator arrived at independently or
 * copied; spec 6.2 asks the two generators to use "different prompt templates
 * and seeds", and a shared 5-token stem is the closest thing to a violation of
 * that this repository has.
 *
 * `phraseOverlap` reports that distribution in full. It is REPORTED and not
 * dropped on: 5 shared tokens out of a 75-token item is not ">0.7 8-gram
 * overlap" under any reading, and inventing a run-length gate to make the
 * number look acted-on would be tuning the check to the data it was pointed at.
  */

export const NGRAM_N = 8;
export const OVERLAP_DROP_THRESHOLD = 0.7;
/**
 * An example shorter than this is a value, not phrasing, and is reported as
 * unscoreable instead of matched. See the module header for the measurement.
 */
export const MIN_EXAMPLE_TOKENS = 3;
/**
 * Longest shared run `phraseOverlap` will look for. A bound on the work, not a
 * judgement: no example in this repository exceeds 14 tokens, so nothing is
 * truncated by it today.
 */
export const MAX_PHRASE_RUN_TOKENS = 24;

/**
 * Word tokens, lowercased, punctuation stripped from the edges. Deliberately
 * crude: the check is about copied phrasing, and a tokenizer that preserved
 * case or punctuation would let a self-test example survive by being requoted.
 */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9@._+/:-]+/)
    .map((t) => t.replace(/^[._:/-]+|[._:/-]+$/g, ""))
    .filter((t) => t.length > 0);
}

export function ngrams(tokens: readonly string[], n: number = NGRAM_N): Set<string> {
  const out = new Set<string>();
  for (let i = 0; i + n <= tokens.length; i += 1) out.add(tokens.slice(i, i + n).join(" "));
  return out;
}

/** |a n b| / |a|. Returns undefined for an empty `a`, which has no containment. */
export function containment(a: ReadonlySet<string>, b: ReadonlySet<string>): number | undefined {
  if (a.size === 0) return undefined;
  let hits = 0;
  for (const g of a) if (b.has(g)) hits += 1;
  return hits / a.size;
}

export type OverlapDirection = "item-in-example" | "example-in-item";

/**
 * The score and which containment produced it. ONE definition of the maximum,
 * used by `overlapScore` and by `checkContamination` alike -- a second copy
 * inside the loop is how a "max" quietly becomes a one-sided ratio in the code
 * that runs while the exported helper keeps testing clean. (Found by mutation:
 * an earlier revision had exactly that split, and turning `overlapScore` into
 * the item-side containment killed one unit test and left the pipeline intact.)
 */
export function directionalOverlap(
  a: ReadonlySet<string>,
  b: ReadonlySet<string>,
): { score: number; direction: OverlapDirection } | undefined {
  const ab = containment(a, b);
  const ba = containment(b, a);
  if (ab === undefined || ba === undefined) return undefined;
  return ab >= ba ? { score: ab, direction: "item-in-example" } : { score: ba, direction: "example-in-item" };
}

/** max of the two directional containments; undefined when either side is empty. */
export function overlapScore(a: ReadonlySet<string>, b: ReadonlySet<string>): number | undefined {
  return directionalOverlap(a, b)?.score;
}

export interface SelfTestExample {
  /** Which file it came from, so a drop is traceable to a source. */
  readonly sourceId: string;
  readonly index: number;
  readonly text: string;
  /**
   * `selftest.ts`'s `CORPUS_TAG`, when the source carries one. Its own comment
   * asks Plan 7 to exclude by provenance as well as by n-gram, "which catches
   * the case n-gram overlap misses". `taggedSources` in the report is that
   * exclusion made visible: this corpus draws no text from a tagged source, so
   * the provenance rule has nothing to exclude, and saying so is more useful
   * than a silent zero.
   */
  readonly corpusTag?: string;
}

export interface ContaminationDrop {
  readonly itemId: string;
  readonly score: number;
  /** Which containment produced the score. Reported because they differ hugely. */
  readonly direction: OverlapDirection;
  readonly sourceId: string;
  readonly exampleIndex: number;
  readonly example: string;
}

export interface PhraseRun {
  readonly itemId: string;
  /** Length of the shared run, in word tokens. */
  readonly tokens: number;
  /** The shared run itself, tokenized and rejoined by single spaces. */
  readonly phrase: string;
  readonly sourceId: string;
  readonly exampleIndex: number;
}

/**
 * The overlap the RATIO cannot see: longest contiguous run of word tokens each
 * item shares with any example, with no n and no threshold anywhere in it.
 *
 * It exists because `maxScoreKept: 0` was being read as "these corpora share no
 * phrasing with the compiler's self-test corpus", and that reading is false.
 * See the module header for the arithmetic that makes the ratio blind here.
 */
export interface PhraseOverlapReport {
  /** Longest run shared by any item with any example. 0 when nothing is shared. */
  readonly maxRunTokens: number;
  /**
   * How many items have each longest-run length. An array and not an object
   * keyed by number: `JSON.stringify` hoists integer-like keys to the front in
   * numeric order whatever the insertion order was, and every other accumulated
   * key set in this pipeline is a sorted id for exactly that reason.
   */
  readonly histogram: readonly { readonly tokens: number; readonly items: number }[];
  /** Every item tied at `maxRunTokens`. Not a top-k: the cut is the maximum itself. */
  readonly worst: readonly PhraseRun[];
  readonly note: string;
}

export interface ContaminationReport {
  readonly n: number;
  readonly threshold: number;
  /** Below this an example is a value, not phrasing, and is not scored at all. */
  readonly minExampleTokens: number;
  readonly sources: readonly { readonly sourceId: string; readonly examples: number }[];
  readonly taggedSources: readonly { readonly corpusTag: string; readonly examples: number }[];
  readonly itemsChecked: number;
  /** Items that were scored and survived. A `maxScoreKept` of 0 with 0 here is vacuous. */
  readonly itemsKept: number;
  readonly dropped: readonly ContaminationDrop[];
  /** Items with fewer than `n` tokens: no 8-gram exists, so none was scored. */
  readonly unscoreable: readonly { readonly itemId: string; readonly tokens: number }[];
  /**
   * Self-test EXAMPLES below `minExampleTokens`, which are not matched against
   * at all. MEASURED over this repository's 723 examples: 115, every one of
   * them a `key value` config fragment. Value-level contamination against the
   * same sources is `leakage.ts`'s job and it runs.
   *
   * This number used to be 263 -- every example under 8 tokens -- because the
   * check scored every example at a fixed n = 8. That was the defect; see the
   * module header.
   */
  readonly examplesUnscoreable: number;
  /**
   * Examples scored at their own length because they are shorter than `n`.
   * These are exactly the ones a fixed n = 8 could not see.
   */
  readonly examplesScoredAtOwnLength: number;
  /**
   * Of the examples that carry at least one full-length `n`-gram, how many have
   * NO `n`-gram free of a digit-bearing token. Such an example can only be
   * matched by an item that reproduces its identifier verbatim, so it bounds
   * what a score of 0 is worth. MEASURED here: 300 of 460.
   */
  readonly examplesMatchableOnlyThroughTheirOwnIdentifier: number;
  /** The highest score among items that were kept. The headroom to the threshold. */
  readonly maxScoreKept: number;
  readonly phraseOverlap: PhraseOverlapReport;
  readonly note: string;
}

/**
 * Grams of an example, at `min(n, tokens.length)` so that an example shorter
 * than `n` is matched at its own length instead of not at all. Returns an empty
 * set below `MIN_EXAMPLE_TOKENS`, which is the one case that stays unscoreable.
 */
export function exampleNgrams(tokens: readonly string[], n: number = NGRAM_N): Set<string> {
  if (tokens.length < MIN_EXAMPLE_TOKENS) return new Set();
  return ngrams(tokens, Math.min(n, tokens.length));
}

/**
 * Every contiguous token run of every example, indexed by run length, so the
 * per-item scan below is a map lookup and not a rescan of the example set.
 * Built once per `checkContamination` call: rebuilding it inside the item loop
 * made the whole test suite visibly slower for no change in the answer.
 */
export function buildRunIndex(
  exampleTokens: readonly (readonly string[])[],
  max: number = MAX_PHRASE_RUN_TOKENS,
): readonly ReadonlyMap<string, number>[] {
  const ceiling = Math.min(max, Math.max(0, ...exampleTokens.map((t) => t.length)));
  const index: Map<string, number>[] = [];
  for (let k = 1; k <= ceiling; k += 1) {
    const m = new Map<string, number>();
    for (let ei = 0; ei < exampleTokens.length; ei += 1) {
      const t = exampleTokens[ei]!;
      for (let i = 0; i + k <= t.length; i += 1) {
        const gram = t.slice(i, i + k).join(" ");
        if (!m.has(gram)) m.set(gram, ei);
      }
    }
    index.push(m);
  }
  return index;
}

/**
 * Longest contiguous token run this item shares with any example, and which
 * example. Scans from the longest candidate down and stops at the first hit, so
 * the answer is the maximum and not a sample of it.
 */
export function longestSharedRun(
  index: readonly ReadonlyMap<string, number>[],
  itemTokens: readonly string[],
): { tokens: number; phrase: string; exampleIndex: number } | undefined {
  for (let k = Math.min(index.length, itemTokens.length); k >= 1; k -= 1) {
    const m = index[k - 1]!;
    for (let i = 0; i + k <= itemTokens.length; i += 1) {
      const gram = itemTokens.slice(i, i + k).join(" ");
      const ei = m.get(gram);
      if (ei !== undefined) return { tokens: k, phrase: gram, exampleIndex: ei };
    }
  }
  return undefined;
}

/**
 * Scores every item against every example and reports the drops. Does not
 * mutate anything: the caller drops, so the generator can record both the
 * corpus and the reason an id is missing from it.
 */
export function checkContamination(
  items: readonly { readonly id: string; readonly text: string }[],
  examples: readonly SelfTestExample[],
  threshold: number = OVERLAP_DROP_THRESHOLD,
): ContaminationReport {
  const exampleTokens = examples.map((e) => tokenize(e.text));
  const exampleGrams = exampleTokens.map((t) => exampleNgrams(t));
  const runIndex = buildRunIndex(exampleTokens);
  const sources = new Map<string, number>();
  const tagged = new Map<string, number>();
  for (const e of examples) {
    sources.set(e.sourceId, (sources.get(e.sourceId) ?? 0) + 1);
    if (e.corpusTag !== undefined) tagged.set(e.corpusTag, (tagged.get(e.corpusTag) ?? 0) + 1);
  }

  const examplesUnscoreable = exampleGrams.filter((g) => g.size === 0).length;
  const examplesScoredAtOwnLength = exampleTokens.filter(
    (t) => t.length >= MIN_EXAMPLE_TOKENS && t.length < NGRAM_N,
  ).length;
  // The reach bound: an example whose every full-length gram carries a digit
  // can only be matched by an item that reproduces its identifier.
  let examplesMatchableOnlyThroughTheirOwnIdentifier = 0;
  for (const t of exampleTokens) {
    if (t.length < NGRAM_N) continue;
    const full = ngrams(t);
    if (full.size === 0) continue;
    if (![...full].some((g) => !/[0-9]/.test(g))) examplesMatchableOnlyThroughTheirOwnIdentifier += 1;
  }

  // Grams of an example are cut at the example's own length when it is short,
  // so the ITEM's grams have to be cut at the same length to be comparable at
  // all -- a 6-gram string is never equal to an 8-gram string. Found by the
  // verbatim-copy test in corpus-contamination.test.ts, which reported zero
  // drops until the item side was cut per pair.
  const exampleN = exampleTokens.map((t) => (t.length < MIN_EXAMPLE_TOKENS ? 0 : Math.min(NGRAM_N, t.length)));

  const dropped: ContaminationDrop[] = [];
  const unscoreable: { itemId: string; tokens: number }[] = [];
  const runs: PhraseRun[] = [];
  let maxScoreKept = 0;
  let itemsKept = 0;
  for (const item of items) {
    const tokens = tokenize(item.text);
    const run = longestSharedRun(runIndex, tokens);
    if (run !== undefined) {
      const e = examples[run.exampleIndex]!;
      runs.push({
        itemId: item.id,
        tokens: run.tokens,
        phrase: run.phrase,
        sourceId: e.sourceId,
        exampleIndex: e.index,
      });
    }

    const itemGramsByN = new Map<number, Set<string>>();
    let worst: ContaminationDrop | undefined;
    let best = 0;
    let comparisons = 0;
    for (let i = 0; i < examples.length; i += 1) {
      const n = exampleN[i]!;
      if (n === 0) continue;
      let itemGrams = itemGramsByN.get(n);
      if (itemGrams === undefined) {
        itemGrams = ngrams(tokens, n);
        itemGramsByN.set(n, itemGrams);
      }
      const overlap = directionalOverlap(itemGrams, exampleGrams[i]!);
      if (overlap === undefined) continue;
      comparisons += 1;
      if (overlap.score > best) best = overlap.score;
      if (overlap.score > threshold && (worst === undefined || overlap.score > worst.score)) {
        const e = examples[i]!;
        worst = {
          itemId: item.id,
          score: overlap.score,
          direction: overlap.direction,
          sourceId: e.sourceId,
          exampleIndex: e.index,
          example: e.text,
        };
      }
    }
    // No comparable example: 0/0 again, and reported rather than scored. An
    // item shorter than every scoreable example lands here.
    if (comparisons === 0) {
      unscoreable.push({ itemId: item.id, tokens: tokens.length });
      continue;
    }
    if (worst !== undefined) dropped.push(worst);
    else {
      itemsKept += 1;
      if (best > maxScoreKept) maxScoreKept = best;
    }
  }

  const maxRunTokens = Math.max(0, ...runs.map((r) => r.tokens));
  const byLength = new Map<number, number>();
  for (const r of runs) byLength.set(r.tokens, (byLength.get(r.tokens) ?? 0) + 1);

  return {
    n: NGRAM_N,
    threshold,
    minExampleTokens: MIN_EXAMPLE_TOKENS,
    sources: [...sources].map(([sourceId, count]) => ({ sourceId, examples: count })).sort((a, b) => a.sourceId.localeCompare(b.sourceId)),
    taggedSources: [...tagged].map(([corpusTag, count]) => ({ corpusTag, examples: count })).sort((a, b) => a.corpusTag.localeCompare(b.corpusTag)),
    itemsChecked: items.length,
    itemsKept,
    dropped,
    unscoreable,
    examplesUnscoreable,
    examplesScoredAtOwnLength,
    examplesMatchableOnlyThroughTheirOwnIdentifier,
    maxScoreKept,
    phraseOverlap: {
      maxRunTokens,
      histogram: [...byLength].sort((a, b) => a[0] - b[0]).map(([tokens, count]) => ({ tokens, items: count })),
      worst: runs
        .filter((r) => r.tokens === maxRunTokens && maxRunTokens > 0)
        .sort((a, b) => a.itemId.localeCompare(b.itemId)),
      note:
        "longest contiguous run of word tokens each item shares with any example. No n and no " +
        "threshold: it is the measurement the >0.7 8-gram ratio cannot make, reported because a " +
        "maxScoreKept of 0 was being read as 'shares no phrasing', which it does not mean.",
    },
    note:
      `score = max(|item n example| / |item|, |item n example| / |example|) over ${NGRAM_N}-grams of ` +
      `lowercased word tokens, an example shorter than ${NGRAM_N} tokens matched at its own length ` +
      `and one shorter than ${MIN_EXAMPLE_TOKENS} not matched at all; strictly greater than ` +
      `${threshold} is dropped. See the module header for why the maximum and not the item-side ` +
      `containment the spec's wording most directly suggests, and for why a score of 0 here is a ` +
      `weaker statement than it looks -- read phraseOverlap beside it.`,
  };
}

// -- loading the compiler's self-test examples ------------------------------

/** `policies/compiled/<policy>.selftest.json`: the committed self-test corpus. */
export function selfTestExamplesFromCompiledCorpus(json: string, sourceId: string): SelfTestExample[] {
  const parsed: unknown = JSON.parse(json);
  if (!Array.isArray(parsed)) throw new Error(`${sourceId} is not a JSON array of self-test cases`);
  return parsed.map((raw, index) => {
    const c = raw as { text?: unknown; corpusTag?: unknown };
    if (typeof c.text !== "string") throw new Error(`${sourceId}[${index}] has no string "text"`);
    return {
      sourceId,
      index,
      text: c.text,
      ...(typeof c.corpusTag === "string" ? { corpusTag: c.corpusTag } : {}),
    };
  });
}

/**
 * `packages/compiler/test/fixtures/llm/SelfTestCases.*.json`: a recorded model
 * response, `{positives: string[], negatives: string[]}`. Both lists are
 * examples for this purpose -- a hard negative is generated text just as a
 * positive is, and an item that copied one is contaminated either way.
 */
export function selfTestExamplesFromLlmFixture(json: string, sourceId: string): SelfTestExample[] {
  const parsed = JSON.parse(json) as { positives?: unknown; negatives?: unknown };
  const out: SelfTestExample[] = [];
  for (const key of ["positives", "negatives"] as const) {
    const list = parsed[key];
    if (!Array.isArray(list)) throw new Error(`${sourceId} has no "${key}" array`);
    for (const text of list) {
      if (typeof text !== "string") throw new Error(`${sourceId}.${key} contains a non-string`);
      out.push({ sourceId: `${sourceId}#${key}`, index: out.length, text });
    }
  }
  return out;
}
