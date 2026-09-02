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
 */

export const NGRAM_N = 8;
export const OVERLAP_DROP_THRESHOLD = 0.7;

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

export interface ContaminationReport {
  readonly n: number;
  readonly threshold: number;
  readonly sources: readonly { readonly sourceId: string; readonly examples: number }[];
  readonly taggedSources: readonly { readonly corpusTag: string; readonly examples: number }[];
  readonly itemsChecked: number;
  readonly dropped: readonly ContaminationDrop[];
  /** Items with fewer than `n` tokens: no 8-gram exists, so none was scored. */
  readonly unscoreable: readonly { readonly itemId: string; readonly tokens: number }[];
  /**
   * Self-test EXAMPLES too short to carry an 8-gram, and therefore invisible to
   * this check no matter what an item does.
   *
   * MEASURED over this repository's sources: 86 of the 280 cases in
   * `policies/compiled/p-fin.selftest.json` are under 8 word tokens
   * ("pan_number: MKLPD7264V", "customer_pan=NPZAK8329R"), i.e. 31% of that
   * corpus cannot be matched against at all. An item that copied one of them
   * verbatim would pass. That is a property of n = 8 and not a bug to fix by
   * lowering n -- a 3-gram check over one-line config fragments would drop
   * honest items by the dozen -- so it is reported instead, and it is exactly
   * the hole `selftest.ts`'s own comment asks the `corpusTag` provenance rule to
   * cover.
   */
  readonly examplesUnscoreable: number;
  /** The highest score among items that were kept. The headroom to the threshold. */
  readonly maxScoreKept: number;
  readonly note: string;
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
  const exampleGrams = examples.map((e) => ngrams(tokenize(e.text)));
  const sources = new Map<string, number>();
  const tagged = new Map<string, number>();
  for (const e of examples) {
    sources.set(e.sourceId, (sources.get(e.sourceId) ?? 0) + 1);
    if (e.corpusTag !== undefined) tagged.set(e.corpusTag, (tagged.get(e.corpusTag) ?? 0) + 1);
  }

  const examplesUnscoreable = exampleGrams.filter((g) => g.size === 0).length;
  const dropped: ContaminationDrop[] = [];
  const unscoreable: { itemId: string; tokens: number }[] = [];
  let maxScoreKept = 0;
  for (const item of items) {
    const tokens = tokenize(item.text);
    const itemGrams = ngrams(tokens);
    if (itemGrams.size === 0) {
      unscoreable.push({ itemId: item.id, tokens: tokens.length });
      continue;
    }
    let worst: ContaminationDrop | undefined;
    let best = 0;
    for (let i = 0; i < examples.length; i += 1) {
      const overlap = directionalOverlap(itemGrams, exampleGrams[i]!);
      if (overlap === undefined) continue;
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
    if (worst !== undefined) dropped.push(worst);
    else if (best > maxScoreKept) maxScoreKept = best;
  }

  return {
    n: NGRAM_N,
    threshold,
    sources: [...sources].map(([sourceId, count]) => ({ sourceId, examples: count })).sort((a, b) => a.sourceId.localeCompare(b.sourceId)),
    taggedSources: [...tagged].map(([corpusTag, count]) => ({ corpusTag, examples: count })).sort((a, b) => a.corpusTag.localeCompare(b.corpusTag)),
    itemsChecked: items.length,
    dropped,
    unscoreable,
    examplesUnscoreable,
    maxScoreKept,
    note:
      `score = max(|item n example| / |item|, |item n example| / |example|) over ${NGRAM_N}-grams of ` +
      `lowercased word tokens; strictly greater than ${threshold} is dropped. See the module header ` +
      `for why the maximum and not the item-side containment the spec's wording most directly suggests.`,
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
