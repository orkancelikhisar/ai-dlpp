import type { PolicyIr } from "@sih/core";
import type { CorpusItem } from "../driver/corpus.js";
import { NEG_PREFIX } from "./labels.js";

/**
 * The four leakage measurements, and the reading of the IR that one of them
 * rests on.
 *
 * Every number this module produces is written into the emitted manifest, so
 * the claim "this corpus does not hand the compiled arm the answer" is a
 * regenerable measurement rather than a sentence. The functions are shared by
 * the manifest and by `corpus-v2.test.ts` deliberately: a test that computed
 * its own version of a rate would be checking a second implementation, not the
 * artifact.
 *
 * The `before` figures quoted throughout are measured on
 * `corpora/generated/injection-p-fin-adjudicated-v1.jsonl`, the corpus this
 * round exists to replace, using these same functions.
 */

// -- contextBoost -----------------------------------------------------------

/**
 * The window and match rule `runTier0` ACTUALLY uses, copied from it.
 *
 * `packages/core/src/detect/tier0.ts` declares `const CONTEXT_WINDOW = 40` and
 * its `hasNearbyKeyword` lowercases
 * `text.slice(start - 40, end + 40)` and asks `window.includes(k)`. Three
 * things follow, and every one of them differs from `BOOST_WINDOW` below:
 * the width is 40 rather than 70, the test is SUBSTRING rather than
 * word-boundary, and the window INCLUDES the matched span's own text.
 *
 * The earlier version of this comment asserted that `runTier0` "does not
 * implement contextBoost, so there is no window in the shipping code to copy".
 * That was false -- the mechanism is at tier0.ts:45, :74 and :90 -- and the
 * symmetry number the manifest published was therefore measured at a width and
 * a match rule no arm uses. This constant and `tier0BoostNear` exist so the
 * published number is the one the shipping detector would see.
 *
 * `corpus-v2.test.ts` pins this copy to `runTier0` DIFFERENTIALLY: it builds
 * texts where the two disagree and asserts the confidence bump `runTier0`
 * emits (0.95 against a 0.9 floor) tracks `tier0BoostNear` and not
 * `boostNear`. A copied constant with no test against its original is how the
 * first version of this comment survived.
 */
export const TIER0_BOOST_WINDOW = 40;

/**
 * How far from a span a contextBoost term counts as "near it", under the WIDER
 * reading.
 *
 * 70 characters either side, and the span's own text is EXCLUDED from the
 * window. The exclusion matters for this reading: `CIF 30045512` contains
 * "cif", `KYC 4471200` contains "kyc", and counting those would score a family
 * as boosted because of a substring of the value the family is asking about,
 * which is not a fact about the surrounding text at all.
 *
 * This is NOT what the shipping detector does -- see `TIER0_BOOST_WINDOW` --
 * and it is kept for two reasons rather than one. It is the width the previous
 * corpus's before-figures were measured at, so the before/after pair stays a
 * comparison; and it is the reading that asks about the surrounding PROSE,
 * which is the channel a prompt-reading arm sees. Both are reported. The one
 * to quote when the subject is the compiled arm is the tier-0 one.
 */
export const BOOST_WINDOW = 70;

export interface BoostTerms {
  /** Every contextBoost term in the IR, lowercased, deduplicated, sorted. */
  readonly all: readonly string[];
  /** entityType id -> the terms its own rules declare. */
  readonly byEntityType: Readonly<Record<string, readonly string[]>>;
}

export function boostTerms(ir: PolicyIr): BoostTerms {
  const all = new Set<string>();
  const by: Record<string, Set<string>> = {};
  for (const rule of ir.rules) {
    for (const term of rule.contextBoost ?? []) {
      const t = term.toLowerCase();
      all.add(t);
      (by[rule.entityType] ??= new Set()).add(t);
    }
  }
  const byEntityType: Record<string, readonly string[]> = {};
  for (const k of Object.keys(by).sort()) byEntityType[k] = [...by[k]!].sort();
  return { all: [...all].sort(), byEntityType };
}

/**
 * Word-boundary matching, not substring, with the boundary class written out
 * because it is narrower than "word".
 *
 * The boundary is `[^a-z0-9]`, so "connect" does not fire on "connection" and
 * "account" does not fire on "accounts" -- which is the case this exists for.
 * IT DOES FIRE INSIDE AN UNDERSCORED IDENTIFIER: `_` is neither a letter nor a
 * digit, so "account" matches in `storage_account_key` and "pan" matches in
 * `pan_number:`. An earlier version of this comment named
 * `storage_account_key` as a case the rule excludes. It does not, and the
 * comment was the wrong half of the pair: MEASURED over every (labelled span x
 * boost term) pair in both corpora, widening the class to `[^a-z0-9_]` changes
 * 0 of 12,426 answers on the emitted corpus and 3 of 8,208 on the corpus it
 * replaces -- all three the key `pan_number:` on a kv line above an `in-pan`
 * span, where the key genuinely does signal the value's type and excluding it
 * would be the wrong answer, not the right one. So the class is left as it is
 * and described as it is.
 *
 * The pattern is built per call rather than cached because a shared `/g` regex
 * is stateful across calls, which this repository has been bitten by before
 * (see `orthographicOrgSweep` in `certify.ts`).
 */
function containsTerm(haystack: string, term: string): boolean {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, "i").test(haystack);
}

export function boostNear(
  text: string,
  span: { readonly start: number; readonly end: number },
  terms: readonly string[],
  window = BOOST_WINDOW,
): boolean {
  const before = text.slice(Math.max(0, span.start - window), span.start);
  const after = text.slice(span.end, Math.min(text.length, span.end + window));
  return terms.some((t) => containsTerm(before, t) || containsTerm(after, t));
}

/**
 * `hasNearbyKeyword` from `packages/core/src/detect/tier0.ts`, transcribed.
 *
 * Deliberately a transcription and not an import: `hasNearbyKeyword` is module
 * private, and re-exporting it to be measured would change the shipping surface
 * to suit a measurement. The risk a transcription carries -- that it drifts
 * from its original -- is what the differential test in `corpus-v2.test.ts`
 * exists to catch, by asserting `runTier0`'s own confidence bump agrees with
 * this function on texts where it disagrees with `boostNear`.
 *
 * The keyword must fall ENTIRELY inside the window, so the effective reach is
 * `window` minus the keyword's own length. That is the shipping behaviour, and
 * tier0.ts says so in the same words.
 */
export function tier0BoostNear(
  text: string,
  span: { readonly start: number; readonly end: number },
  terms: readonly string[],
  window = TIER0_BOOST_WINDOW,
): boolean {
  const w = text
    .slice(Math.max(0, span.start - window), Math.min(text.length, span.end + window))
    .toLowerCase();
  return terms.some((t) => w.includes(t));
}

export interface BoostRates {
  readonly window: number;
  /** Which of the two readings produced these rates. Stated, not inferred from the width. */
  readonly matching: string;
  /** Any contextBoost term of any rule, near a gold span. */
  readonly goldSpans: number;
  readonly goldSpansWithBoost: number;
  readonly goldRate: number;
  /** The same, near a confusable span. */
  readonly confusableSpans: number;
  readonly confusableSpansWithBoost: number;
  readonly confusableRate: number;
  readonly delta: number;
  /**
   * The sharper form: a term declared by a rule of the span's OWN entityType,
   * near a gold span; and near a confusable, a term of the entityType that
   * confusable is a near miss for.
   */
  readonly goldOwnTypeRate: number;
  readonly confusablePairedTypeRate: number;
  readonly ownTypeDelta: number;
}

export interface BoostReport {
  readonly terms: number;
  /**
   * The rates under the mechanism `runTier0` implements. THIS IS THE NUMBER TO
   * READ when the question is whether a contextBoost keyword predicts the label
   * for the compiled arm, because it is the only one measured at the width and
   * match rule that arm uses.
   */
  readonly asTier0Reads: BoostRates;
  /**
   * The rates at 70 characters with word-boundary matching and the span's own
   * text excluded. The width the previous corpus's before-figures were taken
   * at, kept so the before/after pair is a comparison rather than two different
   * definitions, and the reading that asks about surrounding PROSE.
   */
  readonly wideWordBoundary: BoostRates;
  /**
   * What the two readings together support. Generated from the rates rather
   * than typed, apart from the two before-corpus figures it sets them
   * against, which are `BEFORE_CORPUS_TIER0_BOOST`.
   */
  readonly verdict: string;
}

function rates(
  items: readonly CorpusItem[],
  terms: BoostTerms,
  pairedType: Readonly<Record<string, string>>,
  near: (text: string, span: { start: number; end: number }, t: readonly string[]) => boolean,
  window: number,
  matching: string,
): BoostRates {
  let gold = 0;
  let goldBoost = 0;
  let goldOwn = 0;
  let neg = 0;
  let negBoost = 0;
  let negPaired = 0;
  for (const item of items) {
    for (const label of labelSpans(item)) {
      const isNear = near(item.text, label, terms.all);
      if (label.type.startsWith(NEG_PREFIX)) {
        neg += 1;
        if (isNear) negBoost += 1;
        const paired = pairedType[label.type];
        if (paired !== undefined && near(item.text, label, terms.byEntityType[paired] ?? [])) negPaired += 1;
      } else {
        gold += 1;
        if (isNear) goldBoost += 1;
        if (near(item.text, label, terms.byEntityType[label.type] ?? [])) goldOwn += 1;
      }
    }
  }
  const goldRate = gold === 0 ? 0 : goldBoost / gold;
  const confusableRate = neg === 0 ? 0 : negBoost / neg;
  const goldOwnTypeRate = gold === 0 ? 0 : goldOwn / gold;
  const confusablePairedTypeRate = neg === 0 ? 0 : negPaired / neg;
  return {
    window,
    matching,
    goldSpans: gold,
    goldSpansWithBoost: goldBoost,
    goldRate,
    confusableSpans: neg,
    confusableSpansWithBoost: negBoost,
    confusableRate,
    delta: goldRate - confusableRate,
    goldOwnTypeRate,
    confusablePairedTypeRate,
    ownTypeDelta: goldOwnTypeRate - confusablePairedTypeRate,
  };
}

/**
 * The corpus this round replaces, at the tier-0 reading: the `asTier0Reads`
 * DELTAS -- a gold rate minus a confusable rate -- that `measureBoost` returns
 * for `corpora/generated/injection-p-fin-adjudicated-v1.jsonl` under the wave-2
 * paired-type map. Both are deltas because the residuals `verdict` sets them
 * against are deltas.
 *
 * Literals rather than a call: that corpus is not loaded here, and loading a
 * superseded artifact inside a measurement of this one would make this
 * manifest depend on it. `corpus-v2.test.ts` recomputes both from that file
 * with the same function and fails if either drifts, and separately pins the
 * own-type figure against a fraction derived from the span counts.
 *
 * The own-type figure published before this constant existed was +0.4352,
 * which is that corpus's `goldOwnTypeRate` -- 47 of its 108 gold spans -- with
 * nothing subtracted from it: a RATE, set against a delta. The delta is
 * 47/108 - 4/108 = 43/108 = +0.3981. Direction and rough magnitude of the
 * comparison are unchanged; the published statistic was the wrong one.
 */
export const BEFORE_CORPUS_TIER0_BOOST = {
  /** Any contextBoost term near a gold span, minus the same near a confusable. */
  delta: 0.4537,
  /** The sharper own-entityType form of that same difference. */
  ownTypeDelta: 0.3981,
} as const;

/**
 * `pairedType` maps a `neg:` label to the entityType it is a near miss for, so
 * the own-type rate has a counterpart on the negative side. Supplied by the
 * caller rather than derived: only the family catalogue knows which pair a
 * confusable belongs to, and guessing it from the id would be a second,
 * fallible claim.
 *
 * Both readings are computed on every call. Reporting only one was the defect:
 * the manifest published a delta of -0.0075 taken at 70 characters with
 * word-boundary matching, and the same corpus reads +0.0753 at the width and
 * match rule `runTier0` uses. A number that moves that much between two
 * defensible definitions has to carry the definition with it.
 */
export function measureBoost(
  items: readonly CorpusItem[],
  ir: PolicyIr,
  pairedType: Readonly<Record<string, string>>,
): BoostReport {
  const terms = boostTerms(ir);
  const asTier0Reads = rates(
    items,
    terms,
    pairedType,
    (text, span, t) => tier0BoostNear(text, span, t),
    TIER0_BOOST_WINDOW,
    "substring, span's own text INCLUDED in the window -- packages/core/src/detect/tier0.ts hasNearbyKeyword",
  );
  const wideWordBoundary = rates(
    items,
    terms,
    pairedType,
    (text, span, t) => boostNear(text, span, t),
    BOOST_WINDOW,
    "word boundary, span's own text EXCLUDED from the window -- this module's boostNear",
  );
  const symmetric = Math.abs(asTier0Reads.delta) < 0.05 && Math.abs(asTier0Reads.ownTypeDelta) < 0.05;
  return {
    terms: terms.all.length,
    asTier0Reads,
    wideWordBoundary,
    verdict: symmetric
      ? "SYMMETRIC under the tier-0 reading: a contextBoost keyword is as likely near a confusable " +
        "span as near a gold one, within 5 points on both the any-term and the own-type form."
      : "NOT SYMMETRIC under the tier-0 reading. A contextBoost keyword sits near " +
        `${(asTier0Reads.goldRate * 100).toFixed(1)}% of gold spans and ` +
        `${(asTier0Reads.confusableRate * 100).toFixed(1)}% of confusable spans (delta ` +
        `${asTier0Reads.delta >= 0 ? "+" : ""}${asTier0Reads.delta.toFixed(4)}), and on the sharper ` +
        `own-entityType form ${(asTier0Reads.goldOwnTypeRate * 100).toFixed(1)}% against ` +
        `${(asTier0Reads.confusablePairedTypeRate * 100).toFixed(1)}% (delta ` +
        `${asTier0Reads.ownTypeDelta >= 0 ? "+" : ""}${asTier0Reads.ownTypeDelta.toFixed(4)}). The ` +
        "residual favours the compiled arm. It is far smaller than the corpus this replaces -- " +
        `which read +${BEFORE_CORPUS_TIER0_BOOST.delta.toFixed(4)} and ` +
        `+${BEFORE_CORPUS_TIER0_BOOST.ownTypeDelta.toFixed(4)} on the same two forms -- and it is not zero, so a ` +
        "per-entityType accuracy number off this corpus carries a keyword advantage of that size " +
        "for any arm that reads contextBoost.",
  };
}

/** Every labelled span on an item -- gold and `neg:` alike -- read out of `meta.injections`. */
export function labelSpans(
  item: CorpusItem,
): readonly { readonly start: number; readonly end: number; readonly text: string; readonly type: string; readonly family: string }[] {
  const raw = (item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[];
  return raw.map((r) => ({
    start: r["start"] as number,
    end: r["end"] as number,
    text: r["text"] as string,
    type: String(r["type"]),
    family: String(r["family"]),
  }));
}

// -- orthography ------------------------------------------------------------

/**
 * The "flag the odd string" detector, written down so the corpus can be scored
 * against it.
 *
 * This is not a proposed system. It is the null hypothesis: a reader that
 * understands nothing, returns every region of a message that LOOKS unlike
 * running prose, and would be indistinguishable from a real detector on a
 * corpus whose injected spans are its only odd-looking tokens. MEASURED on
 * `injection-p-fin-adjudicated-v1`: it returns 95 of that corpus's 108 gold
 * spans, and on 58 of them it returns nothing else in the message -- so on 54%
 * of the gold spans "return the odd string" is a perfect detector, and its
 * precision over the whole corpus is 0.49.
 *
 * The rules are deliberately crude and are listed rather than tuned: a
 * PEM-style dashed block; a token mixing letters and digits; a run of nine or
 * more digits; something with an at-sign between alphanumerics; anything
 * containing "://"; an all-uppercase run of four or more; sixteen or more
 * base64 characters; and a Title-Case bigram, which is what an organisation
 * name looks like in a lowercase chat message.
 */
const ORACLE_TOKEN = /[A-Za-z0-9@._/:+=~-]{3,}/g;
const ORACLE_TITLE_BIGRAM = /\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b/g;
const ORACLE_PEM = /-----BEGIN[^\n]*-----[\s\S]*?-----END[^\n]*-----/g;

function tokenIsOdd(t: string): boolean {
  if (/[0-9]/.test(t) && /[A-Za-z]/.test(t)) return true;
  if (/^[0-9]{9,}$/.test(t)) return true;
  if (/[A-Za-z0-9]@[A-Za-z0-9]/.test(t)) return true;
  if (t.includes("://")) return true;
  if (/^[A-Z]{4,}$/.test(t)) return true;
  if (t.length >= 16 && /^[A-Za-z0-9+/=]+$/.test(t)) return true;
  return false;
}

export interface OracleSpan {
  readonly start: number;
  readonly end: number;
}

export function orthographicOracle(text: string): readonly OracleSpan[] {
  const spans: OracleSpan[] = [];
  for (const m of text.matchAll(ORACLE_PEM)) spans.push({ start: m.index, end: m.index + m[0].length });
  for (const m of text.matchAll(ORACLE_TOKEN)) {
    const start = m.index;
    const end = start + m[0].length;
    if (spans.some((s) => s.start <= start && end <= s.end)) continue;
    if (tokenIsOdd(m[0])) spans.push({ start, end });
  }
  for (const m of text.matchAll(ORACLE_TITLE_BIGRAM)) spans.push({ start: m.index, end: m.index + m[0].length });
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  // Drop anything strictly contained in another hit, so a PEM block counts once
  // rather than once per line.
  return sorted.filter(
    (s) => !sorted.some((o) => o !== s && o.start <= s.start && s.end <= o.end && !(o.start === s.start && o.end === s.end)),
  );
}

function overlaps(a: { start: number; end: number }, b: { start: number; end: number }): boolean {
  return a.start < b.end && b.start < a.end;
}

export interface OrthographyReport {
  readonly goldSpans: number;
  /** Gold spans the oracle returns at all. Its recall numerator. */
  readonly goldSpansFound: number;
  /**
   * Gold spans the oracle FINDS and that are the only region it returns
   * anywhere in that message.
   *
   * KEPT, AND NO LONGER LOAD-BEARING. This was the statistic that certified
   * "the gold span is not the unique orthographic outlier", and it cannot carry
   * that claim: injecting one same-shape distractor per positive drives it to
   * zero BY CONSTRUCTION -- a second oracle hit anywhere in the message makes
   * the gold span not-alone regardless of whether the distractor is as odd as
   * the gold span, as plausible a candidate, or even in the same clause. A
   * corpus can read 0 here and still be solved by an outlier-only reader. The
   * measurements that survive a distractor are `budgetMatched` and
   * `firstHitIsGold` below, and the honest summary is `verdict`.
   *
   * "Finds AND is alone" rather than "is alone", and the difference is not
   * pedantry: an eight-digit account number is not odd enough for this oracle to
   * return, so a message carrying one and nothing else has no oracle spans at
   * all, and counting that as "the gold span is the unique outlier" would credit
   * the corpus for a span the oracle cannot see. MEASURED during this rebuild:
   * three of four apparent survivors were exactly that.
   */
  readonly goldSpansSolvedByOracle: number;
  readonly solvedRate: number;
  readonly oracleSpans: number;
  readonly oracleSpansOnGold: number;
  readonly oraclePrecision: number;
  readonly oracleRecall: number;
  /**
   * The oracle scored the way `score.ts` scores an ARM: the `overlap` rule with
   * maximum one-to-one matching, so two hits on one gold span count once.
   *
   * These are the figures to compare with an arm, and they are not the same as
   * `oraclePrecision` above: that one divides hits-that-touch-gold by hits,
   * which double-counts a gold span covered twice. The gap is small here (97
   * touching hits against 94 matched) and it is the difference between a number
   * an arm could be placed beside and one it could not.
   */
  readonly asScored: {
    readonly rule: string;
    readonly tp: number;
    readonly fp: number;
    readonly fn: number;
    readonly precision: number;
    readonly recall: number;
    readonly f1: number;
  };
  /**
   * The oracle told how many spans to return on each item -- the item's own
   * gold count -- and returning its first that many in document order.
   *
   * Budget-matched rather than unbounded, because the unbounded oracle buys its
   * recall with 212 false positives that no arm operating under a precision
   * constraint would emit, and comparing an arm's F1 with that is comparing two
   * different tasks. The budget is oracle information the arms do not get, so
   * this is an UPPER reference and not a competitor; what makes it worth
   * publishing is that adding a distractor cannot drive it to a flattering
   * value the way `goldSpansSolvedByOracle` can. A distractor only helps here
   * if it actually displaces the gold span in the ranking.
   *
   * Document order rather than a learned or tuned oddness score: it is the one
   * ranking that needs no threshold and cannot be fitted to the corpus. It is
   * also the ranking the corpus is most exposed to, because `generate.ts`
   * places every shape-matched distractor one slot AFTER the span it shadows --
   * see `measurePosition`.
   */
  readonly budgetMatched: {
    readonly budget: string;
    readonly ranking: string;
    readonly tp: number;
    readonly fp: number;
    readonly fn: number;
    readonly precision: number;
    readonly recall: number;
    readonly f1: number;
  };
  /** Items carrying at least one gold span. The denominator of `firstHitIsGoldRate`. */
  readonly itemsWithGold: number;
  /** Items where the oracle's FIRST hit, in document order, lands on a gold span. */
  readonly firstHitIsGold: number;
  readonly firstHitIsGoldRate: number;
  readonly verdict: string;
}

/**
 * Maximum-cardinality one-to-one matching between oracle hits and gold spans on
 * one item, under the `overlap` rule.
 *
 * The same shape `apps/eval/src/driver/score.ts` uses for an arm, and for the
 * same reason: counting "every hit that overlaps something" would let two hits
 * on one gold span score two true positives. Re-implemented here rather than
 * imported because `scoreArm` takes a `RunRecord` and a `Tier2GoldRow`, neither
 * of which exists at corpus build time; `corpus-v2.test.ts` cross-checks this
 * against `spansMatch("overlap", ...)` so the two definitions cannot drift.
 */
function matchOneToOne(
  hits: readonly OracleSpan[],
  gold: readonly { readonly start: number; readonly end: number }[],
): number {
  const matchedGold = new Array<number>(gold.length).fill(-1);
  let size = 0;
  const augment = (hit: number, seen: boolean[]): boolean => {
    for (let g = 0; g < gold.length; g += 1) {
      if (seen[g] || !overlaps(hits[hit]!, gold[g]!)) continue;
      seen[g] = true;
      if (matchedGold[g] === -1 || augment(matchedGold[g]!, seen)) {
        matchedGold[g] = hit;
        return true;
      }
    }
    return false;
  };
  for (let h = 0; h < hits.length; h += 1) {
    if (augment(h, new Array<boolean>(gold.length).fill(false))) size += 1;
  }
  return size;
}

export function measureOrthography(items: readonly CorpusItem[]): OrthographyReport {
  let gold = 0;
  let found = 0;
  let solved = 0;
  let oracleSpans = 0;
  let oracleOnGold = 0;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let btp = 0;
  let bfp = 0;
  let bfn = 0;
  let itemsWithGold = 0;
  let firstHitIsGold = 0;
  for (const item of items) {
    const hits = orthographicOracle(item.text);
    oracleSpans += hits.length;
    for (const h of hits) if (item.gold.some((g) => overlaps(h, g))) oracleOnGold += 1;
    const m = matchOneToOne(hits, item.gold);
    tp += m;
    fp += hits.length - m;
    fn += item.gold.length - m;
    const budgeted = hits.slice(0, item.gold.length);
    const bm = matchOneToOne(budgeted, item.gold);
    btp += bm;
    bfp += budgeted.length - bm;
    bfn += item.gold.length - bm;
    if (item.gold.length > 0) {
      itemsWithGold += 1;
      if (hits.length > 0 && item.gold.some((g) => overlaps(hits[0]!, g))) firstHitIsGold += 1;
    }
    for (const g of item.gold) {
      gold += 1;
      const hit = hits.some((h) => overlaps(h, g));
      if (!hit) continue;
      found += 1;
      if (hits.every((h) => overlaps(h, g))) solved += 1;
    }
  }
  const f1 = (p: number, r: number) => (p + r === 0 ? 0 : (2 * p * r) / (p + r));
  const precision = oracleSpans === 0 ? 0 : oracleOnGold / oracleSpans;
  const recall = gold === 0 ? 0 : found / gold;
  const op = tp + fp === 0 ? 0 : tp / (tp + fp);
  const orr = tp + fn === 0 ? 0 : tp / (tp + fn);
  const bp = btp + bfp === 0 ? 0 : btp / (btp + bfp);
  const br = btp + bfn === 0 ? 0 : btp / (btp + bfn);
  const firstRate = itemsWithGold === 0 ? 0 : firstHitIsGold / itemsWithGold;
  return {
    goldSpans: gold,
    goldSpansFound: found,
    goldSpansSolvedByOracle: solved,
    solvedRate: gold === 0 ? 0 : solved / gold,
    oracleSpans,
    oracleSpansOnGold: oracleOnGold,
    oraclePrecision: precision,
    oracleRecall: recall,
    asScored: {
      rule: "overlap, maximum one-to-one matching -- the shape apps/eval/src/driver/score.ts uses",
      tp,
      fp,
      fn,
      precision: op,
      recall: orr,
      f1: f1(op, orr),
    },
    budgetMatched: {
      budget: "the item's own gold-span count",
      ranking: "document order -- the oracle's first N hits, left to right",
      tp: btp,
      fp: bfp,
      fn: bfn,
      precision: bp,
      recall: br,
      f1: f1(bp, br),
    },
    itemsWithGold,
    firstHitIsGold,
    firstHitIsGoldRate: firstRate,
    verdict:
      `a reader that returns every orthographically odd region scores P ${op.toFixed(3)} ` +
      `R ${orr.toFixed(3)} on this corpus, scored exactly as an arm is; told how many spans to ` +
      "return on each item it scores " +
      `P ${bp.toFixed(3)} R ${br.toFixed(3)}, and its first hit in document order is a gold span on ` +
      `${firstHitIsGold} of ${itemsWithGold} items carrying one (${(firstRate * 100).toFixed(1)}%). ` +
      "THIS IS THE FLOOR ANY ACCURACY NUMBER OFF THIS CORPUS SITS ON. An arm scoring near these " +
      "figures has not been shown to be reading the policy rather than the character classes. " +
      "solvedRate is reported beside them and certifies nothing: one same-shape distractor per " +
      "positive drives it to zero whatever the rest of the message looks like.",
  };
}

// -- position ---------------------------------------------------------------

export interface PositionReport {
  /** Items carrying at least one gold span AND at least one confusable span. */
  readonly itemsWithBoth: number;
  /** Of those, items whose EARLIEST labelled span is a gold span. */
  readonly goldFirst: number;
  readonly goldFirstRate: number;
  /** (gold, distractor) pairs the generator recorded through `dimensions.distractorFor`. */
  readonly shadowedPairs: number;
  /** Of those, the distractor placed at a higher offset than the span it shadows. */
  readonly distractorAfterGold: number;
  readonly distractorAfterGoldRate: number;
  readonly verdict: string;
}

/**
 * Whether "the earlier of two same-shape odd strings" answers the question.
 *
 * This measurement exists because the fix for the orthographic leak introduced
 * a second one. `generate.ts` places every shape-matched distractor at
 * `(slotIndex + 1) % slots.length` -- one slot AFTER the span it shadows,
 * deterministically, with no randomisation -- so on the great majority of
 * shadowed pairs the gold span is the earlier of the two. Nothing else in this
 * module can see that: `measureOrthography`'s solved rate is satisfied by the
 * distractor's mere existence, and the boost and role measurements do not look
 * at offsets at all.
 *
 * It is measured rather than fixed because fixing it means changing the emitted
 * text, and the emitted text is what two annotators read and labelled. The
 * corpus this round produced carries the bias; the manifest now says how large
 * it is instead of leaving a reader to find it.
 */
export function measurePosition(items: readonly CorpusItem[]): PositionReport {
  let itemsWithBoth = 0;
  let goldFirst = 0;
  let shadowedPairs = 0;
  let distractorAfterGold = 0;
  for (const item of items) {
    const spans = labelSpans(item);
    const golds = spans.filter((s) => !s.type.startsWith(NEG_PREFIX));
    const negs = spans.filter((s) => s.type.startsWith(NEG_PREFIX));
    if (golds.length > 0 && negs.length > 0) {
      itemsWithBoth += 1;
      const earliest = [...spans].sort((a, b) => a.start - b.start)[0]!;
      if (!earliest.type.startsWith(NEG_PREFIX)) goldFirst += 1;
    }
    for (const raw of (item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[]) {
      const dimensions = (raw["dimensions"] ?? {}) as Record<string, string>;
      const shadows = dimensions["distractorFor"];
      if (shadows === undefined) continue;
      const shadowed = spans.find((s) => s.family === shadows);
      if (shadowed === undefined) continue;
      shadowedPairs += 1;
      if ((raw["start"] as number) > shadowed.start) distractorAfterGold += 1;
    }
  }
  const goldFirstRate = itemsWithBoth === 0 ? 0 : goldFirst / itemsWithBoth;
  const afterRate = shadowedPairs === 0 ? 0 : distractorAfterGold / shadowedPairs;
  return {
    itemsWithBoth,
    goldFirst,
    goldFirstRate,
    shadowedPairs,
    distractorAfterGold,
    distractorAfterGoldRate: afterRate,
    verdict:
      `the shape-matched distractor is placed after the span it shadows on ${distractorAfterGold} of ` +
      `${shadowedPairs} pairs (${(afterRate * 100).toFixed(1)}%), and on ${goldFirst} of ` +
      `${itemsWithBoth} items carrying both a gold and a confusable span the EARLIER span is the ` +
      `gold one (${(goldFirstRate * 100).toFixed(1)}% against a 50% coin flip). THIS IS AN OPEN ` +
      "LEAK. `generate.ts` places the distractor one slot along from its gold span with no " +
      "randomisation, so position substitutes for the orthographic discriminator the distractor " +
      "removed. It is not fixed here because fixing it moves every offset in the corpus and the " +
      "labels a blind round wrote are about these offsets.",
  };
}

// -- role readability -------------------------------------------------------

export interface RoleReport {
  readonly orgSpans: number;
  /** Spans whose organisation name appears in only ONE role class corpus-wide. */
  readonly roleLockedSpans: number;
  readonly roleLockedRate: number;
  readonly namesInBothClasses: readonly string[];
  readonly namesInOneClass: readonly { readonly name: string; readonly roleClass: string }[];
  /**
   * The per-name counts, because `roleLockedRate: 0` is a BINARY answer to a
   * question that has a continuous one.
   *
   * A name in both classes is not a name that carries no signal: it can still
   * be 11-to-2 on one side. `nameOnlyAccuracy` is what a classifier that reads
   * the name and nothing else scores by always answering that name's majority
   * class, and `majorityBaseline` is what it scores by always answering the
   * corpus-wide majority. The gap between them is the information the name
   * carries that the locked/unlocked test cannot express.
   */
  readonly perName: readonly { readonly name: string; readonly counts: Readonly<Record<string, number>> }[];
  readonly nameOnlyCorrect: number;
  readonly nameOnlyAccuracy: number;
  readonly majorityBaseline: number;
  readonly nameOnlyLift: number;
}

/**
 * Whether an arm could answer "client or not" from the NAME.
 *
 * A span is role-locked when every occurrence of its name in the whole corpus
 * sits on the same side of the client/non-client line. MEASURED on
 * `injection-p-fin-adjudicated-v1`: 19 of the 39 spans this function counts
 * there, because its client pool, its supplier pool, its competitor pool and
 * its employer name are four disjoint arrays. That corpus also records
 * `constructedRole: "none"` on three families whose clauses do assign a role,
 * which keeps 13 further role-locked spans out of the denominator; counted by
 * hand the figure is 32 of 52.
 *
 * `roleClassOf` is supplied by the caller for the same reason `pairedType` is:
 * the mapping from a written clause to a role class lives in the family
 * catalogue, and re-deriving it here from an id would be a second claim.
 */
export function measureRoles(
  items: readonly CorpusItem[],
  roleClassOf: (dimensions: Readonly<Record<string, string>>, type: string) => string | undefined,
): RoleReport {
  const spans: { name: string; roleClass: string }[] = [];
  for (const item of items) {
    for (const raw of (item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[]) {
      const dimensions = (raw["dimensions"] ?? {}) as Record<string, string>;
      const rc = roleClassOf(dimensions, String(raw["type"]));
      if (rc === undefined) continue;
      spans.push({ name: String(raw["text"]), roleClass: rc });
    }
  }
  const byName = new Map<string, Set<string>>();
  for (const s of spans) {
    const set = byName.get(s.name) ?? new Set<string>();
    set.add(s.roleClass);
    byName.set(s.name, set);
  }
  const locked = spans.filter((s) => byName.get(s.name)!.size === 1);
  const countsByName = new Map<string, Record<string, number>>();
  const countsByClass: Record<string, number> = {};
  for (const s of spans) {
    const row = countsByName.get(s.name) ?? {};
    row[s.roleClass] = (row[s.roleClass] ?? 0) + 1;
    countsByName.set(s.name, row);
    countsByClass[s.roleClass] = (countsByClass[s.roleClass] ?? 0) + 1;
  }
  const nameOnlyCorrect = [...countsByName.values()].reduce((n, row) => n + Math.max(...Object.values(row)), 0);
  const majority = Object.values(countsByClass).length === 0 ? 0 : Math.max(...Object.values(countsByClass));
  const nameOnlyAccuracy = spans.length === 0 ? 0 : nameOnlyCorrect / spans.length;
  const majorityBaseline = spans.length === 0 ? 0 : majority / spans.length;
  return {
    orgSpans: spans.length,
    roleLockedSpans: locked.length,
    roleLockedRate: spans.length === 0 ? 0 : locked.length / spans.length,
    namesInBothClasses: [...byName.entries()].filter(([, v]) => v.size > 1).map(([k]) => k).sort(),
    namesInOneClass: [...byName.entries()]
      .filter(([, v]) => v.size === 1)
      .map(([name, v]) => ({ name, roleClass: [...v][0]! }))
      .sort((a, b) => a.name.localeCompare(b.name)),
    perName: [...countsByName.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([name, counts]) => ({
        name,
        counts: Object.fromEntries(Object.entries(counts).sort((x, y) => x[0].localeCompare(y[0]))),
      })),
    nameOnlyCorrect,
    nameOnlyAccuracy,
    majorityBaseline,
    nameOnlyLift: nameOnlyAccuracy - majorityBaseline,
  };
}

// -- the IR's own counterExample list ---------------------------------------

/**
 * What each of the IR's 23 `counterExamples` entries DENOTES, written down.
 *
 * This is a reading of `policies/compiled/p-fin.ir.json`, not a fact derived
 * from it, and it is here rather than inferred because the inference cannot be
 * made mechanically. "MICR 400240012" denotes the surface "MICR line"; a token
 * test cannot see that, and a corpus family called `micr-code` that mints
 * `MICR <9 digits>` is the same surface with different digits. That identity is
 * the finding this round exists to close -- 13 of 24 confusable families in
 * `injection-p-fin-adjudicated-v1` are 1:1 with an entry below, carrying 59 of
 * its 108 confusable spans -- and closing it requires naming the surfaces.
 *
 * `corpus-v2.test.ts` asserts this table enumerates exactly the counterExamples
 * the IR actually holds, so a recompiled policy that adds one fails here rather
 * than silently escaping the check.
 */
export const IR_COUNTEREXAMPLE_SURFACES: readonly { readonly counterExample: string; readonly surfaceName: string }[] = [
  { counterExample: "ABCDE1234F", surfaceName: "PAN-shaped string with an invalid holder-type character" },
  { counterExample: "AAAPZ1234", surfaceName: "truncated PAN" },
  { counterExample: "27AABCU9603R1ZM", surfaceName: "GSTIN" },
  { counterExample: "PAN application form 49A", surfaceName: "name of a tax form" },
  { counterExample: "1234 5678 9012", surfaceName: "twelve-digit number with a broken Verhoeff digit" },
  { counterExample: "100234567890", surfaceName: "twelve-digit number with a forbidden leading digit" },
  { counterExample: "9876543210", surfaceName: "ten-digit number" },
  { counterExample: "priya.sharma@meridiancap.example.com", surfaceName: "email address" },
  { counterExample: "MICR 400240012", surfaceName: "MICR line" },
  { counterExample: "SWIFT HDFCINBB", surfaceName: "SWIFT/BIC code" },
  { counterExample: "ticket INC0042318", surfaceName: "incident ticket id" },
  { counterExample: "branch code 0247", surfaceName: "bare branch code" },
  { counterExample: "employee id 88213", surfaceName: "employee id" },
  { counterExample: "the client", surfaceName: "unnamed reference to a client" },
  { counterExample: "our counterparty", surfaceName: "unnamed reference to a counterparty" },
  { counterExample: "Mumbai", surfaceName: "place name" },
  { counterExample: "sk-...", surfaceName: "redaction placeholder in credential shape" },
  { counterExample: "AKIA followed by sixteen uppercase characters", surfaceName: "prose description of a key format" },
  { counterExample: "the token has been rotated", surfaceName: "prose statement that a credential was rotated" },
  { counterExample: "https://intranet.example.com/runbook", surfaceName: "internal https URL" },
  { counterExample: "the production database is down", surfaceName: "prose statement about a database" },
  { counterExample: "-----BEGIN CERTIFICATE REQUEST-----", surfaceName: "certificate signing request block" },
  { counterExample: "the public key fingerprint", surfaceName: "public key fingerprint" },
];

export function irCounterExamples(ir: PolicyIr): readonly string[] {
  return ir.entityTypes.flatMap((e) => e.counterExamples ?? []);
}
