import {
  segmentText,
  selectSegments,
  uncertainSegmentStarts,
  UNCERTAIN_BELOW,
  type Finding,
} from "@sih/core";
import type { CorpusItem } from "./corpus.js";

/**
 * How big is the passage a tier-2 judge is actually handed?
 *
 * Plan 5's replacement latency gate says the wall-clock budget comes from a
 * MEASURED segment-size distribution over the tier-0/1 corpus, and until this
 * module existed that measurement did not. A budget without it is a guess, and
 * the kill rule it feeds is as arbitrary as the one it replaced.
 *
 * ## Why this imports the escalation policy rather than filtering by kind
 *
 * The sizes that matter are the sizes of the segments a judge would really see,
 * and that set is decided by spec 4.1's escalation policy -- which excludes code
 * under the predicate branch and re-admits it under the uncertainty branch. A
 * `kind !== "code"` filter written here would be a second copy of half that
 * rule, free to drift from the one `detect` runs, and the drift would silently
 * size the budget for work tier 2 never does.
 *
 * The import is from `@sih/core` and not from `@sih/tier2`, and that is not a
 * shortcut around a missing dependency. `packages/tier2/src/escalate.ts` is
 * itself a bare re-export of `packages/core/src/detect/escalate.ts` -- core
 * cannot depend on the tier it gates, so the definition lives in core and tier2
 * re-exports it, and `packages/tier2/test/escalate.test.ts` asserts the two are
 * the same function objects. So this file imports the same binding Task 7
 * landed, by the shortest route to it. Adding an `@sih/eval` -> `@sih/tier2`
 * dependency would buy nothing and cost something: tier2 pulls in
 * `@mlc-ai/web-llm`, which is WebGPU-only, and this driver is Node.
 *
 * ## What this module does not measure
 *
 * TOKENS. A size here is UTF-16 code units, UTF-8 bytes, or whitespace-delimited
 * words, all three counted directly off the segment. Converting any of them to
 * tokens needs the model's own tokenizer, none of the four pinned tier-2 arms
 * has one cached on this machine, and inventing a chars-per-token ratio would
 * put a fabricated number where the budget reads a measured one.
 *
 * ## The one token claim this module supports, and the one it used to
 *
 * It used to say that characters bound tokens from above for any byte-level BPE
 * vocabulary. That is FALSE, and it is false in the unsafe direction. A
 * byte-level BPE tokenizes UTF-8 BYTES, so its floor is one token per byte, and
 * every non-ASCII character is 2-4 bytes while `String.length` counts UTF-16
 * code units.
 *
 * MEASURED HERE, against a real byte-level BPE that is cached in this repo --
 * `packages/tier1/models/gliner-pii-edge/tokenizer.json`, whose `model.type` is
 * `BPE` and whose pre-tokenizer is `ByteLevel` -- encoded through this app's own
 * `@huggingface/transformers`:
 *
 *     "x" repeated 20x          20 units   20 bytes    4 tokens
 *     U+1F389 repeated 20x      40 units   80 bytes   60 tokens  <- over the units
 *     U+65E5 repeated 20x       20 units   60 bytes   21 tokens  <- over the units
 *     one Devanagari word        6 units   18 bytes    7 tokens  <- over the units
 *
 * So the ceiling is the UTF-8 BYTE count, and `bytes` exists for exactly that:
 * it is the number to carry into a token budget, and `chars` is not.
 *
 * On `corpora/fixtures/smoke.jsonl` the two barely differ -- 1,081 units against
 * 1,093 bytes over the selected segments, and the largest segment is ASCII in
 * both units -- which is why the budget arithmetic Plan 5 quotes off this corpus
 * is unaffected. It matters for the re-run the plan schedules against Plan 7's
 * corpus: that is Indian-context scraped chat, `corpus.ts` already documents
 * lone surrogates from truncated emoji in it, and at 3 bytes per Devanagari code
 * unit a character count would understate the ceiling threefold.
 */

/** Character (UTF-16 code unit) counts, UTF-8 byte counts, word counts, and segments-per-message. */
/**
 * What ONE engine call covers on an arm.
 *
 * Declared here rather than in `bakeoff.ts` because this module is the lower
 * layer -- bakeoff imports segments and not the other way round -- and both
 * `judgedUnitFor`'s return and `SegmentSizeDistribution.unit` have to be the
 * same union or an arm could be paired with a distribution measured over the
 * other one.
 *
 * THREE members and not two, and the third is the reason the union was widened.
 * `WebLlmJudge` partitions its predicates by the scope each was DECLARED in and
 * calls once about the whole message plus once per selected segment, so on a
 * policy declaring BOTH scopes neither of the first two members describes the
 * work: a segment sample omits the whole-message call the arm certainly makes,
 * and a message sample omits every segment call. `"segment+message"` is the
 * union of the two samples, one entry per engine call, which is what
 * `ArmGateReport.judgedUnitChars` claims to be.
 *
 * The alternative considered and rejected was keeping the sample segment-based
 * and letting the row name the message call separately. It was rejected because
 * it reproduces, one call smaller, exactly the defect the third member exists to
 * close: `judgedUnitChars` would describe some of the arm's prompts and be read
 * as all of them. No policy in this repository declares both scopes, so nothing
 * here is measured on one -- see `bakeoff.test.ts`'s constructed IR.
 */
export type JudgedUnit = "segment" | "message" | "segment+message";

export interface SizeStats {
  readonly p50: number;
  readonly p95: number;
  readonly max: number;
  readonly min: number;
}

export interface SegmentSizeDistribution {
  /**
   * What ONE engine call covers on the arm this distribution describes, and
   * therefore what `chars`, `bytes`, `words` and `perItem` are counted over.
   *
   * `"segment"` is the compiled judge on a policy whose semantic predicates are
   * all segment-scoped: escalation picks segments and the judge makes one call
   * per selected segment. `"message"` is Approach B, which does not escalate
   * and does not segment -- it makes one call per message with the whole policy
   * and the whole message in it -- and it is ALSO the compiled judge on a policy
   * whose predicates are all message-scoped, which is what
   * `policies/compiled/p-fin.ir.json` is. `"segment+message"` is the compiled
   * judge on a policy declaring both. So this is a property of the family AND
   * the policy, and `judgedUnitFor` in `bakeoff.ts` is where the two meet.
   *
   * A FIELD rather than a caller's memory of what it asked for, because the
   * distributions are all well-formed and none says so anywhere else. On
   * `smoke.jsonl` a segment distribution reports a p50 of 62 characters and a
   * message distribution 78: a report carrying the wrong one puts a plausible
   * number under a gate whose ceiling was derived at the other unit.
   */
  readonly unit: JudgedUnit;
  /** Corpus items measured. */
  readonly items: number;
  /**
   * Every segment `segmentText` produced, selected or not.
   *
   * A fact about the CORPUS, so it is counted the same way under both units --
   * on a message distribution it is the segmentation Approach B does not use,
   * kept because it is what makes the two units comparable at all.
   */
  readonly segmentsTotal: number;
  /** Judged units SELECTED -- the sample everything below describes. */
  readonly count: number;
  /**
   * Segment size in UTF-16 code units, the unit every offset in this pipeline
   * uses (see the header of `corpus.ts` for why the repo is UTF-16 throughout).
   *
   * `undefined` rather than zeroes when `count` is 0. A `p50: 0` would assert
   * that the median judged segment is empty, which is a statement about a
   * sample that does not exist, and a caller could size a budget off it without
   * ever noticing there was nothing to size against.
   */
  readonly chars: SizeStats | undefined;
  /**
   * Segment size in UTF-8 BYTES. `undefined` when `count` is 0.
   *
   * The unit a token budget has to be read in, and the reason it is a separate
   * field rather than a note on `chars`: a byte-level BPE bottoms out at one
   * token per byte, so THIS is the ceiling on how many tokens a segment can
   * become, while `chars` -- UTF-16 code units -- is smaller than the byte count
   * for every non-ASCII character. See the module header for the measurement.
   */
  readonly bytes: SizeStats | undefined;
  /** Whitespace-delimited words per segment. `undefined` when `count` is 0. */
  readonly words: SizeStats | undefined;
  /**
   * Selected judged units per corpus ITEM, one sample per item, zeroes included.
   *
   * The budget being sized is per message and a judge makes one engine call per
   * selected segment, so this is the multiplier on any per-call latency. It is
   * not derivable from `count / items`: a mean cannot say whether one message
   * costs three calls.
   *
   * `undefined` only when there are no items at all -- an item that selected
   * nothing is a 0 in the sample, not an absence from it. Under
   * `unit: "message"` every sample is 1 by construction, which is the whole of
   * what "one call per message" costs; under `"segment+message"` every sample
   * is the selected count PLUS one, and it is never 0 -- an item escalation
   * selected nothing on still costs the whole-message call.
   */
  readonly perItem: SizeStats | undefined;
  /** The raw samples, so a percentile quoted anywhere else can be re-derived. */
  readonly samples: {
    readonly chars: readonly number[];
    readonly bytes: readonly number[];
    readonly words: readonly number[];
    readonly perItem: readonly number[];
  };
  /**
   * The escalation input this ran under, resolved.
   *
   * A distribution is only readable against the policy shape that produced it:
   * `hasPredicates: false` selects nothing but the uncertain segments, and a
   * different `uncertainBelow` moves which those are. Recorded as fact rather
   * than left to the caller's memory of what it passed.
   */
  readonly escalation: {
    /**
     * Whether escalation SELECTED any of this distribution's sample.
     *
     * False exactly when `unit` is `"message"`: that arm judges every message
     * unconditionally -- Approach B because it never escalates, a compiled arm
     * on a message-only policy because its one call is about the whole message
     * either way -- so `hasPredicates` and `uncertainBelow` below describe
     * inputs that decided nothing here. They are still recorded, because they
     * are what the PAIRED compiled family escalated on and a reader comparing
     * the two families needs both sides -- but read them as the other arm's
     * condition, not as this one's.
     *
     * True under `"segment+message"`, and the word is "any" for that case: the
     * segment half of the sample is escalation's, and the one message entry per
     * item is unconditional.
     */
    readonly applies: boolean;
    readonly hasPredicates: boolean;
    readonly uncertainBelow: number;
    /**
     * Whether a `priorFindings` source was supplied at all.
     *
     * The THIRD escalation input, and the one that separates the two conditions
     * a bake-off must never splice: an arm that runs tier 0 feeds its findings
     * in here and gets the uncertainty branch, an arm that does not feeds
     * nothing and gets only the predicate branch. Both produce a
     * well-formed distribution and neither says so anywhere else, so a report
     * could carry one family's distribution under the other family's name --
     * which is what `gateReport` now refuses with this field.
     *
     * "Supplied", not "found something": a callback that returns no finding on
     * any item is still the tier-0 condition, and a policy whose rules happen
     * to match nothing on one corpus is a measurement rather than a
     * misconfiguration. `count` and `perItem` are where the difference shows up.
     */
    readonly hasPriors: boolean;
  };
}

export interface SegmentSizeOptions {
  /**
   * `ir.semanticPredicates.length > 0` for the policy being sized. Defaults to
   * `true`.
   *
   * True is the default because it is the only input under which tier 2 spends
   * anything: `WebLlmJudge` returns an empty verdict without an engine call when
   * the IR declares no predicates, so a distribution measured at `false` sizes a
   * budget for zero calls. Worth knowing when reading any number this produces:
   * of the THREE IRs under `apps/eval/fixtures`, only `semantic-ir.json`
   * declares a predicate -- `minimal-ir.json` and `multiclass-ir.json` both
   * carry `semanticPredicates: []`, and `minimal` is the page's default. So the
   * default here is the policy shape the bake-off needs and not the one the
   * harness loads unless a caller has asked for `semantic`. (This paragraph
   * named two fixtures until `semantic-ir.json` was added under it; re-read
   * from the files rather than trusted.)
   */
  readonly hasPredicates?: boolean;
  /**
   * The tier-0/1 findings for one item, which decide which of its segments an
   * earlier tier left uncertain. Defaults to none.
   *
   * A callback keyed on the item rather than findings baked in here, because
   * producing them means choosing an IR, and which policy the corpus is scored
   * under is not this module's decision to make.
   */
  readonly priorFindings?: (item: CorpusItem) => readonly Finding[];
  /** Forwarded to `uncertainSegmentStarts`; defaults to core's `UNCERTAIN_BELOW`. */
  readonly uncertainBelow?: number;
  /**
   * What one engine call covers on the arm being sized. Defaults to
   * `"segment"`, which is the compiled judge on a segment-scoped policy and was
   * the only arm this module had when it was written.
   *
   * Under `"message"` escalation is not consulted: the sample is one whole
   * message per item, `perItem` is all 1s, and `escalation.applies` is false.
   * `priorFindings` is still meaningful there and is still recorded through
   * `hasPriors` -- an Approach-B arm with tier 0 in front of it really does run
   * tier 0, and `gateReport` uses that flag to refuse a report carrying the
   * other family's distribution.
   *
   * Under `"segment+message"` both halves are measured: the whole message once
   * per item and every selected segment, which is one entry per engine call
   * `WebLlmJudge` would make on a policy declaring both scopes.
   *
   * This module does NOT decide which of the three an arm gets. That takes the
   * family and the IR's declared scopes together, and `judgedUnitFor` in
   * `bakeoff.ts` is where both are in hand.
   */
  readonly unit?: JudgedUnit;
}

/**
 * The `percent`-th percentile of `values` by the NEAREST-RANK definition: sort
 * ascending, take the `ceil(percent * n / 100)`-th value. Order of `values` is
 * irrelevant; the array is not mutated.
 *
 * Nearest-rank and not linear interpolation, deliberately. Every number this
 * returns is a size some real segment had, so a budget quoting it is quoting an
 * observation. Interpolation would report a p50 of 50.5 characters for a corpus
 * in which no segment is 50.5 characters long.
 *
 * `percent` is an INTEGER, and refusing a fractional one is what lets this
 * arithmetic carry no epsilon. MEASURED on Node 26 over every (n <= 20000,
 * percent 1..100) pair -- 2,000,000 of them -- `Math.ceil((percent * n) / 100)`
 * agreed with exact BigInt arithmetic every single time: `percent * n` is an
 * exact integer, and IEEE-754 division is correctly rounded, so a quotient that
 * is an integer is reproduced exactly and a quotient that is not sits at least
 * 1/100 from the nearest integer -- far outside double error at these
 * magnitudes. The fraction spelling has no such property: in the same sweep
 * `Math.ceil(0.07 * n)` disagreed with the exact rank 153 times, starting at
 * n = 100, because `0.07 * 100` is 7.000000000000001 and ceils to 8.
 */
export function percentile(values: readonly number[], percent: number): number {
  if (values.length === 0) {
    throw new Error("percentile of an empty sample is undefined; check `count` first");
  }
  if (!Number.isInteger(percent)) {
    throw new Error(`percentile percent must be an integer, got ${String(percent)}`);
  }
  if (!(percent > 0 && percent <= 100)) {
    throw new Error(`percentile percent must be in (0, 100], got ${String(percent)}`);
  }
  // Numeric comparator, not the default. `[...]` because sorting the caller's
  // array in place would reorder `samples.chars` under whoever holds it.
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((percent * sorted.length) / 100);
  return sorted[rank - 1]!;
}

/**
 * The four-number summary of a sample, or `undefined` for an empty one.
 *
 * EXPORTED rather than kept private because `bakeoff.ts` summarises latency and
 * token samples with exactly this shape, and a second copy of these four lines
 * is a second definition of "p95" free to drift from this one -- which is how
 * one file's nearest-rank number ends up being compared against another file's
 * interpolated one. The `undefined` on an empty sample is the same refusal
 * `SegmentSizeDistribution.chars` documents: a `p50: 0` over no samples asserts
 * a median that no observation supports.
 */
export function sizeStats(values: readonly number[]): SizeStats | undefined {
  if (values.length === 0) return undefined;
  return {
    p50: percentile(values, 50),
    p95: percentile(values, 95),
    max: percentile(values, 100),
    min: Math.min(...values),
  };
}

/**
 * Segment every item, keep the passages the arm's model would actually be shown
 * -- the segments spec 4.1's escalation policy selects, the whole message, or
 * both -- and describe their sizes.
 *
 * Passages are measured once each: an item contributing two selected segments
 * contributes two samples, because that is two engine calls, and under
 * `"segment+message"` it contributes three.
 */
export function segmentSizeDistribution(
  items: readonly CorpusItem[],
  options: SegmentSizeOptions = {},
): SegmentSizeDistribution {
  const hasPredicates = options.hasPredicates ?? true;
  const uncertainBelow = options.uncertainBelow ?? UNCERTAIN_BELOW;
  // Recorded before the default is applied, because the default is exactly what
  // the flag has to distinguish: `() => []` supplied by a caller and `() => []`
  // filled in here produce identical numbers, and only one of them is an arm
  // that ran tier 0.
  const hasPriors = options.priorFindings !== undefined;
  const priorFindings = options.priorFindings ?? (() => []);
  const unit = options.unit ?? "segment";

  const chars: number[] = [];
  const bytes: number[] = [];
  const words: number[] = [];
  const perItem: number[] = [];
  let segmentsTotal = 0;

  // One passage's three sizes, pushed together so no unit can add a sample to
  // one of the three arrays and not the others.
  const measure = (text: string): void => {
    chars.push(text.length);
    bytes.push(utf8Length(text));
    words.push(countWords(text));
  };

  for (const item of items) {
    const segments = segmentText(item.text);
    segmentsTotal += segments.length;
    if (unit === "message") {
      // No escalation call at all, rather than one whose result is discarded.
      // This arm is shown the message whatever any tier below found, and
      // running `selectSegments` here to throw the answer away would leave a
      // reader of this loop believing the two units differ only in how the
      // samples are aggregated.
      perItem.push(1);
      measure(item.text);
      continue;
    }
    // `uncertainSegmentStarts` is what turns findings into the segment starts
    // `selectSegments` demands, and the conversion is never done by hand here
    // because the natural hand version -- passing finding starts -- is wrong.
    // core REFUSES that rather than absorbing it: `selectSegments` throws
    // ("escalation was given uncertain offset N, which starts no segment"), and
    // its own guard comment says the throw exists precisely because the silent
    // behaviour, selecting nothing, is indistinguishable from a message with
    // nothing uncertain. VERIFIED by calling it with a finding start.
    const uncertain = uncertainSegmentStarts(segments, priorFindings(item), uncertainBelow);
    const selected = selectSegments(segments, { hasPredicates, uncertain });
    // The whole-message call FIRST, in the order `WebLlmJudge.judge` issues
    // them: it makes the message call before entering the segment loop, and its
    // docblock gives the reason (the message call is exactly one call, known
    // before the run, so a tight budget must not leave the policy's message
    // clause covered on short messages and skipped on long ones). Order is
    // irrelevant to every statistic below -- `percentile` sorts -- and is kept
    // because `samples` is exported for re-derivation and a reader checking it
    // against a run's call rows should meet the same sequence.
    const passages = unit === "segment+message" ? [item.text] : [];
    for (const segment of selected) passages.push(segment.text);
    perItem.push(passages.length);
    for (const passage of passages) measure(passage);
  }

  return {
    unit,
    items: items.length,
    segmentsTotal,
    count: chars.length,
    chars: sizeStats(chars),
    bytes: sizeStats(bytes),
    words: sizeStats(words),
    perItem: sizeStats(perItem),
    samples: { chars, bytes, words, perItem },
    escalation: { applies: unit !== "message", hasPredicates, uncertainBelow, hasPriors },
  };
}

// One encoder, reused: `TextEncoder` and not `Buffer.byteLength` so this is the
// same function in the browser half of this app as in the driver.
const UTF8 = new TextEncoder();

/**
 * One segment's size in UTF-8 bytes.
 *
 * The unit is the point, not the spelling: see the module header for the
 * measurement showing that a byte-level BPE can emit more tokens than the
 * string has UTF-16 code units, so bytes and not `String.length` are what
 * ceiling a token count.
 */
function utf8Length(text: string): number {
  return UTF8.encode(text).length;
}

/**
 * Words as whitespace-delimited runs.
 *
 * A proxy for how much a model has to read, never a token count -- see the
 * module header.
 *
 * The `trim` and the empty guard keep an all-whitespace segment at 0. That
 * segment is reachable, and by a route an earlier version of this comment named
 * wrongly: a blank line between two PROSE runs is not its own segment at all --
 * `segmentText` merges same-kind lines into one run, so "a\n\nb" is one prose
 * segment. What does stand alone is a blank run between two runs of a DIFFERENT
 * kind, MEASURED against core's segmenter: "a: 1\n\nb: 2\n" segments as
 * kv / prose "\n" / kv, and a fenced block followed by a blank line and a kv
 * run gives code / prose "\n\n" / kv. The escalation policy selects those prose
 * runs like any other.
 *
 * On such a segment the un-guarded count is 2, not 1: MEASURED,
 * `"\n".split(/\s+/)` is `["", ""]`. (`"".split(/\s+/)` is `[""]`, length 1 --
 * that is the value the `trim` alone would produce, and it is why both halves
 * are here.) Either way it would report words in a segment that has none, and
 * `words.min` would come back as a number no segment's content justifies.
 */
function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/).length;
}
