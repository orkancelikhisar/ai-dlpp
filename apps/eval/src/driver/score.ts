import { z } from "zod";
import { RunRecordSchema, type RunRecord } from "./record.js";

/**
 * Span-level precision, recall and F1 for tier-2 predicate findings against a
 * blind-labelled gold set (`corpora/fixtures/smoke.gold-tier2.jsonl`).
 *
 * ## Why three match rules and not one
 *
 * A span scorer has to decide when a finding and a gold span are "the same
 * thing", and every available answer is a convention rather than a fact. This
 * module computes all three and reports all three, because on the run that
 * motivated it THEY DISAGREE, and the disagreement is the result.
 *
 * MEASURED on `runs/slate-p-fin-02.*`: the compiled judge is asked a
 * MESSAGE-SCOPED predicate (`policies/compiled/p-fin.ir.json` declares
 * `scope: "message"` for `client-relationship-disclosure`), and the models
 * answer it by returning a span covering most or all of the message --
 * `tier2only-Phi-4-mini` returns [0,74) on `pos-client-name-prose` where the
 * adjudicated gold span is [43,59), "Tamarind Grocers". That pair is a match
 * under `overlap`, and not a match under `exact` or `iou50` (IoU 16/74 = 0.216).
 * Picking one rule would therefore have decided the headline number by picking
 * a convention, which is why `winnersByRule` exists below: it makes a
 * rule-dependent ranking visible instead of letting one rule stand in for all.
 *
 * ## What this module deliberately does NOT do
 *
 * It is not a gate. Nothing here can kill an arm; `ArmGateReport.killedOnRunGates`
 * keeps meaning "failed a property of the run" and no accuracy number enters it.
 * `ArmScoringBoundary.accuracyGated` in `bakeoff.ts` states the same boundary
 * from the other side and stays `false`.
 */

// ---------------------------------------------------------------------------
// The gold file
// ---------------------------------------------------------------------------

/**
 * One annotator's original call, carried verbatim so the artifact keeps its
 * provenance and a later reader can re-adjudicate without the labelling round.
 *
 * `span` is present exactly when that annotator said yes -- including on a
 * DISPUTED row, which is the one place a span survives adjudication without
 * being scoreable. That asymmetry is deliberate: the row's own `spans` is
 * emptied on a disputed row (see the schema refine) so no scorer can reach it
 * by accident, while the evidence for the dispute stays readable here.
 */
const AnnotatorCallSchema = z.object({
  satisfies: z.boolean(),
  confidence: z.enum(["clear", "borderline"]),
  span: z
    .object({
      start: z.number().int().nonnegative(),
      end: z.number().int().positive(),
      text: z.string().min(1),
    })
    .optional(),
  rationale: z.string().min(1),
});

/**
 * A gold span for a semantic predicate.
 *
 * OFFSET UNIT is UTF-16 code units, exactly as `GoldSpanSchema` in `corpus.ts`
 * documents at length -- these offsets index the same `text`, so they cannot be
 * anything else.
 *
 * `end > start` IS ENFORCED HERE and is not in `corpus.ts`'s version. That is a
 * real hole rather than a stylistic difference: `GoldSpanSchema` relies on
 * `CorpusItemSchema`'s refine to reject a degenerate span, and that refine only
 * fires because `text` has `.min(1)` -- `slice()` returns "" for an inverted or
 * empty range and "" would compare equal to a claimed text of "". A scorer is
 * the wrong place to depend on that chain: an inverted span here would compute
 * a negative intersection length and quietly move an IoU.
 */
const Tier2GoldSpanSchema = z
  .object({
    start: z.number().int().nonnegative(),
    end: z.number().int().positive(),
    /** Exactly `text.slice(start, end)` on the corpus item's text. */
    text: z.string().min(1),
  })
  .refine((s) => s.end > s.start, { message: "a gold span's end must be after its start" });

export const Tier2GoldRowSchema = z
  .object({
    itemId: z.string().min(1),
    /**
     * The policy the labels were written against, as a name. `smoke.jsonl`'s
     * own items say `minimal-fixture`; these rows say `p-fin`, because they are
     * §3 of `policies/p-fin.md` and nothing else. The two files therefore
     * disagree about `policy` ON PURPOSE, and a scorer that joined them into one
     * denominator would be mixing labels written against two documents.
     */
    policy: z.string().min(1),
    /**
     * sha256 of the policy DOCUMENT the labels were read from, so the join to a
     * run is checkable rather than asserted: a record carries `policyHash` over
     * the same bytes, and `assertGoldMatchesRecords` refuses a run whose policy
     * differs. Unlike `corpus.ts`'s deliberate choice of a name over a hash,
     * this one can be pinned -- these labels were written against one revision
     * of one document, in one sitting, and if that document changes the labels
     * must be re-read rather than carried forward.
     */
    policyHash: z.string().regex(/^[0-9a-f]{64}$/),
    /** The IR's `semanticPredicates[].id` these labels answer. */
    predicateId: z.string().min(1),
    /**
     * The `entityType` a finding for this predicate carries -- the SHADOW id
     * `pred:<predicateId>` that `judge.ts` mints. This is the field the join is
     * actually made on, because it is what appears on a `Finding`.
     */
    entityType: z.string().min(1),
    /**
     * `disputed` rows are excluded from BOTH the numerator and the denominator
     * of every metric here, and their count is reported. A gold set that admits
     * it does not know is worth more than one that forces a call, and the count
     * is reported so "excluded" cannot be mistaken for "absent".
     */
    status: z.enum(["scored", "disputed"]),
    satisfies: z.boolean(),
    /**
     * The adjudicated confidence, which is the WEAKER of the two annotators'
     * where they differed. It is metadata, not a filter: nothing here scores
     * differently on a `borderline` row. It exists so a reader who wants to see
     * how much of a result rests on soft labels can.
     */
    confidence: z.enum(["clear", "borderline"]),
    spans: z.array(Tier2GoldSpanSchema),
    adjudication: z.string().min(1),
    annotators: z.object({ a: AnnotatorCallSchema, b: AnnotatorCallSchema }),
  })
  .refine((r) => r.satisfies || r.spans.length === 0, {
    message: "a row that does not satisfy the predicate cannot carry gold spans",
  })
  .refine((r) => r.status !== "disputed" || r.spans.length === 0, {
    message: "a disputed row must carry no scoreable spans; its evidence lives under `annotators`",
  })
  .refine((r) => r.status !== "scored" || !r.satisfies || r.spans.length > 0, {
    message: "a scored row that satisfies the predicate must name at least one span",
  });

export type Tier2GoldRow = z.infer<typeof Tier2GoldRowSchema>;

/**
 * Parses the gold JSONL. Blank lines are skipped; every other line must parse.
 *
 * Duplicate `itemId`s are refused for the reason `loadCorpus` refuses them: a
 * duplicate does not collide loudly, it silently doubles one item's weight in
 * every ratio below and drops the other.
 */
export function loadTier2Gold(jsonl: string): Tier2GoldRow[] {
  const rows: Tier2GoldRow[] = [];
  const seen = new Set<string>();
  const lines = jsonl.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!.trim();
    if (line === "") continue;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch (cause) {
      throw new Error(`gold line ${i + 1} is not valid JSON`, { cause });
    }
    const parsed = Tier2GoldRowSchema.safeParse(raw);
    if (!parsed.success) {
      throw new Error(`gold line ${i + 1} is not a valid tier-2 gold row: ${z.prettifyError(parsed.error)}`);
    }
    if (seen.has(parsed.data.itemId)) {
      throw new Error(`gold line ${i + 1} repeats itemId "${parsed.data.itemId}"`);
    }
    seen.add(parsed.data.itemId);
    rows.push(parsed.data);
  }
  return rows;
}

/** Parses a run's JSONL through `RunRecordSchema`, so a reshaped file fails here and not in a ratio. */
export function loadRunRecords(jsonl: string): RunRecord[] {
  const records: RunRecord[] = [];
  const lines = jsonl.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!.trim();
    if (line === "") continue;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch (cause) {
      throw new Error(`run line ${i + 1} is not valid JSON`, { cause });
    }
    const parsed = RunRecordSchema.safeParse(raw);
    if (!parsed.success) {
      throw new Error(`run line ${i + 1} is not a valid run record: ${z.prettifyError(parsed.error)}`);
    }
    records.push(parsed.data);
  }
  return records;
}

/**
 * Gold rows grouped by `predicateId`.
 *
 * Score one predicate at a time. A single P/R/F1 over a gold file carrying two
 * predicates is an average whose weights are the corpus's accident, and it
 * hides the case this project actually cares about -- a judge that answers one
 * clause well and another not at all reads as mediocre at both.
 */
export function groupGoldByPredicate(gold: readonly Tier2GoldRow[]): Map<string, Tier2GoldRow[]> {
  const out = new Map<string, Tier2GoldRow[]>();
  for (const row of gold) {
    const bucket = out.get(row.predicateId);
    if (bucket) bucket.push(row);
    else out.set(row.predicateId, [row]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Match rules
// ---------------------------------------------------------------------------

export type MatchRule = "exact" | "overlap" | "iou50";

/** Every rule, in the order reports print them. */
export const MATCH_RULES: readonly MatchRule[] = ["exact", "overlap", "iou50"];

export const MATCH_RULE_MEANING: Readonly<Record<MatchRule, string>> = {
  exact: "the finding's offsets equal the gold span's offsets",
  overlap: "the finding and the gold span share at least one character",
  iou50: "intersection over union of the two spans is at least 0.5",
};

export interface SpanLike {
  readonly start: number;
  readonly end: number;
}

/** Shared length in UTF-16 code units; 0 when the spans do not touch. */
export function intersectionLength(a: SpanLike, b: SpanLike): number {
  return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
}

/**
 * |A ∩ B| / |A ∪ B| for two half-open intervals.
 *
 * The union is computed as `|A| + |B| - |A ∩ B|` rather than as the hull. For
 * two OVERLAPPING intervals the two agree, because the union is contiguous; for
 * two DISJOINT ones they do not, and the definition is the one that is right:
 * the hull of [0,1) and [99,100) is 100 units of which 2 are in the union, so a
 * hull denominator would be answering a different question. Both give 0 here
 * because the numerator is 0, so this is a statement about which formula is
 * being relied on, not a bug that was found.
 */
export function iou(a: SpanLike, b: SpanLike): number {
  const inter = intersectionLength(a, b);
  if (inter === 0) return 0;
  return inter / (a.end - a.start + (b.end - b.start) - inter);
}

export function spansMatch(rule: MatchRule, a: SpanLike, b: SpanLike): boolean {
  switch (rule) {
    case "exact":
      return a.start === b.start && a.end === b.end;
    case "overlap":
      return intersectionLength(a, b) > 0;
    case "iou50":
      return iou(a, b) >= 0.5;
  }
}

// ---------------------------------------------------------------------------
// Matching findings to gold
// ---------------------------------------------------------------------------

/**
 * Maximum-cardinality bipartite matching (Kuhn's augmenting paths).
 *
 * ONE-TO-ONE IS THE POINT. Counting "every finding that matches something" and
 * "every gold span that is matched by something" separately lets two findings
 * on the same gold span score two true positives, which rewards an arm for
 * emitting the same answer twice. Maximum cardinality also beats greedy: greedy
 * in emission order can take a pairing that blocks a second one, undercounting
 * TP. On a 13-item corpus with at most one gold span per item that never bites,
 * which is exactly why it is worth getting right here rather than after a
 * corpus that shows it.
 *
 * Only the COUNT depends on the algorithm's choices, and it does not: every
 * maximum matching of a bipartite graph has the same size, so tp/fp/fn are
 * invariant even though WHICH pairs are chosen is not. The near-miss
 * diagnostic below reads only the unmatched sets, which are likewise
 * determined up to size -- it is a count, not a list of pairs.
 *
 * @returns `matchedGold[g]` is the finding index matched to gold span `g`, or
 *   -1. Recursion depth is bounded by the number of findings on one item.
 */
function maximumMatching(
  adjacency: readonly (readonly number[])[],
  goldCount: number,
): { size: number; matchedGold: number[] } {
  const matchedGold = new Array<number>(goldCount).fill(-1);
  let size = 0;
  const augment = (finding: number, seen: boolean[]): boolean => {
    for (const g of adjacency[finding]!) {
      if (seen[g]) continue;
      seen[g] = true;
      if (matchedGold[g] === -1 || augment(matchedGold[g]!, seen)) {
        matchedGold[g] = finding;
        return true;
      }
    }
    return false;
  };
  for (let f = 0; f < adjacency.length; f += 1) {
    if (augment(f, new Array<boolean>(goldCount).fill(false))) size += 1;
  }
  return { size, matchedGold };
}

// ---------------------------------------------------------------------------
// Scores
// ---------------------------------------------------------------------------

export interface RuleScore {
  readonly rule: MatchRule;
  readonly tp: number;
  readonly fp: number;
  readonly fn: number;
  /**
   * `tp / (tp + fp)`, or UNDEFINED when the arm produced no findings in this
   * predicate's vocabulary on the scored items.
   *
   * Undefined is neither 0 nor 1 and must not be rendered as either. An arm
   * that answered nothing has made no claim to be right or wrong about, so 0
   * would report it as maximally imprecise and 1 as flawless, and on
   * `runs/slate-p-fin-02` SIX of sixteen arms are in exactly that state -- three
   * because their prompt exhausted the 5,000 ms budget before a first token, so
   * the number would be describing a timeout as an accuracy.
   */
  readonly precision: number | undefined;
  /** `tp / (tp + fn)`, or undefined when the scored items carry no gold spans at all. */
  readonly recall: number | undefined;
  /**
   * `2·tp / (2·tp + fp + fn)`, the harmonic mean written so it needs no
   * division by a zero sum -- but reported as UNDEFINED whenever precision or
   * recall is undefined, rather than at its own narrower condition.
   *
   * The narrower condition would leave F1 defined and equal to 0 for an arm
   * with no findings and unmatched gold, beside a precision of undefined. That
   * is arithmetically fine and reads as a precision of 0, which is the exact
   * misreading this type exists to prevent. So F1 is defined here exactly when
   * both of its inputs are.
   */
  readonly f1: number | undefined;
  /**
   * Findings that count as false positives under this rule while overlapping at
   * least one gold span that counts as a false negative: the arm found the
   * right thing at the wrong span.
   *
   * 0 by construction under `overlap` -- any overlap is a match there -- so a
   * non-zero value under `exact` or `iou50` beside a 0 under `overlap` is the
   * signature of a span-convention disagreement rather than a detection
   * failure. On `runs/slate-p-fin-02` this is what separates the three rules.
   */
  readonly nearMisses: number;
}

/**
 * Everything about an arm's scoreability that does not depend on the match
 * rule, so a reader can tell an arm that answered "no" from one that could not
 * answer at all. Without this an empty `findings` array has two causes and one
 * appearance.
 */
export interface ArmCoverage {
  readonly goldRows: number;
  readonly goldRowsScored: number;
  /** Excluded from every numerator and denominator above. Reported, never silent. */
  readonly goldRowsDisputed: number;
  /**
   * Gold rows this run has NO record for -- a slice, or an arm that died
   * mid-corpus. They are NOT counted as false negatives: the arm was never
   * asked. Named for the ROWS for the reason `tiersTheseRowsCannotScore` is in
   * `bakeoff.ts` -- three items of a thirteen-item corpus is a legitimate run
   * and must not read as ten misses.
   */
  readonly goldRowsWithoutRecord: number;
  /** Records whose itemId has no gold row: their findings are ignored, not scored as FP. */
  readonly recordsWithoutGold: number;
  readonly goldSpansScored: number;
  /** Records the arm produced for scored gold rows. */
  readonly recordsScored: number;
  /** Of those, ones carrying a non-null `error`. Their gold DOES count as FN: the arm was asked. */
  readonly recordsWithError: number;
  /**
   * Scored items on which the arm's judge produced at least one completed call.
   *
   * Read off `tier2Stats.calls` / `baselineStats.calls`, which hold one row per
   * ANSWERED call, so an arm whose every call was aborted before a first token
   * shows 0 here while still having a record per item.
   */
  readonly itemsJudgeAnswered: number;
  /** Scored items where it produced none: the arm could not answer within its budget. */
  readonly itemsJudgeUnanswered: number;
  /** `degraded` reasons at tier 2 over the scored records, counted by reason. */
  readonly tier2DegradedReasons: Readonly<Record<string, number>>;
  /** Findings on scored records whose entityType is in this predicate's vocabulary. */
  readonly findingsInVocabulary: number;
  /** Findings on scored records at any other entityType. Not scored, not FP. */
  readonly findingsOutOfVocabulary: number;
  /**
   * The tiers `findingsInVocabulary` came from, counted.
   *
   * The scorer filters by entityType and NOT by tier, deliberately. On
   * `policies/compiled/p-fin.ir.json` the two are equivalent -- MEASURED, its
   * ten rules map to in-pan, in-aadhaar, bank-account-identifier,
   * internal-customer-id, api-credential, db-connection-string and
   * private-key-material, and none to a `pred:` id, so nothing below tier 2 can
   * emit this vocabulary. Filtering by tier would ASSUME that of every future
   * policy; counting instead means a rule that started emitting a predicate id
   * shows up in this record rather than vanishing from the score.
   */
  readonly findingsInVocabularyByTier: Readonly<Record<string, number>>;
  /**
   * One sentence per reason this arm's score is not a statement about its
   * accuracy. Empty is the positive claim that it is.
   */
  readonly caveats: readonly string[];
}

export interface ScoredArm {
  readonly arm: string;
  readonly predicateId: string;
  /** The entityType ids scored, sorted -- the vocabulary read off the gold rows. */
  readonly entityTypes: readonly string[];
  readonly coverage: ArmCoverage;
  readonly byRule: Readonly<Record<MatchRule, RuleScore>>;
}

/**
 * Refuses a gold set and a run that were not written about the same thing.
 *
 * The failure this prevents is silent and total: gold labelled against one
 * revision of `p-fin.md` scored against a run of another produces a complete,
 * schema-valid, entirely fictional table. `policyHash` is on both sides for
 * this check and no other purpose.
 */
export function assertGoldMatchesRecords(
  gold: readonly Tier2GoldRow[],
  records: readonly RunRecord[],
): void {
  const goldHashes = new Set(gold.map((g) => g.policyHash));
  if (goldHashes.size > 1) {
    throw new Error(
      `the gold rows name ${goldHashes.size} different policy documents (${[...goldHashes].sort().join(", ")}); ` +
        `labels written against different documents cannot share a denominator`,
    );
  }
  const goldHash = [...goldHashes][0];
  if (goldHash === undefined) return;
  const byId = new Map(gold.map((g) => [g.itemId, g]));
  for (const record of records) {
    if (!byId.has(record.itemId)) continue;
    if (record.policyHash !== goldHash) {
      throw new Error(
        `record "${record.itemId}" on arm "${record.arm}" ran policy ${record.policyHash} but the gold ` +
          `was labelled against ${goldHash}: scoring them together would compare a run to labels ` +
          `written about a different document`,
      );
    }
  }
}

export interface ScoreArmInput {
  /** One arm's records. Every record must name the same `arm`. */
  readonly records: readonly RunRecord[];
  /** Gold rows for ONE predicate; see `groupGoldByPredicate`. */
  readonly gold: readonly Tier2GoldRow[];
}

/**
 * Scores one arm against one predicate's gold, under all three match rules.
 *
 * @throws when the records span more than one arm, when the gold spans more
 *   than one predicate, or when a gold span's offsets do not slice back to its
 *   own text on the record's `text`. The last one is the cross-check
 *   `corpus.ts` describes: a reader that indexes wrongly must abort rather than
 *   score against fiction, and a scorer is the reader that would otherwise
 *   produce the fiction.
 */
export function scoreArm(input: ScoreArmInput): ScoredArm {
  const { records, gold } = input;

  const arms = new Set(records.map((r) => r.arm));
  if (arms.size > 1) throw new Error(`scoreArm was given ${arms.size} arms (${[...arms].sort().join(", ")}); score one at a time`);
  const predicates = new Set(gold.map((g) => g.predicateId));
  if (predicates.size > 1) {
    throw new Error(
      `scoreArm was given ${predicates.size} predicates (${[...predicates].sort().join(", ")}); ` +
        `use groupGoldByPredicate -- one ratio over two predicates is an average weighted by the corpus`,
    );
  }
  const arm = [...arms][0] ?? "";
  const predicateId = [...predicates][0] ?? "";
  const entityTypes = new Set(gold.map((g) => g.entityType));

  assertGoldMatchesRecords(gold, records);

  const recordsById = new Map<string, RunRecord>();
  for (const record of records) {
    if (recordsById.has(record.itemId)) {
      throw new Error(`arm "${arm}" has two records for item "${record.itemId}"; one of them would be silently dropped`);
    }
    recordsById.set(record.itemId, record);
  }
  const goldById = new Map(gold.map((g) => [g.itemId, g]));

  const scored = gold.filter((g) => g.status === "scored");
  const disputed = gold.length - scored.length;

  // Per item, the finding spans in this predicate's vocabulary and the gold
  // spans, paired up once and reused by all three rules.
  const pairs: { findings: SpanLike[]; goldSpans: SpanLike[] }[] = [];
  let goldRowsWithoutRecord = 0;
  let recordsScored = 0;
  let recordsWithError = 0;
  let itemsJudgeAnswered = 0;
  let findingsInVocabulary = 0;
  let findingsOutOfVocabulary = 0;
  let goldSpansScored = 0;
  const tier2DegradedReasons: Record<string, number> = {};
  const findingsInVocabularyByTier: Record<string, number> = {};

  for (const row of scored) {
    const record = recordsById.get(row.itemId);
    if (record === undefined) {
      goldRowsWithoutRecord += 1;
      continue;
    }
    recordsScored += 1;
    if (record.error !== null && record.error !== undefined) recordsWithError += 1;

    for (const span of row.spans) {
      if (record.text.slice(span.start, span.end) !== span.text) {
        throw new Error(
          `gold span [${span.start},${span.end}) on item "${row.itemId}" holds ${JSON.stringify(span.text)} ` +
            `but the record's text holds ${JSON.stringify(record.text.slice(span.start, span.end))}: ` +
            `the gold and the run disagree about the message, so no score over them means anything`,
        );
      }
    }
    goldSpansScored += row.spans.length;

    const inVocab = record.findings.filter((f) => entityTypes.has(f.entityType));
    findingsInVocabulary += inVocab.length;
    findingsOutOfVocabulary += record.findings.length - inVocab.length;
    for (const f of inVocab) {
      const key = String(f.tier);
      findingsInVocabularyByTier[key] = (findingsInVocabularyByTier[key] ?? 0) + 1;
    }

    for (const notice of record.degraded ?? []) {
      if (notice.tier !== 2) continue;
      tier2DegradedReasons[notice.reason] = (tier2DegradedReasons[notice.reason] ?? 0) + 1;
    }

    const answered = (record.tier2Stats?.calls.length ?? 0) + (record.baselineStats?.calls.length ?? 0);
    if (answered > 0) itemsJudgeAnswered += 1;

    pairs.push({
      findings: inVocab.map((f) => ({ start: f.start, end: f.end })),
      goldSpans: row.spans.map((s) => ({ start: s.start, end: s.end })),
    });
  }

  let recordsWithoutGold = 0;
  for (const record of records) if (!goldById.has(record.itemId)) recordsWithoutGold += 1;

  const byRule = {} as Record<MatchRule, RuleScore>;
  for (const rule of MATCH_RULES) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let nearMisses = 0;
    for (const { findings, goldSpans } of pairs) {
      const adjacency = findings.map((f) => {
        const out: number[] = [];
        for (let g = 0; g < goldSpans.length; g += 1) if (spansMatch(rule, f, goldSpans[g]!)) out.push(g);
        return out;
      });
      const { size, matchedGold } = maximumMatching(adjacency, goldSpans.length);
      tp += size;
      const matchedFindings = new Set(matchedGold.filter((f) => f !== -1));
      const unmatchedFindings = findings.filter((_, i) => !matchedFindings.has(i));
      const unmatchedGold = goldSpans.filter((_, g) => matchedGold[g] === -1);
      fp += unmatchedFindings.length;
      fn += unmatchedGold.length;
      for (const f of unmatchedFindings) {
        if (unmatchedGold.some((g) => intersectionLength(f, g) > 0)) nearMisses += 1;
      }
    }
    const precision = tp + fp === 0 ? undefined : tp / (tp + fp);
    const recall = tp + fn === 0 ? undefined : tp / (tp + fn);
    const f1 = precision === undefined || recall === undefined ? undefined : (2 * tp) / (2 * tp + fp + fn);
    byRule[rule] = { rule, tp, fp, fn, precision, recall, f1, nearMisses };
  }

  const itemsJudgeUnanswered = recordsScored - itemsJudgeAnswered;
  const caveats: string[] = [];
  if (disputed > 0) {
    caveats.push(
      `${disputed} of ${gold.length} gold rows are adjudicated disputed and are excluded from both the ` +
        `numerator and the denominator of every metric here`,
    );
  }
  if (goldRowsWithoutRecord > 0) {
    caveats.push(
      `${goldRowsWithoutRecord} scored gold rows have no record on this arm, so this arm was never asked ` +
        `about them; they are not counted as misses`,
    );
  }
  if (itemsJudgeUnanswered > 0) {
    caveats.push(
      `the judge produced no completed call on ${itemsJudgeUnanswered} of ${recordsScored} scored items, so ` +
        `an absence of findings there is an arm that could not answer, not an arm that answered "no"`,
    );
  }
  if (findingsInVocabulary === 0) {
    caveats.push(
      `this arm emitted no finding at ${[...entityTypes].sort().join(", ")} on any scored item, so its ` +
        `precision is UNDEFINED rather than 0 or 1 under every rule`,
    );
  }
  if (recordsWithError > 0) {
    caveats.push(`${recordsWithError} scored records carry an error; their gold still counts as missed`);
  }

  return {
    arm,
    predicateId,
    entityTypes: [...entityTypes].sort(),
    coverage: {
      goldRows: gold.length,
      goldRowsScored: scored.length,
      goldRowsDisputed: disputed,
      goldRowsWithoutRecord,
      recordsWithoutGold,
      goldSpansScored,
      recordsScored,
      recordsWithError,
      itemsJudgeAnswered,
      itemsJudgeUnanswered,
      tier2DegradedReasons,
      findingsInVocabulary,
      findingsOutOfVocabulary,
      findingsInVocabularyByTier,
      caveats,
    },
    byRule,
  };
}

/** Splits a multi-arm run into one `ScoredArm` per arm, sorted by arm name. */
export function scoreArms(records: readonly RunRecord[], gold: readonly Tier2GoldRow[]): ScoredArm[] {
  const byArm = new Map<string, RunRecord[]>();
  for (const record of records) {
    const bucket = byArm.get(record.arm);
    if (bucket) bucket.push(record);
    else byArm.set(record.arm, [record]);
  }
  return [...byArm.keys()]
    .sort()
    .map((arm) => scoreArm({ records: byArm.get(arm)!, gold }));
}

export interface RuleWinners {
  readonly rule: MatchRule;
  /** Arms tied at the best defined F1, sorted. Empty when no arm has a defined F1. */
  readonly winners: readonly string[];
  readonly bestF1: number | undefined;
  /**
   * Arms with no defined F1 under this rule, sorted. They are NOT ranked last:
   * an arm that made no claim has not lost a comparison, it was not in one.
   */
  readonly unrankable: readonly string[];
}

export interface WinnerDisagreement {
  readonly byRule: readonly RuleWinners[];
  /**
   * True when the three rules do not name the same winning set.
   *
   * When it is true the disagreement IS the finding and must be reported as
   * one -- the choice of match rule is a convention, so a ranking that moves
   * with it is a statement about the convention and not about the arms.
   */
  readonly disagree: boolean;
}

/** Which arm each rule ranks first, and whether the rules agree about it. */
export function winnersByRule(scored: readonly ScoredArm[]): WinnerDisagreement {
  const byRule = MATCH_RULES.map((rule): RuleWinners => {
    const ranked = scored.filter((s) => s.byRule[rule].f1 !== undefined);
    const unrankable = scored.filter((s) => s.byRule[rule].f1 === undefined).map((s) => s.arm).sort();
    if (ranked.length === 0) return { rule, winners: [], bestF1: undefined, unrankable };
    const bestF1 = Math.max(...ranked.map((s) => s.byRule[rule].f1!));
    return {
      rule,
      winners: ranked.filter((s) => s.byRule[rule].f1! === bestF1).map((s) => s.arm).sort(),
      bestF1,
      unrankable,
    };
  });
  const keys = byRule.map((r) => r.winners.join("|"));
  return { byRule, disagree: new Set(keys).size > 1 };
}
