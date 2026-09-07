import { z } from "zod";
import { RunRecordSchema, type RecordFinding, type RunRecord } from "./record.js";

/**
 * Span-level precision, recall and F1 for tier-2 predicate findings against a
 * blind-labelled gold set. Three exist: `corpora/fixtures/smoke.gold-tier2.jsonl`
 * (13 items, the fixture this module was written against),
 * `injection-p-fin-v2.gold-tier2.jsonl` (20 items, 0 positives) and
 * `injection-p-fin-v2.gold-tier2-predicate.jsonl` (189 items, 19 positives).
 * They are different rounds over overlapping items and must never be pooled;
 * `loadTier2Gold` refuses the pooling by rejecting a repeated itemId.
 *
 * READ `TRIVIAL_FLOOR_READERS` BELOW BEFORE QUOTING ANY NUMBER THIS MODULE
 * PRODUCES. On the 189-item predicate gold, no arm of the sixteen measured
 * beats a regular expression that reads neither the policy nor the message.
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
 * answered it by returning a span covering most or all of the message --
 * `tier2only-Phi-4-mini` returned [0,74) on `pos-client-name-prose` where the
 * adjudicated gold span is [43,59), "Tamarind Grocers". That pair is a match
 * under `overlap`, and not a match under `exact` or `iou50` (IoU 16/74 = 0.216).
 * Picking one rule would therefore have decided the headline number by picking
 * a convention, which is why `winnersByRule` exists below: it makes a
 * rule-dependent ranking visible instead of letting one rule stand in for all.
 *
 * ## What changed under that run, and what did NOT
 *
 * That disagreement was not only a scoring convention. `Finding.start/end` is
 * what `applyActions` rewrites, so a whole-message span on a `redact` predicate
 * replaces the whole message -- the arms were being asked for a span that was
 * wrong for the shipping path and unscoreable on two of three rules at once.
 * Both arms now return TWO spans (`packages/tier2/src/spans.ts`): an evidence
 * clause that locates the finding and a mention inside it, and `Finding`
 * carries the mention. So `exact` and `iou50` are REACHABLE where they were
 * structurally unreachable.
 *
 * ## Reachable is not the same as reached: what the next run actually did
 *
 * The paragraph above stops at "reachable" on purpose, because the run taken
 * after the split is WORSE on every accuracy column and nothing else in this
 * repository records it. MEASURED apples-to-apples, both slates through ONE
 * scorer -- this one, with `RunRecordSchema`'s two new counters relaxed to
 * optional in a scratch copy so it can read the older records at all -- summed
 * over all 16 arms and 208 records of each slate:
 *
 *   slate-p-fin-02 (one-span ask)  exact tp 0 fp 13 | overlap tp 3 fp 10 | iou50 tp 0 fp 13
 *   slate-p-fin-03 (two-span ask)  exact tp 0 fp 11 | overlap tp 0 fp 11 | iou50 tp 0 fp 11
 *
 * `tier2Config`, `irHash` (b00e5ce67a7e...), `policyHash` (ebb3cd68d973...) and
 * `temperature: 0` are identical between the two slates, so the intervention is
 * the span convention. Overlap true positives fell 3 to 0; `exact` and `iou50`
 * stayed at 0. Every accuracy column on all 16 arms of the later slate reads
 * 0.000, and it reads 0.000 under all three rules rather than under two.
 *
 * End-to-end coverage of the corpus gold spans moved the same way and barely:
 * over `corpora/fixtures/smoke.jsonl`'s own gold, spans fully covered by the
 * union of a record's shipped findings went 53 of 112 to 51 of 112, and every
 * uncovered case in BOTH slates is a total miss rather than a partial cover. n
 * is 13 items times 16 arms at one draw each, so this is a fact about two
 * recorded slates and not an effect estimate.
 *
 * What the split IS supported by, on the same runs, is message preservation:
 * whole-message `pred:` spans fell from 5 of 13 to 1 of 11. That is the defect
 * it was made for. It should not be written up as improving detection coverage,
 * because on this evidence it did not.
 *
 * The refusal path the split added has no production evidence either way. The
 * counter does not exist on `slate-p-fin-02` at all -- that is why the current
 * `RunRecordSchema` cannot read those records -- and on `slate-p-fin-03`,
 * summed over all 208 records of all 16 arms, `unresolvedMentions` is 0. So its
 * cost is currently unobserved rather than bounded, and every check on it is a
 * unit test.
 *
 * NOTHING HERE CHANGED, and that is deliberate. The three rules still earn
 * their places: a mention off by an article or a possessive is an `overlap`
 * match and an `exact` miss, which is a real disagreement about a real answer
 * rather than a disagreement about a convention. Nor is the gold set touched --
 * it was blind-labelled and adjudicated against §3.1 before any of this, and
 * re-reading it to suit a new span contract is the one move that would make
 * every number below meaningless.
 *
 * What a reader MUST NOT do is treat the cross-slate comparison above as an
 * accuracy comparison. `runs/slate-p-fin-02` was taken under the one-span ask;
 * any later run is taken under the two-span one, and on `exact` and `iou50`
 * those are different questions asked of the model. The numbers are recorded
 * because "the columns are all still zero, and the one non-zero signal that
 * ever existed is gone" is a true and unflattering fact about the change, not
 * because 3 and 0 are two measurements of the same quantity.
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
 * One annotator's original call.
 *
 * WHAT IS VERBATIM AND WHAT IS NOT, because a provenance field that overstates
 * itself is worse than none. `satisfies`, `confidence`, `rationale` and
 * `quote.text` are the annotator's own returned bytes and nothing else -- the
 * predicate round's two return files are committed beside its gold
 * (`injection-p-fin-v2.predicate-annotator-{a,b}.json`) and
 * `corpus-predicate-round.test.ts` fails on any divergence between them and
 * these fields, so this is a checked claim rather than an asserted one. It is
 * checked because it was once false: the round that produced that gold shipped
 * 378 adjudicator-rewritten rationales under this schema's old promise that
 * they were carried verbatim.
 *
 * `quote.start` and `quote.end` are NOT the annotator's. Neither annotator of
 * that round returned offsets; both returned a quoted STRING, and the offsets
 * beside it were located afterwards by searching the item text. That search is
 * only sound where the quote occurs once, which the round's test asserts on all
 * 38 quotes. A round whose annotators return offsets should record those
 * instead and say so in its own record.
 *
 * `quote` is present exactly when that annotator said yes -- including on a
 * DISPUTED row, which is the one place an annotator's span survives
 * adjudication without being scoreable. That asymmetry is deliberate: the row's
 * own `spans` is emptied on a disputed row (see the schema refine) so no scorer
 * can reach it by accident, while the evidence for the dispute stays readable
 * here.
 */
const AnnotatorCallSchema = z.object({
  satisfies: z.boolean(),
  confidence: z.enum(["clear", "borderline"]),
  quote: z
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
     * The opaque queue id this row's labels came back under, and the id of the
     * round that produced them.
     *
     * OPTIONAL because two committed gold files predate them: the 13-row
     * `smoke.gold-tier2.jsonl` fixture and `injection-p-fin-v2.gold-tier2.jsonl`,
     * which is byte-pinned to its builder and cannot gain a field without
     * breaking that pin. They are NOT optional in practice on the file that has
     * them -- `corpus-predicate-round.test.ts` requires both on all 189 rows and
     * joins `rowId` through the committed queue to the corpus item, so a row
     * claiming a queue id the queue does not carry for that message fails there.
     *
     * A gold row without them is a row whose handover cannot be checked, which
     * is the state every gold file in this repository was in until the predicate
     * round: nothing on a row named its round, so two files covering the same
     * item under the same policy hash could disagree about its status and be
     * told apart only by filename. `injection-p-fin-v2.gold-tier2.jsonl` and
     * `injection-p-fin-v2.gold-tier2-predicate.jsonl` do exactly that on
     * `inj-hn08-1` -- disputed in the first, scored in the second.
     */
    rowId: z.string().min(1).optional(),
    round: z.string().min(1).optional(),
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

/** One item's findings and its gold spans, already filtered to the vocabulary. */
export interface ScoringPair {
  readonly findings: readonly SpanLike[];
  readonly goldSpans: readonly SpanLike[];
}

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
  /**
   * `tp / (tp + fn)`, or undefined when the scored items carry no gold spans at
   * all.
   *
   * DELIBERATELY NOT SYMMETRIC WITH `precision`, and the asymmetry is argued
   * here rather than assumed. Precision is undefined when the arm made no
   * claim, because its denominator is the arm's own output and an empty one
   * makes the ratio a statement about nothing. Recall's denominator is the GOLD,
   * which exists whether or not the arm answered, so `tp / (tp + fn)` stays a
   * true statement -- an arm that produced no token at tier 2 did in fact find 0
   * of the 19 gold spans it was asked about.
   *
   * The cost of that is real and is handled by `caveats` rather than by making
   * the number undefined: MEASURED over `runs/slate-rebuild-*`, six of sixteen
   * arms read `P=undefined R=0.000` because their judge produced no completed
   * call at all, beside `tier2only-Ministral` which answered 162 calls and read
   * `P=0.000 R=0.000`. In a recall COLUMN those two zeros look identical.
   * `itemsJudgeUnanswered` and its caveat are what separate them, and F1 is
   * undefined for the six, so `winnersByRule` puts them in `unrankable` rather
   * than ranking them last.
   */
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
 * Scores one set of (findings, gold spans) pairs under all three rules.
 *
 * Shared by `scoreArm` and by the trivial floors below, so the floor and the
 * arm beside it are not two implementations of "the same" matching. A floor
 * scored by a second matcher would be a different measurement wearing the same
 * column heading.
 */
export function scoreEveryRule(pairs: readonly ScoringPair[]): Record<MatchRule, RuleScore> {
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
  return byRule;
}

// ---------------------------------------------------------------------------
// The trivial floor
// ---------------------------------------------------------------------------

/**
 * A reader that answers the predicate question without reading it.
 *
 * WHY THIS IS IN THE SCORER AND NOT AN AFTERTHOUGHT. `corpus/leakage.ts`
 * already measures the tier-1 equivalent -- an orthographic oracle scored
 * exactly as an arm is -- and `corpus-v2.test.ts` calls it "the number an arm
 * has to beat". The predicate gold shipped without one, and the omission was
 * not neutral: MEASURED over `injection-p-fin-v2.gold-tier2-predicate.jsonl`'s
 * 179 scored rows, `capitalised-multiword` scores F1 0.551 under all three
 * rules and `first-capitalised-multiword` 0.571, while the best of sixteen arms
 * scores 0.197. Every arm number in that table is a factor of 2.9 BELOW a
 * regular expression, and none of the sixteen beats any of these three readers
 * under any rule. A table without this column reads as weak-but-real semantic
 * detection. It is not.
 *
 * Every reader here is a function of the MESSAGE ALONE. None is given the
 * policy, the predicate, the corpus's organisation pool, or the gold. A reader
 * that needed any of those would be an oracle and would belong in a different
 * column: for the record, one that returns every occurrence of the five names
 * `families.v2.ts` draws from scores F1 0.585 on the same rows -- higher still,
 * and not a floor, because it has been handed the answer's vocabulary.
 */
export interface TrivialFloorReader {
  readonly id: string;
  /** What it does, in one sentence, so a report can print the rule beside its score. */
  readonly what: string;
  find(text: string): SpanLike[];
}

/**
 * `\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b` -- a run of two or more capitalised words.
 *
 * Shared across calls, which is safe with `matchAll` and would NOT be with
 * `exec`. MEASURED on node 26: `[...text.matchAll(re)]` leaves `re.lastIndex`
 * at 0 -- `String.prototype.matchAll` iterates a clone -- while `re.exec(text)`
 * advances it to the end of the match, so an `exec` loop over a shared `g`
 * regex would start the second item where the first one stopped. This comment
 * replaces one that asserted the opposite mechanism without measuring it.
 */
const CAPITALISED_MULTIWORD = /\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b/g;

function capitalisedMultiword(text: string): SpanLike[] {
  const out: SpanLike[] = [];
  for (const m of text.matchAll(CAPITALISED_MULTIWORD)) {
    if (m.index === undefined) continue;
    out.push({ start: m.index, end: m.index + m[0].length });
  }
  return out;
}

export const TRIVIAL_FLOOR_READERS: readonly TrivialFloorReader[] = [
  {
    id: "capitalised-multiword",
    what: "every run of two or more capitalised words in the message",
    find: capitalisedMultiword,
  },
  {
    id: "first-capitalised-multiword",
    what: "the FIRST run of two or more capitalised words, and nothing else",
    find: (text) => capitalisedMultiword(text).slice(0, 1),
  },
  {
    id: "whole-message",
    what: "one span covering the entire message",
    find: (text) => (text.length > 0 ? [{ start: 0, end: text.length }] : []),
  },
];

export interface TrivialFloor {
  readonly reader: string;
  readonly what: string;
  readonly findings: number;
  readonly byRule: Readonly<Record<MatchRule, RuleScore>>;
}

/** One item as a floor reader sees it: the message, and the gold spans on it. */
export interface FloorItem {
  readonly text: string;
  readonly goldSpans: readonly SpanLike[];
}

/** Scores every reader in `TRIVIAL_FLOOR_READERS` over the same items an arm was scored on. */
export function scoreTrivialFloors(items: readonly FloorItem[]): TrivialFloor[] {
  return TRIVIAL_FLOOR_READERS.map((reader) => {
    let findings = 0;
    const pairs = items.map((item) => {
      const found = reader.find(item.text);
      findings += found.length;
      return { findings: found, goldSpans: item.goldSpans };
    });
    return { reader: reader.id, what: reader.what, findings, byRule: scoreEveryRule(pairs) };
  });
}

/**
 * The best defined floor F1 under one rule, and which readers reach it.
 *
 * Undefined only when no floor has a defined F1 under that rule, which on a
 * gold set carrying any span cannot happen: `whole-message` always emits.
 */
export function bestFloorF1(floors: readonly TrivialFloor[], rule: MatchRule): number | undefined {
  const defined = floors.map((f) => f.byRule[rule].f1).filter((f): f is number => f !== undefined);
  return defined.length === 0 ? undefined : Math.max(...defined);
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
   * In-vocabulary findings this arm made on DISPUTED rows, which the exclusion
   * removed from every ratio.
   *
   * `goldRowsDisputed` counts rows and reads like a symmetric trim of the gold.
   * On `injection-p-fin-v2.gold-tier2-predicate.jsonl` it is not symmetric:
   * MEASURED, all ten disputed rows are negatives carrying no gold span, so the
   * exclusion can remove false positives and nothing else. Over
   * `runs/slate-rebuild-*` it hides 8 of `tier2-Qwen3-4B`'s 50 findings and
   * moves its overlap precision from 0.120 to 0.143 -- a 19% relative
   * improvement produced by the exclusion rather than by the arm. A reader given
   * only the row count cannot size that; this is the number that sizes it.
   */
  readonly findingsOnDisputedRows: number;
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
  /**
   * The trivial floors, scored over EXACTLY the rows this arm was scored on --
   * the same items, the same gold spans, the same matcher, the same three rules.
   *
   * Beside `byRule` and not in a separate report, because the failure this
   * prevents is a table of arm F1s published without the one number that says
   * what they are worth. `caveats` names any rule where this arm does not beat
   * it.
   */
  readonly floors: readonly TrivialFloor[];
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
  records: readonly ScoreableRecord[],
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

/**
 * The fields a scorer actually reads off a row, as a STRUCTURAL type.
 *
 * Two record shapes reach this module and neither is a subtype of the other:
 * `RunRecord` (apps/eval/src/driver/record.ts), written by the in-browser arms,
 * and `CeilingRecord` (apps/eval/src/driver/ceiling.ts), written by the hosted
 * capability-ceiling arm. They differ in the transport telemetry they carry --
 * one has `backend` and `tier2Stats`, the other has `provider` and per-call
 * TTFT -- and agree on everything a SCORE depends on.
 *
 * Widening the signature to this interface, rather than making the ceiling arm
 * emit `RunRecord`s, is the choice `ceiling.ts`'s header argues for at length:
 * a ceiling row cannot honestly claim a `backend` of "wasm" or "webgpu", nor a
 * `detector` of "core-orchestrator" or "approach-b". So the scorer names what
 * it needs and both shapes satisfy it -- one gold, one matcher, one set of
 * floors, and no row asserting a configuration that did not run.
 *
 * `calls` is the ceiling arm's own answered-call list and is how
 * `itemsJudgeAnswered` stays meaningful across both shapes. A `RunRecord` has
 * no such top-level field, so it contributes nothing there and the two
 * tier-specific stats objects still carry it, exactly as before.
 */
export interface ScoreableRecord {
  readonly arm: string;
  readonly itemId: string;
  readonly policyHash: string;
  readonly text: string;
  readonly findings: readonly RecordFinding[];
  readonly error: string | null;
  readonly degraded?: readonly { readonly tier: number; readonly reason: string }[] | undefined;
  readonly tier2Stats?: { readonly calls: readonly unknown[] } | undefined;
  readonly baselineStats?: { readonly calls: readonly unknown[] } | undefined;
  readonly calls?: readonly unknown[] | undefined;
}

export interface ScoreArmInput {
  /** One arm's records. Every record must name the same `arm`. */
  readonly records: readonly ScoreableRecord[];
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

  const recordsById = new Map<string, ScoreableRecord>();
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
  /** The same items, as a floor reader sees them: the message and its gold spans. */
  const floorItems: FloorItem[] = [];
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

    // Across both record shapes: the two in-browser stats objects, and the
    // ceiling arm's own per-call list. A row with none of them is an arm that
    // never got an answer, which is exactly what `itemsJudgeUnanswered` means.
    const answered =
      (record.tier2Stats?.calls.length ?? 0) +
      (record.baselineStats?.calls.length ?? 0) +
      (record.calls?.length ?? 0);
    if (answered > 0) itemsJudgeAnswered += 1;

    pairs.push({
      findings: inVocab.map((f) => ({ start: f.start, end: f.end })),
      goldSpans: row.spans.map((s) => ({ start: s.start, end: s.end })),
    });
    floorItems.push({
      text: record.text,
      goldSpans: row.spans.map((s) => ({ start: s.start, end: s.end })),
    });
  }

  // What the exclusion actually removed FROM THIS ARM. `goldRowsDisputed` is a
  // count of rows and reads like a two-sided trim; on the gold that motivated
  // this it is not one. See `findingsOnDisputedRows`.
  let findingsOnDisputedRows = 0;
  for (const row of gold) {
    if (row.status !== "disputed") continue;
    const record = recordsById.get(row.itemId);
    if (record === undefined) continue;
    findingsOnDisputedRows += record.findings.filter((f) => entityTypes.has(f.entityType)).length;
  }

  let recordsWithoutGold = 0;
  for (const record of records) if (!goldById.has(record.itemId)) recordsWithoutGold += 1;

  const byRule = scoreEveryRule(pairs);
  const floors = scoreTrivialFloors(floorItems);

  const itemsJudgeUnanswered = recordsScored - itemsJudgeAnswered;
  const caveats: string[] = [];
  if (disputed > 0) {
    const disputedRows = gold.filter((g) => g.status === "disputed");
    const oneSided = disputedRows.every((g) => !g.satisfies && g.spans.length === 0);
    caveats.push(
      `${disputed} of ${gold.length} gold rows are adjudicated disputed and are excluded from both the ` +
        `numerator and the denominator of every metric here` +
        (oneSided
          ? `, and on this gold that exclusion is ONE-SIDED: every disputed row is a negative carrying no ` +
            `gold span, so it can only remove false positives and never a miss. It removed ` +
            `${findingsOnDisputedRows} of this arm's ${findingsInVocabulary + findingsOnDisputedRows} ` +
            `in-vocabulary findings, all of which would have been false positives`
          : ""),
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
  // The floor is a caveat and not a footnote: an arm that does not beat it has
  // not been shown to be reading anything. Stated per rule, because an arm can
  // clear it under one rule and not another.
  for (const rule of MATCH_RULES) {
    const best = bestFloorF1(floors, rule);
    const mine = byRule[rule].f1;
    if (best === undefined) continue;
    const winner = floors.find((f) => f.byRule[rule].f1 === best)!;
    if (mine === undefined) {
      caveats.push(
        `this arm has no defined F1 under ${rule}, so it cannot be compared with the trivial floor, which ` +
          `scores ${best.toFixed(3)} there ("${winner.what}")`,
      );
    } else if (mine <= best) {
      caveats.push(
        `under ${rule} this arm's F1 (${mine.toFixed(3)}) DOES NOT BEAT the trivial floor (${best.toFixed(3)}, ` +
          `"${winner.what}"), a reader given neither the policy nor the predicate; the number below is not ` +
          `evidence that this arm read the message`,
      );
    }
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
      findingsOnDisputedRows,
      caveats,
    },
    byRule,
    floors,
  };
}

/** Splits a multi-arm run into one `ScoredArm` per arm, sorted by arm name. */
export function scoreArms(
  records: readonly ScoreableRecord[],
  gold: readonly Tier2GoldRow[],
): ScoredArm[] {
  const byArm = new Map<string, ScoreableRecord[]>();
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
