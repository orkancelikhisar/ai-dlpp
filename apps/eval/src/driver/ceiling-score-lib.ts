/**
 * The decisions behind `pnpm -C apps/eval ceiling:score`, separated from the
 * `console.log` calls that render them.
 *
 * `ceiling-score.ts` used to inline all of this between its print statements
 * and end in a bare `main()`, exporting nothing. Nothing in the workspace could
 * import it, so nothing tested it: a mutation review planted mutants across the
 * file -- disputed gold rows re-admitted into the positive count, the span
 * pairing allowed to reuse one gold span, the 512-token comparison column
 * retargeted at 600 -- and every one of them survived a green suite. Splitting
 * the file is what makes those killable. The renderer above stays a renderer:
 * if a decision moves back into it, it stops being covered.
 *
 * Two things here are NEW behaviour rather than moved behaviour, and both are
 * additive columns beside the existing ones, never replacements:
 *
 *   - ATTEMPTED-ONLY scoring (`attemptedRecords`). 68 of 3,780 rows across
 *     passes 1-2 carry `calls: []` and a 429: the provider rate-limited the
 *     item away and the model never saw it. Scored whole-gold, those rows are
 *     indistinguishable from items the model read and missed.
 *   - The decode-window guard (`decodeColumn`). See its own docblock.
 */
import { measureOrthography } from "../corpus/leakage.js";
import { CeilingRecordSchema, type CeilingRecord } from "./ceiling.js";
import {
  bestFloorF1,
  scoreArms,
  spansMatch,
  type MatchRule,
  type ScoreableRecord,
  type ScoredArm,
  type SpanLike,
  type Tier2GoldRow,
} from "./score.js";

/**
 * `DEFAULT_TIER2_CONFIG.maxTokens` (packages/tier2/src/manifest.ts:116).
 *
 * The ceiling arms ran at 600, the browser arms at 512. The `calls >=512`
 * column below is how a reader checks whether that 88-token asymmetry could
 * have changed anything for a given arm: a 0 there means running at 512 would
 * have produced identical output.
 */
export const LOCAL_ARM_MAX_TOKENS = 512;

/** The `max_tokens` the ceiling arms themselves ran at. Named so the two caps cannot be confused. */
export const CEILING_ARM_MAX_TOKENS = 600;

/**
 * Below this median decode window, `decode tok/s` is suppressed: at that scale
 * the number says when a server-sent-event frame happened to arrive, not how
 * fast the model decodes.
 *
 * MEASURED on this arm's own artifacts (`runs/*.ceiling-*.jsonl`, passes 1-2
 * plus `glmon-01`), not assumed. `ceiling-judge-deepseek-v4-flash-0731
 * [ceiling-01]` has a median decode window of 9.6 ms; **145 of its 185
 * streaming calls have a window under 20 ms**, 17 under 5 ms, 6 under 1 ms, the
 * shortest is 0.174 ms, and the largest rate the formula returns for it is
 * 40,327 tok/s for a 7-token answer. Its per-call rate runs p10/p50/p90 =
 * 126/727/1371 tok/s -- an 11x spread across one model answering one task.
 *
 * 100 ms is a judgement, not a measured breakpoint, and it is stated here
 * rather than buried so a reader can move it: `decodeWindowMsP50` is printed in
 * the table beside the rate for exactly that reason. What the measurements DO
 * establish is that the threshold has to be on the window and cannot be on the
 * family: at 100 ms it suppresses eight arm-passes, all of them judge arms, and
 * spares `judge-nemotron-3-super-120b-a12b` (shortest window 66.1 ms) and
 * `judge-glm-5.3-flash` (median 181.5 ms, shortest 14.4 ms), whose windows are
 * sound.
 */
export const MIN_DECODE_WINDOW_MS = 100;

/**
 * The `1/(n-1)` sensitivity above which the rate is flagged.
 *
 * A decode rate over `n` completion tokens rests on `n - 1` token intervals, so
 * one mis-timed interval moves it by `1/(n-1)`. That is arithmetic, not an
 * estimate: 5% is n = 21. On this arm's data the split is total -- the
 * Approach-B arms sit at n = 55-119 (1.9% down to 0.9%) and
 * `judge-glm-5.3-flash` at n = 191 (0.5%), while every thinking-off judge arm
 * sits at n = 5-7 (16.7% to 25%).
 *
 * The same quantity was, until `ceiling.ts` was corrected on 2026-09-08, also
 * the EXACT overstatement of the rate: the old formula divided `n` tokens by
 * the window that opens when the FIRST one arrives. Every row now in `runs/`
 * predates that fix and carries the overstated figure, which is why the marker
 * is worth printing on this data and not only in principle. It stays correct
 * afterwards, because a rate resting on six intervals is a coarse measurement
 * whichever numerator it uses.
 *
 * SEPARATE from `MIN_DECODE_WINDOW_MS` because the two are independent and the
 * data proves it: `judge-nemotron` has a sound window (shortest 66.1 ms) and
 * n = 7, while `judge-glm-5.3-flash` has a 181.5 ms window and n = 191. A
 * single guard on either quantity alone gets one of those two arms wrong.
 */
export const MAX_DECODE_BIAS = 0.05;

export function fmt(n: number | undefined, digits = 3): string {
  return n === undefined ? "—" : n.toFixed(digits);
}

/** The percentile convention used by every latency column here. */
export function percentile(xs: readonly number[], q: number): number | undefined {
  if (xs.length === 0) return undefined;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

export function loadCeilingRecords(jsonl: string): CeilingRecord[] {
  const out: CeilingRecord[] = [];
  for (const line of jsonl.split("\n")) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    out.push(CeilingRecordSchema.parse(JSON.parse(trimmed)));
  }
  return out;
}

export interface ArmRows {
  readonly label: string;
  readonly kind: "local" | "ceiling";
  readonly records: ScoreableRecord[];
  readonly ceiling: CeilingRecord[];
}

/** The label a ceiling arm-pass is listed under. Taken from the RECORD, never the filename. */
export function ceilingArmLabel(r: CeilingRecord): string {
  // A thinking-ON row is a DIFFERENT CONDITION from every other row in these
  // tables and is labelled so it can never be read as one of them. Taken from
  // the RECORD, not from the filename: the filename is a naming choice, the
  // field is what was asked of the model.
  return `${r.arm} [${r.runId}]${r.thinkingRequested === "on" ? " **thinking ON**" : ""}`;
}

/** The key an arm-pass is bucketed under: `runId` as well as arm, so three passes are three columns. */
export function ceilingArmKey(r: CeilingRecord): string {
  return `ceiling::${r.runId}::${r.arm}`;
}

/**
 * Whether the arm produced any completed call for this item.
 *
 * The same union `scoreArm` uses for `itemsJudgeAnswered` -- the two in-browser
 * stats objects and the ceiling arm's own per-call list -- so the
 * `answered/scored` column and the attempted-only tables cannot come to hold
 * two definitions of "answered".
 */
export function recordWasAttempted(record: ScoreableRecord): boolean {
  return (
    (record.tier2Stats?.calls.length ?? 0) +
      (record.baselineStats?.calls.length ?? 0) +
      (record.calls?.length ?? 0) >
    0
  );
}

/**
 * The rows an arm actually answered.
 *
 * Feeding these to `scoreArm` is what makes attempted-only scoring
 * like-for-like rather than an arm adjusted against an unadjusted floor: a gold
 * row with no record becomes `goldRowsWithoutRecord`, which `scoreArm` excludes
 * from every numerator and denominator AND from the `floorItems` it hands to
 * `scoreTrivialFloors`. So the floors are re-scored over exactly the same
 * subset as the arm, which is the correction
 * `docs/research/2026-09-07-ceiling-arm.md` Sec 7.4 asks for and explicitly
 * does not itself make.
 */
export function attemptedRecords(records: readonly ScoreableRecord[]): ScoreableRecord[] {
  return records.filter((r) => recordWasAttempted(r));
}

export interface UnansweredCounts {
  /** Scored gold rows the arm has a record for but produced no completed call on. */
  readonly unanswered: number;
  /** Of those, ones the gold marks as positives -- recall the arm was never given a chance at. */
  readonly unansweredPositives: number;
}

/** What an arm lost to the provider, counted against the gold rather than against the run. */
export function unansweredCounts(
  records: readonly ScoreableRecord[],
  gold: readonly Tier2GoldRow[],
): UnansweredCounts {
  const byId = new Map(records.map((r) => [r.itemId, r]));
  let unanswered = 0;
  let unansweredPositives = 0;
  for (const row of gold) {
    if (row.status !== "scored") continue;
    const record = byId.get(row.itemId);
    if (record === undefined || recordWasAttempted(record)) continue;
    unanswered += 1;
    if (row.satisfies) unansweredPositives += 1;
  }
  return { unanswered, unansweredPositives };
}

export interface GoldSummary {
  readonly rows: number;
  readonly scored: number;
  readonly disputed: number;
  readonly positives: number;
  readonly goldSpans: number;
}

/**
 * The gold's own shape, counted once.
 *
 * `positives` and `goldSpans` count SCORED rows only. A disputed row is
 * excluded from every numerator and denominator `scoreArm` computes, so
 * counting one here would print a recall denominator no arm is ever scored
 * against.
 */
export function summarizeGold(gold: readonly Tier2GoldRow[]): GoldSummary {
  const scored = gold.filter((g) => g.status === "scored");
  return {
    rows: gold.length,
    scored: scored.length,
    disputed: gold.length - scored.length,
    positives: scored.filter((g) => g.satisfies).length,
    goldSpans: scored.reduce((n, g) => n + g.spans.length, 0),
  };
}

export interface SpanCounts {
  readonly tp: number;
  readonly fp: number;
  readonly fn: number;
}

/**
 * Greedy one-to-one pairing of an item's findings against its gold spans.
 *
 * ONE-TO-ONE is the load-bearing word. `takenGold` is what stops two findings
 * on the same gold span from scoring two true positives, which would reward an
 * arm for emitting the same span twice and let precision exceed what the gold
 * can support. Uses `score.ts`'s OWN `spansMatch` rather than a second
 * inequality written here: the two tables must not be able to disagree about
 * what a match is, and an inlined `a.start < b.end && b.start < a.end` is
 * precisely how they would come to.
 */
export function pairSpansGreedy(
  rule: MatchRule,
  findings: readonly SpanLike[],
  gold: readonly SpanLike[],
): SpanCounts {
  const takenGold = new Set<number>();
  let matched = 0;
  for (const f of findings) {
    for (let g = 0; g < gold.length; g++) {
      if (takenGold.has(g)) continue;
      if (spansMatch(rule, f, gold[g]!)) {
        takenGold.add(g);
        matched += 1;
        break;
      }
    }
  }
  return { tp: matched, fp: findings.length - matched, fn: gold.length - matched };
}

export interface PrecisionRecallF1 {
  readonly precision: number | undefined;
  readonly recall: number | undefined;
  readonly f1: number | undefined;
}

export function prf1(counts: SpanCounts): PrecisionRecallF1 {
  const { tp, fp, fn } = counts;
  const precision = tp + fp === 0 ? undefined : tp / (tp + fp);
  const recall = tp + fn === 0 ? undefined : tp / (tp + fn);
  const f1 =
    precision === undefined || recall === undefined || precision + recall === 0
      ? undefined
      : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1 };
}

/**
 * How the `decode tok/s` cell renders for one arm, and why.
 *
 * Returns the value only when the arm's median decode window clears
 * `MIN_DECODE_WINDOW_MS`. Below it the cell carries the window instead of a
 * number: printing a suppressed column as a bare dash would hide the reason,
 * and the reason is the only thing that lets a reader disagree.
 */
export interface DecodeColumn {
  readonly tokPerSecP50: number | undefined;
  readonly windowMsP50: number | undefined;
  /** Streaming calls whose window is under a fifth of the threshold. The evidence for suppression. */
  readonly shortWindowCalls: number;
  readonly streamingCalls: number;
  readonly suppressed: boolean;
  /**
   * `1/(n-1)` at the median completion: how far one mis-timed token interval
   * moves the rate. On every row now in `runs/` it is also the exact
   * overstatement of the pre-2026-09-08 formula. See `MAX_DECODE_BIAS`.
   */
  readonly biasAtMedianN: number | undefined;
  readonly biased: boolean;
}

export function decodeColumn(calls: readonly CeilingRecord["calls"][number][]): DecodeColumn {
  // A call with one token or fewer has NO decode window to measure: under the
  // corrected `(n-1)/window` formula its rate is undefined, and the rows on
  // disk were written before that correction. Excluded here so the percentile
  // does not carry a value that the corrected driver will not produce.
  const streaming = calls.filter(
    (c) => c.ttftMs !== null && c.decodeTokPerSec !== null && (c.completionTokens ?? 0) > 1,
  );
  const windows = streaming.map((c) => c.wallMs - c.ttftMs!);
  const windowMsP50 = percentile(windows, 0.5);
  const tokPerSecP50 = percentile(
    streaming.map((c) => c.decodeTokPerSec!),
    0.5,
  );
  const medianN = percentile(
    streaming.map((c) => c.completionTokens!),
    0.5,
  );
  const biasAtMedianN = medianN === undefined || medianN <= 1 ? undefined : 1 / (medianN - 1);
  const suppressed = windowMsP50 !== undefined && windowMsP50 < MIN_DECODE_WINDOW_MS;
  return {
    tokPerSecP50: suppressed ? undefined : tokPerSecP50,
    windowMsP50,
    shortWindowCalls: windows.filter((w) => w < MIN_DECODE_WINDOW_MS / 5).length,
    streamingCalls: streaming.length,
    suppressed,
    biasAtMedianN,
    biased: biasAtMedianN !== undefined && biasAtMedianN > MAX_DECODE_BIAS,
  };
}

export interface TransportRow {
  readonly label: string;
  readonly providers: string[];
  readonly pin: string;
  readonly pinHonoured: boolean;
  readonly calls: number;
  readonly reasoningP50: number | undefined;
  readonly reasoningMax: number | undefined;
  readonly completionP50: number | undefined;
  readonly completionMax: number | undefined;
  readonly truncated: number;
  readonly atLocalCap: number;
  readonly ttftP50: number | undefined;
  readonly ttftP95: number | undefined;
  readonly fellBack: number;
  readonly decode: DecodeColumn;
  readonly wallP50: number | undefined;
  readonly wallP95: number | undefined;
  readonly parseFailures: number;
  readonly repairs: number;
  readonly rateLimited: number;
  readonly otherRetries: number;
  readonly itemWallP50: number | undefined;
  readonly costUsd: number;
}

/** Every transport figure for one ceiling arm-pass. Returns undefined when the arm made no call. */
export function transportRow(arm: ArmRows): TransportRow | undefined {
  const calls = arm.ceiling.flatMap((r) => r.calls);
  if (calls.length === 0) return undefined;
  const providers = [...new Set(calls.map((c) => c.provider).filter((p): p is string => p !== null))];
  const pin = arm.ceiling[0]!.requestedProvider;
  const reasoning = calls.map((c) => c.reasoningTokens).filter((n): n is number => n !== null);
  // A `non-stream-fallback` call has null ttftMs BY CONSTRUCTION -- there is no
  // first-token event on that path. It is EXCLUDED here rather than
  // substituted: folding its whole wall time in as a TTFT would report the
  // fallback's total latency as a first-token latency, and the `non-stream`
  // column exists so a reader can see how many were dropped.
  const ttft = calls.map((c) => c.ttftMs).filter((n): n is number => n !== null);
  const wall = calls.map((c) => c.wallMs);
  const completion = calls.map((c) => c.completionTokens).filter((n): n is number => n !== null);
  const allRetries = calls.flatMap((c) => c.retries);
  // 429s are broken out from other retries because they are the ones that
  // contaminate a naive latency reading: they are the provider throttling, not
  // the model thinking. They cost nothing (a 429 is unbilled) and they change
  // no finding, so they belong beside the latency columns and not in the
  // accuracy discussion.
  const rateLimited = allRetries.filter((r) => r.status === 429).length;
  return {
    label: arm.label,
    providers,
    pin,
    pinHonoured: providers.every((p) => p === pin),
    calls: calls.length,
    reasoningP50: percentile(reasoning, 0.5),
    reasoningMax: reasoning.length > 0 ? Math.max(...reasoning) : undefined,
    completionP50: percentile(completion, 0.5),
    completionMax: completion.length > 0 ? Math.max(...completion) : undefined,
    // `finish_reason: "length"` is the provider stating that IT cut the answer
    // off. An arm with a nonzero count here is not measured cleanly at this
    // cap, and its F1 is "this model with N% of its answers truncated".
    truncated: calls.filter((c) => c.finishReason === "length").length,
    // How many calls would have been cut short at the LOCAL arms' 512 but were
    // not here. Zero means the 88-token budget asymmetry was inert for this arm
    // -- the strongest available statement that the two caps are comparable.
    // Compared against the LOCAL cap, never against this arm's own 600: at 600
    // the column would count calls that were not truncated anywhere and answer
    // no question.
    atLocalCap: completion.filter((n) => n >= LOCAL_ARM_MAX_TOKENS).length,
    ttftP50: percentile(ttft, 0.5),
    ttftP95: percentile(ttft, 0.95),
    fellBack: calls.filter((c) => c.transport === "non-stream-fallback").length,
    decode: decodeColumn(calls),
    wallP50: percentile(wall, 0.5),
    wallP95: percentile(wall, 0.95),
    parseFailures: arm.ceiling.reduce((n, r) => n + r.parseFailures, 0),
    repairs: arm.ceiling.reduce((n, r) => n + r.repairs, 0),
    rateLimited,
    otherRetries: allRetries.length - rateLimited,
    // The record's own top-level wallMs: end to end, backoff included. Printed
    // beside the per-attempt figures so the gap between them is visible.
    itemWallP50: percentile(
      arm.ceiling.map((r) => r.wallMs),
      0.5,
    ),
    costUsd: calls.reduce((n, c) => n + (c.costUsd ?? 0), 0),
  };
}

export interface PredicateArmLine {
  readonly label: string;
  readonly kind: string;
  readonly f1: number | undefined;
  readonly scored: ScoredArm;
}

/**
 * Scores every arm over one predicate's gold rows, and the floors beside them.
 *
 * `floors` is the FIRST scored arm's, deliberately: on the whole gold every arm
 * carries a record for all 179 scored rows, so `scoreTrivialFloors` sees the
 * same items whichever arm supplies them, and taking the first states that the
 * floor is one number rather than a per-arm one. Attempted-only scoring is the
 * case where that stops holding -- there the subsets differ by arm, so
 * `scorePredicate` is called per arm and each carries its own floor.
 */
export function scorePredicate(
  arms: Iterable<ArmRows>,
  rows: readonly Tier2GoldRow[],
): { lines: PredicateArmLine[]; floors: ScoredArm["floors"] | undefined } {
  let floors: ScoredArm["floors"] | undefined;
  const lines: PredicateArmLine[] = [];
  for (const arm of arms) {
    const scored = scoreArms(arm.records, rows)[0];
    if (scored === undefined) continue;
    floors ??= scored.floors;
    lines.push({ label: arm.label, kind: arm.kind, f1: scored.byRule.overlap.f1, scored });
  }
  lines.sort((a, b) => (b.f1 ?? -1) - (a.f1 ?? -1) || a.label.localeCompare(b.label));
  return { lines, floors };
}

export interface AttemptedOnlyRow {
  readonly label: string;
  readonly kind: string;
  readonly attempted: number;
  readonly unanswered: number;
  readonly unansweredPositives: number;
  /** Scored over the whole gold: the published number. */
  readonly wholeGoldF1: number | undefined;
  /** Scored over the rows this arm answered, WITH the floors re-scored over the same rows. */
  readonly attemptedF1: number | undefined;
  readonly attemptedBestFloorF1: number | undefined;
  readonly wholeGoldBestFloorF1: number | undefined;
}

/**
 * The attempted-only table: each arm, and every floor, over the intersection of
 * items that arm answered.
 *
 * Reported BESIDE the whole-gold columns and never instead of them. The
 * unanswered rows are a real cost of running against a rate-limited hosted
 * provider; defining them away would turn a provider's throttling into a model
 * result.
 */
export function attemptedOnlyTable(
  arms: Iterable<ArmRows>,
  rows: readonly Tier2GoldRow[],
  rule: MatchRule,
): AttemptedOnlyRow[] {
  const out: AttemptedOnlyRow[] = [];
  for (const arm of arms) {
    const whole = scoreArms(arm.records, rows)[0];
    if (whole === undefined) continue;
    const counts = unansweredCounts(arm.records, rows);
    const attempted = attemptedRecords(arm.records);
    const scoped = attempted.length === 0 ? undefined : scoreArms(attempted, rows)[0];
    out.push({
      label: arm.label,
      kind: arm.kind,
      attempted: attempted.length,
      unanswered: counts.unanswered,
      unansweredPositives: counts.unansweredPositives,
      wholeGoldF1: whole.byRule[rule].f1,
      attemptedF1: scoped?.byRule[rule].f1,
      attemptedBestFloorF1: scoped === undefined ? undefined : bestFloorF1(scoped.floors, rule),
      wholeGoldBestFloorF1: bestFloorF1(whole.floors, rule),
    });
  }
  out.sort((a, b) => (b.attemptedF1 ?? -1) - (a.attemptedF1 ?? -1) || a.label.localeCompare(b.label));
  return out;
}

export interface SpanArmRow {
  readonly label: string;
  readonly kind: string;
  readonly findings: number;
  readonly counts: SpanCounts;
  readonly scores: PrecisionRecallF1;
}

/** The span-level row for one arm against the labelled corpus's entity gold. */
export function scoreSpanArm(
  arm: ArmRows,
  goldById: ReadonlyMap<string, readonly { start: number; end: number }[]>,
  entityTypes: ReadonlySet<string>,
): SpanArmRow | undefined {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let found = 0;
  let sawAny = false;
  for (const record of arm.records) {
    const gold = goldById.get(record.itemId);
    if (gold === undefined) continue;
    const mine = record.findings.filter((f) => entityTypes.has(f.entityType));
    if (mine.length > 0) sawAny = true;
    found += mine.length;
    const counts = pairSpansGreedy("overlap", mine, gold);
    tp += counts.tp;
    fp += counts.fp;
    fn += counts.fn;
  }
  if (!sawAny) return undefined;
  const counts = { tp, fp, fn };
  return { label: arm.label, kind: arm.kind, findings: found, counts, scores: prf1(counts) };
}

/** A local arm whose spans are mostly tier 0's regex output, not its model's. */
export function isTierAssisted(label: string): boolean {
  return /^tier2-|\+tier0-/.test(label);
}

export { bestFloorF1, measureOrthography };
