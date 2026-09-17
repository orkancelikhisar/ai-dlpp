/**
 * Scoring for a judgment arm, which is a different exercise from scoring a
 * generative one.
 *
 * Every other arm produced one decision per message, so one run gave one point:
 * P, R, F1 and nothing else. This arm produces a PROBABILITY per message, so one
 * run gives a curve. What that buys, and what this file computes:
 *
 *   - The whole threshold sweep, so "does it beat the 0.776 no-model floor" can
 *     be answered at the default 0.5 AND at the threshold that suits the cost of
 *     each error, instead of at whichever point the decoder happened to land on.
 *   - A SPLIT-HALF estimate beside the best-threshold one. Picking the threshold
 *     on the same rows you then score is fitting on test data; the honest number
 *     picks it on one half and scores the other. Both are reported, labelled.
 *   - Calibration: does a 0.7 mean seven in ten? A generative arm has no such
 *     question, and a caller who routes on probability needs the answer.
 *   - The filter effect: candidates come from tier 0 and the orthographic
 *     oracle, so the model's only entity job is to KEEP or REJECT them. Counting
 *     correct and wrongful rejections says exactly what the judgment adds over
 *     the regex tier, which is the question the paper could only answer
 *     indirectly.
 *
 * Floors are read from docs/paper/data/numbers.json, so this arm is compared
 * against the same numbers the paper prints rather than against numbers retyped
 * here.
 */
import { pairSpansGreedy } from "./ceiling-score-lib.js";
import { NOT_CONFIDENTIAL, projectFindings, type ProjectedFinding, type TypeSafeRecord } from "./typesafe.js";
import type { PolicyIr } from "@sih/core";

export interface PredicateGold {
  readonly satisfies: ReadonlyMap<string, boolean>;
  readonly scored: number;
  readonly positives: number;
}

export function loadPredicateGold(jsonl: string, predicateId: string): PredicateGold {
  const satisfies = new Map<string, boolean>();
  let positives = 0;
  for (const line of jsonl.split("\n")) {
    if (line.trim() === "") continue;
    const row = JSON.parse(line) as { itemId?: string; predicateId?: string; status?: string; satisfies?: unknown };
    if (row.predicateId !== predicateId || row.status !== "scored" || typeof row.satisfies !== "boolean") continue;
    if (row.itemId === undefined) continue;
    satisfies.set(row.itemId, row.satisfies);
    if (row.satisfies) positives += 1;
  }
  return { satisfies, scored: satisfies.size, positives };
}

export interface SweepPoint {
  readonly threshold: number;
  readonly tp: number;
  readonly fp: number;
  readonly fn: number;
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
}

export function prf1(tp: number, fp: number, fn: number, threshold: number): SweepPoint {
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { threshold, tp, fp, fn, precision, recall, f1 };
}

/** Rows this arm actually answered AND the gold scored, paired up. */
export interface ScoredRow {
  readonly itemId: string;
  readonly probability: number;
  readonly label: boolean;
}

export function pairRows(records: readonly TypeSafeRecord[], gold: PredicateGold): { rows: ScoredRow[]; unanswered: string[] } {
  const rows: ScoredRow[] = [];
  const unanswered: string[] = [];
  for (const r of records) {
    const label = gold.satisfies.get(r.itemId);
    if (label === undefined) continue;
    if (r.predicateProbability === null) {
      unanswered.push(r.itemId);
      continue;
    }
    rows.push({ itemId: r.itemId, probability: r.predicateProbability, label });
  }
  return { rows, unanswered };
}

export function pointAt(rows: readonly ScoredRow[], threshold: number): SweepPoint {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (const row of rows) {
    const fires = row.probability >= threshold;
    if (fires && row.label) tp += 1;
    else if (fires && !row.label) fp += 1;
    else if (!fires && row.label) fn += 1;
  }
  return prf1(tp, fp, fn, threshold);
}

/** Every distinct probability is a threshold, plus 0.5, so no maximum is missed between grid steps. */
export function sweep(rows: readonly ScoredRow[]): SweepPoint[] {
  const thresholds = new Set<number>([0.5]);
  for (const r of rows) thresholds.add(r.probability);
  return [...thresholds].sort((a, b) => a - b).map((t) => pointAt(rows, t));
}

export function bestPoint(points: readonly SweepPoint[]): SweepPoint | undefined {
  return points.reduce<SweepPoint | undefined>((best, p) => (best === undefined || p.f1 > best.f1 ? p : best), undefined);
}

/** Mann-Whitney: P(a positive outranks a negative), ties counted as half. */
export function rocAuc(rows: readonly ScoredRow[]): number | null {
  const pos = rows.filter((r) => r.label).map((r) => r.probability);
  const neg = rows.filter((r) => !r.label).map((r) => r.probability);
  if (pos.length === 0 || neg.length === 0) return null;
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

export function averagePrecision(rows: readonly ScoredRow[]): number | null {
  const sorted = [...rows].sort((a, b) => b.probability - a.probability);
  const positives = sorted.filter((r) => r.label).length;
  if (positives === 0) return null;
  let tp = 0;
  let sum = 0;
  sorted.forEach((row, i) => {
    if (!row.label) return;
    tp += 1;
    sum += tp / (i + 1);
  });
  return sum / positives;
}

export interface CalibrationBin {
  readonly lower: number;
  readonly upper: number;
  readonly n: number;
  readonly meanProbability: number | null;
  readonly observedRate: number | null;
}

export function calibration(rows: readonly ScoredRow[], bins = 10): CalibrationBin[] {
  return Array.from({ length: bins }, (_, i) => {
    const lower = i / bins;
    const upper = (i + 1) / bins;
    const inBin = rows.filter((r) => (i === bins - 1 ? r.probability >= lower && r.probability <= upper : r.probability >= lower && r.probability < upper));
    return {
      lower,
      upper,
      n: inBin.length,
      meanProbability: inBin.length === 0 ? null : inBin.reduce((a, r) => a + r.probability, 0) / inBin.length,
      observedRate: inBin.length === 0 ? null : inBin.filter((r) => r.label).length / inBin.length,
    };
  });
}

/** Deterministic, content-addressed split: the same item lands in the same half on every machine. */
export function halfOf(itemId: string): 0 | 1 {
  let h = 5381;
  for (let i = 0; i < itemId.length; i++) h = ((h << 5) + h + itemId.charCodeAt(i)) >>> 0;
  return (h & 1) as 0 | 1;
}

export interface SplitHalf {
  readonly tauFromA: number | null;
  readonly f1OnB: number | null;
  readonly tauFromB: number | null;
  readonly f1OnA: number | null;
  readonly mean: number | null;
}

/** The threshold is chosen on one half and spent on the other, both ways round. */
export function splitHalf(rows: readonly ScoredRow[]): SplitHalf {
  const a = rows.filter((r) => halfOf(r.itemId) === 0);
  const b = rows.filter((r) => halfOf(r.itemId) === 1);
  const tauA = bestPoint(sweep(a))?.threshold ?? null;
  const tauB = bestPoint(sweep(b))?.threshold ?? null;
  const f1OnB = tauA === null || b.length === 0 ? null : pointAt(b, tauA).f1;
  const f1OnA = tauB === null || a.length === 0 ? null : pointAt(a, tauB).f1;
  const both = [f1OnB, f1OnA].filter((x): x is number => x !== null);
  return { tauFromA: tauA, f1OnB, tauFromB: tauB, f1OnA, mean: both.length === 0 ? null : both.reduce((x, y) => x + y, 0) / both.length };
}

// ------------------------------------------------------------ entity level ---

const nonPred = (s: { entityType: string }): boolean => !s.entityType.startsWith("pred:");
const overlaps = (a: { start: number; end: number }, b: { start: number; end: number }): boolean => a.start < b.end && b.start < a.end;

export interface EntityPoint {
  readonly threshold: number;
  readonly tp: number;
  readonly fp: number;
  readonly fn: number;
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
  readonly leakBearing: number;
  readonly fullyCaught: number;
  readonly leakPrevention: number;
  readonly clean: number;
  readonly overBlocked: number;
  readonly overBlocking: number;
}

export function entityPointWith(ir: PolicyIr, records: readonly TypeSafeRecord[], t: import("./typesafe.js").Thresholds): EntityPoint {
  let tp = 0, fp = 0, fn = 0, leakBearing = 0, fullyCaught = 0, clean = 0, overBlocked = 0;
  for (const r of records) {
    const findings = projectFindings(ir, r, t).filter(nonPred);
    const gold = r.gold.filter(nonPred);
    const counts = pairSpansGreedy("overlap", findings, gold);
    tp += counts.tp; fp += counts.fp; fn += counts.fn;
    if (gold.length > 0) {
      leakBearing += 1;
      if (gold.every((g) => findings.some((f) => overlaps(f, g)))) fullyCaught += 1;
    } else {
      clean += 1;
      if (findings.length > 0) overBlocked += 1;
    }
  }
  const base = prf1(tp, fp, fn, t.candidate);
  return { ...base, leakBearing, fullyCaught, leakPrevention: leakBearing === 0 ? 0 : fullyCaught / leakBearing,
           clean, overBlocked, overBlocking: clean === 0 ? 0 : overBlocked / clean };
}

export function entityPoint(ir: PolicyIr, records: readonly TypeSafeRecord[], threshold: number): EntityPoint {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let leakBearing = 0;
  let fullyCaught = 0;
  let clean = 0;
  let overBlocked = 0;
  for (const r of records) {
    // Predicate threshold 1.1 switches the shadow finding off: this is the ENTITY
    // question, and mixing the predicate into it is what made an earlier table
    // in this project unreadable.
    const findings = projectFindings(ir, r, { predicate: 1.1, candidate: threshold }).filter(nonPred);
    const gold = r.gold.filter(nonPred);
    const counts = pairSpansGreedy("overlap", findings, gold);
    tp += counts.tp;
    fp += counts.fp;
    fn += counts.fn;
    if (gold.length > 0) {
      leakBearing += 1;
      if (gold.every((g) => findings.some((f) => overlaps(f, g)))) fullyCaught += 1;
    } else {
      clean += 1;
      if (findings.length > 0) overBlocked += 1;
    }
  }
  const base = prf1(tp, fp, fn, threshold);
  return {
    ...base,
    leakBearing,
    fullyCaught,
    leakPrevention: leakBearing === 0 ? 0 : fullyCaught / leakBearing,
    clean,
    overBlocked,
    overBlocking: clean === 0 ? 0 : overBlocked / clean,
  };
}

export function entitySweep(ir: PolicyIr, records: readonly TypeSafeRecord[], steps = 20): EntityPoint[] {
  return Array.from({ length: steps + 1 }, (_, i) => entityPoint(ir, records, i / steps));
}

/** Tier 0 alone, with no judgment at all: the same candidates, labelled by the regex that found them. */
export function tier0Baseline(records: readonly TypeSafeRecord[]): EntityPoint {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let leakBearing = 0;
  let fullyCaught = 0;
  let clean = 0;
  let overBlocked = 0;
  for (const r of records) {
    const findings: ProjectedFinding[] = r.candidates
      .filter((c) => c.source === "tier0" && c.tier0EntityType !== null)
      .map((c) => ({ start: c.start, end: c.end, text: c.text, entityType: c.tier0EntityType!, severity: "high", tier: 0, source: "tier0", confidence: 1, action: "block" }));
    const gold = r.gold.filter(nonPred);
    const counts = pairSpansGreedy("overlap", findings, gold);
    tp += counts.tp;
    fp += counts.fp;
    fn += counts.fn;
    if (gold.length > 0) {
      leakBearing += 1;
      if (gold.every((g) => findings.some((f) => overlaps(f, g)))) fullyCaught += 1;
    } else {
      clean += 1;
      if (findings.length > 0) overBlocked += 1;
    }
  }
  const base = prf1(tp, fp, fn, 0);
  return { ...base, leakBearing, fullyCaught, leakPrevention: leakBearing === 0 ? 0 : fullyCaught / leakBearing, clean, overBlocked, overBlocking: clean === 0 ? 0 : overBlocked / clean };
}

export interface FilterEffect {
  readonly tier0Candidates: number;
  readonly onGold: number;
  readonly offGold: number;
  /** Off-gold candidates the model rejected: false positives the regex tier would have filed. */
  readonly correctRejections: number;
  /** On-gold candidates the model rejected: real leaks the judgment threw away. */
  readonly wrongfulRejections: number;
  readonly rejectionRateOffGold: number;
  readonly rejectionRateOnGold: number;
}

/**
 * What the judgment ADDS over the candidate generator, counted directly.
 *
 * This is the arm's whole thesis in four numbers: the regexes over-find on
 * purpose, and the only thing the model can do is keep or reject. A correct
 * rejection is precision the regex tier could not have; a wrongful one is recall
 * this arm destroyed.
 */
export function filterEffect(records: readonly TypeSafeRecord[]): FilterEffect {
  let onGold = 0;
  let offGold = 0;
  let correctRejections = 0;
  let wrongfulRejections = 0;
  let tier0Candidates = 0;
  for (const r of records) {
    const gold = r.gold.filter(nonPred);
    for (const c of r.candidates) {
      if (c.source !== "tier0") continue;
      tier0Candidates += 1;
      const hitsGold = gold.some((g) => overlaps(c, g));
      const rejected = c.choice === null ? false : c.choice === NOT_CONFIDENTIAL;
      if (hitsGold) {
        onGold += 1;
        if (rejected) wrongfulRejections += 1;
      } else {
        offGold += 1;
        if (rejected) correctRejections += 1;
      }
    }
  }
  return {
    tier0Candidates,
    onGold,
    offGold,
    correctRejections,
    wrongfulRejections,
    rejectionRateOffGold: offGold === 0 ? 0 : correctRejections / offGold,
    rejectionRateOnGold: onGold === 0 ? 0 : wrongfulRejections / onGold,
  };
}

export interface CostTimeSummary {
  readonly items: number;
  readonly answered: number;
  readonly costUsd: number;
  readonly costPer1kMessages: number | null;
  readonly inputTokensP50: number | null;
  readonly questionsPerItemP50: number | null;
  readonly itemWallMsP50: number | null;
  readonly itemWallMsP95: number | null;
}

export function costAndTime(records: readonly TypeSafeRecord[]): CostTimeSummary {
  const answered = records.filter((r) => r.predicateProbability !== null);
  const pct = (xs: number[], q: number): number | null => {
    if (xs.length === 0) return null;
    const s = [...xs].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(q * s.length))] ?? null;
  };
  const costUsd = records.reduce((a, r) => a + r.costUsd, 0);
  return {
    items: records.length,
    answered: answered.length,
    costUsd,
    costPer1kMessages: answered.length === 0 ? null : (costUsd / answered.length) * 1000,
    inputTokensP50: pct(answered.map((r) => r.inputTokens ?? 0), 0.5),
    questionsPerItemP50: pct(records.map((r) => r.questionCount), 0.5),
    itemWallMsP50: pct(answered.map((r) => r.itemWallMs), 0.5),
    itemWallMsP95: pct(answered.map((r) => r.itemWallMs), 0.95),
  };
}

// ------------------------------------------------------------------ tuning ---

/**
 * Threshold tuning, and the honesty machinery that has to come with it.
 *
 * The first run scored one global 0.5 with the argmax rule and no overlap
 * merging, because those were defaults, not decisions. Three things were left on
 * the table and all three are measurable from the SAME stored probabilities, at
 * no further cost:
 *
 *   1. firing on 1 - P(not-confidential) rather than on the top option's own
 *      probability, which stops splitting mass between two confidential labels
 *      from reading as doubt;
 *   2. keeping one finding per overlapping cluster;
 *   3. a threshold per entity type.
 *
 * A tuned number is worth nothing without the estimate that says it generalises,
 * so `crossValidateEntity` tunes on four fifths of the MESSAGES and scores the
 * fifth, five times, and that pooled number is the one to quote. Tuning on the
 * rows you then score is how a 0.79 becomes a lie.
 */
export const THRESHOLD_GRID: readonly number[] = Array.from({ length: 21 }, (_, i) => Number((i / 20).toFixed(2)));

/** Deterministic, content-addressed fold assignment: the same message lands in the same fold anywhere. */
export function foldOf(itemId: string, folds: number): number {
  let h = 2166136261;
  for (let i = 0; i < itemId.length; i++) {
    h ^= itemId.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h % folds;
}

export interface TuneOptions {
  readonly rule: "argmax" | "confidential-mass";
  readonly mergeOverlaps: boolean;
  readonly base: number;
  readonly rounds?: number;
}

/** Coordinate ascent over the per-type thresholds; deterministic, and it never reads the fold it is scored on. */
export function tuneThresholds(ir: PolicyIr, records: readonly TypeSafeRecord[], opts: TuneOptions): Record<string, number> {
  const types = ir.entityTypes.filter((e) => !e.id.startsWith("pred:")).map((e) => e.id);
  const perType: Record<string, number> = Object.fromEntries(types.map((t) => [t, opts.base]));
  for (let round = 0; round < (opts.rounds ?? 3); round++) {
    for (const type of types) {
      let bestValue = perType[type] ?? opts.base;
      let bestF1 = -1;
      for (const v of THRESHOLD_GRID) {
        const f1 = entityPointWith(ir, records, { predicate: 1.1, candidate: opts.base, perType: { ...perType, [type]: v }, rule: opts.rule, mergeOverlaps: opts.mergeOverlaps }).f1;
        if (f1 > bestF1) { bestF1 = f1; bestValue = v; }
      }
      perType[type] = bestValue;
    }
  }
  return perType;
}

export interface CrossValidated {
  readonly folds: number;
  readonly point: EntityPoint;
  readonly perFold: readonly { fold: number; items: number; thresholds: Record<string, number> }[];
}

/** The number to quote: every message scored by thresholds tuned without it. */
export function crossValidateEntity(ir: PolicyIr, records: readonly TypeSafeRecord[], opts: TuneOptions, folds = 5): CrossValidated {
  let tp = 0, fp = 0, fn = 0, leakBearing = 0, fullyCaught = 0, clean = 0, overBlocked = 0;
  const perFold: { fold: number; items: number; thresholds: Record<string, number> }[] = [];
  for (let k = 0; k < folds; k++) {
    const train = records.filter((r) => foldOf(r.itemId, folds) !== k);
    const test = records.filter((r) => foldOf(r.itemId, folds) === k);
    if (test.length === 0) continue;
    const perType = tuneThresholds(ir, train, opts);
    perFold.push({ fold: k, items: test.length, thresholds: perType });
    const p = entityPointWith(ir, test, { predicate: 1.1, candidate: opts.base, perType, rule: opts.rule, mergeOverlaps: opts.mergeOverlaps });
    tp += p.tp; fp += p.fp; fn += p.fn;
    leakBearing += p.leakBearing; fullyCaught += p.fullyCaught; clean += p.clean; overBlocked += p.overBlocked;
  }
  const base = prf1(tp, fp, fn, 0);
  return {
    folds,
    perFold,
    point: { ...base, leakBearing, fullyCaught, leakPrevention: leakBearing === 0 ? 0 : fullyCaught / leakBearing,
             clean, overBlocked, overBlocking: clean === 0 ? 0 : overBlocked / clean },
  };
}

/**
 * The predicate threshold, chosen at the MIDDLE of the gap rather than at the
 * best observed value.
 *
 * An F1-argmax lands exactly on some message's probability, so the next run's
 * equivalent message can fall a thousandth below it. The midpoint of the widest
 * empty band is the same decision with margin on both sides, and it is worth
 * 0.006 of split-half F1 here for no cost at all.
 */
export function pickThresholdMidGap(rows: readonly ScoredRow[]): number | null {
  const pos = rows.filter((r) => r.label).map((r) => r.probability).sort((a, b) => a - b);
  const neg = rows.filter((r) => !r.label).map((r) => r.probability).sort((a, b) => a - b);
  if (pos.length === 0 || neg.length === 0) return null;
  const argmax = bestPoint(sweep(rows))?.threshold;
  if (argmax === undefined) return null;
  const below = neg.filter((p) => p < argmax);
  const above = pos.filter((p) => p >= argmax);
  if (above.length === 0) return argmax;
  const lower = below.length === 0 ? 0 : Math.max(...below);
  return Number(((lower + Math.min(...above)) / 2).toFixed(4));
}

/** Split-half with a choosable picker, so two picking rules can be compared on the same rows. */
export function splitHalfWith(rows: readonly ScoredRow[], pick: (r: readonly ScoredRow[]) => number | null): SplitHalf {
  const a = rows.filter((r) => halfOf(r.itemId) === 0);
  const b = rows.filter((r) => halfOf(r.itemId) === 1);
  const tauA = pick(a);
  const tauB = pick(b);
  const f1OnB = tauA === null || b.length === 0 ? null : pointAt(b, tauA).f1;
  const f1OnA = tauB === null || a.length === 0 ? null : pointAt(a, tauB).f1;
  const both = [f1OnB, f1OnA].filter((x): x is number => x !== null);
  return { tauFromA: tauA, f1OnB, tauFromB: tauB, f1OnA, mean: both.length === 0 ? null : both.reduce((x, y) => x + y, 0) / both.length };
}
