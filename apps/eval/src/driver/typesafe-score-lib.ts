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
