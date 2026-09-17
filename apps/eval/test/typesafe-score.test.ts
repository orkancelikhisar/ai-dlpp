import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { loadPolicyIr } from "@sih/core";
import { NOT_CONFIDENTIAL, TYPESAFE_MODEL, type TypeSafeRecord } from "../src/driver/typesafe.js";
import {
  averagePrecision,
  bestPoint,
  calibration,
  costAndTime,
  entityPoint,
  filterEffect,
  halfOf,
  loadPredicateGold,
  pairRows,
  pointAt,
  rocAuc,
  splitHalf,
  sweep,
  tier0Baseline,
  type ScoredRow,
} from "../src/driver/typesafe-score-lib.js";

const IR = loadPolicyIr(readFileSync(new URL("../../../policies/compiled/p-fin.ir.json", import.meta.url), "utf8"));
const GOLD_JSONL = readFileSync(new URL("../../../corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl", import.meta.url), "utf8");
const rows = (...xs: [string, number, boolean][]): ScoredRow[] => xs.map(([itemId, probability, label]) => ({ itemId, probability, label }));

describe("predicate gold", () => {
  it("is the same 179 scored rows and 19 positives the paper reports", () => {
    const gold = loadPredicateGold(GOLD_JSONL, "client-relationship-disclosure");
    expect(gold.scored).toBe(179);
    expect(gold.positives).toBe(19);
  });
  it("ignores rows for another predicate or without a scored status", () => {
    const gold = loadPredicateGold(GOLD_JSONL, "no-such-predicate");
    expect(gold.scored).toBe(0);
  });
});

describe("threshold sweep", () => {
  const r = rows(["a", 0.9, true], ["b", 0.6, true], ["c", 0.4, false], ["d", 0.1, false]);
  it("counts at a given threshold", () => {
    expect(pointAt(r, 0.5)).toMatchObject({ tp: 2, fp: 0, fn: 0, precision: 1, recall: 1, f1: 1 });
    expect(pointAt(r, 0.05)).toMatchObject({ tp: 2, fp: 2, fn: 0 });
    expect(pointAt(r, 0.95)).toMatchObject({ tp: 0, fp: 0, fn: 2, f1: 0 });
  });
  it("uses every observed probability as a threshold so no maximum is missed", () => {
    const points = sweep(r);
    expect(points.map((p) => p.threshold)).toEqual([0.1, 0.4, 0.5, 0.6, 0.9]);
    expect(bestPoint(points)!.f1).toBe(1);
  });
});

describe("ranking and calibration", () => {
  it("ROC-AUC is 1 for perfect separation, 0 when inverted, 0.5 for ties", () => {
    expect(rocAuc(rows(["a", 0.9, true], ["b", 0.1, false]))).toBe(1);
    expect(rocAuc(rows(["a", 0.1, true], ["b", 0.9, false]))).toBe(0);
    expect(rocAuc(rows(["a", 0.5, true], ["b", 0.5, false]))).toBe(0.5);
    expect(rocAuc(rows(["a", 0.5, true]))).toBeNull();
  });
  it("average precision rewards positives ranked first", () => {
    expect(averagePrecision(rows(["a", 0.9, true], ["b", 0.8, true], ["c", 0.1, false]))).toBe(1);
    expect(averagePrecision(rows(["a", 0.9, false], ["b", 0.8, true]))).toBeCloseTo(0.5, 6);
  });
  it("calibration reports observed frequency against mean probability per bin", () => {
    const bins = calibration(rows(["a", 0.05, false], ["b", 0.95, true], ["c", 0.92, false]), 10);
    expect(bins[0]).toMatchObject({ n: 1, observedRate: 0 });
    expect(bins[9]).toMatchObject({ n: 2, observedRate: 0.5 });
    expect(bins[4]!.n).toBe(0);
  });
});

describe("split-half", () => {
  it("is deterministic and reports the threshold from one half spent on the other", () => {
    expect(halfOf("i-1")).toBe(halfOf("i-1"));
    const r = rows(["i-1", 0.9, true], ["i-2", 0.2, false], ["i-3", 0.8, true], ["i-4", 0.1, false], ["i-5", 0.7, true], ["i-6", 0.3, false]);
    const sh = splitHalf(r);
    expect(sh.mean).not.toBeNull();
    expect(sh.mean).toBeLessThanOrEqual(bestPoint(sweep(r))!.f1);
  });
});

const rec = (over: Partial<TypeSafeRecord>): TypeSafeRecord => ({
  schemaVersion: 1, runId: "r", itemId: "i", policy: "p-fin", irHash: "a".repeat(64), policyHash: IR.policyHash, arm: "ts-judgment",
  requestedModelId: TYPESAFE_MODEL, modelId: TYPESAFE_MODEL, gitSha: "b".repeat(40), gitDirty: false, dryRun: true,
  text: "Meridian Capital and account 50100234567890 and nothing else here at all",
  gold: [], predicateId: "client-relationship-disclosure", predicateProbability: 0.5, candidates: [], questionCount: 1,
  inputTokens: 100, outputTokens: 4, costUsd: 0.000004, callWallMs: 120, itemWallMs: 130, attempts: 1, retries: 0, httpStatus: 200, error: null,
  ...over,
});
const cand = (i: number, start: number, end: number, text: string, choice: string | null, p: number, source: "tier0" | "orthographic" = "tier0", t0: string | null = "bank-account-identifier") => ({
  index: i, start, end, text, source, tier0EntityType: t0, choice, probability: p, confidence: p, probabilities: null,
});

describe("entity scoring", () => {
  const leak = rec({
    itemId: "leak",
    gold: [{ start: 29, end: 43, text: "50100234567890", entityType: "bank-account-identifier", action: "block" }],
    candidates: [cand(0, 29, 43, "50100234567890", "bank-account-identifier", 0.9), cand(1, 0, 16, "Meridian Capital", NOT_CONFIDENTIAL, 0.8, "orthographic", null)],
  });
  const clean = rec({ itemId: "clean", gold: [], candidates: [cand(0, 0, 16, "Meridian Capital", "client-name", 0.6, "orthographic", null)] });

  it("pairs findings to gold by overlap and reports prevention and over-blocking", () => {
    const p = entityPoint(IR, [leak, clean], 0.5);
    expect(p).toMatchObject({ tp: 1, fn: 0, leakBearing: 1, fullyCaught: 1, leakPrevention: 1, clean: 1, overBlocked: 1, overBlocking: 1 });
  });

  it("a higher threshold silences the low-probability finding", () => {
    expect(entityPoint(IR, [leak, clean], 0.7)).toMatchObject({ overBlocked: 0, overBlocking: 0, tp: 1 });
  });

  it("scores tier 0 alone, with no judgment, from the same candidates", () => {
    const t0 = tier0Baseline([leak, clean]);
    expect(t0.tp).toBe(1);
    expect(t0.leakPrevention).toBe(1);
  });

  it("counts what the judgment adds: correct rejections against wrongful ones", () => {
    const wrong = rec({
      itemId: "wrong",
      gold: [{ start: 29, end: 43, text: "50100234567890", entityType: "bank-account-identifier", action: "block" }],
      candidates: [cand(0, 29, 43, "50100234567890", NOT_CONFIDENTIAL, 0.9), cand(1, 44, 47, "and", NOT_CONFIDENTIAL, 0.9)],
    });
    const fe = filterEffect([wrong]);
    expect(fe).toMatchObject({ tier0Candidates: 2, onGold: 1, offGold: 1, wrongfulRejections: 1, correctRejections: 1, rejectionRateOnGold: 1, rejectionRateOffGold: 1 });
  });
});

describe("pairing and cost", () => {
  it("drops records the gold does not score and counts unanswered rows separately", () => {
    const gold = { satisfies: new Map([["a", true], ["b", false]]), scored: 2, positives: 1 };
    const { rows: paired, unanswered } = pairRows([rec({ itemId: "a", predicateProbability: 0.9 }), rec({ itemId: "b", predicateProbability: null }), rec({ itemId: "zz", predicateProbability: 0.2 })], gold);
    expect(paired).toEqual([{ itemId: "a", probability: 0.9, label: true }]);
    expect(unanswered).toEqual(["b"]);
  });
  it("reports cost per thousand messages over answered rows only", () => {
    const ct = costAndTime([rec({ itemId: "a", costUsd: 0.00001 }), rec({ itemId: "b", predicateProbability: null, costUsd: 0 })]);
    expect(ct.answered).toBe(1);
    expect(ct.costPer1kMessages).toBeCloseTo(0.01, 9);
  });
});
