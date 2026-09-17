import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { loadPolicyIr } from "@sih/core";
import { NOT_CONFIDENTIAL, TYPESAFE_MODEL, decideCandidate, projectFindings, type CandidateAnswer, type TypeSafeRecord } from "../src/driver/typesafe.js";
import {
  THRESHOLD_GRID,
  blockedShare,
  crossValidateEntity,
  entityPointWith,
  foldOf,
  pickThresholdMidGap,
  splitHalfWith,
  tuneThresholds,
  type ScoredRow,
} from "../src/driver/typesafe-score-lib.js";

const IR = loadPolicyIr(readFileSync(new URL("../../../policies/compiled/p-fin.ir.json", import.meta.url), "utf8"));
const cand = (over: Partial<CandidateAnswer> = {}): CandidateAnswer => ({
  index: 0, start: 0, end: 16, text: "Meridian Capital", source: "tier0", tier0EntityType: null,
  choice: "client-name", probability: 0.6, confidence: 0.6, probabilities: { "client-name": 0.6, [NOT_CONFIDENTIAL]: 0.4 }, ...over,
});
const rec = (over: Partial<TypeSafeRecord> = {}): TypeSafeRecord => ({
  schemaVersion: 1, runId: "r", itemId: "i", policy: "p-fin", irHash: "a".repeat(64), policyHash: IR.policyHash, arm: "ts-judgment",
  requestedModelId: TYPESAFE_MODEL, modelId: TYPESAFE_MODEL, gitSha: "b".repeat(40), gitDirty: false, dryRun: true,
  text: "Meridian Capital and the key material below and some more text here",
  gold: [], predicateId: "client-relationship-disclosure", predicateProbability: null, candidates: [], questionCount: 1,
  inputTokens: 10, outputTokens: 1, costUsd: 0, callWallMs: 1, itemWallMs: 1, attempts: 1, retries: 0, httpStatus: 200, error: null, ...over,
});

describe("decideCandidate", () => {
  it("argmax fires on the top option's own probability", () => {
    expect(decideCandidate(cand(), { predicate: 1, candidate: 0.5 })).toEqual({ entityType: "client-name", probability: 0.6 });
    expect(decideCandidate(cand(), { predicate: 1, candidate: 0.7 })).toBeUndefined();
  });

  it("confidential mass fires when two confidential labels split the mass, where argmax does not", () => {
    const split = cand({ choice: "api-credential", probability: 0.4, probabilities: { "api-credential": 0.4, "db-connection-string": 0.35, [NOT_CONFIDENTIAL]: 0.25 } });
    expect(decideCandidate(split, { predicate: 1, candidate: 0.5 })).toBeUndefined();
    expect(decideCandidate(split, { predicate: 1, candidate: 0.5, rule: "confidential-mass" })).toEqual({ entityType: "api-credential", probability: 0.75 });
  });

  it("stays silent when the model rejected the candidate outright", () => {
    const no = cand({ choice: NOT_CONFIDENTIAL, probability: 0.9, probabilities: { "client-name": 0.1, [NOT_CONFIDENTIAL]: 0.9 } });
    expect(decideCandidate(no, { predicate: 1, candidate: 0.5 })).toBeUndefined();
    expect(decideCandidate(no, { predicate: 1, candidate: 0.5, rule: "confidential-mass" })).toBeUndefined();
  });

  it("takes the per-type threshold over the global one", () => {
    const t = { predicate: 1, candidate: 0.5, perType: { "client-name": 0.95 } };
    expect(decideCandidate(cand(), t)).toBeUndefined();
    expect(decideCandidate(cand({ choice: "in-pan", probability: 0.6, probabilities: { "in-pan": 0.6, [NOT_CONFIDENTIAL]: 0.4 } }), t)).toEqual({ entityType: "in-pan", probability: 0.6 });
  });
});

describe("overlap merging", () => {
  const overlapping = rec({
    candidates: [
      cand({ index: 0, start: 20, end: 40, text: "key material below x", choice: "private-key-material", probability: 0.7, probabilities: { "private-key-material": 0.7, [NOT_CONFIDENTIAL]: 0.3 } }),
      cand({ index: 1, start: 24, end: 36, text: "material belo", choice: "private-key-material", probability: 0.9, probabilities: { "private-key-material": 0.9, [NOT_CONFIDENTIAL]: 0.1 } }),
      cand({ index: 2, start: 0, end: 16, text: "Meridian Capital" }),
    ],
  });
  it("keeps every finding when merging is off", () => {
    expect(projectFindings(IR, overlapping, { predicate: 1.1, candidate: 0.5 })).toHaveLength(3);
  });
  it("keeps one per overlapping cluster, the most confident, and leaves disjoint findings alone", () => {
    const merged = projectFindings(IR, overlapping, { predicate: 1.1, candidate: 0.5, mergeOverlaps: true });
    expect(merged).toHaveLength(2);
    expect(merged.map((f) => [f.start, f.end])).toEqual([[0, 16], [24, 36]]);
    expect(merged[1]!.confidence).toBe(0.9);
  });
});

describe("gating a type on the message predicate", () => {
  const withClient = (predicateProbability: number | null): TypeSafeRecord =>
    rec({ predicateProbability, gold: [], candidates: [cand({ choice: "client-name", probability: 0.8, probabilities: { "client-name": 0.8, [NOT_CONFIDENTIAL]: 0.2 } })] });
  const t = { predicate: 0.375, candidate: 0.5, rule: "confidential-mass" as const, gateTypesOnPredicate: ["client-name"] };

  it("keeps a client-name finding when the message discloses a client relationship", () => {
    expect(projectFindings(IR, withClient(0.9), t).filter((f) => f.entityType === "client-name")).toHaveLength(1);
  });

  it("drops it when the message does not, and when the predicate went unanswered", () => {
    expect(projectFindings(IR, withClient(0.2), t).filter((f) => f.entityType === "client-name")).toHaveLength(0);
    expect(projectFindings(IR, withClient(null), t).filter((f) => f.entityType === "client-name")).toHaveLength(0);
  });

  it("gates only the named types, and not at all when the list is absent", () => {
    const other = rec({ predicateProbability: 0.1, gold: [], candidates: [cand({ choice: "in-pan", probability: 0.8, probabilities: { "in-pan": 0.8, [NOT_CONFIDENTIAL]: 0.2 } })] });
    expect(projectFindings(IR, other, t)).toHaveLength(1);
    expect(projectFindings(IR, withClient(0.2), { ...t, gateTypesOnPredicate: undefined })).toHaveLength(1);
  });
});

describe("blocked is not touched", () => {
  it("counts only clean messages carrying a finding the policy blocks", () => {
    const pseudonymised = rec({ itemId: "p", gold: [], candidates: [cand({ choice: "client-name", probability: 0.9, probabilities: { "client-name": 0.9, [NOT_CONFIDENTIAL]: 0.1 } })] });
    const blocked = rec({ itemId: "b", gold: [], candidates: [cand({ choice: "in-pan", probability: 0.9, probabilities: { "in-pan": 0.9, [NOT_CONFIDENTIAL]: 0.1 } })] });
    const t = { predicate: 1.1, candidate: 0.5, rule: "confidential-mass" as const };
    expect(entityPointWith(IR, [pseudonymised, blocked], t).overBlocking).toBe(1);
    expect(blockedShare(IR, [pseudonymised, blocked], t)).toEqual({ clean: 2, blocked: 1, share: 0.5 });
  });
});

describe("folds", () => {
  it("are deterministic, in range, and spread the corpus", () => {
    const ids = Array.from({ length: 200 }, (_, i) => `inj-o${i}-${i % 3}`);
    expect(ids.map((i) => foldOf(i, 5))).toEqual(ids.map((i) => foldOf(i, 5)));
    expect(ids.every((i) => foldOf(i, 5) >= 0 && foldOf(i, 5) < 5)).toBe(true);
    const sizes = [0, 1, 2, 3, 4].map((k) => ids.filter((i) => foldOf(i, 5) === k).length);
    expect(Math.min(...sizes)).toBeGreaterThan(ids.length / 10);
  });
});

describe("tuning", () => {
  const goldSpan = { start: 20, end: 40, text: "key material below x", entityType: "private-key-material", action: "block" };
  const records = [
    rec({ itemId: "a", gold: [goldSpan], candidates: [cand({ start: 20, end: 40, choice: "private-key-material", probability: 0.9, probabilities: { "private-key-material": 0.9, [NOT_CONFIDENTIAL]: 0.1 } })] }),
    rec({ itemId: "b", gold: [], candidates: [cand({ choice: "client-name", probability: 0.55, probabilities: { "client-name": 0.55, [NOT_CONFIDENTIAL]: 0.45 } })] }),
    rec({ itemId: "c", gold: [], candidates: [cand({ choice: "client-name", probability: 0.6, probabilities: { "client-name": 0.6, [NOT_CONFIDENTIAL]: 0.4 } })] }),
  ];
  const opts = { rule: "confidential-mass" as const, mergeOverlaps: true, base: 0.5 };

  it("only picks thresholds from the published grid", () => {
    const th = tuneThresholds(IR, records, opts);
    expect(Object.values(th).every((v) => THRESHOLD_GRID.includes(v))).toBe(true);
  });

  it("never scores worse than the untuned default on the rows it tuned on", () => {
    const th = tuneThresholds(IR, records, opts);
    const tuned = entityPointWith(IR, records, { predicate: 1.1, candidate: 0.5, perType: th, ...opts }).f1;
    const flat = entityPointWith(IR, records, { predicate: 1.1, candidate: 0.5, ...opts }).f1;
    expect(tuned).toBeGreaterThanOrEqual(flat);
    // and it learns the obvious lesson: silence the label that only ever fires on clean text
    expect(th["client-name"]).toBeGreaterThan(0.6);
  });

  it("cross-validation scores every fold with thresholds tuned without it", () => {
    const cv = crossValidateEntity(IR, records, opts, 3);
    expect(cv.perFold.reduce((a, f) => a + f.items, 0)).toBe(records.length);
    expect(cv.point.tp + cv.point.fn).toBe(1);
  });
});

describe("predicate threshold picking", () => {
  const rows = (...xs: [string, number, boolean][]): ScoredRow[] => xs.map(([itemId, probability, label]) => ({ itemId, probability, label }));
  it("puts the threshold in the middle of the empty band, not on an observed value", () => {
    const r = rows(["a", 0.9, true], ["b", 0.41, true], ["c", 0.29, false], ["d", 0.1, false]);
    expect(pickThresholdMidGap(r)).toBeCloseTo(0.35, 6);
  });
  it("returns nothing when one class is missing", () => {
    expect(pickThresholdMidGap(rows(["a", 0.9, true]))).toBeNull();
  });
  it("beats the best-observed-value picker on held-out halves of separable data", () => {
    const r = rows(...Array.from({ length: 40 }, (_, i) => [`neg-${i}`, 0.05 + (i % 5) / 100, false] as [string, number, boolean]),
                   ...Array.from({ length: 12 }, (_, i) => [`pos-${i}`, 0.5 + (i % 4) / 100, true] as [string, number, boolean]));
    const argmax = splitHalfWith(r, (rs) => rs.reduce<number>((best, x) => (x.label && x.probability < best ? x.probability : best), 1));
    const midgap = splitHalfWith(r, pickThresholdMidGap);
    expect(midgap.mean).toBeGreaterThanOrEqual(argmax.mean ?? 0);
    expect(midgap.mean).toBe(1);
  });
});
