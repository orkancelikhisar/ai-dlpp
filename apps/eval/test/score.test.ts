import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadCorpus } from "../src/driver/corpus.js";
import type { RunRecord } from "../src/driver/record.js";
import {
  MATCH_RULES,
  Tier2GoldRowSchema,
  assertGoldMatchesRecords,
  groupGoldByPredicate,
  intersectionLength,
  iou,
  loadTier2Gold,
  bestFloorF1,
  scoreArm,
  scoreArms,
  scoreTrivialFloors,
  spansMatch,
  winnersByRule,
  type MatchRule,
  type Tier2GoldRow,
} from "../src/driver/score.js";

const REPO = fileURLToPath(new URL("../../..", import.meta.url));
const GOLD_PATH = `${REPO}corpora/fixtures/smoke.gold-tier2.jsonl`;
const CORPUS_PATH = `${REPO}corpora/fixtures/smoke.jsonl`;
const P_FIN_HASH = "ebb3cd68d973175f3ea40faeec00685e2cb9d83c6940e96a57e88ee269e8110a";

const IR_HASH = "b00e5ce67a7e6d1643c81b2e8f29498faad71b5d6b4a6e53b70b644b34f592d0";
const PRED = "pred:client-relationship-disclosure";

interface FindingSpec {
  readonly start: number;
  readonly end: number;
  readonly text: string;
  readonly entityType?: string;
  readonly tier?: 0 | 1 | 2;
}

function makeRecord(spec: {
  itemId: string;
  text: string;
  findings?: readonly FindingSpec[];
  arm?: string;
  policyHash?: string;
  answeredCalls?: number;
  error?: string | null;
  degraded?: readonly { tier: 0 | 1 | 2; reason: string; detail: string }[];
}): RunRecord {
  return {
    schemaVersion: 1,
    runId: "test",
    itemId: spec.itemId,
    policy: "minimal-fixture",
    irHash: IR_HASH,
    policyHash: spec.policyHash ?? P_FIN_HASH,
    arm: spec.arm ?? "tier2-Test-Model",
    detector: "core-orchestrator",
    backend: "webgpu",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: true, t2Model: "Test-Model", backend: "webgpu" },
    tier2Stats: {
      rung1: 0,
      rung2: 0,
      unresolvedQuotes: 0,
      unresolvedMentions: 0,
      wholeClauseMentions: 0,
      unknownPredicates: 0,
      duplicatesDropped: 0,
      repairAttempts: 0,
      failedClosed: 0,
      truncatedResponses: 0,
      abortedResponses: 0,
      segmentsJudged: 0,
      segmentsSkipped: 0,
      messageScopeCalls: 0,
      messageScopeJudged: 0,
      messageScopeFailedClosed: 0,
      deadlineExpiries: 0,
      callerAbortsMidGeneration: 0,
      callerAbortsWhileQueued: 0,
      calls: Array.from({ length: spec.answeredCalls ?? 1 }, () => ({ finishReason: "stop" as const })),
    },
    text: spec.text,
    findings: (spec.findings ?? []).map((f) => ({
      start: f.start,
      end: f.end,
      text: f.text,
      entityType: f.entityType ?? PRED,
      severity: "high" as const,
      tier: f.tier ?? (2 as const),
      source: "Test-Model",
      confidence: 1,
      action: "redact" as const,
    })),
    gold: [],
    timings: { tier0Ms: 0 },
    degraded: spec.degraded ? [...spec.degraded] : [],
    error: spec.error ?? null,
    abandonedWorkInFlight: false,
  } as RunRecord;
}

function goldRow(spec: {
  itemId: string;
  satisfies: boolean;
  spans?: readonly { start: number; end: number; text: string }[];
  status?: "scored" | "disputed";
  policyHash?: string;
}): Tier2GoldRow {
  const call = {
    satisfies: spec.satisfies,
    confidence: "clear" as const,
    rationale: "test fixture",
  };
  return Tier2GoldRowSchema.parse({
    itemId: spec.itemId,
    policy: "p-fin",
    policyHash: spec.policyHash ?? P_FIN_HASH,
    predicateId: "client-relationship-disclosure",
    entityType: PRED,
    status: spec.status ?? "scored",
    satisfies: spec.satisfies,
    confidence: "clear",
    spans: spec.spans ?? [],
    adjudication: "test fixture",
    annotators: { a: call, b: call },
  });
}

// ---------------------------------------------------------------------------
// The shipped artifact
// ---------------------------------------------------------------------------

describe("corpora/fixtures/smoke.gold-tier2.jsonl", () => {
  const gold = loadTier2Gold(readFileSync(GOLD_PATH, "utf8"));
  const corpus = loadCorpus(readFileSync(CORPUS_PATH, "utf8"));

  it("covers exactly the smoke corpus, one row per item", () => {
    expect(gold.map((g) => g.itemId).sort()).toEqual(corpus.map((c) => c.id).sort());
  });

  it("holds the labelling round's outcome: 2 positives, 11 negatives, 0 disputed", () => {
    // Pinned as NUMBERS rather than recomputed, so a later edit to the file has
    // to change this line too. The round is a historical fact about two blind
    // annotators; it does not move because code moved.
    expect(gold.filter((g) => g.satisfies)).toHaveLength(2);
    expect(gold.filter((g) => !g.satisfies)).toHaveLength(11);
    expect(gold.filter((g) => g.status === "disputed")).toHaveLength(0);
    expect(gold.filter((g) => g.satisfies).map((g) => g.itemId).sort()).toEqual([
      "pos-client-name-prose",
      "pos-multiline-pan-and-client",
    ]);
  });

  it("every gold span slices back to its own text on the CORPUS item, not on a copy", () => {
    // The cross-check corpus.ts describes, run against the file the labels were
    // written from. A span that does not slice back is a gold set scoring
    // against fiction, and this is the only place that can catch it before a
    // number is produced.
    const byId = new Map(corpus.map((c) => [c.id, c]));
    for (const row of gold) {
      for (const span of row.spans) {
        expect(byId.get(row.itemId)!.text.slice(span.start, span.end)).toBe(span.text);
      }
    }
  });

  it("pins the policy document the labels were read from", () => {
    expect(new Set(gold.map((g) => g.policyHash))).toEqual(new Set([P_FIN_HASH]));
    expect(new Set(gold.map((g) => g.predicateId))).toEqual(new Set(["client-relationship-disclosure"]));
  });

  it("keeps both annotators' original calls on every row", () => {
    for (const row of gold) {
      expect(row.annotators.a.rationale.length).toBeGreaterThan(0);
      expect(row.annotators.b.rationale.length).toBeGreaterThan(0);
      // The adjudicated call must be one the annotators actually made.
      expect([row.annotators.a.satisfies, row.annotators.b.satisfies]).toContain(row.satisfies);
    }
  });

  it("records the one soft item as borderline rather than laundering it clear", () => {
    const soft = gold.filter((g) => g.confidence === "borderline");
    expect(soft.map((g) => g.itemId)).toEqual(["pos-pan-prose"]);
    // The adjudicated confidence is the WEAKER of the two annotators', which on
    // this row is B's. A file that took A's would report the item as clear.
    expect(soft[0]!.annotators.a.confidence).toBe("clear");
    expect(soft[0]!.annotators.b.confidence).toBe("borderline");
  });
});

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

describe("Tier2GoldRowSchema", () => {
  it("refuses spans on a row that does not satisfy the predicate", () => {
    expect(() => goldRow({ itemId: "x", satisfies: false, spans: [{ start: 0, end: 2, text: "ab" }] })).toThrow();
  });

  it("refuses spans on a disputed row, so no scorer can reach them", () => {
    expect(() =>
      goldRow({ itemId: "x", satisfies: true, status: "disputed", spans: [{ start: 0, end: 2, text: "ab" }] }),
    ).toThrow();
  });

  it("refuses a scored positive with no span", () => {
    expect(() => goldRow({ itemId: "x", satisfies: true, spans: [] })).toThrow();
  });

  it("refuses an inverted or empty span, which corpus.ts's schema admits", () => {
    expect(() => goldRow({ itemId: "x", satisfies: true, spans: [{ start: 5, end: 5, text: "" }] })).toThrow();
    expect(() => goldRow({ itemId: "x", satisfies: true, spans: [{ start: 8, end: 3, text: "abc" }] })).toThrow();
  });

  it("refuses a duplicate itemId rather than double-weighting one item", () => {
    const row = JSON.stringify(goldRow({ itemId: "dup", satisfies: false }));
    expect(() => loadTier2Gold(`${row}\n${row}\n`)).toThrow(/repeats itemId/);
  });

  it("names the offending line number", () => {
    expect(() => loadTier2Gold(`{"itemId":"a"}\n`)).toThrow(/gold line 1/);
  });
});

// ---------------------------------------------------------------------------
// Match rules
// ---------------------------------------------------------------------------

describe("match rules", () => {
  it("agree on an exact pair and disagree on the run's real one", () => {
    // The pair MEASURED on runs/slate-p-fin-02.tier2only-Phi-4-mini: the gold
    // span is "Tamarind Grocers" at [43,59) and the model returned the whole
    // 74-character message. Intersection 16, union 74, IoU 0.216.
    const gold = { start: 43, end: 59 };
    const wholeMessage = { start: 0, end: 74 };
    expect(intersectionLength(gold, wholeMessage)).toBe(16);
    expect(iou(gold, wholeMessage)).toBeCloseTo(16 / 74, 10);
    expect(spansMatch("exact", gold, wholeMessage)).toBe(false);
    expect(spansMatch("overlap", gold, wholeMessage)).toBe(true);
    expect(spansMatch("iou50", gold, wholeMessage)).toBe(false);

    for (const rule of MATCH_RULES) expect(spansMatch(rule, gold, { start: 43, end: 59 })).toBe(true);
  });

  it("requires BOTH offsets to agree under exact, not just one", () => {
    // Found by mutation: `a.start === b.start && a.end === b.end` reduced to
    // `a.start === b.start` survived every other test in this file, because no
    // comparison in it shared one offset and differed in the other. A judge
    // that starts its span at the gold span's first character and runs past it
    // is the most likely wrong answer there is, so this is the mutation that
    // mattered.
    const gold = { start: 43, end: 59 };
    expect(spansMatch("exact", gold, { start: 43, end: 52 })).toBe(false);
    expect(spansMatch("exact", gold, { start: 50, end: 59 })).toBe(false);
  });

  it("treats abutting spans as disjoint", () => {
    // [0,5) and [5,9) share no character. A closed-interval reading would call
    // these adjacent spans overlapping and turn every neighbouring finding into
    // a true positive.
    expect(intersectionLength({ start: 0, end: 5 }, { start: 5, end: 9 })).toBe(0);
    expect(spansMatch("overlap", { start: 0, end: 5 }, { start: 5, end: 9 })).toBe(false);
  });

  it("puts iou50 exactly at 0.5, inclusive", () => {
    // [0,10) against [0,5): intersection 5, union 10, IoU exactly 0.5.
    expect(iou({ start: 0, end: 10 }, { start: 0, end: 5 })).toBe(0.5);
    expect(spansMatch("iou50", { start: 0, end: 10 }, { start: 0, end: 5 })).toBe(true);
    // One character wider and it drops below.
    expect(spansMatch("iou50", { start: 0, end: 11 }, { start: 0, end: 5 })).toBe(false);
  });

  it("computes IoU over the union and not over the hull", () => {
    // Disjoint [0,1) and [99,100): union is 2 units, hull is 100. Both give 0
    // here because the numerator is 0 -- this pins WHICH denominator is used by
    // checking an overlapping pair where the two formulas would agree, and the
    // disjoint pair where a hull denominator is the tempting mistake.
    expect(iou({ start: 0, end: 1 }, { start: 99, end: 100 })).toBe(0);
    expect(iou({ start: 0, end: 4 }, { start: 2, end: 6 })).toBe(2 / 6);
  });
});

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

const TAMARIND = "Can you draft a contract renewal email for Tamarind Grocers before Friday?";

describe("scoreArm", () => {
  it("scores a perfect hit under all three rules", () => {
    const scored = scoreArm({
      records: [
        makeRecord({
          itemId: "hit",
          text: TAMARIND,
          findings: [{ start: 43, end: 59, text: "Tamarind Grocers" }],
        }),
      ],
      gold: [goldRow({ itemId: "hit", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
    });
    for (const rule of MATCH_RULES) {
      expect(scored.byRule[rule]).toMatchObject({ tp: 1, fp: 0, fn: 0, precision: 1, recall: 1, f1: 1, nearMisses: 0 });
    }
  });

  it("splits the three rules on a right-thing-wrong-span finding", () => {
    // The behaviour the whole three-rule design exists for. Under `exact` and
    // `iou50` the same finding is simultaneously a false positive and a missed
    // gold span -- 1 FP and 1 FN off ONE finding -- and `nearMisses` is what
    // says those two counts describe the same object.
    const scored = scoreArm({
      records: [
        makeRecord({ itemId: "wide", text: TAMARIND, findings: [{ start: 0, end: 74, text: TAMARIND }] }),
      ],
      gold: [goldRow({ itemId: "wide", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
    });
    expect(scored.byRule.overlap).toMatchObject({ tp: 1, fp: 0, fn: 0, precision: 1, recall: 1, f1: 1, nearMisses: 0 });
    expect(scored.byRule.exact).toMatchObject({ tp: 0, fp: 1, fn: 1, precision: 0, recall: 0, f1: 0, nearMisses: 1 });
    expect(scored.byRule.iou50).toMatchObject({ tp: 0, fp: 1, fn: 1, precision: 0, recall: 0, f1: 0, nearMisses: 1 });
  });

  it("leaves precision UNDEFINED for an arm with no findings, and not 0 or 1", () => {
    const scored = scoreArm({
      records: [makeRecord({ itemId: "silent", text: TAMARIND, findings: [] })],
      gold: [goldRow({ itemId: "silent", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
    });
    for (const rule of MATCH_RULES) {
      const s = scored.byRule[rule];
      expect(s).toMatchObject({ tp: 0, fp: 0, fn: 1 });
      expect(s.precision).toBeUndefined();
      expect(s.precision).not.toBe(0);
      expect(s.precision).not.toBe(1);
      // Recall IS 0: there was something to find and it was not found.
      expect(s.recall).toBe(0);
      // F1 follows precision into undefined even though 2tp/(2tp+fp+fn) would
      // be a defined 0 here. A defined 0 beside an undefined precision reads as
      // a precision of 0, which is the misreading this rule prevents.
      expect(s.f1).toBeUndefined();
    }
    expect(scored.coverage.caveats.join(" ")).toMatch(/precision is UNDEFINED/);
  });

  it("leaves recall undefined when the scored items carry no gold span at all", () => {
    const scored = scoreArm({
      records: [makeRecord({ itemId: "none", text: TAMARIND, findings: [] })],
      gold: [goldRow({ itemId: "none", satisfies: false })],
    });
    for (const rule of MATCH_RULES) {
      expect(scored.byRule[rule].recall).toBeUndefined();
      expect(scored.byRule[rule].precision).toBeUndefined();
      expect(scored.byRule[rule].f1).toBeUndefined();
    }
  });

  it("counts a finding on a true negative as a false positive", () => {
    const scored = scoreArm({
      records: [
        makeRecord({ itemId: "neg", text: TAMARIND, findings: [{ start: 0, end: 3, text: "Can" }] }),
      ],
      gold: [goldRow({ itemId: "neg", satisfies: false })],
    });
    for (const rule of MATCH_RULES) {
      expect(scored.byRule[rule]).toMatchObject({ tp: 0, fp: 1, fn: 0, precision: 0, nearMisses: 0 });
      expect(scored.byRule[rule].recall).toBeUndefined();
      expect(scored.byRule[rule].f1).toBeUndefined();
    }
  });

  it("excludes a disputed row from BOTH the numerator and the denominator", () => {
    const scored = scoreArm({
      records: [
        makeRecord({ itemId: "clean", text: TAMARIND, findings: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
        // A finding on the disputed item that would be a false positive if the
        // row were scored, and a gold call that would be a miss if it were not.
        makeRecord({ itemId: "argued", text: TAMARIND, findings: [{ start: 0, end: 3, text: "Can" }] }),
      ],
      gold: [
        goldRow({ itemId: "clean", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
        goldRow({ itemId: "argued", satisfies: true, status: "disputed" }),
      ],
    });
    expect(scored.coverage.goldRows).toBe(2);
    expect(scored.coverage.goldRowsScored).toBe(1);
    expect(scored.coverage.goldRowsDisputed).toBe(1);
    for (const rule of MATCH_RULES) {
      expect(scored.byRule[rule]).toMatchObject({ tp: 1, fp: 0, fn: 0, precision: 1, recall: 1 });
    }
    expect(scored.coverage.caveats.join(" ")).toMatch(/1 of 2 gold rows are adjudicated disputed/);
  });

  it("excludes a disputed NEGATIVE, which is the only shape any shipped gold actually has", () => {
    // The fixture above is `satisfies: true, status: disputed`, and no row of
    // that shape exists in any committed gold file. All ten disputed rows of
    // injection-p-fin-v2.gold-tier2-predicate.jsonl are `satisfies: false,
    // spans: []`, and smoke.gold-tier2.jsonl has none at all -- so a filter
    // keying on `satisfies` or on `spans.length` instead of on `status` passed
    // the whole suite. MEASURED: both `g.status === "scored" || !g.satisfies`
    // and `... || (g.spans.length === 0 && !g.satisfies)` survived 662 tests.
    const scored = scoreArm({
      records: [
        makeRecord({ itemId: "clean", text: TAMARIND, findings: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
        makeRecord({ itemId: "argued", text: TAMARIND, findings: [{ start: 0, end: 3, text: "Can" }] }),
      ],
      gold: [
        goldRow({ itemId: "clean", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
        goldRow({ itemId: "argued", satisfies: false, status: "disputed" }),
      ],
    });
    expect(scored.coverage.goldRowsScored).toBe(1);
    expect(scored.coverage.goldRowsDisputed).toBe(1);
    // The finding on the disputed row is suppressed, and the count that says so
    // is reported rather than left to be inferred from the row count.
    expect(scored.coverage.findingsInVocabulary).toBe(1);
    expect(scored.coverage.findingsOnDisputedRows).toBe(1);
    for (const rule of MATCH_RULES) {
      expect(scored.byRule[rule], rule).toMatchObject({ tp: 1, fp: 0, fn: 0, precision: 1, recall: 1 });
    }
    expect(scored.coverage.caveats.join(" ")).toMatch(/that exclusion is ONE-SIDED/);
    expect(scored.coverage.caveats.join(" ")).toMatch(/removed 1 of this arm's 2 in-vocabulary findings/);
  });

  it("keeps a row the arm was never asked about out of the RECALL DENOMINATOR, not only out of fn", () => {
    // The previous test asserts `fn === 0`, which cannot see a denominator that
    // adds `goldRowsWithoutRecord` back: with no true positive, recall is
    // undefined either way. MEASURED: `tp / (tp + fn + goldRowsWithoutRecord)`
    // survived the whole apps/eval suite. This fixture has a true positive, so
    // the two forms differ -- 1.0 against 0.5.
    const scored = scoreArm({
      records: [
        makeRecord({ itemId: "asked", text: TAMARIND, findings: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
      ],
      gold: [
        goldRow({ itemId: "asked", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
        goldRow({ itemId: "unasked", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
      ],
    });
    expect(scored.coverage.goldRowsWithoutRecord).toBe(1);
    for (const rule of MATCH_RULES) {
      expect(scored.byRule[rule].fn, rule).toBe(0);
      expect(scored.byRule[rule].recall, rule).toBe(1);
      expect(scored.byRule[rule].f1, rule).toBe(1);
    }
  });

  it("does not count a gold row the arm was never asked about as a miss", () => {
    const scored = scoreArm({
      records: [makeRecord({ itemId: "asked", text: TAMARIND, findings: [] })],
      gold: [
        goldRow({ itemId: "asked", satisfies: false }),
        goldRow({ itemId: "unasked", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] }),
      ],
    });
    expect(scored.coverage.goldRowsWithoutRecord).toBe(1);
    for (const rule of MATCH_RULES) expect(scored.byRule[rule].fn).toBe(0);
    expect(scored.coverage.caveats.join(" ")).toMatch(/never asked/);
  });

  it("ignores a record whose item has no gold row instead of scoring it a false positive", () => {
    const scored = scoreArm({
      records: [
        makeRecord({ itemId: "known", text: TAMARIND, findings: [] }),
        makeRecord({ itemId: "stranger", text: TAMARIND, findings: [{ start: 0, end: 3, text: "Can" }] }),
      ],
      gold: [goldRow({ itemId: "known", satisfies: false })],
    });
    expect(scored.coverage.recordsWithoutGold).toBe(1);
    for (const rule of MATCH_RULES) expect(scored.byRule[rule].fp).toBe(0);
  });

  it("matches one finding to one gold span, not many to one", () => {
    // Two findings on one gold span must be 1 TP + 1 FP, never 2 TP: an arm
    // that says the same thing twice has not been right twice.
    const scored = scoreArm({
      records: [
        makeRecord({
          itemId: "dup",
          text: TAMARIND,
          findings: [
            { start: 43, end: 59, text: "Tamarind Grocers" },
            { start: 43, end: 52, text: "Tamarind " },
          ],
        }),
      ],
      gold: [goldRow({ itemId: "dup", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
    });
    expect(scored.byRule.overlap).toMatchObject({ tp: 1, fp: 1, fn: 0, precision: 0.5, recall: 1 });
  });

  it("does not credit a finding sharing only one offset with the gold span", () => {
    const scored = scoreArm({
      records: [
        makeRecord({ itemId: "short", text: TAMARIND, findings: [{ start: 43, end: 52, text: "Tamarind " }] }),
      ],
      gold: [goldRow({ itemId: "short", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
    });
    // Same start, 9 of 16 characters: an overlap, not an exact hit. IoU 9/16 is
    // above 0.5, which is the one place these two rules part company in the
    // permissive direction rather than the strict one.
    expect(scored.byRule.exact).toMatchObject({ tp: 0, fp: 1, fn: 1, nearMisses: 1 });
    expect(scored.byRule.overlap).toMatchObject({ tp: 1, fp: 0, fn: 0 });
    expect(scored.byRule.iou50).toMatchObject({ tp: 1, fp: 0, fn: 0 });
  });

  it("counts a disjoint finding on a positive item as a plain miss, not a near miss", () => {
    // MEASURED on runs/slate-p-fin-02.tier2only-Phi-4-mini: on
    // pos-multiline-pan-and-client it returned [90,143) -- the PAN sentence --
    // where the gold span is [38,55). One finding, one gold span, zero shared
    // characters: 1 FP and 1 FN under every rule, and NOT a span-convention
    // near miss. `nearMisses` has to tell those apart or it stops meaning
    // "found the right thing at the wrong span".
    const text = "0123456789abcdefghij";
    const scored = scoreArm({
      records: [makeRecord({ itemId: "apart", text, findings: [{ start: 12, end: 18, text: text.slice(12, 18) }] })],
      gold: [goldRow({ itemId: "apart", satisfies: true, spans: [{ start: 0, end: 4, text: text.slice(0, 4) }] })],
    });
    for (const rule of MATCH_RULES) {
      expect(scored.byRule[rule]).toMatchObject({ tp: 0, fp: 1, fn: 1, nearMisses: 0 });
    }
  });

  it("reads answered calls from an Approach-B record's baselineStats too", () => {
    // Half the arms of the slate are Approach B, and B's answered calls land in
    // `baselineStats`, not `tier2Stats`. A coverage counter that read only the
    // compiled field would report all eight B arms as having answered nothing
    // -- which is TRUE of six of them for a different reason, so the bug would
    // have been invisible on exactly the run this scorer was written for.
    const b = makeRecord({ itemId: "b", text: TAMARIND, findings: [] });
    const bRecord = {
      ...b,
      tier2Stats: undefined,
      baselineStats: { ...b.tier2Stats!, messagesJudged: 1, messageBudgetExpiries: 0, unknownEntityTypes: 0 },
    } as unknown as RunRecord;
    const scored = scoreArm({ records: [bRecord], gold: [goldRow({ itemId: "b", satisfies: false })] });
    expect(scored.coverage.itemsJudgeAnswered).toBe(1);
    expect(scored.coverage.itemsJudgeUnanswered).toBe(0);
  });

  it("finds the maximum matching where taking findings in order would not", () => {
    // Under `overlap`: finding 0 = [0,14) touches BOTH gold spans; finding 1 =
    // [0,2) touches only the first. Walking findings in order and taking the
    // first free gold span gives finding 0 the first gold span, after which
    // finding 1 has nowhere to go -- tp 1, fp 1, fn 1. Kuhn's augmenting path
    // pushes finding 0 onto the second gold span and pairs both.
    //
    // AN EARLIER VERSION OF THIS TEST DID NOT DISCRIMINATE. Its fixture made
    // each finding adjacent to exactly one gold span, where greedy and maximum
    // cardinality agree, and a greedy mutant survived it. The asymmetry -- one
    // finding with two options, one with a single shared option -- is the whole
    // content of the test.
    const text = "0123456789abcdefghij";
    const scored = scoreArm({
      records: [
        makeRecord({
          itemId: "cross",
          text,
          findings: [
            { start: 0, end: 14, text: text.slice(0, 14) },
            { start: 0, end: 2, text: text.slice(0, 2) },
          ],
        }),
      ],
      gold: [
        goldRow({
          itemId: "cross",
          satisfies: true,
          spans: [
            { start: 0, end: 4, text: text.slice(0, 4) },
            { start: 10, end: 14, text: text.slice(10, 14) },
          ],
        }),
      ],
    });
    expect(scored.byRule.overlap).toMatchObject({ tp: 2, fp: 0, fn: 0, precision: 1, recall: 1, f1: 1 });
  });

  it("scores only the predicate's own vocabulary and counts what it skipped", () => {
    const scored = scoreArm({
      records: [
        makeRecord({
          itemId: "mixed",
          text: TAMARIND,
          findings: [
            { start: 43, end: 59, text: "Tamarind Grocers", entityType: "client-name", tier: 1 },
            { start: 43, end: 59, text: "Tamarind Grocers" },
          ],
        }),
      ],
      gold: [goldRow({ itemId: "mixed", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
    });
    // The tier-1 client-name finding is at the RIGHT span and is still not a
    // true positive here: it answers a different clause. Scoring it would let
    // an arm win the §3 predicate with a §3-adjacent entity detector.
    expect(scored.coverage.findingsInVocabulary).toBe(1);
    expect(scored.coverage.findingsOutOfVocabulary).toBe(1);
    expect(scored.byRule.exact).toMatchObject({ tp: 1, fp: 0, fn: 0 });
    expect(scored.coverage.findingsInVocabularyByTier).toEqual({ "2": 1 });
  });

  it("separates an arm that answered nothing from one that answered no", () => {
    const scored = scoreArm({
      records: [
        makeRecord({
          itemId: "timedout",
          text: TAMARIND,
          findings: [],
          answeredCalls: 0,
          degraded: [{ tier: 2, reason: "budget-exhausted", detail: "the 5000ms budget ran out" }],
        }),
        makeRecord({ itemId: "answered", text: TAMARIND, findings: [], answeredCalls: 1 }),
      ],
      gold: [goldRow({ itemId: "timedout", satisfies: false }), goldRow({ itemId: "answered", satisfies: false })],
    });
    expect(scored.coverage.itemsJudgeAnswered).toBe(1);
    expect(scored.coverage.itemsJudgeUnanswered).toBe(1);
    expect(scored.coverage.tier2DegradedReasons).toEqual({ "budget-exhausted": 1 });
    expect(scored.coverage.caveats.join(" ")).toMatch(/could not answer, not an arm that answered "no"/);
  });

  it("counts a gold span on an errored record as missed, because the arm was asked", () => {
    const scored = scoreArm({
      records: [makeRecord({ itemId: "broke", text: TAMARIND, findings: [], error: "engine died" })],
      gold: [goldRow({ itemId: "broke", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
    });
    expect(scored.coverage.recordsWithError).toBe(1);
    for (const rule of MATCH_RULES) expect(scored.byRule[rule].fn).toBe(1);
  });
});

describe("scoreArm refusals", () => {
  it("refuses records from more than one arm", () => {
    expect(() =>
      scoreArm({
        records: [
          makeRecord({ itemId: "a", text: TAMARIND, arm: "one" }),
          makeRecord({ itemId: "b", text: TAMARIND, arm: "two" }),
        ],
        gold: [goldRow({ itemId: "a", satisfies: false }), goldRow({ itemId: "b", satisfies: false })],
      }),
    ).toThrow(/score one at a time/);
  });

  it("refuses two records for the same item on one arm", () => {
    expect(() =>
      scoreArm({
        records: [makeRecord({ itemId: "a", text: TAMARIND }), makeRecord({ itemId: "a", text: TAMARIND })],
        gold: [goldRow({ itemId: "a", satisfies: false })],
      }),
    ).toThrow(/silently dropped/);
  });

  it("refuses a run whose policy is not the one the gold was labelled against", () => {
    const wrong = "0".repeat(64);
    expect(() =>
      assertGoldMatchesRecords(
        [goldRow({ itemId: "a", satisfies: false })],
        [makeRecord({ itemId: "a", text: TAMARIND, policyHash: wrong })],
      ),
    ).toThrow(/written about a different document/);
  });

  it("refuses gold whose span does not slice back on the record's own text", () => {
    expect(() =>
      scoreArm({
        // Same offsets, a different message: the gold and the run disagree
        // about what was judged. Every count would still be computable.
        records: [makeRecord({ itemId: "a", text: "a totally different message of its own length here" })],
        gold: [goldRow({ itemId: "a", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })],
      }),
    ).toThrow(/disagree about the message/);
  });

  it("refuses to average two predicates into one ratio", () => {
    const other = { ...goldRow({ itemId: "b", satisfies: false }), predicateId: "other-predicate" };
    expect(() =>
      scoreArm({
        records: [makeRecord({ itemId: "a", text: TAMARIND }), makeRecord({ itemId: "b", text: TAMARIND })],
        gold: [goldRow({ itemId: "a", satisfies: false }), other],
      }),
    ).toThrow(/groupGoldByPredicate/);
  });
});

describe("winnersByRule", () => {
  it("reports the rules disagreeing rather than picking one", () => {
    // `wide` answers at the whole message; `tight` answers at the gold span.
    // Under `overlap` they tie; under `exact` and `iou50` only `tight` scores.
    const gold = [goldRow({ itemId: "i", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })];
    const scored = scoreArms(
      [
        makeRecord({ itemId: "i", text: TAMARIND, arm: "wide", findings: [{ start: 0, end: 74, text: TAMARIND }] }),
        makeRecord({
          itemId: "i",
          text: TAMARIND,
          arm: "tight",
          findings: [{ start: 43, end: 59, text: "Tamarind Grocers" }],
        }),
      ],
      gold,
    );
    const w = winnersByRule(scored);
    expect(w.disagree).toBe(true);
    const at = (r: MatchRule) => w.byRule.find((x) => x.rule === r)!;
    expect(at("overlap").winners).toEqual(["tight", "wide"]);
    expect(at("exact").winners).toEqual(["tight"]);
    expect(at("iou50").winners).toEqual(["tight"]);
  });

  it("does not rank an arm with an undefined F1 last -- it does not rank it", () => {
    const gold = [goldRow({ itemId: "i", satisfies: true, spans: [{ start: 43, end: 59, text: "Tamarind Grocers" }] })];
    const scored = scoreArms(
      [
        makeRecord({
          itemId: "i",
          text: TAMARIND,
          arm: "answers",
          findings: [{ start: 43, end: 59, text: "Tamarind Grocers" }],
        }),
        makeRecord({ itemId: "i", text: TAMARIND, arm: "silent", findings: [], answeredCalls: 0 }),
      ],
      gold,
    );
    const w = winnersByRule(scored);
    for (const r of w.byRule) {
      expect(r.winners).toEqual(["answers"]);
      expect(r.unrankable).toEqual(["silent"]);
    }
  });
});

describe("the trivial floor", () => {
  // Two capitalised bigrams, one of them the gold span. Worked by hand so the
  // expectations are not read back from the readers: a reader returning both
  // scores 1 tp and 1 fp; a reader returning only the first scores 1 tp and 0
  // fp because the gold name comes first here; a whole-message reader overlaps
  // the gold span and matches neither its offsets nor half its union.
  const TEXT = "we are pitching Ashcombe Holdings next month and Bexmoor Associates leases us the floor";
  const SPAN = { start: 16, end: 33, text: "Ashcombe Holdings" };

  it("slices back, so the hand-worked offsets are the ones being scored", () => {
    expect(TEXT.slice(SPAN.start, SPAN.end)).toBe(SPAN.text);
    expect(TEXT.indexOf("Bexmoor Associates")).toBe(49);
  });

  it("scores each reader over the same items, with the same matcher, under all three rules", () => {
    const floors = scoreTrivialFloors([{ text: TEXT, goldSpans: [{ start: SPAN.start, end: SPAN.end }] }]);
    expect(floors.map((f) => f.reader)).toEqual([
      "capitalised-multiword",
      "first-capitalised-multiword",
      "whole-message",
    ]);
    const byId = new Map(floors.map((f) => [f.reader, f]));
    expect(byId.get("capitalised-multiword")!.findings).toBe(2);
    expect(byId.get("first-capitalised-multiword")!.findings).toBe(1);
    expect(byId.get("whole-message")!.findings).toBe(1);
    for (const rule of MATCH_RULES) {
      expect(byId.get("capitalised-multiword")!.byRule[rule], rule).toMatchObject({ tp: 1, fp: 1, fn: 0 });
      expect(byId.get("first-capitalised-multiword")!.byRule[rule], rule).toMatchObject({ tp: 1, fp: 0, fn: 0 });
    }
    expect(byId.get("whole-message")!.byRule.overlap).toMatchObject({ tp: 1, fp: 0, fn: 0 });
    expect(byId.get("whole-message")!.byRule.exact).toMatchObject({ tp: 0, fp: 1, fn: 1 });
    expect(byId.get("whole-message")!.byRule.iou50).toMatchObject({ tp: 0, fp: 1, fn: 1 });
    expect(bestFloorF1(floors, "overlap")).toBe(1);
  });

  it("finds the same thing on every item, and would catch a switch to a shared exec loop", () => {
    // The capitalised readers share one `g` regex. MEASURED on node 26, that is
    // safe with `matchAll`, which iterates a clone and leaves `lastIndex` at 0;
    // it would NOT be safe with `exec`, which advances it, so the second item
    // would be scanned from where the first stopped. This cannot fail against
    // the current implementation and is recorded as such -- it is a guard on
    // the shape, not a claim that the shape is currently at risk.
    const floors = scoreTrivialFloors([
      { text: TEXT, goldSpans: [] },
      { text: TEXT, goldSpans: [] },
      { text: TEXT, goldSpans: [] },
    ]);
    expect(floors.find((f) => f.reader === "capitalised-multiword")!.findings).toBe(6);
    expect(floors.find((f) => f.reader === "first-capitalised-multiword")!.findings).toBe(3);
  });

  it("rides along on every scored arm and names the rules the arm does not beat it under", () => {
    // The floor is not a separate report a writer can forget: it is computed
    // over exactly the rows the arm was scored on, and any rule where the arm
    // fails to beat it becomes a caveat on the arm.
    const scored = scoreArm({
      records: [makeRecord({ itemId: "one", text: TEXT, findings: [{ start: 0, end: 2, text: "we" }] })],
      gold: [goldRow({ itemId: "one", satisfies: true, spans: [SPAN] })],
    });
    expect(scored.floors.map((f) => f.reader)).toContain("capitalised-multiword");
    expect(scored.byRule.overlap.f1).toBe(0);
    const caveats = scored.coverage.caveats.join(" ");
    for (const rule of MATCH_RULES) {
      expect(caveats, rule).toContain(`under ${rule} this arm's F1 (0.000) DOES NOT BEAT the trivial floor`);
    }
  });

  it("says nothing about an arm that beats it", () => {
    const scored = scoreArm({
      records: [makeRecord({ itemId: "one", text: TEXT, findings: [{ start: SPAN.start, end: SPAN.end, text: SPAN.text }] })],
      gold: [goldRow({ itemId: "one", satisfies: true, spans: [SPAN] })],
    });
    expect(scored.byRule.overlap.f1).toBe(1);
    // It ties the perfect floor here rather than beating it, which is exactly
    // what the caveat should say -- "does not beat" and not "loses to".
    expect(scored.coverage.caveats.join(" ")).toContain("DOES NOT BEAT the trivial floor");
    // An arm that DOES beat it: two items with no capitalised word anywhere, so
    // the capitalised readers find nothing at all, and one scored negative, so
    // the whole-message reader buys its overlap recall with a false positive.
    const harder = scoreArm({
      records: [
        makeRecord({ itemId: "one", text: "the answer is xy", findings: [{ start: 14, end: 16, text: "xy" }] }),
        makeRecord({ itemId: "two", text: "nothing here", findings: [] }),
      ],
      gold: [
        goldRow({ itemId: "one", satisfies: true, spans: [{ start: 14, end: 16, text: "xy" }] }),
        goldRow({ itemId: "two", satisfies: false }),
      ],
    });
    expect(harder.byRule.overlap.f1).toBe(1);
    expect(bestFloorF1(harder.floors, "overlap")).toBeCloseTo(2 / 3, 12);
    expect(bestFloorF1(harder.floors, "exact")).toBe(0);
    expect(harder.coverage.caveats.join(" ")).not.toContain("DOES NOT BEAT");
  });
});

describe("groupGoldByPredicate", () => {
  it("keeps predicates apart", () => {
    const a = goldRow({ itemId: "a", satisfies: false });
    const b = { ...goldRow({ itemId: "b", satisfies: false }), predicateId: "other" };
    const grouped = groupGoldByPredicate([a, b]);
    expect([...grouped.keys()].sort()).toEqual(["client-relationship-disclosure", "other"]);
    expect(grouped.get("other")!.map((r) => r.itemId)).toEqual(["b"]);
  });
});
