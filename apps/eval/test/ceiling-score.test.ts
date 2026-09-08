/**
 * `ceiling-score-lib.ts` -- the decisions behind `pnpm -C apps/eval ceiling:score`.
 *
 * WHY THIS FILE EXISTS. Until the logic was split out, `ceiling-score.ts` ended
 * in a bare `main()` and exported nothing, so nothing in the workspace imported
 * it and a mutation review found every mutant planted in it surviving a green
 * suite -- disputed gold rows re-admitted into the positive count, the span
 * pairing allowed to spend one gold span twice, the 512-token comparison column
 * retargeted at this arm's own 600, non-streaming rows folded into the TTFT
 * percentiles. Each of those has a test below named for it.
 *
 * Fixtures are built to be big enough to SEPARATE the rules they exercise: a
 * percentile fixture where two conventions would agree proves nothing, and a
 * pairing fixture with one finding and one gold span cannot tell one-to-one
 * pairing from reuse.
 */
import { describe, expect, it } from "vitest";
import type { CeilingCall } from "../src/driver/ceiling.js";
import type { ScoreableRecord, Tier2GoldRow } from "../src/driver/score.js";
import {
  CEILING_ARM_MAX_TOKENS,
  LOCAL_ARM_MAX_TOKENS,
  MAX_DECODE_BIAS,
  MIN_DECODE_WINDOW_MS,
  attemptedOnlyTable,
  attemptedRecords,
  decodeColumn,
  pairSpansGreedy,
  percentile,
  prf1,
  recordWasAttempted,
  scorePredicate,
  summarizeGold,
  transportRow,
  unansweredCounts,
  type ArmRows,
} from "../src/driver/ceiling-score-lib.js";

const POLICY_HASH = "a".repeat(64);

const annotator = (satisfies: boolean) =>
  ({ satisfies, confidence: "clear" as const, rationale: "fixture" });

/** A gold row. `text` is the message the span offsets index into, kept beside it by the caller. */
function goldRow(o: {
  itemId: string;
  status: "scored" | "disputed";
  satisfies: boolean;
  spans?: { start: number; end: number; text: string }[];
}): Tier2GoldRow {
  return {
    itemId: o.itemId,
    policy: "p-fin",
    policyHash: POLICY_HASH,
    predicateId: "client-relationship-disclosure",
    entityType: "pred:client-relationship-disclosure",
    status: o.status,
    satisfies: o.satisfies,
    confidence: "clear",
    spans: o.spans ?? [],
    adjudication: "fixture",
    annotators: { a: annotator(o.satisfies), b: annotator(o.satisfies) },
  };
}

function record(o: {
  arm: string;
  itemId: string;
  text: string;
  findings?: { start: number; end: number; text: string }[];
  calls?: number;
}): ScoreableRecord {
  return {
    arm: o.arm,
    itemId: o.itemId,
    policyHash: POLICY_HASH,
    text: o.text,
    findings: (o.findings ?? []).map((f) => ({
      entityType: "pred:client-relationship-disclosure",
      start: f.start,
      end: f.end,
      text: f.text,
      tier: 2 as const,
      severity: "medium" as const,
      source: "fixture",
      confidence: 1,
      action: "pseudonymize" as const,
    })),
    error: null,
    calls: Array.from({ length: o.calls ?? 1 }, () => ({})),
  };
}

function call(o: Partial<CeilingCall>): CeilingCall {
  return {
    promptTokens: 100,
    completionTokens: 60,
    reasoningTokens: 0,
    ttftMs: 500,
    wallMs: 900,
    decodeTokPerSec: 150,
    costUsd: 0.0001,
    finishReason: "stop",
    provider: "DeepInfra",
    modelId: "m",
    transport: "stream",
    repair: false,
    parse: "ok",
    retries: [],
    ...o,
  };
}

function ceilingArm(label: string, calls: CeilingCall[][]): ArmRows {
  return {
    label,
    kind: "ceiling",
    records: [],
    ceiling: calls.map((cs, i) => ({
      arm: label,
      requestedProvider: "DeepInfra",
      calls: cs,
      parseFailures: 0,
      repairs: 0,
      wallMs: 1000 + i,
    })) as unknown as ArmRows["ceiling"],
  };
}

describe("summarizeGold — V02: a disputed row is not a positive", () => {
  // Three scored positives, one scored negative, and TWO disputed rows that a
  // naive `gold.filter(g => g.satisfies)` would count. The disputed rows differ
  // from each other in `satisfies` so the fixture separates "excluded because
  // disputed" from "excluded because negative".
  const gold = [
    goldRow({ itemId: "a", status: "scored", satisfies: true, spans: [{ start: 0, end: 4, text: "Acme" }] }),
    goldRow({ itemId: "b", status: "scored", satisfies: true, spans: [{ start: 0, end: 4, text: "Acme" }] }),
    goldRow({ itemId: "c", status: "scored", satisfies: true, spans: [{ start: 0, end: 4, text: "Acme" }] }),
    goldRow({ itemId: "d", status: "scored", satisfies: false }),
    goldRow({ itemId: "e", status: "disputed", satisfies: true }),
    goldRow({ itemId: "f", status: "disputed", satisfies: false }),
  ];

  it("counts positives over SCORED rows only", () => {
    const s = summarizeGold(gold);
    expect(s.positives).toBe(3);
    expect(s.rows).toBe(6);
    expect(s.scored).toBe(4);
    expect(s.disputed).toBe(2);
    expect(s.goldSpans).toBe(3);
  });

  it("would report a different number if disputed rows were admitted", () => {
    // Guards the fixture itself: if every disputed row were negative, the two
    // rules would agree and the assertion above would prove nothing.
    expect(gold.filter((g) => g.satisfies).length).toBe(4);
    expect(summarizeGold(gold).positives).not.toBe(gold.filter((g) => g.satisfies).length);
  });
});

describe("pairSpansGreedy — V03: one gold span cannot be spent twice", () => {
  const gold = [{ start: 10, end: 20 }];

  it("scores two findings on ONE gold span as one tp and one fp", () => {
    const counts = pairSpansGreedy("overlap", [{ start: 10, end: 20 }, { start: 12, end: 18 }], gold);
    expect(counts).toEqual({ tp: 1, fp: 1, fn: 0 });
    // Precision must reflect the duplicate. Reusing the gold span gives 1.000.
    expect(prf1(counts).precision).toBe(0.5);
  });

  it("scores three findings on TWO gold spans as two tp and one fp", () => {
    const counts = pairSpansGreedy(
      "overlap",
      [{ start: 0, end: 5 }, { start: 1, end: 4 }, { start: 10, end: 15 }],
      [{ start: 0, end: 5 }, { start: 10, end: 15 }],
    );
    expect(counts).toEqual({ tp: 2, fp: 1, fn: 0 });
  });

  it("counts an unmatched gold span as a false negative", () => {
    expect(pairSpansGreedy("overlap", [{ start: 0, end: 5 }], [{ start: 0, end: 5 }, { start: 40, end: 50 }])).toEqual({
      tp: 1,
      fp: 0,
      fn: 1,
    });
  });

  it("honours the match rule it is given, and the rules disagree on this fixture", () => {
    const findings = [{ start: 11, end: 19 }];
    // Deliberately a partial overlap: contained in the gold span but neither
    // identical to it nor at IoU 0.5 (8/10 characters would pass iou50, so the
    // fixture uses a span short enough to fail it).
    const narrow = [{ start: 11, end: 14 }];
    expect(pairSpansGreedy("overlap", findings, gold).tp).toBe(1);
    expect(pairSpansGreedy("exact", findings, gold).tp).toBe(0);
    expect(pairSpansGreedy("iou50", narrow, gold).tp).toBe(0);
    expect(pairSpansGreedy("overlap", narrow, gold).tp).toBe(1);
  });
});

describe("percentile", () => {
  it("uses the same convention for p50 and p95 on a fixture where p95 and p96 differ", () => {
    // n = 100 so that floor(0.95 * n) = 95 and floor(0.96 * n) = 96 select
    // different elements. At n = 25 they would not, and the test would pass
    // against a mutated quantile.
    const xs = Array.from({ length: 100 }, (_, i) => i + 1);
    expect(percentile(xs, 0.5)).toBe(51);
    expect(percentile(xs, 0.95)).toBe(96);
    expect(percentile(xs, 0.96)).toBe(97);
  });

  it("never indexes past the end, and is undefined on an empty set", () => {
    expect(percentile([7], 0.95)).toBe(7);
    expect(percentile([], 0.5)).toBeUndefined();
  });
});

describe("transportRow — V04: `calls >=512` counts the LOCAL cap, not this arm's 600", () => {
  it("the two caps are different numbers", () => {
    expect(LOCAL_ARM_MAX_TOKENS).toBe(512);
    expect(CEILING_ARM_MAX_TOKENS).toBe(600);
    expect(LOCAL_ARM_MAX_TOKENS).not.toBe(CEILING_ARM_MAX_TOKENS);
  });

  it("counts calls at or above 512, including the ones strictly between 512 and 600", () => {
    // 400 | 511 below; 512 at the boundary; 550 and 599 strictly between the
    // two caps -- these are the only calls that can tell the two constants
    // apart; 600 at this arm's own cap. Counting at 600 would report 1.
    const arm = ceilingArm("arm", [
      [
        call({ completionTokens: 400 }),
        call({ completionTokens: 511 }),
        call({ completionTokens: 512 }),
        call({ completionTokens: 550 }),
        call({ completionTokens: 599 }),
        call({ completionTokens: 600 }),
      ],
    ]);
    const t = transportRow(arm)!;
    expect(t.atLocalCap).toBe(4);
    expect(t.calls).toBe(6);
  });
});

describe("transportRow — V05: a non-streaming call has no TTFT and is excluded, not substituted", () => {
  it("keeps the fallback's wall time out of the TTFT percentiles", () => {
    // Three streaming calls with tight TTFTs and two fallbacks whose whole wall
    // is two orders of magnitude larger. Folding the fallbacks in as `ttftMs ??
    // wallMs` moves p95 from 120 to 9000.
    const arm = ceilingArm("arm", [
      [
        call({ ttftMs: 100, wallMs: 700 }),
        call({ ttftMs: 110, wallMs: 700 }),
        call({ ttftMs: 120, wallMs: 700 }),
        call({ transport: "non-stream-fallback", ttftMs: null, decodeTokPerSec: null, wallMs: 9000 }),
        call({ transport: "non-stream-fallback", ttftMs: null, decodeTokPerSec: null, wallMs: 9000 }),
      ],
    ]);
    const t = transportRow(arm)!;
    expect(t.ttftP50).toBe(110);
    expect(t.ttftP95).toBe(120);
    expect(t.fellBack).toBe(2);
    // The wall percentiles DO include them: that column times every attempt.
    expect(t.wallP95).toBe(9000);
  });
});

describe("transportRow — the pin is checked against the response, not asserted", () => {
  it("reports NO when any answered call names a provider other than the pin", () => {
    const ok = transportRow(ceilingArm("arm", [[call({ provider: "DeepInfra" }), call({ provider: "DeepInfra" })]]))!;
    expect(ok.pinHonoured).toBe(true);
    const bad = transportRow(ceilingArm("arm", [[call({ provider: "DeepInfra" }), call({ provider: "Together" })]]))!;
    expect(bad.pinHonoured).toBe(false);
    expect(bad.providers).toEqual(["DeepInfra", "Together"]);
  });

  it("returns undefined for an arm that made no call at all", () => {
    expect(transportRow(ceilingArm("arm", [[]]))).toBeUndefined();
  });
});

describe("decodeColumn — Sec 7.6: the rate is suppressed when the window is not a measurement", () => {
  /** `judge-deepseek-v4-flash-0731 [ceiling-01]`: 7 tokens over a 9.6 ms window. */
  const judgeLike = Array.from({ length: 20 }, () =>
    call({ completionTokens: 7, ttftMs: 1198, wallMs: 1198 + 9.6, decodeTokPerSec: 726.7 }),
  );
  /** `b-nemotron-3-super-120b-a12b [ceiling-01]`: 57 tokens over a 4,204 ms window. */
  const approachBLike = Array.from({ length: 20 }, () =>
    call({ completionTokens: 57, ttftMs: 745, wallMs: 745 + 4203.5, decodeTokPerSec: 13.7 }),
  );

  it("suppresses the rate below the threshold and reports the window that decided it", () => {
    const d = decodeColumn(judgeLike);
    expect(d.suppressed).toBe(true);
    expect(d.tokPerSecP50).toBeUndefined();
    expect(d.windowMsP50).toBeCloseTo(9.6, 6);
    expect(d.shortWindowCalls).toBe(20);
  });

  it("keeps the rate above the threshold", () => {
    const d = decodeColumn(approachBLike);
    expect(d.suppressed).toBe(false);
    expect(d.tokPerSecP50).toBe(13.7);
    expect(d.windowMsP50).toBeCloseTo(4203.5, 6);
    expect(d.shortWindowCalls).toBe(0);
    expect(d.biased).toBe(false);
  });

  it("the two guards are INDEPENDENT: a sound window can still carry a biased rate", () => {
    // `judge-nemotron-3-super-120b-a12b`: median window 491.3 ms -- comfortably
    // above the threshold, shortest window 66.1 ms -- but n = 7, so the
    // published formula overstates by 1/(n-1) = 16.7%. A single guard on the
    // window prints this number unflagged.
    const d = decodeColumn(
      Array.from({ length: 20 }, () => call({ completionTokens: 7, ttftMs: 755, wallMs: 755 + 491.3, decodeTokPerSec: 14.3 })),
    );
    expect(d.suppressed).toBe(false);
    expect(d.tokPerSecP50).toBe(14.3);
    expect(d.biased).toBe(true);
    expect(d.biasAtMedianN).toBeCloseTo(1 / 6, 9);
  });

  it("the two guards are INDEPENDENT: a short window can carry an unbiased rate", () => {
    // `judge-glm-5.3-flash [glmon-01]`: 191 completion tokens (thinking on) over
    // a 181.5 ms median window. Bias 0.5%, and the window clears the threshold,
    // so nothing fires. A guard that keyed on the judge FAMILY would suppress
    // this arm; a guard keyed on the token count alone would not flag
    // judge-deepseek. Neither substitutes for the other.
    const d = decodeColumn(
      Array.from({ length: 20 }, () => call({ completionTokens: 191, ttftMs: 4476, wallMs: 4476 + 181.5, decodeTokPerSec: 1017.4 })),
    );
    expect(d.suppressed).toBe(false);
    expect(d.biased).toBe(false);
    expect(d.biasAtMedianN!).toBeLessThan(MAX_DECODE_BIAS);
  });

  it("excludes a call of one token or fewer, which has no decode window to measure", () => {
    // Under the corrected `(n-1)/window` formula an n <= 1 call has no defined
    // rate. The rows on disk predate that correction and still carry one.
    const d = decodeColumn([
      call({ completionTokens: 1, ttftMs: 10, wallMs: 10.5, decodeTokPerSec: 2000 }),
      call({ completionTokens: 0, ttftMs: 10, wallMs: 10.5, decodeTokPerSec: 1 }),
      call({ completionTokens: 60, ttftMs: 100, wallMs: 100 + 400, decodeTokPerSec: 150 }),
    ]);
    expect(d.streamingCalls).toBe(1);
    expect(d.tokPerSecP50).toBe(150);
    expect(d.windowMsP50).toBeCloseTo(400, 6);
  });

  it("excludes non-streaming calls, which carry no window by construction", () => {
    const d = decodeColumn([
      call({ transport: "non-stream-fallback", ttftMs: null, decodeTokPerSec: null, wallMs: 9000 }),
      call({ completionTokens: 60, ttftMs: 100, wallMs: 100 + 400, decodeTokPerSec: 150 }),
    ]);
    expect(d.streamingCalls).toBe(1);
    expect(d.windowMsP50).toBeCloseTo(400, 6);
  });

  it("the threshold is the stated one", () => {
    expect(MIN_DECODE_WINDOW_MS).toBe(100);
    const just = (w: number) =>
      decodeColumn(Array.from({ length: 5 }, () => call({ completionTokens: 60, ttftMs: 100, wallMs: 100 + w })));
    expect(just(MIN_DECODE_WINDOW_MS - 0.1).suppressed).toBe(true);
    expect(just(MIN_DECODE_WINDOW_MS).suppressed).toBe(false);
  });
});

describe("recordWasAttempted / attemptedRecords", () => {
  it("an item the provider refused is not an item the model answered", () => {
    expect(recordWasAttempted(record({ arm: "a", itemId: "i", text: "x", calls: 0 }))).toBe(false);
    expect(recordWasAttempted(record({ arm: "a", itemId: "i", text: "x", calls: 1 }))).toBe(true);
  });

  it("uses the same union as `itemsJudgeAnswered`, so a browser arm counts too", () => {
    const base = record({ arm: "a", itemId: "i", text: "x", calls: 0 });
    expect(recordWasAttempted({ ...base, tier2Stats: { calls: [{}] } })).toBe(true);
    expect(recordWasAttempted({ ...base, baselineStats: { calls: [{}] } })).toBe(true);
    expect(recordWasAttempted({ ...base, tier2Stats: { calls: [] }, baselineStats: { calls: [] } })).toBe(false);
  });

  it("keeps only the answered records", () => {
    const rs = [
      record({ arm: "a", itemId: "1", text: "x", calls: 1 }),
      record({ arm: "a", itemId: "2", text: "x", calls: 0 }),
      record({ arm: "a", itemId: "3", text: "x", calls: 2 }),
    ];
    expect(attemptedRecords(rs).map((r) => r.itemId)).toEqual(["1", "3"]);
  });
});

describe("unansweredCounts", () => {
  const text = "Acme Holdings retained us last spring.";
  const gold = [
    goldRow({ itemId: "1", status: "scored", satisfies: true, spans: [{ start: 0, end: 13, text: "Acme Holdings" }] }),
    goldRow({ itemId: "2", status: "scored", satisfies: true, spans: [{ start: 0, end: 13, text: "Acme Holdings" }] }),
    goldRow({ itemId: "3", status: "scored", satisfies: false }),
    goldRow({ itemId: "4", status: "disputed", satisfies: true }),
  ];

  it("counts unanswered scored rows and, separately, the positives among them", () => {
    // Item 2 (a positive) and item 3 (a negative) went unanswered; item 4 is
    // disputed and is excluded from every metric, so it must not be counted
    // even though it too went unanswered.
    const records = [
      record({ arm: "a", itemId: "1", text, calls: 1 }),
      record({ arm: "a", itemId: "2", text, calls: 0 }),
      record({ arm: "a", itemId: "3", text, calls: 0 }),
      record({ arm: "a", itemId: "4", text, calls: 0 }),
    ];
    expect(unansweredCounts(records, gold)).toEqual({ unanswered: 2, unansweredPositives: 1 });
  });

  it("does not count a gold row the arm has no record for as unanswered", () => {
    // A sliced run is not a rate-limited one. The arm was never asked.
    expect(unansweredCounts([record({ arm: "a", itemId: "1", text, calls: 1 })], gold)).toEqual({
      unanswered: 0,
      unansweredPositives: 0,
    });
  });
});

describe("attempted-only scoring re-scores the FLOOR over the same rows", () => {
  /**
   * The point of Sec 7.4's correction. The quick figures in the write-up adjust
   * the arm and leave the floor at its whole-gold value; that is not
   * like-for-like, and this fixture is built so the difference is visible.
   *
   * Item 3 is the one the provider rate-limited away, and it is chosen to be a
   * row the FLOOR gets WRONG: its gold entity is lowercase, so every
   * capitalised-multiword reader misses it and fires on an unrelated
   * capitalised name instead. Dropping it therefore moves the floor as well as
   * the arm, which is exactly the effect a not-like-for-like adjustment hides.
   */
  const withCap = "Acme Holdings retained us last spring.";
  const noCap = "we spoke to them about it yesterday afternoon.";
  // Offsets are COMPUTED, never hand-counted: `scoreArm` refuses a gold span
  // whose offsets do not slice back to its own text, and a fixture that got
  // them wrong would fail as a schema error rather than as the assertion below.
  const lowerPrefix = "the retainer for ";
  const lowerEntity = "acme holdings";
  const lowerCap = `${lowerPrefix}${lowerEntity} was signed by Big Corp.`;
  const lowerSpan = { start: lowerPrefix.length, end: lowerPrefix.length + lowerEntity.length, text: lowerEntity };
  const capSpan = { start: 0, end: 13, text: "Acme Holdings" };
  const gold = [
    goldRow({ itemId: "1", status: "scored", satisfies: true, spans: [capSpan] }),
    goldRow({ itemId: "2", status: "scored", satisfies: false }),
    goldRow({ itemId: "3", status: "scored", satisfies: true, spans: [lowerSpan] }),
  ];
  const arm: ArmRows = {
    label: "rate-limited-arm",
    kind: "ceiling",
    records: [
      record({ arm: "x", itemId: "1", text: withCap, findings: [capSpan], calls: 1 }),
      record({ arm: "x", itemId: "2", text: noCap, calls: 1 }),
      // Answered by nobody: `calls: []`, exactly the 429 shape.
      record({ arm: "x", itemId: "3", text: lowerCap, calls: 0 }),
    ],
    ceiling: [],
  };

  const [row] = attemptedOnlyTable([arm], gold, "overlap");

  it("the fixture's gold span really is invisible to the floor readers", () => {
    // Guards the test itself. If the floor happened to find the lowercase
    // entity, the two floor figures would coincide and the assertion below
    // would be vacuous.
    expect(/\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b/.exec(lowerCap)?.[0]).toBe("Big Corp");
    expect(lowerCap.slice(lowerSpan.start, lowerSpan.end)).toBe(lowerEntity);
  });

  it("reports the arm's unanswered rows and the positives among them", () => {
    expect(row!.unanswered).toBe(1);
    expect(row!.unansweredPositives).toBe(1);
    expect(row!.attempted).toBe(2);
  });

  it("moves the FLOOR as well as the arm", () => {
    // This is the assertion the not-like-for-like version fails: adjusting only
    // the arm leaves `attemptedBestFloorF1` equal to `wholeGoldBestFloorF1`.
    expect(row!.attemptedBestFloorF1).not.toBe(row!.wholeGoldBestFloorF1);
    // Whole gold, best reader is `whole-message`: tp 2, fp 1, fn 0 -> F1 0.8.
    expect(row!.wholeGoldBestFloorF1).toBeCloseTo(0.8, 9);
    // Over the two answered rows the capitalised readers are perfect: tp 1, fp 0, fn 0.
    expect(row!.attemptedBestFloorF1).toBe(1);
    expect(row!.attemptedBestFloorF1!).toBeGreaterThan(row!.wholeGoldBestFloorF1!);
  });

  it("raises the arm's own F1 by dropping the row it was never allowed to attempt", () => {
    expect(row!.wholeGoldF1).toBeCloseTo(2 / 3, 9); // 1 tp, 0 fp, 1 fn
    expect(row!.attemptedF1).toBe(1); // 1 tp, 0 fp, 0 fn
  });

  it("still reports the whole-gold column, so the unanswered rows are not defined away", () => {
    expect(row!.wholeGoldF1).toBeDefined();
    expect(row!.wholeGoldBestFloorF1).toBeDefined();
  });
});

describe("scorePredicate — V01: the whole-gold floor is the FIRST arm's, stated once", () => {
  /**
   * On the run artifacts every arm carries a record for all 179 scored rows, so
   * `scoreTrivialFloors` sees the same items whichever arm supplies them and
   * `??=` versus `=` is an equivalent mutation THERE. This fixture makes the
   * subsets differ, which is the case attempted-only scoring creates, so the
   * two are distinguishable.
   */
  const withCap = "Acme Holdings retained us last spring.";
  const noCap = "we spoke to them about it yesterday afternoon.";
  const gold = [
    goldRow({ itemId: "1", status: "scored", satisfies: true, spans: [{ start: 0, end: 13, text: "Acme Holdings" }] }),
    goldRow({ itemId: "2", status: "scored", satisfies: false }),
  ];
  const wide: ArmRows = {
    label: "wide",
    kind: "ceiling",
    records: [
      record({ arm: "wide", itemId: "1", text: withCap, calls: 1 }),
      record({ arm: "wide", itemId: "2", text: noCap, calls: 1 }),
    ],
    ceiling: [],
  };
  const narrow: ArmRows = {
    label: "narrow",
    kind: "ceiling",
    records: [record({ arm: "narrow", itemId: "1", text: withCap, calls: 1 })],
    ceiling: [],
  };

  it("the two arms' own floors differ on this fixture", () => {
    // Guards the test: without this, taking the first or the last arm's floor
    // would be indistinguishable and the assertion below would be vacuous.
    const a = scorePredicate([wide], gold).floors!;
    const b = scorePredicate([narrow], gold).floors!;
    expect(a.find((f) => f.reader === "whole-message")!.findings).toBe(2);
    expect(b.find((f) => f.reader === "whole-message")!.findings).toBe(1);
  });

  it("takes the first arm's floors and does not overwrite them with the last arm's", () => {
    const first = scorePredicate([wide, narrow], gold).floors!;
    expect(first.find((f) => f.reader === "whole-message")!.findings).toBe(2);
    const reversed = scorePredicate([narrow, wide], gold).floors!;
    expect(reversed.find((f) => f.reader === "whole-message")!.findings).toBe(1);
  });

  it("sorts arms by overlap F1, descending", () => {
    const { lines } = scorePredicate([narrow, wide], gold);
    expect(lines.map((l) => l.label).sort()).toEqual(["narrow", "wide"]);
  });
});

describe("prf1", () => {
  it("leaves precision undefined when nothing was predicted and recall undefined when there is no gold", () => {
    expect(prf1({ tp: 0, fp: 0, fn: 3 })).toEqual({ precision: undefined, recall: 0, f1: undefined });
    expect(prf1({ tp: 0, fp: 2, fn: 0 })).toEqual({ precision: 0, recall: undefined, f1: undefined });
    expect(prf1({ tp: 0, fp: 0, fn: 0 })).toEqual({ precision: undefined, recall: undefined, f1: undefined });
  });

  it("computes the harmonic mean, which is not the arithmetic one", () => {
    const s = prf1({ tp: 3, fp: 1, fn: 3 });
    expect(s.precision).toBe(0.75);
    expect(s.recall).toBe(0.5);
    expect(s.f1).toBeCloseTo(0.6, 9);
    expect(s.f1).not.toBeCloseTo(0.625, 3);
  });
});
