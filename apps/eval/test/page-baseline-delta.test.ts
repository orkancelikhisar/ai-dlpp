import { describe, expect, it } from "vitest";
import type { BaselineStats, JudgeCallRecord } from "@sih/tier2";
import { baselineDelta } from "../src/page/baseline-delta.js";

/**
 * `baselineDelta`: the projection that turns `BaselineB`'s CUMULATIVE counters
 * into one `detect`'s worth of work.
 *
 * The twin of `page-judge-delta.test.ts`, and it exists for the same reason
 * that file does: the browser suite loads a fresh page and does one `detect`, so
 * `before` is all-zero there and a projection that returns the TOTAL is
 * indistinguishable from one that returns the difference. Everything below uses
 * a nonzero `before` and all-distinct values, so a counter wired into the wrong
 * slot is visible rather than merely possible.
 *
 * It carries one assertion `judgeDelta`'s cannot: that the two projections are
 * over DIFFERENT counter sets. `BaselineStats` renames `segmentsJudged` to
 * `messagesJudged` and `unknownPredicates` to `unknownEntityTypes`, adds
 * `messageBudgetExpiries` and drops `segmentsSkipped` -- so a copy-paste of the
 * judge's projection compiles for twelve fields and silently reports nothing for
 * the three that matter most to a head-to-head.
 */

const CALL = (over: Partial<JudgeCallRecord> = {}): JudgeCallRecord => ({
  finishReason: "stop",
  promptTokens: 2860,
  completionTokens: 31,
  ttftMs: 1900,
  decodeTokPerSec: 40,
  ...over,
});

/** The arm's state after one message, every counter at a different value. */
const BEFORE: BaselineStats = {
  rung1: 3,
  rung2: 1,
  unresolvedQuotes: 2,
  unknownEntityTypes: 6,
  duplicatesDropped: 4,
  repairAttempts: 5,
  failedClosed: 7,
  truncatedResponses: 8,
  abortedResponses: 9,
  messagesJudged: 10,
  deadlineExpiries: 12,
  messageBudgetExpiries: 15,
  callerAbortsMidGeneration: 13,
  callerAbortsWhileQueued: 14,
  calls: [CALL({ promptTokens: 1 }), CALL({ promptTokens: 2 })],
};

/** The SECOND message's own work, none of it equal to any `BEFORE` value. */
const OWN = {
  rung1: 20,
  rung2: 21,
  unresolvedQuotes: 22,
  unknownEntityTypes: 23,
  duplicatesDropped: 24,
  repairAttempts: 25,
  failedClosed: 26,
  truncatedResponses: 27,
  abortedResponses: 28,
  messagesJudged: 29,
  deadlineExpiries: 31,
  messageBudgetExpiries: 34,
  callerAbortsMidGeneration: 32,
  callerAbortsWhileQueued: 33,
} as const;

const NEW_CALLS = [CALL({ promptTokens: 3 }), CALL({ promptTokens: 4 })];

const AFTER: BaselineStats = {
  rung1: BEFORE.rung1 + OWN.rung1,
  rung2: BEFORE.rung2 + OWN.rung2,
  unresolvedQuotes: BEFORE.unresolvedQuotes + OWN.unresolvedQuotes,
  unknownEntityTypes: BEFORE.unknownEntityTypes + OWN.unknownEntityTypes,
  duplicatesDropped: BEFORE.duplicatesDropped + OWN.duplicatesDropped,
  repairAttempts: BEFORE.repairAttempts + OWN.repairAttempts,
  failedClosed: BEFORE.failedClosed + OWN.failedClosed,
  truncatedResponses: BEFORE.truncatedResponses + OWN.truncatedResponses,
  abortedResponses: BEFORE.abortedResponses + OWN.abortedResponses,
  messagesJudged: BEFORE.messagesJudged + OWN.messagesJudged,
  deadlineExpiries: BEFORE.deadlineExpiries + OWN.deadlineExpiries,
  messageBudgetExpiries: BEFORE.messageBudgetExpiries + OWN.messageBudgetExpiries,
  callerAbortsMidGeneration: BEFORE.callerAbortsMidGeneration + OWN.callerAbortsMidGeneration,
  callerAbortsWhileQueued: BEFORE.callerAbortsWhileQueued + OWN.callerAbortsWhileQueued,
  calls: [...BEFORE.calls, ...NEW_CALLS],
};

describe("baselineDelta", () => {
  it("reports the second message's own work, not the arm's running total", () => {
    // The whole object in one assertion rather than fourteen: `toEqual` on the
    // literal is what makes a counter wired into the wrong slot fail, which
    // per-field assertions written in the implementation's own order would not.
    expect(baselineDelta(BEFORE, AFTER)).toEqual({ ...OWN, calls: NEW_CALLS });
  });

  it("takes calls as a SUFFIX, so the previous message's rows are not counted twice", () => {
    // `createArm` in `baselineB.ts` only ever appends, so this message's rows
    // are everything past the old length. Handing back the whole history would
    // duplicate every earlier row into this row -- and `bakeoff.ts` computes
    // the p95 TTFT, the decode rate and the truncation accounting from exactly
    // these rows, so over an n-item corpus the over-count is quadratic.
    const delta = baselineDelta(BEFORE, AFTER);
    expect(delta.calls).toHaveLength(2);
    // By identity, not by shape: the rows are otherwise near-identical, so a
    // shape comparison would accept the first two of four.
    expect(delta.calls.map((c) => c.promptTokens)).toEqual([3, 4]);
    expect(delta.calls[0]).toBe(NEW_CALLS[0]);
  });

  it("reports zeros for a detect that added nothing, without inventing rows", () => {
    const delta = baselineDelta(BEFORE, { ...BEFORE, calls: [...BEFORE.calls] });
    expect(delta.calls).toEqual([]);
    const { calls: _calls, ...counters } = delta;
    expect(Object.values(counters)).toEqual(new Array(14).fill(0));
  });

  it("projects every counter BaselineStats declares, so an upstream addition cannot go unreported", () => {
    // The runtime half of the claim the `satisfies`-free listing makes at
    // compile time: the key set of the delta is the key set of the stats, in
    // both directions -- a field dropped from the projection and a field
    // invented by it are both failures here.
    expect(Object.keys(baselineDelta(BEFORE, AFTER)).sort()).toEqual(Object.keys(AFTER).sort());
  });

  it("is over a DIFFERENT counter set from the judge's, which is why it is a second function", () => {
    // The assertion that would catch a copy-paste of `judgeDelta`. Three of
    // these four names are the whole reason `baselineStats` is a separate
    // record field: a B arm's message count under a field named for segments
    // would be one event reported under another event's name, and nothing
    // downstream could notice -- both are small non-negative integers.
    const keys = Object.keys(baselineDelta(BEFORE, AFTER));
    expect(keys).toContain("messagesJudged");
    expect(keys).toContain("unknownEntityTypes");
    expect(keys).toContain("messageBudgetExpiries");
    expect(keys).not.toContain("segmentsJudged");
    expect(keys).not.toContain("unknownPredicates");
    expect(keys).not.toContain("segmentsSkipped");
  });
});
