import { describe, expect, it } from "vitest";
import type { JudgeCallRecord, JudgeStats } from "@sih/tier2";
import { judgeDelta } from "../src/page/judge-delta.js";

/**
 * `judgeDelta`: the projection that turns `WebLlmJudge`'s CUMULATIVE counters
 * into one `detect`'s worth of work.
 *
 * ## Why this file exists at all
 *
 * Because the browser suite cannot test it. `test/tier2.spec.ts` loads a fresh
 * page per test, loads an arm (which builds a fresh judge), and then does
 * exactly ONE `detect` -- so `before` is all-zero on every path there,
 * `after - before` and `after` agree for all fourteen counters, and
 * `after.calls.slice(0)` and `after.calls` agree too. The review that found
 * this measured it against the pre-extraction code -- replacing all fourteen
 * counters with literals (`rung1: 0 ... segmentsJudged: 1`) left every tier-2
 * test passing, and so did replacing the `calls` suffix with the judge's whole
 * history, the latter while the bake-off's own console line moved from 3 calls
 * to 4. Those are its numbers; what was re-run here is that each of those three
 * mutations now fails in this file.
 *
 * Everything below therefore uses a NONZERO `before`, which is the state a
 * second message on one arm's judge actually starts from, and values that are
 * all DISTINCT, so a counter projected into the wrong slot is visible. A
 * fixture of zeros cannot tell a producer that subtracts from one that copies.
 */

const CALL = (over: Partial<JudgeCallRecord> = {}): JudgeCallRecord => ({
  finishReason: "stop",
  promptTokens: 1031,
  completionTokens: 24,
  ttftMs: 480,
  decodeTokPerSec: 40,
  ...over,
});

/**
 * The judge's state after one message, with every counter at a different
 * value and two call rows behind it.
 */
const BEFORE: JudgeStats = {
  rung1: 3,
  rung2: 1,
  unresolvedQuotes: 2,
  unknownPredicates: 6,
  duplicatesDropped: 4,
  repairAttempts: 5,
  failedClosed: 7,
  truncatedResponses: 8,
  abortedResponses: 9,
  segmentsJudged: 10,
  segmentsSkipped: 11,
  deadlineExpiries: 12,
  callerAbortsMidGeneration: 13,
  callerAbortsWhileQueued: 14,
  calls: [CALL({ promptTokens: 1 }), CALL({ promptTokens: 2 })],
};

/**
 * The SECOND message's own work, again all distinct and none equal to any
 * `BEFORE` value -- so a projection that returned the total, the difference or
 * a constant produces three different answers here.
 */
const OWN = {
  rung1: 20,
  rung2: 21,
  unresolvedQuotes: 22,
  unknownPredicates: 23,
  duplicatesDropped: 24,
  repairAttempts: 25,
  failedClosed: 26,
  truncatedResponses: 27,
  abortedResponses: 28,
  segmentsJudged: 29,
  segmentsSkipped: 30,
  deadlineExpiries: 31,
  callerAbortsMidGeneration: 32,
  callerAbortsWhileQueued: 33,
} as const;

const NEW_CALLS = [CALL({ promptTokens: 3 }), CALL({ promptTokens: 4 }), CALL({ promptTokens: 5 })];

const AFTER: JudgeStats = {
  rung1: BEFORE.rung1 + OWN.rung1,
  rung2: BEFORE.rung2 + OWN.rung2,
  unresolvedQuotes: BEFORE.unresolvedQuotes + OWN.unresolvedQuotes,
  unknownPredicates: BEFORE.unknownPredicates + OWN.unknownPredicates,
  duplicatesDropped: BEFORE.duplicatesDropped + OWN.duplicatesDropped,
  repairAttempts: BEFORE.repairAttempts + OWN.repairAttempts,
  failedClosed: BEFORE.failedClosed + OWN.failedClosed,
  truncatedResponses: BEFORE.truncatedResponses + OWN.truncatedResponses,
  abortedResponses: BEFORE.abortedResponses + OWN.abortedResponses,
  segmentsJudged: BEFORE.segmentsJudged + OWN.segmentsJudged,
  segmentsSkipped: BEFORE.segmentsSkipped + OWN.segmentsSkipped,
  deadlineExpiries: BEFORE.deadlineExpiries + OWN.deadlineExpiries,
  callerAbortsMidGeneration: BEFORE.callerAbortsMidGeneration + OWN.callerAbortsMidGeneration,
  callerAbortsWhileQueued: BEFORE.callerAbortsWhileQueued + OWN.callerAbortsWhileQueued,
  calls: [...BEFORE.calls, ...NEW_CALLS],
};

describe("judgeDelta", () => {
  it("reports the second message's own work, not the judge's running total", () => {
    // The whole object in one assertion rather than fourteen: `toEqual` on the
    // literal is what makes a counter wired into the wrong slot -- rung2 read
    // from rung1's subtraction, say -- fail, which per-field assertions written
    // in the same order as the implementation would not.
    expect(judgeDelta(BEFORE, AFTER)).toEqual({ ...OWN, calls: NEW_CALLS });
  });

  it("takes calls as a SUFFIX, so the previous message's rows are not counted twice", () => {
    // The documented rule, and the one no browser fixture reaches: `WebLlmJudge`
    // only ever appends, so this message's rows are everything past the old
    // length. Handing back the whole history instead would duplicate every
    // earlier row into this row's `calls` -- and `bakeoff.ts` computes the p95
    // TTFT, the decode rate and the truncation accounting from exactly these
    // rows, so on a 17-item corpus the over-count is quadratic.
    const delta = judgeDelta(BEFORE, AFTER);
    expect(delta.calls).toHaveLength(3);
    // By identity, not by shape: the rows are otherwise near-identical, so a
    // shape comparison would accept the first two of five.
    expect(delta.calls.map((c) => c.promptTokens)).toEqual([3, 4, 5]);
    expect(delta.calls[0]).toBe(NEW_CALLS[0]);
  });

  it("reports zeros for a detect that added nothing, without inventing rows", () => {
    // The honest reading of a `detect` whose escalation selected no segment:
    // the judge was in the loop and did nothing. Distinct from `lastDetect`
    // being absent, which means no `detect` has run at all.
    const delta = judgeDelta(BEFORE, { ...BEFORE, calls: [...BEFORE.calls] });
    expect(delta.calls).toEqual([]);
    const { calls: _calls, ...counters } = delta;
    expect(Object.values(counters)).toEqual(new Array(14).fill(0));
  });

  it("projects every counter JudgeStats declares, so an upstream addition cannot go unreported", () => {
    // `judgeDelta` lists its fields rather than looping, which is what makes a
    // counter added to `JudgeStats` a compile error there. This is the runtime
    // half of the same claim: the key set of the delta is the key set of the
    // stats. Both directions -- a field dropped from the projection and a field
    // invented by it are both failures here.
    const delta = judgeDelta(BEFORE, AFTER);
    expect(Object.keys(delta).sort()).toEqual(Object.keys(AFTER).sort());
  });
});
