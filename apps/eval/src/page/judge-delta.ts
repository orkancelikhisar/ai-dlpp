import type { JudgeCallRecord, JudgeStats } from "@sih/tier2";

/** What one `detect` call cost tier 2, as deltas over the judge's running counters. */
export interface Tier2DetectStats extends Omit<JudgeStats, "calls"> {
  /** The call rows THIS detect appended, not the judge's whole history. */
  readonly calls: readonly JudgeCallRecord[];
}

/**
 * One `detect`'s worth of judge activity, as deltas.
 *
 * ## Why this is its own module rather than a function inside `main.ts`
 *
 * Because `main.ts` is a Vite ENTRY POINT and nothing but a browser can run it,
 * so the only exercise this arithmetic used to get was through `tier2.spec.ts`
 * -- and that fixture cannot separate a delta from a total. Every test there
 * loads a fresh page, loads an arm (which builds a fresh judge), then does
 * exactly ONE `detect`: `before` is all-zero on every path, so `after - before`
 * and `after` agree everywhere, and `after.calls.slice(0)` and `after.calls`
 * agree too. The review that found this measured both halves of that against
 * the pre-extraction code -- replacing all fourteen counters with literals, and
 * replacing the `calls` suffix with the judge's whole history, each left the
 * entire tier-2 browser suite green -- and its report is where those numbers
 * come from, not a run of mine.
 *
 * Extracted here, `test/page-judge-delta.test.ts` drives it under vitest with
 * synthetic `JudgeStats`, where a nonzero `before` and a two-call `after` are
 * one object literal rather than a second GPU model call. The browser spec
 * keeps an end-to-end check that the page WIRES this in -- that is the half a
 * unit test cannot answer.
 *
 * ## The two rules it implements
 *
 * Every counter is listed rather than derived with a loop, for the same reason
 * `statsDelta` lists tier 1's: a counter added to `JudgeStats` upstream then
 * fails to compile here instead of silently going unreported.
 *
 * `calls` is a SUFFIX, not a difference: `WebLlmJudge` only ever appends rows,
 * so everything past the old length is this call's.
 */
export function judgeDelta(before: JudgeStats, after: JudgeStats): Tier2DetectStats {
  return {
    rung1: after.rung1 - before.rung1,
    rung2: after.rung2 - before.rung2,
    unresolvedQuotes: after.unresolvedQuotes - before.unresolvedQuotes,
    unknownPredicates: after.unknownPredicates - before.unknownPredicates,
    duplicatesDropped: after.duplicatesDropped - before.duplicatesDropped,
    repairAttempts: after.repairAttempts - before.repairAttempts,
    failedClosed: after.failedClosed - before.failedClosed,
    truncatedResponses: after.truncatedResponses - before.truncatedResponses,
    abortedResponses: after.abortedResponses - before.abortedResponses,
    segmentsJudged: after.segmentsJudged - before.segmentsJudged,
    segmentsSkipped: after.segmentsSkipped - before.segmentsSkipped,
    deadlineExpiries: after.deadlineExpiries - before.deadlineExpiries,
    callerAbortsMidGeneration: after.callerAbortsMidGeneration - before.callerAbortsMidGeneration,
    callerAbortsWhileQueued: after.callerAbortsWhileQueued - before.callerAbortsWhileQueued,
    calls: after.calls.slice(before.calls.length),
  };
}
