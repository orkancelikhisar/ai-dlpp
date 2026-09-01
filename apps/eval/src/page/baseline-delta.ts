import type { BaselineStats, JudgeCallRecord } from "@sih/tier2";

/** What one Approach-B `detect` cost, as deltas over the arm's running counters. */
export interface BaselineDetectStats extends Omit<BaselineStats, "calls"> {
  /** The call rows THIS detect appended, not the arm's whole history. */
  readonly calls: readonly JudgeCallRecord[];
}

/**
 * One Approach-B `detect`'s worth of activity, as deltas.
 *
 * The twin of `judgeDelta`, in its own module for the same reason and with the
 * same two rules. Read that file's docblock first: it carries the measurement
 * that says why a delta is not optional (`WebLlmJudge.stats` is cumulative, and
 * `BaselineB.stats` accumulates identically -- `baselineB.ts` MEASURED `rung1`
 * after each of four identical items as 1, 2, 3, 4), and why the arithmetic
 * lives outside `main.ts` (a Vite entry point only a browser can run, whose
 * one-detect-per-page specs cannot separate a delta from a total).
 *
 * ## Why this is a SECOND function and not a shared generic one
 *
 * Because the two counter sets are not the same set, and a generic subtraction
 * over `Object.keys` would hide exactly that. `BaselineStats` renames two events
 * -- `messagesJudged` for `segmentsJudged`, `unknownEntityTypes` for
 * `unknownPredicates` -- because B judges a MESSAGE where the judge judges a
 * segment and names an entity class where the judge names a predicate. It adds
 * `messageBudgetExpiries`, which no judge counter reports: `detect` arms the
 * message deadline for the compiled path and B is its own orchestrator, so B is
 * the only thing that can count it. And it has no `segmentsSkipped`, because an
 * arm that makes one call per message has no second unit to skip.
 *
 * Two of those are renames of the same event and one is a genuinely new one, so
 * a bake-off summing them beside the judge's has to do the translation
 * knowingly. Listing every counter here rather than deriving them with a loop is
 * what makes a counter added to `BaselineStats` upstream a compile error instead
 * of a silently unreported number.
 *
 * `calls` is a SUFFIX, not a difference: `createArm` in `baselineB.ts` only ever
 * appends rows, so everything past the old length is this call's.
 */
export function baselineDelta(before: BaselineStats, after: BaselineStats): BaselineDetectStats {
  return {
    rung1: after.rung1 - before.rung1,
    rung2: after.rung2 - before.rung2,
    unresolvedQuotes: after.unresolvedQuotes - before.unresolvedQuotes,
    unknownEntityTypes: after.unknownEntityTypes - before.unknownEntityTypes,
    duplicatesDropped: after.duplicatesDropped - before.duplicatesDropped,
    repairAttempts: after.repairAttempts - before.repairAttempts,
    failedClosed: after.failedClosed - before.failedClosed,
    truncatedResponses: after.truncatedResponses - before.truncatedResponses,
    abortedResponses: after.abortedResponses - before.abortedResponses,
    messagesJudged: after.messagesJudged - before.messagesJudged,
    deadlineExpiries: after.deadlineExpiries - before.deadlineExpiries,
    messageBudgetExpiries: after.messageBudgetExpiries - before.messageBudgetExpiries,
    callerAbortsMidGeneration: after.callerAbortsMidGeneration - before.callerAbortsMidGeneration,
    callerAbortsWhileQueued: after.callerAbortsWhileQueued - before.callerAbortsWhileQueued,
    calls: after.calls.slice(before.calls.length),
  };
}
