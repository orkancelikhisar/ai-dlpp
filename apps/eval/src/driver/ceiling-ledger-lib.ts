/**
 * The join-and-reconcile arithmetic behind `pnpm -C apps/eval ceiling:ledger`,
 * separated from its I/O so it can be tested without a key or a network.
 *
 * `ceiling-ledger.ts` used to hold this at module top level, next to a
 * `readFileSync` sweep, a `fetch` of the key endpoint and a `writeFileSync`.
 * Importing that module ran all of it, so no test could import it, so NOTHING
 * here was covered -- including the sign of the residual, which is the one
 * quantity in the file a reader acts on. Two sign errors reached the published
 * write-up (`docs/research/2026-09-07-ceiling-arm.md` Sec 6 records the
 * correction: a residual of -$0.00749 against a key reading of $0.37858, where
 * the artifact holds +$0.002626 against $0.388698) and a green suite could not
 * have caught either. The split exists for that reason and no other.
 *
 * THE TWO TOTALS ARE NOT THE SAME KIND OF NUMBER, and `CombinedLedger` says so
 * on the object rather than in prose:
 *
 *   - `ledgerTotalUsd` is RE-DERIVED from the segment files on disk. Given the
 *     same files it is the same number, today or next year.
 *   - `keyUsageFinalUsd` is a POINT-IN-TIME read of `GET /auth/key`. Any later
 *     spend on the same key makes it larger, and therefore makes `residualUsd`
 *     larger by exactly the same amount.
 *
 * That second property is what `keyReadIsPointInTime` marks and what the tests
 * exercise: re-reading the key must move the residual by the key's delta and
 * must not move the ledger total at all.
 */

/** One spend segment: which file it comes from, and why it is a separate file. */
export interface LedgerSegmentSpec {
  readonly file: string;
  readonly name: string;
  readonly note: string;
}

export const LEDGER_SEGMENTS: readonly LedgerSegmentSpec[] = [
  {
    file: "ceiling-probe.spend.json",
    name: "probe",
    note: "The 6-model x 2-family probe. Its process was killed deliberately while diagnosing the Alibaba streaming hang; the probe was re-run inside each later launch.",
  },
  {
    file: "ceiling-ceiling-01.part1.spend.json",
    name: "ceiling-01 window 1",
    note: "Arms 1-7: deepseek judge+b, qwen-flash judge+b, mistral judge+b, nemotron judge. The process stopped after the nemotron judge arm with no [stop] line, no error and no done line; the spend guard did not trip and the cause is unknown.",
  },
  {
    file: "ceiling-glmon-01.spend.json",
    name: "glmon-01",
    note: "GLM-5.3-flash with thinking ON, judge arm only; its B arm was skipped after a 0/3 probe. Ran concurrently with window 1, against a different provider.",
  },
  {
    file: "ceiling-ceiling-01.spend.json",
    name: "ceiling-01 window 2",
    note: "Arms 8-10 plus a re-run of the nemotron judge arm, under the same runId, pins and request body as window 1.",
  },
];

/** A segment as it appears in the joined file: the ledger's own fields plus the three added here. */
export type JoinedSegment = Record<string, unknown>;

export interface CombinedLedger {
  readonly what: string;
  readonly regenerateWith: string;
  readonly segments: readonly JoinedSegment[];
  /** Re-derived from the segment files. Stable across re-runs given the same files. */
  readonly ledgerTotalUsd: number;
  readonly ledgerTotalCalls: number;
  /** A live read of `GET /auth/key`, or null when no key was supplied. */
  readonly keyUsageFinalUsd: number | null;
  /** `keyUsageFinalUsd - ledgerTotalUsd`. POSITIVE means the key exceeds the ledgers. */
  readonly residualUsd: number | null;
  readonly residualNote: string;
  /**
   * True whenever a key reading is present, marking `keyUsageFinalUsd` and
   * `residualUsd` as valid only for the instant the key was read. See the
   * module docblock.
   */
  readonly keyReadIsPointInTime: boolean;
  readonly keyLimitUsd: number;
  readonly hardStopUsd: number;
  readonly guardEverTripped: boolean;
  readonly guardNote: string;
}

export interface CombinedLedgerInput {
  readonly segments: readonly JoinedSegment[];
  /** null when `OPENROUTER_API_KEY` was absent or the endpoint did not answer. */
  readonly keyUsageUsd: number | null;
  readonly keyLimitUsd?: number;
  readonly hardStopUsd?: number;
}

/** Attaches the three joiner fields to one segment's own ledger contents. */
export function joinSegment(spec: LedgerSegmentSpec, ledger: Record<string, unknown>): JoinedSegment {
  return { segment: spec.name, ledger: `runs/${spec.file}`, note: spec.note, ...ledger };
}

/**
 * The whole reconciliation, from segments and one key reading.
 *
 * Everything the joined file reports is decided here, so reverting any of it --
 * the residual's sign, the guard's derivation from the segments -- changes this
 * function's return value and nothing else has to be re-wired to notice.
 */
export function buildCombinedLedger(input: CombinedLedgerInput): CombinedLedger {
  const { segments, keyUsageUsd } = input;
  const ledgerTotalUsd = segments.reduce((n, s) => n + Number(s["costUsd"] ?? 0), 0);
  const ledgerTotalCalls = segments.reduce((n, s) => n + Number(s["calls"] ?? 0), 0);
  // key MINUS ledgers. The sign is the whole content of this line: positive
  // means spend the driver never saw, negative means the key's accounting
  // trailing the per-response figures. Reversing the operands inverts the
  // conclusion a reader draws, which is what happened in the write-up.
  const residualUsd = keyUsageUsd === null ? null : Number((keyUsageUsd - ledgerTotalUsd).toFixed(6));

  return {
    what: "Every spend segment of the capability-ceiling arm, joined and reconciled against the key endpoint.",
    regenerateWith: "OPENROUTER_API_KEY=... pnpm -C apps/eval ceiling:ledger",
    segments,
    ledgerTotalUsd: Number(ledgerTotalUsd.toFixed(6)),
    ledgerTotalCalls,
    keyUsageFinalUsd: keyUsageUsd,
    residualUsd,
    residualNote:
      "key usage minus the ledger total. POSITIVE means spend the driver never saw (diagnostic curls made " +
      "outside it). NEGATIVE means the key endpoint's accounting trailing the per-response usage.cost " +
      "figures, or a segment still in flight when the key was read. The sign alone does not settle which.",
    keyReadIsPointInTime: keyUsageUsd !== null,
    keyLimitUsd: input.keyLimitUsd ?? 10,
    hardStopUsd: input.hardStopUsd ?? 7,
    // Read off the segments, never asserted. A segment that tripped its guard
    // records `tripped: true`; a run summary that says otherwise is a record of
    // intent rather than of fact.
    guardEverTripped: segments.some((s) => s["tripped"] === true),
    guardNote:
      "No stop path fired in any segment. This experiment was bounded by wall-clock time, not by budget.",
  };
}

/** The lines `ceiling:ledger` prints, in order. Separated so the numbers can be asserted without a console. */
export function formatLedgerSummary(out: CombinedLedger): string[] {
  const key = out.keyUsageFinalUsd;
  const residual = out.residualUsd;
  return [
    `ledger total $${out.ledgerTotalUsd.toFixed(5)} over ${out.ledgerTotalCalls} calls; ` +
      `key $${key === null ? "unread" : key.toFixed(5)}; residual ` +
      `${residual === null ? "unknown" : `$${residual.toFixed(5)}`}; guard tripped: ${out.guardEverTripped}`,
    ...out.segments.map(
      (s) => `  ${String(s["segment"]).padEnd(22)} calls ${String(s["calls"]).padStart(5)}  $${Number(s["costUsd"]).toFixed(5)}`,
    ),
  ];
}
