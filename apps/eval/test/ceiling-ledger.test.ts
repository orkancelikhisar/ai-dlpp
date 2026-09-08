/**
 * `ceiling-ledger-lib.ts` -- the join and the reconciliation.
 *
 * WHY THIS FILE EXISTS. The residual's sign is the one number in the joiner a
 * reader acts on, and until this file there was no test of it, because
 * `ceiling-ledger.ts` did its arithmetic at module top level between a
 * `readFileSync` sweep, a live `fetch` of the key endpoint and a
 * `writeFileSync` into `runs/`. Importing it ran all of that, so nothing
 * imported it. Two sign errors reached the published write-up in that state
 * (`docs/research/2026-09-07-ceiling-arm.md` Sec 6 and Sec 8.3 record the
 * correction) and the suite stayed green through both.
 *
 * The anchor figures below are read from `runs/ceiling-combined.spend.json`,
 * the artifact the write-up quotes. They are inlined rather than loaded because
 * `runs/` is gitignored and a test that skips when its input is absent is a
 * test that passes when it is absent.
 */
import { describe, expect, it } from "vitest";
import {
  LEDGER_SEGMENTS,
  buildCombinedLedger,
  formatLedgerSummary,
  joinSegment,
  type JoinedSegment,
} from "../src/driver/ceiling-ledger-lib.js";

/**
 * The four segments exactly as `runs/ceiling-combined.spend.json` holds them,
 * to full float precision.
 *
 * Full precision matters: the residual is reported to six decimals and the
 * rounded per-segment figures in the write-up's table ($0.00121, $0.13645,
 * $0.05572, $0.19269) sum to 0.38607, which cannot distinguish +0.002626 from
 * +0.002627.
 */
const REAL_SEGMENTS: JoinedSegment[] = [
  { segment: "probe", calls: 11, costUsd: 0.001213568, tripped: false },
  { segment: "ceiling-01 window 1", calls: 1457, costUsd: 0.13644738799999995, tripped: false },
  { segment: "glmon-01", calls: 225, costUsd: 0.0557229, tripped: false },
  { segment: "ceiling-01 window 2", calls: 768, costUsd: 0.19268732999999996, tripped: false },
];

/** `GET /api/v1/auth/key` -> `data.usage`, as read at the moment pass 1 ended. */
const REAL_KEY_USAGE = 0.388697546;

describe("buildCombinedLedger reproduces the published reconciliation", () => {
  const out = buildCombinedLedger({ segments: REAL_SEGMENTS, keyUsageUsd: REAL_KEY_USAGE });

  it("re-derives the committed total from the segments", () => {
    expect(out.ledgerTotalUsd).toBe(0.386071);
    expect(out.ledgerTotalCalls).toBe(2461);
  });

  it("reports the residual as key MINUS ledgers, so it is POSITIVE here", () => {
    // The published correction. An earlier draft printed -0.00749 against a key
    // of 0.37858; the artifact holds +0.002626 against 0.388697546. Reversing
    // the operands yields -0.002626 and inverts the conclusion a reader draws
    // about where the unattributed spend came from.
    expect(out.residualUsd).toBe(0.002626);
    expect(out.residualUsd!).toBeGreaterThan(0);
    expect(out.keyUsageFinalUsd).toBe(REAL_KEY_USAGE);
  });

  it("keeps the guard flag false when no segment tripped", () => {
    expect(out.guardEverTripped).toBe(false);
  });
});

describe("the residual's sign", () => {
  // Asymmetric on purpose: |key - ledger| differs in the two directions, so a
  // flipped subtraction cannot be hidden by a fixture that is symmetric about
  // the ledger total.
  const segments: JoinedSegment[] = [{ segment: "one", calls: 3, costUsd: 2 }];

  it("is POSITIVE when the key exceeds the ledgers", () => {
    expect(buildCombinedLedger({ segments, keyUsageUsd: 7.5 }).residualUsd).toBe(5.5);
  });

  it("is NEGATIVE when the ledgers exceed the key", () => {
    expect(buildCombinedLedger({ segments, keyUsageUsd: 0.25 }).residualUsd).toBe(-1.75);
  });

  it("is null, not zero, when no key could be read", () => {
    const out = buildCombinedLedger({ segments, keyUsageUsd: null });
    expect(out.residualUsd).toBeNull();
    expect(out.keyUsageFinalUsd).toBeNull();
    // A missing reading is not a reading of zero, and the flag says so.
    expect(out.keyReadIsPointInTime).toBe(false);
    expect(out.ledgerTotalUsd).toBe(2);
  });
});

describe("the two totals are different kinds of number", () => {
  /**
   * `docs/research/2026-09-07-ceiling-arm.md` Sec 6: *"`ledgerTotalUsd`
   * re-derives from four files on disk and is stable; `keyUsageFinalUsd` is
   * read live from the key endpoint, so re-running `ceiling:ledger` after any
   * later pass spends on the same key returns a larger key figure and a
   * correspondingly larger positive residual."*
   *
   * That is a property of the arithmetic, so it is asserted here rather than
   * left in prose. It also kills the sign flip from the other side: under a
   * reversed subtraction a LARGER key reading makes the residual SMALLER.
   */
  it("a later key reading moves the residual by exactly its own delta, and never the ledger total", () => {
    const first = buildCombinedLedger({ segments: REAL_SEGMENTS, keyUsageUsd: REAL_KEY_USAGE });
    const laterSpend = 0.25;
    const second = buildCombinedLedger({
      segments: REAL_SEGMENTS,
      keyUsageUsd: REAL_KEY_USAGE + laterSpend,
    });

    expect(second.ledgerTotalUsd).toBe(first.ledgerTotalUsd);
    expect(second.ledgerTotalCalls).toBe(first.ledgerTotalCalls);
    expect(second.residualUsd! - first.residualUsd!).toBeCloseTo(laterSpend, 9);
    expect(second.residualUsd!).toBeGreaterThan(first.residualUsd!);
    expect(first.keyReadIsPointInTime).toBe(true);
  });
});

describe("guardEverTripped is read off the segments", () => {
  const untripped = (name: string): JoinedSegment => ({ segment: name, calls: 1, costUsd: 1, tripped: false });

  it("is true when ANY ONE of several segments tripped", () => {
    // Three untripped segments and one tripped, so a hardcoded `false` survives
    // the all-false case above and dies here. The tripped segment is not first,
    // so a mutant reading only `segments[0]` dies too.
    const segments = [untripped("a"), untripped("b"), { segment: "c", calls: 1, costUsd: 1, tripped: true }, untripped("d")];
    expect(buildCombinedLedger({ segments, keyUsageUsd: null }).guardEverTripped).toBe(true);
  });

  it("does not treat a truthy non-boolean as a trip", () => {
    // A ledger written by an older driver could carry a string here. `tripped`
    // is a claim that a stop path FIRED; anything that is not the boolean true
    // is not that claim, and reading it loosely would report a budget stop that
    // never happened.
    const segments = [{ segment: "a", calls: 1, costUsd: 1, tripped: "true" }];
    expect(buildCombinedLedger({ segments, keyUsageUsd: null }).guardEverTripped).toBe(false);
  });

  it("is false, not undefined, when no segment carries the field at all", () => {
    expect(buildCombinedLedger({ segments: [{ segment: "a", calls: 1, costUsd: 1 }], keyUsageUsd: null }).guardEverTripped).toBe(
      false,
    );
  });
});

describe("segments are joined once each", () => {
  it("names four distinct files and four distinct segment names", () => {
    // A duplicated entry double-counts its cost and its calls into the total
    // with no visible error. Uniqueness of both keys is the cheapest guard.
    // Uniqueness is derived from the list's own length so a new segment does not
    // require editing this test -- but the FILE LIST is spelled out, so silently
    // dropping a segment (which would understate the total with no error) fails.
    expect(new Set(LEDGER_SEGMENTS.map((s) => s.file)).size).toBe(LEDGER_SEGMENTS.length);
    expect(new Set(LEDGER_SEGMENTS.map((s) => s.name)).size).toBe(LEDGER_SEGMENTS.length);
    expect(LEDGER_SEGMENTS.map((s) => s.file)).toEqual([
      "ceiling-probe.spend.json",
      "ceiling-ceiling-01.part1.spend.json",
      "ceiling-glmon-01.spend.json",
      "ceiling-ceiling-01.spend.json",
      "ceiling-ceiling-02.spend.json",
      "orphaned/thinkon-fullslate-aborted.spend.json",
      "ceiling-thinkonglm-01.spend.json",
    ]);
  });

  it("sums each segment exactly once", () => {
    const one: JoinedSegment = { segment: "x", calls: 100, costUsd: 1.5 };
    expect(buildCombinedLedger({ segments: [one], keyUsageUsd: null }).ledgerTotalCalls).toBe(100);
    expect(buildCombinedLedger({ segments: [one, one], keyUsageUsd: null }).ledgerTotalCalls).toBe(200);
    expect(buildCombinedLedger({ segments: REAL_SEGMENTS, keyUsageUsd: null }).ledgerTotalCalls).toBe(2461);
  });

  it("treats a segment missing costUsd or calls as zero rather than NaN", () => {
    const out = buildCombinedLedger({ segments: [{ segment: "x" }, { segment: "y", calls: 5, costUsd: 2 }], keyUsageUsd: null });
    expect(out.ledgerTotalUsd).toBe(2);
    expect(out.ledgerTotalCalls).toBe(5);
  });

  it("joinSegment attaches the file path and note without losing the ledger's own fields", () => {
    const spec = LEDGER_SEGMENTS[1]!;
    const joined = joinSegment(spec, { calls: 1457, costUsd: 0.13644738799999995, tripped: false });
    expect(joined["segment"]).toBe(spec.name);
    expect(joined["ledger"]).toBe(`runs/${spec.file}`);
    expect(joined["calls"]).toBe(1457);
    expect(joined["tripped"]).toBe(false);
  });
});

describe("the summary a human reads carries the same numbers", () => {
  it("prints the total, the key and the SIGNED residual", () => {
    const lines = formatLedgerSummary(buildCombinedLedger({ segments: REAL_SEGMENTS, keyUsageUsd: REAL_KEY_USAGE }));
    expect(lines[0]).toBe(
      "ledger total $0.38607 over 2461 calls; key $0.38870; residual $0.00263; guard tripped: false",
    );
    expect(lines).toHaveLength(5);
    expect(lines[2]).toContain("ceiling-01 window 1");
    expect(lines[2]).toContain("$0.13645");
  });

  it("shows a negative residual with its minus sign rather than as an absolute value", () => {
    const lines = formatLedgerSummary(
      buildCombinedLedger({ segments: [{ segment: "one", calls: 1, costUsd: 5 }], keyUsageUsd: 1 }),
    );
    expect(lines[0]).toContain("residual $-4.00000");
  });

  it("says the key was unread rather than printing $0.00000", () => {
    const lines = formatLedgerSummary(
      buildCombinedLedger({ segments: [{ segment: "one", calls: 1, costUsd: 5 }], keyUsageUsd: null }),
    );
    expect(lines[0]).toContain("key $unread");
    expect(lines[0]).toContain("residual unknown");
  });
});
