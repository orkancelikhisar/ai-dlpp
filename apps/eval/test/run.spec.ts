import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { UNCERTAIN_BELOW } from "@sih/core";
import { loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema, toJsonl, type RunRecord } from "../src/driver/record.js";
import { runArm } from "../src/driver/run.js";

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const FIXTURES = join(REPO_ROOT, "corpora", "fixtures");
const IR_FIXTURE = join(REPO_ROOT, "apps", "eval", "fixtures", "minimal-ir.json");

/** Any well-formed digest; the stand-in page below only needs to clear the guard. */
const FAKE_IR_HASH = "a".repeat(64);

/**
 * Which entityTypes tier 0 can possibly produce, read from the IR FIXTURE rather
 * than written out here, so adding a tier-0 entity to that file extends this
 * check instead of quietly leaving it behind.
 */
function tier0EntityIds(): Set<string> {
  const ir = JSON.parse(readFileSync(IR_FIXTURE, "utf8")) as {
    entityTypes: { id: string; tier: number }[];
  };
  return new Set(ir.entityTypes.filter((e) => e.tier === 0).map((e) => e.id));
}

/**
 * Wraps a real Page so `evaluate` still runs for real, and records both the
 * high-water mark of concurrent calls and how many were made. An observer, not
 * a stub: every call reaches the same page object with the same arguments and
 * `this` bound to the real page, so what gets measured is unchanged and only the
 * call PATTERN is visible.
 */
function watchEvaluate(page: Page): { page: Page; maxInFlight: () => number; calls: () => number } {
  let live = 0;
  let max = 0;
  let total = 0;
  const watched = new Proxy(page, {
    get(target, prop) {
      // `target` as the receiver rather than the proxy, so a getter on Page
      // resolves its own property reads against the real object instead of
      // recursing back through this trap. Methods are bound to it for the same
      // reason.
      const value = Reflect.get(target, prop, target) as unknown;
      if (typeof value !== "function") return value;
      if (prop !== "evaluate") return value.bind(target);
      return async (...args: unknown[]) => {
        live += 1;
        total += 1;
        max = Math.max(max, live);
        try {
          return await (value as (...a: unknown[]) => Promise<unknown>).apply(target, args);
        } finally {
          live -= 1;
        }
      };
    },
  });
  return { page: watched, maxInFlight: () => max, calls: () => total };
}

/**
 * What the stand-in page does when `detect` is called with a given item's text.
 *
 * `failed-closed` answers exactly as `ok` does but adds a tier-2 notice to the
 * canned `degraded`. It is a VERDICT rather than a second canned result because
 * the property under test is per ITEM -- an arm that failed closed on some of
 * its messages -- and one canned object cannot vary by item.
 */
type FakeVerdict = "ok" | "throw" | "hang" | "kill" | "failed-closed";

/**
 * Installs a stand-in `window.__sih` on `about:blank`.
 *
 * Several specs below need a page that fails in a way the real harness page
 * cannot be made to fail on demand: a `detect` that never settles, a browser
 * that dies mid-arm, a corpus where only the middle item throws. None of them
 * measure detection -- they exercise driver behaviour AROUND detection, and the
 * real page is used everywhere that behaviour of core itself is the subject.
 *
 * Only possible on `about:blank`: the harness page defines `__sih` non-writable
 * and non-configurable, so nothing can stub the real one out. This fake is
 * deliberately `configurable: true` so that the "kill" verdict can delete it and
 * simulate a page that navigated away.
 *
 * `plan` is data rather than a function because it has to cross into the browser
 * as JSON. Any text not named in it is treated as "ok".
 *
 * `tier1` is what `tier1Status().lastDetect` answers. It has a default rather
 * than being optional-and-absent because `RunRecordSchema` requires
 * `tier1Stats` on any record whose `config.tier1` is set and whose `error` is
 * null: a fake that answered nothing would make every tier-1 arm here produce
 * records the schema refuses, for a reason that has nothing to do with the
 * behaviour under test. `inferences: 1` is the honest reading of a fake that
 * did answer a detect call.
 */
const FAKE_TIER1_STATS = {
  inferences: 1,
  droppedWords: 0,
  truncatedWords: 0,
  overWideSpans: 0,
  unmappableSpans: 0,
  nonFiniteScores: 0,
  gpuSubmits: 0,
};

/**
 * `Tier2DetectStats` with every counter at zero and no calls -- the honest
 * reading of a judge that was in the loop and did nothing, which is what a
 * stand-in page is.
 *
 * A default for the same reason `FAKE_TIER1_STATS` is one: `RunRecordSchema`
 * requires `tier2Stats` on any record whose `config.tier2` is set and whose
 * `error` is null, so a fake answering nothing would make every tier-2 arm here
 * produce records the schema refuses for a reason unrelated to what is under
 * test. Tests that care about the values pass their own, and they pass
 * non-zeros -- a fixture of zeros cannot tell a producer that copies from one
 * that hardcodes.
 */
const FAKE_TIER2_STATS = {
  rung1: 0, rung2: 0, unresolvedQuotes: 0, unknownPredicates: 0, duplicatesDropped: 0,
  repairAttempts: 0, failedClosed: 0, truncatedResponses: 0, abortedResponses: 0,
  segmentsJudged: 0, segmentsSkipped: 0, deadlineExpiries: 0,
  callerAbortsMidGeneration: 0, callerAbortsWhileQueued: 0,
  calls: [] as unknown[],
};

/**
 * The resolved tier-2 settings a record must carry when tier 2 ran.
 *
 * `RunRecordSchema` requires it exactly when `config.tier2` is set -- the same
 * coupling `tier1Config` has -- so a tier-2 arm here without one would produce
 * records the schema refuses for a reason unrelated to what is under test.
 * These are `DEFAULT_TIER2_CONFIG`'s values plus the page's default per-call
 * budget, which is what an arm run with no overrides is loaded under.
 */
const FAKE_TIER2_CONFIG = {
  modelId: "Qwen3.5-2B-q4f16_1-MLC",
  contextWindowSize: 8192,
  temperature: 0,
  maxTokens: 512,
  callBudgetMs: 60_000,
} as const;

/**
 * The tier-2 notice a `failed-closed` verdict adds, worded as
 * `WebLlmJudge` words its own: a body still unparseable after the one repair
 * retry. The stand-in does NOT model core's `absent` entries -- it is not core,
 * and inventing them here would be a fake asserting which tiers ran. The
 * real-page spec below is what pins those.
 */
const FAKE_FAILED_CLOSED = {
  tier: 2,
  reason: "failed-closed",
  detail: "the tier-2 response was still unparseable after one repair retry",
};

async function installFakeSih(
  page: Page,
  plan: Record<string, FakeVerdict>,
  result: {
    findings: unknown[];
    timings: { tier0Ms: number; tier1Ms?: number; tier2Ms?: number };
    /**
     * Required, matching `DetectionResult` -- where it is required too, and its
     * doc explains why: "[] is a positive claim; undefined would be silence". A
     * fake allowed to omit it would let `runArm` be written against a shape the
     * real page cannot produce.
     */
    degraded: unknown[];
  },
  tier1: Record<string, number> = FAKE_TIER1_STATS,
  tier2: Record<string, unknown> | undefined = FAKE_TIER2_STATS,
): Promise<void> {
  await page.goto("about:blank");
  await page.evaluate(
    ({
      plan: verdicts,
      result: canned,
      hash,
      tier1: stats,
      tier2: judged,
      FAILED_CLOSED_NOTICE,
    }) => {
      // How many `detect` calls this fake has answered, so `tier2Status` can
      // offer a CUMULATIVE `totals` beside the per-item `lastDetect` -- see
      // there. Nothing else reads it.
      let detects = 0;
      Object.defineProperty(window, "__sih", {
        configurable: true,
        value: {
          irHash: () => Promise.resolve(hash),
          policyHash: () => "test-hash",
          detect: ({ text, config }: { text: string; config: unknown }) => {
            detects += 1;
            // Remembered so a spec can assert what `detect` was HANDED, which no
            // assertion on the returned record can establish: runArm builds one
            // config object and uses it for both, so a record agreeing with
            // itself proves nothing about the call.
            const seen = window as unknown as { __configs?: unknown[] };
            seen.__configs = [...(seen.__configs ?? []), config];
            const verdict = Object.hasOwn(verdicts, text) ? verdicts[text] : "ok";
            if (verdict === "hang") return new Promise(() => {});
            if (verdict === "kill") {
              delete window.__sih;
              return Promise.reject(new Error("the page went away"));
            }
            if (verdict === "throw") return Promise.reject(new Error("detector exploded"));
            if (verdict === "failed-closed") {
              return Promise.resolve({
                ...canned,
                degraded: [...canned.degraded, FAILED_CLOSED_NOTICE],
              });
            }
            return Promise.resolve(canned);
          },
          // Only `lastDetect` is populated: it is the only member runArm reads,
          // and inventing a `load` report would be a fake asserting which rung
          // and which provider ran, which is precisely what this stand-in has
          // no standing to claim.
          tier1Status: () => ({ lastDetect: stats }),
          // `undefined` when the fake was given no tier-2 stats, which is what
          // an unloaded page answers -- so a tier-2 arm run against that fake
          // produces a record the schema refuses, rather than one carrying
          // counters nobody measured.
          //
          // `totals` is deliberately NOT equal to `lastDetect`, and that is the
          // point of it being here at all. `WebLlmJudge.stats` is CUMULATIVE
          // across every message a judge has seen -- its own docblock says so
          // -- so a driver that reads `totals` instead of the per-item delta
          // gets the arm's running sum and inflates every row after the first,
          // with a green suite. This fake models that: each counter multiplied
          // by the number of detects so far, and the call rows repeated. It
          // only bites at TWO OR MORE items, which is why the specs that read
          // these stats run at least two.
          tier2Status: () =>
            judged === undefined
              ? undefined
              : {
                  totals: Object.fromEntries(
                    Object.entries(judged).map(([key, value]) => [
                      key,
                      typeof value === "number"
                        ? value * detects
                        : Array.isArray(value)
                          ? Array.from({ length: detects }, () => value).flat()
                          : value,
                    ]),
                  ),
                  lastDetect: judged,
                },
        },
      });
    },
    { plan, result, hash: FAKE_IR_HASH, tier1, tier2, FAILED_CLOSED_NOTICE: FAKE_FAILED_CLOSED },
  );
}

test("runs the smoke corpus and emits one valid record per item", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));
  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items,
  });

  expect(records).toHaveLength(items.length);
  for (const record of records) {
    // The error text matters more here than anywhere else in the file: this is
    // the assertion a 1,500-item run is most likely to trip, and a bare
    // "expected false to be true" names neither the offending record nor the
    // field. z.prettifyError plus the itemId, mirroring corpus.ts.
    const parsed = RunRecordSchema.safeParse(record);
    expect(
      parsed.success,
      parsed.success ? "" : `record ${record.itemId} is invalid: ${z.prettifyError(parsed.error)}`,
    ).toBe(true);
  }
  // Record order matches corpus order, so a diff between two runs lines up.
  expect(records.map((r) => r.itemId)).toEqual(items.map((i) => i.id));

  // The pass-through fields, every value distinct so a field written into the
  // wrong slot shows up. Nothing else here reads them and RunRecordSchema only
  // asks that they be non-empty strings, so without this a runArm that stamped
  // the arm name where the provider belongs would emit a perfectly valid file
  // that mislabels every row of the comparison it exists to feed. `config` is
  // included because `arm` alone is a name: without the configuration beside
  // it, an all-tiers-off run is indistinguishable from a detector that found
  // nothing.
  expect(records[0]).toMatchObject({
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    // The corpus's own value, carried through untouched. It says
    // "minimal-fixture" rather than "p-fin" because its gold was written
    // against apps/eval/fixtures/minimal-ir.json and not against
    // policies/p-fin.md -- see the `policy` field's note in driver/corpus.ts.
    policy: "minimal-fixture",
    config: { tier0: true, tier1: false, tier2: false },
  });

  // Everything above is satisfied by a runArm that never calls detect at all:
  // RunRecordSchema accepts an empty `findings`, and 7 of these 13 items are
  // negatives that legitimately have none. So pin the one thing only a real
  // detection run produces -- every tier-0 GOLD span is covered by a finding
  // that agrees with it on ENTITY TYPE as well as position. Position alone
  // would let a finding with the wrong label count as a hit, and the label is
  // what Plan 8 scores by.
  //
  // Overlap, not equality, and the corpus is why: in `pos-secret-key-value` the
  // gold span is the secret value alone while tier 0 reports the whole
  // `SESSION_TOKEN=<secret>` run (the entropy alphabet contains `=` and `_`, so
  // the kv line is one candidate run). Requiring equality would fail on a
  // detector that is behaving as documented.
  //
  // Derived from the corpus's own labels, never from what this run returned, so
  // it stays a claim about detection rather than a transcript of it. Tier-1
  // entities are skipped because no tier-1 engine is registered in this arm.
  const tier0 = tier0EntityIds();
  const expectedCovered = items.reduce(
    (n, i) => n + i.gold.filter((g) => tier0.has(g.entityType)).length,
    0,
  );
  let covered = 0;
  for (const record of records) {
    for (const gold of record.gold) {
      if (!tier0.has(gold.entityType)) continue;
      const hit = record.findings.some(
        (f) => f.entityType === gold.entityType && f.start < gold.end && gold.start < f.end,
      );
      expect(
        hit,
        `${record.itemId}: no ${gold.entityType} finding overlaps gold [${gold.start},${gold.end})`,
      ).toBe(true);
      covered += 1;
    }
  }
  // The loop above passes vacuously if `gold` were dropped from the records, so
  // count what the CORPUS says should have been checked and require both that
  // the two agree and that there was anything to check. Derived rather than
  // hardcoded: a later task adding a tier-0 positive must not turn this red.
  expect(expectedCovered).toBeGreaterThan(0);
  expect(covered).toBe(expectedCovered);
});

test("runs items one at a time, never overlapping them", async ({ page }) => {
  // `runArm` documents sequential execution as load-bearing -- latency is a
  // reported metric and concurrent inference on one GPU measures contention
  // rather than the model. The order assertion in the first spec does NOT stand
  // in for this one: MEASURED, running the item loop under
  // `Promise.all(spec.items.map(...))` leaves that spec green, because the
  // records still come out in corpus order. Under the same mutation this spec
  // reads a maximum of 13 in flight -- one per item.
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));
  const watched = watchEvaluate(page);

  const records = await runArm(watched.page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items,
  });

  expect(records).toHaveLength(items.length);
  expect(watched.maxInFlight()).toBe(1);
  // One round trip per item, plus the readiness probe and the one that reads
  // both hashes. Pinned because the comment in run.ts claims this shape: an
  // extra per-item evaluate is latency charged to every measurement, and a
  // second hash read would make that comment wrong.
  expect(watched.calls()).toBe(items.length + 2);
});

test("records a thrown item instead of dropping it", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    // tier1 with no engine registered throws inside detect().
    config: { tier0: true, tier1: true, tier2: false },
    itemTimeoutMs: 10_000,
    items: [{ id: "boom", text: "anything at all", policy: "p-fin", gold: [] }],
  });

  expect(records).toHaveLength(1);
  expect(records[0]!.error).toMatch(/tier1/i);
  expect(records[0]!.findings).toEqual([]);
});

test("keeps going after a thrown item without leaking its error onto the next", async ({ page }) => {
  // The "try lives INSIDE the loop" guarantee, which no homogeneous corpus can
  // check: every other spec here runs items that all succeed or a single one
  // that throws, and both pass whether the try wraps the loop or the item.
  // Three items, only the middle one failing, is the smallest corpus that can
  // tell the difference -- and it also catches the opposite bug, an `error`
  // that is assigned once and never cleared, which would make every item after
  // the first failure look like it failed too.
  await installFakeSih(page, { "middle text": "throw" }, { findings: [], timings: { tier0Ms: 1.5 }, degraded: [] });

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items: [
      { id: "ok-before", text: "first text", policy: "p-fin", gold: [] },
      { id: "boom", text: "middle text", policy: "p-fin", gold: [] },
      { id: "ok-after", text: "last text", policy: "p-fin", gold: [] },
    ],
  });

  expect(records.map((r) => r.itemId)).toEqual(["ok-before", "boom", "ok-after"]);
  expect(records.map((r) => r.error === null)).toEqual([true, false, true]);
  expect(records[1]!.error).toMatch(/detector exploded/);

  // Latency is a headline metric and it can be zeroed with a fully green suite:
  // nothing on the real page can assert `tier0Ms > 0`, because two negatives in
  // the smoke corpus legitimately read exactly 0. A stand-in that returns a
  // known non-zero value is the only way to prove the number survives the trip
  // from `detect` to the record at all.
  expect(records[0]!.timings).toEqual({ tier0Ms: 1.5 });
  expect(records[2]!.timings).toEqual({ tier0Ms: 1.5 });
  // And the failed item carries no borrowed timing from its neighbours.
  expect(records[1]!.timings).toEqual({ tier0Ms: 0 });
});

test("carries timings verbatim and emits only the fields the record declares", async ({ page }) => {
  // Two properties of the copy out of `detect`'s result, both invisible to the
  // real page. Timings: a tier-1 number has to survive as well as tier 0, and
  // no arm here runs tier 1. Fields: findings cross the evaluate boundary as
  // plain JSON and nothing validates a record on the producing path, so a
  // producer that decorates its findings rides the extra key straight into the
  // JSONL unless runArm projects the fields it declares.
  await installFakeSih(
    page,
    {},
    {
      findings: [
        {
          start: 0,
          end: 5,
          text: "hello",
          entityType: "in-pan",
          severity: "high",
          tier: 0,
          source: "pan-rule",
          confidence: 0.9,
          action: "block",
          // Not in RecordFindingSchema. A real tier-1 engine carrying debug
          // state on its findings is the realistic shape of this.
          debugNote: "should not reach the file",
        },
      ],
      timings: { tier0Ms: 12.5, tier1Ms: 3.25 },
      degraded: [],
    },
  );

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0+t1",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: true, tier2: false, t1Model: "fake" },
    itemTimeoutMs: 10_000,
    items: [{ id: "a", text: "hello world", policy: "p-fin", gold: [] }],
  });

  expect(records[0]!.timings).toEqual({ tier0Ms: 12.5, tier1Ms: 3.25 });
  expect(Object.keys(records[0]!.findings[0]!).sort()).toEqual([
    "action",
    "confidence",
    "end",
    "entityType",
    "severity",
    "source",
    "start",
    "text",
    "tier",
  ]);
  // The config that ran, recorded rather than implied by the arm name. It
  // carries `backend` even though this arm's TierConfig did not: runArm
  // reconciles `spec.backend` into the config it forwards, so the label and the
  // config cannot disagree.
  expect(records[0]!.config).toEqual({
    tier0: true,
    tier1: true,
    tier2: false,
    t1Model: "fake",
    backend: "wasm",
  });
  // `detector` is on the RECORD and not in the config: it says which
  // implementation the page ran, which is not something core's `TierConfig`
  // can express -- an Approach-B row reports `tier2: true` too.
  expect(records[0]!.detector).toBe("core-orchestrator");
});

test("times an item out instead of wedging the whole arm on it", async ({ page }) => {
  // A `detect` that never settles would otherwise hang runArm forever, and
  // because records are returned only in bulk at the end, every already
  // completed record would be unreachable. The budget is per item and comes
  // from the caller.
  await installFakeSih(page, { "wedges forever": "hang" }, { findings: [], timings: { tier0Ms: 2 }, degraded: [] });

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 250,
    items: [
      { id: "hangs", text: "wedges forever", policy: "p-fin", gold: [] },
      { id: "after", text: "fine", policy: "p-fin", gold: [] },
    ],
  });

  expect(records.map((r) => r.itemId)).toEqual(["hangs", "after"]);
  expect(records[0]!.error).toMatch(/did not finish within 250ms/);
  expect(records[0]!.findings).toEqual([]);
  // The arm survives the wedged item; the page is still alive, so the loop goes
  // on rather than aborting.
  expect(records[1]!.error).toBeNull();
});

test("aborts the arm when the browser dies, rather than filing the rest as errors", async ({
  page,
}) => {
  // The worst output this harness can produce: confident, complete and wrong.
  // A crashed browser, a closed context or a page that navigated away makes
  // EVERY remaining item throw, and the per-item catch turns each into a
  // schema-valid record with `error` set. Without a liveness probe the arm
  // returns a full-length JSONL file that reads as "this arm scored badly"
  // instead of "this run died", and Plan 8 would score it. Here the third item
  // destroys the page API before failing, which is what a navigated-away page
  // looks like from the driver.
  await installFakeSih(page, { "kills the page": "kill" }, { findings: [], timings: { tier0Ms: 1 }, degraded: [] });

  const items = [
    { id: "ok-1", text: "one", policy: "p-fin", gold: [] },
    { id: "ok-2", text: "two", policy: "p-fin", gold: [] },
    { id: "fatal", text: "kills the page", policy: "p-fin", gold: [] },
    { id: "never-run", text: "four", policy: "p-fin", gold: [] },
  ];
  const armRun = runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items,
  });

  // Which item, and how much was already collected -- otherwise the operator
  // knows only that something died somewhere in a 1,500-item corpus.
  await expect(armRun).rejects.toThrow(/harness died while running item 3 of 4 \("fatal"\)/);
  await expect(armRun).rejects.toThrow(/2 record\(s\) collected/);
});

test("refuses an unprepared page rather than silently measuring a blank one", async ({ page }) => {
  // If runArm navigated on its own, Task 12 would load a tier-1 model and then
  // have it thrown away — producing a full JSONL file of tier-0 results
  // labelled as a tier-1 arm. Nothing downstream could detect that.
  await expect(
    runArm(page, {
      runId: "test-run", arm: "t0", backend: "wasm", detector: "core-orchestrator" as const,
      provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
      itemTimeoutMs: 10_000,
      items: [{ id: "a", text: "hello", policy: "p-fin", gold: [] }],
    }),
  ).rejects.toThrow(/not prepared/i);
});

test("stamps every record with both the IR artifact's digest and the IR's own policy hash", async ({
  page,
}) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));
  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items: items.slice(0, 2),
  });

  // Computed in Node, from the bytes on disk, and deliberately NOT from
  // anything the page handed back. That is the whole property being pinned: a
  // record's irHash must be reproducible from outside the browser --
  // `shasum -a 256 apps/eval/fixtures/minimal-ir.json` prints this same value --
  // so "these numbers came from that artifact" is auditable rather than
  // asserted. Recomputing it from the page's own IR text would prove only that
  // sha256 is a function.
  const expected = createHash("sha256").update(readFileSync(IR_FIXTURE)).digest("hex");
  expect(records.map((r) => r.irHash)).toEqual([expected, expected]);

  // And the IR's OWN policyHash, carried verbatim beside it. The two fields
  // answer different questions -- which artifact ran, versus which policy
  // document it was compiled from -- so a record that dropped this one would
  // break the document -> IR -> numbers chain. Read out of the fixture rather
  // than written here, so it is the artifact's claim and not this file's.
  const declared = (JSON.parse(readFileSync(IR_FIXTURE, "utf8")) as { policyHash: string })
    .policyHash;
  expect(records.map((r) => r.policyHash)).toEqual([declared, declared]);
  // The placeholder that made a single field impossible, asserted rather than
  // described: this value is real in the fixture, and it is not a digest.
  expect(declared).toBe("test-hash");
  expect(records[0]!.irHash).not.toBe(records[0]!.policyHash);
});

test("refuses a page whose irHash is not a digest, before running the corpus", async ({ page }) => {
  // The hash is identical on every record in an arm, so a bad one invalidates
  // the entire output; discovering that after a full corpus has run costs the
  // whole arm. The fixture's literal `policyHash` is used as the bad value
  // because forwarding `ir.policyHash` into `irHash` is exactly the mistake the
  // guard exists to refuse.
  await page.goto("about:blank");
  await page.evaluate(() => {
    Object.defineProperty(window, "__sih", {
      value: {
        detect: () => Promise.reject(new Error("detect must not be reached")),
        irHash: () => Promise.resolve("test-hash"),
        policyHash: () => "test-hash",
      },
    });
  });

  await expect(
    runArm(page, {
      runId: "test-run",
      arm: "t0",
      backend: "wasm",
      detector: "core-orchestrator",
      provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
      itemTimeoutMs: 10_000,
      items: [{ id: "a", text: "hello", policy: "p-fin", gold: [] }],
    }),
  ).rejects.toThrow(/irHash/);
});

test("hands detect the arm's backend rather than leaving the label unbacked", async ({ page }) => {
  // `ArmSpec.backend` used to reach nothing: it was copied onto every record and
  // never plumbed into the config forwarded to `detect`, so the page's
  // reconciliation against what actually executed could not fire unless the
  // CALLER remembered to set `config.backend` too. Asserted on what the page was
  // handed, not on the record, because runArm builds one object and uses it for
  // both -- a record agreeing with itself would prove nothing.
  await installFakeSih(page, {}, { findings: [], timings: { tier0Ms: 1 }, degraded: [] });

  await runArm(page, {
    runId: "test-run",
    arm: "t0+t1",
    backend: "webgpu",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items: [
      { id: "a", text: "one", policy: "p-fin", gold: [] },
      { id: "b", text: "two", policy: "p-fin", gold: [] },
    ],
  });

  const seen = await page.evaluate(
    () => (window as unknown as { __configs?: unknown[] }).__configs ?? [],
  );
  expect(seen).toEqual([
    { tier0: true, tier1: false, tier2: false, backend: "webgpu" },
    { tier0: true, tier1: false, tier2: false, backend: "webgpu" },
  ]);
});

test("refuses an arm whose config names a different backend than the arm does", async ({ page }) => {
  // Overwriting the caller's value silently would pick one of two stated
  // intentions and record it as fact. Checked before the page is touched, so it
  // fails on any page at all -- this one has never been navigated.
  await expect(
    runArm(page, {
      runId: "test-run",
      arm: "t0+t1",
      backend: "wasm",
      detector: "core-orchestrator",
      provider: "claude",
      config: { tier0: true, tier1: true, tier2: false, backend: "webgpu" },
      itemTimeoutMs: 10_000,
      items: [{ id: "a", text: "hello", policy: "p-fin", gold: [] }],
    }),
  ).rejects.toThrow(/one arm cannot measure two runtimes/);
});

test("refuses an Approach-B arm carrying an escalation threshold", async ({ page }) => {
  // The sibling of the check above and checked in the same place, before the
  // page is touched. B judges the WHOLE MESSAGE in one call and never
  // escalates, so `uncertainBelow` selected nothing: a row carrying it would
  // name a knob that turned nothing, which is the intent-as-fact defect
  // arriving through a field that happens to be available.
  //
  // Refused rather than deleted, because deleting it silently would let a
  // caller believe it had varied an experiment variable this arm has no
  // equivalent of. `runBakeoff` cannot reach this throw -- `planBakeoff` omits
  // the threshold on a family whose `runsCompiledJudge` is false, and
  // `bakeoff-run.test.ts` asserts that -- so this is the only test of it, and
  // without it deleting the whole guard leaves the suite green.
  await expect(
    runArm(page, {
      runId: "test-run",
      arm: "baselineB-model",
      backend: "webgpu",
      detector: "approach-b",
      provider: "claude",
      config: { tier0: false, tier1: false, tier2: true, uncertainBelow: 0.6 },
      itemTimeoutMs: 10_000,
      items: [{ id: "a", text: "hello", policy: "p-fin", gold: [] }],
    }),
  ).rejects.toThrow(/the row would name a knob that turned nothing/);
});

test("stamps the tier-1 config it was given onto every record, and nothing when given none", async ({
  page,
}) => {
  // The six-dimensional rung has nowhere else to go: `TierConfig` has no
  // `threshold`, `maxWidth` or `labelForm`, so without this field two arms
  // differing only in those are byte-identical in the output. runArm cannot
  // verify the config describes the tagger the page holds -- it never loads one
  // -- which is why RunRecordSchema couples the two and runMatrix validates
  // every record before writing.
  await installFakeSih(page, {}, { findings: [], timings: { tier0Ms: 1, tier1Ms: 2 }, degraded: [] });
  const items = [{ id: "a", text: "one", policy: "p-fin", gold: [] }];

  const withTier1 = await runArm(page, {
    runId: "test-run",
    arm: "t0+t1",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: true, tier2: false },
    tier1Config: {
      modelId: "gliner-pii-edge-uint8",
      backend: "wasm",
      threshold: 0.02,
      maxWidth: 12,
      labelForm: "id",
    },
    itemTimeoutMs: 10_000,
    items,
  });
  expect(withTier1[0]!.tier1Config).toEqual({
    modelId: "gliner-pii-edge-uint8",
    backend: "wasm",
    threshold: 0.02,
    maxWidth: 12,
    labelForm: "id",
  });

  const withoutTier1 = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items,
  });
  expect(withoutTier1[0]!.tier1Config).toBeUndefined();
});

test("carries the tier-1 tagger's own counters onto every record", async ({ page }) => {
  // The finding this closes: runArm built each record from `findings` and
  // `timings` only and never called `tier1Status()`, so an item whose tail
  // `maxLen` truncated -- or whose span the offset mapper refused -- emitted a
  // record byte-indistinguishable from one where the model read the whole
  // message and found nothing. Both have no findings and a real `tier1Ms`.
  // `Tier1TaggerStats` exists precisely to tell those apart, and the counters
  // were dying one layer above the tagger.
  //
  // The values below are deliberately NOT zeros: a test whose expected stats
  // are all zero passes just as well against a producer that hardcodes zeros.
  await installFakeSih(
    page,
    {},
    { findings: [], timings: { tier0Ms: 1, tier1Ms: 40 }, degraded: [] },
    {
      inferences: 3,
      droppedWords: 2,
      truncatedWords: 17,
      overWideSpans: 4,
      unmappableSpans: 1,
      nonFiniteScores: 5,
      // Present on the page's Tier1DetectStats and deliberately not on a
      // record: it counts the PAGE's GPU submissions, not the tagger's work.
      gpuSubmits: 99,
    },
  );

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t1",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: false, tier1: true, tier2: false },
    tier1Config: {
      modelId: "gliner-pii-edge-uint8",
      backend: "wasm",
      threshold: 0.5,
      maxWidth: 12,
      labelForm: "id",
    },
    itemTimeoutMs: 10_000,
    items: [{ id: "a", text: "one", policy: "minimal-fixture", gold: [] }],
  });

  expect(records[0]!.tier1Stats).toEqual({
    inferences: 3,
    droppedWords: 2,
    truncatedWords: 17,
    overWideSpans: 4,
    unmappableSpans: 1,
    nonFiniteScores: 5,
  });
  // Projected, not spread: `gpuSubmits` is on the object the page hands over
  // and must not ride into the file on a field the schema never declared.
  expect(Object.keys(records[0]!.tier1Stats!).sort()).toEqual([
    "droppedWords",
    "inferences",
    "nonFiniteScores",
    "overWideSpans",
    "truncatedWords",
    "unmappableSpans",
  ]);
  // A truncated tail is now visible in the record, which is the whole point:
  // this row and a row from a model that read everything and found nothing are
  // no longer the same bytes.
  expect(records[0]!.findings).toEqual([]);
  expect(records[0]!.tier1Stats!.truncatedWords).toBeGreaterThan(0);
  expect(RunRecordSchema.safeParse(records[0]).success).toBe(true);
});

test("asks the page for the tier-1 counters without a second round trip per item", async ({
  page,
}) => {
  // The constraint the fix had to respect. runArm makes exactly one evaluate
  // per item plus two fixed ones (the readiness probe and the two hashes), and
  // reading `lastDetect` in its own evaluate would have doubled the per-item
  // cost over a 1,500-item corpus for a value already sitting in the page --
  // and would have raced the loop, since the next `detect` overwrites it.
  await installFakeSih(page, {}, { findings: [], timings: { tier0Ms: 1, tier1Ms: 2 }, degraded: [] });
  const watched = watchEvaluate(page);
  const items = [
    { id: "a", text: "one", policy: "minimal-fixture", gold: [] },
    { id: "b", text: "two", policy: "minimal-fixture", gold: [] },
    { id: "c", text: "three", policy: "minimal-fixture", gold: [] },
  ];
  const records = await runArm(watched.page, {
    runId: "test-run",
    arm: "t1",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: false, tier1: true, tier2: false },
    tier1Config: {
      modelId: "gliner-pii-edge-uint8",
      backend: "wasm",
      threshold: 0.5,
      maxWidth: 12,
      labelForm: "id",
    },
    itemTimeoutMs: 10_000,
    items,
  });
  expect(records.every((r) => r.tier1Stats !== undefined)).toBe(true);
  // 2 fixed + 1 per item. Pinned exactly, because "not many more" is the
  // property a regression would satisfy.
  expect(watched.calls()).toBe(2 + items.length);
  expect(watched.maxInFlight()).toBe(1);
});

test("marks every row measured after a deadline expiry, and no row before it", async ({ page }) => {
  // `page.evaluate` has no cancellation channel, so the abandoned detection
  // keeps running in the browser and every later item is timed under
  // contention with it -- while carrying `error: null`, so a latency aggregate
  // over non-errored rows silently includes them. The arm is flagged rather
  // than aborted; `abandonedWorkInFlight` in record.ts states why.
  await installFakeSih(page, { "wedges forever": "hang" }, { findings: [], timings: { tier0Ms: 2 }, degraded: [] });

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 250,
    items: [
      { id: "before", text: "fine", policy: "minimal-fixture", gold: [] },
      { id: "hangs", text: "wedges forever", policy: "minimal-fixture", gold: [] },
      { id: "after", text: "also fine", policy: "minimal-fixture", gold: [] },
    ],
  });

  expect(records.map((r) => r.itemId)).toEqual(["before", "hangs", "after"]);
  expect(records.map((r) => r.abandonedWorkInFlight)).toEqual([false, false, true]);
  // The timed-out row is not flagged BY ITS OWN expiry -- it already carries
  // `error`, and what the flag marks is a row whose timings were taken under
  // someone else's work. This pairing is what a scorer filters on.
  expect(records[1]!.error).toMatch(/did not finish within 250ms/);
  expect(records[2]!.error).toBeNull();
  expect(records[2]!.timings.tier0Ms).toBe(2);
});

test("does not flag an arm whose items merely THREW", async ({ page }) => {
  // The control the test above needs. A detection that rejects promptly leaves
  // nothing running in the browser, so later latencies are clean -- flagging
  // them would tell Plan 8 to discard good measurements. Only a deadline
  // expiry, which abandons work still executing, sets the flag.
  await installFakeSih(page, { "boom": "throw" }, { findings: [], timings: { tier0Ms: 2 }, degraded: [] });
  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items: [
      { id: "explodes", text: "boom", policy: "minimal-fixture", gold: [] },
      { id: "after", text: "fine", policy: "minimal-fixture", gold: [] },
    ],
  });
  expect(records[0]!.error).toMatch(/detector exploded/);
  expect(records.map((r) => r.abandonedWorkInFlight)).toEqual([false, false]);
});

test("carries core's degradation notices onto every record", async ({ page }) => {
  // The highest-severity thing this file guards, and it is a DROP rather than a
  // mistake: `runArm` projected `DetectionResult` field by field -- `findings`
  // and `timings` -- so `degraded` never reached a record. Adding it to
  // `DetectionResult` produced no type error here and no change in the output,
  // which is exactly why it went unnoticed.
  //
  // Run against the REAL page and the real orchestrator, not the stand-in, so
  // the expectation comes from what core does rather than from canned data.
  // `absent` is the only word a tier-0 arm can produce, and it is produced
  // twice -- the two tiers this TierConfig switched off.
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items: [
      { id: "a", text: "one", policy: "minimal-fixture", gold: [] },
      { id: "b", text: "two", policy: "minimal-fixture", gold: [] },
    ],
  });

  for (const record of records) {
    // Tier and reason asserted exactly; `detail` only for being non-empty,
    // because restating core's sentence here would make this a test of a string
    // literal rather than of the channel.
    expect(record.degraded?.map((d) => [d.tier, d.reason])).toEqual([
      [1, "absent"],
      [2, "absent"],
    ]);
    expect(record.degraded?.every((d) => d.detail.length > 0)).toBe(true);
    expect(RunRecordSchema.safeParse(record).success).toBe(true);
  }
  // Not an empty array: a producer that stubbed the field would satisfy the
  // schema's presence rule and say nothing. This is what says the array came
  // from the orchestrator.
  expect(records[0]!.degraded).toHaveLength(2);
});

test("carries the judge's counters and its per-call rows onto every record", async ({ page }) => {
  // The tier-2 twin of the tier-1 counters spec above, and the same finding one
  // tier up: without these, an item where the judge failed closed on every
  // segment emits a record indistinguishable from one where it read the whole
  // message and found nothing -- both have no findings and a real `tier2Ms`.
  //
  // Values deliberately NOT zeros, and every one of them distinct, so a
  // producer that hardcodes zeros or writes a counter into the wrong slot is
  // visible. `calls` carries TWO rows that disagree on `finishReason`: a
  // message makes one call per selected segment, and this is the shape a single
  // per-message `finishReason` could not express.
  await installFakeSih(
    page,
    {},
    { findings: [], timings: { tier0Ms: 1, tier2Ms: 4210 }, degraded: [] },
    undefined,
    {
      rung1: 3, rung2: 1, unresolvedQuotes: 2, unknownPredicates: 6,
      duplicatesDropped: 4, repairAttempts: 5, failedClosed: 7, truncatedResponses: 8,
      abortedResponses: 9, segmentsJudged: 10, segmentsSkipped: 11, deadlineExpiries: 12,
      callerAbortsMidGeneration: 13, callerAbortsWhileQueued: 14,
      calls: [
        { finishReason: "stop", promptTokens: 1105, completionTokens: 96, ttftMs: 780 },
        { finishReason: "length", promptTokens: 1196, completionTokens: 512, ttftMs: 940 },
      ],
      // Not a field of `JudgeStats`. A page that decorated its stats would ride
      // the extra key into the JSONL unless runArm projects what it declares.
      debugNote: "should not reach the file",
    },
  );

  const watched = watchEvaluate(page);
  const records = await runArm(watched.page, {
    runId: "test-run",
    arm: "t0+t2",
    backend: "webgpu",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: true, t2Model: "Qwen3.5-2B-q4f16_1-MLC" },
    tier2Config: FAKE_TIER2_CONFIG,
    itemTimeoutMs: 10_000,
    // TWO items, not one, and the second is what this spec turns on. The stand-in
    // answers a CUMULATIVE `totals` beside the per-item `lastDetect`, exactly as
    // `WebLlmJudge.stats` does -- and at ONE item the two are equal, so a driver
    // reading the running sum would pass a single-item fixture unchanged. The
    // assertion below is on record[1], where they differ.
    items: [
      { id: "a", text: "one", policy: "minimal-fixture", gold: [] },
      { id: "b", text: "two", policy: "minimal-fixture", gold: [] },
    ],
  });

  expect(records[1]!.tier2Stats).toEqual({
    rung1: 3, rung2: 1, unresolvedQuotes: 2, unknownPredicates: 6,
    duplicatesDropped: 4, repairAttempts: 5, failedClosed: 7, truncatedResponses: 8,
    abortedResponses: 9, segmentsJudged: 10, segmentsSkipped: 11, deadlineExpiries: 12,
    callerAbortsMidGeneration: 13, callerAbortsWhileQueued: 14,
    calls: [
      { finishReason: "stop", promptTokens: 1105, completionTokens: 96, ttftMs: 780 },
      { finishReason: "length", promptTokens: 1196, completionTokens: 512, ttftMs: 940 },
    ],
  });
  // Both rows carry the same per-item numbers, which is what a DELTA means: the
  // second message is not the first message's work plus its own.
  expect(records[0]!.tier2Stats).toEqual(records[1]!.tier2Stats);
  expect(Object.keys(records[1]!.tier2Stats!)).not.toContain("debugNote");
  // The per-call rows are rows, in order: the second call hit the token ceiling
  // while the first stopped cleanly, so this message's judgement is partial.
  // No single per-message value can say that. TWO rows on the second record as
  // well -- reading the cumulative view would have found four there.
  expect(records[1]!.tier2Stats!.calls.map((c) => c.finishReason)).toEqual(["stop", "length"]);
  for (const record of records) expect(RunRecordSchema.safeParse(record).success).toBe(true);
  // And it cost no extra round trip: one evaluate per item plus the readiness
  // probe and the hash read, exactly as the tier-1 counters do. Reading
  // `lastDetect` in its own evaluate would double the per-item cost over a
  // 1,500-item corpus AND race the loop, since the next `detect` overwrites it.
  expect(watched.calls()).toBe(records.length + 2);
});

test("keeps a call row silent about what the engine did not report", async ({ page }) => {
  // The producer half of `Tier2CallSchema`'s optional fields, and the half no
  // fixture in this file used to reach: every other call row here supplies all
  // five values, so the `| undefined` branch of the projection was never
  // exercised HERE -- and `record.test.ts` pinning `calls: [{}]` tests the
  // SCHEMA, which by construction cannot see a producer that filled the blanks
  // in before the schema ever ran. The review that found this measured
  // `finishReason ?? "stop"`, `promptTokens ?? 0` and a `ttftMs` mapping that
  // turns undefined into null each surviving the whole Node suite and all 23
  // tests this file then had.
  //
  // Why the defaults would be wrong rather than merely tidy: 0.2.84 types
  // `ChatCompletion.usage` optional and assigns `finish_reason` from
  // `getFinishReason()`, declared `| undefined`, so both absences are real.
  // `"stop"` would record "we do not know why this stopped" as "it finished
  // normally" -- and `bakeoff.ts` reads truncation accounting straight off this
  // column -- while a 0-token row would enter the decode-rate arithmetic as a
  // call that decoded nothing.
  await installFakeSih(
    page,
    {},
    { findings: [], timings: { tier0Ms: 1, tier2Ms: 60 }, degraded: [] },
    undefined,
    {
      ...FAKE_TIER2_STATS,
      segmentsJudged: 1,
      // An engine that answered and reported nothing about the answer. The
      // second row is the same call with a decoration on it, which is what
      // makes the projection's field list load-bearing one level below
      // `findings`: `run.ts` says naming the fields is what keeps the emitted
      // shape a property of that module, and nothing tested that for a CALL.
      calls: [{}, { debugNote: "should not reach the file" }],
    },
  );

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0+t2",
    backend: "webgpu",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: true },
    tier2Config: FAKE_TIER2_CONFIG,
    itemTimeoutMs: 10_000,
    items: [{ id: "a", text: "one", policy: "minimal-fixture", gold: [] }],
  });

  // Asserted on the SERIALIZED row, which is what Plan 8 reads: an invented
  // `finishReason` or a 0 token count shows up here as a key that should not
  // exist, and `JSON.stringify` drops the undefined-valued keys the projection
  // does set. Both rows, so the decoration is covered too.
  const [line] = toJsonl(records).trim().split("\n");
  const written = JSON.parse(line!) as { tier2Stats: { calls: Record<string, unknown>[] } };
  expect(written.tier2Stats.calls).toEqual([{}, {}]);
  // And in memory, where `undefined` and "absent" are still distinguishable:
  // `toEqual` treats an undefined-valued key as absent, which is exactly the
  // equivalence `JSON.stringify` makes, so a null would fail both.
  expect(records[0]!.tier2Stats!.calls).toEqual([{}, {}]);
  expect(RunRecordSchema.safeParse(JSON.parse(line!)).success).toBe(true);
});

test("projects a degradation notice field by field, like a finding", async ({ page }) => {
  // The `degraded` half of the same projection guarantee. `run.ts` argues for
  // it in as many words -- "nothing validates a record on this path, so a
  // producer that decorates ... would have them ride into the JSONL unchecked"
  // -- and that claim was pinned for `findings` and for the tier2Stats OBJECT
  // and for nothing else. The review that found this measured a cast in place of
  // the notice projection surviving the whole suite, and a spread over the call
  // rows surviving it too.
  //
  // That the extra key really would reach the file: `runMatrix` validates with
  // `safeParse` but writes the RAW records, and zod is not strict here, so a
  // decorated notice is written and then accepted on the way back in.
  await installFakeSih(page, {}, {
    findings: [],
    timings: { tier0Ms: 1 },
    degraded: [{ tier: 1, reason: "absent", detail: "tier 1 was switched off", debugNote: "x" }],
  });

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    items: [{ id: "a", text: "one", policy: "minimal-fixture", gold: [] }],
  });

  expect(records[0]!.degraded).toEqual([
    { tier: 1, reason: "absent", detail: "tier 1 was switched off" },
  ]);
  expect(Object.keys(records[0]!.degraded![0]!).sort()).toEqual(["detail", "reason", "tier"]);
});

test("writes a non-finite time-to-first-token or decode rate as null instead of a NaN", async ({ page }) => {
  // MEASURED with zod 4.4.3: `z.number()` rejects NaN, and `JSON.stringify(NaN)`
  // is the string "null". So a NaN copied straight through produces a file its
  // own reader refuses -- written as null, rejected on the way back in, after
  // the GPU time is spent. `JudgeCallRecord` says the conversion out of
  // `usage.extra.time_to_first_token_s` is deliberately unguarded because a NaN
  // there is a fact about the call, so the mapping has to happen here.
  //
  // MEASURED: Playwright's protocol preserves NaN across `evaluate`, so the NaN
  // really does reach runArm rather than arriving as null already -- without
  // which this spec would pass against a producer that does nothing.
  await page.goto("about:blank");
  expect(await page.evaluate(() => Number.NaN)).toBeNaN();

  await installFakeSih(
    page,
    {},
    { findings: [], timings: { tier0Ms: 1, tier2Ms: 100 }, degraded: [] },
    undefined,
    {
      ...FAKE_TIER2_STATS,
      segmentsJudged: 1,
      // BOTH mappings, and `decodeTokPerSec` is the one that is more reachable:
      // the library computes it as `completion_tokens / decode_time` with no
      // zero guard, and `judge.test.ts` asserts the 0/0 a call interrupted
      // before its first token produces. The plan says most messages on this
      // corpus expire mid-run, so a NaN reaching the record is the ordinary
      // case rather than the exotic one -- and it would make
      // `RunRecordSchema.safeParse` reject the row, which `runBakeoff` turns
      // into "produced an invalid record at row N" AFTER the arm's GPU time is
      // spent and BEFORE the file is written.
      calls: [
        {
          finishReason: "stop",
          promptTokens: 1105,
          completionTokens: 20,
          ttftMs: Number.NaN,
          decodeTokPerSec: Number.NaN,
        },
      ],
    },
  );

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0+t2",
    backend: "webgpu",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: true },
    tier2Config: FAKE_TIER2_CONFIG,
    itemTimeoutMs: 10_000,
    items: [{ id: "a", text: "one", policy: "minimal-fixture", gold: [] }],
  });

  expect(records[0]!.tier2Stats!.calls[0]!.ttftMs).toBeNull();
  expect(records[0]!.tier2Stats!.calls[0]!.decodeTokPerSec).toBeNull();
  // The whole point: it survives the round trip Plan 8 makes.
  const [line] = toJsonl(records).trim().split("\n");
  expect(RunRecordSchema.safeParse(JSON.parse(line!)).success).toBe(true);
});

test("resolves the escalation threshold and hands detect the value it records", async ({ page }) => {
  // `TierConfig.uncertainBelow` is an EXPERIMENT variable the bake-off varies
  // per arm, and `TierConfig` says so in as many words. Two tier-2 arms
  // differing only in it used to emit records identical in every field a scorer
  // can group by.
  //
  // TWO values, one of them not the default, because a spec that only ever
  // exercises the default cannot tell "carries the config" from "hardcodes
  // UNCERTAIN_BELOW" -- the trap this plan has already sprung twice.
  await installFakeSih(page, {}, { findings: [], timings: { tier0Ms: 1, tier2Ms: 9 }, degraded: [] });
  const items = [{ id: "a", text: "one", policy: "minimal-fixture", gold: [] }];
  const spec = {
    runId: "test-run",
    arm: "t0+t2",
    backend: "webgpu" as const,
    detector: "core-orchestrator" as const,
    provider: "claude",
    itemTimeoutMs: 10_000,
    items,
  };

  const explicit = await runArm(page, {
    ...spec,
    config: { tier0: true, tier1: false, tier2: true, uncertainBelow: 0.35 },
    tier2Config: FAKE_TIER2_CONFIG,
  });
  expect(explicit[0]!.config.uncertainBelow).toBe(0.35);

  const defaulted = await runArm(page, {
    ...spec,
    config: { tier0: true, tier1: false, tier2: true },
    tier2Config: FAKE_TIER2_CONFIG,
  });
  // RESOLVED, not left absent. A record that says nothing is a record whose
  // threshold has to be reconstructed from a constant in another package, and
  // the schema refuses it for that reason.
  expect(defaulted[0]!.config.uncertainBelow).toBe(UNCERTAIN_BELOW);

  // ... and `detect` was handed the same object, which no assertion on the
  // record can establish on its own: runArm builds one config and uses it for
  // both, so a record agreeing with itself proves nothing about the call.
  const seen = await page.evaluate(
    () => (window as unknown as { __configs?: { uncertainBelow?: number }[] }).__configs ?? [],
  );
  expect(seen.map((c) => c.uncertainBelow)).toEqual([0.35, UNCERTAIN_BELOW]);

  // A tier-0 arm is not given one: escalation never runs, so there is no
  // threshold to state.
  const tier0 = await runArm(page, {
    ...spec,
    config: { tier0: true, tier1: false, tier2: false },
  });
  expect(tier0[0]!.config.uncertainBelow).toBeUndefined();
});

test("leaves the tier-2 evidence off an item that threw, rather than the previous item's", async ({
  page,
}) => {
  // `tier2Status().lastDetect` is a DELTA over the judge's cumulative counters.
  // On a thrown item that delta belongs to the PREVIOUS item, so copying it
  // would attribute one message's failures to another -- and a row of zeros
  // would assert that nothing failed closed on an item whose judge may never
  // have returned. `degraded` goes with it: `detect` throws whole, so there is
  // no result to read an array off.
  await installFakeSih(
    page,
    { "second text": "throw" },
    { findings: [], timings: { tier0Ms: 1, tier2Ms: 50 }, degraded: [] },
    undefined,
    { ...FAKE_TIER2_STATS, segmentsJudged: 2, failedClosed: 1 },
  );

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0+t2",
    backend: "webgpu",
    detector: "core-orchestrator",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: true },
    tier2Config: FAKE_TIER2_CONFIG,
    itemTimeoutMs: 10_000,
    items: [
      { id: "a", text: "first text", policy: "minimal-fixture", gold: [] },
      { id: "b", text: "second text", policy: "minimal-fixture", gold: [] },
      { id: "c", text: "third text", policy: "minimal-fixture", gold: [] },
    ],
  });

  expect(records[0]!.tier2Stats?.failedClosed).toBe(1);
  expect(records[1]!.error).toMatch(/exploded/);
  expect(records[1]!.tier2Stats).toBeUndefined();
  expect(records[1]!.degraded).toBeUndefined();
  expect(records[2]!.tier2Stats?.failedClosed).toBe(1);
  for (const record of records) expect(RunRecordSchema.safeParse(record).success).toBe(true);
});

test("emits a file that separates failing closed on 40% from finding nothing", async ({ page }) => {
  // The comparison `DetectionResult.degraded` was added for, end to end and
  // read off the FILE rather than off the objects runArm returned -- which is
  // the only version that can fail, since zod strips what a schema does not
  // declare.
  //
  // Both arms find nothing on every message, so recall, precision and every
  // span-derived number are identical. The judge's counters are identical too:
  // the same stand-in stats answer both arms, which is what leaves `degraded`
  // as the only thing that can be carrying the difference.
  const items = [1, 2, 3, 4, 5].map((n) => ({
    id: `m${String(n)}`,
    text: `message ${String(n)}`,
    policy: "minimal-fixture",
    gold: [],
  }));
  const spec = {
    runId: "test-run",
    arm: "t0+t2",
    backend: "webgpu" as const,
    detector: "core-orchestrator" as const,
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: true, uncertainBelow: UNCERTAIN_BELOW },
    tier2Config: FAKE_TIER2_CONFIG,
    itemTimeoutMs: 10_000,
    items,
  };
  const canned = { findings: [], timings: { tier0Ms: 1, tier2Ms: 3000 }, degraded: [] };

  await installFakeSih(page, { "message 2": "failed-closed", "message 4": "failed-closed" }, canned);
  const closed = await runArm(page, spec);
  await installFakeSih(page, {}, canned);
  const quiet = await runArm(page, spec);

  /** Records as Plan 8 will hold them: written as JSONL and parsed back. */
  const throughTheFile = (records: RunRecord[]): RunRecord[] =>
    toJsonl(records)
      .trim()
      .split("\n")
      .map((line) => RunRecordSchema.parse(JSON.parse(line)));

  /**
   * THE SCORER, per MESSAGE. `JudgeStats.failedClosed` counts SEGMENTS and
   * cannot answer "on what fraction of messages", which is the number the
   * bake-off table reports.
   */
  const failedClosedMessageRate = (records: RunRecord[]) => {
    const scoreable = records.filter((r) => r.error === null);
    return (
      scoreable.filter((r) =>
        (r.degraded ?? []).some((d) => d.tier === 2 && d.reason === "failed-closed"),
      ).length / scoreable.length
    );
  };

  const failing = throughTheFile(closed);
  const finding = throughTheFile(quiet);

  // Identical in every field the old projection copied. Timings are dropped
  // from the comparison only because tier0Ms is a wall clock; nothing else is.
  const withoutDegraded = (r: RunRecord) => {
    const { degraded: _dropped, timings: _clock, ...rest } = r;
    return rest;
  };
  expect(failing.map(withoutDegraded)).toEqual(finding.map(withoutDegraded));
  expect(failing.every((r) => r.findings.length === 0)).toBe(true);
  expect(finding.every((r) => r.findings.length === 0)).toBe(true);

  // And two different results once `degraded` is read.
  expect(failedClosedMessageRate(failing)).toBeCloseTo(0.4, 10);
  expect(failedClosedMessageRate(finding)).toBe(0);
});
