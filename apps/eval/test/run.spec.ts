import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema } from "../src/driver/record.js";
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

/** What the stand-in page does when `detect` is called with a given item's text. */
type FakeVerdict = "ok" | "throw" | "hang" | "kill";

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
 */
async function installFakeSih(
  page: Page,
  plan: Record<string, FakeVerdict>,
  result: { findings: unknown[]; timings: { tier0Ms: number; tier1Ms?: number; tier2Ms?: number } },
): Promise<void> {
  await page.goto("about:blank");
  await page.evaluate(
    ({ plan: verdicts, result: canned, hash }) => {
      Object.defineProperty(window, "__sih", {
        configurable: true,
        value: {
          irHash: () => Promise.resolve(hash),
          policyHash: () => "test-hash",
          detect: ({ text }: { text: string }) => {
            const verdict = Object.hasOwn(verdicts, text) ? verdicts[text] : "ok";
            if (verdict === "hang") return new Promise(() => {});
            if (verdict === "kill") {
              delete window.__sih;
              return Promise.reject(new Error("the page went away"));
            }
            if (verdict === "throw") return Promise.reject(new Error("detector exploded"));
            return Promise.resolve(canned);
          },
        },
      });
    },
    { plan, result, hash: FAKE_IR_HASH },
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
    provider: "claude",
    policy: "p-fin",
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
  await installFakeSih(page, { "middle text": "throw" }, { findings: [], timings: { tier0Ms: 1.5 } });

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
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
    },
  );

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0+t1",
    backend: "wasm",
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
  // The config that ran, recorded rather than implied by the arm name.
  expect(records[0]!.config).toEqual({ tier0: true, tier1: true, tier2: false, t1Model: "fake" });
});

test("times an item out instead of wedging the whole arm on it", async ({ page }) => {
  // A `detect` that never settles would otherwise hang runArm forever, and
  // because records are returned only in bulk at the end, every already
  // completed record would be unreachable. The budget is per item and comes
  // from the caller.
  await installFakeSih(page, { "wedges forever": "hang" }, { findings: [], timings: { tier0Ms: 2 } });

  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
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
  await installFakeSih(page, { "kills the page": "kill" }, { findings: [], timings: { tier0Ms: 1 } });

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
      runId: "test-run", arm: "t0", backend: "wasm", provider: "claude",
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
      provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
      itemTimeoutMs: 10_000,
      items: [{ id: "a", text: "hello", policy: "p-fin", gold: [] }],
    }),
  ).rejects.toThrow(/irHash/);
});
