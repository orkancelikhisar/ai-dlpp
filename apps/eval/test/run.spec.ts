import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema } from "../src/driver/record.js";
import { runArm } from "../src/driver/run.js";

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const FIXTURES = join(REPO_ROOT, "corpora", "fixtures");
const IR_FIXTURE = join(REPO_ROOT, "apps", "eval", "fixtures", "minimal-ir.json");

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
 * Wraps a real Page so `evaluate` still runs for real, and records the high-water
 * mark of concurrent calls. An observer, not a stub: every call reaches the same
 * page object with the same arguments and `this` bound to the real page, so what
 * gets measured is unchanged and only the call PATTERN is visible.
 */
function watchEvaluate(page: Page): { page: Page; maxInFlight: () => number } {
  let live = 0;
  let max = 0;
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
        max = Math.max(max, live);
        try {
          return await (value as (...a: unknown[]) => Promise<unknown>).apply(target, args);
        } finally {
          live -= 1;
        }
      };
    },
  });
  return { page: watched, maxInFlight: () => max };
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
    items,
  });

  expect(records).toHaveLength(items.length);
  for (const record of records) {
    expect(RunRecordSchema.safeParse(record).success).toBe(true);
  }
  // Record order matches corpus order, so a diff between two runs lines up.
  expect(records.map((r) => r.itemId)).toEqual(items.map((i) => i.id));

  // The pass-through fields, every value distinct so a field written into the
  // wrong slot shows up. Nothing else here reads them and RunRecordSchema only
  // asks that they be non-empty strings, so without this a runArm that stamped
  // the arm name where the provider belongs would emit a perfectly valid file
  // that mislabels every row of the comparison it exists to feed.
  expect(records[0]).toMatchObject({
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    provider: "claude",
    policy: "p-fin",
  });

  // Everything above is satisfied by a runArm that never calls detect at all:
  // RunRecordSchema accepts an empty `findings`, and 7 of these 13 items are
  // negatives that legitimately have none. So pin the one thing only a real
  // detection run produces -- every tier-0 GOLD span is covered by some finding.
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
  let covered = 0;
  for (const record of records) {
    for (const gold of record.gold) {
      if (!tier0.has(gold.entityType)) continue;
      const hit = record.findings.some((f) => f.start < gold.end && gold.start < f.end);
      expect(
        hit,
        `${record.itemId}: no finding overlaps gold ${gold.entityType} [${gold.start},${gold.end})`,
      ).toBe(true);
      covered += 1;
    }
  }
  // The loop above passes vacuously if `gold` were dropped from the records or
  // the corpus stopped carrying tier-0 positives. This corpus has five.
  expect(covered).toBe(5);
});

test("runs items one at a time, never overlapping them", async ({ page }) => {
  // `runArm` documents sequential execution as load-bearing -- latency is a
  // reported metric and concurrent inference on one GPU measures contention
  // rather than the model. Nothing else in this file can see the difference:
  // MEASURED, replacing the loop with `Promise.all(spec.items.map(...))` leaves
  // every other spec here green -- under that mutation the records still came
  // out in corpus order, so the order assertion in the first spec does not stand
  // in for this one.
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
    items,
  });

  expect(records).toHaveLength(items.length);
  expect(watched.maxInFlight()).toBe(1);
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
    items: [{ id: "boom", text: "anything at all", policy: "p-fin", gold: [] }],
  });

  expect(records).toHaveLength(1);
  expect(records[0]!.error).toMatch(/tier1/i);
  expect(records[0]!.findings).toEqual([]);
});

test("refuses an unprepared page rather than silently measuring a blank one", async ({ page }) => {
  // If runArm navigated on its own, Task 12 would load a tier-1 model and then
  // have it thrown away — producing a full JSONL file of tier-0 results
  // labelled as a tier-1 arm. Nothing downstream could detect that.
  await expect(
    runArm(page, {
      runId: "test-run", arm: "t0", backend: "wasm", provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
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

test("refuses a page whose irHash is not a digest, before running the corpus", async ({
  page,
}) => {
  // The hash is identical on every record in an arm, so a bad one invalidates
  // the entire output; discovering that after a full corpus has run costs the
  // whole arm. This is the one spec that installs a FAKE `__sih` -- it exercises
  // a driver guard and never measures detection. It is only possible because
  // `about:blank` has no `__sih`: the harness page defines its own non-writable
  // and non-configurable, so nothing can stub the real page out.
  await page.goto("about:blank");
  await page.evaluate(() => {
    Object.defineProperty(window, "__sih", {
      value: {
        detect: () => Promise.reject(new Error("detect must not be reached")),
        // The IR fixture's literal `policyHash` field, which is what forwarding
        // `ir.policyHash` into `irHash` would put on every record.
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
      items: [{ id: "a", text: "hello", policy: "p-fin", gold: [] }],
    }),
  ).rejects.toThrow(/irHash/);
});
