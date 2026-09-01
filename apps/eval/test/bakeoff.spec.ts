import { mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIER2_MODELS } from "@sih/tier2";
import { runBakeoff, type ArmGateReport } from "../src/driver/bakeoff.js";
import { loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema } from "../src/driver/record.js";
import { expect, openHarness, test } from "./tier2-profile.js";

/**
 * The bake-off driver's BROWSER half.
 *
 * Everything that can be settled in Node -- planning, the deadline derivation,
 * the gate arithmetic, and `runBakeoff`'s own refusals against a scripted page
 * -- is in `bakeoff.test.ts` and `bakeoff-run.test.ts`, and is not repeated
 * here. What is left needs a real page, and one test here needs a real model.
 *
 * ## What this file deliberately is not
 *
 * It is not the bake-off. Running the slate is four models and the 7.49 GB of
 * weights `tier2-profile.ts` measured, and is a separate decision; this runs ONE
 * arm over a two-item slice,
 * which is a pipe-integrity check on the driver and produces no accuracy number
 * in either direction. The slice exists to keep a suite run to minutes rather
 * than an hour, and it is the FIRST two items of the shipped corpus rather than
 * a hand-picked pair -- picking items by what they produce is how a harness gets
 * tuned to look good.
 */

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const CORPUS = join(REPO_ROOT, "corpora", "fixtures", "smoke.jsonl");
const FIXTURES = join(REPO_ROOT, "apps", "eval", "fixtures");

/** The cheapest arm, which is the one `TIER2_MODELS` puts first. */
const CHEAPEST = TIER2_MODELS[0]!.id;

/**
 * Above the 121,020 ms bound `semantic-ir.json`'s 120,000 ms message budget and
 * the page's 60,000 ms per-call budget produce. `deriveItemTimeoutMs` returns
 * 242,040; this is that, rounded up.
 */
const ITEM_TIMEOUT_MS = 250_000;

/** A warm cache read of a 1 GB model, then a corpus pass, then a second load. */
const ARM_TIMEOUT_MS = 900_000;

function freshOutDir(): string {
  return mkdtempSync(join(tmpdir(), "sih-bakeoff-"));
}

test("refuses to run when the page's IR is not the file the driver planned from", async ({ page }) => {
  // No model is loaded and none needs to be: this refusal happens before
  // planning. It is the check that keeps the ceiling, the escalation and the
  // policy pairing honest -- all three are computed here in Node from a file on
  // disk, while the records carry the PAGE's `irHash`, and nothing else in the
  // pipeline compares the two.
  const out = freshOutDir();
  await openHarness(page);
  await expect(
    runBakeoff(page, {
      runId: "mismatch",
      outDir: out,
      corpus: CORPUS,
      provider: "claude",
      models: [CHEAPEST],
      irName: "semantic",
      // A real fixture, and the wrong one: `minimal-ir.json` is what the page
      // loads by default and is not what `useIr("semantic")` selects.
      irPath: join(FIXTURES, "minimal-ir.json"),
      itemTimeoutMs: ITEM_TIMEOUT_MS,
    }),
  ).rejects.toThrow(/hashes to/);
  // And nothing was written: a refusal that had already created files would
  // leave a directory that looks like a partial run.
  expect(readdirSync(out)).toEqual([]);
});

test("runs one arm end to end and writes the gate verdict beside the records", async ({ page }) => {
  test.setTimeout(ARM_TIMEOUT_MS);
  await openHarness(page);
  test.skip(
    !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
    "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
  );

  const out = freshOutDir();
  // The first two items of the shipped corpus, written out as a corpus of their
  // own so this test costs four model calls rather than eighteen. Sliced, not
  // edited: every field is the shipped item's. (Four and eighteen because the
  // default `compiled` family runs tier 0 and `semantic-ir.json`'s entropy rule
  // re-admits this corpus's code fence -- which is in the second of these two
  // items. It was three and seventeen while that IR carried no rules.)
  const slice = readFileSync(CORPUS, "utf8").trim().split("\n").slice(0, 2);
  const slicePath = join(out, "slice.jsonl");
  writeFileSync(slicePath, slice.join("\n") + "\n");
  expect(loadCorpus(readFileSync(slicePath, "utf8"))).toHaveLength(2);

  const result = await runBakeoff(page, {
    runId: "smoke",
    outDir: out,
    corpus: slicePath,
    provider: "claude",
    models: [CHEAPEST],
    irName: "semantic",
    itemTimeoutMs: ITEM_TIMEOUT_MS,
  });

  // One file per PLANNED arm. The set of arms is fixed before the first model
  // loads and the files are written before any gate is computed, so an arm
  // cannot be dropped for its verdict -- which is this driver's central rule.
  expect(result.written).toHaveLength(result.plan.arms.length);
  expect(result.written).toHaveLength(1);
  const records = readFileSync(result.written[0]!, "utf8").trim().split("\n").map((line) => JSON.parse(line) as unknown);
  expect(records).toHaveLength(2);
  for (const record of records) {
    const parsed = RunRecordSchema.safeParse(record);
    if (!parsed.success) throw new Error(`invalid record: ${JSON.stringify(parsed.error.issues)}`);
    // The arm's label and the model it ran are the same claim in two places.
    expect(parsed.data.arm).toBe(`tier2-${CHEAPEST}`);
    expect(parsed.data.config.t2Model).toBe(CHEAPEST);
  }

  // The gate verdict is a FILE, one line per arm, written whatever the verdict
  // says -- so a killed arm is a row in it rather than an absence.
  const gates = readFileSync(result.gatesPath, "utf8").trim().split("\n").map((l) => JSON.parse(l) as ArmGateReport);
  expect(gates).toHaveLength(result.plan.arms.length);
  const report = gates[0]!;
  console.log(
    `[bakeoff] ${report.arm}: ${report.answeredCalls} call(s), ` +
      report.gates.map((g) => `${g.gate}=${g.verdict}${g.observed === undefined ? "" : `(${g.observed.toFixed(1)})`}`).join(" ") +
      ` | promptTokens ${JSON.stringify(report.promptTokens)} completionTokens ` +
      `${JSON.stringify(report.completionTokens)} segmentChars ${JSON.stringify(report.segmentChars)}` +
      ` | degraded ${JSON.stringify(report.degradedNotices)}`,
  );
  // Printed beside the numbers, not only written into the file: the person
  // reading this console line is the person who just spent the GPU time, and
  // `killedOnRunGates` is the field they will otherwise read as a result.
  console.log(
    `[bakeoff] ${report.arm}: killedOnRunGates=${report.killedOnRunGates} ` +
      `(run gates only, accuracyGated=${report.scoring.accuracyGated}); tiers run ` +
      `[${report.scoring.tiersRun.join(", ")}], this corpus cannot score ` +
      `[${report.scoring.tiersThisCorpusCannotScore.join(", ")}]` +
      report.scoring.cannotScore.map((line) => `\n  - ${line}`).join(""),
  );
  // The model ANSWERED. Without this the file below is a complete, schema-valid
  // transcript of a judge that returned before it touched the engine, and every
  // gate would read "not measured" while the run looked like it happened.
  expect(report.answeredCalls).toBeGreaterThan(0);
  expect(report.ttftCalls).toBeGreaterThan(0);
  // Both halves of the comparability evidence the p95 gate needs are present:
  // the engine's prompt tokens over exactly those calls, and the character
  // sizes of the segments this arm ran.
  expect(report.promptTokens).not.toBeUndefined();
  expect(report.segmentChars).not.toBeUndefined();
  // Every reason word is a key, so a zero is a measurement rather than a gap.
  expect(Object.keys(report.degradedNotices).sort()).toEqual([
    "absent",
    "budget-exhausted",
    "call-budget-exhausted",
    "failed-closed",
    "scope-unjudged",
  ]);
  // Tier 1 is off on every bake-off arm, so its `absent` notice is on every row.
  expect(report.degradedNotices["absent"]).toBe(2);

  // And the row says what it cannot say. This slice is the corpus's first two
  // items, whose four gold spans are all tier 0, while the arm runs tier 0 and
  // tier 2 -- so every tier-2 finding the model just produced is unmatchable,
  // and the row names that rather than leaving it to a scorer to discover.
  expect(report.scoring.tiersRun).toEqual([0, 2]);
  expect(report.scoring.goldSpansByTier[2]).toBe(0);
  expect(report.scoring.tiersThisCorpusCannotScore).toEqual([2]);
  expect(report.scoring.accuracyGated).toBe(false);
});

test("refuses a second run under the same runId rather than overwriting a measurement", async ({ page }) => {
  const out = freshOutDir();
  // Pre-create the file the arm would write. No model loads: the check runs
  // before the first `loadTier2`, which is the whole point -- learning that a
  // runId was reused after an arm has finished costs the arm.
  writeFileSync(join(out, `dup.tier2-${CHEAPEST}.jsonl`), "");
  await openHarness(page);
  await expect(
    runBakeoff(page, {
      runId: "dup",
      outDir: out,
      corpus: CORPUS,
      provider: "claude",
      models: [CHEAPEST],
      irName: "semantic",
      itemTimeoutMs: ITEM_TIMEOUT_MS,
    }),
  ).rejects.toThrow(/already exists/);
});
