import { readFileSync } from "node:fs";
import { TIER2_MODELS } from "@sih/tier2";
import {
  assertItemTimeoutMs,
  itemDeadlineBound,
  runBakeoff,
  slateBakeoffOptions,
  type ArmGateReport,
} from "../src/driver/bakeoff.js";
import { expect, openHarness, test } from "./tier2-profile.js";

/**
 * THE COMMAND that runs the bake-off.
 *
 * ```
 * SIH_BAKEOFF=1 SIH_BAKEOFF_RUN_ID=<name> pnpm -C apps/eval bakeoff
 * ```
 *
 * ## Why this file exists
 *
 * `runBakeoff` had no caller outside the test suite and nothing in this
 * repository ran it over the slate. Two specs drive it -- `bakeoff.spec.ts` with
 * one model over two items and `baseline.spec.ts` with one model over four
 * families and three items -- and both are pipe-integrity checks on the driver.
 * Neither is the four-model bake-off spec 4.2 describes, and an apparatus with
 * no way to invoke it is one a reader cannot tell apart from a finished
 * experiment. As of the commit that added this file **the four-arm bake-off has
 * still never been run**; what changed is that there is now a command for it.
 *
 * ## Why it is a spec rather than a script
 *
 * Everything a slate run needs is already a Playwright concern and none of it is
 * optional. The persistent browser profile is a correctness matter, not
 * convenience -- `tier2-profile.ts` measured an ordinary context reporting a
 * 3,221 MB quota against the 7.49 GB the four pinned arms occupy, and a
 * `QuotaExceededError` mid-download looks exactly like a model that cannot load.
 * `workers: 1`, the `chromium` channel (the headless shell's `requestAdapter()`
 * resolves null and web-llm throws on a null adapter) and the dev server all come
 * from `playwright.config.ts`. A standalone Node CLI would have to re-create all
 * four, and each copy is free to drift from the one the other specs measure
 * under.
 *
 * ## Why it skips by default
 *
 * A slate run is four model loads and hours of GPU time; `pnpm -r test` must not
 * start one. It is gated on `SIH_BAKEOFF`, and a skip here is REPORTED AS A SKIP
 * rather than as a pass -- which matters in this repository, because "the specs
 * passed" has never meant "tier 2 ran". The Node-side test below is not gated
 * and does run in every suite pass: it asserts that the default options are the
 * four-arm slate, so this entry point cannot quietly become a one-model run.
 */

/** Above the 121,020 ms bound this driver derives for `semantic-ir.json`; see `slateBakeoffOptions`. */
const DEFAULT_ITEM_TIMEOUT_MS = 250_000;

/** Four model loads, four corpus passes. Wall clock, not a latency target. */
const SLATE_TIMEOUT_MS = 4 * 60 * 60 * 1000;

test("the entry point's defaults are the four-arm slate, not a one-model smoke run", () => {
  // NOT gated on SIH_BAKEOFF: it costs milliseconds and it is the only thing
  // standing between "the command runs the slate" and "the command runs
  // whatever the last person debugging it left in the defaults".
  const options = slateBakeoffOptions({ SIH_BAKEOFF_RUN_ID: "check" });
  expect(options.models).toEqual(TIER2_MODELS.map((m) => m.id));
  expect(options.models).toHaveLength(4);
  // Left unset, so `planBakeoff`'s own DEFAULT_FAMILIES stays the single
  // definition of what a slate runs.
  expect(options.families).toBeUndefined();
  expect(options.itemTimeoutMs).toBe(DEFAULT_ITEM_TIMEOUT_MS);
  expect(options.corpus.endsWith("corpora/fixtures/smoke.jsonl")).toBe(true);

  // The run id is required and is not generated. It is half of every output
  // file's name and `runBakeoff` refuses to overwrite one, so a timestamp
  // default would make a repeat of one experiment look like two.
  expect(() => slateBakeoffOptions({})).toThrow(/SIH_BAKEOFF_RUN_ID is required/);
  // The EMPTY-STRING half of that guard, which is the reachable one: an unset
  // shell variable and one set to "" are the same thing to a caller and
  // different things to `env[...]`. Without this half the refusal still
  // happens, but from `assertFileSafe` deep inside `planBakeoff`, with a
  // message about file names rather than about why the id is required.
  expect(() => slateBakeoffOptions({ SIH_BAKEOFF_RUN_ID: "" })).toThrow(
    /SIH_BAKEOFF_RUN_ID is required/,
  );

  // `Number` and not `parseInt`, which is the one decision in this function
  // whose comment argues for it and which nothing checked: `parseInt("250s")`
  // is 250, and a silently truncated timeout that still runs is worse than one
  // that refuses. `Number("250s")` is NaN, which falls through to
  // `assertItemTimeoutMs` naming the field and the reason.
  const fatFingered = slateBakeoffOptions({
    SIH_BAKEOFF_RUN_ID: "check",
    SIH_BAKEOFF_ITEM_TIMEOUT_MS: "250s",
  });
  expect(Number.isNaN(fatFingered.itemTimeoutMs)).toBe(true);
  const bound = itemDeadlineBound({
    latencyBudgetMs: 120_000,
    callBudgetMs: 60_000,
    maxCallsPerItem: 6,
    lowerTierAllowanceMs: 1_000,
  });
  expect(() => assertItemTimeoutMs(fatFingered.itemTimeoutMs, bound)).toThrow(/itemTimeoutMs/);
  // The control: the SAME bound accepts the number a clean string produces, so
  // the throw above is about NaN and not about the bound being unsatisfiable.
  expect(() => assertItemTimeoutMs(400_000, bound)).not.toThrow();
  // And a clean numeric string with surrounding whitespace still parses, so the
  // stricter reader has not made the common case fail.
  expect(
    slateBakeoffOptions({ SIH_BAKEOFF_RUN_ID: "check", SIH_BAKEOFF_ITEM_TIMEOUT_MS: " 300000 " })
      .itemTimeoutMs,
  ).toBe(300_000);

  // And every field is really read from the environment. Without a second value
  // per field, "reads the env" and "returns the default" are the same test --
  // the trap this plan has already sprung twice.
  const overridden = slateBakeoffOptions({
    SIH_BAKEOFF_RUN_ID: "other",
    SIH_BAKEOFF_MODELS: `${TIER2_MODELS[1]!.id}, ${TIER2_MODELS[0]!.id}`,
    SIH_BAKEOFF_FAMILIES: "baseline-b,baseline-b-tier0",
    SIH_BAKEOFF_CORPUS: "/tmp/other.jsonl",
    SIH_BAKEOFF_OUT_DIR: "/tmp/out",
    SIH_BAKEOFF_PROVIDER: "deepseek",
    SIH_BAKEOFF_IR: "p-fin",
    SIH_BAKEOFF_IR_PATH: "/tmp/p-fin.ir.json",
    SIH_BAKEOFF_POLICY_PATH: "/tmp/p-fin.md",
    SIH_BAKEOFF_ITEM_TIMEOUT_MS: "400000",
  });
  expect(overridden.runId).toBe("other");
  // Passed through in the caller's order; `planBakeoff` is what sorts the slate
  // cheapest-first, so a reordering here would be a second sort free to drift.
  expect(overridden.models).toEqual([TIER2_MODELS[1]!.id, TIER2_MODELS[0]!.id]);
  expect(overridden.families).toEqual(["baseline-b", "baseline-b-tier0"]);
  expect(overridden.corpus).toBe("/tmp/other.jsonl");
  expect(overridden.outDir).toBe("/tmp/out");
  expect(overridden.provider).toBe("deepseek");
  expect(overridden.irName).toBe("p-fin");
  expect(overridden.irPath).toBe("/tmp/p-fin.ir.json");
  expect(overridden.policyPath).toBe("/tmp/p-fin.md");
  expect(overridden.itemTimeoutMs).toBe(400_000);
});

test("runs the slate", async ({ page }) => {
  test.skip(
    process.env["SIH_BAKEOFF"] !== "1",
    "a slate run is four model loads and hours of GPU time; set SIH_BAKEOFF=1 and " +
      "SIH_BAKEOFF_RUN_ID=<name> to run it. THIS IS A SKIP, NOT A PASS: nothing below ran.",
  );
  test.setTimeout(SLATE_TIMEOUT_MS);
  await openHarness(page);
  test.skip(
    !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
    "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
  );

  const options = slateBakeoffOptions(process.env);
  const result = await runBakeoff(page, options);

  const gates = readFileSync(result.gatesPath, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as ArmGateReport);
  for (const g of gates) {
    console.log(
      `[slate] ${g.arm} (${g.family}, judgedUnit ${g.judgedUnit}): ` +
        `load ${g.engineLoadMs.toFixed(0)}ms + warmup ${g.engineWarmupMs.toFixed(0)}ms, ` +
        `origin storage ${(g.originStorageBytes / 1e9).toFixed(2)} GB, ` +
        `${g.answeredCalls} answered call(s), ttftMs ${JSON.stringify(g.ttftMs)}, ` +
        `ladder ${JSON.stringify(g.ladder)}, ` +
        `gates ${g.gates.map((x) => `${x.gate}=${x.verdict}(n=${x.sample}/min${x.minSample})`).join(" ")}, ` +
        `killedOnRunGates=${g.killedOnRunGates}`,
    );
  }

  // One file and one gates line per PLANNED arm. The set of arms is fixed before
  // the first model loads and each file is written before its gate is computed,
  // so a killed arm is a row here rather than an absence -- which is the one
  // rule this driver must not break.
  expect(result.written).toHaveLength(result.plan.arms.length);
  expect(gates).toHaveLength(result.plan.arms.length);

  // EVERY arm reached the engine. Without this the files are complete,
  // schema-valid transcripts of models that never ran, every gate reads
  // "not measured", and the run looks like it happened.
  for (const g of gates) expect(g.answeredCalls, g.arm).toBeGreaterThan(0);

  // The hardware half of the research question is populated, on every row. A
  // zero here would mean the page reported no load cost and the row recorded
  // that as a measurement.
  for (const g of gates) {
    expect(g.engineLoadMs, g.arm).toBeGreaterThan(0);
    expect(g.engineWarmupMs, g.arm).toBeGreaterThan(0);
    expect(g.originStorageBytes, g.arm).toBeGreaterThan(0);
  }

  // NOT asserted: that any arm passes its gates. A killed arm is a result and
  // spec 4.2's primary criterion is task accuracy, which nothing here computes
  // -- see `ArmGateReport.scoring.verdictMeans` and `experimentScope`, both of
  // which are on every row of the file this just wrote.
});
