import { DEFAULT_TIER2_CONFIG } from "@sih/tier2";
import { WEBLLM_ADAPTER_FLOOR, meetsWebLlmAdapterFloor } from "../src/page/webgpu-floor.js";
import { BASE_URL, expect, openHarness, test, tier2ProfileDir } from "./tier2-profile.js";

/**
 * Tier 2, executing in real Chrome: the engine lifecycle, the context window
 * the engine actually enforces, and the wedge.
 *
 * Serial, and for a reason stronger than tier 1's: these tests share ONE
 * browser profile, and a model loaded into a second page while the first still
 * holds one would put two copies of 2-4 GB of weights on one GPU.
 */
test.describe.configure({ mode: "serial" });

/** A cold load can be a multi-gigabyte download; a warm one is a cache read. */
const LOAD_TIMEOUT_MS = 600_000;

const MODEL = DEFAULT_TIER2_CONFIG.modelId;

/**
 * A message the semantic fixture's one predicate is about, with nothing in it
 * for tier 0 or tier 1 to find. Nothing here asserts the model FINDS anything:
 * measured, these models miss most of what they are shown, and a recall
 * assertion would turn this suite into a model-quality gate that fails for
 * reasons which are the finding rather than a regression.
 */
const MESSAGE = "Please review the Northwind Traders renewal before Friday.";

/**
 * The same subject in TWO segments, so a second `detect` on one arm's judge
 * costs a different amount of work than the first.
 *
 * `segmentText` splits a kv line run from a prose line run, and escalation
 * keeps both (neither is code), so this message selects two segments where
 * `MESSAGE` selects one -- VERIFIED against core's own `segmentText`, which
 * returns `[kv 0..27, prose 27..67]` for exactly this string. That asymmetry is
 * the whole point: a projection that reported the judge's running TOTAL, or a
 * constant, agrees with the delta on a fixture where every message costs the
 * same.
 */
const TWO_SEGMENT_MESSAGE = "Account: Northwind Traders\nPlease review the renewal before Friday.";

const TIER2_ONLY = { tier0: false, tier1: false, tier2: true } as const;

test("the browser profile has room for the pinned arms", async ({ page }) => {
  await openHarness(page);
  const estimate = await page.evaluate(async () => {
    const { usage, quota } = await navigator.storage.estimate();
    return { usage: usage ?? 0, quota: quota ?? 0 };
  });
  console.log(
    `[tier2] profile ${tier2ProfileDir()} at ${BASE_URL}: ` +
      `${(estimate.usage / 1e9).toFixed(2)} GB used of ${(estimate.quota / 1e9).toFixed(2)} GB`,
  );
  // 8 GB, against the 7.49 GB the four pinned arms occupy at this one origin
  // (measured: see `tier2-profile.ts`). This is the assertion that fails when
  // the persistent profile is lost, and it was checked by losing it: with the
  // fixture mutated to `browser.newContext()` this line failed at 4,295 MB, and
  // a standalone ordinary context measured 3,221 MB. The threshold sits between
  // those and the 10,737 MB an empty persistent profile reports, so neither
  // side of it is a rounding error.
  expect(estimate.quota).toBeGreaterThan(8e9);
});

test("webgpuAvailable answers this adapter's real limits, and says which one refused", async ({
  page,
}) => {
  // One of the two tier-2 tests that must NEVER skip, because this one is about
  // the check every other tier-2 test skips on. `test.skip(!webgpuAvailable())`
  // turns any error in that check into a green run of zero tests. MEASURED HERE
  // on the suite as it stands, with `meetsWebLlmAdapterFloor` forced to return
  // false: `playwright test tier2.spec.ts tier2-arms.spec.ts` reports 9 skipped,
  // 2 passed, exit 0 -- of the 76 tests `playwright test --list` counts across
  // the 7 spec files, 11 are tier-2 tests and 9 of them are gated on that call.
  // (The review that found this quoted "all nine tier-2 tests" and "54
  // Playwright tests", which were the counts at the time and are not the counts
  // now; two tier-2 tests have been added since.)
  //
  // WHAT THIS TEST CATCHES, stated narrowly because an earlier version of this
  // comment overstated it. `webgpu-floor.test.ts` owns the comparison itself --
  // both directions, from synthetic limit objects, since no single machine can
  // supply both -- and it re-reads the four floor NUMBERS out of the installed
  // web-llm bundle, which is what guards them. What is left for this test is the
  // WIRING, which only a browser can answer: that the page asks the real
  // adapter, and that its answer is what the tested function gives for those
  // limits.
  //
  // So the expectation below is not a restatement of the PAGE's answer -- it is
  // computed from the adapter's own numbers -- but it IS computed with the same
  // constant table the function under test reads, and that half is tautological:
  // raise a floor and both sides move together, this test stays green, and the
  // whole tier-2 suite skips. Only `webgpu-floor.test.ts` fails then, which is
  // exactly why it re-reads the numbers from the library instead of from here.
  await openHarness(page);
  const observed = await page.evaluate(async (floorKeys: readonly string[]) => {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<null | { limits: Record<string, number | undefined> }> } }).gpu;
    const available = await window.__sih!.webgpuAvailable();
    if (gpu === undefined) return { available, limits: undefined };
    const adapter = await gpu.requestAdapter();
    if (adapter === null) return { available, limits: undefined };
    // Copied key by key: `GPUSupportedLimits` members are on the prototype, so
    // structuredClone of the object itself crosses the boundary empty.
    const limits: Record<string, number | undefined> = {};
    for (const key of floorKeys) limits[key] = adapter.limits[key];
    return { available, limits };
  }, Object.keys(WEBLLM_ADAPTER_FLOOR));

  console.log(`[tier2] adapter limits ${JSON.stringify(observed.limits)} -> available ${String(observed.available)}`);

  if (observed.limits === undefined) {
    // No `navigator.gpu`, or a null adapter. web-llm throws on both, so the
    // only honest answer is false -- and a page that returned true here would
    // send every later spec into a load that cannot start.
    expect(observed.available).toBe(false);
    return;
  }
  expect(observed.available).toBe(meetsWebLlmAdapterFloor(observed.limits));
  // Every limit the floor names was reported by this adapter. Without this the
  // line above is satisfied by an evaluate that returned an empty object and a
  // page that returned false: two wrongs agreeing.
  for (const limit of Object.keys(WEBLLM_ADAPTER_FLOOR)) {
    expect(typeof observed.limits[limit], `adapter did not report ${limit}`).toBe("number");
  }
});

test("loads a tier-2 model in real Chrome and reports what the engine answered", async ({
  page,
}) => {
  test.setTimeout(LOAD_TIMEOUT_MS);
  await openHarness(page);
  const available = await page.evaluate(() => window.__sih!.webgpuAvailable());
  test.skip(!available, "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded");

  const report = await page.evaluate(
    async (modelId: string) => window.__sih!.loadTier2({ modelId }),
    MODEL,
  );
  console.log(
    `[tier2] load ${report.servedModelId}: load ${report.loadMs.toFixed(0)}ms, ` +
      `warm-up ${report.warmupMs.toFixed(0)}ms finish=${String(report.warmupFinishReason)} ` +
      `tokens=${String(report.warmupCompletionTokens)}, storage ` +
      `${(report.storageUsageBytes / 1e9).toFixed(2)} GB`,
  );

  // The ENGINE's answer, not the request. The plan's own version of this test
  // asserted `report.loadedModelId`, which cannot pass: 0.2.84 exposes no
  // accessor for the model an engine loaded -- `MLCEngineInterface` declares
  // none and `MLCEngine.loadedModelIdToPipeline` is private -- so the only
  // channel is `ChatCompletion.model`, and a load that asked the engine nothing
  // has nothing to report. `loadTier2` therefore makes one throwaway
  // completion, and this is that completion's `model`.
  //
  // What it proves is narrower than "these weights ran", and saying so is the
  // point of the name: traced through the bundle, the field is the id we passed
  // to `CreateMLCEngine` laundered through a Map key. What it does prove is that
  // an engine ANSWERED under the id this arm is named for -- which the
  // requested id copied into a new field would claim for a browser whose GPU
  // process had died, and which is exactly the shape of Plan 4's dead browser
  // producing a complete, schema-valid output file.
  expect(report.servedModelId).toBe(MODEL);
  // Every field of the config, because a report that named half of them would
  // describe an arm nobody can reproduce. These are what was REQUESTED --
  // `Tier2LoadReport` says so, and the window one is measured against the
  // engine in the next test rather than asserted as an observation here.
  expect(report.config.modelId).toBe(MODEL);
  expect(report.config.contextWindowSize).toBe(8192);
  expect(report.config.temperature).toBe(0);
  expect(report.config.maxTokens).toBe(512);
  // Never "abort": that is the engine saying an interrupt cut the answer off,
  // and an engine that starts latched fails every later call instantly and
  // emptily. "stop" is what a completed grammar-constrained answer reports.
  expect(report.warmupFinishReason).toBe("stop");
  expect(report.loadMs).toBeGreaterThan(0);
  expect(report.warmupMs).toBeGreaterThan(0);
  // The page's own default, which is what a caller naming no budget gets.
  expect(report.callBudgetMs).toBe(60_000);

  const status = await page.evaluate(() => window.__sih!.tier2Status());
  expect(status?.load.servedModelId).toBe(MODEL);
  // The warm-up went through the engine and not through the judge, so no counter
  // moved and no `detect` has run: a spec asking "did tier 2 run during MY
  // detect" must not be answered with this page's own warm-up call.
  expect(status?.totals.calls).toHaveLength(0);
  expect(status?.totals.segmentsJudged).toBe(0);
  expect(status?.lastDetect).toBeUndefined();

  // ... and the SECOND load, whose only job is to exercise `callBudgetMs` at a
  // value that is not the default. `bakeoff.ts` passes one explicitly and then
  // refuses an arm whose page reports back a different number -- but that check
  // compares 60,000 with 60,000, because `bakeoff.ts` carries its own copy of
  // the same default. So a `loadTier2` that dropped `options.callBudgetMs` and
  // always used the module constant would satisfy every check in this
  // repository, and the review that found this measured that it did: the
  // mutation survived the whole tier-2 suite and the bake-off spec. A test that
  // only ever exercises the default
  // cannot tell "reads the option" from "hardcodes the default", the trap this
  // plan has already sprung twice.
  //
  // What this proves and what it does not: `loadTier2` resolves the budget into
  // ONE `const` and uses that same binding both to construct the judge and to
  // fill this field, so a wrong number here is a wrong number in the judge --
  // but this assertion reads the field, not the judge. `detectWithBudget` at
  // 50 ms in the wedge test below is what shows a per-call budget being
  // ENFORCED, on the path that takes one explicitly.
  const rebudgeted = await page.evaluate(
    async (modelId: string) => window.__sih!.loadTier2({ modelId, callBudgetMs: 45_000 }),
    MODEL,
  );
  expect(rebudgeted.callBudgetMs).toBe(45_000);
  expect(rebudgeted.servedModelId).toBe(MODEL);

  // And a `detect` that NAMES A DIFFERENT MODEL than the loaded engine is
  // refused, which is the tier-2 twin of the backend guard one tier down.
  // `config.t2Model` is copied onto every record by `runArm`, so without this a
  // file can name one model for work another one did -- and it is a cheap
  // mistake to make from outside this harness, where nothing checks a load
  // against a config. The bake-off driver closes the same door one level up,
  // once per arm; this closes it per call, for every other caller of the page.
  //
  // No engine call is made: the refusal is before `detect`, which is why this
  // costs nothing to assert here.
  const refusal = await page.evaluate(
    async ([wrongModel, text]) =>
      window
        .__sih!.detect({
          text: text as string,
          provider: "claude",
          config: { tier0: false, tier1: false, tier2: true, t2Model: wrongModel as string },
        })
        .then(() => "no refusal")
        .catch((cause: unknown) => (cause as Error).message),
    ["Phi-4-mini-instruct-q4f16_1-MLC", MESSAGE] as const,
  );
  expect(refusal).toContain("Phi-4-mini-instruct-q4f16_1-MLC");
  expect(refusal).toContain(MODEL);

  // The control that keeps the check from being "any t2Model is refused": the
  // loaded arm's own id passes, and passing is what every bake-off record does.
  const accepted = await page.evaluate(
    async ([rightModel, text]) =>
      window
        .__sih!.detect({
          text: text as string,
          provider: "claude",
          config: { tier0: false, tier1: false, tier2: true, t2Model: rightModel as string },
        })
        .then(() => "accepted")
        .catch((cause: unknown) => (cause as Error).message),
    [MODEL, MESSAGE] as const,
  );
  expect(accepted).toBe("accepted");
});

test("the requested context window is the one the engine enforces", async ({ page }) => {
  test.setTimeout(LOAD_TIMEOUT_MS);
  await openHarness(page);
  test.skip(
    !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
    "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
  );

  // 2,000 filler words, which is between the two windows on every pinned arm
  // and is not a number to change casually: MEASURED, this generator produces
  // 5,809 tokens on this model and 4,023 on Phi-4-mini, so a smaller count
  // would be accepted by a 4,096 window on the arm with the densest tokenizer
  // and the probe would prove nothing there.
  const WORDS = 2000;
  const PROBE_BUDGET_MS = 300_000;

  const wide = await page.evaluate(
    async ([modelId, words, budgetMs]) => {
      await window.__sih!.loadTier2({ modelId: modelId as string, contextWindowSize: 8192 });
      return window.__sih!.probeContextWindow({ words: words as number, budgetMs: budgetMs as number });
    },
    [MODEL, WORDS, PROBE_BUDGET_MS] as const,
  );
  console.log(`[tier2] window 8192: ${JSON.stringify(wide)}`);

  // The engine prefilled a prompt longer than 4,096 tokens, so the window it is
  // enforcing is larger than the one `prebuiltAppConfig` overrides this model to
  // (all four pinned arms ship `overrides.context_window_size: 4096`). The token
  // count is the ENGINE's own `usage.prompt_tokens`, not an estimate taken in
  // the page, so an arithmetic error here cannot produce this pass.
  expect(wide.accepted).toBe(true);
  expect(wide.requestedContextWindowSize).toBe(8192);
  expect(wide.promptTokens).toBeGreaterThan(4096);
  expect(wide.servedModelId).toBe(MODEL);

  // The CONTROL, and this test is worth little without it: it is what shows the
  // probe can fail. Same model, same prompt, one setting changed -- and it also
  // exercises `contextWindowSize` at a NON-DEFAULT value, where a `loadTier2`
  // that hardcoded 8192 (or ignored the config entirely) would be caught. A
  // test that only ever loads the default cannot tell "reads the config" from
  // "hardcodes the default", and this project has shipped that twice.
  const narrow = await page.evaluate(
    async ([modelId, words, budgetMs]) => {
      const report = await window.__sih!.loadTier2({
        modelId: modelId as string,
        contextWindowSize: 4096,
      });
      const probe = await window.__sih!.probeContextWindow({
        words: words as number,
        budgetMs: budgetMs as number,
      });
      return { report, probe };
    },
    [MODEL, WORDS, PROBE_BUDGET_MS] as const,
  );
  console.log(`[tier2] window 4096: ${JSON.stringify(narrow.probe)}`);

  expect(narrow.report.config.contextWindowSize).toBe(4096);
  expect(narrow.probe.accepted).toBe(false);
  expect(narrow.probe.errorName).toBe("ContextWindowSizeExceededError");
  // The library naming the window it enforced. This is the only place 0.2.84
  // states that number, and it is why the refusal is the measurement rather
  // than a failure: `getInputTokens` throws when
  // `numPromptTokens + filledKVCacheLength > contextWindowSize` and puts both
  // numbers in the message.
  expect(narrow.probe.errorMessage).toContain("context window size: 4096");
});

test("a judged message produces findings whose spans core accepts", async ({ page }) => {
  test.setTimeout(LOAD_TIMEOUT_MS);
  await openHarness(page);
  test.skip(
    !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
    "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
  );

  const run = await page.evaluate(
    async ([modelId, text]) => {
      // The semantic fixture, and the whole test turns on it: the other two
      // IRs declare `semanticPredicates: []`, and on this config -- tier 0 and
      // tier 1 both off -- nothing is left uncertain either, so
      // `selectSegments` keeps no segment, the orchestrator never calls
      // `judge()`, and `timings.tier2Ms` is never set. Against them this test
      // would FAIL, at the `tier2Ms` assertion below, on an undefined; the
      // hazard is that it would fail reading like a broken harness rather than
      // like an IR with no clause for tier 2 to judge.
      await window.__sih!.useIr("semantic");
      await window.__sih!.loadTier2({ modelId });
      const result = await window.__sih!.detect({
        text,
        provider: "claude",
        config: { tier0: false, tier1: false, tier2: true },
      });
      return { result, status: window.__sih!.tier2Status() };
    },
    [MODEL, MESSAGE] as const,
  );

  console.log(
    `[tier2] findings ${JSON.stringify(run.result.findings.map((f) => f.text))}; ` +
      `stats ${JSON.stringify(run.status?.lastDetect)}`,
  );

  // Reaching this line at all means every offset survived normalizeFindings,
  // which enforces `text === message.slice(start, end)` and rejects an
  // entityType the IR does not declare.
  for (const finding of run.result.findings) {
    expect(finding.tier).toBe(2);
    expect(finding.entityType).toBe("pred:unannounced-deal");
    expect(MESSAGE.slice(finding.start, finding.end)).toBe(finding.text);
  }
  expect(run.result.timings.tier2Ms).toBeGreaterThan(0);

  // The part `findings` and `tier2Ms` cannot establish, and the reason
  // `tier2Status().lastDetect` exists. An empty findings array is a legitimate
  // outcome here -- these models miss most entities -- so what has to be
  // asserted is that a model call HAPPENED and the engine answered it.
  const stats = run.status?.lastDetect;
  expect(stats).toBeDefined();
  expect(stats!.calls.length).toBeGreaterThan(0);
  // Prompt tokens come from the engine's own `usage`, so a call row cannot be
  // manufactured by a page that never reached a model.
  expect(stats!.calls[0]!.promptTokens).toBeGreaterThan(0);
  // Exactly the segments the message has: this one is a single prose segment,
  // and the judge's own invariant is that judged + failed-closed + skipped
  // covers every segment it was handed.
  expect(stats!.segmentsJudged + stats!.failedClosed + stats!.segmentsSkipped).toBe(1);
  // Nothing interrupted this run: an "abort" here would mean the engine was
  // latched before the test started.
  expect(stats!.abortedResponses).toBe(0);
  expect(stats!.deadlineExpiries).toBe(0);
});

test("a second detect on one arm reports its own work, not the arm's running total", async ({
  page,
}) => {
  test.setTimeout(LOAD_TIMEOUT_MS);
  await openHarness(page);
  test.skip(
    !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
    "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
  );

  // The shape no other test in this file has, and the reason it is worth a
  // second model call: every other tier-2 test loads a fresh page, loads an arm
  // (which builds a fresh judge) and does exactly ONE `detect`. On that fixture
  // `before` is all-zero, so a projection that returned the judge's cumulative
  // TOTAL, or the whole call history, or a literal, is indistinguishable from
  // the delta -- the review that found this measured all three surviving the
  // suite as it stood.
  //
  // `judge-delta.test.ts` pins the arithmetic itself under vitest, where a
  // nonzero `before` is one object literal. What is left is the WIRING, which
  // only the browser can answer: that the page snapshots the arm's judge around
  // each `detect` and hands `runArm` the difference. The two messages are
  // deliberately different SIZES -- one selected segment against two -- so an
  // arm where every message costs the same cannot hide a total behind a delta.
  const run = await page.evaluate(
    async ([modelId, one, two]) => {
      await window.__sih!.useIr("semantic");
      await window.__sih!.loadTier2({ modelId });
      const config = { tier0: false, tier1: false, tier2: true } as const;
      await window.__sih!.detect({ text: one, provider: "claude", config });
      const first = window.__sih!.tier2Status()!.lastDetect;
      await window.__sih!.detect({ text: two, provider: "claude", config });
      const status = window.__sih!.tier2Status()!;
      return { first, second: status.lastDetect, totals: status.totals };
    },
    [MODEL, MESSAGE, TWO_SEGMENT_MESSAGE] as const,
  );

  console.log(
    `[tier2] delta1 ${JSON.stringify(run.first)}\n[tier2] delta2 ${JSON.stringify(run.second)}` +
      `\n[tier2] totals ${JSON.stringify(run.totals)}`,
  );

  const first = run.first!;
  const second = run.second!;
  // The segment counts, which are properties of the MESSAGES and not of the
  // model: the judge's own invariant is that judged + failed-closed + skipped
  // covers every segment it was handed, and core's segmentation gives these two
  // strings one segment and two. This is the pair a constant cannot satisfy.
  expect(first.segmentsJudged + first.failedClosed + first.segmentsSkipped).toBe(1);
  expect(second.segmentsJudged + second.failedClosed + second.segmentsSkipped).toBe(2);

  // The SUFFIX rule, which is the half a total would break most visibly: the
  // second message's rows are its own, so the judge's history is the two deltas
  // end to end. Reporting the whole history as the second delta would make
  // `bakeoff.ts` count the first message's calls again under the second item --
  // and on a 17-item corpus that over-count is quadratic in the p95 TTFT, the
  // decode rate and the truncation accounting, all of which read these rows.
  expect(first.calls.length).toBeGreaterThan(0);
  expect(second.calls.length).toBeGreaterThan(0);
  expect(run.totals.calls).toHaveLength(first.calls.length + second.calls.length);
  // Not merely the same count: the second delta must be the TAIL of the
  // history, so the rows themselves have to line up.
  expect(second.calls).toEqual(run.totals.calls.slice(first.calls.length));

  // And every counter, mechanically, so a slot this fixture happens to leave at
  // zero is still covered by the arithmetic rather than by nothing. Stated
  // narrowly on purpose: this says the two deltas partition the totals, which a
  // literal 0 satisfies for any counter neither message moved. The counters
  // this fixture does move are asserted above; `judge-delta.test.ts` is where
  // every slot is driven at a distinct nonzero value.
  const counters = Object.keys(run.totals).filter((k) => k !== "calls") as (keyof typeof run.totals)[];
  expect(counters).toHaveLength(14);
  for (const counter of counters) {
    expect(
      (first[counter] as number) + (second[counter] as number),
      `${counter}: the two deltas do not sum to the arm's total`,
    ).toBe(run.totals[counter] as number);
  }
});

test("the engine survives a deadline expiry", async ({ page }) => {
  test.setTimeout(LOAD_TIMEOUT_MS);
  await openHarness(page);
  test.skip(
    !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
    "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
  );

  // The wedge, end to end. Task 3 measured under Node with a fake that a
  // Promise.race abandons the promise but not the generation, and that the
  // interrupt which does stop it leaves an engine-wide flag set that the
  // non-streaming path never clears -- after which every later call returns
  // instantly, empty, with finish_reason "abort". `runWithDeadline` interrupts,
  // waits for the drain, and clears the flag. This is that property against the
  // real engine, which is where it was first observed.
  const run = await page.evaluate(
    async ([modelId, text]) => {
      await window.__sih!.useIr("semantic");
      await window.__sih!.loadTier2({ modelId });
      // 50 ms: far below the ~4.6 s a tier-2 call takes on this arm, so the
      // deadline fires during generation rather than racing it.
      const expired = await window.__sih!.detectWithBudget({
        text,
        provider: "claude",
        config: { tier0: false, tier1: false, tier2: true },
        callBudgetMs: 50,
      });
      const after = await window.__sih!.detect({
        text,
        provider: "claude",
        config: { tier0: false, tier1: false, tier2: true },
      });
      // The SECOND budget, and it is what makes this test able to tell
      // "detectWithBudget reads its caller's budget" from "detectWithBudget
      // hardcodes 50". 50 was the only value any test ever passed, so a judge
      // built with a literal 50 satisfied every assertion above -- and a budget
      // knob that does nothing is a knob every later measurement is taken
      // under. Far above the ~4.6 s a call costs on this arm, so this one must
      // NOT expire.
      const generous = await window.__sih!.detectWithBudget({
        text,
        provider: "claude",
        config: { tier0: false, tier1: false, tier2: true },
        callBudgetMs: 120_000,
      });
      return { expired, after, generous, status: window.__sih!.tier2Status() };
    },
    [MODEL, MESSAGE] as const,
  );

  console.log(
    `[tier2] expired ${JSON.stringify(run.expired.stats)}; after ${JSON.stringify(run.status?.lastDetect)}`,
  );

  // The premise, asserted rather than assumed. Without this the test could pass
  // against a budget that never expired -- which is how the plan's own version
  // read: it checked that the later `detect` returned non-null, and `detect`
  // returns an object on every path including a fully latched engine.
  expect(run.expired.stats.deadlineExpiries).toBe(1);
  expect(
    run.expired.result.degraded.some(
      (notice) => notice.tier === 2 && notice.reason === "call-budget-exhausted",
    ),
  ).toBe(true);

  // The conclusion. A latched engine answers the NEXT call instantly with an
  // empty body and finish_reason "abort", which `WebLlmJudge` counts as an
  // aborted response, files as `failed-closed`, and ends the run on -- so these
  // three assertions are the difference between an engine that survived the
  // interrupt and one that merely returned.
  const after = run.status?.lastDetect;
  expect(after).toBeDefined();
  expect(after!.calls.length).toBeGreaterThan(0);
  expect(after!.abortedResponses).toBe(0);
  expect(after!.failedClosed).toBe(0);
  expect(after!.segmentsJudged).toBe(1);

  // And the budget is the CALLER's. The same function, the same engine, the
  // same message: at 50 ms the call is interrupted and at 120,000 ms it is not,
  // so the only thing that can have changed the outcome is the number passed in.
  // Without this pair, `new WebLlmJudge(engine, {budgetMs: 50})` -- the value
  // hardcoded -- passes every other assertion in this file.
  expect(run.generous.stats.deadlineExpiries).toBe(0);
  expect(run.generous.stats.segmentsJudged).toBe(1);
  expect(
    run.generous.result.degraded.some((notice) => notice.reason === "call-budget-exhausted"),
  ).toBe(false);
});
