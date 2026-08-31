import { DEFAULT_TIER2_CONFIG } from "@sih/tier2";
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

  const status = await page.evaluate(() => window.__sih!.tier2Status());
  expect(status?.load.servedModelId).toBe(MODEL);
  // The warm-up went through the engine and not through the judge, so no counter
  // moved and no `detect` has run: a spec asking "did tier 2 run during MY
  // detect" must not be answered with this page's own warm-up call.
  expect(status?.totals.calls).toHaveLength(0);
  expect(status?.totals.segmentsJudged).toBe(0);
  expect(status?.lastDetect).toBeUndefined();
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
      // The semantic fixture, and the whole test turns on it: the other two IRs
      // declare `semanticPredicates: []`, and `WebLlmJudge.judge` returns an
      // empty verdict on such an IR BEFORE it touches the engine. Against them
      // this test would pass with an empty findings array, a tier2Ms of a few
      // microseconds, and no model call at all.
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
      return { expired, after, status: window.__sih!.tier2Status() };
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
});
