import { TIER2_MODELS } from "@sih/tier2";
import { expect, openHarness, test } from "./tier2-profile.js";

/**
 * Every pinned arm, loaded at the 8,192-token context window the bake-off
 * intends to run them all at.
 *
 * WHY THIS EXISTS. `context_window_size: 8192` was measured on Qwen3.5-2B only.
 * The other three arms each ship `overrides.context_window_size: 4096` in the
 * installed `prebuiltAppConfig` (verified: all four do) and had never been
 * loaded at 8,192. Discovering a refusal as three failed arms in the middle of
 * a bake-off wastes the whole run, and quietly running one arm at 4,096 while
 * the others run at 8,192 would make the bake-off measure context rather than
 * method.
 *
 * WHY LOADING IS THE CHECK. The residual risk is KV-cache VRAM, and the KV
 * cache is allocated AT LOAD: read from the 0.2.84 bundle,
 * `LLMChatPipeline`'s constructor builds the paged KV cache with
 * `max_total_sequence_length = contextWindowSize`, so an arm that cannot afford
 * an 8,192-token cache fails here rather than on some later long prompt.
 *
 * The `cs1k` every arm carries in its model_lib name is the PREFILL CHUNK size
 * and not the window, so nothing about 8,192 is baked into the libraries.
 * MEASURED, with the library's own load-time log at logLevel INFO: all four
 * arms print `Using prefillChunkSize: 1024` -- the same 1024 for every one --
 * while `Using contextWindowSize:` prints whatever this page asked for, 4096 or
 * 8192. The two numbers move independently.
 *
 * WHAT THIS DOES NOT CHECK, deliberately: that the window in force is 8,192
 * rather than the 4,096 the model record asks for. That needs a prompt longer
 * than 4,096 tokens, which costs a 13-45 s prefill per arm on this machine, and
 * `tier2.spec.ts` pays it once on the default arm together with the 4,096
 * control that shows the probe can fail. The merge that decides the window is
 * per-CALL library code (`{...mlc-chat-config.json, ...record.overrides,
 * ...chatOpts}`), not per-model, so repeating it on every arm on every run
 * re-measures the same library behaviour.
 *
 * These tests are independent rather than `mode: "serial"`: a per-arm gate
 * whose second arm is skipped because its first failed answers the question for
 * one model, which is the opposite of what it is for. They still run one at a
 * time, because `playwright.config.ts` pins `workers: 1`.
 */

/** A cold arm is a 1-2.5 GB download; a warm one is a cache read. */
const ARM_TIMEOUT_MS = 900_000;

for (const model of TIER2_MODELS) {
  test(`${model.id} loads and answers at an 8192-token context window`, async ({ page }) => {
    test.setTimeout(ARM_TIMEOUT_MS);
    await openHarness(page);
    test.skip(
      !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
      "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
    );

    const run = await page.evaluate(async (modelId: string) => {
      const before = await navigator.storage.estimate();
      const report = await window.__sih!.loadTier2({ modelId, contextWindowSize: 8192 });
      return { report, beforeBytes: before.usage ?? 0 };
    }, model.id);

    console.log(
      `[tier2 arm] ${model.id}: load ${run.report.loadMs.toFixed(0)}ms, warm-up ` +
        `${run.report.warmupMs.toFixed(0)}ms finish=${String(run.report.warmupFinishReason)}, ` +
        `origin storage ${(run.report.storageUsageBytes / 1e9).toFixed(2)} GB ` +
        `(+${((run.report.storageUsageBytes - run.beforeBytes) / 1e9).toFixed(2)} GB this load) ` +
        `of ${(run.report.storageQuotaBytes / 1e9).toFixed(2)} GB`,
    );

    expect(run.report.config.contextWindowSize).toBe(8192);
    // The engine ANSWERED under this arm's id. A load that allocated its KV
    // cache and then could not run would satisfy every other assertion here:
    // `loadTier2` gets this string from a real completion, so there is no
    // version of this line that passes without one.
    expect(run.report.servedModelId).toBe(model.id);
    // Never "abort" -- an engine that starts latched answers every later call
    // instantly and emptily -- and never "length", which at this arm's
    // `maxTokens` would mean the model could not close a one-key JSON object.
    expect(run.report.warmupFinishReason).toBe("stop");
  });
}
