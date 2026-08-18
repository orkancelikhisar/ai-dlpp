import { expect, test, type Page } from "@playwright/test";

/**
 * WebGPU has no lib.dom typings (they live in @webgpu/types, which nothing in
 * this app needs yet). Naming the single method the environment check calls is
 * cheaper than taking the dependency for a two-line probe; Task 11 can add the
 * real types when it adds the real backend.
 */
type MaybeWebGpu = { gpu?: { requestAdapter(): Promise<unknown> } };

/**
 * Navigate and wait for the page to publish its API.
 *
 * The wait is what fails when anything in main.ts throws -- the IR is parsed at
 * module scope, so a PolicyLoadError leaves `__sih` unpublished -- and on its
 * own it fails as a bare timeout naming nothing. The real error exists only as
 * a `pageerror`, so collect those and re-throw with the message attached: a
 * malformed IR should say which rule or field is at fault, not spend 30 seconds
 * saying "timeout".
 */
async function openHarness(page: Page): Promise<void> {
  const pageErrors: Error[] = [];
  page.on("pageerror", (error) => pageErrors.push(error));
  await page.goto("/");
  try {
    await page.waitForFunction(() => window.__sih !== undefined, undefined, { timeout: 10_000 });
  } catch (cause) {
    if (pageErrors.length > 0) {
      throw new Error(
        `harness never became ready; the page threw: ${pageErrors.map((e) => e.message).join("; ")}`,
        { cause },
      );
    }
    throw cause;
  }
}

test("core detects a tier-0 entity inside real Chrome", async ({ page }) => {
  await openHarness(page);

  const result = await page.evaluate(async () => {
    return window.__sih!.detect({
      text: "My PAN is AFTPD1298Q, please help.",
      provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
    });
  });

  expect(result.findings).toHaveLength(1);
  const finding = result.findings[0]!;

  expect(finding.entityType).toBe("in-pan");
  // Offsets are absolute into the message and must survive the page -> driver
  // boundary intact. That boundary is Playwright's own protocol serialization,
  // NOT structuredClone: the two differ on `undefined` (structuredClone
  // preserves it, the protocol drops it), which is exactly the direction a
  // dropped offset would fail in. Asserted as literals rather than recomputed
  // from the input string -- deriving them here would re-implement in the
  // driver the one thing the page exists to prove core still does.
  expect(finding.start).toBe(10);
  expect(finding.end).toBe(20);
  expect(finding.text).toBe("AFTPD1298Q");

  // The three fields below are the ones a re-implementation cannot fake, and
  // they are why this test means anything. Everything above is satisfiable by a
  // ten-line regex in main.ts; none of these is:
  //
  //   source    -- the IR's rule id, so a match had to come from ir.rules
  //                rather than from a pattern written into the page.
  //   severity  -- NOT copied from the rule that matched. normalizeFindings
  //                re-derives it from ir.entityTypes, so this passing proves
  //                the entity table was consulted, not hardcoded.
  //   action    -- does not exist until resolveAction -> strictestAction ->
  //                winnerAction have all run, so it proves the resolution chain
  //                executed rather than a label being copied off a regex match.
  //
  // What `action` does NOT do, despite being added for it, is make `provider`
  // load-bearing. MEASURED: dropping `provider` from the detect call in main.ts
  // entirely still passes all three assertions below. This IR's only
  // providerOverrides entry is deepseek -> client-name and client-name is tier
  // 1, so no tier-0 finding can resolve differently per provider and the field
  // stays inert here however it is asserted. Closing that needs an override on
  // a tier-0 entity, i.e. a different IR -- left to whoever replaces this
  // placeholder with a compiled policy, because editing the copy in
  // apps/eval/fixtures would falsify main.ts's note that it is a faithful copy
  // of core's fixture.
  expect(finding.source).toBe("pan-rule");
  expect(finding.severity).toBe("high");
  expect(finding.action).toBe("block");

  // Deliberately NOT `> 0`, and the reason is measured rather than assumed.
  // Over 400 warm calls on this page 315 read exactly 0: tier 0 on a 34-char
  // message finishes inside one clock tick, and cross-origin isolation still
  // caps performance.now() at ~5us. The first call on a fresh page does read
  // ~0.15ms (25/25 samples, 0.145-0.200) purely because it is un-JITted, so
  // `> 0` would pass today and turn flaky the moment any task adds a warm-up
  // call before this one. Per types.ts a 0 also genuinely means "tier 0 did not
  // run", so timing cannot separate ran-from-skipped at this size either way --
  // `findings` having exactly one entry is what proves tier 0 ran.
  expect(Number.isFinite(result.timings.tier0Ms)).toBe(true);
  expect(result.timings.tier0Ms).toBeGreaterThanOrEqual(0);
  // The IR's own latencyBudgetMs. Catches a garbage clock or a timer reported
  // in the wrong unit without pretending to be a performance assertion.
  expect(result.timings.tier0Ms).toBeLessThan(5000);
});

/**
 * Guards the two configuration decisions every Task 11 number depends on. Both
 * are invisible to the detection test above: delete the coop-coep plugin from
 * vite.config.ts, or the `channel: "chromium"` line from playwright.config.ts,
 * and that test still passes while the latency figures it exists to enable
 * quietly become measurements of different software.
 */
test("the page is the cross-origin isolated, GPU-backed one Task 11 measures", async ({ page }) => {
  await openHarness(page);

  const env = await page.evaluate(async () => {
    const { gpu } = navigator as Navigator & MaybeWebGpu;
    return {
      crossOriginIsolated: globalThis.crossOriginIsolated,
      hasSharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
      hasWebGpuApi: gpu !== undefined,
      hasAdapter: gpu === undefined ? false : (await gpu.requestAdapter()) !== null,
    };
  });

  // COOP/COEP from vite.config.ts. onnxruntime-web needs SharedArrayBuffer for
  // multi-threaded WASM; without isolation it silently runs single-threaded.
  expect(env.crossOriginIsolated).toBe(true);
  expect(env.hasSharedArrayBuffer).toBe(true);

  // `channel: "chromium"` from playwright.config.ts. The API check is NOT
  // redundant with the adapter check -- Playwright's default headless shell
  // exposes navigator.gpu and returns null from requestAdapter(), so the API
  // alone would pass on the browser this assertion exists to reject.
  expect(env.hasWebGpuApi).toBe(true);
  expect(env.hasAdapter).toBe(true);
});
