import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { WEBLLM_ADAPTER_FLOOR, meetsWebLlmAdapterFloor } from "../src/page/webgpu-floor.js";

/**
 * The check that decides whether the tier-2 browser suite RUNS.
 *
 * ## Why this cannot be a browser test
 *
 * Every tier-2 spec opens with `test.skip(!webgpuAvailable())`, so an error in
 * this comparison is a silent green skip rather than a failure. The review that
 * found this measured both mutations against the pre-extraction code -- raising
 * `maxComputeWorkgroupStorageSize` to `64 << 10` made all nine tier-2 tests
 * skip at exit 0, and replacing the whole floor walk with `return true` was
 * equally green, because this machine clears the real floor either way, so no
 * machine can exercise both directions. Re-run here: each of them now fails in
 * this file.
 *
 * The two halves below are what no machine provides: a limits object sitting
 * exactly ON the floor, and one a single unit below each of the four. And the
 * floor's own numbers are re-read from the INSTALLED bundle rather than
 * restated, so this file's expectation comes from the library rather than from
 * the table it is checking.
 */

const require = createRequire(import.meta.url);

/**
 * The four numbers `detectGPUDevice` demands, read out of the shipped 0.2.84
 * bundle.
 *
 * Regexes over `lib/index.js` rather than an import, because these are local
 * `const`s inside an async function and nothing exports them -- reading the
 * source is the only channel. Each pattern is anchored on the declaration and
 * on nothing else, and every one is asserted to have matched, so a bundle whose
 * shape changed fails here loudly instead of silently matching nothing.
 *
 * `maxBufferSize` and `maxStorageBufferBindingSize` are the BACKUP values (the
 * library requests 1<<30 for each, falls back once, and refuses below the
 * backup), which is why the pattern names `backupRequired...` for those two and
 * `required...` for the two with no fallback.
 */
function floorFromInstalledBundle(): Record<string, number> {
  const entry = require.resolve("@mlc-ai/web-llm");
  const source = readFileSync(join(dirname(entry), "index.js"), "utf8");
  const patterns: Readonly<Record<string, RegExp>> = {
    maxComputeWorkgroupStorageSize: /const requiredMaxComputeWorkgroupStorageSize = (\d+) << (\d+);/,
    maxStorageBuffersPerShaderStage: /const requiredMaxStorageBuffersPerShaderStage = (\d+);/,
    maxBufferSize: /const backupRequiredMaxBufferSize = (\d+) << (\d+);/,
    maxStorageBufferBindingSize: /const backupRequiredMaxStorageBufferBindingSize = (\d+) << (\d+);/,
  };
  const read: Record<string, number> = {};
  for (const [limit, pattern] of Object.entries(patterns)) {
    const match = pattern.exec(source);
    if (match === null) {
      throw new Error(
        `could not find web-llm's ${limit} requirement in ${entry}: the bundle's shape changed, ` +
          `so this test can no longer read the floor the library actually enforces`,
      );
    }
    read[limit] =
      match[2] === undefined ? Number(match[1]) : Number(match[1]) << Number(match[2]);
  }
  return read;
}

/** An adapter that clears the floor exactly, with no headroom anywhere. */
const AT_FLOOR: Record<string, number> = { ...WEBLLM_ADAPTER_FLOOR };

describe("web-llm's adapter floor", () => {
  it("is the floor the installed bundle enforces, number for number", () => {
    // Not a restatement of the table under test: the right-hand side is parsed
    // out of node_modules. A library bump that raises one of these fails HERE,
    // where it reads as "the floor moved", rather than in the browser suite,
    // where it would read as nine skipped tests and an exit code of 0.
    expect({ ...WEBLLM_ADAPTER_FLOOR }).toEqual(floorFromInstalledBundle());
    // The literal values too, so a bundle-reader that silently returned the
    // page's own table (a `require` resolving to the wrong package, say) could
    // not make the line above vacuous.
    expect({ ...WEBLLM_ADAPTER_FLOOR }).toEqual({
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 10,
      maxBufferSize: 268435456,
      maxStorageBufferBindingSize: 134217728,
    });
  });

  it("accepts an adapter sitting exactly on the floor", () => {
    // MEASURED on this machine's Chrome-for-Testing build, two of the four sit
    // exactly here with no headroom -- maxComputeWorkgroupStorageSize 32768 and
    // maxStorageBuffersPerShaderStage 10 -- so `>=` versus `>` is not a
    // hypothetical distinction: `>` would report this machine as unable to run
    // tier 2 at all, and every tier-2 spec would skip.
    expect(meetsWebLlmAdapterFloor(AT_FLOOR)).toBe(true);
  });

  it("refuses an adapter one unit below any single limit", () => {
    // Four cases, not one: a walk that checked only the first limit, or that
    // short-circuited on the wrong one, passes a single-limit test. This is
    // also the direction no machine here can produce -- which is why the whole
    // comparison had to leave `main.ts` to be exercised at all.
    for (const limit of Object.keys(WEBLLM_ADAPTER_FLOOR)) {
      const weak = { ...AT_FLOOR, [limit]: WEBLLM_ADAPTER_FLOOR[limit]! - 1 };
      expect(meetsWebLlmAdapterFloor(weak), `${limit} one below the floor was accepted`).toBe(false);
    }
  });

  it("treats a limit the adapter does not report as below the floor", () => {
    // `GPUSupportedLimits` has a fixed member list, so an absent one means the
    // browser has no such limit. Reading that as "large enough" is the
    // direction that lets a load throw inside a spec that should have skipped,
    // which is the failure `webgpuAvailable` exists to prevent.
    for (const limit of Object.keys(WEBLLM_ADAPTER_FLOOR)) {
      const missing: Record<string, number | undefined> = { ...AT_FLOOR };
      delete missing[limit];
      expect(meetsWebLlmAdapterFloor(missing), `a missing ${limit} was accepted`).toBe(false);
    }
  });

  it("accepts an adapter with headroom everywhere", () => {
    // The control: without it every assertion above is satisfied by a function
    // that returns false unconditionally, which would skip the tier-2 suite on
    // every machine -- exactly the silent failure this file is here to catch.
    const roomy = Object.fromEntries(
      Object.entries(WEBLLM_ADAPTER_FLOOR).map(([limit, floor]) => [limit, floor * 2]),
    );
    expect(meetsWebLlmAdapterFloor(roomy)).toBe(true);
  });
});
