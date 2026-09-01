/**
 * The adapter limits web-llm refuses to start below.
 *
 * READ HERE from the installed 0.2.84 bundle's `detectGPUDevice`, which is the
 * only place that decides: the first two are demanded outright with NO fallback
 * (the source itself notes the WebGPU default for the buffer COUNT is 8, i.e.
 * the library asks for more than the spec guarantees), while `maxBufferSize`
 * and `maxStorageBufferBindingSize` are each requested at 1<<30, fall back once
 * to the values below, and are refused under those.
 *
 * MEASURED on this machine through the same Chrome-for-Testing build the specs
 * use: maxComputeWorkgroupStorageSize 32768 and maxStorageBuffersPerShaderStage
 * 10 -- EXACTLY the minimums, with no headroom -- against maxBufferSize and
 * maxStorageBufferBindingSize of 4,294,967,292 each. So on this machine the two
 * that can fail are the two with no fallback.
 *
 * `test/page-webgpu-floor.test.ts` re-reads all four numbers out of the
 * installed bundle and compares them with this table, so a library bump that
 * moves one fails there rather than by skipping the whole tier-2 suite.
 */
export const WEBLLM_ADAPTER_FLOOR: Readonly<Record<string, number>> = Object.freeze({
  maxComputeWorkgroupStorageSize: 32 << 10,
  maxStorageBuffersPerShaderStage: 10,
  maxBufferSize: 1 << 28,
  maxStorageBufferBindingSize: 1 << 27,
});

/**
 * The floor comparison, over an adapter's reported limits.
 *
 * ## Why this is its own module rather than three lines inside `webgpuAvailable`
 *
 * Because it decides whether every tier-2 spec RUNS, and a decision that gates
 * a suite cannot be tested by that suite. `test.skip(!webgpuAvailable())` turns
 * any error in this comparison into a silent, green skip. The review that found
 * this measured both directions against the pre-extraction code -- raising
 * `maxComputeWorkgroupStorageSize` to `64 << 10` made every tier-2 test skip at
 * exit 0, and replacing the whole walk with `return true` was equally
 * green, because this machine clears the real floor either way -- and those are
 * its numbers rather than a run of mine. Neither direction is observable from
 * inside the browser suite.
 *
 * RE-MEASURED HERE on the suite as it stands, because the review's own count
 * ("all nine tier-2 tests") described a smaller suite and had already gone
 * stale. With this function forced to `return false`,
 * `playwright test tier2.spec.ts tier2-arms.spec.ts` reports 9 SKIPPED, 2
 * PASSED, exit 0. The two that still run are the two that never consult
 * `webgpuAvailable()`: the profile-quota test, and the test that asserts this
 * function's answer against the real adapter -- which under the mutation agrees
 * with the page that the adapter is unusable, and passes. A green run of a
 * suite that measured nothing is the whole argument for this module.
 *
 * Split out, both directions are one object literal each in a Node test: a
 * limits object sitting exactly ON the floor must be accepted, and one a single
 * unit below any of the four must be refused. That is the pair no machine can
 * provide on its own.
 *
 * ## What the answer means
 *
 * Not the same question as `backendAvailable("webgpu")`, and the difference is
 * why the page asks this one for tier 2. Both libraries are handed the same
 * adapter and treat a weak one oppositely: onnxruntime-web falls back to wasm
 * SILENTLY (Task 1 measured that), while web-llm THROWS at init -- so where
 * tier 1 degrades, tier 2 is ABSENT, which is the orchestrator's own word for a
 * tier that did not run.
 *
 * A limit the adapter does not report at all is treated as BELOW the floor.
 * `GPUSupportedLimits` is an interface with a fixed member list rather than a
 * dictionary, so an absent one means this browser has no such limit -- and
 * guessing that an unreported limit is large enough is the direction that lets
 * a load throw inside a spec that should have skipped.
 */
export function meetsWebLlmAdapterFloor(limits: Record<string, number | undefined>): boolean {
  return Object.entries(WEBLLM_ADAPTER_FLOOR).every(([limit, floor]) => {
    const value = limits[limit];
    return value !== undefined && value >= floor;
  });
}
