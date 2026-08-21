import { defineConfig } from "@playwright/test";

export default defineConfig({
  // *.spec.ts only. Sibling tasks add vitest *.test.ts under the same tree for
  // the parts of the harness that are plain Node code, and Playwright's default
  // glob would otherwise pick those up and run them in a browser worker.
  testDir: "test",
  testMatch: "**/*.spec.ts",
  // A stray committed `test.only` greens the suite by running one test and
  // skipping the rest, which on a harness whose whole job is producing numbers
  // is worse than a red build.
  forbidOnly: !!process.env["CI"],
  // ONE worker, for the same physical reason runMatrix runs its arms one at a
  // time: two ONNX graphs loading at once contend for one GPU and one memory
  // budget. Playwright's default is half the cores (5 on this 10-core machine)
  // and it schedules FILES in parallel, so tier1.spec.ts and matrix.spec.ts --
  // which both load real models -- were being run against each other.
  //
  // MEASURED once matrix.spec.ts existed: roughly 1 full run in 5 failed under
  // the default, always inside whichever model-loading file lost the race (seen
  // as `tier-1 runs the real graph in real Chrome on wasm`, which then skipped
  // the 9 tests after it because that file is `mode: "serial"`). At one worker,
  // four consecutive full runs passed. There is no `retries` key here on
  // purpose, so a flaky suite cannot be papered over by re-running it.
  //
  // It also makes the latencies tier1.spec.ts prints mean something: measured
  // while another worker is loading a 665 MB graph, they describe contention.
  //
  // The cost is about nine seconds of wall clock on this machine: three runs at
  // the default took 25.7, 26.4 and 25.7 s, four at one worker took 30.7, 34.0,
  // 35.4 and 35.2 s.
  workers: 1,
  use: {
    baseURL: "http://localhost:5178",
    // `retain-on-failure`, NOT `on-first-retry`. There is no `retries` key
    // here, so the default of 0 applies and a first retry never happens: run
    // against the same induced failure, on-first-retry writes screenshot and
    // error-context and no trace.zip, retain-on-failure writes the trace.
    // Retries stay absent on purpose -- silently re-running a measurement is
    // the wrong reflex for a harness whose output is numbers -- so the trace
    // has to be earned on the first failure instead. `openHarness` in
    // smoke.spec.ts promises the screenshot, so that half must land too.
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "pnpm vite",
    port: 5178,
    reuseExistingServer: !process.env["CI"],
  },
  projects: [
    {
      name: "chromium",
      // `channel: "chromium"` selects the full Chrome-for-Testing build instead
      // of Playwright's default headless shell, and it is load-bearing rather
      // than cosmetic. Measured on this machine, both builds are cross-origin
      // isolated and both expose `navigator.gpu`, but the shell's
      // `requestAdapter()` resolves to null -- there is no adapter behind it.
      // onnxruntime-web reads that as "no webgpu" and falls back to wasm
      // SILENTLY, so a Task 11 webgpu arm run on the shell would report wasm
      // latency under webgpu's name: numbers describing software nobody runs,
      // which is the substitution spec 2.2 exists to forbid.
      use: { browserName: "chromium", channel: "chromium" },
    },
  ],
});
