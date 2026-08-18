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
