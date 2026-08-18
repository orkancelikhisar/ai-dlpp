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
    // The page writes a #status div and surfaces pageerrors; both are only
    // legible after the fact through these. `openHarness` in smoke.spec.ts
    // promises the screenshot, so it has to actually be produced.
    screenshot: "only-on-failure",
    trace: "on-first-retry",
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
