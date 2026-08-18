import { defineConfig } from "@playwright/test";

export default defineConfig({
  // *.spec.ts only. Sibling tasks add vitest *.test.ts under the same tree for
  // the parts of the harness that are plain Node code, and Playwright's default
  // glob would otherwise pick those up and run them in a browser worker.
  testDir: "test",
  testMatch: "**/*.spec.ts",
  use: { baseURL: "http://localhost:5178" },
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
