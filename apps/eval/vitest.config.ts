import { defineConfig } from "vitest/config";

// Separate from vite.config.ts rather than a `test` key inside it, because that
// file sets `root: "src/page"` for the browser harness. MEASURED: with only
// vite.config.ts present, vitest reports `RUN v3.2.7 .../apps/eval/src/page` and
// exits 1 with "No test files found" -- it adopts the page root and never sees
// test/. A vitest.config.ts is loaded in preference to vite.config.ts, so
// declaring one here restores the package root without touching the dev server.
export default defineConfig({
  test: {
    // *.test.ts only, mirroring the *.spec.ts split playwright.config.ts pins
    // from the other side. MEASURED: with vitest's default include
    // (**/*.{test,spec}.?(c|m)[jt]s?(x)) this run collects test/smoke.spec.ts
    // and fails it with Playwright's "did not expect test() to be called here"
    // -- the Playwright runner's fixtures do not exist under vitest, so the
    // browser spec does not merely get skipped, it turns the suite red.
    include: ["test/**/*.test.ts"],
  },
});
