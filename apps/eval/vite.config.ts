import { defineConfig } from "vite";

export default defineConfig({
  root: "src/page",
  // Deliberately no `server.fs.allow`. It reads like an addition and is a
  // strict REPLACEMENT: setting it resolves to exactly the listed dirs plus
  // vite/dist/client, which drops the pnpm workspace root that the default
  // detection supplies -- and with it packages/core, the code this page exists
  // to run. That still appears to work, because Vite's import analysis adds
  // each rewritten module to `safeModulePaths` as it goes, so ES imports
  // resolve while a plain runtime `fetch()` for the same tree gets a 403.
  // Task 11 fetches ONNX weights that way. The default covers everything.
  server: { port: 5178 },
  // Both are reached only through a DYNAMIC import inside main.ts (tier 1 is
  // loaded on demand, so a tier-0 page does not pay for a WASM runtime), which
  // means Vite's scanner does not see them at startup and discovers them
  // mid-run instead. MEASURED: without this, the first tier-1 spec fails with
  // "Execution context was destroyed, most likely because of a navigation",
  // and the dev-server log says why -- "new dependencies optimized:
  // @huggingface/transformers / optimized dependencies changed. reloading".
  // Vite full-reloads the page in the middle of the evaluate that triggered the
  // discovery. Listing them here makes the pre-bundle happen before the browser
  // is ever pointed at the page.
  optimizeDeps: { include: ["onnxruntime-web", "@huggingface/transformers"] },
  // Output escapes the source tree: `root` is src/page, so Vite's default
  // outDir would write dist/ INTO the page sources.
  build: { outDir: "../../dist", emptyOutDir: true },
  plugins: [
    {
      // Cross-origin isolation. onnxruntime-web needs SharedArrayBuffer for
      // multi-threaded WASM in Task 11; setting it now means the page the smoke
      // test exercises is the same page the model test measures later, rather
      // than a different one whose numbers do not transfer.
      name: "coop-coep",
      configureServer(server) {
        server.middlewares.use((_req, res, next) => {
          res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
          res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
          next();
        });
      },
    },
  ],
});
