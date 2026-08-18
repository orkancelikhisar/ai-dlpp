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
