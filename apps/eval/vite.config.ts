import { defineConfig } from "vite";

export default defineConfig({
  root: "src/page",
  server: {
    port: 5178,
    // The IR fixture lives at apps/eval/fixtures, outside the page root. Vite's
    // default allow-list is the pnpm workspace root, so this is belt-and-braces
    // -- it pins the one extra directory the page reads instead of relying on
    // workspace-root detection staying true as the repo grows.
    fs: { allow: [".", "../../fixtures"] },
  },
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
