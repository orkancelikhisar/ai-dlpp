import { defineConfig } from "vite";

/**
 * A vite config for `vite-node` ONLY, and separate from `vite.config.ts` for
 * the same reason `vitest.config.ts` is.
 *
 * `vite.config.ts` sets `root: "src/page"` for the browser harness. vite-node
 * adopts that root and then cannot resolve `src/driver/*` or the workspace
 * packages, so a driver script run under it fails on its first import. This
 * file restores the package root and nothing else; the dev server is untouched.
 */
export default defineConfig({});
