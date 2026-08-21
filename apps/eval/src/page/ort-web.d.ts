/**
 * Types for `onnxruntime-web`, declared here because the package ships none
 * that TypeScript can reach.
 *
 * MEASURED, by importing the literal specifier and running `npm run -s
 * typecheck` from the repo root: tsc reports TS7016, "Could not find a
 * declaration file for module 'onnxruntime-web'.
 * .../dist/ort.bundle.min.mjs implicitly has an 'any' type. There are types at
 * .../onnxruntime-web/types.d.ts, but this result could not be resolved when
 * respecting package.json \"exports\"." -- the exports map publishes no `types`
 * condition, and this app resolves modules the way a bundler does.
 *
 * packages/tier1/src/session.ts hits the same wall and dodges it by importing
 * through a `string`-typed variable, which makes the specifier `any`. That is
 * not available here: MEASURED in this browser, a non-literal specifier is not
 * rewritten by Vite and reaches the page as a bare name, where it throws
 * "Failed to resolve module specifier 'onnxruntime-web'". So the page must use
 * a literal, and a literal needs a declaration.
 *
 * Only the three things the page touches are declared, and two of them are
 * borrowed from `OrtRuntime` rather than restated, so this file cannot drift
 * from the seam `createOrtSession` actually consumes.
 */
declare module "onnxruntime-web" {
  export const InferenceSession: import("@sih/tier1").OrtRuntime["InferenceSession"];
  export const Tensor: import("@sih/tier1").OrtRuntime["Tensor"];
  /**
   * `wasmPaths` is `unknown` rather than its real union: onnxruntime-common
   * types it as `string | { wasm?: string | URL; mjs?: string | URL }`, and the
   * page both writes the object form and reads it back in `assertLocalWasm` to
   * check nothing reassigned it. Narrowing at the read site is honest about the
   * fact that this declaration is the page's, not the package's.
   */
  export const env: { wasm: { wasmPaths?: unknown } };
}
