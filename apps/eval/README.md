# @sih/eval

Evaluation harness for the detection runtime. Playwright drives real Chrome
against a page that imports `@sih/core` unmodified.

The point is spec §2.2: the harness must execute the same detection code, on the
same runtime, as the extension. Numbers measured on `onnxruntime-node` would
describe software nobody runs, and WebGPU latency cannot be measured from Node
at all. **This app may never fork, stub, or re-implement any detection logic** —
it imports core and reaches it only through the page API (`window.__sih`).

## Prerequisite: install the browser

```bash
pnpm -C apps/eval exec playwright install chromium
```

Required once per machine, and in CI. Without it every spec fails at launch with
`Executable doesn't exist`. Root `pnpm test` runs this app, so a checkout that
has never run the line above will fail the whole repo's suite.

### The browser must be full Chromium, not the headless shell

`playwright.config.ts` pins `channel: "chromium"`. `playwright install chromium`
downloads both builds, so the command above is sufficient — but a CI image that
provisions only `chromium-headless-shell` (some slim images do, and
`playwright install --only-shell` does explicitly) **will fail to launch.**

This is not a preference. Measured on both builds: each is cross-origin isolated
and each exposes `navigator.gpu`, but the shell's `requestAdapter()` resolves to
`null`. onnxruntime-web reads a null adapter as "no WebGPU" and falls back to
wasm *silently*, so a WebGPU arm run on the shell reports wasm latency under
WebGPU's name. `test/smoke.spec.ts` asserts a non-null adapter so this fails
loudly rather than quietly producing wrong numbers.

## Running

```bash
pnpm -C apps/eval test        # playwright test
pnpm -C apps/eval typecheck   # tsc --noEmit
pnpm -C apps/eval dev         # vite dev server on :5178, for poking by hand
```

`playwright test` starts its own dev server and reuses one already listening on
5178 outside CI. Specs are `test/*.spec.ts`; `test/*.test.ts` is reserved for
vitest over the plain-Node parts of the harness.

## Two config decisions worth knowing before you edit

**COOP/COEP** (`vite.config.ts`). The `coop-coep` plugin sets
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, which is what makes
`SharedArrayBuffer` available for multi-threaded WASM. Without it onnxruntime-web
runs single-threaded and says nothing. The smoke spec asserts
`crossOriginIsolated`.

**No `server.fs.allow`** (`vite.config.ts`). It looks like an addition and is a
strict replacement: setting it drops the pnpm-workspace-root default and with it
`packages/core`. ES imports keep working anyway — Vite adds each rewritten module
to `safeModulePaths` — while a plain runtime `fetch()` for a file in the same
tree gets a 403. Model weights are fetched that way. Leave it unset.

## The IR fixture

`fixtures/minimal-ir.json` is a placeholder copied from
`packages/core/test/fixtures/minimal-ir.ts`, serialized so the page loads it the
way the extension will: `loadPolicyIr` over JSON text. It is not a baseline and
its numbers mean nothing on their own — it exists so the smoke test has a policy.
Replacing it with a compiled policy is a later task, and doing so also closes the
one gap the smoke spec documents: no tier-0 entity in this IR carries a provider
override, so `provider` cannot change any assertion here.
