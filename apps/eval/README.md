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
pnpm -C apps/eval test        # vitest run && playwright test
pnpm -C apps/eval typecheck   # tsc --noEmit
pnpm -C apps/eval dev         # vite dev server on :5178, for poking by hand
```

Two runners, one script, vitest first: it needs no browser and finishes in under
a second, so a broken schema fails before Playwright spends time launching
Chrome. Root `pnpm -r test` calls this script, so both suites run there too.

Specs are `test/*.spec.ts` and vitest tests are `test/*.test.ts`, and each runner
is pinned to its own half — `testMatch` in `playwright.config.ts`, `include` in
`vitest.config.ts`. Both pins are load-bearing rather than tidiness. Measured:
with vitest's default `include`, vitest collects `test/smoke.spec.ts` and fails
it with Playwright's "did not expect test() to be called here", because
Playwright's fixtures do not exist under vitest.

`vitest.config.ts` is a separate file rather than a `test` key inside
`vite.config.ts` because that file sets `root: "src/page"` for the browser
harness. Measured: with only `vite.config.ts` present, vitest adopts that root
and exits 1 with "No test files found".

## The corpus

`corpora/fixtures/smoke.jsonl` (repo root, not this app) is a 12-item
hand-authored smoke corpus — 6 negatives with empty `gold`, 6 positives — used to
exercise the harness end to end. It covers a PAN in prose, an AWS key in a code
fence, a `key=value` line, a tier-1 client name in prose, a multi-line message,
and a message with emoji before the span, that last one so a UTF-16 offset bug
shows up here rather than in a run whose numbers someone believes.

`src/driver/corpus.ts` and `src/driver/record.ts` are the TS half of the
TypeScript/Python boundary in spec §2.2: this app emits JSONL and computes **no
metrics**; scoring is Python's job and the file is the only thing crossing
between them. `CorpusItemSchema` refuses any gold span whose offsets do not hold
the text it names, which is the labelling equivalent of the span-fidelity
invariant `normalizeFindings` enforces on findings.

One known asymmetry, left as-is deliberately: in `pos-secret-key-value` the gold
span is the secret value alone, while tier 0 reports the whole
`SESSION_TOKEN=<secret>` run — the entropy alphabet contains `=` and `_`, so the
kv line is one candidate run (`tier0.ts` documents that over-inclusion as
intended). The gold labels what must be protected, not what the current detector
happens to emit, so a scorer has to match spans by overlap rather than equality.

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
