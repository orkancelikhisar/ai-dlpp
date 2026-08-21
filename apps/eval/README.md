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

## Prerequisite: fetch the model weights

```bash
pnpm -C packages/tier1 exec vite-node ../../scripts/fetch-models.ts
```

`test/tier1.spec.ts` runs the real ONNX graphs and **fails** without them —
deliberately, and with a message naming the script above rather than a 404. The
weights are ~1.5 GB, gitignored, and verified against `MODEL_MANIFEST`'s pinned
hashes by that script.

The other half of the same coverage, `packages/tier1/test/e2e.test.ts`, **skips**
instead. That asymmetry is intended: the vitest suites in `packages/*` must run
on a checkout that has never downloaded a model, while this app is the harness
whose entire job is running them.

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

`corpora/fixtures/smoke.jsonl` (repo root, not this app) is a 13-item
hand-authored smoke corpus — 7 negatives with empty `gold`, 6 positives — used to
exercise the harness end to end. It covers a PAN in prose, an AWS key in a code
fence, a `key=value` line, a tier-1 client name in prose, a multi-line message,
and a message with emoji before the span, that last one so a UTF-16 offset bug
shows up here rather than in a run whose numbers someone believes.

> **This corpus is a pipe-integrity check. It cannot produce meaningful tier-1
> accuracy numbers, in either direction.** Thirteen hand-written items is far
> too small for a rate to mean anything, and they were authored to exercise
> shapes — code fence, kv line, astral offsets — not to be representative of
> anything. Do not quote a precision, recall or F1 from this file. Building a
> corpus those numbers can come from is Plan 7's job; this one only proves the
> pipe carries spans without corrupting them.

Two known limits worth stating rather than discovering:

- **`action: "none"` ships untested by the corpus.** It is in the gold enum, and
  means "a span worth labelling that this policy deliberately permits" — the
  distinction between a false positive and a correct-but-allowed detection. No
  item exercises it, so a scorer's handling of that branch is unverified here.
  (`test/record.test.ts` does assert the schema accepts it on gold and rejects
  it on a finding.)
- **Tier-1 precision is barely measurable.** `neg-proper-nouns-not-clients` is
  the only negative containing proper nouns (a person, a city, a weekday, a
  product), so it is the only item that can catch a tagger which fires on every
  capitalized token. One item is enough to notice a grossly over-firing model
  and nothing more.

The tier-1 gold values are deliberately **not** the ones in
`fixtures/minimal-ir.json`, whose `client-name` examples are `["Globex"]`. A
model prompted from the IR would otherwise be handed the gold answer in its own
prompt, and its recall here would be contamination rather than detection. The
same applies to the AWS key: `AKIAIOSFODNN7EXAMPLE` is that IR's `aws-key`
example, so the corpus uses a different (still obviously fake) value. Tier 0
matches it by regex, so the example could not have inflated tier-0 recall, but
any tier that sees entityType examples in a prompt would have had the answer.

`src/driver/corpus.ts` and `src/driver/record.ts` are the TS half of the
TypeScript/Python boundary in spec §2.2: this app emits JSONL and computes **no
metrics**; scoring is Python's job and the file is the only thing crossing
between them. `CorpusItemSchema` refuses any gold span whose offsets do not hold
the text it names, which is the labelling equivalent of the span-fidelity
invariant `normalizeFindings` enforces on findings.

### Offsets are UTF-16 code units — Plan 8 must decode, not slice

Every `start`/`end` in both schemas is a JavaScript string index: a **UTF-16
code unit** offset, the unit `String.prototype.slice` takes. JS is the producing
runtime and core's invariant is `text === message.slice(start, end)`, so offsets
are UTF-16 from the moment a detector emits one. Do not "fix" this to code
points — Tasks 6 and 10 build tier-1 span fidelity on the same invariant.

Python's `str` is indexed by **code point**, so the two disagree by one unit per
astral character (emoji, most non-BMP scripts) appearing earlier in the message.
Measured on `pos-emoji-before-pan`, whose gold span is `[33,43)`:

```python
text[33:43]                                          # -> 'CPT1234H t'  WRONG
text.encode('utf-16-le')[66:86].decode('utf-16-le')  # -> 'ABCPT1234H'  correct
```

`len(text)` is 60 there while the message is 62 code units long. Across the
smoke corpus a reader that slices `str` directly gets **6 of 7 spans right and
silently corrupts the seventh** — it fails toward wrong numbers rather than
crashing, and Plan 7's real corpus is ShareChat conversations, which certainly
contain emoji. A Python reader must index via UTF-16, **passing
`surrogatepass`**:

```python
text.encode('utf-16-le', 'surrogatepass')[2*start:2*end] \
    .decode('utf-16-le', 'surrogatepass')
```

The error handler is not optional garnish. A `text` may contain a **lone
surrogate** — scraped chat truncated mid-emoji produces them, which is Plan 7's
exact input — and it survives the whole trip: `JSON.stringify` emits it escaped
as `\ud800`, valid JSON that `json.loads` parses happily. Measured, a bare
`text.encode('utf-16-le')` then raises
`UnicodeEncodeError: 'utf-16-le' codec can't encode character '\ud800'`. With
`surrogatepass` the same string encodes and slices normally.

Every span also carries its own `text`, which is exactly what the offsets should
select. That is the recovery path and the cross-check: verify indexing on every
span read and abort on the first disagreement rather than scoring against
fiction. `test/record.test.ts` pins this with the emoji item, so removing the
astral characters from the corpus fails the suite rather than quietly disarming
the check.

### A record is self-contained

Every run record carries `text`, the message its offsets index into, copied from
the corpus item. **Scoring needs no join back to the corpus**: a reader holds the
spans and the string to check them against, so the UTF-16 cross-check above is
runnable from one line of JSONL. `RunRecordSchema` states it — every
finding AND every gold span must satisfy `start < end`, `end <= len(text)` and
`text.slice(start, end) == span.text` — so a record whose spans were built
against a different message is refused by anything that validates the file.
(The one case this cannot catch: a swap onto a text that coincidentally holds
the same slice at the same offsets.)

**`runArm` does not validate its own output.** The schema is the contract, not a
gate on the producing path: `test/run.spec.ts` runs it over a real arm on every
suite run, but a writer that persists records must run it too rather than assume
it already happened.

Carrying the message costs roughly +23% at this corpus's ~93 B mean message and
+112% at a 500 B mean; the projected worst case for a full Plan 7/8 run
(1,500 items × 8 arms × 3 policies) is around 35 MB, for a file Python reads
once, sequentially.

`toJsonl` also escapes **U+2028 and U+2029**. `JSON.stringify` leaves both raw —
they are legal unescaped JSON string content — while Python's `str.splitlines()`
treats them as line terminators, so a record containing one would split into two
fragments that both fail `json.loads`. Splitting on `"\n"` or iterating the file
handle is safe either way, and now `splitlines()` is too.

### Reading the files: three things the schemas do not enforce

- **`policy` is a NAME, not a hash.** It says which policy *document* an item's
  gold was written against, because the same PAN is `block` under one policy and
  `allow` under another. The exact IR is pinned on the record instead
  (`irHash`, the sha256 of the IR JSON the page loaded), alongside the IR's own
  `policyHash` — see *A record carries two hashes* below. Nothing records which
  IR an *item's labels* were written against, so a policy edited without
  relabelling its corpus fails silently — change the `policy` name whenever
  labels stop matching the prose.
- **`entityType` is unvalidated here.** It is an id from the IR, the same
  namespace as `Finding.entityType`, but this module never loads an IR. A typo
  surfaces as an entity scoring 0 recall, not as a load error, so a scorer
  should report the gold entityTypes it saw and let a human spot an odd one.
- **Gold spans are not sorted and not disjoint.** A message can carry two labels
  at the same offsets, or nested ones. This is the opposite of
  `DetectionResult.findings`, which core guarantees sorted and pairwise
  disjoint, so a scorer must not walk the two arrays in lockstep and must sort
  gold itself if it needs order.

**Unknown keys are silently dropped** at every level — item, gold span, record.
That is zod's default `strip`, not a decision made here, and it means a Plan 7
corpus carrying extra provenance columns loses them without a word. Put
provenance under `meta`, which keeps whatever it is given.

One known asymmetry, left as-is deliberately: in `pos-secret-key-value` the gold
span is the secret value alone, while tier 0 reports the whole
`SESSION_TOKEN=<secret>` run — the entropy alphabet contains `=` and `_`, so the
kv line is one candidate run (`tier0.ts` documents that over-inclusion as
intended). The gold labels what must be protected, not what the current detector
happens to emit, so a scorer has to match spans by overlap rather than equality.

## The run loop

`src/driver/run.ts` turns a corpus into records. `runArm(page, spec)` sends every
item through `window.__sih.detect` and returns one `RunRecord` per item, in
corpus order. It computes **no metrics** — that half of the §2.2 boundary is
Plan 8's Python.

Behaviours worth knowing before calling it:

- **It does not navigate.** It asserts `window.__sih` exists and throws
  otherwise. The caller owns page state, because a later task loads a tier-1
  model into the page *before* calling this, and a `goto()` here would discard it
  and measure a tier-0 run under a tier-1 arm label — a full JSONL file nothing
  downstream could tell apart from a real one.
- **Items run one at a time**, in corpus order. Latency is a reported metric, so
  concurrent inference on one GPU would measure contention rather than the model;
  stable order makes two runs diffable line by line.
- **An item that throws is recorded, not dropped**, with `error` set and
  `findings` empty. An arm that dies on item 300 of 1500 must not report as a
  complete run of 299 items, which scores as a much better arm than it is.
  `detect` throws whole, so no timings are *recoverable* for a thrown item —
  read `error` before reading `tier0Ms`, where `0` means "nothing was recorded".
- **But a dead browser aborts the arm.** After any item fails, `runArm` re-probes
  the page once. A closed, crashed or navigated-away page cannot answer, and
  that is the difference between "this arm scored badly" and "this run died":
  without the probe, every remaining item throws, each becomes a schema-valid
  record with `error` set, and the arm returns a **full-length JSONL file that
  Plan 8 would score**. Confident, complete and wrong is the worst output this
  harness can produce, so it throws instead, naming the item it died on and how
  many records were discarded.
- **Each item has a deadline**, `ArmSpec.itemTimeoutMs`, required rather than
  defaulted because a tier-0 regex pass and a cold WebGPU model differ by orders
  of magnitude. A wedged item becomes an errored record instead of a hung run.
  Note the limit: `page.evaluate` takes no timeout and offers no cancellation, so
  the abandoned call keeps running in the browser — this bounds the driver's
  wait, not the page's work.
- **It records the `TierConfig` that ran**, not just the arm name. `arm` is a
  free string; without the config beside it an all-tiers-off run is
  byte-for-byte identical to a detector that legitimately found nothing.
- **Findings are projected field by field** onto the record rather than copied
  wholesale, so a producer that decorates its findings cannot ride extra keys
  into the JSONL. Nothing validates a record on the producing path.
- **It stamps two different hashes**, described in its own section below.

`ArmSpec.backend` is copied onto every record and **still half-unbacked**. The
chain from that label to what executed was broken in two places; one is now
fixed. The page measures which execution provider really ran and rejects a
`TierConfig.backend` that contradicts it (see *`backend` is a measurement now*
below). But `runArm` still never plumbs `spec.backend` into `spec.config`, so
unless the caller sets `config.backend` as well, nothing reconciles this label
with anything. Do not read a record's `backend` as evidence of what executed
until an arm sets both.

## Tier 1: the model in the page

`window.__sih` grows three methods when a tier-1 model is involved:

- `backendAvailable(backend)` — asks the browser, not a user agent. `"webgpu"`
  requires `navigator.gpu.requestAdapter()` to resolve **non-null**, because
  Playwright's headless shell exposes `navigator.gpu` and returns `null` from it.
- `loadTier1({ backend, modelId?, threshold?, ... })` — loads one rung, builds
  the tokenizer, constructs `GlinerSpanTagger`, runs one warm-up inference, and
  returns a report. Takes a `Partial<Tier1Config>`, so an arm can vary threshold,
  max width and label form; `resolveTier1Config` fills in and validates the rest,
  and the **fully resolved config** comes back on the report.
- `tier1Status()` — the load report, the tagger's cumulative counters, and the
  per-call deltas from the most recent `detect`.

The ONNX file is reached through `import.meta.glob(..., { query: "?url" })`, which
resolves to a `/@fs/...` URL that Vite's **default** `server.fs.allow` already
covers. Measured: a plain `fetch()` of one answers 206. Weights absent (the
directory is gitignored) makes that glob empty, and `loadTier1` then throws a
message naming `scripts/fetch-models.ts` rather than letting onnxruntime fetch a
404 body and fail on a magic word.

Two environment facts, both measured, both invisible until they bite:

- **`env.wasm.wasmPaths` must be set to a local URL.** Vite's dependency
  pre-bundling moves onnxruntime-web to `/node_modules/.vite/deps/`, and ORT
  resolves `ort-wasm-simd-threaded.jsep.wasm` relative to its own module URL, so
  the binary 404s and session creation dies on
  `CompileError: expected magic word 00 61 73 6d, found 3c 21 64 6f` (`<!do` —
  the dev server's index.html fallback). Worse, `@huggingface/transformers`
  shares this exact ORT instance and assigns
  `wasmPaths = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@<v>/dist/"`
  when it finds the field unset — a **different ORT build, fetched over the
  network**, from an origin this page's COEP rejects. `main.ts` sets it and then
  re-checks it.
- **`optimizeDeps.include`** lists both packages in `vite.config.ts`. They are
  reached only through a dynamic import, so Vite discovers them mid-run and
  full-reloads the page in the middle of the `evaluate` that triggered the
  discovery — which surfaces as `Execution context was destroyed`.

### `backend` is a measurement now, not a label

`loadTier1` counts **`GPUQueue.submit` calls around a real warm-up inference**
and reports `observedBackend` from that. If it disagrees with the backend that
was asked for, the load throws and no arm runs. Measured on this page:
`executionProviders: ["wasm"]` submits **0** during a run, `["webgpu"]` submits
40–193 depending on the rung.

The two obvious alternatives do not work. `env.webgpu.device` is documented as a
getter that **creates** a device when read before the first webgpu session, so
reading it to ask "did webgpu initialize" makes the answer yes.
`env.webgpu.adapter` is a real signal but sticky — once any webgpu session has
existed it stays set for the page's lifetime.

`window.__sih.detect` additionally rejects a `TierConfig.backend` that
contradicts the loaded model, which is the one place both values are known.

### ⚠ WebGPU returns WRONG NUMBERS on three of the four loadable rungs

Measured with identical feeds, one fresh page per rung per provider, comparing
raw `logits` element by element:

| rung | spanMode | max &#124;wasm − webgpu&#124; | verdict |
|---|---|---|---|
| `gliner-pii-edge` | token_level | 8.352 | **wrong** |
| `gliner-pii-edge-uint8` | token_level | 8.134 | **wrong** |
| `gliner-pii-base` | markerV0 | **0.000** | exact |
| `gliner-pii-base-uint8` | markerV0 | 30.154 | **wrong** |

Bit-identical across two independent full re-runs, so it is deterministic rather
than numerical noise. The disagreement is a **collapse**, not drift:
`gliner-pii-base-uint8` logits span `[-30.46, +2.05]` on wasm and `[-0.36, -0.16]`
on webgpu, whose sigmoid is ≈0.46 for everything — which is exactly what the
end-to-end findings show, every word scoring 0.44–0.47 in monotone order.

**Nothing reports this.** Session creation succeeds, `run` resolves, the logits
are finite and correctly shaped. `gliner-pii-base` agreeing to 0.000 under the
same page code and readback path is the control that says the harness is not the
cause. `test/tier1.spec.ts` pins the table above, so a future onnxruntime-web
that fixes (or breaks) a rung fails the suite instead of quietly changing what a
number means.

**Do not run a WebGPU arm on any rung except `gliner-pii-base` until this is
retested.** That is also why the smoke arms use the 665 MB fp32 rung rather than
the 46 MB uint8 one: it is the only rung on which a wasm arm and a webgpu arm
measure the same model. The size cost is 2.0–2.7 s of load in this browser.

### The ladder is four rungs, not six

Both `*-fp16` variants fail to create a session, in the **browser** and under
`onnxruntime-node` alike, with the same error:

```
Type Error: Type (tensor(float16)) of output arg (.../Cast_1_output_0)
of node (.../Cast_1) does not match expected type (tensor(float)).
```

Same failure on both runtimes means an upstream export defect, not a runtime
difference. `test/tier1.spec.ts` asserts they fail *loudly* — a rung that cannot
run must throw rather than fall back to one that can.

### Node and the browser agree, on WASM

`packages/tier1/test/e2e.test.ts` runs the assembled tagger under
`onnxruntime-node` (skipped unless the weights are on disk). Cross-checked
against the browser with the same IR, the same message and the same rungs:
identical spans on all four loadable rungs, and identical scores to 3 dp on both
fp32 rungs. Only `gliner-pii-edge-uint8` drifts, by ≤0.03 — two ORT builds'
quantized kernels, not a span difference.

Latency for one 12-word prose segment, `tier1Ms`:

| rung | onnxruntime-node CPU | browser wasm | browser webgpu |
|---|---|---|---|
| `gliner-pii-edge` | 10.9 | 20.1 | 66.9 |
| `gliner-pii-edge-uint8` | 3.4 | 20.7 | 50.3 |
| `gliner-pii-base` | 24.9 | 54.8 | 31.1 |
| `gliner-pii-base-uint8` | 10.0 | 70.1 | 108.1 |

WebGPU is **slower** than WASM on three of four rungs at this message size, and
three of those four webgpu columns are measuring wrong numbers anyway. Only
`gliner-pii-base` both benefits and agrees.

### Telling "ran and found nothing" from "never ran"

A span tagger legitimately returns nothing, so `findings` cannot be the evidence
that the model executed, and `tier1Ms` cannot either. `tier1Status().lastDetect`
carries the tagger's own counters as deltas over the one `detect` call —
`inferences` (actual `session.run` calls), `unmappableSpans` (decoded spans the
offset mapper refused, which are invisible in `findings` and would otherwise read
as a model that found less), and `gpuSubmits`. The page clears `lastDetect` at
the end of `loadTier1`, so the warm-up cannot supply it.

### A record carries two hashes

They answer different questions and neither replaces the other.

`irHash` is the **sha256 of the IR artifact the page loaded** — the bytes of the
JSON file, hashed in the page. It answers *which IR produced these numbers*.
`shasum -a 256 apps/eval/fixtures/minimal-ir.json` prints exactly what lands on
the record, so the provenance is checkable from outside the browser. The cost is
that a whitespace-only reformat changes it while the IR is semantically
identical — the safe direction, since it can call two identical IRs different but
never two different IRs the same. `runArm` checks the digest once, up front,
rather than letting `RunRecordSchema` reject every row after the arm has run.

`policyHash` is the IR's own field carried **verbatim** — the compiler's sha256
of the policy *document* (`packages/compiler/src/stages/emit.ts`). It answers
*which prose that IR was compiled from*, restoring the document → IR → numbers
chain that `irHash` alone cannot.

`policyHash` cannot stand in for `irHash`, and the reason is stronger than
compiler version drift: compilation is **model-driven**, so the same document
compiled twice by the same compiler can yield two different IRs carrying the same
`policyHash`. It identifies the input, never the artifact.

Only `irHash` is constrained to `/^[0-9a-f]{64}$/`. `policyHash` is any non-empty
string, exactly as strict as core's own `PolicyIrSchema`, because it is copied
from whatever IR the page loaded and a hand-written fixture legitimately carries
a placeholder — `fixtures/minimal-ir.json` says `"test-hash"`. Tightening it
would force that fixture to state the hash of a document that does not exist.

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
