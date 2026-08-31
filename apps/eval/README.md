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

**The tier-2 weights are not fetched by that script and are not in the repo at
all.** web-llm downloads them on first load into the browser profile — about
7.5 GB for the four pinned arms, roughly three minutes at 40 MB/s — and every
run after that reads them from the profile. See *Tier 2: the engine in the page*
below for where the profile lives and why it has to be a persistent one.

## Running

```bash
pnpm -C apps/eval test        # vitest run && playwright test
pnpm -C apps/eval typecheck   # tsc --noEmit
pnpm -C apps/eval dev         # vite dev server on :5178, for poking by hand
```

Two runners, one script, vitest first: it needs no browser and finishes in under
a second, so a broken schema fails before Playwright spends time launching
Chrome. Root `pnpm -r test` calls this script, so both suites run there too.

**Playwright runs with `workers: 1`**, for the same physical reason `runMatrix`
runs its arms one at a time: two ONNX graphs loading at once contend for one GPU
and one memory budget. The default is half the cores and it schedules *files* in
parallel, so `tier1.spec.ts` and `matrix.spec.ts` — which both load real models —
were being run against each other. Measured once `matrix.spec.ts` existed: about
one full run in five failed under the default, inside whichever model-loading
file lost the race. At one worker, four consecutive runs passed. The cost is
about nine seconds (25.7–26.4 s against 30.7–35.4 s). There are no `retries`
either, so a flaky suite cannot be papered over by re-running it.

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
it already happened. `runMatrix` does, on every record, before writing.

### `config` is core's, `tier1Config` is the rung

A record carries two configuration objects because one of them cannot hold the
other's fields. `config` is the `TierConfig` `detect` received — three booleans,
two model names, a backend. The tier-1 ladder has **six** dimensions (modelId,
precision, backend, threshold, maxWidth, labelForm) and three of them appear
nowhere in `TierConfig`, so two arms differing only in `threshold`, `maxWidth` or
`labelForm` used to emit records that were byte-identical in every field a scorer
can group by.

`tier1Config` is the **fully resolved** `Tier1Config` the tagger in the page was
constructed with, read off `loadTier1`'s report — never the partial object an arm
asked for. It is present **exactly when `config.tier1` is true**, and the schema
enforces both directions of that, plus that `backend`, `config.backend` and
`tier1Config.backend` are one answer and that `config.t1Model` names the same
rung as `tier1Config.modelId`. Each of those is a way a complete, valid,
scoreable file can describe a run that did not happen.

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

`ArmSpec.backend` is copied onto every record **and plumbed into the
`TierConfig` handed to `detect`**. The chain from that label to what executed was
broken in two places and both are now closed: the page measures which execution
provider really ran and rejects a `TierConfig.backend` that contradicts it (see
*`backend` is a measurement now* below), and `runArm` puts `spec.backend` into
the config it forwards — refusing an arm whose `config.backend` says something
else rather than silently picking one. So on a **tier-1** arm a record's
`backend` is backed by a measurement. On a **tier-0** arm it stays a bare label:
a regex pass runs on no backend at all, and the page's check is skipped when
`config.tier1` is false, so read it there as "which arm of the matrix this row
belongs to" and nothing more.

`runArm` also stamps `tier1Config` — the resolved `Tier1Config` its caller says
the tagger was built with. It cannot verify that, because it never loads a model;
`RunRecordSchema` couples the two fields and `runMatrix` validates every record
before writing one.

## The matrix: arms in, files out

`src/driver/main.ts` is the layer above `runArm`. `runMatrix(page, options)`
takes a list of arms and a corpus path, runs each arm over the whole corpus, and
writes **one JSONL file per arm** named `<runId>.<arm>.<backend>.jsonl`. It
returns the paths it wrote and, like everything else here, computes **no
metrics**.

```ts
await runMatrix(page, {
  runId: "2026-08-21a",
  outDir: "runs/",
  corpus: "corpora/fixtures/smoke.jsonl",
  provider: "claude",
  arms: [
    { arm: "t0", backend: "wasm", itemTimeoutMs: 10_000,
      config: { tier0: true, tier1: false, tier2: false } },
    { arm: "t0+t1-base", backend: "wasm", itemTimeoutMs: 120_000,
      config: { tier0: true, tier1: true, tier2: false },
      tier1Config: { modelId: "gliner-pii-base" } },
  ],
});
```

**Arms run sequentially, each from a freshly navigated page.** Sequential because
two models loading at once contend for one GPU and corrupt every latency number
in both files. Freshly navigated because a tagger left over from the previous arm
would be measured under the next arm's label — `runMatrix` owns page state, which
is exactly why `runArm` refuses to navigate on its own and throws on an
unprepared page.

**`tier1Config` on an arm is a `Partial<Tier1Config>`** — vary `modelId`,
`threshold`, `maxWidth` or `labelForm`. `backend` comes from the arm and naming a
different one inside is refused rather than reconciled. `resolveTier1Config`
fills in and validates the rest, in Node, before anything loads.

A note on which axes belong in a matrix at all: **`labelForm` should be measured
on one model and one policy, not crossed into the ladder.** It asks whether label
conditioning helps, which is a different question from where a rung sits on the
accuracy-versus-latency curve, and crossing it multiplies every arm for an answer
that does not vary by rung. Likewise a tier-0 arm's `backend` is a label and
nothing more: no part of a regex pass runs on a backend, so `{t0} x {wasm,
webgpu}` is the same measurement twice under two names. It is not refused —
`config.tier1` is `false` on both, so the records say so — but it buys nothing.

### Everything `runMatrix` refuses

Almost all of it is one failure mode: **a run that looks complete and measured
nothing.** An arm whose model never loaded, an arm that silently ran tier 0 under
a tier-1 label, an arm whose every row is an error, two arms overwriting each
other's file — each of those ends as a directory of well-formed JSONL that Plan 8
would score without complaint. So:

*Before the browser is touched at all* (a ten-arm matrix is hours of GPU time;
learning on the last arm that its name collides wastes all of it):

- **A WebGPU arm on any rung except `gliner-pii-base`** — see the table below.
  Including an arm that names no rung, since the default resolves to
  `gliner-pii-edge`, which is one of the wrong ones.
- **Tier-1 settings on an arm whose `config.tier1` is false.** Naming a rung and
  forgetting to switch tier 1 on produces a complete, valid, tier-0 file under a
  tier-1 name.
- **A `tier1Config.backend` contradicting the arm's `backend`**, and a
  `config.t1Model` naming a different rung than the tier-1 settings resolve to.
- **Two arms that would write the same file** — same `runId`, `arm` and
  `backend` — which would leave one silently overwritten while the matrix
  reports both.
- **A file an earlier run already wrote.** Re-running under the same `runId`
  replaces a measurement; use a new one. `writeFileSync` also uses the exclusive
  flag, so a file that appears mid-run is refused too.
- **A `runId` or arm name that is not filename-safe**, since both halves reach
  the filesystem verbatim.
- **An empty corpus, or a matrix with no arms.**

*After an arm has run, before its file is written:*

- **Every item errored.** `error` on a record exists so an arm that crashes on
  5% of a corpus is distinguishable from one that scores 0 on it; at 100% the
  file is a complete, schema-valid transcript of nothing having been measured.
  No file is written.
- **A tier-1 arm whose tagger ran no inference.** Not judged from `findings` (a
  span tagger may return none) or from `tier1Ms` (core sets it around the tagger
  call whether or not there was anything to tag) but from
  `tier1Status().totals.inferences`, compared against its value straight after
  the warm-up.
- **Any record `RunRecordSchema` refuses.** `runArm` does not validate its own
  output — the README says a writer must, and this is the writer.
- **A dead browser**, propagated out of `runArm`, which discards the records it
  had rather than returning a short arm as a complete one.

The WebGPU refusal is worth one more line because it is the only guard here
defending against a defect in someone else's code. `BACKEND_AGREEMENT` in
`src/driver/main.ts` is the table, `test/tier1.spec.ts` imports it and re-measures
it end to end on every suite run, and `runMatrix` checks it twice per tier-1 arm:
once on the config resolved in Node, and once on the config the page reports the
tagger was actually built with.

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

The disagreement is a **collapse**, not drift: `gliner-pii-base-uint8` logits
span `[-30.46, +2.05]` on wasm and `[-0.36, -0.16]` on webgpu, whose sigmoid is
≈0.46 for everything — which is exactly what the end-to-end findings show, every
word scoring 0.44–0.47 in monotone order. That scale, not a re-run, is what the
three **wrong** verdicts rest on: they are three to four orders of magnitude
above anything WebGPU varies by on its own (below).

**Nothing reports this.** Session creation succeeds, `run` resolves, the logits
are finite and correctly shaped. `gliner-pii-base` agreeing under the same page
code and readback path is the control that says the harness is not the cause.

That `0.000` is Task 11's, over raw logits on a single short message, and
**"exact" is not "bit for bit."** Measured over the whole 13-item smoke corpus at
threshold 0.02, comparing a real wasm arm against a real webgpu arm: every span
boundary and every entityType is identical on all 13 items (47 tier-1 findings
per arm), while *confidences* differ by ~1e-7 typically and by up to **2.6e-4**
on the two multi-line items — the longest inputs in the corpus.

**WebGPU is not bit-reproducible against itself either.** An earlier version of
this section said the table was "bit-identical across two independent full
re-runs, so it is deterministic rather than numerical noise". Measured: six
consecutive `detect` calls on the *same* text, in one page, on `gliner-pii-base`
at threshold 0.02 —

| provider | 153-char message | 2,148-char message |
|---|---|---|
| wasm | all six bit-identical | all six bit-identical |
| webgpu | every pass differs from the last, max &#124;Δ&#124; 1.7e-4 | first three differ, then settles, max &#124;Δ&#124; 4.9e-3 |

Spans, labels and ordering were identical in all twelve passes. So the `0.000`
was read off one short message where the effect is smallest, and if
`maxAbsLogitDiff` ever becomes an assertion rather than documentation it needs a
tolerance — nothing under ~5e-3 would hold at that input length, and the spread
grows with sequence length. It also puts the wasm-vs-webgpu gap on this rung in
proportion: it is no larger than the provider's disagreement with itself.
Nothing that is *scored* moved, which is why the rung stays usable.
`test/tier1.spec.ts` pins the table above, so a future onnxruntime-web that fixes
(or breaks) a rung fails the suite instead of quietly changing what a number
means.

**`runMatrix` refuses a WebGPU arm on any rung except `gliner-pii-base`**, and
the table above is the constant it reads — `BACKEND_AGREEMENT`, exported from
`src/driver/main.ts` and imported by `test/tier1.spec.ts`, so the guard and the
measurement that justifies it cannot drift apart. The refusal happens before the
browser is touched, and again on the config the page reports the tagger was
actually built with. That is also why the smoke arms use the 665 MB fp32 rung
rather than the 46 MB uint8 one: it is the only rung on which a wasm arm and a
webgpu arm measure the same model. The size cost is real but smaller than it
was recorded as: three cold loads each, median [min–max], `1.65 s [1.25–1.76]`
on wasm and `2.20 s [1.58–2.22]` on webgpu, against `0.26 s [0.25–0.32]` for
`gliner-pii-edge-uint8`. (A previous "2.0–2.7 s" here does not reproduce.)

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

Latency for one **16-word** prose segment, `tier1Ms` in ms, median [min–max].
Browser cells are 3 cold loads x 5 detects; the node column is 15 detects after
one warm-up. Re-measured, because the previous table here — which also called
the segment 12 words — no longer reproduces on any of its three columns:

| rung | onnxruntime-node CPU | browser wasm | browser webgpu |
|---|---|---|---|
| `gliner-pii-edge` | 8.0 [7.3–8.9] | 13.7 [9.2–22.1] | 11.7 [9.6–16.0] |
| `gliner-pii-edge-uint8` | 4.4 [3.3–5.1] | 11.5 [8.8–16.9] | 41.7 [38.8–47.3] |
| `gliner-pii-base` | 26.0 [24.8–28.3] | 41.0 [30.6–58.8] | 30.1 [20.1–63.8] |
| `gliner-pii-base-uint8` | 11.7 [9.9–13.6] | 34.1 [31.4–55.3] | 68.1 [64.6–75.2] |

The ranges are wide enough that a single sample of any cell is not a
measurement, which is how the earlier table came to be wrong. **Read the medians
as this machine on this day, not as a property of the ladder.**

WebGPU is **slower** than WASM on the two uint8 rungs and faster on the two fp32
ones at this message size — an earlier "three of four" no longer holds — and
three of the four webgpu columns are measuring wrong numbers anyway. Only
`gliner-pii-base` both benefits and agrees.

### Telling "ran and found nothing" from "never ran"

A span tagger legitimately returns nothing, so `findings` cannot be the evidence
that the model executed, and `tier1Ms` cannot either. `tier1Status().lastDetect`
carries the tagger's own counters as deltas over the one `detect` call —
`inferences` (actual `session.run` calls), `unmappableSpans` (decoded spans the
offset mapper refused, which are invisible in `findings` and would otherwise read
as a model that found less), and `gpuSubmits`. The page clears `lastDetect` at
the end of `loadTier1`, so the warm-up cannot supply it.

**And they reach the file.** `RunRecordSchema.tier1Stats` carries
`Tier1TaggerStats` whole — `inferences`, `droppedWords`, `truncatedWords`,
`overWideSpans`, `unmappableSpans`, `nonFiniteScores` — per record, present
exactly when `config.tier1` is set *and* `error` is null. Until it existed the
counters stopped at the page: `runArm` built each record from `findings` and
`timings` alone, so an item whose tail `maxLen` cut off emitted a row
byte-indistinguishable from one where the model read the whole message and found
nothing, which is the exact recall miss the counters were introduced to expose.
`error` is part of the coupling because `detect` throws whole — on a thrown item
the page's `lastDetect` still holds the *previous* item's delta, so absent is the
only honest answer and a row of zeros would be a false one. `gpuSubmits` is not
carried: it counts the page's GPU submissions, not the tagger's work.

### Telling "failed closed on 40% of messages" from "found nothing"

The tier-2 version of the same question, and it needs **two** channels because
neither one answers it alone.

`RunRecordSchema.tier2Stats` carries `JudgeStats` **whole** as a per-item delta,
present exactly when `config.tier2` is set *and* `error` is null — the same
coupling `tier1Stats` uses and for the same reason. Read `segmentsJudged` first:
`segmentsJudged + failedClosed + segmentsSkipped` is the number of segments the
judge was handed, and it is the denominator every findings-per-segment or recall
number needs. A **delta**, not the judge's totals: `WebLlmJudge.stats` is
cumulative across every message an arm has processed, so a record built from it
would inflate every row after the first with a fully green suite. The page's
`judgeDelta` subtracts the snapshot it took before the call; `runArm` reads
`tier2Status().lastDetect`.

`RunRecordSchema.degraded` carries `DetectionResult.degraded` whole, present
exactly when `error` is null. It was being **dropped**: `runArm` projected the
detection result field by field — `findings` and `timings` — so adding the array
to `DetectionResult` produced no type error and no change in the output.
Counters cannot stand in for it, and not as a matter of taste: three of the five
reason words have no counter anywhere. `absent` and `scope-unjudged` are the
orchestrator's own facts, no judge is in a position to count them, and the
budget-spent-before-start form of `budget-exhausted` is filed on a path that
makes **no engine call at all**. Read it per entry — `degraded.length > 0` is
not a cleanliness test, because an `absent` entry is filed for every tier the
`TierConfig` switched off, so every tier-0 row carries two of them.

**Per-call rows, not a per-message aggregate.** A message makes one engine call
per selected segment, plus the pinned recipe's one repair retry, so
`tier2Stats.calls` is one row per answered call in the order they were made:
`finishReason`, `promptTokens`, `completionTokens`, `ttftMs`. A single
`finishReason` for a message would be a fact about one call presented as a fact
about the message — one call stopping cleanly while another hits the token
ceiling means the judgement is *partial*, which no single value can say — and the
bake-off's p95 TTFT gate is a quantile over **calls**, which a per-message mean
cannot reconstruct. Sums remain available to a scorer; distributions would not
have been. A `ttftMs` the engine reported as non-finite is written as `null`
rather than as a `NaN`: measured, `JSON.stringify(NaN)` is `"null"` and zod's
`z.number()` rejects `NaN`, so copying one through produces a file this module's
own reader refuses.

**And the escalation threshold travels with them.** `config.uncertainBelow` is
required on a tier-2 record. `TierConfig` calls it an *experiment variable* the
bake-off varies per arm, so two tier-2 arms differing only in it would otherwise
emit records identical in every field a scorer can group by — the same disease as
the three tier-1 ladder dimensions `TierConfig` has no room for. `runArm`
resolves it (filling in `UNCERTAIN_BELOW` when the caller omits it) **into the
object `detect` receives**, so the number recorded is the number `escalate.ts`
compared against rather than a claim about what the default is. `runArm` does
not stamp one on a tier-0 or tier-1 arm — escalation never runs there, and a
threshold on such a row would name a knob that turned nothing — and the schema
does not require one there either. It does not *forbid* one, deliberately: a
caller may hand `detect` a threshold on an arm that never reaches escalation,
and the record's job is to state what `detect` received rather than to tidy it
away.

### `abandonedWorkInFlight`, and why a timeout does not abort the arm

`page.evaluate` accepts no timeout and offers no cancellation channel, so when
`runArm`'s per-item deadline expires the detection **keeps running in the
browser**. Every later item is then timed under contention with work belonging to
a different row — and only the expired row gets an `error`, so a latency
aggregate over `error === null` rows silently includes the contaminated ones.

Every row measured after an expiry therefore carries
`abandonedWorkInFlight: true`. The expired row itself does not: it already
carries `error`, and what the flag marks is a row timed under someone else's
work. Findings on a flagged row are still good — contention moves the clock, not
the spans — so this is a filter for `timings`, not a reason to drop the row from
a recall count.

Aborting the arm instead is defensible, and is what `runArm` already does for a
dead browser. It is rejected because an expiry says nothing about the items
already measured, and aborting discards all of them and — through `runMatrix` —
the rest of the matrix: one slow item at 1,400 of 1,500 would destroy hours of
GPU time that had already produced good rows. A run where this flag is true
anywhere is a deadline to fix, not a tolerable outcome.

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

## Tier 2: the engine in the page

`window.__sih.loadTier2(options)` loads one pinned tier-2 model through
`@sih/tier2`'s `createWebLlmEngine`, builds a `WebLlmJudge` over it, and makes
`detect` use that judge. `test/tier2.spec.ts` covers the lifecycle and
`test/tier2-arms.spec.ts` loads every pinned arm.

### Prerequisite: a browser profile, and about 7.5 GB of downloads

Unlike tier 1's, these weights are not fetched by a script into the repo. web-llm
downloads them from HuggingFace on first load and caches them in the **browser
profile**, keyed by the page's origin. Measured, loading all four pinned arms at
`http://localhost:5178` leaves **7.49 GB** in that origin's storage: 1.08 GB for
Qwen3.5-2B and roughly 2–2.3 GB for each of the other three.

The profile lives at `~/.cache/sih-eval/chrome-profile`
(`SIH_EVAL_PROFILE_DIR` overrides it) — deliberately **outside the repository**,
because a `git add -A` near a multi-gigabyte directory has burned this project
before. Cost, measured on this machine at ~40 MB/s: cold, the four arm loads took
2.1 s, 53.7 s, 56.7 s and 63.4 s; warm, the whole `tier2-arms.spec.ts` file runs
in about 17–20 s and the two tier-2 spec files add roughly 45 s to a suite run.

### `launchPersistentContext` is correctness, not convenience

`test/tier2-profile.ts` launches the tier-2 specs' browser itself, because
Playwright has no config option for a persistent profile. The reason is quota.
Measured against this same dev server: an ordinary Playwright context
(`browser.newContext()`, which the built-in `page` fixture builds on) reports a
`navigator.storage.estimate().quota` of **3,221 MB** standalone and 4,295 MB
from inside this suite, while the persistent profile reports **10,737 MB** empty
and 18,230 MB once it holds the arms. Four arms do not fit the first;
`QuotaExceededError` mid-download then looks exactly like a model that cannot
load. `test/tier2.spec.ts` asserts a quota above 8 GB, which is the line between
the two.

Two consequences of launching the context ourselves, both measured rather than
assumed: `screenshot` and `trace` from `use` **do** still apply (Playwright's
artifacts recorder attaches to any context the client creates), while `baseURL`
does not and is passed explicitly.

### What a load report claims, and what it does not

`Tier2LoadReport.config` is what was **requested**, in every field. 0.2.84
exposes no accessor for what an engine loaded — `MLCEngineInterface` declares
none and `MLCEngine.loadedModelIdToPipeline` is private — so there is no tier-2
equivalent of tier 1's `observedBackend` for the context window.

The one observed field is `servedModelId`, and `loadTier2` gets it by running one
throwaway completion before it returns. That is narrower than "these weights
ran": traced through the bundle, `ChatCompletion.model` is the id handed to
`CreateMLCEngine` laundered through a Map key. What it does buy is that **a load
cannot report success against an engine that did not answer** — the shape of the
Plan 4 failure where a dead browser produced a complete, schema-valid output
file — and that a `reload()` behind our back is followed rather than remembered.

The effective context window is measured separately, because the only channel
0.2.84 has for reporting it is its own refusal of an over-long prompt:
`window.__sih.probeContextWindow({words, budgetMs})` prefills filler text and
returns either the engine's `usage.prompt_tokens` or the
`ContextWindowSizeExceededError` whose message names the window it enforced.

### The 8192-token context window, measured on all four arms

`context_window_size: 8192` had been measured on Qwen3.5-2B only; the other three
arms each ship `overrides.context_window_size: 4096` in the installed
`prebuiltAppConfig` (all four do) and had never been loaded at 8192. All four
load and answer at 8192, and all four enforce it:

| arm | loads at 8192 | prompt accepted at 8192 | same prompt at 4096 |
|---|---|---|---|
| Qwen3.5-2B | yes | 5,809 tokens | refused, "context window size: 4096" |
| Ministral-3-3B | yes | 6,328 tokens | — |
| Qwen3-4B | yes | 5,809 tokens | — |
| Phi-4-mini | yes | 6,023 tokens | refused, "context window size: 4096" |

No arm has to run at 4096, so the bake-off does not have to report an asymmetry.

`tier2-arms.spec.ts` re-runs the **load** half on every arm on every suite run,
because the residual risk was KV-cache VRAM and the KV cache is allocated at load
(`LLMChatPipeline` builds the paged cache with
`max_total_sequence_length = contextWindowSize`). It does **not** re-run the
prompt half per arm: that costs a 13–45 s prefill each, and the merge deciding
the window (`{...mlc-chat-config.json, ...record.overrides, ...chatOpts}`) is
library code that does not vary by model. `tier2.spec.ts` pays it once on the
default arm, together with the 4096 control that shows the probe can fail.

### What the tier-2 specs cannot catch

When `webgpuAvailable()` is false every tier-2 test **skips**, which is a green
suite. Verified by mutation: forcing that function to return false leaves the two
spec files reporting 8 skipped and 1 passed. That is the intended behaviour on a
machine without a GPU — WebGPU absent means tier 2 is *absent*, not degraded —
but it means a bake-off driver must not read "the tier-2 specs passed" as "tier 2
ran".

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

## The IR fixtures

The page carries **three**, and `window.__sih.useIr(name)` switches between them,
returning the new `irHash` so a record's provenance moves with the artifact. A
named registry of `?raw` imports rather than a `loadIr(json)` taking arbitrary
text: `irHash` is reproducible with `shasum -a 256` on a file in this repo, and
injectable IR text would destroy that.

`fixtures/minimal-ir.json` (the default, `"minimal"`) is a placeholder copied
from `packages/core/test/fixtures/minimal-ir.ts`, serialized so the page loads it
the way the extension will: `loadPolicyIr` over JSON text. It is not a baseline
and its numbers mean nothing on their own — it exists so the smoke test has a
policy. Replacing it with a compiled policy is a later task, and doing so also
closes the one gap the smoke spec documents: no tier-0 entity in this IR carries
a provider override, so `provider` cannot change any assertion here.

`fixtures/multiclass-ir.json` (`"multiclass"`) declares **three** tier-1
entityTypes — `client-name`, `person-name`, `email-address`, in that order — and
exists for one reason. The minimal fixture has exactly one, so every browser
assertion ran at `classes = 1`, where `buildLabels` only ever assigns
classIndex 0 and both decoders' class stride multiplies by zero: a class-axis
bug reads the same cells a correct implementation reads, and the only label that
can come back is the only label there was. The multi-class coverage that existed
ran under `onnxruntime-node`, which §2.2 says is not the measured runtime. The
order is the teeth of the assertion — a decoder ignoring the class axis would
label everything `client-name`, and the spans `test/tier1.spec.ts` demands are
classIndex 1 and 2.

`fixtures/semantic-ir.json` (`"semantic"`) is the only one tier 2 can do
anything with. The other two declare `semanticPredicates: []`, and
`WebLlmJudge.judge` returns an empty verdict on such an IR *before it touches the
engine* — with tier 0 and tier 1 off nothing is uncertain either, so escalation
selects no segment, `timings.tier2Ms` is never set, and no model call happens. It
declares one predicate (`unannounced-deal`, scope `segment`) and the shadow
entityType `pred:unannounced-deal` a compiler would mint for it, which is what
makes tier 2 reachable at all.

Its `latencyBudgetMs` is **120,000** where the other two carry 5,000, and that
difference is deliberate. The orchestrator arms the message deadline from that
field; Plan 5 measured 4.6 s for one tier-2 call on the cheapest pinned arm, so
at 5,000 the deadline would fire during the first call of every tier-2 spec and
the suite would only ever exercise the abort path. It is a lifecycle fixture
sized for that job, **not** a claim that tier 2 fits a 5 s budget — Task 9
measured that on this corpus it does not.

## The smoke corpus says `minimal-fixture`, not `p-fin`

`corpora/fixtures/smoke.jsonl` labels every item `policy: "minimal-fixture"`.
Its gold was written against `fixtures/minimal-ir.json`, not against
`policies/p-fin.md`: two of its six positives use entityType ids (`aws-key`,
`generic-secret`) that exist only in that placeholder and that no p-fin
compilation produces, and one labels a secret `redact` where p-fin §4 blocks
credentials outright. It used to claim `p-fin` and did not have it. Relabelling
the *gold* to match p-fin is the other way to resolve that, and is deliberately
not done — that is Plan 7's job, and doing it here would be tuning data to fit a
claim.

## How big is a segment a tier-2 judge sees?

`src/driver/segments.ts` answers that, and Plan 5's tier-2 wall-clock budget is
sized from its answer rather than from a guess. `segmentSizeDistribution(items)`
segments every corpus item with core's `segmentText`, keeps only the segments
spec 4.1's escalation policy would select — `selectSegments` itself, imported
from `@sih/core`, never a second copy of the rule — and reports characters,
UTF-8 bytes, words and selected-segments-per-message.

Measured over `corpora/fixtures/smoke.jsonl` with `hasPredicates: true` and no
prior findings, which is the input a bake-off arm runs under:

| | value |
|---|---|
| items / segments produced / segments selected | 13 / 19 / 17 |
| characters per segment (UTF-16 code units) | p50 **62**, p95 153, max 153, min 22 |
| bytes per segment (UTF-8) | p50 **65**, p95 153, max 153, min 22 |
| words per segment | p50 **9**, p95 25, max 25, min 3 |
| selected segments per message | p50 **1**, p95 2, max 2 |

Supplying the real tier-0 findings (`runTier0` under `fixtures/minimal-ir.json`)
adds one segment — the 66-character AWS-key fence, escalated by an entropy
finding at 0.7 — for 18 selected and a maximum of **3** segments in one message.
Sizes move not at all.

Three things to know before quoting any of these:

- **The p95 is the maximum, and that is arithmetic rather than a finding.**
  Percentiles here are nearest-rank, so at n = 17 the 95th percentile is
  `ceil(0.95 × 17) = 17` — the last rank. Below n = 20 that is true of every
  sample. Read the p95 as "the largest of 17", never as a tail.
- **Sizes are never tokens.** None of the four pinned tier-2 models has a
  tokenizer cached here, and a made-up chars-per-token ratio would put a
  fabricated number under a budget.

  The one token claim these support needs no ratio, but it is the BYTE column
  and not the character one. This section used to say that characters bound
  tokens from above for any byte-level BPE vocabulary; that is false. A
  byte-level BPE tokenizes UTF-8 bytes, so its floor is one token per *byte*,
  while `String.length` counts UTF-16 code units — fewer than the bytes for
  every non-ASCII character. Measured against a real byte-level BPE cached in
  this repo (`packages/tier1/models/gliner-pii-edge/tokenizer.json`: `model.type`
  BPE, pre-tokenizer ByteLevel) through this app's own
  `@huggingface/transformers`: 20 `x` → 4 tokens, but 20 × U+1F389 → 60 tokens
  from 40 code units, and 20 CJK characters → 21 tokens from 20 code units. Two
  segments of this very corpus are already non-ASCII, which is why the two rows
  above differ at the median.

  On this corpus the difference changes nothing downstream — the largest segment
  is ASCII, so the ceiling is 153 either way — but carry the **byte** number
  forward, not the character one.
- **This is the same pipe-integrity fixture the section above warns about.** It
  sizes a budget for the bake-off that runs on *it*. It does not size one for
  real messages; re-run this against Plan 7's corpus before carrying a number
  forward.

`percentile(values, percent)` takes an INTEGER percent, and refusing a
fractional one is what lets the arithmetic carry no epsilon. Measured on Node 26
over every (n ≤ 20000, percent 1..100) pair — 2,000,000 of them —
`Math.ceil((percent * n) / 100)` agreed with exact BigInt arithmetic every time,
while in the same sweep the fraction spelling `Math.ceil(0.07 * n)` disagreed
with the true rank 153 times, starting at n = 100 where `0.07 * 100` is
`7.000000000000001`.
