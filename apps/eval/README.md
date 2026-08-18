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
contain emoji. A Python reader must index via
`text.encode('utf-16-le')[2*start:2*end].decode('utf-16-le')`.

Every span also carries its own `text`, which is exactly what the offsets should
select. That is the recovery path and the cross-check: verify indexing on every
span read and abort on the first disagreement rather than scoring against
fiction. `test/record.test.ts` pins this with the emoji item, so removing the
astral characters from the corpus fails the suite rather than quietly disarming
the check.

### Reading the files: three things the schemas do not enforce

- **`policy` is a NAME, not a hash.** It says which policy *document* an item's
  gold was written against, because the same PAN is `block` under one policy and
  `allow` under another. The exact IR is pinned on the record instead
  (`policyHash`). Nothing records which IR an *item's labels* were written
  against, so a policy edited without relabelling its corpus fails silently —
  change the `policy` name whenever labels stop matching the prose.
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
