# AI-DLPP — AI Data Leak Prevention Plugin

A browser extension that detects confidential data in LLM prompts **entirely on the user's
machine**, conditioned on a data-leak policy written in plain English, and offers to
pseudonymize or remove it before the prompt is sent — rehydrating the real values in the
model's reply.

This is a **research prototype**. The deliverable is a defensible answer to one question:

> How well can policy-conditioned, fully-local models prevent confidential-data leakage in
> LLM prompts, and at what latency and hardware cost?

A labelled corpus and the evaluation numbers are first-class outputs, not an afterthought.

---

## The core idea: the policy is the program

Most PII tooling ships a fixed taxonomy. This one takes a **natural-language policy document**
— the kind a compliance team actually writes — and compiles it into a versioned, hash-stamped
intermediate representation. That IR then drives every layer at runtime:

- which patterns tier 0 matches,
- **which entity classes the tier-1 model is asked to look for**, injected at inference time,
- which semantic predicates tier 2 judges,
- and what happens to each finding, *per LLM provider*.

Changing the policy changes what the system looks for. **No retraining, no code change.**

```mermaid
flowchart LR
  DOC["policy.md<br/><i>plain English</i>"] -->|compile once,<br/>frontier model| IR["Policy IR<br/><i>versioned · hash-stamped</i>"]
  IR --> T0["tier 0<br/>regex · validators · entropy"]
  IR --> T1["tier 1<br/>GLiNER ONNX span tagger"]
  IR --> T2["tier 2<br/>local instruct LLM"]
  T0 & T1 & T2 --> M["merge<br/><i>overlap resolution,<br/>severity-first</i>"]
  M --> A{"resolve action<br/><i>per provider</i>"}
  A -->|block| X["refuse to send"]
  A -->|pseudonymize| V["vault<br/><i>real ⇄ surrogate</i>"]
  A -->|redact| R["strip"]
  V --> P["provider"]
  P -->|streamed reply| RH["rehydrate<br/><i>surrogate → real</i>"]
```

**No user prompt ever leaves the browser.** A frontier model is used exactly once — at
policy-compile time, which touches zero user data.

---

## Headline result so far

The tier-1 span tagger runs in a real browser on both WASM and WebGPU. Getting there produced
a finding that matters beyond this project:

![WebGPU returns wrong logits on 3 of 4 loadable rungs](docs/assets/webgpu-divergence.svg)

`onnxruntime-web`'s WebGPU execution provider **silently returns wrong logits on three of the
four model variants that load at all**. Not slower, not slightly off — a collapse, where every
word scores ≈0.46 and no error is raised. It survives review only because one rung agrees
*exactly* under identical page code, which is what proves the harness is not the cause.

Anyone benchmarking a transformer in the browser on WebGPU should check this before trusting
their numbers.

Two related findings from the same work:

![The precision ladder is four rungs, not six](docs/assets/model-ladder.svg)

![Tier-1 inference latency per rung and runtime](docs/assets/latency.svg)

---

## Why the harness runs a real browser

The evaluation harness is Playwright driving actual Chrome, loading the actual detection
package. That is a deliberate and expensive choice.

Numbers measured under `onnxruntime-node` would describe **software nobody runs**, and WebGPU
latency cannot be measured from Node at all. So `apps/eval` may never fork, stub, or
re-implement any detection logic — it reaches detection only through a single page API, and the
WebGPU finding above is exactly the class of defect that a Node-only harness would have
reported as a clean result.

The TypeScript/Python boundary is a **JSONL file**, never a shared library. The harness emits
records; all scoring happens separately in Python. A record carries the findings, the gold
spans, the timings, the resolved config, the IR hash, and the per-item loss counters — enough
that a coverage claim can be re-derived from the file alone.

---

## Repository layout

| Path | What lives there |
|---|---|
| `packages/core` | The detection runtime. Zero DOM, zero Node — lint- and test-enforced, because it runs in both. Policy IR, segmenter, tier-0 rules, merge, action resolution, pseudonymization vault, streaming rehydrator. |
| `packages/compiler` | Policy document → IR. Node-only CLI. The one component allowed to call a frontier model, with an anti-hallucination gate requiring every rule to quote its source clause verbatim. |
| `packages/tier1` | The GLiNER-class ONNX span tagger. Browser-only. Word splitter, per-word encoder, two decoders (the two model families express spans differently), and the span mapper. |
| `apps/eval` | Playwright harness: corpus in, JSONL out. Computes no metrics. |
| `policies/` | Three deliberately disagreeing policy documents (finance, healthcare, generic corporate) plus the provider manifest. |
| `corpora/` | Corpus fixtures and, later, the build pipeline. Never raw third-party data. |
| `docs/superpowers/` | The design spec and the per-plan implementation plans, including their deviation logs. |

---

## Status

Eight plans; four are done.

| | Plan | State |
|---|---|---|
| 1 | Core foundation | **done** |
| 2 | Vault, pseudonymization, rehydration | **done** — Plans 1&nbsp;+&nbsp;2 are `packages/core`, 276 tests between them |
| 3 | Policy compiler + policy suite | **done** offline — 133 tests. The live compile against a frontier model is deferred by choice. |
| 4 | Tier-1 span tagger + eval harness | **done** — 218 + 96 tests |
| 5 | Tier-2 judge + in-context baseline | not started |
| 6 | Extension (WXT, MV3) | not started |
| 7 | Corpus pipeline | not started |
| 8 | Evaluation and analysis | not started |

**723 tests**, typechecking clean across four packages.

### What is deliberately *not* claimed yet

- **No policy has been compiled by a real frontier model.** Plan 3 is verified against
  hand-authored fixtures, so the *compiler* is tested and the *extraction prompt* is not.
- **No accuracy numbers exist.** The corpus in this repo is a 13-item smoke fixture whose only
  job is to prove the pipe carries data end to end. It cannot produce meaningful tier-1
  accuracy in either direction, and says so.
- **The backend axis of the experiment is compromised** by the WebGPU defect above, and is
  reported as a finding rather than quietly dropped.

---

## Evaluation design

Three policies that **disagree on purpose** — client names are forbidden under the finance
policy and explicitly permitted under the healthcare one; salary data is regulated only by the
corporate one; provider-specific clauses appear only in finance. Those disagreements are what
the adaptivity metric grips onto: the same message, scored under three policies, must produce
three different actions.

Carriers are real conversations; only the confidential span is synthetic. **No prompt gets a
label until it is certified clear** under all three policies, so a "negative" is negative by
construction rather than by assumption.

Metrics: leak-prevention rate (headline) paired with over-blocking rate, span P/R/F1 per data
class, a policy-adaptivity delta, utility preservation on pseudonymized prompts, and
latency/footprint per arm and backend.

---

## Running it

```bash
pnpm install
pnpm -r test          # 723 tests; apps/eval drives real Chrome
pnpm -r typecheck
```

The tier-1 model weights (1.4 GB across six variants) are **not** in this repository. They are
pinned by content hash — every file each model needs, at an immutable upstream revision — and
fetched on demand:

```bash
pnpm -C packages/tier1 exec vite-node ../../scripts/fetch-models.ts
```

Tests that need weights skip cleanly when they are absent.

---

## Notes on how this was built

Every implementation task went through an independent spec-compliance review and a code-quality
review, with findings handed to adversarial verifiers who had to reproduce the evidence by
running it before a finding was accepted.

That process repeatedly overturned the plan rather than the code. The tokenizer approach in
Plan 4 was replaced wholesale after measurement showed the library had never shipped the API it
depended on; the decoder design was refuted by running the actual ONNX graphs, whose declared
axis names turned out to be wrong on both model families; a documented justification was found
to rest on a component the pinned checkpoints do not contain.

The recurring defect was rarely broken logic. It was **confident, plausible, wrong output** —
WASM latency labelled WebGPU, a config field recording intent rather than fact, gold values
leaking into the model's own prompt, a crashed browser filing a complete file of errored rows.
Those fail no test. Finding them is the reason the harness runs a real browser.
