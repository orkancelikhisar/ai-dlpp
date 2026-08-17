# AI Data Leak Prevention Plugin (AI-DLPP) — Design Spec

**Date:** 2026-08-13
**Status:** Approved by user (brainstorming phase complete)
**Target:** Research prototype

## 1. Goal and research framing

AI-DLPP is a browser extension that intercepts every prompt (and text-file upload) sent to LLM providers (ChatGPT, Claude, Gemini, DeepSeek), detects confidential information **locally**, and applies policy-driven actions — allow, pseudonymize, redact, or block — before anything leaves the browser. Pseudonymized values are transparently restored (rehydrated) in the provider's streamed response.

This is a **research prototype**. The deliverable is not only the working extension but a defensible answer to the research question:

> **How well can policy-conditioned, fully-local models prevent confidential-data leakage in LLM prompts, and at what latency/hardware cost?**

The core novel requirement is **policy-adaptivity**: the system is given a natural-language data-leak policy document and dynamically shapes its entire behavior from it — including **provider-specific clauses** (e.g. "customer data may not go to non-enterprise or foreign-hosted services"). The taxonomy of "confidential" is defined at runtime by the policy, not baked into model weights.

### Scope decisions (fixed)

| Dimension | Decision |
|---|---|
| Target bar | Research prototype: correctness of detection + measured evidence; polish secondary |
| Data classes | All four: structured identifiers, credentials/secrets, named entities, semantic/contextual prose |
| Cloud boundary | User prompts/files NEVER leave the browser. Cloud (frontier model) permitted only at **policy-compile time**, which touches zero user data |
| Hardware | Treated as experiment variable: accuracy-vs-latency curve across a model ladder on CPU/WASM and WebGPU |
| Providers | ChatGPT, Claude, Gemini, DeepSeek — with provider-aware policy resolution |
| Timeline / team | Months, solo |
| Files | Text formats only (.txt, .md, .csv, .json, code). PDF/docx parsing = future work |
| Not in v1 | Keystroke-live highlighting; enterprise management console; mobile browsers |

## 2. Architecture

### 2.1 Chosen approach: A (policy compiler + tiered runtime) with B as experimental baseline

Three approaches were considered:

- **A — Policy compiler + tiered runtime** (chosen): offline compilation of the policy document into a validated IR; runtime executes compiled rules + policy-conditioned models in cost tiers.
- **B — Policy in-context**: one local LLM reads policy chunks + prompt at every send. Built as a **baseline arm** (same `Detector` interface, ~2 days work) so "does compiling beat prompting?" becomes a measured result instead of an assumption.
- **C — Per-policy trained classifier**: rejected (no span output → can't pseudonymize; retrain on every policy edit).

### 2.2 Repository layout

The load-bearing constraint: **the eval harness must execute the exact same detection code, on the same runtime, as the extension.** Detection numbers measured on a different runtime (e.g. onnxruntime-node) would describe software nobody runs, and WebGPU latency cannot be measured from Node at all. Therefore the harness is Playwright driving real Chrome loading the real core.

```
sih/
├── packages/
│   ├── core/          # zero DOM, zero chrome.* — runs in Node AND browser (lint-enforced)
│   │   ├── policy/      # IR types, schema validator, resolver (provider-override merge)
│   │   ├── segment/     # text → segments (code-fence / prose / kv-pair aware)
│   │   ├── detect/      # tier0 rules · tier1 span tagger · tier2 judge · orchestrator · baselineB
│   │   └── pseudo/      # vault, surrogate minting, rehydrator
│   ├── adapters/      # per-provider request/response shape knowledge (pure functions)
│   └── compiler/      # policy doc → IR. Node-only CLI. May call frontier model. Self-tests.
├── apps/
│   ├── extension/     # WXT (MV3): MAIN-world interceptor, content UI, SW, offscreen doc, options
│   └── eval/          # Playwright driver: loads core in real Chrome, runs corpus, emits JSONL
├── analysis/          # Python: JSONL → metrics, CIs, plots. Touches only JSONL.
├── corpora/           # dataset build scripts + manifests (never raw third-party data)
└── policies/          # policy documents (markdown) + compiled IRs
```

TS/Python boundary is a **JSONL file**, never a shared library. `packages/core` importing `chrome.*` or `document` is a lint-enforced build error.

### 2.3 Data flow (send path)

```
user hits send
  → [MAIN-world fetch interceptor] adapter identifies provider, extracts {conversationId, userText, writeback}
  → [core.detect] policy IR resolved for THIS provider
      tier0 rules → tier1 span tagger → tier2 judge (escalation only)
  → findings[] {span, entityType, confidence, provenance → policy clause}
  → action resolution: allow | pseudonymize | redact | block
  → [pseudo.vault] real span ⇄ surrogate
  → rewritten body → provider
  → streamed response ← [rehydrator] surrogate → real, incrementally
```

## 3. Policy IR and compiler

### 3.1 Policy IR

One versioned JSON artifact, hash-stamped with its source document. The runtime **refuses to load** an IR whose schema version it doesn't know.

```jsonc
{
  "irVersion": "1",
  "policyHash": "sha256 of source doc",
  "entityTypes": [{
    "id": "client-name",
    "tier": 1,                        // which tier is responsible for catching it
    "nlDefinition": "Name of any current or prospective client organization…",
    "examples": ["…Globex renewal…"],
    "counterExamples": ["…our clients generally…"],
    "severity": "high"
  }],
  "rules": [{                          // tier-0 only
    "id": "pan-card", "entityType": "in-pan",
    "regex": "\\b[A-Z]{5}[0-9]{4}[A-Z]\\b",
    "validator": "pan-structure",      // named function from a fixed library — never generated code
    "contextBoost": ["PAN", "tax"]
  }],
  "semanticPredicates": [{             // tier-2 only; no span-shaped surface form
    "id": "unreleased-financials",
    "nlPredicate": "Discusses revenue, margins, or forecasts not yet public…",
    "scope": "segment"
  }],
  "actions": {
    "default":   { "client-name": "pseudonymize", "credential": "block" },
    "providerOverrides": { "deepseek": { "client-name": "redact" } }
  },
  "failMode": "closed",                // open | closed; per-provider overridable
  "latencyBudgetMs": 5000,
  "provenance": { "pan-card": { "clause": "§3.2", "quote": "…" } }
}
```

Design invariants:

- **Validators are named, not generated.** The compiler emits regexes (data); executable logic comes only from a fixed audited library in `core` (Luhn, Verhoeff, PAN structure, JWT shape, Shannon entropy). A cloud model authoring code that runs in the security layer is an injection surface; authoring data that a fixed engine interprets is not.
- **Every entityType is tier-assigned by the compiler** ("PAN" → 0, "client name" → 1, "unannounced financials" → 2). The assignment is itself testable via compiler self-test.
- **Action resolution** is a pure function `resolve(entityType, provider) → action`, deep-merging `providerOverrides` over `default`. Exhaustively unit-tested — a bug here is a silent leak.
- **Provenance is mandatory**: every rule/entity/predicate links to the policy clause and quote that produced it. Powers the review-sheet explanations and the adaptivity metric.

### 3.2 Compiler (`sih-compile policy.md --provider-manifest providers.json`)

Node CLI. The **only** place a cloud model is ever called; it never sees user data. The extension ships a compiled IR and cannot function without one.

Pipeline (each stage inspectable):

1. **Extract** — frontier model (Claude via API) reads the policy doc, emits candidate entityTypes/rules/predicates/actions, each with the supporting source quote. **Anything without a supporting quote is rejected** (anti-hallucination gate).
2. **Ground** — provider mentions ("Chinese-hosted services", "our enterprise Claude agreement") mapped to adapter IDs via the provider manifest.
3. **Validate** — JSON-schema check; every regex compiled and bounded-time-checked against catastrophic backtracking; every referenced validator exists in the library.
4. **Self-test** — frontier model generates ~20 positives + ~20 hard negatives per entityType; the **actual runtime** (imported from `core`) executes them; per-entity coverage report. Under-threshold entities are flagged for human review — compilation "succeeds with warnings," listing exactly what's weak. Self-test corpus is kept (with contamination controls vs. the eval corpus — §6.2, Splits & contamination).
5. **Emit** — IR + human-readable markdown compilation report an admin can audit.

## 4. Detection runtime

### 4.1 Tiers

| Tier | Engine | Cost | Responsibility |
|---|---|---|---|
| 0 | Compiled regex + named validators + context boost + entropy scanning (sliding-window Shannon on code/config segments; known-prefix secret rules: `sk-`, `AKIA`, `ghp_`, JWT triple-base64) | <1 ms, sync | Structured identifiers, credentials |
| 1 | Zero-shot span tagger (GLiNER-class, ONNX via onnxruntime-web, Web Worker; WASM/CPU baseline, WebGPU when available) — **labels injected at inference are exactly `entityTypes[].{id, nlDefinition}` from the IR**. This is the policy-adaptivity mechanism: no retraining, ever | ~20–80 ms | Named entities |
| 2 | Local instruct LLM via WebLLM (WebGPU only). Fires only on segments where tier 0/1 left uncertainty above threshold, or where the IR has `semanticPredicates` (those always require tier 2 on prose segments). Input: segment + relevant predicates + tier 0/1 findings. Output: strict JSON findings, schema-validated, one repair retry, then **fail-closed to "flag for user review"** — a model that won't emit valid JSON never silently passes text through | ~200 ms–2 s | Semantic/contextual prose |

**Orchestrator** (per segment): tier0 always → tier1 on prose+kv segments → tier2 iff (uncertain ∨ semanticPredicates present) ∧ `TierConfig.tier2`. Findings merged with span-overlap resolution (highest severity wins); confidence calibrated per tier.

`TierConfig = {tier0, tier1, tier2, t1Model, t2Model, backend: wasm|webgpu}` — **the ablation ladder is a config parameter, not a code branch.**

### 4.2 Model slate (researched 2026-08-13)

Tier 1:

- **Primary:** `knowledgator/gliner-pii-edge-v1.0` — quantization-aware ONNX (197 MB uint8 / 330 MB fp16), browser-proven path (GLiNER.js/ORT-web), Apache-2.0. Ladder siblings: `gliner-pii-base`, `gliner-pii-large`.
- **Stretch arm:** `fastino/gliner2-privacy-filter-PII-multi` (GLiNER2-PII, 0.3 B; arXiv 2605.09973) — label-conditioned schema-as-input, best span-F1 on SPY, beats OpenAI Privacy Filter. In-browser ONNX path unproven → attempt ONNX-web export in week 1; keep if it runs, else report as future work.

Tier 2 — **two-stage selection, not a fixed pick** (user decision: model family is itself an experiment variable):

- **Stage 1 bake-off** (dev slice ~200 segments): `Qwen3-4B`, `Qwen3.5-2B`, `Gemma-3-4b-it`, `Phi-4-mini` (3.8 B), `Ministral-3-3B` — all q4f16_1 prebuilt MLC builds (verified present in mlc-ai HF registry), thinking modes disabled. Measured: JSON-schema compliance (pre/post one repair), span F1, p50/p95 latency + tok/s on WebGPU, VRAM, cold-load. Kill rules: compliance <95 % pre-repair, or p95 segment latency >3 s.
- **Stage 2:** winner joins the full ladder as mid rung; size axis runs within the winner's family (e.g. Qwen3.5 0.8B/2B). If the winner has no size siblings (Phi-4-mini), the size axis stays on Qwen3 (0.6/1.7/4 B) and family/size are reported as separate comparisons. Selection = best F1 subject to latency + compliance constraints; full Pareto frontier reported.
- **Watch list (no MLC build → cannot run in WebLLM today):** Gemma 4 E2B/E4B (MatFormer, LiteRT-only for now), SmolLM3-3B, LFM2/2.5. Revisit if MLC builds land.
- **Risk:** Qwen3.5/Ministral-3 MLC weights exist, but current `@mlc-ai/web-llm` npm may lag their architectures. Verify week 1; fallback slate = Qwen3 + Gemma-3 + Phi-4-mini (all confirmed working).

### 4.3 Approach-B baseline

`detect/baselineB.ts`: same `Detector` interface; implementation = tier-2 LLM given (policy chunks + full message), no compiler, no tiers. Runs in the eval harness as just another arm.

## 5. Extension runtime

### 5.1 Process topology (MV3)

```
page world (MAIN)           extension world (ISOLATED)         extension processes
┌──────────────────┐        ┌───────────────────┐    ┌──────────────────────────────┐
│ fetch interceptor │◀──────▶│ content script     │◀──▶│ service worker (router ONLY) │
│ patches window.   │ post-  │ + shadow-DOM UI    │    │        │                     │
│ fetch @doc_start  │ Message│ (review sheet)     │    │        ▼                     │
└──────────────────┘        └───────────────────┘    │ offscreen document:          │
                                                      │  tier0 · ORT-web (t1)        │
                                                      │  WebLLM (t2) · vault         │
                                                      └──────────────────────────────┘
```

- **Models live in an offscreen document, not the service worker.** MV3 kills idle SWs in ~30 s; tier-2 cold load is tens of seconds. Offscreen doc is persistent, owns WebGPU + workers, hosts `core` + vault. SW is a dumb message router.
- MAIN-world script patches `window.fetch` at `document_start` — installed before the provider app's JS runs, so no request escapes unscanned. `fetch` being promise-based makes holding a request while detection runs natural.
- Adapters (matched by hostname + URL pattern) classify requests: chat-send / file-upload / irrelevant; extract `{conversationId, userText, writeback}`. Pure functions — replayable in the harness without a browser tab.

### 5.2 Send flow

1. Adapter extracts text → offscreen runs the detect pipeline (policy resolved for this provider).
2. **No findings → request proceeds untouched, zero UI.** Must remain the overwhelmingly common path.
3. Findings → **review sheet** (shadow-DOM overlay): each span in context, entityType, provenance (policy clause + quote), policy-default action pre-selected. Per-span user overrides allowed and **logged** (research data). One click: Apply & send / Cancel. `block`-severity findings disable "send unmodified."
4. Actions applied → body rewritten via adapter `writeback` → original fetch continues.

### 5.3 Failure semantics are policy, not code

IR carries `failMode: open | closed` (per-provider overridable) and `latencyBudgetMs`:

- Tier 2 over budget → degrade to tier 0/1 findings + "semantic scan skipped" warning in the sheet.
- Engine crash / not yet loaded → `failMode` decides: `closed` blocks the send with an explicit, logged "send anyway" escape hatch; `open` allows with warning.
- Tier 0 is compiled into the bundle and synchronous → some protection exists from browser startup.

### 5.4 Vault + rehydration

> **Implementation note (Plan 2, shipped):** two details below are superseded by what was built — the seed-key formula (now salted with a per-install secret, D1) and the streaming holdback (the `maxSurrogateLen − 1` sketch was replaced after two review rounds). The design intent in this section is unchanged and still governs; for current behaviour read the Deviations log in `docs/superpowers/plans/2026-08-14-02-vault-pseudonymization.md` before implementing against these paragraphs.

- **Surrogates are format-preserving fakes** ("Priya Sharma" → "Anjali Verma"; PAN → structurally-valid fake PAN; client "Globex" → "Vantor"), not markers like ⟦P1⟧ — markers get mangled by the model and destroy answer utility. Deterministic per conversation: surrogate = seeded generator keyed on `hash(conversationId ‖ realValue)` → referential integrity across turns.
- **Credentials are never pseudonymized** — only redacted or blocked. A format-valid fake API key is a lie waiting to be pasted somewhere.
- **Storage:** IndexedDB; values AES-GCM-encrypted (WebCrypto), key in `chrome.storage.session` (evaporates on browser close; exportable deliberately for the harness). Honest framing: this is a reversible mapping table — encryption at rest is hygiene; the security property is that real values never leave.
- **Rehydration:** transport-level. Adapter wraps the response body in a `TransformStream` that parses the provider's SSE/JSON-chunk framing and rewrites surrogate→real **inside string values only**, re-emitting valid frames. Surrogates split across chunks handled by a tail holdback of `maxSurrogateLen − 1` chars, flushed on frame end. Non-streamed responses: buffer, rewrite, re-serialize. Per-provider fallback (likely needed for Gemini's batchexecute): DOM-side MutationObserver rewriting rendered text, contained inside that one adapter.

### 5.5 Options page + audit log

Minimal: load compiled IR, set `TierConfig`, view/export audit log. Audit log records every interception: findings, actions, overrides, latencies → IndexedDB, JSONL export. **The audit log is also the instrumentation layer for the in-extension eval arm.**

## 6. Evaluation design

### 6.1 Policy suite — three deliberately disagreeing policies

Authored as markdown documents, the way an admin would write them:

- **P-FIN** (fintech): PAN/Aadhaar/account numbers forbidden everywhere; client names pseudonymized; provider clause: "no customer data to non-enterprise or foreign-hosted services" → stricter actions for DeepSeek/Gemini.
- **P-MED** (healthcare): patient identifiers + clinical details blocked; client/company names **explicitly permitted**; salary/financial info unregulated.
- **P-CORP** (generic corporate): credentials blocked; unreleased financials + M&A as semantic predicates; internal codenames pseudonymized; permissive about generic personal names.

The deliberate disagreements (client names: forbidden in P-FIN, allowed in P-MED; salaries: P-CORP only; provider clauses: P-FIN only) are what the adaptivity metric grips onto.

### 6.2 Corpus

Research context (established via literature sweep 2026-08-13): the 32-benchmark PII survey (arXiv 2608.02616) contains **zero** policy-conditioned or business-confidential chat benchmarks — all are fixed-taxonomy PII, and the only conversational one is customer support (Kiji). The corpus below fills a documented gap and is **releasable by construction** (CC-BY carriers + fictional entities) — a standalone contribution.

**Carrier certification — no prompt gets a label until certified.** Every carrier (negative or injection-bound) passes:

1. Tier-0 sweep, thresholds at max recall — any hit → quarantine.
2. High-recall model sweep. Certification runs offline (Python, in `corpora/` scripts), so browser constraints don't apply: use the strongest available detectors regardless of ONNX-web support (e.g. GLiNER2-PII and gliner-pii-large via their Python runtimes) at deliberately low thresholds, configured differently from the arms under test.
3. Frontier-model adjudication against the **union of all three policies**. Certified **clear** only if clean under all three (one corpus is scored under each policy).

Quarantine triage: genuinely sensitive → dropped entirely; sensitive-looking but adjudicated benign (public figures, example.com, tutorial API keys) → **hard-negative stratum**, human confirmation required.

Invariant bought: injection happens only into certified-clear carriers ⇒ on positives, gold spans are exactly the injected ones; any finding outside them is a true false positive. Negatives contain nothing sensitive by certification, not assumption.

Honesty mechanisms: (a) **residual-contamination audit** — human review of ~200 random certified-clear carriers → reported bound ("contamination ≤ X % at 95 % confidence"); (b) **circularity disclosure** — cleaning negatives with detectors biases them "easy" for related families; mitigated by frontier adjudicator as deciding voice + max-recall sweeps; discard/quarantine rates reported.

**Negatives (~1,500):** primary source **ShareChat** (arXiv 2512.17843; 142,808 real conversations / 660 K turns from publicly-shared ChatGPT/Perplexity/Grok/Gemini/Claude URLs, Apr 2023–Oct 2025, CC-BY-4.0) — fresher, permissively licensed, and drawn from the very providers we intercept. WildChat-1M as volume supplement. Filtered to English, deduped, certified as above. Hard negatives are a deliberate, reported stratum.

**Positives (~1,500):** **real carriers, controlled injection** — carriers are actual certified-clear ShareChat/WildChat conversations; only the confidential span is synthetic.

- **Fictional universe per policy** (constraint-driven, GLiNER2-PII-style): a small consistent world (company, employees, clients, patients, deal terms) with stable attributes → enables multi-turn positives (secret in turn 2, oblique reference in turn 6 — tests cross-turn detection + vault referential integrity) and coherent utility eval.
- **Controlled dimensions** (REDACT-style, failures diagnosable): surface format, position, density, register match, difficulty tier — verbatim / paraphrased / implicit ("our biggest client, the Cupertino fruit company").
- **Per-policy labels:** every span carries `{span, type, violatesUnder: {P-FIN: block, P-MED: none, P-CORP: pseudonymize}}` — one corpus, three ground truths. Message-level gold per policy: `CLEAR` or `FLAGGED{span → action}`.
- **Realism gates:** (a) frontier-judge naturalness score, bottom decile discarded; (b) **adversarial style probe** — classifier trained to distinguish injected vs pristine prompts *with the sensitive span masked*; if it beats chance meaningfully, injections carry a stylistic tell → generator iterates; (c) ~100-item human spot check.

**Splits & contamination:** dev slice (~200, for bake-off + threshold tuning) disjoint from test; test frozen before any tuning. Compiler self-test generator uses different prompt templates and seeds from the corpus generator; test items with >0.7 8-gram overlap vs any compiler self-test example are dropped.

Known bias, disclosed: ShareChat conversations were publicly shared (selection toward "interesting"; platform-side scrubbing) — which is precisely why injection is needed.

### 6.3 Experiment matrix

```
arms:      T0 · T0+T1(edge) · T0+T1(base) · T0+T1+T2(winner) · [T1=GLiNER2-PII if ONNX-web lands]
           + Approach-B (policy-in-context, no compiler)
bake-off:  T2 ∈ {Qwen3-4B, Qwen3.5-2B, Gemma-3-4B, Phi-4-mini, Ministral-3-3B} — dev slice only
backends:  WASM/CPU · WebGPU        (tier-1 both; tier-2 WebGPU only)
policies:  P-FIN · P-MED · P-CORP
full test set runs: {all arms} × {backends} × {3 policies}; bake-off losers do not graduate
```

### 6.4 Metrics

1. **Leak-prevention rate** (headline): % of positive messages where *every* violating span was caught-and-actioned. Paired with **over-blocking rate**: % of negatives with any false finding. One 2-axis plot; every arm is a point.
2. **Span P/R/F1** per data class per arm — partial-overlap credit at ≥50 % span IoU; exact-match also reported.
3. **Policy-adaptivity delta:** same messages under the three policies — action-accuracy per policy vs per-policy ground truth + confusion matrix over the deliberate disagreement cases. Approach-B runs this too: the head-to-head for compiler-vs-prompting.
4. **Utility preservation** (~200-pair subset): pseudonymized vs original prompt → same provider → rehydrate → frontier-judge scores answer equivalence (win/tie/loss + 1–5 usefulness). Safe because injected spans are fictional. Judge prompt frozen before scoring; 30-pair human spot check validates the judge.
5. **Latency/footprint** (Playwright, real Chrome): per-tier p50/p95, cold-load, VRAM, per arm × backend.

**Statistics:** 95 % bootstrap CIs on all rates; McNemar's test for paired arm comparisons (same messages across arms). n ≈ 1,500/side → ±2–3 pp CIs.

**Reproducibility:** `make eval` → harness emits JSONL (config hash, corpus hash, policy hash in every record) → `analysis/` rebuilds every table and figure from JSONL alone. No number in the writeup that isn't regenerable.

## 7. Error handling summary

| Failure | Behavior |
|---|---|
| Unknown IR schema version | Refuse to load; extension inert with visible error |
| Tier-2 JSON invalid after 1 repair | Fail closed → "flag for user review"; never silent pass-through |
| Tier-2 over `latencyBudgetMs` | Degrade to tier 0/1 findings + warning |
| Engine crash / model not loaded | `failMode` decides (closed = block + logged escape hatch) |
| Adapter can't parse a provider request | Treat as irrelevant traffic → pass through untouched (interception is allow-list, not deny-list) |
| Rehydration stream error | Pass remaining stream through unmodified + notify (response-side failure must not eat the user's answer) |
| Regex catastrophic backtracking | Prevented at compile time (bounded-time validation) |

## 8. Testing strategy

- **Unit:** action resolver (exhaustive over entityType × provider × override permutations), vault determinism/referential integrity, segmenter, SSE reframing with surrogates split across chunk boundaries, every named validator.
- **Compiler:** golden-file tests (policy doc → expected IR skeleton); self-test harness on the three policy docs; quote-grounding rejection tests.
- **Adapters:** recorded request/response fixtures per provider, replayed in Node (pure functions — no browser needed); a small live smoke-test checklist per provider for UI drift.
- **Integration:** Playwright — extension loaded in real Chrome against recorded provider pages; end-to-end pseudonymize→rehydrate on streamed fixtures.
- **Eval harness doubles as the regression suite:** any detection change reruns the dev slice; metric regressions fail CI.

## 9. Risks

| Risk | Mitigation |
|---|---|
| GLiNER2-PII has no ONNX-web path | Stretch arm only; primary is browser-proven gliner-pii-edge |
| web-llm npm lags Qwen3.5/Ministral-3 architectures | Verify week 1; fallback slate Qwen3 + Gemma-3 + Phi-4-mini (confirmed) |
| Provider UI/API drift breaks adapters | Adapters are thin pure functions over recorded fixtures; drift localized per adapter |
| Gemini batchexecute resists stream rewriting | Documented fallback: DOM-side rehydration for that adapter only |
| MV3 SW lifetime kills models | Offscreen document hosts all models (design §5.1) |
| Synthetic positives unrealistic | Real carriers + three realism gates incl. adversarial style probe (§6.2) |
| Dirty carriers corrupt labels | Carrier certification pipeline + residual-contamination audit (§6.2) |
| Frontier model hallucinates policy rules | Quote-grounding gate + compiler self-test + human-audited compilation report |

## 10. Deliverables

1. Working extension (Chrome, MV3) implementing the full pipeline on 4 providers.
2. Policy compiler CLI + three authored policies with compiled IRs.
3. Released corpus: policy-conditioned confidential-data benchmark over real LLM-chat carriers (first of its kind per the 2026 literature sweep).
4. Eval results: all five metrics across the full experiment matrix, reproducible via `make eval`.
5. Written report of findings (ladder curve, compiler-vs-prompting head-to-head, adaptivity results).
