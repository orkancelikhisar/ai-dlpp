# Tier-2 Semantic Judge and Approach-B Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a local instruct LLM in the browser as tier 2 — judging semantic predicates the pattern and span layers cannot express — and build the Approach-B baseline that tests whether compiling a policy beats simply prompting with it.

**Architecture:** A new `packages/tier2` wraps `@mlc-ai/web-llm` behind core's existing `SemanticJudge` interface: grammar-constrained JSON out, one repair retry, then fail-closed. Because a model returns text and core demands exact character offsets, a **span-recovery ladder** turns model-quoted text back into offsets and reports which rung each finding needed. Approach B implements core's `Detector` type directly — same interface, no compiler, no tiers — and runs in the eval harness as just another arm.

**Tech Stack:** `@mlc-ai/web-llm@0.2.84` (TVM/MLC → WGSL, WebGPU only), Playwright with `launchPersistentContext`, zod, vitest.

---

## Read this before Task 1

Everything below was **measured on this machine** during Plan 5 feasibility research, in real Chrome against `@mlc-ai/web-llm@0.2.84`. Where this plan and a docs page disagree, the measurement wins.

### Tier 2 is viable, and the Plan 4 WebGPU defect does not transfer

Plan 4 found `onnxruntime-web`'s WebGPU provider silently returning wrong logits on 3 of 4 rungs. **That does not apply here.** The shipped web-llm bundle contains **zero** occurrences of `onnx` (against 300 of `TVM`), and unlike ORT it **fails loudly**: `detectGPUDevice` throws on a null adapter or any limit below its minimum. There is no WASM fallback and therefore no silent-substitution failure mode. Measured: 77 calls, 143-4,360 prompt tokens, exactly one error (a deliberate context-overflow test); 26/26 byte-identical completions at temperature 0.

**When WebGPU is unavailable, tier 2 is ABSENT, not degraded** — the engine throws at init. Route that to the same fail-closed path as an unparseable response.

### The pinned recipe — each item avoids a specific measured failure

```ts
// The call that works. Every clause here was established by measurement.
await engine.chat.completions.create({
  messages,
  temperature: 0,
  max_tokens: 512,
  response_format: { type: "json_object", schema: JSON.stringify(SCHEMA) },
  // NOTE: no `enable_thinking` key at all. See below.
});
```

- **`schema` must be a STRINGIFIED JSON schema**, not an object. Grammar constraint is applied to logits before sampling, via bundled xgrammar 0.1.27.
- **Never pass `enable_thinking` under constrained decoding.** Measured: it injects a literal think tag into `message.content`, which then fails `JSON.parse`. Omit the key entirely.
- **Never use `structural_tag`** — it hangs forever (upstream issue).
- **`type: "number"` in the constrained schema: MEASURED, and it is safe.** This was previously flagged UNMEASURED and blocking, on the grounds that an uncompilable grammar surfaces as a **hang, not an error**. It no longer needs a browser probe.

  The grammar compiler is reachable from Node. `@mlc-ai/web-llm` inlines xgrammar's WASM as base64 rather than fetching it, and the standalone `@mlc-ai/web-xgrammar@0.1.27` ships a **byte-identical binary** — sha256 `80eb86a9e61e8148a45d60973ec29ffb85077a3e42cee8f042e1452fffd63774`, 727,906 bytes, found in both files. (The bundle also carries a second, larger blob, `f261cfa5…`, 3,479,304 bytes: the TVM runtime.) So compiling a schema under Node exercises the exact binary the browser runs. Two mechanical notes for anyone repeating this: the package is a UMD bundle inside a `"type": "module"` package, so Node gives it **no named exports** and it attaches to `globalThis.xgrammar`; and its emscripten glue contains a `require()` alongside top-level await, which Node refuses to classify — replacing the three `pathToFileURL(__filename).href` script-dir shims with the equivalent literal makes it load, and the WASM hash is unchanged by that edit.

  Measured: 4 schema variants x 3 entry points, each in a child process under an **external** 45 s watchdog (a hang inside WASM blocks Node's event loop, so an in-process timer would never fire). The entry points include `grammarCompiler.compileJSONSchema`, which is the exact call `llm_chat` makes. **All 12 compiled. Nothing hung.** An all-string control ran alongside and is what identified an early harness fault as a harness fault rather than a schema one.

- **Bounds are compilable too, and they are worth taking.** Since the compiler was in hand, `minimum`/`maximum` were measured on the same binary. xgrammar folds them into the grammar — `root_prop_0 ::= ( "0" | "1" | "0" "." [0-9]{1,6} | "1" "." [0-9]{1,6} )` — so the model **cannot emit** an out-of-range value; the logit mask forecloses it before sampling.

  | `confidence` | `0.85` | `95` | `1.5` | `-0.5` | `1e999` | `0.5000001` |
  |---|---|---|---|---|---|---|
  | `{ type: "number" }` (shipped today) | accept | accept | accept | accept | accept | accept |
  | `{ type: "number", minimum: 0, maximum: 1 }` | accept | **REJECT** | accept | **REJECT** | **REJECT** | **REJECT** |

  `schema.ts` currently defends `95` (a model answering on a percentage scale), `-0.5`, and `1e999` (JSON has no `Infinity` literal, but `1e999` parses to one) in zod, *after* generation. Adding the bounds to `JUDGE_SCHEMA` moves that defence a stage earlier. **Keep the zod check regardless:** the grammar is a sound over-approximation, not an exact one — `1.5` still passes, because the rule pins only the leading digit. And note the `{1,6}` fractional cap: a model wanting a 7th decimal place is masked into stopping at six. That is a precision limit, not a failure, but it should be stated rather than discovered.

- **The 8192 context window: narrowed, still needs a probe — but a cheaper one.** All four pinned arms resolve to the same `cs1k` library variant (`<model>-q4f16_1_cs1k-webgpu.wasm`) and all four ship the same `overrides.context_window_size: 4096`, so the lift is structurally identical across arms; `cs1k` is the prefill chunk size, not the context window, so nothing about 8192 is baked into the model library. The residual risk is therefore **KV-cache VRAM, not library capability**, and it concentrates on the two largest arms (`Qwen3-4B` 3,432 MB, `Phi-4-mini` 3,438 MB) where doubling the window doubles the cache. That makes the required probe a **load-only** check per model rather than a full generation run.
- **`launchPersistentContext(userDataDir)` in the harness**, or 3 of 4 cold model loads fail with `QuotaExceededError`. Measured: 2/8 cold loads succeed on an ephemeral context vs 3/3 persistent. A flaky load looks exactly like a model defect, so this is a correctness matter, not convenience.
- **One engine per arm.** Swapping models in one page leaks VRAM.
- **`context_window_size: 8192` loads fine** on the shipped lib (1.7 s warm) and prefills a 4,360-token prompt at 452 tok/s. The 4096 default is a WebLLM override and is liftable — raise it for **both** tier 2 and Approach B so B does not fail on long messages for a reason unrelated to its design.

  **Task 9 measured what actually needs the window, and it is not tier 2.** A tier-2 prompt is the fixed 776-character system turn plus the predicates plus one SEGMENT, and over `corpora/fixtures/smoke.jsonl` the largest selected segment is 153 characters — a whole prompt of **1,196 characters**, 1,105 at the median. The ceiling on tokens is the UTF-8 BYTE count, not the character count — a byte-level BPE tokenizes bytes, and measured against a real one, 20 U+1F389 are 40 UTF-16 code units and 60 tokens. Here the two barely differ: that prompt's segment and the fixed system turn are both ASCII (measured: the largest selected segment is 153 code units and 153 bytes; `judge.ts` contains no non-ASCII character), and only the interpolated predicate text could add any, at most a byte or two per em dash. So the byte ceiling is ~1,196 and tier 2 fits the unlifted 4096 with 3.4x headroom, nowhere near 8192. Re-derive it in BYTES on any other corpus. The window lift is therefore justified by **Approach B alone**, whose prompt carries the whole policy (`policies/p-fin.md` is 5,320 characters) plus the whole message rather than one segment. Keep the lift — the two arms must share a window or the head-to-head measures context rather than method, which is this bullet's original argument and is unaffected — but stop attributing the requirement to tier 2, and note that the load-only 8192 probe this plan still owes is a probe for B's benefit.

  **But that was measured on `Qwen3.5-2B` ONLY.** The other three arms each ship their own `overrides.context_window_size: 4096` and are **unmeasured at 8192**. Before Task 12 commits four arms to it, probe each remaining model at 8192 and record what happens — a model that refuses the larger window, or that loads but thrashes, is a finding, and discovering it as three failed arms mid-bake-off would waste a full run. If a model cannot take 8192, the honest options are to run that arm at 4096 and **report the asymmetry**, or to drop the arm; silently mixing window sizes across arms would make the comparison measure context rather than method.

### Cancellation must interrupt, drain, AND clear — corrected by Task 3 against the real engine

The original note here said a naive `Promise.race` leaves the engine deadlocked for 8 s. Task 3 measured that on the **pinned non-streaming recipe** and the mechanism is different: a bare race costs **10.2 s of latency and then self-recovers**. The 8 s figure came from the streaming path, where abandoning a `for await` never releases the lock.

**Far more importantly, interrupt-and-drain alone is NOT sufficient, and the plan's original `cancel.ts` was defective.** Draining returns the engine to idle with its `interruptSignal` flag still **set**, and on the non-streaming path nothing clears it. Read out of the shipped 0.2.84 bundle: `_generate` clears the flag on entry, but `chatCompletion` tests it *before deciding whether to call `_generate`* — and that poisoned branch sets the output to `""` and returns without ever reaching the clear. `resetChat()` does not touch the field. Only the streaming path clears it unconditionally.

Measured consequence, driving the repo's own module in real Chrome:

| step | plan's original | after the fix |
|---|---|---|
| expire(1500 ms) | `DeadlineExpired` @ 1509 ms, flag left **true** | `DeadlineExpired` @ 1519 ms, flag **false** |
| **next call** | **0 ms, `finish_reason: "abort"`, text `""`** | **171 ms, `stop`, `"OK"`** |
| the call after that | **0 ms, `abort`, `""`** | 164 ms, `stop`, `"OK"` |

**This is worse than the wedge it was written to prevent.** Not a hang anyone would notice, but an instant empty answer that a judge reads as "no findings" — on every message from then on, permanently. So `Interruptible` requires a `clearInterrupt()` alongside `interruptGenerate()`, called after the drain, and it is **required rather than optional** because optionality is exactly how this defect survives review.

**Two consequences for later tasks:**

- **Task 12 must assert the call following any expiry returns a NON-EMPTY body.** The plan's own Step 4 check — "the second call returns" — would have passed against the broken implementation, because it did return, in 0 ms, empty. `clearInterrupt` writes a field TypeScript marks `private`; if upstream renames it the clear silently no-ops and the poisoning returns with a fully green unit suite. No Node test can catch that; the bake-off assertion is the cheapest real guard.
- **Reentrancy was broken and is now serialized per engine.** Measured: two overlapping calls, and the second was silently killed by the first's timeout — resolving with `abort` and 0 characters. The judge calls this per segment, so that was not hypothetical. Note `budgetMs` excludes queue time by design; a caller needing a bound on total elapsed time must pass the `outer` signal, which is honoured while queued.

### The bake-off slate is four arms

| Model | Size | Note |
|---|---|---|
| `Qwen3.5-2B-q4f16_1-MLC` | 2245 MB | Fastest; recommended primary. Decode 33-46 tok/s |
| `Phi-4-mini-instruct-q4f16_1-MLC` | 3438 MB | No thinking mode. Decode 20-29 tok/s |
| `Qwen3-4B-q4f16_1-MLC` | 3432 MB | Likely killed on throughput |
| `Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC` | 2864 MB | Weak: returned **empty** findings on a message full of secrets |

**Note on that id:** the first draft of this plan wrote it without the `-q4f16_1-MLC` suffix. That is a strict PREFIX of the real `prebuiltAppConfig` entry, and `MLCEngine` resolves ids by string equality — so the arm would have thrown at load with an error reading like a bug in our code, exactly the way the dropped `gemma3-4b` would. **Verify every model id against the installed `prebuiltAppConfig` rather than against this plan.**

**`Gemma-3-4b-it` is dropped** — weights exist, but no WebGPU lib is compiled and it is absent from `prebuiltAppConfig`. Run the bake-off **cheapest-first**; decode rate falls with context on both measured models.

### The kill rules changed, and the reason matters

The spec's original rules were replaced (see the amended spec §4.2) because both measured the wrong thing and would have killed all four arms for reasons unrelated to capability:

- JSON compliance is **near-vacuous** under constrained decoding — 0 malformed outputs in 77 calls. It would pass an arm that finds nothing at all.

  **Qualified during Task 1, and it matters for the budget.** "0 malformed" is true and is not the whole picture: **3 of 6 constrained Phi-4-mini calls in the probe corpus failed to `JSON.parse`**, all three identically (`Unterminated string in JSON at position 1959`), because the model emitted a duplicate loop — `"quote": "Halcyon"` about nineteen times — that exhausted the token budget mid-string. The grammar was satisfied throughout, so these are **truncated, not malformed**: a different cause with the same downstream effect, since the parse throws and an unparseable response routes to fail-closed.

  Two consequences. First, this validates Task 2's decision to distinguish `truncated` from `malformed` as separate parse outcomes — it is not a hypothetical distinction. Second, **those calls ran at `maxTokens: 600`, and this plan pins the default at 512** — tighter than a budget already observed truncating. So an arm can be killed by the token budget rather than by capability, which is precisely the failure the amended kill rules exist to prevent. Before the bake-off, either raise the default with a measurement behind the new number, or **count truncations per arm and report them beside the gate verdict** so a budget-killed arm is distinguishable from an incapable one. Do not leave it implicit.

  **Task 9 settled which of those two to take: take the second, and leave 512 alone.** A judge's answer is bounded by the passage it is judging, because every `quote` must be copied from it. Measured over `corpora/fixtures/smoke.jsonl`, the largest segment the escalation policy selects is **153 characters**. A maximal honest answer for that segment — one finding quoting the entire passage — is `{"findings":[{"predicateId":"client-relationship-disclosure","quote":<153 chars>,"confidence":0.85}]}`, about **243 characters**. That is ~61 tokens at a conventional English ratio and **243 tokens at the absolute ceiling of one token per UTF-8 BYTE**, which is the most a byte-level BPE vocabulary can emit — 243 bytes here because that segment is ASCII. (One token per CHARACTER is not a ceiling: measured against a real byte-level BPE, 20 U+1F389 are 40 UTF-16 code units and 60 tokens. On a corpus with non-ASCII in its longest segments, redo this in bytes: `segmentSizeDistribution` reports a `bytes` row for that.) So 512 has 2x headroom over this corpus's worst case under an assumption nobody has to accept, and 8x under the ordinary one: on the corpus that exists, **512 cannot be what kills an arm.**

  Two things that does NOT say. It does not vindicate 512 for Plan 7's corpus, whose segments are not yet measured — re-run Task 9's `segmentSizeDistribution` against it and redo this arithmetic, which is one line. And it does not make the observed truncations harmless: they were a duplicate loop (`"quote": "Halcyon"` about nineteen times) that satisfied the grammar the whole way, and **raising the cap does not fix a loop, it buys a longer one** while spending more wall clock per call against gates that are about wall clock. Report `truncatedResponses` per arm.
- **No model achieves p95 ≤ 3 s.** A tier-2-shaped call is 4.6 s on the cheapest model; an Approach-B call is 7.5 s, whose TTFT alone (2.8-3.9 s) exceeds the budget before one token is emitted.

New gates: **semantic correctness** (required fields populated, spans resolvable at rung ≤ 2, no duplicate-only output) and **normalized throughput** (p95 TTFT ≤ 1.5 s at tier-2 segment size, decode ≥ 25 tok/s). The wall-clock budget comes from a measured segment-size distribution — which is why Task 9 runs **before** the bake-off.

### The live risk is model capability, not backend correctness

On one message containing a name, email, salary, codename and AWS key: `Qwen3.5-2B` with the full policy found **only the AWS key**, three times over; with a short system prompt it found only the salary and missed the key; `Ministral-3-3B` found nothing; `Qwen3-4B` false-positived `"The weather is nice today."` as a secret. **Expect duplicates, expect misses, and do not tune the corpus to hide either.**

---

## File structure

```
packages/tier2/                    NEW -- browser-only, WebGPU only
  src/
    manifest.ts     the four pinned model ids + their measured costs   (pure)
    schema.ts       the JSON schema the model is constrained to        (pure)
    spans.ts        the span-recovery ladder: quote -> offsets         (pure, Node-tested)
    engine.ts       WebLlmEngine seam + the pinned call recipe         (browser)
    cancel.ts       interrupt-and-drain, never Promise.race            (browser)
    judge.ts        WebLlmJudge implements core's SemanticJudge        (browser)
    escalate.ts     which segments are worth a judge at all            (pure)
    baselineB.ts    Approach B: implements core's Detector             (browser)
    index.ts        barrel
  test/                            vitest, Node -- pure modules + fakes

apps/eval/
  src/driver/record.ts             MODIFIED: tier2Stats, rung distribution
  src/page/main.ts                 MODIFIED: loadTier2, useDetector
  src/driver/bakeoff.ts            NEW: the four-arm dev-slice driver
  playwright.config.ts             MODIFIED: launchPersistentContext
```

**Boundary discipline unchanged:** `apps/eval` emits JSONL and computes **no metrics**. Every new number this plan produces is a field in a record, not a score.

---

### Task 1: Package scaffold and the pinned model manifest

**Files:**
- Create: `packages/tier2/package.json`, `tsconfig.json`, `src/manifest.ts`, `src/index.ts`
- Test: `packages/tier2/test/manifest.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { TIER2_MODELS, resolveTier2Config, DEFAULT_TIER2_CONFIG } from "../src/manifest.js";

describe("TIER2_MODELS", () => {
  it("lists only models measured to run, cheapest first", () => {
    // Order is load-bearing: the bake-off runs cheapest-first because decode
    // rate falls with context, so an expensive arm that will be killed on
    // throughput should not be paid for before a cheap one has been measured.
    expect(TIER2_MODELS.map((m) => m.id)).toEqual([
      "Qwen3.5-2B-q4f16_1-MLC",
      "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
      "Qwen3-4B-q4f16_1-MLC",
      "Phi-4-mini-instruct-q4f16_1-MLC",
    ]);
  });

  it("does not list gemma3-4b, which has no compiled WebGPU lib", () => {
    // Weights exist in the registry; no lib is compiled and it is absent from
    // prebuiltAppConfig, so the shipped runtime cannot execute it. Listing it
    // would fail at load with an error that reads like a bug in our code.
    expect(TIER2_MODELS.map((m) => m.id).join(" ")).not.toMatch(/gemma/i);
  });

  it("records the measured cost of each model, not a guess", () => {
    for (const m of TIER2_MODELS) {
      expect(m.sizeMb, m.id).toBeGreaterThan(0);
      expect(m.decodeTokPerSec, m.id).toBeGreaterThan(0);
      expect(typeof m.hasThinkingMode, m.id).toBe("boolean");
    }
  });
});

describe("resolveTier2Config", () => {
  it("defaults to the fastest measured model with a lifted context window", () => {
    const c = resolveTier2Config({});
    expect(c.modelId).toBe("Qwen3.5-2B-q4f16_1-MLC");
    // 4096 is a WebLLM override, not a model limit. Measured: 8192 loads in
    // 1.7 s warm and prefills 4,360 tokens at 452 tok/s. Approach B needs the
    // headroom and both arms must have the same window or the comparison is
    // measuring context, not method.
    expect(c.contextWindowSize).toBe(8192);
    expect(c.temperature).toBe(0);
  });

  it("rejects a model id that is not in the manifest, naming what is", () => {
    expect(() => resolveTier2Config({ modelId: "gemma3-4b-it" })).toThrow(/Qwen3.5-2B/);
  });

  it("rejects a non-zero temperature rather than silently allowing it", () => {
    // The bake-off compares models. A non-zero temperature makes a rerun
    // disagree with itself and the comparison stops being one.
    expect(() => resolveTier2Config({ temperature: 0.7 })).toThrow(/temperature/);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/tier2 exec vitest run`
Expected: FAIL — cannot find module `../src/manifest.js`.

- [ ] **Step 3: Implement**

`packages/tier2/package.json` — mirror `packages/tier1/package.json`, with `"@mlc-ai/web-llm": "0.2.84"` pinned **exactly** (no caret: the recipe below was measured against this build, and 0.2.82 ships neither Qwen3.5-2B nor Phi-4-mini).

`packages/tier2/tsconfig.json` — same shape as `packages/tier1/tsconfig.json` (`lib: ["ES2022","DOM","DOM.Iterable"]`).

`packages/tier2/src/manifest.ts`:

```ts
export interface Tier2Model {
  readonly id: string;
  /** Download size in MB, from prebuiltAppConfig. */
  readonly sizeMb: number;
  /** MEASURED decode rate on the development machine, tokens/sec. Falls with context. */
  readonly decodeTokPerSec: number;
  /**
   * Whether the model has a thinking mode. Recorded and NEVER acted on by
   * passing `enable_thinking`: measured, that key under constrained decoding
   * injects a literal think tag into message.content and breaks JSON.parse.
   * It is here so a reader knows why a model's raw output looks the way it does.
   */
  readonly hasThinkingMode: boolean;
  /** One line on what measurement said about this model's task behaviour. */
  readonly note: string;
}

/**
 * The four models measured to load AND run. Ordered cheapest-first because the
 * bake-off should pay for the expensive arms last.
 */
export const TIER2_MODELS: readonly Tier2Model[] = [
  { id: "Qwen3.5-2B-q4f16_1-MLC", sizeMb: 2245, decodeTokPerSec: 40, hasThinkingMode: true,
    note: "Fastest measured. Found only the AWS key on a 5-entity message, three times over." },
  { id: "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC", sizeMb: 2864, decodeTokPerSec: 30, hasThinkingMode: false,
    note: "Schema-valid but returned EMPTY findings on a message full of secrets." },
  { id: "Qwen3-4B-q4f16_1-MLC", sizeMb: 3432, decodeTokPerSec: 25, hasThinkingMode: true,
    note: "False-positived 'The weather is nice today.' as a secret." },
  { id: "Phi-4-mini-instruct-q4f16_1-MLC", sizeMb: 3438, decodeTokPerSec: 24, hasThinkingMode: false,
    note: "Cold load 73.6 s / 2.18 GB. Deterministic; slowest decode measured." },
];

export interface Tier2Config {
  readonly modelId: string;
  readonly contextWindowSize: number;
  readonly temperature: number;
  readonly maxTokens: number;
}

export const DEFAULT_TIER2_CONFIG: Tier2Config = {
  modelId: "Qwen3.5-2B-q4f16_1-MLC",
  contextWindowSize: 8192,
  temperature: 0,
  maxTokens: 512,
};

export function resolveTier2Config(overrides: Partial<Tier2Config>): Tier2Config {
  // Explicitly-undefined keys must not clobber defaults -- the natural CLI shape
  // passes `{ modelId: undefined }` for "not specified". Task 5 of Plan 4 shipped
  // this bug and it took a review round to find.
  const clean = Object.fromEntries(Object.entries(overrides).filter(([, v]) => v !== undefined));
  const config = { ...DEFAULT_TIER2_CONFIG, ...clean } as Tier2Config;
  if (!TIER2_MODELS.some((m) => m.id === config.modelId)) {
    throw new Error(
      `unknown tier-2 model "${config.modelId}"; measured to run: ` +
        TIER2_MODELS.map((m) => m.id).join(", "),
    );
  }
  if (config.temperature !== 0) {
    throw new Error(`tier-2 temperature must be 0 for a reproducible bake-off, got ${config.temperature}`);
  }
  if (!(Number.isInteger(config.maxTokens) && config.maxTokens > 0)) {
    throw new Error(`tier-2 maxTokens must be a positive integer, got ${config.maxTokens}`);
  }
  return config;
}
```

- [ ] **Step 4: Verify pass**

Run: `pnpm -C packages/tier2 exec vitest run`
Expected: 6 passed. Then from root: `pnpm -r test` and `npm run -s typecheck`, both green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier2): package scaffold with the measured model slate"
```

---

### Task 2: The JSON contract — schema, one repair, fail closed

**Files:**
- Create: `packages/tier2/src/schema.ts`
- Test: `packages/tier2/test/schema.test.ts`

Grammar-constrained decoding makes malformed JSON nearly impossible, but **truncation is real** — measured, 100% of parse failures were truncation, not malformation. So the repair path exists for a failure mode the constraint cannot prevent.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { JUDGE_SCHEMA, JudgeResponseSchema, parseJudgeResponse } from "../src/schema.js";

describe("JUDGE_SCHEMA", () => {
  it("is a plain JSON schema, serializable for the grammar constraint", () => {
    // WebLLM takes response_format.schema as a STRING. A schema carrying a
    // function, a RegExp or a cycle stringifies to something xgrammar cannot
    // compile, and the failure surfaces as a hang rather than an error.
    expect(() => JSON.stringify(JUDGE_SCHEMA)).not.toThrow();
    expect(JSON.parse(JSON.stringify(JUDGE_SCHEMA))).toEqual(JUDGE_SCHEMA);
  });

  it("asks for a verbatim quote, never for offsets", () => {
    // A model asked for character offsets returns wrong ones. The quote is
    // what the span ladder resolves; see spans.ts.
    const props = JUDGE_SCHEMA.properties.findings.items.properties;
    expect(Object.keys(props)).toContain("quote");
    expect(Object.keys(props)).not.toContain("start");
    expect(Object.keys(props)).not.toContain("end");
  });
});

describe("parseJudgeResponse", () => {
  it("accepts a well-formed response", () => {
    const r = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"Acme Corp is our client","confidence":0.9}]}');
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.findings[0]!.quote).toBe("Acme Corp is our client");
  });

  it("reports truncation distinctly from malformation", () => {
    // The two need different responses: truncation says raise max_tokens or
    // shorten the prompt; malformation says the constraint is not working.
    // Measured: every real parse failure was truncation.
    const truncated = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"Acme');
    expect(truncated.ok).toBe(false);
    if (!truncated.ok) expect(truncated.reason).toBe("truncated");
  });

  it("reports a schema mismatch as its own reason", () => {
    const wrong = parseJudgeResponse('{"findings":[{"predicateId":"p1"}]}');
    expect(wrong.ok).toBe(false);
    if (!wrong.ok) expect(wrong.reason).toBe("schema");
  });

  it("rejects an empty quote rather than passing it to the span ladder", () => {
    // An empty quote matches at offset 0 of every message. That is a finding
    // pointing at the wrong text, which is worse than no finding.
    const r = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"","confidence":0.9}]}');
    expect(r.ok).toBe(false);
  });

  it("accepts an empty findings array as a real answer", () => {
    // "Nothing here" is a legitimate judgement, not a failure. Conflating it
    // with a parse failure would make a silent model look like a broken one.
    const r = parseJudgeResponse('{"findings":[]}');
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.findings).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/schema.js`.

- [ ] **Step 3: Implement**

```ts
import { z } from "zod";

/**
 * The schema the model is grammar-constrained to. Deliberately small: every
 * field is one the model can actually produce.
 *
 * `quote` and not `start`/`end`. A small model asked for character offsets
 * returns wrong ones -- and a wrong offset is worse than no finding, because
 * core re-derives `text` from the span, so a mis-located finding is
 * schema-valid and passes the fidelity check. Offsets are recovered locally by
 * `spans.ts` instead, and which rung recovered them is reported.
 */
export const JUDGE_SCHEMA = {
  type: "object",
  properties: {
    findings: {
      type: "array",
      items: {
        type: "object",
        properties: {
          predicateId: { type: "string" },
          quote: { type: "string" },
          confidence: { type: "number" },
        },
        required: ["predicateId", "quote", "confidence"],
      },
    },
  },
  required: ["findings"],
} as const;

export const JudgeResponseSchema = z.object({
  findings: z.array(
    z.object({
      predicateId: z.string().min(1),
      // min(1): an empty quote matches at offset 0 of every message, which is a
      // finding pointing at the wrong text.
      quote: z.string().min(1),
      confidence: z.number(),
    }),
  ),
});

export type JudgeResponse = z.infer<typeof JudgeResponseSchema>;

export type ParseResult =
  | { ok: true; value: JudgeResponse }
  | { ok: false; reason: "truncated" | "malformed" | "schema"; detail: string };

/**
 * Parse a model response, distinguishing the three failure modes because they
 * call for different responses: truncation means raise max_tokens or shorten
 * the prompt, malformation means the grammar constraint is not working, and a
 * schema mismatch means the prompt and the schema disagree.
 *
 * MEASURED: under grammar-constrained decoding, 0 of 77 calls produced
 * malformed JSON, and 100% of parse failures were truncation. The repair retry
 * therefore exists for truncation, which the constraint cannot prevent.
 */
export function parseJudgeResponse(raw: string): ParseResult {
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (cause) {
    // A truncated response is unterminated: the JSON parser reaches the end of
    // input mid-value. Malformation is a syntax error somewhere earlier.
    const truncated = /Unexpected end of (JSON )?input|Unterminated/i.test(String(cause));
    return { ok: false, reason: truncated ? "truncated" : "malformed", detail: String(cause) };
  }
  const parsed = JudgeResponseSchema.safeParse(json);
  if (!parsed.success) return { ok: false, reason: "schema", detail: z.prettifyError(parsed.error) };
  return { ok: true, value: parsed.data };
}
```

- [ ] **Step 4: Verify pass** — 6 passed; root suite and typecheck green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier2): grammar-constrained JSON contract with truncation-aware parsing"
```

---

### Task 3: Cancellation that interrupts and drains

**Files:**
- Create: `packages/tier2/src/cancel.ts`
- Test: `packages/tier2/test/cancel.test.ts`

**This task exists because of one measurement.** A naive `Promise.race` timeout around a WebLLM call leaves the engine **permanently wedged** — measured on the real pipeline in the browser, the next call did not return within 8 s. Core's `SemanticJudge` is handed an `AbortSignal` -- on `JudgeRequest.signal` since Task 6, not as a positional argument -- precisely so a tier-2 run exceeding `latencyBudgetMs` can degrade to the tier 0/1 findings, and doing that wrong costs every subsequent message.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it, vi } from "vitest";
import { runWithDeadline, DeadlineExpired } from "../src/cancel.js";

/** A fake engine that records whether it was interrupted and drained. */
function fakeEngine(opts: { hangs?: boolean } = {}) {
  const state = { interrupted: 0, drained: 0, calls: 0, wedged: false };
  return {
    state,
    async interruptGenerate() { state.interrupted += 1; },
    async create(signal?: AbortSignal): Promise<string> {
      state.calls += 1;
      if (state.wedged) throw new Error("engine is wedged: a previous call was abandoned");
      if (opts.hangs) {
        // Resolves only when interrupted -- which is what a real engine does
        // once interruptGenerate lands. If the caller never interrupts, the
        // engine stays busy and the NEXT call is the one that suffers.
        await new Promise<void>((resolve) => {
          const t = setInterval(() => { if (state.interrupted > 0) { clearInterval(t); resolve(); } }, 1);
          signal?.addEventListener("abort", () => { clearInterval(t); resolve(); });
        });
        state.drained += 1;
        return "";
      }
      return '{"findings":[]}';
    },
  };
}

describe("runWithDeadline", () => {
  it("returns the value when the call finishes in time", async () => {
    const e = fakeEngine();
    await expect(runWithDeadline(e, (s) => e.create(s), 1000)).resolves.toBe('{"findings":[]}');
    expect(e.state.interrupted).toBe(0);
  });

  it("interrupts AND drains on expiry, so the next call still works", async () => {
    // The whole point. A Promise.race would reject here and leave the engine
    // generating; measured on the real pipeline, the next call then never
    // returns. Interrupting and awaiting the drain is what keeps the engine
    // usable.
    const e = fakeEngine({ hangs: true });
    await expect(runWithDeadline(e, (s) => e.create(s), 20)).rejects.toBeInstanceOf(DeadlineExpired);
    expect(e.state.interrupted).toBe(1);
    expect(e.state.drained).toBe(1);
    await expect(runWithDeadline(e, (s) => e.create(s), 1000)).resolves.toBeDefined();
  });

  it("does not resolve before the drain completes", async () => {
    // If the deadline path resolves while the engine is still generating, the
    // caller starts the next segment against a busy engine -- the wedge, one
    // level up.
    const e = fakeEngine({ hangs: true });
    const p = runWithDeadline(e, (s) => e.create(s), 10).catch(() => "rejected");
    await p;
    expect(e.state.drained).toBe(1);
  });

  it("propagates an outer abort the same way as a deadline", async () => {
    const e = fakeEngine({ hangs: true });
    const ac = new AbortController();
    const p = runWithDeadline(e, (s) => e.create(s), 5000, ac.signal);
    ac.abort();
    await expect(p).rejects.toBeInstanceOf(DeadlineExpired);
    expect(e.state.interrupted).toBe(1);
  });

  it("clears its timer on the success path", async () => {
    // A leaked timer keeps the process alive after the suite finishes, which
    // shows up as vitest hanging rather than as a failure.
    vi.useFakeTimers();
    const e = fakeEngine();
    await runWithDeadline(e, (s) => e.create(s), 1000);
    expect(vi.getTimerCount()).toBe(0);
    vi.useRealTimers();
  });
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/cancel.js`.

- [ ] **Step 3: Implement**

```ts
/** Thrown when a call exceeded its budget or the caller aborted. */
export class DeadlineExpired extends Error {
  constructor(readonly budgetMs: number) {
    super(`tier-2 call exceeded its ${budgetMs}ms budget and was interrupted`);
    this.name = "DeadlineExpired";
  }
}

/** The part of a WebLLM engine this wrapper needs. */
export interface Interruptible {
  interruptGenerate(): Promise<void>;
}

/**
 * Run one engine call under a deadline, interrupting AND DRAINING on expiry.
 *
 * MEASURED, and the reason this function exists: a `Promise.race` between the
 * call and a timer leaves the engine generating after the race resolves, and
 * the NEXT call then never returns -- the engine is permanently wedged. The
 * upstream issue has been open two years.
 *
 * So on expiry this asks the engine to stop, then AWAITS the original promise
 * before rejecting. The await is the load-bearing part: resolving earlier hands
 * control back to a caller that will immediately start the next segment against
 * a still-busy engine, which is the same wedge one level up.
 */
export async function runWithDeadline<T>(
  engine: Interruptible,
  call: (signal: AbortSignal) => Promise<T>,
  budgetMs: number,
  outer?: AbortSignal,
): Promise<T> {
  const ac = new AbortController();
  let expired = false;

  const timer = setTimeout(() => {
    expired = true;
    ac.abort();
    void engine.interruptGenerate();
  }, budgetMs);

  const onOuterAbort = () => {
    expired = true;
    ac.abort();
    void engine.interruptGenerate();
  };
  outer?.addEventListener("abort", onOuterAbort, { once: true });
  if (outer?.aborted === true) onOuterAbort();

  try {
    const value = await call(ac.signal);
    // Deliberately checked AFTER the await: the interrupt makes the engine
    // return early rather than throw, so a drained call resolves normally with
    // a partial or empty body. Returning that as a real answer would report a
    // truncated judgement as a complete one.
    if (expired) throw new DeadlineExpired(budgetMs);
    return value;
  } finally {
    clearTimeout(timer);
    outer?.removeEventListener("abort", onOuterAbort);
  }
}
```

- [ ] **Step 4: Verify pass** — 5 passed.

**Then verify against the real engine, because the fake cannot prove the property.** Write a throwaway script (namespaced `task3-*`, outside the package dirs) that loads `Qwen3.5-2B` in real Chrome, issues a long generation, expires it through `runWithDeadline`, and then issues a second call. Assert the second call returns. Report the measured timings. Do **not** commit that script — the unit tests run without a 2 GB model, and a test that needs one would make the suite unrunnable.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier2): interrupt-and-drain cancellation, never a timeout race"
```

---

### Task 4: The span-recovery ladder

**Files:**
- Create: `packages/tier2/src/spans.ts`
- Test: `packages/tier2/test/spans.test.ts`

**The hardest task in this plan, and the one most likely to rig the comparison if done badly.** The model returns a quote; core demands offsets satisfying `text === message.slice(start, end)`. Get this wrong and Approach B fails on offset arithmetic rather than on judgement, handing the compiler a win it did not earn.

**A design that was tested and failed.** The obvious approach — ask the model for the ~20 characters preceding its quote, then match `before + quote` — **resolved 0 times out of 5** on real models. They return whole sentences, the entire prefix, or a whitespace-truncated fragment. Measured examples: asked for a 20-char anchor, `Qwen3.5-2B` returned `"Can someone pull the AWS key"` (missing a trailing space, so no match) and 165 characters of preceding text; `Phi-4-mini` returned the whole enclosing sentence twice, and once a "quote" that was not verbatim at all.

**What works instead:** ask for a longer verbatim quote and resolve it locally. Measured ambiguity at a 1,000-character window: one-word capitalized quotes are non-unique 22-51% of the time, two-word 7-38%, **three-word ≈ 0%**.

**Three defects Task 4 found in this plan's own `spans.ts`, by installing it verbatim and testing it — it passed all 9 of its own tests and failed 7 of the adversarial ones:**

1. **The fold map desynchronizes on a casefold that expands.** `out.push(ch.toLowerCase())` pushes a *string* while `map.push(i)` pushes one entry, assuming `toLowerCase()` preserves length per code unit. Enumerating all 65,536 BMP code units found **exactly one that does not: U+0130, Turkish dotted capital İ**, which lowercases to two code units. From there the two arrays are off by one for the rest of the string. Measured: a quote in `"İstanbul office: … codename is Bluebird …"` resolved to `{start: 18}` instead of `{start: 17}`, yielding `"he merger codename is Bluebird "`. **`text === message.slice(start, end)` still holds, so core accepts it** — this is the mis-location hazard in its purest form, and `applyActions` would vault the wrong string and leave the `t` behind.

2. **Rung 2 sheds the word carrying the secret — a security defect, not a quality one.** Word-level prefix descent drops a whole word to shed one appended punctuation mark, and for a quote shaped `<label> <secret>.` the last word *is* the secret. Measured: `"the AWS key AKIAIOSFODNN7EXAMPLE."` resolved to `"the AWS key"` — **a finding that names the credential without covering it**, so the vault takes the label and the key stays in the message. Not hypothetical: Phi-4-mini appended exactly that trailing punctuation in this very corpus. Fixed by stripping trailing non-word characters *before* the word descent, which returns a longer and better-located span without adding a rung or ever word-searching.

3. **A span boundary could split a surrogate pair**, producing a lone surrogate that still satisfies core's fidelity check.

**Also worth knowing:** this plan's test `"never returns a span whose text disagrees with its offsets"` is **tautological** — the implementation defines `text` as the slice, so it cannot fail. The tests that actually catch drift assert offsets against `indexOf` ground truth. And the ladder was validated against the **six real model quotes** in the probe corpus, nothing selected out; the one non-verbatim quote came from the **Approach B arm** specifically, which is the arm that must not lose on offset arithmetic.

**Rung 2 was rebuilt after review, and the numbers it produces must be read differently.** The first fix — strip trailing non-word characters, then descend by whole words — was found by two independent reviews to reproduce the defect it was written to cure. Stripping was **greedy**, so a secret ending in `=`, `_`, `!` or `)` lost that character to the span while it stayed in the message; and the word descent still shed the final word whenever a model perturbed it with a *word* character rather than punctuation (`"…AKIAIOSFODNN7EXAMPLF"` resolved to `"the AWS key"` — 0 of 20 credential characters).

Rung 2 is now an **incremental code-point peel**: remove one code point from the tail, retest for a unique occurrence, stop at the first hit, trimming only trailing spaces. Measured against the reviews' 11 concrete cases, 6 strictly better / 5 identical / 0 worse; over a 40,000-pair differential fuzz, **0 mis-located, 0 start-offset moves, 0 spans shorter than the old ladder**, 2,766 longer, 778 newly resolved. Containment was *proved*, not merely fuzzed: same start, end never smaller.

Two consequences worth carrying into the bake-off:

- **Rung-2 spans now end inside a word 31.6% of the time** (the word descent: 0.0%). That is the improvement stated honestly — the `…EXAMPLF` case recovers 19 of 20 credential characters instead of 0 — but it means a rung-2 span is no longer guaranteed to be word-aligned, and a reader comparing arms should not treat a mid-word end as a defect.
- **The three-word floor no longer bounds what it admits.** The "three-word quotes are ~0% ambiguous" measurement was taken over *whole* words; the peel will accept `"the deploy p"`. The floor still stops the ladder from descending into one- and two-word territory, but it is no longer the ambiguity guarantee its name suggests.

A third defect surfaced only under differential fuzzing, and neither review found it: `return at(...)` on a unique hit meant that a candidate `at()` correctly **refused** — a boundary splitting a surrogate pair — ended the whole ladder instead of continuing to a shorter candidate. Fixed at both rungs; it accounted for 259 of the 778 newly resolved cases.

**The honesty requirement.** Because a recovered span is sliced from the message, a *mis-located* span is schema-valid and passes core's fidelity check — that check catches incoherence, not mis-location. So **which rung resolved each finding is a first-class reported metric**, not an implementation detail. An arm whose findings mostly resolve at rung 3 is reporting guesses.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { buildFoldMap, resolveQuote } from "../src/spans.js";

const MSG = "Hi team,\n\nAcme  Corp is our biggest client. Please don't tell Acme Corp's rival.";

describe("buildFoldMap", () => {
  it("maps every folded index back to an original index", () => {
    const { folded, map } = buildFoldMap(MSG);
    expect(map).toHaveLength(folded.length);
    for (let i = 0; i < folded.length; i += 1) {
      expect(map[i]).toBeGreaterThanOrEqual(0);
      expect(map[i]).toBeLessThan(MSG.length);
    }
  });

  it("collapses whitespace runs and casefolds", () => {
    // "Acme  Corp" (two spaces) and a newline run both fold to single spaces,
    // so a model that reflows whitespace still matches.
    expect(buildFoldMap(MSG).folded).toContain("acme corp is our biggest client");
  });
});

describe("resolveQuote", () => {
  it("rung 1: resolves an exact quote and slices the ORIGINAL text", () => {
    const r = resolveQuote(MSG, "Acme  Corp is our biggest client");
    expect(r?.rung).toBe(1);
    expect(MSG.slice(r!.start, r!.end)).toBe(r!.text);
  });

  it("rung 1: resolves a quote whose whitespace and case differ from the message", () => {
    // The model reflowed the double space and lowercased. The span must still
    // land on the ORIGINAL characters, double space included.
    const r = resolveQuote(MSG, "acme corp is our biggest client");
    expect(r?.rung).toBe(1);
    expect(MSG.slice(r!.start, r!.end)).toBe("Acme  Corp is our biggest client");
  });

  it("refuses an ambiguous quote rather than picking an occurrence", () => {
    // "Acme Corp" appears twice. Guessing one is a coin flip that produces a
    // finding pointing at text the model was not talking about -- and it would
    // slice cleanly, so nothing downstream could object.
    const r = resolveQuote(MSG, "Acme Corp");
    expect(r).toBeUndefined();
  });

  it("rung 2: falls back to the longest unique prefix of the quote", () => {
    // Models append or alter trailing punctuation. Measured: one returned a
    // quote with an added "?" and five words dropped.
    const r = resolveQuote(MSG, "Acme  Corp is our biggest client, obviously!!");
    expect(r?.rung).toBe(2);
    expect(MSG.slice(r!.start, r!.end)).toBe(r!.text);
  });

  it("returns undefined rather than a guess when nothing matches", () => {
    expect(resolveQuote(MSG, "a sentence that is nowhere in this message")).toBeUndefined();
  });

  it("never returns a span whose text disagrees with its offsets", () => {
    // The invariant core enforces by throwing. Checked here over every rung so
    // a rung added later cannot violate it quietly.
    for (const q of ["Acme  Corp is our biggest client", "acme corp is our biggest client",
                     "Acme  Corp is our biggest client!!", "nope"]) {
      const r = resolveQuote(MSG, q);
      if (r !== undefined) expect(MSG.slice(r.start, r.end)).toBe(r.text);
    }
  });

  it("resolves correctly when an emoji precedes the quote", () => {
    // UTF-16 again. Offsets are JS string indices throughout this system.
    const msg = "\u{1F389} Acme Corp is our client";
    const r = resolveQuote(msg, "Acme Corp is our client");
    expect(r).toBeDefined();
    expect(msg.slice(r!.start, r!.end)).toBe("Acme Corp is our client");
  });
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/spans.js`.

- [ ] **Step 3: Implement**

```ts
export interface ResolvedQuote {
  readonly start: number;
  readonly end: number;
  /** Always `message.slice(start, end)`. Sliced, never assembled. */
  readonly text: string;
  /** Which rung resolved it. REPORTED per finding -- see the ladder note. */
  readonly rung: 1 | 2;
}

/**
 * A fold of `text` that is easier to match against, plus a map from every
 * folded index back to the ORIGINAL index it came from.
 *
 * The map is the whole point. Matching happens in folded space, where a model's
 * reflowed whitespace and altered case still line up; the span is then sliced
 * from the UNTOUCHED original, so the returned text is exactly what the message
 * says. Normalizing the message and slicing THAT would return text the user
 * never wrote.
 */
export function buildFoldMap(text: string): { folded: string; map: number[] } {
  const out: string[] = [];
  const map: number[] = [];
  let inWhitespace = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i]!;
    if (/\s/.test(ch)) {
      // A whitespace RUN folds to one space, mapped to the run's first index.
      if (!inWhitespace) { out.push(" "); map.push(i); inWhitespace = true; }
      continue;
    }
    inWhitespace = false;
    // Smart quotes and dashes: models rewrite these freely.
    const folded = ch.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[–—]/g, "-");
    out.push(folded.toLowerCase());
    map.push(i);
  }
  return { folded: out.join(""), map };
}

/** Fold a quote the same way, without needing a map back. */
function foldQuote(q: string): string {
  return buildFoldMap(q).folded.trim();
}

/**
 * Turn a model-supplied quote into character offsets into `message`.
 *
 * The ladder, and why each rung exists:
 *
 *   rung 1 -- the folded quote occurs EXACTLY ONCE. The strong case.
 *   rung 2 -- it does not occur, but its longest prefix that occurs exactly
 *             once does. Models append punctuation and drop trailing words;
 *             measured, one returned a quote with an added "?" and five words
 *             missing.
 *
 * Ambiguity is refused, never resolved by picking. A quote occurring twice
 * gives no evidence about which the model meant, and a guessed span slices
 * cleanly -- so core accepts it and nothing downstream can object. Measured
 * ambiguity at a 1,000-char window: one-word quotes are non-unique 22-51% of
 * the time, two-word 7-38%, three-word about 0%. Prompt for whole clauses.
 *
 * There is deliberately no rung that searches for the quote's individual words.
 * That is the rung that would let a model score by accident.
 */
export function resolveQuote(message: string, quote: string): ResolvedQuote | undefined {
  const { folded, map } = buildFoldMap(message);
  const needle = foldQuote(quote);
  if (needle.length === 0) return undefined;

  const at = (rung: 1 | 2, fi: number, flen: number): ResolvedQuote | undefined => {
    const start = map[fi];
    // The end maps from the LAST folded character, +1 in original coordinates,
    // because a folded index maps to where its character began.
    const lastIdx = map[fi + flen - 1];
    if (start === undefined || lastIdx === undefined) return undefined;
    const end = lastIdx + 1;
    if (!(start >= 0 && start < end && end <= message.length)) return undefined;
    return { start, end, text: message.slice(start, end), rung };
  };

  const occurrences = (n: string): number[] => {
    const hits: number[] = [];
    for (let i = folded.indexOf(n); i !== -1; i = folded.indexOf(n, i + 1)) hits.push(i);
    return hits;
  };

  const exact = occurrences(needle);
  if (exact.length === 1) return at(1, exact[0]!, needle.length);
  if (exact.length > 1) return undefined; // ambiguous: refuse

  // Rung 2: longest unique prefix, word by word so a prefix never ends mid-word.
  const words = needle.split(" ").filter((w) => w.length > 0);
  for (let n = words.length - 1; n >= 3; n -= 1) {
    const prefix = words.slice(0, n).join(" ");
    const hits = occurrences(prefix);
    if (hits.length === 1) return at(2, hits[0]!, prefix.length);
  }
  return undefined;
}
```

Note the floor of **3 words** in rung 2: below that, measured ambiguity rises sharply, and a two-word prefix that happens to be unique in one message will not be in the next.

- [ ] **Step 4: Verify pass** — 8 passed; root suite and typecheck green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier2): span-recovery ladder that refuses ambiguity instead of guessing"
```

---

### Task 5: The engine seam and the pinned call recipe

**Files:**
- Create: `packages/tier2/src/engine.ts`
- Test: `packages/tier2/test/engine.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { buildCallParams, WebLlmEngine } from "../src/engine.js";
import { JUDGE_SCHEMA } from "../src/schema.js";
import { DEFAULT_TIER2_CONFIG } from "../src/manifest.js";

describe("buildCallParams", () => {
  const p = buildCallParams([{ role: "user", content: "hi" }], DEFAULT_TIER2_CONFIG);

  it("stringifies the schema, because WebLLM takes a string", () => {
    // MEASURED: passing the schema as an object does not constrain generation.
    expect(typeof p.response_format.schema).toBe("string");
    expect(JSON.parse(p.response_format.schema)).toEqual(JUDGE_SCHEMA);
    expect(p.response_format.type).toBe("json_object");
  });

  it("never sends enable_thinking", () => {
    // MEASURED: under constrained decoding that key injects a literal think tag
    // into message.content, which then fails JSON.parse. Omitting the key is
    // not the same as setting it false -- the key must be absent.
    expect(Object.keys(p)).not.toContain("enable_thinking");
    expect(JSON.stringify(p)).not.toContain("enable_thinking");
  });

  it("never sends structural_tag", () => {
    // MEASURED upstream: hangs forever. json_object + schema does the same job.
    expect(JSON.stringify(p)).not.toContain("structural_tag");
  });

  it("pins temperature to 0 so a rerun agrees with itself", () => {
    expect(p.temperature).toBe(0);
  });
});

describe("WebLlmEngine", () => {
  it("reports the model it actually loaded, not the one requested", async () => {
    // The same disease Plan 4 found twice: a field recording intent rather
    // than fact. A record must say what ran.
    const fake = {
      chat: { completions: { create: async () => ({ choices: [{ message: { content: "{}" }, finish_reason: "stop" }] }) } },
      interruptGenerate: async () => {},
      unload: async () => {},
    };
    const e = new WebLlmEngine(fake as never, "Qwen3.5-2B-q4f16_1-MLC");
    expect(e.loadedModelId).toBe("Qwen3.5-2B-q4f16_1-MLC");
  });

  it("surfaces finish_reason, so truncation is distinguishable from a short answer", async () => {
    // A model that stopped because it hit max_tokens produced a partial
    // judgement. Reporting that as a complete one understates recall with no
    // way to notice.
    const fake = {
      chat: { completions: { create: async () => ({ choices: [{ message: { content: '{"findings":[]}' }, finish_reason: "length" }] }) } },
      interruptGenerate: async () => {},
      unload: async () => {},
    };
    const e = new WebLlmEngine(fake as never, "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete([{ role: "user", content: "hi" }], DEFAULT_TIER2_CONFIG);
    expect(r.finishReason).toBe("length");
  });
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/engine.js`.

- [ ] **Step 3: Implement**

`packages/tier2/src/engine.ts` exports `buildCallParams(messages, config)` returning the pinned recipe, and a `WebLlmEngine` class wrapping the MLC engine with `complete(messages, config)` returning `{ content, finishReason, usage }`.

Also export `createWebLlmEngine(modelId, config, onProgress?)` which calls `CreateMLCEngine` with `{ initProgressCallback, context_window_size: config.contextWindowSize }`, and **records the model id the engine reports** rather than the one requested.

Two things the doc comment must state, because both were measured:

- The engine **throws at init** when WebGPU is unavailable or an adapter limit is below its minimum — the good failure mode, and the opposite of onnxruntime-web. Tier 2 is therefore *absent* on such a machine, not degraded, and the caller routes to fail-closed.
- Two adapter limits on the development machine sit at exactly the required minimum with zero headroom (`maxStorageBuffersPerShaderStage` 10, `maxComputeWorkgroupStorageSize` 32768), so a device one unit below throws outright.

**Four defects Task 5 found in this task, by reading the installed `@mlc-ai/web-llm` 0.2.84 rather than trusting the text above:**

1. **`context_window_size` is in the WRONG ARGUMENT, and the failure is silent.** This task says to call `CreateMLCEngine` with `{ initProgressCallback, context_window_size }` — that is the *second* argument, `MLCEngineConfig`, which declares exactly four fields (`appConfig`, `initProgressCallback`, `logitProcessorRegistry`, `logLevel`) and is read for nothing else. The setting is dropped without a warning. `reloadInternal` builds the effective chat config as `{...mlc-chat-config.json, ...modelRecord.overrides, ...chatOpts}`, and **all four pinned arms ship `overrides.context_window_size: 4096` in the installed `prebuiltAppConfig`** — verified by enumerating them — so only `chatOpts`, the *third* argument, can raise it. Following this task as written would have run every arm at 4096 while the record claimed 8192, with a fully green suite: exactly the wrong-number-with-no-symptom failure the plan's own preamble worries about. The test therefore asserts the **argument position**, not the value.

2. **`new WebLlmEngine(fake, "Qwen3.5-2B…"); expect(e.loadedModelId).toBe("Qwen3.5-2B…")` is tautological** — the expected value *is* the constructor argument, and the fake engine reports no model at all, so the assertion cannot distinguish a loaded id from a requested one. The ninth instance of this defect across Plans 3-5. Replaced: the fake reports a **different** id, and the test asserts `requestedModelId` and `loadedModelId` diverge. Related: 0.2.84 exposes **no accessor for the loaded model** — `MLCEngineInterface` has none and `MLCEngine.loadedModelIdToPipeline` is private. The one channel through which the engine names what it loaded is `ChatCompletion.model`, set from that private map's keys, so `loadedModelId` is `undefined` until the first completion and is never seeded with the requested id.

3. **`expect(Object.keys(p)).not.toContain("enable_thinking")` cannot fail on this version.** The key is not top-level on 0.2.84 — it lives under `extra_body` — so a top-level key check passes even for a request that sets it. Worse, `ChatCompletionRequestBase` has no index signature, so TypeScript already rejects it at the top level. The assertion that can fail is a **deep key walk**. The mechanism, read out of the bundle, confirms the plan's measurement and sharpens it: the pipeline tests `enable_thinking === false`, and *only* that branch encodes `<think>…</think>` and pushes those tokens onto `outputIds`. `undefined` and `true` both take the harmless branch — absent and false are genuinely different, and `true` is not the dangerous value.

4. **`ChatCompletion.usage` is optional and `Choice.finish_reason` is narrower than reality.** `usage?: CompletionUsage` is declared optional (though `MLCEngine` populates it unconditionally on the non-streaming path); `Choice.finish_reason` is declared non-optional but is assigned from `getFinishReason(): ChatCompletionFinishReason | undefined`, so `undefined` reaches the caller and defaulting it to `"stop"` would invent a fact. The full union is exactly `"stop" | "length" | "tool_calls" | "abort"` — Task 3's `"abort"` confirmed. Also: every rate in `usage.extra` is an unguarded division, so an interrupted call (`completion_tokens: 0`) yields NaN, which `JSON.stringify` writes as `null`. And `message.content` is `string | null`; `String(null)` would hand the parser four characters that parse as valid JSON and misreport a missing body as a bad model.

**Two signature changes, both deliberate.** `complete(messages, config, { budgetMs, signal })` — the budget is **required**, because both known ways this engine stops responding present as a call that never returns, so a forgotten budget yields no error and no answer. And `createWebLlmEngine` **throws when `modelId` disagrees with `config.modelId`**: two requested ids in one call is how a record ends up naming a model that never ran.

- [ ] **Step 4: Verify pass** — 26 passed (the block above is 6; the rest cover the four defects, the deadline path, and per-engine serialization).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier2): engine seam with the measured call recipe"
```

---

### Task 6: WebLlmJudge — tier 2 proper

**Files:**
- Create: `packages/tier2/src/judge.ts`
- Test: `packages/tier2/test/judge.test.ts`

Implements core's `SemanticJudge`. **The obligation core has been carrying since Plan 3:** a tier-2 finding must name the shadow entityType `pred:<predicateId>`, not the bare predicate id. `normalizeFindings` throws on an entityType the IR does not contain, so getting it wrong is loud — but getting it *right* is this task's job.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import type { Finding, PolicyIr, Segment } from "@sih/core";
import { detect, loadPolicyIr } from "@sih/core";
import { WebLlmJudge } from "../src/judge.js";
import { fakeEngine, predicateIr } from "./helpers.js";

const MSG = "Please review the Northwind Traders renewal before Friday.";

// CORRECTED after Task 6 shipped. `SemanticJudge.judge` takes ONE object and
// returns a `JudgeVerdict`, not a `Finding[]` -- the positional form these
// snippets were written against no longer compiles. `packages/tier2/test/judge.test.ts`
// is the authority; these two helpers are the minimum that keeps the snippets
// below runnable.
const seg = (text = MSG, start = 0): Segment[] => [
  { kind: "prose", start, end: start + text.length, text },
];
const judged = async (
  instance: WebLlmJudge, segments: Segment[], ir: PolicyIr,
  priorFindings: Finding[] = [], text = MSG,
): Promise<Finding[]> =>
  (await instance.judge({ text, segments, ir, priorFindings, budgetMs: 30_000 })).findings;

describe("WebLlmJudge", () => {
  it("emits the SHADOW entityType, not the bare predicate id", async () => {
    // The contract core has documented since Plan 3. A bare predicate id makes
    // normalizeFindings throw, killing the whole message.
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [{ predicateId: "client-relationship", quote: "Northwind Traders renewal", confidence: 0.9 }] }),
      { budgetMs: 30_000 },
    );
    const found = await judged(judge, seg(), predicateIr());
    expect(found[0]!.entityType).toBe("pred:client-relationship");
  });

  it("survives normalizeFindings inside a real detect() call", async () => {
    // Core is the judge of whether these offsets are right, and it throws
    // rather than warning.
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(predicateIr())),
      provider: "claude",
      text: MSG,
      config: { tier0: false, tier1: false, tier2: true },
      engines: { tier2: new WebLlmJudge(
        fakeEngine({ findings: [{ predicateId: "client-relationship", quote: "Northwind Traders renewal", confidence: 0.9 }] }),
        { budgetMs: 30_000 },
      ) },
    });
    expect(result.findings.map((f) => f.text)).toEqual(["Northwind Traders renewal"]);
  });

  it("drops a finding naming a predicate the IR does not declare", async () => {
    // A model will invent predicate ids. Passing one through makes core throw
    // and loses the whole message, including the findings that were fine.
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [{ predicateId: "not-a-predicate", quote: "Northwind Traders renewal", confidence: 0.9 }] }),
      { budgetMs: 30_000 },
    );
    const found = await judged(judge, seg(), predicateIr());
    expect(found).toEqual([]);
  });

  it("drops a finding whose quote does not resolve, and counts it", async () => {
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [{ predicateId: "client-relationship", quote: "text that is not in the message", confidence: 0.9 }] }),
      { budgetMs: 30_000 },
    );
    const found = await judged(judge, seg(), predicateIr());
    expect(found).toEqual([]);
    expect(judge.stats.unresolvedQuotes).toBe(1);
  });

  it("records the rung that resolved each finding", async () => {
    // A first-class metric, not a detail: an arm whose findings mostly resolve
    // at rung 2 is reporting weaker evidence than one resolving at rung 1, and
    // core's fidelity check cannot tell them apart.
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [{ predicateId: "client-relationship", quote: "Northwind Traders renewal", confidence: 0.9 }] }),
      { budgetMs: 30_000 },
    );
    await judged(judge, seg(), predicateIr());
    expect(judge.stats.rung1 + judge.stats.rung2).toBe(1);
  });

  it("emits offsets absolute into the MESSAGE when the segment starts late", async () => {
    // The same failure Plan 4 spent a task on. The judge sees a segment; core
    // demands offsets into the whole message.
    const prefix = "intro line\n";
    const whole = prefix + MSG;
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [{ predicateId: "client-relationship", quote: "Northwind Traders renewal", confidence: 0.9 }] }),
      { budgetMs: 30_000 },
    );
    const found = await judged(judge, seg(MSG, prefix.length), predicateIr(), [], whole);
    expect(whole.slice(found[0]!.start, found[0]!.end)).toBe("Northwind Traders renewal");
  });

  it("returns no findings and spends no call when the IR declares no predicates", async () => {
    // A policy with no semanticPredicates is legitimate. Calling a 2 GB model
    // to ask about nothing costs seconds per message.
    let calls = 0;
    const judge = new WebLlmJudge(fakeEngine({ findings: [], onCall: () => { calls += 1; } }), { budgetMs: 30_000 });
    const found = await judged(judge, seg(), predicateIr({ predicates: [] }));
    expect(found).toEqual([]);
    expect(calls).toBe(0);
  });

  it("fails closed on an unparseable response after one repair", async () => {
    // The spec's rule: schema-validated, ONE repair retry, then fail closed to
    // "flag for user review" -- a model that will not emit valid JSON must
    // never silently pass text through.
    const judge = new WebLlmJudge(fakeEngine({ raw: "not json at all" }), { budgetMs: 30_000 });
    const found = await judged(judge, seg(), predicateIr());
    expect(judge.stats.repairAttempts).toBe(1);
    expect(judge.stats.failedClosed).toBe(1);
    expect(found).toEqual([]);
  });
});
```

Write `packages/tier2/test/helpers.ts` in this task: `predicateIr(options)` returns a loadable IR carrying one `semanticPredicate` (`client-relationship`) and its minted shadow entityType `pred:client-relationship` with a default action; `fakeEngine(options)` returns an `Interruptible` engine whose `complete` returns either a canned `findings` array as JSON or a raw string.

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/judge.js`.

- [ ] **Step 3: Implement**

`WebLlmJudge implements SemanticJudge`. Per `judge(request)` -- ONE object, `{ text, segments, ir, priorFindings, budgetMs, signal }`, returning a `JudgeVerdict` of `{ findings, scopesJudged, degraded? }`. The positional signature this line used to name was replaced when Task 6 needed `text` (for `scope: "message"`) and `budgetMs` (for the per-message budget) reachable without breaking every implementor again:

1. If `ir.semanticPredicates` is empty, return `[]` **without calling the engine**.
2. Build one prompt per segment carrying: the predicates (`id` + `nlPredicate`), the segment text, and a compact summary of `priorFindings` overlapping that segment. Prompt for a **verbatim quote of the whole enclosing clause, at least three words** — measured, three-word quotes are ~0% ambiguous while one-word are non-unique 22-51% of the time.
3. Call through `runWithDeadline`. On `DeadlineExpired`, count it and return what has been collected — the spec says an over-budget tier-2 run degrades to the lower tiers' findings.
4. Parse with `parseJudgeResponse`, **passing the engine's `finish_reason` as its second argument**. Task 2 established this is not optional politeness: `finish_reason: "length"` is recorded on every truncation in the probe corpus, so threading it turns a heuristic into a fact. Without it the parser must infer truncation from the thrown `JSON.parse` message, and Task 2 measured that inference misclassifying 6 of 16 truncation boundaries. On failure, **one** repair retry appending the parse error; on a second failure, count `failedClosed` and emit nothing for that segment.
5. For each returned finding: reject an unknown `predicateId`; resolve the quote with `resolveQuote` against the **segment** text; drop and count an unresolved one; then offset into the message by `segment.start`.
6. Emit `entityType: shadowIdFor(predicateId)`, `tier: 2`, `source:` **the completion's `model` field**, `confidence` clamped to a finite `[0,1]` — core's merge assumes finite confidence and a NaN destroys its ordering.

**Four things Task 5's reviews established that this task must not rediscover:**

- **Do NOT use `engine.loadedModelId` for `source`.** It is `undefined` until the engine has answered at least once, and `Finding.source` is `string` (`packages/core/src/detect/types.ts:14`) — so it is a type error, and the tempting repair (`?? requestedModelId`) restores the intent-as-fact defect this project has shipped twice. `Tier2Completion.model` is typed `string`, is per-call, and is the channel Task 5 built for exactly this.
- **Handle `finish_reason: "abort"` explicitly.** The union is `"stop" | "length" | "tool_calls" | "abort"` plus `undefined`. `parseJudgeResponse` branches on `"length"` and nothing else, so a poisoned engine's empty body is currently filed as **truncation** — which points at the wrong fix ("raise `max_tokens`") and corrupts the per-arm truncation count the bake-off depends on. An `"abort"` is a cancelled call, not a long answer.
- **`WebLlmEngine` cannot be faked structurally.** Its `#private` fields make TypeScript type it *nominally*, so a plain object literal is not assignable and `as never` is the only escape — which is the tautology trap this plan already fell into once. Task 5 must export an interface seam (the way `cancel.ts` exports `Interruptible`) and this task must take it as a parameter.
- **The library enforces message order.** `postInitAndCheckFields` requires system messages only at index 0 and the **last message to be `user` or `tool`**, and it indexes `messages[length - 1]` with no length check. Build the prompt accordingly or get a `MessageOrderError` — or a bare `TypeError` on an empty list.

**Before you can call `shadowIdFor`, it has to move.** It currently lives in `packages/compiler/src/stages/predicates.ts`, and `packages/compiler` is **Node-only** — importing it from a browser-only package would drag Node dependencies into the page. `SHADOW_PREFIX` and `shadowIdFor` are a *contract* between the compiler that mints shadow entityTypes and the tier-2 engine that must name them, so the contract belongs in `packages/core`, which both already depend on and which is DOM- and Node-free by design.

Move both symbols to `packages/core/src/policy/predicates.ts`, export them from core's barrel, and have `packages/compiler` re-export from core so nothing in Plan 3 breaks. Verify with core's own firewall test that the move introduces no Node import. A duplicated prefix constant is the wrong fix: the two copies would drift, and the failure when they do is a `normalizeFindings` throw on every tier-2 message.

Expose a readonly `stats` carrying at least `rung1`, `rung2`, `unresolvedQuotes`, `unknownPredicates`, `repairAttempts`, `failedClosed`, `deadlineExpiries`, `duplicatesDropped`.

- [ ] **Step 4: Verify pass** — 8 passed; root suite and typecheck green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier2): WebLlmJudge emitting shadow entityTypes with resolved spans"
```

---

**Five things Task 6 established that later tasks depend on.** Two are corrections to this plan; three were gaps, and the degraded-channel commit closed all three — see each entry for what it left open.

*Corrections:*

1. **"Call through `runWithDeadline`" (step 3) is not possible from the judge, and should not be.** `runWithDeadline` takes an `Interruptible`; Task 5's `Tier2Engine` seam exposes only `complete(messages, { budgetMs, signal })`, which already performs the deadline, drain, interrupt-clear and per-engine serialization internally. The judge passes a budget; the engine owns the deadline. This plan text predates the seam.
2. **"confidence clamped to a finite `[0,1]`" (step 6) would be dead code, and contradicts `schema.ts`.** `JudgeResponseSchema` is `z.number().min(0).max(1)`, which rejects NaN, both infinities and out-of-range values before `parseJudgeResponse` returns `ok`. And the schema **rejects rather than clamps** on purpose — clamping a model's `95` to `1.0` would turn a misread scale into a maximally confident finding and hide it from the per-arm failure counts. Confidence is passed through; the validation boundary is `schema.ts`, and it is documented there.

*Gaps as Task 6 left them. All three were closed by the degraded-channel commit; each entry below now records what closed it and what it left open:*

3. **CLOSED (`feat(core)`, the degraded-channel commit) — `SemanticPredicate.scope` was unhonoured.** Every predicate is still judged per segment, including `scope: "message"`, but the gap is now *reported* rather than silent: `JudgeVerdict.scopesJudged` is required, `WebLlmJudge` answers `["segment"]`, and the orchestrator turns "the policy declares a scope you did not evaluate" into a `scope-unjudged` notice on every affected row. `JudgeRequest.text` now carries the whole message, so a judge that wants to answer the message scope can; doing so is Task 7's, and until it does the bake-off says so in every row instead of in a write-up nobody reads.
4. **CLOSED — nothing bounded the whole-message tier-2 budget. Owner: THE ORCHESTRATOR.** `detect` reads `ir.latencyBudgetMs`, subtracts what tiers 0 and 1 already spent (`remainingBudgetMs`), refuses to call the judge at all on a spent budget, and arms a single `AbortController` deadline at what is left — a deadline, never a `Promise.race`, since Task 3 measured that racing wedges the engine. The judge still spends a per-call budget per segment; what the orchestrator adds is the ceiling across N of them. The remaining seam is that `JudgeRequest.budgetMs` is informational — `WebLlmJudge` reads it for nothing and sizes its calls from its constructor budget — so sizing per-call budgets from what is left of the message is still open, and belongs with Task 12, which measures the result.
5. **CLOSED — "fail closed to flag for user review" now has a channel.** `DetectionResult.degraded` is a REQUIRED `DegradedNotice[]`: `{ tier, reason, detail }`, with `reason` one of `failed-closed`, `call-budget-exhausted`, `budget-exhausted`, `absent`, `scope-unjudged`. An empty tier-2 return is no longer indistinguishable from a clean message. Read the field's own docblock before consuming it: `absent` is filed for every deliberately-disabled tier, so `degraded.length` is NOT a cleanliness test, and tier 1 has no channel into it at all (`SpanTagger.tag` returns findings with no verdict), so a truncated tier-1 read is still invisible here. Giving tier 1 a channel is the one piece of this gap still open, and has no owner.

Also settled: on `DeadlineExpired` and on a latched-engine `"abort"`, the judge **stops the run** rather than continuing to the next segment, and returns what it has collected. Continuing is not merely wasteful — Task 3 measured that a latched engine returns an instant empty answer to every subsequent call, which a judge reads as "no findings", so continuing would manufacture clean segments.

---

### Task 7: Escalation — which segments are worth a judge

**Files:**
- Create: `packages/tier2/src/escalate.ts`
- Test: `packages/tier2/test/escalate.test.ts`

Spec §4.1: tier 2 fires **iff** (uncertain ∨ `semanticPredicates` present) ∧ `TierConfig.tier2`. Core's orchestrator currently hands the judge every segment as a conservative placeholder and names this task as the owner.

At 4.6 s per call on the cheapest model, escalation is the difference between a usable system and an unusable one.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { selectSegments } from "../src/escalate.js";

const seg = (kind: "prose" | "code" | "kv", text: string, start = 0) =>
  ({ kind, text, start, end: start + text.length });

describe("selectSegments", () => {
  it("selects every prose segment when the policy has predicates", () => {
    // A semantic predicate is about prose by construction -- it exists because
    // no pattern can express it -- so prose is always worth judging.
    const segs = [seg("prose", "Acme is our client"), seg("prose", "and also this")];
    expect(selectSegments(segs, { hasPredicates: true, uncertain: [] })).toHaveLength(2);
  });

  it("skips code segments even when the policy has predicates", () => {
    // Tier 0 owns code (entropy scans it) and a judge reads it as a wall of
    // false positives while burning seconds per segment.
    const segs = [seg("code", "const k = 'AKIA...'"), seg("prose", "Acme is our client")];
    expect(selectSegments(segs, { hasPredicates: true, uncertain: [] }).map((s) => s.kind)).toEqual(["prose"]);
  });

  it("selects nothing when there are no predicates and nothing is uncertain", () => {
    // The expensive default. Without this, every message pays 4.6 s to ask a
    // model about a policy with no semantic clauses.
    const segs = [seg("prose", "Acme is our client")];
    expect(selectSegments(segs, { hasPredicates: false, uncertain: [] })).toEqual([]);
  });

  it("selects a segment flagged uncertain even with no predicates", () => {
    const segs = [seg("prose", "Acme is our client", 0), seg("prose", "second", 20)];
    expect(selectSegments(segs, { hasPredicates: false, uncertain: [20] }).map((s) => s.start)).toEqual([20]);
  });

  it("never returns the same segment twice", () => {
    // A segment both uncertain and prose-with-predicates must be judged once.
    // Judging it twice doubles the cost and produces duplicate findings that
    // look like model behaviour.
    const segs = [seg("prose", "Acme is our client", 0)];
    expect(selectSegments(segs, { hasPredicates: true, uncertain: [0] })).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/escalate.js`.

- [ ] **Step 3: Implement** `selectSegments(segments, opts)` returning segments in input order with no duplicates, per the rules above. Document why `code` is excluded and why the no-predicate no-uncertainty case returns nothing.

- [ ] **Step 4: Verify pass** — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier2): escalation policy so a judge is spent only where it can help"
```

---

**Four things Task 7 established, two of them corrections to this plan.**

*Corrections:*

1. **`packages/tier2/src/escalate.ts` cannot hold the policy.** `detect` is the
   caller; `packages/core/package.json` declares no dependency on `@sih/tier2`
   while `@sih/tier2` depends on core, so importing the escalation policy from
   the tier it gates is a cycle. It lives in
   `packages/core/src/detect/escalate.ts`; `packages/tier2/src/escalate.ts` is a
   re-export, the way `@sih/compiler` re-exports `shadowIdFor`, and
   `packages/tier2/test/escalate.test.ts` asserts the two are the same function
   objects so a second copy cannot be introduced quietly.

2. **The Step-1 snippet's code test cannot fail for the right reason.** Its
   fixture holds only prose and code, and over that fixture
   `kind === "prose"` and `kind !== "code"` select exactly the same segments —
   so it does not pin which rule is implemented. A kv segment is what makes the
   two distinguishable, and kv is IN: spec 4.1 writes "prose segments", but tier
   1 is already handed prose and kv by the same orchestrator, and excluding kv
   would leave a message that is entirely a config paste (`client: Northwind
   Traders`) with no tier-2 coverage while tier 1 read every line of it.

*Established:*

3. **`uncertain` has a definition and a producer.**
   `uncertainSegmentStarts(segments, priorFindings, below)` reads
   `Finding.confidence` off what tiers 0 and 1 found — the two tiers spec 4.1
   names — and a finding marks every segment it OVERLAPS, since tier 0's regex
   rules scan the whole message and a match can straddle a boundary. The
   threshold is `TierConfig.uncertainBelow`, default `UNCERTAIN_BELOW` = 0.8:
   the midpoint of tier 0's two clusters (entropy fixed at 0.7, regex 0.9 rising
   to 0.95 boosted and capped at 0.99), so neither sits on the boundary. It is
   an experiment variable, not policy, which is why it is on `TierConfig` beside
   `t2Model` rather than in the IR.

   What the branch does NOT buy, so Task 12 does not measure it expecting more:
   tier 2 cannot resolve a tier-0/1 finding's uncertainty. `WebLlmJudge` emits
   only `pred:` shadow entityTypes and no channel exists for one tier to revise
   another's confidence, so escalating a segment because tier 0 hedged gets that
   segment's PREDICATES judged, not that secret re-scored. With no predicates
   declared the judge returns an empty verdict before touching the engine, so an
   uncertainty-only escalation currently costs nothing and yields nothing.

4. **A fully-skipped message is a degradation only when the policy asked for
   something, and needs no new reason word.** With no predicates and nothing
   uncertain, nothing is filed: `WebLlmJudge.judge` returns
   `{findings: [], scopesJudged: []}` on an IR with no `semanticPredicates`
   before it touches the engine, so calling it would have produced the same
   `findings` and the same `degraded`, and filing a notice would report a
   degradation the pipeline does not report when the judge really runs.
   `timings.tier2Ms` — set on the call path, unset on this one — is what
   answers "did tier 2 run?". With predicates declared and no segment selected
   (a message that is entirely a code fence) the run IS weaker and files
   `scope-unjudged`, which already means "the policy declares predicates in a
   scope nothing evaluated". The escalation branch is tested BEFORE the budget
   branch, because `budget-exhausted`'s detail asserts a cause ("already spent …
   so the judge was not called") that would be wrong here: no budget produces a
   call on a message with no escalatable segment, and an operator who raised
   `ir.latencyBudgetMs` in response would see no change.

---

### Task 8: Approach B — the baseline that makes the thesis falsifiable

**Files:**
- Create: `packages/tier2/src/baselineB.ts`
- Test: `packages/tier2/test/baselineB.test.ts`

**This arm exists to threaten the compiler.** Same `Detector` type, but the model gets policy chunks plus the full message, with no compiler and no tiers. If it matches the compiled pipeline, the compiler is not earning its place — and that is a finding worth having, not one to engineer around.

**How a badly-built Approach B would rig the comparison, and the countermeasure for each:**

| Way to rig it | Countermeasure |
|---|---|
| Chunk the policy so a clause is missing from the chunk that sees the message — making a violation undetectable *in principle* | **Do not chunk.** Measured: `p-corp.md` is 1,227 tokens and a full Approach-B prompt is 1,443, well inside an 8192 window. If a policy genuinely will not fit, say so and report it rather than silently truncating |
| Give B a smaller context window than tier 2 | Both arms use `contextWindowSize: 8192` |
| Let B fail on offset arithmetic rather than judgement | B uses the **same** `resolveQuote` ladder as tier 2, and its rung distribution is reported the same way |
| Compare B against the full tiered pipeline only | Also run **B + tier 0**, the mandatory intermediate arm — otherwise the head-to-head conflates "compiling the policy helps" with "having deterministic patterns helps" |
| Prompt B worse than tier 2 is prompted | Both prompts are committed side by side in this file, and the reviewer is asked to compare them |

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { createBaselineB } from "../src/baselineB.js";
import { fakeEngine, minimalIr } from "./helpers.js";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const POLICY = readFileSync(join(import.meta.dirname, "..", "..", "..", "policies", "p-corp.md"), "utf8");
const MSG = "Please review the Northwind Traders renewal before Friday.";

describe("createBaselineB", () => {
  it("satisfies core's Detector type and returns a DetectionResult", async () => {
    // Same interface as the compiled pipeline, so the harness runs it as just
    // another arm rather than through a second code path.
    const detector = createBaselineB({
      engine: fakeEngine({ findings: [{ predicateId: "confidential", quote: "Northwind Traders renewal", confidence: 0.8 }] }),
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector({ ir: minimalIr(), provider: "claude", text: MSG, config: { tier0: false, tier1: false, tier2: false } });
    expect(Array.isArray(r.findings)).toBe(true);
    expect(typeof r.timings.tier0Ms).toBe("number");
  });

  it("sends the WHOLE policy, never a chunk of it", async () => {
    // Chunking such that a clause is absent makes a violation undetectable in
    // principle, which would rig the comparison. Measured: the policy fits.
    let prompt = "";
    const detector = createBaselineB({
      engine: fakeEngine({ findings: [], onPrompt: (p: string) => { prompt = p; } }),
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await detector({ ir: minimalIr(), provider: "claude", text: MSG, config: { tier0: false, tier1: false, tier2: false } });
    // Every section heading of the policy must be present in the prompt.
    for (const heading of POLICY.match(/^##? .+$/gm) ?? []) {
      expect(prompt).toContain(heading.trim());
    }
  });

  it("throws rather than truncating when the policy does not fit", async () => {
    // The honest failure. A silently truncated policy produces an arm that
    // loses for a reason nobody can see in the numbers.
    const huge = "x".repeat(200_000);
    const detector = createBaselineB({ engine: fakeEngine({ findings: [] }), policyText: huge, budgetMs: 60_000 });
    await expect(
      detector({ ir: minimalIr(), provider: "claude", text: MSG, config: { tier0: false, tier1: false, tier2: false } }),
    ).rejects.toThrow(/does not fit|too large/i);
  });

  it("resolves spans through the same ladder as tier 2, and reports the rung", async () => {
    const detector = createBaselineB({
      engine: fakeEngine({ findings: [{ predicateId: "confidential", quote: "Northwind Traders renewal", confidence: 0.8 }] }),
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector({ ir: minimalIr(), provider: "claude", text: MSG, config: { tier0: false, tier1: false, tier2: false } });
    expect(MSG.slice(r.findings[0]!.start, r.findings[0]!.end)).toBe(r.findings[0]!.text);
    expect(r.baselineStats!.rung1 + r.baselineStats!.rung2).toBe(1);
  });

  it("emits an entityType the IR declares, so the merge can resolve an action", async () => {
    // B has no compiler and therefore no shadow entityTypes minted for it. It
    // must still name something the IR contains or core throws -- see the
    // note in the implementation about why this is B's hardest constraint.
    const detector = createBaselineB({
      engine: fakeEngine({ findings: [{ predicateId: "confidential", quote: "Northwind Traders renewal", confidence: 0.8 }] }),
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector({ ir: minimalIr(), provider: "claude", text: MSG, config: { tier0: false, tier1: false, tier2: false } });
    const declared = minimalIr().entityTypes.map((e) => e.id);
    for (const f of r.findings) expect(declared).toContain(f.entityType);
  });
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/baselineB.js`.

- [ ] **Step 3: Implement**

`createBaselineB({engine, policyText, budgetMs, entityTypeFor?})` returns a `Detector`.

**The hardest constraint, and it must be solved honestly.** B has no compiler, so no shadow entityTypes are minted for it — but every `Finding` must name an `entityType` the IR declares or `normalizeFindings` throws. Resolve it by asking the model to name **which entity class from a list** each finding belongs to, where the list is `ir.entityTypes.map(e => e.id)` — the IR's *vocabulary* without its compiled rules. State plainly in the doc comment that this is a concession: B is given the entity taxonomy but not the compiled patterns, provider overrides, or actions, and that boundary is exactly what the comparison measures. A reviewer must be able to see the concession and judge whether it is fair.

Prompt requirements, mirroring tier 2's so the comparison is about method rather than prompt quality:
- the whole policy document, verbatim,
- the entity vocabulary,
- the full message,
- an instruction to quote the **whole enclosing clause, at least three words, verbatim**,
- the same JSON schema and the same grammar constraint.

Return core's `DetectionResult` **unchanged in shape** — it is core's type and B must satisfy `Detector` exactly, so B cannot add a field to it. Expose the counters the same way `WebLlmJudge` does: as a readonly `stats` property on the object `createBaselineB` returns, alongside the callable.

```ts
export interface BaselineB {
  (input: DetectInput): Promise<DetectionResult>;
  readonly stats: BaselineStats;
}
```

The test's `r.baselineStats` therefore becomes `detector.stats` — read it off the detector, not off the result. **`stats` is CUMULATIVE across every message the arm has processed** (measured: `rung1` after each of four identical items is 1, 2, 3, 4), while the record's `tier2Stats` is a per-ITEM field. So Task 11 must record a DELTA — snapshot before the call, subtract after — exactly as `tier1Stats` is built. Copying the snapshot straight onto the row puts running totals on every item, inflating any counter summed over an n-item corpus by about (n+1)/2, and the counters the amended kill rules read (`failedClosed`, `truncatedResponses`, `deadlineExpiries`) are precisely the ones that would then report an arm failing on more messages than the corpus contains. `WebLlmJudge.stats` accumulates the same way, so both arms must be handled identically or the head-to-head compares a delta against a total.

- [ ] **Step 4: Also build the mandatory B + tier 0 arm**

The spec amendment makes this arm **required**, because without it the head-to-head conflates two different claims: "compiling the policy helps" and "having deterministic patterns helps". B alone versus the full pipeline cannot separate them.

Export `createBaselineBPlusTier0({engine, policyText, budgetMs, ir})`, also a `BaselineB`. It runs core's `runTier0(ir, text, segments)` first, then B over the same message, then core's own `mergeFindings` — so the two sources are combined by the *same* overlap resolution the compiled pipeline uses, not by a bespoke union that would advantage one side.

Add a test asserting the composed arm returns at least the tier-0 findings, and that a span found by both sources appears once rather than twice:

```ts
it("merges tier-0 and B findings through core's own resolution, not a union", async () => {
  const detector = createBaselineBPlusTier0({
    engine: fakeEngine({ findings: [{ predicateId: "confidential", quote: "PAN AFTPD1298Q on file", confidence: 0.8 }] }),
    policyText: POLICY, budgetMs: 60_000, ir: minimalIr(),
  });
  const msg = "We have PAN AFTPD1298Q on file for the client.";
  const r = await detector({ ir: minimalIr(), provider: "claude", text: msg, config: { tier0: true, tier1: false, tier2: false } });
  // The PAN is found by tier 0 AND covered by B's quote. Core's merge keeps one.
  const overlapping = r.findings.filter((f) => f.start < msg.indexOf("on file") && f.end > msg.indexOf("AFTPD"));
  expect(overlapping).toHaveLength(1);
});
```

- [ ] **Step 5: Verify pass** — 6 passed; root suite and typecheck green.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(tier2): Approach-B baseline and the B-plus-tier-0 arm"
```

---

**What Task 8 established, and six places this plan was wrong.**

*Corrections:*

1. **Every Step-1 snippet passes `config: {tier0: false, tier1: false, tier2: false}`, and that
   contradicts Task 11.** Task 11 couples `tier2Stats` on the record to `config.tier2 && error
   === null`, so a B row recorded under `tier2: false` either carries no stats at all or is
   rejected by the schema — and the counters this whole arm exists to produce would have nowhere
   to go. B now **requires `tier2: true`**, and the word means "a model read this message", not
   "the compiled tier-2 judge ran": every `Finding` B emits carries `tier: 2` because
   `Finding.tier` is `0 | 1 | 2` and a model produced it. Which arm ran is the record's arm
   field. Both factories also refuse a config that disagrees with what they do (`tier1` at all,
   `tier0` for the wrong factory), because the harness records the config it passed and a row
   claiming coverage the arm never had is the records-state-intent defect in the one place
   nothing downstream could catch it.

2. **The `"emits an entityType the IR declares"` test is vacuous as written.** Its fake emits
   `predicateId: "confidential"`, which `minimalIr()` does not declare, so B drops it, `r.findings`
   is empty, and the `for` loop asserts nothing at all. Rewritten to assert a non-empty
   `findings` first, with the drop-and-count path as its own test.

3. **`createBaselineB({engine, policyText, budgetMs})` cannot perform the fit check the same task
   asks it to perform.** Nothing on the `Tier2Engine` seam exposes the context window or
   `max_tokens`, and 0.2.84 exposes no accessor for reading either off a loaded engine. The
   signature is `{engine, config, policyText, budgetMs}` where `config` is the `Tier2Config` the
   engine was loaded under, and the constructor refuses an engine whose `requestedModelId`
   disagrees with `config.modelId` — the same guard `createWebLlmEngine` performs, for the same
   reason. `entityTypeFor?` was dropped: nothing needs it.

4. **`createBaselineBPlusTier0({..., ir})` takes an IR that `DetectInput` already carries.** Two
   IRs in one call is how an arm runs tier 0 against one policy while the record names another.
   Dropped; the caller's is the only one.

5. **"the same JSON schema" for both arms is not achievable and should not be.** B names an
   entity class, and a wire field called `predicateId` holding an entityType id is the
   field-says-one-thing-holds-another defect one layer down. `BASELINE_B_SCHEMA` is
   `JUDGE_SCHEMA` with exactly one property renamed — same keyword set, same
   `additionalProperties: false`, same `[0,1]` bounds — asserted structurally in
   `schema.test.ts`, and reaches the pinned request through a new
   `CompleteOptions.responseSchemaJson` that replaces `response_format.schema` and nothing else.
   `parseJudgeResponse` and `parseBaselineResponse` are two entry points onto one private core,
   so the arms cannot get different failure *classification*.

6. **The Step-1 snippets do not compile against the shipped helpers.** `fakeEngine` has `onCall`,
   not `onPrompt`; `minimalIr` is core's test fixture and is not reachable from
   `packages/tier2/test/helpers.js`; and `r.baselineStats` is the field Step 3 already corrects
   to `detector.stats` without updating the snippet.

*Established:*

7. **The remaining asymmetry, unfixed and documented rather than hidden: B has one message's
   worth of completion budget where the judge has one per segment.** `max_tokens` is fixed on the
   engine, so a five-segment message gives the compiled arm five times B's output allowance for
   the same input, and Plan 5 already recorded truncation as the dominant parse failure at a
   budget larger than the pinned 512. B cannot correct it without trading one asymmetry for
   another, so `BaselineStats.truncatedResponses` plus per-call `finishReason` and
   `completionTokens` are what make a budget-killed arm distinguishable from an incapable one.
   Task 12 owns the decision.

8. **`finishReason: "length"` has two causes, and neither arm can separate them.** READ from the
   shipped 0.2.84 bundle: "Stop condition 4" sets `finishReason = "length"` when
   `filledKVCacheLength` reaches `contextWindowSize`, which is byte-for-byte the signal
   `max_tokens` produces. Only the PREFILL side is loud (`ContextWindowSizeExceededError`, on
   `numPromptTokens + filledKVCacheLength > contextWindowSize`). So a prompt that fits at prefill
   and exhausts the window mid-answer is indistinguishable from a verbose model — which is why
   B's fit check reserves `config.maxTokens` up front, and why `truncatedResponses` must not be
   read as "raise max_tokens" alone. `judge.ts`'s counter doc said `max_tokens` and has been
   corrected.

9. **Two functions moved onto core's package index**, because `detect` is no longer the only
   orchestrator. `resolveFindings(ir, provider, text, raw)` is the extracted tail of `detect` —
   normalize, cluster, merge, cluster-strictest action — and `detect` now calls it, so there is
   one implementation rather than one per arm; a bespoke union in B would have made the
   head-to-head measure action resolution. `remainingBudgetMs` is exported for the same reason:
   B must arm the spec-5.3 message budget itself, and a second copy would drift on the `> 0`
   boundary and the `MAX_TIMER_DELAY_MS` clamp.

10. **Mutation testing found one defect the whole suite could not see, and it is the exact
    failure this task exists to prevent.** With `WebLlmEngine.complete` dropping
    `opts.responseSchemaJson`, all 251 tests stayed green — every Approach-B test drives a FAKE
    engine that never reaches `buildCallParams`. In real Chrome that arm would be
    grammar-constrained to `predicateId` while its prompt asked for `entityType`, so every B
    response would fail the schema and B would fail closed on every message. Fixed by an
    `engine.test.ts` test that drives a real `WebLlmEngine` through the `createEngine` seam and
    reads the request off the wire. A second survivor is documented rather than fixed: replacing
    `severity: entity.severity` in B's `collect` with a literal leaves the suite green, because
    `resolveFindings` re-derives severity afterwards — the assignment is belt and braces and no
    test can distinguish it, and the test that claimed to pin it now says so.

11. **The standing brief's baseline test counts are stale.** Before this task: core 341 (not
    308), tier2 193 (not 190), compiler 133, tier1 218, eval 42 — 927 vitest, not 891 — plus 54
    Playwright.

---

### Task 9: Measure the segment-size distribution — BEFORE the bake-off

**Files:**
- Create: `apps/eval/src/driver/segments.ts`
- Test: `apps/eval/test/segments.test.ts`

The replacement latency gate says the wall-clock budget comes from a **measured** segment-size distribution over the tier-0/1 corpus. That measurement does not exist yet, and it must exist before the bake-off runs — otherwise the budget is a guess and the kill rule is as arbitrary as the one it replaced.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { segmentSizeDistribution } from "../src/driver/segments.js";
import { loadCorpus } from "../src/driver/corpus.js";

const FIXTURES = join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures");

describe("segmentSizeDistribution", () => {
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));

  it("reports percentiles over the segments a judge would actually see", () => {
    // Not over all segments: code is excluded by the escalation policy, so
    // including it would size the budget for work tier 2 never does.
    const d = segmentSizeDistribution(items);
    expect(d.count).toBeGreaterThan(0);
    expect(d.p50).toBeLessThanOrEqual(d.p95);
    expect(d.p95).toBeLessThanOrEqual(d.max);
  });

  it("counts characters and words, because the model bills in tokens not items", () => {
    const d = segmentSizeDistribution(items);
    expect(d.p50Chars).toBeGreaterThan(0);
    expect(d.p50Words).toBeGreaterThan(0);
  });

  it("is empty-safe rather than dividing by zero", () => {
    expect(segmentSizeDistribution([]).count).toBe(0);
  });
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/driver/segments.js`.

- [ ] **Step 3: Implement** `segmentSizeDistribution(items)` which segments each corpus item with core's `segmentText`, keeps only the segments the escalation policy would select, and returns `{count, p50, p95, max, p50Chars, p95Chars, p50Words, p95Words}`.

Then **run it and write the numbers into the plan's Task 12 budget**, replacing the placeholder there. Report the distribution in the commit message so the budget's provenance is in the history.

- [ ] **Step 4: Verify pass** — 17 passed, not the 3 this plan predicted; the percentile arithmetic needs hand-computed fixtures of its own and the escalation wiring needs both halves of the union.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): measure the segment-size distribution that sizes the tier-2 budget"
```

**What Task 9 measured, and five places this plan was wrong.**

*The measurement.* `apps/eval/src/driver/segments.ts`, over `corpora/fixtures/smoke.jsonl` — the
only corpus in this repository — keeping the segments spec 4.1's escalation policy selects, with
`hasPredicates: true` and no priors, which is the input the bake-off runs under:

| | value |
|---|---|
| corpus items | 13 |
| segments produced | 19 |
| segments **selected** | 17 (two code fences excluded) |
| characters | **p50 62**, p95 153, max 153, min 22 |
| words | **p50 9**, p95 25, max 25, min 3 |
| selected segments per message | **p50 1**, p95 2, max 2 |

The whole ascending sample, in characters, because 17 numbers fit and a percentile nobody can
re-derive is not a measurement: `22, 28, 36, 39, 39, 45, 48, 49, 62, 65, 70, 71, 72, 74, 78, 130,
153`. Supplying the real tier-0 findings (`runTier0` under `apps/eval/fixtures/minimal-ir.json`)
adds exactly one segment — the 66-character AWS-key fence, escalated by an entropy finding at 0.7 —
taking the count to 18 and **segments per message to a maximum of 3**. It moves neither the median
nor the maximum size.

*Corrections:*

1. **The plan's field list for this distribution names both `p50` and `p50Chars`, and a bare `p50`
   next to a `p50Chars` states no unit.** Shipped as `chars`, `words` and `perItem`, each a
   `{p50, p95, max, min}`, so every number carries the unit it is in. The result also carries
   `samples`, `segmentsTotal`, and the `escalation` input it ran under: a distribution measured at
   `hasPredicates: false` describes a different pipeline, and the number should not be quotable
   without the input that produced it.

2. **`perItem` is not in the plan's field list and is the number the budget most needs.** The
   orchestrator arms ONE deadline over the whole `judge()` call, and a judge spends one engine call
   per selected segment, so a size distribution alone cannot say what a message costs. It is also
   not `count / items`: a mean cannot say whether one message costs three calls.

3. **Task 9 cannot import `selectSegments` from `packages/tier2/src/escalate.ts`, and does not need
   to.** That file is a bare re-export of `packages/core/src/detect/escalate.ts` — core is where the
   definition had to live, because `detect` is the caller and core cannot depend on the tier it
   gates. `apps/eval` already depends on `@sih/core`, so importing from there is importing the same
   function object Task 7 landed (`packages/tier2/test/escalate.test.ts` asserts the identity).
   Adding an `@sih/eval` -> `@sih/tier2` dependency would have bought nothing and cost the pull-in of
   `@mlc-ai/web-llm`, which is WebGPU-only, into a Node driver.

4. **`maxTokens: 512` is not what will kill an arm on this corpus, and raising it is the wrong
   response to the observed truncations.** See the amended kill-rules section: the largest honest
   answer a 153-character passage can require is ~243 characters — and, since that passage is ASCII,
   243 UTF-8 bytes, which is under 512 tokens at the true one-token-per-byte ceiling.

5. **The 8192 context window is not required by tier 2.** The largest tier-2 prompt this corpus can
   produce is 1,196 characters, and about as many UTF-8 bytes: its segment is ASCII (measured, 153
   code units and 153 bytes) and so is the fixed system turn, leaving only the predicate text able
   to add any. Bytes, not characters, are the unit that ceilings a token count — see the last bullet
   below. The lift is Approach B's requirement, and should be attributed to it.

*Established, and it is the uncomfortable one:*

6. **Tier 2 does not fit `ir.latencyBudgetMs` on this corpus, and no arm choice changes that.**
   Every IR fixture here carries `latencyBudgetMs: 5000`. The segments-per-message figure depends on
   whether tier 0's own findings supply the uncertainty, and the two conditions must not be mixed —
   an earlier version of this bullet paired a p95 from one with a maximum from the other, which no
   single distribution can produce, since at n = 13 the nearest-rank p95 IS the maximum
   (`ceil(0.95 × 13) = 13`). RE-MEASURED, both conditions: with tier-0 priors, the realistic one
   because tier 0 runs first, **p50 1, p95 3, max 3**; with no priors at all, **p50 1, p95 2,
   max 2**. At this plan's own 4.6 s per call on the CHEAPEST pinned arm, an escalating message
   therefore costs **13.8 s at p95 under priors** (2.8x the budget) and 9.2 s at p95 without them
   (1.8x). `detect` will file `budget-exhausted` or leave scopes unjudged on
   most such messages. Task 12 must count those notices per arm and report them beside the gate
   verdict; it must not turn them into a kill rule, which would kill all four arms for a reason that
   is not about capability — the exact failure the amended rules were written to avoid.

*Two limits on every number above, stated because the budget rests on them:*

- **n = 13 items and 17 segments, all hand-authored, none longer than 153 characters.** At n = 17 the
  nearest-rank 95th percentile IS the maximum — `ceil(0.95 x 17) = 17`, the last rank — so the p95
  above carries nothing the max does not, and that is a property of the sample size rather than of
  the corpus. This is a smoke fixture whose own README calls it a pipe-integrity check. It sizes a
  budget for the bake-off that runs on it; **it does not size a budget for real messages**, and
  Plan 7's corpus must be re-measured with the same function before any of these numbers is carried
  forward.
- **Sizes are characters, UTF-8 bytes and words, never tokens.** No pinned arm's tokenizer is cached
  on this machine and inventing a chars-per-token ratio would put a fabricated number under the
  budget. The one token claim these numbers do support needs no ratio, and it is the BYTE column:
  a byte-level BPE tokenizes UTF-8 bytes, so its floor is one token per byte. CHARACTERS are not a
  ceiling — an earlier version of this bullet said they were. MEASURED against a real byte-level
  BPE cached in this repo (`packages/tier1/models/gliner-pii-edge/tokenizer.json`, `model.type` BPE,
  pre-tokenizer ByteLevel) through `@huggingface/transformers`: 20 `x` give 4 tokens, but 20 U+1F389
  give 60 tokens from 40 UTF-16 code units, and 20 CJK characters give 21 tokens from 20 code units.
  Two selected segments of this very corpus are non-ASCII, so its byte median is 65 against a
  character median of 62. Nothing downstream moves here, because the LARGEST segment is ASCII — but
  on Plan 7's corpus redo the arithmetic in bytes, which `segmentSizeDistribution` now reports
  directly.

---

### Task 10: Harness integration — engine lifecycle and the persistent profile

**Files:**
- Modify: `apps/eval/src/page/main.ts`, `apps/eval/playwright.config.ts`
- Test: `apps/eval/test/tier2.spec.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { expect, test } from "@playwright/test";

test.describe.configure({ mode: "serial", timeout: 600_000 });

test("loads a tier-2 model in real Chrome and reports what actually loaded", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const available = await page.evaluate(() => window.__sih!.webgpuAvailable());
  test.skip(!available, "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded");

  const report = await page.evaluate(() => window.__sih!.loadTier2({ modelId: "Qwen3.5-2B-q4f16_1-MLC" }));
  // MEASURED by Task 5's review: this assertion CANNOT pass as written.
  // 0.2.84 exposes no accessor for the loaded model -- the only channel is
  // ChatCompletion.model -- so `loadedModelId` is undefined until the engine
  // has answered at least once, and `loadTier2` has not asked it anything.
  //
  // Do NOT repair this by seeding the field from the requested id: that is
  // the intent-as-fact defect Plan 4 had to fix twice, and it would leave the
  // comment above standing there green and false. Either run one throwaway
  // completion as part of the load report and assert what the engine then
  // reports, or record `requestedModelId` under a name that says so.
  expect(report.loadedModelId).toBe("Qwen3.5-2B-q4f16_1-MLC");
  expect(report.contextWindowSize).toBe(8192);
});

test("a judged message produces findings whose spans core accepts", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  test.skip(!(await page.evaluate(() => window.__sih!.webgpuAvailable())), "no WebGPU");
  await page.evaluate(() => window.__sih!.loadTier2({ modelId: "Qwen3.5-2B-q4f16_1-MLC" }));

  const r = await page.evaluate(() =>
    window.__sih!.detect({
      text: "Please review the Northwind Traders renewal before Friday.",
      provider: "claude",
      config: { tier0: false, tier1: false, tier2: true },
    }),
  );
  // Reaching this line at all means every offset survived normalizeFindings.
  for (const f of r.findings) expect(f.tier).toBe(2);
  expect(r.timings.tier2Ms).toBeGreaterThan(0);
  // Quality is NOT asserted. Measured, these models miss most entities; a
  // recall assertion here would make the suite a model-quality gate and would
  // fail for reasons that are the finding, not a regression.
  console.log(`[tier2] ${JSON.stringify(r.findings.map((f) => f.text))}`);
});

test("the engine survives a deadline expiry", async ({ page }) => {
  // The wedge, end to end. Measured under Node with a fake in Task 3; this is
  // the same property against the real engine, which is where it was first
  // observed.
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  test.skip(!(await page.evaluate(() => window.__sih!.webgpuAvailable())), "no WebGPU");
  await page.evaluate(() => window.__sih!.loadTier2({ modelId: "Qwen3.5-2B-q4f16_1-MLC" }));

  const survived = await page.evaluate(async () => {
    await window.__sih!.detectWithBudget({ text: "Write an essay about clouds.", budgetMs: 50 }).catch(() => null);
    const after = await window.__sih!.detect({
      text: "Please review the Northwind Traders renewal.",
      provider: "claude",
      config: { tier0: false, tier1: false, tier2: true },
    });
    return after !== null;
  });
  expect(survived).toBe(true);
});
```

- [ ] **Step 2: Run to verify failure** — `window.__sih.loadTier2 is not a function`.

- [ ] **Step 3: Implement**

Extend `SihPageApi` with `webgpuAvailable()`, `loadTier2(overrides)` returning a report `{loadedModelId, contextWindowSize, loadMs}`, and `detectWithBudget({text, budgetMs})`.

**`playwright.config.ts` must use a persistent context.** Measured: 2 of 8 cold model loads succeed on an ephemeral context versus 3 of 3 persistent — the rest fail with `QuotaExceededError`. A flaky load looks exactly like a model defect, so this is a correctness matter. Add a smoke assertion that `navigator.storage.estimate().quota > 8e9`.

Keep `workers: 1` (Plan 4 established that parallel workers race over model loading), and note that a 2-4 GB cold load needs a timeout far above Playwright's default.

- [ ] **Step 4: Verify pass** — 3 passed, or skipped with a legible reason when WebGPU is unavailable.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): tier-2 engine lifecycle on a persistent browser profile"
```

---

### Task 11: Record the tier-2 evidence the JSONL is missing

**Files:**
- Modify: `apps/eval/src/driver/record.ts`, `apps/eval/src/driver/run.ts`
- Test: `apps/eval/test/record.test.ts`

The spec amendment added reporting requirements this schema does not carry. Without them a Plan 8 scorer cannot distinguish an arm that judged well from one that failed closed on every segment.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { RunRecordSchema } from "../src/driver/record.js";

const base = { /* a valid tier-0 record, as in the existing tests */ } as never;

describe("RunRecordSchema tier-2 fields", () => {
  it("requires tier2Stats exactly when tier 2 ran and the item did not throw", () => {
    // Same coupling the tier-1 stats use: absent is the only honest answer for
    // a thrown item, because the page's last-detect delta then belongs to the
    // PREVIOUS item.
    expect(RunRecordSchema.safeParse({ ...base, config: { tier0: true, tier1: false, tier2: true }, error: null }).success).toBe(false);
  });

  it("carries the rung distribution, so guessed spans are visible", () => {
    // A recovered span is sliced from the message, so a MIS-LOCATED span is
    // schema-valid and passes core's fidelity check. The rung is the only
    // signal that separates strong evidence from weak.
    const rec = { ...base, config: { tier0: false, tier1: false, tier2: true }, error: null,
      tier2Stats: { rung1: 3, rung2: 1, unresolvedQuotes: 2, unknownPredicates: 0,
                    repairAttempts: 1, failedClosed: 0, deadlineExpiries: 0, duplicatesDropped: 4,
                    finishReason: "stop", promptTokens: 412, completionTokens: 96, ttftMs: 780 } };
    expect(RunRecordSchema.safeParse(rec).success).toBe(true);
  });

  it("rejects a negative counter", () => {
    const rec = { ...base, config: { tier0: false, tier1: false, tier2: true }, error: null,
      tier2Stats: { rung1: -1, rung2: 0, unresolvedQuotes: 0, unknownPredicates: 0,
                    repairAttempts: 0, failedClosed: 0, deadlineExpiries: 0, duplicatesDropped: 0,
                    finishReason: "stop", promptTokens: 1, completionTokens: 1, ttftMs: 1 } };
    expect(RunRecordSchema.safeParse(rec).success).toBe(false);
  });

  it("carries finishReason, so truncation is distinguishable from a short answer", () => {
    // A model that stopped at max_tokens produced a partial judgement.
    // Scoring it as complete understates recall with nothing to notice.
    const rec = { ...base, config: { tier0: false, tier1: false, tier2: true }, error: null,
      tier2Stats: { rung1: 0, rung2: 0, unresolvedQuotes: 0, unknownPredicates: 0,
                    repairAttempts: 0, failedClosed: 1, deadlineExpiries: 0, duplicatesDropped: 0,
                    finishReason: "length", promptTokens: 4000, completionTokens: 512, ttftMs: 3900 } };
    expect(RunRecordSchema.safeParse(rec).success).toBe(true);
  });
});
```

- [ ] **Step 2: Run to verify failure** — the schema has no `tier2Stats`.

- [ ] **Step 3: Implement** — add `tier2Stats` to `RunRecordSchema`, coupled to `config.tier2 && error === null`, carrying every counter above plus `finishReason`, `promptTokens`, `completionTokens`, `ttftMs`. Populate it in `runArm` **in the round trip it already makes**, exactly as `tier1Stats` is.

  **Counters are not sufficient, and this step as written does not deliver what Task 12 needs.** `RunRecordSchema` has no `degraded` field and `runArm` projects `DetectionResult` field by field (`findings`, `timings`), so the array is silently dropped — adding it to `DetectionResult` produced no type error and no output change. Three of the five reason words have no counter equivalent: `absent` and `scope-unjudged` are the orchestrator's own facts and no judge counts them, and the budget-spent-before-start path makes **no judge call at all**, so nothing is there to count. Carry `degraded` onto the record alongside `tier2Stats`, or Task 12's table still cannot separate the two arms this whole channel exists to separate: "failed closed on 40% of messages" and "found nothing".

- [ ] **Step 4: Verify pass** — 4 passed; root suite green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): carry the tier-2 rung distribution and token accounting into the JSONL"
```

---

### Task 12: The four-arm bake-off driver

**Files:**
- Create: `apps/eval/src/driver/bakeoff.ts`
- Test: `apps/eval/test/bakeoff.spec.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { expect, test } from "@playwright/test";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runBakeoff, GATES } from "../src/driver/bakeoff.js";
import { RunRecordSchema } from "../src/driver/record.js";

test("runs the slate cheapest-first and writes one file per arm", async ({ page }) => {
  const out = mkdtempSync(join(tmpdir(), "sih-bakeoff-"));
  const written = await runBakeoff(page, {
    runId: "bake", outDir: out,
    corpus: join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures", "smoke.jsonl"),
    models: ["Qwen3.5-2B-q4f16_1-MLC"],
    provider: "claude", itemTimeoutMs: 120_000,
  });
  expect(written).toHaveLength(1);
  for (const line of readFileSync(written[0]!, "utf8").trim().split("\n")) {
    expect(RunRecordSchema.safeParse(JSON.parse(line)).success).toBe(true);
  }
});

test("the gates are the amended ones, not the spec's original pair", () => {
  // The originals would have killed all four arms for reasons unrelated to
  // capability: JSON compliance is vacuous under grammar constraint (0
  // malformed in 77 calls) and NO model achieves p95 3 s (cheapest is 4.6 s).
  expect(GATES.maxP95TtftMs).toBe(1500);
  expect(GATES.minDecodeTokPerSec).toBe(25);
  expect(GATES).not.toHaveProperty("maxP95LatencyMs");
  expect(GATES).not.toHaveProperty("minJsonCompliance");
});

test("records a killed arm rather than omitting it", async () => {
  // An arm that fails a gate is a RESULT. Dropping it from the output makes
  // the bake-off look like it had fewer contenders than it did.
  expect(GATES.recordKilledArms).toBe(true);
});
```

- [ ] **Step 2: Run to verify failure** — cannot find module `../src/driver/bakeoff.js`.

- [ ] **Step 3: Implement**

`runBakeoff(page, options)` runs each model in `TIER2_MODELS` order (cheapest first), one engine per arm on a fresh page, writing `<runId>.tier2-<modelId>.jsonl`.

Export `GATES`:

```ts
export const GATES = {
  /**
   * p95 time-to-first-token at the MEASURED tier-2 segment size.
   *
   * Task 9 measured that size over `corpora/fixtures/smoke.jsonl`, the only
   * corpus in this repository, keeping the 17 of 19 segments the escalation
   * policy selects: p50 62 characters, max 153, and 9 words at the median.
   * Assembled by `buildMessages` with the one semantic predicate the repo's
   * compiled extraction fixture carries, that is a WHOLE PROMPT of 1,105
   * characters at the median segment and 1,196 at the largest. So a TTFT
   * measured against a prompt materially bigger than ~1.2 kB is not measuring
   * this gate, and an arm must not be killed on a number taken at a different
   * prompt size.
   */
  maxP95TtftMs: 1500,
  /** Sustained decode rate. Latency here is dominated by output length. */
  minDecodeTokPerSec: 25,
  /** Semantic correctness: fraction of findings resolving at rung <= 2. */
  minResolvableRate: 0.8,
  /** An arm returning only duplicates has found nothing. */
  maxDuplicateRate: 0.5,
  /** A killed arm is a result and stays in the output. */
  recordKilledArms: true,
  /**
   * After ANY deadline expiry, the next call must return a non-empty body.
   * Task 3 measured the engine latching its interrupt flag on the pinned
   * non-streaming path, after which every later call returns instantly and
   * empty -- which a judge reads as "no findings" forever. `clearInterrupt`
   * fixes it by writing a field TypeScript marks private, so an upstream
   * rename would silently restore the poisoning with a green unit suite.
   * This assertion is the only guard that would notice.
   */
  assertNonEmptyAfterExpiry: true,
} as const;
```

**Compute the gates, do not enforce them by dropping data.** Every arm writes its file; the gate verdict is a field. Plan 8 decides what to do with a killed arm.

**The wall-clock ceiling, derived from Task 9's measured distribution rather than guessed.** The
number that has to be sized is `itemTimeoutMs`, and `run.ts` is explicit that it exists to catch a
wedge and not to enforce a latency target, so it is sized from the worst case and doubled:

```
  3 selected segments   -- Task 9 max over smoke.jsonl WITH tier-0 priors (2 without)
x 2 calls per segment   -- the pinned recipe's one repair retry
x ~10 s per call        -- Plan 5 measured 4.6 s on Qwen3.5-2B (33-46 tok/s decode);
                           Phi-4-mini decodes at 20-29, so ~1.6x, rounded up for a first call
= ~60 s, doubled       -> itemTimeoutMs: 120_000
```

The 120,000 in the Step-1 snippet is therefore right, and now has a derivation behind it. Two
honesty notes on the inputs: the 4.6 s was measured at a prompt size nobody recorded, while these
prompts are ~1.2 kB, so the per-call term is if anything generous — which is the safe direction for
a wedge catcher; and 3 segments is both the maximum and the nearest-rank p95 over 13 items under
tier-0 priors, not a tail — at n = 13 those are the same rank.

**What this ceiling does NOT do is make tier 2 fit `ir.latencyBudgetMs`, and the bake-off must
record that rather than gate on it.** The orchestrator arms ONE deadline over the whole `judge()`
call from what is left of the message budget, and a judge spends one engine call per selected
segment. Every IR fixture in this repo carries `latencyBudgetMs: 5000`. Task 9's
segments-per-message figure has two conditions and they must not be mixed: with tier-0 priors (the
realistic one, since tier 0 runs before tier 2) p50 1, p95 3, max 3; with no priors p50 1, p95 2,
max 2. At n = 13 the nearest-rank p95 is the maximum, so no single distribution has p95 2 and max 3.
At the plan's own 4.6 s per call on the CHEAPEST arm, one message costs **13.8 s at p95 under
priors — 2.8x the entire budget** — and 9.2 s at p95 without them. So on this corpus tier 2 expires
mid-run on most messages that escalate more than one segment, whatever arm wins.
Derive no kill rule from that: it would kill all four arms for a reason that is not about capability,
which is exactly why the spec's original rules were replaced. Count the `budget-exhausted` and
`scope-unjudged` notices per arm and report them beside the gate verdict, the same way the token
budget's truncations are reported.

- [ ] **Step 4: Verify pass** — 3 passed; full root suite and typecheck green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): four-arm tier-2 bake-off against the amended gates"
```

---

## Done criteria for this plan

A local instruct LLM runs in a real browser as tier 2, judging the semantic predicates a compiled policy declares, emitting shadow entityTypes core can resolve actions for, with spans recovered by a ladder that refuses ambiguity rather than guessing and reports which rung it needed. Cancellation interrupts and drains, so an over-budget run degrades to the lower tiers instead of wedging the engine for every message after it. Approach B runs the same corpus through the same interface with the whole policy and no compiler, and **B + tier 0** runs between them, so the head-to-head separates "compiling the policy helps" from "having deterministic patterns helps". Four model arms are measured against gates derived from what the hardware actually does.

**Explicitly NOT in this plan:** the real corpus (Plan 7), any metric or plot (Plan 8), the extension (Plan 6). The smoke corpus remains a pipe-integrity check and cannot produce meaningful accuracy in either direction.

**Carried forward as known-unretired risks:**

- **Model capability is the live failure, not backend correctness.** Measured, these models miss most entities, duplicate findings, and one false-positived `"The weather is nice today."` as a secret. Expect the bake-off to be a search for *any* usable arm rather than a ranking of good ones.
- **xgrammar issue #807** (`Invalid token id` on Apple Silicon) did not reproduce in 36 constrained calls, which does not retire an intermittent bug. Instrument the dev-slice run with an error counter over at least 200 constrained calls; if it fires, fall back to unconstrained JSON plus the repair retry, which is the regime the original kill rule was written for.
- **One unexplained determinism break.** In one session three identical constrained calls at temperature 0 returned 175 / 588 / 599 completion tokens. Three later sessions were 26/26 byte-identical. Unexplained is not benign — log completion-token counts per call so a recurrence is visible.
- **Upstream issue #844** (prefill over 120 tokens throwing) did not reproduce here across 36 calls at 1,480-4,360 prompt tokens, but was reported on integrated AMD/Windows. It is a portability risk, not a local one.

**Next plans:** 6 — the extension; 7 — corpus pipeline; 8 — evaluation and analysis.

## Deviations log

Append an entry per task: what the plan said, what you did instead, and why. Several claims in this plan are measurements from a specific machine on a specific day; if one fails to reproduce, that is a finding and it belongs here.

### Tasks 10-12 review round: three of this plan's own numbers were wrong

**`GATES.maxDuplicateRate` is 0.9, not the 0.5 in this plan's Task 12 snippet.**
0.5 has no derivation anywhere -- not in the snippet, not in spec 4.2, which
asks for "no duplicate-**only** output" and gives no number. Applied to this
plan's own measurement of the arm it recommends as primary -- Qwen3.5-2B "found
only the AWS key, three times over", i.e. one distinct span and two restatements
-- the observed rate is 0.667 and a 0.5 ceiling **kills that arm**, for exactly
the behaviour this plan tells the bake-off to expect ("expect duplicates, expect
misses, and do not tune the corpus to hide either"). The new ceiling is
bracketed by the two duplicate behaviours this plan measured: above 0.667, below
the ~0.95 of Phi-4-mini's `"quote": "Halcyon"` loop. Two points on one message
each are not a distribution and the constant says so.

`minResolvableRate` stays at 0.8 and is now documented as a **chosen** floor
rather than a derived one, because no run in this repository has produced a
distribution of resolvable rates to derive it from. That is a gap against this
plan's done criterion "gates derived from what the hardware actually does": two
of the four numeric gates are derived (`minDecodeTokPerSec` from `manifest.ts`'s
per-arm rates; `maxP95TtftMs` from the measured prompt size), and two are not.

**The `resolvable-rate` gate was computed over the wrong population.** Read from
`WebLlmJudge.#collect`: `resolveQuote` runs BEFORE the duplicate check, so a
dropped duplicate is a quote the ladder DID place. The denominator omitted it
from both halves, so an arm whose ladder works but whose model restates itself
was reported as an arm whose quotes do not resolve -- a different diagnosis, and
one `duplicate-rate` already covers.

**The prompt size the p95 TTFT gate is derived at is 1,031/1,122 characters,
not the 1,105/1,196 this plan states.** Re-measured here by driving the real
`WebLlmJudge` over this corpus's median and largest segments with a capturing
engine: the plan's figures are correct *under its own stated condition* ("with
the one semantic predicate the repo's compiled extraction fixture carries"),
whose predicate line is 74 characters longer than `semantic-ir.json`'s. The
bake-off is hard-required to run `semantic-ir.json` -- `planBakeoff` throws on
any IR with no `semanticPredicates` -- so 1,105/1,196 describe work no arm of
this bake-off can perform. The same measurement reproduces both pairs exactly,
one predicate each; the fixed system turn is 776 characters, matching the plan.

**`RunRecordSchema` gained `tier2Config`,** which Task 11 did not specify. The
record carried `config.t2Model` and nothing else about tier 2, so two arms
differing only in context window, token ceiling, temperature or per-call budget
emitted rows byte-identical in every field a scorer can group by -- and this
plan's own fallback for a model that cannot take 8,192 ("run that arm at 4,096
and report the asymmetry") was unexpressible in the output. `ArmGateReport`
gained the matching `run` block for the same reason: the gates file could not be
joined to the IR, the corpus or the run id it came from.
