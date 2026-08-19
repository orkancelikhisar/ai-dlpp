# Tier-1 Span Tagger and Playwright Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run policy-conditioned named-entity detection with a GLiNER-class ONNX model in a real browser, and measure the whole detection pipeline on the exact runtime the extension will ship.

**Architecture:** A new `packages/tier1` implements core's existing `SpanTagger` interface using onnxruntime-web plus a transformers.js tokenizer; labels are injected at inference from `ir.entityTypes` filtered to `tier === 1`, which is the policy-adaptivity mechanism — no retraining, ever. A new `apps/eval` is a Vite page that imports `@sih/core` unmodified and a Playwright driver that loads it in real Chrome, feeds a corpus, and writes JSONL. The harness is built FIRST against tier 0 alone, so it is working, measurable software before any model is involved.

**Tech Stack:** onnxruntime-web (WASM + WebGPU EPs), @huggingface/transformers (tokenizer only), Vite, Playwright (chromium), vitest, TypeScript.

---

## Why this shape

Three constraints from earlier work bind every task here. Read them before writing code.

**1. The harness must run the same code as the extension.** Spec §2.2: detection numbers measured on onnxruntime-node would describe software nobody runs, and WebGPU latency cannot be measured from Node at all. So the harness is Playwright driving real Chrome loading the real `@sih/core`. `apps/eval` may not fork, stub, or re-implement any detection logic.

**2. `normalizeFindings` throws on span drift, and it is aimed directly at this plan.** From `packages/core/src/detect/orchestrator.ts`:

> `types.ts` states `text === message.slice(start, end)` as a contract and `SpanTagger` tells implementers to re-derive it, but nothing enforced it, and **a tokenizer-backed model breaks it routinely**.

A tier-1 finding whose offsets are off by one does not degrade quality — it throws, and `detect()` fails the whole message. Two independent offset translations happen in this plan (subword token → character within a segment, and segment-local → absolute message) and both must be exact. Tasks 6 and 10 exist for this and nothing else.

**3. Tier-1 entityTypes are the ONLY ones the tagger may emit.** `ir.entityTypes` mixes tiers. Sending tier-0 entities to the model double-detects every PAN and credential; emitting an entityType absent from the IR throws in `normalizeFindings`. The label set is `ir.entityTypes.filter(e => e.tier === 1)` and the decoder maps model class index → that exact id.

**Deviation from spec §2.2, decided here:** the spec's layout diagram puts the tier-1 tagger inside `packages/core/detect/`. It cannot live there. Core's tsconfig is `lib: ["ES2022"]` with no DOM, and `packages/core/test/firewall.test.ts` enforces a Node/DOM-free source tree; onnxruntime-web and transformers.js both require DOM and WebGPU types. Core keeps the `SpanTagger` *interface* — which is the correct seam and already exists — and the implementation lives in `packages/tier1`. Record this in the Deviations log at Task 1.

---

## File structure

```
packages/tier1/                  NEW — browser-only, DOM types allowed
  src/
    labels.ts        IR → label set + class-index ↔ entityType id map   (pure, Node-tested)
    offsets.ts       subword token spans → character spans              (pure, Node-tested)
    decode.ts        model output tensor → segment-local spans          (pure, Node-tested)
    session.ts       OnnxSession seam + onnxruntime-web implementation  (browser)
    tokenizer.ts     Tokenizer seam + transformers.js implementation    (browser)
    tagger.ts        GlinerSpanTagger implements core's SpanTagger      (browser)
    config.ts        Tier1Config: model id, backend, threshold, maxWidth
    index.ts         barrel
  models/            gitignored — fetched by scripts/fetch-models.ts
  test/              vitest, Node — pure modules only

apps/eval/                       NEW
  src/page/main.ts   the page: imports @sih/core, exposes window.__sih
  src/page/index.html
  src/driver/run.ts      Playwright driver: corpus in → JSONL out
  src/driver/corpus.ts   corpus JSONL schema + loader
  src/driver/record.ts   JSONL record schema (the TS↔Python boundary)
  test/              vitest + Playwright
  playwright.config.ts

scripts/fetch-models.ts          NEW — checksummed model download
corpora/fixtures/smoke.jsonl     NEW — 13-item hand-authored corpus
```

**Boundary discipline:** `apps/eval` emits JSONL and computes **no metrics**. Spec §2.2 makes the TS/Python boundary a JSONL file; scoring lives in `analysis/` (Plan 8). Harness tests assert on JSONL *contents*, never on a computed F1.

---

### Task 1: Eval app scaffold — core running in real Chrome

**Files:**
- Create: `apps/eval/package.json`, `apps/eval/tsconfig.json`, `apps/eval/vite.config.ts`
- Create: `apps/eval/src/page/index.html`, `apps/eval/src/page/main.ts`
- Create: `apps/eval/playwright.config.ts`
- Test: `apps/eval/test/smoke.spec.ts`

The point of this task is one fact: `@sih/core` executes unmodified in real Chrome. Everything later rests on it.

- [ ] **Step 1: Write the failing test**

`apps/eval/test/smoke.spec.ts`:
```ts
import { expect, test } from "@playwright/test";

test("core detects a tier-0 entity inside real Chrome", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);

  const result = await page.evaluate(async () => {
    return window.__sih.detect({
      text: "My PAN is AFTPD1298Q, please help.",
      provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
    });
  });

  expect(result.findings).toHaveLength(1);
  expect(result.findings[0].entityType).toBe("in-pan");
  // Offsets are absolute into the message and must survive the structured-clone
  // boundary between the page and the driver intact.
  expect(result.findings[0].text).toBe("AFTPD1298Q");
  expect(typeof result.timings.tier0Ms).toBe("number");
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C apps/eval exec playwright test`
Expected: FAIL — no config / page does not exist.

- [ ] **Step 3: Implement**

`apps/eval/package.json`:
```json
{
  "name": "@sih/eval",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "test": "playwright test",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@sih/core": "workspace:*"
  },
  "devDependencies": {
    "@playwright/test": "^1.50.0",
    "@types/node": "^20.0.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0"
  }
}
```

`apps/eval/tsconfig.json`:
```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["node"],
    "moduleResolution": "bundler",
    "noEmit": true
  },
  "include": ["src", "test", "playwright.config.ts", "vite.config.ts"]
}
```

`apps/eval/vite.config.ts`:
```ts
import { defineConfig } from "vite";

export default defineConfig({
  root: "src/page",
  server: { port: 5178 },
  // Cross-origin isolation. onnxruntime-web needs SharedArrayBuffer for
  // multi-threaded WASM in Task 11; setting it now means the page the smoke
  // test measures is the same page the model test measures later, rather than
  // a different one whose numbers do not transfer.
  plugins: [
    {
      name: "coop-coep",
      configureServer(server) {
        server.middlewares.use((_req, res, next) => {
          res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
          res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
          next();
        });
      },
    },
  ],
});
```

`apps/eval/src/page/index.html`:
```html
<!doctype html>
<html>
  <head><meta charset="utf-8" /><title>sih eval</title></head>
  <body>
    <div id="status">loading</div>
    <script type="module" src="./main.ts"></script>
  </body>
</html>
```

`apps/eval/src/page/main.ts`:
```ts
import { detect, loadPolicyIr, type DetectionResult, type TierConfig } from "@sih/core";
import irJson from "../../../../packages/core/test/fixtures/minimal-ir.json?raw";

export interface DetectRequest {
  text: string;
  provider: string;
  config: TierConfig;
}

/**
 * The page's whole API surface. Playwright reaches detection ONLY through this,
 * so the harness cannot accidentally measure a re-implementation: everything
 * below `detect` is core, imported unmodified.
 */
export interface SihPageApi {
  detect(request: DetectRequest): Promise<DetectionResult>;
}

declare global {
  interface Window {
    __sih: SihPageApi;
  }
}

const ir = loadPolicyIr(irJson);

window.__sih = {
  detect: ({ text, provider, config }) => detect({ ir, provider, text, config }),
};

document.getElementById("status")!.textContent = "ready";
```

`apps/eval/playwright.config.ts`:
```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "test",
  use: { baseURL: "http://localhost:5178" },
  webServer: {
    command: "pnpm vite",
    port: 5178,
    reuseExistingServer: !process.env.CI,
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
```

Install: `pnpm -C apps/eval add -D @playwright/test vite typescript @types/node` then `pnpm -C apps/eval exec playwright install chromium`.

If `packages/core/test/fixtures/minimal-ir.json` does not exist as a JSON file (Plan 1 may export it from a `.ts` module), create the JSON file from that fixture rather than importing the `.ts` — the page must load an IR the same way the extension will, through `loadPolicyIr` on a JSON string.

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C apps/eval exec playwright test`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): Playwright harness running core in real Chrome"
```

---

### Task 2: JSONL record schema — the TS/Python boundary

**Files:**
- Create: `apps/eval/src/driver/record.ts`
- Create: `apps/eval/src/driver/corpus.ts`
- Create: `corpora/fixtures/smoke.jsonl`
- Test: `apps/eval/test/record.test.ts`

This schema is what Plan 8's Python reads. Getting it wrong is expensive later, so it is pinned by tests now.

- [ ] **Step 1: Write the failing test**

`apps/eval/test/record.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CorpusItemSchema, loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema } from "../src/driver/record.js";

const FIXTURES = join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures");

describe("corpus", () => {
  it("loads the smoke corpus, rejecting any malformed line by number", () => {
    const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));
    expect(items.length).toBeGreaterThanOrEqual(12);
    expect(items.every((i) => CorpusItemSchema.safeParse(i).success)).toBe(true);
  });

  it("names the line number when a record is malformed", () => {
    expect(() => loadCorpus('{"id":"a","text":"x","policy":"p-fin","gold":[]}\n{"id":\n')).toThrow(
      /line 2/,
    );
  });

  it("requires every gold span to quote the text it claims", () => {
    // The corpus is ground truth. A gold span whose offsets do not hold the
    // text it names would silently score every arm against fiction — the same
    // failure normalizeFindings refuses for findings, applied to labels.
    const bad = { id: "a", text: "hello world", policy: "p-fin", gold: [{ start: 0, end: 5, text: "WRONG", entityType: "client-name", action: "block" }] };
    expect(CorpusItemSchema.safeParse(bad).success).toBe(false);
  });
});

describe("RunRecordSchema", () => {
  it("accepts a record carrying everything Plan 8 needs to score without re-running", () => {
    const record = {
      schemaVersion: 1,
      runId: "r1",
      itemId: "a",
      policy: "p-fin",
      policyHash: "0".repeat(64),
      arm: "t0",
      backend: "wasm",
      provider: "claude",
      findings: [{ start: 0, end: 5, text: "hello", entityType: "client-name", severity: "high", tier: 0, source: "rule", confidence: 0.9, action: "block" }],
      gold: [{ start: 0, end: 5, text: "hello", entityType: "client-name", action: "block" }],
      timings: { tier0Ms: 0.4 },
      error: null,
    };
    expect(RunRecordSchema.safeParse(record).success).toBe(true);
  });

  it("carries the policy hash, so a record can never be attributed to the wrong IR", () => {
    const { policyHash: _omitted, ...withoutHash } = {
      schemaVersion: 1, runId: "r1", itemId: "a", policy: "p-fin", policyHash: "0".repeat(64),
      arm: "t0", backend: "wasm", provider: "claude", findings: [], gold: [],
      timings: { tier0Ms: 0 }, error: null,
    };
    expect(RunRecordSchema.safeParse(withoutHash).success).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C apps/eval exec vitest run test/record.test.ts`
Expected: FAIL — cannot find module `../src/driver/corpus.js`.

- [ ] **Step 3: Implement**

`apps/eval/src/driver/corpus.ts`:
```ts
import { z } from "zod";

export const GoldSpanSchema = z.object({
  start: z.number().int().nonnegative(),
  end: z.number().int().positive(),
  text: z.string().min(1),
  entityType: z.string().min(1),
  /** What the policy demands for this span. "none" means the span is not a violation under this policy. */
  action: z.enum(["none", "allow", "pseudonymize", "redact", "block"]),
});

export const CorpusItemSchema = z
  .object({
    id: z.string().min(1),
    text: z.string().min(1),
    policy: z.string().min(1),
    gold: z.array(GoldSpanSchema),
    /** Free-form provenance from Plan 7 (carrier source, injection dimensions). Passed through untouched. */
    meta: z.record(z.string(), z.unknown()).optional(),
  })
  .refine(
    (item) => item.gold.every((g) => g.end <= item.text.length && item.text.slice(g.start, g.end) === g.text),
    { message: "a gold span's offsets do not hold the text it names" },
  );

export type CorpusItem = z.infer<typeof CorpusItemSchema>;

/**
 * Parses JSONL. Blank lines are skipped; every other line must be a valid item.
 * The line number is in the error because a 1,500-item corpus with one bad line
 * is otherwise unfixable.
 */
export function loadCorpus(jsonl: string): CorpusItem[] {
  const items: CorpusItem[] = [];
  const lines = jsonl.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!.trim();
    if (line === "") continue;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch (cause) {
      throw new Error(`corpus line ${i + 1} is not valid JSON`, { cause });
    }
    const parsed = CorpusItemSchema.safeParse(raw);
    if (!parsed.success) {
      throw new Error(`corpus line ${i + 1} is not a valid corpus item: ${z.prettifyError(parsed.error)}`);
    }
    items.push(parsed.data);
  }
  return items;
}
```

`apps/eval/src/driver/record.ts`:
```ts
import { z } from "zod";
import { GoldSpanSchema } from "./corpus.js";

/**
 * Bump when a field is removed or its meaning changes. Plan 8's Python reads
 * this first and refuses a version it does not know, so a silently reshaped
 * record cannot be scored as if it were the old one.
 */
export const RECORD_SCHEMA_VERSION = 1;

export const RunRecordSchema = z.object({
  schemaVersion: z.literal(RECORD_SCHEMA_VERSION),
  runId: z.string().min(1),
  itemId: z.string().min(1),
  policy: z.string().min(1),
  /**
   * sha256 of the policy document the IR was compiled from. Required, not
   * optional: an arm's numbers are meaningless without knowing exactly which
   * compiled policy produced them, and "which IR was that?" is unanswerable
   * after the fact.
   */
  policyHash: z.string().regex(/^[0-9a-f]{64}$/),
  arm: z.string().min(1),
  backend: z.enum(["wasm", "webgpu"]),
  provider: z.string().min(1),
  findings: z.array(
    z.object({
      start: z.number().int().nonnegative(),
      end: z.number().int().positive(),
      text: z.string(),
      entityType: z.string(),
      severity: z.string(),
      tier: z.number().int(),
      source: z.string(),
      confidence: z.number(),
      action: z.string(),
    }),
  ),
  /** Copied from the corpus item so a record scores standalone, without a join. */
  gold: z.array(GoldSpanSchema),
  timings: z.object({
    tier0Ms: z.number(),
    tier1Ms: z.number().optional(),
    tier2Ms: z.number().optional(),
  }),
  /**
   * Set when detection THREW for this item. The record is still written: an arm
   * that crashes on 5% of the corpus and one that scores 0 on it are different
   * results, and dropping the row makes them look identical.
   */
  error: z.string().nullable(),
});

export type RunRecord = z.infer<typeof RunRecordSchema>;

export function toJsonl(records: readonly RunRecord[]): string {
  return records.map((r) => JSON.stringify(r)).join("\n") + "\n";
}
```

`corpora/fixtures/smoke.jsonl` — hand-authored items, one per line. At least six negatives with `"gold": []` and six positives whose gold spans quote their text exactly. (Task 2 shipped 13: seven negatives, six positives — the extra negative carries a non-client proper noun so tier-1 over-firing is measurable at all.) Cover: a PAN in prose, an AWS key in a code fence, a value in a `key=value` line, a client name in prose (tier-1, `entityType: "client-name"`), a multi-line message, and a message containing an emoji before the span (so any UTF-16 offset bug surfaces here rather than in Task 6). Example line:

```json
{"id":"pos-pan-prose","text":"Client sent his PAN HGRPS4821M for the KYC file.","policy":"p-fin","gold":[{"start":20,"end":30,"text":"HGRPS4821M","entityType":"in-pan","action":"block"}]}
```

Verify every offset before committing — the schema's refine will reject a wrong one, which is the point.

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C apps/eval exec vitest run test/record.test.ts`
Expected: 5 passed.

Add `vitest` and `zod` to `apps/eval` devDependencies, and a `vitest.config.ts` if the Playwright `testDir` collides with vitest discovery — keep Playwright specs in `test/*.spec.ts` and vitest tests in `test/*.test.ts`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): corpus and JSONL record schemas with span-fidelity checks"
```

---

### Task 3: Driver run loop — corpus in, JSONL out

**Files:**
- Create: `apps/eval/src/driver/run.ts`
- Test: `apps/eval/test/run.spec.ts`

- [ ] **Step 1: Write the failing test**

`apps/eval/test/run.spec.ts`:
```ts
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema } from "../src/driver/record.js";
import { runArm } from "../src/driver/run.js";

const FIXTURES = join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures");

test("runs the smoke corpus and emits one valid record per item", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));
  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    items,
  });

  expect(records).toHaveLength(items.length);
  for (const record of records) {
    expect(RunRecordSchema.safeParse(record).success).toBe(true);
  }
  // Record order matches corpus order, so a diff between two runs lines up.
  expect(records.map((r) => r.itemId)).toEqual(items.map((i) => i.id));
});

test("records a thrown item instead of dropping it", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);
  const records = await runArm(page, {
    runId: "test-run",
    arm: "t0",
    backend: "wasm",
    provider: "claude",
    // tier1 with no engine registered throws inside detect().
    config: { tier0: true, tier1: true, tier2: false },
    items: [{ id: "boom", text: "anything at all", policy: "p-fin", gold: [] }],
  });

  expect(records).toHaveLength(1);
  expect(records[0]!.error).toMatch(/tier1/i);
  expect(records[0]!.findings).toEqual([]);
});

test("refuses an unprepared page rather than silently measuring a blank one", async ({ page }) => {
  // If runArm navigated on its own, Task 12 would load a tier-1 model and then
  // have it thrown away — producing a full JSONL file of tier-0 results
  // labelled as a tier-1 arm. Nothing downstream could detect that.
  await expect(
    runArm(page, {
      runId: "test-run", arm: "t0", backend: "wasm", provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
      items: [{ id: "a", text: "hello", policy: "p-fin", gold: [] }],
    }),
  ).rejects.toThrow(/not prepared/i);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C apps/eval exec playwright test test/run.spec.ts`
Expected: FAIL — cannot find module `../src/driver/run.js`.

- [ ] **Step 3: Implement**

Add to `apps/eval/src/page/main.ts`, replacing the existing `window.__sih` assignment:

```ts
window.__sih = {
  detect: ({ text, provider, config }) => detect({ ir, provider, text, config }),
  policyHash: () => ir.policyHash,
};
```
and add `policyHash(): string;` to `SihPageApi`.

`apps/eval/src/driver/run.ts`:
```ts
import type { Page } from "@playwright/test";
import type { TierConfig } from "@sih/core";
import type { CorpusItem } from "./corpus.js";
import { RECORD_SCHEMA_VERSION, type RunRecord } from "./record.js";

export interface ArmSpec {
  runId: string;
  arm: string;
  backend: "wasm" | "webgpu";
  provider: string;
  config: TierConfig;
  items: readonly CorpusItem[];
}

/**
 * One arm over one corpus.
 *
 * Items run SEQUENTIALLY and in corpus order. Two reasons, both load-bearing:
 * latency is a reported metric and concurrent inference on one GPU would
 * measure contention rather than the model, and stable order makes two runs
 * diffable line by line.
 */
export async function runArm(page: Page, spec: ArmSpec): Promise<RunRecord[]> {
  // Deliberately does NOT navigate. Task 12 loads a tier-1 model into the page
  // before calling this, and a goto() here would discard it and silently
  // measure a tier-0 run under a tier-1 arm label. The caller owns page state.
  const ready = await page.evaluate(() => window.__sih !== undefined);
  if (!ready) {
    throw new Error("page is not prepared: navigate to '/' and await window.__sih before calling runArm");
  }
  const policyHash = await page.evaluate(() => window.__sih.policyHash());

  const records: RunRecord[] = [];
  for (const item of spec.items) {
    // The try lives INSIDE the loop: one item that throws must not end the arm.
    // An arm that dies on item 300 of 1500 would otherwise report as a complete
    // run of 299 items, which scores as a much better arm than it is.
    let findings: RunRecord["findings"] = [];
    let timings: RunRecord["timings"] = { tier0Ms: 0 };
    let error: string | null = null;
    try {
      const result = await page.evaluate(
        ([text, provider, config]) =>
          window.__sih.detect({ text: text as string, provider: provider as string, config: config as never }),
        [item.text, spec.provider, spec.config] as const,
      );
      findings = result.findings as RunRecord["findings"];
      timings = result.timings;
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }

    records.push({
      schemaVersion: RECORD_SCHEMA_VERSION,
      runId: spec.runId,
      itemId: item.id,
      policy: item.policy,
      policyHash,
      arm: spec.arm,
      backend: spec.backend,
      provider: spec.provider,
      findings,
      gold: item.gold,
      timings,
      error,
    });
  }
  return records;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C apps/eval exec playwright test test/run.spec.ts`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): arm run loop emitting one JSONL record per corpus item"
```

---

### Task 4: Tier-1 package scaffold and model acquisition

**Files:**
- Create: `packages/tier1/package.json`, `tsconfig.json`, `src/config.ts`, `src/index.ts`
- Create: `scripts/fetch-models.ts`
- Create: `packages/tier1/models/.gitignore`
- Test: `packages/tier1/test/config.test.ts`

Model weights are ~200 MB and never enter git. This task makes acquisition reproducible and tamper-evident.

- [ ] **Step 1: Write the failing test**

`packages/tier1/test/config.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { MODEL_MANIFEST, resolveTier1Config } from "../src/config.js";

describe("MODEL_MANIFEST", () => {
  it("pins every model by sha256, not by tag", () => {
    // A HuggingFace tag is mutable. Weights that change under a fixed id would
    // silently invalidate every number already measured against them.
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      expect(entry.sha256, id).toMatch(/^[0-9a-f]{64}$/);
      expect(entry.url, id).toMatch(/^https:\/\//);
      expect(entry.bytes, id).toBeGreaterThan(0);
    }
  });

  it("includes the primary and the ladder siblings the experiment matrix names", () => {
    expect(Object.keys(MODEL_MANIFEST)).toEqual(
      expect.arrayContaining(["gliner-pii-edge", "gliner-pii-base"]),
    );
  });
});

describe("resolveTier1Config", () => {
  it("defaults to the primary model on wasm", () => {
    const config = resolveTier1Config({});
    expect(config.modelId).toBe("gliner-pii-edge");
    expect(config.backend).toBe("wasm");
  });

  it("rejects a model id that is not in the manifest, naming what is available", () => {
    expect(() => resolveTier1Config({ modelId: "not-a-model" })).toThrow(/gliner-pii-edge/);
  });

  it("rejects a threshold outside (0,1] rather than silently clamping", () => {
    // A clamped threshold produces a run whose reported config does not match
    // the config that ran — the single worst failure for a measurement tool.
    expect(() => resolveTier1Config({ threshold: 0 })).toThrow(/threshold/);
    expect(() => resolveTier1Config({ threshold: 1.5 })).toThrow(/threshold/);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/tier1 exec vitest run`
Expected: FAIL — cannot find module `../src/config.js`.

- [ ] **Step 3: Implement**

`packages/tier1/package.json`:
```json
{
  "name": "@sih/tier1",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "main": "src/index.ts",
  "scripts": { "test": "vitest run", "typecheck": "tsc --noEmit" },
  "dependencies": {
    "@huggingface/transformers": "^3.3.0",
    "@sih/core": "workspace:*",
    "onnxruntime-web": "^1.21.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.6.0",
    "vitest": "^3.0.0"
  }
}
```

`packages/tier1/tsconfig.json` — same shape as `apps/eval`'s, `lib: ["ES2022", "DOM", "DOM.Iterable"]`.

`packages/tier1/src/config.ts`:
```ts
export interface ModelEntry {
  readonly url: string;
  readonly sha256: string;
  readonly bytes: number;
  /** Tokenizer repo id for transformers.js. Separate from the ONNX url: the two are versioned independently upstream. */
  readonly tokenizer: string;
}

/**
 * Every tier-1 model the experiment matrix can select, pinned by content hash.
 *
 * Fill `sha256` and `bytes` by running scripts/fetch-models.ts, which prints
 * both for a downloaded file. They are deliberately NOT optional — an unpinned
 * entry would let a silently-updated upstream file invalidate measurements that
 * were already taken and reported.
 */
export const MODEL_MANIFEST: Readonly<Record<string, ModelEntry>> = {
  "gliner-pii-edge": {
    url: "https://huggingface.co/knowledgator/gliner-pii-edge-v1.0/resolve/main/onnx/model.onnx",
    sha256: "<fill from fetch-models.ts>",
    bytes: 0,
    tokenizer: "knowledgator/gliner-pii-edge-v1.0",
  },
  "gliner-pii-base": {
    url: "https://huggingface.co/knowledgator/gliner-pii-base-v1.0/resolve/main/onnx/model.onnx",
    sha256: "<fill from fetch-models.ts>",
    bytes: 0,
    tokenizer: "knowledgator/gliner-pii-base-v1.0",
  },
};

export interface Tier1Config {
  readonly modelId: string;
  readonly backend: "wasm" | "webgpu";
  /** Span score below this is discarded. An experiment variable, not a constant. */
  readonly threshold: number;
  /** Widest span in WORDS the model may propose. GLiNER span mode enumerates all widths up to this. */
  readonly maxWidth: number;
}

export const DEFAULT_TIER1_CONFIG: Tier1Config = {
  modelId: "gliner-pii-edge",
  backend: "wasm",
  threshold: 0.5,
  maxWidth: 12,
};

export function resolveTier1Config(overrides: Partial<Tier1Config>): Tier1Config {
  const config = { ...DEFAULT_TIER1_CONFIG, ...overrides };
  if (!Object.hasOwn(MODEL_MANIFEST, config.modelId)) {
    throw new Error(
      `unknown tier-1 model "${config.modelId}"; available: ${Object.keys(MODEL_MANIFEST).join(", ")}`,
    );
  }
  if (!(config.threshold > 0 && config.threshold <= 1)) {
    throw new Error(`tier-1 threshold must be in (0, 1], got ${config.threshold}`);
  }
  if (!(Number.isInteger(config.maxWidth) && config.maxWidth > 0)) {
    throw new Error(`tier-1 maxWidth must be a positive integer, got ${config.maxWidth}`);
  }
  return config;
}
```

`scripts/fetch-models.ts`:
```ts
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { MODEL_MANIFEST } from "../packages/tier1/src/config.js";

const MODELS_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "packages", "tier1", "models");
const PLACEHOLDER = "<fill from fetch-models.ts>";

const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

/**
 * Downloads every manifest entry and verifies it against its pinned hash.
 * Network, no credentials, safe to re-run: a present file whose hash already
 * matches is skipped.
 */
async function main(): Promise<number> {
  mkdirSync(MODELS_DIR, { recursive: true });
  let unpinned = 0;

  for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
    const path = join(MODELS_DIR, `${id}.onnx`);
    if (existsSync(path) && entry.sha256 !== PLACEHOLDER) {
      if (sha256(readFileSync(path)) === entry.sha256) {
        console.log(`${id}: present and verified`);
        continue;
      }
      console.log(`${id}: on-disk hash disagrees with the manifest; re-downloading`);
    }

    console.log(`${id}: downloading ${entry.url}`);
    const response = await fetch(entry.url);
    if (!response.ok) {
      console.error(`${id}: HTTP ${response.status} ${response.statusText} for ${entry.url}`);
      return 1;
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    const digest = sha256(bytes);

    if (entry.sha256 === PLACEHOLDER) {
      // Written, but the manifest stays unpinned until a human pastes the
      // values in. Auto-writing them would defeat the pin entirely: the file
      // would "verify" against whatever it happened to download.
      writeFileSync(path, bytes);
      console.log(`${id}: UNPINNED — paste into MODEL_MANIFEST:\n    sha256: "${digest}",\n    bytes: ${bytes.byteLength},`);
      unpinned += 1;
      continue;
    }
    if (digest !== entry.sha256) {
      console.error(`${id}: hash mismatch — manifest ${entry.sha256}, downloaded ${digest}. NOT written.`);
      return 1;
    }
    writeFileSync(path, bytes);
    console.log(`${id}: verified and written (${bytes.byteLength} bytes)`);
  }

  if (unpinned > 0) {
    console.error(`\n${unpinned} manifest entry/entries unpinned — paste the values above and re-run.`);
    return 1;
  }
  return 0;
}

process.exitCode = await main();
```

`packages/tier1/models/.gitignore` containing `*` — weights never enter git.

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C packages/tier1 exec vitest run`
Expected: 5 passed — with the manifest hashes filled in from a real `pnpm exec vite-node scripts/fetch-models.ts` run. **Do not fabricate hashes**: run the script, paste what it prints. If the download fails or the URL 404s, report it rather than inventing a value — the URL shape above is a starting point, not a verified fact, and `knowledgator` may publish the ONNX under a different filename.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier1): package scaffold with hash-pinned model manifest"
```

---

### Task 5: Label construction from the IR

**Files:**
- Create: `packages/tier1/src/labels.ts`
- Test: `packages/tier1/test/labels.test.ts`

This is the policy-adaptivity mechanism: what the model looks for is read out of the IR at inference time.

- [ ] **Step 1: Write the failing test**

`packages/tier1/test/labels.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import type { EntityType, PolicyIr } from "@sih/core";
import { buildLabels } from "../src/labels.js";

const entity = (over: Partial<EntityType>): EntityType => ({
  id: "client-name",
  tier: 1,
  nlDefinition: "The name of a client organisation.",
  examples: [],
  counterExamples: [],
  severity: "high",
  surrogateKind: "company",
  ...over,
} as EntityType);

const ir = (entityTypes: EntityType[]): PolicyIr => ({ entityTypes } as PolicyIr);

describe("buildLabels", () => {
  it("includes only tier-1 entityTypes", () => {
    // Tier 0 already caught these deterministically. Handing them to the model
    // duplicates every finding and spends latency to do it; tier 2's engine
    // does not exist here at all.
    const labels = buildLabels(
      ir([entity({ id: "client-name", tier: 1 }), entity({ id: "in-pan", tier: 0 }), entity({ id: "pred:x", tier: 2 })]),
    );
    expect(labels.map((l) => l.entityType)).toEqual(["client-name"]);
  });

  it("maps class index to entityType id positionally", () => {
    const labels = buildLabels(ir([entity({ id: "a", tier: 1 }), entity({ id: "b", tier: 1 })]));
    expect(labels[0]!.classIndex).toBe(0);
    expect(labels[1]!.classIndex).toBe(1);
    expect(labels.map((l) => l.entityType)).toEqual(["a", "b"]);
  });

  it("sends a natural-language phrase to the model, never the kebab-case id", () => {
    // GLiNER was trained on phrases like "person" and "credit card number".
    // "client-name" is an identifier, not language, and scores worse. The id
    // stays as the KEY the finding is emitted under; only the prompt differs.
    const labels = buildLabels(ir([entity({ id: "client-name", nlDefinition: "The name of a client organisation." })]));
    expect(labels[0]!.prompt).not.toContain("-");
    expect(labels[0]!.prompt.toLowerCase()).toContain("client");
  });

  it("returns an empty label set when the policy has no tier-1 entities", () => {
    // Not an error: P-MED may legitimately have none, and the tagger must then
    // skip inference entirely rather than call a model with zero classes.
    expect(buildLabels(ir([entity({ tier: 0 })]))).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/tier1 exec vitest run test/labels.test.ts`
Expected: FAIL — cannot find module `../src/labels.js`.

- [ ] **Step 3: Implement**

`packages/tier1/src/labels.ts`:
```ts
import type { PolicyIr } from "@sih/core";

export interface Tier1Label {
  /** Position in the label array handed to the model; the model's class index. */
  readonly classIndex: number;
  /** The IR id this class maps back to. Emitted on the Finding; must exist in ir.entityTypes. */
  readonly entityType: string;
  /** What the model actually sees. Natural language, because that is what GLiNER was trained on. */
  readonly prompt: string;
}

/**
 * Derive the model's label set from the compiled policy.
 *
 * This function IS the policy-adaptivity mechanism named in spec §4.1: a new
 * policy changes what the model looks for with no retraining and no code
 * change, because the labels are read out of the IR at inference time.
 *
 * The prompt form is a CONFIG VARIABLE (`Tier1Config.labelForm`), defaulting to
 * the id-derived phrase. Kebab-case becomes spaces so the model sees language.
 *
 * The default is chosen on measured token cost, NOT on a claim about the label
 * encoder: an id renders to ~2 tokens while a real compiler-authored
 * nlDefinition renders to 26-37, and label text shares one sequence with the
 * message against the model's `max_len`, multiplied by the tier-1 class count.
 *
 * An earlier draft of this plan justified the choice by asserting that GLiNER's
 * label encoder was trained on short noun phrases and that sentences degrade
 * span F1. That was never measured, and Task 5 disproved its premise: both
 * pinned `gliner_config.json` files carry `labels_encoder: null` — these
 * checkpoints have no label encoder at all. Spec §4.1 names `{id, nlDefinition}`,
 * so `id` alone is a genuine deviation, which is exactly why the form is a
 * config value with all three arms reachable rather than a hardcoded choice.
 * What settles it: one corpus, one policy, arms differing only in `labelForm`,
 * compared on span F1 (spec §6.4 metric 2).
 */
export function buildLabels(ir: PolicyIr, config: Tier1Config): readonly Tier1Label[] {
  // `config` is REQUIRED, deliberately. A default here would let a caller that
  // omits it run `id` labels while the run record reports `labelForm:
  // "definition"` -- silently invalidating that arm with nothing to notice.
  return ir.entityTypes
    .filter((entity) => entity.tier === 1)
    .map((entity, classIndex) => ({
      classIndex,
      entityType: entity.id,
      prompt: promptFor(entity, config.labelForm),
    }));
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C packages/tier1 exec vitest run test/labels.test.ts`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier1): derive model label set from the compiled policy"
```

---

### Task 6: Token-to-character offset mapping

**Files:**
- Create: `packages/tier1/src/offsets.ts`
- Test: `packages/tier1/test/offsets.test.ts`

**The highest-risk code in this plan.** `normalizeFindings` throws on a one-character error, and Plan 1 already shipped a UTF-16-vs-codepoint bug in `shannonEntropy` that counted code points and divided by `s.length`.

- [ ] **Step 1: Write the failing test**

`packages/tier1/test/offsets.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { spanFromTokens, type TokenOffset } from "../src/offsets.js";

/** Mirrors what a fast tokenizer's offset_mapping gives: [start, end) in JS string indices. */
const toks = (...pairs: [number, number][]): TokenOffset[] => pairs.map(([start, end]) => ({ start, end }));

describe("spanFromTokens", () => {
  it("spans from the first token's start to the last token's end", () => {
    const text = "call Acme Corp today";
    const span = spanFromTokens(text, toks([0, 4], [5, 9], [10, 14], [15, 20]), 1, 2);
    expect(span).toEqual({ start: 5, end: 14, text: "Acme Corp" });
  });

  it("re-derives text from the string, never from concatenated token pieces", () => {
    // Subword pieces carry markers ("##Corp", "▁Acme") and joining them
    // reconstructs something that is NOT a substring of the message. The
    // contract is text === message.slice(start, end); slicing is the only way
    // to satisfy it.
    const text = "Acme Corporation";
    expect(spanFromTokens(text, toks([0, 4], [5, 16]), 0, 1).text).toBe("Acme Corporation");
  });

  it("survives an emoji before the span", () => {
    // "🙂" is ONE code point and TWO UTF-16 units. Offsets are JS string
    // indices throughout this system because String.prototype.slice is, and
    // core's contract is expressed in slice terms.
    const text = "🙂 Acme Corp";
    const start = text.indexOf("Acme");
    expect(start).toBe(3);
    const span = spanFromTokens(text, toks([0, 2], [3, 7], [8, 12]), 1, 2);
    expect(span).toEqual({ start: 3, end: 12, text: "Acme Corp" });
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
  });

  it("survives a combining mark inside the span", () => {
    const text = "café Ltd";
    const span = spanFromTokens(text, toks([0, 4], [5, 8]), 0, 1);
    expect(text.slice(span!.start, span!.end)).toBe(span!.text);
  });

  it("returns undefined rather than a bad span when indices are out of range", () => {
    // A model may propose a span index past the token array on a truncated
    // input. Returning undefined drops the finding; returning a clamped span
    // sends a wrong offset into normalizeFindings, which throws and kills the
    // whole message.
    expect(spanFromTokens("short", toks([0, 5]), 0, 3)).toBeUndefined();
    expect(spanFromTokens("short", toks([0, 5]), 2, 2)).toBeUndefined();
  });

  it("returns undefined for an inverted or zero-width span", () => {
    expect(spanFromTokens("hello", toks([0, 0], [0, 5]), 0, 0)).toBeUndefined();
  });

  it("offsets a segment-local span into absolute message coordinates", () => {
    const message = "intro line\ncall Acme Corp today";
    const segmentStart = message.indexOf("call");
    const local = spanFromTokens(message.slice(segmentStart), toks([0, 4], [5, 9], [10, 14], [15, 20]), 1, 2)!;
    const absolute = { start: local.start + segmentStart, end: local.end + segmentStart, text: local.text };
    expect(message.slice(absolute.start, absolute.end)).toBe("Acme Corp");
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/tier1 exec vitest run test/offsets.test.ts`
Expected: FAIL — cannot find module `../src/offsets.js`.

- [ ] **Step 3: Implement**

`packages/tier1/src/offsets.ts`:
```ts
export interface TokenOffset {
  /** JS string index, inclusive. */
  readonly start: number;
  /** JS string index, exclusive. */
  readonly end: number;
}

export interface CharSpan {
  readonly start: number;
  readonly end: number;
  readonly text: string;
}

/**
 * Turn a token-index span into a character span over `text`.
 *
 * Every offset in this system is a JS string index — a UTF-16 code unit
 * position — because `String.prototype.slice` is, and core's contract is
 * `text === message.slice(start, end)`. Code-point indices would be defensible
 * in isolation and are wrong here; an emoji before the span shifts every
 * subsequent index by one and `normalizeFindings` throws.
 *
 * Returns `undefined` on anything malformed rather than repairing it. A
 * repaired span is a wrong span that reaches `applyActions`, which rewrites BY
 * SPAN — so a drifted finding sends a neighbouring word to the vault and leaves
 * the real value in the message. Dropping one uncertain finding is strictly
 * better than that, and the caller counts drops.
 */
export function spanFromTokens(
  text: string,
  tokens: readonly TokenOffset[],
  firstToken: number,
  lastToken: number,
): CharSpan | undefined {
  if (!Number.isInteger(firstToken) || !Number.isInteger(lastToken)) return undefined;
  if (firstToken < 0 || lastToken < firstToken) return undefined;
  if (lastToken >= tokens.length) return undefined;

  const start = tokens[firstToken]!.start;
  const end = tokens[lastToken]!.end;
  if (!(Number.isInteger(start) && Number.isInteger(end))) return undefined;
  if (!(start >= 0 && start < end && end <= text.length)) return undefined;

  // Sliced, never assembled from token pieces: subword markers ("##", "▁") do
  // not appear in the message, so a joined string fails the fidelity check even
  // when the offsets are perfect.
  return { start, end, text: text.slice(start, end) };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C packages/tier1 exec vitest run test/offsets.test.ts`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier1): token-to-character span mapping with UTF-16 fidelity"
```

---

### Task 7: Probe the model's real IO signature

**Files:**
- Create: `packages/tier1/scripts/probe-model.ts`
- Create: `packages/tier1/test/fixtures/model-signature.json`
- Test: `packages/tier1/test/signature.test.ts`

**Do not write the decoder against a guessed tensor layout.** GLiNER ONNX exports differ between publishers in input names, tensor ranks, and output shape. This task discovers the truth and commits it; Task 8 decodes against the committed fact.

- [ ] **Step 1: Write the probe script**

`packages/tier1/scripts/probe-model.ts`:
```ts
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

const modelId = process.argv[2] ?? "gliner-pii-edge";
const path = join(dirname(fileURLToPath(import.meta.url)), "..", "models", `${modelId}.onnx`);
const session = await ort.InferenceSession.create(path);

/**
 * onnxruntime-node has moved the metadata accessor between versions
 * (`inputMetadata` as a record vs. an array vs. absent). Names are always
 * available; dims may not be. Record whatever this version gives and note in
 * the Deviations log if dims came back empty — an empty dims array is a real
 * observation, not a failure to record.
 */
const meta = (names: readonly string[], table: unknown) =>
  names.map((name) => {
    const entry = (table as Record<string, { type?: string; shape?: unknown[] }> | undefined)?.[name];
    return { name, type: entry?.type ?? "unknown", dims: entry?.shape ?? [] };
  });

console.log(
  JSON.stringify(
    {
      modelId,
      sha256: createHash("sha256").update(readFileSync(path)).digest("hex"),
      inputs: meta(session.inputNames, (session as never as { inputMetadata?: unknown }).inputMetadata),
      outputs: meta(session.outputNames, (session as never as { outputMetadata?: unknown }).outputMetadata),
    },
    null,
    2,
  ),
);
```

Add `onnxruntime-node` as a tier1 **devDependency** — used only by this script; the measured path stays onnxruntime-web. Run it and save stdout verbatim to `packages/tier1/test/fixtures/model-signature.json`. Expected shape:

```
{
  "modelId": "gliner-pii-edge",
  "sha256": "<from the manifest>",
  "inputs":  [{ "name": "input_ids", "type": "int64", "dims": ["batch", "sequence"] }, ...],
  "outputs": [{ "name": "logits", "type": "float32", "dims": ["batch", "num_words", "max_width", "num_classes"] }]
}
```



- [ ] **Step 2: Write the test that pins it**

`packages/tier1/test/signature.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import signature from "./fixtures/model-signature.json" with { type: "json" };
import { MODEL_MANIFEST } from "../src/config.js";

describe("model signature", () => {
  it("is pinned to the manifest hash it was probed from", () => {
    // A signature recorded against different weights describes a different
    // model. Tying the two together makes a weight swap fail here, loudly,
    // instead of producing silently wrong spans at inference.
    expect(signature.sha256).toBe(MODEL_MANIFEST["gliner-pii-edge"]!.sha256);
  });

  it("names every input the encoder must supply", () => {
    const names = signature.inputs.map((i) => i.name);
    // Assert against the ACTUAL probed names. If the probe shows this model
    // does not take span_idx/span_mask, it is a token-classification GLiNER
    // rather than span-mode — record that in the Deviations log and decode
    // accordingly in Task 8. Do not force the model to match this list.
    expect(names).toContain("input_ids");
    expect(names).toContain("attention_mask");
    expect(names.length).toBeGreaterThan(0);
  });

  it("has exactly one output whose last dimension is the class axis", () => {
    expect(signature.outputs).toHaveLength(1);
    expect(signature.outputs[0]!.dims.length).toBeGreaterThanOrEqual(3);
  });
});
```

- [ ] **Step 3: Run to verify it passes**

Run: `pnpm -C packages/tier1 exec vitest run test/signature.test.ts`
Expected: 3 passed.

If the probe reveals a layout that contradicts the assumptions in Task 8's decoder sketch, **the probe wins**. Rewrite Task 8's decode against the real shape and record the difference in the Deviations log.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(tier1): pin the ONNX model IO signature probed from real weights"
```

---

### Task 8: Decode model output to segment-local spans

**Files:**
- Create: `packages/tier1/src/decode.ts`
- Test: `packages/tier1/test/decode.test.ts`

Pure, Node-tested, no ORT import: takes a plain `Float32Array` plus dimensions.

- [ ] **Step 1: Write the failing test**

`packages/tier1/test/decode.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { decodeSpans } from "../src/decode.js";

/**
 * Builds a logits array shaped [numWords, maxWidth, numClasses] with one hot
 * cell. Widths are 0-indexed: width w covers w+1 words.
 */
function logitsWith(
  dims: { numWords: number; maxWidth: number; numClasses: number },
  hot: { word: number; width: number; cls: number; score: number },
): Float32Array {
  const data = new Float32Array(dims.numWords * dims.maxWidth * dims.numClasses).fill(-10);
  data[(hot.word * dims.maxWidth + hot.width) * dims.numClasses + hot.cls] = hot.score;
  return data;
}

const DIMS = { numWords: 4, maxWidth: 3, numClasses: 2 };

describe("decodeSpans", () => {
  it("returns the span whose score clears the threshold", () => {
    const spans = decodeSpans(logitsWith(DIMS, { word: 1, width: 1, cls: 0, score: 5 }), DIMS, 0.5);
    expect(spans).toEqual([{ firstToken: 1, lastToken: 2, classIndex: 0, score: expect.any(Number) }]);
    expect(spans[0]!.score).toBeGreaterThan(0.5);
  });

  it("applies sigmoid, not softmax, over the class axis", () => {
    // GLiNER span scoring is per-class binary — two entity types can both be
    // present in one message. Softmax would force them to compete and suppress
    // the weaker one, which is a silent recall loss no test downstream sees.
    const data = new Float32Array(1 * 1 * 2);
    data[0] = 2;
    data[1] = 2;
    const spans = decodeSpans(data, { numWords: 1, maxWidth: 1, numClasses: 2 }, 0.5);
    expect(spans).toHaveLength(2);
    expect(spans[0]!.score).toBeCloseTo(spans[1]!.score, 6);
  });

  it("drops everything below the threshold", () => {
    expect(decodeSpans(logitsWith(DIMS, { word: 0, width: 0, cls: 0, score: -5 }), DIMS, 0.5)).toEqual([]);
  });

  it("never proposes a span running past the last word", () => {
    // Width 2 at word 3 would cover words 3..5 in a 4-word input. GLiNER
    // enumerates the full [word × width] grid regardless, so the invalid
    // corner is always present in the tensor and must be rejected here.
    const spans = decodeSpans(logitsWith(DIMS, { word: 3, width: 2, cls: 0, score: 5 }), DIMS, 0.5);
    expect(spans).toEqual([]);
  });

  it("rejects a logits array whose length disagrees with the dimensions", () => {
    // A silent reshape misreads every score at a shifted stride and produces
    // plausible-looking garbage.
    expect(() => decodeSpans(new Float32Array(5), DIMS, 0.5)).toThrow(/length/i);
  });

  it("returns spans sorted by descending score", () => {
    const data = new Float32Array(DIMS.numWords * DIMS.maxWidth * DIMS.numClasses).fill(-10);
    data[(0 * DIMS.maxWidth + 0) * DIMS.numClasses + 0] = 1;
    data[(2 * DIMS.maxWidth + 0) * DIMS.numClasses + 0] = 4;
    const spans = decodeSpans(data, DIMS, 0.5);
    expect(spans.map((s) => s.firstToken)).toEqual([2, 0]);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/tier1 exec vitest run test/decode.test.ts`
Expected: FAIL — cannot find module `../src/decode.js`.

- [ ] **Step 3: Implement**

`packages/tier1/src/decode.ts`:
```ts
export interface SpanDims {
  readonly numWords: number;
  readonly maxWidth: number;
  readonly numClasses: number;
}

export interface DecodedSpan {
  readonly firstToken: number;
  readonly lastToken: number;
  readonly classIndex: number;
  /** Sigmoid of the logit, in (0, 1). Used directly as Finding.confidence. */
  readonly score: number;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Decode GLiNER span-mode logits, shaped [numWords, maxWidth, numClasses] in
 * row-major order, into candidate spans.
 *
 * Per-class SIGMOID, not softmax across classes: GLiNER scores each (span,
 * class) pair as an independent binary decision, and a message may contain a
 * client name and a project codename at once. Softmax would make the labels
 * compete for one probability mass and silently suppress the weaker of two
 * genuine findings.
 *
 * Overlaps are NOT resolved here. Core's merge owns overlap resolution for all
 * tiers together (severity-first, then confidence), and a tier that pre-filtered
 * its own overlaps would hide candidates the merge might have preferred.
 */
export function decodeSpans(logits: Float32Array, dims: SpanDims, threshold: number): DecodedSpan[] {
  const expected = dims.numWords * dims.maxWidth * dims.numClasses;
  if (logits.length !== expected) {
    throw new Error(
      `logits length ${logits.length} disagrees with dimensions ` +
        `[${dims.numWords}, ${dims.maxWidth}, ${dims.numClasses}] (expected ${expected})`,
    );
  }

  const spans: DecodedSpan[] = [];
  for (let word = 0; word < dims.numWords; word += 1) {
    for (let width = 0; width < dims.maxWidth; width += 1) {
      const lastToken = word + width;
      // The [word x width] grid is rectangular, so its bottom-right corner
      // always describes spans past the end of the input. Always present in the
      // tensor, never valid.
      if (lastToken >= dims.numWords) break;
      for (let cls = 0; cls < dims.numClasses; cls += 1) {
        const score = sigmoid(logits[(word * dims.maxWidth + width) * dims.numClasses + cls]!);
        if (score < threshold) continue;
        spans.push({ firstToken: word, lastToken, classIndex: cls, score });
      }
    }
  }
  return spans.sort((a, b) => b.score - a.score);
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C packages/tier1 exec vitest run test/decode.test.ts`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier1): decode span logits with per-class sigmoid scoring"
```

---

### Task 9: Session and tokenizer seams

**Files:**
- Create: `packages/tier1/src/session.ts`, `packages/tier1/src/tokenizer.ts`
- Test: `packages/tier1/test/session.test.ts`

Interfaces plus real implementations. The interfaces let Task 10 be tested in Node with a fake; the implementations are exercised in Chrome in Task 11.

- [ ] **Step 1: Write the failing test**

`packages/tier1/test/session.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { assertSignature } from "../src/session.js";
import signature from "./fixtures/model-signature.json" with { type: "json" };

describe("assertSignature", () => {
  it("accepts a session whose inputs match the pinned signature", () => {
    const names = signature.inputs.map((i) => i.name);
    expect(() => assertSignature({ inputNames: names, outputNames: signature.outputs.map((o) => o.name) })).not.toThrow();
  });

  it("rejects a session missing a pinned input, naming the input", () => {
    // A model swapped underneath a matching filename would otherwise be fed
    // tensors it does not read, and return scores for something else entirely.
    const names = signature.inputs.map((i) => i.name).slice(1);
    expect(() => assertSignature({ inputNames: names, outputNames: signature.outputs.map((o) => o.name) })).toThrow(
      new RegExp(signature.inputs[0]!.name),
    );
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/tier1 exec vitest run test/session.test.ts`
Expected: FAIL — cannot find module `../src/session.js`.

- [ ] **Step 3: Implement**

`packages/tier1/src/session.ts`:
```ts
import * as ort from "onnxruntime-web";
import signature from "../test/fixtures/model-signature.json" with { type: "json" };

export interface OnnxTensorLike {
  readonly data: Float32Array;
  readonly dims: readonly number[];
}

export interface OnnxSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OnnxTensorLike>>;
}

/**
 * Fail fast when the loaded model is not the model the decoder was written
 * against. ORT will happily run a session with an unexpected graph and return a
 * tensor of the wrong shape; decode.ts then reads it at the wrong stride and
 * produces confident nonsense that looks exactly like a real result.
 */
export function assertSignature(session: Pick<OnnxSession, "inputNames" | "outputNames">): void {
  for (const input of signature.inputs) {
    if (!session.inputNames.includes(input.name)) {
      throw new Error(
        `loaded model is missing input "${input.name}" (it has [${session.inputNames.join(", ")}]) — ` +
          `these weights do not match test/fixtures/model-signature.json`,
      );
    }
  }
  for (const output of signature.outputs) {
    if (!session.outputNames.includes(output.name)) {
      throw new Error(
        `loaded model is missing output "${output.name}" (it has [${session.outputNames.join(", ")}])`,
      );
    }
  }
}

export async function createOrtSession(
  modelUrl: string,
  backend: "wasm" | "webgpu",
): Promise<OnnxSession> {
  // Threads help WASM only, and only under cross-origin isolation — which
  // vite.config.ts arranges via COOP/COEP. Requesting them without it makes ORT
  // fall back silently, so the check is explicit rather than hopeful.
  ort.env.wasm.numThreads =
    backend === "wasm" && globalThis.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency ?? 1)
      : 1;

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: [backend],
    graphOptimizationLevel: "all",
  });
  assertSignature(session);

  return {
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    async run(feeds) {
      const result = await session.run(feeds as never);
      const out: Record<string, OnnxTensorLike> = {};
      for (const name of session.outputNames) {
        const tensor = result[name]!;
        out[name] = { data: tensor.data as Float32Array, dims: tensor.dims };
      }
      return out;
    },
  };
}
```

`packages/tier1/src/tokenizer.ts`:
```ts
import { AutoTokenizer } from "@huggingface/transformers";
import type { TokenOffset } from "./offsets.js";

export interface Encoded {
  readonly inputIds: BigInt64Array;
  readonly attentionMask: BigInt64Array;
  /** JS string indices into the text passed to encode() — see offsets.ts. */
  readonly offsets: TokenOffset[];
}

export interface Tokenizer {
  encode(text: string): Promise<Encoded>;
}

export async function createTransformersTokenizer(repoId: string): Promise<Tokenizer> {
  const tokenizer = await AutoTokenizer.from_pretrained(repoId);
  return {
    async encode(text) {
      const encoded = tokenizer(text, { return_offsets_mapping: true, add_special_tokens: true });
      const mapping = (encoded as { offset_mapping?: [number, number][] }).offset_mapping;
      if (mapping === undefined) {
        // A blocker, deliberately not worked around. The alternative — locating
        // each token by searching the text for its surface form — is precisely
        // the drift normalizeFindings throws on, and it fails silently on
        // repeated substrings.
        throw new Error(
          `tokenizer "${repoId}" returned no offset_mapping; character spans cannot be derived ` +
            `without it (see offsets.ts)`,
        );
      }
      return {
        inputIds: encoded.input_ids.data as BigInt64Array,
        attentionMask: encoded.attention_mask.data as BigInt64Array,
        offsets: mapping.map(([start, end]) => ({ start, end })),
      };
    },
  };
}
```

**MEASURED IN TASK 6 AGAINST THE PINNED TOKENIZER — these are facts, not cautions, and each one shipped a wrong span in the plan's original draft:**

1. **Python `tokenizers` returns CODE-POINT offsets, and JS strings are UTF-16.** Measured: `👨` inside a ZWJ family sequence is reported as `(7, 8)`; `slice(7, 8)` in JS is a **lone high surrogate** (verified via `TextEncoder` → `ef bf bd`, i.e. U+FFFD). **You must measure which unit `@huggingface/transformers` uses** — do not assume that because it operates on JS strings it reports UTF-16. If it mirrors the Python convention, the conversion belongs HERE in the tokenizer adapter, not in `offsets.ts`, which is documented as taking JS string indices. Pin the answer with a test over an astral character.
2. **Whitespace is INSIDE the token.** `▁Acme` in `"call Acme Corp today"` is `(4, 9)` → `" Acme"`, not `(5, 9)`. `spanFromTokens` trims this, but the adapter must not "helpfully" pre-adjust as well or the span loses its first real character.
3. **`[CLS]`/`[SEP]` carry `(0, 0)` offsets**, and a *paired* encoding carries a **third `(0,0)` `[SEP]` mid-sequence with sequence B's offsets restarting at 0**. Decide explicitly whether the returned arrays include special tokens, state it in the type, and pin it with a test.
4. **The one hazard nothing downstream can catch:** if a caller passes the full token array (specials included) but indices derived from a text-only view, the one-off shift produces a span that **slices cleanly and names the wrong word** — `normalizeFindings` cannot object because the text does match the offsets. Only the index-0 case is detectable. The token ids and the offsets this adapter returns must therefore be the *same* view, and that must be tested, not assumed.

**Verify at implementation time** that the installed `@huggingface/transformers` returns `offset_mapping` for this tokenizer and that `ort.InferenceSession.create` accepts `"webgpu"` in the installed onnxruntime-web. Both are version-sensitive. If either disagrees, follow the installed library and record it — the code above is written from the documented API, not from a run.

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C packages/tier1 exec vitest run test/session.test.ts`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier1): ONNX session and tokenizer seams with signature assertion"
```

---

### Task 10: GlinerSpanTagger

**Files:**
- Create: `packages/tier1/src/tagger.ts`
- Modify: `packages/tier1/src/index.ts`
- Test: `packages/tier1/test/tagger.test.ts`

Implements core's `SpanTagger`. Tested in Node against a fake session — no model, no browser.

- [ ] **Step 1: Write the failing test**

`packages/tier1/test/tagger.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { detect, loadPolicyIr } from "@sih/core";
import { GlinerSpanTagger } from "../src/tagger.js";
import { fakeSession, fakeTokenizer, tier1Ir } from "./helpers.js";

describe("GlinerSpanTagger", () => {
  it("emits absolute offsets when the segment does not start at zero", async () => {
    // THE bug this class exists to avoid. The model sees one segment and
    // reports offsets into it; findings must be absolute into the message, or
    // normalizeFindings throws and the whole message fails.
    const message = "intro line\ncall Acme Corp today";
    const tagger = new GlinerSpanTagger(fakeSession({ hit: "Acme Corp" }), fakeTokenizer(), { threshold: 0.5, maxWidth: 12 });
    const findings = await tagger.tag(
      [{ kind: "prose", start: message.indexOf("call"), end: message.length, text: message.slice(message.indexOf("call")) }],
      tier1Ir(),
    );
    expect(findings).toHaveLength(1);
    expect(message.slice(findings[0]!.start, findings[0]!.end)).toBe("Acme Corp");
    expect(findings[0]!.text).toBe("Acme Corp");
  });

  it("survives normalizeFindings inside a real detect() call", async () => {
    // The integration that matters: core is the judge of whether these offsets
    // are right, and it throws rather than warning.
    const message = "🙂 please email Acme Corp about the renewal";
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(tier1Ir())),
      provider: "claude",
      text: message,
      config: { tier0: false, tier1: true, tier2: false },
      engines: { tier1: new GlinerSpanTagger(fakeSession({ hit: "Acme Corp" }), fakeTokenizer(), { threshold: 0.5, maxWidth: 12 }) },
    });
    expect(result.findings.map((f) => f.text)).toEqual(["Acme Corp"]);
  });

  it("emits nothing and never calls the model when the policy has no tier-1 entities", async () => {
    let runs = 0;
    const session = fakeSession({ hit: "Acme Corp", onRun: () => { runs += 1; } });
    const tagger = new GlinerSpanTagger(session, fakeTokenizer(), { threshold: 0.5, maxWidth: 12 });
    const findings = await tagger.tag([{ kind: "prose", start: 0, end: 5, text: "hello" }], tier1Ir({ tier: 0 }));
    expect(findings).toEqual([]);
    expect(runs).toBe(0);
  });

  it("drops a span whose offsets do not survive mapping, and does not throw", async () => {
    // A model span past the token array must cost one finding, not the message.
    const tagger = new GlinerSpanTagger(fakeSession({ outOfRange: true }), fakeTokenizer(), { threshold: 0.5, maxWidth: 12 });
    const findings = await tagger.tag([{ kind: "prose", start: 0, end: 5, text: "hello" }], tier1Ir());
    expect(findings).toEqual([]);
  });

  it("labels each finding with the entityType its class index maps to", async () => {
    const tagger = new GlinerSpanTagger(fakeSession({ hit: "Acme Corp", classIndex: 1 }), fakeTokenizer(), { threshold: 0.5, maxWidth: 12 });
    const ir = tier1Ir({ extraTier1Id: "project-codename" });
    const findings = await tagger.tag([{ kind: "prose", start: 0, end: 30, text: "call Acme Corp today about it" }], ir);
    expect(findings[0]!.entityType).toBe("project-codename");
  });

  it("aborts in-flight work when the signal fires", async () => {
    const controller = new AbortController();
    controller.abort();
    const tagger = new GlinerSpanTagger(fakeSession({ hit: "Acme Corp" }), fakeTokenizer(), { threshold: 0.5, maxWidth: 12 });
    await expect(
      tagger.tag([{ kind: "prose", start: 0, end: 30, text: "call Acme Corp today about it" }], tier1Ir(), controller.signal),
    ).resolves.toEqual([]);
  });
});
```

Write `packages/tier1/test/helpers.ts` in this task: `tier1Ir(options)` returns a loadable IR whose `client-name` is tier 1 (plus an optional second tier-1 entity); `fakeTokenizer()` splits on spaces and returns real JS-index offsets; `fakeSession(options)` returns logits that make the requested substring the winning span.

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/tier1 exec vitest run test/tagger.test.ts`
Expected: FAIL — cannot find module `../src/tagger.js`.

- [ ] **Step 3: Implement**

`packages/tier1/src/tagger.ts`:
```ts
import type { Finding, PolicyIr, Segment, SpanTagger } from "@sih/core";
import { buildLabels } from "./labels.js";
import { decodeSpans } from "./decode.js";
import { spanFromTokens } from "./offsets.js";
import type { OnnxSession } from "./session.js";
import type { Tokenizer } from "./tokenizer.js";

export const TIER1_SOURCE = "gliner";

export class GlinerSpanTagger implements SpanTagger {
  constructor(
    private readonly session: OnnxSession,
    private readonly tokenizer: Tokenizer,
    // The WHOLE Tier1Config, not a subset: `buildLabels` needs `labelForm`, and a
    // narrower options bag is how an arm silently runs labels it did not configure.
    private readonly options: Tier1Config,
  ) {}

  async tag(segments: Segment[], ir: PolicyIr, signal?: AbortSignal): Promise<Finding[]> {
    const labels = buildLabels(ir, this.options);
    // No tier-1 entities is a legitimate policy (P-MED may have none), not an
    // error — and running a zero-class model would waste the budget to return
    // nothing.
    if (labels.length === 0) return [];

    const findings: Finding[] = [];
    for (const segment of segments) {
      if (signal?.aborted === true) break;

      const encoded = await this.tokenizer.encode(segment.text);
      const output = await this.session.run({
        input_ids: encoded.inputIds,
        attention_mask: encoded.attentionMask,
        labels: labels.map((l) => l.prompt),
      });
      // Feed names and output name come from the probed signature (Task 7);
      // adjust both to whatever it recorded rather than to this sketch.
      const logits = output["logits"]!;
      const decoded = decodeSpans(
        logits.data,
        {
          numWords: encoded.offsets.length,
          maxWidth: this.options.maxWidth,
          numClasses: labels.length,
        },
        this.options.threshold,
      );

      for (const span of decoded) {
        const local = spanFromTokens(segment.text, encoded.offsets, span.firstToken, span.lastToken);
        // Dropped, not repaired. spanFromTokens already refused to guess; a
        // finding invented here reaches applyActions, which rewrites BY SPAN.
        if (local === undefined) continue;
        const label = labels[span.classIndex];
        if (label === undefined) continue;

        findings.push({
          // Segment-local to absolute. The single most likely place for this
          // whole package to be wrong, and normalizeFindings throws on it.
          start: segment.start + local.start,
          end: segment.start + local.end,
          text: local.text,
          entityType: label.entityType,
          // Re-derived by normalizeFindings from the IR; this value is a
          // placeholder that core overwrites, and must not be trusted here.
          severity: "low",
          tier: 1,
          source: TIER1_SOURCE,
          confidence: span.score,
        });
      }
    }
    return findings;
  }
}
```

Note the `text` field: it comes from `spanFromTokens`, which slices the *segment*, while core checks it against a slice of the *message*. Those agree only when the absolute offsets are right — which is precisely the property under test.

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C packages/tier1 exec vitest run`
Expected: all tier1 tests pass, including the six above.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(tier1): GlinerSpanTagger emitting absolute message offsets"
```

---

### Task 11: Real model in real Chrome

**Files:**
- Modify: `apps/eval/src/page/main.ts`, `apps/eval/package.json`
- Test: `apps/eval/test/tier1.spec.ts`

First execution of the actual weights. Everything before this ran against fakes.

- [ ] **Step 1: Write the failing test**

`apps/eval/test/tier1.spec.ts`:
```ts
import { expect, test } from "@playwright/test";

test.describe.configure({ mode: "serial", timeout: 180_000 });

for (const backend of ["wasm", "webgpu"] as const) {
  test(`tier-1 finds a client name in real Chrome on ${backend}`, async ({ page }) => {
    await page.goto("/");
    await page.waitForFunction(() => window.__sih !== undefined);

    const supported = await page.evaluate(async (b) => window.__sih.backendAvailable(b), backend);
    test.skip(!supported, `${backend} not available in this browser`);

    await page.evaluate(async (b) => window.__sih.loadTier1({ backend: b }), backend);

    const result = await page.evaluate(() =>
      window.__sih.detect({
        text: "Please draft a renewal note for Acme Corporation before Friday.",
        provider: "claude",
        config: { tier0: false, tier1: true, tier2: false },
      }),
    );

    // The assertion is span fidelity, not model quality: reaching this line at
    // all means normalizeFindings accepted every offset the model produced.
    for (const finding of result.findings) {
      expect(finding.tier).toBe(1);
      expect(finding.confidence).toBeGreaterThan(0);
    }
    expect(result.timings.tier1Ms).toBeGreaterThan(0);
    // Recorded, not asserted on: a hard recall assertion here would make the
    // suite a model-quality gate, and quality is Plan 8's measurement over a
    // real corpus, not a one-sentence smoke test.
    console.log(`[${backend}] findings=${JSON.stringify(result.findings.map((f) => f.text))}`);
  });
}
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C apps/eval exec playwright test test/tier1.spec.ts`
Expected: FAIL — `window.__sih.loadTier1 is not a function`.

- [ ] **Step 3: Implement**

Extend `SihPageApi` with:
```ts
backendAvailable(backend: "wasm" | "webgpu"): Promise<boolean>;
loadTier1(options: { backend: "wasm" | "webgpu"; modelId?: string }): Promise<void>;
```

`backendAvailable` returns `true` for `"wasm"`, and for `"webgpu"` checks `navigator.gpu !== undefined` and that `requestAdapter()` resolves non-null. Real detection, not a user-agent guess — CI machines and headless runs frequently lack a GPU adapter, and the test skips rather than failing.

`loadTier1` constructs the tokenizer and session from `@sih/tier1` and stores the tagger in a module-level variable; `detect` then passes `engines: { tier1: tagger }` when `config.tier1` is set. Serve the ONNX from `packages/tier1/models/` — add a Vite alias or static-serve directory so the page fetches it over HTTP exactly as the extension will.

**Two facts established in Task 1 that this task depends on — do not rediscover them the hard way:**

- **Do NOT add `server.fs.allow` to `apps/eval/vite.config.ts`.** It is deliberately unset, and the Vite default — the workspace root — already covers `packages/core`, `packages/tier1/models/`, and everything else in this repo. Setting it **replaces** that default rather than extending it: Task 1 originally set it to `[".", "../../fixtures"]`, which resolved to `[src/page, fixtures, vite/dist/client]` and silently *excluded* `packages/core`, the one directory the page actually reads. That configuration appeared to work only because Vite's import analysis adds resolved modules to `safeModulePaths` — which does not cover a runtime `fetch()`, so a non-module asset under `packages/core` 403'd with the block present and 200'd without it. It was removed in `c1fae3c`, with a comment in the config recording why it stays gone, because the setting reads like an addition. If the ONNX weights ever do 403, the fix is to check what path is being requested, **not** to add an allow-list. Importing the model through Vite (`?url`) avoids the raw-fetch path entirely and is the preferred route.

- **Use a QUANTIZED variant, not the fp32 weights, and measure the choice.** Task 4 established the real artifacts (verified against HuggingFace's LFS oids, not just self-consistent): `gliner-pii-edge` fp32 is **181 MB** and `gliner-pii-base` fp32 is **665 MB**. Each repo also publishes `model_fp16.onnx` (91 MB / 333 MB) and `model_quint8.onnx` (46 MB / 197 MB) at the same revision. 665 MB of fp32 through the WASM backend is not a realistic browser load, and the spec's §4.2 figures ("197 MB uint8 / 330 MB fp16") describe the quantized artifacts rather than the default `model.onnx` this manifest currently pins. Add the quantized variants to `MODEL_MANIFEST` as their own ids so precision becomes an explicit rung of the ladder, and report load time and accuracy per variant rather than silently picking one — the accuracy-vs-latency trade is the experiment, not an implementation detail.

- **`backend` is currently a LABEL that nothing verifies — this task must make it a measurement.** Confirmed in Task 3: `grep -rn "\.backend" packages/core/src/` finds nothing; `TierConfig.backend` is declared in `detect/types.ts` and never read. So today a record's `backend: "webgpu"` asserts only what was *requested*. Combined with the null-adapter finding below, that is two independent paths to reporting WASM latency under WebGPU's name. After creating the session, **verify which execution provider actually initialized** and fail loudly if it is not the requested one — do not let the run continue and label itself. A record must say what executed, not what was asked for.

- **`channel: "chromium"` in `playwright.config.ts` is load-bearing for the WebGPU arm.** Measured on the real COOP/COEP page: Playwright's bundled headless shell exposes `navigator.gpu` but `requestAdapter()` resolves **null**, while `channel: "chromium"` returns a real adapter. onnxruntime-web reads a null adapter as "no WebGPU" and falls back to WASM **silently** — so without this the webgpu arm would report WASM latency under WebGPU's name, which is exactly the runtime substitution spec §2.2 forbids. `backendAvailable("webgpu")` must therefore check `requestAdapter()` resolving non-null, not merely `navigator.gpu !== undefined`.

Add `@sih/tier1` to `apps/eval` dependencies.

**Do not commit model weights.** If `models/` is empty, `loadTier1` must throw a message naming `scripts/fetch-models.ts`, and the Playwright test should fail with that message rather than a 404.

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C apps/eval exec playwright test test/tier1.spec.ts`
Expected: 2 passed, or 1 passed + 1 skipped when no WebGPU adapter exists.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): tier-1 model executing in real Chrome on both backends"
```

---

### Task 12: Arm matrix and JSONL output

**Files:**
- Create: `apps/eval/src/driver/main.ts`
- Modify: `apps/eval/src/driver/run.ts`
- Test: `apps/eval/test/matrix.spec.ts`

Ties it together: run `{t0, t0+t1} × {wasm, webgpu}` over the smoke corpus and write JSONL Plan 8 can read.

- [ ] **Step 1: Write the failing test**

`apps/eval/test/matrix.spec.ts`:
```ts
import { expect, test } from "@playwright/test";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { RunRecordSchema } from "../src/driver/record.js";
import { runMatrix } from "../src/driver/main.js";

test("writes one JSONL file per arm, each line a valid record", async ({ page }) => {
  const out = mkdtempSync(join(tmpdir(), "sih-eval-"));
  const written = await runMatrix(page, {
    runId: "matrix-test",
    outDir: out,
    corpus: join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures", "smoke.jsonl"),
    arms: [{ arm: "t0", backend: "wasm", config: { tier0: true, tier1: false, tier2: false } }],
    provider: "claude",
  });

  expect(written).toHaveLength(1);
  const lines = readFileSync(written[0]!, "utf8").trim().split("\n");
  expect(lines.length).toBeGreaterThanOrEqual(12);
  for (const line of lines) {
    expect(RunRecordSchema.safeParse(JSON.parse(line)).success).toBe(true);
  }
});

test("names each output file by run, arm and backend so two runs never collide", async ({ page }) => {
  const out = mkdtempSync(join(tmpdir(), "sih-eval-"));
  const written = await runMatrix(page, {
    runId: "abc",
    outDir: out,
    corpus: join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures", "smoke.jsonl"),
    arms: [{ arm: "t0", backend: "wasm", config: { tier0: true, tier1: false, tier2: false } }],
    provider: "claude",
  });
  expect(written[0]).toMatch(/abc.*t0.*wasm\.jsonl$/);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C apps/eval exec playwright test test/matrix.spec.ts`
Expected: FAIL — cannot find module `../src/driver/main.js`.

- [ ] **Step 3: Implement**

**Before writing `main.ts`: the record's `config` field must carry the TIER-1 config, not just core's `TierConfig`.** As of Task 5 the ladder has six dimensions — `modelId × precision × backend × threshold × maxWidth × labelForm` — but `RunRecordSchema.config` records only `tier0/1/2`, `t1Model`, `t2Model`, `backend`. So two arms differing in `labelForm`, `threshold` or `maxWidth` emit **byte-identical config** in the JSONL, and Plan 8 cannot tell them apart. That is the same "intent recorded in place of fact" disease the record's own comment diagnoses for `backend`. Extend the schema to carry the resolved `Tier1Config` whenever tier 1 ran, and stamp the object the tagger was actually constructed with — not the one the caller intended.

Note also that `labelForm` should be measured on ONE model and ONE policy rather than crossed into the full matrix: it is a question about label conditioning, not an axis of the accuracy-vs-latency ladder.

`apps/eval/src/driver/main.ts`:
```ts
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Page } from "@playwright/test";
import type { TierConfig } from "@sih/core";
import { loadCorpus } from "./corpus.js";
import { toJsonl } from "./record.js";
import { runArm } from "./run.js";

export interface ArmDefinition {
  readonly arm: string;
  readonly backend: "wasm" | "webgpu";
  readonly config: TierConfig;
  readonly modelId?: string;
}

export interface MatrixOptions {
  readonly runId: string;
  readonly outDir: string;
  readonly corpus: string;
  readonly arms: readonly ArmDefinition[];
  readonly provider: string;
}

/**
 * Run every arm over one corpus, one JSONL file per arm.
 *
 * Arms run SEQUENTIALLY and each starts from a fresh page. Sequential because
 * two models loading at once contend for the same GPU and corrupt every latency
 * number in both files; fresh page because a tagger left over from the previous
 * arm would be measured under the next arm's label — this function owns page
 * state, which is exactly why runArm refuses to navigate on its own.
 */
export async function runMatrix(page: Page, options: MatrixOptions): Promise<string[]> {
  const items = loadCorpus(readFileSync(options.corpus, "utf8"));
  const written: string[] = [];

  for (const arm of options.arms) {
    await page.goto("/");
    await page.waitForFunction(() => window.__sih !== undefined);
    if (arm.config.tier1) {
      await page.evaluate(
        (o) => window.__sih.loadTier1(o as never),
        { backend: arm.backend, modelId: arm.modelId },
      );
    }

    const records = await runArm(page, {
      runId: options.runId,
      arm: arm.arm,
      backend: arm.backend,
      provider: options.provider,
      config: arm.config,
      items,
    });

    // Named by run, arm and backend: two runs of the same arm must never
    // overwrite each other, and a file whose name does not say which arm
    // produced it is unattributable once it leaves this directory.
    const path = join(options.outDir, `${options.runId}.${arm.arm}.${arm.backend}.jsonl`);
    writeFileSync(path, toJsonl(records));
    written.push(path);
  }
  return written;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm -C apps/eval exec playwright test`
Expected: all specs pass.

Then from root: `pnpm test && pnpm typecheck` — everything green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(eval): arm matrix driver writing per-arm JSONL"
```

---

## Done criteria for this plan

A GLiNER-class ONNX model runs inside real Chrome on both WASM and WebGPU, looking for exactly the entity classes the compiled policy declares at tier 1 — changing the policy changes what it looks for, with no retraining and no code change. Every span it produces survives `normalizeFindings` in the same `detect()` the extension will call. The Playwright harness runs an arm matrix over a JSONL corpus and writes JSONL records carrying findings, gold, timings, policy hash, and errors — the complete input Plan 8's Python needs, with no metric computed on the TypeScript side.

**Explicitly NOT in this plan:** tier-2 (Plan 5), the Approach-B baseline (Plan 5), the real corpus (Plan 7), any metric or plot (Plan 8), the extension (Plan 6). The smoke corpus is 13 hand-authored items whose only job is to prove the pipe carries data end to end.

**Three spec items deliberately deferred, found in the spec sweep for this plan — none are silent omissions:**

1. **Web Worker for tier 1.** Spec §4.1 puts the span tagger in a worker. It runs on the main thread here. In the harness nothing else competes for the thread, so a worker would change no measurement; in the extension it matters, because blocking the page for 80 ms per keystroke-triggered scan is user-visible. **Plan 6 owns this**, and `GlinerSpanTagger` is already worker-ready — it depends only on the `OnnxSession` and `Tokenizer` seams, so moving it behind a message port changes no logic in this package.
2. **GLiNER2-PII stretch arm** (`fastino/gliner2-privacy-filter-PII-multi`). Spec §4.2 makes it conditional on an ONNX-web export existing at all. Adding it is a `MODEL_MANIFEST` entry plus whatever the probe in Task 7 reveals about its IO — attempt it only after the primary arm is measured end to end, and report it as future work if the export does not run.
3. **Multi-policy iteration in `runMatrix`.** Spec §6.3 runs every arm across P-FIN/P-MED/P-CORP. The record schema carries `policy` and `policyHash` per row so this costs nothing later, but the harness loads a single fixture IR because **the compiled IRs do not exist yet** — that is Plan 3's deferred live compile. Adding the policy axis is a loop over IRs in `runMatrix` once `policies/compiled/` is populated.

**Carried forward from Plan 3:** the live policy compile is still deferred, so `apps/eval` loads a test-fixture IR rather than a compiled one. When the live compile lands, point `main.ts` at `policies/compiled/p-fin.ir.json` and the `policyHash` in every record becomes the real one automatically — the field exists now precisely so that switch is traceable.

**Next plans:** 5 — tier-2 (WebLLM judge) + Approach-B baseline; 6 — extension (WXT MV3); 7 — corpus pipeline; 8 — eval + analysis.

## Deviations log

Append an entry per task: what the plan said, what you did instead, and why. Task 7's probe is expected to contradict Task 8's decoder sketch in at least the tensor names — that is the probe working as designed, not a failure.
