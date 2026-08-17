# AI-DLPP Plan 3: Policy Compiler + Policy Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `@sih/compiler` — the Node-only CLI that turns a natural-language data-leak policy document into a validated, hash-stamped Policy IR plus a human-auditable compilation report — and author the three policy documents (P-FIN / P-MED / P-CORP) the evaluation depends on.

**Architecture:** Five inspectable stages (extract → ground → predicates → validate → self-test → emit). The frontier model is reached only through an injectable `LlmClient` seam: tests replay committed JSON fixtures and never touch the network; one manual script hits the real API. Extraction uses structured outputs (`output_config.format` + `zodOutputFormat`) so the model returns schema-valid JSON rather than prose we parse. Every candidate must carry a verbatim source quote or it is rejected. Self-test executes the **actual `@sih/core` runtime** against model-generated cases — the compiler depends on core, which is the point.

**Tech Stack:** TypeScript 5 (strict), vitest, zod 4, `@anthropic-ai/sdk`, `claude-opus-5`. Node-only package — `node:` imports are allowed here (the DOM/Node firewall applies to `packages/core` only).

**Spec:** `docs/superpowers/specs/2026-08-13-ai-dlpp-design.md` §3.2 (compiler), §3.1 (IR), §6.1 (policy suite). Note §5.4 carries an implementation-note pointer added in Plan 2 — the Deviations logs of Plans 1–2 supersede stale spec prose.

**Baseline:** `main` @ `187373f` — 276 tests green, typecheck clean. Work on branch `feat/policy-compiler`.

**Established conventions (binding):** strict TDD per task; own-key discipline on record reads; every quirk pinned by a test; comments record decisions with alternatives; deviations recorded in this file's Deviations log; error messages never echo confidential values.

**Decisions this plan resolves (parked since Plan 1):**

- **semanticPredicate → action path.** The compiler mints a **shadow entityType** for every semanticPredicate (id `pred:<predicateId>`, tier 2). Predicate findings then flow through the existing `entityType → action` path with zero runtime changes, and `orchestrator.ts`'s open-decision comment is closed. Chosen over a `Finding.predicateId` field because that would require touching normalize/merge/resolve/apply in four places for the same outcome.
- **entityType ids are outbound-visible** (`[REDACTED:<id>]` ships to the provider). The extraction prompt forbids deriving ids from confidential nouns; validate warns on id/example token overlap; the report lists every id under an explicit audit heading.

---

### Task 1: Compiler package scaffold and the LLM seam

**Files:**
- Create: `packages/compiler/package.json`, `packages/compiler/tsconfig.json`
- Create: `packages/compiler/src/llm/client.ts`, `packages/compiler/src/llm/fixture.ts`
- Test: `packages/compiler/test/llm/fixture.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/llm/fixture.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { FixtureLlmClient, requestHash } from "../../src/llm/fixture.js";
import type { LlmRequest } from "../../src/llm/client.js";

const Shape = z.object({ answer: z.string() });
const req: LlmRequest = { system: "sys", user: "usr", schemaName: "Shape", maxTokens: 100 };

describe("requestHash", () => {
  it("is stable for identical requests", () => {
    expect(requestHash(req)).toBe(requestHash({ ...req }));
  });

  it("changes when any field changes", () => {
    const base = requestHash(req);
    expect(requestHash({ ...req, user: "other" })).not.toBe(base);
    expect(requestHash({ ...req, schemaName: "Other" })).not.toBe(base);
    expect(requestHash({ ...req, maxTokens: 101 })).not.toBe(base);
  });

  it("does not collide across a field boundary", () => {
    // "ab"+"c" and "a"+"bc" must not hash alike — the separator is load-bearing.
    expect(requestHash({ ...req, system: "ab", user: "c" })).not.toBe(
      requestHash({ ...req, system: "a", user: "bc" }),
    );
  });
});

describe("FixtureLlmClient", () => {
  it("replays a recorded response and validates it against the schema", async () => {
    const client = new FixtureLlmClient(new Map([[requestHash(req), { answer: "42" }]]));
    expect(await client.complete(req, Shape)).toEqual({ answer: "42" });
  });

  it("throws a recordable error on a miss, naming the hash", async () => {
    const client = new FixtureLlmClient(new Map());
    await expect(client.complete(req, Shape)).rejects.toThrow(
      new RegExp(`no fixture.*${requestHash(req)}`, "i"),
    );
  });

  it("rejects a fixture that does not match the schema", async () => {
    const client = new FixtureLlmClient(new Map([[requestHash(req), { answer: 42 }]]));
    await expect(client.complete(req, Shape)).rejects.toThrow(/fixture.*schema/i);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/compiler test`
Expected: FAIL — `Cannot find module '../../src/llm/fixture.js'`.

- [ ] **Step 3: Implement**

`packages/compiler/package.json`:
```json
{
  "name": "@sih/compiler",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "main": "src/index.ts",
  "bin": { "sih-compile": "src/cli.ts" },
  "scripts": {
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.110.0",
    "@sih/core": "workspace:*",
    "zod": "^4.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.6.0",
    "vitest": "^3.0.0"
  }
}
```

`packages/compiler/tsconfig.json` — note this package is **Node-only by design**; unlike `core` it may import `node:*`:
```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "lib": ["ES2022"],
    "types": ["node"]
  },
  "include": ["src", "test"]
}
```

`packages/compiler/src/llm/client.ts`:
```ts
import type { z } from "zod";

/**
 * A single frontier-model call. Deliberately narrow: system + user + a named
 * schema. Everything the compiler asks of the model is a structured extraction,
 * so there is no free-text completion path to abuse.
 */
export interface LlmRequest {
  system: string;
  user: string;
  /** Names the response schema; part of the fixture key so schema changes miss. */
  schemaName: string;
  maxTokens: number;
}

/**
 * The compiler's ONLY route to a frontier model (spec §3.2: the sole place a
 * cloud model is called, and it never sees user data). Injectable so tests
 * replay committed fixtures and never touch the network.
 */
export interface LlmClient {
  complete<T>(request: LlmRequest, schema: z.ZodType<T>): Promise<T>;
}
```

`packages/compiler/src/llm/fixture.ts`:
```ts
import { createHash } from "node:crypto";
import type { z } from "zod";
import type { LlmClient, LlmRequest } from "./client.js";

/**
 * Fixture key. NUL-separated because concatenation alone is ambiguous:
 * ("ab","c") and ("a","bc") would otherwise share a key and silently replay
 * each other's response.
 */
export function requestHash(request: LlmRequest): string {
  return createHash("sha256")
    .update(
      [request.system, request.user, request.schemaName, String(request.maxTokens)].join("\u0000"),
    )
    .digest("hex")
    .slice(0, 16);
}

/**
 * Replays committed responses. Tests use this exclusively — a compiler test
 * that reaches the network would be non-deterministic and cost money, so the
 * real client is never constructed inside the suite.
 */
export class FixtureLlmClient implements LlmClient {
  constructor(private readonly fixtures: ReadonlyMap<string, unknown>) {}

  async complete<T>(request: LlmRequest, schema: z.ZodType<T>): Promise<T> {
    const hash = requestHash(request);
    if (!this.fixtures.has(hash)) {
      throw new Error(
        `no fixture for request ${hash} (schema ${request.schemaName}); ` +
          `record it with scripts/record-fixtures.ts`,
      );
    }
    const parsed = schema.safeParse(this.fixtures.get(hash));
    if (!parsed.success) {
      throw new Error(`fixture ${hash} does not match schema ${request.schemaName}: ${parsed.error.message}`);
    }
    return parsed.data;
  }
}
```

- [ ] **Step 4: Verify pass**

Run: `pnpm install` (new workspace package), then `pnpm -C packages/compiler test` (6 passing) and `pnpm typecheck` from root (clean).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(compiler): package scaffold and fixture-replay LLM seam"
```

---

### Task 2: The three policy documents and the provider manifest

**Files:**
- Create: `policies/providers.json`
- Create: `policies/p-fin.md`, `policies/p-med.md`, `policies/p-corp.md`
- Test: `packages/compiler/test/policies.test.ts`

These are the compiler's input and the evaluation's independent variable (spec §6.1). They must **disagree with each other on purpose** — the disagreements are what the policy-adaptivity metric measures.

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/policies.test.ts`:
```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = join(import.meta.dirname, "..", "..", "..", "policies");
const read = (f: string) => readFileSync(join(ROOT, f), "utf8");

describe("policy suite", () => {
  const policies = { "p-fin.md": read("p-fin.md"), "p-med.md": read("p-med.md"), "p-corp.md": read("p-corp.md") };

  it("each policy is substantial prose, not a stub", () => {
    for (const [name, text] of Object.entries(policies)) {
      expect(text.length, name).toBeGreaterThan(1500);
    }
  });

  it("each policy carries numbered clauses the compiler can quote", () => {
    for (const [name, text] of Object.entries(policies)) {
      expect(text, name).toMatch(/§\s?\d/);
    }
  });

  it("the deliberate disagreements are present", () => {
    // Client names: forbidden under P-FIN, explicitly permitted under P-MED.
    expect(policies["p-fin.md"]).toMatch(/client/i);
    expect(policies["p-med.md"]).toMatch(/client organisation names|client names/i);
    // Provider-specific clauses exist ONLY in P-FIN.
    expect(policies["p-fin.md"]).toMatch(/foreign-hosted|non-enterprise/i);
    expect(policies["p-med.md"]).not.toMatch(/foreign-hosted/i);
    expect(policies["p-corp.md"]).not.toMatch(/foreign-hosted/i);
    // Salary figures are a P-CORP-only concern.
    expect(policies["p-corp.md"]).toMatch(/salary/i);
    expect(policies["p-med.md"]).not.toMatch(/salary/i);
  });

  it("provider manifest maps adapter ids with aliases", () => {
    const manifest = JSON.parse(read("providers.json")) as {
      providers: Array<{ id: string; displayName: string; aliases: string[] }>;
    };
    const ids = manifest.providers.map((p) => p.id);
    expect(ids).toEqual(expect.arrayContaining(["chatgpt", "claude", "gemini", "deepseek"]));
    for (const p of manifest.providers) {
      expect(p.aliases.length, p.id).toBeGreaterThan(0);
    }
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/compiler test -- policies`
Expected: FAIL — `ENOENT` on `policies/p-fin.md`.

- [ ] **Step 3: Implement**

`policies/providers.json` — the grounding table the compiler resolves NL provider mentions against:
```json
{
  "manifestVersion": "1",
  "providers": [
    {
      "id": "chatgpt",
      "displayName": "OpenAI ChatGPT",
      "vendor": "OpenAI",
      "hostingRegion": "us",
      "enterpriseAgreement": false,
      "aliases": ["chatgpt", "openai", "gpt", "gpt-4", "gpt-5"]
    },
    {
      "id": "claude",
      "displayName": "Anthropic Claude",
      "vendor": "Anthropic",
      "hostingRegion": "us",
      "enterpriseAgreement": true,
      "aliases": ["claude", "anthropic", "claude enterprise"]
    },
    {
      "id": "gemini",
      "displayName": "Google Gemini",
      "vendor": "Google",
      "hostingRegion": "us",
      "enterpriseAgreement": false,
      "aliases": ["gemini", "google", "bard"]
    },
    {
      "id": "deepseek",
      "displayName": "DeepSeek",
      "vendor": "DeepSeek",
      "hostingRegion": "cn",
      "enterpriseAgreement": false,
      "aliases": ["deepseek", "chinese-hosted", "foreign-hosted"]
    }
  ]
}
```

Author the three policy documents as an admin would write them. Each needs numbered clauses (`§1`, `§2.1`, …), because the compiler's anti-hallucination gate quotes them verbatim. Each must be **≥1500 characters of real prose** — the compiler is being evaluated on its ability to read a realistic document, so stubs invalidate the experiment.

`policies/p-fin.md` — fintech. Must contain, at minimum:
- §1 scope and definitions (what "customer data" means at this firm)
- §2 identifiers: PAN, Aadhaar, bank account numbers, customer IDs — **forbidden outright**
- §3 client and counterparty organisation names — **pseudonymize**
- §4 credentials: API keys, DB connection strings — **blocked**
- §5 **provider clause**: "No customer data of any classification may be sent to non-enterprise or foreign-hosted services." Name DeepSeek and Gemini as examples of services the firm has not contracted with; name Claude as the firm's enterprise agreement.

`policies/p-med.md` — healthcare. Must contain:
- §1 scope (patient confidentiality basis)
- §2 patient identifiers (name, MRN, date of birth, contact) — **blocked**
- §3 clinical details tied to an identifiable patient — **blocked**
- §4 an **explicit permission**: "Client organisation names (hospitals, clinics, and partner institutions) are matters of public record and may be shared freely." This is the deliberate disagreement with P-FIN.
- §5 credentials — blocked
- **No** provider-specific clauses, **no** mention of salary. Their absence is what the adaptivity metric reads.

`policies/p-corp.md` — generic corporate. Must contain:
- §1 scope
- §2 credentials and secrets — **blocked**
- §3 unreleased financial results, forecasts, and M&A activity — a **semantic predicate**, no fixed surface form
- §4 **salary and compensation figures** — the P-CORP-only class
- §5 internal project codenames — **pseudonymize**
- §6 an explicit permissiveness: generic personal names of employees, absent other confidential context, need not be redacted.

Two authoring rules that make compilation testable:
1. **Quote-ability** — every rule must be stated in one self-contained sentence the compiler can lift verbatim. Rules spread across three paragraphs cannot be quoted, and the anti-hallucination gate will reject the candidate.
2. **No identifier naming leaks** — do not write clauses like "Project Titan must never be mentioned." The codename is itself confidential and would end up in an entityType id, which ships outbound in `[REDACTED:<id>]`. Write "internal project codenames" instead.

- [ ] **Step 4: Verify pass**

Run: `pnpm -C packages/compiler test -- policies` — 4 tests pass. If the disagreement assertions fail, the policy prose is wrong, not the test.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(policies): P-FIN, P-MED, P-CORP documents and provider manifest"
```

---

### Task 3: Extract stage with the quote-grounding gate

**Files:**
- Create: `packages/compiler/src/stages/extract.ts`
- Create: `packages/compiler/test/fixtures/llm/` (hand-authored fixtures)
- Test: `packages/compiler/test/stages/extract.test.ts`

The anti-hallucination gate (spec §3.2 stage 1): **every candidate must carry a `sourceQuote` that appears verbatim in the policy document**, after whitespace normalization. A candidate without a grounded quote is dropped and reported.

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/stages/extract.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { extract, groundQuotes, normalizeForQuoteMatch } from "../../src/stages/extract.js";
import { FixtureLlmClient, requestHash } from "../../src/llm/fixture.js";

const POLICY = [
  "# Test Policy",
  "",
  "§1 PAN card numbers must never be shared with any external service.",
  "§2 Client organisation names must be pseudonymized before transmission.",
].join("\n");

describe("normalizeForQuoteMatch", () => {
  it("collapses whitespace and newlines so re-wrapped quotes still match", () => {
    expect(normalizeForQuoteMatch("a  b\n c")).toBe(normalizeForQuoteMatch("a b c"));
  });

  it("preserves character identity (does not lowercase)", () => {
    expect(normalizeForQuoteMatch("PAN")).not.toBe(normalizeForQuoteMatch("pan"));
  });
});

describe("groundQuotes", () => {
  const candidates = [
    { id: "in-pan", sourceQuote: "PAN card numbers must never be shared" },
    { id: "invented", sourceQuote: "Blood type must never be shared" },
  ];

  it("keeps candidates whose quote appears verbatim in the document", () => {
    const { grounded } = groundQuotes(POLICY, candidates);
    expect(grounded.map((c) => c.id)).toEqual(["in-pan"]);
  });

  it("rejects candidates whose quote does not appear (anti-hallucination gate)", () => {
    const { rejected } = groundQuotes(POLICY, candidates);
    expect(rejected).toHaveLength(1);
    expect(rejected[0]!.id).toBe("invented");
    expect(rejected[0]!.reason).toMatch(/not found/i);
  });

  it("tolerates a quote re-wrapped across lines", () => {
    const { grounded } = groundQuotes(POLICY, [
      { id: "client-name", sourceQuote: "Client organisation names must be\n  pseudonymized" },
    ]);
    expect(grounded).toHaveLength(1);
  });

  it("rejects an empty or whitespace-only quote", () => {
    const { rejected } = groundQuotes(POLICY, [{ id: "blank", sourceQuote: "   " }]);
    expect(rejected[0]!.reason).toMatch(/empty/i);
  });
});

describe("extract", () => {
  it("calls the model once and grounds every candidate it returns", async () => {
    const request = {
      system: expect.any(String),
      user: expect.any(String),
      schemaName: "Extraction",
      maxTokens: 16000,
    };
    void request; // shape documented; the fixture map below is keyed by the real hash
    const client = new FixtureLlmClient(loadTestFixtures());
    const result = await extract(client, POLICY);
    expect(result.entityTypes.map((e) => e.id)).toContain("in-pan");
    expect(result.rejected).toEqual([]);
  });
});
```

Plus a `loadTestFixtures()` helper in `packages/compiler/test/fixtures/index.ts` that reads every `*.json` in `test/fixtures/llm/` into a `Map<hash, unknown>` (filename = hash). **Hand-author** the fixture for this task's extraction call: run the test once, read the hash from the miss error, then write `test/fixtures/llm/<hash>.json` containing a realistic model response. Task 9's live script re-records these against the real API.

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/compiler test -- extract`
Expected: FAIL — module not found; then, after implementing, a fixture-miss error naming the hash to record.

- [ ] **Step 3: Implement**

`packages/compiler/src/stages/extract.ts`:
```ts
import { z } from "zod";
import type { LlmClient } from "../llm/client.js";

/**
 * The model's raw output shape. Every element carries a sourceQuote — that is
 * the whole anti-hallucination mechanism (spec §3.2 stage 1): a candidate the
 * model cannot ground in the document is a candidate the model invented.
 */
export const ExtractionSchema = z.object({
  entityTypes: z.array(
    z.object({
      id: z.string().min(1),
      tier: z.union([z.literal(0), z.literal(1), z.literal(2)]),
      nlDefinition: z.string().min(1),
      examples: z.array(z.string()),
      counterExamples: z.array(z.string()),
      severity: z.enum(["low", "medium", "high", "critical"]),
      surrogateKind: z.enum(["person-name", "org-name", "id-number", "opaque"]).optional(),
      neverPseudonymize: z.boolean().optional(),
      sourceQuote: z.string(),
    }),
  ),
  rules: z.array(
    z.object({
      id: z.string().min(1),
      entityType: z.string().min(1),
      regex: z.string().optional(),
      validator: z.string().optional(),
      contextBoost: z.array(z.string()).optional(),
      entropyThreshold: z.number().optional(),
      minLength: z.number().optional(),
      sourceQuote: z.string(),
    }),
  ),
  semanticPredicates: z.array(
    z.object({
      id: z.string().min(1),
      nlPredicate: z.string().min(1),
      scope: z.enum(["segment", "message"]),
      severity: z.enum(["low", "medium", "high", "critical"]),
      sourceQuote: z.string(),
    }),
  ),
  actions: z.array(
    z.object({
      entityType: z.string().min(1),
      action: z.enum(["allow", "pseudonymize", "redact", "block"]),
      /** Natural-language provider mention; grounded to adapter ids in Task 4. */
      providerMention: z.string().optional(),
      sourceQuote: z.string(),
    }),
  ),
  failMode: z.enum(["open", "closed"]),
});

export type Extraction = z.infer<typeof ExtractionSchema>;

const SYSTEM = [
  "You compile a natural-language data-leak policy into a structured detection IR.",
  "",
  "Rules you must follow:",
  "1. Every item you emit carries a sourceQuote: a span copied VERBATIM from the policy",
  "   document that states the rule. Items without a grounded quote are discarded, so an",
  "   invented rule is wasted output. Quote the sentence, not the section heading.",
  "2. entityType ids are OUTBOUND-VISIBLE: they ship to the LLM provider inside redaction",
  "   markers like [REDACTED:<id>]. Never derive an id from a confidential noun. Write",
  "   'internal-codename', not 'project-titan'. Ids are lowercase kebab-case.",
  "3. Assign each entityType a tier: 0 for anything with a fixed surface form a regex can",
  "   match (identifiers, key prefixes), 1 for named entities needing a span model,",
  "   2 for classes with no surface form at all.",
  "4. Credentials and secrets get neverPseudonymize: true. A format-valid fake credential",
  "   is a lie waiting to be pasted somewhere.",
  "5. Emit regexes as data only. For checksum logic name a validator from this fixed list:",
  "   luhn, verhoeff, pan-structure, jwt-shape. Never invent a validator name.",
  "6. A regex must never match the empty string.",
  "7. Semantic predicates are for classes with no surface form (unreleased financials,",
  "   clinical narrative). Do not emit a regex for those.",
].join("\n");

export interface RejectedCandidate {
  id: string;
  kind: string;
  reason: string;
}

/** Whitespace-insensitive, case- and character-sensitive. */
export function normalizeForQuoteMatch(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

/**
 * The anti-hallucination gate. Case sensitivity is deliberate: a model that
 * changed "PAN" to "pan" did not copy the document, it paraphrased it, and a
 * paraphrase is exactly the failure this gate exists to catch.
 */
export function groundQuotes<T extends { id: string; sourceQuote: string }>(
  document: string,
  candidates: readonly T[],
  kind = "candidate",
): { grounded: T[]; rejected: RejectedCandidate[] } {
  const haystack = normalizeForQuoteMatch(document);
  const grounded: T[] = [];
  const rejected: RejectedCandidate[] = [];
  for (const candidate of candidates) {
    const needle = normalizeForQuoteMatch(candidate.sourceQuote);
    if (needle.length === 0) {
      rejected.push({ id: candidate.id, kind, reason: "sourceQuote is empty" });
    } else if (!haystack.includes(needle)) {
      rejected.push({ id: candidate.id, kind, reason: "sourceQuote not found in policy document" });
    } else {
      grounded.push(candidate);
    }
  }
  return { grounded, rejected };
}

export interface ExtractResult extends Extraction {
  rejected: RejectedCandidate[];
}

export async function extract(client: LlmClient, document: string): Promise<ExtractResult> {
  const raw = await client.complete(
    { system: SYSTEM, user: document, schemaName: "Extraction", maxTokens: 16000 },
    ExtractionSchema,
  );

  const entityTypes = groundQuotes(document, raw.entityTypes, "entityType");
  const rules = groundQuotes(document, raw.rules, "rule");
  const semanticPredicates = groundQuotes(document, raw.semanticPredicates, "semanticPredicate");
  const actions = groundQuotes(
    document,
    raw.actions.map((a) => ({ ...a, id: `${a.entityType}:${a.providerMention ?? "default"}` })),
    "action",
  );

  return {
    entityTypes: entityTypes.grounded,
    rules: rules.grounded,
    semanticPredicates: semanticPredicates.grounded,
    actions: actions.grounded,
    failMode: raw.failMode,
    rejected: [
      ...entityTypes.rejected,
      ...rules.rejected,
      ...semanticPredicates.rejected,
      ...actions.rejected,
    ],
  };
}
```

- [ ] **Step 4: Verify pass** — all extract tests green; full suite green; typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(compiler): extract stage with verbatim quote-grounding gate"
```

---

### Task 4: Ground stage — provider resolution

**Files:**
- Create: `packages/compiler/src/stages/ground.ts`
- Test: `packages/compiler/test/stages/ground.test.ts`

Maps NL provider mentions ("Chinese-hosted services", "our enterprise Claude agreement") to adapter ids via the manifest. An unresolvable mention is a **warning that keeps the default action**, never a silent drop — a dropped provider clause is a silent policy downgrade.

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/stages/ground.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { groundProviders, loadManifest } from "../../src/stages/ground.js";

const MANIFEST = loadManifest({
  manifestVersion: "1",
  providers: [
    { id: "claude", displayName: "Anthropic Claude", vendor: "Anthropic", hostingRegion: "us", enterpriseAgreement: true, aliases: ["claude", "anthropic"] },
    { id: "deepseek", displayName: "DeepSeek", vendor: "DeepSeek", hostingRegion: "cn", enterpriseAgreement: false, aliases: ["deepseek", "foreign-hosted", "chinese-hosted"] },
  ],
});

describe("groundProviders", () => {
  const actions = [
    { entityType: "client-name", action: "pseudonymize" as const, sourceQuote: "q" },
    { entityType: "client-name", action: "redact" as const, providerMention: "Chinese-hosted services", sourceQuote: "q" },
    { entityType: "client-name", action: "allow" as const, providerMention: "our enterprise Claude agreement", sourceQuote: "q" },
  ];

  it("routes an unqualified action to the default map", () => {
    const { defaults } = groundProviders(MANIFEST, actions);
    expect(defaults["client-name"]).toBe("pseudonymize");
  });

  it("resolves an alias mention to its adapter id", () => {
    const { providerOverrides } = groundProviders(MANIFEST, actions);
    expect(providerOverrides["deepseek"]!["client-name"]).toBe("redact");
  });

  it("resolves a mention containing a provider name among other words", () => {
    const { providerOverrides } = groundProviders(MANIFEST, actions);
    expect(providerOverrides["claude"]!["client-name"]).toBe("allow");
  });

  it("warns rather than dropping an unresolvable mention", () => {
    const { warnings, providerOverrides } = groundProviders(MANIFEST, [
      { entityType: "client-name", action: "block" as const, providerMention: "Foocorp AI", sourceQuote: "q" },
    ]);
    expect(warnings[0]).toMatch(/Foocorp AI/);
    expect(Object.keys(providerOverrides)).toHaveLength(0);
  });

  it("prefers the longest matching alias when several match", () => {
    // "chinese-hosted" and "deepseek" both belong to deepseek; a mention naming
    // both must resolve once, not twice.
    const { providerOverrides } = groundProviders(MANIFEST, [
      { entityType: "x", action: "block" as const, providerMention: "DeepSeek and other Chinese-hosted services", sourceQuote: "q" },
    ]);
    expect(Object.keys(providerOverrides)).toEqual(["deepseek"]);
  });

  it("rejects a manifest with duplicate ids or aliases", () => {
    expect(() =>
      loadManifest({
        manifestVersion: "1",
        providers: [
          { id: "a", displayName: "A", vendor: "A", hostingRegion: "us", enterpriseAgreement: false, aliases: ["x"] },
          { id: "b", displayName: "B", vendor: "B", hostingRegion: "us", enterpriseAgreement: false, aliases: ["x"] },
        ],
      }),
    ).toThrow(/alias "x"/i);
  });
});
```

- [ ] **Step 2: Run to verify failure** — module not found.

- [ ] **Step 3: Implement** `packages/compiler/src/stages/ground.ts` with:
- `ProviderManifestSchema` (zod) + `loadManifest(raw)` that validates and rejects duplicate ids **and** duplicate aliases across providers (an ambiguous alias would resolve arbitrarily).
- `groundProviders(manifest, actions)` → `{ defaults, providerOverrides, warnings }`. Matching: lowercase the mention, test each alias as a substring, pick the **longest** matching alias per provider, and dedupe to at most one override entry per (provider, entityType). Actions without a mention populate `defaults`. Unresolvable mentions append a warning naming the mention verbatim and contribute nothing.
- Own-key discipline throughout (`Object.hasOwn` / `Object.create(null)` accumulators) — provider ids and entityType ids are model-authored strings and could be `toString`.

- [ ] **Step 4: Verify pass** — 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(compiler): ground provider mentions to adapter ids"
```

---

### Task 5: Semantic predicates become shadow entityTypes

**Files:**
- Create: `packages/compiler/src/stages/predicates.ts`
- Modify: `packages/core/src/detect/orchestrator.ts` (close the parked decision comment)
- Test: `packages/compiler/test/stages/predicates.test.ts`

This resolves the open decision parked in `orchestrator.ts` since Plan 1. Every `semanticPredicate` gets a shadow entityType so predicate findings reach an action through the path that already exists.

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/stages/predicates.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { SHADOW_PREFIX, mintShadowEntityTypes, shadowIdFor } from "../../src/stages/predicates.js";

const predicates = [
  { id: "unreleased-financials", nlPredicate: "Discusses revenue not yet public", scope: "segment" as const, severity: "high" as const, sourceQuote: "q" },
];

describe("mintShadowEntityTypes", () => {
  it("mints one tier-2 entityType per predicate", () => {
    const { entityTypes } = mintShadowEntityTypes(predicates);
    expect(entityTypes).toHaveLength(1);
    expect(entityTypes[0]!.id).toBe(shadowIdFor("unreleased-financials"));
    expect(entityTypes[0]!.tier).toBe(2);
  });

  it("carries the predicate text as the entityType definition", () => {
    const { entityTypes } = mintShadowEntityTypes(predicates);
    expect(entityTypes[0]!.nlDefinition).toContain("revenue not yet public");
  });

  it("inherits the predicate's severity", () => {
    expect(mintShadowEntityTypes(predicates).entityTypes[0]!.severity).toBe("high");
  });

  it("defaults shadow entityTypes to redact, never pseudonymize", () => {
    // A semantic finding has no stable surface form, so a surrogate could not be
    // rehydrated coherently; redaction is the only honest treatment.
    const { defaultActions } = mintShadowEntityTypes(predicates);
    expect(defaultActions[shadowIdFor("unreleased-financials")]).toBe("redact");
    expect(mintShadowEntityTypes(predicates).entityTypes[0]!.neverPseudonymize).toBe(true);
  });

  it("shadow ids are namespaced so they cannot collide with authored ids", () => {
    expect(shadowIdFor("x")).toMatch(new RegExp(`^${SHADOW_PREFIX}`));
  });

  it("rejects a predicate whose id would collide with an existing entityType", () => {
    expect(() => mintShadowEntityTypes(predicates, new Set([shadowIdFor("unreleased-financials")]))).toThrow(
      /collides/i,
    );
  });
});
```

- [ ] **Step 2: Run to verify failure** — module not found.

- [ ] **Step 3: Implement** `packages/compiler/src/stages/predicates.ts`:
- `export const SHADOW_PREFIX = "pred:";` and `shadowIdFor(predicateId)` → `` `${SHADOW_PREFIX}${predicateId}` ``.
- `mintShadowEntityTypes(predicates, existingIds = new Set())` → `{ entityTypes, defaultActions }`. Each shadow: `tier: 2`, `severity` inherited, `nlDefinition` = the predicate text, `examples: []`, `counterExamples: []`, `neverPseudonymize: true`. Default action `redact`. Throws if a minted id collides with `existingIds`.
- A docblock recording the decision **and the rejected alternative** (a `Finding.predicateId` field, which would have required changes in normalize, merge, resolve, and apply for the same outcome).

Then in `packages/core/src/detect/orchestrator.ts`, replace the parked open-decision comment with the resolution: predicates arrive as `pred:`-prefixed entityTypes minted by the compiler, so `strictestAction` needs no predicate-specific path; tier-2 engines (Plan 5) must emit findings whose `entityType` is the shadow id.

- [ ] **Step 4: Verify pass** — 6 tests green; `pnpm test` from root green (core's 276 unaffected — the change is comment-only there).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(compiler): mint shadow entityTypes for semantic predicates; close parked decision"
```

---

### Task 6: Validate stage — regex safety, validators, id hygiene

**Files:**
- Create: `packages/compiler/src/stages/validate.ts`
- Test: `packages/compiler/test/stages/validate.test.ts`

Spec §3.2 stage 3. The highest-risk stage: the model authors regexes, and a catastrophic-backtracking pattern in a browser extension is a hang the user cannot escape.

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/stages/validate.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { checkRegexSafety, validateIdHygiene, validateRules } from "../../src/stages/validate.js";

describe("checkRegexSafety", () => {
  it("accepts a well-behaved pattern", async () => {
    await expect(checkRegexSafety("\\b[A-Z]{5}[0-9]{4}[A-Z]\\b")).resolves.toBeUndefined();
  });

  it("rejects a catastrophic-backtracking pattern within the time budget", async () => {
    // (a+)+$ against a long non-matching input is the classic ReDoS shape.
    await expect(checkRegexSafety("^(a+)+$")).rejects.toThrow(/timed out|backtrack/i);
  }, 15_000);

  it("rejects a syntactically invalid pattern", async () => {
    await expect(checkRegexSafety("([unclosed")).rejects.toThrow(/invalid/i);
  });

  it("rejects a pattern that can match the empty string", async () => {
    await expect(checkRegexSafety("X*")).rejects.toThrow(/empty string/i);
  });
});

describe("validateRules", () => {
  it("rejects a rule naming a validator outside the fixed library", async () => {
    await expect(
      validateRules([{ id: "r", entityType: "e", regex: "a", validator: "invented", sourceQuote: "q" }]),
    ).rejects.toThrow(/unknown validator "invented"/i);
  });

  it("accepts every validator the core library actually exports", async () => {
    for (const v of ["luhn", "verhoeff", "pan-structure", "jwt-shape"]) {
      await expect(
        validateRules([{ id: `r-${v}`, entityType: "e", regex: "\\d{2,}", validator: v, sourceQuote: "q" }]),
      ).resolves.toBeDefined();
    }
  });
});

describe("validateIdHygiene", () => {
  it("warns when an entityType id shares a token with its own examples", () => {
    // Ids ship outbound in [REDACTED:<id>]; an id derived from the confidential
    // value defeats the redaction.
    const warnings = validateIdHygiene([
      { id: "project-titan", examples: ["Project Titan"], nlDefinition: "d", tier: 1, counterExamples: [], severity: "high", sourceQuote: "q" },
    ]);
    expect(warnings[0]).toMatch(/project-titan/);
  });

  it("stays silent on a generic id", () => {
    expect(
      validateIdHygiene([
        { id: "internal-codename", examples: ["Project Titan"], nlDefinition: "d", tier: 1, counterExamples: [], severity: "high", sourceQuote: "q" },
      ]),
    ).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify failure** — module not found.

- [ ] **Step 3: Implement** `packages/compiler/src/stages/validate.ts`:

**`checkRegexSafety(source, timeoutMs = 1000)`** — the bounded-time check. Compile first (invalid → throw). Reject empty-matchable patterns via `new RegExp(source).test("")` (the same gate `loadPolicyIr` applies, caught here with a better message). Then run the pattern against adversarial inputs **inside a `node:worker_threads` Worker with `eval: true`**, racing a timeout and calling `worker.terminate()` on expiry:

```ts
const WORKER_SOURCE = `
  const { parentPort, workerData } = require("node:worker_threads");
  const re = new RegExp(workerData.source, "g");
  for (const input of workerData.inputs) { re.lastIndex = 0; re.test(input); }
  parentPort.postMessage("ok");
`;
```

A synchronous regex cannot be interrupted on the main thread — a worker is the only way to bound it, which is why this stage owns a worker rather than a timing heuristic. Adversarial inputs: repeats of `a`, of the pattern's first literal character, and of a two-character alternation, each ~20k long with a non-matching suffix.

**`validateRules(rules)`** — for each rule: `checkRegexSafety` on its regex; `hasValidator` from `@sih/core` for its validator name (the fixed-library invariant, enforced here rather than trusted from the prompt); variant-consistency mirroring the IR schema.

**`validateIdHygiene(entityTypes)`** — warn when an id's kebab tokens overlap tokens in that entityType's `examples[]` (case-insensitive, ignoring tokens shorter than 4 chars to avoid noise). A warning, not an error: the compiler cannot know which nouns are confidential, so this flags for the human audit the report demands.

- [ ] **Step 4: Verify pass** — 8 tests green. The ReDoS test has a generous `testTimeout` because it must actually wait out the budget.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(compiler): validate stage with worker-bounded ReDoS check"
```

---

### Task 7: Self-test stage

**Files:**
- Create: `packages/compiler/src/stages/selftest.ts`
- Test: `packages/compiler/test/stages/selftest.test.ts`

Spec §3.2 stage 4: the model generates positives and hard negatives per entityType; the **actual core runtime** executes them; per-entity coverage is reported. Under-threshold entities produce warnings — compilation "succeeds with warnings", listing exactly what is weak.

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/stages/selftest.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { runSelfTest } from "../../src/stages/selftest.js";
import { FixtureLlmClient } from "../../src/llm/fixture.js";
import { loadTestFixtures } from "../fixtures/index.js";
import { minimalCompiledIr } from "../fixtures/minimal-compiled-ir.js";

describe("runSelfTest", () => {
  const ir = loadPolicyIr(JSON.stringify(minimalCompiledIr()));

  it("scores each tier-0 entityType against the real runtime", async () => {
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    const pan = report.entities.find((e) => e.entityType === "in-pan")!;
    expect(pan.positives).toBeGreaterThan(0);
    expect(pan.recall).toBeGreaterThan(0.8);
  });

  it("flags an entityType whose rules catch nothing", async () => {
    // The fixture supplies positives for a deliberately unmatched entity.
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    expect(report.warnings.some((w) => /below threshold/i.test(w))).toBe(true);
  });

  it("counts false positives from hard negatives", async () => {
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    const pan = report.entities.find((e) => e.entityType === "in-pan")!;
    expect(pan.falsePositives).toBeGreaterThanOrEqual(0);
    expect(pan.negatives).toBeGreaterThan(0);
  });

  it("skips tier-1 and tier-2 entityTypes with an explicit note", async () => {
    // Only tier 0 is executable today; scoring tier 1/2 here would report zero
    // recall for entities whose engines do not exist until Plans 4-5.
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    const client = report.entities.find((e) => e.entityType === "client-name");
    expect(client?.skipped).toBe(true);
    expect(client?.skipReason).toMatch(/tier 1/i);
  });

  it("tags every generated case so Plan 7 can exclude them from the eval corpus", async () => {
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    expect(report.corpusTag).toMatch(/selftest/i);
  });
});
```

- [ ] **Step 2: Run to verify failure** — module not found.

Two fixtures this test needs, both created in this task:
- `packages/compiler/test/fixtures/minimal-compiled-ir.ts` — exports `minimalCompiledIr()` returning a fresh `structuredClone` of a compiler-shaped IR: the four entityTypes from core's own fixture (`in-pan` tier 0, `aws-key` tier 0 `neverPseudonymize`, `generic-secret` tier 0 `neverPseudonymize`, `client-name` **tier 1**) plus one deliberately unmatchable tier-0 entity so the below-threshold warning has something to fire on.
- The LLM fixtures for each per-entity generation call, hand-authored the same way as Task 3's (run, read the hash from the miss error, write the file).

- [ ] **Step 3: Implement** `packages/compiler/src/stages/selftest.ts`:
- Per tier-0 entityType, one LLM call producing `{positives: string[], negatives: string[]}` (~20 each), with a system prompt that (a) forbids reusing text from the policy document, and (b) demands negatives be *hard* — text that looks like the entity but is not.
- Execute each case with `detect({ir, provider: "selftest", text, config: {tier0: true, tier1: false, tier2: false}})` from `@sih/core`. A positive counts as caught if any finding carries that entityType; a negative counts as a false positive if any finding does.
- Tier 1/2 entityTypes are **skipped with a reason**, not scored — their engines land in Plans 4–5, and reporting 0% recall for them would be a false alarm every compile.
- `RECALL_THRESHOLD = 0.8`, `MAX_FP_RATE = 0.1`; breaches append warnings naming the entity and its numbers.
- `corpusTag` = `"selftest-v1"`, stamped on the returned cases so Plan 7's contamination filter can exclude them by tag as well as by n-gram overlap.
- Cases are returned (not discarded) so `emit` can write them alongside the IR.

- [ ] **Step 4: Verify pass** — 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(compiler): self-test stage executing the real core runtime"
```

---

### Task 8: Emit — IR, report, and the loadPolicyIr round-trip

**Files:**
- Create: `packages/compiler/src/stages/emit.ts`, `packages/compiler/src/report.ts`
- Create: `packages/compiler/src/compile.ts`
- Test: `packages/compiler/test/compile.test.ts`

The end-to-end pipeline test. The emitted IR **must load through `loadPolicyIr`** — the compiler cannot be trusted to produce IRs the runtime rejects.

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/compile.test.ts`:
```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr, resolveAction } from "@sih/core";
import { compilePolicy } from "../src/compile.js";
import { FixtureLlmClient } from "../src/llm/fixture.js";
import { loadTestFixtures } from "./fixtures/index.js";

const POLICIES = join(import.meta.dirname, "..", "..", "..", "policies");

describe("compilePolicy", () => {
  const run = () =>
    compilePolicy({
      client: new FixtureLlmClient(loadTestFixtures()),
      document: readFileSync(join(POLICIES, "p-fin.md"), "utf8"),
      manifest: JSON.parse(readFileSync(join(POLICIES, "providers.json"), "utf8")),
      policyName: "p-fin",
    });

  it("emits an IR the runtime loader accepts", async () => {
    const { ir } = await run();
    expect(() => loadPolicyIr(JSON.stringify(ir))).not.toThrow();
  });

  it("stamps the IR with a hash of the source document", async () => {
    const { ir } = await run();
    expect(ir.policyHash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("carries provenance for every rule, entity and predicate", async () => {
    const { ir } = await run();
    for (const rule of ir.rules) {
      expect(ir.provenance[rule.id], rule.id).toBeDefined();
      expect(ir.provenance[rule.id]!.quote.length).toBeGreaterThan(0);
    }
  });

  it("resolves the provider clause end-to-end", async () => {
    const { ir } = await run();
    const loaded = loadPolicyIr(JSON.stringify(ir));
    // P-FIN §5: customer data may not go to non-enterprise or foreign-hosted services.
    expect(resolveAction(loaded, "client-name", "deepseek")).not.toBe(
      resolveAction(loaded, "client-name", "claude"),
    );
  });

  it("produces a markdown report listing outbound-visible ids for audit", async () => {
    const { report } = await run();
    expect(report).toMatch(/# Compilation report/i);
    expect(report).toMatch(/outbound-visible/i);
    for (const id of (await run()).ir.entityTypes.map((e) => e.id)) {
      expect(report).toContain(id);
    }
  });

  it("succeeds with warnings rather than failing on a weak entity", async () => {
    const { warnings, ok } = await run();
    expect(ok).toBe(true);
    expect(Array.isArray(warnings)).toBe(true);
  });

  it("fails hard when a rejected candidate leaves an entityType actionless", async () => {
    // Grounding drops a candidate → schema would reject the IR. The compiler must
    // catch that itself with a legible message, not emit an IR the loader rejects.
    await expect(
      compilePolicy({
        client: new FixtureLlmClient(loadTestFixtures()),
        document: "# Empty\n\n§1 Nothing to see here.",
        manifest: JSON.parse(readFileSync(join(POLICIES, "providers.json"), "utf8")),
        policyName: "empty",
      }),
    ).rejects.toThrow(/no entityTypes|actionless/i);
  });
});
```

- [ ] **Step 2: Run to verify failure** — module not found.

- [ ] **Step 3: Implement**
- `src/stages/emit.ts` — assembles the `PolicyIr`: `irVersion: "1"`, `policyHash` = `sha256(document)`, entityTypes (authored + shadow), rules, semanticPredicates, actions (`default` + `providerOverrides`), `failMode`, `latencyBudgetMs` (default 5000), and `provenance` keyed by rule/entity/predicate id carrying `{clause, quote}` derived from each item's `sourceQuote` (clause = the `§n` marker found nearest before the quote in the document, else `"unmarked"`).
- `src/report.ts` — renders the markdown report: source hash, stage-by-stage counts, **rejected candidates with their reasons** (the anti-hallucination gate's audit trail), self-test coverage per entity, all warnings, and an **"Outbound-visible identifiers"** section listing every entityType id with the note that these ship inside `[REDACTED:<id>]`.
- `src/compile.ts` — `compilePolicy({client, document, manifest, policyName})` runs extract → ground → predicates → validate → self-test → emit, collecting warnings, and **validates its own output** by calling `loadPolicyIr` before returning. Returns `{ir, report, warnings, ok, selfTestCases}`. Throws with a legible message when the IR would be structurally unusable (no entityTypes, or an entityType with no action).

- [ ] **Step 4: Verify pass** — 7 tests green; full suite + typecheck green from root.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(compiler): emit IR with provenance plus auditable compilation report"
```

---

### Task 9: CLI, live compilation script, and the committed IRs

**Files:**
- Create: `packages/compiler/src/cli.ts`, `packages/compiler/src/llm/anthropic.ts`
- Create: `scripts/record-fixtures.ts`
- Create: `policies/compiled/p-fin.ir.json`, `p-med.ir.json`, `p-corp.ir.json` + `.report.md` for each
- Test: `packages/compiler/test/cli.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/compiler/test/cli.test.ts`:
```ts
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { runCli } from "../src/cli.js";
import { loadTestFixtures } from "./fixtures/index.js";

const POLICIES = join(import.meta.dirname, "..", "..", "..", "policies");

describe("runCli", () => {
  it("writes an IR and a report, exiting 0", async () => {
    const out = mkdtempSync(join(tmpdir(), "sih-cli-"));
    const code = await runCli(
      ["--policy", join(POLICIES, "p-fin.md"), "--providers", join(POLICIES, "providers.json"), "--out", out, "--name", "p-fin"],
      { fixtures: loadTestFixtures() },
    );
    expect(code).toBe(0);
    const ir = readFileSync(join(out, "p-fin.ir.json"), "utf8");
    expect(() => loadPolicyIr(ir)).not.toThrow();
    expect(readFileSync(join(out, "p-fin.report.md"), "utf8")).toMatch(/# Compilation report/i);
  });

  it("exits 1 with a legible message on a missing policy file", async () => {
    const code = await runCli(["--policy", "/nope.md", "--providers", join(POLICIES, "providers.json"), "--out", "/tmp", "--name", "x"], {
      fixtures: loadTestFixtures(),
    });
    expect(code).toBe(1);
  });

  it("refuses to run live without an explicit --live flag", async () => {
    // Guards against a test or CI run silently spending money.
    const code = await runCli(["--policy", join(POLICIES, "p-fin.md"), "--providers", join(POLICIES, "providers.json"), "--out", "/tmp", "--name", "x"], {});
    expect(code).toBe(1);
  });
});
```

- [ ] **Step 2: Run to verify failure** — module not found.

- [ ] **Step 3: Implement**

`src/llm/anthropic.ts` — the real client. Per the API reference: model `claude-opus-5`, structured outputs via `client.messages.parse()` with `zodOutputFormat`, `output_config.effort: "high"`, and streaming for the large extraction call (`max_tokens` 16000 is at the non-streaming ceiling; use `client.messages.stream()` + `finalMessage()` for anything larger). Handle `stop_reason === "refusal"` explicitly — a data-leak policy names credentials and security topics, so a classifier decline is a realistic failure here, not a hypothetical. Opt into server-side fallbacks by default:

```ts
import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";

const MODEL = "claude-opus-5";

export class AnthropicLlmClient implements LlmClient {
  private readonly client = new Anthropic();

  async complete<T>(request: LlmRequest, schema: z.ZodType<T>): Promise<T> {
    const response = await this.client.beta.messages.parse({
      model: MODEL,
      max_tokens: request.maxTokens,
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      output_config: { effort: "high", format: zodOutputFormat(schema as never) },
      system: request.system,
      messages: [{ role: "user", content: request.user }],
    });
    if (response.stop_reason === "refusal") {
      throw new Error(
        `model declined to compile this policy (category ${response.stop_details?.category ?? "unknown"}); ` +
          `see the compilation report for which clause triggered it`,
      );
    }
    if (response.parsed_output === null) {
      throw new Error(`model returned output that did not match schema ${request.schemaName}`);
    }
    return response.parsed_output as T;
  }
}
```

`src/cli.ts` — `runCli(argv, deps)`. Flags: `--policy`, `--providers`, `--out`, `--name`, `--live`. With `deps.fixtures` it uses `FixtureLlmClient`; with `--live` it constructs `AnthropicLlmClient`; with neither it **exits 1** rather than defaulting to the network. Exit codes: 0 = compiled (warnings allowed and printed), 1 = hard failure. Writes `<name>.ir.json`, `<name>.report.md`, and `<name>.selftest.json` to `--out`.

`scripts/record-fixtures.ts` — the manual live pass: compiles all three policies with `AnthropicLlmClient` wrapped in a recorder that writes each `(hash, response)` to `packages/compiler/test/fixtures/llm/`, then re-runs the whole suite against the recorded fixtures to prove they replay. Documented as requiring `ANTHROPIC_API_KEY` or an `ant auth login` profile, and as the **only** thing in the repo that spends money.

- [ ] **Step 4: Verify pass**

Run: `pnpm test` and `pnpm typecheck` from root — everything green (276 core + ~45 compiler).

- [ ] **Step 5: Live compile and commit the IRs**

```bash
pnpm -C packages/compiler exec tsx ../../scripts/record-fixtures.ts
pnpm test
```

Review each generated `policies/compiled/*.report.md` by hand — especially the **Outbound-visible identifiers** and **Rejected candidates** sections — before committing. A compiled IR is a security artifact; it does not land unread.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(compiler): CLI, live Anthropic client, and compiled IRs for the policy suite"
```

---

## Done criteria for this plan

`sih-compile` turns any of the three authored policy documents into an IR that `loadPolicyIr` accepts, with every rule traceable to a verbatim policy quote, every model-authored regex proven non-catastrophic in bounded time, every validator name checked against core's fixed library, per-entity tier-0 coverage measured by the real runtime, and a markdown report a human can audit before shipping. Semantic predicates reach actions through minted shadow entityTypes, closing the decision parked since Plan 1. Tests never touch the network.

**Next plans:** 4 — tier-1 (ORT-web span tagger) + the Playwright eval harness; 5 — tier-2 (WebLLM judge) + the Approach-B baseline.

## Deviations log

(entries added during execution, same convention as Plans 1–2)

**Task 1 (compiler scaffold + LLM seam).**

- **Plan-file encoding fix.** Task 1's `requestHash` code block carried a raw NUL byte (0x00) inside `.join(...)` — the same corruption Plan 2 hit and logged. Replaced with the literal TS escape text `\u0000` (six source characters). This was the file's only raw NUL; the tree is now NUL-free (`git ls-files -co --exclude-standard | xargs perl -ne '/\x00/ && ...'` finds nothing — note BSD `grep -P` is broken on this machine, so perl is the scan of record). `fixture.ts` was written with the escape from the start and byte-verified with `od -c`.
- **Step order: package.json and tsconfig.json created at Step 1, not Step 3.** Step 2's expected failure (`Cannot find module '../../src/llm/fixture.js'`) presupposes that vitest can run, which requires the package manifest to exist. The scaffold was therefore created alongside the test, `pnpm install` run, and the fail-first check then produced exactly the expected module-resolution error. No content deviation — package.json, tsconfig.json, client.ts and fixture.ts are byte-for-byte the plan's, modulo the NUL escape above.
- **Task 1 result:** 6 compiler tests green, core's 276 unaffected (282 total), `pnpm typecheck` clean across both packages. `@anthropic-ai/sdk@^0.110.0` resolves and installs, though nothing imports it until Task 3.

### Task 1 review fixes

- **`system` pinned in the hash-sensitivity test.** The reviewer mutation-verified a real hole: deleting `system` from `requestHash` passed all six original tests, because the boundary test varies `user` and the FixtureLlmClient tests key their maps with `requestHash` itself (self-consistent under any hash). Two prompts differing only in system prompt would have shared a fixture. Re-verified after the fix: the mutation now fails `changes when any field changes`.
- **Compile-time exhaustiveness guard.** `requestHash` destructures `LlmRequest` and asserts the rest is `never`, so a field added later but forgotten in the hash is a typecheck error rather than silent fixture sharing. Verified by adding an `effort` field to `LlmRequest`: `TS2322: Type 'true' is not assignable to type 'never'`.
- **Miss-error remediation names the hand-authoring convention** (`write test/fixtures/llm/<hash>.json by hand`) alongside the recorder script, which does not exist until Task 9 — Tasks 3 and 7 author fixtures from this message.
- **`src/index.ts` barrel created.** `package.json` `main` pointed at it and no task in the plan created it; the first cross-package import would have failed module-not-found.
- Minors: truncation rationale commented (64 bits, repo-authored corpus, filename-friendly); schema-mismatch error now pinned to name the hash, matching the miss contract; `z.prettifyError` replaces zod v4's JSON-dump `.message`.
- Applied by the coordinator directly: the implementer subagent hit a server-side 529 on receiving the fix list, before starting work. The NUL trap fired again during the edit and was repaired with the logged `chr(92)` perl technique.

**Task 2 (policy suite + provider manifest).** No deviations: `test/policies.test.ts` and `providers.json` are byte-for-byte the plan's, and every required clause in all three per-policy lists is present. Authoring notes, all inside the plan's "at minimum" latitude:

- **Clause layout is one sentence per line, no inline emphasis.** Each clause is a bare `§N.M <single sentence>` paragraph rather than a bullet or a bolded run-in heading, so the anti-hallucination gate's verbatim lift is a clean substring of the document with no markdown to strip. 26 / 27 / 28 clause lines in P-FIN / P-MED / P-CORP; document lengths 5,272 / 5,488 / 5,705 characters against the test's 1,500 floor.
- **P-FIN §5 names all four manifest providers, not the plan's three.** The plan requires DeepSeek and Gemini as non-contracted and Claude as the enterprise agreement; ChatGPT was added on the same footing as Gemini (§5.5) so every id in `providers.json` is reachable from prose, which gives Task 4's provider resolution a real four-way grounding target instead of three plus an unexercised row. §5.6 adds the personal/trial-subscription case, since "non-enterprise" otherwise reads as vendor-level only.
- **The absences were verified mechanically, not just by the test's three negative assertions.** P-MED and P-CORP contain zero occurrences of any provider name or manifest alias (`deepseek|gemini|chatgpt|openai|anthropic|claude|bard|non-enterprise|foreign-hosted`), and P-MED contains zero occurrences of `salar|compensation|bonus|payroll` — so the policy-adaptivity metric reads a real silence. Note `/salary/i` alone would not catch "salaries", hence the broader stem check.
- **P-CORP §1.3 makes its provider silence deliberate** ("governs the content of a prompt rather than the choice of assistant, and the list of assistants approved for business use is maintained separately"), so a compiler that emits no provider rule under P-CORP is agreeing with the document rather than missing something.
- **Voices deliberately differ**, per spec §6.1's "as an admin would write them": P-FIN is an India-context compliance memo (absolute prohibitions, incident timers, PAN/Aadhaar/UPI), P-MED a committee-approved hospital information-governance policy (duty-of-confidence rationale, permissions foregrounded, Caldicott sign-off), P-CORP a dry standards document (revision table, MUST/MUST NOT conformance language, control-numbered sections). Robustness to that variation is part of what Task 3 is being tested on.
- **Task 2 result:** 287 tests green (283 before, +4 policies), typecheck clean across both packages, perl NUL scan clean over all four new files (no escapes were needed).

**Task 3 (extract stage + quote-grounding gate).** `src/stages/extract.ts` is the plan's code verbatim (`ExtractionSchema`, `SYSTEM`, `normalizeForQuoteMatch`, `groundQuotes`, `extract`), plus one comment on the synthesized action id. Deviations, all in the test and fixture layer:

- **Fixture filenames are `<schemaName>.<hash>.json`, keyed by the trailing hash segment.** The plan says "filename = hash"; a directory of bare hex names is unreviewable, and this repo requires humans to audit compiler artifacts before they ship. `loadTestFixtures(dir?)` in `test/fixtures/index.ts` therefore splits on `.` and keys by the last segment, which must match `/^[0-9a-f]{16}$/` (requestHash's width). Two guards, both pinned: a `*.json` whose trailing segment is not a hash throws rather than keying itself unreachably by its own name, and two files claiming one hash throw rather than letting one silently answer a request it was not recorded for. Non-JSON files (README, `.DS_Store`) are skipped. 5 tests.
- **The miss error in `src/llm/fixture.ts` now names that convention** (`write test/fixtures/llm/${request.schemaName}.${hash}.json by hand`). Task 1's message named `<hash>.json`, which would have had every later task author files the loader accepts but a human cannot browse. One-line change; Task 1's error-contract tests still pass unchanged.
- **The plan's test block called `loadTestFixtures()` without importing it, and imported `requestHash` without using it.** Added the import, dropped the unused one.
- **The `void request` documentation const was dropped for a comment.** An object of `expect.any(String)` matchers that is never passed to a matcher reads like a forgotten assertion. The comment says what it was there to say and adds why no assertion is needed: `requestHash` covers system, user, schemaName and maxTokens, so any drift in the request misses the committed fixture and fails the test by name — a harder pin than an equality check.
- **Two extract tests added beyond the plan's one, because the plan's one does not test the gate.** Mutation-verified: replacing `extract`'s whole return with `{...raw, rejected: []}` — the gate deleted outright — passed all 23 tests. Every quote in an honestly-authored fixture grounds, so the plan's assertions (`entityTypes` contains `in-pan`, `rejected` is empty) hold identically with and without the gate, and `groundQuotes`'s unit tests never prove it was wired in. The new tests replay a second fixture in which the model invents a `blood-type` entity and its action, asserting the invented pair is dropped and reported; the same mutation now fails both. They also pin the synthesized action id (`blood-type:default`), which is the string an auditor reads in the report's rejected-candidates section.
- **Both fixtures are hand-authored, per the plan's Task 9 hand-off.** `Extraction.c2079a920065d32e.json` answers the test's POLICY: `in-pan` (tier 0, `pan-structure`, regex `\b[A-Z]{5}[0-9]{4}[A-Z]\b`) and `client-name` (tier 1, org-name surrogate), two actions, no predicates, `failMode: "closed"`. `Extraction.f1724da884e9fcf7.json` answers a second sparse document with one grounded entity and one hallucinated one. Verified mechanically rather than by eye: every `sourceQuote` in both files was normalized and substring-checked against its own document (5 grounded / 2 rejected, exactly as intended), and `in-pan`'s three examples pass core's real `panStructure` while all four counterExamples fail it — `ABCDE1234F` is shaped like a PAN but carries an invalid holder type, so it is a hard negative rather than filler.
- **Task 3 result:** 301 tests green (287 before, +14: 9 extract, 5 fixture loader), typecheck clean across both packages, perl NUL scan clean over the tree including untracked files.
