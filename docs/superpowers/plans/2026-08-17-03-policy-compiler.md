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

### Task 3 review fix: fragment quotes

- **`groundQuotes` gained a minimum quote length (`MIN_QUOTE_CHARS = 24`), exported.** The reviewer found that bare substring matching with no floor proves only that some characters appear somewhere: against the test policy, `sourceQuote: "PAN"` grounded, and `"the"` would ground against essentially any policy document ever written. A model returning one-word quotes therefore cleared the anti-hallucination gate having grounded nothing — the failure mode that looks like a pass. The SYSTEM prompt's "Quote the sentence, not the section heading" was unenforced prose. Fail-first evidence: the new fragment test failed with `expected [ { id: 'fragment' } ] to deeply equal []`, i.e. `"PAN"` demonstrably grounding before the fix.
- **24 was verified against the real suite, not assumed.** Measured across the 81 numbered clauses of P-FIN / P-MED / P-CORP: the shortest clause is **63 characters** sentence-only (68 with its `§` marker) — P-FIN §2.2, "Aadhaar numbers must never be sent to an external AI assistant."; per-document minima are 63 / 75 / 79. Real quotes therefore clear the floor by 2.6x and only fragments are caught. The floor is deliberately not tuned up toward sentence length: a genuinely short clause in some future policy would be silently dropped, and dropping a real rule is the expensive direction of this trade. It is also not a heading filter — `## §6 Governance` (16) and `## §6 Enforcement` (17) fall under it, but `## §1 Scope and definitions` (27) does not.
- **Distinct rejection reason, checked before presence:** `sourceQuote too short to ground a rule (N chars, minimum 24)`. A fragment and an invention need different fixes (quote the whole sentence vs. stop inventing rules), and a short quote that happens to appear in the document is still a fragment, so length is tested before the substring check. Only the length is named — the quote itself is never echoed, per the error-message convention.
- **Substring matching stays boundary-free, now as a stated decision:** a quote of `"PANCAKE"` grounds against a document containing `"PANCAKES"`. A model that copied 24+ characters of a real word demonstrably read the document, and boundary-anchored matching would reject genuine quotes trimmed mid-token. Commented at `groundQuotes` so a future reader finds a decision rather than an oversight.
- **Three mutations, all fatal** (they were not, before these tests): removing the length check fails all 3 new tests; `<` → `<=` fails the boundary test; and the earlier gate-bypass mutation (`return {...raw, rejected: []}`) still fails both invention tests.
- **Review-fix result:** 304 tests green (301 before, +3 groundQuotes), typecheck clean, NUL scan clean. Both committed fixtures still ground unchanged — every quote in them is a full clause of 60+ characters.

**Task 4 (ground stage — provider resolution).** `src/stages/ground.ts` implements the plan's `loadManifest` / `groundProviders` contract and the plan's six tests are verbatim. `policies/providers.json` changed (see below); no policy document changed, and `test/policies.test.ts` passes unmodified.

- **The §5.1 decision: category terms are now first-class manifest entries with an attribute predicate, and no longer live in any provider's alias list.** Task 2's review found that P-FIN §5.1 — "No customer data of any classification may be sent to non-enterprise or foreign-hosted services", the firm's *blanket* rule — contains the literal string `foreign-hosted`, which `providers.json` carried as an alias of `deepseek`. Under per-provider alias matching the broadest clause in the policy resolved to one vendor, and ChatGPT and Gemini (which §5.5 and §5.4 also place outside the firm's agreements) silently kept the permissive default. The fix is structural, not a special case for this sentence: **a category is not a provider name and must not be stored as one.** `providers.json` gained a `categories` block whose entries carry an explicit `match` over the attributes the manifest already records — `non-enterprise` → `{enterpriseAgreement: false}`, `chinese-hosted` → `{hostingRegionIn: ["cn"]}` — and `deepseek`'s aliases shrank to `["deepseek"]`. A mention matching a category resolves to **every** provider satisfying it, expanded once at load time. §5.1 now grounds onto `chatgpt`, `deepseek`, `gemini` and — correctly — not onto `claude`, which §5.2 approves. Providers and categories share one alias namespace and `loadManifest` rejects a collision between them, so a category can never quietly become a vendor synonym again.
- **Why an attribute predicate rather than a wider alias list:** the alias form encodes "foreign-hosted *means* DeepSeek", which is true only by accident of the manifest having one non-US vendor today. The day a second China-hosted adapter is added, the blanket clause keeps resolving to DeepSeek alone — the same silent downgrade, re-armed. The predicate form is decided by the manifest's own data, so the new adapter is covered on the day it is added.
- **Rejected — leave the aliases where they were.** Mutation-verified as the live bug, not a hypothetical: restoring `["deepseek", "chinese-hosted", "foreign-hosted"]` and deleting the `categories` block makes the real-manifest tests fail with `expected [ 'deepseek' ] to deeply equal [ 'chatgpt', 'deepseek', 'gemini' ]`.
- **Rejected — drop the category terms with no replacement** (the plan's first suggested option), letting §5.3/§5.4/§5.5's named vendors carry §5. Mutation-verified: the blanket clause then resolves to nothing and produces only a warning (`expected [] to deeply equal [ 'chatgpt', 'deepseek', 'gemini' ]`). Warnings keep the default, and the default is the permissive outcome — so the crown-jewel clause of a fintech policy would depend on a human noticing a report line. It is also incomplete on its own terms: §5.6 (personal/trial subscriptions) names no vendor at all, so nothing would carry it. Warning is this stage's *fallback*, not its goal; where a mention can be resolved correctly and mechanically, it should be.
- **Rejected — define `foreign-hosted` as a category.** "Foreign" is decidable only against a home jurisdiction the manifest does not record, and both available guesses are wrong. `hostingRegionNotIn: ["us"]` reproduces exactly the narrowing being fixed (`{deepseek}`) while dressing a US-centric default as a fact. `hostingRegionNotIn: ["in"]` — P-FIN is an India-context firm — sweeps in all four providers including Claude, contradicting §5.2's explicit approval; combined with the most-restrictive-wins rule below, that would block the firm's only approved assistant outright. So the term is deliberately left undefined: a bare "foreign-hosted" mention warns rather than guessing, pinned by its own test. It costs nothing on §5.1, where the `non-enterprise` disjunct already covers every foreign-hosted provider in this manifest. Note P-FIN glosses the term by example itself (§5.3, "DeepSeek, which is foreign-hosted"), which is a per-document reading and does not belong in a manifest three policies share.
- **§5.6 is deliberately left unresolvable too.** No subscription aliases were added to `non-enterprise`: §5.6's point is that a *personal Claude subscription* is outside the enterprise agreement, but the manifest models one adapter per vendor, so expanding it over `enterpriseAgreement: false` would produce `{chatgpt, gemini, deepseek}` — confidently omitting the one provider the clause is about. Modelling that needs a separate adapter id (`claude-personal`) and is out of scope here. A warning a human resolves beats a wrong answer stated precisely.
- **Longest-matching-alias-wins is scoped per provider, and is now observable rather than vestigial.** Taking a single global winner could let one long alias suppress a different provider matched by a shorter one — dropping a provider from a clause that named it. Matching therefore unions across providers and dedupes within one. Because the plan's dedupe test cannot distinguish "longest" from "any", the winning alias is *returned*: `resolveProviderMention` is exported and reports `{providerId, alias, via, source}`, which doubles as the audit trail Task 8's report needs ("§5.1 → category non-enterprise → gemini"). Mutation-verified: removing the length sort makes the alias test fail (`"deepseek"` instead of `"chinese-hosted"`).
- **Colliding actions resolve to the more restrictive one, independent of input order.** The plan says "dedupe to at most one override per (provider, entityType)" without saying which survives; first-wins would make the IR depend on the order the model happened to emit clauses in. P-FIN makes the collision real — §5.1 restricts customer data on Gemini while §5.4 permits Gemini for public material — so the rule is `max()` over `allow < pseudonymize < redact < block`, plus a warning naming both. Over-blocking is recoverable by reading the report; under-blocking is a leak. Same rule for `defaults`. Mutation-verified: last-write-wins fails the both-orders test.
- **A blank `providerMention` warns instead of falling through to `defaults`.** The extract schema allows `""`, and folding it into `defaults` would promote a provider-scoped `allow` into a firm-wide one — a downgrade manufactured by an empty string. Absent means default; present-but-blank means unresolvable.
- **Own-key discipline, black-box tested.** Both accumulators are `Object.create(null)` and every write is `Object.hasOwn`-guarded. The pin is behavioural rather than white-box: `providerOverrides["toString"]` must read `undefined`, which a `{}` accumulator answers with `Function.prototype.toString`. Mutation-verified with `{}` + `key in target`.
- **`loadManifest` rejects more than the plan asked.** Beyond duplicate ids and duplicate aliases across providers: duplicate aliases across the provider/category namespace, a category id colliding with a provider id, the same owner listing one alias twice, a blank alias, and a category whose `match` constrains nothing (an empty match selects every provider, which would turn one stray word into a firm-wide block). Aliases are compared lowercased, so `"X"` and `"x"` collide.
- **Residual, accepted:** `google` remains an alias of `gemini`, so a mention of an unrelated Google product would resolve to the Gemini adapter. That is a vendor name doing vendor-name duty, the over-blocking direction, and Task 2's manifest data is otherwise unchanged.
- **Task 4 result:** 329 tests green (304 before, +25 ground: the plan's 6 plus 19 covering the real manifest against real P-FIN §5, longest-alias reporting, conflict resolution, blank mentions, prototype-named keys, and the extra manifest rejections). Typecheck clean across both packages; perl NUL scan clean over tracked and untracked files. Six mutations run, all fatal to exactly the intended tests.

**Task 5 (semantic predicates become shadow entityTypes).** `src/stages/predicates.ts` implements the plan's contract (`SHADOW_PREFIX`, `shadowIdFor`, `mintShadowEntityTypes`) and `test/stages/predicates.test.ts` carries the plan's 6 tests verbatim (plus 2 appended, below). Fail-first: `Cannot find module '../../src/stages/predicates.js'`. The core edit is comment-only, as the plan states — core stayed at exactly 276 tests, untouched.

- **The parked comment is now a decision, and says what does NOT change.** `strictestAction`'s docblock previously read "The open decision is whether the compiler mints shadow entityTypes for them (Plan 3) or Finding grows a `predicateId` that resolution consults (Plan 5); either way it lands in this function." It now records that Plan 3 chose the shadow, that *nothing in orchestrator.ts changes* as a result (a predicate finding names its shadow id in `entityType`, so `resolveAction` — provider overrides included — and `winnerAction`'s `neverPseudonymize` check both work unmodified), and that the residual obligation moved to the producer: tier-2 engines must emit `entityType: "pred:<id>"`, not the bare predicate id. It points at `predicates.ts` for the rejected alternative rather than restating it, so the reasoning has one home.
- **Types are imported from `@sih/core` rather than redeclared locally** — the first cross-package type import in the compiler. `entityTypes` is `readonly EntityType[]` and `defaultActions` is `Record<string, Action>` using core's own types, so a drift in the IR's `EntityType` shape is a compiler typecheck failure rather than a `loadPolicyIr` rejection in Task 8. `extract.ts` and `ground.ts` declare their shapes locally because they model the *model's* output; a minted shadow is IR, and should be typed as IR.
- **One throw beyond the plan: two predicates sharing an id.** The plan specifies the `existingIds` collision only. A model emitting a duplicate predicate id would mint the same shadow twice, and `PolicyIrSchema.checkDuplicateIds` would then reject the IR in Task 8 with an error anchored at an array index far from the cause. Separate message (`collides with another minted shadow (duplicate predicate id)`) so the two failures are distinguishable. Neither message echoes `nlPredicate`; ids only, per the error-message convention — and a shadow id is outbound-visible by design, so it is not a confidential value.
- **`neverPseudonymize: true` is belt AND braces with the `redact` default, deliberately.** The default action alone would not hold: a provider override or a cluster's strictest action could still hand a shadow a `pseudonymize` stamp at runtime. With the flag set, `winnerAction` escalates that combination back to `redact` and the IR schema rejects an authored pseudonymize action for the shadow outright. The reason it must never be pseudonymized is on the constant: Plan 2's vault mints a format-preserving surrogate *keyed to a real surface value* and rehydrates it back; a semantic finding is a span judged to be **about** something and has no such value, so there is nothing to key a surrogate to and nothing coherent to rehydrate into.
- **`SHADOW_PREFIX = "pred:"` — the `:` is load-bearing.** The extraction prompt constrains authored ids to lowercase kebab-case, which cannot contain a colon, so the two id spaces are disjoint before the explicit `existingIds` check ever runs. The check stays as defense in depth against a future prompt change.
- **Task 6 hand-off, checked rather than assumed: shadow ids reach `validateIdHygiene` only if Task 8 concatenates before validating, and even then the check is vacuous on them.** Task 8's pipeline order (`extract → ground → predicates → validate → self-test → emit`) does put minting before validation, so passing `[...extracted.entityTypes, ...shadow.entityTypes]` to `validateIdHygiene` is natural — but nothing forces it, and the obvious wiring (validate the extract stage's output) would miss shadows entirely. More importantly, the plan's hygiene heuristic compares an id's kebab tokens against **that entityType's `examples[]`**, and a shadow's `examples` is `[]` by construction, so it can never warn on a shadow no matter how it is wired. Task 6 should therefore (a) receive authored + shadow ids, and (b) for shadows, compare tokens against `nlDefinition` (the predicate text) instead of `examples`, stripping `SHADOW_PREFIX` first so `"pred"` is not itself scored as a token. Not implemented here — recorded for Task 6. The audit path that *does* work for free is Task 8's report: its outbound-visible section iterates `ir.entityTypes`, which includes shadows.
- **Two tests added beyond the plan's six, both mutation-verified as real holes.** The plan's six are unmodified and verbatim; these are appended. (1) *The namespacing test cannot fail.* `expect(shadowIdFor("x")).toMatch(new RegExp(`^${SHADOW_PREFIX}`))` builds its regex from the constant under test, so it is self-consistent under **any** prefix — setting `SHADOW_PREFIX = ""` passed all six while deleting the namespace outright, which is the same self-referential shape the Task 1 review caught in the fixture hashes. The added test pins `"pred:"` independently of itself, and the `:` specifically, since a colon is what makes the shadow space disjoint from kebab-case authored ids. (2) The intra-batch duplicate throw had no test at all. Six mutations run in total — dropped `neverPseudonymize`, `redact`→`block`, `tier: 2`→`1`, dropped `existingIds` throw, empty prefix, dropped duplicate throw — all six now fatal to exactly one test each; the last two were survivors before these additions.
- **Task 5 result:** 337 tests green (329 before, +8 predicates), core still exactly 276 and comment-only, typecheck clean across both packages, perl NUL scan clean over tracked and untracked files.

### Task 6

- **`validateIdHygiene` shadow fix.** The plan scores every id against that entityType's `examples[]`, but a shadow minted by `mintShadowEntityTypes` has `examples: []` by construction, so the heuristic is vacuous on shadows however the stage is wired — while `pred:` ids are outbound-visible exactly like authored ones and their predicate-id half is model-authored. Shadows are now scored against `nlDefinition` instead, with `SHADOW_PREFIX` stripped first so `"pred"` (4 chars, exactly at the threshold) is not itself a token. Fail-first captured: the plan-literal heuristic returned `[]` on the shadow case while all 8 plan tests passed.
- **ReDoS check verified end to end.** Worker probe on Node 26 confirmed `eval: true` + `require` works; a safe pattern returns in 12ms, `^(a+)+$` against 20k chars times out at the full budget rather than throwing — so termination, not an exception, is the detection path. The unbounded control run (same pattern, no worker, no timeout) never completed and had to be killed, which is the property the whole stage exists to prevent.
- **Worker cleanup verified by the coordinator**: `vitest run` completes in ~2.15s wall clock including the 1s ReDoS budget, so no worker handle is leaked (a leak would hang the process after tests finish).
- Committed by the coordinator: the implementing agent stalled after writing both files and going 9/9 green — its own unbounded-regex control experiment hung its shell. No work was lost.

**Task 7 (self-test stage executing the real core runtime).** `test/stages/selftest.test.ts` carries the plan's 5 tests verbatim; `src/stages/selftest.ts`, `test/fixtures/minimal-compiled-ir.ts` and 5 hand-authored LLM fixtures are new. Fail-first: `Cannot find module '../../src/stages/selftest.js'`, then five identical fixture-miss errors naming `SelfTestCases.58bd28ec04a4f507.json`.

- **Prompt determinism, which this stage needs harder than any before it.** The fixture key hashes the whole `user` prompt, so one varying byte means every committed fixture misses on every run. Guarantees, all structural rather than incidental: `userPromptFor` writes each field out **by name in a fixed order** and nothing anywhere iterates an object's keys or calls `JSON.stringify` on an entityType (the parsed IR's key order is an artifact of zod's shape and of the JSON it came from, neither a contract); arrays render in IR order, which is array order; entityTypes are visited in `ir.entityTypes` order; there is no clock, no randomness, no counter, and no locale-sensitive formatting. Pinned by three tests, one of them mutation-verified with a mutation that is byte-identical under the canonical key order — reading one field through `Object.keys(entity)[0]` — so it passes all fourteen other tests and fails only the key-order test. That is the real bug shape: the fixtures were recorded against whatever insertion order this process produced, so no other test in the file can see the dependence.
- **The prompt deliberately withholds the detection rules.** A generator shown `\bLEGACY-EMP-[0-9]{8}\b` emits twenty strings matching it, reports 100% recall, and has measured its own guess rather than the policy. Cases are generated from `nlDefinition` + `examples` + `counterExamples` only, and SYSTEM rule 3 says so explicitly. `surrogateKind` and `neverPseudonymize` are withheld too — they decide what happens to a detected value, not what the class is, so including them would churn fixture keys whenever the action side of a policy changed.
- **`minimal-compiled-ir.ts` reuses core's four entityTypes verbatim** (`in-pan`, `aws-key`, `generic-secret`, `client-name` at tier 1) so this stage is scored by the rules core's own detection tests are written against; a divergent entity table would make the coverage numbers unfalsifiable against the runtime they claim to measure. Two differences: a real sha256 `policyHash` (matching what Task 8 stamps), and the plan's added unmatchable entity, `legacy-employee-id` — a tier-0 rule that is well-formed, safe, passes every validate-stage check, and simply describes a format the entity does not use (`LEGACY-EMP-[0-9]{8}` against real-world `EMP/2019/04417`). That is the realistic compiler failure, and nothing before this stage can see it.
- **Skipping tier 1/2 is a designed refusal, not an omission.** They cost no model call and produce no cases, and their `recall`/`fpRate` are left `undefined` rather than 0: 0 means "the rules caught nothing", `undefined` means "nothing was run", and collapsing the two puts a 0% recall bar next to every semantic and span-model entity on every compile until Plans 4–5 land — a warning that fires unconditionally and trains its readers to ignore the list the genuinely weak entities are also in.
- **Fixture scoring is the honest number, not a tuned one.** Measured against the real `detect(...)`: `in-pan` 20/20 recall with 1/20 FPs (a PAN-shaped internal vendor code that passes `pan-structure` — a true positive of the rule and a genuine false positive of the policy), `aws-key` 20/20 with 1/20 (the `AKIAXXXXXXXXXXXXXXXX` README placeholder), `generic-secret` 20/20 with **3/20**, `legacy-employee-id` **0/20**. The two warnings emitted are therefore a below-threshold recall on `legacy-employee-id` and an FP-rate breach on `generic-secret`. The breach was left in rather than authored away: the entropy rule at 4.0 bits/char genuinely fires on long snake_case and kebab-case identifiers (`/var/log/application/service-worker-output`, `DEFAULT_RETRY_BACKOFF_MILLISECONDS`, `v2024.05.17-hotfix-payments-rollback`), which is exactly what a *hard* negative is for and exactly the finding this stage exists to surface. No plan test requires it.
- **Positives were placed with the merge in mind, and one placement was corrected because of it.** A PAN in `pan_number=WQSAJ6207H` glues into a single 21-character entropy run scoring 4.30 bits/char; `generic-secret` is critical and `in-pan` is high, so merge resolution drops the PAN finding and the positive is silently missed. Changed to `pan_number: WQSAJ6207H` (the `:` plus space splits the run below `minLength`). AWS keys need no such care: the regex rule's 0.9 confidence beats entropy's fixed 0.7 at equal severity, so `aws-key` wins its own overlap either way. This is the difference between scoring `detect(...)`'s resolved findings — the set a rewriter would actually act on — and scoring raw pre-merge findings, which would credit the policy with catches the runtime then discards.
- **Six tests appended beyond the plan's five, each mutation-verified as a real hole**, plus 2 unmutated property tests. Fatal mutations, all of which pass the plan's five: `cases: []` and `corpusTag: ""` on cases (the plan's corpusTag test reads the *report's* tag, not the cases' — Plan 7's contamination filter reads the cases'); deleting the `MAX_FP_RATE` branch (`falsePositives >= 0` is true of every possible implementation); `skipReasonFor` returning `undefined` for tier 2 (only tier 1 is pinned by the plan); `recall = caught / positives` with no zero guard (NaN < 0.8 is false, so no warning fires and NaN serializes to `null` — an entity nothing was run against, reported as one that passed); `detected` set from the case's label rather than from what detection did (the committed corpus then cannot re-derive the coverage claim printed beside it). Eight mutations run in total, each fatal to exactly the intended tests.
- **Warnings name the entityType id and the counts and never a case.** A generated positive is a sensitive-*shaped* string — an identifier, a key, a credential — and the convention that diagnostics never carry one does not get an exemption because this particular string was invented by a model rather than lifted from a policy. Pinned by a test that cross-checks every warning against every case.
- **One extra fixture beyond the four the plan implies** (`SelfTestCases.7808f1de86789d52.json`, positives empty) to drive the zero-denominator path from a test-local one-entity IR. The tier-2 skip test needs no fixture at all, which is itself the assertion: it runs against `new FixtureLlmClient(new Map())`, so any model call would throw.
- **Task 7 result:** 361 tests green (346 before, +15: the plan's 5 plus 10 appended), 276 core / 85 compiler, typecheck clean across both packages, perl NUL scan clean over tracked and untracked files. No core file touched.

**Task 7 (coordinator review, after the implementing agent's commit `c4cd56f`).** The plan specifies "a positive counts as caught if any finding carries that entityType". Followed literally, that conflates two different outcomes. The implementer hit the conflation as a symptom: a PAN in `pan_number=WQSAJ6207H` glues into one 21-char entropy run at 4.30 bits/char, `generic-secret` (critical) beats `in-pan` (high) in overlap resolution, and the `in-pan` label never survives — so the case scored as a missed positive. Their response was to edit the corpus (`=` → `:`) so the case would not arise.

That is corpus tuning toward a nicer number, and `pan_number=VALUE` is exactly how a PAN appears in a config line or a query string. Verified against the runtime before changing anything: `pan_number=AFTPD1298Q` yields `generic-secret[0,21)` and no `in-pan` finding, but the resolved action is still `block` — cluster-strictest resolution keeps the strictest action across the overlap, so the *action* is correct and only the *label* is not this entity's. The value does not leak. Scoring it as a miss therefore made the warning assert something false ("its rules do not match the values its own definition describes") — the same cry-wolf shape the tier-1/2 skip rule exists to prevent, and misaligned with the project's headline metric, which is message-level leak prevention.

Fixed the measurement, not the data. `=` restored to the corpus; `classify` now distinguishes **shadowed** (no own-label finding, but another entityType caught it) from a genuine miss (nothing caught it). `recall` = `(caught + shadowed) / positives` is leak-prevention recall and is what `RECALL_THRESHOLD` gates; `labelRecall` = `caught / positives` is reported alongside, because the surviving label decides which surrogate the user sees and which clause the UI cites. Shadowing emits a note worded deliberately as *not* a threshold breach. Hard negatives are unchanged: another entity firing on them is not this entity's false positive, or per-entity `fpRate` would inherit every neighbour's over-firing.

Measured after the fix: `in-pan` recall 1.00 / labelRecall 0.95 / shadowed 1, `aws-key` 1.00, `generic-secret` 1.00 with fpRate 0.15 (genuine breach, left in — the 4.0 bits/char rule really does fire on long snake_case identifiers), `legacy-employee-id` recall 0.00 (the deliberate below-threshold case, still firing the real alarm). 3 tests added, all four mutations verified fatal to exactly the intended test: shadowed-as-miss (in-pan recall 0.95, still above 0.8, so every plan test passes while the report is wrong), shadowed-folded-into-caught, the shadow note worded as "below threshold", and dropping `shadowedBy` from the emitted case.

**Task 8 must not render `recall`/`labelRecall`/`fpRate` as 0% when `undefined`** — undefined means unmeasured, and the report's whole point is that the two are different.

**Task 8 (emit + the end-to-end compile pipeline).** `test/compile.test.ts` carries the plan's 7 tests verbatim; `src/stages/emit.ts`, `src/report.ts`, `src/compile.ts`, 2 hand-authored `Extraction` fixtures and 7 hand-authored `SelfTestCases` fixtures (280 cases) are new. Fail-first: `Failed to load url ../src/compile.js`, then a fixture miss naming `Extraction.b31ccfa2b2cadd6c.json`, then one naming `Extraction.fdea482aec3e9626.json`, then seven naming the per-entity `SelfTestCases` files.

- **Pipeline order: emit runs BEFORE self-test, not after.** The plan's prose says `extract → ground → predicates → validate → self-test → emit`, but the self-test stage measures coverage by executing the compiled policy through core's real `detect(...)`, so it takes a `PolicyIr` and the IR has to exist first. `compilePolicy` therefore assembles, round-trips through `loadPolicyIr`, and measures the LOADED IR — which is strictly better than measuring the assembled one, since the numbers then describe exactly the artifact a runtime would run. The report is rendered last, so it still quotes the measurement. Nothing else about the order moved.
- **`emit` reconciles dangling references instead of shipping an IR the loader rejects.** Not in the plan, and a real consequence of the grounding gate rejecting candidates one at a time: a model can ground its rule's quote and fail its entityType's, leaving a rule (or a default action, or a provider override) pointing at an entityType that is not in the IR. `loadPolicyIr` rejects that — correctly, but with a zod path and no mention of the quote that caused it. `reconcile` drops such items, records each in `dropped` with a reason, and warns. A drop rather than a throw: throwing discards an otherwise good compile over one ungrounded quote, and keeping it emits an unloadable IR; what must never happen is dropping silently, so every drop reaches the report's rejected-candidates table. An emptied per-provider map is removed rather than left as `{}`, so the report cannot show a provider as if it carried policy.
- **An entityType with no action is still a throw, per the plan.** `assertShippable` runs before the loader so the message speaks the document's vocabulary ("no clause resolved an action for it") rather than zod's. Both branches are also `loadPolicyIr` rejections, so this is a legibility layer, not a second authority — mutation-verified: deleting the call makes the plan's own seventh test fail.
- **`ok` is always `true` on a successful return, by construction, and says so in its docblock.** Every structural failure throws, because an IR the runtime rejects is not something a caller can act on. It is computed from the round-trip (`loadPolicyIr` succeeded in this process) rather than hardcoded, so it cannot drift from the claim it makes, and Task 9's CLI branches on it. Rejected: making `ok` false for a weak entity — the plan's sixth test requires `ok === true` alongside warnings, and gating shippability on a coverage number would make the compiler's headline flag depend on how good a model's generated corpus happened to be.
- **Provenance clause = nearest `§` marker at or before the quote, resolved in NORMALIZED space.** `clauseLocator` collapses whitespace exactly as the grounding gate does, so any quote the gate accepted can be located even when it spans a line break, and marker offsets come from the same normalized string so the two coordinate systems agree. A quote before any marker (P-FIN's preamble) or one that cannot be found at all reads `"unmarked"` — never a guess, because a guessed clause sends an auditor to the wrong sentence and it reads as confirmation. Shadow entityTypes inherit their predicate's clause and quote, so `pred:<id>` — the one id in the IR with no clause of its own — is still traceable.
- **The model's `sourceQuote` is stripped from the IR, field by field rather than by spread.** Two reasons, both about a committed artifact: provenance is the single home for a policy quote (a spread writes a second, unversioned copy of every quote into the IR), and naming the fields fixes the JSON key order so recompiling an unchanged policy produces an unchanged file. Mutation-verified: spreading passes all 7 plan tests, because the loader strips unknown keys from nested objects.
- **Stage counts pass `grounded` explicitly instead of deriving it from the IR — a bug found by reading the first rendered report.** The IR's action maps are DEDUPED (one entry per entityType, one per provider/entityType pair), so counting them made the extract row read `27/31 actions survived the quote-grounding gate` when all 31 grounded and 4 pairs had merged. That is the report accusing the gate of rejections it never made. The row now reports grounding only (`31/31 action clauses quote the document verbatim`) and the ground row reports the merge (`31 action clauses resolved to 8 policy defaults and 18 overrides across 3 providers`).
- **Report shape, beyond the plan's list.** Sections: header (policy hash, IR version, fail mode, latency budget), Stages, Entity types, Outbound-visible identifiers, Rejected candidates, Self-test coverage, Warnings, Provenance. Three deliberate choices: (1) ids in the outbound section are printed as `[REDACTED:<id>]`, i.e. as the string that actually leaves the machine, since the point is to read them as a third party will; (2) an unmeasured metric prints `not measured`, a measured zero prints `0% (0/20)`, and both are pinned by tests that fail if either collapses into the other; (3) `labelRecall` sits beside `recall` with a note naming the shadowing entity, because the two answer different questions — whether the value leaks, and whether this entity's own label, surrogate and cited clause are what the user sees. The Provenance section prints policy quotes verbatim; that is not a breach of the no-echo convention, which governs errors and warnings — the report is committed next to the policy document itself, and traceability is what it exists for.
- **No clock anywhere, and a test that says so.** Task 9 commits an IR and a report per policy; a timestamp would make every recompile look like a policy edit and a real edit unreadable in the diff. `compilePolicy` run twice returns byte-identical IR, report and warnings.
- **The empty-document fixture cannot ground anything, structurally.** The plan's seventh test compiles `"# Empty\n\n§1 Nothing to see here."`, whose longest normalized span is 23 characters — one below `MIN_QUOTE_CHARS` (24). So no candidate quoting that document can pass the gate no matter what the model returns, and `no entityTypes` is the only reachable branch. The fixture is authored as the failure the test's own comment describes: a model that invents a plausible entity, its rule and its action, and invents the sentence justifying all three. All four candidates are rejected as "sourceQuote not found in policy document", and the thrown message lists them with their reasons.
- **The P-FIN extraction fixture is a thorough-but-careful reading, and one thing it deliberately does NOT say.** 8 authored entityTypes (`in-pan`, `in-aadhaar`, `bank-account-identifier`, `internal-customer-id`, `client-name` at tier 1, `api-credential`, `db-connection-string`, `private-key-material`), 10 rules, 1 semantic predicate (`client-relationship-disclosure`, grounded on §3.4 — the clause that says the relationship itself is the confidential part), and 31 action clauses: 8 defaults plus §5.1 over the five customer-data classes, §5.3 over all eight, and §5.4/§5.5 over the five. It emits no `allow` for Claude from §5.2: that clause approves an assistant for customer-data work, it does not relieve §2's absolute prohibitions or §3's pseudonymization, and §2.1 says "never sent to an external AI assistant" without exception. Emitting the `allow` would have passed the plan's provider test just as well and shipped a policy that forwards client names to Claude in the clear. Every one of the 50 quotes was machine-checked against `p-fin.md` (normalized, case-sensitive, ≥24 chars) before the fixture was committed.
- **The compile's actual output, kept exactly as measured.** 9 entityTypes (8 authored + 1 shadow), 10 rules, 9 default actions, 18 provider overrides across chatgpt/gemini/deepseek and none for claude, 20 provenance entries, 0 rejected candidates, 6 warnings, 280 self-test cases. Coverage: `in-pan` 1.00/1.00 recall with fpRate 0.05, `in-aadhaar` 1.00/1.00 with 0.05, `bank-account-identifier` 0.95/0.95 with 0.10, `internal-customer-id` **0.50**, `api-credential` 0.90/0.90 with fpRate **0.25**, `db-connection-string` 0.85 recall / 0.80 label recall (1 shadowed), `private-key-material` **0.75**; `client-name` (tier 1) and `pred:client-relationship-disclosure` (tier 2) not measured.
- **Nothing was tuned to make those numbers nicer, and every one of them has a cause worth reading.** `internal-customer-id` at 0.50 is a genuinely weak rule: `\b(?:CIF|CRN|KYC)[ /:#-]?[0-9]{6,10}\b` is uppercase-only and requires the prefix adjacent to the digits, so `cif_number: 30112847`, `KYC case 4471200`, `KYC/2026/004182` and every "customer reference number 8830125" phrasing miss — exactly the failure a model-authored regex makes and exactly what this stage exists to surface. `api-credential`'s 0.25 FP rate is the documented AWS example key (`AKIAIOSFODNN7EXAMPLE`, which is a real match of a correct rule) plus four entropy hits at 4.0 bits/char over 24+ characters on a long SCREAMING_SNAKE constant, a log path, a SHA256 fingerprint and a git sha — the same true over-firing Task 7 recorded. `private-key-material` at 0.75 misses a PGP key block, a certificate block and two HSM key labels, all of which its own `nlDefinition` covers and its PEM-header regex does not. `db-connection-string`'s shadowed positive is the UPI VPA rule matching `events@rabbit` inside `amqps://events@rabbit-02.internal:5671/audit`, and its 3 outright misses are the ADO.NET, libpq and `jdbc:oracle:thin:` forms, none of which are `scheme://` URIs. `in-pan`'s single false positive is `SVCPL2024A`, a payroll vendor code that is a structurally valid PAN; `in-aadhaar`'s is `202604180001`, a twelve-digit invoice reference that passes Verhoeff by coincidence (a 1-in-10 event, and the honest limit of a checksum). `bank-account-identifier`'s two are e-mail addresses, which the VPA rule reads as `handle@provider`.
- **Two id-hygiene warnings fire and are left standing.** `private-key-material` shares "private" with its own examples and `pred:client-relationship-disclosure` shares "client" with its predicate text. Both are generic class words rather than confidential values, so both are arguably false alarms — but the check is explicitly a flag for human judgement, not a gate, and suppressing it by renaming the entities would be tuning the input to quiet a check that is doing what Task 6 designed it to do.
- **22 tests appended beyond the plan's 7** (8 in `test/stages/emit.test.ts`, 7 in `test/report.test.ts`, 7 appended to `test/compile.test.ts`), **11 mutations run, all fatal to exactly the intended tests and all survivable by the plan's 7**: rendering `undefined` metrics as 0%; deleting shadow provenance; spreading candidates so `sourceQuote` reaches the IR; keeping orphan rules; `clauseLocator` always returning "unmarked"; hashing a constant instead of the document; returning `selfTestCases: []`; deleting the label-recall column; summarising the rejected-candidates table away; truncating the warning list in the report; and (as a control that the plan's own test bites) deleting the `assertShippable` call, which fails plan test 7 and nothing else.
- **`compilePolicy` and its two types were added to `src/index.ts`.** The barrel is the package `main`, and Task 9's CLI is the first consumer.
- **Task 8 result:** 393 tests green (364 before, +29), 276 core / 117 compiler, typecheck clean across both packages, perl NUL scan clean over tracked and untracked files. No core file touched.

**Task 8 (coordinator review, after the implementing agent's commit `9f3a8d4`).** Found by resolving every entityType against every provider on the real p-fin compile and reading the matrix as a whole. All eight authored entityTypes resolved to `block` on DeepSeek; the shadow predicate resolved to `redact` — while P-FIN §5.3 states "no Firm information of any kind may be sent to DeepSeek".

Structural, not a fixture omission. `ExtractionSchema.actions[]` keys every action by `entityType`, and a semantic predicate is not an entityType at extraction time — its shadow is minted downstream in the predicates stage. A provider clause therefore has no way to name a predicate, and `mintShadowEntityTypes` returns only `defaultActions`, never overrides. Provider-conditioned behaviour is the requirement this compiler exists to serve, and one of the four data classes was silently exempt from all of it.

Fixed in `emit` — the stage where shadows and actions become one object. A shadow's override for provider P is now the strictest action any surviving authored entityType carries for P, written only when that is strictly stricter than what the shadow already resolves to. The direction is justified: a predicate exists to catch disclosures the identifier rules miss, so it is at least as sensitive as the entities the policy names. The guard makes the extension monotone — a laxer stated clause changes nothing, and `pseudonymize` can never be written for a shadow (default `redact`), which matters because the IR schema rejects that pairing and the compiler would otherwise emit an IR its own loader refuses. Every extension emits a warning: this is the one place the compiler applies a clause to something the document did not literally name, so it says so.

`ACTION_RANK` is now **exported from core** (`detect/orchestrator.ts`, re-exported from the barrel) rather than restated in the compiler. A second copy would let the compiler emit a policy the runtime resolves differently, and the two drifting apart is the bug no test in either package would catch.

p-fin after the fix: `pred:client-relationship-disclosure` resolves `block` on deepseek/gemini/chatgpt and `redact` on claude — §5.3/§5.4/§5.5 and §5.2 respectively. 4 tests added, 4 mutations verified fatal.

Two of those mutations survived the first attempt, both my own errors and both the tautology shape this plan keeps producing: the "strictest, not first" test used a second entityType that `reconcile` drops as dangling, so the strictest was also the only candidate and keeping the first would have passed; and `stricter(strictest, fallback)` was dead code, since the `<= fallback` guard already prevents loosening on its own — the comment claimed a guarantee the line did not provide. Test rewritten with two surviving entities in laxer-first order; redundant call removed and the guard documented as the actual mechanism.
