# AI-DLPP Plan 1: Core Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `@sih/core`'s foundation: the Policy IR (types, validation, loading), the action resolver, the named-validator library, the text segmenter, the tier-0 detection engine, and the orchestrator skeleton — a package that takes a compiled policy IR + text and returns actionable findings, fully testable in Node.

**Architecture:** pnpm monorepo. `packages/core` is pure TypeScript with **no DOM and no chrome.* access — enforced by tsconfig `lib: ["ES2022"]` (no DOM lib) so any `document`/`window`/`chrome` reference fails typecheck.** Zod validates the IR at load; version mismatch refuses to load. Tier-0 = compiled regex rules + named validators + context boost + entropy scanning over code/kv segments. The orchestrator defines the `Detector`/`TierConfig` seams that tiers 1–2 (Plans 4–5) will plug into.

**Tech Stack:** TypeScript 5 (strict), pnpm workspaces, vitest, zod. Node ≥ 20.

**Spec:** `docs/superpowers/specs/2026-08-13-ai-dlpp-design.md` §2.2, §3.1, §4.1

---

### Task 1: Monorepo scaffold

**Files:**
- Create: `package.json`, `pnpm-workspace.yaml`, `tsconfig.base.json`, `.gitignore`
- Create: `packages/core/package.json`, `packages/core/tsconfig.json`
- Create: `packages/core/src/index.ts`, `packages/core/test/smoke.test.ts`

- [ ] **Step 1: Verify pnpm is available**

Run: `pnpm --version || npm i -g pnpm`
Expected: a version number (≥ 9).

- [ ] **Step 2: Create root files**

`package.json`:
```json
{
  "name": "sih",
  "private": true,
  "scripts": {
    "test": "pnpm -r test",
    "typecheck": "pnpm -r typecheck"
  }
}
```

`pnpm-workspace.yaml`:
```yaml
packages:
  - "packages/*"
  - "apps/*"
```

`tsconfig.base.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true,
    "declaration": true
  }
}
```

`.gitignore`:
```
node_modules/
dist/
*.tsbuildinfo
```

- [ ] **Step 3: Create the core package**

`packages/core/package.json`:
```json
{
  "name": "@sih/core",
  "version": "0.0.1",
  "type": "module",
  "main": "src/index.ts",
  "scripts": {
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "zod": "^4.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.6.0",
    "vitest": "^3.0.0"
  }
}
```

`packages/core/tsconfig.json` — note `lib` has **no DOM** and `types` allows **only Node globals** (`performance`, `atob`, `Buffer` for tests) — `document`/`window`/`chrome` stay undeclared. This is the browser-API firewall from spec §2.2:
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

`packages/core/src/index.ts`:
```ts
export const CORE_VERSION = "0.0.1";
```

`packages/core/test/smoke.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { CORE_VERSION } from "../src/index.js";

describe("smoke", () => {
  it("imports the package", () => {
    expect(CORE_VERSION).toBe("0.0.1");
  });
});
```

- [ ] **Step 4: Install and run**

Run: `pnpm install && pnpm test`
Expected: 1 test passes.

- [ ] **Step 5: Verify the DOM firewall works**

Append `const x = document.title;` to `packages/core/src/index.ts`, run `pnpm -C packages/core typecheck`.
Expected: FAIL with `Cannot find name 'document'`. **Remove the line**, rerun, expect PASS. (`vitest` may still execute DOM code because Node polyfills nothing here — the *typecheck* script is the enforced gate; it runs in CI and pre-merge.)

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: scaffold pnpm monorepo with DOM-firewalled @sih/core"
```

---

### Task 2: Policy IR types and zod schema

**Files:**
- Create: `packages/core/src/policy/types.ts`, `packages/core/src/policy/schema.ts`
- Test: `packages/core/test/policy/schema.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/policy/schema.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { PolicyIrSchema } from "../../src/policy/schema.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

describe("PolicyIrSchema", () => {
  it("accepts a valid IR", () => {
    expect(() => PolicyIrSchema.parse(minimalIr())).not.toThrow();
  });

  it("rejects an entityType with no action mapping", () => {
    const ir = minimalIr();
    ir.entityTypes.push({
      id: "orphan", tier: 1, nlDefinition: "x", examples: [], counterExamples: [], severity: "low",
    });
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/orphan.*no action/i);
  });

  it("rejects a rule referencing an unknown entityType", () => {
    const ir = minimalIr();
    ir.rules[0]!.entityType = "nope";
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/unknown entityType/i);
  });

  it("rejects a rule with neither regex nor entropyThreshold", () => {
    const ir = minimalIr();
    delete (ir.rules[0] as Record<string, unknown>)["regex"];
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/regex or entropyThreshold/i);
  });
});
```

`packages/core/test/fixtures/minimal-ir.ts` (test fixture used everywhere; returns a fresh deep copy each call):
```ts
import type { PolicyIrInput } from "../../src/policy/types.js";

export function minimalIr(): PolicyIrInput {
  return structuredClone(BASE);
}

const BASE: PolicyIrInput = {
  irVersion: "1",
  policyHash: "test-hash",
  entityTypes: [
    { id: "in-pan", tier: 0, nlDefinition: "Indian PAN card number", examples: ["ABCPD1234E"], counterExamples: [], severity: "high" },
    { id: "aws-key", tier: 0, nlDefinition: "AWS access key ID", examples: ["AKIAIOSFODNN7EXAMPLE"], counterExamples: [], severity: "critical" },
    { id: "generic-secret", tier: 0, nlDefinition: "High-entropy secret string", examples: [], counterExamples: [], severity: "critical" },
    { id: "client-name", tier: 1, nlDefinition: "Name of a client organization", examples: ["Globex"], counterExamples: [], severity: "high" },
  ],
  rules: [
    { id: "pan-rule", entityType: "in-pan", regex: "\\b[A-Z]{5}[0-9]{4}[A-Z]\\b", validator: "pan-structure", contextBoost: ["PAN", "tax"] },
    { id: "aws-rule", entityType: "aws-key", regex: "\\bAKIA[0-9A-Z]{16}\\b" },
    { id: "entropy-rule", entityType: "generic-secret", entropyThreshold: 4.0, minLength: 20 },
  ],
  semanticPredicates: [],
  actions: {
    default: { "in-pan": "block", "aws-key": "block", "generic-secret": "redact", "client-name": "pseudonymize" },
    providerOverrides: { deepseek: { "client-name": "redact" } },
  },
  failMode: "closed",
  latencyBudgetMs: 5000,
  provenance: {
    "pan-rule": { clause: "§3.2", quote: "PAN numbers must never be shared." },
  },
};
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- schema`
Expected: FAIL — `Cannot find module '../../src/policy/schema.js'`

- [ ] **Step 3: Implement types and schema**

`packages/core/src/policy/types.ts`:
```ts
export type Severity = "low" | "medium" | "high" | "critical";
export type Action = "allow" | "pseudonymize" | "redact" | "block";
export type FailMode = "open" | "closed";
export type Tier = 0 | 1 | 2;

export interface EntityType {
  id: string;
  tier: Tier;
  nlDefinition: string;
  examples: string[];
  counterExamples: string[];
  severity: Severity;
}

export interface Rule {
  id: string;
  entityType: string;
  /** Regex source (no flags); present for pattern rules. */
  regex?: string;
  /** Name of a validator in the fixed library; never generated code. */
  validator?: string;
  /** Nearby keywords that raise confidence. */
  contextBoost?: string[];
  /** Present for entropy rules (bits/char over sliding windows). */
  entropyThreshold?: number;
  /** Minimum candidate length for entropy rules. */
  minLength?: number;
}

export interface SemanticPredicate {
  id: string;
  nlPredicate: string;
  scope: "segment" | "message";
}

export interface Actions {
  default: Record<string, Action>;
  providerOverrides?: Record<string, Record<string, Action>>;
}

export interface Provenance {
  clause: string;
  quote: string;
}

export interface PolicyIr {
  irVersion: "1";
  policyHash: string;
  entityTypes: EntityType[];
  rules: Rule[];
  semanticPredicates: SemanticPredicate[];
  actions: Actions;
  failMode: FailMode;
  latencyBudgetMs: number;
  provenance: Record<string, Provenance>;
}

/** Input shape before zod validation (what JSON.parse gives us). */
export type PolicyIrInput = PolicyIr;
```

`packages/core/src/policy/schema.ts`:
```ts
import { z } from "zod";

const SeveritySchema = z.enum(["low", "medium", "high", "critical"]);
const ActionSchema = z.enum(["allow", "pseudonymize", "redact", "block"]);

const EntityTypeSchema = z.object({
  id: z.string().min(1),
  tier: z.union([z.literal(0), z.literal(1), z.literal(2)]),
  nlDefinition: z.string().min(1),
  examples: z.array(z.string()),
  counterExamples: z.array(z.string()),
  severity: SeveritySchema,
});

const RuleSchema = z
  .object({
    id: z.string().min(1),
    entityType: z.string().min(1),
    regex: z.string().optional(),
    validator: z.string().optional(),
    contextBoost: z.array(z.string()).optional(),
    entropyThreshold: z.number().positive().optional(),
    minLength: z.number().int().positive().optional(),
  })
  .refine((r) => r.regex !== undefined || r.entropyThreshold !== undefined, {
    message: "rule must have regex or entropyThreshold",
  });

export const PolicyIrSchema = z
  .object({
    irVersion: z.string(),
    policyHash: z.string().min(1),
    entityTypes: z.array(EntityTypeSchema).min(1),
    rules: z.array(RuleSchema),
    semanticPredicates: z.array(
      z.object({ id: z.string().min(1), nlPredicate: z.string().min(1), scope: z.enum(["segment", "message"]) }),
    ),
    actions: z.object({
      default: z.record(z.string(), ActionSchema),
      providerOverrides: z.record(z.string(), z.record(z.string(), ActionSchema)).optional(),
    }),
    failMode: z.enum(["open", "closed"]),
    latencyBudgetMs: z.number().int().positive(),
    provenance: z.record(z.string(), z.object({ clause: z.string(), quote: z.string() })),
  })
  .superRefine((ir, ctx) => {
    const ids = new Set(ir.entityTypes.map((e) => e.id));
    for (const e of ir.entityTypes) {
      if (!(e.id in ir.actions.default)) {
        ctx.addIssue({ code: "custom", message: `entityType "${e.id}" has no action mapping` });
      }
    }
    for (const r of ir.rules) {
      if (!ids.has(r.entityType)) {
        ctx.addIssue({ code: "custom", message: `rule "${r.id}" references unknown entityType "${r.entityType}"` });
      }
    }
  });
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm -C packages/core test -- schema`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): policy IR types and zod schema with cross-reference refinements"
```

---

### Task 3: Policy loader with version refusal

**Files:**
- Create: `packages/core/src/policy/load.ts`
- Test: `packages/core/test/policy/load.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/policy/load.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { loadPolicyIr, PolicyLoadError, PolicyVersionError } from "../../src/policy/load.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

describe("loadPolicyIr", () => {
  it("loads a valid IR from JSON text", () => {
    const ir = loadPolicyIr(JSON.stringify(minimalIr()));
    expect(ir.entityTypes).toHaveLength(4);
  });

  it("refuses an unknown irVersion (spec §7: never best-effort)", () => {
    const raw = { ...minimalIr(), irVersion: "2" };
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(PolicyVersionError);
  });

  it("throws PolicyLoadError on malformed JSON", () => {
    expect(() => loadPolicyIr("{nope")).toThrow(PolicyLoadError);
  });

  it("throws PolicyLoadError on an invalid regex", () => {
    const raw = minimalIr();
    raw.rules[0]!.regex = "([unclosed";
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/invalid regex/i);
  });

  it("throws PolicyLoadError when a named validator does not exist", () => {
    const raw = minimalIr();
    raw.rules[0]!.validator = "not-a-real-validator";
    expect(() => loadPolicyIr(JSON.stringify(raw))).toThrow(/unknown validator/i);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- load`
Expected: FAIL — `Cannot find module '../../src/policy/load.js'`

- [ ] **Step 3: Implement the loader**

`packages/core/src/policy/load.ts`:
```ts
import { PolicyIrSchema } from "./schema.js";
import type { PolicyIr } from "./types.js";
import { hasValidator } from "../detect/validators.js";

export class PolicyLoadError extends Error {}
export class PolicyVersionError extends PolicyLoadError {}

export const SUPPORTED_IR_VERSION = "1";

export function loadPolicyIr(jsonText: string): PolicyIr {
  let raw: unknown;
  try {
    raw = JSON.parse(jsonText);
  } catch (e) {
    throw new PolicyLoadError(`IR is not valid JSON: ${(e as Error).message}`);
  }

  const version = (raw as { irVersion?: unknown })?.irVersion;
  if (version !== SUPPORTED_IR_VERSION) {
    throw new PolicyVersionError(
      `unsupported irVersion ${JSON.stringify(version)}; this runtime supports "${SUPPORTED_IR_VERSION}"`,
    );
  }

  const parsed = PolicyIrSchema.safeParse(raw);
  if (!parsed.success) {
    throw new PolicyLoadError(`IR failed validation: ${parsed.error.message}`);
  }
  const ir = parsed.data as PolicyIr;

  for (const rule of ir.rules) {
    if (rule.regex !== undefined) {
      try {
        // "u" alone breaks common IR escapes like \- ; plain compile matches tier-0 usage.
        new RegExp(rule.regex, "g");
      } catch (e) {
        throw new PolicyLoadError(`rule "${rule.id}" has invalid regex: ${(e as Error).message}`);
      }
    }
    if (rule.validator !== undefined && !hasValidator(rule.validator)) {
      throw new PolicyLoadError(`rule "${rule.id}" names unknown validator "${rule.validator}"`);
    }
  }
  return ir;
}
```

`packages/core/src/detect/validators.ts` (stub for now — Task 5 fills it):
```ts
export type Validator = (candidate: string) => boolean;

const REGISTRY: Record<string, Validator> = {
  "pan-structure": () => true, // implemented in Task 5
};

export function hasValidator(name: string): boolean {
  return name in REGISTRY;
}

export function getValidator(name: string): Validator {
  const v = REGISTRY[name];
  if (!v) throw new Error(`unknown validator "${name}"`);
  return v;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm -C packages/core test -- load`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): policy loader with version refusal and rule sanity checks"
```

---

### Task 4: Action resolver

**Files:**
- Create: `packages/core/src/policy/resolve.ts`
- Test: `packages/core/test/policy/resolve.test.ts`

Spec §3.1: pure function, two-level merge (`providerOverrides[provider]` over `default`), exhaustively tested — a bug here is a silent leak. Unknown entityType throws (the schema guarantees every entityType has a default action, so a miss is a programmer error, not a policy state).

- [ ] **Step 1: Write the failing test**

`packages/core/test/policy/resolve.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { resolveAction } from "../../src/policy/resolve.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));

describe("resolveAction", () => {
  it("returns the default action when no override exists", () => {
    expect(resolveAction(ir, "client-name", "chatgpt")).toBe("pseudonymize");
  });

  it("applies a provider override", () => {
    expect(resolveAction(ir, "client-name", "deepseek")).toBe("redact");
  });

  it("does not leak overrides to other entityTypes on the same provider", () => {
    expect(resolveAction(ir, "in-pan", "deepseek")).toBe("block");
  });

  it("falls back to default for unknown providers", () => {
    expect(resolveAction(ir, "client-name", "some-new-provider")).toBe("pseudonymize");
  });

  it("throws on an unknown entityType (programmer error)", () => {
    expect(() => resolveAction(ir, "nope", "chatgpt")).toThrow(/unknown entityType/i);
  });

  it("covers every entityType × provider without throwing", () => {
    for (const e of ir.entityTypes) {
      for (const p of ["chatgpt", "claude", "gemini", "deepseek", "unknown"]) {
        expect(["allow", "pseudonymize", "redact", "block"]).toContain(resolveAction(ir, e.id, p));
      }
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- resolve`
Expected: FAIL — `Cannot find module '../../src/policy/resolve.js'`

- [ ] **Step 3: Implement**

`packages/core/src/policy/resolve.ts`:
```ts
import type { Action, PolicyIr } from "./types.js";

export function resolveAction(ir: PolicyIr, entityTypeId: string, providerId: string): Action {
  const base = ir.actions.default[entityTypeId];
  if (base === undefined) {
    throw new Error(`unknown entityType "${entityTypeId}" — IR validation should have prevented this`);
  }
  return ir.actions.providerOverrides?.[providerId]?.[entityTypeId] ?? base;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm -C packages/core test -- resolve`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): action resolver with provider overrides"
```

---

### Task 5: Validator library — Luhn, Verhoeff, PAN structure

**Files:**
- Modify: `packages/core/src/detect/validators.ts` (replace the stub registry)
- Test: `packages/core/test/detect/validators.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/detect/validators.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { getValidator } from "../../src/detect/validators.js";

describe("luhn", () => {
  const luhn = getValidator("luhn");
  it("accepts known-valid numbers", () => {
    expect(luhn("79927398713")).toBe(true);
    expect(luhn("4539148803436467")).toBe(true); // Visa test number
    expect(luhn("4539 1488 0343 6467")).toBe(true); // with spaces
  });
  it("rejects invalid numbers and junk", () => {
    expect(luhn("79927398714")).toBe(false);
    expect(luhn("")).toBe(false);
    expect(luhn("abcd")).toBe(false);
  });
});

describe("verhoeff", () => {
  const verhoeff = getValidator("verhoeff");
  it("accepts the canonical example 2363 (check digit of 236 is 3)", () => {
    expect(verhoeff("2363")).toBe(true);
  });
  it("exactly one check digit makes any base valid", () => {
    const base = "23629958402";
    const valid = [..."0123456789"].filter((d) => verhoeff(base + d));
    expect(valid).toHaveLength(1);
  });
  it("a single transposed digit invalidates", () => {
    const base = "23629958402";
    const check = [..."0123456789"].find((d) => verhoeff(base + d))!;
    const full = base + check;
    const swapped = full[1]! + full[0]! + full.slice(2);
    expect(verhoeff(swapped)).toBe(false);
  });
});

describe("pan-structure", () => {
  const pan = getValidator("pan-structure");
  it("accepts a well-formed PAN with a valid 4th character", () => {
    expect(pan("ABCPD1234E")).toBe(true);
  });
  it("rejects an invalid holder-type character", () => {
    expect(pan("ABCXD1234E")).toBe(false); // X is not a holder type
  });
  it("rejects wrong shapes", () => {
    expect(pan("ABCP1234E")).toBe(false);
    expect(pan("abcpd1234e")).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- validators`
Expected: FAIL — `unknown validator "luhn"`

- [ ] **Step 3: Implement**

Replace `packages/core/src/detect/validators.ts` entirely:
```ts
export type Validator = (candidate: string) => boolean;

function luhn(candidate: string): boolean {
  const digits = candidate.replace(/[\s-]/g, "");
  if (!/^\d{2,}$/.test(digits)) return false;
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = digits.charCodeAt(i) - 48;
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}

// Verhoeff dihedral-group tables (used by Aadhaar).
const D = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
  [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
  [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
  [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
  [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
  [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
  [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
  [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
] as const;
const P = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
  [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
  [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
  [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
  [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
  [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
  [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
] as const;

function verhoeff(candidate: string): boolean {
  const digits = candidate.replace(/[\s-]/g, "");
  if (!/^\d{2,}$/.test(digits)) return false;
  let c = 0;
  const reversed = [...digits].reverse();
  for (let i = 0; i < reversed.length; i++) {
    c = D[c]![P[i % 8]![reversed[i]!.charCodeAt(0) - 48]!]!;
  }
  return c === 0;
}

// Indian PAN: AAAPA9999A; 4th char = holder type.
const PAN_HOLDER_TYPES = new Set(["A", "B", "C", "F", "G", "H", "J", "L", "P", "T"]);

function panStructure(candidate: string): boolean {
  if (!/^[A-Z]{5}[0-9]{4}[A-Z]$/.test(candidate)) return false;
  return PAN_HOLDER_TYPES.has(candidate[3]!);
}

const REGISTRY: Record<string, Validator> = {
  luhn,
  verhoeff,
  "pan-structure": panStructure,
};

export function hasValidator(name: string): boolean {
  return name in REGISTRY;
}

export function getValidator(name: string): Validator {
  const v = REGISTRY[name];
  if (!v) throw new Error(`unknown validator "${name}"`);
  return v;
}
```

- [ ] **Step 4: Run all tests (loader relied on the stub — ensure nothing broke)**

Run: `pnpm -C packages/core test`
Expected: all tests PASS (validators + schema + load + resolve + smoke).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): luhn, verhoeff, and PAN-structure validators"
```

---

### Task 6: Validator library — JWT shape and Shannon entropy

**Files:**
- Modify: `packages/core/src/detect/validators.ts`
- Test: `packages/core/test/detect/entropy.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/detect/entropy.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { getValidator, shannonEntropy } from "../../src/detect/validators.js";

describe("jwt-shape", () => {
  const jwt = getValidator("jwt-shape");
  const b64url = (s: string) => Buffer.from(s).toString("base64url");

  it("accepts a structurally valid JWT", () => {
    const token = `${b64url('{"alg":"HS256","typ":"JWT"}')}.${b64url('{"sub":"1"}')}.${b64url("sig")}`;
    expect(jwt(token)).toBe(true);
  });
  it("rejects three dot-separated non-JWT parts", () => {
    expect(jwt("aaa.bbb.ccc")).toBe(false); // header decodes but has no alg
    expect(jwt("not a token")).toBe(false);
  });
});

describe("shannonEntropy", () => {
  it("is 0 for a single repeated character", () => {
    expect(shannonEntropy("aaaaaaaa")).toBe(0);
  });
  it("is 1 bit/char for a two-symbol alternation", () => {
    expect(shannonEntropy("abababab")).toBeCloseTo(1.0, 5);
  });
  it("is high for a random-looking secret", () => {
    expect(shannonEntropy("x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ")).toBeGreaterThan(4.0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- entropy`
Expected: FAIL — `unknown validator "jwt-shape"` / no export `shannonEntropy`.

- [ ] **Step 3: Implement**

Add to `packages/core/src/detect/validators.ts` (before `REGISTRY`; also export `shannonEntropy` and register `"jwt-shape"`):
```ts
export function shannonEntropy(s: string): number {
  if (s.length === 0) return 0;
  const counts = new Map<string, number>();
  for (const ch of s) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  let h = 0;
  for (const n of counts.values()) {
    const p = n / s.length;
    h -= p * Math.log2(p);
  }
  return h;
}

function decodeBase64Url(part: string): string | undefined {
  if (!/^[A-Za-z0-9_-]+$/.test(part)) return undefined;
  try {
    // atob is ES-level in Node 20+ and browsers; no DOM lib needed.
    const b64 = part.replace(/-/g, "+").replace(/_/g, "/");
    return atob(b64 + "=".repeat((4 - (b64.length % 4)) % 4));
  } catch {
    return undefined;
  }
}

function jwtShape(candidate: string): boolean {
  const parts = candidate.split(".");
  if (parts.length !== 3) return false;
  const header = decodeBase64Url(parts[0]!);
  if (header === undefined) return false;
  try {
    const obj = JSON.parse(header) as Record<string, unknown>;
    return typeof obj["alg"] === "string";
  } catch {
    return false;
  }
}
```

In `REGISTRY`, add: `"jwt-shape": jwtShape,`

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm -C packages/core test -- entropy`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): jwt-shape validator and shannon entropy"
```

---

### Task 7: Segmenter

**Files:**
- Create: `packages/core/src/segment/segment.ts`
- Test: `packages/core/test/segment/segment.test.ts`

Spec §2.2: text → segments, code-fence / prose / kv-pair aware. Offsets are **absolute** into the original string — every downstream span depends on this invariant.

- [ ] **Step 1: Write the failing test**

`packages/core/test/segment/segment.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { segmentText } from "../../src/segment/segment.js";

const SAMPLE = [
  "Please review my setup.",
  "```",
  "API_KEY=abc123",
  "```",
  "username: john",
  "password: hunter2",
  "Thanks for the help!",
].join("\n");

describe("segmentText", () => {
  it("splits into prose / code / kv / prose", () => {
    const segs = segmentText(SAMPLE);
    expect(segs.map((s) => s.kind)).toEqual(["prose", "code", "kv", "prose"]);
  });

  it("keeps absolute offsets: text === original.slice(start, end)", () => {
    for (const s of segmentText(SAMPLE)) {
      expect(s.text).toBe(SAMPLE.slice(s.start, s.end));
    }
  });

  it("covers the whole input with no gaps between segment bounds", () => {
    const segs = segmentText(SAMPLE);
    expect(segs[0]!.start).toBe(0);
    expect(segs[segs.length - 1]!.end).toBe(SAMPLE.length);
    for (let i = 1; i < segs.length; i++) {
      expect(segs[i]!.start).toBe(segs[i - 1]!.end);
    }
  });

  it("treats an unclosed fence as code to the end", () => {
    const segs = segmentText("hello\n```\nSECRET=x");
    expect(segs.map((s) => s.kind)).toEqual(["prose", "code"]);
  });

  it("returns a single prose segment for plain text", () => {
    expect(segmentText("just words here")).toEqual([
      { start: 0, end: 15, kind: "prose", text: "just words here" },
    ]);
  });

  it("handles empty input", () => {
    expect(segmentText("")).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- segment`
Expected: FAIL — `Cannot find module '../../src/segment/segment.js'`

- [ ] **Step 3: Implement**

`packages/core/src/segment/segment.ts`:
```ts
export type SegmentKind = "prose" | "code" | "kv";

export interface Segment {
  start: number;
  end: number;
  kind: SegmentKind;
  text: string;
}

const KV_LINE = /^\s*[A-Za-z_][A-Za-z0-9_.-]*\s*[=:]\s*\S/;

/**
 * Split text into contiguous, gap-free segments with absolute offsets.
 * Fenced blocks (```...```) are "code"; runs of key=value / key: value
 * lines are "kv"; everything else is "prose".
 */
export function segmentText(text: string): Segment[] {
  if (text.length === 0) return [];

  // Pass 1: fence boundaries. The closing fence consumes its trailing newline
  // so the next region starts cleanly on the following line.
  const regions: Array<{ start: number; end: number; kind: "code" | "other" }> = [];
  const fence = /```[\s\S]*?(?:```\n?|$)/g;
  let last = 0;
  for (let m = fence.exec(text); m !== null; m = fence.exec(text)) {
    if (m.index > last) regions.push({ start: last, end: m.index, kind: "other" });
    regions.push({ start: m.index, end: m.index + m[0].length, kind: "code" });
    last = m.index + m[0].length;
  }
  if (last < text.length) regions.push({ start: last, end: text.length, kind: "other" });

  // Pass 2: split "other" regions into kv / prose line runs.
  const out: Segment[] = [];
  for (const region of regions) {
    if (region.kind === "code") {
      out.push({ ...region, kind: "code", text: text.slice(region.start, region.end) });
      continue;
    }
    let runStart = region.start;
    let runKind: SegmentKind | undefined;
    let cursor = region.start;
    while (cursor < region.end) {
      const nl = text.indexOf("\n", cursor);
      const lineEnd = nl === -1 || nl >= region.end ? region.end : nl + 1;
      const line = text.slice(cursor, lineEnd);
      const kind: SegmentKind = KV_LINE.test(line) ? "kv" : "prose";
      if (runKind === undefined) {
        runKind = kind;
      } else if (kind !== runKind) {
        out.push({ start: runStart, end: cursor, kind: runKind, text: text.slice(runStart, cursor) });
        runStart = cursor;
        runKind = kind;
      }
      cursor = lineEnd;
    }
    if (runKind !== undefined && runStart < region.end) {
      out.push({ start: runStart, end: region.end, kind: runKind, text: text.slice(runStart, region.end) });
    }
  }
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm -C packages/core test -- segment`
Expected: 6 tests PASS. If the first test fails on kind sequencing, check whether blank/whitespace-only lines between kv lines break the run — the sample has none; do not add speculative handling (YAGNI).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): gap-free segmenter (prose/code/kv) with absolute offsets"
```

---

### Task 8: Detection types

**Files:**
- Create: `packages/core/src/detect/types.ts`
- Test: covered by usage in Tasks 9–11 (types only — no runtime behavior to test)

- [ ] **Step 1: Create the shared detection types**

`packages/core/src/detect/types.ts`:
```ts
import type { Action, PolicyIr, Severity, Tier } from "../policy/types.js";
import type { Segment } from "../segment/segment.js";

export interface Finding {
  /** Absolute character offsets into the original message text. */
  start: number;
  end: number;
  /** Exact matched text: always === message.slice(start, end). */
  text: string;
  entityType: string;
  severity: Severity;
  tier: Tier;
  /** Rule id (tier 0) or model identifier (tiers 1-2). */
  source: string;
  /** 0..1 */
  confidence: number;
}

export interface ResolvedFinding extends Finding {
  action: Action;
}

export interface TierConfig {
  tier0: boolean;
  tier1: boolean;
  tier2: boolean;
  t1Model?: string;
  t2Model?: string;
  backend?: "wasm" | "webgpu";
}

/** Seam for tiers 1-2 (Plans 4-5). Labels come from ir.entityTypes at inference. */
export interface SpanTagger {
  tag(segments: Segment[], ir: PolicyIr): Promise<Finding[]>;
}
export interface SemanticJudge {
  judge(segments: Segment[], ir: PolicyIr, priorFindings: Finding[]): Promise<Finding[]>;
}

export interface DetectorEngines {
  tier1?: SpanTagger;
  tier2?: SemanticJudge;
}

export interface DetectionResult {
  findings: ResolvedFinding[];
  timings: { tier0Ms: number; tier1Ms?: number; tier2Ms?: number };
}
```

- [ ] **Step 2: Typecheck**

Run: `pnpm -C packages/core typecheck`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(core): detection types and tier engine seams"
```

---

### Task 9: Tier-0 — regex rules with validators and context boost

**Files:**
- Create: `packages/core/src/detect/tier0.ts`
- Test: `packages/core/test/detect/tier0.test.ts`

Behavior: each regex rule scans the **full message text** (absolute spans for free). A rule with a `validator` drops matches the validator rejects. Confidence: 0.9 base, +0.05 if any `contextBoost` keyword appears (case-insensitive) within ±40 chars of the match, capped at 0.99.

- [ ] **Step 1: Write the failing test**

`packages/core/test/detect/tier0.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { runTier0 } from "../../src/detect/tier0.js";
import { segmentText } from "../../src/segment/segment.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const run = (text: string) => runTier0(ir, text, segmentText(text));

describe("runTier0 — regex rules", () => {
  it("finds a PAN with absolute span and provenance source", () => {
    const text = "my PAN is ABCPD1234E for tax filing";
    const [f] = run(text);
    expect(f).toBeDefined();
    expect(f!.entityType).toBe("in-pan");
    expect(f!.text).toBe("ABCPD1234E");
    expect(text.slice(f!.start, f!.end)).toBe("ABCPD1234E");
    expect(f!.source).toBe("pan-rule");
    expect(f!.tier).toBe(0);
  });

  it("applies context boost when a keyword is nearby", () => {
    const boosted = run("my PAN is ABCPD1234E for tax filing")[0]!;
    const plain = run("the code ABCPD1234E appeared in the log")[0]!;
    expect(boosted.confidence).toBeCloseTo(0.95, 5);
    expect(plain.confidence).toBeCloseTo(0.9, 5);
  });

  it("drops regex matches that fail the validator", () => {
    // Regex-shaped but 4th char X is not a PAN holder type.
    expect(run("code ABCXD1234E here")).toHaveLength(0);
  });

  it("finds an AWS key with no validator configured", () => {
    const [f] = run("creds: AKIAIOSFODNN7EXAMPLE");
    expect(f!.entityType).toBe("aws-key");
    expect(f!.severity).toBe("critical");
  });

  it("finds multiple occurrences with distinct spans", () => {
    const text = "ABCPD1234E and again ABCPD1234E";
    const findings = run(text);
    expect(findings).toHaveLength(2);
    expect(findings[0]!.start).not.toBe(findings[1]!.start);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- tier0`
Expected: FAIL — `Cannot find module '../../src/detect/tier0.js'`

- [ ] **Step 3: Implement**

`packages/core/src/detect/tier0.ts`:
```ts
import type { PolicyIr, Rule } from "../policy/types.js";
import type { Segment } from "../segment/segment.js";
import type { Finding } from "./types.js";
import { getValidator } from "./validators.js";

const BASE_CONFIDENCE = 0.9;
const CONTEXT_BONUS = 0.05;
const CONTEXT_WINDOW = 40;

function severityOf(ir: PolicyIr, entityTypeId: string) {
  const e = ir.entityTypes.find((et) => et.id === entityTypeId);
  if (!e) throw new Error(`rule references unknown entityType "${entityTypeId}"`);
  return e.severity;
}

function hasNearbyKeyword(text: string, start: number, end: number, keywords: string[]): boolean {
  const window = text
    .slice(Math.max(0, start - CONTEXT_WINDOW), Math.min(text.length, end + CONTEXT_WINDOW))
    .toLowerCase();
  return keywords.some((k) => window.includes(k.toLowerCase()));
}

function runRegexRule(ir: PolicyIr, rule: Rule, text: string): Finding[] {
  const findings: Finding[] = [];
  const re = new RegExp(rule.regex!, "g");
  const validator = rule.validator ? getValidator(rule.validator) : undefined;
  for (let m = re.exec(text); m !== null; m = re.exec(text)) {
    if (m[0].length === 0) break; // guard against zero-width loops
    if (validator && !validator(m[0])) continue;
    const boosted = rule.contextBoost?.length
      ? hasNearbyKeyword(text, m.index, m.index + m[0].length, rule.contextBoost)
      : false;
    findings.push({
      start: m.index,
      end: m.index + m[0].length,
      text: m[0],
      entityType: rule.entityType,
      severity: severityOf(ir, rule.entityType),
      tier: 0,
      source: rule.id,
      confidence: Math.min(0.99, BASE_CONFIDENCE + (boosted ? CONTEXT_BONUS : 0)),
    });
  }
  return findings;
}

export function runTier0(ir: PolicyIr, text: string, segments: Segment[]): Finding[] {
  const findings: Finding[] = [];
  for (const rule of ir.rules) {
    if (rule.regex !== undefined) {
      findings.push(...runRegexRule(ir, rule, text));
    }
    // entropy rules: Task 10 (uses `segments`)
  }
  void segments;
  return findings.sort((a, b) => a.start - b.start);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm -C packages/core test -- tier0`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): tier-0 regex rules with validators and context boost"
```

---

### Task 10: Tier-0 — entropy rules over code/kv segments

**Files:**
- Modify: `packages/core/src/detect/tier0.ts`
- Test: `packages/core/test/detect/tier0-entropy.test.ts`

Behavior (spec §4.1): entropy rules scan **only code and kv segments**. Candidates = maximal runs of secret-alphabet chars (`[A-Za-z0-9+/=_-]`) of at least `minLength`; a candidate fires when its Shannon entropy ≥ `entropyThreshold`. Confidence is fixed 0.7 (entropy is heuristic; tiers above it or the user confirm).

- [ ] **Step 1: Write the failing test**

`packages/core/test/detect/tier0-entropy.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { runTier0 } from "../../src/detect/tier0.js";
import { segmentText } from "../../src/segment/segment.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const run = (text: string) => runTier0(ir, text, segmentText(text));
const SECRET = "x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ";

describe("runTier0 — entropy rules", () => {
  it("fires on a high-entropy string inside a code fence", () => {
    const text = "here:\n```\ntoken = " + SECRET + "\n```";
    const hits = run(text).filter((f) => f.entityType === "generic-secret");
    expect(hits).toHaveLength(1);
    expect(hits[0]!.text).toBe(SECRET);
    expect(text.slice(hits[0]!.start, hits[0]!.end)).toBe(SECRET);
    expect(hits[0]!.confidence).toBe(0.7);
  });

  it("fires in kv segments", () => {
    const hits = run("SECRET_TOKEN=" + SECRET).filter((f) => f.entityType === "generic-secret");
    expect(hits).toHaveLength(1);
  });

  it("does NOT fire on the same string in prose", () => {
    const hits = run("I saw the string " + SECRET + " on a slide today").filter(
      (f) => f.entityType === "generic-secret",
    );
    expect(hits).toHaveLength(0);
  });

  it("ignores low-entropy and short strings in code", () => {
    const text = "```\nname = aaaaaaaaaaaaaaaaaaaaaaaaaaa\nport = x9K2mQ8v\n```";
    expect(run(text).filter((f) => f.entityType === "generic-secret")).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- tier0-entropy`
Expected: FAIL — entropy hits are empty (rule not implemented).

- [ ] **Step 3: Implement**

In `packages/core/src/detect/tier0.ts`, add below `runRegexRule`:
```ts
const SECRET_RUN = /[A-Za-z0-9+/=_-]+/g;
const ENTROPY_CONFIDENCE = 0.7;

function runEntropyRule(ir: PolicyIr, rule: Rule, text: string, segments: Segment[]): Finding[] {
  const findings: Finding[] = [];
  const minLength = rule.minLength ?? 20;
  for (const seg of segments) {
    if (seg.kind === "prose") continue;
    for (let m = SECRET_RUN.exec(seg.text); m !== null; m = SECRET_RUN.exec(seg.text)) {
      if (m[0].length < minLength) continue;
      if (shannonEntropy(m[0]) < rule.entropyThreshold!) continue;
      const start = seg.start + m.index;
      findings.push({
        start,
        end: start + m[0].length,
        text: m[0],
        entityType: rule.entityType,
        severity: severityOf(ir, rule.entityType),
        tier: 0,
        source: rule.id,
        confidence: ENTROPY_CONFIDENCE,
      });
    }
  }
  return findings;
}
```

Update imports: `import { getValidator, shannonEntropy } from "./validators.js";`

Replace the loop body in `runTier0`:
```ts
  for (const rule of ir.rules) {
    if (rule.regex !== undefined) {
      findings.push(...runRegexRule(ir, rule, text));
    } else if (rule.entropyThreshold !== undefined) {
      findings.push(...runEntropyRule(ir, rule, text, segments));
    }
  }
```
and delete the `void segments;` line.

- [ ] **Step 4: Run the full tier-0 suite**

Run: `pnpm -C packages/core test -- tier0`
Expected: all 9 tier-0 tests PASS (both files).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): tier-0 entropy scanning over code/kv segments"
```

---

### Task 11: Finding merge (overlap resolution)

**Files:**
- Create: `packages/core/src/detect/merge.ts`
- Test: `packages/core/test/detect/merge.test.ts`

Spec §4.1: overlapping spans resolved by highest severity; tie → higher confidence; equal → wider span. Non-overlapping findings pass through sorted by start.

- [ ] **Step 1: Write the failing test**

`packages/core/test/detect/merge.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { mergeFindings } from "../../src/detect/merge.js";
import type { Finding } from "../../src/detect/types.js";

const f = (over: Partial<Finding>): Finding => ({
  start: 0, end: 10, text: "0123456789", entityType: "x", severity: "low",
  tier: 0, source: "t", confidence: 0.9, ...over,
});

describe("mergeFindings", () => {
  it("keeps non-overlapping findings, sorted by start", () => {
    const out = mergeFindings([f({ start: 20, end: 25, text: "aaaaa" }), f({ start: 0, end: 5, text: "bbbbb" })]);
    expect(out.map((x) => x.start)).toEqual([0, 20]);
  });

  it("keeps the higher-severity finding on overlap", () => {
    const out = mergeFindings([
      f({ start: 0, end: 10, severity: "low" }),
      f({ start: 5, end: 15, severity: "critical", text: "abcdefghij" }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0]!.severity).toBe("critical");
  });

  it("breaks severity ties by confidence", () => {
    const out = mergeFindings([
      f({ start: 0, end: 10, confidence: 0.7 }),
      f({ start: 5, end: 15, confidence: 0.95, text: "abcdefghij" }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0]!.confidence).toBe(0.95);
  });

  it("touching spans (end === start) do not overlap", () => {
    const out = mergeFindings([f({ start: 0, end: 5, text: "aaaaa" }), f({ start: 5, end: 10, text: "bbbbb" })]);
    expect(out).toHaveLength(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- merge`
Expected: FAIL — `Cannot find module '../../src/detect/merge.js'`

- [ ] **Step 3: Implement**

`packages/core/src/detect/merge.ts`:
```ts
import type { Severity } from "../policy/types.js";
import type { Finding } from "./types.js";

const SEVERITY_RANK: Record<Severity, number> = { low: 0, medium: 1, high: 2, critical: 3 };

function wins(a: Finding, b: Finding): boolean {
  const sev = SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity];
  if (sev !== 0) return sev > 0;
  if (a.confidence !== b.confidence) return a.confidence > b.confidence;
  return a.end - a.start >= b.end - b.start;
}

export function mergeFindings(findings: Finding[]): Finding[] {
  const sorted = [...findings].sort((a, b) => a.start - b.start || b.end - a.end);
  const out: Finding[] = [];
  for (const cur of sorted) {
    const prev = out[out.length - 1];
    if (prev && cur.start < prev.end) {
      if (wins(cur, prev)) out[out.length - 1] = cur;
      // else: drop cur
    } else {
      out.push(cur);
    }
  }
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm -C packages/core test -- merge`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): overlap resolution for findings (severity > confidence > width)"
```

---

### Task 12: Orchestrator (tier-0 only) and public API

**Files:**
- Create: `packages/core/src/detect/orchestrator.ts`
- Modify: `packages/core/src/index.ts`
- Test: `packages/core/test/detect/orchestrator.test.ts`

The orchestrator is async and accepts `DetectorEngines` so tiers 1–2 (Plans 4–5) plug in without signature changes. In this plan only tier 0 executes.

- [ ] **Step 1: Write the failing test**

`packages/core/test/detect/orchestrator.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { detect } from "../../src/detect/orchestrator.js";
import { loadPolicyIr } from "../../src/policy/load.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const config = { tier0: true, tier1: false, tier2: false };
const SECRET = "x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ";

const MESSAGE = [
  "Hey, my PAN is ABCPD1234E for the tax form.",
  "```",
  "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
  "session = " + SECRET,
  "```",
].join("\n");

describe("detect (tier-0 only)", () => {
  it("returns resolved findings end-to-end", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    const byType = Object.fromEntries(result.findings.map((x) => [x.entityType, x]));
    expect(byType["in-pan"]!.action).toBe("block");
    expect(byType["aws-key"]!.action).toBe("block");
    expect(byType["generic-secret"]!.action).toBe("redact");
    expect(result.findings).toHaveLength(3);
  });

  it("all spans are faithful to the message", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    for (const x of result.findings) {
      expect(x.text).toBe(MESSAGE.slice(x.start, x.end));
    }
  });

  it("records tier-0 timing", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    expect(result.timings.tier0Ms).toBeGreaterThanOrEqual(0);
    expect(result.timings.tier1Ms).toBeUndefined();
  });

  it("returns no findings on clean text", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: "what is a monad?", config });
    expect(result.findings).toEqual([]);
  });

  it("tier0=false disables tier 0", async () => {
    const result = await detect({
      ir, provider: "chatgpt", text: MESSAGE, config: { tier0: false, tier1: false, tier2: false },
    });
    expect(result.findings).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm -C packages/core test -- orchestrator`
Expected: FAIL — `Cannot find module '../../src/detect/orchestrator.js'`

- [ ] **Step 3: Implement**

`packages/core/src/detect/orchestrator.ts`:
```ts
import type { PolicyIr } from "../policy/types.js";
import { resolveAction } from "../policy/resolve.js";
import { segmentText } from "../segment/segment.js";
import { mergeFindings } from "./merge.js";
import { runTier0 } from "./tier0.js";
import type { DetectionResult, DetectorEngines, Finding, TierConfig } from "./types.js";

export interface DetectInput {
  ir: PolicyIr;
  provider: string;
  text: string;
  config: TierConfig;
  engines?: DetectorEngines;
}

export async function detect(input: DetectInput): Promise<DetectionResult> {
  const { ir, provider, text, config } = input;
  const segments = segmentText(text);
  const raw: Finding[] = [];
  const timings: DetectionResult["timings"] = { tier0Ms: 0 };

  if (config.tier0) {
    const t0 = performance.now();
    raw.push(...runTier0(ir, text, segments));
    timings.tier0Ms = performance.now() - t0;
  }
  // Tiers 1-2 attach here via input.engines (Plans 4-5).

  const findings = mergeFindings(raw).map((f) => ({
    ...f,
    action: resolveAction(ir, f.entityType, provider),
  }));
  return { findings, timings };
}
```

Replace `packages/core/src/index.ts`:
```ts
export const CORE_VERSION = "0.0.1";

export type * from "./policy/types.js";
export { PolicyIrSchema } from "./policy/schema.js";
export { loadPolicyIr, PolicyLoadError, PolicyVersionError, SUPPORTED_IR_VERSION } from "./policy/load.js";
export { resolveAction } from "./policy/resolve.js";
export { segmentText, type Segment, type SegmentKind } from "./segment/segment.js";
export { getValidator, hasValidator, shannonEntropy, type Validator } from "./detect/validators.js";
export { runTier0 } from "./detect/tier0.js";
export { mergeFindings } from "./detect/merge.js";
export { detect, type DetectInput } from "./detect/orchestrator.js";
export type * from "./detect/types.js";
```

- [ ] **Step 4: Run the entire suite and typecheck**

Run: `pnpm test && pnpm typecheck`
Expected: all tests PASS (≈ 53), typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): detection orchestrator (tier-0) and public API"
```

---

## Done criteria for this plan

`@sih/core` loads a policy IR, refuses bad/unknown ones, and `detect()` turns raw message text into span-accurate, provenance-linked, provider-resolved findings using tier-0 rules — with the engine seams (`SpanTagger`, `SemanticJudge`, `TierConfig`) in place for Plans 4–5. Everything runs and is tested in plain Node.

**Next plans:** 2 — vault + pseudonymization + rehydration; 3 — policy compiler CLI + the three policy documents.

---

## Deviations log

- **Task 2 (Policy IR types + zod schema) — schema tightened post-review**, beyond the inline code in this plan, after a code-quality review returned "With fixes":
  - **Rule variant exclusivity:** `regex` and `entropyThreshold` are now mutually exclusive, and variant-mismatched fields are rejected (`validator`/`contextBoost` only on regex rules, `minLength` only on entropy rules). The planned schema allowed a rule carrying both; Task 9/10's if/else-if dispatch would silently drop one check. That dispatch is now provably safe.
  - **Id uniqueness:** duplicate `entityTypes[].id`, `rules[].id`, and `semanticPredicates[].id` are rejected.
  - **Referential integrity:** every key of `actions.default` and of `actions.providerOverrides[*]` must be a declared entityType id (a misspelled override key otherwise falls back to the default action silently — the spec names this a silent-leak surface); every `provenance` key must reference a declared rule, entityType, or semanticPredicate id. Provenance *completeness* is deliberately NOT enforced here — it stays the compiler's Extract-stage guarantee (Plan 3).
  - **Strict top level:** the IR object rejects unknown top-level keys rather than stripping them, since it is a hash-stamped artifact and silent stripping would hide drift or tampering. Inner objects stay non-strict.
  - **Issue paths:** superRefine issues are anchored at the offending element (e.g. `["rules", i, "entityType"]`).
  - **Types:** `PolicyIrInput` widened to `Omit<PolicyIr, "irVersion"> & { irVersion: string }` (JSON cannot guarantee the `"1"` literal; the Task 3 loader narrows), plus a zero-runtime schema/type drift assertion in `schema.ts` that fails typecheck on divergence.
  - **No downstream churn:** `test/fixtures/minimal-ir.ts` was unaffected — it already satisfies every new constraint and passes unmodified — and later-task snippets in this plan need no changes.

- **Task 7 (segmenter) — `KV_LINE` widened in Task 10 per review.** Task 7's reviewer flagged the key pattern as under-inclusive and deferred the fix to Task 10, where it first has consequences: `/^\s*[A-Za-z_][A-Za-z0-9_.-]*\s*[=:]\s*\S/` → `/^\s*(?:-\s+)?[A-Za-z0-9_][A-Za-z0-9_.-]*\s*[=:]\s*\S/`.
  - **Why it matters now:** tier-0 entropy rules scan code and kv segments ONLY, so a key shape the regex misses classifies as prose and is *never scanned for secrets*. YAML-list keys (`- api_key: <secret>`) and digit-initial keys (`2fa_secret: ...`) both fell through. Under-inclusion is the dangerous direction; a false kv only costs an extra segment scan.
  - **Not widened further:** the separator plus a non-space value still gates the match, so a bare list item (`- just a list item`) stays prose. Pinned by tests in `test/segment/segment.test.ts`, plus an end-to-end entropy test on a YAML-list secret in `test/detect/tier0-entropy.test.ts` — the kv/prose boundary is what decides whether entropy looks at a line, so it is pinned at both levels.
  - **Segmenter property test unaffected:** it asserts only tiling and kind-domain invariants (gap-free coverage, `text === input.slice(start, end)`, `kind` ∈ {prose, code, kv}), never that a given line lands in a given kind — so reclassifying lines cannot break it. All 300 generated documents pass unmodified. (Its generator does not actually emit the two new key shapes; adding them would strengthen it, but the invariants it checks are kind-independent either way.)

- **Task 10 (tier-0 entropy) — candidates are maximal runs, not sliding windows.** The plan's `entropyThreshold` field doc described "bits/char over sliding windows"; the implementation scores each maximal run of secret-alphabet characters (`[A-Za-z0-9+/=_-]+`) at or above `minLength`. Field doc in `policy/types.ts` corrected to match.
  - **Why runs:** a run is the token as the user actually typed it, so the reported span is a real lexical unit and needs no window-overlap dedup. Sliding windows would emit many overlapping findings per secret and push an arbitrary window size into policy authoring.
  - **Known consequence, pinned by test:** the alphabet contains `=` and `_`, so a `KEY=<secret>` kv line is ONE run and the finding spans key and value together. Over-inclusive, which is the safe direction, and asserted explicitly in `test/detect/tier0-entropy.test.ts` because Task 11's merge step resolves this span against overlapping regex findings.
  - **Alphabet excludes `.`:** including it would glue filenames, version strings, and dotted paths into single runs; the cost is that a JWT fragments into three runs, which is acceptable since whole-JWT detection belongs to the `jwt-shape` regex rule.
  - **Dead thresholds now rejected at load:** entropy is bounded by log2(67) ≈ 6.07 over this alphabet, so `entropyThreshold > 6.08` describes a string that cannot exist. Such a rule previously parsed and silently never fired — an author would believe an entity class was covered while nothing watched it. The schema rejects it. Related guidance now documented at the constant: threshold 4.0 with minLength 20 requires ≥17 distinct characters, and hex-only secrets cap at exactly log2(16) = 4.0, so covering sha/md5-style tokens needs a threshold nearer 3.5.

- **Task 12 (orchestrator) — actions resolved per overlap CLUSTER, plus engine guards and finding normalization**, beyond the inline code in this plan, per review:
  - **Cluster-strictest actions replace the snippet's per-winner `resolveAction`:** the plan resolved each merge winner's own action. Merge resolution is severity-first, so a critical `generic-secret` entropy run whose action is `redact` beats a high-severity `in-pan` inside it whose action is `block` — and the message that policy says to block gets redacted instead, with the losing finding already discarded by the time actions are read. Every winner now carries the strictest action (`block` > `redact` > `pseudonymize` > `allow`) resolved over ALL members of its cluster, using the per-cluster composition recipe documented in `detect/merge.ts` (cluster once, `mergeFindings` per cluster, one action per cluster — four lines of glue, so no combined export was folded back into `merge.ts`).
  - **Strictest is cluster-WIDE, deliberately:** clusters are transitive, so in a chain A–B–C a winner overlapping only A can still be escalated by C. Errs strict, which is the safe direction for a data-loss filter; the alternative silently under-protects text the policy did call sensitive.
  - **Engine-presence guards:** `config.tier1` with no `engines.tier1` throws, likewise tier 2. A silently skipped tier corrupts an experiment arm — T0's latency and recall reported under T0+T1's name. Tier 2 without tier 1 is explicitly NOT an error: tier 2 judges semantic predicates over segments and consumes no tier-1 spans, so predicates-only escalation is a valid configuration.
  - **Finding normalization before merge:** every raw finding's `entityType` must be declared in `ir.entityTypes` or detection throws naming both the entityType and the producing source, and severity is re-derived from the IR exactly as `tier0.ts` does. A no-op for tier 0 (the schema already validates rule entityTypes), but it is what turns a future tier-1/2 hallucinated label into a diagnosable error instead of a `resolveAction` crash deep in the pipeline, and it stops a model inflating its own severity from overruling policy about which finding survives the merge. Pinned by stub-engine tests.
  - **Open decision recorded, not made — semantic predicates have no action path:** `SemanticPredicate` carries no entityType, so predicate findings cannot be resolved by `resolveAction` at all. Whether the compiler mints shadow entityTypes (Plan 3) or `Finding` grows a `predicateId` that resolution consults (Plan 5) is noted at the resolution site so tier 2 does not quietly default to `allow` when it starts emitting them.
  - **`failMode` is the caller's, and stated as such:** `detect()` throws on engine crashes, invalid findings and guard violations, and never silently degrades; mapping those to `ir.failMode` (spec §5.3) belongs to the caller, which is the only layer that knows whether there is a user to warn. A `degraded`/`warnings` field on `DetectionResult` is expected in Plan 5, when tier-2 latency-budget degradation gives it a second producer.
  - **`Detector` type alias exported** (`(input: DetectInput) => Promise<DetectionResult>`) — spec §4.3's promised interface, and the shape the Approach-B baseline is measured through. `clusterOverlapping` is also exported from the barrel alongside `mergeFindings`.
  - **Tier seams wired now, not later:** tier-1 and tier-2 invocation paths exist so the guards, normalization and cluster resolution are exercised by stub engines today. Tier 1 receives prose + kv segments only — the orchestrator filters, engines stay dumb — and tier 2 receives all segments plus a *snapshot copy* of the prior findings (the live accumulator would let an engine observe or mutate findings it never saw; the `Finding` objects inside are deliberately the same objects, since `merge.ts` joins clusters to winners by reference). Segment escalation and `latencyBudgetMs`/`AbortSignal` enforcement remain Plan 5's. Each tier is timed around the tier call alone, with normalization outside the window, so orchestrator overhead is never charged to a tier.
