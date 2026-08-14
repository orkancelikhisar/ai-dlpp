# AI-DLPP Plan 2: Vault + Pseudonymization + Rehydration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `@sih/core`'s `pseudo/` layer: deterministic format-preserving surrogate minting, a per-conversation vault with referential integrity, AES-GCM at-rest helpers, action application (pseudonymize/redact/block) over `ResolvedFinding` spans, and streaming rehydration (surrogate → real) with chunk-boundary holdback — all pure TS, fully testable in Node.

**Architecture:** Everything lives in `packages/core/src/pseudo/` behind the existing DOM/Node firewall (WebCrypto `crypto.subtle`, `TransformStream`, `btoa`/`atob` are WHATWG globals in both runtimes — no `node:` imports). Surrogates are format-preserving fakes seeded by FNV-1a64(conversationId ‖ entityType ‖ realValue) — deterministic ⇒ referential integrity across turns; the seed is NOT a security boundary (the security property is that real values never leave; encryption-at-rest is hygiene). The IR gains two optional EntityType fields: `surrogateKind` (which generator) and `neverPseudonymize` (credentials — schema-rejected from pseudonymize actions AND vault-refused: defense in depth). Storage is an abstract async `VaultStore`; core ships `MemoryVaultStore`; Plan 6's IndexedDB store wires the AES-GCM helpers. Rehydration: pure `rehydrateText` + a `TransformStream` with a tail holdback of `maxSurrogateLen − 1` so surrogates split across chunks are never missed.

**Tech Stack:** TypeScript 5 (strict), vitest, zod 4 (schema additions). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-13-ai-dlpp-design.md` §5.4 (vault + rehydration), §3.1 (IR)

**Baseline:** `main` @ `37a5904` — 144 tests green, typecheck clean. Work on branch `feat/vault-pseudonymization`.

**Established conventions (from Plan 1 — binding):** strict TDD per task; `Object.hasOwn`/own-key discipline on record reads; span fidelity; comments record decisions with alternatives; deviations recorded in this file's Deviations log; every quirk pinned by a test.

---

### Task 1: IR extension — surrogateKind and neverPseudonymize

**Files:**
- Modify: `packages/core/src/policy/types.ts`, `packages/core/src/policy/schema.ts`
- Modify: `packages/core/test/fixtures/minimal-ir.ts`
- Test: `packages/core/test/policy/schema.test.ts` (additions)

- [ ] **Step 1: Write the failing tests** (append to the existing `PolicyIrSchema` describe block):

```ts
  it("accepts surrogateKind and neverPseudonymize on entityTypes", () => {
    // Fixture now carries both fields (client-name: surrogateKind; aws-key/generic-secret: neverPseudonymize).
    expect(() => PolicyIrSchema.parse(minimalIr())).not.toThrow();
  });

  it("rejects a neverPseudonymize entityType with default action pseudonymize", () => {
    const ir = minimalIr();
    ir.entityTypes.find((e) => e.id === "aws-key")!.neverPseudonymize = true;
    ir.actions.default["aws-key"] = "pseudonymize";
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/never be pseudonymized/i);
  });

  it("rejects a neverPseudonymize entityType with a pseudonymize provider override", () => {
    const ir = minimalIr();
    ir.actions.providerOverrides!["deepseek"] = { "generic-secret": "pseudonymize" };
    expect(() => PolicyIrSchema.parse(ir)).toThrow(/never be pseudonymized/i);
  });

  it("accepts neverPseudonymize with block and redact actions", () => {
    const ir = minimalIr(); // aws-key: block, generic-secret: redact — both neverPseudonymize
    expect(() => PolicyIrSchema.parse(ir)).not.toThrow();
  });
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm -C packages/core test`
Expected: the two rejection tests FAIL (`expected function to throw`) — the schema doesn't know the fields yet, and the fixture additions in Step 3 haven't landed. (The two acceptance tests may fail on unknown-key grounds only if EntityType were strict — it is not; capture actual output.)

- [ ] **Step 3: Implement**

`packages/core/src/policy/types.ts` — add above `EntityType` and extend it:

```ts
/** Which format-preserving generator mints surrogates for this entityType (Plan 2). */
export type SurrogateKind = "person-name" | "org-name" | "id-number" | "opaque";
```

```ts
export interface EntityType {
  id: string;
  tier: Tier;
  nlDefinition: string;
  examples: string[];
  counterExamples: string[];
  severity: Severity;
  /** Generator used when the resolved action is "pseudonymize". Missing → "opaque". */
  surrogateKind?: SurrogateKind;
  /**
   * Credentials-class marker (spec §5.4): a format-valid fake credential is a lie
   * waiting to be pasted somewhere. Schema rejects pseudonymize actions for these,
   * and the vault refuses to mint them — defense in depth.
   */
  neverPseudonymize?: boolean;
}
```

`packages/core/src/policy/schema.ts` — in `EntityTypeSchema` add:

```ts
  surrogateKind: z.enum(["person-name", "org-name", "id-number", "opaque"]).optional(),
  neverPseudonymize: z.boolean().optional(),
```

In the existing `superRefine`, after the current checks (reuse the entity Map/Set already built there):

```ts
    // neverPseudonymize × pseudonymize is contradictory policy — reject at compile
    // time rather than trusting the vault's runtime refusal alone (defense in depth).
    const neverPseudo = new Set(ir.entityTypes.filter((e) => e.neverPseudonymize).map((e) => e.id));
    for (const [entityId, action] of Object.entries(ir.actions.default)) {
      if (action === "pseudonymize" && neverPseudo.has(entityId)) {
        ctx.addIssue({
          code: "custom",
          message: `entityType "${entityId}" is marked neverPseudonymize and must never be pseudonymized`,
          path: ["actions", "default", entityId],
        });
      }
    }
    for (const [provider, overrides] of Object.entries(ir.actions.providerOverrides ?? {})) {
      for (const [entityId, action] of Object.entries(overrides)) {
        if (action === "pseudonymize" && neverPseudo.has(entityId)) {
          ctx.addIssue({
            code: "custom",
            message: `entityType "${entityId}" is marked neverPseudonymize and must never be pseudonymized`,
            path: ["actions", "providerOverrides", provider, entityId],
          });
        }
      }
    }
```

`packages/core/test/fixtures/minimal-ir.ts` — update three entityTypes in `BASE`:

```ts
    { id: "in-pan", tier: 0, nlDefinition: "Indian PAN card number", examples: ["ABCPD1234E"], counterExamples: [], severity: "high", surrogateKind: "id-number" },
    { id: "aws-key", tier: 0, nlDefinition: "AWS access key ID", examples: ["AKIAIOSFODNN7EXAMPLE"], counterExamples: [], severity: "critical", neverPseudonymize: true },
    { id: "generic-secret", tier: 0, nlDefinition: "High-entropy secret string", examples: [], counterExamples: [], severity: "critical", neverPseudonymize: true },
    { id: "client-name", tier: 1, nlDefinition: "Name of a client organization", examples: ["Globex"], counterExamples: [], severity: "high", surrogateKind: "org-name" },
```

- [ ] **Step 4: Verify pass**

Run: `pnpm -C packages/core test` and `pnpm -C packages/core typecheck`
Expected: 148 tests green (144 + 4), typecheck clean (the type-drift `Expect<>` assertions in schema.ts must still compile — optional fields on both sides).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): surrogateKind and neverPseudonymize on IR entityTypes"
```

---

### Task 2: Deterministic seeding — FNV-1a64 + PRNG

**Files:**
- Create: `packages/core/src/pseudo/seed.ts`
- Test: `packages/core/test/pseudo/seed.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/seed.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { fnv1a64, mulberry32, seededRng } from "../../src/pseudo/seed.js";

describe("fnv1a64", () => {
  it("matches published FNV-1a 64 test vectors", () => {
    expect(fnv1a64("")).toBe(0xcbf29ce484222325n);
    expect(fnv1a64("a")).toBe(0xaf63dc4c8601ec8cn);
    expect(fnv1a64("foobar")).toBe(0x85944171f73967e8n);
  });

  it("is deterministic and input-sensitive", () => {
    expect(fnv1a64("conv1\u0000client-name\u0000Globex")).toBe(fnv1a64("conv1\u0000client-name\u0000Globex"));
    expect(fnv1a64("conv1")).not.toBe(fnv1a64("conv2"));
  });
});

describe("mulberry32 / seededRng", () => {
  it("same seed → same sequence; different seed → different sequence", () => {
    const a1 = mulberry32(42); const a2 = mulberry32(42); const b = mulberry32(43);
    const seq = (r: () => number) => [r(), r(), r()];
    expect(seq(a1)).toEqual(seq(a2));
    expect(seq(mulberry32(42))).not.toEqual(seq(b));
  });

  it("seededRng derives from the full 64-bit hash and stays in [0, 1)", () => {
    const r = seededRng("some-key");
    for (let i = 0; i < 100; i++) {
      const x = r();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
    expect(seededRng("k1")()).not.toBe(seededRng("k2")());
  });
});
```

- [ ] **Step 2: Run to verify failure** — `Cannot find module '../../src/pseudo/seed.js'`.

- [ ] **Step 3: Implement**

`packages/core/src/pseudo/seed.ts`:

```ts
/**
 * Deterministic seeding for surrogate generation (spec §5.4: surrogate =
 * seeded generator keyed on hash(conversationId ‖ realValue)).
 *
 * NOT a security boundary: FNV-1a is not cryptographic, deliberately. The
 * vault's security property is that real values never leave the machine;
 * surrogate seeding only needs determinism (referential integrity across
 * turns). At-rest encryption is crypto.ts's job.
 */

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK64 = 0xffffffffffffffffn;

/** FNV-1a over UTF-16 code units (deterministic across JS runtimes). */
export function fnv1a64(s: string): bigint {
  let h = FNV_OFFSET;
  for (let i = 0; i < s.length; i++) {
    h ^= BigInt(s.charCodeAt(i));
    h = (h * FNV_PRIME) & MASK64;
  }
  return h;
}

/** Small fast PRNG; adequate for picking fake names, nothing more. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** PRNG seeded from both halves of the 64-bit hash of `key`. */
export function seededRng(key: string): () => number {
  const h = fnv1a64(key);
  return mulberry32(Number(h & 0xffffffffn) ^ Number((h >> 32n) & 0xffffffffn));
}
```

- [ ] **Step 4: Verify pass** — 152 tests green, typecheck clean. (FNV vectors are for ASCII input, where UTF-16 code units equal bytes; if a vector fails, STOP and report — do not adjust constants.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): FNV-1a64 seeding and PRNG for deterministic surrogates"
```

---

### Task 3: Surrogate generators

**Files:**
- Create: `packages/core/src/pseudo/generators.ts`
- Test: `packages/core/test/pseudo/generators.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/generators.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { generateSurrogate } from "../../src/pseudo/generators.js";
import { getValidator } from "../../src/detect/validators.js";

const KEY = "conv1\u0000client-name\u0000Globex";

describe("generateSurrogate", () => {
  it("is deterministic per seed key", () => {
    expect(generateSurrogate("org-name", "Globex", KEY)).toBe(generateSurrogate("org-name", "Globex", KEY));
  });

  it("person-name yields a first/last pair, never the real value", () => {
    const s = generateSurrogate("person-name", "Priya Sharma", "k1");
    expect(s).toMatch(/^[A-Z][a-z]+ [A-Z][a-z]+$/);
    expect(s.toLowerCase()).not.toBe("priya sharma");
  });

  it("org-name re-rolls when the pick collides with the real value", () => {
    // Deterministic construction: whatever org key "kX" yields, minting THAT org
    // under the same key must yield something else (salted retry).
    const first = generateSurrogate("org-name", "zzz-no-collision", "kX");
    const rerolled = generateSurrogate("org-name", first, "kX");
    expect(rerolled).not.toBe(first);
  });

  it("id-number preserves PAN shape and holder type, passes the validator, differs from real", () => {
    const s = generateSurrogate("id-number", "ABCPD1234E", "k2");
    expect(s).toMatch(/^[A-Z]{5}[0-9]{4}[A-Z]$/);
    expect(s[3]).toBe("P");
    expect(getValidator("pan-structure")(s)).toBe(true);
    expect(s).not.toBe("ABCPD1234E");
  });

  it("id-number falls back to class-preserving scramble for non-PAN shapes", () => {
    const s = generateSurrogate("id-number", "AC-42-9917", "k3");
    expect(s).toMatch(/^[A-Z]{2}-[0-9]{2}-[0-9]{4}$/);
    expect(s).not.toBe("AC-42-9917");
  });

  it("opaque preserves length and character classes", () => {
    const s = generateSurrogate("opaque", "x9K2mQ8vL4jR7nT3wY6z", "k4");
    expect(s).toHaveLength(20);
    expect(s).toMatch(/^[a-z][0-9][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z][A-Z][0-9][a-z]$/);
    expect(s).not.toBe("x9K2mQ8vL4jR7nT3wY6z");
  });

  it("different seed keys give different surrogates (spot check)", () => {
    expect(generateSurrogate("opaque", "abcdefgh12345678", "kA")).not.toBe(
      generateSurrogate("opaque", "abcdefgh12345678", "kB"),
    );
  });
});
```

- [ ] **Step 2: Run to verify failure** — `Cannot find module '../../src/pseudo/generators.js'`.

- [ ] **Step 3: Implement**

`packages/core/src/pseudo/generators.ts`:

```ts
import type { SurrogateKind } from "../policy/types.js";
import { seededRng } from "./seed.js";

/**
 * Format-preserving surrogate generators (spec §5.4). Surrogates are realistic
 * fakes, not ⟦P1⟧ markers — markers get mangled by the model and wreck answer
 * utility. Name/org lists are deliberately small and fictional; the compiler
 * (Plan 3) or extension config may extend them later.
 */

const FIRST = ["Anjali", "Rohan", "Meera", "Arjun", "Kavya", "Nikhil", "Priyanka", "Vikram", "Sneha", "Aditya", "Ishita", "Rahul"] as const;
const LAST = ["Verma", "Iyer", "Kapoor", "Nair", "Deshpande", "Chatterjee", "Menon", "Bhatt", "Rao", "Kulkarni", "Sethi", "Joshi"] as const;
const ORGS = ["Vantor", "Corvex Systems", "Nimbria Labs", "Atlas Forge", "Zephyrline", "Quantelle", "Meridian Ops", "Bluecrest Analytics", "Solstice Works", "Kitehill", "Novabound", "Praxeon", "Vellum & Gray", "Orchid Dynamics", "Statlerhouse", "Ironvale"] as const;

const UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const LOWER = "abcdefghijklmnopqrstuvwxyz";
const DIGITS = "0123456789";
/** Valid PAN 4th-char holder types — keep in sync with validators.ts. */
const PAN_HOLDER = "ABCFGHJLPT";

function pick(rng: () => number, pool: string | readonly string[]): string {
  return pool[Math.floor(rng() * pool.length)]!;
}

/**
 * Deterministic under (kind, seedKey); salted retry guarantees the surrogate
 * never equals the real value (case-insensitive), which would be a non-
 * pseudonymization.
 */
export function generateSurrogate(kind: SurrogateKind, real: string, seedKey: string): string {
  for (let salt = 0; ; salt++) {
    const rng = seededRng(salt === 0 ? seedKey : `${seedKey}#${salt}`);
    const candidate = generate(kind, real, rng);
    if (candidate.toLowerCase() !== real.toLowerCase()) return candidate;
  }
}

function generate(kind: SurrogateKind, real: string, rng: () => number): string {
  switch (kind) {
    case "person-name":
      return `${pick(rng, FIRST)} ${pick(rng, LAST)}`;
    case "org-name":
      return pick(rng, ORGS);
    case "id-number":
      return idNumber(real, rng);
    case "opaque":
      return scramble(real, rng);
  }
}

function idNumber(real: string, rng: () => number): string {
  // PAN-shaped reals keep their holder type (4th char) — format preservation
  // means a fake PAN should still read as the same kind of PAN.
  if (/^[A-Z]{5}[0-9]{4}[A-Z]$/.test(real)) {
    const holder = PAN_HOLDER.includes(real[3]!) ? real[3]! : "P";
    return (
      pick(rng, UPPER) + pick(rng, UPPER) + pick(rng, UPPER) + holder + pick(rng, UPPER) +
      pick(rng, DIGITS) + pick(rng, DIGITS) + pick(rng, DIGITS) + pick(rng, DIGITS) +
      pick(rng, UPPER)
    );
  }
  return scramble(real, rng);
}

/** Per-character class-preserving scramble; non-alphanumerics pass through. */
function scramble(real: string, rng: () => number): string {
  let out = "";
  for (const ch of real) {
    if (UPPER.includes(ch)) out += pick(rng, UPPER);
    else if (LOWER.includes(ch)) out += pick(rng, LOWER);
    else if (DIGITS.includes(ch)) out += pick(rng, DIGITS);
    else out += ch;
  }
  return out;
}
```

- [ ] **Step 4: Verify pass** — 159 tests green, typecheck clean. If the opaque class-pattern test fails, the real string's classes were transcribed wrong in the regex — fix the TEST, the generator contract is per-character class preservation.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): format-preserving surrogate generators"
```

---

### Task 4: Vault with referential integrity

**Files:**
- Create: `packages/core/src/pseudo/vault.ts`
- Test: `packages/core/test/pseudo/vault.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/vault.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { generateSurrogate } from "../../src/pseudo/generators.js";
import { MemoryVaultStore, Vault } from "../../src/pseudo/vault.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));

describe("Vault", () => {
  it("mints deterministically: same (conversation, real, entityType) → same surrogate", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const s1 = await vault.mint("conv1", "Globex", "client-name", ir);
    const s2 = await vault.mint("conv1", "Globex", "client-name", ir);
    expect(s1).toBe(s2);
    expect(s1).not.toBe("Globex");
  });

  it("isolates conversations: conv2's map has no conv1 entries", async () => {
    const vault = new Vault(new MemoryVaultStore());
    await vault.mint("conv1", "Globex", "client-name", ir);
    expect((await vault.rehydrationMap("conv2")).size).toBe(0);
  });

  it("re-mints on surrogate collision with a different real value", async () => {
    const store = new MemoryVaultStore();
    const vault = new Vault(store);
    // Deterministic construction: precompute what "Initech" WOULD get unsalted,
    // then occupy that surrogate for a different real value first.
    const wouldGet = generateSurrogate("org-name", "Initech", "conv1\u0000client-name\u0000Initech");
    await store.put("conv1", { entries: [{ real: "Other Corp", surrogate: wouldGet, entityType: "client-name" }] });
    const minted = await vault.mint("conv1", "Initech", "client-name", ir);
    expect(minted).not.toBe(wouldGet);
  });

  it("refuses to mint neverPseudonymize entityTypes (defense in depth)", async () => {
    const vault = new Vault(new MemoryVaultStore());
    await expect(vault.mint("conv1", "AKIAIOSFODNN7EXAMPLE", "aws-key", ir)).rejects.toThrow(/never be pseudonymized/i);
  });

  it("throws on unknown entityType", async () => {
    const vault = new Vault(new MemoryVaultStore());
    await expect(vault.mint("conv1", "x", "nope", ir)).rejects.toThrow(/unknown entityType/i);
  });

  it("rehydrationMap inverts the minting", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const s = await vault.mint("conv1", "Globex", "client-name", ir);
    const map = await vault.rehydrationMap("conv1");
    expect(map.get(s)).toBe("Globex");
    expect(map.size).toBe(1);
  });

  it("uses the entityType's surrogateKind (id-number for in-pan)", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const s = await vault.mint("conv1", "ABCPD1234E", "in-pan", ir);
    expect(s).toMatch(/^[A-Z]{5}[0-9]{4}[A-Z]$/);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `Cannot find module '../../src/pseudo/vault.js'`.

- [ ] **Step 3: Implement**

`packages/core/src/pseudo/vault.ts`:

```ts
import type { PolicyIr } from "../policy/types.js";
import { generateSurrogate } from "./generators.js";

export interface VaultEntry {
  real: string;
  surrogate: string;
  entityType: string;
}

export interface VaultRecord {
  entries: VaultEntry[];
}

/**
 * Storage seam: MemoryVaultStore for Node (tests, eval harness); Plan 6 adds an
 * IndexedDB store in the extension that encrypts values with crypto.ts helpers.
 * Async throughout because IndexedDB is.
 */
export interface VaultStore {
  get(conversationId: string): Promise<VaultRecord | undefined>;
  put(conversationId: string, record: VaultRecord): Promise<void>;
}

export class MemoryVaultStore implements VaultStore {
  private readonly records = new Map<string, VaultRecord>();
  async get(conversationId: string): Promise<VaultRecord | undefined> {
    return this.records.get(conversationId);
  }
  async put(conversationId: string, record: VaultRecord): Promise<void> {
    this.records.set(conversationId, record);
  }
}

/**
 * Per-conversation real ⇄ surrogate mapping (spec §5.4). Honest framing: this
 * is a reversible mapping table, not cryptography — the security property is
 * that real values never leave the machine.
 */
export class Vault {
  constructor(private readonly store: VaultStore) {}

  /**
   * Deterministic per (conversationId, entityType, real): re-minting returns the
   * existing surrogate (referential integrity across turns). Surrogates are
   * unique per conversation — a collision with another real value's surrogate
   * re-rolls with a salt.
   */
  async mint(conversationId: string, real: string, entityTypeId: string, ir: PolicyIr): Promise<string> {
    const entity = ir.entityTypes.find((e) => e.id === entityTypeId);
    if (!entity) throw new Error(`unknown entityType "${entityTypeId}"`);
    if (entity.neverPseudonymize) {
      // Schema already rejects pseudonymize actions for these; this is the
      // runtime backstop (defense in depth — a fake credential is a lie).
      throw new Error(`entityType "${entityTypeId}" must never be pseudonymized`);
    }

    const record = (await this.store.get(conversationId)) ?? { entries: [] };
    const existing = record.entries.find((e) => e.real === real && e.entityType === entityTypeId);
    if (existing) return existing.surrogate;

    const kind = entity.surrogateKind ?? "opaque";
    const baseKey = `${conversationId}\u0000${entityTypeId}\u0000${real}`;
    let surrogate: string;
    for (let salt = 0; ; salt++) {
      surrogate = generateSurrogate(kind, real, salt === 0 ? baseKey : `${baseKey}!${salt}`);
      if (!record.entries.some((e) => e.surrogate === surrogate)) break;
    }

    record.entries.push({ real, surrogate, entityType: entityTypeId });
    await this.store.put(conversationId, record);
    return surrogate;
  }

  /** surrogate → real, for response rehydration. */
  async rehydrationMap(conversationId: string): Promise<Map<string, string>> {
    const record = await this.store.get(conversationId);
    const map = new Map<string, string>();
    for (const e of record?.entries ?? []) map.set(e.surrogate, e.real);
    return map;
  }
}
```

- [ ] **Step 4: Verify pass** — 166 tests green, typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): per-conversation vault with deterministic minting and referential integrity"
```

---

### Task 5: AES-GCM at-rest helpers

**Files:**
- Create: `packages/core/src/pseudo/crypto.ts`
- Test: `packages/core/test/pseudo/crypto.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/crypto.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  decryptString, encryptString, exportVaultKey, generateVaultKey, importVaultKey,
} from "../../src/pseudo/crypto.js";

describe("vault crypto (AES-GCM)", () => {
  it("round-trips a string", async () => {
    const key = await generateVaultKey();
    const payload = await encryptString(key, "Globex ⇄ Vantor");
    expect(await decryptString(key, payload)).toBe("Globex ⇄ Vantor");
  });

  it("uses a fresh IV per encryption (same plaintext, different ciphertext)", async () => {
    const key = await generateVaultKey();
    expect(await encryptString(key, "same")).not.toBe(await encryptString(key, "same"));
  });

  it("fails to decrypt with the wrong key", async () => {
    const k1 = await generateVaultKey();
    const k2 = await generateVaultKey();
    const payload = await encryptString(k1, "secret");
    await expect(decryptString(k2, payload)).rejects.toThrow();
  });

  it("export/import round-trips the key", async () => {
    const key = await generateVaultKey();
    const imported = await importVaultKey(await exportVaultKey(key));
    const payload = await encryptString(key, "hello");
    expect(await decryptString(imported, payload)).toBe("hello");
  });
});
```

- [ ] **Step 2: Run to verify failure** — `Cannot find module '../../src/pseudo/crypto.js'`.

- [ ] **Step 3: Implement**

`packages/core/src/pseudo/crypto.ts`:

```ts
/**
 * AES-GCM helpers for vault-at-rest encryption (spec §5.4). WebCrypto only —
 * `crypto.subtle` and `crypto.getRandomValues` are WHATWG globals in browsers
 * and Node 20+; no node: imports (firewall). Plan 6's IndexedDB store encrypts
 * VaultRecord values with these; the session key lives in chrome.storage.session
 * there. Encryption-at-rest is hygiene, not the core security property.
 */

const IV_BYTES = 12;

function bytesToBase64(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function generateVaultKey(): Promise<CryptoKey> {
  return crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt", "decrypt"]);
}

export async function exportVaultKey(key: CryptoKey): Promise<string> {
  return bytesToBase64(new Uint8Array(await crypto.subtle.exportKey("raw", key)));
}

export async function importVaultKey(b64: string): Promise<CryptoKey> {
  return crypto.subtle.importKey("raw", base64ToBytes(b64), { name: "AES-GCM" }, true, ["encrypt", "decrypt"]);
}

/** Returns base64(iv ‖ ciphertext); IV is random per call — never reuse under GCM. */
export async function encryptString(key: CryptoKey, plaintext: string): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const ct = new Uint8Array(
    await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, new TextEncoder().encode(plaintext)),
  );
  const packed = new Uint8Array(iv.length + ct.length);
  packed.set(iv, 0);
  packed.set(ct, iv.length);
  return bytesToBase64(packed);
}

export async function decryptString(key: CryptoKey, payload: string): Promise<string> {
  const packed = base64ToBytes(payload);
  const iv = packed.slice(0, IV_BYTES);
  const ct = packed.slice(IV_BYTES);
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
  return new TextDecoder().decode(pt);
}
```

- [ ] **Step 4: Verify pass** — 170 tests green, typecheck clean, firewall test green (no Buffer — base64 goes through btoa/atob).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): AES-GCM vault-at-rest helpers via WebCrypto"
```

---

### Task 6: Action application

**Files:**
- Create: `packages/core/src/pseudo/apply.ts`
- Test: `packages/core/test/pseudo/apply.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/apply.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { detect } from "../../src/detect/orchestrator.js";
import { loadPolicyIr } from "../../src/policy/load.js";
import { applyActions } from "../../src/pseudo/apply.js";
import { MemoryVaultStore, Vault } from "../../src/pseudo/vault.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const config = { tier0: true, tier1: false, tier2: false };

describe("applyActions", () => {
  it("pseudonymizes and redacts by span, leaves allow untouched", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const text = "PAN ABCPD1234E and key AKIAIOSFODNN7EXAMPLE here";
    const { findings } = await detect({ ir, provider: "chatgpt", text, config });
    // in-pan: block, aws-key: block → both present; craft manually instead:
    const manual = findings.map((f) =>
      f.entityType === "in-pan" ? { ...f, action: "pseudonymize" as const } : { ...f, action: "redact" as const },
    );
    const result = await applyActions(text, manual, vault, "conv1", ir);
    expect(result.blocked).toBe(false);
    expect(result.text).not.toContain("ABCPD1234E");
    expect(result.text).not.toContain("AKIAIOSFODNN7EXAMPLE");
    expect(result.text).toContain("[REDACTED:aws-key]");
    expect(result.text).toMatch(/PAN [A-Z]{5}[0-9]{4}[A-Z] and key/);
  });

  it("applied records slice the rewritten text exactly", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const text = "PAN ABCPD1234E and key AKIAIOSFODNN7EXAMPLE here";
    const { findings } = await detect({ ir, provider: "chatgpt", text, config });
    const manual = findings.map((f) =>
      f.entityType === "in-pan" ? { ...f, action: "pseudonymize" as const } : { ...f, action: "redact" as const },
    );
    const result = await applyActions(text, manual, vault, "conv1", ir);
    for (const a of result.applied) {
      expect(result.text.slice(a.newStart, a.newEnd)).toBe(a.replacement);
      expect(text.slice(a.start, a.end)).not.toBe(a.replacement);
    }
  });

  it("sets blocked when any finding's action is block, and still rewrites the rest", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const text = "PAN ABCPD1234E and key AKIAIOSFODNN7EXAMPLE here";
    const { findings } = await detect({ ir, provider: "chatgpt", text, config });
    // Fixture actions: in-pan → block, aws-key → block. Soften one to redact.
    const mixed = findings.map((f) =>
      f.entityType === "aws-key" ? { ...f, action: "redact" as const } : f,
    );
    const result = await applyActions(text, mixed, vault, "conv1", ir);
    expect(result.blocked).toBe(true);
    expect(result.text).toContain("ABCPD1234E"); // block does not rewrite
    expect(result.text).toContain("[REDACTED:aws-key]");
  });

  it("is the identity on empty findings", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const result = await applyActions("hello world", [], vault, "conv1", ir);
    expect(result).toEqual({ text: "hello world", blocked: false, applied: [] });
  });

  it("keeps referential integrity across messages in one conversation", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const mk = async (text: string) => {
      const { findings } = await detect({ ir, provider: "chatgpt", text, config });
      const manual = findings.map((f) => ({ ...f, action: "pseudonymize" as const }));
      return applyActions(text, manual, vault, "convX", ir);
    };
    const r1 = await mk("first mention ABCPD1234E ok");
    const r2 = await mk("second mention ABCPD1234E ok");
    const fake1 = r1.applied[0]!.replacement;
    expect(r2.applied[0]!.replacement).toBe(fake1);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `Cannot find module '../../src/pseudo/apply.js'`.

- [ ] **Step 3: Implement**

`packages/core/src/pseudo/apply.ts`:

```ts
import type { ResolvedFinding } from "../detect/types.js";
import type { PolicyIr } from "../policy/types.js";
import type { Vault } from "./vault.js";

export interface AppliedReplacement {
  /** Original-text span (== the finding's span). */
  start: number;
  end: number;
  /** Span of `replacement` in the rewritten text. */
  newStart: number;
  newEnd: number;
  replacement: string;
  entityType: string;
  action: "pseudonymize" | "redact";
}

export interface ApplyResult {
  text: string;
  /**
   * True when any finding's action is "block". Block does NOT rewrite its span —
   * the caller decides whether the message may be sent at all (spec §5.2: block-
   * severity findings disable "send unmodified"); other findings are still
   * rewritten so the review sheet can show the would-be result.
   */
  blocked: boolean;
  applied: AppliedReplacement[];
}

/**
 * Applies resolved actions to the message text. Findings are guaranteed
 * pairwise disjoint and sorted by start (DetectionResult contract), so a single
 * forward pass assembles the output; new offsets fall out of the assembly.
 */
export async function applyActions(
  text: string,
  findings: ResolvedFinding[],
  vault: Vault,
  conversationId: string,
  ir: PolicyIr,
): Promise<ApplyResult> {
  const blocked = findings.some((f) => f.action === "block");
  const sorted = [...findings].sort((a, b) => a.start - b.start);

  let out = "";
  let cursor = 0;
  const applied: AppliedReplacement[] = [];

  for (const f of sorted) {
    if (f.action !== "pseudonymize" && f.action !== "redact") continue;
    const replacement =
      f.action === "pseudonymize"
        ? await vault.mint(conversationId, f.text, f.entityType, ir)
        : `[REDACTED:${f.entityType}]`;
    out += text.slice(cursor, f.start);
    const newStart = out.length;
    out += replacement;
    applied.push({
      start: f.start,
      end: f.end,
      newStart,
      newEnd: out.length,
      replacement,
      entityType: f.entityType,
      action: f.action,
    });
    cursor = f.end;
  }
  out += text.slice(cursor);

  return { text: out, blocked, applied };
}
```

- [ ] **Step 4: Verify pass** — 175 tests green, typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): action application — pseudonymize/redact over disjoint finding spans"
```

---

### Task 7: Text rehydration

**Files:**
- Create: `packages/core/src/pseudo/rehydrate.ts`
- Test: `packages/core/test/pseudo/rehydrate.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/rehydrate.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { maxSurrogateLength, rehydrateText, surrogatePattern } from "../../src/pseudo/rehydrate.js";

describe("rehydrateText", () => {
  it("replaces every known surrogate", () => {
    const map = new Map([["Vantor", "Globex"], ["Anjali Verma", "Priya Sharma"]]);
    expect(rehydrateText("Vantor hired Anjali Verma. Vantor won.", map)).toBe(
      "Globex hired Priya Sharma. Globex won.",
    );
  });

  it("prefers the longest surrogate on containment", () => {
    const map = new Map([["AB", "x"], ["ABC", "y"]]);
    expect(rehydrateText("ABC AB", map)).toBe("y x");
  });

  it("escapes regex metacharacters in surrogates", () => {
    const map = new Map([["Vellum & Gray (Ltd)", "RealCo"]]);
    expect(rehydrateText("per Vellum & Gray (Ltd) filing", map)).toBe("per RealCo filing");
  });

  it("is the identity on an empty map", () => {
    expect(rehydrateText("nothing here", new Map())).toBe("nothing here");
  });
});

describe("helpers", () => {
  it("maxSurrogateLength over the map's keys", () => {
    expect(maxSurrogateLength(new Map([["ab", "1"], ["abcd", "2"]]))).toBe(4);
    expect(maxSurrogateLength(new Map())).toBe(0);
  });

  it("surrogatePattern is undefined for an empty map", () => {
    expect(surrogatePattern(new Map())).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run to verify failure** — `Cannot find module '../../src/pseudo/rehydrate.js'`.

- [ ] **Step 3: Implement**

`packages/core/src/pseudo/rehydrate.ts` (transform stream lands in Task 8 — this task is the pure-text half):

```ts
/**
 * Rehydration: surrogate → real in provider responses (spec §5.4). Pure string
 * layer here; the streaming TransformStream builds on it. SSE/JSON framing is
 * adapter territory (Plan 6) — these functions operate on decoded text.
 */

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * One alternation over all surrogates, longest-first so a surrogate containing
 * another as a prefix is matched whole. Undefined when the map is empty.
 */
export function surrogatePattern(map: Map<string, string>): RegExp | undefined {
  if (map.size === 0) return undefined;
  const alts = [...map.keys()].sort((a, b) => b.length - a.length).map(escapeRegex);
  return new RegExp(alts.join("|"), "g");
}

export function maxSurrogateLength(map: Map<string, string>): number {
  let max = 0;
  for (const s of map.keys()) max = Math.max(max, s.length);
  return max;
}

export function rehydrateText(text: string, map: Map<string, string>): string {
  const pattern = surrogatePattern(map);
  if (!pattern) return text;
  return text.replace(pattern, (m) => map.get(m)!);
}
```

- [ ] **Step 4: Verify pass** — 181 tests green, typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): text rehydration with longest-first surrogate matching"
```

---

### Task 8: Streaming rehydration transform

**Files:**
- Modify: `packages/core/src/pseudo/rehydrate.ts`
- Test: `packages/core/test/pseudo/rehydrate-stream.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/rehydrate-stream.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { createRehydrateTransform, rehydrateText } from "../../src/pseudo/rehydrate.js";
import { mulberry32 } from "../../src/pseudo/seed.js";

/** Push chunks through the transform, collect the full output string. */
async function pump(chunks: string[], map: Map<string, string>): Promise<string> {
  const transform = createRehydrateTransform(map);
  const writer = transform.writable.getWriter();
  const reader = transform.readable.getReader();
  const outParts: string[] = [];
  const readAll = (async () => {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      outParts.push(value);
    }
  })();
  for (const c of chunks) await writer.write(c);
  await writer.close();
  await readAll;
  return outParts.join("");
}

const MAP = new Map([["Vantor", "Globex"], ["Anjali Verma", "Priya Sharma"]]);
const SAMPLE = "Regarding Vantor: Anjali Verma of Vantor signed. Thanks, Anjali Verma.";

describe("createRehydrateTransform", () => {
  it("matches rehydrateText for every single split point of the sample", async () => {
    const expected = rehydrateText(SAMPLE, MAP);
    for (let i = 1; i < SAMPLE.length; i++) {
      const got = await pump([SAMPLE.slice(0, i), SAMPLE.slice(i)], MAP);
      expect(got, `split at ${i}`).toBe(expected);
    }
  });

  it("handles many tiny chunks (seeded random chunking, 50 rounds)", async () => {
    const expected = rehydrateText(SAMPLE, MAP);
    const rng = mulberry32(0xc0ffee);
    for (let round = 0; round < 50; round++) {
      const chunks: string[] = [];
      let i = 0;
      while (i < SAMPLE.length) {
        const step = 1 + Math.floor(rng() * 7);
        chunks.push(SAMPLE.slice(i, i + step));
        i += step;
      }
      expect(await pump(chunks, MAP)).toBe(expected);
    }
  });

  it("passes text through untouched on an empty map", async () => {
    expect(await pump(["hello ", "world"], new Map())).toBe("hello world");
  });

  it("flushes a trailing partial that never completes", async () => {
    // "Vanto" is a prefix of "Vantor" but the stream ends — it must be emitted as-is.
    expect(await pump(["ends with Vanto"], MAP)).toBe("ends with Vanto");
  });

  it("rehydrates a surrogate that is the entire final chunk", async () => {
    expect(await pump(["deal with ", "Vantor"], MAP)).toBe("deal with Globex");
  });
});
```

- [ ] **Step 2: Run to verify failure** — no export `createRehydrateTransform`.

- [ ] **Step 3: Implement** — append to `packages/core/src/pseudo/rehydrate.ts`:

```ts
/**
 * Streaming rehydration (spec §5.4): replaces complete surrogates as chunks
 * arrive, holding back the last maxSurrogateLen−1 chars so a surrogate split
 * across a chunk boundary is never emitted half-replaced. flush() drains the
 * tail. Operates on decoded text chunks — provider SSE/JSON framing is the
 * adapter's job (Plan 6).
 *
 * Known limitation (accepted): the held-back tail is re-scanned with the next
 * chunk after replacement, so if a REAL value's suffix + following text happens
 * to spell a surrogate, it would be falsely replaced. Surrogates are generated
 * fakes, so this needs an adversarial coincidence; revisit if Plan 8's eval
 * ever observes it.
 */
export function createRehydrateTransform(map: Map<string, string>): TransformStream<string, string> {
  const pattern = surrogatePattern(map);
  const holdback = Math.max(0, maxSurrogateLength(map) - 1);
  let tail = "";
  const replaceAll = (s: string): string => (pattern ? s.replace(pattern, (m) => map.get(m)!) : s);

  return new TransformStream<string, string>({
    transform(chunk, controller) {
      if (!pattern) {
        controller.enqueue(chunk);
        return;
      }
      const buffer = replaceAll(tail + chunk);
      const emitEnd = Math.max(0, buffer.length - holdback);
      tail = buffer.slice(emitEnd);
      if (emitEnd > 0) controller.enqueue(buffer.slice(0, emitEnd));
    },
    flush(controller) {
      if (tail) controller.enqueue(replaceAll(tail));
    },
  });
}
```

- [ ] **Step 4: Verify pass** — 186 tests green, typecheck clean. (TransformStream is a WHATWG global in Node ≥18; no imports.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): streaming rehydration transform with chunk-boundary holdback"
```

---

### Task 9: Public API + end-to-end integration

**Files:**
- Modify: `packages/core/src/index.ts`
- Test: `packages/core/test/pseudo/e2e.test.ts`

- [ ] **Step 1: Write the failing test**

`packages/core/test/pseudo/e2e.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  applyActions, createRehydrateTransform, detect, loadPolicyIr, MemoryVaultStore, Vault,
  type Finding, type SpanTagger,
} from "../../src/index.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));

/** Stub tier-1 tagger that flags every "Globex" occurrence as client-name. */
const globexTagger: SpanTagger = {
  async tag(segments, taggedIr): Promise<Finding[]> {
    const findings: Finding[] = [];
    for (const seg of segments) {
      for (let idx = seg.text.indexOf("Globex"); idx !== -1; idx = seg.text.indexOf("Globex", idx + 1)) {
        findings.push({
          start: seg.start + idx, end: seg.start + idx + 6, text: "Globex",
          entityType: "client-name",
          severity: taggedIr.entityTypes.find((e) => e.id === "client-name")!.severity,
          tier: 1, source: "stub-tagger", confidence: 0.9,
        });
      }
    }
    return findings;
  },
};

describe("end-to-end: detect → apply → rehydrate", () => {
  it("round-trips a pseudonymized conversation through a streamed response", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const text = "Draft an email to Globex about the renewal. Globex prefers Q3.";

    const { findings } = await detect({
      ir, provider: "chatgpt", text,
      config: { tier0: true, tier1: true, tier2: false },
      engines: { tier1: globexTagger },
    });
    expect(findings).toHaveLength(2);
    expect(findings.every((f) => f.action === "pseudonymize")).toBe(true);

    const outbound = await applyActions(text, findings, vault, "conv-e2e", ir);
    expect(outbound.blocked).toBe(false);
    expect(outbound.text).not.toContain("Globex");
    const surrogate = outbound.applied[0]!.replacement;
    expect(outbound.applied[1]!.replacement).toBe(surrogate); // referential integrity

    // Simulate the provider echoing the surrogate in a streamed response,
    // split mid-surrogate to exercise the holdback.
    const response = `Sure — here's the email to ${surrogate} about their renewal.`;
    const cut = response.indexOf(surrogate) + Math.ceil(surrogate.length / 2);
    const transform = createRehydrateTransform(await vault.rehydrationMap("conv-e2e"));
    const writer = transform.writable.getWriter();
    const reader = transform.readable.getReader();
    const parts: string[] = [];
    const readAll = (async () => {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
      }
    })();
    await writer.write(response.slice(0, cut));
    await writer.write(response.slice(cut));
    await writer.close();
    await readAll;

    expect(parts.join("")).toBe(`Sure — here's the email to Globex about their renewal.`);
  });

  it("provider override changes the outbound treatment end-to-end", async () => {
    const vault = new Vault(new MemoryVaultStore());
    const text = "Summarize the Globex contract.";
    const { findings } = await detect({
      ir, provider: "deepseek", text,
      config: { tier0: true, tier1: true, tier2: false },
      engines: { tier1: globexTagger },
    });
    expect(findings[0]!.action).toBe("redact"); // deepseek override in the fixture
    const outbound = await applyActions(text, findings, vault, "conv-ds", ir);
    expect(outbound.text).toBe("Summarize the [REDACTED:client-name] contract.");
  });
});
```

- [ ] **Step 2: Run to verify failure** — the barrel does not export the pseudo layer yet (`applyActions` etc. missing).

- [ ] **Step 3: Implement** — append to `packages/core/src/index.ts`:

```ts
export { fnv1a64, mulberry32, seededRng } from "./pseudo/seed.js";
export { generateSurrogate } from "./pseudo/generators.js";
export { MemoryVaultStore, Vault, type VaultEntry, type VaultRecord, type VaultStore } from "./pseudo/vault.js";
export { decryptString, encryptString, exportVaultKey, generateVaultKey, importVaultKey } from "./pseudo/crypto.js";
export { applyActions, type AppliedReplacement, type ApplyResult } from "./pseudo/apply.js";
export { createRehydrateTransform, maxSurrogateLength, rehydrateText, surrogatePattern } from "./pseudo/rehydrate.js";
```

- [ ] **Step 4: Verify pass** — full suite ≈188 tests green, `pnpm test` and `pnpm typecheck` clean FROM ROOT.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): pseudo layer public API and end-to-end pseudonymize/rehydrate test"
```

---

## Done criteria for this plan

`@sih/core` can take a message + `DetectionResult`, rewrite it per policy (deterministic format-preserving surrogates with per-conversation referential integrity; credentials un-mintable at two layers), and reverse the mapping in a streamed response including surrogates split across chunk boundaries — all in plain Node, with the storage and crypto seams Plan 6's extension needs.

**Next plans:** 3 — policy compiler CLI + the three policies; 4 — tier-1 (ORT-web) + Playwright harness.

## Deviations log

(entries added during execution, same convention as Plan 1)
- **Plan-file encoding fix (post Task 2):** the seed-key separator was written as raw NUL bytes (0x00) in four code blocks, corrupting the markdown. Replaced with the literal TS escape text `\u0000` (Task 2 determinism test, Task 3 KEY, Task 4 baseKey + collision-test precompute). Task 2's already-committed test uses spaces in its determinism assertion — semantically neutral (asserts f(x)===f(x) only) and left as-is. Tasks 3+ MUST use the `\u0000` escape separator consistently; a space separator would be ambiguity-prone for real values containing spaces.

**Task 4 (review-mandated, D1-D5).** The Task 4 code block above predates three review rounds; where it disagrees with the entries below, the entries below win.

- **D1 - per-install salt in the seed key.** `Vault`'s constructor becomes `new Vault(store, installSalt: string)`, and the seed key becomes `installSalt` + `conversationId` + `entityTypeId` + `real`, `\u0000`-separated (this also supersedes the plan's `!${salt}` salt suffix - see D2). Rationale: every other component of the key is known to the other side (conversationId is provider-assigned, entityType ids are policy-public), so an unsalted deterministic key lets a provider precompute `surrogate(convId, type, candidate)` over a dictionary of plausible reals and read the mapping straight off the wire. The salt is not a secret key and FNV is not a PRF - this closes an offline-precomputation channel, nothing stronger. The caller generates it once and persists it beside the vault (Plan 6: `crypto.getRandomValues` to hex); tests use any fixed string. Pinned by: same inputs under two install salts give different surrogates (asserted on both a pool kind and the large id-number space, so it is not a 1-in-16 coincidence); determinism within one salt unchanged.
- **D2 - bounded uniqueness retries, then deterministic suffix expansion.** The planned `for (let salt = 0; ; salt++)` uniqueness loop cannot terminate once a pool kind runs out of candidates: the org pool holds 16 names, so a 17th distinct org real in one conversation livelocks the browser (person-name: the 145th, and in practice earlier). Replaced with 64 salted tries (`\u0000u${salt}` appended to the base key - the `u` keeps this ladder disjoint from the generator's own internal salt keys, which append a bare number to the same base) followed by `${basePick} ${k}` for k = 2, 3, ... until unused, where `basePick` is the salt-0 candidate. The suffix separator is a **space, not `\u0000`**: unlike the key, a surrogate is text that ships inside the message and has to come back through rehydration. Suffixed candidates are leak-checked with the generator's own `leaksReal`, now exported - a suffix can reintroduce the real ("Ledger 10" against a real of "Ledger 10 Holdings"). Termination is provable: only finitely many k collide with a finite entry set, and only finitely many can leak (equality fixes one k; a numeric token of the real, a handful more). Probe evidence: 17 distinct org reals in one conversation yield the 16 pool names then `Meridian Ops 2`; 30 reals stay unique; 200 person-name reals in one conversation stay unique. Pinned by the 17-real test (17 unique, at least one suffixed), a 5-real test (no suffixes - the normal path is untouched), and a leak-skip test that is mutation-verified: dropping the `leaksReal` call makes it mint `Nimbria Labs 10` for a real of "Ledger 10 Holdings".
- **D3 - empty/degenerate real rejected at mint.** `real` must be non-empty after trim, else `cannot pseudonymize an empty value`. The generator's degenerate guard only covers the scramble kinds; a pool kind would cheerfully mint "Vantor" for `""` and store an entry that rehydrates to nothing. This is a caller bug, not a policy decision, so it throws rather than routing through failMode.
- **D4 - injectable salt cap in `generateSurrogate`** (Task 3 file, landed with Task 4). Optional trailing `maxSalt = 64`; at cap 0 the exhaustion path fires immediately. That makes the previously-untestable no-echo privacy contract pinnable: the exhaustion error must carry neither the real value nor any of its tokens, because unlike the degenerate guard it can fire on a genuinely sensitive input. No behaviour change at the default.
- **D5 - leak tokenizer splits on `/[^a-z0-9]+/` instead of `/\s+/`** (Task 3 file, landed with Task 4). A real of "Rohan-Mehta" was a single opaque token, so a candidate of "Rohan Kapoor" cleared the leak check while handing the real first name back verbatim (probe-verified at roughly 7% of seed keys). The single-character exemption is unchanged. The same fix applies to the test file's `sharedTokens` helper.
- **Task 4 result:** 181 tests green (165 before, +3 generators, +13 vault), typecheck clean, no raw NULs in the tree. Landed as two commits, the generator changes (D4, D5, `leaksReal` export) ahead of the vault itself.

**Task 4 quality review (three Important fixes, landed in one follow-up commit).** All cheap in-file now, expensive once Plan 6 writes an IndexedDB store against the ambiguous contract.

- **R1 - mints serialize per conversation.** `mint` is a read-modify-write over one record, and a message with a dozen detected entities mints them together, so every mint read the record before any wrote it and all but the last entry was lost - surrogates on the wire that rehydrate to nothing. MemoryVaultStore's live-object aliasing accidentally rescued the already-has-a-record case but never the first mint; an IDB store (structured clone) loses every case. Fixed inside `Vault` with a promise chain keyed by conversationId: same conversation serializes, different conversations stay parallel, a rejected mint cannot wedge the queue, and the chain link drops itself once it is the settled tail. The `VaultStore` docstring now states the contract both ways (stores need not serialize; `Vault` does; `get`/`put` are snapshots). Fail-first: 12 concurrent mints of distinct reals gave a rehydration map of size 1. Mutation-verified after the fix by bypassing the queue - same failure returns, so the serialization and not the snapshot change is what carries the test.
- **R2 - MemoryVaultStore hands out snapshots.** `get` returned its live record, so a mutate-without-`put` persisted in Node and would silently vanish in the extension; `structuredClone` on both `get` and `put` makes the memory store behave like IndexedDB. Fail-first: pushing onto a get-returned record changed the store (length 2, expected 1).
- **R3 - cross-entry leak check** (privacy). Candidates were checked only against their own real, so with "Meridian Traders" already in the conversation a later value could draw the pool name "Meridian Ops" and ship a real client's distinctive token under another value's cover. The uniqueness predicate now rejects any candidate that leaks any existing entry's real, on both the salt ladder and the suffix path. Termination survives: each entry blocks finitely many k (equality fixes one; numeric tokens a handful more). Fail-first: the second mint returned exactly "Meridian Ops".
- **R4 - blank `installSalt` throws** at construction, since it silently reopens the precomputation channel D1 exists to close while still looking like a working vault.
- **R5 - key-composition comment corrected.** "NUL is the one character a real value cannot hold" was unenforced and false. The real argument is positional: `real` is the one segment we cannot constrain and nothing follows it, so it cannot shift a boundary ahead of it; a NUL in an earlier segment could alias two keys, but the cost is a shared seed, not a shared surrogate, because the uniqueness check still separates them. Rejected the alternative of stripping NULs from `real` - rehydration has to return the exact original text.
- **R6 - docstrings:** the suffix fallback's abandonment of format preservation is now stated as the deliberate trade it is (pool kinds are the design case; a suffixed PAN beats an ambiguous mapping), as are salt-rotation semantics (existing mappings survive in the store, only new mints diverge; Plan 6 decides regenerate-on-loss).
- **R7 - test gaps:** same-salt determinism from two FRESH stores (the cross-instance test shares a store and would pass even if minting were random), and re-mint stability on the suffixed fallback path (17 orgs, re-mint the 17th, same value back).
- **Result:** 189 tests green (181 before, +8 vault), typecheck clean, no raw NULs.

**Task 5 (AES-GCM at-rest helpers).**

- **`CryptoKey` derived as a local type alias, not imported.** The plan's Task 5 code does not typecheck as written: `CryptoKey` is a global *type* only under the DOM lib, and @types/node keeps it inside the `webcrypto` namespace, so all five signatures failed with `TS2304: Cannot find name 'CryptoKey'` under this package's `lib: ["ES2022"]` / `types: ["node"]`. (The global `crypto` *value* is declared, which is why `crypto.subtle`, `btoa`/`atob` and `TextEncoder` were fine — only the bare type name was unresolvable.) Fixed with a type-only alias, `export type CryptoKey = Parameters<typeof crypto.subtle.encrypt>[1]`; `subtle.encrypt` has a single non-overloaded signature, so the parameter resolves unambiguously. Runtime code is byte-identical to the plan. Rejected alternatives: `import type { webcrypto } from "node:crypto"` trips the firewall regex even as a type-only import, and adding `"DOM"` to `lib` would pull `fetch`/`localStorage`/`window` into scope package-wide and blunt the portability constraint the firewall exists to enforce. Deriving from the global `crypto` value also tracks whichever lib is in scope, so the signatures stay honest in the browser and in Node without hardcoding either. Exported so declaration emit stays valid (`declaration: true` in the base config). Probe-verified under the real project config that the alias is the true interface and not `any`: `0 extends 1 & Alias` is false, it carries `algorithm`/`extractable`/`type`/`usages`, it is bidirectionally assignable with `Awaited<ReturnType<typeof crypto.subtle.importKey>>`, and a plain object is rejected (the `@ts-expect-error` fires rather than going unused).
- **Task 5 result:** 193 tests green (189 before, +4 crypto), typecheck clean, firewall green (base64 through btoa/atob, no Buffer). Probes beyond the plan's four: empty-string round-trip; emoji/CJK/combining-mark round-trip; tampered ciphertext byte and tampered IV each rejected with `OperationError` (GCM auth holds); truncated payload rejected rather than silently returning garbage; 400 KB payload round-trips with no stack overflow (the char-at-a-time `+=` loop avoids the spread/`apply` trap); framing is exactly iv(12) + ct + tag(16). Noted, not a defect: a lone UTF-16 surrogate comes back as U+FFFD, inherent to any UTF-8 round-trip and irrelevant to vault values, which originate from valid text.

**Task 5 quality review (two Important pins, two Minor hardenings; one commit).** The probes above were run once by hand and then thrown away, which pins nothing in CI — these promote them.

- **Q1 - GCM contracts pinned as tests.** Tamper-evidence was verified only in a throwaway scratchpad script, so a silent AEAD downgrade (unauthenticated mode, a stripped tag, a decrypt returning garbage instead of throwing) would have left at-rest records forgeable with the suite still green. `crypto.test.ts` now pins: flipped ciphertext byte rejects, flipped IV byte rejects, truncated payload rejects, exact framing (`IV_BYTES + utf8len + TAG_BYTES` for a known plaintext), empty-string round-trip, multi-byte round-trip, and one ~400 KB payload (the only expensive case; it guards the char-at-a-time base64 loop against a spread/`apply` regression). Mutation-verified: `IV_BYTES` 12 to 16 fails the framing pin.
- **Q2 - `CryptoKey` alias shape pinned** in `test/pseudo/crypto.types.test-d.ts` (type-level asserts, checked by `tsc --noEmit` via the tsconfig `test` include; the `.test-d.ts` suffix keeps vitest from collecting it). Guards the one fragile property of the derivation: if a future lib gives `subtle.encrypt` a second overload, `Parameters<>` silently switches to the last one and `[1]` could retarget every signature in the module with no runtime test going red. Mutation-verified: aliasing to `any` fails both the `IsAny` assert and the now-unused `@ts-expect-error`.
- **Q3 - `importVaultKey` imports non-extractable** (changes a plan-snippet line: `true` to `false` in the `importKey` call). Nothing re-exports an imported key - the base64 the caller already holds *is* the persistable form - so this narrows the handle Plan 6 passes around. Not a containment boundary: whoever holds the base64 can import an extractable key. The plan's export/import test is unaffected (it only decrypts with the imported key). Pinned by `extractable === false` plus a rejecting `exportKey`; the scratchpad probe that asserted byte-identical re-export was dropped, since it is now impossible by design.
- **Q4 - legible truncation error** (adds a line to the plan snippet): `packed.length < IV_BYTES + TAG_BYTES` throws `payload too short` before `subtle.decrypt`, which otherwise sees a nonsense IV and returns a generic `OperationError` indistinguishable from a wrong key or real tampering. Mutation-verified: dropping the guard fails the truncation test.
- **Result:** 201 tests green (193 before, +8 crypto), typecheck clean, firewall green.

**Task 6 (review-mandated additions A1-A4, ahead of and inside the apply work).**

- **A1 - orchestrator allow-gap fix** (Plan 1 Task 1 quality review; landed as its own commit BEFORE the apply work, since it is a Plan-1 defect that Task 6 would have turned into a crash). `strictestAction` is cluster-wide, and `allow`(0) < `pseudonymize`(1) in `ACTION_RANK`, so a `neverPseudonymize` entityType sitting at `allow` -- legal policy, since the schema rejects only `pseudonymize` for these -- that overlaps a `pseudonymize` neighbour and WINS its cluster on severity was stamped `pseudonymize`. Task 6's `applyActions` then calls `vault.mint`, which refuses `neverPseudonymize` by design: a guaranteed message-level exception on a schema-valid IR, raised at apply time after detection reported success. Fixed with a per-winner `winnerAction(ir, winner, clusterAction)` in the winner loop: a `pseudonymize` cluster action on a `neverPseudonymize` winner escalates to `redact` -- the strict direction this file already commits to, and the one that still rewrites the span without minting a format-valid fake credential. Per WINNER, not per cluster: a non-credential winner in the same cluster keeps `pseudonymize`. Rejected `allow` as the resolution (forwards text the cluster said to rewrite) and rejected fixing it in `applyActions` (the wrong layer -- the IR-to-action mapping is the orchestrator's, and a caller reading `DetectionResult` should never see an unhonourable action). Fail-first: both new tests reported `pseudonymize` where they expect `redact`. Pinned by an escalation test (aws-key at `allow` overlapping a `client-name`, credential wins on critical severity) and a per-winner-granularity test (chain A-B-C where the credential and a second `client-name` both survive one cluster; only the credential escalates) -- the second is what stops a "simpler" cluster-level escalation.
- **A2 - apply-stage failures are blocking regardless of `failMode`** (doc contract on `applyActions`, plus a test). A rejection out of `applyActions` (vault refusal, generator exhaustion) means the text was not fully rewritten and no partial result exists -- the only string in hand is the original, real values and all. Routing that through `ir.failMode` inverts what failMode means: `open` is a decision about a message nobody managed to SCAN, and reusing it here turns "we could not check this" into "we checked it, found things the policy said to rewrite, failed to rewrite them, and sent them anyway", which leaks precisely what the policy named. Documented on `applyActions`; pinned by a test that constructs a `pseudonymize` action on a `neverPseudonymize` entityType by hand (a hostile or buggy caller -- neither the schema nor the orchestrator can produce it, especially after A1) and asserts the vault's refusal propagates. Mutation-verified: wrapping the `vault.mint` call in `try/catch { continue; }` fails that test alone, so it is the propagation and not some other assertion carrying it.
- **A3 - minting stays sequential** (`await` inside the loop, comment only). Deliberately NOT `Promise.all`: `Vault` already serializes mints per conversation through its promise chain (Task 4 review R1), so gathering buys no concurrency at all, while making the order in which candidates claim surrogates depend on scheduling -- which makes the uniqueness ladder's suffix assignment (`Vantor`, `Vantor 2`) nondeterministic for the same input. The comment says so, since "await in a loop" reads as an oversight to a future reader or linter.
- **A4 - residual spans are knowingly uncovered** (doc, carried from Plan 1's merge review). A merge LOSER can be wider than the winner that displaced it (merge.ts: winners keep their own span), so a cluster can leave remainder characters some rule once called sensitive covered by no surviving finding. `applyActions` rewrites exactly the resolved spans and nothing else. The motivating case argues for that: "SECRET_TOKEN=<secret>" scored as one entropy run against a regex matching just the secret leaves the key NAME as the remainder, and rewriting it costs utility for nothing. Widening to cluster unions (for critical clusters, say) is the obvious alternative and is a deliberate non-feature for now -- revisit if Plan 8's eval surfaces leaks through residuals. `clusterOverlapping` already exposes the losers such a policy would need, so nothing forecloses it.
- **Vault-signature test adaptation.** The plan's Task 6 tests construct `new Vault(new MemoryVaultStore())`, which predates D1's `installSalt` parameter. Every construction in `apply.test.ts` goes through a `newVault()` helper passing a fixed `"test-salt"`; behaviour is unaffected (the salt only shifts which surrogate a value draws, and no assertion names a specific surrogate).
- **Task 6 result:** 209 tests green (201 before, +2 orchestrator escalation, +6 apply), typecheck clean from root, firewall green, no raw NULs. Landed as two commits, the A1 orchestrator fix ahead of the apply work. Note for Task 9: `applyActions` is deliberately not exported from `index.ts` yet -- the whole pseudo layer's barrel exports are Task 9's step.

**Task 7 (rehydration, with review-mandated B1 and Task 6's ride-alongs).**

- **B1 - digit-boundary lookahead on `surrogatePattern`** (mandated by Task 4's review; changes the plan's Task 7 snippet). The pattern becomes `` new RegExp(`(?:${alts.join("|")})(?![0-9])`, "g") ``. The plan's unboundaried alternation is unsafe precisely because of D2: the pool-exhaustion ladder mints digit-suffixed surrogates as a matter of routine ("Vantor", then "Vantor 2"), so a conversation holding both meets ordinary text like "Vantor 2024 report" — and longest-first then matches "Vantor 2", rehydrating to the SECOND entity's real value spliced onto "024". That is a misattributed mention plus a mangled year, in the direction that hurts most (the user reads the wrong client's name). With the lookahead, "Vantor 2" fails at the "2|024" seam, the alternation retries — longest-first still holds among the candidates that clear the boundary — "Vantor" matches, and the following space passes. Fail-first evidence against the plan's snippet, implemented verbatim first: `"Vantor 2024 report"` produced `"Borealis Ltd024 report"` and `"Vantor 22"` produced `"Borealis Ltd2"`; both green after the lookahead, with the other 8 tests unaffected (so the boundary is what carries them).
- **B1 scope: digits only, deliberately NOT `(?!\w)`.** A letter boundary would refuse to match "Vantor" inside "Vantors", and models pluralize surrogates freely, so that mention would reach the user still wearing its fake name — the real value lost entirely, which is a worse failure than an imperfect one. The digit rule errs toward base-name replacement instead: "Vantors" rehydrates to "Globexs" (one wrong trailing letter, meaning intact), and the genuinely ambiguous "Vantor 22" resolves to base + " 22" rather than "Vantor 2" + "2". Both trades are pinned as tests so they read as decisions, not drift, and the rationale is in the `surrogatePattern` docstring. The plan's containment test ("ABC AB" → "y x") is unaffected. Also pinned: a surrogate whose own last character is a digit still matches when the text does not continue the run ("Vantor 2 today").
- **Task 7 result:** 219 tests green (209 before, +10 rehydrate: the plan's 6 plus B1's 4), typecheck clean, no raw NULs. `rehydrate.ts` is otherwise byte-identical to the plan snippet; the streaming transform and barrel exports remain Tasks 8-9.

**Task 6 quality-review ride-alongs (second commit, R1-R3).**

- **R1 - `DetectionResult.findings` doc corrected.** It still told rewriters to walk the array in REVERSE and splice in place, which is guidance for an implementation that no longer exists: the shipped `applyActions` walks forward and assembles a new string. The comment now describes forward assembly and says why it is the better shape here — new offsets fall out of the output length as it is built, so a different-length replacement needs no offset bookkeeping and `AppliedReplacement.newStart/newEnd` come free, whereas reverse splicing cannot report new offsets without a second pass. The DISJOINT-and-sorted contract statement is unchanged; only the recommendation moved.
- **R2 - the allow branch is now actually exercised.** The test titled "leaves allow untouched" contained no allow finding, so the `continue` for non-rewriting actions was untested from that test's side. It now appends a hand-built `client-name` finding at `allow` (hand-built because client-name is tier 1, which is off in this config, and the fixture maps it to pseudonymize anyway) and asserts both halves of the contract: the span survives verbatim in the output, AND it is absent from `applied[]` — a caller diffing that list would otherwise be told about a replacement that never happened, with meaningless newStart/newEnd. Mutation-verified: dropping the action guard rewrites the span to `[REDACTED:client-name]` and fails the new assertion.
- **R3 - three boundary pins on the forward assembly.** Every off-by-one in a cursor walk hides at the edges; a mid-string finding with slack on both sides passes even when the gap copy is wrong. Added: adjacent findings (`end === next.start`, both rewritten, asserted `applied[0].newEnd === applied[1].newStart` so the zero-length gap stays zero-length), a finding starting at offset 0, and a finding ending at `text.length` (asserted `newEnd === result.text.length`, i.e. nothing appended after the last span). Mutation-verified: `cursor = f.end - 1` fails all three. Honest scope note — the end-at-length case is a contract pin more than a mutation detector, since dropping the trailing `out += text.slice(cursor)` leaves it passing (there is nothing left to append); it exists to stop a future rewrite that reasons with slice arithmetic from mishandling `end === length`.
- **A test-fixture helper** (`findingAt`) re-derives `text` from `source.slice(start, end)` rather than taking it as an argument, which is the Finding contract stated in `detect/types.ts` and keeps the hand-built fixtures from drifting when a literal is edited.
- **Ride-along result:** 222 tests green (219 after Task 7's commit, +3 boundary pins; the allow assertions extended an existing test rather than adding one), typecheck clean.
