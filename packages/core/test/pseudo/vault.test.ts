import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { generateSurrogate } from "../../src/pseudo/generators.js";
import { MemoryVaultStore, Vault } from "../../src/pseudo/vault.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));

const SALT = "install-salt-a";

/** The documented key composition, restated here so a silent change fails loudly. */
function baseKey(conversationId: string, entityTypeId: string, real: string): string {
  return `${SALT}\u0000${conversationId}\u0000${entityTypeId}\u0000${real}`;
}

/**
 * The org pool by observation rather than import: minting is only ever supposed
 * to draw from it, so a test that needs "every pool member is taken" can collect
 * it from the generator itself. Deterministic (fixed keys) and self-checking.
 */
function orgPool(): string[] {
  const seen = new Set<string>();
  for (let i = 0; i < 500; i++) seen.add(generateSurrogate("org-name", "zzz-no-collision", `pool${i}`));
  expect(seen.size).toBe(16);
  return [...seen];
}

function occupy(surrogates: string[]) {
  return { entries: surrogates.map((surrogate, i) => ({ real: `occupant-${i}`, surrogate, entityType: "client-name" })) };
}

const SUFFIXED = /\s\d+$/;

describe("Vault", () => {
  it("mints deterministically: same (conversation, real, entityType) → same surrogate", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const s1 = await vault.mint("conv1", "Globex", "client-name", ir);
    const s2 = await vault.mint("conv1", "Globex", "client-name", ir);
    expect(s1).toBe(s2);
    expect(s1).not.toBe("Globex");
  });

  it("isolates conversations: conv2's map has no conv1 entries", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    await vault.mint("conv1", "Globex", "client-name", ir);
    expect((await vault.rehydrationMap("conv2")).size).toBe(0);
  });

  it("re-mints on surrogate collision with a different real value", async () => {
    const store = new MemoryVaultStore();
    const vault = new Vault(store, SALT);
    // Deterministic construction: precompute what "Initech" WOULD get at salt 0,
    // then occupy that surrogate for a different real value first.
    const wouldGet = generateSurrogate("org-name", "Initech", baseKey("conv1", "client-name", "Initech"));
    await store.put("conv1", { entries: [{ real: "Other Corp", surrogate: wouldGet, entityType: "client-name" }] });
    const minted = await vault.mint("conv1", "Initech", "client-name", ir);
    expect(minted).not.toBe(wouldGet);
  });

  it("refuses to mint neverPseudonymize entityTypes (defense in depth)", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    await expect(vault.mint("conv1", "AKIAIOSFODNN7EXAMPLE", "aws-key", ir)).rejects.toThrow(/never be pseudonymized/i);
  });

  it("throws on unknown entityType", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    await expect(vault.mint("conv1", "x", "nope", ir)).rejects.toThrow(/unknown entityType/i);
  });

  it("rehydrationMap inverts the minting", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const s = await vault.mint("conv1", "Globex", "client-name", ir);
    const map = await vault.rehydrationMap("conv1");
    expect(map.get(s)).toBe("Globex");
    expect(map.size).toBe(1);
  });

  it("uses the entityType's surrogateKind (id-number for in-pan)", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const s = await vault.mint("conv1", "ABCPD1234E", "in-pan", ir);
    expect(s).toMatch(/^[A-Z]{5}[0-9]{4}[A-Z]$/);
  });

  // D1 — per-install salt.

  it("gives different surrogates under different install salts", async () => {
    const a = await new Vault(new MemoryVaultStore(), "install-salt-a").mint("conv1", "Globex", "client-name", ir);
    const b = await new Vault(new MemoryVaultStore(), "install-salt-b").mint("conv1", "Globex", "client-name", ir);
    expect(a).not.toBe(b);
    // The id-number space is large enough that this one is a real signal, not
    // a 1-in-16 pool coincidence.
    const panA = await new Vault(new MemoryVaultStore(), "install-salt-a").mint("conv1", "ABCPD1234E", "in-pan", ir);
    const panB = await new Vault(new MemoryVaultStore(), "install-salt-b").mint("conv1", "ABCPD1234E", "in-pan", ir);
    expect(panA).not.toBe(panB);
  });

  it("stays deterministic across vault instances sharing an install salt and store", async () => {
    const store = new MemoryVaultStore();
    const first = await new Vault(store, SALT).mint("conv1", "Globex", "client-name", ir);
    const second = await new Vault(store, SALT).mint("conv1", "Globex", "client-name", ir);
    expect(second).toBe(first);
  });

  // D2 — provable termination when the pool runs out.

  it("mints 17 distinct orgs in one conversation, falling back to suffixes", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const minted: string[] = [];
    for (let i = 0; i < 17; i++) minted.push(await vault.mint("conv1", `Realco${i}`, "client-name", ir));
    // The org pool holds 16 names, so the 17th real cannot draw an unused one at
    // any salt — without the suffix fallback the salt loop would never converge.
    expect(new Set(minted).size).toBe(17);
    expect(minted.filter((s) => SUFFIXED.test(s)).length).toBeGreaterThanOrEqual(1);
  });

  it("leaves the normal path unsuffixed when the pool is not exhausted", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const minted: string[] = [];
    for (let i = 0; i < 5; i++) minted.push(await vault.mint("conv1", `Realco${i}`, "client-name", ir));
    expect(new Set(minted).size).toBe(5);
    expect(minted.filter((s) => SUFFIXED.test(s))).toEqual([]);
  });

  it("skips a suffixed candidate that would leak a token of the real", async () => {
    const store = new MemoryVaultStore();
    const real = "Ledger 10 Holdings";
    const basePick = generateSurrogate("org-name", real, baseKey("conv1", "client-name", real));
    // Whole pool taken plus suffixes 2..9, so the next free suffix is 10 — which
    // reuses the real's own "10" token and must be skipped like any other leak.
    await store.put("conv1", occupy([...orgPool(), ...[2, 3, 4, 5, 6, 7, 8, 9].map((k) => `${basePick} ${k}`)]));
    const minted = await new Vault(store, SALT).mint("conv1", real, "client-name", ir);
    expect(minted).toBe(`${basePick} 11`);
  });

  // D3 — degenerate reals.

  it("refuses to mint an empty or whitespace-only real", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    await expect(vault.mint("conv1", "", "client-name", ir)).rejects.toThrow(/cannot pseudonymize an empty value/i);
    await expect(vault.mint("conv1", "   ", "client-name", ir)).rejects.toThrow(/cannot pseudonymize an empty value/i);
    // Pool kinds would happily mint a surrogate for "" — the generator's
    // degenerate guard only covers the scramble kinds.
    expect((await vault.rehydrationMap("conv1")).size).toBe(0);
  });
});
