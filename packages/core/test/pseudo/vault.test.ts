import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { generateSurrogate } from "../../src/pseudo/generators.js";
import { MemoryVaultStore, Vault, type VaultRecord } from "../../src/pseudo/vault.js";
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

/**
 * Occupants' reals fuse their index into one token ("zzOccupant11", not
 * "occupant-11"): a bare numeric token would be blocked by the cross-entry leak
 * check and quietly move the suffix ladder, testing something other than what
 * the test names.
 */
function occupy(surrogates: string[]) {
  return { entries: surrogates.map((surrogate, i) => ({ real: `zzOccupant${i}`, surrogate, entityType: "client-name" })) };
}

/** The first probe real whose unsalted pick is `target`, for cross-entry setups. */
function realDrawing(target: string): string {
  for (let i = 0; i < 2000; i++) {
    const real = `Probe${i}`;
    if (generateSurrogate("org-name", real, baseKey("conv1", "client-name", real)) === target) return real;
  }
  throw new Error(`no probe real draws ${target}`);
}

function tokensOf(s: string): string[] {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 1);
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

  it("mints the same surrogate from a fresh store under the same install salt", async () => {
    // The test above shares a store, so a lookup of the stored entry satisfies
    // it even if minting were random. Two empty stores pin the derivation.
    const first = await new Vault(new MemoryVaultStore(), SALT).mint("conv1", "Globex", "client-name", ir);
    const second = await new Vault(new MemoryVaultStore(), SALT).mint("conv1", "Globex", "client-name", ir);
    expect(second).toBe(first);
  });

  it("rejects a blank install salt", () => {
    expect(() => new Vault(new MemoryVaultStore(), "")).toThrow(/install salt/i);
    expect(() => new Vault(new MemoryVaultStore(), "   ")).toThrow(/install salt/i);
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

  it("returns the same suffixed surrogate when a pool-exhausted real is re-minted", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const minted: string[] = [];
    for (let i = 0; i < 17; i++) minted.push(await vault.mint("conv1", `Realco${i}`, "client-name", ir));
    expect(minted[16]).toMatch(SUFFIXED);
    // Referential integrity has to hold on the fallback path too, or the 18th
    // turn renames a client the model has already been told about.
    expect(await vault.mint("conv1", "Realco16", "client-name", ir)).toBe(minted[16]);
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

  // Concurrency, store semantics, cross-entry leaks.

  it("keeps every entry when one conversation's mints run concurrently", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const reals = Array.from({ length: 12 }, (_, i) => `Realco${i}`);
    // A message with 12 detected entities mints them together. Read-modify-write
    // without serialization drops all but the last: every mint reads the record
    // before any of them writes it.
    const minted = await Promise.all(reals.map((real) => vault.mint("conv1", real, "client-name", ir)));
    const map = await vault.rehydrationMap("conv1");
    expect(map.size).toBe(12);
    expect(new Set(minted).size).toBe(12);
    for (const [surrogate, real] of map) expect(reals).toContain(real);
    expect([...map.keys()].sort()).toEqual([...minted].sort());
  });

  it("serializes per conversation without serializing across conversations", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    await Promise.all([
      ...Array.from({ length: 6 }, (_, i) => vault.mint("convA", `Realco${i}`, "client-name", ir)),
      ...Array.from({ length: 6 }, (_, i) => vault.mint("convB", `Realco${i}`, "client-name", ir)),
    ]);
    expect((await vault.rehydrationMap("convA")).size).toBe(6);
    expect((await vault.rehydrationMap("convB")).size).toBe(6);
  });

  it("does not let a rejected mint wedge the conversation's queue", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    const [failed, ok] = await Promise.allSettled([
      vault.mint("conv1", "x", "nope", ir),
      vault.mint("conv1", "Globex", "client-name", ir),
    ]);
    expect(failed!.status).toBe("rejected");
    expect(ok!.status).toBe("fulfilled");
  });

  it("MemoryVaultStore hands out snapshots, not live records", async () => {
    // IndexedDB structured-clones on the way out; a memory store that aliases
    // its own record would let a forgotten put pass every test here and fail
    // only in the extension.
    const store = new MemoryVaultStore();
    await store.put("conv1", { entries: [{ real: "Globex", surrogate: "Vantor", entityType: "client-name" }] });
    const fetched = (await store.get("conv1"))!;
    fetched.entries.push({ real: "Initech", surrogate: "Kitehill", entityType: "client-name" });
    expect((await store.get("conv1"))!.entries).toHaveLength(1);
    // ...and the caller's record cannot be mutated through the store either.
    const held: VaultRecord = { entries: [] };
    await store.put("conv2", held);
    held.entries.push({ real: "Initech", surrogate: "Kitehill", entityType: "client-name" });
    expect((await store.get("conv2"))!.entries).toHaveLength(0);
  });

  it("will not draw a surrogate that leaks another entry's real", async () => {
    const vault = new Vault(new MemoryVaultStore(), SALT);
    // "Meridian Traders" shares its distinctive token with the pool name
    // "Meridian Ops". Mint it, then mint a second real whose own unsalted pick
    // IS "Meridian Ops" — checking candidates only against their own real would
    // ship the first client's name to the provider under a second one's cover.
    await vault.mint("conv1", "Meridian Traders", "client-name", ir);
    const second = await vault.mint("conv1", realDrawing("Meridian Ops"), "client-name", ir);
    expect(second).not.toBe("Meridian Ops");
    expect(tokensOf(second)).not.toContain("meridian");
  });
});
