import type { PolicyIr, SurrogateKind } from "../policy/types.js";
import { generateSurrogate, leaksReal } from "./generators.js";

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

/** Uniqueness retries before the pool is presumed exhausted. */
const UNIQUENESS_SALTS = 64;

/**
 * Per-conversation real ⇄ surrogate mapping (spec §5.4). Honest framing: this
 * is a reversible mapping table, not cryptography — the security property is
 * that real values never leave the machine.
 *
 * `installSalt` is a per-install random string the caller generates once and
 * persists beside the vault (Plan 6: crypto.getRandomValues → hex). Without it
 * the seed key is built entirely from things the other side already knows —
 * conversationId is provider-assigned and entityType ids are policy-public — so
 * a provider could precompute surrogate(convId, type, candidate) over a
 * dictionary of real values and read the map straight off the wire. The salt is
 * not a secret key and FNV is not a PRF; this closes an offline-precomputation
 * channel, nothing stronger.
 */
export class Vault {
  constructor(
    private readonly store: VaultStore,
    private readonly installSalt: string,
  ) {}

  /**
   * Deterministic per (installSalt, conversationId, entityType, real):
   * re-minting returns the existing surrogate (referential integrity across
   * turns). Surrogates are unique per conversation — a collision with another
   * real value's surrogate re-rolls with a salt.
   */
  async mint(conversationId: string, real: string, entityTypeId: string, ir: PolicyIr): Promise<string> {
    const entity = ir.entityTypes.find((e) => e.id === entityTypeId);
    if (!entity) throw new Error(`unknown entityType "${entityTypeId}"`);
    if (entity.neverPseudonymize) {
      // Schema already rejects pseudonymize actions for these; this is the
      // runtime backstop (defense in depth — a fake credential is a lie).
      throw new Error(`entityType "${entityTypeId}" must never be pseudonymized`);
    }
    // The generator's degenerate guard only covers scramble kinds; a pool kind
    // would cheerfully mint "Vantor" for "" and add an entry whose surrogate
    // rehydrates to nothing. Nothing upstream should ask, so this is a bug
    // signal, not a policy decision.
    if (real.trim() === "") throw new Error("cannot pseudonymize an empty value");

    const record = (await this.store.get(conversationId)) ?? { entries: [] };
    const existing = record.entries.find((e) => e.real === real && e.entityType === entityTypeId);
    if (existing) return existing.surrogate;

    const kind = entity.surrogateKind ?? "opaque";
    // NUL separators so a real value containing the delimiter cannot forge a
    // different entity's key; NUL is the one character a real value cannot hold.
    const baseKey = `${this.installSalt}\u0000${conversationId}\u0000${entityTypeId}\u0000${real}`;
    const taken = new Set(record.entries.map((e) => e.surrogate));
    const surrogate = mintUnique(kind, real, baseKey, taken);

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

/**
 * A surrogate no other entry in the conversation already holds — uniqueness is
 * what makes rehydration a function rather than a guess.
 *
 * Salted re-rolls alone cannot deliver that: the org pool holds 16 names, so
 * the 17th distinct org in one conversation has no unused candidate to draw at
 * any salt (person-name: the 145th), and an unbounded retry loop would spin
 * forever on a perfectly ordinary conversation. After UNIQUENESS_SALTS tries we
 * presume the pool is exhausted and expand deterministically instead —
 * "Vantor", "Vantor 2", "Vantor 3" — which terminates because k is unbounded
 * while the entries that could collide with it are finite, and only finitely
 * many k can leak (equality with the real fixes one k; a numeric token of the
 * real, a handful more).
 */
function mintUnique(kind: SurrogateKind, real: string, baseKey: string, taken: Set<string>): string {
  let basePick = "";
  for (let salt = 0; salt < UNIQUENESS_SALTS; salt++) {
    // "u" prefix keeps this ladder disjoint from the generator's own internal
    // salt keys, which append a bare number to the same base.
    const candidate = generateSurrogate(kind, real, salt === 0 ? baseKey : `${baseKey}\u0000u${salt}`);
    if (salt === 0) basePick = candidate;
    if (!taken.has(candidate)) return candidate;
  }
  for (let k = 2; ; k++) {
    // A space, not a NUL: this one is surrogate text that goes out in the
    // message and comes back through rehydration, not key material. The suffix
    // still has to clear the leak check — "Ledger 10" would hand back a token
    // of a real named "Ledger 10 Holdings".
    const candidate = `${basePick} ${k}`;
    if (!taken.has(candidate) && !leaksReal(candidate, real)) return candidate;
  }
}
