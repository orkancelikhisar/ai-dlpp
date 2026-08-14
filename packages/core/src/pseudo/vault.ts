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
 *
 * Two contracts an implementation owes its caller:
 *
 * - **Snapshots, not aliases.** `get` returns a record the caller may mutate
 *   freely without touching storage, and `put` copies what it is handed. This
 *   is what IndexedDB does anyway (structured clone); a store that aliased its
 *   own record would let a forgotten `put` pass in Node and lose data in the
 *   extension.
 * - **No atomicity required.** A store need not serialize concurrent
 *   read-modify-write cycles — `Vault` serializes mints per conversation
 *   before they reach the store.
 */
export interface VaultStore {
  get(conversationId: string): Promise<VaultRecord | undefined>;
  put(conversationId: string, record: VaultRecord): Promise<void>;
}

export class MemoryVaultStore implements VaultStore {
  private readonly records = new Map<string, VaultRecord>();
  async get(conversationId: string): Promise<VaultRecord | undefined> {
    const record = this.records.get(conversationId);
    return record && structuredClone(record);
  }
  async put(conversationId: string, record: VaultRecord): Promise<void> {
    this.records.set(conversationId, structuredClone(record));
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
 *
 * Rotating the salt does not invalidate anything already minted: existing
 * mappings live in the store and rehydrate as before, and only newly minted
 * values diverge from what the old salt would have produced. What happens when
 * the salt is *lost* is Plan 6's call (regenerate and accept the divergence, or
 * treat the vault as unreadable).
 */
export class Vault {
  /** Tail of each conversation's mint chain; see `mint`. */
  private readonly pending = new Map<string, Promise<void>>();

  constructor(
    private readonly store: VaultStore,
    private readonly installSalt: string,
  ) {
    // A blank salt silently reopens the precomputation channel the salt exists
    // to close, and looks like a working vault while doing it.
    if (installSalt.trim() === "") throw new Error("vault requires a non-empty install salt");
  }

  /**
   * Deterministic per (installSalt, conversationId, entityType, real):
   * re-minting returns the existing surrogate (referential integrity across
   * turns). Surrogates are unique per conversation — a collision with another
   * real value's surrogate re-rolls with a salt.
   *
   * Mints for one conversation are serialized through a promise chain, because
   * minting is a read-modify-write over a single record and a message with a
   * dozen detected entities mints them together: without this, every mint reads
   * the record before any of them writes it and all but the last entry is lost
   * — surrogates on the wire that rehydrate to nothing. Different conversations
   * never wait on each other. The chain link swallows failures so one rejected
   * mint cannot wedge the queue, and drops itself once it is the settled tail.
   */
  async mint(conversationId: string, real: string, entityTypeId: string, ir: PolicyIr): Promise<string> {
    const previous = this.pending.get(conversationId) ?? Promise.resolve();
    const result = previous.then(() => this.mintOne(conversationId, real, entityTypeId, ir));
    let tail: Promise<void>;
    tail = result.then(
      () => {},
      () => {},
    ).then(() => {
      if (this.pending.get(conversationId) === tail) this.pending.delete(conversationId);
    });
    this.pending.set(conversationId, tail);
    return result;
  }

  private async mintOne(conversationId: string, real: string, entityTypeId: string, ir: PolicyIr): Promise<string> {
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
    // NUL separators, with `real` last. Nothing here validates that the earlier
    // segments are NUL-free, so injectivity rests on position instead: `real` is
    // the segment we cannot constrain, and nothing follows it, so no real value
    // can shift a boundary between the segments ahead of it. An earlier segment
    // carrying a NUL could still alias two keys, and the cost of that is a
    // shared seed, not a shared surrogate — the uniqueness check below hands the
    // second value its own regardless.
    const baseKey = `${this.installSalt}\u0000${conversationId}\u0000${entityTypeId}\u0000${real}`;
    const surrogate = mintUnique(kind, real, baseKey, record.entries);

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
 * A surrogate no other entry in the conversation already holds, and one that
 * leaks no entry's real. Uniqueness is what makes rehydration a function rather
 * than a guess; the leak half matters because the generator only ever checks a
 * candidate against its OWN real, so with "Rohan Mehta" already in the
 * conversation a later value could draw "Rohan Kapoor" and ship a real person's
 * first name under someone else's cover.
 *
 * Salted re-rolls alone cannot deliver uniqueness: the org pool holds 16 names,
 * so the 17th distinct org in one conversation has no unused candidate to draw
 * at any salt (person-name: the 145th), and an unbounded retry loop would spin
 * forever on a perfectly ordinary conversation. After UNIQUENESS_SALTS tries we
 * presume the pool is exhausted and expand deterministically instead —
 * "Vantor", "Vantor 2", "Vantor 3".
 *
 * That fallback gives up format preservation, which is a real cost but the
 * right trade: pool kinds are the design case (a 16-name list runs out in
 * ordinary use), while a scramble kind only lands here against an adversarially
 * stuffed store, and a suffixed PAN that no longer looks like a PAN still beats
 * a duplicate surrogate that makes rehydration ambiguous.
 *
 * Termination: k is unbounded while everything that can block it is finite —
 * finitely many entries to collide with, and finitely many leaks (equality with
 * a real fixes one k per entry; a numeric token in a real blocks a handful
 * more).
 */
function mintUnique(kind: SurrogateKind, real: string, baseKey: string, entries: VaultEntry[]): string {
  const taken = new Set(entries.map((e) => e.surrogate));
  const usable = (candidate: string) => !taken.has(candidate) && !entries.some((e) => leaksReal(candidate, e.real));
  let basePick = "";
  for (let salt = 0; salt < UNIQUENESS_SALTS; salt++) {
    // "u" prefix keeps this ladder disjoint from the generator's own internal
    // salt keys, which append a bare number to the same base.
    const candidate = generateSurrogate(kind, real, salt === 0 ? baseKey : `${baseKey}\u0000u${salt}`);
    if (salt === 0) basePick = candidate;
    if (usable(candidate)) return candidate;
  }
  for (let k = 2; ; k++) {
    // A space, not a NUL: this one is surrogate text that goes out in the
    // message and comes back through rehydration, not key material. The suffix
    // still has to clear the leak checks — "Ledger 10" would hand back a token
    // of a real named "Ledger 10 Holdings".
    const candidate = `${basePick} ${k}`;
    if (usable(candidate) && !leaksReal(candidate, real)) return candidate;
  }
}
