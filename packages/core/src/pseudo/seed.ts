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
