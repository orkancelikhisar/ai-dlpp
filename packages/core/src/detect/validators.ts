/**
 * Validator library -- spec 3.1.
 *
 * This library is NAMED, AUDITED, and FIXED. Policy rules refer to validators
 * by name only; the policy compiler emits names, never code, so no policy can
 * introduce executable logic into the detection path. Adding or changing a
 * validator is therefore a human-reviewed change to this file, and the review
 * is what the whole design leans on -- keep each contract below exact.
 */

export type Validator = (candidate: string) => boolean;

/**
 * Shared input contract for the digit-checksum validators (luhn, verhoeff):
 * strip the separators real-world numbers are written with, then require at
 * least two ASCII digits and nothing else. Returns undefined when the candidate
 * does not qualify, so callers reject instead of checksumming junk. Both
 * validators share this one audited definition -- do not inline it back.
 */
function normalizeDigits(candidate: string): string | undefined {
  const digits = candidate.replace(/[\s-]/g, "");
  return /^\d{2,}$/.test(digits) ? digits : undefined;
}

/** Luhn mod-10 checksum. Strips `[\s-]`; requires >=2 ASCII digits. */
function luhn(candidate: string): boolean {
  const digits = normalizeDigits(candidate);
  if (digits === undefined) return false;
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

// Verhoeff (1969) checksum over the dihedral group D5 (used by Aadhaar).
// D is the group's multiplication table. The P rows are successive powers of
// the permutation s = (0 1 5 8 9 4 2 7)(3 6), P[0] being the identity; s has
// order 8, which is why the scheme cycles P with `i % 8`.
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

/** Verhoeff checksum. Strips `[\s-]`; requires >=2 ASCII digits. */
function verhoeff(candidate: string): boolean {
  const digits = normalizeDigits(candidate);
  if (digits === undefined) return false;
  let c = 0;
  const reversed = [...digits].reverse();
  for (let i = 0; i < reversed.length; i++) {
    // Each index below is in range by construction, which is what makes the
    // non-null assertions sound under noUncheckedIndexedAccess:
    //   digit       -- normalizeDigits admits only ASCII digits, so 0..9;
    //   permutation -- i % 8 is 0..7 and P.length === 8;
    //   c           -- every D and P entry is itself 0..9, so the running
    //                  check digit stays a valid D row index across iterations.
    const digit = reversed[i]!.charCodeAt(0) - 48;
    const permutation = P[i % 8]!;
    c = D[c]![permutation[digit]!]!;
  }
  return c === 0;
}

// Indian PAN: AAAPA9999A; 4th char = holder type.
const PAN_HOLDER_TYPES = new Set(["A", "B", "C", "F", "G", "H", "J", "L", "P", "T"]);

/**
 * Indian PAN structure. Deliberately asymmetric with the checksum validators
 * above: this does NO normalization, requiring an exact 10-character uppercase
 * match. A PAN is an identifier written as one token, so whitespace or dashes
 * inside a candidate mean the match was mis-scoped, not that it needs cleaning.
 */
function panStructure(candidate: string): boolean {
  if (!/^[A-Z]{5}[0-9]{4}[A-Z]$/.test(candidate)) return false;
  return PAN_HOLDER_TYPES.has(candidate[3]!);
}

/**
 * Character-frequency Shannon entropy in bits/char, over code points. Not a
 * Validator: rules use it as a numeric gate behind a regex (high-entropy
 * strings are secret-shaped), so it returns a score rather than a verdict.
 */
export function shannonEntropy(s: string): number {
  // Count and divide over the SAME unit. Iterating a string yields code points
  // while s.length counts UTF-16 code units, so dividing by s.length would make
  // the probabilities sum to <1 for any astral input (an emoji pair scoring 0.5
  // bits/char instead of 0). Materializing the code points keeps both in sync.
  const chars = [...s];
  if (chars.length === 0) return 0;
  const counts = new Map<string, number>();
  for (const ch of chars) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  let h = 0;
  for (const n of counts.values()) {
    const p = n / chars.length;
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

/**
 * JWT shape: three base64url parts whose header decodes to a JSON object
 * carrying a string `alg`. Structure only -- this never verifies the signature,
 * so `alg: "none"` and expired or forged tokens still match. That is intended:
 * a leaked token is reportable whether or not it would authenticate.
 */
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

const REGISTRY: Record<string, Validator> = {
  luhn,
  verhoeff,
  "pan-structure": panStructure,
  "jwt-shape": jwtShape,
};

export function hasValidator(name: string): boolean {
  // Own keys only: `in` would consult the prototype chain, so a rule naming
  // "toString" would pass the loader's check and then fail at getValidator time.
  return Object.hasOwn(REGISTRY, name);
}

export function getValidator(name: string): Validator {
  // Gate on the same own-key check rather than a bare REGISTRY[name]: that
  // lookup walks the prototype chain and would return Object.prototype.toString
  // for a rule named "toString" -- a "validator" returning a truthy
  // "[object ...]" string for every candidate, i.e. failing OPEN. Sharing one
  // check keeps hasValidator and getValidator over the same key set by design.
  if (!hasValidator(name)) throw new Error(`unknown validator "${name}"`);
  return REGISTRY[name]!;
}
