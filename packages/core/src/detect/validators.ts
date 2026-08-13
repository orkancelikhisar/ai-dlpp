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
