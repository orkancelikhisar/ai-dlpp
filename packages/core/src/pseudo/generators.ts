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
