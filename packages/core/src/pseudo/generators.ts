import { PAN_HOLDER_TYPES } from "../detect/validators.js";
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

const PAN_SHAPE = /^[A-Z]{5}[0-9]{4}[A-Z]$/;
const SCRAMBLEABLE = /[A-Za-z0-9]/;
/** Retry ceiling: an unsatisfiable request must surface as an error, not a hang. */
const MAX_SALT = 64;
/** Split point for leak tokens: any run of non-alphanumerics, not just spaces. */
const TOKEN_SEPARATOR = /[^a-z0-9]+/;

function pick(rng: () => number, pool: string | readonly string[]): string {
  return pool[Math.floor(rng() * pool.length)]!;
}

/**
 * Deterministic under (kind, seedKey); salted retry guarantees the surrogate
 * never leaks the real value, which would be a non-pseudonymization.
 *
 * `maxSalt` exists to make the exhaustion path testable — at the default 64,
 * constructing a real whose every candidate leaks is impractical, so the
 * no-echo contract on that error would otherwise go unpinned. Callers have no
 * reason to pass it; the vault's own uniqueness retries are a separate loop.
 */
export function generateSurrogate(kind: SurrogateKind, real: string, seedKey: string, maxSalt = MAX_SALT): string {
  // A scramble returns its own input when there is nothing in it to scramble
  // ("", "----"), so the retry below could never converge — it would spin
  // forever. Fail fast instead and let the caller's failMode decide (Task 4).
  // Echoing `real` is safe in THIS message only: the guard fires exactly when
  // it holds no alphanumerics, i.e. punctuation that cannot be a secret.
  if (usesScramble(kind, real) && !SCRAMBLEABLE.test(real)) {
    throw new Error(`cannot generate surrogate for "${real}": no scrambleable characters`);
  }
  for (let salt = 0; salt < maxSalt; salt++) {
    // NUL separator, matching the base key's: a "#" would let a real value
    // containing "#1" alias a different entity's salted key.
    const rng = seededRng(salt === 0 ? seedKey : `${seedKey}\u0000${salt}`);
    const candidate = generate(kind, real, rng);
    if (!leaksReal(candidate, real)) return candidate;
  }
  // Reachable when every candidate collides — a person-name real that lists the
  // whole first-name pool, say. Deliberately does NOT echo `real`: unlike the
  // guard above, this one can fire on a genuinely sensitive value.
  throw new Error(`exhausted ${maxSalt} retries generating a ${kind} surrogate`);
}

/** Kinds whose output is a per-character scramble of the real (not a pool pick). */
function usesScramble(kind: SurrogateKind, real: string): boolean {
  return kind === "opaque" || (kind === "id-number" && !PAN_SHAPE.test(real));
}

/**
 * A candidate leaks if it equals the real OR reuses any of its multi-character
 * tokens: real "Rohan Mehta" drawing "Rohan Kapoor" clears a whole-string check
 * while handing back the real first name verbatim. Single-character tokens are
 * exempt — an initial is not a name, and counting one as a leak makes some
 * inputs unsatisfiable (a class-preserving scramble of "a b c … z" can never
 * clear the check, so a perfectly generatable value would start throwing).
 *
 * Tokens split on any non-alphanumeric run, not just whitespace: "Rohan-Mehta"
 * and "Rohan.Mehta" are as much a first name plus a last name as the spaced
 * form is, and treating them as one opaque token let the first name through.
 *
 * Exported for the vault, whose pool-exhaustion suffixes ("Vantor 2") are
 * candidates the generator never sees and so must be leak-checked there.
 */
export function leaksReal(candidate: string, real: string): boolean {
  if (candidate.toLowerCase() === real.toLowerCase()) return true;
  const realTokens = new Set(tokens(real));
  return tokens(candidate).some((t) => realTokens.has(t));
}

function tokens(s: string): string[] {
  return s.toLowerCase().split(TOKEN_SEPARATOR).filter((t) => t.length > 1);
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
  if (PAN_SHAPE.test(real)) {
    const holder = PAN_HOLDER_TYPES.has(real[3]!) ? real[3]! : "P";
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
