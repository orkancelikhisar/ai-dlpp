import { getValidator } from "@sih/core";

/**
 * The fictional universe and the injection catalogue -- spec 6.2's "fictional
 * universe per policy" and "controlled dimensions".
 *
 * ## Everything here is invented, and there is a test that says so
 *
 * Two independent leaks would make this corpus measure the wrong thing, and
 * both have happened in this repository before:
 *
 * - A gold value that also appears in the IR the model is prompted with. One
 *   earlier corpus had its only tier-1 gold value sitting in the IR's
 *   `examples` array, so the model was shown the answer. `corpus-universe.test.ts`
 *   asserts no minted value and no universe name occurs in
 *   `policies/compiled/p-fin.ir.json`.
 * - A value shared with the compiler's self-test corpus, which is generated
 *   from the same policy and is the thing `contamination.ts` exists to keep
 *   separate. The same test asserts absence there too.
 *
 * ## Why identifiers are minted rather than listed
 *
 * A listed identifier is a constant, and a constant that has to satisfy a
 * checksum is a constant somebody eventually edits by hand and breaks. Every
 * minted value here is checked against the REAL validator from `@sih/core`
 * (`pan-structure`, `verhoeff`) at mint time and the mint throws if it fails --
 * so the corpus cannot contain a "valid PAN" the shipping validator rejects,
 * and it cannot contain an "invalid" one the validator accepts. That check is
 * against the same code the extension runs, not a second copy of the rule.
 *
 * ## What that consistency also costs, stated because it is not free
 *
 * The same gate that makes the labels consistent SYSTEMATICALLY EXCLUDES the
 * compiled arm's own tier-0 failure mode. A validator disagreeing with the
 * world -- a real PAN `pan-structure` rejects, a non-PAN it accepts -- is the
 * one class of item on which tier 0 is wrong, and by construction no such item
 * can be minted here or admitted as a confusable (`surfaces.ts:161` throws if
 * the TAN confusable happens to pass `panStructure`). So per-type tier-0 recall
 * on `in-pan` and `in-aadhaar`, and tier-0's zero false-positive rate on the
 * PAN confusable, are upper bounds this corpus manufactured rather than
 * measurements of the world, and a prompting arm reasoning about the format
 * from the policy text gets no equivalent floor. The v2 manifest carries this
 * in `unvalidated`; it is repeated here because this is the file that does it.
 */

// -- the world --------------------------------------------------------------

/** The employer whose policy p-fin is. Never a client of itself. */
export const FIRM = "Harborline Capital Advisors";

/**
 * Organisations the universe places in a client / prospective-client /
 * counterparty role. Under p-fin §3.1 a mention of one of these IN THAT ROLE is
 * a `client-name`; the same string in a supplier role is not, which is the
 * whole point of `NON_CLIENT_ORGS` below.
 */
export const CLIENT_ORGS = [
  "Vetiver Logistics",
  "Saltmarsh Foods",
  "Kestrel Ironworks",
  "Palewater Shipping",
  "Umbral Textiles",
  "Ridgeline Dairy",
] as const;

/**
 * Organisations the universe places in roles p-fin §3 does not reach: a
 * software supplier, a printer, a venue. A span tagger will label these ORG
 * exactly as readily as a client name -- they are the same orthographic shape --
 * and p-fin demands nothing of them. They are the corpus's answer to "does the
 * arm classify, or does it just find capitalised nouns".
 */
export const NON_CLIENT_ORGS = [
  "Brightline Cloud Services",
  "Cobblestone Print Works",
  "Ferndale Conference Centre",
  "Sablewood Facilities",
] as const;

// -- minting ----------------------------------------------------------------

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/** The 10 PAN holder-type codes `pan-structure` accepts, in the validator's own set order. */
const HOLDER_TYPES = "ABCFGHJLPT";
/** The 16 uppercase letters that are NOT holder types; the invalid-PAN family draws from these. */
const NON_HOLDER_TYPES = [...LETTERS].filter((c) => !HOLDER_TYPES.includes(c)).join("");

function pick<T>(rng: () => number, xs: ArrayLike<T>): T {
  return xs[Math.floor(rng() * xs.length)]!;
}

function digits(rng: () => number, n: number): string {
  let s = "";
  for (let i = 0; i < n; i += 1) s += Math.floor(rng() * 10).toString();
  return s;
}

const panStructure = getValidator("pan-structure");
const verhoeff = getValidator("verhoeff");

/** A PAN the shipping `pan-structure` validator accepts. Throws if it does not. */
export function mintPan(rng: () => number): string {
  const pan =
    pick(rng, LETTERS) +
    pick(rng, LETTERS) +
    pick(rng, LETTERS) +
    pick(rng, HOLDER_TYPES) +
    pick(rng, LETTERS) +
    digits(rng, 4) +
    pick(rng, LETTERS);
  if (!panStructure(pan)) throw new Error(`minted PAN ${pan} fails pan-structure`);
  return pan;
}

/**
 * A PAN-SHAPED string the validator rejects, because its 4th character is not a
 * holder type. This is the confusable that separates "matched the regex" from
 * "is a PAN": p-fin's `in-pan-format` rule pairs the regex with the validator,
 * so a correct arm does not fire here and `certify.ts`'s max-recall sweep,
 * which drops validators on purpose, does.
 */
export function mintInvalidPan(rng: () => number): string {
  const pan =
    pick(rng, LETTERS) +
    pick(rng, LETTERS) +
    pick(rng, LETTERS) +
    pick(rng, NON_HOLDER_TYPES) +
    pick(rng, LETTERS) +
    digits(rng, 4) +
    pick(rng, LETTERS);
  if (panStructure(pan)) throw new Error(`"invalid" PAN ${pan} was accepted by pan-structure`);
  return pan;
}

/**
 * A 12-digit Aadhaar with a correct Verhoeff check digit, first digit 2-9 as
 * p-fin's rule requires. The check digit is SEARCHED for against the real
 * validator rather than computed from a second copy of the Verhoeff tables --
 * one copy of those tables in this repository is enough, and a second would be
 * a place for the corpus and the runtime to disagree.
 */
export function mintAadhaar(rng: () => number): string {
  const head = pick(rng, "23456789") + digits(rng, 10);
  for (let d = 0; d < 10; d += 1) {
    const candidate = head + d.toString();
    if (verhoeff(candidate)) return candidate;
  }
  // Verhoeff is a complete check digit scheme: exactly one d in 0..9 works for
  // any 11-digit head, so this is unreachable unless the validator changed.
  throw new Error(`no Verhoeff check digit completes ${head}`);
}

/** 12 digits, first 2-9, that the real `verhoeff` rejects. */
export function mintBadAadhaar(rng: () => number): string {
  const head = pick(rng, "23456789") + digits(rng, 10);
  for (let d = 0; d < 10; d += 1) {
    const candidate = head + d.toString();
    if (!verhoeff(candidate)) return candidate;
  }
  throw new Error(`every check digit completes ${head}, which Verhoeff cannot do`);
}

/** "#### #### ####" -- the spaced surface p-fin's rule also accepts. */
export function spaceAadhaar(a: string): string {
  return `${a.slice(0, 4)} ${a.slice(4, 8)} ${a.slice(8, 12)}`;
}

const UPPER_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const LOWER_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789";
const HEX = "0123456789abcdef";

/** `[A-Z]{4}0[A-Z0-9]{6}` -- an IFSC-shaped branch code. */
export function mintIfsc(rng: () => number): string {
  let s = "";
  for (let i = 0; i < 4; i += 1) s += pick(rng, LETTERS);
  s += "0";
  for (let i = 0; i < 6; i += 1) s += pick(rng, UPPER_ALNUM);
  return s;
}

/**
 * A SWIFT/BIC-shaped code: 8 uppercase letters. Confusable with an IFSC by
 * eye, and structurally distinct -- position 5 is a letter where an IFSC
 * requires "0", so p-fin's `ifsc-qualified-account` regex does not match it.
 */
export function mintBic(rng: () => number): string {
  let s = "";
  for (let i = 0; i < 8; i += 1) s += pick(rng, LETTERS);
  if (/^[A-Z]{4}0[A-Z0-9]{6}$/.test(s)) throw new Error(`BIC ${s} is IFSC-shaped`);
  return s;
}

export function mintUpiVpa(rng: () => number, handle: string): string {
  let s = "";
  for (let i = 0; i < 7; i += 1) s += pick(rng, LOWER_ALNUM);
  return `${s}@${handle}`;
}

export function mintAccountDigits(rng: () => number): string {
  return digits(rng, 14);
}

export function mintCustomerId(rng: () => number, prefix: "CIF" | "CRN" | "KYC", sep: string): string {
  return `${prefix}${sep}${digits(rng, 8)}`;
}

/** `INC` + 7 digits: a ticket id. Not a CIF/CRN/KYC, so p-fin's rule does not match it. */
export function mintTicketId(rng: () => number): string {
  return `INC${digits(rng, 7)}`;
}

export function mintAwsKey(rng: () => number): string {
  let s = "AKIA";
  for (let i = 0; i < 16; i += 1) s += pick(rng, UPPER_ALNUM);
  return s;
}

export function mintSkKey(rng: () => number): string {
  let s = "sk-";
  for (let i = 0; i < 28; i += 1) s += pick(rng, UPPER_ALNUM + LOWER_ALNUM);
  return s;
}

export function mintSlackToken(rng: () => number): string {
  const part = (n: number) => {
    let s = "";
    for (let i = 0; i < n; i += 1) s += pick(rng, UPPER_ALNUM + LOWER_ALNUM);
    return s;
  };
  return `xoxb-${part(4)}-${part(4)}-${part(16)}`;
}

/** 40 lowercase hex: a git commit id. High-looking, not a secret. */
export function mintGitSha(rng: () => number): string {
  let s = "";
  for (let i = 0; i < 40; i += 1) s += pick(rng, HEX);
  return s;
}

export function mintMicr(rng: () => number): string {
  return `MICR ${digits(rng, 9)}`;
}

export function mintEmail(rng: () => number, domain: string): string {
  let s = "";
  for (let i = 0; i < 6; i += 1) s += pick(rng, "abcdefghijklmnopqrstuvwxyz");
  return `${s}.${pick(rng, ["rao", "iyer", "menon", "shah", "bose"])}@${domain}`;
}

export function mintDbUrl(rng: () => number, scheme: string): string {
  return `${scheme}://ledger-${digits(rng, 3)}.harborline.example:5432/reconciliation`;
}

/**
 * A PEM block with a short invented body.
 *
 * The gold span for this family covers the WHOLE block, header to footer, not
 * just the header p-fin's `pem-private-key-header` regex matches. Reason: the
 * body is the secret and the header is a marker for it, so a span covering only
 * the header would label the least sensitive line of the injection.
 *
 * MEASURED on the emitted corpus, both consequences, stated here rather than
 * discovered in a results table. Running p-fin's own tier-0 rules over an item
 * carrying this family returns THREE findings inside the 174-character gold
 * span and none outside it:
 *
 * - `private-key-material` on the 31-character header. An `overlap` match
 *   against this gold and an `exact` miss, so the private-key family reads 0 on
 *   exact-match by construction.
 * - `api-credential` twice, on the two 44-character body lines, because
 *   `api-credential-entropy` fires on them at its stock threshold. These sit
 *   INSIDE a gold span of a different entityType, which is a case a scorer
 *   joining on `entityType` has to have an answer for. Shortening the body
 *   until the entropy rule stopped firing would make the number prettier and
 *   the corpus weaker, so the body stays and the fact is written down.
 */
export function mintPemBlock(rng: () => number): string {
  const line = (n: number) => {
    let s = "";
    for (let i = 0; i < n; i += 1) s += pick(rng, UPPER_ALNUM + LOWER_ALNUM + "+/");
    return s;
  };
  return `-----BEGIN RSA PRIVATE KEY-----\n${line(44)}\n${line(44)}\n${line(20)}==\n-----END RSA PRIVATE KEY-----`;
}
