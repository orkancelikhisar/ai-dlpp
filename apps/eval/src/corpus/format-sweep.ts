import type { Sweep } from "./certify.js";

/**
 * A certification sweep that reads no IR and calls no detector this project
 * ships.
 *
 * ## The finding this answers
 *
 * `certify.ts` refuses to stand `@sih/tier1` in for spec 6.2's stage 2, on the
 * stated ground that tier 1 IS an arm under test. The identical objection
 * applies to stage 1, which runs `runTier0` from `@sih/core` over a widened
 * copy of the very IR the corpus is scored under -- and `certify.ts` never
 * raised it. So the carriers that survive certification are exactly the
 * carriers the tier-0 arm is silent on, chosen by that arm, and the negatives
 * are easy for it by construction.
 *
 * Spec 6.2 prescribes stage 1 as "tier-0 sweep, thresholds at max recall", so
 * the fix is NOT to replace it -- that would be a different stage than the one
 * the spec names. Stage 1 still runs and still quarantines. What is added is
 * this second, independent sweep running beside it as a SUPPLEMENTARY sweep,
 * which can also quarantine, and a count in the manifest of how many carriers
 * the tier-0 arm quarantined that this sweep did NOT. That count is the size of
 * the circularity, measured rather than argued about.
 *
 * ## What "independent" means here, exactly
 *
 * - It does not import `runTier0`, `segmentText`, `getValidator` or any IR. The
 *   only import is the `Sweep` type.
 * - Its patterns are written from published format rules, not from p-fin's
 *   compiled regexes: PAN and TAN from the Income Tax Department's layouts,
 *   twelve digits from UIDAI's, eleven characters from RBI's IFSC layout, an
 *   at-sign handle from NPCI's VPA grammar, PEM dashes from RFC 7468, the
 *   vendor credential prefixes from the vendors.
 * - It carries NO validator. A structurally invalid PAN and a broken Verhoeff
 *   digit both quarantine here, because a certification sweep wants recall and
 *   a carrier that merely looks like it carries an identifier is a carrier not
 *   worth the argument.
 * - It scans the WHOLE text, including prose. `runTier0` scopes its entropy
 *   rules to code and kv segments; this one does not, which is a difference in
 *   coverage and not only in code.
 * - Its Shannon entropy is a second implementation, deliberately. Calling
 *   core's `shannonEntropy` would import the arm's own arithmetic into the
 *   thing that is supposed to be independent of it. The duplication is the
 *   point, and it is the only duplication in this file.
 *
 * What it does NOT claim: independence of the AUTHOR. Both this sweep and the
 * carriers were written in this repository, and a carrier written to pass one
 * of them was written by someone who could read the other. Spec 6.2's stage 2
 * is what would fix that, and it is still unrun.
 */

/** A second implementation, on purpose. See the header. */
function entropyBits(s: string): number {
  const counts = new Map<string, number>();
  for (const ch of s) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  let h = 0;
  for (const n of counts.values()) {
    const p = n / s.length;
    h -= p * Math.log2(p);
  }
  return h;
}

/**
 * Deliberately low, and low for a stated reason rather than a tuned one: a
 * certification sweep wants to over-quarantine, and 3.0 bits over 16 characters
 * is roughly "a token with more variety than an English word of that length".
 * The stock `api-credential-entropy` rule sits at 4.0 over 24 characters, so
 * this is wider on both axes, which is what a max-recall pass is for.
 */
const ENTROPY_BITS = 3.0;
const ENTROPY_MIN_LENGTH = 16;

interface Pattern {
  readonly label: string;
  readonly why: string;
  readonly re: RegExp;
}

/**
 * Patterns are recreated per call rather than held in a module constant.
 *
 * MEASURED in this repository on a `/g` literal shared across calls: two
 * successive `[...text.matchAll(re)]` are fine because `matchAll` clones the
 * regex, but `re.exec` or `re.test` advance `lastIndex` and silently skip the
 * head of every second input. Building them per call removes the trap rather
 * than documenting it.
 */
function patterns(): readonly Pattern[] {
  return [
    { label: "pan-shaped", why: "Income Tax Department PAN layout: five letters, four digits, a letter", re: /\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b/g },
    { label: "tan-shaped", why: "Income Tax Department TAN layout: four letters, five digits, a letter", re: /\b[A-Za-z]{4}[0-9]{5}[A-Za-z]\b/g },
    { label: "twelve-digit-run", why: "UIDAI Aadhaar length, in any of the printed groupings", re: /\b[0-9]{4}[ -]?[0-9]{4}[ -]?[0-9]{4}\b/g },
    { label: "long-digit-run", why: "nine or more consecutive digits: a core banking account number or longer", re: /\b[0-9]{9,}\b/g },
    { label: "ifsc-shaped", why: "RBI IFSC layout: four letters, a zero, six alphanumerics", re: /\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b/g },
    { label: "at-handle", why: "NPCI VPA grammar, which an email address also satisfies", re: /\b[A-Za-z0-9][A-Za-z0-9._-]{2,}@[A-Za-z][A-Za-z0-9]{2,}\b/g },
    { label: "prefixed-customer-id", why: "the Firm's own CIF / CRN / KYC reference layouts", re: /\b(?:CIF|CRN|KYC)[ /:#-]?[0-9]{5,12}\b/gi },
    { label: "credential-prefix", why: "published vendor key prefixes: AWS, GitHub, Slack, OpenAI-style", re: /\b(?:AKIA|ASIA)[0-9A-Za-z]{10,}|\bghp_[A-Za-z0-9]{20,}|\bxox[abprs]-[A-Za-z0-9-]{8,}|\bsk-[A-Za-z0-9-]{10,}/g },
    { label: "pem-encapsulation", why: "RFC 7468 encapsulation boundary, whatever the label between the dashes", re: /-----BEGIN [A-Z0-9 ]+-----/g },
    { label: "driver-url", why: "a URL whose scheme names a datastore driver or a file transfer", re: /\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|mssql|redis|amqp|sftp|ftps?|jdbc:[a-z]+):\/\/[^\s"']+/g },
    { label: "masked-tail", why: "a run of masking characters followed by a numeric tail", re: /\bX{4,}[ -]?(?:X{4}[ -]?)*[0-9]{4}\b/g },
  ];
}

const TOKEN = /[A-Za-z0-9+/=_-]{16,}/g;

export const FORMAT_SPEC_SWEEP_ID = "format-spec-sweep";

/**
 * Everything this sweep considers identifier-shaped. Any hit quarantines the
 * carrier, exactly as spec 6.2's "any hit -> quarantine" says of stage 1.
 */
export const formatSpecSweep: Sweep = (text) => {
  const hits: { sweep: string; start: number; end: number; text: string; label: string; note: string }[] = [];
  for (const p of patterns()) {
    for (const m of text.matchAll(p.re)) {
      hits.push({ sweep: FORMAT_SPEC_SWEEP_ID, start: m.index, end: m.index + m[0].length, text: m[0], label: p.label, note: p.why });
    }
  }
  for (const m of text.matchAll(TOKEN)) {
    if (m[0].length < ENTROPY_MIN_LENGTH) continue;
    const bits = entropyBits(m[0]);
    if (bits < ENTROPY_BITS) continue;
    hits.push({
      sweep: FORMAT_SPEC_SWEEP_ID,
      start: m.index,
      end: m.index + m[0].length,
      text: m[0],
      label: "high-entropy-token",
      note: `${bits.toFixed(2)} bits over ${m[0].length} characters, against a floor of ${ENTROPY_BITS} over ${ENTROPY_MIN_LENGTH}; scanned in prose as well as in code, which runTier0 does not do`,
    });
  }
  return hits.sort((a, b) => a.start - b.start || a.end - b.end || a.label.localeCompare(b.label));
};

/** Exported so a test can show the entropy floor moves something rather than trusting the constant. */
export const FORMAT_SWEEP_ENTROPY = { bits: ENTROPY_BITS, minLength: ENTROPY_MIN_LENGTH } as const;
