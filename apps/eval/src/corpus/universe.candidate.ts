import { getValidator } from "@sih/core";
import { mintAadhaar } from "./universe.js";

/**
 * Universe additions for the second carrier/family wave -- the pools and mints
 * the near-miss families in `families.candidate.ts` draw from.
 *
 * ## Why this is a separate file and not more of `universe.ts`
 *
 * `CLIENT_ORGS` and `NON_CLIENT_ORGS` are indexed by `Math.floor(rng() * n)`,
 * so appending one name to either array changes which organisation every
 * existing item draws and the committed corpus stops reproducing. The wave-2
 * pools therefore live here, disjoint from the wave-1 ones, and the committed
 * artifact keeps reproducing byte for byte while this file is unwired. See the
 * header of `carriers.candidate.ts` for what wiring costs.
 *
 * ## The confound this file exists to remove
 *
 * `CLIENT_ORGS` are industrial ("Kestrel Ironworks", "Ridgeline Dairy") and
 * `NON_CLIENT_ORGS` are services ("Cobblestone Print Works", "Ferndale
 * Conference Centre"), and `corpus-universe.test.ts` asserts the two pools do
 * not overlap. So in the wave-1 corpus the client/non-client question can be
 * answered from the NAME: anything ending in "Print Works" or "Conference
 * Centre" is a supplier, and no role clause needs reading. An arm that scores
 * well on that pair has not been shown to classify by role.
 *
 * `DUAL_ROLE_ORGS` is the fix. One pool, deliberately flavour-neutral, drawn by
 * a client family and by a vendor family alike, so the only signal separating
 * `client-name` from `neg:dual-role-org-vendor` is the clause around the name.
 * That is the pair the run-03 result says this corpus has to contain: a 2B model
 * put a finding on exactly the right sixteen characters of an organisation name
 * and called it `in-pan`, which is a classification failure that no amount of
 * additional span-finding measures.
 *
 * ## Fiction, and how far it was checked
 *
 * Every organisation and product name below was web-searched as a quoted string
 * on 2026-09-02 before being used, and none returned an exact match; the
 * closest hits were different companies with a shared word ("Ashmore Group",
 * "Fenway Partners", "Watermill Group"). "Stonecrop Advisory" was drafted and
 * DROPPED at that step, because the search returned Stonecrop Wealth Advisors
 * and Stonecrop Capital. That is a name check, not a trademark clearance, and
 * it is worth exactly what a search engine's first page is worth.
 *
 * Person names are a weaker claim and are marked as such: every surname a
 * person could plausibly have belongs to real people, so `PERSON_NAMES` is
 * fictional in the sense that no entry is intended to denote anyone, not in the
 * sense that nobody is called this. They are used only in a colleague role --
 * never as a customer -- which is also what keeps their `neg:` label
 * defensible; see `families.candidate.ts`.
 */

// -- organisations ----------------------------------------------------------

/**
 * Organisations the corpus places in BOTH a client role and a supplier role,
 * across different items. Deliberately neutral: nothing in "Marrowfield Group"
 * says whether the Firm sells to it or buys from it.
 *
 * Under p-fin §3.1 an occurrence in a client / prospective-client /
 * counterparty clause is a `client-name`; the same string in a supplier clause
 * is nothing p-fin reaches. Same span, same orthography, opposite answer.
 */
export const DUAL_ROLE_ORGS = [
  "Marrowfield Group",
  "Ashcombe Holdings",
  "Tarnwick Industries",
  "Lensfield Trading",
  "Bexmoor Associates",
  "Halvergate Company",
] as const;

/** A rival firm. Named, capitalised, and not a client -- p-fin §3 does not reach it. */
export const COMPETITOR_ORGS = ["Fenwold Advisory", "Brackwater Partners"] as const;

/**
 * A listed company the user read about in the trade press. The most
 * over-detected organisation shape there is: it is famous-sounding, it is in a
 * financial sentence, and the Firm has no relationship with it at all.
 */
export const LISTED_COMPANY_ORGS = ["Cindermill Motors", "Quarrywood Energy"] as const;

/** Software the Firm uses. An ORG tagger will tag these; p-fin demands nothing of them. */
export const PRODUCT_NAMES = ["Quillrun", "Wickerpost"] as const;

/**
 * Colleagues. Not customers: p-fin §1.2 defines customer data as information
 * identifying a person or entity HOLDING AN ACCOUNT with the Firm, so a
 * colleague's name is outside it, and the compiled IR has no personal-name
 * entityType to read an obligation from either. Both readings agree, which is
 * why this family is labelled `neg:` rather than left to an adjudication round.
 */
export const PERSON_NAMES = ["Meera Talwatkar", "Arjun Beheria", "Sunita Marlekar"] as const;

// -- minting ----------------------------------------------------------------

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const HOLDER_TYPES = "ABCFGHJLPT";
const UPPER_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const LOWER_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789";
const BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const HEX = "0123456789abcdef";

function pick<T>(rng: () => number, xs: ArrayLike<T>): T {
  return xs[Math.floor(rng() * xs.length)]!;
}

function digits(rng: () => number, n: number): string {
  let s = "";
  for (let i = 0; i < n; i += 1) s += Math.floor(rng() * 10).toString();
  return s;
}

function run(rng: () => number, alphabet: string, n: number): string {
  let s = "";
  for (let i = 0; i < n; i += 1) s += pick(rng, alphabet);
  return s;
}

const panStructure = getValidator("pan-structure");

/** The compiled rule, copied so a mint can assert what tier 0 will and will not do. */
const PAN_RULE = /\b[A-Z]{5}[0-9]{4}[A-Z]\b/;

/**
 * A GSTIN: two state digits, a structurally valid PAN, an entity digit, "Z" and
 * a check character.
 *
 * The embedded PAN is minted through the SAME holder-type set the shipping
 * `pan-structure` validator accepts, so the ten characters in the middle of
 * this string are a real PAN by every test the runtime has. What stops tier 0
 * is arithmetic, not semantics: `in-pan-format` is anchored with `\b`, and
 * inside "27AABCU9603R1ZM" neither end of the PAN sits on a word boundary. So
 * this is the sharpest confusable in the file -- p-fin's own counterExample
 * list names a GSTIN, tier 0 is silent on it, and a model reading the middle
 * ten characters has every reason to call it a PAN.
 *
 * Both halves are asserted at mint time rather than described: the embedded PAN
 * must pass the validator, and the whole string must not match the rule.
 */
export function mintGstin(rng: () => number): string {
  const pan =
    run(rng, LETTERS, 3) + pick(rng, HOLDER_TYPES) + pick(rng, LETTERS) + digits(rng, 4) + pick(rng, LETTERS);
  if (!panStructure(pan)) throw new Error(`GSTIN core ${pan} is not a structurally valid PAN`);
  const gstin = `${digits(rng, 2)}${pan}${digits(rng, 1)}Z${pick(rng, UPPER_ALNUM)}`;
  if (PAN_RULE.test(gstin)) throw new Error(`GSTIN ${gstin} matches the in-pan rule, so it is not a confusable`);
  return gstin;
}

/** "####-####-####" -- the dashed surface `in-aadhaar-format` also accepts. */
export function dashAadhaar(a: string): string {
  return `${a.slice(0, 4)}-${a.slice(4, 8)}-${a.slice(8, 12)}`;
}

export function mintDashedAadhaar(rng: () => number): string {
  return dashAadhaar(mintAadhaar(rng));
}

/**
 * Fourteen bare digits: a core banking account number written the way a
 * colleague pastes one, with no "A/C" and no "account" in front of it.
 *
 * Deliberately invisible to tier 0, and the reason is worth stating because it
 * looks like an oversight. `labelled-account-number` needs the word "account"
 * (or A/C, A/c, Acct) within twelve non-digit characters, and the glue for this
 * family carries none of those; `in-aadhaar-format` cannot match a 14-digit run
 * because neither `\b` lands where the pattern needs it. p-fin §2.3 forbids
 * bank account numbers outright, so the span is a genuine positive that only a
 * tier above 0 can reach -- which is the point. A corpus whose every positive
 * is regex-findable measures the regex.
 */
export function mintBareAccountDigits(rng: () => number): string {
  return digits(rng, 14);
}

/** `EMP` + six digits. p-fin's counterExample list names an employee id; no rule reaches it. */
export function mintEmployeeId(rng: () => number): string {
  return `EMP${digits(rng, 6)}`;
}

/** `ghp_` + 36 alphanumerics: the GitHub shape `api-credential-prefix` names. */
export function mintGhpToken(rng: () => number): string {
  return `ghp_${run(rng, UPPER_ALNUM + LOWER_ALNUM, 36)}`;
}

/**
 * A 32-character mixed-case client secret with no recognisable prefix.
 *
 * This family exists to exercise the ENTROPY path rather than the prefix path:
 * every other `api-credential` family in the corpus is caught by
 * `api-credential-prefix`, so without this one the entropy rule has no positive
 * of its own and its behaviour is only ever observed as a false positive on
 * private-key bodies. 32 characters over a 62-symbol alphabet clears the stock
 * rule's 24-character floor with room to spare.
 */
export function mintClientSecret(rng: () => number): string {
  return run(rng, UPPER_ALNUM + LOWER_ALNUM, 32);
}

/**
 * AWS's own documented example access key, verbatim and constant.
 *
 * This is spec 6.2's hard-negative triage case in one string: "sensitive-looking
 * but adjudicated benign (public figures, example.com, tutorial API keys)". It
 * matches `api-credential-prefix` exactly, so tier 0 fires on it and, under the
 * injection invariant, that finding is a true false positive.
 *
 * Its LABEL is the one call in this file a reader should treat as contested.
 * p-fin §4.1 forbids API keys without qualifying the sentence, and a strict
 * reading blocks any key-shaped string; spec 6.2 says a tutorial key is a hard
 * negative requiring human confirmation. Neither this file nor the generator
 * can settle that, so the family is named for what it is and flagged for the
 * adjudication round that the whole `pred:` side of this corpus is already
 * waiting on. It is constant rather than minted precisely so a human can find
 * every instance with one grep.
 *
 * Two things about the literal, both measured rather than assumed:
 *
 * - It is NOT AWS's own "AKIAIOSFODNN7EXAMPLE". That string is already in
 *   `policies/compiled/p-fin.selftest.json` (example 180, "AKIAIOSFODNN7EXAMPLE
 *   is the key id used throughout the AWS documentation"), so injecting it
 *   would put a corpus value inside the compiler self-test corpus that
 *   `contamination.ts` exists to keep this corpus disjoint from -- and it would
 *   trip `corpus-generate.test.ts`'s leak check on the first run.
 * - That self-test example is also the corroboration for the label. The
 *   compiler's own self-test carries the documented key as a hard NEGATIVE, so
 *   "a documentation key is not a credential" is a reading this repository
 *   already committed to somewhere else, rather than this file's opinion.
 *
 * The substitute embeds "EXAMPLEDOCSKEY", which cannot collide with a live AWS
 * key, and still matches `api-credential-prefix` exactly.
 */
export const DOC_EXAMPLE_AWS_KEY = "AKIAEXAMPLEDOCSKEY00";

/**
 * A redaction placeholder that still matches `api-credential-prefix`
 * (`sk-[A-Za-z0-9-]{20,}`), which is what makes it a confusable rather than a
 * curiosity: the rule cannot tell a key from a sentence about a key written in
 * key-shaped characters. p-fin's own counterExample list carries "sk-...", the
 * short version of the same idea. Constant, because a redaction marker that
 * varied per item would be a strange thing to find in a runbook.
 */
export const REDACTED_CREDENTIAL_PLACEHOLDER = "sk-live-REDACTED-BY-THE-SECURITY-DESK";

/** A v4-shaped UUID: a correlation id. High-looking, not a secret, and not high-entropy enough to be one. */
export function mintUuid(rng: () => number): string {
  return `${run(rng, HEX, 8)}-${run(rng, HEX, 4)}-4${run(rng, HEX, 3)}-${pick(rng, "89ab")}${run(rng, HEX, 3)}-${run(rng, HEX, 12)}`;
}

/**
 * A fragment of a base64-encoded image, the kind that arrives when somebody
 * pastes a data URI into a chat window. Injected inside a code fence on
 * purpose: `runTier0` scopes entropy rules to code and kv segments, so in prose
 * this would be a confusable nothing ever confuses. In a fence it trips
 * `api-credential-entropy` and becomes a measured false positive rather than a
 * hypothetical one.
 */
export function mintBase64Fragment(rng: () => number): string {
  return run(rng, BASE64, 64);
}

/** An internal https URL. p-fin's counterExample for `db-connection-string` is exactly this shape. */
export function mintInternalUrl(rng: () => number): string {
  const page = pick(rng, ["settlement-cutover", "month-end-close", "feed-replay", "cache-warm"]);
  return `https://intranet.harborline.example/runbooks/${page}-${digits(rng, 2)}`;
}

export function mintMongoUrl(rng: () => number): string {
  return `mongodb+srv://cluster-${digits(rng, 2)}.harborline.example/servicing`;
}

export function mintJdbcUrl(rng: () => number): string {
  return `jdbc:postgresql://ledger-arch-${digits(rng, 2)}.harborline.example:5432/settlements`;
}

/** A PEM private key in the OPENSSH flavour -- the second header `pem-private-key-header` names. */
export function mintOpensshPem(rng: () => number): string {
  return [
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    run(rng, BASE64, 44),
    run(rng, BASE64, 44),
    `${run(rng, BASE64, 18)}==`,
    "-----END OPENSSH PRIVATE KEY-----",
  ].join("\n");
}

/**
 * A certificate SIGNING REQUEST block. Same five dashes, same base64 body, same
 * genre -- and p-fin's counterExample list for `private-key-material` names it
 * explicitly, because a CSR carries a PUBLIC key. `pem-private-key-header`
 * requires the literal "PRIVATE KEY", so tier 0 is correctly silent on the
 * header; the body lines still trip the entropy rule inside a fence, exactly as
 * a real private key's do, which is what makes this a fair test of whether an
 * arm reads the header or the shape.
 */
export function mintCsrPem(rng: () => number): string {
  return [
    "-----BEGIN CERTIFICATE REQUEST-----",
    run(rng, BASE64, 44),
    run(rng, BASE64, 44),
    `${run(rng, BASE64, 18)}==`,
    "-----END CERTIFICATE REQUEST-----",
  ].join("\n");
}

/** An ssh host key fingerprint: public material, and the counterExample p-fin names. */
export function mintPubkeyFingerprint(rng: () => number): string {
  return `SHA256:${run(rng, BASE64, 43)}`;
}

/**
 * The label naming a key inside a hardware security module. p-fin §4.3 and the
 * `private-key-material` nlDefinition both cover it, and no tier-0 rule looks
 * for it, so this is a positive an arm can only reach by understanding the
 * entity rather than by matching a header.
 */
export function mintHsmKeyLabel(rng: () => number): string {
  const purpose = pick(rng, ["signing", "settlement", "archive", "issuance"]);
  return `hsm-prod-${purpose}-${digits(rng, 2)}`;
}
