import { getValidator } from "@sih/core";
import {
  mintAadhaar,
  mintAwsKey,
  mintGitSha,
  mintIfsc,
  mintPan,
  mintSlackToken,
  mintUpiVpa,
} from "./universe.js";
import {
  mintBase64Fragment,
  mintClientSecret,
  mintDashedAadhaar,
  mintGhpToken,
  mintHsmKeyLabel,
  mintUuid,
} from "./universe.candidate.js";

/**
 * Every surface the wave-3 catalogue injects, minted from the FORMAT SPEC that
 * defines it and never from `policies/compiled/p-fin.ir.json`.
 *
 * ## The defect this file exists to remove
 *
 * MEASURED on `corpora/generated/injection-p-fin-adjudicated-v1.jsonl`: 13 of
 * its 24 confusable families are 1:1 with an entry in the IR's own
 * `counterExamples` lists, and those 13 carry 59 of its 108 confusable spans.
 * A counterExample is text the compiled arm is SHOWN. So on more than half the
 * confusable spans the corpus was asking "is this a PAN?" of a model that had
 * already been handed "ABCDE1234F is not a PAN" in its prompt. The positives
 * are the same story from the other side: an IFSC, a UPI address, a labelled
 * account number, a CIF, a CRN, a KYC id, an AKIA key, an xoxb token, a
 * postgres URL and a mongodb+srv URL are the IR's `examples` array read out
 * surface for surface.
 *
 * So the rule here, and `corpus-surfaces.test.ts` enforces both halves:
 *
 * 1. Every mint is written from a published format rule -- the Income Tax
 *    Department's PAN and TAN layouts, UIDAI's twelve digits and Verhoeff check,
 *    RBI's eleven-character IFSC, NPCI's VPA grammar, RFC 7468's PEM
 *    encapsulation, RFC 7519's JWT, the vendors' own published key prefixes.
 *    The spec is quoted on each mint. No mint reads the IR.
 * 2. No minted value occurs in the IR or in the compiler self-test corpus, over
 *    hundreds of seeds -- the check `corpus-universe.test.ts` already made for
 *    wave 1, applied to every surface here.
 *
 * The part that is NOT claimed: a positive family's ENTITY CLASS necessarily
 * matches the IR, because the IR is the compilation of the policy the corpus
 * scores against. A PAN family tests §2.1 and §2.1 is `in-pan`. What is
 * removable is the surface inventory being an enumeration of the IR's lists,
 * and what is added below is the other direction: surfaces the format specs
 * produce and the IR names nowhere -- a lowercase-typed PAN, a UIDAI-masked
 * Aadhaar, a bare account number with no keyword in front of it, a JWT, an
 * Azure storage key, an EC and an encrypted PEM block, a CERTIFICATE block,
 * and a bare relationship number.
 *
 * ## Shape classes, and what they are for
 *
 * MEASURED on the same corpus by `measureOrthography`: the crude "return the
 * odd-looking string" oracle in `leakage.ts` finds 95 of its 108 gold spans, and
 * on 58 of them it returns NOTHING ELSE in the message -- so on 54% of the gold
 * spans a reader that understands nothing is a perfect detector, and its
 * precision over the whole corpus is 0.49. That score would not transfer to real
 * text.
 * `shapeClass` is the fix's index: every positive family declares one, every
 * class holds at least one confusable family of the same shape, and the
 * generator injects a same-class confusable alongside every positive. The
 * carriers cannot carry those distractors themselves -- they are the pool
 * certification cleaned, and a carrier holding a UUID or a second organisation
 * name is a carrier the sweeps quarantine -- so the distractor is an injected,
 * LABELLED span rather than uncertified carrier text.
 */

export const SHAPE_CLASSES = [
  "alnum-id-10",
  "digits-12",
  "digits-14",
  "digits-8",
  "masked-tail",
  "code-11",
  "handle-at-domain",
  "prefixed-id",
  "opaque-token",
  "driver-url",
  "pem-block",
  "kebab-label",
  "alias-word",
  "org-name",
  "person-name",
  "product-name",
] as const;
export type ShapeClass = (typeof SHAPE_CLASSES)[number];

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const UPPER_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
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

/**
 * Reserved by RFC 2606 §2, so nothing minted here can name a host that
 * resolves. Every invented hostname in this file sits under it. The wave-1 and
 * wave-2 mints put their hosts under `harborline.example`, the Firm's own
 * domain; wave 3 does not name the Firm at all (see `orgs.ts` for why), so the
 * hosts move to a domain that names no party.
 */
export const INTERNAL_DOMAIN = "intraledger.example";

// -- in-pan (§2.1) ----------------------------------------------------------

/**
 * A PAN typed in lowercase, which is how it arrives when somebody retypes one
 * out of a scan rather than pasting it.
 *
 * Spec: the value is a PAN by the Income Tax Department's layout; case is not
 * part of that layout, and §2.5 refuses relief for masking, truncating or
 * reversing a value, so a case change cannot be the thing that makes a PAN stop
 * being one. `in-pan-format`'s regex is `[A-Z]{5}[0-9]{4}[A-Z]` and does not
 * match it, so this is a §2.1 positive no tier-0 rule reaches -- which is the
 * point: the IR's `examples` are all uppercase and this surface is nowhere in
 * it.
 */
export function mintLowercasePan(rng: () => number): string {
  return mintPan(rng).toLowerCase();
}

const panStructure = getValidator("pan-structure");

/**
 * A TAN: the Income Tax Department's Tax Deduction and Collection Account
 * Number, four letters, five digits and a check letter.
 *
 * Spec: ITD's TAN layout (`AAAA99999A`), which is the same alphabet and the
 * same ten characters as a PAN with the letter/digit split moved by one. That
 * is what makes it the sharpest PAN confusable available, and the IR names no
 * such thing -- its `in-pan` counterExamples are a structurally invalid PAN, a
 * truncated PAN, a GSTIN and a form name.
 *
 * Asserted at mint time rather than described: `pan-structure` must REJECT it,
 * or it is not a confusable.
 */
export function mintTan(rng: () => number): string {
  const tan = run(rng, LETTERS, 4) + digits(rng, 5) + pick(rng, LETTERS);
  if (panStructure(tan)) throw new Error(`TAN ${tan} was accepted by pan-structure, so it is not a confusable`);
  return tan;
}

// -- in-aadhaar (§2.2) ------------------------------------------------------

/**
 * A UIDAI-masked Aadhaar: the first eight digits replaced by X, the last four
 * left, which is the form UIDAI's own masked e-Aadhaar prints and the form a
 * servicing console shows.
 *
 * Spec: UIDAI masked-Aadhaar presentation. The LABEL is §2.5, verbatim: the
 * §2 prohibition "is not relieved by masking part of the value". So this is an
 * `in-aadhaar` positive that `in-aadhaar-format` cannot see, because twelve
 * digits are no longer there to match.
 */
export function mintMaskedAadhaar(rng: () => number): string {
  return `XXXX XXXX ${mintAadhaar(rng).slice(8)}`;
}

/**
 * A masked mobile number, the same X-run-then-tail shape a masked Aadhaar has.
 *
 * p-fin regulates PAN, Aadhaar, bank account identifiers and internal customer
 * ids (§2), client and counterparty names (§3), and credentials (§4). A
 * telephone number is in none of those lists, masked or not, so §2.5 has
 * nothing to bite on. It is the shape neighbour for the masked Aadhaar.
 */
export function mintMaskedMobile(rng: () => number): string {
  return `XXXXXX${digits(rng, 4)}`;
}

/**
 * Twelve digits, leading 2-9, carrying a VALID Verhoeff check digit -- and used
 * as an NPCI retrieval reference number rather than as an Aadhaar.
 *
 * The check digit is valid on purpose and the reason is arithmetic. A twelve
 * digit number with a BROKEN Verhoeff digit is exactly the IR's own
 * `in-aadhaar` counterExample ("1234 5678 9012"), and one starting with 1 is
 * its other one ("100234567890"); minting either would put this family back in
 * the list decision 1 exists to empty. A valid one is a string the shipping
 * validator cannot separate from an Aadhaar at all, so tier 0 fires on it and
 * only the clause says what it is.
 *
 * Its label is `neg:` because §2's list is closed and enumerated -- PAN,
 * Aadhaar, bank account numbers (including IFSC-qualified strings and UPI
 * addresses), internal customer identifiers -- and a retrieval reference names
 * a TRANSACTION, is issued by the payment network rather than by the Firm, and
 * is not a customer reference number, a CIF or a KYC case id. That reading is
 * recorded on the family with the clause that could overturn it (§1.2's
 * "can be combined with other information to identify"), and the item carries a
 * label question for the adjudication round rather than a settled claim.
 */
export function mintRetrievalReference(rng: () => number): string {
  return mintAadhaar(rng);
}

/** Twelve digits, valid Verhoeff, used as a settlement batch sequence. Same reasoning as the RRN. */
export function mintBatchSequence12(rng: () => number): string {
  return mintAadhaar(rng);
}

// -- bank-account-identifier (§2.3) -----------------------------------------

/**
 * Eleven characters in the RBI's IFSC layout, used as an internal SCHEME code
 * rather than as a branch code.
 *
 * Same mint as an IFSC on purpose: `ifsc-qualified-account` fires on it, so
 * this is a measured false positive rather than a hypothetical one, and the
 * only thing separating it from the positive is the clause. §2.3's operative
 * prohibition is on "bank account numbers, including IFSC-qualified account
 * strings"; a product scheme code qualifies no account and names no customer.
 */
export const mintSchemeCode = mintIfsc;

/**
 * An npm package specifier, `name@dist-tag`.
 *
 * Spec: npm's own `<name>@<tag>` grammar. MEASURED against p-fin's
 * `upi-vpa-format` regex `\b[a-zA-Z0-9][a-zA-Z0-9._-]{2,}@[a-zA-Z][a-zA-Z0-9]{2,}\b`:
 * it matches, so tier 0 reports a `bank-account-identifier` on a lockfile line.
 * That is the over-firing this pair exists to measure, and it is NOT the IR's
 * counterExample for the same rule, which is an email address.
 */
export function mintPackageSpec(rng: () => number): string {
  const head = pick(rng, ["ledger", "settle", "recon", "vault", "audit", "cutover"]);
  const tail = pick(rng, ["parser", "client", "codec", "shim", "runner"]);
  return `${head}-${tail}@${pick(rng, ["latest", "next", "canary", "stable"])}`;
}

/** Fourteen bare digits: a core banking account number with no keyword in front of it. */
export function mintBareAccount14(rng: () => number): string {
  return digits(rng, 14);
}

/** Fourteen bare digits used as a settlement batch number. The shape neighbour for the account number. */
export function mintBatchNumber14(rng: () => number): string {
  return digits(rng, 14);
}

/** Eight bare digits: a POS terminal id. Names a terminal, not an account and not a person. */
export function mintTerminalId(rng: () => number): string {
  return digits(rng, 8);
}

// -- internal-customer-id (§2.4) --------------------------------------------

/**
 * Eight bare digits used as a relationship number.
 *
 * §2.4 forbids "customer reference numbers" outright and does not require them
 * to carry a prefix; `internal-customer-id-format` only matches CIF, CRN or KYC
 * followed by digits, so this is a §2.4 positive tier 0 cannot see. The IR's
 * `examples` are all prefixed, which is exactly why this surface is here.
 */
export function mintRelationshipNumber(rng: () => number): string {
  return digits(rng, 8);
}

/** `CIF`/`CRN`/`KYC` + separator + digits: the Firm's own prefixed customer references. */
export function mintPrefixedCustomerId(rng: () => number, prefix: "CIF" | "CRN" | "KYC", sep: string): string {
  return `${prefix}${sep}${digits(rng, 8)}`;
}

/**
 * `KYC ` plus a seven digit document number, used as the version of the Firm's
 * KYC STANDARD rather than as a case id.
 *
 * `internal-customer-id-format` fires on it -- the rule cannot tell a case from
 * a document -- and §2.4's list is of customer references, CIF numbers and KYC
 * case ids, none of which a policy document version is.
 */
export function mintKycStandardVersion(rng: () => number): string {
  // Hyphenated rather than spaced, and the reason is the orthographic
  // measurement rather than the rule: `internal-customer-id-format` accepts
  // either separator, but "KYC 1234567" is two tokens -- a three-letter
  // uppercase run and a seven-digit run -- and neither is odd enough for the
  // orthographic oracle to return. A distractor the "flag the odd string"
  // reader cannot see is not a distractor. MEASURED: with the space, the
  // `crn-id` gold spans it shadows stayed the only region the oracle returned
  // in their message.
  return `KYC-${digits(rng, 7)}`;
}

/** A supplier invoice number. Names an invoice, not a customer. */
export function mintInvoiceNumber(rng: () => number): string {
  return `INV-2026-${digits(rng, 4)}`;
}

// -- api-credential (§4.1) --------------------------------------------------

/**
 * A JWT: three base64url segments separated by dots, per RFC 7519 §3.1.
 *
 * A bearer token is an access token, which §4.1 names outright. The IR's
 * `api-credential` examples are an AWS key, an OpenAI-style key and a Slack
 * token, and its `api-credential-prefix` regex knows those three prefixes and
 * GitHub's; none of them is a JWT, so this is a §4.1 positive whose only
 * tier-0 route is the entropy rule, and only inside a code or kv segment.
 */
export function mintJwt(rng: () => number): string {
  return `${run(rng, BASE64URL, 36)}.${run(rng, BASE64URL, 52)}.${run(rng, BASE64URL, 43)}`;
}

/**
 * An Azure storage account key: 64 random bytes, base64, so 88 characters
 * ending in "==".
 *
 * Spec: Azure Storage's published key length. Another §4.1 credential with no
 * prefix for `api-credential-prefix` to match.
 */
export function mintAzureStorageKey(rng: () => number): string {
  return `${run(rng, BASE64, 86)}==`;
}

/** An OCI image digest, `sha256:` + 64 lowercase hex. Content addressing, not a secret. */
export function mintArtifactDigest(rng: () => number): string {
  return `sha256:${run(rng, HEX, 64)}`;
}

/**
 * An ssh PUBLIC key line: algorithm, base64 blob, comment.
 *
 * §4.3 reaches "private keys, certificate material, and HSM key labels". The
 * public half of a keypair is none of the three, and publishing it is what it
 * is for. It is not the IR's counterExample either -- that one is a
 * FINGERPRINT of a public key, which is a different string and a different
 * family, and it is removed from this corpus for exactly that reason.
 */
export function mintSshPublicKeyLine(rng: () => number): string {
  return `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5${run(rng, BASE64, 44)} ops-desk-${digits(rng, 2)}`;
}

// -- db-connection-string (§4.2) --------------------------------------------

/** `postgresql://user@host:port/db` on a reserved-TLD host. */
export function mintPostgresUrl(rng: () => number): string {
  return `postgresql://svc_recon@ledger-${digits(rng, 2)}.${INTERNAL_DOMAIN}:5432/settlements`;
}

/** `jdbc:postgresql://host:port/db`. */
export function mintJdbcUrl(rng: () => number): string {
  return `jdbc:postgresql://archive-${digits(rng, 2)}.${INTERNAL_DOMAIN}:5432/statements`;
}

/** `redis://host:port/db`. A cache endpoint is still a datastore connection string under §4.2. */
export function mintRedisUrl(rng: () => number): string {
  return `redis://cache-${digits(rng, 2)}.${INTERNAL_DOMAIN}:6379/3`;
}

/**
 * An sftp file-transfer endpoint on the same internal domain.
 *
 * §4.2 governs "database connection strings", and its stated reason is that the
 * host and schema names disclose internal topology. An sftp endpoint names no
 * driver and no schema, so the operative sentence does not reach it -- and the
 * reason plainly does, which is why the family carries the contesting clause on
 * its face and the item carries a label question. `db-connection-string-format`
 * knows nine driver schemes and sftp is not one of them.
 */
export function mintSftpEndpoint(rng: () => number): string {
  return `sftp://transfer-${digits(rng, 2)}.${INTERNAL_DOMAIN}/outbound/settlements`;
}

/**
 * A datasource ALIAS: the name a config uses for a connection whose host and
 * credentials live somewhere else.
 *
 * It names no host, no schema and no driver, so §4.2 -- operative sentence and
 * stated reason alike -- has nothing to reach. Injected next to the word
 * "datasource", which is a `db-connection-string-format` contextBoost term, so
 * the boost keyword sits beside a negative as readily as beside a positive.
 */
export function mintDatasourceAlias(rng: () => number): string {
  return `${pick(rng, ["settlements", "statements", "recon", "cutover"])}_${pick(rng, ["ro", "rw", "batch"])}`;
}

// -- private-key-material (§4.3) --------------------------------------------

function pemBlock(rng: () => number, label: string): string {
  return [
    `-----BEGIN ${label}-----`,
    run(rng, BASE64, 44),
    run(rng, BASE64, 44),
    `${run(rng, BASE64, 18)}==`,
    `-----END ${label}-----`,
  ].join("\n");
}

/** RFC 7468 encapsulation, "RSA PRIVATE KEY". */
export function mintRsaPem(rng: () => number): string {
  return pemBlock(rng, "RSA PRIVATE KEY");
}

/** RFC 7468 encapsulation, OpenSSH's own private-key label. */
export function mintOpensshPem(rng: () => number): string {
  return pemBlock(rng, "OPENSSH PRIVATE KEY");
}

/** RFC 7468 encapsulation, "EC PRIVATE KEY". The IR's examples name RSA and OPENSSH and not this. */
export function mintEcPem(rng: () => number): string {
  return pemBlock(rng, "EC PRIVATE KEY");
}

/** RFC 7468 encapsulation, "ENCRYPTED PRIVATE KEY". Encryption is not relief: §4.3 is unqualified. */
export function mintEncryptedPem(rng: () => number): string {
  return pemBlock(rng, "ENCRYPTED PRIVATE KEY");
}

/**
 * An X.509 CERTIFICATE block, RFC 7468 §5.
 *
 * This is a POSITIVE, and the reasoning is worth stating precisely rather than
 * dramatically, because the IR and the policy agree in one place and diverge in
 * another.
 *
 * They AGREE at the entityType: p-fin §4.3 reads "Private keys, certificate
 * material, and HSM key labels must never be included in a prompt", and the
 * IR's `private-key-material` nlDefinition reads "Private key or certificate
 * material, including a PEM private key block and the label naming a key held
 * in a hardware security module". Both cover a certificate. So the label here
 * is not a reading against the IR; it is the IR's own definition.
 *
 * They DIVERGE in two artifacts under it. `pem-private-key-header` matches only
 * the literal "PRIVATE KEY" headers, so no tier-0 rule reaches a CERTIFICATE
 * block -- which makes this a §4.3 positive the ladder has to climb for, the
 * same shape of gap as `client-name` and the HSM label. And the entityType's
 * counterExample list names a CERTIFICATE REQUEST, which is the hardest of the
 * three to reconcile with "certificate material": that is why the wave-2 CSR
 * family is DELETED rather than relabelled. A CSR carries a public key and sits
 * exactly on that fault line, so it is a question for the adjudication round
 * and not a span to score anybody on.
 */
export function mintCertificatePem(rng: () => number): string {
  return pemBlock(rng, "CERTIFICATE");
}

/**
 * A `DH PARAMETERS` block: the same five dashes, the same base64 body, and no
 * key at all -- Diffie-Hellman parameters are public constants.
 *
 * Not a private key, not certificate material, not an HSM key label, so §4.3
 * reaches none of it; and the IR names no such block, which is what makes it a
 * legal shape neighbour for the PEM positives.
 */
export function mintDhParametersBlock(rng: () => number): string {
  return pemBlock(rng, "DH PARAMETERS");
}

/** A kubernetes deployment name, the shape neighbour for an HSM key label. Names a workload, not a key. */
export function mintDeploymentName(rng: () => number): string {
  return `svc-prod-${pick(rng, ["settlement", "recon", "statements", "cutover"])}-${digits(rng, 2)}`;
}

// -- re-exported wave-1/wave-2 mints ----------------------------------------

/**
 * Re-exported rather than re-implemented. Each was already written from its
 * published format rule -- PAN and Aadhaar against the shipping validators,
 * IFSC and VPA against RBI's and NPCI's layouts, AKIA/ghp_/xoxb against the
 * vendors' own published prefixes -- and a second copy of a format rule in this
 * repository is a place for two files to disagree.
 *
 * `mintSkKey` is deliberately NOT among them: `sk-` is the one prefix that
 * appears in the IR twice, as an `api-credential` example and as its own
 * counterExample ("sk-..."), so a family minting it is on both sides of decision
 * 1 at once. The GitHub, AWS and Slack prefixes stay because they are published
 * credential formats that the IR happens to list, which is the opposite
 * direction of copying.
 */
export {
  mintAadhaar,
  mintAwsKey,
  mintBase64Fragment,
  mintClientSecret,
  mintDashedAadhaar,
  mintGhpToken,
  mintGitSha,
  mintHsmKeyLabel,
  mintIfsc,
  mintPan,
  mintSlackToken,
  mintUpiVpa,
  mintUuid,
};

/** "#### #### ####", the spaced surface UIDAI prints and `in-aadhaar-format` accepts. */
export function mintSpacedAadhaar(rng: () => number): string {
  const a = mintAadhaar(rng);
  return `${a.slice(0, 4)} ${a.slice(4, 8)} ${a.slice(8, 12)}`;
}

/** A VPA at an invented handle. `okaxis` and the other live PSP handles are never minted here. */
export function mintFictionalVpa(rng: () => number): string {
  return mintUpiVpa(rng, "okbluecrest");
}

/** A 32-character mixed-case client secret with no prefix: the entropy path's only positive. */
export const mintOpaqueClientSecret = mintClientSecret;
