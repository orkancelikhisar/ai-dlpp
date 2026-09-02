import type { Companion, ConstructedRole, Family } from "./families.js";
import { NEG_PREFIX } from "./labels.js";
import { ORG_POOL, PERSON_NAMES, PRODUCT_NAMES, nextOrg, pickFrom, pickOrg, roleClass, type OrgRole } from "./orgs.js";
import {
  type ShapeClass,
  mintArtifactDigest,
  mintAwsKey,
  mintAzureStorageKey,
  mintAadhaar,
  mintBareAccount14,
  mintBase64Fragment,
  mintBatchNumber14,
  mintBatchSequence12,
  mintCertificatePem,
  mintDashedAadhaar,
  mintDatasourceAlias,
  mintDeploymentName,
  mintDhParametersBlock,
  mintEcPem,
  mintEncryptedPem,
  mintFictionalVpa,
  mintGhpToken,
  mintGitSha,
  mintHsmKeyLabel,
  mintIfsc,
  mintInvoiceNumber,
  mintJdbcUrl,
  mintJwt,
  mintKycStandardVersion,
  mintLowercasePan,
  mintMaskedAadhaar,
  mintMaskedMobile,
  mintOpaqueClientSecret,
  mintOpensshPem,
  mintPackageSpec,
  mintPan,
  mintPostgresUrl,
  mintPrefixedCustomerId,
  mintRedisUrl,
  mintRelationshipNumber,
  mintRetrievalReference,
  mintRsaPem,
  mintSchemeCode,
  mintSftpEndpoint,
  mintSlackToken,
  mintSpacedAadhaar,
  mintSshPublicKeyLine,
  mintTan,
  mintTerminalId,
  mintUuid,
} from "./surfaces.js";

/**
 * The wave-3 injection catalogue: the inputs rebuilt so that none of them
 * answers the question the corpus is asking.
 *
 * ## What each field on a wave-3 family is for
 *
 * `surfaceName` -- what a reader would call this surface. It is the key to the
 * decision-1 check: `IR_COUNTEREXAMPLE_SURFACES` in `leakage.ts` names what
 * each of the IR's 23 counterExamples denotes, and no confusable family here
 * may share a surface name with one. That check exists because the mechanical
 * version cannot work: a family minting `MICR <9 digits>` is the IR's
 * "MICR 400240012" with different digits, and no token comparison sees it.
 * MEASURED before: 13 of 24 confusable families in
 * `injection-p-fin-adjudicated-v1` were 1:1 with an entry in that list, and
 * they carried 59 of its 108 confusable spans.
 *
 * `boost` -- whether the family's GLUE puts an IR `contextBoost` term within
 * `BOOST_WINDOW` of the span. Declared here and MEASURED in `corpus-v2.test.ts`
 * against the real term list, so a declaration that stops matching the glue is
 * a failing test rather than a stale comment. contextBoost is a compiled-arm
 * feature, so a corpus where the boost keyword predicts the label hands that
 * arm the answer; MEASURED before, by `measureBoost` at a 70-character window
 * that excludes the span's own text: a boost term sat near 40.7% of gold spans
 * and near 0.9% of confusable spans -- 44 of 108 against 1 of 108.
 *
 * `shapeClass` -- the orthographic class the value belongs to. Every positive
 * family's class must also hold a confusable family, and the generator injects
 * one of those alongside every positive, so the gold span stops being the only
 * odd-looking thing in its message. MEASURED before: on 58 of 108 gold spans the
 * orthographic oracle returned that span and nothing else.
 *
 * `labelBasis` -- the clause of `policies/p-fin.md` that settles the label, the
 * reading, and where a reader disagrees, the clause under which it flips.
 * Wave 2 shipped three confusables the policy contradicts or makes arguable and
 * said nothing on their face; a family that cannot cite a clause is removed
 * rather than shipped with a hopeful comment, and the removals are in
 * `REMOVED_FAMILIES` with their reasons.
 *
 * `orgRole` -- for organisation families only, the role the clause writes. All
 * of them draw from one pool (`orgs.ts`), so the name carries no signal and the
 * clause carries all of it.
 */

export interface LabelBasis {
  /** Clauses of policies/p-fin.md, e.g. ["§2.3"]. */
  readonly clauses: readonly string[];
  /** The reading those clauses support, in one sentence. */
  readonly reading: string;
  /**
   * The clause under which a careful reader reaches the opposite answer, when
   * one exists. A family carrying this still ships, and every span it produces
   * is excluded from the corpus's false-positive claim until the label
   * adjudication round answers it -- see `questions.ts`.
   */
  readonly contestedBy?: string;
}

export interface V2Family extends Family {
  readonly surfaceName: string;
  readonly shapeClass: ShapeClass;
  readonly boost: "carries" | "absent";
  readonly labelBasis: LabelBasis;
  /** Confusables only: the entityType this surface is a near miss for. */
  readonly nearMissFor?: string;
  /** Organisation families only. */
  readonly orgRole?: OrgRole;
  /** Positives that write their own shape neighbour and must not be given another. */
  readonly selfDistracting?: boolean;
}

const g = (prefix: string, suffix: string) => () => ({ prefix, suffix });

/**
 * The block the cross-segment pair writes between the two organisation names
 * and the sentence that assigns their roles.
 *
 * A fenced block, and the choice is mechanical: `segmentText` splits on fences
 * and on kv/prose line runs, never on sentence boundaries, so two sentences
 * separated by a blank line stay inside ONE prose segment and a segment-scoped
 * judge would still see both. A fence forces three segments.
 *
 * The block's CONTENT is identical in both halves of the pair, which the wave-2
 * version's was not: it wrote `fee_basis: retainer` on the client side and
 * `rate_card: annual` on the vendor side, so the middle segment alone answered
 * the question and the pair measured nothing about scope. Here the only
 * difference between the two halves is the final clause, and that clause names
 * no organisation.
 */
const CROSS_SEGMENT_BLOCK = "\n\n```\nreview: april\nowner: ops desk\n```\n\n";

/** Asserted byte-identical between the two cross-segment families by `corpus-v2.test.ts`. */
export const CROSS_SEGMENT_PREFIX = " the sheet i am reading lists ";
export const CROSS_SEGMENT_JOIN = " and ";
export const CROSS_SEGMENT_TAIL_HEAD = ` in that order.${CROSS_SEGMENT_BLOCK}`;
/** The client half's discriminating clause. Contains no organisation name. */
export const CROSS_SEGMENT_CLIENT_CLAUSE =
  "the first of those two is the one we act for, and the second only leases us the floor.";
/** The vendor half's. Same shape, same length of reasoning, no name. */
export const CROSS_SEGMENT_VENDOR_CLAUSE =
  "the first of those two prints our statements, and the second only leases us the floor.";

function crossSegmentCompanion(clause: string, type: string, family: string) {
  return (value: string): Companion => ({
    prefix: "",
    value: nextOrg(value),
    suffix: `${CROSS_SEGMENT_TAIL_HEAD}${clause}`,
    type,
    family,
    constructedRole: "landlord",
  });
}

// -- positives --------------------------------------------------------------

export const V2_POSITIVE_FAMILIES: readonly V2Family[] = [
  // in-pan (§2.1)
  {
    id: "pan-prose-boosted",
    type: "in-pan",
    surfaceName: "PAN in running prose",
    shapeClass: "alnum-id-10",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintPan,
    glue: g(" the permanent account number on the file reads ", "."),
    labelBasis: { clauses: ["§2.1"], reading: "PAN values must never be sent to an external assistant." },
  },
  {
    id: "pan-kv",
    type: "in-pan",
    surfaceName: "PAN on a configuration-style key line",
    shapeClass: "alnum-id-10",
    boost: "absent",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintPan,
    // Short on purpose: the whole line stays under `api-credential-entropy`'s
    // 24-character floor, so the item carries the regex hit and not an entropy
    // finding stacked on top of it.
    glue: g("\ntaxpayer_ref: ", "\n"),
    labelBasis: { clauses: ["§2.1"], reading: "PAN values must never be sent to an external assistant." },
  },
  {
    id: "pan-lowercase-prose",
    type: "in-pan",
    surfaceName: "PAN typed in lowercase",
    shapeClass: "alnum-id-10",
    boost: "absent",
    surface: "prose",
    difficulty: "paraphrased",
    register: "casual",
    constructedRole: "none",
    mint: mintLowercasePan,
    glue: g(" she had typed it in lowercase as ", " and the portal matched it anyway."),
    labelBasis: {
      clauses: ["§2.1", "§2.5"],
      reading:
        "the value is a PAN by the Income Tax Department's layout, and §2.5 refuses relief for masking, " +
        "truncating or reversing a value, so a change of case cannot be what stops it being one.",
    },
  },
  // in-aadhaar (§2.2)
  {
    id: "aadhaar-uidai-boosted",
    type: "in-aadhaar",
    surfaceName: "Aadhaar in the spaced form UIDAI prints",
    shapeClass: "digits-12",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintSpacedAadhaar,
    glue: g(" the uidai record we hold shows ", "."),
    labelBasis: { clauses: ["§2.2"], reading: "Aadhaar numbers must never be sent to an external assistant." },
  },
  {
    id: "aadhaar-plain",
    type: "in-aadhaar",
    surfaceName: "Aadhaar written as twelve unbroken digits",
    shapeClass: "digits-12",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintAadhaar,
    glue: g(" the twelve digit id on the e-kyc form is ", "."),
    labelBasis: { clauses: ["§2.2"], reading: "Aadhaar numbers must never be sent to an external assistant." },
  },
  {
    id: "aadhaar-dashed",
    type: "in-aadhaar",
    surfaceName: "Aadhaar written with hyphens",
    shapeClass: "digits-12",
    boost: "absent",
    surface: "prose",
    difficulty: "paraphrased",
    register: "casual",
    constructedRole: "none",
    mint: mintDashedAadhaar,
    glue: g(" she sent it over hyphenated as ", " and i typed it in exactly like that."),
    labelBasis: { clauses: ["§2.2"], reading: "Aadhaar numbers must never be sent to an external assistant." },
  },
  {
    id: "aadhaar-masked",
    type: "in-aadhaar",
    surfaceName: "UIDAI-masked Aadhaar showing only the last four digits",
    shapeClass: "masked-tail",
    boost: "carries",
    surface: "prose",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "none",
    mint: mintMaskedAadhaar,
    glue: g(" the aadhaar copy on file is masked and reads ", ", and the last four still match."),
    labelBasis: {
      clauses: ["§2.2", "§2.5"],
      reading: "§2.5 states the §2 prohibition is not relieved by masking part of the value.",
    },
  },
  // bank-account-identifier (§2.3)
  {
    id: "ifsc-neft-boosted",
    type: "bank-account-identifier",
    surfaceName: "IFSC branch code qualifying a transfer",
    shapeClass: "code-11",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintIfsc,
    glue: g(" the neft slip shows the code ", " against that transfer."),
    labelBasis: { clauses: ["§2.3"], reading: "§2.3 names IFSC-qualified account strings outright." },
  },
  {
    id: "ifsc-plain",
    type: "bank-account-identifier",
    surfaceName: "IFSC printed on a passbook",
    shapeClass: "code-11",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: mintIfsc,
    glue: g(" the eleven character code printed on the passbook is ", " and the branch has moved since."),
    labelBasis: { clauses: ["§2.3"], reading: "§2.3 names IFSC-qualified account strings outright." },
  },
  {
    id: "upi-vpa-boosted",
    type: "bank-account-identifier",
    surfaceName: "UPI virtual payment address",
    shapeClass: "handle-at-domain",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: mintFictionalVpa,
    glue: g(" the collect request went out to ", " and it has still not been accepted."),
    labelBasis: { clauses: ["§2.3"], reading: "§2.3 names UPI virtual payment addresses outright." },
  },
  {
    id: "upi-vpa-plain",
    type: "bank-account-identifier",
    surfaceName: "UPI address quoted as a payment handle",
    shapeClass: "handle-at-domain",
    boost: "absent",
    surface: "prose",
    difficulty: "paraphrased",
    register: "casual",
    constructedRole: "none",
    mint: mintFictionalVpa,
    glue: g(" she pays everything from the handle ", " these days."),
    labelBasis: { clauses: ["§2.3"], reading: "§2.3 names UPI virtual payment addresses outright." },
  },
  {
    id: "labelled-account-credited",
    type: "bank-account-identifier",
    surfaceName: "core banking account number after the word account",
    shapeClass: "digits-14",
    boost: "carries",
    surface: "labelled",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "none",
    mint: mintBareAccount14,
    // The gold span is the DIGITS: §2.3 forbids bank account numbers and the
    // number is the digits. p-fin's own `labelled-account-number` rule matches
    // wider, from the word "account" through them, so this family scores as an
    // overlap match and an exact-match miss by construction. Lengthening the
    // glue between "account" and the digits would push the IoU under 0.5 and
    // make it an iou50 miss too, which is why it is this short.
    glue: g(" the amount was credited to account ", " rather than the nominated one."),
    labelBasis: { clauses: ["§2.3"], reading: "§2.3 forbids bank account numbers outright." },
  },
  {
    id: "bare-account-digits",
    type: "bank-account-identifier",
    surfaceName: "core banking account number with no keyword in front of it",
    shapeClass: "digits-14",
    boost: "carries",
    surface: "prose",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintBareAccount14,
    // "beneficiary" is a contextBoost term and is NOT one of the four words
    // `labelled-account-number` requires (account, A/C, A/c, Acct), so this is
    // a boosted positive that tier 0 still cannot see -- which is the pair the
    // decision-2 measurement needs and the ladder needs at the same time.
    glue: g(" the money went to the beneficiary number ", " at the same office."),
    labelBasis: { clauses: ["§2.3"], reading: "§2.3 forbids bank account numbers outright." },
  },
  // internal-customer-id (§2.4)
  {
    id: "cif-id",
    type: "internal-customer-id",
    surfaceName: "CIF number",
    shapeClass: "prefixed-id",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: (rng) => mintPrefixedCustomerId(rng, "CIF", " "),
    glue: g(" the servicing console shows ", " against that relationship."),
    labelBasis: { clauses: ["§2.4"], reading: "§2.4 names CIF numbers outright." },
  },
  {
    id: "crn-id",
    type: "internal-customer-id",
    surfaceName: "customer reference number",
    shapeClass: "prefixed-id",
    boost: "absent",
    surface: "labelled",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: (rng) => mintPrefixedCustomerId(rng, "CRN", "-"),
    glue: g(" the reference quoted on the form is ", "."),
    labelBasis: { clauses: ["§2.4"], reading: "§2.4 names customer reference numbers outright." },
  },
  {
    id: "kyc-case-id",
    type: "internal-customer-id",
    surfaceName: "KYC case id",
    shapeClass: "prefixed-id",
    boost: "absent",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: (rng) => mintPrefixedCustomerId(rng, "KYC", "/"),
    glue: g(" the case raised at onboarding is ", " and it is still open."),
    labelBasis: { clauses: ["§2.4"], reading: "§2.4 names KYC case ids outright." },
  },
  {
    id: "relationship-number",
    type: "internal-customer-id",
    surfaceName: "unprefixed relationship number",
    shapeClass: "digits-8",
    boost: "carries",
    surface: "prose",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintRelationshipNumber,
    // §2.4 forbids customer reference numbers and does not require a prefix;
    // `internal-customer-id-format` only matches CIF, CRN or KYC followed by
    // digits, so this is a §2.4 positive no tier-0 rule reaches.
    glue: g(" the customer reference on that relationship is just ", " with no prefix at all."),
    labelBasis: { clauses: ["§2.4"], reading: "§2.4 forbids customer reference numbers; it does not require a prefix." },
  },
  // api-credential (§4.1)
  {
    id: "aws-key-fence",
    type: "api-credential",
    surfaceName: "AWS access key id in a shell snippet",
    shapeClass: "opaque-token",
    boost: "absent",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintAwsKey,
    glue: g("\n\n```sh\nexport AWS_ACCESS_KEY_ID=", "\n```\n\n"),
    labelBasis: { clauses: ["§4.1"], reading: "§4.1 forbids API keys and access tokens outright." },
  },
  {
    id: "ghp-token-prose",
    type: "api-credential",
    surfaceName: "GitHub personal access token",
    shapeClass: "opaque-token",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintGhpToken,
    glue: g(" the runner is still using that access token, ", ", and nobody has revoked it."),
    labelBasis: {
      clauses: ["§4.1"],
      reading: "§4.1 forbids access tokens including when the sender believes the credential is already revoked.",
    },
  },
  {
    id: "slack-token-kv",
    type: "api-credential",
    surfaceName: "Slack bot token on a key line",
    shapeClass: "opaque-token",
    boost: "absent",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintSlackToken,
    glue: g("\nbot_token: ", "\n"),
    labelBasis: { clauses: ["§4.1"], reading: "§4.1 forbids access tokens outright." },
  },
  {
    id: "client-secret-entropy-kv",
    type: "api-credential",
    surfaceName: "unprefixed client secret",
    shapeClass: "opaque-token",
    boost: "absent",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintOpaqueClientSecret,
    // The only positive in the corpus that reaches tier 0 through the ENTROPY
    // rule rather than a prefix regex. Without it `api-credential-entropy` is
    // only ever observed firing on private-key bodies, i.e. only as a false
    // positive, and a rule whose true positives are never exercised is a rule
    // nobody has measured.
    glue: g("\nclient_secret: ", "\n"),
    labelBasis: { clauses: ["§4.1"], reading: "§4.1 names client secrets outright." },
  },
  {
    id: "jwt-prose",
    type: "api-credential",
    surfaceName: "JWT bearer token",
    shapeClass: "opaque-token",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintJwt,
    glue: g(" the authorization header carried the bearer token ", " and it had not expired."),
    labelBasis: { clauses: ["§4.1"], reading: "a bearer token is an access token, which §4.1 forbids outright." },
  },
  {
    id: "azure-storage-key-kv",
    type: "api-credential",
    surfaceName: "Azure storage account key",
    shapeClass: "opaque-token",
    boost: "absent",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintAzureStorageKey,
    glue: g("\nstorage_key: ", "\n"),
    labelBasis: { clauses: ["§4.1"], reading: "a storage account key is an API key, which §4.1 forbids outright." },
  },
  // db-connection-string (§4.2)
  {
    id: "postgres-url-prose",
    type: "db-connection-string",
    surfaceName: "postgres connection URL",
    shapeClass: "driver-url",
    boost: "carries",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintPostgresUrl,
    // The suffix begins with a SPACE, not a full stop: `db-connection-string-format`
    // matches `[^\s"']+` after the scheme, so a suffix starting with punctuation
    // would be swallowed into the match and every finding would be one character
    // wider than the gold span -- an exact-match miss manufactured by the glue.
    glue: g(" the datasource on the archive box is ", " and the schema is read only."),
    labelBasis: { clauses: ["§4.2"], reading: "§4.2 forbids database connection strings outright." },
  },
  {
    id: "jdbc-url-kv",
    type: "db-connection-string",
    surfaceName: "JDBC connection URL",
    shapeClass: "driver-url",
    boost: "carries",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintJdbcUrl,
    glue: g("\n\nthe dsn on the archive box is\n\njdbc_url: ", "\n\n"),
    labelBasis: { clauses: ["§4.2"], reading: "§4.2 forbids database connection strings outright." },
  },
  {
    id: "redis-url-kv",
    type: "db-connection-string",
    surfaceName: "redis connection URL",
    shapeClass: "driver-url",
    boost: "absent",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintRedisUrl,
    glue: g("\ncache_endpoint: ", "\n"),
    labelBasis: {
      clauses: ["§4.2"],
      reading:
        "§4.2 forbids database connection strings, and its stated reason is that host and schema names " +
        "disclose internal topology; a redis URL carries both.",
    },
  },
  // private-key-material (§4.3)
  {
    id: "rsa-pem-fence",
    type: "private-key-material",
    surfaceName: "RSA private key block",
    shapeClass: "pem-block",
    boost: "absent",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintRsaPem,
    // The gold span covers the WHOLE block, header to footer, not just the
    // header `pem-private-key-header` matches: the body is the secret and the
    // header is a marker for it. MEASURED consequence, stated rather than
    // discovered in a results table: p-fin's own rules return three findings
    // inside the span -- `private-key-material` on the header and
    // `api-credential` twice on the body lines, because the entropy rule fires
    // on them -- and none outside it, so this family reads 0 on exact-match by
    // construction.
    glue: g("\n\n```\n", "\n```\n\n"),
    labelBasis: { clauses: ["§4.3"], reading: "§4.3 forbids private keys outright." },
  },
  {
    id: "openssh-pem-fence",
    type: "private-key-material",
    surfaceName: "OpenSSH private key block",
    shapeClass: "pem-block",
    boost: "absent",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintOpensshPem,
    glue: g("\n\n```\n", "\n```\n\n"),
    labelBasis: { clauses: ["§4.3"], reading: "§4.3 forbids private keys outright." },
  },
  {
    id: "ec-pem-fence",
    type: "private-key-material",
    surfaceName: "EC private key block",
    shapeClass: "pem-block",
    boost: "absent",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintEcPem,
    glue: g("\n\n```\n", "\n```\n\n"),
    labelBasis: { clauses: ["§4.3"], reading: "§4.3 forbids private keys outright." },
  },
  {
    id: "encrypted-pem-fence",
    type: "private-key-material",
    surfaceName: "encrypted private key block",
    shapeClass: "pem-block",
    boost: "absent",
    surface: "code-fence",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintEncryptedPem,
    glue: g("\n\nthe key is encrypted at rest and this is the file:\n\n```\n", "\n```\n\n"),
    labelBasis: { clauses: ["§4.3"], reading: "§4.3 is unqualified: encryption at rest is not relief from it." },
  },
  {
    id: "certificate-pem-fence",
    type: "private-key-material",
    surfaceName: "X.509 certificate block",
    shapeClass: "pem-block",
    boost: "carries",
    surface: "code-fence",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintCertificatePem,
    glue: g("\n\nthe certificate material we were sent is\n\n```\n", "\n```\n\n"),
    labelBasis: {
      clauses: ["§4.3"],
      reading:
        "§4.3 reads 'Private keys, certificate material, and HSM key labels must never be included', and the " +
        "IR's own private-key-material nlDefinition reads 'Private key or certificate material'. Both cover a " +
        "certificate, so the label follows the IR as well as the document. What no artifact covers is the " +
        "MATCH: pem-private-key-header matches only literal PRIVATE KEY headers, so this is a §4.3 positive " +
        "no tier-0 rule reaches.",
    },
  },
  {
    id: "hsm-key-label",
    type: "private-key-material",
    surfaceName: "HSM key label",
    shapeClass: "kebab-label",
    boost: "carries",
    surface: "labelled",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintHsmKeyLabel,
    glue: g(" the hsm key label on the module is ", " if that narrows it down."),
    labelBasis: { clauses: ["§4.3"], reading: "§4.3 names HSM key labels outright." },
  },
  // client-name (§3.1, §3.3)
  {
    id: "client-org-mandate",
    type: "client-name",
    surfaceName: "named organisation in a mandate clause",
    shapeClass: "org-name",
    boost: "absent",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "client",
    orgRole: "client",
    mint: pickOrg,
    glue: g(" the mandate we hold for ", " renews in march and nobody has picked it up."),
    labelBasis: { clauses: ["§3.1"], reading: "§3.1 requires client organisation names to be pseudonymized." },
  },
  {
    id: "client-org-prospect",
    type: "client-name",
    surfaceName: "named organisation being pitched to",
    shapeClass: "org-name",
    boost: "absent",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "prospect",
    orgRole: "prospect",
    mint: pickOrg,
    glue: g(" we are pitching ", " next month and the deck is nowhere near ready."),
    labelBasis: { clauses: ["§3.3"], reading: "§3.3 treats prospective clients exactly as existing clients." },
  },
  {
    id: "client-org-nda",
    type: "client-name",
    surfaceName: "named party to a confidentiality agreement",
    shapeClass: "org-name",
    boost: "absent",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "nda-party",
    orgRole: "nda-party",
    mint: pickOrg,
    glue: g(" the party we signed the confidentiality agreement with is ", " and the term runs to december."),
    labelBasis: { clauses: ["§3.3"], reading: "§3.3 treats parties under a non-disclosure agreement as client names." },
  },
  {
    id: "client-org-trade-counterparty",
    type: "client-name",
    surfaceName: "named organisation on the other side of a trade",
    shapeClass: "org-name",
    boost: "absent",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "counterparty",
    orgRole: "counterparty",
    mint: pickOrg,
    glue: g(" we are on the other side of that trade from ", ", so the netting question actually matters."),
    labelBasis: { clauses: ["§3.1", "§3.3"], reading: "§3.1 and §3.3 reach deal counterparty names." },
  },
  {
    id: "client-org-fee-note",
    type: "client-name",
    surfaceName: "named organisation being billed a fee",
    shapeClass: "org-name",
    boost: "absent",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "client",
    orgRole: "client",
    mint: pickOrg,
    glue: g(" the fee note we send to ", " is still on the schedule we agreed two years ago."),
    labelBasis: { clauses: ["§3.1"], reading: "§3.1 requires client organisation names to be pseudonymized." },
  },
  {
    id: "client-org-onboarding",
    type: "client-name",
    surfaceName: "named organisation part-way through onboarding",
    shapeClass: "org-name",
    boost: "absent",
    surface: "prose",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "prospect",
    orgRole: "prospect",
    mint: pickOrg,
    glue: g(" the onboarding pack for ", " has been sitting with legal for a fortnight."),
    labelBasis: { clauses: ["§3.3"], reading: "§3.3 treats prospective clients exactly as existing clients." },
  },
  {
    id: "client-org-cross-segment",
    type: "client-name",
    surfaceName: "named organisation whose role is assigned in a later segment",
    shapeClass: "org-name",
    boost: "absent",
    surface: "prose",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "client",
    orgRole: "client",
    selfDistracting: true,
    mint: pickOrg,
    glue: g(CROSS_SEGMENT_PREFIX, CROSS_SEGMENT_JOIN),
    companion: crossSegmentCompanion(
      CROSS_SEGMENT_CLIENT_CLAUSE,
      `${NEG_PREFIX}org-cross-segment-landlord`,
      "client-org-cross-segment-landlord",
    ),
    labelBasis: {
      clauses: ["§3.1"],
      reading:
        "the message says a named organisation is one the Firm acts for; no single segment does, which is " +
        "what makes the predicate's declared message scope observable.",
    },
  },
];

// -- confusables ------------------------------------------------------------

export const V2_CONFUSABLE_FAMILIES: readonly V2Family[] = [
  {
    id: "tan-boosted",
    type: `${NEG_PREFIX}tan`,
    surfaceName: "TAN (tax deduction and collection account number)",
    shapeClass: "alnum-id-10",
    boost: "carries",
    nearMissFor: "in-pan",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintTan,
    glue: g(" the income tax challan carries the deduction account number ", ", which is the firm's own."),
    labelBasis: {
      clauses: ["§1.2", "§2"],
      reading:
        "§2's list is closed -- PAN, Aadhaar, bank account numbers, internal customer identifiers -- and a " +
        "TAN is the Firm's own account with the tax department, not information identifying a person or " +
        "entity holding an account WITH the Firm.",
    },
  },
  {
    id: "tan-plain",
    type: `${NEG_PREFIX}tan`,
    surfaceName: "TAN (tax deduction and collection account number)",
    shapeClass: "alnum-id-10",
    boost: "absent",
    nearMissFor: "in-pan",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintTan,
    glue: g(" the deduction reference on the challan is ", " and it has not changed in years."),
    labelBasis: {
      clauses: ["§1.2", "§2"],
      reading: "as tan-boosted: §2's list is closed and a TAN is the Firm's own tax account with the department.",
    },
  },
  {
    id: "imps-rrn",
    type: `${NEG_PREFIX}retrieval-reference`,
    surfaceName: "NPCI retrieval reference number",
    shapeClass: "digits-12",
    boost: "carries",
    nearMissFor: "in-aadhaar",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintRetrievalReference,
    glue: g(" the branch quoted the retrieval reference on that credit to the resident as ", "."),
    labelBasis: {
      clauses: ["§2.3", "§2.4"],
      reading:
        "§2.3 enumerates bank account numbers, IFSC-qualified account strings and UPI addresses; §2.4 " +
        "enumerates customer reference numbers, CIF numbers and KYC case ids. A retrieval reference names a " +
        "TRANSACTION and is issued by the payment network, not by the Firm, so it is in neither list.",
      contestedBy:
        "§1.2, if 'can be combined with other information to identify' is read as operative rather than as a " +
        "definition feeding §2-§4: a retrieval reference joins to the Firm's own records.",
    },
  },
  {
    id: "settlement-batch-sequence",
    type: `${NEG_PREFIX}batch-sequence`,
    surfaceName: "settlement batch sequence number",
    shapeClass: "digits-12",
    boost: "absent",
    nearMissFor: "in-aadhaar",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintBatchSequence12,
    glue: g(" the batch sequence on last night's file was ", " and it has not rolled over."),
    labelBasis: {
      clauses: ["§1.2", "§2"],
      reading: "a batch sequence names a processing run; it identifies no person and no entity, so §1.2 does not reach it.",
    },
  },
  {
    id: "masked-mobile",
    type: `${NEG_PREFIX}masked-mobile`,
    surfaceName: "masked telephone number",
    shapeClass: "masked-tail",
    boost: "carries",
    nearMissFor: "in-aadhaar",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: mintMaskedMobile,
    glue: g(" the uidai sms went to ", " which is not a number she recognises."),
    labelBasis: {
      clauses: ["§2"],
      reading: "§2's list carries no telephone number, masked or otherwise, so §2.5 has nothing to relieve.",
    },
  },
  {
    id: "scheme-code",
    type: `${NEG_PREFIX}scheme-code`,
    surfaceName: "internal product scheme code",
    shapeClass: "code-11",
    boost: "carries",
    nearMissFor: "bank-account-identifier",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintSchemeCode,
    // The same mint as an IFSC on purpose: `ifsc-qualified-account` fires on
    // it, so this is a measured false positive and the only thing separating it
    // from the positive is the clause.
    glue: g(" the product scheme code on the statement is ", " and every branch uses the same one."),
    labelBasis: {
      clauses: ["§2.3"],
      reading: "§2.3's prohibition is on IFSC-qualified ACCOUNT strings; a product scheme code qualifies no account.",
    },
  },
  {
    id: "package-spec-plain",
    type: `${NEG_PREFIX}package-spec`,
    surfaceName: "npm package specifier",
    shapeClass: "handle-at-domain",
    boost: "absent",
    nearMissFor: "bank-account-identifier",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintPackageSpec,
    glue: g(" the lockfile still pins ", " and the build box resolves the old one."),
    labelBasis: {
      clauses: ["§2.3"],
      reading: "§2.3 reaches UPI virtual payment addresses; a package specifier names software and no account.",
    },
  },
  {
    // The boosted half of the pair. `upi-vpa-boosted` puts "collect request"
    // beside a positive and `upi-vpa-plain` puts nothing beside one, so the
    // confusable side needs both halves too, or the presence of a boost term
    // near a handle@domain string becomes a signal for the label.
    id: "package-spec-boosted",
    type: `${NEG_PREFIX}package-spec`,
    surfaceName: "npm package specifier",
    shapeClass: "handle-at-domain",
    boost: "carries",
    nearMissFor: "bank-account-identifier",
    surface: "prose",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintPackageSpec,
    glue: g(" the lockfile pins ", " and the current build cannot resolve it at all."),
    labelBasis: {
      clauses: ["§2.3"],
      reading: "as package-spec-plain: a package specifier names software and no account.",
    },
  },
  {
    id: "settlement-batch-number",
    type: `${NEG_PREFIX}batch-number`,
    surfaceName: "settlement batch number",
    shapeClass: "digits-14",
    boost: "carries",
    nearMissFor: "bank-account-identifier",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintBatchNumber14,
    glue: g(" the batch number on the run that credited them was ", " and it is in the header of every row."),
    labelBasis: { clauses: ["§2.3"], reading: "§2.3 forbids account numbers; a batch number names a run." },
  },
  {
    id: "pos-terminal-id",
    type: `${NEG_PREFIX}terminal-id`,
    surfaceName: "card terminal id",
    shapeClass: "digits-8",
    boost: "carries",
    nearMissFor: "internal-customer-id",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: mintTerminalId,
    glue: g(" the card terminal id on the charge slip is ", " and the servicing console shows nothing against it."),
    labelBasis: {
      clauses: ["§2.3", "§2.4"],
      reading: "a terminal id names a device in a shop; it is neither an account number nor a customer reference.",
    },
  },
  {
    id: "kyc-standard-version",
    type: `${NEG_PREFIX}standard-version`,
    surfaceName: "version number of an internal policy document",
    shapeClass: "prefixed-id",
    boost: "carries",
    nearMissFor: "internal-customer-id",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintKycStandardVersion,
    // `internal-customer-id-format` fires on it: the rule cannot tell a case
    // from a document.
    glue: g(" the kyc standard we work to is ", " and it was reissued in march."),
    labelBasis: {
      clauses: ["§2.4"],
      reading: "§2.4's list is customer reference numbers, CIF numbers and KYC case ids; a document version is none of them.",
    },
  },
  {
    id: "vendor-invoice-number",
    type: `${NEG_PREFIX}invoice-number`,
    surfaceName: "supplier invoice number",
    shapeClass: "prefixed-id",
    boost: "absent",
    nearMissFor: "internal-customer-id",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintInvoiceNumber,
    glue: g(" the supplier invoice we are chasing is ", " and it is six weeks old."),
    labelBasis: {
      clauses: ["§1.2", "§2.4"],
      reading: "an invoice number names an invoice from a supplier, who holds no account with the Firm.",
    },
  },
  {
    id: "uuid",
    type: `${NEG_PREFIX}uuid`,
    surfaceName: "correlation id",
    shapeClass: "opaque-token",
    boost: "carries",
    nearMissFor: "api-credential",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintUuid,
    // Injected in PROSE deliberately: a UUID is 36 characters over a 17-symbol
    // alphabet, so in a kv or code segment it sits right on the stock 4.0
    // entropy threshold and the family would be measuring a threshold rather
    // than a classifier.
    glue: g(" the correlation id printed next to the access token line is ", "."),
    labelBasis: {
      clauses: ["§4.1"],
      reading: "§4.1 reaches credentials that authenticate a caller; a correlation id authenticates nothing.",
    },
  },
  {
    id: "git-commit-sha",
    type: `${NEG_PREFIX}git-commit-sha`,
    surfaceName: "git commit id",
    shapeClass: "opaque-token",
    boost: "absent",
    nearMissFor: "api-credential",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintGitSha,
    glue: g(" the rotation landed in commit ", " last thursday."),
    labelBasis: { clauses: ["§4.1"], reading: "a commit id addresses a revision; it authenticates nothing." },
  },
  {
    id: "base64-image-fragment",
    type: `${NEG_PREFIX}base64-image-fragment`,
    surfaceName: "fragment of a base64-encoded image",
    shapeClass: "opaque-token",
    boost: "absent",
    nearMissFor: "api-credential",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintBase64Fragment,
    // Injected in a FENCE deliberately, for the opposite reason to the UUID:
    // entropy rules only scan code and kv segments, so in prose this would be a
    // confusable that confuses nothing.
    glue: g("\n\nnothing sensitive was in the paste, this is what came through:\n\n```\n", "\n```\n\n"),
    labelBasis: { clauses: ["§4.1"], reading: "image bytes authenticate no caller." },
  },
  {
    id: "artifact-digest",
    type: `${NEG_PREFIX}artifact-digest`,
    surfaceName: "container image digest",
    shapeClass: "opaque-token",
    boost: "absent",
    nearMissFor: "api-credential",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintArtifactDigest,
    glue: g(" the image we deployed is pinned by digest ", " and that has not moved."),
    labelBasis: { clauses: ["§4.1"], reading: "a content digest addresses bytes; it authenticates nothing." },
  },
  {
    id: "ssh-public-key-line",
    type: `${NEG_PREFIX}ssh-public-key`,
    surfaceName: "ssh public key line",
    shapeClass: "opaque-token",
    boost: "carries",
    nearMissFor: "private-key-material",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintSshPublicKeyLine,
    glue: g("\n\nthe public half of the keypair is\n\n```\n", "\n```\n\n"),
    labelBasis: {
      clauses: ["§4.3"],
      reading: "§4.3 reaches private keys, certificate material and HSM key labels; a public key is none of the three.",
    },
  },
  {
    id: "sftp-endpoint",
    type: `${NEG_PREFIX}sftp-endpoint`,
    surfaceName: "sftp file-transfer endpoint",
    shapeClass: "driver-url",
    boost: "carries",
    nearMissFor: "db-connection-string",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintSftpEndpoint,
    glue: g(" the box we connect to for the file drop is ", " and it holds nothing but the outbound files."),
    labelBasis: {
      clauses: ["§4.2"],
      reading: "§4.2's operative sentence governs database connection strings; an sftp endpoint names no driver and no schema.",
      contestedBy:
        "§4.2's stated reason -- that host and schema names disclose the Firm's internal topology -- which an " +
        "internal sftp hostname does disclose.",
    },
  },
  {
    id: "datasource-alias",
    type: `${NEG_PREFIX}datasource-alias`,
    surfaceName: "datasource alias",
    // Not `driver-url`: an alias is a bare word and would be an invisible
    // distractor beside a URL, which is the opposite of what a shape neighbour
    // is for. It stays a confusable in its own right and is never drawn as one.
    shapeClass: "alias-word",
    boost: "absent",
    nearMissFor: "db-connection-string",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintDatasourceAlias,
    glue: g("\n\nthe one here is named by alias and the host lives in the vault.\n\nalias: ", "\n\n"),
    labelBasis: {
      clauses: ["§4.2"],
      reading: "an alias names no host, no schema and no driver, so neither §4.2's sentence nor its reason reaches it.",
    },
  },
  {
    id: "dh-parameters-block",
    type: `${NEG_PREFIX}dh-parameters-block`,
    surfaceName: "Diffie-Hellman parameters block",
    shapeClass: "pem-block",
    boost: "absent",
    nearMissFor: "private-key-material",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintDhParametersBlock,
    glue: g("\n\nthe tls handshake parameters we ship are\n\n```\n", "\n```\n\n"),
    labelBasis: {
      clauses: ["§4.3"],
      reading:
        "Diffie-Hellman parameters are public constants: not a private key, not a certificate, not an HSM key label.",
    },
  },
  {
    id: "deployment-name",
    type: `${NEG_PREFIX}deployment-name`,
    surfaceName: "kubernetes deployment name",
    shapeClass: "kebab-label",
    boost: "carries",
    nearMissFor: "private-key-material",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintDeploymentName,
    glue: g(" the workload that reads that hsm key is the deployment ", " and nothing else touches the module."),
    labelBasis: { clauses: ["§4.3"], reading: "§4.3 reaches HSM key labels; a workload name is not a key label." },
  },
  {
    id: "org-vendor",
    type: `${NEG_PREFIX}org-vendor`,
    surfaceName: "named organisation in a supplier clause",
    shapeClass: "org-name",
    boost: "absent",
    nearMissFor: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "vendor",
    orgRole: "vendor",
    mint: pickOrg,
    glue: g(" the office fit-out contract is with ", " and they have not invoiced us since june."),
    labelBasis: {
      clauses: ["§3.1", "§3.3"],
      reading: "§3 reaches clients, prospective clients, deal counterparties and NDA parties; a supplier is none of them.",
    },
  },
  {
    id: "org-landlord",
    type: `${NEG_PREFIX}org-landlord`,
    surfaceName: "named organisation in a lease clause",
    shapeClass: "org-name",
    boost: "absent",
    nearMissFor: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "landlord",
    orgRole: "landlord",
    mint: pickOrg,
    glue: g(" the floor we sit on is leased from ", " and the lift has been out for a week."),
    labelBasis: { clauses: ["§3.1", "§3.3"], reading: "a landlord is not a client, a prospect, a counterparty or an NDA party." },
  },
  {
    id: "org-competitor",
    type: `${NEG_PREFIX}org-competitor`,
    surfaceName: "named organisation in a competitor clause",
    shapeClass: "org-name",
    boost: "absent",
    nearMissFor: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "competitor",
    orgRole: "competitor",
    mint: pickOrg,
    glue: g(" our nearest competitor ", " has started quoting a lower fee for the same work."),
    labelBasis: { clauses: ["§3.1", "§3.3"], reading: "a competitor is not a client, a prospect, a counterparty or an NDA party." },
  },
  {
    id: "org-listed-company",
    type: `${NEG_PREFIX}org-listed-company`,
    surfaceName: "named organisation read about in the trade press",
    shapeClass: "org-name",
    boost: "absent",
    nearMissFor: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "listed-company",
    orgRole: "listed-company",
    mint: pickOrg,
    glue: g(" the trade press has been writing about ", " all week and none of it touches us."),
    labelBasis: { clauses: ["§3.1", "§3.3"], reading: "a company the Firm has no relationship with is outside §3 entirely." },
  },
  {
    id: "org-vendor-cross-segment",
    type: `${NEG_PREFIX}org-cross-segment-supplier`,
    surfaceName: "named organisation whose supplier role is assigned in a later segment",
    shapeClass: "org-name",
    boost: "absent",
    nearMissFor: "client-name",
    surface: "prose",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "vendor",
    orgRole: "vendor",
    selfDistracting: true,
    mint: pickOrg,
    glue: g(CROSS_SEGMENT_PREFIX, CROSS_SEGMENT_JOIN),
    companion: crossSegmentCompanion(
      CROSS_SEGMENT_VENDOR_CLAUSE,
      `${NEG_PREFIX}org-cross-segment-landlord`,
      "org-vendor-cross-segment-landlord",
    ),
    labelBasis: {
      clauses: ["§3.1", "§3.3"],
      reading: "read as a whole message the two organisations are a printer and a landlord; §3 reaches neither.",
    },
  },
  {
    id: "person-name",
    type: `${NEG_PREFIX}person-name`,
    surfaceName: "colleague's name",
    shapeClass: "person-name",
    boost: "absent",
    nearMissFor: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: (rng) => pickFrom(rng, PERSON_NAMES),
    glue: g(" the note was drafted by ", " on the operations desk."),
    labelBasis: {
      clauses: ["§1.2", "§3.1"],
      reading:
        "§1.2 scopes customer data to a person or entity holding an account with the Firm, and §3 reaches " +
        "organisation names; a colleague is neither.",
    },
  },
  {
    id: "product-name",
    type: `${NEG_PREFIX}product-name`,
    surfaceName: "internal software product name",
    shapeClass: "product-name",
    boost: "absent",
    nearMissFor: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: (rng) => pickFrom(rng, PRODUCT_NAMES),
    glue: g(" the scheduler we run everything through is called ", " and it has no retry setting at all."),
    labelBasis: { clauses: ["§3.1"], reading: "§3 reaches organisation names; a piece of software is not an organisation." },
  },
];

export const V2_FAMILIES: readonly V2Family[] = [...V2_POSITIVE_FAMILIES, ...V2_CONFUSABLE_FAMILIES];

/**
 * Every wave-1 and wave-2 family that wave 3 REMOVES, with the reason.
 *
 * A register rather than a deletion, because "the corpus no longer measures
 * this" is a fact a reader of the numbers needs. It reaches the emitted
 * manifest verbatim.
 */
export const REMOVED_FAMILIES: readonly { readonly id: string; readonly why: string }[] = [
  {
    id: "pan-shaped-invalid",
    why: "1:1 with the IR counterExample 'ABCDE1234F' (a PAN-shaped string with an invalid holder-type character). tan-boosted / tan-plain replace it with a surface the IR names nowhere.",
  },
  {
    id: "aadhaar-shaped-invalid",
    why: "1:1 with the IR counterExamples '1234 5678 9012' and '100234567890'. Replaced by imps-rrn and settlement-batch-sequence, both of which carry a VALID Verhoeff digit, so the shipping validator cannot separate them from an Aadhaar and only the clause can.",
  },
  { id: "email-address", why: "1:1 with the IR counterExample 'priya.sharma@meridiancap.example.com'. Replaced by package-spec, which the same `upi-vpa-format` rule also over-fires on." },
  { id: "swift-bic", why: "1:1 with the IR counterExample 'SWIFT HDFCINBB'." },
  { id: "micr-code", why: "1:1 with the IR counterExample 'MICR 400240012'." },
  { id: "ticket-id", why: "1:1 with the IR counterExample 'ticket INC0042318'. Replaced by kyc-standard-version, which the rule actually fires on." },
  { id: "employee-id", why: "1:1 with the IR counterExample 'employee id 88213'." },
  { id: "gstin", why: "1:1 with the IR counterExample '27AABCU9603R1ZM'." },
  { id: "internal-http-url", why: "1:1 with the IR counterExample 'https://intranet.example.com/runbook'. Replaced by sftp-endpoint and datasource-alias." },
  {
    id: "redacted-credential-placeholder",
    why: "1:1 with the IR counterExample 'sk-...', which is the same idea in fewer characters. What goes with it is a real capability -- it was the family that showed `api-credential-prefix` firing on a SENTENCE about a key rather than on a key -- and nothing in wave 3 replaces it, because every credential-prefix confusable that is not the IR's own counterExample is also one the policy makes arguable.",
  },
  {
    id: "tutorial-api-key",
    why: "1:1 with the IR counterExample 'AKIA followed by sixteen uppercase characters', and its `neg:` label is contested by §4.1, which forbids API keys without qualifying the sentence. Removed rather than relabelled: it is a genuine question for the label adjudication round and a bad span to score anyone on.",
  },
  { id: "public-key-fingerprint", why: "1:1 with the IR counterExample 'the public key fingerprint'. Replaced by ssh-public-key-line, which the IR names nowhere and which the entropy rule fires on inside a fence." },
  {
    id: "csr-pem-block",
    why: "1:1 with the IR counterExample '-----BEGIN CERTIFICATE REQUEST-----', AND its `neg:` label is contradicted by §4.3, which forbids 'certificate material'. Both halves are disqualifying. certificate-pem-fence takes the §4.3 reading as a POSITIVE, and dh-parameters-block supplies a dashed-block confusable the policy does settle.",
  },
  {
    id: "own-employer-org",
    why: "the employer is the one organisation role that is fixed for the whole corpus, so its name is role-locked by construction and no clause can unlock it. Removing it costs the corpus the question 'is the Firm's own name a client name', which is stated rather than hidden.",
  },
  { id: "non-client-org", why: "drew from NON_CLIENT_ORGS, a pool disjoint from the client pool and different in flavour, so the role was readable off the name. Replaced by org-vendor drawing from the one shared pool." },
  { id: "client-org-client-role", why: "drew from CLIENT_ORGS, likewise role-locked. Replaced by client-org-mandate drawing from the one shared pool." },
  { id: "client-org-counterparty-role", why: "as client-org-client-role: it drew from CLIENT_ORGS, a pool no supplier family ever draws from, so the counterparty role was readable off the name. client-org-trade-counterparty draws from the shared pool." },
  { id: "competitor-org", why: "drew from COMPETITOR_ORGS, a pool that can only ever be a competitor. Replaced by org-competitor drawing from the shared pool." },
  { id: "listed-company-in-news", why: "drew from LISTED_COMPANY_ORGS, likewise. Replaced by org-listed-company drawing from the shared pool." },
  {
    id: "pan-under-aadhaar-context",
    why: "the idea -- context words pointing at the WRONG entity type -- is now a property of the whole catalogue rather than of one family: `boost` is declared per family and the manifest measures how often a term of the span's own type sits near it, on both the positive and the confusable side. The PAN-in-prose surface is covered by pan-prose-boosted and pan-kv.",
  },
  { id: "aadhaar-under-pan-context", why: "as pan-under-aadhaar-context: the misleading-context idea became a catalogue-wide balance rather than a family, and the Aadhaar-in-prose surface is covered by aadhaar-plain, aadhaar-uidai-boosted and aadhaar-dashed." },
  { id: "ifsc-kv", why: "the IFSC kv surface is not carried forward. ifsc-neft-boosted and ifsc-plain cover the surface in prose in both boost states, and the kv slot in the bank-account class goes to nothing; that is a loss of one surface and it is stated rather than absorbed." },
  { id: "mongo-url-kv", why: "mongodb+srv is one of the two driver URLs the IR's db-connection-string examples name. redis-url-kv takes the kv slot with a driver the IR names nowhere." },
  { id: "upi-vpa", why: "minted at okaxis, a live UPI handle of a real bank. upi-vpa-boosted and upi-vpa-plain mint at the invented handle okbluecrest." },
  { id: "sk-key-prose", why: "'sk-' is the one credential prefix the IR carries twice, as an api-credential example and as its own counterExample. jwt-prose and azure-storage-key-kv cover the prefix-less credential surfaces instead." },
  {
    id: "client-org-message-scope / dual-role-org-vendor-message-scope",
    why: "the pair did not isolate scope. Its two halves differed in the fenced block as well as the final clause (fee_basis: retainer against rate_card: annual), so the middle segment alone answered the question, and its final clause named the relationship type outright. client-org-cross-segment and org-vendor-cross-segment share a byte-identical block and differ only in a final clause that names no organisation.",
  },
];

/**
 * Wave-1 and wave-2 families that wave 3 CARRIES FORWARD under a different id.
 *
 * Separate from `REMOVED_FAMILIES` because the two answers a reader wants are
 * different: "this surface is gone from the corpus" and "this surface is still
 * here, under another name, with these changes". Merging them would let a
 * rename read as a removal and hide a real loss inside a list of cosmetic ones.
 */
export const RENAMED_FAMILIES: readonly { readonly from: string; readonly to: string; readonly why: string }[] = [
  { from: "pan-prose", to: "pan-prose-boosted", why: "renamed for what it now declares: it is the PAN family that carries a contextBoost term." },
  { from: "aadhaar-spaced", to: "aadhaar-uidai-boosted", why: "same spaced surface; the glue changed from 'the resident identifier we hold is' to 'the uidai record we hold shows' so the family's boost state is declared and measured." },
  { from: "ifsc-branch", to: "ifsc-neft-boosted", why: "same IFSC surface; the glue no longer says 'branch code', which is the phrase the IR's internal-customer-id counterExample 'branch code 0247' uses." },
  { from: "labelled-account", to: "labelled-account-credited", why: "same surface and the same measured overlap-not-exact behaviour; renamed for the boost term the glue carries." },
  { from: "db-url-prose", to: "postgres-url-prose", why: "same postgres surface, on a host under a domain that names no party rather than under the Firm's own." },
  { from: "pem-fence", to: "rsa-pem-fence", why: "renamed because wave 3 carries four private-key labels and a certificate, not one." },
  { from: "git-sha", to: "git-commit-sha", why: "same surface; the glue no longer says 'api key', so the family is a boost-absent confusable and the opaque-token class balances." },
  { from: "upi-vpa-fictional-handle", to: "upi-vpa-boosted", why: "same mint at the same invented handle; wave 3 adds upi-vpa-plain as its boost-absent half." },
  { from: "jdbc-url-prose", to: "jdbc-url-kv", why: "same jdbc surface, moved to a kv line so the driver-url class carries both surfaces and a boost term." },
  { from: "dual-role-org-mandate", to: "client-org-mandate", why: "the pool it draws from is no longer a special 'dual role' array; every organisation family draws from ORG_POOL, so the name said something that is now true of all of them." },
  { from: "dual-role-org-prospect", to: "client-org-prospect", why: "as dual-role-org-mandate: one shared organisation pool makes the old name redundant." },
  { from: "dual-role-org-nda", to: "client-org-nda", why: "as dual-role-org-mandate: one shared organisation pool makes the old name redundant." },
  { from: "dual-role-org-trade-counterparty", to: "client-org-trade-counterparty", why: "as dual-role-org-mandate: one shared organisation pool makes the old name redundant." },
  { from: "dual-role-org-vendor", to: "org-vendor", why: "as dual-role-org-mandate; the glue also dropped 'branch', which was the only contextBoost term on the organisation side." },
];

/** `neg:` type -> the entityType it is a near miss for. Read by the contextBoost measurement. */
export const PAIRED_TYPE: Readonly<Record<string, string>> = Object.fromEntries(
  V2_CONFUSABLE_FAMILIES.filter((f) => f.nearMissFor !== undefined).map((f) => [f.type, f.nearMissFor!]),
);

/**
 * The role class of an injection, for the decision-4 measurement.
 *
 * Reads `dimensions.constructedRole`, which is what the generator WROTE, and
 * maps it through `orgs.ts`'s own client-side/non-client split. Returns
 * `undefined` for anything that is not an organisation, so persons, products
 * and identifier families are outside the denominator.
 */
export function orgRoleClassOf(
  dimensions: Readonly<Record<string, string>>,
  _type: string,
): string | undefined {
  const role = dimensions["constructedRole"];
  if (role === undefined || role === "none") return undefined;
  return roleClass(role as OrgRole);
}

/** Every family that mints an organisation name, for the "one pool" assertion. */
export const ORG_FAMILY_IDS: readonly string[] = V2_FAMILIES.filter((f) => f.orgRole !== undefined).map((f) => f.id);

export function isOrgName(value: string): boolean {
  return (ORG_POOL as readonly string[]).includes(value);
}

export type { ConstructedRole };
