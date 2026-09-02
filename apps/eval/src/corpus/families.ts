import type { Register } from "./carriers.js";
import { NEG_PREFIX } from "./labels.js";
import {
  CLIENT_ORGS,
  NON_CLIENT_ORGS,
  mintAadhaar,
  mintAccountDigits,
  mintAwsKey,
  mintBadAadhaar,
  mintBic,
  mintCustomerId,
  mintDbUrl,
  mintEmail,
  mintGitSha,
  mintIfsc,
  mintInvalidPan,
  mintMicr,
  mintPan,
  mintPemBlock,
  mintSkKey,
  mintSlackToken,
  mintTicketId,
  mintUpiVpa,
} from "./universe.js";

/**
 * The injection catalogue: spec 6.2's "controlled dimensions (REDACT-style,
 * failures diagnosable)".
 *
 * ## The families are the measurement
 *
 * The last measured slate says span extraction is not this system's binding
 * constraint and CLASSIFICATION is: run 03's `baselineB-Qwen3.5-2B` placed a
 * finding at exactly the blind-labelled gold span, byte-identical, and called
 * it `in-pan` when it was an organisation name. A corpus that only adds more
 * spans to find cannot see that failure. So the families below are organised in
 * confusable PAIRS, and every pair differs in exactly the dimension a
 * classifier has to get right:
 *
 *   in-pan                    vs  neg:pan-shaped-invalid-holder
 *     -- same 10-character regex shape; the 4th character is or is not a PAN
 *        holder-type code, which is the whole difference between a taxpayer
 *        identifier and a string.
 *   in-aadhaar                vs  neg:aadhaar-shaped-bad-verhoeff
 *     -- same 12 digits, same leading-digit constraint; the check digit is or
 *        is not right.
 *   bank-account-identifier   vs  neg:email-address
 *     -- both are `handle@domain`. MEASURED: p-fin's `upi-vpa-format` rule fires
 *        on an ordinary email address (on "varsha.menon@brightline.example.com"
 *        it returns "varsha.menon@brightline"), so this pair is a live
 *        over-firing family and not a hypothetical one.
 *   bank-account-identifier   vs  neg:swift-bic  /  neg:micr-code
 *     -- an IFSC branch code, a BIC and a MICR line are all short banking codes;
 *        only the first qualifies an account.
 *   internal-customer-id      vs  neg:ticket-id
 *     -- `CIF 30045512` and `INC0042318` are the same genre of internal
 *        reference; p-fin covers one of them.
 *   api-credential            vs  neg:git-commit-sha
 *     -- both are long opaque hex-or-base62 tokens pasted out of a terminal.
 *   client-name               vs  neg:non-client-org
 *     -- THE SAME ORTHOGRAPHIC SHAPE, and often the same universe. What
 *        separates them is the role the surrounding clause puts the
 *        organisation in, which is a semantic judgement and not a lexical one.
 *        p-fin §3.1 reaches client, prospective-client and counterparty
 *        organisations; it does not reach a software supplier or a venue.
 *
 * ## `constructedRole` states what was written, not what is true
 *
 * For the organisation families the generator writes a clause that names the
 * role explicitly ("our client X", "our stationery supplier X"). Recording
 * `constructedRole` is therefore a fact about generation. It is deliberately
 * NOT a tier-2 predicate label: `pred:client-relationship-disclosure` gold in
 * this repository is blind-labelled by two annotators and adjudicated (see
 * `corpora/fixtures/smoke.gold-tier2.jsonl`), and a single author's call
 * dropped into that pipeline would be a provenance lie. The generator's job
 * here is to produce a stratified, labelling-READY set and to say so.
 */

/**
 * Known cosmetic artefact of block surfaces, written down rather than left to
 * be noticed: a carrier segment begins with a space (segments join with no
 * separator), so a `code-fence` or `kv-line` injection ending in a newline is
 * followed by "\n she said ..." -- a stray space after a line break. It is not
 * repaired, because every repair costs something worse: trimming after the
 * splice would move every offset computed before it and break the one property
 * this corpus rests on, and re-punctuating the carriers would mean editing the
 * pool to suit the injections. It affects rendering, not tokenization, and no
 * gold span includes it.
 */
export type Surface = "prose" | "labelled" | "kv-line" | "code-fence";
/**
 * Spec 6.2 names three difficulty tiers -- verbatim, paraphrased, implicit.
 * Only two are generated. An implicit reference ("our biggest client, the
 * Cupertino fruit company") injects no literal value, so there is no written
 * span to derive a gold span from, and the invariant that makes every gold span
 * here trustworthy does not hold for it. Generating implicit items would mean
 * re-finding spans by search, which `inject.ts` exists to forbid. The tier is
 * listed as a named gap in the manifest rather than approximated.
 */
export type Difficulty = "verbatim" | "paraphrased";
/**
 * The role the generator's clause puts an organisation in.
 *
 * Widened for wave 3, which draws every organisation from one pool and lets
 * only the clause assign the role (see `orgs.ts`). The wave-1 and wave-2
 * values are unchanged and still mean what they meant, so the two committed
 * corpora reproduce byte for byte; the additions are the roles p-fin §3 does
 * NOT reach beyond a supplier -- a landlord, a competitor, a listed company
 * read about in the trade press -- plus the two §3.3 roles wave 1 folded into
 * "client" and "counterparty".
 */
export type ConstructedRole =
  | "client"
  | "counterparty"
  | "vendor"
  | "none"
  | "prospect"
  | "nda-party"
  | "landlord"
  | "competitor"
  | "listed-company";

/**
 * A SECOND labelled span written into the same injection, immediately after the
 * family's own value and derived from it.
 *
 * It exists for one construction and is `undefined` everywhere else: the
 * cross-segment pair, which names two organisations in one clause and assigns
 * their roles by ordinal in a later segment ("the first of those two is the one
 * we act for"). Both names have to be labelled -- an organisation sitting in
 * the glue with no label is precisely the defect that made 25 of the previous
 * corpus's positives carry an unlabelled relationship disclosure -- and both
 * have to land in the same clause, which two independently slotted injections
 * cannot do.
 *
 * `value` is derived from the family's own value rather than drawn from the
 * rng because the two names must differ: a redraw can repeat, and "those two"
 * naming one organisation twice is nonsense no invariant here would catch.
 */
export interface Companion {
  readonly prefix: string;
  readonly value: string;
  readonly suffix: string;
  readonly type: string;
  readonly family: string;
  readonly constructedRole: ConstructedRole;
}

export interface Family {
  readonly id: string;
  /** The corpus label type: an IR entityType id, or a `neg:` confusable id. */
  readonly type: string;
  readonly surface: Surface;
  readonly difficulty: Difficulty;
  /** The register the GLUE is written in. Compared with the carrier's. */
  readonly register: Register;
  readonly constructedRole: ConstructedRole;
  /** Mints the value. Must be a pure function of the rng. */
  readonly mint: (rng: () => number) => string;
  /** Glue around the value. The span covers the value only. */
  readonly glue: (value: string) => { prefix: string; suffix: string };
  /**
   * A second labelled span in the same clause. See `Companion`. Absent on every
   * wave-1 and wave-2 family, which is why adding it left both committed
   * corpora byte-identical.
   */
  readonly companion?: (value: string) => Companion;
}

const p = (prefix: string, suffix: string) => () => ({ prefix, suffix });

export const POSITIVE_FAMILIES: readonly Family[] = [
  {
    id: "pan-prose",
    type: "in-pan",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintPan,
    glue: p(" the permanent account number on the file reads ", "."),
  },
  {
    id: "pan-kv",
    type: "in-pan",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintPan,
    // Short on purpose: the whole line is 22 characters, under the stock
    // entropy rule's minLength of 24, so the kv line does not add an entropy
    // finding of its own on top of the regex hit.
    glue: p("\npan_number: ", "\n"),
  },
  {
    id: "aadhaar-plain",
    type: "in-aadhaar",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintAadhaar,
    glue: p(" the resident identifier we hold is ", "."),
  },
  {
    id: "aadhaar-spaced",
    type: "in-aadhaar",
    surface: "prose",
    difficulty: "paraphrased",
    register: "casual",
    constructedRole: "none",
    mint: (rng) => {
      const a = mintAadhaar(rng);
      return `${a.slice(0, 4)} ${a.slice(4, 8)} ${a.slice(8, 12)}`;
    },
    glue: p(" they read it out as ", " over the phone."),
  },
  {
    id: "ifsc-branch",
    type: "bank-account-identifier",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintIfsc,
    glue: p(" the branch code on the transfer was ", "."),
  },
  {
    id: "upi-vpa",
    type: "bank-account-identifier",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: (rng) => mintUpiVpa(rng, "okaxis"),
    glue: p(" the collect request went to ", " instead."),
  },
  {
    id: "labelled-account",
    type: "bank-account-identifier",
    surface: "labelled",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "none",
    mint: mintAccountDigits,
    // The gold span is the DIGITS, because p-fin §2.3 forbids "bank account
    // numbers" and the number is the digits. p-fin's own
    // `labelled-account-number` rule matches wider, from the word "account"
    // through the digits. MEASURED consequence of the glue below: the rule's
    // match is 22 characters against a 14-character gold span, IoU 0.64, so
    // this family scores as an `overlap` and `iou50` match and an `exact` miss.
    // Lengthening the glue past "account " would push IoU under 0.5 and make it
    // an iou50 miss too, which is why the glue is this short.
    glue: p(" the credit landed in account ", " rather than the nominated one."),
  },
  {
    id: "cif-id",
    type: "internal-customer-id",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    // The span INCLUDES the "CIF" prefix, following the IR's own definition of
    // the entity ("a customer reference number, a CIF number, or a KYC case
    // id") and its examples, which are prefixed.
    mint: (rng) => mintCustomerId(rng, "CIF", " "),
    glue: p(" the servicing console shows ", " against that relationship."),
  },
  {
    id: "crn-id",
    type: "internal-customer-id",
    surface: "labelled",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: (rng) => mintCustomerId(rng, "CRN", "-"),
    glue: p(" the reference quoted on the ticket is ", "."),
  },
  {
    id: "aws-key-fence",
    type: "api-credential",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintAwsKey,
    glue: p("\n\n```sh\nexport AWS_ACCESS_KEY_ID=", "\n```\n\n"),
  },
  {
    id: "sk-key-prose",
    type: "api-credential",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: mintSkKey,
    glue: p(" someone pasted ", " into the shared channel this morning."),
  },
  {
    id: "slack-token-kv",
    type: "api-credential",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintSlackToken,
    glue: p("\nbot_token: ", "\n"),
  },
  {
    id: "db-url-prose",
    type: "db-connection-string",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: (rng) => mintDbUrl(rng, "postgresql"),
    glue: p(" the datasource is ", " if that matters."),
  },
  {
    id: "pem-fence",
    type: "private-key-material",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintPemBlock,
    glue: p("\n\n```\n", "\n```\n\n"),
  },
  {
    id: "client-org-client-role",
    type: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "client",
    mint: (rng) => CLIENT_ORGS[Math.floor(rng() * CLIENT_ORGS.length)]!,
    glue: p(" our client ", " has asked for the same breakdown."),
  },
  {
    id: "client-org-counterparty-role",
    type: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "counterparty",
    mint: (rng) => CLIENT_ORGS[Math.floor(rng() * CLIENT_ORGS.length)]!,
    glue: p(" the counterparty on that mandate is ", "."),
  },
];

export const CONFUSABLE_FAMILIES: readonly Family[] = [
  {
    id: "pan-shaped-invalid",
    type: `${NEG_PREFIX}pan-shaped-invalid-holder`,
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintInvalidPan,
    glue: p(" the code stamped on the form is ", " which is not a taxpayer number."),
  },
  {
    id: "aadhaar-shaped-invalid",
    type: `${NEG_PREFIX}aadhaar-shaped-bad-verhoeff`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintBadAadhaar,
    glue: p(" the placeholder we use in fixtures is ", " which no portal accepts."),
  },
  {
    id: "email-address",
    type: `${NEG_PREFIX}email-address`,
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: (rng) => mintEmail(rng, "sablewood.example.com"),
    glue: p(" you can reach the coordinator at ", " if that helps."),
  },
  {
    id: "swift-bic",
    type: `${NEG_PREFIX}swift-bic`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintBic,
    glue: p(" the swift code printed on the advice is ", "."),
  },
  {
    id: "micr-code",
    type: `${NEG_PREFIX}micr-code`,
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintMicr,
    glue: p(" the cheque carried ", " along the bottom edge."),
  },
  {
    id: "ticket-id",
    type: `${NEG_PREFIX}ticket-id`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintTicketId,
    glue: p(" the incident raised for this was ", "."),
  },
  {
    id: "git-sha",
    type: `${NEG_PREFIX}git-commit-sha`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintGitSha,
    glue: p(" the change landed in commit ", " last thursday."),
  },
  {
    id: "non-client-org",
    type: `${NEG_PREFIX}non-client-org`,
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "vendor",
    mint: (rng) => NON_CLIENT_ORGS[Math.floor(rng() * NON_CLIENT_ORGS.length)]!,
    glue: p(" our stationery supplier ", " sent the wrong boxes again."),
  },
];

export const ALL_FAMILIES: readonly Family[] = [...POSITIVE_FAMILIES, ...CONFUSABLE_FAMILIES];
