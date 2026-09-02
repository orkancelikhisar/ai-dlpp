import type { Family } from "./families.js";
import { NEG_PREFIX } from "./labels.js";
import {
  CLIENT_ORGS,
  FIRM,
  mintAadhaar,
  mintCustomerId,
  mintIfsc,
  mintPan,
  mintUpiVpa,
} from "./universe.js";
import {
  COMPETITOR_ORGS,
  DOC_EXAMPLE_AWS_KEY,
  DUAL_ROLE_ORGS,
  LISTED_COMPANY_ORGS,
  PERSON_NAMES,
  PRODUCT_NAMES,
  REDACTED_CREDENTIAL_PLACEHOLDER,
  mintBareAccountDigits,
  mintBase64Fragment,
  mintClientSecret,
  mintCsrPem,
  mintDashedAadhaar,
  mintEmployeeId,
  mintGhpToken,
  mintGstin,
  mintHsmKeyLabel,
  mintInternalUrl,
  mintJdbcUrl,
  mintMongoUrl,
  mintOpensshPem,
  mintPubkeyFingerprint,
  mintUuid,
} from "./universe.candidate.js";

/**
 * Wave-2 injection templates: the near-miss pairs.
 *
 * ## What this wave is for, in one measurement
 *
 * Run 03: `baselineB-Qwen3.5-2B` returned a finding at exactly the gold span --
 * byte-identical to a span two annotators had blind-labelled -- and typed it
 * `in-pan` when the sixteen characters were an organisation name. Span
 * extraction is not the binding constraint; CLASSIFICATION is. A corpus that
 * adds more spans to find cannot see that failure, so every family below exists
 * as one half of a pair whose two halves LOOK alike and are LABELLED
 * differently. The pairs, and what each one isolates:
 *
 * | pair | held constant | what has to be read |
 * |---|---|---|
 * | `dual-role-org-mandate` / `neg:dual-role-org-vendor` | the organisation NAME -- both draw from `DUAL_ROLE_ORGS` | the role clause around it |
 * | `client-org-message-scope` / `neg:dual-role-org-vendor-message-scope` | the whole shape: name, then a fenced block, then a role sentence about "they" | the two prose sentences JOINTLY -- neither one alone says it |
 * | `client-name` families / `neg:person-name`, `neg:product-name` | Title-Case-in-prose orthography | organisation vs person vs software |
 * | `client-name` families / `neg:competitor-org`, `neg:listed-company-in-news`, `neg:own-employer-org` | "a named organisation in a financial sentence" | which side of the Firm's relationship it is on -- p-fin §3 reaches clients, prospects and counterparties, and nothing else |
 * | `pan-under-aadhaar-context` / `aadhaar-under-pan-context` | the surrounding words | the VALUE, against words that point at the other type |
 * | `in-pan` / `neg:gstin` | a structurally valid PAN, character for character | whether ten characters embedded in fifteen are still a PAN |
 * | `api-credential` families / `neg:uuid`, `neg:base64-image-fragment`, `neg:git-commit-sha` | "a long opaque token from a terminal" | secret vs identifier vs payload |
 * | `db-connection-string` families / `neg:internal-http-url` | "a URL naming an internal host" | driver scheme vs https |
 * | `private-key-material` families / `neg:csr-pem-block`, `neg:public-key-fingerprint` | five dashes and a base64 body | the header word, and private vs public |
 * | `api-credential` families / `neg:redacted-credential-placeholder`, `neg:tutorial-api-key` | the credential PREFIX, which the tier-0 rule matches | whether the string is a credential at all |
 *
 * ## The shared organisation pool is the point of the first two rows
 *
 * Wave 1 draws client names from `CLIENT_ORGS` and non-client names from
 * `NON_CLIENT_ORGS`, and the two pools differ in flavour -- "Kestrel Ironworks"
 * against "Cobblestone Print Works". An arm can answer that pair from the name
 * alone. Every wave-2 organisation family draws from `DUAL_ROLE_ORGS`, one
 * pool, so "Marrowfield Group" appears as a client in one item and as a
 * facilities supplier in another and the name carries no signal at all.
 *
 * ## What is NOT claimed here
 *
 * `constructedRole` records the clause the generator WROTE. It is not a tier-2
 * predicate label and no `pred:` gold is emitted from it: predicate gold in this
 * repository is blind-labelled by two annotators and adjudicated (see
 * `corpora/fixtures/smoke.gold-tier2.jsonl`), and one author's call dropped into
 * that pipeline would be a provenance lie. The message-scope families in
 * particular are built to be LABELLING-READY for that round -- they are the
 * items where `scope: "message"` has something to bite on -- and not to be
 * labelled here.
 */

const p = (prefix: string, suffix: string) => () => ({ prefix, suffix });

function pickFrom(rng: () => number, xs: readonly string[]): string {
  return xs[Math.floor(rng() * xs.length)]!;
}

/**
 * The block a message-scope family writes BETWEEN the organisation name and the
 * sentence that establishes the relationship.
 *
 * A fenced block, not a blank line, and the difference is mechanical.
 * `segmentText` splits on fences and on kv/prose line runs, never on sentence
 * boundaries, so two sentences separated by a blank line stay inside ONE prose
 * segment and a segment-scoped judge would still see both. A fence forces three
 * segments -- prose, code, prose -- with the name in the first and the role in
 * the third. The predicate p-fin declares is `scope: "message"`, and this is
 * the construction that makes the difference between message and segment scope
 * observable rather than asserted.
 */
const scopeSplit = (kvLines: string, relationship: string) => () => ({
  prefix: " the agreement i am working from names ",
  suffix: `.\n\n\`\`\`\n${kvLines}\n\`\`\`\n\n${relationship}`,
});

export const CANDIDATE_POSITIVE_FAMILIES: readonly Family[] = [
  {
    // The context words point at in-aadhaar and the value is a PAN. A
    // classifier keying on nearby words rather than on the string gets this
    // wrong; tier 0 gets it right, because a regex cannot read.
    id: "pan-under-aadhaar-context",
    type: "in-pan",
    surface: "prose",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "none",
    mint: mintPan,
    glue: p(" the e-kyc file has no aadhaar on it at all, only the tax number ", ", which is what the branch keyed in."),
  },
  {
    // The mirror image. "permanent account number" sits 40-odd characters from
    // a twelve-digit value, which is deliberately outside
    // `labelled-account-number`'s twelve-non-digit-character reach -- close
    // enough to mislead a reader, far enough that no second tier-0 rule fires
    // across the gold span.
    id: "aadhaar-under-pan-context",
    type: "in-aadhaar",
    surface: "prose",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintAadhaar,
    glue: p(
      " the income tax page wanted a permanent account number and the field we actually stored is ",
      ", which is why every lookup fails.",
    ),
  },
  {
    id: "aadhaar-dashed",
    type: "in-aadhaar",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: mintDashedAadhaar,
    glue: p(" she sent it over as ", " and i typed it in exactly like that."),
  },
  {
    id: "kyc-case-id",
    type: "internal-customer-id",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: (rng) => mintCustomerId(rng, "KYC", "/"),
    glue: p(" the case raised at onboarding is ", " and it is still open."),
  },
  {
    // Short key on purpose: the whole kv line stays under the stock entropy
    // rule's 24-character floor, so this item carries one finding (the IFSC
    // regex) rather than an entropy finding stacked on top of it.
    id: "ifsc-kv",
    type: "bank-account-identifier",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintIfsc,
    glue: p("\nifsc: ", "\n"),
  },
  {
    // A fictional UPI handle. Wave 1's `upi-vpa` family mints against "okaxis",
    // which is the live UPI handle of a real bank, so every VPA it produces is
    // a well-formed address at a real payment provider. `families.ts` is still
    // not edited -- changing it would change the committed v1 corpus -- so the
    // fix is an exclusion instead: `build-adjudicated.ts` drops `upi-vpa` from
    // the catalogue it passes and this family covers the same entityType and
    // surface at an invented handle. `injection-p-fin-adjudicated-v1` contains
    // no address in a real payment namespace; the test asserts the file has no
    // "@okaxis" in it.
    id: "upi-vpa-fictional-handle",
    type: "bank-account-identifier",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: (rng) => mintUpiVpa(rng, "okbluecrest"),
    glue: p(" the collect request went out to ", " and it has still not been accepted."),
  },
  {
    // Invisible to tier 0 by construction; see `mintBareAccountDigits`. The
    // glue must never put "account", "acct" or "a/c" within twelve non-digit
    // characters of the value, or `labelled-account-number` fires with a span
    // wider than the gold one and this family stops being what it is for.
    id: "bare-account-digits",
    type: "bank-account-identifier",
    surface: "prose",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintBareAccountDigits,
    glue: p(" the credit was routed to the beneficiary number ", " at the same branch."),
  },
  {
    id: "ghp-token-prose",
    type: "api-credential",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintGhpToken,
    glue: p(" the runner is still authenticating with ", " and nobody has revoked it."),
  },
  {
    // The only positive family in the corpus that reaches tier 0 through the
    // ENTROPY rule rather than a prefix regex. Without it `api-credential-entropy`
    // is only ever observed firing on private-key bodies, i.e. only as a false
    // positive, and a rule whose true positives are never exercised is a rule
    // nobody has measured.
    id: "client-secret-entropy-kv",
    type: "api-credential",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintClientSecret,
    glue: p("\nclient_secret: ", "\n"),
  },
  {
    id: "mongo-url-kv",
    type: "db-connection-string",
    surface: "kv-line",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintMongoUrl,
    glue: p("\ndsn: ", "\n"),
  },
  {
    // The suffix begins with a SPACE, not a full stop. `db-connection-string-format`
    // matches `[^\s"']+` after the scheme, so a suffix starting with punctuation
    // would be swallowed into the match and every finding would be one character
    // wider than the gold span -- an exact-match miss manufactured by the glue.
    id: "jdbc-url-prose",
    type: "db-connection-string",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintJdbcUrl,
    glue: p(" the datasource on the archive box is ", " and the schema is read only."),
  },
  {
    id: "openssh-pem-fence",
    type: "private-key-material",
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintOpensshPem,
    glue: p("\n\n```\n", "\n```\n\n"),
  },
  {
    // p-fin §4.3 and the `private-key-material` nlDefinition both name HSM key
    // labels; no tier-0 rule looks for one. Like `client-name`, this family
    // reads 0 recall on a tier-0-only arm by construction, which is what makes
    // the ladder curve say something.
    id: "hsm-key-label",
    type: "private-key-material",
    surface: "labelled",
    difficulty: "paraphrased",
    register: "technical",
    constructedRole: "none",
    mint: mintHsmKeyLabel,
    glue: p(" the label on the key inside the module is ", " if that narrows it down."),
  },
  {
    // p-fin §3.3: prospective clients are treated exactly as existing clients.
    id: "dual-role-org-prospect",
    type: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "client",
    mint: (rng) => pickFrom(rng, DUAL_ROLE_ORGS),
    glue: p(" we are pitching ", " next month and the deck is nowhere near ready."),
  },
  {
    // p-fin §3.3 again: a party under an NDA.
    id: "dual-role-org-nda",
    type: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "counterparty",
    mint: (rng) => pickFrom(rng, DUAL_ROLE_ORGS),
    glue: p(" the party we signed the confidentiality agreement with is ", " and the term runs to december."),
  },
  {
    id: "dual-role-org-mandate",
    type: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "client",
    mint: (rng) => pickFrom(rng, DUAL_ROLE_ORGS),
    glue: p(" the mandate we hold for ", " renews in march and nobody has picked it up."),
  },
  {
    id: "dual-role-org-trade-counterparty",
    type: "client-name",
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "counterparty",
    mint: (rng) => pickFrom(rng, DUAL_ROLE_ORGS),
    glue: p(" we are on the other side of that trade from ", ", so the netting question actually matters."),
  },
  {
    // The message-scope positive. No single sentence says "X is our client":
    // the first names the organisation, a fenced block breaks the segment, and
    // the third says "they have been on our books". Read one segment at a time
    // this is nothing; read as a message it is p-fin §3.1 and §3.4 exactly.
    id: "client-org-message-scope",
    type: "client-name",
    surface: "prose",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "client",
    mint: (rng) => pickFrom(rng, DUAL_ROLE_ORGS),
    glue: scopeSplit(
      "fee_basis: retainer\nreview: april",
      "they have been on our books since that was signed, and the retainer line above is the one we agreed with them at the time.",
    ),
  },
];

export const CANDIDATE_CONFUSABLE_FAMILIES: readonly Family[] = [
  {
    // A COLLEAGUE, never a customer. p-fin §1.2 scopes customer data to a
    // person or entity holding an account with the Firm, and the compiled IR
    // has no personal-name entityType at all, so both readings agree that
    // p-fin demands nothing here. Put the same name in a customer clause and
    // that stops being true, which is why the clause is fixed.
    id: "person-name",
    type: `${NEG_PREFIX}person-name`,
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: (rng) => pickFrom(rng, PERSON_NAMES),
    glue: p(" the note was drafted by ", " on the operations desk."),
  },
  {
    id: "product-name",
    type: `${NEG_PREFIX}product-name`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: (rng) => pickFrom(rng, PRODUCT_NAMES),
    glue: p(" the scheduler we run everything through is called ", " and it has no retry setting at all."),
  },
  {
    id: "competitor-org",
    type: `${NEG_PREFIX}competitor-org`,
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: (rng) => pickFrom(rng, COMPETITOR_ORGS),
    glue: p(" our nearest competitor ", " has started quoting a lower fee for the same work."),
  },
  {
    id: "listed-company-in-news",
    type: `${NEG_PREFIX}listed-company-in-news`,
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: (rng) => pickFrom(rng, LISTED_COMPANY_ORGS),
    glue: p(" the trade press has been writing about ", " all week and none of it touches us."),
  },
  {
    // The Firm is never a client of itself. This is the organisation an arm is
    // most likely to over-flag, because it is the one that appears in the
    // policy document the arm may also be shown.
    id: "own-employer-org",
    type: `${NEG_PREFIX}own-employer-org`,
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: () => FIRM,
    glue: p(" i work on the operations desk at ", " and the intranet has been down since monday."),
  },
  {
    // Same pool as the client families. Same span, opposite answer.
    id: "dual-role-org-vendor",
    type: `${NEG_PREFIX}dual-role-org-vendor`,
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "vendor",
    mint: (rng) => pickFrom(rng, DUAL_ROLE_ORGS),
    glue: p(" the facilities contract is with ", " and they have not invoiced us since june."),
  },
  {
    // The message-scope negative, built to the same template as
    // `client-org-message-scope` down to the fenced block, so the pair differs
    // in the last sentence and in nothing else.
    id: "dual-role-org-vendor-message-scope",
    type: `${NEG_PREFIX}dual-role-org-vendor-message-scope`,
    surface: "prose",
    difficulty: "paraphrased",
    register: "formal",
    constructedRole: "vendor",
    mint: (rng) => pickFrom(rng, DUAL_ROLE_ORGS),
    glue: scopeSplit(
      "rate_card: annual\nreview: april",
      "they have been supplying our print and postage since that was signed, and the rate card above is the one we agreed with them at the time.",
    ),
  },
  {
    // Injected in PROSE deliberately. A UUID is 36 characters over a 17-symbol
    // alphabet and `-` is inside the entropy candidate alphabet, so in a kv or
    // code segment it would sit right on the stock 4.0 threshold and the family
    // would be measuring a threshold rather than a classifier.
    id: "uuid",
    type: `${NEG_PREFIX}uuid`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintUuid,
    glue: p(" the correlation id in the trace is ", " if you want to look it up."),
  },
  {
    // Injected in a FENCE deliberately, for the opposite reason: entropy rules
    // only scan code and kv segments, so in prose this would be a confusable
    // that confuses nothing. Here it is a measured false positive.
    id: "base64-image-fragment",
    type: `${NEG_PREFIX}base64-image-fragment`,
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintBase64Fragment,
    glue: p("\n\nthe paste came through as this, which is most of a screenshot:\n\n```\n", "\n```\n\n"),
  },
  {
    id: "internal-http-url",
    type: `${NEG_PREFIX}internal-http-url`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintInternalUrl,
    glue: p(" the runbook for this is at ", " and it is badly out of date."),
  },
  {
    id: "gstin",
    type: `${NEG_PREFIX}gstin`,
    surface: "prose",
    difficulty: "verbatim",
    register: "formal",
    constructedRole: "none",
    mint: mintGstin,
    glue: p(" the invoice carries the tax registration ", " which is the supplier's and not ours."),
  },
  {
    id: "employee-id",
    type: `${NEG_PREFIX}employee-id`,
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: mintEmployeeId,
    glue: p(" the roster lists him as ", " because that system does not use names."),
  },
  {
    // Matches `api-credential-prefix` and is a sentence about a key rather than
    // a key. p-fin's own counterExample list carries the short form, "sk-...".
    id: "redacted-credential-placeholder",
    type: `${NEG_PREFIX}redacted-credential-placeholder`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: () => REDACTED_CREDENTIAL_PLACEHOLDER,
    glue: p(" the runbook prints the key as ", " which is not a key at all."),
  },
  {
    // Spec 6.2's hard-negative triage case, and the one label in this file a
    // reader should treat as contested. See `DOC_EXAMPLE_AWS_KEY`.
    id: "tutorial-api-key",
    type: `${NEG_PREFIX}tutorial-api-key`,
    surface: "prose",
    difficulty: "verbatim",
    register: "casual",
    constructedRole: "none",
    mint: () => DOC_EXAMPLE_AWS_KEY,
    glue: p(" the tutorial tells you to paste ", " and then watch the call fail."),
  },
  {
    id: "csr-pem-block",
    type: `${NEG_PREFIX}csr-pem-block`,
    surface: "code-fence",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintCsrPem,
    glue: p("\n\n```\n", "\n```\n\n"),
  },
  {
    id: "public-key-fingerprint",
    type: `${NEG_PREFIX}public-key-fingerprint`,
    surface: "prose",
    difficulty: "verbatim",
    register: "technical",
    constructedRole: "none",
    mint: mintPubkeyFingerprint,
    glue: p(" the fingerprint it printed was ", " and i have nothing to compare it against."),
  },
];

export const CANDIDATE_FAMILIES: readonly Family[] = [
  ...CANDIDATE_POSITIVE_FAMILIES,
  ...CANDIDATE_CONFUSABLE_FAMILIES,
];

/**
 * The wave-1 client pool, re-exported so the candidate test can assert that no
 * wave-2 organisation name collides with a wave-1 one. A collision would break
 * the "occurs exactly as many times as injected" invariant the first time both
 * landed in one message.
 */
export const WAVE1_CLIENT_ORGS = CLIENT_ORGS;
