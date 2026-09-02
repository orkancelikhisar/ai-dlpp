import type { StageHit, Sweep } from "./certify.js";

/**
 * The blind double-adjudication round over the wave-2 carrier pool, recorded
 * verbatim, plus the admission rule the corpus builder reads out of it.
 *
 * ## What this is, and what it deliberately is NOT
 *
 * Spec 6.2's stage 3 is "frontier-model adjudication against the UNION of all
 * three policies. Certified clear only if clean under all three." What ran here
 * is two independent certifiers, working blind of each other and of the
 * generator, reading `policies/p-fin.md` and the wave-2 carrier file and
 * nothing else. One policy of three. So this is not stage 3, it does not make
 * `certifyCarrier` return `"certified-clear"`, and it is wired as a
 * SUPPLEMENTARY sweep -- `certify.ts`'s existing category for a sweep this
 * repository invented, which "can quarantine a carrier; it can never move a
 * carrier toward certified-clear". The manifest still reports
 * `frontier-adjudication` as an unrun stage with its blockers, because it is.
 *
 * Understating is the only safe direction here. A round that covered one third
 * of the policy union but was booked as stage 3 would flip `claim` to
 * "CERTIFIED" the moment stage 2 ever lands, on the strength of a p-fin-only
 * reading -- the same intent-as-fact defect the standing conventions record.
 *
 * ## The admission rule, and why `borderline` refuses
 *
 * A carrier is admitted only when BOTH certifiers returned `clear: true` and
 * NEITHER flagged `borderline`. The asymmetry is the one both certifiers
 * independently argued for: a wrong clearance puts an invisible hole in the
 * corpus invariant (a finding on an uninjected instance is scored a false
 * positive when it is actually correct), while a wrong refusal costs one
 * carrier. `borderline` is a certifier saying "an adjudicator could reasonably
 * disagree", and a carrier that two readers both found arguable cannot support
 * the sentence "any finding outside the injected spans is a TRUE false
 * positive".
 *
 * ## Carriers with no verdict are refused, not waved through
 *
 * `adjudicationSweep` quarantines a carrier it has no verdict for. The 25
 * wave-1 carriers (`carriers.ts`) are in that position: the round was scoped to
 * wave 2 and both certifiers stayed out of the wave-1 carrier text on purpose,
 * so nothing has adjudicated them. Passing them through would be the exact
 * hole this round exists to close -- stage 1 is blind to organisation names and
 * to relationship disclosures, which is what an adjudicator reads for.
 *
 * ## Blindness: what was verified, and the one breach that was disclosed
 *
 * See `ADJUDICATION_ROUND.blindness`. The short version, because it bounds what
 * the agreement number below is worth: both certifiers read the target file,
 * and that file's own header comment states which carriers were written to fail
 * and which one is contestable. Neither certifier could avoid it. So the round
 * is blind of the generator, of the labels and of each other, and is NOT blind
 * of the authoring intent.
 *
 * ## Two channels, and only one of them used to be audited
 *
 * The blindness record checked what the certifiers READ -- their `filesRead`
 * lists, item by item, against the files that would have given the answer away.
 * It said nothing about what they were TOLD. An annotator's brief is the other
 * route by which the answer reaches them, and it is the route that leaves no
 * trace in a file list: a brief saying "four of these were written to fail"
 * would produce a perfectly clean `filesRead` and a worthless round.
 *
 * `blindness.channels` names both channels and states, per channel, whether it
 * was audited and against what evidence. The told channel comes out UNAUDITED,
 * and it is recorded as unaudited rather than assumed clean, because the
 * verbatim briefs were not retained -- `blindness.rulesGiven` is this round's
 * own summary of the instruction, not a transcript of it. `blindnessAudit()`
 * returns the same structure for a machine, and `corpus-adjudication.test.ts`
 * fails if a channel is listed with neither evidence nor a declared gap.
 */

export type Confidence = "clear" | "borderline";

/** One certifier's call on one carrier, transcribed from their returned verdict. */
export interface CertifierVerdict {
  /** Their `clear` field: is this carrier free of p-fin-confidential content? */
  readonly clear: boolean;
  /** Their `confidence` field. `borderline` means "reasonable people differ". */
  readonly confidence: Confidence;
  /** Verbatim. Never summarised: the reasoning is the provenance. */
  readonly rationale: string;
}

export interface CarrierVerdicts {
  readonly carrierId: string;
  readonly A: CertifierVerdict;
  readonly B: CertifierVerdict;
}

export const ADJUDICATION_ROUND = {
  id: "wave2-blind-double-adjudication-p-fin",
  policy: "policies/p-fin.md",
  scope:
    "the 32 wave-2 carriers in apps/eval/src/corpus/carriers.candidate.ts " +
    "(16 CANDIDATE_ORDINARY, 12 CANDIDATE_HARD_NEGATIVE, 4 CANDIDATE_DIRTY)",
  certifiers: ["A", "B"] as const,
  independence:
    "Each certifier worked without sight of the other's verdicts, notes or existence of a second " +
    "round, and returned a structured verdict per carrier plus a free-text method note.",
  specStage: {
    stage: "frontier-adjudication",
    satisfied: false,
    why:
      "spec 6.2 stage 3 requires adjudication against the union of P-FIN, P-MED and P-CORP and " +
      "certifies clear only if clean under all three. Both certifiers read policies/p-fin.md " +
      "alone, so one policy of three was adjudicated. This round is recorded as a supplementary " +
      "sweep and the stage stays unrun.",
  },
  blindness: {
    rulesGiven:
      "read policies/p-fin.md and the wave-2 carrier file only; no families/inject/certify/labels " +
      "modules, no compiled IR, no gold fixtures, nothing under runs/, no git log, no plan",
    /**
     * The channels by which the answer can reach an annotator, each with what
     * was actually checked. `audited: false` is a finding, not a formatting
     * choice: it means this round cannot say the channel was clean.
     */
    channels: [
      {
        channel: "read",
        what: "the files each certifier opened, from their returned filesRead list",
        audited: true,
        evidence:
          "each certifier's returned filesRead list was compared item by item against the modules " +
          "that carry the answer; the results are in blindness.verified below, including the one " +
          "range Certifier A disclosed and the line numbers that range was checked against",
        gap:
          "a filesRead list is self-reported. It bounds what each certifier SAID they opened, and " +
          "this repository has no independent record of file access to check it against.",
      },
      {
        channel: "told",
        what: "the brief each certifier was given: task, scope, decision rule, and any framing in it",
        audited: false,
        evidence:
          "rulesGiven above is this round's summary of the instruction, and both returned method " +
          "notes are consistent with it -- A's opens 'Scope: the 32 wave-2 carriers', B's 'Read " +
          "policies/p-fin.md and the wave-2 carrier file only'. Consistency with a summary is not " +
          "an audit of the text.",
        gap:
          "NO VERBATIM BRIEF WAS RETAINED. Neither certifier's prompt is committed anywhere in " +
          "this repository, so nothing here can rule out answer-bearing framing inside it. The " +
          "concrete instance: both certifiers demonstrably knew the stratum structure -- A's note " +
          "counts '16 ordinary carriers' and '12 hard negatives', B's says the file 'told B that " +
          "the d0* carriers were written to fail' -- and the round attributes that to the target " +
          "file's own header (see breaches, STRUCTURAL). " +
          "The brief is a second route to the same knowledge and it cannot be excluded, because " +
          "it cannot be read. Any future round must retain the briefs verbatim for this channel " +
          "to be auditable at all.",
      },
    ],
    verified: [
      "Neither filesRead list contains families.candidate.ts, families.ts, inject.ts, certify.ts, " +
        "labels.ts, universe*.ts, generate.ts, policies/compiled/p-fin.ir.json, " +
        "corpora/fixtures/smoke.gold-tier2.jsonl, or anything under runs/.",
      "Certifier A additionally read carriers.ts lines 1-40 and 55-85 and disclosed it. VERIFIED " +
        "against the file: the provenance docblock ends at line 43 and CLEAN_CARRIERS begins at " +
        "line 101, so both ranges are docblock and type declarations and neither contains wave-1 " +
        "carrier text. The docblock does disclose that the wave-1 carriers were authored to pass " +
        "certify.ts's sweeps and that DIRTY_CARRIERS exist to fail them.",
    ],
    breaches: [
      "DISCLOSED BY B: B ran `git log --oneline -1` on the carrier file and received one subject " +
        "line, 'feat(eval): wave-2 corpus inputs, built as near-miss pairs'. The rules said no git " +
        "log. No verdict-bearing content was returned, but the breach is on the record because B " +
        "put it there.",
      "STRUCTURAL, AFFECTS BOTH: the target file's own header docblock (lines 1-70 of " +
        "carriers.candidate.ts, above the carriers) states that the d0* carriers were written to " +
        "FAIL certification, that hn01 is the contestable one, and gives the entropy sweep's " +
        "thresholds. B disclosed reading it. A did not list it separately but demonstrably read it " +
        "-- A's d07 rationale argues against 'the file's own framing' and against what 'the intent " +
        "was'. Neither certifier could have reached the carriers without passing it. The round is " +
        "therefore blind of the generator, the labels and each other, and NOT blind of the " +
        "authoring intent, which is a plausible cause of at least the d0*/hn01 agreement below.",
    ],
  },
  /**
   * Both certifiers' prose summaries say "28 clear, 4 not clear" while both
   * their own structured verdict arrays say 29 clear, 3 not clear -- the
   * difference is d07, which both marked `clear: true` and argued at length was
   * clear ("Cleared against the policy document", "the policy verdict is
   * clear"), while both prose lines counted the whole d0* stratum as not clear.
   *
   * The structured field is what a machine reads and what each rationale
   * supports, so the verdict table below is transcribed from it. Recorded
   * rather than silently reconciled: a reader who quotes either note will get a
   * number that disagrees with this file, and should know why.
   */
  discrepancies: [
    "Certifier A's note states 'Result: 28 clear, 4 not clear'; A's own verdict array is 29 clear, " +
      "3 not clear. The difference is d07-entropy-fence, which A recorded clear: true.",
    "Certifier B's note states '28 of 32 clear ... 4 not clear (the d0* pool)'; B's own verdict " +
      "array is 29 clear, 3 not clear. The difference is d07-entropy-fence again, which B also " +
      "recorded clear: true and argued at length was clear on the document.",
  ],
  notes: {
    A:
      "Certifier A, blind. Scope: the 32 wave-2 carriers. Decision rule where torn: the two errors " +
      "are not symmetric. A wrong 'clear' silently punches a hole in the corpus invariant and is " +
      "invisible downstream; a wrong 'not clear' costs one carrier. So on a genuine tie A withholds " +
      "clearance. Result: all 16 ordinary carriers clear and none close; all 12 hard negatives " +
      "clear, each naming a regulated category without carrying an instance of it. Three " +
      "borderlines called honestly: hn01 (cleared, but the weakest clearance in the set -- if " +
      "§1.2's 'can be combined with other information to identify' is read as operative through " +
      "§5.1 rather than as a definition feeding §2-§4, it flips); d06 (withheld -- §2.3's operative " +
      "prohibition is on account NUMBERS and an IFSC branch code is publicly published, but §2.5 " +
      "refuses relief for partial values); d07 (cleared -- a checksum is in none of §4's enumerated " +
      "categories, and if the intent was that d07 fail on policy grounds, that intent does not " +
      "survive the document). Accuracy note recorded rather than buried: QRSDE7741K is PAN-shaped " +
      "but its fourth character 'D' is not a valid PAN holder-type code, so it is not a " +
      "structurally valid PAN; clearance was still withheld under §2.5.",
    B:
      "Certifier B, blind round. Read policies/p-fin.md and the wave-2 carrier file only. Two " +
      "disclosures, because a blind round is only worth what it admits: (1) B ran one `git log " +
      "--oneline -1` on the carrier file, against the rules, and received one subject line; (2) the " +
      "carrier file's own header comment is not blind -- it told B that the d0* carriers were " +
      "written to fail, that hn01 is contestable, and the entropy sweep's thresholds. Tie-breaker " +
      "used on the arguable carriers: if a detector fired on this string, would that be a TRUE " +
      "false positive under p-fin as written? If yes the carrier is clear; if the policy names the " +
      "category outright, certifying clear would mislabel a defensible hit as over-blocking. That " +
      "splits d06 (IFSC named outright in §2.3 -> not clear) from d07 (§4's list is closed and a " +
      "checksum is in none of it -> clear on the document, but B states explicitly this is not a " +
      "recommendation to inject: the token is secret-shaped and should stay in the dirty pool). " +
      "hn01 clear, borderline, and named as the one carrier in the clean pool B would want a " +
      "second opinion on before the invariant leans on it.",
  },
} as const;

/**
 * Every verdict from both certifiers, transcribed from their returned
 * structured fields. Rationales are verbatim; nothing here is summarised,
 * re-worded or reconciled. Where the two disagree the disagreement stands in
 * the table and `admits` refuses the carrier.
 */
export const CARRIER_VERDICTS: readonly CarrierVerdicts[] = [
  {
    carrierId: "o01",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "CSV reader throwing an index error on a trailing blank line. No organisation name, no identifier value, no credential, no connection string. Nothing in any p-fin category.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "CSV reader throwing an index error on a trailing blank line. No identifier, no name, no credential, no connection string. Nothing in §2, §3 or §4.",
    },
  },
  {
    carrierId: "o02",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Splitting household bills with a flatmate. The flatmate is not named and no account, amount or identifier appears. §1.2 customer data concerns persons holding an account with the Firm; nobody here is identified at all.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Flatmates splitting bills after different move-in dates. Domestic money talk with no account number, no institution named, no value of any kind. §2 and §3 untouched.",
    },
  },
  {
    carrierId: "o03",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Drafting a complaint about an unpaid refund. 'The company' is referred to only by that word, so §3.1's obligation to pseudonymize a counterparty name has nothing to bite on. No transaction identifiers.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Drafting a complaint about a thrice-promised refund. The company is referred to only as 'the company' and never named, so §3 is not engaged; no reference number, no amount, no account.",
    },
  },
  {
    carrierId: "o04",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "CSS grid collapsing at a breakpoint with the footer above the sidebar. Purely technical, no names, values or secrets.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "CSS grid collapsing to one column with the footer above the sidebar. Pure front-end question, no data of any kind.",
    },
  },
  {
    carrierId: "o05",
    A: {
      clear: true,
      confidence: "clear",
      rationale: "Bread coming out dense despite rising in the bowl. No policy-regulated content of any kind.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale: "Bread coming out dense despite rising in the bowl. Nothing remotely within p-fin's scope.",
    },
  },
  {
    carrierId: "o06",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "How much detail belongs in a handover note. Mentions that some content concerns how particular colleagues prefer to be approached, but no colleague is named and colleagues are not customer data under §1.2, which is scoped to account holders.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "How much detail belongs in a handover note when the successor is unnamed. Mentions that some content concerns how colleagues prefer to be approached, but names no colleague and no customer; §1.2 covers account holders, not staff, and no identifier appears.",
    },
  },
  {
    carrierId: "o07",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A nightly staging-to-main merge that doubled in duration. Tables are referred to generically ('a staging table', 'the main one') with no host, schema, database or connection string, so §4.2's topology concern is not engaged. Row volumes are not customer data.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Nightly staging-to-main merge that doubled in runtime with no change to the job. §4.2's concern is host and schema names disclosing internal topology; this carrier gives neither -- 'a staging table' and 'the main one' are generic. No connection string, no credential.",
    },
  },
  {
    carrierId: "o08",
    A: { clear: true, confidence: "clear", rationale: "Choosing a talk topic for a mixed audience. Nothing regulated." },
    B: {
      clear: true,
      confidence: "clear",
      rationale: "Choosing a talk topic for a mixed audience in six weeks. No event named, no person named, nothing in scope.",
    },
  },
  {
    carrierId: "o09",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Contains a code fence, but it is a three-line generic filter loop over `rows` with no host, credential, endpoint or identifier in it. Nothing under §4, and nothing under §2.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Three-line loop over rows appending to a list, asking for a more idiomatic form. The fenced code contains only generic identifiers (rows, r.ok, out.add) -- no key, token, secret, host or schema. §4 not engaged.",
    },
  },
  {
    carrierId: "o10",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Cutting a policy document from fourteen pages to four. Discusses the editing problem in the abstract and reproduces no clause, obligation text or internal content. No p-fin category applies to the mere existence of an internal document.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Cutting a fourteen-page policy document to four without dropping an obligation. Discusses an internal document in the abstract; quotes none of it and names no client, customer or system.",
    },
  },
  {
    carrierId: "o11",
    A: { clear: true, confidence: "clear", rationale: "Planning four unstructured days off in November. Nothing regulated." },
    B: {
      clear: true,
      confidence: "clear",
      rationale: "Planning four unstructured days off in November. No location named, no booking reference, nothing in scope.",
    },
  },
  {
    carrierId: "o12",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Contains a code fence, but it is two lines of pseudo-config (`setup: db.fresh()` / `teardown: db.drop()`) with no host, schema, username or password. Not a database connection string under §4.2 and not a credential under §4.1.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Eleven-minute test suite dominated by one file that spins a fresh database per case. The fence holds pseudo-code (db.fresh() / db.drop()) with no host, port, schema, DSN or credential, so §4.2 is not engaged.",
    },
  },
  {
    carrierId: "o13",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A colleague describing shared work in a technically accurate but misleading way. The colleague is not named and the work is not described. Nothing regulated.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale: "A colleague describing shared work misleadingly in meetings. Interpersonal, no names, no customer data.",
    },
  },
  {
    carrierId: "o14",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Framing a paper on why a manual check outlived the automation around it, with the candid admission that nobody trusted the automation. Internally awkward but not customer data, not a client or counterparty name, and not a credential. p-fin does not regulate internal process candour.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Framing a paper on why a manual check outlived the automation around it. Candid about internal distrust of the automation, which is awkward rather than confidential -- it is not a §2 identifier, a §3 name or §4 secret material.",
    },
  },
  {
    carrierId: "o15",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A daily step change in a latency graph, possibly coinciding with a cache warm job on an uncalled service. Services are described by role only, with no hostnames, endpoints or identifiers. Nothing under §4.2.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Daily step change in a latency graph coinciding with a cache warm job. The other service is described as 'a service we do not call' and never named; no host, no schema, no identifier.",
    },
  },
  {
    carrierId: "o16",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Explaining the author's own job to a child, where the job involves reconciliation. Discloses a job function, not customer data, no client name, no identifier.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Explaining a reconciliation job to an eleven-year-old for a school report. Names the writer's own function, not any customer, account or client.",
    },
  },
  {
    carrierId: "hn01",
    A: {
      clear: true,
      confidence: "borderline",
      rationale:
        "A short-notice call from 'our biggest client' with a fee arrangement the author will not discuss until the manager returns. No organisation is named, and §3.1/§3.2 impose an obligation on client organisation NAMES while §3.4 frames the disclosure as the pairing of a client NAME with an operational question -- none of which is triggered without a name. Borderline, and I flag it as the weakest clearance in the set: §1.2 defines customer data to include what 'can be combined with other information to identify' an entity, and the superlative 'our biggest client', combined with knowledge of the Firm, is precisely such a combinable handle. If §1.2 is read as operative (via §5.1's bar on sending customer data to non-enterprise services) rather than as a definition feeding §2-§4, this carrier is not clear. A detector firing on this is arguable rather than plainly wrong.",
    },
    B: {
      clear: true,
      confidence: "borderline",
      rationale:
        "'Our biggest client' asks for a short-notice call and will press on the fee arrangement. No organisation is named, and §3.1/§3.3 operate on client and counterparty NAMES -- so on the operative clause the correct finding is nothing. Genuinely arguable, and flagged: §1.2 extends to information that 'can be combined with other information to identify' an account holder, and 'our biggest client' is a distinguishing descriptor rather than a generic one, with a fee arrangement attached to it. I read §1.2 as a definition that §3 and §5.1 then act on, not as a standalone prohibition, so I land clear -- but this is the one carrier in the clean pool I would want a second certifier to confirm before the invariant leans on it.",
    },
  },
  {
    carrierId: "hn02",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A counterparty on a mandate went quiet after requesting an extension and not signing. §3.3 treats deal counterparties exactly as client names, but no name, no mandate identifier and no deal terms appear -- 'this mandate' and 'they' are the only references. The §3 obligation has nothing to substitute.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "A counterparty went quiet after requesting a two-week extension and not signing. §3.3 treats deal counterparties exactly as clients, but the clause protects the NAME and none is given -- 'the counterparty on this mandate' carries no distinguishing attribute, no sector, no size, no jurisdiction. Nothing in §2 or §4.",
    },
  },
  {
    carrierId: "hn03",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Discusses a token that was pasted into a shared channel and rotated within the hour, and whether it had to be reported. §4.1 forbids including keys, tokens and secrets in a prompt; no token value appears anywhere in the text. The §4.4 rotate-and-report question is discussed in the abstract, and p-fin does not classify incident discussion itself as confidential.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "A token was pasted into a shared channel and rotated within the hour; the argument is whether it was reportable. The token VALUE never appears, and §4.1 forbids including a credential in a prompt, not discussing one. §4.4 is the rule being debated, not a rule this text breaks. Clear.",
    },
  },
  {
    carrierId: "hn04",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A ten-minute production database outage with an ambiguous failover. No connection string, no host, no schema, no credential -- §4.2's rationale (host and schema names disclosing internal topology) is not reached. No customer data.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Production database unreachable for ten minutes with no incident-channel trace. §4.2's rationale is that host and schema names disclose internal topology, and this carrier names neither -- no DSN, no hostname, no schema, no credential. An outage narrative is not itself a p-fin category.",
    },
  },
  {
    carrierId: "hn05",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "Asks where the trusted copy of a host key fingerprint normally comes from. §4.3 covers private keys, certificate material and HSM key labels; a host key fingerprint is public key material and, decisively, no fingerprint value is reproduced -- the author only says it is visible on screen.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Setting up a new machine and not knowing what to compare a host key fingerprint against. The fingerprint is described as visible on screen and is never transcribed, so no §4.3 certificate or key material is present. Names the concept, carries no instance.",
    },
  },
  {
    carrierId: "hn06",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A PAN card application where the portal rejects the father's date of birth. §2.1 forbids PAN values; no PAN appears. The date of birth is referred to as a rejected field and its value is never given. The father is not named and is not identified as a Firm account holder.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Helping a father with a PAN card application that rejects his date of birth. §2.1 forbids PAN VALUES; no PAN appears, and the date of birth is described as rejected without ever being stated. A relative is also not an account holder of the Firm under §1.2. Topic word only.",
    },
  },
  {
    carrierId: "hn07",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "The Aadhaar helpline and an enrolment centre sending the mother back and forth over an address update. §2.2 forbids Aadhaar numbers; none appears, nor does the address value or the mother's name.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Mother sent back and forth between the Aadhaar helpline and a centre over an address update. §2.2 forbids Aadhaar numbers; none appears, and no address is given either. Topic word only.",
    },
  },
  {
    carrierId: "hn08",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "An unexplained statement credit from 'an account ending in the same four digits as my own'. §2.5 refuses relief for masked or truncated values, but here not even the four digits are stated -- the text discloses a coincidence of digits without disclosing any digit. No account number, IFSC string or UPI address is present.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "An unplaceable statement credit 'from an account ending in the same four digits as my own'. Built to bait §2.5's anti-masking rule, but §2.5 forecloses masking or truncating a value -- it does not reach a sentence that states no digits at all. The text contains zero numerals; a mechanical digit sweep over the file confirms it. Nothing in §2.3.",
    },
  },
  {
    carrierId: "hn09",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "An internal directory keyed by employee number, and a proposal to allow name search. §2.4 covers internal CUSTOMER identifiers -- customer reference numbers, CIF numbers, KYC case ids -- and an employee number is not one; independently, no number value appears. No employee is named.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "An internal directory keyed by employee number, and whether a name-search page is a privacy problem. §2.4's 'internal customer identifiers' are customer reference numbers, CIF numbers and KYC case ids -- employee numbers are staff identifiers, not customer ones, and in any case no number is given.",
    },
  },
  {
    carrierId: "hn10",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A cross-border supplier payment bounced with a SWIFT code complaint. §2.3 forbids account numbers and IFSC-qualified account strings; no code, account number or bank name is given, and the supplier and bank are both unnamed, so §3.1 has nothing to pseudonymize.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "A cross-border supplier payment bounced with a SWIFT code error. No SWIFT code, no account number, no IFSC string and no supplier name appear -- §2.3 and §3.1 both need a value or a name and neither is present.",
    },
  },
  {
    carrierId: "hn11",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "An unnamed cloud provider raised prices by a third on sixty days notice, and the team is absorbing it. Discloses a commercial posture, but §3 attaches to client and counterparty NAMES and the vendor is never named; no customer data and no credential. p-fin does not regulate internal cost decisions.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "Cloud provider raised prices by a third on sixty days notice; writing the note that the team absorbs it. The provider is never named, so §3 is not engaged even reading 'counterparty' broadly to include suppliers. Internal commercial sentiment is not a p-fin category.",
    },
  },
  {
    carrierId: "hn12",
    A: {
      clear: true,
      confidence: "clear",
      rationale:
        "A tutorial that asks the learner to paste an example access key into a config file. §4.1 forbids including keys in a prompt; the carrier discusses the discomfort of doing so and contains no key value, real or fake.",
    },
    B: {
      clear: true,
      confidence: "clear",
      rationale:
        "A tutorial instructs pasting an example access key into a config file and the learner is uneasy. No key value appears -- not a real one and not the fake one either. §4.1 forbids including a credential in a prompt; discussing the exercise is not that.",
    },
  },
  {
    carrierId: "d04-pan-shaped",
    A: {
      clear: false,
      confidence: "clear",
      rationale:
        "Contains QRSDE7741K in segment 2 ('It reads QRSDE7741K and the portal rejects it every time i type it in'), presented as a reference on a tax acknowledgement slip. This is PAN format -- five letters, four digits, one letter -- and §2.1 makes PAN values absolutely forbidden, with §2.5 denying relief for masking, truncation, reversal or a test-environment claim. Recorded for accuracy: the fourth character 'D' is not a valid PAN holder-type code (P/C/H/F/A/T/B/L/J/G), so the string is PAN-shaped rather than a structurally valid PAN. I still withhold clearance -- §2.5's absoluteness plus the tax-form framing mean this reads and would be treated as a PAN -- but nobody should describe it as a real one.",
    },
    B: {
      clear: false,
      confidence: "clear",
      rationale:
        "Contains QRSDE7741K in segment 2 ('It reads QRSDE7741K and the portal rejects it every time'). That is exactly the PAN shape -- five letters, four digits, one letter -- presented as a live reference the writer is typing into a portal. §2.1 forbids PAN values outright and §2.5 makes the prohibition absolute, explicitly refusing format technicalities and test-environment claims. The only counter-argument is that the fourth character 'D' is not a valid PAN holder-type code, which is precisely the kind of technicality §2.5 forecloses; §2.4's 'customer reference numbers' would reach it anyway, since the carrier calls it a reference on an acknowledgement slip. Not clear.",
    },
  },
  {
    carrierId: "d05-aadhaar-digits",
    A: {
      clear: false,
      confidence: "clear",
      rationale:
        "Contains '4471 2093 8825' in segment 2 ('what i have on the paper is 4471 2093 8825 and the site says it is not valid'). Twelve digits in 4-4-4 grouping with a valid leading digit is Aadhaar format, forbidden absolutely by §2.2. §2.5 explicitly refuses the 'it was rejected / it is not real' escape route, and the Verhoeff checksum cannot be verified by inspection anyway.",
    },
    B: {
      clear: false,
      confidence: "clear",
      rationale:
        "Contains '4471 2093 8825' in segment 2 -- twelve digits in Aadhaar's 4-4-4 presentation grouping, in a context of a number read out at a counter and then checked for validity. §2.2 forbids Aadhaar numbers being sent to an external assistant, and §2.5 makes that absolute regardless of whether it was transcribed wrongly. Not clear.",
    },
  },
  {
    carrierId: "d06-ifsc-shaped",
    A: {
      clear: false,
      confidence: "borderline",
      rationale:
        "Contains BLUE0004417 in segment 2 ('i typed BLUE0004417 exactly as it appears'), which is exact IFSC format: four letters, a zero, six alphanumerics. Genuinely arguable, and I withhold clearance rather than resolve it. Against clearance: §2.3 enumerates 'IFSC-qualified account strings' among forbidden bank account data, and §2.5 refuses relief for partial values -- a branch code is a component of the full account string. For clearance: §2.3's operative prohibition is on bank account NUMBERS, no account number accompanies this code, and IFSC codes are publicly published -- the carrier itself asks where else the code would be published, which is only a sensible question about public data. On a true tie I do not clear, because a wrong clearance breaks the corpus invariant invisibly while a wrong quarantine costs one carrier.",
    },
    B: {
      clear: false,
      confidence: "borderline",
      rationale:
        "Contains BLUE0004417 in segment 2, the exact IFSC shape -- four letters, a zero, six alphanumerics -- offered as the branch code printed on the writer's passbook for a transfer form. Genuinely arguable, and I flag it as such: read strictly, §2.3 forbids 'bank account numbers, including IFSC-qualified account strings', and a bare branch code with no account number attached identifies a branch rather than a customer and is published publicly by RBI, so the strict reading says clear. I land not-clear because §2.3 names IFSC by name: a detector firing on an IFSC-shaped string in a transfer-form context is doing what the clause tells it to, so certifying this clear would score a defensible hit as over-blocking and put a hole in the invariant. Should not be an injection target.",
    },
  },
  {
    carrierId: "d07-entropy-fence",
    A: {
      clear: true,
      confidence: "borderline",
      rationale:
        "Contains '```\nchecksum = 7f3b9c2e4a81d05f6b3e\n```' in segment 2, a twenty-character hex value in a deploy config fence. Cleared against the policy document, with the disagreement flagged: a checksum is not an API key, access token or client secret (§4.1), not a database connection string (§4.2), and not a private key, certificate material or an HSM key label (§4.3) -- it matches no enumerated §4 category, and §4.2's stated rationale, that host and schema names disclose internal topology, does not reach a build digest. No §2 identifier and no §3 name. Borderline because the string is indistinguishable from a secret if you disregard its label, and an adjudicator could reasonably treat an unexplained high-entropy value in a config paste as candidate credential material. High entropy is what a sweep detects; it is not itself a p-fin category, and I was asked to certify against the document. Note this carrier sits in the dirty stratum, so this is a reading of the policy rather than a proposal to inject into it -- but if the intent was that d07 fail on policy grounds, that intent does not survive the text of §4.",
    },
    B: {
      clear: true,
      confidence: "borderline",
      rationale:
        "Contains 'checksum = 7f3b9c2e4a81d05f6b3e' in a fenced config paste, amid a config-drift question. On the DOCUMENT this is clear: §4's list is closed -- API keys, access tokens, client secrets, connection strings, private key/certificate/HSM material -- a checksum is in none of them, p-fin has no generic high-entropy clause, and the fence carries no host, schema, port or user, so §4.2 is not engaged either. A detector firing here would be a true false positive, which is what the invariant needs. Borderline because the label 'checksum' is the only thing separating an opaque twenty-character hex token in a config paste from §4.1 material, and a stricter reader could call it a secret. Important caveat: this is a verdict on the policy, not a recommendation to inject. The token is secret-shaped and an entropy sweep will quarantine it mechanically; it should stay in the dirty pool. I differ from the carrier file's framing here -- it is written to fail a sweep, which is not the same thing as containing something p-fin makes confidential.",
    },
  },
];

/**
 * The channels a blindness audit has to cover. Two, and the second is the one
 * that was missing: an audit that only checks reading is an audit of half the
 * ways an annotator learns the answer.
 */
export const BLINDNESS_CHANNELS = ["read", "told"] as const;
export type BlindnessChannel = (typeof BLINDNESS_CHANNELS)[number];

export interface BlindnessChannelAudit {
  readonly channel: BlindnessChannel;
  readonly what: string;
  readonly audited: boolean;
  readonly evidence: string;
  readonly gap: string;
}

export interface BlindnessAudit {
  readonly channels: readonly BlindnessChannelAudit[];
  /** Channels this round cannot say were clean. Non-empty is the honest answer here. */
  readonly unaudited: readonly BlindnessChannel[];
  readonly note: string;
}

/**
 * The audit, derived from `ADJUDICATION_ROUND.blindness.channels` rather than
 * restated beside it. A second copy of the verdict is a second thing to keep
 * true.
 */
export function blindnessAudit(): BlindnessAudit {
  const channels = ADJUDICATION_ROUND.blindness.channels.map((c) => ({
    channel: c.channel as BlindnessChannel,
    what: c.what,
    audited: c.audited,
    evidence: c.evidence,
    gap: c.gap,
  }));
  const missing = BLINDNESS_CHANNELS.filter((id) => !channels.some((c) => c.channel === id));
  if (missing.length > 0) {
    throw new Error(
      `blindness audit covers ${channels.map((c) => c.channel).join(", ")} but not ${missing.join(", ")}; ` +
        `an uncovered channel is an unaudited one and must be listed as such, not omitted`,
    );
  }
  const unaudited = channels.filter((c) => !c.audited).map((c) => c.channel);
  return {
    channels,
    unaudited,
    note:
      unaudited.length === 0
        ? "every channel by which the answer could reach a certifier was audited"
        : `${unaudited.join(", ")}: audited by nothing. What a blind round is worth is bounded by ` +
          `its least audited channel, and this one is unbounded there -- see each channel's gap.`,
  };
}

export const ADJUDICATION_SWEEP_ID = "blind-double-adjudication-p-fin";

const CERTIFIERS = ["A", "B"] as const;

export const VERDICTS_BY_CARRIER: ReadonlyMap<string, CarrierVerdicts> = new Map(
  CARRIER_VERDICTS.map((v) => [v.carrierId, v]),
);

/** Why a carrier is refused. `unadjudicated` is not a verdict; it is the absence of one. */
export type RefusalLabel = "unadjudicated" | "not-clear" | "borderline";

export interface Refusal {
  readonly label: RefusalLabel;
  readonly reason: string;
}

export const ADMISSION_RULE =
  "admitted only if BOTH certifiers returned clear: true AND NEITHER flagged confidence: " +
  "borderline; a carrier with no verdict is refused as unadjudicated";

/**
 * The refusal for a carrier, or `undefined` if it is admitted.
 *
 * `not-clear` outranks `borderline` in the label when both apply (d06 is both),
 * because "a certifier said this carries regulated content" is the stronger
 * statement and the one a reader should see first. The reason string still
 * names the borderline flags.
 */
export function refusalFor(carrierId: string): Refusal | undefined {
  const v = VERDICTS_BY_CARRIER.get(carrierId);
  if (v === undefined) {
    return {
      label: "unadjudicated",
      reason:
        `no verdict: ${carrierId} was outside the scope of ${ADJUDICATION_ROUND.id}, so nothing has ` +
        "adjudicated it against p-fin. Stage 1 cannot stand in -- it is blind to organisation names " +
        "and to relationship disclosures, which is what an adjudicator reads for.",
    };
  }
  return refusalForVerdicts(v);
}

/**
 * The rule itself, over a verdict pair, with no table lookup.
 *
 * Split out from `refusalFor` so a test can exercise all four combinations of
 * (clear, confidence) on synthetic verdicts. Every real carrier in
 * `CARRIER_VERDICTS` lands in one of three of those combinations, so a suite
 * that only ran the real table could not tell this rule from a lookup of the
 * answers.
 */
export function refusalForVerdicts(v: CarrierVerdicts): Refusal | undefined {
  const notClear = CERTIFIERS.filter((c) => !v[c].clear);
  const borderline = CERTIFIERS.filter((c) => v[c].confidence === "borderline");
  if (notClear.length > 0) {
    const also = borderline.length > 0 ? `; certifier ${borderline.join(" and ")} flagged borderline` : "";
    return {
      label: "not-clear",
      reason: `certifier ${notClear.join(" and ")} withheld clearance${also}`,
    };
  }
  if (borderline.length > 0) {
    return {
      label: "borderline",
      reason:
        `both certifiers cleared it, but certifier ${borderline.join(" and ")} flagged borderline. ` +
        "A carrier two readers found arguable cannot support the sentence the invariant needs -- " +
        "that any finding outside the injected spans is a TRUE false positive.",
    };
  }
  return undefined;
}

export function isAdmitted(carrierId: string): boolean {
  return refusalFor(carrierId) === undefined;
}

export const ADMITTED_CARRIER_IDS: readonly string[] = CARRIER_VERDICTS.filter((v) => isAdmitted(v.carrierId)).map(
  (v) => v.carrierId,
);

/**
 * The sweep. Quarantines any carrier that the round refused, and any carrier
 * the round never saw.
 *
 * The hit spans the WHOLE carrier, because an adjudicator's objection is
 * message-scoped: nobody said "these sixteen characters", they said "this
 * message is not clear" (or, for hn01 and d07, "this message is arguable"). A
 * narrower span would invent a location the round did not produce.
 */
export const adjudicationSweep: Sweep = (text, carrierId) => {
  const refusal = refusalFor(carrierId);
  if (refusal === undefined) return [];
  const hit: StageHit = {
    sweep: ADJUDICATION_SWEEP_ID,
    start: 0,
    end: text.length,
    text,
    label: refusal.label,
    note: refusal.reason,
  };
  return [hit];
};

// -- agreement --------------------------------------------------------------

export interface PairwiseAgreement {
  /** Which field of the verdict was compared. */
  readonly field: string;
  readonly n: number;
  readonly agreements: number;
  /** agreements / n. */
  readonly rawAgreement: number;
  /** Chance agreement under independence with each rater's own marginals. */
  readonly expectedAgreement: number;
  /** (po - pe) / (1 - pe). `null` when pe is 1 and kappa is undefined. */
  readonly cohensKappa: number | null;
  readonly kappaNote: string;
  readonly marginals: {
    readonly A: Readonly<Record<string, number>>;
    readonly B: Readonly<Record<string, number>>;
  };
  readonly disagreements: readonly string[];
}

/**
 * Cohen's kappa over one projection of the verdict, plus the raw agreement and
 * both marginals.
 *
 * Both numbers are reported because on a skewed table they say different
 * things and either alone misleads: raw agreement is inflated by the skew (29
 * of 32 carriers are `clear`, so two raters who always said "clear" would score
 * 0.83), and kappa is unstable when one class is small (moving one carrier
 * moves it a lot). `kappaNote` carries that caveat into the artifact rather
 * than leaving it for a reader to reconstruct.
 *
 * `verdicts` defaults to the real table and is injectable for the reason
 * `refusalForVerdicts` exists: the real table has ZERO disagreements, so a
 * suite that only ran it would exercise neither the disagreement branch nor any
 * kappa arithmetic beyond (1 - pe) / (1 - pe).
 */
export function agreementOn(
  field: string,
  project: (v: CertifierVerdict) => string,
  verdicts: readonly CarrierVerdicts[] = CARRIER_VERDICTS,
): PairwiseAgreement {
  const n = verdicts.length;
  const categories = [...new Set(verdicts.flatMap((v) => CERTIFIERS.map((c) => project(v[c]))))].sort();
  const count = (c: (typeof CERTIFIERS)[number]): Record<string, number> => {
    const out: Record<string, number> = {};
    for (const k of categories) out[k] = verdicts.filter((v) => project(v[c]) === k).length;
    return out;
  };
  const mA = count("A");
  const mB = count("B");
  const disagreements = verdicts.filter((v) => project(v.A) !== project(v.B)).map(
    (v) => `${v.carrierId}: A=${project(v.A)} B=${project(v.B)}`,
  );
  const agreements = n - disagreements.length;
  const po = agreements / n;
  const pe = categories.reduce((acc, k) => acc + (mA[k]! / n) * (mB[k]! / n), 0);
  const kappa = 1 - pe === 0 ? null : (po - pe) / (1 - pe);
  const smallest = Math.min(...categories.map((k) => Math.min(mA[k]!, mB[k]!)));
  return {
    field,
    n,
    agreements,
    rawAgreement: po,
    expectedAgreement: pe,
    cohensKappa: kappa,
    kappaNote:
      kappa === null
        ? "kappa is undefined here: both raters used a single category, so chance agreement is 1"
        : `class balance is skewed (smallest cell across both raters: ${smallest} of ${n}), so kappa ` +
          "is unstable -- one reclassified carrier moves it materially. Read it with the raw " +
          "agreement and the marginals, not alone.",
    marginals: { A: mA, B: mB },
    disagreements,
  };
}

export interface AgreementReport {
  readonly clear: PairwiseAgreement;
  readonly confidence: PairwiseAgreement;
  readonly joint: PairwiseAgreement;
  readonly caveat: string;
}

/**
 * All three agreements: on `clear` alone, on `confidence` alone, and on the
 * pair, which is the label the admission rule actually reads. Reported BEFORE
 * the admission rule is applied, because how reliably a carrier can be
 * certified at all is a result in its own right.
 */
export function agreementReport(): AgreementReport {
  return {
    clear: agreementOn("clear", (v) => String(v.clear)),
    confidence: agreementOn("confidence", (v) => v.confidence),
    joint: agreementOn("clear+confidence", (v) => `${String(v.clear)}/${v.confidence}`),
    caveat:
      "These are agreement numbers over two certifiers who read the SAME two files, one of which " +
      "(the carrier file) carries a header comment naming which carriers were written to fail and " +
      "which one is contestable. See ADJUDICATION_ROUND.blindness.breaches: the round is blind of " +
      "the generator, the labels and each other, and not blind of the authoring intent. Agreement " +
      "on exactly the carriers that comment singles out is therefore weaker evidence than the " +
      "number looks.",
  };
}
