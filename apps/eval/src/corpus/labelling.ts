import { PREDICATE_ID } from "./questions.js";

/**
 * The blind two-annotator LABELLING round over
 * `corpora/generated/injection-p-fin-v2.labelling-queue.jsonl`, recorded
 * verbatim, adjudicated here, and -- this is the part a reader should not skip
 * -- SHORT of what it was commissioned to produce.
 *
 * ## What was asked for, and what came back
 *
 * `questions.ts` built a queue of 209 questions over the 189 items of
 * `injection-p-fin-v2`: one `message-predicate` question per item, asking
 * whether the finished message discloses that a NAMED organisation is a client,
 * prospective client, deal counterparty or NDA party of the Firm, plus 20
 * `contested-span-label` questions on the two confusable families whose p-fin
 * reading the generator itself recorded as arguable. The deliverable was
 * `pred:` gold at a scale above the two items in
 * `corpora/fixtures/smoke.gold-tier2.jsonl`.
 *
 * Both annotators answered the 20 span-bearing rows and neither answered the
 * other 169. Both say why in their own scope notes, and both say the same
 * thing: the brief described the unit of work as "the message text and one
 * highlighted span with a proposed type", and 169 of the 189 items have no
 * span. So the round covered 20 of 189 items (10.6%), every one of them
 * selected for SPAN contestation rather than for predicate content.
 *
 * The consequence is arithmetic, not opinion. The generator predicts 19 items
 * satisfy the predicate (`meta.predicateConstruction.constructed`), and the
 * intersection of those 19 with the 20 covered is EMPTY -- checked in
 * `coverageReport()`, not asserted. All 20 predicate answers came back "no",
 * from both annotators. So this round produced 20 predicate labels, 0 of them
 * positive, and `pred:` gold at scale WAS NOT DELIVERED. Tier-2 gold in this
 * repository is still the two items in the smoke fixture. That sentence is the
 * headline result of the round and it is repeated in the manifest's
 * `unvalidated` block rather than softened.
 *
 * Both annotators independently volunteered where the positives are: both
 * quote `inj-o01-1`'s "the party we signed the confidentiality agreement with
 * is Marrowfield Group" -- a clean §3.3 NDA-party hit -- which each of them saw
 * only because it happened to be in the first lines of the file, and which
 * neither was asked about.
 *
 * ## Why this is a separate module from `adjudication.ts`
 *
 * `adjudication.ts` records a different round (which CARRIERS may be injected
 * into) and its exported values are serialized verbatim into
 * `injection-p-fin-v2.manifest.json`. That manifest is byte-compared against
 * the committed file on every `pnpm -r test`, and the queue those bytes
 * describe is the artifact these annotators were handed. Editing it to make
 * room for a second round would rewrite the record of what was handed over, so
 * nothing here touches it. The one thing the two rounds share is the agreement
 * arithmetic, and `corpus-labelling.test.ts` proves the copy here reproduces
 * `adjudication.ts`'s numbers on `adjudication.ts`'s own table rather than
 * leaving two definitions of kappa in one package.
 */

export const LABEL_ROUND_ID = "v2-blind-double-labelling-p-fin";

export type PredicateConfidence = "clear" | "borderline";
export const ANNOTATORS = ["A", "B"] as const;
export type Annotator = (typeof ANNOTATORS)[number];

/** One annotator's answers on one item, transcribed from their returned array. */
export interface AnnotatorLabel {
  readonly itemId: string;
  /** Their answer to "is the generator's proposed span type correct?". */
  readonly spanLabelCorrect: boolean;
  /** Present only when `spanLabelCorrect` is false: the type they would put there instead. */
  readonly correctedType?: string;
  /** Their answer to the message-scoped predicate question. */
  readonly satisfiesPredicate: boolean;
  readonly predicateConfidence: PredicateConfidence;
  /** Verbatim. Never summarised: the reasoning is the provenance. */
  readonly rationale: string;
}

export const BRIEF_PARAPHRASE_FRAGMENT =
  "a named organisation is a client, prospective client, or counterparty of the Firm";

/**
 * The round, its scope, and what the two channels an answer can travel by came
 * back with.
 *
 * `blindness.channels` follows `adjudication.ts`'s structure deliberately: one
 * entry per channel, each saying what was checked and against what evidence,
 * with `audited: "partial"` and `"unaudited"` recorded as findings rather than
 * smoothed. Two channels are new here. `structural` names leaks that travel
 * inside the blind artifact itself, which neither a filesRead list nor a brief
 * transcript can catch; `scope` names the one that cost this round its
 * deliverable.
 */
export const LABEL_ROUND = {
  id: LABEL_ROUND_ID,
  policy: "policies/p-fin.md",
  artifactUnderLabel: {
    queue: "corpora/generated/injection-p-fin-v2.labelling-queue.jsonl",
    corpus: "corpora/generated/injection-p-fin-v2.jsonl",
    why:
      "the queue is what was handed over and the corpus is what the labels attach to. Both are " +
      "byte-compared in corpus-labelling.test.ts against the sha256 the v2 BUILDER produces in that " +
      "same process -- not against a digest copied out of the committed manifest -- so a later edit " +
      "to either cannot silently reinterpret these labels.",
  },
  annotators: ANNOTATORS,
  independence:
    "Each annotator worked without sight of the other's answers or reasoning, and returned a " +
    "structured label per item plus a free-text method note, a filesRead list and a wasToldAbout " +
    "disclosure.",
  questionsAsked: {
    inQueue: 209,
    messagePredicate: 189,
    contestedSpanLabel: 20,
    answeredByBothAnnotators: 20,
    questionsUnanswered: 189,
    /**
     * NOT the same 189. 209 questions minus the 20 answered leaves 189
     * QUESTIONS unanswered, and it is a coincidence that the corpus also holds
     * 189 items. Each of the 20 answered items carried a predicate question
     * too, so the items with no predicate answer number 169. Both numbers are
     * recorded because both annotators' notes conflate them -- see
     * `discrepancies`.
     */
    itemsWithNoPredicateAnswer: 169,
    why:
      "both annotators answered every contested-span row and neither answered a span-less row. " +
      "See blindness.channels[scope]: the brief described the unit of work as a message plus one " +
      "highlighted span, and 169 of the 189 items have no span.",
  },
  blindness: {
    rulesGiven:
      "read policies/p-fin.md and the queue's message text and span only; no families file, no " +
      "manifest, no compiled IR, no gold fixtures, nothing under runs/, no git log, no plan",
    channels: [
      {
        channel: "read",
        what: "the files each annotator opened, from their returned filesRead list",
        audited: "partial",
        evidence:
          "both filesRead lists were compared item by item against the modules and artifacts that carry the answer. Neither lists families.v2.ts, generate.ts, inject.ts, labels.ts, questions.ts, build-v2.ts, either manifest, policies/compiled/p-fin.ir.json, corpora/fixtures/smoke.gold-tier2.jsonl, or anything under runs/. Both DID open corpora/generated/injection-p-fin-v2.jsonl, which carries gold and meta and is absent from that list only because the list names modules; both disclosed it, both bounded what they took from it, and A additionally probed the two superseded corpora. See breaches.",
        gap:
          "a filesRead list is self-reported, and this repository still has no independent record of file access to check it against. Same gap as the carrier round.",
        breaches: [
          "BOTH: injection-p-fin-v2.jsonl was opened, which carries gold and meta. Forced by the task rather than careless -- the queue deliberately omits the proposed span type, so the only place in the repository to recover the claim they were asked to check is the corpus file. A read start/end/text/entityType and saw meta's KEY NAMES but no meta values; B probed the schema for key names, wrote all 20 predicate answers to disk BEFORE opening gold, then read entityType at the 20 contested offsets and nothing else. The queue design is what should change: a round asked to check a proposed type must be handed the proposed type.",
          "A ONLY: injection-p-fin-v1.jsonl and injection-p-fin-adjudicated-v1.jsonl were queried programmatically for whether the 20 contested span strings appear in their gold (all absent). Neither file was rendered. Disclosed by A.",
          "B ONLY: an orienting `head -c 2000` of the queue rendered the full text of inj-o01-0 and inj-o01-1 before the task was scoped. Neither is among B's 20. inj-o01-1 is the NDA-party positive B later cites, so B saw one uncovered positive; it is outside the labelled set and changes no label here. Disclosed by B.",
        ],
      },
      {
        channel: "told",
        what: "the brief each annotator was given: task, scope, decision rule, and any framing in it",
        audited: "partial",
        evidence:
          "this is the first round in this repository where the told channel could be examined at all. The previous round recorded it audited: false because no brief was retained. No brief was retained this round either, but both annotators returned a wasToldAbout disclosure quoting the instruction, and the two quotations agree sentence by sentence -- which is corroboration from two independent readers, not a transcript. One of the three breaches below does not rest on self-report at all: it is verified by string comparison against the compiled IR in corpus-labelling.test.ts.",
        gap:
          "the brief itself is still not committed anywhere, so what is audited here is two reconstructions of it. Anything both annotators failed to notice is invisible. Retain the verbatim brief.",
        breaches: [
          "TOLD-1, AND THE SERIOUS ONE: the brief handed both annotators a paraphrase of the predicate containing the exact phrase BRIEF_PARAPHRASE_FRAGMENT, which is policies/compiled/p-fin.ir.json's own semanticPredicates[0].nlPredicate. The compiled IR is the artifact the arm under test detects against and it is on the forbidden list. MEASURED, not inferred: that phrase occurs in the IR, and occurs in neither policies/p-fin.md nor questions.ts's PREDICATE_QUESTION, so the IR is its only source in the policy inputs -- corpus-labelling.test.ts asserts all three. Both annotators independently noticed the paraphrase DROPS a limb (§3.3's 'parties under a non-disclosure agreement') and both restored it from the document; neither knew where the wording came from. The bias it carries is toward the compiled arm's own reading, in the direction of under-firing on NDA parties. The corpus holds 3 nda-party items and none is among the 20 labelled, so no label in this round is affected -- but a round that had covered the other 169 would have been scored against a key contaminated by the arm it was going to score.",
          "TOLD-2: the brief supplied a DIRECTIONAL prior on the answer key -- that several proposed types were found contradicted by the policy last round, with a worked example of a certificate block labelled non-confidential against §4.3. That prior is STALE with respect to the artifact under label: the family it describes is csr-pem-block, and families.v2.ts's REMOVED_FAMILIES removes it from v2 for exactly that reason. So both annotators were told the key under-labels, about a defect already removed from the file they were labelling, and every one of the 20 proposals they saw was of the shape the prior named. Both reported the pull; B states outright that the seven flips cannot be claimed uncontaminated by it. This is the strongest candidate explanation for the one disagreement in the round.",
          "TOLD-3: distributional hints in both directions -- that marking every item false, or every item true because organisation names appear, would make the work worthless, and that 'borderline' should be used honestly and often. Both tell the annotator the key is mixed and that arguable cases exist in quantity. Both annotators flagged them, both returned a uniformly false predicate column anyway, and both said in terms that they were holding an answer the hint argued against. Recorded because the hints made the honest answer harder to report, which is the wrong direction for a hint to push.",
        ],
      },
      {
        channel: "structural",
        what: "answer-bearing content travelling inside the blind artifact itself",
        audited: "audited",
        evidence:
          "both annotators found these and neither audit channel above would have. They are checkable against the committed queue and corpus: the id prefixes are in the corpus, and the question text is in the queue.",
        gap:
          "none for what is listed -- both leaks are in committed files and can be read directly. The gap is that this class was not looked for until an annotator raised it.",
        breaches: [
          "STRUCTURAL-1: item ids encode the stratum. `inj-hn*` against `inj-o*` reads as hard-negative against ordinary, which is generator intent shipped inside the identifier and visible to anyone who touches the corpus. B mitigated it by relabelling to opaque keys B00-B19 under a hash of the text, and states the mapping stayed recoverable. 8 of the 20 labelled items carry the hn prefix.",
          "STRUCTURAL-2: the queue's own `question` field carries the generator's argument. It names the span's semantic type in prose ('The quoted span is a sftp file-transfer endpoint', 'The quoted span is a NPCI retrieval reference number') -- which neither annotator could have derived with confidence from a bare 12-digit number -- and it pre-argues both readings with the clause numbers attached. The predicate question additionally bounds the reading to 'sections 3.1, 3.3 and 3.4 only'. A read two of these in full before finishing p-fin.md and says the §4.2 and §1.2 analyses START from a framing the generator wrote; B says it constrained where the reasoning looked, and found the decisive point in §5, outside the nominated sections. questions.ts's own docblock claims the queue 'carries no family id, no constructedRole and no gold, so an annotator cannot read the generator's intent off the row'. These two disclosures falsify that claim, and the claim is in a committed manifest.",
        ],
      },
      {
        channel: "scope",
        what: "what the brief defined as the unit of work, and what that silently excluded",
        audited: "audited",
        evidence:
          "both annotators opened their notes with the same scope statement and the same reason, and the arithmetic is checkable: coverageReport() counts the covered items against the corpus and against meta.predicateConstruction.",
        gap:
          "none -- the consequence is measured. It is recorded as a breach because a scoping choice that removes 89% of the questions is not a detail of the brief, it is the result of the round.",
        breaches: [
          "SCOPE-1: the brief described the unit of work as a message plus one highlighted span with a proposed type. 169 of the 189 items have no span, so 169 predicate questions were never put. The 20 that were are exactly the items selected for SPAN contestation, which is a selection on a property unrelated to the predicate -- both annotators say so, and both say the resulting all-false predicate column is a property of the selection rather than of their reading. The 19 items the generator predicts satisfy the predicate are disjoint from the 20 covered, so the round tested the construction prediction only where it predicts 'no'.",
        ],
      },
    ],
  },
  /**
   * Prose that disagrees with the same annotator's own structured answers.
   * Recorded rather than reconciled, for the reason the carrier round records
   * its d07 discrepancy: a reader who quotes the note will get a number that
   * disagrees with this file and should know why. The structured array is what
   * a machine reads and what each per-item rationale supports, so the tables
   * below are transcribed from it.
   */
  discrepancies: [
    "Annotator A's note counts the span families as 'an internal sftp file-drop URL (14 items) and a 12-digit NPCI retrieval reference number (6 items)' and repeats 14 and 6 in FINDING 1 and FINDING 2. A's own 20 per-item rationales split 13 sftp and 7 RRN, which is what the committed corpus says (meta.labels types: neg:sftp-endpoint x13, neg:retrieval-reference x7, asserted in corpus-labelling.test.ts). B's note says 13 and 7. The tables here use 13 and 7.",
    "Both annotators describe the queue as '189 message-predicate questions and 20 contested-span questions', and A's scope line reads 'the other 189 are message-predicate-only'. The queue holds 209 rows: 189 message-predicate plus 20 contested-span, and the 20 contested-span rows name items that ALSO carry a message-predicate row. So the items left unanswered number 169, not 189. The counts in questionsAsked above are read off the committed queue.",
  ],
  notes: {
    A:
      "SCOPE: I labelled the 20 items in injection-p-fin-v2.labelling-queue.jsonl carrying kind \"contested-span-label\" (the only queue entries that pair a message with a highlighted span; the other 189 are message-predicate-only and have no span). Two span classes account for all 20: an internal sftp file-drop URL (14 items) and a 12-digit NPCI retrieval reference number (6 items).\n\nWHAT \"THE PROPOSED TYPE\" WAS. The queue's span objects carry no type field. I recovered the generator's claim by checking each contested span against that item's own gold span list in injection-p-fin-v2.jsonl: none of the 20 appears there (nor in v1 or adjudicated-v1). So the proposed type for every item is effectively \"none / no entity type / not withheld\", and every spanLabelCorrect below is a judgement on that claim. I read only start/end/text/entityType from gold; I did not read the meta object on any item.\n\nVOCABULARY. The corpus's entityType vocabulary is api-credential, bank-account-identifier, client-name, db-connection-string, in-aadhaar, in-pan, internal-customer-id, private-key-material — i.e. exactly the §2/§3/§4 must-withhold classes. There is no general \"customer data\" type. That constrains what a correctedType could even say, and it matters for the RRN finding below.\n\nFINDING 1 — sftp endpoints (14 items): the generator is right, and the reasoning is not close in the way the queue's framing suggests. §4.2's operative subject is \"Database connection strings\". An sftp URL names no driver, no database and no schema. The trailing \"because the host and schema names disclose the Firm's internal topology\" is a purposive clause; it explains the rule, it does not create a second one, and the document's own preamble (\"Where a clause is absolute, it says so\") plus the explicit anti-circumvention drafting in §2.5 and §6.3 show a drafter who broadens by saying so. §4 is titled \"Credentials and secrets\" and an unauthenticated file path is neither. I checked every other clause: §2 (no customer identifier), §3 (intraledger.example is the Firm's own internal domain, not a client or counterparty), §4.1, §4.3, §5.1 (a settlement server does not identify an account holder). Nothing reaches it. Worth flagging to the project as a policy gap rather than a labelling error: p-fin.md protects internal topology only through the database-connection-string rule, so an sftp host that discloses the same topology falls outside it. That is a defect in COMP-2026-11, not in the corpus.\n\nFINDING 2 — NPCI retrieval reference numbers (6 items): the generator is right within the §2–§4 classification the vocabulary encodes, but this is the genuinely contested one and the project should know why. §2.3 enumerates bank account numbers, IFSC-qualified strings and UPI VPAs; §2.4 enumerates \"Internal customer identifiers, including customer reference numbers, CIF numbers, and KYC case ids\". An RRN is issued by the payment network, not the Firm (so not \"internal\"), and keys a transaction, not a customer (so not a \"customer identifier\"). inj-o03-1 is the clean control: it carries a real §2.4 item — \"the customer reference on that relationship is just 92133109\" — alongside the RRN, and the two are plainly different objects. HOWEVER: an RRN does fall inside §1.2's definition of customer data, because it \"can be combined with other information to identify\" the account holder — joining it to the Firm's own records is exactly what a retrieval reference is for, and the carrier phrase names the counterparty of the credit as \"the resident\". §1.2 is not inert: §5.1 (\"No customer data of any classification may be sent to non-enterprise or foreign-hosted services\") and §5.2 make it operative. So the honest position is that an RRN is customer data whose transmission is restricted by DESTINATION, not an identifier forbidden outright by §2. Because §5.2 expressly permits customer data to go to the approved assistant, \"no §2–§4 entity type\" is the correct label; but if this corpus is ever scored against a model that also decides §5 destination questions, these six spans are not neutral and should not be treated as clean negatives. I kept spanLabelCorrect: true rather than flipping, because flipping would assert §2 coverage that the enumeration does not support — but I want the §1.2/§5.1 reading on the record for adjudication.\n\nFINDING 3 — the predicate came out false on all 20, and I want to say plainly why that is not laziness. These 20 items were selected for SPAN contestation, so their carriers are whatever happened to host an sftp URL or an RRN; the predicate distribution in this subset is incidental. Eighteen of the twenty name no organisation at all — the organisations that appear are unnamed by design (\"The company\", \"Our cloud provider\", \"a supplier abroad\", \"the bank\"). Exactly two name an organisation, and both are near-misses in the interesting direction: inj-o15-1's \"Marrowfield Group\" is named with the relationship expressly DENIED (\"none of it touches us\"), and inj-hn08-1's \"Quillrun\" is named with a real relationship but the wrong KIND of relationship (a scheduler the Firm runs on, not a client, prospective client, deal counterparty or NDA party). Both are marked borderline. If the wider 189-item queue is scored, I would expect the true-predicate mass to sit there, not here — I saw enough of the head of that file to know carriers like \"the party we signed the confidentiality agreement with is Marrowfield Group\" exist, which is a clear §3.3 NDA-party hit.\n\nONE CAUTION ON §3.4. §3.4 says \"the pairing of a client name with a routine operational question is itself a disclosure of the relationship.\" It is tempting to read that as making inj-o15-1 a hit, since Marrowfield is paired with a latency investigation. It does not: §3.4 presupposes the name IS a client name and explains why §3.1 exists; it does not convert an arbitrary named company into a client. A model that treats §3.4 as an independent trigger will over-fire on exactly this item, which makes it a valuable item to keep.",
    B:
      "SCOPE. The queue file holds 209 questions over 189 messages: 189 `message-predicate` questions (no span) and 20 `contested-span-label` questions (span + a proposed reading). Only those 20 match this task's description (\"the message text and one highlighted span with a proposed type\"), and their itemIds are a strict subset of the 189. I labelled those 20 and answered both questions for each. I did not fabricate a `spanLabelCorrect` for the 189 span-less items; if the intent was to label all 189 on the predicate alone, that is a separate pass and the schema should drop the span field for it.\n\nWHERE THE \"PROPOSED TYPE\" CAME FROM. The queue's span objects carry only {start,end,text} — no type. The proposed type exists only in the corpus `gold` array (keys: start/end/text/entityType/action). To avoid contaminating the predicate judgement I wrote all 20 predicate answers to lab-B/predicate-committed.json BEFORE opening `gold`, then extracted only the entityType for the exact contested offsets (lab-B/proposed-types.json) and read nothing else from gold. Result: all 20 contested spans are ABSENT from gold, i.e. the uniform proposal is \"not confidential / permit\".\n\nQ1 VERDICTS. Two span families, and they split.\n- 13 sftp file-drop endpoints (sftp://transfer-NN.intraledger.example/outbound/settlements): proposal CORRECT. §4.2's operative subject is \"Database connection strings\". An sftp URL names no driver and no schema; \"/outbound/settlements\" is a directory. The trailing \"because the host and schema names disclose the Firm's internal topology\" is a stated reason, not a second operative sentence. The document broadens explicitly when it means to (\"including chat interfaces…\" §1.1; \"including customer reference numbers, CIF numbers, and KYC case ids\" §2.4); §4.2 has no such broadening. §6.1 also requires a blocked prompt to be logged \"with the clause that blocked it\", which presupposes a determinate clause — you cannot cite §4.2 here without rewriting its subject. The purposive reading is genuinely available, but it would also forbid every redis/jdbc host in the corpus under a clause that never says so.\n- 7 NPCI retrieval reference numbers: proposal WRONG. This is the load-bearing argument, so I state it once here rather than seven times. §2.3 enumerates account numbers/IFSC strings/UPI addresses — an RRN is none. §2.4's genus is \"internal customer identifiers\"; an NPCI reference is externally issued and names a transaction, so it fails the genus even though \"including\" makes the list open. But §1.2 defines customer data as information that \"can be combined with other information to identify\" an accountholder, and every one of these is framed as \"the retrieval reference on that credit to the resident\" — a specific customer's credit, quoted from the branch. Decisively, the phrase \"customer data\" never appears in §2, §3 or §4; it appears in the title, the preamble, and §5.1/§5.2/§5.4/§5.5. So §1.2 does not \"feed §2–§4\" as the queue's framing suggests — it feeds §5. That makes the two layers non-redundant rather than one swallowing the other: §2 = never send anywhere, absolute, no exceptions (§2.5, §6.3); §1.2+§5 = customer data, so approved enterprise assistant only. An RRN is in the second layer, and a span typed \"none\" misses a real operative restriction. §6.2 (\"A breach of §2 or §4 is a reportable security incident\") corroborates that §5 breaches are breaches that simply are not reportable incidents.\nThe asymmetry between the two families is principled, not me splitting the difference: an sftp hostname identifies a server, so §1.2 cannot reach it; an RRN tied to a named customer's credit is exactly what §1.2 describes.\n\nQ2 — ALL TWENTY ARE FALSE, AND WHY THAT IS NOT LAZINESS. I was warned against blanket-false, so I checked mechanically as well as by reading: a regex sweep for mid-sentence capitalised tokens across all 20 texts returns proper nouns in only three (Marrowfield Group, Arjun Beheria, Quillrun) plus the sentence-initial \"Our\"/\"Kindly\". Marrowfield Group is named with the relationship expressly disclaimed (\"none of it touches us\"); Arjun Beheria is a person on the Firm's own operations desk, and §3 governs organisation names; Quillrun is the Firm's scheduling tool, i.e. the Firm is the customer, not the counterparty. Several items are near-misses in the other direction — \"a payment to a supplier abroad\" (B15/inj-hn10-1) and \"the customer reference on that relationship\" (B18/inj-o03-1) both state a relationship with no name. §3.1/§3.3 require a NAMED client, prospective client, deal counterparty or NDA party, and §3.4's rationale presupposes a name that is in fact a client's. Independently, I know positives exist in the wider corpus: the first 2KB of the queue, which I read before scoping the task, contains \"the party we signed the confidentiality agreement with is Marrowfield Group\" (item inj-o01-1, NOT in my 20) — a clear §3.3 NDA-party positive. So the uniform false here is a property of which 20 items were selected for span contestation, not of my reading. I flagged one borderline (Quillrun) rather than manufacturing spread.\n\nINCIDENTAL, OUT OF SCOPE. B15/inj-hn10-1 contains an \"-----BEGIN EC PRIVATE KEY-----\" block, squarely within §4.3, and a \"-----BEGIN DH PARAMETERS-----\" block, which is NOT private-key or certificate material and is a fair question. Neither is the span I was asked about, so neither is in my labels. If the \"certificate block labelled non-confidential\" error mentioned in my instructions is still live, this item is where I would look next.",
  },
  filesRead: {
    A: [
      "/private/tmp/.../scratchpad/lab-A/p-fin.md (copy of /Users/orce/Desktop/Repositories/sih/policies/p-fin.md — read in full)",
      "/private/tmp/.../scratchpad/lab-A/items.jsonl (my own extract: itemId, text, span offsets/text for the 20 contested-span items — read in full)",
      "corpora/generated/injection-p-fin-v2.labelling-queue.jsonl (source of the 20 items; see wasToldAbout for exactly which fields I saw rendered)",
      "corpora/generated/injection-p-fin-v2.jsonl (message text for the 20 items, plus gold span start/end/text/entityType read programmatically; meta never rendered — only its key names)",
      "corpora/generated/injection-p-fin-adjudicated-v1.jsonl (never rendered; queried programmatically only, for whether the 20 contested span strings appear in gold — all absent)",
      "corpora/generated/injection-p-fin-v1.jsonl (never rendered; same programmatic absence check only)",
      "Directory listings only, no contents: corpora/generated/ and policies/ (both listings show a manifest filename; I opened none of them)",
    ],
    B: [
      "/Users/orce/Desktop/Repositories/sih/policies/p-fin.md",
      "/private/tmp/.../scratchpad/lab-B/p-fin.md",
      "corpora/generated/injection-p-fin-v2.labelling-queue.jsonl",
      "corpora/generated/injection-p-fin-v2.jsonl",
      "/private/tmp/.../scratchpad/lab-B/items.json",
      "/private/tmp/.../scratchpad/lab-B/proposed-types.json",
      "/private/tmp/.../scratchpad/lab-B/predicate-committed.json",
    ],
  },
  wasToldAbout: {
    A:
      "Answering the audit question directly: the prompt told me a substantial amount about the corpus's construction, and some of it is prejudicial in a way reading-only audits would miss. Listing every channel I can identify, including ones that cut against me.\n\nFROM THE PROMPT ITSELF:\n\n1. That the items are GENERATED by a generator that attaches a proposed span type, and that the proposed type is \"a CLAIM, not the answer\". This tells me the labels have an author with an intent, and invites me to disagree.\n\n2. The strongest anchor, and the one I want on the record: \"Several were found contradicted by the policy last round -- one labelled a certificate block as non-confidential when §4.3 forbids 'certificate material' outright.\" This supplies a prior with a DIRECTION (the generator's errors are false negatives), a MAGNITUDE (\"several\"), and even a worked example with a clause number. It creates real pressure to flip proposed-none spans to confidential. I noticed the pull, and it is relevant that both of my span classes are proposed-none. I resisted it by walking every clause of §2–§5 against each span independently, and I ended up affirming the generator on all 20. If an adjudicator disagrees with me on the RRN class, this prompt-supplied prior is a live explanation for why I might have been pushed the OTHER way and wasn't — but it is also the reason I documented the §1.2/§5.1 counter-reading at length rather than silently affirming. I note incidentally that the §4.3 example is not idle: item inj-hn10-1 contains a \"-----BEGIN DH PARAMETERS-----\" block, so I was primed on a clause that is live in my item set, even though it is not the span I was asked about.\n\n3. A PARAPHRASE OF THE PREDICATE, supplied before I had read the clause: \"does the MESSAGE AS A WHOLE disclose that a named organisation is a client, prospective client, or counterparty of the Firm?\" The prompt did warn me not to take it as the definition, and that warning was warranted — the paraphrase DROPS a limb. §3.3 covers \"prospective clients, deal counterparties, and parties under a non-disclosure agreement\"; the paraphrase omits NDA parties and flattens \"deal counterparties\" to \"counterparty\". I restored the NDA limb from the document. It did not change any of my 20 answers, but on a corpus whose carriers demonstrably use NDA phrasing it could change others, and an annotator who trusted the paraphrase would under-fire.\n\n4. Distributional warnings in both directions: \"Marking every item satisfiesPredicate: false because it is safe, or true because organisation names appear.\" This tells me the true labels are mixed rather than uniform. My subset came out uniformly false, so this warning worked against my actual answer, and I am flagging that I held the answer anyway.\n\n5. \"Use predicateConfidence: 'borderline' honestly and often where it is arguable\" — an instruction that borderline cases EXIST in quantity, which is itself information about how the corpus was built (near-miss pairs).\n\n6. That there is a \"manifest\", a \"families file\", and that families \"state the answer the generator intended\". This reveals the corpus is family-structured: items come in designed groups sharing an intended answer. Knowing that, seeing 14 near-identical sftp spans and 6 near-identical RRN spans tells me they are two families with one intended answer each — so my 20 judgements are really 2 judgements, and any error I make is correlated across a family rather than independent. That is a construction fact I learned from the prompt, not from a file.\n\n7. That a previous round happened, had a blindness audit, and that these items are \"contested\" enough to have survived into a second version.\n\n8. Where the corpus lives (corpora/generated/) and that items carry text plus a span — enough to locate it without exploring.\n\nA CHANNEL THIS PROMPT'S COPY INSTRUCTION DID NOT ANTICIPATE, and the one I would fix first:\n\n9. The queue's own `question` field travels in the SAME FILE as the text and the span, so \"copy only the text and the span\" does not exclude it. That field does two things the instruction plainly meant to prevent. It NAMES the span's semantic type in prose — \"The quoted span is a sftp file-transfer endpoint\", \"The quoted span is a NPCI retrieval reference number\" — which I could not have derived with confidence from \"sftp://transfer-58.intraledger.example/outbound/settlements\" or from a bare 12-digit number. And it pre-argues the case with clause numbers attached, laying out both readings: for the sftp, §4.2's operative sentence versus §4.2's stated reason; for the RRN, the §2.3/§2.4 enumerations versus §1.2's \"can be combined with other information to identify\" read as operative. I read two of these in full (inj-o01-4#1 and inj-o03-1#1) before I had finished reading p-fin.md. My §4.2 and §1.2 analyses therefore START from a framing the generator wrote. I did read the whole policy afterwards and did check every clause against every span myself, and I reached a conclusion the framing did not dictate (it is deliberately even-handed and names no winner) — but I cannot claim I generated that issue-framing independently, and it would be wrong to let this pass as blind. The same field also states the intended scoping for the predicate questions (\"Answer from policies/p-fin.md sections 3.1, 3.3 and 3.4 only\" and \"A relationship stated without a name, and a name stated without a relationship, are both 'no'\"), which is a generator-authored decision rule for the exact judgement I was asked to make independently; I saw it in the first three lines of the file before I knew what the file contained.\n\n10. Minor: I saw the meta object's KEY NAMES while checking what to avoid — generator, generatorVersion, seed, carrierId, carrierSource, carrierRegister, irHash, density, injections, labels, predicateConstruction, labelQuestions, scoringScope, predicateLabelling, carrierStratum. No values. The names alone confirm the corpus is built by injecting spans into carriers at a controlled density, and that predicate labels are constructed rather than observed. I stopped there and read no meta values.\n\nWHAT I DID NOT TOUCH: nothing under runs/; policies/compiled/p-fin.ir.json; any manifest; any families file; smoke.gold-tier2.jsonl; no git log or plan. I ran no git commands at all.",
    B:
      "Reading-channel disclosures first, then the telling-channel, which is what you asked me to audit.\n\nREAD. I read the policy, the labelling queue in full, and two things from the corpus file itself that deserve naming. (1) A structural probe of injection-p-fin-v2.jsonl printing only KEY NAMES revealed the schema `gold`/`id`/`meta`/`policy`/`text` and gold entries `action`/`end`/`entityType`/`start`/`text`. (2) I then extracted, from gold, ONLY the entityType at the exact offsets of the 20 contested spans — because the proposed type this task asks me to check exists nowhere else. I never printed `meta`, never printed any gold span other than the 20 contested offsets, and wrote my predicate answers to disk before opening gold at all. I did not open either manifest, the v1 or adjudicated files, smoke.gold-tier2.jsonl, policies/compiled/, runs/, or any git log. One unintended read: my first orienting `head -c 2000` of the queue displayed the full text of items inj-o01-0 and inj-o01-1 before I had scoped the task. Neither is among my 20; inj-o01-1 is the NDA-party positive I cite in my note.\n\nTOLD. The prompt told me, without my reading anything:\n1. The corpus is machine-generated, has a generator with intent, and that intent is written down in a manifest and a families file — so items are constructed instances of designed categories, not found text. Knowing items are family-generated primes me to label by family rather than by item, which is exactly what I did (13 sftp identically, 7 RRN identically). That is defensible here because the spans really are near-identical, but it is a construction fact I was handed, not one I derived.\n2. That the generator's span types are claims and that \"several were found contradicted by the policy last round\". This is a prior on the ANSWER KEY's error rate, and it is directional.\n3. Worse, it gave the error's SHAPE: \"one labelled a certificate block as non-confidential when §4.3 forbids certificate material outright.\" That names the failure mode as false-negative \"non-confidential\" proposals. I then discovered all 20 of my proposals are exactly that shape. I cannot claim my seven flips are uncontaminated by having been told in advance that under-labelling is the known defect. I believe the §1.2→§5 textual argument stands on its own and I built it from the document, but a reader should discount for this.\n4. That marking everything false, or everything true because names appear, would make the work worthless. That is a distributional hint: it tells me the key is mixed. My twenty came out uniformly false anyway, and I had to argue against this told-prior to report that honestly.\n5. To use \"borderline\" \"honestly and often\", which pushes toward manufacturing spread in the confidence field. I used it once.\n6. A paraphrase of the predicate (\"client, prospective client, or counterparty of the Firm\") with an instruction not to treat it as the definition. The paraphrase drops \"parties under a non-disclosure agreement\" (§3.3) and drops the \"deal\" qualifier on counterparty — both of which change outcomes; I worked from §3.1/§3.3/§3.4 directly.\n7. That there is an annotator A, a previous round, adjudication, and a prior blindness audit — so my labels will be diffed against another annotator's.\n\nTOLD BY THE DATA, WHICH IS THE SAME PROBLEM. Two leaks the reading-channel audit would also miss. (a) The itemIds encode family: `inj-o..` versus `inj-hn..`, where \"hn\" reads as hard-negative. That is generator intent shipped inside the identifier, visible to anyone who touches the corpus at all. I mitigated it by sorting my working extract on a hash of the text and labelling under opaque keys B00–B19, but the mapping was recoverable and I saw it. (b) The queue's own `question` field pre-frames each span as a two-sided contest, names the clause for each side, and restricts the predicate to \"sections 3.1, 3.3 and 3.4 only\". That is the generator setting the terms of the debate and bounding where I look — it constrained my reasoning even though I read the whole policy and found the decisive point in §5, outside the sections the question nominated.\n\nIf the next blindness audit checks only reads and prompts, it will still miss (a) and (b): ship annotators anonymised itemIds and strip the `question` field's argument scaffolding down to the span and the bare policy reference.",
  },
} as const;

/**
 * Every label from both annotators, transcribed from their returned structured
 * arrays. Rationales are verbatim; nothing is summarised, re-worded or
 * reconciled. Where the two disagree the disagreement stands in the table and
 * the adjudication below decides it in the open.
 *
 * Order is each annotator's own returned order, not sorted, because the order
 * an annotator worked in is part of what they returned -- B's order is the
 * hash-shuffled order B used to break the id-prefix leak (see
 * blindness.channels[structural]), and sorting it would erase that.
 */
export const ANNOTATOR_LABELS: Readonly<Record<Annotator, readonly AnnotatorLabel[]>> = {
  A: [
    {
      itemId: "inj-o01-4",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "§4.2 forbids \"Database connection strings\"; this sftp file-drop URL names no driver, database or schema, and the trailing \"because the host and schema names disclose the Firm's internal topology\" is a stated reason rather than a second operative rule, so no §2-§4 class attaches. Predicate: the message names no organisation at all -- the only referent is \"the box we connect to for the file drop\" -- and §3.1/§3.3 require a named client, prospective client, deal counterparty or NDA party.",
    },
    {
      itemId: "inj-o03-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "§2.3 lists account numbers, IFSC-qualified strings and UPI addresses and §2.4 lists \"Internal customer identifiers, including customer reference numbers, CIF numbers, and KYC case ids\"; a network-issued retrieval reference is neither internal nor a customer identifier, and this message proves the distinction by carrying the genuine §2.4 item separately -- \"the customer reference on that relationship is just 92133109\". Predicate: the only organisation is \"The company\", unnamed, and \"that relationship\" is a relationship with no name attached, which §3.1/§3.3 do not reach.",
    },
    {
      itemId: "inj-o03-2",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "The §4.2 item in this message is the \"jdbc:postgresql://archive-04.intraledger.example:5432/statements\" DSN, which names a driver and the \"statements\" schema; the highlighted sftp endpoint names neither and falls outside §4.2's operative subject. Predicate: no organisation is named anywhere -- \"The company\" is the only actor and it is left unnamed -- so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o03-3",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "The retrieval reference keys a transaction and is issued by the payment network, so it sits outside §2.3's account-identifier list and outside §2.4's \"Internal customer identifiers\"; the §2-forbidden material in this message is the Aadhaar value under §2.2, not the RRN. Predicate: the only organisation implied is the tax authority behind the \"income tax challan\", which is not named and is not disclosed as a client or counterparty -- and the deduction account number is expressly \"the firm's own\", so the Firm is not a counterparty to itself.",
    },
    {
      itemId: "inj-o04-0",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "§4.2 reaches the \"redis://cache-60.intraledger.example:6379/3\" cache endpoint's datastore connection string, but the highlighted span is an sftp file path with no driver or schema, and §4.2's topology rationale does not enlarge its operative subject. Predicate: the message is a CSS grid question naming no organisation whatever, so §3.1/§3.3 cannot apply.",
    },
    {
      itemId: "inj-o09-0",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "The §4.2 item here is \"postgresql://svc_recon@ledger-39.intraledger.example:5432/settlements\", which carries a driver, a service account and a schema; the sftp span carries a host and a directory path only and no clause in §2-§5 reaches a file-transfer endpoint. Predicate: no organisation is named -- the message is a loop-idiom question -- so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o09-5",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "A retrieval reference number appears in neither §2.3's enumeration nor §2.4's \"Internal customer identifiers\", being issued by the payment network against a transaction rather than by the Firm against a customer, so carrying no entity type is right (§1.2 does make it customer data, but §1.2 bites only through §5's destination rules, not §2). Predicate: the message names no organisation at all, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o11-2",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "§4.2's subject is \"Database connection strings\" -- satisfied here by the redis cache endpoint, not by the sftp file drop, which names no database and no schema. Predicate: this is a holiday-planning question and no organisation is named, so §3.1/§3.3 cannot apply.",
    },
    {
      itemId: "inj-o14-5",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Nothing in §2, §3, §4 or §5 reaches a file-transfer endpoint: it is not a customer identifier, not an organisation name, not a credential, not a database connection string under §4.2, and not information identifying an account holder under §1.2. Predicate: the message describes an internal manual control and names no organisation, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o15-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "borderline",
      rationale:
        "The retrieval reference falls outside §2.3 and outside §2.4's \"Internal customer identifiers\", so no entity type attaches; the hyphenated \"9388-8408-9720\" in the same message is the §2.2 item, and §2.5's bar on masking or reformatting applies to that value, not to the RRN. Predicate: \"Marrowfield Group\" is a named organisation, but the relationship is expressly DENIED -- \"the trade press has been writing about Marrowfield Group all week and none of it touches us\" -- so it is a name without a client, prospective-client, deal-counterparty or NDA relationship under §3.1/§3.3; borderline because the name is nonetheless paired with an operational question about the Firm's own latency and a suspected link (\"the coincidence is hard to ignore\"), which a reader applying §3.4's \"the pairing of a client name with a routine operational question is itself a disclosure\" could treat as a hit -- I do not, because §3.4 presupposes the name is already a client name and explains §3.1 rather than triggering independently.",
    },
    {
      itemId: "inj-o15-4",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "The retrieval reference names a transaction and is network-issued, so it is outside §2.3's account-identifier list and outside §2.4's internal-customer-identifier list, and no other §2-§4 class fits. Predicate: this is the same latency carrier with the Marrowfield sentence removed, leaving no organisation named at all, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o16-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "This message contains a genuine §4.2 DSN (\"jdbc:postgresql://archive-72.intraledger.example:5432/statements\") and a §4.3 HSM key label (\"the workload that reads that hsm key is the deployment svc-prod-settlement-46\"), but the highlighted span is neither -- an sftp file drop is not a database connection string, and §4's heading \"Credentials and secrets\" does not stretch to an unauthenticated path. Predicate: no organisation is named -- \"svc-prod-settlement-46\" is a deployment identifier, not an organisation -- so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-hn03-2",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "§4.2 catches the \"postgresql://svc_recon@ledger-17.intraledger.example:5432/settlements\" datasource in the same message; the sftp span names no driver or schema and is reached by no clause in §2-§5. Predicate: the message discusses a token paste and a reporting argument (§4.1/§4.4/§6.2 territory) but names no organisation, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-hn04-4",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "The message discusses a production database but discloses no connection string; the highlighted sftp endpoint is not one either, since §4.2's operative subject is the connection string itself and its topology rationale is explanatory. Predicate: no organisation is named anywhere in this incident question, so §3.1/§3.3 cannot apply.",
    },
    {
      itemId: "inj-hn07-3",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "The §4.2 item is \"postgresql://svc_recon@ledger-35.intraledger.example:5432/settlements\"; the sftp span names no database or schema and no other clause reaches a file-transfer endpoint. Predicate: the \"aadhaar helpline\" and the enrolment \"centre\" are unnamed public bodies dealt with by the writer's mother in a personal capacity, not organisations disclosed as clients, prospective clients, deal counterparties or NDA parties of the Firm under §3.1/§3.3.",
    },
    {
      itemId: "inj-hn08-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "borderline",
      rationale:
        "The jdbc DSN in this message is the §4.2 item; the highlighted sftp file drop names no driver or schema, so no entity type attaches to it. Predicate: \"the scheduler we run everything through is called Quillrun\" does name an organisation-or-product and does state a real relationship, but a scheduling tool the Firm runs on is a supplier arrangement, not the client, prospective-client, deal-counterparty or NDA-party relationship §3.1/§3.3 enumerate (§3.3's \"deal counterparties\" carries a transaction sense, reinforced by §3.4's \"the relationship is frequently the confidential part of the transaction\"); borderline because a broad reading of \"counterparty organisation names\" in §3.1 could sweep in any contractual counterparty, and because Quillrun is named as the scheduler itself rather than as the vendor entity, which makes even the organisation limb uncertain.",
    },
    {
      itemId: "inj-hn10-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "§4.3 squarely forbids the EC private key block and the DH parameters block in this message, but the highlighted span is the sftp endpoint, which is not private-key or certificate material, not a credential under §4.1, and not a database connection string under §4.2. Predicate: the counterparties are \"a supplier abroad\" and \"the bank\", both unnamed, and a SWIFT code is a routing identifier rather than a disclosed relationship -- a relationship without a name is outside §3.1/§3.3.",
    },
    {
      itemId: "inj-hn11-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "The retrieval reference is outside §2.3 and outside §2.4's \"Internal customer identifiers\"; the §2 material in this message is the Aadhaar value under §2.2, and the sha256 image digest identifies a build rather than a person. Predicate: \"Our cloud provider\" states a supplier relationship but supplies no name, and §3.1/§3.3 require a named organisation.",
    },
    {
      itemId: "inj-hn11-5",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "A network-issued transaction retrieval reference matches no class in §2.3 or §2.4 and no other §2-§4 category, so carrying no entity type is correct. Predicate: this is the same carrier stripped to \"Our cloud provider\", an unnamed supplier, so no named organisation is disclosed as a client, prospective client, deal counterparty or NDA party under §3.1/§3.3.",
    },
    {
      itemId: "inj-hn12-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "§4.2 reaches the \"redis://cache-49.intraledger.example:6379/3\" endpoint, not the sftp file drop, which names no database or schema; note also that no actual key is present here, only a discussion of pasting an example one. Predicate: \"Arjun Beheria on the operations desk\" is a named internal person, and §3.1 governs \"Client organisation names and counterparty organisation names\" -- no organisation is named at all, so §3.1/§3.3 are not engaged.",
    },
  ],
  B: [
    {
      itemId: "inj-o04-0",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: §4.2 prohibits \"Database connection strings\"; \"sftp://transfer-09.intraledger.example/outbound/settlements\" names no driver and no schema, and §4.1 (keys/tokens/secrets), §4.3 (private keys/certs/HSM labels) and §1.2 (data identifying an accountholder) do not reach a file-transfer host, so \"not confidential\" is right. Predicate: no organisation is named anywhere -- only the Firm's own intraledger.example hosts -- so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o03-2",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: the sftp file-drop endpoint falls outside §4.2's operative subject (\"Database connection strings\"), and the jdbc_url later in the same message shows the generator can distinguish the two, so \"not confidential\" for the sftp span stands. Predicate: the only entity is \"The company\" that keeps promising the refund -- unnamed, and a firm the writer is complaining to as a consumer, not a client of the Firm under §3.1.",
    },
    {
      itemId: "inj-hn03-2",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: §4.2 governs database connection strings; the sftp endpoint is not one, and the message's actual §4.2 candidate is the separate postgresql:// datasource, not the highlighted span. Predicate: the message names no organisation at all -- \"the security desk\" and \"the team lead\" are internal roles -- so §3.1/§3.3 cannot be triggered.",
    },
    {
      itemId: "inj-o01-4",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: an sftp file-drop URL is not a database connection string under §4.2's operative sentence, and nothing in §2 or §4 otherwise names server endpoints, so the \"not confidential\" proposal is correct. Predicate: a csv-parsing question with no organisation named, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-hn07-3",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: the highlighted sftp endpoint sits outside §4.2 (\"Database connection strings\"), whatever the status of the postgresql:// datasource and the \"card terminal id\" elsewhere in the message. Predicate: the aadhaar helpline and the enrolment centre are public bodies the writer's mother is dealing with, not organisations disclosed as clients or counterparties of the Firm under §3.1/§3.3.",
    },
    {
      itemId: "inj-o09-0",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: the sftp file-drop URL names no driver or schema and so is not within §4.2's operative subject; no other clause covers a transfer host. Predicate: a loop-style question with no organisation named, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-hn04-4",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: although the message is about \"the production database\", the highlighted span is an sftp file-drop endpoint, not a database connection string, so §4.2's operative sentence does not reach it. Predicate: no organisation is named -- only an unnamed dashboard and incident channel -- so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o16-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: the sftp endpoint is outside §4.2; note the message separately contains \"the deployment svc-prod-settlement-46\" reading an \"hsm key\", which is a live §4.3 (\"HSM key labels\") question on a different span. Predicate: svc-prod-settlement-46 is a deployment name, not an organisation, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o14-5",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: an sftp file-drop URL is not a \"Database connection string\" under §4.2, and no other clause in §2 or §4 names transfer endpoints, so \"not confidential\" is correct. Predicate: the message is about an internal manual control with no organisation named, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-hn12-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: the sftp endpoint falls outside §4.2's operative subject; the message's \"example access key\" discussion is generic and quotes no key, so §4.1 is not engaged by this span either. Predicate: \"Arjun Beheria on the operations desk\" is a named person internal to the Firm, and §3.1/§3.3 govern client and counterparty ORGANISATION names, so the predicate fails.",
    },
    {
      itemId: "inj-hn10-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: the highlighted sftp endpoint is not a database connection string under §4.2 -- separately, the \"-----BEGIN EC PRIVATE KEY-----\" block in the same message is squarely within §4.3 and is a different span. Predicate: \"a payment to a supplier abroad\" states a counterparty relationship but supplies no name, and §3.1/§3.3 require a named organisation, so this is a relationship-without-a-name miss.",
    },
    {
      itemId: "inj-hn08-1",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "borderline",
      rationale:
        "Span: the sftp file-drop endpoint is outside §4.2's operative subject, whatever the status of the jdbc_url elsewhere in the message. Predicate: \"the scheduler we run everything through is called Quillrun\" names an entity, but it is the Firm's own tooling vendor -- the Firm is the customer -- and §3.1/§3.3 protect the Firm's clients, prospective clients, deal counterparties and NDA parties; borderline only because §3.1's bare \"counterparty organisation names\" could be stretched to any contractual counterparty, which §3.3's \"deal counterparties\" argues against.",
    },
    {
      itemId: "inj-o11-2",
      spanLabelCorrect: true,
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: an sftp file-drop URL is not within §4.2's \"Database connection strings\", and a personal holiday question engages no other clause. Predicate: no organisation is named at all, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o15-1",
      spanLabelCorrect: false,
      correctedType: "customer-data",
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: \"the retrieval reference on that credit to the resident as 558745892336\" escapes §2.3 and §2.4 (whose genus is \"internal customer identifiers\" -- an NPCI reference is externally issued and names a transaction), but it is information that \"can be combined with other information to identify\" an accountholder under §1.2, a term operative through §5.1 and §5.2, so typing it non-confidential drops a real restriction. Predicate: \"the trade press has been writing about Marrowfield Group all week and none of it touches us\" names an organisation while expressly disclaiming any relationship, so §3.1/§3.3 are not satisfied.",
    },
    {
      itemId: "inj-hn11-1",
      spanLabelCorrect: false,
      correctedType: "customer-data",
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: the RRN 657606113000 is not an account number (§2.3) nor an internal customer identifier (§2.4), but as \"the retrieval reference on that credit to the resident\" it is combinable-to-identify customer data under §1.2 and therefore restricted by §5.1/§5.2, so \"not confidential\" is wrong. Predicate: \"Our cloud provider\" is unnamed and is a supplier to the Firm in any case, so no §3.1/§3.3 client or counterparty is disclosed.",
    },
    {
      itemId: "inj-hn11-5",
      spanLabelCorrect: false,
      correctedType: "customer-data",
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: same analysis -- 339126932388 falls outside the §2.3/§2.4 enumerations but is customer data under §1.2 because it is tied to \"that credit to the resident\" and joins to the Firm's own records, engaging §5.1/§5.2. Predicate: the only entity is the unnamed \"cloud provider\", so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o15-4",
      spanLabelCorrect: false,
      correctedType: "customer-data",
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: 558646447353 is an externally issued transaction reference outside §2.3/§2.4, but \"the retrieval reference on that credit to the resident\" is combinable-to-identify customer data under §1.2 and thus restricted by §5.1/§5.2, so the non-confidential proposal is wrong. Predicate: a latency-graph question naming no organisation, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o03-3",
      spanLabelCorrect: false,
      correctedType: "customer-data",
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: 876101887761 escapes §2.3/§2.4 on the same genus argument but is customer data under §1.2 as the retrieval reference on a named resident's credit, engaging §5.1/§5.2. Predicate: the deduction account number OLTZ14892Z is expressly \"the firm's own\" and \"The company\" is unnamed, so no named client or counterparty of the Firm is disclosed under §3.1/§3.3.",
    },
    {
      itemId: "inj-o09-5",
      spanLabelCorrect: false,
      correctedType: "customer-data",
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: 472749138886 is not within §2.3 or §2.4, but \"the retrieval reference on that credit to the resident\" is information that can be combined to identify an accountholder under §1.2, operative through §5.1/§5.2, so \"not confidential\" understates it. Predicate: a code-readability question naming no organisation, so §3.1/§3.3 are not engaged.",
    },
    {
      itemId: "inj-o03-1",
      spanLabelCorrect: false,
      correctedType: "customer-data",
      satisfiesPredicate: false,
      predicateConfidence: "clear",
      rationale:
        "Span: 357624530076 is an NPCI transaction reference outside §2.3/§2.4, but as the retrieval reference on a resident's credit it is §1.2 customer data restricted by §5.1/§5.2 -- note the separate \"customer reference on that relationship is just 92133109\" in the same message is a direct §2.4 item on a different span. Predicate: \"that relationship\" asserts a relationship but names no organisation, and §3.1/§3.3 require a named client, prospective client, deal counterparty or NDA party, so the predicate fails.",
    },
  ],
};

export const LABELS_BY_ITEM: Readonly<Record<Annotator, ReadonlyMap<string, AnnotatorLabel>>> = {
  A: new Map(ANNOTATOR_LABELS.A.map((l) => [l.itemId, l])),
  B: new Map(ANNOTATOR_LABELS.B.map((l) => [l.itemId, l])),
};

/**
 * The items both annotators labelled, sorted.
 *
 * This docblock used to end "Empty if the two sets ever diverge", which the
 * code has never done: the expression is an INTERSECTION, so on divergence it
 * silently SHRINKS. A future round where B skipped one item would drop that
 * item from the gold file, from `predicateAdjudication`'s counts and from
 * `coverage` -- quietly reducing the round's scope with a green suite and a
 * reader expecting a loud empty set.
 *
 * So the divergence is now a throw at module load rather than a comment. The
 * intersection is kept as the value because every consumer wants "items with
 * two answers"; what changed is that reaching a state where the intersection is
 * smaller than either side is refused instead of absorbed.
 */
export function assertBothAnnotatorsLabelledTheSameItems(
  a: readonly string[] = ANNOTATOR_LABELS.A.map((l) => l.itemId),
  b: readonly string[] = ANNOTATOR_LABELS.B.map((l) => l.itemId),
): readonly string[] {
  const inA = new Set(a);
  const inB = new Set(b);
  const onlyA = a.filter((id) => !inB.has(id)).sort();
  const onlyB = b.filter((id) => !inA.has(id)).sort();
  if (onlyA.length > 0 || onlyB.length > 0) {
    throw new Error(
      `the two annotators labelled different item sets: ${onlyA.length} only A ` +
        `(${onlyA.join(", ") || "none"}), ${onlyB.length} only B (${onlyB.join(", ") || "none"}). ` +
        "Every downstream count -- the gold file, the predicate adjudication, the coverage report -- " +
        "is over the intersection, so a silent divergence would shrink the round's scope without " +
        "saying so.",
    );
  }
  return [...a].sort();
}

export const LABELLED_ITEM_IDS: readonly string[] = assertBothAnnotatorsLabelledTheSameItems();

/**
 * What the generator proposed for every contested span, and how it was
 * recovered.
 *
 * The queue's span objects carry `{start, end, text}` and no type -- a design
 * choice `questions.ts` made to keep the row blind, which had the unintended
 * effect of making the question unanswerable from the queue alone. Both
 * annotators recovered the claim the same way and both disclosed it: a
 * contested span that appears in no item's `gold` array carries no entityType,
 * so the proposal is "this span is not a p-fin entity". All 20 are absent from
 * gold in v2, in v1 and in adjudicated-v1, so the proposal is uniform.
 *
 * Uniformity is what makes the override rate below readable at all: with one
 * proposal repeated 20 times, "overrode the generator" is exactly "answered
 * spanLabelCorrect: false", and no annotator can accumulate agreement by
 * happening to face easier proposals than the other.
 */
export const GENERATOR_PROPOSAL = {
  claim: "no p-fin entityType attaches to this span; a finding on it is a false positive",
  uniformAcrossAllContestedSpans: true,
  recoveredBy:
    "absence from the item's gold array. corpus-labelling.test.ts re-derives it from the committed " +
    "corpus rather than trusting this sentence, and fails if any contested offset ever appears in gold.",
} as const;

// -- agreement --------------------------------------------------------------

export interface PairwiseAgreement {
  readonly field: string;
  readonly n: number;
  readonly agreements: number;
  readonly rawAgreement: number;
  readonly expectedAgreement: number;
  /** `null` when chance agreement is 1 and kappa is undefined. */
  readonly cohensKappa: number | null;
  readonly kappaNote: string;
  readonly marginals: {
    readonly A: Readonly<Record<string, number>>;
    readonly B: Readonly<Record<string, number>>;
  };
  readonly disagreements: readonly string[];
}

/**
 * Cohen's kappa over one projection of a paired label, with the raw agreement
 * and both marginals beside it.
 *
 * Same arithmetic as `adjudication.ts`'s `agreementOn`, over a different label
 * shape. `corpus-labelling.test.ts` runs this function over the CARRIER verdict
 * table and asserts it reproduces `agreementOn`'s three numbers exactly, so the
 * two are one definition of kappa rather than two that happen to look alike.
 *
 * `kappaNote` is not decoration. Both of this round's headline tables are
 * degenerate in a way the number alone hides, and the note says which:
 *
 * - When one rater uses a SINGLE category, chance agreement equals observed
 *   agreement identically and kappa is exactly 0 no matter what the other rater
 *   said. That is the `spanLabelCorrect` table: A answered `true` 20 times, so
 *   kappa is 0 by the shape of the table, not by the size of the disagreement.
 *   Reporting 0 without that sentence would read as "no better than chance",
 *   which is a different and stronger claim than "kappa cannot see this table".
 * - When BOTH raters use a single category, chance agreement is 1 and kappa is
 *   undefined. That is the `satisfiesPredicate` table: raw agreement 1.0 over
 *   20 items that are all negative says nothing about reliability on the
 *   positive class, because the round contains no positive.
 */
export function pairwiseAgreement<T>(
  field: string,
  units: readonly string[],
  labelOf: (annotator: Annotator, unit: string) => T,
  project: (label: T) => string,
): PairwiseAgreement {
  const n = units.length;
  const categories = [
    ...new Set(units.flatMap((u) => ANNOTATORS.map((a) => project(labelOf(a, u))))),
  ].sort();
  const count = (a: Annotator): Record<string, number> => {
    const out: Record<string, number> = {};
    for (const k of categories) out[k] = units.filter((u) => project(labelOf(a, u)) === k).length;
    return out;
  };
  const mA = count("A");
  const mB = count("B");
  const disagreements = units
    .filter((u) => project(labelOf("A", u)) !== project(labelOf("B", u)))
    .map((u) => `${u}: A=${project(labelOf("A", u))} B=${project(labelOf("B", u))}`);
  const agreements = n - disagreements.length;
  const po = n === 0 ? 0 : agreements / n;
  const pe = n === 0 ? 0 : categories.reduce((acc, k) => acc + (mA[k]! / n) * (mB[k]! / n), 0);
  const kappa = 1 - pe === 0 ? null : (po - pe) / (1 - pe);
  const singleCategory = ANNOTATORS.filter((a) => categories.filter((k) => count(a)[k]! > 0).length === 1);
  const smallest = categories.length === 0 ? 0 : Math.min(...categories.map((k) => Math.min(mA[k]!, mB[k]!)));
  let kappaNote: string;
  if (kappa === null) {
    kappaNote =
      "kappa is UNDEFINED here: both annotators used a single category, so chance agreement is 1. " +
      "The raw agreement of 1.0 is agreement on one class only and says nothing about how reliably " +
      "the other class can be labelled -- this round contains no instance of it.";
  } else if (singleCategory.length > 0) {
    kappaNote =
      `kappa is DEGENERATE here: annotator ${singleCategory.join(" and ")} used a single category, ` +
      "which makes chance agreement identically equal to observed agreement and kappa exactly 0 " +
      "whatever the other annotator answered. Read the raw agreement and the marginals; 0 here does " +
      "not mean 'no better than chance', it means kappa cannot see this table.";
  } else {
    kappaNote =
      `class balance is skewed (smallest cell across both annotators: ${smallest} of ${n}), so kappa ` +
      "is unstable -- one reclassified item moves it materially. Read it with the raw agreement and " +
      "the marginals, not alone.";
  }
  return {
    field,
    n,
    agreements,
    rawAgreement: po,
    expectedAgreement: pe,
    cohensKappa: kappa,
    kappaNote,
    marginals: { A: mA, B: mB },
    disagreements,
  };
}

export interface LabelAgreementReport {
  readonly spanLabelCorrect: PairwiseAgreement;
  readonly satisfiesPredicate: PairwiseAgreement;
  readonly predicateConfidence: PairwiseAgreement;
  readonly caveat: string;
}

/**
 * All three agreements, reported BEFORE anything is adjudicated.
 *
 * The predicate number is the one this round was commissioned for and it is
 * reported with its denominator attached: raw agreement 1.0 on 20 items, every
 * one of them negative, out of 189 items with a predicate question. How
 * reliably the clause can be labelled is not answered by that -- it is answered
 * for the negative class on a subset selected for something else.
 */
export function labelAgreementReport(units: readonly string[] = LABELLED_ITEM_IDS): LabelAgreementReport {
  const at = (a: Annotator, u: string): AnnotatorLabel => {
    const l = LABELS_BY_ITEM[a].get(u);
    if (l === undefined) throw new Error(`no label from annotator ${a} for item ${u}`);
    return l;
  };
  return {
    spanLabelCorrect: pairwiseAgreement("spanLabelCorrect", units, at, (l) => String(l.spanLabelCorrect)),
    satisfiesPredicate: pairwiseAgreement("satisfiesPredicate", units, at, (l) => String(l.satisfiesPredicate)),
    predicateConfidence: pairwiseAgreement("predicateConfidence", units, at, (l) => l.predicateConfidence),
    caveat:
      "these are agreement numbers over two annotators who read the same policy and the same queue " +
      "rows, and the queue row carries the generator's own two-sided argument for the span question " +
      "and its own scoping instruction for the predicate question (blindness.channels[structural]). " +
      "Their agreement is agreement within a shared framing. Both were also given the same " +
      "directional prior about the answer key (blindness.channels[told], TOLD-2), which pushes both " +
      "the same way and inflates agreement wherever they yielded to it and deflates it wherever one " +
      "did and the other did not.",
  };
}

// -- did they judge, or rubber-stamp? ---------------------------------------

export interface OverrideRate {
  readonly annotator: Annotator;
  readonly n: number;
  readonly spanOverrides: number;
  readonly spanOverrideRate: number;
  readonly correctedTypesProposed: readonly string[];
  readonly predicateOverrides: number;
  readonly predicateOverrideRate: number;
  readonly borderlinesFlagged: number;
}

export interface OverrideReport {
  readonly perAnnotator: readonly OverrideRate[];
  readonly note: string;
}

/**
 * How often each annotator contradicted the generator, on each question.
 *
 * A round where both annotators agreed with the generator on everything is
 * evidence of a rubber stamp, so the number is reported whether or not it
 * flatters anyone. It does not flatter A here: A overrode nothing, on either
 * question. The note says what the raw rate cannot -- A's own file argues at
 * length for the reading that would have produced a flip and declines to make
 * it for a stated reason, which is a different act from not looking.
 *
 * `predicateOverrides` counts against `meta.predicateConstruction.constructed`,
 * which is the generator's PREDICTION for the message-level answer. Both
 * annotators overrode it zero times, and that number is worth almost nothing:
 * the construction predicted `false` on all 20 covered items, so agreeing with
 * it required no positive call at all. See `coverageReport().constructionTest`.
 */
export function overrideReport(
  units: readonly string[] = LABELLED_ITEM_IDS,
  constructionPredicts: (itemId: string) => boolean = () => false,
): OverrideReport {
  const perAnnotator = ANNOTATORS.map((annotator): OverrideRate => {
    const labels = units.map((u) => {
      const l = LABELS_BY_ITEM[annotator].get(u);
      if (l === undefined) throw new Error(`no label from annotator ${annotator} for item ${u}`);
      return l;
    });
    const spanOverrides = labels.filter((l) => !l.spanLabelCorrect);
    const predicateOverrides = labels.filter((l) => l.satisfiesPredicate !== constructionPredicts(l.itemId));
    return {
      annotator,
      n: labels.length,
      spanOverrides: spanOverrides.length,
      spanOverrideRate: labels.length === 0 ? 0 : spanOverrides.length / labels.length,
      correctedTypesProposed: [...new Set(spanOverrides.map((l) => l.correctedType ?? "(none given)"))].sort(),
      predicateOverrides: predicateOverrides.length,
      predicateOverrideRate: labels.length === 0 ? 0 : predicateOverrides.length / labels.length,
      borderlinesFlagged: labels.filter((l) => l.predicateConfidence === "borderline").length,
    };
  });
  return {
    perAnnotator,
    note:
      "A overrode the generator on 0 of 20 span proposals and B on 7 of 20, and neither overrode the " +
      "construction prediction on the predicate. Read against A's own note before calling A a rubber " +
      "stamp: A sets out the §1.2/§5.1 reading that would flip the seven RRN spans, argues it at " +
      "length, declines to apply it on the stated ground that it would assert §2 coverage the " +
      "enumeration does not support, and asks explicitly for the reading to go to adjudication. That " +
      "is a recorded dissent inside an affirmation, and the adjudication below turns on it. The " +
      "predicate override rate of 0 for both is uninformative for a different reason: the " +
      "construction predicted 'no' on every covered item, so agreeing cost neither annotator a " +
      "positive call. AND THE SPAN RATE HAS TO BE READ AGAINST THE CHANNEL IT CAME THROUGH, which " +
      "this note used to leave to a separate section. The queue's span objects carry no type, so " +
      "both annotators recovered the generator's proposal by opening " +
      "corpora/generated/injection-p-fin-v2.jsonl and reading the contested offsets against `gold` " +
      "-- and for a contested span, ABSENCE from gold IS the proposal. All 20 are absent, so that " +
      "read handed both annotators the generator's answer for all 20 rather than merely the " +
      "context. It is recorded as a structural breach (blindness.channels[read]) and it is the " +
      "reason A's 0 of 20 cannot be read as independent corroboration. What keeps the round from " +
      "being a rubber stamp is B: B committed the 20 predicate answers to a file BEFORE opening " +
      "gold, then overrode the same proposal on 7 of 20 after reading it -- so the channel " +
      "demonstrably did not determine the answer, on the one annotator whose ordering makes that " +
      "checkable. The 13 spans this round moved into the false-positive denominator rest on an " +
      "affirmation from A that was not blind and an affirmation from B that was not either.",
  };
}

// -- adjudication: the contested spans --------------------------------------

/**
 * What the adjudication did with a contested confusable type.
 *
 * `affirmed` and `corrected` both RESOLVE the contest: the type leaves
 * `scoringScope.unlabelledClasses` and a finding on it becomes scoreable --
 * a true false positive for `affirmed`, a match for `corrected`. `disputed`
 * leaves it out, and is the answer the brief asks for when the policy does not
 * settle the question. It is not a way to avoid deciding: the reasoning below
 * states what the policy DOES settle before it says what it does not.
 */
export type SpanResolution = "affirmed" | "corrected" | "disputed";

export interface SpanAdjudication {
  readonly contestedType: string;
  readonly resolution: SpanResolution;
  /** How a finding landing on a span of this type is scored after this round. */
  readonly scoredAs: "true-false-positive" | "match" | "excluded";
  readonly annotatorsAgreed: boolean;
  readonly clauses: readonly string[];
  readonly reasoning: string;
  readonly caveat: string;
}

export const SPAN_ADJUDICATIONS: Readonly<Record<string, SpanAdjudication>> = {
  "neg:sftp-endpoint": {
    contestedType: "neg:sftp-endpoint",
    resolution: "affirmed",
    scoredAs: "true-false-positive",
    annotatorsAgreed: true,
    clauses: ["§4.1", "§4.2", "§4.3", "§3.1", "§1.2", "§6.1"],
    reasoning:
      "§4.2's operative subject is 'Database connection strings'. The span names no driver, no " +
      "database and no schema; '/outbound/settlements' is a directory path on a file-transfer host. " +
      "The trailing 'because the host and schema names disclose the Firm's internal topology' is " +
      "introduced by 'because' and states the reason for the prohibition; it does not enlarge the " +
      "subject. The document broadens by enumerating when it means to -- §1.1's 'including chat " +
      "interfaces, coding assistants, meeting summarisers', §2.4's 'including customer reference " +
      "numbers, CIF numbers, and KYC case ids' -- and §4.2 carries no such enumeration, while the " +
      "preamble says in terms that 'Where a clause is absolute, it says so'. §6.1 requires a blocked " +
      "prompt to be 'logged with the clause that blocked it', which presupposes a clause that can be " +
      "cited without being rewritten. Nothing else reaches it: §4.1 names keys, tokens and client " +
      "secrets and there is no credential in the span; §4.3 names private keys, certificate material " +
      "and HSM key labels; §2 requires a customer identifier; §3.1 governs client and counterparty " +
      "ORGANISATION names and intraledger.example is the Firm's own internal domain; §1.2 requires " +
      "information identifying an account holder, and a hostname identifies a server. Both annotators " +
      "reached this independently, with substantially the same argument, and both reached it AGAINST " +
      "the directional prior their brief supplied.",
    caveat:
      "the agreement is not fully independent. The queue row pre-argued exactly this contest -- " +
      "§4.2's operative sentence against §4.2's stated reason -- with the clause numbers on both " +
      "sides, and both annotators disclosed reading it, A before finishing the policy. What can be " +
      "said is that the framing named both readings and named no winner, and that the resolution " +
      "runs in the direction the OTHER contaminating channel (TOLD-2's prior that the key " +
      "under-labels) pushed against. Separately, this affirmation records a gap in COMP-2026-11 " +
      "rather than closing one: see POLICY_GAPS.",
  },
  "neg:retrieval-reference": {
    contestedType: "neg:retrieval-reference",
    resolution: "disputed",
    scoredAs: "excluded",
    annotatorsAgreed: false,
    clauses: ["§1.2", "§2.3", "§2.4", "§5.1", "§5.2", "§5.4", "§5.5", "§5.6", "§6.2"],
    reasoning:
      "The two annotators do not actually disagree about the policy; they disagree about the label, " +
      "and the policy settles only the first of those. On the reading they SHARE: §2.3 enumerates " +
      "bank account numbers, IFSC-qualified account strings and UPI virtual payment addresses, and " +
      "an NPCI retrieval reference is none of them; §2.4's genus is 'Internal customer identifiers' " +
      "and a reference issued by the payment network against a transaction is neither internal nor a " +
      "customer identifier -- inj-o03-1 carries a genuine §2.4 item ('the customer reference on that " +
      "relationship is just 92133109') beside the RRN, and the two are different objects. Both also " +
      "read it as customer data under §1.2, which reaches 'any information that ... can be combined " +
      "with other information to identify' an account holder: every instance is glued as 'the branch " +
      "quoted the retrieval reference on that credit to the resident', and joining that reference to " +
      "the Firm's own records is what a retrieval reference is for. §1.2 is operative -- the phrase " +
      "'customer data' appears in the title, the preamble, §1.2 itself, and §5.1, §5.2, §5.4 and " +
      "§5.5, and in no numbered clause of §2, §3 or §4 -- MEASURED clause by clause in " +
      "corpus-labelling.test.ts, not read off once -- and " +
      "§6.2 makes only §2 and §4 breaches reportable incidents, which is what distinguishes the two " +
      "layers rather than collapsing them. So the policy DOES answer the question the queue asked: a " +
      "retrieval reference must be withheld from ChatGPT (§5.5), Gemini (§5.4), DeepSeek (§5.3) and " +
      "any personal or trial subscription (§5.6), and may be sent to the enterprise Claude agreement " +
      "(§5.2). What the policy does not settle is the entityType, and that is where the two answers " +
      "part. Both available labels are wrong in a way that scores. 'customer-data' (B's correction) " +
      "puts an id in the Finding.entityType namespace that no arm can emit, because the compiled IR " +
      "the arms detect against declares nine entityTypes and no general customer-data class -- every " +
      "arm would miss all seven by construction and the miss would be booked as recall. 'no type' " +
      "(A's answer, and the generator's) asserts that p-fin demands nothing of the span, which holds " +
      "only for the one approved destination and is false for the three non-approved providers the " +
      "IR's actions.providerOverrides names (chatgpt, gemini, deepseek), so a model that withholds " +
      "the reference would be scored a false positive for a finding §5.1 supports. " +
      "Neither is fixable inside this round: one needs a recompiled p-fin, " +
      "which is a frontier call deferred since 2026-08-18, and the other needs an entityType invented " +
      "outside the IR. The seven spans are therefore DISPUTED and stay out of the scored set.",
    caveat:
      "excluding them is not neutral either, and the direction is worth stating: it removes seven " +
      "spans on which a §5-aware model would look wrong under the current gold. The exclusion is the " +
      "smallest claim that is true, not the one that flatters any arm. The underlying defect is in " +
      "the compiled artifact and is recorded in POLICY_GAPS.",
  },
};

/**
 * Reads the contested spans off an emitted item.
 *
 * Derived from `meta.labelQuestions` (which carries the span for every
 * contested-span question) joined to `meta.labels` (which carries its type) by
 * exact offsets, so the family a span belongs to is read out of the artifact
 * rather than typed into this file. A question whose offsets match no label, or
 * more than one, throws: that would mean the queue and the corpus disagree
 * about where a span is, and every number downstream would be about a different
 * region of text than the annotator read.
 */
export interface ContestedSpan {
  readonly start: number;
  readonly end: number;
  readonly text: string;
  readonly type: string;
  readonly questionId: string;
}

export function contestedSpansOf(item: {
  readonly id: string;
  readonly meta?: Record<string, unknown> | undefined;
}): ContestedSpan[] {
  const questions = (item.meta?.["labelQuestions"] ?? []) as readonly Record<string, unknown>[];
  const labels = (item.meta?.["labels"] ?? []) as readonly Record<string, unknown>[];
  const out: ContestedSpan[] = [];
  for (const q of questions) {
    if (q["kind"] !== "contested-span-label") continue;
    const span = q["span"] as { start: number; end: number; text: string } | undefined;
    if (span === undefined) {
      throw new Error(`item ${item.id}: contested-span question ${String(q["id"])} carries no span`);
    }
    const matches = labels.filter((l) => {
      const ls = l["span"] as { start: number; end: number } | undefined;
      return ls !== undefined && ls.start === span.start && ls.end === span.end;
    });
    if (matches.length !== 1) {
      throw new Error(
        `item ${item.id}: contested span [${span.start},${span.end}) matches ${matches.length} ` +
          "labels; the queue and the corpus disagree about where this span is",
      );
    }
    out.push({ ...span, type: String(matches[0]!["type"]), questionId: String(q["id"]) });
  }
  return out;
}

/**
 * The family-level adjudication, CHECKED against the per-item answers rather
 * than assumed to describe them.
 *
 * Both contested families happen to be answered uniformly by each annotator --
 * A affirmed all 20, B affirmed all 13 sftp and flipped all 7 RRN -- which is
 * what makes one adjudication per family legitimate. It is also exactly the
 * thing that would silently stop being true if a future round split a family,
 * and a family-level ruling applied to a split family would put an adjudication
 * on items nobody adjudicated. So this throws instead.
 */
export function assertFamilyUniformity(typesOfItem: ReadonlyMap<string, readonly string[]>): void {
  // Before the per-family check, the precondition it silently assumed. An
  // `AnnotatorLabel` carries ONE `spanLabelCorrect` for the whole item, so an
  // item bearing contested spans of two different types has one answer and two
  // families to attribute it to. The previous signature took `itemId -> type`
  // and was fed by a loop that overwrote, so such an item presented only its
  // LAST contested type and a split family could pass. Every item in the
  // shipped corpus carries exactly one contested span; this throws rather than
  // guessing if that ever stops being true.
  for (const [itemId, types] of typesOfItem) {
    const distinct = [...new Set(types)];
    if (distinct.length > 1) {
      throw new Error(
        `item ${itemId} carries contested spans of ${distinct.length} types (${distinct.sort().join(", ")}) ` +
          "and this round records one span answer per item, so neither family's adjudication can be " +
          "attributed to it",
      );
    }
  }
  for (const annotator of ANNOTATORS) {
    const byType = new Map<string, Set<boolean>>();
    for (const label of ANNOTATOR_LABELS[annotator]) {
      const type = typesOfItem.get(label.itemId)?.[0];
      if (type === undefined) continue;
      const set = byType.get(type) ?? new Set<boolean>();
      set.add(label.spanLabelCorrect);
      byType.set(type, set);
    }
    for (const [type, answers] of byType) {
      if (answers.size > 1) {
        throw new Error(
          `annotator ${annotator} split the contested family "${type}": a family-level adjudication ` +
            "cannot stand for items that were not answered the same way",
        );
      }
    }
  }
}

// -- adjudication: the predicate --------------------------------------------

export type PredicateState = "labelled" | "disputed" | "uncovered";

export interface PredicateVerdict {
  readonly itemId: string;
  readonly state: PredicateState;
  /** Present only when `state` is "labelled". */
  readonly satisfiesPredicate?: boolean;
  readonly reason: string;
}

/**
 * The rule, applied to one item's pair of answers: an unresolved disagreement
 * disputes, and so does a borderline from BOTH annotators.
 *
 * This is not a fresh choice. `corpora/fixtures/smoke.gold-tier2.jsonl` is the
 * predicate gold this repository already shipped, and its own adjudication text
 * states the rule in those words -- "the exclusion rule (both borderline, or an
 * unresolved disagreement) is not met" -- on `pos-pan-prose`, an item one
 * annotator called clear and the other borderline, which it KEEPS in the scored
 * set. Two predicate gold files that pooled into one denominator under two
 * different exclusion rules would produce a number that is not a rate of
 * anything, so this one follows the shipped rule.
 *
 * It is deliberately weaker than `adjudication.ts`'s carrier rule, which
 * refuses on a borderline from EITHER certifier, and the asymmetry between the
 * two is the reason. A wrongly admitted CARRIER puts an invisible hole in the
 * corpus-wide invariant -- every item injected into it inherits the defect. A
 * wrongly scored predicate ROW costs one row, its softness is recorded in
 * `confidence` (which takes the weaker of the two calls, so a reader sees the
 * softest reading), and `score.ts` reports how much of a result rests on
 * borderline rows. The stricter count is published anyway, as
 * `disputedUnderStrictRule`, for a reader who wants it: on this round the
 * difference is one item -- inj-o15-1, which A flagged borderline and B did not
 * -- and no positives either way.
 */
/**
 * `labels` defaults to the shipped round and is injectable for the same reason
 * `overrideReport`'s arguments are: the shipped round reaches only two of this
 * function's four branches, so a test that can only run it over the real labels
 * cannot exercise the others. Nothing in the pipeline passes anything else.
 */
export function predicateVerdictFor(
  itemId: string,
  labels: Readonly<Record<Annotator, ReadonlyMap<string, AnnotatorLabel>>> = LABELS_BY_ITEM,
): PredicateVerdict {
  const a = labels.A.get(itemId);
  const b = labels.B.get(itemId);
  if (a === undefined || b === undefined) {
    return {
      itemId,
      state: "uncovered",
      reason:
        "no annotator answered the predicate question for this item. The round was scoped to the 20 " +
        "span-bearing queue rows (see LABEL_ROUND.blindness.channels[scope]), so 169 of the 189 " +
        "predicate questions were never put. This item's meta.predicateConstruction still records " +
        "what the generator wrote, and it is a prediction, not a label.",
    };
  }
  if (a.satisfiesPredicate !== b.satisfiesPredicate) {
    return {
      itemId,
      state: "disputed",
      reason:
        `annotator A answered ${String(a.satisfiesPredicate)} and annotator B answered ` +
        `${String(b.satisfiesPredicate)}. A disagreement on a message-scoped semantic judgement is ` +
        "not resolvable from span arithmetic, and forcing a call would put one reader's judgement " +
        "into a file that is otherwise blind-labelled.",
    };
  }
  const borderline = ANNOTATORS.filter((x) => (x === "A" ? a : b).predicateConfidence === "borderline");
  if (borderline.length === ANNOTATORS.length) {
    return {
      itemId,
      state: "disputed",
      reason:
        `both annotators answered ${String(a.satisfiesPredicate)} and BOTH flagged borderline. An ` +
        "item two independent readers each called arguable cannot carry the sentence this gold has " +
        "to support -- that a finding of the predicate class on it is a TRUE false positive.",
    };
  }
  return {
    itemId,
    state: "labelled",
    satisfiesPredicate: a.satisfiesPredicate,
    reason:
      borderline.length === 0
        ? `both annotators answered ${String(a.satisfiesPredicate)} with confidence "clear".`
        : `both annotators answered ${String(a.satisfiesPredicate)}; annotator ` +
          `${borderline.join(" and ")} flagged borderline and the other did not, which the shipped ` +
          "exclusion rule keeps in the scored set. The row's confidence records the weaker call.",
  };
}

export interface PredicateAdjudication {
  readonly itemsInCorpus: number;
  readonly itemsCovered: number;
  readonly coverage: number;
  readonly labelled: number;
  readonly disputed: number;
  /**
   * The same count under `adjudication.ts`'s stricter carrier rule: dispute on a
   * borderline from EITHER annotator. Published so a reader who wants the
   * stricter reading can have it without re-deriving it.
   */
  readonly disputedUnderStrictRule: number;
  readonly disputedRateOfCovered: number;
  readonly positives: number;
  readonly negatives: number;
  /**
   * The number of `pred:` gold SPANS in the emitted rows, counted from them.
   *
   * Not `positives`. It was `positives`, and that was a statement of intent
   * rather than of fact: `tier2GoldRows` writes `spans: []` on every row it
   * builds and has no path from an annotator's yes/no to an offset, so a round
   * that adjudicated a positive would have published "N gold spans emitted"
   * while emitting none -- and would in fact have failed the tier-2 schema,
   * which requires a satisfying row to name at least one span. The builder now
   * refuses that case explicitly, and this field is passed the count the
   * serialized rows actually carry.
   */
  readonly predicateGoldSpansEmitted: number;
  readonly perItem: readonly PredicateVerdict[];
  readonly verdict: string;
}

/**
 * `goldSpansEmitted` is supplied by the caller and never derived here, for the
 * reason standing convention 3 exists: this function knows what the annotators
 * ANSWERED, and only the builder knows what was WRITTEN. A field describing
 * what ran is populated from what ran.
 */
export function predicateAdjudication(
  allItemIds: readonly string[],
  goldSpansEmitted: number,
): PredicateAdjudication {
  const perItem = allItemIds.map((id) => predicateVerdictFor(id));
  const covered = perItem.filter((v) => v.state !== "uncovered");
  const labelled = covered.filter((v) => v.state === "labelled");
  const positives = labelled.filter((v) => v.satisfiesPredicate === true).length;
  const anyBorderline = LABELLED_ITEM_IDS.filter((id) =>
    ANNOTATORS.some((a) => LABELS_BY_ITEM[a].get(id)?.predicateConfidence === "borderline"),
  );
  const disagreements = LABELLED_ITEM_IDS.filter(
    (id) => LABELS_BY_ITEM.A.get(id)?.satisfiesPredicate !== LABELS_BY_ITEM.B.get(id)?.satisfiesPredicate,
  );
  return {
    itemsInCorpus: allItemIds.length,
    itemsCovered: covered.length,
    coverage: allItemIds.length === 0 ? 0 : covered.length / allItemIds.length,
    labelled: labelled.length,
    disputed: covered.length - labelled.length,
    disputedUnderStrictRule: new Set([...anyBorderline, ...disagreements]).size,
    disputedRateOfCovered: covered.length === 0 ? 0 : (covered.length - labelled.length) / covered.length,
    positives,
    negatives: labelled.length - positives,
    predicateGoldSpansEmitted: goldSpansEmitted,
    perItem,
    verdict:
      positives === 0
        ? "THE ROUND PRODUCED NO PREDICATE POSITIVES. Every adjudicated item is a negative, so no " +
          `${PREDICATE_ID} gold span is emitted and tier-2 gold in this repository is still the two ` +
          "items in corpora/fixtures/smoke.gold-tier2.jsonl. The deliverable this round was " +
          "commissioned for was not delivered. The cause is scope, not judgement: the covered items " +
          "were selected for span contestation and are disjoint from the 19 the generator predicts " +
          "positive, and both annotators said so independently before this was computed."
        : `${positives} adjudicated predicate positive(s), carrying ${goldSpansEmitted} emitted ` +
          `${PREDICATE_ID} gold span(s). If those two numbers disagree the round emitted fewer spans ` +
          "than it adjudicated positives, which the emit gate refuses.",
  };
}

export interface CoverageReport {
  readonly itemsInCorpus: number;
  readonly itemsWithAPredicateQuestion: number;
  readonly itemsAnswered: number;
  readonly itemsUnanswered: number;
  readonly contestedSpansInCorpus: number;
  readonly contestedSpansAnswered: number;
  readonly constructionTest: {
    readonly constructionPositives: number;
    readonly constructionPositivesCovered: number;
    readonly constructionNegativesCovered: number;
    readonly labelledOfCovered: number;
    readonly agreementOnLabelled: number;
    readonly what: string;
  };
}

/**
 * What the round reached, counted against the committed corpus.
 *
 * `constructionTest` is the measurement `questions.ts` said it wanted -- how
 * often the generator's construction predicts the adjudicated label -- and it
 * comes back one-sided. The construction predicted "no" on all 20 covered items
 * and the adjudication agreed on all 20, which is 20/20 on the negative side
 * and 0 of 19 on the positive side. A prediction confirmed only where it
 * predicts the absence of a thing has not been tested against the thing.
 */
export function coverageReport(
  items: readonly { readonly id: string; readonly meta?: Record<string, unknown> | undefined }[],
): CoverageReport {
  const constructionPositives = items.filter(
    (i) => ((i.meta?.["predicateConstruction"] ?? {}) as Record<string, unknown>)["constructed"] === true,
  );
  const answered = new Set(LABELLED_ITEM_IDS);
  const covered = constructionPositives.filter((i) => answered.has(i.id));
  const withQuestion = items.filter((i) =>
    ((i.meta?.["labelQuestions"] ?? []) as readonly Record<string, unknown>[]).some(
      (q) => q["kind"] === "message-predicate",
    ),
  );
  const contested = items.flatMap((i) => contestedSpansOf(i));
  const labelled = LABELLED_ITEM_IDS.filter((id) => predicateVerdictFor(id).state === "labelled");
  const agreed = LABELLED_ITEM_IDS.filter((id) => {
    const item = items.find((i) => i.id === id);
    const constructed =
      ((item?.meta?.["predicateConstruction"] ?? {}) as Record<string, unknown>)["constructed"] === true;
    const verdict = predicateVerdictFor(id);
    return verdict.state === "labelled" && verdict.satisfiesPredicate === constructed;
  }).length;
  return {
    itemsInCorpus: items.length,
    itemsWithAPredicateQuestion: withQuestion.length,
    itemsAnswered: answered.size,
    itemsUnanswered: items.length - answered.size,
    contestedSpansInCorpus: contested.length,
    contestedSpansAnswered: contested.filter((c) => answered.has(c.questionId.split("#")[0]!)).length,
    constructionTest: {
      constructionPositives: constructionPositives.length,
      constructionPositivesCovered: covered.length,
      constructionNegativesCovered: answered.size - covered.length,
      labelledOfCovered: labelled.length,
      agreementOnLabelled: agreed,
      what:
        "how often meta.predicateConstruction.constructed predicted the adjudicated label. It is " +
        "one-sided: the round covered 0 of the items the construction calls positive, so the " +
        "prediction is confirmed only where it predicts 'no'. Nothing here supports the sentence " +
        "'the construction is a good predictor of the predicate'.",
    },
  };
}

// -- what the policy documents got wrong, as opposed to the corpus ----------

export interface PolicyGap {
  readonly id: string;
  readonly artifact: string;
  readonly clause: string;
  readonly finding: string;
  readonly consequence: string;
  readonly foundBy: string;
}

/**
 * Three defects this round found in the POLICY artifacts, kept separate from
 * the corpus findings because the fix is in a different file and none of them
 * is fixed here.
 *
 * Recording rather than fixing is the deliberate half. Two of the three are in
 * `policies/compiled/p-fin.ir.json`, and rewriting a compiled artifact by hand
 * to fit a label is precisely the move this project's standing conventions
 * forbid -- the artifact is supposed to be what the compiler produced, and
 * editing it would make `policyHash` a lie. The third is in the prose standard
 * itself, which no one here is authorised to amend.
 */
export const POLICY_GAPS: readonly PolicyGap[] = [
  {
    id: "PGAP-1",
    artifact: "policies/p-fin.md",
    clause: "§4.2",
    finding:
      "the standard protects the Firm's internal topology only through the database-connection-string " +
      "rule. §4.2 states topology disclosure as its REASON, so a host that discloses exactly the same " +
      "topology by another route -- the sftp file-drop endpoint this round affirmed as permitted -- " +
      "falls outside the standard as written.",
    consequence:
      "the affirmation of neg:sftp-endpoint is correct AGAINST THIS DOCUMENT and would be wrong " +
      "against a document that said what §4.2 gives as its reason. If p-fin is ever revised, these 13 " +
      "spans flip and this corpus's negatives are stale.",
    foundBy: "annotator A, FINDING 1; annotator B reached the same reading independently",
  },
  {
    id: "PGAP-2",
    artifact: "policies/compiled/p-fin.ir.json",
    clause: "§1.2 with §5.1",
    finding:
      "the compiled IR has no representation of §1.2 'customer data' as a class. Its entityTypes are " +
      "the eight §2-§4 identifier and secret classes plus the one semantic predicate, and " +
      "actions.providerOverrides encodes §5 by re-listing those same types at a stricter action. So a " +
      "§1.2 item that is not ALSO a §2-§4 type -- the NPCI retrieval reference is the worked example " +
      "-- cannot be expressed in the IR, cannot be emitted by any arm that detects against it, and " +
      "cannot be scored.",
    consequence:
      "this is the reason neg:retrieval-reference is disputed rather than corrected. The gap is in " +
      "the compiled policy, and no labelling decision inside the corpus can close it.",
    foundBy:
      "annotator B's correctedType proposal, which named a type the IR does not declare; confirmed " +
      "here by reading the IR's entityTypes and providerOverrides",
  },
  {
    id: "PGAP-3",
    artifact: "policies/compiled/p-fin.ir.json",
    clause: "§3.3",
    finding:
      "the IR's semanticPredicates[0].nlPredicate reads 'a client, prospective client, or " +
      "counterparty of the Firm' and omits §3.3's fourth limb, 'parties under a non-disclosure " +
      "agreement', while flattening 'deal counterparties' to 'counterparty'. Both annotators noticed " +
      "the omission in the paraphrase their brief gave them, without knowing the paraphrase came from " +
      "this artifact.",
    consequence:
      "a tier-2 arm reading this IR will under-fire on NDA-party disclosures. The corpus carries 3 " +
      "items whose constructed role is nda-party, and it has no predicate gold on any of them, so " +
      "nothing in this repository would currently catch that. This is the single most useful thing " +
      "the uncovered 169 items would have measured.",
    foundBy:
      "both annotators, in wasToldAbout; the IR as the source of the wording is verified by string " +
      "comparison in corpus-labelling.test.ts",
  },
];
