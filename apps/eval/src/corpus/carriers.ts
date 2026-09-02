/**
 * The carrier pool.
 *
 * ## Provenance, stated plainly: these are hand-authored, not ShareChat
 *
 * Spec 6.2 specifies real conversations from ShareChat (arXiv 2512.17843,
 * CC-BY-4.0) with WildChat-1M as a volume supplement. Neither is here. Ingesting
 * them needs a network fetch this session cannot make and a licensing review
 * this session cannot do, and spec 2.2 forbids raw third-party data in
 * `corpora/` regardless. So every carrier below was written for this file.
 *
 * That is a real weakness and it is the one spec 6.2 itself names: "ShareChat
 * conversations were publicly shared (selection toward 'interesting';
 * platform-side scrubbing) -- which is precisely why injection is needed". A
 * hand-authored carrier is worse than a scraped one in the same direction and
 * further along it: it is written by the same person who wrote the injections,
 * so the "adversarial style probe" realism gate (a classifier distinguishing
 * injected from pristine prompts with the span masked) would almost certainly
 * win against this pool. That gate is not implemented, and this is the carrier
 * pool it would have failed. Every item emitted carries
 * `meta.carrierSource: "hand-authored"` so no downstream reader can mistake
 * these for the corpus spec 6.2 describes.
 *
 * ## Segments and slots
 *
 * A carrier is a list of text segments joined with no separator. Injection
 * points ("slots") are the boundaries BETWEEN segments plus the end of the
 * text, derived by summing segment lengths. Nobody types an offset: a
 * hand-typed offset is a number that silently rots the first time a word is
 * fixed, and every gold span in this corpus is arithmetic downstream of these.
 *
 * ## What the carriers were written to avoid, and why that is disclosed
 *
 * They are prose, they contain no "@", no run of nine or more digits, no
 * all-uppercase token, and no Title-Case bigram -- i.e. they were authored to
 * pass `certify.ts`'s stage-1 sweep and its supplementary orthographic sweep.
 * Authoring to pass a filter is exactly the "cleaning negatives with detectors
 * biases them easy" circularity spec 6.2 asks to be disclosed, and this
 * paragraph is that disclosure. `DIRTY_CARRIERS` is the partial answer: three
 * carriers written to FAIL, so the quarantine rate in the manifest is a
 * measured number over a pool that contains failures rather than a zero over a
 * pool curated until nothing fails.
 */

export type Register = "casual" | "technical" | "formal";

export interface Carrier {
  readonly id: string;
  readonly register: Register;
  /** Joined with "" to form the text; boundaries between them are the slots. */
  readonly segments: readonly string[];
}

export function carrierText(c: Carrier): string {
  return c.segments.join("");
}

/**
 * Offsets at which an injection may be placed: after each segment, including
 * after the last (the end of the text). Never 0 -- an injection at offset 0
 * would open the message with glue and read as a different genre entirely.
 */
export function carrierSlots(c: Carrier): number[] {
  const slots: number[] = [];
  let acc = 0;
  for (const s of c.segments) {
    acc += s.length;
    slots.push(acc);
  }
  return slots;
}

export const SLOT_POSITIONS = ["head", "middle", "tail"] as const;
export type SlotPosition = (typeof SLOT_POSITIONS)[number];

/** Slot index -> position label, by thirds. Reported as a controlled dimension. */
export function slotPosition(index: number, total: number): SlotPosition {
  if (total <= 1) return "tail";
  const third = index / (total - 1);
  if (third < 0.34) return "head";
  if (third < 0.67) return "middle";
  return "tail";
}

export const CLEAN_CARRIERS: readonly Carrier[] = [
  {
    id: "c01",
    register: "technical",
    segments: [
      "i keep hitting a timeout when the nightly reconciliation job writes to the reporting table.",
      " the log shows the retry counter going up but there is no error message anywhere.",
      " any idea what to look at first?",
    ],
  },
  {
    id: "c02",
    register: "casual",
    segments: [
      "my manager wants a one page summary of last quarter by tomorrow morning and i have no idea where to start.",
      " she said it should be readable by someone who does not work in operations.",
      " can you give me a rough outline?",
    ],
  },
  {
    id: "c03",
    register: "formal",
    segments: [
      "please review the attached draft of the quarterly servicing report and advise whether the tone is appropriate for an external audience.",
      " the second paragraph in particular reads more defensive than i intended.",
      " a suggested rewrite would be welcome.",
    ],
  },
  {
    id: "c04",
    register: "technical",
    segments: [
      "the reconciliation script runs fine locally and then fails in staging with a connection reset after about forty seconds.",
      " both environments use the same driver version as far as i can tell.",
      " what would you check next?",
    ],
  },
  {
    id: "c05",
    register: "casual",
    segments: [
      "i have to explain to a very impatient colleague why the settlement file did not arrive on friday.",
      " the honest answer is that nobody noticed the upstream feed was late.",
      " how do i say that without sounding like i am blaming anyone?",
    ],
  },
  {
    id: "c06",
    register: "technical",
    segments: [
      "our batch job writes about eleven thousand rows a night and lately the last few hundred are missing.",
      " the job reports success and the row counter in the summary looks right.",
      " where would you start looking?",
    ],
  },
  {
    id: "c07",
    register: "formal",
    segments: [
      "kindly advise on the correct retention period for onboarding paperwork under our internal standard.",
      " the operations handbook and the compliance intranet page appear to disagree by two years.",
      " which of the two would ordinarily prevail?",
    ],
  },
  {
    id: "c08",
    register: "casual",
    segments: [
      "someone on my team keeps rewriting my emails before they go out and it is driving me a bit mad.",
      " the edits are not wrong exactly, they just remove anything that sounds like an opinion.",
      " how would you raise this without making it awkward?",
    ],
  },
  {
    id: "c09",
    register: "technical",
    segments: [
      "i want to add a retry with backoff around the settlement upload but i am not sure how many attempts is sensible.",
      " the upstream service is usually fine and then unavailable for about ninety seconds at a time.",
      " what would you pick?",
    ],
  },
  {
    id: "c10",
    register: "formal",
    segments: [
      "we are preparing a short note for the operations committee on delays in the month end close.",
      " the note needs to acknowledge the delay without inviting a full review of the process.",
      " could you suggest a structure?",
    ],
  },
  {
    id: "c11",
    register: "casual",
    segments: [
      "i am trying to write a handover document before i go on leave and it keeps turning into a novel.",
      " most of what i do is knowing who to ask rather than following a procedure.",
      " any advice on how to write that down?",
    ],
  },
  {
    id: "c12",
    register: "technical",
    segments: [
      "the query that builds the daily position report takes about four minutes and it used to take twenty seconds.",
      " nothing obvious changed in the schema and the row counts are similar to last month.",
      " what are the usual suspects?",
    ],
  },
  {
    id: "c13",
    register: "casual",
    segments: [
      "i have a meeting in an hour about why our team missed a deadline that i do not think was ever agreed.",
      " there is an email thread that supports my version but it is long and nobody will read it.",
      " how do i summarise it fairly?",
    ],
  },
  {
    id: "c14",
    register: "formal",
    segments: [
      "please draft a polite reminder to a counterparty who has not returned signed documentation after three requests.",
      " the relationship is otherwise good and i do not wish to escalate yet.",
      " a short paragraph would be sufficient.",
    ],
  },
  {
    id: "c15",
    register: "technical",
    segments: [
      "i need to move about nine gigabytes of archived statements off an old file share before it is decommissioned.",
      " the files are organised by year and then by branch which makes scripting awkward.",
      " what would be the least painful approach?",
    ],
  },
  {
    id: "c16",
    register: "casual",
    segments: [
      "my laptop takes almost five minutes to become usable after i log in each morning.",
      " the support desk says this is expected which i find hard to believe.",
      " is there anything i can check myself before i argue with them?",
    ],
  },
  {
    id: "c17",
    register: "technical",
    segments: [
      "we have a spreadsheet that six people edit at once and the totals disagree depending on who opens it.",
      " i suspect a stale cached copy but i cannot prove it.",
      " how would you go about demonstrating that?",
    ],
  },
  {
    id: "c18",
    register: "formal",
    segments: [
      "we intend to update the internal guidance on the use of external assistants for routine drafting work.",
      " the current wording is widely regarded as unclear on what may be pasted.",
      " could you propose a clearer formulation?",
    ],
  },
  {
    id: "c19",
    register: "casual",
    segments: [
      "i am doing a talk next week for people who have never worked in operations and i keep using jargon without noticing.",
      " my practice audience looked politely confused for twenty minutes.",
      " how do i catch that in my own writing?",
    ],
  },
  {
    id: "c20",
    register: "technical",
    segments: [
      "the alerting rule for failed uploads fires about thirty times a day and almost all of them are noise.",
      " people have started ignoring the channel entirely which is obviously the worst outcome.",
      " how would you rework it?",
    ],
  },
  {
    id: "c21",
    register: "formal",
    segments: [
      "we have been asked to provide a short written explanation of a discrepancy identified during a routine review.",
      " the discrepancy has since been corrected and the cause is understood.",
      " what tone would you recommend?",
    ],
  },
  {
    id: "c22",
    register: "casual",
    segments: [
      "there is a recurring meeting on my calendar that nobody seems to own and everyone attends anyway.",
      " it has run every week for two years and produces no decisions at all.",
      " how would you go about cancelling it?",
    ],
  },
];

/**
 * Carriers written to FAIL certification, one per sweep that can fail them.
 * They exist so the manifest's quarantine count is a measurement over a pool
 * containing failures. Removing them would make the reported rate 0 by
 * construction and say nothing about whether the sweeps run.
 */
export const DIRTY_CARRIERS: readonly Carrier[] = [
  {
    id: "d01-email",
    register: "casual",
    segments: [
      "can you help me word a reply to a colleague who cc'd the whole floor by mistake?",
      " the thread went to varsha.menon@brightline.example.com and about forty other people.",
      " i want to be gracious about it.",
    ],
  },
  {
    id: "d02-titlecase",
    register: "casual",
    segments: [
      "i am trying to plan a team offsite and the only free week is the one before the audit.",
      " everyone would have to travel to Ferndale Conference Centre on the tuesday.",
      " is that a terrible idea?",
    ],
  },
  {
    id: "d03-kv",
    register: "technical",
    segments: [
      "the deploy keeps picking up the wrong settings and i cannot see why.\n",
      "LOG_LEVEL=debug\nRETRY_BUDGET=30\nUPLOAD_MODE=production_readonly\n",
      "the same file works on my machine.",
    ],
  },
];

export const ALL_CARRIERS: readonly Carrier[] = [...CLEAN_CARRIERS, ...DIRTY_CARRIERS];
