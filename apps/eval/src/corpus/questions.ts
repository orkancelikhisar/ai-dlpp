import type { CorpusItem } from "../driver/corpus.js";
import { SHADOW_PREFIX } from "@sih/core";
import { CLIENT_SIDE_ROLES } from "./orgs.js";
import type { V2Family } from "./families.v2.js";

/**
 * The label questions the next phase answers, attached to every item.
 *
 * ## The defect this turns into the deliverable
 *
 * A review of `injection-p-fin-adjudicated-v1` found that 25 of its 108
 * positives contain a real, message-scoped `pred:client-relationship-disclosure`
 * violation BY CONSTRUCTION -- the client-name glue makes the finished message
 * satisfy p-fin's only semantic predicate -- and that the corpus emits no gold
 * for it. Under the injection invariant ("any finding outside the gold spans is
 * a true false positive") a model that correctly flags the relationship is
 * scored as a false positive, and the manifest attributed the resulting gap to
 * a different cause.
 *
 * Stripping the glue is not the fix. A client name with no relationship context
 * is barely a client name, and removing it would make the carriers read like
 * nothing anybody types. The fix is to treat the predicate as a first-class
 * LABEL: every item -- positive, confusable-only and pristine negative alike --
 * carries the question "does this message disclose that a named organisation is
 * a client, prospective client or counterparty of the Firm", in a form a blind
 * two-annotator round can answer, and the corpus's false-positive claim
 * explicitly excludes the predicate class until that round has run.
 *
 * That is the thing this project most needs: tier-2 gold at a scale above the
 * two items in `corpora/fixtures/smoke.gold-tier2.jsonl`.
 *
 * ## What the generator DOES know, and why it is not gold
 *
 * `predicateConstruction` records what the generator wrote and the argument
 * that the written clause is the only thing in the message that could satisfy
 * the predicate: the carrier was read end to end by two independent certifiers
 * against `policies/p-fin.md` and cleared by both, and a relationship
 * disclosure is exactly what an adjudicator reads for. That is a real argument
 * and it is NOT gold, for a reason the span invariant does not share: a span
 * offset is arithmetic over a splice, while "does this finished text disclose a
 * relationship" is a semantic judgement about a sentence. It is recorded as a
 * PREDICTION with its provenance, so the adjudication round can be scored
 * against it -- how often the construction predicts the adjudicated label is
 * itself a measurement, and one nobody can make if the construction is thrown
 * away.
 */

export const PREDICATE_ID = `${SHADOW_PREFIX}client-relationship-disclosure`;

export interface LabelQuestion {
  /** Stable within the item: `<itemId>#<n>`. */
  readonly id: string;
  readonly kind: "message-predicate" | "contested-span-label";
  readonly about: string;
  readonly question: string;
  readonly options: readonly string[];
  readonly state: "unanswered";
  readonly answeredBy: string;
  /** Contested-span questions only. */
  readonly span?: { readonly start: number; readonly end: number; readonly text: string };
}

export const PREDICATE_QUESTION =
  "Read the whole message. Does it disclose that a NAMED organisation is a client, a prospective " +
  "client, a deal counterparty, or a party under a non-disclosure agreement OF THE FIRM? Answer from " +
  "policies/p-fin.md sections 3.1, 3.3 and 3.4 only, and quote the words the answer rests on. A " +
  "relationship stated without a name, and a name stated without a relationship, are both 'no'.";

export const PREDICATE_ANSWERED_BY =
  "a blind two-annotator round with adjudication, run on the queue artifact and never on this file: " +
  "the queue carries the message text and the question and carries no family id, no constructedRole " +
  "and no gold, so an annotator cannot read the generator's intent off the row.";

export interface PredicateConstruction {
  readonly predicate: string;
  readonly scope: "message";
  /** What the generator's clause wrote. NOT a label. */
  readonly constructed: boolean;
  readonly constructedFrom: readonly string[];
  readonly state: "unadjudicated";
  readonly basis: string;
  readonly why: string;
}

const CONSTRUCTION_BASIS =
  "the carrier was cleared end to end by two independent certifiers against policies/p-fin.md, and a " +
  "relationship disclosure is what an adjudicator reads for, so the injected clause is the only thing " +
  "in this message that can satisfy a message-scoped relationship predicate.";

const CONSTRUCTION_WHY =
  "a span offset is arithmetic over a splice and can be asserted; whether a finished sentence discloses " +
  "a relationship is a semantic judgement, and one author's judgement dropped into a gold file that is " +
  "otherwise blind-labelled and adjudicated would be a provenance lie. Recorded as a prediction so the " +
  "adjudication round can be scored against it.";

/**
 * Whether the clause the generator wrote puts an organisation on the client
 * side of p-fin §3. Derived from `dimensions.constructedRole`, which is a fact
 * about generation, and mapped through the same `CLIENT_SIDE_ROLES` list
 * `orgs.ts` uses, so the two cannot drift.
 */
export interface InjectionView {
  readonly start: number;
  readonly end: number;
  readonly text: string;
  readonly family: string;
  readonly dimensions: Readonly<Record<string, string>>;
}

/**
 * Reads the injections back out of an emitted item.
 *
 * The generator hands `itemMetaExtra` the same values in memory, so nothing in
 * the emit path goes through this; it exists for the queue, which is built from
 * the SERIALIZED corpus. That is deliberate for the same reason
 * `verifyOrRefuse` re-reads the serialized bytes: the queue is what a human
 * round is handed, and it should be derived from the file, not from objects
 * that only ever existed in one process.
 */
export function injectionsOfItem(item: CorpusItem): InjectionView[] {
  return ((item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[]).map((raw) => ({
    start: raw["start"] as number,
    end: raw["end"] as number,
    text: String(raw["text"]),
    family: String(raw["family"]),
    dimensions: (raw["dimensions"] ?? {}) as Record<string, string>,
  }));
}

export function predicateConstructedFor(injections: readonly InjectionView[]): PredicateConstruction {
  const roles: string[] = [];
  for (const inj of injections) {
    const role = inj.dimensions["constructedRole"];
    if (role !== undefined && role !== "none") roles.push(role);
  }
  const constructed = roles.some((r) => (CLIENT_SIDE_ROLES as readonly string[]).includes(r));
  return {
    predicate: PREDICATE_ID,
    scope: "message",
    constructed,
    constructedFrom: [...roles].sort(),
    state: "unadjudicated",
    basis: CONSTRUCTION_BASIS,
    why: CONSTRUCTION_WHY,
  };
}

/**
 * The questions for one item: the predicate question, always, plus one per
 * contested confusable span.
 *
 * "Always" is the whole point. A predicate question only on the items that
 * carry an organisation would hand the annotator half the answer and would
 * leave the negative side of the eventual gold empty, which is the shape of
 * corpus that produces a recall number and no precision number.
 */
export function labelQuestionsFor(
  itemId: string,
  injections: readonly InjectionView[],
  familyById: ReadonlyMap<string, V2Family>,
): readonly LabelQuestion[] {
  const questions: LabelQuestion[] = [
    {
      id: `${itemId}#0`,
      kind: "message-predicate",
      about: PREDICATE_ID,
      question: PREDICATE_QUESTION,
      options: ["yes", "no", "unclear"],
      state: "unanswered",
      answeredBy: PREDICATE_ANSWERED_BY,
    },
  ];
  let n = 1;
  for (const inj of injections) {
    const family = familyById.get(inj.family);
    const contested = family?.labelBasis.contestedBy;
    if (family === undefined || contested === undefined) continue;
    questions.push({
      id: `${itemId}#${n}`,
      kind: "contested-span-label",
      about: family.type,
      question:
        `The quoted span is a ${family.surfaceName}. Under policies/p-fin.md, must it be withheld from an ` +
        `external AI assistant? One reading says no: ${family.labelBasis.reading} Another says yes: ${contested} ` +
        "Answer from the policy document alone and name the clause you relied on.",
      options: ["withhold", "permit", "unclear"],
      state: "unanswered",
      answeredBy: PREDICATE_ANSWERED_BY,
      span: { start: inj.start, end: inj.end, text: inj.text },
    });
    n += 1;
  }
  return questions;
}

/**
 * The item-side projection of a question: identity and state, no prose.
 *
 * The question TEXT is 500 characters and would otherwise be repeated on all
 * 189 items -- MEASURED, it took the emitted corpus from 0.6 MB to 1.0 MB -- so
 * it lives once in the manifest and once per row in the queue, which is the
 * artifact an annotator actually reads. What stays on the item is what a reader
 * of a single lifted JSONL line needs: that a question exists, what it is
 * about, that nobody has answered it, and where to find it.
 */
export function compactQuestions(
  questions: readonly LabelQuestion[],
  queuePath: string,
): readonly Record<string, unknown>[] {
  return questions.map((q) => ({
    id: q.id,
    kind: q.kind,
    about: q.about,
    state: q.state,
    ...(q.span === undefined ? {} : { span: q.span }),
    askedIn: queuePath,
  }));
}

export interface ScoringScope {
  readonly goldIsComplete: false;
  readonly unlabelledClasses: readonly string[];
  readonly why: string;
}

/**
 * What a scorer must NOT count as a false positive on this corpus.
 *
 * The injection invariant's second clause -- "any finding outside the gold
 * spans is a true false positive" -- is true of the tier-0 and tier-1 entity
 * classes and is FALSE of the predicate class, because the predicate is
 * unlabelled and the messages that satisfy it are in the file. Saying so in a
 * machine-readable field on every item is the difference between a disclosed
 * limit and a footnote nobody reads: an item lifted out of the JSONL carries
 * this with it, and a scorer that ignores it is ignoring something explicit.
 */
export function scoringScopeFor(contestedTypes: readonly string[]): ScoringScope {
  return {
    goldIsComplete: false,
    unlabelledClasses: [PREDICATE_ID, ...[...contestedTypes].sort()],
    why:
      "a finding of any class listed here is neither a match nor a false positive: the predicate is " +
      "constructed on both sides of this corpus and deliberately unlabelled, and the contested types " +
      "are confusables the policy can be read either way about. meta.labelQuestions carries the " +
      "questions; every other class is covered by the injection invariant.",
  };
}

export interface QueueRow {
  readonly questionId: string;
  readonly itemId: string;
  readonly text: string;
  readonly kind: string;
  readonly question: string;
  readonly options: readonly string[];
  readonly span?: { readonly start: number; readonly end: number; readonly text: string };
}

/**
 * The blind queue: message text, question, options, and nothing else.
 *
 * Deliberately NOT the corpus with an extra column. `meta.injections` names the
 * family and the constructed role, `gold` names the answer to the neighbouring
 * question, and `predicateConstruction` states outright what the generator
 * intended -- an annotator who saw any of it would be scoring the generator's
 * confidence rather than reading the message. The queue is a separate artifact
 * so that "the round was blind" is a property of what was handed over rather
 * than a claim about what somebody looked at.
 */
export function queueRows(items: readonly CorpusItem[], familyById: ReadonlyMap<string, V2Family>): QueueRow[] {
  const rows: QueueRow[] = [];
  for (const item of items) {
    for (const q of labelQuestionsFor(item.id, injectionsOfItem(item), familyById)) {
      rows.push({
        questionId: q.id,
        itemId: item.id,
        text: item.text,
        kind: q.kind,
        question: q.question,
        options: q.options,
        ...(q.span === undefined ? {} : { span: q.span }),
      });
    }
  }
  return rows;
}

export function serializeQueue(rows: readonly QueueRow[]): string {
  return rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
}
