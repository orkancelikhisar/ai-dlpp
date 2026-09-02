import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadCorpus, type CorpusItem } from "../driver/corpus.js";
import { loadTier2Gold, type Tier2GoldRow } from "../driver/score.js";
import { IR_PATH, OUT_DIR, REPO_ROOT } from "./build.js";
import {
  V2_CORPUS_PATH,
  V2_CORPUS_RELPATH,
  V2_QUEUE_PATH,
  V2_QUEUE_RELPATH,
  buildV2Artifacts,
} from "./build-v2.js";
import { serializeCorpus } from "./generate.js";
import {
  LABELLED_ITEM_IDS,
  LABELS_BY_ITEM,
  LABEL_ROUND,
  LABEL_ROUND_ID,
  POLICY_GAPS,
  SPAN_ADJUDICATIONS,
  assertFamilyUniformity,
  contestedSpansOf,
  coverageReport,
  labelAgreementReport,
  overrideReport,
  predicateAdjudication,
  predicateVerdictFor,
  type CoverageReport,
  type LabelAgreementReport,
  type OverrideReport,
  type PolicyGap,
  type PredicateAdjudication,
  type SpanAdjudication,
} from "./labelling.js";
import { PREDICATE_ID } from "./questions.js";

/**
 * What the blind labelling round leaves behind: a tier-2 predicate gold file, a
 * corpus whose scoring scope reflects the adjudication, and a manifest that
 * says plainly what the round could not support.
 *
 * ## Three artifacts, and why none of them is an edit
 *
 * `injection-p-fin-v2.jsonl` and its labelling queue are the bytes the two
 * annotators read. Editing either would rewrite the record of what was handed
 * over, so `assertSourceBytesUnchanged` re-reads both from disk, hashes them,
 * and refuses to emit unless they still match the sha256 the v2 manifest
 * recorded at build time. Everything this round produces is new files.
 *
 * ## The tier-2 gold is a SIDECAR, not a `pred:` span in `gold`
 *
 * `corpora/fixtures/smoke.gold-tier2.jsonl` already established the shape a
 * predicate label takes in this repository -- `Tier2GoldRowSchema` in
 * `../driver/score.ts` -- and it is a separate file keyed by `itemId`, carrying
 * `status`, both annotators' calls verbatim, and an adjudication string. It is
 * a better shape than an entry in the item's `gold` array for a reason that is
 * structural rather than stylistic: a message-scoped predicate that is NOT
 * satisfied has no span, and `gold` has nowhere to put "we asked, the answer
 * was no". A row with `satisfies: false` and `spans: []` says exactly that, and
 * `build-adjudicated.ts`'s emit gate refuses a `pred:` type inside `gold` for
 * the same reason. So this round writes the same shape the scorer already
 * reads, and `loadTier2Gold` parses the serialized bytes before they are
 * written.
 *
 * ## What it does not contain, stated here because it is the result
 *
 * 19 scored rows, 1 disputed, ZERO positives. The round covered 20 of 189
 * items and the 20 were selected for span contestation, disjoint from the 19
 * the generator predicts satisfy the predicate. So this file roughly triples
 * the tier-2 NEGATIVE pool (11 negatives in the smoke fixture, 19 here) and
 * adds nothing at all to the 2 positives. Predicate recall still cannot be
 * computed on anything but the smoke fixture's two items. See
 * `LABELLING_MANIFEST.canSupport`.
 */

export const LABELLED_CORPUS_RELPATH = "corpora/generated/injection-p-fin-v2.labelled.jsonl";
export const LABELLED_GOLD_RELPATH = "corpora/generated/injection-p-fin-v2.gold-tier2.jsonl";
export const LABELLED_MANIFEST_RELPATH = "corpora/generated/injection-p-fin-v2.labelled.manifest.json";
export const LABELLED_CORPUS_PATH = join(OUT_DIR, "injection-p-fin-v2.labelled.jsonl");
export const LABELLED_GOLD_PATH = join(OUT_DIR, "injection-p-fin-v2.gold-tier2.jsonl");
export const LABELLED_MANIFEST_PATH = join(OUT_DIR, "injection-p-fin-v2.labelled.manifest.json");
export const POLICY_DOC_PATH = join(REPO_ROOT, "policies/p-fin.md");
export const SMOKE_TIER2_GOLD_PATH = join(REPO_ROOT, "corpora/fixtures/smoke.gold-tier2.jsonl");

export function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

export interface SourceBytesCheck {
  readonly corpusPath: string;
  readonly corpusSha256: string;
  readonly queuePath: string;
  readonly queueSha256: string;
  readonly checkedAgainst: string;
  readonly note: string;
}

/**
 * The gate that makes these labels mean anything.
 *
 * A label is a judgement about a specific string. If the string moved, the
 * judgement is about text nobody read. So both source artifacts are re-read
 * from disk and hashed, and compared against the digests the v2 manifest
 * recorded when it built them -- which are themselves recomputed here by
 * running the v2 builder, so the comparison is against a value this process
 * derived rather than a number copied out of a file.
 */
export function assertSourceBytesUnchanged(expected: {
  readonly corpusSha256: string;
  readonly queueSha256: string;
}): SourceBytesCheck {
  const corpus = readFileSync(V2_CORPUS_PATH, "utf8");
  const queue = readFileSync(V2_QUEUE_PATH, "utf8");
  const corpusSha256 = sha256(corpus);
  const queueSha256 = sha256(queue);
  if (corpusSha256 !== expected.corpusSha256) {
    throw new Error(
      `refusing to emit: ${V2_CORPUS_RELPATH} is sha256 ${corpusSha256}, not the ` +
        `${expected.corpusSha256} its builder produces. These labels were written about the old ` +
        "bytes and cannot be carried onto new ones.",
    );
  }
  if (queueSha256 !== expected.queueSha256) {
    throw new Error(
      `refusing to emit: ${V2_QUEUE_RELPATH} is sha256 ${queueSha256}, not the ` +
        `${expected.queueSha256} its builder produces. The annotators read the old queue.`,
    );
  }
  return {
    corpusPath: V2_CORPUS_RELPATH,
    corpusSha256,
    queuePath: V2_QUEUE_RELPATH,
    queueSha256,
    checkedAgainst:
      "the sha256 buildV2Artifacts() produces in this process, not a digest copied from the " +
      "committed manifest",
    note:
      "both files are byte-identical to what the two annotators were handed. Nothing in this round " +
      "edits either; every artifact it produces is a new file.",
  };
}

// ---------------------------------------------------------------------------
// The tier-2 predicate gold
// ---------------------------------------------------------------------------

/**
 * One gold row per covered item, in `Tier2GoldRowSchema`'s shape.
 *
 * `confidence` is the WEAKER of the two annotators' calls, which is the
 * convention `smoke.gold-tier2.jsonl` already uses and states ("`confidence`
 * records B's weaker call, not A's, so a reader of this file sees the softest
 * reading of the item"). Following it means a reader can pool the two files
 * without a per-file rule.
 *
 * `spans` is empty on every row here, and that is not a shortcut: a row may
 * carry spans only when it satisfies the predicate, and no row does.
 */
export function tier2GoldRows(items: readonly CorpusItem[], policyHash: string): Tier2GoldRow[] {
  const byId = new Map(items.map((i) => [i.id, i]));
  const rows: Tier2GoldRow[] = [];
  for (const itemId of LABELLED_ITEM_IDS) {
    const item = byId.get(itemId);
    if (item === undefined) {
      throw new Error(
        `refusing to emit: annotators labelled item "${itemId}", which is not in ${V2_CORPUS_RELPATH}`,
      );
    }
    const a = LABELS_BY_ITEM.A.get(itemId)!;
    const b = LABELS_BY_ITEM.B.get(itemId)!;
    const verdict = predicateVerdictFor(itemId);
    if (verdict.state === "uncovered") {
      throw new Error(`refusing to emit: item "${itemId}" is in the labelled set but has no verdict`);
    }
    const satisfies = verdict.state === "labelled" ? verdict.satisfiesPredicate === true : a.satisfiesPredicate;
    const borderline = a.predicateConfidence === "borderline" || b.predicateConfidence === "borderline";
    rows.push({
      itemId,
      policy: "p-fin",
      policyHash,
      predicateId: PREDICATE_ID.slice("pred:".length),
      entityType: PREDICATE_ID,
      status: verdict.state === "labelled" ? "scored" : "disputed",
      satisfies,
      confidence: borderline ? "borderline" : "clear",
      spans: [],
      adjudication:
        `${verdict.state === "labelled" ? "AGREED" : "DISPUTED"} ${String(satisfies)} ` +
        `(A ${a.predicateConfidence}, B ${b.predicateConfidence}). ${verdict.reason} ` +
        `ROUND: ${LABEL_ROUND_ID}; see ${LABELLED_MANIFEST_RELPATH} for the blindness audit, which ` +
        "records three told-channel breaches and two structural leaks in the artifact these two " +
        "read. This row's predicate answer is unaffected by all five -- the item names no " +
        "organisation, or names one whose relationship is denied -- but the audit bounds what the " +
        "round as a whole is worth.",
      annotators: {
        a: { satisfies: a.satisfiesPredicate, confidence: a.predicateConfidence, rationale: a.rationale },
        b: { satisfies: b.satisfiesPredicate, confidence: b.predicateConfidence, rationale: b.rationale },
      },
    });
  }
  return rows;
}

export function serializeTier2Gold(rows: readonly Tier2GoldRow[]): string {
  return rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// The labelled corpus
// ---------------------------------------------------------------------------

/**
 * The contested types this round RESOLVED, so a finding on one is scoreable.
 *
 * Derived from `SPAN_ADJUDICATIONS` rather than listed, so a resolution that
 * changes in `labelling.ts` cannot leave a stale list here.
 */
export const RESOLVED_CONTESTED_TYPES: readonly string[] = Object.values(SPAN_ADJUDICATIONS)
  .filter((a) => a.resolution !== "disputed")
  .map((a) => a.contestedType)
  .sort();

export const DISPUTED_CONTESTED_TYPES: readonly string[] = Object.values(SPAN_ADJUDICATIONS)
  .filter((a) => a.resolution === "disputed")
  .map((a) => a.contestedType)
  .sort();

/**
 * v2's items with their labelling recorded, and NOTHING ELSE CHANGED.
 *
 * `text`, `gold` and `policy` are carried through untouched, so the injection
 * invariant that `build-adjudicated.ts` verified on v2 holds here by identity
 * rather than by a second verification of the same arithmetic --
 * `verifyLabelledOrRefuse` asserts the identity itself, which is the stronger
 * check.
 *
 * Three `meta` fields change, all of them fields that previously said "nobody
 * has answered this":
 *
 * - `predicateLabelling` moves from `unlabelled` to the adjudicated verdict,
 *   or, on the 169 items nobody was asked about, stays unlabelled with the
 *   REASON replaced: it is no longer "a round has not run" but "a round ran and
 *   did not reach this item", which is a different fact and points at a
 *   different fix.
 * - `spanLabelAdjudication` is new: per contested span in this item, what the
 *   round decided and whether it is scoreable.
 * - `scoringScope.unlabelledClasses` drops the resolved types and keeps the
 *   disputed ones, and drops the predicate on items that now carry a verdict.
 *   That field exists so an item lifted out of the JSONL carries its own
 *   limits; leaving it saying `neg:sftp-endpoint` is unlabelled after two
 *   annotators and an adjudication agreed on it would be a record of intent
 *   rather than fact.
 */
export function labelledItems(items: readonly CorpusItem[]): CorpusItem[] {
  return items.map((item) => {
    const verdict = predicateVerdictFor(item.id);
    const contested = contestedSpansOf(item);
    const stillUnlabelled: string[] = [];
    if (verdict.state !== "labelled") stillUnlabelled.push(PREDICATE_ID);
    for (const c of contested) {
      const adj = SPAN_ADJUDICATIONS[c.type];
      if (adj === undefined || adj.resolution === "disputed") stillUnlabelled.push(c.type);
    }
    const a = LABELS_BY_ITEM.A.get(item.id);
    const b = LABELS_BY_ITEM.B.get(item.id);
    return {
      ...item,
      meta: {
        ...item.meta,
        labelRound: {
          id: LABEL_ROUND_ID,
          covered: verdict.state !== "uncovered",
          goldPath: LABELLED_GOLD_RELPATH,
          manifestPath: LABELLED_MANIFEST_RELPATH,
          supersedes: {
            corpus: V2_CORPUS_RELPATH,
            fields: ["meta.predicateLabelling", "meta.scoringScope"],
            why:
              "v2 is left byte-identical because it is what the annotators read. Its per-item " +
              "predicateLabelling and scoringScope are therefore frozen at 'no round has run', " +
              "which stopped being true when this one did. Read this file for those two fields.",
          },
        },
        predicateLabelling:
          verdict.state === "uncovered"
            ? {
                state: "unlabelled",
                blockedOn: "a labelling round that reaches this item",
                why: verdict.reason,
              }
            : {
                state: verdict.state,
                predicate: PREDICATE_ID,
                satisfiesPredicate: verdict.satisfiesPredicate ?? null,
                scoredIn: LABELLED_GOLD_RELPATH,
                why: verdict.reason,
                annotators: {
                  A: { satisfiesPredicate: a!.satisfiesPredicate, confidence: a!.predicateConfidence },
                  B: { satisfiesPredicate: b!.satisfiesPredicate, confidence: b!.predicateConfidence },
                },
              },
        spanLabelAdjudication: contested.map((c) => {
          const adj = SPAN_ADJUDICATIONS[c.type];
          return {
            span: { start: c.start, end: c.end, text: c.text },
            type: c.type,
            resolution: adj?.resolution ?? "disputed",
            scoredAs: adj?.scoredAs ?? "excluded",
            annotators: {
              A: { spanLabelCorrect: a?.spanLabelCorrect ?? null, correctedType: a?.correctedType ?? null },
              B: { spanLabelCorrect: b?.spanLabelCorrect ?? null, correctedType: b?.correctedType ?? null },
            },
            clauses: adj?.clauses ?? [],
          };
        }),
        scoringScope: {
          goldIsComplete: false,
          unlabelledClasses: [...new Set(stillUnlabelled)].sort(),
          why:
            "a finding of any class listed here is neither a match nor a false positive on this " +
            "item. Every class NOT listed is now covered by the injection invariant, including the " +
            `contested types this round resolved (${RESOLVED_CONTESTED_TYPES.join(", ") || "none"}). ` +
            `The predicate is listed only where ${LABELLED_GOLD_RELPATH} has no scored row for this ` +
            "item; where it does, the row is the ground truth and a finding is scoreable against it.",
        },
      },
    };
  });
}

// ---------------------------------------------------------------------------
// The emit gate
// ---------------------------------------------------------------------------

export const LABELLED_VERIFICATION_CHECKS: readonly string[] = [
  "the source corpus and the labelling queue still hash to what the v2 builder produces, so these " +
    "labels are about the bytes the annotators read",
  "every labelled item exists in the source corpus, matched by id",
  "every emitted item's text, policy and gold array are IDENTICAL to the source item's, so v2's " +
    "injection invariant carries over by identity rather than by re-derivation",
  "every gold span still slices back: text.slice(start, end) === span.text. The refusal comes from " +
    "CorpusItemSchema's refine when loadCorpus parses the serialized bytes, which fires before the " +
    "loop below reaches the same span; the loop keeps its copy as a second line rather than relying " +
    "on a refine in another module staying where it is",
  "no gold span carries a pred: type -- predicate labels live in the tier-2 sidecar, which is the " +
    "only shape that can record 'we asked and the answer was no'",
  "every tier-2 gold row parses through loadTier2Gold FROM THE SERIALIZED BYTES, names an item in " +
    "the corpus, and carries spans that slice back on that item's text",
  "every contested span in the corpus has an adjudication, and every adjudicated type is one the " +
    "corpus actually contains",
  "no annotator split a contested family, so a family-level adjudication stands for items that " +
    "were all answered the same way",
];

export interface LabelledVerificationReport {
  readonly source: SourceBytesCheck;
  readonly itemsChecked: number;
  readonly goldSpansChecked: number;
  readonly contestedSpansChecked: number;
  readonly tier2RowsChecked: number;
  readonly tier2SpansChecked: number;
  readonly checks: readonly string[];
}

/**
 * Throws rather than returning a flag, for the reason `verifyOrRefuse` does: a
 * caller that could ignore the result is a caller that will.
 *
 * It reads the tier-2 gold back out of the SERIALIZED bytes through
 * `loadTier2Gold`, not from the objects that produced them. The JSON round trip
 * and the zod refines are what every downstream consumer depends on, and a
 * check against the in-memory rows would leave both unexercised.
 */
export function verifyLabelledOrRefuse(
  source: SourceBytesCheck,
  sourceItems: readonly CorpusItem[],
  emittedJsonl: string,
  goldJsonl: string,
): LabelledVerificationReport {
  const emitted = loadCorpus(emittedJsonl);
  const sourceById = new Map(sourceItems.map((i) => [i.id, i]));
  let goldSpansChecked = 0;
  let contestedSpansChecked = 0;

  if (emitted.length !== sourceItems.length) {
    throw new Error(
      `refusing to emit: ${emitted.length} labelled items against ${sourceItems.length} source items`,
    );
  }
  for (const item of emitted) {
    const src = sourceById.get(item.id);
    if (src === undefined) throw new Error(`refusing to emit: labelled item ${item.id} is not in the source corpus`);
    if (item.text !== src.text) {
      throw new Error(`refusing to emit: item ${item.id} text differs from the source corpus`);
    }
    if (item.policy !== src.policy) {
      throw new Error(`refusing to emit: item ${item.id} policy differs from the source corpus`);
    }
    if (JSON.stringify(item.gold) !== JSON.stringify(src.gold)) {
      throw new Error(
        `refusing to emit: item ${item.id} gold differs from the source corpus. This round adjudicates ` +
          "labels; it does not re-mint spans.",
      );
    }
    for (const gold of item.gold) {
      if (gold.end > item.text.length || item.text.slice(gold.start, gold.end) !== gold.text) {
        throw new Error(
          `refusing to emit: item ${item.id} gold span [${gold.start},${gold.end}) is ` +
            `${JSON.stringify(item.text.slice(gold.start, gold.end))}, not ${JSON.stringify(gold.text)}`,
        );
      }
      if (gold.entityType.startsWith("pred:")) {
        throw new Error(
          `refusing to emit: item ${item.id} carries a pred: span in gold. Predicate labels belong in ` +
            `${LABELLED_GOLD_RELPATH}, which can express a negative answer.`,
        );
      }
      goldSpansChecked += 1;
    }
    for (const contested of contestedSpansOf(item)) {
      if (SPAN_ADJUDICATIONS[contested.type] === undefined) {
        throw new Error(
          `refusing to emit: item ${item.id} carries a contested span of type "${contested.type}" ` +
            "that this round did not adjudicate",
        );
      }
      if (item.text.slice(contested.start, contested.end) !== contested.text) {
        throw new Error(
          `refusing to emit: item ${item.id} contested span [${contested.start},${contested.end}) ` +
            "does not slice back",
        );
      }
      contestedSpansChecked += 1;
    }
  }

  // In the GATE rather than only in the builder, and the difference is not
  // cosmetic. MEASURED by mutation: deleting the builder's call to
  // `assertFamilyUniformity` survived the whole suite, because the only test
  // that exercised it called the function directly. A guard nothing on the emit
  // path can be shown to run is not a guard.
  const typeOfItem = new Map<string, string>();
  for (const item of sourceItems) for (const c of contestedSpansOf(item)) typeOfItem.set(item.id, c.type);
  assertFamilyUniformity(typeOfItem);

  const typesInCorpus = new Set(emitted.flatMap((i) => contestedSpansOf(i).map((c) => c.type)));
  for (const type of Object.keys(SPAN_ADJUDICATIONS)) {
    if (!typesInCorpus.has(type)) {
      throw new Error(
        `refusing to emit: this round adjudicates "${type}", which no contested span in the corpus ` +
          "carries. An adjudication of nothing is a claim with no subject.",
      );
    }
  }

  const rows = loadTier2Gold(goldJsonl);
  let tier2SpansChecked = 0;
  for (const row of rows) {
    const item = sourceById.get(row.itemId);
    if (item === undefined) {
      throw new Error(`refusing to emit: tier-2 gold row names item "${row.itemId}", which is not in the corpus`);
    }
    for (const span of row.spans) {
      if (span.end > item.text.length || item.text.slice(span.start, span.end) !== span.text) {
        throw new Error(
          `refusing to emit: tier-2 gold row ${row.itemId} span [${span.start},${span.end}) does not slice back`,
        );
      }
      tier2SpansChecked += 1;
    }
  }

  return {
    source,
    itemsChecked: emitted.length,
    goldSpansChecked,
    contestedSpansChecked,
    tier2RowsChecked: rows.length,
    tier2SpansChecked,
    checks: LABELLED_VERIFICATION_CHECKS,
  };
}

// ---------------------------------------------------------------------------
// What this corpus can and cannot support
// ---------------------------------------------------------------------------

export interface ClassCapability {
  readonly className: string;
  readonly goldPositives: number;
  readonly scoredNegatives: number | null;
  /** The smallest change in recall this class can express: 1 / goldPositives. */
  readonly recallStep: number | null;
  readonly note: string;
}

/**
 * The other half of every accuracy number: what an arm can be WRONG about
 * without it being a false positive.
 *
 * Counted from the corpus rather than stated, because this round changed it:
 * 13 sftp spans moved from "excluded, the policy could be read either way" to
 * "scoreable, a finding here is a true false positive", and 7 retrieval
 * references stayed out.
 */
export interface NegativeSurface {
  readonly pristineNegativeItems: number;
  readonly confusableSpans: number;
  readonly scoredConfusableSpans: number;
  readonly excludedConfusableSpans: number;
  readonly movedIntoScopeByThisRound: number;
  readonly note: string;
}

export interface CapabilityReport {
  readonly perEntityType: readonly ClassCapability[];
  readonly goldSpans: number;
  readonly negativeSurface: NegativeSurface;
  readonly predicate: ClassCapability;
  readonly canRank: readonly string[];
  readonly cannotRank: readonly string[];
  readonly verdict: string;
}

/**
 * Recall is a fraction with an integer numerator, so with `p` gold positives it
 * can only take the values 0, 1/p, 2/p, ... 1. Two arms whose true recall
 * differs by less than 1/p CANNOT be ordered by this corpus, however many
 * decimal places the report prints.
 *
 * The one-sided 95% bound for a class that yields ZERO events is the exact
 * Clopper-Pearson upper limit, `1 - 0.05^(1/n)`. It is what a reader can
 * legitimately say about an arm that never fires on the predicate: not "the
 * false-positive rate is 0" but "it is at most this".
 */
export function zeroEventUpperBound95(n: number): number | null {
  return n <= 0 ? null : 1 - Math.pow(0.05, 1 / n);
}

export function capabilityReport(
  items: readonly CorpusItem[],
  predicate: PredicateAdjudication,
  smokeGold: readonly Tier2GoldRow[],
): CapabilityReport {
  const byType = new Map<string, number>();
  for (const item of items) for (const g of item.gold) byType.set(g.entityType, (byType.get(g.entityType) ?? 0) + 1);
  const perEntityType = [...byType.entries()]
    .sort((x, y) => x[0].localeCompare(y[0]))
    .map(([className, goldPositives]): ClassCapability => ({
      className,
      goldPositives,
      scoredNegatives: null,
      recallStep: 1 / goldPositives,
      note:
        `per-type recall moves in steps of ${(100 / goldPositives).toFixed(1)} percentage points; no ` +
        "difference smaller than that between two arms is resolvable on this corpus.",
    }));

  const confusableSpans = items.flatMap(
    (i) => (i.meta?.["labels"] ?? []) as readonly Record<string, unknown>[],
  ).filter((l) => String(l["type"]).startsWith("neg:")).length;
  const contested = items.flatMap((i) => contestedSpansOf(i));
  const excludedConfusableSpans = contested.filter(
    (c) => (SPAN_ADJUDICATIONS[c.type]?.resolution ?? "disputed") === "disputed",
  ).length;
  const resolvedByThisRound = contested.length - excludedConfusableSpans;
  const negativeSurface: NegativeSurface = {
    pristineNegativeItems: items.filter((i) => i.gold.length === 0).length,
    confusableSpans,
    scoredConfusableSpans: confusableSpans - excludedConfusableSpans,
    excludedConfusableSpans,
    movedIntoScopeByThisRound: resolvedByThisRound,
    note:
      "before this round every contested span was excluded from the false-positive claim. The " +
      `adjudication resolved ${resolvedByThisRound} of them (${RESOLVED_CONTESTED_TYPES.join(", ")}), ` +
      `so they are now scoreable, and left ${excludedConfusableSpans} out ` +
      `(${DISPUTED_CONTESTED_TYPES.join(", ")}).`,
  };
  const goldSpans = items.reduce((n, i) => n + i.gold.length, 0);
  const steps = perEntityType.map((c) => 100 / c.goldPositives);

  const smokePositives = smokeGold.filter((r) => r.status === "scored" && r.satisfies).length;
  const smokeNegatives = smokeGold.filter((r) => r.status === "scored" && !r.satisfies).length;
  const pooledPositives = predicate.positives + smokePositives;
  const pooledNegatives = predicate.negatives + smokeNegatives;
  const predicateCapability: ClassCapability = {
    className: PREDICATE_ID,
    goldPositives: predicate.positives,
    scoredNegatives: predicate.negatives,
    recallStep: predicate.positives === 0 ? null : 1 / predicate.positives,
    note:
      `this round contributes ${predicate.positives} positive(s) and ${predicate.negatives} scored ` +
      `negative(s). RECALL IS NOT DEFINED on this file alone -- a fraction with a zero denominator ` +
      `-- and pooling it with corpora/fixtures/smoke.gold-tier2.jsonl (${smokePositives} positives, ` +
      `${smokeNegatives} negatives) still leaves ${pooledPositives} positives in total, so pooled ` +
      `predicate recall is quantised to steps of ${(100 / Math.max(pooledPositives, 1)).toFixed(1)} ` +
      "percentage points and can express only three values. What this round DOES buy is the " +
      `negative side: ${pooledNegatives} scored negatives pooled, so an over-firing rate moves in ` +
      `steps of ${(100 / Math.max(pooledNegatives, 1)).toFixed(1)} points, and an arm that fires on ` +
      `none of them supports "predicate false-positive rate at most ` +
      `${((zeroEventUpperBound95(pooledNegatives) ?? 0) * 100).toFixed(1)}% with 95% confidence".`,
  };

  const canRank = perEntityType
    .filter((c) => c.goldPositives >= 15)
    .map((c) => `${c.className} (${c.goldPositives} gold spans, ${(100 / c.goldPositives).toFixed(1)} pp steps)`);
  const cannotRank = [
    ...perEntityType
      .filter((c) => c.goldPositives < 15)
      .map(
        (c) =>
          `${c.className}: ${c.goldPositives} gold spans. Recall steps of ` +
          `${(100 / c.goldPositives).toFixed(1)} pp, so only a very large arm difference is visible.`,
      ),
    `${PREDICATE_ID} RECALL: ${predicate.positives} positives from this round. Not rankable at all -- ` +
      "there is no denominator. Pooled with the smoke fixture there are 2, which can express recall " +
      "0, 0.5 or 1 and cannot order two arms in any useful sense.",
    ...DISPUTED_CONTESTED_TYPES.map(
      (t) =>
        `${t}: the adjudication left this type DISPUTED, so its spans are outside both the match set ` +
        "and the false-positive set. Nothing about an arm's behaviour on them is scoreable.",
    ),
  ];

  return {
    perEntityType,
    goldSpans,
    negativeSurface,
    predicate: predicateCapability,
    canRank,
    cannotRank,
    verdict:
      "THIS CORPUS CAN SUPPORT A COARSE MODEL RANKING ON ENTITY-TYPE SPAN RECALL AND ON " +
      `OVER-BLOCKING, AND CANNOT SUPPORT ONE ON THE PREDICATE. The entity side has ${goldSpans} gold ` +
      `spans over ${perEntityType.length} types with per-type recall steps between ` +
      `${Math.min(...steps).toFixed(1)} and ${Math.max(...steps).toFixed(1)} percentage points, and ` +
      `the false-positive side has ${negativeSurface.pristineNegativeItems} pristine negative items ` +
      `plus ${negativeSurface.scoredConfusableSpans} scored confusable spans -- enough to order arms ` +
      "that differ substantially, not enough to resolve small differences on the scarce types " +
      `(${perEntityType.filter((c) => c.goldPositives < 10).map((c) => `${c.className} at ${c.goldPositives}`).join(", ")}). ` +
      "The predicate side is the deliverable this round was commissioned for and it did not arrive: " +
      `${predicate.labelled} scored rows, ${predicate.positives} of them positive, so an arm can be ` +
      "measured for OVER-FIRING on the predicate and cannot be measured for CATCHING it. Any headline " +
      "that reads 'tier-2 predicate accuracy' off this corpus would be reporting a precision-only " +
      "number as if it were both halves.",
  };
}

// ---------------------------------------------------------------------------
// The manifest
// ---------------------------------------------------------------------------

export interface LabellingManifest {
  readonly round: typeof LABEL_ROUND;
  readonly generator: { readonly seed: string; readonly generatorVersion: string; readonly irHash: string };
  readonly source: SourceBytesCheck;
  readonly agreement: LabelAgreementReport;
  readonly overrides: OverrideReport;
  readonly coverage: CoverageReport;
  readonly spanAdjudication: {
    readonly perType: readonly SpanAdjudication[];
    readonly contestedSpans: number;
    readonly resolvedSpans: number;
    readonly disputedSpans: number;
    readonly disputedRate: number;
  };
  readonly predicateAdjudication: PredicateAdjudication;
  readonly policyGaps: readonly PolicyGap[];
  readonly canSupport: CapabilityReport;
  readonly artifact: {
    readonly corpusPath: string;
    readonly corpusSha256: string;
    readonly goldPath: string;
    readonly goldSha256: string;
  };
  readonly verification: LabelledVerificationReport;
  readonly unvalidated: readonly string[];
}

export interface BuiltLabelledArtifacts {
  readonly corpusJsonl: string;
  readonly goldJsonl: string;
  readonly manifestJson: string;
  readonly items: readonly CorpusItem[];
  readonly goldRows: readonly Tier2GoldRow[];
  readonly manifest: LabellingManifest;
}

/**
 * Two-space JSON with a trailing newline, matching `generate.ts`'s
 * `serializeManifest` byte for byte. Written out rather than reused because
 * that function is typed to `CorpusManifest` and this manifest is not one --
 * it describes a labelling round over an existing corpus, not a generation --
 * and casting to borrow a formatter would put a lie in the type.
 */
export function serializeLabellingManifest(manifest: LabellingManifest): string {
  return JSON.stringify(manifest, null, 2) + "\n";
}

export function buildLabelledArtifacts(): BuiltLabelledArtifacts {
  const v2 = buildV2Artifacts();
  const source = assertSourceBytesUnchanged({
    corpusSha256: v2.manifest.artifact.corpusSha256,
    queueSha256: v2.manifest.artifact.queueSha256,
  });
  const sourceItems = v2.items;

  // The family-uniformity guard lives in `verifyLabelledOrRefuse`, which runs
  // below on the serialized bytes. It is not repeated here: a second call would
  // be the one a mutation test cannot tell from the first.
  const policyHash = sha256(readFileSync(POLICY_DOC_PATH, "utf8"));
  const goldRows = tier2GoldRows(sourceItems, policyHash);
  const goldJsonl = serializeTier2Gold(goldRows);
  const items = labelledItems(sourceItems);
  const corpusJsonl = serializeCorpus(items);
  const verification = verifyLabelledOrRefuse(source, sourceItems, corpusJsonl, goldJsonl);

  const predicate = predicateAdjudication(sourceItems.map((i) => i.id));
  const contestedSpans = sourceItems.flatMap((i) => contestedSpansOf(i));
  const disputedSpans = contestedSpans.filter(
    (c) => (SPAN_ADJUDICATIONS[c.type]?.resolution ?? "disputed") === "disputed",
  ).length;
  const smokeGold = loadTier2Gold(readFileSync(SMOKE_TIER2_GOLD_PATH, "utf8"));

  const manifest: LabellingManifest = {
    round: LABEL_ROUND,
    generator: {
      seed: String(sourceItems[0]?.meta?.["seed"] ?? ""),
      generatorVersion: String(sourceItems[0]?.meta?.["generatorVersion"] ?? ""),
      irHash: sha256(readFileSync(IR_PATH, "utf8")),
    },
    source,
    agreement: labelAgreementReport(),
    overrides: overrideReport(LABELLED_ITEM_IDS, (itemId) => {
      const item = sourceItems.find((i) => i.id === itemId);
      return (
        ((item?.meta?.["predicateConstruction"] ?? {}) as Record<string, unknown>)["constructed"] === true
      );
    }),
    coverage: coverageReport(sourceItems),
    spanAdjudication: {
      perType: Object.values(SPAN_ADJUDICATIONS),
      contestedSpans: contestedSpans.length,
      resolvedSpans: contestedSpans.length - disputedSpans,
      disputedSpans,
      disputedRate: contestedSpans.length === 0 ? 0 : disputedSpans / contestedSpans.length,
    },
    predicateAdjudication: predicate,
    policyGaps: POLICY_GAPS,
    canSupport: capabilityReport(sourceItems, predicate, smokeGold),
    artifact: {
      corpusPath: LABELLED_CORPUS_RELPATH,
      corpusSha256: sha256(corpusJsonl),
      goldPath: LABELLED_GOLD_RELPATH,
      goldSha256: sha256(goldJsonl),
    },
    verification,
    unvalidated: [
      "THE DELIVERABLE DID NOT ARRIVE. This round was commissioned to produce pred: gold at a scale " +
        "above the two positives in corpora/fixtures/smoke.gold-tier2.jsonl. It produced 19 scored " +
        "rows and ZERO positives, because both annotators were scoped to the 20 span-bearing queue " +
        "rows and those 20 are disjoint from the 19 items the generator predicts satisfy the " +
        "predicate. Predicate RECALL is still unmeasurable outside the smoke fixture.",
      "THE ROUND'S TOLD CHANNEL CARRIED THREE BREACHES, one of them first-order: the brief handed " +
        "both annotators a paraphrase of the predicate that is the COMPILED IR's own nlPredicate " +
        "(round.blindness.channels[told], TOLD-1). No label in this round is affected -- the phrase " +
        "omits the NDA limb and no NDA-party item was covered -- but a round that had covered the " +
        "other 169 would have been keyed against text taken from the arm under test.",
      "THE BLIND ARTIFACT ITSELF LEAKS. The queue's question field names each span's semantic type " +
        "and pre-argues both readings with clause numbers, and item ids encode the hard-negative " +
        "stratum (round.blindness.channels[structural]). questions.ts's docblock claims the queue " +
        "carries nothing an annotator could read the generator's intent off; that claim is false and " +
        "is committed in the v2 manifest.",
      "AGREEMENT ON THE PREDICATE IS 1.0 AND MEANS LESS THAN IT LOOKS. Both annotators answered 'no' " +
        "on all 20, so chance agreement is 1 and kappa is undefined. The number says the two agree " +
        "about the negative class on a subset selected for something else; it says nothing about how " +
        "reliably the clause can be labelled where it bites.",
      "SEVEN SPANS ARE DISPUTED AND EXCLUDED, and the exclusion is not neutral: it removes the seven " +
        "spans on which a §5-aware model would look wrong under the current gold. The underlying " +
        "defect is PGAP-2, a compiled IR with no representation of §1.2 customer data.",
      "THE CARRIERS, THE REALISM GATES, CERTIFICATION STAGES 2 AND 3, AND p-med/p-corp ARE UNCHANGED " +
        "AND STILL UNVALIDATED. Everything in injection-p-fin-v2.manifest.json's unvalidated block " +
        "still applies; this round adjudicated labels and touched none of it.",
    ],
  };

  return { corpusJsonl, goldJsonl, manifestJson: serializeLabellingManifest(manifest), items, goldRows, manifest };
}

export function writeLabelledArtifacts(): BuiltLabelledArtifacts {
  const built = buildLabelledArtifacts();
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(LABELLED_CORPUS_PATH, built.corpusJsonl, "utf8");
  writeFileSync(LABELLED_GOLD_PATH, built.goldJsonl, "utf8");
  writeFileSync(LABELLED_MANIFEST_PATH, built.manifestJson, "utf8");
  return built;
}

/**
 * Not runnable under plain node, for the reason `build-v2.ts` records: node's
 * type stripper cannot resolve `@sih/core`. `corpus-labelling.test.ts`
 * regenerates all three files in memory on every `pnpm -r test` and compares
 * them to the committed bytes, so the reproducibility guarantee does not depend
 * on this entry point running.
 */
if (process.argv[1] !== undefined && import.meta.url === `file://${process.argv[1]}`) {
  const built = writeLabelledArtifacts();
  process.stdout.write(
    `${LABEL_ROUND_ID}: ${built.goldRows.length} tier-2 gold rows ` +
      `(${built.manifest.predicateAdjudication.labelled} scored, ` +
      `${built.manifest.predicateAdjudication.disputed} disputed, ` +
      `${built.manifest.predicateAdjudication.positives} positive), ` +
      `${built.manifest.spanAdjudication.resolvedSpans}/${built.manifest.spanAdjudication.contestedSpans} ` +
      `contested spans resolved, sha256 ${built.manifest.artifact.corpusSha256}\n`,
  );
}
