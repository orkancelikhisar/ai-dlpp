import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadPolicyIr, type PolicyIr } from "@sih/core";
import { loadCorpus, type CorpusItem } from "../driver/corpus.js";
import {
  ADJUDICATION_ROUND,
  ADJUDICATION_SWEEP_ID,
  ADMISSION_RULE,
  ADMITTED_CARRIER_IDS,
  CARRIER_VERDICTS,
  adjudicationSweep,
  agreementReport,
  isAdmitted,
  type AgreementReport,
} from "./adjudication.js";
import { IR_PATH, OUT_DIR, SELFTEST_PATH, loadSelfTestExamples } from "./build.js";
import { OFFERED_CARRIERS, verifyOrRefuse, type VerificationReport } from "./build-adjudicated.js";
import { orthographicOrgSweep } from "./certify.js";
import { NEG_PREFIX } from "./labels.js";
import type { Family } from "./families.js";
import {
  ORG_FAMILY_IDS,
  PAIRED_TYPE,
  REMOVED_FAMILIES,
  RENAMED_FAMILIES,
  V2_CONFUSABLE_FAMILIES,
  V2_FAMILIES,
  V2_POSITIVE_FAMILIES,
  orgRoleClassOf,
  type V2Family,
} from "./families.v2.js";
import { FORMAT_SPEC_SWEEP_ID, formatSpecSweep } from "./format-sweep.js";
import { ORG_POOL } from "./orgs.js";
import {
  GENERATOR_NAME,
  GENERATOR_VERSION,
  generateCorpus,
  serializeCorpus,
  serializeManifest,
  type CorpusManifest,
  type GeneratedCorpus,
} from "./generate.js";
import {
  IR_COUNTEREXAMPLE_SURFACES,
  measureBoost,
  measureOrthography,
  measureRoles,
  type BoostReport,
  type OrthographyReport,
  type RoleReport,
} from "./leakage.js";
import {
  PREDICATE_ANSWERED_BY,
  PREDICATE_ID,
  PREDICATE_QUESTION,
  compactQuestions,
  injectionsOfItem,
  labelQuestionsFor,
  predicateConstructedFor,
  queueRows,
  scoringScopeFor,
  serializeQueue,
} from "./questions.js";

/**
 * The third corpus, and the first whose numbers could be published.
 *
 * ## Why there is a third artifact rather than an edit
 *
 * A review of `corpora/generated/injection-p-fin-adjudicated-v1` found it
 * systematically biased TOWARD the compiled arm, in four independent ways, and
 * the accuracy numbers it produced were discarded unread rather than published.
 * The machinery was sound -- gold spans known by construction, an injection
 * invariant enforced by arithmetic, an emit gate that verifies serialized bytes
 * -- and the INPUTS were contaminated. This file rebuilds the inputs and keeps
 * the machinery.
 *
 * Both earlier corpora still reproduce byte for byte from their own builders,
 * and their tests still hold that line: an artifact nobody can regenerate is
 * not an artifact, and deleting the record of a measurement that was made is
 * not the same as correcting it. They are SUPERSEDED, which the manifest here
 * says in the field a reader will find, and nothing should be scored on them.
 *
 * ## The four leaks, before and after, all measured by `leakage.ts`
 *
 * 1. SURFACES FROM THE IR. 13 of the 24 confusable families in the adjudicated
 *    corpus were 1:1 with an entry in the IR's own `counterExamples` lists,
 *    carrying 59 of its 108 confusable spans -- so on more than half of them the
 *    corpus asked "is this a PAN?" of an arm whose prompt already said
 *    "ABCDE1234F is not a PAN". Wave 3 mints every surface from its published
 *    format rule and shares no surface with that list. After: 0 of 27.
 * 2. contextBoost ASYMMETRY. A `contextBoost` term -- a compiled-arm-only
 *    feature -- sat near 40.7% of gold spans and near 0.9% of confusable spans,
 *    so the keyword predicted the label. After: 45.4% against 46.1%, a delta of
 *    -0.007, and the sharper own-entityType form falls from 0.333 to 0.024.
 * 3. THE GOLD SPAN AS THE UNIQUE ODD TOKEN. On 58 of 108 gold spans the crude
 *    orthographic oracle returned that span and nothing else. Wave 3 injects a
 *    shape-matched, LABELLED confusable beside every positive. After: 0 of 108,
 *    and the oracle's precision over the corpus falls from 0.49 to 0.32.
 * 4. ROLE READABLE OFF THE NAME. 19 of 39 organisation spans carried a name that
 *    appears in only one role class corpus-wide, because the client pool, the
 *    supplier pool, the competitor pool and the employer name were four disjoint
 *    arrays. Wave 3 draws every organisation from one pool. After: 0 of 59, with
 *    all five names appearing on both sides.
 *
 * ## What this corpus does NOT fix
 *
 * The carriers are unchanged, and unchanged on purpose: they are the 27 a blind
 * two-certifier round admitted, and editing one voids its verdict. They are
 * still hand-authored rather than the ShareChat/WildChat conversations spec 6.2
 * specifies, none of spec 6.2's three realism gates has run, stages 2 and 3 of
 * certification are still unrun, and p-med and p-corp are still uncompiled.
 * Every one of those is in `unvalidated` and in the manifest's `gaps`.
 */

export const V2_SEED = "sih-p7-unleaked-v2";
export const V2_CORPUS_RELPATH = "corpora/generated/injection-p-fin-v2.jsonl";
export const V2_MANIFEST_RELPATH = "corpora/generated/injection-p-fin-v2.manifest.json";
export const V2_QUEUE_RELPATH = "corpora/generated/injection-p-fin-v2.labelling-queue.jsonl";
export const V2_CORPUS_PATH = join(OUT_DIR, "injection-p-fin-v2.jsonl");
export const V2_MANIFEST_PATH = join(OUT_DIR, "injection-p-fin-v2.manifest.json");
export const V2_QUEUE_PATH = join(OUT_DIR, "injection-p-fin-v2.labelling-queue.jsonl");

export const FAMILY_BY_ID: ReadonlyMap<string, V2Family> = new Map(
  V2_FAMILIES.flatMap((f) => {
    const rows: [string, V2Family][] = [[f.id, f]];
    // The cross-segment companions are written under their own family id and
    // must be findable by it, or a contested-label question on one would be
    // silently dropped. They inherit the parent's record, which is correct:
    // the companion is the same surface in the same clause.
    if (f.companion !== undefined) rows.push([f.companion(ORG_POOL[0]).family, f]);
    return rows;
  }),
);

/**
 * Confusable families indexed by orthographic shape, for the distractor draw.
 *
 * A family that supplies its own shape neighbour is excluded from BOTH sides:
 * it gets no distractor (it already has one in its own clause) and it is never
 * used as one, because injecting a whole cross-segment block as somebody else's
 * distractor would put two ordinal role clauses in one message and make the
 * cross-segment measurement meaningless.
 */
const CONFUSABLES_BY_SHAPE = new Map<string, V2Family[]>();
for (const f of V2_CONFUSABLE_FAMILIES) {
  if (f.selfDistracting === true) continue;
  const list = CONFUSABLES_BY_SHAPE.get(f.shapeClass) ?? [];
  list.push(f);
  CONFUSABLES_BY_SHAPE.set(f.shapeClass, list);
}

/**
 * Throws rather than returning `undefined` for a positive with no shape
 * neighbour. A positive family whose class holds no confusable is a gold span
 * that is the only odd-looking thing in its message -- the exact defect the
 * distractor exists to remove -- and letting it through silently would restore
 * it for that one family with nothing in the manifest to say so.
 */
export function v2DistractorFor(family: Family, rng: () => number): Family | undefined {
  // Read off the family the generator actually dealt, not looked up by id. The
  // generator deals from `V2_POSITIVE_FAMILIES`, so the object already carries
  // `shapeClass`; an id lookup would make this untestable with a synthetic
  // family and would silently return `undefined` for one whose id had drifted.
  const v2 = family as Family & Partial<V2Family>;
  if (v2.shapeClass === undefined) return undefined;
  if (v2.type.startsWith(NEG_PREFIX)) return undefined;
  if (v2.selfDistracting === true) return undefined;
  const pool = CONFUSABLES_BY_SHAPE.get(v2.shapeClass);
  if (pool === undefined || pool.length === 0) {
    throw new Error(
      `positive family "${family.id}" has shapeClass "${v2.shapeClass}" and no confusable family ` +
        `shares it, so its gold span would be the only orthographically odd region in its message`,
    );
  }
  return pool[Math.floor(rng() * pool.length)]!;
}

/** Confusable types whose p-fin reading a careful reader can reach either way. */
export const CONTESTED_TYPES: readonly string[] = [
  ...new Set(V2_CONFUSABLE_FAMILIES.filter((f) => f.labelBasis.contestedBy !== undefined).map((f) => f.type)),
].sort();

export interface LeakageReport {
  readonly boost: BoostReport;
  readonly orthography: OrthographyReport;
  readonly roles: RoleReport;
  readonly irOverlap: {
    readonly labelledSpans: number;
    readonly spanValuesFoundInIr: readonly string[];
    readonly spanValuesFoundInSelfTest: readonly string[];
    readonly confusableFamilies: number;
    readonly confusableFamiliesSharingAnIrCounterExampleSurface: readonly string[];
    readonly irCounterExamplesAnnotated: number;
  };
  readonly comparedWith: string;
}

function measureLeakage(items: readonly CorpusItem[], ir: PolicyIr, irText: string, selfTestText: string): LeakageReport {
  const irSurfaces = new Set(IR_COUNTEREXAMPLE_SURFACES.map((c) => c.surfaceName));
  const values = new Set<string>();
  let labelled = 0;
  for (const item of items) {
    for (const inj of injectionsOfItem(item)) {
      labelled += 1;
      values.add(inj.text);
    }
  }
  return {
    boost: measureBoost(items, ir, PAIRED_TYPE),
    orthography: measureOrthography(items),
    roles: measureRoles(items, orgRoleClassOf),
    irOverlap: {
      labelledSpans: labelled,
      spanValuesFoundInIr: [...values].filter((v) => irText.includes(v)).sort(),
      spanValuesFoundInSelfTest: [...values].filter((v) => selfTestText.includes(v)).sort(),
      confusableFamilies: V2_CONFUSABLE_FAMILIES.length,
      confusableFamiliesSharingAnIrCounterExampleSurface: V2_CONFUSABLE_FAMILIES.filter((f) =>
        irSurfaces.has(f.surfaceName),
      ).map((f) => f.id),
      irCounterExamplesAnnotated: IR_COUNTEREXAMPLE_SURFACES.length,
    },
    comparedWith:
      "the before figures quoted in the module headers are these same functions run over " +
      "corpora/generated/injection-p-fin-adjudicated-v1.jsonl; corpus-v2.test.ts recomputes both.",
  };
}

export interface V2Manifest extends CorpusManifest {
  readonly supersedes: {
    readonly corpora: readonly string[];
    readonly why: string;
  };
  readonly inputs: {
    readonly carrierModules: readonly string[];
    readonly familyModules: readonly string[];
    readonly surfaceModule: string;
    readonly carriersOffered: readonly string[];
    readonly positiveFamilies: readonly string[];
    readonly confusableFamilies: readonly string[];
    readonly removedFamilies: readonly { readonly id: string; readonly why: string }[];
    readonly renamedFamilies: readonly { readonly from: string; readonly to: string; readonly why: string }[];
    readonly orgFamilies: readonly string[];
    readonly note: string;
  };
  readonly leakage: LeakageReport;
  readonly labelling: {
    readonly queuePath: string;
    readonly questions: number;
    readonly itemsWithAPredicateQuestion: number;
    readonly predicateConstructedTrue: number;
    readonly predicateConstructedFalse: number;
    readonly contestedTypes: readonly string[];
    readonly contestedSpanQuestions: number;
    /** The predicate question verbatim, once. The items carry only its id and state. */
    readonly predicateQuestion: string;
    readonly answeredBy: string;
    readonly why: string;
  };
  readonly adjudication: {
    readonly round: typeof ADJUDICATION_ROUND;
    readonly admissionRule: string;
    readonly agreement: AgreementReport;
    readonly admitted: readonly string[];
  };
  readonly verification: VerificationReport;
  readonly unvalidated: readonly string[];
  readonly artifact: {
    readonly corpusSha256: string;
    readonly corpusBytes: number;
    readonly corpusPath: string;
    readonly queueSha256: string;
    readonly queueBytes: number;
    readonly queuePath: string;
  };
}

export interface BuiltV2Artifacts {
  readonly corpusJsonl: string;
  readonly queueJsonl: string;
  readonly manifestJson: string;
  readonly generated: GeneratedCorpus;
  readonly items: readonly CorpusItem[];
  readonly manifest: V2Manifest;
}

export function buildV2Artifacts(seed: string = V2_SEED): BuiltV2Artifacts {
  const irText = readFileSync(IR_PATH, "utf8");
  const ir = loadPolicyIr(irText);
  const selfTestExamples = loadSelfTestExamples();
  const selfTestText = readFileSync(SELFTEST_PATH, "utf8");
  const scoringScope = scoringScopeFor(CONTESTED_TYPES);

  const generated = generateCorpus({
    seed,
    ir,
    irSource: "policies/compiled/p-fin.ir.json",
    irHash: createHash("sha256").update(irText, "utf8").digest("hex"),
    carriers: OFFERED_CARRIERS,
    positiveFamilies: V2_POSITIVE_FAMILIES,
    confusableFamilies: V2_CONFUSABLE_FAMILIES,
    selfTestExamples,
    distractorFor: v2DistractorFor,
    certifyOptions: {
      supplementarySweeps: [
        { id: "orthographic-org-sweep", sweep: orthographicOrgSweep },
        { id: ADJUDICATION_SWEEP_ID, sweep: adjudicationSweep },
        // The IR-free sweep. It can quarantine, and the certification summary
        // reports how many carriers stage 1 removed that it would have kept --
        // which is the size of the circularity stage 1 carries by being the arm
        // under test. See format-sweep.ts.
        { id: FORMAT_SPEC_SWEEP_ID, sweep: formatSpecSweep },
      ],
    },
    itemMetaExtra: ({ carrier, recipeIndex, written }) => {
      const injections = written.map((w) => ({
        start: w.span.start,
        end: w.span.end,
        text: w.span.text,
        family: w.family,
        dimensions: w.dimensions,
      }));
      // The id the generator is about to give this item. Duplicated here rather
      // than passed in, and asserted equal in corpus-v2.test.ts: a question id
      // that did not match its item's id would split the queue from the corpus
      // and no invariant in this pipeline would notice.
      const itemId = recipeIndex === undefined ? `neg-${carrier.id}` : `inj-${carrier.id}-${recipeIndex}`;
      return {
        predicateConstruction: predicateConstructedFor(injections),
        labelQuestions: compactQuestions(labelQuestionsFor(itemId, injections, FAMILY_BY_ID), V2_QUEUE_RELPATH),
        scoringScope,
      };
    },
  });

  const corpusJsonl = serializeCorpus(generated.items);
  // Read back from the serialized bytes, not from `generated.items`: the JSON
  // round trip is the one step every downstream consumer depends on, and
  // verifying the in-memory objects would leave it unchecked.
  const items = loadCorpus(corpusJsonl);
  const verification = verifyOrRefuse(items, generated.certifications, V2_CORPUS_RELPATH);
  const rows = queueRows(items, FAMILY_BY_ID);
  const queueJsonl = serializeQueue(rows);

  const constructed = items.map((i) => predicateConstructedFor(injectionsOfItem(i)));
  const manifest: V2Manifest = {
    ...generated.manifest,
    supersedes: {
      corpora: [
        "corpora/generated/injection-p-fin-v1.jsonl",
        "corpora/generated/injection-p-fin-adjudicated-v1.jsonl",
      ],
      why:
        "both were built from inputs that leak the answer to the compiled arm: their confusable " +
        "inventories are largely an enumeration of the IR's own counterExamples, a contextBoost term " +
        "predicts the label, the gold span is usually the only odd-looking token in its message, and " +
        "the organisation role is readable off the name. The exact rates are in this manifest's " +
        "leakage block, measured by the same functions on both corpora. They are kept, and " +
        "reproducible, because deleting a measurement that was made is not correcting it.",
    },
    inputs: {
      carrierModules: ["apps/eval/src/corpus/carriers.ts", "apps/eval/src/corpus/carriers.candidate.ts"],
      familyModules: ["apps/eval/src/corpus/families.v2.ts"],
      surfaceModule: "apps/eval/src/corpus/surfaces.ts",
      carriersOffered: OFFERED_CARRIERS.map((c) => c.id),
      positiveFamilies: V2_POSITIVE_FAMILIES.map((f) => f.id),
      confusableFamilies: V2_CONFUSABLE_FAMILIES.map((f) => f.id),
      removedFamilies: REMOVED_FAMILIES,
      renamedFamilies: RENAMED_FAMILIES,
      orgFamilies: ORG_FAMILY_IDS,
      note:
        "the carriers are the wave-1 and wave-2 pools UNCHANGED, because the 27 this corpus injects " +
        "into are the ones a blind two-certifier round admitted and editing a carrier voids its " +
        "verdict. Everything else -- surfaces, the organisation universe, the families, the " +
        "distractor composition -- is rebuilt.",
    },
    leakage: measureLeakage(items, ir, irText, selfTestText),
    labelling: {
      queuePath: V2_QUEUE_RELPATH,
      questions: rows.length,
      itemsWithAPredicateQuestion: rows.filter((r) => r.kind === "message-predicate").length,
      predicateConstructedTrue: constructed.filter((c) => c.constructed).length,
      predicateConstructedFalse: constructed.filter((c) => !c.constructed).length,
      contestedTypes: CONTESTED_TYPES,
      contestedSpanQuestions: rows.filter((r) => r.kind === "contested-span-label").length,
      predicateQuestion: PREDICATE_QUESTION,
      answeredBy: PREDICATE_ANSWERED_BY,
      why:
        `every item carries the ${PREDICATE_ID} question, positives and negatives alike, so the blind ` +
        "round that answers it produces gold with both classes in it. No pred: gold is emitted here, " +
        "and meta.scoringScope names the predicate as unlabelled on every item, so a finding of that " +
        "class is neither a match nor a false positive until the round has run. The previous corpus " +
        "constructed the same violation in 25 of its positives, emitted no gold for it, and scored a " +
        "model that flagged it as wrong.",
    },
    adjudication: {
      round: ADJUDICATION_ROUND,
      admissionRule: ADMISSION_RULE,
      agreement: agreementReport(),
      admitted: ADMITTED_CARRIER_IDS,
    },
    verification,
    unvalidated: [
      "CARRIER REALISM IS UNVALIDATED. The carriers are hand-authored, not the ShareChat/WildChat " +
        "conversations spec 6.2 specifies, and none of its three realism gates ran: no frontier " +
        "naturalness score, no adversarial style probe, no human spot check. The style probe is the " +
        "one this pool would most likely fail -- the carriers and the injections share an author.",
      "CERTIFICATION STAGES 2 AND 3 ARE STILL UNRUN, so no carrier is certified-clear and this corpus " +
        "is NOT certified. What is new is that stage 1's own circularity is now measured rather than " +
        "unmentioned: certification.circularity counts the carriers the tier-0 arm quarantined that " +
        "the IR-free format-spec sweep would have kept.",
      "p-med AND p-corp LABELS ARE UNPOPULATED. Any per-policy or policy-adaptivity number computed " +
        "from this corpus is a p-fin number.",
      `NO pred: GOLD, AND THE PREDICATE IS CONSTRUCTED ON BOTH SIDES. meta.predicateConstruction ` +
        "records what the generator wrote as a PREDICTION with its provenance; meta.labelQuestions " +
        "carries the question a blind round answers; meta.scoringScope excludes the class from the " +
        "false-positive claim until it does.",
      "THE CONTESTED CONFUSABLES ARE EXCLUDED FROM THE FALSE-POSITIVE CLAIM TOO. Their p-fin reading " +
        "can be reached either way from the document, each family says which clause would overturn it, " +
        "and each span carries its own question in the queue.",
      "THE ADJUDICATION ROUND THAT ADMITTED THE CARRIERS WAS NOT BLIND OF THE AUTHORING INTENT. See " +
        "adjudication.round.blindness.breaches. Nothing about that changed this round.",
    ],
    artifact: {
      corpusSha256: createHash("sha256").update(corpusJsonl, "utf8").digest("hex"),
      corpusBytes: Buffer.byteLength(corpusJsonl, "utf8"),
      corpusPath: V2_CORPUS_RELPATH,
      queueSha256: createHash("sha256").update(queueJsonl, "utf8").digest("hex"),
      queueBytes: Buffer.byteLength(queueJsonl, "utf8"),
      queuePath: V2_QUEUE_RELPATH,
    },
  };

  return { corpusJsonl, queueJsonl, manifestJson: serializeManifest(manifest), generated, items, manifest };
}

export function writeV2Artifacts(): BuiltV2Artifacts {
  const built = buildV2Artifacts();
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(V2_CORPUS_PATH, built.corpusJsonl, "utf8");
  writeFileSync(V2_MANIFEST_PATH, built.manifestJson, "utf8");
  writeFileSync(V2_QUEUE_PATH, built.queueJsonl, "utf8");
  return built;
}

/**
 * Not runnable under plain node, for the reason `build-adjudicated.ts` records
 * and MEASURED again here: `node --experimental-strip-types` cannot resolve
 * `@sih/core`, whose package `main` is `src/index.ts` with `.js` internal
 * specifiers. What writes the committed artifacts is vitest's bundled vite-node
 * applying vite's resolver. The reproducibility guarantee does not depend on
 * either: `corpus-v2.test.ts` regenerates all three files in memory on every
 * `pnpm -r test` and compares them to the committed bytes.
 */
if (process.argv[1] !== undefined && import.meta.url === `file://${process.argv[1]}`) {
  const built = writeV2Artifacts();
  process.stdout.write(
    `${GENERATOR_NAME} v${GENERATOR_VERSION} seed ${V2_SEED}: ${built.manifest.counts.items} items, ` +
      `${built.manifest.counts.goldSpans} gold spans, ${built.manifest.labelling.questions} label questions, ` +
      `sha256 ${built.manifest.artifact.corpusSha256}\n`,
  );
}
