import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadPolicyIr } from "@sih/core";
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
  refusalFor,
  type AgreementReport,
} from "./adjudication.js";
import { IR_PATH, OUT_DIR, loadSelfTestExamples } from "./build.js";
import { ALL_CARRIERS } from "./carriers.js";
import { CANDIDATE_CARRIERS } from "./carriers.candidate.js";
import { orthographicOrgSweep, type CarrierCertification } from "./certify.js";
import { CONFUSABLE_FAMILIES, POSITIVE_FAMILIES, type Family } from "./families.js";
import { CANDIDATE_CONFUSABLE_FAMILIES, CANDIDATE_POSITIVE_FAMILIES } from "./families.candidate.js";
import {
  GENERATOR_NAME,
  GENERATOR_VERSION,
  generateCorpus,
  serializeCorpus,
  serializeManifest,
  type CorpusManifest,
  type GeneratedCorpus,
} from "./generate.js";
import { isIrBacked } from "./labels.js";

/**
 * The second corpus: wave-1 and wave-2 families injected into the carriers that
 * a blind double-adjudication round admitted, and nothing else.
 *
 * ## Why this is a SECOND artifact and not an edit of the first
 *
 * `corpora/generated/injection-p-fin-v1` reproduces byte for byte from
 * `build.ts`, and `corpus-artifact.test.ts` holds that line. Widening
 * `ALL_CARRIERS` or `ALL_FAMILIES` would silently rewrite it, so nothing here
 * touches either: the carrier pool and the family catalogues are passed
 * explicitly. The two corpora share a generator and share no item id -- v1's
 * ids are built from wave-1 carrier ids (`c01`..`c22`, `d01`..`d03`) and this
 * one's from wave-2 ids (`o01`..`o16`, `hn02`..`hn12`) -- so records scored
 * against one can never be pooled with records scored against the other by
 * accident. The seed differs too and reaches `meta.seed` on every item.
 *
 * ## The name says "adjudicated", not "certified", on purpose
 *
 * `certificationSummary().claim` on this corpus still reads "NOT CERTIFIED" and
 * every carrier in it is `provisional-clear`, because spec 6.2's stage 2
 * (high-recall Python model sweep) never ran and its stage 3 wants the union of
 * three policies where this round read one. What IS new is that every admitted
 * carrier was read end to end by two independent certifiers against
 * `policies/p-fin.md` and cleared by both without reservation. Naming the file
 * "certified" would claim the two unrun stages.
 *
 * ## What the invariant is, stated as what was checked
 *
 * `verifyOrRefuse` re-reads the SERIALIZED corpus -- the bytes that will be on
 * disk, through `loadCorpus`, not the in-memory objects the generator returned
 * -- and refuses to emit unless, for every item: every gold span satisfies
 * `text.slice(start, end) === span.text`, every gold span is one the generator
 * wrote (matched against `meta.injections` by offset AND by type, not by
 * search), every injected value occurs exactly as many times as it was
 * injected, and the item's carrier is one both certifiers cleared with no
 * borderline flag and that no sweep hit.
 */

export const ADJUDICATED_SEED = "sih-p7-adjudicated-v1";
/**
 * Repo-relative, and used as such everywhere it reaches the manifest.
 *
 * `OUT_DIR` is absolute -- it is derived from `import.meta.url` -- so writing
 * the joined path into the artifact would stamp the build machine's home
 * directory into a committed file and make the byte-for-byte reproduction check
 * fail on every other checkout. Found by emitting once and reading the result.
 */
export const ADJUDICATED_CORPUS_RELPATH = "corpora/generated/injection-p-fin-adjudicated-v1.jsonl";
export const ADJUDICATED_MANIFEST_RELPATH = "corpora/generated/injection-p-fin-adjudicated-v1.manifest.json";
export const ADJUDICATED_CORPUS_PATH = join(OUT_DIR, "injection-p-fin-adjudicated-v1.jsonl");
export const ADJUDICATED_MANIFEST_PATH = join(OUT_DIR, "injection-p-fin-adjudicated-v1.manifest.json");

/**
 * Wave 1's `upi-vpa` family is excluded, and it is the only exclusion.
 *
 * It mints `<7 random lowercase alnum>@okaxis`, and `okaxis` is a live UPI
 * handle belonging to a real bank -- so every value it produces is a
 * well-formed address in a real payment namespace, which is the one thing spec
 * 6.2's "releasable by construction (CC-BY carriers + fictional entities)"
 * rules out. `families.candidate.ts` already flagged it ("worth fixing in the
 * wiring commit") and could not fix it in place without rewriting the committed
 * v1 corpus. `upi-vpa-fictional-handle` mints the same shape at the invented
 * handle `okbluecrest` and carries the same `bank-account-identifier` type, so
 * the surface is not lost.
 */
export const EXCLUDED_FAMILY_IDS: readonly string[] = ["upi-vpa"];

const EXCLUSION_REASONS: Readonly<Record<string, string>> = {
  "upi-vpa":
    "mints a syntactically valid UPI address at okaxis, a live handle of a real bank; " +
    "upi-vpa-fictional-handle covers the same entityType and surface at an invented handle",
};

/**
 * Every carrier this repository has, offered to the pipeline together. The
 * quarantined ones are offered deliberately: dropping them upstream would make
 * the reported quarantine rate a number about a pool that was curated first.
 */
export const OFFERED_CARRIERS = [...ALL_CARRIERS, ...CANDIDATE_CARRIERS];

export const ADJUDICATED_POSITIVE_FAMILIES: readonly Family[] = [
  ...POSITIVE_FAMILIES.filter((f) => !EXCLUDED_FAMILY_IDS.includes(f.id)),
  ...CANDIDATE_POSITIVE_FAMILIES,
];
export const ADJUDICATED_CONFUSABLE_FAMILIES: readonly Family[] = [
  ...CONFUSABLE_FAMILIES.filter((f) => !EXCLUDED_FAMILY_IDS.includes(f.id)),
  ...CANDIDATE_CONFUSABLE_FAMILIES,
];

export interface VerificationReport {
  readonly source: string;
  readonly itemsChecked: number;
  readonly positives: number;
  readonly negatives: number;
  readonly goldSpansChecked: number;
  readonly carriersUsed: readonly string[];
  readonly checks: readonly string[];
}

const VERIFICATION_CHECKS: readonly string[] = [
  "every gold span: end <= text.length and text.slice(start, end) === span.text",
  "every gold span is one the generator wrote: matched 1:1 against meta.injections by offset AND " +
    "by type, never by searching the text for the value",
  "every injected value occurs in the text exactly as many times as it was injected, so a finding " +
    "on it is provably a match or provably a false positive",
  "no gold span carries a neg: or pred: type",
  "the item's carrier was cleared by BOTH certifiers with NEITHER flagging borderline",
  "the item's carrier came through every sweep that ran with zero hits",
];

/**
 * The gate. Throws rather than returning a flag: a caller that could ignore the
 * result is a caller that will, and the whole point is that a violation stops
 * the emit.
 *
 * `items` must be the corpus read back out of its serialized form. Verifying
 * the generator's in-memory objects would leave the JSON round trip -- the one
 * step a downstream consumer actually depends on -- unchecked.
 */
export function verifyOrRefuse(
  items: readonly CorpusItem[],
  certifications: readonly CarrierCertification[],
  source: string,
): VerificationReport {
  const certById = new Map(certifications.map((c) => [c.carrierId, c]));
  const carriersUsed = new Set<string>();
  let goldSpansChecked = 0;
  let positives = 0;

  for (const item of items) {
    const carrierId = item.meta?.["carrierId"];
    if (typeof carrierId !== "string") {
      throw new Error(`refusing to emit: item ${item.id} records no meta.carrierId`);
    }
    carriersUsed.add(carrierId);

    if (!isAdmitted(carrierId)) {
      const refusal = refusalFor(carrierId)!;
      throw new Error(
        `refusing to emit: item ${item.id} was injected into carrier ${carrierId}, which the ` +
          `adjudication round refused (${refusal.label}): ${refusal.reason}`,
      );
    }
    const cert = certById.get(carrierId);
    if (cert === undefined) {
      throw new Error(`refusing to emit: item ${item.id} names carrier ${carrierId}, which was never certified`);
    }
    if (cert.status === "quarantined" || cert.hits.length > 0) {
      throw new Error(
        `refusing to emit: item ${item.id} was injected into carrier ${carrierId}, which is ` +
          `${cert.status} with ${cert.hits.length} sweep hit(s)`,
      );
    }

    const injections = (item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[];
    const counts = new Map<string, number>();
    for (const inj of injections) {
      const value = inj["text"];
      if (typeof value !== "string") {
        throw new Error(`refusing to emit: item ${item.id} has an injection with no text`);
      }
      counts.set(value, (counts.get(value) ?? 0) + 1);
    }
    for (const [value, expected] of counts) {
      let seen = 0;
      for (let i = item.text.indexOf(value); i !== -1; i = item.text.indexOf(value, i + 1)) seen += 1;
      if (seen !== expected) {
        throw new Error(
          `refusing to emit: item ${item.id} injected ${JSON.stringify(value)} ${expected} time(s) ` +
            `but it occurs ${seen} time(s) in the text`,
        );
      }
    }

    if (item.gold.length > 0) positives += 1;
    for (const gold of item.gold) {
      if (gold.end > item.text.length || item.text.slice(gold.start, gold.end) !== gold.text) {
        throw new Error(
          `refusing to emit: item ${item.id} gold span [${gold.start},${gold.end}) is ` +
            `${JSON.stringify(item.text.slice(gold.start, gold.end))}, not ${JSON.stringify(gold.text)}`,
        );
      }
      if (!isIrBacked(gold.entityType) || gold.entityType.startsWith("pred:")) {
        throw new Error(
          `refusing to emit: item ${item.id} gold span carries type "${gold.entityType}", which is ` +
            `not an IR entityType this corpus labels`,
        );
      }
      const written = injections.filter(
        (inj) => inj["start"] === gold.start && inj["end"] === gold.end && inj["text"] === gold.text,
      );
      if (written.length !== 1) {
        throw new Error(
          `refusing to emit: item ${item.id} gold span [${gold.start},${gold.end}) matches ` +
            `${written.length} recorded injections; a gold span must be exactly one thing the ` +
            `generator wrote`,
        );
      }
      if (written[0]!["type"] !== gold.entityType) {
        throw new Error(
          `refusing to emit: item ${item.id} gold span [${gold.start},${gold.end}) is typed ` +
            `"${gold.entityType}" but the injection that wrote it was typed ` +
            `"${String(written[0]!["type"])}"`,
        );
      }
      goldSpansChecked += 1;
    }
  }

  return {
    source,
    itemsChecked: items.length,
    positives,
    negatives: items.length - positives,
    goldSpansChecked,
    carriersUsed: [...carriersUsed].sort(),
    checks: VERIFICATION_CHECKS,
  };
}

export interface AdjudicationManifestBlock {
  readonly round: typeof ADJUDICATION_ROUND;
  readonly admissionRule: string;
  readonly agreement: AgreementReport;
  readonly counts: {
    readonly adjudicated: number;
    readonly admitted: number;
    readonly refused: number;
    readonly quarantineRateOfAdjudicatedPool: number;
    readonly carriersOffered: number;
    readonly carriersUnadjudicated: number;
    readonly quarantineRateOfOfferedPool: number;
  };
  readonly perCarrier: readonly {
    readonly carrierId: string;
    readonly admitted: boolean;
    readonly refusal?: { readonly label: string; readonly reason: string };
    readonly A: { readonly clear: boolean; readonly confidence: string; readonly rationale: string };
    readonly B: { readonly clear: boolean; readonly confidence: string; readonly rationale: string };
  }[];
  readonly unadjudicated: readonly string[];
  readonly invariantScopeAddendum: string;
}

export interface AdjudicatedManifest extends CorpusManifest {
  readonly inputs: {
    readonly carrierModules: readonly string[];
    readonly carriersOffered: readonly string[];
    readonly familyModules: readonly string[];
    readonly positiveFamilies: readonly string[];
    readonly confusableFamilies: readonly string[];
    readonly excludedFamilies: readonly { readonly id: string; readonly why: string }[];
    readonly note: string;
  };
  readonly adjudication: AdjudicationManifestBlock;
  readonly verification: VerificationReport;
  readonly unvalidated: readonly string[];
  readonly artifact: {
    readonly corpusSha256: string;
    readonly corpusBytes: number;
    readonly corpusPath: string;
    readonly goldNote: string;
  };
}

export interface BuiltAdjudicatedArtifacts {
  readonly corpusJsonl: string;
  readonly manifestJson: string;
  readonly generated: GeneratedCorpus;
  readonly items: readonly CorpusItem[];
  readonly manifest: AdjudicatedManifest;
  readonly verification: VerificationReport;
}

export function buildAdjudicatedArtifacts(seed: string = ADJUDICATED_SEED): BuiltAdjudicatedArtifacts {
  const irText = readFileSync(IR_PATH, "utf8");
  const generated = generateCorpus({
    seed,
    ir: loadPolicyIr(irText),
    irSource: "policies/compiled/p-fin.ir.json",
    irHash: createHash("sha256").update(irText, "utf8").digest("hex"),
    carriers: OFFERED_CARRIERS,
    positiveFamilies: ADJUDICATED_POSITIVE_FAMILIES,
    confusableFamilies: ADJUDICATED_CONFUSABLE_FAMILIES,
    selfTestExamples: loadSelfTestExamples(),
    certifyOptions: {
      // Order matters only for reading: both run, and either can quarantine.
      // The adjudication sweep is SUPPLEMENTARY, not stage 3 -- see
      // adjudication.ts for why one policy of three cannot be booked as the
      // stage that requires all three.
      supplementarySweeps: [
        { id: "orthographic-org-sweep", sweep: orthographicOrgSweep },
        { id: ADJUDICATION_SWEEP_ID, sweep: adjudicationSweep },
      ],
    },
  });

  const corpusJsonl = serializeCorpus(generated.items);
  // Read back from the serialized bytes, not from `generated.items`.
  const items = loadCorpus(corpusJsonl);
  const verification = verifyOrRefuse(items, generated.certifications, ADJUDICATED_CORPUS_RELPATH);

  const offeredIds = OFFERED_CARRIERS.map((c) => c.id);
  const adjudicatedIds = new Set(CARRIER_VERDICTS.map((v) => v.carrierId));
  const unadjudicated = offeredIds.filter((id) => !adjudicatedIds.has(id));
  const refused = CARRIER_VERDICTS.filter((v) => !isAdmitted(v.carrierId));

  const manifest: AdjudicatedManifest = {
    ...generated.manifest,
    inputs: {
      carrierModules: ["apps/eval/src/corpus/carriers.ts", "apps/eval/src/corpus/carriers.candidate.ts"],
      carriersOffered: offeredIds,
      familyModules: ["apps/eval/src/corpus/families.ts", "apps/eval/src/corpus/families.candidate.ts"],
      positiveFamilies: ADJUDICATED_POSITIVE_FAMILIES.map((f) => f.id),
      confusableFamilies: ADJUDICATED_CONFUSABLE_FAMILIES.map((f) => f.id),
      excludedFamilies: EXCLUDED_FAMILY_IDS.map((id) => ({ id, why: EXCLUSION_REASONS[id]! })),
      note:
        "the generator and its version are shared with corpora/generated/injection-p-fin-v1; the " +
        "inputs are not, and neither is the seed. This block is what distinguishes the two, " +
        "because generator.version alone cannot: it tracks a change in the generated bytes for a " +
        "FIXED seed and input set.",
    },
    adjudication: {
      round: ADJUDICATION_ROUND,
      admissionRule: ADMISSION_RULE,
      agreement: agreementReport(),
      counts: {
        adjudicated: CARRIER_VERDICTS.length,
        admitted: ADMITTED_CARRIER_IDS.length,
        refused: refused.length,
        quarantineRateOfAdjudicatedPool: refused.length / CARRIER_VERDICTS.length,
        carriersOffered: offeredIds.length,
        carriersUnadjudicated: unadjudicated.length,
        quarantineRateOfOfferedPool: (refused.length + unadjudicated.length) / offeredIds.length,
      },
      perCarrier: CARRIER_VERDICTS.map((v) => {
        const refusal = refusalFor(v.carrierId);
        return {
          carrierId: v.carrierId,
          admitted: refusal === undefined,
          ...(refusal === undefined ? {} : { refusal: { label: refusal.label, reason: refusal.reason } }),
          A: { clear: v.A.clear, confidence: v.A.confidence, rationale: v.A.rationale },
          B: { clear: v.B.clear, confidence: v.B.confidence, rationale: v.B.rationale },
        };
      }),
      unadjudicated,
      invariantScopeAddendum:
        "certification.invariantScope reads 'stage 1 only' because spec 6.2's stages 2 and 3 did " +
        "not run, and that sentence is left alone because it errs conservative. What this corpus " +
        "adds on top of it: every carrier injected into was read end to end by two independent " +
        "certifiers against policies/p-fin.md and cleared by both with no borderline flag, which " +
        "is a guard stage 1 cannot provide for client-name (tier 1) or for " +
        "pred:client-relationship-disclosure (tier 2). It is NOT a guard against p-med or p-corp " +
        "content, which no certifier read.",
    },
    verification,
    unvalidated: [
      "CARRIER REALISM IS UNVALIDATED. Carriers are hand-authored, not the ShareChat/WildChat " +
        "conversations spec 6.2 specifies, and none of its three realism gates ran: no frontier " +
        "naturalness score, no adversarial style probe, no human spot check. The style probe is " +
        "the one this pool would most likely fail -- the carriers and the injections share an " +
        "author. No claim is made that any item here reads like a real prompt.",
      "p-med AND p-corp LABELS ARE UNPOPULATED. Every span carries violatesUnder for all three " +
        "policies; only p-fin holds an action. policies/p-med.md and policies/p-corp.md have never " +
        "been compiled, so there is no actions table to read from, and reading an action out of an " +
        "unpopulated label throws rather than defaulting. Any per-policy or policy-adaptivity " +
        "number computed from this corpus is a p-fin number.",
      "NO pred: GOLD. Items carrying an organisation are stratified by injections[].dimensions." +
        "constructedRole and marked meta.predicateLabelling.state = 'unlabelled'. constructedRole " +
        "records the clause the generator wrote, which is a fact about generation and not an " +
        "adjudicated judgement about the finished text.",
      "THE ADJUDICATION ROUND WAS NOT BLIND OF THE AUTHORING INTENT. Both certifiers read the " +
        "carrier file, whose header comment names which carriers were written to fail and which " +
        "one is contestable. See adjudication.round.blindness.breaches.",
    ],
    artifact: {
      corpusSha256: createHash("sha256").update(corpusJsonl, "utf8").digest("hex"),
      corpusBytes: Buffer.byteLength(corpusJsonl, "utf8"),
      corpusPath: ADJUDICATED_CORPUS_RELPATH,
      goldNote:
        "gold lives INSIDE the corpus, one `gold` array per item, plus the full three-policy label " +
        "in meta.labels. There is no companion gold file: a second copy of the same spans is a " +
        "second thing to keep in sync. The only separate gold set in this repository is " +
        "corpora/fixtures/smoke.gold-tier2.jsonl, which holds tier-2 PREDICATE labels; this corpus " +
        "emits none, so it adds nothing to that file.",
    },
  };

  return { corpusJsonl, manifestJson: serializeManifest(manifest), generated, items, manifest, verification };
}

export function writeAdjudicatedArtifacts(): BuiltAdjudicatedArtifacts {
  const built = buildAdjudicatedArtifacts();
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(ADJUDICATED_CORPUS_PATH, built.corpusJsonl, "utf8");
  writeFileSync(ADJUDICATED_MANIFEST_PATH, built.manifestJson, "utf8");
  return built;
}

/**
 * MEASURED on node v26.0.0, because `build.ts`'s own comment claims a command
 * that does not work: `node --experimental-strip-types <this file>` fails with
 * ERR_MODULE_NOT_FOUND -- `@sih/core`'s package `main` is `src/index.ts` and its
 * internal specifiers end in `.js`, which node's type stripping does not remap
 * -- and adding a resolver hook that does remap them then fails with
 * ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX. What ran, and what wrote the committed
 * artifact, is vitest's bundled vite-node applying vite's resolver:
 *
 *   node node_modules/.pnpm/vite-node@<v>/node_modules/vite-node/vite-node.mjs \
 *     --options.root=apps/eval <entry calling writeAdjudicatedArtifacts()>
 *
 * The entry has to call the writer: vite-node sets `process.argv[1]` to its own
 * path, so the guard below never fires under it. `vite-node` is a transitive
 * dependency with no bin link and no package script, so that command is not
 * stable across installs and is recorded rather than recommended.
 *
 * None of that is the reproducibility guarantee. `corpus-adjudicated.test.ts`
 * regenerates both artifacts in memory on every `pnpm -r test` and compares
 * them to the committed bytes, which is the check that the file on disk is the
 * one this code produces.
 */
if (process.argv[1] !== undefined && import.meta.url === `file://${process.argv[1]}`) {
  const built = writeAdjudicatedArtifacts();
  process.stdout.write(
    `${GENERATOR_NAME} v${GENERATOR_VERSION} seed ${ADJUDICATED_SEED}: ${built.manifest.counts.items} items, ` +
      `${built.manifest.counts.positives} positives, ${built.manifest.counts.goldSpans} gold spans, ` +
      `${built.manifest.certification.carriers.quarantined} of ${built.manifest.certification.carriers.total} ` +
      `carriers quarantined, sha256 ${built.manifest.artifact.corpusSha256}\n`,
  );
}
