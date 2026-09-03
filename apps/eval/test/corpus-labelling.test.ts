import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { loadCorpus, type CorpusItem } from "../src/driver/corpus.js";
import { loadTier2Gold } from "../src/driver/score.js";
import { IR_PATH, REPO_ROOT } from "../src/corpus/build.js";
import { V2_CORPUS_PATH, V2_QUEUE_PATH, buildV2Artifacts } from "../src/corpus/build-v2.js";
import {
  DISPUTED_CONTESTED_TYPES,
  LABELLED_CORPUS_PATH,
  LABELLED_GOLD_PATH,
  LABELLED_MANIFEST_PATH,
  LABELLED_VERIFICATION_CHECKS,
  RESOLVED_CONTESTED_TYPES,
  SMOKE_TIER2_GOLD_PATH,
  assertSourceBytesUnchanged,
  buildLabelledArtifacts,
  labelledItems,
  compiledArmReachByType,
  serializeTier2Gold,
  sha256,
  tier2GoldRows,
  verifyLabelledOrRefuse,
  zeroEventUpperBound95,
} from "../src/corpus/build-labelled.js";
import {
  ANNOTATORS,
  ANNOTATOR_LABELS,
  BRIEF_PARAPHRASE_FRAGMENT,
  assertBothAnnotatorsLabelledTheSameItems,
  LABELLED_ITEM_IDS,
  LABELS_BY_ITEM,
  LABEL_ROUND,
  SPAN_ADJUDICATIONS,
  assertFamilyUniformity,
  contestedSpansOf,
  labelAgreementReport,
  overrideReport,
  pairwiseAgreement,
  predicateAdjudication,
  predicateVerdictFor,
  type AnnotatorLabel,
} from "../src/corpus/labelling.js";
import { CARRIER_VERDICTS, agreementOn } from "../src/corpus/adjudication.js";
import { REMOVED_FAMILIES } from "../src/corpus/families.v2.js";
import { PREDICATE_ANSWERED_BY, PREDICATE_QUESTION } from "../src/corpus/questions.js";
import { serializeCorpus } from "../src/corpus/generate.js";

/**
 * The round's numbers, its blindness audit, and the gate that refuses to emit
 * gold about text nobody read.
 *
 * Every agreement figure asserted below is worked out by hand from the 20-row
 * label table and written here as a literal, not read back from the function
 * that computes it -- a test whose expectation is the output of the thing under
 * test proves only that the code is deterministic. The corpus-derived counts
 * (how many contested spans, of which types, how many construction positives)
 * are recomputed here from the COMMITTED JSONL with independent code, for the
 * same reason.
 */

const built = buildLabelledArtifacts();
const V2 = buildV2Artifacts();
const IR_TEXT = readFileSync(IR_PATH, "utf8");
const IR = loadPolicyIr(IR_TEXT);
const POLICY_TEXT = readFileSync(join(REPO_ROOT, "policies/p-fin.md"), "utf8");
const COMMITTED_V2 = loadCorpus(readFileSync(V2_CORPUS_PATH, "utf8"));
const COMMITTED_QUEUE = readFileSync(V2_QUEUE_PATH, "utf8")
  .split("\n")
  .filter((l) => l.trim() !== "")
  .map((l) => JSON.parse(l) as Record<string, unknown>);

/** Contested spans recomputed from the committed queue and corpus, without the module's helper. */
const CONTESTED_BY_TYPE = new Map<string, string[]>();
for (const row of COMMITTED_QUEUE) {
  if (row["kind"] !== "contested-span-label") continue;
  const span = row["span"] as { start: number; end: number };
  const item = COMMITTED_V2.find((i) => i.id === row["itemId"])!;
  const label = ((item.meta?.["labels"] ?? []) as { span: { start: number; end: number }; type: string }[]).find(
    (l) => l.span.start === span.start && l.span.end === span.end,
  )!;
  CONTESTED_BY_TYPE.set(label.type, [...(CONTESTED_BY_TYPE.get(label.type) ?? []), item.id]);
}

describe("the committed artifacts are what this code produces", () => {
  it("reproduces the labelled corpus, the tier-2 gold and the manifest byte for byte", () => {
    expect(readFileSync(LABELLED_CORPUS_PATH, "utf8")).toBe(built.corpusJsonl);
    expect(readFileSync(LABELLED_GOLD_PATH, "utf8")).toBe(built.goldJsonl);
    expect(readFileSync(LABELLED_MANIFEST_PATH, "utf8")).toBe(built.manifestJson);
  });

  it("records digests of the bytes it wrote", () => {
    expect(built.manifest.artifact.corpusSha256).toBe(sha256(built.corpusJsonl));
    expect(built.manifest.artifact.goldSha256).toBe(sha256(built.goldJsonl));
  });
});

describe("the corpus and queue the annotators read are byte-unchanged", () => {
  it("hashes both source files to what the v2 builder produces", () => {
    expect(sha256(readFileSync(V2_CORPUS_PATH, "utf8"))).toBe(V2.manifest.artifact.corpusSha256);
    expect(sha256(readFileSync(V2_QUEUE_PATH, "utf8"))).toBe(V2.manifest.artifact.queueSha256);
    expect(built.manifest.source.corpusSha256).toBe(V2.manifest.artifact.corpusSha256);
    expect(built.manifest.source.queueSha256).toBe(V2.manifest.artifact.queueSha256);
  });

  it("refuses to emit if either source file has moved", () => {
    expect(() =>
      assertSourceBytesUnchanged({ corpusSha256: "0".repeat(64), queueSha256: V2.manifest.artifact.queueSha256 }),
    ).toThrow(/injection-p-fin-v2\.jsonl is sha256/);
    expect(() =>
      assertSourceBytesUnchanged({ corpusSha256: V2.manifest.artifact.corpusSha256, queueSha256: "0".repeat(64) }),
    ).toThrow(/labelling-queue\.jsonl is sha256/);
  });
});

describe("agreement, reported before anything is resolved", () => {
  const report = labelAgreementReport();

  it("computes raw agreement and kappa the hand-worked values", () => {
    // 20 items. A answered spanLabelCorrect true on all 20; B answered true on
    // 13 and false on 7, so they agree on 13. po = 13/20 = 0.65. Chance
    // agreement with A's marginals (20 true, 0 false) and B's (13, 7) is
    // (20/20)(13/20) + (0/20)(7/20) = 0.65, so kappa = (0.65-0.65)/(1-0.65) = 0.
    expect(report.spanLabelCorrect.n).toBe(20);
    expect(report.spanLabelCorrect.agreements).toBe(13);
    expect(report.spanLabelCorrect.rawAgreement).toBeCloseTo(0.65, 12);
    expect(report.spanLabelCorrect.expectedAgreement).toBeCloseTo(0.65, 12);
    expect(report.spanLabelCorrect.cohensKappa).toBeCloseTo(0, 12);
    expect(report.spanLabelCorrect.marginals).toEqual({ A: { false: 0, true: 20 }, B: { false: 7, true: 13 } });

    // Both answered false on all 20, so po = 1 and pe = 1 and kappa is undefined.
    expect(report.satisfiesPredicate.rawAgreement).toBe(1);
    expect(report.satisfiesPredicate.expectedAgreement).toBe(1);
    expect(report.satisfiesPredicate.cohensKappa).toBeNull();

    // A flagged 2 borderline, B flagged 1, and they differ on exactly one item.
    // po = 19/20 = 0.95; pe = (18/20)(19/20) + (2/20)(1/20) = 0.855 + 0.005 = 0.86;
    // kappa = (0.95 - 0.86) / (1 - 0.86) = 0.09/0.14 = 9/14.
    expect(report.predicateConfidence.rawAgreement).toBeCloseTo(0.95, 12);
    expect(report.predicateConfidence.expectedAgreement).toBeCloseTo(0.86, 12);
    expect(report.predicateConfidence.cohensKappa).toBeCloseTo(9 / 14, 12);
    expect(report.predicateConfidence.disagreements).toEqual(["inj-o15-1: A=borderline B=clear"]);
  });

  it("says WHY each kappa cannot be read as a reliability figure", () => {
    expect(report.spanLabelCorrect.kappaNote).toContain("DEGENERATE");
    expect(report.spanLabelCorrect.kappaNote).toContain("annotator A used a single category");
    expect(report.satisfiesPredicate.kappaNote).toContain("UNDEFINED");
    expect(report.predicateConfidence.kappaNote).toContain("skewed");
  });

  it("is the same arithmetic as the carrier round's, checked on the carrier round's own table", () => {
    // The two rounds label different things and neither table can exercise the
    // other's branches, so the check is that this function reproduces
    // `agreementOn`'s numbers when handed `agreementOn`'s data. Two definitions
    // of kappa in one package that silently disagreed would be the defect.
    for (const [field, project] of [
      ["clear", (v: { clear: boolean }) => String(v.clear)],
      ["confidence", (v: { confidence: string }) => v.confidence],
    ] as const) {
      const theirs = agreementOn(field, project as never);
      const mine = pairwiseAgreement(
        field,
        CARRIER_VERDICTS.map((v) => v.carrierId),
        (annotator, unit) => CARRIER_VERDICTS.find((v) => v.carrierId === unit)![annotator],
        project as never,
      );
      expect(mine.rawAgreement).toBe(theirs.rawAgreement);
      expect(mine.expectedAgreement).toBeCloseTo(theirs.expectedAgreement, 12);
      expect(mine.cohensKappa === null ? null : Number(mine.cohensKappa.toFixed(12))).toBe(
        theirs.cohensKappa === null ? null : Number(theirs.cohensKappa.toFixed(12)),
      );
      expect(mine.marginals).toEqual(theirs.marginals);
    }
  });

  it("computes a non-degenerate kappa on a table that has a genuine spread", () => {
    // Neither real table has both annotators using both categories, so neither
    // reaches the ordinary branch. 8 units: 3 agree yes, 3 agree no, A says yes
    // and B no once, A no and B yes once. po = 6/8 = 0.75; A's marginals are
    // (4 yes, 4 no) and B's the same, so pe = 0.5 and kappa = 0.5.
    const synthetic: Record<string, Record<"A" | "B", boolean>> = {
      u1: { A: true, B: true },
      u2: { A: true, B: true },
      u3: { A: true, B: true },
      u4: { A: false, B: false },
      u5: { A: false, B: false },
      u6: { A: false, B: false },
      u7: { A: true, B: false },
      u8: { A: false, B: true },
    };
    const got = pairwiseAgreement(
      "synthetic",
      Object.keys(synthetic),
      (a, u) => synthetic[u]![a],
      (v) => String(v),
    );
    expect(got.rawAgreement).toBeCloseTo(0.75, 12);
    expect(got.expectedAgreement).toBeCloseTo(0.5, 12);
    expect(got.cohensKappa).toBeCloseTo(0.5, 12);
    expect(got.kappaNote).toContain("skewed");
  });
});

describe("did the annotators judge or rubber-stamp", () => {
  const report = overrideReport(LABELLED_ITEM_IDS, (id) => {
    const item = COMMITTED_V2.find((i) => i.id === id)!;
    return (item.meta?.["predicateConstruction"] as { constructed: boolean }).constructed;
  });

  it("reports A at zero overrides and B at seven, on the span question", () => {
    const a = report.perAnnotator.find((r) => r.annotator === "A")!;
    const b = report.perAnnotator.find((r) => r.annotator === "B")!;
    expect(a.spanOverrides).toBe(0);
    expect(a.spanOverrideRate).toBe(0);
    expect(b.spanOverrides).toBe(7);
    expect(b.spanOverrideRate).toBeCloseTo(0.35, 12);
    expect(b.correctedTypesProposed).toEqual(["customer-data"]);
  });

  it("reports zero predicate overrides, and says why that number is worth nothing", () => {
    for (const r of report.perAnnotator) expect(r.predicateOverrides).toBe(0);
    expect(report.note).toContain("agreeing cost neither annotator a positive call");
  });

  it("proves B's correctedType is outside the IR's namespace, which is why it cannot be emitted", () => {
    expect(IR.entityTypes.map((e) => e.id)).not.toContain("customer-data");
    expect(Object.keys(IR.actions.default)).not.toContain("customer-data");
  });
});

describe("the blindness audit, checked against the artifacts rather than believed", () => {
  it("lists every channel with evidence and either an audit or a stated gap", () => {
    expect(LABEL_ROUND.blindness.channels.length).toBeGreaterThanOrEqual(4);
    for (const channel of LABEL_ROUND.blindness.channels) {
      expect(channel.what.length).toBeGreaterThan(0);
      expect(["audited", "partial", "unaudited"]).toContain(channel.audited);
      expect(channel.evidence.length).toBeGreaterThan(0);
      // A channel that is not fully audited must say what it cannot rule out.
      if (channel.audited !== "audited") expect(channel.gap.length).toBeGreaterThan(0);
    }
    expect(LABEL_ROUND.blindness.channels.map((c) => c.channel)).toEqual(["read", "told", "structural", "scope"]);
  });

  it("TOLD-1: the brief's predicate paraphrase is the COMPILED IR's own nlPredicate", () => {
    // The whole breach rests on where that wording comes from, and that is
    // checkable rather than a matter of trusting two self-reports.
    const nl = IR.semanticPredicates.map((p) => p.nlPredicate).join("\n");
    expect(nl).toContain(BRIEF_PARAPHRASE_FRAGMENT);
    expect(POLICY_TEXT).not.toContain(BRIEF_PARAPHRASE_FRAGMENT);
    expect(PREDICATE_QUESTION).not.toContain(BRIEF_PARAPHRASE_FRAGMENT);
    const told = LABEL_ROUND.blindness.channels.find((c) => c.channel === "told")!;
    expect(told.breaches.some((b) => b.startsWith("TOLD-1"))).toBe(true);
  });

  it("PGAP-3: that same nlPredicate drops the NDA limb the policy's §3.3 carries", () => {
    const nl = IR.semanticPredicates.map((p) => p.nlPredicate).join("\n").toLowerCase();
    expect(POLICY_TEXT.toLowerCase()).toContain("parties under a non-disclosure agreement");
    expect(nl).not.toContain("non-disclosure");
    expect(nl).not.toContain("nda");
    // And the corpus does carry items built on that limb, so the omission is live.
    const ndaItems = COMMITTED_V2.filter((i) =>
      ((i.meta?.["predicateConstruction"] as { constructedFrom: string[] }).constructedFrom ?? []).includes(
        "nda-party",
      ),
    );
    expect(ndaItems.length).toBeGreaterThan(0);
    expect(ndaItems.every((i) => !LABELLED_ITEM_IDS.includes(i.id))).toBe(true);
  });

  it("TOLD-2: the prior the brief supplied was about a family already removed from this corpus", () => {
    const removed = REMOVED_FAMILIES.find((f) => f.id === "csr-pem-block");
    expect(removed).toBeDefined();
    expect(removed!.why).toContain("§4.3");
    const typesInCorpus = new Set(
      COMMITTED_V2.flatMap((i) => ((i.meta?.["labels"] ?? []) as { type: string }[]).map((l) => l.type)),
    );
    expect(typesInCorpus.has("neg:csr-pem-block")).toBe(false);
  });

  it("STRUCTURAL-1: item ids leak the stratum, and 8 of the 20 labelled items carry it", () => {
    expect(LABELLED_ITEM_IDS.filter((id) => id.startsWith("inj-hn")).length).toBe(8);
    expect(COMMITTED_V2.some((i) => i.id.startsWith("inj-hn"))).toBe(true);
    expect(COMMITTED_V2.some((i) => i.id.startsWith("inj-o"))).toBe(true);
  });

  it("STRUCTURAL-2: the queue's question field names the span's type and pre-argues both readings", () => {
    const contested = COMMITTED_QUEUE.filter((r) => r["kind"] === "contested-span-label");
    expect(contested.length).toBe(20);
    for (const row of contested) {
      const q = String(row["question"]);
      expect(q).toMatch(/The quoted span is a (sftp file-transfer endpoint|NPCI retrieval reference number)/);
      expect(q).toContain("One reading says no:");
      expect(q).toContain("Another says yes:");
      expect(q).toContain("§");
    }
    const predicateRow = COMMITTED_QUEUE.find((r) => r["kind"] === "message-predicate")!;
    expect(String(predicateRow["question"])).toContain("sections 3.1, 3.3 and 3.4 only");
    // The claim these two disclosures falsify is committed in questions.ts and
    // serialized into the v2 manifest, so it is quoted here rather than described.
    expect(PREDICATE_ANSWERED_BY).toContain("an annotator cannot read the generator's intent off the row");
  });

  it("counts the queue it describes, rather than describing it from memory", () => {
    const q = LABEL_ROUND.questionsAsked;
    expect(q.inQueue).toBe(COMMITTED_QUEUE.length);
    expect(q.messagePredicate).toBe(COMMITTED_QUEUE.filter((r) => r["kind"] === "message-predicate").length);
    expect(q.contestedSpanLabel).toBe(COMMITTED_QUEUE.filter((r) => r["kind"] === "contested-span-label").length);
    expect(q.answeredByBothAnnotators).toBe(LABELLED_ITEM_IDS.length);
    expect(q.questionsUnanswered).toBe(q.inQueue - q.answeredByBothAnnotators);
    // The number the two annotators' notes get wrong, and the reason the field
    // is spelled out beside the question count rather than left to arithmetic.
    expect(q.itemsWithNoPredicateAnswer).toBe(COMMITTED_V2.length - LABELLED_ITEM_IDS.length);
    expect(q.itemsWithNoPredicateAnswer).not.toBe(q.questionsUnanswered);
  });

  it("SCOPE-1: the 20 covered items are disjoint from the 19 the generator predicts positive", () => {
    const constructionPositives = COMMITTED_V2.filter(
      (i) => (i.meta?.["predicateConstruction"] as { constructed: boolean }).constructed,
    ).map((i) => i.id);
    expect(constructionPositives.length).toBe(19);
    expect(constructionPositives.filter((id) => LABELLED_ITEM_IDS.includes(id))).toEqual([]);
    expect(built.manifest.coverage.constructionTest.constructionPositivesCovered).toBe(0);
    expect(built.manifest.coverage.itemsUnanswered).toBe(169);
  });
});

describe("the adjudication", () => {
  it("faces exactly two contested families, 13 sftp and 7 retrieval references", () => {
    // Recomputed above from the committed queue joined to the committed corpus,
    // which is also the fact that settles annotator A's own 14/6 prose count.
    expect([...CONTESTED_BY_TYPE.keys()].sort()).toEqual(["neg:retrieval-reference", "neg:sftp-endpoint"]);
    expect(CONTESTED_BY_TYPE.get("neg:sftp-endpoint")!.length).toBe(13);
    expect(CONTESTED_BY_TYPE.get("neg:retrieval-reference")!.length).toBe(7);
    expect(LABEL_ROUND.discrepancies.some((d) => d.includes("13 sftp and 7 RRN"))).toBe(true);
  });

  it("measures the clause claim the retrieval-reference ruling turns on", () => {
    // The adjudication says §1.2 is operative through §5 and not through §2-§4,
    // and rests that on where the phrase "customer data" occurs. Measured over
    // the numbered clauses rather than asserted from one reading.
    const clauses = [...POLICY_TEXT.matchAll(/^(§\d+\.\d+)([^\n]*)$/gm)].map((m) => ({
      id: m[1]!,
      body: m[2]!.toLowerCase(),
    }));
    expect(clauses.length).toBeGreaterThan(20);
    expect(clauses.filter((c) => c.body.includes("customer data")).map((c) => c.id)).toEqual([
      "§1.2",
      "§5.1",
      "§5.2",
      "§5.4",
      "§5.5",
    ]);
    expect(SPAN_ADJUDICATIONS["neg:retrieval-reference"]!.reasoning).toContain(
      "in no numbered clause of §2, §3 or §4",
    );
    // And the other half of the ruling: the corrected type has no home in the IR.
    expect(IR.entityTypes.map((e) => e.id).length).toBe(9);
    expect(Object.keys(IR.actions.providerOverrides ?? {}).sort()).toEqual(["chatgpt", "deepseek", "gemini"]);
  });

  it("re-derives the generator's proposal: no contested span appears in any gold array", () => {
    for (const item of COMMITTED_V2) {
      for (const span of contestedSpansOf(item)) {
        expect(item.gold.some((g) => g.start === span.start && g.end === span.end)).toBe(false);
      }
    }
  });

  it("affirms the sftp family and disputes the retrieval references", () => {
    expect(SPAN_ADJUDICATIONS["neg:sftp-endpoint"]!.resolution).toBe("affirmed");
    expect(SPAN_ADJUDICATIONS["neg:sftp-endpoint"]!.scoredAs).toBe("true-false-positive");
    expect(SPAN_ADJUDICATIONS["neg:retrieval-reference"]!.resolution).toBe("disputed");
    expect(SPAN_ADJUDICATIONS["neg:retrieval-reference"]!.scoredAs).toBe("excluded");
    expect(RESOLVED_CONTESTED_TYPES).toEqual(["neg:sftp-endpoint"]);
    expect(DISPUTED_CONTESTED_TYPES).toEqual(["neg:retrieval-reference"]);
    expect(built.manifest.spanAdjudication.resolvedSpans).toBe(13);
    expect(built.manifest.spanAdjudication.disputedSpans).toBe(7);
    expect(built.manifest.spanAdjudication.disputedRate).toBeCloseTo(0.35, 12);
  });

  it("moves the resolved type into the false-positive claim and leaves the disputed one out", () => {
    const byId = new Map(built.items.map((i) => [i.id, i]));
    for (const [type, itemIds] of CONTESTED_BY_TYPE) {
      for (const id of itemIds) {
        const scope = byId.get(id)!.meta!["scoringScope"] as { unlabelledClasses: string[] };
        if (type === "neg:sftp-endpoint") expect(scope.unlabelledClasses).not.toContain(type);
        else expect(scope.unlabelledClasses).toContain(type);
      }
    }
  });

  it("refuses a family-level ruling when an annotator split the family", () => {
    // Every real item is answered uniformly within its family, so the real
    // table cannot reach this branch and a suite that only ran it would not
    // know the guard exists.
    const split = new Map<string, string[]>();
    for (const label of ANNOTATOR_LABELS.B) split.set(label.itemId, ["one-family"]);
    expect(() => assertFamilyUniformity(split)).toThrow(/split the contested family/);
    const uniform = new Map<string, string[]>();
    for (const [type, ids] of CONTESTED_BY_TYPE) for (const id of ids) uniform.set(id, [type]);
    expect(() => assertFamilyUniformity(uniform)).not.toThrow();
    // The precondition the guard used to assume. One item carrying contested
    // spans of two types has ONE annotator answer and two families to attribute
    // it to; the builder used to hand over only the last type, so this case
    // reached the family check looking uniform.
    const twoTypes = new Map(uniform);
    twoTypes.set([...uniform.keys()][0]!, ["neg:sftp-endpoint", "neg:retrieval-reference"]);
    expect(() => assertFamilyUniformity(twoTypes)).toThrow(/carries contested spans of 2 types/);
  });

  it("keeps a one-borderline item scored and disputes the two-borderline one", () => {
    // The rule corpora/fixtures/smoke.gold-tier2.jsonl already ships.
    expect(predicateVerdictFor("inj-o15-1").state).toBe("labelled");
    expect(predicateVerdictFor("inj-hn08-1").state).toBe("disputed");
    expect(predicateVerdictFor("inj-o01-0").state).toBe("uncovered");
    expect(built.manifest.predicateAdjudication.labelled).toBe(19);
    expect(built.manifest.predicateAdjudication.disputed).toBe(1);
    expect(built.manifest.predicateAdjudication.disputedUnderStrictRule).toBe(2);
  });

  it("disputes an item where the two annotators disagree outright", () => {
    // No real item does, so the branch is exercised with a synthetic pair.
    const flipped: AnnotatorLabel = { ...LABELS_BY_ITEM.B.get("inj-o15-1")!, satisfiesPredicate: true };
    const saved = LABELS_BY_ITEM.B.get("inj-o15-1")!;
    (LABELS_BY_ITEM.B as Map<string, AnnotatorLabel>).set("inj-o15-1", flipped);
    try {
      const verdict = predicateVerdictFor("inj-o15-1");
      expect(verdict.state).toBe("disputed");
      expect(verdict.reason).toContain("A answered false and annotator B answered true");
    } finally {
      (LABELS_BY_ITEM.B as Map<string, AnnotatorLabel>).set("inj-o15-1", saved);
    }
    expect(predicateVerdictFor("inj-o15-1").state).toBe("labelled");
  });
});

describe("the tier-2 predicate gold", () => {
  // Parsed from what the BUILDER produced, not from the committed file.
  // MEASURED by mutation: reading the committed bytes made every assertion here
  // insensitive to the code, and a mutant that forced every row's confidence to
  // "clear" was caught only by the byte-comparison test -- which says the file
  // changed, not that the rule is wrong. The committed bytes are checked
  // against built.goldJsonl separately, at the top of this file.
  const rows = loadTier2Gold(built.goldJsonl);
  const smoke = loadTier2Gold(readFileSync(SMOKE_TIER2_GOLD_PATH, "utf8"));

  it("parses through the scorer's own loader and names 20 items in the corpus", () => {
    expect(rows.length).toBe(20);
    const ids = new Set(COMMITTED_V2.map((i) => i.id));
    for (const row of rows) expect(ids.has(row.itemId)).toBe(true);
    expect(rows.map((r) => r.itemId).sort()).toEqual([...LABELLED_ITEM_IDS].sort());
  });

  it("is 19 scored, 1 disputed, and ZERO positives -- the round's headline result", () => {
    expect(rows.filter((r) => r.status === "scored").length).toBe(19);
    expect(rows.filter((r) => r.status === "disputed").length).toBe(1);
    expect(rows.filter((r) => r.satisfies).length).toBe(0);
    for (const row of rows) expect(row.spans).toEqual([]);
    expect(built.manifest.predicateAdjudication.verdict).toContain("NO PREDICATE POSITIVES");
  });

  it("carries the WEAKER confidence, matching the shipped fixture's convention", () => {
    for (const row of rows) {
      const a = LABELS_BY_ITEM.A.get(row.itemId)!;
      const b = LABELS_BY_ITEM.B.get(row.itemId)!;
      const weaker = a.predicateConfidence === "borderline" || b.predicateConfidence === "borderline";
      expect(row.confidence).toBe(weaker ? "borderline" : "clear");
      expect(row.annotators.a.rationale).toBe(a.rationale);
      expect(row.annotators.b.rationale).toBe(b.rationale);
    }
    expect(rows.filter((r) => r.confidence === "borderline").map((r) => r.itemId).sort()).toEqual([
      "inj-hn08-1",
      "inj-o15-1",
    ]);
  });

  it("pools with the smoke fixture without colliding on an itemId or a policy hash", () => {
    const smokeIds = new Set(smoke.map((r) => r.itemId));
    for (const row of rows) expect(smokeIds.has(row.itemId)).toBe(false);
    for (const row of rows) expect(row.policyHash).toBe(smoke[0]!.policyHash);
    expect(rows[0]!.policyHash).toBe(sha256(POLICY_TEXT));
  });
});

describe("the predicate gold this builder can and cannot produce", () => {
  const withLabels = (itemId: string, patch: Partial<AnnotatorLabel>) => {
    const a = new Map(LABELS_BY_ITEM.A);
    const b = new Map(LABELS_BY_ITEM.B);
    a.set(itemId, { ...a.get(itemId)!, ...patch });
    b.set(itemId, { ...b.get(itemId)!, ...patch });
    return { A: a as ReadonlyMap<string, AnnotatorLabel>, B: b as ReadonlyMap<string, AnnotatorLabel> };
  };

  it("refuses an adjudicated positive rather than emitting a row the schema will reject", () => {
    // The round produced zero positives, so this branch is unreachable from the
    // shipped labels. MEASURED before the guard existed: flipping one item to
    // `satisfiesPredicate: true, predicateConfidence: "clear"` produced
    // `{"status":"scored","satisfies":true,"spans":[]}` and the build then died
    // inside loadTier2Gold with "a scored row that satisfies the predicate must
    // name at least one span" -- a message naming the symptom, not the cause.
    const id = LABELLED_ITEM_IDS[0]!;
    const labels = withLabels(id, { satisfiesPredicate: true, predicateConfidence: "clear" });
    expect(() => tier2GoldRows(COMMITTED_V2, "hash", labels)).toThrow(
      /adjudicated as SATISFYING .*this builder cannot emit a span for it/s,
    );
    // And the same labels with the flag left alone still emit, so the refusal
    // is about the positive and not about the injected map.
    expect(tier2GoldRows(COMMITTED_V2, "hash", LABELS_BY_ITEM)).toHaveLength(20);
  });

  it("reports predicateGoldSpansEmitted from the rows, not from the count of positives", () => {
    // The field said `positives` and the builder writes `spans: []` on every
    // row, so a round with a positive would have published a span count it did
    // not emit. It is now passed what the serialized rows carry.
    const rows = tier2GoldRows(COMMITTED_V2, "hash");
    const emitted = rows.reduce((n, r) => n + r.spans.length, 0);
    expect(emitted).toBe(0);
    expect(built.manifest.predicateAdjudication.predicateGoldSpansEmitted).toBe(emitted);
    // And the field is not simply pinned to zero: it reports what it is given.
    expect(predicateAdjudication(COMMITTED_V2.map((i) => i.id), 7).predicateGoldSpansEmitted).toBe(7);
    expect(predicateAdjudication(COMMITTED_V2.map((i) => i.id), 0).positives).toBe(0);
  });

  it("names both annotators on a disputed row instead of presenting A's answer as the verdict", () => {
    // `satisfies` on a disputed row is A's call, and the adjudication string
    // used to render it as "DISPUTED false" on an item where a reader would
    // find B saying true. Swapping A for B in the builder changed nothing in
    // the suite, because A and B agree on all 20 real items.
    const id = LABELLED_ITEM_IDS[0]!;
    const a = new Map(LABELS_BY_ITEM.A);
    const b = new Map(LABELS_BY_ITEM.B);
    b.set(id, { ...b.get(id)!, satisfiesPredicate: !b.get(id)!.satisfiesPredicate });
    const rows = tier2GoldRows(COMMITTED_V2, sha256(POLICY_TEXT), { A: a, B: b });
    const disputed = rows.find((r) => r.itemId === id)!;
    expect(disputed.status).toBe("disputed");
    expect(disputed.adjudication).toContain("NO AGREED ANSWER");
    expect(disputed.adjudication).toContain("annotator A's false");
    expect(disputed.adjudication).toContain("annotator B answered true");
    // The other disputed cause -- both annotators agreeing and both calling it
    // borderline -- reads differently, because there IS an agreed answer there
    // and rendering it as "no agreed answer" would be the mirror of the defect.
    const bothBorderline = tier2GoldRows(COMMITTED_V2, sha256(POLICY_TEXT)).find(
      (r) => r.itemId === "inj-hn08-1",
    )!;
    expect(bothBorderline.status).toBe("disputed");
    expect(bothBorderline.adjudication).toContain("both annotators answered false");
    expect(bothBorderline.adjudication).not.toContain("NO AGREED ANSWER");
    // The row still parses, so the honesty is in the text and not bought by
    // emitting something the scorer cannot read.
    expect(loadTier2Gold(serializeTier2Gold(rows))).toHaveLength(20);
  });
});

describe("the emit gate refuses, once per advertised check", () => {
  const source = built.manifest.source;
  const good = () => verifyLabelledOrRefuse(source, COMMITTED_V2, built.corpusJsonl, built.goldJsonl);
  const withItems = (mutate: (items: CorpusItem[]) => CorpusItem[]): string =>
    serializeCorpus(mutate(labelledItems(COMMITTED_V2).map((i) => ({ ...i, gold: [...i.gold] }))));

  it("passes on the real artifacts, and counts what it checked", () => {
    const report = good();
    expect(report.itemsChecked).toBe(189);
    expect(report.goldSpansChecked).toBe(108);
    expect(report.contestedSpansChecked).toBe(20);
    expect(report.tier2RowsChecked).toBe(20);
    expect(report.checks).toEqual(LABELLED_VERIFICATION_CHECKS);
  });

  it("refuses when an item's text drifts from the source corpus", () => {
    const jsonl = withItems((items) => items.map((i) => (i.id === "inj-o01-4" ? { ...i, text: `${i.text} ` } : i)));
    expect(() => verifyLabelledOrRefuse(source, COMMITTED_V2, jsonl, built.goldJsonl)).toThrow(
      /text differs from the source corpus/,
    );
  });

  it("refuses when the gold array drifts from the source corpus", () => {
    const jsonl = withItems((items) =>
      items.map((i) => (i.gold.length > 0 ? { ...i, gold: i.gold.slice(1) } : i)),
    );
    expect(() => verifyLabelledOrRefuse(source, COMMITTED_V2, jsonl, built.goldJsonl)).toThrow(
      /gold differs from the source corpus/,
    );
  });

  it("refuses a gold span whose offsets no longer hold its text", () => {
    // Mutating BOTH sides so the source-identity check passes first. MEASURED
    // while writing this: the refusal comes from CorpusItemSchema's refine
    // inside loadCorpus, not from the loop in verifyLabelledOrRefuse, because
    // the gate parses the serialized bytes before it walks them. The advertised
    // check now says so rather than claiming a refusal this path never reaches.
    const broken = COMMITTED_V2.map((i) =>
      i.gold.length > 0 ? { ...i, gold: [{ ...i.gold[0]!, start: i.gold[0]!.start + 1 }, ...i.gold.slice(1)] } : i,
    );
    const jsonl = serializeCorpus(labelledItems(broken));
    expect(() => verifyLabelledOrRefuse(source, broken, jsonl, built.goldJsonl)).toThrow(
      /offsets do not hold the text it names/,
    );
  });

  it("refuses a pred: type inside the corpus gold array", () => {
    const withPred = COMMITTED_V2.map((i) =>
      i.id === "inj-o01-4"
        ? {
            ...i,
            gold: [
              ...i.gold,
              { start: 0, end: 3, text: i.text.slice(0, 3), entityType: "pred:client-relationship-disclosure", action: "redact" as const },
            ],
          }
        : i,
    );
    const jsonl = serializeCorpus(labelledItems(withPred));
    expect(() => verifyLabelledOrRefuse(source, withPred, jsonl, built.goldJsonl)).toThrow(/pred: span in gold/);
  });

  it("refuses a tier-2 gold row naming an item the corpus does not have", () => {
    const rows = tier2GoldRows(COMMITTED_V2, sha256(POLICY_TEXT));
    const bad = serializeTier2Gold([{ ...rows[0]!, itemId: "no-such-item" }, ...rows.slice(1)]);
    expect(() => verifyLabelledOrRefuse(source, COMMITTED_V2, built.corpusJsonl, bad)).toThrow(
      /names item "no-such-item"/,
    );
  });

  it("refuses a tier-2 gold span that does not slice back on the item's text", () => {
    const rows = tier2GoldRows(COMMITTED_V2, sha256(POLICY_TEXT));
    const bad = serializeTier2Gold([
      { ...rows[0]!, satisfies: true, spans: [{ start: 0, end: 5, text: "XXXXX" }] },
      ...rows.slice(1),
    ]);
    expect(() => verifyLabelledOrRefuse(source, COMMITTED_V2, built.corpusJsonl, bad)).toThrow(/does not slice back/);
  });

  it("refuses a contested span this round did not adjudicate", () => {
    const unknown = COMMITTED_V2.map((i) => {
      if (i.id !== "inj-o01-4") return i;
      const labels = (i.meta!["labels"] as { span: { start: number; end: number }; type: string }[]).map((l) => {
        const q = (i.meta!["labelQuestions"] as { kind: string; span?: { start: number; end: number } }[]).find(
          (x) => x.kind === "contested-span-label",
        )!;
        return l.span.start === q.span!.start && l.span.end === q.span!.end ? { ...l, type: "neg:unadjudicated" } : l;
      });
      return { ...i, meta: { ...i.meta, labels } };
    });
    const jsonl = serializeCorpus(labelledItems(unknown));
    expect(() => verifyLabelledOrRefuse(source, unknown, jsonl, built.goldJsonl)).toThrow(
      /this round did not adjudicate/,
    );
  });

  it("refuses an emitted item that is not in the source corpus", () => {
    // A branch nothing reached before: MEASURED by mutation, rewording its
    // message changed no test outcome. The item count has to stay equal or the
    // length check fires first, so one id is renamed rather than one removed.
    const renamed = labelledItems(COMMITTED_V2).map((i) => (i.id === "inj-o01-4" ? { ...i, id: "inj-ghost" } : i));
    expect(() => verifyLabelledOrRefuse(source, COMMITTED_V2, serializeCorpus(renamed), built.goldJsonl)).toThrow(
      /labelled item inj-ghost is not in the source corpus/,
    );
  });

  it("refuses to emit when an annotator's answers split a contested family", () => {
    // Retyping one sftp span as a retrieval reference splits B's answers within
    // that family: B affirmed the 7 real retrieval references' proposal as
    // wrong and this one as right. The guard has to be on the EMIT path for
    // this to fire -- it was not, until a mutation showed the builder's call
    // could be deleted with the suite still green.
    const split = COMMITTED_V2.map((i) => {
      if (i.id !== "inj-o01-4") return i;
      const q = (i.meta!["labelQuestions"] as { kind: string; span?: { start: number; end: number } }[]).find(
        (x) => x.kind === "contested-span-label",
      )!;
      const labels = (i.meta!["labels"] as { span: { start: number; end: number }; type: string }[]).map((l) =>
        l.span.start === q.span!.start && l.span.end === q.span!.end
          ? { ...l, type: "neg:retrieval-reference" }
          : l,
      );
      return { ...i, meta: { ...i.meta, labels } };
    });
    expect(() =>
      verifyLabelledOrRefuse(source, split, serializeCorpus(labelledItems(split)), built.goldJsonl),
    ).toThrow(/split the contested family/);
  });

  it("refuses an adjudication of a type no span in the corpus carries", () => {
    const thinned = COMMITTED_V2.filter((i) => !CONTESTED_BY_TYPE.get("neg:retrieval-reference")!.includes(i.id));
    const jsonl = serializeCorpus(labelledItems(thinned));
    const rows = serializeTier2Gold(
      tier2GoldRows(COMMITTED_V2, sha256(POLICY_TEXT)).filter((r) => thinned.some((i) => i.id === r.itemId)),
    );
    expect(() => verifyLabelledOrRefuse(source, thinned, jsonl, rows)).toThrow(
      /An adjudication of nothing is a claim with no subject/,
    );
  });
});

describe("the labelled items say only what the certification supports", () => {
  const emitted = loadCorpus(built.corpusJsonl);

  it("does not claim the injection invariant covers classes no stage that ran can see", () => {
    // The defect: every one of the 189 items said "Every class NOT listed is
    // now covered by the injection invariant" while the v2 manifest's own
    // certification.invariantScope says the invariant is established for
    // TIER-0 entity classes only, and client-name is tier 1. A scorer reading
    // the item would count a tier-1 finding outside gold as a true false
    // positive on the authority of a sentence the certification contradicts.
    const ir = loadPolicyIr(readFileSync(IR_PATH, "utf8"));
    const tier1Classes = ir.entityTypes.filter((e) => e.tier > 0).map((e) => e.id);
    expect(tier1Classes).toContain("client-name");
    for (const item of emitted) {
      const scope = item.meta!["scoringScope"] as { why: string; unlabelledClasses: string[] };
      expect([item.id, scope.why.includes("ONLY AS FAR AS THAT INVARIANT REACHES")]).toEqual([item.id, true]);
      expect([item.id, scope.why.includes("TIER-1")]).toEqual([item.id, true]);
      // The old sentence, which claimed more than the certification does.
      expect([item.id, scope.why.includes("Every class NOT listed is now covered")]).toEqual([item.id, false]);
    }
    // And the two records now agree rather than contradicting inside one commit.
    const v2Scope = buildV2Artifacts().manifest.certification.invariantScope;
    expect(v2Scope).toContain("stage 1 only");
    expect(v2Scope).toContain("uninjected tier-1 name");
  });
});

describe("the guards a mutation survived, exercised", () => {
  // Each of these was replaced with `if (false)` in a pristine copy and left
  // the suite green. Mostly unreachable by construction TODAY -- which is the
  // reason to test them now rather than after a refactor deletes one.
  const contestedItem = COMMITTED_V2.find((i) => contestedSpansOf(i).length > 0)!;

  it("refuses a contested span whose offsets match no label, or two", () => {
    const labels = (contestedItem.meta!["labels"] as Record<string, unknown>[]);
    const contested = contestedSpansOf(contestedItem)[0]!;
    const dup = {
      ...contestedItem,
      meta: { ...contestedItem.meta, labels: [...labels, labels.find((l) => {
        const s = l["span"] as { start: number; end: number };
        return s.start === contested.start && s.end === contested.end;
      })!] },
    };
    expect(() => contestedSpansOf(dup)).toThrow(/matches 2 labels/);
    const none = {
      ...contestedItem,
      meta: { ...contestedItem.meta, labels: labels.filter((l) => {
        const s = l["span"] as { start: number; end: number };
        return !(s.start === contested.start && s.end === contested.end);
      }) },
    };
    expect(() => contestedSpansOf(none)).toThrow(/matches 0 labels/);
    // With a duplicate label present and this guard gone, the span would be
    // typed from whichever copy sorted first -- silently, which is why the
    // count is asserted rather than the mere presence of a match.
    expect(contestedSpansOf(contestedItem)).toHaveLength(1);
  });

  it("refuses a contested-span question carrying no span", () => {
    const questions = contestedItem.meta!["labelQuestions"] as Record<string, unknown>[];
    const stripped = {
      ...contestedItem,
      meta: {
        ...contestedItem.meta,
        labelQuestions: questions.map((q) => (q["kind"] === "contested-span-label" ? { ...q, span: undefined } : q)),
      },
    };
    expect(() => contestedSpansOf(stripped)).toThrow(/carries no span/);
  });

  it("refuses a gold row for an item the corpus does not hold, and for one nobody answered", () => {
    expect(() => tier2GoldRows(COMMITTED_V2, sha256(POLICY_TEXT), LABELS_BY_ITEM, ["no-such-item"])).toThrow(
      /which is not in/,
    );
    const unanswered = COMMITTED_V2.find((i) => !LABELLED_ITEM_IDS.includes(i.id))!;
    expect(() => tier2GoldRows(COMMITTED_V2, sha256(POLICY_TEXT), LABELS_BY_ITEM, [unanswered.id])).toThrow(
      /is in the labelled set but has no verdict/,
    );
  });

  it("refuses when the emitted corpus is a different size from the source", () => {
    const short = serializeCorpus(labelledItems(COMMITTED_V2).slice(1));
    expect(() => verifyLabelledOrRefuse(built.manifest.source, COMMITTED_V2, short, built.goldJsonl)).toThrow(
      /188 labelled items against 189 source items/,
    );
  });

  it("refuses a contested span in the emitted meta whose offsets no longer hold its text", () => {
    // The gate checks the contested span independently of the gold spans,
    // because a contested span lives in meta and never in `gold` -- so the
    // corpus schema's own refine cannot see it. Replacing the check with
    // `if (false)` left the suite green.
    const drifted = labelledItems(COMMITTED_V2).map((i) => {
      const questions = (i.meta!["labelQuestions"] ?? []) as Record<string, unknown>[];
      if (!questions.some((q) => q["kind"] === "contested-span-label")) return i;
      return {
        ...i,
        meta: {
          ...i.meta,
          labelQuestions: questions.map((q) =>
            q["kind"] === "contested-span-label"
              ? { ...q, span: { ...(q["span"] as Record<string, unknown>), text: "not-what-is-there" } }
              : q,
          ),
        },
      };
    });
    expect(() =>
      verifyLabelledOrRefuse(built.manifest.source, COMMITTED_V2, serializeCorpus(drifted), built.goldJsonl),
    ).toThrow(/contested span \[\d+,\d+\) does not slice back/);
  });

  it("refuses when an item's policy drifts from the source corpus", () => {
    // Unreachable today -- `labelledItems` spreads `...item`, so policy is
    // carried by identity -- and a refactor that rebuilt the item field by
    // field would make it reachable with nothing else to catch it.
    const drifted = labelledItems(COMMITTED_V2).map((i, n) => (n === 0 ? { ...i, policy: "p-med" } : i));
    expect(() =>
      verifyLabelledOrRefuse(built.manifest.source, COMMITTED_V2, serializeCorpus(drifted), built.goldJsonl),
    ).toThrow(/policy differs from the source corpus/);
  });
});

describe("what the corpus can support", () => {
  const support = built.manifest.canSupport;

  it("quantises per-type recall by the gold count recomputed from the committed file", () => {
    const counts = new Map<string, number>();
    for (const item of COMMITTED_V2) for (const g of item.gold) counts.set(g.entityType, (counts.get(g.entityType) ?? 0) + 1);
    expect(support.goldSpans).toBe([...counts.values()].reduce((a, b) => a + b, 0));
    for (const c of support.perEntityType) {
      expect(c.goldPositives).toBe(counts.get(c.className));
      expect(c.recallStep).toBeCloseTo(1 / counts.get(c.className)!, 12);
    }
    expect(counts.get("in-pan")).toBe(7);
  });

  it("counts the negative surface, and records the 13 spans this round moved into it", () => {
    const confusables = COMMITTED_V2.flatMap((i) => (i.meta!["labels"] as { type: string }[])).filter((l) =>
      l.type.startsWith("neg:"),
    ).length;
    expect(support.negativeSurface.confusableSpans).toBe(confusables);
    expect(support.negativeSurface.excludedConfusableSpans).toBe(7);
    expect(support.negativeSurface.scoredConfusableSpans).toBe(confusables - 7);
    expect(support.negativeSurface.movedIntoScopeByThisRound).toBe(13);
    expect(support.negativeSurface.pristineNegativeItems).toBe(COMMITTED_V2.filter((i) => i.gold.length === 0).length);
  });

  it("says plainly that predicate recall has no denominator", () => {
    expect(support.predicate.goldPositives).toBe(0);
    expect(support.predicate.recallStep).toBeNull();
    expect(support.predicate.scoredNegatives).toBe(19);
    expect(support.cannotRank.some((s) => s.includes("Not rankable at all"))).toBe(true);
    expect(support.verdict).toContain("CANNOT SUPPORT ONE ON THE PREDICATE");
  });

  it("measures what the compiled arm's own rules can reach, per match rule, with runTier0", () => {
    // The defect: the artifact listed private-key-material under canRank with
    // no caveat while the class is UNSCOREABLE for every compiled arm on two of
    // the scorer's three rules. Gold is the whole PEM block; p-fin's only rule
    // matches the BEGIN line. Recomputed here against the shipping detector
    // rather than read out of the manifest.
    const ir = loadPolicyIr(readFileSync(IR_PATH, "utf8"));
    const reach = compiledArmReachByType(COMMITTED_V2, ir);
    for (const c of support.perEntityType) {
      expect([c.className, c.compiledArmReach]).toEqual([c.className, reach.get(c.className) ?? null]);
    }
    const pem = reach.get("private-key-material")!;
    expect([pem.exact, pem.iou50, pem.overlap, pem.unreachable]).toEqual([0, 0, 12, 6]);
    // Non-vacuity: a class whose rule DOES produce the gold span reads
    // differently, so the zeros above are about this class and not about the
    // measurement returning zero for everything.
    expect(reach.get("db-connection-string")!.exact).toBe(9);
    // And the exact/iou50 gap the account-number convention costs, which the
    // manifest also did not carry.
    const account = reach.get("bank-account-identifier")!;
    expect(account.exact).toBeLessThan(account.iou50);
    // A tier-1 class has no tier-0 rule, so the field is null rather than a
    // zero that would read as "the arm cannot detect it".
    expect(reach.get("client-name")).toBeUndefined();
    expect(support.perEntityType.find((c) => c.className === "client-name")!.compiledArmReach).toBeNull();
  });

  it("caveats the rankability claim with the match rules the class cannot express", () => {
    const pemRow = support.canRank.find((r) => r.startsWith("private-key-material"))!;
    expect(pemRow).toContain("UNDER overlap ONLY");
    expect(support.cannotRank).toContain(
      support.cannotRank.find((r) => r.startsWith("private-key-material under exact")),
    );
    expect(support.cannotRank.some((r) => r.startsWith("private-key-material under iou50"))).toBe(true);
    // And a class with no blocked rule carries no caveat, so the suffix is
    // derived from the measurement and not appended to everything.
    expect(support.canRank.find((r) => r.startsWith("api-credential"))).not.toContain("UNDER");
  });

  it("computes the zero-event 95% upper bound as the exact Clopper-Pearson limit", () => {
    // 1 - 0.05^(1/n), computed independently: n=19 -> 0.1458685033, n=30 ->
    // 0.0950338529. n=1 is the sanity anchor -- one trial with no event bounds
    // the rate at 0.95, which is the rule the whole formula generalises.
    expect(zeroEventUpperBound95(19)).toBeCloseTo(0.1458685033, 9);
    expect(zeroEventUpperBound95(30)).toBeCloseTo(0.0950338529, 9);
    expect(zeroEventUpperBound95(0)).toBeNull();
    expect(zeroEventUpperBound95(1)).toBeCloseTo(0.95, 12);
  });
});

describe("every annotator label attaches to something real", () => {
  it("names an item in the corpus and a span the corpus carries at those offsets", () => {
    for (const annotator of ANNOTATORS) {
      expect(ANNOTATOR_LABELS[annotator].length).toBe(20);
      for (const label of ANNOTATOR_LABELS[annotator]) {
        const item = COMMITTED_V2.find((i) => i.id === label.itemId);
        expect(item, `annotator ${annotator} labelled ${label.itemId}`).toBeDefined();
        const spans = contestedSpansOf(item!);
        expect(spans.length).toBe(1);
        expect(item!.text.slice(spans[0]!.start, spans[0]!.end)).toBe(spans[0]!.text);
      }
    }
    expect(LABELLED_ITEM_IDS.length).toBe(20);
  });

  it("refuses a round where the two annotators labelled different item sets", () => {
    // The docblock said "Empty if the two sets ever diverge" and the code is an
    // intersection, so divergence SHRANK the set instead: a future round where
    // B skipped one item would have dropped it from the gold file, from the
    // predicate counts and from coverage, quietly. Mutating the intersection
    // filter to `filter(() => true)` survived the whole suite.
    const ids = [...LABELLED_ITEM_IDS];
    expect(assertBothAnnotatorsLabelledTheSameItems(ids, ids)).toEqual([...ids].sort());
    expect(() => assertBothAnnotatorsLabelledTheSameItems(ids, ids.slice(1))).toThrow(
      /labelled different item sets: 1 only A/,
    );
    expect(() => assertBothAnnotatorsLabelledTheSameItems(ids, [...ids, "extra"])).toThrow(
      /1 only B \(extra\)/,
    );
  });

  it("connects the 0 span-override rate to the channel it came through", () => {
    // Both annotators recovered the generator's proposal by reading `gold` at
    // the contested offsets, and for a contested span ABSENCE from gold IS the
    // proposal -- so that read handed them the answer, not merely the context.
    // The manifest recorded the read and the 0 rate in separate sections.
    const contested = COMMITTED_V2.flatMap((i) => contestedSpansOf(i));
    expect(contested).toHaveLength(20);
    const inGold = contested.filter((c) =>
      COMMITTED_V2.some((i) => i.gold.some((g) => g.start === c.start && g.end === c.end)),
    );
    expect(inGold).toEqual([]);
    const note = overrideReport().note;
    expect(note).toContain("ABSENCE from gold IS the proposal");
    expect(note).toContain("cannot be read as independent corroboration");
    // And the note names what keeps the round from being a rubber stamp, which
    // is B's ordering plus B's 7 overrides -- both facts, recomputed here.
    const b = overrideReport().perAnnotator.find((r) => r.annotator === "B")!;
    expect([b.n, b.spanOverrides]).toEqual([20, 7]);
    const a = overrideReport().perAnnotator.find((r) => r.annotator === "A")!;
    expect([a.n, a.spanOverrides]).toEqual([20, 0]);
  });

  it("carries a correctedType only where the annotator overrode the proposal", () => {
    for (const annotator of ANNOTATORS) {
      for (const label of ANNOTATOR_LABELS[annotator]) {
        if (label.spanLabelCorrect) expect(label.correctedType).toBeUndefined();
      }
    }
  });
});
