import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr, runTier0, seededRng, segmentText, type PolicyIr } from "@sih/core";
import { loadCorpus } from "../src/driver/corpus.js";
import { IR_PATH, REPO_ROOT, SELFTEST_PATH, loadSelfTestExamples } from "../src/corpus/build.js";
import {
  CONTESTED_TYPES,
  V2_CORPUS_PATH,
  V2_MANIFEST_PATH,
  V2_QUEUE_PATH,
  buildV2Artifacts,
  v2DistractorFor,
} from "../src/corpus/build-v2.js";
import { CONFUSABLE_FAMILIES, POSITIVE_FAMILIES } from "../src/corpus/families.js";
import { CANDIDATE_CONFUSABLE_FAMILIES, CANDIDATE_POSITIVE_FAMILIES } from "../src/corpus/families.candidate.js";
import {
  CROSS_SEGMENT_CLIENT_CLAUSE,
  CROSS_SEGMENT_JOIN,
  CROSS_SEGMENT_PREFIX,
  CROSS_SEGMENT_VENDOR_CLAUSE,
  ORG_FAMILY_IDS,
  PAIRED_TYPE,
  REMOVED_FAMILIES,
  RENAMED_FAMILIES,
  V2_CONFUSABLE_FAMILIES,
  V2_FAMILIES,
  V2_POSITIVE_FAMILIES,
  orgRoleClassOf,
  type V2Family,
} from "../src/corpus/families.v2.js";
import { formatSpecSweep } from "../src/corpus/format-sweep.js";
import {
  BOOST_WINDOW,
  IR_COUNTEREXAMPLE_SURFACES,
  boostNear,
  boostTerms,
  irCounterExamples,
  measureBoost,
  measureOrthography,
  measureRoles,
  orthographicOracle,
} from "../src/corpus/leakage.js";
import { ORG_POOL, roleClass } from "../src/corpus/orgs.js";
import { PREDICATE_ID } from "../src/corpus/questions.js";

/**
 * The corpus this round exists to produce, and the four measurements that say
 * it does not hand the compiled arm the answer.
 *
 * Every "before" number asserted here is recomputed from
 * `corpora/generated/injection-p-fin-adjudicated-v1.jsonl` by the SAME function
 * that produces the after number, rather than quoted from a review. A
 * before/after pair computed by two different definitions is not a comparison.
 */

const IR_TEXT = readFileSync(IR_PATH, "utf8");
const IR: PolicyIr = loadPolicyIr(IR_TEXT);
const SELFTEST_TEXT = readFileSync(SELFTEST_PATH, "utf8");
const SELFTEST_EXAMPLE_TEXT = loadSelfTestExamples().map((e) => e.text).join("\n");

const built = buildV2Artifacts();
const items = built.items;
const manifest = built.manifest;

const BEFORE = loadCorpus(
  readFileSync(join(REPO_ROOT, "corpora/generated/injection-p-fin-adjudicated-v1.jsonl"), "utf8"),
);

/** Renders a family's own glue around one minted value. The declaration test's input. */
function renderGlue(family: V2Family, seed: string): { text: string; span: { start: number; end: number } } {
  const value = family.mint(seededRng(seed));
  const { prefix, suffix } = family.glue(value);
  return { text: `${prefix}${value}${suffix}`, span: { start: prefix.length, end: prefix.length + value.length } };
}

describe("the committed artifacts are what this code produces", () => {
  it("reproduces the corpus, the manifest and the labelling queue byte for byte", () => {
    expect(readFileSync(V2_CORPUS_PATH, "utf8")).toBe(built.corpusJsonl);
    expect(readFileSync(V2_MANIFEST_PATH, "utf8")).toBe(built.manifestJson);
    expect(readFileSync(V2_QUEUE_PATH, "utf8")).toBe(built.queueJsonl);
  });

  it("is deterministic in the seed", () => {
    expect(buildV2Artifacts().corpusJsonl).toBe(built.corpusJsonl);
    expect(buildV2Artifacts("a-different-seed").corpusJsonl).not.toBe(built.corpusJsonl);
  });
});

describe("decision 1: surfaces are minted from format specs, not read off the IR", () => {
  it("annotates exactly the counterExamples the IR actually holds", () => {
    // Without this the surface-name check silently stops covering a
    // counterExample the next compile adds.
    expect(IR_COUNTEREXAMPLE_SURFACES.map((c) => c.counterExample).sort()).toEqual([...irCounterExamples(IR)].sort());
  });

  it("shares no confusable surface with an IR counterExample, where the previous corpus shared 13", () => {
    const irSurfaces = new Set(IR_COUNTEREXAMPLE_SURFACES.map((c) => c.surfaceName));
    expect(V2_CONFUSABLE_FAMILIES.filter((f) => irSurfaces.has(f.surfaceName)).map((f) => f.id)).toEqual([]);
    // The other half of the claim, as a fact about the corpus this replaces:
    // 13 of its 24 confusable families were 1:1 with an entry in that list.
    // Named here rather than in prose so the comparison is checkable.
    const wave12Confusables = [...CONFUSABLE_FAMILIES, ...CANDIDATE_CONFUSABLE_FAMILIES].map((f) => f.id);
    const oneToOne = [
      "pan-shaped-invalid",
      "aadhaar-shaped-invalid",
      "email-address",
      "swift-bic",
      "micr-code",
      "ticket-id",
      "internal-http-url",
      "gstin",
      "employee-id",
      "redacted-credential-placeholder",
      "tutorial-api-key",
      "csr-pem-block",
      "public-key-fingerprint",
    ];
    expect(oneToOne.filter((id) => wave12Confusables.includes(id))).toHaveLength(13);
    expect(wave12Confusables).toHaveLength(24);
  });

  it("puts no minted value into the IR or the compiler self-test, over 120 seeds a family", () => {
    const leaks: string[] = [];
    for (const family of V2_FAMILIES) {
      for (let i = 0; i < 120; i += 1) {
        const value = family.mint(seededRng(`v2-leak|${family.id}|${i}`));
        if (IR_TEXT.includes(value)) leaks.push(`${family.id} -> IR`);
        if (SELFTEST_TEXT.includes(value)) leaks.push(`${family.id} -> selftest`);
      }
    }
    expect(leaks).toEqual([]);
    expect([...ORG_POOL].filter((n) => IR_TEXT.includes(n) || SELFTEST_TEXT.includes(n))).toEqual([]);
  });

  it("and the haystacks are the real ones, so the check above is not vacuous", () => {
    expect(IR_TEXT).toContain("AAAPZ1234C");
    expect(SELFTEST_TEXT).toContain("AKIAIOSFODNN7EXAMPLE");
    expect(SELFTEST_EXAMPLE_TEXT).toContain("AKIAIOSFODNN7EXAMPLE");
  });

  it("emits no span value that appears in either haystack", () => {
    expect(manifest.leakage.irOverlap.spanValuesFoundInIr).toEqual([]);
    expect(manifest.leakage.irOverlap.spanValuesFoundInSelfTest).toEqual([]);
    expect(manifest.leakage.irOverlap.labelledSpans).toBeGreaterThan(300);
  });

  it("accounts for every wave-1 and wave-2 family, as kept, renamed or removed", () => {
    const removed = new Set(REMOVED_FAMILIES.flatMap((r) => r.id.split(" / ")));
    for (const r of RENAMED_FAMILIES) removed.add(r.from);
    const kept = new Set(V2_FAMILIES.map((f) => f.id));
    // A rename must actually land on a family that exists, or the register
    // sends a reader looking for something that is not there.
    for (const r of RENAMED_FAMILIES) expect([r.from, kept.has(r.to)]).toEqual([r.from, true]);
    // And a family cannot be both renamed and removed.
    for (const r of RENAMED_FAMILIES) {
      expect([r.from, REMOVED_FAMILIES.some((x) => x.id === r.from)]).toEqual([r.from, false]);
    }
    const orphaned = [...POSITIVE_FAMILIES, ...CONFUSABLE_FAMILIES, ...CANDIDATE_POSITIVE_FAMILIES, ...CANDIDATE_CONFUSABLE_FAMILIES]
      .map((f) => f.id)
      .filter((id) => !kept.has(id) && !removed.has(id));
    expect(orphaned).toEqual([]);
    for (const r of REMOVED_FAMILIES) expect([r.id, r.why.length > 40]).toEqual([r.id, true]);
    for (const r of RENAMED_FAMILIES) expect([r.from, r.why.length > 40]).toEqual([r.from, true]);
  });
});

describe("decision 2: a contextBoost term is as likely near a confusable as near a positive", () => {
  const terms = boostTerms(IR);

  it("declares boost per family and the declaration matches the family's own glue", () => {
    // Measured on the GLUE, not on the finished item: a neighbouring injection
    // can put a boost term inside the window and that is not a fact about this
    // family's declaration. Two seeds, because a mint that happened to produce
    // a boost-looking value would otherwise pass on one.
    const wrong: string[] = [];
    for (const family of V2_FAMILIES) {
      for (const seed of ["boost-a", "boost-b"]) {
        const { text, span } = renderGlue(family, `${seed}|${family.id}`);
        const measured = boostNear(text, span, terms.all) ? "carries" : "absent";
        if (measured !== family.boost) wrong.push(`${family.id} declares ${family.boost}, glue is ${measured}`);
      }
    }
    expect(wrong).toEqual([]);
  });

  it("balances the two rates on the emitted corpus, where the previous one did not", () => {
    const after = measureBoost(items, IR, PAIRED_TYPE);
    const before = measureBoost(BEFORE, IR, WAVE2_PAIRED_TYPE);
    // The defect, restated as a measurement: before, a boost term sat near
    // 40.7% of gold spans and near 0.9% of confusable spans.
    expect(before.goldRate).toBeGreaterThan(0.35);
    expect(before.confusableRate).toBeLessThan(0.05);
    expect(before.delta).toBeGreaterThan(0.3);
    // After. 0.08 is the bound this corpus is claimed to meet, not a bound
    // anything was tuned to: the catalogue declares `boost` per family and the
    // rates fall out of the deal.
    expect(Math.abs(after.delta)).toBeLessThan(0.08);
    expect(Math.abs(after.ownTypeDelta)).toBeLessThan(0.08);
    // Non-vacuous on both sides: a corpus with no boost terms at all would also
    // have a delta of zero and would be a different kind of unrealistic.
    expect(after.goldRate).toBeGreaterThan(0.25);
    expect(after.confusableRate).toBeGreaterThan(0.25);
  });

  it("carries the measurement into the manifest rather than only into this test", () => {
    expect(manifest.leakage.boost.window).toBe(BOOST_WINDOW);
    expect(Math.abs(manifest.leakage.boost.delta)).toBeLessThan(0.08);
  });
});

/** Wave 2's `neg:` ids mapped to the entityType each was a near miss for, so the before/after own-type rates are comparable. */
const WAVE2_PAIRED_TYPE: Readonly<Record<string, string>> = {
  "neg:pan-shaped-invalid-holder": "in-pan",
  "neg:gstin": "in-pan",
  "neg:aadhaar-shaped-bad-verhoeff": "in-aadhaar",
  "neg:email-address": "bank-account-identifier",
  "neg:swift-bic": "bank-account-identifier",
  "neg:micr-code": "bank-account-identifier",
  "neg:ticket-id": "internal-customer-id",
  "neg:employee-id": "internal-customer-id",
  "neg:git-commit-sha": "api-credential",
  "neg:uuid": "api-credential",
  "neg:base64-image-fragment": "api-credential",
  "neg:redacted-credential-placeholder": "api-credential",
  "neg:tutorial-api-key": "api-credential",
  "neg:internal-http-url": "db-connection-string",
  "neg:csr-pem-block": "private-key-material",
  "neg:public-key-fingerprint": "private-key-material",
  "neg:non-client-org": "client-name",
  "neg:dual-role-org-vendor": "client-name",
  "neg:dual-role-org-vendor-message-scope": "client-name",
  "neg:person-name": "client-name",
  "neg:product-name": "client-name",
  "neg:competitor-org": "client-name",
  "neg:listed-company-in-news": "client-name",
  "neg:own-employer-org": "client-name",
};

describe("decision 3: the gold span is not the unique orthographic outlier", () => {
  it("gives every positive family a same-shape confusable to be shadowed by", () => {
    const missing: string[] = [];
    for (const family of V2_POSITIVE_FAMILIES) {
      if (family.selfDistracting === true) {
        // These write their own shape neighbour into the same clause.
        expect([family.id, family.companion !== undefined]).toEqual([family.id, true]);
        continue;
      }
      const drawn = new Set<string>();
      for (let i = 0; i < 40; i += 1) {
        const d = v2DistractorFor(family, seededRng(`d|${family.id}|${i}`));
        if (d === undefined) missing.push(family.id);
        else drawn.add(d.id);
      }
      for (const id of drawn) {
        const conf = V2_CONFUSABLE_FAMILIES.find((f) => f.id === id)!;
        expect([family.id, id, conf.shapeClass]).toEqual([family.id, id, family.shapeClass]);
      }
    }
    expect(missing).toEqual([]);
  });

  it("throws rather than silently skipping a positive whose class has no confusable", () => {
    // Every class in the shipped catalogue has one, which is what the previous
    // test asserts -- so the failure path needs a family with a class nothing
    // declares. A silent `undefined` here would restore the defect for one
    // family with nothing in the manifest to say so.
    const real = V2_POSITIVE_FAMILIES[0]!;
    const orphan = { ...real, id: "orphan", shapeClass: "no-such-shape" as unknown as typeof real.shapeClass };
    expect(() => v2DistractorFor(orphan, seededRng("x"))).toThrowError(/no confusable family shares it/);
    // And the real one still resolves, so the throw is about the class and not
    // about the synthetic family.
    expect(v2DistractorFor(real, seededRng("x"))).toBeDefined();
  });

  it("drops the oracle's solved rate from 58 of 108 to zero", () => {
    const before = measureOrthography(BEFORE);
    const after = measureOrthography(items);
    expect([before.goldSpans, before.goldSpansFound, before.goldSpansSolvedByOracle]).toEqual([108, 95, 58]);
    expect(after.goldSpansSolvedByOracle).toBe(0);
    // Non-vacuity, and it is the load-bearing half: an oracle that found
    // nothing would also solve nothing. It must still find most of the gold.
    expect(after.oracleRecall).toBeGreaterThan(0.8);
    expect(after.oraclePrecision).toBeLessThan(before.oraclePrecision);
  });

  it("puts at least one other odd region in every message that carries a gold span", () => {
    const bare: string[] = [];
    for (const item of items) {
      if (item.gold.length === 0) continue;
      const hits = orthographicOracle(item.text);
      for (const g of item.gold) {
        if (!hits.some((h) => h.start < g.end && g.start < h.end)) continue;
        if (hits.every((h) => h.start < g.end && g.start < h.end)) bare.push(`${item.id}:${g.entityType}`);
      }
    }
    expect(bare).toEqual([]);
  });
});

describe("decision 4: the organisation's role is not readable off its name", () => {
  it("draws every organisation family from the one pool", () => {
    for (const id of ORG_FAMILY_IDS) {
      const family = V2_FAMILIES.find((f) => f.id === id)!;
      const drawn = new Set(Array.from({ length: 200 }, (_, i) => family.mint(seededRng(`pool|${id}|${i}`))));
      expect([id, [...drawn].sort()]).toEqual([id, [...ORG_POOL].sort()]);
    }
    // And both sides of the line are actually written, or the pool sharing
    // would be a property nothing exercises.
    const classes = new Set(ORG_FAMILY_IDS.map((id) => roleClass(V2_FAMILIES.find((f) => f.id === id)!.orgRole!)));
    expect([...classes].sort()).toEqual(["client-side", "non-client"]);
  });

  it("leaves no name locked to one role class, where the previous corpus locked 19 of 39", () => {
    const before = measureRoles(BEFORE, (d) => {
      const r = d["constructedRole"];
      if (r === undefined || r === "none") return undefined;
      return r === "client" || r === "counterparty" ? "client-side" : "non-client";
    });
    expect([before.orgSpans, before.roleLockedSpans]).toEqual([39, 19]);
    const after = measureRoles(items, orgRoleClassOf);
    expect(after.roleLockedSpans).toBe(0);
    expect([...after.namesInBothClasses].sort()).toEqual([...ORG_POOL].sort());
    expect(after.orgSpans).toBeGreaterThan(40);
  });
});

describe("decision 5: message scope is observable, because no segment holds both halves", () => {
  const clientFamily = V2_POSITIVE_FAMILIES.find((f) => f.id === "client-org-cross-segment")!;
  const vendorFamily = V2_CONFUSABLE_FAMILIES.find((f) => f.id === "org-vendor-cross-segment")!;

  it("differs between the two halves in the final clause and in nothing else", () => {
    // Wave 2's pair failed here: it wrote `fee_basis: retainer` on the client
    // side and `rate_card: annual` on the vendor side, so the fenced block alone
    // answered the question and the pair measured nothing about scope.
    const name = ORG_POOL[0];
    const c = clientFamily.glue(name);
    const v = vendorFamily.glue(name);
    expect([c.prefix, c.suffix]).toEqual([CROSS_SEGMENT_PREFIX, CROSS_SEGMENT_JOIN]);
    expect([v.prefix, v.suffix]).toEqual([CROSS_SEGMENT_PREFIX, CROSS_SEGMENT_JOIN]);
    const ct = clientFamily.companion!(name);
    const vt = vendorFamily.companion!(name);
    expect(ct.value).toBe(vt.value);
    const shared = ct.suffix.slice(0, ct.suffix.length - CROSS_SEGMENT_CLIENT_CLAUSE.length);
    expect(vt.suffix).toBe(`${shared}${CROSS_SEGMENT_VENDOR_CLAUSE}`);
    expect(shared).toContain("```");
    expect(CROSS_SEGMENT_CLIENT_CLAUSE).not.toBe(CROSS_SEGMENT_VENDOR_CLAUSE);
  });

  it("names no organisation in the clause that assigns the roles", () => {
    for (const clause of [CROSS_SEGMENT_CLIENT_CLAUSE, CROSS_SEGMENT_VENDOR_CLAUSE]) {
      expect([clause, [...ORG_POOL].filter((n) => clause.includes(n))]).toEqual([clause, []]);
      // And not merely no name from THIS pool: any capitalised multi-word name
      // in the role clause would let a segment-scoped reader answer, which is
      // the whole thing this pair exists to make impossible. MEASURED by
      // mutation: naming an organisation in the clause is caught by the
      // injection invariant only when the name is one the corpus also injects.
      expect([clause, clause.match(/\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b/g) ?? []]).toEqual([clause, []]);
    }
  });

  it("puts the names and the role clause in different segments in every emitted item", () => {
    const roleMarkers = ["is the one we act for", "prints our statements"];
    let seen = 0;
    for (const item of items) {
      const injections = item.meta!["injections"] as { family: string; start: number; end: number; text: string }[];
      const cross = injections.filter((i) => i.family.includes("cross-segment"));
      if (cross.length === 0) continue;
      seen += 1;
      expect([item.id, cross.length]).toEqual([item.id, 2]);
      const segments = segmentText(item.text);
      const nameSegments = segments.filter((s) => cross.some((c) => s.start <= c.start && c.end <= s.end));
      const roleSegments = segments.filter((s) => roleMarkers.some((m) => s.text.includes(m)));
      expect([item.id, nameSegments.length, roleSegments.length]).toEqual([item.id, 1, 1]);
      expect([item.id, nameSegments[0] === roleSegments[0]]).toEqual([item.id, false]);
      // The clause segment must name nobody, and the name segment must assign
      // no role. Either one alone is a "no" for a message-scoped predicate; only
      // the two together answer it.
      expect([item.id, [...ORG_POOL].filter((n) => roleSegments[0]!.text.includes(n))]).toEqual([item.id, []]);
      expect([item.id, roleMarkers.filter((m) => nameSegments[0]!.text.includes(m))]).toEqual([item.id, []]);
      expect([item.id, segments.some((s) => s.kind === "code")]).toEqual([item.id, true]);
    }
    expect(seen).toBeGreaterThan(4);
  });
});

describe("decision 6: the predicate is a question the next round answers, on every item", () => {
  it("asks it of positives and negatives alike", () => {
    for (const item of items) {
      const questions = item.meta!["labelQuestions"] as { id: string; kind: string; about: string }[];
      expect([item.id, questions[0]!.kind, questions[0]!.about]).toEqual([item.id, "message-predicate", PREDICATE_ID]);
      expect([item.id, questions[0]!.id]).toEqual([item.id, `${item.id}#0`]);
    }
    expect(manifest.labelling.itemsWithAPredicateQuestion).toBe(items.length);
    // Both answers have to be reachable, or the round produces gold with one
    // class in it. The counts are what the generator CONSTRUCTED, not labels.
    expect(manifest.labelling.predicateConstructedTrue).toBeGreaterThan(15);
    expect(manifest.labelling.predicateConstructedFalse).toBeGreaterThan(100);
  });

  it("emits no pred: gold and says so on every item", () => {
    for (const item of items) {
      for (const g of item.gold) expect([item.id, g.entityType.startsWith("pred:")]).toEqual([item.id, false]);
      const scope = item.meta!["scoringScope"] as { goldIsComplete: boolean; unlabelledClasses: string[] };
      expect([item.id, scope.goldIsComplete]).toEqual([item.id, false]);
      expect([item.id, scope.unlabelledClasses.includes(PREDICATE_ID)]).toEqual([item.id, true]);
      for (const t of CONTESTED_TYPES) expect([item.id, t, scope.unlabelledClasses.includes(t)]).toEqual([item.id, t, true]);
    }
  });

  it("records the construction as a prediction with its provenance, never as a label", () => {
    const clientItems = items.filter((i) => i.gold.some((g) => g.entityType === "client-name"));
    expect(clientItems.length).toBeGreaterThan(15);
    for (const item of clientItems) {
      const pc = item.meta!["predicateConstruction"] as { constructed: boolean; state: string; basis: string };
      expect([item.id, pc.constructed, pc.state]).toEqual([item.id, true, "unadjudicated"]);
      expect(pc.basis.length).toBeGreaterThan(60);
    }
    // A vendor-only item constructs the opposite, so the field is derived and
    // not a constant.
    const vendorOnly = items.filter(
      (i) =>
        i.gold.length === 0 &&
        (i.meta!["injections"] as { dimensions: Record<string, string> }[]).some(
          (x) => x.dimensions["constructedRole"] === "vendor",
        ),
    );
    expect(vendorOnly.length).toBeGreaterThan(0);
    for (const item of vendorOnly) {
      expect([item.id, (item.meta!["predicateConstruction"] as { constructed: boolean }).constructed]).toEqual([item.id, false]);
    }
  });

  it("hands the annotator a queue with nothing of the generator's intent in it", () => {
    const rows = built.queueJsonl.trim().split("\n").map((l) => JSON.parse(l) as Record<string, unknown>);
    expect(rows.length).toBe(manifest.labelling.questions);
    const allowed = new Set(["questionId", "itemId", "text", "kind", "question", "options", "span"]);
    for (const row of rows) for (const k of Object.keys(row)) expect([k, allowed.has(k)]).toEqual([k, true]);
    for (const forbidden of ["family", "constructedRole", "predicateConstruction", "entityType", "labelBasis"]) {
      expect([forbidden, built.queueJsonl.includes(`"${forbidden}"`)]).toEqual([forbidden, false]);
    }
    for (const family of V2_FAMILIES) {
      expect([family.id, built.queueJsonl.includes(`"${family.id}"`)]).toEqual([family.id, false]);
    }
    // Every question on every item is in the queue, and every queue row names a
    // real item: a queue that silently dropped rows would look blind and be
    // incomplete.
    const byItem = new Map(items.map((i) => [i.id, i]));
    for (const row of rows) expect([row["questionId"], byItem.has(row["itemId"] as string)]).toEqual([row["questionId"], true]);
    const expected = items.flatMap((i) => (i.meta!["labelQuestions"] as { id: string }[]).map((q) => q.id));
    expect(rows.map((r) => r["questionId"])).toEqual(expected);
  });

  it("queues a question for every span whose label a careful reader could reach either way", () => {
    expect(CONTESTED_TYPES.length).toBeGreaterThan(0);
    for (const family of V2_CONFUSABLE_FAMILIES) {
      const contested = CONTESTED_TYPES.includes(family.type);
      expect([family.id, family.labelBasis.contestedBy !== undefined]).toEqual([family.id, contested]);
    }
    const contestedSpans = items.flatMap((i) =>
      (i.meta!["injections"] as { type: string }[]).filter((x) => CONTESTED_TYPES.includes(x.type)),
    );
    expect(manifest.labelling.contestedSpanQuestions).toBe(contestedSpans.length);
  });

  it("cites a policy clause on every family, contested or not", () => {
    for (const family of V2_FAMILIES) {
      expect([family.id, family.labelBasis.clauses.length > 0]).toEqual([family.id, true]);
      for (const c of family.labelBasis.clauses) expect([family.id, /^§\d/.test(c)]).toEqual([family.id, true]);
      expect([family.id, family.labelBasis.reading.length > 30]).toEqual([family.id, true]);
    }
  });
});

describe("the injection invariant still holds, and stage 1's circularity is now measured", () => {
  it("puts no tier-0 finding on carrier text", () => {
    const orphan: string[] = [];
    for (const item of items) {
      const injections = item.meta!["injections"] as { start: number; end: number }[];
      for (const f of runTier0(IR, item.text, segmentText(item.text))) {
        if (!injections.some((i) => f.start < i.end && i.start < f.end)) orphan.push(`${item.id}:${f.entityType}`);
      }
    }
    expect(orphan).toEqual([]);
  });

  it("holds every gold span's text at its offsets", () => {
    for (const item of items) {
      for (const g of item.gold) expect([item.id, item.text.slice(g.start, g.end)]).toEqual([item.id, g.text]);
    }
    expect(manifest.verification.goldSpansChecked).toBe(manifest.counts.goldSpans);
  });

  it("runs an IR-free sweep beside stage 1 and reports what stage 1 decided alone", () => {
    const c = manifest.certification.circularity;
    expect(c).toBeDefined();
    expect(c!.independentSweep).toBe("format-spec-sweep");
    // The number that matters: stage 1 is the tier-0 arm, and this is the set
    // of carriers it removed that an IR-free reading of the same format specs
    // would have kept. Empty means the arm made no quarantine call of its own.
    expect(c!.stage1Only).toEqual([]);
    expect(c!.both.length).toBeGreaterThan(4);
    expect(manifest.certification.supplementarySweepsRun).toContain("format-spec-sweep");
  });

  it("has an independent sweep that actually fires, and fires on things tier 0 misses", () => {
    // Otherwise "stage1Only is empty" would be a statement about a sweep that
    // returns nothing.
    expect(formatSpecSweep("the reference on the form reads ABCDE1234F next to the signature.", "t").map((h) => h.label)).toEqual([
      "pan-shaped",
    ]);
    // A structurally invalid PAN: p-fin's own rule rejects it through the
    // validator, and a certification sweep should not care.
    expect(runTier0(IR, "the reference on the form reads ABCDE1234F.", segmentText("the reference on the form reads ABCDE1234F."))).toEqual([]);
    // And a shape the IR knows nothing about.
    expect(formatSpecSweep("the deduction account number is MUMB12345C.", "t").map((h) => h.label)).toEqual(["tan-shaped"]);
    expect(formatSpecSweep("nothing here but ordinary words about a meeting.", "t")).toEqual([]);
  });

  it("is not certified, and says so", () => {
    expect(manifest.certification.claim).toBe("NOT CERTIFIED");
    expect(manifest.certification.carriers.certifiedClear).toBe(0);
    expect(manifest.certification.stagesUnrun.map((s) => s.stage)).toEqual([
      "high-recall-model-sweep",
      "frontier-adjudication",
    ]);
  });

  it("names what it supersedes and why", () => {
    expect(manifest.supersedes.corpora).toContain("corpora/generated/injection-p-fin-adjudicated-v1.jsonl");
    expect(manifest.supersedes.why.length).toBeGreaterThan(100);
    expect(manifest.unvalidated.length).toBeGreaterThan(4);
  });
});
