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
  BEFORE_CORPUS_TIER0_BOOST,
  BOOST_WINDOW,
  IR_COUNTEREXAMPLE_SURFACES,
  TIER0_BOOST_WINDOW,
  boostNear,
  boostTerms,
  irCounterExamples,
  measureBoost,
  measureOrthography,
  measurePosition,
  measureRoles,
  orthographicOracle,
  tier0BoostNear,
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

  it("pins both surfaceName columns, because the disjointness check is a string equality", () => {
    // `confusableFamiliesSharingAnIrCounterExampleSurface` compares two
    // hand-typed English phrases. Nothing else reads either column, so
    // MEASURED by mutation: changing "MICR line" to "MICR line MUTATED" left
    // all 604 tests green and the manifest bytes unmoved. These two frozen
    // lists are what makes such an edit visible.
    //
    // WHAT THIS DOES NOT CHECK, stated so it is not mistaken for more: it
    // catches a CHANGE to a name, not a family that is an IR counterExample
    // surface under a DIFFERENT name. That gap is real and open -- the
    // disjointness guard is only as good as whoever types the next
    // surfaceName, and there is no mechanical test for "these two English
    // phrases denote the same surface".
    expect([...IR_COUNTEREXAMPLE_SURFACES].map((c) => c.surfaceName).sort()).toEqual([
      "GSTIN",
      "MICR line",
      "PAN-shaped string with an invalid holder-type character",
      "SWIFT/BIC code",
      "bare branch code",
      "certificate signing request block",
      "email address",
      "employee id",
      "incident ticket id",
      "internal https URL",
      "name of a tax form",
      "place name",
      "prose description of a key format",
      "prose statement about a database",
      "prose statement that a credential was rotated",
      "public key fingerprint",
      "redaction placeholder in credential shape",
      "ten-digit number",
      "truncated PAN",
      "twelve-digit number with a broken Verhoeff digit",
      "twelve-digit number with a forbidden leading digit",
      "unnamed reference to a client",
      "unnamed reference to a counterparty",
    ]);
    expect(V2_CONFUSABLE_FAMILIES.map((f) => [f.id, f.surfaceName]).sort()).toEqual([
      ["artifact-digest", "container image digest"],
      ["base64-image-fragment", "fragment of a base64-encoded image"],
      ["datasource-alias", "datasource alias"],
      ["deployment-name", "kubernetes deployment name"],
      ["dh-parameters-block", "Diffie-Hellman parameters block"],
      ["git-commit-sha", "git commit id"],
      ["imps-rrn", "NPCI retrieval reference number"],
      ["kyc-standard-version", "version number of an internal policy document"],
      ["masked-mobile", "masked telephone number"],
      ["org-competitor", "named organisation in a competitor clause"],
      ["org-landlord", "named organisation in a lease clause"],
      ["org-listed-company", "named organisation read about in the trade press"],
      ["org-vendor", "named organisation in a supplier clause"],
      ["org-vendor-cross-segment", "named organisation whose supplier role is assigned in a later segment"],
      ["package-spec-boosted", "npm package specifier"],
      ["package-spec-plain", "npm package specifier"],
      ["person-name", "colleague's name"],
      ["pos-terminal-id", "card terminal id"],
      ["product-name", "internal software product name"],
      ["scheme-code", "internal product scheme code"],
      ["settlement-batch-number", "settlement batch number"],
      ["settlement-batch-sequence", "settlement batch sequence number"],
      ["sftp-endpoint", "sftp file-transfer endpoint"],
      ["ssh-public-key-line", "ssh public key line"],
      ["tan-boosted", "TAN (tax deduction and collection account number)"],
      ["tan-plain", "TAN (tax deduction and collection account number)"],
      ["uuid", "correlation id"],
      ["vendor-invoice-number", "supplier invoice number"],
    ]);
  });

  it("says which arm the counterExample-surface check protects, and does not say 'the prompt'", () => {
    // The claim this replaces: "a counterExample is text the compiled arm is
    // SHOWN". MEASURED false here rather than in a comment. The strings reach
    // the compiler's self-test generator and nothing else.
    const irStrings = IR.entityTypes.flatMap((e) => [...(e.counterExamples ?? []), ...(e.examples ?? [])]);
    expect(irStrings.length).toBeGreaterThan(40);
    const policyDoc = readFileSync(join(REPO_ROOT, "policies/p-fin.md"), "utf8");
    expect(irStrings.filter((c) => policyDoc.includes(c))).toEqual([]);
    expect(manifest.leakage.irOverlap.whatTheSurfaceCheckProtects).toContain("NOT a prompt");
    expect(manifest.leakage.irOverlap.whatTheSurfaceCheckProtects).toContain("TIER-0 rules");
    // And the exemption on the other half is a field rather than a silence.
    expect(manifest.leakage.irOverlap.carriersCheckedAgainstIrSurfaces).toBe(false);
  });

  it("reports the IR surfaces that DO occur verbatim, which the value-level check cannot see", () => {
    // `spanValuesFoundInIr` compares whole span VALUES, so a minted PEM block
    // never matches -- its body is random. Two IR `examples` entries are
    // nonetheless in the emitted text, and an empty value-level result beside
    // them would read as "no IR text reaches the corpus".
    const found = manifest.leakage.irOverlap.irSurfacesFoundInCorpusText;
    expect(found.map((f) => f.text)).toEqual([
      "-----BEGIN OPENSSH PRIVATE KEY-----",
      "-----BEGIN RSA PRIVATE KEY-----",
    ]);
    for (const f of found) {
      expect([f.text, f.kind, f.entityType]).toEqual([f.text, "examples", "private-key-material"]);
      expect([f.text, f.items]).toEqual([f.text, 3]);
      // Each really is in the IR and really is in the corpus, so the report is
      // not a list this test and the builder both invented.
      expect([f.text, IR_TEXT.includes(f.text)]).toEqual([f.text, true]);
      expect([f.text, items.filter((i) => i.text.includes(f.text)).length]).toEqual([f.text, f.items]);
    }
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

  it("matches on word boundaries in the wide reading, and on substrings in the tier-0 one", () => {
    // `containsTerm`'s boundary rule is documented as load-bearing and had no
    // test at all: replacing the regex with `includes` moved no number
    // anywhere, because no span in the shipped corpus sits near a string the
    // two matchers disagree about. These assertions pin the rule to what it
    // does, INCLUDING the case the old comment got backwards.
    const span = { start: 30, end: 40 };
    const pad = (before: string) => `${before.padStart(30, " ")}0123456789 tail`;
    // Excluded: the term is a prefix of a longer alphanumeric word.
    expect(boostNear(pad("connection "), span, ["connect"])).toBe(false);
    expect(boostNear(pad("accounts "), span, ["account"])).toBe(false);
    // Matched: the term stands alone.
    expect(boostNear(pad("the account "), span, ["account"])).toBe(true);
    // NOT excluded, and the comment used to say it was: `_` is outside
    // [a-z0-9], so the boundary class treats it as a separator.
    expect(boostNear(pad("storage_account_key "), span, ["account"])).toBe(true);
    // The tier-0 reading makes none of these distinctions, because
    // `hasNearbyKeyword` is `window.includes(k)`. That is why both are
    // reported rather than one standing in for the other.
    expect(tier0BoostNear(pad("connection "), span, ["connect"])).toBe(true);
    expect(tier0BoostNear(pad("accounts "), span, ["account"])).toBe(true);
  });

  it("copies runTier0's window and match rule, checked against runTier0 itself", () => {
    // The defect this test exists for: `BOOST_WINDOW` was documented as a free
    // choice because "runTier0 does not implement contextBoost". It does, at 40
    // characters, by substring, over a window that includes the span. Each
    // assertion below is a case where `tier0BoostNear` and `boostNear` give
    // different answers, and runTier0's own confidence -- 0.9 base, +0.05 when
    // boosted -- is the arbiter.
    const boosted = (text: string) => {
      const findings = runTier0(IR, text, segmentText(text));
      expect([text, findings.length]).toEqual([text, 1]);
      return findings[0]!.confidence > 0.9;
    };
    const spanOf = (text: string) => {
      const f = runTier0(IR, text, segmentText(text))[0]!;
      return { start: f.start, end: f.end };
    };
    const cases: { text: string; why: string }[] = [
      {
        // "cif" is inside the matched span itself. tier0's window includes it.
        text: "the reference on the ticket reads CIF 300455 and that is all we have.",
        why: "boost term inside the span",
      },
      {
        // "account" 20 characters before the match: inside 40, inside 70.
        text: "the beneficiary account was set up wrong, IFSC0012345 is what they sent.",
        why: "term within both windows",
      },
      {
        // "income tax" ends 49 characters before the match: outside tier0's 40,
        // inside boostNear's 70.
        text: "the income tax paperwork is on my desk and the number on it is ABCPZ1234C, which I cannot place.",
        why: "term between the two widths",
      },
    ];
    for (const c of cases) {
      const span = spanOf(c.text);
      const terms = boostTerms(IR).all;
      expect([c.why, tier0BoostNear(c.text, span, terms)]).toEqual([c.why, boosted(c.text)]);
    }
    // And the two functions really do disagree on the first and third, or the
    // agreement above would be a property of the examples rather than of the
    // mechanism.
    const disagreements = cases.filter((c) => {
      const span = spanOf(c.text);
      const terms = boostTerms(IR).all;
      return tier0BoostNear(c.text, span, terms) !== boostNear(c.text, span, terms);
    });
    expect(disagreements.map((c) => c.why)).toEqual(["boost term inside the span", "term between the two widths"]);
    expect(TIER0_BOOST_WINDOW).toBe(40);
    expect(BOOST_WINDOW).toBe(70);
  });

  it("reports both readings, and does not claim symmetry under the one that applies", () => {
    const after = measureBoost(items, IR, PAIRED_TYPE);
    const before = measureBoost(BEFORE, IR, WAVE2_PAIRED_TYPE);
    // The defect, restated as a measurement: before, a boost term sat near
    // 40.7% of gold spans and near 0.9% of confusable spans at the wide width,
    // and near 51.9% against 6.5% at the width runTier0 uses.
    expect(before.wideWordBoundary.goldRate).toBeGreaterThan(0.35);
    expect(before.wideWordBoundary.confusableRate).toBeLessThan(0.05);
    expect(before.wideWordBoundary.delta).toBeGreaterThan(0.3);
    expect(before.asTier0Reads.delta).toBeGreaterThan(0.4);
    // After, at the wide width: the -0.007 the previous emission published.
    expect(Math.abs(after.wideWordBoundary.delta)).toBeLessThan(0.08);
    expect(Math.abs(after.wideWordBoundary.ownTypeDelta)).toBeLessThan(0.08);
    // After, at the width and match rule the detector uses. THESE ARE
    // REGRESSION PINS ON A MEASURED RESIDUAL, not a symmetry claim: the corpus
    // does NOT meet the 0.08 bound on the own-type form under this reading, and
    // the manifest says so in `verdict` rather than the bound being widened to
    // fit. What is asserted structurally is the improvement, which is real and
    // large.
    expect(after.asTier0Reads.delta).toBeCloseTo(0.0753, 3);
    expect(after.asTier0Reads.ownTypeDelta).toBeCloseTo(0.0979, 3);
    expect(after.asTier0Reads.delta).toBeLessThan(before.asTier0Reads.delta / 5);
    expect(after.asTier0Reads.ownTypeDelta).toBeLessThan(before.asTier0Reads.ownTypeDelta / 4);
    expect(after.verdict).toContain("NOT SYMMETRIC");
    // Non-vacuous on both sides: a corpus with no boost terms at all would also
    // have a delta of zero and would be a different kind of unrealistic.
    expect(after.asTier0Reads.goldRate).toBeGreaterThan(0.25);
    expect(after.asTier0Reads.confusableRate).toBeGreaterThan(0.25);
    // And the two readings are not the same number wearing two labels.
    expect(after.asTier0Reads.delta).not.toBeCloseTo(after.wideWordBoundary.delta, 2);
  });

  it("publishes the before corpus's own-type DELTA and not its own-type RATE", () => {
    const before = measureBoost(BEFORE, IR, WAVE2_PAIRED_TYPE).asTier0Reads;
    // Hand-derivation, from the two counts rather than from `rates()`: at the
    // tier-0 width the before corpus carries 108 gold spans, 47 of which have a
    // boost term of their OWN entityType near them, and 108 confusable spans, 4
    // of which have a term of the type they are a near miss for. The rate is
    // 47/108 = 0.43519. The DELTA -- the statistic the residuals below are --
    // is (47 - 4)/108 = 43/108 = 0.39815. The manifest published 0.4352 for it,
    // which is the rate with nothing subtracted, and no test read the literal.
    const goldOwnTypeSpans = 47;
    const confusablePairedTypeSpans = 4;
    const spansOfEachKind = 108;
    expect([before.goldSpans, before.confusableSpans]).toEqual([spansOfEachKind, spansOfEachKind]);
    expect(before.goldOwnTypeRate).toBeCloseTo(goldOwnTypeSpans / spansOfEachKind, 12);
    expect(before.confusablePairedTypeRate).toBeCloseTo(confusablePairedTypeSpans / spansOfEachKind, 12);
    expect(before.goldOwnTypeRate).toBeCloseTo(0.43519, 5);
    expect(before.ownTypeDelta).toBeCloseTo(
      (goldOwnTypeSpans - confusablePairedTypeSpans) / spansOfEachKind,
      12,
    );
    expect(before.ownTypeDelta).toBeCloseTo(0.39815, 5);
    // The two are 4.8 points apart, so a manifest quoting one for the other is
    // not a rounding difference.
    expect(before.goldOwnTypeRate - before.ownTypeDelta).toBeGreaterThan(0.03);

    // The literal the module publishes is that delta at four places, and the
    // any-term figure beside it is that reading's delta too.
    expect(BEFORE_CORPUS_TIER0_BOOST.ownTypeDelta).toBe(Number(before.ownTypeDelta.toFixed(4)));
    expect(BEFORE_CORPUS_TIER0_BOOST.delta).toBe(Number(before.delta.toFixed(4)));
    expect(BEFORE_CORPUS_TIER0_BOOST).toEqual({ delta: 0.4537, ownTypeDelta: 0.3981 });

    // And the emitted manifest carries them, in that order, with the rate absent.
    expect(manifest.leakage.boost.verdict).toContain(
      `+${before.delta.toFixed(4)} and +${before.ownTypeDelta.toFixed(4)} on the same two forms`,
    );
    expect(manifest.leakage.boost.verdict).not.toContain(before.goldOwnTypeRate.toFixed(4));
  });

  it("carries the measurement into the manifest rather than only into this test", () => {
    expect(manifest.leakage.boost.asTier0Reads.window).toBe(TIER0_BOOST_WINDOW);
    expect(manifest.leakage.boost.wideWordBoundary.window).toBe(BOOST_WINDOW);
    expect(manifest.leakage.boost.verdict).toContain("NOT SYMMETRIC");
    // The manifest must not be able to publish a symmetry claim the numbers do
    // not support: the verdict is generated from the rates, so this pins the
    // two together.
    expect(manifest.leakage.boost.verdict).toContain(manifest.leakage.boost.asTier0Reads.delta.toFixed(4));
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

  it("drops the oracle's solved rate from 58 of 108 to zero, and says why that certifies nothing", () => {
    const before = measureOrthography(BEFORE);
    const after = measureOrthography(items);
    expect([before.goldSpans, before.goldSpansFound, before.goldSpansSolvedByOracle]).toEqual([108, 95, 58]);
    expect(after.goldSpansSolvedByOracle).toBe(0);
    // Non-vacuity, and it is the load-bearing half: an oracle that found
    // nothing would also solve nothing. It must still find most of the gold.
    expect(after.oracleRecall).toBeGreaterThan(0.8);
    expect(after.oraclePrecision).toBeLessThan(before.oraclePrecision);
    // AND THE STATISTIC IS ZEROED BY CONSTRUCTION, which is the point of the
    // rest of this block. Demonstrated rather than asserted: put ONE extra odd
    // token anywhere in every message of the corpus this replaces -- changing
    // nothing about how findable its gold span is -- and its solved rate goes
    // to zero too.
    const beforeWithADistractor = BEFORE.map((i) => ({ ...i, text: `${i.text}\nref ZZQQ7788XX9\n` }));
    expect(measureOrthography(beforeWithADistractor).goldSpansSolvedByOracle).toBe(0);
    expect(measureOrthography(beforeWithADistractor).oracleRecall).toBeCloseTo(before.oracleRecall, 6);
  });

  it("measures what the trivial reader actually scores, which is the number an arm has to beat", () => {
    const after = measureOrthography(items);
    // No bound is asserted on these. They are the corpus's own floor and the
    // honest reading of them is in `verdict`: a reader that understands nothing
    // scores in the same range as the arms measured on this corpus, so the
    // distractor injection reduced the leak and did not close it. Pinning a
    // threshold here would turn a disclosure back into a certification.
    expect(after.asScored.f1).toBeGreaterThan(0.4);
    // Scored as an arm is, so the two numbers are comparable: the oracle's
    // one-to-one precision is BELOW the touch-any-gold figure beside it, and
    // the verdict quotes the comparable one.
    expect(after.asScored.precision).toBeLessThan(after.oraclePrecision);
    expect(after.asScored.tp + after.asScored.fn).toBe(after.goldSpans);
    expect(after.verdict).toContain(after.asScored.precision.toFixed(3));
    expect(after.budgetMatched.tp + after.budgetMatched.fn).toBe(after.goldSpans);
    expect(after.budgetMatched.precision).toBeGreaterThan(0.6);
    expect(after.budgetMatched.recall).toBeGreaterThan(0.6);
    expect(after.firstHitIsGold).toBeGreaterThan(after.itemsWithGold / 2);
    expect(after.verdict).toContain("THIS IS THE FLOOR");
    expect(after.verdict).toContain("certifies nothing");
    // The budget-matched form is the one a distractor cannot flatter: adding an
    // odd token to every message does not move it the way it moves solvedRate.
    const padded = items.map((i) => ({ ...i, text: `${i.text}\nref ZZQQ7788XX9\n` }));
    const paddedReport = measureOrthography(padded);
    expect(paddedReport.goldSpansSolvedByOracle).toBe(0);
    expect(paddedReport.budgetMatched.recall).toBeCloseTo(after.budgetMatched.recall, 6);
    expect(manifest.leakage.orthography.verdict).toBe(after.verdict);
  });

  it("measures the positional discriminator the distractor introduced", () => {
    const position = measurePosition(items);
    // `generate.ts` places the distractor at (slotIndex + 1) % slots.length, so
    // this is a fact about the generator and not about the draw. It is measured
    // rather than fixed, and the manifest says so; the assertion is that the
    // number is PUBLISHED, not that it is small.
    expect(position.shadowedPairs).toBeGreaterThan(100);
    expect(position.distractorAfterGoldRate).toBeGreaterThan(0.5);
    expect(position.goldFirstRate).toBeGreaterThan(0.5);
    expect(position.verdict).toContain("THIS IS AN OPEN LEAK");
    expect(manifest.leakage.position).toEqual(position);
    // The corpus this replaces had no distractors at all, so it has no pairs --
    // which is what makes this a NEW leak rather than an inherited one.
    expect(measurePosition(BEFORE).shadowedPairs).toBe(0);
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

  it("publishes the per-name marginals, because 'locked' is a binary answer to a continuous question", () => {
    const after = measureRoles(items, orgRoleClassOf);
    // Every name appears on both sides, so roleLockedRate is 0 -- and the
    // counts are still lopsided enough that a name-only classifier beats the
    // majority class. The residual is what the binary statistic cannot say.
    expect(after.perName.map((p) => p.name).sort()).toEqual([...ORG_POOL].sort());
    for (const row of after.perName) {
      expect([row.name, Object.keys(row.counts).sort()]).toEqual([row.name, ["client-side", "non-client"]]);
    }
    expect(after.nameOnlyCorrect).toBe(
      after.perName.reduce((n, r) => n + Math.max(...Object.values(r.counts)), 0),
    );
    expect(after.nameOnlyLift).toBeGreaterThan(0);
    expect(after.nameOnlyAccuracy).toBeGreaterThan(after.majorityBaseline);
    expect(manifest.leakage.roles.perName).toEqual(after.perName);
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

  it("puts the circularity in the sentence a reader quotes, not only in a side block", () => {
    // certification.circularity has counted this since wave 3, but
    // invariantScope -- the field that says what the corpus is worth -- did not
    // mention that stage 1 IS an arm under test. The same objection this module
    // raises against standing tier 1 in for stage 2.
    const scope = manifest.certification.invariantScope;
    const c = manifest.certification.circularity!;
    expect(scope).toContain("CIRCULARITY");
    expect(scope).toContain("the tier-0 arm under test");
    // The numbers in the sentence come from the block beside it rather than
    // from a second count that could drift.
    expect(scope).toContain(`Measured against the IR-free ${c.independentSweep}`);
    expect(scope).toContain(`${c.stage1Only.length} carrier(s) quarantined by stage 1 alone`);
    expect(scope).toContain(`${c.both.length} by both`);
  });

  it("states the resolution of its own per-type rates", () => {
    // The shipped corpus's scarcest gold type carries 7 spans, so its recall
    // moves in 14.3-point steps. That is a coarser grid than "de-quantised"
    // suggests and the manifest now says the number rather than implying none.
    const r = manifest.counts.perTypeResolution;
    expect(r.minGoldSpansPerType).toBe(7);
    expect(r.scarcestTypes).toEqual(["in-pan"]);
    expect(r.note).toContain("14.3 percentage points");
    expect(r.minGoldSpansPerType).toBe(Math.min(...Object.values(manifest.counts.goldSpansByType)));
    expect(r.recallStepAtMin).toBeCloseTo(1 / 7, 12);
  });

  it("reports the phrase overlap the 8-gram ratio scores at zero", () => {
    const c = manifest.contamination;
    expect([c.dropped.length, c.maxScoreKept, c.itemsKept]).toEqual([0, 0, 189]);
    // Five items share a five-token sentence stem with the compiler self-test
    // corpus. The ratio cannot see it and the report says so beside the zero.
    expect(c.phraseOverlap.maxRunTokens).toBe(5);
    expect(c.phraseOverlap.worst).toHaveLength(5);
    for (const w of c.phraseOverlap.worst) {
      expect([w.itemId, w.sourceId]).toEqual([w.itemId, "policies/compiled/p-fin.selftest.json"]);
    }
  });

  it("marks the adjudication round's told channel unaudited", () => {
    expect(manifest.adjudication.blindness.unaudited).toEqual(["told"]);
    expect(manifest.unvalidated.some((u) => u.startsWith("THE TOLD CHANNEL"))).toBe(true);
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
