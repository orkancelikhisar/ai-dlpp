import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { getValidator, loadPolicyIr, runTier0, seededRng, segmentText, type PolicyIr } from "@sih/core";
import { loadSelfTestExamples } from "../src/corpus/build.js";
import { ALL_CARRIERS, carrierSlots, carrierText } from "../src/corpus/carriers.js";
import {
  CANDIDATE_CARRIERS,
  CANDIDATE_CLEAN_CARRIERS,
  CANDIDATE_DIRTY_CARRIERS,
  CANDIDATE_HARD_NEGATIVE_CARRIERS,
  CANDIDATE_ORDINARY_CARRIERS,
} from "../src/corpus/carriers.candidate.js";
import { certifyCarrier } from "../src/corpus/certify.js";
import { CONFUSABLE_FAMILIES, POSITIVE_FAMILIES } from "../src/corpus/families.js";
import {
  CANDIDATE_CONFUSABLE_FAMILIES,
  CANDIDATE_FAMILIES,
  CANDIDATE_POSITIVE_FAMILIES,
} from "../src/corpus/families.candidate.js";
import { generateCorpus } from "../src/corpus/generate.js";
import { applyInjections } from "../src/corpus/inject.js";
import { CLIENT_ORGS, FIRM, NON_CLIENT_ORGS } from "../src/corpus/universe.js";
import {
  COMPETITOR_ORGS,
  DOC_EXAMPLE_AWS_KEY,
  DUAL_ROLE_ORGS,
  LISTED_COMPANY_ORGS,
  PERSON_NAMES,
  PRODUCT_NAMES,
  mintGstin,
  mintUuid,
} from "../src/corpus/universe.candidate.js";

/**
 * The wave-2 carrier pool and family catalogue, exercised without being
 * emitted.
 *
 * Spec 6.2 puts certification before labelling -- "no prompt gets a label until
 * certified" -- so this suite runs the whole generator over wave 1 plus wave 2
 * IN MEMORY and asserts what the corpus would be, while
 * `corpora/generated/injection-p-fin-v1` stays exactly what it was and
 * `corpus-artifact.test.ts` keeps reproducing it.
 *
 * The dry run below is deliberately kept as the UNADJUDICATED baseline: it uses
 * the default sweeps only, so its 350 items over 50 usable carriers are what
 * the pipeline would emit if the only gate were the automated ones. The corpus
 * actually emitted is `injection-p-fin-adjudicated-v1` (189 items over 27
 * carriers), which adds the blind double-adjudication sweep; the difference
 * between the two numbers is what that round cost, and
 * `corpus-adjudicated.test.ts` holds the other end. Neither wave is in
 * `ALL_CARRIERS` or `ALL_FAMILIES`, which is why v1 still reproduces.
 */

const REPO = fileURLToPath(new URL("../../..", import.meta.url));
const IR_TEXT = readFileSync(`${REPO}policies/compiled/p-fin.ir.json`, "utf8");
const IR: PolicyIr = loadPolicyIr(IR_TEXT);
const SELFTEST = loadSelfTestExamples();
const SELFTEST_TEXT = SELFTEST.map((e) => e.text).join("\n");

/** The wave-1 + wave-2 corpus as it WOULD be. Never written to disk. */
const dryRun = generateCorpus({
  ir: IR,
  irSource: "policies/compiled/p-fin.ir.json",
  irHash: createHash("sha256").update(IR_TEXT, "utf8").digest("hex"),
  carriers: [...ALL_CARRIERS, ...CANDIDATE_CARRIERS],
  positiveFamilies: [...POSITIVE_FAMILIES, ...CANDIDATE_POSITIVE_FAMILIES],
  confusableFamilies: [...CONFUSABLE_FAMILIES, ...CANDIDATE_CONFUSABLE_FAMILIES],
  selfTestExamples: SELFTEST,
});

const ALL_NAMES = [
  ...DUAL_ROLE_ORGS,
  ...COMPETITOR_ORGS,
  ...LISTED_COMPANY_ORGS,
  ...PERSON_NAMES,
  ...PRODUCT_NAMES,
];

/** Every wave-2 family minted over 60 seeds. Property tests run over this. */
const MINTED = CANDIDATE_FAMILIES.flatMap((f) =>
  Array.from({ length: 60 }, (_, i) => ({ family: f.id, type: f.type, value: f.mint(seededRng(`cand|${f.id}|${i}`)) })),
);

function injectOnce(carrierId: string, familyId: string, slot = 0) {
  const carrier = [...ALL_CARRIERS, ...CANDIDATE_CARRIERS].find((c) => c.id === carrierId)!;
  const family = CANDIDATE_FAMILIES.find((f) => f.id === familyId)!;
  const value = family.mint(seededRng(`inject|${carrierId}|${familyId}`));
  const { prefix, suffix } = family.glue(value);
  const slots = carrierSlots(carrier);
  const result = applyInjections(carrierText(carrier), [
    { at: slots[slot]!, prefix, value, suffix, type: family.type, family: family.id, dimensions: {} },
  ]);
  return { ...result, value, span: result.injections[0]!.span };
}

describe("certification of the wave-2 carriers", () => {
  it("clears every ordinary and hard-negative carrier with no hit at all", () => {
    // "provisional-clear", never "certified-clear": two of spec 6.2's three
    // stages cannot run in this repository. See certify.ts.
    for (const carrier of CANDIDATE_CLEAN_CARRIERS) {
      const cert = certifyCarrier(carrier.id, carrierText(carrier), { ir: IR });
      expect([carrier.id, cert.status, cert.hits]).toEqual([carrier.id, "provisional-clear", []]);
    }
  });

  it("quarantines every dirty carrier, each through the sweep it was written for", () => {
    // Asserting the SWEEP and the LABEL, not just the status. A dirty carrier
    // that quarantined for an unintended reason would still be "quarantined",
    // and the pool would silently stop covering the arm it was written to cover.
    const expected: Record<string, [string, string]> = {
      "d04-pan-shaped": ["tier0-sweep", "in-pan"],
      "d05-aadhaar-digits": ["tier0-sweep", "in-aadhaar"],
      "d06-ifsc-shaped": ["tier0-sweep", "bank-account-identifier"],
      "d07-entropy-fence": ["tier0-sweep", "api-credential"],
    };
    expect(CANDIDATE_DIRTY_CARRIERS.map((c) => c.id).sort()).toEqual(Object.keys(expected).sort());
    for (const carrier of CANDIDATE_DIRTY_CARRIERS) {
      const cert = certifyCarrier(carrier.id, carrierText(carrier), { ir: IR });
      expect([carrier.id, cert.status]).toEqual([carrier.id, "quarantined"]);
      expect([carrier.id, cert.hits.map((h) => [h.sweep, h.label])]).toEqual([carrier.id, [expected[carrier.id]!]]);
    }
  });

  it("keeps three of the four dirty carriers invisible to the STOCK rules", () => {
    // The whole claim of `maxRecallIr` is that dropping validators and lowering
    // the entropy floor widens what a certification sweep sees. These three
    // carriers are the evidence: p-fin as shipped returns nothing on them, and
    // only the widened IR quarantines them. d06 is the control -- its rule has
    // no validator to drop, so the stock IR catches it too, and a change that
    // accidentally made the max-recall transform a no-op would leave d06
    // passing while the other three flipped.
    const stock = (id: string) => {
      const text = carrierText(CANDIDATE_DIRTY_CARRIERS.find((c) => c.id === id)!);
      return runTier0(IR, text, segmentText(text)).map((f) => f.entityType);
    };
    expect(stock("d04-pan-shaped")).toEqual([]);
    expect(stock("d05-aadhaar-digits")).toEqual([]);
    expect(stock("d07-entropy-fence")).toEqual([]);
    expect(stock("d06-ifsc-shaped")).toEqual(["bank-account-identifier"]);
  });

  it("leaves no clean carrier carrying a stock tier-0 finding either", () => {
    // The pristine negatives come from these carriers verbatim. A finding here
    // would be a false positive the corpus manufactured rather than measured.
    for (const carrier of CANDIDATE_CLEAN_CARRIERS) {
      const text = carrierText(carrier);
      expect([carrier.id, runTier0(IR, text, segmentText(text))]).toEqual([carrier.id, []]);
    }
  });
});

describe("nothing the arms are shown contains a corpus value", () => {
  it("no minted value and no universe name is in the compiled IR", () => {
    expect(MINTED.filter((m) => IR_TEXT.includes(m.value)).map((m) => m.family)).toEqual([]);
    expect(ALL_NAMES.filter((n) => IR_TEXT.includes(n))).toEqual([]);
  });

  it("no minted value and no universe name is in a compiler self-test example", () => {
    expect(MINTED.filter((m) => SELFTEST_TEXT.includes(m.value)).map((m) => m.family)).toEqual([]);
    expect(ALL_NAMES.filter((n) => SELFTEST_TEXT.includes(n))).toEqual([]);
  });

  it("and the check is not vacuous, on both sides", () => {
    // A value that IS in each source, to prove the haystacks are the real ones.
    expect(IR_TEXT).toContain("AAAPZ1234C");
    expect(SELFTEST_TEXT).toContain("AKIAIOSFODNN7EXAMPLE");
    // AWS's documented key is the natural literal for the tutorial-key family
    // and is exactly what this check forbids: it is already a hard negative in
    // the compiler's own self-test corpus (example 180). The family therefore
    // mints a different EXAMPLE-bearing key, and this assertion is what stops
    // anyone putting the obvious one back.
    expect(DOC_EXAMPLE_AWS_KEY).not.toBe("AKIAIOSFODNN7EXAMPLE");
    expect(MINTED.length).toBeGreaterThan(900);
  });
});

describe("the near-miss pairs", () => {
  it("draws client and vendor organisations from ONE pool", () => {
    // Wave 1's client pool and non-client pool are disjoint AND differ in
    // flavour ("Kestrel Ironworks" against "Cobblestone Print Works"), so the
    // role can be read off the name. Wave 2 removes that: over enough seeds the
    // client families and the vendor families draw the SAME set of names, which
    // is the only construction under which a right answer requires reading the
    // clause.
    const drawn = (ids: string[]) =>
      new Set(
        ids.flatMap((id) => {
          const f = CANDIDATE_FAMILIES.find((x) => x.id === id)!;
          return Array.from({ length: 200 }, (_, i) => f.mint(seededRng(`pool|${id}|${i}`)));
        }),
      );
    const clientDraws = drawn(["dual-role-org-prospect", "dual-role-org-mandate", "client-org-message-scope"]);
    const vendorDraws = drawn(["dual-role-org-vendor", "dual-role-org-vendor-message-scope"]);
    expect([...clientDraws].sort()).toEqual([...DUAL_ROLE_ORGS].sort());
    expect([...vendorDraws].sort()).toEqual([...DUAL_ROLE_ORGS].sort());
  });

  it("puts the message-scope pair's name and role clause in different segments", () => {
    // p-fin declares one predicate and its scope is "message". This pair is
    // what makes that word measurable: the organisation is named in the first
    // prose segment, a fenced block forces a segment break, and the sentence
    // that establishes the relationship is in the third. No segment holds both,
    // so a segment-scoped judge cannot reach the answer and a message-scoped
    // one can.
    for (const [familyId, marker] of [
      ["client-org-message-scope", "on our books"],
      ["dual-role-org-vendor-message-scope", "supplying our print"],
    ] as const) {
      const { text, span, value } = injectOnce("c01", familyId);
      const segments = segmentText(text);
      const nameSegment = segments.find((s) => s.start <= span.start && span.end <= s.end)!;
      const roleSegments = segments.filter((s) => s.text.includes(marker));
      expect([familyId, nameSegment.kind]).toEqual([familyId, "prose"]);
      expect([familyId, roleSegments.length]).toEqual([familyId, 1]);
      expect([familyId, nameSegment.text.includes(marker)]).toEqual([familyId, false]);
      expect([familyId, roleSegments[0]!.text.includes(value)]).toEqual([familyId, false]);
      expect([familyId, segments.some((s) => s.kind === "code")]).toEqual([familyId, true]);
    }
  });

  it("mints a GSTIN whose embedded PAN the shipping validator accepts", () => {
    // The pair that separates "is a PAN" from "contains a PAN". `pan-structure`
    // accepts the middle ten characters, and `in-pan-format` still does not
    // match, because neither end of them is a word boundary.
    const panStructure = getValidator("pan-structure");
    const panRule = new RegExp(IR.rules.find((r) => r.id === "in-pan-format")!.regex!);
    for (let i = 0; i < 100; i += 1) {
      const gstin = mintGstin(seededRng(`gstin|${i}`));
      expect([gstin, gstin.length]).toEqual([gstin, 15]);
      expect([gstin, panStructure(gstin.slice(2, 12))]).toEqual([gstin, true]);
      expect([gstin, panRule.test(gstin)]).toEqual([gstin, false]);
    }
  });

  it("mints a UUID that no p-fin rule reaches", () => {
    for (let i = 0; i < 100; i += 1) {
      const uuid = mintUuid(seededRng(`uuid|${i}`));
      expect([uuid, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(uuid)]).toEqual([uuid, true]);
    }
  });

  it("names no organisation that is a substring of another", () => {
    // A substring pair would make "occurs exactly as many times as injected"
    // ambiguous the first time both landed in one message.
    const all = [...ALL_NAMES, ...CLIENT_ORGS, ...NON_CLIENT_ORGS, FIRM];
    const collisions: string[] = [];
    for (const a of all) for (const b of all) if (a !== b && b.includes(a)) collisions.push(`${a} in ${b}`);
    expect(collisions).toEqual([]);
  });
});

describe("what p-fin's own tier-0 rules do to each wave-2 family", () => {
  // Characterisation, measured over one reference carrier. These are the
  // numbers that say what each family ASKS of an arm, and pinning them means a
  // later glue edit that quietly changes the task shows up here rather than in
  // a results table.
  const classify = (familyId: string) => {
    const { text, span } = injectOnce("c01", familyId);
    const findings = runTier0(IR, text, segmentText(text));
    const exact = findings.filter((f) => f.start === span.start && f.end === span.end);
    const inside = findings.filter((f) => f.start >= span.start && f.end <= span.end && !(f.start === span.start && f.end === span.end));
    const elsewhere = findings.filter((f) => f.end <= span.start || f.start >= span.end);
    return {
      exact: exact.map((f) => f.entityType),
      inside: inside.map((f) => f.entityType),
      elsewhere: elsewhere.map((f) => f.entityType),
    };
  };

  it("matches the gold span exactly on ten of the eighteen positive families", () => {
    const exactly = CANDIDATE_POSITIVE_FAMILIES.filter((f) => classify(f.id).exact.includes(f.type));
    expect(exactly.map((f) => f.id).sort()).toEqual([
      "aadhaar-dashed",
      "aadhaar-under-pan-context",
      "client-secret-entropy-kv",
      "ghp-token-prose",
      "ifsc-kv",
      "jdbc-url-prose",
      "kyc-case-id",
      "mongo-url-kv",
      "pan-under-aadhaar-context",
      "upi-vpa-fictional-handle",
    ]);
  });

  it("leaves seven positive families entirely to the tiers above 0", () => {
    // Five client names (tier 1), one HSM key label and one bare account
    // number. This is the corpus asking the ladder a question tier 0 cannot
    // answer, which is the only way a ladder curve says anything.
    const invisible = CANDIDATE_POSITIVE_FAMILIES.filter((f) => {
      const c = classify(f.id);
      return c.exact.length === 0 && c.inside.length === 0 && c.elsewhere.length === 0;
    });
    expect(invisible.map((f) => f.id).sort()).toEqual([
      "bare-account-digits",
      "client-org-message-scope",
      "dual-role-org-mandate",
      "dual-role-org-nda",
      "dual-role-org-prospect",
      "dual-role-org-trade-counterparty",
      "hsm-key-label",
    ]);
  });

  it("fires on exactly four confusable families, which is what makes them hard", () => {
    // A confusable no rule fires on is only a test of the model. These four are
    // also a measured false-positive source for tier 0, and under the injection
    // invariant they are TRUE false positives rather than corpus artifacts.
    const noisy = CANDIDATE_CONFUSABLE_FAMILIES.filter((f) => {
      const c = classify(f.id);
      return c.exact.length + c.inside.length + c.elsewhere.length > 0;
    });
    expect(noisy.map((f) => f.id).sort()).toEqual([
      "base64-image-fragment",
      "csr-pem-block",
      "redacted-credential-placeholder",
      "tutorial-api-key",
    ]);
    // The CSR is the sharpest of the four: p-fin is right about the header and
    // wrong about the body, so the finding lands inside the span with the wrong
    // entityType rather than on it.
    expect(classify("csr-pem-block")).toEqual({ exact: [], inside: ["api-credential", "api-credential"], elsewhere: [] });
    expect(classify("tutorial-api-key").exact).toEqual(["api-credential"]);
  });
});

describe("the wave-1 + wave-2 corpus, generated in memory and not written", () => {
  const m = dryRun.manifest;

  it("more than doubles the corpus and lifts every per-type denominator", () => {
    expect([m.counts.items, m.counts.positives, m.counts.negatives]).toEqual([350, 200, 150]);
    expect([m.counts.goldSpans, m.counts.confusableSpans]).toEqual([200, 200]);
    const scorable = IR.entityTypes.filter((e) => e.tier !== 2).map((e) => e.id);
    for (const id of scorable) expect([id, m.counts.goldSpansByType[id]! >= 17]).toEqual([id, true]);
    // 24 confusable types, none below 8. The corpus this replaces had two
    // tier-2 gold positives and a recall quantised to {0, 0.5, 1}.
    expect(Object.keys(m.counts.confusableSpansByType)).toHaveLength(24);
    for (const [type, n] of Object.entries(m.counts.confusableSpansByType)) expect([type, n >= 8]).toEqual([type, true]);
  });

  it("quarantines the seven dirty carriers and nothing else", () => {
    expect(m.certification.carriers).toEqual({
      total: 57,
      certifiedClear: 0,
      provisionalClear: 50,
      quarantined: 7,
    });
    expect(m.certification.claim).toBe("NOT CERTIFIED");
  });

  it("carries the hard-negative stratum into the items", () => {
    const hard = dryRun.items.filter((i) => i.meta?.["carrierStratum"] === "hard-negative");
    expect(hard).toHaveLength(CANDIDATE_HARD_NEGATIVE_CARRIERS.length * 7);
    // One pristine negative per hard-negative carrier: no injection at all, so
    // any finding on it is over-blocking on the stratum spec 6.2 asks to be
    // reported separately.
    const pristine = hard.filter((i) => i.gold.length === 0 && i.meta?.["density"] === 0);
    expect(pristine).toHaveLength(CANDIDATE_HARD_NEGATIVE_CARRIERS.length);
    // And no ordinary carrier picked the field up.
    const ordinaryIds = new Set(CANDIDATE_ORDINARY_CARRIERS.map((c) => c.id));
    expect(dryRun.items.filter((i) => ordinaryIds.has(i.meta?.["carrierId"] as string) && i.meta?.["carrierStratum"] !== undefined)).toEqual([]);
  });

  it("never labels one string with two types in one message", () => {
    // The generator re-mints on a shared-pool collision. Without that,
    // `assertInjectionInvariant`'s fourth check refuses the item -- measured at
    // 70 of 40,730 possible (positive family x confusable family x carrier)
    // pairings, every one of them two organisation families drawing one name.
    for (const item of dryRun.items) {
      const byValue = new Map<string, Set<string>>();
      for (const inj of (item.meta?.["injections"] ?? []) as { text: string; type: string }[]) {
        const set = byValue.get(inj.text) ?? new Set<string>();
        set.add(inj.type);
        byValue.set(inj.text, set);
      }
      for (const [value, types] of byValue) expect([item.id, value, types.size]).toEqual([item.id, value, 1]);
    }
  });

  it("re-mints its way out of a shared-pool collision, and gives up when it cannot", () => {
    // MEASURED, by deleting the re-mint loop and re-running this file: the test
    // above passes either way. At the default seed no shared-pool collision
    // actually occurs in the 350-item dry run, so that test proves the corpus
    // is clean and proves nothing about the loop that keeps it clean. This one
    // forces the collision instead of waiting for it.
    //
    // A pool of TWO: every dual-injection item draws twice from it, so a naive
    // generator collides with probability 1/2 per item and the fourth invariant
    // check refuses the corpus. A pool of ONE: every draw collides and no
    // number of re-mints can escape, which is the bounded loop's other end --
    // it has to give up and say why rather than spin.
    const pool = (names: readonly string[]) => (rng: () => number) => names[Math.floor(rng() * names.length)]!;
    const shape = { surface: "prose", difficulty: "verbatim", register: "formal" } as const;
    const families = (names: readonly string[]) => ({
      positiveFamilies: [
        {
          id: "t-client",
          type: "client-name",
          ...shape,
          constructedRole: "client" as const,
          mint: pool(names),
          glue: () => ({ prefix: " our client ", suffix: " has asked again." }),
        },
      ],
      confusableFamilies: [
        {
          id: "t-vendor",
          type: "neg:t-vendor",
          ...shape,
          constructedRole: "vendor" as const,
          mint: pool(names),
          glue: () => ({ prefix: " the facilities supplier is ", suffix: "." }),
        },
      ],
    });
    const run = (names: readonly string[]) =>
      generateCorpus({
        seed: "collision-probe",
        ir: IR,
        irSource: "policies/compiled/p-fin.ir.json",
        irHash: "0".repeat(64),
        carriers: CANDIDATE_ORDINARY_CARRIERS.slice(0, 2),
        ...families(names),
        selfTestExamples: SELFTEST,
      });

    const twoNames = run(["Aldermoor Group", "Bramfield Textiles"]);
    for (const item of twoNames.items) {
      const types = new Map<string, string>();
      for (const inj of (item.meta?.["injections"] ?? []) as { text: string; type: string }[]) {
        expect([item.id, inj.text, types.get(inj.text) ?? inj.type]).toEqual([item.id, inj.text, inj.type]);
        types.set(inj.text, inj.type);
      }
    }
    // Both roles were actually written, so the collision had somewhere to come from.
    const roles = new Set(
      twoNames.items.flatMap((i) =>
        ((i.meta?.["injections"] ?? []) as { dimensions: Record<string, string> }[]).map((x) => x.dimensions["constructedRole"]),
      ),
    );
    expect([...roles].sort()).toEqual(["client", "vendor"]);

    expect(() => run(["Aldermoor Group"])).toThrowError(/injected as both "client-name" and "neg:t-vendor"/);
  });

  it("refuses the contradiction directly, so the previous test is not vacuous", () => {
    const org = DUAL_ROLE_ORGS[0]!;
    expect(() =>
      applyInjections("a carrier sentence that ends here.", [
        { at: 0, prefix: "our client ", value: org, suffix: " asked. ", type: "client-name", family: "a", dimensions: {} },
        { at: 33, prefix: " the supplier is ", value: org, suffix: ".", type: "neg:dual-role-org-vendor", family: "b", dimensions: {} },
      ]),
    ).toThrowError(/injected as both "client-name" and "neg:dual-role-org-vendor"/);
  });

  it("fires outside a gold span only on an injected confusable", () => {
    // The invariant spec 6.2 buys: on positives the gold spans are exactly the
    // injected ones, so any finding outside them is a true false positive. Here
    // every one of them lands on a confusable that was injected to provoke it,
    // and none lands on carrier text.
    const outside: string[] = [];
    for (const item of dryRun.items) {
      const injections = (item.meta?.["injections"] ?? []) as { start: number; end: number; type: string }[];
      for (const f of runTier0(IR, item.text, segmentText(item.text))) {
        if (item.gold.some((g) => f.start < g.end && g.start < f.end)) continue;
        const on = injections.find((x) => f.start < x.end && x.start < f.end);
        outside.push(`${f.entityType} <- ${on?.type ?? "carrier text"}`);
      }
    }
    const tally: Record<string, number> = {};
    for (const o of outside) tally[o] = (tally[o] ?? 0) + 1;
    expect(tally).toEqual({
      "bank-account-identifier <- neg:email-address": 9,
      "api-credential <- neg:base64-image-fragment": 8,
      "api-credential <- neg:csr-pem-block": 18,
      "api-credential <- neg:redacted-credential-placeholder": 8,
      "api-credential <- neg:tutorial-api-key": 8,
    });
  });

  it("finds nothing to drop for contamination against the compiler self-test", () => {
    expect(m.contamination.dropped).toEqual([]);
    expect(m.contamination.maxScoreKept).toBe(0);
    expect(m.contamination.itemsChecked).toBe(350);
  });

  it("still emits no pred: gold and still marks the predicate labelling unrun", () => {
    for (const item of dryRun.items) {
      for (const g of item.gold) expect(g.entityType.startsWith("pred:")).toBe(false);
    }
    const orgItems = dryRun.items.filter((i) =>
      ((i.meta?.["injections"] ?? []) as { dimensions: Record<string, string> }[]).some(
        (x) => x.dimensions["constructedRole"] !== "none",
      ),
    );
    expect(orgItems.length).toBeGreaterThan(60);
    for (const i of orgItems) expect(i.meta?.["predicateLabelling"]).toMatchObject({ state: "unlabelled" });
  });
});
