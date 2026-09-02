import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { runTier0, segmentText } from "@sih/core";
import { loadCorpus } from "../src/driver/corpus.js";
import { ADMITTED_CARRIER_IDS, adjudicationSweep, isAdmitted } from "../src/corpus/adjudication.js";
import { IR_PATH, loadSelfTestExamples } from "../src/corpus/build.js";
import {
  ADJUDICATED_CORPUS_PATH,
  ADJUDICATED_CORPUS_RELPATH,
  ADJUDICATED_MANIFEST_PATH,
  EXCLUDED_FAMILY_IDS,
  OFFERED_CARRIERS,
  buildAdjudicatedArtifacts,
  verifyOrRefuse,
} from "../src/corpus/build-adjudicated.js";
import { carrierText } from "../src/corpus/carriers.js";
import { certifyCarrier, orthographicOrgSweep } from "../src/corpus/certify.js";
import { checkContamination } from "../src/corpus/contamination.js";
import { loadPolicyIr } from "@sih/core";

/**
 * `corpora/generated/injection-p-fin-adjudicated-v1`: the artifact, and the
 * invariant it is emitted under.
 *
 * The pattern is `corpus-artifact.test.ts`'s -- regenerate through the real
 * builder and compare to the committed bytes, so a hand-edit or a stale seed
 * cannot pass. What is added here is the other half: the emit gate is exercised
 * against corpora that VIOLATE the invariant, because a gate that has only ever
 * been run on a clean corpus has not been shown to stop a dirty one.
 */
const built = buildAdjudicatedArtifacts();
const IR = loadPolicyIr(readFileSync(IR_PATH, "utf8"));

describe("the committed artifact", () => {
  it("reproduces the committed corpus byte for byte", () => {
    expect(readFileSync(ADJUDICATED_CORPUS_PATH, "utf8")).toBe(built.corpusJsonl);
  });

  it("reproduces the committed manifest byte for byte", () => {
    expect(readFileSync(ADJUDICATED_MANIFEST_PATH, "utf8")).toBe(built.manifestJson);
  });

  it("matches the sha256 and byte count the manifest records", () => {
    const manifest = JSON.parse(readFileSync(ADJUDICATED_MANIFEST_PATH, "utf8")) as {
      artifact: { corpusSha256: string; corpusBytes: number; corpusPath: string };
    };
    expect(manifest.artifact.corpusSha256).toBe(built.manifest.artifact.corpusSha256);
    expect(manifest.artifact.corpusBytes).toBe(Buffer.byteLength(readFileSync(ADJUDICATED_CORPUS_PATH, "utf8"), "utf8"));
    expect(manifest.artifact.corpusPath).toBe(ADJUDICATED_CORPUS_RELPATH);
  });

  it("records no absolute path anywhere in the manifest", () => {
    // An absolute path would stamp the build machine's home directory into a
    // committed file and break the reproduction check on every other checkout.
    // Found exactly that way: the first emit wrote `verification.source` as an
    // absolute path.
    expect(built.manifestJson).not.toMatch(/"\/[A-Za-z]/);
    expect(built.manifestJson).not.toContain("/Users/");
    expect(built.manifestJson).not.toContain("/home/");
  });

  it("loads through the harness reader", () => {
    expect(loadCorpus(readFileSync(ADJUDICATED_CORPUS_PATH, "utf8"))).toHaveLength(189);
  });

  it("contains no raw control characters", () => {
    const bytes = readFileSync(ADJUDICATED_CORPUS_PATH, "utf8");
    expect([...bytes].filter((c) => c !== "\n" && c.charCodeAt(0) < 0x20)).toEqual([]);
  });

  it("is a new file and touches neither the smoke fixtures nor the v1 corpus", () => {
    expect(ADJUDICATED_CORPUS_PATH).toContain("corpora/generated/");
    expect(ADJUDICATED_CORPUS_PATH).not.toContain("fixtures");
    expect(ADJUDICATED_CORPUS_PATH).not.toContain("injection-p-fin-v1");
    // No item id can collide with the v1 corpus: v1's carriers are c01..c22 and
    // d01..d03, this corpus's are o01..o16 and hn02..hn12, so records scored
    // against one can never be pooled with records scored against the other.
    for (const item of built.items) expect([item.id, /^(inj|neg)-(o\d\d|hn\d\d)/.test(item.id)]).toEqual([item.id, true]);
  });
});

describe("the injection invariant, re-derived from the file", () => {
  const onDisk = loadCorpus(readFileSync(ADJUDICATED_CORPUS_PATH, "utf8"));

  it("holds text.slice(start, end) === gold.text on every gold span", () => {
    let checked = 0;
    for (const item of onDisk) {
      for (const g of item.gold) {
        expect([item.id, g.start, item.text.slice(g.start, g.end)]).toEqual([item.id, g.start, g.text]);
        checked += 1;
      }
    }
    expect(checked).toBe(108);
  });

  it("puts every gold span on a value the generator wrote, and every injected value once", () => {
    for (const item of onDisk) {
      const injections = (item.meta?.["injections"] ?? []) as { start: number; end: number; text: string; type: string }[];
      for (const g of item.gold) {
        const written = injections.filter((i) => i.start === g.start && i.end === g.end && i.text === g.text);
        expect([item.id, g.start, written.length]).toEqual([item.id, g.start, 1]);
        expect([item.id, g.start, written[0]!.type]).toEqual([item.id, g.start, g.entityType]);
      }
      for (const inj of injections) {
        const expected = injections.filter((i) => i.text === inj.text).length;
        let seen = 0;
        for (let i = item.text.indexOf(inj.text); i !== -1; i = item.text.indexOf(inj.text, i + 1)) seen += 1;
        expect([item.id, inj.text, seen]).toEqual([item.id, inj.text, expected]);
      }
    }
  });

  it("was injected only into carriers both certifiers cleared with no reservation", () => {
    const used = new Set(onDisk.map((i) => i.meta?.["carrierId"] as string));
    expect([...used].sort()).toEqual([...ADMITTED_CARRIER_IDS].sort());
    for (const id of used) expect([id, isAdmitted(id)]).toEqual([id, true]);
    // And the carriers that were offered and refused really are absent.
    for (const id of ["hn01", "d04-pan-shaped", "d05-aadhaar-digits", "d06-ifsc-shaped", "d07-entropy-fence", "c01"]) {
      expect([id, used.has(id)]).toEqual([id, false]);
    }
  });

  it("emits no pred: gold and no neg: gold", () => {
    for (const item of onDisk) {
      for (const g of item.gold) {
        expect([item.id, g.entityType.startsWith("pred:"), g.entityType.startsWith("neg:")]).toEqual([
          item.id,
          false,
          false,
        ]);
      }
    }
  });

  it("fires tier 0 outside a gold span only on an injected confusable, never on carrier text", () => {
    // This is what the invariant buys, measured: on these items every tier-0
    // finding that is not a gold span lands on a value the generator injected
    // to provoke it. A finding on carrier text would mean the corpus is
    // manufacturing false positives and attributing them to the arm.
    const tally: Record<string, number> = {};
    for (const item of onDisk) {
      const injections = (item.meta?.["injections"] ?? []) as { start: number; end: number; type: string }[];
      for (const f of runTier0(IR, item.text, segmentText(item.text))) {
        if (item.gold.some((g) => f.start < g.end && g.start < f.end)) continue;
        const on = injections.find((x) => f.start < x.end && x.start < f.end);
        const key = `${f.entityType} <- ${on?.type ?? "CARRIER TEXT"}`;
        tally[key] = (tally[key] ?? 0) + 1;
      }
    }
    expect(tally).toEqual({
      "api-credential <- neg:base64-image-fragment": 4,
      "api-credential <- neg:csr-pem-block": 8,
      "api-credential <- neg:redacted-credential-placeholder": 5,
      "api-credential <- neg:tutorial-api-key": 4,
      "bank-account-identifier <- neg:email-address": 4,
    });
  });
});

describe("the emit gate refuses a corpus that violates the invariant", () => {
  const clone = () => JSON.parse(JSON.stringify(built.items)) as typeof built.items;

  it("refuses a gold span whose offsets do not hold its text", () => {
    // ISOLATED, and the isolation is the point. An earlier version of this test
    // appended a character to `gold.text` and asserted /gold span \[/. MEASURED
    // by mutation: with the slice comparison replaced by `gold.text !==
    // gold.text` -- always false -- that test still passed, because the doctored
    // text then failed the NEXT check and threw a different message the loose
    // regex also matched. Three of this gate's messages begin "gold span [".
    //
    // So this shifts the gold span AND the injection that wrote it by the same
    // one character, leaving every other check satisfiable: the value still
    // occurs exactly once, the injection lookup still matches by offset and by
    // text, and the only broken property is the one under test.
    const items = clone();
    const victim = items.find(
      (i) => i.gold.length > 0 && i.gold[0]!.end + 1 <= i.text.length,
    )!;
    const gold = victim.gold[0]! as { start: number; end: number; text: string };
    const inj = ((victim.meta as { injections: { start: number; end: number; text: string }[] }).injections).find(
      (x) => x.start === gold.start && x.end === gold.end,
    )!;
    expect(victim.text.slice(gold.start + 1, gold.end + 1)).not.toBe(gold.text);
    gold.start += 1;
    gold.end += 1;
    inj.start += 1;
    inj.end += 1;
    expect(() => verifyOrRefuse(items, built.generated.certifications, "probe")).toThrowError(
      /gold span \[\d+,\d+\) is ".*", not "/,
    );
  });

  it("refuses a gold span typed with a confusable or predicate id", () => {
    // `neg:` and `pred:` ids live in a different namespace from
    // `Finding.entityType`; a scorer joining on string equality would count one
    // as a permanent recall miss. This corpus emits neither, and the gate is
    // what makes that a checked property rather than a habit of the generator.
    const items = clone();
    const victim = items.find((i) => i.gold.length > 0)!;
    (victim.gold[0] as { entityType: string }).entityType = "neg:dual-role-org-vendor";
    expect(() => verifyOrRefuse(items, built.generated.certifications, "probe")).toThrowError(
      /carries type "neg:dual-role-org-vendor"/,
    );
  });

  it("refuses a gold span the generator did not write", () => {
    const items = clone();
    const victim = items.find((i) => i.gold.length > 0)!;
    (victim.meta as { injections: unknown[] }).injections = [];
    expect(() => verifyOrRefuse(items, built.generated.certifications, "probe")).toThrowError(
      /matches 0 recorded injections/,
    );
  });

  it("refuses a gold span typed differently from the injection that wrote it", () => {
    const items = clone();
    const victim = items.find((i) => i.gold.length > 0)!;
    (victim.gold[0] as { entityType: string }).entityType = "client-name";
    const first = (victim.meta as { injections: { type: string }[] }).injections[0]!;
    if (first.type === "client-name") first.type = "in-pan";
    expect(() => verifyOrRefuse(items, built.generated.certifications, "probe")).toThrowError(
      /but the injection that wrote it was typed/,
    );
  });

  it("refuses an item whose carrier the round refused", () => {
    const items = clone();
    (items.find((i) => i.gold.length > 0)!.meta as { carrierId: string }).carrierId = "hn01";
    expect(() => verifyOrRefuse(items, built.generated.certifications, "probe")).toThrowError(
      /adjudication round refused \(borderline\)/,
    );
  });

  it("refuses an item whose carrier nobody adjudicated", () => {
    const items = clone();
    (items.find((i) => i.gold.length > 0)!.meta as { carrierId: string }).carrierId = "c01";
    expect(() => verifyOrRefuse(items, built.generated.certifications, "probe")).toThrowError(
      /adjudication round refused \(unadjudicated\)/,
    );
  });

  it("refuses an injected value that occurs more often than it was injected", () => {
    const items = clone();
    const victim = items.find((i) => i.gold.length > 0)!;
    const value = ((victim.meta as { injections: { text: string }[] }).injections[0]!).text;
    (victim as { text: string }).text = `${victim.text} ${value}`;
    expect(() => verifyOrRefuse(items, built.generated.certifications, "probe")).toThrowError(/but it occurs 2 time/);
  });

  it("accepts the real corpus, so the refusals above are not a gate stuck shut", () => {
    const report = verifyOrRefuse(built.items, built.generated.certifications, "probe");
    expect([report.itemsChecked, report.positives, report.negatives, report.goldSpansChecked]).toEqual([
      189, 108, 81, 108,
    ]);
  });
});

describe("what the manifest reports", () => {
  const m = built.manifest;

  it("counts 189 items over 27 carriers, with per-type denominators in double figures", () => {
    expect([m.counts.items, m.counts.positives, m.counts.negatives]).toEqual([189, 108, 81]);
    expect([m.counts.goldSpans, m.counts.confusableSpans]).toEqual([108, 108]);
    expect(m.counts.goldSpansByType).toEqual({
      "api-credential": 16,
      "bank-account-identifier": 15,
      "client-name": 25,
      "db-connection-string": 10,
      "in-aadhaar": 14,
      "in-pan": 9,
      "internal-customer-id": 9,
      "private-key-material": 10,
    });
    // Every scorable entityType the IR declares carries gold. A type with zero
    // is a type whose recall is undefined rather than zero.
    const scorable = IR.entityTypes.filter((e) => e.tier !== 2).map((e) => e.id).sort();
    expect(Object.keys(m.counts.goldSpansByType).sort()).toEqual(scorable);
    expect(Object.keys(m.counts.confusableSpansByType)).toHaveLength(24);
    for (const [t, n] of Object.entries(m.counts.confusableSpansByType)) expect([t, n >= 4]).toEqual([t, true]);
  });

  it("reports the quarantine rate over the pool it was offered, not a curated one", () => {
    expect(m.certification.carriers).toEqual({
      total: 57,
      certifiedClear: 0,
      provisionalClear: 27,
      quarantined: 30,
    });
    expect(m.certification.claim).toBe("NOT CERTIFIED");
    expect(m.adjudication.counts.adjudicated).toBe(32);
    expect(m.adjudication.counts.admitted).toBe(27);
    expect(m.adjudication.counts.refused).toBe(5);
    expect(m.adjudication.counts.quarantineRateOfAdjudicatedPool).toBeCloseTo(5 / 32, 12);
    expect(m.adjudication.counts.carriersUnadjudicated).toBe(25);
    expect(m.adjudication.counts.quarantineRateOfOfferedPool).toBeCloseTo(30 / 57, 12);
    expect(m.certification.quarantined).toHaveLength(30);
    // Every quarantine carries its reason.
    for (const q of m.certification.quarantined) {
      expect([q.carrierId, q.hits.length > 0]).toEqual([q.carrierId, true]);
    }
  });

  it("shows the adjudication changing the admitted set by exactly one carrier", () => {
    // The unflattering measurement, and the reason it is in a test rather than
    // a paragraph: over the 32 adjudicated carriers, the automated sweeps
    // already quarantine the four d0* carriers. The blind round's UNIQUE
    // contribution to this pool is hn01 -- one carrier that every sweep passes
    // and two readers both found arguable. It also, separately, withholds the
    // 25 wave-1 carriers nobody adjudicated.
    const sweptOnly = OFFERED_CARRIERS.filter(
      (c) =>
        certifyCarrier(c.id, carrierText(c), {
          ir: IR,
          supplementarySweeps: [{ id: "orthographic-org-sweep", sweep: orthographicOrgSweep }],
        }).hits.length > 0,
    ).map((c) => c.id);
    const refusedByRound = OFFERED_CARRIERS.filter((c) => !isAdmitted(c.id)).map((c) => c.id);
    expect(sweptOnly.sort()).toEqual([
      "d01-email",
      "d02-titlecase",
      "d03-kv",
      "d04-pan-shaped",
      "d05-aadhaar-digits",
      "d06-ifsc-shaped",
      "d07-entropy-fence",
    ]);
    const uniqueToRound = refusedByRound.filter((id) => !sweptOnly.includes(id));
    const adjudicatedOnly = uniqueToRound.filter((id) => adjudicationSweep("x", id)[0]!.label !== "unadjudicated");
    expect(adjudicatedOnly).toEqual(["hn01"]);
  });

  it("drops nothing for contamination, and the check is not vacuous", () => {
    expect(m.contamination.dropped).toEqual([]);
    expect(m.contamination.itemsChecked).toBe(189);
    expect(m.contamination.maxScoreKept).toBe(0);
    expect(m.contamination.unscoreable).toEqual([]);
    // Zero drops at maxScoreKept 0 means no item shares even ONE 8-gram with
    // any self-test example -- which is either a clean corpus or a check that
    // never fires. Feed it an example verbatim and it must drop.
    const examples = loadSelfTestExamples();
    const long = examples.filter((e) => e.text.split(/\s+/).length >= 12);
    expect(long.length).toBeGreaterThan(50);
    const probe = checkContamination([{ id: "planted", text: long[0]!.text }], examples);
    expect(probe.dropped.map((d) => d.itemId)).toEqual(["planted"]);
    expect(probe.dropped[0]!.score).toBe(1);
  });

  it("states, in the artifact, what is unvalidated and what is unpopulated", () => {
    const joined = m.unvalidated.join(" ");
    expect(joined).toContain("CARRIER REALISM IS UNVALIDATED");
    expect(joined).toContain("p-med AND p-corp LABELS ARE UNPOPULATED");
    expect(joined).toContain("NO pred: GOLD");
    expect(joined).toContain("NOT BLIND OF THE AUTHORING INTENT");
    expect(m.labels.populated).toEqual(["p-fin"]);
    expect(m.labels.unpopulated.map((u) => u.policy)).toEqual(["p-med", "p-corp"]);
    // And every item's own labels agree, so an item lifted out of the JSONL
    // carries the same caveat as the manifest.
    for (const item of built.items) {
      for (const l of (item.meta?.["labels"] ?? []) as { violatesUnder: Record<string, { state: string }> }[]) {
        expect(l.violatesUnder["p-fin"]!.state).toBe("populated");
        expect(l.violatesUnder["p-med"]!.state).toBe("unpopulated");
        expect(l.violatesUnder["p-corp"]!.state).toBe("unpopulated");
      }
    }
  });

  it("records the one family it excluded and why", () => {
    expect(m.inputs.excludedFamilies.map((e) => e.id)).toEqual([...EXCLUDED_FAMILY_IDS]);
    expect(m.inputs.excludedFamilies[0]!.why).toContain("okaxis");
    expect(m.inputs.positiveFamilies).not.toContain("upi-vpa");
    expect(m.inputs.positiveFamilies).toContain("upi-vpa-fictional-handle");
    // No emitted value sits in a real payment namespace as a result.
    expect(built.corpusJsonl).not.toContain("@okaxis");
  });

  it("splits dev from test, stratified and disjoint", () => {
    expect(m.splits.disjoint).toBe(true);
    expect(m.splits.dev.length + m.splits.test.length).toBe(189);
    expect(m.splits.dev).toHaveLength(39);
    const devPositives = m.splits.dev.filter((id) => built.items.find((i) => i.id === id)!.gold.length > 0);
    expect(devPositives.length).toBeGreaterThan(0);
    expect(devPositives.length).toBeLessThan(m.splits.dev.length);
  });
});
