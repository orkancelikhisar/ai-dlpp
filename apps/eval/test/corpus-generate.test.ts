import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadPolicyIr, runTier0, segmentText, type PolicyIr } from "@sih/core";
import { loadCorpus } from "../src/driver/corpus.js";
import { buildArtifacts, loadSelfTestExamples } from "../src/corpus/build.js";
import { CLEAN_CARRIERS, DIRTY_CARRIERS, carrierSlots, carrierText, slotPosition } from "../src/corpus/carriers.js";
import { CONFUSABLE_FAMILIES, POSITIVE_FAMILIES } from "../src/corpus/families.js";
import { DEFAULT_SEED, generateCorpus, serializeCorpus } from "../src/corpus/generate.js";
import { isIrBacked } from "../src/corpus/labels.js";
import { CLIENT_ORGS, FIRM, NON_CLIENT_ORGS } from "../src/corpus/universe.js";

const REPO = fileURLToPath(new URL("../../..", import.meta.url));
const IR_TEXT = readFileSync(`${REPO}policies/compiled/p-fin.ir.json`, "utf8");
const IR: PolicyIr = loadPolicyIr(IR_TEXT);
const SELFTEST = loadSelfTestExamples();

const built = buildArtifacts();
const { items, manifest } = { items: built.generated.items, manifest: built.manifest };

interface MetaInjection {
  start: number;
  end: number;
  text: string;
  type: string;
  family: string;
  dimensions: Record<string, string>;
}

function injectionsOf(item: (typeof items)[number]): MetaInjection[] {
  return (item.meta?.["injections"] ?? []) as MetaInjection[];
}

describe("the emitted corpus", () => {
  it("round-trips through the reader the harness actually uses", () => {
    expect(loadCorpus(serializeCorpus(items)).map((i) => i.id)).toEqual(items.map((i) => i.id));
  });

  it("holds every gold span at the offsets it claims", () => {
    for (const item of items) {
      for (const g of item.gold) expect([item.id, item.text.slice(g.start, g.end)]).toEqual([item.id, g.text]);
    }
  });

  it("re-derives the injection invariant from the emitted file alone", () => {
    // Independent of the generator's own assertion: this reads only what a
    // downstream consumer can read, the text and meta.injections.
    let checked = 0;
    for (const item of items) {
      for (const inj of injectionsOf(item)) {
        expect([item.id, item.text.slice(inj.start, inj.end)]).toEqual([item.id, inj.text]);
        const occurrences = item.text.split(inj.text).length - 1;
        const injectedSame = injectionsOf(item).filter((o) => o.text === inj.text).length;
        expect([item.id, occurrences]).toEqual([item.id, injectedSame]);
        checked += 1;
      }
    }
    expect(checked).toBe(manifest.injection.spansChecked);
    expect(checked).toBe(176);
  });

  it("derives gold from the injections and never from a search", () => {
    for (const item of items) {
      const irBacked = injectionsOf(item).filter((i) => isIrBacked(i.type));
      expect([item.id, item.gold.map((g) => [g.start, g.end, g.entityType])]).toEqual([
        item.id,
        irBacked.map((i) => [i.start, i.end, i.type]),
      ]);
    }
  });

  it("carries the three-policy label on every span, populated only for p-fin", () => {
    for (const item of items) {
      const labels = (item.meta?.["labels"] ?? []) as {
        type: string;
        violatesUnder: Record<string, { state: string }>;
      }[];
      expect(labels).toHaveLength(injectionsOf(item).length);
      for (const l of labels) {
        expect(Object.keys(l.violatesUnder).sort()).toEqual(["p-corp", "p-fin", "p-med"]);
        expect(l.violatesUnder["p-fin"]!.state).toBe("populated");
        expect(l.violatesUnder["p-med"]!.state).toBe("unpopulated");
        expect(l.violatesUnder["p-corp"]!.state).toBe("unpopulated");
      }
    }
  });

  it("never emits a pred: gold span, and marks the predicate labelling unrun", () => {
    for (const item of items) {
      for (const g of item.gold) expect(g.entityType.startsWith("pred:")).toBe(false);
      const hasOrg = injectionsOf(item).some((i) => i.dimensions["constructedRole"] !== "none");
      const labelling = item.meta?.["predicateLabelling"] as { state: string } | undefined;
      expect([item.id, hasOrg, labelling?.state]).toEqual([item.id, hasOrg, hasOrg ? "unlabelled" : undefined]);
    }
  });

  it("pins the IR its labels were read from, closing the gap corpus.ts names", () => {
    for (const item of items) {
      expect(item.policy).toBe("p-fin");
      expect(item.meta?.["irHash"]).toBe(manifest.policy.irHash);
    }
    expect(manifest.policy.policyHash).toBe(IR.policyHash);
  });

  it("excludes every quarantined carrier", () => {
    const carriersUsed = new Set(items.map((i) => i.meta?.["carrierId"]));
    for (const dirty of DIRTY_CARRIERS) expect(carriersUsed.has(dirty.id)).toBe(false);
    expect(carriersUsed.size).toBe(CLEAN_CARRIERS.length);
  });
});

describe("what the corpus is for: discrimination", () => {
  it("gives every tier-0 and tier-1 entityType a denominator worth measuring", () => {
    // The corpus this replaces had 2 tier-2 gold positives, so recall was
    // quantised to {0, 0.5, 1}. Every scorable entityType here has at least six.
    const scorable = IR.entityTypes.filter((e) => e.tier !== 2).map((e) => e.id);
    for (const id of scorable) {
      expect([id, manifest.counts.goldSpansByType[id] ?? 0]).toEqual([id, expect.any(Number)]);
      expect(manifest.counts.goldSpansByType[id] ?? 0).toBeGreaterThanOrEqual(6);
    }
    expect(Object.keys(manifest.counts.goldSpansByType).sort()).toEqual([...scorable].sort());
  });

  it("pairs every confusable family against a real entity family", () => {
    // A confusable that no positive family shadows measures nothing.
    expect(CONFUSABLE_FAMILIES).toHaveLength(8);
    expect(new Set(CONFUSABLE_FAMILIES.map((f) => f.type)).size).toBe(8);
    for (const family of CONFUSABLE_FAMILIES) {
      expect(manifest.counts.confusableSpansByType[family.type]).toBeGreaterThanOrEqual(6);
    }
  });

  it("puts the same organisation names in client and non-client roles", () => {
    const roles = new Set(
      items.flatMap((i) => injectionsOf(i).map((inj) => inj.dimensions["constructedRole"])),
    );
    expect([...roles].sort()).toEqual(["client", "counterparty", "none", "vendor"]);
  });

  it("varies surface, position and register match", () => {
    const dims = (key: string) =>
      new Set(items.flatMap((i) => injectionsOf(i).map((inj) => inj.dimensions[key])));
    expect([...dims("surface")].sort()).toEqual(["code-fence", "kv-line", "labelled", "prose"]);
    expect([...dims("position")].sort()).toEqual(["head", "middle", "tail"]);
    expect([...dims("registerMatch")].sort()).toEqual(["false", "true"]);
    expect([...dims("difficulty")].sort()).toEqual(["paraphrased", "verbatim"]);
  });

  it("varies density, so an item can hold a real value and a look-alike at once", () => {
    const densities = new Set(items.map((i) => i.meta?.["density"]));
    expect([...densities].sort()).toEqual([0, 1, 2]);
    const mixed = items.filter(
      (i) => injectionsOf(i).some((x) => isIrBacked(x.type)) && injectionsOf(i).some((x) => !isIrBacked(x.type)),
    );
    expect(mixed.length).toBeGreaterThan(20);
  });
});

describe("what p-fin's own tier-0 rules do to this corpus", () => {
  // Characterisation, not a target. These numbers say what the corpus asks of
  // an arm, and pinning them means a later carrier or glue edit that changes
  // the shape of the task shows up here instead of in a results table.
  const outside: string[] = [];
  const insideWrongType: string[] = [];
  for (const item of items) {
    for (const f of runTier0(IR, item.text, segmentText(item.text))) {
      const overlapping = item.gold.filter((g) => f.start < g.end && g.start < f.end);
      if (overlapping.length === 0) outside.push(`${item.id}|${f.entityType}|${f.start}`);
      else if (!overlapping.some((g) => g.entityType === f.entityType))
        insideWrongType.push(`${item.id}|${f.entityType}|${overlapping.map((g) => g.entityType).join(",")}`);
    }
  }

  it("fires outside a gold span only on an injected confusable", () => {
    // MEASURED: 11 findings, every one of them p-fin's `upi-vpa-format` rule
    // matching the local part of a neg:email-address injection. Under the
    // invariant these are true false positives, and they are the ones the
    // family was injected to provoke -- not an artifact of the glue.
    expect(outside).toHaveLength(11);
    for (const o of outside) {
      const [id] = o.split("|");
      const item = items.find((i) => i.id === id)!;
      expect([o, injectionsOf(item).map((x) => x.type)]).toEqual([o, expect.arrayContaining(["neg:email-address"])]);
      expect(o).toContain("|bank-account-identifier|");
    }
  });

  it("fires inside a gold span under a different type only on private-key bodies", () => {
    // MEASURED: 12 findings, two per private-key item -- `api-credential-entropy`
    // on each base64 body line of the PEM block. See mintPemBlock for why the
    // block is not shortened until they go away.
    expect(insideWrongType).toHaveLength(12);
    for (const o of insideWrongType) expect(o).toContain("|api-credential|private-key-material");
  });
});

describe("no gold value leaks into anything the arms are shown", () => {
  // The standing conventions record a corpus whose only tier-1 gold value was
  // also in the IR's `examples`, i.e. the answer was in the prompt.
  const values = [
    ...items.flatMap((i) => injectionsOf(i).map((inj) => inj.text)),
    ...CLIENT_ORGS,
    ...NON_CLIENT_ORGS,
    FIRM,
  ];

  it("no injected value or universe name appears in the compiled IR", () => {
    const offenders = values.filter((v) => IR_TEXT.includes(v));
    expect(offenders).toEqual([]);
  });

  it("no injected value or universe name appears in a compiler self-test example", () => {
    const corpus = SELFTEST.map((e) => e.text).join("\n");
    const offenders = values.filter((v) => corpus.includes(v));
    expect(offenders).toEqual([]);
  });

  it("and there is something to find, so the check is not vacuous", () => {
    expect(values.length).toBeGreaterThan(150);
    expect(IR_TEXT).toContain("AAAPZ1234C");
    expect(values).not.toContain("AAAPZ1234C");
  });
});

describe("splits", () => {
  it("are disjoint, exhaustive and stratified", () => {
    const dev = new Set(manifest.splits.dev);
    const test = new Set(manifest.splits.test);
    expect(dev.size + test.size).toBe(items.length);
    expect([...dev].filter((id) => test.has(id))).toEqual([]);
    expect(manifest.splits.disjoint).toBe(true);
    const isPositive = new Map(items.map((i) => [i.id, i.gold.length > 0]));
    expect([...dev].some((id) => isPositive.get(id) === true)).toBe(true);
    expect([...dev].some((id) => isPositive.get(id) === false)).toBe(true);
  });

  it("puts the requested fraction of each stratum in dev", () => {
    const positives = items.filter((i) => i.gold.length > 0).length;
    const negatives = items.length - positives;
    const dev = new Set(manifest.splits.dev);
    const devPositives = items.filter((i) => i.gold.length > 0 && dev.has(i.id)).length;
    expect(devPositives).toBe(Math.ceil(manifest.splits.devFraction * positives));
    expect(dev.size - devPositives).toBe(Math.ceil(manifest.splits.devFraction * negatives));
  });
});

describe("determinism", () => {
  const regenerate = (seed: string) =>
    serializeCorpus(
      generateCorpus({
        seed,
        ir: IR,
        irSource: "policies/compiled/p-fin.ir.json",
        irHash: manifest.policy.irHash,
        selfTestExamples: SELFTEST,
      }).items,
    );

  it("the same seed produces byte-identical output", () => {
    expect(regenerate(DEFAULT_SEED)).toBe(regenerate(DEFAULT_SEED));
    expect(regenerate(DEFAULT_SEED)).toBe(serializeCorpus(items));
  });

  it("a different seed produces different output", () => {
    // Otherwise "seeded" would be decoration over constants.
    expect(regenerate("some-other-seed")).not.toBe(regenerate(DEFAULT_SEED));
  });

  it("records the seed and the generator version in the corpus itself", () => {
    for (const item of items) {
      expect(item.meta?.["seed"]).toBe(DEFAULT_SEED);
      expect(item.meta?.["generatorVersion"]).toBe(manifest.generator.version);
    }
  });
});

describe("the manifest states fact, not intent", () => {
  it("does not claim certification", () => {
    expect(manifest.certification.claim).toBe("NOT CERTIFIED");
    expect(manifest.certification.stagesRun).toEqual(["tier0-sweep"]);
    expect(manifest.certification.carriers).toEqual({
      total: CLEAN_CARRIERS.length + DIRTY_CARRIERS.length,
      certifiedClear: 0,
      provisionalClear: CLEAN_CARRIERS.length,
      quarantined: DIRTY_CARRIERS.length,
    });
  });

  it("names every gap with a blocker", () => {
    expect(manifest.gaps.length).toBeGreaterThanOrEqual(8);
    for (const gap of manifest.gaps) {
      expect(gap.blockedOn.length).toBeGreaterThan(10);
      expect(gap.what.length).toBeGreaterThan(30);
    }
    expect(manifest.gaps.map((g) => g.id)).toContain("certification-stage-2");
    expect(manifest.gaps.map((g) => g.id)).toContain("certification-stage-3");
    expect(manifest.gaps.map((g) => g.id)).toContain("labels-p-med-p-corp");
  });

  it("reports the contamination check it ran, including what it dropped", () => {
    expect(manifest.contamination.n).toBe(8);
    expect(manifest.contamination.threshold).toBe(0.7);
    expect(manifest.contamination.itemsChecked).toBe(items.length);
    // 280 committed self-test cases plus 443 across the 12 recorded compiler
    // fixtures (one of which has an empty positives list, which is why there
    // are 24 source ids and not 25).
    expect(manifest.contamination.sources.length).toBe(24);
    expect(manifest.contamination.sources.reduce((n, s) => n + s.examples, 0)).toBe(723);
    expect(manifest.contamination.taggedSources).toEqual([{ corpusTag: "selftest-v1", examples: 280 }]);
    expect(manifest.contamination.dropped).toEqual([]);
    expect(manifest.contamination.unscoreable).toEqual([]);
    // 263 of the 723 examples are one-line fragments under 8 tokens, so the
    // check is structurally blind to 36% of what it is checking against. That
    // is reported, not hidden inside a passing zero.
    expect(manifest.contamination.examplesUnscoreable).toBe(263);
    // The check found nothing, and the reason is headroom rather than a check
    // that cannot fire: the highest score any kept item reached is reported,
    // and corpus-contamination.test.ts drops a deliberately copied item.
    // MEASURED: exactly 0. No emitted item shares a single 8-gram with any
    // self-test example, which is what a hand-authored carrier pool about
    // operational chat versus a generated identifier corpus should look like.
    expect(manifest.contamination.maxScoreKept).toBe(0);
  });

  it("counts what it emitted", () => {
    expect(manifest.counts.items).toBe(items.length);
    expect(manifest.counts.positives).toBe(items.filter((i) => i.gold.length > 0).length);
    expect(manifest.counts.negatives).toBe(items.filter((i) => i.gold.length === 0).length);
    expect(manifest.counts.goldSpans).toBe(items.reduce((n, i) => n + i.gold.length, 0));
  });

  it("says its carriers are hand-authored", () => {
    expect(manifest.injection.carrierSource).toBe("hand-authored");
    for (const item of items) expect(item.meta?.["carrierSource"]).toBe("hand-authored");
  });
});

describe("carriers", () => {
  it("derive slots from segment lengths rather than typed offsets", () => {
    for (const carrier of CLEAN_CARRIERS) {
      const slots = carrierSlots(carrier);
      expect(slots).toHaveLength(carrier.segments.length);
      expect(slots.at(-1)).toBe(carrierText(carrier).length);
      expect([...slots].sort((a, b) => a - b)).toEqual(slots);
    }
  });

  it("label slot positions by thirds", () => {
    expect([0, 1, 2].map((i) => slotPosition(i, 3))).toEqual(["head", "middle", "tail"]);
  });

  it("offer every family a distinct value generator", () => {
    expect(new Set([...POSITIVE_FAMILIES, ...CONFUSABLE_FAMILIES].map((f) => f.id)).size).toBe(24);
  });
});
