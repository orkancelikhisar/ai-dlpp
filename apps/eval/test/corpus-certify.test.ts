import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadPolicyIr, runTier0, segmentText, type PolicyIr } from "@sih/core";
import { ALL_CARRIERS, CLEAN_CARRIERS, DIRTY_CARRIERS, carrierText } from "../src/corpus/carriers.js";
import {
  CERTIFICATION_STAGES,
  MAX_RECALL_ENTROPY_THRESHOLD,
  certificationSummary,
  certifyCarrier,
  maxRecallIr,
  orthographicOrgSweep,
  tier0Sweep,
  type CarrierCertification,
  type Sweep,
} from "../src/corpus/certify.js";
import { FORMAT_SPEC_SWEEP_ID, formatSpecSweep } from "../src/corpus/format-sweep.js";

const REPO = fileURLToPath(new URL("../../..", import.meta.url));
const IR: PolicyIr = loadPolicyIr(readFileSync(`${REPO}policies/compiled/p-fin.ir.json`, "utf8"));

/**
 * A `Sweep` takes the carrier id as well as the text, for the one sweep whose
 * input is a recorded per-carrier judgement rather than the text. Every sweep
 * in this file ignores it; `PROBE` is the id these direct calls run under.
 */
const PROBE = "probe-carrier";

const CLEAR: Sweep = () => [];
const hit = (label: string): Sweep => (text) => [{ sweep: "stub", start: 0, end: 1, text: text.slice(0, 1), label }];

describe("maxRecallIr", () => {
  it("drops validators, which is the only way to widen a regex rule", () => {
    // ABCDE1234F is PAN-SHAPED and not a PAN: its 4th character D is not one of
    // the ten holder-type codes pan-structure accepts. The stock IR pairs the
    // regex with that validator; a max-recall sweep must see it anyway.
    const text = "The reference on the form reads ABCDE1234F next to the signature.";
    // Stock: runTier0 straight off the compiled IR, not through this module.
    expect(runTier0(IR, text, segmentText(text))).toEqual([]);
    expect(tier0Sweep(IR)(text, PROBE).map((h) => `${h.label}:${h.text}`)).toEqual(["in-pan:ABCDE1234F"]);
    expect(IR.rules.some((r) => r.validator !== undefined)).toBe(true);
    expect(maxRecallIr(IR).rules.every((r) => r.validator === undefined)).toBe(true);
  });

  it("takes the MINIMUM of the existing threshold and the max-recall floor", () => {
    // A policy that already scans lower than the sweep's floor must not be
    // RAISED by a pass whose name is "max recall". The stock p-fin rule is at
    // 4.0, so the default direction is exercised by the second case only; this
    // first case is the one a naive assignment would break.
    const lower: PolicyIr = {
      ...IR,
      rules: IR.rules.map((r) => (r.entropyThreshold === undefined ? r : { ...r, entropyThreshold: 2.0, minLength: 8 })),
    };
    const widened = maxRecallIr(lower).rules.find((r) => r.entropyThreshold !== undefined)!;
    expect(widened.entropyThreshold).toBe(2.0);
    expect(widened.minLength).toBe(8);

    const stockWidened = maxRecallIr(IR).rules.find((r) => r.entropyThreshold !== undefined)!;
    expect(stockWidened.entropyThreshold).toBe(MAX_RECALL_ENTROPY_THRESHOLD);
  });

  it("lowering the entropy floor really does find more in a kv block", () => {
    const kv = "settings:\nLOG_LEVEL=debug\nMODE=production_readonly\nTIMEOUT_SECONDS=30\n";
    const wide = tier0Sweep(IR)(kv, PROBE);
    expect(wide.map((h) => h.text)).toEqual([
      "LOG_LEVEL=debug",
      "MODE=production_readonly",
      "TIMEOUT_SECONDS=30",
    ]);
  });
});

describe("orthographicOrgSweep", () => {
  it("finds a Title-Case bigram", () => {
    expect(orthographicOrgSweep("we met at Ferndale Conference Centre on tuesday", PROBE).map((h) => h.text)).toEqual([
      "Ferndale Conference Centre",
    ]);
  });

  it("does not fire on a lone Title-Case word", () => {
    // The bigram requirement is the whole design: relaxing it to a unigram
    // would quarantine every sentence-initial capital, which is to say every
    // real carrier. MEASURED by mutation -- with `+` relaxed to `*` in the
    // pattern, nothing else in this suite noticed.
    expect(orthographicOrgSweep("Tuesday was fine and the report from Ferndale went out", PROBE)).toEqual([]);
    // And the blind spot, stated: a lowercase organisation name is invisible.
    expect(orthographicOrgSweep("we bank with vetiver logistics", PROBE)).toEqual([]);
  });

  it("is not stateful across calls", () => {
    // A module-level /g regex shares lastIndex; the second call would silently
    // start mid-string and return nothing.
    const text = "the venue was Ferndale Conference Centre again";
    expect(orthographicOrgSweep(text, PROBE)).toEqual(orthographicOrgSweep(text, PROBE));
    expect(orthographicOrgSweep(text, PROBE)).toHaveLength(1);
  });
});

describe("certifyCarrier", () => {
  it("never returns certified-clear with the runners this repository can supply", () => {
    for (const carrier of ALL_CARRIERS) {
      const cert = certifyCarrier(carrier.id, carrierText(carrier), { ir: IR });
      expect(cert.status).not.toBe("certified-clear");
      expect(cert.unrun).toEqual(["high-recall-model-sweep", "frontier-adjudication"]);
    }
  });

  it("DOES return certified-clear when all three stages are supplied and clear", () => {
    // The other half of the previous test. Without this one, a hardcoded
    // "provisional-clear" would pass the whole file.
    const cert = certifyCarrier("c01", carrierText(CLEAN_CARRIERS[0]!), {
      ir: IR,
      modelSweep: CLEAR,
      frontierAdjudication: CLEAR,
    });
    expect(cert.status).toBe("certified-clear");
    expect(cert.unrun).toEqual([]);
    expect(cert.stages.every((s) => s.ran)).toBe(true);
  });

  it("two of three stages is still not certified", () => {
    const cert = certifyCarrier("c01", carrierText(CLEAN_CARRIERS[0]!), { ir: IR, modelSweep: CLEAR });
    expect(cert.status).toBe("provisional-clear");
    expect(cert.unrun).toEqual(["frontier-adjudication"]);
  });

  it("a hit from any stage quarantines, even when every stage ran", () => {
    const cert = certifyCarrier("c01", carrierText(CLEAN_CARRIERS[0]!), {
      ir: IR,
      modelSweep: CLEAR,
      frontierAdjudication: hit("adjudicator-flag"),
    });
    expect(cert.status).toBe("quarantined");
    expect(cert.hits.map((h) => h.label)).toEqual(["adjudicator-flag"]);
  });

  it("a supplementary sweep can quarantine but cannot certify", () => {
    const cert = certifyCarrier("x", "we met at Ferndale Conference Centre", {
      ir: IR,
      modelSweep: CLEAR,
      frontierAdjudication: CLEAR,
    });
    expect(cert.status).toBe("quarantined");
    expect(cert.supplementary.map((s) => s.sweep)).toEqual(["orthographic-org-sweep"]);
  });

  it("attributes each dirty carrier to the sweep that caught it", () => {
    const caught = DIRTY_CARRIERS.map((c) => {
      const cert = certifyCarrier(c.id, carrierText(c), { ir: IR });
      return [c.id, cert.status, cert.hits.map((h) => `${h.sweep}/${h.label}`).join(",")];
    });
    expect(caught).toEqual([
      ["d01-email", "quarantined", "tier0-sweep/bank-account-identifier"],
      ["d02-titlecase", "quarantined", "orthographic-org-sweep/title-case-bigram"],
      [
        "d03-kv",
        "quarantined",
        "tier0-sweep/api-credential,tier0-sweep/api-credential,tier0-sweep/api-credential",
      ],
    ]);
  });

  it("every clean carrier survives stage 1 and the orthographic sweep", () => {
    for (const carrier of CLEAN_CARRIERS) {
      const cert = certifyCarrier(carrier.id, carrierText(carrier), { ir: IR });
      expect([carrier.id, cert.hits]).toEqual([carrier.id, []]);
    }
  });
});

describe("certificationSummary", () => {
  it("says NOT CERTIFIED whenever a stage did not run, however clean the carriers", () => {
    const certs = CLEAN_CARRIERS.map((c) => certifyCarrier(c.id, carrierText(c), { ir: IR }));
    const summary = certificationSummary(certs);
    expect(summary.claim).toBe("NOT CERTIFIED");
    expect(summary.carriers.quarantined).toBe(0);
    expect(summary.stagesRun).toEqual(["tier0-sweep"]);
    expect(summary.stagesUnrun.map((s) => s.stage)).toEqual([
      "high-recall-model-sweep",
      "frontier-adjudication",
    ]);
    for (const unrun of summary.stagesUnrun) {
      expect(unrun.blockedOn.length).toBeGreaterThan(20);
      expect(unrun.why.length).toBeGreaterThan(20);
    }
    expect(summary.invariantScope).toContain("stage 1 only");
  });

  it("says CERTIFIED only when all three ran and every carrier is clear", () => {
    const certs = CLEAN_CARRIERS.map((c) =>
      certifyCarrier(c.id, carrierText(c), { ir: IR, modelSweep: CLEAR, frontierAdjudication: CLEAR }),
    );
    const summary = certificationSummary(certs);
    expect(summary.claim).toBe("CERTIFIED");
    expect(summary.carriers.certifiedClear).toBe(CLEAN_CARRIERS.length);
    expect(summary.invariantScope).toContain("all three");
  });

  it("says NOT CERTIFIED when all three ran but a carrier was quarantined", () => {
    const certs = [
      ...CLEAN_CARRIERS.map((c) =>
        certifyCarrier(c.id, carrierText(c), { ir: IR, modelSweep: CLEAR, frontierAdjudication: CLEAR }),
      ),
      certifyCarrier("dirty", carrierText(DIRTY_CARRIERS[0]!), {
        ir: IR,
        modelSweep: CLEAR,
        frontierAdjudication: CLEAR,
      }),
    ];
    const summary = certificationSummary(certs);
    expect(summary.claim).toBe("NOT CERTIFIED");
    expect(summary.quarantined.map((q) => q.carrierId)).toEqual(["dirty"]);
  });

  it("cannot say CERTIFIED while a stage is unrun, even with every carrier clear", () => {
    // The conjunct `allStagesRan &&` in `claim:` had no test that could fail on
    // its removal: every existing case is also caught by the OTHER clause,
    // because `certifyCarrier` never returns certified-clear with a stage
    // unrun. So the input has to be built by hand -- which is the only way to
    // separate the two clauses, and is exactly the state a future stage-2
    // runner reporting `ran: false` would produce.
    const clearWithAStageMissing: CarrierCertification = {
      carrierId: "hand-built",
      status: "certified-clear",
      stages: [
        { sweep: "tier0-sweep", ran: true, detector: "stub", hits: [] },
        { sweep: "high-recall-model-sweep", ran: false, blockedOn: "a model", why: "no model here" },
        { sweep: "frontier-adjudication", ran: false, blockedOn: "a frontier model", why: "none here" },
      ],
      supplementary: [],
      hits: [],
      unrun: ["high-recall-model-sweep", "frontier-adjudication"],
    };
    const summary = certificationSummary([clearWithAStageMissing]);
    // Both halves: the count clause is SATISFIED here (1 of 1 certified-clear),
    // so only the stage clause can be producing this answer.
    expect(summary.carriers).toEqual({ total: 1, certifiedClear: 1, provisionalClear: 0, quarantined: 0 });
    expect(summary.claim).toBe("NOT CERTIFIED");
    expect(summary.stagesUnrun.map((s) => s.stage)).toEqual([
      "high-recall-model-sweep",
      "frontier-adjudication",
    ]);
    // And the same object with every stage reported run DOES certify, so the
    // refusal above is about the stages and not about the hand-built shape.
    const allRan: CarrierCertification = {
      ...clearWithAStageMissing,
      stages: clearWithAStageMissing.stages.map((st) => ({
        sweep: st.sweep,
        ran: true as const,
        detector: "stub",
        hits: [],
      })),
      unrun: [],
    };
    expect(certificationSummary([allRan]).claim).toBe("CERTIFIED");
  });

  it("reports the carriers stage 1 quarantined ALONE, and can produce a non-empty answer", () => {
    // `circularity.stage1Only` is the round's headline honesty number and its
    // only observed value is []. A test asserting `toEqual([])` against the
    // shipped corpus cannot tell a working filter from `filter(() => false)`.
    // These two cases drive it both ways on inputs built here.
    const tier0Only = certifyCarrier(
      "tier0-only",
      "the api key in the config was sk-liveAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA and it broke",
      { ir: IR, supplementarySweeps: [{ id: FORMAT_SPEC_SWEEP_ID, sweep: CLEAR }] },
    );
    expect(tier0Only.hits.some((h) => h.sweep === "tier0-sweep")).toBe(true);
    const only = certificationSummary([tier0Only]).circularity!;
    expect(only.stage1Only).toEqual(["tier0-only"]);
    expect(only.both).toEqual([]);
    expect(only.independentOnly).toEqual([]);

    // The mirror: a surface the IR names nowhere, so the IR-free sweep fires
    // and the widened tier-0 arm does not. Without this the test above would
    // pass on a summary that put every carrier in `stage1Only`.
    const independentOnly = certifyCarrier("format-only", "the deduction account number is MUMB12345C.", {
      ir: IR,
      supplementarySweeps: [{ id: FORMAT_SPEC_SWEEP_ID, sweep: formatSpecSweep }],
    });
    const mirrored = certificationSummary([independentOnly]).circularity!;
    expect(mirrored.independentOnly).toEqual(["format-only"]);
    expect(mirrored.stage1Only).toEqual([]);

    // And a carrier both sweeps catch lands in `both` and in neither list.
    const bothSweeps = certifyCarrier("both", "the reference on the form reads ABCDE1234F next to the signature.", {
      ir: IR,
      supplementarySweeps: [{ id: FORMAT_SPEC_SWEEP_ID, sweep: formatSpecSweep }],
    });
    const b = certificationSummary([bothSweeps]).circularity!;
    expect([b.both, b.stage1Only, b.independentOnly]).toEqual([["both"], [], []]);
  });

  it("counts exactly three spec stages, so a fourth sweep cannot inflate the count", () => {
    expect(CERTIFICATION_STAGES).toHaveLength(3);
    const certs = [
      certifyCarrier("c01", carrierText(CLEAN_CARRIERS[0]!), {
        ir: IR,
        modelSweep: CLEAR,
        frontierAdjudication: CLEAR,
        supplementarySweeps: [
          { id: "orthographic-org-sweep", sweep: orthographicOrgSweep },
          { id: "orthographic-org-sweep", sweep: CLEAR },
        ],
      }),
    ];
    expect(certificationSummary(certs).stagesRun).toHaveLength(3);
  });
});

describe("the vacuous certification: an empty pool must not certify everything", () => {
  // The defect this replaces: certificationSummary([]) returned
  // claim "CERTIFIED" with stagesRun [] and an invariantScope naming all three
  // stages. The general path derives "all stages ran" from "no stage result
  // said ran: false", and with no carrier there are no stage results at all --
  // so the most permissive answer in the module belonged to its emptiest input.
  const summary = certificationSummary([]);

  it("refuses rather than certifies", () => {
    expect(summary.claim).toBe("NOT CERTIFIED");
  });

  it("lists all three stages as unrun, blocked on having a carrier at all", () => {
    expect(summary.stagesRun).toEqual([]);
    expect(summary.stagesUnrun.map((s) => s.stage)).toEqual([...CERTIFICATION_STAGES]);
    for (const stage of summary.stagesUnrun) {
      expect([stage.stage, stage.blockedOn]).toEqual([stage.stage, "at least one carrier"]);
      // Not UNRUN_STAGES' reason: "unreachable: the tier-0 sweep is always
      // supplied" is a false sentence about a pool with no carriers in it.
      expect(stage.why).not.toContain("unreachable");
    }
  });

  it("says the invariant is established for nothing, not for a clean pool", () => {
    expect(summary.carriers).toEqual({ total: 0, certifiedClear: 0, provisionalClear: 0, quarantined: 0 });
    expect(summary.invariantScope).toContain("no carrier was submitted");
    expect(summary.invariantScope).toContain("empty one");
    expect(summary.quarantined).toEqual([]);
    expect(summary.supplementarySweepsRun).toEqual([]);
  });

  it("still certifies a non-empty pool with all three stages clear, so the guard is not a blanket refusal", () => {
    const clear: Sweep = () => [];
    const cert = certifyCarrier(PROBE, "nothing here at all", {
      ir: IR,
      modelSweep: clear,
      frontierAdjudication: clear,
      supplementarySweeps: [],
    });
    expect(certificationSummary([cert]).claim).toBe("CERTIFIED");
  });
});

describe("invariantScope names stage 1's circularity", () => {
  // The module refuses to stand @sih/tier1 in for stage 2 because tier 1 is an
  // arm under test, then runs runTier0 for stage 1 without raising it. The
  // resolution the round took: keep stage 1 -- spec 6.2 prescribes it in those
  // words -- and put the circularity in the sentence a reader quotes.
  const clean = CLEAN_CARRIERS.slice(0, 3);

  it("says the size is UNMEASURED when no IR-free sweep ran beside stage 1", () => {
    const certs = clean.map((c) => certifyCarrier(c.id, carrierText(c), { ir: IR, supplementarySweeps: [] }));
    const scope = certificationSummary(certs).invariantScope;
    expect(scope).toContain("CIRCULARITY");
    expect(scope).toContain("runTier0 over maxRecallIr(ir)");
    expect(scope).toContain("UNMEASURED");
    expect(certificationSummary(certs).circularity).toBeUndefined();
  });

  it("gives the measured split when the format-spec sweep ran beside it", () => {
    const pool = [...clean, ...DIRTY_CARRIERS];
    const certs = pool.map((c) =>
      certifyCarrier(c.id, carrierText(c), {
        ir: IR,
        supplementarySweeps: [{ id: FORMAT_SPEC_SWEEP_ID, sweep: formatSpecSweep }],
      }),
    );
    const summary = certificationSummary(certs);
    const circ = summary.circularity!;
    expect(summary.invariantScope).toContain("CIRCULARITY");
    expect(summary.invariantScope).not.toContain("UNMEASURED");
    // The numbers in the sentence are the numbers in the block, not a second
    // count that could drift from it.
    expect(summary.invariantScope).toContain(`${circ.stage1Only.length} carrier(s) quarantined by stage 1 alone`);
    expect(summary.invariantScope).toContain(`${circ.both.length} by both`);
    expect(summary.invariantScope).toContain(`${circ.independentOnly.length} by the independent sweep alone`);
    // And the split is real: the dirty carriers were caught, by both sweeps.
    expect(circ.both.length).toBeGreaterThan(0);
  });
});
