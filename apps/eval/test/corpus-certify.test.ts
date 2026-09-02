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
  type Sweep,
} from "../src/corpus/certify.js";

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
