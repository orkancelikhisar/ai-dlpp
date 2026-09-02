import { describe, expect, it } from "vitest";
import {
  ADJUDICATION_ROUND,
  ADJUDICATION_SWEEP_ID,
  ADMITTED_CARRIER_IDS,
  BLINDNESS_CHANNELS,
  CARRIER_VERDICTS,
  adjudicationSweep,
  agreementOn,
  agreementReport,
  blindnessAudit,
  isAdmitted,
  refusalFor,
  refusalForVerdicts,
  type CarrierVerdicts,
  type CertifierVerdict,
} from "../src/corpus/adjudication.js";
import { ALL_CARRIERS, carrierText } from "../src/corpus/carriers.js";
import { CANDIDATE_CARRIERS } from "../src/corpus/carriers.candidate.js";

/**
 * The blind double-adjudication round: the record, the agreement arithmetic and
 * the admission rule it feeds.
 *
 * The record itself is transcribed data, so most of what can be checked about
 * it is internal consistency and coverage against an INDEPENDENT source -- the
 * carrier pool the round was scoped to. What can be tested properly is the
 * arithmetic and the rule, and both are exercised on synthetic inputs as well
 * as on the real table, because the real table has no disagreements and only
 * three of the four possible verdict combinations.
 */

const verdict = (clear: boolean, confidence: "clear" | "borderline"): CertifierVerdict => ({
  clear,
  confidence,
  rationale: "synthetic verdict for a rule test; not a transcription of anything",
});
const pair = (a: CertifierVerdict, b: CertifierVerdict, id = "synthetic"): CarrierVerdicts => ({
  carrierId: id,
  A: a,
  B: b,
});

describe("the recorded round", () => {
  it("covers exactly the wave-2 carrier pool, no more and no less", () => {
    // Coverage checked against carriers.candidate.ts rather than against a
    // literal list: a carrier added to the pool without being adjudicated must
    // fail here, and a verdict for a carrier that does not exist must too.
    expect(CARRIER_VERDICTS.map((v) => v.carrierId).sort()).toEqual(CANDIDATE_CARRIERS.map((c) => c.id).sort());
    expect(CARRIER_VERDICTS).toHaveLength(32);
  });

  it("carries a real rationale from both certifiers on every carrier", () => {
    // MEASURED over the transcribed table: the shortest rationale is 62
    // characters (o08, certifier A, "Choosing a talk topic for a mixed
    // audience. Nothing regulated.") and the longest is 1,123 (d07, A). The
    // floor is set under the measured minimum, so this catches a truncated or
    // dropped transcription rather than pinning a length nobody chose.
    for (const v of CARRIER_VERDICTS) {
      for (const c of ["A", "B"] as const) {
        expect([v.carrierId, c, v[c].rationale.length >= 50]).toEqual([v.carrierId, c, true]);
      }
      // The two are independent readings, not one copied into both slots.
      expect([v.carrierId, v.A.rationale === v.B.rationale]).toEqual([v.carrierId, false]);
    }
  });

  it("records the discrepancy between each certifier's prose count and their own verdicts", () => {
    // Both certifiers' notes say "28 clear, 4 not clear"; both verdict arrays
    // say 29 and 3. MEASURED off the table rather than asserted from the prose.
    const clearCount = (c: "A" | "B") => CARRIER_VERDICTS.filter((v) => v[c].clear).length;
    expect([clearCount("A"), clearCount("B")]).toEqual([29, 29]);
    const d07 = CARRIER_VERDICTS.find((v) => v.carrierId === "d07-entropy-fence")!;
    expect([d07.A.clear, d07.B.clear]).toEqual([true, true]);
    expect(ADJUDICATION_ROUND.discrepancies).toHaveLength(2);
    for (const d of ADJUDICATION_ROUND.discrepancies) expect(d).toContain("d07");
  });

  it("does not claim spec 6.2's stage 3", () => {
    expect(ADJUDICATION_ROUND.specStage.stage).toBe("frontier-adjudication");
    expect(ADJUDICATION_ROUND.specStage.satisfied).toBe(false);
    expect(ADJUDICATION_ROUND.specStage.why).toContain("union");
    // And the breach that bounds what the agreement number is worth is on the
    // record, not only in a report somewhere.
    expect(ADJUDICATION_ROUND.blindness.breaches).toHaveLength(2);
    expect(ADJUDICATION_ROUND.blindness.breaches.join(" ")).toContain("git log");
    expect(ADJUDICATION_ROUND.blindness.breaches.join(" ")).toContain("header");
  });
});

describe("agreement arithmetic", () => {
  it("computes kappa on a table with a known disagreement", () => {
    // Hand-computed. 10 carriers: 7 true/true, 1 true/false, 2 false/false.
    // A marginals 8/2, B marginals 7/3. po = 0.9.
    // pe = 0.8*0.7 + 0.2*0.3 = 0.62. kappa = 0.28/0.38 = 0.7368421052631579.
    const table: CarrierVerdicts[] = [
      ...Array.from({ length: 7 }, (_, i) => pair(verdict(true, "clear"), verdict(true, "clear"), `t${i}`)),
      pair(verdict(true, "clear"), verdict(false, "clear"), "x"),
      ...Array.from({ length: 2 }, (_, i) => pair(verdict(false, "clear"), verdict(false, "clear"), `f${i}`)),
    ];
    const got = agreementOn("clear", (v) => String(v.clear), table);
    expect(got.n).toBe(10);
    expect(got.agreements).toBe(9);
    expect(got.rawAgreement).toBe(0.9);
    expect(got.expectedAgreement).toBeCloseTo(0.62, 12);
    expect(got.cohensKappa).toBeCloseTo(0.28 / 0.38, 12);
    expect(got.marginals).toEqual({ A: { false: 2, true: 8 }, B: { false: 3, true: 7 } });
    expect(got.disagreements).toEqual(["x: A=true B=false"]);
  });

  it("returns null, not 1, when chance agreement is 1", () => {
    // Both raters used one category. po = pe = 1 and kappa is 0/0. Returning 1
    // there would report perfect agreement beyond chance for two raters who
    // never discriminated anything.
    const table = Array.from({ length: 5 }, (_, i) => pair(verdict(true, "clear"), verdict(true, "clear"), `a${i}`));
    const got = agreementOn("clear", (v) => String(v.clear), table);
    expect(got.rawAgreement).toBe(1);
    expect(got.expectedAgreement).toBe(1);
    expect(got.cohensKappa).toBeNull();
    expect(got.kappaNote).toContain("undefined");
  });

  it("reports the real round: 32 of 32 on every projection, kappa 1", () => {
    const report = agreementReport();
    for (const [name, a] of Object.entries({
      clear: report.clear,
      confidence: report.confidence,
      joint: report.joint,
    })) {
      expect([name, a.n, a.agreements, a.rawAgreement, a.cohensKappa]).toEqual([name, 32, 32, 1, 1]);
      expect([name, a.disagreements]).toEqual([name, []]);
    }
    expect(report.clear.marginals).toEqual({ A: { false: 3, true: 29 }, B: { false: 3, true: 29 } });
    expect(report.confidence.marginals).toEqual({
      A: { borderline: 3, clear: 29 },
      B: { borderline: 3, clear: 29 },
    });
    // Raw agreement of 1 on a 29/3 table is worth less than it looks, and both
    // the skew caveat and the not-blind-of-intent caveat ship with the number.
    expect(report.clear.kappaNote).toContain("skewed");
    expect(report.caveat).toContain("not blind of the authoring intent");
  });
});

describe("the admission rule", () => {
  it("admits only both-clear and neither-borderline, over all four combinations", () => {
    // The real table never produces (clear, borderline) from ONE certifier
    // only, nor a disagreement on `clear`, so these are the cases the corpus
    // cannot exercise.
    const clear = verdict(true, "clear");
    const soft = verdict(true, "borderline");
    const no = verdict(false, "clear");
    expect(refusalForVerdicts(pair(clear, clear))).toBeUndefined();
    expect(refusalForVerdicts(pair(soft, clear))?.label).toBe("borderline");
    expect(refusalForVerdicts(pair(clear, soft))?.label).toBe("borderline");
    expect(refusalForVerdicts(pair(no, clear))?.label).toBe("not-clear");
    expect(refusalForVerdicts(pair(clear, no))?.label).toBe("not-clear");
    // Not-clear outranks borderline in the LABEL and the reason still names
    // both, which is d06's actual shape.
    const both = refusalForVerdicts(pair(verdict(false, "borderline"), verdict(false, "borderline")))!;
    expect(both.label).toBe("not-clear");
    expect(both.reason).toContain("borderline");
  });

  it("refuses a carrier nobody adjudicated", () => {
    const unknown = refusalFor("c01")!;
    expect(unknown.label).toBe("unadjudicated");
    expect(unknown.reason).toContain("c01");
    // Every wave-1 carrier is in that position; the round was scoped to wave 2.
    for (const c of ALL_CARRIERS) expect([c.id, isAdmitted(c.id)]).toEqual([c.id, false]);
  });

  it("admits 27 of the 32 adjudicated carriers, and names the five it refuses", () => {
    expect(ADMITTED_CARRIER_IDS).toHaveLength(27);
    const refused = CARRIER_VERDICTS.filter((v) => !isAdmitted(v.carrierId)).map((v) => [
      v.carrierId,
      refusalFor(v.carrierId)!.label,
    ]);
    expect(refused).toEqual([
      ["hn01", "borderline"],
      ["d04-pan-shaped", "not-clear"],
      ["d05-aadhaar-digits", "not-clear"],
      ["d06-ifsc-shaped", "not-clear"],
      ["d07-entropy-fence", "borderline"],
    ]);
    // hn01 and d07 are the two carriers BOTH certifiers cleared and both
    // flagged. Losing them is the price of the borderline rule, and it is the
    // rule doing work rather than a restatement of "not clear".
    for (const id of ["hn01", "d07-entropy-fence"]) {
      const v = CARRIER_VERDICTS.find((x) => x.carrierId === id)!;
      expect([id, v.A.clear, v.B.clear]).toEqual([id, true, true]);
    }
  });
});

describe("the sweep", () => {
  const textOf = (id: string) => carrierText(CANDIDATE_CARRIERS.find((c) => c.id === id)!);

  it("returns nothing for an admitted carrier", () => {
    expect(adjudicationSweep(textOf("o01"), "o01")).toEqual([]);
  });

  it("returns one message-scoped hit for a refused carrier", () => {
    const text = textOf("hn01");
    const hits = adjudicationSweep(text, "hn01");
    expect(hits).toHaveLength(1);
    const [hit] = hits;
    // The span is the whole message because the objection was message-scoped:
    // no certifier pointed at a substring. `text` must be exactly what the
    // offsets select, the same contract every other StageHit holds.
    expect([hit!.sweep, hit!.start, hit!.end]).toEqual([ADJUDICATION_SWEEP_ID, 0, text.length]);
    expect(hit!.text).toBe(text.slice(hit!.start, hit!.end));
    expect(hit!.label).toBe("borderline");
    expect(hit!.note).toContain("borderline");
  });

  it("keys on the carrier id, not on the text", () => {
    // Feeding an admitted carrier's TEXT under a refused carrier's ID must
    // quarantine: the verdict belongs to the carrier, and re-deriving identity
    // from the text would be a second, fallible claim.
    expect(adjudicationSweep(textOf("o01"), "hn01")).toHaveLength(1);
    expect(adjudicationSweep(textOf("hn01"), "o01")).toEqual([]);
  });
});

describe("the blindness audit covers both channels, not only what was read", () => {
  /**
   * The defect: `blindness.verified` audited `filesRead` and nothing else. What
   * an annotator is TOLD is the other route to the answer, and it leaves no
   * trace in a file list -- a brief saying "four of these were written to fail"
   * produces a spotless `filesRead` and a worthless round.
   */
  const audit = blindnessAudit();

  it("names every channel, so one cannot be dropped by being unmentioned", () => {
    expect(audit.channels.map((c) => c.channel)).toEqual([...BLINDNESS_CHANNELS]);
  });

  it("gives every channel both evidence and a stated gap", () => {
    for (const channel of audit.channels) {
      expect([channel.channel, channel.what.length > 0]).toEqual([channel.channel, true]);
      expect([channel.channel, channel.evidence.length > 0]).toEqual([channel.channel, true]);
      // A channel with no gap would be a claim of perfect blindness, which no
      // round in this repository can make.
      expect([channel.channel, channel.gap.length > 0]).toEqual([channel.channel, true]);
    }
  });

  it("marks the told channel UNAUDITED rather than assuming it clean", () => {
    expect(audit.unaudited).toEqual(["told"]);
    const told = audit.channels.find((c) => c.channel === "told")!;
    expect(told.audited).toBe(false);
    expect(told.gap).toContain("NO VERBATIM BRIEF WAS RETAINED");
    expect(audit.note).toContain("told");
  });

  it("still records what the read channel was checked against", () => {
    const read = audit.channels.find((c) => c.channel === "read")!;
    expect(read.audited).toBe(true);
    expect(read.evidence).toContain("filesRead");
    // Self-reported, and said so: this repository has no independent record of
    // file access to check either list against.
    expect(read.gap).toContain("self-reported");
  });

  it("refuses to report an audit that omits a channel", () => {
    // The function derives its channels from the record; if that record ever
    // loses one, the audit must fail rather than quietly cover less.
    const round = ADJUDICATION_ROUND as unknown as { blindness: { channels: unknown[] } };
    const saved = round.blindness.channels;
    round.blindness.channels = saved.filter((c) => (c as { channel: string }).channel !== "told");
    try {
      expect(() => blindnessAudit()).toThrowError(/but not told/);
    } finally {
      round.blindness.channels = saved;
    }
    expect(blindnessAudit().channels).toHaveLength(BLINDNESS_CHANNELS.length);
  });

  it("keeps the told channel's evidence consistent with the notes it cites", () => {
    // The gap cites what each certifier's own note says. Those notes are in the
    // record; the citation is checked against them rather than trusted.
    expect(ADJUDICATION_ROUND.notes.A).toContain("Scope: the 32 wave-2 carriers");
    expect(ADJUDICATION_ROUND.notes.B).toContain("Read policies/p-fin.md and the wave-2 carrier file only");
    expect(ADJUDICATION_ROUND.notes.A).toContain("16 ordinary carriers");
    expect(ADJUDICATION_ROUND.notes.B).toContain("the d0* carriers were written to fail");
    const told = audit.channels.find((c) => c.channel === "told")!;
    expect(told.evidence).toContain("Scope: the 32 wave-2 carriers");
    expect(told.gap).toContain("the d0* carriers were written to fail");
  });
});
