import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { runTier0 } from "../../src/detect/tier0.js";
import { segmentText } from "../../src/segment/segment.js";
import { minimalIr } from "../fixtures/minimal-ir.js";
import type { PolicyIrInput } from "../../src/policy/types.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const run = (text: string) => runTier0(ir, text, segmentText(text));

/** The fixture IR with its rules swapped out; provenance is keyed by rule id, so it goes too. */
const irWithRules = (rules: PolicyIrInput["rules"]) =>
  loadPolicyIr(JSON.stringify({ ...minimalIr(), rules, provenance: {} }));

describe("runTier0 — regex rules", () => {
  it("finds a PAN with absolute span and provenance source", () => {
    const text = "my PAN is ABCPD1234E for tax filing";
    const [f] = run(text);
    expect(f).toBeDefined();
    expect(f!.entityType).toBe("in-pan");
    expect(f!.text).toBe("ABCPD1234E");
    expect(text.slice(f!.start, f!.end)).toBe("ABCPD1234E");
    expect(f!.source).toBe("pan-rule");
    expect(f!.tier).toBe(0);
  });

  it("applies context boost when a keyword is nearby", () => {
    const boosted = run("my PAN is ABCPD1234E for tax filing")[0]!;
    const plain = run("the code ABCPD1234E appeared in the log")[0]!;
    expect(boosted.confidence).toBeCloseTo(0.95, 5);
    expect(plain.confidence).toBeCloseTo(0.9, 5);
  });

  it("drops regex matches that fail the validator", () => {
    // Regex-shaped but 4th char X is not a PAN holder type.
    expect(run("code ABCXD1234E here")).toHaveLength(0);
  });

  it("finds an AWS key with no validator configured", () => {
    const [f] = run("creds: AKIAIOSFODNN7EXAMPLE");
    expect(f!.entityType).toBe("aws-key");
    expect(f!.severity).toBe("critical");
  });

  it("finds multiple occurrences with distinct spans", () => {
    const text = "ABCPD1234E and again ABCPD1234E";
    const findings = run(text);
    expect(findings).toHaveLength(2);
    expect(findings[0]!.start).not.toBe(findings[1]!.start);
  });

  // The loader rejects regexes that match "" at offset 0, but a lookbehind slips
  // through and still matches empty mid-string. Abandoning the rule there would
  // silently drop its real matches — a fail-open — so the scan skips and continues.
  it("skips a zero-width match and keeps scanning the rest of the text", () => {
    const lookbehindIr = irWithRules([{ id: "lookbehind-rule", entityType: "in-pan", regex: "(?<=:)\\w*" }]);
    const text = "a:! then :word"; // empty match after the first ":", real match after the second
    const findings = runTier0(lookbehindIr, text, segmentText(text));
    expect(findings.map((f) => f.text)).toEqual(["word"]);
    expect(text.slice(findings[0]!.start, findings[0]!.end)).toBe("word");
  });

  // Rules run in declaration order, so a rule declared first can match later text.
  // pan-rule is declared before aws-rule in the fixture; the AWS key comes first here.
  it("returns findings in ascending start order across rules", () => {
    const findings = run("AKIAIOSFODNN7EXAMPLE then ABCPD1234E");
    expect(findings.map((f) => f.source)).toEqual(["aws-rule", "pan-rule"]);
    expect(findings[0]!.start).toBeLessThan(findings[1]!.start);
  });
});

describe("runTier0 — context boost", () => {
  // The keyword must fit ENTIRELY inside [start-40, end+40): the window is a slice,
  // not a distance test, so a keyword straddling the edge is invisible.
  it("boosts for a keyword inside the window but not for one straddling its edge", () => {
    const inside = `tax${" ".repeat(37)}ABCPD1234E`; // window starts at 0 -> "tax" intact
    const straddling = `tax${" ".repeat(38)}ABCPD1234E`; // window starts at 1 -> only "ax"
    expect(run(inside)[0]!.confidence).toBeCloseTo(0.95, 5);
    expect(run(straddling)[0]!.confidence).toBeCloseTo(0.9, 5);
  });

  it("matches keywords case-insensitively (fixture keyword is uppercase PAN)", () => {
    expect(run("my pan is ABCPD1234E")[0]!.confidence).toBeCloseTo(0.95, 5);
  });
});

describe("runTier0 — span fidelity", () => {
  // Offsets are UTF-16 code units, matching Segment offsets. Pinned before tiers 1-2
  // land, since a model tokenizer reporting code points would silently disagree.
  it("keeps spans faithful across astral characters", () => {
    const text = "\u{1F600} PAN: ABCPD1234E";
    const [f] = run(text);
    expect(f!.text).toBe(text.slice(f!.start, f!.end));
    expect(f!.text).toBe("ABCPD1234E");
    expect(f!.start).toBe(8); // the emoji is two UTF-16 units, not one
  });
});
