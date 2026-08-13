import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { runTier0 } from "../../src/detect/tier0.js";
import { segmentText } from "../../src/segment/segment.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const run = (text: string) => runTier0(ir, text, segmentText(text));

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
});
