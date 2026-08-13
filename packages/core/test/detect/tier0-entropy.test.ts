import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { runTier0 } from "../../src/detect/tier0.js";
import { segmentText } from "../../src/segment/segment.js";
import { minimalIr } from "../fixtures/minimal-ir.js";
import type { PolicyIrInput } from "../../src/policy/types.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const run = (text: string) => runTier0(ir, text, segmentText(text));
const SECRET = "x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ";

/** The fixture IR with its rules swapped out; provenance is keyed by rule id, so it goes too. */
const irWithRules = (rules: PolicyIrInput["rules"]) =>
  loadPolicyIr(JSON.stringify({ ...minimalIr(), rules, provenance: {} }));

describe("runTier0 — entropy rules", () => {
  it("fires on a high-entropy string inside a code fence", () => {
    const text = "here:\n```\ntoken = " + SECRET + "\n```";
    const hits = run(text).filter((f) => f.entityType === "generic-secret");
    expect(hits).toHaveLength(1);
    expect(hits[0]!.text).toBe(SECRET);
    expect(text.slice(hits[0]!.start, hits[0]!.end)).toBe(SECRET);
    expect(hits[0]!.confidence).toBe(0.7);
  });

  // DOCUMENTED QUIRK, pinned deliberately (same convention as the segmenter's
  // CRLF pins): "=" and "_" are in the secret alphabet, so a kv line is ONE
  // maximal run and the span covers key AND value, not the value alone. This is
  // load-bearing for Task 11 — the merge step resolves this span against the
  // regex findings it overlaps, so a silent narrowing here would change which
  // finding wins. Change the alphabet and this assertion must be revisited.
  it("fires in kv segments, spanning the whole key=value run", () => {
    const text = "SECRET_TOKEN=" + SECRET;
    const hits = run(text).filter((f) => f.entityType === "generic-secret");
    expect(hits).toHaveLength(1);
    expect(hits[0]!.text).toBe("SECRET_TOKEN=" + SECRET);
    expect(text.slice(hits[0]!.start, hits[0]!.end)).toBe(hits[0]!.text);
  });

  it("does NOT fire on the same string in prose", () => {
    const hits = run("I saw the string " + SECRET + " on a slide today").filter(
      (f) => f.entityType === "generic-secret",
    );
    expect(hits).toHaveLength(0);
  });

  it("ignores low-entropy and short strings in code", () => {
    const text = "```\nname = aaaaaaaaaaaaaaaaaaaaaaaaaaa\nport = x9K2mQ8v\n```";
    expect(run(text).filter((f) => f.entityType === "generic-secret")).toHaveLength(0);
  });

  // Exercises the `rule.minLength ?? DEFAULT_MIN_LENGTH` arm, which no fixture
  // rule reaches (the fixture sets minLength: 20 explicitly). Both runs are 19
  // and 20 DISTINCT characters, so both clear the 4.0 threshold on entropy
  // alone (log2(19) = 4.25) — length is the only thing separating them.
  it("defaults minLength to 20 when a rule omits it", () => {
    const localIr = irWithRules([
      { id: "no-minlength", entityType: "generic-secret", entropyThreshold: 4.0 },
    ]);
    const fire = (run19or20: string) => {
      const text = "```\nk = " + run19or20 + "\n```";
      return runTier0(localIr, text, segmentText(text));
    };
    expect(fire("aB3dE6gH9jK2mN5pQ8r")).toHaveLength(0); // 19 chars — under the default
    expect(fire("aB3dE6gH9jK2mN5pQ8rT")).toHaveLength(1); // 20 chars — at the default
  });

  // The kv/prose boundary decides whether entropy ever looks at a line, so the
  // widened KV_LINE (YAML-list keys) is pinned end-to-end, not just in the
  // segmenter: a regression there silently stops scanning these lines.
  it("fires on a secret on a YAML list line", () => {
    const text = "config:\n- api_key: " + SECRET;
    const hits = run(text).filter((f) => f.entityType === "generic-secret");
    expect(hits).toHaveLength(1);
    expect(hits[0]!.text).toBe(SECRET);
    expect(text.slice(hits[0]!.start, hits[0]!.end)).toBe(SECRET);
  });
});
