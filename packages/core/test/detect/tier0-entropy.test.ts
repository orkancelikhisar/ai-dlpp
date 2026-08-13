import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "../../src/policy/load.js";
import { runTier0 } from "../../src/detect/tier0.js";
import { segmentText } from "../../src/segment/segment.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const run = (text: string) => runTier0(ir, text, segmentText(text));
const SECRET = "x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ";

describe("runTier0 — entropy rules", () => {
  it("fires on a high-entropy string inside a code fence", () => {
    const text = "here:\n```\ntoken = " + SECRET + "\n```";
    const hits = run(text).filter((f) => f.entityType === "generic-secret");
    expect(hits).toHaveLength(1);
    expect(hits[0]!.text).toBe(SECRET);
    expect(text.slice(hits[0]!.start, hits[0]!.end)).toBe(SECRET);
    expect(hits[0]!.confidence).toBe(0.7);
  });

  it("fires in kv segments", () => {
    const hits = run("SECRET_TOKEN=" + SECRET).filter((f) => f.entityType === "generic-secret");
    expect(hits).toHaveLength(1);
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
