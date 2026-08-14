import { describe, expect, it } from "vitest";
import type { ResolvedFinding } from "../../src/detect/types.js";
import { detect } from "../../src/detect/orchestrator.js";
import type { Action } from "../../src/policy/types.js";
import { loadPolicyIr } from "../../src/policy/load.js";
import { applyActions } from "../../src/pseudo/apply.js";
import { MemoryVaultStore, Vault } from "../../src/pseudo/vault.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const config = { tier0: true, tier1: false, tier2: false };

/** Any fixed string; the vault only requires it to be non-empty (Task 4, D1). */
const SALT = "test-salt";
const newVault = () => new Vault(new MemoryVaultStore(), SALT);

const TEXT = "PAN ABCPD1234E and key AKIAIOSFODNN7EXAMPLE here";

/**
 * Hand-built resolved finding. `text` is re-derived from the source string
 * rather than passed in, which is the Finding contract (offsets are the truth)
 * and keeps these fixtures from drifting when a literal is edited.
 */
function findingAt(
  source: string,
  start: number,
  end: number,
  action: Action,
  entityType = "aws-key",
): ResolvedFinding {
  return {
    start,
    end,
    text: source.slice(start, end),
    entityType,
    severity: "critical",
    tier: 0,
    source: "hand-built",
    confidence: 1,
    action,
  };
}

describe("applyActions", () => {
  it("pseudonymizes and redacts by span, leaves allow untouched", async () => {
    const vault = newVault();
    // `client-name` is a tier-1 entity and tier 1 is off here, so the allow
    // finding is appended by hand -- which is also the only way to exercise the
    // branch: the fixture maps client-name to pseudonymize.
    const text = `${TEXT} for Globex`;
    const clientStart = text.indexOf("Globex");
    const { findings } = await detect({ ir, provider: "chatgpt", text, config });
    // The fixture's actions are block/block for the detected two, so the mix
    // this test is about is crafted here rather than resolved from policy.
    const manual: ResolvedFinding[] = [
      ...findings.map((f) =>
        f.entityType === "in-pan" ? { ...f, action: "pseudonymize" as const } : { ...f, action: "redact" as const },
      ),
      findingAt(text, clientStart, clientStart + "Globex".length, "allow", "client-name"),
    ];
    const result = await applyActions(text, manual, vault, "conv1", ir);
    expect(result.blocked).toBe(false);
    expect(result.text).not.toContain("ABCPD1234E");
    expect(result.text).not.toContain("AKIAIOSFODNN7EXAMPLE");
    expect(result.text).toContain("[REDACTED:aws-key]");
    // The surrogate is format-preserving, so the PAN shape survives.
    expect(result.text).toMatch(/PAN [A-Z]{5}[0-9]{4}[A-Z] and key/);
    // Allow is a rewrite of nothing: the span survives verbatim, and it must not
    // appear in `applied` either -- a caller diffing that list would otherwise
    // report a replacement that never happened (and one whose newStart/newEnd
    // would be meaningless).
    expect(result.text).toContain(" for Globex");
    expect(result.text.endsWith(" for Globex")).toBe(true);
    expect(result.applied.map((a) => a.entityType)).not.toContain("client-name");
    expect(result.applied).toHaveLength(2);
  });

  it("applied records slice the rewritten text exactly", async () => {
    const vault = newVault();
    const { findings } = await detect({ ir, provider: "chatgpt", text: TEXT, config });
    const manual = findings.map((f) =>
      f.entityType === "in-pan" ? { ...f, action: "pseudonymize" as const } : { ...f, action: "redact" as const },
    );
    const result = await applyActions(TEXT, manual, vault, "conv1", ir);
    expect(result.applied).toHaveLength(2);
    for (const a of result.applied) {
      expect(result.text.slice(a.newStart, a.newEnd)).toBe(a.replacement);
      expect(TEXT.slice(a.start, a.end)).not.toBe(a.replacement);
    }
  });

  it("sets blocked when any finding's action is block, and still rewrites the rest", async () => {
    const vault = newVault();
    const { findings } = await detect({ ir, provider: "chatgpt", text: TEXT, config });
    // Fixture actions: in-pan -> block, aws-key -> block. Soften one to redact.
    const mixed = findings.map((f) => (f.entityType === "aws-key" ? { ...f, action: "redact" as const } : f));
    const result = await applyActions(TEXT, mixed, vault, "conv1", ir);
    expect(result.blocked).toBe(true);
    expect(result.text).toContain("ABCPD1234E"); // block does not rewrite
    expect(result.text).toContain("[REDACTED:aws-key]");
  });

  // The forward assembly is a cursor walk, and every off-by-one in it hides at
  // the edges: a mid-string finding surrounded by slack passes even if the gap
  // copy is wrong by a character. These three pin the edges themselves.
  describe("span boundaries", () => {
    it("handles adjacent findings with no gap or overlap between them", async () => {
      const text = "AAAABBBB tail";
      const findings = [
        findingAt(text, 0, 4, "redact", "aws-key"),
        findingAt(text, 4, 8, "redact", "generic-secret"),
      ];
      const result = await applyActions(text, findings, newVault(), "conv1", ir);
      expect(result.text).toBe("[REDACTED:aws-key][REDACTED:generic-secret] tail");
      // The zero-length gap between them stays zero-length: no dropped or
      // duplicated character where the two replacements meet.
      expect(result.applied[0]!.newEnd).toBe(result.applied[1]!.newStart);
      for (const a of result.applied) {
        expect(result.text.slice(a.newStart, a.newEnd)).toBe(a.replacement);
      }
    });

    it("handles a finding that starts at offset 0", async () => {
      const text = "AAAA tail";
      const result = await applyActions(text, [findingAt(text, 0, 4, "redact")], newVault(), "conv1", ir);
      expect(result.text).toBe("[REDACTED:aws-key] tail");
      expect(result.applied[0]!.newStart).toBe(0);
    });

    it("handles a finding that ends at text.length", async () => {
      const text = "tail AAAA";
      const result = await applyActions(
        text,
        [findingAt(text, 5, text.length, "redact")],
        newVault(),
        "conv1",
        ir,
      );
      expect(result.text).toBe("tail [REDACTED:aws-key]");
      // Nothing is appended after the last span, so the replacement has to run
      // to the very end of the output.
      expect(result.applied[0]!.newEnd).toBe(result.text.length);
    });
  });

  it("is the identity on empty findings", async () => {
    const result = await applyActions("hello world", [], newVault(), "conv1", ir);
    expect(result).toEqual({ text: "hello world", blocked: false, applied: [] });
  });

  it("keeps referential integrity across messages in one conversation", async () => {
    const vault = newVault();
    const mk = async (text: string) => {
      const { findings } = await detect({ ir, provider: "chatgpt", text, config });
      const manual = findings.map((f) => ({ ...f, action: "pseudonymize" as const }));
      return applyActions(text, manual, vault, "convX", ir);
    };
    const r1 = await mk("first mention ABCPD1234E ok");
    const r2 = await mk("second mention ABCPD1234E ok");
    const fake1 = r1.applied[0]!.replacement;
    expect(r2.applied[0]!.replacement).toBe(fake1);
  });

  // A failure inside apply means the text was NOT fully rewritten, so nothing
  // here may be caught and continued: a swallowed mint leaves the real value
  // verbatim inside text the caller believes was rewritten, and the caller's
  // failMode is about DETECTION failures, not this one.
  it("propagates a vault refusal instead of returning half-rewritten text", async () => {
    const vault = newVault();
    const text = "key AKIAIOSFODNN7EXAMPLE here";
    const { findings } = await detect({ ir, provider: "chatgpt", text, config });
    // Hostile/buggy caller: `aws-key` is neverPseudonymize, so neither the schema
    // nor the orchestrator can produce this action for it. Constructed by hand to
    // reach the vault's runtime backstop directly.
    const forged = findings.map((f) => ({ ...f, action: "pseudonymize" as const }));
    expect(forged.some((f) => f.entityType === "aws-key")).toBe(true);
    await expect(applyActions(text, forged, vault, "conv1", ir)).rejects.toThrow(/never be pseudonymized/);
  });
});
