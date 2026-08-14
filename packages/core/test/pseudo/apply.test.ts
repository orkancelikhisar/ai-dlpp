import { describe, expect, it } from "vitest";
import { detect } from "../../src/detect/orchestrator.js";
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

describe("applyActions", () => {
  it("pseudonymizes and redacts by span, leaves allow untouched", async () => {
    const vault = newVault();
    const { findings } = await detect({ ir, provider: "chatgpt", text: TEXT, config });
    // The fixture's actions are block/block for these two, so the mix this test
    // is about is crafted here rather than resolved from policy.
    const manual = findings.map((f) =>
      f.entityType === "in-pan" ? { ...f, action: "pseudonymize" as const } : { ...f, action: "redact" as const },
    );
    const result = await applyActions(TEXT, manual, vault, "conv1", ir);
    expect(result.blocked).toBe(false);
    expect(result.text).not.toContain("ABCPD1234E");
    expect(result.text).not.toContain("AKIAIOSFODNN7EXAMPLE");
    expect(result.text).toContain("[REDACTED:aws-key]");
    // The surrogate is format-preserving, so the PAN shape survives.
    expect(result.text).toMatch(/PAN [A-Z]{5}[0-9]{4}[A-Z] and key/);
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
