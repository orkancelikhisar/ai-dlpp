import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = join(import.meta.dirname, "..", "..", "..", "policies");
const read = (f: string) => readFileSync(join(ROOT, f), "utf8");

describe("policy suite", () => {
  const policies = { "p-fin.md": read("p-fin.md"), "p-med.md": read("p-med.md"), "p-corp.md": read("p-corp.md") };

  it("each policy is substantial prose, not a stub", () => {
    for (const [name, text] of Object.entries(policies)) {
      expect(text.length, name).toBeGreaterThan(1500);
    }
  });

  it("each policy carries numbered clauses the compiler can quote", () => {
    for (const [name, text] of Object.entries(policies)) {
      expect(text, name).toMatch(/§\s?\d/);
    }
  });

  it("the deliberate disagreements are present", () => {
    // Client names: forbidden under P-FIN, explicitly permitted under P-MED.
    expect(policies["p-fin.md"]).toMatch(/client/i);
    expect(policies["p-med.md"]).toMatch(/client organisation names|client names/i);
    // Provider-specific clauses exist ONLY in P-FIN.
    expect(policies["p-fin.md"]).toMatch(/foreign-hosted|non-enterprise/i);
    expect(policies["p-med.md"]).not.toMatch(/foreign-hosted/i);
    expect(policies["p-corp.md"]).not.toMatch(/foreign-hosted/i);
    // Salary figures are a P-CORP-only concern.
    expect(policies["p-corp.md"]).toMatch(/salary/i);
    expect(policies["p-med.md"]).not.toMatch(/salary/i);
  });

  it("provider manifest maps adapter ids with aliases", () => {
    const manifest = JSON.parse(read("providers.json")) as {
      providers: Array<{ id: string; displayName: string; aliases: string[] }>;
    };
    const ids = manifest.providers.map((p) => p.id);
    expect(ids).toEqual(expect.arrayContaining(["chatgpt", "claude", "gemini", "deepseek"]));
    for (const p of manifest.providers) {
      expect(p.aliases.length, p.id).toBeGreaterThan(0);
    }
  });
});
