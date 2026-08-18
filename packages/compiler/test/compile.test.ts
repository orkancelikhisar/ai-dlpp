import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr, resolveAction } from "@sih/core";
import { compilePolicy } from "../src/compile.js";
import { FixtureLlmClient } from "../src/llm/fixture.js";
import { loadTestFixtures } from "./fixtures/index.js";

const POLICIES = join(import.meta.dirname, "..", "..", "..", "policies");

describe("compilePolicy", () => {
  const run = () =>
    compilePolicy({
      client: new FixtureLlmClient(loadTestFixtures()),
      document: readFileSync(join(POLICIES, "p-fin.md"), "utf8"),
      manifest: JSON.parse(readFileSync(join(POLICIES, "providers.json"), "utf8")),
      policyName: "p-fin",
    });

  it("emits an IR the runtime loader accepts", async () => {
    const { ir } = await run();
    expect(() => loadPolicyIr(JSON.stringify(ir))).not.toThrow();
  });

  it("stamps the IR with a hash of the source document", async () => {
    const { ir } = await run();
    expect(ir.policyHash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("carries provenance for every rule, entity and predicate", async () => {
    const { ir } = await run();
    for (const rule of ir.rules) {
      expect(ir.provenance[rule.id], rule.id).toBeDefined();
      expect(ir.provenance[rule.id]!.quote.length).toBeGreaterThan(0);
    }
  });

  it("resolves the provider clause end-to-end", async () => {
    const { ir } = await run();
    const loaded = loadPolicyIr(JSON.stringify(ir));
    // P-FIN §5: customer data may not go to non-enterprise or foreign-hosted services.
    expect(resolveAction(loaded, "client-name", "deepseek")).not.toBe(
      resolveAction(loaded, "client-name", "claude"),
    );
  });

  it("produces a markdown report listing outbound-visible ids for audit", async () => {
    const { report } = await run();
    expect(report).toMatch(/# Compilation report/i);
    expect(report).toMatch(/outbound-visible/i);
    for (const id of (await run()).ir.entityTypes.map((e) => e.id)) {
      expect(report).toContain(id);
    }
  });

  it("succeeds with warnings rather than failing on a weak entity", async () => {
    const { warnings, ok } = await run();
    expect(ok).toBe(true);
    expect(Array.isArray(warnings)).toBe(true);
  });

  it("fails hard when a rejected candidate leaves an entityType actionless", async () => {
    // Grounding drops a candidate → schema would reject the IR. The compiler must
    // catch that itself with a legible message, not emit an IR the loader rejects.
    await expect(
      compilePolicy({
        client: new FixtureLlmClient(loadTestFixtures()),
        document: "# Empty\n\n§1 Nothing to see here.",
        manifest: JSON.parse(readFileSync(join(POLICIES, "providers.json"), "utf8")),
        policyName: "empty",
      }),
    ).rejects.toThrow(/no entityTypes|actionless/i);
  });
});

/**
 * Appended beyond the plan's seven, each verified by mutation to survive them.
 * The plan's set proves the pipeline runs end to end and that the IR loads;
 * these pin the parts an auditor (or Task 9's committed artifacts) depends on
 * and that nothing else watches.
 */
describe("compilePolicy (properties the plan's seven do not pin)", () => {
  const document = readFileSync(join(POLICIES, "p-fin.md"), "utf8");
  const run = () =>
    compilePolicy({
      client: new FixtureLlmClient(loadTestFixtures()),
      document,
      manifest: JSON.parse(readFileSync(join(POLICIES, "providers.json"), "utf8")),
      policyName: "p-fin",
    });

  it("stamps the hash of THIS document, not merely something hex-shaped", async () => {
    // MUTATION: hashing the policy name, or a constant, passes the plan's
    // /^[0-9a-f]{64}$/ assertion while breaking the one thing the stamp is for.
    const { ir } = await run();
    expect(ir.policyHash).toBe(createHash("sha256").update(document, "utf8").digest("hex"));
  });

  it("traces every entityType and predicate to a numbered clause, not just rules", async () => {
    // The plan checks provenance for rules only. An entityType id is the label
    // a user sees on a redaction and the id that ships outbound, so it is the
    // one an auditor is most likely to want the sentence for.
    const { ir } = await run();
    for (const entity of ir.entityTypes) {
      const provenance = ir.provenance[entity.id];
      expect(provenance, entity.id).toBeDefined();
      expect(provenance!.clause, entity.id).toMatch(/^§[0-9]+(?:\.[0-9]+)*$/);
      expect(document).toContain(provenance!.quote);
    }
    for (const predicate of ir.semanticPredicates) {
      expect(ir.provenance[predicate.id], predicate.id).toBeDefined();
    }
    expect(ir.provenance["in-pan"]!.clause).toBe("§2.1");
  });

  it("keeps the model's sourceQuote out of the shipped IR", async () => {
    // MUTATION: spreading candidates into the IR passes every plan test, since
    // the loader strips unknown nested keys — and writes a second copy of every
    // policy quote into an artifact Task 9 commits.
    const { ir } = await run();
    expect(JSON.stringify(ir)).not.toContain("sourceQuote");
  });

  it("returns the self-test corpus that its coverage numbers were measured from", async () => {
    // MUTATION: returning `[]` passes every plan test. The corpus is the
    // evidence behind the coverage claim; without it the claim cannot be
    // re-executed by anyone auditing the compile.
    const { ir, selfTestCases } = await run();
    expect(selfTestCases.length).toBeGreaterThan(0);
    const ids = new Set(ir.entityTypes.map((e) => e.id));
    for (const testCase of selfTestCases) {
      expect(ids.has(testCase.entityType), testCase.entityType).toBe(true);
      expect(testCase.corpusTag).toMatch(/selftest/);
    }
  });

  it("carries every collected warning into the report", async () => {
    // MUTATION: rendering the warnings only into the returned array, or only
    // into the report, passes every plan test — which asserts that `warnings`
    // is an array and never that the two agree. The report is what a human
    // reads before shipping.
    const { report, warnings } = await run();
    expect(warnings.length).toBeGreaterThan(0);
    for (const warning of warnings) expect(report).toContain(warning);
  });

  it("produces byte-identical output when nothing changed", async () => {
    // Task 9 commits an IR and a report beside each policy. A clock, a counter
    // or an unordered iteration anywhere in emit or report would make every
    // recompile look like a policy edit, and a real edit unreadable in the diff.
    // MUTATION: stamping `new Date().toISOString()` into the report header
    // passes every other test in this file.
    const [first, second] = await Promise.all([run(), run()]);
    expect(JSON.stringify(second!.ir)).toBe(JSON.stringify(first!.ir));
    expect(second!.report).toBe(first!.report);
    expect(second!.warnings).toEqual(first!.warnings);
  });

  it("reports an unmeasurable entity as not measured rather than as zero", async () => {
    // End-to-end wiring of the rule the self-test stage states and the report
    // renders: `client-name` is tier 1 and `pred:...` is tier 2, so no case was
    // executed against either. Scoring them 0% would put a false alarm beside
    // every semantic entity on every compile until Plans 4-5 land.
    const { report } = await run();
    const coverage = report.slice(report.indexOf("## Self-test coverage"));
    const table = coverage.slice(0, coverage.indexOf("\n## "));
    for (const line of table.split("\n")) {
      if (line.startsWith("| `client-name` |") || line.startsWith("| `pred:")) {
        expect(line).toContain("not measured");
        expect(line).not.toContain("%");
      }
    }
    expect(report).toMatch(/tier 1 engine arrives in Plan 4/);
  });
});
