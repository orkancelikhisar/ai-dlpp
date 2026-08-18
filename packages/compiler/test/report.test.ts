import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { renderReport, type ReportInput } from "../src/report.js";
import { CORPUS_TAG, type SelfTestReport } from "../src/stages/selftest.js";
import { minimalCompiledIr } from "./fixtures/minimal-compiled-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalCompiledIr()));

/**
 * A self-test result carrying one of each shape the renderer has to tell apart:
 * a measured entity that was shadowed, a measured entity whose numbers are
 * genuinely zero, and an entity nothing was executed against.
 */
const selfTest: SelfTestReport = {
  corpusTag: CORPUS_TAG,
  cases: [],
  warnings: [],
  entities: [
    {
      entityType: "in-pan",
      tier: 0,
      positives: 20,
      negatives: 20,
      caught: 19,
      shadowed: 1,
      falsePositives: 1,
      recall: 1,
      labelRecall: 0.95,
      fpRate: 0.05,
      skipped: false,
    },
    {
      entityType: "legacy-employee-id",
      tier: 0,
      positives: 20,
      negatives: 20,
      caught: 0,
      shadowed: 0,
      falsePositives: 0,
      recall: 0,
      labelRecall: 0,
      fpRate: 0,
      skipped: false,
    },
    {
      entityType: "client-name",
      tier: 1,
      positives: 0,
      negatives: 0,
      caught: 0,
      shadowed: 0,
      falsePositives: 0,
      skipped: true,
      skipReason: "entityType is tier 1; only tier 0 executes today",
    },
  ],
};

const input = (over: Partial<ReportInput> = {}): ReportInput => ({
  policyName: "toy",
  ir,
  proposed: { entityTypes: 5, rules: 4, semanticPredicates: 0, actions: 6 },
  grounded: { entityTypes: 5, rules: 4, semanticPredicates: 0, actions: 6 },
  rejected: [],
  selfTest,
  warnings: [],
  ...over,
});

/** The self-test table's row for one entity, scoped to that section. */
function rowFor(report: string, entityTypeId: string): string {
  const section = report.slice(report.indexOf("## Self-test coverage"));
  const row = section
    .slice(0, section.indexOf("\n## ") === -1 ? undefined : section.indexOf("\n## "))
    .split("\n")
    .find((line) => line.startsWith(`| \`${entityTypeId}\` |`));
  expect(row, `no self-test row for ${entityTypeId}`).toBeDefined();
  return row!;
}

describe("renderReport self-test coverage", () => {
  it("prints an unmeasured metric as not measured, never as 0%", () => {
    // MUTATION: `rate()` returning "0%" for undefined passes every plan test in
    // compile.test.ts. It also puts a 0% recall bar next to every tier-1 and
    // tier-2 entity on every compile until Plans 4-5 land — a false alarm that
    // fires unconditionally and teaches a reader to skim the very list the
    // genuinely weak entities are in.
    const report = renderReport(input());
    const row = rowFor(report, "client-name");
    expect(row).toContain("not measured");
    expect(row).not.toContain("%");
    expect(report).toContain("only tier 0 executes today");
  });

  it("still prints a measured zero as 0%, so the two cannot be confused", () => {
    // The complement of the test above, and the reason it cannot be satisfied
    // by rendering "not measured" everywhere: an entity that caught nothing
    // must read as caught nothing.
    const row = rowFor(renderReport(input()), "legacy-employee-id");
    expect(row).toContain("0% (0/20)");
    expect(row).not.toContain("not measured");
  });

  it("reports labelRecall beside recall, with a note when they differ", () => {
    // MUTATION: dropping the label-recall column passes every plan test. The
    // two numbers answer different questions — whether the value leaks, and
    // whether this entity's own label (its surrogate, its cited clause) is what
    // the user sees — so a report carrying only one of them is a report that
    // silently picks which question the reader gets to ask.
    const report = renderReport(input());
    expect(report).toMatch(/label recall/i);
    const row = rowFor(report, "in-pan");
    expect(row).toContain("100% (20/20)");
    expect(row).toContain("95% (19/20)");
    expect(report).toContain("caught under another entityType's label");
  });
});

describe("renderReport rejected candidates", () => {
  it("prints every rejected candidate with its id, kind and reason", () => {
    // MUTATION: rendering only the count, or nothing at all, passes every plan
    // test — p-fin's own compile rejects nothing. A compile that dropped half
    // the policy must not read like one that dropped nothing.
    const report = renderReport(
      input({
        rejected: [
          { id: "blood-type", kind: "entityType", reason: "sourceQuote not found in policy document" },
          { id: "vague", kind: "rule", reason: "sourceQuote too short to ground a rule (3 chars, minimum 24)" },
        ],
      }),
    );
    expect(report).toContain("blood-type");
    expect(report).toContain("sourceQuote not found in policy document");
    expect(report).toContain("vague");
    expect(report).toContain("minimum 24");
    expect(report).not.toMatch(/None — every candidate grounded/);
  });

  it("says so explicitly when nothing was rejected", () => {
    expect(renderReport(input())).toMatch(/None — every candidate grounded/);
  });
});

describe("renderReport outbound-visible identifiers", () => {
  it("lists every entityType id inside the marker it actually ships in", () => {
    const report = renderReport(input());
    for (const entity of ir.entityTypes) {
      expect(report).toContain(`[REDACTED:${entity.id}]`);
    }
  });
});

describe("renderReport warnings", () => {
  it("carries every warning through verbatim", () => {
    // MUTATION: truncating the list, or summarising it as a count, passes every
    // plan test (which only checks that `warnings` is an array).
    const warnings = ["first warning", "second warning", "third warning"];
    const report = renderReport(input({ warnings }));
    for (const warning of warnings) expect(report).toContain(warning);
  });
});
