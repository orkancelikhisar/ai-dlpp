import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { runCli } from "../src/cli.js";
import { loadTestFixtures } from "./fixtures/index.js";

const REPO = join(import.meta.dirname, "..", "..", "..");
const POLICIES = join(REPO, "policies");
const COMPILED = join(POLICIES, "compiled");

/**
 * Every committed compiled artifact, by policy name.
 *
 * Read off the DIRECTORY rather than from a list, so adding a policy to
 * `policies/compiled/` puts it under this test without anyone remembering to,
 * and deleting one is caught by the emptiness check below rather than by the
 * loop silently having nothing to do.
 */
function committedPolicies(): string[] {
  if (!existsSync(COMPILED)) return [];
  return readdirSync(COMPILED)
    .filter((f) => f.endsWith(".ir.json"))
    .map((f) => f.slice(0, -".ir.json".length))
    .sort();
}

/**
 * The compiled artifacts under `policies/compiled/` are real compiler output and
 * this is what says so.
 *
 * WHY THIS TEST EXISTS. `apps/eval` runs Approach B against
 * `policies/compiled/p-fin.ir.json` paired with `policies/p-fin.md`, and
 * `planBakeoff` refuses that pairing unless `ir.policyHash` is the sha256 of the
 * document. So the artifact is not documentation: it is the thing that makes a
 * compiler-versus-prompting head-to-head runnable at all, and a hand-edited one
 * would rig that comparison invisibly. Recompiling it here from the same
 * committed LLM fixtures and requiring byte equality is what makes "compiled
 * output" a checkable claim rather than a filename.
 *
 * WHAT IT DOES NOT CLAIM. These fixtures are hand-authored stand-ins for a live
 * frontier-model pass (see `scripts/record-fixtures.ts`), so the entity
 * vocabulary behind the IR is not a frontier model's. Every stage of the
 * compiler did run over it; nothing here says the model responses were live.
 * `scripts/compile-policies.ts` regenerates the directory.
 */
describe("policies/compiled", () => {
  const names = committedPolicies();

  it("holds at least one compiled artifact", () => {
    // Without this the loop below vacuously passes on an empty directory, and
    // the failure mode is not academic: `apps/eval/src/page/main.ts` imports
    // p-fin.ir.json, so a missing artifact breaks the eval harness at build
    // time while this file reported green.
    expect(names.length).toBeGreaterThan(0);
  });

  for (const name of names) {
    describe(name, () => {
      const irPath = join(COMPILED, `${name}.ir.json`);
      const irText = readFileSync(irPath, "utf8");

      it("reproduces byte for byte from the committed LLM fixtures", async () => {
        const out = mkdtempSync(join(tmpdir(), `sih-compiled-${name}-`));
        const code = await runCli(
          [
            "--policy",
            join(POLICIES, `${name}.md`),
            "--providers",
            join(POLICIES, "providers.json"),
            "--out",
            out,
            "--name",
            name,
          ],
          // Warnings are silenced rather than printed: this compile emits nine
          // of them and they are the report's subject, not this test's.
          { fixtures: loadTestFixtures(), write: () => {}, writeError: () => {} },
        );
        expect(code).toBe(0);
        // All three artifacts, not only the IR. The report is what a human
        // audits before an IR ships and the self-test cases are the evidence
        // behind its coverage table, so an artifact that drifted from either
        // would be an IR whose audit describes a different compile.
        for (const suffix of ["ir.json", "report.md", "selftest.json"]) {
          expect(
            readFileSync(join(out, `${name}.${suffix}`), "utf8"),
            `${name}.${suffix} does not reproduce`,
          ).toBe(readFileSync(join(COMPILED, `${name}.${suffix}`), "utf8"));
        }
      });

      it("parses as a PolicyIr", () => {
        expect(() => loadPolicyIr(irText)).not.toThrow();
      });

      it("carries the sha256 of its own policy document as policyHash", () => {
        // The equality `planBakeoff` enforces before it will pair an Approach-B
        // arm with a compiled one. Checked here as well as there because there
        // it is a refusal at run time and here it is a property of the
        // committed file: an artifact that fails this can never be paired with
        // the document it was compiled from, and nothing else would say why.
        const document = readFileSync(join(POLICIES, `${name}.md`), "utf8");
        expect(loadPolicyIr(irText).policyHash).toBe(
          createHash("sha256").update(document, "utf8").digest("hex"),
        );
      });
    });
  }
});
