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

/**
 * `scripts/compile-policies.ts` is the only path that PRODUCES those artifacts,
 * and it is imported by no test.
 *
 * The determinism claim the round rests on -- "produced offline by
 * `scripts/compile-policies.ts` replaying the committed LLM fixtures with no
 * network call" -- was verified for the compiler and not for this script: the
 * test above drives `runCli` directly with `loadTestFixtures()` and never loads
 * the file. An inert declaration inserted at the top of it survived the whole
 * compiler suite, which is what says the file is not loaded.
 *
 * Running it is not the answer: `main()` executes at module scope and WRITES
 * into `policies/compiled/`, so importing it from a test would mutate the
 * repository on every suite run. What can be checked without running it is its
 * PROVENANCE -- that its only source of model answers is the committed fixture
 * set, that it still compiles every policy document rather than a subset, and
 * that it reports the ones it cannot. Those are the three properties the claim
 * is made of, and the test above catches a drifted artifact independently.
 */
describe("the offline compile script's provenance", () => {
  const SCRIPT = join(REPO, "scripts", "compile-policies.ts");
  const source = readFileSync(SCRIPT, "utf8");

  it("takes its model answers only from the committed fixtures", () => {
    // The ONE call that decides whether a compile is offline: `runCli`'s options
    // carry either `fixtures` or a live client, and this script must pass the
    // former on every path.
    expect(source).toContain("loadTestFixtures");
    expect(source).toContain("{ fixtures }");
    // And nothing that could reach a network or a key. `record-fixtures.ts` is
    // the script that spends money; naming it in prose is fine, importing
    // anything it imports is not.
    const imports = [...source.matchAll(/^import[^;]*from\s+"([^"]+)";/gm)].map((m) => m[1]!);
    expect(imports).toEqual(["node:fs", "node:path", "node:url", "../packages/compiler/src/index.js", "../packages/compiler/test/fixtures/index.js"]);
    expect(source).not.toMatch(/\bfetch\s*\(|@anthropic-ai|process\.env/);
  });

  it("compiles every policy document in the repository, not a hand-kept subset", () => {
    // A compiled directory holding one of three policies must not look like a
    // compiled directory holding three, so the script iterates ALL of them and
    // reports the misses. A `POLICIES` list that fell behind `policies/*.md`
    // would silently stop trying the new one.
    const documents = readdirSync(POLICIES)
      .filter((f) => f.endsWith(".md"))
      .map((f) => f.slice(0, -".md".length))
      .sort();
    expect(documents.length).toBeGreaterThan(0);
    const listed = /const POLICIES = \[([^\]]*)\] as const;/.exec(source)?.[1] ?? "";
    const names = [...listed.matchAll(/"([^"]+)"/g)].map((m) => m[1]!).sort();
    expect(names).toEqual(documents);
    // And the ones it expects to succeed offline are exactly the ones with a
    // committed artifact -- the two directions the script's own exit code
    // checks, checked here against the directory rather than against itself.
    const expected = /const OFFLINE_POLICIES: readonly string\[\] = \[([^\]]*)\];/.exec(source)?.[1] ?? "";
    const offline = [...expected.matchAll(/"([^"]+)"/g)].map((m) => m[1]!).sort();
    expect(offline).toEqual(committedPolicies());
  });
});
