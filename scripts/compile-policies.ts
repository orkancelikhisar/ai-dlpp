/**
 * compile-policies.ts — the OFFLINE compile, replaying committed model responses.
 *
 *   pnpm -C packages/compiler exec vite-node ../../scripts/compile-policies.ts
 *
 * Sibling of `record-fixtures.ts` and deliberately its opposite: that script is
 * the only thing here that spends money, and this one is the only thing that can
 * produce a compiled artifact without a network call. It drives the SAME
 * `runCli` through the SAME `compilePolicy`, with `FixtureLlmClient` answering
 * from `packages/compiler/test/fixtures/llm/` instead of the API.
 *
 * ## What that makes the artifacts, stated so nobody has to guess
 *
 * `policies/compiled/p-fin.*` is real compiler output: extraction, grounding,
 * predicate minting, regex validation, the self-test against the real tier-0
 * runtime, and emission all ran. What it is NOT is a live frontier-model
 * compile. The two model-driven stages (`Extraction`, `SelfTestCases`) were
 * answered from committed fixtures, and `record-fixtures.ts`'s own docblock says
 * what those are: hand-authored stand-ins for a live pass that Plan 5 defers.
 * So read the IR as "the compiler's output given those responses" — every
 * clause quote, every regex, every action and every warning in the report is the
 * compiler's work, and the entity vocabulary behind them is not a frontier
 * model's.
 *
 * Re-running this after `record-fixtures.ts` recompiles the same documents
 * against the recorded LIVE responses, which is the intended upgrade path and
 * needs no change here.
 *
 * ## Determinism
 *
 * MEASURED: three consecutive runs of `p-fin` produced byte-identical
 * `p-fin.ir.json`, `p-fin.report.md` and `p-fin.selftest.json` (md5 compared).
 * That is what makes the output committable and what
 * `packages/compiler/test/compiled.test.ts` asserts on every suite run — a
 * committed artifact that no longer reproduces is either a hand-edit or a
 * compiler change, and both must be visible.
 *
 * ## Which policies compile today
 *
 * ONLY `p-fin`. The committed fixture set answers p-fin's extraction and
 * self-test prompts and nothing else: MEASURED, `p-med` and `p-corp` both exit 1
 * on a fixture miss, because `requestHash` keys a fixture by the prompt and
 * those prompts carry a different document. This script therefore compiles the
 * policies it can and REPORTS the ones it cannot rather than skipping them
 * quietly — a compiled directory holding one of three policies must not look
 * like a compiled directory holding three.
 *
 * `OFFLINE_POLICIES` below states which ones are expected to land, and the exit
 * code checks that expectation IN BOTH DIRECTIONS. An expected policy that fails
 * is a regression; an unexpected one that SUCCEEDS means the fixture set grew
 * and this constant went stale, which would otherwise be a silent pass. A script
 * whose exit code can only ever be 1 carries no information, which is what
 * "non-zero because two policies need a live pass" would amount to.
 */
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { runCli } from "../packages/compiler/src/index.js";
import { loadTestFixtures } from "../packages/compiler/test/fixtures/index.js";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..");
const POLICY_DIR = join(REPO, "policies");
const COMPILED_DIR = join(POLICY_DIR, "compiled");

/**
 * Every policy document in the repository, not only the one that compiles.
 *
 * Listing all three is the point: a run prints one line per policy, so the two
 * that need a live pass are named on every run rather than being absent from a
 * list nobody re-derives.
 */
const POLICIES = ["p-fin", "p-med", "p-corp"] as const;

/**
 * The policies the committed fixture set can answer, as of the fixtures on disk.
 *
 * Deliberately NOT exported and NOT read by any test. The suite's check is
 * `packages/compiler/test/compiled.test.ts`, which walks `policies/compiled/`
 * and requires every artifact it finds to reproduce byte for byte — a stronger
 * statement than agreeing with a list, and one that keeps working when this
 * constant changes. This is the script's own expectation about its own run.
 */
const OFFLINE_POLICIES: readonly string[] = ["p-fin"];

async function main(): Promise<number> {
  mkdirSync(COMPILED_DIR, { recursive: true });
  // Loaded ONCE and shared, so every policy is offered exactly the same fixture
  // set and a miss is a statement about the prompt rather than about which
  // files happened to be readable on that pass.
  const fixtures = loadTestFixtures();

  const compiled: string[] = [];
  const missing: string[] = [];
  for (const name of POLICIES) {
    const code = await runCli(
      [
        "--policy",
        join(POLICY_DIR, `${name}.md`),
        "--providers",
        join(POLICY_DIR, "providers.json"),
        "--out",
        COMPILED_DIR,
        "--name",
        name,
      ],
      { fixtures },
    );
    if (code === 0) compiled.push(name);
    else missing.push(name);
  }

  console.log(`\ncompiled offline into ${COMPILED_DIR}: ${compiled.join(", ") || "none"}`);
  if (missing.length > 0) {
    console.error(
      `no committed fixture answers the compile prompts for: ${missing.join(", ")}.\n` +
        "These need a live pass (scripts/record-fixtures.ts --yes, which spends money),\n" +
        "which Plan 5 defers. Nothing was written for them.",
    );
  }

  const expected = [...OFFLINE_POLICIES].sort().join(", ");
  const actual = [...compiled].sort().join(", ");
  if (expected !== actual) {
    console.error(
      `\nOFFLINE_POLICIES says [${expected}] compiles offline and this run compiled ` +
        `[${actual}]. If more compiled than expected, the fixture set grew and that ` +
        `constant is stale; if fewer, a policy that used to compile no longer does.`,
    );
    return 1;
  }
  console.log(
    "\nA compiled IR is a security artifact: read each *.report.md by hand — the\n" +
      "Outbound-visible identifiers and Rejected candidates sections especially —\n" +
      "before committing.",
  );
  return 0;
}

process.exitCode = await main();
