/**
 * record-fixtures.ts — THE ONLY THING IN THIS REPOSITORY THAT SPENDS MONEY.
 *
 * Compiles all three policy documents against the real Claude API, recording
 * every model response as a committed fixture so that from then on the whole
 * test suite replays those responses offline and never touches the network.
 *
 * ## Running it
 *
 *   pnpm -C packages/compiler exec vite-node ../../scripts/record-fixtures.ts --yes
 *
 * (The plan specifies `tsx`; this tree has no tsx and installing one needs the
 * network, while `vite-node` already ships as part of vitest and resolves the
 * repo's `.js`-suffixed TypeScript imports identically. Node's own type
 * stripping cannot run this file: it resolves `./x.js` literally and every
 * source file in this repo imports that way.)
 *
 * ## What it needs
 *
 * A credential. `ANTHROPIC_API_KEY` in the environment, or an `ant auth login`
 * profile — the SDK resolves either, and a bare `new AnthropicLlmClient()`
 * throws immediately if it finds neither. That check runs before any policy is
 * read, so a run without credentials costs nothing and changes nothing.
 *
 * ## What it costs, and what it destroys
 *
 * Three policy documents x (one extraction call + one self-test call per
 * entityType) of Claude Opus 5 at effort "high". Budget for real money and
 * several minutes; there is no dry-run mode that exercises the same path,
 * because the whole point of the path is that it is live.
 *
 * It is also DESTRUCTIVE to the committed test basis. The fixture key is a hash
 * of the prompt, so re-recording p-fin writes over the hand-authored
 * `Extraction.*.json` and `SelfTestCases.*.json` files that Tasks 3, 7 and 8
 * committed — the ones every current assertion about entity ids, coverage and
 * warning counts is written against. That replacement is the intent of this
 * script (those fixtures were authored as a stand-in for exactly this pass), but
 * expect assertions tuned to the hand-authored corpus to need revisiting
 * afterwards. `git diff packages/compiler/test/fixtures/llm/` is the review.
 *
 * Because it both spends money and overwrites committed fixtures, it refuses to
 * do anything without an explicit `--yes`.
 *
 * ## Afterwards
 *
 * Every compiled artifact lands in `policies/compiled/`. A compiled IR is a
 * security artifact: read each `*.report.md` by hand — the Outbound-visible
 * identifiers and Rejected candidates sections especially — before committing.
 */
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AnthropicLlmClient,
  requestHash,
  runCli,
  type LlmClient,
  type LlmRequest,
  type LlmSchema,
} from "../packages/compiler/src/index.js";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURE_DIR = join(REPO, "packages", "compiler", "test", "fixtures", "llm");
const POLICY_DIR = join(REPO, "policies");
const COMPILED_DIR = join(POLICY_DIR, "compiled");
const POLICIES = ["p-fin", "p-med", "p-corp"] as const;

/**
 * Wraps the live client and writes each response to disk on the way back.
 *
 * It records the PARSED value rather than the raw HTTP body, because that is
 * exactly what `FixtureLlmClient` re-validates on replay: recording anything
 * else would produce files that the replay path then rejects. Filenames follow
 * the committed convention `<schemaName>.<hash>.json` — only the trailing hash
 * is load-bearing, the prefix is there so a human can browse the directory.
 */
class RecordingLlmClient implements LlmClient {
  public recorded = 0;

  constructor(
    private readonly inner: LlmClient,
    private readonly dir: string,
  ) {}

  async complete<T>(request: LlmRequest, schema: LlmSchema<T>): Promise<T> {
    const value = await this.inner.complete(request, schema);
    const name = `${request.schemaName}.${requestHash(request)}.json`;
    writeFileSync(join(this.dir, name), `${JSON.stringify(value, null, 2)}\n`);
    this.recorded += 1;
    console.log(`    recorded ${name}`);
    return value;
  }
}

function refuse(): never {
  console.error(
    [
      "record-fixtures: refusing to run without --yes.",
      "",
      "This is the only script in the repository that spends money. It calls the",
      "Claude API once per policy plus once per entityType, across three policy",
      "documents, and it OVERWRITES the committed fixtures that the current test",
      "suite is written against.",
      "",
      "  pnpm -C packages/compiler exec vite-node ../../scripts/record-fixtures.ts --yes",
    ].join("\n"),
  );
  process.exit(2);
}

async function main(): Promise<number> {
  if (!process.argv.includes("--yes")) refuse();

  // Constructed once, up front, so a missing credential fails here — before a
  // single policy is read and before anything on disk is touched.
  const live = new AnthropicLlmClient();
  const recorder = new RecordingLlmClient(live, FIXTURE_DIR);

  mkdirSync(FIXTURE_DIR, { recursive: true });
  mkdirSync(COMPILED_DIR, { recursive: true });

  for (const name of POLICIES) {
    console.log(`\ncompiling ${name} (live)...`);
    // Driven through the real CLI rather than compilePolicy directly, so what is
    // recorded is produced by exactly the code path `sih-compile --live` runs.
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
        "--live",
      ],
      { liveClient: () => recorder },
    );
    if (code !== 0) {
      console.error(`\n${name} failed to compile; stopping before the remaining policies.`);
      return 1;
    }
  }

  console.log(`\nrecorded ${recorder.recorded} responses into ${FIXTURE_DIR}`);
  console.log("re-running the suite against the recorded fixtures to prove they replay...\n");

  // Proof of replay: the suite constructs FixtureLlmClient only, so if a
  // recorded file does not answer the prompt that produced it, this goes red.
  const suite = spawnSync("pnpm", ["-r", "test"], { cwd: REPO, stdio: "inherit" });
  if (suite.status !== 0) {
    console.error(
      "\nthe suite does not pass against the newly recorded fixtures. The recordings are\n" +
        "on disk and are not lost — read `git diff packages/compiler/test/fixtures/llm/`\n" +
        "and decide whether the model's output changed, or an assertion was tuned to the\n" +
        "hand-authored corpus these replaced.",
    );
    return 1;
  }

  console.log(
    `\ndone. Review each ${COMPILED_DIR}/*.report.md by hand before committing —\n` +
      "a compiled IR is a security artifact and does not land unread.",
  );
  return 0;
}

process.exitCode = await main();
