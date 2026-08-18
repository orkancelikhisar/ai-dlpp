#!/usr/bin/env node
import { mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { compilePolicy } from "./compile.js";
import { AnthropicLlmClient } from "./llm/anthropic.js";
import type { LlmClient } from "./llm/client.js";
import { FixtureLlmClient } from "./llm/fixture.js";

/**
 * `sih-compile` — a policy document in, an auditable IR out.
 *
 * The one invariant this file exists to hold: THE NETWORK IS NEVER THE DEFAULT.
 * A run reaches a frontier model only when `--live` is passed explicitly. With
 * neither `--live` nor injected fixtures the CLI exits 1 rather than picking a
 * client for you, because the failure mode of guessing is a CI job that quietly
 * spends money on every push.
 *
 * Importing this module is free: `AnthropicLlmClient` resolves credentials in
 * its CONSTRUCTOR, not at import, so the static import above costs nothing on a
 * run that never goes live.
 */

const FLAGS_WITH_VALUES = ["--policy", "--providers", "--out", "--name"] as const;
type ValueFlag = (typeof FLAGS_WITH_VALUES)[number];

const USAGE =
  "usage: sih-compile --policy <file.md> --providers <providers.json> " +
  "--out <dir> --name <name> [--live]";

export interface CliDeps {
  /**
   * Committed LLM responses. Present means "replay these" — the test suite's
   * only mode, and the reason no test can reach the network.
   */
  readonly fixtures?: ReadonlyMap<string, unknown>;
  /**
   * Builds the live client. A seam rather than a bare `new AnthropicLlmClient()`
   * at the call site so a test can prove this is NEVER invoked without `--live`:
   * injecting a factory that throws turns "the CLI went live by accident" from
   * an unbilled silent pass into a red test.
   */
  readonly liveClient?: () => LlmClient;
  readonly write?: (line: string) => void;
  readonly writeError?: (line: string) => void;
}

interface ParsedArgs {
  readonly policy: string;
  readonly providers: string;
  readonly out: string;
  readonly name: string;
  readonly live: boolean;
}

/**
 * `--name` becomes three filenames under `--out`. Left unchecked it is also a
 * path: `--name ../../etc/thing` writes outside the output directory. A compiled
 * IR is a security artifact, so the name is constrained to something that cannot
 * traverse.
 */
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

function parseArgs(argv: readonly string[]): ParsedArgs | { error: string } {
  const values = new Map<ValueFlag, string>();
  let live = false;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]!;
    if (arg === "--live") {
      live = true;
      continue;
    }
    if (!(FLAGS_WITH_VALUES as readonly string[]).includes(arg)) {
      // Silently ignoring an unrecognised flag is how "--liv" becomes a run that
      // did something other than what was asked, and still exits 0.
      return { error: `unknown flag "${arg}"\n${USAGE}` };
    }
    const flag = arg as ValueFlag;
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      return { error: `${flag} needs a value\n${USAGE}` };
    }
    if (values.has(flag)) {
      return { error: `${flag} given twice; the second value would silently win` };
    }
    values.set(flag, value);
    i += 1;
  }

  const missing = FLAGS_WITH_VALUES.filter((f) => !values.has(f));
  if (missing.length > 0) {
    return { error: `missing required ${missing.join(", ")}\n${USAGE}` };
  }

  const name = values.get("--name")!;
  if (!SAFE_NAME.test(name)) {
    return {
      error: `--name "${name}" is not a plain file basename (letters, digits, ".", "-", "_")`,
    };
  }

  return {
    policy: values.get("--policy")!,
    providers: values.get("--providers")!,
    out: values.get("--out")!,
    name,
    live,
  };
}

/**
 * Picks the compiler's model client, or refuses.
 *
 * Both branches are explicit and neither is a default: fixtures replay, `--live`
 * calls the API. Supplying both is an error rather than a precedence rule,
 * because a precedence rule is something a reader must remember correctly in
 * order to know whether a command costs money.
 */
function selectClient(
  live: boolean,
  fixtures: ReadonlyMap<string, unknown> | undefined,
  makeLiveClient: () => LlmClient,
): LlmClient | { error: string } {
  if (fixtures !== undefined && live) {
    return {
      error: "--live was passed alongside injected fixtures; pick one — refusing to guess which",
    };
  }
  if (fixtures !== undefined) return new FixtureLlmClient(fixtures);
  if (live) return makeLiveClient();
  return {
    error:
      "refusing to run: no fixtures were provided and --live was not passed. " +
      "Pass --live to call the model (this costs money), or supply fixtures. " +
      "There is deliberately no default.",
  };
}

export async function runCli(argv: readonly string[], deps: CliDeps): Promise<number> {
  const write = deps.write ?? ((line: string) => console.log(line));
  const writeError = deps.writeError ?? ((line: string) => console.error(line));
  const makeLiveClient = deps.liveClient ?? (() => new AnthropicLlmClient());

  const args = parseArgs(argv);
  if ("error" in args) {
    writeError(args.error);
    return 1;
  }

  try {
    // Everything local happens BEFORE a client is built, so a typo in a path
    // fails without having resolved credentials or opened a connection.
    const document = readFileSync(args.policy, "utf8");
    const manifest: unknown = JSON.parse(readFileSync(args.providers, "utf8"));
    mkdirSync(args.out, { recursive: true });

    const client = selectClient(args.live, deps.fixtures, makeLiveClient);
    if ("error" in client) {
      writeError(client.error);
      return 1;
    }

    const result = await compilePolicy({
      client,
      document,
      manifest,
      policyName: args.name,
    });

    writeFileSync(join(args.out, `${args.name}.ir.json`), `${JSON.stringify(result.ir, null, 2)}\n`);
    writeFileSync(join(args.out, `${args.name}.report.md`), result.report);
    writeFileSync(
      join(args.out, `${args.name}.selftest.json`),
      `${JSON.stringify(result.selfTestCases, null, 2)}\n`,
    );

    // Warnings do not change the exit code — a policy that is 90% right is still
    // worth emitting — but they are the reason the report exists, so they are
    // printed rather than left for someone to happen upon in a file.
    for (const warning of result.warnings) writeError(`warning: ${warning}`);
    write(
      `compiled ${args.name}: ${result.ir.entityTypes.length} entityTypes, ` +
        `${result.ir.rules.length} rules, ${result.warnings.length} warnings -> ${args.out}`,
    );
    return 0;
  } catch (e) {
    writeError(`error: ${(e as Error).message}`);
    return 1;
  }
}

/** True only when this file is the process entrypoint, never under a test runner. */
function isEntrypoint(): boolean {
  const invoked = process.argv[1];
  if (invoked === undefined) return false;
  try {
    return realpathSync(invoked) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

if (isEntrypoint()) {
  process.exitCode = await runCli(process.argv.slice(2), {});
}
