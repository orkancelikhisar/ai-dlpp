import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Committed LLM responses, replayed by FixtureLlmClient so no test touches the
 * network.
 *
 * Filename convention: `<schemaName>.<hash>.json`, e.g.
 * `Extraction.4f1c9ab2c0d3e5f7.json`. FixtureLlmClient looks these up by
 * requestHash alone, so only the trailing segment is load-bearing — the
 * schemaName prefix exists so a human can tell at a glance which stage a file
 * belongs to. A directory of bare hex names is unreviewable, and this repo
 * requires humans to audit compiler artifacts before they ship.
 */
const DEFAULT_DIR = join(import.meta.dirname, "llm");

/** Matches requestHash's output: sha256 truncated to 16 lowercase hex chars. */
const HASH_SEGMENT = /^[0-9a-f]{16}$/;

export function loadTestFixtures(dir: string = DEFAULT_DIR): Map<string, unknown> {
  const fixtures = new Map<string, unknown>();
  for (const name of readdirSync(dir).sort()) {
    // Only *.json claims to be a fixture; a README or a macOS .DS_Store beside
    // them is not an error.
    if (!name.endsWith(".json")) continue;
    const hash = name.slice(0, -".json".length).split(".").at(-1) ?? "";
    if (!HASH_SEGMENT.test(hash)) {
      // Keying such a file by its own name would make it silently unreachable:
      // the miss error would quote a hash the reader can see sitting right there
      // in the directory, and no reading of the filename would explain why.
      throw new Error(
        `fixture ${name} is not named <schemaName>.<hash>.json (hash = 16 lowercase hex chars)`,
      );
    }
    if (fixtures.has(hash)) {
      // Two files, one key: whichever lost the race would never replay, and the
      // winner would answer a request it was not recorded for.
      throw new Error(`two fixtures claim hash ${hash}; delete the stale one (found ${name})`);
    }
    fixtures.set(hash, JSON.parse(readFileSync(join(dir, name), "utf8")));
  }
  return fixtures;
}
