import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadTestFixtures } from "./index.js";

const HASH = "0123456789abcdef";

function fixtureDir(files: Record<string, string>): string {
  const dir = mkdtempSync(join(tmpdir(), "sih-fixtures-"));
  for (const [name, body] of Object.entries(files)) writeFileSync(join(dir, name), body);
  return dir;
}

describe("loadTestFixtures", () => {
  it("keys the map by the trailing hash segment, not the whole filename", () => {
    // The schemaName prefix exists for human browsability only; FixtureLlmClient
    // looks up by hash alone.
    const map = loadTestFixtures(fixtureDir({ [`Extraction.${HASH}.json`]: '{"ok":true}' }));
    expect(map.get(HASH)).toEqual({ ok: true });
  });

  it("ignores non-JSON files so a README or a .DS_Store cannot break the suite", () => {
    const map = loadTestFixtures(
      fixtureDir({ [`Extraction.${HASH}.json`]: "{}", "README.md": "notes", ".DS_Store": "" }),
    );
    expect(map.size).toBe(1);
  });

  it("rejects a JSON file whose trailing segment is not a request hash", () => {
    // Such a file would key the map by "notes" and then never replay: the miss
    // error would name a hash the reader can already see sitting in the directory.
    expect(() => loadTestFixtures(fixtureDir({ "notes.json": "{}" }))).toThrow(/notes\.json/);
  });

  it("rejects two fixtures sharing one hash rather than letting one win", () => {
    expect(() =>
      loadTestFixtures(fixtureDir({ [`A.${HASH}.json`]: "{}", [`B.${HASH}.json`]: "{}" })),
    ).toThrow(new RegExp(HASH));
  });

  it("loads the committed fixture directory by default", () => {
    expect(loadTestFixtures().size).toBeGreaterThan(0);
  });
});
