import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadCorpus } from "../src/driver/corpus.js";
import { CORPUS_PATH, MANIFEST_PATH, buildArtifacts } from "../src/corpus/build.js";

/**
 * The committed artifact must be exactly what the generator produces. Without
 * this the corpus would be a file somebody once made -- a hand-edit, a partial
 * regeneration or a stale seed would all pass silently, and every number
 * measured against it would describe a corpus nobody can rebuild.
 *
 * Modelled on `packages/compiler/test/compiled.test.ts`, which holds the same
 * line for `policies/compiled/`.
 */
describe("corpora/generated/injection-p-fin-v1", () => {
  const built = buildArtifacts();

  it("reproduces the committed corpus byte for byte", () => {
    expect(readFileSync(CORPUS_PATH, "utf8")).toBe(built.corpusJsonl);
  });

  it("reproduces the committed manifest byte for byte", () => {
    expect(readFileSync(MANIFEST_PATH, "utf8")).toBe(built.manifestJson);
  });

  it("matches the sha256 the manifest records for it", () => {
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8")) as {
      artifact: { corpusSha256: string; corpusBytes: number };
    };
    expect(manifest.artifact.corpusSha256).toBe(built.manifest.artifact.corpusSha256);
    expect(manifest.artifact.corpusBytes).toBe(Buffer.byteLength(readFileSync(CORPUS_PATH, "utf8"), "utf8"));
  });

  it("loads through the harness reader", () => {
    expect(loadCorpus(readFileSync(CORPUS_PATH, "utf8"))).toHaveLength(built.generated.items.length);
  });

  it("contains no raw control characters", () => {
    // The Write/Edit tooling used on this repository decodes escapes into raw
    // characters, and a literal control byte in a JSONL corpus is invisible in
    // review and fatal to a reader. Newline is the line separator; nothing else
    // below 0x20 may appear.
    const bytes = readFileSync(CORPUS_PATH, "utf8");
    const offenders = [...bytes].filter((c) => c !== "\n" && c.charCodeAt(0) < 0x20);
    expect(offenders).toEqual([]);
  });

  it("does not touch the smoke fixtures", () => {
    expect(CORPUS_PATH).toContain("corpora/generated/");
    expect(CORPUS_PATH).not.toContain("fixtures");
  });
});
