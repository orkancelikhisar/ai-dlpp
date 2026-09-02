import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPolicyIr } from "@sih/core";
import {
  selfTestExamplesFromCompiledCorpus,
  selfTestExamplesFromLlmFixture,
  type SelfTestExample,
} from "./contamination.js";
import { generateCorpus, serializeCorpus, serializeManifest, type CorpusManifest, type GeneratedCorpus } from "./generate.js";

/**
 * Reads this repository's inputs, runs the generator, and writes the two
 * artifacts. Shared by the CLI entry at the bottom and by
 * `corpus-artifact.test.ts`, deliberately: if the test built its inputs its own
 * way, it would prove the generator is deterministic and prove nothing about
 * the committed file.
 */

export const REPO_ROOT = fileURLToPath(new URL("../../../../", import.meta.url));
export const IR_PATH = join(REPO_ROOT, "policies/compiled/p-fin.ir.json");
export const SELFTEST_PATH = join(REPO_ROOT, "policies/compiled/p-fin.selftest.json");
export const COMPILER_FIXTURE_DIR = join(REPO_ROOT, "packages/compiler/test/fixtures/llm");
export const OUT_DIR = join(REPO_ROOT, "corpora/generated");
export const CORPUS_PATH = join(OUT_DIR, "injection-p-fin-v1.jsonl");
export const MANIFEST_PATH = join(OUT_DIR, "injection-p-fin-v1.manifest.json");

/**
 * Every self-test example this repository holds, from BOTH shapes:
 *
 * - `policies/compiled/p-fin.selftest.json`, the committed self-test corpus the
 *   compiler actually executed against p-fin, carrying `corpusTag`.
 * - `packages/compiler/test/fixtures/llm/SelfTestCases.*.json`, the recorded
 *   model responses the compiler's own tests replay.
 *
 * The brief names the second. The first is included because it is the corpus
 * the contamination rule is actually about -- it is generated from the very
 * policy this corpus is scored under, which is the circularity spec 6.2 asks to
 * be broken. Filenames are sorted before reading: `readdirSync` order is a
 * filesystem property and this pipeline's outputs must not depend on one.
 */
export function loadSelfTestExamples(): SelfTestExample[] {
  const examples: SelfTestExample[] = [
    ...selfTestExamplesFromCompiledCorpus(readFileSync(SELFTEST_PATH, "utf8"), "policies/compiled/p-fin.selftest.json"),
  ];
  const names = readdirSync(COMPILER_FIXTURE_DIR)
    .filter((n) => n.startsWith("SelfTestCases.") && n.endsWith(".json"))
    .sort();
  for (const name of names) {
    examples.push(
      ...selfTestExamplesFromLlmFixture(
        readFileSync(join(COMPILER_FIXTURE_DIR, name), "utf8"),
        `packages/compiler/test/fixtures/llm/${name}`,
      ),
    );
  }
  return examples;
}

export interface EmittedManifest extends CorpusManifest {
  /**
   * Hash of the emitted JSONL, so a run record can pin the corpus it scored
   * against. Lives here rather than inside the corpus for the obvious reason
   * that a file cannot contain its own hash.
   */
  readonly artifact: { readonly corpusSha256: string; readonly corpusBytes: number; readonly corpusPath: string };
}

export interface BuiltArtifacts {
  readonly corpusJsonl: string;
  readonly manifestJson: string;
  readonly generated: GeneratedCorpus;
  readonly manifest: EmittedManifest;
}

export function buildArtifacts(seed?: string): BuiltArtifacts {
  const irText = readFileSync(IR_PATH, "utf8");
  const generated = generateCorpus({
    ...(seed === undefined ? {} : { seed }),
    ir: loadPolicyIr(irText),
    irSource: "policies/compiled/p-fin.ir.json",
    irHash: createHash("sha256").update(irText, "utf8").digest("hex"),
    selfTestExamples: loadSelfTestExamples(),
  });
  const corpusJsonl = serializeCorpus(generated.items);
  const manifest: EmittedManifest = {
    ...generated.manifest,
    artifact: {
      corpusSha256: createHash("sha256").update(corpusJsonl, "utf8").digest("hex"),
      corpusBytes: Buffer.byteLength(corpusJsonl, "utf8"),
      corpusPath: "corpora/generated/injection-p-fin-v1.jsonl",
    },
  };
  return { corpusJsonl, manifestJson: serializeManifest(manifest), generated, manifest };
}

export function writeArtifacts(): BuiltArtifacts {
  const built = buildArtifacts();
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(CORPUS_PATH, built.corpusJsonl, "utf8");
  writeFileSync(MANIFEST_PATH, built.manifestJson, "utf8");
  return built;
}

// Regenerates both artifacts in place -- but NOT under plain node. MEASURED on
// node v26.0.0: `node --experimental-strip-types apps/eval/src/corpus/build.ts`
// fails with ERR_MODULE_NOT_FOUND, because `@sih/core`'s package `main` is
// `src/index.ts` and its internal specifiers end in `.js`, which node's type
// stripping does not remap to `.ts`; adding a resolver hook that does remap them
// then fails with ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX. This comment previously
// claimed that command works. See `build-adjudicated.ts` for what actually ran.
// The guarantee that matters does not depend on any of it:
// `corpus-artifact.test.ts` regenerates in memory and compares to the committed
// bytes on every `pnpm -r test`.
if (process.argv[1] !== undefined && import.meta.url === `file://${process.argv[1]}`) {
  const built = writeArtifacts();
  process.stdout.write(
    `${built.generated.items.length} items, ${built.manifest.counts.goldSpans} gold spans, ` +
      `sha256 ${built.manifest.artifact.corpusSha256}\n`,
  );
}
