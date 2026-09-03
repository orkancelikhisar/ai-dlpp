import { randomBytes, createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../driver/corpus.js";
import {
  SHUFFLE_CONSTRAINT,
  buildPredicateQueue,
  forbiddenTokensOf,
  serializePredicateQueue,
  type BuiltPredicateQueue,
  type QueueSourceItem,
} from "./predicate-queue.js";

/**
 * Emits the blind predicate queue and the reverse mapping, into a directory
 * given on the command line.
 *
 * ## Why the output directory is an argument and the salt is not committed
 *
 * The queue is a LIVE handover while a round is running, so it is written to a
 * directory the caller names rather than into the tree. The SALT and the SEED
 * must not be committed at all: the seed alone reconstructs the row ORDER from
 * the source corpus, which re-groups the strata the shuffle exists to break,
 * and the salt alone inverts every row id by hashing the 189 known ids. Both
 * are generated here and written only to the mapping file.
 *
 * WHAT HAPPENS AFTER THE ROUND IS DIFFERENT, and the first version of this
 * comment was wrong about it. A finished round's queue bytes ARE committed --
 * `corpora/generated/injection-p-fin-v2.predicate-queue.jsonl` is the file the
 * predicate round's two annotators read -- because a handover nobody can
 * inspect is a blindness claim nobody can check, which is exactly what that
 * round shipped. The salt and the seed still stay out; `predicate-round.ts`
 * records sha256 commitments to them and states plainly that the queue can
 * therefore be verified but not regenerated.
 *
 * Pass `--salt` and `--seed` to reproduce a previous emission from its mapping
 * file; omit them and this generates 128 bits of each.
 *
 * ## How it is run, and why there is no `import.meta.url === argv[1]` guard
 *
 * Not runnable under plain `node`, for `build-v2.ts`'s reason: node's type
 * stripper does not rewrite the `./x.js` specifiers TypeScript requires
 * (MEASURED this session -- `ERR_MODULE_NOT_FOUND` on `../driver/corpus.js`).
 * It runs under vite-node, and under vite-node the self-execution guard the
 * other build modules use CANNOT fire: MEASURED, `process.argv[1]` is
 * vite-node's own `cli.mjs`, never the script. So this module exports `main`
 * and a two-line driver calls it:
 *
 *   echo 'import {main} from "<repo>/apps/eval/src/corpus/build-predicate-queue.ts";
 *         process.stdout.write(main(process.argv));' > driver.ts
 *   node node_modules/.pnpm/vite-node@<v>/node_modules/vite-node/dist/cli.mjs \
 *     driver.ts -- --out <dir> [--salt <hex>] [--seed <hex>]
 *
 * A guard that cannot fire under the only runtime that can load the file would
 * be a comment claiming a run path that does not exist, which is this
 * repository's most-repeated defect. `corpus-predicate-queue.test.ts` builds
 * the same rows in memory on every `pnpm -r test`, so every property the queue
 * is required to have is checked whether or not this entry point runs again.
 */

export const REPO_ROOT = fileURLToPath(new URL("../../../../", import.meta.url));
export const PREDICATE_QUEUE_SOURCE = join(
  REPO_ROOT,
  "corpora/generated/injection-p-fin-v2.labelled.jsonl",
);
export const QUEUE_FILENAME = "predicate-queue.jsonl";
export const MAP_FILENAME = "rowid-map.json";

export function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

export interface EmittedPredicateQueue {
  readonly built: BuiltPredicateQueue;
  readonly queueJsonl: string;
  readonly mapJson: string;
  readonly items: readonly QueueSourceItem[];
}

/**
 * The emitter's own refusal: it re-reads the SERIALIZED queue and throws if any
 * corpus family id, entity type, `neg:` label, role, carrier id or item id
 * survives into the part of a row that is not the message.
 *
 * DELIBERATELY RESTRICTED TO THE NON-`text` PART, because the message text
 * legitimately contains some of those strings as ordinary English -- MEASURED
 * on the committed corpus: "client" in 7 items, "competitor" in 8,
 * "counterparty" in 7 (all seven of them carrier hn02's, whose opening sentence
 * is about a counterparty), and the carrier ids "o14" and "o07" as accidental
 * substrings of two random tokens. Scanning the whole line would fail on the
 * message an annotator is supposed to read.
 *
 * EXPORTED so it can be exercised on inputs the committed corpus cannot
 * produce. It was inline and therefore untestable: the test only ever builds
 * from the one committed corpus at the one committed shape, so both guards
 * could be disabled -- MEASURED, `if (false && ...)` on either -- with a green
 * suite. A guard broken since it was written looks identical to a working one
 * until the next round emits through it.
 *
 * @throws when a row has any key but `rowId` and `text`, when a forbidden token
 *   appears anywhere outside the message, or when a row's whole message IS one
 *   of the tokens.
 */
export function assertQueueCarriesNothingElse(queueJsonl: string, tokens: readonly string[]): void {
  for (const [i, line] of queueJsonl.split("\n").entries()) {
    if (line === "") continue;
    const parsed = JSON.parse(line) as Record<string, unknown>;
    const keys = Object.keys(parsed).sort();
    if (keys.length !== 2 || keys[0] !== "rowId" || keys[1] !== "text") {
      throw new Error(`queue line ${i + 1} carries keys ${JSON.stringify(keys)}`);
    }
    const withoutText = JSON.stringify({ rowId: parsed["rowId"] });
    for (const token of tokens) {
      if (withoutText.includes(token)) {
        throw new Error(`queue line ${i + 1} leaks ${JSON.stringify(token)} outside the message`);
      }
      if (parsed["text"] === token) {
        throw new Error(`queue line ${i + 1}'s text is exactly ${JSON.stringify(token)}`);
      }
    }
  }
}

/** Builds both files' bytes, refusing through `assertQueueCarriesNothingElse` before returning. */
export function buildPredicateQueueArtifacts(options: {
  readonly salt: string;
  readonly seed: string;
  readonly sourcePath?: string;
}): EmittedPredicateQueue {
  const sourcePath = options.sourcePath ?? PREDICATE_QUEUE_SOURCE;
  const sourceText = readFileSync(sourcePath, "utf8");
  const items = loadCorpus(sourceText) as unknown as readonly QueueSourceItem[];
  const built = buildPredicateQueue(items, { salt: options.salt, seed: options.seed });
  const queueJsonl = serializePredicateQueue(built.rows);

  assertQueueCarriesNothingElse(queueJsonl, forbiddenTokensOf(items));

  const mapJson =
    JSON.stringify(
      {
        what: "the reverse mapping for " + QUEUE_FILENAME + ". NEVER hand an annotator this file, " +
          "or any path in this directory other than the queue itself.",
        shuffle: {
          algorithm:
            "seeded Fisher-Yates over the source order, then seeded repair swaps that each strictly " +
            "reduce same-" + SHUFFLE_CONSTRAINT + " adjacency, until it is zero",
          constraint: SHUFFLE_CONSTRAINT,
          seed: options.seed,
        },
        source: {
          path: relative(REPO_ROOT, sourcePath),
          sha256: sha256(sourceText),
          items: items.length,
        },
        artifact: {
          path: QUEUE_FILENAME,
          sha256: sha256(queueJsonl),
          rows: built.rows.length,
        },
        ...built.map,
      },
      null,
      2,
    ) + "\n";

  return { built, queueJsonl, mapJson, items };
}

export function writePredicateQueueArtifacts(
  outDir: string,
  options: { readonly salt: string; readonly seed: string },
): EmittedPredicateQueue {
  const emitted = buildPredicateQueueArtifacts(options);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, QUEUE_FILENAME), emitted.queueJsonl, "utf8");
  writeFileSync(join(outDir, MAP_FILENAME), emitted.mapJson, "utf8");
  return emitted;
}

function argOf(argv: readonly string[], name: string): string | undefined {
  const at = argv.indexOf(`--${name}`);
  return at < 0 ? undefined : argv[at + 1];
}

/**
 * Emits both files and returns the summary a caller should print.
 *
 * The summary reports every stratum dimension, INCLUDING the two the shuffle
 * makes worse: source order interleaves the waves and the densities, so
 * breaking the carrier blocks necessarily pushes those two UP toward chance.
 * Reporting only the dimensions that improved would be picking the numbers
 * that look good.
 */
export function main(argv: readonly string[]): string {
  const outDir = argOf(argv, "out");
  if (outDir === undefined) {
    throw new Error("usage: --out <dir> [--salt <hex>] [--seed <hex>]");
  }
  const salt = argOf(argv, "salt") ?? randomBytes(32).toString("hex");
  const seed = argOf(argv, "seed") ?? randomBytes(16).toString("hex");
  const emitted = writePredicateQueueArtifacts(outDir, { salt, seed });
  const source = emitted.built.map.adjacency.sourceOrder;
  const queue = emitted.built.map.adjacency.queueOrder;
  const sorted = emitted.built.map.adjacency.rowIdSortedOrder;
  const lines = [
    `${emitted.built.rows.length} rows -> ${join(outDir, QUEUE_FILENAME)}`,
    `queue sha256 ${sha256(emitted.queueJsonl)}`,
    `map          ${join(outDir, MAP_FILENAME)}`,
    "dimension                 source        queue      rowIdSort       chance",
  ];
  for (const [i, before] of source.entries()) {
    const after = queue[i]!;
    const byId = sorted[i]!;
    lines.push(
      `  ${before.dimension.padEnd(22)} ` +
        `${before.rate.toFixed(4)} (${String(before.sameStratumPairs).padStart(3)})  ` +
        `${after.rate.toFixed(4)} (${String(after.sameStratumPairs).padStart(3)})  ` +
        `${byId.rate.toFixed(4)} (${String(byId.sameStratumPairs).padStart(3)})  ` +
        `${before.chanceRate.toFixed(4)}`,
    );
  }
  return lines.join("\n") + "\n";
}
