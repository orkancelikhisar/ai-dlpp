/**
 * fetch-models.ts -- downloads the tier-1 ONNX weights and verifies them
 * against the content hashes pinned in MODEL_MANIFEST.
 *
 * ## Running it
 *
 *   pnpm -C packages/tier1 exec vite-node ../../scripts/fetch-models.ts
 *
 * (Same runner as scripts/record-fixtures.ts and for the same reason: this
 * tree has no tsx, and Node's own type stripping resolves `./x.js` literally
 * while every source file here imports that way.)
 *
 * Public files, no credentials. The weights land in packages/tier1/models/,
 * which is gitignored -- they are ~846 MB and never enter git.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { MODEL_MANIFEST } from "../packages/tier1/src/config.js";

const MODELS_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "packages", "tier1", "models");
const PLACEHOLDER = "<fill from fetch-models.ts>";

const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

/**
 * Downloads every manifest entry and verifies it against its pinned hash.
 * Safe to re-run: a present file whose hash already matches is skipped without
 * touching the network. MEASURED -- the second run over the pinned pair prints
 * "present and verified" for both and exits 0 in ~1.1 s wall, which is the cost
 * of re-hashing 846 MB off disk, not a network round trip.
 */
async function main(): Promise<number> {
  mkdirSync(MODELS_DIR, { recursive: true });
  let unpinned = 0;

  for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
    const path = join(MODELS_DIR, `${id}.onnx`);
    if (existsSync(path) && entry.sha256 !== PLACEHOLDER) {
      if (sha256(readFileSync(path)) === entry.sha256) {
        console.log(`${id}: present and verified`);
        continue;
      }
      console.log(`${id}: on-disk hash disagrees with the manifest; re-downloading`);
    }

    console.log(`${id}: downloading ${entry.url}`);
    const response = await fetch(entry.url);
    if (!response.ok) {
      console.error(`${id}: HTTP ${response.status} ${response.statusText} for ${entry.url}`);
      return 1;
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    const digest = sha256(bytes);

    if (entry.sha256 === PLACEHOLDER) {
      // Written, but the manifest stays unpinned until a human pastes the
      // values in. Auto-writing them would defeat the pin entirely: the file
      // would "verify" against whatever it happened to download.
      writeFileSync(path, bytes);
      console.log(`${id}: UNPINNED -- paste into MODEL_MANIFEST:\n    sha256: "${digest}",\n    bytes: ${bytes.byteLength},`);
      unpinned += 1;
      continue;
    }
    if (digest !== entry.sha256) {
      console.error(`${id}: hash mismatch -- manifest ${entry.sha256}, downloaded ${digest}. NOT written.`);
      return 1;
    }
    writeFileSync(path, bytes);
    console.log(`${id}: verified and written (${bytes.byteLength} bytes)`);
  }

  if (unpinned > 0) {
    console.error(`\n${unpinned} manifest entry/entries unpinned -- paste the values above and re-run.`);
    return 1;
  }
  return 0;
}

process.exitCode = await main();
