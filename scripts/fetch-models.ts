/**
 * fetch-models.ts -- downloads the tier-1 model files and verifies them against
 * the content hashes pinned in MODEL_MANIFEST.
 *
 * ## Running it
 *
 *   pnpm -C packages/tier1 exec vite-node ../../scripts/fetch-models.ts
 *
 * (Same runner as scripts/record-fixtures.ts and for the same reason: this
 * tree has no tsx, and Node's own type stripping resolves `./x.js` literally
 * while every source file here imports that way.)
 *
 * Public files, no credentials. They land in packages/tier1/models/<id>/, which
 * is gitignored -- the weights alone are ~846 MB and never enter git.
 *
 * Safe to re-run: a file already present at its pinned size and hash is skipped
 * without touching the network. MEASURED, warm: the whole 12-file verify pass
 * is ~0.73 s wall, of which ~0.42 s is vite-node startup (a no-op script costs
 * the same) and ~0.33 s is sha256 over the 846 MB of weights. So the re-run
 * cost is dominated by process startup, not by hashing.
 *
 * ## Where trust comes from
 *
 * For an already-pinned entry the manifest is the authority. For an UNPINNED
 * one it cannot be, or the pin would be circular: hashing whatever arrived and
 * pasting that in verifies the file against itself, so an HTML login wall
 * becomes a permanently "verified" model. The bootstrap therefore verifies
 * every freshly downloaded file against the Hub tree API, which reports an
 * independent digest per file at the pinned revision -- `lfs.oid` (sha256) for
 * LFS blobs, and the git blob sha1 for everything else. Nothing is written or
 * printed unless that check passes.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_FILE_PLACEHOLDER,
  MODEL_MANIFEST,
  modelFileUrl,
  type ModelEntry,
} from "../packages/tier1/src/manifest.js";

const MODELS_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "packages", "tier1", "models");

const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

/** Git's blob id: sha1 over "blob <len>\0" + content. The Hub reports this for non-LFS files. */
const gitBlobSha1 = (bytes: Uint8Array): string =>
  createHash("sha1")
    .update(`blob ${bytes.byteLength}\0`)
    .update(bytes)
    .digest("hex");

interface TreeFile {
  readonly size: number;
  /** git blob sha1 */
  readonly oid: string;
  readonly lfsSha256: string | undefined;
}

/** Per-path upstream digests at one revision, used only to bootstrap an unpinned entry. */
async function fetchTree(entry: ModelEntry): Promise<Map<string, TreeFile>> {
  const url = `https://huggingface.co/api/models/${entry.repo}/tree/${entry.revision}?recursive=1`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`tree API HTTP ${response.status} ${response.statusText} for ${url}`);
  }
  const raw: unknown = await response.json();
  if (!Array.isArray(raw)) throw new Error(`tree API did not return an array for ${url}`);
  const tree = new Map<string, TreeFile>();
  for (const item of raw as Array<Record<string, unknown>>) {
    if (item["type"] !== "file") continue;
    const lfs = item["lfs"];
    tree.set(String(item["path"]), {
      size: Number(item["size"]),
      oid: String(item["oid"]),
      lfsSha256:
        typeof lfs === "object" && lfs !== null && "oid" in lfs
          ? String((lfs as Record<string, unknown>)["oid"])
          : undefined,
    });
  }
  return tree;
}

/**
 * Confirms the Hub itself vouches for these exact bytes at this path, so a
 * login wall or an error page can never be pasted in as a pin.
 */
function attestedByTree(path: string, bytes: Uint8Array, tree: Map<string, TreeFile>): string | undefined {
  const upstream = tree.get(path);
  if (upstream === undefined) return `tree API lists no file at ${path}`;
  if (upstream.size !== bytes.byteLength) {
    return `tree API says ${path} is ${upstream.size} bytes, downloaded ${bytes.byteLength}`;
  }
  const expected = upstream.lfsSha256 ?? upstream.oid;
  const actual = upstream.lfsSha256 === undefined ? gitBlobSha1(bytes) : sha256(bytes);
  const kind = upstream.lfsSha256 === undefined ? "git blob sha1" : "lfs sha256";
  if (expected !== actual) {
    return `${path} ${kind} disagrees with the tree API: upstream ${expected}, downloaded ${actual}`;
  }
  return undefined;
}

async function download(url: string, expectedBytes: number | undefined): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText} for ${url}`);
  }
  // Checked before reading the body so a truncated transfer is caught as such
  // rather than surfacing later as an unexplained hash mismatch.
  const declared = response.headers.get("content-length");
  if (declared !== null && expectedBytes !== undefined && Number(declared) !== expectedBytes) {
    throw new Error(`${url}: manifest says ${expectedBytes} bytes, server declares ${declared}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (declared !== null && Number(declared) !== bytes.byteLength) {
    throw new Error(`${url}: declared ${declared} bytes, received ${bytes.byteLength}`);
  }
  return bytes;
}

async function main(): Promise<number> {
  let unpinned = 0;
  let failed = 0;

  for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
    const bootstrapping = Object.values(entry.files).some(
      (file) => file.sha256 === MODEL_FILE_PLACEHOLDER,
    );
    let tree: Map<string, TreeFile> | undefined;
    if (bootstrapping) {
      try {
        tree = await fetchTree(entry);
      } catch (error) {
        console.error(`${id}: ${String(error)}`);
        failed += 1;
        continue;
      }
    }
    const pasteLines: string[] = [];

    for (const [path, file] of Object.entries(entry.files)) {
      const dest = join(MODELS_DIR, id, path);
      const pinned = file.sha256 !== MODEL_FILE_PLACEHOLDER;

      if (existsSync(dest) && pinned) {
        // Size first: it rejects the common truncated-download case without
        // re-hashing, which for the weights means not reading 846 MB.
        if (statSync(dest).size === file.bytes && sha256(readFileSync(dest)) === file.sha256) {
          console.log(`${id}/${path}: present and verified`);
          continue;
        }
        console.log(`${id}/${path}: on-disk contents disagree with the manifest; re-downloading`);
      }

      let bytes: Uint8Array;
      try {
        bytes = await download(modelFileUrl(entry, path), pinned ? file.bytes : undefined);
      } catch (error) {
        console.error(`${id}/${path}: ${String(error)}`);
        failed += 1;
        continue;
      }

      if (!pinned) {
        const problem = attestedByTree(path, bytes, tree ?? new Map());
        if (problem !== undefined) {
          console.error(`${id}/${path}: REFUSING to pin -- ${problem}`);
          failed += 1;
          continue;
        }
        mkdirSync(dirname(dest), { recursive: true });
        writeFileSync(dest, bytes);
        pasteLines.push(
          `      "${path}": { sha256: "${sha256(bytes)}", bytes: ${bytes.byteLength} },`,
        );
        unpinned += 1;
        continue;
      }

      const digest = sha256(bytes);
      if (digest !== file.sha256) {
        console.error(
          `${id}/${path}: hash mismatch -- manifest ${file.sha256}, downloaded ${digest}. NOT written.`,
        );
        failed += 1;
        continue;
      }
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, bytes);
      console.log(`${id}/${path}: verified and written (${bytes.byteLength} bytes)`);
    }

    if (pasteLines.length > 0) {
      console.log(`\n${id}: UNPINNED, attested by the tree API -- paste into MODEL_MANIFEST:`);
      console.log(pasteLines.join("\n"));
    }
  }

  if (failed > 0) {
    console.error(`\n${failed} file(s) could not be verified.`);
    return 1;
  }
  if (unpinned > 0) {
    console.error(`\n${unpinned} file(s) unpinned -- paste the values above and re-run.`);
    return 1;
  }
  return 0;
}

process.exitCode = await main();
