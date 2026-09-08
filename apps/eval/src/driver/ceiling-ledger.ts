/**
 * Joins every spend segment of the capability-ceiling arm into one file and
 * reconciles it against the key endpoint: `pnpm -C apps/eval ceiling:ledger`.
 *
 * It exists because the per-run ledgers are keyed by `runId` and one run reused
 * an id: the completion run relaunched under `ceiling-01` and overwrote
 * `runs/ceiling-ceiling-01.spend.json`, whose earlier contents had been copied
 * to `...part1.spend.json` first. Neither file alone is the run's cost, and a
 * total that lives only in a report is a number nobody can re-derive.
 *
 * The RESIDUAL between the ledgers and the key is reported with its sign and
 * without a story attached. It has two known contributors pulling in opposite
 * directions, and this script cannot tell how much of each:
 *
 *   - POSITIVE residual (key > ledgers): spend made outside the driver, which
 *     no ledger sees -- the `/endpoints` listings, the streaming-hang isolation
 *     probes, and the provider searches, all made with plain curl.
 *   - NEGATIVE residual (ledgers > key): the key endpoint's own accounting
 *     lagging the per-response `usage.cost` figures, or a segment still in
 *     flight when the key was read.
 *
 * Do not read the sign as evidence for either on its own.
 *
 * THIS FILE IS I/O ONLY. Every decision -- the totals, the residual and its
 * sign, the guard flag -- lives in `ceiling-ledger-lib.ts` so that a test can
 * import it without a key, a network or a write into `runs/`. Keep it that way:
 * anything moved back up here becomes uncovered again the moment it arrives.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  LEDGER_SEGMENTS,
  buildCombinedLedger,
  formatLedgerSummary,
  joinSegment,
  type JoinedSegment,
} from "./ceiling-ledger-lib.js";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const RUNS = join(REPO, "runs");

async function keyUsage(): Promise<number | null> {
  const apiKey = process.env["OPENROUTER_API_KEY"];
  if (apiKey === undefined || apiKey === "") return null;
  const r = await fetch("https://openrouter.ai/api/v1/auth/key", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!r.ok) return null;
  const b = (await r.json()) as { data?: { usage?: unknown } };
  return typeof b.data?.usage === "number" ? b.data.usage : null;
}

const segments: JoinedSegment[] = LEDGER_SEGMENTS.filter((s) => existsSync(join(RUNS, s.file))).map((s) =>
  joinSegment(s, JSON.parse(readFileSync(join(RUNS, s.file), "utf8")) as Record<string, unknown>),
);

const out = buildCombinedLedger({ segments, keyUsageUsd: await keyUsage() });

writeFileSync(join(RUNS, "ceiling-combined.spend.json"), `${JSON.stringify(out, null, 2)}\n`);
for (const line of formatLedgerSummary(out)) console.log(line);
