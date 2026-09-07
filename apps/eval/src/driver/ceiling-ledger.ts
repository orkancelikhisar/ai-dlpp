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
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const RUNS = join(REPO, "runs");

interface Segment {
  readonly file: string;
  readonly name: string;
  readonly note: string;
}

const SEGMENTS: readonly Segment[] = [
  {
    file: "ceiling-probe.spend.json",
    name: "probe",
    note: "The 6-model x 2-family probe. Its process was killed deliberately while diagnosing the Alibaba streaming hang; the probe was re-run inside each later launch.",
  },
  {
    file: "ceiling-ceiling-01.part1.spend.json",
    name: "ceiling-01 window 1",
    note: "Arms 1-7: deepseek judge+b, qwen-flash judge+b, mistral judge+b, nemotron judge. The process stopped after the nemotron judge arm with no [stop] line, no error and no done line; the spend guard did not trip and the cause is unknown.",
  },
  {
    file: "ceiling-glmon-01.spend.json",
    name: "glmon-01",
    note: "GLM-5.3-flash with thinking ON, judge arm only; its B arm was skipped after a 0/3 probe. Ran concurrently with window 1, against a different provider.",
  },
  {
    file: "ceiling-ceiling-01.spend.json",
    name: "ceiling-01 window 2",
    note: "Arms 8-10 plus a re-run of the nemotron judge arm, under the same runId, pins and request body as window 1.",
  },
];

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

const segments: Record<string, unknown>[] = SEGMENTS.filter((s) => existsSync(join(RUNS, s.file))).map((s) => {
  const d = JSON.parse(readFileSync(join(RUNS, s.file), "utf8")) as Record<string, unknown>;
  return { segment: s.name, ledger: `runs/${s.file}`, note: s.note, ...d };
});

const ledgerTotalUsd = segments.reduce((n, s) => n + Number(s["costUsd"] ?? 0), 0);
const ledgerTotalCalls = segments.reduce((n, s) => n + Number(s["calls"] ?? 0), 0);
const key = await keyUsage();
const residual = key === null ? null : Number((key - ledgerTotalUsd).toFixed(6));

const out = {
  what: "Every spend segment of the capability-ceiling arm, joined and reconciled against the key endpoint.",
  regenerateWith: "OPENROUTER_API_KEY=... pnpm -C apps/eval ceiling:ledger",
  segments,
  ledgerTotalUsd: Number(ledgerTotalUsd.toFixed(6)),
  ledgerTotalCalls,
  keyUsageFinalUsd: key,
  residualUsd: residual,
  residualNote:
    "key usage minus the ledger total. POSITIVE means spend the driver never saw (diagnostic curls made " +
    "outside it). NEGATIVE means the key endpoint's accounting trailing the per-response usage.cost " +
    "figures, or a segment still in flight when the key was read. The sign alone does not settle which.",
  keyLimitUsd: 10,
  hardStopUsd: 7,
  guardEverTripped: segments.some((s) => s["tripped"] === true),
  guardNote:
    "No stop path fired in any segment. This experiment was bounded by wall-clock time, not by budget.",
};

writeFileSync(join(RUNS, "ceiling-combined.spend.json"), `${JSON.stringify(out, null, 2)}\n`);
console.log(
  `ledger total $${out.ledgerTotalUsd.toFixed(5)} over ${ledgerTotalCalls} calls; ` +
    `key $${key === null ? "unread" : key.toFixed(5)}; residual ` +
    `${residual === null ? "unknown" : `$${residual.toFixed(5)}`}; guard tripped: ${out.guardEverTripped}`,
);
for (const s of segments) {
  console.log(`  ${String(s["segment"]).padEnd(22)} calls ${String(s["calls"]).padStart(5)}  $${Number(s["costUsd"]).toFixed(5)}`);
}
