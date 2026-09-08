/** Oracle floors over the rows an arm answered, via the REAL measureOrthography. Used by extract-numbers.py. */
import { readFileSync } from "node:fs";
import { measureOrthography } from "../../apps/eval/src/corpus/leakage.js";
const rows = readFileSync(process.argv[2]!, "utf8").trim().split("\n").map((l) => JSON.parse(l) as Record<string, unknown>);
const item = (r: Record<string, unknown>) => ({ text: r["text"] as string, gold: (r["gold"] ?? []) as { start: number; end: number }[] });
const all = rows.map(item);
const answered = rows.filter((r) => Array.isArray(r["calls"]) && (r["calls"] as unknown[]).length > 0).map(item);
for (const [label, items] of [[`ALL ${all.length}`, all], [`ANSWERED ${answered.length}`, answered]] as const) {
  const o = measureOrthography(items as never) as unknown as Record<string, { precision: number; recall: number; f1: number }>;
  console.log(`${label}: budget-matched F1 ${o["budgetMatched"]!.f1.toFixed(3)} (P ${o["budgetMatched"]!.precision.toFixed(3)} R ${o["budgetMatched"]!.recall.toFixed(3)}) | unbudgeted F1 ${o["asScored"]!.f1.toFixed(3)}`);
}
