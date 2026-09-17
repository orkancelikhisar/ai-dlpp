/**
 * Renders the TypeSafe arm's numbers: `pnpm -C apps/eval typesafe:score`.
 *
 * THIS FILE RENDERS; it decides nothing. Every count, curve and ratio comes from
 * `typesafe-score-lib.ts`, which is importable and tested, because the last time
 * a driver in this repository inlined its arithmetic a mutation review found the
 * whole file uncovered.
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadIr } from "./ceiling-run.js";
import { loadTypeSafeRecords } from "./typesafe.js";
import {
  averagePrecision,
  bestPoint,
  calibration,
  costAndTime,
  entityPoint,
  entitySweep,
  filterEffect,
  loadPredicateGold,
  pairRows,
  pointAt,
  rocAuc,
  splitHalf,
  sweep,
  tier0Baseline,
} from "./typesafe-score-lib.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");
const RUNS = join(REPO, "runs");
const f = (n: number | null | undefined, d = 3): string => (n === null || n === undefined ? "—" : n.toFixed(d));

function main(): void {
  const runId = process.env.SIH_TS_RUN_ID;
  const files = readdirSync(RUNS).filter((n) => n.endsWith(".jsonl") && n.includes(".ts-") && !n.includes(".gates.") && (runId === undefined || n.startsWith(`${runId}.`)));
  if (files.length === 0) {
    console.error(`no TypeSafe arm files in runs/${runId === undefined ? "" : ` for ${runId}`}`);
    process.exitCode = 1;
    return;
  }
  const { ir } = loadIr();
  const predicateId = ir.semanticPredicates.find((p) => p.scope === "message")?.id ?? "";
  const gold = loadPredicateGold(readFileSync(join(REPO, "corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl"), "utf8"), predicateId);
  const paper = JSON.parse(readFileSync(join(REPO, "docs/paper/data/numbers.json"), "utf8")) as {
    floor_message_level_all: { F1: number };
    local_best_message: { F1: number };
    message_level: Record<string, Record<string, { F1: number }>>;
  };
  const floor = paper.floor_message_level_all.F1;
  const hostedBest = Math.max(...[1, 2, 3].map((p) => paper.message_level["judge-deepseek-v4-flash-0731"]?.[`ceiling-0${p}`]?.F1 ?? 0));

  for (const file of files) {
    const records = loadTypeSafeRecords(readFileSync(join(RUNS, file), "utf8"));
    const dry = records.some((r) => r.dryRun);
    const { rows, unanswered } = pairRows(records, gold);
    const points = sweep(rows);
    const best = bestPoint(points);
    const half = pointAt(rows, 0.5);
    const sh = splitHalf(rows);
    const ct = costAndTime(records);
    const ent = entitySweep(ir, records, 20);
    const entBest = ent.reduce((a, b) => (b.f1 > a.f1 ? b : a), ent[0]!);
    const t0 = tier0Baseline(records);
    const fe = filterEffect(records);

    console.log(`\n=== ${file}${dry ? "   [DRY RUN: stub answers, no model was called]" : ""}`);
    console.log(`rows scored ${rows.length} of gold ${gold.scored} (${gold.positives} positive) · unanswered ${unanswered.length} · model ${records[0]?.modelId ?? "—"}`);
    console.log("\nPREDICATE, message level (a noul: no clause comes back, so there is no span-wise number)");
    console.log(`  floor, no model          F1 ${f(floor)}`);
    console.log(`  in-browser best (paper)  F1 ${f(paper.local_best_message.F1)}`);
    console.log(`  hosted judge best (paper) F1 ${f(hostedBest)}`);
    console.log(`  at 0.5                   F1 ${f(half.f1)}  P ${f(half.precision)}  R ${f(half.recall)}  tp ${half.tp} fp ${half.fp} fn ${half.fn}`);
    console.log(`  best threshold ${f(best?.threshold)}   F1 ${f(best?.f1)}  P ${f(best?.precision)}  R ${f(best?.recall)}   <- FITTED on these rows`);
    console.log(`  split-half (honest)      F1 ${f(sh.mean)}  (tau ${f(sh.tauFromA)} -> ${f(sh.f1OnB)}, tau ${f(sh.tauFromB)} -> ${f(sh.f1OnA)})`);
    console.log(`  ranking                  ROC-AUC ${f(rocAuc(rows))}  average precision ${f(averagePrecision(rows))}`);
    console.log("\n  calibration (does 0.7 mean seven in ten?)");
    for (const b of calibration(rows)) {
      if (b.n === 0) continue;
      console.log(`    [${b.lower.toFixed(1)}, ${b.upper.toFixed(1)})  n ${String(b.n).padStart(3)}  mean p ${f(b.meanProbability)}  observed ${f(b.observedRate)}`);
    }
    console.log("\nENTITY SPANS, overlap rule (code proposes the span, the model keeps or rejects it)");
    console.log(`  tier 0 alone, no model   F1 ${f(t0.f1)}  P ${f(t0.precision)}  R ${f(t0.recall)}   prevention ${f(t0.leakPrevention)}  over-blocking ${f(t0.overBlocking)}`);
    console.log(`  best threshold ${f(entBest.threshold, 2)}      F1 ${f(entBest.f1)}  P ${f(entBest.precision)}  R ${f(entBest.recall)}   prevention ${f(entBest.leakPrevention)}  over-blocking ${f(entBest.overBlocking)}`);
    const at50 = entityPoint(ir, records, 0.5);
    console.log(`  at 0.5                   F1 ${f(at50.f1)}  P ${f(at50.precision)}  R ${f(at50.recall)}   prevention ${f(at50.leakPrevention)}  over-blocking ${f(at50.overBlocking)}`);
    console.log("\nFILTER EFFECT on tier 0's candidates (what the judgment adds over the regexes)");
    console.log(`  candidates ${fe.tier0Candidates}  on gold ${fe.onGold}  off gold ${fe.offGold}`);
    console.log(`  correct rejections ${fe.correctRejections}/${fe.offGold} (${f(fe.rejectionRateOffGold)})   wrongful rejections ${fe.wrongfulRejections}/${fe.onGold} (${f(fe.rejectionRateOnGold)})`);
    console.log("\nCOST AND TIME");
    console.log(`  $${ct.costUsd.toFixed(6)} for ${ct.answered} answered · $${f(ct.costPer1kMessages, 4)} per 1,000 messages · input tokens p50 ${f(ct.inputTokensP50, 0)} · questions/item p50 ${f(ct.questionsPerItemP50, 0)}`);
    console.log(`  per message p50 ${f(ct.itemWallMsP50, 0)} ms · p95 ${f(ct.itemWallMsP95, 0)} ms   (one request per message; no TTFT and no decode rate exist for this transport)`);

    const out = {
      file,
      dryRun: dry,
      predicateId,
      gold: { scored: gold.scored, positives: gold.positives },
      rowsScored: rows.length,
      unanswered: unanswered.length,
      floors: { noModelMessageLevel: floor, inBrowserBest: paper.local_best_message.F1, hostedJudgeBest: hostedBest },
      predicate: { at0_5: half, best, splitHalf: sh, rocAuc: rocAuc(rows), averagePrecision: averagePrecision(rows), calibration: calibration(rows) },
      entity: { tier0Baseline: t0, at0_5: at50, best: entBest, sweep: ent },
      filterEffect: fe,
      costAndTime: ct,
      sweep: points,
    };
    const outName = file.replace(/\.jsonl$/, ".score.json");
    writeFileSync(join(RUNS, outName), JSON.stringify(out, null, 2) + "\n", "utf8");
    console.log(`\nwrote runs/${outName}`);
  }
}


main();
