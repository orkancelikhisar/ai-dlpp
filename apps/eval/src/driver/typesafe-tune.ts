/**
 * Tunes the judgment arm's decision rule against the stored probabilities:
 * `pnpm -C apps/eval typesafe:tune`.
 *
 * No model is called. Every number here comes from probabilities already on
 * disk, which is the whole argument for storing them raw: the first run's
 * defaults -- one global 0.5, fire on the top option, keep every overlapping
 * finding -- turn out to cost a third of the achievable F1, and finding that out
 * costs nothing but arithmetic.
 *
 * The fitted numbers are printed beside the cross-validated ones deliberately.
 * The fitted one is what the grid search reached; the cross-validated one is
 * what a new message should expect, and it is the only one to quote.
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadIr } from "./ceiling-run.js";
import { loadTypeSafeRecords, type Thresholds } from "./typesafe.js";
import {
  blockedShare,
  crossValidateEntity,
  entityPointWith,
  loadPredicateGold,
  pairRows,
  pickThresholdMidGap,
  splitHalf,
  splitHalfWith,
  sweep,
  bestPoint,
  tuneThresholds,
  type TuneOptions,
} from "./typesafe-score-lib.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");
const RUNS = join(REPO, "runs");
const f = (n: number | null | undefined, d = 3): string => (n === null || n === undefined ? "—" : n.toFixed(d));

function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 === 1 ? s[(s.length - 1) / 2]! : ((s[s.length / 2 - 1]! + s[s.length / 2]!) / 2);
}

function main(): void {
  const files = readdirSync(RUNS).filter((n) => n.endsWith(".jsonl") && n.includes(".ts-") && !n.includes(".gates.") && !n.startsWith("ts-dryrun") && !n.startsWith("ts-probe"));
  if (files.length === 0) { console.error("no TypeSafe arm files in runs/"); process.exitCode = 1; return; }
  const { ir } = loadIr();
  const passes = files.map((file) => ({ file, records: loadTypeSafeRecords(readFileSync(join(RUNS, file), "utf8")) }));
  const predicateId = ir.semanticPredicates.find((p) => p.scope === "message")?.id ?? "";
  const gold = loadPredicateGold(readFileSync(join(REPO, "corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl"), "utf8"), predicateId);

  const VARIANTS: { name: string; t: Thresholds }[] = [
    { name: "as run: argmax, no merge", t: { predicate: 1.1, candidate: 0.5 } },
    { name: "+ merge overlaps", t: { predicate: 1.1, candidate: 0.5, mergeOverlaps: true } },
    { name: "+ confidential mass", t: { predicate: 1.1, candidate: 0.5, rule: "confidential-mass" } },
    { name: "+ both", t: { predicate: 1.1, candidate: 0.5, rule: "confidential-mass", mergeOverlaps: true } },
    { name: "+ client-name gated on the predicate", t: { predicate: 0.375, candidate: 0.5, rule: "confidential-mass", mergeOverlaps: true, gateTypesOnPredicate: ["client-name"] } },
  ];
  console.log("\nENTITY SPANS at the 0.5 default, one decision rule at a time (mean over passes)");
  for (const v of VARIANTS) {
    const pts = passes.map((p) => entityPointWith(ir, p.records, v.t));
    const mean = (pick: (x: (typeof pts)[number]) => number): number => pts.reduce((a, x) => a + pick(x), 0) / pts.length;
    const blocked = passes.map((p) => blockedShare(ir, p.records, v.t).share).reduce((a, b) => a + b, 0) / passes.length;
    console.log(`  ${v.name.padEnd(38)} F1 ${f(mean((x) => x.f1))}  prevention ${f(mean((x) => x.leakPrevention))}  clean touched ${f(mean((x) => x.overBlocking))}  clean blocked ${f(blocked)}`);
  }

  const opts: TuneOptions = { rule: "confidential-mass", mergeOverlaps: true, base: 0.5 };
  console.log("\nPER-TYPE THRESHOLDS, tuned by coordinate ascent");
  const tuned = passes.map((p) => ({ ...p, perType: tuneThresholds(ir, p.records, opts) }));
  for (const p of tuned) {
    const fit = entityPointWith(ir, p.records, { predicate: 1.1, candidate: 0.5, perType: p.perType, ...opts });
    console.log(`  ${p.file.split(".")[0]}: fitted F1 ${f(fit.f1)}   ${Object.entries(p.perType).map(([k, v]) => `${k.replace(/-/g, "")}=${v}`).join(" ")}`);
  }
  const shipped: Record<string, number> = {};
  for (const type of Object.keys(tuned[0]!.perType)) shipped[type] = median(tuned.map((p) => p.perType[type] ?? 0.5));
  console.log(`  shipped (median over passes): ${Object.entries(shipped).map(([k, v]) => `${k}=${v}`).join(", ")}`);

  console.log("\nCROSS-VALIDATED, five folds of MESSAGES: every message scored by thresholds tuned without it");
  const cvs = passes.map((p) => ({ file: p.file, cv: crossValidateEntity(ir, p.records, opts, 5) }));
  for (const c of cvs) {
    const x = c.cv.point;
    console.log(`  ${c.file.split(".")[0]}: F1 ${f(x.f1)}  P ${f(x.precision)}  R ${f(x.recall)}   prevention ${f(x.leakPrevention)}  over-blocking ${f(x.overBlocking)}`);
  }
  const pooledF1 = cvs.reduce((a, c) => a + c.cv.point.f1, 0) / cvs.length;
  const baseline = passes.map((p) => entityPointWith(ir, p.records, VARIANTS[0]!.t).f1).reduce((a, b) => a + b, 0) / passes.length;
  console.log(`  mean cross-validated F1 ${f(pooledF1)} against ${f(baseline)} as run: ${f(pooledF1 - baseline)} of F1 for no extra call`);
  const shippedPts = passes.map((p) => entityPointWith(ir, p.records, { predicate: 1.1, candidate: 0.5, perType: shipped, ...opts }));
  console.log(`  shipped thresholds, per pass: F1 ${shippedPts.map((x) => f(x.f1)).join(" / ")}  prevention ${shippedPts.map((x) => f(x.leakPrevention)).join(" / ")}  over-blocking ${shippedPts.map((x) => f(x.overBlocking)).join(" / ")}`);

  console.log("\nPREDICATE: where to put the threshold");
  const perPass = passes.map((p) => {
    const { rows } = pairRows(p.records, gold);
    return { file: p.file, rows, argmax: splitHalf(rows), midgap: splitHalfWith(rows, pickThresholdMidGap), best: bestPoint(sweep(rows))?.threshold ?? null, mid: pickThresholdMidGap(rows) };
  });
  for (const p of perPass) {
    console.log(`  ${p.file.split(".")[0]}: split-half F1 ${f(p.argmax.mean)} at the best observed value, ${f(p.midgap.mean)} at the gap midpoint (best ${f(p.best, 2)}, midpoint ${f(p.mid, 2)})`);
  }
  const shippedPredicate = median(perPass.map((p) => p.mid ?? 0.5));
  console.log(`  shipped predicate threshold (median gap midpoint): ${f(shippedPredicate, 3)}`);

  const out = {
    generated: new Date().toISOString().slice(0, 10),
    passes: passes.map((p) => p.file),
    rule: opts,
    entity: {
      asRun: passes.map((p) => entityPointWith(ir, p.records, VARIANTS[0]!.t)),
      variantsAtHalf: VARIANTS.map((v) => ({ name: v.name, points: passes.map((p) => entityPointWith(ir, p.records, v.t)) })),
      tunedPerPass: tuned.map((p) => ({ file: p.file, thresholds: p.perType })),
      crossValidated: cvs.map((c) => ({ file: c.file, point: c.cv.point, folds: c.cv.perFold })),
      shipped: { thresholds: shipped, points: shippedPts },
    },
    predicate: { shippedThreshold: shippedPredicate, perPass: perPass.map((p) => ({ file: p.file, argmax: p.argmax, midgap: p.midgap, best: p.best, midpoint: p.mid })) },
  };
  writeFileSync(join(RUNS, "typesafe-tuning.json"), JSON.stringify(out, null, 2) + "\n", "utf8");
  console.log("\nwrote runs/typesafe-tuning.json");
}

main();
