/**
 * Scores the capability-ceiling arms beside the local ones, in ONE table, with
 * the floors in it: `pnpm -C apps/eval ceiling:score`.
 *
 * Everything here is read from run artifacts. Nothing is typed in from memory,
 * and nothing is recomputed by a second implementation -- the arms and the
 * floors go through `score.ts`'s own `scoreArms`, and the span-level
 * orthographic oracle goes through `leakage.ts`'s own `measureOrthography`. The
 * point of the exercise is that a reader can re-run this and get the table in
 * the write-up.
 *
 * Two levels are reported because the two questions are different:
 *
 *   - PREDICATE level, against both tier-2 golds, three match rules, with the
 *     capitalised-multiword floors `scoreArm` computes over the same rows.
 *   - SPAN level, against the labelled corpus's ENTITY gold, with the
 *     budget-matched orthographic oracle as the floor. Only the Approach-B
 *     family has entity findings to score there; the judge family emits the
 *     shadow predicate alone, by construction.
 *
 * Passes are scored SEPARATELY and never pooled. `scoreArm` refuses two records
 * for one item on one arm -- rightly, since one would be silently dropped -- and
 * pooling three repeats of the same arm is exactly that. Variance across passes
 * is reported as the spread of the per-pass numbers.
 *
 * THIS FILE RENDERS; it decides nothing. Every count, ratio, threshold and
 * pairing lives in `ceiling-score-lib.ts`, which is importable and tested. It
 * was not always so: this file inlined all of it and ended in a bare `main()`,
 * exporting nothing, so a mutation review found the whole file uncovered.
 */
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "./corpus.js";
import { MATCH_RULES, groupGoldByPredicate, loadRunRecords, loadTier2Gold, type MatchRule } from "./score.js";
import {
  CEILING_ARM_MAX_TOKENS,
  LOCAL_ARM_MAX_TOKENS,
  MIN_DECODE_WINDOW_MS,
  attemptedOnlyTable,
  bestFloorF1,
  ceilingArmKey,
  ceilingArmLabel,
  fmt,
  isTierAssisted,
  loadCeilingRecords,
  measureOrthography,
  scorePredicate,
  scoreSpanArm,
  summarizeGold,
  transportRow,
  type ArmRows,
} from "./ceiling-score-lib.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");
const RUNS = join(REPO, "runs");

function collect(): Map<string, ArmRows> {
  const byKey = new Map<string, ArmRows>();
  for (const file of readdirSync(RUNS).sort()) {
    if (!file.endsWith(".jsonl") || file.endsWith(".gates.jsonl")) continue;
    const jsonl = readFileSync(join(RUNS, file), "utf8");
    if (file.startsWith("slate-rebuild-")) {
      for (const r of loadRunRecords(jsonl)) {
        const key = `local::${r.arm}`;
        const bucket = byKey.get(key) ?? { label: r.arm, kind: "local" as const, records: [], ceiling: [] };
        bucket.records.push(r);
        byKey.set(key, bucket);
      }
    } else if (file.includes(".ceiling-")) {
      // Matched on the ARM name inside the filename rather than on a run-id
      // prefix, so a run under any id (`ceiling-01`, `glmon-01`) is picked up.
      for (const r of loadCeilingRecords(jsonl)) {
        const key = ceilingArmKey(r);
        const bucket = byKey.get(key) ?? {
          label: ceilingArmLabel(r),
          kind: "ceiling" as const,
          records: [],
          ceiling: [],
        };
        bucket.records.push(r);
        bucket.ceiling.push(r);
        byKey.set(key, bucket);
      }
    }
  }
  return byKey;
}

function predicateTable(title: string, goldPath: string, arms: Map<string, ArmRows>): void {
  const gold = loadTier2Gold(readFileSync(goldPath, "utf8"));
  const groups = groupGoldByPredicate(gold);
  const summary = summarizeGold(gold);
  console.log(`\n## ${title}`);
  console.log(
    `${summary.rows} gold rows, ${summary.scored} scored, ${summary.disputed} disputed, ` +
      `${summary.positives} positives, ${summary.goldSpans} gold spans`,
  );
  if (summary.positives === 0) {
    console.log(
      "NOTE: this gold carries ZERO positives and ZERO gold spans, so tp = fn = 0 for every arm and " +
        "RECALL AND F1 ARE UNDEFINED for all of them under every rule. It can only measure precision " +
        "(0.000 for any arm that fired, undefined for one that did not). It cannot rank anything.",
    );
  }

  for (const [predicateId, rows] of groups) {
    console.log(`\npredicate: ${predicateId}`);
    console.log(
      "| arm | kind | " +
        MATCH_RULES.map((r) => `${r} P | ${r} R | ${r} F1`).join(" | ") +
        " | inVocab | answered/scored |",
    );
    console.log(`|${"---|".repeat(4 + MATCH_RULES.length * 3)}`);
    const { lines, floors } = scorePredicate(arms.values(), rows);
    for (const l of lines) {
      const cells = MATCH_RULES.flatMap((rule) => {
        const s = l.scored.byRule[rule];
        return [fmt(s.precision), fmt(s.recall), fmt(s.f1)];
      });
      console.log(
        `| ${l.label} | ${l.kind} | ${cells.join(" | ")} | ` +
          `${l.scored.coverage.findingsInVocabulary} | ` +
          `${l.scored.coverage.itemsJudgeAnswered}/${l.scored.coverage.recordsScored} |`,
      );
    }
    if (floors !== undefined) {
      for (const floor of floors) {
        const cells = MATCH_RULES.flatMap((rule) => {
          const s = floor.byRule[rule];
          return [fmt(s.precision), fmt(s.recall), fmt(s.f1)];
        });
        console.log(`| **FLOOR ${floor.reader}** | floor | ${cells.join(" | ")} | ${floor.findings} | — |`);
      }
      for (const rule of MATCH_RULES) {
        const best = bestFloorF1(floors, rule);
        const beat = lines.filter((l) => l.f1 !== undefined && best !== undefined && (l.f1 ?? 0) > best);
        console.log(
          `\nunder ${rule}: best floor F1 ${fmt(best)}; arms beating it: ${beat.length === 0 ? "NONE" : beat.map((b) => b.label).join(", ")}`,
        );
      }
    }
    attemptedOnlySection(arms, rows, predicateId);
  }
}

/**
 * The attempted-only table, printed BESIDE the whole-gold one above and never
 * instead of it.
 *
 * 68 of the 3,780 rows across passes 1-2 carry `calls: []`, `provider: null`
 * and a 429: the provider's rate limiter refused the item and the model never
 * saw it. Scored against the whole gold, such a row is indistinguishable from
 * an item the model read and missed, and the losses are very uneven across arms
 * -- 25 on one, 0 on thirteen.
 *
 * The floor moves with the arm here. Each row's `attempted floor` is
 * `scoreTrivialFloors` re-run over exactly the subset that arm answered, so the
 * comparison is like-for-like; `docs/research/2026-09-07-ceiling-arm.md` Sec
 * 7.4 gives quick figures that adjust the arm against an UNADJUSTED floor and
 * says so. These are the corrected ones.
 */
function attemptedOnlySection(arms: Map<string, ArmRows>, rows: Parameters<typeof attemptedOnlyTable>[1], predicateId: string): void {
  const rule: MatchRule = "overlap";
  const table = attemptedOnlyTable(arms.values(), rows, rule);
  const affected = table.filter((r) => r.unanswered > 0);
  console.log(
    `\n### ATTEMPTED-ONLY — ${predicateId}, ${rule} rule. Beside the table above, never instead of it.`,
  );
  console.log(
    `An item the provider rate-limited away scores identically to one the model read and missed. Here ` +
      `each arm AND every floor is re-scored over the intersection of items that arm answered, so the ` +
      `comparison is like-for-like. The unanswered rows are a real cost of a rate-limited hosted ` +
      `provider and are counted, not defined away.`,
  );
  if (affected.length === 0) {
    console.log(`\nevery arm answered every scored gold row under this predicate; the two scorings coincide.`);
    return;
  }
  console.log(
    "\n| arm | kind | unanswered | of those, gold-positive | whole-gold F1 | whole-gold best floor | " +
      "attempted-only F1 | attempted-only best floor | F1 delta |",
  );
  console.log(`|${"---|".repeat(9)}`);
  for (const r of affected) {
    const delta =
      r.attemptedF1 === undefined || r.wholeGoldF1 === undefined ? undefined : r.attemptedF1 - r.wholeGoldF1;
    console.log(
      `| ${r.label} | ${r.kind} | ${r.unanswered} | ${r.unansweredPositives} | ${fmt(r.wholeGoldF1)} | ` +
        `${fmt(r.wholeGoldBestFloorF1)} | ${fmt(r.attemptedF1)} | ${fmt(r.attemptedBestFloorF1)} | ` +
        `${delta === undefined ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`} |`,
    );
  }
  const flips = affected.filter(
    (r) =>
      r.wholeGoldF1 !== undefined &&
      r.attemptedF1 !== undefined &&
      r.wholeGoldBestFloorF1 !== undefined &&
      r.attemptedBestFloorF1 !== undefined &&
      r.wholeGoldF1 > r.wholeGoldBestFloorF1 !== r.attemptedF1 > r.attemptedBestFloorF1,
  );
  console.log(
    `\narms whose verdict against the floor CHANGES under attempted-only scoring: ` +
      `${flips.length === 0 ? "NONE" : flips.map((f) => f.label).join(", ")}`,
  );
}

/**
 * The SPAN-level table: entity findings against the labelled corpus's own gold,
 * with the budget-matched orthographic oracle beside them.
 *
 * Scored here rather than through `scoreArm` because that function takes a
 * `Tier2GoldRow` -- a predicate row -- and the entity gold lives on the corpus
 * item. The per-span predicate is `score.ts`'s OWN exported `spansMatch`, so the
 * two tables cannot hold two definitions of a match.
 *
 * The PAIRING here is greedy first-fit, where `scoreArm` runs a maximum
 * -cardinality matching. On this gold that cannot differ: every gold-carrying
 * item in the labelled corpus has exactly ONE entity gold span (108 spans over
 * 108 items), so there is nothing for a maximum matching to improve on. Stated
 * rather than assumed, because on a multi-span gold the two WOULD diverge and
 * this function would need `maximumMatching`, which `score.ts` keeps private.
 */
function spanTable(arms: Map<string, ArmRows>): void {
  const items = loadCorpus(readFileSync(join(REPO, "corpora/generated/injection-p-fin-v2.labelled.jsonl"), "utf8"));
  const goldById = new Map(items.map((i) => [i.id, i.gold]));
  const entityTypes = new Set(items.flatMap((i) => i.gold.map((g) => g.entityType)));
  const orth = measureOrthography(items);

  console.log(`\n## SPAN level — entity gold from the labelled corpus`);
  console.log(
    `${items.length} items, ${[...goldById.values()].reduce((n, g) => n + g.length, 0)} gold ENTITY spans, ` +
      `types: ${[...entityTypes].sort().join(", ")}`,
  );
  console.log(
    `\nFLOOR — orthographic oracle, budget-matched (first N hits in document order, N = the item's own ` +
      `gold count): P ${fmt(orth.budgetMatched.precision)} R ${fmt(orth.budgetMatched.recall)} ` +
      `F1 ${fmt(orth.budgetMatched.f1)}`,
  );
  console.log(
    `FLOOR — orthographic oracle, unbudgeted (every hit): P ${fmt(orth.asScored.precision)} ` +
      `R ${fmt(orth.asScored.recall)} F1 ${fmt(orth.asScored.f1)}`,
  );
  console.log(
    `\nNOTE: only the Approach-B family emits entity findings. The judge family emits the shadow ` +
      `predicate alone by construction, so it has no span score and is omitted rather than shown as 0.`,
  );
  console.log(
    `NOTE, AND IT DECIDES HOW THIS TABLE READS: a local arm whose name is \`tier2-*\` or ` +
      `\`baselineB+tier0-*\` RAN TIER 0, so its entity spans are mostly the compiled regex layer's ` +
      `and not its model's -- which is why those eight rows cluster at P~0.36 R~0.65 whatever model ` +
      `they name. The MODEL-ONLY rows are \`baselineB-*\` (no tier 0) and every \`ceiling-b-*\` row, ` +
      `which runs no tier 0 either. Compare ceiling-b against baselineB-*, never against tier2-*.`,
  );
  console.log("\n| arm | kind | findings | tp | fp | fn | P | R | F1 |");
  console.log(`|${"---|".repeat(9)}`);

  const rows: { label: string; f1: number | undefined; line: string }[] = [];
  for (const arm of arms.values()) {
    const row = scoreSpanArm(arm, goldById, entityTypes);
    if (row === undefined) continue;
    const { tp, fp, fn } = row.counts;
    const { precision, recall, f1 } = row.scores;
    rows.push({
      label: row.label,
      f1,
      line:
        `| ${row.label} | ${row.kind} | ${row.findings} | ${tp} | ${fp} | ${fn} | ` +
        `${fmt(precision)} | ${fmt(recall)} | ${fmt(f1)} |`,
    });
  }
  rows.sort((a, b) => (b.f1 ?? -1) - (a.f1 ?? -1) || a.label.localeCompare(b.label));
  // SPLIT, not merely annotated. A `tier2-*` or `baselineB+tier0-*` row's spans
  // are mostly the compiled REGEX layer's, not its model's -- which is why
  // eight of them cluster at P~0.36 whatever model they name. Printed in the
  // same table they invite exactly one misreading: that a 2B browser model is
  // within 0.05 of a 120B hosted one at span finding. It is not; tier 0 is.
  const modelOnly = rows.filter((r) => !isTierAssisted(r.label));
  const tierAssisted = rows.filter((r) => isTierAssisted(r.label));
  for (const row of modelOnly) console.log(row.line);
  if (modelOnly.length === 0) console.log("| (no arm emitted an entity finding) | | | | | | | | |");
  if (tierAssisted.length > 0) {
    console.log(
      `\nEXCLUDED from the table above -- these ${tierAssisted.length} local arms RAN TIER 0, so their ` +
        `spans are the compiled regex layer's and not their model's. Listed for completeness, and not ` +
        `comparable with any row above:`,
    );
    console.log("\n| arm (tier-0 assisted, NOT model-only) | kind | findings | tp | fp | fn | P | R | F1 |");
    console.log(`|${"---|".repeat(9)}`);
    for (const row of tierAssisted) console.log(row.line);
  }
}

/** Per-model transport facts: what was asked, what happened, and what it cost. */
function transportTable(arms: Map<string, ArmRows>): void {
  console.log(`\n## Transport — reasoning, latency, structured-output compliance`);
  console.log(
    `\nLATENCY, AND WHICH CLOCK: \`TTFT\`, \`decode\` and \`wall\` time the SUCCESSFUL ATTEMPT ONLY -- ` +
      `\`callChat\` restarts its clock inside the retry loop, so backoff sleeps and failed attempts ` +
      `are excluded (proved by a test, not assumed). \`item wall\` is the record's own top-level ` +
      `figure: end to end, every retry and backoff included. The \`429s\` column says how much ` +
      `throttling stood between them -- a 429 is unbilled and changes no finding, but a latency ` +
      `column that silently absorbed one would be a fact about the aggregator wearing the model's name. ` +
      `\`non-stream\` counts calls whose stream hung and were retried without streaming: they have NO ` +
      `TTFT and NO decode rate by construction and are EXCLUDED from those two percentiles, so a large ` +
      `count there means the latency columns describe a subset of the arm.`,
  );
  console.log(
    `\nTRUNCATION FIRST: an arm with a nonzero \`finish=length\` count was cut off by the ` +
      `${CEILING_ARM_MAX_TOKENS}-token \`max_tokens\` and its accuracy row reads "this model with that many answers ` +
      `truncated", not "this model". \`calls >=${LOCAL_ARM_MAX_TOKENS}\` counts calls that would ` +
      `ALSO have been cut at the browser arms' ${LOCAL_ARM_MAX_TOKENS}; a 0 there means the ` +
      `88-token budget asymmetry between the two experiments was inert for this arm.`,
  );
  console.log(
    "\nThe two columns that TRANSFER to other hardware are `reasoning tok` and `completion tok`: " +
      "they are properties of the model and the task. TTFT and decode rate are PROVIDER FACTS -- they " +
      "describe whose GPU answered, under what load, behind which aggregator, and say nothing about " +
      "what the same weights would do elsewhere. `s @60tok/s` projects the median completion onto a " +
      "60 tok/s decode budget (a DGX Spark-class figure for a model of this size); it is arithmetic " +
      "over a measured token count, NOT a measurement, and it is labelled so.",
  );
  console.log(
    `\nDECODE RATE, AND WHEN IT IS NOT ONE: the \`decode window p50\` column beside the rate is the ` +
      `median of \`wallMs - ttftMs\`, the interval the rate is measured over. Where that median is ` +
      `under ${MIN_DECODE_WINDOW_MS} ms the rate is SUPPRESSED and the cell reads \`not measured\`: at ` +
      `that scale the figure describes when a server-sent-event frame happened to arrive, not how fast ` +
      `the model decodes. Measured, not assumed -- \`judge-deepseek-v4-flash-0731 [ceiling-01]\` has a ` +
      `median window of 9.6 ms, 145 of its 185 streaming calls under 20 ms, a shortest window of ` +
      `0.174 ms and a largest computed rate of 40,327 tok/s for a seven-token answer. A \`(bias N%)\` ` +
      `marker is SEPARATE and means something else: a rate over n completion tokens rests on n-1 token ` +
      `intervals, so one mis-timed interval moves it by 1/(n-1) -- and on every row now in \`runs/\`, ` +
      `written before \`ceiling.ts\` was corrected on 2026-09-08, that is also the exact amount the old ` +
      `n/window formula overstated it by. The two guards are ` +
      `independent -- \`judge-nemotron\` has a sound window and a 16.7% bias, \`judge-glm-5.3-flash\` a ` +
      `181.5 ms window and a 0.5% one -- so neither substitutes for the other.`,
  );
  console.log(
    "| arm | provider (response) | pin honoured | calls | reasoning tok (p50/max) | " +
      "completion tok p50/max | **truncated (finish=length)** | calls >=512 | s @60tok/s | " +
      "TTFT p50/p95 ms | **non-stream (no TTFT)** | decode tok/s p50 | **decode window p50 ms** | " +
      "wall p50/p95 ms | parse fail | repairs | **429s** | other retries | item wall p50 ms | cost $ |",
  );
  console.log(`|${"---|".repeat(20)}`);
  for (const arm of arms.values()) {
    if (arm.kind !== "ceiling") continue;
    const t = transportRow(arm);
    if (t === undefined) continue;
    const d = t.decode;
    const decodeCell = d.suppressed
      ? `**not measured**`
      : `${fmt(d.tokPerSecP50, 1)}${d.biased ? ` (bias ${(100 * d.biasAtMedianN!).toFixed(1)}%)` : ""}`;
    const windowCell =
      d.windowMsP50 === undefined
        ? "—"
        : `${d.windowMsP50.toFixed(1)}${d.shortWindowCalls > 0 ? ` (${d.shortWindowCalls}/${d.streamingCalls} <${MIN_DECODE_WINDOW_MS / 5}ms)` : ""}`;
    console.log(
      `| ${t.label} | ${t.providers.join("|") || "—"} | ${t.pinHonoured ? "yes" : "**NO**"} | ` +
        `${t.calls} | ${fmt(t.reasoningP50, 0)}/${t.reasoningMax ?? "—"} | ` +
        `${fmt(t.completionP50, 0)}/${t.completionMax ?? "—"} | ` +
        `${t.truncated}${t.truncated > 0 ? ` (**${((100 * t.truncated) / t.calls).toFixed(1)}%**)` : ""} | ` +
        `${t.atLocalCap} | ` +
        `${fmt(t.completionP50 === undefined ? undefined : t.completionP50 / 60, 2)} | ` +
        `${fmt(t.ttftP50, 0)}/${fmt(t.ttftP95, 0)} | ${t.fellBack}${t.fellBack > 0 ? ` (${((100 * t.fellBack) / t.calls).toFixed(0)}%)` : ""} | ` +
        `${decodeCell} | ${windowCell} | ` +
        `${fmt(t.wallP50, 0)}/${fmt(t.wallP95, 0)} | ` +
        `${t.parseFailures} | ${t.repairs} | ${t.rateLimited} | ${t.otherRetries} | ` +
        `${fmt(t.itemWallP50, 0)} | ${t.costUsd.toFixed(5)} |`,
    );
  }
}

function main(): void {
  const arms = collect();
  const ceilingArms = [...arms.values()].filter((a) => a.kind === "ceiling").length;
  const localArms = [...arms.values()].filter((a) => a.kind === "local").length;
  console.log(`# Ceiling vs local — every number read from runs/`);
  console.log(`${localArms} local arms, ${ceilingArms} ceiling arm-passes`);
  const shas = new Set(
    [...arms.values()].flatMap((a) => a.ceiling.map((r) => `${r.gitSha.slice(0, 12)}${r.gitDirty ? "+dirty" : ""}`)),
  );
  if (shas.size > 0) console.log(`ceiling rows were produced at git ${[...shas].join(", ")}`);

  predicateTable(
    "PREDICATE level — gold-tier2-predicate (189 rows, the full blind round)",
    join(REPO, "corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl"),
    arms,
  );
  predicateTable(
    "PREDICATE level — gold-tier2 (20 rows, the earlier contested-span round)",
    join(REPO, "corpora/generated/injection-p-fin-v2.gold-tier2.jsonl"),
    arms,
  );
  spanTable(arms);
  transportTable(arms);
}

main();
