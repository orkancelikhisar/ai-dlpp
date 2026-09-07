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
 */
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { measureOrthography } from "../corpus/leakage.js";
import { loadCorpus } from "./corpus.js";
import { CeilingRecordSchema, type CeilingRecord } from "./ceiling.js";
import {
  MATCH_RULES,
  bestFloorF1,
  groupGoldByPredicate,
  loadRunRecords,
  loadTier2Gold,
  scoreArms,
  spansMatch,
  type MatchRule,
  type ScoreableRecord,
  type ScoredArm,
} from "./score.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");
const RUNS = join(REPO, "runs");

/**
 * `DEFAULT_TIER2_CONFIG.maxTokens` (packages/tier2/src/manifest.ts:116).
 *
 * The ceiling arms ran at 600, the browser arms at 512. The `calls >=512`
 * column below is how a reader checks whether that 88-token asymmetry could
 * have changed anything for a given arm: a 0 there means running at 512 would
 * have produced identical output.
 */
const LOCAL_ARM_MAX_TOKENS = 512;

function fmt(n: number | undefined, digits = 3): string {
  return n === undefined ? "—" : n.toFixed(digits);
}

function loadCeilingRecords(jsonl: string): CeilingRecord[] {
  const out: CeilingRecord[] = [];
  for (const line of jsonl.split("\n")) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    out.push(CeilingRecordSchema.parse(JSON.parse(trimmed)));
  }
  return out;
}

interface ArmRows {
  readonly label: string;
  readonly kind: "local" | "ceiling";
  readonly records: ScoreableRecord[];
  readonly ceiling: CeilingRecord[];
}

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
        // Keyed on runId as well as arm, so three passes are three columns and
        // never one silently-deduplicated column.
        const key = `ceiling::${r.runId}::${r.arm}`;
        const bucket = byKey.get(key) ?? {
          label:
            // A thinking-ON row is a DIFFERENT CONDITION from every other row
            // in these tables and is labelled so it can never be read as one of
            // them. Taken from the RECORD, not from the filename: the filename
            // is a naming choice, the field is what was asked of the model.
            `${r.arm} [${r.runId}]${r.thinkingRequested === "on" ? " **thinking ON**" : ""}`,
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
  const positives = gold.filter((g) => g.status === "scored" && g.satisfies).length;
  const scored = gold.filter((g) => g.status === "scored").length;
  console.log(`\n## ${title}`);
  console.log(
    `${gold.length} gold rows, ${scored} scored, ${gold.length - scored} disputed, ${positives} positives, ` +
      `${gold.reduce((n, g) => n + (g.status === "scored" ? g.spans.length : 0), 0)} gold spans`,
  );
  if (positives === 0) {
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
    let floors: ScoredArm["floors"] | undefined;
    const lines: { label: string; kind: string; f1: number | undefined; line: string }[] = [];
    for (const arm of arms.values()) {
      const scoredArm = scoreArms(arm.records, rows)[0];
      if (scoredArm === undefined) continue;
      floors ??= scoredArm.floors;
      const cells = MATCH_RULES.flatMap((rule) => {
        const s = scoredArm.byRule[rule];
        return [fmt(s.precision), fmt(s.recall), fmt(s.f1)];
      });
      lines.push({
        label: arm.label,
        kind: arm.kind,
        f1: scoredArm.byRule.overlap.f1,
        line:
          `| ${arm.label} | ${arm.kind} | ${cells.join(" | ")} | ` +
          `${scoredArm.coverage.findingsInVocabulary} | ` +
          `${scoredArm.coverage.itemsJudgeAnswered}/${scoredArm.coverage.recordsScored} |`,
      });
    }
    lines.sort((a, b) => (b.f1 ?? -1) - (a.f1 ?? -1) || a.label.localeCompare(b.label));
    for (const l of lines) console.log(l.line);
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
  }
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
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let found = 0;
    let sawAny = false;
    for (const record of arm.records) {
      const gold = goldById.get(record.itemId);
      if (gold === undefined) continue;
      const mine = record.findings.filter((f) => entityTypes.has(f.entityType));
      if (mine.length > 0) sawAny = true;
      found += mine.length;
      // Greedy one-to-one, using `score.ts`'s OWN `spansMatch` rather than a
      // second inequality written here. The two tables must not be able to
      // disagree about what a match is, and an inlined `a.start < b.end &&
      // b.start < a.end` is precisely how they would come to.
      const takenGold = new Set<number>();
      let matched = 0;
      for (const f of mine) {
        for (let g = 0; g < gold.length; g++) {
          if (takenGold.has(g)) continue;
          if (spansMatch("overlap", f, gold[g]!)) {
            takenGold.add(g);
            matched += 1;
            break;
          }
        }
      }
      tp += matched;
      fp += mine.length - matched;
      fn += gold.length - matched;
    }
    if (!sawAny) continue;
    const p = tp + fp === 0 ? undefined : tp / (tp + fp);
    const r = tp + fn === 0 ? undefined : tp / (tp + fn);
    const f1 = p === undefined || r === undefined || p + r === 0 ? undefined : (2 * p * r) / (p + r);
    rows.push({
      label: arm.label,
      f1,
      line: `| ${arm.label} | ${arm.kind} | ${found} | ${tp} | ${fp} | ${fn} | ${fmt(p)} | ${fmt(r)} | ${fmt(f1)} |`,
    });
  }
  rows.sort((a, b) => (b.f1 ?? -1) - (a.f1 ?? -1) || a.label.localeCompare(b.label));
  // SPLIT, not merely annotated. A `tier2-*` or `baselineB+tier0-*` row's spans
  // are mostly the compiled REGEX layer's, not its model's -- which is why
  // eight of them cluster at P~0.36 whatever model they name. Printed in the
  // same table they invite exactly one misreading: that a 2B browser model is
  // within 0.05 of a 120B hosted one at span finding. It is not; tier 0 is.
  const modelOnly = rows.filter((r) => !/^tier2-|\+tier0-/.test(r.label));
  const tierAssisted = rows.filter((r) => /^tier2-|\+tier0-/.test(r.label));
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
      `${600}-token \`max_tokens\` and its accuracy row reads "this model with that many answers ` +
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
    "| arm | provider (response) | pin honoured | calls | reasoning tok (p50/max) | " +
      "completion tok p50/max | **truncated (finish=length)** | calls >=512 | s @60tok/s | " +
      "TTFT p50/p95 ms | **non-stream (no TTFT)** | decode tok/s p50 | wall p50/p95 ms | " +
      "parse fail | repairs | **429s** | other retries | item wall p50 ms | cost $ |",
  );
  console.log(`|${"---|".repeat(19)}`);
  for (const arm of arms.values()) {
    if (arm.kind !== "ceiling" || arm.ceiling.length === 0) continue;
    const calls = arm.ceiling.flatMap((r) => r.calls);
    if (calls.length === 0) continue;
    const providers = [...new Set(calls.map((c) => c.provider).filter((p): p is string => p !== null))];
    const pin = arm.ceiling[0]!.requestedProvider;
    const reasoning = calls.map((c) => c.reasoningTokens).filter((n): n is number => n !== null);
    const pct = (xs: number[], q: number): number | undefined => {
      if (xs.length === 0) return undefined;
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.min(s.length - 1, Math.floor(q * s.length))];
    };
    const ttft = calls.map((c) => c.ttftMs).filter((n): n is number => n !== null);
    const decode = calls.map((c) => c.decodeTokPerSec).filter((n): n is number => n !== null);
    const wall = calls.map((c) => c.wallMs);
    const cost = calls.reduce((n, c) => n + (c.costUsd ?? 0), 0);
    const completion = calls.map((c) => c.completionTokens).filter((n): n is number => n !== null);
    const medianCompletion = pct(completion, 0.5);
    // `finish_reason: "length"` is the provider stating that IT cut the answer
    // off. An arm with a nonzero count here is not measured cleanly at this
    // cap, and its F1 is "this model with N% of its answers truncated".
    const truncated = calls.filter((c) => c.finishReason === "length").length;
    // How many calls would have been cut short at the LOCAL arms' 512 but were
    // not here. Zero means the 88-token budget asymmetry was inert for this arm
    // -- the strongest available statement that the two caps are comparable.
    const atLocalCap = completion.filter((n) => n >= LOCAL_ARM_MAX_TOKENS).length;
    // 429s are broken out from other retries because they are the ones that
    // contaminate a naive latency reading: they are the provider throttling,
    // not the model thinking. They cost nothing (a 429 is unbilled) and they
    // change no finding, so they belong beside the latency columns and not in
    // the accuracy discussion.
    const allRetries = calls.flatMap((c) => c.retries);
    const rateLimited = allRetries.filter((r) => r.status === 429).length;
    const otherRetries = allRetries.length - rateLimited;
    // The record's own top-level wallMs: end to end, backoff included. Printed
    // beside the per-attempt figures so the gap between them is visible.
    const itemWall = arm.ceiling.map((r) => r.wallMs);
    // Calls that fell back to the non-streaming path after a stream timeout.
    // They carry NO ttft and NO decode rate by construction, so they are absent
    // from those two percentiles above -- stating how many were excluded is the
    // difference between a percentile over 155 calls and one silently over 189.
    const fellBack = calls.filter((c) => c.transport === "non-stream-fallback").length;
    console.log(
      `| ${arm.label} | ${providers.join("|") || "—"} | ${providers.every((p) => p === pin) ? "yes" : "**NO**"} | ` +
        `${calls.length} | ${fmt(pct(reasoning, 0.5), 0)}/${reasoning.length ? Math.max(...reasoning) : "—"} | ` +
        `${fmt(medianCompletion, 0)}/${completion.length ? Math.max(...completion) : "—"} | ` +
        `${truncated}${truncated > 0 ? ` (**${((100 * truncated) / calls.length).toFixed(1)}%**)` : ""} | ` +
        `${atLocalCap} | ` +
        `${fmt(medianCompletion === undefined ? undefined : medianCompletion / 60, 2)} | ` +
        `${fmt(pct(ttft, 0.5), 0)}/${fmt(pct(ttft, 0.95), 0)} | ${fellBack}${fellBack > 0 ? ` (${((100 * fellBack) / calls.length).toFixed(0)}%)` : ""} | ` +
        `${fmt(pct(decode, 0.5), 1)} | ` +
        `${fmt(pct(wall, 0.5), 0)}/${fmt(pct(wall, 0.95), 0)} | ` +
        `${arm.ceiling.reduce((n, r) => n + r.parseFailures, 0)} | ` +
        `${arm.ceiling.reduce((n, r) => n + r.repairs, 0)} | ` +
        `${rateLimited} | ${otherRetries} | ${fmt(pct(itemWall, 0.5), 0)} | ${cost.toFixed(5)} |`,
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
