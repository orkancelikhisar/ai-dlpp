/**
 * The capability-ceiling arm's entry point: `pnpm -C apps/eval ceiling`.
 *
 * Reads `OPENROUTER_API_KEY` from the ENVIRONMENT and nowhere else. The key is
 * never printed, never written to a run artifact, never placed on a record and
 * never included in an error message -- `callChat` puts it in one header and
 * that header is the only place it exists in this process.
 *
 * Environment:
 *   OPENROUTER_API_KEY   required
 *   SIH_CEILING_RUN_ID   run id; default "ceiling-01"
 *   SIH_CEILING_PASSES   how many passes to attempt; default 1
 *   SIH_CEILING_PROBE    "1" to run the 3-item probe and stop
 *   SIH_CEILING_LIMIT    cap items per arm (probe/debug); default the whole corpus
 *   SIH_CEILING_MODELS   comma-separated OpenRouter ids to restrict the slate to
 *   SIH_CEILING_THINKING "off" (default, the experiment's condition) or "on".
 *                        "on" exists only for z-ai/glm-5.3-flash, which refuses
 *                        every thinking-off request on all eleven of its
 *                        providers. Rows from an "on" run carry
 *                        thinkingRequested: "on" and are a DIFFERENT CONDITION
 *                        -- never pool them with the off arms.
 *   SIH_CEILING_PIN      re-pin models, e.g. "z-ai/glm-5.3-flash=NextBit".
 *                        Changes what is ASKED; the response still decides what
 *                        `provider` the record carries.
 *   SIH_CEILING_PASS_START  1-based number of the FIRST pass to write, so a
 *                        later pass can be run on its own without overwriting
 *                        an earlier one's files. See passRunIdFor.
 *
 * The order of operations is chosen so that the expensive thing never happens
 * before the cheap check that would have stopped it: read the key's balance,
 * then probe three items per model per family, then run the corpus
 * cheapest-model-first. The spend guard is consulted before every call.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPolicyIr, type PolicyIr } from "@sih/core";
import {
  CEILING_MODELS,
  applyPinOverrides,
  REPRESENTATIVE_COMPLETION_TOKENS,
  REPRESENTATIVE_PROMPT_TOKENS,
  SPEND_HARD_STOP_USD,
  SpendGuard,
  armName,
  estimateCostUsd,
  mapWithConcurrency,
  passRunIdFor,
  runCeilingItem,
  toJsonl,
  type ThinkingRequest,
  type CeilingFamily,
  type CeilingModel,
  type CeilingRecord,
  type ClientDeps,
  type ItemInput,
} from "./ceiling.js";
import { loadCorpus } from "./corpus.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");

/** Concurrency 3, as briefed: enough to make the run finish, few enough to stay under rate limits. */
const CONCURRENCY = 3;
/** Items per family per model in the probe. */
const PROBE_ITEMS = 3;
/**
 * The completion ceiling per call.
 *
 * ## This is NOT the local arms' value, and the comment that said it was, lied
 *
 * An earlier version of this docblock read "The local arms run at 600
 * (`DEFAULT_TIER2_CONFIG`). Kept the same here so a truncation means the same
 * thing." That was false and nobody had read the constant: MEASURED at
 * `packages/tier2/src/manifest.ts:116`, `DEFAULT_TIER2_CONFIG.maxTokens` is
 * **512**, and its own docblock at :81 says it is "UNCHANGED at 512 through the
 * two-span schema change". So the hosted arms ran with **88 more completion
 * tokens** than the browser arms.
 *
 * ## Which way it cuts, and where it is inert
 *
 * It FAVOURS the ceiling arms: more room to finish an answer. That is the
 * conservative direction for this experiment's headline -- the ceiling arms had
 * the advantage and still did not beat the trivial floor -- and the flattering
 * direction for any local-vs-ceiling gap, which must be read with that in mind.
 *
 * It is also MEASURABLY INERT for every thinking-off arm, which is a stronger
 * statement than arguing about the direction. Counted over every call written
 * so far: the judge family's completions top out at 103 tokens (p95 61) and
 * Approach B's at 455 (p95 263). **Not one thinking-off call reached 512**, so
 * running at 512 would have produced byte-identical output. The 88 tokens only
 * ever mattered for the thinking-ON arm, where 55 of 216 calls exceeded 512 and
 * 48 hit 600 exactly. `ceiling-score.ts` prints the per-arm `>=512` and
 * `finish=length` counts so this stays checkable rather than asserted.
 *
 * NOT changed to 512 mid-experiment: the paid run was already in flight, and a
 * slate half-measured at each value would be worse than one measured at a
 * documented 600. Every row records the value it ran at.
 *
 * A thinking-ON phase cannot be measured at either cap -- see the doc -- and
 * needs a uniformly larger ceiling (8,192) applied to every model.
 */
const MAX_TOKENS = 600;

/** `DEFAULT_TIER2_CONFIG.maxTokens`, restated so the asymmetry above is checkable here. */
const LOCAL_ARM_MAX_TOKENS = 512;

function env(name: string): string | undefined {
  const v = process.env[name];
  return v === undefined || v === "" ? undefined : v;
}

function gitProvenance(): { gitSha: string; gitDirty: boolean } {
  const gitSha = execFileSync("git", ["rev-parse", "HEAD"], { cwd: REPO, encoding: "utf8" }).trim();
  const status = execFileSync("git", ["status", "--porcelain"], { cwd: REPO, encoding: "utf8" });
  return { gitSha, gitDirty: status.trim().length > 0 };
}

interface KeyState {
  usage: number;
  limit: number | null;
}

async function readKeyState(apiKey: string): Promise<KeyState> {
  const response = await fetch("https://openrouter.ai/api/v1/auth/key", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!response.ok) {
    // The status, never the body: an auth error body can echo the key prefix.
    throw new Error(`/auth/key returned ${response.status}`);
  }
  const body = (await response.json()) as { data?: { usage?: unknown; limit?: unknown } };
  const usage = typeof body.data?.usage === "number" ? body.data.usage : 0;
  const limit = typeof body.data?.limit === "number" ? body.data.limit : null;
  return { usage, limit };
}

function loadIr(): { ir: PolicyIr; irHash: string } {
  const path = join(REPO, "policies/compiled/p-fin.ir.json");
  const bytes = readFileSync(path);
  // Hashed from the ARTIFACT's own bytes, exactly as the page does it, so
  // `shasum -a 256` on the file reproduces what the record claims.
  const irHash = createHash("sha256").update(bytes).digest("hex");
  // The repo's own loader, not a bare schema parse: it is what narrows
  // `irVersion` and what every other IR reader here goes through, so a
  // divergence in validation between arms is impossible.
  const ir = loadPolicyIr(bytes.toString("utf8"));
  return { ir, irHash };
}

function loadItems(): ItemInput[] {
  const jsonl = readFileSync(join(REPO, "corpora/generated/injection-p-fin-v2.labelled.jsonl"), "utf8");
  return loadCorpus(jsonl).map((item) => ({
    id: item.id,
    text: item.text,
    policy: item.policy,
    gold: item.gold.map((g) => ({
      start: g.start,
      end: g.end,
      text: g.text,
      entityType: g.entityType,
      action: g.action,
    })),
  }));
}

const RUNS = join(REPO, "runs");

function writeSpend(runId: string, guard: SpendGuard, extra: Record<string, unknown>): void {
  mkdirSync(RUNS, { recursive: true });
  writeFileSync(
    join(RUNS, `ceiling-${runId}.spend.json`),
    `${JSON.stringify({ ...guard.snapshot(), ...extra }, null, 2)}\n`,
  );
}

interface ProbeResult {
  readonly model: string;
  readonly family: CeilingFamily;
  readonly attempted: number;
  readonly parsed: number;
  readonly providersSeen: string[];
  readonly providerMatchesPin: boolean;
  readonly reasoningTokens: (number | null)[];
  readonly reasoningAllZero: boolean;
  readonly ttftMs: (number | null)[];
  readonly wallMs: number[];
  readonly errors: string[];
  readonly verdict: "pass" | "skip";
}

function summarizeProbe(
  model: CeilingModel,
  family: CeilingFamily,
  records: readonly CeilingRecord[],
): ProbeResult {
  const calls = records.flatMap((r) => r.calls);
  const providersSeen = [...new Set(calls.map((c) => c.provider).filter((p): p is string => p !== null))];
  const reasoning = calls.map((c) => c.reasoningTokens);
  const errors = records.map((r) => r.error).filter((e): e is string => e !== null);
  const parsed = records.filter((r) => r.error === null && r.calls.some((c) => c.parse === "ok")).length;
  return {
    model: model.id,
    family,
    attempted: records.length,
    parsed,
    providersSeen,
    // A pin is a request; this compares it with the fact. A mismatch does not
    // stop the arm -- the row records both -- but it is reported loudly,
    // because the latency columns would then be about routing.
    providerMatchesPin: providersSeen.length > 0 && providersSeen.every((p) => p === model.provider),
    reasoningTokens: reasoning,
    // "All zero" only counts MEASURED zeros. A null is "the provider reported
    // no usage", which is not evidence that thinking was off.
    reasoningAllZero: reasoning.length > 0 && reasoning.every((r) => r === 0),
    ttftMs: calls.map((c) => c.ttftMs),
    wallMs: calls.map((c) => c.wallMs),
    errors,
    // A model whose probe parses NOTHING is skipped, not retried into the
    // budget. One or two failures out of three is a model that is answering,
    // and the full run's own counters are where that gets measured.
    verdict: parsed > 0 ? "pass" : "skip",
  };
}

async function main(): Promise<void> {
  const apiKey = env("OPENROUTER_API_KEY");
  if (apiKey === undefined) {
    throw new Error("OPENROUTER_API_KEY is not set; the ceiling arm reads its key from the environment only");
  }
  const runId = env("SIH_CEILING_RUN_ID") ?? "ceiling-01";
  const passes = Number(env("SIH_CEILING_PASSES") ?? "1");
  const probeOnly = env("SIH_CEILING_PROBE") === "1";
  const limit = env("SIH_CEILING_LIMIT") === undefined ? undefined : Number(env("SIH_CEILING_LIMIT"));
  const only = env("SIH_CEILING_MODELS")?.split(",").map((s) => s.trim());
  const thinking: ThinkingRequest = env("SIH_CEILING_THINKING") === "on" ? "on" : "off";
  const passStart = Number(env("SIH_CEILING_PASS_START") ?? "1");
  const repinned = applyPinOverrides(CEILING_MODELS, env("SIH_CEILING_PIN"));
  const slate = only === undefined ? repinned : repinned.filter((m) => only.includes(m.id));
  if (slate.length === 0) throw new Error(`SIH_CEILING_MODELS matched no model on the slate`);

  const { gitSha, gitDirty } = gitProvenance();
  const { ir, irHash } = loadIr();
  const policyText = readFileSync(join(REPO, "policies/p-fin.md"), "utf8");
  const allItems = loadItems();
  const items = limit === undefined ? allItems : allItems.slice(0, limit);

  const keyAtStart = await readKeyState(apiKey);
  const guard = new SpendGuard({
    hardStopUsd: SPEND_HARD_STOP_USD,
    keyUsageAtStart: keyAtStart.usage,
    keyLimit: keyAtStart.limit,
    gitSha,
    gitDirty,
  });

  const deps: ClientDeps = {
    fetch: globalThis.fetch,
    now: () => performance.now(),
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    apiKey,
  };

  console.log(
    `[ceiling] runId=${runId} passes=${passes} models=${slate.length} items=${items.length} ` +
      `thinking=${thinking} ` +
      `git=${gitSha.slice(0, 12)}${gitDirty ? "+dirty" : ""}`,
  );
  console.log(
    `[ceiling] max_tokens=${MAX_TOKENS} (local arms run at ${LOCAL_ARM_MAX_TOKENS}; ` +
      `+${MAX_TOKENS - LOCAL_ARM_MAX_TOKENS} favouring this arm)\n` +
      `[ceiling] key usage=$${keyAtStart.usage.toFixed(4)} limit=${keyAtStart.limit === null ? "none" : `$${keyAtStart.limit}`} ` +
      `hard stop=$${SPEND_HARD_STOP_USD}`,
  );
  console.log(
    `[ceiling] slate order (cheapest first at ${REPRESENTATIVE_PROMPT_TOKENS}/${REPRESENTATIVE_COMPLETION_TOKENS} tokens): ` +
      slate
        .map(
          (m) =>
            `${m.id}@${m.provider} ~$${(
              estimateCostUsd(m, REPRESENTATIVE_PROMPT_TOKENS, REPRESENTATIVE_COMPLETION_TOKENS) * items.length
            ).toFixed(3)}`,
        )
        .join(", "),
  );

  console.log(
    `[ceiling] will write passes: ${Array.from({ length: passes }, (_, i) => passRunIdFor(runId, i + 1, passStart)).join(", ")}`,
  );

  const families: CeilingFamily[] = ["judge", "b"];
  const probes: ProbeResult[] = [];
  const stopNotes: string[] = [];

  const runOne = (
    model: CeilingModel,
    family: CeilingFamily,
    item: ItemInput,
    thisRunId: string,
  ): Promise<CeilingRecord> =>
    runCeilingItem({
      deps,
      model,
      family,
      ir,
      irHash,
      policyText,
      item,
      runId: thisRunId,
      maxTokens: MAX_TOKENS,
      localArmMaxTokens: LOCAL_ARM_MAX_TOKENS,
      // The bake-off's own default destination provider, so an action resolved
      // here matches an action resolved in the browser arms.
      destinationProvider: "claude",
      thinking,
      gitSha,
      gitDirty,
      onCall: (entry) => {
        guard.record(entry);
        if (guard.dueForCheckpoint()) {
          // Fire and forget: the guard trips on the summed costs meanwhile, and
          // awaiting a key read inside a call callback would serialize the run.
          void readKeyState(apiKey)
            .then((s) => guard.noteCheckpoint(s.usage))
            .catch(() => undefined);
        }
        writeSpend(runId, guard, { runId, passes, probes, stopNotes });
      },
    });

  // ---- PROBE ------------------------------------------------------------
  const probeItems = items.slice(0, PROBE_ITEMS);
  for (const model of slate) {
    for (const family of families) {
      if (guard.tripped()) {
        stopNotes.push(`spend guard tripped during probe, before ${model.id}/${family}`);
        break;
      }
      const { results } = await mapWithConcurrency(probeItems, CONCURRENCY, () => guard.tripped(), (item) =>
        runOne(model, family, item, `${runId}-probe`),
      );
      const probe = summarizeProbe(model, family, results.filter((r): r is CeilingRecord => r !== undefined));
      probes.push(probe);
      console.log(
        `[probe] ${model.id} ${family}: parsed ${probe.parsed}/${probe.attempted} ` +
          `provider=${probe.providersSeen.join("|") || "none"} ${probe.providerMatchesPin ? "(pinned)" : "(NOT THE PIN)"} ` +
          `reasoning=${probe.reasoningTokens.join(",")} ` +
          `ttft=${probe.ttftMs.map((t) => (t === null ? "null" : Math.round(t))).join(",")}ms ` +
          `wall=${probe.wallMs.map((t) => Math.round(t)).join(",")}ms ` +
          `verdict=${probe.verdict}` +
          (probe.errors.length > 0 ? ` errors=${probe.errors.slice(0, 2).join(" | ")}` : ""),
      );
    }
  }
  writeSpend(runId, guard, { runId, passes, probes, stopNotes });
  if (probeOnly) {
    console.log(`[ceiling] probe only; spent $${guard.totalUsd.toFixed(5)} so far`);
    return;
  }

  // ---- PASSES -----------------------------------------------------------
  for (let pass = 1; pass <= passes; pass++) {
    const passRunId = passRunIdFor(runId, pass, passStart);
    for (const model of slate) {
      for (const family of families) {
        const probe = probes.find((p) => p.model === model.id && p.family === family);
        if (probe?.verdict === "skip") {
          const note = `${armName(family, model)} SKIPPED: probe parsed 0 of ${probe.attempted}`;
          stopNotes.push(note);
          console.log(`[skip] ${note}`);
          continue;
        }
        if (guard.tripped()) {
          const note =
            `SPEND GUARD TRIPPED before ${armName(family, model)} on pass ${pass}: ` +
            `$${guard.keySpendUsd.toFixed(4)} of $${SPEND_HARD_STOP_USD}`;
          stopNotes.push(note);
          console.log(`[stop] ${note}`);
          break;
        }
        const started = performance.now();
        const { results, completed, stoppedAtIndex } = await mapWithConcurrency(
          items,
          CONCURRENCY,
          () => guard.tripped(),
          (item) => runOne(model, family, item, passRunId),
        );
        const records = results.filter((r): r is CeilingRecord => r !== undefined);
        if (stoppedAtIndex !== null) {
          const note =
            `SPEND GUARD TRIPPED mid-arm ${armName(family, model)} pass ${pass}: ` +
            `stopped at item index ${stoppedAtIndex} (${items[stoppedAtIndex]?.id ?? "?"}), ` +
            `${completed} of ${items.length} items written`;
          stopNotes.push(note);
          console.log(`[stop] ${note}`);
        }
        if (records.length > 0) {
          mkdirSync(RUNS, { recursive: true });
          writeFileSync(join(RUNS, `${passRunId}.${armName(family, model)}.jsonl`), toJsonl(records));
        }
        console.log(
          `[arm] ${passRunId} ${armName(family, model)}: ${records.length} rows in ` +
            `${Math.round(performance.now() - started)}ms, spend so far $${guard.totalUsd.toFixed(5)}`,
        );
        writeSpend(runId, guard, { runId, passes, probes, stopNotes });
      }
    }
    // Passes 2 and 3 only when pass 1 left room, so a repeat can never be the
    // thing that exhausts the key.
    if (pass === 1 && passes > 1 && guard.keySpendUsd >= 2.5) {
      const note = `passes 2+ SKIPPED: pass 1 spent $${guard.keySpendUsd.toFixed(4)}, over the $2.50 gate`;
      stopNotes.push(note);
      console.log(`[stop] ${note}`);
      break;
    }
  }

  const keyAtEnd = await readKeyState(apiKey);
  guard.noteCheckpoint(keyAtEnd.usage);
  writeSpend(runId, guard, { runId, passes, probes, stopNotes, keyAtEnd });
  console.log(
    `[ceiling] done. calls=${guard.calls} summed=$${guard.totalUsd.toFixed(5)} ` +
      `estimate=$${guard.estimatedUsd.toFixed(5)} key usage=$${keyAtEnd.usage.toFixed(5)} ` +
      `of ${keyAtEnd.limit === null ? "no limit" : `$${keyAtEnd.limit}`}`,
  );
}

await main();
