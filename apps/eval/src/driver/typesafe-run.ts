/**
 * Runs the TypeSafe arm over the study corpus: one request per message, raw
 * probabilities out, thresholds left to the scorer.
 *
 * Deliberately NOT built on `ceiling-run.ts`. That runner's plan carries
 * providers, quantization, reasoning requests and a per-call completion cap,
 * none of which exist here, and its spend guard reads an OpenRouter key
 * endpoint that has no TypeSafe equivalent. Sharing it would mean carrying
 * fields that cannot be true. What IS shared is the discipline: refuse to
 * overwrite an artifact, stamp every row with the IR hash and the commit, write
 * the ledger next to the rows, and make the whole path runnable with no key so
 * the structure can be reviewed before a penny is spent.
 *
 *   pnpm -C apps/eval typesafe            # needs TYPESAFE_API_KEY
 *   SIH_TS_DRY_RUN=1 pnpm -C apps/eval typesafe   # no key, no network
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gitProvenance, loadIr, loadItems } from "./ceiling-run.js";
import type { ItemInput } from "./ceiling.js";
import {
  MAX_CANDIDATES,
  PREDICATE_QUESTION_ID,
  PRICE_SOURCE,
  TYPESAFE_MODEL,
  TYPESAFE_SCHEMA_VERSION,
  buildRequest,
  callSystemOne,
  candidateQuestionId,
  candidateSpans,
  costUsdFor,
  predicateQuestion,
  stubAnswers,
  toJsonl,
  type CandidateAnswer,
  type SystemOneResponse,
  type TypeSafeRecord,
} from "./typesafe.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");
const RUNS = join(REPO, "runs");

export const DEFAULT_CONCURRENCY = 4;
export const DEFAULT_TIMEOUT_MS = 60_000;
export const DEFAULT_MAX_ATTEMPTS = 5;
/** A cap, not a budget: the whole corpus is ~190 requests of ~600 input tokens. */
export const DEFAULT_SPEND_CAP_USD = 2;

export interface TypeSafePlan {
  readonly runId: string;
  readonly arm: string;
  readonly limit: number | undefined;
  readonly concurrency: number;
  readonly dryRun: boolean;
  readonly maxCandidates: number;
  readonly spendCapUsd: number;
  readonly timeoutMs: number;
  readonly maxAttempts: number;
  readonly baseUrl: string | undefined;
}

function intFrom(bag: Record<string, string | undefined>, key: string, fallback: number): number {
  const raw = bag[key];
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`${key} must be a positive number, got "${raw}"`);
  return n;
}

/** Pure: the plan is what a test dumps and what the run header prints. */
export function resolveTypeSafePlan(bag: Record<string, string | undefined>): TypeSafePlan {
  const runId = (bag.SIH_TS_RUN_ID ?? "").trim();
  if (runId === "") throw new Error("SIH_TS_RUN_ID is required: it names the artifacts and must be new");
  if (!/^[a-z0-9][a-z0-9-]*$/.test(runId)) throw new Error(`SIH_TS_RUN_ID must be kebab-case, got "${runId}"`);
  return {
    runId,
    arm: (bag.SIH_TS_ARM ?? "ts-judgment").trim(),
    limit: bag.SIH_TS_LIMIT === undefined ? undefined : intFrom(bag, "SIH_TS_LIMIT", 0),
    concurrency: intFrom(bag, "SIH_TS_CONCURRENCY", DEFAULT_CONCURRENCY),
    dryRun: bag.SIH_TS_DRY_RUN === "1",
    maxCandidates: intFrom(bag, "SIH_TS_MAX_CANDIDATES", MAX_CANDIDATES),
    spendCapUsd: intFrom(bag, "SIH_TS_SPEND_CAP_USD", DEFAULT_SPEND_CAP_USD),
    timeoutMs: intFrom(bag, "SIH_TS_TIMEOUT_MS", DEFAULT_TIMEOUT_MS),
    maxAttempts: intFrom(bag, "SIH_TS_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
    baseUrl: bag.SIH_TS_BASE_URL,
  };
}

export function armFileName(plan: TypeSafePlan): string {
  return `${plan.runId}.${plan.arm}.jsonl`;
}
export function gatesFileName(plan: TypeSafePlan): string {
  return `${plan.runId}.${plan.arm}.gates.jsonl`;
}
export function spendFileName(plan: TypeSafePlan): string {
  return `${plan.runId}.${plan.arm}.spend.json`;
}

/**
 * The key comes from the environment, or from a gitignored `.env` at the repo
 * root. Read here and passed as an argument from here on, so no other module
 * touches the environment and nothing ever writes it to an artifact.
 */
export function readApiKey(env: Record<string, string | undefined>, repo: string = REPO): string {
  const direct = (env.TYPESAFE_API_KEY ?? "").trim();
  if (direct !== "") return direct;
  const dotenv = join(repo, ".env");
  if (existsSync(dotenv)) {
    for (const line of readFileSync(dotenv, "utf8").split("\n")) {
      const m = /^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.+?)\s*$/.exec(line);
      if (m?.[1] !== undefined) return m[1].replace(/^["']|["']$/g, "").trim();
    }
  }
  throw new Error("no TYPESAFE_API_KEY in the environment or in .env at the repo root (or set SIH_TS_DRY_RUN=1)");
}

export async function mapWithLimit<T, R>(items: readonly T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i]!, i);
    }
  });
  await Promise.all(workers);
  return out;
}

export function percentile(xs: readonly number[], q: number): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))] ?? null;
}

export interface TypeSafeRunDeps {
  readonly env: Record<string, string | undefined>;
  readonly now: () => number;
  readonly sleep: (ms: number) => Promise<void>;
  readonly fetchImpl: typeof fetch;
  readonly loadIr: () => { ir: ReturnType<typeof loadIr>["ir"]; irHash: string };
  readonly loadItems: () => ItemInput[];
  readonly gitProvenance: () => { gitSha: string; gitDirty: boolean };
  readonly readApiKey: (env: Record<string, string | undefined>) => string;
  readonly writeFile: (name: string, contents: string) => void;
  readonly fileExists: (name: string) => boolean;
  readonly log: (line: string) => void;
}

export function defaultTypeSafeRunDeps(): TypeSafeRunDeps {
  return {
    env: process.env,
    now: () => performance.now(),
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    fetchImpl: fetch,
    loadIr: () => loadIr(),
    loadItems: () => loadItems(),
    gitProvenance: () => gitProvenance(),
    readApiKey: (env) => readApiKey(env),
    writeFile: (name, contents) => {
      mkdirSync(RUNS, { recursive: true });
      writeFileSync(join(RUNS, name), contents, "utf8");
    },
    fileExists: (name) => existsSync(join(RUNS, name)),
    log: (line) => console.log(line),
  };
}

function answerToCandidate(response: SystemOneResponse | undefined, index: number, c: { start: number; end: number; text: string; source: "tier0" | "orthographic"; tier0EntityType?: string }): CandidateAnswer {
  const a = response?.answers[candidateQuestionId(index)];
  if (a === undefined || a.type !== "choice") {
    return { index, start: c.start, end: c.end, text: c.text, source: c.source, tier0EntityType: c.tier0EntityType ?? null, choice: null, probability: null, confidence: null, probabilities: null };
  }
  return {
    index,
    start: c.start,
    end: c.end,
    text: c.text,
    source: c.source,
    tier0EntityType: c.tier0EntityType ?? null,
    choice: a.choice,
    probability: a.probabilities[a.choice] ?? null,
    confidence: a.confidence,
    probabilities: a.probabilities,
  };
}

export async function runTypeSafe(overrides: Partial<TypeSafeRunDeps> = {}): Promise<void> {
  const deps: TypeSafeRunDeps = { ...defaultTypeSafeRunDeps(), ...overrides };
  const plan = resolveTypeSafePlan(deps.env);
  const armFile = armFileName(plan);
  if (deps.fileExists(armFile)) throw new Error(`${armFile} exists; pick a new SIH_TS_RUN_ID rather than overwriting a measurement`);

  const { ir, irHash } = deps.loadIr();
  const items = deps.loadItems();
  const chosen = plan.limit === undefined ? items : items.slice(0, plan.limit);
  const { gitSha, gitDirty } = deps.gitProvenance();
  const apiKey = plan.dryRun ? "dry-run" : deps.readApiKey(deps.env);
  const pred = predicateQuestion(ir);
  deps.log(
    `typesafe: run=${plan.runId} arm=${plan.arm} model=${TYPESAFE_MODEL} items=${chosen.length} concurrency=${plan.concurrency} dryRun=${plan.dryRun} predicate=${pred?.predicateId ?? "none"}`,
  );

  let spentUsd = 0;
  let stoppedEarly = false;
  const records = await mapWithLimit(chosen, plan.concurrency, async (item): Promise<TypeSafeRecord> => {
    const itemStart = deps.now();
    const candidates = candidateSpans(ir, item.text, plan.maxCandidates);
    const body = buildRequest(ir, item.text, candidates);
    let response: SystemOneResponse | undefined;
    let callWallMs: number | null = null;
    let attempts = 1;
    let retries = 0;
    let httpStatus: number | null = null;
    let error: string | null = null;
    let costUsd = 0;
    if (stoppedEarly) {
      error = "spend cap reached before this item";
    } else if (plan.dryRun) {
      response = stubAnswers(item.id, body);
      callWallMs = 0;
      costUsd = costUsdFor(response.usage);
    } else {
      const outcome = await callSystemOne(
        { fetchImpl: deps.fetchImpl, apiKey, now: deps.now, sleep: deps.sleep, timeoutMs: plan.timeoutMs, maxAttempts: plan.maxAttempts, baseUrl: plan.baseUrl },
        body,
      );
      response = outcome.response;
      callWallMs = outcome.response === undefined ? null : outcome.wallMs;
      attempts = outcome.attempts;
      retries = outcome.retries;
      httpStatus = outcome.status ?? null;
      error = outcome.error ?? null;
      costUsd = outcome.costUsd;
    }
    spentUsd += costUsd;
    if (spentUsd >= plan.spendCapUsd && !stoppedEarly) {
      stoppedEarly = true;
      deps.log(`typesafe: SPEND CAP $${plan.spendCapUsd} reached at $${spentUsd.toFixed(6)}; remaining items are recorded unanswered`);
    }
    const predAnswer = response?.answers[PREDICATE_QUESTION_ID];
    return {
      schemaVersion: TYPESAFE_SCHEMA_VERSION,
      runId: plan.runId,
      itemId: item.id,
      policy: item.policy,
      irHash,
      policyHash: ir.policyHash,
      arm: plan.arm,
      requestedModelId: TYPESAFE_MODEL,
      modelId: response?.model ?? null,
      gitSha,
      gitDirty,
      dryRun: plan.dryRun,
      text: item.text,
      gold: item.gold.map((g) => ({ start: g.start, end: g.end, text: g.text, entityType: g.entityType, action: g.action })),
      predicateId: pred?.predicateId ?? null,
      predicateProbability: predAnswer !== undefined && predAnswer.type === "noul" ? predAnswer.noul : null,
      candidates: candidates.map((c, i) => answerToCandidate(response, i, c)),
      questionCount: Object.keys(body.questions).length,
      inputTokens: response?.usage.input_tokens ?? null,
      outputTokens: response?.usage.output_tokens ?? null,
      costUsd,
      callWallMs,
      itemWallMs: deps.now() - itemStart,
      attempts,
      retries,
      httpStatus,
      error,
    };
  });

  deps.writeFile(armFile, toJsonl(records));

  const answered = records.filter((r) => r.predicateProbability !== null);
  const itemWalls = answered.map((r) => r.itemWallMs);
  const gates = {
    runId: plan.runId,
    arm: plan.arm,
    model: TYPESAFE_MODEL,
    modelReturned: records.find((r) => r.modelId !== null)?.modelId ?? null,
    dryRun: plan.dryRun,
    items: records.length,
    answered: answered.length,
    errored: records.filter((r) => r.error !== null).length,
    retries: records.reduce((a, r) => a + r.retries, 0),
    rateLimited: records.filter((r) => r.httpStatus === 429 || r.httpStatus === 529).length,
    questionsPerItem: { min: Math.min(...records.map((r) => r.questionCount)), max: Math.max(...records.map((r) => r.questionCount)) },
    candidatesPerItem: { min: Math.min(...records.map((r) => r.candidates.length)), max: Math.max(...records.map((r) => r.candidates.length)) },
    itemWallMsP50: percentile(itemWalls, 0.5),
    itemWallMsP95: percentile(itemWalls, 0.95),
    inputTokensTotal: records.reduce((a, r) => a + (r.inputTokens ?? 0), 0),
    outputTokensTotal: records.reduce((a, r) => a + (r.outputTokens ?? 0), 0),
    costUsd: Number(spentUsd.toFixed(6)),
    // Named rather than silently absent: one JSON body, no stream, so these are
    // not slow numbers, they are numbers this transport does not produce.
    notMeasured: {
      ttftMs: "no stream: /v1/systemone returns one body",
      decodeTokPerSec: "no decode window to measure",
      reasoningTokens: "System One returns a judgment, not reasoning",
      spanWisePredicate: "a noul returns a probability and no clause, so the predicate has no span to score",
    },
    unresolvedQuotes: 0,
    unresolvedQuotesNote: "zero BY CONSTRUCTION: code proposes spans and the model only labels them",
    gitSha,
    gitDirty,
  };
  deps.writeFile(gatesFileName(plan), JSON.stringify(gates) + "\n");
  deps.writeFile(
    spendFileName(plan),
    JSON.stringify(
      {
        runId: plan.runId,
        arm: plan.arm,
        model: TYPESAFE_MODEL,
        calls: records.filter((r) => r.modelId !== null).length,
        inputTokens: gates.inputTokensTotal,
        outputTokens: gates.outputTokensTotal,
        costUsd: Number(spentUsd.toFixed(6)),
        capUsd: plan.spendCapUsd,
        stoppedEarly,
        priceSource: PRICE_SOURCE,
        dryRun: plan.dryRun,
      },
      null,
      2,
    ) + "\n",
  );
  deps.log(
    `typesafe: wrote ${armFile} (${records.length} rows, ${answered.length} answered), gates and ledger; $${spentUsd.toFixed(6)} spent`,
  );
}
