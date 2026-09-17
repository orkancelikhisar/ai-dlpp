/**
 * The TypeSafe arm: a JUDGMENT arm, not a generation arm.
 *
 * Every other tier-2 arm in this repository asks a model to WRITE something --
 * JSON carrying a label and a quoted clause -- and then recovers spans from the
 * text it wrote. TypeSafe's System One models (Jev) cannot do that: they return
 * a typed answer and a probability and nothing else. So the decomposition is
 * inverted, and that inversion is the point of this arm:
 *
 *   CODE proposes, the MODEL judges, CODE places the span.
 *
 *   - The semantic predicate becomes one `noul`: P(the message discloses a
 *     client relationship). No clause comes back, so this arm can be scored at
 *     MESSAGE level and cannot be scored span-wise. Stated, not papered over.
 *   - Entity spans become one `choice` per CANDIDATE span, where the candidates
 *     are produced here by tier 0 and the orthographic oracle -- both already in
 *     this repository, both recall-tuned by design. The model only labels them.
 *     Offsets therefore come from the regex that found them, so the span ladder
 *     that every generative arm needs (quote -> mention -> nothing) does not
 *     exist here and `unresolvedQuotes` is zero BY CONSTRUCTION rather than by
 *     good behaviour. That is a structural advantage and must be reported as
 *     one, never as the model being better at quoting.
 *
 * ## What this arm does not have, and will not pretend to have
 *
 * `POST /v1/systemone` returns one JSON body. There is no stream, so there is no
 * time to first token and no decode window; there are no reasoning tokens, no
 * provider pin, and no quantization to read off an endpoint. The ceiling arms'
 * record schema carries all of those, so this arm gets its OWN schema instead of
 * borrowing that one and filling the missing half with nulls that later read as
 * measurements. Cross-arm comparison happens at scoring time, over the metrics
 * both sides genuinely have: accuracy, prevention, per-message wall time, cost.
 *
 * ## Raw probabilities are the artifact
 *
 * Records carry the probability, never a thresholded boolean. One run then
 * yields the whole precision/recall curve, the calibration table and any
 * threshold a later reader wants to argue for; a run that stored booleans would
 * have to be paid for again to answer the next question. `projectFindings` is
 * the only place a threshold turns a probability into a finding.
 *
 * Contract source: https://docs.typesafe.ai/api.md, primitives/{noul,choice}.md
 * and models.md, read 2026-09-17.
 */
import { createHash } from "node:crypto";
import { z } from "zod";
import { resolveAction, runTier0, segmentText, type PolicyIr } from "@sih/core";
import { orthographicOracle } from "../corpus/leakage.js";

export const TYPESAFE_SCHEMA_VERSION = 1;
export const TYPESAFE_BASE_URL = "https://api.typesafe.ai";
export const SYSTEMONE_PATH = "/v1/systemone";

/**
 * The concrete model, not the `jev-latest` alias.
 *
 * Same reason the hosted arms pin a provider: an alias that moves under a run
 * makes two passes incomparable without saying so. The response echoes `model`,
 * and the record carries both what was asked and what came back.
 */
export const TYPESAFE_MODEL = "jev-1.13.0";

/**
 * docs.typesafe.ai/models.md, read 2026-09-17: input $42 per BILLION tokens,
 * output free. Kept as a per-token number so the arithmetic in the ledger is the
 * same shape as the OpenRouter one, and named with its source because a price
 * copied without a date is the defect this project keeps finding in itself.
 */
export const INPUT_USD_PER_TOKEN = 42 / 1_000_000_000;
export const OUTPUT_USD_PER_TOKEN = 0;
export const PRICE_SOURCE = "docs.typesafe.ai/models.md read 2026-09-17: input $42/1e9 tokens, output free";

/** Recall-tuned, so the cap is generous; every arm records how many it used. */
export const MAX_CANDIDATES = 24;
/** The option that lets the model reject a candidate the regexes over-found. */
export const NOT_CONFIDENTIAL = "not-confidential";
export const PREDICATE_QUESTION_ID = "predicate";

// ---------------------------------------------------------------- request ---

export interface NoulQuestion {
  readonly type: "noul";
  readonly instructions: string;
  readonly criteria?: { readonly true: string; readonly false: string };
}
export interface ChoiceQuestion {
  readonly type: "choice";
  readonly instructions: string;
  readonly criteria: Readonly<Record<string, string>>;
}
export type TsQuestion = NoulQuestion | ChoiceQuestion;

export interface SystemOneRequest {
  readonly state: unknown;
  readonly model: string;
  readonly questions: Readonly<Record<string, TsQuestion>>;
}

// --------------------------------------------------------------- response ---

const NoulAnswerSchema = z.object({ type: z.literal("noul"), noul: z.number().min(0).max(1) });
const ChoiceAnswerSchema = z.object({
  type: z.literal("choice"),
  choice: z.string(),
  probabilities: z.record(z.string(), z.number()),
  confidence: z.number().min(0).max(1),
});
const AnswerSchema = z.discriminatedUnion("type", [NoulAnswerSchema, ChoiceAnswerSchema]);
export const SystemOneResponseSchema = z.object({
  model: z.string().min(1),
  answers: z.record(z.string(), AnswerSchema),
  usage: z.object({ input_tokens: z.number().int().nonnegative(), output_tokens: z.number().int().nonnegative() }),
});
export type SystemOneResponse = z.infer<typeof SystemOneResponseSchema>;
export type NoulAnswer = z.infer<typeof NoulAnswerSchema>;
export type ChoiceAnswer = z.infer<typeof ChoiceAnswerSchema>;

export function costUsdFor(usage: SystemOneResponse["usage"]): number {
  return usage.input_tokens * INPUT_USD_PER_TOKEN + usage.output_tokens * OUTPUT_USD_PER_TOKEN;
}

// ----------------------------------------------------------------- client ---

export interface TypeSafeClientDeps {
  readonly fetchImpl: typeof fetch;
  readonly apiKey: string;
  readonly now: () => number;
  readonly sleep: (ms: number) => Promise<void>;
  readonly timeoutMs: number;
  readonly maxAttempts: number;
  readonly baseUrl?: string;
}

export interface CallOutcome {
  readonly response: SystemOneResponse | undefined;
  /** Wall time of the ATTEMPT THAT ANSWERED, excluding backoff, like the hosted arms. */
  readonly wallMs: number;
  /** End to end including every failed attempt and every sleep. */
  readonly totalMs: number;
  readonly attempts: number;
  readonly retries: number;
  readonly status: number | undefined;
  readonly costUsd: number;
  readonly error: string | undefined;
}

/** 429 and 529 are the documented retryable pair; 5xx and transport faults join them. */
export function isRetryable(status: number | undefined): boolean {
  if (status === undefined) return true;
  return status === 429 || status === 529 || (status >= 500 && status < 600);
}

export function backoffMs(attempt: number, retryAfterSeconds?: number): number {
  if (retryAfterSeconds !== undefined && Number.isFinite(retryAfterSeconds)) {
    return Math.min(30_000, Math.max(0, retryAfterSeconds) * 1000);
  }
  return Math.min(16_000, 500 * 2 ** (attempt - 1));
}

export async function callSystemOne(deps: TypeSafeClientDeps, body: SystemOneRequest): Promise<CallOutcome> {
  const url = `${deps.baseUrl ?? TYPESAFE_BASE_URL}${SYSTEMONE_PATH}`;
  const startedAll = deps.now();
  let attempts = 0;
  let lastStatus: number | undefined;
  let lastError = "no attempt ran";
  while (attempts < deps.maxAttempts) {
    attempts += 1;
    const started = deps.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), deps.timeoutMs);
    try {
      const res = await deps.fetchImpl(url, {
        method: "POST",
        headers: {
          // The key is read from the environment and never written to any
          // artifact this run produces; the ledger records spend, not identity.
          authorization: `Bearer ${deps.apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      lastStatus = res.status;
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        lastError = `HTTP ${res.status}: ${text.slice(0, 200)}`;
        if (!isRetryable(res.status) || attempts >= deps.maxAttempts) break;
        const retryAfter = Number(res.headers.get("retry-after") ?? Number.NaN);
        await deps.sleep(backoffMs(attempts, Number.isNaN(retryAfter) ? undefined : retryAfter));
        continue;
      }
      const parsed = SystemOneResponseSchema.safeParse(await res.json());
      if (!parsed.success) {
        // A schema miss is NOT retried: the same body would come back. It is a
        // contract change, and it should stop the run loudly.
        lastError = `response failed schema: ${parsed.error.issues.map((i) => i.path.join(".") + " " + i.message).join("; ").slice(0, 200)}`;
        break;
      }
      return {
        response: parsed.data,
        wallMs: deps.now() - started,
        totalMs: deps.now() - startedAll,
        attempts,
        retries: attempts - 1,
        status: res.status,
        costUsd: costUsdFor(parsed.data.usage),
        error: undefined,
      };
    } catch (err) {
      lastError = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      if (attempts >= deps.maxAttempts) break;
      await deps.sleep(backoffMs(attempts));
    } finally {
      clearTimeout(timer);
    }
  }
  return {
    response: undefined,
    wallMs: 0,
    totalMs: deps.now() - startedAll,
    attempts,
    retries: Math.max(0, attempts - 1),
    status: lastStatus,
    costUsd: 0,
    error: lastError,
  };
}

// ------------------------------------------------------------- candidates ---

export interface Candidate {
  readonly start: number;
  readonly end: number;
  readonly text: string;
  /** Which code-side finder proposed it. Both are recall-tuned on purpose. */
  readonly source: "tier0" | "orthographic";
  /** Tier 0's own guess at the type, kept so the model's answer can be compared with it. */
  readonly tier0EntityType?: string;
}

/**
 * Candidates are the arm's recall ceiling: the model cannot choose a span nobody
 * offered it. Tier 0 brings the structured identifiers, the orthographic oracle
 * brings capitalised multiword names and odd tokens -- the same generator the
 * paper's no-model floor uses, which is deliberate: it makes "what does the
 * judgment add over the floor's own candidates" a measurable question.
 */
export function candidateSpans(ir: PolicyIr, text: string, limit = MAX_CANDIDATES): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const push = (c: Candidate): void => {
    const key = `${c.start}:${c.end}`;
    if (seen.has(key) || c.end <= c.start) return;
    seen.add(key);
    out.push(c);
  };
  for (const f of runTier0(ir, text, segmentText(text))) {
    push({ start: f.start, end: f.end, text: text.slice(f.start, f.end), source: "tier0", tier0EntityType: f.entityType });
  }
  for (const s of orthographicOracle(text)) {
    push({ start: s.start, end: s.end, text: text.slice(s.start, s.end), source: "orthographic" });
  }
  return out.sort((a, b) => a.start - b.start || a.end - b.end).slice(0, limit);
}

// -------------------------------------------------------------- questions ---

export function entityCriteria(ir: PolicyIr): Record<string, string> {
  const criteria: Record<string, string> = {};
  for (const et of ir.entityTypes) {
    if (et.id.startsWith("pred:")) continue;
    const ex = et.examples.length > 0 ? ` Examples: ${et.examples.slice(0, 3).join("; ")}.` : "";
    const no = et.counterExamples.length > 0 ? ` Not this: ${et.counterExamples.slice(0, 3).join("; ")}.` : "";
    criteria[et.id] = `${et.nlDefinition}${ex}${no}`;
  }
  criteria[NOT_CONFIDENTIAL] =
    "The highlighted value is none of the categories above: ordinary prose, a public identifier, a number with no bearing on a customer, or text the policy does not govern.";
  return criteria;
}

export function predicateQuestion(ir: PolicyIr): { id: string; question: NoulQuestion; predicateId: string } | undefined {
  const predicate = ir.semanticPredicates.find((p) => p.scope === "message");
  if (predicate === undefined) return undefined;
  return {
    id: PREDICATE_QUESTION_ID,
    predicateId: predicate.id,
    question: {
      type: "noul",
      // The IR's own words, verbatim. The compiled judge is handed the same
      // sentence, so the two arms differ in MECHANISM and not in what was asked.
      instructions: `${predicate.nlPredicate} Judge the message as a whole.`,
      criteria: {
        true: "The message states or clearly implies that a named organisation is a client, prospective client, or counterparty of the Firm.",
        false:
          "No organisation is named, or an organisation is named only as a vendor, tool, employer, public body or topic of discussion, with no client, prospective-client or counterparty relationship to the Firm.",
      },
    },
  };
}

export function candidateQuestionId(index: number): string {
  return `cand_${index}`;
}

/**
 * One request per message: the predicate plus one classification per candidate.
 *
 * The docs are explicit that questions in a request are answered independently
 * and that the document dominates the bill, so batching is both cheaper and
 * free of cross-talk. Question ids never reach the model, so each candidate's
 * instructions repeat the span verbatim.
 */
export function buildRequest(ir: PolicyIr, text: string, candidates: readonly Candidate[]): SystemOneRequest {
  const questions: Record<string, TsQuestion> = {};
  const pred = predicateQuestion(ir);
  if (pred !== undefined) questions[pred.id] = pred.question;
  const criteria = entityCriteria(ir);
  candidates.forEach((c, i) => {
    questions[candidateQuestionId(i)] = {
      type: "choice",
      instructions: `In the message, the value "${c.text}" appears at character offset ${c.start}. Which category does that value belong to under the Firm's policy?`,
      criteria,
    };
  });
  return { state: { message: text }, model: TYPESAFE_MODEL, questions };
}

// ----------------------------------------------------------------- record ---

const CandidateAnswerSchema = z.object({
  index: z.number().int().nonnegative(),
  start: z.number().int().nonnegative(),
  end: z.number().int().positive(),
  text: z.string(),
  source: z.enum(["tier0", "orthographic"]),
  tier0EntityType: z.string().nullable(),
  /** The label Jev picked, and the whole distribution it picked from. */
  choice: z.string().nullable(),
  probability: z.number().min(0).max(1).nullable(),
  confidence: z.number().min(0).max(1).nullable(),
  probabilities: z.record(z.string(), z.number()).nullable(),
});

export const TypeSafeRecordSchema = z.object({
  schemaVersion: z.literal(TYPESAFE_SCHEMA_VERSION),
  runId: z.string().min(1),
  itemId: z.string().min(1),
  policy: z.string().min(1),
  irHash: z.string().regex(/^[0-9a-f]{64}$/),
  policyHash: z.string().min(1),
  arm: z.string().min(1),
  requestedModelId: z.string().min(1),
  /** What the response echoed. Null when nothing answered. */
  modelId: z.string().nullable(),
  gitSha: z.string().regex(/^[0-9a-f]{40}$/),
  gitDirty: z.boolean(),
  dryRun: z.boolean(),
  text: z.string().min(1),
  gold: z.array(z.object({ start: z.number(), end: z.number(), text: z.string(), entityType: z.string(), action: z.string() })),
  predicateId: z.string().nullable(),
  /** P(predicate holds). RAW: no threshold has been applied to this number. */
  predicateProbability: z.number().min(0).max(1).nullable(),
  candidates: z.array(CandidateAnswerSchema),
  questionCount: z.number().int().nonnegative(),
  inputTokens: z.number().int().nonnegative().nullable(),
  outputTokens: z.number().int().nonnegative().nullable(),
  costUsd: z.number().nonnegative(),
  /** The answering attempt, and the end-to-end including retries and sleeps. */
  callWallMs: z.number().nonnegative().nullable(),
  itemWallMs: z.number().nonnegative(),
  attempts: z.number().int().positive(),
  retries: z.number().int().nonnegative(),
  httpStatus: z.number().int().nullable(),
  error: z.string().nullable(),
});
export type TypeSafeRecord = z.infer<typeof TypeSafeRecordSchema>;
export type CandidateAnswer = z.infer<typeof CandidateAnswerSchema>;

export function toJsonl(records: readonly TypeSafeRecord[]): string {
  return records.map((r) => JSON.stringify(r)).join("\n") + (records.length > 0 ? "\n" : "");
}

export function loadTypeSafeRecords(jsonl: string): TypeSafeRecord[] {
  return jsonl
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => TypeSafeRecordSchema.parse(JSON.parse(l)));
}

// ------------------------------------------------------------- projection ---

export interface ProjectedFinding {
  start: number;
  end: number;
  text: string;
  entityType: string;
  severity: string;
  tier: number;
  source: string;
  confidence: number;
  action: string;
}

export interface Thresholds {
  readonly predicate: number;
  /** The fallback when no per-type threshold is given. */
  readonly candidate: number;
  /** Per entity type, because one number cannot serve a PEM block and a client name. */
  readonly perType?: Readonly<Record<string, number>>;
  /**
   * How a candidate's probability is read.
   *
   * `"argmax"` is what the first run used: fire on the top option's own
   * probability. `"confidential-mass"` fires on 1 - P(not-confidential) and
   * labels with the best confidential option, which is the honest question when
   * the mass splits across two entity types that are both confidential.
   */
  readonly rule?: "argmax" | "confidential-mass";
  /**
   * Keep one finding per overlapping cluster, highest probability first.
   *
   * MEASURED: 55 of the 139 span false positives in pass 1 were a second
   * candidate inside a finding already reported -- four orthographic hits inside
   * one PEM block, say. Gold pairs one-to-one, so every extra is a false
   * positive for text already caught.
   *
   * REPORTING ONLY. Keeping the most confident member DROPS the others, so a
   * short high-confidence token inside a long low-confidence key block survives
   * and the block does not. Here that moves a score and nothing else: these
   * findings are consumed by `typesafe-score-lib` alone and no transform acts on
   * them. In an engine whose redactor consumes findings it would blank twenty
   * characters of a hundred-character private key and send the rest, so build
   * the redaction set from every finding that fired, BEFORE any merge. Raised by
   * the system1-dlp session, which hit it while lifting this function.
   */
  readonly mergeOverlaps?: boolean;
}

const overlapsSpan = (a: { start: number; end: number }, b: { start: number; end: number }): boolean =>
  a.start < b.end && b.start < a.end;

/** The firing decision for one candidate: its label and the probability that fired it, or nothing. */
export function decideCandidate(c: CandidateAnswer, t: Thresholds): { entityType: string; probability: number } | undefined {
  if (c.choice === null) return undefined;
  const rule = t.rule ?? "argmax";
  const threshold = (type: string): number => t.perType?.[type] ?? t.candidate;
  if (rule === "argmax") {
    if (c.choice === NOT_CONFIDENTIAL || c.probability === null) return undefined;
    return c.probability >= threshold(c.choice) ? { entityType: c.choice, probability: c.probability } : undefined;
  }
  const probs = c.probabilities;
  if (probs === null) return undefined;
  let best: string | undefined;
  for (const [k, v] of Object.entries(probs)) {
    if (k === NOT_CONFIDENTIAL) continue;
    if (best === undefined || v > (probs[best] ?? 0)) best = k;
  }
  if (best === undefined) return undefined;
  const confidential = 1 - (probs[NOT_CONFIDENTIAL] ?? 0);
  return confidential >= threshold(best) ? { entityType: best, probability: confidential } : undefined;
}

/**
 * Turns one record's probabilities into findings at the given thresholds.
 *
 * The ONLY place a probability becomes a decision. The predicate finding spans
 * the whole message because a `noul` returns no clause: message-level scoring is
 * exact, span-wise scoring of the predicate is not available for this arm and
 * must not be reported as if it were.
 */
export function projectFindings(ir: PolicyIr, record: TypeSafeRecord, t: Thresholds): ProjectedFinding[] {
  const out: ProjectedFinding[] = [];
  if (record.predicateId !== null && record.predicateProbability !== null && record.predicateProbability >= t.predicate) {
    const id = `pred:${record.predicateId}`;
    out.push({
      start: 0,
      end: record.text.length,
      text: record.text,
      entityType: id,
      severity: "high",
      tier: 2,
      source: `typesafe/${record.modelId ?? record.requestedModelId}`,
      confidence: record.predicateProbability,
      action: resolveAction(ir, id, "default"),
    });
  }
  const fired: ProjectedFinding[] = [];
  for (const c of record.candidates) {
    const decision = decideCandidate(c, t);
    if (decision === undefined) continue;
    if (!ir.entityTypes.some((e) => e.id === decision.entityType)) continue;
    fired.push({
      start: c.start,
      end: c.end,
      text: c.text,
      entityType: decision.entityType,
      severity: ir.entityTypes.find((e) => e.id === decision.entityType)?.severity ?? "high",
      tier: 2,
      source: `typesafe/${record.modelId ?? record.requestedModelId}`,
      confidence: decision.probability,
      action: resolveAction(ir, decision.entityType, "default"),
    });
  }
  const kept: ProjectedFinding[] = [];
  if (t.mergeOverlaps === true) {
    for (const f of [...fired].sort((a, b) => b.confidence - a.confidence || a.start - b.start)) {
      if (!kept.some((k) => overlapsSpan(f, k))) kept.push(f);
    }
  } else {
    kept.push(...fired);
  }
  out.push(...kept);
  return out.sort((a, b) => a.start - b.start || a.end - b.end);
}

/** Deterministic stand-in used by the dry run, so the whole path runs with no key and no network. */
export function stubAnswers(itemId: string, body: SystemOneRequest): SystemOneResponse {
  const answers: Record<string, NoulAnswer | ChoiceAnswer> = {};
  for (const [id, q] of Object.entries(body.questions)) {
    const h = createHash("sha256").update(`${itemId}:${id}`).digest();
    const p = h.readUInt16BE(0) / 65_535;
    if (q.type === "noul") {
      answers[id] = { type: "noul", noul: Number(p.toFixed(4)) };
    } else {
      const options = Object.keys(q.criteria);
      const pick = options[h.readUInt16BE(2) % options.length] ?? NOT_CONFIDENTIAL;
      const probabilities: Record<string, number> = {};
      for (const o of options) probabilities[o] = o === pick ? Number(p.toFixed(4)) : Number(((1 - p) / (options.length - 1)).toFixed(4));
      answers[id] = { type: "choice", choice: pick, probabilities, confidence: Number(p.toFixed(4)) };
    }
  }
  const inputTokens = JSON.stringify(body).length / 4;
  return {
    model: TYPESAFE_MODEL,
    answers,
    usage: { input_tokens: Math.round(inputTokens), output_tokens: Object.keys(body.questions).length * 4 },
  };
}
