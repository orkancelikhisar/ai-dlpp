import { describe, expect, it, vi } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { readFileSync } from "node:fs";
import {
  INPUT_USD_PER_TOKEN,
  NOT_CONFIDENTIAL,
  PREDICATE_QUESTION_ID,
  TYPESAFE_MODEL,
  backoffMs,
  buildRequest,
  callSystemOne,
  candidateQuestionId,
  candidateSpans,
  costUsdFor,
  entityCriteria,
  isRetryable,
  predicateQuestion,
  projectFindings,
  stubAnswers,
  type TypeSafeRecord,
} from "../src/driver/typesafe.js";

const IR = loadPolicyIr(readFileSync(new URL("../../../policies/compiled/p-fin.ir.json", import.meta.url), "utf8"));
const TEXT = "Meridian Capital asked about A/C No. 50100234567890 before the call with Sable Wood Partners.";

function deps(fetchImpl: typeof fetch, over: Partial<Parameters<typeof callSystemOne>[0]> = {}) {
  let t = 0;
  return { fetchImpl, apiKey: "k", now: () => (t += 10), sleep: vi.fn(async () => {}), timeoutMs: 50, maxAttempts: 3, ...over };
}
const ok = (body: unknown): Response => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
const GOOD = { model: TYPESAFE_MODEL, answers: { [PREDICATE_QUESTION_ID]: { type: "noul", noul: 0.81 } }, usage: { input_tokens: 500, output_tokens: 8 } };

describe("pricing", () => {
  it("bills input tokens at $42 per billion and output at zero", () => {
    expect(costUsdFor({ input_tokens: 1_000_000, output_tokens: 1_000_000 })).toBeCloseTo(0.042, 9);
    expect(INPUT_USD_PER_TOKEN).toBeCloseTo(4.2e-8, 12);
  });
});

describe("retry policy", () => {
  it("retries 429, 529, 5xx and transport faults, and never a 422", () => {
    expect([429, 529, 500, 503].every(isRetryable)).toBe(true);
    expect(isRetryable(422)).toBe(false);
    expect(isRetryable(401)).toBe(false);
    expect(isRetryable(undefined)).toBe(true);
  });
  it("honours retry-after over exponential backoff, and caps both", () => {
    expect(backoffMs(1)).toBe(500);
    expect(backoffMs(4)).toBe(4000);
    expect(backoffMs(99)).toBe(16_000);
    expect(backoffMs(1, 3)).toBe(3000);
    expect(backoffMs(1, 9999)).toBe(30_000);
  });
});

describe("callSystemOne", () => {
  it("sends bearer auth and the pinned model, and returns the parsed answer with its cost", async () => {
    const fetchImpl = vi.fn(async () => ok(GOOD)) as unknown as typeof fetch;
    const d = deps(fetchImpl);
    const out = await callSystemOne(d, { state: { message: TEXT }, model: TYPESAFE_MODEL, questions: {} });
    const [, init] = (fetchImpl as unknown as { mock: { calls: [string, RequestInit][] } }).mock.calls[0]!;
    expect((init.headers as Record<string, string>).authorization).toBe("Bearer k");
    expect(JSON.parse(String(init.body)).model).toBe(TYPESAFE_MODEL);
    expect(out.response?.answers[PREDICATE_QUESTION_ID]).toEqual({ type: "noul", noul: 0.81 });
    expect(out.costUsd).toBeCloseTo(500 * INPUT_USD_PER_TOKEN, 12);
    expect(out.retries).toBe(0);
  });

  it("retries a 429 and reports the retry count", async () => {
    let n = 0;
    const fetchImpl = vi.fn(async () => (++n === 1 ? new Response("slow down", { status: 429 }) : ok(GOOD))) as unknown as typeof fetch;
    const d = deps(fetchImpl);
    const out = await callSystemOne(d, { state: {}, model: TYPESAFE_MODEL, questions: {} });
    expect(out.response).toBeDefined();
    expect(out.attempts).toBe(2);
    expect(out.retries).toBe(1);
    expect(d.sleep).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry a body that fails the schema: the same body would come back", async () => {
    const fetchImpl = vi.fn(async () => ok({ model: "x", answers: { a: { type: "noul", noul: 4 } }, usage: { input_tokens: 1, output_tokens: 0 } })) as unknown as typeof fetch;
    const d = deps(fetchImpl);
    const out = await callSystemOne(d, { state: {}, model: TYPESAFE_MODEL, questions: {} });
    expect(out.response).toBeUndefined();
    expect(out.attempts).toBe(1);
    expect(out.error).toContain("schema");
  });

  it("gives up after maxAttempts and reports the last status without a body dump", async () => {
    const fetchImpl = vi.fn(async () => new Response("overloaded", { status: 529 })) as unknown as typeof fetch;
    const out = await callSystemOne(deps(fetchImpl), { state: {}, model: TYPESAFE_MODEL, questions: {} });
    expect(out.response).toBeUndefined();
    expect(out.attempts).toBe(3);
    expect(out.status).toBe(529);
    expect(out.costUsd).toBe(0);
  });
});

describe("candidates", () => {
  it("proposes tier-0 and orthographic spans whose offsets are exact", () => {
    const cands = candidateSpans(IR, TEXT);
    expect(cands.length).toBeGreaterThan(1);
    for (const c of cands) expect(TEXT.slice(c.start, c.end)).toBe(c.text);
    expect(cands.some((c) => c.source === "tier0")).toBe(true);
    expect(cands.some((c) => c.source === "orthographic")).toBe(true);
    expect(cands.some((c) => c.text.includes("Meridian Capital"))).toBe(true);
  });

  it("dedupes identical spans and respects the cap", () => {
    const cands = candidateSpans(IR, TEXT, 2);
    expect(cands).toHaveLength(2);
    expect(new Set(cands.map((c) => `${c.start}:${c.end}`)).size).toBe(2);
  });
});

describe("questions", () => {
  it("asks the predicate as a noul in the IR's own words", () => {
    const q = predicateQuestion(IR);
    expect(q?.predicateId).toBe("client-relationship-disclosure");
    expect(q?.question.type).toBe("noul");
    expect(q?.question.instructions.startsWith(IR.semanticPredicates[0]!.nlPredicate)).toBe(true);
  });

  it("offers every policy entity type plus a rejection option, and no shadow predicate", () => {
    const criteria = entityCriteria(IR);
    expect(Object.keys(criteria)).toContain("in-aadhaar");
    expect(Object.keys(criteria)).toContain(NOT_CONFIDENTIAL);
    expect(Object.keys(criteria).some((k) => k.startsWith("pred:"))).toBe(false);
  });

  it("batches one predicate question and one choice per candidate, naming the span verbatim", () => {
    const cands = candidateSpans(IR, TEXT);
    const req = buildRequest(IR, TEXT, cands);
    expect(Object.keys(req.questions)).toHaveLength(cands.length + 1);
    expect(req.questions[PREDICATE_QUESTION_ID]!.type).toBe("noul");
    const first = req.questions[candidateQuestionId(0)]!;
    expect(first.type).toBe("choice");
    expect(first.instructions).toContain(cands[0]!.text);
    expect(req.state).toEqual({ message: TEXT });
  });
});

const record = (over: Partial<TypeSafeRecord> = {}): TypeSafeRecord => ({
  schemaVersion: 1,
  runId: "r",
  itemId: "i",
  policy: "p-fin",
  irHash: "a".repeat(64),
  policyHash: IR.policyHash,
  arm: "ts-judgment",
  requestedModelId: TYPESAFE_MODEL,
  modelId: TYPESAFE_MODEL,
  gitSha: "b".repeat(40),
  gitDirty: false,
  dryRun: true,
  text: TEXT,
  gold: [],
  predicateId: "client-relationship-disclosure",
  predicateProbability: 0.9,
  candidates: [],
  questionCount: 1,
  inputTokens: 10,
  outputTokens: 1,
  costUsd: 0,
  callWallMs: 1,
  itemWallMs: 2,
  attempts: 1,
  retries: 0,
  httpStatus: 200,
  error: null,
  ...over,
});

describe("projectFindings: the only place a probability becomes a decision", () => {
  it("fires the predicate over the whole message above the threshold, and not below it", () => {
    const r = record({ predicateProbability: 0.6 });
    const on = projectFindings(IR, r, { predicate: 0.5, candidate: 0.5 });
    expect(on).toHaveLength(1);
    expect(on[0]!.entityType).toBe("pred:client-relationship-disclosure");
    expect(on[0]!.start).toBe(0);
    expect(on[0]!.end).toBe(TEXT.length);
    expect(on[0]!.action).toBe("redact");
    expect(projectFindings(IR, r, { predicate: 0.7, candidate: 0.5 })).toHaveLength(0);
  });

  it("keeps a labelled candidate, drops a rejected one, and drops a label the policy does not define", () => {
    const base = { index: 0, start: 0, end: 16, text: "Meridian Capital", source: "tier0" as const, tier0EntityType: null, confidence: 0.9, probabilities: null };
    const r = record({
      predicateProbability: null,
      candidates: [
        { ...base, index: 0, choice: "client-name", probability: 0.8 },
        { ...base, index: 1, start: 30, end: 48, text: "50100234567890", choice: NOT_CONFIDENTIAL, probability: 0.99 },
        { ...base, index: 2, start: 50, end: 60, text: "Sable Wood", choice: "invented-type", probability: 0.99 },
      ],
    });
    const out = projectFindings(IR, r, { predicate: 0.5, candidate: 0.5 });
    expect(out).toHaveLength(1);
    expect(out[0]!.entityType).toBe("client-name");
    expect(out[0]!.action).toBe("pseudonymize");
  });
});

describe("dry-run stub", () => {
  it("is deterministic per item and question, and answers every question asked", () => {
    const req = buildRequest(IR, TEXT, candidateSpans(IR, TEXT));
    const a = stubAnswers("item-1", req);
    const b = stubAnswers("item-1", req);
    expect(a).toEqual(b);
    expect(Object.keys(a.answers).sort()).toEqual(Object.keys(req.questions).sort());
    expect(stubAnswers("item-2", req).answers[PREDICATE_QUESTION_ID]).not.toEqual(a.answers[PREDICATE_QUESTION_ID]);
  });
});
