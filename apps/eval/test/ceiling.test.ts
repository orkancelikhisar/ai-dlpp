import { describe, expect, it } from "vitest";
import type { PolicyIr } from "@sih/core";
import {
  CEILING_MODELS,
  passRunIdFor,
  spendLedgerFileFor,
  REASONING_OFF,
  reasoningRequestFor,
  applyPinOverrides,
  REASONING_ON,
  runCeilingItem,
  REPRESENTATIVE_COMPLETION_TOKENS,
  REPRESENTATIVE_PROMPT_TOKENS,
  CeilingRecordSchema,
  OUTPUT_MECHANISM,
  SPEND_HARD_STOP_USD,
  SpendGuard,
  armName,
  baselineMessagesFor,
  buildRequestBody,
  callChat,
  collectBaseline,
  collectJudge,
  estimateCostUsd,
  judgeMessagesFor,
  messagePassage,
} from "../src/driver/ceiling.js";

/**
 * THE MOCKED HTTP LAYER.
 *
 * Every test in this file drives `ceiling.ts` through a fake `fetch` and a fake
 * clock. The LIVE run is the measurement, not a test: a test that talks to
 * OpenRouter would spend the key's budget, would be non-deterministic in exactly
 * the four columns this module exists to record, and would go red when a
 * provider had a bad minute. What is asserted here is that the module reads a
 * real OpenRouter response correctly -- and every fixture below is shaped from
 * a REAL one, captured from `deepseek/deepseek-v4-flash-0731` on DeepInfra
 * before any of this was written (the `: OPENROUTER PROCESSING` comment lines,
 * the trailing `data: [DONE]`, and `usage` arriving only on the final chunk are
 * all copied from that capture, not imagined).
 */

/** One SSE frame as OpenRouter writes it, `data: ` prefix and blank line included. */
function sse(obj: unknown): string {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

function contentChunk(text: string, extra: Record<string, unknown> = {}): string {
  return sse({
    id: "gen-1",
    object: "chat.completion.chunk",
    model: "deepseek/deepseek-v4-flash-0731",
    provider: "DeepInfra",
    choices: [{ index: 0, delta: { content: text, role: "assistant" }, finish_reason: null }],
    ...extra,
  });
}

function finalChunk(usage: Record<string, unknown>, finishReason = "stop"): string {
  return sse({
    id: "gen-1",
    object: "chat.completion.chunk",
    model: "deepseek/deepseek-v4-flash-0731",
    provider: "DeepInfra",
    choices: [{ index: 0, delta: { content: "" }, finish_reason: finishReason }],
    usage,
  });
}

const USAGE = {
  prompt_tokens: 26,
  completion_tokens: 61,
  total_tokens: 87,
  cost: 1.254e-5,
  completion_tokens_details: { reasoning_tokens: 0 },
};

/**
 * A `fetch` that hands back one SSE body, writing each frame only when the
 * consumer pulls -- and ADVANCING THE FAKE CLOCK between frames, so a test can
 * state what TTFT and wall time must come out as. A body delivered in one gulp
 * would make every timing assertion below vacuous.
 */
function streamingFetch(
  frames: readonly string[],
  clock: { t: number },
  msPerFrame: number,
  status = 200,
): typeof fetch {
  return (async () => {
    const encoder = new TextEncoder();
    let i = 0;
    // `highWaterMark: 0` is load-bearing, not tidiness. At the DEFAULT of 1 the
    // stream pulls one chunk AHEAD of the consumer, so the clock had already
    // advanced for frame N+1 by the time `read()` handed back frame N -- and
    // every TTFT assertion below read exactly one frame late (measured: 50ms
    // where the frames say 40ms). The mock, not the module, was wrong; this is
    // the line that makes "frame i is visible at time i*msPerFrame" true.
    const body = new ReadableStream<Uint8Array>(
      {
        pull(controller) {
          if (i >= frames.length) {
            controller.close();
            return;
          }
          clock.t += msPerFrame;
          controller.enqueue(encoder.encode(frames[i]!));
          i += 1;
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(body, { status, headers: { "content-type": "text/event-stream" } });
  }) as unknown as typeof fetch;
}

function deps(fetchImpl: typeof fetch, clock: { t: number }) {
  return {
    fetch: fetchImpl,
    now: () => clock.t,
    sleep: async (ms: number) => {
      clock.t += ms;
    },
    apiKey: "test-key-never-real",
  };
}

const MODEL = CEILING_MODELS.find((m) => m.id === "deepseek/deepseek-v4-flash-0731")!;

const CALL = {
  model: MODEL,
  messages: [{ role: "system" as const, content: "sys" }, { role: "user" as const, content: "usr" }],
  schemaName: "judge_response",
  schema: { type: "object" },
  maxTokens: 600,
};

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

describe("callChat over a mocked stream", () => {
  it("assembles the content across deltas and ignores SSE comment lines", async () => {
    const clock = { t: 1000 };
    const frames = [
      ": OPENROUTER PROCESSING\n\n",
      ": OPENROUTER PROCESSING\n\n",
      contentChunk('{"fin'),
      contentChunk('dings":[]}'),
      finalChunk(USAGE),
      "data: [DONE]\n\n",
    ];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 10), clock), CALL);
    expect(outcome.content).toBe('{"findings":[]}');
    expect(outcome.finishReason).toBe("stop");
  });

  it("times TTFT to the first CONTENT delta, not to the first byte of the stream", async () => {
    // The two comment frames and one role-only frame all arrive BEFORE any
    // content. Timed from the first byte, TTFT would be 10ms; the honest
    // number is 40ms, when the model's first token actually appeared. This is
    // the assertion that makes the latency column mean what the doc says.
    const clock = { t: 0 };
    const frames = [
      ": OPENROUTER PROCESSING\n\n",
      ": OPENROUTER PROCESSING\n\n",
      contentChunk(""), // role-only opener: a delta carrying no content
      contentChunk("abc"),
      finalChunk(USAGE),
      "data: [DONE]\n\n",
    ];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 10), clock), CALL);
    expect(outcome.ttftMs).toBe(40);
    expect(outcome.wallMs).toBe(60);
  });

  it("computes decode rate over the DECODE window (wall minus TTFT), not over wall time", async () => {
    const clock = { t: 0 };
    const frames = [contentChunk("a"), contentChunk("b"), finalChunk(USAGE), "data: [DONE]\n\n"];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 100), clock), CALL);
    // ttft 100ms, wall 400ms -> decode window 300ms -> 61 tok / 0.3 s
    expect(outcome.ttftMs).toBe(100);
    expect(outcome.wallMs).toBe(400);
    expect(outcome.decodeTokPerSec).toBeCloseTo(61 / 0.3, 6);
  });

  it("reports decode rate as undefined rather than Infinity when the decode window is zero", async () => {
    // One frame carries all the content AND is the last frame, so wall === ttft.
    // A naive division makes this Infinity, which would enter the report as a
    // model that decoded infinitely fast.
    const clock = { t: 0 };
    const frames = [contentChunk("x") + finalChunk(USAGE) + "data: [DONE]\n\n"];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 5), clock), CALL);
    expect(outcome.wallMs).toBe(outcome.ttftMs);
    expect(outcome.decodeTokPerSec).toBeUndefined();
  });

  it("reads usage, cost, reasoning tokens and the answering provider off the stream", async () => {
    const clock = { t: 0 };
    const frames = [
      contentChunk("{}"),
      finalChunk({ ...USAGE, completion_tokens_details: { reasoning_tokens: 384 } }),
      "data: [DONE]\n\n",
    ];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 1), clock), CALL);
    expect(outcome.promptTokens).toBe(26);
    expect(outcome.completionTokens).toBe(61);
    expect(outcome.reasoningTokens).toBe(384);
    expect(outcome.costUsd).toBe(1.254e-5);
    expect(outcome.provider).toBe("DeepInfra");
    expect(outcome.modelId).toBe("deepseek/deepseek-v4-flash-0731");
  });

  it("leaves every usage number undefined -- never 0 -- when the stream reported no usage", async () => {
    // A 0 would say the call used no tokens and emitted no reasoning. The
    // reasoning column is the one where that lie matters: a missing field would
    // read as a measured zero, which is exactly the finding this arm is looking
    // for.
    const clock = { t: 0 };
    const frames = [contentChunk("{}"), "data: [DONE]\n\n"];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 1), clock), CALL);
    expect(outcome.promptTokens).toBeUndefined();
    expect(outcome.completionTokens).toBeUndefined();
    expect(outcome.reasoningTokens).toBeUndefined();
    expect(outcome.costUsd).toBeUndefined();
  });

  it("records the provider the RESPONSE carried even when it is not the one pinned", async () => {
    // A pin is a request; the response is the fact. The outcome must carry what
    // answered, so the caller can refuse it -- see the assertion test below.
    const clock = { t: 0 };
    // Every frame of a re-routed call names the provider that answered, the
    // final one included -- so this fixture changes it on both, which is what
    // OpenRouter actually sends.
    const frames = [
      contentChunk("{}", { provider: "Novita" }),
      sse({
        id: "gen-1",
        model: "deepseek/deepseek-v4-flash-0731",
        provider: "Novita",
        choices: [{ index: 0, delta: { content: "" }, finish_reason: "stop" }],
        usage: USAGE,
      }),
      "data: [DONE]\n\n",
    ];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 1), clock), CALL);
    expect(outcome.provider).toBe("Novita");
  });
});

describe("the non-streaming fallback", () => {
  it("falls back to a non-streamed call after a stream times out, and marks the row", async () => {
    // MEASURED against the live slate: `qwen/qwen3.8-flash` on Alibaba streamed
    // fine for one family and then returned zero bytes and no headers on every
    // streamed request within the same hour, while the identical body
    // non-streamed answered in 2.5s. Without this path that arm is simply lost.
    const clock = { t: 0 };
    let calls = 0;
    const seen: boolean[] = [];
    const fetchImpl = (async (_url: string, init: RequestInit) => {
      calls += 1;
      seen.push(JSON.parse(String(init.body)).stream === true);
      if (calls === 1) {
        return new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new Error("The operation was aborted.")));
        });
      }
      return new Response(
        JSON.stringify({
          model: "deepseek/deepseek-v4-flash-0731",
          provider: "DeepInfra",
          choices: [{ finish_reason: "stop", message: { content: '{"findings":[]}' } }],
          usage: USAGE,
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;
    const { outcome, retries } = await callChat(deps(fetchImpl, clock), CALL, { timeoutMs: 5 });
    expect(seen).toEqual([true, false]); // streamed first, then not
    expect(outcome.transport).toBe("non-stream-fallback");
    expect(outcome.content).toBe('{"findings":[]}');
    // Accuracy, cost and reasoning survive the fallback; the two columns a
    // single timestamp cannot produce do NOT, and are null rather than imputed.
    expect(outcome.completionTokens).toBe(61);
    expect(outcome.reasoningTokens).toBe(0);
    expect(outcome.costUsd).toBe(1.254e-5);
    expect(outcome.ttftMs).toBeUndefined();
    expect(outcome.decodeTokPerSec).toBeUndefined();
    expect(retries[0]!.detail).toMatch(/retrying WITHOUT streaming/);
  });

  it("does NOT fall back when the caller asked to measure the streaming path alone", async () => {
    const clock = { t: 0 };
    const streams: boolean[] = [];
    const fetchImpl = (async (_url: string, init: RequestInit) => {
      streams.push(JSON.parse(String(init.body)).stream === true);
      return new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(new Error("The operation was aborted.")));
      });
    }) as unknown as typeof fetch;
    await expect(
      callChat(deps(fetchImpl, clock), CALL, { timeoutMs: 5, maxAttempts: 2, allowNonStreamFallback: false }),
    ).rejects.toThrow(/network failure/);
    expect(streams).toEqual([true, true]);
  });

  it("does NOT fall back on a network error that is not a timeout", async () => {
    // The fallback exists for one measured failure: a streaming path that
    // accepts the connection and sends nothing. A connection reset is a
    // different thing, it is transient, and silently dropping to non-streaming
    // on it would quietly cost the TTFT column on every flaky call.
    const clock = { t: 0 };
    const streams: boolean[] = [];
    let calls = 0;
    const fetchImpl = (async (_url: string, init: RequestInit) => {
      calls += 1;
      streams.push(JSON.parse(String(init.body)).stream === true);
      if (calls === 1) throw new Error("ECONNRESET");
      return streamingFetch([contentChunk("{}"), finalChunk(USAGE), "data: [DONE]\n\n"], clock, 1)("", {});
    }) as unknown as typeof fetch;
    const { outcome, retries } = await callChat(deps(fetchImpl, clock), CALL, { timeoutMs: 60_000 });
    expect(streams).toEqual([true, true]); // still streaming on the retry
    expect(outcome.transport).toBe("stream");
    expect(retries[0]!.detail).not.toMatch(/WITHOUT streaming/);
  });

  it("leaves the fallback's usage numbers undefined too when the provider reported none", async () => {
    // The same lie, one code path over: a non-streamed response with no usage
    // block must not report 0 reasoning tokens. Caught by a mutant that
    // survived the first round because only the streamed reader was tested.
    const clock = { t: 0 };
    let calls = 0;
    const fetchImpl = (async (_url: string, init: RequestInit) => {
      calls += 1;
      if (calls === 1) {
        return new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new Error("The operation was aborted.")));
        });
      }
      return new Response(
        JSON.stringify({
          model: "m",
          provider: "DeepInfra",
          choices: [{ finish_reason: "stop", message: { content: "{}" } }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;
    const { outcome } = await callChat(deps(fetchImpl, clock), CALL, { timeoutMs: 5 });
    expect(outcome.transport).toBe("non-stream-fallback");
    expect(outcome.promptTokens).toBeUndefined();
    expect(outcome.completionTokens).toBeUndefined();
    expect(outcome.reasoningTokens).toBeUndefined();
    expect(outcome.costUsd).toBeUndefined();
  });

  it("a streamed row is marked as such", async () => {
    const clock = { t: 0 };
    const frames = [contentChunk("{}"), finalChunk(USAGE), "data: [DONE]\n\n"];
    const { outcome } = await callChat(deps(streamingFetch(frames, clock, 1), clock), CALL);
    expect(outcome.transport).toBe("stream");
  });
});

describe("the request body", () => {
  it("pins the provider with fallbacks off, asks for usage, and disables reasoning", () => {
    const body = buildRequestBody(CALL);
    expect(body.provider).toEqual({ order: ["DeepInfra"], allow_fallbacks: false });
    expect(body.reasoning).toEqual({ enabled: false });
    expect(body.usage).toEqual({ include: true });
    expect(body.stream).toBe(true);
    expect(body.temperature).toBe(0);
  });

  it("sends the schema as a strict provider-side json_schema response_format", () => {
    const body = buildRequestBody(CALL);
    expect(body.response_format).toEqual({
      type: "json_schema",
      json_schema: { name: "judge_response", strict: true, schema: { type: "object" } },
    });
  });

  it("never lets allow_fallbacks be true, because the latency column would then name a routing decision", () => {
    for (const model of CEILING_MODELS) {
      const body = buildRequestBody({ ...CALL, model });
      expect(body.provider.allow_fallbacks).toBe(false);
      expect(body.provider.order).toEqual([model.provider]);
    }
  });
});

describe("the thinking condition", () => {
  it("asks for reasoning OFF by default -- the experiment's condition", () => {
    expect(buildRequestBody(CALL).reasoning).toEqual({ enabled: false });
  });

  it("asks for reasoning ON only when explicitly told to", () => {
    // Exists for `z-ai/glm-5.3-flash` alone, which answers HTTP 400
    // "Reasoning is mandatory for this endpoint and cannot be disabled" on all
    // eleven of its providers. Rows from such a run are a DIFFERENT CONDITION.
    expect(buildRequestBody({ ...CALL, thinking: "on" }).reasoning).toEqual({ enabled: true });
    expect(REASONING_ON).toEqual({ enabled: true });
  });

  it("records on the row what was ASKED, beside what happened", async () => {
    const clock = { t: 0 };
    const fetchImpl = (async () =>
      streamingFetch(
        [contentChunk('{"findings":[]}'), finalChunk({ ...USAGE, completion_tokens_details: { reasoning_tokens: 512 } }), "data: [DONE]\n\n"],
        clock,
        1,
      )("", {})) as unknown as typeof fetch;
    const record = await runCeilingItem({
      deps: deps(fetchImpl, clock),
      model: MODEL,
      family: "judge",
      ir: IR,
      irHash: "b".repeat(64),
      policyText: "policy",
      item: { id: "i1", text: TEXT, policy: "p-fin", gold: [] },
      runId: "t",
      maxTokens: 600,
      destinationProvider: "claude",
      thinking: "on",
      gitSha: "0".repeat(40),
      gitDirty: true,
    });
    expect(record.thinkingRequested).toBe("on");
    expect(record.reasoningRequest).toBe('{"enabled":true}');
    // The ask and the outcome are separate columns, so a model that ignores the
    // request -- in either direction -- is visible rather than assumed away.
    expect(record.calls[0]!.reasoningTokens).toBe(512);
  });
});

describe("the request body did not change between the two run windows", () => {
  /**
   * Arms 1-7 and arms 8-10 were produced by two launches of the driver with
   * edits in between, and both stamp the same dirty gitSha -- so the stamps
   * alone cannot show the code was equivalent. These assertions are what shows
   * it: the request body is pinned against a LITERAL of what the first window
   * sent, field for field. Any edit that changed a request would fail here.
   */
  it("is byte-identical to what the first window sent, for a thinking-off call", () => {
    expect(buildRequestBody(CALL)).toEqual({
      model: "deepseek/deepseek-v4-flash-0731",
      provider: { order: ["DeepInfra"], allow_fallbacks: false },
      reasoning: { enabled: false },
      usage: { include: true },
      stream: true,
      temperature: 0,
      max_tokens: 600,
      messages: CALL.messages,
      response_format: {
        type: "json_schema",
        json_schema: { name: "judge_response", strict: true, schema: { type: "object" } },
      },
    });
  });

  it("routes the default through reasoningRequestFor to the SAME frozen object the literal used", () => {
    // The one line in the request path that changed between the windows was
    // `reasoning: REASONING_OFF` becoming
    // `reasoning: reasoningRequestFor(call.thinking ?? "off")`. This is the
    // identity that makes that a no-op for every thinking-off call.
    expect(reasoningRequestFor("off")).toBe(REASONING_OFF);
    expect(buildRequestBody(CALL).reasoning).toBe(buildRequestBody({ ...CALL, thinking: "off" }).reasoning);
  });

  it("leaves the slate untouched when no pin override is given", () => {
    // The other addition. Run 2 passed no SIH_CEILING_PIN, so this is the path
    // it took.
    expect(applyPinOverrides(CEILING_MODELS, undefined)).toEqual([...CEILING_MODELS]);
  });
});

describe("retries", () => {
  it("retries a 429 with exponential backoff and records every attempt", async () => {
    const clock = { t: 0 };
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      if (calls <= 2) return new Response("rate limited", { status: 429 });
      const frames = [contentChunk("{}"), finalChunk(USAGE), "data: [DONE]\n\n"];
      return streamingFetch(frames, clock, 1)("", {});
    }) as unknown as typeof fetch;
    const { outcome, retries } = await callChat(deps(fetchImpl, clock), CALL);
    expect(outcome.content).toBe("{}");
    expect(retries).toHaveLength(2);
    expect(retries.map((r) => r.status)).toEqual([429, 429]);
    // Exponential, not flat: a flat backoff against a rate limiter is the
    // failure mode that turns one 429 into a wall of them.
    expect(retries[1]!.delayMs).toBeGreaterThan(retries[0]!.delayMs);
  });

  it("retries a 5xx", async () => {
    const clock = { t: 0 };
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      if (calls === 1) return new Response("upstream boom", { status: 503 });
      return streamingFetch([contentChunk("{}"), finalChunk(USAGE), "data: [DONE]\n\n"], clock, 1)("", {});
    }) as unknown as typeof fetch;
    const { retries } = await callChat(deps(fetchImpl, clock), CALL);
    expect(retries.map((r) => r.status)).toEqual([503]);
  });

  it("times the SUCCESSFUL attempt only -- backoff sleeps and failed attempts are excluded", async () => {
    // The load-bearing claim for every latency column in the report. DeepInfra
    // rate-limited 45-50% of the DeepSeek calls with 429s, so if `ttftMs` and
    // `wallMs` were measured from the FIRST attempt's start they would be
    // mostly the aggregator's rate limiter wearing the model's name -- a fact
    // about routing in a column labelled with a model. `started` is
    // re-initialised inside the attempt loop; this is what proves it.
    const clock = { t: 0 };
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      if (calls <= 2) return new Response("rate limited", { status: 429 });
      return streamingFetch([contentChunk("{}"), finalChunk(USAGE), "data: [DONE]\n\n"], clock, 10)("", {});
    }) as unknown as typeof fetch;
    const { outcome, retries } = await callChat(deps(fetchImpl, clock), CALL, { baseBackoffMs: 5000 });
    // 5s + 10s of backoff elapsed on the fake clock before the winning attempt.
    const backoff = retries.reduce((n, r) => n + r.delayMs, 0);
    expect(backoff).toBe(15_000);
    // The successful attempt is 3 frames x 10ms. If backoff leaked in, these
    // would be 15_000-odd.
    expect(outcome.ttftMs).toBe(10);
    expect(outcome.wallMs).toBe(30);
    expect(clock.t).toBeGreaterThan(15_000); // the sleeps really did happen
  });

  it("does NOT retry a 400, because a malformed request is not a transient failure", async () => {
    const clock = { t: 0 };
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      return new Response('{"error":{"message":"schema not supported"}}', { status: 400 });
    }) as unknown as typeof fetch;
    await expect(callChat(deps(fetchImpl, clock), CALL)).rejects.toThrow(/400/);
    expect(calls).toBe(1);
  });

  it("aborts a call that hangs past the timeout, and retries it as a network failure", async () => {
    // The first LIVE probe produced exactly this: one streamed call to
    // `qwen/qwen3.8-flash` on Alibaba stayed open past ten minutes with no
    // further frames and no error, stalling every arm queued behind it. An
    // unfinished call also has no cost, so the spend guard cannot end it.
    const clock = { t: 0 };
    let calls = 0;
    const fetchImpl = (async (_url: string, init: RequestInit) => {
      calls += 1;
      if (calls === 1) {
        // Never resolves on its own; only the abort signal ends it.
        return new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new Error("The operation was aborted.")));
        });
      }
      return streamingFetch([contentChunk("{}"), finalChunk(USAGE), "data: [DONE]\n\n"], clock, 1)("", {});
    }) as unknown as typeof fetch;
    // `allowNonStreamFallback: false` so this test still exercises the RETRY
    // path. With the default the second attempt is non-streamed, which is the
    // separate behaviour asserted in "the non-streaming fallback" above.
    const { outcome, retries } = await callChat(deps(fetchImpl, clock), CALL, {
      timeoutMs: 5,
      allowNonStreamFallback: false,
    });
    expect(outcome.content).toBe("{}");
    expect(outcome.transport).toBe("stream");
    expect(retries).toHaveLength(1);
    expect(retries[0]!.status).toBe("network");
    expect(retries[0]!.detail).toMatch(/abort/i);
  });

  it("gives up after the attempt ceiling rather than retrying into the budget forever", async () => {
    const clock = { t: 0 };
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      return new Response("nope", { status: 429 });
    }) as unknown as typeof fetch;
    await expect(callChat(deps(fetchImpl, clock), CALL, { maxAttempts: 3 })).rejects.toThrow(/429/);
    expect(calls).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// The spend guard
// ---------------------------------------------------------------------------

describe("SpendGuard", () => {
  it("accumulates the cost the RESPONSE reported, in preference to the price-table estimate", () => {
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 0, keyLimit: 10 });
    guard.record({ costUsd: 0.25, estimateUsd: 0.9 });
    guard.record({ costUsd: 0.25, estimateUsd: 0.9 });
    expect(guard.totalUsd).toBeCloseTo(0.5, 10);
    expect(guard.estimatedUsd).toBeCloseTo(1.8, 10);
    expect(guard.calls).toBe(2);
  });

  it("falls back to the estimate when the response carried no cost", () => {
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 0, keyLimit: 10 });
    guard.record({ costUsd: undefined, estimateUsd: 0.4 });
    expect(guard.totalUsd).toBeCloseTo(0.4, 10);
  });

  it("TRIPS at the hard stop", () => {
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 0, keyLimit: 10 });
    guard.record({ costUsd: 6.99, estimateUsd: 6.99 });
    expect(guard.tripped()).toBe(false);
    guard.record({ costUsd: 0.02, estimateUsd: 0.02 });
    expect(guard.tripped()).toBe(true);
  });

  it("trips exactly AT the stop, not only above it", () => {
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 0, keyLimit: 10 });
    guard.record({ costUsd: 7, estimateUsd: 7 });
    expect(guard.tripped()).toBe(true);
  });

  it("counts spend that was already on the key before this run", () => {
    // The stop is against what the KEY has spent, not what this process has. A
    // guard that ignored `keyUsageAtStart` would let a second run spend the
    // whole limit again.
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 6.5, keyLimit: 10 });
    guard.record({ costUsd: 0.6, estimateUsd: 0.6 });
    expect(guard.tripped()).toBe(true);
    expect(guard.totalUsd).toBeCloseTo(0.6, 10);
    expect(guard.keySpendUsd).toBeCloseTo(7.1, 10);
  });

  it("asks for a key re-read every 50 calls and not otherwise", () => {
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 0, keyLimit: 10, checkpointEvery: 50 });
    for (let i = 0; i < 49; i++) guard.record({ costUsd: 0.0001, estimateUsd: 0.0001 });
    expect(guard.dueForCheckpoint()).toBe(false);
    guard.record({ costUsd: 0.0001, estimateUsd: 0.0001 });
    expect(guard.dueForCheckpoint()).toBe(true);
    guard.noteCheckpoint(0.005);
    expect(guard.dueForCheckpoint()).toBe(false);
    expect(guard.keyUsageLatest).toBe(0.005);
  });

  it("trips on the key's OWN reported usage at a checkpoint, even when the summed costs disagree", () => {
    // The per-response `cost` is what OpenRouter said each call cost; the key
    // endpoint is what it actually billed. If they diverge, the billed number
    // is the one that can exhaust the key, so it must be able to stop the run
    // on its own.
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 0, keyLimit: 10 });
    guard.record({ costUsd: 0.01, estimateUsd: 0.01 });
    expect(guard.tripped()).toBe(false);
    guard.noteCheckpoint(8.2);
    expect(guard.tripped()).toBe(true);
  });

  it("the shipped hard stop leaves margin under the key's $10 cap", () => {
    expect(SPEND_HARD_STOP_USD).toBe(7);
  });

  it("snapshots the whole ledger, so a crash leaves an accurate file behind", () => {
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 1.5, keyLimit: 10 });
    guard.record({ costUsd: 0.2, estimateUsd: 0.3, model: "deepseek/deepseek-v4-flash-0731", family: "judge" });
    guard.record({ costUsd: 0.1, estimateUsd: 0.1, model: "deepseek/deepseek-v4-flash-0731", family: "b" });
    const snap = guard.snapshot();
    expect(snap.calls).toBe(2);
    expect(snap.costUsd).toBeCloseTo(0.3, 10);
    expect(snap.keyUsageAtStart).toBe(1.5);
    expect(snap.keyLimit).toBe(10);
    expect(snap.byModel["deepseek/deepseek-v4-flash-0731"]).toBeCloseTo(0.3, 10);
    expect(snap.byFamily["judge"]).toBeCloseTo(0.2, 10);
    expect(snap.byFamily["b"]).toBeCloseTo(0.1, 10);
  });
});

describe("estimateCostUsd", () => {
  it("prices prompt and completion tokens off the pinned endpoint's own per-million rates", () => {
    // DeepSeek/DeepInfra: $0.06/M in, $0.18/M out.
    expect(estimateCostUsd(MODEL, 1_000_000, 0)).toBeCloseTo(0.06, 10);
    expect(estimateCostUsd(MODEL, 0, 1_000_000)).toBeCloseTo(0.18, 10);
  });

  it("treats missing token counts as zero rather than NaN, so one bad response cannot poison the ledger", () => {
    expect(estimateCostUsd(MODEL, undefined, undefined)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

const GOOD_ROW = {
  schemaVersion: 1 as const,
  runId: "ceiling-01",
  itemId: "inj-o01-0",
  policy: "p-fin",
  irHash: "b".repeat(64),
  policyHash: "ebb3cd68d973175f3ea40faeec00685e2cb9d83c6940e96a57e88ee269e8110a",
  arm: "ceiling-judge-deepseek-v4-flash",
  family: "judge" as const,
  requestedModelId: "deepseek/deepseek-v4-flash-0731",
  modelId: "deepseek/deepseek-v4-flash-0731",
  requestedProvider: "DeepInfra",
  provider: "DeepInfra",
  quantization: "fp8",
  outputMechanism: OUTPUT_MECHANISM,
  thinkingRequested: "off" as const,
  reasoningRequest: '{"enabled":false}',
  gitSha: "0".repeat(40),
  gitDirty: true,
  text: "Tamarind Grocers is our client.",
  findings: [
    {
      start: 0,
      end: 16,
      text: "Tamarind Grocers",
      entityType: "pred:client-relationship-disclosure",
      severity: "high" as const,
      tier: 2 as const,
      source: "deepseek/deepseek-v4-flash-0731",
      confidence: 0.9,
      action: "redact" as const,
    },
  ],
  gold: [],
  calls: [
    {
      promptTokens: 26,
      completionTokens: 61,
      reasoningTokens: 0,
      ttftMs: 120,
      wallMs: 640,
      decodeTokPerSec: 117.3,
      costUsd: 1.254e-5,
      finishReason: "stop",
      provider: "DeepInfra",
      modelId: "deepseek/deepseek-v4-flash-0731",
      transport: "stream" as const,
      repair: false,
      parse: "ok" as const,
      retries: [],
    },
  ],
  parseFailures: 0,
  repairs: 0,
  unresolvedQuotes: 0,
  unresolvedMentions: 0,
  unknownLabels: 0,
  duplicatesDropped: 0,
  wholeClauseMentions: 0,
  wallMs: 640,
  error: null,
};

describe("CeilingRecordSchema", () => {
  it("accepts a well-formed row", () => {
    expect(() => CeilingRecordSchema.parse(GOOD_ROW)).not.toThrow();
  });

  it("refuses a finding whose offsets do not slice back to its own text", () => {
    // The same cross-check RunRecordSchema runs, and for the same reason: a
    // scorer joining findings to gold must be able to verify the offsets from
    // the record alone.
    const bad = { ...GOOD_ROW, findings: [{ ...GOOD_ROW.findings[0]!, end: 8 }] };
    expect(() => CeilingRecordSchema.parse(bad)).toThrow(/do not hold the text/);
  });

  it("requires all four timing columns on every call row", () => {
    for (const field of ["ttftMs", "wallMs", "reasoningTokens", "decodeTokPerSec"] as const) {
      const call = { ...GOOD_ROW.calls[0]! } as Record<string, unknown>;
      delete call[field];
      const row = { ...GOOD_ROW, calls: [call] };
      // Only the two that are always measurable are required outright; the two
      // that a provider can legitimately fail to report must be PRESENT and
      // explicitly null, never simply missing.
      expect(() => CeilingRecordSchema.parse(row), `${field} must be accounted for`).toThrow();
    }
  });

  it("allows a null decode rate and a null reasoning count, because 'not reported' is not zero", () => {
    const row = {
      ...GOOD_ROW,
      calls: [{ ...GOOD_ROW.calls[0]!, decodeTokPerSec: null, reasoningTokens: null, costUsd: null }],
    };
    expect(() => CeilingRecordSchema.parse(row)).not.toThrow();
  });

  it("pins outputMechanism to the provider-side json_schema mechanism", () => {
    expect(OUTPUT_MECHANISM).toBe("provider-json-schema");
    expect(() => CeilingRecordSchema.parse({ ...GOOD_ROW, outputMechanism: "xgrammar" })).toThrow();
  });

  it("refuses a row that claims thinking was off while naming no reasoning request", () => {
    expect(() => CeilingRecordSchema.parse({ ...GOOD_ROW, reasoningRequest: "" })).toThrow();
  });

  it("keeps requestedProvider and provider as SEPARATE fields, so a pin can be checked against the fact", () => {
    const drifted = { ...GOOD_ROW, provider: "Novita" };
    const parsed = CeilingRecordSchema.parse(drifted);
    expect(parsed.requestedProvider).toBe("DeepInfra");
    expect(parsed.provider).toBe("Novita");
  });

  it("carries the completion cap it ran at, beside the local arms' own value", async () => {
    // The 88-token asymmetry (600 here vs DEFAULT_TIER2_CONFIG's 512) came from
    // a comment nobody had checked. A budget asymmetry that is not on the row
    // is one a later reader cannot find.
    const clock = { t: 0 };
    const fetchImpl = (async () =>
      streamingFetch([contentChunk('{"findings":[]}'), finalChunk(USAGE), "data: [DONE]\n\n"], clock, 1)(
        "",
        {},
      )) as unknown as typeof fetch;
    const record = await runCeilingItem({
      deps: deps(fetchImpl, clock),
      model: MODEL,
      family: "judge",
      ir: IR,
      irHash: "b".repeat(64),
      policyText: "policy",
      item: { id: "i1", text: TEXT, policy: "p-fin", gold: [] },
      runId: "t",
      maxTokens: 600,
      localArmMaxTokens: 512,
      destinationProvider: "claude",
      gitSha: "0".repeat(40),
      gitDirty: true,
    });
    expect(record.maxTokens).toBe(600);
    expect(record.localArmMaxTokens).toBe(512);
  });

  it("still accepts a row from the first paid run, which predates the cap fields", () => {
    // OPTIONAL rather than required, deliberately: the first run's rows cannot
    // be back-filled without inventing provenance, and refusing to parse them
    // would throw away the measurement this arm exists for.
    const { maxTokens: _a, localArmMaxTokens: _b, ...older } = { ...GOOD_ROW, maxTokens: 600, localArmMaxTokens: 512 };
    expect(() => CeilingRecordSchema.parse(older)).not.toThrow();
  });

  it("requires a 40-hex git sha and a dirty flag, so a row is traceable to a revision", () => {
    // No artifact already in `runs/` names the code revision that produced it.
    // A row that omitted this, or carried a short sha, would inherit that.
    expect(() => CeilingRecordSchema.parse({ ...GOOD_ROW, gitSha: "abc123" })).toThrow();
    const { gitDirty: _drop, ...withoutFlag } = GOOD_ROW;
    expect(() => CeilingRecordSchema.parse(withoutFlag)).toThrow();
    expect(CeilingRecordSchema.parse(GOOD_ROW).gitDirty).toBe(true);
  });

  it("is structurally scoreable: it carries every field scoreArm reads off a run record", () => {
    const parsed = CeilingRecordSchema.parse(GOOD_ROW);
    expect(parsed.arm).toBeTypeOf("string");
    expect(parsed.itemId).toBeTypeOf("string");
    expect(parsed.policyHash).toBeTypeOf("string");
    expect(parsed.text).toBeTypeOf("string");
    expect(parsed.error).toBeNull();
    expect(Array.isArray(parsed.findings)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// The two families: the prompts are IMPORTED, never rebuilt
// ---------------------------------------------------------------------------

const IR: PolicyIr = {
  irVersion: 1,
  policyHash: "ebb3cd68d973175f3ea40faeec00685e2cb9d83c6940e96a57e88ee269e8110a",
  entityTypes: [
    {
      id: "client-name",
      nlDefinition: "a client organisation name",
      severity: "high",
      examples: [],
      counterExamples: [],
    },
    {
      id: "pred:client-relationship-disclosure",
      nlDefinition: "shadow",
      severity: "high",
      examples: [],
      counterExamples: [],
    },
  ],
  rules: [],
  semanticPredicates: [
    {
      id: "client-relationship-disclosure",
      nlPredicate: "The message discloses that a named organisation is a client of the Firm.",
      scope: "message",
    },
  ],
  // Core's own `Actions` shape: a `default` map keyed by entityType, not a
  // list. `resolveFindings` reads it through `resolveAction`, so a list here
  // throws inside core rather than failing a shape check here.
  actions: {
    default: { "client-name": "pseudonymize", "pred:client-relationship-disclosure": "redact" },
  },
  failMode: "fail-closed",
  latencyBudgetMs: 5000,
  provenance: {
    "client-name": { clause: "S3.1", quote: "client names" },
    "pred:client-relationship-disclosure": { clause: "S3.3", quote: "relationship" },
  },
} as unknown as PolicyIr;

const TEXT = "Tamarind Grocers is our client and the retainer starts in June.";

describe("the judge family", () => {
  it("sends the compiled judge's own two turns, with the predicate and the whole message", () => {
    const messages = judgeMessagesFor(IR, TEXT);
    expect(messages).toHaveLength(2);
    expect(messages[0]!.role).toBe("system");
    // Keyed on the judge prompt's own words. If this drifts, the ceiling arm is
    // no longer running the local arm's method and the comparison is void.
    expect(messages[0]!.content).toContain("You audit one passage of a message against a list of policy predicates.");
    expect(messages[0]!.content).toContain('"predicateId"');
    expect(messages[1]!.content).toContain("client-relationship-disclosure: The message discloses");
    expect(messages[1]!.content).toContain(TEXT);
  });

  it("tells the model that earlier tiers found nothing -- the tier-2-only arm's own prior line", () => {
    expect(judgeMessagesFor(IR, TEXT)[1]!.content).toContain("Earlier tiers found nothing in this passage.");
  });

  it("judges the whole message as one passage, because p-fin's only predicate is message-scoped", () => {
    const passage = messagePassage(TEXT);
    expect(passage.start).toBe(0);
    expect(passage.end).toBe(TEXT.length);
    expect(passage.text).toBe(TEXT);
  });
});

describe("the Approach-B family", () => {
  it("sends B's own two turns, carrying the whole policy document and every entityType id", () => {
    const policyText = "# p-fin\n\nSection 3.1: client names must be pseudonymised.\n";
    const messages = baselineMessagesFor(IR, policyText, TEXT);
    expect(messages[0]!.content).toContain("You audit one message against a policy document.");
    expect(messages[0]!.content).toContain('"entityType"');
    expect(messages[1]!.content).toContain(policyText);
    expect(messages[1]!.content).toContain("- client-name");
    expect(messages[1]!.content).toContain("- pred:client-relationship-disclosure");
    expect(messages[1]!.content).toContain(TEXT);
  });

  it("carries NO prior-tier line, because this arm runs no tier 0", () => {
    const messages = baselineMessagesFor(IR, "policy", TEXT);
    expect(messages[1]!.content).not.toContain("Earlier tiers");
  });
});

describe("collecting findings through the shared span ladder", () => {
  it("places the judge's mention inside its quote and emits the shadow entityType", () => {
    const out = collectJudge(IR, TEXT, {
      findings: [
        {
          predicateId: "client-relationship-disclosure",
          quote: "Tamarind Grocers is our client and the retainer starts in June.",
          mention: "Tamarind Grocers",
          confidence: 0.8,
        },
      ],
    }, "m");
    expect(out.findings).toHaveLength(1);
    expect(out.findings[0]!.entityType).toBe("pred:client-relationship-disclosure");
    expect(out.findings[0]!.start).toBe(0);
    expect(out.findings[0]!.end).toBe(16);
    expect(out.findings[0]!.text).toBe("Tamarind Grocers");
  });

  it("drops an invented predicateId and counts it rather than throwing the message away", () => {
    const out = collectJudge(IR, TEXT, {
      findings: [{ predicateId: "not-a-predicate", quote: TEXT, mention: "Tamarind Grocers", confidence: 0.5 }],
    }, "m");
    expect(out.findings).toHaveLength(0);
    expect(out.counters.unknownLabels).toBe(1);
  });

  it("counts an unplaceable quote apart from an unplaceable mention", () => {
    const a = collectJudge(IR, TEXT, {
      findings: [{ predicateId: "client-relationship-disclosure", quote: "nowhere in this message at all", mention: "x", confidence: 0.5 }],
    }, "m");
    expect(a.counters.unresolvedQuotes).toBe(1);
    expect(a.counters.unresolvedMentions).toBe(0);
    const b = collectJudge(IR, TEXT, {
      findings: [{ predicateId: "client-relationship-disclosure", quote: TEXT, mention: "Vetiver Logistics", confidence: 0.5 }],
    }, "m");
    expect(b.counters.unresolvedQuotes).toBe(0);
    expect(b.counters.unresolvedMentions).toBe(1);
  });

  it("drops a duplicate span rather than counting one piece of evidence twice", () => {
    const one = {
      predicateId: "client-relationship-disclosure",
      quote: TEXT,
      mention: "Tamarind Grocers",
      confidence: 0.7,
    };
    const out = collectJudge(IR, TEXT, { findings: [one, { ...one, confidence: 0.6 }] }, "m");
    expect(out.findings).toHaveLength(1);
    expect(out.counters.duplicatesDropped).toBe(1);
  });

  it("places Approach B's findings under the same ladder and keeps its own entityType", () => {
    const out = collectBaseline(IR, TEXT, {
      findings: [
        { entityType: "client-name", quote: TEXT, mention: "Tamarind Grocers", confidence: 0.9 },
      ],
    }, "m");
    expect(out.findings).toHaveLength(1);
    expect(out.findings[0]!.entityType).toBe("client-name");
    expect(out.findings[0]!.start).toBe(0);
    expect(out.findings[0]!.end).toBe(16);
  });

  it("drops an invented entityType on the B side too", () => {
    const out = collectBaseline(IR, TEXT, {
      findings: [{ entityType: "made-up", quote: TEXT, mention: "Tamarind Grocers", confidence: 0.9 }],
    }, "m");
    expect(out.findings).toHaveLength(0);
    expect(out.counters.unknownLabels).toBe(1);
  });

  it("counts a whole-clause mention, so an arm that never narrows is visible", () => {
    const out = collectJudge(IR, TEXT, {
      findings: [{ predicateId: "client-relationship-disclosure", quote: TEXT, mention: TEXT, confidence: 0.5 }],
    }, "m");
    expect(out.counters.wholeClauseMentions).toBe(1);
    expect(out.findings[0]!.start).toBe(0);
    expect(out.findings[0]!.end).toBe(TEXT.length);
  });
});

describe("runCeilingItem: one repair, then fail closed", () => {
  const ITEM = { id: "i1", text: TEXT, policy: "p-fin", gold: [] };
  const base = (fetchImpl: typeof fetch, clock: { t: number }) => ({
    deps: deps(fetchImpl, clock),
    model: MODEL,
    family: "judge" as const,
    ir: IR,
    irHash: "b".repeat(64),
    policyText: "policy",
    item: ITEM,
    runId: "t",
    maxTokens: 600,
    destinationProvider: "claude",
    gitSha: "0".repeat(40),
    gitDirty: true,
  });

  /** A fetch that answers with each body in turn. */
  function bodies(list: readonly string[], clock: { t: number }): typeof fetch {
    let i = 0;
    return (async () => {
      const body = list[Math.min(i, list.length - 1)]!;
      i += 1;
      return streamingFetch([contentChunk(body), finalChunk(USAGE), "data: [DONE]\n\n"], clock, 1)("", {});
    }) as unknown as typeof fetch;
  }

  it("retries ONCE on an unparseable body and keeps the repaired answer", async () => {
    const clock = { t: 0 };
    const good = JSON.stringify({
      findings: [
        {
          predicateId: "client-relationship-disclosure",
          quote: TEXT,
          mention: "Tamarind Grocers",
          confidence: 0.8,
        },
      ],
    });
    const record = await runCeilingItem(base(bodies(["not json at all", good], clock), clock));
    expect(record.repairs).toBe(1);
    expect(record.parseFailures).toBe(1);
    expect(record.calls).toHaveLength(2);
    expect(record.calls[0]!.repair).toBe(false);
    expect(record.calls[1]!.repair).toBe(true);
    // The repair row is a real call and carries its own timings and cost, so a
    // repair is visible in the spend and latency columns rather than free.
    expect(record.calls[1]!.costUsd).toBe(1.254e-5);
    expect(record.findings).toHaveLength(1);
  });

  it("stops after ONE repair and emits NO findings rather than retrying into compliance", async () => {
    // An arm that retries until the JSON parses is measuring persistence, not
    // capability. Two calls, then the answer is dropped.
    const clock = { t: 0 };
    const record = await runCeilingItem(base(bodies(["nope", "still nope", "nope"], clock), clock));
    expect(record.calls).toHaveLength(2);
    expect(record.repairs).toBe(1);
    expect(record.parseFailures).toBe(2);
    expect(record.findings).toEqual([]);
    expect(record.error).toBeNull();
  });

  it("records the answering provider and the pin separately on the row", async () => {
    const clock = { t: 0 };
    const record = await runCeilingItem(base(bodies(['{"findings":[]}'], clock), clock));
    expect(record.requestedProvider).toBe("DeepInfra");
    expect(record.provider).toBe("DeepInfra");
    expect(record.thinkingRequested).toBe("off");
    expect(record.reasoningRequest).toBe('{"enabled":false}');
    expect(record.calls[0]!.reasoningTokens).toBe(0);
  });

  it("writes a row with an error rather than throwing, when every attempt fails", async () => {
    const clock = { t: 0 };
    const fetchImpl = (async () => new Response("bad request", { status: 400 })) as unknown as typeof fetch;
    const record = await runCeilingItem(base(fetchImpl, clock));
    expect(record.error).toMatch(/400/);
    expect(record.findings).toEqual([]);
    expect(record.calls).toEqual([]);
  });
});

describe("passRunIdFor", () => {
  it("numbers passes from the base id", () => {
    expect(passRunIdFor("ceiling-01", 1)).toBe("ceiling-01");
    expect(passRunIdFor("ceiling-01", 2)).toBe("ceiling-02");
    expect(passRunIdFor("ceiling-01", 3)).toBe("ceiling-03");
  });

  it("does NOT collapse a later run id onto pass 1 -- the overwrite this function exists to stop", () => {
    // The inline version this replaced returned "ceiling-01" for BOTH of these,
    // so launching a follow-up run as `ceiling-02` would have silently
    // overwritten pass 1's ten arm files.
    expect(passRunIdFor("ceiling-02", 1, 2)).toBe("ceiling-02");
    expect(passRunIdFor("ceiling-01", 1, 3)).toBe("ceiling-03");
    expect(passRunIdFor("ceiling-01", 2, 2)).toBe("ceiling-03");
  });

  it("gives a relaunch a LEDGER FILE that cannot overwrite the earlier run's", () => {
    // This asserts the FUNCTION THE DRIVER CALLS, not the id arithmetic behind
    // it. An earlier version of this test pinned `passRunIdFor` alone and a
    // mutant reverting the call site to the base `runId` SURVIVED it -- the
    // "expectation that cannot fail" defect. `spendLedgerFileFor` exists so the
    // wiring is what gets asserted.
    expect(spendLedgerFileFor("ceiling-01", 1)).toBe("ceiling-ceiling-01.spend.json");
    expect(spendLedgerFileFor("ceiling-01", 2)).toBe("ceiling-ceiling-02.spend.json");
    expect(spendLedgerFileFor("ceiling-01", 3)).toBe("ceiling-ceiling-03.spend.json");
    // The property that was violated on disk: a relaunch must not resolve to the
    // original run's ledger.
    expect(spendLedgerFileFor("ceiling-01", 2)).not.toBe(spendLedgerFileFor("ceiling-01", 1));
  });

  it("numbers a relaunch's passes from passStart", () => {
    // The second half of the same footgun, and it was live: `passRunIdFor`
    // fixed the ARM path, but every `writeSpend` call still passed the base
    // `runId`, so a relaunch at passStart=2 wrote `ceiling-ceiling-01.spend.json`
    // -- pass 1's window-2 ledger, 768 calls and $0.19269 -- while its arm files
    // were correctly named `ceiling-02.*`. MEASURED on disk: that file held
    // calls=403 / $0.02033 eight seconds after the relaunch began.
    //
    // `ceiling-main.ts` now derives `ledgerRunId = passRunIdFor(runId, 1, passStart)`,
    // so the property that has to hold is that the FIRST pass id differs
    // between an original run and a relaunch.
    const original = passRunIdFor("ceiling-01", 1, 1);
    const relaunch = passRunIdFor("ceiling-01", 1, 2);
    expect(original).toBe("ceiling-01");
    expect(relaunch).toBe("ceiling-02");
    expect(relaunch).not.toBe(original);
    expect(`ceiling-${relaunch}.spend.json`).not.toBe(`ceiling-${original}.spend.json`);
  });

  it("keeps a base id that carries no trailing number", () => {
    expect(passRunIdFor("glmon", 1)).toBe("glmon-01");
  });
});

describe("arm naming", () => {
  it("names an arm by family and model, as the score table reads it", () => {
    expect(armName("judge", MODEL)).toBe("ceiling-judge-deepseek-v4-flash-0731");
    expect(armName("b", MODEL)).toBe("ceiling-b-deepseek-v4-flash-0731");
  });

  it("gives every model a distinct arm name in both families", () => {
    const names = CEILING_MODELS.flatMap((m) => [armName("judge", m), armName("b", m)]);
    expect(new Set(names).size).toBe(names.length);
  });
});

describe("applyPinOverrides", () => {
  it("re-pins the named model and leaves every other alone", () => {
    const out = applyPinOverrides(CEILING_MODELS, "z-ai/glm-5.3-flash=NextBit");
    expect(out.find((m) => m.id === "z-ai/glm-5.3-flash")!.provider).toBe("NextBit");
    expect(out.find((m) => m.id === "deepseek/deepseek-v4-flash-0731")!.provider).toBe("DeepInfra");
  });

  it("drops the quantization it can no longer vouch for", () => {
    // The old pin's `fp8` was a fact about BaseTen. Carrying it onto NextBit
    // would assert a property of an endpoint that was never asked.
    const out = applyPinOverrides(CEILING_MODELS, "z-ai/glm-5.3-flash=NextBit");
    expect(CEILING_MODELS.find((m) => m.id === "z-ai/glm-5.3-flash")!.quantization).toBe("fp8");
    expect(out.find((m) => m.id === "z-ai/glm-5.3-flash")!.quantization).toBe("unknown");
  });

  it("returns the slate unchanged for an empty or absent spec", () => {
    expect(applyPinOverrides(CEILING_MODELS, undefined)).toEqual([...CEILING_MODELS]);
    expect(applyPinOverrides(CEILING_MODELS, "  ")).toEqual([...CEILING_MODELS]);
  });

  it("REFUSES a model id that is not on the slate, rather than silently doing nothing", () => {
    expect(() => applyPinOverrides(CEILING_MODELS, "not/a-model=Somewhere")).toThrow(/not on the slate/);
  });

  it("REFUSES a malformed spec", () => {
    expect(() => applyPinOverrides(CEILING_MODELS, "z-ai/glm-5.3-flash")).toThrow(/not <modelId>=<provider>/);
  });
});

describe("the model slate", () => {
  it("runs cheapest-first FOR THIS WORKLOAD, so a tripped guard leaves the most expensive arm partial", () => {
    // Ordered on the cost of the run that is actually made, not on
    // `in + out` unweighted. This corpus is prompt-heavy and output-light --
    // Approach B alone carries the whole 5,320-char policy on every call -- so
    // the input rate dominates, and the two orderings genuinely disagree:
    // by the unweighted sum GLM-Flash (0.65) sorts ahead of Qwen-Flash (0.62),
    // and by this workload's mix Qwen-Flash is the cheaper of the two.
    const cost = CEILING_MODELS.map((m) =>
      estimateCostUsd(m, REPRESENTATIVE_PROMPT_TOKENS, REPRESENTATIVE_COMPLETION_TOKENS),
    );
    expect([...cost].sort((a, b) => a - b)).toEqual(cost);
  });

  it("pins a provider and a quantization for every model, with 'unknown' a permitted VALUE", () => {
    for (const m of CEILING_MODELS) {
      expect(m.provider.length).toBeGreaterThan(0);
      expect(m.quantization.length).toBeGreaterThan(0);
      expect(m.pricePerMTokIn).toBeGreaterThan(0);
      expect(m.pricePerMTokOut).toBeGreaterThan(0);
    }
  });

  it("holds the six models the ceiling arm is defined over", () => {
    expect(CEILING_MODELS).toHaveLength(6);
    expect(CEILING_MODELS.map((m) => m.id).sort()).toEqual([
      "deepseek/deepseek-v4-flash-0731",
      "mistralai/mistral-small-2603",
      "nvidia/nemotron-3-super-120b-a12b",
      "qwen/qwen3.8-27b",
      "qwen/qwen3.8-flash",
      "z-ai/glm-5.3-flash",
    ]);
  });
});
