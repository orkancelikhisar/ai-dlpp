/**
 * THE CAPABILITY-CEILING ARM (spec 4.2b), over hosted open-weight models.
 *
 * ## What this is, and what it is emphatically not
 *
 * Spec 4.2b asks for a ceiling arm to separate two explanations of the tier-2
 * result: is the task hard, or are 2-4B browser-runnable models too small? It
 * describes that arm as a LOCAL server (llama.cpp/Ollama), which keeps the
 * cloud boundary this project cares about intact. **This module does not do
 * that.** It runs six larger open-weight models over the OpenRouter API, so
 * every corpus message leaves the machine. That is a real difference from the
 * spec and it is stated here rather than buried: this arm answers the
 * CAPABILITY question 4.2b poses -- the models are open-weight and could be
 * served locally on adequate hardware -- and it answers NOTHING about the
 * privacy boundary. It is not a shippable configuration, it is not a product
 * column, and no number it produces may be presented beside the browser arms
 * without that label.
 *
 * What is preserved from the local arms, exactly, is the METHOD. Both prompts,
 * both response schemas, the span ladder and the parse-and-one-repair rule are
 * IMPORTED from `@sih/tier2` -- not reimplemented here -- because the whole
 * claim of this arm is "same method, bigger model". A second copy of either
 * prompt would turn the comparison into a comparison of prompts. `judge.ts` and
 * `baselineB.ts` gained four aliased exports so that this file could reuse
 * them; `packages/tier2/test/prompts.test.ts` pins both system turns line by
 * line and is the check that nothing drifted in the process.
 *
 * ## The one mechanism that genuinely differs, and is therefore recorded
 *
 * The browser arms constrain decoding with xgrammar, compiled from the schema
 * inside WebLLM. This arm sends the same schema as an OpenRouter
 * `response_format: {type: "json_schema", strict: true}` and the PROVIDER
 * constrains decoding, by whatever means it uses. Those are different
 * mechanisms with different failure modes, so every row carries
 * `outputMechanism: "provider-json-schema"` and the write-up says so. Parse
 * failures and repairs are counted per model for the same reason: a model that
 * needed repairs on many calls is a finding about the mechanism, not only about
 * the model.
 *
 * ## Why a sibling record schema rather than an extension of RunRecordSchema
 *
 * `CeilingRecordSchema` deliberately does NOT extend `RunRecordSchema`, and the
 * reason is the defect that file spends most of its comments warning about:
 * intent recorded in place of fact. To satisfy `RunRecordSchema` a row from
 * here would have to claim
 *
 *   - `backend: "wasm" | "webgpu"` -- an HTTPS call to a remote GPU is neither;
 *   - `detector: "core-orchestrator" | "approach-b"` -- neither implementation
 *     ran; this file did;
 *   - a `tier2Config` naming a context window and per-call budget that a hosted
 *     endpoint does not expose;
 *   - and, through the refines, a `Tier2Stats` or `BaselineStats` object whose
 *     fields describe events (engine latching, caller aborts mid-generation)
 *     that cannot occur over HTTP.
 *
 * Four lies to reuse one schema. Instead this row carries the same identity and
 * scoring fields VERBATIM -- `runId`, `itemId`, `policy`, `irHash`,
 * `policyHash`, `arm`, `text`, `findings`, `gold`, `error` -- plus honest
 * transport telemetry, and `score.ts` reads both through the structural
 * `ScoreableRecord` type. One scorer, one gold, one set of floors; two record
 * shapes, each stating what actually happened.
 */
import { z } from "zod";
import type { Finding, PolicyIr, Segment, SemanticPredicate } from "@sih/core";
import { resolveFindings, shadowIdFor } from "@sih/core";
import {
  BASELINE_B_SCHEMA,
  JUDGE_SCHEMA,
  buildBaselineMessages,
  buildJudgeMessages,
  baselineRepairMessage,
  judgeRepairMessage,
  locateFinding,
  parseBaselineResponse,
  parseJudgeResponse,
  type BaselineResponse,
  type JudgeResponse,
} from "@sih/tier2";
import { RecordFindingSchema } from "./record.js";

// ---------------------------------------------------------------------------
// The slate
// ---------------------------------------------------------------------------

/**
 * The mechanism the provider used to constrain decoding, as a VALUE on every
 * row. The browser arms' equivalent is xgrammar; a table pooling the two
 * without this column would attribute a provider's JSON-mode quirks to a model.
 */
export const OUTPUT_MECHANISM = "provider-json-schema";

/**
 * The cumulative-spend ceiling, in USD, below the key's own $10 cap.
 *
 * The margin is not decoration. The run's own accounting can lag the key's --
 * a response's `cost` is what OpenRouter reported for that call, and the key
 * endpoint is what it billed -- and the final `/auth/key` read after the run
 * must itself succeed. $3 of headroom covers both plus any in-flight
 * concurrency at the moment the guard trips.
 */
export const SPEND_HARD_STOP_USD = 7;

export interface CeilingModel {
  /** The OpenRouter model id, as `/api/v1/models` spells it. */
  readonly id: string;
  /**
   * The provider `provider.order` is pinned to, VERIFIED against
   * `/api/v1/models/{id}/endpoints` before the run. A pin is a REQUEST; the
   * `provider` a response carries is the fact, and both go on every row.
   */
  readonly provider: string;
  /**
   * The pinned endpoint's own `quantization` string. `"unknown"` is a real
   * value here and is never guessed at: two of these six endpoints report it,
   * and recording fp8 for one of them because the model is usually served that
   * way would be a fabricated fact about the thing under test.
   */
  readonly quantization: string;
  /** USD per million prompt tokens at the pinned endpoint. */
  readonly pricePerMTokIn: number;
  /** USD per million completion tokens at the pinned endpoint. */
  readonly pricePerMTokOut: number;
  /** Where the pinned endpoint is hosted, as far as the provider states it. */
  readonly hosting: string;
}

/**
 * The token mix `CEILING_MODELS` is ordered by, and the only honest way to say
 * "cheapest first" for a workload this lopsided.
 *
 * ESTIMATED, not measured, at the time the slate was written: the compiled
 * judge's prompt ran 252-280 tokens on the local bake-off and Approach B's ran
 * 1,410 (it carries the whole 5,320-char policy document), so one item across
 * both families is ~1,700 prompt tokens; completions on the local arms sat near
 * 100 each. The real numbers land in every record's `promptTokens` /
 * `completionTokens`, and the write-up checks these two against them.
 *
 * WHICH COMPARISON THIS ACTUALLY DECIDES. Only one, and it is not the pair an
 * earlier version of this comment named. COMPUTED from the six price pairs
 * below: an ascending sort on the unweighted `in + out` sum gives 0.24 < 0.62 <
 * 0.65 < 0.75 < 0.95 < 2.44, which is the shipped order EXACTLY -- so on this
 * slate the sum and the weighting do not disagree anywhere, and no example of
 * the form "the sum would have ordered X before Y" exists to be given.
 * GLM-Flash and Qwen-Flash in particular cannot be that example twice over:
 * 0.62 < 0.65 puts Qwen-Flash first under the sum as well, and their input
 * rates are EQUAL (0.15 each), so no positive weighting can separate them at
 * all -- the comparison reduces to the output rate under every mix.
 *
 * The one pair a mix CAN reorder is nemotron (0.30/0.65) against qwen-27b
 * (0.24/2.20). They cost the same when `p * 0.06 == c * 1.55`, i.e. at a
 * prompt/completion ratio of 25.83; below it nemotron is cheaper, above it
 * qwen-27b is. The estimate above sits at 8.5 and the shipped order follows.
 *
 * The MIX THAT ACTUALLY RAN sits far closer to that edge than the estimate did.
 * MEASURED here on 2026-09-08 over `runs/ceiling-01.*` (first call per item;
 * 941 judge items, 913 Approach-B items): median prompt tokens 433 for judge
 * and 1,560 for B, median completions 7 and 71 -- so one item across both
 * families is 1,993 prompt and 78 completion tokens, a ratio of 25.55. That is
 * 1.1% UNDER the 25.83 crossover: the shipped order survives the correction,
 * but nemotron beats qwen-27b on this workload by 0.2%, not by a margin. The
 * estimate below was low on both prompt figures and high on completions; it is
 * left unchanged because it is what the slate was ordered by at the time, and
 * editing it now would rewrite the run's own provenance.
 */
export const REPRESENTATIVE_PROMPT_TOKENS = 1700;
export const REPRESENTATIVE_COMPLETION_TOKENS = 200;

/**
 * The six models, CHEAPEST FIRST.
 *
 * The ordering is the spend guard's only mitigation that costs nothing: if the
 * guard trips, the arm left partial is the most expensive one, so the cheap
 * arms are always complete. Prices and quantizations were read from
 * `/api/v1/models/{id}/endpoints` on 2026-09-07 and every pin was confirmed to
 * advertise `structured_outputs`.
 */
export const CEILING_MODELS: readonly CeilingModel[] = [
  {
    id: "deepseek/deepseek-v4-flash-0731",
    provider: "DeepInfra",
    quantization: "fp8",
    pricePerMTokIn: 0.06,
    pricePerMTokOut: 0.18,
    hosting: "US",
  },
  {
    id: "qwen/qwen3.8-flash",
    provider: "Alibaba",
    quantization: "unknown",
    pricePerMTokIn: 0.15,
    pricePerMTokOut: 0.47,
    hosting: "US-routed",
  },
  {
    id: "z-ai/glm-5.3-flash",
    provider: "BaseTen",
    quantization: "fp8",
    pricePerMTokIn: 0.15,
    pricePerMTokOut: 0.5,
    hosting: "US",
  },
  {
    id: "mistralai/mistral-small-2603",
    provider: "Mistral",
    // Reported as `unknown` by the endpoint. Recorded as unknown; not guessed.
    quantization: "unknown",
    pricePerMTokIn: 0.15,
    pricePerMTokOut: 0.6,
    // The only first-party EU endpoint on this slate. The other five pins are
    // US providers reached through a US-headquartered aggregator, which is a
    // fact an EU-hostable tier's write-up can use and which no other column
    // records.
    hosting: "EU (Mistral first-party)",
  },
  {
    id: "nvidia/nemotron-3-super-120b-a12b",
    provider: "DigitalOcean",
    quantization: "unknown",
    pricePerMTokIn: 0.3,
    pricePerMTokOut: 0.65,
    hosting: "US",
  },
  {
    id: "qwen/qwen3.8-27b",
    provider: "Parasail",
    quantization: "fp8",
    pricePerMTokIn: 0.24,
    pricePerMTokOut: 2.2,
    hosting: "US",
  },
];

/**
 * Re-pins one or more models to a different provider, from a spec like
 * `"z-ai/glm-5.3-flash=NextBit,qwen/qwen3.8-27b=Reka"`.
 *
 * This exists for a measured reason and not for convenience. The slate's pins
 * were chosen for the thinking-OFF condition; `z-ai/glm-5.3-flash` refuses that
 * condition on every provider, and the only way to get any data from it is a
 * separate thinking-ON run -- for which its slate pin (BaseTen) was, when
 * tried, rate-limited upstream while NextBit answered. Overriding the pin is
 * therefore part of the measurement, not a workaround around it.
 *
 * It is safe to expose because the record separates `requestedProvider` from
 * `provider`: an override changes what was ASKED and cannot change what is
 * recorded as having answered.
 */
export function applyPinOverrides(
  models: readonly CeilingModel[],
  spec: string | undefined,
): CeilingModel[] {
  if (spec === undefined || spec.trim() === "") return [...models];
  const overrides = new Map<string, string>();
  for (const pair of spec.split(",")) {
    const [id, provider] = pair.split("=").map((x) => x.trim());
    if (id === undefined || provider === undefined || id === "" || provider === "") {
      throw new Error(`pin override "${pair}" is not <modelId>=<provider>`);
    }
    overrides.set(id, provider);
  }
  const known = new Set(models.map((m) => m.id));
  for (const id of overrides.keys()) {
    if (!known.has(id)) throw new Error(`pin override names "${id}", which is not on the slate`);
  }
  return models.map((m) => {
    const provider = overrides.get(m.id);
    if (provider === undefined) return m;
    // The quantization belonged to the OLD endpoint. Carrying it over would
    // assert a fact about a provider that was never asked.
    return { ...m, provider, quantization: "unknown", hosting: "unknown (pin overridden)" };
  });
}

/**
 * The run id one PASS writes under, given the base run id and a 1-based pass
 * number.
 *
 * Extracted here, and tested, because the inline version was a live footgun:
 * it stripped a trailing `-\d+` and re-appended the PASS number, so launching
 * with `runId=ceiling-02` and `passes=1` produced `ceiling-01` and would have
 * silently OVERWRITTEN pass 1's ten arm files. Caught before passes 2-3 were
 * launched, by printing the mapping rather than trusting it.
 *
 * `passStart` is what makes a later pass runnable on its own: pass 2 and 3 are
 * `passStart=2, passes=2`, which yields `ceiling-02` and `ceiling-03` instead
 * of clobbering `ceiling-01`.
 */
export function passRunIdFor(runId: string, pass: number, passStart = 1): string {
  const base = runId.replace(/-\d+$/, "");
  return `${base}-${String(passStart + pass - 1).padStart(2, "0")}`;
}

/**
 * The spend-ledger filename for a run, keyed by the FIRST pass it writes.
 *
 * Exported so the WIRING is testable, not just the id arithmetic. MEASURED, and
 * the reason: a relaunch at `passStart=2` named its arm files `ceiling-02.*`
 * correctly while every `writeSpend` call still used the base `runId`, so its
 * ledger overwrote `ceiling-ceiling-01.spend.json` -- pass 1's window-2 ledger,
 * 768 calls and $0.19269 -- eight seconds in, replaced by calls=403 / $0.02033.
 * A test that pinned only `passRunIdFor` did NOT catch reverting the call site,
 * because nothing exercised the call site.
 *
 * NOTE, and it is the whole lesson: extracting THIS function was still not
 * enough. A mutant that ignores the returned value and rebuilds
 * `ceiling-${runId}.spend.json` at the point of use is invisible to every test
 * of this function, because this function keeps returning the right string and
 * nothing consumes it -- the same defect one layer further out. What actually
 * kills that mutant is `writeSpend` being an INJECTED dependency in
 * `ceiling-run.ts`, so a test reads the filename each write really received.
 * See `ceiling-main.test.ts`.
 */
export function spendLedgerFileFor(runId: string, passStart = 1): string {
  return `ceiling-${passRunIdFor(runId, 1, passStart)}.spend.json`;
}

export type CeilingFamily = "judge" | "b";

/**
 * The arm name a score table groups by. The model's OpenRouter id loses only
 * its vendor prefix -- `qwen/qwen3.8-27b` becomes `qwen3.8-27b` -- because the
 * vendor is already implied and the remainder is what distinguishes the row.
 */
export function armName(family: CeilingFamily, model: CeilingModel): string {
  const short = model.id.includes("/") ? model.id.slice(model.id.indexOf("/") + 1) : model.id;
  return `ceiling-${family}-${short}`;
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

export interface ChatMessage {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

export interface ChatCall {
  readonly model: CeilingModel;
  /** Defaults to "off", which is the experiment's condition. See REASONING_ON. */
  readonly thinking?: ThinkingRequest;
  readonly messages: readonly ChatMessage[];
  readonly schemaName: string;
  readonly schema: unknown;
  readonly maxTokens: number;
}

/**
 * The literal `reasoning` object every request carries.
 *
 * `{enabled: false}` and not `{effort: "none"}`: OpenRouter documents `effort`
 * as the OpenAI/Grok spelling, five of these six models advertise
 * `reasoning_effort` and one (`qwen/qwen3.8-flash`) does not, and `enabled` is
 * the unified field all six accept. What was ASKED is recorded on every row as
 * this exact string; what HAPPENED is `reasoningTokens`, read back from usage.
 * The two are separate columns precisely so a model that ignores the request is
 * visible rather than assumed away.
 */
export const REASONING_OFF = { enabled: false } as const;

/**
 * The `reasoning` object for the one condition that is NOT the experiment's:
 * thinking left on.
 *
 * It exists because a model on this slate leaves no alternative. MEASURED
 * 2026-09-07 against all ELEVEN of its providers, `z-ai/glm-5.3-flash` answers
 * every `{enabled: false}` request with HTTP 400 "Reasoning is mandatory for
 * this endpoint and cannot be disabled." OpenRouter's own metadata does not
 * advertise this -- the model's `reasoning_config` is null and its
 * `supported_parameters` lists `reasoning` -- so only a live call reveals it.
 *
 * An arm run this way is NOT comparable with the thinking-off arms and must
 * never be pooled with them. `thinkingRequested` on the row is `"on"`, which is
 * how a reader tells them apart, and the write-up reports it in its own table.
 */
export const REASONING_ON = { enabled: true } as const;

export type ThinkingRequest = "off" | "on";

export interface OpenRouterRequestBody {
  readonly model: string;
  readonly provider: { readonly order: readonly string[]; readonly allow_fallbacks: false };
  readonly reasoning: typeof REASONING_OFF | typeof REASONING_ON;
  readonly usage: { readonly include: true };
  readonly stream: boolean;
  readonly temperature: number;
  readonly max_tokens: number;
  readonly messages: readonly ChatMessage[];
  readonly response_format: {
    readonly type: "json_schema";
    readonly json_schema: { readonly name: string; readonly strict: true; readonly schema: unknown };
  };
}

export function reasoningRequestFor(thinking: ThinkingRequest): typeof REASONING_OFF | typeof REASONING_ON {
  return thinking === "off" ? REASONING_OFF : REASONING_ON;
}

export function buildRequestBody(call: ChatCall, stream = true): OpenRouterRequestBody {
  return {
    model: call.model.id,
    // `allow_fallbacks: false` is load-bearing, not caution. With fallbacks on,
    // a pinned provider having a bad minute silently routes to another one and
    // the latency columns become facts about routing wearing the model's name.
    // A hard failure is the honest outcome; the caller records it.
    provider: { order: [call.model.provider], allow_fallbacks: false },
    reasoning: reasoningRequestFor(call.thinking ?? "off"),
    // Makes the response carry `usage.cost`: what OpenRouter says this call
    // cost, which is what the spend guard should count rather than a price
    // table multiplied by token counts.
    usage: { include: true },
    // Streamed so TTFT is measurable at all. A non-streamed call has one
    // timestamp and cannot separate queueing from decoding -- which is exactly
    // why the non-streaming FALLBACK below records `ttftMs: null` rather than a
    // number it cannot know.
    stream,
    // The local arms decode at the WebLLM default; 0 here removes sampling
    // variance from a measurement whose whole point is comparing models, and
    // makes the three repeat passes measure provider and routing variance
    // rather than the sampler.
    temperature: 0,
    max_tokens: call.maxTokens,
    messages: call.messages,
    response_format: {
      type: "json_schema",
      json_schema: { name: call.schemaName, strict: true, schema: call.schema },
    },
  };
}

/** One HTTP attempt that failed and was retried. Every one is recorded. */
export interface RetryEvent {
  readonly attempt: number;
  readonly status: number | "network";
  readonly delayMs: number;
  readonly detail: string;
}

export interface CallOutcome {
  /**
   * Which path produced this row.
   *
   * `"stream"` is the normal one and the only one that can report TTFT or a
   * decode rate. `"non-stream-fallback"` exists because a provider on this
   * slate needed it: MEASURED 2026-09-07, `qwen/qwen3.8-flash` on Alibaba
   * streamed normally for one family (TTFT 3.9-5.7s over three items) and then,
   * within the same hour, began returning ZERO BYTES and no headers on every
   * streamed request while the IDENTICAL body non-streamed answered in 2.5s.
   * Isolated with two further probes: it hangs with the json_schema removed and
   * it hangs with a 60-character prompt, so it is neither the schema nor the
   * prompt size -- it is that endpoint's streaming path.
   *
   * Falling back rather than losing the arm is the deliberate trade: accuracy,
   * reasoning tokens and cost are all still measurable without a stream, and
   * they are the questions this experiment is for. What is NOT measurable is
   * TTFT and decode rate, so both are recorded as `null` on a fallback row and
   * never imputed. A latency table must exclude these rows by this field.
   */
  readonly transport: "stream" | "non-stream-fallback";
  readonly content: string;
  readonly finishReason: string | undefined;
  /** The provider that ANSWERED, off the stream. Never the pin. */
  readonly provider: string | undefined;
  readonly modelId: string | undefined;
  readonly promptTokens: number | undefined;
  readonly completionTokens: number | undefined;
  /**
   * `usage.completion_tokens_details.reasoning_tokens`. A MODEL property and
   * the transferable number on this whole run: unlike TTFT and decode rate it
   * does not depend on whose GPU answered. `undefined` means the provider
   * reported no usage at all -- never 0, which would be the measured claim that
   * the model emitted no reasoning.
   */
  readonly reasoningTokens: number | undefined;
  readonly costUsd: number | undefined;
  /**
   * Milliseconds to the first delta carrying non-empty `content`. SSE comment
   * frames (`: OPENROUTER PROCESSING`) and role-only openers are deliberately
   * not counted: they arrive before the model has produced anything, and timing
   * to them measures the aggregator's connection handling. PROVIDER-DEPENDENT.
   */
  readonly ttftMs: number | undefined;
  /** Request start to final chunk. What a user waits through. */
  readonly wallMs: number;
  /**
   * `(completionTokens - 1) / ((wallMs - ttftMs) / 1000)`. PROVIDER-DEPENDENT.
   *
   * `n - 1` and not `n`: TTFT ENDS when the first token arrives, so only the
   * remaining tokens decode inside this window. MEASURED here on 2026-09-08 by
   * recomputing both formulas over pass 1's ten arm files: dividing by `n`
   * overstates the median rate by 0.8-4.7% on the Approach-B arms (median
   * 57-118 completion tokens) and by 14.2-21.8% on the judge arms (median 5-7),
   * which is larger than the spread this column is used to compare judge arms
   * with.
   *
   * `undefined` when the decode window is zero or negative, when either input
   * is missing, or when `n <= 1` -- an Infinity here would enter the report as
   * a model that decoded infinitely fast, and one token measures nothing but
   * TTFT. MEASURED: of the 4,844 calls across passes 1-3 that reported usage,
   * the smallest completion was 5 tokens, so the `n <= 1` arm guards a case
   * this run never hit rather than changing any published figure.
   */
  readonly decodeTokPerSec: number | undefined;
}

export interface ClientDeps {
  readonly fetch: typeof fetch;
  readonly now: () => number;
  readonly sleep: (ms: number) => Promise<void>;
  readonly apiKey: string;
}

export interface CallOptions {
  readonly maxAttempts?: number;
  readonly baseBackoffMs?: number;
  readonly timeoutMs?: number;
  /**
   * Whether a stream that times out may be retried WITHOUT streaming. Default
   * true; set false to measure the streaming path alone.
   */
  readonly allowNonStreamFallback?: boolean;
}

const DEFAULT_MAX_ATTEMPTS = 5;
const DEFAULT_BASE_BACKOFF_MS = 1000;
/**
 * The per-call wall ceiling, and it exists because the FIRST live probe needed
 * it: `qwen/qwen3.8-flash` on Alibaba left a single streamed call open past ten
 * minutes with no error and no further frames, which stalled the whole run
 * behind one item. Without a bound, one slow endpoint costs the arms queued
 * after it, and the spend guard cannot help -- an unfinished call has no cost
 * to record.
 *
 * 60s is deliberately far above every healthy call measured in that probe
 * (885-9,323 ms wall), so it bounds a hang rather than truncating slow but
 * working generation. A stream timeout switches the remaining attempts to the
 * NON-STREAMING path (see `CallOutcome.transport`), so a dead streaming
 * endpoint costs one timeout rather than five.
 */
const DEFAULT_TIMEOUT_MS = 60_000;

/** 429 and 5xx are transient; a 4xx that is not 429 is a request this run will never get right. */
function isRetryable(status: number): boolean {
  return status === 429 || status >= 500;
}

/**
 * One chat completion, streamed, timed, and retried on transient failures.
 *
 * The key reaches this function through `deps.apiKey` and is written into one
 * header. It is never logged, never returned, and never placed on a record.
 */
export async function callChat(
  deps: ClientDeps,
  call: ChatCall,
  options: CallOptions = {},
): Promise<{ outcome: CallOutcome; retries: RetryEvent[] }> {
  const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
  const base = options.baseBackoffMs ?? DEFAULT_BASE_BACKOFF_MS;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const allowFallback = options.allowNonStreamFallback ?? true;
  const retries: RetryEvent[] = [];
  // Flips to false for the remaining attempts once a stream has timed out, so
  // an endpoint whose streaming path is dead costs ONE timeout and not five.
  let stream = true;

  for (let attempt = 1; ; attempt++) {
    const body = JSON.stringify(buildRequestBody(call, stream));
    const started = deps.now();
    let response: Response;
    // Armed around the WHOLE call, request and stream both: a provider that
    // accepts the connection and then stops sending frames is the hang this
    // bounds, and a timeout on the request alone would not see it. Cleared in
    // `finally` so a completed call does not leave a ref'd timer behind.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      try {
        response = await deps.fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${deps.apiKey}`,
            "Content-Type": "application/json",
          },
          body,
          signal: controller.signal,
        });
      } catch (cause) {
        if (attempt >= maxAttempts) throw new Error(`network failure after ${attempt} attempts: ${String(cause)}`);
        const delayMs = base * 2 ** (attempt - 1);
        const timedOut = controller.signal.aborted;
        retries.push({
          attempt,
          status: "network",
          delayMs,
          detail: timedOut && stream && allowFallback
            ? `${String(cause)} (stream timed out after ${timeoutMs}ms; retrying WITHOUT streaming, so this call's ttftMs and decodeTokPerSec will be null)`
            : String(cause),
        });
        if (timedOut && stream && allowFallback) stream = false;
        await deps.sleep(delayMs);
        continue;
      }

      if (!response.ok) {
        // Read the body for the message and then discard it; an error body is
        // small and naming the reason is what makes a failed arm diagnosable.
        const detail = (await response.text().catch(() => "")).slice(0, 500);
        if (!isRetryable(response.status) || attempt >= maxAttempts) {
          throw new Error(`OpenRouter returned ${response.status} for ${call.model.id}: ${detail}`);
        }
        const delayMs = base * 2 ** (attempt - 1);
        retries.push({ attempt, status: response.status, delayMs, detail });
        await deps.sleep(delayMs);
        continue;
      }

      const outcome = stream
        ? await consumeStream(response, deps.now, started)
        : await consumeWhole(response, deps.now, started);
      return { outcome, retries };
    } finally {
      clearTimeout(timer);
    }
  }
}

/**
 * Reads one SSE body, assembling content and taking the two timestamps.
 *
 * Written against a REAL capture rather than against the SSE spec in the
 * abstract: OpenRouter opens with `: OPENROUTER PROCESSING` comment frames,
 * sends `usage` only on the final data frame, and terminates with
 * `data: [DONE]`. All three are handled explicitly because each one, mishandled,
 * corrupts a different column -- the comments would halve TTFT, a missing
 * final-frame read would lose cost and reasoning tokens entirely, and treating
 * `[DONE]` as JSON would throw at the end of every successful call.
 */
async function consumeStream(
  response: Response,
  now: () => number,
  started: number,
): Promise<CallOutcome> {
  const reader = response.body?.getReader();
  let content = "";
  let finishReason: string | undefined;
  let provider: string | undefined;
  let modelId: string | undefined;
  let usage: Record<string, unknown> | undefined;
  let ttftMs: number | undefined;
  let buffer = "";

  const handleFrame = (frame: string): void => {
    // SSE comment: not data, and in particular not a token.
    if (frame.startsWith(":")) return;
    if (!frame.startsWith("data:")) return;
    const payload = frame.slice(5).trim();
    if (payload === "" || payload === "[DONE]") return;
    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      // A frame we cannot read is skipped rather than fatal: the columns it
      // would have contributed to are all cumulative, and one unreadable frame
      // must not lose an answered call.
      return;
    }
    if (typeof chunk["provider"] === "string") provider = chunk["provider"];
    if (typeof chunk["model"] === "string") modelId = chunk["model"];
    if (chunk["usage"] && typeof chunk["usage"] === "object") {
      usage = chunk["usage"] as Record<string, unknown>;
    }
    const choices = chunk["choices"];
    if (!Array.isArray(choices) || choices.length === 0) return;
    const choice = choices[0] as Record<string, unknown>;
    if (typeof choice["finish_reason"] === "string") finishReason = choice["finish_reason"];
    const delta = choice["delta"] as Record<string, unknown> | undefined;
    const piece = delta?.["content"];
    if (typeof piece === "string" && piece.length > 0) {
      // The FIRST non-empty content delta, and nothing earlier, is TTFT.
      if (ttftMs === undefined) ttftMs = now() - started;
      content += piece;
    }
  };

  if (reader !== undefined) {
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Frames are separated by a blank line. Split on the separator and keep
      // the trailing partial in the buffer -- a chunk boundary lands mid-frame
      // routinely and parsing a half frame would drop tokens.
      let sep = buffer.indexOf("\n\n");
      while (sep !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of frame.split("\n")) handleFrame(line.trim());
        sep = buffer.indexOf("\n\n");
      }
    }
    for (const line of buffer.split("\n")) handleFrame(line.trim());
  }

  const wallMs = now() - started;
  const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
  const details = usage?.["completion_tokens_details"] as Record<string, unknown> | undefined;
  const completionTokens = num(usage?.["completion_tokens"]);
  const decodeWindowMs = ttftMs === undefined ? undefined : wallMs - ttftMs;
  const decodeTokPerSec =
    completionTokens !== undefined &&
    completionTokens > 1 &&
    decodeWindowMs !== undefined &&
    decodeWindowMs > 0
      ? (completionTokens - 1) / (decodeWindowMs / 1000)
      : undefined;

  return {
    transport: "stream",
    content,
    finishReason,
    provider,
    modelId,
    promptTokens: num(usage?.["prompt_tokens"]),
    completionTokens,
    reasoningTokens: num(details?.["reasoning_tokens"]),
    costUsd: num(usage?.["cost"]),
    ttftMs,
    wallMs,
    decodeTokPerSec,
  };
}

/**
 * Reads one NON-streamed response: the fallback path.
 *
 * Everything a stream reports is here EXCEPT the two columns a single
 * timestamp cannot produce. `ttftMs` and `decodeTokPerSec` are `undefined`, and
 * they stay that way to the record -- there is exactly one timestamp, so any
 * TTFT written here would be the whole call's duration wearing a different
 * name, which is the intent-as-fact defect with a latency label on it.
 */
async function consumeWhole(
  response: Response,
  now: () => number,
  started: number,
): Promise<CallOutcome> {
  const body = (await response.json()) as Record<string, unknown>;
  const wallMs = now() - started;
  const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
  const usage = body["usage"] as Record<string, unknown> | undefined;
  const details = usage?.["completion_tokens_details"] as Record<string, unknown> | undefined;
  const choice = (body["choices"] as Record<string, unknown>[] | undefined)?.[0];
  const message = choice?.["message"] as Record<string, unknown> | undefined;
  return {
    transport: "non-stream-fallback",
    content: typeof message?.["content"] === "string" ? (message["content"] as string) : "",
    finishReason: typeof choice?.["finish_reason"] === "string" ? (choice["finish_reason"] as string) : undefined,
    provider: typeof body["provider"] === "string" ? (body["provider"] as string) : undefined,
    modelId: typeof body["model"] === "string" ? (body["model"] as string) : undefined,
    promptTokens: num(usage?.["prompt_tokens"]),
    completionTokens: num(usage?.["completion_tokens"]),
    reasoningTokens: num(details?.["reasoning_tokens"]),
    costUsd: num(usage?.["cost"]),
    ttftMs: undefined,
    wallMs,
    decodeTokPerSec: undefined,
  };
}

// ---------------------------------------------------------------------------
// The spend guard
// ---------------------------------------------------------------------------

export interface SpendEntry {
  readonly costUsd: number | undefined;
  readonly estimateUsd: number;
  readonly model?: string;
  readonly family?: string;
}

export interface SpendSnapshot {
  readonly gitSha: string;
  readonly gitDirty: boolean;
  readonly calls: number;
  readonly costUsd: number;
  readonly estimatedCostUsd: number;
  readonly keyUsageAtStart: number;
  readonly keyLimit: number | null;
  readonly keyUsageLatest: number | null;
  readonly keySpendUsd: number;
  readonly hardStopUsd: number;
  readonly tripped: boolean;
  readonly byModel: Record<string, number>;
  readonly byFamily: Record<string, number>;
  readonly updatedAt: string;
}

/**
 * The run's ledger, and the only thing standing between this arm and the key's
 * $10 cap.
 *
 * Two independent numbers can stop the run and BOTH are checked, because they
 * can disagree and the pessimistic one is the one that matters:
 *
 *   1. the sum of every response's own `usage.cost`, added to whatever the key
 *      had already spent before this run started; and
 *   2. the key endpoint's own `usage`, re-read every 50 calls.
 *
 * A guard reading only (1) would miss billing the responses under-report; a
 * guard reading only (2) would overshoot by up to 50 calls. Either tripping
 * stops the run.
 */
export class SpendGuard {
  readonly #hardStopUsd: number;
  readonly #keyUsageAtStart: number;
  readonly #keyLimit: number | null;
  readonly #checkpointEvery: number;
  readonly #gitSha: string;
  readonly #gitDirty: boolean;
  #calls = 0;
  #costUsd = 0;
  #estimatedUsd = 0;
  #keyUsageLatest: number | null = null;
  #callsAtLastCheckpoint = 0;
  readonly #byModel = new Map<string, number>();
  readonly #byFamily = new Map<string, number>();

  constructor(options: {
    hardStopUsd: number;
    keyUsageAtStart: number;
    keyLimit: number | null;
    checkpointEvery?: number;
    gitSha?: string;
    gitDirty?: boolean;
  }) {
    this.#gitSha = options.gitSha ?? "unknown";
    this.#gitDirty = options.gitDirty ?? false;
    this.#hardStopUsd = options.hardStopUsd;
    this.#keyUsageAtStart = options.keyUsageAtStart;
    this.#keyLimit = options.keyLimit;
    this.#checkpointEvery = options.checkpointEvery ?? 50;
  }

  record(entry: SpendEntry): void {
    this.#calls += 1;
    // The response's own number in preference to the price table: the table is
    // what the endpoint advertised, the response is what OpenRouter charged,
    // and on a slate where one pin (`Mistral`) resolves to three endpoints at
    // two different prices only the latter is knowable in advance.
    const cost = entry.costUsd ?? entry.estimateUsd;
    this.#costUsd += cost;
    this.#estimatedUsd += entry.estimateUsd;
    if (entry.model !== undefined) this.#byModel.set(entry.model, (this.#byModel.get(entry.model) ?? 0) + cost);
    if (entry.family !== undefined) this.#byFamily.set(entry.family, (this.#byFamily.get(entry.family) ?? 0) + cost);
  }

  get calls(): number {
    return this.#calls;
  }

  get totalUsd(): number {
    return this.#costUsd;
  }

  get estimatedUsd(): number {
    return this.#estimatedUsd;
  }

  /** What the KEY has spent: what it had before this run, plus what this run added. */
  get keySpendUsd(): number {
    return this.#keyUsageAtStart + this.#costUsd;
  }

  get keyUsageLatest(): number | null {
    return this.#keyUsageLatest;
  }

  tripped(): boolean {
    if (this.keySpendUsd >= this.#hardStopUsd) return true;
    return this.#keyUsageLatest !== null && this.#keyUsageLatest >= this.#hardStopUsd;
  }

  dueForCheckpoint(): boolean {
    return this.#calls - this.#callsAtLastCheckpoint >= this.#checkpointEvery;
  }

  noteCheckpoint(keyUsage: number): void {
    this.#keyUsageLatest = keyUsage;
    this.#callsAtLastCheckpoint = this.#calls;
  }

  snapshot(): SpendSnapshot {
    return {
      gitSha: this.#gitSha,
      gitDirty: this.#gitDirty,
      calls: this.#calls,
      costUsd: this.#costUsd,
      estimatedCostUsd: this.#estimatedUsd,
      keyUsageAtStart: this.#keyUsageAtStart,
      keyLimit: this.#keyLimit,
      keyUsageLatest: this.#keyUsageLatest,
      keySpendUsd: this.keySpendUsd,
      hardStopUsd: this.#hardStopUsd,
      tripped: this.tripped(),
      byModel: Object.fromEntries(this.#byModel),
      byFamily: Object.fromEntries(this.#byFamily),
      updatedAt: new Date().toISOString(),
    };
  }
}

/**
 * The price-table estimate, kept beside the response's own cost as a
 * cross-check rather than as the primary number.
 *
 * Missing token counts contribute 0 rather than NaN: one provider omitting
 * usage must not poison the whole ledger into unusability, and the two totals
 * disagreeing is itself information the snapshot carries.
 */
export function estimateCostUsd(
  model: CeilingModel,
  promptTokens: number | undefined,
  completionTokens: number | undefined,
): number {
  const p = promptTokens ?? 0;
  const c = completionTokens ?? 0;
  return (p / 1e6) * model.pricePerMTokIn + (c / 1e6) * model.pricePerMTokOut;
}

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

const CEILING_SCHEMA_VERSION = 1;

/**
 * One model call. Repair retries get their own row, exactly as `JudgeCallRecord`
 * does for the local arms, because a repair spent tokens and time.
 *
 * The four timing fields are all present on every row and are NULLABLE rather
 * than optional. The distinction is deliberate: `null` is the positive claim
 * "the provider did not report this", and a missing key would be
 * indistinguishable from a writer that forgot. `reasoningTokens: null` in
 * particular must never be read as zero -- zero is the answer this run is
 * looking for, so it has to be a measurement and not a default.
 */
export const CeilingCallSchema = z.object({
  /** Prompt tokens the provider billed, or null if it reported no usage. */
  promptTokens: z.number().int().nonnegative().nullable(),
  completionTokens: z.number().int().nonnegative().nullable(),
  /** MODEL property; the transferable number. See CallOutcome.reasoningTokens. */
  reasoningTokens: z.number().int().nonnegative().nullable(),
  /** PROVIDER-DEPENDENT. Milliseconds to the first non-empty content delta. */
  ttftMs: z.number().nonnegative().nullable(),
  /**
   * THIS ATTEMPT's request start to its final chunk, in milliseconds.
   *
   * Per-ATTEMPT, and the distinction decides whether every latency number in
   * the report means anything. `callChat` re-initialises its clock inside the
   * retry loop, so this covers ONLY the attempt that succeeded: it excludes
   * every failed attempt and every backoff sleep. That matters here and not in
   * the abstract -- DeepInfra 429'd 45-50% of the DeepSeek calls, some four
   * times over, and a wall time measured from the first attempt would be mostly
   * the aggregator's rate limiter wearing the model's name.
   *
   * The END-TO-END number, which DOES include every retry, every backoff sleep
   * and the repair turn, is the RECORD's own top-level `wallMs`. Both are
   * carried deliberately: this one is the closest thing available to a model
   * latency, that one is what a caller actually waited through. Never sum these
   * across a row and call it either.
   *
   * `retries` on this row names what was paid for the difference.
   */
  wallMs: z.number().nonnegative(),
  /** PROVIDER-DEPENDENT. (completionTokens - 1) / (wallMs - ttftMs), tok/s. See CallOutcome. */
  decodeTokPerSec: z.number().positive().nullable(),
  /** What OpenRouter reported this call cost, in USD. */
  costUsd: z.number().nonnegative().nullable(),
  finishReason: z.string().nullable(),
  /** The provider that ANSWERED this call. */
  provider: z.string().nullable(),
  modelId: z.string().nullable(),
  /**
   * Which path produced this row. A `non-stream-fallback` row has null `ttftMs`
   * and null `decodeTokPerSec` BY CONSTRUCTION; exclude it from any latency
   * aggregate rather than treating the nulls as missing data.
   */
  transport: z.enum(["stream", "non-stream-fallback"]),
  /** True when this row is the one repair retry rather than the original call. */
  repair: z.boolean(),
  /** How the body was read; `schema.ts`'s own words plus "ok". */
  parse: z.enum(["ok", "aborted", "truncated", "malformed", "schema"]),
  retries: z.array(
    z.object({
      attempt: z.number().int().positive(),
      status: z.union([z.number().int(), z.literal("network")]),
      delayMs: z.number().nonnegative(),
      detail: z.string(),
    }),
  ),
});

const GoldSpanRowSchema = z.object({
  start: z.number().int().nonnegative(),
  end: z.number().int().positive(),
  text: z.string().min(1),
  entityType: z.string().min(1),
  action: z.string().min(1),
});

export const CeilingRecordSchema = z
  .object({
    schemaVersion: z.literal(CEILING_SCHEMA_VERSION),
    runId: z.string().min(1),
    itemId: z.string().min(1),
    policy: z.string().min(1),
    /** sha256 of the IR artifact that ran, hashed from its own bytes. */
    irHash: z.string().regex(/^[0-9a-f]{64}$/),
    /** `PolicyIr.policyHash` verbatim -- what `score.ts` joins gold on. */
    policyHash: z.string().min(1),
    arm: z.string().min(1),
    family: z.enum(["judge", "b"]),
    /** The model id ASKED for. */
    requestedModelId: z.string().min(1),
    /** The model id the response NAMED, or null when no call answered. */
    modelId: z.string().nullable(),
    /** The provider `provider.order` pinned. A request. */
    requestedProvider: z.string().min(1),
    /** The provider the response CARRIED. The fact. Null when no call answered. */
    provider: z.string().nullable(),
    /** The pinned endpoint's own quantization string; "unknown" is a value. */
    quantization: z.string().min(1),
    outputMechanism: z.literal(OUTPUT_MECHANISM),
    /**
     * What was ASKED of the model's reasoning. Compare with
     * `calls[].reasoningTokens`, which is what HAPPENED. Two separate columns
     * precisely so a model that ignores the request is visible.
     *
     * `"on"` rows are a DIFFERENT CONDITION and must never be pooled with
     * `"off"` rows. Only `z-ai/glm-5.3-flash` produces them, because it refuses
     * to run any other way; see REASONING_ON.
     */
    thinkingRequested: z.enum(["off", "on"]),
    /** The literal `reasoning` object sent, serialized, so the ask is on the row. */
    reasoningRequest: z.string().min(1),
    /**
     * The `max_tokens` this call was given, and the local arms' value beside it.
     *
     * Present because a comment in `ceiling-main.ts` asserted the two were equal
     * and they are not: `DEFAULT_TIER2_CONFIG.maxTokens` is 512
     * (packages/tier2/src/manifest.ts:116) and this arm ran at 600. A budget
     * asymmetry that is not on the row is one a later reader cannot find, so
     * both numbers go here rather than into prose.
     *
     * OPTIONAL, and that is deliberate rather than lax: rows written by the
     * first paid run predate this field and cannot be back-filled without
     * inventing provenance. Every one of them ran at maxTokens 600 against a
     * local 512 -- stated here, in the doc, and checkable from the per-arm
     * completion-token distribution, which is the honest substitute for a value
     * the file does not carry.
     */
    maxTokens: z.number().int().positive().optional(),
    localArmMaxTokens: z.number().int().positive().optional(),
    /**
     * `git rev-parse HEAD` at run start, and whether the tree was dirty.
     *
     * Added because a survey of `runs/` found that NO existing artifact names
     * the code revision that produced it, so no number already in that
     * directory can be traced to a commit. This arm is the first that can be.
     * `gitDirty` is recorded honestly: these runs were made from a working tree
     * carrying this file before it was committed, so it is `true`, and
     * committing early to make it read `false` would be the intent-as-fact
     * defect in its purest form.
     */
    gitSha: z.string().regex(/^[0-9a-f]{40}$/),
    gitDirty: z.boolean(),
    text: z.string().min(1),
    findings: z.array(RecordFindingSchema),
    gold: z.array(GoldSpanRowSchema),
    calls: z.array(CeilingCallSchema),
    parseFailures: z.number().int().nonnegative(),
    repairs: z.number().int().nonnegative(),
    unresolvedQuotes: z.number().int().nonnegative(),
    unresolvedMentions: z.number().int().nonnegative(),
    unknownLabels: z.number().int().nonnegative(),
    duplicatesDropped: z.number().int().nonnegative(),
    wholeClauseMentions: z.number().int().nonnegative(),
    /**
     * END TO END for this item: every attempt, every backoff sleep, the repair
     * turn, and the successful call. What a caller waited through.
     *
     * Deliberately NOT comparable with `calls[].wallMs`, which times the
     * successful attempt alone. On a rate-limited provider the two differ by
     * seconds, and the gap is the aggregator's, not the model's.
     */
    wallMs: z.number().nonnegative(),
    /** Set when the item threw. The row is still written; see RunRecordSchema. */
    error: z.string().nullable(),
  })
  .refine(
    (r) =>
      [...r.findings, ...r.gold].every(
        (s) => s.start < s.end && s.end <= r.text.length && r.text.slice(s.start, s.end) === s.text,
      ),
    { message: "a span's offsets do not hold the text it names" },
  );

export type CeilingRecord = z.infer<typeof CeilingRecordSchema>;
export type CeilingCall = z.infer<typeof CeilingCallSchema>;

export function toJsonl(records: readonly CeilingRecord[]): string {
  return (
    records
      .map((r) => JSON.stringify(r).replace(/\u{2028}/gu, "\\u2028").replace(/\u{2029}/gu, "\\u2029"))
      .join("\n") + "\n"
  );
}

// ---------------------------------------------------------------------------
// The two families
// ---------------------------------------------------------------------------

/**
 * The whole message as one passage.
 *
 * p-fin's only predicate declares `scope: "message"`, and predicates PARTITION
 * by declared scope, so the compiled arm makes exactly one call per message
 * here. This arm does the same: one call per item, over the whole text.
 */
export function messagePassage(text: string): Segment {
  return { start: 0, end: text.length, kind: "prose", text };
}

function messagePredicates(ir: PolicyIr): readonly SemanticPredicate[] {
  return ir.semanticPredicates.filter((p) => p.scope === "message");
}

/**
 * The compiled judge's own two turns, from `@sih/tier2`.
 *
 * `priorFindings` is `[]` because this arm is the ceiling equivalent of
 * `compiled-tier2-only`: no tier 0, no tier 1, so the prior line reads "Earlier
 * tiers found nothing in this passage." -- which is true, and is the same
 * sentence the local tier2-only arm sends.
 */
export function judgeMessagesFor(ir: PolicyIr, text: string): ChatMessage[] {
  return buildJudgeMessages(messagePassage(text), messagePredicates(ir), []) as ChatMessage[];
}

/** Approach B's own two turns, from `@sih/tier2`: the whole policy and every entityType id. */
export function baselineMessagesFor(ir: PolicyIr, policyText: string, text: string): ChatMessage[] {
  return buildBaselineMessages(
    policyText,
    ir.entityTypes.map((e) => e.id),
    text,
    // undefined, not a "found nothing" line: this arm runs no tier 0, and B's
    // own no-tier-0 configuration omits the line entirely rather than asserting
    // an empty one. Matching `createBaselineB` exactly is the point.
    undefined,
  ) as ChatMessage[];
}

export function judgeRepairTurn(reason: string, detail: string): ChatMessage {
  return judgeRepairMessage(reason, detail) as ChatMessage;
}

export function baselineRepairTurn(reason: string, detail: string): ChatMessage {
  return baselineRepairMessage(reason, detail) as ChatMessage;
}

export interface CollectCounters {
  unresolvedQuotes: number;
  unresolvedMentions: number;
  unknownLabels: number;
  duplicatesDropped: number;
  wholeClauseMentions: number;
}

function zeroCounters(): CollectCounters {
  return {
    unresolvedQuotes: 0,
    unresolvedMentions: 0,
    unknownLabels: 0,
    duplicatesDropped: 0,
    wholeClauseMentions: 0,
  };
}

export interface CollectResult {
  readonly findings: Finding[];
  readonly counters: CollectCounters;
}

/**
 * `judge.ts`'s `#collect`, over an HTTP response instead of a WebLLM one.
 *
 * The span ladder, the shadow-id mapping, the duplicate key and every counter
 * are the judge's, so a finding placed here and a finding placed by the browser
 * arm are placed by the same rules. What is NOT reimplemented is any of the
 * engine-specific handling around it -- there is no engine to latch.
 */
export function collectJudge(
  ir: PolicyIr,
  text: string,
  response: JudgeResponse,
  model: string,
): CollectResult {
  const counters = zeroCounters();
  const findings: Finding[] = [];
  const emitted = new Set<string>();
  const severityOf = new Map<string, Finding["severity"]>();
  for (const predicate of ir.semanticPredicates) {
    const shadow = ir.entityTypes.find((e) => e.id === shadowIdFor(predicate.id));
    if (shadow !== undefined) severityOf.set(predicate.id, shadow.severity);
  }

  for (const finding of response.findings) {
    const severity = severityOf.get(finding.predicateId);
    if (severity === undefined) {
      counters.unknownLabels += 1;
      continue;
    }
    const located = locateFinding(text, finding.quote, finding.mention);
    if (!located.ok) {
      if (located.refused === "evidence") counters.unresolvedQuotes += 1;
      else counters.unresolvedMentions += 1;
      continue;
    }
    const action = located.at.action;
    const entityType = shadowIdFor(finding.predicateId);
    const key = `${action.start}:${action.end}:${entityType}`;
    if (emitted.has(key)) {
      counters.duplicatesDropped += 1;
      continue;
    }
    emitted.add(key);
    if (located.at.actionIsWholeEvidence) counters.wholeClauseMentions += 1;
    findings.push({
      start: action.start,
      end: action.end,
      text: action.text,
      entityType,
      severity,
      tier: 2,
      source: model,
      confidence: finding.confidence,
    });
  }
  return { findings, counters };
}

/** `baselineB.ts`'s `collect`, over an HTTP response. Same ladder, same counters. */
export function collectBaseline(
  ir: PolicyIr,
  text: string,
  response: BaselineResponse,
  model: string,
): CollectResult {
  const counters = zeroCounters();
  const findings: Finding[] = [];
  const emitted = new Set<string>();

  for (const finding of response.findings) {
    const entity = ir.entityTypes.find((e) => e.id === finding.entityType);
    if (entity === undefined) {
      counters.unknownLabels += 1;
      continue;
    }
    const located = locateFinding(text, finding.quote, finding.mention);
    if (!located.ok) {
      if (located.refused === "evidence") counters.unresolvedQuotes += 1;
      else counters.unresolvedMentions += 1;
      continue;
    }
    const action = located.at.action;
    const key = `${action.start}:${action.end}:${entity.id}`;
    if (emitted.has(key)) {
      counters.duplicatesDropped += 1;
      continue;
    }
    emitted.add(key);
    if (located.at.actionIsWholeEvidence) counters.wholeClauseMentions += 1;
    findings.push({
      start: action.start,
      end: action.end,
      text: action.text,
      entityType: entity.id,
      severity: entity.severity,
      tier: 2,
      source: model,
      confidence: finding.confidence,
    });
  }
  return { findings, counters };
}

// ---------------------------------------------------------------------------
// Running one item
// ---------------------------------------------------------------------------

export interface ItemInput {
  readonly id: string;
  readonly text: string;
  readonly policy: string;
  readonly gold: readonly { start: number; end: number; text: string; entityType: string; action: string }[];
}

export interface RunItemOptions {
  readonly deps: ClientDeps;
  readonly model: CeilingModel;
  readonly family: CeilingFamily;
  readonly ir: PolicyIr;
  readonly irHash: string;
  readonly policyText: string;
  readonly item: ItemInput;
  readonly runId: string;
  readonly maxTokens: number;
  /** `DEFAULT_TIER2_CONFIG.maxTokens`, recorded beside `maxTokens`. See the record field. */
  readonly localArmMaxTokens?: number;
  /**
   * The DESTINATION provider the policy's actions are resolved for -- core's
   * `resolveFindings(ir, provider, ...)` argument, `"claude"` in every bake-off
   * run so far.
   *
   * NOT the serving provider. The record's `provider` field is the OpenRouter
   * endpoint that answered; this is which downstream LLM the message was
   * notionally headed to, which is what decides a finding's action. Two
   * different meanings for one English word, kept in two differently named
   * fields so a reader cannot conflate them.
   */
  readonly destinationProvider: string;
  /** Defaults to "off". See REASONING_ON for the one model that forces "on". */
  readonly thinking?: ThinkingRequest;
  /** See CeilingRecordSchema.gitSha. */
  readonly gitSha: string;
  readonly gitDirty: boolean;
  readonly onCall?: (entry: SpendEntry) => void;
}

/**
 * One corpus item on one arm: build the prompt, call, parse, repair once,
 * collect, resolve.
 *
 * The parse-and-one-repair rule is the local arms' rule exactly, including the
 * part that is easy to soften: after one repair the answer is DROPPED, never
 * retried into compliance. An arm that retries until the JSON parses is
 * measuring persistence, not capability.
 */
export async function runCeilingItem(options: RunItemOptions): Promise<CeilingRecord> {
  const { deps, model, family, ir, item } = options;
  const thinking: ThinkingRequest = options.thinking ?? "off";
  const schema = family === "judge" ? JUDGE_SCHEMA : BASELINE_B_SCHEMA;
  const schemaName = family === "judge" ? "judge_response" : "baseline_response";
  let messages: ChatMessage[] =
    family === "judge"
      ? judgeMessagesFor(ir, item.text)
      : baselineMessagesFor(ir, options.policyText, item.text);

  const calls: CeilingCall[] = [];
  const counters = zeroCounters();
  let raw: Finding[] = [];
  let repairs = 0;
  let parseFailures = 0;
  let lastProvider: string | null = null;
  let lastModelId: string | null = null;
  let error: string | null = null;
  const itemStarted = deps.now();

  try {
    let repaired = false;
    for (;;) {
      const { outcome, retries } = await callChat(deps, {
        model,
        thinking,
        messages,
        schemaName,
        schema,
        maxTokens: options.maxTokens,
      });
      options.onCall?.({
        costUsd: outcome.costUsd,
        estimateUsd: estimateCostUsd(model, outcome.promptTokens, outcome.completionTokens),
        model: model.id,
        family,
      });
      lastProvider = outcome.provider ?? null;
      lastModelId = outcome.modelId ?? null;

      const parsed =
        family === "judge"
          ? parseJudgeResponse(outcome.content, outcome.finishReason)
          : parseBaselineResponse(outcome.content, outcome.finishReason);

      calls.push({
        promptTokens: outcome.promptTokens ?? null,
        completionTokens: outcome.completionTokens ?? null,
        reasoningTokens: outcome.reasoningTokens ?? null,
        ttftMs: outcome.ttftMs ?? null,
        wallMs: outcome.wallMs,
        decodeTokPerSec: outcome.decodeTokPerSec ?? null,
        costUsd: outcome.costUsd ?? null,
        finishReason: outcome.finishReason ?? null,
        provider: outcome.provider ?? null,
        modelId: outcome.modelId ?? null,
        transport: outcome.transport,
        repair: repaired,
        parse: parsed.ok ? "ok" : parsed.reason,
        retries: retries.map((r) => ({ ...r })),
      });

      if (parsed.ok) {
        const collected =
          family === "judge"
            ? collectJudge(ir, item.text, parsed.value as JudgeResponse, outcome.modelId ?? model.id)
            : collectBaseline(ir, item.text, parsed.value as BaselineResponse, outcome.modelId ?? model.id);
        raw = collected.findings;
        counters.unresolvedQuotes += collected.counters.unresolvedQuotes;
        counters.unresolvedMentions += collected.counters.unresolvedMentions;
        counters.unknownLabels += collected.counters.unknownLabels;
        counters.duplicatesDropped += collected.counters.duplicatesDropped;
        counters.wholeClauseMentions += collected.counters.wholeClauseMentions;
        break;
      }

      parseFailures += 1;
      if (repaired) break; // one repair, then fail closed
      repaired = true;
      repairs += 1;
      const turn =
        family === "judge"
          ? judgeRepairTurn(parsed.reason, parsed.detail)
          : baselineRepairTurn(parsed.reason, parsed.detail);
      messages = [...messages, turn];
    }
  } catch (cause) {
    error = String(cause);
  }

  // Core's own cluster resolution, so a ceiling finding carries the same action
  // a browser finding over the same span would.
  const resolved =
    error === null ? resolveFindings(ir, options.destinationProvider, item.text, raw) : [];

  return CeilingRecordSchema.parse({
    schemaVersion: CEILING_SCHEMA_VERSION,
    runId: options.runId,
    itemId: item.id,
    policy: item.policy,
    irHash: options.irHash,
    policyHash: ir.policyHash,
    arm: armName(family, model),
    family,
    requestedModelId: model.id,
    modelId: lastModelId,
    requestedProvider: model.provider,
    provider: lastProvider,
    quantization: model.quantization,
    outputMechanism: OUTPUT_MECHANISM,
    thinkingRequested: thinking,
    reasoningRequest: JSON.stringify(reasoningRequestFor(thinking)),
    maxTokens: options.maxTokens,
    localArmMaxTokens: options.localArmMaxTokens,
    gitSha: options.gitSha,
    gitDirty: options.gitDirty,
    text: item.text,
    findings: resolved.map((f) => ({
      start: f.start,
      end: f.end,
      text: f.text,
      entityType: f.entityType,
      severity: f.severity,
      tier: f.tier,
      source: f.source,
      confidence: f.confidence,
      action: f.action,
    })),
    gold: item.gold.map((g) => ({ ...g })),
    calls,
    parseFailures,
    repairs,
    unresolvedQuotes: counters.unresolvedQuotes,
    unresolvedMentions: counters.unresolvedMentions,
    unknownLabels: counters.unknownLabels,
    duplicatesDropped: counters.duplicatesDropped,
    wholeClauseMentions: counters.wholeClauseMentions,
    wallMs: deps.now() - itemStarted,
    error,
  });
}

/**
 * Runs `items` with at most `limit` in flight, stopping early when `stop()`
 * says the guard has tripped.
 *
 * The stop is checked before each task STARTS rather than only between batches,
 * so a tripped guard costs at most the calls already in flight.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  stop: () => boolean,
  run: (item: T, index: number) => Promise<R>,
): Promise<{ results: R[]; completed: number; stoppedAtIndex: number | null }> {
  const results: R[] = new Array(items.length);
  let next = 0;
  let completed = 0;
  let stoppedAtIndex: number | null = null;

  const worker = async (): Promise<void> => {
    for (;;) {
      if (stop()) {
        if (stoppedAtIndex === null && next < items.length) stoppedAtIndex = next;
        return;
      }
      const index = next++;
      if (index >= items.length) return;
      results[index] = await run(items[index]!, index);
      completed += 1;
    }
  };

  await Promise.all(Array.from({ length: Math.max(1, limit) }, worker));
  return { results, completed, stoppedAtIndex };
}
