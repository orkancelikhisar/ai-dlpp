import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import { z } from "zod";
import type { LlmClient, LlmRequest } from "./client.js";

/**
 * The compiler's only route to a real frontier model. Nothing in the test suite
 * constructs this class — `scripts/record-fixtures.ts` and `sih-compile --live`
 * are its only callers, and they are the only things in this repo that cost
 * money.
 */
const MODEL = "claude-opus-5";

/**
 * Server-side fallback target. Claude Opus 5 runs elevated cybersecurity
 * classifiers, and a data-leak-prevention policy names credentials, API keys and
 * private-key material by construction — a refusal here is a realistic failure,
 * not a hypothetical. `claude-opus-4-8` carries different classifiers, so the
 * fallback genuinely recovers the request rather than relabelling the failure.
 */
const FALLBACK_MODEL = "claude-opus-4-8";

/**
 * Beta gate for the `fallbacks` ARRAY form.
 *
 * The plan specifies `betas: ["server-side-fallback-2026-07-01"]` with
 * `fallbacks: "default"`. That pairing is real, but the installed SDK
 * (@anthropic-ai/sdk@0.110.0) does not type it: `fallbacks` is declared
 * `Array<BetaFallbackParam> | null`, so the scalar `"default"` is a type error,
 * and `AnthropicBeta`'s union lists `server-side-fallback-2026-06-01` with no
 * `-07-01` member. Pairing either header with the other form is a documented
 * 400, so the two must move together — this is the array form, which the
 * installed SDK types end to end. See the plan's Task 9 deviations.
 */
const SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-06-01";

/**
 * Above this, a non-streaming request risks an HTTP timeout, so the call is made
 * over the streaming endpoint instead. Both paths build the same params object
 * and both return a parsed message, so the branch changes the transport and
 * nothing else.
 */
const NON_STREAMING_MAX_TOKENS = 16000;

export class AnthropicLlmClient implements LlmClient {
  private readonly client: Anthropic;

  /**
   * The client is a constructor parameter so a caller can inject a configured
   * one (a different base URL, a longer timeout). Constructing the default reads
   * credentials from the environment — which is why this happens at
   * construction, not at import: importing this module must never touch
   * credentials, or `sih-compile` without `--live` could not safely import it.
   */
  constructor(client: Anthropic = new Anthropic()) {
    this.client = client;
  }

  async complete<T>(request: LlmRequest, schema: z.ZodType<T>): Promise<T> {
    const params = {
      model: MODEL,
      max_tokens: request.maxTokens,
      betas: [SERVER_SIDE_FALLBACK_BETA],
      fallbacks: [{ model: FALLBACK_MODEL }],
      output_config: {
        effort: "high" as const,
        format: betaZodOutputFormat(schema),
      },
      system: request.system,
      messages: [{ role: "user" as const, content: request.user }],
    };

    const response =
      request.maxTokens > NON_STREAMING_MAX_TOKENS
        ? await this.client.beta.messages.stream(params).finalMessage()
        : await this.client.beta.messages.parse(params);

    // Checked BEFORE `content` is read, per the refusal contract: a pre-output
    // refusal returns a 200 with an empty content array, so any code that
    // indexes the response first breaks here rather than reporting the refusal.
    if (response.stop_reason === "refusal") {
      throw new Error(
        `model declined to compile this policy (category ${response.stop_details?.category ?? "unknown"}); ` +
          `see the compilation report for which clause triggered it`,
      );
    }

    // Thinking is ON BY DEFAULT on this model and `max_tokens` caps thinking
    // plus response text together, so a truncated structured output is a live
    // possibility rather than a theoretical one. Named separately because the
    // fix is a larger budget or a lower effort — not a schema change, which is
    // what a bare parse failure would send a reader looking for.
    if (response.stop_reason === "max_tokens") {
      throw new Error(
        `model hit the ${request.maxTokens}-token ceiling before completing schema ` +
          `${request.schemaName}; raise maxTokens for this stage or lower output_config.effort`,
      );
    }

    if (response.parsed_output === null) {
      throw new Error(`model returned output that did not match schema ${request.schemaName}`);
    }

    // Re-validated against the caller's schema rather than trusted and cast.
    // FixtureLlmClient validates every replayed response the same way, and the
    // two clients have to reject the same shapes: a response the live client
    // waved through would be recorded as a fixture the replay then rejects.
    const parsed = schema.safeParse(response.parsed_output);
    if (!parsed.success) {
      throw new Error(
        `model response does not match schema ${request.schemaName}: ${z.prettifyError(parsed.error)}`,
      );
    }
    return parsed.data;
  }
}
