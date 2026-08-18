import type { z } from "zod";

/**
 * A single frontier-model call. Deliberately narrow: system + user + a named
 * schema. Everything the compiler asks of the model is a structured extraction,
 * so there is no free-text completion path to abuse.
 */
export interface LlmRequest {
  system: string;
  user: string;
  /** Names the response schema; part of the fixture key so schema changes miss. */
  schemaName: string;
  maxTokens: number;
}

/**
 * The schema a response is validated against.
 *
 * Aliased rather than written as `z.ZodType<T>` at every use site so that code
 * OUTSIDE this package can implement `LlmClient` without resolving zod itself —
 * `scripts/record-fixtures.ts` lives at the repo root, where pnpm's strict
 * layout puts zod out of reach, and it needs this signature to wrap a client.
 */
export type LlmSchema<T> = z.ZodType<T>;

/**
 * The compiler's ONLY route to a frontier model (spec §3.2: the sole place a
 * cloud model is called, and it never sees user data). Injectable so tests
 * replay committed fixtures and never touch the network.
 */
export interface LlmClient {
  complete<T>(request: LlmRequest, schema: LlmSchema<T>): Promise<T>;
}
