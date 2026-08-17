import { createHash } from "node:crypto";
import type { z } from "zod";
import type { LlmClient, LlmRequest } from "./client.js";

/**
 * Fixture key. NUL-separated because concatenation alone is ambiguous:
 * ("ab","c") and ("a","bc") would otherwise share a key and silently replay
 * each other's response.
 */
export function requestHash(request: LlmRequest): string {
  return createHash("sha256")
    .update(
      [request.system, request.user, request.schemaName, String(request.maxTokens)].join("\u0000"),
    )
    .digest("hex")
    .slice(0, 16);
}

/**
 * Replays committed responses. Tests use this exclusively — a compiler test
 * that reaches the network would be non-deterministic and cost money, so the
 * real client is never constructed inside the suite.
 */
export class FixtureLlmClient implements LlmClient {
  constructor(private readonly fixtures: ReadonlyMap<string, unknown>) {}

  async complete<T>(request: LlmRequest, schema: z.ZodType<T>): Promise<T> {
    const hash = requestHash(request);
    if (!this.fixtures.has(hash)) {
      throw new Error(
        `no fixture for request ${hash} (schema ${request.schemaName}); ` +
          `record it with scripts/record-fixtures.ts`,
      );
    }
    const parsed = schema.safeParse(this.fixtures.get(hash));
    if (!parsed.success) {
      throw new Error(`fixture ${hash} does not match schema ${request.schemaName}: ${parsed.error.message}`);
    }
    return parsed.data;
  }
}
