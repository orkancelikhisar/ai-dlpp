import { createHash } from "node:crypto";
import { z } from "zod";
import type { LlmClient, LlmRequest } from "./client.js";

/** Separator for the fixture key. See requestHash. */
const FIELD_SEP = "\u0000";

/**
 * Fixture key. NUL-separated because concatenation alone is ambiguous:
 * ("ab","c") and ("a","bc") would otherwise share a key and silently replay
 * each other's response.
 *
 * Truncated to 16 hex chars (64 bits). The corpus is dozens of fixtures, all
 * repo-authored with no adversarial input, so collision probability is
 * negligible; the short form earns its keep as the fixture filename.
 *
 * The destructure below is an exhaustiveness guard, not style: every field of
 * LlmRequest must reach the hash, and a field added later but forgotten here
 * would silently make two different requests share one fixture — the failure
 * mode that looks like a pass. `_guard` stops compiling the moment `rest` is
 * non-empty.
 */
export function requestHash(request: LlmRequest): string {
  const { system, user, schemaName, maxTokens, ...rest } = request;
  // Adding a field to LlmRequest? Add it to the join below — and re-record fixtures.
  const _guard: keyof typeof rest extends never ? true : never = true;
  void _guard;
  return createHash("sha256")
    .update([system, user, schemaName, String(maxTokens)].join(FIELD_SEP))
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
      // Names the hand-authoring convention as well as the recorder: tasks that
      // predate scripts/record-fixtures.ts write these files by hand from this
      // very message, and pointing only at a script that does not exist yet
      // leaves that reader with nowhere to go.
      throw new Error(
        `no fixture for request ${hash} (schema ${request.schemaName}); ` +
          `write test/fixtures/llm/${hash}.json by hand, ` +
          `or record live via scripts/record-fixtures.ts (Task 9)`,
      );
    }
    const parsed = schema.safeParse(this.fixtures.get(hash));
    if (!parsed.success) {
      // prettifyError, not .message: zod v4's message is a JSON dump of the
      // issue array, which is unreadable in a vitest failure.
      throw new Error(
        `fixture ${hash} does not match schema ${request.schemaName}: ${z.prettifyError(parsed.error)}`,
      );
    }
    return parsed.data;
  }
}
