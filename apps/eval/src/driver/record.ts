import { z } from "zod";
import { GoldSpanSchema } from "./corpus.js";

/**
 * Bump when a field is removed or its meaning changes. Plan 8's Python reads
 * this first and refuses a version it does not know, so a silently reshaped
 * record cannot be scored as if it were the old one.
 */
export const RECORD_SCHEMA_VERSION = 1;

export const RunRecordSchema = z.object({
  schemaVersion: z.literal(RECORD_SCHEMA_VERSION),
  runId: z.string().min(1),
  itemId: z.string().min(1),
  policy: z.string().min(1),
  /**
   * sha256 of the policy document the IR was compiled from. Required, not
   * optional: an arm's numbers are meaningless without knowing exactly which
   * compiled policy produced them, and "which IR was that?" is unanswerable
   * after the fact.
   */
  policyHash: z.string().regex(/^[0-9a-f]{64}$/),
  arm: z.string().min(1),
  backend: z.enum(["wasm", "webgpu"]),
  provider: z.string().min(1),
  /**
   * Offsets here carry the SAME unit contract as gold spans: UTF-16 code units
   * into the corpus item's text, not code points. GoldSpanSchema in corpus.ts
   * documents it in full, including the measured Python failure and the
   * `utf-16-le` decode a Python reader needs; findings arrive straight from
   * core, whose invariant is `text === message.slice(start, end)`, so they are
   * UTF-16 by construction and a scorer must compare them against gold in that
   * same unit. `text` is the per-span cross-check.
   */
  findings: z.array(
    z.object({
      /** UTF-16 code-unit offset. See GoldSpanSchema in corpus.ts. */
      start: z.number().int().nonnegative(),
      /** Exclusive UTF-16 code-unit offset. See GoldSpanSchema in corpus.ts. */
      end: z.number().int().positive(),
      /** Exactly the message slice at [start, end) -- the non-JS reader's check. */
      text: z.string(),
      entityType: z.string(),
      severity: z.string(),
      tier: z.number().int(),
      source: z.string(),
      confidence: z.number(),
      action: z.string(),
    }),
  ),
  /**
   * Copied from the corpus item so a record scores standalone, without a join.
   * Same UTF-16 code-unit offsets as it had in the corpus -- nothing rebases it.
   */
  gold: z.array(GoldSpanSchema),
  timings: z.object({
    tier0Ms: z.number(),
    tier1Ms: z.number().optional(),
    tier2Ms: z.number().optional(),
  }),
  /**
   * Set when detection THREW for this item. The record is still written: an arm
   * that crashes on 5% of the corpus and one that scores 0 on it are different
   * results, and dropping the row makes them look identical.
   */
  error: z.string().nullable(),
});

export type RunRecord = z.infer<typeof RunRecordSchema>;

export function toJsonl(records: readonly RunRecord[]): string {
  return records.map((r) => JSON.stringify(r)).join("\n") + "\n";
}
