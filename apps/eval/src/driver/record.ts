import { z } from "zod";
import { GoldSpanSchema } from "./corpus.js";

/**
 * Bump when a field is removed or its meaning changes. Plan 8's Python reads
 * this first and refuses a version it does not know, so a silently reshaped
 * record cannot be scored as if it were the old one.
 */
export const RECORD_SCHEMA_VERSION = 1;

/**
 * A finding as core produced it, validated against core's ACTUAL unions rather
 * than widened to strings.
 *
 * The widening is worth naming because it was the original shape here and it is
 * the tempting one: `findings` crosses a process boundary as JSON, so typing it
 * loosely feels like tolerance. It is not. Plan 8 compares `findings[].action`
 * against `gold[].action`, and gold's action is a validated enum -- so a widened
 * findings side means one comparison with a checked vocabulary on the left and
 * an unchecked one on the right, where `"blocked"` silently never matches
 * `"block"` and reads as a miss rather than a bug. MEASURED on the widened
 * version: a finding with severity "banana", tier -7, confidence 42, action
 * "obliterate" and empty entityType/source validated cleanly.
 *
 * Unions kept in step with packages/core/src/policy/types.ts (Severity, Tier,
 * Action) and detect/types.ts (Finding). If core adds an action, this fails
 * loudly on the next run, which is the intended coupling.
 */
export const RecordFindingSchema = z.object({
  /** UTF-16 code-unit offset into the record's `text`. See GoldSpanSchema in corpus.ts. */
  start: z.number().int().nonnegative(),
  /** Exclusive UTF-16 code-unit offset. See GoldSpanSchema in corpus.ts. */
  end: z.number().int().positive(),
  /**
   * Exactly `text.slice(start, end)` on this record's own `text`. Enforced by
   * the refine at the bottom of RunRecordSchema, so it is a real cross-check a
   * non-JS reader can run rather than a claim.
   *
   * `.min(1)` matches GoldSpanSchema.text and is load-bearing rather than
   * symmetry for its own sake: without it a degenerate span walks straight
   * through that refine, because `slice()` returns "" for any inverted or
   * zero-width range and "" compares equal to a claimed text of "". MEASURED on
   * the version without it, all three ACCEPTED as findings while gold rejected
   * every one: {start:999,end:1}, {start:5,end:5}, {start:8,end:3}.
   */
  text: z.string().min(1),
  /** An entityType id from the IR that ran -- same namespace as a gold span's. */
  entityType: z.string().min(1),
  severity: z.enum(["low", "medium", "high", "critical"]),
  /** Core's Tier: 0 rules, 1 span tagger, 2 semantic judge. */
  tier: z.union([z.literal(0), z.literal(1), z.literal(2)]),
  /** Rule id (tier 0) or model identifier (tiers 1-2); never empty. */
  source: z.string().min(1),
  /** 0..1, per Finding.confidence. */
  confidence: z.number().min(0).max(1),
  /**
   * Core's Action union exactly. Note this is NARROWER than a gold span's,
   * which also admits "none": a resolved finding always carries a real action,
   * whereas gold can label a span the policy deliberately permits.
   */
  action: z.enum(["allow", "pseudonymize", "redact", "block"]),
});

export const RunRecordSchema = z
  .object({
    schemaVersion: z.literal(RECORD_SCHEMA_VERSION),
    runId: z.string().min(1),
    itemId: z.string().min(1),
    /** The corpus item's `policy` name, carried through. See CorpusItemSchema. */
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
     * The message the offsets below index into, copied verbatim from the corpus
     * item. This is what makes "scores standalone, without a join" true rather
     * than aspirational: without it a reader holds spans and no text to index,
     * so the UTF-16 cross-check every offset comment points at is impossible
     * from a record alone.
     *
     * MEASURED cost at projected Plan 7/8 volume (1,500 items x 8 arms x 3
     * policies = 36,000 records): the baseline record is ~456 B, so the run is
     * ~16.4 MB without text. Carrying text adds ~3.7 MB at this corpus's
     * ~93 B mean message (20.1 MB total, +23%) and ~18.4 MB at a 500 B mean
     * (34.8 MB, +112%). Each message is repeated 24 times, once per
     * arm x policy. Tens of megabytes for a file Python reads once is a price
     * worth paying to make every record self-verifying.
     */
    text: z.string().min(1),
    findings: z.array(RecordFindingSchema),
    /**
     * Copied from the corpus item so a record scores standalone, without a join.
     * Same UTF-16 code-unit offsets as it had in the corpus -- nothing rebases
     * it -- and, like gold in the corpus, NOT necessarily sorted or disjoint,
     * unlike `findings`.
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
  })
  .refine(
    (r) =>
      [...r.findings, ...r.gold].every(
        (s) =>
          // `start < end` is checked explicitly, not left to the slice
          // comparison. `end` being positive was meant to bar degenerate spans
          // and an inverted `start` walks around it: slice() clamps an inverted
          // range to "" rather than throwing, so {start:999,end:1} agrees with
          // a claimed text of "" and an out-of-range start never surfaces.
          s.start < s.end &&
          s.end <= r.text.length &&
          r.text.slice(s.start, s.end) === s.text,
      ),
    { message: "a span's offsets do not hold the text it names" },
  );

export type RunRecord = z.infer<typeof RunRecordSchema>;
export type RecordFinding = z.infer<typeof RecordFindingSchema>;

/**
 * Serializes records as JSONL, one per line, with a terminating newline.
 *
 * U+2028 and U+2029 are escaped explicitly because `JSON.stringify` does NOT
 * escape them -- they are valid unescaped JSON string content -- while Python
 * treats both as line terminators. MEASURED: a record whose text contains a raw
 * U+2028 is emitted as one line by this function, and Python's
 * `data.splitlines()` then yields TWO fragments, both of which fail
 * `json.loads` with "Unterminated string". That is a loud failure rather than a
 * silent one, so it ranks well below the offset hazard -- but it costs one
 * replace to remove, and scraped chat is exactly where a stray U+2028 comes
 * from. `\\u2028` is valid JSON escape syntax, so the output stays parseable by
 * any conformant reader.
 */
export function toJsonl(records: readonly RunRecord[]): string {
  return (
    records
      .map((r) => JSON.stringify(r).replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029"))
      .join("\n") + "\n"
  );
}
