import { z } from "zod";
import { TIER1_BACKENDS, TIER1_LABEL_FORMS } from "@sih/tier1";
import { GoldSpanSchema } from "./corpus.js";

/**
 * Bump when a field is removed or its meaning changes. Plan 8's Python reads
 * this first and refuses a version it does not know, so a silently reshaped
 * record cannot be scored as if it were the old one.
 *
 * STILL 1 despite this shape having changed twice since the constant was
 * written -- `text` was added, and the single `policyHash` became `irHash` plus
 * a verbatim `policyHash`. That looks like the rule above being broken in the
 * same file that states it, and is not, for one reason worth writing down
 * rather than leaving to be re-derived: NO RECORD OF ANY EARLIER SHAPE HAS EVER
 * BEEN WRITTEN. This schema landed before its only producer (`runArm` in
 * driver/run.ts), and nothing in the repo calls `toJsonl` outside this module's
 * own tests, so there is no file anywhere for a version 2 to be distinguished
 * FROM. Bumping would advertise a compatibility boundary that does not exist
 * and oblige Plan 8 to carry a reader for a shape nothing ever emitted.
 *
 * Version 1 therefore describes the only record shape that has ever existed.
 * From the first run that writes a file, the rule above applies literally.
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
     * sha256 of the IR ARTIFACT that ran, lowercase hex: the bytes of the JSON
     * the page loaded, hashed by the page itself (apps/eval/src/page/main.ts).
     *
     * Named `irHash`, not `policyHash`, because `PolicyIr.policyHash` already
     * exists and means something else -- it is the field directly below. Two
     * different values sharing one name across one codebase would leave Plan 8
     * joining a record to an IR, finding the two disagree, and having no way to
     * tell an intended difference from corruption.
     *
     * Required, not optional: an arm's numbers are meaningless without knowing
     * exactly which compiled IR produced them, and "which IR was that?" is
     * unanswerable after the fact. Hashing the artifact's own bytes is also what
     * makes the answer checkable from outside the browser -- `shasum -a 256` on
     * the IR file reproduces it, so a reader can confirm the provenance instead
     * of taking the record's word for it.
     */
    irHash: z.string().regex(/^[0-9a-f]{64}$/),
    /**
     * `PolicyIr.policyHash` carried verbatim -- the COMPILER's sha256 of the
     * policy DOCUMENT (packages/compiler/src/stages/emit.ts). It restores the
     * link `irHash` cannot: `irHash` says which IR ran, this says which prose
     * that IR was compiled from, and scoring wants the whole
     * document -> IR -> numbers chain.
     *
     * It cannot stand in for `irHash`, and the reason is stronger than compiler
     * version drift: compilation is MODEL-DRIVEN, so the same document compiled
     * twice by the same compiler can yield two different IRs both carrying this
     * same value. It identifies the input, never the artifact.
     *
     * Deliberately NOT constrained to /^[0-9a-f]{64}$/ the way `irHash` is, even
     * though a compiled IR's really is 64 hex. This value is copied out of
     * whatever IR the page loaded, and a hand-written fixture legitimately
     * carries a placeholder -- apps/eval/fixtures/minimal-ir.json says
     * "test-hash". Tightening this would force that fixture to state the hash of
     * a policy document that does not exist: the harness lying about its own
     * provenance so its own schema would pass. `min(1)` is exactly what core's
     * PolicyIrSchema requires of the field (packages/core/src/policy/schema.ts),
     * so this is as strict as the value's own source and no stricter.
     */
    policyHash: z.string().min(1),
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
    /**
     * The `TierConfig` handed to `detect` for this item, recorded because
     * otherwise the record asserts an arm NAME and nothing else. `arm` is a free
     * string the caller chooses; without the config beside it, a run with every
     * tier switched off is byte-for-byte identical to a detector that legitimately
     * found nothing, and both read as "this arm scored zero". This is the same
     * disease as `backend` -- intent recorded in place of fact -- except that here
     * the fact is available, because this object is the exact one `detect`
     * received.
     *
     * Kept in step with TierConfig in packages/core/src/detect/types.ts. The three
     * booleans are required there and so are required here; the model names and
     * `backend` are optional there and stay optional, so a tier-0 arm does not
     * have to invent values for tiers it never ran.
     */
    config: z.object({
      tier0: z.boolean(),
      tier1: z.boolean(),
      tier2: z.boolean(),
      t1Model: z.string().optional(),
      t2Model: z.string().optional(),
      backend: z.enum(["wasm", "webgpu"]).optional(),
    }),
    /**
     * The FULLY RESOLVED `Tier1Config` the tagger in the page was constructed
     * with. Present exactly when tier 1 ran, absent otherwise.
     *
     * `config` above is core's `TierConfig`: three booleans, two model names and
     * a backend. The tier-1 ladder has six dimensions -- modelId, precision,
     * backend, threshold, maxWidth, labelForm -- and three of them appear
     * nowhere in `TierConfig`. So two arms differing only in `threshold`,
     * `maxWidth` or `labelForm` used to emit records that were byte-identical in
     * every field a scorer can group by: a matrix that ran four arms and handed
     * Plan 8 four indistinguishable ones.
     *
     * What lands here is the config off `loadTier1`'s report -- the object the
     * GlinerSpanTagger was constructed with, after `resolveTier1Config` filled in
     * and validated every field -- and never the partial object an arm asked
     * for. Same distinction `backend` above is a warning about: intent is not
     * evidence of what ran.
     *
     * The vocabularies are IMPORTED from @sih/tier1 rather than restated, unlike
     * core's unions higher up in this file. Those are types, with no runtime
     * value to import; `TIER1_BACKENDS` and `TIER1_LABEL_FORMS` are exported
     * arrays, so importing them makes drift impossible instead of merely
     * discouraged.
     */
    tier1Config: z
      .object({
        modelId: z.string().min(1),
        backend: z.enum(TIER1_BACKENDS),
        /** Restates resolveTier1Config's own bound, (0, 1]. */
        threshold: z.number().gt(0).lte(1),
        maxWidth: z.number().int().positive(),
        labelForm: z.enum(TIER1_LABEL_FORMS),
      })
      .optional(),
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
  )
  // The four checks below tie the record's LABELS to the configuration that
  // produced them. Each one is a way a complete, schema-valid, perfectly
  // scoreable file can describe a run that did not happen, and each was
  // reachable before Task 12: nothing validates a record on the producing path,
  // so the writer runs this schema and these are the checks it runs.
  .refine((r) => r.config.backend === undefined || r.config.backend === r.backend, {
    message: "config.backend contradicts the record's own backend",
  })
  .refine((r) => r.config.tier1 === (r.tier1Config !== undefined), {
    // Both directions. Tier 1 on with no config is an arm whose six-dimensional
    // rung went unrecorded; a config with tier 1 off is a tier-0 run wearing a
    // tier-1 label, which is the exact substitution runArm refuses to navigate
    // in order to prevent.
    message: "tier1Config must be present exactly when config.tier1 is true",
  })
  .refine((r) => r.tier1Config === undefined || r.tier1Config.backend === r.backend, {
    message: "tier1Config.backend contradicts the record's own backend",
  })
  .refine(
    (r) =>
      r.tier1Config === undefined ||
      r.config.t1Model === undefined ||
      r.config.t1Model === r.tier1Config.modelId,
    { message: "config.t1Model names a different rung than tier1Config.modelId" },
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
