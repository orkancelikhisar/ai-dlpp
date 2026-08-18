import { z } from "zod";

/**
 * OFFSET UNIT -- read before writing a reader for this file.
 *
 * `start` and `end` are absolute offsets into the item's `text` measured in
 * UTF-16 CODE UNITS: plain JavaScript string indices, the unit
 * `String.prototype.slice` takes. That is not a preference. JS is the producing
 * runtime, and core's span-fidelity invariant is literally
 * `text === message.slice(start, end)` (see Finding.text in
 * packages/core/src/detect/types.ts), so every offset in this pipeline is
 * UTF-16 from the moment a detector produces it.
 *
 * This matters most to the Python side of the spec 2.2 boundary, because
 * Python's `str` is indexed by CODE POINT, not code unit. The two agree until
 * an astral character (emoji, most non-BMP scripts) appears earlier in the
 * message, after which Python runs 1 unit short per astral character. MEASURED
 * on corpora/fixtures/smoke.jsonl, item `pos-emoji-before-pan`, gold span
 * [33,43):
 *
 *   text[33:43]                                       -> "CPT1234H t"   WRONG
 *   text.encode("utf-16-le")[66:86].decode("utf-16-le") -> "ABCPT1234H"  correct
 *
 * len(text) is 60 there while the message is 62 code units long. Across that
 * corpus a reader slicing `str` directly gets 6 of 7 spans right and silently
 * corrupts the seventh -- it fails toward wrong numbers, not a crash, which is
 * why this is stated here rather than left to be discovered.
 *
 * So a Python reader MUST index via UTF-16, and MUST pass `surrogatepass`:
 *
 *   text.encode("utf-16-le", "surrogatepass")[2*start:2*end] \
 *       .decode("utf-16-le", "surrogatepass")
 *
 * The error handler is not defensive noise. This schema accepts a `text`
 * containing a LONE SURROGATE -- scraped chat truncated mid-emoji produces
 * them, which is Plan 7's exact input -- and JSON survives the round trip.
 * MEASURED: JS `JSON.stringify` emits it escaped as `\ud800`, valid JSON that
 * `json.loads` parses happily, and then a bare
 * `text.encode("utf-16-le")` raises
 * `UnicodeEncodeError: 'utf-16-le' codec can't encode character '\ud800'`.
 * With `surrogatepass` the same string encodes and slices normally.
 *
 * `text` below (and `findings[].text` on a record) is the recovery path and the
 * cross-check: it holds exactly what the offsets are supposed to select, so a
 * reader can verify its own indexing on every span it reads and abort on the
 * first disagreement instead of scoring against fiction. The refine at the
 * bottom of CorpusItemSchema enforces that agreement on the JS side.
 */
export const GoldSpanSchema = z.object({
  /** UTF-16 code-unit offset into the item's `text`. See the note above. */
  start: z.number().int().nonnegative(),
  /** Exclusive UTF-16 code-unit offset. See the note above. */
  end: z.number().int().positive(),
  /** Exactly `text.slice(start, end)` -- the cross-check for a non-JS reader. */
  text: z.string().min(1),
  /**
   * An entityType id from the IR this item is labelled against -- the same
   * namespace as `Finding.entityType`, which is what makes a finding and a gold
   * span comparable at all. NOT validated against any IR here: this module
   * never loads one, and the corpus is authored before the policy it will be
   * scored under is necessarily compiled. A typo therefore surfaces as an
   * entity that scores 0 recall, not as a load error, so a scorer should report
   * the gold entityTypes it saw and let a human notice an unexpected one.
   */
  entityType: z.string().min(1),
  /**
   * What the policy demands for this span. "none" means the span is not a
   * violation under this policy -- i.e. a span worth labelling (a detector may
   * well fire on it) that the policy deliberately permits, which is how a
   * false positive gets distinguished from a correct-but-allowed detection.
   * The other four are core's `Action` union verbatim.
   */
  action: z.enum(["none", "allow", "pseudonymize", "redact", "block"]),
});

export const CorpusItemSchema = z
  .object({
    id: z.string().min(1),
    text: z.string().min(1),
    /**
     * Which policy DOCUMENT this item's gold labels were written against, as a
     * short human-chosen name (the smoke corpus uses "p-fin"). Labels are only
     * meaningful relative to a policy -- the same PAN is "block" under one and
     * "allow" under another -- so an item scored against a different policy
     * than it was labelled for is measuring nothing.
     *
     * Deliberately a NAME, not a hash: the corpus is authored by hand against a
     * policy in prose, and it must stay valid while that policy is recompiled.
     * The exact IR is pinned on the RECORD instead (`irHash`), which is the
     * side that knows what actually ran. Nothing in this file records which IR
     * an item's labels were written against, so a policy edited without
     * relabelling its corpus fails silently -- a real gap, and the reason
     * `policy` should change name whenever the labels stop matching the prose.
     */
    policy: z.string().min(1),
    /**
     * NOT required to be sorted, and NOT required to be disjoint. A message can
     * carry two labels at the same offsets (a value that is both a credential
     * and a client identifier) or nested ones. This is the opposite of
     * `DetectionResult.findings`, which core guarantees sorted and pairwise
     * disjoint -- so a scorer must not assume it can walk the two arrays in
     * lockstep, and must sort gold itself if it needs order.
     */
    gold: z.array(GoldSpanSchema),
    /** Free-form provenance from Plan 7 (carrier source, injection dimensions). Passed through untouched. */
    meta: z.record(z.string(), z.unknown()).optional(),
  })
  .refine(
    (item) =>
      item.gold.every((g) => g.end <= item.text.length && item.text.slice(g.start, g.end) === g.text),
    { message: "a gold span's offsets do not hold the text it names" },
  );

export type CorpusItem = z.infer<typeof CorpusItemSchema>;

/**
 * Parses JSONL. Blank lines are skipped; every other line must be a valid item.
 * The line number is in the error because a 1,500-item corpus with one bad line
 * is otherwise unfixable.
 *
 * UNKNOWN KEYS ARE SILENTLY DROPPED, at every level -- item, gold span, and
 * record. That is zod's default `strip` behaviour, not a decision made here,
 * and it is worth knowing because a Plan 7 corpus carrying extra provenance
 * columns loses them here without a word unless they are nested under `meta`,
 * which is typed to keep whatever it is given. Put provenance in `meta`.
 */
export function loadCorpus(jsonl: string): CorpusItem[] {
  const items: CorpusItem[] = [];
  // Ids must be unique: Plan 8 joins records back to items by `itemId`, so a
  // duplicate does not collide loudly -- it double-counts one item and drops
  // the other, shifting every aggregate by an amount nobody can see afterwards.
  const seen = new Set<string>();
  const lines = jsonl.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!.trim();
    if (line === "") continue;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch (cause) {
      throw new Error(`corpus line ${i + 1} is not valid JSON`, { cause });
    }
    const parsed = CorpusItemSchema.safeParse(raw);
    if (!parsed.success) {
      throw new Error(
        `corpus line ${i + 1} is not a valid corpus item: ${z.prettifyError(parsed.error)}`,
      );
    }
    if (seen.has(parsed.data.id)) {
      throw new Error(`corpus line ${i + 1} repeats id "${parsed.data.id}"`);
    }
    seen.add(parsed.data.id);
    items.push(parsed.data);
  }
  return items;
}
