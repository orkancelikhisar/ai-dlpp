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
 * So a Python reader MUST index via UTF-16, e.g.
 * `text.encode("utf-16-le")[2*start:2*end].decode("utf-16-le")`.
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
  entityType: z.string().min(1),
  /** What the policy demands for this span. "none" means the span is not a violation under this policy. */
  action: z.enum(["none", "allow", "pseudonymize", "redact", "block"]),
});

export const CorpusItemSchema = z
  .object({
    id: z.string().min(1),
    text: z.string().min(1),
    policy: z.string().min(1),
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
 */
export function loadCorpus(jsonl: string): CorpusItem[] {
  const items: CorpusItem[] = [];
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
    items.push(parsed.data);
  }
  return items;
}
