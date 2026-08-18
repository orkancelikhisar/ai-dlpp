import { z } from "zod";

export const GoldSpanSchema = z.object({
  start: z.number().int().nonnegative(),
  end: z.number().int().positive(),
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
