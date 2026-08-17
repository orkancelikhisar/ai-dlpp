import { z } from "zod";
import type { LlmClient } from "../llm/client.js";

/**
 * The model's raw output shape. Every element carries a sourceQuote — that is
 * the whole anti-hallucination mechanism (spec §3.2 stage 1): a candidate the
 * model cannot ground in the document is a candidate the model invented.
 */
export const ExtractionSchema = z.object({
  entityTypes: z.array(
    z.object({
      id: z.string().min(1),
      tier: z.union([z.literal(0), z.literal(1), z.literal(2)]),
      nlDefinition: z.string().min(1),
      examples: z.array(z.string()),
      counterExamples: z.array(z.string()),
      severity: z.enum(["low", "medium", "high", "critical"]),
      surrogateKind: z.enum(["person-name", "org-name", "id-number", "opaque"]).optional(),
      neverPseudonymize: z.boolean().optional(),
      sourceQuote: z.string(),
    }),
  ),
  rules: z.array(
    z.object({
      id: z.string().min(1),
      entityType: z.string().min(1),
      regex: z.string().optional(),
      validator: z.string().optional(),
      contextBoost: z.array(z.string()).optional(),
      entropyThreshold: z.number().optional(),
      minLength: z.number().optional(),
      sourceQuote: z.string(),
    }),
  ),
  semanticPredicates: z.array(
    z.object({
      id: z.string().min(1),
      nlPredicate: z.string().min(1),
      scope: z.enum(["segment", "message"]),
      severity: z.enum(["low", "medium", "high", "critical"]),
      sourceQuote: z.string(),
    }),
  ),
  actions: z.array(
    z.object({
      entityType: z.string().min(1),
      action: z.enum(["allow", "pseudonymize", "redact", "block"]),
      /** Natural-language provider mention; grounded to adapter ids in Task 4. */
      providerMention: z.string().optional(),
      sourceQuote: z.string(),
    }),
  ),
  failMode: z.enum(["open", "closed"]),
});

export type Extraction = z.infer<typeof ExtractionSchema>;

const SYSTEM = [
  "You compile a natural-language data-leak policy into a structured detection IR.",
  "",
  "Rules you must follow:",
  "1. Every item you emit carries a sourceQuote: a span copied VERBATIM from the policy",
  "   document that states the rule. Items without a grounded quote are discarded, so an",
  "   invented rule is wasted output. Quote the sentence, not the section heading.",
  "2. entityType ids are OUTBOUND-VISIBLE: they ship to the LLM provider inside redaction",
  "   markers like [REDACTED:<id>]. Never derive an id from a confidential noun. Write",
  "   'internal-codename', not 'project-titan'. Ids are lowercase kebab-case.",
  "3. Assign each entityType a tier: 0 for anything with a fixed surface form a regex can",
  "   match (identifiers, key prefixes), 1 for named entities needing a span model,",
  "   2 for classes with no surface form at all.",
  "4. Credentials and secrets get neverPseudonymize: true. A format-valid fake credential",
  "   is a lie waiting to be pasted somewhere.",
  "5. Emit regexes as data only. For checksum logic name a validator from this fixed list:",
  "   luhn, verhoeff, pan-structure, jwt-shape. Never invent a validator name.",
  "6. A regex must never match the empty string.",
  "7. Semantic predicates are for classes with no surface form (unreleased financials,",
  "   clinical narrative). Do not emit a regex for those.",
].join("\n");

export interface RejectedCandidate {
  id: string;
  kind: string;
  reason: string;
}

/** Whitespace-insensitive, case- and character-sensitive. */
export function normalizeForQuoteMatch(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

/**
 * The anti-hallucination gate. Case sensitivity is deliberate: a model that
 * changed "PAN" to "pan" did not copy the document, it paraphrased it, and a
 * paraphrase is exactly the failure this gate exists to catch.
 */
export function groundQuotes<T extends { id: string; sourceQuote: string }>(
  document: string,
  candidates: readonly T[],
  kind = "candidate",
): { grounded: T[]; rejected: RejectedCandidate[] } {
  const haystack = normalizeForQuoteMatch(document);
  const grounded: T[] = [];
  const rejected: RejectedCandidate[] = [];
  for (const candidate of candidates) {
    const needle = normalizeForQuoteMatch(candidate.sourceQuote);
    if (needle.length === 0) {
      rejected.push({ id: candidate.id, kind, reason: "sourceQuote is empty" });
    } else if (!haystack.includes(needle)) {
      rejected.push({ id: candidate.id, kind, reason: "sourceQuote not found in policy document" });
    } else {
      grounded.push(candidate);
    }
  }
  return { grounded, rejected };
}

export interface ExtractResult extends Extraction {
  rejected: RejectedCandidate[];
}

export async function extract(client: LlmClient, document: string): Promise<ExtractResult> {
  const raw = await client.complete(
    { system: SYSTEM, user: document, schemaName: "Extraction", maxTokens: 16000 },
    ExtractionSchema,
  );

  const entityTypes = groundQuotes(document, raw.entityTypes, "entityType");
  const rules = groundQuotes(document, raw.rules, "rule");
  const semanticPredicates = groundQuotes(document, raw.semanticPredicates, "semanticPredicate");
  const actions = groundQuotes(
    document,
    // Actions have no id of their own, so one is synthesized purely to name the
    // action in a rejection report — "client-name:default" reads; an array index
    // would not. It stays on the grounded object, which is harmless: the ground
    // stage (Task 4) reads entityType/action/providerMention only.
    raw.actions.map((a) => ({ ...a, id: `${a.entityType}:${a.providerMention ?? "default"}` })),
    "action",
  );

  return {
    entityTypes: entityTypes.grounded,
    rules: rules.grounded,
    semanticPredicates: semanticPredicates.grounded,
    actions: actions.grounded,
    failMode: raw.failMode,
    rejected: [
      ...entityTypes.rejected,
      ...rules.rejected,
      ...semanticPredicates.rejected,
      ...actions.rejected,
    ],
  };
}
