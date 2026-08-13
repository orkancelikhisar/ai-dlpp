import { z } from "zod";

const SeveritySchema = z.enum(["low", "medium", "high", "critical"]);
const ActionSchema = z.enum(["allow", "pseudonymize", "redact", "block"]);

const EntityTypeSchema = z.object({
  id: z.string().min(1),
  tier: z.union([z.literal(0), z.literal(1), z.literal(2)]),
  nlDefinition: z.string().min(1),
  examples: z.array(z.string()),
  counterExamples: z.array(z.string()),
  severity: SeveritySchema,
});

const RuleSchema = z
  .object({
    id: z.string().min(1),
    entityType: z.string().min(1),
    regex: z.string().optional(),
    validator: z.string().optional(),
    contextBoost: z.array(z.string()).optional(),
    entropyThreshold: z.number().positive().optional(),
    minLength: z.number().int().positive().optional(),
  })
  .refine((r) => r.regex !== undefined || r.entropyThreshold !== undefined, {
    message: "rule must have regex or entropyThreshold",
  });

export const PolicyIrSchema = z
  .object({
    irVersion: z.string(),
    policyHash: z.string().min(1),
    entityTypes: z.array(EntityTypeSchema).min(1),
    rules: z.array(RuleSchema),
    semanticPredicates: z.array(
      z.object({ id: z.string().min(1), nlPredicate: z.string().min(1), scope: z.enum(["segment", "message"]) }),
    ),
    actions: z.object({
      default: z.record(z.string(), ActionSchema),
      providerOverrides: z.record(z.string(), z.record(z.string(), ActionSchema)).optional(),
    }),
    failMode: z.enum(["open", "closed"]),
    latencyBudgetMs: z.number().int().positive(),
    provenance: z.record(z.string(), z.object({ clause: z.string(), quote: z.string() })),
  })
  .superRefine((ir, ctx) => {
    const ids = new Set(ir.entityTypes.map((e) => e.id));
    for (const e of ir.entityTypes) {
      if (!(e.id in ir.actions.default)) {
        ctx.addIssue({ code: "custom", message: `entityType "${e.id}" has no action mapping` });
      }
    }
    for (const r of ir.rules) {
      if (!ids.has(r.entityType)) {
        ctx.addIssue({ code: "custom", message: `rule "${r.id}" references unknown entityType "${r.entityType}"` });
      }
    }
  });
