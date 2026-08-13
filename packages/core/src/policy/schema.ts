import { z } from "zod";
import type { PolicyIrInput } from "./types.js";

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

/**
 * A rule is exactly one of two variants: regex or entropy. The variants are
 * mutually exclusive (detection dispatch is if/else-if, so a rule carrying both
 * would silently drop one check), and each variant's optional fields are only
 * legal on that variant.
 */
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
  .superRefine((r, ctx) => {
    const isRegexRule = r.regex !== undefined;
    const isEntropyRule = r.entropyThreshold !== undefined;

    if (isRegexRule && isEntropyRule) {
      ctx.addIssue({
        code: "custom",
        message: `rule "${r.id}" cannot have both regex and entropyThreshold`,
        path: ["entropyThreshold"],
      });
    } else if (!isRegexRule && !isEntropyRule) {
      ctx.addIssue({ code: "custom", message: `rule "${r.id}" must have regex or entropyThreshold` });
    }

    if (!isRegexRule) {
      if (r.validator !== undefined) {
        ctx.addIssue({
          code: "custom",
          message: `rule "${r.id}": validator is only valid on regex rules`,
          path: ["validator"],
        });
      }
      if (r.contextBoost !== undefined) {
        ctx.addIssue({
          code: "custom",
          message: `rule "${r.id}": contextBoost is only valid on regex rules`,
          path: ["contextBoost"],
        });
      }
    }
    if (!isEntropyRule && r.minLength !== undefined) {
      ctx.addIssue({
        code: "custom",
        message: `rule "${r.id}": minLength is only valid on entropy rules`,
        path: ["minLength"],
      });
    }
  });

const SemanticPredicateSchema = z.object({
  id: z.string().min(1),
  nlPredicate: z.string().min(1),
  scope: z.enum(["segment", "message"]),
});

/** Reports every id that appears more than once, anchored at the offending element. */
function checkDuplicateIds(
  items: ReadonlyArray<{ id: string }>,
  field: string,
  label: string,
  ctx: z.RefinementCtx,
): Set<string> {
  const seen = new Set<string>();
  items.forEach((item, i) => {
    if (seen.has(item.id)) {
      ctx.addIssue({ code: "custom", message: `duplicate ${label} id "${item.id}"`, path: [field, i, "id"] });
    }
    seen.add(item.id);
  });
  return seen;
}

/**
 * The IR is a hash-stamped artifact, so the top level is strict: unknown keys are
 * rejected rather than stripped, since silent stripping would hide drift or tampering.
 */
export const PolicyIrSchema = z
  .strictObject({
    irVersion: z.string().min(1),
    policyHash: z.string().min(1),
    entityTypes: z.array(EntityTypeSchema).min(1),
    rules: z.array(RuleSchema),
    semanticPredicates: z.array(SemanticPredicateSchema),
    actions: z.object({
      default: z.record(z.string(), ActionSchema),
      providerOverrides: z.record(z.string(), z.record(z.string(), ActionSchema)).optional(),
    }),
    failMode: z.enum(["open", "closed"]),
    latencyBudgetMs: z.number().int().positive(),
    // Keys must reference something declared, but completeness (every rule HAS
    // provenance) is the compiler's Extract-stage guarantee, not a schema invariant.
    provenance: z.record(z.string(), z.object({ clause: z.string(), quote: z.string() })),
  })
  .superRefine((ir, ctx) => {
    const entityIds = checkDuplicateIds(ir.entityTypes, "entityTypes", "entityType", ctx);
    const ruleIds = checkDuplicateIds(ir.rules, "rules", "rule", ctx);
    const predicateIds = checkDuplicateIds(ir.semanticPredicates, "semanticPredicates", "semanticPredicate", ctx);

    // Own keys only: `in` would consult the prototype chain, so an entityType named
    // "toString" would appear to have an action mapping it does not have.
    const defaultKeys = new Set(Object.keys(ir.actions.default));

    ir.entityTypes.forEach((e, i) => {
      if (!defaultKeys.has(e.id)) {
        ctx.addIssue({
          code: "custom",
          message: `entityType "${e.id}" has no action mapping`,
          path: ["entityTypes", i, "id"],
        });
      }
    });

    ir.rules.forEach((r, i) => {
      if (!entityIds.has(r.entityType)) {
        ctx.addIssue({
          code: "custom",
          message: `rule "${r.id}" references unknown entityType "${r.entityType}"`,
          path: ["rules", i, "entityType"],
        });
      }
    });

    for (const key of defaultKeys) {
      if (!entityIds.has(key)) {
        ctx.addIssue({
          code: "custom",
          message: `actions.default references unknown entityType "${key}"`,
          path: ["actions", "default", key],
        });
      }
    }

    // A misspelled override key would otherwise fall back to the default action silently.
    for (const [provider, overrides] of Object.entries(ir.actions.providerOverrides ?? {})) {
      for (const key of Object.keys(overrides)) {
        if (!entityIds.has(key)) {
          ctx.addIssue({
            code: "custom",
            message: `actions.providerOverrides["${provider}"] references unknown entityType "${key}"`,
            path: ["actions", "providerOverrides", provider, key],
          });
        }
      }
    }

    for (const key of Object.keys(ir.provenance)) {
      if (!ruleIds.has(key) && !entityIds.has(key) && !predicateIds.has(key)) {
        ctx.addIssue({
          code: "custom",
          message: `provenance key "${key}" references no declared rule, entityType, or semanticPredicate`,
          path: ["provenance", key],
        });
      }
    }
  });

/**
 * Zero-runtime drift guard: if the schema and the hand-written types diverge, one of
 * these stops being `true` and typecheck fails.
 */
type Expect<T extends true> = T;
type _SchemaAssignableToTypes = Expect<z.infer<typeof PolicyIrSchema> extends PolicyIrInput ? true : false>;
type _TypesAssignableToSchema = Expect<PolicyIrInput extends z.infer<typeof PolicyIrSchema> ? true : false>;
