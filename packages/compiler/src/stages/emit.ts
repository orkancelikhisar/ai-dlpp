import { createHash } from "node:crypto";
import type {
  Action,
  EntityType,
  FailMode,
  PolicyIr,
  Provenance,
  Rule,
  SemanticPredicate,
} from "@sih/core";
import { normalizeForQuoteMatch, type RejectedCandidate } from "./extract.js";
import { SHADOW_PREFIX } from "./predicates.js";

/**
 * Stage 5: assemble the artifact.
 *
 * Everything upstream produced fragments that were each checked on their own
 * terms — quotes grounded, providers resolved, regexes raced, predicates
 * shadowed. This stage is where they become one object, and its whole job is to
 * make that object TRACEABLE and CONSISTENT:
 *
 * - Traceable: every rule, entityType and predicate gets a `provenance` entry
 *   carrying the clause it came from and the verbatim quote that justified it.
 *   An IR nobody can trace back to policy text is an IR nobody can audit, and an
 *   unauditable security artifact is one that gets trusted for the wrong reasons.
 * - Consistent: a rule or an action whose entityType did not survive the
 *   grounding gate is dropped HERE, with a reason, rather than shipped into an
 *   IR the runtime loader would reject. See `reconcile` for why that is a drop
 *   and not a throw.
 *
 * Deliberately NOT this stage's job: deciding whether the result is shippable.
 * `compilePolicy` owns that (an IR with no entityTypes, or one whose entityType
 * has no action, is unusable rather than merely thin), so this file stays a pure
 * assembler that can be unit-tested on inputs a real compile would reject.
 */

/**
 * Spec §5.3's budget for a full detection pass. A default rather than a
 * per-policy knob because no clause in an authored policy talks about
 * milliseconds — the number is a property of the runtime, not of the document.
 */
export const DEFAULT_LATENCY_BUDGET_MS = 5000;

/** Clause recorded when a quote sits before any `§` marker in the document. */
export const UNMARKED_CLAUSE = "unmarked";

/**
 * `§2`, `§2.1`, `§2.1.3` — the marker shapes the authored suite uses. Matched
 * against the NORMALIZED document (see `clauseLocator`), so a marker split
 * across a line break is still one token.
 */
const CLAUSE_MARKER = /§[0-9]+(?:\.[0-9]+)*/g;

/**
 * sha256 of the source document, hex. Stamped into the IR so a shipped artifact
 * can be tied to the exact bytes it was compiled from: an IR whose policy has
 * since been edited is not a stale IR you can eyeball, it is one that quietly
 * enforces a superseded standard.
 */
export function policyHash(document: string): string {
  return createHash("sha256").update(document, "utf8").digest("hex");
}

/**
 * Builds a `quote -> clause` locator over one document.
 *
 * Works entirely in NORMALIZED space (`normalizeForQuoteMatch`, the same
 * whitespace-insensitive form the grounding gate matched in) so that a quote the
 * gate accepted is a quote this can locate. Marker offsets are collected once
 * from the same normalized string, so the two coordinate systems agree.
 *
 * "Nearest marker at or before the quote" and not "the marker inside the quote":
 * a model quotes the sentence, and the sentence follows its marker. A marker
 * sitting exactly at the quote's start still counts (the model quoted the whole
 * clause line, marker included), which is why the comparison is `<=`.
 */
export function clauseLocator(document: string): (quote: string) => string {
  const haystack = normalizeForQuoteMatch(document);
  const markers: Array<{ index: number; marker: string }> = [];
  for (const match of haystack.matchAll(CLAUSE_MARKER)) {
    markers.push({ index: match.index, marker: match[0] });
  }

  return (quote: string): string => {
    const needle = normalizeForQuoteMatch(quote);
    // An ungrounded quote has no clause. Reachable only when this stage is
    // called with candidates that never passed the gate (a unit test, or a
    // future caller); `UNMARKED_CLAUSE` says "no marker" and never guesses one.
    if (needle.length === 0) return UNMARKED_CLAUSE;
    const at = haystack.indexOf(needle);
    if (at < 0) return UNMARKED_CLAUSE;

    let clause = UNMARKED_CLAUSE;
    for (const marker of markers) {
      if (marker.index > at) break;
      clause = marker.marker;
    }
    return clause;
  };
}

// -- inputs -----------------------------------------------------------------

/** Every grounded candidate carries the quote that grounded it. */
interface Quoted {
  readonly sourceQuote: string;
}

export type EntityTypeCandidate = EntityType & Quoted;
export type RuleCandidate = Rule & Quoted;
export type PredicateCandidate = SemanticPredicate & Quoted;

export interface EmitInput {
  /** The exact bytes compiled: hashed, and searched for clause markers. */
  readonly document: string;
  /** Authored entityTypes that survived the grounding gate, in model order. */
  readonly entityTypes: readonly EntityTypeCandidate[];
  readonly rules: readonly RuleCandidate[];
  readonly semanticPredicates: readonly PredicateCandidate[];
  /** Minted by `mintShadowEntityTypes`; provenance is inherited from the predicate. */
  readonly shadowEntityTypes: readonly EntityType[];
  readonly actions: {
    readonly default: Record<string, Action>;
    readonly providerOverrides: Record<string, Record<string, Action>>;
  };
  readonly failMode: FailMode;
  readonly latencyBudgetMs?: number;
}

export interface EmitResult {
  readonly ir: PolicyIr;
  /** Items dropped for referencing an entityType that did not survive. */
  readonly dropped: RejectedCandidate[];
  readonly warnings: string[];
}

/** Own-key discipline: every key here is a model-authored id. */
function emptyMap<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

// -- assembly ---------------------------------------------------------------

/**
 * Fields are copied out BY NAME rather than spread, for two reasons that both
 * matter to a committed artifact: the model's `sourceQuote` must not ride along
 * inside the IR (provenance is where a quote belongs, and one copy of a policy
 * quote is enough), and naming the fields fixes the key order of the emitted
 * JSON so that recompiling an unchanged policy produces an unchanged file.
 */
function toEntityType(candidate: EntityType): EntityType {
  return {
    id: candidate.id,
    tier: candidate.tier,
    nlDefinition: candidate.nlDefinition,
    examples: [...candidate.examples],
    counterExamples: [...candidate.counterExamples],
    severity: candidate.severity,
    ...(candidate.surrogateKind === undefined ? {} : { surrogateKind: candidate.surrogateKind }),
    ...(candidate.neverPseudonymize === undefined
      ? {}
      : { neverPseudonymize: candidate.neverPseudonymize }),
  };
}

function toRule(candidate: RuleCandidate): Rule {
  return {
    id: candidate.id,
    entityType: candidate.entityType,
    ...(candidate.regex === undefined ? {} : { regex: candidate.regex }),
    ...(candidate.validator === undefined ? {} : { validator: candidate.validator }),
    ...(candidate.contextBoost === undefined ? {} : { contextBoost: [...candidate.contextBoost] }),
    ...(candidate.entropyThreshold === undefined
      ? {}
      : { entropyThreshold: candidate.entropyThreshold }),
    ...(candidate.minLength === undefined ? {} : { minLength: candidate.minLength }),
  };
}

function toPredicate(candidate: PredicateCandidate): SemanticPredicate {
  // `severity` is deliberately absent: the extract stage carries it so that
  // `mintShadowEntityTypes` can put it on the shadow entityType, which is where
  // the runtime reads severity from. A second copy on the predicate would be a
  // field nothing reads and a value that could disagree with the shadow's.
  return { id: candidate.id, nlPredicate: candidate.nlPredicate, scope: candidate.scope };
}

/**
 * Drops rules and action mappings that name an entityType which is not in the
 * IR, and says so.
 *
 * This is a real consequence of the grounding gate rejecting candidates one at a
 * time: a model can ground its rule's quote and fail its entityType's, leaving a
 * rule pointing at nothing. `loadPolicyIr` rejects that IR — correctly, but with
 * an error anchored at an array index and no mention of the quote that started
 * it. Dropping here keeps the failure legible and keeps the rest of the policy
 * shippable.
 *
 * A DROP and not a throw, because the alternative is worse in both directions:
 * throwing would discard an otherwise good compile over one ungrounded quote,
 * and keeping the rule would emit an IR the runtime refuses to load. What must
 * never happen is dropping it silently — every drop lands in `dropped`, which
 * the report prints beside the anti-hallucination gate's own rejections.
 */
function reconcile(
  input: EmitInput,
  entityIds: ReadonlySet<string>,
): {
  rules: RuleCandidate[];
  defaults: Record<string, Action>;
  providerOverrides: Record<string, Record<string, Action>>;
  dropped: RejectedCandidate[];
  warnings: string[];
} {
  const dropped: RejectedCandidate[] = [];
  const warnings: string[] = [];

  const rules = input.rules.filter((rule) => {
    if (entityIds.has(rule.entityType)) return true;
    dropped.push({
      id: rule.id,
      kind: "rule",
      reason: `references entityType "${rule.entityType}", which is not in the compiled IR`,
    });
    warnings.push(
      `rule "${rule.id}" was dropped: its entityType "${rule.entityType}" is not in the compiled ` +
        `IR, so the rule could never resolve an action`,
    );
    return false;
  });

  const defaults = emptyMap<Action>();
  for (const entityId of Object.keys(input.actions.default)) {
    if (!entityIds.has(entityId)) {
      dropped.push({
        id: entityId,
        kind: "action",
        reason: `default action names entityType "${entityId}", which is not in the compiled IR`,
      });
      warnings.push(
        `default action for entityType "${entityId}" was dropped: that entityType is not in the ` +
          `compiled IR`,
      );
      continue;
    }
    defaults[entityId] = input.actions.default[entityId]!;
  }

  const providerOverrides = emptyMap<Record<string, Action>>();
  for (const providerId of Object.keys(input.actions.providerOverrides)) {
    const overrides = input.actions.providerOverrides[providerId]!;
    const kept = emptyMap<Action>();
    let keptAny = false;
    for (const entityId of Object.keys(overrides)) {
      if (!entityIds.has(entityId)) {
        dropped.push({
          id: `${providerId}:${entityId}`,
          kind: "providerOverride",
          reason: `override names entityType "${entityId}", which is not in the compiled IR`,
        });
        warnings.push(
          `provider override "${providerId}" → "${entityId}" was dropped: that entityType is not ` +
            `in the compiled IR`,
        );
        continue;
      }
      kept[entityId] = overrides[entityId]!;
      keptAny = true;
    }
    // An empty per-provider map is legal in the schema but says nothing; keeping
    // it would show a provider in the report as if it carried policy.
    if (keptAny) providerOverrides[providerId] = kept;
  }

  return { rules, defaults, providerOverrides, dropped, warnings };
}

/**
 * Assemble the IR.
 *
 * Order is stable and meaningful: authored entityTypes in the order the model
 * emitted them, then minted shadows in predicate order. Nothing is sorted —
 * sorting would also be deterministic, but it would discard the document's own
 * ordering, which is the order a human auditor reads the report in.
 */
export function emitIr(input: EmitInput): EmitResult {
  const entityTypes = [
    ...input.entityTypes.map(toEntityType),
    ...input.shadowEntityTypes.map(toEntityType),
  ];
  const entityIds = new Set(entityTypes.map((e) => e.id));
  const { rules, defaults, providerOverrides, dropped, warnings } = reconcile(input, entityIds);

  const clauseOf = clauseLocator(input.document);
  const provenance = emptyMap<Provenance>();
  const record = (id: string, quote: string): void => {
    // First writer wins. Two candidates sharing an id is a duplicate the IR
    // schema rejects by id, so overwriting here would only change which quote a
    // rejected IR carried.
    if (Object.hasOwn(provenance, id)) return;
    provenance[id] = { clause: clauseOf(quote), quote };
  };

  for (const entity of input.entityTypes) record(entity.id, entity.sourceQuote);
  for (const rule of rules) record(rule.id, rule.sourceQuote);
  for (const predicate of input.semanticPredicates) record(predicate.id, predicate.sourceQuote);
  // A shadow has no quote of its own — it exists because a predicate did. It
  // inherits that predicate's clause and quote so the audit trail for
  // `pred:<id>` lands on the sentence that created it rather than on nothing.
  for (const shadow of input.shadowEntityTypes) {
    const predicateId = shadow.id.startsWith(SHADOW_PREFIX)
      ? shadow.id.slice(SHADOW_PREFIX.length)
      : shadow.id;
    if (Object.hasOwn(provenance, predicateId)) {
      record(shadow.id, provenance[predicateId]!.quote);
    }
  }

  const ir: PolicyIr = {
    irVersion: "1",
    policyHash: policyHash(input.document),
    entityTypes,
    rules: rules.map(toRule),
    semanticPredicates: input.semanticPredicates.map(toPredicate),
    actions: { default: defaults, providerOverrides },
    failMode: input.failMode,
    latencyBudgetMs: input.latencyBudgetMs ?? DEFAULT_LATENCY_BUDGET_MS,
    provenance,
  };

  return { ir, dropped, warnings };
}
