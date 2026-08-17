import type { Action, EntityType, Severity } from "@sih/core";

/**
 * Semantic predicates become SHADOW entityTypes.
 *
 * This closes the decision parked in `packages/core/src/detect/orchestrator.ts`
 * since Plan 1. A semantic predicate ("discusses unreleased financials") has no
 * entityType, and every action path in the runtime — `resolveAction`,
 * `strictestAction`, `winnerAction`, `applyActions` — is keyed by entityType. A
 * tier-2 finding therefore had no way to reach an action at all, and would have
 * defaulted to `allow`: a policy clause that silently evaporates, which is the
 * failure class this project exists to prevent.
 *
 * DECIDED (Plan 3): the compiler mints one tier-2 entityType per predicate,
 * `pred:<predicateId>`, and a default action for it. Predicate findings then
 * flow through the entityType → action path that already exists, and no runtime
 * code changes: resolution, merge, and apply cannot tell a shadow from an
 * authored entityType, which is the point. Tier-2 engines (Plan 5) emit findings
 * whose `entityType` is the shadow id.
 *
 * REJECTED: growing `Finding` a `predicateId` field that resolution consults.
 * Same observable outcome, but it forks every stage the finding passes through —
 * `normalizeFindings` (which validates `entityType` against the IR and would
 * need a second lookup table), `mergeFindings`/`clusterOverlapping` (severity
 * comparison across two kinds of finding), `resolveAction` (a second key space,
 * including provider overrides), and `applyActions` (`neverPseudonymize` has no
 * predicate-side equivalent) — to arrive where minting an id arrives for free.
 * A second key space through four stages is four places to forget the predicate
 * case; the shadow entityType is zero.
 *
 * The cost, recorded honestly: `pred:` ids are outbound-visible like any other
 * entityType id, since they ship inside `[REDACTED:<id>]`. They come from the
 * extraction model, whose SYSTEM prompt already forbids deriving ids from
 * confidential nouns, and Task 6's id hygiene check sees them the same as any
 * authored id.
 */

/**
 * Namespace for minted ids. A prefix rather than a suffix so a shadow is
 * recognizable at a glance in a redaction marker and in the IR, and `:` because
 * the extraction prompt constrains authored ids to lowercase kebab-case — an
 * authored id cannot contain a colon, so the two spaces cannot overlap even
 * before the explicit collision check below.
 */
export const SHADOW_PREFIX = "pred:";

export function shadowIdFor(predicateId: string): string {
  return `${SHADOW_PREFIX}${predicateId}`;
}

/** A predicate as the extract stage produces it; `sourceQuote` is not read here. */
export interface PredicateInput {
  readonly id: string;
  readonly nlPredicate: string;
  readonly scope: "segment" | "message";
  readonly severity: Severity;
  readonly sourceQuote?: string | undefined;
}

export interface ShadowEntityTypes {
  /** One per predicate, in input order. Typed as core's `EntityType` so a drift
   * in the IR schema breaks this file rather than the emitted IR. */
  readonly entityTypes: readonly EntityType[];
  /** Shadow id → default action, to be merged into `ir.actions.default`. */
  readonly defaultActions: Record<string, Action>;
}

/**
 * Why `redact` and never `pseudonymize`: pseudonymization mints a
 * format-preserving surrogate keyed to a real surface value, and rehydration
 * maps that surrogate back to the value it replaced (Plan 2's vault). A semantic
 * finding has no such value — it is a span the model judged to be *about*
 * something, so there is no stable surface form to key a surrogate to and
 * nothing coherent to rehydrate into. `neverPseudonymize` is set as well as the
 * default action being `redact`, so that a provider override or a cluster's
 * strictest action cannot hand a shadow a `pseudonymize` stamp later:
 * `winnerAction` escalates that combination back to `redact`, and the IR schema
 * rejects an authored pseudonymize action for these outright.
 */
const SHADOW_ACTION: Action = "redact";

/** Own-key discipline: predicate ids are model-authored strings. */
function emptyMap<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

/**
 * Mints a shadow entityType and a default action for each predicate.
 *
 * `existingIds` is every entityType id already in the IR. A collision is thrown,
 * not renamed: a silently renamed shadow would leave the tier-2 engine emitting
 * findings under an id the IR does not contain, and a silently merged one would
 * give a predicate an unrelated entityType's action. Both are policy changes
 * nobody asked for; a compile error is visible.
 */
export function mintShadowEntityTypes(
  predicates: readonly PredicateInput[],
  existingIds: ReadonlySet<string> = new Set(),
): ShadowEntityTypes {
  const entityTypes: EntityType[] = [];
  const defaultActions = emptyMap<Action>();
  const minted = new Set<string>();

  for (const predicate of predicates) {
    const id = shadowIdFor(predicate.id);
    if (existingIds.has(id)) {
      throw new Error(`shadow entityType "${id}" collides with an existing entityType id`);
    }
    // Two predicates sharing an id would mint the same shadow twice, which the
    // IR schema rejects as a duplicate entityType far from its cause.
    if (minted.has(id)) {
      throw new Error(`shadow entityType "${id}" collides with another minted shadow (duplicate predicate id)`);
    }
    minted.add(id);

    entityTypes.push({
      id,
      // Tier 2 by construction: a predicate exists precisely because the class
      // has no surface form for tier 0 or tier 1 to match.
      tier: 2,
      // Verbatim, not decorated: this text is the tier-2 engine's prompt and the
      // line a human reads in the IR, and rewording it here would put words the
      // policy never said into both.
      nlDefinition: predicate.nlPredicate,
      examples: [],
      counterExamples: [],
      severity: predicate.severity,
      neverPseudonymize: true,
    });
    defaultActions[id] = SHADOW_ACTION;
  }

  return { entityTypes, defaultActions };
}
