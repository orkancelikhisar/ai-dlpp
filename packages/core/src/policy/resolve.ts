import type { Action, PolicyIr } from "./types.js";

/**
 * Own-key read. Both action maps are JSON-parsed plain objects inheriting from
 * Object.prototype, and nothing constrains entityType ids — or provider ids, which are
 * validated against nothing at all — to avoid names like "toString", "constructor" or
 * "name". A bare index read answers those lookups from the prototype chain, yielding a
 * function or a stray string (`providerOverrides["toString"].name` is "toString") that
 * is neither an Action nor undefined: it slips past a miss check and is returned as if
 * it were policy.
 */
function own<T>(map: Record<string, T>, key: string): T | undefined {
  return Object.hasOwn(map, key) ? map[key] : undefined;
}

/**
 * Resolves the effective action for one entityType against one provider: a two-level
 * merge of `actions.providerOverrides[provider][entityType]` over
 * `actions.default[entityType]`. Pure, and the single point where a wrong answer is a
 * silent leak.
 */
export function resolveAction(ir: PolicyIr, entityTypeId: string, providerId: string): Action {
  const base = own(ir.actions.default, entityTypeId);
  // The schema guarantees every declared entityType has a default action, so a miss
  // means the caller invented an id — a programmer error, not a policy state.
  if (base === undefined) {
    throw new Error(`unknown entityType "${entityTypeId}" — IR validation should have prevented this`);
  }

  const byProvider = ir.actions.providerOverrides;
  const forProvider = byProvider === undefined ? undefined : own(byProvider, providerId);
  const override = forProvider === undefined ? undefined : own(forProvider, entityTypeId);
  return override ?? base;
}
