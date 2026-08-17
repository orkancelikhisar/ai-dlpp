import { z } from "zod";

/**
 * Stage 2: resolve the natural-language provider mentions the extract stage
 * lifted out of the policy ("Chinese-hosted services", "our enterprise Claude
 * agreement") to the adapter ids the runtime actually routes to.
 *
 * The governing rule of this stage: **an unresolvable mention is a warning that
 * keeps the default action, never a silent drop.** A provider clause that
 * vanishes between the document and the IR is a policy downgrade nobody sees,
 * which is the failure class this project exists to prevent.
 */

export type PolicyAction = "allow" | "pseudonymize" | "redact" | "block";

/**
 * How restrictive each action is. Used to settle a collision between two
 * clauses that name the same (provider, entityType): the stricter one wins.
 * Over-blocking is recoverable by reading the report; under-blocking is a leak.
 */
const RESTRICTIVENESS: Record<PolicyAction, number> = {
  allow: 0,
  pseudonymize: 1,
  redact: 2,
  block: 3,
};

const ProviderSchema = z.object({
  id: z.string().min(1),
  displayName: z.string().min(1),
  vendor: z.string().min(1),
  hostingRegion: z.string().min(1),
  enterpriseAgreement: z.boolean(),
  aliases: z.array(z.string().min(1)).min(1),
});

/**
 * A category's membership test, expressed over the attributes the manifest
 * actually records. Every key present must hold (AND).
 *
 * Deliberately tiny and closed: a category exists so that a blanket policy
 * clause ("non-enterprise services") lands on every provider it covers, and a
 * predicate language richer than the manifest's own fields would be the
 * compiler inventing facts about vendors rather than reading them.
 */
const CategoryMatchSchema = z
  .object({
    enterpriseAgreement: z.boolean().optional(),
    hostingRegionIn: z.array(z.string().min(1)).min(1).optional(),
    hostingRegionNotIn: z.array(z.string().min(1)).min(1).optional(),
  })
  .refine((match) => Object.values(match).some((value) => value !== undefined), {
    message:
      "category match must constrain at least one attribute (an empty match selects every provider)",
  });

/**
 * A class of providers named by attribute rather than by vendor.
 *
 * This exists because of a real bug found reviewing the manifest: P-FIN §5.1
 * ("No customer data ... may be sent to non-enterprise or foreign-hosted
 * services") is the firm's blanket rule, but "foreign-hosted" was sitting in
 * DeepSeek's alias list. Under plain alias matching the broadest clause in the
 * policy resolved to one vendor, and ChatGPT and Gemini — which §5.4 and §5.5
 * also place outside the firm's agreements — silently kept the permissive
 * default. A category term is not a provider name and must not be stored as one.
 */
const CategorySchema = z.object({
  id: z.string().min(1),
  /** Why this predicate is the right reading of the term. Read by humans. */
  description: z.string().min(1),
  aliases: z.array(z.string().min(1)).min(1),
  match: CategoryMatchSchema,
});

export const ProviderManifestSchema = z.object({
  manifestVersion: z.literal("1"),
  providers: z.array(ProviderSchema).min(1),
  categories: z.array(CategorySchema).default([]),
});

export type Provider = z.infer<typeof ProviderSchema>;
export type ProviderCategory = z.infer<typeof CategorySchema>;

/** One searchable term, lowercased, with the adapter ids it stands for. */
export interface AliasEntry {
  readonly alias: string;
  readonly via: "provider" | "category";
  /** Provider id or category id — whichever declared this alias. */
  readonly source: string;
  /** Empty only for a category no provider currently satisfies. */
  readonly providerIds: readonly string[];
}

export interface ProviderManifest {
  readonly manifestVersion: string;
  readonly providers: readonly Provider[];
  readonly categories: readonly ProviderCategory[];
  /** Every alias in one namespace, longest first (see resolveProviderMention). */
  readonly aliases: readonly AliasEntry[];
}

function normalizeTerm(term: string): string {
  return term.trim().toLowerCase();
}

function satisfiesCategory(category: ProviderCategory, provider: Provider): boolean {
  const { enterpriseAgreement, hostingRegionIn, hostingRegionNotIn } = category.match;
  if (enterpriseAgreement !== undefined && provider.enterpriseAgreement !== enterpriseAgreement) {
    return false;
  }
  const region = normalizeTerm(provider.hostingRegion);
  if (hostingRegionIn !== undefined && !hostingRegionIn.some((r) => normalizeTerm(r) === region)) {
    return false;
  }
  if (hostingRegionNotIn !== undefined && hostingRegionNotIn.some((r) => normalizeTerm(r) === region)) {
    return false;
  }
  return true;
}

/**
 * Validates a raw manifest and builds its alias index.
 *
 * Rejects duplicate ids and duplicate aliases **across providers and
 * categories alike**: an alias two entries claim resolves to whichever one the
 * loader happened to index, so an ambiguous manifest would make the compiled
 * IR depend on file ordering rather than on the policy.
 */
export function loadManifest(raw: unknown): ProviderManifest {
  const parsed = ProviderManifestSchema.safeParse(raw);
  if (!parsed.success) {
    throw new Error(`invalid provider manifest:\n${z.prettifyError(parsed.error)}`);
  }
  const { manifestVersion, providers, categories } = parsed.data;

  // Map, not object literal: ids and aliases are authored strings and could be
  // "toString" or "constructor".
  const idOwners = new Map<string, string>();
  for (const provider of providers) {
    if (idOwners.has(provider.id)) {
      throw new Error(`duplicate provider id "${provider.id}"`);
    }
    idOwners.set(provider.id, `provider "${provider.id}"`);
  }
  for (const category of categories) {
    const existing = idOwners.get(category.id);
    if (existing !== undefined) {
      throw new Error(`category id "${category.id}" collides with ${existing}`);
    }
    idOwners.set(category.id, `category "${category.id}"`);
  }

  const aliasOwners = new Map<string, string>();
  const aliases: AliasEntry[] = [];
  const claim = (
    rawAlias: string,
    via: AliasEntry["via"],
    source: string,
    providerIds: readonly string[],
  ): void => {
    const alias = normalizeTerm(rawAlias);
    const label = `${via} "${source}"`;
    if (alias.length === 0) {
      throw new Error(`${label} declares a blank alias`);
    }
    const existing = aliasOwners.get(alias);
    if (existing !== undefined) {
      throw new Error(
        existing === label
          ? `${label} lists alias "${alias}" twice`
          : `duplicate alias "${alias}": claimed by both ${existing} and ${label}`,
      );
    }
    aliasOwners.set(alias, label);
    aliases.push({ alias, via, source, providerIds });
  };

  for (const provider of providers) {
    for (const alias of provider.aliases) {
      claim(alias, "provider", provider.id, [provider.id]);
    }
  }
  for (const category of categories) {
    // Expanded once, at load time, against the manifest as it stands. Every
    // adapter the runtime can reach is in this manifest, so the expansion is
    // complete with respect to reachable destinations.
    const members = providers.filter((p) => satisfiesCategory(category, p)).map((p) => p.id);
    for (const alias of category.aliases) {
      claim(alias, "category", category.id, members);
    }
  }

  // Longest first, ties alphabetically: resolveProviderMention takes the first
  // matching alias per provider, so this ordering *is* the longest-wins rule.
  aliases.sort((a, b) => b.alias.length - a.alias.length || a.alias.localeCompare(b.alias));

  return { manifestVersion, providers, categories, aliases };
}

export interface MentionMatch {
  readonly providerId: string;
  /** The longest alias of this provider found in the mention. */
  readonly alias: string;
  readonly via: AliasEntry["via"];
  readonly source: string;
}

export interface MentionResolution {
  /** One entry per provider, in manifest order. Empty means unresolvable. */
  readonly matches: readonly MentionMatch[];
  /** Aliases that appeared but expanded to no provider (an empty category). */
  readonly emptyMatches: readonly string[];
}

/**
 * Matches a mention against the alias index: lowercase, substring, longest
 * alias per provider, at most one match per provider.
 *
 * Longest-wins is scoped **per provider**, not globally. A global winner would
 * let one long alias suppress a different provider matched by a shorter one —
 * dropping a provider from a clause that named it, which is the expensive
 * direction of this trade. Union across providers, dedupe within one.
 *
 * The winning alias is returned rather than discarded: it is the audit trail
 * for why a clause landed on this adapter ("§5.1 → non-enterprise → gemini").
 */
export function resolveProviderMention(
  manifest: ProviderManifest,
  mention: string,
): MentionResolution {
  const haystack = mention.toLowerCase();
  const claimed = new Map<string, MentionMatch>();
  const emptyMatches: string[] = [];

  for (const entry of manifest.aliases) {
    if (!haystack.includes(entry.alias)) continue;
    if (entry.providerIds.length === 0) {
      emptyMatches.push(entry.alias);
      continue;
    }
    for (const providerId of entry.providerIds) {
      if (claimed.has(providerId)) continue;
      claimed.set(providerId, { providerId, alias: entry.alias, via: entry.via, source: entry.source });
    }
  }

  const matches = manifest.providers.flatMap((provider) => {
    const match = claimed.get(provider.id);
    return match === undefined ? [] : [match];
  });
  return { matches, emptyMatches };
}

/**
 * An action as the extract stage produces it. `sourceQuote` and the synthesized
 * `id` are carried for rejection reporting and are deliberately ignored here.
 */
export interface ProviderActionInput {
  readonly entityType: string;
  readonly action: PolicyAction;
  readonly providerMention?: string | undefined;
  readonly sourceQuote?: string | undefined;
  readonly id?: string | undefined;
}

export interface GroundedActions {
  /** entityType → action, for actions that named no provider. */
  readonly defaults: Record<string, PolicyAction>;
  /** provider id → entityType → action. */
  readonly providerOverrides: Record<string, Record<string, PolicyAction>>;
  readonly warnings: string[];
}

/** Own-key discipline: entityType and provider ids are authored strings. */
function emptyMap<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

function mergeAction(
  target: Record<string, PolicyAction>,
  key: string,
  next: PolicyAction,
  scope: string,
  warnings: string[],
): void {
  if (!Object.hasOwn(target, key)) {
    target[key] = next;
    return;
  }
  const current = target[key]!;
  if (current === next) return;
  // Order-independent by construction: max() over the pair, not last-write-wins,
  // so the IR does not depend on the order the model happened to emit clauses in.
  const winner = RESTRICTIVENESS[next] > RESTRICTIVENESS[current] ? next : current;
  warnings.push(
    `conflicting actions for ${scope}: "${current}" and "${next}"; kept the more restrictive "${winner}"`,
  );
  target[key] = winner;
}

export function groundProviders(
  manifest: ProviderManifest,
  actions: readonly ProviderActionInput[],
): GroundedActions {
  const defaults = emptyMap<PolicyAction>();
  const providerOverrides = emptyMap<Record<string, PolicyAction>>();
  const warnings: string[] = [];

  for (const action of actions) {
    const entity = action.entityType;
    const mention = action.providerMention;

    if (mention === undefined) {
      mergeAction(defaults, entity, action.action, `entityType "${entity}"`, warnings);
      continue;
    }

    if (mention.trim().length === 0) {
      // Not folded into `defaults`: a blank mention there would promote a
      // provider-scoped "allow" into a firm-wide one.
      warnings.push(
        `blank provider mention on the "${action.action}" action for entityType "${entity}"; ` +
          `no override was emitted and the default for "${entity}" stands`,
      );
      continue;
    }

    const { matches, emptyMatches } = resolveProviderMention(manifest, mention);
    if (matches.length === 0) {
      // The mention is echoed verbatim (it is a provider name from the policy,
      // not a confidential value); the sourceQuote never is.
      const cause =
        emptyMatches.length > 0
          ? `matched ${emptyMatches.map((a) => `category "${a}"`).join(", ")}, which no provider in the manifest satisfies`
          : "matched no provider or category in the manifest";
      warnings.push(
        `could not resolve provider mention "${mention}" (entityType "${entity}", action "${action.action}"): ` +
          `${cause}; no override was emitted and the default for "${entity}" stands`,
      );
      continue;
    }

    for (const match of matches) {
      if (!Object.hasOwn(providerOverrides, match.providerId)) {
        providerOverrides[match.providerId] = emptyMap<PolicyAction>();
      }
      mergeAction(
        providerOverrides[match.providerId]!,
        entity,
        action.action,
        `provider "${match.providerId}", entityType "${entity}"`,
        warnings,
      );
    }
  }

  return { defaults, providerOverrides, warnings };
}
