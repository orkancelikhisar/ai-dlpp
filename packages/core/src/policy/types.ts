export type Severity = "low" | "medium" | "high" | "critical";
export type Action = "allow" | "pseudonymize" | "redact" | "block";
export type FailMode = "open" | "closed";
export type Tier = 0 | 1 | 2;

/** Which format-preserving generator mints surrogates for this entityType (Plan 2). */
export type SurrogateKind = "person-name" | "org-name" | "id-number" | "opaque";

export interface EntityType {
  id: string;
  tier: Tier;
  nlDefinition: string;
  examples: string[];
  counterExamples: string[];
  severity: Severity;
  /** Generator used when the resolved action is "pseudonymize". Missing → "opaque". */
  surrogateKind?: SurrogateKind;
  /**
   * Credentials-class marker (spec §5.4): a format-valid fake credential is a lie
   * waiting to be pasted somewhere. Schema rejects pseudonymize actions for these,
   * and the vault refuses to mint them — defense in depth.
   */
  neverPseudonymize?: boolean;
}

export interface Rule {
  id: string;
  entityType: string;
  /** Regex source (no flags); present for pattern rules. */
  regex?: string;
  /** Name of a validator in the fixed library; never generated code. */
  validator?: string;
  /** Nearby keywords that raise confidence. */
  contextBoost?: string[];
  /**
   * Present for entropy rules: minimum bits/char, scored over each maximal run
   * of secret-alphabet characters (not a sliding window). Capped by the schema
   * at the alphabet's own maximum entropy, since a higher threshold can never
   * fire.
   */
  entropyThreshold?: number;
  /** Minimum candidate-run length for entropy rules. */
  minLength?: number;
}

/**
 * What a semantic predicate is asked about: one segment at a time, or the whole
 * message at once.
 *
 * Named rather than written inline in both places that need it, because the two
 * places are a QUESTION and an ANSWER -- the policy declares a scope here, and
 * `JudgeVerdict.scopesJudged` reports which scopes a judge evaluated. Two
 * independent spellings of the same union would let the answer stop covering
 * the question without a compile error.
 */
export type PredicateScope = "segment" | "message";

export interface SemanticPredicate {
  id: string;
  nlPredicate: string;
  scope: PredicateScope;
}

export interface Actions {
  default: Record<string, Action>;
  providerOverrides?: Record<string, Record<string, Action>>;
}

export interface Provenance {
  clause: string;
  quote: string;
}

export interface PolicyIr {
  irVersion: "1";
  policyHash: string;
  entityTypes: EntityType[];
  rules: Rule[];
  semanticPredicates: SemanticPredicate[];
  actions: Actions;
  failMode: FailMode;
  latencyBudgetMs: number;
  provenance: Record<string, Provenance>;
}

/**
 * Input shape before zod validation (what JSON.parse gives us): identical to
 * `PolicyIr` except that `irVersion` is any string — JSON cannot guarantee the "1"
 * literal, and the loader is what narrows it.
 */
export type PolicyIrInput = Omit<PolicyIr, "irVersion"> & { irVersion: string };
