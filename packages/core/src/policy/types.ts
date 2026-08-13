export type Severity = "low" | "medium" | "high" | "critical";
export type Action = "allow" | "pseudonymize" | "redact" | "block";
export type FailMode = "open" | "closed";
export type Tier = 0 | 1 | 2;

export interface EntityType {
  id: string;
  tier: Tier;
  nlDefinition: string;
  examples: string[];
  counterExamples: string[];
  severity: Severity;
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
  /** Present for entropy rules (bits/char over sliding windows). */
  entropyThreshold?: number;
  /** Minimum candidate length for entropy rules. */
  minLength?: number;
}

export interface SemanticPredicate {
  id: string;
  nlPredicate: string;
  scope: "segment" | "message";
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

/** Input shape before zod validation (what JSON.parse gives us). */
export type PolicyIrInput = PolicyIr;
