import type { PolicyIrInput } from "@sih/core";

/**
 * A compiler-SHAPED IR: what `emit` (Task 8) would hand the self-test stage,
 * rather than the hand-minimal one core uses to exercise its loader.
 *
 * Shared with core's `test/fixtures/minimal-ir.ts` on purpose — the same four
 * entityTypes, the same rules, the same actions — so a self-test run here is
 * scored by the very rules core's own detection tests are written against. A
 * second, divergent entity table would make this stage's coverage numbers
 * unfalsifiable against the runtime they claim to measure. Two differences:
 *
 * - `policyHash` is a real sha256 rather than `"test-hash"`, matching what emit
 *   stamps (Task 8 pins `/^[0-9a-f]{64}$/`).
 * - `legacy-employee-id` is added: a tier-0 entityType with a rule that cannot
 *   match the values its own definition describes. See below.
 *
 * A fresh `structuredClone` per call because `loadPolicyIr` hands the parsed
 * object straight to detection: a test that mutated a shared literal would
 * silently rewrite the policy every later test runs under.
 */
export function minimalCompiledIr(): PolicyIrInput {
  return structuredClone(BASE);
}

const BASE: PolicyIrInput = {
  irVersion: "1",
  policyHash: "478cdc8e20010f95aba0ab3553890aee05efa4eb2f187d764788f55ec8b2536f",
  entityTypes: [
    {
      id: "in-pan",
      tier: 0,
      nlDefinition: "Indian PAN card number",
      examples: ["ABCPD1234E"],
      counterExamples: [],
      severity: "high",
      surrogateKind: "id-number",
    },
    {
      id: "aws-key",
      tier: 0,
      nlDefinition: "AWS access key ID",
      examples: ["AKIAIOSFODNN7EXAMPLE"],
      counterExamples: [],
      severity: "critical",
      neverPseudonymize: true,
    },
    {
      id: "generic-secret",
      tier: 0,
      nlDefinition: "High-entropy secret string",
      examples: [],
      counterExamples: [],
      severity: "critical",
      neverPseudonymize: true,
    },
    {
      id: "client-name",
      tier: 1,
      nlDefinition: "Name of a client organisation",
      examples: ["Globex"],
      counterExamples: [],
      severity: "high",
      surrogateKind: "org-name",
    },
    /**
     * The deliberately unmatchable one, and the reason this stage exists.
     *
     * Its rule is well-formed, safe, and passes every check the validate stage
     * applies — it simply describes a format the entity does not actually use.
     * That is the realistic compiler failure: the model reads "legacy employee
     * identifier", invents a plausible-looking pattern, and nothing in the
     * pipeline before this stage can tell the difference between a pattern that
     * matches the world and one that only matches the model's idea of it.
     *
     * It is also why `runSelfTest` never shows the model the rules when it asks
     * for cases: a generator that saw `LEGACY-EMP-[0-9]{8}` would dutifully emit
     * twenty strings matching it, report 100% recall, and prove nothing at all.
     */
    {
      id: "legacy-employee-id",
      tier: 0,
      nlDefinition:
        "Legacy employee identifier as printed on internal HR records and payslips",
      examples: [],
      counterExamples: [],
      severity: "medium",
      surrogateKind: "id-number",
    },
  ],
  rules: [
    {
      id: "pan-rule",
      entityType: "in-pan",
      regex: "\\b[A-Z]{5}[0-9]{4}[A-Z]\\b",
      validator: "pan-structure",
      contextBoost: ["PAN", "tax"],
    },
    { id: "aws-rule", entityType: "aws-key", regex: "\\bAKIA[0-9A-Z]{16}\\b" },
    { id: "entropy-rule", entityType: "generic-secret", entropyThreshold: 4.0, minLength: 20 },
    { id: "legacy-emp-rule", entityType: "legacy-employee-id", regex: "\\bLEGACY-EMP-[0-9]{8}\\b" },
  ],
  semanticPredicates: [],
  actions: {
    default: {
      "in-pan": "block",
      "aws-key": "block",
      "generic-secret": "redact",
      "client-name": "pseudonymize",
      "legacy-employee-id": "redact",
    },
    providerOverrides: { deepseek: { "client-name": "redact" } },
  },
  failMode: "closed",
  latencyBudgetMs: 5000,
  provenance: {
    "in-pan": { clause: "§2.1", quote: "PAN numbers must never be shared." },
    "aws-key": { clause: "§4.1", quote: "Cloud access keys must never leave the network." },
    "generic-secret": { clause: "§4.2", quote: "Secrets and tokens must never be pasted." },
    "client-name": { clause: "§3.1", quote: "Client names must be pseudonymized." },
    "legacy-employee-id": {
      clause: "§6.3",
      quote: "Legacy employee identifiers must be redacted before transmission.",
    },
    "pan-rule": { clause: "§2.1", quote: "PAN numbers must never be shared." },
    "aws-rule": { clause: "§4.1", quote: "Cloud access keys must never leave the network." },
    "entropy-rule": { clause: "§4.2", quote: "Secrets and tokens must never be pasted." },
    "legacy-emp-rule": {
      clause: "§6.3",
      quote: "Legacy employee identifiers must be redacted before transmission.",
    },
  },
};
