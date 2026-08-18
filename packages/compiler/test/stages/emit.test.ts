import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { clauseLocator, emitIr, policyHash, UNMARKED_CLAUSE } from "../../src/stages/emit.js";
import { mintShadowEntityTypes } from "../../src/stages/predicates.js";

const POLICIES = join(import.meta.dirname, "..", "..", "..", "..", "policies");
const PFIN = readFileSync(join(POLICIES, "p-fin.md"), "utf8");

const DOC = [
  "# Toy standard",
  "",
  "Nothing in this preamble carries a clause marker.",
  "",
  "§1.1 Account identifiers must never be sent to an external assistant.",
  "",
  "§2.4 Client organisation names must be replaced with a stable pseudonym.",
].join("\n");

const ACCOUNT_QUOTE = "Account identifiers must never be sent to an external assistant.";
const CLIENT_QUOTE = "Client organisation names must be replaced with a stable pseudonym.";

function accountEntity() {
  return {
    id: "account-id",
    tier: 0 as const,
    nlDefinition: "An account identifier.",
    examples: ["00123456789"],
    counterExamples: [],
    severity: "critical" as const,
    surrogateKind: "id-number" as const,
    sourceQuote: ACCOUNT_QUOTE,
  };
}

function accountRule() {
  return {
    id: "account-id-format",
    entityType: "account-id",
    regex: "\\b[0-9]{11}\\b",
    sourceQuote: ACCOUNT_QUOTE,
  };
}

describe("policyHash", () => {
  it("is the sha256 of the document bytes", () => {
    // MUTATION: hashing a constant, or the policy NAME, satisfies the plan's
    // /^[0-9a-f]{64}$/ assertion while stamping every IR with the same value —
    // which is the whole point of the stamp (tying an IR to the bytes it came
    // from) removed.
    expect(policyHash(PFIN)).toBe(createHash("sha256").update(PFIN, "utf8").digest("hex"));
    expect(policyHash(PFIN)).not.toBe(policyHash(`${PFIN}\n`));
  });
});

describe("clauseLocator", () => {
  it("resolves a quote to the nearest marker at or before it", () => {
    const clauseOf = clauseLocator(DOC);
    expect(clauseOf(ACCOUNT_QUOTE)).toBe("§1.1");
    expect(clauseOf(CLIENT_QUOTE)).toBe("§2.4");
  });

  it("says unmarked rather than guessing when no marker precedes the quote", () => {
    // A guessed clause is worse than none: an auditor checking provenance would
    // read the wrong sentence and find the rule justified.
    expect(clauseLocator(DOC)("Nothing in this preamble carries a clause marker.")).toBe(
      UNMARKED_CLAUSE,
    );
    expect(clauseLocator(DOC)("a sentence that is not in this document at all")).toBe(
      UNMARKED_CLAUSE,
    );
  });

  it("matches across a line break, like the grounding gate it mirrors", () => {
    // The gate normalizes whitespace before matching, so a quote it accepted
    // must be locatable here or provenance would read "unmarked" for a rule
    // that is perfectly well grounded.
    const wrapped = "§3.2 Bank account numbers must never\nbe sent to an external assistant.";
    expect(clauseLocator(wrapped)("Bank account numbers must never be sent")).toBe("§3.2");
  });
});

describe("emitIr", () => {
  const baseInput = () => ({
    document: DOC,
    entityTypes: [accountEntity()],
    rules: [accountRule()],
    semanticPredicates: [],
    shadowEntityTypes: [],
    actions: { default: { "account-id": "block" as const }, providerOverrides: {} },
    failMode: "closed" as const,
  });

  it("keeps the model's sourceQuote out of the IR, where provenance owns it", () => {
    // MUTATION: spreading the candidate (`{...candidate}`) instead of copying
    // fields by name passes every plan test — the loader strips unknown keys
    // from nested objects — while writing a second, unversioned copy of every
    // policy quote into a committed artifact.
    const { ir } = emitIr(baseInput());
    expect(JSON.stringify(ir)).not.toContain("sourceQuote");
    expect(ir.provenance["account-id"]).toEqual({ clause: "§1.1", quote: ACCOUNT_QUOTE });
  });

  it("drops a rule whose entityType did not survive, with a reason", () => {
    // The gate rejects candidates one at a time, so a model can ground its
    // rule's quote and fail its entityType's. Emitting the rule anyway produces
    // an IR loadPolicyIr rejects; dropping it silently loses the audit trail.
    const input = {
      ...baseInput(),
      rules: [accountRule(), { ...accountRule(), id: "orphan-rule", entityType: "gone" }],
    };
    const { ir, dropped, warnings } = emitIr(input);
    expect(ir.rules.map((r) => r.id)).toEqual(["account-id-format"]);
    expect(dropped).toEqual([
      { id: "orphan-rule", kind: "rule", reason: expect.stringContaining("gone") },
    ]);
    expect(warnings.join("\n")).toContain("orphan-rule");
    expect(() => loadPolicyIr(JSON.stringify(ir))).not.toThrow();
  });

  it("drops action mappings that name an entityType which is not in the IR", () => {
    const input = {
      ...baseInput(),
      actions: {
        default: { "account-id": "block" as const, gone: "redact" as const },
        providerOverrides: { deepseek: { gone: "block" as const } },
      },
    };
    const { ir, dropped } = emitIr(input);
    expect(Object.keys(ir.actions.default)).toEqual(["account-id"]);
    // An emptied provider carries no policy, so it is not listed as if it did.
    expect(ir.actions.providerOverrides).toEqual({});
    expect(dropped.map((d) => d.kind)).toEqual(["action", "providerOverride"]);
    expect(() => loadPolicyIr(JSON.stringify(ir))).not.toThrow();
  });

  it("gives a minted shadow the provenance of the predicate that created it", () => {
    // MUTATION: skipping the shadow-provenance loop passes every plan test —
    // the schema does not require provenance for an entityType — and leaves the
    // one id in the IR that a human cannot trace to a sentence.
    const predicate = {
      id: "client-relationship",
      nlPredicate: "The message reveals that an organisation is a client.",
      scope: "message" as const,
      severity: "high" as const,
      sourceQuote: CLIENT_QUOTE,
    };
    const shadow = mintShadowEntityTypes([predicate], new Set(["account-id"]));
    const { ir } = emitIr({
      ...baseInput(),
      semanticPredicates: [predicate],
      shadowEntityTypes: shadow.entityTypes,
      actions: {
        default: { "account-id": "block" as const, ...shadow.defaultActions },
        providerOverrides: {},
      },
    });
    expect(ir.provenance["pred:client-relationship"]).toEqual({
      clause: "§2.4",
      quote: CLIENT_QUOTE,
    });
    expect(() => loadPolicyIr(JSON.stringify(ir))).not.toThrow();
  });
});
