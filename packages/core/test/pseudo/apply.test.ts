import { describe, expect, it } from "vitest";
import type { ResolvedFinding } from "../../src/detect/types.js";
import { detect } from "../../src/detect/orchestrator.js";
import type { Action } from "../../src/policy/types.js";
import { loadPolicyIr } from "../../src/policy/load.js";
import { applyActions } from "../../src/pseudo/apply.js";
import { MemoryVaultStore, Vault } from "../../src/pseudo/vault.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const config = { tier0: true, tier1: false, tier2: false };

/** Any fixed string; the vault only requires it to be non-empty (Task 4, D1). */
const SALT = "test-salt";
const newVault = () => new Vault(new MemoryVaultStore(), SALT);

const TEXT = "PAN ABCPD1234E and key AKIAIOSFODNN7EXAMPLE here";

/**
 * Hand-built resolved finding. `text` is re-derived from the source string
 * rather than passed in, which is the Finding contract (offsets are the truth)
 * and keeps these fixtures from drifting when a literal is edited.
 */
function findingAt(
  source: string,
  start: number,
  end: number,
  action: Action,
  entityType = "aws-key",
): ResolvedFinding {
  return {
    start,
    end,
    text: source.slice(start, end),
    entityType,
    severity: "critical",
    tier: 0,
    source: "hand-built",
    confidence: 1,
    action,
  };
}

describe("applyActions", () => {
  it("pseudonymizes and redacts by span, leaves allow untouched", async () => {
    const vault = newVault();
    // `client-name` is a tier-1 entity and tier 1 is off here, so the allow
    // finding is appended by hand -- which is also the only way to exercise the
    // branch: the fixture maps client-name to pseudonymize.
    const text = `${TEXT} for Globex`;
    const clientStart = text.indexOf("Globex");
    const { findings } = await detect({ ir, provider: "chatgpt", text, config });
    // The fixture's actions are block/block for the detected two, so the mix
    // this test is about is crafted here rather than resolved from policy.
    const manual: ResolvedFinding[] = [
      ...findings.map((f) =>
        f.entityType === "in-pan" ? { ...f, action: "pseudonymize" as const } : { ...f, action: "redact" as const },
      ),
      findingAt(text, clientStart, clientStart + "Globex".length, "allow", "client-name"),
    ];
    const result = await applyActions(text, manual, vault, "conv1", ir);
    expect(result.blocked).toBe(false);
    expect(result.text).not.toContain("ABCPD1234E");
    expect(result.text).not.toContain("AKIAIOSFODNN7EXAMPLE");
    expect(result.text).toContain("[REDACTED:aws-key]");
    // The surrogate is format-preserving, so the PAN shape survives.
    expect(result.text).toMatch(/PAN [A-Z]{5}[0-9]{4}[A-Z] and key/);
    // Allow is a rewrite of nothing: the span survives verbatim, and it must not
    // appear in `applied` either -- a caller diffing that list would otherwise
    // report a replacement that never happened (and one whose newStart/newEnd
    // would be meaningless).
    expect(result.text.endsWith(" for Globex")).toBe(true);
    expect(result.applied.map((a) => a.entityType)).not.toContain("client-name");
    expect(result.applied).toHaveLength(2);
  });

  it("applied records slice the rewritten text exactly", async () => {
    const vault = newVault();
    const { findings } = await detect({ ir, provider: "chatgpt", text: TEXT, config });
    const manual = findings.map((f) =>
      f.entityType === "in-pan" ? { ...f, action: "pseudonymize" as const } : { ...f, action: "redact" as const },
    );
    const result = await applyActions(TEXT, manual, vault, "conv1", ir);
    expect(result.applied).toHaveLength(2);
    for (const a of result.applied) {
      expect(result.text.slice(a.newStart, a.newEnd)).toBe(a.replacement);
      expect(TEXT.slice(a.start, a.end)).not.toBe(a.replacement);
    }
  });

  it("sets blocked when any finding's action is block, and still rewrites the rest", async () => {
    const vault = newVault();
    const { findings } = await detect({ ir, provider: "chatgpt", text: TEXT, config });
    // Fixture actions: in-pan -> block, aws-key -> block. Soften one to redact.
    const mixed = findings.map((f) => (f.entityType === "aws-key" ? { ...f, action: "redact" as const } : f));
    const result = await applyActions(TEXT, mixed, vault, "conv1", ir);
    expect(result.blocked).toBe(true);
    expect(result.text).toContain("ABCPD1234E"); // block does not rewrite
    expect(result.text).toContain("[REDACTED:aws-key]");
  });

  // The forward assembly is a cursor walk, and every off-by-one in it hides at
  // the edges: a mid-string finding surrounded by slack passes even if the gap
  // copy is wrong by a character. These three pin the edges themselves.
  describe("span boundaries", () => {
    it("handles adjacent findings with no gap or overlap between them", async () => {
      const text = "AAAABBBB tail";
      const findings = [
        findingAt(text, 0, 4, "redact", "aws-key"),
        findingAt(text, 4, 8, "redact", "generic-secret"),
      ];
      const result = await applyActions(text, findings, newVault(), "conv1", ir);
      expect(result.text).toBe("[REDACTED:aws-key][REDACTED:generic-secret] tail");
      // The zero-length gap between them stays zero-length: no dropped or
      // duplicated character where the two replacements meet.
      expect(result.applied[0]!.newEnd).toBe(result.applied[1]!.newStart);
      for (const a of result.applied) {
        expect(result.text.slice(a.newStart, a.newEnd)).toBe(a.replacement);
      }
    });

    it("handles a finding that starts at offset 0", async () => {
      const text = "AAAA tail";
      const result = await applyActions(text, [findingAt(text, 0, 4, "redact")], newVault(), "conv1", ir);
      expect(result.text).toBe("[REDACTED:aws-key] tail");
      expect(result.applied[0]!.newStart).toBe(0);
    });

    it("handles a finding that ends at text.length", async () => {
      const text = "tail AAAA";
      const result = await applyActions(
        text,
        [findingAt(text, 5, text.length, "redact")],
        newVault(),
        "conv1",
        ir,
      );
      expect(result.text).toBe("tail [REDACTED:aws-key]");
      // Nothing is appended after the last span, so the replacement has to run
      // to the very end of the output.
      expect(result.applied[0]!.newEnd).toBe(result.text.length);
    });
  });

  it("is the identity on empty findings", async () => {
    const result = await applyActions("hello world", [], newVault(), "conv1", ir);
    expect(result).toEqual({ text: "hello world", blocked: false, applied: [], skipped: [] });
  });

  /**
   * `applied` covers only the spans that were rewritten, so a caller holding an
   * ApplyResult knows WHAT was blocked (it has the findings) but not WHERE the
   * blocked span sits in the text it is about to render -- the offsets it has
   * are into the original, and every rewrite before them moved the text. The
   * only way to recover them from `applied` alone is to replay the deltas by
   * hand, which is Plan 6's review sheet re-deriving something the forward pass
   * already knew for free. `skipped` reports it: same text, shifted offsets.
   */
  describe("skipped spans", () => {
    // Both skipped spans deliberately sit AFTER a length-changing rewrite
    // ([REDACTED:aws-key] is 18 chars against a 20-char key, so everything to
    // its right moves by -2). An identity mapping would pass otherwise.
    const skipText = "key AKIAIOSFODNN7EXAMPLE then Globex and ABCPD1234E end";
    const at = (needle: string) => skipText.indexOf(needle);

    const mixed: ResolvedFinding[] = [
      findingAt(skipText, at("AKIA"), at("AKIA") + 20, "redact", "aws-key"),
      findingAt(skipText, at("Globex"), at("Globex") + 6, "block", "client-name"),
      findingAt(skipText, at("ABCPD"), at("ABCPD") + 10, "allow", "in-pan"),
    ];

    it("reports block and allow spans at their position in the REWRITTEN text", async () => {
      const result = await applyActions(skipText, mixed, newVault(), "conv1", ir);
      expect(result.skipped.map((s) => [s.entityType, s.action])).toEqual([
        ["client-name", "block"],
        ["in-pan", "allow"],
      ]);
      for (const s of result.skipped) {
        // The defining property: identical text, shifted offsets.
        expect(result.text.slice(s.newStart, s.newEnd)).toBe(skipText.slice(s.start, s.end));
        // ...and the shift is real, so this is not an accidental identity.
        expect(s.newStart).toBe(s.start - 2);
        expect(s.newEnd - s.newStart).toBe(s.end - s.start);
      }
      expect(result.blocked).toBe(true);
    });

    it("keeps skipped spans out of applied, and rewritten spans out of skipped", async () => {
      const result = await applyActions(skipText, mixed, newVault(), "conv1", ir);
      expect(result.applied.map((a) => a.entityType)).toEqual(["aws-key"]);
      expect(result.skipped.map((s) => s.entityType)).not.toContain("aws-key");
    });

    // One preceding rewrite cannot distinguish "accumulates every delta" from
    // "remembers the last one" -- they agree at n=1. Two rewrites with
    // DIFFERENT deltas separate them: [REDACTED:aws-key] is 18 characters over
    // a 20-character key (-2) and [REDACTED:generic-secret] is 25 over a
    // 10-character token (+15), so the correct cumulative shift is +13 and
    // every plausible wrong answer (-2, +15, 0) is a different number.
    it("accumulates the shift across several rewrites of differing length", async () => {
      const text = "key AKIAIOSFODNN7EXAMPLE tok SECRETTOKN then Globex and ABCPD1234E end";
      const span = (needle: string) => [text.indexOf(needle), text.indexOf(needle) + needle.length] as const;
      const findings: ResolvedFinding[] = [
        findingAt(text, ...span("AKIAIOSFODNN7EXAMPLE"), "redact", "aws-key"),
        findingAt(text, ...span("SECRETTOKN"), "redact", "generic-secret"),
        findingAt(text, ...span("Globex"), "block", "client-name"),
        findingAt(text, ...span("ABCPD1234E"), "allow", "in-pan"),
      ];
      const result = await applyActions(text, findings, newVault(), "conv1", ir);
      expect(result.skipped).toHaveLength(2);
      for (const s of result.skipped) {
        expect(result.text.slice(s.newStart, s.newEnd)).toBe(text.slice(s.start, s.end));
        expect(s.newStart).toBe(s.start + 13);
      }
    });

    // A block span before any rewrite must not pick up a phantom shift.
    it("reports an unshifted span when nothing before it was rewritten", async () => {
      const text = "Globex then key AKIAIOSFODNN7EXAMPLE end";
      const findings: ResolvedFinding[] = [
        findingAt(text, 0, 6, "block", "client-name"),
        findingAt(text, text.indexOf("AKIA"), text.indexOf("AKIA") + 20, "redact", "aws-key"),
      ];
      const result = await applyActions(text, findings, newVault(), "conv1", ir);
      expect(result.skipped).toEqual([
        { start: 0, end: 6, newStart: 0, newEnd: 6, entityType: "client-name", action: "block" },
      ]);
    });
  });

  it("keeps referential integrity across messages in one conversation", async () => {
    const vault = newVault();
    const mk = async (text: string) => {
      const { findings } = await detect({ ir, provider: "chatgpt", text, config });
      const manual = findings.map((f) => ({ ...f, action: "pseudonymize" as const }));
      return applyActions(text, manual, vault, "convX", ir);
    };
    const r1 = await mk("first mention ABCPD1234E ok");
    const r2 = await mk("second mention ABCPD1234E ok");
    const fake1 = r1.applied[0]!.replacement;
    expect(r2.applied[0]!.replacement).toBe(fake1);
  });

  // A failure inside apply means the text was NOT fully rewritten, so nothing
  // here may be caught and continued: a swallowed mint leaves the real value
  // verbatim inside text the caller believes was rewritten, and the caller's
  // failMode is about DETECTION failures, not this one.
  it("propagates a vault refusal instead of returning half-rewritten text", async () => {
    const vault = newVault();
    const text = "key AKIAIOSFODNN7EXAMPLE here";
    const { findings } = await detect({ ir, provider: "chatgpt", text, config });
    // Hostile/buggy caller: `aws-key` is neverPseudonymize, so neither the schema
    // nor the orchestrator can produce this action for it. Constructed by hand to
    // reach the vault's runtime backstop directly.
    const forged = findings.map((f) => ({ ...f, action: "pseudonymize" as const }));
    expect(forged.some((f) => f.entityType === "aws-key")).toBe(true);
    await expect(applyActions(text, forged, vault, "conv1", ir)).rejects.toThrow(/never be pseudonymized/);
  });

  /**
   * The forward assembly assumes the DetectionResult contract (spans disjoint,
   * sorted, in bounds). Violating it did not fail — it silently emitted REAL
   * text out of a function whose whole job is that no real text survives, which
   * is the worst possible failure mode for this layer and invisible from the
   * outside because the returned `applied` records stay internally consistent.
   * Measured before the guard existed, on "SECRETVALUE tail":
   *
   * - overlap [0,10) then [5,8) -> "[REDACTED:aws-key][REDACTED:aws-key]LUE tail":
   *   the cursor regressed (slice(10,5) === ""), then jumped to 8, so chars 8-9
   *   of a span the policy said to redact were re-emitted verbatim.
   * - start -4 on "hello world" -> "hello w[REDACTED:aws-key] world": JS reads a
   *   negative slice index from the END, so the replacement landed in an
   *   unrelated position and the whole original string survived around it.
   * - end 500 on "short" -> "[REDACTED:aws-key]": the tail was swallowed silently.
   *
   * A throw is the only safe answer: there is no partial result to forward
   * (same argument as the vault-refusal test above).
   */
  describe("span-sanity guard", () => {
    it("throws on overlapping findings instead of re-emitting the covered text", async () => {
      const text = "SECRETVALUE tail";
      const findings = [findingAt(text, 0, 10, "redact"), findingAt(text, 5, 8, "redact")];
      const run = applyActions(text, findings, newVault(), "conv1", ir);
      await expect(run).rejects.toThrow(/\[5, 8\)/);
      // The message has to name the offending producer, or a hallucinating
      // tier-1 engine is diagnosed by reading apply.ts instead of the error.
      await expect(applyActions(text, findings, newVault(), "conv2", ir)).rejects.toThrow(/hand-built/);
      await expect(applyActions(text, findings, newVault(), "conv3", ir)).rejects.toThrow(/aws-key/);
    });

    it("throws on a span that runs past the end of the text", async () => {
      const text = "short";
      const bad = { ...findingAt(text, 0, text.length, "redact"), end: 500 };
      await expect(applyActions(text, [bad], newVault(), "conv1", ir)).rejects.toThrow(/past text length 5/);
    });

    it("throws on a negative start", async () => {
      const text = "hello world";
      const bad = { ...findingAt(text, 0, 5, "redact"), start: -4 };
      await expect(applyActions(text, [bad], newVault(), "conv1", ir)).rejects.toThrow(/negative start/);
    });

    it("throws on an inverted span", async () => {
      const text = "hello world";
      const bad = { ...findingAt(text, 0, 5, "redact"), start: 5, end: 2 };
      await expect(applyActions(text, [bad], newVault(), "conv1", ir)).rejects.toThrow(/end before start/);
    });

    // The guard tracks the previous finding's end, not the rewrite cursor, so a
    // span overlapping a BLOCKED one is caught too -- block leaves the cursor
    // where it was, so a cursor-based check cannot see this collision at all,
    // and the skipped-span offsets reported below would be silently wrong.
    it("throws when a rewritten span overlaps a blocked one", async () => {
      const text = "SECRETVALUE tail";
      const findings = [findingAt(text, 0, 10, "block"), findingAt(text, 5, 8, "redact")];
      await expect(applyActions(text, findings, newVault(), "conv1", ir)).rejects.toThrow(/\[5, 8\)/);
    });

    /**
     * The guard's first form was written as four NEGATED comparisons, which is
     * the shape that lets exactly the values with no ordering through: every
     * comparison against NaN is false, so a non-numeric offset satisfied all
     * four and reached `slice`. Measured on "0123456789 tail", redacting:
     *
     * - `start: NaN` (or a JSON-borne `null`) -> "[REDACTED:aws-key] tail":
     *   `slice(0, NaN)` is "", so the ten characters before the span vanish
     *   from a message the caller believes was only rewritten.
     * - `end: NaN`/`null` -> "[REDACTED:aws-key]0123456789 tail": the marker is
     *   emitted AND the whole original survives after it -- a leak, not a
     *   swallow, and one that looks like a successful redaction.
     * - `[2.5, 6.5)` -> "01[REDACTED:aws-key]6789 tail": slice truncates, so
     *   characters 2-6 were rewritten while `applied` reports [2.5, 6.5) --
     *   offsets no consumer can trust.
     *
     * Rewritten as a single positive predicate: real integers, in range,
     * non-empty, disjoint from what came before. Anything that is not a number
     * fails the integer test rather than sliding through a comparison.
     */
    it("throws on a zero-width span", async () => {
      const text = "hello world";
      const bad = { ...findingAt(text, 0, 5, "redact"), start: 5, end: 5 };
      await expect(applyActions(text, [bad], newVault(), "conv1", ir)).rejects.toThrow(/zero-width/);
    });

    it("throws on non-numeric offsets instead of slicing with them", async () => {
      const text = "0123456789 tail";
      const base = findingAt(text, 0, 10, "redact");
      const cases: ResolvedFinding[] = [
        { ...base, start: NaN },
        { ...base, end: NaN },
        // What a finding deserialized from JSON carries: `undefined` does not
        // survive a round trip, and a producer that omitted a field yields null.
        { ...base, start: null as unknown as number },
        { ...base, end: null as unknown as number },
        { ...base, start: undefined as unknown as number },
      ];
      for (const bad of cases) {
        await expect(applyActions(text, [bad], newVault(), "conv1", ir)).rejects.toThrow(
          /non-integer offsets/,
        );
      }
    });

    it("throws on fractional offsets", async () => {
      const text = "0123456789 tail";
      const bad = { ...findingAt(text, 0, 10, "redact"), start: 2.5, end: 6.5 };
      await expect(applyActions(text, [bad], newVault(), "conv1", ir)).rejects.toThrow(
        /non-integer offsets/,
      );
    });

    /**
     * The other half of the Finding contract: `text === message.slice(start,
     * end)`. `normalizeFindings` enforces it at the pipeline entrance, and the
     * argument for repeating it here is the same one the span half rests on --
     * `applyActions` is an exported entrance in its own right, and findings can
     * be hand-built, replayed from storage, or deserialized from a worker
     * message without ever passing through `detect`.
     *
     * The damage is action-dependent, and the guard is deliberately NOT:
     *
     * - `pseudonymize` reads `f.text` and mints it as the REAL value. A drifted
     *   finding stores the wrong real, and the outbound message is fine (the
     *   span is fully replaced either way) -- the corruption surfaces one turn
     *   later, when rehydration splices the drifted text into the user's view
     *   as though the model had said it.
     * - `block`/`allow` do not read `f.text` here, but `skipped` exists so a
     *   review sheet can render WHAT is blocked next to WHERE it sits; that
     *   render reads the finding's text. Drift shows the user the wrong string
     *   as the reason their message cannot send, at offsets pointing elsewhere.
     * - `redact` alone is genuinely harmless: the span is replaced by a marker
     *   built from the entityType, and `f.text` is never read.
     *
     * Enforced for all four anyway. Exempting the one harmless branch buys
     * nothing and costs a rule that has to be re-derived on every read, and
     * `redact` is precisely the branch a policy edit flips to `pseudonymize` --
     * at which point a latent drift that was "harmless" becomes a vault-
     * poisoning bug with no code change to point at.
     */
    it("throws when a finding's text disagrees with its span", async () => {
      const text = "Globex renewed the contract.";
      const drifted = { ...findingAt(text, 0, 6, "pseudonymize", "client-name"), text: "Globe" };
      const run = () => applyActions(text, [drifted], newVault(), "conv1", ir);
      await expect(run()).rejects.toThrow(/text does not match the span/);
      await expect(run()).rejects.toThrow(/hand-built/);
      // Value-free, like every other message this guard emits: a finding's text
      // is the sensitive string, and an exception is a loggable place.
      await expect(run()).rejects.not.toThrow(/Globe/);
    });

    it("enforces text fidelity on every action, not just the branch that reads it", async () => {
      const text = "Globex renewed the contract.";
      for (const action of ["pseudonymize", "redact", "block", "allow"] as const) {
        const drifted = { ...findingAt(text, 0, 6, action, "client-name"), text: "Globe" };
        await expect(applyActions(text, [drifted], newVault(), "conv1", ir)).rejects.toThrow(
          /text does not match the span/,
        );
      }
    });
  });
});
