import { describe, expect, it } from "vitest";
import { detect, remainingBudgetMs } from "../../src/detect/orchestrator.js";
import { loadPolicyIr } from "../../src/policy/load.js";
import { resolveAction } from "../../src/policy/resolve.js";
import { segmentText, type Segment } from "../../src/segment/segment.js";
import type {
  DetectionResult,
  EngineDegradedNotice,
  Finding,
  JudgeRequest,
  JudgeVerdict,
  SemanticJudge,
  SpanTagger,
} from "../../src/detect/types.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));
const config = { tier0: true, tier1: false, tier2: false };
const SECRET = "x9K2mQ8vL4jR7nT3wY6zB1cD5fG0hJpZ";

const MESSAGE = [
  "Hey, my PAN is ABCPD1234E for the tax form.",
  "```",
  "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
  "session = " + SECRET,
  "```",
].join("\n");

describe("detect (tier-0 only)", () => {
  it("returns resolved findings end-to-end", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    const byType = Object.fromEntries(result.findings.map((x) => [x.entityType, x]));
    expect(byType["in-pan"]!.action).toBe("block");
    expect(byType["aws-key"]!.action).toBe("block");
    expect(byType["generic-secret"]!.action).toBe("redact");
    expect(result.findings).toHaveLength(3);
  });

  it("all spans are faithful to the message", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    for (const x of result.findings) {
      expect(x.text).toBe(MESSAGE.slice(x.start, x.end));
    }
  });

  it("records tier-0 timing", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    expect(result.timings.tier0Ms).toBeGreaterThanOrEqual(0);
    expect(result.timings.tier1Ms).toBeUndefined();
  });

  it("returns no findings on clean text", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: "what is a monad?", config });
    expect(result.findings).toEqual([]);
  });

  it("tier0=false disables tier 0", async () => {
    const result = await detect({
      ir, provider: "chatgpt", text: MESSAGE, config: { tier0: false, tier1: false, tier2: false },
    });
    expect(result.findings).toEqual([]);
  });
});

/** Stub engines: the tier 1/2 seams Plans 4-5 fill in with real models. */
const tagger = (findings: Finding[]): SpanTagger => ({ tag: async () => findings });
/**
 * `scopesJudged: ["segment"]` is what `WebLlmJudge` really reports, so a stub
 * claiming more would let a scope-coverage bug pass here and fail in tier 2.
 */
const judge = (findings: Finding[] = []): SemanticJudge => ({
  judge: async () => ({ findings, scopesJudged: ["segment"] }),
});

/** A judge whose whole verdict the test dictates, including what it claims to have done. */
const verdictJudge = (verdict: Partial<JudgeVerdict>): SemanticJudge => ({
  judge: async () => ({ findings: [], scopesJudged: ["segment"], ...verdict }),
});

const finding = (over: Partial<Finding> & Pick<Finding, "start" | "end" | "entityType">): Finding => ({
  text: "", severity: "high", tier: 1, source: "stub-t1", confidence: 0.8, ...over,
});

describe("engine-presence guards", () => {
  // A silently skipped tier corrupts an experiment arm: you believe you measured
  // T0+T1 and you measured T0. Enabling a tier without its engine is a caller bug.
  it("throws when tier 1 is enabled with no tier-1 engine", async () => {
    await expect(
      detect({ ir, provider: "chatgpt", text: MESSAGE, config: { tier0: true, tier1: true, tier2: false } }),
    ).rejects.toThrow(/tier1/);
  });

  it("throws when tier 2 is enabled with no tier-2 engine", async () => {
    await expect(
      detect({ ir, provider: "chatgpt", text: MESSAGE, config: { tier0: true, tier1: false, tier2: true } }),
    ).rejects.toThrow(/tier2/);
  });

  // Tier 2 judges semantic predicates, which do not depend on tier-1 spans, so
  // predicates-only escalation is a legitimate configuration -- not a guard case.
  it("allows tier 2 without tier 1", async () => {
    const result = await detect({
      ir, provider: "chatgpt", text: "what is a monad?",
      config: { tier0: true, tier1: false, tier2: true },
      engines: { tier2: judge() },
    });
    expect(result.findings).toEqual([]);
    expect(result.timings.tier2Ms).toBeGreaterThanOrEqual(0);
  });
});

describe("finding normalization", () => {
  // The guard that turns a future tier-1/2 hallucinated label into a diagnosable
  // error here instead of a resolveAction crash deeper in the pipeline.
  it("rejects a finding whose entityType is not declared in the IR", async () => {
    const run = () =>
      detect({
        ir, provider: "chatgpt", text: "Globex signed the deal.",
        config: { tier0: false, tier1: true, tier2: false },
        engines: { tier1: tagger([finding({ start: 0, end: 6, text: "Globex", entityType: "client-nmae" })]) },
      });
    await expect(run()).rejects.toThrow(/client-nmae/);
    await expect(run()).rejects.toThrow(/stub-t1/);
  });

  it("re-derives severity from the IR, never trusting the producer", async () => {
    const text = "Globex signed the deal.";
    const claimed = finding({ start: 0, end: 6, text: "Globex", entityType: "client-name", severity: "low" });
    const result = await detect({
      ir, provider: "chatgpt", text,
      config: { tier0: true, tier1: true, tier2: false },
      engines: { tier1: tagger([claimed]) },
    });
    expect(result.findings).toHaveLength(1);
    // "client-name" is `high` in the IR; the producer claimed `low`.
    expect(result.findings[0]!.severity).toBe("high");
    expect(result.findings[0]!.action).toBe("pseudonymize");
    // Normalization is pure: the engine's own object is never rewritten.
    expect(claimed.severity).toBe("low");
  });

  /**
   * Spans were trusted entirely, and every consumer downstream is built on the
   * assumption in `types.ts` that `text === message.slice(start, end)`:
   * `applyActions` rewrites BY OFFSET, so a drifted span sends a neighbouring
   * word to the vault and leaves the real value in place, and the merge orders
   * and clusters by offsets that describe nothing. Tier 0 cannot violate this
   * (it slices the message itself); a tokenizer-backed tier-1/2 model can and
   * routinely does, which is why `SpanTagger` already tells implementers to
   * re-derive `text`. The contract is now enforced at the point the pipeline
   * takes ownership rather than restated in a docstring.
   */
  const drifted = "Globex signed the deal.";
  const run = (f: Finding) =>
    detect({
      ir, provider: "chatgpt", text: drifted,
      config: { tier0: false, tier1: true, tier2: false },
      engines: { tier1: tagger([f]) },
    });

  it("rejects a finding whose text disagrees with its span", async () => {
    // Classic tokenizer drift: the offsets say [0,6), the model says "Globe".
    const f = finding({ start: 0, end: 6, text: "Globe", entityType: "client-name" });
    await expect(run(f)).rejects.toThrow(/does not match its span/);
    await expect(run(f)).rejects.toThrow(/stub-t1/);
    // The message reports lengths, never the values: a finding's text is the
    // sensitive string this whole system exists to keep out of places it may be
    // logged, and an exception message is exactly such a place.
    await expect(run(f)).rejects.not.toThrow(/Globe/);
  });

  it("rejects a finding whose span runs past the end of the text", async () => {
    const f = finding({ start: 0, end: drifted.length + 5, text: drifted, entityType: "client-name" });
    await expect(run(f)).rejects.toThrow(/out-of-range span/);
    await expect(run(f)).rejects.toThrow(/stub-t1/);
  });

  it("rejects a negative start", async () => {
    const f = finding({ start: -1, end: 6, text: "Globex", entityType: "client-name" });
    await expect(run(f)).rejects.toThrow(/out-of-range span/);
  });

  // Zero-width is rejected rather than tolerated: tier 0 never emits one, and a
  // finding covering no characters means nothing downstream -- it would mint a
  // surrogate for the empty string (the vault refuses), or splice a redaction
  // marker into text at a point where nothing was detected.
  it("rejects a zero-width finding", async () => {
    const f = finding({ start: 3, end: 3, text: "", entityType: "client-name" });
    await expect(run(f)).rejects.toThrow(/out-of-range span/);
  });

  /**
   * The range check is NaN-safe by construction (every comparison against NaN
   * is false, so the positive predicate fails), but `null` is a different
   * animal: it coerces to 0, so `null >= 0` is TRUE and `null < 6` is TRUE, and
   * `slice(null, 6)` is `slice(0, 6)` -- which means a null start cleared the
   * text-fidelity check too and reached the merge, where it sorts as 0. It
   * could not leak (applyActions rejects it), but it was diagnosed a stage late
   * and under the wrong name. An offset must be an integer before any ordering
   * question about it means anything.
   */
  it("rejects offsets that are not integers", async () => {
    const cases = [
      { start: null as unknown as number, end: 6, text: "Globex" },
      { start: 0, end: null as unknown as number, text: "" },
      { start: NaN, end: 6, text: "Globex" },
      { start: 0.5, end: 6, text: "Globex" },
    ];
    for (const c of cases) {
      await expect(run(finding({ ...c, entityType: "client-name" }))).rejects.toThrow(
        /non-integer offsets/,
      );
    }
  });

  it("validates tier-2 findings on the same terms", async () => {
    const bad = finding({ start: 0, end: 6, text: "Globe", entityType: "client-name", tier: 2, source: "stub-t2" });
    await expect(
      detect({
        ir, provider: "chatgpt", text: drifted,
        config: { tier0: false, tier1: false, tier2: true },
        engines: { tier2: judge([bad]) },
      }),
    ).rejects.toThrow(/stub-t2/);
  });
});

describe("cluster-strictest action resolution", () => {
  it("escalates a winner's action to the strictest in its overlap cluster", async () => {
    // A critical entropy finding (`generic-secret` -> redact) swallows an
    // overlapping high-severity finding (`in-pan` -> block) in the merge. Taking
    // the winner's own action alone would silently downgrade block to redact.
    const text = ["```", "session = " + SECRET, "```"].join("\n");
    const at = text.indexOf(SECRET);
    const loser = finding({
      start: at + 4, end: at + 14, text: text.slice(at + 4, at + 14),
      entityType: "in-pan", confidence: 0.99,
    });
    const result = await detect({
      ir, provider: "chatgpt", text,
      config: { tier0: true, tier1: true, tier2: false },
      engines: { tier1: tagger([loser]) },
    });
    expect(result.findings).toHaveLength(1);
    const winner = result.findings[0]!;
    expect(winner.entityType).toBe("generic-secret");
    expect(winner.text).toBe(SECRET);
    // Premise: on its own the winner would only be redacted.
    expect(resolveAction(ir, "generic-secret", "chatgpt")).toBe("redact");
    expect(winner.action).toBe("block");
  });

  it("escalates cluster-WIDE, through a bridging finding", async () => {
    // Chain cluster A-B-C: A and C are disjoint, B overlaps both. The strict
    // action lives at C, the far end; the winner (A) overlaps only the mild B.
    // Narrowing resolution to a winner's DIRECT overlaps would still pass every
    // other test in this file -- this is the test that stops that "optimization".
    const text = "0123456789abcdefghijklmnopqrst";
    const a = finding({ start: 0, end: 10, text: text.slice(0, 10), entityType: "generic-secret", confidence: 0.7 });
    const b = finding({ start: 8, end: 20, text: text.slice(8, 20), entityType: "client-name", confidence: 0.8 });
    const c = finding({ start: 18, end: 28, text: text.slice(18, 28), entityType: "in-pan", confidence: 0.9 });
    const result = await detect({
      ir, provider: "chatgpt", text,
      config: { tier0: false, tier1: true, tier2: false },
      engines: { tier1: tagger([a, b, c]) },
    });
    // B loses to A (critical beats high); C survives because it never overlapped A.
    expect(result.findings.map((x) => x.entityType)).toEqual(["generic-secret", "in-pan"]);
    const winner = result.findings[0]!;
    const farEnd = result.findings[1]!;
    expect(winner.end).toBeLessThanOrEqual(farEnd.start); // premise: the two are disjoint
    expect(resolveAction(ir, "client-name", "chatgpt")).toBe("pseudonymize"); // premise: the bridge is mild
    expect(winner.action).toBe("block"); // escalated across B, from C
  });

  it("leaves a non-overlapping finding on its own action", async () => {
    const result = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    const secret = result.findings.find((x) => x.entityType === "generic-secret")!;
    expect(secret.action).toBe("redact");
  });
});

/**
 * A neverPseudonymize entityType whose OWN action is `allow`. Legal policy: the
 * schema rejects only `pseudonymize` for these, so `allow` on a credential class
 * a policy does not care about loads fine -- and it is the one input from which
 * cluster-strictest can stamp `pseudonymize` onto a credential.
 */
const allowCredentialIr = (() => {
  const raw = minimalIr();
  raw.actions.default["aws-key"] = "allow";
  return loadPolicyIr(JSON.stringify(raw));
})();

describe("neverPseudonymize winners escalate rather than pseudonymize", () => {
  it("escalates a neverPseudonymize winner from pseudonymize to redact", async () => {
    // `aws-key` is neverPseudonymize/critical at `allow`; the overlapping
    // `client-name` is high at `pseudonymize`. allow(0) < pseudonymize(1), so the
    // cluster resolves to `pseudonymize` and the credential WINS the merge on
    // severity -- stamping the credential with an action the vault refuses to
    // mint, i.e. a guaranteed throw in the apply stage on a schema-valid IR.
    const text = "key AKIAIOSFODNN7EXAMPLE here";
    const at = text.indexOf("AKIA");
    const neighbour = finding({
      start: at + 2, end: at + 12, text: text.slice(at + 2, at + 12),
      entityType: "client-name", confidence: 0.9,
    });
    const result = await detect({
      ir: allowCredentialIr, provider: "chatgpt", text,
      config: { tier0: true, tier1: true, tier2: false },
      engines: { tier1: tagger([neighbour]) },
    });
    expect(result.findings).toHaveLength(1);
    const winner = result.findings[0]!;
    expect(winner.entityType).toBe("aws-key");
    // Premises: the credential alone is allowed, and the neighbour is what drags
    // the cluster up to `pseudonymize`.
    expect(resolveAction(allowCredentialIr, "aws-key", "chatgpt")).toBe("allow");
    expect(resolveAction(allowCredentialIr, "client-name", "chatgpt")).toBe("pseudonymize");
    expect(winner.action).toBe("redact");
  });

  it("escalates per winner, not per cluster", async () => {
    // Chain A-B-C: the credential A overlaps a bridging client-name B, which
    // overlaps a second client-name C that is disjoint from A. A and C both
    // survive the merge inside one `pseudonymize` cluster. Only A escalates --
    // `pseudonymize` is a legal, and more useful, action for C, and blanket
    // cluster-level escalation would throw that utility away.
    const text = "key AKIAIOSFODNN7EXAMPLE here plus more text";
    const bridge = finding({
      start: 20, end: 30, text: text.slice(20, 30), entityType: "client-name", confidence: 0.7,
    });
    const far = finding({
      start: 28, end: 38, text: text.slice(28, 38), entityType: "client-name", confidence: 0.9,
    });
    const result = await detect({
      ir: allowCredentialIr, provider: "chatgpt", text,
      config: { tier0: true, tier1: true, tier2: false },
      engines: { tier1: tagger([bridge, far]) },
    });
    expect(result.findings.map((x) => x.entityType)).toEqual(["aws-key", "client-name"]);
    const credential = result.findings[0]!;
    const survivor = result.findings[1]!;
    expect(credential.end).toBeLessThanOrEqual(survivor.start); // premise: disjoint, bridged by B
    expect(credential.action).toBe("redact");
    expect(survivor.action).toBe("pseudonymize");
  });
});

describe("engine failures propagate", () => {
  // Spec 5.3: detection THROWS and never silently degrades -- mapping an engine
  // crash to ir.failMode belongs to the caller. These pin that contract before
  // Plan 5 adds latency-budget degradation, when a try/catch around an engine
  // call becomes tempting: catching here would quietly reclassify a crashed
  // model as a clean scan, and every other test in this file would still pass.
  it("propagates a tier-1 engine rejection", async () => {
    const boom: SpanTagger = { tag: async () => { throw new Error("t1 engine exploded"); } };
    await expect(
      detect({
        ir, provider: "chatgpt", text: MESSAGE,
        config: { tier0: true, tier1: true, tier2: false },
        engines: { tier1: boom },
      }),
    ).rejects.toThrow("t1 engine exploded");
  });

  it("propagates a tier-2 engine rejection", async () => {
    const boom: SemanticJudge = { judge: async () => { throw new Error("t2 engine exploded"); } };
    await expect(
      detect({
        ir, provider: "chatgpt", text: MESSAGE,
        config: { tier0: true, tier1: false, tier2: true },
        engines: { tier2: boom },
      }),
    ).rejects.toThrow("t2 engine exploded");
  });
});

describe("tier seams (Plans 4-5)", () => {
  it("passes only prose and kv segments to tier 1, and records its timing", async () => {
    let seen: Segment[] = [];
    const spy: SpanTagger = { tag: async (segments) => { seen = segments; return []; } };
    const result = await detect({
      ir, provider: "chatgpt", text: MESSAGE,
      config: { tier0: true, tier1: true, tier2: false },
      engines: { tier1: spy },
    });
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((s) => s.kind !== "code")).toBe(true);
    // The fixture really does contain a code segment, so the filter is load-bearing.
    expect(segmentText(MESSAGE).some((s) => s.kind === "code")).toBe(true);
    expect(result.timings.tier1Ms).toBeGreaterThanOrEqual(0);
    expect(result.timings.tier2Ms).toBeUndefined();
  });

  it("gives tier 2 every segment plus the prior findings, and merges what it returns", async () => {
    let seen: Segment[] = [];
    let priors: Finding[] = [];
    const extra = finding({
      start: 0, end: 3, text: MESSAGE.slice(0, 3), entityType: "client-name", tier: 2, source: "stub-t2",
    });
    const spy: SemanticJudge = {
      judge: async (request) => {
        seen = request.segments;
        priors = request.priorFindings;
        return { findings: [extra], scopesJudged: ["segment"] };
      },
    };
    const result = await detect({
      ir, provider: "chatgpt", text: MESSAGE,
      config: { tier0: true, tier1: false, tier2: true },
      engines: { tier2: spy },
    });
    expect(seen).toEqual(segmentText(MESSAGE));
    expect(priors.map((p) => p.entityType).sort()).toEqual(["aws-key", "generic-secret", "in-pan"]);
    const added = result.findings.find((x) => x.entityType === "client-name")!;
    expect(added.action).toBe("pseudonymize");
    expect(result.findings).toHaveLength(4);
    expect(result.timings.tier2Ms).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// Plan 5's three orchestrator gaps: the degraded channel, predicate scope, and
// a latency budget that spans segments.
// ---------------------------------------------------------------------------

/**
 * All three tiers on, so `degraded` holds exactly what the tier under test
 * produced. Leaving tier 1 off would put an `absent` entry in every assertion
 * below and hide the entry that is actually being checked.
 */
const T2 = { tier0: true, tier1: true, tier2: true };

/** Tier 1 present and silent, so only tier 2 can add to `degraded`. */
const withT2 = (tier2: SemanticJudge) => ({ tier1: tagger([]), tier2 });

/** A loadable IR built from the fixture, so every field the schema polices is real. */
const irWith = (over: Record<string, unknown>) =>
  loadPolicyIr(JSON.stringify({ ...minimalIr(), ...over }));

const predicate = (id: string, scope: "segment" | "message") => ({
  id,
  nlPredicate: `test predicate ${id}`,
  scope,
});

/** Tier and reason only: the details are prose and are asserted where they are produced. */
const kinds = (result: DetectionResult) =>
  result.degraded.map(({ tier, reason }) => ({ tier, reason }));

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

const slowTagger = (ms: number): SpanTagger => ({
  tag: async () => { await sleep(ms); return []; },
});

const aborted = (signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (signal.aborted) resolve();
    else signal.addEventListener("abort", () => { resolve(); }, { once: true });
  });

describe("degraded channel", () => {
  it("stays empty when every configured tier ran and answered", async () => {
    const result = await detect({
      ir, provider: "chatgpt", text: "what is a monad?",
      config: { tier0: true, tier1: true, tier2: true },
      engines: { tier1: tagger([]), tier2: judge() },
    });
    // Empty findings AND empty degraded is the only combination that entitles a
    // caller to say the message is clean.
    expect(result.findings).toEqual([]);
    expect(result.degraded).toEqual([]);
  });

  it("separates a clean message from one tier 2 refused to judge", async () => {
    // The reason the field exists. Both runs return zero findings; before this
    // channel the difference lived only on the judge's own counters, which
    // DetectionResult gave a caller no way to reach -- so a model that will not
    // emit valid JSON silently passed text through, which spec section 7 forbids.
    const text = "what is a monad?";
    const clean = await detect({ ir, provider: "chatgpt", text, config: T2, engines: withT2(judge()) });
    const refused = await detect({
      ir, provider: "chatgpt", text, config: T2,
      engines: withT2(
        verdictJudge({
          degraded: [{ reason: "failed-closed", detail: "2 of 3 segments unparseable after one repair" }],
        }),
      ),
    });
    expect(refused.findings).toEqual(clean.findings);
    expect(clean.degraded).toEqual([]);
    expect(refused.degraded).toEqual([
      { tier: 2, reason: "failed-closed", detail: "2 of 3 segments unparseable after one repair" },
    ]);
  });

  it("stamps the tier it called, never a tier the engine names", async () => {
    // Engines are not trusted to name a tier, for the same reason
    // normalizeFindings does not trust one to name a severity: a result saying
    // tier 0 failed closed when tier 2 did is a confident, wrong record. The
    // notice type carries no `tier` at all, so this can only arrive from
    // untyped JS -- which the eval harness's page boundary is.
    const liar: SemanticJudge = {
      judge: async () => ({
        findings: [],
        scopesJudged: ["segment"],
        degraded: [{ tier: 0, reason: "failed-closed", detail: "d" } as unknown as EngineDegradedNotice],
      }),
    };
    const result = await detect({
      ir, provider: "chatgpt", text: "what is a monad?", config: T2, engines: withT2(liar),
    });
    expect(result.degraded).toEqual([{ tier: 2, reason: "failed-closed", detail: "d" }]);
  });

  it("names every tier that did not run", async () => {
    const result = await detect({
      ir, provider: "chatgpt", text: MESSAGE,
      config: { tier0: false, tier1: false, tier2: false },
    });
    expect(kinds(result)).toEqual([
      { tier: 0, reason: "absent" },
      { tier: 1, reason: "absent" },
      { tier: 2, reason: "absent" },
    ]);
    for (const notice of result.degraded) expect(notice.detail.length).toBeGreaterThan(0);
  });

  it("tells a tier 0 that did not run from a tier 0 that ran", async () => {
    // timings.tier0Ms cannot: it reads 0 for both, as its own doc warns.
    const off = await detect({
      ir, provider: "chatgpt", text: MESSAGE, config: { tier0: false, tier1: false, tier2: false },
    });
    const on = await detect({ ir, provider: "chatgpt", text: MESSAGE, config });
    expect(kinds(off)).toContainEqual({ tier: 0, reason: "absent" });
    expect(kinds(on)).not.toContainEqual({ tier: 0, reason: "absent" });
  });

  it("does not file an engine crash as a degradation", async () => {
    // Spec 5.3 maps a crash to ir.failMode, which is the CALLER's decision.
    // A notice here would be detection deciding it for them, and quietly.
    const boom: SemanticJudge = { judge: async () => { throw new Error("t2 engine exploded"); } };
    await expect(
      detect({ ir, provider: "chatgpt", text: MESSAGE, config: T2, engines: withT2(boom) }),
    ).rejects.toThrow("t2 engine exploded");
  });
});

describe("semantic predicate scope", () => {
  it("hands the judge the whole message, with segment offsets that index into it", async () => {
    // A judge cannot reconstruct the message from segments it may have been
    // handed a filtered list of, so scope: "message" is unanswerable without this.
    let seen: JudgeRequest | undefined;
    const spy: SemanticJudge = {
      judge: async (request) => { seen = request; return { findings: [], scopesJudged: ["segment"] }; },
    };
    await detect({
      ir: irWith({ semanticPredicates: [predicate("p1", "message")] }),
      provider: "chatgpt", text: MESSAGE, config: T2, engines: withT2(spy),
    });
    expect(seen?.text).toBe(MESSAGE);
    for (const segment of seen!.segments) {
      expect(seen!.text.slice(segment.start, segment.end)).toBe(segment.text);
    }
  });

  it("reports a scope the policy declares and the judge did not evaluate", async () => {
    const result = await detect({
      ir: irWith({ semanticPredicates: [predicate("p1", "message")] }),
      provider: "chatgpt", text: MESSAGE, config: T2,
      engines: withT2(judge()),
    });
    expect(kinds(result)).toEqual([{ tier: 2, reason: "scope-unjudged" }]);
    expect(result.degraded[0]!.detail).toContain("message");
  });

  it("reports nothing when the judge evaluated every scope the policy declares", async () => {
    const result = await detect({
      ir: irWith({ semanticPredicates: [predicate("p1", "message")] }),
      provider: "chatgpt", text: MESSAGE, config: T2,
      engines: withT2(verdictJudge({ scopesJudged: ["segment", "message"] })),
    });
    expect(result.degraded).toEqual([]);
  });

  it("reports nothing about a scope the policy declares no predicate in", async () => {
    // The fixture IR has no semanticPredicates at all, and a judge that says it
    // evaluated only segments must not be reported for the message scope it was
    // never asked about.
    const result = await detect({
      ir, provider: "chatgpt", text: MESSAGE, config: T2, engines: withT2(judge()),
    });
    expect(result.degraded).toEqual([]);
  });

  it("reports each unevaluated scope separately", async () => {
    const result = await detect({
      ir: irWith({ semanticPredicates: [predicate("p1", "message"), predicate("p2", "segment")] }),
      provider: "chatgpt", text: MESSAGE, config: T2,
      engines: withT2(verdictJudge({ scopesJudged: [] })),
    });
    expect(kinds(result)).toEqual([
      { tier: 2, reason: "scope-unjudged" },
      { tier: 2, reason: "scope-unjudged" },
    ]);
    expect(result.degraded.map((d) => d.detail).join(" ")).toContain("segment");
    expect(result.degraded.map((d) => d.detail).join(" ")).toContain("message");
  });
});

describe("remainingBudgetMs", () => {
  // The boundary a wall clock cannot be steered onto. Every expectation here
  // comes from what setTimeout does with the number, not from the arithmetic:
  // MEASURED on Node v26.0.0 and on Chrome 148, setTimeout fires within ~1 ms
  // for a delay of 0, -1, NaN, 2147483648 and Infinity, and does not fire
  // within 400 ms for 2147483647. So a budget that is spent must not be handed
  // on as a number at all, and one too large for the field must be clamped.
  it("is spent at exactly the budget", () => {
    expect(remainingBudgetMs(100, 100)).toBeUndefined();
  });

  it("is spent past the budget", () => {
    expect(remainingBudgetMs(100, 100.5)).toBeUndefined();
  });

  it("is what is left just inside the budget", () => {
    expect(remainingBudgetMs(100, 99)).toBe(1);
  });

  it("clamps to the largest delay a timer holds", () => {
    expect(remainingBudgetMs(2 ** 31, 0)).toBe(2_147_483_647);
  });

  it("treats a budget that is not a number as spent", () => {
    // Unreachable through loadPolicyIr, which is the validation boundary --
    // but NaN reaching setTimeout is a 1 ms deadline, so the direction of the
    // failure is chosen here rather than left to the timer.
    expect(remainingBudgetMs(Number.NaN, 0)).toBeUndefined();
  });
});

describe("message latency budget", () => {
  it("hands tier 2 what is LEFT of the message budget, not the whole of it", async () => {
    // A per-call budget that ignores the earlier tiers is how a 12-segment
    // message spends 12x its budget with every individual call compliant.
    let seen: number | undefined;
    const spy: SemanticJudge = {
      judge: async (request) => { seen = request.budgetMs; return { findings: [], scopesJudged: ["segment"] }; },
    };
    await detect({
      ir: irWith({ latencyBudgetMs: 5000 }), provider: "chatgpt", text: MESSAGE,
      config: { tier0: true, tier1: true, tier2: true },
      engines: { tier1: slowTagger(60), tier2: spy },
    });
    expect(seen).toBeGreaterThan(0);
    // Tier 1 slept 60 ms of the 5000, so anything at or above 4940 means the
    // budget was passed through rather than spent down.
    expect(seen).toBeLessThan(4940);
  });

  it("hands the judge a signal that has not already fired", async () => {
    let signalled: boolean | undefined;
    const spy: SemanticJudge = {
      judge: async (request) => {
        signalled = request.signal?.aborted;
        return { findings: [], scopesJudged: ["segment"] };
      },
    };
    await detect({ ir, provider: "chatgpt", text: MESSAGE, config: T2, engines: withT2(spy) });
    expect(signalled).toBe(false);
  });

  it("does not call tier 2 at all when the earlier tiers spent the budget", async () => {
    let called = false;
    const never: SemanticJudge = {
      judge: async () => { called = true; return { findings: [], scopesJudged: ["segment"] }; },
    };
    const result = await detect({
      ir: irWith({ latencyBudgetMs: 10 }), provider: "chatgpt", text: MESSAGE,
      config: { tier0: true, tier1: true, tier2: true },
      engines: { tier1: slowTagger(60), tier2: never },
    });
    expect(called).toBe(false);
    // No call means no timing: a 0 here would read as a tier that ran instantly.
    expect(result.timings.tier2Ms).toBeUndefined();
    expect(kinds(result)).toEqual([{ tier: 2, reason: "budget-exhausted" }]);
    // Spec 5.3: an over-budget tier 2 degrades TO the lower tiers' findings.
    expect(result.findings.length).toBeGreaterThan(0);
  });

  it("cuts a tier-2 run short when the message budget expires, and keeps what it collected", async () => {
    const collected = finding({
      start: 0, end: 3, text: MESSAGE.slice(0, 3), entityType: "client-name", tier: 2, source: "stub-t2",
    });
    const slow: SemanticJudge = {
      judge: async (request) => {
        await aborted(request.signal!);
        return { findings: [collected], scopesJudged: ["segment"] };
      },
    };
    const result = await detect({
      ir: irWith({ latencyBudgetMs: 30 }), provider: "chatgpt", text: MESSAGE, config: T2,
      engines: withT2(slow),
    });
    expect(kinds(result)).toEqual([{ tier: 2, reason: "budget-exhausted" }]);
    expect(result.findings.some((f) => f.entityType === "client-name")).toBe(true);
  });

  it("does not report a budget expiry when the judge answered inside it", async () => {
    const result = await detect({
      ir: irWith({ latencyBudgetMs: 5000 }), provider: "chatgpt", text: MESSAGE, config: T2,
      engines: withT2(judge()),
    });
    expect(result.degraded).toEqual([]);
  });

  it("clears its budget timer when the judge throws", async () => {
    // A timer left running outlives the message that armed it. Its abort then
    // fires during whatever detect() call happens to be in flight next, which
    // reads as a slow model on a message that was never slow.
    const cleared: unknown[] = [];
    const realClear = globalThis.clearTimeout;
    globalThis.clearTimeout = ((handle: never) => {
      cleared.push(handle);
      realClear(handle);
    }) as typeof clearTimeout;
    const boom: SemanticJudge = { judge: async () => { throw new Error("t2 engine exploded"); } };
    try {
      await expect(
        detect({
          ir: irWith({ latencyBudgetMs: 5000 }), provider: "chatgpt", text: MESSAGE, config: T2,
          engines: withT2(boom),
        }),
      ).rejects.toThrow("t2 engine exploded");
    } finally {
      globalThis.clearTimeout = realClear;
    }
    expect(cleared).toHaveLength(1);
  });
});
