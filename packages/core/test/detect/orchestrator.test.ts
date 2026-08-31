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

/** A loadable IR built from the fixture, so every field the schema polices is real. */
const irWith = (over: Record<string, unknown>) =>
  loadPolicyIr(JSON.stringify({ ...minimalIr(), ...over }));

const predicate = (id: string, scope: "segment" | "message") => ({
  id,
  nlPredicate: `test predicate ${id}`,
  scope,
});

/**
 * The policy every test uses whose subject is what the JUDGE does rather than
 * whether it is called at all.
 *
 * `minimalIr()` declares no `semanticPredicates`, and escalation (spec 4.1)
 * spends a judge only where one can help -- so under that policy a clean
 * message escalates nothing and tier 2 is never reached. A degraded-channel or
 * budget test written against `ir` and a clean message would be asserting about
 * a call that did not happen, and would pass for the wrong reason.
 *
 * Scoped to `segment` deliberately: the judge stubs below report
 * `scopesJudged: ["segment"]`, so a message-scoped predicate would add a
 * `scope-unjudged` entry next to the one under test.
 */
const irPredicate = irWith({ semanticPredicates: [predicate("p1", "segment")] });
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
  // `irPredicate`, because the point is that the judge really RUNS here: under a
  // policy with no predicates escalation selects nothing on a clean message and
  // `tier2Ms` would be undefined for a reason that has nothing to do with tier 1.
  it("allows tier 2 without tier 1", async () => {
    const result = await detect({
      ir: irPredicate, provider: "chatgpt", text: "what is a monad?",
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
        // A policy with predicates, so escalation reaches the judge at all.
        ir: irPredicate, provider: "chatgpt", text: drifted,
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

  it("gives tier 2 everything the earlier tiers found, and merges what it returns", async () => {
    let priors: Finding[] = [];
    const extra = finding({
      start: 0, end: 3, text: MESSAGE.slice(0, 3), entityType: "client-name", tier: 2, source: "stub-t2",
    });
    const spy: SemanticJudge = {
      judge: async (request) => {
        priors = request.priorFindings;
        return { findings: [extra], scopesJudged: ["segment"] };
      },
    };
    const result = await detect({
      ir: irPredicate, provider: "chatgpt", text: MESSAGE,
      config: { tier0: true, tier1: false, tier2: true },
      engines: { tier2: spy },
    });
    // Priors are the whole message's findings, NOT just the escalated segments'.
    // A judge weighs a segment against what the pipeline already knows, and
    // filtering these to the escalated set would hide the tier-0 hit sitting one
    // segment away.
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

// ---------------------------------------------------------------------------
// Escalation -- spec 4.1: "tier2 iff (uncertain OR semanticPredicates present)
// AND TierConfig.tier2". At the seconds-per-call a judge costs, this decides
// whether the pipeline answers inside `ir.latencyBudgetMs` at all.
// ---------------------------------------------------------------------------

/** prose, kv and code in one message, with nothing in any of them for tier 0 to find. */
const MIXED = "Renewal notes.\nclient: Northwind Traders\n```\nconst total = a + b;\n```\n";
/** That fenced block alone: a message with no segment escalation can select. */
const CODE_ONLY = "```\nconst total = a + b;\n```\n";

/** A judge that answers nothing and records every request it was handed. */
const spyJudge = () => {
  const calls: JudgeRequest[] = [];
  return {
    calls,
    engine: {
      judge: async (request: JudgeRequest) => {
        calls.push(request);
        return { findings: [], scopesJudged: ["segment"] as const };
      },
    } satisfies SemanticJudge,
  };
};

describe("tier-2 escalation", () => {
  it("premise: MIXED and CODE_ONLY carry nothing for tier 0 to find", async () => {
    // Every test below reads escalation, and a stray tier-0 hit would add an
    // uncertainty flag and quietly change which segments get selected -- so the
    // fixtures' emptiness is asserted rather than assumed.
    for (const text of [MIXED, CODE_ONLY]) {
      const result = await detect({
        ir, provider: "chatgpt", text, config: { tier0: true, tier1: false, tier2: false },
      });
      expect(result.findings).toEqual([]);
    }
    // And MIXED really does contain one of each kind, which is what makes the
    // prose/kv/code assertions below able to fail.
    expect(segmentText(MIXED).map((s) => s.kind)).toEqual(["prose", "kv", "code"]);
  });

  it("does not call the judge at all when nothing qualifies", async () => {
    // The expensive default. Without it every message pays a model call per
    // segment to ask about a policy that declares no semantic clauses.
    const { calls, engine } = spyJudge();
    const result = await detect({
      ir, provider: "chatgpt", text: MIXED, config: T2, engines: { tier1: tagger([]), tier2: engine },
    });
    expect(calls).toEqual([]);
    // No call, no timing: a 0 here would read as a tier that ran instantly,
    // which is the ambiguity `timings`' own doc warns about.
    expect(result.timings.tier2Ms).toBeUndefined();
    // And NOT a degradation. `degraded` means "weaker than a full three-tier
    // run", and a full run is not stronger here: `WebLlmJudge.judge` returns
    // `{findings: [], scopesJudged: []}` on an IR with no `semanticPredicates`
    // before it touches the engine, so calling it would have produced the same
    // findings and the same degraded array. `timings.tier2Ms` above is the one
    // field that differs, and it is what tells a caller the judge did not run.
    expect(result.degraded).toEqual([]);
  });

  it("selects prose and kv but not code when the policy declares predicates", async () => {
    const { calls, engine } = spyJudge();
    await detect({
      ir: irPredicate, provider: "chatgpt", text: MIXED, config: T2,
      engines: { tier1: tagger([]), tier2: engine },
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.segments.map((s) => s.kind)).toEqual(["prose", "kv"]);
  });

  it("selects a code segment tier 0 left uncertain, with no predicates at all", async () => {
    // The uncertainty branch, wired to the confidences tier 0 really emits:
    // MESSAGE's fenced block holds an entropy finding, and `tier0.ts` fixes
    // entropy below every regex rule's floor precisely because it cannot tell a
    // random-looking string from a secret. Nothing here is hand-fed -- the
    // flag comes from running tier 0 over this message.
    const { calls, engine } = spyJudge();
    const result = await detect({
      ir, provider: "chatgpt", text: MESSAGE, config: T2, engines: { tier1: tagger([]), tier2: engine },
    });
    expect(segmentText(MESSAGE).map((s) => s.kind)).toEqual(["prose", "code"]);
    expect(calls).toHaveLength(1);
    // The code segment ALONE: the prose segment holds only a validated PAN,
    // which no judge call can improve on, and there are no predicates to make
    // it qualify on the other branch.
    expect(calls[0]!.segments.map((s) => s.kind)).toEqual(["code"]);
    expect(result.degraded).toEqual([]);
  });

  it("reads TierConfig.uncertainBelow instead of the built-in threshold", async () => {
    // THREE thresholds, because a test exercising only the default cannot tell
    // "reads the config" from "hardcodes 0.8" -- the failure this project has
    // shipped twice. The expectations come from tier 0's own confidences
    // (entropy 0.7 in the code segment, a boosted PAN 0.95 in the prose one),
    // not from anything escalation computed.
    const run = async (uncertainBelow?: number) => {
      const { calls, engine } = spyJudge();
      await detect({
        ir, provider: "chatgpt", text: MESSAGE,
        config: { ...T2, uncertainBelow },
        engines: { tier1: tagger([]), tier2: engine },
      });
      return calls.map((c) => c.segments.map((s) => s.kind));
    };
    // Under 0.7: nothing is uncertain, so the judge is never called.
    expect(await run(0.5)).toEqual([]);
    // Over 0.95: every tier-0 finding qualifies, so both segments are judged.
    expect(await run(0.96)).toEqual([["prose", "code"]]);
    // Omitted falls back to the default, which sits between the two.
    expect(await run()).toEqual([["code"]]);
  });

  it("refuses a threshold that would disable the branch silently", async () => {
    // `confidence < NaN` is false for every finding, so an unvalidated NaN
    // reports "nothing was uncertain" on every message an arm ever sees.
    await expect(
      detect({
        ir, provider: "chatgpt", text: MESSAGE,
        config: { ...T2, uncertainBelow: Number.NaN },
        engines: { tier1: tagger([]), tier2: judge() },
      }),
    ).rejects.toThrow(/threshold/i);
  });

  it("reports the predicates it could not judge when escalation selects nothing", async () => {
    // The other half of the skip decision. Here a full run WOULD have been
    // stronger: the policy declares a clause and no segment qualified to carry
    // it, so the run is weaker and says so. `scope-unjudged` already means
    // exactly "the policy declares predicates in a scope nothing evaluated"; a
    // new reason word would split that count across two spellings.
    const { calls, engine } = spyJudge();
    const result = await detect({
      ir: irPredicate, provider: "chatgpt", text: CODE_ONLY, config: T2,
      engines: { tier1: tagger([]), tier2: engine },
    });
    expect(calls).toEqual([]);
    expect(result.timings.tier2Ms).toBeUndefined();
    expect(kinds(result)).toEqual([{ tier: 2, reason: "scope-unjudged" }]);
    // The detail must say escalation was the reason, not that a judge answered
    // for fewer scopes than it was asked: no judge ran at all.
    expect(result.degraded[0]!.detail).toContain("escalation selected none");
    expect(result.degraded[0]!.detail).toContain("was not called");
  });

  it("reports nothing when the skipped message had no predicate to judge either", async () => {
    const result = await detect({
      ir, provider: "chatgpt", text: CODE_ONLY, config: T2,
      engines: { tier1: tagger([]), tier2: judge() },
    });
    expect(result.degraded).toEqual([]);
  });

  it("names escalation, not the budget, when both would have stopped the call", async () => {
    // The budget notice claims a cause -- "already spent ... so the judge was
    // not called" -- and here it would be the wrong one: no budget at all
    // produces a call on a message with no escalatable segment, so an operator
    // who raised `ir.latencyBudgetMs` in response would see no change.
    const result = await detect({
      ir: irWith({ semanticPredicates: [predicate("p1", "segment")], latencyBudgetMs: 10 }),
      provider: "chatgpt", text: CODE_ONLY, config: T2,
      engines: { tier1: slowTagger(60), tier2: judge() },
    });
    expect(kinds(result)).toEqual([{ tier: 2, reason: "scope-unjudged" }]);
  });
});

describe("degraded channel", () => {
  it("stays empty when all three tiers ran and answered", async () => {
    // `irPredicate`, so tier 2 really is one of the three that ran: escalation
    // skips the judge under a policy with no semantic clauses, and this test's
    // name would then be false while its assertion still held.
    const result = await detect({
      ir: irPredicate, provider: "chatgpt", text: "what is a monad?",
      config: { tier0: true, tier1: true, tier2: true },
      engines: { tier1: tagger([]), tier2: judge() },
    });
    expect(result.findings).toEqual([]);
    expect(result.degraded).toEqual([]);
  });

  it("files absent for a tier the config disabled, so length is not a cleanliness test", async () => {
    // All-three-on is the ONLY configuration that can produce an empty array,
    // and this test exists because every assertion about emptiness above uses
    // it -- so "every CONFIGURED tier ran" and "all three tiers ran" were
    // indistinguishable. A tier-0-only arm on a genuinely clean message is a
    // complete run of the pipeline it was asked for, and it still reports two
    // entries. A consumer reading `degraded.length > 0` as "this run was
    // weakened" therefore marks every tier-0 row in the bake-off matrix, and
    // every WebGPU-less machine, as weakened.
    const result = await detect({
      ir, provider: "chatgpt", text: "what is a monad?",
      config: { tier0: true, tier1: false, tier2: false },
    });
    expect(result.findings).toEqual([]);
    expect(result.degraded).not.toEqual([]);
    expect(kinds(result)).toEqual([
      { tier: 1, reason: "absent" },
      { tier: 2, reason: "absent" },
    ]);
    // The test the field's docblock actually offers a caller, in place of the
    // one it used to promise.
    expect(result.degraded.every((d) => d.reason === "absent")).toBe(true);
  });

  it("separates a clean message from one tier 2 refused to judge", async () => {
    // The reason the field exists. Both runs return zero findings; before this
    // channel the difference lived only on the judge's own counters, which
    // DetectionResult gave a caller no way to reach -- so a model that will not
    // emit valid JSON silently passed text through, which spec section 7 forbids.
    const text = "what is a monad?";
    const clean = await detect({ ir: irPredicate, provider: "chatgpt", text, config: T2, engines: withT2(judge()) });
    const refused = await detect({
      ir: irPredicate, provider: "chatgpt", text, config: T2,
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
      ir: irPredicate, provider: "chatgpt", text: "what is a monad?", config: T2, engines: withT2(liar),
    });
    expect(result.degraded).toEqual([{ tier: 2, reason: "failed-closed", detail: "d" }]);
  });

  it("refuses a reason word the engine is not entitled to say", async () => {
    // The other half of the same defence, and it has to be a REFUSAL rather
    // than an override: the orchestrator knows which tier it called, so it can
    // correct `tier`, but what went wrong inside a model is exactly what the
    // engine is being asked, so there is no right value to substitute. `absent`
    // is the case that makes this matter -- it means "this tier did not run",
    // and an engine that reached this line demonstrably ran, so passing it
    // through files the inverted record the `tier` defence exists to prevent.
    const claims = (reason: string): SemanticJudge => ({
      judge: async () => ({
        findings: [], scopesJudged: ["segment"],
        degraded: [{ reason, detail: "d" } as unknown as EngineDegradedNotice],
      }),
    });
    const run = (reason: string) =>
      detect({
        ir: irPredicate, provider: "chatgpt", text: "what is a monad?", config: T2,
        engines: withT2(claims(reason)),
      });
    for (const reason of ["absent", "budget-exhausted", "scope-unjudged", "ranch-dressing"]) {
      await expect(run(reason)).rejects.toThrow(/may not name/);
      await expect(run(reason)).rejects.toThrow(reason);
    }
    // Both legal engine words still pass through untouched.
    for (const reason of ["failed-closed", "call-budget-exhausted"] as const) {
      expect(kinds(await run(reason))).toEqual([{ tier: 2, reason }]);
    }
  });

  it("refuses an engine notice with no detail to carry", async () => {
    // `detail` is the entire human-readable payload of a notice; `reason` only
    // says the category. An empty one from the same untyped boundary is typed
    // `string` on the result and is not one.
    const silent: SemanticJudge = {
      judge: async () => ({
        findings: [], scopesJudged: ["segment"],
        degraded: [{ reason: "failed-closed" } as unknown as EngineDegradedNotice],
      }),
    };
    await expect(
      detect({ ir: irPredicate, provider: "chatgpt", text: "what is a monad?", config: T2, engines: withT2(silent) }),
    ).rejects.toThrow(/no detail/);
  });

  it("copies the two fields a notice carries and nothing else", async () => {
    // `EngineDegradedNotice`'s docblock forbids a notice carrying message text,
    // a finding's text or model output, because notices get logged. Field by
    // field is what keeps that promise; `{ ...notice, tier: 2 }` reads
    // identically at the call site and carries whatever else the producer
    // attached straight into the log.
    const chatty: SemanticJudge = {
      judge: async () => ({
        findings: [], scopesJudged: ["segment"],
        degraded: [
          { reason: "failed-closed", detail: "d", offendingBody: "my PAN is ABCPD1234E" } as unknown as EngineDegradedNotice,
        ],
      }),
    };
    const result = await detect({
      ir: irPredicate, provider: "chatgpt", text: "what is a monad?", config: T2, engines: withT2(chatty),
    });
    expect(result.degraded).toEqual([{ tier: 2, reason: "failed-closed", detail: "d" }]);
    expect(Object.keys(result.degraded[0]!).sort()).toEqual(["detail", "reason", "tier"]);
    expect(JSON.stringify(result.degraded)).not.toContain("ABCPD1234E");
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
    // Not just non-empty: a one-character detail passed the length check, and
    // `detail` is the entire human-readable payload of a notice.
    for (const notice of result.degraded) {
      expect(notice.detail).toContain(`tier ${notice.tier}`);
      expect(notice.detail).toContain("not enabled");
    }
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
    //
    // MIXED rather than MESSAGE, and that choice is what gives this test teeth.
    // Escalation drops MIXED's code segment, so the segments the judge receives
    // no longer tile the message and `segments.map(s => s.text).join("")` is
    // strictly shorter than it. Passing that join in place of `text` was a
    // KNOWN EQUIVALENT MUTANT here -- with every segment escalated the two
    // strings are byte-identical -- and the three assertions below are what
    // stop it being equivalent again.
    let seen: JudgeRequest | undefined;
    const spy: SemanticJudge = {
      judge: async (request) => { seen = request; return { findings: [], scopesJudged: ["segment"] }; },
    };
    await detect({
      ir: irWith({ semanticPredicates: [predicate("p1", "message")] }),
      provider: "chatgpt", text: MIXED, config: T2, engines: withT2(spy),
    });
    expect(seen?.text).toBe(MIXED);
    // Premise: the judge really was handed less than the whole message, so the
    // assertion above is about `text` and not about a segment list that happens
    // to reconstruct it.
    expect(seen!.segments.length).toBeLessThan(segmentText(MIXED).length);
    expect(seen!.segments.map((s) => s.text).join("")).not.toBe(MIXED);
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
    const SLEPT = 300;
    // TWO budgets, because 5000 is also `minimalIr()`'s own default and a test
    // that exercises only the default value cannot tell "reads the IR" from
    // "hardcodes the default" -- `budgetMs: 1000` survived the whole suite.
    for (const budget of [5000, 2000]) {
      await detect({
        ir: irWith({ latencyBudgetMs: budget }), provider: "chatgpt", text: MESSAGE,
        config: { tier0: true, tier1: true, tier2: true },
        engines: { tier1: slowTagger(SLEPT), tier2: spy },
      });
      // BOTH bounds are load-bearing, and the window is the test's own
      // arithmetic over its own constants rather than anything the orchestrator
      // computed. Upper: elapsed ignored, i.e. the whole budget passed through.
      // Lower: elapsed OVER-charged, which is the double-count this seam is
      // exposed to since the orchestrator is the one layer subtracting a
      // per-message number -- doubling the subtraction survived a one-sided
      // assertion, and in the results it reads as a slow model rather than as a
      // mis-subtracted budget.
      expect(seen).toBeLessThan(budget - SLEPT + 150);
      expect(seen).toBeGreaterThan(budget - SLEPT - 180);
    }
  });

  it("arms the deadline at what is LEFT of the budget, not at the whole of it", async () => {
    // The delay handed to `setTimeout` is the ONLY thing that enforces the
    // cross-segment budget, and nothing measured WHEN it fires: dividing it by
    // 1000, adding 2000 to it, and arming it with the whole `ir.latencyBudgetMs`
    // each left the suite green. `request.budgetMs` is not a stand-in for it --
    // `JudgeRequest` calls that field informational and names `signal` as the
    // enforcement.
    //
    // Measured between the judge being ENTERED and its signal firing, against
    // bounds derived from the test's own constants: tier 1 sleeps 300 ms of a
    // 600 ms budget, so about 300 ms should remain.
    let entered = 0;
    let firedAt = 0;
    const watcher: SemanticJudge = {
      judge: async (request) => {
        entered = performance.now();
        await aborted(request.signal!);
        firedAt = performance.now();
        return { findings: [], scopesJudged: ["segment"] };
      },
    };
    const result = await detect({
      ir: irWith({ latencyBudgetMs: 600 }), provider: "chatgpt", text: MESSAGE,
      config: { tier0: true, tier1: true, tier2: true },
      engines: { tier1: slowTagger(300), tier2: watcher },
    });
    const waited = firedAt - entered;
    // A timer fires late, never early, so the upper bound is what catches a
    // deadline armed with the whole budget (600) or with `remaining` plus a
    // constant; the lower bound catches one shortened by any real factor.
    expect(waited).toBeGreaterThan(150);
    expect(waited).toBeLessThan(480);
    // And the record agrees with the clock: this run really was cut short.
    expect(kinds(result)).toEqual([{ tier: 2, reason: "budget-exhausted" }]);
  });

  it("charges the earlier tiers' time to them, not to the judge", async () => {
    // `timings.tier2Ms` was only ever asserted `>= 0`, so starting its window
    // at `messageStarted` -- which charges tier 0, tier 1 and segmentation to
    // the judge -- survived, reporting a 0.15 ms judge as a 202 ms one. Per-tier
    // latency between arms is the comparison `timings` exists for.
    const result = await detect({
      ir: irWith({ latencyBudgetMs: 10_000 }), provider: "chatgpt", text: MESSAGE,
      config: { tier0: true, tier1: true, tier2: true },
      engines: { tier1: slowTagger(200), tier2: judge() },
    });
    expect(result.timings.tier1Ms).toBeGreaterThan(150);
    expect(result.timings.tier2Ms).toBeLessThan(50);
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
    // Blanking this detail survived the suite, and so would swapping it for the
    // mid-run sentence: `reason` is the same word on both paths, so `detail` is
    // the only thing that tells an operator a budget spent BEFORE the call from
    // one that expired during it.
    expect(result.degraded[0]!.detail).toContain("10ms message latency budget was already spent");
    expect(result.degraded[0]!.detail).toContain("the judge was not called");
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
    // The other half of the pair above: this sentence must say the budget ran
    // out DURING the run, and must name what the judge was actually given.
    expect(result.degraded[0]!.detail).toContain("30ms message latency budget ran out during the tier-2 run");
    expect(result.degraded[0]!.detail).toMatch(/given the \d+ms that remained/);
  });

  it("does not report a budget expiry when the judge answered inside it", async () => {
    const result = await detect({
      ir: irWith({ latencyBudgetMs: 5000 }), provider: "chatgpt", text: MESSAGE, config: T2,
      engines: withT2(judge()),
    });
    expect(result.degraded).toEqual([]);
  });

  it("clears its budget timer when the judge throws", async () => {
    // What a leaked timer costs, measured rather than assumed -- this comment
    // used to claim it aborts the NEXT detect(), and a probe says it cannot:
    // the AbortController is built per call, so a leaked timer aborts a
    // controller nobody is listening to. What it does cost is a ref'd handle
    // (measured on Node v26.0.0: a process whose only remaining work was an
    // uncleared 800 ms timer exited at 803 ms instead of at 0), so a batch
    // harness lingers for the rest of every abandoned budget. See the
    // orchestrator's own comment for both measurements.
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
