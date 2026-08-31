import { describe, expect, it } from "vitest";
import type { Finding, Segment } from "@sih/core";
import { detect, loadPolicyIr } from "@sih/core";
import { DeadlineExpired, MINIMUM_CANDIDATE_WORDS } from "../src/index.js";
import { WebLlmJudge } from "../src/judge.js";
import { fakeEngine, predicateIr } from "./helpers.js";

const MSG = "Please review the Northwind Traders renewal before Friday.";
const QUOTE = "Northwind Traders renewal";
const BUDGET = { budgetMs: 30_000 } as const;

/** The whole message as one prose segment, which is what `segmentText(MSG)` gives. */
const whole = (text = MSG, start = 0): Segment[] => [
  { kind: "prose", start, end: start + text.length, text },
];

const hit = (quote = QUOTE, confidence = 0.9, predicateId = "client-relationship") => ({
  predicateId,
  quote,
  confidence,
});

describe("WebLlmJudge", () => {
  it("emits the SHADOW entityType, not the bare predicate id", async () => {
    // The contract core has documented since Plan 3. A bare predicate id makes
    // normalizeFindings throw, killing the whole message.
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(found[0]!.entityType).toBe("pred:client-relationship");
  });

  it("survives normalizeFindings inside a real detect() call", async () => {
    // Core is the judge of whether these offsets are right, and it throws
    // rather than warning.
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(predicateIr())),
      provider: "claude",
      text: MSG,
      config: { tier0: false, tier1: false, tier2: true },
      engines: { tier2: new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET) },
    });
    expect(result.findings.map((f) => f.text)).toEqual([QUOTE]);
    // The action the shadow entityType resolves to, which is the whole reason
    // for naming a shadow rather than a bare predicate id.
    expect(result.findings[0]!.action).toBe("redact");
  });

  it("drops a finding naming a predicate the IR does not declare", async () => {
    // A model will invent predicate ids. Passing one through makes core throw
    // and loses the whole message, including the findings that were fine.
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [hit(QUOTE, 0.9, "not-a-predicate"), hit()] }),
      BUDGET,
    );
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(found.map((f) => f.entityType)).toEqual(["pred:client-relationship"]);
    expect(judge.stats.unknownPredicates).toBe(1);
  });

  it("drops a finding whose quote does not resolve, and counts it", async () => {
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [hit("text that is not in the message")] }),
      BUDGET,
    );
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(found).toEqual([]);
    expect(judge.stats.unresolvedQuotes).toBe(1);
  });

  it("records rung 1 for a verbatim quote, and rung 2 for nothing", async () => {
    // A first-class metric, not a detail: an arm whose findings mostly resolve
    // at rung 2 is reporting weaker evidence than one resolving at rung 1, and
    // core's fidelity check cannot tell them apart. `rung1 + rung2 === 1` --
    // the assertion this replaces -- cannot tell the two apart either, which
    // is the entire thing the field exists to say.
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    await judge.judge(whole(), predicateIr(), []);
    expect(judge.stats.rung1).toBe(1);
    expect(judge.stats.rung2).toBe(0);
  });

  it("records rung 2 for a quote that only resolves after peeling", async () => {
    // The other half of the metric. This quote is not in the message; the
    // ladder peels the invented tail off and lands on a unique prefix, which is
    // weaker evidence and has to be reported as such.
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [hit("the Northwind Traders renewal package")] }),
      BUDGET,
    );
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(judge.stats.rung2).toBe(1);
    expect(judge.stats.rung1).toBe(0);
    expect(MSG.slice(found[0]!.start, found[0]!.end)).toBe("the Northwind Traders renewal");
    // `text` is the SLICE, never the model's quote. This is the one assertion
    // that separates them: everywhere the model quotes verbatim the two strings
    // are equal, so only a peeled quote can tell a pass-through apart from a
    // re-derivation.
    expect(found[0]!.text).toBe("the Northwind Traders renewal");
    expect(found[0]!.text).not.toBe("the Northwind Traders renewal package");
  });

  it("survives normalizeFindings when the span came off rung 2", async () => {
    // Core throws when text !== message.slice(start, end), and a passed-through
    // model quote fails that check for every span the ladder had to peel --
    // losing the whole message, not the one finding.
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(predicateIr())),
      provider: "claude",
      text: MSG,
      config: { tier0: false, tier1: false, tier2: true },
      engines: {
        tier2: new WebLlmJudge(
          fakeEngine({ findings: [hit("the Northwind Traders renewal package")] }),
          BUDGET,
        ),
      },
    });
    expect(result.findings.map((f) => f.text)).toEqual(["the Northwind Traders renewal"]);
  });

  it("emits offsets absolute into the MESSAGE when the segment starts late", async () => {
    // The same failure Plan 4 spent a task on. The judge sees a segment; core
    // demands offsets into the whole message.
    const prefix = "intro line\n";
    const wholeText = prefix + MSG;
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    const found = await judge.judge(whole(MSG, prefix.length), predicateIr(), []);
    expect(wholeText.slice(found[0]!.start, found[0]!.end)).toBe(QUOTE);
    expect(found[0]!.text).toBe(QUOTE);
  });

  it("returns no findings and spends no call when the IR declares no predicates", async () => {
    // A policy with no semanticPredicates is legitimate. Calling a 2 GB model
    // to ask about nothing costs seconds per message.
    const engine = fakeEngine({ findings: [] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(whole(), predicateIr({ predicates: [] }), []);
    expect(found).toEqual([]);
    expect(engine.calls).toHaveLength(0);
  });

  it("fails closed on an unparseable response after exactly one repair", async () => {
    // The spec's rule: schema-validated, ONE repair retry, then fail closed --
    // a model that will not emit valid JSON must never silently pass text
    // through. The call count is the assertion that makes `repairAttempts: 1`
    // mean "it retried" rather than "it counted".
    const engine = fakeEngine({ raw: "not json at all" });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(engine.calls).toHaveLength(2);
    expect(judge.stats.repairAttempts).toBe(1);
    expect(judge.stats.failedClosed).toBe(1);
    expect(found).toEqual([]);
  });

  it("keeps the findings when the repair retry succeeds", async () => {
    // The other side of the same counter. Without this, an implementation that
    // never actually reads the second response passes the fail-closed test.
    const engine = fakeEngine({
      script: [{ raw: "{ oh no" }, { findings: [hit()] }],
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(engine.calls).toHaveLength(2);
    expect(judge.stats.repairAttempts).toBe(1);
    expect(judge.stats.failedClosed).toBe(0);
    expect(found.map((f) => f.text)).toEqual([QUOTE]);
  });

  it("tells the repair call what was wrong with the first answer", async () => {
    const engine = fakeEngine({ script: [{ raw: "{ oh no" }, { findings: [hit()] }] });
    await new WebLlmJudge(engine, BUDGET).judge(whole(), predicateIr(), []);
    const repair = engine.promptOf(1);
    expect(repair).not.toBe(engine.promptOf(0));
    expect(repair).toMatch(/could not be parsed/i);
  });

  it("names the model that ANSWERED as the finding's source, not the one requested", async () => {
    // `source` is what a bake-off row attributes a finding to. The requested id
    // is intent; the completion's `model` is what served the call, and Task 5
    // left `loadedModelId` off the seam so that reaching for it is a compile
    // error rather than a warning.
    const engine = fakeEngine({
      findings: [hit()],
      requestedModelId: "Qwen3.5-2B-q4f16_1-MLC",
      model: "Phi-4-mini-instruct-q4f16_1-MLC",
    });
    const found = await new WebLlmJudge(engine, BUDGET).judge(whole(), predicateIr(), []);
    expect(found[0]!.source).toBe("Phi-4-mini-instruct-q4f16_1-MLC");
    expect(found[0]!.source).not.toBe(engine.requestedModelId);
  });

  it("drops a repeated finding and counts it", async () => {
    // Models restate a finding, and Plan 5's own probe corpus has a model
    // looping one finding until the token budget ran out. Two findings over the
    // same span are one piece of evidence counted twice, and they would reach
    // core's merge as a self-overlapping cluster.
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit(), hit(), hit(QUOTE, 0.4)] }), BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(found).toHaveLength(1);
    expect(judge.stats.duplicatesDropped).toBe(2);
  });

  it("does not treat the NEXT message's findings as duplicates of this one's", async () => {
    // Offsets repeat across messages -- [18, 43) is [18, 43) in every message.
    // De-duplication state that outlives one judge() call silently deletes the
    // second message's findings, while the counters keep accumulating as the
    // bake-off needs them to.
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    await judge.judge(whole(), predicateIr(), []);
    const second = await judge.judge(whole(), predicateIr(), []);
    expect(second).toHaveLength(1);
    expect(judge.stats.duplicatesDropped).toBe(0);
    expect(judge.stats.rung1).toBe(2);
  });

  it("does not repair an ABORTED call, and does not file it as truncation", async () => {
    // Task 3 measured the state that produces this: after an interrupt the
    // engine's flag stays set and every later call returns instantly with an
    // empty body and finish_reason "abort". A repair retry spends a second call
    // on an engine that cannot answer, and counting it as truncation tells the
    // bake-off this model is too verbose for its token budget.
    const engine = fakeEngine({ raw: "", finishReason: "abort" });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(engine.calls).toHaveLength(1);
    expect(judge.stats.repairAttempts).toBe(0);
    expect(judge.stats.abortedResponses).toBe(1);
    expect(judge.stats.truncatedResponses).toBe(0);
    expect(judge.stats.failedClosed).toBe(1);
    expect(found).toEqual([]);
  });

  it("DOES repair a truncated call, and counts the truncation", async () => {
    // The contrast that makes the previous test mean something: a truncated
    // answer is a long answer, and asking again is the right response to it.
    const engine = fakeEngine({
      script: [
        { raw: '{"findings":[{"predicateId":"client-relationship","quote":"Northwind Trad', finishReason: "length" },
        { findings: [hit()] },
      ],
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(engine.calls).toHaveLength(2);
    expect(judge.stats.truncatedResponses).toBe(1);
    expect(judge.stats.abortedResponses).toBe(0);
    expect(judge.stats.repairAttempts).toBe(1);
    expect(found.map((f) => f.text)).toEqual([QUOTE]);
  });

  it("keeps what it collected when a later segment blows the budget, and stops", async () => {
    // Spec 5.3: an over-budget tier-2 run degrades to the lower tiers' findings
    // rather than losing the message. Continuing to the third segment would
    // spend another full budget on an engine that just proved it is too slow.
    const other = "Renew the Contoso Industries agreement this quarter.";
    const engine = fakeEngine({
      script: [{ findings: [hit()] }, { throws: new DeadlineExpired("budget", 30_000, true) }],
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(
      [...whole(), ...whole(other, 100), ...whole(other, 300)],
      predicateIr(),
      [],
    );
    expect(found.map((f) => f.text)).toEqual([QUOTE]);
    expect(engine.calls).toHaveLength(2);
    expect(judge.stats.deadlineExpiries).toBe(1);
    expect(judge.stats.callerAborts).toBe(0);
  });

  it("counts a caller abort separately from a budget expiry", async () => {
    // Different events with different fixes: a blown budget says the model is
    // too slow for this arm, a caller abort says nobody waited for the answer.
    // One counter serving both puts a fabricated timeout in every cancelled row.
    const engine = fakeEngine({ throws: new DeadlineExpired("aborted", 30_000, false) });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(found).toEqual([]);
    expect(judge.stats.callerAborts).toBe(1);
    expect(judge.stats.deadlineExpiries).toBe(0);
  });

  it("asks for a quote of at least as many words as the ladder will accept", async () => {
    // Task 4's review flagged the drift this prevents: the span ladder refuses
    // a candidate shorter than MINIMUM_CANDIDATE_WORDS, so a prompt asking for
    // fewer produces quotes the ladder throws away and blames on the model.
    // The number is read back OUT of the prompt and compared to the constant,
    // rather than the constant being interpolated into the expectation -- a
    // prompt saying "three" fails here, and so does one hardcoding 3 after the
    // ladder's floor moves.
    const engine = fakeEngine({ findings: [] });
    await new WebLlmJudge(engine, BUDGET).judge(whole(), predicateIr(), []);
    const asked = /at least (\S+) words/.exec(engine.promptOf(0));
    expect(asked).not.toBeNull();
    expect(Number(asked![1])).toBe(MINIMUM_CANDIDATE_WORDS);
  });

  it("puts every predicate's id and text in the prompt, and no entityType examples", async () => {
    const engine = fakeEngine({ findings: [] });
    const ir = predicateIr({
      predicates: [
        { id: "client-relationship", nlPredicate: "names a client" },
        { id: "unreleased-financials", nlPredicate: "discusses unreleased financials" },
      ],
    });
    await new WebLlmJudge(engine, BUDGET).judge(whole(), ir, []);
    const prompt = engine.promptOf(0);
    expect(prompt).toContain("client-relationship");
    expect(prompt).toContain("discusses unreleased financials");
  });

  it("never restates a prior finding's TEXT in the prompt", async () => {
    // Standing rule: a gold value must not reach the model's own prompt. A
    // prior finding's `text` IS the value tier 0/1 detected, and echoing it
    // hands the model an answer key it can score against for free. The label
    // and the count carry the context without the value.
    const prior: Finding[] = [
      {
        start: 0,
        end: MSG.length,
        // Deliberately not a substring of the segment: if the implementation
        // interpolates `text`, this shows up in the prompt and nothing else can
        // have put it there.
        text: "SENTINEL-PRIOR-VALUE",
        entityType: "client-name",
        severity: "medium",
        tier: 1,
        source: "gliner-pii-base",
        confidence: 0.9,
      },
    ];
    const engine = fakeEngine({ findings: [] });
    await new WebLlmJudge(engine, BUDGET).judge(whole(), predicateIr(), prior);
    const prompt = engine.promptOf(0);
    expect(prompt).not.toContain("SENTINEL-PRIOR-VALUE");
    // ... and the priors did reach the prompt, so the assertion above is not
    // passing merely because they were dropped on the floor.
    expect(prompt).toContain("client-name");
  });

  it("summarises only the priors that overlap the segment it is asking about", async () => {
    const prior: Finding[] = [
      {
        start: 500,
        end: 510,
        text: "elsewhere",
        entityType: "client-name",
        severity: "medium",
        tier: 1,
        source: "gliner-pii-base",
        confidence: 0.9,
      },
    ];
    const engine = fakeEngine({ findings: [] });
    await new WebLlmJudge(engine, BUDGET).judge(whole(), predicateIr(), prior);
    expect(engine.promptOf(0)).not.toContain("client-name");
  });

  it("refuses an IR whose predicate has no shadow entityType, before spending a call", async () => {
    // A predicate the compiler never minted a shadow for is a policy clause
    // that cannot reach an action: every finding naming it would be thrown out
    // by normalizeFindings, so the clause silently does nothing. That is the
    // failure class this project exists to prevent, so it is loud.
    const engine = fakeEngine({ findings: [hit()] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const ir = predicateIr({
      predicates: [{ id: "client-relationship", nlPredicate: "names a client", shadow: false }],
    });
    await expect(judge.judge(whole(), ir, [])).rejects.toThrow(/pred:client-relationship/);
    expect(engine.calls).toHaveLength(0);
  });

  it("refuses a budget setTimeout would silently reinterpret", async () => {
    // Infinity is the natural spelling of "no budget" and produces a 1 ms
    // deadline, not an infinite one. Caught at construction rather than on the
    // first engine call, which is many segments later in a bake-off arm.
    for (const budgetMs of [Number.POSITIVE_INFINITY, Number.NaN, 0, -1, 2_147_483_648]) {
      expect(() => new WebLlmJudge(fakeEngine(), { budgetMs })).toThrow(/budgetMs/);
    }
    expect(() => new WebLlmJudge(fakeEngine(), { budgetMs: 2_147_483_647 })).not.toThrow();
  });

  it("refuses a budget that is not a number at all", async () => {
    // The range check alone accepts "5000": a string compares fine against both
    // bounds, so `Number.isFinite` is the only clause that catches it. A judge
    // is built from an eval driver's JSON config, where a quoted number is one
    // typo away and TypeScript is not present to object.
    expect(
      () => new WebLlmJudge(fakeEngine(), { budgetMs: "5000" as unknown as number }),
    ).toThrow(/budgetMs/);
  });

  it("hands the engine the budget it was configured with, and the caller's signal", async () => {
    // A second, non-default value on purpose: a test that only ever exercises
    // the value the implementation might have hardcoded cannot tell "reads the
    // config" from "hardcodes the default".
    const engine = fakeEngine({ findings: [] });
    const controller = new AbortController();
    await new WebLlmJudge(engine, { budgetMs: 1234 }).judge(
      whole(),
      predicateIr(),
      [],
      controller.signal,
    );
    expect(engine.calls[0]!.opts.budgetMs).toBe(1234);
    expect(engine.calls[0]!.opts.signal).toBe(controller.signal);
  });

  it("asks once per segment", async () => {
    const engine = fakeEngine({ findings: [] });
    await new WebLlmJudge(engine, BUDGET).judge(
      [...whole(), ...whole("Renew the Contoso Industries agreement.", 100)],
      predicateIr(),
      [],
    );
    expect(engine.calls).toHaveLength(2);
    expect(engine.promptOf(0)).toContain(MSG);
    expect(engine.promptOf(1)).toContain("Contoso Industries");
  });

  it("takes severity from the IR and confidence from the model", async () => {
    // Severity is policy, and a model does not get to set it -- core re-derives
    // it anyway, so a hardcoded one here would be invisible until a policy
    // changed it. Confidence is the model's own and is passed through.
    const ir = predicateIr({
      predicates: [{ id: "client-relationship", nlPredicate: "names a client", severity: "low" }],
    });
    const found = await new WebLlmJudge(fakeEngine({ findings: [hit(QUOTE, 0.42)] }), BUDGET).judge(
      whole(),
      ir,
      [],
    );
    expect(found[0]!.severity).toBe("low");
    expect(found[0]!.confidence).toBe(0.42);
    expect(found[0]!.tier).toBe(2);
  });

  it("drops a finding whose confidence the schema refuses, without losing the good one", async () => {
    // A model answering on a percentage scale. Clamping 95 to 1.0 would turn a
    // misunderstanding into a maximally confident finding; the schema rejects
    // the whole response, so the repair retry is what recovers the segment.
    const engine = fakeEngine({
      script: [{ findings: [hit(QUOTE, 95)] }, { findings: [hit(QUOTE, 0.5)] }],
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judge.judge(whole(), predicateIr(), []);
    expect(judge.stats.repairAttempts).toBe(1);
    expect(found.map((f) => f.confidence)).toEqual([0.5]);
  });

  it("starts every counter at zero", async () => {
    const judge = new WebLlmJudge(fakeEngine(), BUDGET);
    expect(Object.values(judge.stats).every((v) => v === 0)).toBe(true);
  });
});
