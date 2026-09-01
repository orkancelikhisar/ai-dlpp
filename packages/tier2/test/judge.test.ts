import { describe, expect, it, vi } from "vitest";
import type { Finding, JudgeRequest, PolicyIr, Segment } from "@sih/core";
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

/**
 * A `JudgeRequest` for segments this file already builds by hand.
 *
 * `text` is REBUILT from the segments rather than defaulted to MSG, because the
 * seam's contract is that every segment offset indexes into `text` -- several
 * tests here place a segment at an offset inside a longer message, and a
 * hardcoded MSG would hand the judge a request no orchestrator could produce.
 * Gaps between segments are filled with spaces: invented, and invented on
 * purpose, since no test asserts on what is between two segments.
 */
const req = (
  segments: Segment[],
  ir: PolicyIr,
  priorFindings: Finding[] = [],
  over: Partial<JudgeRequest> = {},
): JudgeRequest => {
  const end = segments.reduce((max, s) => Math.max(max, s.end), 0);
  let text = " ".repeat(end);
  for (const s of segments) text = text.slice(0, s.start) + s.text + text.slice(s.end);
  return { text, segments, ir, priorFindings, budgetMs: 30_000, ...over };
};

/**
 * One run's FINDINGS, for the many tests below that assert only on those.
 *
 * The seam returns a verdict now -- findings plus what the judge says about its
 * own run -- and unwrapping it at every call site would bury the tests that do
 * assert on the rest. Those call `judge()` directly.
 */
const judged = async (
  instance: WebLlmJudge,
  segments: Segment[],
  ir: PolicyIr,
  priorFindings: Finding[] = [],
): Promise<Finding[]> => (await instance.judge(req(segments, ir, priorFindings))).findings;

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
    const found = await judged(judge, whole(), predicateIr(), []);
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
    const found = await judged(judge, whole(), predicateIr(), []);
    expect(found.map((f) => f.entityType)).toEqual(["pred:client-relationship"]);
    expect(judge.stats.unknownPredicates).toBe(1);
  });

  it("drops a quote that is nowhere in the segment, rather than placing it somewhere", async () => {
    // A model that paraphrases instead of quoting. The alternative to dropping
    // it is a span chosen by something other than the model's own evidence,
    // which slices cleanly and so passes every check core makes.
    const judge = new WebLlmJudge(
      fakeEngine({ findings: [hit("text that is not in the message")] }),
      BUDGET,
    );
    const found = await judged(judge, whole(), predicateIr(), []);
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
    await judged(judge, whole(), predicateIr(), []);
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
    const found = await judged(judge, whole(), predicateIr(), []);
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
    const found = await judged(judge, whole(MSG, prefix.length), predicateIr(), []);
    expect(wholeText.slice(found[0]!.start, found[0]!.end)).toBe(QUOTE);
    expect(found[0]!.text).toBe(QUOTE);
  });

  it("returns no findings and spends no call when the IR declares no predicates", async () => {
    // A policy with no semanticPredicates is legitimate. Calling a 2 GB model
    // to ask about nothing costs seconds per message.
    const engine = fakeEngine({ findings: [] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judged(judge, whole(), predicateIr({ predicates: [] }), []);
    expect(found).toEqual([]);
    expect(engine.calls).toHaveLength(0);
    // And it counts NOTHING, rather than filing the segment as skipped. A
    // policy with no semantic clauses is not a run a stop cut short, and the
    // stats header's segment invariant is scoped to calls that reached the
    // loop precisely so this stays distinguishable.
    expect(judge.stats.segmentsSkipped).toBe(0);
    expect(judge.stats.segmentsJudged).toBe(0);
  });

  it("fails closed on an unparseable response after exactly one repair", async () => {
    // The spec's rule: schema-validated, ONE repair retry, then fail closed --
    // a model that will not emit valid JSON must never silently pass text
    // through. The call count is the assertion that makes `repairAttempts: 1`
    // mean "it retried" rather than "it counted".
    const engine = fakeEngine({ raw: "not json at all" });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judged(judge, whole(), predicateIr(), []);
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
    const found = await judged(judge, whole(), predicateIr(), []);
    expect(engine.calls).toHaveLength(2);
    expect(judge.stats.repairAttempts).toBe(1);
    expect(judge.stats.failedClosed).toBe(0);
    expect(found.map((f) => f.text)).toEqual([QUOTE]);
  });

  it("tells the repair call what was wrong with the first answer", async () => {
    const engine = fakeEngine({ script: [{ raw: "{ oh no" }, { findings: [hit()] }] });
    await judged(new WebLlmJudge(engine, BUDGET), whole(), predicateIr(), []);
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
    const found = await judged(new WebLlmJudge(engine, BUDGET), whole(), predicateIr(), []);
    expect(found[0]!.source).toBe("Phi-4-mini-instruct-q4f16_1-MLC");
    expect(found[0]!.source).not.toBe(engine.requestedModelId);
  });

  it("drops a repeated finding and counts it", async () => {
    // Models restate a finding, and Plan 5's own probe corpus has a model
    // looping one finding until the token budget ran out. Two findings over the
    // same span are one piece of evidence counted twice, and they would reach
    // core's merge as a self-overlapping cluster.
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit(), hit(), hit(QUOTE, 0.4)] }), BUDGET);
    const found = await judged(judge, whole(), predicateIr(), []);
    expect(found).toHaveLength(1);
    expect(judge.stats.duplicatesDropped).toBe(2);
  });

  it("does not treat the NEXT message's findings as duplicates of this one's", async () => {
    // Offsets repeat across messages -- [18, 43) is [18, 43) in every message.
    // De-duplication state that outlives one judge() call silently deletes the
    // second message's findings, while the counters keep accumulating as the
    // bake-off needs them to.
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    await judged(judge, whole(), predicateIr(), []);
    const second = await judged(judge, whole(), predicateIr(), []);
    expect(second).toHaveLength(1);
    expect(judge.stats.duplicatesDropped).toBe(0);
    expect(judge.stats.rung1).toBe(2);
  });

  it("stops the whole run on a latched engine, and does not file it as truncation", async () => {
    // Task 3 measured the state that produces this: after an interrupt the
    // engine's flag stays set and every later call returns instantly with an
    // empty body and finish_reason "abort". A repair retry spends a second call
    // on an engine that cannot answer, and counting it as truncation tells the
    // bake-off this model is too verbose for its token budget.
    //
    // THREE segments, not one, and that is the assertion. With one segment
    // `return findings` and `break` are observationally identical, so nothing
    // pinned that the run stops rather than spending two more calls
    // manufacturing clean segments out of an engine that answers everything
    // instantly and emptily.
    const engine = fakeEngine({ raw: "", finishReason: "abort" });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judged(judge, 
      [...whole(), ...whole(MSG, 200), ...whole(MSG, 400)],
      predicateIr(),
      [],
    );
    expect(engine.calls).toHaveLength(1);
    expect(judge.stats.repairAttempts).toBe(0);
    expect(judge.stats.abortedResponses).toBe(1);
    expect(judge.stats.truncatedResponses).toBe(0);
    expect(judge.stats.failedClosed).toBe(1);
    expect(judge.stats.segmentsSkipped).toBe(2);
    expect(judge.stats.segmentsJudged).toBe(0);
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
    const found = await judged(judge, whole(), predicateIr(), []);
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
    const found = await judged(judge, 
      [...whole(), ...whole(other, 100), ...whole(other, 300)],
      predicateIr(),
      [],
    );
    expect(found.map((f) => f.text)).toEqual([QUOTE]);
    expect(engine.calls).toHaveLength(2);
    expect(judge.stats.deadlineExpiries).toBe(1);
    expect(judge.stats.callerAbortsMidGeneration).toBe(0);
    expect(judge.stats.callerAbortsWhileQueued).toBe(0);
  });

  it("counts every segment an early stop abandoned, not just the stop", async () => {
    // `deadlineExpiries` is 1 whether the budget blew on segment 2 of 3 or on
    // segment 39 of 40 -- it is bounded at 1 per call by construction. Without
    // segmentsSkipped a scorer computing findings-per-segment from an
    // early-stopped run divides by a denominator that never happened, and
    // nothing in the record says so.
    const engine = fakeEngine({
      script: [{ findings: [hit()] }, { throws: new DeadlineExpired("budget", 30_000, true) }],
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    const segments = [...whole(), ...whole(MSG, 200), ...whole(MSG, 400), ...whole(MSG, 600)];
    await judged(judge, segments, predicateIr(), []);
    expect(judge.stats.segmentsJudged).toBe(1);
    // The segment whose own call blew the budget, plus the two never asked about.
    expect(judge.stats.segmentsSkipped).toBe(3);
    expect(judge.stats.deadlineExpiries).toBe(1);
    // The invariant the header promises: nothing falls between the counters.
    const s = judge.stats;
    expect(s.segmentsJudged + s.failedClosed + s.segmentsSkipped).toBe(segments.length);
  });

  it("counts a mid-generation caller abort apart from a queued one", async () => {
    // `cancel.ts` carries `interrupted` precisely because these are different
    // events: one interrupted a running generation, the other withdrew a call
    // that never started and therefore spent no model time at all. A latency
    // row that attributes a queued abort to the model is reporting a number the
    // model never produced.
    const mid = new WebLlmJudge(
      fakeEngine({ throws: new DeadlineExpired("aborted", 30_000, true) }),
      BUDGET,
    );
    expect(await judged(mid, whole(), predicateIr(), [])).toEqual([]);
    expect(mid.stats.callerAbortsMidGeneration).toBe(1);
    expect(mid.stats.callerAbortsWhileQueued).toBe(0);

    const queued = new WebLlmJudge(
      fakeEngine({ throws: new DeadlineExpired("aborted", 30_000, false) }),
      BUDGET,
    );
    expect(await judged(queued, whole(), predicateIr(), [])).toEqual([]);
    expect(queued.stats.callerAbortsWhileQueued).toBe(1);
    expect(queued.stats.callerAbortsMidGeneration).toBe(0);

    // And neither is a budget expiry, which is the conflation that would put a
    // fabricated timeout in every cancelled row.
    expect(mid.stats.deadlineExpiries).toBe(0);
    expect(queued.stats.deadlineExpiries).toBe(0);
  });

  it("propagates an engine error that is NOT a stop, rather than degrading it", async () => {
    // `engine.ts` throws on a response with no choices instead of folding it
    // into "" -- the comment there says treating it as an empty answer "would
    // report it as a clean segment". Catching it here would restore exactly
    // that false negative, AND file it as a caller abort, because a plain Error
    // has no `reason` and the else-branch owns everything that is not "budget".
    const boom = new Error(
      `tier-2 engine "x" returned a response with no choices; treating that as an ` +
        `empty answer would report it as a clean segment`,
    );
    const judge = new WebLlmJudge(fakeEngine({ throws: boom }), BUDGET);
    await expect(judged(judge, whole(), predicateIr(), [])).rejects.toThrow(/no choices/);
    expect(judge.stats.callerAbortsWhileQueued).toBe(0);
    expect(judge.stats.callerAbortsMidGeneration).toBe(0);
    expect(judge.stats.deadlineExpiries).toBe(0);
  });

  it("asks for the word count the ladder's PEEL rung enforces, as a number", async () => {
    // What the floor actually gates, since the prompt used to tell the model
    // otherwise: MINIMUM_CANDIDATE_WORDS bounds rung 2 only. Rung 1 has no word
    // check -- measured, the one-word quote "Northwind" resolves at rung 1 --
    // so a short quote is not refused for being short. Asking for the whole
    // clause is still right, because a longer quote is likelier to occur
    // exactly once and leaves the peel somewhere to descend to.
    //
    // The number is read back OUT of the prompt rather than interpolated into
    // the expectation, so a prompt spelling it "three" fails here too.
    const engine = fakeEngine({ findings: [] });
    await judged(new WebLlmJudge(engine, BUDGET), whole(), predicateIr(), []);
    const asked = /at least (\S+) words/.exec(engine.promptOf(0));
    expect(asked).not.toBeNull();
    expect(Number(asked![1])).toBe(MINIMUM_CANDIDATE_WORDS);
  });

  it("follows the ladder's floor when it MOVES, rather than restating today's value", async () => {
    // The standing rule this project has broken twice: a test that only ever
    // exercises the DEFAULT cannot tell "reads the constant" from "hardcodes
    // the literal". Against the shipped floor of 3, a prompt writing out
    // `at least 3 words` passes the test above -- and so does moving the floor
    // to 4 while the prompt keeps interpolating, since the two sides of that
    // assertion would move together. Only a second value separates them.
    vi.resetModules();
    vi.doMock("../src/spans.js", async () => {
      const actual = await vi.importActual<typeof import("../src/spans.js")>("../src/spans.js");
      return { ...actual, MINIMUM_CANDIDATE_WORDS: 7 };
    });
    try {
      const { WebLlmJudge: RebuiltJudge } = await import("../src/judge.js");
      const engine = fakeEngine({ findings: [] });
      await judged(new RebuiltJudge(engine, BUDGET), whole(), predicateIr(), []);
      expect(engine.promptOf(0)).toContain("at least 7 words");
      // Positive control: 7 is the mocked floor and not the real one, so this
      // cannot be passing because the prompt happens to say 7 anyway.
      expect(MINIMUM_CANDIDATE_WORDS).not.toBe(7);
    } finally {
      vi.doUnmock("../src/spans.js");
      vi.resetModules();
    }
  });

  it("puts every predicate's id and text in the prompt, and no entityType examples", async () => {
    // The prompt's own doc says examples and counterExamples are deliberately
    // absent: they are authored strings, and in one corpus the only tier-1 gold
    // value was also the IR's `examples` entry -- a model handed the answer
    // scores without doing the work. `predicateIr` gives the AUTHORED entityType
    // a sentinel example list so this has something to catch; with the empty
    // lists it used to carry, an implementation that interpolated
    // `entity.examples` straight into the prompt would pass. Shadow entityTypes
    // stay empty here because that is what the compiler mints.
    const engine = fakeEngine({ findings: [] });
    const ir = predicateIr({
      predicates: [
        { id: "client-relationship", nlPredicate: "names a client" },
        { id: "unreleased-financials", nlPredicate: "discusses unreleased financials" },
      ],
    });
    await judged(new WebLlmJudge(engine, BUDGET), whole(), ir, []);
    const prompt = engine.promptOf(0);
    expect(prompt).toContain("client-relationship");
    expect(prompt).toContain("discusses unreleased financials");
    for (const sentinel of ir.entityTypes.flatMap((e) => [...e.examples, ...e.counterExamples])) {
      expect(prompt).not.toContain(sentinel);
    }
    // ... and the IR really did carry some, so the loop above is not vacuous.
    expect(ir.entityTypes.flatMap((e) => e.examples)).toContain("SENTINEL-AUTHORED-EXAMPLE");
    expect(ir.entityTypes.flatMap((e) => e.counterExamples)).toContain(
      "SENTINEL-AUTHORED-COUNTEREXAMPLE",
    );
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
    await judged(new WebLlmJudge(engine, BUDGET), whole(), predicateIr(), prior);
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
    await judged(new WebLlmJudge(engine, BUDGET), whole(), predicateIr(), prior);
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
    await expect(judged(judge, whole(), ir, [])).rejects.toThrow(/pred:client-relationship/);
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
      req(whole(), predicateIr(), [], { signal: controller.signal }),
    );
    expect(engine.calls[0]!.opts.budgetMs).toBe(1234);
    expect(engine.calls[0]!.opts.signal).toBe(controller.signal);
  });

  it("shows each segment only its own text, so one cannot answer for another", async () => {
    // One call per segment, and each prompt carries exactly one passage. A
    // single call over the concatenation would let a quote resolve against text
    // the finding's segment does not contain, which is the mis-location
    // `spans.ts` exists to make impossible.
    const engine = fakeEngine({ findings: [] });
    await judged(new WebLlmJudge(engine, BUDGET), 
      [...whole(), ...whole("Renew the Contoso Industries agreement.", 100)],
      predicateIr(),
      [],
    );
    expect(engine.calls).toHaveLength(2);
    expect(engine.promptOf(0)).toContain(MSG);
    expect(engine.promptOf(0)).not.toContain("Contoso Industries");
    expect(engine.promptOf(1)).toContain("Contoso Industries");
    expect(engine.promptOf(1)).not.toContain(MSG);
  });

  it("emits findings from TWO segments in one call, each offset into the message", async () => {
    // Nothing else covers two segments both producing findings: the other
    // multi-segment tests either return nothing or stop on segment 2. So an
    // implementation that keyed dedup on segment-RELATIVE offsets, or that
    // forgot `segment.start` on any but the first, went unnoticed.
    //
    // The two quotes sit at the SAME offset within their own segments and are
    // the same length, which is what makes a relative key collide: keyed
    // relatively, the second finding is dropped as a duplicate of the first.
    const second = "Please review the Northwind Traders renewal again.";
    expect(second.indexOf(QUOTE)).toBe(MSG.indexOf(QUOTE));
    const engine = fakeEngine({ findings: [hit()] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const message = `${MSG}\n${second}`;
    const found = await judged(judge, 
      [...whole(), ...whole(second, MSG.length + 1)],
      predicateIr(),
      [],
    );
    expect(found.map((f) => [f.start, f.end])).toEqual([
      [18, 43],
      [77, 102],
    ]);
    for (const f of found) expect(message.slice(f.start, f.end)).toBe(f.text);
    expect(judge.stats.duplicatesDropped).toBe(0);
    expect(judge.stats.segmentsJudged).toBe(2);
    expect(judge.stats.segmentsSkipped).toBe(0);
  });

  it("emits offsets in UTF-16 code units when the prefix is astral, not code points", async () => {
    // Plan 4 put an emoji fixture in the tier-1 corpus so any UTF-16 offset bug
    // would surface there rather than here; this reciprocates. The prefix is
    // built so all three plausible answers differ -- 35 UTF-16 code units, 33
    // code points, 46 UTF-8 bytes -- because a prefix of pure ASCII (which is
    // what the other late-segment test uses) makes every convention agree and
    // pins nothing. Core slices with `String.prototype.slice`, which is UTF-16.
    //
    // Spelled with escapes rather than literal characters on purpose: an editor
    // that NFC-normalised the combining diaeresis into a precomposed letter would
    // quietly change all three counts and take the test's whole point with it.
    const prefix =
      "Kickoff \u{1F680}\u{1F3FD} caf\u{E9} nai\u{308}ve \u{4E2D}\u{6587} notes\n\n";
    expect(prefix.length).toBe(35);
    expect([...prefix].length).toBe(33);
    expect(Buffer.byteLength(prefix, "utf8")).toBe(45);
    const message = prefix + MSG;
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    const found = await judged(judge, whole(MSG, prefix.length), predicateIr(), []);
    expect(found[0]!.start).toBe(53);
    // The code-point answer, spelled out so the assertion above cannot be read
    // as "whatever the implementation produced".
    expect(found[0]!.start).not.toBe(51);
    expect(message.slice(found[0]!.start, found[0]!.end)).toBe(QUOTE);
    expect(found[0]!.text).toBe(QUOTE);

    // The other half: astral characters INSIDE the segment, before the quote.
    // The prefix above only pins that `segment.start` is added unchanged; this
    // pins the offset the ladder returned, which an implementation counting
    // code points would report one short per astral character.
    const inner = `Update \u{1F680}: ${MSG}`;
    const innerJudge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    const innerFound = await judged(innerJudge, 
      [{ kind: "prose", start: prefix.length, end: prefix.length + inner.length, text: inner }],
      predicateIr(),
      [],
    );
    // 35 units of prefix + 11 units of "Update <rocket>: " + 18 into MSG.
    expect(innerFound[0]!.start).toBe(64);
    expect(innerFound[0]!.start).not.toBe(63);
    expect((prefix + inner).slice(innerFound[0]!.start, innerFound[0]!.end)).toBe(QUOTE);
  });

  it("keeps two predicates' findings over the SAME span, one per entityType", async () => {
    // The dedup key is `start:end:entityType`, and the entityType component is
    // the load-bearing part: two predicates can be satisfied by one clause, and
    // core resolves an ACTION per entityType. Dropping the component silently
    // loses one predicate's finding and with it whatever action that predicate
    // carried, while the counters report a duplicate the model never sent.
    const ir = predicateIr({
      predicates: [
        { id: "client-relationship", nlPredicate: "names a client", severity: "high" },
        { id: "renewal-terms", nlPredicate: "discusses renewal terms", severity: "low" },
      ],
    });
    const judge = new WebLlmJudge(
      fakeEngine({
        findings: [hit(QUOTE, 0.9, "client-relationship"), hit(QUOTE, 0.8, "renewal-terms")],
      }),
      BUDGET,
    );
    const found = await judged(judge, whole(), ir, []);
    expect(found.map((f) => f.entityType)).toEqual([
      "pred:client-relationship",
      "pred:renewal-terms",
    ]);
    expect(found.map((f) => [f.start, f.end])).toEqual([
      [18, 43],
      [18, 43],
    ]);
    expect(judge.stats.duplicatesDropped).toBe(0);
  });

  it("takes severity from the IR and confidence from the model", async () => {
    // Severity is policy, and a model does not get to set it -- core re-derives
    // it anyway, so a hardcoded one here would be invisible until a policy
    // changed it. Confidence is the model's own and is passed through.
    const ir = predicateIr({
      predicates: [{ id: "client-relationship", nlPredicate: "names a client", severity: "low" }],
    });
    const found = await judged(new WebLlmJudge(fakeEngine({ findings: [hit(QUOTE, 0.42)] }), BUDGET), 
      whole(),
      ir,
      [],
    );
    expect(found[0]!.severity).toBe("low");
    expect(found[0]!.confidence).toBe(0.42);
    expect(found[0]!.tier).toBe(2);
  });

  it("rejects the WHOLE response over one out-of-range confidence, and repairs it", async () => {
    // A model answering on a percentage scale. Clamping 95 to 1.0 would turn a
    // misunderstanding into a maximally confident finding, so the schema
    // refuses instead -- and it refuses the response, not the finding, because
    // `JudgeResponseSchema` validates the parsed body as one object. There is
    // deliberately no per-finding salvage: the repair retry is the whole
    // recovery, and this name used to promise a granularity that does not
    // exist.
    const engine = fakeEngine({
      script: [
        { findings: [hit(QUOTE, 95), hit("before Friday", 0.5)] },
        { findings: [hit(QUOTE, 0.5)] },
      ],
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    const found = await judged(judge, whole(), predicateIr(), []);
    expect(judge.stats.repairAttempts).toBe(1);
    // The in-range sibling from the first response is gone too, not kept.
    expect(found.map((f) => f.text)).toEqual([QUOTE]);
    expect(found.map((f) => f.confidence)).toEqual([0.5]);
  });

  it("records what each ANSWERED call cost, per call rather than per message", async () => {
    // `WebLlmJudge` is the only holder of a `Tier2Completion` inside a detect()
    // run, so a field it does not copy is unobtainable by the bake-off row that
    // needs it -- and `engine.ts` passes `usage` through verbatim precisely so
    // it can be. Per call and not aggregated: a message makes one call per
    // segment, so one finishReason for the message would be a fact about one
    // call reported as a fact about the message.
    const engine = fakeEngine({
      script: [
        {
          findings: [hit()],
          usage: {
            prompt_tokens: 411,
            completion_tokens: 37,
            total_tokens: 448,
            extra: {
              e2e_latency_s: 1.5,
              prefill_tokens_per_s: 900,
              decode_tokens_per_s: 30,
              time_to_first_token_s: 0.42,
              time_per_output_token_s: 0.03,
            },
          },
        },
        { findings: [], finishReason: "length" },
      ],
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    await judged(judge, [...whole(), ...whole(MSG, 200)], predicateIr(), []);
    expect(judge.stats.calls).toEqual([
      {
        finishReason: "stop",
        promptTokens: 411,
        completionTokens: 37,
        ttftMs: 420,
        // The ENGINE's own decode rate, copied rather than derived. Nothing
        // downstream can reconstruct it: a record carries `timings.tier2Ms` for
        // the whole message and no per-call elapsed time, so dividing tokens by
        // anything available there charges the judge's prompt assembly, JSON
        // parse and span ladder to the model. The bake-off's
        // `minDecodeTokPerSec` gate is a floor, so that error kills good arms.
        decodeTokPerSec: 30,
      },
      // The second call reported no usage at all. `undefined` and not 0: a 0
      // would claim the model answered instantly on no prompt tokens.
      {
        finishReason: "length",
        promptTokens: undefined,
        completionTokens: undefined,
        ttftMs: undefined,
        decodeTokPerSec: undefined,
      },
    ]);
  });

  it("passes a poisoned usage number through instead of zeroing it", async () => {
    // `engine.ts` records that an interrupted call leaves NaN or Infinity in
    // `usage.extra`, because every rate there is a division with no zero guard.
    // A `?? 0` or a finite-check that substitutes 0 would report an interrupted
    // call as an instantaneous one, which is the silently-zeroed-NaN defect a
    // sibling task's review already caught once.
    const engine = fakeEngine({
      findings: [],
      usage: {
        prompt_tokens: 400,
        completion_tokens: 0,
        total_tokens: 400,
        extra: {
          e2e_latency_s: 0.1,
          prefill_tokens_per_s: Number.NaN,
          decode_tokens_per_s: Number.NaN,
          time_to_first_token_s: Number.NaN,
          time_per_output_token_s: Number.NaN,
        },
      },
    });
    const judge = new WebLlmJudge(engine, BUDGET);
    await judged(judge, whole(), predicateIr(), []);
    expect(judge.stats.calls[0]!.ttftMs).toBeNaN();
    expect(judge.stats.calls[0]!.completionTokens).toBe(0);
    // VERIFIED in the installed 0.2.84 bundle: `decode_tokens_per_s:
    // completion_tokens / decode_time`, a plain division, so an interrupt that
    // landed before the first token leaves 0/0. A finite-check substituting 0
    // here would report the call as an infinitely slow one and a `?? 25` would
    // report it as passing the floor gate; both are inventions. The driver
    // excludes non-finite rates from the gate instead, which it can only do if
    // the non-finite value survives to the record.
    expect(judge.stats.calls[0]!.decodeTokPerSec).toBeNaN();
  });

  it("exposes every documented counter, all of them at zero, before any call", async () => {
    // `Object.values(stats).every(v => v === 0)` was the assertion here, and it
    // is vacuously true for `{}` -- a stats getter returning an empty object
    // passed it. The key list is what makes the zeroes mean something.
    const judge = new WebLlmJudge(fakeEngine(), BUDGET);
    expect(Object.keys(judge.stats).sort()).toEqual([
      "abortedResponses",
      "callerAbortsMidGeneration",
      "callerAbortsWhileQueued",
      "calls",
      "deadlineExpiries",
      "duplicatesDropped",
      "failedClosed",
      "messageScopeCalls",
      "messageScopeFailedClosed",
      "messageScopeJudged",
      "repairAttempts",
      "rung1",
      "rung2",
      "segmentsJudged",
      "segmentsSkipped",
      "truncatedResponses",
      "unknownPredicates",
      "unresolvedQuotes",
    ]);
    const { calls, ...counters } = judge.stats;
    expect(calls).toEqual([]);
    expect(Object.values(counters)).toEqual(Object.values(counters).map(() => 0));
  });

  it("hands out a SNAPSHOT of stats, not the counters a bake-off row is built from", async () => {
    // The getter's own docblock promises this. A live reference lets a consumer
    // -- or a harness copying `stats` into a record and then tidying it --
    // rewrite numbers that are supposed to be facts about calls that happened.
    // The `calls` array needs the same treatment as the counters: spreading the
    // outer object alone would hand out the judge's own array.
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    await judged(judge, whole(), predicateIr(), []);

    const taken = judge.stats as unknown as Record<string, unknown>;
    taken["rung1"] = 999;
    (taken["calls"] as unknown[]).push({ finishReason: "invented" });
    (judge.stats.calls[0] as unknown as Record<string, unknown>)["promptTokens"] = 999;

    expect(judge.stats.rung1).toBe(1);
    expect(judge.stats.calls).toHaveLength(1);
    expect(judge.stats.calls[0]!.finishReason).toBe("stop");
    expect(judge.stats.calls[0]!.promptTokens).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// The verdict: what the judge says about its own run, beyond its findings.
// ---------------------------------------------------------------------------

describe("judge verdict", () => {
  it("says it evaluated the segment scope, and only that", async () => {
    // The honest answer, and the one the orchestrator turns into a
    // `scope-unjudged` notice when the policy declares a message-scoped
    // predicate. Claiming "message" here would make that notice disappear
    // without a single message-scoped judgement being made.
    const judge = new WebLlmJudge(fakeEngine({ findings: [] }), BUDGET);
    const verdict = await judge.judge(req(whole(), predicateIr()));
    expect(verdict.scopesJudged).toEqual(["segment"]);
  });

  it("claims no scope at all when the policy has no semantic predicates", async () => {
    // Nothing was evaluated, so naming a scope would be a claim about a run
    // that never happened -- and the orchestrator ignores scopes the policy
    // declares no predicate in, so nothing is lost by being accurate.
    const judge = new WebLlmJudge(fakeEngine({ findings: [] }), BUDGET);
    const verdict = await judge.judge(req(whole(), predicateIr({ predicates: [] })));
    expect(verdict.scopesJudged).toEqual([]);
  });

  it("reports nothing degraded when every segment was judged", async () => {
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    const verdict = await judge.judge(req(whole(), predicateIr()));
    expect(verdict.degraded ?? []).toEqual([]);
  });

  it("reports failing closed, not only counting it", async () => {
    // Spec section 7: a body still invalid after one repair is flagged for user
    // review, never a silent pass-through. Before the verdict, "flagged" meant
    // a counter on this object that DetectionResult gave no caller a way to
    // read -- so an unparseable model looked exactly like a clean message.
    const judge = new WebLlmJudge(fakeEngine({ raw: "not json at all" }), BUDGET);
    const verdict = await judge.judge(req(whole(), predicateIr()));
    expect(verdict.findings).toEqual([]);
    expect(judge.stats.failedClosed).toBe(1);
    expect(verdict.degraded?.map((n) => n.reason)).toEqual(["failed-closed"]);
  });

  it("reports a blown per-call budget against the PER-CALL budget, not the message's", async () => {
    // `call-budget-exhausted`, not `budget-exhausted`. The number that expired
    // is the one this judge was constructed with; `ir.latencyBudgetMs` is
    // 30000 here and untouched, and a bake-off aggregating on `reason` would
    // otherwise count model slowness against the spec 5.3 message budget.
    const judge = new WebLlmJudge(
      fakeEngine({ throws: new DeadlineExpired("budget", 30_000, true) }),
      BUDGET,
    );
    const verdict = await judge.judge(req(whole(), predicateIr()));
    expect(judge.stats.deadlineExpiries).toBe(1);
    expect(verdict.degraded?.map((n) => n.reason)).toEqual(["call-budget-exhausted"]);
    // The message budget is what the orchestrator's own word is about, and no
    // part of it was consumed by this fake.
    expect(verdict.degraded?.[0]!.detail).toContain("per-call budget");
  });

  it("says nothing about an abort its own caller raised", async () => {
    // The caller performed the abort and already knows; the orchestrator files
    // its own budget notice from the timer it armed. A second notice from here
    // would double-count one event across two layers, and this judge cannot
    // tell WHY a caller withdrew -- a budget is only one of the reasons.
    const judge = new WebLlmJudge(
      fakeEngine({ throws: new DeadlineExpired("aborted", 30_000, true) }),
      BUDGET,
    );
    const verdict = await judge.judge(req(whole(), predicateIr()));
    expect(judge.stats.callerAbortsMidGeneration).toBe(1);
    expect(verdict.degraded ?? []).toEqual([]);
  });

  it("reports a latched engine's aborted response as failing closed", async () => {
    const judge = new WebLlmJudge(fakeEngine({ raw: "", finishReason: "abort" }), BUDGET);
    const verdict = await judge.judge(req(whole(), predicateIr()));
    expect(judge.stats.failedClosed).toBe(1);
    expect(verdict.degraded?.map((n) => n.reason)).toEqual(["failed-closed"]);
  });

  // Both early returns below leave the loop mid-message, and both were asserted
  // only on `stats` and `degraded[].reason` -- so `scopesJudged: []` on either
  // path survived the whole suite. It is not a cosmetic field: the orchestrator
  // turns "the policy declares a scope you did not evaluate" into a
  // `scope-unjudged` notice, so a judge that answered k segments and then
  // stopped would collect a SECOND notice for the same event, one of them
  // saying those predicates were never judged in their own scope. That is the
  // records-state-fact defect inside the field added to prevent it.
  const twoSegments = (): Segment[] => [
    { kind: "prose", start: 0, end: MSG.length, text: MSG },
    { kind: "prose", start: MSG.length + 1, end: MSG.length + 1 + MSG.length, text: MSG },
  ];

  it("still names the scope it judged when a per-call budget ends the run early", async () => {
    const judge = new WebLlmJudge(
      fakeEngine({
        script: [{ findings: [hit()] }, { throws: new DeadlineExpired("budget", 30_000, true) }],
      }),
      BUDGET,
    );
    const verdict = await judge.judge(req(twoSegments(), predicateIr()));
    // Segment 1 really was judged in the segment scope: its finding is here.
    expect(verdict.findings).toHaveLength(1);
    expect(judge.stats.segmentsJudged).toBe(1);
    expect(verdict.scopesJudged).toEqual(["segment"]);
  });

  it("still names the scope it judged when a latched engine ends the run early", async () => {
    const judge = new WebLlmJudge(
      fakeEngine({ script: [{ findings: [hit()] }, { raw: "", finishReason: "abort" }] }),
      BUDGET,
    );
    const verdict = await judge.judge(req(twoSegments(), predicateIr()));
    expect(verdict.findings).toHaveLength(1);
    expect(judge.stats.segmentsJudged).toBe(1);
    expect(verdict.scopesJudged).toEqual(["segment"]);
  });

  it("files no scope-unjudged notice for a run its own budget cut short", async () => {
    // The two notices this pairing must not produce together, through the real
    // orchestrator: one saying the run stopped, one saying the scope was never
    // evaluated. Only the first is true.
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(predicateIr())),
      provider: "claude",
      text: MSG,
      config: { tier0: false, tier1: false, tier2: true },
      engines: {
        tier2: new WebLlmJudge(
          fakeEngine({ throws: new DeadlineExpired("budget", 30_000, true) }),
          BUDGET,
        ),
      },
    });
    expect(result.degraded.filter((d) => d.reason === "scope-unjudged")).toEqual([]);
    expect(result.degraded.filter((d) => d.reason === "call-budget-exhausted")).toHaveLength(1);
  });

  it("never puts message text or model output in a notice", async () => {
    // A notice is a diagnostic and diagnostics get logged. The message text is
    // precisely what this system exists to keep out of logs, and a model's
    // broken body is message text it has been rearranging.
    const SECRET = "Northwind";
    const MODEL_LEAK = "ZZQXLEAKZZ";
    const judge = new WebLlmJudge(fakeEngine({ raw: `{"findings": ${MODEL_LEAK}` }), BUDGET);
    const verdict = await judge.judge(req(whole(), predicateIr()));
    const details = (verdict.degraded ?? []).map((n) => n.detail).join(" ");
    expect(details.length).toBeGreaterThan(0);
    expect(details).not.toContain(SECRET);
    expect(details).not.toContain(MODEL_LEAK);
  });

  it("surfaces failing closed all the way onto DetectionResult", async () => {
    // The whole chain, through the real orchestrator: an empty findings list
    // that a caller can tell apart from a clean message without reaching into
    // this judge's counters.
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(predicateIr())),
      provider: "claude",
      text: MSG,
      config: { tier0: false, tier1: false, tier2: true },
      engines: { tier2: new WebLlmJudge(fakeEngine({ raw: "not json at all" }), BUDGET) },
    });
    expect(result.findings).toEqual([]);
    expect(result.degraded.filter((d) => d.reason === "failed-closed")).toHaveLength(1);
    expect(result.degraded.every((d) => d.tier === 2 || d.reason === "absent")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Message scope: the predicates the policy declared about the WHOLE message.
// ---------------------------------------------------------------------------

/**
 * `PredicateScope`'s own definition in core is what these pin: "what a semantic
 * predicate is asked about: one segment at a time, or the whole message at
 * once". So the two scopes PARTITION the predicate list -- a message-scoped
 * predicate is asked once, about everything, and is not also sent per segment.
 *
 * `policies/compiled/p-fin.ir.json`, the only compiled policy in this
 * repository, declares exactly one semantic predicate and its scope is
 * "message", so the all-message shape below is the shipped one rather than a
 * corner case.
 */
describe("message-scoped predicates", () => {
  const messagePredicate = (id = "client-relationship") => ({
    id,
    nlPredicate: `whether the message discloses ${id}`,
    scope: "message" as const,
  });
  const segmentPredicate = (id: string) => ({
    id,
    nlPredicate: `whether this passage contains ${id}`,
    scope: "segment" as const,
  });

  /** Two prose segments with a one-character gap, as `req` assembles them. */
  const LEFT = "Northwind Traders is the counterparty";
  const RIGHT = "on the renewal we discussed on Friday.";
  const twoSegments = (): Segment[] => [
    { kind: "prose", start: 0, end: LEFT.length, text: LEFT },
    { kind: "prose", start: LEFT.length + 1, end: LEFT.length + 1 + RIGHT.length, text: RIGHT },
  ];

  it("asks one call about the whole message, carrying every message-scoped predicate", async () => {
    // Both halves of decision 1: ONE extra call (not one per predicate), and
    // all the message-scoped predicates together in it. TWO message-scoped
    // predicates, because at one the batched and per-predicate shapes make the
    // same number of calls and the test cannot tell them apart.
    const ir = predicateIr({
      predicates: [messagePredicate("m1"), messagePredicate("m2"), segmentPredicate("s1")],
    });
    const engine = fakeEngine({ findings: [] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const request = req(twoSegments(), ir);
    await judge.judge(request);

    // One message call, then one per segment.
    expect(engine.calls).toHaveLength(3);
    const message = engine.promptOf(0);
    expect(message).toContain("- m1: ");
    expect(message).toContain("- m2: ");
    // The segment-scoped clause is NOT in the message call: it was declared
    // about a segment, and asking it here would judge it in a scope the policy
    // did not name for it.
    expect(message).not.toContain("- s1: ");
    // The whole message, including the gap `req` puts between the segments --
    // which is the text no segment call is ever shown.
    expect(message).toContain(request.text);

    for (const index of [1, 2]) {
      expect(engine.promptOf(index)).toContain("- s1: ");
      expect(engine.promptOf(index)).not.toContain("- m1: ");
      expect(engine.promptOf(index)).not.toContain("- m2: ");
    }
  });

  it("resolves a message-scoped quote to ABSOLUTE offsets, not offsets into a segment", async () => {
    // The hazard `spans.ts` was built around: a span shifted by a segment's
    // start still satisfies core's `text === message.slice(start, end)` check,
    // because the shifted slice is sliced from the same string -- so it passes
    // every guard and points at the wrong words. The segment here starts at a
    // NON-ZERO offset, which is what makes a `+ segment.start` visible: at
    // offset 0 the right answer and the wrong one are the same number.
    const ir = predicateIr({ predicates: [messagePredicate()] });
    const segments: Segment[] = [{ kind: "prose", start: 12, end: 12 + MSG.length, text: MSG }];
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET);
    const request = req(segments, ir);
    const verdict = await judge.judge(request);

    // The expectation is read off the REQUEST's own text, not off anything the
    // judge computed.
    const at = request.text.indexOf(QUOTE);
    expect(at).toBeGreaterThan(0);
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.findings[0]!.start).toBe(at);
    expect(verdict.findings[0]!.end).toBe(at + QUOTE.length);
    expect(request.text.slice(verdict.findings[0]!.start, verdict.findings[0]!.end)).toBe(QUOTE);
    // And the ladder reports on it like any other finding: a message-scoped
    // resolution is not exempt from the rung the bake-off reads as evidence
    // strength. `rung1 + rung2` is documented as exactly the findings returned.
    expect(judge.stats.rung1).toBe(1);
    expect(judge.stats.rung2).toBe(0);
  });

  it("answers a quote that straddles two segments, which no segment call can", async () => {
    // The reason `JudgeRequest.text` exists, in core's own words: "a
    // message-scoped predicate whose evidence spans two segments is invisible
    // to a judge that only sees segments". The quote below crosses the gap
    // between the two segments, so it occurs in NEITHER of them.
    const straddling = "counterparty on the renewal";
    const segments = twoSegments();
    expect(segments.every((s) => !s.text.includes(straddling))).toBe(true);

    const message = predicateIr({ predicates: [messagePredicate()] });
    const judge = new WebLlmJudge(fakeEngine({ findings: [hit(straddling)] }), BUDGET);
    const request = req(segments, message);
    const found = (await judge.judge(request)).findings;
    expect(found).toHaveLength(1);
    expect(request.text.slice(found[0]!.start, found[0]!.end)).toBe(straddling);

    // THE CONTROL, and it is what makes the assertion above about the message
    // scope rather than about the ladder: the identical quote under a
    // SEGMENT-scoped predicate resolves nowhere, once per segment.
    const perSegment = predicateIr({ predicates: [segmentPredicate("client-relationship")] });
    const other = new WebLlmJudge(fakeEngine({ findings: [hit(straddling)] }), BUDGET);
    expect((await other.judge(req(segments, perSegment))).findings).toEqual([]);
    expect(other.stats.unresolvedQuotes).toBe(2);
  });

  it("spends NO segment call when every predicate is message-scoped", async () => {
    // The shipped shape: `p-fin` declares one predicate, message-scoped. A
    // segment call would carry an empty predicate list -- seconds of a 2 GB
    // model spent asking about nothing -- so the loop is not entered at all,
    // and the segments are not filed as skipped either: no stop cut anything
    // short.
    const ir = predicateIr({ predicates: [messagePredicate()] });
    const engine = fakeEngine({ findings: [hit()] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const verdict = await judge.judge(req(twoSegments(), ir));

    expect(engine.calls).toHaveLength(1);
    expect(verdict.scopesJudged).toEqual(["message"]);
    expect(judge.stats.messageScopeCalls).toBe(1);
    expect(judge.stats.messageScopeJudged).toBe(1);
    expect(judge.stats.segmentsJudged).toBe(0);
    expect(judge.stats.segmentsSkipped).toBe(0);
  });

  it("spends NO message call when every predicate is segment-scoped", async () => {
    // The other half of the partition, and the arm that was already shipping:
    // an extra whole-message call under a policy with no message-scoped clause
    // would be a call spent on nothing.
    const engine = fakeEngine({ findings: [] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const verdict = await judge.judge(req(twoSegments(), predicateIr()));
    expect(engine.calls).toHaveLength(2);
    expect(verdict.scopesJudged).toEqual(["segment"]);
    expect(judge.stats.messageScopeCalls).toBe(0);
    expect(judge.stats.messageScopeJudged).toBe(0);
  });

  it("names both scopes, in the order it asked about them, when the policy declares both", async () => {
    const ir = predicateIr({ predicates: [messagePredicate("m1"), segmentPredicate("s1")] });
    const judge = new WebLlmJudge(fakeEngine({ findings: [] }), BUDGET);
    const verdict = await judge.judge(req(twoSegments(), ir));
    expect(verdict.scopesJudged).toEqual(["message", "segment"]);
  });

  it("judges the message scope even when it was handed no segment at all", async () => {
    // Escalation is a decision about SEGMENTS. A message-scoped predicate is
    // not a segment question, so an empty segment list is not a reason to skip
    // it -- the orchestrator hands one over precisely when nothing qualified.
    const ir = predicateIr({ predicates: [messagePredicate()] });
    const engine = fakeEngine({ findings: [hit()] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const verdict = await judge.judge({
      text: MSG,
      segments: [],
      ir,
      priorFindings: [],
      budgetMs: 30_000,
    });
    expect(engine.calls).toHaveLength(1);
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.scopesJudged).toEqual(["message"]);
  });

  it("claims no scope when a segment-scoped policy was handed no segment", async () => {
    // The mirror, and it is a fact this judge used to get wrong: with an empty
    // segment list it made no call and still answered `["segment"]`, which
    // erased the orchestrator's `scope-unjudged` notice for a scope nothing had
    // asked about.
    const engine = fakeEngine({ findings: [] });
    const judge = new WebLlmJudge(engine, BUDGET);
    const verdict = await judge.judge({
      text: MSG,
      segments: [],
      ir: predicateIr(),
      priorFindings: [],
      budgetMs: 30_000,
    });
    expect(engine.calls).toEqual([]);
    expect(verdict.scopesJudged).toEqual([]);
  });

  it("counts the message call's repair retry against the message, and rows it like any other", async () => {
    // `messageScopeCalls` counts ISSUED calls, so the retry is in it; the
    // per-call rows carry both, because a call that failed to parse still spent
    // its tokens and its time-to-first-token.
    const ir = predicateIr({ predicates: [messagePredicate()] });
    const judge = new WebLlmJudge(
      fakeEngine({ script: [{ raw: "not json at all" }, { findings: [hit()] }] }),
      BUDGET,
    );
    const verdict = await judge.judge(req(whole(), ir));
    expect(verdict.findings).toHaveLength(1);
    expect(judge.stats.messageScopeCalls).toBe(2);
    expect(judge.stats.repairAttempts).toBe(1);
    expect(judge.stats.messageScopeJudged).toBe(1);
    expect(judge.stats.calls).toHaveLength(2);
  });

  it("charges a failed-closed message call to the message, never to the segment accounting", async () => {
    // `failedClosed` is documented and consumed as a count of SEGMENTS, and it
    // is one of the three terms of the segment invariant. A message-scope
    // failure landing there makes that sum exceed the segments handed over,
    // which is the denominator a recall number is built from.
    const ir = predicateIr({ predicates: [messagePredicate("m1"), segmentPredicate("s1")] });
    const judge = new WebLlmJudge(
      fakeEngine({
        script: [
          { raw: "not json" },
          { raw: "still not json" },
          { findings: [hit(QUOTE, 0.9, "s1")] },
        ],
      }),
      BUDGET,
    );
    const segments = twoSegments();
    const verdict = await judge.judge(req(segments, ir));

    expect(judge.stats.messageScopeFailedClosed).toBe(1);
    expect(judge.stats.failedClosed).toBe(0);
    // The run CONTINUES: one unit failing closed is not a stop.
    expect(judge.stats.segmentsJudged).toBe(2);
    expect(judge.stats.segmentsJudged + judge.stats.failedClosed + judge.stats.segmentsSkipped).toBe(
      segments.length,
    );
    expect(verdict.degraded?.map((n) => n.reason)).toEqual(["failed-closed"]);
    expect(verdict.degraded?.[0]!.detail).toContain("the whole message");
  });

  it("files the segments as skipped when a stop ends the run on the message call", async () => {
    // Message-first means a stop there costs the segments, and a scorer's
    // denominator has to say so -- `deadlineExpiries` is 1 whether the budget
    // blew on the message call or on segment 39.
    const ir = predicateIr({ predicates: [messagePredicate("m1"), segmentPredicate("s1")] });
    const judge = new WebLlmJudge(
      fakeEngine({ throws: new DeadlineExpired("budget", 30_000, true) }),
      BUDGET,
    );
    const segments = twoSegments();
    const verdict = await judge.judge(req(segments, ir));

    expect(judge.stats.messageScopeCalls).toBe(1);
    expect(judge.stats.messageScopeJudged).toBe(0);
    expect(judge.stats.messageScopeFailedClosed).toBe(0);
    expect(judge.stats.deadlineExpiries).toBe(1);
    expect(judge.stats.segmentsSkipped).toBe(segments.length);
    expect(judge.stats.segmentsJudged).toBe(0);
    // The scope was ASKED about -- see the verdict's own docblock -- so it is
    // named, and the budget word is what says the answer never arrived.
    expect(verdict.scopesJudged).toEqual(["message"]);
    expect(verdict.degraded?.[0]!.detail).toContain("the whole message");
    expect(verdict.degraded?.[0]!.detail).toContain("2 segment(s) were not judged");
  });

  it("skips no segment when the stopped message call was the only work there was", async () => {
    // The control for the count above: with no segment-scoped predicate the
    // loop would not have run, so nothing was skipped and the notice says what
    // was actually lost.
    const ir = predicateIr({ predicates: [messagePredicate()] });
    const judge = new WebLlmJudge(
      fakeEngine({ throws: new DeadlineExpired("budget", 30_000, true) }),
      BUDGET,
    );
    const verdict = await judge.judge(req(twoSegments(), ir));
    expect(judge.stats.segmentsSkipped).toBe(0);
    expect(verdict.degraded?.[0]!.detail).toContain("1 message-scoped predicate(s) were not judged");
  });

  it("keeps two clauses that name the same span, and drops one clause restated", async () => {
    // The de-duplication key is `start:end:entityType`, and it is still the
    // right key across the two calls because a predicate has exactly ONE scope:
    // no entityType is carried by both the message call and the segment calls,
    // so a collision means one clause restated and never two clauses agreeing.
    const ir = predicateIr({ predicates: [messagePredicate("m1"), segmentPredicate("s1")] });
    const judge = new WebLlmJudge(
      fakeEngine({
        script: [
          // The message call restates its own finding, which IS a duplicate.
          { findings: [hit(QUOTE, 0.9, "m1"), hit(QUOTE, 0.7, "m1")] },
          { findings: [hit(QUOTE, 0.9, "s1")] },
        ],
      }),
      BUDGET,
    );
    const found = (await judge.judge(req(whole(), ir))).findings;
    expect(judge.stats.duplicatesDropped).toBe(1);
    // Two POLICY CLAUSES over one span survive as two findings: core's cluster
    // resolution is what decides the action for the overlap, and collapsing
    // them here would silently drop one clause's severity.
    expect(found.map((f) => f.entityType)).toEqual(["pred:m1", "pred:s1"]);
    expect(new Set(found.map((f) => `${f.start}:${f.end}`)).size).toBe(1);
  });

  it("hands the MESSAGE call the caller's signal and the configured budget", async () => {
    // The equivalent segment-call assertion cannot cover this one: it builds
    // `predicateIr()`, whose default predicate is segment-scoped, so it never
    // issues a message call at all. `engine.ts` passes `opts.signal` as
    // `runWithDeadline`'s outer abort, and that signal is the ONLY channel by
    // which the orchestrator's per-message deadline can stop a call -- an
    // unsignalled call runs to the judge's per-call budget instead, which on a
    // bake-off arm is 60,000 ms against p-fin's 5,000 ms message budget. Under
    // message-first ordering this is the FIRST call of every message, so a
    // dropped signal would leave spec 5.3's per-message bound unenforced on
    // every item.
    //
    // A second, non-default budget on purpose: a test that only ever exercises
    // the value the implementation might have hardcoded cannot tell "reads the
    // config" from "hardcodes the default".
    const ir = predicateIr({ predicates: [messagePredicate("m1"), segmentPredicate("s1")] });
    const engine = fakeEngine({ findings: [] });
    const controller = new AbortController();
    await new WebLlmJudge(engine, { budgetMs: 1234 }).judge(
      req(twoSegments(), ir, [], { signal: controller.signal }),
    );
    // Call 0 is the message call -- the prompt is what says so, not the index.
    expect(engine.promptOf(0)).toContain("- m1: ");
    expect(engine.calls[0]!.opts.signal).toBe(controller.signal);
    expect(engine.calls[0]!.opts.budgetMs).toBe(1234);
    // And the segment calls after it, so a fix that wired only the first call
    // cannot pass.
    for (const index of [1, 2]) {
      expect(engine.calls[index]!.opts.signal, `call ${index}`).toBe(controller.signal);
      expect(engine.calls[index]!.opts.budgetMs, `call ${index}`).toBe(1234);
    }
  });

  it("tells the message call about EVERY prior on the message, as labels and counts", async () => {
    // The fairness premise of the whole head-to-head, pinned. `priorFindingsLine`'s
    // own docblock states it: the whole-message call carries every prior on the
    // message, "which is the same set Approach B's own prompt carries -- so the
    // two arms' models are told the same thing about the same lower tier when
    // they are shown the same text". Nothing asserted it on this side. The B
    // side IS asserted (`baselineB.test.ts`, "in-pan (x1)"), so a message call
    // that silently lost its priors would leave `baseline-b-tier0` with tier-0
    // context and `compiled` without it -- the compiled arm looking worse at
    // the task for a reason no counter, notice or gate reports.
    //
    // TWO priors, one in each segment, and the counts are the assertion. A
    // single prior cannot tell "the passage is the whole message" from "the
    // passage is the first segment", and the message passage's `end` is a
    // hand-written literal whose only consumer is this line: `priorFindingsLine`
    // drops a prior with `prior.start >= passage.end`, so an `end` that is too
    // small discards the tail of the message silently.
    const ir = predicateIr({ predicates: [messagePredicate("m1"), segmentPredicate("s1")] });
    const segments = twoSegments();
    const priors: Finding[] = segments.map((segment, index) => ({
      start: segment.start,
      end: segment.end,
      // Not a substring of the message: if the prompt interpolates a prior's
      // `text`, nothing else can have put this string there. The value a lower
      // tier detected is exactly the answer key this project keeps out of
      // prompts.
      text: `SENTINEL-PRIOR-${index}`,
      entityType: "client-name",
      severity: "medium",
      tier: 1,
      source: "gliner-pii-base",
      confidence: 0.9,
    }));
    const engine = fakeEngine({ findings: [] });
    await new WebLlmJudge(engine, BUDGET).judge(req(segments, ir, priors));

    const message = engine.promptOf(0);
    expect(message).toContain("client-name (x2)");
    expect(message).not.toContain("SENTINEL-PRIOR-0");
    expect(message).not.toContain("SENTINEL-PRIOR-1");
    // Each segment call sees only its own, which is what makes the x2 above a
    // statement about the message passage rather than about the prior list.
    expect(engine.promptOf(1)).toContain("client-name (x1)");
    expect(engine.promptOf(2)).toContain("client-name (x1)");
  });

  it("charges a LATCHED message call to the message, and ends the run there", async () => {
    // The latch is the one outcome that is both a failure and a stop, and on
    // the message call neither half was pinned. Task 3 measured the state: after
    // an interrupt the engine's flag stays set and every later call returns
    // instantly with an empty body and `finish_reason: "abort"`, so continuing
    // into the segment loop spends real calls on an engine that cannot answer
    // and files the results as SEGMENT failures -- a message-scope failure
    // landing inside the segment denominator, which is the units error the
    // three messageScope* counters exist to prevent.
    const ir = predicateIr({ predicates: [messagePredicate("m1"), segmentPredicate("s1")] });
    const engine = fakeEngine({ raw: "", finishReason: "abort" });
    const judge = new WebLlmJudge(engine, BUDGET);
    const segments = twoSegments();
    const verdict = await judge.judge(req(segments, ir));

    // The run ENDED: no segment call was attempted.
    expect(engine.calls).toHaveLength(1);
    expect(judge.stats.messageScopeCalls).toBe(1);
    expect(judge.stats.messageScopeJudged).toBe(0);
    expect(judge.stats.messageScopeFailedClosed).toBe(1);
    expect(judge.stats.abortedResponses).toBe(1);
    expect(judge.stats.truncatedResponses).toBe(0);
    expect(judge.stats.repairAttempts).toBe(0);
    // ... and it is charged to the MESSAGE, never to the segment accounting.
    expect(judge.stats.failedClosed).toBe(0);
    expect(judge.stats.segmentsJudged).toBe(0);
    expect(judge.stats.segmentsSkipped).toBe(segments.length);
    expect(verdict.scopesJudged).toEqual(["message"]);
    expect(verdict.degraded?.map((n) => n.reason)).toEqual(["failed-closed"]);
    expect(verdict.degraded?.[0]!.detail).toContain("the whole message");
  });

  it("files no scope-unjudged notice through detect() for a message-scoped policy", async () => {
    // The end of the chain, and the defect this closes: `p-fin`'s one predicate
    // is message-scoped, so every message used to carry a `scope-unjudged`
    // notice while the prompting baseline answered the same clause in one call.
    const ir = predicateIr({ predicates: [messagePredicate()] });
    const result = await detect({
      ir: loadPolicyIr(JSON.stringify(ir)),
      provider: "claude",
      text: MSG,
      config: { tier0: false, tier1: false, tier2: true },
      engines: { tier2: new WebLlmJudge(fakeEngine({ findings: [hit()] }), BUDGET) },
    });
    expect(result.degraded.filter((d) => d.reason === "scope-unjudged")).toEqual([]);
    expect(result.findings.map((f) => f.text)).toEqual([QUOTE]);
  });
});
