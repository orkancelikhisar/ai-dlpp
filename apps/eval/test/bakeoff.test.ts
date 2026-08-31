import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { UNCERTAIN_BELOW, loadPolicyIr, type PolicyIr } from "@sih/core";
import {
  GATES,
  INTERRUPT_DRAIN_OVERSHOOT_MS,
  armName,
  assertPageCanRun,
  deriveItemTimeoutMs,
  familyShape,
  gateReport,
  itemDeadlineBound,
  planBakeoff,
  type ArmFamily,
  type BakeoffOptions,
} from "../src/driver/bakeoff.js";
import { loadCorpus } from "../src/driver/corpus.js";
import { RECORD_SCHEMA_VERSION, RunRecordSchema, type RunRecord } from "../src/driver/record.js";
import { segmentSizeDistribution } from "../src/driver/segments.js";

/**
 * The bake-off driver: arms in, one JSONL file per arm and a gate verdict
 * beside each of them.
 *
 * Everything here is the NODE half -- planning, the deadline derivation and the
 * gate arithmetic -- because that is the half whose correctness can be settled
 * without twelve gigabytes of weights. The browser half has its own spec.
 *
 * The rule the whole file is arranged around: a gate is COMPUTED, never
 * enforced by dropping data. An arm that fails one is a result, and a driver
 * that omitted it would report a bake-off with fewer contenders than it had.
 */

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const CORPUS = join(REPO_ROOT, "corpora", "fixtures", "smoke.jsonl");
const ITEMS = loadCorpus(readFileSync(CORPUS, "utf8"));

/** The page fixture the bake-off has to run: the only IR here that declares a predicate. */
const SEMANTIC_IR = JSON.parse(
  readFileSync(join(REPO_ROOT, "apps", "eval", "fixtures", "semantic-ir.json"), "utf8"),
) as PolicyIr;

/**
 * A policy document and an IR that honestly names it.
 *
 * The digest below is a LITERAL, produced once by `shasum -a 256` on exactly
 * these bytes and pasted here, never recomputed by the test. A test that hashed
 * the string with the same function the implementation uses would pass whatever
 * that function did -- the hash-keyed-by-the-hash defect this project has
 * already shipped once.
 */
/**
 * The predicate IR with `minimal-ir.json`'s rules grafted on.
 *
 * A COMPOSED unit fixture and not a claim about any shipped policy: it exists
 * because the two escalation conditions are only distinguishable on an IR that
 * has both a semantic predicate and a tier-0 rule, and no such IR is shipped
 * here. `minimal-ir.json`'s `entropy-rule` is the one Task 9 measured firing on
 * this corpus's code fence at confidence 0.7, which is under `UNCERTAIN_BELOW`
 * and therefore re-admits that fence to escalation.
 */
const MINIMAL_IR = loadPolicyIr(readFileSync(join(REPO_ROOT, "apps", "eval", "fixtures", "minimal-ir.json"), "utf8"));
const MIXED_IR: PolicyIr = {
  ...SEMANTIC_IR,
  entityTypes: [...SEMANTIC_IR.entityTypes, ...MINIMAL_IR.entityTypes],
  rules: MINIMAL_IR.rules,
  actions: { default: { ...SEMANTIC_IR.actions.default, ...MINIMAL_IR.actions.default } },
};

const POLICY_TEXT = "# Test standard\n\nSection 1. Do not paste secrets.\n";
const POLICY_SHA256 = "2e337b6929ddd64f2fe2ae62feb134823c4883cf3dadd18a98fe8e2aba158396";
const PAIRED_IR: PolicyIr = { ...SEMANTIC_IR, policyHash: POLICY_SHA256 };

const MODEL = "Qwen3.5-2B-q4f16_1-MLC";

function options(overrides: Partial<BakeoffOptions> = {}): BakeoffOptions {
  return {
    runId: "bake",
    outDir: "/tmp/sih-bakeoff-test",
    corpus: CORPUS,
    provider: "claude",
    models: [MODEL],
    itemTimeoutMs: 400_000,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Record fixtures
// ---------------------------------------------------------------------------

type Call = NonNullable<RunRecord["tier2Stats"]>["calls"][number];
type Stats = NonNullable<RunRecord["tier2Stats"]>;

const ZERO_STATS: Omit<Stats, "calls"> = {
  rung1: 0,
  rung2: 0,
  unresolvedQuotes: 0,
  unknownPredicates: 0,
  duplicatesDropped: 0,
  repairAttempts: 0,
  failedClosed: 0,
  truncatedResponses: 0,
  abortedResponses: 0,
  segmentsJudged: 0,
  segmentsSkipped: 0,
  deadlineExpiries: 0,
  callerAbortsMidGeneration: 0,
  callerAbortsWhileQueued: 0,
};

/**
 * One schema-valid tier-2 record, VALIDATED here rather than merely typed.
 *
 * The validation is the point: a gate computed off a shape `RunRecordSchema`
 * would reject is a gate computed off a file that cannot exist. This project
 * has shipped a test whose fixture the real writer could not have produced.
 */
function rec(overrides: Partial<RunRecord> = {}): RunRecord {
  const record: RunRecord = {
    schemaVersion: RECORD_SCHEMA_VERSION,
    runId: "bake",
    itemId: "item-1",
    policy: "semantic-fixture",
    irHash: "a".repeat(64),
    policyHash: "test-hash",
    arm: "tier2-" + MODEL,
    backend: "webgpu",
    provider: "claude",
    config: { tier0: false, tier1: false, tier2: true, uncertainBelow: UNCERTAIN_BELOW },
    // Required exactly when `config.tier2` is set, and the values are what a
    // page loaded with no overrides reports back: `DEFAULT_TIER2_CONFIG` plus
    // the page's default per-call budget.
    tier2Config: {
      modelId: MODEL,
      contextWindowSize: 8192,
      temperature: 0,
      maxTokens: 512,
      callBudgetMs: 60_000,
    },
    text: "hello world",
    findings: [],
    gold: [],
    timings: { tier0Ms: 0.4, tier2Ms: 900 },
    degraded: [],
    tier2Stats: { ...ZERO_STATS, calls: [] },
    error: null,
    abandonedWorkInFlight: false,
    ...overrides,
  };
  const parsed = RunRecordSchema.safeParse(record);
  if (!parsed.success) throw new Error(`fixture is not a valid record: ${parsed.error.message}`);
  return record;
}

/** A record whose judge made `calls` and whose counters are otherwise supplied. */
function judged(calls: readonly Call[], counters: Partial<Omit<Stats, "calls">> = {}, extra: Partial<RunRecord> = {}): RunRecord {
  return rec({ tier2Stats: { ...ZERO_STATS, ...counters, calls: [...calls] }, ...extra });
}

const call = (over: Partial<Call> = {}): Call => ({
  finishReason: "stop",
  promptTokens: 400,
  completionTokens: 60,
  ttftMs: 500,
  decodeTokPerSec: 40,
  ...over,
});

const SEGMENTS = segmentSizeDistribution(ITEMS, { hasPredicates: true });

/**
 * The three settings no record carries, as `runBakeoff` would pass them for a
 * run of the shipped corpus against `semantic-ir.json`.
 */
const RUN_CONTEXT = {
  corpus: CORPUS,
  itemTimeoutMs: 250_000,
  latencyBudgetMs: SEMANTIC_IR.latencyBudgetMs,
} as const;

function report(records: readonly RunRecord[], family: ArmFamily = "compiled") {
  return gateReport({
    arm: "tier2-" + MODEL,
    family,
    modelId: MODEL,
    records,
    segments: SEGMENTS,
    ...RUN_CONTEXT,
  });
}

const outcome = (r: ReturnType<typeof report>, gate: string) => {
  const found = r.gates.find((g) => g.gate === gate);
  if (found === undefined) throw new Error(`no gate "${gate}" in [${r.gates.map((g) => g.gate).join(", ")}]`);
  return found;
};

// ---------------------------------------------------------------------------

describe("the amended gates", () => {
  it("is the amended pair, not the spec's original one", () => {
    // The originals would have killed all four arms for reasons unrelated to
    // capability: JSON compliance is vacuous under grammar constraint (0
    // malformed in 77 calls) and NO model achieves p95 3 s (cheapest is 4.6 s).
    expect(GATES.maxP95TtftMs).toBe(1500);
    expect(GATES.minDecodeTokPerSec).toBe(25);
    expect(GATES).not.toHaveProperty("maxP95LatencyMs");
    expect(GATES).not.toHaveProperty("minJsonCompliance");
  });

  it("records a killed arm rather than omitting it", () => {
    expect(GATES.recordKilledArms).toBe(true);
  });

  it("is exactly these six knobs, so a new one cannot arrive unnoticed", () => {
    // A key list rather than a pattern match, because the property being pinned
    // is "no gate is derived from the message latency budget" and a pattern is
    // both too loose and too tight for it: `assertNonEmptyAfterExpiry` contains
    // the word "expiry" and has nothing to do with the budget, while a new gate
    // called `maxUnjudgedScopes` would contain none of the obvious words and
    // would be exactly the rule this file must not have. Adding any gate breaks
    // this line and forces the argument to be made again.
    expect(Object.keys(GATES).sort()).toEqual([
      "assertNonEmptyAfterExpiry",
      "maxDuplicateRate",
      "maxP95TtftMs",
      "minDecodeTokPerSec",
      "minResolvableRate",
      "recordKilledArms",
    ]);
  });
});

describe("the per-item wall-clock ceiling", () => {
  it("takes the SMALLER of the two bounds the code actually arms", () => {
    // Two independent stops exist on a tier-2 item, and the shorter one is what
    // ends the run: the orchestrator's ONE deadline over the whole judge() call,
    // armed from what is left of `ir.latencyBudgetMs`; and the judge's per-CALL
    // budget spent on each of the calls it makes. Neither alone is the ceiling.
    const messageBinds = itemDeadlineBound({
      latencyBudgetMs: 5_000,
      callBudgetMs: 60_000,
      maxCallsPerItem: 4,
      lowerTierAllowanceMs: 0,
    });
    expect(messageBinds.bindingBound).toBe("message-budget");
    expect(messageBinds.boundMs).toBe(5_000 + INTERRUPT_DRAIN_OVERSHOOT_MS);

    const callsBind = itemDeadlineBound({
      latencyBudgetMs: 600_000,
      callBudgetMs: 10_000,
      maxCallsPerItem: 4,
      lowerTierAllowanceMs: 0,
    });
    expect(callsBind.bindingBound).toBe("call-budget");
    expect(callsBind.boundMs).toBe(4 * (10_000 + INTERRUPT_DRAIN_OVERSHOOT_MS));
  });

  it("adds the lower tiers on top rather than assuming the budget covers them", () => {
    // The orchestrator checks `ir.latencyBudgetMs` at TIER 2's turn and nowhere
    // else, so a slow tier-0 or tier-1 pass overruns it with nothing to stop it.
    // A ceiling that treated the message budget as covering the whole item would
    // convert a legitimately slow lower tier into an errored record.
    const bound = itemDeadlineBound({
      latencyBudgetMs: 5_000,
      callBudgetMs: 60_000,
      maxCallsPerItem: 4,
      lowerTierAllowanceMs: 2_500,
    });
    expect(bound.boundMs).toBe(2_500 + 5_000 + INTERRUPT_DRAIN_OVERSHOOT_MS);
  });

  it("doubles the bound, because it catches a wedge rather than enforcing a latency", () => {
    const input = { latencyBudgetMs: 5_000, callBudgetMs: 60_000, maxCallsPerItem: 4, lowerTierAllowanceMs: 0 };
    expect(deriveItemTimeoutMs(input)).toBe(2 * itemDeadlineBound(input).boundMs);
  });

  it("refuses a ceiling at or below the bound, which is where the plan's literal sits", () => {
    // The plan writes `itemTimeoutMs: 120_000` and the semantic IR this bake-off
    // has to run writes `latencyBudgetMs: 120_000`. Those are the SAME number,
    // so an item that legitimately spends its whole message budget races the
    // driver's own deadline -- and when the driver wins, the row becomes an
    // error AND every later row of the arm is stamped `abandonedWorkInFlight`,
    // which invalidates the arm's latencies. Not a tuning preference: a ceiling
    // equal to the thing it is supposed to sit above catches the good case.
    expect(SEMANTIC_IR.latencyBudgetMs).toBe(120_000);
    expect(() =>
      planBakeoff({ options: options({ itemTimeoutMs: 120_000 }), ir: SEMANTIC_IR, items: ITEMS }),
    ).toThrow(/120000/);
    // And one comfortably above it is accepted.
    expect(() =>
      planBakeoff({ options: options({ itemTimeoutMs: 400_000 }), ir: SEMANTIC_IR, items: ITEMS }),
    ).not.toThrow();
  });

  it("counts a baseline arm's calls per MESSAGE and a compiled arm's per SEGMENT", () => {
    // The two families do not make the same number of calls, and the difference
    // is intrinsic to the method rather than a harness asymmetry: the compiled
    // judge makes one engine call per SELECTED SEGMENT plus the one repair
    // retry, while Approach B makes one call per MESSAGE plus the same retry.
    expect(familyShape("compiled").judgedUnit).toBe("segment");
    expect(familyShape("baseline-b").judgedUnit).toBe("message");
    expect(familyShape("baseline-b-tier0").judgedUnit).toBe("message");
    const plan = planBakeoff({
      options: options({ families: ["compiled", "baseline-b"] }),
      ir: PAIRED_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    const compiled = plan.arms.find((a) => a.family === "compiled")!;
    const baseline = plan.arms.find((a) => a.family === "baseline-b")!;
    // The compiled arm pays for every selected segment; B pays once per message
    // however many segments the message has.
    expect(compiled.maxCallsPerItem).toBe(2 * compiled.segments.perItem!.max);
    expect(compiled.segments.perItem!.max).toBeGreaterThan(1);
    expect(baseline.maxCallsPerItem).toBe(2);
  });
});

describe("the two segments-per-message conditions, never mixed", () => {
  it("measures each arm's escalation under that arm's OWN tier-0 setting", () => {
    // Task 9's figures have two conditions -- with tier-0 priors and without --
    // and a previous version of the plan spliced them into one distribution with
    // p95 2 and max 3, which no single sample can have: at n = 13 the
    // nearest-rank p95 IS the maximum. The driver cannot splice them because it
    // never holds one distribution for two arms: it recomputes per arm, from the
    // IR that arm runs, and stamps the escalation input it used.
    const plan = planBakeoff({
      options: options({ families: ["compiled", "compiled-tier2-only"] }),
      ir: SEMANTIC_IR,
      items: ITEMS,
    });
    const withTier0 = plan.arms.find((a) => a.family === "compiled")!;
    const without = plan.arms.find((a) => a.family === "compiled-tier2-only")!;
    expect(withTier0.config.tier0).toBe(true);
    expect(without.config.tier0).toBe(false);
    for (const arm of [withTier0, without]) {
      // p95 and max come from ONE sample, so they can never disagree the way
      // the spliced pair did.
      expect(arm.segments.perItem!.p95).toBeLessThanOrEqual(arm.segments.perItem!.max);
      expect(arm.segments.escalation.uncertainBelow).toBe(arm.config.uncertainBelow);
    }
  });

  it("gives the tier-0 arm the same escalation as the tier-2-only arm ON THIS IR, and that is a fact about the IR", () => {
    // Not an oversight and not a splice. `semantic-ir.json` carries `rules: []`,
    // so `runTier0` finds nothing on any item, so there are no priors to make a
    // segment uncertain -- and the tier-0 arm's escalation is identical to the
    // tier-2-only arm's. Task 9's "with priors" condition (p95 3, max 3) was
    // measured against `minimal-ir.json`, which HAS entropy rules and is not an
    // IR any tier-2 arm can run. Anything quoting 3 for this bake-off is quoting
    // a different policy's number.
    expect(SEMANTIC_IR.rules).toEqual([]);
    const plan = planBakeoff({
      options: options({ families: ["compiled", "compiled-tier2-only"] }),
      ir: SEMANTIC_IR,
      items: ITEMS,
    });
    const withTier0 = plan.arms.find((a) => a.family === "compiled")!;
    const without = plan.arms.find((a) => a.family === "compiled-tier2-only")!;
    expect(withTier0.segments.perItem).toEqual(without.segments.perItem);
    expect(withTier0.segments.perItem).toEqual({ p50: 1, p95: 2, max: 2, min: 1 });
  });
});

describe("the p95 time-to-first-token gate", () => {
  it("is a percentile over CALLS, not over per-message means", () => {
    // A p95 of per-message averages is a different statistic from a p95 over
    // calls, and it is not the one the gate names. Here two messages carry two
    // calls each; every per-message mean is 1000, so a mean-of-means gate would
    // read 1000 and pass, while the call sample's 95th value is 1900.
    const r = report([
      judged([call({ ttftMs: 100 }), call({ ttftMs: 1900 })]),
      judged([call({ ttftMs: 100 }), call({ ttftMs: 1900 })]),
    ]);
    expect(r.answeredCalls).toBe(4);
    expect(r.ttftMs).toEqual({ p50: 100, p95: 1900, max: 1900, min: 100 });
    expect(outcome(r, "p95-ttft").observed).toBe(1900);
    expect(outcome(r, "p95-ttft").verdict).toBe("fail");
  });

  it("reports the prompt size the TTFT was measured at, over the same calls", () => {
    // `maxP95TtftMs: 1500` was derived at a WHOLE PROMPT of ~1.1 kB -- 1,031
    // characters at this corpus's median segment, 1,122 at its largest, measured
    // through the real `buildMessages` with the IR this driver must run -- so a TTFT
    // taken against a materially bigger prompt is not measuring this gate. The
    // driver cannot enforce that -- it does not hold the prompt -- but it must
    // put the size beside the number so an arm is not killed on a measurement
    // taken at different work.
    const r = report([judged([call({ ttftMs: 400, promptTokens: 380 }), call({ ttftMs: 600, promptTokens: 4000 })])]);
    expect(r.promptTokens).toEqual({ p50: 380, p95: 4000, max: 4000, min: 380 });
    // Same sample size as the latency it stands beside; a prompt column taken
    // over a different set of calls could not be checked against it.
    expect(outcome(r, "p95-ttft").sample).toBe(2);
    // And the segment sizes the arm actually ran over, which ARE in characters
    // and so are directly comparable to the ~1.1 kB the threshold was derived at.
    expect(r.segmentChars).toEqual(SEGMENTS.chars);
  });

  it("does not kill an arm that produced no call to measure", () => {
    // An arm whose every message ran out of budget before its first call has no
    // TTFT sample at all. Killing it here would be killing it for the budget
    // overrun by the back door.
    const r = report([judged([], {}, { degraded: [{ tier: 2, reason: "budget-exhausted", detail: "spent" }] })]);
    expect(outcome(r, "p95-ttft").verdict).toBe("not-measured");
    expect(outcome(r, "p95-ttft").observed).toBeUndefined();
    expect(r.killed).toBe(false);
  });

  it("excludes a call whose engine reported no time-to-first-token", () => {
    const r = report([judged([call({ ttftMs: 900 }), call({ ttftMs: undefined }), call({ ttftMs: null })])]);
    expect(outcome(r, "p95-ttft").sample).toBe(1);
    expect(r.ttftMs).toEqual({ p50: 900, p95: 900, max: 900, min: 900 });
  });
});

describe("the decode-rate gate", () => {
  it("is the engine's own rate, weighted by the tokens each call decoded", () => {
    // Token-weighted and not a plain mean: the gate says "sustained", and a
    // 5-token call that decoded slowly must not outvote a 500-token call that
    // decoded fast. 10 tokens at 10 tok/s is 1 s; 90 tokens at 90 tok/s is 1 s;
    // 100 tokens in 2 s is 50 tok/s, where the plain mean of the two rates is 50
    // as well only by coincidence -- so the fixture below breaks the tie.
    const r = report([
      judged([
        call({ completionTokens: 10, decodeTokPerSec: 10 }),
        call({ completionTokens: 990, decodeTokPerSec: 100 }),
      ]),
    ]);
    // 10/10 + 990/100 = 1 + 9.9 = 10.9 s for 1000 tokens -> 91.74 tok/s.
    expect(r.sustainedDecodeTokPerSec).toBeCloseTo(1000 / 10.9, 6);
    // A plain mean of the rates would be 55, which is a different answer.
    expect(r.sustainedDecodeTokPerSec).toBeGreaterThan(60);
    expect(outcome(r, "decode-rate").verdict).toBe("pass");
  });

  it("fails an arm below the floor", () => {
    const r = report([judged([call({ completionTokens: 100, decodeTokPerSec: 20 })])]);
    expect(outcome(r, "decode-rate").observed).toBeCloseTo(20, 6);
    expect(outcome(r, "decode-rate").verdict).toBe("fail");
    expect(r.killed).toBe(true);
  });

  it("excludes the 0/0 a call interrupted before its first token leaves", () => {
    // VERIFIED in the installed 0.2.84 bundle: `decode_tokens_per_s` is
    // `completion_tokens / decode_time`, a plain division. `runArm` maps the
    // resulting NaN to null so the file round-trips; treating that null as a 0
    // would drag the weighted rate to zero and kill the arm on a call that
    // decoded nothing.
    const r = report([
      judged([call({ completionTokens: 100, decodeTokPerSec: 40 }), call({ completionTokens: 0, decodeTokPerSec: null })]),
    ]);
    expect(outcome(r, "decode-rate").sample).toBe(1);
    expect(outcome(r, "decode-rate").observed).toBeCloseTo(40, 6);
  });

  it("excludes a reported rate of zero rather than dividing by it", () => {
    // Unreachable from a stock engine -- `completion_tokens / decode_time` is 0
    // only when `completion_tokens` is 0, which the token guard already excludes
    // -- and guarded anyway, because the schema admits a 0 and the arithmetic
    // does not: `tokens / 0` is Infinity, which drags the weighted rate to 0 and
    // fails a FLOOR gate on a call that reported nothing.
    const r = report([
      judged([
        call({ completionTokens: 100, decodeTokPerSec: 40 }),
        call({ completionTokens: 5, decodeTokPerSec: 0 }),
      ]),
    ]);
    expect(outcome(r, "decode-rate").sample).toBe(1);
    expect(outcome(r, "decode-rate").observed).toBeCloseTo(40, 6);
    expect(outcome(r, "decode-rate").verdict).toBe("pass");
  });

  it("is not measured when no call reported a usable rate", () => {
    const r = report([judged([call({ decodeTokPerSec: null }), call({ decodeTokPerSec: undefined })])]);
    expect(outcome(r, "decode-rate").verdict).toBe("not-measured");
    expect(r.killed).toBe(false);
  });
});

describe("the semantic gates", () => {
  it("scores resolvable findings against every quote the ladder was given", () => {
    // The denominator is quotes the model produced, not findings that survived:
    // an arm whose quotes never resolve has found nothing, and dividing by the
    // survivors would report it as perfect.
    const r = report([judged([call()], { rung1: 6, rung2: 2, unresolvedQuotes: 2 })]);
    expect(outcome(r, "resolvable-rate").observed).toBeCloseTo(0.8, 10);
    expect(outcome(r, "resolvable-rate").verdict).toBe("pass");
    const bad = report([judged([call()], { rung1: 3, rung2: 0, unresolvedQuotes: 7 })]);
    expect(outcome(bad, "resolvable-rate").observed).toBeCloseTo(0.3, 10);
    expect(outcome(bad, "resolvable-rate").verdict).toBe("fail");
  });

  it("counts a dropped duplicate as a quote that RESOLVED, in both halves", () => {
    // READ from `WebLlmJudge.#collect`: `resolveQuote` runs BEFORE the
    // duplicate check, so an unresolved quote is dropped first and a duplicate
    // is by definition a quote the ladder DID place. Leaving duplicates out
    // omitted them from the numerator and the denominator at once, which is how
    // an arm whose ladder works was reported as one whose quotes do not resolve.
    //
    // The fixture is the shape Plan 5 says to expect on this hardware: a model
    // that finds one thing and restates it. 8 quotes resolved (1 first
    // occurrence + 7 restatements) out of 10 given to the ladder is 0.8, and
    // the OLD arithmetic reported "1 of 3 ... 0.333" and killed the arm.
    const r = report([judged([call()], { rung1: 1, duplicatesDropped: 7, unresolvedQuotes: 2 })]);
    expect(outcome(r, "resolvable-rate").sample).toBe(10);
    expect(outcome(r, "resolvable-rate").observed).toBeCloseTo(0.8, 10);
    expect(outcome(r, "resolvable-rate").verdict).toBe("pass");
    // And the detail says where the 8 came from, so a reader can check it
    // against `ladder` without re-deriving the rule.
    expect(outcome(r, "resolvable-rate").detail).toContain("8 of 10 quote(s) resolved");
    expect(outcome(r, "resolvable-rate").detail).toContain("duplicate");
    // The gate can still fail, and on the thing it is named for: quotes the
    // ladder refused. Same 8 resolutions, more refusals.
    const unresolvable = report([judged([call()], { rung1: 1, duplicatesDropped: 7, unresolvedQuotes: 6 })]);
    expect(outcome(unresolvable, "resolvable-rate").observed).toBeCloseTo(8 / 14, 10);
    expect(outcome(unresolvable, "resolvable-rate").verdict).toBe("fail");
  });

  it("scores duplicates against every finding that resolved to a span", () => {
    const r = report([judged([call()], { rung1: 4, rung2: 0, duplicatesDropped: 6 })]);
    expect(outcome(r, "duplicate-rate").observed).toBeCloseTo(0.6, 10);
    // 0.6 PASSES, and that is the correction: the ceiling was 0.5, which is
    // under the 0.667 Plan 5 measured for the arm it recommends as primary.
    expect(outcome(r, "duplicate-rate").verdict).toBe("pass");
  });

  it("does not kill the restatement behaviour Plan 5 told the bake-off to expect", () => {
    // Plan 5's measurement of the recommended primary arm, verbatim:
    // Qwen3.5-2B "found only the AWS key, three times over" -- one distinct
    // span and two restatements, a rate of 0.667 -- with the instruction
    // "expect duplicates, expect misses, and do not tune the corpus to hide
    // either". A ceiling that kills that kills the primary arm for the
    // behaviour the plan predicted, which the 0.5 ceiling did.
    const primary = report([judged([call()], { rung1: 1, duplicatesDropped: 2 })]);
    expect(outcome(primary, "duplicate-rate").observed).toBeCloseTo(2 / 3, 10);
    expect(outcome(primary, "duplicate-rate").verdict).toBe("pass");
    expect(primary.killed).toBe(false);

    // ... and it still kills the pathology it is for: Plan 5 measured
    // Phi-4-mini looping `"quote": "Halcyon"` about nineteen times until the
    // token budget ran out mid-string. One distinct span and eighteen
    // restatements is 0.947, an answer that is nothing but one thing said again.
    const looping = report([judged([call()], { rung1: 1, duplicatesDropped: 18 })]);
    expect(outcome(looping, "duplicate-rate").observed).toBeCloseTo(18 / 19, 10);
    expect(outcome(looping, "duplicate-rate").verdict).toBe("fail");
    expect(looping.killed).toBe(true);
  });

  it("is not measured on an arm that emitted no finding at all", () => {
    const r = report([judged([call()])]);
    expect(outcome(r, "resolvable-rate").verdict).toBe("not-measured");
    expect(outcome(r, "duplicate-rate").verdict).toBe("not-measured");
  });
});

describe("the engine-poisoning assertion", () => {
  it("flags a run whose first call after a stop came back aborted", () => {
    // Task 3 measured the engine latching its interrupt flag on the pinned
    // non-streaming path, after which every later call returns instantly and
    // empty -- which a judge reads as "no findings" forever. `clearInterrupt`
    // fixes it by writing a field TypeScript marks private, so an upstream
    // rename would silently restore the poisoning with a green unit suite. This
    // is the only assertion that would notice.
    const r = report([
      judged([call()]),
      judged([], { deadlineExpiries: 1, segmentsSkipped: 1 }),
      judged([call({ finishReason: "abort", completionTokens: 0 })], { failedClosed: 1, abortedResponses: 1 }),
    ]);
    expect(outcome(r, "non-empty-after-stop").verdict).toBe("fail");
    expect(r.killed).toBe(true);
  });

  it("flags a poisoned call on EITHER term, not only on both at once", () => {
    // The gate is `finishReason === "abort" || completionTokens === 0`, and the
    // only fixture that used to reach it set BOTH -- so dropping either half
    // left the whole suite green. The review that found this measured both:
    // removing the token term survived 133 tests, and so did removing the abort
    // term.
    //
    // The zero-token half is the one that matters most, and it is not
    // hypothetical: this repository's own reproduction of a real interrupted
    // call (`packages/tier2/test/judge.test.ts`, "passes a poisoned usage
    // number through") has `completion_tokens` 0 with the fake's DEFAULT finish
    // reason -- i.e. not "abort". A gate simplified to the abort half alone
    // would report that engine as `non-empty-after-stop: pass`.
    const stop = judged([], { deadlineExpiries: 1, segmentsSkipped: 1 });

    const abortOnly = report([
      judged([call()]),
      stop,
      // "abort" with real tokens behind it: the engine says an interrupt cut
      // the answer off, which is the event, whatever the token count says.
      judged([call({ finishReason: "abort", completionTokens: 7 })], { abortedResponses: 1 }),
    ]);
    expect(outcome(abortOnly, "non-empty-after-stop").verdict).toBe("fail");
    expect(outcome(abortOnly, "non-empty-after-stop").detail).toContain('finishReason "abort"');

    const emptyOnly = report([
      judged([call()]),
      stop,
      // An empty body under an ordinary finish reason -- the shape judge.test.ts
      // reproduces. The judge reads an empty body as "no findings".
      judged([call({ finishReason: "stop", completionTokens: 0 })], { failedClosed: 1 }),
    ]);
    expect(outcome(emptyOnly, "non-empty-after-stop").verdict).toBe("fail");
    expect(outcome(emptyOnly, "non-empty-after-stop").detail).toContain("0 completion token(s)");
  });

  it("does not flag a call the engine reported no usage for", () => {
    // The control that keeps the token term from being "anything but a
    // positive number": `completionTokens` undefined means the engine reported
    // no `usage` at all, which is a different fact from an empty body and not
    // evidence of a latched engine. Without this, `!call.completionTokens`
    // would pass every test above.
    const r = report([
      judged([call()]),
      judged([], { deadlineExpiries: 1, segmentsSkipped: 1 }),
      judged([call({ finishReason: "stop", completionTokens: undefined })], { rung1: 1 }),
    ]);
    expect(outcome(r, "non-empty-after-stop").verdict).toBe("pass");
  });

  it("passes when the call after the stop answered normally", () => {
    const r = report([
      judged([], { deadlineExpiries: 1 }),
      judged([call()], { rung1: 1 }),
      judged([call({ finishReason: "abort", completionTokens: 0 })]),
    ]);
    // Only the FIRST call after the FIRST stop is the evidence: a later abort
    // belongs to whatever stopped before it, and reading the whole tail would
    // report one poisoning as several.
    expect(outcome(r, "non-empty-after-stop").verdict).toBe("pass");
  });

  it("is not measured when nothing stopped", () => {
    expect(outcome(report([judged([call()])]), "non-empty-after-stop").verdict).toBe("not-measured");
  });
});

describe("the budget overrun is counted and never gated on", () => {
  it("reports budget-exhausted and scope-unjudged beside the verdict, and kills nothing", () => {
    // On this corpus, at the measured per-call cost, tier 2 expires mid-run on
    // most messages that escalate more than one segment WHATEVER arm wins. An
    // arm whose every message overran but whose measured numbers are all healthy
    // must survive: the alternative kills all four for a reason that is not
    // about capability.
    const records = [
      judged([call({ ttftMs: 400 })], { rung1: 5 }, {
        degraded: [
          { tier: 2, reason: "budget-exhausted", detail: "the 5000ms message latency budget ran out" },
          { tier: 2, reason: "scope-unjudged", detail: "predicates were not judged in the scope" },
        ],
      }),
      judged([call({ ttftMs: 500 })], { rung1: 5 }, {
        degraded: [{ tier: 2, reason: "budget-exhausted", detail: "already spent" }],
      }),
    ];
    const r = report(records);
    expect(r.degradedNotices["budget-exhausted"]).toBe(2);
    expect(r.degradedNotices["scope-unjudged"]).toBe(1);
    expect(r.degradedItems["budget-exhausted"]).toBe(2);
    expect(r.degradedItems["scope-unjudged"]).toBe(1);
    expect(r.killed).toBe(false);
    expect(r.gates.every((g) => g.verdict !== "fail")).toBe(true);
  });

  it("counts NOTICES and ITEMS apart, because one item can carry two", () => {
    const r = report([
      judged([call()], {}, {
        degraded: [
          { tier: 2, reason: "scope-unjudged", detail: "segment scope" },
          { tier: 2, reason: "scope-unjudged", detail: "message scope" },
        ],
      }),
    ]);
    expect(r.degradedNotices["scope-unjudged"]).toBe(2);
    expect(r.degradedItems["scope-unjudged"]).toBe(1);
  });

  it("names every reason word, so a zero is a measurement rather than a missing key", () => {
    const r = report([judged([call()])]);
    expect(Object.keys(r.degradedNotices).sort()).toEqual([
      "absent",
      "budget-exhausted",
      "call-budget-exhausted",
      "failed-closed",
      "scope-unjudged",
    ]);
  });
});

describe("what a gate report says about the run itself", () => {
  it("separates an errored item from an item that scored nothing", () => {
    const r = report([
      rec({ tier2Stats: undefined, degraded: undefined, error: "boom" }),
      judged([call()]),
      judged([call()], {}, { abandonedWorkInFlight: true }),
    ]);
    expect(r.items).toBe(3);
    expect(r.itemsErrored).toBe(1);
    expect(r.itemsAbandonedWorkInFlight).toBe(1);
  });

  it("states the settings the numbers were produced under", () => {
    // WHY. Neither output file used to record its own configuration: this
    // report had no runId, no IR, no corpus and no budgets, and a record
    // carried `config.t2Model` and nothing else about tier 2. Two runs at
    // different context windows or per-call budgets were byte-identical in
    // every field a scorer can group by, and the gates file could not be joined
    // to the IR it ran against at all. Plan 5's own fallback for a model that
    // cannot take 8,192 -- "run that arm at 4,096 and report the asymmetry" --
    // was unexpressible in the output.
    const r = report([rec(), rec({ itemId: "item-2" })]);
    expect(r.run).toEqual({
      runId: "bake",
      // The basename, not the path: the path is machine-specific and is not
      // evidence about the run.
      corpus: "smoke.jsonl",
      irHash: "a".repeat(64),
      policyHash: "test-hash",
      recordSchemaVersion: RECORD_SCHEMA_VERSION,
      itemTimeoutMs: 250_000,
      latencyBudgetMs: SEMANTIC_IR.latencyBudgetMs,
      uncertainBelow: UNCERTAIN_BELOW,
      tier2Config: {
        modelId: MODEL,
        contextWindowSize: 8192,
        temperature: 0,
        maxTokens: 512,
        callBudgetMs: 60_000,
      },
    });
    // And it comes off the ROWS rather than off anything the caller asserted:
    // an arm run at the smaller window says so here, which is what makes a
    // deliberately asymmetric run distinguishable from a symmetric one.
    const narrow = report([
      rec({ tier2Config: { modelId: MODEL, contextWindowSize: 4096, temperature: 0, maxTokens: 512, callBudgetMs: 30_000 } }),
    ]);
    expect(narrow.run.tier2Config?.contextWindowSize).toBe(4096);
    expect(narrow.run.tier2Config?.callBudgetMs).toBe(30_000);
  });

  it("refuses records that disagree about the settings they ran under", () => {
    // The same refusal the foreign-arm check makes, applied to the
    // configuration: a gates row summarising rows written under two windows, or
    // two IRs, or two run ids, would be a confident number describing neither
    // -- and every field of it would look well-formed.
    const wrongWindow = rec({
      itemId: "item-2",
      tier2Config: { modelId: MODEL, contextWindowSize: 4096, temperature: 0, maxTokens: 512, callBudgetMs: 60_000 },
    });
    expect(() => report([rec(), wrongWindow])).toThrow(/tier2Config/);
    expect(() => report([rec(), rec({ itemId: "item-2", irHash: "b".repeat(64) })])).toThrow(/irHash/);
    expect(() => report([rec(), rec({ itemId: "item-2", runId: "other" })])).toThrow(/runId/);
  });

  it("refuses records that disagree about which arm they belong to", () => {
    // A report summing two arms' rows would be a confident number describing no
    // run at all, and nothing downstream could see it.
    expect(() =>
      gateReport({
        arm: "tier2-" + MODEL,
        family: "compiled",
        modelId: MODEL,
        records: [rec(), rec({ arm: "someone-else" })],
        segments: SEGMENTS,
        ...RUN_CONTEXT,
      }),
    ).toThrow(/someone-else/);
  });
});

describe("planning the slate", () => {
  it("orders the models cheapest-first and names one file per arm", () => {
    const plan = planBakeoff({
      options: options({
        models: ["Phi-4-mini-instruct-q4f16_1-MLC", "Qwen3.5-2B-q4f16_1-MLC"],
        families: ["compiled"],
      }),
      ir: SEMANTIC_IR,
      items: ITEMS,
    });
    // Cheapest first by the manifest's order, whatever order the caller listed.
    expect(plan.arms.map((a) => a.modelId)).toEqual([
      "Qwen3.5-2B-q4f16_1-MLC",
      "Phi-4-mini-instruct-q4f16_1-MLC",
    ]);
    expect(plan.arms.map((a) => a.path.endsWith(armName("compiled", a.modelId) + ".jsonl"))).toEqual([true, true]);
  });

  it("refuses two arms that would write the same file", () => {
    expect(() =>
      planBakeoff({
        options: options({ models: [MODEL, MODEL], families: ["compiled"] }),
        ir: SEMANTIC_IR,
        items: ITEMS,
      }),
    ).toThrow(/would both write/);
  });

  it("refuses an unknown model rather than discovering it at load", () => {
    expect(() =>
      planBakeoff({ options: options({ models: ["Qwen3.5-2B"] }), ir: SEMANTIC_IR, items: ITEMS }),
    ).toThrow(/Qwen3\.5-2B/);
  });

  it("refuses an IR that declares no semantic predicate", () => {
    // `WebLlmJudge.judge` returns before it touches the engine on such an IR, so
    // every arm would produce a complete, schema-valid file of zero model calls
    // and the bake-off would report four arms that never ran.
    const empty: PolicyIr = { ...SEMANTIC_IR, semanticPredicates: [] };
    expect(() => planBakeoff({ options: options(), ir: empty, items: ITEMS })).toThrow(/semanticPredicate/);
  });

  it("gives every arm the same corpus, provider, timeout and window", () => {
    const plan = planBakeoff({
      options: options({ families: ["compiled", "compiled-tier2-only"], models: [MODEL, "Qwen3-4B-q4f16_1-MLC"] }),
      ir: SEMANTIC_IR,
      items: ITEMS,
    });
    expect(plan.arms).toHaveLength(4);
    // ONE value each, held on the plan rather than per arm, so no arm can be
    // given a different one: a bake-off in which one arm had a longer deadline
    // than another is measuring the deadline.
    expect(typeof plan.itemTimeoutMs).toBe("number");
    expect(new Set(plan.arms.map((a) => a.contextWindowSize)).size).toBe(1);
    expect(new Set(plan.arms.map((a) => a.callBudgetMs)).size).toBe(1);
  });

  it("sizes the shared ceiling from the family that needs the most, not the least", () => {
    // The compiled family makes up to 2 calls per selected segment and the
    // baseline family 2 per message. One shared ceiling is what keeps the
    // comparison fair, and it has to be the LARGER of the two bounds, or the
    // family with more calls is squeezed by a deadline sized for the other.
    //
    // `callBudgetMs: 10_000` is what makes this test able to fail at all. At the
    // page's 60,000 ms default both families' bounds collapse onto the same
    // number -- the semantic IR's 120,000 ms message budget binds for both, so
    // 4 calls and 2 calls produce the same ceiling and a "take the smallest"
    // implementation is indistinguishable from a correct one. This was caught by
    // mutation: the earlier version of this test survived exactly that change.
    const both = planBakeoff({
      options: options({ families: ["compiled", "baseline-b"], callBudgetMs: 10_000 }),
      ir: PAIRED_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    const compiled = both.arms.find((a) => a.family === "compiled")!;
    const baseline = both.arms.find((a) => a.family === "baseline-b")!;
    // 4 calls x (10,000 + 20) against 2 x (10,000 + 20): the per-call budget is
    // now the binding bound for both, and it separates them.
    expect(compiled.bound.bindingBound).toBe("call-budget");
    expect(compiled.bound.boundMs).toBe(1_000 + 4 * 10_020);
    expect(baseline.bound.boundMs).toBe(1_000 + 2 * 10_020);
    expect(both.bound.boundMs).toBe(compiled.bound.boundMs);
    expect(both.bound.boundMs).toBeGreaterThan(baseline.bound.boundMs);
  });
});

describe("the baseline family and the policy it would be shown", () => {
  it("refuses a baseline arm whose policy text is not what the IR was compiled from", () => {
    // `PolicyIr.policyHash` is the compiler's sha256 of the source DOCUMENT
    // (packages/compiler/src/stages/emit.ts). Approach B's whole premise is
    // being shown that document; shown a different one, the head-to-head
    // compares two policies rather than two methods, with a green suite.
    expect(() =>
      planBakeoff({
        options: options({ families: ["baseline-b"] }),
        ir: SEMANTIC_IR,
        items: ITEMS,
        policyText: "# Some other standard\n",
      }),
    ).toThrow(/policyHash/);
  });

  it("names the mismatch on this repository's own fixture, which cannot host a baseline arm", () => {
    // Not a hypothetical. `apps/eval/fixtures/semantic-ir.json` is hand-written
    // with `policyHash: "test-hash"`, which is not a sha256 of anything, so no
    // document in `policies/` can satisfy it. Until a compiled IR exists this is
    // the honest state of the baseline family, and it is a refusal rather than a
    // silently different comparison.
    expect(SEMANTIC_IR.policyHash).toBe("test-hash");
    let message = "";
    try {
      planBakeoff({
        options: options({ families: ["baseline-b-tier0"] }),
        ir: SEMANTIC_IR,
        items: ITEMS,
        policyText: readFileSync(join(REPO_ROOT, "policies", "p-corp.md"), "utf8"),
      });
    } catch (cause) {
      message = (cause as Error).message;
    }
    expect(message).toContain("test-hash");
    expect(message).toMatch(/[0-9a-f]{64}/);
  });

  it("refuses a baseline arm with no policy document at all", () => {
    expect(() =>
      planBakeoff({ options: options({ families: ["baseline-b"] }), ir: SEMANTIC_IR, items: ITEMS }),
    ).toThrow(/policy/i);
  });

  it("pairs each baseline arm with the compiled arm that runs the same tiers", () => {
    // B alone against a tier-0+tier-2 compiled arm would credit tier 0's
    // deterministic findings to the compiler. The families are declared in
    // pairs so the comparison a reader makes is between arms that ran the same
    // deterministic half.
    expect(familyShape("baseline-b").runsTier0).toBe(false);
    expect(familyShape("compiled-tier2-only").runsTier0).toBe(false);
    expect(familyShape("baseline-b-tier0").runsTier0).toBe(true);
    expect(familyShape("compiled").runsTier0).toBe(true);
  });
});

describe("what this harness can actually execute", () => {
  it("lets a compiled-only slate through", () => {
    expect(() =>
      assertPageCanRun(planBakeoff({ options: options(), ir: SEMANTIC_IR, items: ITEMS })),
    ).not.toThrow();
  });

  it("refuses a baseline arm, naming the door that is missing and the three things behind it", () => {
    // A valid experiment with nowhere to run. The refusal is separate from
    // planning on purpose: the plan is a statement about fairness and this is a
    // statement about the browser half, and conflating them would make a fair
    // slate look invalid.
    const plan = planBakeoff({
      options: options({ families: ["baseline-b", "baseline-b-tier0"] }),
      ir: PAIRED_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    expect(plan.arms).toHaveLength(2);
    let message = "";
    try {
      assertPageCanRun(plan);
    } catch (cause) {
      message = (cause as Error).message;
    }
    // The door.
    expect(message).toContain("createBaselineB");
    expect(message).toContain("window.__sih.detect");
    // And the three reasons adding it would not be enough on its own.
    expect(message).toContain("compiled from");
    expect(message).toContain("rules");
    expect(message).toContain("BaselineStats");
  });
});

describe("numbers that reach a setTimeout", () => {
  it("refuses the spellings of \"no budget\" that become a 1 ms deadline", () => {
    // `Infinity` is the natural spelling and it is the dangerous one: measured
    // elsewhere in this repo, it produces a ~1 ms timeout rather than an
    // infinite one, and NaN, negative and > 2^31 all do the same. Every number
    // below is a summand of a ceiling that ends up in `setTimeout`.
    const base = { latencyBudgetMs: 5_000, callBudgetMs: 60_000, maxCallsPerItem: 4, lowerTierAllowanceMs: 0 };
    expect(() => itemDeadlineBound({ ...base, latencyBudgetMs: Infinity })).toThrow(/latencyBudgetMs/);
    expect(() => itemDeadlineBound({ ...base, callBudgetMs: Number.NaN })).toThrow(/callBudgetMs/);
    expect(() => itemDeadlineBound({ ...base, lowerTierAllowanceMs: -1 })).toThrow(/lowerTierAllowanceMs/);
    expect(() => itemDeadlineBound({ ...base, maxCallsPerItem: 1.5 })).toThrow(/maxCallsPerItem/);
    // A numeric STRING compares fine against both bounds and `setTimeout`
    // coerces and honours it, so it would time correctly and be recorded in a
    // field typed number.
    expect(() => itemDeadlineBound({ ...base, callBudgetMs: "60000" as unknown as number })).toThrow(/callBudgetMs/);
  });

  it("refuses a zero budget, which is a deadline that fires immediately", () => {
    // The spelling the check used to ACCEPT, and the comment above it already
    // named: `>= 0` let `callBudgetMs: 0` and `latencyBudgetMs: 0` through
    // planning, and a 0 budget was then refused only by `WebLlmJudge`'s
    // constructor INSIDE THE BROWSER -- i.e. after the model load that
    // `planBakeoff` exists to happen before. `cancel.ts` refuses both with
    // `x > 0 && x <= MAX_BUDGET_MS`, and this now matches it.
    const base = { latencyBudgetMs: 5_000, callBudgetMs: 60_000, maxCallsPerItem: 4, lowerTierAllowanceMs: 0 };
    expect(() => itemDeadlineBound({ ...base, latencyBudgetMs: 0 })).toThrow(/latencyBudgetMs/);
    expect(() => itemDeadlineBound({ ...base, callBudgetMs: 0 })).toThrow(/callBudgetMs/);
    // The allowance is the one number that MAY be 0, and the asymmetry is
    // deliberate: it is a caller saying "give the deterministic tiers no
    // headroom", not a timer set to fire at once. The `base` above already
    // exercises that, so this is only making it explicit.
    expect(itemDeadlineBound({ ...base, lowerTierAllowanceMs: 0 }).boundMs).toBe(5_020);
  });

  it("refuses a budget above what setTimeout can hold", () => {
    // MEASURED on Node v26.0.0: `setTimeout(fn, 6_000_002_040)` prints
    // "TimeoutOverflowWarning: ... Timeout duration was set to 1" and fired
    // after 5 ms. The old check accepted these -- `Number.isFinite(3e9)` is
    // true -- so a `lowerTierAllowanceMs` or an `ir.latencyBudgetMs` above
    // 2^31-1 (core's schema is `z.number().int().positive()` with no upper
    // bound) derived a ceiling `run.ts` fires in ~1 ms on every item, turning
    // the whole arm into errored rows and stamping every later row
    // `abandonedWorkInFlight`.
    const base = { latencyBudgetMs: 5_000, callBudgetMs: 60_000, maxCallsPerItem: 4, lowerTierAllowanceMs: 0 };
    const over = 2_147_483_648;
    expect(() => itemDeadlineBound({ ...base, latencyBudgetMs: over })).toThrow(/latencyBudgetMs/);
    expect(() => itemDeadlineBound({ ...base, callBudgetMs: over })).toThrow(/callBudgetMs/);
    expect(() => itemDeadlineBound({ ...base, lowerTierAllowanceMs: over })).toThrow(/lowerTierAllowanceMs/);
    // Exactly on the bound is accepted, so the refusal is a ceiling and not an
    // off-by-one that also rejects the largest legal value.
    expect(() => itemDeadlineBound({ ...base, lowerTierAllowanceMs: 2_147_483_647 })).not.toThrow();
  });

  it("refuses a ceiling that is not a duration at all", () => {
    const bound = itemDeadlineBound({ latencyBudgetMs: 5_000, callBudgetMs: 1_000, maxCallsPerItem: 2, lowerTierAllowanceMs: 0 });
    expect(() => planBakeoff({ options: options({ itemTimeoutMs: Infinity }), ir: SEMANTIC_IR, items: ITEMS })).toThrow(
      /itemTimeoutMs/,
    );
    // And a ceiling `setTimeout` cannot hold, for the reason above: this number
    // reaches a real timer in `run.ts`, so one over 2^31-1 fires in ~1 ms
    // rather than never -- which errors every item instead of catching a wedge.
    expect(() =>
      planBakeoff({ options: options({ itemTimeoutMs: 2_147_483_648 }), ir: SEMANTIC_IR, items: ITEMS }),
    ).toThrow(/itemTimeoutMs/);
    expect(bound.boundMs).toBeGreaterThan(0);
  });
});

describe("arm names reach the filesystem", () => {
  it("is file-safe for every family crossed with every pinned model", () => {
    // Both halves of a file name end up on disk, so neither may contain a path
    // separator, "..", or anything a shell glob has an opinion about. The model
    // ids carry dots, digits and underscores, which is exactly the set this has
    // to survive.
    const safe = /^[A-Za-z0-9][A-Za-z0-9._+-]*$/;
    const families: ArmFamily[] = ["compiled", "compiled-tier2-only", "baseline-b", "baseline-b-tier0"];
    const names = new Set<string>();
    for (const family of families) {
      for (const model of ["Qwen3.5-2B-q4f16_1-MLC", "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC", "Qwen3-4B-q4f16_1-MLC", "Phi-4-mini-instruct-q4f16_1-MLC"]) {
        const name = armName(family, model);
        expect(name).toMatch(safe);
        // And distinct: two families sharing a name would write one file.
        expect(names.has(name)).toBe(false);
        names.add(name);
      }
    }
    expect(names.size).toBe(16);
  });
});

describe("killed is the disjunction of the failures", () => {
  it("is set by any one failing gate and by no not-measured one", () => {
    const oneFailure = report([judged([call({ ttftMs: 9000 })], { rung1: 10 })]);
    expect(oneFailure.gates.filter((g) => g.verdict === "fail").map((g) => g.gate)).toEqual(["p95-ttft"]);
    expect(oneFailure.killed).toBe(true);

    const twoFailures = report([judged([call({ ttftMs: 9000, completionTokens: 100, decodeTokPerSec: 2 })], { rung1: 1, unresolvedQuotes: 9 })]);
    expect(twoFailures.gates.filter((g) => g.verdict === "fail").map((g) => g.gate).sort()).toEqual([
      "decode-rate",
      "p95-ttft",
      "resolvable-rate",
    ]);
    expect(twoFailures.killed).toBe(true);

    const nothingMeasured = report([judged([])]);
    expect(nothingMeasured.gates.every((g) => g.verdict === "not-measured")).toBe(true);
    expect(nothingMeasured.killed).toBe(false);
  });

  it("every outcome carries the numbers behind it, so a verdict is checkable", () => {
    const r = report([judged([call({ ttftMs: 1400 })], { rung1: 9, unresolvedQuotes: 1, duplicatesDropped: 1 })]);
    for (const gate of r.gates) {
      expect(gate.detail.length).toBeGreaterThan(20);
      if (gate.verdict !== "not-measured") expect(gate.sample).toBeGreaterThan(0);
    }
    expect(outcome(r, "p95-ttft").verdict).toBe("pass");
    expect(outcome(r, "p95-ttft").detail).toContain("1400");
  });
});

describe("the tier-0 priors reach only the arms that run tier 0", () => {
  it("selects MORE segments for the tier-0 arm on an IR whose tier 0 finds something", () => {
    // The structural half of "never mix the two conditions": the driver does not
    // hold one distribution and label it twice, it runs `runTier0` for the arms
    // that run tier 0 and passes nothing for the arms that do not. On
    // `semantic-ir.json` the two coincide because it declares no rules, which is
    // exactly why that fixture cannot prove this -- hence MIXED_IR.
    //
    // 17 and 18 are Task 9's numbers, measured in test/segments.test.ts against
    // the same corpus and the same escalation policy but through a different
    // caller, so they are not this module's arithmetic restated.
    const plan = planBakeoff({
      options: options({ families: ["compiled", "compiled-tier2-only"] }),
      ir: MIXED_IR,
      items: ITEMS,
    });
    const withTier0 = plan.arms.find((a) => a.family === "compiled")!;
    const without = plan.arms.find((a) => a.family === "compiled-tier2-only")!;
    expect(without.segments.count).toBe(17);
    expect(withTier0.segments.count).toBe(18);
    expect(without.segments.perItem).toEqual({ p50: 1, p95: 2, max: 2, min: 1 });
    expect(withTier0.segments.perItem).toEqual({ p50: 1, p95: 3, max: 3, min: 1 });
    // And the ceiling follows: 3 segments is 6 calls, not 4.
    expect(withTier0.maxCallsPerItem).toBe(6);
    expect(without.maxCallsPerItem).toBe(4);
  });
});

describe("the ceiling's boundary", () => {
  it("refuses a ceiling EQUAL to the bound, not merely one below it", () => {
    // The off-by-one that matters. A ceiling exactly on the bound fires on the
    // item that used exactly its budget, which on this corpus is the normal case
    // rather than the pathological one -- and the cost of that firing is the
    // arm's whole latency column, because every later row is stamped
    // `abandonedWorkInFlight`.
    const plan = planBakeoff({ options: options(), ir: SEMANTIC_IR, items: ITEMS });
    // Hand-derived from the parts, not read off the plan: 1,000 ms for the tiers
    // below, plus the semantic IR's 120,000 ms message budget, plus Task 3's
    // 20 ms interrupt-and-drain overshoot.
    expect(plan.bound.boundMs).toBe(1_000 + 120_000 + 20);
    expect(() =>
      planBakeoff({ options: options({ itemTimeoutMs: plan.bound.boundMs }), ir: SEMANTIC_IR, items: ITEMS }),
    ).toThrow(/not above/);
    expect(() =>
      planBakeoff({ options: options({ itemTimeoutMs: plan.bound.boundMs + 1 }), ir: SEMANTIC_IR, items: ITEMS }),
    ).not.toThrow();
  });
});

describe("the latency sample and the prompt sample are the same calls", () => {
  it("leaves a call with prompt tokens but no time-to-first-token out of BOTH", () => {
    // A prompt column taken over a different set of calls than the latency it
    // stands beside cannot be checked against it -- which is the whole job of
    // that column, since `maxP95TtftMs` is only meaningful at the prompt size it
    // was derived for.
    const r = report([
      judged([
        call({ ttftMs: 700, promptTokens: 410 }),
        call({ ttftMs: undefined, promptTokens: 9999 }),
      ]),
    ]);
    expect(r.ttftCalls).toBe(1);
    expect(r.promptTokens).toEqual({ p50: 410, p95: 410, max: 410, min: 410 });
    expect(r.completionTokens).toEqual({ p50: 60, p95: 60, max: 60, min: 60 });
  });
});

describe("a stop ends the run, so the stopped item's own calls came before it", () => {
  it("does not read a call made BEFORE the stop as the call after it", () => {
    // Both loops -- `WebLlmJudge.judge` and `baselineB.ts`'s `createArm` --
    // return on every `DeadlineExpired`, so no call is made on an item after its
    // stop. An implementation that latched the flag before walking the item's
    // calls would read that item's LAST answered call as the evidence, which is
    // a call the engine made while it was healthy.
    const r = report([
      // One segment answered, then the second call blew the per-call budget.
      judged([call({ finishReason: "abort", completionTokens: 0 })], { deadlineExpiries: 1, segmentsSkipped: 1 }),
      judged([call({ finishReason: "stop", completionTokens: 40 })], { rung1: 1 }),
    ]);
    // The aborted row belongs to the item that stopped and is not evidence of a
    // latched engine; the healthy call on the next item is.
    expect(outcome(r, "non-empty-after-stop").verdict).toBe("pass");
  });
});
