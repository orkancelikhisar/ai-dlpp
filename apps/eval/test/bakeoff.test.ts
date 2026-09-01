import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { UNCERTAIN_BELOW, loadPolicyIr, runTier0, segmentText, type PolicyIr } from "@sih/core";
import {
  COMPILER_DEFAULT_LATENCY_BUDGET_MS,
  DEFAULT_LOWER_TIER_ALLOWANCE_MS,
  DEFAULT_TIER2_CALL_BUDGET_MS,
  GATES,
  INTERRUPT_DRAIN_OVERSHOOT_MS,
  armName,
  assertPageCanRun,
  deriveItemTimeoutMs,
  familyShape,
  gateReport,
  itemDeadlineBound,
  judgedUnitFor,
  planBakeoff,
  runBakeoff,
  type ArmFamily,
  type BakeoffOptions,
  type GateReportInput,
} from "../src/driver/bakeoff.js";
import { loadCorpus } from "../src/driver/corpus.js";
import { RECORD_SCHEMA_VERSION, RunRecordSchema, type RunRecord } from "../src/driver/record.js";
import { percentile, segmentSizeDistribution } from "../src/driver/segments.js";

/**
 * The bake-off driver: arms in, one JSONL file per arm and a gate verdict
 * beside each of them.
 *
 * Everything here is the NODE half -- planning, the deadline derivation, the
 * gate arithmetic and, through a scripted page, `runBakeoff`'s own refusals --
 * because that is the half whose correctness can be settled without the 7.49 GB
 * of weights `test/tier2-profile.ts` measured. The browser half has its own spec.
 *
 * The rule the whole file is arranged around: a gate is COMPUTED, never
 * enforced by dropping data. An arm that fails one is a result, and a driver
 * that omitted it would report a bake-off with fewer contenders than it had.
 */

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const CORPUS = join(REPO_ROOT, "corpora", "fixtures", "smoke.jsonl");
const ITEMS = loadCorpus(readFileSync(CORPUS, "utf8"));

/**
 * The hand-written page fixture most of this file plans against: one of the two
 * IRs here that declare a semantic predicate. `policies/compiled/p-fin.ir.json`
 * is the other, and it is the one `test/baseline.spec.ts` and
 * `test/bakeoff-run.test.ts`'s Approach-B block run, because only a compiled
 * policy can be paired with the document B is shown.
 */
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
  messageScopeCalls: 0,
  messageScopeJudged: 0,
  messageScopeFailedClosed: 0,
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
    detector: "core-orchestrator",
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

/**
 * The two escalation conditions, each measured the way `planBakeoff` measures
 * it for the family that runs under it.
 *
 * TWO of them and never one, because `gateReport` now refuses a distribution
 * whose condition disagrees with the arm's family -- `judgedUnitChars`,
 * `judgedUnitsPerItem` and `escalation` are the only fields on a gate report that
 * do not come off the rows, so the wrong one would describe the other family's
 * work under this arm's name. A single shared constant was what let this file
 * hand a no-priors distribution to a `compiled` report for the whole of the
 * previous round.
 *
 * `undefined` priors and `() => []` are deliberately different callers here:
 * the first is the tier-2-only condition and the second would not be.
 */
const SEGMENTS = segmentSizeDistribution(ITEMS, { hasPredicates: true });
const SEGMENTS_TIER0 = segmentSizeDistribution(ITEMS, {
  hasPredicates: true,
  priorFindings: (item) => runTier0(SEMANTIC_IR, item.text, segmentText(item.text)),
});

/**
 * The same two conditions again for the MESSAGE-judged families, which is a
 * third dimension rather than a variation on those two.
 *
 * Approach B makes one call per message however many segments the text has, so
 * `judgedUnitChars` on a B report is a distribution over whole messages and
 * `judgedUnitsPerItem` is all 1s. `gateReport` refuses a family/unit mismatch
 * for the reason it refuses the priors mismatch: a segment p50 of 62 characters
 * under an arm whose model was shown whole messages is a plausible number
 * describing the other method's work.
 */
const MESSAGES_B = segmentSizeDistribution(ITEMS, { hasPredicates: true, unit: "message" });
const MESSAGES_B_TIER0 = segmentSizeDistribution(ITEMS, {
  hasPredicates: true,
  unit: "message",
  priorFindings: (item) => runTier0(SEMANTIC_IR, item.text, segmentText(item.text)),
});

/**
 * The three settings no record carries, as `runBakeoff` would pass them for a
 * run of the shipped corpus against `semantic-ir.json`.
 *
 * `itemTimeoutMs` is a number this driver chose and is reported, not checked,
 * so it need not be the one `planBakeoff` would derive.
 */
const RUN_CONTEXT = {
  corpus: CORPUS,
  itemTimeoutMs: 250_000,
  latencyBudgetMs: SEMANTIC_IR.latencyBudgetMs,
  // The fourth thing no record carries: the tier each gold entityType belongs
  // to, which only an IR knows. `runBakeoff` passes the IR it verified against
  // the page's own digest, so this is that array.
  entityTypes: SEMANTIC_IR.entityTypes,
  // The fifth: the scopes the policy declares, which is what decides the unit
  // this arm's model was shown. `SEMANTIC_IR`'s only predicate is
  // segment-scoped, so every report built through this constant is a
  // segment-judged one -- the shape the whole file assumed while the unit was a
  // per-family constant. The message-scoped and both-scopes shapes are their
  // own fixtures, in the judged-unit block.
  semanticPredicates: SEMANTIC_IR.semanticPredicates,
  // The fifth, and the only one of the five the PAGE produces rather than the
  // driver: `runBakeoff` reads all three off the same `Tier2LoadReport` whose
  // `servedModelId` it has already refused the arm on. Three values that share
  // no digits, so a report that copied one field into another is a red test.
  load: { engineLoadMs: 1_234, engineWarmupMs: 567, originStorageBytes: 8_900_000_000 },
  // The sixth: what the WHOLE run crosses, which one arm's rows cannot show.
  // Deliberately NOT the four-model four-family slate, so a scope sentence that
  // went back to naming "the four pinned arms" whatever ran is a red test here
  // rather than a false claim on every row of a one-family run.
  slate: { models: [MODEL], families: ["compiled", "compiled-tier2-only"] },
} as const;

/**
 * The distribution `planBakeoff` would hand `gateReport` for this family.
 *
 * Both dimensions, because both are checked: the unit follows `judgedUnit` and
 * the priors condition follows `runsTier0`. Written as a lookup off
 * `familyShape` rather than a per-test literal so a test cannot accidentally
 * hand an arm the other family's numbers -- which is the mistake this file made
 * for a whole round on the priors axis alone.
 */
function distributionFor(family: ArmFamily) {
  const shape = familyShape(family);
  // `runsCompiledJudge` and not the unit, because this file's default IR fixes
  // the unit: `SEMANTIC_IR` declares one segment-scoped predicate, so the
  // compiled families are judged per segment under `RUN_CONTEXT` and only the
  // Approach-B families are judged per message. A report built here against a
  // differently-scoped policy has to say so and pass its own distribution.
  if (!shape.runsCompiledJudge) return shape.runsTier0 ? MESSAGES_B_TIER0 : MESSAGES_B;
  return shape.runsTier0 ? SEGMENTS_TIER0 : SEGMENTS;
}

/** The distribution `planBakeoff` would hand `gateReport` for this family. */
function report(
  records: readonly RunRecord[],
  family: ArmFamily = "compiled",
  over: Partial<GateReportInput> = {},
) {
  return gateReport({
    arm: "tier2-" + MODEL,
    family,
    modelId: MODEL,
    records,
    segments: distributionFor(family),
    ...RUN_CONTEXT,
    ...over,
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
    //
    // On THIS IR. `PAIRED_IR`'s only predicate is segment-scoped, which is the
    // one shape on which "count the selected segments" is the whole story --
    // a message-scoped clause costs one more call per item, and the block
    // "the call ceiling is the policy's, not only the family's" below is what
    // covers that.
    expect(PAIRED_IR.semanticPredicates.map((p) => p.scope)).toEqual(["segment"]);
    expect(judgedUnitFor("compiled", PAIRED_IR.semanticPredicates)).toBe("segment");
    expect(judgedUnitFor("baseline-b", PAIRED_IR.semanticPredicates)).toBe("message");
    expect(judgedUnitFor("baseline-b-tier0", PAIRED_IR.semanticPredicates)).toBe("message");
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

describe("the judged unit and the call ceiling are the policy's, not only the family's", () => {
  /**
   * WHAT a compiled arm's model is shown, and how many times, is decided by the
   * SCOPES the IR declares: `WebLlmJudge` partitions its predicates by scope and
   * never enters a loop it has no predicate for, so a message-scoped clause
   * costs exactly one whole-message call an item and a policy with no
   * segment-scoped clause costs no segment call at all however many segments
   * escalation selected.
   *
   * Every other planning test in this file runs `SEMANTIC_IR`, whose only
   * predicate is segment-scoped -- the one shape on which counting segments and
   * counting segments-plus-the-message-call give the same answer, so none of
   * them can tell the two rules apart. These can.
   */
  const MESSAGE_CLAUSE: PolicyIr["semanticPredicates"][number] = {
    id: "board-confidential",
    nlPredicate: "whether the message discusses an unannounced board decision",
    scope: "message",
  };

  /** The shadow entityType the compiler mints for a predicate; see mintShadowEntityTypes. */
  const MESSAGE_SHADOW: PolicyIr["entityTypes"][number] = {
    id: "pred:board-confidential",
    tier: 2,
    nlDefinition: MESSAGE_CLAUSE.nlPredicate,
    examples: [],
    counterExamples: [],
    severity: "high",
  };

  /** The shipped fixture's predicate, moved to the scope `p-fin`'s real one declares. */
  const MESSAGE_ONLY_IR: PolicyIr = {
    ...PAIRED_IR,
    semanticPredicates: PAIRED_IR.semanticPredicates.map((p) => ({ ...p, scope: "message" })),
  };

  /** Both scopes at once, which is the shape neither `p-fin` nor the fixture has. */
  const BOTH_SCOPES_IR: PolicyIr = {
    ...PAIRED_IR,
    semanticPredicates: [...PAIRED_IR.semanticPredicates, MESSAGE_CLAUSE],
    entityTypes: [...PAIRED_IR.entityTypes, MESSAGE_SHADOW],
    actions: {
      ...PAIRED_IR.actions,
      default: { ...PAIRED_IR.actions.default, [MESSAGE_SHADOW.id]: "redact" },
    },
  };

  /**
   * Two messages that are nothing but a fenced block, and no high-entropy
   * string in either -- so escalation selects NOTHING on them under both tier-0
   * conditions. MEASURED against core's own `selectSegments` in the first
   * assertion below rather than asserted here, since the whole point of the
   * corpus is a precondition this file does not control.
   */
  const CODE_ONLY_ITEMS = loadCorpus(
    [
      {
        id: "code-only-a",
        text: "```\nconst total = price * quantity;\n```",
        policy: "minimal-fixture",
        gold: [],
      },
      {
        id: "code-only-b",
        text: "```js\nfunction add(a, b) {\n  return a + b;\n}\n```",
        policy: "minimal-fixture",
        gold: [],
      },
    ]
      .map((item) => JSON.stringify(item))
      .join("\n"),
  );

  it("adds the whole-message call to a compiled arm's ceiling, and only to that arm's", () => {
    // The undercount this closes: `maxCallsPerItem` reaches `itemDeadlineBound`
    // as bound (b), and a ceiling below an arm's legitimate worst case turns a
    // merely-slow item into an errored row AND stamps every later row of the
    // arm `abandonedWorkInFlight` -- the arm's whole latency column.
    const plan = (ir: PolicyIr) =>
      planBakeoff({
        options: options({ families: ["compiled", "baseline-b"] }),
        ir,
        items: ITEMS,
        policyText: POLICY_TEXT,
      });
    const armOf = (p: ReturnType<typeof plan>, family: ArmFamily) =>
      p.arms.find((a) => a.family === family)!;

    const both = plan(BOTH_SCOPES_IR);
    const segmentOnly = plan(PAIRED_IR);
    const compiledBoth = armOf(both, "compiled");
    const compiledSegmentOnly = armOf(segmentOnly, "compiled");

    // The same corpus and the same escalation, so the SEGMENT term is identical
    // and the only thing that moved is the declared scope -- which adds exactly
    // one judged unit per item, the whole-message call.
    expect(compiledSegmentOnly.segments.perItem!.max).toBe(3);
    expect(compiledBoth.segments.perItem!.max).toBe(4);
    expect(compiledBoth.segments.perItem).toEqual({ p50: 2, p95: 4, max: 4, min: 2 });
    expect(compiledSegmentOnly.segments.perItem).toEqual({ p50: 1, p95: 3, max: 3, min: 1 });
    expect(compiledSegmentOnly.maxCallsPerItem).toBe(6);
    expect(compiledBoth.maxCallsPerItem).toBe(8);
    // Approach B is shown the message whatever the policy declares.
    expect(armOf(both, "baseline-b").maxCallsPerItem).toBe(2);
    expect(armOf(segmentOnly, "baseline-b").maxCallsPerItem).toBe(2);
  });

  it("charges NO segment call on a policy with no segment-scoped clause", () => {
    // The other half, and the one `p-fin` is: escalation still selects up to 3
    // segments per item, the judge is still handed them, and it makes no call
    // about any of them -- so a ceiling of 2 x 3 would size the arm at three
    // times the calls it can make.
    const plan = planBakeoff({
      options: options({ families: ["compiled"] }),
      ir: MESSAGE_ONLY_IR,
      items: ITEMS,
    });
    const compiled = plan.arms[0]!;
    // What escalation would select under this arm's own tier-0 condition, which
    // is unchanged by the scope the predicate is declared in -- the rules are
    // the same rules. It is the number the ceiling used to be built from.
    expect(SEGMENTS_TIER0.perItem!.max).toBe(3);
    // And what the arm is actually planned over: one whole message per item.
    expect(compiled.segments.unit).toBe("message");
    expect(compiled.segments.perItem!.max).toBe(1);
    expect(compiled.maxCallsPerItem).toBe(2);
  });

  it("derives the judged unit from the family AND the policy's declared scopes", () => {
    // The rule `familyShape` could not state, because it cannot see the policy.
    // Three shapes, three answers, and the middle one is what
    // `policies/compiled/p-fin.ir.json` is -- the only compiled policy here.
    expect(PAIRED_IR.semanticPredicates.map((p) => p.scope)).toEqual(["segment"]);
    expect(MESSAGE_ONLY_IR.semanticPredicates.map((p) => p.scope)).toEqual(["message"]);
    expect(BOTH_SCOPES_IR.semanticPredicates.map((p) => p.scope)).toEqual(["segment", "message"]);
    for (const family of ["compiled", "compiled-tier2-only"] as const) {
      expect(judgedUnitFor(family, PAIRED_IR.semanticPredicates)).toBe("segment");
      expect(judgedUnitFor(family, MESSAGE_ONLY_IR.semanticPredicates)).toBe("message");
      expect(judgedUnitFor(family, BOTH_SCOPES_IR.semanticPredicates)).toBe("segment+message");
    }
    // Approach B is the CONTROL and reads no scope at all: it has no compiler,
    // is shown the whole document and the whole message, and makes one call
    // whatever the policy declares. A B arm whose unit moved with the policy
    // would be B doing something the method does not do.
    for (const family of ["baseline-b", "baseline-b-tier0"] as const) {
      for (const ir of [PAIRED_IR, MESSAGE_ONLY_IR, BOTH_SCOPES_IR]) {
        expect(judgedUnitFor(family, ir.semanticPredicates)).toBe("message");
      }
    }
  });

  it("refuses to name a unit for a compiled arm with no predicate to judge", () => {
    // Not a defensive flourish: `WebLlmJudge.judge` returns an empty verdict
    // before it touches the engine when the IR declares no predicate, so such an
    // arm judges no unit of any kind and every answer here would be a made-up
    // one. `planBakeoff` refuses that IR one layer up; this is the same refusal
    // where the unit is named, for the direct caller that skipped the planner.
    expect(() => judgedUnitFor("compiled", [])).toThrow(/declares no semanticPredicates/);
    // B still answers, and correctly: it is shown the document rather than the
    // predicates, so a policy with no semantic clause costs it exactly the same
    // one call per message.
    expect(judgedUnitFor("baseline-b", [])).toBe("message");
  });

  it("plans a both-scopes arm over the union of the two samples, one per call", () => {
    // The hard case, and the decision it records: on a policy declaring both
    // scopes no single existing unit describes the work, so the union is
    // measured and the row SAYS it is a union. The rejected alternative was
    // keeping the sample segment-based and naming the message call elsewhere,
    // which leaves `judgedUnitChars` describing some of the arm's prompts while
    // reading as all of them.
    const plan = planBakeoff({
      options: options({ families: ["compiled", "baseline-b"] }),
      ir: BOTH_SCOPES_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    const compiled = plan.arms.find((a) => a.family === "compiled")!;
    expect(compiled.segments.unit).toBe("segment+message");
    // One sample per engine call: every selected segment PLUS one whole message
    // an item. 13 items, 18 segments selected under the tier-0 condition.
    expect(compiled.segments.count).toBe(SEGMENTS_TIER0.count + ITEMS.length);
    expect(SEGMENTS_TIER0.count).toBe(18);
    // THE ORACLE, and it is not this module's own arithmetic: the union's
    // character samples are exactly the segment rule's samples plus one entry
    // per item holding that ITEM'S OWN LENGTH, which is a fact about the corpus
    // file. Under the rejected rule the second half is missing entirely.
    const ascending = (xs: readonly number[]) => [...xs].sort((a, b) => a - b);
    expect(ascending(compiled.segments.samples.chars)).toEqual(
      ascending([...SEGMENTS_TIER0.samples.chars, ...ITEMS.map((i) => i.text.length)]),
    );
    // The maxima cannot separate the two rules on this corpus and the median
    // can: the largest item is 153 characters and escalation selects a segment
    // covering all of it, so both rules report max 153, while the medians are
    // 62 (segments), 78 (messages) and 71 for the union of the two. The 71 is
    // the nearest-rank 16th of the 31 samples the assertion above just pinned
    // as a multiset, so it is that multiset's median and not a number read off
    // this arm's own `chars` summary.
    expect(compiled.segments.chars!.max).toBe(153);
    expect(SEGMENTS_TIER0.chars!.max).toBe(153);
    expect(compiled.segments.count).toBe(31);
    expect(compiled.segments.chars!.p50).toBe(71);
    expect(SEGMENTS_TIER0.chars!.p50).toBe(62);
    expect(MESSAGES_B_TIER0.chars!.p50).toBe(78);
    // Approach B is untouched by the second scope.
    const b = plan.arms.find((a) => a.family === "baseline-b")!;
    expect(b.segments.unit).toBe("message");
    expect(b.segments.count).toBe(ITEMS.length);
  });

  it("plans a message-scoped arm on a corpus where escalation selects nothing", () => {
    // The refusal below exists to stop an arm writing "a complete, schema-valid
    // transcript of a model that never ran". On a message-scoped policy that is
    // not what a zero segment count means: the judge calls the model once about
    // the whole message whether or not a single segment was selected. Refusing
    // here would drop the compiled arm from a code-heavy corpus while both B
    // arms ran -- an artifact of the harness, not of either method.
    const plan = planBakeoff({
      options: options({ families: ["compiled", "compiled-tier2-only"] }),
      ir: MESSAGE_ONLY_IR,
      items: CODE_ONLY_ITEMS,
    });
    // The precondition, and it has to be measured on the SEGMENT rule now that
    // the arm is not planned under it: segments exist on this corpus, and
    // escalation selects none of them under either tier-0 condition.
    for (const priors of [undefined, { priorFindings: () => [] }]) {
      const bySegment = segmentSizeDistribution(CODE_ONLY_ITEMS, {
        hasPredicates: true,
        ...priors,
      });
      expect(bySegment.segmentsTotal).toBe(2);
      expect(bySegment.count).toBe(0);
      expect(bySegment.perItem).toEqual({ p50: 0, p95: 0, max: 0, min: 0 });
    }
    for (const arm of plan.arms) {
      // And the arm is planned over the unit it is actually shown: one whole
      // message per item, which is the call it makes whatever escalation did.
      expect(arm.segments.unit, arm.arm).toBe("message");
      expect(arm.segments.segmentsTotal, arm.arm).toBe(2);
      expect(arm.segments.count, arm.arm).toBe(2);
      expect(arm.segments.perItem, arm.arm).toEqual({ p50: 1, p95: 1, max: 1, min: 1 });
      expect(arm.maxCallsPerItem, arm.arm).toBe(2);
    }
  });

  it("still refuses an arm whose ONLY calls would be segment calls it will not make", () => {
    // The control for the test above: the refusal has to keep firing where it
    // is true, or the fix has traded a one-sided refusal for a one-sided
    // silence. Same corpus, same families, and the only difference is the scope
    // the predicate is declared in.
    expect(() =>
      planBakeoff({
        options: options({ families: ["compiled-tier2-only"] }),
        ir: PAIRED_IR,
        items: CODE_ONLY_ITEMS,
      }),
    ).toThrow(/would make no engine call on any of the 2 corpus items/);
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

  it("escalates further WITH tier 0 than without, on the IR the bake-off actually runs", () => {
    // Task 9 measured two conditions and the plan quotes both: with tier-0
    // priors p50 1, p95 3, max 3; without them p50 1, p95 2, max 2. They were
    // measured against `minimal-ir.json`, and for one commit this fixture --
    // then the only IR a tier-2 arm could run -- carried `rules: []`, so both
    // numbers described a
    // policy the bake-off never executed, and the two arms escalated alike.
    //
    // The premise assertion below is the guard. Strip the rules again and this
    // fails here, loudly, rather than in a results table where two arms agree
    // for a reason that is not about tier 0.
    expect(SEMANTIC_IR.rules.length).toBeGreaterThan(0);
    const plan = planBakeoff({
      options: options({ families: ["compiled", "compiled-tier2-only"] }),
      ir: SEMANTIC_IR,
      items: ITEMS,
    });
    const withTier0 = plan.arms.find((a) => a.family === "compiled")!;
    const without = plan.arms.find((a) => a.family === "compiled-tier2-only")!;
    expect(withTier0.segments.perItem).not.toEqual(without.segments.perItem);
    expect(withTier0.segments.perItem).toEqual({ p50: 1, p95: 3, max: 3, min: 1 });
    expect(without.segments.perItem).toEqual({ p50: 1, p95: 2, max: 2, min: 1 });
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
    // And the segment sizes escalation selects for the arm, which ARE in
    // characters and so are directly comparable to the ~1.1 kB the threshold
    // was derived at.
    expect(r.judgedUnitChars).toEqual(SEGMENTS_TIER0.chars);
  });

  it("carries THIS family's escalation, on the one field where the two differ", () => {
    // `judgedUnitChars` cannot tell the two conditions apart on this corpus and
    // that is a measured fact, not a weakness of the fixture: the segment tier 0
    // re-admits is 66 characters, larger than nine of the seventeen the
    // predicate branch selects and smaller than the largest, so it moves neither
    // the median nor the maximum. Asserting `judgedUnitChars` alone therefore
    // proves nothing about WHICH distribution a report carries. `judgedUnitsPerItem`
    // is where the two separate -- max 3 against max 2 -- so it is the field
    // this test reads.
    expect(SEGMENTS_TIER0.chars).toEqual(SEGMENTS.chars);
    expect(SEGMENTS_TIER0.perItem).not.toEqual(SEGMENTS.perItem);

    const tier0 = report([judged([call()])], "compiled");
    expect(tier0.judgedUnitsPerItem).toEqual({ p50: 1, p95: 3, max: 3, min: 1 });
    expect(tier0.escalation.hasPriors).toBe(true);

    const tier2Only = report([judged([call()])], "compiled-tier2-only");
    expect(tier2Only.judgedUnitsPerItem).toEqual({ p50: 1, p95: 2, max: 2, min: 1 });
    expect(tier2Only.escalation.hasPriors).toBe(false);
  });

  it("refuses the other family's distribution rather than reporting it as this arm's", () => {
    // The population check. `judgedUnitChars`, `judgedUnitsPerItem` and `escalation`
    // are the only fields on a gate report that do not come off the rows -- they
    // are the PLAN's, measured before the arm ran -- so nothing else in the
    // report contradicts a distribution belonging to the other family. This file
    // itself shipped that mistake for a round: one shared no-priors constant,
    // handed to `compiled` reports.
    expect(() =>
      gateReport({
        arm: "tier2-" + MODEL,
        family: "compiled",
        modelId: MODEL,
        records: [judged([call()])],
        segments: SEGMENTS,
        ...RUN_CONTEXT,
      }),
    ).toThrow(/WITHOUT tier-0 priors/);
    expect(() =>
      gateReport({
        arm: "tier2only-" + MODEL,
        family: "compiled-tier2-only",
        modelId: MODEL,
        records: [judged([call()], {}, { arm: "tier2only-" + MODEL })],
        segments: SEGMENTS_TIER0,
        ...RUN_CONTEXT,
      }),
    ).toThrow(/WITH tier-0 priors/);
    // And a distribution selected at a different escalation threshold is the
    // same defect through the other input: `uncertainBelow` is an experiment
    // variable every record carries, so plan and run can disagree on it.
    expect(() =>
      gateReport({
        arm: "tier2-" + MODEL,
        family: "compiled",
        modelId: MODEL,
        records: [judged([call()])],
        segments: segmentSizeDistribution(ITEMS, {
          hasPredicates: true,
          uncertainBelow: 0.99,
          priorFindings: (item) => runTier0(SEMANTIC_IR, item.text, segmentText(item.text)),
        }),
        ...RUN_CONTEXT,
      }),
    ).toThrow(/0\.99/);
  });

  it("does not kill an arm that produced no call to measure", () => {
    // An arm whose every message ran out of budget before its first call has no
    // TTFT sample at all. Killing it here would be killing it for the budget
    // overrun by the back door.
    const r = report([judged([], {}, { degraded: [{ tier: 2, reason: "budget-exhausted", detail: "spent" }] })]);
    expect(outcome(r, "p95-ttft").verdict).toBe("not-measured");
    expect(outcome(r, "p95-ttft").observed).toBeUndefined();
    expect(r.killedOnRunGates).toBe(false);
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
    expect(r.killedOnRunGates).toBe(true);
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
    expect(r.killedOnRunGates).toBe(false);
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
    expect(primary.killedOnRunGates).toBe(false);

    // ... and it still kills the pathology it is for: Plan 5 measured
    // Phi-4-mini looping `"quote": "Halcyon"` about nineteen times until the
    // token budget ran out mid-string. One distinct span and eighteen
    // restatements is 0.947, an answer that is nothing but one thing said again.
    const looping = report([judged([call()], { rung1: 1, duplicatesDropped: 18 })]);
    expect(outcome(looping, "duplicate-rate").observed).toBeCloseTo(18 / 19, 10);
    expect(outcome(looping, "duplicate-rate").verdict).toBe("fail");
    expect(looping.killedOnRunGates).toBe(true);
  });

  it("is not measured on an arm that emitted no finding at all", () => {
    const r = report([judged([call()])]);
    expect(outcome(r, "resolvable-rate").verdict).toBe("not-measured");
    expect(outcome(r, "duplicate-rate").verdict).toBe("not-measured");
  });
});

describe("every gate, exactly ON its threshold", () => {
  // The four numeric gates are four comparators, and until this block existed
  // three of them had never been exercised at the number they compare against:
  // every fixture sat comfortably on one side. So `<=` against `<` and `>=`
  // against `>` were free changes, and an arm sitting exactly on a gate could
  // have been killed or passed either way -- on a threshold that is a judgement
  // call in three cases out of four.
  //
  // The direction each gate is inclusive in is a decision, and it is the same
  // one every time: the threshold VALUE PASSES. A ceiling of 1,500 ms means
  // 1,500 is allowed, and a floor of 25 tok/s means 25 is allowed -- which
  // matters most for `minDecodeTokPerSec`, whose own comment records that
  // Qwen3-4B's measured 25 tok/s sits "exactly on the line". Under `>` that
  // comment would describe a killed arm.

  it("passes a p95 time-to-first-token exactly at the ceiling, and fails one millisecond over", () => {
    expect(outcome(report([judged([call({ ttftMs: GATES.maxP95TtftMs })])]), "p95-ttft").verdict).toBe("pass");
    expect(outcome(report([judged([call({ ttftMs: GATES.maxP95TtftMs + 1 })])]), "p95-ttft").verdict).toBe("fail");
  });

  it("passes a sustained decode rate exactly at the floor, and fails just under", () => {
    // 50 tokens at 25 tok/s is 2 s, and 50/2 is 25 exactly -- both halves are
    // the engine's own numbers, so no rounding stands between the fixture and
    // the comparison. Qwen3-4B's manifest rate is this number.
    const onTheLine = report([judged([call({ completionTokens: 50, decodeTokPerSec: GATES.minDecodeTokPerSec })])]);
    expect(onTheLine.sustainedDecodeTokPerSec).toBe(GATES.minDecodeTokPerSec);
    expect(outcome(onTheLine, "decode-rate").verdict).toBe("pass");
    expect(onTheLine.killedOnRunGates).toBe(false);

    const under = report([judged([call({ completionTokens: 50, decodeTokPerSec: 24 })])]);
    expect(outcome(under, "decode-rate").verdict).toBe("fail");
  });

  it("passes a resolvable rate exactly at the floor, and fails just under", () => {
    // 4 of 5 is 0.8, and the double 4/5 rounds to is the double the literal 0.8
    // is, so this really is the comparator at its threshold rather than near it.
    const onTheLine = report([judged([call()], { rung1: 4, unresolvedQuotes: 1 })]);
    expect(outcome(onTheLine, "resolvable-rate").observed).toBe(GATES.minResolvableRate);
    expect(outcome(onTheLine, "resolvable-rate").verdict).toBe("pass");

    const under = report([judged([call()], { rung1: 79, unresolvedQuotes: 21 })]);
    expect(outcome(under, "resolvable-rate").observed).toBeCloseTo(0.79, 10);
    expect(outcome(under, "resolvable-rate").verdict).toBe("fail");
  });

  it("passes a duplicate rate exactly at the ceiling, and fails just over", () => {
    const onTheLine = report([judged([call()], { rung1: 1, duplicatesDropped: 9 })]);
    expect(outcome(onTheLine, "duplicate-rate").observed).toBe(GATES.maxDuplicateRate);
    expect(outcome(onTheLine, "duplicate-rate").verdict).toBe("pass");

    const over = report([judged([call()], { rung1: 9, duplicatesDropped: 91 })]);
    expect(outcome(over, "duplicate-rate").observed).toBeCloseTo(0.91, 10);
    expect(outcome(over, "duplicate-rate").verdict).toBe("fail");
  });
});

describe("the p95 gate is the 95th percentile and not a neighbour of it", () => {
  it("reads the nearest-rank 95th value, which no fixture under n = 20 can distinguish", () => {
    // The fixture-too-small-for-the-rank trap, and this project has already
    // written the note about it in `segments.test.ts`: at n < 20 the nearest-rank
    // p95 IS the maximum, so every one- and two-call fixture in this file is
    // equally satisfied by p95, p96, p99 and max. n = 100 is the size at which
    // p94, p95, p96 and max are four different answers.
    //
    // 10..1000 by tens: rank ceil(95 * 100 / 100) = 95 -> 950. The neighbours
    // are 940 and 960 and the maximum is 1000, so a gate computing any of them
    // reports a different number here.
    const tens = Array.from({ length: 100 }, (_, i) => call({ ttftMs: (i + 1) * 10 }));
    const spread = report([judged(tens)]);
    expect(spread.ttftCalls).toBe(100);
    expect(outcome(spread, "p95-ttft").observed).toBe(950);
    expect(spread.ttftMs).toEqual({ p50: 500, p95: 950, max: 1000, min: 10 });

    // And the rank decides the VERDICT, not just the number. 95 calls at 1,000 ms
    // and 5 at 9,000: the 95th value is 1,000 and passes, while the maximum --
    // or a p96, or a mean -- is 9,000 and kills the arm. An arm whose slowest
    // twentieth is slow is exactly what a p95 exists to tolerate.
    const tail = report([
      judged([
        ...Array.from({ length: 95 }, () => call({ ttftMs: 1000 })),
        ...Array.from({ length: 5 }, () => call({ ttftMs: 9000 })),
      ]),
    ]);
    expect(outcome(tail, "p95-ttft").observed).toBe(1000);
    expect(outcome(tail, "p95-ttft").verdict).toBe("pass");
    expect(tail.ttftMs!.max).toBe(9000);
    expect(tail.killedOnRunGates).toBe(false);
  });

  it("says on a small sample that the p95 it reports is the maximum", () => {
    // The fact above is true of every run this repository can currently take --
    // 13 items, at most 18 calls on a compiled arm and 13 on a B arm -- and it
    // was recorded nowhere a reader of the number would meet it. A ceiling
    // called "p95" that is really a ceiling on the single slowest call is a
    // different gate, and one slow first call kills the arm under it.
    const small = report([judged([call({ ttftMs: 100 }), call({ ttftMs: 900 })])]);
    expect(outcome(small, "p95-ttft").sample).toBe(2);
    expect(outcome(small, "p95-ttft").observed).toBe(900);
    expect(outcome(small, "p95-ttft").detail).toContain("nearest-rank p95 IS THE MAXIMUM");

    // And it stops saying it once the sample is big enough for the two to
    // differ, rather than being a fixed sentence. 20 is the first such size.
    const twenty = report([judged(Array.from({ length: 20 }, (_, i) => call({ ttftMs: (i + 1) * 10 })))]);
    expect(outcome(twenty, "p95-ttft").sample).toBe(20);
    expect(outcome(twenty, "p95-ttft").observed).toBe(190);
    expect(twenty.ttftMs!.max).toBe(200);
    expect(outcome(twenty, "p95-ttft").detail).not.toContain("IS THE MAXIMUM");

    // THE NUMBER 20 ITSELF, computed here rather than trusted. The two
    // assertions above constrain it only to the interval (2, 20]: at 10, or at
    // 3, the caveat would silently vanish from the real production run -- this
    // corpus produces up to 18 calls on a compiled arm, inside the window --
    // and the sentence emitted would read "ceil(0.95*n) = n for every n < 3",
    // which is false. `P95_EQUALS_MAX_BELOW`'s docblock calls the value
    // COMPUTED; this is the computation, from `percentile`'s own nearest-rank
    // rule and not from the constant.
    let firstDivergence: number | undefined;
    for (let n = 1; n <= 40 && firstDivergence === undefined; n += 1) {
      const sample = Array.from({ length: n }, (_, i) => (i + 1) * 10);
      if (percentile(sample, 95) !== Math.max(...sample)) firstDivergence = n;
    }
    expect(firstDivergence).toBe(20);
    // And the constant the DETAIL interpolates is that number, read out of the
    // string a reader meets rather than imported: the docblock says two places
    // must not disagree about it and nothing enforced that either.
    expect(outcome(small, "p95-ttft").detail).toContain(
      `ceil(0.95*n) = n for every n < ${String(firstDivergence)}`,
    );
  });

  it("names the prompt-size mismatch only on the arm that has one", () => {
    // The clause used to read "so on a message-judged arm (judgedUnit
    // \"segment\")" on EVERY row -- a sentence contradicting the value it quotes,
    // on the one line a reader checks a gate's derivation against. The ceiling
    // was derived at a whole prompt built from one SEGMENT, so the mismatch is
    // real on a B arm and absent on a compiled one, and the detail now says
    // which of the two this row is.
    expect(outcome(report([judged([call()])]), "p95-ttft").detail).not.toContain(
      "judged per MESSAGE",
    );
  });
});

describe("every ladder counter reaches the report", () => {
  it("sums all eleven across items, each at a distinct value", () => {
    // Seven of the eleven were summed into `ArmGateReport.ladder` and asserted
    // nowhere, so a swapped or dropped `+=` shipped silently -- and `ladder` is
    // what the gate comments tell a reader to recompute a disputed rate from.
    // Distinct primes per counter, so no two can be confused for each other, and
    // TWO items, so a `=` written for a `+=` is caught as well.
    const counters = {
      rung1: 2,
      rung2: 3,
      unresolvedQuotes: 5,
      duplicatesDropped: 7,
      unknownPredicates: 11,
      failedClosed: 13,
      truncatedResponses: 17,
      abortedResponses: 19,
      repairAttempts: 23,
      segmentsJudged: 29,
      segmentsSkipped: 31,
      messageScopeCalls: 37,
      messageScopeJudged: 41,
      messageScopeFailedClosed: 43,
    } as const;
    const r = report([judged([call()], counters), judged([call()], counters)]);
    // The two counters `BaselineStats` renames arrive under the report's own
    // method-neutral names -- see `ArmGateReport.ladder` -- so this asserts the
    // MAPPING as well as the sum: a `normalizeArmStats` that read
    // `segmentsSkipped` where it means `segmentsJudged` swaps 29 and 31 here.
    //
    // `unitsJudged` is the one entry that is a SUM rather than a rename: the
    // compiled judge has two kinds of judged unit and B has one, so it is
    // `segmentsJudged + messageScopeJudged`, 2 x (29 + 41). The primes still
    // separate every misread of it -- taking `segmentsSkipped` for
    // `segmentsJudged` gives 144, `messageScopeCalls` for `messageScopeJudged`
    // gives 132, and dropping either term gives 58 or 82.
    expect(r.ladder).toEqual({
      rung1: 4,
      rung2: 6,
      unresolvedQuotes: 10,
      duplicatesDropped: 14,
      unknownLabels: 22,
      failedClosed: 26,
      truncatedResponses: 34,
      abortedResponses: 38,
      repairAttempts: 46,
      unitsJudged: 140,
      unitsSkipped: 62,
      // `undefined` and not 0 on a compiled arm: the judge has no message-budget
      // counter at all -- `detect` files a `budget-exhausted` notice instead --
      // and a 0 would be the positive claim that the event happened zero times.
      messageBudgetExpiries: undefined,
      // The whole-message calls, which are a compiled-arm event: Approach B's
      // single call IS its message call and reporting it here too would
      // double-count it.
      messageScopeCalls: 74,
      messageScopeJudged: 82,
      messageScopeFailedClosed: 86,
    });
    // The three stop counters are NOT in `ladder` and must not be: they are the
    // stop-gate's evidence, not the ladder's, and a report that summed them here
    // would be reporting an interruption as a judgement.
    expect(Object.keys(r.ladder).sort()).toEqual([
      "abortedResponses",
      "duplicatesDropped",
      "failedClosed",
      "messageBudgetExpiries",
      "messageScopeCalls",
      "messageScopeFailedClosed",
      "messageScopeJudged",
      "repairAttempts",
      "rung1",
      "rung2",
      "truncatedResponses",
      "unitsJudged",
      "unitsSkipped",
      "unknownLabels",
      "unresolvedQuotes",
    ]);
  });
});

describe("the duplicate rate's denominator counts both rungs", () => {
  it("includes rung 2, which no other fixture drives beside a duplicate", () => {
    // `quotesResolved = rung1 + rung2 + duplicatesDropped` is the denominator of
    // the duplicate rate and the numerator of the resolvable one, and no fixture
    // had both `rung2 > 0` and `duplicatesDropped > 0` -- so dropping the rung2
    // term from that sum changed nothing anywhere. It is not a rare shape: rung 2
    // is the ladder's normalised match, which is how a model that rewrites
    // whitespace or casing gets placed at all.
    const r = report([judged([call()], { rung1: 1, rung2: 1, duplicatesDropped: 2 })]);
    expect(outcome(r, "duplicate-rate").sample).toBe(4);
    expect(outcome(r, "duplicate-rate").observed).toBe(0.5);
    // Without the rung2 term the same fixture reads 2 of 3, a rate of 0.667 --
    // a 33% overstatement of how much of this arm's output was restatement.
    expect(outcome(r, "duplicate-rate").observed).not.toBeCloseTo(2 / 3, 3);
    expect(outcome(r, "resolvable-rate").observed).toBe(1);
    expect(outcome(r, "resolvable-rate").sample).toBe(4);
    expect(outcome(r, "resolvable-rate").detail).toContain("rung 2: 1");
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
    expect(r.killedOnRunGates).toBe(true);
  });

  it("names the FIRST stop, not the last, when two items stop", () => {
    // The latch is `if (stoppedAt === undefined && stats.stops > 0)`, and no
    // fixture in this file had two stopping items -- so last-write-wins survived
    // the whole suite. `stoppedAt` is interpolated into this gate's detail three
    // times, and under the mutant a two-stop arm reads "the first engine call
    // after the stop on item B answered normally (on item B)": self-contradictory,
    // and naming the wrong item as the one the engine had to survive.
    const stop = (itemId: string) =>
      judged([], { deadlineExpiries: 1, segmentsSkipped: 1 }, { itemId });
    const r = report([
      judged([call()], {}, { itemId: "item-1" }),
      stop("item-2"),
      judged([call()], {}, { itemId: "item-3" }),
      stop("item-4"),
      judged([call()], {}, { itemId: "item-5" }),
    ]);
    const detail = outcome(r, "non-empty-after-stop").detail;
    expect(detail).toContain('the stop on item "item-2"');
    expect(detail).not.toContain('the stop on item "item-4"');
    // The call the engine had to survive is the one right after the FIRST stop.
    expect(detail).toContain('answered normally (on item "item-3")');
    expect(outcome(r, "non-empty-after-stop").verdict).toBe("pass");
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

  it("arms on ALL THREE stop counters, not only on a deadline expiry", () => {
    // `deadlineExpiries + callerAbortsMidGeneration + callerAbortsWhileQueued`
    // is the stop condition, and only the first term had a fixture -- so two
    // stop causes could have been dropped from that sum without a test noticing,
    // and an engine latched by a caller abort would have been reported as
    // `not-measured` rather than as the fail it is.
    //
    // The two abort counters are not hypothetical: `cancel.ts` distinguishes an
    // abort that arrives MID-GENERATION from one that arrives while the call is
    // still QUEUED, and the orchestrator's message deadline can deliver either.
    for (const counter of ["deadlineExpiries", "callerAbortsMidGeneration", "callerAbortsWhileQueued"] as const) {
      const poisoned = report([
        judged([call()]),
        judged([], { [counter]: 1, segmentsSkipped: 1 }),
        judged([call({ finishReason: "abort", completionTokens: 0 })], { abortedResponses: 1 }),
      ]);
      expect(outcome(poisoned, "non-empty-after-stop").verdict, counter).toBe("fail");

      const survived = report([
        judged([call()]),
        judged([], { [counter]: 1, segmentsSkipped: 1 }),
        judged([call()], { rung1: 1 }),
      ]);
      expect(outcome(survived, "non-empty-after-stop").verdict, counter).toBe("pass");
    }
  });

  it("is not measured when a stop was the LAST engine work in the arm", () => {
    // The third branch of the gate, and it had no test at all: something
    // stopped, and nothing after it ever reached the engine. Both other answers
    // are wrong here. "Pass" would assert an engine survived an interrupt
    // nothing afterwards asked it to survive -- the single most valuable claim
    // this gate makes, invented. "Fail" would kill the arm for running out of
    // budget on its last item, which is the one rule this module must not have.
    //
    // It is the ordinary shape of a stopped run, not a corner: the message
    // budget is spent monotonically across an arm, so the stop that ends the run
    // tends to be near the end of the corpus.
    const r = report([
      judged([call()], { rung1: 1 }, { itemId: "healthy" }),
      judged([call()], { deadlineExpiries: 1, segmentsSkipped: 1 }, { itemId: "the-stop" }),
      // A later item, but one that made no engine call: the budget was already
      // spent before its first segment, so the judge never reached the engine.
      judged([], { segmentsSkipped: 2 }, {
        itemId: "after-the-stop",
        degraded: [{ tier: 2, reason: "budget-exhausted", detail: "spent" }],
      }),
    ]);
    const stop = outcome(r, "non-empty-after-stop");
    expect(stop.verdict).toBe("not-measured");
    expect(stop.detail).toContain("no later item made an engine call");
    // Named, so a reader knows WHICH stop went unexamined rather than being told
    // only that something did -- and it is the STOPPED item that is named, not
    // the healthy one before it or the silent one after.
    expect(stop.detail).toContain("the-stop");
    expect(stop.detail).not.toContain("healthy");
    expect(stop.detail).not.toContain("after-the-stop");
    expect(stop.sample).toBe(0);
    expect(r.killedOnRunGates).toBe(false);
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
    expect(r.killedOnRunGates).toBe(false);
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
      // The caveat AUDIT-10 asked for, in the machine-readable half: every
      // latency on this report was taken at 24x the budget the compiler emits
      // for a policy that does not name one.
      compilerDefaultLatencyBudgetMs: COMPILER_DEFAULT_LATENCY_BUDGET_MS,
      latencyBudgetTimesCompilerDefault: 24,
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

  it("states what the arm cost to stand up, from the caller's load report", () => {
    // WHY. Every other latency on this report is a per-CALL number taken once
    // the model is resident, so nothing in either output file said what getting
    // it there cost -- and "at what latency AND HARDWARE COST" is half the
    // question this project exists to answer. The page measures all three and
    // `runBakeoff` already holds the report they come from.
    const r = report([rec()]);
    expect(r.engineLoadMs).toBe(1_234);
    expect(r.engineWarmupMs).toBe(567);
    expect(r.originStorageBytes).toBe(8_900_000_000);

    // A SECOND load, because a report that hardcoded the first set -- or that
    // wrote `engineLoadMs` into all three -- would satisfy the block above.
    const other = report([rec()], "compiled", {
      load: { engineLoadMs: 90_001, engineWarmupMs: 2, originStorageBytes: 33 },
    });
    expect(other.engineLoadMs).toBe(90_001);
    expect(other.engineWarmupMs).toBe(2);
    expect(other.originStorageBytes).toBe(33);
  });

  it("names the axes of the experiment a run of this driver cannot cross", () => {
    // A gates file is read by someone without the spec open, and the available
    // misreading is that it answers the research question. It does not: the
    // backend, policy and hardware axes of spec 6.3's matrix are one point each
    // here, and no accuracy metric exists in this repository at all.
    //
    // Pinned as CONTENT rather than as a non-empty string, because a sentence
    // that stopped saying the load-bearing half would still be a sentence.
    const scope = report([rec()]).experimentScope;
    expect(scope).toContain("WebGPU or absent");
    expect(scope).toContain("one of the three policy documents has a compiled artifact");
    expect(scope).toContain("one machine and one GPU");
    expect(scope).toContain("It answers NOTHING about 'how well ... prevent leakage'");
    // The model and method points are READ OFF THE SLATE, not asserted. This
    // was a constant naming "the four pinned arms" and four methods whatever
    // ran -- so `SIH_BAKEOFF=1 ... pnpm -C apps/eval bakeoff`, whose
    // `DEFAULT_FAMILIES` is `["compiled"]` alone, put a claim of a
    // compiled-versus-Approach-B head-to-head on every row of a one-method run.
    // RUN_CONTEXT's slate is one model and two compiled families, so a
    // hardcoded four survives nothing here.
    expect(scope).toContain("MODEL axis at 1 point(s) (" + MODEL + ")");
    expect(scope).toContain(
      "METHOD axis at 2 point(s) (tier 0 + compiled judge, compiled judge)",
    );
    expect(scope).not.toContain("the four pinned arms");
    // A DIFFERENT slate produces a different sentence, which is what separates
    // "reads the slate" from "happens to agree with this fixture".
    const four = report([rec()], "compiled", {
      slate: {
        models: [MODEL, "Phi-4-mini-instruct-q4f16_1-MLC"],
        families: ["compiled", "baseline-b"],
      },
    }).experimentScope;
    expect(four).toContain("MODEL axis at 2 point(s)");
    expect(four).toContain("METHOD axis at 2 point(s) (tier 0 + compiled judge, Approach B)");

    // Identical across the arms of ONE run on the axes that come off the plan,
    // because the plan is one object: two rows of a gates file disagreeing
    // about what the RUN crossed would be the worse artifact. They may differ
    // on the scoring clause, and should -- two families run different tier sets
    // -- so the comparison is of the sentence up to that clause.
    const upToScoring = (text: string) => text.slice(0, text.indexOf("and because "));
    expect(upToScoring(report([rec()], "compiled-tier2-only").experimentScope)).toBe(
      upToScoring(scope),
    );
  });

  it("points the accuracy clause at this row's own scoring rather than restating it", () => {
    // The clause used to end "the corpus carries no gold for the tier every arm
    // here runs" -- a property of `options.corpus`, which SIH_BAKEOFF_CORPUS
    // makes free and which Plan 7 exists to change. On a corpus carrying tier-2
    // gold the SAME ROW would have said `tiersTheseRowsCannotScore: []` in the
    // data and "there is no gold" in the prose.
    const withGold = rec({
      gold: [{ start: 0, end: 5, text: "hello", entityType: "pred:unannounced-deal", action: "redact" }],
    });
    const clean = report([withGold], "compiled-tier2-only");
    expect(clean.scoring.tiersTheseRowsCannotScore).toEqual([]);
    expect(clean.scoring.cannotScore).toEqual([]);
    expect(clean.experimentScope).toContain("scoring.cannotScore on this row is empty");
    expect(clean.experimentScope).not.toContain("way(s) these rows cannot be joined to gold");

    const bare = report([rec()], "compiled-tier2-only");
    expect(bare.scoring.cannotScore.length).toBeGreaterThan(0);
    expect(bare.experimentScope).toContain(
      `scoring.cannotScore on this row names ${String(bare.scoring.cannotScore.length)} way(s)`,
    );
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

  it("reads each option that has a default, at a value that is NOT the default", () => {
    // The rule this project learned twice in one task: a test that exercises
    // only the default config cannot tell "reads the option" from "hardcodes the
    // default". Four `BakeoffOptions` knobs were only ever seen at their
    // defaults here -- `contextWindowSize`, `callBudgetMs`, `lowerTierAllowanceMs`
    // and `uncertainBelow` -- and each of them reaches something that decides how
    // an arm runs or how its rows are read.
    //
    // Every value below differs from the default AND from every other value in
    // this file, so a hardcode shows up as the default coming back.
    const plan = planBakeoff({
      options: options({
        contextWindowSize: 4096,
        callBudgetMs: 12_345,
        lowerTierAllowanceMs: 2_500,
        uncertainBelow: 0.42,
        itemTimeoutMs: 400_000,
      }),
      ir: SEMANTIC_IR,
      items: ITEMS,
    });
    const arm = plan.arms[0]!;
    expect(arm.contextWindowSize).toBe(4096);
    expect(arm.callBudgetMs).toBe(12_345);
    // Into the TierConfig `detect` receives and every record carries, not merely
    // into the plan: `uncertainBelow` is the escalation threshold a scorer groups
    // rows by.
    expect(arm.config.uncertainBelow).toBe(0.42);
    // And into the escalation the ceiling is sized from. 0.42 is below the
    // corpus's tier-0 confidences, so the tier-0 arm's extra segment goes away
    // and this arm plans 2 segments rather than 3 -- the knob is not merely
    // stored, it changed the plan.
    expect(arm.segments.escalation.uncertainBelow).toBe(0.42);
    expect(arm.segments.perItem).toEqual({ p50: 1, p95: 2, max: 2, min: 1 });
    expect(arm.maxCallsPerItem).toBe(4);
    // The allowance is the third summand of the bound, and 2,500 is not 1,000.
    // At a 12,345 ms per-call budget the CALL bound binds rather than the
    // message budget -- 4 x 12,365 is under 120,020 -- so both knobs are visible
    // in this one number and the default for either would change it.
    expect(arm.bound.input.lowerTierAllowanceMs).toBe(2_500);
    expect(arm.bound.bindingBound).toBe("call-budget");
    expect(arm.bound.boundMs).toBe(2_500 + 4 * (12_345 + INTERRUPT_DRAIN_OVERSHOOT_MS));

    // The defaults, asserted beside them, so "reads the option" and "applies the
    // documented default" are two separate claims rather than one.
    const defaults = planBakeoff({ options: options(), ir: SEMANTIC_IR, items: ITEMS }).arms[0]!;
    expect(defaults.callBudgetMs).toBe(DEFAULT_TIER2_CALL_BUDGET_MS);
    expect(defaults.config.uncertainBelow).toBe(UNCERTAIN_BELOW);
    expect(defaults.bound.input.lowerTierAllowanceMs).toBe(DEFAULT_LOWER_TIER_ALLOWANCE_MS);
  });

  it("holds the same per-call budget the PAGE defaults to, checked against its source", () => {
    // The page declares its own `DEFAULT_TIER2_CALL_BUDGET_MS` and does not
    // export it -- it is a Vite entry point, not a module this driver can import
    // -- so the driver keeps a copy, and a copy with nothing comparing it is a
    // number free to drift. `runBakeoff` does compare them at run time, by
    // passing this value to `loadTier2` and refusing an arm whose page resolved
    // a different one, but that refusal needs a GPU and this ceiling is derived
    // on every machine.
    //
    // So the page's literal is read out of its SOURCE. Anchored on the
    // declaration and asserted to have matched, so a rename fails here loudly
    // rather than by matching nothing.
    expect(DEFAULT_TIER2_CALL_BUDGET_MS).toBe(60_000);
    const pageSource = readFileSync(join(REPO_ROOT, "apps", "eval", "src", "page", "main.ts"), "utf8");
    const declared = /^const DEFAULT_TIER2_CALL_BUDGET_MS = ([\d_]+);$/m.exec(pageSource);
    expect(declared, "the page no longer declares DEFAULT_TIER2_CALL_BUDGET_MS").not.toBeNull();
    expect(Number(declared![1]!.replaceAll("_", ""))).toBe(DEFAULT_TIER2_CALL_BUDGET_MS);

    // And it is the number the plan actually uses, not just a constant sitting
    // beside one: the ceiling is derived from it.
    const plan = planBakeoff({ options: options(), ir: SEMANTIC_IR, items: ITEMS });
    expect(plan.arms.map((a) => a.callBudgetMs)).toEqual([DEFAULT_TIER2_CALL_BUDGET_MS]);
    expect(plan.arms[0]!.bound.callBudgetBoundMs).toBe(
      plan.arms[0]!.maxCallsPerItem * (DEFAULT_TIER2_CALL_BUDGET_MS + INTERRUPT_DRAIN_OVERSHOOT_MS),
    );
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
    // 6 calls and 2 calls produce the same ceiling and a "take the smallest"
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
    // 6 calls x (10,000 + 20) against 2 x (10,000 + 20): the per-call budget is
    // now the binding bound for both, and it separates them. Six because the
    // compiled family runs tier 0, whose priors re-admit a third segment to
    // escalation (max 3 segments x the one repair retry); the baseline family
    // makes one call per message by design.
    expect(compiled.bound.bindingBound).toBe("call-budget");
    expect(compiled.bound.boundMs).toBe(1_000 + 6 * 10_020);
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

  it("demands the document when ANY family needs it, not only when every one does", () => {
    // `needsPolicy = families.some(...)`, and every fixture that exercised it
    // passed a slate of baseline arms ONLY -- on which `some` and `every` return
    // the same answer, so the quantifier was a free change. Under `every` the
    // mixed slate below plans happily and the B arm runs with no document, which
    // is Approach B's entire premise missing: it would be shown nothing and lose
    // to the compiled arm for a reason invisible in every number of the run.
    const mixed: ArmFamily[] = ["compiled", "baseline-b"];
    expect(mixed.some((f) => !familyShape(f).runsCompiledJudge)).toBe(true);
    expect(mixed.every((f) => !familyShape(f).runsCompiledJudge)).toBe(false);
    expect(() =>
      planBakeoff({ options: options({ families: mixed }), ir: SEMANTIC_IR, items: ITEMS }),
    ).toThrow(/policyPath/);
    // The same slate with a document that does not match the IR is refused on
    // the hash instead, which proves the demand reached the check rather than
    // stopping at "some string was supplied".
    expect(() =>
      planBakeoff({
        options: options({ families: mixed }),
        ir: SEMANTIC_IR,
        items: ITEMS,
        policyText: "# Some other standard\n",
      }),
    ).toThrow(/policyHash/);
    // And a compiled-only slate never asks for one.
    expect(() =>
      planBakeoff({ options: options({ families: ["compiled"] }), ir: SEMANTIC_IR, items: ITEMS }),
    ).not.toThrow();
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

  it("lets a paired baseline slate through, which it refused outright before", () => {
    // The three reasons this function used to give -- no page door, no record
    // field, no compiled IR -- are all closed, and this is what says so from
    // the outside. Each is now checked where it can actually be checked:
    // `loadBaseline` in the page, `baselineStats` plus `detector` in
    // `RunRecordSchema`, and `policies/compiled/p-fin.ir.json` whose
    // `policyHash` is a real sha256. `assertPageCanRun` is left with the one
    // condition it can still decide from a plan alone.
    const plan = planBakeoff({
      options: options({ families: ["baseline-b", "baseline-b-tier0"] }),
      ir: PAIRED_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    expect(plan.arms).toHaveLength(2);
    expect(() => assertPageCanRun(plan)).not.toThrow();
    // And the plan carries the document, which is the thing the arms need.
    // `name` here is `DEFAULT_IR_NAME`, because `options()` sets neither
    // `policyName` nor `irName`.
    expect(plan.policy).toEqual({
      name: "semantic",
      sha256: POLICY_SHA256,
      chars: POLICY_TEXT.length,
    });
  });

  it("takes the policy registry name from policyName, then irName, then the default", () => {
    // ALL THREE LEVELS, because only the innermost was ever exercised and
    // hardcoding the whole expression therefore survived: `name: "semantic"`,
    // `name: DEFAULT_IR_NAME` and `name: options.irName ?? DEFAULT_IR_NAME` each
    // passed the entire suite. `plan.policy.name` is what `runBakeoff` hands the
    // page for `policyDocHash(name)` and `loadBaseline({policy: name})`, and the
    // page's POLICY_FIXTURES holds only "p-fin" -- so a regression here aborts a
    // slate with "unknown policy semantic" rather than failing a test. It is
    // also what made the field's docblock disagree with the code for a round.
    const named = (over: Partial<Parameters<typeof options>[0]>) =>
      planBakeoff({
        options: options({ families: ["baseline-b"], ...over }),
        ir: PAIRED_IR,
        items: ITEMS,
        policyText: POLICY_TEXT,
      }).policy!.name;
    expect(named({ policyName: "p-fin", irName: "semantic" })).toBe("p-fin");
    // The default, and the reason it is `irName`: the page's two registries are
    // keyed by the SAME name for the two halves of one compile.
    expect(named({ irName: "p-fin" })).toBe("p-fin");
    expect(named({})).toBe("semantic");
  });

  it("refuses a baseline arm whose plan carries no policy document", () => {
    // Unreachable through `planBakeoff`, which refuses a baseline family with no
    // policy one layer up -- so this drives the spliced shape a direct caller
    // can build. It is worth keeping rather than deleting as dead: Approach B
    // IS the policy document, and an arm built without one would show its model
    // an empty prompt and record the result as "the method found nothing".
    const compiled = planBakeoff({ options: options(), ir: SEMANTIC_IR, items: ITEMS });
    const spliced = {
      ...compiled,
      arms: [{ ...compiled.arms[0]!, arm: "baselineB-x", family: "baseline-b" as const }],
    };
    expect(() => assertPageCanRun(spliced)).toThrow(/no policy document/);
  });

  it("no longer claims tier 0 finds nothing, which was true for one commit", () => {
    // `semantic-ir.json` shipped `rules: []`, so tier 0 found nothing on any
    // item and `baseline-b-tier0` versus `baseline-b` -- the pair that separates
    // "compiling helps" from "patterns help" -- would have been two runs of the
    // same thing. The fixture carries three rules now and that pair really does
    // differ, which is what makes the intermediate arm worth running.
    expect(SEMANTIC_IR.rules.length).toBeGreaterThan(0);
    const plan = planBakeoff({
      options: options({ families: ["baseline-b", "baseline-b-tier0"] }),
      ir: PAIRED_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    const [b, bTier0] = plan.arms;
    expect(b!.segments.escalation.hasPriors).toBe(false);
    expect(bTier0!.segments.escalation.hasPriors).toBe(true);
    // Both are MESSAGE distributions, because B makes one call per message
    // however many segments the text has -- so unlike the compiled pair these
    // two agree on every size and differ only in whether tier 0 ran.
    expect(b!.segments.unit).toBe("message");
    expect(bTier0!.segments.unit).toBe("message");
    expect(b!.segments.chars).toEqual(bTier0!.segments.chars);
    expect(b!.segments.perItem).toEqual({ p50: 1, p95: 1, max: 1, min: 1 });
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

describe("killedOnRunGates is the disjunction of the failures", () => {
  it("is set by any one failing gate and by no not-measured one", () => {
    const oneFailure = report([judged([call({ ttftMs: 9000 })], { rung1: 10 })]);
    expect(oneFailure.gates.filter((g) => g.verdict === "fail").map((g) => g.gate)).toEqual(["p95-ttft"]);
    expect(oneFailure.killedOnRunGates).toBe(true);

    const twoFailures = report([judged([call({ ttftMs: 9000, completionTokens: 100, decodeTokPerSec: 2 })], { rung1: 1, unresolvedQuotes: 9 })]);
    expect(twoFailures.gates.filter((g) => g.verdict === "fail").map((g) => g.gate).sort()).toEqual([
      "decode-rate",
      "p95-ttft",
      "resolvable-rate",
    ]);
    expect(twoFailures.killedOnRunGates).toBe(true);

    const nothingMeasured = report([judged([])]);
    expect(nothingMeasured.gates.every((g) => g.verdict === "not-measured")).toBe(true);
    expect(nothingMeasured.killedOnRunGates).toBe(false);
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
    // Proved on the SHIPPED fixture, which is the only one the bake-off can run.
    // For one commit `semantic-ir.json` declared `rules: []`, so `runTier0` found
    // nothing, no prior made a segment uncertain, and these two arms escalated
    // identically -- two of the four contrasts null by construction, with the
    // published numbers agreeing for a reason that had nothing to do with tier 0.
    //
    // 17 and 18 are Task 9's numbers, measured in test/segments.test.ts against
    // the same corpus and the same escalation policy but through a different
    // caller, so they are not this module's arithmetic restated.
    const plan = planBakeoff({
      options: options({ families: ["compiled", "compiled-tier2-only"] }),
      ir: SEMANTIC_IR,
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

// ---------------------------------------------------------------------------
// What this corpus can score, and what it cannot
// ---------------------------------------------------------------------------

describe("what the shipped corpus can and cannot score", () => {
  it("carries no gold at all for the tier the bake-off exists to measure", () => {
    // Established on the two DATA files, not on this module's arithmetic: the
    // counts below were taken by hand off `corpora/fixtures/smoke.jsonl` and
    // `apps/eval/fixtures/semantic-ir.json` and are written here as literals, so
    // a change to either file fails this rather than moving with it.
    const tierOf = new Map(SEMANTIC_IR.entityTypes.map((e) => [e.id, e.tier]));
    expect([...tierOf].sort()).toEqual([
      ["aws-key", 0],
      ["client-name", 1],
      ["generic-secret", 0],
      ["in-pan", 0],
      ["pred:unannounced-deal", 2],
    ]);

    const gold = ITEMS.flatMap((item) => item.gold);
    expect(ITEMS).toHaveLength(13);
    expect(gold).toHaveLength(7);
    expect([...new Set(gold.map((g) => g.entityType))].sort()).toEqual([
      "aws-key",
      "client-name",
      "generic-secret",
      "in-pan",
    ]);
    // The IR declares every entityType the gold uses, so the gap is tier 2's
    // alone and not a corpus labelled against a policy this IR does not know.
    expect(gold.filter((g) => !tierOf.has(g.entityType))).toEqual([]);
    // 5 at tier 0, 2 at tier 1, and NONE at tier 2 -- while every arm of this
    // bake-off runs tier 2 and only tier 2.
    expect(gold.filter((g) => tierOf.get(g.entityType) === 0)).toHaveLength(5);
    expect(gold.filter((g) => tierOf.get(g.entityType) === 1)).toHaveLength(2);
    expect(gold.filter((g) => tierOf.get(g.entityType) === 2)).toEqual([]);
  });

  it("carries items whose TEXT satisfies the predicate and no label saying so", () => {
    // The half that makes the missing gold an inversion rather than a blank: a
    // model that answers the predicate CORRECTLY on these three produces
    // findings with nothing to match, so a scorer joining findings to gold
    // counts each of them a false positive and ranks the arm that found nothing
    // as the most precise. The three readings are mine, from the text:
    //
    //   pos-client-name-prose         "draft a contract renewal email for
    //                                  Tamarind Grocers" -- a customer contract
    //                                  renewal, in an unsent internal request.
    //   pos-multiline-pan-and-client  "The Halcyon Logistics account needs a
    //                                  renewal quote" -- a renewal being priced.
    //   pos-emoji-before-pan          "Deal closed!" -- a concluded negotiation,
    //                                  announced to accounts rather than
    //                                  publicly. The weakest of the three.
    //
    // Asserted as SUBSTRINGS of the shipped text rather than as a verdict,
    // because whether a passage satisfies an NL predicate is the model's
    // judgement and not a test's. What the test pins is that the words are
    // there and the label is not.
    expect(SEMANTIC_IR.semanticPredicates.map((p) => p.id)).toEqual(["unannounced-deal"]);
    expect(SEMANTIC_IR.semanticPredicates[0]!.nlPredicate).toContain(
      "customer contract, renewal, or negotiation that has not been publicly announced",
    );
    const suspects = [
      ["pos-client-name-prose", "contract renewal"],
      ["pos-multiline-pan-and-client", "renewal quote"],
      ["pos-emoji-before-pan", "Deal closed"],
    ] as const;
    for (const [id, phrase] of suspects) {
      const item = ITEMS.find((i) => i.id === id);
      expect(item, `corpus no longer carries ${id}`).toBeDefined();
      expect(item!.text).toContain(phrase);
      expect(item!.gold.map((g) => g.entityType)).not.toContain("pred:unannounced-deal");
    }
  });

  it("escalates each of those to the judge, so the wrongly-scored finding is reachable", () => {
    // Without this the gap would be theoretical: a segment escalation never
    // selects is a segment no arm judges, and a finding no arm can produce
    // cannot be miscounted. Measured through the driver's OWN distribution
    // function rather than a second copy of the escalation rule.
    const ids = ["pos-client-name-prose", "pos-multiline-pan-and-client", "pos-emoji-before-pan"];
    const three = ITEMS.filter((item) => ids.includes(item.id));
    expect(three).toHaveLength(3);
    // One prose segment each, and escalation keeps all three under BOTH
    // conditions -- so this holds for the tier-0 families and the tier-2-only
    // families alike.
    const withoutPriors = segmentSizeDistribution(three, { hasPredicates: true });
    const withPriors = segmentSizeDistribution(three, {
      hasPredicates: true,
      priorFindings: (item) => runTier0(SEMANTIC_IR, item.text, segmentText(item.text)),
    });
    expect(withoutPriors.segmentsTotal).toBe(3);
    expect(withoutPriors.count).toBe(3);
    expect(withPriors.count).toBe(3);
  });
});

describe("a tier the corpus cannot score is a named result, not a silent zero", () => {
  /** Gold and a finding that agree, on the fixture record's own text. */
  const PRED = "pred:unannounced-deal";
  const tier2Gold = { start: 0, end: 5, text: "hello", entityType: PRED, action: "redact" } as const;
  const tier0Gold = { start: 6, end: 11, text: "world", entityType: "in-pan", action: "block" } as const;
  const tier2Finding = {
    start: 0,
    end: 5,
    text: "hello",
    entityType: PRED,
    severity: "high",
    tier: 2,
    source: MODEL,
    confidence: 0.9,
    action: "redact",
  } as const;

  /** A tier-0-and-tier-2 arm, so the only scoring gap in a fixture is the one under test. */
  const BOTH_TIERS = { tier0: true, tier1: false, tier2: true, uncertainBelow: UNCERTAIN_BELOW };

  it("names tier 2 unscorable when the arm ran it and the rows carry no gold for it", () => {
    const r = report([judged([call()], { rung1: 1 }, { config: BOTH_TIERS, gold: [tier0Gold] })]);
    expect(r.scoring.tiersTheseRowsCannotScore).toEqual([2]);
    expect(r.scoring.goldSpansByTier).toEqual({ 0: 1, 1: 0, 2: 0 });
    expect(r.scoring.cannotScore).toHaveLength(1);
    expect(r.scoring.cannotScore[0]).toContain("tier 2");
    // The consequence, named in the artifact rather than left to be worked out.
    expect(r.scoring.cannotScore[0]).toContain("false positive");
    // WHICH entityTypes the IR declares at the unscorable tier, which is the
    // half of the sentence only the IR knows and the only place a gates row
    // says it. Asserted against the fixture IR's own tier-2 ids rather than
    // against a literal, so "declares 0 entityType(s)" and "declares every
    // tier's ids" are both red -- both survived the whole suite before this.
    const tier2Ids = SEMANTIC_IR.entityTypes.filter((e) => e.tier === 2).map((e) => e.id);
    expect(tier2Ids.length).toBeGreaterThan(0);
    expect(r.scoring.cannotScore[0]).toContain(
      `The IR declares ${String(tier2Ids.length)} entityType(s) at tier 2 (${tier2Ids.join(", ")})`,
    );
    const tier0Ids = SEMANTIC_IR.entityTypes.filter((e) => e.tier === 0).map((e) => e.id);
    for (const id of tier0Ids) expect(r.scoring.cannotScore[0]).not.toContain(id);
  });

  it("names NO tier unscorable once the rows do carry gold at that tier", () => {
    // The discriminator. Without it, a field hardcoded to "[2]" -- which is the
    // right answer for every corpus in this repository -- passes the test above.
    const r = report([
      judged([call()], { rung1: 1 }, { config: BOTH_TIERS, gold: [tier0Gold, tier2Gold] }),
    ]);
    expect(r.scoring.tiersTheseRowsCannotScore).toEqual([]);
    expect(r.scoring.tiersWithGoldThisArmDidNotRun).toEqual([]);
    expect(r.scoring.cannotScore).toEqual([]);
    expect(r.scoring.goldSpansByTier).toEqual({ 0: 1, 1: 0, 2: 1 });
    expect(r.scoring.tiers.find((t) => t.tier === 2)!.goldEntityTypes).toEqual([PRED]);
  });

  it("names gold at a tier the arm did NOT run, which caps recall by construction", () => {
    // THE MIRROR, and it was missing: the tier/gold join comes back empty in
    // two directions and only one of them had a field. On the shipped corpus
    // this is `client-name` at tier 1 -- two of seven gold spans, at a tier
    // `planBakeoff` switches off on every arm it plans -- so every arm's recall
    // denominator silently included spans it could not produce.
    const r = report([
      judged([call()], { rung1: 1 }, { gold: [tier0Gold, tier2Gold] }),
    ]);
    expect(r.scoring.tiersRun).toEqual([2]);
    expect(r.scoring.tiersWithGoldThisArmDidNotRun).toEqual([0]);
    expect(r.scoring.tiersTheseRowsCannotScore).toEqual([]);
    expect(r.scoring.cannotScore).toHaveLength(1);
    // The BOUND, not just the fact: a recall over record.gold cannot exceed
    // 1 of the 2 gold spans on this row whatever the model does.
    expect(r.scoring.cannotScore[0]).toContain("did NOT run tier 0");
    expect(r.scoring.cannotScore[0]).toContain("bounded above by 1/2");
    expect(r.scoring.cannotScore[0]).toContain("(in-pan)");
  });

  it("reads which tiers RAN off config on the rows, not off the family label", () => {
    // Standing rule: a field describing what ran is populated from what ran.
    // Both reports below are the same family and differ only in the config
    // `detect` was handed, so a `familyShape(family).runsTier0` implementation
    // answers identically for the two and fails here.
    const cfg = (tier0: boolean) => ({
      tier0,
      tier1: false,
      tier2: true,
      uncertainBelow: UNCERTAIN_BELOW,
    });
    const withTier0 = report([judged([call()], { rung1: 1 }, { config: cfg(true), gold: [tier0Gold] })]);
    const withoutTier0 = report([judged([call()], { rung1: 1 }, { config: cfg(false), gold: [tier0Gold] })]);
    expect(withTier0.scoring.tiersRun).toEqual([0, 2]);
    expect(withoutTier0.scoring.tiersRun).toEqual([2]);
    // And a tier with gold that the arm did NOT run IS a complaint, in the
    // mirror field: the gold is there, this arm cannot produce it, and a recall
    // denominator that includes it is wrong by exactly that much.
    expect(withoutTier0.scoring.tiers.find((t) => t.tier === 0)!.goldSpans).toBe(1);
    expect(withoutTier0.scoring.tiersTheseRowsCannotScore).toEqual([2]);
    expect(withoutTier0.scoring.tiersWithGoldThisArmDidNotRun).toEqual([0]);
    expect(withTier0.scoring.tiersWithGoldThisArmDidNotRun).toEqual([]);
  });

  it("names a gold entityType the IR declares no tier for, which scores as nothing at all", () => {
    // The other way a join comes back empty: gold written against a different
    // policy, which is exactly what `smoke.jsonl`'s `policy: "minimal-fixture"`
    // warns about. Such a span belongs to no tier bucket, so the per-tier counts
    // alone would report it as absent rather than as unmatchable.
    //
    // TWO strays, in reverse alphabetical order on the row, because one cannot
    // tell a sorted list from an insertion-ordered one -- and this list is read
    // by a human comparing two arms' gates rows.
    const salary = { start: 6, end: 11, text: "world", entityType: "salary", action: "redact" } as const;
    const codename = { start: 0, end: 5, text: "hello", entityType: "codename", action: "redact" } as const;
    const r = report([judged([call()], { rung1: 1 }, { gold: [tier2Gold, salary, codename] })]);
    expect(r.scoring.goldEntityTypesNotInIr).toEqual(["codename", "salary"]);
    expect(r.scoring.cannotScore.join(" ")).toContain("salary");
    // Its tier-2 sibling still counts, so this is an addition and not a veto.
    expect(r.scoring.goldSpansByTier[2]).toBe(1);
  });

  it("reports the policy the gold was labelled under, off the rows", () => {
    const r = report([
      // `p-fin` FIRST, so insertion order is not sorted order: a row a human
      // compares against another arm's has to be stably ordered.
      judged([call()], { rung1: 1 }, { itemId: "a", policy: "p-fin" }),
      judged([call()], { rung1: 1 }, { itemId: "b", policy: "minimal-fixture" }),
    ]);
    // A SET, not a single value: a corpus may hold items labelled under two
    // policies, and refusing that here would be a new rule about corpora rather
    // than a report of one.
    expect(r.scoring.goldPolicies).toEqual(["minimal-fixture", "p-fin"]);
  });

  it("refuses rows that ran DIFFERENT tiers, which describe no single arm", () => {
    // The same refusal `gateReport` already makes for two IRs or two context
    // windows, applied to the one setting `scoring.tiersRun` is read from. A
    // report summing a tier-0 row and a tier-2-only row would state one tier set
    // for work done under two, and `tiersTheseRowsCannotScore` would be
    // computed against a tier half the rows never ran.
    const cfg = (tier0: boolean) => ({
      tier0,
      tier1: false,
      tier2: true,
      uncertainBelow: UNCERTAIN_BELOW,
    });
    expect(() =>
      report([
        judged([call()], { rung1: 1 }, { itemId: "a", config: cfg(true) }),
        judged([call()], { rung1: 1 }, { itemId: "b", config: cfg(false) }),
      ]),
    ).toThrow(/disagree on config's tier switches/);
  });
});

describe("no verdict on a gate report is a selection", () => {
  const PRED = "pred:unannounced-deal";
  const finding = {
    start: 0,
    end: 5,
    text: "hello",
    entityType: PRED,
    severity: "high",
    tier: 2,
    source: MODEL,
    confidence: 0.9,
    action: "redact",
  } as const;
  const matching = { start: 0, end: 5, text: "hello", entityType: PRED, action: "redact" } as const;

  it("gives two arms the SAME verdict when only their correctness differs", () => {
    // The whole of AUDIT-3 in one assertion. Arm A answered the predicate and
    // its answer is labelled; arm B produced the identical finding against a
    // corpus that labels nothing. One is right and one is unmatchable, their
    // throughput is byte-identical, and every gate verdict is identical --
    // because no gate here reads a gold label. `killedOnRunGates` says so in its
    // name; `scoring` is the only field that notices the difference.
    const rows = (gold: readonly (typeof matching)[]) => [
      judged([call()], { rung1: 1 }, { findings: [finding], gold: [...gold] }),
    ];
    const right = report(rows([matching]));
    const unmatchable = report(rows([]));
    expect(right.killedOnRunGates).toBe(unmatchable.killedOnRunGates);
    expect(right.gates).toEqual(unmatchable.gates);
    expect(right.scoring.tiersTheseRowsCannotScore).toEqual([]);
    expect(unmatchable.scoring.tiersTheseRowsCannotScore).toEqual([2]);
  });

  it("carries no gate whose subject is accuracy", () => {
    const r = report([judged([call()], { rung1: 1 })]);
    // A list, not a pattern: the property is "every gate is a property of the
    // run", and a new gate must break this line and force the argument.
    expect(r.gates.map((g) => g.gate)).toEqual([
      "p95-ttft",
      "decode-rate",
      "resolvable-rate",
      "duplicate-rate",
      "non-empty-after-stop",
    ]);
    expect(r.scoring.accuracyGated).toBe(false);
  });

  it("states the boundary in the ARTIFACT, not only in the plan", () => {
    const r = report([judged([call()], { rung1: 1 })]);
    // A reader holding only the gates file has to be able to learn that spec
    // 4.2's primary criterion was never applied. These substrings are the
    // contract; changing the wording is fine, dropping the facts is not.
    expect(r.scoring.verdictMeans).toContain("accuracy");
    expect(r.scoring.verdictMeans).toContain("killedOnRunGates");
    expect(r.scoring.verdictMeans).toContain("Plan 8");
    expect(r).not.toHaveProperty("killed");
  });
});

describe("the message budget these latencies were taken under", () => {
  it("is the compiler's own default, so a drift is a failure here", () => {
    // The same treatment `DEFAULT_TIER2_CALL_BUDGET_MS` gets one layer down, and
    // for the same reason: `@sih/compiler` is not a dependency of `@sih/eval`
    // (it is Node-only and shells out to an SDK), so the number cannot be shared
    // by reference and a copy with nothing comparing it is free to drift.
    expect(COMPILER_DEFAULT_LATENCY_BUDGET_MS).toBe(5_000);
    const emitSource = readFileSync(
      join(REPO_ROOT, "packages", "compiler", "src", "stages", "emit.ts"),
      "utf8",
    );
    const declared = /^export const DEFAULT_LATENCY_BUDGET_MS = ([\d_]+);$/m.exec(emitSource);
    expect(declared, "the compiler no longer declares DEFAULT_LATENCY_BUDGET_MS").not.toBeNull();
    expect(Number(declared![1]!.replaceAll("_", ""))).toBe(COMPILER_DEFAULT_LATENCY_BUDGET_MS);
  });

  it("is reported as a multiple of that default, and read from the run", () => {
    // TWO budgets, because a report taken only at the fixture's 120,000 cannot
    // tell "computes the multiple" from "hardcodes 24".
    const r = report([judged([call({ ttftMs: 700 })], { rung1: 1 })]);
    expect(r.run.latencyBudgetMs).toBe(120_000);
    expect(r.run.compilerDefaultLatencyBudgetMs).toBe(5_000);
    expect(r.run.latencyBudgetTimesCompilerDefault).toBe(24);

    const shipped = report([judged([call({ ttftMs: 700 })], { rung1: 1 })], "compiled", {
      latencyBudgetMs: 7_500,
    });
    expect(shipped.run.latencyBudgetTimesCompilerDefault).toBe(1.5);
  });

  it("caveats the latency numbers where a reader of them would look", () => {
    // AUDIT-10: every latency here was taken at 24x the budget a compiled
    // policy emits, so the degradation a shipped policy would cause is not in
    // this file. The structured fields above are the machine-readable half; a
    // reader looking at the p95 itself has to be told beside the number.
    const r = report([judged([call({ ttftMs: 700 })], { rung1: 1 })]);
    for (const gate of ["p95-ttft", "decode-rate"]) {
      const detail = outcome(r, gate).detail;
      expect(detail, `${gate} does not name the budget its numbers were taken under`).toContain(
        "120000ms",
      );
      expect(detail).toContain("24x");
    }
    // And the caveat moves with the budget rather than being a fixed sentence.
    const shipped = report([judged([call({ ttftMs: 700 })], { rung1: 1 })], "compiled", {
      latencyBudgetMs: 5_000,
    });
    expect(outcome(shipped, "p95-ttft").detail).toContain("1x");
  });
});

/**
 * The gate report over an APPROACH-B arm.
 *
 * Everything in this describe was a SURVIVING MUTANT before it existed. No
 * vitest fixture built a `baselineStats` row, so `normalizeArmStats` could map
 * `unitsJudged` to `deadlineExpiries`, the family/detector check and the
 * family/unit check could both be deleted, and `unitsSkipped` could report 0
 * on an arm that has no such counter -- all with the whole suite green. The
 * browser spec exercises the real path, but a browser spec costs a model load
 * and cannot drive a counter to a chosen value.
 *
 * The mapping is the subject. `BaselineStats` renames two events and adds one
 * the judge does not have, so `ArmGateReport.ladder` is method-neutral and
 * `normalizeArmStats` is where a rename can silently become a swap. Every
 * counter below is a distinct prime for exactly that reason.
 */
describe("the gate report over an Approach-B arm", () => {
  type BStats = NonNullable<RunRecord["baselineStats"]>;

  const ZERO_B: Omit<BStats, "calls"> = {
    rung1: 0,
    rung2: 0,
    unresolvedQuotes: 0,
    unknownEntityTypes: 0,
    duplicatesDropped: 0,
    repairAttempts: 0,
    failedClosed: 0,
    truncatedResponses: 0,
    abortedResponses: 0,
    messagesJudged: 0,
    deadlineExpiries: 0,
    messageBudgetExpiries: 0,
    callerAbortsMidGeneration: 0,
    callerAbortsWhileQueued: 0,
  };

  /** One schema-valid Approach-B record, VALIDATED here rather than merely typed. */
  function bRec(
    calls: readonly Call[],
    counters: Partial<Omit<BStats, "calls">> = {},
    over: Partial<RunRecord> = {},
  ): RunRecord {
    const record: RunRecord = {
      schemaVersion: RECORD_SCHEMA_VERSION,
      runId: "bake",
      itemId: "item-1",
      policy: "semantic-fixture",
      irHash: "a".repeat(64),
      policyHash: "test-hash",
      arm: "baselineB-" + MODEL,
      backend: "webgpu",
      detector: "approach-b",
      provider: "claude",
      // No `uncertainBelow`: B does not escalate, and the schema refuses one.
      config: { tier0: false, tier1: false, tier2: true, t2Model: MODEL },
      tier2Config: {
        modelId: MODEL,
        contextWindowSize: 8192,
        temperature: 0,
        maxTokens: 512,
        callBudgetMs: 60_000,
      },
      text: "Please review the Northwind Traders renewal before Friday.",
      findings: [],
      gold: [],
      timings: { tier0Ms: 0.2, tier2Ms: 3600 },
      degraded: [],
      error: null,
      abandonedWorkInFlight: false,
      baselineStats: { ...ZERO_B, ...counters, calls: [...calls] },
      ...over,
    };
    const parsed = RunRecordSchema.safeParse(record);
    if (!parsed.success) throw new Error(`fixture is not a valid record: ${parsed.error.message}`);
    return record;
  }

  const bReport = (records: readonly RunRecord[], family: ArmFamily = "baseline-b") =>
    gateReport({
      arm: "baselineB-" + MODEL,
      family,
      modelId: MODEL,
      records,
      segments: distributionFor(family),
      ...RUN_CONTEXT,
    });

  it("maps every renamed counter into the report's method-neutral names", () => {
    // Distinct primes per counter, so a field read from the wrong slot is
    // visible rather than merely possible, and TWO records so a `=` written for
    // a `+=` fails too. The three that are not the judge's carry the three
    // largest values.
    const counters = {
      rung1: 2,
      rung2: 3,
      unresolvedQuotes: 5,
      duplicatesDropped: 7,
      unknownEntityTypes: 11,
      failedClosed: 13,
      truncatedResponses: 17,
      abortedResponses: 19,
      repairAttempts: 23,
      messagesJudged: 29,
      messageBudgetExpiries: 31,
    } as const;
    const r = bReport([bRec([call()], counters), bRec([call()], counters)]);
    expect(r.ladder).toEqual({
      rung1: 4,
      rung2: 6,
      unresolvedQuotes: 10,
      duplicatesDropped: 14,
      // `unknownEntityTypes` on this arm, `unknownPredicates` on the other.
      unknownLabels: 22,
      failedClosed: 26,
      truncatedResponses: 34,
      abortedResponses: 38,
      repairAttempts: 46,
      // `messagesJudged` here, `segmentsJudged` there -- and they are DIFFERENT
      // events, not a rename of taste: one counts messages and the other
      // segments, and both are the denominator every rate is taken over.
      unitsJudged: 58,
      // `undefined` and not 0: an arm that makes one call per message has no
      // second unit for a stop to skip past, and a 0 would be the positive
      // claim that it skipped none.
      unitsSkipped: undefined,
      messageBudgetExpiries: 62,
      // `undefined` for the same reason: B's ONE call is its message call and
      // is already counted as a judged unit, so a message-scope column here
      // would report the same call twice.
      messageScopeCalls: undefined,
      messageScopeJudged: undefined,
      messageScopeFailedClosed: undefined,
    });
  });

  it("carries the message distribution, not the segment one, and refuses the swap", () => {
    const r = bReport([bRec([call()])]);
    expect(r.judgedUnit).toBe("message");
    // One judged unit per item, which is the whole of what "one call per
    // message" costs -- against max 2 or 3 on the compiled families.
    expect(r.judgedUnitsPerItem).toEqual({ p50: 1, p95: 1, max: 1, min: 1 });
    // Whole MESSAGES. MEASURED over this corpus, and the shape is worth
    // stating because half of it is counter-intuitive: the message p50 is 78
    // characters against the selected-segment p50 of 62, and the two MAXIMA are
    // both 153 -- because the largest item, `pos-multiline-pan-and-client`, is
    // 153 characters and escalation selects a segment covering all of it. So a
    // report that carried the segment distribution under a B arm's name would
    // agree on the maximum and disagree on the median and the count (13 judged
    // units against 17), which is exactly why the swap is refused rather than
    // spotted.
    expect(r.judgedUnitChars).toEqual({ p50: 78, p95: 153, max: 153, min: 48 });
    expect(SEGMENTS.chars).toEqual({ p50: 62, p95: 153, max: 153, min: 22 });
    expect(r.judgedUnitChars!.max).toBe(Math.max(...ITEMS.map((i) => i.text.length)));
    // Escalation did not select this sample, and the row says so rather than
    // reporting a threshold that decided nothing.
    expect(r.escalation.applies).toBe(false);
    expect(SEGMENTS.escalation.applies).toBe(true);
    // And the swap is refused rather than reported.
    expect(() =>
      gateReport({
        arm: "baselineB-" + MODEL,
        family: "baseline-b",
        modelId: MODEL,
        records: [bRec([call()])],
        segments: SEGMENTS,
        ...RUN_CONTEXT,
      }),
    ).toThrow(/one message per engine call.*measured over segments/s);
  });

  it("refuses to file an Approach-B arm's rows under a compiled family", () => {
    // The check that reads `detector` off the ROWS rather than trusting the
    // family a caller named. The gates are the same numbers either way and
    // would look perfectly well-formed; what would be wrong is which METHOD
    // they are attributed to, which is the only question the bake-off asks.
    // The arm NAME is made to agree deliberately, so this drives the detector
    // check and not the foreign-arm one that guards it. A mislabelled family is
    // exactly the case where the name would agree: `armName` builds it from the
    // family the plan asked for.
    expect(() =>
      gateReport({
        arm: "tier2-" + MODEL,
        family: "compiled-tier2-only",
        modelId: MODEL,
        records: [bRec([call()], {}, { arm: "tier2-" + MODEL })],
        segments: SEGMENTS,
        ...RUN_CONTEXT,
      }),
    ).toThrow(/rows say the detector was "approach-b"/);
    // And the mirror: a compiled arm's rows filed under a baseline family.
    expect(() =>
      gateReport({
        arm: "tier2-" + MODEL,
        family: "baseline-b",
        modelId: MODEL,
        records: [judged([call()])],
        segments: MESSAGES_B,
        ...RUN_CONTEXT,
      }),
    ).toThrow(/rows say the detector was "core-orchestrator"/);
  });

  it("computes the same gates over B's calls as over the judge's", () => {
    // The point of normalising into one shape: every gate below is one
    // expression over both methods. A B arm's TTFT is B's TTFT, and the ceiling
    // it is compared against is the same 1,500 ms -- which is exactly why
    // `judgedUnit` and `promptTokens` are on the row: that ceiling was derived
    // at a ~1.1 kB prompt built from ONE SEGMENT, and B's prompt carries the
    // whole policy document.
    const r = bReport([bRec([call({ ttftMs: 2700, promptTokens: 1433 })], { rung1: 1, messagesJudged: 1 })]);
    expect(r.answeredCalls).toBe(1);
    expect(outcome(r, "p95-ttft").verdict).toBe("fail");
    expect(outcome(r, "p95-ttft").detail).toContain("judgedUnitChars");
    // The mismatch clause, which appears on a message-judged arm and not on a
    // segment-judged one -- see "names the prompt-size mismatch only on the arm
    // that has one". It used to be printed on both, quoting a `judgedUnit` that
    // contradicted the sentence around it.
    expect(outcome(r, "p95-ttft").detail).toContain("this arm is judged per MESSAGE");
    expect(r.promptTokens).toEqual({ p50: 1433, p95: 1433, max: 1433, min: 1433 });
    expect(outcome(r, "resolvable-rate").observed).toBe(1);
  });

  it("reads B's own message-budget stop as a stop, which the judge has no counter for", () => {
    // `messageBudgetExpiries` is in the fold `normalizeArmStats` hands the
    // engine-poisoning walk, and it belongs there: `createArm` in
    // `baselineB.ts` `break`s the loop on it exactly as it does on a per-call
    // expiry, so it is a stop that ended the run for that message.
    const stopped = bRec([call()], { messageBudgetExpiries: 1 }, { itemId: "stopped" });
    const after = bRec([call({ finishReason: "abort", completionTokens: 0 })], {}, { itemId: "later" });
    const r = bReport([stopped, after]);
    expect(outcome(r, "non-empty-after-stop").verdict).toBe("fail");
    expect(outcome(r, "non-empty-after-stop").detail).toContain("stopped");
  });

  it("separates the two Approach-B families by their tier-0 half", () => {
    // The pair that makes "compiling helps" separable from "patterns help".
    // Both are message-judged, so their size distributions agree exactly; what
    // differs is whether tier 0 ran in front of the model.
    const plain = bReport([bRec([call()])], "baseline-b");
    const withTier0 = bReport(
      [bRec([call()], {}, { config: { tier0: true, tier1: false, tier2: true, t2Model: MODEL } })],
      "baseline-b-tier0",
    );
    expect(plain.escalation.hasPriors).toBe(false);
    expect(withTier0.escalation.hasPriors).toBe(true);
    expect(plain.judgedUnitChars).toEqual(withTier0.judgedUnitChars);
    expect(plain.scoring.tiersRun).toEqual([2]);
    expect(withTier0.scoring.tiersRun).toEqual([0, 2]);
  });
});

describe("a gates row reports the unit its arm was actually judged on", () => {
  /**
   * The head-to-head's shape, on a gate report rather than in the planner.
   *
   * `policies/compiled/p-fin.ir.json` declares ONE predicate and declares it
   * `scope: "message"`, so a COMPILED arm on it judges whole messages and makes
   * no segment call at all. Every fixture elsewhere in this file runs
   * `SEMANTIC_IR`, whose one predicate is segment-scoped, so none of them can
   * tell "the unit is the family's" from "the unit is the family's and the
   * policy's". These can, and the run they describe is the one
   * `test/baseline.spec.ts` performs on a GPU.
   */
  const MESSAGE_SCOPED: PolicyIr["semanticPredicates"] = SEMANTIC_IR.semanticPredicates.map((p) => ({
    ...p,
    scope: "message",
  }));

  /** A compiled+tier0 arm's rows as the head-to-head produced them: one collected message call. */
  const messageJudged = (over: Partial<Stats> = {}) =>
    judged([call()], { segmentsJudged: 0, messageScopeCalls: 1, messageScopeJudged: 1, ...over }, {
      config: { tier0: true, tier1: false, tier2: true, uncertainBelow: UNCERTAIN_BELOW },
    });

  const compiledOnMessagePolicy = (segments: Parameters<typeof gateReport>[0]["segments"]) =>
    gateReport({
      arm: "tier2-" + MODEL,
      family: "compiled",
      modelId: MODEL,
      records: [messageJudged()],
      ...RUN_CONTEXT,
      semanticPredicates: MESSAGE_SCOPED,
      segments,
    });

  it("names MESSAGE on a compiled arm whose policy declares only message scope", () => {
    const r = compiledOnMessagePolicy(MESSAGES_B_TIER0);
    expect(r.judgedUnit).toBe("message");
    // The distribution is the MESSAGE one, which on this policy is the same
    // population Approach B's arm is planned over -- and that identity is the
    // finding, not a fixture accident: on a message-only policy the compiled
    // judge and B are shown the same text, and only the policy document B
    // carries makes the prompts differ.
    expect(r.judgedUnitChars).toEqual({ p50: 78, p95: 153, max: 153, min: 48 });
    expect(r.judgedUnitsPerItem).toEqual({ p50: 1, p95: 1, max: 1, min: 1 });
    // The segment distribution this row used to carry, for contrast: a p50 of
    // 62 characters and up to 3 units an item, for an arm that judged one whole
    // message and no segment.
    expect(SEGMENTS_TIER0.chars!.p50).toBe(62);
    expect(SEGMENTS_TIER0.perItem!.max).toBe(3);
    // The whole-message call is a JUDGED UNIT, so the column the planned
    // distribution is compared against counts it. It read 0 here.
    expect(r.ladder.unitsJudged).toBe(1);
    expect(r.ladder.messageScopeJudged).toBe(1);
    // And there is no segment loop on this arm, so there is no second unit for
    // a stop to skip past: `undefined` rather than the 0 a segment-judged arm
    // reports.
    expect(r.ladder.unitsSkipped).toBeUndefined();
    // The p95 gate says the ceiling was not derived at this arm's prompt size,
    // which it did not say on a compiled row before.
    expect(outcome(r, "p95-ttft").detail).toContain("this arm is judged per MESSAGE");
    // Escalation decided nothing here and the row says so, rather than quoting
    // a threshold that selected segments no call was made about.
    expect(r.escalation.applies).toBe(false);
    expect(r.escalation.uncertainBelow).toBe(UNCERTAIN_BELOW);
  });

  it("refuses the segment distribution its family alone would have named", () => {
    // The defect this closes, as a refusal: the same arm, the same rows, and
    // the distribution `familyShape` used to supply. It is well-formed, it is
    // this corpus, it is even this arm's own tier-0 condition -- and it
    // describes passages the model was never shown.
    expect(() => compiledOnMessagePolicy(SEGMENTS_TIER0)).toThrow(
      /is shown one message per engine call, but its size distribution was measured over segments/s,
    );
    // And the control: the same distribution under the same family is CORRECT
    // when the policy declares the predicate segment-scoped, so the refusal is
    // about the scopes and not about the family.
    expect(SEMANTIC_IR.semanticPredicates.map((p) => p.scope)).toEqual(["segment"]);
    expect(report([judged([call()])], "compiled").judgedUnit).toBe("segment");
  });

  it("reports a both-scopes arm on the union, and counts both scopes' judged units", () => {
    // No policy in this repository declares both scopes; this is the
    // constructed one. The row says `segment+message` rather than picking a
    // half, and `unitsJudged` counts the segment calls and the message call
    // together because both are units this arm's model was shown.
    const both: PolicyIr["semanticPredicates"] = [
      ...SEMANTIC_IR.semanticPredicates,
      { id: "board-confidential", nlPredicate: "whether the message names a board decision", scope: "message" },
    ];
    const segments = segmentSizeDistribution(ITEMS, {
      hasPredicates: true,
      unit: "segment+message",
      priorFindings: (item) => runTier0(SEMANTIC_IR, item.text, segmentText(item.text)),
    });
    const r = gateReport({
      arm: "tier2-" + MODEL,
      family: "compiled",
      modelId: MODEL,
      records: [messageJudged({ segmentsJudged: 2 })],
      ...RUN_CONTEXT,
      semanticPredicates: both,
      segments,
    });
    expect(r.judgedUnit).toBe("segment+message");
    expect(r.judgedUnitsPerItem).toEqual({ p50: 2, p95: 4, max: 4, min: 2 });
    // Two segments and one message on the one item that has rows.
    expect(r.ladder.unitsJudged).toBe(3);
    expect(r.ladder.messageScopeJudged).toBe(1);
    // This arm DOES have a segment loop, so the skip counter is a number again.
    expect(r.ladder.unitsSkipped).toBe(0);
    expect(r.escalation.applies).toBe(true);
    // And neither single-unit distribution is accepted for it.
    for (const wrong of [SEGMENTS_TIER0, MESSAGES_B_TIER0]) {
      expect(() =>
        gateReport({
          arm: "tier2-" + MODEL,
          family: "compiled",
          modelId: MODEL,
          records: [messageJudged({ segmentsJudged: 2 })],
          ...RUN_CONTEXT,
          semanticPredicates: both,
          segments: wrong,
        }),
      ).toThrow(/is shown one segment\+message per engine call/);
    }
  });
});

describe("what planBakeoff gives an Approach-B arm", () => {
  it("does not stamp an escalation threshold on it, and does on the compiled arms", () => {
    // `uncertainBelow` decides which SEGMENTS escalate. B judges the whole
    // message in one call and never escalates, so on a B arm it is a knob that
    // turned nothing -- and `gateReport` compares it against the threshold the
    // planned distribution was measured at, which would make the two agree
    // about work no B arm performs. `RunRecordSchema` refuses such a row, so
    // stamping it here would fail the whole arm after its GPU time was spent.
    const plan = planBakeoff({
      options: options({
        families: ["compiled", "compiled-tier2-only", "baseline-b", "baseline-b-tier0"],
        uncertainBelow: 0.42,
      }),
      ir: PAIRED_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    const byFamily = new Map(plan.arms.map((a) => [a.family, a]));
    expect(byFamily.get("compiled")!.config.uncertainBelow).toBe(0.42);
    expect(byFamily.get("compiled-tier2-only")!.config.uncertainBelow).toBe(0.42);
    expect(byFamily.get("baseline-b")!.config.uncertainBelow).toBeUndefined();
    expect(byFamily.get("baseline-b-tier0")!.config.uncertainBelow).toBeUndefined();
    // A non-default threshold, deliberately: a plan exercised only at
    // `UNCERTAIN_BELOW` cannot tell "carries the caller's value" from
    // "hardcodes the default".
    expect(0.42).not.toBe(UNCERTAIN_BELOW);
  });

  it("sizes a B arm at one call per message and the compiled arms at one per segment", () => {
    const plan = planBakeoff({
      options: options({ families: ["compiled", "baseline-b"] }),
      ir: PAIRED_IR,
      items: ITEMS,
      policyText: POLICY_TEXT,
    });
    const byFamily = new Map(plan.arms.map((a) => [a.family, a]));
    // One judged unit plus the single repair retry.
    expect(byFamily.get("baseline-b")!.maxCallsPerItem).toBe(2);
    // The compiled arm's worst message selects three segments on this corpus.
    expect(byFamily.get("compiled")!.maxCallsPerItem).toBe(6);
  });
});
