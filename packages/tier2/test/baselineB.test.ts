import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import {
  DeadlineExpired,
  WebLlmJudge,
  createBaselineB,
  createBaselineBPlusTier0,
  resolveTier2Config,
  BASELINE_B_SCHEMA,
  type CompleteOptions,
  type Tier2Completion,
  type Tier2Engine,
} from "../src/index.js";
import { runTier0, segmentText } from "@sih/core";
import type {
  Action,
  DegradedNotice,
  DetectInput,
  Detector,
  EntityType,
  PolicyIr,
  TierConfig,
} from "@sih/core";
import { fakeEngine, predicateIr } from "./helpers.js";

/**
 * Approach B -- the arm that has to be able to WIN.
 *
 * Every test below is either a symmetry check against the compiled arm or a
 * refusal B must perform for the comparison to mean anything. The failure this
 * file exists to prevent is not "B is broken": it is "B lost, and the reason is
 * in the harness rather than in the method".
 */

const POLICY = readFileSync(
  join(import.meta.dirname, "..", "..", "..", "policies", "p-corp.md"),
  "utf8",
);

const MSG = "Please review the Northwind Traders renewal before Friday.";

/** The default arm's config: what a bake-off arm is actually built with. */
const CONFIG = resolveTier2Config({});

/**
 * An IR with tier-0 rules, an authored entityType, and one shadow entityType.
 *
 * Written out here rather than reusing `predicateIr` because B needs something
 * `predicateIr` does not have -- real `rules`, so the B+tier-0 arm has a tier 0
 * that finds anything -- and because an independent fixture is what lets a test
 * disagree with the module under test.
 *
 * The `examples` and `counterExamples` are SENTINELS that appear nowhere else,
 * for the same reason `predicateIr`'s are: "B's prompt carries the vocabulary
 * and not the compiled definitions" is a vacuous assertion against empty lists.
 * One corpus in this project already shipped its only gold value as the IR's
 * own `examples` entry, and B is handed the IR.
 */
function baselineIr(): PolicyIr {
  const entityTypes: EntityType[] = [
    {
      id: "in-pan",
      tier: 0,
      nlDefinition: "SENTINEL-PAN-DEFINITION",
      examples: ["SENTINEL-PAN-EXAMPLE"],
      counterExamples: ["SENTINEL-PAN-COUNTEREXAMPLE"],
      severity: "high",
      surrogateKind: "id-number",
    },
    {
      id: "client-name",
      tier: 1,
      nlDefinition: "SENTINEL-CLIENT-DEFINITION",
      examples: ["SENTINEL-CLIENT-EXAMPLE"],
      counterExamples: ["SENTINEL-CLIENT-COUNTEREXAMPLE"],
      severity: "medium",
      surrogateKind: "org-name",
    },
    {
      id: "pred:unreleased-financials",
      tier: 2,
      nlDefinition: "discusses financial results that have not been published",
      examples: [],
      counterExamples: [],
      severity: "critical",
      neverPseudonymize: true,
    },
  ];
  const defaultActions: Record<string, Action> = {
    "in-pan": "block",
    "client-name": "pseudonymize",
    "pred:unreleased-financials": "redact",
  };
  return {
    irVersion: "1",
    policyHash: "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    entityTypes,
    rules: [
      {
        id: "pan-rule",
        entityType: "in-pan",
        regex: "\\b[A-Z]{5}[0-9]{4}[A-Z]\\b",
        validator: "pan-structure",
        contextBoost: ["PAN", "tax"],
      },
    ],
    semanticPredicates: [
      {
        id: "unreleased-financials",
        nlPredicate: "discusses financial results that have not been published",
        scope: "segment",
      },
    ],
    actions: {
      default: defaultActions,
      providerOverrides: { deepseek: { "client-name": "redact" } },
    },
    failMode: "closed",
    latencyBudgetMs: 30_000,
    provenance: {},
  };
}

/** The TierConfig each arm requires; spelled out so a test can bend one field. */
const B_ONLY: TierConfig = { tier0: false, tier1: false, tier2: true };
const B_PLUS_T0: TierConfig = { tier0: true, tier1: false, tier2: true };

/**
 * A fake engine that agrees with `CONFIG` about which model it loaded.
 *
 * `createBaselineB` refuses a disagreement -- the fit check would otherwise
 * read a window the engine was not loaded at -- so every arm below goes through
 * here rather than repeating the id. `fakeEngine`'s own default id is
 * deliberately NOT a real model, which is what makes that guard reachable.
 */
function armEngine(options: Parameters<typeof fakeEngine>[0] = {}) {
  return fakeEngine({ requestedModelId: CONFIG.modelId, ...options });
}

/**
 * The same engine, answering `ms` later.
 *
 * A wrapper rather than a `fakeEngine` option: what has to be slow is the awaited
 * `complete`, and wrapping the shipped double keeps every other behaviour --
 * the message-order preconditions, the script, the recorded calls -- exactly the
 * one the rest of the file exercises.
 */
function slowEngine(base: Tier2Engine, ms: number): Tier2Engine {
  return {
    requestedModelId: base.requestedModelId,
    complete: async (messages, opts) => {
      await new Promise((resolve) => setTimeout(resolve, ms));
      return base.complete(messages, opts);
    },
    unload: () => base.unload(),
  };
}

/**
 * A stock `usage` block, shaped as 0.2.84 declares it.
 *
 * The two numbers the assertions read are the plan's own measurements for an
 * Approach-B call -- a 1,443-token prompt and a 2.8 s time to first token --
 * so a reader of the test sees the arm's real cost rather than round numbers.
 */
const USAGE = {
  prompt_tokens: 1443,
  completion_tokens: 12,
  total_tokens: 1455,
  extra: {
    e2e_latency_s: 7.5,
    prefill_tokens_per_s: 452,
    decode_tokens_per_s: 40,
    time_to_first_token_s: 2.8,
    time_per_output_token_s: 0.025,
  },
};

/**
 * One scripted Approach-B answer, in B's own wire shape.
 *
 * `mention` defaults to `quote`, which is the model's "no smaller span will do"
 * answer, so a case that does not name one is asserting on the WHOLE-CLAUSE
 * path and its offsets are the clause's. That is deliberate for the cases where
 * the span is incidental -- budgets, aborts, repairs, entity naming -- and it
 * means those cases cannot show that `mention` is read at all. The "narrows the
 * action span" block is where that is asserted.
 */
function answer(
  ...findings: Array<{
    entityType: string;
    quote: string;
    mention?: string;
    confidence: number;
  }>
): string {
  return JSON.stringify({
    findings: findings.map((f) => ({ ...f, mention: f.mention ?? f.quote })),
  });
}

function input(overrides: Partial<DetectInput> = {}): DetectInput {
  return {
    ir: baselineIr(),
    provider: "claude",
    text: MSG,
    config: B_ONLY,
    ...overrides,
  };
}

function reasons(degraded: readonly DegradedNotice[]): string[] {
  return degraded.map((d) => `${d.tier}:${d.reason}`);
}

/** The system message's content, which is where both arms keep their rules. */
function systemTurn(messages: readonly ChatCompletionMessageParam[]): string {
  const system = messages[0]!;
  expect(system.role).toBe("system");
  return typeof system.content === "string" ? system.content : JSON.stringify(system.content);
}

// ---------------------------------------------------------------------------
// The Detector contract
// ---------------------------------------------------------------------------

describe("createBaselineB: the Detector contract", () => {
  it("satisfies core's Detector type and returns a whole DetectionResult", async () => {
    // Same interface as the compiled pipeline, so the harness runs B as just
    // another arm rather than through a second code path. The annotation is
    // the assertion: if B's shape drifts from `Detector` this stops compiling.
    const detector: Detector = createBaselineB({
      engine: armEngine({
        raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(Array.isArray(r.findings)).toBe(true);
    expect(typeof r.timings.tier0Ms).toBe("number");
    expect(Array.isArray(r.degraded)).toBe(true);
  });

  it("attaches a provider-resolved action to every finding it returns", async () => {
    const build = (provider: string) =>
      createBaselineB({
        engine: armEngine({
          raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
        }),
        config: CONFIG,
        policyText: POLICY,
        budgetMs: 60_000,
      })(input({ provider }));

    // Two providers, because a single one cannot tell "resolves the action" from
    // "hardcodes the default": the IR's default for client-name is pseudonymize
    // and its deepseek override is redact.
    expect((await build("claude")).findings.map((f) => f.action)).toEqual(["pseudonymize"]);
    expect((await build("deepseek")).findings.map((f) => f.action)).toEqual(["redact"]);
  });

  it("reports tier2Ms for the model call and leaves tier0Ms at zero when tier 0 did not run", async () => {
    // `tier2Ms` is asserted against a call that really took time, because every
    // earlier assertion on it (`>= 0`, `typeof === "number"`) is satisfied by a
    // hardcoded 0 -- and that mutation survived the suite. apps/eval carries
    // `timings` verbatim onto every record, and latency is the whole of B's
    // structural disadvantage, so a confident zero in that column is the
    // records-state-fact defect in the one place it costs most.
    const SLOW_MS = 30;
    const detector = createBaselineB({
      engine: slowEngine(armEngine({ raw: answer() }), SLOW_MS),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const wallStarted = performance.now();
    const r = await detector(input());
    const wall = performance.now() - wallStarted;
    expect(r.timings.tier0Ms).toBe(0);
    // A timer fires late, never early. Half the delay is the floor a wrong
    // clock cannot reach; the wall clock is the ceiling a window opened too
    // early (at `messageStarted` rather than at the call) would exceed.
    expect(r.timings.tier2Ms!).toBeGreaterThan(SLOW_MS / 2);
    expect(r.timings.tier2Ms!).toBeLessThanOrEqual(wall);
  });
});

// ---------------------------------------------------------------------------
// What the record says about the arm
// ---------------------------------------------------------------------------

describe("createBaselineB: the TierConfig it will accept", () => {
  const build = () =>
    createBaselineB({
      engine: armEngine({ raw: answer() }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });

  it("refuses a config claiming tier 1 ran, since B has no span tagger", async () => {
    await expect(build()(input({ config: { tier0: false, tier1: true, tier2: true } }))).rejects.toThrow(
      /tier 1/i,
    );
  });

  it("refuses a config claiming tier 0 ran, because this arm does not run it", async () => {
    // The record-states-intent defect, in the one place it would be invisible:
    // a harness row reading `tier0: true` against an arm that never called
    // runTier0 reports deterministic coverage the arm never had.
    await expect(build()(input({ config: B_PLUS_T0 }))).rejects.toThrow(/tier 0/i);
  });

  it("refuses a config saying no model read the message", async () => {
    // `config.tier2` is what couples `tier2Stats` onto the harness record, and
    // every finding B emits carries `tier: 2`. A false here contradicts both.
    await expect(build()(input({ config: { tier0: false, tier1: false, tier2: false } }))).rejects.toThrow(
      /tier 2/i,
    );
  });

  it("files an absent notice per tier it did not run, and none for the model it did run", async () => {
    const r = await build()(input());
    expect(reasons(r.degraded)).toEqual(["0:absent", "1:absent"]);
  });
});

// ---------------------------------------------------------------------------
// The whole policy, or an honest refusal
// ---------------------------------------------------------------------------

describe("createBaselineB: the policy is never chunked", () => {
  it("sends the WHOLE policy document, heading for heading", async () => {
    // Chunking such that a clause is absent from the chunk that sees the
    // message makes a violation undetectable IN PRINCIPLE, which rigs the
    // comparison in the compiler's favour before a model is even loaded.
    const engine = armEngine({ raw: answer() });
    const detector = createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 });
    await detector(input());

    const headings = POLICY.match(/^##? .+$/gm) ?? [];
    expect(headings.length).toBeGreaterThan(5);
    const prompt = engine.promptOf(0);
    for (const heading of headings) expect(prompt).toContain(heading.trim());
  });

  it("refuses at construction when the policy alone cannot fit the window", async () => {
    // The honest failure. A silently truncated policy produces an arm that
    // loses for a reason nobody can see in the numbers. Refused when the arm is
    // BUILT rather than on its first message, so a bake-off does not start.
    expect(() =>
      createBaselineB({
        engine: armEngine({ raw: answer() }),
        config: CONFIG,
        policyText: "x".repeat(200_000),
        budgetMs: 60_000,
      }),
    ).toThrow(/does not fit/i);
  });

  it("refuses a message that pushes the prompt past the window", async () => {
    const detector = createBaselineB({
      engine: armEngine({ raw: answer() }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await expect(detector(input({ text: "x".repeat(200_000) }))).rejects.toThrow(/does not fit/i);
  });

  it("reads the window off the config it was given, not off the 8192 default", async () => {
    // A test exercising only the DEFAULT config cannot tell "reads the config"
    // from "hardcodes 8192" -- that exact mutation survived a 114-test suite in
    // this package once. So the same policy and the same message are run at two
    // windows and must disagree.
    //
    // 16384 is not a window any model here was measured at. It is here to make
    // the arithmetic read its input; nothing about it is a claim that a model
    // would load at it.
    const text = "x".repeat(20_000);
    const wide = createBaselineB({
      engine: armEngine({ raw: answer() }),
      config: resolveTier2Config({ contextWindowSize: 16_384 }),
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await expect(wide(input({ text }))).resolves.toBeDefined();

    const narrow = createBaselineB({
      engine: armEngine({ raw: answer() }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await expect(narrow(input({ text }))).rejects.toThrow(/does not fit/i);
  });

  it("reserves the answer's own tokens, so a bigger max_tokens shrinks what fits", async () => {
    // The window has to hold the prompt AND the completion: read out of the
    // shipped 0.2.84 bundle, the pipeline throws ContextWindowSizeExceededError
    // on `numPromptTokens + filledKVCacheLength > contextWindowSize`, and the
    // KV cache is what generation grows.
    const text = "x".repeat(9_000);
    const roomy = createBaselineB({
      engine: armEngine({ raw: answer() }),
      config: resolveTier2Config({ contextWindowSize: 8_192, maxTokens: 16 }),
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await expect(roomy(input({ text }))).resolves.toBeDefined();

    const greedy = createBaselineB({
      engine: armEngine({ raw: answer() }),
      config: resolveTier2Config({ contextWindowSize: 8_192, maxTokens: 4_000 }),
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await expect(greedy(input({ text }))).rejects.toThrow(/does not fit/i);
  });
});

// ---------------------------------------------------------------------------
// The prompt: same discipline as the compiled arm, no compiled rules
// ---------------------------------------------------------------------------

describe("createBaselineB: what the prompt carries", () => {
  it("asks for a quote under the same rules the compiled arm asks under", async () => {
    // The countermeasure for "prompt B worse than tier 2 is prompted", made
    // mechanical instead of left to a reviewer's eye. The two prompts are read
    // off REAL calls, not off exported constants, so a change to either arm's
    // wording has to be made in both.
    const judgeEngine = fakeEngine({ findings: [] });
    await new WebLlmJudge(judgeEngine, { budgetMs: 60_000 }).judge({
      text: MSG,
      segments: [{ start: 0, end: MSG.length, kind: "prose", text: MSG }],
      ir: predicateIr(),
      priorFindings: [],
      budgetMs: 60_000,
    });

    const bEngine = armEngine({ raw: answer() });
    await createBaselineB({
      engine: bEngine,
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    })(input());

    // The SYSTEM turn on each side, which is where both arms keep their rules.
    // Not the whole prompt: the judge's user turn lists its predicates as "- "
    // bullets too, and folding those in would compare a rule against a policy
    // clause.
    const judgeRules = systemTurn(judgeEngine.calls[0]!.messages);
    const bPrompt = systemTurn(bEngine.calls[0]!.messages);

    // Every rule bullet the judge states, restated to B with the two nouns that
    // MUST differ substituted: B reads a message rather than one passage of
    // one, and B names an entity class rather than a predicate.
    const shared = judgeRules
      .split("\n")
      .filter((line) => line.startsWith("- ") || line.startsWith("  "))
      .map((line) => line.replaceAll("passage", "message").replaceAll("predicateId", "entityType"));
    expect(shared.length).toBeGreaterThanOrEqual(6);
    for (const line of shared) {
      // The one bullet that cannot map is the empty-answer instruction: the
      // judge's names predicates, B's names the policy. It is asserted on its
      // own below rather than fudged into this loop.
      if (line.includes("satisfies any predicate")) continue;
      expect(bPrompt).toContain(line);
    }
    expect(bPrompt).toMatch(/If nothing in the message is restricted by the policy/);

    // THE MIRROR, and until this round there was none. The map above rewrites
    // the JUDGE's side, so a judge bullet that has ALREADY drifted to B's noun
    // is left alone by the substitution and then found in B's prompt verbatim
    // -- the check passed on exactly the change it exists to refuse. MEASURED
    // on the version before this one: renaming "passage" to "message" in all
    // four of the judge's remaining rule bullets left the whole tier2 suite
    // green, and so did renaming it in one. That is the compiled arm being told
    // to quote from a message it is never shown -- it is handed ONE PASSAGE and
    // its own task sentence says so -- with no test and no report row recording
    // the change of method.
    //
    // So the same comparison is run the other way: every rule bullet B states
    // must appear in the judge's with the substitution reversed.
    const mirrored = bPrompt
      .split("\n")
      .filter((line) => line.startsWith("- ") || line.startsWith("  "))
      .map((line) => line.replaceAll("message", "passage").replaceAll("entityType", "predicateId"));
    // Same count both ways, which is the third thing neither direction's
    // containment can see on its own: a rule bullet ADDED to one arm only.
    expect(mirrored.length).toBe(shared.length);
    for (const line of mirrored) {
      // B's half of the one bullet that cannot map, for the same reason.
      if (line.includes("restricted by the policy")) continue;
      expect(judgeRules).toContain(line);
    }
    expect(judgeRules).toMatch(/If nothing in the passage satisfies any predicate/);
  });

  it("lists the entity vocabulary and never the IR's definitions or examples", async () => {
    // The concession B is given is the entity TAXONOMY -- the ids, so a finding
    // can name something normalizeFindings accepts. It is deliberately not
    // given the compiled definitions, and never the examples: one corpus in
    // this project shipped its only gold value as the IR's own examples entry,
    // and a model handed the answer scores without doing the work.
    const engine = armEngine({ raw: answer() });
    await createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 })(input());
    const prompt = engine.promptOf(0);

    for (const id of ["in-pan", "client-name", "pred:unreleased-financials"]) {
      expect(prompt).toContain(id);
    }
    expect(prompt).not.toMatch(/SENTINEL-/);
  });

  it("sends the whole message in ONE call rather than a call per segment", async () => {
    // The structural difference between the arms, and the one that must not be
    // quietly undone: tier 2 is handed escalated segments and calls per
    // segment; B is handed the message. A B that segmented would be a second
    // tier-2 with a longer prompt.
    const text = ["Northwind Traders renew in March.", "```", "const k = 1;", "```", "key: value"].join(
      "\n",
    );
    const engine = armEngine({ raw: answer() });
    await createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 })(
      input({ text }),
    );
    expect(engine.calls).toHaveLength(1);
    const prompt = engine.promptOf(0);
    expect(prompt).toContain("Northwind Traders renew in March.");
    expect(prompt).toContain("const k = 1;");
    expect(prompt).toContain("key: value");
  });

  it("asks for the wire field its grammar masks for, not the judge's", async () => {
    // The one line of B's prompt that MUST differ from the judge's is excluded
    // from the cross-arm comparison above by construction: that test filters to
    // lines starting "- " or two spaces, and the JSON-shape line starts with
    // "{". Renaming `entityType` to `predicateId` there survived the suite.
    //
    // Checked against the SCHEMA, because the schema is what the logit mask is
    // compiled from. A prompt asking for one key under a grammar that forces
    // another puts the instruction and the mask in disagreement on every call;
    // constrained decoding still yields the right key, so it does not fail
    // loudly -- it just degrades B's answers for a harness reason. That is the
    // same defect class this task already found one layer down, where
    // `complete` dropped `opts.responseSchemaJson`.
    //
    // The three names are written out HERE rather than read off the schema
    // alone, so this file is the independent oracle: renaming the field in both
    // the schema and the prompt at once has to fail here rather than agree with
    // itself.
    const engine = armEngine({ raw: answer() });
    await createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 })(input());
    const shape = systemTurn(engine.calls[0]!.messages)
      .split("\n")
      .find((line) => line.startsWith("{"));
    expect(shape).toBeDefined();
    expect(BASELINE_B_SCHEMA.properties.findings.items.required).toEqual([
      "entityType",
      "quote",
      "mention",
      "confidence",
    ]);
    for (const field of ["entityType", "quote", "mention", "confidence"]) {
      expect(shape).toContain(`"${field}"`);
    }
    // The judge's name for the same slot, which is what a copy-paste from
    // `judge.ts` would leave here.
    expect(shape).not.toContain("predicateId");
  });

  it("uses the pinned call recipe, with the schema the only thing changed", async () => {
    const engine = armEngine({ raw: answer() });
    await createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 })(input());
    const opts = engine.calls[0]!.opts as CompleteOptions;
    expect(opts.responseSchemaJson).toBe(JSON.stringify(BASELINE_B_SCHEMA));
    expect(opts.budgetMs).toBe(60_000);
    expect(opts.signal).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Spans: the same ladder, over the whole message
// ---------------------------------------------------------------------------

describe("createBaselineB: span recovery", () => {
  it("resolves a verbatim quote at rung 1 and reports offsets core would accept", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(r.findings).toHaveLength(1);
    const f = r.findings[0]!;
    // Against indexOf, not against the implementation's own arithmetic: `text`
    // is the slice by construction, so comparing them proves nothing.
    expect(f.start).toBe(MSG.indexOf("Northwind Traders renewal"));
    expect(f.end).toBe(f.start + "Northwind Traders renewal".length);
    expect(MSG.slice(f.start, f.end)).toBe(f.text);
    expect(detector.stats.rung1).toBe(1);
    expect(detector.stats.rung2).toBe(0);
  });

  it("falls to rung 2 when the model perturbs the tail, and says so", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({
          entityType: "client-name",
          quote: "Northwind Traders renewal beforX",
          confidence: 0.5,
        }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(r.findings).toHaveLength(1);
    expect(r.findings[0]!.text).toBe("Northwind Traders renewal befor");
    expect(detector.stats.rung1).toBe(0);
    expect(detector.stats.rung2).toBe(1);
  });

  it("refuses a quote lifted from the POLICY rather than from the message", async () => {
    // B is the only arm shown a second document, so it is the only arm that can
    // quote one. A ladder that searched anything but the message would let the
    // policy's own words resolve to a span of the user's text.
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({
          entityType: "client-name",
          quote: "API keys, access tokens, and passwords MUST NOT",
          confidence: 0.9,
        }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(r.findings).toEqual([]);
    expect(detector.stats.unresolvedQuotes).toBe(1);
  });

  it("resolves a quote that straddles two segments, which a per-segment arm cannot", async () => {
    // Not a bug being pinned -- an ADVANTAGE B has, recorded so a reader of the
    // bake-off knows which direction it points. Tier 2 searches the segment it
    // was handed; B searches the message it was shown.
    const text = "alpha: Northwind\nThe Northwind Traders renewal is signed.";
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal is signed", confidence: 0.7 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input({ text }));
    expect(r.findings).toHaveLength(1);
    expect(text.slice(r.findings[0]!.start, r.findings[0]!.end)).toBe(
      "Northwind Traders renewal is signed",
    );
  });

  it("drops a duplicate span once rather than counting one piece of evidence twice", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer(
          { entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 },
          { entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.6 },
        ),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(r.findings).toHaveLength(1);
    expect(detector.stats.duplicatesDropped).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Naming what it found without a compiler
// ---------------------------------------------------------------------------

describe("createBaselineB: entity naming", () => {
  it("emits an entityType the IR declares, so the merge can resolve an action", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({
          entityType: "pred:unreleased-financials",
          quote: "Northwind Traders renewal",
          confidence: 0.8,
        }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    // Non-vacuously: an empty findings array would satisfy a bare for-loop.
    expect(r.findings).toHaveLength(1);
    const declared = baselineIr().entityTypes.map((e) => e.id);
    for (const f of r.findings) expect(declared).toContain(f.entityType);
  });

  it("carries the IR's severity and tier 2, whatever the arm itself put there", async () => {
    // HONEST LABEL, from mutation testing: this pins a PIPELINE guarantee, not
    // one this module provides alone. Replacing `severity: entity.severity` in
    // `collect` with a literal "low" leaves the whole suite green, because
    // `resolveFindings` re-derives severity from `ir.entityTypes` after B has
    // spoken -- exactly as it does for the compiled arm. The assignment in
    // `collect` is belt and braces and no test can distinguish it; what IS
    // load-bearing, and is what this asserts, is that a finding leaving this
    // arm carries the policy's severity rather than a model's opinion.
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({
          entityType: "pred:unreleased-financials",
          quote: "Northwind Traders renewal",
          confidence: 0.8,
        }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(r.findings[0]!.severity).toBe("critical");
    expect(r.findings[0]!.tier).toBe(2);
  });

  it("records the model that ANSWERED, never the id that was requested", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        requestedModelId: "Qwen3.5-2B-q4f16_1-MLC",
        model: "some-other-model-actually-served-this",
        raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(r.findings[0]!.source).toBe("some-other-model-actually-served-this");
  });

  it("drops a label the IR does not declare and counts it, rather than losing the message", async () => {
    // Models invent labels. Passing one through makes core reject the whole
    // message, which costs the findings that were fine; the compiled arm drops
    // an invented predicateId for the same reason, so both arms lose one
    // finding rather than one message.
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer(
          { entityType: "confidential", quote: "Northwind Traders renewal", confidence: 0.9 },
          { entityType: "client-name", quote: "review the Northwind", confidence: 0.7 },
        ),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input());
    expect(r.findings.map((f) => f.entityType)).toEqual(["client-name"]);
    expect(detector.stats.unknownEntityTypes).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// One repair, then fail closed
// ---------------------------------------------------------------------------

describe("createBaselineB: parse failures", () => {
  it("retries once and collects the repaired answer", async () => {
    const engine = armEngine({
      script: [
        { raw: "here are the findings:" },
        { raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }) },
      ],
    });
    const detector = createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 });
    const r = await detector(input());
    expect(engine.calls).toHaveLength(2);
    expect(r.findings).toHaveLength(1);
    expect(detector.stats.repairAttempts).toBe(1);
    expect(detector.stats.failedClosed).toBe(0);
  });

  it("tells the repair call what was wrong with the first answer", async () => {
    // The compiled arm has exactly this test (`judge.test.ts`); B did not, and
    // deleting `repairMessage(...)` from the retry -- so the second call
    // re-sends the byte-identical prompt -- left all 252 tier-2 tests green.
    // The two tests that touch this path only count calls and stats, and
    // `fakeEngine` answers from a script whatever the prompt says, so a repair
    // turn that was never sent is invisible to them.
    //
    // Not cosmetic: `Tier2Config.temperature` is pinned to 0 and Plan 5
    // measured 26/26 byte-identical completions at that setting, so re-sending
    // an unchanged prompt returns the same unparseable body by construction.
    // B's "one repair, then fail closed" would collapse to "fail closed", every
    // malformed answer a repair would have fixed would become a failedClosed
    // row, and B would lose the head-to-head for a harness reason.
    const engine = armEngine({ script: [{ raw: "here are the findings:" }, { raw: answer() }] });
    await createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 })(input());
    expect(engine.calls).toHaveLength(2);
    const repair = engine.promptOf(1);
    expect(repair).not.toBe(engine.promptOf(0));
    expect(repair).toMatch(/could not be parsed/i);
    // The original turns are kept and the repair is APPENDED, which is what
    // makes the second call a continuation rather than a fresh ask.
    expect(repair.startsWith(engine.promptOf(0))).toBe(true);
  });

  it("fails closed after one repair, and says so on the result", async () => {
    const engine = armEngine({ raw: "still not JSON" });
    const detector = createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 });
    const r = await detector(input());
    expect(engine.calls).toHaveLength(2);
    expect(r.findings).toEqual([]);
    expect(detector.stats.failedClosed).toBe(1);
    expect(reasons(r.degraded)).toContain("2:failed-closed");
  });

  it("does not retry a latched engine, because a retry is answered instantly and emptily", async () => {
    // Task 3 measured it: an interrupt sets an engine-wide flag the
    // non-streaming path never clears, and every later call returns "" with
    // finish_reason "abort" until something writes it back.
    const engine = armEngine({ raw: "", finishReason: "abort" });
    const detector = createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 });
    const r = await detector(input());
    expect(engine.calls).toHaveLength(1);
    expect(detector.stats.repairAttempts).toBe(0);
    expect(detector.stats.failedClosed).toBe(1);
    expect(detector.stats.abortedResponses).toBe(1);
    expect(reasons(r.degraded)).toContain("2:failed-closed");
  });

  it("counts a truncated response even when its body happens to parse", async () => {
    // B is the arm most exposed to the token budget: it makes ONE call for the
    // whole message where tier 2 makes one per segment, so it has a fraction of
    // the completion budget for the same input. An arm killed by max_tokens has
    // to be distinguishable from an incapable one.
    const detector = createBaselineB({
      engine: armEngine({ raw: answer(), finishReason: "length" }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await detector(input());
    expect(detector.stats.truncatedResponses).toBe(1);
    expect(detector.stats.messagesJudged).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Budgets: the per-call one and the message's
// ---------------------------------------------------------------------------

describe("createBaselineB: budgets", () => {
  it("refuses a per-call budget setTimeout would reinterpret", () => {
    // Infinity is the natural spelling of "no budget" and fires in under 1 ms.
    expect(() =>
      createBaselineB({
        engine: armEngine({ raw: answer() }),
        config: CONFIG,
        policyText: POLICY,
        budgetMs: Number.POSITIVE_INFINITY,
      }),
    ).toThrow(/budgetMs/);
  });

  it("files call-budget-exhausted when its own call budget expires", async () => {
    const detector = createBaselineB({
      engine: armEngine({ throws: new DeadlineExpired("budget", 500, true) }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 500,
    });
    const r = await detector(input());
    expect(r.findings).toEqual([]);
    expect(reasons(r.degraded)).toContain("2:call-budget-exhausted");
    expect(reasons(r.degraded)).not.toContain("2:budget-exhausted");
    expect(detector.stats.deadlineExpiries).toBe(1);
  });

  it("enforces ir.latencyBudgetMs itself, because nothing else will", async () => {
    // B replaces `detect`, so the message budget spec 5.3 writes down has no
    // other enforcer on this arm. An arm with no message budget would beat one
    // that has one for a reason that is not method.
    const waiting: Tier2Engine = {
      requestedModelId: CONFIG.modelId,
      async complete(_messages, opts: CompleteOptions): Promise<Tier2Completion> {
        await new Promise<void>((resolve) => {
          if (opts.signal?.aborted === true) resolve();
          else opts.signal?.addEventListener("abort", () => resolve(), { once: true });
        });
        throw new DeadlineExpired("aborted", opts.budgetMs, true);
      },
      async unload() {},
    };
    const detector = createBaselineB({
      engine: waiting,
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const ir = baselineIr();
    ir.latencyBudgetMs = 30;
    const r = await detector(input({ ir }));
    expect(reasons(r.degraded)).toContain("2:budget-exhausted");
    expect(detector.stats.messageBudgetExpiries).toBe(1);
    expect(detector.stats.callerAbortsMidGeneration).toBe(1);
  });

  it("hands the engine its fixed per-call budget and bounds the message with the signal", async () => {
    // NOT clamped to what is left of ir.latencyBudgetMs, and the reason is a
    // race rather than a preference: clamping arms the engine's own timer and
    // this arm's deadline on the same instant, so which notice gets filed --
    // call-budget-exhausted or budget-exhausted -- becomes a coin flip. `detect`
    // does not clamp the judge's budget either; the signal is the stop with one
    // owner, and the test above proves it fires.
    const engine = armEngine({ raw: answer() });
    const detector = createBaselineB({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 });
    const ir = baselineIr();
    ir.latencyBudgetMs = 2_000;
    await detector(input({ ir }));
    expect(engine.calls[0]!.opts.budgetMs).toBe(60_000);
    expect(engine.calls[0]!.opts.signal).toBeDefined();
  });

  it("refuses an engine loaded for a model the config does not name", async () => {
    // Two requested ids in one call is how a record ends up naming a run that
    // never happened -- and here it is worse than a label, because the fit
    // check reads the window off the config while the prompt goes to the
    // engine. `createWebLlmEngine` refuses the same pairing for the same reason.
    expect(() =>
      createBaselineB({
        engine: fakeEngine({ requestedModelId: "Phi-4-mini-instruct-q4f16_1-MLC" }),
        config: CONFIG,
        policyText: POLICY,
        budgetMs: 60_000,
      }),
    ).toThrow(/Phi-4-mini/);
  });
});

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

describe("createBaselineB: stats", () => {
  it("hands out a copy, so a reader cannot rewrite the numbers a row is built from", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await detector(input());
    const snapshot = detector.stats;
    (snapshot as { rung1: number }).rung1 = 99;
    (snapshot.calls as unknown as unknown[]).length = 0;
    expect(detector.stats.rung1).toBe(1);
    expect(detector.stats.calls).toHaveLength(1);
  });

  it("keeps one call row per answered call, and none for a call that was stopped", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        script: [
          { raw: "not JSON", usage: USAGE },
          { raw: answer(), usage: USAGE },
        ],
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await detector(input());
    expect(detector.stats.calls).toHaveLength(2);
    expect(detector.stats.calls[0]!.promptTokens).toBe(1443);
    expect(detector.stats.calls[0]!.ttftMs).toBe(2800);
  });

  it("accumulates across messages, which is what a recall denominator needs", async () => {
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await detector(input());
    await detector(input());
    expect(detector.stats.messagesJudged).toBe(2);
    expect(detector.stats.rung1).toBe(2);
  });

  it("de-duplicates within a message and not across them", async () => {
    // Offsets repeat across messages: [18, 43) is [18, 43) in every one of
    // them, so de-duplication state that outlived a call would silently delete
    // the next message's finding.
    const detector = createBaselineB({
      engine: armEngine({
        raw: answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    expect((await detector(input())).findings).toHaveLength(1);
    expect((await detector(input())).findings).toHaveLength(1);
    expect(detector.stats.duplicatesDropped).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// B + tier 0
// ---------------------------------------------------------------------------

describe("createBaselineBPlusTier0", () => {
  const PAN_MSG = "We have PAN AFTPD1298Q on file for the client.";

  it("merges tier-0 and B findings through core's own resolution, not a union", async () => {
    const detector = createBaselineBPlusTier0({
      engine: armEngine({
        raw: answer({ entityType: "in-pan", quote: "PAN AFTPD1298Q on file", confidence: 0.8 }),
      }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input({ text: PAN_MSG, config: B_PLUS_T0 }));
    const overlapping = r.findings.filter(
      (f) => f.start < PAN_MSG.indexOf("on file") && f.end > PAN_MSG.indexOf("AFTPD"),
    );
    expect(overlapping).toHaveLength(1);
    // Tier 0's own span survived, not B's wider one: the merge is severity then
    // confidence, and tier 0's boosted 0.95 beats B's 0.8.
    expect(overlapping[0]!.start).toBe(PAN_MSG.indexOf("AFTPD"));
  });

  it("returns the tier-0 findings even when the model answers with nothing", async () => {
    const detector = createBaselineBPlusTier0({
      engine: armEngine({ raw: answer() }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input({ text: PAN_MSG, config: B_PLUS_T0 }));
    expect(r.findings.map((f) => f.entityType)).toEqual(["in-pan"]);
    expect(r.findings[0]!.tier).toBe(0);
    expect(r.timings.tier0Ms).toBeGreaterThanOrEqual(0);
  });

  it("runs tier 0 over CORE's segmentation, which is what its entropy rules need", async () => {
    // Every other test in this block uses PAN_MSG, and `segmentText` returns
    // that as ONE prose segment -- so replacing `segmentText(text)` with a
    // hand-built whole-message prose segment produced a literally identical
    // segment list on the only input the suite ever exercised, and survived.
    //
    // What that would cost is not cosmetic: `tier0.ts` skips prose in its
    // entropy scanner (`if (seg.kind === "prose") continue;`), so an arm that
    // lost core's segmentation would run with every entropy rule disabled. On
    // this repo's own corpus that is exactly the AWS-key fence Task 9 measured
    // -- the one segment the compiled pipeline escalates on an entropy finding
    // at 0.7 -- so the arm would lose recall against the compiled pipeline for
    // a reason nowhere visible in the numbers.
    const FENCED = "Here is the deploy config.\n```\nAWS_ACCESS_KEY_ID=AKIAZZ7EXAMPLE4XQ2LN\n```\n";
    const base = baselineIr();
    const withEntropy: PolicyIr = {
      ...base,
      entityTypes: [
        ...base.entityTypes,
        {
          id: "generic-secret",
          tier: 0,
          nlDefinition: "SENTINEL-SECRET-DEFINITION",
          examples: [],
          counterExamples: [],
          severity: "critical",
          neverPseudonymize: true,
        },
      ],
      rules: [
        ...base.rules,
        { id: "entropy-rule", entityType: "generic-secret", entropyThreshold: 4.0, minLength: 20 },
      ],
      actions: {
        ...base.actions,
        default: { ...base.actions.default, "generic-secret": "redact" },
      },
    };

    // The premise, asserted rather than assumed, and it is what makes the
    // assertion below non-vacuous: this message is TWO segments, and the same
    // tier 0 over one whole-message PROSE segment finds nothing at all.
    expect(segmentText(FENCED).map((seg) => seg.kind)).toEqual(["prose", "code"]);
    expect(
      runTier0(withEntropy, FENCED, [{ start: 0, end: FENCED.length, kind: "prose", text: FENCED }]),
    ).toEqual([]);

    const detector = createBaselineBPlusTier0({
      engine: armEngine({ raw: answer() }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input({ ir: withEntropy, text: FENCED, config: B_PLUS_T0 }));
    expect(r.findings.map((f) => [f.entityType, f.tier])).toEqual([["generic-secret", 0]]);
    // And the tier-0 clock really ran. `tier0Ms` hardcoded to 0 survived the
    // suite because every assertion on it was `>= 0` or `toBe(0)` on the arm
    // that does not run tier 0. MEASURED over 400 runs of this arm, the
    // smallest value this field ever took was 0.00125 ms; it was never 0.
    expect(r.timings.tier0Ms).toBeGreaterThan(0);
  });

  it("tells the model WHAT tier 0 found as labels and counts, never as the text it found", async () => {
    // The prior line is the same one the compiled judge sends, so the two arms
    // hand the model the same kind of context. Putting the value itself in the
    // prompt would be an answer key the model can score against without doing
    // the work.
    const engine = armEngine({ raw: answer() });
    await createBaselineBPlusTier0({ engine, config: CONFIG, policyText: POLICY, budgetMs: 60_000 })(
      input({ text: PAN_MSG, config: B_PLUS_T0 }),
    );
    const prompt = engine.promptOf(0);
    expect(prompt).toContain("in-pan (x1)");
    // The message itself carries the value; the CONTEXT line must not repeat it.
    const contextLine = prompt.split("\n").find((l) => l.includes("in-pan (x1)"))!;
    expect(contextLine).not.toContain("AFTPD1298Q");
  });

  it("files no absent notice for tier 0, and one for tier 1", async () => {
    const detector = createBaselineBPlusTier0({
      engine: armEngine({ raw: answer() }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input({ text: PAN_MSG, config: B_PLUS_T0 }));
    expect(reasons(r.degraded)).toEqual(["1:absent"]);
  });

  it("refuses a config that says tier 0 was off, since this arm is defined by running it", async () => {
    const detector = createBaselineBPlusTier0({
      engine: armEngine({ raw: answer() }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    await expect(detector(input({ text: PAN_MSG, config: B_ONLY }))).rejects.toThrow(/tier 0/i);
  });

  it("keeps the tier-0 findings when the model fails closed", async () => {
    // The composition's whole point: B failing is not tier 0 failing, and an
    // arm that lost its deterministic findings to a model's bad JSON would be
    // measuring the wrong thing.
    const detector = createBaselineBPlusTier0({
      engine: armEngine({ raw: "not JSON at all" }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });
    const r = await detector(input({ text: PAN_MSG, config: B_PLUS_T0 }));
    expect(r.findings.map((f) => f.entityType)).toEqual(["in-pan"]);
    expect(reasons(r.degraded)).toContain("2:failed-closed");
  });
});

// ---------------------------------------------------------------------------
// The mention: the same four rules the compiled arm places spans under
//
// SYMMETRY IS THE POINT of this block. `locateFinding` in spans.ts is the one
// implementation both arms call, so these assertions and their twins in
// judge.test.ts are the pin that says the change was applied to both arms and
// applied the same way. An arm placing spans under a looser rule than the other
// would win exact-match columns for a harness reason.
// ---------------------------------------------------------------------------

describe("createBaselineB: narrows the action span", () => {
  const armFor = (raw: string) =>
    createBaselineB({
      engine: armEngine({ raw }),
      config: CONFIG,
      policyText: POLICY,
      budgetMs: 60_000,
    });

  it("carries the MENTION's offsets, not the clause's", async () => {
    const detector = armFor(
      answer({
        entityType: "client-name",
        quote: "review the Northwind Traders renewal",
        mention: "Northwind Traders",
        confidence: 0.8,
      }),
    );
    const r = await detector(input());
    expect(r.findings).toHaveLength(1);
    const f = r.findings[0]!;
    expect(f.start).toBe(MSG.indexOf("Northwind Traders"));
    expect(f.end).toBe(f.start + "Northwind Traders".length);
    expect(f.text).toBe("Northwind Traders");
    // The clause located it and is not what the finding carries.
    expect(MSG.indexOf("review the Northwind Traders renewal")).toBeLessThan(f.start);
    expect(detector.stats.rung1).toBe(1);
    expect(detector.stats.wholeClauseMentions).toBe(0);
  });

  it("counts a repeated quote as a whole-clause answer and still emits it", async () => {
    const detector = armFor(
      answer({ entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 }),
    );
    const r = await detector(input());
    expect(r.findings).toHaveLength(1);
    expect(r.findings[0]!.text).toBe("Northwind Traders renewal");
    expect(detector.stats.wholeClauseMentions).toBe(1);
  });

  it("REFUSES a mention outside its own quote, apart from an unplaced quote", async () => {
    const detector = armFor(
      answer({
        entityType: "client-name",
        quote: "review the Northwind Traders renewal",
        mention: "Friday",
        confidence: 0.8,
      }),
    );
    const r = await detector(input());
    expect(r.findings).toEqual([]);
    expect(detector.stats.unresolvedMentions).toBe(1);
    expect(detector.stats.unresolvedQuotes).toBe(0);
    expect(MSG).toContain("Friday");
  });

  it("REFUSES a mention the clause states twice", async () => {
    const text = "Please note: Acme sued Acme last year, per counsel.";
    const detector = armFor(
      answer({
        entityType: "client-name",
        quote: "Acme sued Acme last year",
        mention: "Acme",
        confidence: 0.8,
      }),
    );
    const r = await detector(input({ text }));
    expect(r.findings).toEqual([]);
    expect(detector.stats.unresolvedMentions).toBe(1);
  });

  it("files an unplaceable CLAUSE under unresolvedQuotes and never looks at the mention", async () => {
    const detector = armFor(
      answer({
        entityType: "client-name",
        quote: "a clause that is nowhere in this message",
        mention: "Northwind Traders",
        confidence: 0.8,
      }),
    );
    const r = await detector(input());
    expect(r.findings).toEqual([]);
    expect(detector.stats.unresolvedQuotes).toBe(1);
    expect(detector.stats.unresolvedMentions).toBe(0);
  });

  it("de-duplicates on the ACTION span, so two clauses over one mention are one finding", async () => {
    // The judge's twin, and a gap this block did not have: its only duplicate
    // fixture restates an IDENTICAL quote, where the evidence span and the
    // action span coincide and the two keying rules cannot disagree. FOUND BY
    // MUTATION -- re-keying B's duplicate check on `located.at.evidence` left
    // tier2, apps/eval and core green, while the identical swap in judge.ts was
    // killed by its own version of this test.
    //
    // Two DIFFERENT clauses, both containing the name. One piece of evidence
    // about one range of the message, so one finding; keyed on the clause it
    // would be two, `duplicatesDropped` would read 0, and B's `duplicate-rate`
    // and `resolvable-rate` -- both numeric gates that can kill an arm -- would
    // shift for one arm only.
    const detector = armFor(
      answer(
        {
          entityType: "client-name",
          quote: "review the Northwind Traders renewal",
          mention: "Northwind Traders",
          confidence: 0.8,
        },
        {
          entityType: "client-name",
          quote: "the Northwind Traders renewal before Friday",
          mention: "Northwind Traders",
          confidence: 0.5,
        },
      ),
    );
    const r = await detector(input());
    // The two clauses really are different, and both really do place -- so this
    // cannot pass because the second one was refused.
    expect(MSG).toContain("review the Northwind Traders renewal");
    expect(MSG).toContain("the Northwind Traders renewal before Friday");
    expect(r.findings).toHaveLength(1);
    expect(detector.stats.duplicatesDropped).toBe(1);
    expect(detector.stats.rung1).toBe(1);
    expect(detector.stats.unresolvedQuotes).toBe(0);
    expect(detector.stats.unresolvedMentions).toBe(0);
    expect(detector.stats.wholeClauseMentions).toBe(0);
  });

  it("counts a whole-clause answer ONCE when the model restates it", async () => {
    // The judge's twin, and the same mutation-found gap: the counter is over
    // findings EMITTED, so a model that restates one finding contributes one.
    // Without this, `wholeClauseMentions / (rung1 + rung2)` can exceed 1 on
    // exactly the looping model Plan 5's probe corpus caught.
    const f = { entityType: "client-name", quote: "Northwind Traders renewal", confidence: 0.8 };
    const detector = armFor(answer(f, f, { ...f, confidence: 0.4 }));
    const r = await detector(input());
    expect(r.findings).toHaveLength(1);
    expect(detector.stats.duplicatesDropped).toBe(2);
    expect(detector.stats.rung1).toBe(1);
    expect(detector.stats.wholeClauseMentions).toBe(1);
  });

  it("places the SAME span the compiled judge places for the same answer", async () => {
    // The cross-arm pin. Both arms are handed the same message, the same clause
    // and the same mention, and must return the same offsets -- which is what
    // "both arms changed, identically" means operationally. The two wire shapes
    // differ (`entityType` against `predicateId`) and nothing else does.
    const quote = "review the Northwind Traders renewal";
    const mention = "Northwind Traders";

    const b = armFor(answer({ entityType: "client-name", quote, mention, confidence: 0.8 }));
    const bFindings = (await b(input())).findings;

    const judge = new WebLlmJudge(
      fakeEngine({
        findings: [{ predicateId: "client-relationship", quote, mention, confidence: 0.8 }],
      }),
      { budgetMs: 60_000 },
    );
    const verdict = await judge.judge({
      text: MSG,
      segments: [{ kind: "prose", start: 0, end: MSG.length, text: MSG }],
      ir: predicateIr(),
      priorFindings: [],
      budgetMs: 60_000,
    });

    expect(bFindings).toHaveLength(1);
    expect(verdict.findings).toHaveLength(1);
    expect({ start: bFindings[0]!.start, end: bFindings[0]!.end }).toEqual({
      start: verdict.findings[0]!.start,
      end: verdict.findings[0]!.end,
    });
    expect(bFindings[0]!.text).toBe(verdict.findings[0]!.text);
    // And the counters agree, which is what a bake-off puts side by side.
    expect(b.stats.rung1).toBe(judge.stats.rung1);
    expect(b.stats.wholeClauseMentions).toBe(judge.stats.wholeClauseMentions);
  });
});
