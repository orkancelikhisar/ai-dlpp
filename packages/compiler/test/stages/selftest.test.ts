import { describe, expect, it } from "vitest";
import type { ZodType } from "zod";
import { loadPolicyIr, type EntityType } from "@sih/core";
import { MAX_FP_RATE, runSelfTest, userPromptFor } from "../../src/stages/selftest.js";
import type { LlmClient, LlmRequest } from "../../src/llm/client.js";
import { FixtureLlmClient, requestHash } from "../../src/llm/fixture.js";
import { loadTestFixtures } from "../fixtures/index.js";
import { minimalCompiledIr } from "../fixtures/minimal-compiled-ir.js";

describe("runSelfTest", () => {
  const ir = loadPolicyIr(JSON.stringify(minimalCompiledIr()));

  it("scores each tier-0 entityType against the real runtime", async () => {
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    const pan = report.entities.find((e) => e.entityType === "in-pan")!;
    expect(pan.positives).toBeGreaterThan(0);
    expect(pan.recall).toBeGreaterThan(0.8);
  });

  it("flags an entityType whose rules catch nothing", async () => {
    // The fixture supplies positives for a deliberately unmatched entity.
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    expect(report.warnings.some((w) => /below threshold/i.test(w))).toBe(true);
  });

  it("counts false positives from hard negatives", async () => {
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    const pan = report.entities.find((e) => e.entityType === "in-pan")!;
    expect(pan.falsePositives).toBeGreaterThanOrEqual(0);
    expect(pan.negatives).toBeGreaterThan(0);
  });

  it("skips tier-1 and tier-2 entityTypes with an explicit note", async () => {
    // Only tier 0 is executable today; scoring tier 1/2 here would report zero
    // recall for entities whose engines do not exist until Plans 4-5.
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    const client = report.entities.find((e) => e.entityType === "client-name");
    expect(client?.skipped).toBe(true);
    expect(client?.skipReason).toMatch(/tier 1/i);
  });

  it("tags every generated case so Plan 7 can exclude them from the eval corpus", async () => {
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);
    expect(report.corpusTag).toMatch(/selftest/i);
  });
});

/**
 * Appended beyond the plan's five, each verified by mutation to survive them.
 * The plan's set proves the stage runs and scores; these pin the parts a
 * downstream task or a human auditor depends on and nothing else watches.
 */
describe("runSelfTest (properties the plan's five do not pin)", () => {
  const ir = loadPolicyIr(JSON.stringify(minimalCompiledIr()));
  const run = () => runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);

  it("returns every executed case, tagged, so emit can commit the corpus", async () => {
    // MUTATION: `cases: []`, or dropping `corpusTag` from the case objects,
    // passes all five plan tests — the corpusTag test reads the report's tag,
    // not the cases'. Plan 7's contamination filter reads the cases'.
    const report = await run();
    expect(report.cases.length).toBeGreaterThan(0);
    expect(report.cases.every((c) => c.corpusTag === report.corpusTag)).toBe(true);
    expect(report.cases.some((c) => c.kind === "positive")).toBe(true);
    expect(report.cases.some((c) => c.kind === "negative")).toBe(true);
    // The count has to reconcile with what was scored, or the corpus is not the
    // corpus the numbers came from.
    const scored = report.entities.filter((e) => !e.skipped);
    const executed = scored.reduce((n, e) => n + e.positives + e.negatives, 0);
    expect(report.cases).toHaveLength(executed);
  });

  it("records what the runtime did with each case, not just the label", async () => {
    // Without `detected` the committed corpus is a list of strings and the
    // coverage claim beside it cannot be re-derived from it.
    const report = await run();
    const pan = report.entities.find((e) => e.entityType === "in-pan")!;
    const panCases = report.cases.filter((c) => c.entityType === "in-pan");
    expect(panCases.filter((c) => c.kind === "positive" && c.detected)).toHaveLength(pan.caught);
    expect(panCases.filter((c) => c.kind === "negative" && c.detected)).toHaveLength(
      pan.falsePositives,
    );
  });

  it("warns when an entityType over-fires on its own hard negatives", async () => {
    // MUTATION: deleting the MAX_FP_RATE branch passes all five plan tests —
    // `falsePositives >= 0` is true of every possible implementation. The
    // entropy rule genuinely over-fires here (long snake_case and kebab-case
    // identifiers clear 4.0 bits/char), which is exactly the finding this stage
    // exists to surface rather than a fixture tuned to produce one.
    const report = await run();
    const secret = report.entities.find((e) => e.entityType === "generic-secret")!;
    expect(secret.fpRate).toBeGreaterThan(MAX_FP_RATE);
    const warning = report.warnings.find((w) => /false-positive rate/i.test(w))!;
    expect(warning).toContain("generic-secret");
    expect(warning).toContain(String(secret.falsePositives));
  });

  it("never echoes a generated case in a warning", async () => {
    // A generated positive is a sensitive-SHAPED string. The repo convention
    // that diagnostics never carry one does not get an exemption because this
    // particular string was invented by a model rather than lifted from a policy.
    const report = await run();
    for (const warning of report.warnings) {
      for (const testCase of report.cases) {
        expect(warning).not.toContain(testCase.text);
      }
    }
  });

  it("leaves recall undefined rather than 0 for an entity it did not measure", async () => {
    // 0 means "the rules caught nothing", undefined means "nothing was run".
    // Collapsing the two is the false alarm the skip rule exists to prevent.
    const report = await run();
    const client = report.entities.find((e) => e.entityType === "client-name")!;
    expect(client.recall).toBeUndefined();
    expect(client.fpRate).toBeUndefined();
  });

  it("skips tier 2 as well as tier 1, and spends no model call doing it", async () => {
    // MUTATION: `skipReasonFor` returning undefined for tier 2 only passes all
    // five plan tests. It would also cost a fixture per shadow entityType and
    // report 0% recall for every semantic predicate on every compile.
    const shadowOnly = loadPolicyIr(
      JSON.stringify({
        ...minimalCompiledIr(),
        entityTypes: [
          {
            id: "pred:discusses-unreleased-financials",
            tier: 2,
            nlDefinition: "The message discusses unreleased financial results",
            examples: [],
            counterExamples: [],
            severity: "high",
            neverPseudonymize: true,
          },
        ],
        rules: [],
        semanticPredicates: [
          {
            id: "discusses-unreleased-financials",
            nlPredicate: "The message discusses unreleased financial results",
            scope: "message",
          },
        ],
        actions: { default: { "pred:discusses-unreleased-financials": "redact" } },
        provenance: {},
      }),
    );
    // A client with no fixtures at all: reaching the model would throw, so this
    // asserts the absence of the call rather than trusting the report.
    const report = await runSelfTest(new FixtureLlmClient(new Map()), shadowOnly);
    const shadow = report.entities[0]!;
    expect(shadow.skipped).toBe(true);
    expect(shadow.skipReason).toMatch(/tier 2/i);
    expect(report.cases).toEqual([]);
  });
});

/**
 * Prompt determinism. The fixture key hashes the whole `user` prompt, so a
 * prompt that varies by one byte between runs makes every committed fixture
 * miss — with a hash nobody can reproduce, since the request that produced it
 * is gone. In-process drift (a clock, a counter) is already fatal to the tests
 * above; what those cannot see is a prompt whose bytes depend on the ORDER the
 * IR's keys happened to arrive in, because within one process that order is
 * stable and the fixtures were recorded against it.
 */
/**
 * Coordinator review, after Task 7 shipped. The plan specifies "a positive
 * counts as caught if any finding carries that entityType", and the implementer
 * followed it — then hit a case where a real PAN in `pan_number=VALUE` glues
 * into one entropy run that `generic-secret` (critical) wins over `in-pan`
 * (high), and edited the corpus so the case would not arise.
 *
 * That is the wrong lever. `pan_number=VALUE` is how a PAN really appears in a
 * config line, the runtime CATCHES it and resolves `block` for it (cluster
 * resolution keeps the strictest action across the overlap, so the action is
 * right even though the label is not this entity's), and scoring it as a missed
 * positive makes the warning assert something false. Tuning corpus data until
 * the number looks better is the bias this project's carrier certification
 * exists to prevent elsewhere; it does not get a pass here.
 */
describe("positives caught under a different label", () => {
  const ir = loadPolicyIr(JSON.stringify(minimalCompiledIr()));
  const run = () => runSelfTest(new FixtureLlmClient(loadTestFixtures()), ir);

  it("counts a shadowed positive as prevented, and says which entity took it", async () => {
    // MUTATION: scoring a shadowed positive as a plain miss (the shipped
    // behaviour) leaves in-pan recall at 0.95 — still above 0.8, so every plan
    // test passes while the report claims a value that is caught and blocked
    // was missed.
    const report = await run();
    const pan = report.entities.find((e) => e.entityType === "in-pan")!;
    expect(pan.shadowed).toBe(1);
    expect(pan.recall).toBe(1);
    // MUTATION: folding shadowed into `caught` also gives recall 1 — and loses
    // the fact that in-pan's own rule never fires there.
    expect(pan.labelRecall).toBeLessThan(pan.recall!);

    const shadowedCase = report.cases.find((c) => c.entityType === "in-pan" && c.shadowedBy);
    expect(shadowedCase?.kind).toBe("positive");
    expect(shadowedCase?.detected).toBe(false);
    expect(shadowedCase?.shadowedBy).toBe("generic-secret");
  });

  it("still scores a positive nothing caught as a genuine miss", async () => {
    // The distinction has to cut both ways, or it is just a way of never
    // reporting a failure.
    const report = await run();
    const legacy = report.entities.find((e) => e.entityType === "legacy-employee-id")!;
    expect(legacy.recall).toBe(0);
    expect(legacy.shadowed).toBe(0);
    expect(report.cases.filter((c) => c.entityType === "legacy-employee-id")).not.toContainEqual(
      expect.objectContaining({ shadowedBy: expect.anything() }),
    );
  });

  it("does not word shadowing as a threshold breach", async () => {
    // MUTATION: wording the shadow note with "below threshold" passes every
    // other test here and makes the plan's below-threshold assertion pass even
    // when no entity is actually weak — the real alarm stops being findable.
    const report = await run();
    const shadow = report.warnings.filter((w) => /shadowed/i.test(w));
    expect(shadow).toHaveLength(1);
    expect(shadow[0]).not.toMatch(/below threshold/i);
    expect(shadow[0]).toContain("in-pan");
  });
});

describe("self-test prompt determinism", () => {
  const entity: EntityType = {
    id: "in-pan",
    tier: 0,
    nlDefinition: "Indian PAN card number",
    examples: ["ABCPD1234E"],
    counterExamples: [],
    severity: "high",
    surrogateKind: "id-number",
  };

  it("does not depend on the order an entityType's keys were inserted in", () => {
    // MUTATION (verified): reading one field through `Object.keys(entity)[0]`
    // instead of by name is byte-identical for the canonical order — every
    // fixture still hits and all fourteen other tests pass — and fails only
    // here. That is the real shape of the bug: `JSON.stringify(entity)` and any
    // `Object.entries` walk emit string keys in INSERTION order, so the same
    // entity parsed from a differently-ordered JSON document produces different
    // prompt bytes, a different hash, and a fixture miss on the machine that
    // read the IR back from disk rather than building it in memory. Nothing
    // else in this file can see that, because within one process the order is
    // whatever the fixtures were recorded against.
    const reordered: EntityType = {
      surrogateKind: "id-number",
      severity: "high",
      counterExamples: [],
      examples: ["ABCPD1234E"],
      nlDefinition: "Indian PAN card number",
      tier: 0,
      id: "in-pan",
    };
    expect(userPromptFor(reordered)).toBe(userPromptFor(entity));
  });

  it("distinguishes two entityTypes that differ only in id", () => {
    // The converse guard: a prompt that dropped a field would silently make two
    // classes share one fixture, and each would replay the other's cases.
    expect(userPromptFor({ ...entity, id: "other-id" })).not.toBe(userPromptFor(entity));
    expect(userPromptFor({ ...entity, nlDefinition: "Something else" })).not.toBe(
      userPromptFor(entity),
    );
    expect(userPromptFor({ ...entity, examples: ["QRSPT5678X"] })).not.toBe(userPromptFor(entity));
  });

  it("issues byte-identical requests across two independent runs", async () => {
    class RecordingClient implements LlmClient {
      readonly hashes: string[] = [];
      constructor(private readonly inner: LlmClient) {}
      async complete<T>(request: LlmRequest, schema: ZodType<T>): Promise<T> {
        this.hashes.push(requestHash(request));
        return this.inner.complete(request, schema);
      }
    }
    const runOnce = async (): Promise<string[]> => {
      const client = new RecordingClient(new FixtureLlmClient(loadTestFixtures()));
      // A fresh parse each time: `minimalCompiledIr` hands back a fresh clone,
      // so the two runs share no object identity anywhere.
      await runSelfTest(client, loadPolicyIr(JSON.stringify(minimalCompiledIr())));
      return client.hashes;
    };
    const [first, second] = [await runOnce(), await runOnce()];
    expect(second).toEqual(first);
    // One call per tier-0 entityType, in IR order — not per case, and none for
    // the tier-1 entity sitting between them.
    expect(first).toHaveLength(4);
    expect(new Set(first).size).toBe(4);
  });
});

describe("runSelfTest with a model that returns no positives", () => {
  /** One tier-0 entityType, no rules, and a fixture whose `positives` is empty. */
  const barrenIr = loadPolicyIr(
    JSON.stringify({
      ...minimalCompiledIr(),
      entityTypes: [
        {
          id: "unsampled-entity",
          tier: 0,
          nlDefinition: "A class the generator declined to produce positives for",
          examples: [],
          counterExamples: [],
          severity: "low",
        },
      ],
      rules: [],
      actions: { default: { "unsampled-entity": "redact" } },
      provenance: {},
    }),
  );

  it("reports the entity as unmeasured instead of dividing by zero", async () => {
    // MUTATION: `recall = caught / positives` with no zero guard yields NaN.
    // `NaN < RECALL_THRESHOLD` is false, so no warning fires, and NaN serializes
    // to `null` in the emitted report — an entity nothing was ever run against,
    // presented as one that passed. Every plan test still passes.
    const report = await runSelfTest(new FixtureLlmClient(loadTestFixtures()), barrenIr);
    const entity = report.entities[0]!;
    expect(entity.skipped).toBe(false);
    expect(entity.positives).toBe(0);
    expect(entity.recall).toBeUndefined();
    expect(report.warnings.some((w) => /unmeasured/i.test(w))).toBe(true);
    expect(report.warnings.some((w) => /below threshold/i.test(w))).toBe(false);
  });
});
