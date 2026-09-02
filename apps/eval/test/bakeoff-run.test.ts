import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Page } from "@playwright/test";
import { afterEach, describe, expect, it } from "vitest";
import { RunRecordSchema, type RunRecord } from "../src/driver/record.js";
import {
  resolveIrPath,
  runBakeoff,
  type ArmGateReport,
  type BakeoffOptions,
} from "../src/driver/bakeoff.js";

/**
 * `runBakeoff` itself: the guards between a plan and a directory of files.
 *
 * ## Why this file exists
 *
 * No VITEST test imported `runBakeoff`. `bakeoff.spec.ts` does, and reaches two
 * of its refusals without a model -- the IR mismatch and the repeated runId --
 * but it is a Playwright spec, so it needs a browser and a dev server, and the
 * one test there that runs an arm skips without WebGPU. The other nine refusals
 * had no test anywhere: they are the checks that decide whether a run costing
 * hours and gigabytes may start, and the only way to execute them was to start
 * one.
 *
 * Everything the driver can decide without a page is in `bakeoff.test.ts`;
 * `runBakeoff` is the function that puts those together, and this file is that
 * seam.
 *
 * ## What a scripted page is and is not
 *
 * `page.evaluate(fn, arg)` is `fn(arg)` here, against a `window.__sih` this file
 * writes. That is enough because every guard under test is DRIVER logic reading
 * a value the page reported: an IR hash, an availability boolean, a load report,
 * a record. It proves nothing whatever about the browser -- not that WebGPU
 * works, not that a model loads, not that `detect` is core. Those live in
 * `tier2.spec.ts` and cost what they cost.
 *
 * The one thing this arrangement DOES have to be honest about: a real
 * `page.evaluate` serialises through the CDP protocol, so a fake that passes
 * live object references would let a test pass on shapes the real path cannot
 * carry. Every value this page returns is built from JSON primitives for that
 * reason, and the records the driver produces are validated with
 * `RunRecordSchema` at the end, which is the same check `runBakeoff` runs.
 */

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const CORPUS = join(REPO_ROOT, "corpora", "fixtures", "smoke.jsonl");
const IR_PATH = join(REPO_ROOT, "apps", "eval", "fixtures", "semantic-ir.json");
const HARNESS_DIR = join(import.meta.dirname, "..");
const MODEL = "Qwen3.5-2B-q4f16_1-MLC";
const CORPUS_ITEMS = readFileSync(CORPUS, "utf8").trim().split("\n").length;

/**
 * The COMPILED policy and its IR, which is what an Approach-B arm needs.
 *
 * `semantic-ir.json` cannot host one: its `policyHash` is a hand-written string
 * rather than a sha256, so `planBakeoff` refuses a B family against it before
 * any of the guards below are reached (the last test in the first block above
 * is that refusal). Only `policies/compiled/p-fin.ir.json` pairs with a
 * document, which is why every B test here runs against it -- and it is also
 * the pair `test/baseline.spec.ts` drives on a GPU, so these tests exercise the
 * same options that spec does without needing one.
 */
const B_IR_PATH = join(REPO_ROOT, "policies", "compiled", "p-fin.ir.json");
const B_POLICY_PATH = join(REPO_ROOT, "policies", "p-fin.md");
const B_IR_HASH = createHash("sha256").update(readFileSync(B_IR_PATH, "utf8"), "utf8").digest("hex");
const B_POLICY_SHA256 = createHash("sha256")
  .update(readFileSync(B_POLICY_PATH, "utf8"), "utf8")
  .digest("hex");
/**
 * Above `itemDeadlineBound` at p-fin's 5,000 ms message budget.
 *
 * p-fin carries the COMPILER's default rather than `semantic-ir.json`'s lifted
 * 120,000, so the bound here is 6,020 rather than 121,020 and the 400,000 the
 * tests above use would be accepted but absurd. `assertItemTimeoutMs` refuses
 * anything at or below the bound, so a wrong number here is a red test.
 */
const B_ITEM_TIMEOUT_MS = 20_000;

/**
 * The digest the page would report for the shipped IR.
 *
 * Computed with the same algorithm the driver uses, which is a tautology this
 * file accepts deliberately and narrowly: what is under test here is the
 * COMPARISON -- that a driver whose page reports a different digest refuses --
 * and the mismatch cases below supply a digest this function did not produce.
 * `run.spec.ts` reproduces a record's `irHash` the same way.
 */
const IR_HASH = createHash("sha256").update(readFileSync(IR_PATH, "utf8"), "utf8").digest("hex");

const dirs: string[] = [];
function outDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "sih-bakeoff-"));
  dirs.push(dir);
  return dir;
}
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  delete (globalThis as { window?: unknown }).window;
});

interface PageScript {
  harnessDir?: string;
  /** What `useIr` and `irHash` return. Defaults to the real digest of the file. */
  irHash?: string;
  /** A second value returned from the per-arm `useIr`, if it differs. */
  irHashAfterNavigation?: string;
  webgpu?: boolean;
  servedModelId?: string;
  contextWindowSize?: number;
  callBudgetMs?: number;
  /** Throw on every item, as an engine that stopped answering would. */
  detectThrows?: boolean;
  /** Fail on the Nth item of the Nth arm instead -- 1-based arm, 1-based item. */
  detectThrowsOnArm?: number;
  /** Emit a finding whose span does not lie inside the item text. */
  emitBadSpan?: boolean;
  /** What `policyDocHash` answers. Defaults to the real digest of the document. */
  policyDocHash?: string;
  /** What `loadBaseline` reports for the document it built on. */
  baselinePolicyDocSha256?: string;
  /** What `loadBaseline` reports as the IR it paired that document with. */
  baselineIrPolicyHash?: string;
  /** What `loadBaseline` reports as the engine it was built on. */
  baselineServedModelId?: string;
  /** What `loadBaseline` reports as its per-call budget. */
  baselineCallBudgetMs?: number;
  /**
   * Counter values merged into BOTH arms' `lastDetect`, so a projection can be
   * driven at something other than its default.
   *
   * A test that only ever sees 0 cannot tell "reads the field" from "writes a
   * literal 0" -- the defect the standing conventions name, and the one that
   * left `run.ts`'s Approach-B projection of `unresolvedMentions` and
   * `wholeClauseMentions` unpinned at any value.
   */
  counters?: Readonly<Record<string, number>>;
  /**
   * Files to create when the FIRST arm's engine loads.
   *
   * The only seam that lands between `runBakeoff`'s pre-flight `existsSync`
   * checks and its first write, which is the window the exclusive-create flags
   * exist for: a repeated runId is caught up front, and a file that appears an
   * hour later -- another process starting the same run -- is caught only by
   * "wx" on the arm file and "ax" on the gates file's first row.
   */
  createOnFirstLoad?: string[];
}

/** One `detect` answer, in the shape the real page's JSON would arrive in. */
function detection(text: string, script: PageScript): unknown {
  return {
    findings: script.emitBadSpan
      ? [
          {
            start: 0,
            end: text.length + 50,
            text: "not in this message",
            entityType: "generic-secret",
            severity: "critical",
            tier: 2,
            source: "pred:unannounced-deal",
            confidence: 0.9,
            action: "redact",
          },
        ]
      : [],
    timings: { tier0Ms: 0.2, tier2Ms: 12 },
    degraded: [],
  };
}

/**
 * A `Page` that runs the driver's own evaluate callbacks against a scripted API.
 *
 * `goto` publishes the API, exactly as a navigation does, so the driver's
 * `waitForFunction(() => window.__sih !== undefined)` is answered by the same
 * mechanism it is in a browser rather than by a stub that always says yes.
 */
function scriptedPage(script: PageScript = {}): {
  page: Page;
  unloads: () => number;
  /** Every `loadBaseline` argument, in order, so a test can assert what the driver ASKED for. */
  baselineLoads: () => { family: string; policy: string }[];
} {
  let firstUseIr = true;
  let unloads = 0;
  // 1-based index of the arm currently running, incremented by each `loadTier2`.
  let arms = 0;
  const baselineLoads: { family: string; policy: string }[] = [];
  // The budget the tier-2 load resolved, so `loadBaseline` can ECHO it the way
  // the real page does -- it reads B's per-call deadline off the judge's engine
  // rather than off the driver, and a stub inventing its own would make the
  // driver's cross-check a comparison of two constants.
  let loadedCallBudgetMs = 0;
  const api = {
    harnessDir: () => script.harnessDir ?? HARNESS_DIR,
    useIr: async (_name: string) => {
      const hash =
        firstUseIr || script.irHashAfterNavigation === undefined
          ? (script.irHash ?? IR_HASH)
          : script.irHashAfterNavigation;
      firstUseIr = false;
      return hash;
    },
    irHash: async () => script.irHash ?? IR_HASH,
    policyHash: () => "test-hash",
    // The page's own digest of the document it BUNDLED, which is the only thing
    // that can catch a dev server left running by another worktree:
    // `playwright.config.ts` sets `reuseExistingServer: !CI`, and no record
    // field carries a digest of the text B was shown.
    policyDocHash: async (_name: string) => script.policyDocHash ?? B_POLICY_SHA256,
    webgpuAvailable: async () => script.webgpu ?? true,
    loadTier2: async (options: { modelId: string; contextWindowSize: number; callBudgetMs: number }) => {
      arms += 1;
      if (arms === 1) for (const path of script.createOnFirstLoad ?? []) writeFileSync(path, "");
      loadedCallBudgetMs = script.callBudgetMs ?? options.callBudgetMs;
      return {
      config: {
        modelId: options.modelId,
        contextWindowSize: script.contextWindowSize ?? options.contextWindowSize,
        temperature: 0,
        maxTokens: 512,
      },
      callBudgetMs: loadedCallBudgetMs,
      servedModelId: script.servedModelId ?? options.modelId,
      // Three values that share no digits, and not the 1/1/0 they were. These
      // are the hardware-cost half of the research question, and `runBakeoff`
      // now carries them onto the gates row -- so a report that invented them,
      // or copied one field into another, has to be a red test rather than a
      // plausible-looking number in an output file nobody re-derives.
      loadMs: 4_321,
      warmupMs: 765,
      warmupFinishReason: "stop",
      warmupCompletionTokens: 5,
      storageUsageBytes: 9_876_543_210,
      storageQuotaBytes: 0,
      };
    },
    // The Approach-B door. Every field the driver checks is scriptable, because
    // each of the four checks is a DIFFERENT failure: a document the driver
    // never read, a document paired with the wrong IR, an engine reloaded
    // between the two loads, and two arms at different per-call deadlines.
    loadBaseline: async (options: { family: string; policy: string }) => {
      baselineLoads.push({ family: options.family, policy: options.policy });
      return {
      family: options.family,
      policy: options.policy,
      policyDocSha256: script.baselinePolicyDocSha256 ?? B_POLICY_SHA256,
      irPolicyHash: script.baselineIrPolicyHash ?? B_POLICY_SHA256,
      policyChars: 12_345,
      callBudgetMs: script.baselineCallBudgetMs ?? loadedCallBudgetMs,
      servedModelId: script.baselineServedModelId ?? MODEL,
      };
    },
    // Distinct from `tier2Status`, and that is the point of it being here: a
    // driver reading the judge's counters on a B arm would produce a row whose
    // `tier2Stats` is populated and whose `baselineStats` is absent, which
    // `RunRecordSchema` refuses -- but only on a machine that can load a model.
    baselineStatus: () => ({
      lastDetect: {
        rung1: 1,
        rung2: 0,
        unresolvedQuotes: 2,
        unresolvedMentions: 0,
        wholeClauseMentions: 0,
        unknownEntityTypes: 0,
        duplicatesDropped: 0,
        repairAttempts: 0,
        failedClosed: 0,
        truncatedResponses: 0,
        abortedResponses: 0,
        messagesJudged: 1,
        deadlineExpiries: 0,
        messageBudgetExpiries: 0,
        callerAbortsMidGeneration: 0,
        callerAbortsWhileQueued: 0,
        calls: [
          { finishReason: "stop", promptTokens: 2_100, completionTokens: 55, ttftMs: 900, decodeTokPerSec: 30 },
        ],
        // LAST, so a test can drive any one counter off its default. Spread
        // first it would be overwritten by the literals above and the knob
        // would silently do nothing.
        ...script.counters,
      },
    }),
    // COUNTED rather than a bare no-op: `runBakeoff` releases each arm's engine
    // when the arm finishes instead of leaving it to the next navigation, and an
    // arm that is never released is a 1-3 GB allocation the next arm's model
    // load competes with. A stub that silently accepted the call would let that
    // release be deleted with this suite green.
    unloadTier2: async () => {
      unloads += 1;
    },
    detect: async (request: { text: string }) => {
      if (script.detectThrows === true) throw new Error("the model did not answer");
      // Per ARM rather than per call, so a run can be made to die AFTER an arm
      // has finished and written its file -- which is the only state in which
      // the gates file's ordering is observable.
      if (script.detectThrowsOnArm !== undefined && arms === script.detectThrowsOnArm) {
        throw new Error("the model did not answer");
      }
      return detection(request.text, script);
    },
    tier2Status: () => ({
      lastDetect: {
        rung1: 1,
        rung2: 0,
        unresolvedQuotes: 0,
        unresolvedMentions: 0,
        wholeClauseMentions: 0,
        unknownPredicates: 0,
        duplicatesDropped: 0,
        repairAttempts: 0,
        failedClosed: 0,
        truncatedResponses: 0,
        abortedResponses: 0,
        segmentsJudged: 1,
        segmentsSkipped: 0,
        messageScopeCalls: 0,
        messageScopeJudged: 0,
        messageScopeFailedClosed: 0,
        deadlineExpiries: 0,
        callerAbortsMidGeneration: 0,
        callerAbortsWhileQueued: 0,
        calls: [
          { finishReason: "stop", promptTokens: 400, completionTokens: 40, ttftMs: 480, decodeTokPerSec: 45 },
        ],
        // LAST, for the reason `baselineStatus` states.
        ...script.counters,
      },
    }),
  };
  const publish = (): void => {
    (globalThis as { window?: unknown }).window = { __sih: api };
  };
  publish();
  const page = {
    on: () => undefined,
    off: () => undefined,
    goto: async () => {
      publish();
      return null;
    },
    waitForFunction: async (fn: () => unknown) => {
      if (fn() !== true) throw new Error("harness never became ready");
      return undefined;
    },
    evaluate: async (fn: (arg?: unknown) => unknown, arg?: unknown) => fn(arg),
  };
  return { page: page as unknown as Page, unloads: () => unloads, baselineLoads: () => baselineLoads };
}

function bakeoffOptions(dir: string, overrides: Partial<BakeoffOptions> = {}): BakeoffOptions {
  return {
    runId: "run1",
    outDir: dir,
    corpus: CORPUS,
    provider: "claude",
    models: [MODEL],
    itemTimeoutMs: 400_000,
    irName: "semantic",
    irPath: IR_PATH,
    ...overrides,
  };
}

describe("runBakeoff, against a scripted page", () => {
  it("writes one file per arm plus a gates file, and every record validates", async () => {
    const dir = outDir();
    const { page, unloads } = scriptedPage();
    const result = await runBakeoff(page, bakeoffOptions(dir, { families: ["compiled", "compiled-tier2-only"] }));

    // ONE release per arm, including the last. `runBakeoff` navigates between
    // arms rather than reloading, and navigation destroys the JS context
    // without promising the GPU allocation goes with it -- so an arm left
    // loaded is a 1-3 GB allocation the next arm's model load competes with.
    // Asserted here because the browser suite cannot see it: `unloadTier2`
    // resolving says nothing about what the GPU did.
    expect(unloads()).toBe(2);

    expect(result.written).toHaveLength(2);
    expect(result.reports).toHaveLength(2);
    for (const path of result.written) expect(existsSync(path)).toBe(true);
    expect(existsSync(result.gatesPath)).toBe(true);

    const rows = readFileSync(result.written[0]!, "utf8").trim().split("\n").map((line) => JSON.parse(line) as unknown);
    expect(rows).toHaveLength(CORPUS_ITEMS);
    for (const row of rows) expect(RunRecordSchema.safeParse(row).success).toBe(true);

    // The settings this driver chose reach the rows, and they are the ones it
    // asked the page for rather than defaults: `provider` and the tier-2 config
    // are both fields a scorer groups by.
    const first = rows[0] as RunRecord;
    expect(first.provider).toBe("claude");
    expect(first.arm).toBe(result.plan.arms[0]!.arm);
    expect(first.irHash).toBe(IR_HASH);
    expect(first.tier2Config?.callBudgetMs).toBe(result.plan.arms[0]!.callBudgetMs);
    expect(first.tier2Config?.contextWindowSize).toBe(result.plan.arms[0]!.contextWindowSize);
    expect(first.config.tier0).toBe(true);

    // Each report carries ITS OWN arm's escalation, which is the coupling
    // `gateReport`'s population check cannot see from inside: the two families
    // differ here, 3 selected segments per message against 2.
    const [compiled, tier2Only] = result.reports;
    expect(compiled!.family).toBe("compiled");
    expect(compiled!.escalation.hasPriors).toBe(true);
    expect(compiled!.judgedUnitsPerItem).toEqual({ p50: 1, p95: 3, max: 3, min: 1 });
    expect(tier2Only!.escalation.hasPriors).toBe(false);
    expect(tier2Only!.judgedUnitsPerItem).toEqual({ p50: 1, p95: 2, max: 2, min: 1 });
    expect(compiled!.judgedUnitChars).toEqual(result.plan.arms[0]!.segments.chars);

    // One gates line per arm, and it is the report, so the file a reader gets is
    // the object this function returned.
    const gates = readFileSync(result.gatesPath, "utf8").trim().split("\n");
    expect(gates).toHaveLength(2);
    expect(JSON.parse(gates[0]!)).toEqual(JSON.parse(JSON.stringify(result.reports[0])));
  });

  it("carries the page's own load cost onto every gates row", async () => {
    // The hardware half of "at what latency and hardware cost?". The page
    // measured all three of these on every arm and this driver dropped them, so
    // neither output file could answer the question at all -- every other
    // latency here is a per-CALL number taken once the model is already
    // resident.
    //
    // Asserted against the SCRIPTED page's three distinct values, so this is a
    // check that the driver reads the load report rather than that it can
    // produce a plausible number: with 1/1/0 in the fake, a report hardcoding
    // any of them, or copying `engineLoadMs` into `engineWarmupMs`, passed.
    const dir = outDir();
    const { page } = scriptedPage();
    const result = await runBakeoff(page, bakeoffOptions(dir, { families: ["compiled", "compiled-tier2-only"] }));
    for (const row of result.reports) {
      expect(row.engineLoadMs).toBe(4_321);
      expect(row.engineWarmupMs).toBe(765);
      expect(row.originStorageBytes).toBe(9_876_543_210);
    }
  });

  it("says on every gates line that this corpus cannot score the tier the arms ran", async () => {
    // End to end, over the SHIPPED corpus and the SHIPPED IR, because that is
    // the pair a bake-off run today would use: `smoke.jsonl` carries seven gold
    // spans and none of them is at tier 2, while every arm here runs tier 2 and
    // nothing else. A scorer joining findings to gold would count every correct
    // tier-2 finding as a false positive and rank the arm that found nothing
    // first. The run is still taken -- every gate on this report is a property
    // of the run and needs no label -- and the gap is named in the artifact
    // instead of left to be discovered downstream.
    const dir = outDir();
    const { page } = scriptedPage();
    const result = await runBakeoff(
      page,
      bakeoffOptions(dir, { families: ["compiled", "compiled-tier2-only"] }),
    );

    const gates = readFileSync(result.gatesPath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as ArmGateReport);
    expect(gates).toHaveLength(2);
    for (const row of gates) {
      expect(row.scoring.goldSpansByTier).toEqual({ 0: 5, 1: 2, 2: 0 });
      // SORTED, not in the order the corpus happens to mention them: the corpus
      // introduces in-pan, then aws-key, then generic-secret, so an
      // insertion-ordered list differs here. Two arms' rows are read side by
      // side and an unstable order makes them look different when they are not.
      expect(row.scoring.tiers.find((t) => t.tier === 0)!.goldEntityTypes).toEqual([
        "aws-key",
        "generic-secret",
        "in-pan",
      ]);
      expect(row.scoring.tiersTheseRowsCannotScore).toEqual([2]);
      expect(row.scoring.goldPolicies).toEqual(["minimal-fixture"]);
      expect(row.scoring.cannotScore.join(" ")).toContain("tier 2");
      expect(row.scoring.accuracyGated).toBe(false);
      // The verdict does not become "unknown" for it: a run gate is still a run
      // gate, and dropping the arm is what this driver must never do.
      expect(row.killedOnRunGates).toBe(false);
    }
    // And the tier-0 family's extra tier is read off ITS rows, not off a shared
    // constant: the two families ran different tier sets.
    expect(gates.find((g) => g.family === "compiled")!.scoring.tiersRun).toEqual([0, 2]);
    expect(gates.find((g) => g.family === "compiled-tier2-only")!.scoring.tiersRun).toEqual([2]);
  });

  it("refuses when the page's IR is not the file this process planned against", async () => {
    // The driver would plan the ceiling and the escalation from one policy while
    // the page ran another, and every record would carry the page's hash -- so
    // nothing in the output would disagree with itself.
    const dir = outDir();
    const { page } = scriptedPage({ irHash: "b".repeat(64) });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/would plan the bake-off against one policy/);
  });

  it("refuses when the page's IR changes between the plan and an arm", async () => {
    // A second `useIr` after each arm's navigation, checked separately, because
    // a page that answered correctly once can be re-navigated to a different
    // server or reloaded with a different bundle mid-run.
    const dir = outDir();
    const { page } = scriptedPage({ irHashAfterNavigation: "c".repeat(64) });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/different IR hash after navigating/);
  });

  it("refuses a page served by another checkout", async () => {
    // `playwright.config.ts` reuses an existing dev server outside CI, so a
    // server left running by another worktree serves the page every number in
    // this bake-off would be measured on -- uniformly across arms, so never as a
    // disagreement between two of them.
    const dir = outDir();
    const { page } = scriptedPage({ harnessDir: "/somewhere/else/apps/eval" });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/built by \/somewhere\/else/);
  });

  it("refuses to run at all when WebGPU is absent", async () => {
    // Tier 2 is ABSENT rather than degraded without it, so every arm would write
    // a complete, schema-valid file of zero model calls.
    const dir = outDir();
    const { page } = scriptedPage({ webgpu: false });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/WebGPU is not available/);
  });

  it("refuses an engine that answered as a different model", async () => {
    const dir = outDir();
    const { page } = scriptedPage({ servedModelId: "Phi-4-mini-instruct-q4f16_1-MLC" });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/every finding would be recorded against a model that did not run/);
  });

  it("refuses a page that resolved a different context window than the arm asked for", async () => {
    // Two arms at different windows measure context rather than method, and the
    // page resolving one silently is how that happens: `contextWindowSize` in
    // the wrong argument position to `CreateMLCEngine` is dropped by the library.
    const dir = outDir();
    const { page } = scriptedPage({ contextWindowSize: 4096 });
    await expect(runBakeoff(page, bakeoffOptions(dir, { contextWindowSize: 8192 }))).rejects.toThrow(
      /two arms at different windows measure context rather than method/,
    );
  });

  it("refuses a page that resolved a different per-call budget than the arm asked for", async () => {
    // The per-item ceiling was derived from the budget this driver asked for, so
    // an arm running under a different one is not bounded by it.
    const dir = outDir();
    const { page } = scriptedPage({ callBudgetMs: 30_000 });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/the per-item ceiling was derived from/);
  });

  it("refuses an arm whose every item failed", async () => {
    // The file would be a complete, schema-valid, perfectly scoreable transcript
    // of nothing having been measured, which Plan 8 reads as an arm with no
    // recall rather than as a run that died.
    const dir = outDir();
    const { page } = scriptedPage({ detectThrows: true });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/failed on all \d+ of its items/);
    // And nothing was written: the refusal is before the file, not after it.
    expect(existsSync(join(dir, "run1.tier2-" + MODEL + ".jsonl"))).toBe(false);
  });

  it("refuses a record the schema would reject, before writing the file", async () => {
    // `RunRecordSchema` re-checks every finding's span against the message text,
    // and nothing on `runArm`'s path validates. Without this check the arm would
    // write a file its own reader refuses -- discovered after the GPU hours, by
    // whoever tried to score it.
    const dir = outDir();
    const { page } = scriptedPage({ emitBadSpan: true });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/produced an invalid record at row 1/);
    expect(existsSync(join(dir, "run1.tier2-" + MODEL + ".jsonl"))).toBe(false);
  });

  it("refuses a runId whose arm file already exists, before the first model loads", async () => {
    // Checked up front rather than at write time, so a repeated runId costs no
    // GPU hours: overwriting would replace one measurement with another.
    const dir = outDir();
    const armPath = join(dir, `run1.tier2-${MODEL}.jsonl`);
    writeFileSync(armPath, "");
    const { page } = scriptedPage();
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/already exists/);
    // Unchanged: the refusal did not truncate the file it was protecting.
    expect(readFileSync(armPath, "utf8")).toBe("");
  });

  it("refuses a gates file that appears AFTER the pre-flight check", async () => {
    // The window the exclusive-create flag exists for, and it is not the one the
    // two tests below cover: those are caught by `existsSync` before a model
    // loads. This one appears while the run is going -- another process starting
    // the same runId an hour in -- and plain "a" would silently append this
    // run's verdicts to that one's file.
    const dir = outDir();
    const gatesPath = join(dir, "run1.gates.jsonl");
    const { page } = scriptedPage({ createOnFirstLoad: [gatesPath] });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(
      /run1\.gates\.jsonl was created while this bake-off was running/,
    );
    // Unchanged: the refusal did not append to the file it was protecting.
    expect(readFileSync(gatesPath, "utf8")).toBe("");
  });

  it("refuses an arm file that appears AFTER the pre-flight check", async () => {
    // The same window, one file over. `wx` on the arm write is what catches it.
    const dir = outDir();
    const armPath = join(dir, `run1.tier2-${MODEL}.jsonl`);
    const { page } = scriptedPage({ createOnFirstLoad: [armPath] });
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(
      /was created while this bake-off was running/,
    );
    expect(readFileSync(armPath, "utf8")).toBe("");
  });

  it("refuses a runId whose gates file already exists", async () => {
    const dir = outDir();
    writeFileSync(join(dir, "run1.gates.jsonl"), "");
    const { page } = scriptedPage();
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/use a different runId/);
  });

  it("writes each arm's gates row as that arm finishes, not after the last one", async () => {
    // THE ARTIFACT A PARTIAL RUN LEAVES. The gates file used to be written once,
    // after every arm, on the argument that "every verdict in this file is
    // recomputable from" the JSONL. It is not, and `GateReportInput` says so in
    // three docblocks: `entityTypes`, `itemTimeoutMs`/`latencyBudgetMs` and the
    // load costs are each "the Nth thing no record carries". So a run that died
    // on arm 2 left arm 1's complete, schema-valid, fully scoreable file with no
    // `scoring`, no `cannotScore` and no `experimentScope` anywhere on disk --
    // the silent zero-precision arm, in the one output that survives.
    const dir = outDir();
    const { page } = scriptedPage({ detectThrowsOnArm: 2 });
    const gatesPath = join(dir, "run1.gates.jsonl");
    await expect(
      runBakeoff(page, bakeoffOptions(dir, { families: ["compiled", "compiled-tier2-only"] })),
    ).rejects.toThrow(/failed on all \d+ of its items/);

    // Arm 1 finished, so arm 1's file AND arm 1's gates row are both on disk.
    const armPath = join(dir, `run1.tier2-${MODEL}.jsonl`);
    expect(existsSync(armPath)).toBe(true);
    expect(existsSync(gatesPath)).toBe(true);
    const gates = readFileSync(gatesPath, "utf8").trim().split("\n").map((l) => JSON.parse(l) as ArmGateReport);
    // ONE row, for the ONE arm that produced a file: a row for an arm that
    // never wrote one would be a verdict on nothing.
    expect(gates).toHaveLength(1);
    expect(gates[0]!.arm).toBe(`tier2-${MODEL}`);
    expect(gates[0]!.scoring.cannotScore.length).toBeGreaterThan(0);
    expect(gates[0]!.experimentScope).toContain("SCOPE OF THIS RUN");
    // And arm 2 wrote neither: its refusal is before its file.
    expect(existsSync(join(dir, `run1.tier2only-${MODEL}.jsonl`))).toBe(false);
  });

  it("says on every gates row that gold at a tier no arm ran caps recall", async () => {
    // The mirror of the tier-2 gap, and it was unnamed until this round. The
    // shipped corpus carries two `client-name` spans, `client-name` is tier 1 in
    // both runnable IRs, and `planBakeoff` sets `tier1: false` on every arm --
    // so 2 of 7 gold spans are in a recall denominator no arm here can fill,
    // and nothing in the artifact said so.
    const dir = outDir();
    const { page } = scriptedPage();
    const result = await runBakeoff(page, bakeoffOptions(dir, { families: ["compiled"] }));
    const row = result.reports[0]!;
    expect(row.scoring.tiersWithGoldThisArmDidNotRun).toEqual([1]);
    expect(row.scoring.goldSpansByTier).toEqual({ 0: 5, 1: 2, 2: 0 });
    expect(row.scoring.cannotScore.join(" ")).toContain("did NOT run tier 1");
    expect(row.scoring.cannotScore.join(" ")).toContain("bounded above by 5/7");
    expect(row.scoring.cannotScore.join(" ")).toContain("client-name");
  });

  it("names the methods and models THIS run crossed, not a constant four", async () => {
    // `experimentScope` was a constant asserting "the four pinned arms" and four
    // methods whatever ran. `slateBakeoffOptions` leaves `families` unset and
    // `DEFAULT_FAMILIES` is `["compiled"]`, so the shipped slate command put a
    // claim of a compiled-versus-Approach-B head-to-head -- the project's
    // central question -- on every row of a one-method run.
    const dir = outDir();
    const { page } = scriptedPage();
    const result = await runBakeoff(page, bakeoffOptions(dir));
    expect(result.plan.arms).toHaveLength(1);
    const scope = result.reports[0]!.experimentScope;
    expect(scope).toContain("MODEL axis at 1 point(s)");
    expect(scope).toContain("METHOD axis at 1 point(s) (tier 0 + compiled judge)");
    expect(scope).not.toContain("Approach B");
  });

  describe("the Approach-B door", () => {
    /**
     * A four-family head-to-head against the compiled policy, which is the only
     * pairing that exists here.
     *
     * These are the same options `test/baseline.spec.ts` passes, minus the GPU:
     * every guard under test is DRIVER logic reading a value the page reported,
     * and until this block none of the five had a negative test on any machine.
     * Deleting any one of them left the whole suite green -- measured, one
     * mutant at a time, against a `git archive HEAD` copy.
     */
    const bOptions = (dir: string, over: Partial<BakeoffOptions> = {}): BakeoffOptions =>
      bakeoffOptions(dir, {
        families: ["compiled", "baseline-b"],
        irName: "p-fin",
        irPath: B_IR_PATH,
        policyPath: B_POLICY_PATH,
        itemTimeoutMs: B_ITEM_TIMEOUT_MS,
        ...over,
      });

    it("projects the two SPAN counters off the page, on both arms, at values that are not 0", async () => {
      // FOUND BY MUTATION -- replacing `run.ts`'s Approach-B projection of
      // `unresolvedMentions` and `wholeClauseMentions` with literal `0`s left
      // the whole apps/eval suite green. Every existing driver test sees both
      // counters at 0 on both arms, and a test that only exercises the DEFAULT
      // value cannot tell "reads the field" from "writes the default".
      //
      // Distinct primes per counter and per arm, so a projection that read the
      // wrong field, or read the judge's counters on the B row, fails here
      // rather than agreeing with itself. Both arms in ONE test because the
      // symmetry is the property: `resolvable-rate` puts `unresolvedMentions`
      // in its denominator and `wholeClauseMentions` in the whole-clause
      // column, and an arm whose counters silently read 0 while it is really
      // losing mentions moves a gate that can kill an arm.
      const dir = outDir();
      const { page } = scriptedPage({
        irHash: B_IR_HASH,
        counters: { unresolvedMentions: 13, wholeClauseMentions: 17 },
      });
      const result = await runBakeoff(page, bOptions(dir, { families: ["compiled", "baseline-b"] }));
      const [compiled, baseline] = result.written.map((path) =>
        readFileSync(path, "utf8").trim().split("\n").map((l) => JSON.parse(l) as RunRecord),
      );
      for (const row of compiled!) {
        expect(row.tier2Stats?.unresolvedMentions).toBe(13);
        expect(row.tier2Stats?.wholeClauseMentions).toBe(17);
      }
      for (const row of baseline!) {
        expect(row.baselineStats?.unresolvedMentions).toBe(13);
        expect(row.baselineStats?.wholeClauseMentions).toBe(17);
      }
      // The stub really does default these to 0, so the assertions above are
      // reading the knob and not a coincidence.
      const { page: plain } = scriptedPage({ irHash: B_IR_HASH });
      const plainResult = await runBakeoff(
        plain,
        bOptions(outDir(), { families: ["baseline-b"] }),
      );
      const plainRows = readFileSync(plainResult.written[0]!, "utf8")
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l) as RunRecord);
      for (const row of plainRows) {
        expect(row.baselineStats?.unresolvedMentions).toBe(0);
        expect(row.baselineStats?.wholeClauseMentions).toBe(0);
      }
    });

    it("runs a B arm end to end and files its counters under baselineStats", async () => {
      // The whole B branch of `runArm` -- the detector routing, the `isBaseline`
      // flag, the `baselineStatus()` read and the 14-field projection -- was
      // exercised by no test that runs without WebGPU: all 27 arms in
      // `run.spec.ts` pass `detector: "core-orchestrator"` and `baselineStatus`
      // appeared in no test file at all.
      const dir = outDir();
      const { page, baselineLoads } = scriptedPage({ irHash: B_IR_HASH });
      const result = await runBakeoff(
        page,
        bOptions(dir, { families: ["compiled", "baseline-b", "baseline-b-tier0"] }),
      );
      expect(result.written).toHaveLength(3);
      const rows = result.written.map((path) =>
        readFileSync(path, "utf8").trim().split("\n").map((l) => JSON.parse(l) as RunRecord),
      );
      const [compiled, baseline, baselineTier0] = rows;
      // WHICH B CONSTRUCTOR the page was asked for, per arm. The two B families
      // differ only in whether core's tier 0 runs in front of the model, and
      // the driver picks between them from `familyShape(arm.family).runsTier0`
      // -- so a `loadBaseline` call hardcoded to "baseline-b" would build the
      // same arm twice and file it under two names, with every record, every
      // gates row and the whole suite green.
      expect(baselineLoads()).toEqual([
        { family: "baseline-b", policy: "p-fin" },
        { family: "baseline-b-tier0", policy: "p-fin" },
      ]);
      // And the tier-0 half really is on in one and off in the other, read off
      // the rows rather than off the family label.
      expect(new Set(baseline!.map((r) => r.config.tier0))).toEqual(new Set([false]));
      expect(new Set(baselineTier0!.map((r) => r.config.tier0))).toEqual(new Set([true]));
      for (const row of baselineTier0!) expect(row.detector).toBe("approach-b");
      for (const row of compiled!) {
        expect(row.detector).toBe("core-orchestrator");
        expect(row.tier2Stats).toBeDefined();
        expect(row.baselineStats).toBeUndefined();
        // The compiled arm escalates, so its threshold is a knob that turned.
        expect(row.config.uncertainBelow).toBeDefined();
      }
      for (const row of baseline!) {
        expect(row.detector).toBe("approach-b");
        // The 14-field projection, off the page's OWN counters rather than the
        // judge's: `unresolvedQuotes` is 2 in the stub and `unknownEntityTypes`
        // is a field `tier2Stats` does not have at all.
        expect(row.baselineStats?.unresolvedQuotes).toBe(2);
        expect(row.baselineStats?.messagesJudged).toBe(1);
        expect(row.baselineStats?.calls).toHaveLength(1);
        expect(row.tier2Stats).toBeUndefined();
        // B does not escalate, so a threshold here would name a knob that
        // turned nothing -- and `RunRecordSchema` refuses one.
        expect(row.config.uncertainBelow).toBeUndefined();
      }
      // And the gates row is filed under the family that produced it, with B's
      // one-call-per-MESSAGE unit rather than the judge's per-segment one.
      const b = result.reports.find((r) => r.family === "baseline-b")!;
      expect(b.judgedUnit).toBe("message");
      expect(b.run.uncertainBelow).toBeUndefined();
    });

    it("names the gold ids the COMPILED policy does not declare, which is not empty", async () => {
      // `goldEntityTypesNotInIr`'s docblock called this "a standing candidate"
      // that "is empty today". It is not: `smoke.jsonl`'s gold is labelled under
      // `minimal-fixture` and carries `aws-key` and `generic-secret`, which
      // `semantic-ir.json` declares and `p-fin.ir.json` does not -- so the one
      // head-to-head this repository actually performs populates it on every
      // row. MEASURED here over the same three-item slice `baseline.spec.ts`
      // uses, so this is that run's number and not a derivation.
      const dir = outDir();
      const slicePath = join(dir, "slice.jsonl");
      writeFileSync(slicePath, readFileSync(CORPUS, "utf8").trim().split("\n").slice(0, 3).join("\n") + "\n");
      const { page } = scriptedPage({ irHash: B_IR_HASH });
      const result = await runBakeoff(page, bOptions(dir, { corpus: slicePath }));
      for (const row of result.reports) {
        expect(row.scoring.goldEntityTypesNotInIr).toEqual(["aws-key", "generic-secret"]);
        expect(row.scoring.goldSpansByTier).toEqual({ 0: 1, 1: 0, 2: 0 });
        expect(row.scoring.goldPolicies).toEqual(["minimal-fixture"]);
        expect(row.scoring.cannotScore.join(" ")).toContain("are not declared by the IR this arm");
      }
      // And the multiple this run's latencies were taken at is 1, not 24: p-fin
      // carries the compiler's own default budget. This is the assertion
      // `baseline.spec.ts` makes on a GPU, made here without one.
      for (const row of result.reports) {
        expect(row.run.latencyBudgetTimesCompilerDefault).toBe(1);
        expect(row.run.latencyBudgetMs).toBe(5_000);
      }
      // So the p95 detail must NOT carry the 24x caveat.
      const ttft = result.reports[0]!.gates.find((g) => g.gate === "p95-ttft")!;
      expect(ttft.detail).toContain("that IS a compiled policy's own default budget");
      expect(ttft.detail).not.toContain("degradation that difference causes is measured nowhere");
    });

    it("refuses when the page's policy document is not the one this driver read", async () => {
      // `reuseExistingServer: !CI` means a dev server from another worktree
      // serves the document B is shown, and NO record field carries a digest of
      // it -- `policyHash` on a row is the IR's field, the hash of the document
      // the IR was COMPILED from, not of the text B was handed.
      const dir = outDir();
      const { page } = scriptedPage({ irHash: B_IR_HASH, policyDocHash: "d".repeat(64) });
      await expect(runBakeoff(page, bOptions(dir))).rejects.toThrow(
        /Approach B would be shown a document this driver never read/,
      );
    });

    it("refuses when the page BUILT B on a document other than the planned one", async () => {
      // A different check from the one above and at a different moment: that one
      // compares digests before any model loads, this one reads back what
      // `loadBaseline` says it actually built on.
      const dir = outDir();
      const { page } = scriptedPage({ irHash: B_IR_HASH, baselinePolicyDocSha256: "e".repeat(64) });
      await expect(runBakeoff(page, bOptions(dir))).rejects.toThrow(
        /built its Approach-B arm on a document hashing/,
      );
    });

    it("refuses when the page paired B's document with a different IR", async () => {
      // The strong one of the four: the page compares its own digest of the
      // document with the loaded IR's `policyHash` and refuses, so this reads
      // back a refusal that has already run -- and catches a page that loaded a
      // stale IR between `useIr` and `loadBaseline`.
      const dir = outDir();
      const { page } = scriptedPage({ irHash: B_IR_HASH, baselineIrPolicyHash: "f".repeat(64) });
      await expect(runBakeoff(page, bOptions(dir))).rejects.toThrow(
        /paired its Approach-B document with an IR whose policyHash/,
      );
    });

    it("refuses when B was built on an engine answering as another model", async () => {
      // B runs on the tier-2 load's engine, so a reload between the two calls
      // would otherwise go unnoticed: every B finding would be recorded against
      // a model that did not produce it.
      const dir = outDir();
      const { page } = scriptedPage({
        irHash: B_IR_HASH,
        baselineServedModelId: "Phi-4-mini-instruct-q4f16_1-MLC",
      });
      await expect(runBakeoff(page, bOptions(dir))).rejects.toThrow(
        /the Approach-B arm was built on an engine answering as/,
      );
    });

    it("refuses two arms at different per-call budgets", async () => {
      // The deadline is the one setting that decides how much output an arm gets
      // to produce, so two arms holding different ones measure the deadline
      // rather than the method.
      const dir = outDir();
      const { page } = scriptedPage({ irHash: B_IR_HASH, baselineCallBudgetMs: 30_000 });
      await expect(runBakeoff(page, bOptions(dir))).rejects.toThrow(
        /two arms at different per-call budgets measure the deadline rather than the method/,
      );
    });

    it("strips the escalation threshold from a B arm's config rather than passing it on", async () => {
      // MEASURED rather than assumed, because `runArm` has a guard that throws
      // on exactly this input and it is UNREACHABLE from here -- `planBakeoff`
      // already omits `uncertainBelow` on a family whose `runsCompiledJudge` is
      // false, so the throw is defence for a direct `runArm` caller and is
      // tested there (`run.spec.ts`). What this asserts is the reason it is
      // unreachable: an explicit non-default threshold reaches the COMPILED
      // arm's rows and no B row at all, so the two families differ on the one
      // knob only one of them has.
      const dir = outDir();
      const { page } = scriptedPage({ irHash: B_IR_HASH });
      const result = await runBakeoff(page, bOptions(dir, { uncertainBelow: 0.55 }));
      const rowsOf = (path: string) =>
        readFileSync(path, "utf8").trim().split("\n").map((l) => JSON.parse(l) as RunRecord);
      const compiled = result.plan.arms.findIndex((a) => a.family === "compiled");
      const baseline = result.plan.arms.findIndex((a) => a.family === "baseline-b");
      for (const row of rowsOf(result.written[compiled]!)) {
        expect(row.config.uncertainBelow).toBe(0.55);
      }
      for (const row of rowsOf(result.written[baseline]!)) {
        expect(row.config.uncertainBelow).toBeUndefined();
      }
      expect(result.reports[compiled]!.run.uncertainBelow).toBe(0.55);
      expect(result.reports[baseline]!.run.uncertainBelow).toBeUndefined();
    });
  });

  it("refuses a baseline arm here rather than running one with nowhere to run", async () => {
    // `assertPageCanRun` is a separate question from planning and this is where
    // the two meet. The shipped fixture cannot host a baseline arm at all --
    // its policyHash is not a sha256 -- so planning refuses first, and the
    // message names the document check rather than the missing door.
    const dir = outDir();
    const { page } = scriptedPage();
    await expect(
      runBakeoff(page, bakeoffOptions(dir, { families: ["baseline-b"], policyPath: join(REPO_ROOT, "policies", "p-corp.md") })),
    ).rejects.toThrow(/policyHash/);
  });
});

/**
 * Where this driver looks for an IR, given only the name the PAGE serves it by.
 *
 * ## The defect these pin
 *
 * MEASURED before the fix, by running the documented invocation: the page's
 * `IR_FIXTURES` genuinely registers "p-fin" (it `?raw`-imports
 * `policies/compiled/p-fin.ir.json`), `slateBakeoffOptions` genuinely accepts
 * `SIH_BAKEOFF_IR=p-fin`, and `runBakeoff` then resolved
 * `apps/eval/fixtures/p-fin-ir.json` -- a path this repository has never had --
 * and died inside `readFileSync` with a bare `ENOENT`. The requirement was
 * written down (`slateBakeoffOptions`'s docblock said p-fin "needs
 * SIH_BAKEOFF_IR_PATH and SIH_BAKEOFF_POLICY_PATH together") and enforced
 * nowhere, which is this project's recurring shape: the knowledge lives in a
 * comment and not in the code.
 */
describe("the IR name the page serves and the file this driver reads", () => {
  it("has no file where the fixture convention would put the compiled policy's IR", () => {
    // The PREMISE of everything below, asserted rather than assumed: if someone
    // ever adds `apps/eval/fixtures/p-fin-ir.json`, the convention starts
    // resolving and the two tests after this stop testing what they claim.
    expect(existsSync(join(REPO_ROOT, "apps", "eval", "fixtures", "p-fin-ir.json"))).toBe(false);
  });

  it("resolves the compiled policy's IR from the registry name alone", () => {
    // Not a second copy of the page's registry standing unchecked: `runBakeoff`
    // compares its own sha256 of whatever this resolved against the digest
    // `useIr(name)` returns from the page's bundled bytes, and refuses the run
    // on a mismatch ("refuses when the page's IR is not the file this process
    // planned against", above). What this asserts is that the resolution lands
    // on the file whose bytes the page actually carries.
    expect(resolveIrPath({ irName: "p-fin" })).toBe(B_IR_PATH);
    expect(existsSync(resolveIrPath({ irName: "p-fin" }))).toBe(true);
  });

  it("still resolves the three fixtures the convention does cover", () => {
    // The CONTROL for the line above: the off-convention entry must not have
    // replaced the convention, or `SIH_BAKEOFF_IR=multiclass` -- which works
    // today and is nobody's defect -- would start refusing.
    for (const name of ["minimal", "multiclass", "semantic"]) {
      const resolved = resolveIrPath({ irName: name });
      expect(resolved).toBe(join(REPO_ROOT, "apps", "eval", "fixtures", `${name}-ir.json`));
      expect(existsSync(resolved)).toBe(true);
    }
    // And the default, which is the one a run with no `SIH_BAKEOFF_IR` takes.
    expect(resolveIrPath({})).toBe(IR_PATH);
  });

  it("keeps an explicit path exactly as the caller gave it", () => {
    // `irPath` wins over both the table and the convention, because a caller
    // who named a file is not asking to be second-guessed -- and it is how
    // `test/baseline.spec.ts` and every B test above select p-fin today.
    expect(resolveIrPath({ irName: "p-fin", irPath: "/tmp/somewhere-else.ir.json" })).toBe(
      "/tmp/somewhere-else.ir.json",
    );
  });

  it("refuses a name it has no file for, naming the cause rather than the syscall", () => {
    // The failure mode the fix is FOR. Before it, a name outside the convention
    // produced `ENOENT: no such file or directory, open '.../typo-ir.json'`
    // from four layers inside `runBakeoff` -- a path that was never going to
    // exist, about a name the page may well serve.
    let thrown: unknown;
    try {
      resolveIrPath({ irName: "typo" });
    } catch (error) {
      thrown = error;
    }
    const message = String(thrown);
    expect(message).toContain("typo");
    // The path it TRIED, so a reader can see what the convention would have
    // wanted and decide whether to move the file or pass the option.
    expect(message).toContain(join("fixtures", "typo-ir.json"));
    // Both spellings of the way out, once: the option a programmatic caller
    // passes and the variable the shipped command reads.
    expect(message).toContain("irPath");
    expect(message).toContain("SIH_BAKEOFF_IR_PATH");
    // And it says WHY the name being accepted elsewhere is not evidence this
    // driver can find it -- the exact confusion that produced the defect.
    expect(message).toContain("registry");
    // Not a re-thrown syscall error.
    expect(message).not.toContain("ENOENT");
  });

  it("runs the whole slate from `irName: p-fin` with no path at all", async () => {
    // End to end through `runBakeoff`, because that is where the ENOENT was:
    // the resolution above is only worth anything if the function that reads
    // the file uses it. The page is scripted to report p-fin's digest, so the
    // two-sided hash check is a real comparison here and not a tautology --
    // hand it `IR_HASH` instead and this test goes red on the mismatch.
    const dir = outDir();
    const { page } = scriptedPage({ irHash: B_IR_HASH });
    const result = await runBakeoff(
      page,
      bakeoffOptions(dir, { irName: "p-fin", irPath: undefined, itemTimeoutMs: B_ITEM_TIMEOUT_MS }),
    );
    expect(result.written).toHaveLength(1);
    // Every record names the IR the page served, which is p-fin's and not the
    // default fixture's -- so the run really did switch policies rather than
    // resolving the old file under a new name.
    const rows = readFileSync(result.written[0]!, "utf8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l) as RunRecord);
    expect(rows).toHaveLength(CORPUS_ITEMS);
    for (const row of rows) expect(row.irHash).toBe(B_IR_HASH);
    expect(B_IR_HASH).not.toBe(IR_HASH);
  });
});
