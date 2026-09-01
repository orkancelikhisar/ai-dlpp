import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Page } from "@playwright/test";
import { afterEach, describe, expect, it } from "vitest";
import { RunRecordSchema, type RunRecord } from "../src/driver/record.js";
import { runBakeoff, type ArmGateReport, type BakeoffOptions } from "../src/driver/bakeoff.js";

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
  /** Emit a finding whose span does not lie inside the item text. */
  emitBadSpan?: boolean;
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
function scriptedPage(script: PageScript = {}): { page: Page; unloads: () => number } {
  let firstUseIr = true;
  let unloads = 0;
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
    webgpuAvailable: async () => script.webgpu ?? true,
    loadTier2: async (options: { modelId: string; contextWindowSize: number; callBudgetMs: number }) => ({
      config: {
        modelId: options.modelId,
        contextWindowSize: script.contextWindowSize ?? options.contextWindowSize,
        temperature: 0,
        maxTokens: 512,
      },
      callBudgetMs: script.callBudgetMs ?? options.callBudgetMs,
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
      return detection(request.text, script);
    },
    tier2Status: () => ({
      lastDetect: {
        rung1: 1,
        rung2: 0,
        unresolvedQuotes: 0,
        unknownPredicates: 0,
        duplicatesDropped: 0,
        repairAttempts: 0,
        failedClosed: 0,
        truncatedResponses: 0,
        abortedResponses: 0,
        segmentsJudged: 1,
        segmentsSkipped: 0,
        deadlineExpiries: 0,
        callerAbortsMidGeneration: 0,
        callerAbortsWhileQueued: 0,
        calls: [
          { finishReason: "stop", promptTokens: 400, completionTokens: 40, ttftMs: 480, decodeTokPerSec: 45 },
        ],
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
  return { page: page as unknown as Page, unloads: () => unloads };
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
      expect(row.scoring.tiersThisCorpusCannotScore).toEqual([2]);
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

  it("refuses a runId whose gates file already exists", async () => {
    const dir = outDir();
    writeFileSync(join(dir, "run1.gates.jsonl"), "");
    const { page } = scriptedPage();
    await expect(runBakeoff(page, bakeoffOptions(dir))).rejects.toThrow(/use a different runId/);
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
