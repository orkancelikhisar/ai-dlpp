import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIER2_MODELS } from "@sih/tier2";
import { runBakeoff, type ArmFamily, type ArmGateReport } from "../src/driver/bakeoff.js";
import { RunRecordSchema, type RunRecord } from "../src/driver/record.js";
import { expect, openHarness, test } from "./tier2-profile.js";

/**
 * The compiler-versus-prompting head-to-head, running.
 *
 * This is the arm that makes the project's central claim falsifiable, and for
 * most of Plan 5 it could not run at all: the page published only core's
 * orchestrator, `RunRecordSchema` had nowhere to put `BaselineStats`, and every
 * IR here carried `policyHash: "test-hash"` so no document could be paired with
 * one. All three are closed, and this file is the end-to-end proof rather than
 * a claim about it -- everything below drives the real `runBakeoff` against a
 * real model and reads the files it wrote.
 *
 * ## What it is NOT
 *
 * It is not the bake-off and it produces no accuracy number in either
 * direction. `corpora/fixtures/smoke.jsonl` is labelled under `minimal-fixture`
 * and its gold entityTypes are not p-fin's, so a scorer joining findings to
 * gold here would score nothing meaningful -- which is why every assertion
 * below is STRUCTURAL: which stats field a row carries, which prompt each arm
 * was given, which notices each files. Not one of them reads what the model
 * said, because that varies and because tuning a corpus until a model looks
 * good is the failure mode this project has already had to revert once.
 *
 * Three items rather than thirteen keeps a suite run to well under a minute
 * (MEASURED: 41 s for all four arms over three items on the cheapest arm,
 * including four warm model loads). They are the FIRST three of the shipped
 * corpus, sliced and not selected.
 */

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const CORPUS = join(REPO_ROOT, "corpora", "fixtures", "smoke.jsonl");
/** The only compiled artifact in the repository; see policies/compiled. */
const IR_PATH = join(REPO_ROOT, "policies", "compiled", "p-fin.ir.json");
const POLICY_PATH = join(REPO_ROOT, "policies", "p-fin.md");

/** The cheapest arm, which is the one `TIER2_MODELS` puts first. */
const CHEAPEST = TIER2_MODELS[0]!.id;

/**
 * Well above the bound `p-fin.ir.json`'s 5,000 ms message budget and the page's
 * 60,000 ms per-call budget produce for either family. It exists to catch a
 * wedge, not to enforce a latency target.
 */
const ITEM_TIMEOUT_MS = 400_000;

/** Four warm cache reads of a 1 GB model, plus four corpus passes. */
const RUN_TIMEOUT_MS = 900_000;

/**
 * The four families, in the order that puts each PAIR side by side.
 *
 *   compiled-tier2-only (judge alone)         <->  baseline-b
 *   compiled            (tier 0 + judge)      <->  baseline-b-tier0
 *
 * The pairing is what separates "compiling the policy helps" from "having
 * deterministic patterns helps": comparing B alone against a tier-0-plus-judge
 * pipeline credits tier 0's findings to the compiler.
 */
const FAMILIES: readonly ArmFamily[] = [
  "compiled-tier2-only",
  "baseline-b",
  "compiled",
  "baseline-b-tier0",
];

function rowsOf(path: string): RunRecord[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .map((line) => {
      const parsed = RunRecordSchema.safeParse(JSON.parse(line));
      if (!parsed.success) {
        throw new Error(`invalid record in ${path}: ${JSON.stringify(parsed.error.issues)}`);
      }
      return parsed.data;
    });
}

test("runs Approach B against the compiled arms over one compiled policy", async ({ page }) => {
  test.setTimeout(RUN_TIMEOUT_MS);
  await openHarness(page);
  test.skip(
    !(await page.evaluate(() => window.__sih!.webgpuAvailable())),
    "WebGPU unavailable; tier 2 is ABSENT on this machine, not degraded",
  );

  const out = mkdtempSync(join(tmpdir(), "sih-baseline-"));
  const slice = readFileSync(CORPUS, "utf8").trim().split("\n").slice(0, 3);
  const slicePath = join(out, "slice.jsonl");
  writeFileSync(slicePath, slice.join("\n") + "\n");

  const result = await runBakeoff(page, {
    runId: "headtohead",
    outDir: out,
    corpus: slicePath,
    provider: "claude",
    models: [CHEAPEST],
    families: FAMILIES,
    irName: "p-fin",
    irPath: IR_PATH,
    policyPath: POLICY_PATH,
    itemTimeoutMs: ITEM_TIMEOUT_MS,
  });

  expect(result.written).toHaveLength(4);
  const gates = readFileSync(result.gatesPath, "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l) as ArmGateReport);
  expect(gates).toHaveLength(4);
  const byFamily = new Map(gates.map((g) => [g.family, g]));
  const rowsByFamily = new Map(
    result.plan.arms.map((arm, i) => [arm.family, rowsOf(result.written[i]!)]),
  );

  for (const g of gates) {
    console.log(
      `[head-to-head] ${g.arm} (${g.family}, judgedUnit ${g.judgedUnit}): ` +
        `${g.answeredCalls} answered call(s), promptTokens ${JSON.stringify(g.promptTokens)}, ` +
        `ttftMs ${JSON.stringify(g.ttftMs)}, ladder ${JSON.stringify(g.ladder)}, ` +
        `gates ${g.gates.map((x) => `${x.gate}=${x.verdict}`).join(" ")}, ` +
        `killedOnRunGates=${g.killedOnRunGates}`,
    );
  }

  // EVERY arm reached the engine. Without this the four files below are
  // complete, schema-valid transcripts of models that never ran, and every gate
  // would read "not measured" while the run looked like it happened.
  for (const g of gates) expect(g.answeredCalls, g.arm).toBeGreaterThan(0);

  // ONE experiment: the same IR artifact, the same source policy, the same
  // model, the same corpus and the same per-item ceiling on all four arms. This
  // is what makes the four numbers comparable at all, and it is checked off the
  // ROWS rather than off the options this test passed.
  const allRows = [...rowsByFamily.values()].flat();
  expect(new Set(allRows.map((r) => r.irHash)).size).toBe(1);
  expect(new Set(allRows.map((r) => r.policyHash)).size).toBe(1);
  expect(new Set(allRows.map((r) => r.config.t2Model))).toEqual(new Set([CHEAPEST]));
  expect(new Set(gates.map((g) => g.run.itemTimeoutMs)).size).toBe(1);
  expect(new Set(gates.map((g) => JSON.stringify(g.run.tier2Config))).size).toBe(1);
  // The compiled IR's budget is the compiler's own default, not a fixture's
  // lifted one -- so unlike every earlier run in this repository these
  // latencies are taken at 1x a shipped policy's message budget.
  expect(new Set(gates.map((g) => g.run.latencyBudgetTimesCompilerDefault))).toEqual(new Set([1]));

  // Each row says which implementation produced it, and carries that
  // implementation's counters and no other's. This is the record half of the
  // gap: before it, a B arm's counters had nowhere to go but `tier2Stats`,
  // where `messagesJudged` would have been written under a field named for
  // segments.
  for (const [family, rows] of rowsByFamily) {
    const isBaseline = family.startsWith("baseline-b");
    for (const row of rows) {
      expect(row.detector, `${family}/${row.itemId}`).toBe(
        isBaseline ? "approach-b" : "core-orchestrator",
      );
      expect(row.error).toBeNull();
      expect(row.baselineStats !== undefined, `${family} baselineStats`).toBe(isBaseline);
      expect(row.tier2Stats !== undefined, `${family} tier2Stats`).toBe(!isBaseline);
      // Approach B never escalates, so a threshold on its row would name a knob
      // that turned nothing. The compiled arms must carry one.
      expect(row.config.uncertainBelow === undefined, `${family} uncertainBelow`).toBe(isBaseline);
    }
  }

  // ONE CALL PER MESSAGE against one per selected segment, which is the one
  // place the two methods legitimately differ in cost. Asserted off the CALL
  // ROWS, so it is what the engine did rather than what the plan expected.
  const callsPerItem = (family: ArmFamily): number[] =>
    rowsByFamily
      .get(family)!
      .map((r) => (r.baselineStats ?? r.tier2Stats)!.calls.length);
  expect(callsPerItem("baseline-b")).toEqual([1, 1, 1]);
  expect(callsPerItem("baseline-b-tier0")).toEqual([1, 1, 1]);
  // The compiled arms make more, because escalation selects several segments
  // from these three messages. `> 3` rather than an exact figure: the exact
  // count is `planBakeoff`'s business and `bakeoff.test.ts` pins it there.
  expect(callsPerItem("compiled").reduce((a, b) => a + b, 0)).toBeGreaterThan(3);

  // THE ASYMMETRY THAT DECIDES HOW TO READ THE LATENCY GATE, asserted rather
  // than described. B carries the whole 5,272-character policy document in
  // every prompt where the judge carries one segment, so B's prompts are
  // several times larger -- and `GATES.maxP95TtftMs` was derived at the
  // judge's ~1.1 kB. A B arm failing that gate is failing a threshold set for
  // different work, which is why `judgedUnit` and `promptTokens` are on the row
  // beside the verdict.
  const b = byFamily.get("baseline-b")!;
  const compiledOnly = byFamily.get("compiled-tier2-only")!;
  expect(b.judgedUnit).toBe("message");
  expect(compiledOnly.judgedUnit).toBe("segment");
  expect(b.promptTokens!.min).toBeGreaterThan(compiledOnly.promptTokens!.max);
  expect(b.gates.find((g) => g.gate === "p95-ttft")!.detail).toContain("judgedUnitChars");

  // THE SCOPE DIFFERENCE, which is B's advantage and is intrinsic rather than a
  // harness artifact. p-fin's one semantic predicate is declared
  // `scope: "message"`; `WebLlmJudge` evaluates every predicate against a
  // SEGMENT and reports `scopesJudged: ["segment"]`, so the orchestrator files
  // one `scope-unjudged` notice per message. B puts the whole message in one
  // call, so nothing goes unasked and it files none.
  expect(compiledOnly.degradedNotices["scope-unjudged"]).toBe(3);
  expect(byFamily.get("compiled")!.degradedNotices["scope-unjudged"]).toBe(3);
  expect(b.degradedNotices["scope-unjudged"]).toBe(0);
  expect(byFamily.get("baseline-b-tier0")!.degradedNotices["scope-unjudged"]).toBe(0);

  // And the tier-0 half really does differ between the paired arms, which is
  // what makes the intermediate arm worth its GPU time: `absent` is filed once
  // per tier a config switched off, so a tier-0 arm files one per message and a
  // tier-2-only arm two.
  expect(byFamily.get("baseline-b-tier0")!.degradedNotices["absent"]).toBe(3);
  expect(b.degradedNotices["absent"]).toBe(6);
});

test("refuses to show Approach B a document the loaded IR was not compiled from", async ({
  page,
}) => {
  // The pairing check, in the page rather than only in the driver. No model is
  // loaded and none is needed: this refusal happens when the arm is built.
  //
  // It is the check that makes a head-to-head a comparison of two METHODS.
  // Shown a different policy than the IR came from, B wins or loses for a
  // reason invisible in every number the run produces -- every record would
  // still validate, every gate would still compute.
  await openHarness(page);
  const message = await page.evaluate(async () => {
    // `semantic` is a hand-written fixture whose policyHash is the literal
    // "test-hash", so no document in this repository can pair with it.
    await window.__sih!.useIr("semantic");
    try {
      await window.__sih!.loadBaseline({ family: "baseline-b", policy: "p-fin" });
      return "no throw";
    } catch (cause) {
      return (cause as Error).message;
    }
  });
  expect(message).toContain("test-hash");
  expect(message).toContain("two policies rather than two methods");
});

test("publishes the policy document's digest, so a driver can check the page served the file it planned from", async ({
  page,
}) => {
  // The twin of `irHash`, and the failure it exists for is the same one:
  // `playwright.config.ts` reuses an existing dev server outside CI, so a page
  // built by another worktree would show B a different document while every
  // record looked correct. Unlike the IR, no record field carries a digest of
  // what B was shown -- `policyHash` on a row is the IR's field, the hash of
  // the document the IR was COMPILED from -- so this accessor is the only place
  // the two can be tied together.
  await openHarness(page);
  const pageHash = await page.evaluate(() => window.__sih!.policyDocHash("p-fin"));
  // Reproduced here from the file on disk, which is what makes this a check
  // rather than an echo.
  const { createHash } = await import("node:crypto");
  const fileHash = createHash("sha256")
    .update(readFileSync(POLICY_PATH, "utf8"), "utf8")
    .digest("hex");
  expect(pageHash).toBe(fileHash);
  // And it is the IR's own policyHash, which is the equality `loadBaseline`
  // enforces and `planBakeoff` refuses a pairing without.
  const irPolicyHash = (JSON.parse(readFileSync(IR_PATH, "utf8")) as { policyHash: string })
    .policyHash;
  expect(pageHash).toBe(irPolicyHash);
});
