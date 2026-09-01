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

  // WHAT ONE MESSAGE COSTS EACH METHOD, off the CALL ROWS -- what the engine
  // did, not what the plan expected.
  const callsPerItem = (family: ArmFamily): number[] =>
    rowsByFamily
      .get(family)!
      .map((r) => (r.baselineStats ?? r.tier2Stats)!.calls.length);
  expect(callsPerItem("baseline-b")).toEqual([1, 1, 1]);
  expect(callsPerItem("baseline-b-tier0")).toEqual([1, 1, 1]);
  // AND ONE ON THE COMPILED ARMS TOO, on THIS policy. `p-fin` declares exactly
  // one semantic predicate and its scope is "message", so the judge asks it
  // once about the whole message and enters no segment loop at all -- there is
  // no segment-scoped clause for a per-segment call to carry, and a call
  // carrying an empty predicate list is seconds of a 2 GB model spent on
  // nothing. A policy declaring both scopes costs `selectedSegments + 1`; this
  // one costs 1.
  //
  // What that replaced, in the units it was paid in: the arm used to spend one
  // call per SELECTED SEGMENT answering this message-scoped predicate in a
  // scope the policy never declared it in. On these three items escalation
  // selects 1, 2 and 2 segments without tier-0 priors and 1, 3 and 2 with them
  // -- MEASURED by calling core's `selectSegments` over this same slice, not
  // read off a run -- so the saving is 5 calls to 3 on `compiled-tier2-only`
  // and 6 to 3 on `compiled`, never more than 3 calls to 1 on any one message.
  // The plan's record of the pre-change run reports exactly those 5 and 6
  // answered calls (docs/superpowers/plans/2026-08-30-05-tier2-judge-baseline.md),
  // which is where the totals are checkable; the per-item counts above are the
  // half this file can compute for itself.
  expect(callsPerItem("compiled")).toEqual([1, 1, 1]);
  expect(callsPerItem("compiled-tier2-only")).toEqual([1, 1, 1]);
  // ATTRIBUTED, so "one call" cannot be a judge that did nothing: every one of
  // those calls was a whole-message call and every one of them was answered and
  // collected. `unitsJudged` counts SEGMENTS and is 0 for exactly that reason.
  for (const family of ["compiled", "compiled-tier2-only"] as const) {
    const g = byFamily.get(family)!;
    expect(g.ladder.messageScopeCalls, family).toBe(3);
    expect(g.ladder.messageScopeJudged, family).toBe(3);
    expect(g.ladder.messageScopeFailedClosed, family).toBe(0);
    expect(g.ladder.unitsJudged, family).toBe(0);
    // Approach B's single call IS its message call, so it has no separate
    // message-scope column and a 0 there would claim it made none.
    expect(byFamily.get("baseline-b")!.ladder.messageScopeCalls).toBeUndefined();
  }
  // AND THE MESSAGE BUDGET HELD. `p-fin` carries the compiler's real 5,000 ms
  // and the orchestrator arms ONE deadline over the whole `judge()` call, so a
  // second call per message is the thing that could have blown it. It did not,
  // on this machine and this model: no arm filed a budget notice. This is a
  // MEASUREMENT and not a guarantee -- it is one call per message here because
  // the policy is message-only, and a policy declaring both scopes would spend
  // this call and then start the segment loop inside the same 5,000 ms.
  for (const g of gates) {
    expect(g.degradedNotices["budget-exhausted"], g.arm).toBe(0);
    expect(g.degradedNotices["call-budget-exhausted"], g.arm).toBe(0);
  }

  // THE ASYMMETRY THAT DECIDES HOW TO READ THE LATENCY GATE, asserted rather
  // than described. B carries the whole 5,272-character policy document in
  // every prompt; the judge carries the passage its model was shown and no
  // policy text at all -- on THIS policy that passage is the whole message,
  // which is the same text B is shown minus the document. So B's prompts are
  // several times larger, and the difference is the document.
  //
  // `GATES.maxP95TtftMs` was derived at the judge's ~1.1 kB, which was a
  // ONE-SEGMENT prompt: the judge on this policy no longer builds one, so the
  // ratio below is not the ratio the ceiling was set at. A B arm failing that
  // gate is failing a threshold set for different work either way, which is why
  // `judgedUnit` and `promptTokens` are on the row beside the verdict.
  const b = byFamily.get("baseline-b")!;
  const compiledOnly = byFamily.get("compiled-tier2-only")!;
  expect(b.judgedUnit).toBe("message");
  // The family's PLANNED unit, and on this policy it describes work the arm no
  // longer does: every call asserted above was a whole-message call.
  // `judgedUnitChars` and `judgedUnitsPerItem` are built from that planned
  // segment distribution, so on a message-only policy the two of them describe
  // passages nobody was shown. `ladder.messageScopeCalls` on the same row is
  // what says so; closing the gap means making the judged unit a function of
  // the IR's scopes rather than of the family alone, which is recorded as a
  // carried risk in the README rather than done here.
  expect(compiledOnly.judgedUnit).toBe("segment");
  expect(b.promptTokens!.min).toBeGreaterThan(compiledOnly.promptTokens!.max);
  expect(b.gates.find((g) => g.gate === "p95-ttft")!.detail).toContain("judgedUnitChars");

  // THE SCOPE DIFFERENCE, CLOSED, and the count is the record of it. p-fin's
  // one semantic predicate is declared `scope: "message"`. Until the judge
  // honoured that it evaluated every predicate against a SEGMENT and reported
  // `scopesJudged: ["segment"]`, so the orchestrator filed one `scope-unjudged`
  // notice per message on BOTH compiled families -- 3 of 3 here -- while both
  // Approach-B families filed none, because B puts the whole message in one
  // call. That was read as B's structural advantage on the only real compiled
  // policy in this repository, and it was not: it was an unimplemented feature,
  // and every head-to-head number taken before this line changed was taken with
  // the compiled arm unable to answer the one predicate the policy declares.
  //
  // All four are 0 now, and only TWO of the four zeros moved. The compiled
  // arms' zero is the measurement: `unjudgedScopes(p-fin, ["message"])` returns
  // nothing because the judge named the scope it asked about, and it named
  // "segment" before this. The B arms' zero is a structural constant -- the
  // notice is filed inside core's `detect`, which a B arm never enters
  // (`createBaselineB` is a `Detector`, not a `SemanticJudge`), so a B row
  // reads 0 under any judge and is not evidence about what B asks its model.
  // Asserted on all four anyway: the B rows are the control that says the
  // column is 0 where nothing could have filed it.
  for (const g of gates) expect(g.degradedNotices["scope-unjudged"], g.arm).toBe(0);
  // And the 0s are not vacuous: this policy really does declare a predicate in
  // a scope `unjudgedScopes` enumerates, so a judge that stopped naming
  // "message" would put the 3s straight back. Read off the IR FILE, which is
  // the independent oracle -- reading it off the run would be the record
  // agreeing with itself.
  const declaredScopes = (
    JSON.parse(readFileSync(IR_PATH, "utf8")) as { semanticPredicates: { scope: string }[] }
  ).semanticPredicates.map((p) => p.scope);
  expect(declaredScopes).toEqual(["message"]);

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
