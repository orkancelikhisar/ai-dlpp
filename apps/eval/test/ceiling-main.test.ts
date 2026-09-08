import { describe, expect, it } from "vitest";
import type { PolicyIr } from "@sih/core";
import {
  CEILING_MODELS,
  SPEND_HARD_STOP_USD,
  SpendGuard,
  spendLedgerFileFor,
  type CeilingCall,
  type CeilingFamily,
  type CeilingModel,
  type CeilingRecord,
  type ClientDeps,
  type ItemInput,
  type RunItemOptions,
  type SpendEntry,
} from "../src/driver/ceiling.js";
import {
  LATER_PASS_GATE_USD,
  LOCAL_ARM_MAX_TOKENS,
  MAX_TOKENS,
  PROBE_ITEMS,
  armFileName,
  gitProvenance,
  laterPassGate,
  makeCallHook,
  resolveRunPlan,
  runCeiling,
  summarizeProbe,
  type CeilingRunDeps,
  type KeyState,
} from "../src/driver/ceiling-run.js";

/**
 * WHAT THIS FILE IS FOR, and why it is not `ceiling.test.ts`.
 *
 * `ceiling.test.ts` covers the units: `SpendGuard` arithmetic, `passRunIdFor`,
 * `spendLedgerFileFor`, `callChat`. All of them were green on 2026-09-08 while
 * this mutation, applied to the driver by exact-string replace with md5
 * confirmed changed, left `vitest run` at `Tests 791 passed (791)`, exit 0:
 *
 *     -  const ledgerFile = spendLedgerFileFor(runId, passStart);
 *     +  const ledgerFile = `ceiling-${runId}.spend.json`;
 *
 * That is byte-for-byte the defect that overwrote a 768-call spend ledger. The
 * helper had been extracted precisely so the mutation would be killable, and it
 * still was not, because a test that pins a helper cannot see whether anything
 * CALLS it.
 *
 * So every test below asserts a value at the point where it crosses a boundary:
 * the filename `writeSpend` is actually handed, the `maxTokens` that actually
 * reaches `runCeilingItem`, whether the guard is actually fed. Nothing here
 * re-derives an expectation from the thing under test -- the ledger names are
 * spelled out as literal strings, not computed from `spendLedgerFileFor`.
 */

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const MODEL: CeilingModel = {
  id: "vendor/model-x",
  provider: "PinnedProv",
  quantization: "fp8",
  pricePerMTokIn: 0.1,
  pricePerMTokOut: 0.2,
  hosting: "US",
};

function call(over: Partial<CeilingCall> = {}): CeilingCall {
  return {
    promptTokens: 100,
    completionTokens: 20,
    reasoningTokens: 0,
    ttftMs: 5,
    wallMs: 10,
    decodeTokPerSec: 4000,
    costUsd: 0.0001,
    finishReason: "stop",
    provider: "PinnedProv",
    modelId: "vendor/model-x",
    transport: "stream",
    repair: false,
    parse: "ok",
    retries: [],
    ...over,
  };
}

function record(over: Partial<CeilingRecord> = {}): CeilingRecord {
  return {
    schemaVersion: 1,
    runId: "ceiling-01",
    itemId: "item-0",
    policy: "p-fin",
    irHash: "a".repeat(64),
    policyHash: "b".repeat(64),
    arm: "ceiling-judge-model-x",
    family: "judge",
    requestedModelId: "vendor/model-x",
    modelId: "vendor/model-x",
    requestedProvider: "PinnedProv",
    provider: "PinnedProv",
    quantization: "fp8",
    outputMechanism: "provider-json-schema",
    thinkingRequested: "off",
    reasoningRequest: '{"enabled":false}',
    gitSha: "0".repeat(40),
    gitDirty: false,
    text: "some text",
    findings: [],
    gold: [],
    calls: [call()],
    parseFailures: 0,
    repairs: 0,
    unresolvedQuotes: 0,
    unresolvedMentions: 0,
    unknownLabels: 0,
    duplicatesDropped: 0,
    wholeClauseMentions: 0,
    wallMs: 12,
    error: null,
    ...over,
  } as CeilingRecord;
}

function items(n: number): ItemInput[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `item-${i}`,
    text: `text ${i}`,
    policy: "p-fin",
    gold: [],
  }));
}

// ---------------------------------------------------------------------------
// resolveRunPlan -- the ledger call site, and everything else the env decides
// ---------------------------------------------------------------------------

describe("resolveRunPlan", () => {
  const KEY = { OPENROUTER_API_KEY: "sk-not-a-real-key" };

  it("names the ledger after the FIRST PASS this process writes, not the base run id", () => {
    // THE mutant this file exists for. The two cases differ only in
    // SIH_CEILING_PASS_START, and the base run id is `ceiling-01` in both, so
    // `ceiling-${runId}.spend.json` gives the SAME name twice and this fails.
    // Spelled as literals rather than computed from `spendLedgerFileFor`: an
    // expectation derived from the helper cannot detect the helper not being
    // called.
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "ceiling-01" }).ledgerFile).toBe(
      "ceiling-ceiling-01.spend.json",
    );
    expect(
      resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "ceiling-01", SIH_CEILING_PASS_START: "2" }).ledgerFile,
    ).toBe("ceiling-ceiling-02.spend.json");
    expect(
      resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "ceiling-01", SIH_CEILING_PASS_START: "3" }).ledgerFile,
    ).toBe("ceiling-ceiling-03.spend.json");
  });

  it("gives a relaunch at pass 2 a ledger name that cannot collide with pass 1's", () => {
    // The live failure, stated as the property it violated: the pass-2 relaunch
    // wrote over `ceiling-ceiling-01.spend.json` -- 768 calls, $0.19269.
    const first = resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "ceiling-01" }).ledgerFile;
    const relaunch = resolveRunPlan({
      ...KEY,
      SIH_CEILING_RUN_ID: "ceiling-01",
      SIH_CEILING_PASS_START: "2",
    }).ledgerFile;
    expect(relaunch).not.toBe(first);
  });

  it("lists one pass run id per pass, offset by passStart", () => {
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_PASSES: "3" }).passRunIds).toEqual([
      "ceiling-01",
      "ceiling-02",
      "ceiling-03",
    ]);
    expect(
      resolveRunPlan({ ...KEY, SIH_CEILING_PASSES: "2", SIH_CEILING_PASS_START: "2" }).passRunIds,
    ).toEqual(["ceiling-02", "ceiling-03"]);
  });

  it("defaults the run id, passes and passStart", () => {
    const plan = resolveRunPlan(KEY);
    expect(plan.runId).toBe("ceiling-01");
    expect(plan.passes).toBe(1);
    expect(plan.passStart).toBe(1);
    expect(plan.probeOnly).toBe(false);
    expect(plan.limit).toBeUndefined();
    expect(plan.only).toBeUndefined();
  });

  it("reads a non-default run id rather than hardcoding ceiling-01", () => {
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "glmon" }).runId).toBe("glmon");
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "glmon" }).ledgerFile).toBe(
      "ceiling-glmon-01.spend.json",
    );
  });

  it("RENUMBERS a run id that already carries a pass suffix", () => {
    // MEASURED, and surprising enough to pin: `passRunIdFor` strips a trailing
    // `-\d+` and re-appends `passStart + pass - 1`, so a launch at
    // `SIH_CEILING_RUN_ID=glmon-07` with passStart 1 writes `glmon-01`, NOT
    // `glmon-07`. The base id survives on the ledger's `runId` field; only the
    // filenames are renumbered.
    const plan = resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "glmon-07" });
    expect(plan.runId).toBe("glmon-07");
    expect(plan.passRunIds).toEqual(["glmon-01"]);
    expect(plan.ledgerFile).toBe("ceiling-glmon-01.spend.json");
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "glmon-07", SIH_CEILING_PASS_START: "7" })
      .ledgerFile).toBe("ceiling-glmon-07.spend.json");
  });

  it("treats thinking as OFF unless the value is exactly \"on\"", () => {
    expect(resolveRunPlan(KEY).thinking).toBe("off");
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_THINKING: "off" }).thinking).toBe("off");
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_THINKING: "on" }).thinking).toBe("on");
    // Not a truthiness test: "true"/"1"/"ON" are the experiment's condition, off.
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_THINKING: "true" }).thinking).toBe("off");
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_THINKING: "ON" }).thinking).toBe("off");
  });

  it("restricts the slate to SIH_CEILING_MODELS and throws when nothing matches", () => {
    const wanted = CEILING_MODELS[1]!.id;
    const plan = resolveRunPlan({ ...KEY, SIH_CEILING_MODELS: ` ${wanted} ` });
    expect(plan.slate.map((m) => m.id)).toEqual([wanted]);
    expect(() => resolveRunPlan({ ...KEY, SIH_CEILING_MODELS: "nope/nope" })).toThrow(
      /matched no model/,
    );
  });

  it("treats an EMPTY SIH_CEILING_MODELS as the whole slate, not as a slate of none", () => {
    // `""` reaching the filter would produce a zero-length slate and throw --
    // the empty-string-is-absent rule is what stops an exported-but-unset shell
    // variable from aborting the run.
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_MODELS: "" }).slate).toHaveLength(
      CEILING_MODELS.length,
    );
  });

  it("applies pin overrides, which change what is ASKED and not what is recorded", () => {
    const target = CEILING_MODELS[0]!;
    const plan = resolveRunPlan({ ...KEY, SIH_CEILING_PIN: `${target.id}=NextBit` });
    const pinned = plan.slate.find((m) => m.id === target.id)!;
    expect(pinned.provider).toBe("NextBit");
    expect(pinned.provider).not.toBe(target.provider);
    // The quantization belonged to the OLD endpoint and must not be carried.
    expect(pinned.quantization).toBe("unknown");
  });

  it("carries the limit and the probe flag", () => {
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_LIMIT: "17" }).limit).toBe(17);
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_PROBE: "1" }).probeOnly).toBe(true);
    expect(resolveRunPlan({ ...KEY, SIH_CEILING_PROBE: "0" }).probeOnly).toBe(false);
  });

  it("never puts the API key on the plan", () => {
    // The plan is the natural thing to log or dump in a failure. The key lives
    // in one Authorization header and nowhere else, so it must not be reachable
    // from an object that travels.
    const plan = resolveRunPlan({ ...KEY, SIH_CEILING_RUN_ID: "ceiling-01" });
    expect(JSON.stringify(plan)).not.toContain("sk-not-a-real-key");
  });
});

// ---------------------------------------------------------------------------
// armFileName
// ---------------------------------------------------------------------------

describe("armFileName", () => {
  it("keys the arm file by the PASS run id, never the base run id", () => {
    expect(armFileName("ceiling-02", "judge", MODEL)).toBe(
      "ceiling-02.ceiling-judge-model-x.jsonl",
    );
    expect(armFileName("ceiling-02", "b", MODEL)).toBe("ceiling-02.ceiling-b-model-x.jsonl");
    expect(armFileName("ceiling-01", "judge", MODEL)).not.toBe(
      armFileName("ceiling-02", "judge", MODEL),
    );
  });
});

// ---------------------------------------------------------------------------
// laterPassGate
// ---------------------------------------------------------------------------

describe("laterPassGate", () => {
  it("stops passes 2+ AT the gate, and not one cent below it", () => {
    // The boundary is what distinguishes 2.5 from 25 and from 0.25. Asserting
    // only "a big number stops it" leaves both alive.
    expect(laterPassGate(1, 3, 2.5).stop).toBe(true);
    expect(laterPassGate(1, 3, 2.4999).stop).toBe(false);
    expect(laterPassGate(1, 3, 3).stop).toBe(true);
    expect(laterPassGate(1, 3, 2.6).stop).toBe(true);
    expect(LATER_PASS_GATE_USD).toBe(2.5);
  });

  it("only gates AFTER pass 1, and only when there are later passes to gate", () => {
    expect(laterPassGate(2, 3, 99).stop).toBe(false);
    expect(laterPassGate(1, 1, 99).stop).toBe(false);
  });

  it("reports the spend and the gate in the note", () => {
    const gate = laterPassGate(1, 3, 2.5);
    expect(gate.note).toBe("passes 2+ SKIPPED: pass 1 spent $2.5000, over the $2.50 gate");
    expect(laterPassGate(1, 3, 2.4999).note).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// makeCallHook -- that the guard is FED, in the right order, and re-checked
// ---------------------------------------------------------------------------

function hookHarness(guardOptions: Partial<ConstructorParameters<typeof SpendGuard>[0]> = {}) {
  const guard = new SpendGuard({
    hardStopUsd: 7,
    keyUsageAtStart: 0,
    keyLimit: 10,
    ...guardOptions,
  });
  const writes: { ledgerFile: string; calls: number; costUsd: number }[] = [];
  const keyReads: string[] = [];
  const hook = makeCallHook({
    guard,
    ledgerFile: "ceiling-ceiling-02.spend.json",
    apiKey: "sk-not-a-real-key",
    extra: { runId: "ceiling-01" },
    writeSpend: (ledgerFile, g) => {
      const snap = g.snapshot();
      writes.push({ ledgerFile, calls: snap.calls, costUsd: snap.costUsd });
    },
    readKeyState: async (apiKey) => {
      keyReads.push(apiKey);
      return { usage: 1.25, limit: 10 };
    },
  });
  return { guard, writes, keyReads, hook };
}

const ENTRY: SpendEntry = { costUsd: 0.01, estimateUsd: 0.02, model: "vendor/model-x", family: "judge" };

describe("makeCallHook", () => {
  it("feeds the guard and then writes the ledger, in that order", () => {
    // The ordering is the assertion. `writeSpend` running BEFORE `guard.record`
    // would write a ledger that is one call behind forever -- and on the last
    // call of a run, one call short. The spy snapshots the guard at write time,
    // so a reordered hook reports calls=0 here.
    const h = hookHarness();
    h.hook(ENTRY);
    expect(h.writes).toHaveLength(1);
    expect(h.writes[0]!.calls).toBe(1);
    expect(h.writes[0]!.costUsd).toBeCloseTo(0.01, 10);
  });

  it("writes to the ledger file it was given, on every call", () => {
    const h = hookHarness();
    h.hook(ENTRY);
    h.hook(ENTRY);
    h.hook(ENTRY);
    expect(h.writes.map((w) => w.ledgerFile)).toEqual([
      "ceiling-ceiling-02.spend.json",
      "ceiling-ceiling-02.spend.json",
      "ceiling-ceiling-02.spend.json",
    ]);
    expect(h.writes.map((w) => w.calls)).toEqual([1, 2, 3]);
  });

  it("re-reads the key endpoint on the 50th call and not before", async () => {
    // The guard's second, independent stop condition. Deleting the checkpoint
    // block leaves the run relying on summed `usage.cost` alone, which is the
    // number that can under-report.
    const h = hookHarness();
    for (let i = 0; i < 49; i++) h.hook(ENTRY);
    expect(h.keyReads).toHaveLength(0);
    h.hook(ENTRY);
    expect(h.keyReads).toEqual(["sk-not-a-real-key"]);
    // Fire-and-forget: the value lands on a later turn, and the hook must not
    // have awaited it.
    expect(h.guard.keyUsageLatest).toBeNull();
    await new Promise<void>((r) => setTimeout(r, 0));
    expect(h.guard.keyUsageLatest).toBe(1.25);
  });

  it("survives a failing key read rather than taking the run down with it", async () => {
    const guard = new SpendGuard({ hardStopUsd: 7, keyUsageAtStart: 0, keyLimit: 10, checkpointEvery: 1 });
    const hook = makeCallHook({
      guard,
      ledgerFile: "l.json",
      apiKey: "k",
      extra: {},
      writeSpend: () => undefined,
      readKeyState: async () => {
        throw new Error("/auth/key returned 503");
      },
    });
    expect(() => hook(ENTRY)).not.toThrow();
    await new Promise<void>((r) => setTimeout(r, 0));
    expect(guard.keyUsageLatest).toBeNull();
    expect(guard.calls).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// summarizeProbe
// ---------------------------------------------------------------------------

describe("summarizeProbe", () => {
  it("skips a model whose probe parses NOTHING and passes one that parses any", () => {
    const allBad = [
      record({ calls: [call({ parse: "malformed" })] }),
      record({ calls: [call({ parse: "schema" })] }),
      record({ error: "timeout", calls: [] }),
    ];
    expect(summarizeProbe(MODEL, "judge", allBad).verdict).toBe("skip");
    expect(summarizeProbe(MODEL, "judge", allBad).parsed).toBe(0);

    const oneGood = [...allBad, record({ calls: [call({ parse: "ok" })] })];
    expect(summarizeProbe(MODEL, "judge", oneGood).verdict).toBe("pass");
    expect(summarizeProbe(MODEL, "judge", oneGood).parsed).toBe(1);
  });

  it("does not count a parsed call on a record that errored", () => {
    const errored = [record({ error: "aborted after repair", calls: [call({ parse: "ok" })] })];
    expect(summarizeProbe(MODEL, "judge", errored).parsed).toBe(0);
    expect(summarizeProbe(MODEL, "judge", errored).verdict).toBe("skip");
    expect(summarizeProbe(MODEL, "judge", errored).errors).toEqual(["aborted after repair"]);
  });

  it("reports providerMatchesPin as FALSE when the answering provider is not the pin", () => {
    // The pin is a request; this compares it with the fact. A hardcoded `true`
    // would make the record's own provider-pin claim unfalsifiable.
    const off = [record({ calls: [call({ provider: "SomeoneElse" })] })];
    expect(summarizeProbe(MODEL, "judge", off).providerMatchesPin).toBe(false);
    expect(summarizeProbe(MODEL, "judge", off).providersSeen).toEqual(["SomeoneElse"]);

    const mixed = [record({ calls: [call({ provider: "PinnedProv" }), call({ provider: "Other" })] })];
    expect(summarizeProbe(MODEL, "judge", mixed).providerMatchesPin).toBe(false);

    const onPin = [record({ calls: [call({ provider: "PinnedProv" })] })];
    expect(summarizeProbe(MODEL, "judge", onPin).providerMatchesPin).toBe(true);
  });

  it("reports providerMatchesPin as FALSE when NO provider answered at all", () => {
    // `[].every(...)` is true. Without the length guard, a probe where every
    // call reported a null provider would claim the pin was honoured.
    const none = [record({ calls: [call({ provider: null })] })];
    expect(summarizeProbe(MODEL, "judge", none).providersSeen).toEqual([]);
    expect(summarizeProbe(MODEL, "judge", none).providerMatchesPin).toBe(false);
    expect(summarizeProbe(MODEL, "judge", []).providerMatchesPin).toBe(false);
  });

  it("counts only MEASURED zeros as reasoningAllZero -- a null is not a zero", () => {
    // The distinction the whole thinking-off claim rests on: null means "the
    // provider reported no usage", which is not evidence that thinking was off.
    const nulls = [record({ calls: [call({ reasoningTokens: null }), call({ reasoningTokens: null })] })];
    expect(summarizeProbe(MODEL, "judge", nulls).reasoningTokens).toEqual([null, null]);
    expect(summarizeProbe(MODEL, "judge", nulls).reasoningAllZero).toBe(false);

    const mixed = [record({ calls: [call({ reasoningTokens: 0 }), call({ reasoningTokens: null })] })];
    expect(summarizeProbe(MODEL, "judge", mixed).reasoningAllZero).toBe(false);

    const zeros = [record({ calls: [call({ reasoningTokens: 0 }), call({ reasoningTokens: 0 })] })];
    expect(summarizeProbe(MODEL, "judge", zeros).reasoningAllZero).toBe(true);

    const some = [record({ calls: [call({ reasoningTokens: 0 }), call({ reasoningTokens: 41 })] })];
    expect(summarizeProbe(MODEL, "judge", some).reasoningAllZero).toBe(false);

    expect(summarizeProbe(MODEL, "judge", []).reasoningAllZero).toBe(false);
  });

  it("carries the latency columns per CALL and the attempt count per RECORD", () => {
    const recs = [
      record({ calls: [call({ ttftMs: 5, wallMs: 10 }), call({ ttftMs: null, wallMs: 20 })] }),
      record({ calls: [call({ ttftMs: 7, wallMs: 30 })] }),
    ];
    const probe = summarizeProbe(MODEL, "b", recs);
    expect(probe.attempted).toBe(2);
    expect(probe.ttftMs).toEqual([5, null, 7]);
    expect(probe.wallMs).toEqual([10, 20, 30]);
    expect(probe.model).toBe("vendor/model-x");
    expect(probe.family).toBe("b");
  });
});

// ---------------------------------------------------------------------------
// gitProvenance
// ---------------------------------------------------------------------------

describe("gitProvenance", () => {
  it("reports the tree as dirty when `git status --porcelain` prints anything", () => {
    const dirty = gitProvenance(
      (file, args) => (args[0] === "rev-parse" ? `${"a".repeat(40)}\n` : " M apps/eval/x.ts\n"),
      "/repo",
    );
    expect(dirty).toEqual({ gitSha: "a".repeat(40), gitDirty: true });
  });

  it("reports the tree as clean only on EMPTY porcelain output, whitespace included", () => {
    expect(gitProvenance(() => "", "/repo").gitDirty).toBe(false);
    expect(gitProvenance((f, a) => (a[0] === "rev-parse" ? "sha\n" : "\n  \n"), "/repo").gitDirty).toBe(
      false,
    );
  });

  it("asks git for HEAD and for the porcelain status, in the repo root", () => {
    const seen: { args: readonly string[]; cwd: string }[] = [];
    gitProvenance((file, args, options) => {
      seen.push({ args, cwd: options.cwd });
      return file === "git" && args[0] === "rev-parse" ? "sha\n" : "";
    }, "/some/repo");
    expect(seen.map((s) => s.args.join(" "))).toEqual(["rev-parse HEAD", "status --porcelain"]);
    expect(seen.every((s) => s.cwd === "/some/repo")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// runCeiling -- the WIRING. Every assertion below is a value observed at the
// point it crosses out of the orchestration.
// ---------------------------------------------------------------------------

interface Seen {
  spend: { ledgerFile: string; calls: number; keyUsageLatest: number | null; hardStopUsd: number; keyLimit: number | null; extra: Record<string, unknown> }[];
  arms: { fileName: string; rows: number }[];
  runItem: RunItemOptions[];
  logs: string[];
  keyReads: number;
}

function ceilingHarness(
  env: Record<string, string | undefined>,
  options: {
    itemCount?: number;
    keyUsage?: number[];
    keyLimit?: number | null;
    costPerCall?: number;
  } = {},
) {
  const itemCount = options.itemCount ?? 4;
  const keyUsage = options.keyUsage ?? [0];
  const costPerCall = options.costPerCall ?? 0.0001;
  const seen: Seen = { spend: [], arms: [], runItem: [], logs: [], keyReads: 0 };
  let clock = 0;

  const deps: Partial<CeilingRunDeps> = {
    env: { OPENROUTER_API_KEY: "sk-not-a-real-key", ...env },
    gitProvenance: () => ({ gitSha: "c".repeat(40), gitDirty: true }),
    loadIr: () => ({ ir: {} as PolicyIr, irHash: "d".repeat(64) }),
    loadPolicyText: () => "the policy text",
    loadItems: () => items(itemCount),
    readKeyState: async (): Promise<KeyState> => {
      const usage = keyUsage[Math.min(seen.keyReads, keyUsage.length - 1)]!;
      seen.keyReads += 1;
      return { usage, limit: options.keyLimit === undefined ? 10 : options.keyLimit };
    },
    runItem: async (opts: RunItemOptions) => {
      seen.runItem.push(opts);
      opts.onCall?.({
        costUsd: costPerCall,
        estimateUsd: costPerCall / 2,
        model: opts.model.id,
        family: opts.family,
      });
      return record({ runId: opts.runId, itemId: opts.item.id, family: opts.family });
    },
    writeSpend: (ledgerFile, guard, extra) => {
      const s = guard.snapshot();
      seen.spend.push({
        ledgerFile,
        calls: s.calls,
        keyUsageLatest: s.keyUsageLatest,
        hardStopUsd: s.hardStopUsd,
        keyLimit: s.keyLimit,
        extra,
      });
    },
    writeArm: (fileName, jsonl) => {
      seen.arms.push({ fileName, rows: jsonl.trimEnd().split("\n").length });
    },
    clientDeps: (apiKey) => ({ fetch: (async () => new Response("")) as unknown as ClientDeps["fetch"], now: () => 0, sleep: async () => undefined, apiKey }),
    log: (line) => {
      seen.logs.push(line);
    },
    now: () => (clock += 1),
  };
  return { seen, deps };
}

const ONE_MODEL = CEILING_MODELS[0]!.id;

describe("runCeiling wiring", () => {
  it("hands EVERY ledger write the passStart-keyed filename", async () => {
    // The call site, not the helper. The base run id is `ceiling-01`, so a
    // reverted call site writes `ceiling-ceiling-01.spend.json` -- which is
    // pass 1's ledger, the file the live defect destroyed.
    const h = ceilingHarness({
      SIH_CEILING_RUN_ID: "ceiling-01",
      SIH_CEILING_PASS_START: "2",
      SIH_CEILING_MODELS: ONE_MODEL,
    });
    await runCeiling(h.deps);
    expect(h.seen.spend.length).toBeGreaterThan(0);
    expect([...new Set(h.seen.spend.map((s) => s.ledgerFile))]).toEqual([
      "ceiling-ceiling-02.spend.json",
    ]);
  });

  it("names arm files by the PASS run id, so a relaunch cannot overwrite pass 1", async () => {
    const h = ceilingHarness({
      SIH_CEILING_RUN_ID: "ceiling-01",
      SIH_CEILING_PASS_START: "2",
      SIH_CEILING_MODELS: ONE_MODEL,
    });
    await runCeiling(h.deps);
    const short = ONE_MODEL.slice(ONE_MODEL.indexOf("/") + 1);
    expect(h.seen.arms.map((a) => a.fileName)).toEqual([
      `ceiling-02.ceiling-judge-${short}.jsonl`,
      `ceiling-02.ceiling-b-${short}.jsonl`,
    ]);
    expect(h.seen.arms.every((a) => a.rows === 4)).toBe(true);
    // And the rows themselves are stamped with the pass id, not the base id.
    expect([...new Set(h.seen.runItem.filter((o) => !o.runId.endsWith("-probe")).map((o) => o.runId))]).toEqual([
      "ceiling-02",
    ]);
  });

  it("FEEDS the spend guard: every item run reports its call into the ledger", async () => {
    // Nothing else in the suite proves the guard is connected to anything. Drop
    // `onCall` from the item options and the run spends the key with a ledger
    // that reads calls=0 the whole way.
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL }, { itemCount: 5 });
    await runCeiling(h.deps);
    // probe: PROBE_ITEMS per family; passes: 5 items per family.
    const expected = 2 * PROBE_ITEMS + 2 * 5;
    expect(h.seen.runItem).toHaveLength(expected);
    expect(h.seen.runItem.every((o) => typeof o.onCall === "function")).toBe(true);
    expect(h.seen.spend.at(-1)!.calls).toBe(expected);
  });

  it("uses the RUN's hard stop, never the key's own limit", async () => {
    // A key with a $42 limit must not license a $42 run. 42 is chosen so the
    // two numbers cannot be confused.
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL }, { keyLimit: 42 });
    await runCeiling(h.deps);
    expect(h.seen.spend.at(-1)!.hardStopUsd).toBe(SPEND_HARD_STOP_USD);
    expect(h.seen.spend.at(-1)!.hardStopUsd).toBe(7);
    expect(h.seen.spend.at(-1)!.keyLimit).toBe(42);
  });

  it("sends the experiment's token ceiling and records the local arms' beside it", async () => {
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL });
    await runCeiling(h.deps);
    expect(h.seen.runItem.every((o) => o.maxTokens === 600)).toBe(true);
    expect(h.seen.runItem.every((o) => o.localArmMaxTokens === 512)).toBe(true);
    // The asymmetry is the point: collapsing one onto the other erases the 88
    // tokens this arm ran with in its favour.
    expect(MAX_TOKENS).not.toBe(LOCAL_ARM_MAX_TOKENS);
    expect(h.seen.runItem.every((o) => o.maxTokens !== o.localArmMaxTokens)).toBe(true);
  });

  it("sends the OVERRIDDEN token ceiling all the way to every call, not just to the plan", async () => {
    // Sec 7.2: a thinking-ON phase cannot be measured at 600 -- GLM truncated 22%
    // of its answers there, and all eight of its missed gold positives were on
    // truncated calls. The override exists for that run.
    //
    // Asserted at `runItem`, NOT at `resolveRunPlan`, deliberately. A mutant that
    // resolves the value correctly and then passes the constant at the call site
    // is exactly the shape that survived three fixes of the ledger defect --
    // see standing-conventions Sec 10.
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL, SIH_CEILING_MAX_TOKENS: "8192" });
    await runCeiling(h.deps);
    expect(h.seen.runItem.length).toBeGreaterThan(0);
    expect(h.seen.runItem.every((o) => o.maxTokens === 8192)).toBe(true);
    // The local-arm figure is a fact about the browser arms and must NOT move with it.
    expect(h.seen.runItem.every((o) => o.localArmMaxTokens === 512)).toBe(true);
    expect(h.seen.logs.some((l) => l.includes("max_tokens=8192"))).toBe(true);
  });

  it("REFUSES a malformed token ceiling rather than sending max_tokens: null", async () => {
    // Number("8k") is NaN, which serialises to null and makes every arm run at
    // whatever default its provider happens to use -- unrecorded, and different
    // per provider. Failing loudly is the only safe behaviour.
    for (const bad of ["8k", "", "0", "-1", "1.5", "abc"]) {
      const bag = { SIH_CEILING_MODELS: ONE_MODEL, SIH_CEILING_MAX_TOKENS: bad };
      if (bad === "") {
        // An empty value reads as "unset" and must fall back to the default.
        expect(resolveRunPlan({ ...bag }).maxTokens).toBe(600);
        continue;
      }
      expect(() => resolveRunPlan({ ...bag })).toThrow(/SIH_CEILING_MAX_TOKENS/);
    }
  });

  it("requests thinking OFF by default and ON only when asked", async () => {
    const off = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL });
    await runCeiling(off.deps);
    expect(off.seen.runItem.every((o) => o.thinking === "off")).toBe(true);

    const on = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL, SIH_CEILING_THINKING: "on" });
    await runCeiling(on.deps);
    expect(on.seen.runItem.every((o) => o.thinking === "on")).toBe(true);
  });

  it("passes the git provenance and the destination provider onto every item", async () => {
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL });
    await runCeiling(h.deps);
    expect(h.seen.runItem.every((o) => o.gitSha === "c".repeat(40) && o.gitDirty === true)).toBe(true);
    expect(h.seen.runItem.every((o) => o.destinationProvider === "claude")).toBe(true);
    expect(h.seen.runItem.every((o) => o.irHash === "d".repeat(64))).toBe(true);
    expect(h.seen.runItem.every((o) => o.policyText === "the policy text")).toBe(true);
  });

  it("writes the closing ledger with the key's END state, checkpointed", async () => {
    // Deleting the final write leaves the ledger reporting whatever the last
    // per-call write happened to hold and no `keyAtEnd` at all.
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL }, { keyUsage: [1.5, 2.25] });
    await runCeiling(h.deps);
    const last = h.seen.spend.at(-1)!;
    expect(last.extra.keyAtEnd).toEqual({ usage: 2.25, limit: 10 });
    expect(last.keyUsageLatest).toBe(2.25);
    // ...and no earlier write claimed to be the closing one.
    expect(h.seen.spend.slice(0, -1).every((s) => s.extra.keyAtEnd === undefined)).toBe(true);
  });

  it("stops mid-arm when the guard trips, and records where it stopped", async () => {
    // Also the only coverage `mapWithConcurrency`'s stop predicate has: ignore
    // it and all 12 items run, stoppedAtIndex stays null, and this fails.
    // 6.0 start + 6 probe calls * 0.1 = 6.6; the arm trips 4 calls later.
    const h = ceilingHarness(
      { SIH_CEILING_MODELS: ONE_MODEL },
      { itemCount: 12, keyUsage: [6.0], costPerCall: 0.1 },
    );
    await runCeiling(h.deps);
    const notes = h.seen.spend.at(-1)!.extra.stopNotes as string[];
    expect(notes.some((n) => /TRIPPED mid-arm/.test(n))).toBe(true);
    const midArm = notes.find((n) => /TRIPPED mid-arm/.test(n))!;
    expect(midArm).toMatch(/stopped at item index \d+ \(item-\d+\), \d+ of 12 items written/);
    // Fewer than the full arm's calls were made.
    const armCalls = h.seen.runItem.filter((o) => !o.runId.endsWith("-probe")).length;
    expect(armCalls).toBeGreaterThan(0);
    expect(armCalls).toBeLessThan(12);
    expect(h.seen.arms[0]!.rows).toBeLessThan(12);
  });

  it("BREAKS out of the family loop once tripped instead of running the next arm", async () => {
    // Without the `break`, the note is pushed and the arm runs anyway: an
    // `[arm] ... 0 rows` line appears and a second TRIPPED-before note follows.
    const h = ceilingHarness(
      { SIH_CEILING_MODELS: ONE_MODEL },
      { itemCount: 12, keyUsage: [6.0], costPerCall: 0.1 },
    );
    await runCeiling(h.deps);
    const notes = h.seen.spend.at(-1)!.extra.stopNotes as string[];
    expect(notes.filter((n) => /TRIPPED before/.test(n))).toHaveLength(1);
    expect(h.seen.logs.filter((l) => l.startsWith("[arm] "))).toHaveLength(1);
    expect(h.seen.arms).toHaveLength(1);
  });

  it("skips an arm whose probe parsed nothing, without spending a pass on it", async () => {
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL }, { itemCount: 4 });
    await runCeiling({
      ...h.deps,
      runItem: async (opts: RunItemOptions) => {
        h.seen.runItem.push(opts);
        opts.onCall?.({ costUsd: 0.0001, estimateUsd: 0.0001, model: opts.model.id, family: opts.family });
        return record({ runId: opts.runId, itemId: opts.item.id, calls: [call({ parse: "malformed" })] });
      },
    });
    const notes = h.seen.spend.at(-1)!.extra.stopNotes as string[];
    expect(notes.filter((n) => /SKIPPED: probe parsed 0 of 3/.test(n))).toHaveLength(2);
    expect(h.seen.arms).toEqual([]);
    // Only the probe ran.
    expect(h.seen.runItem).toHaveLength(2 * PROBE_ITEMS);
  });

  it("abandons passes 2+ when pass 1 already spent past the gate", async () => {
    // 3 passes requested; pass 1's 2 arms x 4 items x $0.35 plus the probe
    // carries the key past $2.50.
    const h = ceilingHarness(
      { SIH_CEILING_MODELS: ONE_MODEL, SIH_CEILING_PASSES: "3" },
      { itemCount: 4, costPerCall: 0.35 },
    );
    await runCeiling(h.deps);
    const notes = h.seen.spend.at(-1)!.extra.stopNotes as string[];
    expect(notes.some((n) => /passes 2\+ SKIPPED/.test(n))).toBe(true);
    expect([...new Set(h.seen.arms.map((a) => a.fileName.split(".")[0]))]).toEqual(["ceiling-01"]);
  });

  it("runs all three passes when pass 1 stayed under the gate", async () => {
    // The other side of the same gate: raising it to 25 must not be invisible.
    const h = ceilingHarness(
      { SIH_CEILING_MODELS: ONE_MODEL, SIH_CEILING_PASSES: "3" },
      { itemCount: 4, costPerCall: 0.0001 },
    );
    await runCeiling(h.deps);
    const notes = h.seen.spend.at(-1)!.extra.stopNotes as string[];
    expect(notes.some((n) => /passes 2\+ SKIPPED/.test(n))).toBe(false);
    expect([...new Set(h.seen.arms.map((a) => a.fileName.split(".")[0]))]).toEqual([
      "ceiling-01",
      "ceiling-02",
      "ceiling-03",
    ]);
  });

  it("writes the probe ledger and stops when SIH_CEILING_PROBE is set", async () => {
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL, SIH_CEILING_PROBE: "1" });
    await runCeiling(h.deps);
    expect(h.seen.arms).toEqual([]);
    expect(h.seen.runItem).toHaveLength(2 * PROBE_ITEMS);
    expect(h.seen.runItem.every((o) => o.runId === "ceiling-01-probe")).toBe(true);
    expect(h.seen.spend.at(-1)!.extra.keyAtEnd).toBeUndefined();
    // Exactly one closing key read at start; the run must not have paid for a
    // second round trip it does not use.
    expect(h.seen.keyReads).toBe(1);
  });

  it("caps items with SIH_CEILING_LIMIT", async () => {
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL, SIH_CEILING_LIMIT: "2" }, { itemCount: 30 });
    await runCeiling(h.deps);
    expect(h.seen.arms.every((a) => a.rows === 2)).toBe(true);
    expect(h.seen.runItem.filter((o) => !o.runId.endsWith("-probe"))).toHaveLength(4);
  });

  it("refuses to start without a key, before any network or file work", async () => {
    const h = ceilingHarness({});
    await expect(
      runCeiling({ ...h.deps, env: { SIH_CEILING_MODELS: ONE_MODEL } }),
    ).rejects.toThrow(/OPENROUTER_API_KEY is not set/);
    expect(h.seen.keyReads).toBe(0);
    expect(h.seen.runItem).toEqual([]);
    expect(h.seen.spend).toEqual([]);
  });

  it("hands the key to the client deps and to the key endpoint, and nowhere else", async () => {
    const h = ceilingHarness({ SIH_CEILING_MODELS: ONE_MODEL });
    await runCeiling(h.deps);
    expect(h.seen.runItem.every((o) => (o.deps as ClientDeps).apiKey === "sk-not-a-real-key")).toBe(true);
    // The key never reaches a log line or a ledger payload.
    expect(h.seen.logs.some((l) => l.includes("sk-not-a-real-key"))).toBe(false);
    expect(JSON.stringify(h.seen.spend).includes("sk-not-a-real-key")).toBe(false);
  });

  it("announces the ledger path it will actually write", async () => {
    const h = ceilingHarness({
      SIH_CEILING_RUN_ID: "ceiling-01",
      SIH_CEILING_PASS_START: "2",
      SIH_CEILING_MODELS: ONE_MODEL,
    });
    await runCeiling(h.deps);
    // A log line that names a different file from the one written would have
    // been the cheapest possible warning the live defect could have given.
    const announced = h.seen.logs.find((l) => l.startsWith("[ceiling] ledger:"))!;
    expect(announced).toBe("[ceiling] ledger: runs/ceiling-ceiling-02.spend.json");
    expect(announced).toContain(h.seen.spend[0]!.ledgerFile);
    expect(h.seen.logs).toContain("[ceiling] will write passes: ceiling-02");
  });
});

// A guard on the fixtures themselves: `spendLedgerFileFor` is the helper this
// file deliberately does NOT derive its expectations from, so pin the two
// against each other once, here, rather than inside the tests above.
describe("the literal ledger names above are the helper's own", () => {
  it("agrees with spendLedgerFileFor", () => {
    expect(spendLedgerFileFor("ceiling-01", 1)).toBe("ceiling-ceiling-01.spend.json");
    expect(spendLedgerFileFor("ceiling-01", 2)).toBe("ceiling-ceiling-02.spend.json");
  });
});

// Unused-import guard for the type-only fixtures above.
export type { CeilingFamily };
