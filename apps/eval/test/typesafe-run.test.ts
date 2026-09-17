import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { loadPolicyIr } from "@sih/core";
import { TypeSafeRecordSchema, TYPESAFE_MODEL } from "../src/driver/typesafe.js";
import { armFileName, gatesFileName, mapWithLimit, percentile, readApiKey, resolveTypeSafePlan, runTypeSafe, spendFileName } from "../src/driver/typesafe-run.js";

const IR = loadPolicyIr(readFileSync(new URL("../../../policies/compiled/p-fin.ir.json", import.meta.url), "utf8"));
const ITEMS = [
  { id: "i-1", text: "Meridian Capital is our client and the A/C No. 50100234567890 is theirs.", policy: "p-fin", gold: [{ start: 45, end: 59, text: "50100234567890", entityType: "bank-account-identifier", action: "block" }] },
  { id: "i-2", text: "the deploy pipeline is red again and nobody knows why", policy: "p-fin", gold: [] },
  { id: "i-3", text: "Sable Wood Partners asked for the PAN AAAPZ1234C on file", policy: "p-fin", gold: [{ start: 38, end: 48, text: "AAAPZ1234C", entityType: "in-pan", action: "block" }] },
];

function harness(env: Record<string, string | undefined>) {
  const files = new Map<string, string>();
  const logs: string[] = [];
  return {
    files,
    logs,
    deps: {
      env,
      now: () => 0,
      sleep: async () => {},
      fetchImpl: vi.fn(async () => new Response("should not be called in a dry run", { status: 500 })) as unknown as typeof fetch,
      loadIr: () => ({ ir: IR, irHash: "c".repeat(64) }),
      loadItems: () => ITEMS,
      gitProvenance: () => ({ gitSha: "d".repeat(40), gitDirty: true }),
      readApiKey: () => "unused-in-dry-run",
      writeFile: (name: string, contents: string) => void files.set(name, contents),
      fileExists: (name: string) => files.has(name),
      log: (line: string) => void logs.push(line),
    },
  };
}

describe("resolveTypeSafePlan", () => {
  it("requires a kebab-case run id, because the id names every artifact", () => {
    expect(() => resolveTypeSafePlan({})).toThrow(/SIH_TS_RUN_ID is required/);
    expect(() => resolveTypeSafePlan({ SIH_TS_RUN_ID: "Bad Id" })).toThrow(/kebab-case/);
    expect(resolveTypeSafePlan({ SIH_TS_RUN_ID: "ts-01" }).runId).toBe("ts-01");
  });

  it("defaults concurrency, attempts, candidates and the spend cap, and reads the dry-run flag", () => {
    const p = resolveTypeSafePlan({ SIH_TS_RUN_ID: "ts-01" });
    expect(p).toMatchObject({ arm: "ts-judgment", concurrency: 4, maxAttempts: 5, maxCandidates: 24, spendCapUsd: 2, dryRun: false, limit: undefined });
    expect(resolveTypeSafePlan({ SIH_TS_RUN_ID: "ts-01", SIH_TS_DRY_RUN: "1" }).dryRun).toBe(true);
    expect(() => resolveTypeSafePlan({ SIH_TS_RUN_ID: "ts-01", SIH_TS_CONCURRENCY: "0" })).toThrow(/positive/);
  });
});

describe("readApiKey", () => {
  it("prefers the environment and falls back to a .env line, quotes stripped", () => {
    expect(readApiKey({ TYPESAFE_API_KEY: "env-key" })).toBe("env-key");
    expect(() => readApiKey({}, "/nonexistent-repo-path")).toThrow(/no TYPESAFE_API_KEY/);
  });
});

describe("runTypeSafe, dry run", () => {
  it("writes schema-valid rows, gates and a ledger without a key or a network call", async () => {
    const h = harness({ SIH_TS_RUN_ID: "ts-dry", SIH_TS_DRY_RUN: "1" });
    await runTypeSafe(h.deps);
    const plan = resolveTypeSafePlan(h.deps.env);
    const rows = h.files.get(armFileName(plan))!.trim().split("\n").map((l) => TypeSafeRecordSchema.parse(JSON.parse(l)));
    expect(rows).toHaveLength(3);
    expect(h.deps.fetchImpl).not.toHaveBeenCalled();
    expect(rows.every((r) => r.dryRun && r.modelId === TYPESAFE_MODEL)).toBe(true);
    expect(rows.every((r) => r.predicateProbability !== null)).toBe(true);
    // Raw probabilities, never a thresholded boolean: the whole curve is recoverable later.
    expect(rows.every((r) => r.candidates.every((c) => c.probability === null || (c.probability >= 0 && c.probability <= 1)))).toBe(true);
    expect(rows[0]!.candidates.length).toBeGreaterThan(0);
    expect(rows[0]!.questionCount).toBe(rows[0]!.candidates.length + 1);

    const gates = JSON.parse(h.files.get(gatesFileName(plan))!) as Record<string, unknown>;
    expect(gates.items).toBe(3);
    expect(gates.answered).toBe(3);
    expect(gates.unresolvedQuotes).toBe(0);
    expect(Object.keys(gates.notMeasured as object)).toEqual(["ttftMs", "decodeTokPerSec", "reasoningTokens", "spanWisePredicate"]);

    const spend = JSON.parse(h.files.get(spendFileName(plan))!) as { costUsd: number; priceSource: string; dryRun: boolean };
    expect(spend.dryRun).toBe(true);
    expect(spend.priceSource).toContain("$42/1e9");
    expect(spend.costUsd).toBeGreaterThan(0);
  });

  it("refuses to overwrite an existing arm file", async () => {
    const h = harness({ SIH_TS_RUN_ID: "ts-dry", SIH_TS_DRY_RUN: "1" });
    await runTypeSafe(h.deps);
    await expect(runTypeSafe(h.deps)).rejects.toThrow(/exists; pick a new SIH_TS_RUN_ID/);
  });

  it("stops spending at the cap and records the remaining items as unanswered", async () => {
    const h = harness({ SIH_TS_RUN_ID: "ts-cap", SIH_TS_DRY_RUN: "1", SIH_TS_SPEND_CAP_USD: "0.0000001", SIH_TS_CONCURRENCY: "1" });
    await runTypeSafe(h.deps);
    const plan = resolveTypeSafePlan(h.deps.env);
    const rows = h.files.get(armFileName(plan))!.trim().split("\n").map((l) => TypeSafeRecordSchema.parse(JSON.parse(l)));
    expect(rows[0]!.predicateProbability).not.toBeNull();
    expect(rows.slice(1).every((r) => r.predicateProbability === null && r.error === "spend cap reached before this item")).toBe(true);
    expect(h.logs.some((l) => l.includes("SPEND CAP"))).toBe(true);
  });

  it("honours the item limit", async () => {
    const h = harness({ SIH_TS_RUN_ID: "ts-lim", SIH_TS_DRY_RUN: "1", SIH_TS_LIMIT: "2" });
    await runTypeSafe(h.deps);
    expect(h.files.get(armFileName(resolveTypeSafePlan(h.deps.env)))!.trim().split("\n")).toHaveLength(2);
  });
});

describe("helpers", () => {
  it("mapWithLimit preserves order under concurrency", async () => {
    const out = await mapWithLimit([1, 2, 3, 4, 5], 2, async (n) => n * 2);
    expect(out).toEqual([2, 4, 6, 8, 10]);
  });
  it("percentile is nearest-rank and null on an empty sample", () => {
    expect(percentile([], 0.5)).toBeNull();
    expect(percentile([10, 20, 30, 40], 0.5)).toBe(30);
  });
});
