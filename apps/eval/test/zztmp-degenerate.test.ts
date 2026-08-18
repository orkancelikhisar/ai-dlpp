import { it } from "vitest";
import { RunRecordSchema } from "../src/driver/record.js";
import { CorpusItemSchema } from "../src/driver/corpus.js";
it("probe", () => {
  const base = {
    schemaVersion: 1 as const, runId: "r1", itemId: "a", policy: "p-fin",
    policyHash: "0".repeat(64), arm: "t0", backend: "wasm" as const, provider: "claude",
    text: "hello world", findings: [], gold: [], timings: { tier0Ms: 0 }, error: null,
  };
  const cases = [
    ["{start:999, end:1, text:''}", { start: 999, end: 1 }],
    ["{start:5,   end:5, text:''}", { start: 5, end: 5 }],
    ["{start:8,   end:3, text:''}", { start: 8, end: 3 }],
  ] as const;
  for (const [label, o] of cases) {
    const f = { ...o, text: "", entityType: "in-pan", severity: "high", tier: 0, source: "r", confidence: 0.9, action: "block" };
    const g = { ...o, text: "", entityType: "in-pan", action: "block" };
    const fOk = RunRecordSchema.safeParse({ ...base, findings: [f] }).success;
    const gOk = CorpusItemSchema.safeParse({ id: "a", text: "hello world", policy: "p", gold: [g] }).success;
    console.log(`${label}  findings: ${fOk ? "ACCEPTED" : "rejected"}   gold: ${gOk ? "ACCEPTED" : "rejected"}`);
  }
});
