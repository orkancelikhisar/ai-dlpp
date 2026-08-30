import { prebuiltAppConfig } from "@mlc-ai/web-llm";
import { describe, expect, it } from "vitest";
import { TIER2_MODELS, resolveTier2Config, DEFAULT_TIER2_CONFIG } from "../src/manifest.js";

describe("TIER2_MODELS", () => {
  it("lists only models measured to run, cheapest first", () => {
    // Order is load-bearing: the bake-off runs cheapest-first because decode
    // rate falls with context, so an expensive arm that will be killed on
    // throughput should not be paid for before a cheap one has been measured.
    expect(TIER2_MODELS.map((m) => m.id)).toEqual([
      "Qwen3.5-2B-q4f16_1-MLC",
      "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
      "Qwen3-4B-q4f16_1-MLC",
      "Phi-4-mini-instruct-q4f16_1-MLC",
    ]);
  });

  it("does not list gemma3-4b, which has no compiled WebGPU lib", () => {
    // Weights exist in the registry; no lib is compiled and it is absent from
    // prebuiltAppConfig, so the shipped runtime cannot execute it. Listing it
    // would fail at load with an error that reads like a bug in our code.
    expect(TIER2_MODELS.map((m) => m.id).join(" ")).not.toMatch(/gemma/i);
  });

  it("records the measured cost of each model, not a guess", () => {
    for (const m of TIER2_MODELS) {
      expect(m.sizeMb, m.id).toBeGreaterThan(0);
      expect(m.decodeTokPerSec, m.id).toBeGreaterThan(0);
      expect(typeof m.hasThinkingMode, m.id).toBe("boolean");
    }
  });

  it("is ordered by non-decreasing download size, which is what cheapest-first means", () => {
    // The list-equality test above pins today's order but not the rule behind
    // it, so an edit that reorders the slate and updates that literal would
    // pass. This asserts the invariant a future arm has to satisfy too.
    const sizes = TIER2_MODELS.map((m) => m.sizeMb);
    expect(sizes).toEqual([...sizes].sort((a, b) => a - b));
  });

  it("names model ids the shipped web-llm build can actually resolve", () => {
    // The id is not free text: MLCEngine looks it up in prebuiltAppConfig and
    // throws at load if it is absent. A truncated or stale id therefore fails
    // exactly like the dropped gemma3-4b would -- with an error that reads as
    // a bug in our code. Cross-check against the pinned library, not a memory
    // of it. (This caught "Ministral-3-3B-Instruct-2512-BF16", which is a
    // prefix of the real id and resolves to nothing.)
    for (const m of TIER2_MODELS) {
      const record = prebuiltAppConfig.model_list.find((r) => r.model_id === m.id);
      expect(record, `${m.id} is not in prebuiltAppConfig`).toBeDefined();
      // A record with no compiled lib is the gemma3-4b failure mode exactly.
      expect(record!.model_lib, `${m.id} has no compiled WebGPU lib`).toBeTruthy();
      expect(Math.round(record!.vram_required_MB!), m.id).toBe(m.sizeMb);
    }
  });
});

describe("resolveTier2Config", () => {
  it("defaults to the fastest measured model with a lifted context window", () => {
    const c = resolveTier2Config({});
    expect(c.modelId).toBe("Qwen3.5-2B-q4f16_1-MLC");
    // 4096 is a WebLLM override, not a model limit. Measured: 8192 loads in
    // 1.7 s warm and prefills 4,360 tokens at 452 tok/s. Approach B needs the
    // headroom and both arms must have the same window or the comparison is
    // measuring context, not method.
    expect(c.contextWindowSize).toBe(8192);
    expect(c.temperature).toBe(0);
  });

  it("rejects a model id that is not in the manifest, naming what is", () => {
    expect(() => resolveTier2Config({ modelId: "gemma3-4b-it" })).toThrow(/Qwen3.5-2B/);
  });

  it("rejects a non-zero temperature rather than silently allowing it", () => {
    // The bake-off compares models. A non-zero temperature makes a rerun
    // disagree with itself and the comparison stops being one.
    expect(() => resolveTier2Config({ temperature: 0.7 })).toThrow(/temperature/);
    // Not just "too hot": a negative or NaN temperature is equally not a
    // reproducible run, and `> 0` would let both through.
    expect(() => resolveTier2Config({ temperature: -1 })).toThrow(/temperature/);
    expect(() => resolveTier2Config({ temperature: Number.NaN })).toThrow(/temperature/);
  });

  it("treats an explicitly-undefined key as absent, not as a clobber", () => {
    // This is the natural shape of a CLI or env-var reader: every option is
    // present as a key and undefined when unset. Plain object spread copies
    // the undefined over the default, so `{ modelId: undefined }` would make
    // resolve() throw "unknown tier-2 model undefined" -- i.e. omitting an
    // option is punished harder than passing a wrong one. Plan 4 shipped
    // exactly this and it took a review round to find. None of the tests
    // above touch it: they either pass no keys at all or pass real values.
    const fromCli = {
      modelId: undefined,
      contextWindowSize: undefined,
      temperature: undefined,
      maxTokens: undefined,
    };
    expect(resolveTier2Config(fromCli)).toEqual(DEFAULT_TIER2_CONFIG);
    // and one key at a time, so a filter that only works on a full sweep fails
    expect(resolveTier2Config({ modelId: undefined }).modelId).toBe(DEFAULT_TIER2_CONFIG.modelId);
    expect(resolveTier2Config({ temperature: undefined }).temperature).toBe(0);
    expect(resolveTier2Config({ maxTokens: undefined }).maxTokens).toBe(DEFAULT_TIER2_CONFIG.maxTokens);
    expect(resolveTier2Config({ contextWindowSize: undefined }).contextWindowSize).toBe(8192);
  });

  it("still applies a defined override, so the undefined filter is not a mute button", () => {
    const c = resolveTier2Config({
      modelId: "Phi-4-mini-instruct-q4f16_1-MLC",
      contextWindowSize: 4096,
      maxTokens: 128,
    });
    expect(c).toEqual({
      modelId: "Phi-4-mini-instruct-q4f16_1-MLC",
      contextWindowSize: 4096,
      temperature: 0,
      maxTokens: 128,
    });
  });

  it("rejects a maxTokens that is not a positive integer", () => {
    // max_tokens goes straight to the engine; 0 or a fraction is a silent
    // empty completion rather than a loud failure, which in the bake-off
    // reads as "the model found nothing".
    expect(() => resolveTier2Config({ maxTokens: 0 })).toThrow(/maxTokens/);
    expect(() => resolveTier2Config({ maxTokens: -1 })).toThrow(/maxTokens/);
    expect(() => resolveTier2Config({ maxTokens: 1.5 })).toThrow(/maxTokens/);
  });

  it("does not hand back a reference a caller can mutate into the defaults", () => {
    const c = resolveTier2Config({});
    expect(c).not.toBe(DEFAULT_TIER2_CONFIG);
    expect(DEFAULT_TIER2_CONFIG.modelId).toBe("Qwen3.5-2B-q4f16_1-MLC");
    expect(DEFAULT_TIER2_CONFIG.contextWindowSize).toBe(8192);
    expect(DEFAULT_TIER2_CONFIG.maxTokens).toBe(512);
  });
});
