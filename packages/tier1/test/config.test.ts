import { describe, expect, it } from "vitest";
import { MODEL_MANIFEST, resolveTier1Config } from "../src/config.js";

describe("MODEL_MANIFEST", () => {
  it("pins every model by sha256, not by tag", () => {
    // A HuggingFace tag is mutable. Weights that change under a fixed id would
    // silently invalidate every number already measured against them.
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      expect(entry.sha256, id).toMatch(/^[0-9a-f]{64}$/);
      expect(entry.url, id).toMatch(/^https:\/\//);
      expect(entry.bytes, id).toBeGreaterThan(0);
    }
  });

  it("includes the primary and the ladder siblings the experiment matrix names", () => {
    expect(Object.keys(MODEL_MANIFEST)).toEqual(
      expect.arrayContaining(["gliner-pii-edge", "gliner-pii-base"]),
    );
  });
});

describe("resolveTier1Config", () => {
  it("defaults to the primary model on wasm", () => {
    const config = resolveTier1Config({});
    expect(config.modelId).toBe("gliner-pii-edge");
    expect(config.backend).toBe("wasm");
  });

  it("rejects a model id that is not in the manifest, naming what is available", () => {
    expect(() => resolveTier1Config({ modelId: "not-a-model" })).toThrow(/gliner-pii-edge/);
  });

  it("rejects an inherited Object key rather than resolving it up the prototype chain", () => {
    // MEASURED: swapping the Object.hasOwn guard for `!MODEL_MANIFEST[id]` leaves
    // the other tests green, because "not-a-model" is undefined either way.
    // "constructor" is the case that separates them -- it resolves to a truthy
    // function through the prototype, so the bare form admits a model that does
    // not exist and every downstream record would name it.
    expect(() => resolveTier1Config({ modelId: "constructor" })).toThrow(/gliner-pii-edge/);
  });

  it("rejects a maxWidth that is not a positive integer", () => {
    // MEASURED: deleting the maxWidth guard entirely leaves the other tests
    // green. A fractional width silently changes how many spans GLiNER
    // enumerates, so an unvalidated value is a config that cannot be reproduced.
    expect(() => resolveTier1Config({ maxWidth: 0 })).toThrow(/maxWidth/);
    expect(() => resolveTier1Config({ maxWidth: 2.5 })).toThrow(/maxWidth/);
  });

  it("rejects a threshold outside (0,1] rather than silently clamping", () => {
    // A clamped threshold produces a run whose reported config does not match
    // the config that ran — the single worst failure for a measurement tool.
    expect(() => resolveTier1Config({ threshold: 0 })).toThrow(/threshold/);
    expect(() => resolveTier1Config({ threshold: 1.5 })).toThrow(/threshold/);
  });
});
