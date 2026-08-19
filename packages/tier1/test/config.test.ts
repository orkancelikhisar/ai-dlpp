import { describe, expect, it } from "vitest";
import {
  DEFAULT_TIER1_CONFIG,
  MODEL_MANIFEST,
  modelFileUrl,
  resolveTier1Config,
} from "../src/config.js";

describe("MODEL_MANIFEST", () => {
  it("pins every file by sha256, not by tag", () => {
    // A HuggingFace tag is mutable. Weights that change under a fixed id would
    // silently invalidate every number already measured against them.
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      for (const [path, file] of Object.entries(entry.files)) {
        expect(file.sha256, `${id}/${path}`).toMatch(/^[0-9a-f]{64}$/);
        expect(file.bytes, `${id}/${path}`).toBeGreaterThan(0);
      }
    }
  });

  it("includes the primary and the ladder siblings the experiment matrix names", () => {
    expect(Object.keys(MODEL_MANIFEST)).toEqual(
      expect.arrayContaining([
        "gliner-pii-edge",
        "gliner-pii-edge-fp16",
        "gliner-pii-edge-uint8",
        "gliner-pii-base",
        "gliner-pii-base-fp16",
        "gliner-pii-base-uint8",
      ]),
    );
  });

  it("resolves every file to an immutable commit sha, never a branch name", () => {
    // MEASURED: gliner_config.json changed max_len 1024 -> 2048 in BOTH repos
    // under a stable filename. A `main` url would have served the new file
    // against weights whose hash never moved, so the revision is the pin that
    // actually holds the model still.
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      expect(entry.revision, id).toMatch(/^[0-9a-f]{40}$/);
      for (const path of Object.keys(entry.files)) {
        expect(modelFileUrl(entry, path), `${id}/${path}`).toBe(
          `https://huggingface.co/${entry.repo}/resolve/${entry.revision}/${path}`,
        );
        expect(modelFileUrl(entry, path), `${id}/${path}`).toMatch(
          /^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\//,
        );
      }
    }
  });

  it("names each entry after the repo it pins, so an id cannot point at a sibling", () => {
    // MEASURED: without this, swapping gliner-pii-edge's repo to the base repo
    // passes the whole suite. The id is what every run record carries, so an id
    // pointing at a sibling would mislabel every measurement taken under it.
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      const withoutPrecision = id.replace(/-(fp16|uint8)$/, "");
      expect(entry.repo, id).toMatch(new RegExp(`/${withoutPrecision}-v[0-9.]+$`));
    }
  });

  it("pins the files that define the model, not only the weights", () => {
    // The weights alone do not determine behaviour: gliner_config.json carries
    // max_len, max_width and span_mode, and the tokenizer decides where token
    // boundaries -- and therefore span offsets -- fall.
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      expect(Object.keys(entry.files), id).toEqual(
        expect.arrayContaining([
          entry.weightsPath,
          "gliner_config.json",
          "tokenizer.json",
          "tokenizer_config.json",
          "special_tokens_map.json",
        ]),
      );
    }
  });

  it("offers every precision as its own id, so the trade is selected not assumed", () => {
    // fp32 base is 665 MB, which is not a realistic WASM load. Precision has to
    // be a rung the matrix can select and report, not a choice buried in a
    // loader, because accuracy-vs-latency across these IS the experiment.
    const precisionsByRepo = new Map<string, string[]>();
    for (const entry of Object.values(MODEL_MANIFEST)) {
      precisionsByRepo.set(entry.repo, [
        ...(precisionsByRepo.get(entry.repo) ?? []),
        entry.precision,
      ]);
    }
    expect(precisionsByRepo.size).toBe(2);
    for (const [repo, precisions] of precisionsByRepo) {
      expect([...precisions].sort(), repo).toEqual(["fp16", "fp32", "uint8"]);
    }
  });

  it("ties each precision to the artifact upstream actually publishes for it", () => {
    const published: Record<string, string> = {
      fp32: "onnx/model.onnx",
      fp16: "onnx/model_fp16.onnx",
      uint8: "onnx/model_quint8.onnx",
    };
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      expect(entry.weightsPath, id).toBe(published[entry.precision]);
      expect(Object.keys(entry.files), id).toContain(entry.weightsPath);
    }
  });

  it("records the graph inputs that differ between the two span modes", () => {
    // MEASURED by parsing each ONNX graph: these two are not interchangeable
    // rungs of one ladder. A loader written against the 4-input token_level
    // graph cannot feed the 6-input markerV0 one.
    const edge = MODEL_MANIFEST["gliner-pii-edge"];
    const base = MODEL_MANIFEST["gliner-pii-base"];
    expect(edge?.spanMode).toBe("token_level");
    expect(base?.spanMode).toBe("markerV0");
    expect(edge?.inputNames).toEqual([
      "input_ids",
      "attention_mask",
      "words_mask",
      "text_lengths",
    ]);
    expect(base?.inputNames).toEqual([
      "input_ids",
      "attention_mask",
      "words_mask",
      "text_lengths",
      "span_idx",
      "span_mask",
    ]);
  });

  it("records the max_len that drifted upstream, so a re-pin has to restate it", () => {
    for (const [id, entry] of Object.entries(MODEL_MANIFEST)) {
      expect(entry.maxLen, id).toBe(2048);
    }
  });
});

describe("resolveTier1Config", () => {
  it("defaults to the primary model on wasm", () => {
    const config = resolveTier1Config({});
    expect(config.modelId).toBe("gliner-pii-edge");
    expect(config.backend).toBe("wasm");
  });

  it("pins the whole default config, since every field shapes a measured number", () => {
    // threshold, maxWidth and labelForm are experiment variables. A silent edit
    // to any of them would move every reported number with nothing in the
    // record to show it. MEASURED: adding labelForm failed this assertion
    // before it was restated here, which is the whole point of pinning it.
    expect(DEFAULT_TIER1_CONFIG).toEqual({
      modelId: "gliner-pii-edge",
      backend: "wasm",
      threshold: 0.5,
      maxWidth: 12,
      labelForm: "id",
    });
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

  it("treats an explicitly undefined override as not specified", () => {
    // The matrix and CLI both build overrides positionally, so an unset option
    // arrives as an own key holding undefined. A bare spread copies that over
    // the default and turns "not specified" into a hard failure.
    const config = resolveTier1Config({
      modelId: undefined,
      backend: undefined,
      threshold: undefined,
      maxWidth: undefined,
      labelForm: undefined,
    });
    expect(config).toEqual(DEFAULT_TIER1_CONFIG);
  });

  it("rejects a threshold outside (0,1] rather than silently clamping", () => {
    // A clamped threshold produces a run whose reported config does not match
    // the config that ran — the single worst failure for a measurement tool.
    expect(() => resolveTier1Config({ threshold: 0 })).toThrow(/threshold/);
    expect(() => resolveTier1Config({ threshold: 1.5 })).toThrow(/threshold/);
  });

  it("rejects a maxWidth that is not a positive integer", () => {
    // MEASURED: deleting the maxWidth guard entirely leaves the other tests
    // green. A fractional width silently changes how many spans GLiNER
    // enumerates, so an unvalidated value is a config that cannot be reproduced.
    expect(() => resolveTier1Config({ maxWidth: 0 })).toThrow(/maxWidth/);
    expect(() => resolveTier1Config({ maxWidth: 2.5 })).toThrow(/maxWidth/);
  });

  it("rejects a labelForm outside the three the return type declares", () => {
    // labelForm decides the text every class is prompted with, so an unchecked
    // value would reach buildLabels' switch, fall through every case and prompt
    // the model with undefined while the run record named a real form.
    expect(() => resolveTier1Config({ labelForm: "nl-definition" as never })).toThrow(/labelForm/);
  });

  it("rejects a backend outside the two the return type declares", () => {
    // Without this guard the function returns a Tier1Config whose `backend` is
    // a string the type says is impossible, and the run record would name a
    // backend that was never executed.
    expect(() => resolveTier1Config({ backend: "cuda" as never })).toThrow(/backend/);
  });
});
