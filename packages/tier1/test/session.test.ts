import { describe, expect, it, vi } from "vitest";
import { MODEL_MANIFEST } from "../src/config.js";
import {
  assertSignature,
  createOrtSession,
  type OnnxSession,
  type OrtRuntime,
} from "../src/session.js";

const EDGE = MODEL_MANIFEST["gliner-pii-edge"]!;
const BASE = MODEL_MANIFEST["gliner-pii-base"]!;

const session = (inputNames: readonly string[], outputNames: readonly string[] = ["logits"]) =>
  ({
    inputNames,
    outputNames,
    run: () => Promise.resolve({}),
    release: () => Promise.resolve(),
  }) as OnnxSession;

describe("assertSignature", () => {
  it("accepts a graph whose inputs are exactly what the manifest pins", () => {
    expect(() => assertSignature(session(EDGE.inputNames), EDGE, "gliner-pii-edge")).not.toThrow();
    expect(() => assertSignature(session(BASE.inputNames), BASE, "gliner-pii-base")).not.toThrow();
  });

  it("rejects a token_level graph loaded against the span-mode entry", () => {
    // The realistic mix-up: the two rungs differ by span_idx/span_mask, and a
    // feed built for one and run on the other either throws deep inside
    // onnxruntime or silently ignores the extra inputs.
    expect(() => assertSignature(session(EDGE.inputNames), BASE, "gliner-pii-base")).toThrow(
      /span_idx/,
    );
  });

  it("rejects a graph that takes inputs the manifest does not name", () => {
    expect(() =>
      assertSignature(session([...EDGE.inputNames, "token_type_ids"]), EDGE, "gliner-pii-edge"),
    ).toThrow(/token_type_ids/);
  });

  it("rejects a graph that does not return logits", () => {
    expect(() => assertSignature(session(EDGE.inputNames, ["start_logits"]), EDGE, "x")).toThrow(
      /logits/,
    );
  });

  it("names the model in the message, so a six-rung matrix is debuggable", () => {
    expect(() => assertSignature(session([]), EDGE, "gliner-pii-edge-uint8")).toThrow(
      /gliner-pii-edge-uint8/,
    );
  });

  it("does not care about declaration order", () => {
    // Task 7 measured runtime inputNames matching the declared order on every
    // rung that loads, so order is not evidence of anything; requiring it would
    // only make an onnxruntime upgrade look like a corrupted model.
    expect(() =>
      assertSignature(session([...BASE.inputNames].reverse()), BASE, "gliner-pii-base"),
    ).not.toThrow();
  });
});

describe("createOrtSession", () => {
  it("asks the runtime for the backend it was given", async () => {
    const created = session(EDGE.inputNames);
    const create = vi.fn(() => Promise.resolve(created));
    const runtime: OrtRuntime = { InferenceSession: { create } };
    const got = await createOrtSession("https://example.test/model.onnx", "webgpu", () =>
      Promise.resolve(runtime),
    );
    expect(got).toBe(created);
    expect(create).toHaveBeenCalledWith("https://example.test/model.onnx", {
      executionProviders: ["webgpu"],
    });
  });

  it("passes the wasm backend through unchanged", async () => {
    const create = vi.fn(() => Promise.resolve(session(EDGE.inputNames)));
    await createOrtSession("m.onnx", "wasm", () =>
      Promise.resolve({ InferenceSession: { create } }),
    );
    expect(create).toHaveBeenCalledWith("m.onnx", { executionProviders: ["wasm"] });
  });
});
