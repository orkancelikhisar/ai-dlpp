import { describe, expect, it, vi } from "vitest";
import { MODEL_MANIFEST } from "../src/config.js";
import {
  assertSignature,
  createOrtSession,
  type OnnxSession,
  type OnnxTensor,
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

/** Stands in for the runtime's tensor class, so `instanceof` means something. */
class StubTensor {
  constructor(
    readonly type: string,
    readonly data: never,
    readonly dims: readonly number[],
  ) {}
}

/**
 * A runtime stub. `Tensor` is required because `createOrtSession` no longer
 * returns the runtime's session directly -- it wraps it so every feed is
 * converted to the runtime's own tensor class on the way in.
 */
const runtimeStub = (create: OrtRuntime["InferenceSession"]["create"]): OrtRuntime => ({
  InferenceSession: { create },
  Tensor: StubTensor as unknown as OrtRuntime["Tensor"],
});

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
    const got = await createOrtSession("https://example.test/model.onnx", "webgpu", () =>
      Promise.resolve(runtimeStub(create)),
    );

    // ONE provider, and exactly the one asked for. A fallback list would let
    // onnxruntime answer with the other backend, and the whole point of the
    // wasm/webgpu rungs is to measure them apart -- a webgpu arm silently
    // served by wasm reports wasm latency under webgpu's name.
    expect(create).toHaveBeenCalledWith("https://example.test/model.onnx", {
      executionProviders: ["webgpu"],
    });

    // Deliberately NOT `toBe(created)`: createOrtSession now wraps the
    // runtime's session so feeds are converted to its tensor class. Identity
    // would pin the absence of that wrapper. What matters is that the wrapper
    // delegates rather than inventing its own answers.
    expect(got).not.toBe(created);
    expect(got.inputNames).toBe(created.inputNames);
    expect(got.outputNames).toBe(created.outputNames);
  });

  it("rebuilds every feed as one of the runtime's own tensors", async () => {
    // MEASURED, and the reason the wrapper exists at all: onnxruntime-node
    // 1.21.0 rejects a plain `{ dims, type, data }` feed with `Tensor.location
    // must be a string.` before it reaches the graph. Everything upstream of
    // this file describes a tensor as a plain object -- which is what makes the
    // tagger testable without a runtime -- so the conversion has to happen on
    // the way through, on every feed, or the browser path fails on its first
    // real run while every fake-session test still passes.
    let seen: Record<string, OnnxTensor> | undefined;
    const created = {
      inputNames: EDGE.inputNames,
      outputNames: ["logits"],
      run: (feeds: Record<string, OnnxTensor>) => {
        seen = feeds;
        return Promise.resolve({});
      },
      release: () => Promise.resolve(),
    } as unknown as OnnxSession;
    const got = await createOrtSession("m.onnx", "wasm", () =>
      Promise.resolve(runtimeStub(() => Promise.resolve(created))),
    );

    // EVERY feed, not the first one: the two rungs send four and six tensors,
    // and a converter that stopped after one would be caught by no
    // single-tensor test while failing on the graph's second input.
    const ids = BigInt64Array.from([1n, 2n]);
    const mask = Uint8Array.from([1, 0]);
    await got.run({
      input_ids: { dims: [1, 2], type: "int64", data: ids },
      span_mask: { dims: [1, 2], type: "bool", data: mask },
    });

    const converted = seen?.["input_ids"];
    expect(converted).toBeInstanceOf(StubTensor);
    expect(converted?.type).toBe("int64");
    expect(converted?.dims).toEqual([1, 2]);
    // The buffer is handed over, not copied: these feeds are the whole message
    // and copying every one of them per segment is pure cost.
    expect(converted?.data).toBe(ids);

    expect(seen?.["span_mask"]).toBeInstanceOf(StubTensor);
    expect(seen?.["span_mask"]?.type).toBe("bool");
    expect(seen?.["span_mask"]?.data).toBe(mask);
  });

  it("passes the wasm backend through unchanged", async () => {
    const create = vi.fn(() => Promise.resolve(session(EDGE.inputNames)));
    await createOrtSession("m.onnx", "wasm", () => Promise.resolve(runtimeStub(create)));
    expect(create).toHaveBeenCalledWith("m.onnx", { executionProviders: ["wasm"] });
  });
});
