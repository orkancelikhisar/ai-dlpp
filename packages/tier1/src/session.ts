import type { Tier1Backend } from "./config.js";
import type { ModelEntry } from "./manifest.js";

/**
 * One tensor as onnxruntime hands it over.
 *
 * Structural, and `data` is `unknown`: the four feeds are INT64 and base's
 * `span_mask` is BOOL, which in onnxruntime-web are a BigInt64Array and a
 * Uint8Array respectively -- and Task 7 measured that the logits come back
 * float32 on every rung that loads at all, including the uint8 ones, which
 * dequantise inside the graph. Naming a concrete array type here would state a
 * union that the caller has to narrow anyway.
 */
export interface OnnxTensor {
  readonly dims: readonly number[];
  readonly type: string;
  readonly data: unknown;
}

/**
 * The seam the runtime is reached through.
 *
 * Deliberately not `ort.InferenceSession`: the measured runtime path is
 * onnxruntime-web in a browser, the tests run in Node, and `onnxruntime-node`
 * is a devDependency used only by scripts/probe-model.ts. A structural type is
 * what lets the two share a decoder without either importing the other's
 * package.
 */
export interface OnnxSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Readonly<Record<string, OnnxTensor>>): Promise<Record<string, OnnxTensor>>;
  release(): Promise<void>;
}

/** Just enough of onnxruntime-web's shape to create a session. */
export interface OrtRuntime {
  readonly InferenceSession: {
    create(path: string, options: { executionProviders: string[] }): Promise<OnnxSession>;
  };
}

/**
 * The output every pinned graph returns, measured by Task 7 on all four rungs
 * that load: `outputNames` is exactly `["logits"]` on edge and base alike.
 */
const LOGITS = "logits";

/**
 * Checks a loaded graph against the signature the manifest pins for it.
 *
 * This is not a formality. The two models take DIFFERENT inputs -- markerV0
 * adds `span_idx` and `span_mask` that token_level has no use for -- and the
 * feed builder branches on that. Getting the pairing wrong is a realistic
 * mistake in a six-rung experiment matrix where all six files are called
 * `model.onnx` or `model_quantized.onnx`, and it does not fail cleanly: an
 * edge feed run on base is missing two required inputs, while a base feed run
 * on edge carries two the graph ignores, and the second of those returns
 * confident, wrongly shaped logits.
 *
 * Order is deliberately not checked. Task 7 measured `session.inputNames`
 * matching the manifest's declaration order on every rung that loads, so order
 * carries no information the set does not, and requiring it would make an
 * onnxruntime upgrade that reorders look like a corrupted model file.
 */
export function assertSignature(
  session: OnnxSession,
  entry: ModelEntry,
  modelId: string,
): void {
  const got = new Set(session.inputNames);
  const want = new Set(entry.inputNames);
  const missing = entry.inputNames.filter((name) => !got.has(name));
  const unexpected = session.inputNames.filter((name) => !want.has(name));
  const problems: string[] = [];
  if (missing.length > 0) problems.push(`missing input(s) ${missing.join(", ")}`);
  if (unexpected.length > 0) problems.push(`unexpected input(s) ${unexpected.join(", ")}`);
  if (!session.outputNames.includes(LOGITS)) {
    problems.push(`no ${LOGITS} output, got ${session.outputNames.join(", ") || "none"}`);
  }
  if (problems.length > 0) {
    throw new Error(`${modelId}: loaded graph does not match the pinned signature -- ${problems.join("; ")}`);
  }
}

/**
 * The specifier goes through a `string`-typed variable rather than a literal.
 *
 * MEASURED, by putting a literal specifier here and running `npm run -s
 * typecheck` from the repo root: tsc reports TS7016, "Could not find a
 * declaration file for module 'onnxruntime-web'. .../dist/ort.bundle.min.mjs
 * implicitly has an 'any' type. There are types at
 * .../onnxruntime-web/types.d.ts, but this result could not be resolved when
 * respecting package.json \"exports\"." for this pinned dev build.
 * scripts/probe-model.ts hit the same thing and took the same way out. Only
 * `InferenceSession.create` is used, and `OrtRuntime` above states that more
 * precisely than the package's own types would.
 */
const WEB_RUNTIME: string = "onnxruntime-web";

const importOnnxruntimeWeb = async (): Promise<OrtRuntime> =>
  (await import(WEB_RUNTIME)) as unknown as OrtRuntime;

/**
 * Loads one graph on one backend.
 *
 * `loadRuntime` is injectable so the plumbing below is testable without pulling
 * a WASM runtime into a Node test process; the default is the real import, and
 * nothing else in this package may import onnxruntime-web directly.
 */
export async function createOrtSession(
  modelUrl: string,
  backend: Tier1Backend,
  loadRuntime: () => Promise<OrtRuntime> = importOnnxruntimeWeb,
): Promise<OnnxSession> {
  const runtime = await loadRuntime();
  // One provider, never a fallback list: the whole point of the wasm/webgpu
  // rungs is to measure them apart, and a list would let onnxruntime silently
  // answer with the other one.
  return runtime.InferenceSession.create(modelUrl, { executionProviders: [backend] });
}
