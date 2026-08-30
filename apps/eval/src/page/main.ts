import { detect, loadPolicyIr, type DetectionResult, type TierConfig } from "@sih/core";
import {
  GlinerSpanTagger,
  MODEL_MANIFEST,
  TIER1_BACKENDS,
  createOrtSession,
  resolveTier1Config,
  tokenizerFromEncoder,
  type OrtRuntime,
  type Tier1Backend,
  type Tier1Config,
  type Tier1TaggerStats,
} from "@sih/tier1";
// Copied from packages/core/test/fixtures/minimal-ir.ts and owned by this app:
// core's test fixture is a TypeScript module, and importing it would both make
// a test-only artifact a runtime dependency of the harness and bypass the thing
// worth exercising here -- the extension receives a compiled IR as JSON text and
// parses it with loadPolicyIr, so the page must too. Later tasks replace this
// with a real compiled policy; until then it is a placeholder, not a baseline.
import irJson from "../../fixtures/minimal-ir.json?raw";
// Reached by PATH rather than by the package's own subpath export, because
// onnxruntime-web's `exports` map publishes no `./dist/*` entry -- there is no
// specifier that names this file. See `loadOrtRuntime` for why the page has to
// name it at all.
import jsepWasmUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url";

export interface DetectRequest {
  text: string;
  provider: string;
  config: TierConfig;
}

/**
 * Which rung to load, and under what settings.
 *
 * `backend` is required because there is no defensible default -- the whole
 * point of the ladder is measuring wasm and webgpu apart. Everything else is
 * optional and filled in by `resolveTier1Config`, which is also what validates
 * it; passing the whole `Partial<Tier1Config>` through rather than a hand-picked
 * subset is what keeps the settings a caller can vary and the settings the
 * report states the same set.
 */
export type Tier1LoadOptions = Partial<Tier1Config> & { backend: Tier1Backend };

/**
 * What a tier-1 load actually did, as opposed to what it was asked to do.
 *
 * `config.backend` is the label; `observedBackend` is a MEASUREMENT taken around
 * a real warm-up inference -- see `loadTier1`. They are separate fields rather
 * than one because the interesting case is exactly when they disagree, and
 * `loadTier1` refuses to return a report where they do.
 */
export interface Tier1LoadReport {
  /**
   * The FULLY RESOLVED config this rung is running under -- every field, not
   * just the two the caller named. `threshold`, `maxWidth` and `labelForm` all
   * change what the model reports, so an arm recorded under a partial config is
   * an arm nobody can reproduce.
   */
  readonly config: Tier1Config;
  readonly observedBackend: Tier1Backend;
  /** The URL the graph was actually fetched from. */
  readonly weightsUrl: string;
  /**
   * `content-length` of that URL, so a caller holding MODEL_MANIFEST can check
   * the page loaded the pinned artifact rather than a 404 body or a different
   * precision variant. Reported, never judged here: the manifest lives in
   * @sih/tier1 and the driver is where the comparison belongs.
   */
  readonly weightsBytes: number;
  /** Wall clock for the whole load: fetch, graph compile, tokenizer, warm-up. */
  readonly loadMs: number;
  /** `session.run` calls the warm-up made. Zero would mean the graph never ran. */
  readonly warmupInferences: number;
  /** `GPUQueue.submit` calls counted during the warm-up. This is `observedBackend`'s evidence. */
  readonly warmupGpuSubmits: number;
}

/** What one `detect` call cost tier 1, as deltas over the tagger's running counters. */
export interface Tier1DetectStats extends Tier1TaggerStats {
  /** `GPUQueue.submit` calls during that one call. */
  gpuSubmits: number;
}

export interface Tier1Status {
  readonly load: Tier1LoadReport;
  /** The tagger's counters since it was constructed. */
  readonly totals: Tier1TaggerStats;
  /**
   * The most recent `detect` that ran tier 1, as deltas. `undefined` until one
   * has. This is what separates "the model ran and found nothing" from "the
   * model never ran" -- findings alone cannot, and neither can a timing.
   */
  readonly lastDetect: Tier1DetectStats | undefined;
}

/**
 * The page's whole API surface: every spec reaches detection through this, and
 * everything below `detect` is core, imported unmodified from the package root
 * exactly as the extension will import it.
 *
 * A CONVENTION, not a sandbox. The property is frozen and non-writable below,
 * so a spec cannot swap the implementation out from under the page -- but
 * nothing stops one importing @sih/core (or anything else) into the browser
 * context and measuring that instead. Keeping the harness honest about what it
 * measures is a review obligation; this interface is only what makes the
 * intended path the easy one.
 */
export interface SihPageApi {
  detect(request: DetectRequest): Promise<DetectionResult>;
  /**
   * sha256 of the IR artifact this page loaded, lowercase hex. Async because it
   * is a WebCrypto digest; see `irHash` below for why this is not the IR's own
   * `policyHash` field.
   */
  irHash(): Promise<string>;
  /**
   * The loaded IR's `policyHash` field verbatim -- the compiler's hash of the
   * policy DOCUMENT, which answers a different question from `irHash` and is
   * carried alongside it rather than instead of it. Synchronous: a field read,
   * not a digest.
   */
  policyHash(): string;
  /**
   * Whether this browser can actually run `backend`, asked of the browser
   * rather than guessed from a user agent.
   *
   * For `"webgpu"` that means `requestAdapter()` RESOLVING NON-NULL, not merely
   * `navigator.gpu` existing. Task 1 measured Playwright's bundled headless
   * shell exposing `navigator.gpu` while its `requestAdapter()` resolves null,
   * and onnxruntime-web reads a null adapter as "no webgpu" and falls back to
   * wasm silently -- so the weaker check would report the shell as GPU-capable
   * and every latency measured on it would name the wrong runtime.
   */
  backendAvailable(backend: Tier1Backend): Promise<boolean>;
  /**
   * Load one rung of the tier-1 ladder into this page and make `detect` use it.
   *
   * Returns a report rather than void (which is what the plan specified)
   * because everything worth asserting about a load is a measurement taken
   * inside the page: which execution provider really initialized, how many
   * bytes of weights were served, whether the graph ran at all. A void return
   * would leave a spec asserting only that the call did not throw.
   */
  loadTier1(options: Tier1LoadOptions): Promise<Tier1LoadReport>;
  /** `undefined` until `loadTier1` has succeeded. */
  tier1Status(): Tier1Status | undefined;
}

declare global {
  interface Window {
    /**
     * Optional because it genuinely is: the driver navigates, then polls for
     * this property to appear. Typing it as always-present would make the wait
     * that every spec opens with look like dead code.
     */
    __sih?: SihPageApi;
  }
}

// Parsed at module scope so a malformed IR fails BEFORE `__sih` is published,
// rather than a spec receiving an API that throws on first use. The cost is
// that the failure reaches the driver only as a pageerror -- `openHarness` in
// smoke.spec.ts listens for exactly that and re-throws it with the message.
const ir = loadPolicyIr(irJson);

/**
 * sha256 of `irJson` -- the exact bytes of the IR file this page loaded. This is
 * what a record's `irHash` carries; `ir.policyHash` is carried separately and
 * verbatim, because the two answer different questions.
 *
 * The immediate reason they cannot be one field is that this fixture's
 * `policyHash` is the literal string "test-hash" while a record's `irHash`
 * requires /^[0-9a-f]{64}$/. The lasting reason survives that fixture being
 * replaced by a compiled policy: `ir.policyHash` is the compiler's hash of the
 * policy DOCUMENT, while a record has to answer "which IR produced these
 * numbers". Compilation is model-driven, so the same document compiled twice by
 * the same compiler can yield two different IRs carrying that identical
 * `policyHash` -- it identifies the input, never the artifact.
 *
 * Hashing the raw TEXT rather than a re-serialization of the parsed IR is what
 * makes the answer checkable from outside the browser. MEASURED: for the current
 * fixture this returns
 * cf82e7e925ef6b80036f96225d056516ad1bd885efcb0b6264f903e9fae8271a, which is
 * exactly what `shasum -a 256 apps/eval/fixtures/minimal-ir.json` prints -- so
 * someone holding only a JSONL file can confirm the artifact instead of taking
 * the record's word for it. test/run.spec.ts asserts that equality on every run.
 *
 * The cost is that a whitespace-only reformat of the file changes the hash while
 * the IR is semantically identical. That is the direction to err in: it can call
 * two identical IRs different, never two different IRs the same.
 *
 * Computed once at module scope and handed out as a promise rather than awaited
 * here. A top-level await would turn this file into an async module, and I have
 * not measured whether a rejected async module evaluation still reaches
 * Playwright's `pageerror` listener -- which `openHarness` in smoke.spec.ts
 * relies on to report a malformed IR by name. Keeping the module synchronous
 * leaves that path exactly as Task 1 left it.
 */
const irHash = crypto.subtle
  .digest("SHA-256", new TextEncoder().encode(irJson))
  .then((digest) =>
    Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join(""),
  );

// -- tier 1 ------------------------------------------------------------------

/**
 * Every pinned model file, as URLs Vite resolved at transform time.
 *
 * MEASURED in this browser: these globs resolve to `/@fs/<absolute path>` URLs
 * and a plain `fetch()` of one answers 206 to a Range request, so Vite's DEFAULT
 * `server.fs.allow` (the workspace root) already covers packages/tier1/models.
 * Nothing needs adding to vite.config.ts -- see the comment there for why adding
 * it would in fact take coverage away.
 *
 * `eager` because the whole point is that the map's KEYS answer "is this rung on
 * disk", which a lazy glob could only answer by importing. models/ is gitignored
 * and routinely absent, in which case this is `{}` and `loadTier1` says so by
 * name instead of letting onnxruntime fetch a 404 body and fail on a magic word.
 */
const MODELS_GLOB_PREFIX = "../../../../packages/tier1/models/";
const MODEL_FILE_URLS: Readonly<Record<string, string>> = Object.fromEntries(
  Object.entries({
    ...import.meta.glob("../../../../packages/tier1/models/*/onnx/*.onnx", {
      query: "?url",
      import: "default",
      eager: true,
    }),
    ...import.meta.glob("../../../../packages/tier1/models/*/tokenizer.json", {
      query: "?url",
      import: "default",
      eager: true,
    }),
  }).map(([key, url]) => [key.slice(MODELS_GLOB_PREFIX.length), url as string]),
);

const FETCH_MODELS_HINT =
  "run scripts/fetch-models.ts (pnpm -C packages/tier1 exec vite-node ../../scripts/fetch-models.ts)";

function modelFileUrlOrThrow(modelId: string, repoPath: string): string {
  const key = `${modelId}/${repoPath}`;
  if (!Object.hasOwn(MODEL_FILE_URLS, key)) {
    throw new Error(
      `tier-1 weights missing: packages/tier1/models/${key} is not on disk; ${FETCH_MODELS_HINT}`,
    );
  }
  return MODEL_FILE_URLS[key] as string;
}

/**
 * Counts every `GPUQueue.submit` this page makes, from before anything can make
 * one.
 *
 * This is how `backend` stops being a label. onnxruntime-web dispatches its
 * WebGPU work by submitting command buffers to the device queue, so a count
 * taken around an inference answers "did the GPU do this" independently of what
 * the session was asked for, what onnxruntime reports, and what the arm is
 * named. MEASURED on this page across all four rungs that load: a session
 * created with `executionProviders: ["wasm"]` submits 0 during a run, and one
 * created with `["webgpu"]` submits 40 to 193 depending on the graph.
 *
 * Neither obvious alternative works. `env.webgpu.device` is documented as a
 * getter that CREATES a device when read before the first webgpu session, so
 * reading it to ask "did webgpu initialize" would make the answer yes.
 * `env.webgpu.adapter` is a real signal -- measured undefined after a wasm
 * session and defined after a webgpu one -- but a sticky one: once any webgpu
 * session has existed it stays set for the page's lifetime, so it cannot say
 * whether THIS session used the GPU.
 */
let gpuSubmits = 0;
{
  type GpuQueueCtor = { prototype: { submit: (...args: never[]) => unknown } };
  const gpuQueue = (globalThis as unknown as { GPUQueue?: GpuQueueCtor }).GPUQueue;
  if (gpuQueue !== undefined) {
    const original = gpuQueue.prototype.submit;
    gpuQueue.prototype.submit = function counted(this: never, ...args: never[]): unknown {
      gpuSubmits += 1;
      return original.apply(this, args);
    };
  }
}

/** WebGPU has no lib.dom typings here; naming the one method used is cheaper than the dependency. */
type MaybeWebGpu = { gpu?: { requestAdapter(): Promise<unknown> } };

async function backendAvailable(backend: Tier1Backend): Promise<boolean> {
  if (!TIER1_BACKENDS.includes(backend)) return false;
  if (backend === "wasm") return true;
  const { gpu } = navigator as Navigator & MaybeWebGpu;
  if (gpu === undefined) return false;
  return (await gpu.requestAdapter()) !== null;
}

/**
 * The onnxruntime-web module, with the one piece of environment setup that a
 * bundled page cannot do without.
 *
 * Injected through `createOrtSession`'s `loadRuntime` seam rather than left to
 * its default, because MEASURED: that default is `import(WEB_RUNTIME)` with a
 * `string`-typed variable specifier, and in the browser it throws "Failed to
 * resolve module specifier 'onnxruntime-web'". Vite's import analysis rewrites
 * only LITERAL specifiers, so the bare package name reaches the browser
 * unresolved. The variable exists to dodge a TS7016 on the package's own types
 * (session.ts records that), which means the default loader cannot run in any
 * bundler-served page -- this seam is the supported way in, not a test hook.
 *
 * `wasmPaths` is set for two reasons, one measured and one prevented.
 *
 * MEASURED: without it, session creation fails with
 * `CompileError: WebAssembly.instantiate(): expected magic word 00 61 73 6d,
 * found 3c 21 64 6f` -- those four bytes are "<!do", so what arrived was an HTML
 * document. onnxruntime resolves `ort-wasm-simd-threaded.jsep.wasm` relative to
 * its own module URL, which after Vite's dependency pre-bundling is
 * /node_modules/.vite/deps/, where the file does not exist and the dev server
 * serves index.html instead.
 *
 * PREVENTED: @huggingface/transformers shares this exact ORT instance (measured
 * -- it pins the same version and keeps `onnxruntime-web` external in its web
 * bundle), and its module body contains
 *   `if (!ONNX_ENV.wasm.wasmPaths) ONNX_ENV.wasm.wasmPaths =
 *      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@<v>/dist/"`
 * which would point the pinned runtime at a different onnxruntime build fetched
 * over the network. Setting the field before transformers is imported makes that
 * assignment a no-op, since its guard is `!wasmPaths`. That assignment has NOT
 * been observed here (this page has always set the field first), so what the
 * fetch itself would do under COEP: require-corp is unmeasured -- `assertLocalWasm`
 * exists so the question never has to be answered.
 */
interface OrtWebModule extends OrtRuntime {
  readonly env: { wasm: { wasmPaths?: unknown } };
}

let ortModule: OrtWebModule | undefined;

async function loadOrtRuntime(): Promise<OrtWebModule> {
  // No cast: src/page/ort-web.d.ts declares exactly this shape, borrowing two of
  // its three members from `OrtRuntime` so the two cannot drift.
  const loaded: OrtWebModule = ortModule ?? (await import("onnxruntime-web"));
  // Assigned on EVERY load, not only the first. Idempotent, and it means the
  // second rung a page loads gets the same guarantee as the first without
  // depending on nothing having touched the field in between.
  loaded.env.wasm.wasmPaths = { wasm: jsepWasmUrl };
  ortModule = loaded;
  return loaded;
}

function assertLocalWasm(runtime: OrtWebModule): void {
  const paths = runtime.env.wasm.wasmPaths;
  const wasm = (paths as { wasm?: unknown } | undefined)?.wasm;
  if (wasm !== jsepWasmUrl) {
    throw new Error(
      "onnxruntime wasm binary is not the pinned local one; something reassigned " +
        `env.wasm.wasmPaths (now ${JSON.stringify(paths)})`,
    );
  }
}

/**
 * The subword tokenizer for one rung, through @huggingface/transformers.
 *
 * `AutoTokenizer.from_pretrained` reads exactly `tokenizer.json` and
 * `tokenizer_config.json` (measured in its web bundle), joining
 * `env.localModelPath` with `<modelId>/<file>`. Vite gives absolute URLs per
 * FILE, so the shared prefix is recovered by stripping the known suffix off one
 * of them -- checked rather than assumed, because a silently wrong prefix would
 * become a remote fetch, and `allowRemoteModels = false` is what makes that a
 * refusal rather than a download.
 *
 * Deliberately the same entry point Task 10 used from Node, so a browser/Node
 * score difference cannot be blamed on two different tokenizer constructions.
 */
async function loadTokenizer(modelId: string): Promise<ReturnType<typeof tokenizerFromEncoder>> {
  const suffix = `/${modelId}/tokenizer.json`;
  const tokenizerUrl = modelFileUrlOrThrow(modelId, "tokenizer.json");
  if (!tokenizerUrl.endsWith(suffix)) {
    throw new Error(`tokenizer URL ${tokenizerUrl} does not end with ${suffix}`);
  }
  const { AutoTokenizer, env: hfEnv } = await import("@huggingface/transformers");
  // `allowLocalModels` defaults to FALSE in a browser -- transformers.js derives
  // it from whether it is running under Node. MEASURED: without this line
  // from_pretrained throws "Invalid configuration detected: both local and
  // remote models are disabled", because the next line has already switched the
  // remote half off. Both flags have to move for a purely local load.
  hfEnv.allowLocalModels = true;
  hfEnv.allowRemoteModels = false;
  hfEnv.localModelPath = tokenizerUrl.slice(0, -suffix.length);
  const hf = await AutoTokenizer.from_pretrained(modelId);
  return tokenizerFromEncoder((text: string, addSpecialTokens: boolean) =>
    Array.from(
      hf(text, { add_special_tokens: addSpecialTokens }).input_ids.data as BigInt64Array,
      Number,
    ),
  );
}

interface Tier1State {
  readonly tagger: GlinerSpanTagger;
  readonly report: Tier1LoadReport;
  readonly release: () => Promise<void>;
}

let tier1: Tier1State | undefined;
let lastDetect: Tier1DetectStats | undefined;

const statsDelta = (
  before: Tier1TaggerStats,
  after: Tier1TaggerStats,
  gpu: number,
): Tier1DetectStats => ({
  inferences: after.inferences - before.inferences,
  droppedWords: after.droppedWords - before.droppedWords,
  truncatedWords: after.truncatedWords - before.truncatedWords,
  overWideSpans: after.overWideSpans - before.overWideSpans,
  unmappableSpans: after.unmappableSpans - before.unmappableSpans,
  nonFiniteScores: after.nonFiniteScores - before.nonFiniteScores,
  gpuSubmits: gpu,
});

/**
 * One `detect` through core, with the tier-1 counters read either side of it.
 *
 * Shared by the public `detect` and by the warm-up, so the warm-up exercises the
 * SAME path a measured call takes -- segmentation, the code-segment filter,
 * normalizeFindings and all -- rather than a shortcut that could succeed where
 * the real one would not.
 */
async function measuredDetect(
  request: DetectRequest,
  tagger: GlinerSpanTagger | undefined,
): Promise<{ result: DetectionResult; tier1: Tier1DetectStats | undefined }> {
  // Forwarded field by field rather than spread, so the set of things a spec can
  // influence is exactly the three fields of DetectRequest. `engines` stays
  // absent when no tagger is loaded, so core throws its own "tier1 enabled but
  // no tier-1 engine provided" rather than this page inventing a message for
  // the same mistake.
  const { provider, text, config } = request;
  if (tagger === undefined) {
    return { result: await detect({ ir, provider, text, config }), tier1: undefined };
  }
  const before = { ...tagger.stats };
  const gpuBefore = gpuSubmits;
  const result = await detect({ ir, provider, text, config, engines: { tier1: tagger } });
  return {
    result,
    tier1: config.tier1 ? statsDelta(before, tagger.stats, gpuSubmits - gpuBefore) : undefined,
  };
}

/**
 * Short, but real prose that survives segmentation and reaches the graph.
 *
 * The warm-up is not decoration: it is the only place the page can watch one
 * inference from both sides at once, which is what turns the requested backend
 * into an observed one. Nothing here depends on the model FINDING anything in
 * it -- `stats.inferences` counts `session.run` calls, so an empty result still
 * proves the graph executed. It should also amortise whatever first-call cost
 * each provider has out of the first measured `detect`, but how large that is
 * has not been measured either way.
 *
 * Deliberately contains no token from `fixtures/minimal-ir.json`'s entityType
 * examples. Nothing scores this string, so contamination is not possible today;
 * the point is that it stays impossible when a compiled policy replaces that
 * fixture.
 */
const WARMUP_TEXT = "Please renew the maintenance agreement before the quarter closes.";

async function loadTier1(options: Tier1LoadOptions): Promise<Tier1LoadReport> {
  const started = performance.now();
  // resolveTier1Config is the validator for every field, not a convenience: an
  // unknown modelId, a backend outside TIER1_BACKENDS, a threshold outside
  // (0, 1] all throw here naming what was available, instead of turning into a
  // 404 or an onnxruntime error two layers down. It also drops own keys holding
  // `undefined`, so an options object built positionally takes the defaults.
  const config = resolveTier1Config(options);
  const entry = MODEL_MANIFEST[config.modelId];
  if (entry === undefined) throw new Error(`unknown tier-1 model "${config.modelId}"`);

  const weightsUrl = modelFileUrlOrThrow(config.modelId, entry.weightsPath);
  // HEAD, not a full read: the point is only to report what the server says it
  // is about to serve, so a caller can check it against the manifest's pinned
  // size. onnxruntime fetches the body itself a few lines below.
  const head = await fetch(weightsUrl, { method: "HEAD" });
  if (!head.ok) {
    throw new Error(
      `tier-1 weights at ${weightsUrl} answered ${String(head.status)}; ${FETCH_MODELS_HINT}`,
    );
  }
  const weightsBytes = Number(head.headers.get("content-length") ?? "0");

  // Replacing an already-loaded rung: release first. Two live sessions would
  // hold two copies of the weights, and on webgpu two device contexts.
  if (tier1 !== undefined) {
    await tier1.release();
    tier1 = undefined;
    lastDetect = undefined;
  }

  const runtime = await loadOrtRuntime();
  const session = await createOrtSession(weightsUrl, config.backend, () => Promise.resolve(runtime));
  // AFTER the session exists, because @huggingface/transformers is imported by
  // loadTokenizer below and this is the assignment it would have made.
  const tokenizer = await loadTokenizer(config.modelId);
  // After loadTokenizer, because that is where @huggingface/transformers is
  // imported and its module body is what would have reassigned wasmPaths.
  assertLocalWasm(runtime);

  const tagger = new GlinerSpanTagger(session, tokenizer, config);
  const warmup = await measuredDetect(
    { text: WARMUP_TEXT, provider: "claude", config: { tier0: false, tier1: true, tier2: false } },
    tagger,
  );
  const warmupInferences = warmup.tier1?.inferences ?? 0;
  const warmupGpuSubmits = warmup.tier1?.gpuSubmits ?? 0;
  if (warmupInferences === 0) {
    await session.release();
    throw new Error(
      `${config.modelId}: the warm-up produced no inference, so nothing verified which ` +
        "execution provider ran (check that the IR declares at least one tier-1 entityType)",
    );
  }
  const observedBackend: Tier1Backend = warmupGpuSubmits > 0 ? "webgpu" : "wasm";
  if (observedBackend !== config.backend) {
    await session.release();
    throw new Error(
      `${config.modelId}: asked for the ${config.backend} backend but ${String(warmupGpuSubmits)} ` +
        `GPUQueue.submit calls over ${String(warmupInferences)} warm-up inference(s) say ` +
        `${observedBackend} executed; refusing to report ${config.backend} latency for ` +
        `${observedBackend} work`,
    );
  }

  const report: Tier1LoadReport = {
    config,
    observedBackend,
    weightsUrl,
    weightsBytes,
    loadMs: performance.now() - started,
    warmupInferences,
    warmupGpuSubmits,
  };
  tier1 = { tagger, report, release: () => session.release() };
  // Deliberately cleared: the warm-up is this page's own call, and a spec asking
  // "did tier 1 run during MY detect" must not be answered with it.
  lastDetect = undefined;
  return report;
}

const api: SihPageApi = {
  // The tier-1 engine is NOT something a spec can pass in: it is constructed by
  // `loadTier1` in the page, held module-level, and named in `tier1Status()`. An
  // engine crossing the evaluate boundary would be a stub by construction, and
  // the harness would report its latency as core's. What a spec CAN influence is
  // exactly the three fields of DetectRequest -- see measuredDetect.
  detect: async (request) => {
    // The one place both the loaded backend and the caller's TierConfig are
    // known. Nothing under packages/core/src reads `TierConfig.backend`, and
    // runArm copies its own `spec.backend` onto every record, so without this
    // check a record can carry `backend: "webgpu"` for a run served by wasm --
    // the substitution the observed/requested split above exists to prevent,
    // arriving through the other door.
    if (request.config.tier1 && tier1 !== undefined && request.config.backend !== undefined) {
      if (request.config.backend !== tier1.report.observedBackend) {
        throw new Error(
          `config.backend is ${request.config.backend} but the loaded tier-1 model is running on ` +
            `${tier1.report.observedBackend}`,
        );
      }
    }
    const { result, tier1: stats } = await measuredDetect(request, tier1?.tagger);
    if (stats !== undefined) lastDetect = stats;
    return result;
  },
  irHash: () => irHash,
  policyHash: () => ir.policyHash,
  backendAvailable,
  loadTier1,
  tier1Status: () =>
    tier1 === undefined
      ? undefined
      : { load: tier1.report, totals: { ...tier1.tagger.stats }, lastDetect },
};

// Non-writable and non-configurable, not just assigned. Reassigning
// `window.__sih` from a spec would redirect every later measurement in that
// worker to something that is not core, and silently.
Object.defineProperty(window, "__sih", {
  value: Object.freeze(api),
  writable: false,
  configurable: false,
});

// Not what the driver waits on -- `__sih` is -- but it makes a headed run and
// the only-on-failure screenshot say whether the module reached its end.
document.getElementById("status")!.textContent = "ready";
