import {
  detect,
  loadPolicyIr,
  type DetectionResult,
  type DetectorEngines,
  type PolicyIr,
  type TierConfig,
} from "@sih/core";
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
import {
  WebLlmJudge,
  createBaselineB,
  createBaselineBPlusTier0,
  createWebLlmEngine,
  resolveTier2Config,
  type BaselineB,
  type BaselineStats,
  type JudgeStats,
  type Tier2Config,
  type WebLlmEngine,
} from "@sih/tier2";
// Both extracted from this file so they can be exercised under vitest: this
// module is a Vite entry point, so anything declared inside it is reachable
// only through the browser suite -- and that suite cannot separate a delta from
// a total (one detect per page) nor an adapter below web-llm's floor from one
// above it (one machine). Each module's docblock carries the mutation that
// proved it.
import { baselineDelta, type BaselineDetectStats } from "./baseline-delta.js";
import { judgeDelta, type Tier2DetectStats } from "./judge-delta.js";
import { meetsWebLlmAdapterFloor } from "./webgpu-floor.js";
// Copied from packages/core/test/fixtures/minimal-ir.ts and owned by this app:
// core's test fixture is a TypeScript module, and importing it would both make
// a test-only artifact a runtime dependency of the harness and bypass the thing
// worth exercising here -- the extension receives a compiled IR as JSON text and
// parses it with loadPolicyIr, so the page must too. It is still a placeholder
// and not a baseline -- `policyHash: "test-hash"` -- but it is no longer the
// only kind of IR here: `p-fin` below is real compiler output, and this one
// stays because it is the default every tier-0 and tier-1 spec is written
// against and because ONE tier-1 entityType is a shape the compiled policy does
// not have.
import minimalIrJson from "../../fixtures/minimal-ir.json?raw";
// The SECOND fixture, and the reason it exists rather than a variation on the
// first. `minimal-ir.json` declares exactly ONE tier-1 entityType, so every
// browser assertion in this harness has run with `classes = 1`, where
// `buildLabels` only ever assigns classIndex 0 and `decodeEdgeSpans`'
// `(word * classes + classIndex) * slots` degenerates to `word * slots`. A
// class-axis stride bug is invisible at that width -- on the ONE path spec 2.2
// exists to protect, since the multi-class end-to-end test in
// packages/tier1/test/e2e.test.ts runs under onnxruntime-NODE. This file
// declares three, in a fixed order, so the stride is exercised in Chrome.
import multiclassIrJson from "../../fixtures/multiclass-ir.json?raw";
// The THIRD fixture, and the only one tier 2 can do anything with. Both of the
// others declare `semanticPredicates: []`, and on the config every tier-2 spec
// runs (`{tier0: false, tier1: false, tier2: true}`) that is enough to keep the
// judge out of the run entirely: `selectSegments` keeps a segment only when
// something below marked it uncertain or the IR HAS predicates, and with the
// lower tiers off neither holds -- so escalation selects nothing, the
// orchestrator never calls `judge()`, and `timings.tier2Ms` is never set.
// (VERIFIED against escalate.ts and orchestrator.ts, whose own test pins
// `tier2Ms` undefined on that path.) This file declares one predicate and the
// shadow entityType the compiler would mint for it, which is what makes an
// engine call happen at all.
//
// `latencyBudgetMs` is 120,000 here against 5,000 in the other two, and that is
// a deliberate difference rather than a copy that drifted. The orchestrator
// arms the message deadline from that field, Plan 5 measured 4.6 s for one
// tier-2 call on the CHEAPEST pinned arm, and at 5,000 the deadline would fire
// during the first call of every tier-2 spec: every run would exercise the
// abort path and nothing would exercise a completed judgement. It is a
// lifecycle fixture and the number is sized for that; it is NOT a claim that
// tier 2 fits a 5 s budget, and Task 9 measured that on this corpus it does not.
import semanticIrJson from "../../fixtures/semantic-ir.json?raw";
// The FOURTH IR, and the only one in this page that a compiler produced. The
// other three are hand-written fixtures whose `policyHash` is the literal string
// "test-hash", and that is what kept Approach B unrunnable: `planBakeoff`
// refuses to pair a B arm with a compiled arm unless the IR's `policyHash` is
// the sha256 of a document this repository holds, and no hand-written fixture
// can satisfy that for any real file. This one carries
// ebb3cd68d973175f3ea40faeec00685e2cb9d83c6940e96a57e88ee269e8110a, which is
// `shasum -a 256 policies/p-fin.md`.
//
// It is real compiler output -- extraction, grounding, predicate minting, regex
// validation, the tier-0 self-test and emission all ran -- from the OFFLINE
// path: `scripts/compile-policies.ts` replays the committed LLM fixtures rather
// than calling the API, and `packages/compiler/test/compiled.test.ts` recompiles
// it on every suite run and requires byte equality. What it is NOT is a live
// frontier-model compile; the model responses behind it are the hand-authored
// stand-ins `scripts/record-fixtures.ts` describes, and Plan 5 defers the live
// pass.
//
// `latencyBudgetMs` is 5,000 here -- the compiler's default, since p-fin.md
// states no budget -- against `semantic-ir.json`'s 120,000. That is a 24x
// difference in what a tier-2 arm is allowed to spend per message and it is NOT
// a fixture that drifted: this is the number a shipped policy carries, and Plan
// 5 measured one tier-2 call at 4.6 s on the cheapest pinned arm. So an arm run
// against THIS IR will degrade on the budget, loudly, in `degraded`. That is the
// true number and it stays.
import pFinIrJson from "../../../../policies/compiled/p-fin.ir.json?raw";
// The policy DOCUMENT, which only Approach B is shown. Imported beside the IR it
// was compiled from, because the pairing is the whole premise of the arm: B
// reads the prose, the compiled arm reads the IR, and shown two different
// policies they compare two policies rather than two methods. `loadBaseline`
// refuses a pairing whose sha256 does not match the loaded IR's `policyHash`,
// so the two cannot come apart inside this page either.
import pFinPolicyText from "../../../../policies/p-fin.md?raw";
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
 * Which model to load and under what settings.
 *
 * Everything is optional because `resolveTier2Config` is both the filler and
 * the VALIDATOR: it refuses a modelId outside `TIER2_MODELS`, a non-zero
 * temperature, and a non-integer or non-positive `maxTokens` or
 * `contextWindowSize`, naming what was available. Passing the whole
 * `Partial<Tier2Config>` through rather than a hand-picked subset is what keeps
 * "the settings a caller can vary" and "the settings the report states" the
 * same set -- the tier-1 options above take the same shape for the same reason.
 */
export type Tier2LoadOptions = Partial<Tier2Config> & {
  /**
   * The judge's per ENGINE CALL budget, in ms. One segment's budget is not
   * spent by the segment before it; `ir.latencyBudgetMs` is the per-MESSAGE
   * bound and the orchestrator holds that one.
   *
   * Defaulted here rather than in @sih/tier2 because there is no defensible
   * package-level value: it has to be larger than the slowest legitimate call
   * on the slowest pinned arm or every arm degrades, and that number is what
   * the bake-off is FOR. `DEFAULT_TIER2_CALL_BUDGET_MS` is a ceiling for
   * lifecycle work, not a latency target -- see it.
   */
  readonly callBudgetMs?: number;
};

/**
 * What a tier-2 load actually did.
 *
 * The asymmetry with `Tier1LoadReport` is deliberate and is the whole design
 * question this type had to answer. There, `observedBackend` is a MEASUREMENT
 * (GPU submits counted around a real inference) standing beside the requested
 * `config.backend`. Tier 2 has no equivalent for the context window: 0.2.84
 * exposes NO accessor for the chat config an engine loaded -- `MLCEngineInterface`
 * declares none and `MLCEngine.loadedModelIdToPipeline` is private (READ from the
 * installed bundle) -- so `config.contextWindowSize` here is what was ASKED for
 * and nothing in this report upgrades it to what ran. `tier2.spec.ts` measures
 * the effective window per arm through the only channel the library has, which
 * is its own refusal of an over-long prompt; see `probeContextWindow`.
 *
 * What IS observed is `servedModelId`, and `loadTier2` refuses to return a
 * report without it.
 */
export interface Tier2LoadReport {
  /**
   * The FULLY RESOLVED config this engine is running under -- every field, not
   * just the ones the caller named. What was REQUESTED, in every field: see the
   * type's own note above for why no field here is an observation.
   */
  readonly config: Tier2Config;
  /**
   * The judge's per-call budget, as resolved. Not part of `Tier2Config` because
   * it is not a property of the loaded model; recorded here because an arm's
   * `deadlineExpiries` count means nothing without it.
   */
  readonly callBudgetMs: number;
  /**
   * `ChatCompletion.model` from the warm-up call: which loaded pipeline the
   * engine says served it.
   *
   * NARROWER than "which weights ran", and the honest reading is the one
   * `engine.ts` traced through the bundle: `chatCompletion` assigns
   * `model: selectedModelId`, `getModelIdToUse` picks it from the keys of
   * `loadedModelIdToPipeline`, and those keys are the strings handed to
   * `CreateMLCEngine`. It is the requested id laundered through a Map key.
   *
   * It is still the strongest thing available, and it is not the requested id
   * copied into a new field: it comes back only if the engine ANSWERED, it
   * follows a `reload()` behind our back, and `getModelIdToUse` throws rather
   * than guessing when several models are loaded. Reporting
   * `config.modelId` here instead would be the intent-recorded-as-fact defect
   * this project has shipped twice -- and it would still read "Qwen3.5-2B" for
   * a browser whose GPU process had died.
   */
  readonly servedModelId: string;
  /** Wall clock for `CreateMLCEngine` alone: fetch or cache read, then compile. */
  readonly loadMs: number;
  /** Wall clock for the warm-up completion alone. */
  readonly warmupMs: number;
  /**
   * `choices[0].finish_reason` for the warm-up. `undefined` is reachable: the
   * declared type is narrower than the bundle's own `getFinishReason()`.
   */
  readonly warmupFinishReason: string | undefined;
  /** `usage.completion_tokens` for the warm-up; `undefined` when the engine reported no usage. */
  readonly warmupCompletionTokens: number | undefined;
  /**
   * `navigator.storage.estimate()` AFTER the load, in bytes.
   *
   * Here because the failure this whole task is shaped around is a storage one:
   * MEASURED on this machine, an ephemeral Playwright context
   * (`browser.newContext()`) reports a 3,221 MB quota where a persistent
   * profile reports 10,737 MB empty and 18,230 MB holding the four pinned arms
   * -- which are 7.49 GB of weights at this one origin. A `QuotaExceededError`
   * mid-download looks exactly like a model defect, so the number that explains
   * it travels with the load.
   *
   * `?? 0` on both: `StorageEstimate` declares them optional. MEASURED, Chrome
   * has populated both on every load this suite has run, and the default fails
   * in the safe direction -- a 0 quota fails `tier2.spec.ts`'s check rather
   * than reading as a large one.
   */
  readonly storageUsageBytes: number;
  readonly storageQuotaBytes: number;
}

// Declared in `judge-delta.ts` beside the function that builds one, and
// re-exported here because this file is the page's public shape.
export type { Tier2DetectStats };

export interface Tier2Status {
  readonly load: Tier2LoadReport;
  /** The judge's counters since it was constructed, plus every call row. */
  readonly totals: JudgeStats;
  /**
   * The most recent `detect` that ran tier 2, as deltas. `undefined` until one
   * has. Same job as `Tier1Status.lastDetect`: findings alone cannot separate
   * "the model ran and found nothing" from "the model never ran".
   *
   * A timing narrows that but does not settle it, and the difference is why
   * this field exists. `orchestrator.ts` sets `timings.tier2Ms` only on the
   * branch that CALLS the judge, so an absent one does say the judge never ran
   * -- but a present one says the JUDGE ran, not that the ENGINE was asked
   * anything. Two live paths set it without a model call: an IR declaring no
   * `semanticPredicates` whose segments escalation still selected (a lower
   * tier left one uncertain), where `WebLlmJudge.judge` returns an empty
   * verdict before touching the engine; and a first call the caller had
   * already aborted, which the judge counts and files without a call row.
   * `calls` is what separates those from a model that answered.
   */
  readonly lastDetect: Tier2DetectStats | undefined;
}

// Declared in `baseline-delta.ts` beside the function that builds one, for the
// reason `Tier2DetectStats` is declared in `judge-delta.ts`.
export type { BaselineDetectStats };

/** Which Approach-B arm to build. The two differ only in whether core's tier 0 runs. */
export type BaselineFamily = "baseline-b" | "baseline-b-tier0";

export interface BaselineLoadOptions {
  readonly family: BaselineFamily;
  /** A key of `POLICY_FIXTURES`. The document B is shown, verbatim and entire. */
  readonly policy: string;
}

/**
 * What an Approach-B load actually did, as opposed to what it was asked to do.
 *
 * `policyDocSha256` and `irPolicyHash` are BOTH here and are the same value on
 * every successful load, which looks redundant and is not: the load is refused
 * when they differ, so carrying both is what lets a driver read the pairing off
 * the report rather than trusting that the refusal exists. `policyChars` is the
 * one number that says how much bigger B's prompt is than the compiled judge's
 * -- the judge is shown one segment, B is shown this whole document on every
 * call -- and the p95 TTFT gate's ceiling was derived at ~1.1 kB.
 */
export interface BaselineLoadReport {
  readonly family: BaselineFamily;
  readonly policy: string;
  /** sha256 of the document text this page holds, lowercase hex. */
  readonly policyDocSha256: string;
  /** The loaded IR's `policyHash`, verbatim. Equal to the above or the load threw. */
  readonly irPolicyHash: string;
  /** UTF-16 code units of policy text in every Approach-B prompt. */
  readonly policyChars: number;
  /**
   * The per ENGINE CALL budget this arm holds, which is the judge's own
   * `callBudgetMs` off the tier-2 load report and never a number of this
   * function's choosing. Two arms at different per-call budgets measure the
   * deadline rather than the method.
   */
  readonly callBudgetMs: number;
  /** The tier-2 engine this arm runs on, from `loadTier2`'s observed id. */
  readonly servedModelId: string;
}

export interface BaselineStatus {
  readonly load: BaselineLoadReport;
  /** The arm's counters since it was constructed, plus every call row. */
  readonly totals: BaselineStats;
  /**
   * The most recent `detect` this arm ran, as deltas. `undefined` until one has.
   *
   * Same job as `Tier2Status.lastDetect` and one caveat lighter: B makes one
   * call per message unconditionally, so unlike the judge there is no path
   * where it returns a verdict without touching the engine. A `lastDetect`
   * whose `calls` is empty therefore means the call was stopped -- by the
   * per-call budget, by the message budget, or by an abort -- rather than never
   * attempted, and the counters say which.
   */
  readonly lastDetect: BaselineDetectStats | undefined;
}

/** A `detect` run under a per-call budget this page chooses, with that run's judge stats. */
export interface BudgetedDetectRequest extends DetectRequest {
  /**
   * The per ENGINE CALL budget for this run only, in ms. Not
   * `ir.latencyBudgetMs`: that one is the message's and belongs to the IR, and
   * conflating the two is what `call-budget-exhausted` and `budget-exhausted`
   * exist as separate reason words to prevent.
   */
  readonly callBudgetMs: number;
}

export interface BudgetedDetectResult {
  readonly result: DetectionResult;
  /**
   * The stats of the judge that ran THIS call, which is a fresh one -- so these
   * are already this call's numbers and not a delta. The arm's judge is left
   * untouched, deliberately: a deliberately-blown budget is not part of an
   * arm's totals.
   */
  readonly stats: JudgeStats;
}

/**
 * What the engine did with a prompt too long for a 4096-token window.
 *
 * The one channel 0.2.84 has for reporting the context window it is actually
 * enforcing. READ from the installed bundle: `LLMChatPipeline.getInputTokens`
 * throws `ContextWindowSizeExceededError` when
 * `numPromptTokens + filledKVCacheLength > contextWindowSize`, and the message
 * names both numbers.
 */
export interface Tier2WindowProbe {
  /** What the loaded engine was asked for, so a reader can see both halves. */
  readonly requestedContextWindowSize: number;
  readonly promptWords: number;
  /** True when the engine prefilled the prompt and answered. */
  readonly accepted: boolean;
  /**
   * `usage.prompt_tokens` -- the ENGINE's count of the prompt it accepted, not
   * a count taken here. `undefined` when the call was refused, since there is
   * then no response to read one from.
   */
  readonly promptTokens: number | undefined;
  readonly finishReason: string | undefined;
  readonly servedModelId: string | undefined;
  /** `ContextWindowSizeExceededError` when the window refused the prompt. */
  readonly errorName: string | undefined;
  readonly errorMessage: string | undefined;
  readonly ms: number;
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
   * Select which of this page's pinned IR fixtures `detect` runs against, and
   * get the new `irHash` back.
   *
   * `"minimal"` (the default) has ONE tier-1 entityType and `"multiclass"` has
   * three. That difference is the reason this method exists: at one tier-1
   * class the model's class axis has extent 1, `buildLabels` only ever assigns
   * classIndex 0, and both decoders' class stride multiplies by zero -- so
   * every stride bug in `decodeEdgeSpans` and `decodeBaseSpans` reads the same
   * cell as a correct implementation would. Without a way to widen that axis
   * IN THE BROWSER, the only multi-class coverage in the repo runs under
   * onnxruntime-node, which spec 2.2 says is not the measured runtime.
   *
   * Returns the digest rather than void so a caller cannot forget that the
   * record's provenance moved with it.
   */
  useIr(name: string): Promise<string>;
  /**
   * sha256 of one of this page's pinned policy DOCUMENTS, lowercase hex.
   *
   * The `irHash` of the other half of a compile. A driver plans an Approach-B
   * arm from a document it read off disk and the page shows B the document it
   * bundled, and nothing else in the pipeline compares the two -- so a bake-off
   * could plan against `policies/p-fin.md` while the page served a stale copy,
   * and every record would look correct. `shasum -a 256 policies/p-fin.md`
   * reproduces this.
   */
  policyDocHash(name: string): Promise<string>;
  /**
   * Build an Approach-B arm on the loaded tier-2 engine and make `detect` use
   * it INSTEAD of core's orchestrator.
   *
   * This is the door Approach B had no other way through. `detect` is core's
   * orchestrator with a `WebLlmJudge` behind it; `createBaselineB` is a
   * `Detector` in its own right -- no compiler, no tiers, the whole policy and
   * the whole message in one call -- so no amount of `TierConfig` reaches it.
   *
   * Requires `loadTier2` first, and takes the engine, the resolved
   * `Tier2Config` and the per-call budget from that load rather than from this
   * caller. That is the single biggest fairness lever available here: the model
   * id, the context window, the temperature, the token ceiling, the
   * grammar-constrained decoding path and the per-call deadline are then the
   * same OBJECT the compiled arm ran on, not the same numbers by convention.
   *
   * Refuses a policy document whose sha256 is not the loaded IR's `policyHash`.
   * B's entire premise is being shown the document the compiled arm's IR came
   * from; shown a different one it wins or loses for a reason invisible in every
   * number the run produces.
   */
  loadBaseline(options: BaselineLoadOptions): Promise<BaselineLoadReport>;
  /** `undefined` until `loadBaseline` has succeeded. */
  baselineStatus(): BaselineStatus | undefined;
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
  /**
   * Whether a tier-2 engine can START here, asked of the adapter's LIMITS and
   * not merely of `navigator.gpu`. Separate from `backendAvailable("webgpu")`
   * because web-llm refuses adapters onnxruntime-web quietly falls back from --
   * see the implementation for the four limits and which have no fallback.
   *
   * A false here means tier 2 is ABSENT on this machine, which is the
   * orchestrator's own word for a tier that did not run. It is not a
   * degradation and a spec's skip should say so.
   */
  webgpuAvailable(): Promise<boolean>;
  /**
   * Load one tier-2 model into this page and make `detect` use it.
   *
   * Returns a report for the reason `loadTier1` does -- everything worth
   * asserting about a load is measured inside the page -- and runs one
   * throwaway completion before returning, which is what makes `servedModelId`
   * a fact rather than an echo of the request.
   */
  loadTier2(options: Tier2LoadOptions): Promise<Tier2LoadReport>;
  /** `undefined` until `loadTier2` has succeeded. */
  tier2Status(): Tier2Status | undefined;
  /**
   * Release the loaded tier-2 engine and everything built on it.
   *
   * `loadTier2` already unloads before it replaces an arm, and its docblock
   * gives the reason with the numbers: two live engines hold two copies of the
   * weights on one GPU and the two largest pinned arms are 3,432 and 3,438 MB.
   * The same argument applies BETWEEN arms of a bake-off, where `runBakeoff`
   * navigates rather than reloads -- and navigation destroys the JS context
   * without promising the GPU allocation goes with it. That promise is the part
   * nothing here can make: no measurement in this repository says when Chrome
   * releases a `GPUDevice` whose page has gone away, so this is the explicit
   * release rather than a claim about what the implicit one does.
   *
   * A no-op when nothing is loaded, so a caller need not track that.
   */
  unloadTier2(): Promise<void>;
  /**
   * One `detect` under a per-CALL budget of this caller's choosing, plus the
   * judge stats that run produced. The loaded engine is reused, so what a tiny
   * budget tests is whether that engine survives being interrupted.
   */
  detectWithBudget(request: BudgetedDetectRequest): Promise<BudgetedDetectResult>;
  /**
   * Prefill a prompt of `words` filler words on the loaded engine and report
   * what the engine did with it.
   *
   * The only channel 0.2.84 has for reporting the context window it is
   * enforcing: there is no accessor for an engine's effective chat config, so
   * an over-long prompt and the library's own refusal are the measurement.
   */
  probeContextWindow(options: { words: number; budgetMs: number }): Promise<Tier2WindowProbe>;
  /**
   * The `apps/eval` directory of the checkout whose Vite server built this page.
   *
   * Provenance, and it exists because nothing else in this harness had any.
   * `playwright.config.ts` sets `reuseExistingServer: !CI`, so a dev server left
   * running on 5178 by another worktree serves the page every tier-2 number is
   * measured on -- quietly, and uniformly across arms, so it shows up as
   * numbers attributed to the wrong code rather than as a discrepancy anyone
   * would notice. `irHash` covers the IR ARTIFACT and only that; this covers the
   * code.
   *
   * Compiled in by `vite.config.ts` as a `define`, so it is that config file's
   * own directory rather than anything the browser or the driver could compute.
   */
  harnessDir(): string;
}

/**
 * Injected by `vite.config.ts`'s `define`, which substitutes the literal into
 * the source text at transform time.
 *
 * So a page served by a config without that `define` keeps the bare identifier,
 * and `harnessDir()` throws a `ReferenceError` when it is called rather than
 * returning something wrong. Called is the operative word -- the reference is
 * inside an arrow function, so the module still evaluates and `__sih` still
 * appears; the failure lands on the provenance check in `openHarness`, which is
 * where a harness that cannot say which tree served it should fail.
 */
declare const __SIH_HARNESS_DIR__: string;

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

/**
 * Every IR this page can run, by name.
 *
 * A NAMED REGISTRY of `?raw` fixtures rather than a `loadIr(json)` that takes
 * arbitrary text, and the difference is `irHash`'s whole value. A record's
 * `irHash` is checkable from outside the browser -- `shasum -a 256` on a file
 * in this repo reproduces it -- and a spec that could inject IR text would make
 * half the records in a run name an artifact nobody can produce. `p-fin` is
 * that compiled policy, added as one line, and it keeps the property: its
 * digest is `shasum -a 256 policies/compiled/p-fin.ir.json`.
 *
 * Three of the four are hand-written fixtures whose `policyHash` is the literal
 * string "test-hash". Only `p-fin` can be paired with a policy DOCUMENT, which
 * is what an Approach-B arm needs -- see `POLICY_FIXTURES` and `loadBaseline`.
 */
const IR_FIXTURES: Readonly<Record<string, string>> = {
  minimal: minimalIrJson,
  multiclass: multiclassIrJson,
  semantic: semanticIrJson,
  "p-fin": pFinIrJson,
};

/**
 * Every policy DOCUMENT this page can show an Approach-B arm, by name.
 *
 * A named registry for exactly the reason `IR_FIXTURES` is one: `policyDocHash`
 * is checkable from outside the browser with `shasum -a 256`, and a
 * `loadBaseline(policyText)` taking arbitrary text would let a spec show B a
 * document nobody can reproduce. The names are the same names, so `"p-fin"`
 * here and `"p-fin"` there are the two halves of one compile -- and
 * `loadBaseline` checks that rather than trusting it.
 *
 * Only p-fin, because only p-fin has a compiled IR: `policies/p-med.md` and
 * `policies/p-corp.md` are in the repository, but no committed LLM fixture
 * answers their compile prompts (MEASURED -- `scripts/compile-policies.ts`
 * reports both as needing a live pass), so pairing either with an IR is
 * impossible today and a document here without one would be a door onto a
 * comparison that cannot be made.
 */
const POLICY_FIXTURES: Readonly<Record<string, string>> = {
  "p-fin": pFinPolicyText,
};

export type IrName = keyof typeof IR_FIXTURES & string;

/** The default, so every existing spec and every record keeps the IR it had. */
const DEFAULT_IR: IrName = "minimal";

const digestHex = async (text: string): Promise<string> =>
  Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");

// BOTH fixtures are parsed at module scope, not just the selected one, so a
// malformed IR fails BEFORE `__sih` is published rather than on the first spec
// that happens to select it. The cost is that the failure reaches the driver
// only as a pageerror -- `openHarness` in smoke.spec.ts listens for exactly
// that and re-throws it with the message.
const PARSED_IRS: Readonly<Record<string, PolicyIr>> = Object.fromEntries(
  Object.entries(IR_FIXTURES).map(([name, json]) => [name, loadPolicyIr(json)]),
);

let irJson: string = IR_FIXTURES[DEFAULT_IR] as string;
let ir: PolicyIr = PARSED_IRS[DEFAULT_IR] as PolicyIr;

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
let irHash: Promise<string> = digestHex(irJson);

/**
 * Switch this page to another pinned IR, and re-derive the digest with it.
 *
 * The digest is recomputed rather than left alone, which is the only part of
 * this that can go wrong quietly: a record stamped with the old `irHash` while
 * the new IR produced the findings is exactly the unfalsifiable provenance the
 * two-hash split exists to prevent.
 *
 * Any tier-1 model already loaded stays loaded and is REUSED, deliberately.
 * `buildLabels` reads `ir.entityTypes` on every `tag` call, so the same graph
 * answers under the new label set -- which is the point: the class axis widens
 * without reloading 665 MB. What does NOT follow the switch is `loadTier1`'s
 * warm-up, which ran under whatever IR was active then; nothing reads it after
 * the load returns.
 */
async function useIr(name: string): Promise<string> {
  if (!Object.hasOwn(IR_FIXTURES, name)) {
    throw new Error(
      `unknown IR "${name}"; this page carries ${Object.keys(IR_FIXTURES).join(", ")}`,
    );
  }
  irJson = IR_FIXTURES[name] as string;
  ir = PARSED_IRS[name] as PolicyIr;
  irHash = digestHex(irJson);
  // A baseline arm holds a policy document paired with the IR that was loaded
  // when it was built, and switching IRs breaks that pairing silently: the arm
  // would keep showing its model p-fin's prose while `detect` resolved actions
  // against another policy's entityTypes. Dropped rather than re-checked,
  // because re-checking would make `useIr` succeed or fail depending on which
  // arm happens to be loaded, and a driver navigates a fresh page per arm
  // anyway.
  baseline = undefined;
  lastBaselineDetect = undefined;
  return irHash;
}

/** sha256 of one pinned policy document. See `SihPageApi.policyDocHash`. */
async function policyDocHash(name: string): Promise<string> {
  if (!Object.hasOwn(POLICY_FIXTURES, name)) {
    throw new Error(
      `unknown policy "${name}"; this page carries ${Object.keys(POLICY_FIXTURES).join(", ")}`,
    );
  }
  return digestHex(POLICY_FIXTURES[name] as string);
}

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

/**
 * WebGPU has no lib.dom typings here; naming the members used is cheaper than
 * the dependency. `limits` is indexed by name rather than declared field by
 * field because `webgpuAvailable` walks a table of them, and every value is
 * optional so a browser missing one reads as "does not clear the floor" rather
 * than as `undefined >= n`, which is false but for the wrong reason.
 */
type WebGpuAdapterLike = { readonly limits: Readonly<Partial<Record<string, number>>> };
type MaybeWebGpu = { gpu?: { requestAdapter(): Promise<WebGpuAdapterLike | null> } };

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
  // The concrete judge, not the `SemanticJudge` seam core takes: the deltas
  // below read `stats`, which is this class's and is not on that interface.
  judge: WebLlmJudge | undefined,
): Promise<{
  result: DetectionResult;
  tier1: Tier1DetectStats | undefined;
  tier2: Tier2DetectStats | undefined;
}> {
  // Forwarded field by field rather than spread, so the set of things a spec can
  // influence is exactly the three fields of DetectRequest. An engine a tier
  // needs but this page has not loaded stays ABSENT from `engines`, so core
  // throws its own "tierN enabled but no tier-N engine provided" rather than
  // this page inventing a message for the same mistake -- an empty object still
  // takes that path, since the guard tests the field and not the object.
  const { provider, text, config } = request;
  // The TIER-2 twin of `detect`'s backend check, and it lives here rather than
  // beside that one because this is where both doors onto the loaded engine
  // meet: `detect` and `detectWithBudget`, the second of which builds a fresh
  // judge over the SAME engine and would have walked past a check written up
  // there.
  //
  // `TierConfig.t2Model` is an experiment variable the bake-off varies per arm
  // and `runArm` copies the arm's own config onto every record, so without this
  // a record can name one model for a detect the engine holding another served
  // -- the substitution the observed/requested split exists to prevent, one tier
  // up from where it was first closed.
  //
  // `servedModelId`, not `report.config.modelId`: the first came off a real
  // completion in `loadTier2` and follows a `reload()` behind our back, the
  // second is only what this page was asked for. The observed one is what a
  // record has to agree with.
  //
  // Today `runBakeoff` also refuses this one level up, once per arm at load
  // time, so for that driver this is a second lock on a closed door. It is the
  // only lock for every other caller of the page API -- a manual
  // `page.evaluate` session, Plan 6's extension harness, or a driver that loads
  // once and varies `config.t2Model` per item.
  if (config.tier2 && judge !== undefined && tier2 !== undefined && config.t2Model !== undefined) {
    if (config.t2Model !== tier2.report.servedModelId) {
      throw new Error(
        `config.t2Model is "${config.t2Model}" but the loaded tier-2 engine answered as ` +
          `"${tier2.report.servedModelId}"; refusing to report findings for one model under ` +
          `the other's name`,
      );
    }
  }
  const engines: DetectorEngines = {};
  if (tagger !== undefined) engines.tier1 = tagger;
  if (judge !== undefined) engines.tier2 = judge;

  const tier1Before = tagger === undefined ? undefined : { ...tagger.stats };
  const gpuBefore = gpuSubmits;
  // A snapshot, and `WebLlmJudge.stats` already returns one: it copies the
  // counters and rebuilds the call array on every read, so this cannot be a
  // live view of the numbers the delta is taken against.
  const tier2Before = judge === undefined ? undefined : judge.stats;

  const result = await detect({ ir, provider, text, config, engines });

  return {
    result,
    tier1:
      config.tier1 && tagger !== undefined && tier1Before !== undefined
        ? statsDelta(tier1Before, tagger.stats, gpuSubmits - gpuBefore)
        : undefined,
    tier2:
      config.tier2 && judge !== undefined && tier2Before !== undefined
        ? judgeDelta(tier2Before, judge.stats)
        : undefined,
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

  // EVERYTHING from here to the assignment at the bottom runs inside this try,
  // and the reason is arithmetic. Between the line above and the only
  // `release()` there were four throw paths with no cleanup on them --
  // `loadTokenizer` (a missing or unparseable tokenizer.json), `assertLocalWasm`
  // (transformers.js having reassigned wasmPaths), the GlinerSpanTagger
  // constructor (a signature mismatch), and the warm-up `detect` itself -- and
  // an onnxruntime session holds the whole graph. For `gliner-pii-base` that is
  // 665 MB abandoned per failed attempt, in a page that `test/tier1.spec.ts`
  // deliberately drives through repeated failing loads. The two explicit
  // releases that were already here are now redundant with this and have been
  // folded into it, so there is exactly one place that lets go of the session.
  //
  // Rethrows rather than swallowing: the caller's error is the diagnosis, and
  // the release is only bookkeeping on the way out.
  let report: Tier1LoadReport;
  let tagger: GlinerSpanTagger;
  try {
    // AFTER the session exists, because @huggingface/transformers is imported by
    // loadTokenizer below and this is the assignment it would have made.
    const tokenizer = await loadTokenizer(config.modelId);
    // After loadTokenizer, because that is where @huggingface/transformers is
    // imported and its module body is what would have reassigned wasmPaths.
    assertLocalWasm(runtime);

    tagger = new GlinerSpanTagger(session, tokenizer, config);
    const warmup = await measuredDetect(
      { text: WARMUP_TEXT, provider: "claude", config: { tier0: false, tier1: true, tier2: false } },
      tagger,
      // No judge: this warm-up exists to observe which execution provider ran
      // tier 1, and handing it a tier-2 engine would spend a model call on it.
      undefined,
    );
    const warmupInferences = warmup.tier1?.inferences ?? 0;
    const warmupGpuSubmits = warmup.tier1?.gpuSubmits ?? 0;
    if (warmupInferences === 0) {
      throw new Error(
        `${config.modelId}: the warm-up produced no inference, so nothing verified which ` +
          "execution provider ran (check that the IR declares at least one tier-1 entityType)",
      );
    }
    const observedBackend: Tier1Backend = warmupGpuSubmits > 0 ? "webgpu" : "wasm";
    if (observedBackend !== config.backend) {
      throw new Error(
        `${config.modelId}: asked for the ${config.backend} backend but ${String(warmupGpuSubmits)} ` +
          `GPUQueue.submit calls over ${String(warmupInferences)} warm-up inference(s) say ` +
          `${observedBackend} executed; refusing to report ${config.backend} latency for ` +
          `${observedBackend} work`,
      );
    }

    report = {
      config,
      observedBackend,
      weightsUrl,
      weightsBytes,
      loadMs: performance.now() - started,
      warmupInferences,
      warmupGpuSubmits,
    };
  } catch (cause) {
    // Awaited, not fire-and-forget: a rejected release would otherwise be an
    // unhandled rejection reaching Playwright's `pageerror` listener and
    // reported as the failure instead of the real one. If it does reject, that
    // is the more interesting fact and it carries the original as `cause`.
    await session.release().catch((releaseFailure: unknown) => {
      throw new Error(
        `${config.modelId}: failed to load, and releasing the session then failed too ` +
          `(${String(releaseFailure)})`,
        { cause },
      );
    });
    throw cause;
  }
  tier1 = { tagger, report, release: () => session.release() };
  // Deliberately cleared: the warm-up is this page's own call, and a spec asking
  // "did tier 1 run during MY detect" must not be answered with it.
  lastDetect = undefined;
  return report;
}

// -- tier 2 ------------------------------------------------------------------

/**
 * The per-call budget a load takes when the caller names none.
 *
 * A CEILING for lifecycle work, not a latency target, and deliberately far
 * above any latency this project wants: `runWithDeadline` interrupts the engine
 * when it expires, so a budget below a legitimate call turns every segment into
 * a `call-budget-exhausted` notice and an arm measures its own timeout instead
 * of its model. The bake-off's real number is Task 12's to choose and to
 * record, which is why this one is only a default and is reported on every load.
 */
const DEFAULT_TIER2_CALL_BUDGET_MS = 60_000;

/**
 * Whether a tier-2 engine can start here, asked of the adapter's LIMITS.
 *
 * The limits half lives in `webgpu-floor.ts`, where both directions of the
 * comparison can be driven from synthetic limit objects; this function is the
 * browser half that has to exist here -- getting an adapter, and treating a
 * null one as absent. `test/tier2.spec.ts` asserts the two agree on the real
 * adapter, which is the wiring a Node test cannot reach.
 *
 * The consequence for a spec is the whole point: `test.skip(!webgpuAvailable())`
 * must skip on exactly the machines where a load would throw. Using
 * `backendAvailable("webgpu")` instead would let a machine that clears
 * `requestAdapter()` but sits below web-llm's floor RUN the tier-2 specs, which
 * would then fail on a load throw -- reporting a missing GPU capability as a
 * broken harness.
 */
async function webgpuAvailable(): Promise<boolean> {
  const { gpu } = navigator as Navigator & MaybeWebGpu;
  if (gpu === undefined) return false;
  const adapter = await gpu.requestAdapter();
  if (adapter === null) return false;
  // `GPUSupportedLimits` is an interface, not an index signature, so its
  // members are read through the same string keys the floor table is written
  // in. The cast names that rather than widening the adapter type.
  return meetsWebLlmAdapterFloor(adapter.limits as unknown as Record<string, number | undefined>);
}

/**
 * The warm-up's grammar: one boolean, and nothing else legal.
 *
 * NOT the judge's schema, which is the schema every real call uses. The warm-up
 * exists to prove the engine answers, and under the judge's grammar a model is
 * free to spend the whole `maxTokens` budget on a findings array: at the 24 to
 * 40 tokens/sec `TIER2_MODELS` records for these arms, 512 tokens is 13 to 21
 * seconds paid on every load for an answer nobody reads. One required boolean
 * bounds the body instead -- MEASURED across the four arms, the warm-up
 * completes in 0.4 to 2.3 s and Qwen3.5-2B spends 5 completion tokens on it.
 * The judge's own grammar is compiled and exercised by the first real call, so
 * nothing is skipped by not compiling it here.
 */
const TIER2_WARMUP_SCHEMA_JSON = JSON.stringify({
  type: "object",
  properties: { ready: { type: "boolean" } },
  required: ["ready"],
  additionalProperties: false,
});

/**
 * The warm-up turn. Deliberately carries no policy-shaped content: nothing
 * scores it, and the point is that it stays impossible to score when a compiled
 * policy replaces these fixtures.
 */
const TIER2_WARMUP_MESSAGES = [
  { role: "user", content: 'Reply with {"ready": true} and nothing else.' },
] as const;

interface Tier2State {
  readonly engine: WebLlmEngine;
  /** The ARM's judge: its counters accumulate across every `detect` on this load. */
  readonly judge: WebLlmJudge;
  readonly report: Tier2LoadReport;
}

let tier2: Tier2State | undefined;
let lastTier2Detect: Tier2DetectStats | undefined;

/**
 * Load one tier-2 model into this page and make `detect` use it.
 *
 * The throwaway completion in the middle is the part worth defending. Without
 * it this function can only report what it ASKED for: 0.2.84 has no accessor
 * for what an engine loaded, so a report built from the config alone would be
 * satisfied by an engine that compiled its shaders and can no longer answer --
 * which is precisely the failure Plan 4 hit, where a DEAD browser produced a
 * complete, schema-valid output file. `servedModelId` is a string this page
 * cannot produce without a response, so a load cannot report success against an
 * engine that did not answer.
 *
 * What that check is NOT: an attestation of which weights ran. `engine.ts`
 * traced `ChatCompletion.model` through the bundle to the key of
 * `loadedModelIdToPipeline`, i.e. the id we handed to `CreateMLCEngine`. It
 * catches a `reload()` behind our back and it catches a dead engine; it cannot
 * catch a mislabelled checkpoint on HuggingFace, and nothing available here can.
 */
async function loadTier2(options: Tier2LoadOptions): Promise<Tier2LoadReport> {
  // The validator for every model field, not a convenience: an unknown modelId,
  // a non-zero temperature, a non-integer window or token budget all throw here
  // naming what was available, rather than reaching the library. `manifest.ts`
  // records what each of those costs when it does.
  const config = resolveTier2Config(options);
  const callBudgetMs = options.callBudgetMs ?? DEFAULT_TIER2_CALL_BUDGET_MS;

  // Replacing an already-loaded arm: unload FIRST. Two live engines hold two
  // copies of the weights on one GPU, and the two largest pinned arms are 3,432
  // and 3,438 MB -- `WebLlmEngine.unload` exists for exactly this and its
  // docblock carries the arithmetic.
  if (tier2 !== undefined) {
    await tier2.engine.unload();
    tier2 = undefined;
    lastTier2Detect = undefined;
  }

  const started = performance.now();
  const engine = await createWebLlmEngine(config.modelId, config);
  const loadMs = performance.now() - started;

  let report: Tier2LoadReport;
  let judge: WebLlmJudge;
  try {
    // BEFORE the warm-up, so a budget `setTimeout` cannot hold fails with the
    // judge's own message rather than as a 1 ms deadline on the warm-up. It
    // still costs the load: nothing can validate a judge's budget without an
    // engine to build the judge on, and duplicating the rule here would be a
    // second copy free to drift from the one in @sih/tier2.
    judge = new WebLlmJudge(engine, { budgetMs: callBudgetMs });

    const warmupStarted = performance.now();
    const warmup = await engine.complete(TIER2_WARMUP_MESSAGES, {
      budgetMs: callBudgetMs,
      responseSchemaJson: TIER2_WARMUP_SCHEMA_JSON,
    });
    const warmupMs = performance.now() - warmupStarted;
    if (warmup.model !== config.modelId) {
      // The tier-1 half of this page refuses an arm whose observed backend
      // disagrees with the requested one; this is the same refusal. An engine
      // answering under another id means something reloaded it, and every
      // finding of this arm would be recorded against a model the arm was not
      // named for.
      throw new Error(
        `tier-2 engine was loaded as "${config.modelId}" but answered as "${warmup.model}"; ` +
          `refusing to report findings for one model under the other's name`,
      );
    }

    const storage = await navigator.storage.estimate();
    report = {
      config,
      callBudgetMs,
      servedModelId: warmup.model,
      loadMs,
      warmupMs,
      warmupFinishReason: warmup.finishReason,
      warmupCompletionTokens: warmup.usage?.completion_tokens,
      storageUsageBytes: storage.usage ?? 0,
      storageQuotaBytes: storage.quota ?? 0,
    };
  } catch (cause) {
    // Awaited, not fire-and-forget, and for the reason `loadTier1` gives: a
    // rejected release would otherwise reach Playwright's `pageerror` listener
    // and be reported instead of the real failure. A model this size abandoned
    // on the GPU also shrinks what the next arm can load.
    await engine.unload().catch((unloadFailure: unknown) => {
      throw new Error(
        `${config.modelId}: failed to load, and unloading the engine then failed too ` +
          `(${String(unloadFailure)})`,
        { cause },
      );
    });
    throw cause;
  }

  tier2 = { engine, judge, report };
  // Cleared for the same reason the tier-1 one is: the warm-up is this page's
  // own call, and a spec asking "did tier 2 run during MY detect" must not be
  // answered with it. The warm-up does not go through the judge at all, so this
  // is belt and braces rather than a correction.
  lastTier2Detect = undefined;
  // A baseline arm is built ON an engine. Replacing the engine leaves the arm
  // holding an unloaded one, so it goes with it -- and `loadBaseline` has to be
  // called again, which is what makes its report describe the engine it runs on.
  baseline = undefined;
  lastBaselineDetect = undefined;
  return report;
}

/**
 * Release the loaded engine and everything built on it. See `SihPageApi`.
 *
 * The baseline arm goes with it and is not merely cleared: it holds this engine
 * through `BaselineBOptions.engine`, so an arm surviving an unload would call
 * `complete` on a released engine and the failure would arrive as whatever
 * web-llm does with one rather than as "no arm is loaded".
 */
async function unloadTier2(): Promise<void> {
  if (tier2 === undefined) return;
  const { engine } = tier2;
  // Cleared BEFORE the await, so a `detect` racing this cannot find a state
  // whose engine is halfway through being released.
  tier2 = undefined;
  lastTier2Detect = undefined;
  baseline = undefined;
  lastBaselineDetect = undefined;
  await engine.unload();
}

// -- Approach B --------------------------------------------------------------

interface BaselineState {
  /** The arm itself: a `Detector`, with the arm's cumulative counters on it. */
  readonly arm: BaselineB;
  readonly report: BaselineLoadReport;
}

let baseline: BaselineState | undefined;
let lastBaselineDetect: BaselineDetectStats | undefined;

/**
 * Build an Approach-B arm on the loaded tier-2 engine.
 *
 * See `SihPageApi.loadBaseline` for what this door is for. What is worth having
 * beside the code is the list of things this function deliberately does NOT
 * choose, because each one is a lever that would tilt the head-to-head:
 *
 * - the ENGINE, taken from `tier2.engine`, so both arms decode through one
 *   object rather than two configured alike;
 * - the `Tier2Config`, taken from `tier2.report.config`, which is the object
 *   `createWebLlmEngine` was called with -- `baselineB.ts` refuses an engine
 *   whose `requestedModelId` disagrees with it, and that refusal is only worth
 *   anything if this passes the real one;
 * - the per-call budget, taken from `tier2.report.callBudgetMs`, which is the
 *   number the judge holds;
 * - the IR, which is not passed at all: B reads it off each `DetectInput`, so
 *   the arm cannot run tier 0 against a policy the record names differently.
 *
 * The one thing it does choose is which of the two constructors to call, and
 * that follows from `family` alone. `createBaselineBPlusTier0` is the mandatory
 * intermediate arm: without it a B-versus-compiled result conflates "compiling
 * the policy helps" with "having deterministic patterns helps".
 */
async function loadBaseline(options: BaselineLoadOptions): Promise<BaselineLoadReport> {
  // The two PAIRING checks run before the engine requirement, deliberately.
  // Both are configuration errors that cost nothing to detect, and a bake-off
  // whose policy does not match its IR should learn that before it spends a
  // 1 GB model load rather than after -- the same ordering `planBakeoff` uses
  // for every refusal it can make without touching the page. It also means the
  // spec that drives this refusal needs no GPU.
  if (!Object.hasOwn(POLICY_FIXTURES, options.policy)) {
    throw new Error(
      `unknown policy "${options.policy}"; this page carries ` +
        `${Object.keys(POLICY_FIXTURES).join(", ")}`,
    );
  }
  const policyText = POLICY_FIXTURES[options.policy] as string;
  const policyDocSha256 = await digestHex(policyText);
  // The pairing, refused HERE as well as in `planBakeoff`, and the two are not
  // redundant. That one checks a file the driver read against an IR the driver
  // parsed; this one checks the text THIS PAGE will put in the prompt against
  // the IR THIS PAGE loaded. A driver planning correctly over a page serving a
  // stale bundle passes the first and fails this one, which is the case worth
  // catching -- `playwright.config.ts` reuses an existing dev server outside CI.
  if (policyDocSha256 !== ir.policyHash) {
    throw new Error(
      `policy "${options.policy}" hashes to ${policyDocSha256} and the loaded IR's policyHash is ` +
        `"${ir.policyHash}"; Approach B is shown the document and the compiled arm is shown the ` +
        `IR, so running them against different policies compares two policies rather than two ` +
        `methods. Select the IR compiled from this document with useIr() first.`,
    );
  }

  if (tier2 === undefined) {
    throw new Error(
      "loadBaseline needs a loaded tier-2 engine; call loadTier2 first. Approach B runs on the " +
        "SAME engine as the compiled judge -- that is what makes the two arms' decoding settings " +
        "equal by construction rather than by convention",
    );
  }

  const build = options.family === "baseline-b-tier0" ? createBaselineBPlusTier0 : createBaselineB;
  const arm = build({
    engine: tier2.engine,
    config: tier2.report.config,
    policyText,
    budgetMs: tier2.report.callBudgetMs,
  });

  const report: BaselineLoadReport = {
    family: options.family,
    policy: options.policy,
    policyDocSha256,
    irPolicyHash: ir.policyHash,
    policyChars: policyText.length,
    callBudgetMs: tier2.report.callBudgetMs,
    // The OBSERVED id off the tier-2 load's warm-up completion, never
    // `config.modelId`, for the reason `measuredDetect` gives one tier down.
    servedModelId: tier2.report.servedModelId,
  };
  baseline = { arm, report };
  lastBaselineDetect = undefined;
  return report;
}

/**
 * One Approach-B `detect`, with the arm's counters read either side of it.
 *
 * The twin of `measuredDetect`, and a separate function rather than a branch
 * inside it because the two share no engine plumbing: B takes no tagger and no
 * judge, holds its own message deadline, and is itself the `Detector`. A branch
 * would have had to explain, at every line, which arm each half belonged to.
 *
 * The one check it does share is the model-substitution one, and it is here for
 * the same reason: `runArm` copies the arm's `TierConfig` onto every record, so
 * without it a row can name one model for a run the engine holding another
 * served.
 */
async function measuredBaselineDetect(
  request: DetectRequest,
  state: BaselineState,
): Promise<{ result: DetectionResult; stats: BaselineDetectStats }> {
  const { provider, text, config } = request;
  if (config.t2Model !== undefined && config.t2Model !== state.report.servedModelId) {
    throw new Error(
      `config.t2Model is "${config.t2Model}" but the engine this Approach-B arm runs on answered ` +
        `as "${state.report.servedModelId}"; refusing to report findings for one model under ` +
        `the other's name`,
    );
  }
  // A snapshot: `BaselineB.stats` rebuilds the counters and the call array on
  // every read, so this cannot be a live view of the numbers the delta is taken
  // against.
  const before = state.arm.stats;
  const result = await state.arm({ ir, provider, text, config });
  return { result, stats: baselineDelta(before, state.arm.stats) };
}

/**
 * One `detect` under a per-call budget this caller chooses, on the loaded
 * engine, with the stats that run produced.
 *
 * A FRESH judge over the SAME engine, which is what makes the returned stats
 * this call's without a delta -- and what keeps a deliberately-blown budget out
 * of the arm's totals. The engine is shared because the property under test is
 * an engine property: `runWithDeadline` interrupts a real generation, drains
 * it, and clears the interrupt flag the drain leaves set, and whether the
 * ENGINE survives that is answerable only against the engine that was
 * interrupted.
 *
 * Neither `lastDetect` moves. This call belongs to the caller that asked for the
 * odd budget, not to the arm, so `tier1Status()` and `tier2Status()` keep
 * reporting the last ordinary `detect` -- and the stats of the run that did
 * happen are returned rather than dropped.
 */
async function detectWithBudget(request: BudgetedDetectRequest): Promise<BudgetedDetectResult> {
  if (tier2 === undefined) {
    throw new Error("detectWithBudget needs a loaded tier-2 engine; call loadTier2 first");
  }
  const judge = new WebLlmJudge(tier2.engine, { budgetMs: request.callBudgetMs });
  const { result } = await measuredDetect(request, tier1?.tagger, judge);
  return { result, stats: judge.stats };
}

/**
 * A prompt of `words` meaningless words.
 *
 * Cycled rather than one word repeated, so the count grows with `words` under
 * any tokenizer. What the probe reports is the ENGINE's own
 * `usage.prompt_tokens` and never an estimate taken here -- the ratio is per
 * model and it is not stable. MEASURED with this exact generator at 2,000
 * words: 5,809 tokens on Qwen3.5-2B and on Qwen3-4B, 6,328 on Ministral-3-3B,
 * and 4,023 on Phi-4-mini. That last one is the reason the probe reads the
 * count back instead of trusting the word count: 4,023 is BELOW 4,096, so on
 * Phi-4-mini a 2,000-word prompt would have been accepted by the very window
 * size the probe exists to rule out.
 */
function fillerPrompt(words: number): string {
  return Array.from({ length: words }, (_, index) => `token${String(index % 97)}`).join(" ");
}

/**
 * Ask the loaded engine to prefill a prompt and report what it did with it.
 *
 * This is the ONLY way 0.2.84 will say which context window it is enforcing.
 * There is no accessor for the effective chat config, so the alternative to
 * measuring it is reporting the requested value under an observed-sounding
 * name, which is the intent-as-fact defect. READ from the installed bundle:
 * `getInputTokens` throws `ContextWindowSizeExceededError` when
 * `numPromptTokens + filledKVCacheLength > contextWindowSize`, and the message
 * names the window it enforced.
 *
 * Errors are RETURNED rather than thrown because a refusal is the probe's
 * expected outcome half the time -- see the 4,096 control in `tier2.spec.ts`.
 * A caller therefore has to assert on `errorName`, and any other failure
 * (a blown `budgetMs`, a latched engine) arrives as a different name rather
 * than as a pass.
 */
async function probeContextWindow(options: {
  words: number;
  budgetMs: number;
}): Promise<Tier2WindowProbe> {
  if (tier2 === undefined) {
    throw new Error("probeContextWindow needs a loaded tier-2 engine; call loadTier2 first");
  }
  const { words, budgetMs } = options;
  if (!(Number.isInteger(words) && words > 0)) {
    throw new Error(`probeContextWindow needs a positive integer word count, got ${String(words)}`);
  }
  const requestedContextWindowSize = tier2.report.config.contextWindowSize;
  const started = performance.now();
  try {
    const completion = await tier2.engine.complete([{ role: "user", content: fillerPrompt(words) }], {
      budgetMs,
      // The warm-up's one-boolean grammar again, for the same reason: the
      // prefill is what is being measured and the answer is not read.
      responseSchemaJson: TIER2_WARMUP_SCHEMA_JSON,
    });
    return {
      requestedContextWindowSize,
      promptWords: words,
      accepted: true,
      promptTokens: completion.usage?.prompt_tokens,
      finishReason: completion.finishReason,
      servedModelId: completion.model,
      errorName: undefined,
      errorMessage: undefined,
      ms: performance.now() - started,
    };
  } catch (cause) {
    const error = cause as { name?: unknown; message?: unknown };
    return {
      requestedContextWindowSize,
      promptWords: words,
      accepted: false,
      promptTokens: undefined,
      finishReason: undefined,
      servedModelId: undefined,
      errorName: typeof error.name === "string" ? error.name : String(cause),
      errorMessage: typeof error.message === "string" ? error.message : String(cause),
      ms: performance.now() - started,
    };
  }
}

const api: SihPageApi = {
  // The tier-1 engine is NOT something a spec can pass in: it is constructed by
  // `loadTier1` in the page, held module-level, and named in `tier1Status()`. An
  // engine crossing the evaluate boundary would be a stub by construction, and
  // the harness would report its latency as core's. What a spec CAN influence is
  // exactly the three fields of DetectRequest -- see measuredDetect.
  detect: async (request) => {
    // Approach B REPLACES core's orchestrator rather than sitting inside it, so
    // the routing is here and it is exclusive: an arm is one method or the
    // other, and a page holding a B arm must never quietly answer with the
    // compiled pipeline. `loadBaseline` is the only thing that sets this, and
    // `useIr` and `loadTier2` both clear it, so a run cannot drift into the
    // wrong branch between items.
    if (baseline !== undefined) {
      const { result, stats } = await measuredBaselineDetect(request, baseline);
      lastBaselineDetect = stats;
      return result;
    }
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
    // The TIER-2 twin of that check is NOT here, and deliberately: it is in
    // `measuredDetect`, which is the one function both doors onto the loaded
    // engine go through. `detectWithBudget` is the second door -- a fresh judge
    // over the SAME engine -- and a check written beside this one would leave it
    // open.
    const {
      result,
      tier1: tier1Stats,
      tier2: tier2Stats,
    } = await measuredDetect(request, tier1?.tagger, tier2?.judge);
    if (tier1Stats !== undefined) lastDetect = tier1Stats;
    if (tier2Stats !== undefined) lastTier2Detect = tier2Stats;
    return result;
  },
  irHash: () => irHash,
  policyHash: () => ir.policyHash,
  useIr,
  policyDocHash,
  loadBaseline,
  // `BaselineB.stats` copies on every read, one level deep, so `totals` is a
  // snapshot rather than a handle a spec could rewrite the arm's numbers
  // through.
  baselineStatus: () =>
    baseline === undefined
      ? undefined
      : { load: baseline.report, totals: baseline.arm.stats, lastDetect: lastBaselineDetect },
  backendAvailable,
  loadTier1,
  tier1Status: () =>
    tier1 === undefined
      ? undefined
      : { load: tier1.report, totals: { ...tier1.tagger.stats }, lastDetect },
  webgpuAvailable,
  loadTier2,
  unloadTier2,
  // `WebLlmJudge.stats` copies on every read, so `totals` is a snapshot rather
  // than a handle a spec could rewrite the arm's numbers through.
  tier2Status: () =>
    tier2 === undefined
      ? undefined
      : { load: tier2.report, totals: tier2.judge.stats, lastDetect: lastTier2Detect },
  detectWithBudget,
  probeContextWindow,
  harnessDir: () => __SIH_HARNESS_DIR__,
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
