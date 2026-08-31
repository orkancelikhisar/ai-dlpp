import { CreateMLCEngine } from "@mlc-ai/web-llm";
import type {
  ChatCompletionFinishReason,
  ChatCompletionMessageParam,
  ChatCompletionRequestNonStreaming,
  ChatOptions,
  CompletionUsage,
  InitProgressCallback,
  MLCEngineConfig,
  MLCEngineInterface,
} from "@mlc-ai/web-llm";
import { mlcInterruptible, runWithDeadline, type Interruptible } from "./cancel.js";
import type { Tier2Config } from "./manifest.js";
import { JUDGE_SCHEMA } from "./schema.js";

/**
 * The pinned request, with the two fields the recipe depends on narrowed from
 * optional to present.
 *
 * Intersecting with `ChatCompletionRequestNonStreaming` is load-bearing rather
 * than tidy: that interface has no index signature, so an object literal typed
 * as this is excess-property-checked against the real 0.2.84 request shape, and
 * a misplaced key is a compile error instead of a silently ignored one.
 *
 * What that does NOT buy is protection from either banned key in the position
 * it actually occupies, and this comment used to imply otherwise. MEASURED HERE
 * with `tsc` on the installed 0.2.84 types, four literals:
 *
 * - `enable_thinking` at the top level -> TS2353. But it has no top-level
 *   position; it lives under `extra_body`, and `extra_body: { enable_thinking:
 *   false }` COMPILES CLEAN, because `extra_body` is a declared property of
 *   `ChatCompletionRequestBase` and `enable_thinking` a declared property of it.
 * - `structural_tag` at the top level -> TS2353. It has no top-level position
 *   either; it lives at `response_format.structural_tag`
 *   (chat_completion.d.ts:832), and putting it there COMPILES CLEAN for the
 *   same reason -- the intersection widens `response_format` to include every
 *   key of `ResponseFormat`.
 *
 * So the type catches both keys only where nobody would put them. The guard
 * that catches them where they are dangerous is the DEEP KEY WALK in
 * `engine.test.ts`, which is why that test exists and why it walks rather than
 * checking `Object.keys`.
 */
export type Tier2CallParams = ChatCompletionRequestNonStreaming & {
  response_format: { type: "json_object"; schema: string };
  temperature: number;
  max_tokens: number;
};

/**
 * What one engine call produced. Every field is copied from the response; none
 * is copied from the request.
 */
export interface Tier2Completion {
  /**
   * `choices[0].message.content`, with the declared `null` case flattened to
   * "". Never the four characters of `String(null)`, which would parse as valid
   * JSON and be reported as a schema failure rather than as an empty body.
   *
   * "" here always means the engine produced an empty body, never that there
   * was no `choices[0]` to read: `complete` throws on a response with no
   * choices rather than folding it into this field. The two are not the same
   * event and the difference is the whole point -- an empty string reaching a
   * judge reads as "this segment is clean".
   */
  readonly content: string;
  /**
   * `choices[0].finish_reason`. `undefined` is reachable even though the
   * declared `Choice.finish_reason` is not optional -- see `complete`.
   */
  readonly finishReason: ChatCompletionFinishReason | undefined;
  /**
   * `ChatCompletion.model`: which loaded pipeline served THIS call.
   *
   * Narrower than it looks, and the earlier wording here oversold it. TRACED
   * through the 0.2.84 bundle: `chatCompletion` sets `model: selectedModelId`;
   * `selectedModelId` comes from `getModelIdToUse(Array.from(
   * loadedModelIdToPipeline.keys()), request.model, ...)`; and that map is
   * keyed by `loadedModelIdToPipeline.set(modelId, ...)` in `reloadInternal`,
   * where `modelId` is the very string handed to `CreateMLCEngine`. So it is
   * the requested id laundered through a Map key -- NOT an independent
   * attestation of which weights ran, and not a claim this field can support.
   *
   * What it does buy, which is real and is why every record goes through it:
   *
   * - A `reload()` behind our back changes the key, so the field follows the
   *   engine instead of remembering what we asked for.
   * - With several models loaded on one engine and no `model` in the request,
   *   `getModelIdToUse` THROWS `UnclearModelToUseError` rather than picking
   *   one, so this can never quietly name the wrong sibling.
   * - It is per-call and always a `string`, which `loadedModelId` is not.
   *
   * Never `undefined`: `complete` throws rather than passing an unnamed
   * response through, because two later consumers are typed `string` and the
   * tempting repair -- falling back to the requested id -- would be the
   * intent-recorded-as-fact defect this project has already shipped twice.
   */
  readonly model: string;
  /**
   * `ChatCompletion.usage`, verbatim -- no renaming, no defaults, no derived
   * fields. Declared optional in 0.2.84; READ out of the shipped bundle,
   * `MLCEngine.chatCompletion` builds it unconditionally on the non-streaming
   * path, so on that path `undefined` means something other than a stock
   * `MLCEngine` answered.
   *
   * CAUTION for whoever records these: every rate in `usage.extra` is a plain
   * division with no zero guard, so an interrupted call poisons them.
   * `time_per_output_token_s` is `decode_time / completion_tokens` and
   * `grammar_per_token_s` divides by the same count, which is 0 when the
   * interrupt landed before the first token -- both are then NaN. A zero
   * denominator under a non-zero numerator gives Infinity instead. Verified
   * here: `JSON.stringify` writes `null` for NaN and for both infinities, so
   * either one reaches a JSONL record as a missing number rather than as a
   * visibly broken one.
   */
  readonly usage: CompletionUsage | undefined;
}

export interface CompleteOptions {
  /**
   * Required, not optional, and that is the point. Two ways this engine stops
   * responding are on record in the plan -- an uncompilable grammar, and
   * `structural_tag` -- and both were reported as a call that never returns
   * rather than as an error. Neither was re-run here. A caller that forgets a
   * budget therefore gets no error and no answer, so the budget is asked for
   * rather than defaulted.
   *
   * Measured from when the call reaches the engine, not from when it was
   * requested; time queued behind another call on the same engine does not
   * count. Pass `signal` for a bound on total elapsed time.
   */
  readonly budgetMs: number;
  /** Honoured while queued as well as while generating. */
  readonly signal?: AbortSignal;
}

/**
 * What a tier-2 judge needs from an engine, and nothing else.
 *
 * Exported for the same reason `cancel.ts` exports `Interruptible`, and the
 * reason is mechanical rather than aesthetic: `WebLlmEngine` has `#private`
 * fields, which makes TypeScript type it NOMINALLY. A structural stand-in --
 * the obvious way to test a judge without a GPU -- is rejected with
 * `TS2739 ... missing #engine, #interruptible, #loadedModelId`, and the only
 * escape is `as never`, which is a cast that asserts nothing and is exactly the
 * tautology trap this plan has already fallen into once. A consumer takes this
 * interface and gets a real fake.
 *
 * `loadedModelId` is deliberately NOT on it. It is `undefined` until the engine
 * has answered at least once, so a consumer reaching for it either gets a type
 * error or reaches for the `?? requestedModelId` repair that turns a request
 * back into a claim about what ran. `Tier2Completion.model` is per-call, always
 * a string, and is the channel built for that job. Leaving the field off the
 * seam makes the wrong choice unavailable rather than merely discouraged.
 */
export interface Tier2Engine {
  /** What was ASKED for. Never presented as what ran. */
  readonly requestedModelId: string;
  complete(
    messages: readonly ChatCompletionMessageParam[],
    opts: CompleteOptions,
  ): Promise<Tier2Completion>;
  unload(): Promise<void>;
}

/**
 * Stringified once. `response_format.schema` is declared `string` on 0.2.84
 * ("A schema string in the format of the schema of a JSON file"), so this is a
 * type conversion, not a serialization convenience -- an object there is a
 * value the grammar compiler cannot read.
 */
const JUDGE_SCHEMA_JSON = JSON.stringify(JUDGE_SCHEMA);

/**
 * Build the one call shape Plan 5 measured to work.
 *
 * Three omissions matter more than anything present:
 *
 * - **No `enable_thinking`, at any depth.** On 0.2.84 the key is not top-level
 *   at all; it lives under `extra_body`. VERIFIED HERE by reading the bundle:
 *   the pipeline tests it with `=== false`, and that branch encodes an empty
 *   think block and pushes those tokens onto `outputIds`, so the literal tag
 *   lands in `message.content` and `JSON.parse` then throws. `undefined` and
 *   `true` both take the other branch -- which is why omitting the key and
 *   setting it false are not the same thing, and only `false` is harmful.
 * - **No `structural_tag`.** Measured upstream to hang forever;
 *   `json_object` + `schema` constrains the same grammar.
 * - **No `stream`.** The permanent engine wedge is on the streaming path, where
 *   abandoning a `for await` never releases the per-model lock. It is also what
 *   selects the overload: `stream: true` returns an AsyncIterable.
 *
 * Exported from this MODULE for its own tests, and deliberately NOT re-exported
 * from the package index. A caller holding the pinned request can hand it
 * straight to `engine.chat.completions.create`, which skips the deadline, the
 * drain, the interrupt-clear and the per-engine serialization -- every one of
 * the measured hazards this module exists to hold. `schema.ts` keeps
 * `classifyJsonPrefix` private and `orchestrator.ts` keeps `normalizeFindings`
 * private for the same reason; the package surface is `Tier2Engine` and the
 * factory.
 *
 * The result is deliberately NOT frozen. `postInitAndCheckFields` updates the
 * request in place -- it assigns `request.response_format` and unshifts a
 * system message onto `request.messages` -- on the Hermes function-calling
 * branch. The pinned recipe sends no `tools` and so never reaches that branch,
 * but freezing would be a bet that no other path ever does, and the payoff
 * would be a TypeError raised far from its cause.
 */
export function buildCallParams(
  messages: readonly ChatCompletionMessageParam[],
  config: Tier2Config,
): Tier2CallParams {
  if (messages.length === 0) {
    // `postInitAndCheckFields` reads `messages[messages.length - 1].role` with
    // no length check, so an empty list surfaces as a bare TypeError from
    // inside the library, which reads like a library bug rather than a caller
    // bug.
    throw new Error("tier-2 call needs at least one message; the engine indexes the last one");
  }
  return {
    // Copied so a caller that keeps appending to its own array cannot change
    // what this request sends after it was built. The copy is SHALLOW: mutating
    // a message object in place still shows through, and nothing here prevents
    // that.
    messages: [...messages],
    temperature: config.temperature,
    max_tokens: config.maxTokens,
    response_format: { type: "json_object", schema: JUDGE_SCHEMA_JSON },
  };
}

/**
 * One loaded model behind the pinned recipe.
 *
 * WebGPU failures arrive as a THROW at init, never as a degraded result, and
 * this is the opposite of the `onnxruntime-web` defect Plan 4 found. VERIFIED
 * HERE by reading `detectGPUDevice` in the shipped 0.2.84 bundle rather than by
 * running it:
 *
 * - No `navigator.gpu` -> the function returns undefined and `reloadInternal`
 *   throws `WebGPUNotAvailableError`.
 * - `requestAdapter()` resolves null -> it throws "Unable to find a compatible
 *   GPU".
 * - `maxComputeWorkgroupStorageSize` below `32 << 10`, or
 *   `maxStorageBuffersPerShaderStage` below 10, -> it throws outright. Unlike
 *   `maxBufferSize` and `maxStorageBufferBindingSize`, which each fall back
 *   once before failing, these two have no fallback at all. The source itself
 *   notes the WebGPU default for the buffer-count limit is 8, i.e. the library
 *   demands more than the spec guarantees.
 *
 * PRIOR MEASUREMENT, from Plan 5 feasibility research and not re-run here: on
 * the development machine those two adapter limits sit at exactly the required
 * minimum -- `maxStorageBuffersPerShaderStage` 10 and
 * `maxComputeWorkgroupStorageSize` 32768 -- so a device one unit below either
 * throws rather than degrading.
 *
 * The consequence for callers: where WebGPU is unavailable tier 2 is ABSENT,
 * not weakened, and the caller routes construction failure to the same
 * fail-closed path as an unparseable response.
 */
export class WebLlmEngine implements Tier2Engine {
  readonly #engine: MLCEngineInterface;
  /**
   * Built once, in the constructor, because `runWithDeadline` keys its
   * one-call-at-a-time queue on this object's identity. A fresh adapter per
   * call type-checks, passes every single-call test, and silently drops the
   * serialization that stops one segment's timeout emptying the next segment's
   * answer.
   */
  readonly #interruptible: Interruptible;
  /**
   * The config this engine was LOADED under, held so `complete` cannot be
   * handed a second one. See `complete`.
   */
  readonly #config: Tier2Config;
  #loadedModelId: string | undefined;

  /**
   * PRIVATE. `createWebLlmEngine` is the only way in, and that is the point:
   * it is the one place that can check `modelId` against `config.modelId`
   * before a model is loaded. A public constructor would let
   * `new WebLlmEngine(someEngine, someOtherConfig)` walk straight past that
   * check with an engine loaded from a different id, which is the record-names-
   * a-model-that-never-ran failure the factory exists to refuse.
   *
   * This constructor cannot perform the check itself, and saying so is the
   * honest version: 0.2.84 exposes no accessor for what an engine loaded --
   * `MLCEngineInterface` has none and `MLCEngine.loadedModelIdToPipeline` is
   * private -- so at construction time there is nothing to compare against.
   * Tests build one through the factory's `createEngine` seam.
   */
  private constructor(engine: MLCEngineInterface, config: Tier2Config) {
    this.#engine = engine;
    this.#interruptible = mlcInterruptible(engine);
    this.#config = config;
  }

  /**
   * Internal door for `createWebLlmEngine`, which is the only caller. Exists
   * because the constructor is private and TypeScript's `private` is per-class,
   * not per-module, so a free function in this file cannot call it.
   */
  static forLoadedEngine(engine: MLCEngineInterface, config: Tier2Config): WebLlmEngine {
    return new WebLlmEngine(engine, config);
  }

  /** What was asked for, from the config this engine was loaded under. */
  get requestedModelId(): string {
    return this.#config.modelId;
  }

  /**
   * What the engine SAID served the last call, or undefined until it has
   * answered anything.
   *
   * Deliberately NOT seeded with the requested id: that would make the field a
   * lie for the whole window before the first call, and it is precisely the
   * intent-recorded-as-fact defect this project has shipped twice. Updated on
   * every completion, so a `reload()` behind our back is followed rather than
   * remembered.
   *
   * NOT on the `Tier2Engine` seam, and not the field to record a finding
   * against -- `Tier2Completion.model` is. See `Tier2Engine`. This getter is
   * for a caller that wants to ask an engine, outside any particular call,
   * whether it has spoken yet.
   */
  get loadedModelId(): string | undefined {
    return this.#loadedModelId;
  }

  /**
   * Run one call under `opts.budgetMs`.
   *
   * Takes NO config, and that is a fix rather than an omission. It used to take
   * a second, unrelated `Tier2Config` per call, which reopened the exact hole
   * `createWebLlmEngine` throws to close: an engine loaded as one model
   * accepted a config naming another and ran anyway, reporting the model it had
   * actually loaded. The factory refuses two requested ids in one call; there
   * was no reason for this method to accept them.
   *
   * The second half of that defect was quieter. `Tier2Config` has four fields
   * and this method read two of them: `modelId` is fixed at load, and
   * `contextWindowSize` is consumed by `CreateMLCEngine`'s third argument and
   * CANNOT change per call -- so a caller passing a config with a different
   * window got no error, no warning, and a request built at the loaded window.
   * That is a config capturing half its own dimensions, which this project has
   * shipped before. Holding the load-time config removes both cases instead of
   * guarding them: there is now exactly one config, and it is the one that ran.
   */
  async complete(
    messages: readonly ChatCompletionMessageParam[],
    opts: CompleteOptions,
  ): Promise<Tier2Completion> {
    const params = buildCallParams(messages, this.#config);
    // Never a Promise.race: it abandons the promise but not the generation,
    // which keeps the engine's per-model lock. Task 3 measured the cost and
    // `cancel.ts` carries the numbers; `runWithDeadline` interrupts, waits for
    // the drain, and then clears the interrupt flag the drain leaves set --
    // without that last step every later call returns instantly and empty.
    const response = await runWithDeadline(
      this.#interruptible,
      // The signal is accepted and ignored on purpose: `create` takes no
      // AbortSignal on 0.2.84, and `interruptGenerate()` is the only stop.
      () => this.#engine.chat.completions.create(params),
      opts.budgetMs,
      opts.signal,
    );

    // Both of these are declared non-optional on 0.2.84 and both are checked
    // anyway, because this file already defends four other declared-narrower-
    // than-reality fields and these two are the ones whose absence would be
    // INVISIBLE. `choices[0]?.message.content ?? ""` folds "the engine returned
    // no choices at all" into the same "" as "the engine returned an empty
    // body", and a "" reaching a judge reads as "this segment is clean" -- the
    // false negative this whole module exists to prevent. `model` typed
    // `string` while holding undefined poisons `loadedModelId` and both
    // downstream consumers that are themselves typed `string`.
    //
    // Neither is reachable through a stock `MLCEngine`: `chatCompletion` builds
    // `choices` from the generation and assigns `model: selectedModelId`
    // unconditionally on the non-streaming path. So these are tripwires for
    // something other than a stock MLCEngine answering, and they fail loudly
    // instead of inventing a clean segment or an unnamed model.
    const choice = response.choices[0];
    if (choice === undefined) {
      throw new Error(
        `tier-2 engine "${this.requestedModelId}" returned a response with no choices; ` +
          `treating that as an empty answer would report it as a clean segment`,
      );
    }
    if (typeof response.model !== "string" || response.model.length === 0) {
      throw new Error(
        `tier-2 engine "${this.requestedModelId}" answered without naming a model; ` +
          `every tier-2 finding is recorded against this field and it cannot be invented`,
      );
    }

    this.#loadedModelId = response.model;
    return {
      content: choice.message.content ?? "",
      // No default. `Choice.finish_reason` is declared non-optional, but the
      // bundle assigns it from `LLMChatPipeline.getFinishReason()`, declared
      // `ChatCompletionFinishReason | undefined` -- so the declared type is
      // narrower than reality and undefined reaches here. Defaulting it to
      // "stop" would turn "we do not know why this stopped" into "it finished
      // normally", and the reason that matters most, "abort", is exactly the
      // one an interrupted engine pairs with an empty body.
      finishReason: choice.finish_reason,
      model: response.model,
      usage: response.usage,
    };
  }

  /**
   * Release the model, and the VRAM it holds.
   *
   * Not bookkeeping: the plan pins ONE engine per arm because swapping models
   * inside a single page leaks VRAM, so an arm that finishes without unloading
   * shrinks the budget available to the next one. The two largest pinned arms
   * are 3,432 MB and 3,438 MB, so two leaked arms is the difference between a
   * bake-off that completes and one that dies partway with an out-of-memory
   * failure attributable to nothing in particular.
   */
  async unload(): Promise<void> {
    await this.#engine.unload();
  }
}

/**
 * The subset of `CreateMLCEngine` this module uses. Injectable so a Node test
 * can observe WHICH argument each setting was passed in, which is the only way
 * to catch a setting the library would drop in silence.
 *
 * Exported from this MODULE for those tests and deliberately NOT re-exported
 * from the package index: it is a seam, not API. A consumer of `@sih/tier2`
 * that wants a substitutable engine takes `Tier2Engine`.
 */
export type CreateEngineFn = (
  modelId: string,
  engineConfig?: MLCEngineConfig,
  chatOpts?: ChatOptions,
) => Promise<MLCEngineInterface>;

/**
 * Load `modelId` and wrap it.
 *
 * `context_window_size` goes in the THIRD argument, and the position is the
 * whole point. Read out of 0.2.84's `reloadInternal`, the effective chat config
 * is
 *
 *   {...mlc-chat-config.json, ...modelRecord.overrides, ...chatOpts}
 *
 * and all four pinned arms ship `overrides.context_window_size: 4096` in
 * `prebuiltAppConfig`, so only `chatOpts` -- last in that spread -- can raise
 * it. `MLCEngineConfig`, the second argument, declares exactly four fields
 * (`appConfig`, `initProgressCallback`, `logitProcessorRegistry`, `logLevel`)
 * and nothing reads anything else off it: a `context_window_size` passed there
 * is dropped without a warning, and the arm then runs at 4096 while the record
 * claims 8192. That is a wrong number with a green suite, so the test asserts
 * the argument position rather than the value.
 *
 * This is the ONLY way to build a `WebLlmEngine`; its constructor is private.
 * That is what makes the guard below meaningful rather than advisory -- a
 * public constructor would let a caller assemble the same wrong pairing without
 * passing through it.
 *
 * @param createEngine a seam, defaulted to the real `CreateMLCEngine`. It is
 *   how a Node test observes which ARGUMENT each setting was passed in, which
 *   is the only way to catch a setting the library drops in silence. Not
 *   exported from the package index.
 * @throws when `modelId` disagrees with `config.modelId`. Two requested ids in
 *   one call is how a record ends up naming a model that never ran: the engine
 *   loads the argument while the record reports the config. The returned engine
 *   then HOLDS this config, so there is no second chance to introduce a
 *   disagreement later -- `complete` takes no config of its own.
 */
export async function createWebLlmEngine(
  modelId: string,
  config: Tier2Config,
  onProgress?: InitProgressCallback,
  createEngine: CreateEngineFn = CreateMLCEngine,
): Promise<WebLlmEngine> {
  if (modelId !== config.modelId) {
    throw new Error(
      `tier-2 engine asked to load "${modelId}" under a config pinned to ` +
        `"${config.modelId}"; the record would name a model that never ran`,
    );
  }
  const engine = await createEngine(
    modelId,
    { initProgressCallback: onProgress },
    { context_window_size: config.contextWindowSize },
  );
  return WebLlmEngine.forLoadedEngine(engine, config);
}
