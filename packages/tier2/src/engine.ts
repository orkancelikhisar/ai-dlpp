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
 * as this is excess-property-checked against the real 0.2.84 request shape.
 * A stray `enable_thinking` or `structural_tag` at the top level is a compile
 * error, not a silently ignored key.
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
   */
  readonly content: string;
  /**
   * `choices[0].finish_reason`. `undefined` is reachable even though the
   * declared `Choice.finish_reason` is not optional -- see `complete`.
   */
  readonly finishReason: ChatCompletionFinishReason | undefined;
  /**
   * `ChatCompletion.model`: the id the ENGINE says it ran, read out of its own
   * loaded-pipeline map. Not the id anyone asked for.
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
export class WebLlmEngine {
  readonly #engine: MLCEngineInterface;
  /**
   * Built once, in the constructor, because `runWithDeadline` keys its
   * one-call-at-a-time queue on this object's identity. A fresh adapter per
   * call type-checks, passes every single-call test, and silently drops the
   * serialization that stops one segment's timeout emptying the next segment's
   * answer.
   */
  readonly #interruptible: Interruptible;
  #loadedModelId: string | undefined;

  /**
   * @param engine an already-loaded engine.
   * @param requestedModelId the id that was ASKED for. Named for what it is:
   *   0.2.84 exposes no accessor for what the engine actually loaded --
   *   `MLCEngineInterface` has none, and `MLCEngine`'s
   *   `loadedModelIdToPipeline` is private -- so at construction time there is
   *   no better source, and calling this field a loaded id would be a claim
   *   nothing here can support.
   */
  constructor(engine: MLCEngineInterface, requestedModelId: string) {
    this.#engine = engine;
    this.#interruptible = mlcInterruptible(engine);
    this.requestedModelId = requestedModelId;
  }

  /** What was asked for. See the constructor. */
  readonly requestedModelId: string;

  /**
   * What the engine SAID it ran, or undefined until it has said anything.
   *
   * `ChatCompletion.model` is set from the engine's own
   * `loadedModelIdToPipeline` keys, so it is a report rather than an echo: it
   * is the only channel on 0.2.84 through which the engine names what it
   * loaded. Updated on every completion, so a `reload()` behind our back is
   * followed rather than remembered.
   *
   * Deliberately NOT seeded with `requestedModelId`: that would make the field
   * a lie for the whole window before the first call, and it is precisely the
   * intent-recorded-as-fact defect this project has shipped twice.
   */
  get loadedModelId(): string | undefined {
    return this.#loadedModelId;
  }

  async complete(
    messages: readonly ChatCompletionMessageParam[],
    config: Tier2Config,
    opts: CompleteOptions,
  ): Promise<Tier2Completion> {
    const params = buildCallParams(messages, config);
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

    this.#loadedModelId = response.model;
    const choice = response.choices[0];
    return {
      content: choice?.message.content ?? "",
      // No default. `Choice.finish_reason` is declared non-optional, but the
      // bundle assigns it from `LLMChatPipeline.getFinishReason()`, declared
      // `ChatCompletionFinishReason | undefined` -- so the declared type is
      // narrower than reality and undefined reaches here. Defaulting it to
      // "stop" would turn "we do not know why this stopped" into "it finished
      // normally", and the reason that matters most, "abort", is exactly the
      // one an interrupted engine pairs with an empty body.
      finishReason: choice?.finish_reason,
      model: response.model,
      usage: response.usage,
    };
  }

  /**
   * Release the model. The plan pins one engine per arm because swapping models
   * inside one page leaks VRAM, so the bake-off unloads between arms.
   */
  async unload(): Promise<void> {
    await this.#engine.unload();
  }
}

/**
 * The subset of `CreateMLCEngine` this module uses. Injectable so a Node test
 * can observe WHICH argument each setting was passed in, which is the only way
 * to catch a setting the library would drop in silence.
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
 * @throws when `modelId` disagrees with `config.modelId`. Two requested ids in
 *   one call is how a record ends up naming a model that never ran: the engine
 *   loads the argument while the record reports the config.
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
  return new WebLlmEngine(engine, modelId);
}
