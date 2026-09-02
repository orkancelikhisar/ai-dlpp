export interface Tier2Model {
  /**
   * The MLC model id, exactly as it appears in web-llm's `prebuiltAppConfig`.
   * MLCEngine resolves this by string equality and throws at load if it is
   * absent, so a truncated or stale id fails like a bug in our own code.
   */
  readonly id: string;
  /**
   * Peak VRAM in MB needed to run the model: `vram_required_MB` from
   * prebuiltAppConfig, rounded. This is NOT the download size and must not be
   * used to predict transfer time. Measured: a full cold load of Qwen3.5-2B
   * leaves 1079 MB of origin storage against the 2245 recorded here, and
   * Qwen2.5-0.5B downloads 278 MB against a 945 VRAM figure -- 2.1x and 3.4x
   * apart. VRAM is used here only as a monotone stand-in for cost, because it
   * is the one size the shipped library reports for every model offline.
   */
  readonly vramRequiredMb: number;
  /**
   * MEASURED decode rate on the development machine, tokens/sec. It falls with
   * context -- roughly 28% from a 157-token to a 1960-token prompt -- so these
   * are deliberately conservative long-context estimates, not the peak each
   * model hits on a short prompt. Ranking, not absolute value, is the point.
   */
  readonly decodeTokPerSec: number;
  /**
   * Whether the model has a thinking mode. Recorded and NEVER acted on by
   * passing `enable_thinking`: measured, that key under constrained decoding
   * injects a literal think tag into message.content and breaks JSON.parse.
   * It is here so a reader knows why a model's raw output looks the way it does.
   */
  readonly hasThinkingMode: boolean;
  /** One line on what measurement said about this model's task behaviour. */
  readonly note: string;
}

/**
 * The four models measured to load AND run. Ordered cheapest-first -- by VRAM,
 * the only per-model size the library reports offline -- because the bake-off
 * should pay for the expensive arms last. Frozen: `readonly` is erased at
 * runtime and this is process-wide shared state.
 */
export const TIER2_MODELS: readonly Tier2Model[] = Object.freeze([
  Object.freeze({
    id: "Qwen3.5-2B-q4f16_1-MLC",
    vramRequiredMb: 2245,
    decodeTokPerSec: 40,
    hasThinkingMode: true,
    note: "Fastest measured. Found only the AWS key on a 5-entity message, three times over.",
  }),
  Object.freeze({
    id: "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
    vramRequiredMb: 2864,
    decodeTokPerSec: 30,
    hasThinkingMode: false,
    note: "Schema-valid but returned EMPTY findings on a message full of secrets.",
  }),
  Object.freeze({
    id: "Qwen3-4B-q4f16_1-MLC",
    vramRequiredMb: 3432,
    decodeTokPerSec: 25,
    hasThinkingMode: true,
    note: "False-positived 'The weather is nice today.' as a secret.",
  }),
  Object.freeze({
    id: "Phi-4-mini-instruct-q4f16_1-MLC",
    vramRequiredMb: 3438,
    decodeTokPerSec: 24,
    hasThinkingMode: false,
    note: "Cold load 73.6 s (out-e5.json phi.load.ms=73592). Slowest decode measured.",
  }),
]);

export interface Tier2Config {
  readonly modelId: string;
  readonly contextWindowSize: number;
  readonly temperature: number;
  readonly maxTokens: number;
}

/**
 * `maxTokens` is UNCHANGED at 512 through the two-span schema change, and that
 * is a decision rather than a carry-over.
 *
 * The change made every finding carry a second required string on the wire --
 * the literal `,"mention":"..."`, 12 characters plus the value, on top of
 * `entityType`/`predicateId` + `quote` + `confidence` -- and this number did
 * not move with it. `max_tokens` is spent PER CALL (`engine.ts` feeds it to
 * `buildCallParams`), so where the judge makes several calls for one message it
 * has several times B's output allowance: `schema.ts` names B "the arm most
 * exposed to it" and `baselineB.ts`'s module docblock states the shape in full.
 * `fitsContext` charges only the PROMPT against `contextWindowSize -
 * maxTokens`; nothing charges the larger answer.
 *
 * How big that is on the runs this repository actually holds: ZERO. `p-fin` is
 * the only compiled policy here and its one predicate is message-scoped, so the
 * judge makes exactly one whole-message call and both arms spend one budget --
 * MEASURED, and already recorded on `baselineB.ts`: `test/baseline.spec.ts`
 * reports 3 answered calls and `truncatedResponses: 0` on each of the four
 * arms. So the extra field has not yet been shown to cost anything, and the
 * reason to leave 512 alone is not that the risk is imaginary.
 *
 * It is that raising it would be tuning an input so a result looks better, and
 * it would move ONE arm's headroom more than the other's on any policy with
 * segment-scoped clauses -- the harness difference the bake-off exists not to
 * measure. `truncatedResponses` is on both rows precisely so an arm killed by
 * its token budget is distinguishable from one with nothing to say. If that
 * counter rises for B under the two-span ask, the answer is a recorded,
 * symmetric budget change with a measurement behind it, not a nudge here. NO
 * GPU measurement of the new field's token cost was taken; this is an argument
 * from the shipped config and from what the run records already say.
 */
export const DEFAULT_TIER2_CONFIG: Tier2Config = Object.freeze({
  modelId: "Qwen3.5-2B-q4f16_1-MLC",
  contextWindowSize: 8192,
  temperature: 0,
  maxTokens: 512,
});

export function resolveTier2Config(overrides: Partial<Tier2Config>): Tier2Config {
  // Explicitly-undefined keys must not clobber defaults -- the natural CLI shape
  // passes `{ modelId: undefined }` for "not specified". Task 5 of Plan 4 shipped
  // this bug and it took a review round to find.
  const clean = Object.fromEntries(Object.entries(overrides).filter(([, v]) => v !== undefined));
  // The cast is only honest because every one of the four fields is checked
  // below: Object.fromEntries erases value types, so without those checks a
  // string "8192" would arrive here typed as a number.
  const config = { ...DEFAULT_TIER2_CONFIG, ...clean } as Tier2Config;
  if (!TIER2_MODELS.some((m) => m.id === config.modelId)) {
    throw new Error(
      `unknown tier-2 model "${config.modelId}"; measured to run: ` +
        TIER2_MODELS.map((m) => m.id).join(", "),
    );
  }
  if (config.temperature !== 0) {
    throw new Error(
      `tier-2 temperature must be 0 for a reproducible bake-off, got ${config.temperature}`,
    );
  }
  if (!(Number.isInteger(config.maxTokens) && config.maxTokens > 0)) {
    throw new Error(`tier-2 maxTokens must be a positive integer, got ${config.maxTokens}`);
  }
  // web-llm will NOT catch a bad window for us: its chat path only tests
  // `contextWindowSize !== -1`, and the `<= 0` guard lives in the embedding
  // pipeline. A 0 therefore reaches KV-cache allocation and resurfaces much
  // later as a ContextWindowSizeExceededError on the first prompt.
  if (!(Number.isInteger(config.contextWindowSize) && config.contextWindowSize > 0)) {
    throw new Error(
      `tier-2 contextWindowSize must be a positive integer, got ${config.contextWindowSize}`,
    );
  }
  return config;
}
