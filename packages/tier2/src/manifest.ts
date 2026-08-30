export interface Tier2Model {
  /**
   * The MLC model id, exactly as it appears in web-llm's `prebuiltAppConfig`.
   * MLCEngine resolves this by string equality and throws at load if it is
   * absent, so a truncated or stale id fails like a bug in our own code.
   */
  readonly id: string;
  /** Download size in MB, from prebuiltAppConfig. */
  readonly sizeMb: number;
  /** MEASURED decode rate on the development machine, tokens/sec. Falls with context. */
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
 * The four models measured to load AND run. Ordered cheapest-first because the
 * bake-off should pay for the expensive arms last.
 */
export const TIER2_MODELS: readonly Tier2Model[] = [
  {
    id: "Qwen3.5-2B-q4f16_1-MLC",
    sizeMb: 2245,
    decodeTokPerSec: 40,
    hasThinkingMode: true,
    note: "Fastest measured. Found only the AWS key on a 5-entity message, three times over.",
  },
  {
    id: "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
    sizeMb: 2864,
    decodeTokPerSec: 30,
    hasThinkingMode: false,
    note: "Schema-valid but returned EMPTY findings on a message full of secrets.",
  },
  {
    id: "Qwen3-4B-q4f16_1-MLC",
    sizeMb: 3432,
    decodeTokPerSec: 25,
    hasThinkingMode: true,
    note: "False-positived 'The weather is nice today.' as a secret.",
  },
  {
    id: "Phi-4-mini-instruct-q4f16_1-MLC",
    sizeMb: 3438,
    decodeTokPerSec: 24,
    hasThinkingMode: false,
    note: "Cold load 73.6 s / 2.18 GB. Deterministic; slowest decode measured.",
  },
];

export interface Tier2Config {
  readonly modelId: string;
  readonly contextWindowSize: number;
  readonly temperature: number;
  readonly maxTokens: number;
}

export const DEFAULT_TIER2_CONFIG: Tier2Config = {
  modelId: "Qwen3.5-2B-q4f16_1-MLC",
  contextWindowSize: 8192,
  temperature: 0,
  maxTokens: 512,
};

export function resolveTier2Config(overrides: Partial<Tier2Config>): Tier2Config {
  // Explicitly-undefined keys must not clobber defaults -- the natural CLI shape
  // passes `{ modelId: undefined }` for "not specified". Task 5 of Plan 4 shipped
  // this bug and it took a review round to find.
  const clean = Object.fromEntries(Object.entries(overrides).filter(([, v]) => v !== undefined));
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
  return config;
}
