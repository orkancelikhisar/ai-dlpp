import { describe, expect, it } from "vitest";
import { prebuiltAppConfig } from "@mlc-ai/web-llm";
import type {
  ChatCompletion,
  ChatCompletionMessageParam,
  MLCEngineInterface,
} from "@mlc-ai/web-llm";
import { buildCallParams, WebLlmEngine, createWebLlmEngine } from "../src/engine.js";
import { JUDGE_SCHEMA } from "../src/schema.js";
import { DEFAULT_TIER2_CONFIG, resolveTier2Config } from "../src/manifest.js";
import { DeadlineExpired } from "../src/cancel.js";

const MSGS: ChatCompletionMessageParam[] = [{ role: "user", content: "hi" }];

/** Every key name appearing anywhere in a value, at any depth. */
function deepKeys(value: unknown, out = new Set<string>()): Set<string> {
  if (Array.isArray(value)) {
    for (const v of value) deepKeys(v, out);
  } else if (value !== null && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      out.add(k);
      deepKeys(v, out);
    }
  }
  return out;
}

interface FakeOpts {
  /** `ChatCompletionMessage.content` is declared `string | null` on 0.2.84. */
  content?: string | null;
  /**
   * Deliberately `unknown`: `ChatCompletion.Choice.finish_reason` is declared
   * non-optional, but the bundle assigns it from
   * `LLMChatPipeline.getFinishReason(): ChatCompletionFinishReason | undefined`
   * (llm_chat.d.ts:103), so `undefined` is reachable and the declared type is
   * narrower than reality.
   */
  finishReason?: unknown;
  /** What the ENGINE says it ran. The bundle sets this to its own loaded id. */
  model?: string;
  usage?: unknown;
  /** Omit `usage` entirely -- `ChatCompletion.usage` is declared optional. */
  omitUsage?: boolean;
  /**
   * How many of the first calls generate until interrupted, the way a real
   * 512-token decode does. A count rather than a flag: the serialization test
   * needs the FIRST call to hang and the one queued behind it to succeed.
   */
  hangsForCalls?: number;
}

/**
 * A fake modelled on what `@mlc-ai/web-llm` 0.2.84 was READ to do, not on a
 * convenient API. Three behaviours are copied from the shipped bundle because
 * each one lets a plausible implementation through if faked away:
 *
 * 1. `create` takes no AbortSignal. Only `interruptGenerate()` stops work.
 * 2. The interrupt flag is sticky on the non-streaming path: `chatCompletion`
 *    tests `this.interruptSignal` BEFORE deciding to call `_generate`, and that
 *    branch returns `""` with `finish_reason: "abort"` without ever clearing it.
 * 3. Calls serialize on a per-model lock acquired before that test, so a queued
 *    call cannot dodge the poisoned branch.
 */
function fakeEngine(opts: FakeOpts = {}) {
  const state = {
    calls: 0,
    poisoned: 0,
    interrupted: 0,
    unloaded: 0,
    /** Incremented when a generation actually finishes -- i.e. it was drained. */
    settled: 0,
    seen: [] as unknown[],
  };
  let lock: Promise<void> = Promise.resolve();

  const reply = (content: string | null, finishReason: unknown): ChatCompletion => {
    const base: Record<string, unknown> = {
      id: "fake",
      object: "chat.completion",
      created: 0,
      model: opts.model ?? "the-engine-did-not-say",
      choices: [{ index: 0, logprobs: null, message: { role: "assistant", content }, finish_reason: finishReason }],
    };
    if (opts.omitUsage !== true) base["usage"] = opts.usage ?? DEFAULT_USAGE;
    // `??` is deliberately NOT used for content or finish_reason: `null` and
    // `undefined` are two of the values under test, and coalescing them away
    // would make those two cases silently assert the happy path instead.
    return base as unknown as ChatCompletion;
  };

  const engine = {
    /** MLCEngine's own field; `mlcInterruptible` writes it to recover the engine. */
    interruptSignal: false,
    state,
    interruptGenerate(): void {
      state.interrupted += 1;
      engine.interruptSignal = true;
    },
    async unload(): Promise<void> {
      state.unloaded += 1;
    },
    chat: {
      completions: {
        async create(request: unknown): Promise<ChatCompletion> {
          const prior = lock;
          let release!: () => void;
          lock = new Promise<void>((r) => {
            release = r;
          });
          await prior;
          try {
            state.calls += 1;
            state.seen.push(request);
            if (engine.interruptSignal) {
              state.poisoned += 1;
              return reply("", "abort");
            }
            if (state.calls <= (opts.hangsForCalls ?? 0)) {
              await new Promise<void>((resolve) => {
                const t = setInterval(() => {
                  if (engine.interruptSignal) {
                    clearInterval(t);
                    resolve();
                  }
                }, 1);
              });
              return reply("", "abort");
            }
            return reply(
              "content" in opts ? (opts.content as string | null) : '{"findings":[]}',
              "finishReason" in opts ? opts.finishReason : "stop",
            );
          } finally {
            state.settled += 1;
            release();
          }
        },
      },
    },
  };
  return engine;
}

const DEFAULT_USAGE = {
  completion_tokens: 12,
  prompt_tokens: 300,
  total_tokens: 312,
  extra: {
    e2e_latency_s: 1.5,
    prefill_tokens_per_s: 452,
    decode_tokens_per_s: 40,
    time_to_first_token_s: 0.66,
    time_per_output_token_s: 0.025,
    grammar_init_s: 0.1,
    grammar_per_token_s: 0.001,
  },
};

const asEngine = (f: ReturnType<typeof fakeEngine>) => f as unknown as MLCEngineInterface;

// ---------------------------------------------------------------------------

describe("buildCallParams", () => {
  const p = buildCallParams(MSGS, DEFAULT_TIER2_CONFIG);

  it("stringifies the schema, because 0.2.84 declares response_format.schema as a string", () => {
    // chat_completion.d.ts:812 -- `schema?: string`, "A schema string in the
    // format of the schema of a JSON file". An object there is not a narrower
    // type, it is a different one, and grammar compilation gets a value it
    // cannot read.
    expect(typeof p.response_format.schema).toBe("string");
    expect(JSON.parse(p.response_format.schema)).toEqual(JUDGE_SCHEMA);
    // Not decoration: `postInitAndCheckFields` throws InvalidResponseFormatError
    // when `schema` is set and `type` is anything but "json_object".
    expect(p.response_format.type).toBe("json_object");
  });

  it("never sends enable_thinking, at any depth", () => {
    // On 0.2.84 this key is NOT top-level -- it lives under `extra_body`
    // (chat_completion.d.ts:196), so checking only Object.keys(p) can never
    // fail. The walk is the assertion that can.
    //
    // Read out of the bundle: `enable_thinking === false` encodes an empty
    // think block and pushes it onto `outputIds`, so the literal tag lands in
    // message.content and JSON.parse then throws. `undefined` and `true` both
    // take the harmless branch -- which is why "absent" is required and "false"
    // is the one value that must never be sent.
    expect(deepKeys(p)).not.toContain("enable_thinking");
    expect(deepKeys(p)).not.toContain("extra_body");
  });

  it("never sends structural_tag", () => {
    // Measured upstream: hangs forever. json_object + schema does the same job.
    expect(deepKeys(p)).not.toContain("structural_tag");
    expect(p.response_format.type).not.toBe("structural_tag");
  });

  it("pins temperature to 0 so a rerun agrees with itself", () => {
    expect(p.temperature).toBe(0);
  });

  it("takes max_tokens from the config rather than a literal", () => {
    // 512 is the plan's pinned default. A hardcoded 512 would satisfy that and
    // silently ignore an arm that raises the budget, which is exactly the knob
    // the bake-off needs in order to tell a budget-killed arm from an
    // incapable one.
    expect(p.max_tokens).toBe(512);
    const tight = buildCallParams(MSGS, resolveTier2Config({ maxTokens: 128 }));
    expect(tight.max_tokens).toBe(128);
  });

  it("does not stream", () => {
    // The permanent engine wedge Task 3 measured is on the streaming path:
    // abandoning a `for await` never releases the per-model lock. The pinned
    // recipe is non-streaming, and `create`'s overloads key off this field --
    // `stream: true` returns an AsyncIterable, not a ChatCompletion.
    expect(p.stream).not.toBe(true);
  });

  it("passes the caller's messages through, and does not alias the array", () => {
    const mine: ChatCompletionMessageParam[] = [
      { role: "system", content: "you are a judge" },
      { role: "user", content: "segment" },
    ];
    const snapshot = structuredClone(mine);
    const out = buildCallParams(mine, DEFAULT_TIER2_CONFIG);
    expect(out.messages).toEqual(snapshot);
    expect(mine).toEqual(snapshot);

    // The direction that actually bites, and the one a "did we mutate the
    // caller's array" assertion cannot see: a judge reusing one array across
    // segments would otherwise rewrite a request it had already built. Only the
    // array is protected -- the copy is shallow, so this deliberately does not
    // claim the message objects are.
    mine.push({ role: "user", content: "a later segment" });
    expect(out.messages).toHaveLength(2);
  });

  it("refuses an empty message list", () => {
    // `postInitAndCheckFields` reads `request.messages[messages.length - 1].role`
    // with no length check, so an empty list surfaces as a bare TypeError from
    // inside the library, reading like a bug in the library rather than in the
    // caller.
    expect(() => buildCallParams([], DEFAULT_TIER2_CONFIG)).toThrow(/message/i);
  });
});

// ---------------------------------------------------------------------------

describe("WebLlmEngine", () => {
  it("records the model the engine reported, not the one requested", async () => {
    // The same disease Plan 4 shipped twice: a field recording intent rather
    // than fact. Asserting that `loadedModelId` equals the constructor argument
    // cannot fail -- it is the argument. So the fake reports a DIFFERENT id,
    // which is the only version of this test an intent-recording field fails.
    const f = fakeEngine({ model: "Qwen3-4B-q4f16_1-MLC" });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });

    expect(e.requestedModelId).toBe("Qwen3.5-2B-q4f16_1-MLC");
    expect(r.model).toBe("Qwen3-4B-q4f16_1-MLC");
    expect(e.loadedModelId).toBe("Qwen3-4B-q4f16_1-MLC");
  });

  it("has no loadedModelId until the engine has actually reported one", () => {
    // Seeding it with the requested id would make the field a lie for the whole
    // window before the first call, and no later assertion would notice.
    const e = new WebLlmEngine(asEngine(fakeEngine()), "Qwen3.5-2B-q4f16_1-MLC");
    expect(e.loadedModelId).toBeUndefined();
  });

  it("surfaces finish_reason, so truncation is distinguishable from a short answer", async () => {
    // A model that stopped at max_tokens produced a partial judgement.
    // Reporting that as a complete one understates recall with no way to notice.
    const f = fakeEngine({ content: '{"findings":[]}', finishReason: "length" });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });
    expect(r.finishReason).toBe("length");
  });

  it("surfaces abort rather than passing an empty body off as a clean segment", async () => {
    // Task 3's measurement: a poisoned engine answers in 0 ms with
    // finish_reason "abort" and zero characters. A judge that cannot see the
    // reason reads that as "no findings", permanently, on every later segment.
    const f = fakeEngine({ content: "", finishReason: "abort" });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });
    expect(r.finishReason).toBe("abort");
    expect(r.content).toBe("");
  });

  it("surfaces an absent finish_reason as undefined, not as a stop", async () => {
    // getFinishReason() is declared `ChatCompletionFinishReason | undefined`
    // while Choice.finish_reason is declared non-optional, so the response can
    // carry undefined. Defaulting that to "stop" would invent a fact.
    const f = fakeEngine({ finishReason: undefined });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });
    expect(r.finishReason).toBeUndefined();
  });

  it("turns a null content into an empty string, never the text null", async () => {
    // ChatCompletionMessage.content is declared `string | null`; the bundle
    // sets it to null on the function-calling branch. `String(null)` would hand
    // the parser the four characters n-u-l-l, which parse as valid JSON and
    // then fail the schema, misreporting a missing body as a bad model.
    const f = fakeEngine({ content: null });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });
    expect(r.content).toBe("");
  });

  it("passes usage through verbatim", async () => {
    const f = fakeEngine();
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });
    expect(r.usage).toEqual(DEFAULT_USAGE);
  });

  it("reports a missing usage as undefined instead of inventing zeros", async () => {
    // ChatCompletion.usage is declared optional. 0.2.84's MLCEngine always
    // populates it on the non-streaming path, but a zero-filled stand-in would
    // put fabricated throughput numbers into a bake-off record, which is the
    // failure mode this project keeps shipping.
    const f = fakeEngine({ omitUsage: true });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    const r = await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });
    expect(r.usage).toBeUndefined();
  });

  it("sends the pinned recipe to the engine", async () => {
    const f = fakeEngine();
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    await e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 1000 });

    const sent = f.state.seen[0];
    expect(sent).toEqual(buildCallParams(MSGS, DEFAULT_TIER2_CONFIG));
    expect(deepKeys(sent)).not.toContain("enable_thinking");
    expect(deepKeys(sent)).not.toContain("structural_tag");
    expect(typeof (sent as { response_format: { schema: unknown } }).response_format.schema).toBe(
      "string",
    );
  });

  it("interrupts, drains and clears on deadline expiry", async () => {
    // A Promise.race would reject here with the generation still running and
    // the flag never set; measured, that costs 10.2 s of latency on the
    // non-streaming path and wedges the streaming one forever. The drain is
    // what makes `settled` non-zero by the time we reject, and the clear is
    // what stops the NEXT call returning instantly and emptily.
    const f = fakeEngine({ hangsForCalls: 1 });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    await expect(e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 20 })).rejects.toBeInstanceOf(
      DeadlineExpired,
    );
    expect(f.state.interrupted).toBe(1);
    expect(f.state.settled).toBe(1);
    expect(f.interruptSignal).toBe(false);
  });

  it("honours an outer abort signal, which bounds total time including the queue", async () => {
    // budgetMs starts when the call reaches the engine, so it alone cannot
    // bound a caller that is queued behind another segment. `signal` is the
    // only bound on total elapsed time, and an ignored one looks exactly like a
    // working one until something is slow.
    const f = fakeEngine();
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    await expect(
      e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 5000, signal: AbortSignal.abort() }),
    ).rejects.toBeInstanceOf(DeadlineExpired);
    // Never started: interrupting a generation that was not ours to interrupt
    // would poison the engine for whoever runs next.
    expect(f.state.calls).toBe(0);
    expect(f.state.interrupted).toBe(0);
  });

  it("builds the request from the config it was handed, not from the default", async () => {
    const f = fakeEngine();
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    await e.complete(MSGS, resolveTier2Config({ maxTokens: 77 }), { budgetMs: 1000 });
    expect((f.state.seen[0] as { max_tokens: number }).max_tokens).toBe(77);
  });

  it("serializes calls on one engine, so a neighbour's timeout cannot empty this answer", async () => {
    // MEASURED upstream and recorded in cancel.ts: two overlapping calls on one
    // engine, only the first with a short budget, and the SECOND resolved with
    // finish_reason "abort" and a zero-length body. The judge issues one call
    // per segment, so this was never hypothetical. Building a fresh
    // Interruptible per call silently loses the serialization, because
    // runWithDeadline keys its queue on that object's identity.
    const f = fakeEngine({ hangsForCalls: 1 });
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");

    const doomed = e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 20 });
    const neighbour = e.complete(MSGS, DEFAULT_TIER2_CONFIG, { budgetMs: 5000 });

    await expect(doomed).rejects.toBeInstanceOf(DeadlineExpired);
    const r = await neighbour;
    expect(f.state.poisoned).toBe(0);
    expect(r.finishReason).not.toBe("abort");
    expect(r.content.length).toBeGreaterThan(0);
  });

  it("delegates unload to the engine", async () => {
    const f = fakeEngine();
    const e = new WebLlmEngine(asEngine(f), "Qwen3.5-2B-q4f16_1-MLC");
    await e.unload();
    expect(f.state.unloaded).toBe(1);
  });
});

// ---------------------------------------------------------------------------

describe("createWebLlmEngine", () => {
  function spyCreate() {
    const seen: { modelId: unknown; engineConfig: unknown; chatOpts: unknown }[] = [];
    const create = async (modelId: string, engineConfig?: unknown, chatOpts?: unknown) => {
      seen.push({ modelId, engineConfig, chatOpts });
      return asEngine(fakeEngine({ model: modelId }));
    };
    return { seen, create };
  }

  it("puts context_window_size where it can beat the model record's own override", async () => {
    // Grounded in the INSTALLED library, not in our implementation. Every
    // pinned arm ships `overrides.context_window_size: 4096` in
    // prebuiltAppConfig, and reloadInternal merges
    //   {...mlc-chat-config.json, ...modelRecord.overrides, ...chatOpts}
    // so `chatOpts` -- the THIRD argument of CreateMLCEngine -- is the only
    // position that wins. MLCEngineConfig (the second argument) declares only
    // appConfig / initProgressCallback / logitProcessorRegistry / logLevel; a
    // context_window_size passed there is dropped without a word, and the arm
    // runs at 4096 while the record claims 8192.
    const record = prebuiltAppConfig.model_list.find(
      (m) => m.model_id === DEFAULT_TIER2_CONFIG.modelId,
    );
    expect(record?.overrides?.context_window_size).toBe(4096);
    expect(DEFAULT_TIER2_CONFIG.contextWindowSize).toBeGreaterThan(4096);

    const { seen, create } = spyCreate();
    await createWebLlmEngine(DEFAULT_TIER2_CONFIG.modelId, DEFAULT_TIER2_CONFIG, undefined, create);

    expect(seen).toHaveLength(1);
    const call = seen[0]!;
    expect(call.modelId).toBe(DEFAULT_TIER2_CONFIG.modelId);
    expect((call.chatOpts as { context_window_size?: number }).context_window_size).toBe(
      DEFAULT_TIER2_CONFIG.contextWindowSize,
    );
    expect(deepKeys(call.engineConfig)).not.toContain("context_window_size");
  });

  it("passes initProgressCallback in the engine config, where 0.2.84 declares it", async () => {
    const { seen, create } = spyCreate();
    const onProgress = () => {};
    await createWebLlmEngine(DEFAULT_TIER2_CONFIG.modelId, DEFAULT_TIER2_CONFIG, onProgress, create);
    expect((seen[0]!.engineConfig as { initProgressCallback?: unknown }).initProgressCallback).toBe(
      onProgress,
    );
  });

  it("does not claim a loaded model id before the engine has reported one", async () => {
    const { create } = spyCreate();
    const e = await createWebLlmEngine(
      DEFAULT_TIER2_CONFIG.modelId,
      DEFAULT_TIER2_CONFIG,
      undefined,
      create,
    );
    expect(e.requestedModelId).toBe(DEFAULT_TIER2_CONFIG.modelId);
    expect(e.loadedModelId).toBeUndefined();
  });

  it("refuses a modelId that disagrees with the config it will be recorded against", async () => {
    // Two requested ids in one call is how a record ends up naming a model that
    // never ran: the engine loads the argument, the record reports the config.
    const { create } = spyCreate();
    await expect(
      createWebLlmEngine("Qwen3-4B-q4f16_1-MLC", DEFAULT_TIER2_CONFIG, undefined, create),
    ).rejects.toThrow(/Qwen3-4B-q4f16_1-MLC/);
  });
});
