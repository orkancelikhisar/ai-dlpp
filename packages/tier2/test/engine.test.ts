import { describe, expect, it } from "vitest";
import { prebuiltAppConfig } from "@mlc-ai/web-llm";
import type {
  ChatCompletion,
  ChatCompletionMessageParam,
  MLCEngineInterface,
} from "@mlc-ai/web-llm";
import { buildCallParams, createWebLlmEngine } from "../src/engine.js";
import type { Tier2Engine } from "../src/engine.js";
import { JUDGE_SCHEMA } from "../src/schema.js";
import { DEFAULT_TIER2_CONFIG, resolveTier2Config } from "../src/manifest.js";
import type { Tier2Config } from "../src/manifest.js";
import { DeadlineExpired } from "../src/cancel.js";

const MSGS: ChatCompletionMessageParam[] = [{ role: "user", content: "hi" }];

/**
 * Every key name appearing anywhere in a value, at any depth.
 *
 * Every use of this below is a NEGATIVE assertion -- "this key is not in
 * there" -- so the helper is one line away from disarming all of them at once:
 * a `deepKeys` that returned an empty set unconditionally would leave the whole
 * suite green while the two banned keys went unchecked. Verified: that mutation
 * survives everything here except the positive control paired with each walk.
 * So every `not.toContain` in this file is accompanied by a `toContain` on the
 * SAME call, proving the walk reached the depth in question.
 */
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
  /**
   * Omit `model` entirely. `ChatCompletion.model` is declared NON-optional and
   * a stock MLCEngine always sets it, so this fakes a non-stock engine -- the
   * only thing that can produce the `string`-typed-but-undefined case.
   */
  omitModel?: boolean;
  /**
   * Return zero choices. Also declared non-optional and also unreachable via a
   * stock MLCEngine; the point is that `choices[0]?.content ?? ""` would fold
   * it into an empty answer, which a judge reads as a clean segment.
   */
  emptyChoices?: boolean;
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
      choices: opts.emptyChoices === true
        ? []
        : [{ index: 0, logprobs: null, message: { role: "assistant", content }, finish_reason: finishReason }],
    };
    if (opts.omitModel !== true) base["model"] = opts.model ?? "the-engine-did-not-say";
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

/**
 * Build a `WebLlmEngine` over a fake, through the only door there is.
 *
 * The class's constructor is private so that `createWebLlmEngine` is the single
 * place a modelId can be checked against the config it will be recorded
 * against. These tests go through the same door rather than around it, which
 * also means the config the engine HOLDS is the config it was loaded with --
 * there is no longer a second one to pass at call time.
 */
const engineOver = (f: ReturnType<typeof fakeEngine>, config: Tier2Config = DEFAULT_TIER2_CONFIG) =>
  createWebLlmEngine(config.modelId, config, undefined, async () => asEngine(f));

/**
 * The DeadlineExpired a call rejected with.
 *
 * `await p.catch((e) => e)` types as `Tier2Completion | DeadlineExpired`, which
 * makes every field access a cast, and it silently yields the RESOLVED value
 * when the call did not reject at all -- so an assertion written that way goes
 * vacuous the moment the rejection stops happening. This throws instead.
 */
async function rejection(p: Promise<unknown>): Promise<DeadlineExpired> {
  try {
    await p;
  } catch (e) {
    return e as DeadlineExpired;
  }
  throw new Error("expected this call to reject, but it resolved");
}

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
    // (chat_completion.d.ts:201, `enable_thinking` at :208), so checking only
    // Object.keys(p) can never fail. Nor does the TYPE catch it: verified with
    // tsc against the installed 0.2.84 types, `extra_body: { enable_thinking:
    // false }` compiles clean, because both are declared properties. The walk
    // is the only assertion here that can fail.
    //
    // Read out of the bundle: `enable_thinking === false` encodes an empty
    // think block and pushes it onto `outputIds`, so the literal tag lands in
    // message.content and JSON.parse then throws. `undefined` and `true` both
    // take the harmless branch -- which is why "absent" is required and "false"
    // is the one value that must never be sent.
    const keys = deepKeys(p);
    // Positive control on the same call: `schema` lives two levels down, inside
    // `response_format`. Without it a deepKeys that returned nothing would
    // satisfy every negative assertion in this file.
    expect(keys).toContain("schema");
    expect(keys).not.toContain("enable_thinking");
    expect(keys).not.toContain("extra_body");
  });

  it("never sends structural_tag, which lives inside response_format", () => {
    // Measured upstream: hangs forever. json_object + schema does the same job.
    //
    // Its real position is `response_format.structural_tag`
    // (chat_completion.d.ts:832), NOT the top level -- and verified with tsc,
    // putting it there compiles clean, because the intersection widens
    // `response_format` to every key of `ResponseFormat`. So the deep walk is
    // what guards this, and it has to reach inside `response_format` to do it.
    const keys = deepKeys(p);
    // Positive control, and specifically one INSIDE response_format: this is
    // the exact nesting level at which a structural_tag would hide.
    expect(keys).toContain("response_format");
    expect(keys).toContain("schema");
    expect(keys).not.toContain("structural_tag");
  });

  it("takes temperature from the config rather than a literal", () => {
    expect(p.temperature).toBe(0);
    // A hardcoded `temperature: 0` passes the line above and every other test
    // in this suite, because DEFAULT_TIER2_CONFIG.temperature IS 0 -- the same
    // config-vs-literal blind spot that hid a hardcoded context_window_size.
    // Verified: `temperature: 0` in buildCallParams survives the whole suite
    // without this second config.
    //
    // Built as a literal rather than through resolveTier2Config on purpose:
    // that function REFUSES a non-zero temperature today, which is why this is
    // harmless in production. The point is not that 0.3 is allowed -- it is
    // that the field must be READ, so a future temperature-sensitivity arm
    // cannot report a value the engine never received.
    const warm: Tier2Config = { ...DEFAULT_TIER2_CONFIG, temperature: 0.3 };
    expect(buildCallParams(MSGS, warm).temperature).toBe(0.3);
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

  it("stays off the streaming path, where the wedge is permanent", () => {
    // HONEST LABEL: this assertion is a runtime restatement of a type-level
    // guarantee and cannot fail without a cast. `Tier2CallParams` intersects
    // `ChatCompletionRequestNonStreaming`, which declares `stream?: false |
    // null`, so `stream: true` is already a compile error. Kept, and kept
    // named for the hazard rather than deleted, because the streaming path is
    // where Task 3 measured the PERMANENT wedge -- abandoning a `for await`
    // never releases the per-model lock -- and if that intersection is ever
    // loosened this is the line that should start failing.
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
    const e = await engineOver(f);
    const r = await e.complete(MSGS, { budgetMs: 1000 });

    expect(e.requestedModelId).toBe("Qwen3.5-2B-q4f16_1-MLC");
    expect(r.model).toBe("Qwen3-4B-q4f16_1-MLC");
    expect(e.loadedModelId).toBe("Qwen3-4B-q4f16_1-MLC");
  });

  it("has no loadedModelId until the engine has actually reported one", async () => {
    // Seeding it with the requested id would make the field a lie for the whole
    // window before the first call, and no later assertion would notice.
    const e = await engineOver(fakeEngine());
    expect(e.loadedModelId).toBeUndefined();
  });

  it("throws rather than pass on a response that names no model", async () => {
    // `ChatCompletion.model` is declared `string` (chat_completion.d.ts:244) and
    // a stock MLCEngine assigns it unconditionally, so this is a tripwire for
    // something else answering. It matters because `Tier2Completion.model` is
    // also declared `string` and two later consumers are typed on that: a
    // silent undefined would flow into a finding's `source` and into the load
    // report, and the tempting repair -- falling back to the requested id --
    // turns a request back into a claim about what ran.
    const e = await engineOver(fakeEngine({ omitModel: true }));
    await expect(e.complete(MSGS, { budgetMs: 1000 })).rejects.toThrow(/naming a model/i);
  });

  it("throws on a response with no choices, instead of reporting a clean segment", async () => {
    // `choices[0]?.message.content ?? ""` folds THREE different events into one
    // empty string: a null content, an empty body, and no choices at all. The
    // third is the dangerous one -- an empty string reaching a judge reads as
    // "this segment is clean", which is the false negative this module exists
    // to prevent -- and it silently takes finishReason and usage to undefined
    // with it.
    const e = await engineOver(fakeEngine({ emptyChoices: true }));
    await expect(e.complete(MSGS, { budgetMs: 1000 })).rejects.toThrow(/no choices/i);
  });

  it("surfaces finish_reason, so truncation is distinguishable from a short answer", async () => {
    // A model that stopped at max_tokens produced a partial judgement.
    // Reporting that as a complete one understates recall with no way to notice.
    const f = fakeEngine({ content: '{"findings":[]}', finishReason: "length" });
    const e = await engineOver(f);
    const r = await e.complete(MSGS, { budgetMs: 1000 });
    expect(r.finishReason).toBe("length");
  });

  it("surfaces abort rather than passing an empty body off as a clean segment", async () => {
    // Task 3's measurement: a poisoned engine answers in 0 ms with
    // finish_reason "abort" and zero characters. A judge that cannot see the
    // reason reads that as "no findings", permanently, on every later segment.
    const f = fakeEngine({ content: "", finishReason: "abort" });
    const e = await engineOver(f);
    const r = await e.complete(MSGS, { budgetMs: 1000 });
    expect(r.finishReason).toBe("abort");
    expect(r.content).toBe("");
  });

  it("surfaces an absent finish_reason as undefined, not as a stop", async () => {
    // getFinishReason() is declared `ChatCompletionFinishReason | undefined`
    // while Choice.finish_reason is declared non-optional, so the response can
    // carry undefined. Defaulting that to "stop" would invent a fact.
    const f = fakeEngine({ finishReason: undefined });
    const e = await engineOver(f);
    const r = await e.complete(MSGS, { budgetMs: 1000 });
    expect(r.finishReason).toBeUndefined();
  });

  it("turns a null content into an empty string, never the text null", async () => {
    // ChatCompletionMessage.content is declared `string | null`, and THAT is
    // the justification -- a declared type we must handle, not a branch we
    // expect to hit. (The bundle's null-content branch is the function-calling
    // one, which needs `request.tools`; buildCallParams never sends tools, so
    // it cannot fire under the pinned recipe.) `String(null)` would hand the
    // parser the four characters n-u-l-l, which parse as valid JSON and then
    // fail the schema, misreporting a missing body as a bad model.
    const f = fakeEngine({ content: null });
    const e = await engineOver(f);
    const r = await e.complete(MSGS, { budgetMs: 1000 });
    expect(r.content).toBe("");
  });

  it("passes usage through verbatim, NaN and Infinity included", async () => {
    const f = fakeEngine();
    const e = await engineOver(f);
    const r = await e.complete(MSGS, { budgetMs: 1000 });
    expect(r.usage).toEqual(DEFAULT_USAGE);
  });

  it("does not sanitise a poisoned usage into plausible-looking numbers", async () => {
    // This is the case that matters and the all-finite fixture above cannot
    // reach. Every rate in `usage.extra` is an unguarded division by
    // `completion_tokens`, which is 0 when an interrupt landed before the first
    // token: `time_per_output_token_s` and `grammar_per_token_s` come back NaN,
    // and a non-zero numerator over that zero gives Infinity.
    //
    // Verified: adding a `sanitizeUsage()` that rewrites non-finite values to 0
    // leaves the whole suite green without this test -- and it is the WORST
    // repair available, because JSON.stringify writes `null` for NaN and both
    // infinities, so an unrepaired number reaches a record as visibly missing
    // while a zeroed one reaches it as a real measurement of zero. Every
    // interrupted bake-off row would then claim perfect throughput.
    const poisoned = {
      completion_tokens: 0,
      prompt_tokens: 300,
      total_tokens: 300,
      extra: {
        e2e_latency_s: 1.5,
        prefill_tokens_per_s: 452,
        decode_tokens_per_s: Number.POSITIVE_INFINITY,
        time_to_first_token_s: 0.66,
        time_per_output_token_s: Number.NaN,
        grammar_init_s: 0.1,
        grammar_per_token_s: Number.NaN,
      },
    };
    const e = await engineOver(fakeEngine({ usage: poisoned }));
    const r = await e.complete(MSGS, { budgetMs: 1000 });
    const extra = r.usage?.extra as Record<string, number> | undefined;
    // Written as Number.isNaN rather than toEqual: toEqual treats NaN as equal
    // to NaN, so `expect(r.usage).toEqual(poisoned)` would also pass against an
    // implementation that had replaced these with... NaN. It would NOT catch a
    // zeroing sanitizer, which is the mutation under test here.
    expect(Number.isNaN(extra?.["time_per_output_token_s"])).toBe(true);
    expect(Number.isNaN(extra?.["grammar_per_token_s"])).toBe(true);
    expect(extra?.["decode_tokens_per_s"]).toBe(Number.POSITIVE_INFINITY);
    expect(extra?.["prefill_tokens_per_s"]).toBe(452);
  });

  it("reports a missing usage as undefined instead of inventing zeros", async () => {
    // ChatCompletion.usage is declared optional. 0.2.84's MLCEngine always
    // populates it on the non-streaming path, but a zero-filled stand-in would
    // put fabricated throughput numbers into a bake-off record, which is the
    // failure mode this project keeps shipping.
    const f = fakeEngine({ omitUsage: true });
    const e = await engineOver(f);
    const r = await e.complete(MSGS, { budgetMs: 1000 });
    expect(r.usage).toBeUndefined();
  });

  it("sends a request carrying the schema and neither banned key", async () => {
    const f = fakeEngine();
    const e = await engineOver(f);
    await e.complete(MSGS, { budgetMs: 1000 });

    const sent = f.state.seen[0] as Record<string, unknown>;
    // NOT compared against a fresh buildCallParams(...) call: that is the same
    // builder producing both sides, so it holds for any recipe the builder
    // emits -- including a wrong one. Verified: that comparison passed under
    // two mutations the lines below caught. These assert the properties
    // themselves.
    const keys = deepKeys(sent);
    expect(keys).toContain("response_format");
    expect(keys).toContain("schema");
    expect(keys).not.toContain("enable_thinking");
    expect(keys).not.toContain("structural_tag");
    const rf = sent["response_format"] as { type: unknown; schema: unknown };
    expect(rf.type).toBe("json_object");
    expect(typeof rf.schema).toBe("string");
    expect(JSON.parse(rf.schema as string)).toEqual(JUDGE_SCHEMA);
    expect(sent["messages"]).toEqual(MSGS);
    expect(sent["temperature"]).toBe(DEFAULT_TIER2_CONFIG.temperature);
    expect(sent["max_tokens"]).toBe(DEFAULT_TIER2_CONFIG.maxTokens);
  });

  it("interrupts, drains and clears on deadline expiry", async () => {
    // A Promise.race would reject here with the generation still running and
    // the flag never set; measured, that costs 10.2 s of latency on the
    // non-streaming path and wedges the streaming one forever. The drain is
    // what makes `settled` non-zero by the time we reject, and the clear is
    // what stops the NEXT call returning instantly and emptily.
    const f = fakeEngine({ hangsForCalls: 1 });
    const e = await engineOver(f);
    await expect(e.complete(MSGS, { budgetMs: 20 })).rejects.toBeInstanceOf(DeadlineExpired);
    expect(f.state.interrupted).toBe(1);
    expect(f.state.settled).toBe(1);
    expect(f.interruptSignal).toBe(false);
  });

  it("says a budget expired only when one did, and names the interrupt honestly", async () => {
    // A budget overrun and a caller giving up are different events with
    // different fixes, and one message describing both states a falsehood in
    // one of them. The old message -- "exceeded its 5000ms budget and was
    // interrupted" -- was wrong twice over on the abort path below: 0 ms of
    // 5000 had been spent and nothing had been interrupted, which the very next
    // assertions prove.
    const slow = fakeEngine({ hangsForCalls: 1 });
    const slowEngine = await engineOver(slow);
    const expired = await rejection(slowEngine.complete(MSGS, { budgetMs: 20 }));
    expect(expired).toBeInstanceOf(DeadlineExpired);
    expect(expired.reason).toBe("budget");
    expect(expired.interrupted).toBe(true);
    expect(expired.message).toMatch(/exceeded its 20ms budget/);

    const idle = fakeEngine();
    const idleEngine = await engineOver(idle);
    const aborted = await rejection(
      idleEngine.complete(MSGS, { budgetMs: 5000, signal: AbortSignal.abort() }),
    );
    expect(aborted).toBeInstanceOf(DeadlineExpired);
    expect(aborted.reason).toBe("aborted");
    expect(aborted.interrupted).toBe(false);
    // The message must not claim a budget was exceeded, because it was not.
    expect(aborted.message).not.toMatch(/exceeded/);
    expect(aborted.message).toMatch(/aborted by its caller/);
    // ...and these are what make the claim above a fact rather than a wording
    // preference: nothing ran and nothing was interrupted.
    expect(idle.state.calls).toBe(0);
    expect(idle.state.interrupted).toBe(0);
  });

  it("refuses a budget setTimeout would silently turn into an instant deadline", async () => {
    // MEASURED on Node 26: setTimeout coerces Infinity, NaN, 0, -1 and 2^31 all
    // to a 1 ms delay. So POSITIVE_INFINITY -- the natural spelling of "no
    // budget" -- would interrupt the engine before it answered and report
    // "exceeded its Infinityms budget". Rejected rather than read as "no
    // deadline", because a required budget is what stops a hung engine from
    // presenting as a call that never returns.
    const f = fakeEngine();
    const e = await engineOver(f);
    for (const bad of [Number.POSITIVE_INFINITY, Number.NaN, 0, -1, 2_147_483_648]) {
      await expect(e.complete(MSGS, { budgetMs: bad })).rejects.toThrow(/budgetMs/);
    }
    // Refused before the engine was touched -- not interrupted, not poisoned,
    // and no queue turn consumed.
    expect(f.state.calls).toBe(0);
    expect(f.state.interrupted).toBe(0);
    // The boundary is usable, not just the safe middle.
    expect((await e.complete(MSGS, { budgetMs: 2_147_483_647 })).content).toBe('{"findings":[]}');
  });

  it("honours an outer abort signal, which bounds total time including the queue", async () => {
    // budgetMs starts when the call reaches the engine, so it alone cannot
    // bound a caller that is queued behind another segment. `signal` is the
    // only bound on total elapsed time, and an ignored one looks exactly like a
    // working one until something is slow.
    const f = fakeEngine();
    const e = await engineOver(f);
    await expect(
      e.complete(MSGS, { budgetMs: 5000, signal: AbortSignal.abort() }),
    ).rejects.toBeInstanceOf(DeadlineExpired);
    // Never started: interrupting a generation that was not ours to interrupt
    // would poison the engine for whoever runs next.
    expect(f.state.calls).toBe(0);
    expect(f.state.interrupted).toBe(0);
  });

  it("builds every request from the config it was LOADED under", async () => {
    // `complete` takes no config, and this is the runtime half of why. It used
    // to take a second, unrelated Tier2Config per call with no check at all --
    // reopening the hole createWebLlmEngine throws to close, since an engine
    // loaded as one model would accept a config naming another and run anyway.
    // The compile-time half is that there is no longer a parameter to pass one
    // through.
    //
    // Asserted on TWO calls rather than one: a held config and a config read
    // from the first call's arguments are indistinguishable on a single call.
    const f = fakeEngine();
    const e = await engineOver(f, resolveTier2Config({ maxTokens: 77 }));
    await e.complete(MSGS, { budgetMs: 1000 });
    await e.complete(MSGS, { budgetMs: 1000 });
    expect(f.state.seen).toHaveLength(2);
    for (const sent of f.state.seen) {
      expect((sent as { max_tokens: number }).max_tokens).toBe(77);
    }
  });

  it("offers a seam a judge can fake without a cast", async () => {
    // WebLlmEngine has #private fields, which makes TypeScript type it
    // NOMINALLY: a structural stand-in is rejected with `TS2739 ... missing
    // #engine, #interruptible, #loadedModelId`, leaving `as never` -- a cast
    // that asserts nothing and is the tautology trap this plan already fell
    // into once. The assertion here is that this literal COMPILES as a
    // Tier2Engine with no cast; if the seam ever regains a #private-bearing
    // member, this stops compiling rather than quietly forcing the next task
    // back to `as never`.
    const fake: Tier2Engine = {
      requestedModelId: "Qwen3.5-2B-q4f16_1-MLC",
      complete: async () => ({
        content: '{"findings":[]}',
        finishReason: "stop",
        model: "Qwen3.5-2B-q4f16_1-MLC",
        usage: undefined,
      }),
      unload: async () => {},
    };
    expect((await fake.complete(MSGS, { budgetMs: 1000 })).model).toBe("Qwen3.5-2B-q4f16_1-MLC");

    // And the real class satisfies the same seam, so the fake is not a fiction
    // a judge could pass while the production type diverged.
    const real: Tier2Engine = await engineOver(fakeEngine());
    expect(real.requestedModelId).toBe(DEFAULT_TIER2_CONFIG.modelId);
  });

  it("serializes calls on one engine, so a neighbour's timeout cannot empty this answer", async () => {
    // MEASURED upstream and recorded in cancel.ts: two overlapping calls on one
    // engine, only the first with a short budget, and the SECOND resolved with
    // finish_reason "abort" and a zero-length body. The judge issues one call
    // per segment, so this was never hypothetical. Building a fresh
    // Interruptible per call silently loses the serialization, because
    // runWithDeadline keys its queue on that object's identity.
    const f = fakeEngine({ hangsForCalls: 1 });
    const e = await engineOver(f);

    const doomed = e.complete(MSGS, { budgetMs: 20 });
    const neighbour = e.complete(MSGS, { budgetMs: 5000 });

    await expect(doomed).rejects.toBeInstanceOf(DeadlineExpired);
    const r = await neighbour;
    expect(f.state.poisoned).toBe(0);
    expect(r.finishReason).not.toBe("abort");
    expect(r.content.length).toBeGreaterThan(0);
  });

  it("releases the model's VRAM, which one engine per arm depends on", async () => {
    // Not delegation for its own sake: the plan pins one engine per arm because
    // swapping models inside a page leaks VRAM, and the two largest arms are
    // 3,432 MB and 3,438 MB. An unload that never reached the engine would show
    // up as a later arm dying out of memory for no attributable reason.
    const f = fakeEngine();
    const e = await engineOver(f);
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
    const engineKeys = deepKeys(call.engineConfig);
    // Positive control on the same walk. Without it a deepKeys that returned an
    // empty set would satisfy the negative assertion below while checking
    // nothing.
    expect(engineKeys).toContain("initProgressCallback");
    expect(engineKeys).not.toContain("context_window_size");
  });

  it("takes context_window_size from the config rather than a literal", async () => {
    // Separate from the argument-position test above, because that one cannot
    // fail on this: DEFAULT_TIER2_CONFIG.contextWindowSize IS 8192, so a
    // hardcoded `context_window_size: 8192` satisfies it and every other test
    // here. Verified -- that mutation survives the entire suite without this
    // second config. The plan's own preamble turns on this knob: three of the
    // four arms are unmeasured at 8192 and may have to run at 4096, and an
    // engine that ignored the field would load them all at 8192 while the
    // record faithfully reported whatever the config said.
    const { seen, create } = spyCreate();
    const narrow = resolveTier2Config({ contextWindowSize: 4096 });
    await createWebLlmEngine(narrow.modelId, narrow, undefined, create);
    expect((seen[0]!.chatOpts as { context_window_size?: number }).context_window_size).toBe(4096);
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

  it("keeps the deadline bypass off the package's public surface", async () => {
    // `buildCallParams` returns the pinned request. A caller holding one can
    // hand it straight to `engine.chat.completions.create`, which skips the
    // deadline, the drain, the interrupt-clear and the per-engine
    // serialization -- every measured hazard this package exists to hold. It is
    // exported from engine.ts for the tests above and must stop there.
    //
    // `CreateEngineFn` is a type and erases, so it cannot be asserted here; it
    // is covered by the same rule in index.ts.
    const surface = await import("../src/index.js");
    expect(Object.keys(surface)).not.toContain("buildCallParams");
    // Positive control: this walk is worthless if the module failed to load or
    // the key list came back empty.
    expect(Object.keys(surface)).toContain("createWebLlmEngine");
    expect(Object.keys(surface)).toContain("parseJudgeResponse");
    // And the class is a TYPE-only export: its constructor is private, so
    // `createWebLlmEngine` is the only way to build one and therefore the only
    // place a modelId can be checked against the config it is recorded against.
    expect(Object.keys(surface)).not.toContain("WebLlmEngine");
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
