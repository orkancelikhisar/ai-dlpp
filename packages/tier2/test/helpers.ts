import type { ChatCompletionMessageParam, CompletionUsage } from "@mlc-ai/web-llm";
import type { Action, EntityType, PolicyIr, Severity } from "@sih/core";
import type { CompleteOptions, Tier2Completion, Tier2Engine } from "../src/index.js";

/**
 * Test doubles for `WebLlmJudge`. Two of them, and each exists because the real
 * thing cannot be used here: a `PolicyIr` is normally minted by the Node-only
 * compiler, and a `Tier2Engine` is normally a 2 GB model behind WebGPU.
 */

// ---------------------------------------------------------------------------
// predicateIr
// ---------------------------------------------------------------------------

export interface PredicateSpec {
  readonly id: string;
  readonly nlPredicate: string;
  readonly scope?: "segment" | "message";
  readonly severity?: Severity;
  /**
   * Set false to declare the predicate WITHOUT its shadow entityType -- the
   * shape a compiler bug would emit, and the one the judge must refuse loudly
   * rather than judge silently.
   */
  readonly shadow?: boolean;
}

export interface PredicateIrOptions {
  readonly predicates?: readonly PredicateSpec[];
}

const DEFAULT_PREDICATES: readonly PredicateSpec[] = [
  {
    id: "client-relationship",
    nlPredicate: "names a client the company does commercial business with",
    severity: "high",
  },
];

/**
 * A loadable IR carrying one semantic predicate and the shadow entityType the
 * compiler would have minted for it.
 *
 * The shadow id is spelled `pred:${id}` HERE, deliberately, and not built by
 * calling core's `shadowIdFor`. A test that mints its expectation with the
 * function under test proves only that the function agrees with itself: this
 * file is the independent oracle, so changing the prefix in core has to fail
 * here rather than quietly rename both sides at once.
 *
 * The non-shadow `client-name` entityType is always present because the IR
 * schema requires at least one entityType, and `predicateIr({ predicates: [] })`
 * would otherwise be unloadable. It doubles as a prior-findings label that is
 * not a shadow.
 *
 * The AUTHORED `client-name` entityType carries sentinel `examples` and
 * `counterExamples` that appear nowhere else. That is what makes "the prompt
 * carries no examples" a real assertion rather than a vacuous one: with empty
 * lists everywhere, a judge that interpolated `entity.examples` straight into
 * its prompt would pass.
 *
 * SHADOW entityTypes keep `examples: []`, because that is what the compiler
 * actually mints (`mintShadowEntityTypes` hardcodes it). Giving them sentinels
 * would make the assertion catch one more hypothetical mutation, at the cost of
 * handing every test in this suite an IR shape the compiler cannot produce --
 * and this fixture is shared by all of them. A shadow list that is empty in
 * production cannot leak anything, so nothing is lost: the mutation that
 * matters, interpolating examples at all, is still caught by the authored one.
 */
export function predicateIr(options: PredicateIrOptions = {}): PolicyIr {
  const predicates = options.predicates ?? DEFAULT_PREDICATES;
  const entityTypes: EntityType[] = [
    {
      id: "client-name",
      tier: 1,
      nlDefinition: "the name of a client organisation or contact",
      examples: ["SENTINEL-AUTHORED-EXAMPLE"],
      counterExamples: ["SENTINEL-AUTHORED-COUNTEREXAMPLE"],
      severity: "medium",
    },
  ];
  const defaultActions: Record<string, Action> = { "client-name": "pseudonymize" };

  for (const predicate of predicates) {
    if (predicate.shadow === false) continue;
    const shadowId = `pred:${predicate.id}`;
    entityTypes.push({
      id: shadowId,
      tier: 2,
      nlDefinition: predicate.nlPredicate,
      // Empty, as `mintShadowEntityTypes` mints them. See the note above.
      examples: [],
      counterExamples: [],
      severity: predicate.severity ?? "high",
      neverPseudonymize: true,
    });
    defaultActions[shadowId] = "redact";
  }

  return {
    irVersion: "1",
    policyHash: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    entityTypes,
    rules: [],
    semanticPredicates: predicates.map((p) => ({
      id: p.id,
      nlPredicate: p.nlPredicate,
      scope: p.scope ?? "segment",
    })),
    actions: { default: defaultActions },
    failMode: "closed",
    latencyBudgetMs: 30_000,
    provenance: {},
  };
}

// ---------------------------------------------------------------------------
// fakeEngine
// ---------------------------------------------------------------------------

export interface FakeFinding {
  readonly predicateId: string;
  readonly quote: string;
  /**
   * The span an action rewrites, inside `quote`.
   *
   * REQUIRED, with no default here, deliberately. A default of `quote` would
   * make every existing test scripting a finding assert the whole-clause path
   * without saying so, which is exactly the answer this field exists to make
   * visible -- and the suite would then be blind to a judge that stopped
   * reading `mention` at all. Call sites that mean "the whole clause" say so.
   */
  readonly mention: string;
  readonly confidence: number;
}

/** One scripted answer. `raw` wins over `findings` when both are present. */
export interface FakeCompletionSpec {
  readonly findings?: readonly FakeFinding[];
  readonly raw?: string;
  readonly finishReason?: Tier2Completion["finishReason"];
  readonly model?: string;
  readonly usage?: CompletionUsage;
  /** Thrown instead of resolving, for the `DeadlineExpired` paths. */
  readonly throws?: unknown;
}

export interface FakeEngineOptions extends FakeCompletionSpec {
  readonly requestedModelId?: string;
  /**
   * Scripted per call, the LAST entry repeating for every later call. A single
   * repeating answer is what makes "bad JSON twice" -- and so a repair attempt
   * that genuinely happened -- expressible.
   */
  readonly script?: readonly FakeCompletionSpec[];
  readonly onCall?: (messages: readonly ChatCompletionMessageParam[], opts: CompleteOptions) => void;
}

export interface RecordedCall {
  readonly messages: readonly ChatCompletionMessageParam[];
  readonly opts: CompleteOptions;
}

export interface FakeEngine extends Tier2Engine {
  /** Every `complete` call in order, including the ones that threw. */
  readonly calls: readonly RecordedCall[];
  readonly unloadCount: number;
  /** All message content of one call, joined -- what the model would read. */
  promptOf(callIndex: number): string;
}

/**
 * The message-order preconditions the real library enforces, mirrored here.
 *
 * READ out of the installed `@mlc-ai/web-llm` 0.2.84 bundle
 * (`lib/index.js`, `postInitAndCheckFields`) rather than assumed: it throws
 * `SystemMessageOrderError` for a `system` message at any index but 0, throws
 * `MessageOrderError` unless the LAST message is `user` or `tool`, and reaches
 * that last check through `request.messages[request.messages.length - 1].role`
 * with no length guard -- so an empty list is a bare `TypeError` raised from
 * inside the library, which reads like a library bug rather than a caller bug.
 * `getConversationFromChatCompletionRequest` repeats both checks.
 *
 * Nothing in the bundle requires user and assistant turns to ALTERNATE; the
 * conversation builder appends each message in order and the prompt assembler
 * renders one role prefix per message. So two consecutive `user` messages are
 * permitted. That is a claim about the LIBRARY, read from its source; it is not
 * a claim about how any model responds to such a prompt, which needs a GPU to
 * find out and was not measured.
 *
 * Asserted on EVERY fake call, not in one dedicated test, so a prompt-shape
 * regression fails in whichever test introduced it.
 */
function assertLibraryMessageOrder(messages: readonly ChatCompletionMessageParam[]): void {
  if (messages.length === 0) {
    throw new Error("web-llm indexes messages[length - 1] with no length check; an empty list is a TypeError");
  }
  messages.forEach((message, index) => {
    if (message.role === "system" && index !== 0) {
      throw new Error(`web-llm throws SystemMessageOrderError: system message at index ${index}`);
    }
  });
  const last = messages[messages.length - 1]!;
  if (last.role !== "user" && last.role !== "tool") {
    throw new Error(`web-llm throws MessageOrderError: last message is "${last.role}", not user or tool`);
  }
}

/**
 * A `Tier2Engine` that answers from a script.
 *
 * Structural, and that is only possible because Task 5 exported the
 * `Tier2Engine` seam: `WebLlmEngine` itself carries `#private` fields, so
 * TypeScript types it nominally and an object literal is rejected outright --
 * the only escape being `as never`, a cast that asserts nothing.
 */
export function fakeEngine(options: FakeEngineOptions = {}): FakeEngine {
  const calls: RecordedCall[] = [];
  const script: readonly FakeCompletionSpec[] = options.script ?? [options];
  let unloadCount = 0;

  return {
    requestedModelId: options.requestedModelId ?? "fake-model-that-was-requested",
    calls,
    get unloadCount() {
      return unloadCount;
    },
    promptOf(callIndex: number): string {
      const call = calls[callIndex];
      if (call === undefined) throw new Error(`no call at index ${callIndex}; ${calls.length} were made`);
      return call.messages.map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content))).join("\n");
    },
    async complete(messages, opts): Promise<Tier2Completion> {
      assertLibraryMessageOrder(messages);
      const index = calls.length;
      calls.push({ messages, opts });
      options.onCall?.(messages, opts);
      // The last entry repeats: a fake that runs out of script and starts
      // answering differently would make a repair test pass for the wrong
      // reason.
      const spec = script[Math.min(index, script.length - 1)] ?? {};
      if (spec.throws !== undefined) throw spec.throws;
      return {
        content: spec.raw ?? JSON.stringify({ findings: spec.findings ?? [] }),
        finishReason: spec.finishReason ?? "stop",
        model: spec.model ?? "fake-model-that-answered",
        usage: spec.usage,
      };
    },
    async unload(): Promise<void> {
      unloadCount += 1;
    },
  };
}
