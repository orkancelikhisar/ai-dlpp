import type { ChatCompletionFinishReason, ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import { shadowIdFor } from "@sih/core";
import type {
  EngineDegradedNotice,
  Finding,
  JudgeRequest,
  JudgeVerdict,
  PolicyIr,
  PredicateScope,
  SemanticJudge,
  SemanticPredicate,
  Severity,
} from "@sih/core";
import { DeadlineExpired, MAX_BUDGET_MS } from "./cancel.js";
import type { Tier2Completion, Tier2Engine } from "./engine.js";
import { parseJudgeResponse, type JudgeResponse } from "./schema.js";
import { MINIMUM_CANDIDATE_WORDS, resolveQuote } from "./spans.js";

/**
 * Tier 2: the semantic judge core has had a seam for since Plan 1.
 *
 * This module is composition, not new machinery. Every hazard it navigates was
 * measured by an earlier task in this plan and is held by the module that
 * measured it -- the deadline and the interrupt-drain-clear sequence by
 * `cancel.ts`, the pinned request shape by `engine.ts`, the four parse failure
 * modes by `schema.ts`, the refusal to guess a span by `spans.ts`. What is left
 * here is the part none of them could own: turning a model's quote into a
 * `Finding` core will accept, and REFUSING everything that would make core
 * accept a wrong one.
 *
 * The single obligation core has carried since Plan 3 is the entityType. A
 * predicate finding must name the SHADOW id `pred:<predicateId>` that the
 * compiler minted, never the bare predicate id: `normalizeFindings` throws on an
 * entityType the IR does not contain, and that throw loses the whole message,
 * not one finding.
 */

/**
 * What one ANSWERED engine call cost, kept per call rather than aggregated.
 *
 * Task 11's bake-off row wants `finishReason`, `promptTokens`,
 * `completionTokens` and `ttftMs`, and a message makes one engine call per
 * selected segment plus one for the whole message when the policy declares a
 * message-scoped predicate. A single `finishReason` for the message would
 * therefore be a fact about one call presented as a fact about the message, and
 * a summed `ttftMs` would be a latency nothing ever experienced. So the rows are kept whole and
 * the harness decides how to reduce them; this module does not decide for it.
 *
 * Every field is copied from the response and none from the request, and
 * nothing is defaulted to 0. `undefined` means the engine reported no `usage`
 * at all: READ from the shipped 0.2.84 bundle, `MLCEngine.chatCompletion`
 * builds `usage` unconditionally on the non-streaming path, so on that path
 * `undefined` means something other than a stock `MLCEngine` answered. A
 * NON-FINITE value is passed through unchanged for the same reason -- `engine.ts`
 * records that an interrupted call leaves NaN and Infinity in `usage.extra`,
 * and zeroing one would report an interrupted call as an instantaneous one.
 */
export interface JudgeCallRecord {
  /** `choices[0].finish_reason` for THIS call, never the message's. */
  readonly finishReason: ChatCompletionFinishReason | undefined;
  /** `usage.prompt_tokens`, verbatim. */
  readonly promptTokens: number | undefined;
  /** `usage.completion_tokens`, verbatim. */
  readonly completionTokens: number | undefined;
  /**
   * `usage.extra.time_to_first_token_s`, converted to milliseconds and
   * otherwise untouched. READ from the bundle: it is assigned from an
   * accumulated `prefill_time` rather than from a division, so it does not
   * carry the zero-denominator NaN its sibling rate fields do -- but the
   * conversion below still does not guard it, because a NaN that reached here
   * is a fact about the call and not a number to invent a replacement for.
   */
  readonly ttftMs: number | undefined;
  /**
   * `usage.extra.decode_tokens_per_s`, verbatim: tokens per second of
   * autoregressive decoding for THIS call, excluding prefill.
   *
   * Here because the bake-off's `minDecodeTokPerSec` gate is otherwise
   * uncomputable from a run's output. A record carries `timings.tier2Ms` for
   * the whole message and no per-call elapsed time, so the only rate derivable
   * downstream is `completionTokens / (tier2Ms - ttft)` -- which charges the
   * judge's prompt assembly, JSON parse and span ladder to the model, on a
   * gate that is a FLOOR. That error kills capable arms.
   *
   * READ from the installed 0.2.84 bundle rather than assumed: the field is
   * assigned `completion_tokens / decode_time`, where `decode_time` is
   * `pipeline.getCurRoundDecodingTotalTime()` -- an accumulation over this
   * round's decode steps only. So it is the engine's own measurement of the
   * quantity the gate names, and no harness time is inside it.
   *
   * It is a PLAIN DIVISION with no zero guard, which is why nothing here
   * guards it either: a call interrupted before its first token has
   * `completion_tokens` 0 and `decode_time` 0, and 0/0 is NaN. That NaN is a
   * fact about the call. `JSON.stringify` writes NaN and both infinities as
   * `null`, so whoever records this has to map it -- see `Tier2CallSchema` in
   * apps/eval/src/driver/record.ts, which does, for `ttftMs` and for this.
   */
  readonly decodeTokPerSec: number | undefined;
}

/**
 * Per-run counters, all of them facts about calls that actually happened.
 *
 * The bake-off reads these to tell an arm that judged well from one that failed
 * closed on every segment. Most name exactly one event; the two that AGGREGATE
 * -- `unresolvedQuotes` and `failedClosed` -- say what they aggregate on their
 * own field, because a header claiming a one-to-one mapping was false three
 * fields below itself.
 *
 * The four stop-related counters are separate on purpose, and they are the
 * three events `cancel.ts` refuses to merge plus the latched engine: a blown
 * budget says this model is too slow for this arm; a caller abort
 * mid-generation says a generation was interrupted; a caller abort while queued
 * says nothing ran at all; and an aborted RESPONSE says the engine was
 * interrupted by something we did not do. Folding them together would put a
 * fabricated timeout in every cancelled row.
 *
 * Cumulative across `judge()` calls, so one judge per arm accumulates the arm's
 * totals. Two invariants a scorer can rely on:
 *
 * - `rung1 + rung2` is exactly the number of findings returned across every
 *   call. Rungs are counted where a finding is EMITTED, not where a quote
 *   happens to resolve, so a dropped duplicate does not inflate the
 *   distribution the bake-off reads as evidence strength.
 * - `segmentsJudged + failedClosed + segmentsSkipped` is exactly the number of
 *   segments handed to every `judge()` call THAT HAD SEGMENT-SCOPED WORK. That
 *   is the denominator a findings-per-segment or recall number needs, and
 *   without `segmentsSkipped` it could not be computed at all -- see that
 *   field. A call with no such work counts nothing, and deliberately: an IR
 *   with no `semanticPredicates` -- or none the policy declared
 *   `scope: "segment"` -- spends no per-segment engine call and judges no
 *   segment, so filing its segments as skipped would conflate a policy with no
 *   segment-scoped clauses with a run a stop cut short. A malformed IR does not
 *   reach the loop either -- it throws.
 *
 *   The three `messageScope*` counters are OUTSIDE that sum, which is why they
 *   are separate fields rather than message-scope events folded into
 *   `failedClosed`: the whole-message call is not a segment, and adding it to a
 *   segment denominator is the units error `BaselineStats` exists to avoid one
 *   package over. Their own invariant is
 *   `messageScopeJudged + messageScopeFailedClosed <= 1` per `judge()` call,
 *   with the gap being a stop.
 */
export interface JudgeStats {
  /**
   * Findings whose quote occurred exactly once in the segment, at the ladder's
   * first rung. The strong case -- but "verbatim" would overstate it, and used
   * to: rung 1 matches in FOLDED space, so a quote differing from the passage
   * only in capitalisation, in the length of its whitespace runs, or in smart
   * versus ASCII punctuation lands here too. MEASURED against this ladder:
   * "NORTHWIND TRADERS RENEWAL" and "Northwind   Traders\n renewal" both
   * resolve at rung 1 against a passage reading "Northwind Traders renewal". So
   * an arm whose model rewrites case scores identically here to one that copies
   * exactly, and this counter cannot tell them apart. `spans.ts` owns the fold.
   */
  readonly rung1: number;
  /**
   * Findings whose quote only matched after the ladder peeled its tail.
   * Weaker evidence, and reported as such. NOT word-aligned: `spans.ts` peels
   * one code point at a time, so a rung-2 span can end inside a word.
   */
  readonly rung2: number;
  /**
   * Quotes the ladder refused, for ANY of its reasons. Four of them share this
   * counter: the quote is absent from the segment; it occurs more than once, at
   * either rung, and ambiguity is refused rather than guessed; the peel reached
   * the word floor with no unique candidate; or every candidate's boundary
   * would have split a surrogate pair. They share it because `resolveQuote`
   * returns a bare `undefined` for all four and reports no reason -- telling
   * them apart means widening its return type, which nothing downstream has yet
   * asked for.
   *
   * `resolveQuote`'s fifth refusal, a quote that folds to nothing, cannot reach
   * this counter: `JudgeResponseSchema` requires a non-whitespace character in
   * `quote` and folding never empties such a string, so an empty quote is
   * refused a stage earlier and lands in `failedClosed` or `repairAttempts`.
   */
  readonly unresolvedQuotes: number;
  /** Findings naming a predicate the IR does not declare. Models invent ids. */
  readonly unknownPredicates: number;
  /** Findings resolving to a span this run had already emitted. */
  readonly duplicatesDropped: number;
  /** Segments that got a second call because the first answer would not parse. */
  readonly repairAttempts: number;
  /**
   * Segments the engine ANSWERED that still yielded no judgement. Two causes
   * share this counter, and they share it because both leave a segment unjudged
   * with the model in the loop: a body still unparseable after the one repair
   * retry, and a body the engine marked `"abort"` -- the latched engine, which
   * also ends the run. Segments never REACHED are `segmentsSkipped` and not
   * this: folding those in would report a model failure for a call that never
   * exercised the model.
   */
  readonly failedClosed: number;
  /**
   * Completions the engine reported cut off, by `finishReason === "length"`.
   *
   * NOT only `max_tokens`, which is what this line used to say. READ from the
   * shipped 0.2.84 bundle, "Stop condition 4" sets the same `finishReason` when
   * `filledKVCacheLength` reaches `contextWindowSize`, so a prompt that fits at
   * prefill and then exhausts the window mid-answer lands here too and is
   * indistinguishable from a verbose model. At tier-2 segment sizes that is
   * unlikely rather than impossible; it is the Approach-B arm, whose prompt
   * carries a whole policy, where it is a live cause.
   */
  readonly truncatedResponses: number;
  /** Completions the engine reported as interrupted. */
  readonly abortedResponses: number;
  /**
   * Segments whose answer parsed and was collected. This is the denominator for
   * findings-per-segment and for recall, and `segments.length` is not, because
   * a run can stop before it reaches the end of its list.
   */
  readonly segmentsJudged: number;
  /**
   * Segments this run never asked about because a stop ended it early, plus the
   * one whose own call raised that stop -- it produced no answer either.
   *
   * Nothing else records this. `deadlineExpiries` is 1 whether the budget blew
   * on segment 1 of 40 or on segment 39, so a scorer computing recall from an
   * early-stopped run had a wrong denominator and no way to notice.
   */
  readonly segmentsSkipped: number;
  /**
   * Engine calls this run ISSUED with the WHOLE MESSAGE as the passage -- the
   * cost of honouring `SemanticPredicate.scope: "message"`, and the only place
   * it is countable.
   *
   * At most 2 per `judge()` call: one for every message-scoped predicate
   * together (never one per predicate), plus the same single repair retry a
   * segment gets. 0 when the policy declares no message-scoped predicate.
   *
   * ISSUED, not answered, and the difference is the point: a call a stop cut
   * short spent real model time and produced no row in `calls`, so counting
   * answers here would report the feature as free in exactly the runs where it
   * blew the budget. Where the two differ is recoverable --
   * `messageScopeJudged + messageScopeFailedClosed` counts the answers, and
   * `repairAttempts` is shared with the segment loop so it cannot separate them
   * on its own.
   */
  readonly messageScopeCalls: number;
  /**
   * 0 or 1: the whole-message call answered and its findings were collected.
   *
   * NOT the same claim `scopesJudged` makes. That names a scope this run ASKED
   * the model about, so it survives a stop mid-answer (see `judge()`); this
   * counts the asks that came back with something to collect. A run with
   * `scopesJudged` naming "message" and a 0 here is one whose message-scope
   * call was issued and produced no judgement.
   */
  readonly messageScopeJudged: number;
  /**
   * 0 or 1: the whole-message call answered and still yielded no judgement --
   * a body unparseable after one repair retry, or a body the engine marked
   * `"abort"` (the latched engine, which also ends the run).
   *
   * SEPARATE from `failedClosed` rather than folded into it. `failedClosed` is
   * documented and consumed as a count of SEGMENTS, and it is one of the three
   * terms of the segment invariant above; a message-scope failure landing there
   * would make that sum exceed the number of segments handed over, which is the
   * denominator a recall number is built from.
   */
  readonly messageScopeFailedClosed: number;
  /**
   * Runs stopped by a budget expiry. NOT a per-call rate: `judge()` returns on
   * the first `DeadlineExpired`, so this is at most 1 per call by construction
   * and `deadlineExpiries / segments` is not a number that means anything. What
   * it does decide is whether an arm is too slow to finish its corpus.
   */
  readonly deadlineExpiries: number;
  /**
   * Runs the caller's `AbortSignal` stopped after generation had started: a
   * generation really was interrupted, and `runWithDeadline` cleared the flag
   * it set.
   */
  readonly callerAbortsMidGeneration: number;
  /**
   * Runs the caller's `AbortSignal` stopped while the call was still QUEUED
   * behind another on the same engine. Nothing ran and nothing was interrupted.
   *
   * Split from the counter above because `cancel.ts` carries `interrupted`
   * precisely to separate them -- one message for both "would state a falsehood
   * in two of them" -- and because a queued abort spent no model time at all,
   * which a latency row must not attribute to the model.
   */
  readonly callerAbortsWhileQueued: number;
  /**
   * One row per engine call that ANSWERED, in the order they were made, repair
   * retries included. A call stopped by a budget or an abort produces no row,
   * because it produced no response to copy one from; those are the four stop
   * counters above.
   */
  readonly calls: readonly JudgeCallRecord[];
}

export interface WebLlmJudgeOptions {
  /**
   * Per engine CALL, not per message: one segment's budget is not spent by the
   * segment before it. `engine.complete` measures it from the moment the call
   * reaches the engine.
   */
  readonly budgetMs: number;
}

/** Every field of `JudgeStats` except the per-call rows, which are their own array. */
type Counters = Omit<JudgeStats, "calls">;
type MutableCounters = { -readonly [K in keyof Counters]: Counters[K] };

/**
 * The passage one engine call is shown, and the offset it starts at in the
 * message.
 *
 * A `Segment` is one of these structurally, and the whole message is the other:
 * `{ text, start: 0, end: text.length }`. Written as its own type rather than
 * synthesising a `Segment` for the message, because a `Segment` carries a
 * `kind` and a message spanning prose AND a fenced block is not any one of
 * them -- a synthetic `kind: "prose"` would be a field stating something the
 * segmenter never said.
 *
 * `start` is what makes span resolution correct for both: a resolution is
 * against `passage.text` and is then offset by `passage.start`, which is 0 for
 * the message, so a message-scoped finding's offsets stay absolute. Adding a
 * segment's start to a message-relative offset produces a span that still
 * slices cleanly and points at the wrong words -- the mis-location `spans.ts`
 * exists to make impossible rather than unlikely.
 */
export interface JudgedPassage {
  readonly text: string;
  /** Absolute offset of `text[0]` in the message. 0 for the whole message. */
  readonly start: number;
  /** Absolute offset one past `text`'s last character. */
  readonly end: number;
}

/** What one passage's call did, as the four outcomes its caller must tell apart. */
type PassageOutcome =
  /** The answer parsed and its findings were collected. */
  | "collected"
  /** The answer would not parse after one repair retry. The run continues. */
  | "failed-closed"
  /** The engine reported an abort we did not raise: it is latched, so the run ends. */
  | "latched"
  /** A `DeadlineExpired` -- a blown per-call budget or a caller abort. The run ends. */
  | "stopped";

const ZERO_COUNTERS: Counters = {
  rung1: 0,
  rung2: 0,
  unresolvedQuotes: 0,
  unknownPredicates: 0,
  duplicatesDropped: 0,
  repairAttempts: 0,
  failedClosed: 0,
  truncatedResponses: 0,
  abortedResponses: 0,
  segmentsJudged: 0,
  segmentsSkipped: 0,
  messageScopeCalls: 0,
  messageScopeJudged: 0,
  messageScopeFailedClosed: 0,
  deadlineExpiries: 0,
  callerAbortsMidGeneration: 0,
  callerAbortsWhileQueued: 0,
};

/**
 * The instructions, fixed for every call so two arms differ only by their model.
 *
 * The word floor is interpolated from `MINIMUM_CANDIDATE_WORDS` rather than
 * written out, so a second copy here cannot drift from the one enforced. Same
 * reason `ACTION_RANK` is exported from core's orchestrator instead of being
 * restated in the compiler.
 *
 * What that floor actually gates, since this comment used to overstate it and
 * the prompt repeated the overstatement to the model: it gates RUNG 2 only,
 * inside the peel loop. Rung 1 has no word check at all -- MEASURED against
 * this ladder, the one-word quote "Northwind" resolves at rung 1, and so does
 * the two-word "Northwind Traders", because both occur exactly once. Nothing
 * here refuses a short quote for being short.
 *
 * Asking for the whole clause anyway is still the right instruction, for two
 * reasons that are about uniqueness and recoverability rather than about a
 * refusal: a longer quote is likelier to occur exactly once, which is the only
 * thing rung 1 tests; and a quote already AT the floor leaves rung 2 nothing to
 * work with past its final word. MEASURED against a message reading "Ship the
 * deploy pin today ...": the three-word quote "the deploy pXn" recovers "the
 * deploy p", but "the dXploy pin" -- perturbed one word earlier -- recovers
 * nothing at all, because reaching a matching prefix would take the candidate
 * below the floor and the descent stops there.
 *
 * What is deliberately NOT in this prompt: the IR's `examples` and
 * `counterExamples`. Those are authored strings that in at least one corpus
 * held the very value the run was scored on, and a model handed the answer
 * scores without doing the work. Shadow entityTypes carry empty example lists
 * by construction, so reading them would buy nothing even where it were safe.
 */
const SYSTEM_PROMPT = [
  "You audit one passage of a message against a list of policy predicates.",
  "Report every span of the passage that satisfies one of the predicates.",
  "",
  "Answer with JSON and nothing else, in exactly this shape:",
  '{"findings":[{"predicateId":"<id>","quote":"<text from the passage>","confidence":<number 0 to 1>}]}',
  "",
  "Rules:",
  "- predicateId must be one of the ids listed below. Never invent one.",
  "- quote must be copied from the passage character for character, including",
  "  its punctuation and capitalisation.",
  `- Quote the whole clause that carries the evidence, and at least ${MINIMUM_CANDIDATE_WORDS} words.`,
  "  A quote that occurs more than once in the passage is discarded, not guessed at.",
  "- Never quote anything that is not in the passage.",
  '- If nothing in the passage satisfies any predicate, answer {"findings":[]}.',
].join("\n");

export class WebLlmJudge implements SemanticJudge {
  readonly #engine: Tier2Engine;
  readonly #budgetMs: number;
  readonly #counters: MutableCounters = { ...ZERO_COUNTERS };
  readonly #calls: JudgeCallRecord[] = [];

  /**
   * @param engine the `Tier2Engine` SEAM, never a `WebLlmEngine` directly. The
   *   concrete class has `#private` fields, which makes TypeScript type it
   *   nominally and a test double unassignable without `as never` -- a cast
   *   that asserts nothing about the thing it is applied to.
   * @throws when `budgetMs` is not a finite duration `setTimeout` can hold.
   *   `runWithDeadline` performs the same check, but not until the first engine
   *   call: MEASURED on Node 26, Infinity -- the natural spelling of "no
   *   budget" -- fires in under 1 ms, so a judge built with it interrupts a
   *   model that has not answered yet, once per segment, for a whole arm.
   *   Checked at construction so it fails before a bake-off starts rather than
   *   during it.
   *
   *   The two clauses catch disjoint things and both are load-bearing. The
   *   RANGE clause catches every bad number, and MEASURED on Node 26 those are
   *   the values `setTimeout` reinterprets: Infinity, NaN, 0, -1 and
   *   2147483648 each fired in under 2 ms while 2147483647 did not fire within
   *   400 ms. `Number.isFinite` catches something else entirely -- a numeric
   *   STRING, which compares fine against both bounds. MEASURED on the same
   *   run, `setTimeout(fn, "300")` fired at 302 ms: coerced and honoured, not
   *   collapsed. So a quoted number in an eval driver's JSON config would run
   *   correctly here and be recorded as a string in a field typed `number`,
   *   which is the record-states-something-other-than-fact defect rather than a
   *   timing one, and it is caught for that reason.
   */
  constructor(engine: Tier2Engine, options: WebLlmJudgeOptions) {
    if (
      !(
        Number.isFinite(options.budgetMs) &&
        options.budgetMs > 0 &&
        options.budgetMs <= MAX_BUDGET_MS
      )
    ) {
      throw new Error(
        `tier-2 judge budgetMs must be a finite number of milliseconds in ` +
          `(0, ${MAX_BUDGET_MS}], got ${options.budgetMs} (${typeof options.budgetMs}); ` +
          `setTimeout rejects no alternative -- it reinterprets a nonsense number as a ` +
          `1ms deadline and coerces a numeric string -- so a bad budget is never ` +
          `reported as one`,
      );
    }
    this.#engine = engine;
    this.#budgetMs = options.budgetMs;
  }

  /**
   * A snapshot, copied on every read, one level deep. The counters and the call
   * rows are this object's own state and a caller holding a live reference
   * could rewrite the numbers a bake-off row is built from -- so the array is
   * rebuilt and each row is re-spread, not just the outer object.
   */
  get stats(): JudgeStats {
    return { ...this.#counters, calls: this.#calls.map((row) => ({ ...row })) };
  }

  /**
   * Judge the message's predicates in the scope each one was DECLARED in: the
   * message-scoped ones against the whole message, in one call; the
   * segment-scoped ones against each segment, one call apiece.
   *
   * Returns absolute offsets into the MESSAGE, not into the passage it was
   * asked about: core re-derives nothing and `applyActions` rewrites by offset,
   * so a passage-relative span vaults a neighbouring word and leaves the real
   * one in the message.
   *
   * ## The partition, and why it is a partition rather than an addition
   *
   * `PredicateScope`'s own docblock in core says what a scope is: "what a
   * semantic predicate is asked about: one segment at a time, or the whole
   * message at once". So a `scope: "message"` predicate is NOT also sent per
   * segment. Sending it both ways would ask a declared-once question N+1 times,
   * put its findings in a scope the policy did not declare for it, and multiply
   * the arm's cost by the segment count for no reading of `scope` that the type
   * supports.
   *
   * The consequence is visible and worth stating rather than discovering: on a
   * policy whose predicates are ALL message-scoped -- which
   * `policies/compiled/p-fin.ir.json`, the only compiled policy in this
   * repository, is -- the segment loop makes no call at all, and this judge
   * costs ONE call per message rather than one per selected segment. On a
   * policy carrying both, it costs `selectedSegments + 1`.
   *
   * ## Why the message call goes FIRST
   *
   * It is exactly one call, known before the run; the segment list is not. Under
   * a budget that admits k calls -- and `p-fin` carries the compiler's real
   * 5,000 ms against Plan 5's measured ~4.6 s per call, so k is about 1 -- a
   * message-LAST order makes coverage of the policy's message clause depend on
   * how many segments the message happened to have: answered on short messages,
   * silently skipped on long ones, with the difference invisible in any
   * aggregate. Message-first also spends that one affordable call on the same
   * unit the Approach-B baseline spends its own single call on, so a budget that
   * admits one call does not end up comparing a segment answer against a
   * message answer.
   *
   * What it costs, honestly: on a message with segment-scoped work, the message
   * call takes the budget the first segment would have had, and a stop there
   * files every segment as `segmentsSkipped`. That is a reordering of which
   * work a tight budget loses, not new loss.
   *
   * Degrades rather than throwing on a slow or cancelled engine -- spec 5.3
   * says an over-budget tier-2 run falls back to the lower tiers' findings.
   * `verdict.degraded` is where that now shows up, and it travels: the
   * orchestrator stamps each notice with the tier and puts it on
   * `DetectionResult`, so a caller no longer has to hold this object to learn
   * that an empty `findings` is not the claim "this passage is clean".
   *
   * The counters are still the finer record and are not replaced by it: a
   * notice says a segment was not judged, while `failedClosed`,
   * `segmentsSkipped`, `deadlineExpiries` and the two caller-abort counters
   * accumulate across every message an arm ran, which is what a recall
   * denominator needs. One caller abort is deliberately in the counters and NOT
   * in the notices -- see the stop handling below.
   *
   * Anything OTHER than a `DeadlineExpired` from the engine propagates
   * untouched. `engine.ts` throws on a response with no choices and on one that
   * names no model, and both exist because folding them into an empty answer
   * would have a judge report the segment clean. Degrading those here would
   * undo the tripwire and, since such an error carries no `reason`, would file
   * the event as a caller abort as well.
   *
   * `scopesJudged` names a scope this run ASKED the model about -- it pushes
   * the word next to the call, before the answer. That is the reading
   * `JudgeVerdict.scopesJudged` already documents ("naming a scope does NOT
   * claim every predicate in it was reached: a run cut short reports that
   * separately, in `degraded`"), and the same rule is applied to both scopes so
   * a bake-off comparing them is not comparing two rules. A scope with no
   * declared predicate, and a scope whose call was never issued, are both
   * absent from it -- which is what turns "the judge never asked" into the
   * orchestrator's `scope-unjudged` notice and leaves "the judge asked and was
   * cut off" to the budget and failed-closed words that name the real cause.
   *
   * `request.budgetMs` is likewise READ BY NOTHING here. The per-call budget
   * stays the one this judge was constructed with; what bounds the message is
   * `request.signal`, which the orchestrator raises when `ir.latencyBudgetMs`
   * expires and which the stop handling below already treats as a caller abort.
   * Sizing per-call budgets from what is left of the message is a change to how
   * this judge spends its time, and it belongs with the task that measures the
   * result.
   *
   * That is also why a blown budget here reports `call-budget-exhausted` and
   * not `budget-exhausted`: this judge cannot observe the message's budget at
   * all, so naming it would be a record stating a number nothing here measured.
   */
  async judge(request: JudgeRequest): Promise<JudgeVerdict> {
    const { text, segments, ir, priorFindings, signal } = request;
    const predicates = ir.semanticPredicates;
    // A policy with no semantic clauses is legitimate, and asking a 2 GB model
    // about nothing costs seconds per message. Checked before anything else so
    // no call is spent. `scopesJudged` is empty rather than `["segment"]`: no
    // predicate was evaluated in any scope, and the orchestrator only asks
    // about scopes the policy declares a predicate in, so accuracy costs
    // nothing here.
    if (predicates.length === 0) return { findings: [], scopesJudged: [] };

    const severityOf = shadowSeverities(ir, predicates);
    // The partition. Both halves are computed before either call, so a policy
    // that declares only one scope never reaches the other's loop at all --
    // which is what makes "no segment-scoped predicate" cost zero segment
    // calls rather than N calls carrying an empty predicate list.
    const messageScoped = predicates.filter((p) => p.scope === "message");
    const segmentScoped = predicates.filter((p) => p.scope === "segment");

    const findings: Finding[] = [];
    // Per RUN, like `emitted` below and unlike the counters: a notice names the
    // passage it is about, and an array that outlived the call would attach one
    // message's failures to the next message's result.
    const notices: EngineDegradedNotice[] = [];
    // Per CALL, never per instance, and shared by BOTH scopes' calls. Offsets
    // repeat across messages -- [18, 43) is [18, 43) in every one of them -- so
    // de-duplication state that outlived a judge() call would silently delete
    // the next message's findings.
    //
    // Sharing it across the two scopes is safe under the key
    // `start:end:entityType` for a reason the partition above supplies: an
    // entityType here is `pred:<predicateId>` and a predicate has exactly ONE
    // scope, so no entityType is carried by both the message call and the
    // segment calls. Two findings colliding on this key therefore came from the
    // same predicate in the same scope and are a restatement, which is what the
    // counter is for. Two DIFFERENT predicates that name the same span survive
    // as two findings, and should: they are two policy clauses, and core's
    // cluster resolution is what decides the action for the overlap.
    const emitted = new Set<string>();
    // Pushed beside the CALL, not beside the answer -- see the docblock.
    const scopesJudged: PredicateScope[] = [];
    // Frozen on the way out for the reason the old shared constant was: it is
    // returned by reference, and a caller that sorted it in place would rewrite
    // what this run claims. A fresh array per call, so nothing else can leak.
    const verdict = (): JudgeVerdict => ({
      findings,
      scopesJudged: Object.freeze([...scopesJudged]),
      degraded: notices,
    });

    // How many segment calls this run still owes, which is what a stop during
    // the message call costs. 0 when there is no segment-scoped predicate: the
    // loop below would not have run, so nothing was skipped.
    const segmentCallsOwed = segmentScoped.length > 0 ? segments.length : 0;

    if (messageScoped.length > 0) {
      scopesJudged.push("message");
      const outcome = await this.#judgePassage({
        // `start: 0` is the whole point: a resolution against this passage is
        // ALREADY absolute, and `#collect` adds `passage.start` to it. See
        // `JudgedPassage`.
        passage: { text, start: 0, end: text.length },
        // Every message-scoped predicate in ONE call, matching what the segment
        // loop does per segment. One call per predicate would multiply the cost
        // by the number of clauses for no gain the batched prompt does not
        // already give.
        predicates: messageScoped,
        priorFindings,
        signal,
        severityOf,
        emitted,
        findings,
        notices,
        unit: "the whole message",
        lost:
          segmentCallsOwed > 0
            ? `that call and this message's ${segmentCallsOwed} segment(s) were not judged`
            : `the ${messageScoped.length} message-scoped predicate(s) were not judged`,
        onCallIssued: () => {
          this.#counters.messageScopeCalls += 1;
        },
      });
      if (outcome === "collected") this.#counters.messageScopeJudged += 1;
      if (outcome === "failed-closed" || outcome === "latched") {
        this.#counters.messageScopeFailedClosed += 1;
      }
      if (outcome === "latched" || outcome === "stopped") {
        // Both end the RUN -- a latched engine answers every later call
        // instantly and emptily, and a stop means nobody is waiting -- so the
        // segments this run would have asked about got no answer either.
        this.#counters.segmentsSkipped += segmentCallsOwed;
        return verdict();
      }
    }

    if (segmentScoped.length > 0 && segments.length > 0) {
      scopesJudged.push("segment");
      for (let index = 0; index < segments.length; index += 1) {
        const outcome = await this.#judgePassage({
          passage: segments[index]!,
          predicates: segmentScoped,
          priorFindings,
          signal,
          severityOf,
          emitted,
          findings,
          notices,
          unit: `segment ${index + 1} of ${segments.length}`,
          lost: `that segment and the ${segments.length - index - 1} after it were not judged`,
        });
        if (outcome === "collected") {
          this.#counters.segmentsJudged += 1;
          continue;
        }
        if (outcome === "failed-closed") {
          this.#counters.failedClosed += 1;
          continue;
        }
        if (outcome === "latched") {
          this.#counters.failedClosed += 1;
          this.#counters.segmentsSkipped += segments.length - index - 1;
          return verdict();
        }
        // A stop. The segment this call was for is counted as skipped along
        // with the ones after it: it got no answer either.
        this.#counters.segmentsSkipped += segments.length - index;
        return verdict();
      }
    }

    return verdict();
  }

  /**
   * One passage, one prompt, one repair retry, and the four ways it can end.
   *
   * ONE implementation for both scopes, deliberately. The two calls differ in
   * exactly two things -- the passage and which predicates it carries -- so
   * everything else (the pinned request shape, the parse, the repair, the four
   * stop counters, the call row, the truncation and abort counts) is shared by
   * construction rather than by two loops kept in step by review. A bake-off
   * that compared the two scopes' resolvable rates against differently-strict
   * parsers would be comparing the parsers.
   *
   * What it deliberately does NOT do is COUNT the passage: `segmentsJudged`,
   * `failedClosed`, `segmentsSkipped` and the three `messageScope*` counters
   * are the caller's, because the same outcome means different things about
   * different units and a shared counter is exactly how a message-scope failure
   * would end up inside a segment denominator.
   *
   * @param unit names the passage in a notice -- "segment 2 of 5", "the whole
   *   message". Never the passage's TEXT, which is the string this system
   *   exists to keep out of logs.
   * @param lost what a run-ending stop costs BEYOND this call, as a clause the
   *   notice appends. Computed by the caller, which is the only side that knows
   *   what work was still owed.
   */
  async #judgePassage(input: {
    readonly passage: JudgedPassage;
    readonly predicates: readonly SemanticPredicate[];
    readonly priorFindings: readonly Finding[];
    readonly signal: AbortSignal | undefined;
    readonly severityOf: ReadonlyMap<string, Severity>;
    readonly emitted: Set<string>;
    readonly findings: Finding[];
    readonly notices: EngineDegradedNotice[];
    readonly unit: string;
    readonly lost: string;
    readonly onCallIssued?: () => void;
  }): Promise<PassageOutcome> {
    let messages = buildMessages(input.passage, input.predicates, input.priorFindings);
    let repaired = false;

    for (;;) {
      let completion: Tier2Completion;
      input.onCallIssued?.();
      try {
        completion = await this.#engine.complete(messages, {
          budgetMs: this.#budgetMs,
          signal: input.signal,
        });
      } catch (cause) {
        // NOT a stop we own, so not ours to degrade. `engine.ts` throws on a
        // response with no choices and on one naming no model precisely
        // because folding either into an empty answer would report the
        // passage CLEAN; swallowing it here would restore that false negative
        // and, since such an error has no `reason`, file it as a caller abort
        // on top.
        if (!(cause instanceof DeadlineExpired)) throw cause;
        // Every stop ends the RUN, not just this passage, and the honest
        // reason is narrower than "the model is too slow". A budget expiry
        // says THIS call did not finish in THIS budget; whether the next
        // one would is unknown, because passages differ in length and a
        // short one can meet a budget a long one blew. What is known is that
        // we cannot tell a slow model from a slow passage from here, and
        // continuing risks spending a full budget per remaining passage to
        // find out. A caller abort needs no such argument: nobody is waiting
        // for any of it. Either way the message degrades to whatever the
        // lower tiers found, which is what has been collected.
        if (cause.reason === "budget") {
          this.#counters.deadlineExpiries += 1;
          // `call-budget-exhausted`, never `budget-exhausted`. The number that
          // expired is `this.#budgetMs`, fixed when this judge was
          // constructed; `ir.latencyBudgetMs` may have almost all of itself
          // left, and the orchestrator files its own word for that from the
          // timer IT armed. One word for both would have a bake-off count
          // per-call model slowness against the spec 5.3 message budget.
          input.notices.push({
            reason: "call-budget-exhausted",
            detail:
              `the tier-2 call for ${input.unit} did not answer within its ` +
              `${this.#budgetMs}ms per-call budget; ${input.lost}`,
          });
        } else if (cause.interrupted) this.#counters.callerAbortsMidGeneration += 1;
        else this.#counters.callerAbortsWhileQueued += 1;
        // No notice on either caller abort, deliberately. The caller withdrew
        // and already knows; the orchestrator files its own budget notice
        // from the timer IT armed, and a second one from here would report
        // one event twice. This judge also cannot say why a caller withdrew
        // -- a latency budget is only one of the reasons -- so any reason it
        // named would be a guess in a field that must state fact.
        return "stopped";
      }

      // One row per answered call, repair retries included, and the
      // message-scope call is in here like any other: `JudgeCallRecord` is
      // where an arm's token and TTFT columns come from, and a call kept out of
      // it is a cost the bake-off cannot see.
      this.#calls.push(completionCallRecord(completion));

      // Counted from what the ENGINE reported, before and independently of
      // whether the body parsed: a truncated response that happens to parse
      // is still a response the model did not finish, and the bake-off reads
      // this to tell a budget-killed arm from an incapable one.
      if (completion.finishReason === "length") this.#counters.truncatedResponses += 1;
      if (completion.finishReason === "abort") this.#counters.abortedResponses += 1;

      // `finishReason` is threaded rather than left out: Task 2 measured
      // inferring truncation from the thrown SyntaxError's wording putting 6
      // of 16 boundary cases in the wrong bucket, and "length" is the engine
      // stating that IT cut the response off.
      const parsed = parseJudgeResponse(completion.content, completion.finishReason);
      if (parsed.ok) {
        this.#collect(
          parsed.value,
          input.passage,
          completion.model,
          input.severityOf,
          input.emitted,
          input.findings,
        );
        return "collected";
      }

      // "aborted" is handled apart from "truncated", DELIBERATELY, and the
      // decision is not to retry. Task 3 measured the engine state behind it:
      // an interrupt sets an engine-wide flag that the non-streaming path
      // never clears, and every later call returns instantly with an empty
      // body and finish_reason "abort" until something writes the flag back.
      // `runWithDeadline` clears the flag it set itself, so an "abort"
      // arriving here is an interrupt WE did not raise -- the engine is
      // latched, a repair retry would be answered instantly and emptily, and
      // so would every remaining passage. Retrying would also file the event
      // as a model failure when nothing about the model was exercised.
      //
      // Nothing here recovers the engine, and that is worth stating rather
      // than leaving to be discovered: `clearInterrupt` is deliberately NOT
      // on the `Tier2Engine` seam, and `runWithDeadline` clears only the flag
      // it set itself. So a judge holding a latched engine fails this way on
      // its NEXT message too, and the one after -- one wasted call per
      // message, indefinitely, until whoever set the flag clears it or the
      // engine is rebuilt.
      if (parsed.reason === "aborted") {
        input.notices.push({
          reason: "failed-closed",
          detail:
            `the engine reported the response for ${input.unit} aborted by an interrupt this ` +
            `judge did not raise; the engine is latched, so ${input.lost}`,
        });
        return "latched";
      }

      if (repaired) {
        // One repair, then fail closed. A model that will not emit valid JSON
        // twice must never have its prose passed through as a judgement.
        //
        // `parsed.reason` is one of `schema.ts`'s fixed words. `parsed.detail`
        // is NOT included: it is built from the body the model produced, and
        // a model that will not emit JSON is emitting rearranged message text.
        input.notices.push({
          reason: "failed-closed",
          detail:
            `the answer for ${input.unit} could not be parsed after one repair retry ` +
            `(${parsed.reason}), so it was not judged`,
        });
        return "failed-closed";
      }
      repaired = true;
      this.#counters.repairAttempts += 1;
      messages = [...messages, repairMessage(parsed.reason, parsed.detail)];
    }
  }

  #collect(
    response: JudgeResponse,
    passage: JudgedPassage,
    model: string,
    severityOf: ReadonlyMap<string, Severity>,
    emitted: Set<string>,
    out: Finding[],
  ): void {
    for (const finding of response.findings) {
      const severity = severityOf.get(finding.predicateId);
      // A model invents predicate ids. Passing one through makes
      // normalizeFindings throw, which loses the message including the findings
      // that were fine -- so it is dropped here, where it costs one finding.
      if (severity === undefined) {
        this.#counters.unknownPredicates += 1;
        continue;
      }

      // Against the PASSAGE's text, which is the only text the model was shown.
      // Searching more than that would let a quote resolve somewhere the model
      // never read; searching less would refuse a message-scoped quote that
      // straddles two segments, which is the whole reason the message call
      // exists.
      const resolved = resolveQuote(passage.text, finding.quote);
      if (resolved === undefined) {
        this.#counters.unresolvedQuotes += 1;
        continue;
      }

      const entityType = shadowIdFor(finding.predicateId);
      // The PASSAGE's own start, which is 0 for the whole message -- so a
      // message-scoped resolution is already absolute and is not shifted. A
      // segment's start added to a message-relative offset produces a span that
      // slices cleanly, satisfies core's `text === message.slice(start, end)`
      // check, and points at the wrong words.
      const start = passage.start + resolved.start;
      const end = passage.start + resolved.end;
      // `start` and `end` are integers, so the first two colons separate the
      // key unambiguously whatever an entityType contains.
      const key = `${start}:${end}:${entityType}`;
      if (emitted.has(key)) {
        // Models restate a finding, and Plan 5's own probe corpus caught one
        // looping a single finding until its token budget ran out. Two findings
        // over one span are one piece of evidence counted twice.
        this.#counters.duplicatesDropped += 1;
        continue;
      }
      emitted.add(key);
      if (resolved.rung === 1) this.#counters.rung1 += 1;
      else this.#counters.rung2 += 1;

      out.push({
        start,
        end,
        // Sliced by the ladder from the segment, never assembled from the
        // model's quote: core requires text === message.slice(start, end), and
        // a segment's text is by construction the message's slice.
        text: resolved.text,
        entityType,
        // From the IR's shadow entityType. Core re-derives severity anyway, so
        // a value invented here would be invisible until a policy changed it.
        severity,
        tier: 2,
        // The model that ANSWERED, from the completion. Never
        // `engine.requestedModelId`, which is what was asked for: a record
        // naming a model that never ran is the intent-as-fact defect this
        // project has shipped twice, and `Tier2Engine` deliberately does not
        // expose `loadedModelId` so the wrong choice is a compile error.
        source: model,
        // Passed through, not clamped. `JudgeResponseSchema` is the boundary
        // that validated it -- `z.number().min(0).max(1)` rejects NaN, both
        // infinities and everything outside the range -- so every value
        // reaching here is already a finite number in [0, 1] and a clamp would
        // be unreachable code claiming to protect merge's ordering. Rejecting
        // rather than clamping is also the deliberate choice recorded on the
        // schema: clamping a model's `95` to 1.0 would turn a misread scale
        // into a maximally confident finding.
        confidence: finding.confidence,
      });
    }
  }
}

/**
 * Copy the four numbers a bake-off row needs off one completion.
 *
 * `engine.ts` went to real trouble to pass `usage` through verbatim, NaN caveat
 * included, and the judge is the only holder of a `Tier2Completion` inside a
 * `detect()` run -- so anything not copied here is unobtainable downstream.
 *
 * No `?? 0` anywhere, deliberately. Every one of these is `undefined` when the
 * engine reported no `usage`, and `undefined` is the honest value for "not
 * reported": a 0 would say the call used no prompt tokens and answered
 * instantly, which is a claim about the model rather than about a missing
 * field. Seconds become milliseconds and nothing else is transformed.
 *
 * Exported for the Approach-B arm, which records the same rows for the same
 * bake-off. Two projections would let the arms' token and TTFT columns drift --
 * one arm converting seconds and the other not, one defaulting a missing usage
 * to 0 -- and the comparison would then be between the projections.
 */
export function completionCallRecord(completion: Tier2Completion): JudgeCallRecord {
  const ttftSeconds = completion.usage?.extra.time_to_first_token_s;
  return {
    finishReason: completion.finishReason,
    promptTokens: completion.usage?.prompt_tokens,
    completionTokens: completion.usage?.completion_tokens,
    ttftMs: ttftSeconds === undefined ? undefined : ttftSeconds * 1000,
    // Copied, not converted: the library already reports tokens per SECOND, so
    // the unit the gate is written in is the unit the engine hands over.
    decodeTokPerSec: completion.usage?.extra.decode_tokens_per_s,
  };
}

/**
 * Every predicate's shadow entityType severity, and a refusal when one is
 * missing.
 *
 * THROWS rather than skipping. A predicate the compiler never minted a shadow
 * for cannot reach an action: every finding naming it is rejected by
 * `normalizeFindings`, so the clause is present in the policy and silently does
 * nothing. A policy clause that evaporates is the failure class this project
 * exists to prevent, and core's orchestrator already documents that detection
 * throws rather than degrading quietly. Raised before any engine call, so a
 * malformed IR costs nothing.
 */
function shadowSeverities(
  ir: PolicyIr,
  predicates: readonly SemanticPredicate[],
): ReadonlyMap<string, Severity> {
  const severities = new Map<string, Severity>();
  for (const predicate of predicates) {
    const shadowId = shadowIdFor(predicate.id);
    const entity = ir.entityTypes.find((e) => e.id === shadowId);
    if (entity === undefined) {
      throw new Error(
        `IR declares semanticPredicate "${predicate.id}" but no shadow entityType ` +
          `"${shadowId}"; every tier-2 finding for it would be rejected by ` +
          `normalizeFindings, so the clause would silently do nothing`,
      );
    }
    severities.set(predicate.id, entity.severity);
  }
  return severities;
}

/**
 * One prompt per passage: a system turn holding the instructions and a user
 * turn holding this passage's predicates, context and text.
 *
 * IDENTICAL in shape for a segment and for the whole message, and that is a
 * decision rather than an omission. The two calls are meant to be comparable --
 * same instructions, same JSON contract, same quoting rule, same word floor --
 * so the only things that differ are the predicates the policy declared for
 * that scope and the passage they are asked about. A second system prompt for
 * the message call would be a second thing to keep in step, and would make the
 * two scopes' resolvable and duplicate rates incomparable within one arm.
 *
 * Nothing tells the model which scope it is in, deliberately: a message-scoped
 * predicate's own `nlPredicate` says "the message" (p-fin's reads "The message
 * discloses that a named organisation is..."), and the instruction that matters
 * -- quote from the passage in front of you -- is the same either way.
 *
 * The shape is dictated by the library, not by taste. READ from the shipped
 * 0.2.84 bundle: `postInitAndCheckFields` throws `SystemMessageOrderError` for
 * a system message at any index but 0, throws `MessageOrderError` unless the
 * LAST message is `user` or `tool`, and reaches that check through
 * `messages[messages.length - 1].role` with no length guard -- so an empty list
 * surfaces as a bare TypeError from inside the library.
 */
function buildMessages(
  passage: JudgedPassage,
  predicates: readonly SemanticPredicate[],
  priorFindings: readonly Finding[],
): ChatCompletionMessageParam[] {
  const lines = ["Predicates:"];
  for (const predicate of predicates) {
    lines.push(`- ${predicate.id}: ${predicate.nlPredicate}`);
  }
  lines.push("", priorFindingsLine(passage, priorFindings), "", "Passage:", passage.text);
  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: lines.join("\n") },
  ];
}

/**
 * What the lower tiers already found in this segment, as LABELS AND COUNTS and
 * never as text.
 *
 * A prior finding's `text` is precisely the value tier 0 or tier 1 detected. Put
 * in the prompt it becomes an answer key the model can score against without
 * doing the work, which is the leak this project has already shipped once -- a
 * corpus whose only tier-1 gold value was also the IR's `examples` entry. The
 * label says what kind of thing is nearby, which is the whole context a judge
 * needs; the passage itself carries the value.
 *
 * Restricted to priors that OVERLAP this passage, since a label from three
 * paragraphs away is noise about a passage the model cannot see. For the
 * whole-message call that is every prior on the message, which is the same set
 * Approach B's own prompt carries -- so the two arms' models are told the same
 * thing about the same lower tier when they are shown the same text.
 *
 * Exported for the Approach-B-plus-tier-0 arm, which hands its model the same
 * context about the same lower tier. The leak this function exists to prevent
 * is the one that matters most for a bake-off, so there is one implementation
 * of it rather than one per arm; B passes the whole message as its passage,
 * which is exactly what its model was shown.
 */
export function priorFindingsLine(
  passage: JudgedPassage,
  priorFindings: readonly Finding[],
): string {
  const counts = new Map<string, number>();
  for (const prior of priorFindings) {
    if (prior.start >= passage.end || prior.end <= passage.start) continue;
    counts.set(prior.entityType, (counts.get(prior.entityType) ?? 0) + 1);
  }
  if (counts.size === 0) return "Earlier tiers found nothing in this passage.";
  const parts = [...counts].map(([entityType, count]) => `${entityType} (x${count})`);
  return `Earlier tiers already flagged, in this passage: ${parts.join(", ")}.`;
}

/**
 * The one repair turn, appended after the original user turn.
 *
 * A `user` message and not an `assistant` one carrying the broken output: the
 * body that failed is the thing that caused the failure, and on the common
 * failure -- truncation, 3 of 6 calls in one probe run -- re-sending it spends
 * the same token budget again and makes a second truncation MORE likely, not
 * less. The parse error names what was wrong without repeating it.
 *
 * Two consecutive `user` messages are permitted: READ from the 0.2.84 bundle,
 * neither `postInitAndCheckFields` nor `getConversationFromChatCompletionRequest`
 * requires roles to alternate, and the prompt assembler renders one role prefix
 * per message in order. That is a claim about the library's source, not a
 * measurement of how a model answers such a prompt -- that needs a GPU and was
 * not run here.
 */
function repairMessage(reason: string, detail: string): ChatCompletionMessageParam {
  return {
    role: "user",
    content: [
      `Your previous answer could not be parsed (${reason}): ${detail}`,
      "Answer again with JSON only, in the shape given above, and nothing else.",
    ].join("\n"),
  };
}
