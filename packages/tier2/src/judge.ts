import type { ChatCompletionFinishReason, ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import { shadowIdFor } from "@sih/core";
import type { Finding, PolicyIr, SemanticJudge, SemanticPredicate, Segment, Severity } from "@sih/core";
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
 * segment. A single `finishReason` for the message would therefore be a fact
 * about one call presented as a fact about the message, and a summed `ttftMs`
 * would be a latency nothing ever experienced. So the rows are kept whole and
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
 *   segments handed to every `judge()` call THAT REACHED ITS SEGMENT LOOP. That
 *   is the denominator a findings-per-segment or recall number needs, and
 *   without `segmentsSkipped` it could not be computed at all -- see that
 *   field. A call that returned before the loop counts nothing, and
 *   deliberately: an IR with no `semanticPredicates` spends no engine call and
 *   judges nothing, so filing its segments as skipped would conflate a policy
 *   with no semantic clauses with a run a stop cut short. A malformed IR does
 *   not reach the loop either -- it throws.
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
  /** Completions the engine reported as cut off at `max_tokens`. */
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
   * Judge each segment against the IR's semantic predicates.
   *
   * Returns absolute offsets into the MESSAGE, not into the segment it was
   * asked about: core re-derives nothing and `applyActions` rewrites by offset,
   * so a segment-relative span vaults a neighbouring word and leaves the real
   * one in the message.
   *
   * Degrades rather than throwing on a slow or cancelled engine -- spec 5.3
   * says an over-budget tier-2 run falls back to the lower tiers' findings --
   * and the counters are where that shows up. An empty return is NOT the same
   * claim as "this passage is clean", and a caller reading recall from these
   * findings must read `failedClosed`, `segmentsSkipped`, `deadlineExpiries`
   * and the two caller-abort counters alongside them.
   *
   * Anything OTHER than a `DeadlineExpired` from the engine propagates
   * untouched. `engine.ts` throws on a response with no choices and on one that
   * names no model, and both exist because folding them into an empty answer
   * would have a judge report the segment clean. Degrading those here would
   * undo the tripwire and, since such an error carries no `reason`, would file
   * the event as a caller abort as well.
   *
   * `scope` on a `SemanticPredicate` is not honoured yet: every predicate is
   * judged per segment, including one declared `scope: "message"`. The judge is
   * handed segments, not the message, and the orchestrator is free to hand it a
   * FILTERED list (tier 1 already drops code segments, and Task 7 owns the
   * tier-2 filter), so the message cannot be reassembled here without
   * inventing text. The cost is real and is worth stating: evidence for a
   * message-scoped predicate that spans two segments is not seen by either
   * call.
   */
  async judge(
    segments: Segment[],
    ir: PolicyIr,
    priorFindings: Finding[],
    signal?: AbortSignal,
  ): Promise<Finding[]> {
    const predicates = ir.semanticPredicates;
    // A policy with no semantic clauses is legitimate, and asking a 2 GB model
    // about nothing costs seconds per message. Checked before anything else so
    // no call is spent.
    if (predicates.length === 0) return [];

    const severityOf = shadowSeverities(ir, predicates);
    const findings: Finding[] = [];
    // Per CALL, never per instance. Offsets repeat across messages -- [18, 43)
    // is [18, 43) in every one of them -- so de-duplication state that outlived
    // a judge() call would silently delete the next message's findings.
    const emitted = new Set<string>();

    for (let index = 0; index < segments.length; index += 1) {
      const segment = segments[index]!;
      let messages = buildMessages(segment, predicates, priorFindings);
      let repaired = false;

      for (;;) {
        let completion: Tier2Completion;
        try {
          completion = await this.#engine.complete(messages, {
            budgetMs: this.#budgetMs,
            signal,
          });
        } catch (cause) {
          // NOT a stop we own, so not ours to degrade. `engine.ts` throws on a
          // response with no choices and on one naming no model precisely
          // because folding either into an empty answer would report the
          // segment CLEAN; swallowing it here would restore that false negative
          // and, since such an error has no `reason`, file it as a caller abort
          // on top.
          if (!(cause instanceof DeadlineExpired)) throw cause;
          // Every stop ends the RUN, not just this segment, and the honest
          // reason is narrower than "the model is too slow". A budget expiry
          // says THIS call did not finish in THIS budget; whether the next
          // segment would is unknown, because segments differ in length and a
          // short one can meet a budget a long one blew. What is known is that
          // we cannot tell a slow model from a slow segment from here, and
          // continuing risks spending a full budget per remaining segment to
          // find out. A caller abort needs no such argument: nobody is waiting
          // for any of it. Either way the message degrades to whatever the
          // lower tiers found, which is what has been collected.
          //
          // The segment this call was for is counted as skipped along with the
          // ones after it: it got no answer either.
          this.#counters.segmentsSkipped += segments.length - index;
          if (cause.reason === "budget") this.#counters.deadlineExpiries += 1;
          else if (cause.interrupted) this.#counters.callerAbortsMidGeneration += 1;
          else this.#counters.callerAbortsWhileQueued += 1;
          return findings;
        }

        // One row per answered call, repair retries included. Recorded before
        // the body is looked at, because a call that will fail to parse still
        // spent its tokens and its time-to-first-token.
        this.#calls.push(callRecord(completion));

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
          this.#counters.segmentsJudged += 1;
          this.#collect(parsed.value, segment, completion.model, severityOf, emitted, findings);
          break;
        }

        // "aborted" is handled apart from "truncated", DELIBERATELY, and the
        // decision is not to retry. Task 3 measured the engine state behind it:
        // an interrupt sets an engine-wide flag that the non-streaming path
        // never clears, and every later call returns instantly with an empty
        // body and finish_reason "abort" until something writes the flag back.
        // `runWithDeadline` clears the flag it set itself, so an "abort"
        // arriving here is an interrupt WE did not raise -- the engine is
        // latched, a repair retry would be answered instantly and emptily, and
        // so would every remaining segment. Retrying would also file the event
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
          this.#counters.failedClosed += 1;
          this.#counters.segmentsSkipped += segments.length - index - 1;
          return findings;
        }

        if (repaired) {
          // One repair, then fail closed. A model that will not emit valid JSON
          // twice must never have its prose passed through as a judgement.
          this.#counters.failedClosed += 1;
          break;
        }
        repaired = true;
        this.#counters.repairAttempts += 1;
        messages = [...messages, repairMessage(parsed.reason, parsed.detail)];
      }
    }

    return findings;
  }

  #collect(
    response: JudgeResponse,
    segment: Segment,
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

      // Against the SEGMENT's text, which is the only text the model was shown.
      // Searching the whole message would let a quote resolve to a passage the
      // model never read.
      const resolved = resolveQuote(segment.text, finding.quote);
      if (resolved === undefined) {
        this.#counters.unresolvedQuotes += 1;
        continue;
      }

      const entityType = shadowIdFor(finding.predicateId);
      const start = segment.start + resolved.start;
      const end = segment.start + resolved.end;
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
 */
function callRecord(completion: Tier2Completion): JudgeCallRecord {
  const ttftSeconds = completion.usage?.extra.time_to_first_token_s;
  return {
    finishReason: completion.finishReason,
    promptTokens: completion.usage?.prompt_tokens,
    completionTokens: completion.usage?.completion_tokens,
    ttftMs: ttftSeconds === undefined ? undefined : ttftSeconds * 1000,
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
 * One prompt per segment: a system turn holding the instructions and a user
 * turn holding this segment's predicates, context and text.
 *
 * The shape is dictated by the library, not by taste. READ from the shipped
 * 0.2.84 bundle: `postInitAndCheckFields` throws `SystemMessageOrderError` for
 * a system message at any index but 0, throws `MessageOrderError` unless the
 * LAST message is `user` or `tool`, and reaches that check through
 * `messages[messages.length - 1].role` with no length guard -- so an empty list
 * surfaces as a bare TypeError from inside the library.
 */
function buildMessages(
  segment: Segment,
  predicates: readonly SemanticPredicate[],
  priorFindings: readonly Finding[],
): ChatCompletionMessageParam[] {
  const lines = ["Predicates:"];
  for (const predicate of predicates) {
    lines.push(`- ${predicate.id}: ${predicate.nlPredicate}`);
  }
  lines.push("", priorLine(segment, priorFindings), "", "Passage:", segment.text);
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
 * Restricted to priors that OVERLAP this segment, since a label from three
 * paragraphs away is noise about a passage the model cannot see.
 */
function priorLine(segment: Segment, priorFindings: readonly Finding[]): string {
  const counts = new Map<string, number>();
  for (const prior of priorFindings) {
    if (prior.start >= segment.end || prior.end <= segment.start) continue;
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
