import type { ChatCompletionMessageParam } from "@mlc-ai/web-llm";
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
 * Per-run counters, all of them facts about calls that actually happened.
 *
 * The bake-off reads these to tell an arm that judged well from one that failed
 * closed on every segment, so each counter names one event and no counter is
 * shared by two. The three stop-related counters in particular are separate on
 * purpose: a blown budget says this model is too slow for this arm, a caller
 * abort says nobody waited for the answer, and an aborted RESPONSE says the
 * engine was interrupted by something we did not do. Folding them together
 * would put a fabricated timeout in every cancelled row.
 *
 * Cumulative across `judge()` calls, so one judge per arm accumulates the arm's
 * totals. `rung1 + rung2` is exactly the number of findings returned across
 * every call: rungs are counted where a finding is EMITTED, not where a quote
 * happens to resolve, so a dropped duplicate does not inflate the distribution
 * the bake-off reads as evidence strength.
 */
export interface JudgeStats {
  /** Findings whose quote was in the segment verbatim. The strong case. */
  readonly rung1: number;
  /**
   * Findings whose quote only matched after the ladder peeled its tail.
   * Weaker evidence, and reported as such. NOT word-aligned: `spans.ts` peels
   * one code point at a time, so a rung-2 span can end inside a word.
   */
  readonly rung2: number;
  /** Quotes the ladder refused: absent, ambiguous, or below its word floor. */
  readonly unresolvedQuotes: number;
  /** Findings naming a predicate the IR does not declare. Models invent ids. */
  readonly unknownPredicates: number;
  /** Findings resolving to a span this run had already emitted. */
  readonly duplicatesDropped: number;
  /** Segments that got a second call because the first answer would not parse. */
  readonly repairAttempts: number;
  /**
   * Segments the engine answered but that yielded no judgement: an unparseable
   * body still unparseable after the repair retry, or an aborted one. Segments
   * never REACHED, because a stop ended the run early, are deliberately not
   * counted here -- `deadlineExpiries` and `callerAborts` are where those show
   * up, and folding them together would report a model failure for a call that
   * never exercised the model.
   */
  readonly failedClosed: number;
  /** Completions the engine reported as cut off at `max_tokens`. */
  readonly truncatedResponses: number;
  /** Completions the engine reported as interrupted. */
  readonly abortedResponses: number;
  /** Calls that exceeded `budgetMs`. */
  readonly deadlineExpiries: number;
  /** Calls the caller's `AbortSignal` stopped. */
  readonly callerAborts: number;
}

export interface WebLlmJudgeOptions {
  /**
   * Per engine CALL, not per message: one segment's budget is not spent by the
   * segment before it. `engine.complete` measures it from the moment the call
   * reaches the engine.
   */
  readonly budgetMs: number;
}

type MutableStats = { -readonly [K in keyof JudgeStats]: JudgeStats[K] };

const ZERO_STATS: JudgeStats = {
  rung1: 0,
  rung2: 0,
  unresolvedQuotes: 0,
  unknownPredicates: 0,
  duplicatesDropped: 0,
  repairAttempts: 0,
  failedClosed: 0,
  truncatedResponses: 0,
  abortedResponses: 0,
  deadlineExpiries: 0,
  callerAborts: 0,
};

/**
 * The instructions, fixed for every call so two arms differ only by their model.
 *
 * The word floor is interpolated from `MINIMUM_CANDIDATE_WORDS` rather than
 * written out. The ladder REFUSES a candidate spanning fewer words than that,
 * so a prompt asking for fewer produces quotes the ladder throws away and a
 * report that blames the model for it; a second copy of the number here would
 * be free to drift from the one enforced. Same reason `ACTION_RANK` is exported
 * from core's orchestrator instead of being restated in the compiler.
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
  "  A shorter quote cannot be located in the passage and is discarded.",
  "- Never quote anything that is not in the passage.",
  '- If nothing in the passage satisfies any predicate, answer {"findings":[]}.',
].join("\n");

export class WebLlmJudge implements SemanticJudge {
  readonly #engine: Tier2Engine;
  readonly #budgetMs: number;
  readonly #stats: MutableStats = { ...ZERO_STATS };

  /**
   * @param engine the `Tier2Engine` SEAM, never a `WebLlmEngine` directly. The
   *   concrete class has `#private` fields, which makes TypeScript type it
   *   nominally and a test double unassignable without `as never` -- a cast
   *   that asserts nothing about the thing it is applied to.
   * @throws when `budgetMs` is not a finite duration `setTimeout` can hold.
   *   `runWithDeadline` performs the same check, but not until the first engine
   *   call: measured on Node 26, Infinity -- the natural spelling of "no
   *   budget" -- fires in 1-4 ms, so a judge built with it interrupts a model
   *   that has not answered yet, once per segment, for a whole arm. Checked at
   *   construction so it fails before a bake-off starts rather than during it.
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
          `(0, ${MAX_BUDGET_MS}], got ${options.budgetMs}; setTimeout silently turns ` +
          `anything else into a 1ms deadline`,
      );
    }
    this.#engine = engine;
    this.#budgetMs = options.budgetMs;
  }

  /**
   * A snapshot, copied on every read. The counters are this object's own state
   * and a caller holding a live reference could rewrite the numbers a bake-off
   * row is built from.
   */
  get stats(): JudgeStats {
    return { ...this.#stats };
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
   * findings must read `failedClosed`, `deadlineExpiries` and `callerAborts`
   * alongside them.
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

    for (const segment of segments) {
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
          if (!(cause instanceof DeadlineExpired)) throw cause;
          // Both stops end the RUN, not just this segment. A budget expiry says
          // the engine is too slow for this arm, and the next segment would
          // spend another full budget proving it again; a caller abort says
          // nobody is waiting for any of it. Either way the message degrades to
          // whatever the lower tiers found, which is what has been collected.
          if (cause.reason === "budget") this.#stats.deadlineExpiries += 1;
          else this.#stats.callerAborts += 1;
          return findings;
        }

        // Counted from what the ENGINE reported, before and independently of
        // whether the body parsed: a truncated response that happens to parse
        // is still a response the model did not finish, and the bake-off reads
        // this to tell a budget-killed arm from an incapable one.
        if (completion.finishReason === "length") this.#stats.truncatedResponses += 1;
        if (completion.finishReason === "abort") this.#stats.abortedResponses += 1;

        // `finishReason` is threaded rather than left out: Task 2 measured
        // inferring truncation from the thrown SyntaxError's wording putting 6
        // of 16 boundary cases in the wrong bucket, and "length" is the engine
        // stating that IT cut the response off.
        const parsed = parseJudgeResponse(completion.content, completion.finishReason);
        if (parsed.ok) {
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
        if (parsed.reason === "aborted") {
          this.#stats.failedClosed += 1;
          return findings;
        }

        if (repaired) {
          // One repair, then fail closed. A model that will not emit valid JSON
          // twice must never have its prose passed through as a judgement.
          this.#stats.failedClosed += 1;
          break;
        }
        repaired = true;
        this.#stats.repairAttempts += 1;
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
        this.#stats.unknownPredicates += 1;
        continue;
      }

      // Against the SEGMENT's text, which is the only text the model was shown.
      // Searching the whole message would let a quote resolve to a passage the
      // model never read.
      const resolved = resolveQuote(segment.text, finding.quote);
      if (resolved === undefined) {
        this.#stats.unresolvedQuotes += 1;
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
        this.#stats.duplicatesDropped += 1;
        continue;
      }
      emitted.add(key);
      if (resolved.rung === 1) this.#stats.rung1 += 1;
      else this.#stats.rung2 += 1;

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
