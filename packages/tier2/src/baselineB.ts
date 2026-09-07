import type { ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import { remainingBudgetMs, resolveFindings, runTier0, segmentText } from "@sih/core";
import type {
  DegradedNotice,
  DetectInput,
  DetectionResult,
  Finding,
  PolicyIr,
  Segment,
  TierConfig,
} from "@sih/core";
import { DeadlineExpired, MAX_BUDGET_MS } from "./cancel.js";
import type { Tier2Completion, Tier2Engine } from "./engine.js";
import { completionCallRecord, priorFindingsLine, type JudgeCallRecord } from "./judge.js";
import type { Tier2Config } from "./manifest.js";
import { BASELINE_B_SCHEMA, parseBaselineResponse, type BaselineResponse } from "./schema.js";
import { MINIMUM_CANDIDATE_WORDS, locateFinding } from "./spans.js";

/**
 * Approach B: the whole policy, the whole message, one model call, no compiler.
 *
 * This arm exists to THREATEN the compiler, and the project has no result until
 * it can. The thesis is that compiling a policy into an IR beats simply
 * prompting a model with the policy text; B is "simply prompting". If B matches
 * the compiled pipeline, the compiler is not earning its place -- and that is a
 * finding worth having. If B loses for a reason that is not its design, the
 * headline number is an artifact of this file.
 *
 * So the governing rule here is not "make B work", it is **every difference
 * between B and the compiled path is a defect unless it is intrinsic to the
 * method being compared**. What follows is the full list of differences that
 * remain, each with which kind it is. A reviewer's job is to disagree with any
 * of these that look like the harness rather than the method.
 *
 * ## Equal by construction, because B shares the engine
 *
 * B takes a `Tier2Engine` -- the same seam `WebLlmJudge` takes -- and calls
 * `complete`. That is deliberate and is the single biggest fairness lever in
 * the file: the model id, the context window, `temperature`, `max_tokens`,
 * grammar-constrained decoding, the non-streaming path, the interrupt-drain-
 * clear on cancellation and the per-engine serialization are then not "the same
 * for both arms by convention", they are the same object. The one thing B
 * changes is `response_format.schema`, through `CompleteOptions.responseSchemaJson`,
 * because B names an entity class where the judge names a predicate.
 *
 * Equal for the same reason, one level up: the one-repair-then-fail-closed
 * rule, the four-way parse classification (`parseBaselineResponse` is the
 * judge's parser with the other zod schema), the two-span placement
 * (`locateFinding`, the same rungs, the same four edge-case rules, the same
 * refusals, the same counters), the drop-and-count treatment of an invented
 * label, `normalizeFindings`, the overlap merge and the cluster-strictest
 * action resolution (`resolveFindings` in core, which `detect` itself now
 * calls).
 *
 * ## Intrinsic to the method, and therefore kept
 *
 * - **One call per MESSAGE, not one per segment.** B is not a tier and does not
 *   escalate; it is shown the message. That is what "simply prompting" means.
 * - **The whole policy in every prompt.** Not chunked -- see `assertFits`, which
 *   refuses rather than truncates. Chunking so that a clause is missing from the
 *   chunk that sees the message would make a violation undetectable IN
 *   PRINCIPLE, which rigs the comparison before a model loads.
 * - **A bigger haystack for the EVIDENCE clause, on the calls that are per
 *   segment.** B resolves a clause against the whole message; the judge
 *   resolves against the passage its model was shown, which is one segment for
 *   a segment-scoped predicate and the whole message for a message-scoped one.
 *   Uniqueness is harder in a longer string, so rung 1 is harder for B than for
 *   a segment call -- and equally hard for both on a message-scoped predicate.
 *   Each follows from what the model was shown, so neither is corrected here.
 *
 *   The MENTION step does not inherit that asymmetry, and the two-span split is
 *   what removed it: a mention is searched inside the clause the ladder just
 *   placed, and that clause is the same string whichever arm placed it. So
 *   `unresolvedMentions` is comparable between the arms in a way
 *   `unresolvedQuotes` is not.
 *
 * ## Scope: no longer a difference between the methods
 *
 * This bullet used to say `SemanticPredicate.scope` was where B was stronger,
 * and it WAS: the judge asked every predicate per segment, reported
 * `scopesJudged: ["segment"]`, and `detect` filed a `scope-unjudged` notice for
 * the message scope on every message. That was an unimplemented feature, not a
 * property of either method, and it is now implemented -- `WebLlmJudge` asks
 * message-scoped predicates once about the whole message, so the COMPILED
 * families file zero `scope-unjudged` on `p-fin` where they used to file one
 * per message (MEASURED, `test/baseline.spec.ts`).
 *
 * A B arm's zero on the same column is not the same fact and must not be read
 * as one. `scope-unjudged` is filed by core's `detect`, off
 * `unjudgedScopes(ir, verdict.scopesJudged)`, and this module returns a
 * `Detector` rather than a `SemanticJudge`: B never reaches that code, names no
 * scope, and would file zero under a judge that answered nothing at all. B's 0
 * is a structural constant; only the compiled arms' 0 moved.
 *
 * What remains true and is worth keeping: ONE notice per unjudged SCOPE, not
 * one per predicate -- MEASURED against `detect` with three message-scoped
 * predicates, which produced a single notice carrying "declares 3 semantic
 * predicate(s)" in its detail. So a bake-off sizing "how much more does the
 * compiled arm degrade" off notice CARDINALITY reads at most one row per scope
 * however many clauses went unevaluated; the count it wants is in the detail,
 * and `unjudgedScopes` in core's orchestrator is where that decision lives.
 *
 * ## The concession, stated so it can be judged
 *
 * Every `Finding` must name an `entityType` the IR declares or
 * `normalizeFindings` throws and the whole message is lost. B has no compiler
 * and so no vocabulary of its own. It is therefore handed
 * `ir.entityTypes.map(e => e.id)` -- **the ids and nothing else**.
 *
 * That is a real concession and it favours B: B is given the taxonomy the
 * compiler derived from the policy. What it is NOT given is the rest of the
 * compiler's output -- no `nlDefinition`, no `examples` or `counterExamples`, no
 * `rules`, no `actions`, no provider overrides, no `semanticPredicates` text.
 * The boundary is exactly "names, not knowledge", and it is where it is for two
 * reasons: below it B cannot produce a finding core will accept at all, and
 * above it B would be scored on the compiler's work. The examples in particular
 * are excluded on evidence rather than taste -- one corpus in this project
 * shipped its only gold tier-1 value as the IR's own `examples` entry, and a
 * model handed the answer scores without doing the work.
 *
 * ## The asymmetry that is NOT fixed here, and how to see it in the numbers
 *
 * **B has one message's worth of completion budget where the judge has one per
 * CALL, and how many calls the judge makes is the policy's business.**
 * `max_tokens` is fixed on the engine (`Tier2Config.maxTokens`, consumed by
 * `buildCallParams`), so it is spent per call rather than per message: a policy
 * with a segment-scoped clause and a message split into five selected segments
 * gives the compiled arm five times B's output allowance for the same input.
 *
 * The size of that handicap is therefore a property of the POLICY, and on the
 * only compiled policy in this repository it is zero. `p-fin`'s one predicate
 * is message-scoped, so the judge makes exactly one whole-message call and gets
 * exactly one call's worth of `maxTokens` -- the same allowance B gets, since
 * both arms are built from the same `Tier2Config` (MEASURED,
 * `test/baseline.spec.ts`: 3 answered calls and `truncatedResponses: 0` on each
 * of the four arms). Read this bullet as the shape of the asymmetry on a policy
 * with segment-scoped clauses, not as a handicap the numbers in this repository
 * were taken under.
 *
 * Plan 5 already recorded truncation as the dominant parse failure -- 3 of 6
 * constrained Phi-4-mini calls, all three at `finish_reason: "length"` -- at a
 * budget LARGER than the one pinned here, and B is the arm most exposed to it
 * wherever the gap is real.
 *
 * This module cannot correct it: raising `max_tokens` for B alone would trade
 * one asymmetry for another. What it does instead is make it visible, which is
 * the countermeasure the plan itself calls for. `BaselineStats.truncatedResponses`
 * counts every completion the engine cut off, and `BaselineStats.calls` carries
 * `finishReason` and `completionTokens` per call, so a bake-off can report an
 * arm killed by its token budget beside one that had nothing to say. Whoever
 * runs the bake-off owns the decision; this file owns not hiding it.
 */

/**
 * Stringified once, for the same reason `JUDGE_SCHEMA_JSON` is:
 * `response_format.schema` is declared `string` on 0.2.84, so this is a type
 * conversion and not a serialization convenience.
 */
const BASELINE_B_SCHEMA_JSON = JSON.stringify(BASELINE_B_SCHEMA);

/**
 * Characters per token assumed by the pre-flight fit check.
 *
 * NOT a measurement, and it must not be read as one. No tokenizer for these
 * models exists outside a multi-gigabyte model download, so nothing here has
 * counted a token. It is a deliberate UNDER-estimate of how many characters a
 * token covers, which makes the token estimate an OVER-estimate and the check
 * conservative -- it refuses earlier than the engine would, never later.
 *
 * For scale: Plan 5 records `p-corp.md` at 1,227 tokens -- ITS measurement, not
 * mine -- and MEASURED HERE that file is 5,705 UTF-16 code units, which is what
 * `String.length` counts and therefore what this check divides. (It is 5,747
 * BYTES, and the 42-unit gap is the section signs and em-dashes; the two are
 * not interchangeable and the byte count is the wrong one to divide.) So the
 * one policy anyone has counted runs at about 4.65 characters per token, and 2
 * is roughly 2.3x conservative against it.
 *
 * CHECKED AGAINST A REAL TOKENIZER, once, which nothing here could do when this
 * comment was written. `apps/eval/test/baseline.spec.ts` runs this arm against
 * `policies/p-fin.md` on `Qwen3.5-2B-q4f16_1-MLC` and the ENGINE reported
 * 1,407-1,451 prompt tokens over three messages. The exact prompt length is not
 * derivable from outside this module, but a LOWER bound is: the policy alone is
 * 5,272 UTF-16 code units and is in every prompt, so the real ratio is at least
 * 5,272 / 1,451 = 3.63 characters per token. That makes 2 at least 1.8x
 * conservative against a measurement rather than only against `p-corp.md`'s
 * arithmetic, and the direction is the intended one. It is a bound and not the
 * ratio; the true figure is higher, because the instructions, the entity ids and
 * the message are all in the numerator and none of them is counted here.
 *
 * The estimate does not have to be tight, because it is not the enforcement.
 * READ from the shipped 0.2.84 bundle, `LLMChatPipeline` throws
 * `ContextWindowSizeExceededError` when
 * `numPromptTokens + filledKVCacheLength > contextWindowSize`, so a prompt that
 * really is too long fails loudly at prefill either way. What this check buys is
 * that the failure names the POLICY as the cause, before a model call is spent,
 * and -- through the construction-time probe -- before a bake-off starts at all.
 */
const CHARS_PER_TOKEN_FLOOR = 2;

/**
 * B's instructions, to be read side by side with `SYSTEM_PROMPT` in `judge.ts`.
 *
 * The plan asks for exactly that comparison, so the two are kept as parallel as
 * the arms allow: every rule bullet below is the judge's with two substitutions
 * -- "passage" becomes "message", because B is shown the whole thing, and
 * "predicateId" becomes "entityType", because B names an entity class. The one
 * bullet that cannot be a substitution is the last: the judge's names
 * predicates, and B has none.
 *
 * WHERE THAT IS PINNED, and this paragraph has now understated it as well as
 * overstated it. The cross-arm check in `baselineB.test.ts` reads both prompts
 * off real calls rather than off exported constants and requires every RULE
 * BULLET of EACH arm's to appear in the other's with the two nouns substituted
 * -- so a bullet that drifts in one arm and not the other fails there. The
 * second of those two directions is new: the check used to substitute on the
 * judge's side only, which meant a judge bullet already carrying B's noun was
 * found in B's prompt unchanged. MEASURED before it was fixed, renaming
 * "passage" to "message" in the judge's rule bullets -- all four of them, or
 * one -- left the whole tier2 suite green. It is still relative and still
 * partial: it filters to lines starting `"- "` or two spaces, which excludes
 * the two task sentences, the JSON-only instruction, the wire-shape line and
 * the `Rules:` header, and it cannot fire on the two arms drifting together.
 * `prompts.test.ts` is the other half: it pins each arm's system turn line by
 * line against that arm's OWN contract, both directions -- a dropped
 * instruction and an added one each fail -- and each clause there keys on the
 * words carrying the instruction's polarity, so a rule REVERSED in both arms at
 * once fails there even though nothing relative can see it.
 *
 * The word floor is interpolated from `MINIMUM_CANDIDATE_WORDS` rather than
 * written out, so this copy cannot drift from the one the ladder enforces.
 * What that floor gates is rung 2 of the EVIDENCE ladder only, inside the peel;
 * rung 1 has no word check at all, and `mention` is not subject to it (see
 * `resolveMention`). Asking for the whole clause anyway is still right, for the
 * two reasons `judge.ts` gives: a longer quote is likelier to occur exactly
 * once, and a quote already at the floor leaves rung 2 nothing to work with.
 *
 * ## The mention bullets, and why they are word-for-word the judge's
 *
 * The four lines about `mention` carry NEITHER of the two nouns the arms differ
 * on -- not "passage"/"message", not "predicate"/"policy" -- so they are byte
 * identical in both prompts rather than identical after a substitution. That is
 * deliberate and it is the honest shape: the rule "point at the shortest run a
 * rewrite must cover, inside the clause you just quoted" is a property of the
 * SPAN CONTRACT, which both arms are scored against, and not of either method.
 * Wording it per arm would be inventing a difference in order to have one.
 *
 * The change applies to both arms for the reason the standing rule about never
 * moving the control does NOT forbid: this is a change to what is ASKED, made
 * identically on each side. Leaving B on the old clause convention while the
 * judge moved to mentions would score the two against different targets and
 * hand the compiled arm an unearned win on every exact-match and IoU column.
 *
 * The policy document itself is in the USER turn rather than here. The library
 * requires a system message at index 0 and nothing else, so either would load;
 * the user turn is where the message and the vocabulary already are, and
 * keeping the instructions fixed across every call is what lets two arms differ
 * only by their model.
 */
const BASELINE_B_SYSTEM_PROMPT = [
  "You audit one message against a policy document.",
  "Report every span of the message that the policy restricts.",
  "",
  "Answer with JSON and nothing else, in exactly this shape:",
  '{"findings":[{"entityType":"<id>","quote":"<clause from the message>","mention":"<part of that clause>","confidence":<number 0 to 1>}]}',
  "",
  "Rules:",
  "- entityType must be one of the ids listed below. Never invent one.",
  "- quote must be copied from the message character for character, including",
  "  its punctuation and capitalisation.",
  `- Quote the whole clause that carries the evidence, and at least ${MINIMUM_CANDIDATE_WORDS} words.`,
  "  A quote that occurs more than once in the message is discarded, not guessed at.",
  "- mention must be copied from inside quote, character for character.",
  "  Make mention the shortest run of words a rewrite must cover: a name, an identifier, a value.",
  "- When nothing shorter than the whole clause will do, repeat quote as mention.",
  "  A mention that occurs more than once inside quote is discarded, not guessed at.",
  "- Never quote anything that is not in the message.",
  '- If nothing in the message is restricted by the policy, answer {"findings":[]}.',
].join("\n");

/**
 * Per-run counters for one Approach-B arm, all of them facts about calls that
 * happened.
 *
 * Deliberately named after `JudgeStats` wherever the event is the same one, so
 * a bake-off can put the two arms in one table without a translation layer that
 * would be the place a definition quietly changes. Where the event genuinely
 * differs the name differs too: B counts MESSAGES judged rather than segments,
 * because B makes one call per message and a segment denominator would divide
 * by a number B never used.
 *
 * Cumulative across calls to the detector, so one arm accumulates the arm's
 * totals. Never a per-item number -- see `BaselineB.stats`.
 *
 * The invariant a scorer can rely on: `rung1 + rung2` is exactly the number of
 * findings B's span LADDER emitted across every message. Rungs are counted
 * where a finding is emitted, so a dropped duplicate does not inflate the rung
 * distribution the bake-off reads as evidence strength.
 *
 * That total is NOT `DetectionResult.findings.length`, and the gap is measured
 * rather than argued. `findings` is what `resolveFindings` returned, and it
 * clusters overlapping spans and keeps one winner per cluster: two overlapping
 * model quotes are two rung-1 emissions and ONE returned finding (probed:
 * rung1 2, findings 1). On the B+tier-0 arm `findings` additionally holds rows
 * the ladder never produced, and the shipped merge test is a case where tier
 * 0's tighter span wins the cluster outright -- rung1 1, and the single
 * returned finding is tier 0's. So divide rung counts by `rung1 + rung2`, never
 * by `findings.length`; used as a consistency check the other way round it
 * fires on correct runs.
 */
export interface BaselineStats {
  /** Findings whose quote occurred exactly once in the message, in folded space. */
  readonly rung1: number;
  /** Findings that only matched after the ladder peeled the quote's tail. */
  readonly rung2: number;
  /**
   * EVIDENCE quotes the ladder refused, for any of its reasons: absent from the
   * message, ambiguous at either rung, peeled to the word floor without a
   * unique candidate, or every candidate boundary splitting a surrogate pair.
   *
   * B is the arm most likely to accumulate these and the reason is structural
   * rather than a defect: it is the only arm shown a SECOND document, so it is
   * the only arm that can quote the policy back instead of the message. A quote
   * from the policy resolves against nothing here, which is the correct answer
   * -- the alternative is a span of the user's text the model never pointed at.
   *
   * A finding lost at the MENTION step is NOT here -- its clause resolved. See
   * `unresolvedMentions`.
   */
  readonly unresolvedQuotes: number;
  /**
   * Findings whose clause placed but whose MENTION did not: absent from the
   * clause the model itself quoted, occurring more than once inside it,
   * starting or ending between two alphanumeric characters (a mention
   * truncated inside the value it names), carrying no letter or digit at all,
   * or on a boundary that would split a surrogate pair. The judge's counter of
   * the same name counts the same event; `spans.ts` owns all five refusals, so
   * the two arms cannot diverge on them.
   *
   * B's bigger haystack does not reach here DIRECTLY -- the mention is searched
   * inside the already-placed clause rather than in the message. It does reach
   * here INDIRECTLY, and the claim this comment used to make, that the clause
   * "is the same size whichever arm placed it", is false: the placed clause is
   * the model's quote only at RUNG 1, and at rung 2 `resolveQuote` has peeled
   * the tail, so the mention's haystack is a strict PREFIX of what the model
   * wrote. MEASURED on the passage "Please rotate the staging key for Tamarind
   * Grocers today": perturbing the quote's last token gives a rung-2 clause of
   * 48 characters and REFUSES the mention "today"; leaving it alone gives a
   * rung-1 clause of 49 and places the same mention. Rung-2 frequency depends
   * on haystack size, the one dimension the arms differ on, so this counter is
   * comparable between arms only as far as their rung distributions match --
   * read it beside `rung1` and `rung2`.
   */
  readonly unresolvedMentions: number;
  /**
   * Findings whose mention resolved to the WHOLE evidence clause -- the model
   * answering that no smaller span will do. Legitimate for a clause with no
   * extractable entity, and the one path by which a model can restore the
   * whole-clause action spans this contract exists to end, so it is counted
   * rather than trusted. Read against `rung1 + rung2`.
   *
   * It watches only that direction. A model narrowing to the WRONG words inside
   * its own quote is invisible in every counter on this row; the judge's twin
   * docblock says why a counter cannot close that and what can.
   */
  readonly wholeClauseMentions: number;
  /**
   * Findings naming an entityType the IR does not declare. Models invent
   * labels; the judge's `unknownPredicates` counts the same event on the other
   * arm. Dropped rather than thrown on, because throwing loses the message
   * including the findings that were fine.
   */
  readonly unknownEntityTypes: number;
  /** Findings resolving to a span this message had already emitted. */
  readonly duplicatesDropped: number;
  /** Messages that got a second call because the first answer would not parse. */
  readonly repairAttempts: number;
  /**
   * Messages the engine ANSWERED that still yielded no judgement: a body still
   * unparseable after the one repair retry, or a body the engine marked
   * "abort", which is the latched engine and also ends the run.
   */
  readonly failedClosed: number;
  /**
   * Completions the engine reported cut off, by `finishReason === "length"`.
   *
   * The counter that makes this arm's worst structural disadvantage visible.
   * See the module docblock: B spends one completion budget on a whole message
   * where the compiled arm spends one per segment, so an arm killed by
   * `max_tokens` must be distinguishable from an arm with nothing to say.
   *
   * It has TWO causes and cannot separate them. READ from the 0.2.84 bundle,
   * `finishReason` is set to "length" both when `max_tokens` is reached and
   * when the KV cache reaches the context window ("Stop condition 4"). For this
   * arm the second is a live possibility rather than a footnote, because its
   * prompt carries the whole policy on every call -- so read a high count as
   * "the answer was cut off", never as "raise max_tokens" on its own.
   * `assertFits` reserves `maxTokens` up front partly to keep the second cause
   * out of reach.
   */
  readonly truncatedResponses: number;
  /** Completions the engine reported as interrupted. */
  readonly abortedResponses: number;
  /** Messages whose answer parsed and was collected. The recall denominator. */
  readonly messagesJudged: number;
  /**
   * Runs stopped by B's own PER-CALL budget -- `BaselineBOptions.budgetMs`,
   * fixed when the arm was built. Same meaning as `JudgeStats.deadlineExpiries`
   * and named the same for that reason. Says nothing about
   * `ir.latencyBudgetMs`, which can be almost entirely unspent when this fires.
   */
  readonly deadlineExpiries: number;
  /**
   * Runs stopped by the MESSAGE's `ir.latencyBudgetMs`.
   *
   * The judge has no counterpart because it cannot see that budget: `detect`
   * arms the deadline and files the notice. B is its own orchestrator, so B is
   * the only thing that can count it, and without it a bake-off reading only
   * `deadlineExpiries` would report every over-budget B message as clean.
   */
  readonly messageBudgetExpiries: number;
  /** Runs the deadline signal stopped after generation had started. */
  readonly callerAbortsMidGeneration: number;
  /** Runs the deadline signal stopped while the call was still queued. */
  readonly callerAbortsWhileQueued: number;
  /**
   * One row per engine call that ANSWERED, in order, repair retries included.
   * The same shape the judge records, from the same projection, so the two
   * arms' token and TTFT columns cannot drift apart.
   */
  readonly calls: readonly JudgeCallRecord[];
}

type Counters = Omit<BaselineStats, "calls">;
type MutableCounters = { -readonly [K in keyof Counters]: Counters[K] };

const ZERO_COUNTERS: Counters = {
  rung1: 0,
  rung2: 0,
  unresolvedQuotes: 0,
  unresolvedMentions: 0,
  wholeClauseMentions: 0,
  unknownEntityTypes: 0,
  duplicatesDropped: 0,
  repairAttempts: 0,
  failedClosed: 0,
  truncatedResponses: 0,
  abortedResponses: 0,
  messagesJudged: 0,
  deadlineExpiries: 0,
  messageBudgetExpiries: 0,
  callerAbortsMidGeneration: 0,
  callerAbortsWhileQueued: 0,
};

export interface BaselineBOptions {
  /**
   * The `Tier2Engine` SEAM, never a `WebLlmEngine` directly -- the concrete
   * class has `#private` fields, so TypeScript types it nominally and a test
   * double is unassignable without `as never`.
   *
   * Sharing this type with the judge is what makes the two arms' decoding
   * settings equal by construction rather than by convention. See the module
   * docblock.
   */
  readonly engine: Tier2Engine;
  /**
   * The config the engine was LOADED under -- the same object handed to
   * `createWebLlmEngine`.
   *
   * Needed because the fit check has to know the window and how much of it the
   * answer will claim, and 0.2.84 exposes no accessor for either: nothing can
   * be read back off a loaded engine. Two configs in one call is how a record
   * ends up describing a run that never happened, so the constructor refuses an
   * engine whose `requestedModelId` disagrees with `config.modelId` -- the same
   * guard, for the same reason, that `createWebLlmEngine` performs.
   */
  readonly config: Tier2Config;
  /** The policy document, verbatim and entire. Never chunked; see `assertFits`. */
  readonly policyText: string;
  /**
   * Per engine CALL, not per message, exactly as `WebLlmJudgeOptions.budgetMs`
   * is. It is NOT clamped to what is left of `ir.latencyBudgetMs`, and that is
   * deliberate: `detect` does not clamp the judge's either, and clamping would
   * arm two timers on the same deadline and make which notice gets filed --
   * `call-budget-exhausted` or `budget-exhausted` -- a race. The message budget
   * is enforced by the signal instead, which is a stop with one owner.
   */
  readonly budgetMs: number;
}

/**
 * A `Detector` with the arm's counters hanging off it.
 *
 * The counters are NOT on `DetectionResult`: that is core's type, B must
 * satisfy `Detector` exactly, and an arm that widened the shared result type
 * would be an arm the harness reads through a second code path.
 *
 * `stats` is CUMULATIVE, and Task 11's `tier2Stats` is a per-ITEM field, so the
 * two do not connect by assignment. MEASURED by reading `stats.rung1` after
 * each of four identical items through one detector: 1, 2, 3, 4 -- every
 * counter is the arm's running total, which is what `BaselineStats` says and
 * what `accumulates across messages` pins. A per-item row therefore has to be a
 * DELTA: snapshot before the call, subtract after, the way `tier1Stats` is
 * built. Stamping the snapshot straight onto the row puts running totals on
 * every item, which inflates a counter summed over an n-item corpus by about
 * (n+1)/2 -- and the counters the amended kill rules read (`failedClosed`,
 * `truncatedResponses`, `deadlineExpiries`) are exactly the ones that would
 * then report an arm failing on more messages than the corpus contains.
 * `WebLlmJudge.stats` accumulates identically, so whatever Task 11 does it must
 * do to both arms; one arm on a delta and the other on a total is a
 * head-to-head that means nothing.
 */
export interface BaselineB {
  (input: DetectInput): Promise<DetectionResult>;
  readonly stats: BaselineStats;
}

/**
 * Approach B alone: no tiers at all, one model call against the whole policy.
 *
 * @throws when `budgetMs` is not a duration `setTimeout` can hold, when the
 *   engine and the config name different models, or when the policy alone
 *   cannot fit the window. All three are refused at CONSTRUCTION so a bake-off
 *   fails before it starts rather than partway through.
 */
export function createBaselineB(options: BaselineBOptions): BaselineB {
  return createArm(options, false);
}

/**
 * Approach B with core's tier 0 in front of it -- the mandatory intermediate
 * arm.
 *
 * Required rather than optional, because without it the head-to-head conflates
 * two different claims: "compiling the policy helps" and "having deterministic
 * patterns helps". B alone versus the full pipeline cannot separate them.
 *
 * The composition is deliberately not a bespoke union. Tier 0 runs over core's
 * own segmentation, B runs over the same message, and the two finding sets go
 * into `resolveFindings` -- the same normalization, the same overlap clustering
 * and the same cluster-strictest action resolution `detect` applies to tiers 0,
 * 1 and 2. A union would let this arm keep two findings where the compiled
 * pipeline keeps one, and the extra row would read as recall.
 *
 * The model is also told WHAT tier 0 found, as labels and counts and never as
 * the text, through the same `priorFindingsLine` the judge uses. Without it
 * this arm would be a strictly weaker composition than the compiled pipeline,
 * where tier 2 is handed `priorFindings`.
 *
 * The `ir` the plan's signature passes here is deliberately absent: `DetectInput`
 * already carries one, and an arm holding a second could run tier 0 against a
 * policy the record names something else. One IR, the caller's.
 */
export function createBaselineBPlusTier0(options: BaselineBOptions): BaselineB {
  return createArm(options, true);
}

function createArm(options: BaselineBOptions, withTier0: boolean): BaselineB {
  const { engine, config, policyText, budgetMs } = options;

  // Checked at construction, exactly as `WebLlmJudge`'s is, and for the
  // measured reason: `setTimeout` does not reject a nonsense number, it
  // reinterprets one as a ~1 ms deadline, so Infinity -- the natural spelling
  // of "no budget" -- would interrupt a model that has not answered yet, once
  // per message, for a whole arm. `Number.isFinite` additionally catches a
  // numeric STRING, which compares fine against both bounds and which
  // `setTimeout` coerces and honours: that would time correctly and be recorded
  // as a string in a field typed `number`.
  if (!(Number.isFinite(budgetMs) && budgetMs > 0 && budgetMs <= MAX_BUDGET_MS)) {
    throw new Error(
      `Approach B budgetMs must be a finite number of milliseconds in ` +
        `(0, ${MAX_BUDGET_MS}], got ${budgetMs} (${typeof budgetMs})`,
    );
  }
  if (engine.requestedModelId !== config.modelId) {
    throw new Error(
      `Approach B was given an engine loaded for "${engine.requestedModelId}" under a config ` +
        `pinned to "${config.modelId}"; the fit check would read a window the engine was not ` +
        `loaded at, and the record would name a model that never ran`,
    );
  }
  // The floor of what every call will send: the instructions and the policy,
  // with no vocabulary and no message. A policy failing here can never fit, so
  // it is refused before an arm is built rather than on the first item of a
  // corpus.
  assertFits(
    BASELINE_B_SYSTEM_PROMPT.length + userTurn(policyText, [], "", undefined).length,
    config,
    "the policy document and the fixed instructions alone",
  );

  const counters: MutableCounters = { ...ZERO_COUNTERS };
  const calls: JudgeCallRecord[] = [];

  const run = async (input: DetectInput): Promise<DetectionResult> => {
    const { ir, provider, text, config: tiers } = input;
    assertArmConfig(tiers, withTier0);

    // The message's own clock, started before tier 0 because tier 0 is work the
    // user waits through. `ir.latencyBudgetMs` is a per-MESSAGE number
    // (spec 5.3) and B is the only thing on this arm that can enforce it.
    const messageStarted = performance.now();
    const degraded: DegradedNotice[] = [];
    const timings: DetectionResult["timings"] = { tier0Ms: 0 };
    const raw: Finding[] = [];

    // In tier order, so a caller reading the array top to bottom finds the
    // earliest thing that weakened the result first -- the order `detect`
    // produces and the one `DetectionResult.degraded` documents.
    if (!withTier0) {
      degraded.push(absent(0));
    } else {
      const started = performance.now();
      // Core's own tier 0 over core's own segmentation, so this arm's
      // deterministic half is the SAME deterministic half the compiled
      // pipeline runs. A reimplementation here would make "having patterns
      // helps" a statement about two different pattern layers.
      raw.push(...runTier0(ir, text, segmentText(text)));
      timings.tier0Ms = performance.now() - started;
    }
    // Always, on both arms: B has no span tagger and never will. `absent` is a
    // statement of coverage, not of failure.
    degraded.push(absent(1));

    const priors = withTier0 ? priorFindingsLine(wholeMessage(text), raw) : undefined;
    let messages = buildBaselineMessages(
      policyText,
      ir.entityTypes.map((e) => e.id),
      text,
      priors,
    );
    // Re-checked per message, because the message is part of what has to fit
    // and the construction probe could only see the policy.
    assertFits(promptChars(messages), config, "the policy document and this message");

    const remaining = remainingBudgetMs(ir.latencyBudgetMs, performance.now() - messageStarted);
    if (remaining === undefined) {
      // Not called at all, rather than called with a budget of zero: a
      // `setTimeout(0)` fires in about 1 ms, so the call would be started and
      // aborted before its first token -- one model call spent to produce
      // nothing. `timings.tier2Ms` stays unset for the same reason it does in
      // `detect`: a 0 there reads as a tier that ran instantly.
      degraded.push({
        tier: 2,
        reason: "budget-exhausted",
        detail:
          `the ${ir.latencyBudgetMs}ms message latency budget was already spent before the ` +
          `Approach-B model call, so no call was made`,
      });
      return { findings: resolveFindings(ir, provider, text, raw), timings, degraded };
    }

    // A DEADLINE, not a race -- Plan 5 measured that racing a timeout against a
    // WebLLM call wedges the engine, and `cancel.ts` holds the interrupt, drain
    // and clear that make the signal the only safe stop.
    const controller = new AbortController();
    let expired = false;
    const deadline = setTimeout(() => {
      expired = true;
      controller.abort();
    }, remaining);
    const notices: DegradedNotice[] = [];
    // Per MESSAGE, never per arm. Offsets repeat across messages -- [18, 43) is
    // [18, 43) in every one of them -- so de-duplication state that outlived a
    // call would silently delete the next message's findings.
    const emitted = new Set<string>();
    const started = performance.now();

    try {
      let repaired = false;
      for (;;) {
        let completion: Tier2Completion;
        try {
          completion = await engine.complete(messages, {
            budgetMs,
            signal: controller.signal,
            responseSchemaJson: BASELINE_B_SCHEMA_JSON,
          });
        } catch (cause) {
          // Not a stop we own, so not ours to degrade. `engine.ts` throws on a
          // response with no choices and on one naming no model precisely
          // because folding either into an empty answer would report the
          // message CLEAN.
          if (!(cause instanceof DeadlineExpired)) throw cause;
          if (cause.reason === "budget") {
            counters.deadlineExpiries += 1;
            // `call-budget-exhausted`, never `budget-exhausted`: what expired
            // is the per-call budget this arm was built with, and the message
            // budget may have almost all of itself left. The message's own word
            // is filed below, from the timer THIS function armed.
            notices.push({
              tier: 2,
              reason: "call-budget-exhausted",
              detail:
                `the Approach-B call did not answer within its ${budgetMs}ms per-call budget, ` +
                `so this message was not judged`,
            });
          } else if (cause.interrupted) counters.callerAbortsMidGeneration += 1;
          else counters.callerAbortsWhileQueued += 1;
          // No notice on either abort here, deliberately and for the reason
          // `judge.ts` gives: the only signal B passes is the one it armed from
          // `ir.latencyBudgetMs`, and that event is filed once, below, from the
          // flag. A second notice would report one stop twice.
          break;
        }

        // Recorded before the body is looked at, because a call that will fail
        // to parse still spent its tokens and its time-to-first-token.
        calls.push(completionCallRecord(completion));
        // From what the ENGINE reported, independently of whether the body
        // parsed: a truncated response that happens to parse is still one the
        // model did not finish.
        if (completion.finishReason === "length") counters.truncatedResponses += 1;
        if (completion.finishReason === "abort") counters.abortedResponses += 1;

        const parsed = parseBaselineResponse(completion.content, completion.finishReason);
        if (parsed.ok) {
          counters.messagesJudged += 1;
          collect(parsed.value, ir, text, completion.model, emitted, raw, counters);
          break;
        }

        if (parsed.reason === "aborted") {
          // Not retried, and the reason is measured rather than stylistic: an
          // interrupt sets an engine-wide flag the non-streaming path never
          // clears, so a retry is answered instantly with an empty body and so
          // is every later message. Retrying would also file the event as a
          // model failure when nothing about the model was exercised.
          counters.failedClosed += 1;
          notices.push({
            tier: 2,
            reason: "failed-closed",
            detail:
              `the engine reported the Approach-B response aborted by an interrupt this arm did ` +
              `not raise; the engine is latched, so this message was not judged`,
          });
          break;
        }

        if (repaired) {
          // One repair, then fail closed -- the judge's rule, unchanged. A
          // model that will not emit valid JSON twice must never have its prose
          // passed through as a judgement.
          counters.failedClosed += 1;
          // `parsed.reason` is one of `schema.ts`'s fixed words. `parsed.detail`
          // is NOT included: it is built from the body the model produced, and
          // a model that will not emit JSON is emitting rearranged message text.
          notices.push({
            tier: 2,
            reason: "failed-closed",
            detail:
              `the Approach-B response could not be parsed after one repair retry ` +
              `(${parsed.reason}), so this message was not judged`,
          });
          break;
        }
        repaired = true;
        counters.repairAttempts += 1;
        messages = [...messages, repairMessage(parsed.reason, parsed.detail)];
      }
    } finally {
      // In `finally` because the throw path is the one that leaks. A leaked
      // timer here cannot abort a later message -- `controller` is built per
      // call -- but `setTimeout` returns a REF'd handle, so a batch harness
      // lingers for the rest of every abandoned budget.
      clearTimeout(deadline);
    }
    timings.tier2Ms = performance.now() - started;

    // Recorded from the TIMER, not from what the engine returned: an engine
    // that ignores the signal and answers in full still ran past the budget,
    // and that is a fact about this message's latency either way.
    if (expired) {
      counters.messageBudgetExpiries += 1;
      degraded.push({
        tier: 2,
        reason: "budget-exhausted",
        detail:
          `the ${ir.latencyBudgetMs}ms message latency budget ran out during the Approach-B ` +
          `call: it was given the ${remaining.toFixed(0)}ms that remained of it and was then ` +
          `sent an abort`,
      });
    }
    degraded.push(...notices);

    return { findings: resolveFindings(ir, provider, text, raw), timings, degraded };
  };

  // The property is attached before the cast is used for anything, so the
  // assertion is backed by the line above it rather than asserted into
  // existence. A getter and not a value: the counters are live, and a snapshot
  // taken at construction would report zeros for the whole run.
  const detector = run as BaselineB;
  Object.defineProperty(detector, "stats", {
    enumerable: true,
    // Copied on every read, one level deep: the counters and the call rows are
    // this arm's own state, and a caller holding a live reference could rewrite
    // the numbers a bake-off row is built from. The array is rebuilt and each
    // row re-spread, not just the outer object.
    get: (): BaselineStats => ({ ...counters, calls: calls.map((row) => ({ ...row })) }),
  });
  return detector;
}

/**
 * Refuse a `TierConfig` that describes a different run from the one this arm
 * performs.
 *
 * Not pedantry: the harness records the `TierConfig` it passed, so a config
 * that disagrees with the arm is a row asserting coverage the arm never had --
 * the records-state-intent defect, in the one place nothing downstream could
 * catch it.
 *
 * `tier2` must be TRUE, and the word means "a model read this message", not
 * "the compiled tier-2 judge ran". Every `Finding` B emits carries `tier: 2`,
 * because `Finding.tier` is `0 | 1 | 2` and a model produced it, so `false`
 * would contradict every row the arm writes.
 *
 * WHICH arm ran is NOT this flag, and it is no longer the record's free-text
 * `arm` field either. `RunRecordSchema` in apps/eval/src/driver/record.ts grew
 * a required `detector` -- `"core-orchestrator"` or `"approach-b"` -- when the
 * page gained a door onto this file, precisely because `config.tier2` is true
 * on both methods and cannot separate them. That field is what the record's
 * `tier2Stats`/`baselineStats` couplings key off; the paragraph here used to
 * name `config.tier2 && error === null` as the coupling and that was true only
 * while B had no record field of its own.
 */
function assertArmConfig(config: TierConfig, withTier0: boolean): void {
  if (config.tier1) {
    throw new Error(
      "Approach B was handed a TierConfig with tier 1 enabled; this arm has no span tagger, " +
        "so the row would claim coverage nothing produced",
    );
  }
  if (!config.tier2) {
    throw new Error(
      "Approach B was handed a TierConfig with tier 2 disabled; every finding it emits carries " +
        "tier 2 and its stats are recorded under that flag, so false would contradict the row " +
        "it appears on",
    );
  }
  if (config.tier0 !== withTier0) {
    throw new Error(
      `Approach B ${withTier0 ? "with" : "without"} tier 0 was handed a TierConfig with tier 0 ` +
        `${config.tier0 ? "enabled" : "disabled"}; use ` +
        `${config.tier0 ? "createBaselineBPlusTier0" : "createBaselineB"} rather than recording ` +
        `an arm as something it is not`,
    );
  }
}

/**
 * The tier that did not run, recorded as the fact it is.
 *
 * Worded as `detect`'s is, because a consumer reads both from the same field
 * and a second phrasing for the same event reads as a second event. WHY a tier
 * is off -- an arm of an experiment, no WebGPU, a model that failed to load --
 * is not knowledge this function has.
 */
function absent(tier: 0 | 1): DegradedNotice {
  return {
    tier,
    reason: "absent",
    detail: `tier ${tier} was not enabled in this TierConfig, so nothing it detects was looked for`,
  };
}

/** The whole message as one passage, which is exactly what B's model is shown. */
function wholeMessage(text: string): Segment {
  return { start: 0, end: text.length, kind: "prose", text };
}

/**
 * The user turn: the policy, the vocabulary, optionally what tier 0 found, and
 * the message.
 *
 * The vocabulary is IDS ONLY. See the module docblock for why that boundary is
 * where it is; the mechanical half is that `EntityType` also carries
 * `nlDefinition`, `examples` and `counterExamples`, and interpolating the
 * object rather than the id would hand a model the compiler's work and, in at
 * least one corpus this project has shipped, the answer itself.
 */
function userTurn(
  policyText: string,
  entityTypeIds: readonly string[],
  text: string,
  priorLine: string | undefined,
): string {
  const lines = ["Policy document:", policyText, "", "Entity class ids:"];
  for (const id of entityTypeIds) lines.push(`- ${id}`);
  if (priorLine !== undefined) lines.push("", priorLine);
  lines.push("", "Message:", text);
  return lines.join("\n");
}

/**
 * The two-message shape the library requires.
 *
 * READ from the shipped 0.2.84 bundle: `postInitAndCheckFields` throws
 * `SystemMessageOrderError` for a system message at any index but 0, throws
 * `MessageOrderError` unless the LAST message is `user` or `tool`, and reaches
 * that check through `messages[messages.length - 1].role` with no length guard.
 */
function messagesFor(userContent: string): ChatCompletionMessageParam[] {
  return [
    { role: "system", content: BASELINE_B_SYSTEM_PROMPT },
    { role: "user", content: userContent },
  ];
}

/**
 * The two turns B sends for one message, composed once.
 *
 * Extracted from `createArm`'s own call site rather than written beside it, and
 * `createArm` now goes through here -- so the capability-ceiling arm
 * (apps/eval/src/driver/ceiling.ts), which sends B's prompt over a hosted API
 * instead of to WebLLM, cannot drift from the arm this package ships. A second
 * composition would make the ceiling comparison a comparison of two prompts.
 */
export function buildBaselineMessages(
  policyText: string,
  entityTypeIds: readonly string[],
  text: string,
  priorLine: string | undefined,
): ChatCompletionMessageParam[] {
  return messagesFor(userTurn(policyText, entityTypeIds, text, priorLine));
}

/**
 * The one repair turn, appended after the original user turn -- the judge's
 * rule and the judge's reasoning.
 *
 * A `user` message and not an `assistant` one carrying the broken output: on
 * the common failure, truncation, re-sending the body spends the same token
 * budget again and makes a second truncation MORE likely, not less. That
 * argument is sharper for B than for the judge, since B has one completion
 * budget for a whole message.
 */
export function repairMessage(reason: string, detail: string): ChatCompletionMessageParam {
  return {
    role: "user",
    content: [
      `Your previous answer could not be parsed (${reason}): ${detail}`,
      "Answer again with JSON only, in the shape given above, and nothing else.",
    ].join("\n"),
  };
}

function promptChars(messages: readonly ChatCompletionMessageParam[]): number {
  return messages.reduce(
    (n, m) => n + (typeof m.content === "string" ? m.content.length : JSON.stringify(m.content).length),
    0,
  );
}

/**
 * Refuse a prompt that cannot fit, rather than truncating one that does not.
 *
 * The plan's countermeasure for the first and worst way to rig this comparison:
 * a silently truncated policy produces an arm that loses for a reason nobody
 * can see in the numbers. There is no chunking anywhere in this file, and this
 * is the function that makes that a promise rather than an omission.
 *
 * `maxTokens` is subtracted because the window has to hold the answer too, and
 * READ from the shipped 0.2.84 bundle the two ends of that window fail
 * DIFFERENTLY, which is the part worth knowing:
 *
 * - At prefill, the pipeline throws `ContextWindowSizeExceededError` when
 *   `numPromptTokens + filledKVCacheLength > contextWindowSize` (guarded on
 *   `slidingWindowSize == -1`, which is every pinned arm). Loud.
 * - During decode, "Stop condition 4" sets `stopTriggered` and
 *   `finishReason = "length"` when `filledKVCacheLength == contextWindowSize`.
 *   That is byte-for-byte the same signal as running out of `max_tokens`.
 *
 * So a prompt that fits at prefill and then exhausts the window mid-answer is
 * indistinguishable from the outside from a model that was merely verbose.
 * Reserving the completion budget up front is what keeps this arm out of that
 * state, and it is the conservative direction: if it refuses a prompt the
 * runtime would have accepted, the cost is a loud refusal instead of a
 * truncation nobody can attribute.
 *
 * The estimate is conservative by construction -- see `CHARS_PER_TOKEN_FLOOR`,
 * which is not a measurement and says so.
 */
function assertFits(chars: number, config: Tier2Config, what: string): void {
  const estimated = Math.ceil(chars / CHARS_PER_TOKEN_FLOOR);
  const allowance = config.contextWindowSize - config.maxTokens;
  if (estimated > allowance) {
    throw new Error(
      `Approach B does not fit its context window: ${what} is ${chars} characters, at most ` +
        `${estimated} tokens, against ${allowance} available (a ${config.contextWindowSize}-token ` +
        `window less ${config.maxTokens} reserved for the answer). The policy is never chunked ` +
        `and never truncated -- an arm that silently dropped part of it would lose for a reason ` +
        `invisible in the results`,
    );
  }
}

/**
 * Turn one parsed response into findings, refusing everything that would make
 * core accept a wrong one.
 *
 * Offsets are absolute with no rebasing, unlike the judge's: B searched the
 * whole message because the whole message is what its model was shown.
 */
function collect(
  response: BaselineResponse,
  ir: PolicyIr,
  text: string,
  model: string,
  emitted: Set<string>,
  out: Finding[],
  counters: MutableCounters,
): void {
  for (const finding of response.findings) {
    const entity = ir.entityTypes.find((e) => e.id === finding.entityType);
    // A model invents labels. Passing one through makes `normalizeFindings`
    // throw, which loses the message including the findings that were fine --
    // so it is dropped here, where it costs one finding. The judge drops an
    // invented predicateId in the same place for the same reason.
    if (entity === undefined) {
      counters.unknownEntityTypes += 1;
      continue;
    }

    // The same two-span placement the judge performs, from the same module, so
    // the arms cannot drift on either the rules or the refusals. B's haystack
    // for the CLAUSE is the whole message rather than one passage -- that
    // asymmetry is intrinsic and is kept (see the module docblock) -- and the
    // mention is then searched inside the PLACED clause, which is not the same
    // size for both arms whenever their rung distributions differ: at rung 2
    // the placed clause is a peeled prefix of what the model wrote. See
    // `unresolvedMentions` for the measurement.
    const located = locateFinding(text, finding.quote, finding.mention);
    if (!located.ok) {
      if (located.refused === "evidence") counters.unresolvedQuotes += 1;
      else counters.unresolvedMentions += 1;
      continue;
    }
    const action = located.at.action;

    // `start` and `end` are integers, so the first two colons separate the key
    // unambiguously whatever an entityType contains. Keyed on the ACTION span,
    // which is what this finding carries: two clauses naming the same mention
    // are one piece of evidence about one range of the message.
    const key = `${action.start}:${action.end}:${entity.id}`;
    if (emitted.has(key)) {
      // Models restate a finding, and Plan 5's own probe corpus caught one
      // looping a single finding until its token budget ran out.
      counters.duplicatesDropped += 1;
      continue;
    }
    emitted.add(key);
    if (located.at.rung === 1) counters.rung1 += 1;
    else counters.rung2 += 1;
    // After the duplicate check, with the rungs, so all three are over the
    // findings this arm EMITTED -- see the judge, which counts it in the same
    // place for the same reason.
    if (located.at.actionIsWholeEvidence) counters.wholeClauseMentions += 1;

    out.push({
      // The ACTION span. `applyActions` rewrites exactly this range, so the
      // evidence clause must not land here: on a `redact` entityType that turns
      // the sentence around a name into a marker.
      start: action.start,
      end: action.end,
      // Sliced by the ladder from the message, never assembled from the model's
      // quote: core requires text === message.slice(start, end).
      text: action.text,
      entityType: entity.id,
      // From the IR. Core re-derives it anyway in `resolveFindings`, so a value
      // invented here would be invisible until a policy changed it.
      severity: entity.severity,
      // A model produced this, which is what tier 2 means in `Finding.tier`.
      // NOT a claim that the compiled tier-2 judge ran: which arm produced a
      // row is the record's arm field. `Finding.tier` is `0 | 1 | 2` and there
      // is no truer value available.
      tier: 2,
      // The model that ANSWERED, from the completion -- never
      // `engine.requestedModelId`, which is what was asked for.
      source: model,
      // Passed through, not clamped. `BaselineResponseSchema` is the boundary
      // that validated it, under the same field validator the judge's schema
      // uses, so every value reaching here is a finite number in [0, 1].
      confidence: finding.confidence,
    });
  }
}
