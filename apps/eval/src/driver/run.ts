import type { Page } from "@playwright/test";
import { UNCERTAIN_BELOW, type TierConfig } from "@sih/core";
import type { Tier1Config } from "@sih/tier1";
import type { JudgeCallRecord } from "@sih/tier2";
import type { CorpusItem } from "./corpus.js";
import {
  RECORD_SCHEMA_VERSION,
  type RecordDetector,
  type RunRecord,
  type Tier2RunConfig,
} from "./record.js";

export interface ArmSpec {
  runId: string;
  arm: string;
  /**
   * Copied onto every record verbatim AND plumbed into the `TierConfig` that
   * `detect` receives, so the label and the run are one value in two places
   * rather than two independent claims.
   *
   * The chain from this label to what executed was broken in two places, and
   * both are now closed:
   *
   *   1. `runArm` never plumbed this into `spec.config`. `spec.config` was
   *      forwarded to `detect` untouched, so the only backend core could ever
   *      see was whatever the caller happened to put in `config.backend`, which
   *      nothing here reconciled against this field. Closed below: the config
   *      handed to `detect` carries this value, and a caller that put a
   *      different one in `config.backend` is refused rather than quietly
   *      corrected.
   *   2. Nothing under packages/core/src reads `TierConfig.backend`, so even a
   *      faithfully plumbed value reaches no reader inside core. Measured in
   *      Task 3 and re-run here: `grep -rn backend packages/core/src` returns
   *      exactly one line, the field's own declaration in detect/types.ts.
   *      Closed by Task 11 in the PAGE instead: `loadTier1` counts
   *      GPUQueue.submit calls around a warm-up inference and refuses to finish
   *      when that disagrees with the backend asked for, and
   *      `window.__sih.detect` rejects a `TierConfig.backend` contradicting the
   *      loaded model.
   *
   * So on a TIER-1 arm this field is now backed by a measurement: the record's
   * value is the config's value, the page checked that config against a count of
   * real GPU work, and the load would not have completed had the two disagreed.
   * On a TIER-0 arm it stays a bare label -- a regex pass runs on no backend at
   * all, and the page's check is skipped when `config.tier1` is false -- so read
   * it there as "which arm of the matrix this row belongs to" and nothing more.
   */
  backend: "wasm" | "webgpu";
  /**
   * Which implementation the PAGE will run for this arm, stamped onto every
   * record. See `DetectorSchema` in record.ts for what the two values mean.
   *
   * Required with no default, and the caller has to be right about it: this
   * function decides which page counters to read from it (`tier2Status` versus
   * `baselineStatus`) and the schema couples both stats fields to it, so a
   * mislabelled arm is a row with the wrong counters or no counters at all
   * rather than a row with a wrong word on it.
   *
   * `runArm` does not check the coupling and cannot: it never loads anything,
   * so it has no way to know which detector the page holds. The caller that
   * loaded it does -- see `runBakeoff` in driver/bakeoff.ts, which calls
   * `loadBaseline` and passes `approach-b` on the same branch.
   */
  detector: RecordDetector;
  provider: string;
  config: TierConfig;
  /**
   * The resolved `Tier1Config` the tier-1 tagger in the page was built with,
   * stamped onto every record. Required by RunRecordSchema exactly when
   * `config.tier1` is set, because `TierConfig` has no room for `threshold`,
   * `maxWidth` or `labelForm` and two arms differing only in those are otherwise
   * indistinguishable in the output.
   *
   * `runArm` does not check the coupling and cannot: it never loads a model, so
   * it has no way to know whether this describes the tagger the page holds. The
   * caller that loaded it does -- see runMatrix in driver/main.ts, which reads
   * this off `loadTier1`'s report and validates every record before writing.
   */
  tier1Config?: Tier1Config;
  /**
   * The resolved tier-2 settings the engine in the page was loaded with, plus
   * the per-call budget the judge over it holds. Required by RunRecordSchema
   * exactly when `config.tier2` is set, for the reason `tier1Config` is
   * required one tier down: `TierConfig` carries `t2Model` and nothing else, so
   * two arms differing only in their context window or their per-call budget
   * are otherwise byte-identical in every field a scorer can group by -- and
   * running one arm at 4,096 while the others run at 8,192 is the fallback
   * Plan 5 names for a model that cannot take the larger window.
   *
   * `runArm` does not check the coupling and cannot: it never loads a model, so
   * it has no way to know whether this describes the engine the page holds. The
   * caller that loaded it does -- see `runBakeoff` in driver/bakeoff.ts, which
   * reads this off `loadTier2`'s report and refuses an arm whose page resolved
   * different numbers than it asked for.
   */
  tier2Config?: Tier2RunConfig;
  /**
   * Per-item deadline, in milliseconds. Required rather than defaulted: the
   * right budget for a tier-0 regex pass and for a tier-1 ONNX model on a cold
   * WebGPU adapter differ by orders of magnitude, and a default here would be
   * silently wrong for exactly one of them. A caller that has not thought about
   * the number should be made to.
   *
   * Too small is not a safe direction to err in either -- it converts slow items
   * into errored records, which scores as a worse arm rather than as a
   * misconfiguration. Set it well above the arm's expected worst case; it exists
   * to catch a wedge, not to enforce a latency target.
   */
  itemTimeoutMs: number;
  items: readonly CorpusItem[];
}

/**
 * A deadline expiry, distinguishable from a detection failure.
 *
 * Its own class rather than a string match on the message, because the loop
 * below has to tell the two apart to decide whether later rows are still
 * measuring what they claim -- and a message match would be broken silently by
 * anyone rewording the error.
 */
class DeadlineExpired extends Error {}

/**
 * Rejects if `work` has not settled within `ms`.
 *
 * The limit of this mechanism, stated because it is easy to assume otherwise:
 * `page.evaluate` accepts no timeout and offers no cancellation channel, so the
 * abandoned call KEEPS RUNNING in the browser. This bounds the driver's wait,
 * not the page's work. That is still worth having -- a single wedged item would
 * otherwise hang `runArm` forever, and because records are only returned in bulk
 * at the end, every already-completed record would be unreachable -- but a run
 * that trips this repeatedly is leaking in-flight work into later items'
 * latencies. Every row measured after an expiry is therefore stamped
 * `abandonedWorkInFlight: true`; see that field in record.ts for why the arm is
 * flagged rather than aborted.
 */
async function withDeadline<T>(work: Promise<T>, ms: number, what: string): Promise<T> {
  // `Promise.race` attaches a handler to `work` immediately, so a rejection that
  // arrives after the deadline has already won is handled rather than unhandled.
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(new DeadlineExpired(`${what} did not finish within ${ms}ms`)),
          ms,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * One arm's engine-call rows, mapped so they survive the JSON round trip.
 *
 * ONE function for both arms, and that is the point rather than tidiness: the
 * bake-off's p95 TTFT gate and its decode-rate floor are computed over exactly
 * this column on the compiled arm and on Approach B, so two copies of this
 * mapping would be two definitions of "a non-finite rate" free to drift on the
 * one number a FLOOR gate reads.
 *
 * Two fields need converting rather than copying, and both were measured:
 *
 * `ttftMs` -- `JudgeCallRecord` leaves a non-finite time-to-first-token
 * deliberately unguarded, because a NaN there is a fact about the call. But
 * MEASURED, `JSON.stringify(NaN)` is "null" and zod's `z.number()` rejects NaN,
 * so copying it would write a file this module's own reader refuses. Null is
 * that same fact spelled so it survives the round trip; `undefined` stays
 * `undefined`, because "the engine reported none" and "the engine reported
 * something that is not a number" are different facts.
 *
 * `decodeTokPerSec` -- the same mapping, for the same reason, on the field where
 * the non-finite case is MORE reachable rather than less: the library computes
 * it as `completion_tokens / decode_time` with no zero guard, so a call
 * interrupted before its first token is a literal 0/0. `bakeoff.ts` computes the
 * `minDecodeTokPerSec` gate from this column and excludes the nulls, which it
 * can only do because the null arrives.
 */
function callRows(calls: readonly JudgeCallRecord[]): NonNullable<RunRecord["tier2Stats"]>["calls"] {
  return calls.map((call) => ({
    finishReason: call.finishReason,
    promptTokens: call.promptTokens,
    completionTokens: call.completionTokens,
    ttftMs: call.ttftMs === undefined || Number.isFinite(call.ttftMs) ? call.ttftMs : null,
    decodeTokPerSec:
      call.decodeTokPerSec === undefined || Number.isFinite(call.decodeTokPerSec)
        ? call.decodeTokPerSec
        : null,
  }));
}

/**
 * Did the browser survive the item that just threw?
 *
 * This is the difference between "this arm scored badly" and "this run died",
 * and nothing else in the loop can tell them apart. A closed or crashed page
 * makes `evaluate` itself throw; a page that navigated away answers, but has no
 * `__sih`; a genuine detection failure -- a missing tier-1 engine, an engine
 * that hallucinated an entityType -- leaves the page perfectly alive.
 */
async function pageIsAlive(page: Page): Promise<boolean> {
  return page.evaluate(() => window.__sih !== undefined).catch(() => false);
}

/**
 * One arm over one corpus.
 *
 * Items run SEQUENTIALLY and in corpus order. Two reasons, both load-bearing:
 * latency is a reported metric and concurrent inference on one GPU would
 * measure contention rather than the model, and stable order makes two runs
 * diffable line by line.
 */
export async function runArm(page: Page, spec: ArmSpec): Promise<RunRecord[]> {
  // An arm that names one backend beside its label and another inside its config
  // holds two intentions, and neither is the safe one to keep silently. Checked
  // before the page is touched at all, because it is an argument error rather
  // than anything about this run.
  if (spec.config.backend !== undefined && spec.config.backend !== spec.backend) {
    throw new Error(
      `arm "${spec.arm}" is labelled ${spec.backend} while its TierConfig says ` +
        `${spec.config.backend}; one arm cannot measure two runtimes`,
    );
  }
  // Refused rather than dropped. Approach B does not escalate, so a threshold on
  // its config selected nothing; silently deleting it here would let a caller
  // believe it had varied an experiment variable that this arm has no
  // equivalent of. Checked before the page is touched, like the one above,
  // because it is an argument error rather than anything about this run.
  if (spec.detector === "approach-b" && spec.config.uncertainBelow !== undefined) {
    throw new Error(
      `arm "${spec.arm}" runs Approach B and its TierConfig carries uncertainBelow ` +
        `${spec.config.uncertainBelow}; that threshold decides which SEGMENTS escalate to the ` +
        `compiled judge, and B judges the whole message in one call without escalating, so the ` +
        `row would name a knob that turned nothing`,
    );
  }
  // ONE object, built here and used for both the call and the record, so the two
  // cannot drift: whatever `detect` was given is literally what gets stamped.
  const config: TierConfig = { ...spec.config, backend: spec.backend };
  // The escalation threshold RESOLVED rather than left to a default two packages
  // away, and resolved into the object `detect` receives rather than only onto
  // the record -- so the number stamped is the number `escalate.ts` compared
  // against, not a claim about what the default is. `TierConfig` calls this an
  // EXPERIMENT variable the bake-off varies per arm; a record that cannot say
  // which value produced it cannot be compared with the row beside it.
  //
  // Only when tier 2 is on. On a tier-0 or tier-1 arm escalation never runs, and
  // stamping a threshold there would be a record naming a knob that turned
  // nothing -- the intent-as-fact defect, arriving through a field that happens
  // to be available.
  //
  // And only on the CORE path, for exactly the same reason one layer over.
  // Approach B reports `tier2: true` -- a model read the message -- but it does
  // not escalate and does not segment: it makes one call per message, always.
  // So on a B arm this threshold selects nothing, and `RunRecordSchema` refuses
  // a B row that carries one.
  if (config.tier2 && spec.detector === "core-orchestrator") {
    config.uncertainBelow = spec.config.uncertainBelow ?? UNCERTAIN_BELOW;
  }
  // Deliberately does NOT navigate. Task 12 loads a tier-1 model into the page
  // before calling this, and a goto() here would discard it and silently
  // measure a tier-0 run under a tier-1 arm label. The caller owns page state.
  const ready = await page.evaluate(() => window.__sih !== undefined);
  if (!ready) {
    throw new Error("page is not prepared: navigate to '/' and await window.__sih before calling runArm");
  }
  // Both hashes in ONE evaluate because they are read together and neither is
  // useful without the other -- one round trip for one answer, not a
  // concurrency measure. What keeps this function's in-flight evaluate count at
  // 1 is that every call here is AWAITED; batching changes the number of round
  // trips, not their overlap. MEASURED over the 13-item smoke corpus: splitting
  // this into two awaited evaluates still reads a maximum of 1 in flight (and
  // 16 round trips instead of 15), while running the item loop under
  // `Promise.all` reads 13 -- one per item. test/run.spec.ts pins both numbers.
  const { irHash, policyHash } = await page.evaluate(async () => ({
    irHash: await window.__sih!.irHash(),
    policyHash: window.__sih!.policyHash(),
  }));
  // Checked once, up front, rather than leaving whoever validates the file to
  // reject the same way 1,500 times. The hash is identical on every record in
  // the arm, so a bad one invalidates the whole output -- and learning that only
  // after a GPU arm has finished costs the entire run. The regex duplicates
  // RunRecordSchema's on purpose: nothing validates records on this path, so
  // this is the only place the check can happen BEFORE the work rather than
  // after it. The value is not echoed because a malformed one can be arbitrary
  // page-supplied text; its length locates the bug.
  //
  // `policyHash` gets no matching guard, deliberately. It is copied verbatim
  // from the loaded IR and the schema only asks that it be non-empty, which
  // core's own PolicyIrSchema already enforces at load
  // (packages/core/src/policy/schema.ts, `policyHash: z.string().min(1)`) --
  // so a guard here would be unreachable code standing in for a check that has
  // already run.
  if (!/^[0-9a-f]{64}$/.test(irHash)) {
    throw new Error(
      `the page returned an irHash that is not a sha256 digest (${irHash.length} characters); ` +
        "every record in this arm would be rejected by RunRecordSchema",
    );
  }

  const records: RunRecord[] = [];
  // Latched, never cleared. Once an evaluate has been abandoned there is no
  // signal that says it finished -- `page.evaluate` has no cancellation channel
  // and the driver has already stopped listening -- so the honest claim for
  // every later row is "work of another row's may have been in flight", not
  // "was". See `abandonedWorkInFlight` in record.ts.
  let abandonedWorkInFlight = false;
  for (const [index, item] of spec.items.entries()) {
    // The try lives INSIDE the loop: one item that throws must not end the arm.
    // An arm that dies on item 300 of 1500 would otherwise report as a complete
    // run of 299 items, which scores as a much better arm than it is.
    let findings: RunRecord["findings"] = [];
    // `detect` throws whole -- it returns a result or nothing -- so no timings
    // are RECOVERABLE for a thrown item, which is not the same as none having
    // been taken: once a tier-1 engine is wired in, orchestrator.ts assigns
    // `tier0Ms` before tier 1 can throw, so tier 0 really did run and its
    // measurement is simply unreachable from here. Either way, read `error`
    // before reading any timing on this path: a `tier0Ms` of 0 means "nothing
    // was recorded", never "instant".
    let timings: RunRecord["timings"] = { tier0Ms: 0 };
    let tier1Stats: RunRecord["tier1Stats"];
    let tier2Stats: RunRecord["tier2Stats"];
    let baselineStats: RunRecord["baselineStats"];
    // Absent on a thrown item, like the two stats fields and for a reason that
    // is the same one stated the other way round: `detect` throws whole, so
    // there is no result to read an account of degradation off. An empty array
    // would be the positive claim that nothing was skipped -- which is what
    // `DetectionResult.degraded` says `[]` means -- on an item that may never
    // have finished.
    let degraded: RunRecord["degraded"];
    let error: string | null = null;
    // Set for THIS item only; `abandonedWorkInFlight` is advanced from it after
    // the record is built, so the row that timed out is not itself flagged --
    // it already carries `error`, and what the flag marks is a row whose
    // timings were taken under someone else's work.
    let timedOut = false;
    try {
      const result = await withDeadline(
        page.evaluate(
          // The tier-1 counters are read HERE, inside the same evaluate that
          // ran the detection, for two reasons. One: a second `page.evaluate`
          // per item would double this loop's round trips over a 1,500-item
          // corpus for a value that is already sitting in the page. Two:
          // `lastDetect` is overwritten by the next `detect`, so anything that
          // read it in a later round trip would be racing the loop it belongs
          // to. Asked for only when tier 1 is enabled, so a tier-0 arm neither
          // needs `tier1Status` to exist nor carries a value it cannot justify.
          ({ request, isBaseline }) =>
            window.__sih!.detect(request).then((detection) => ({
              detection,
              tier1: request.config.tier1 ? window.__sih!.tier1Status()?.lastDetect : undefined,
              // Read in the SAME evaluate as tier 1's and for the same two
              // reasons: a second round trip per item would double this loop's
              // cost over a 1,500-item corpus, and `lastDetect` is overwritten
              // by the next `detect`, so anything reading it later would race
              // the loop it belongs to.
              //
              // Which of the two the page is asked for follows from the ARM's
              // detector, not from what happens to be loaded: `baselineStatus`
              // on a compiled arm and `tier2Status` on an Approach-B arm both
              // return `undefined` rather than throwing, so reading the wrong
              // one would produce a row missing its counters and a schema
              // rejection naming a field rather than the arm.
              tier2:
                !isBaseline && request.config.tier2
                  ? window.__sih!.tier2Status()?.lastDetect
                  : undefined,
              baseline: isBaseline ? window.__sih!.baselineStatus()?.lastDetect : undefined,
            })),
          {
            request: { text: item.text, provider: spec.provider, config },
            isBaseline: spec.detector === "approach-b",
          },
        ),
        spec.itemTimeoutMs,
        `detection for item "${item.id}"`,
      ).then((both) => {
        // Projected field by field, like `findings` below and for the same
        // reason: `Tier1DetectStats` also carries `gpuSubmits`, which is the
        // page's counter rather than the tagger's and has no field on a record.
        if (both.tier1 !== undefined) {
          tier1Stats = {
            inferences: both.tier1.inferences,
            droppedWords: both.tier1.droppedWords,
            truncatedWords: both.tier1.truncatedWords,
            overWideSpans: both.tier1.overWideSpans,
            unmappableSpans: both.tier1.unmappableSpans,
            nonFiniteScores: both.tier1.nonFiniteScores,
          };
        }
        // `JudgeStats` WHOLE, projected field by field like tier 1's. These are
        // already a DELTA -- `judgeDelta` in the page subtracts the snapshot it
        // took before the call -- which is the part that has to stay true:
        // `WebLlmJudge.stats` is CUMULATIVE across every message the judge has
        // seen, so a record populated from the totals would inflate every row
        // after the first with a fully green suite.
        if (both.tier2 !== undefined) {
          tier2Stats = {
            rung1: both.tier2.rung1,
            rung2: both.tier2.rung2,
            unresolvedQuotes: both.tier2.unresolvedQuotes,
            unresolvedMentions: both.tier2.unresolvedMentions,
            wholeClauseMentions: both.tier2.wholeClauseMentions,
            unknownPredicates: both.tier2.unknownPredicates,
            duplicatesDropped: both.tier2.duplicatesDropped,
            repairAttempts: both.tier2.repairAttempts,
            failedClosed: both.tier2.failedClosed,
            truncatedResponses: both.tier2.truncatedResponses,
            abortedResponses: both.tier2.abortedResponses,
            segmentsJudged: both.tier2.segmentsJudged,
            segmentsSkipped: both.tier2.segmentsSkipped,
            messageScopeCalls: both.tier2.messageScopeCalls,
            messageScopeJudged: both.tier2.messageScopeJudged,
            messageScopeFailedClosed: both.tier2.messageScopeFailedClosed,
            deadlineExpiries: both.tier2.deadlineExpiries,
            callerAbortsMidGeneration: both.tier2.callerAbortsMidGeneration,
            callerAbortsWhileQueued: both.tier2.callerAbortsWhileQueued,
            // ROWS, kept as rows. A message makes one engine call per selected
            // segment plus the one repair retry, so a single `finishReason`
            // for the message would be a fact about one call presented as a
            // fact about the message; see Tier2CallSchema for the whole
            // argument, including why the bake-off's p95 TTFT gate cannot be
            // computed from a per-message aggregate.
            calls: callRows(both.tier2.calls),
          };
        }
        // `BaselineStats` WHOLE, projected the same way and already a DELTA for
        // the same reason: `baselineDelta` in the page subtracts the snapshot
        // it took, because `BaselineB.stats` is CUMULATIVE across every message
        // the arm has processed. One arm on a delta and the other on a total is
        // a head-to-head that means nothing, so both are handled identically.
        //
        // Not `tier2Stats`: three of these counters are different events from
        // the judge's and one of them the judge does not have at all. See
        // BaselineStatsSchema in record.ts.
        if (both.baseline !== undefined) {
          baselineStats = {
            rung1: both.baseline.rung1,
            rung2: both.baseline.rung2,
            unresolvedQuotes: both.baseline.unresolvedQuotes,
            unresolvedMentions: both.baseline.unresolvedMentions,
            wholeClauseMentions: both.baseline.wholeClauseMentions,
            unknownEntityTypes: both.baseline.unknownEntityTypes,
            duplicatesDropped: both.baseline.duplicatesDropped,
            repairAttempts: both.baseline.repairAttempts,
            failedClosed: both.baseline.failedClosed,
            truncatedResponses: both.baseline.truncatedResponses,
            abortedResponses: both.baseline.abortedResponses,
            messagesJudged: both.baseline.messagesJudged,
            deadlineExpiries: both.baseline.deadlineExpiries,
            messageBudgetExpiries: both.baseline.messageBudgetExpiries,
            callerAbortsMidGeneration: both.baseline.callerAbortsMidGeneration,
            callerAbortsWhileQueued: both.baseline.callerAbortsWhileQueued,
            // The SAME projection the judge's rows get, from the same function.
            // B makes one call per message plus the one repair retry, so a row
            // per call is what the p95 TTFT and decode gates are computed over
            // on both arms -- and two copies of this mapping would be two
            // definitions of "a non-finite rate", free to drift on exactly the
            // column a floor gate reads.
            calls: callRows(both.baseline.calls),
          };
        }
        return both.detection;
      });
      // Projected field by field rather than assigned wholesale. `result`
      // crossed the evaluate boundary as plain JSON, and nothing validates a
      // record on this path, so a producer that decorates its findings with
      // extra keys would have them ride into the JSONL unchecked. Naming the
      // fields makes the emitted shape a property of this file rather than of
      // whatever produced them; a field added to RecordFindingSchema shows up
      // here as a type error rather than as silently absent output.
      findings = result.findings.map((f) => ({
        start: f.start,
        end: f.end,
        text: f.text,
        entityType: f.entityType,
        severity: f.severity,
        tier: f.tier,
        source: f.source,
        confidence: f.confidence,
        action: f.action,
      }));
      timings = {
        tier0Ms: result.timings.tier0Ms,
        tier1Ms: result.timings.tier1Ms,
        tier2Ms: result.timings.tier2Ms,
      };
      // The field this projection used to DROP, and the drop was invisible in
      // both directions. MEASURED against the commit before this one:
      // `grep -c degraded apps/eval/src/driver/run.ts` returns 0 and
      // `tsc --noEmit` is clean, even though `DetectionResult.degraded` is
      // REQUIRED -- a projection that names its fields cannot be told it has
      // missed one. And nothing downstream noticed either, because
      // `RunRecordSchema` had no field for it to be missing from.
      //
      // Projected entry by entry for the same reason `findings` is: the result
      // crossed the evaluate boundary as plain JSON and nothing validates a
      // record on this path.
      degraded = result.degraded.map((notice) => ({
        tier: notice.tier,
        reason: notice.reason,
        detail: notice.detail,
      }));
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
      timedOut = cause instanceof DeadlineExpired;
      // A DETECTION failure and a HARNESS failure both land here, and they must
      // not produce the same output. If the browser crashed, the context closed,
      // or the page navigated away, every remaining item throws too, each
      // becomes a schema-valid record with `error` set, and the arm returns a
      // full-length file that reads as "this arm scored badly" rather than "this
      // run died" -- confident, complete, wrong, and scoreable by Plan 8. So
      // probe once before deciding to continue, and abort the arm if the page is
      // gone. A live page means the failure was the item's, which is exactly
      // what the per-item catch exists to record.
      if (!(await pageIsAlive(page))) {
        throw new Error(
          `the harness died while running item ${index + 1} of ${spec.items.length} ` +
            `("${item.id}"): the page is closed, crashed or navigated away, so it can no ` +
            `longer answer. ${records.length} record(s) collected before this point are ` +
            `discarded rather than returned, because a short arm is obviously incomplete ` +
            `while a full one of errored records is not. The item's own failure was: ${error}`,
          { cause },
        );
      }
    }

    records.push({
      schemaVersion: RECORD_SCHEMA_VERSION,
      runId: spec.runId,
      itemId: item.id,
      policy: item.policy,
      irHash,
      policyHash,
      arm: spec.arm,
      // Which implementation produced this row, from the same value that chose
      // which page counters to read above. `arm` is free text and a label; this
      // is the fact a scorer needs in order to know which stats field to read
      // and which comparison the row belongs to.
      detector: spec.detector,
      backend: spec.backend,
      provider: spec.provider,
      // The exact object handed to `detect` above -- the caller's config with
      // `backend` reconciled into it -- so `arm` is a label with the
      // configuration that produced it sitting next to it rather than a claim on
      // its own.
      config,
      // What `TierConfig` has no room for. Absent on a tier-0 arm, which is what
      // RunRecordSchema requires; a tier-1 arm that reaches here without it
      // produces records the schema refuses, which is where runMatrix catches a
      // rung that was never recorded.
      tier1Config: spec.tier1Config,
      // What `TierConfig` has no room for one tier up: the window, the token
      // ceiling, the temperature and the per-call budget. Absent on any arm
      // that did not run tier 2, which is what RunRecordSchema requires; a
      // tier-2 arm reaching here without it produces records the schema
      // refuses, which is where `runBakeoff` catches a rung that went
      // unrecorded -- before the file is written.
      tier2Config: spec.tier2Config,
      // What the tagger did on THIS item, absent when detection threw -- see
      // record.ts, which couples the two and states why a row of zeros would be
      // a worse answer than no row at all.
      tier1Stats,
      // The same, one tier up: the judge's rung distribution, its stop
      // accounting and one row per engine call.
      tier2Stats,
      // Approach B's counterpart to that, on the arms where a model read the
      // whole message against the whole policy instead. Exactly one of the two
      // is populated on a returned row, and RunRecordSchema keys which one off
      // `detector`.
      baselineStats,
      // The message the offsets in `findings` and `gold` index into, and the
      // one detection actually ran on -- both read from the same `item.text`,
      // so a record cannot carry findings produced from a different string than
      // the one it ships. RunRecordSchema's refine re-checks every span against
      // this text wherever the file is validated; nothing on this path runs it.
      text: item.text,
      findings,
      gold: item.gold,
      timings,
      // Everything this result is short of a full three-tier run, straight from
      // the orchestrator. Read per ENTRY: `degraded.length > 0` is not a
      // cleanliness test, because an `absent` entry is filed for every tier the
      // TierConfig switched off.
      degraded,
      error,
      abandonedWorkInFlight,
    });
    // AFTER the push, so the row that expired is not flagged by its own expiry.
    if (timedOut) abandonedWorkInFlight = true;
  }
  return records;
}
