import type { Page } from "@playwright/test";
import type { TierConfig } from "@sih/core";
import type { CorpusItem } from "./corpus.js";
import { RECORD_SCHEMA_VERSION, type RunRecord } from "./record.js";

export interface ArmSpec {
  runId: string;
  arm: string;
  /**
   * Copied onto every record verbatim, and today verified by nothing. The chain
   * from this label to what actually executed is broken in TWO places, and
   * repairing either one alone would still leave the label unbacked:
   *
   *   1. `runArm` never plumbs this into `spec.config`. `spec.config` is
   *      forwarded to `detect` untouched, so the only backend core could ever
   *      see is whatever the caller happened to put in `config.backend`, which
   *      nothing here reconciles against this field.
   *   2. MEASURED: nothing under packages/core/src reads `TierConfig.backend`
   *      at all -- grep finds only its declaration in detect/types.ts -- so
   *      even a faithfully plumbed value would reach no reader.
   *
   * So this is a label the caller asserts. Task 11 owns making it real, and
   * does so by confirming which execution provider actually initialized rather
   * than by labelling harder. Until then do not read a record's `backend` as
   * evidence of what executed. Note that `config` IS recorded (see the record's
   * own field): unlike this one, that value is the object `detect` received.
   */
  backend: "wasm" | "webgpu";
  provider: string;
  config: TierConfig;
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
 * Rejects if `work` has not settled within `ms`.
 *
 * The limit of this mechanism, stated because it is easy to assume otherwise:
 * `page.evaluate` accepts no timeout and offers no cancellation channel, so the
 * abandoned call KEEPS RUNNING in the browser. This bounds the driver's wait,
 * not the page's work. That is still worth having -- a single wedged item would
 * otherwise hang `runArm` forever, and because records are only returned in bulk
 * at the end, every already-completed record would be unreachable -- but a run
 * that trips this repeatedly is leaking in-flight work into later items'
 * latencies, so treat it as a failure to be fixed rather than a tolerable one.
 */
async function withDeadline<T>(work: Promise<T>, ms: number, what: string): Promise<T> {
  // `Promise.race` attaches a handler to `work` immediately, so a rejection that
  // arrives after the deadline has already won is handled rather than unhandled.
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(`${what} did not finish within ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
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
    let error: string | null = null;
    try {
      const result = await withDeadline(
        page.evaluate(
          (request) => window.__sih!.detect(request),
          { text: item.text, provider: spec.provider, config: spec.config },
        ),
        spec.itemTimeoutMs,
        `detection for item "${item.id}"`,
      );
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
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
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
      backend: spec.backend,
      provider: spec.provider,
      // The exact object handed to `detect` above, so `arm` is a label with the
      // configuration that produced it sitting next to it rather than a claim on
      // its own.
      config: spec.config,
      // The message the offsets in `findings` and `gold` index into, and the
      // one detection actually ran on -- both read from the same `item.text`,
      // so a record cannot carry findings produced from a different string than
      // the one it ships. RunRecordSchema's refine re-checks every span against
      // this text wherever the file is validated; nothing on this path runs it.
      text: item.text,
      findings,
      gold: item.gold,
      timings,
      error,
    });
  }
  return records;
}
