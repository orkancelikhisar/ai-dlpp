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
   * evidence of what executed.
   */
  backend: "wasm" | "webgpu";
  provider: string;
  config: TierConfig;
  items: readonly CorpusItem[];
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
  // Both hashes in ONE evaluate, not two: they are read together, and a single
  // round trip keeps this method's in-flight count at one so the sequential
  // guarantee below stays observable.
  const { irHash, policyHash } = await page.evaluate(async () => ({
    irHash: await window.__sih!.irHash(),
    policyHash: window.__sih!.policyHash(),
  }));
  // Checked once, up front, rather than left to RunRecordSchema to reject the
  // same way 1,500 times. The hash is identical on every record in the arm, so
  // a bad one invalidates the whole output -- and learning that only after a GPU
  // arm has finished costs the entire run. The regex duplicates the schema's on
  // purpose: this is the producing side, and it should fail before the work
  // rather than after it. The value is not echoed because a malformed one can
  // be arbitrary page-supplied text; its length locates the bug.
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
  for (const item of spec.items) {
    // The try lives INSIDE the loop: one item that throws must not end the arm.
    // An arm that dies on item 300 of 1500 would otherwise report as a complete
    // run of 299 items, which scores as a much better arm than it is.
    let findings: RunRecord["findings"] = [];
    // `detect` throws whole -- there is no partial result to salvage -- so a
    // thrown item has no timings at all. Read `error` before reading any timing
    // here: per DetectionResult.timings a `tier0Ms` of 0 means "tier 0 did not
    // run", and on this path it means "nothing was measured", never "instant".
    let timings: RunRecord["timings"] = { tier0Ms: 0 };
    let error: string | null = null;
    try {
      const result = await page.evaluate(
        (request) => window.__sih!.detect(request),
        { text: item.text, provider: spec.provider, config: spec.config },
      );
      findings = result.findings;
      timings = result.timings;
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
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
      // The message the offsets in `findings` and `gold` index into, and the
      // one detection actually ran on -- both read from the same `item.text`,
      // so a record cannot carry findings produced from a different string
      // than the one it ships. RunRecordSchema re-checks every span against it.
      text: item.text,
      findings,
      gold: item.gold,
      timings,
      error,
    });
  }
  return records;
}
