import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import type { Page } from "@playwright/test";
import type { TierConfig } from "@sih/core";
import {
  MODEL_MANIFEST,
  resolveTier1Config,
  type Tier1Backend,
  type Tier1Config,
} from "@sih/tier1";
import { z } from "zod";
import { loadCorpus } from "./corpus.js";
import { RunRecordSchema, toJsonl } from "./record.js";
import { runArm } from "./run.js";

/**
 * Whether a rung's WebGPU output agrees with its WASM output.
 *
 * MEASURED in Task 11 -- one fresh page per rung per provider, identical feeds
 * built by the shipped encoder, raw `logits` compared element by element -- and
 * re-measured end to end on every suite run by the per-rung tests in
 * `test/tier1.spec.ts`, which read this table. It lives here, in the module that
 * has to act on it, rather than in the spec that checks it: a guard reading a
 * constant it cannot see is a convention, and this one has to be a rule.
 *
 * `agrees` is the judgement and `maxAbsLogitDiff` is its evidence; the two are
 * separate fields rather than one derived from the other because "agrees" is a
 * claim about a rung being usable, not an arithmetic property of a float.
 *
 * The disagreement is a COLLAPSE rather than drift, and nothing reports it:
 * session creation succeeds, `run` resolves, the logits are finite and correctly
 * shaped. `gliner-pii-base` agreeing under the same page code and the same
 * readback path is the control that says the harness is not the cause.
 *
 * ## What the 0 is, and what it is NOT
 *
 * `maxAbsLogitDiff` is Task 11's number, over raw logits on ONE short message,
 * and is not mine. An earlier version of this block called the whole table
 * "bit-identical over two independent full re-runs, so it is deterministic
 * rather than numerical noise". THAT IS WRONG, and I measured it wrong:
 *
 * Six consecutive `detect` calls on the SAME text, in one page, on
 * `gliner-pii-base` at threshold 0.02. On WASM all six return bit-identical
 * confidences, at 153 characters and at 2,148. On WEBGPU every pass differs
 * from the one before it at 153 characters (max |delta| 1.71e-4), and the first
 * three differ before it settles at 2,148 (max |delta| 4.89e-3). Spans, labels
 * and ordering were identical in all twelve passes.
 *
 * So webgpu output on this rung is not reproducible run to run, and the 0 below
 * was read from a single short message where the effect is smallest. If this
 * field ever becomes an assertion rather than documentation it needs a
 * tolerance, and no tolerance under ~5e-3 would hold at this input length --
 * the spread grows with sequence length and nothing here bounds it above.
 *
 * NONE OF WHICH WEAKENS THE TABLE. The three rungs marked `agrees: false`
 * disagree by 8.352, 8.134 and 30.154 -- three to four orders of magnitude
 * above that jitter -- and end to end at threshold 0.02 all three return
 * different spans and different CONFIDENCES from wasm on every run. Not
 * different LABELS, which this comment claimed until 2026-09-01: the divergence
 * cases run under the default `minimal-ir.json`, whose only tier-1 entityType
 * is `client-name`, so every finding on every rung on both providers carries
 * that one label and a label difference is not expressible there at all.
 * MEASURED -- `test/tier1.spec.ts`'s own eight arrays are labelled
 * `client-name` throughout. (The "collapse of the whole
 * logit range" this comment used to assert of all three is characterised on ONE:
 * see `test/tier1.spec.ts`, where the confidences the three actually produce are
 * measured and are not the same pattern.) And `gliner-pii-base` stays usable for
 * the reason it always did:
 * nothing that is SCORED moves. Running the whole 13-item smoke corpus through
 * both providers at 0.02 gives identical spans and labels on every item, with
 * confidences differing by ~1e-7 typically and up to 2.6e-4 on the two
 * multi-line items -- the same order as the run-to-run jitter above, which is
 * the point: on this rung the provider difference is no larger than the
 * provider's disagreement with itself.
 */
export const BACKEND_AGREEMENT: readonly {
  readonly modelId: string;
  readonly agrees: boolean;
  readonly maxAbsLogitDiff: number;
}[] = [
  { modelId: "gliner-pii-edge", agrees: false, maxAbsLogitDiff: 8.352 },
  { modelId: "gliner-pii-edge-uint8", agrees: false, maxAbsLogitDiff: 8.134 },
  { modelId: "gliner-pii-base", agrees: true, maxAbsLogitDiff: 0 },
  { modelId: "gliner-pii-base-uint8", agrees: false, maxAbsLogitDiff: 30.154 },
];

const WEBGPU_TRUSTWORTHY: readonly string[] = BACKEND_AGREEMENT.filter((r) => r.agrees).map(
  (r) => r.modelId,
);

/**
 * Refuses an arm that would run a rung on WebGPU where WebGPU is known to return
 * different numbers from WASM.
 *
 * Called TWICE per tier-1 arm on purpose: once on the config resolved here in
 * Node, so a bad arm costs milliseconds rather than a 665 MB load, and once on
 * the config the page reports the tagger was actually built with. The first is a
 * check on intent and the second is a check on fact, and only the second can
 * catch a rung substituted anywhere between this process and the graph.
 *
 * A rung not in the table at all is refused too. Six rungs are pinned in
 * MODEL_MANIFEST and only these four load, so an unmeasured rung reaching a
 * webgpu arm means the ladder changed and nobody re-ran the comparison.
 */
function assertWebgpuTrustworthy(arm: string, config: Tier1Config): void {
  if (config.backend !== "webgpu") return;
  if (WEBGPU_TRUSTWORTHY.includes(config.modelId)) return;
  const measured = BACKEND_AGREEMENT.find((r) => r.modelId === config.modelId);
  const evidence =
    measured === undefined
      ? "its webgpu output has never been compared against its wasm output"
      : `its webgpu logits differ from its wasm logits by up to ${String(measured.maxAbsLogitDiff)}`;
  throw new Error(
    `arm "${arm}" would run ${config.modelId} on webgpu, where it returns WRONG NUMBERS: ` +
      `${evidence}, silently -- session creation succeeds, the run resolves, and the logits ` +
      `are finite and correctly shaped, so nothing downstream can notice. The only rung whose ` +
      `webgpu output was measured identical to its wasm output is ` +
      `${WEBGPU_TRUSTWORTHY.join(", ")}. Run ${config.modelId} on wasm instead, or switch the ` +
      `arm to that rung. See BACKEND_AGREEMENT in apps/eval/src/driver/main.ts and the ` +
      `per-rung tests in apps/eval/test/tier1.spec.ts.`,
  );
}

export interface ArmDefinition {
  /**
   * The arm's name, and half of its file name. Free text, so it is a label and
   * never evidence -- but it must be unique across a matrix, because two arms
   * that agree on `arm` and `backend` write the same path.
   */
  readonly arm: string;
  readonly backend: Tier1Backend;
  readonly config: TierConfig;
  /**
   * Per-item deadline, forwarded to `ArmSpec.itemTimeoutMs`. Required for the
   * same reason it is required there, and more so here: a matrix mixes a tier-0
   * regex pass with a cold ONNX model, and one number chosen for the matrix
   * would be wrong for one of them.
   */
  readonly itemTimeoutMs: number;
  /**
   * What this arm varies about tier 1: any of `modelId`, `threshold`,
   * `maxWidth`, `labelForm`. `backend` comes from the field above and naming a
   * different one here is refused rather than reconciled.
   *
   * Refused outright when `config.tier1` is false, because an arm that names a
   * rung and forgets to switch tier 1 on produces a complete, valid, tier-0 file
   * under a tier-1 name -- the exact substitution this whole driver is arranged
   * to prevent.
   *
   * On what to cross into a matrix: `labelForm` belongs on ONE model and ONE
   * policy, not on the ladder. It asks whether label conditioning helps, which is
   * a different question from where a rung sits on the accuracy-versus-latency
   * curve, and crossing it multiplies every arm for an answer that does not vary
   * by rung.
   */
  readonly tier1Config?: Partial<Tier1Config>;
}

export interface MatrixOptions {
  readonly runId: string;
  readonly outDir: string;
  /** Path to a JSONL corpus; read once and shared by every arm. */
  readonly corpus: string;
  readonly arms: readonly ArmDefinition[];
  readonly provider: string;
}

/** An arm that has passed validation, with everything the run loop needs precomputed. */
interface PreparedArm {
  readonly definition: ArmDefinition;
  readonly path: string;
  /**
   * Resolved HERE, in Node, and used only to fail fast and to cross-check. What
   * gets recorded is the page's own resolution, off `loadTier1`'s report.
   */
  readonly tier1: Tier1Config | undefined;
}

/**
 * Both halves of a file name end up on disk, so neither may contain a path
 * separator, `..`, or anything a shell glob has an opinion about.
 */
const FILE_SAFE = /^[A-Za-z0-9][A-Za-z0-9._+-]*$/;

/**
 * EXPORTED because `bakeoff.ts` names files by the same two-part rule, and a
 * second copy of this regex is a second definition of "safe" free to drift from
 * this one -- which on a rule about path separators means one driver refusing a
 * name the other writes.
 */
export function assertFileSafe(what: string, value: string): void {
  if (!FILE_SAFE.test(value)) {
    throw new Error(
      `${what} "${value}" cannot be used in a file name; use letters, digits, and . _ + - only`,
    );
  }
}

function tier1ConfigFor(arm: ArmDefinition): Tier1Config | undefined {
  if (!arm.config.tier1) {
    if (arm.tier1Config !== undefined || arm.config.t1Model !== undefined) {
      throw new Error(
        `arm "${arm.arm}" names tier-1 settings but tier 1 is not enabled in its TierConfig; ` +
          `it would produce a complete tier-0 file under a tier-1 name`,
      );
    }
    return undefined;
  }
  const asked = arm.tier1Config ?? {};
  if (asked.backend !== undefined && asked.backend !== arm.backend) {
    throw new Error(
      `arm "${arm.arm}" is labelled backend ${arm.backend} while its tier1Config asks for ` +
        `backend ${asked.backend}; one arm cannot measure two runtimes`,
    );
  }
  // resolveTier1Config is the validator, not a convenience: an unknown modelId,
  // a threshold outside (0, 1] or a labelForm outside the union throws here
  // naming what was available, rather than 400 seconds later inside the page.
  const resolved = resolveTier1Config({ ...asked, backend: arm.backend });
  if (arm.config.t1Model !== undefined && arm.config.t1Model !== resolved.modelId) {
    throw new Error(
      `arm "${arm.arm}" sets TierConfig.t1Model to ${arm.config.t1Model} while its tier-1 ` +
        `settings resolve to ${resolved.modelId}`,
    );
  }
  assertWebgpuTrustworthy(arm.arm, resolved);
  return resolved;
}

/**
 * Validates EVERY arm before any of them runs.
 *
 * A ten-arm matrix is hours of GPU time, and learning on the last arm that its
 * name collides or its rung is untrustworthy on webgpu wastes all of it. Nothing
 * in here touches the page or the disk.
 */
function prepareArms(options: MatrixOptions): PreparedArm[] {
  if (options.arms.length === 0) {
    throw new Error("a matrix with no arms measures nothing; give runMatrix at least one arm");
  }
  assertFileSafe("runId", options.runId);
  const prepared: PreparedArm[] = [];
  const owners = new Map<string, number>();
  for (const [index, definition] of options.arms.entries()) {
    assertFileSafe("arm name", definition.arm);
    const tier1 = tier1ConfigFor(definition);
    const path = join(
      options.outDir,
      `${options.runId}.${definition.arm}.${definition.backend}.jsonl`,
    );
    const owner = owners.get(path);
    if (owner !== undefined) {
      throw new Error(
        `arms ${String(owner + 1)} and ${String(index + 1)} would both write ` +
          `${basename(path)}: a file is named by runId, arm and backend, so two arms agreeing ` +
          `on those three leave one silently overwritten and the matrix reporting more arms ` +
          `than it produced. Give them different arm names.`,
      );
    }
    owners.set(path, index);
    prepared.push({ definition, path, tier1 });
  }
  return prepared;
}

/**
 * Checks that the page fetched the artifact MODEL_MANIFEST pins for this rung.
 *
 * Two independent facts, because they fail differently. The URL says which file
 * was asked for -- a rung wired to the wrong `weightsPath` names the right
 * model and loads a different precision. The byte count says what the server
 * said it would serve -- a 404 body, a truncated download, or a file replaced
 * on disk since it was fetched all answer at the pinned path with the wrong
 * size. Neither catches a same-size substitution; the sha256 in the manifest
 * does, and verifying it means reading 665 MB back out of the browser on every
 * arm, which is why `scripts/fetch-models.ts` is where that check lives.
 *
 * Read from @sih/tier1 here in Node, never from the page: a page answering from
 * a stub cannot match a number it was not given.
 */
function assertPinnedArtifact(
  arm: string,
  modelId: string,
  loaded: { readonly weightsUrl: string; readonly weightsBytes: number },
): void {
  const entry = MODEL_MANIFEST[modelId];
  // Unreachable while resolveTier1Config has already rejected an unknown
  // modelId, and kept because the alternative is reading `.files` off
  // `undefined` if that ever stops being true.
  if (entry === undefined) throw new Error(`arm "${arm}" loaded unknown rung ${modelId}`);
  const pinned = entry.files[entry.weightsPath];
  if (pinned === undefined) {
    throw new Error(
      `arm "${arm}": MODEL_MANIFEST names ${entry.weightsPath} as ${modelId}'s weights but ` +
        `pins no file at that path`,
    );
  }
  if (!loaded.weightsUrl.endsWith(entry.weightsPath)) {
    throw new Error(
      `arm "${arm}" loaded ${modelId} from a URL ending "${loaded.weightsUrl.slice(-40)}", which ` +
        `is not the pinned ${entry.weightsPath}; the arm would be recorded under a rung it did ` +
        `not run`,
    );
  }
  if (loaded.weightsBytes !== pinned.bytes) {
    throw new Error(
      `arm "${arm}": ${modelId}'s pinned weights are ${String(pinned.bytes)} bytes but the page ` +
        `was served ${String(loaded.weightsBytes)}. Either the file on disk is not the pinned ` +
        `artifact, or the request answered with something that is not the graph at all. Re-fetch ` +
        `with scripts/fetch-models.ts, which verifies every file against its sha256.`,
    );
  }
}

/** Every field of a resolved config, so a drift in any one of the six is caught. */
function sameTier1Config(a: Tier1Config, b: Tier1Config): boolean {
  return (
    a.modelId === b.modelId &&
    a.backend === b.backend &&
    a.threshold === b.threshold &&
    a.maxWidth === b.maxWidth &&
    a.labelForm === b.labelForm
  );
}

function describe(config: Tier1Config): string {
  return (
    `${config.modelId}/${config.backend} threshold=${String(config.threshold)} ` +
    `maxWidth=${String(config.maxWidth)} labelForm=${config.labelForm}`
  );
}

/**
 * Run every arm over one corpus, one JSONL file per arm.
 *
 * Arms run SEQUENTIALLY and each starts from a freshly navigated page. Sequential
 * because two models loading at once contend for the same GPU and corrupt every
 * latency number in both files; freshly navigated because a tagger left over from
 * the previous arm would be measured under the next arm's label -- this function
 * owns page state, which is exactly why `runArm` refuses to navigate on its own.
 *
 * It computes NO metrics. Per spec 2.2 the file is the whole boundary and all
 * scoring is Plan 8's Python; what this function owes that side is files whose
 * every row says truthfully which configuration produced it.
 *
 * Most of the code below is therefore about one failure: a run that LOOKS
 * complete and measured nothing. An arm whose model never loaded, an arm that
 * silently ran tier 0 under a tier-1 label, an arm whose every row is an error,
 * two arms overwriting each other's file -- each ends in a directory of
 * well-formed JSONL that Plan 8 would score without complaint. Every guard here
 * exists to turn one of those into a loud failure and no file.
 */
export async function runMatrix(page: Page, options: MatrixOptions): Promise<string[]> {
  const arms = prepareArms(options);
  const items = loadCorpus(readFileSync(options.corpus, "utf8"));
  if (items.length === 0) {
    throw new Error(
      `corpus ${options.corpus} holds no items; every arm would write an empty file, which is ` +
        `indistinguishable from a run that scored nothing`,
    );
  }
  mkdirSync(options.outDir, { recursive: true });
  for (const { path } of arms) {
    // Checked before the first arm runs, not at write time, so a repeated runId
    // costs no GPU hours. `writeFileSync` still uses the exclusive flag below --
    // this is the early warning, that is the guarantee.
    if (existsSync(path)) {
      throw new Error(
        `${path} already exists: a run under this runId has already written this arm, and ` +
          `overwriting it would silently replace one measurement with another. Use a different ` +
          `runId, or move the earlier output aside.`,
      );
    }
  }

  const written: string[] = [];
  for (const { definition, path, tier1 } of arms) {
    await page.goto("/");
    await page.waitForFunction(() => window.__sih !== undefined, undefined, { timeout: 30_000 });

    let loaded: Tier1Config | undefined;
    let inferencesBefore = 0;
    if (tier1 === undefined) {
      // The fresh-page claim, asked of the page rather than assumed of `goto`.
      // A leftover tagger would not change a tier-0 arm's findings -- core only
      // reaches an engine when `config.tier1` is set -- but it would mean this
      // function does not own page state after all, and the next tier-1 arm
      // would be the one measuring the wrong model.
      const stale = await page.evaluate(() => window.__sih!.tier1Status() !== undefined);
      if (stale) {
        throw new Error(
          `arm "${definition.arm}" runs no tier-1 model but the page already holds one; ` +
            `navigating did not reset it, so no arm after this one can be trusted`,
        );
      }
    } else {
      // Load and read the inference counter in ONE round trip: the counter is
      // cumulative from the tagger's construction, so its value here is exactly
      // the warm-up's, and the difference after the corpus is what says the
      // model ran on the corpus rather than only on the warm-up.
      const report = await page.evaluate(
        async (loadOptions) => {
          const loadedReport = await window.__sih!.loadTier1(loadOptions);
          return {
            config: loadedReport.config,
            weightsUrl: loadedReport.weightsUrl,
            weightsBytes: loadedReport.weightsBytes,
            inferences: window.__sih!.tier1Status()!.totals.inferences,
          };
        },
        { ...definition.tier1Config, backend: definition.backend },
      );
      // The check `Tier1LoadReport.weightsBytes` was added FOR, finally
      // performed. The page reports the size and refuses to judge it -- "the
      // manifest lives in @sih/tier1 and the driver is where the comparison
      // belongs" -- and until now no driver made it: `runMatrix` read
      // `config` and `inferences` off this report and discarded the other two
      // fields. So an arm could serve any bytes at the pinned path, from a 404
      // body to a different precision variant, and every record would still
      // name the manifest's rung. test/tier1.spec.ts already compares this way;
      // this is the same comparison on the path that writes files.
      //
      // Before the corpus, not after: the alternative is learning that the
      // wrong graph ran once the arm has finished.
      assertPinnedArtifact(definition.arm, report.config.modelId, report);
      // The guard again, now on the object the tagger was CONSTRUCTED with
      // rather than on what this process resolved from the arm.
      assertWebgpuTrustworthy(definition.arm, report.config);
      if (!sameTier1Config(report.config, tier1)) {
        throw new Error(
          `arm "${definition.arm}" asked for ${describe(tier1)} but the page built its tagger ` +
            `with ${describe(report.config)}`,
        );
      }
      // `observedBackend` is deliberately NOT re-checked here. `loadTier1`
      // refuses to return a report whose measured provider disagrees with the
      // one asked for, so a report reaching this line has already been through
      // that check -- a guard here would be unreachable code standing in for one
      // that has run, the same reason runArm declines to re-check `policyHash`.
      loaded = report.config;
      inferencesBefore = report.inferences;
    }

    const records = await runArm(page, {
      runId: options.runId,
      arm: definition.arm,
      backend: definition.backend,
      // Every arm of a MATRIX runs core's orchestrator. This driver has no door
      // onto Approach B and is not going to grow one: a matrix varies the
      // tier-1 ladder over one policy, and B has no tier 1 at all. The
      // bake-off in driver/bakeoff.ts is where the two methods meet.
      detector: "core-orchestrator",
      provider: options.provider,
      // `t1Model` filled from the rung that actually loaded, so core's own
      // TierConfig names the model too instead of leaving the only mention of it
      // in a field core has never heard of.
      config:
        loaded === undefined
          ? definition.config
          : { ...definition.config, t1Model: loaded.modelId },
      tier1Config: loaded,
      itemTimeoutMs: definition.itemTimeoutMs,
      items,
    });

    // An arm every one of whose items failed is not a bad result, it is not a
    // result. `error` on a record exists so an arm that crashes on 5% of a
    // corpus is distinguishable from one that scores 0 on it; at 100% the file
    // is a complete, schema-valid, perfectly scoreable transcript of nothing
    // having been measured, and Plan 8 would read it as an arm with no recall.
    // runArm already refuses this shape for the one cause it can detect -- a
    // dead browser -- and this covers the causes it cannot, such as a tier
    // enabled with no engine behind it.
    const failures = records.filter((r) => r.error !== null);
    if (failures.length === records.length) {
      throw new Error(
        `arm "${definition.arm}" failed on all ${String(records.length)} of its items, so it ` +
          `measured nothing; refusing to write a file that would score as an arm with zero ` +
          `recall. The first failure was: ${String(failures[0]?.error)}`,
      );
    }

    if (loaded !== undefined) {
      // Did the graph run over THIS corpus? Not findings -- a span tagger
      // legitimately returns none -- and not `tier1Ms`, which core sets even for
      // a call with nothing taggable in it. `inferences` counts `session.run`
      // calls, so a value that has not moved since the warm-up means every item
      // went through a tier-1 arm without the model executing once.
      const after = await page.evaluate(
        () => window.__sih!.tier1Status()?.totals.inferences ?? 0,
      );
      if (after <= inferencesBefore) {
        throw new Error(
          `arm "${definition.arm}" is a tier-1 arm but the tagger ran no inference over ` +
            `${String(records.length)} items (${String(after)} total, ${String(inferencesBefore)} ` +
            `of them from the warm-up); its numbers would be tier 0's under a tier-1 name`,
        );
      }
    }

    // The README asks every writer to validate, because nothing on the producing
    // path does: runArm builds records field by field and hands them back
    // unchecked. This is the last point at which an invalid row is a thrown
    // error rather than a line in a file someone scores.
    for (const [row, record] of records.entries()) {
      const parsed = RunRecordSchema.safeParse(record);
      if (!parsed.success) {
        throw new Error(
          `arm "${definition.arm}" produced an invalid record at row ${String(row + 1)} ` +
            `(item "${record.itemId}"): ${z.prettifyError(parsed.error)}`,
        );
      }
    }

    // Named by run, arm and backend: two runs of the same arm must never
    // overwrite each other, and a file whose name does not say which arm
    // produced it is unattributable once it leaves this directory. The
    // exclusive flag is what makes the first half of that true rather than
    // merely intended -- a repeated runId is refused, not honoured.
    try {
      writeFileSync(path, toJsonl(records), { flag: "wx" });
    } catch (cause) {
      if ((cause as NodeJS.ErrnoException).code === "EEXIST") {
        throw new Error(
          `${path} was created while this matrix was running; refusing to overwrite a ` +
            `measurement`,
          { cause },
        );
      }
      throw cause;
    }
    written.push(path);
  }
  return written;
}
