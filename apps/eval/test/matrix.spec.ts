import { expect, test, type Page } from "@playwright/test";
import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { z } from "zod";
import { MODEL_MANIFEST } from "@sih/tier1";
import { loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema, type RunRecord } from "../src/driver/record.js";
import { runMatrix, type ArmDefinition } from "../src/driver/main.js";

/**
 * The matrix driver: arms in, JSONL files out. Nothing here computes a metric --
 * that half of the spec 2.2 boundary is Plan 8's Python -- so every assertion is
 * about the CONTENTS of a record, never about an F1.
 *
 * Tests here are NOT configured serial, and several of them load a whole ONNX
 * graph. Ordering is already what it needs to be: playwright.config.ts sets
 * `workers: 1` (added for this file, because two graphs loading at once made the
 * suite flaky) and does not set `fullyParallel`. `mode: "serial"` would add only
 * one thing on top of that -- SKIPPING every later test after the first failure
 * -- which on a file of independent guards hides more than it protects.
 * MEASURED: with it, one bad assertion skipped the other 16 tests; without it,
 * all 19 ran and exactly the 2 broken ones failed.
 */

const REPO_ROOT = join(import.meta.dirname, "..", "..", "..");
const CORPUS = join(REPO_ROOT, "corpora", "fixtures", "smoke.jsonl");
const MODELS_DIR = join(REPO_ROOT, "packages", "tier1", "models");
const ITEMS = loadCorpus(readFileSync(CORPUS, "utf8"));

/** Cold load of an ONNX graph plus a corpus pass, twice over in some tests. */
const TIER1_TIMEOUT_MS = 300_000;

/**
 * The cheap rung, used wherever an arm only has to be a REAL tier-1 arm.
 *
 * 46 MB against `gliner-pii-base`'s 665 MB, and every test here runs it on wasm,
 * which is the provider its numbers are trustworthy on -- the WebGPU defect this
 * file guards against does not touch the wasm column.
 */
const CHEAP_RUNG = "gliner-pii-edge-uint8";

/** The one rung measured to agree between wasm and webgpu. See BACKEND_AGREEMENT. */
const WEBGPU_SAFE_RUNG = "gliner-pii-base";

const FETCH_MODELS_HINT =
  "tier-1 weights are not on disk; run scripts/fetch-models.ts " +
  "(pnpm -C packages/tier1 exec vite-node ../../scripts/fetch-models.ts)";

/** Every pinned file of a rung, not just the graph: the tokenizer is loaded too. */
function weightsOnDisk(modelId: string): boolean {
  const entry = MODEL_MANIFEST[modelId];
  if (entry === undefined) return false;
  return Object.keys(entry.files).every((file) => existsSync(join(MODELS_DIR, modelId, file)));
}

function freshOutDir(): string {
  return mkdtempSync(join(tmpdir(), "sih-eval-"));
}

/** A tier-0 arm; the deadline is far above a regex pass so it catches a wedge, not slowness. */
function t0Arm(overrides: Partial<ArmDefinition> = {}): ArmDefinition {
  return {
    arm: "t0",
    backend: "wasm",
    config: { tier0: true, tier1: false, tier2: false },
    itemTimeoutMs: 10_000,
    ...overrides,
  };
}

/**
 * Reads a written file back the way Plan 8 will: line by line, each validated.
 * Throws rather than expect()s so a bad line names itself instead of arriving as
 * "expected false to be true" somewhere up the stack.
 */
function readRecords(path: string): RunRecord[] {
  const text = readFileSync(path, "utf8");
  // Terminating newline: Python iterating the file handle gets no ragged last
  // line, and appending is safe.
  expect(text.endsWith("\n")).toBe(true);
  return text
    .trim()
    .split("\n")
    .map((line, i) => {
      const parsed = RunRecordSchema.safeParse(JSON.parse(line));
      if (!parsed.success) {
        throw new Error(`line ${String(i + 1)} of ${path} is not a valid record: ${z.prettifyError(parsed.error)}`);
      }
      return parsed.data;
    });
}

const NAVIGATION_SENTINEL = "SENTINEL: runMatrix navigated";

/**
 * A page that refuses to be navigated.
 *
 * Every guard below has to fire BEFORE the browser is touched, because the whole
 * value of validating an arm is learning about it in milliseconds rather than
 * after a 665 MB load and a corpus pass. A rejection alone cannot show that --
 * the run could have failed anywhere -- so these tests assert on WHICH error
 * came back, and this proxy makes "it navigated first" a distinguishable one.
 */
function pageThatRefusesToNavigate(page: Page): Page {
  return new Proxy(page, {
    get(target, prop) {
      if (prop === "goto") return () => Promise.reject(new Error(NAVIGATION_SENTINEL));
      // `target` as the receiver so a getter on Page resolves its own property
      // reads against the real object rather than recursing through this trap.
      const value = Reflect.get(target, prop, target) as unknown;
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

/**
 * A page that closes itself just before its `nth` evaluate, which is what a
 * crashed browser looks like from the driver.
 *
 * Deterministic where waiting for a real crash is not: runArm's evaluates are
 * countable -- one readiness probe, one that reads both hashes, then one per
 * item -- so `nth` selects the item to die on.
 */
function pageThatDiesOnEvaluate(page: Page, nth: number): Page {
  let calls = 0;
  return new Proxy(page, {
    get(target, prop) {
      const value = Reflect.get(target, prop, target) as unknown;
      if (typeof value !== "function") return value;
      if (prop !== "evaluate") return value.bind(target);
      return async (...args: unknown[]) => {
        calls += 1;
        if (calls === nth) await target.close();
        return (value as (...a: unknown[]) => Promise<unknown>).apply(target, args);
      };
    },
  });
}

// -- the file the plan asked for -------------------------------------------

test("writes one JSONL file per arm, each line a valid record", async ({ page }) => {
  const out = freshOutDir();
  const written = await runMatrix(page, {
    runId: "matrix-test",
    outDir: out,
    corpus: CORPUS,
    arms: [t0Arm()],
    provider: "claude",
  });

  expect(written).toHaveLength(1);
  // The returned list and the directory agree: an arm that wrote a file nobody
  // was told about, or a path that names no file, both read as success here
  // without this.
  expect(readdirSync(out)).toEqual([basename(written[0]!)]);

  const lines = readFileSync(written[0]!, "utf8").trim().split("\n");
  expect(lines.length).toBeGreaterThanOrEqual(12);
  for (const line of lines) {
    expect(RunRecordSchema.safeParse(JSON.parse(line)).success).toBe(true);
  }

  // Everything above is satisfied by a file of 13 errored records, which is the
  // exact shape of "this run looked complete and measured nothing". The rest of
  // this test refuses that reading.
  const records = readRecords(written[0]!);
  expect(records).toHaveLength(ITEMS.length);
  expect(records.map((r) => r.itemId)).toEqual(ITEMS.map((i) => i.id));
  expect(records.map((r) => r.error)).toEqual(ITEMS.map(() => null));
  expect(records.reduce((n, r) => n + r.findings.length, 0)).toBeGreaterThan(0);
  expect(records.every((r) => r.arm === "t0" && r.backend === "wasm")).toBe(true);
  expect(records.every((r) => r.runId === "matrix-test")).toBe(true);
  // The arm's backend reached the config `detect` actually received, rather than
  // living only on the label beside it. Without this, `backend` is a free string
  // nothing reconciles -- the disease ArmSpec.backend's own comment names.
  expect(records.every((r) => r.config.backend === "wasm")).toBe(true);
  // A tier-0 arm ran no tier-1 model and must not claim one.
  expect(records.every((r) => r.tier1Config === undefined)).toBe(true);
});

test("names each output file by run, arm and backend so two runs never collide", async ({ page }) => {
  const out = freshOutDir();
  const written = await runMatrix(page, {
    runId: "abc",
    outDir: out,
    corpus: CORPUS,
    arms: [t0Arm()],
    provider: "claude",
  });
  expect(written[0]).toMatch(/abc.*t0.*wasm\.jsonl$/);
  // Pinned exactly as well: the regex above passes on a name that also carries a
  // timestamp or a pid, and Plan 8 joins these files by name.
  expect(basename(written[0]!)).toBe("abc.t0.wasm.jsonl");
  expect(dirname(written[0]!)).toBe(out);
});

// -- arms that would produce an unattributable or missing file ---------------

test("refuses two arms whose files would collide, before running either", async ({ page }) => {
  // The silent version of this is the worst kind of complete-looking run: two
  // arms differing only in threshold, both named "t0+t1" on wasm, produce ONE
  // file -- the second overwriting the first -- and a matrix reported as four
  // arms leaves three files with nothing to say which arm is missing.
  const out = freshOutDir();
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "dup",
      outDir: out,
      corpus: CORPUS,
      arms: [
        t0Arm({ arm: "t0+t1", config: { tier0: true, tier1: false, tier2: false } }),
        t0Arm({ arm: "t0+t1", config: { tier0: false, tier1: false, tier2: false } }),
      ],
      provider: "claude",
    }),
  // Names the file, not just the fact: with a dozen arms the operator needs to
  // know which two.
  ).rejects.toThrow(/both write dup\.t0\+t1\.wasm\.jsonl/);
  expect(readdirSync(out)).toEqual([]);
});

test("refuses a runId or arm name that would not survive being a file name", async ({ page }) => {
  // Both halves of the name reach the filesystem verbatim. A separator in either
  // writes outside outDir -- silently, since mkdir already ran -- and a glob
  // character makes the file unfindable by the shell that collects the run.
  const out = freshOutDir();
  for (const bad of ["../escape", "a/b", "run id", "*"]) {
    await expect(
      runMatrix(pageThatRefusesToNavigate(page), {
        runId: bad,
        outDir: out,
        corpus: CORPUS,
        arms: [t0Arm()],
        provider: "claude",
      }),
    ).rejects.toThrow(/cannot be used in a file name/);
    await expect(
      runMatrix(pageThatRefusesToNavigate(page), {
        runId: "ok",
        outDir: out,
        corpus: CORPUS,
        arms: [t0Arm({ arm: bad })],
        provider: "claude",
      }),
    ).rejects.toThrow(/cannot be used in a file name/);
  }
  expect(readdirSync(out)).toEqual([]);
});

test("refuses to overwrite an arm file an earlier run already wrote", async ({ page }) => {
  // Re-running a matrix under a runId that already has output silently replaces
  // a measurement. The file name promises two runs never collide; that is only
  // true if a repeated runId is refused rather than honoured.
  const out = freshOutDir();
  const [first] = await runMatrix(page, {
    runId: "same-id",
    outDir: out,
    corpus: CORPUS,
    arms: [t0Arm()],
    provider: "claude",
  });
  const before = readFileSync(first!, "utf8");

  await expect(
    runMatrix(page, {
      runId: "same-id",
      outDir: out,
      corpus: CORPUS,
      arms: [t0Arm()],
      provider: "claude",
    }),
  ).rejects.toThrow(/already exists/i);
  expect(readFileSync(first!, "utf8")).toBe(before);
});

test("refuses an empty corpus rather than writing an empty file", async ({ page }) => {
  const out = freshOutDir();
  const empty = join(out, "empty.jsonl");
  writeFileSync(empty, "\n\n");
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "e",
      outDir: out,
      corpus: empty,
      arms: [t0Arm()],
      provider: "claude",
    }),
  ).rejects.toThrow(/no items/i);
});

test("refuses a matrix with no arms rather than reporting a successful run of nothing", async ({
  page,
}) => {
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "e",
      outDir: freshOutDir(),
      corpus: CORPUS,
      arms: [],
      provider: "claude",
    }),
  ).rejects.toThrow(/no arms/i);
});

// -- the WebGPU guard --------------------------------------------------------

/**
 * WebGPU returns WRONG LOGITS on three of the four rungs that load at all.
 * Measured in Task 11 and re-measured on every suite run by the per-rung tests
 * in test/tier1.spec.ts, which are the source of the table this guard reads.
 *
 * The failure is silent -- session creation succeeds, `run` resolves, the logits
 * are finite and correctly shaped -- so nothing downstream can notice. An arm is
 * the last place it can be refused.
 */
for (const modelId of ["gliner-pii-edge", "gliner-pii-edge-uint8", "gliner-pii-base-uint8"]) {
  test(`refuses a webgpu arm on ${modelId}, whose webgpu output is known wrong`, async ({ page }) => {
    const out = freshOutDir();
    const run = runMatrix(pageThatRefusesToNavigate(page), {
      runId: "gpu",
      outDir: out,
      corpus: CORPUS,
      arms: [
        {
          arm: "t0+t1",
          backend: "webgpu",
          config: { tier0: true, tier1: true, tier2: false },
          itemTimeoutMs: 60_000,
          tier1Config: { modelId },
        },
      ],
      provider: "claude",
    });
    await expect(run).rejects.toThrow(new RegExp(modelId));
    // Names the rung that IS safe, so the message is actionable rather than a
    // refusal, and says the numbers are wrong rather than merely unsupported.
    await expect(run).rejects.toThrow(new RegExp(WEBGPU_SAFE_RUNG));
    // Before the browser was touched: no navigation, no 665 MB load.
    await expect(run).rejects.not.toThrow(new RegExp(NAVIGATION_SENTINEL));
    expect(readdirSync(out)).toEqual([]);
  });
}

test("refuses a webgpu arm that names no rung at all, because the default is a wrong one", async ({
  page,
}) => {
  // `resolveTier1Config` fills `modelId` in with `gliner-pii-edge`, which is one
  // of the three. A guard that only inspected what the caller WROTE would wave
  // this through and measure garbage under the default rung's name.
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "gpu",
      outDir: freshOutDir(),
      corpus: CORPUS,
      arms: [
        {
          arm: "t0+t1",
          backend: "webgpu",
          config: { tier0: true, tier1: true, tier2: false },
          itemTimeoutMs: 60_000,
        },
      ],
      provider: "claude",
    }),
  ).rejects.toThrow(/gliner-pii-edge/);
});

test(`lets a webgpu arm through on ${WEBGPU_SAFE_RUNG}`, async ({ page }) => {
  // The control. A guard that refused every webgpu arm would pass every test
  // above while making the whole webgpu half of the ladder unmeasurable, and
  // nothing above can tell the two apart. Reaching the navigation sentinel is
  // exactly "the guard let this arm through".
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "gpu",
      outDir: freshOutDir(),
      corpus: CORPUS,
      arms: [
        {
          arm: "t0+t1",
          backend: "webgpu",
          config: { tier0: true, tier1: true, tier2: false },
          itemTimeoutMs: 60_000,
          tier1Config: { modelId: WEBGPU_SAFE_RUNG },
        },
      ],
      provider: "claude",
    }),
  ).rejects.toThrow(new RegExp(NAVIGATION_SENTINEL));
});

test("validates every arm before running the first one", async ({ page }) => {
  // A ten-arm matrix whose last arm is malformed must not spend an hour finding
  // out. Nothing is written, including for the arm that was fine.
  const out = freshOutDir();
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "late",
      outDir: out,
      corpus: CORPUS,
      arms: [
        t0Arm(),
        {
          arm: "t0+t1",
          backend: "webgpu",
          config: { tier0: true, tier1: true, tier2: false },
          itemTimeoutMs: 60_000,
          tier1Config: { modelId: CHEAP_RUNG },
        },
      ],
      provider: "claude",
    }),
  ).rejects.toThrow(new RegExp(CHEAP_RUNG));
  expect(readdirSync(out)).toEqual([]);
});

test("refuses tier-1 settings on an arm that never enables tier 1", async ({ page }) => {
  // The exact shape of "ran tier 0 under a tier-1 label": a rung, a threshold
  // and a label form are named, `config.tier1` is left false, and the resulting
  // file is a perfectly valid tier-0 run that an operator reads as tier 1.
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "mislabel",
      outDir: freshOutDir(),
      corpus: CORPUS,
      arms: [t0Arm({ tier1Config: { modelId: CHEAP_RUNG, threshold: 0.3 } })],
      provider: "claude",
    }),
  ).rejects.toThrow(/tier1Config.*tier 1 is not enabled|tier 1 is not enabled/i);
});

test("refuses a tier-1 backend that contradicts the arm's own backend", async ({ page }) => {
  await expect(
    runMatrix(pageThatRefusesToNavigate(page), {
      runId: "contradiction",
      outDir: freshOutDir(),
      corpus: CORPUS,
      arms: [
        {
          arm: "t0+t1",
          backend: "wasm",
          config: { tier0: true, tier1: true, tier2: false },
          itemTimeoutMs: 60_000,
          tier1Config: { modelId: CHEAP_RUNG, backend: "webgpu" },
        },
      ],
      provider: "claude",
    }),
  ).rejects.toThrow(/labelled backend wasm .* asks for backend webgpu/);
});

// -- arms that ran but measured nothing --------------------------------------

test("aborts an arm in which every single item failed", async ({ page }) => {
  // `error` on a record is deliberate: an arm that crashes on 5% of a corpus and
  // one that scores 0 on it are different results. At 100% they are not a
  // result at all -- the file is a complete, schema-valid, scoreable transcript
  // of nothing having been measured. Induced with a tier that has no engine,
  // which is the realistic version: core throws per item and the page survives,
  // so the per-item catch files every row exactly as designed.
  const out = freshOutDir();
  await expect(
    runMatrix(page, {
      runId: "allbad",
      outDir: out,
      corpus: CORPUS,
      arms: [t0Arm({ arm: "t0+t2", config: { tier0: true, tier1: false, tier2: true } })],
      provider: "claude",
    }),
  ).rejects.toThrow(/every|all 13/i);
  // And the file it would have written does not exist: a rejection that still
  // left the transcript on disk would be scored by whatever globs the directory.
  expect(readdirSync(out)).toEqual([]);
});

test("aborts a tier-1 arm whose model never ran an inference", async ({ page }) => {
  test.skip(!weightsOnDisk(CHEAP_RUNG), FETCH_MODELS_HINT);
  test.setTimeout(TIER1_TIMEOUT_MS);

  // The quietest way for a tier-1 arm to be tier 0: the model loads, every item
  // succeeds, `tier1Ms` is a real number on every record -- core sets it around
  // the tagger call whether or not there was anything to tag -- and `findings`
  // is empty, which a span tagger is entitled to return. Nothing in the output
  // distinguishes that from a model that ran and found nothing.
  //
  // Induced with a corpus of fenced code, which the orchestrator filters out
  // before the tagger sees it, so the tagger is called with no segments and
  // never reaches the graph. `inferences` counts session.run calls and is the
  // only signal that separates the two.
  const out = freshOutDir();
  const codeOnly = join(out, "code-only.jsonl");
  writeFileSync(
    codeOnly,
    ITEMS.slice(0, 2)
      .map((item, i) =>
        JSON.stringify({
          id: `code-${String(i)}`,
          text: "```py\nwhile True:\n    total = total + 1\n```",
          policy: item.policy,
          gold: [],
        }),
      )
      .join("\n") + "\n",
  );

  await expect(
    runMatrix(page, {
      runId: "noinfer",
      outDir: out,
      corpus: codeOnly,
      arms: [
        {
          arm: "t1",
          backend: "wasm",
          config: { tier0: false, tier1: true, tier2: false },
          itemTimeoutMs: 120_000,
          tier1Config: { modelId: CHEAP_RUNG, threshold: 0.02 },
        },
      ],
      provider: "claude",
    }),
  ).rejects.toThrow(/ran no inference/);
  expect(readdirSync(out)).toEqual(["code-only.jsonl"]);
});

test("aborts the whole matrix when the browser dies mid-arm", async ({ page }) => {
  // Task 3's property, re-checked through runMatrix: a dead page makes every
  // remaining item throw, and without the liveness probe the arm returns a
  // full-length file that reads as "this arm scored badly" rather than "this run
  // died".
  //
  // Three evaluates precede the first item, so closing before the sixth kills
  // item 3: runMatrix's own check that no tier-1 model is left over, then
  // runArm's readiness probe, then the one that reads both hashes. Pinned rather
  // than approximated, the way run.spec.ts pins runArm's round-trip count -- an
  // extra per-item evaluate is latency charged to every measurement.
  const out = freshOutDir();
  await expect(
    runMatrix(pageThatDiesOnEvaluate(page, 6), {
      runId: "dead",
      outDir: out,
      corpus: CORPUS,
      arms: [t0Arm()],
      provider: "claude",
    }),
  ).rejects.toThrow(/harness died while running item 3 of 13/);
  expect(readdirSync(out)).toEqual([]);
});

// -- tier-1 arms, with the model actually running -----------------------------

test("records the tier-1 config the tagger was built with, per arm", async ({ page }) => {
  test.skip(!weightsOnDisk(CHEAP_RUNG), FETCH_MODELS_HINT);
  test.setTimeout(TIER1_TIMEOUT_MS);

  // Two arms whose ONLY difference is `threshold`. Before this task their
  // records were byte-identical in every field Plan 8 could group by, so the two
  // were unscoreable apart. `arm` differs too because the file name must,
  // which is precisely why the name cannot be the evidence.
  const out = freshOutDir();
  const arm = (name: string, threshold: number): ArmDefinition => ({
    arm: name,
    backend: "wasm",
    config: { tier0: false, tier1: true, tier2: false },
    itemTimeoutMs: 120_000,
    tier1Config: { modelId: CHEAP_RUNG, threshold },
  });
  const written = await runMatrix(page, {
    runId: "t1",
    outDir: out,
    corpus: CORPUS,
    arms: [arm("t1-lo", 0.02), arm("t1-hi", 0.95)],
    provider: "claude",
  });

  expect(written).toHaveLength(2);
  const lo = readRecords(written[0]!);
  const hi = readRecords(written[1]!);
  expect(lo).toHaveLength(ITEMS.length);
  expect(hi).toHaveLength(ITEMS.length);
  expect(lo.every((r) => r.error === null)).toBe(true);
  expect(hi.every((r) => r.error === null)).toBe(true);

  // FULLY resolved, not the partial object the caller handed in: `maxWidth` and
  // `labelForm` were never named by this test, and a record carrying only what
  // was asked for is an arm nobody can reproduce.
  for (const record of [...lo, ...hi]) {
    expect(record.tier1Config).toBeDefined();
    expect(Object.keys(record.tier1Config!).sort()).toEqual([
      "backend",
      "labelForm",
      "maxWidth",
      "modelId",
      "threshold",
    ]);
    expect(record.tier1Config!.modelId).toBe(CHEAP_RUNG);
    expect(record.tier1Config!.backend).toBe("wasm");
    expect(record.tier1Config!.maxWidth).toBe(12);
    expect(record.tier1Config!.labelForm).toBe("id");
    // The model that ran is named in the TierConfig too, so `config` alone is
    // no longer identical between two arms on different rungs.
    expect(record.config.t1Model).toBe(CHEAP_RUNG);
    expect(record.timings.tier1Ms).toBeDefined();
  }
  expect(lo.every((r) => r.tier1Config!.threshold === 0.02)).toBe(true);
  expect(hi.every((r) => r.tier1Config!.threshold === 0.95)).toBe(true);

  // And the recorded threshold is the one the graph ran under, not a label:
  // 0.02 admits spans this rung scores in the 0.1-0.2 band on this corpus while
  // 0.95 admits none of them. A stamped-but-unused threshold would leave these
  // two counts equal.
  const spans = (records: RunRecord[]): number => records.reduce((n, r) => n + r.findings.length, 0);
  expect(spans(lo)).toBeGreaterThan(0);
  expect(spans(lo)).toBeGreaterThan(spans(hi));
  // Tier 1 is what produced them -- tier 0 is off in these arms, so a tier-0
  // finding here would mean the config never reached core.
  expect(lo.every((r) => r.findings.every((f) => f.tier === 1))).toBe(true);
});

test("starts each arm from a page with no model left over from the last one", async ({ page }) => {
  test.skip(!weightsOnDisk(CHEAP_RUNG), FETCH_MODELS_HINT);
  test.setTimeout(TIER1_TIMEOUT_MS);

  // runMatrix owns page state, which is why runArm refuses to navigate. A tagger
  // surviving into the next arm would be measured under that arm's label.
  const out = freshOutDir();
  const written = await runMatrix(page, {
    runId: "fresh",
    outDir: out,
    corpus: CORPUS,
    arms: [
      {
        arm: "t1",
        backend: "wasm",
        config: { tier0: false, tier1: true, tier2: false },
        itemTimeoutMs: 120_000,
        tier1Config: { modelId: CHEAP_RUNG, threshold: 0.02 },
      },
      t0Arm(),
    ],
    provider: "claude",
  });

  const withModel = readRecords(written[0]!);
  const without = readRecords(written[1]!);
  expect(withModel.some((r) => r.findings.length > 0)).toBe(true);
  expect(without.every((r) => r.tier1Config === undefined)).toBe(true);
  expect(without.every((r) => r.findings.every((f) => f.tier === 0))).toBe(true);
  expect(without.every((r) => r.timings.tier1Ms === undefined)).toBe(true);
  // Asked of the page rather than inferred from the records: after the tier-0
  // arm the harness holds no tier-1 engine at all.
  expect(await page.evaluate(() => window.__sih!.tier1Status())).toBeUndefined();
});

test(`runs a real webgpu arm on ${WEBGPU_SAFE_RUNG} end to end`, async ({ page }) => {
  test.skip(!weightsOnDisk(WEBGPU_SAFE_RUNG), FETCH_MODELS_HINT);
  test.setTimeout(TIER1_TIMEOUT_MS);
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined, undefined, { timeout: 30_000 });
  test.skip(
    !(await page.evaluate(() => window.__sih!.backendAvailable("webgpu"))),
    "webgpu not available in this browser",
  );

  // The end-to-end control for the guard: the one rung whose webgpu output was
  // measured identical to its wasm output, run for real through the matrix.
  const out = freshOutDir();
  const [written] = await runMatrix(page, {
    runId: "gpu-real",
    outDir: out,
    corpus: CORPUS,
    arms: [
      {
        arm: "t0+t1",
        backend: "webgpu",
        config: { tier0: true, tier1: true, tier2: false },
        itemTimeoutMs: 120_000,
        tier1Config: { modelId: WEBGPU_SAFE_RUNG, threshold: 0.02 },
      },
    ],
    provider: "claude",
  });

  const records = readRecords(written!);
  expect(records).toHaveLength(ITEMS.length);
  expect(records.every((r) => r.error === null)).toBe(true);
  expect(records.every((r) => r.backend === "webgpu")).toBe(true);
  expect(records.every((r) => r.config.backend === "webgpu")).toBe(true);
  expect(records.every((r) => r.tier1Config!.backend === "webgpu")).toBe(true);
  // The model ran on the GPU rather than the arm merely being labelled so: the
  // page refuses to finish loadTier1 when the GPUQueue.submit count around its
  // warm-up disagrees with the backend asked for, and `detect` refuses a
  // TierConfig.backend that contradicts the loaded model -- so a webgpu arm
  // served by wasm cannot reach this line with every record error-free.
  expect(records.some((r) => r.findings.some((f) => f.tier === 1))).toBe(true);
});

test("refuses an arm served bytes that are not the pinned artifact", async ({ page }) => {
  // `Tier1LoadReport.weightsBytes` exists so a caller holding MODEL_MANIFEST
  // can tell the pinned graph from a 404 body or a different precision
  // variant, and the page deliberately reports it without judging it. Nothing
  // on this path made the comparison: runMatrix read `config` and `inferences`
  // off the load report and discarded `weightsUrl`/`weightsBytes`, so an arm
  // could serve anything at the pinned path and every record would still name
  // the manifest rung.
  //
  // Driven by lying about the SIZE rather than by swapping a file: the point of
  // the check is that the driver compares the page's answer against the
  // manifest, so a page whose answer is wrong must be refused whatever made it
  // wrong. A proxy on `evaluate` is the only way to make the real page give a
  // wrong answer on demand.
  test.skip(!weightsOnDisk(CHEAP_RUNG), FETCH_MODELS_HINT);
  const out = freshOutDir();
  let calls = 0;
  const lying = new Proxy(page, {
    get(target, prop) {
      const value = Reflect.get(target, prop, target) as unknown;
      if (typeof value !== "function") return value;
      if (prop !== "evaluate") return value.bind(target);
      return async (...args: unknown[]) => {
        const result = await (value as (...a: unknown[]) => Promise<unknown>).apply(target, args);
        calls += 1;
        // The load report is the only evaluate that answers with these fields.
        if (result !== null && typeof result === "object" && "weightsBytes" in result) {
          return { ...result, weightsBytes: 1234 };
        }
        return result;
      };
    },
  }) as Page;

  await expect(
    runMatrix(lying, {
      runId: "pinned",
      outDir: out,
      corpus: CORPUS,
      arms: [
        {
          arm: "t1",
          backend: "wasm",
          config: { tier0: false, tier1: true, tier2: false },
          tier1Config: { modelId: CHEAP_RUNG },
          itemTimeoutMs: 120_000,
        },
      ],
      provider: "claude",
    }),
  ).rejects.toThrow(/pinned weights are \d+ bytes but the page was served 1234/);

  // Refused BEFORE the corpus ran, which is the difference between losing a
  // second and losing an arm: the load is one evaluate, so a run that reached
  // the items would have made many more than this.
  expect(calls).toBeLessThanOrEqual(3);
  // And no file, so nothing downstream can score an arm that was refused.
  expect(readdirSync(out)).toEqual([]);
});

test("lets an arm through when the bytes served are the pinned ones", async ({ page }) => {
  // The control. Without it the test above passes against a driver that
  // refuses every arm, which is the classic way a guard test proves nothing.
  test.skip(!weightsOnDisk(CHEAP_RUNG), FETCH_MODELS_HINT);
  const out = freshOutDir();
  const written = await runMatrix(page, {
    runId: "pinned-ok",
    outDir: out,
    corpus: CORPUS,
    arms: [
      {
        arm: "t1",
        backend: "wasm",
        config: { tier0: false, tier1: true, tier2: false },
        tier1Config: { modelId: CHEAP_RUNG },
        itemTimeoutMs: 120_000,
      },
    ],
    provider: "claude",
  });
  const records = readRecords(written[0]!);
  expect(records).toHaveLength(ITEMS.length);
  // The manifest's own number, read here rather than restated, so a re-pin
  // moves both sides together.
  const entry = MODEL_MANIFEST[CHEAP_RUNG]!;
  expect(entry.files[entry.weightsPath]!.bytes).toBeGreaterThan(0);
  // Every row carries the tier-1 counters, which is the other half of what a
  // written file now owes Plan 8.
  expect(records.every((r) => r.tier1Stats !== undefined)).toBe(true);
  expect(records.every((r) => r.tier1Stats!.inferences > 0)).toBe(true);
  expect(records.every((r) => r.abandonedWorkInFlight === false)).toBe(true);
});
