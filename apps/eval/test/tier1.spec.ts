import { expect, test, type Page } from "@playwright/test";
import { MODEL_MANIFEST, TIER1_BACKENDS, type Tier1Backend } from "@sih/tier1";
import type { ResolvedFinding } from "@sih/core";

/**
 * The tier-1 model, executing in real Chrome. Everything before this task ran
 * against fakes, or against onnxruntime-NODE, so this is the first place the
 * shipped runtime path is exercised at all.
 *
 * Serial: each arm loads a whole ONNX graph into one browser, and two competing
 * for the same GPU would measure contention rather than the model.
 */
test.describe.configure({ mode: "serial" });

/**
 * The rung the smoke arms use, and NOT the cheapest one.
 *
 * `gliner-pii-base` fp32 is 665 MB where `gliner-pii-edge-uint8` is 46 MB, and
 * size was the obvious selection criterion. It is the wrong one. MEASURED here
 * (see `BACKEND_AGREEMENT` below): this is the ONLY rung of the four that load
 * at all whose WebGPU logits match its WASM logits -- to 0.000, bit for bit --
 * so it is the only rung on which a wasm arm and a webgpu arm are measuring the
 * same model. On the other three, a webgpu arm silently returns different
 * numbers, which is a far worse source of flakiness than a large file.
 *
 * The size cost is small and measured: 2.0-2.7 s to load in this browser from
 * local disk, against 0.5-0.6 s for edge-uint8.
 */
const SMOKE_MODEL = "gliner-pii-base";

/** Cold load of a 665 MB graph, a warm-up inference, then detection. */
const ARM_TIMEOUT_MS = 300_000;

/**
 * Task 10's end-to-end message, reused verbatim so a browser/Node difference is
 * attributable. Three things at once: a fenced code block (which the
 * orchestrator must keep away from the tagger), an astral character before every
 * span (so a UTF-16 offset bug shows up as a failure rather than as a plausible
 * neighbour), and prose holding a person, an email and an organisation.
 */
const MESSAGE =
  "```\nconst apiKey = \"redacted\";\n```\n" +
  "\u{1F642} Contact Priya Sharma at priya@acme.io about the Northwind Traders renewal.";

/** Navigate and wait for the page to publish its API, reporting a pageerror by message. */
async function openHarness(page: Page): Promise<void> {
  const pageErrors: Error[] = [];
  page.on("pageerror", (error) => pageErrors.push(error));
  await page.goto("/");
  try {
    await page.waitForFunction(() => window.__sih !== undefined, undefined, { timeout: 30_000 });
  } catch (cause) {
    if (pageErrors.length > 0) {
      throw new Error(
        `harness never became ready; the page threw: ${pageErrors.map((e) => e.message).join("; ")}`,
        { cause },
      );
    }
    throw cause;
  }
}

async function webgpuAvailable(page: Page): Promise<boolean> {
  return page.evaluate(async () => window.__sih!.backendAvailable("webgpu"));
}

/** Everything about a finding that must survive a change of execution provider. */
const shape = (f: ResolvedFinding): string =>
  `${f.entityType}[${String(f.start)},${String(f.end)})=${JSON.stringify(f.text)}@${f.confidence.toFixed(3)}`;

for (const backend of TIER1_BACKENDS) {
  test(`tier-1 runs the real graph in real Chrome on ${backend}`, async ({ page }) => {
    test.setTimeout(ARM_TIMEOUT_MS);
    await openHarness(page);

    const supported = await page.evaluate(
      async (b: Tier1Backend) => window.__sih!.backendAvailable(b),
      backend,
    );
    test.skip(!supported, `${backend} not available in this browser`);

    const report = await page.evaluate(
      async (args: { backend: Tier1Backend; modelId: string }) => window.__sih!.loadTier1(args),
      { backend, modelId: SMOKE_MODEL },
    );

    // WHAT EXECUTED, not what was asked for. `config.backend` is the label the
    // arm carried in; `observedBackend` is measured inside loadTier1 by counting
    // GPUQueue.submit calls around a real warm-up inference. Asserting both, and
    // that they agree, is what stops a webgpu arm silently served by wasm from
    // reporting wasm latency under webgpu's name.
    expect(report.config.backend).toBe(backend);
    expect(report.observedBackend).toBe(backend);
    expect(report.config.modelId).toBe(SMOKE_MODEL);
    // The rest of the resolved config, because threshold, width and label form
    // all move every score: an arm recorded under a partial config is an arm
    // nobody can reproduce.
    expect(report.config.threshold).toBe(0.5);
    expect(report.config.maxWidth).toBe(12);
    expect(report.config.labelForm).toBe("id");

    // The bytes actually served, against the manifest's pinned size for this
    // rung -- read here in Node from @sih/tier1, never from the page. A page
    // answering from a stub, a 404 body or a different precision variant cannot
    // match this number by accident.
    const entry = MODEL_MANIFEST[SMOKE_MODEL]!;
    expect(report.weightsUrl).toContain(SMOKE_MODEL);
    expect(report.weightsUrl.endsWith(entry.weightsPath)).toBe(true);
    expect(report.weightsBytes).toBe(entry.files[entry.weightsPath]!.bytes);

    const result = await page.evaluate(
      async (text: string) =>
        window.__sih!.detect({
          text,
          provider: "claude",
          config: { tier0: false, tier1: true, tier2: false },
        }),
      MESSAGE,
    );

    for (const finding of result.findings) {
      expect(finding.tier).toBe(1);
      expect(finding.confidence).toBeGreaterThan(0);
      expect(finding.confidence).toBeLessThanOrEqual(1);
      // Span fidelity re-checked HERE, against the message this driver sent.
      // normalizeFindings enforces it inside the page, but once a finding has
      // crossed the protocol nothing about it is self-checking any more.
      expect(MESSAGE.slice(finding.start, finding.end)).toBe(finding.text);
      // The rung, not a family name: the record has to say which of the six
      // produced the number.
      expect(finding.source).toBe(SMOKE_MODEL);
    }

    // ---- ran, versus never ran ----------------------------------------------
    // An arm that loads nothing, finds nothing and reports tier1Ms 0 is
    // indistinguishable from a clean pass on findings alone, and this test is
    // deliberately not a recall gate -- model quality is Plan 8's measurement
    // over a real corpus, not a one-message smoke test. So the evidence that the
    // graph executed comes from the tagger's own counters instead.
    const status = await page.evaluate(() => window.__sih!.tier1Status());
    expect(status).toBeDefined();
    expect(status!.lastDetect).toBeDefined();
    // session.run calls the tagger made during THAT detect call. The page clears
    // `lastDetect` at the end of loadTier1, so the warm-up cannot supply this.
    expect(status!.lastDetect!.inferences).toBeGreaterThan(0);
    // Nothing was silently thrown away on the way from a decoded span to a
    // character range; offsets.ts asks callers to count these precisely because
    // a dropped span looks exactly like a model that found nothing.
    expect(status!.lastDetect!.unmappableSpans).toBe(0);

    // The same question asked of the GPU rather than of our own bookkeeping: the
    // webgpu arm must have dispatched work and the wasm arm must not have.
    if (backend === "webgpu") {
      expect(status!.lastDetect!.gpuSubmits).toBeGreaterThan(0);
    } else {
      expect(status!.lastDetect!.gpuSubmits).toBe(0);
    }

    expect(Number.isFinite(result.timings.tier1Ms)).toBe(true);
    expect(result.timings.tier1Ms).toBeGreaterThan(0);

    console.log(
      `[${backend}] loadMs=${report.loadMs.toFixed(0)} tier1Ms=${(result.timings.tier1Ms ?? 0).toFixed(1)} ` +
        `inferences=${String(status!.lastDetect!.inferences)} ` +
        `gpuSubmits=${String(status!.lastDetect!.gpuSubmits)} ` +
        `findings=${JSON.stringify(result.findings.map(shape))}`,
    );
  });
}

/**
 * Whether a rung's WebGPU output agrees with its WASM output, MEASURED on this
 * machine, and pinned here so that a change in either direction is a test
 * failure rather than a quiet change in what the numbers mean.
 *
 * This is a characterization test of an UPSTREAM DEFECT, not an aspiration.
 * Method (scratch harness, one fresh page per rung per provider, identical feeds
 * built by the shipped encoder): compare the raw `logits` element by element.
 *
 *   gliner-pii-edge         token_level   max |wasm - webgpu| = 8.352
 *   gliner-pii-edge-uint8   token_level   max |wasm - webgpu| = 8.134
 *   gliner-pii-base         markerV0      max |wasm - webgpu| = 0.000
 *   gliner-pii-base-uint8   markerV0      max |wasm - webgpu| = 30.154
 *
 * Bit-identical over two independent full re-runs, so it is deterministic
 * rather than numerical noise, and the shape of the disagreement is a COLLAPSE
 * rather than drift: base-uint8's logits span [-30.46, +2.05] on wasm and
 * [-0.36, -0.16] on webgpu, whose sigmoid is ~0.46 for everything -- which is
 * exactly the "every word scores 0.44-0.47" pattern the findings below show.
 *
 * Nothing reports this. Session creation succeeds, `run` resolves, the logits
 * are finite and correctly shaped. `gliner-pii-base` agreeing to 0.000 under the
 * same page code and the same readback path is the control that says the
 * harness is not the cause.
 *
 * The threshold is 0.02 rather than the default 0.5 because at 0.5 THREE of
 * these four rungs return nothing on either provider, so the arms would agree by
 * both being empty and this test would pass while proving nothing.
 */
const BACKEND_AGREEMENT: readonly { readonly modelId: string; readonly agrees: boolean }[] = [
  { modelId: "gliner-pii-edge", agrees: false },
  { modelId: "gliner-pii-edge-uint8", agrees: false },
  { modelId: "gliner-pii-base", agrees: true },
  { modelId: "gliner-pii-base-uint8", agrees: false },
];

for (const { modelId, agrees } of BACKEND_AGREEMENT) {
  test(`webgpu ${agrees ? "agrees with" : "DISAGREES with"} wasm on ${modelId}`, async ({ page }) => {
    test.setTimeout(ARM_TIMEOUT_MS);
    await openHarness(page);
    test.skip(!(await webgpuAvailable(page)), "webgpu not available in this browser");

    const run = async (backend: Tier1Backend): Promise<string[]> =>
      page.evaluate(
        async (args: { backend: Tier1Backend; modelId: string; text: string }) => {
          await window.__sih!.loadTier1({
            backend: args.backend,
            modelId: args.modelId,
            threshold: 0.02,
          });
          const result = await window.__sih!.detect({
            text: args.text,
            provider: "claude",
            config: { tier0: false, tier1: true, tier2: false },
          });
          return result.findings.map(
            (f) =>
              `${f.entityType}[${String(f.start)},${String(f.end)})=${JSON.stringify(f.text)}@${f.confidence.toFixed(3)}`,
          );
        },
        { backend, modelId, text: MESSAGE },
      );

    // wasm first, then webgpu into the same page: loadTier1 releases the
    // previous session before creating the next. MEASURED that this ordering
    // does not itself change the answer -- the scratch harness reproduced the
    // same per-rung numbers with one fresh page per provider.
    const onWasm = await run("wasm");
    const onWebgpu = await run("webgpu");
    console.log(`[${modelId}] wasm=${JSON.stringify(onWasm)}`);
    console.log(`[${modelId}] webgpu=${JSON.stringify(onWebgpu)}`);

    // Not `toEqual`/`not.toEqual` on the arrays alone: both being empty would
    // satisfy "agrees" without either provider having produced anything.
    expect(onWasm.length).toBeGreaterThan(0);
    expect(onWebgpu.length).toBeGreaterThan(0);
    if (agrees) {
      expect(onWebgpu).toEqual(onWasm);
    } else {
      expect(onWebgpu).not.toEqual(onWasm);
    }
  });
}

/**
 * The two fp16 rungs, which do not load anywhere.
 *
 * Task 7 measured both failing under onnxruntime-node with a `Cast` whose
 * declared output type contradicts its consumer. This asserts the BROWSER agrees
 * -- which is what makes it an upstream export defect rather than a runtime
 * difference, and what justifies a four-rung ladder instead of a six-rung one.
 * It also pins that the failure is LOUD: a rung that cannot run must throw here
 * rather than fall back to something that can.
 */
for (const modelId of ["gliner-pii-edge-fp16", "gliner-pii-base-fp16"]) {
  test(`${modelId} refuses to load rather than falling back`, async ({ page }) => {
    test.setTimeout(ARM_TIMEOUT_MS);
    await openHarness(page);
    const message = await page.evaluate(
      async (id: string) =>
        window.__sih!.loadTier1({ backend: "wasm", modelId: id }).then(
          () => "LOADED",
          (e: unknown) => (e instanceof Error ? e.message : String(e)),
        ),
      modelId,
    );
    expect(message).toContain("tensor(float16)");
    expect(message).toContain("does not match expected type (tensor(float))");
    // And the page is left with no tier-1 engine, so a caller cannot go on to
    // measure some earlier rung under this one's name.
    expect(await page.evaluate(() => window.__sih!.tier1Status())).toBeUndefined();
  });
}

/**
 * Span fidelity under a FLOOD of spans.
 *
 * The smoke arms above check fidelity on whatever the model happens to report at
 * threshold 0.5, which on this message is one span -- so they barely exercise
 * the segment-relative-to-absolute arithmetic that tagger.ts calls its most
 * likely place to be wrong. Dropping the threshold to 0.02 is not tuning for a
 * better-looking result: it maximises the number of spans the offset mapping has
 * to get right, over a message with an astral character before every one of
 * them, and every span still has to slice out of the message exactly.
 */
test("every span the model reports maps back to real characters", async ({ page }) => {
  test.setTimeout(ARM_TIMEOUT_MS);
  await openHarness(page);

  const { findings, status } = await page.evaluate(
    async (args: { modelId: string; text: string }) => {
      await window.__sih!.loadTier1({
        backend: "wasm",
        modelId: args.modelId,
        threshold: 0.02,
      });
      const result = await window.__sih!.detect({
        text: args.text,
        provider: "claude",
        config: { tier0: false, tier1: true, tier2: false },
      });
      return { findings: result.findings, status: window.__sih!.tier1Status() };
    },
    { modelId: SMOKE_MODEL, text: MESSAGE },
  );

  // Guaranteed by the threshold, not by the model being good at anything: at
  // 0.02 this graph reports spans for most word ranges it enumerates.
  expect(findings.length).toBeGreaterThan(0);
  for (const finding of findings) {
    expect(MESSAGE.slice(finding.start, finding.end)).toBe(finding.text);
    expect(finding.start).toBeGreaterThanOrEqual(0);
    expect(finding.end).toBeLessThanOrEqual(MESSAGE.length);
  }

  // The emoji is 2 UTF-16 units and sits before every span, so a code-point
  // offset would put every one of them one unit early -- and still slice
  // plausible-looking text. This is the assertion that separates the two.
  const emojiEnd = MESSAGE.indexOf("Contact");
  expect(emojiEnd).toBe(38);
  for (const finding of findings) expect(finding.start).toBeGreaterThanOrEqual(35);

  // Nothing in the fenced code block, which the orchestrator filters out before
  // the tagger sees it. The fence ends at index 34.
  for (const finding of findings) expect(finding.start).toBeGreaterThan(33);

  // Counted, not merely absent: a span the mapper refused is invisible in
  // `findings` and would otherwise read as a model that found less.
  expect(status!.lastDetect!.unmappableSpans).toBe(0);
  expect(status!.lastDetect!.inferences).toBeGreaterThan(0);
  console.log(`[fidelity] ${String(findings.length)} spans, all slicing correctly`);
});

/**
 * `TierConfig.backend` reconciled against what is actually loaded.
 *
 * MEASURED in Task 3 and still true: nothing under `packages/core/src` reads
 * `TierConfig.backend`, and `runArm` copies its own `ArmSpec.backend` onto every
 * record without plumbing it into the config it forwards. So a caller can hand
 * `detect` a config saying `webgpu` while the page is running wasm, and every
 * record from that arm would carry the wrong provider with nothing to catch it.
 * The page is the one place both values are known.
 *
 * The wasm arm is used because it is the one that exists on every machine, so
 * this cannot become a test that only runs where there is a GPU.
 */
test("detect refuses a TierConfig whose backend contradicts the loaded model", async ({ page }) => {
  test.setTimeout(ARM_TIMEOUT_MS);
  await openHarness(page);

  const outcome = await page.evaluate(
    async (args: { modelId: string; text: string }) => {
      const report = await window.__sih!.loadTier1({ backend: "wasm", modelId: args.modelId });
      const attempt = (backend: "wasm" | "webgpu"): Promise<string> =>
        window.__sih!.detect({
          text: args.text,
          provider: "claude",
          config: { tier0: false, tier1: true, tier2: false, backend },
        }).then(
          () => "OK",
          (e: unknown) => (e instanceof Error ? e.message : String(e)),
        );
      return {
        observed: report.observedBackend,
        matching: await attempt("wasm"),
        contradicting: await attempt("webgpu"),
      };
    },
    { modelId: SMOKE_MODEL, text: MESSAGE },
  );

  expect(outcome.observed).toBe("wasm");
  // The agreeing config is not merely tolerated -- it still detects.
  expect(outcome.matching).toBe("OK");
  expect(outcome.contradicting).toContain("config.backend is webgpu");
  expect(outcome.contradicting).toContain("running on wasm");
});
