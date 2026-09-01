import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { MODEL_MANIFEST, TIER1_BACKENDS, type Tier1Backend } from "@sih/tier1";
import type { ResolvedFinding } from "@sih/core";
import { BACKEND_AGREEMENT } from "../src/driver/main.js";

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
 * (see `BACKEND_AGREEMENT`, imported from src/driver/main.ts): this is the ONLY
 * rung of the four that load at all whose WebGPU output agrees with its WASM
 * output, so it is the only rung on which a wasm arm and a webgpu arm are
 * measuring the same model. On the other three, a webgpu arm silently returns
 * different numbers, which is a far worse source of flakiness than a large file.
 *
 * "Agrees" is not "bit for bit". Task 11 diffed raw logits element-wise on ONE
 * message and got 0.000. Running the whole 13-item smoke corpus through both
 * providers at threshold 0.02 instead: every span boundary and every entityType
 * is identical on all 13 items (47 tier-1 findings per arm), while CONFIDENCES
 * differ by ~1e-7 typically and by up to 2.6e-4 on the two multi-line items --
 * the longest inputs in the corpus.
 *
 * Nor is it bit for bit against ITSELF, which is the sharper statement and the
 * one that says why the rung is still usable. MEASURED here: six consecutive
 * `detect` calls on the same text in one page, on this rung at 0.02. WASM
 * returns bit-identical confidences all six times at 153 characters and at
 * 2,148. WEBGPU differs from the previous pass every time at 153 characters
 * (max |delta| 1.71e-4) and on the first three at 2,148 (max |delta| 4.89e-3).
 * So the provider difference above is no larger than the provider's
 * disagreement with itself, and it stays usable for the reason it always did:
 * nothing that is SCORED moved. No span, no label, no ordering -- in all twelve
 * of those passes.
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
 * machine, and pinned so that a change in either direction is a test failure
 * rather than a quiet change in what the numbers mean.
 *
 * The table itself lives in `src/driver/main.ts`, imported above, because
 * `runMatrix` REFUSES a webgpu arm on any rung it says disagrees. Keeping one
 * copy is what makes that guard and this measurement impossible to drift apart:
 * a rung moved into the trusting column here immediately changes which arms the
 * driver will run, and vice versa.
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
 * What makes these three verdicts safe despite the jitter recorded above is the
 * MAGNITUDE: three to four orders of magnitude separates 8.352 / 8.134 / 30.154
 * from the ~5e-3 webgpu varies by between two runs of the same input, so no
 * re-run could turn a "wrong" verdict into an "exact" one. An earlier version of
 * this comment claimed the four numbers were "bit-identical over two independent
 * full re-runs, so it is deterministic rather than numerical noise"; the
 * determinism half of that is false and the scale argument, which is what the
 * verdicts actually rest on, is not.
 *
 * That same version said the disagreement is "a COLLAPSE rather than drift:
 * base-uint8's logits span [-30.46, +2.05] on wasm and [-0.36, -0.16] on webgpu,
 * whose sigmoid is ~0.46 for everything -- exactly the 'every word scores
 * 0.44-0.47' pattern the findings below show". The last clause is WRONG and the
 * tests in this file are what disprove it. MEASURED twice, byte-identical, at
 * threshold 0.02 on MESSAGE below:
 *
 *   gliner-pii-edge         webgpu confidences 0.103 .. 0.204
 *   gliner-pii-edge-uint8   webgpu confidences 0.341 .. 0.475  <- the 0.44-0.47 run
 *   gliner-pii-base-uint8   webgpu confidences 0.023 .. 0.066
 *
 * So the 0.44-0.47 pattern is EDGE-uint8's, not base-uint8's. And on a markerV0
 * rung `confidence` is `sigmoid(logit)` with nothing in between (decode.ts), so
 * a 0.023 needs a logit near -3.75 and base-uint8's webgpu logits cannot all lie
 * in [-0.36, -0.16] on this message. That range is Task 11's, over the raw
 * tensor on a different short message; it is not re-derivable from this suite
 * and is left standing as that measurement rather than restated as this one's.
 * The verdicts do not depend on it.
 *
 * Nothing reports this. Session creation succeeds, `run` resolves, the logits
 * are finite and correctly shaped. `gliner-pii-base` agreeing to 0.000 under the
 * same page code and the same readback path is the control that says the
 * harness is not the cause.
 *
 * The threshold is 0.02 rather than the default 0.5, and the reason is
 * measured. At 0.5 on this message and the 1-class minimal IR:
 *
 *   gliner-pii-edge         wasm 0 findings   webgpu 0
 *   gliner-pii-edge-uint8   wasm 0            webgpu 0
 *   gliner-pii-base         wasm 1            webgpu 0
 *   gliner-pii-base-uint8   wasm 1            webgpu 0
 *
 * So TWO of the four -- not three, as this comment used to say -- return
 * nothing on BOTH providers, and those two arms would agree by being equally
 * empty while this test passed proving nothing. The other two are worse for
 * this test rather than better: they return one finding on wasm and none on
 * webgpu, so `expect(onWebgpu.length).toBeGreaterThan(0)` below would fail on a
 * rung the table calls exact. 0.02 is what makes every cell of that grid
 * non-empty.
 */
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
 * More than one tier-1 class, in the browser.
 *
 * THE GAP THIS CLOSES. `fixtures/minimal-ir.json` declares exactly one tier-1
 * entityType, so every other browser assertion in this file runs at
 * `classes = 1`. At that width `buildLabels` only ever assigns classIndex 0,
 * `decodeEdgeSpans`' `(word * classes + classIndex) * slots` collapses to
 * `word * slots`, and `decodeBaseSpans`' `(firstWord * widths + width) * classes
 * + classIndex` collapses to `firstWord * widths + width` -- so a class-axis
 * stride bug reads exactly the cells a correct implementation reads, and every
 * label that comes back is the only label there was. The multi-class coverage
 * that did exist ran under onnxruntime-NODE (packages/tier1/test/e2e.test.ts),
 * which spec 2.2 says is not the runtime being measured.
 *
 * `fixtures/multiclass-ir.json` declares three, in the order client-name,
 * person-name, email-address, and `window.__sih.useIr` switches the page onto
 * it. The order is the assertion's teeth: a decoder that ignored the class axis
 * would label every span `client-name`, which is classIndex 0 -- and the two
 * spans this test demands are classIndex 1 and 2.
 *
 * Both rungs, because the two span modes have DIFFERENT class strides and only
 * one of them can be checked by any one graph. Both on wasm: three of the four
 * loadable rungs return wrong numbers on webgpu (see BACKEND_AGREEMENT), so a
 * webgpu arm here would be asserting against a known-broken provider.
 *
 * MEASURED in this browser, and identical to what packages/tier1/test/e2e.test.ts
 * gets from onnxruntime-node on the same message with the same three labels --
 * `gliner-pii-edge` 0.722/0.586 and `gliner-pii-base` 0.635/0.990/0.643. Scores
 * are deliberately NOT asserted: the spans and the labels are what a stride bug
 * moves.
 */
const MULTICLASS_IR = join(import.meta.dirname, "..", "fixtures", "multiclass-ir.json");

const MULTICLASS_EXPECTED: Readonly<Record<string, readonly string[]>> = {
  "gliner-pii-edge": [
    'person-name[46,58)="Priya Sharma"',
    'email-address[62,75)="priya@acme.io"',
  ],
  "gliner-pii-base": [
    'person-name[46,58)="Priya Sharma"',
    'email-address[62,75)="priya@acme.io"',
    'client-name[86,103)="Northwind Traders"',
  ],
};

for (const modelId of Object.keys(MULTICLASS_EXPECTED)) {
  test(`${modelId} labels spans of different tier-1 classes distinctly`, async ({ page }) => {
    test.setTimeout(ARM_TIMEOUT_MS);
    await openHarness(page);

    // Read out of the fixture FILE, in Node, so this test states which class
    // index each label has rather than trusting the page to agree with itself.
    // buildLabels assigns classIndex over the tier-1 entityTypes in IR order.
    const fixture = readFileSync(MULTICLASS_IR, "utf8");
    const tier1Ids = (JSON.parse(fixture) as { entityTypes: { id: string; tier: number }[] })
      .entityTypes.filter((e) => e.tier === 1)
      .map((e) => e.id);
    expect(tier1Ids.length).toBeGreaterThanOrEqual(3);
    const classZero = tier1Ids[0]!;

    const out = await page.evaluate(
      async (args: { modelId: string; text: string }) => {
        const irHash = await window.__sih!.useIr("multiclass");
        const report = await window.__sih!.loadTier1({ backend: "wasm", modelId: args.modelId });
        const result = await window.__sih!.detect({
          text: args.text,
          provider: "claude",
          config: { tier0: false, tier1: true, tier2: false },
        });
        return {
          irHash,
          policyHash: window.__sih!.policyHash(),
          classes: report.config.labelForm,
          findings: result.findings,
          status: window.__sih!.tier1Status(),
        };
      },
      { modelId, text: MESSAGE },
    );

    // The page really switched artifact, and says so in the field a record
    // carries: without this the whole test could be running the 1-class IR.
    expect(out.irHash).toBe(createHash("sha256").update(fixture).digest("hex"));
    expect(out.policyHash).toBe("multiclass-fixture-hash");
    // Prompts are built from the entityType ids (labelForm "id"), which is what
    // makes the label set policy-derived rather than baked into the model.
    expect(out.classes).toBe("id");

    const spans = out.findings.map(
      (f) => `${f.entityType}[${String(f.start)},${String(f.end)})=${JSON.stringify(f.text)}`,
    );
    expect(spans).toEqual(MULTICLASS_EXPECTED[modelId]);

    // Said again, as the property rather than as a list, because THIS is the
    // thing the one-class fixture could not express: more than one class came
    // back, and not the one a collapsed stride would produce.
    const labels = new Set(out.findings.map((f) => f.entityType));
    expect(labels.size).toBeGreaterThanOrEqual(2);
    expect(labels.has(classZero)).toBe(modelId === "gliner-pii-base");
    for (const f of out.findings) {
      expect(tier1Ids).toContain(f.entityType);
      expect(MESSAGE.slice(f.start, f.end)).toBe(f.text);
      expect(f.tier).toBe(1);
      expect(f.source).toBe(modelId);
    }

    // Ran, versus never ran, and nothing silently thrown away on the way.
    expect(out.status!.lastDetect!.inferences).toBeGreaterThan(0);
    expect(out.status!.lastDetect!.unmappableSpans).toBe(0);
    expect(out.status!.lastDetect!.nonFiniteScores).toBe(0);
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

  // Guaranteed by the threshold, not by the model being good at anything -- but
  // NOT a flood, which is what this comment used to claim ("at 0.02 this graph
  // reports spans for most word ranges it enumerates"). MEASURED: the prose
  // segment of MESSAGE is 16 words, so at one class and width <= 12 the decoder
  // enumerates 126 in-range word ranges; at 0.02 it decodes 5 of them (4%) and
  // 3 survive core's merge, against 1 at the default 0.5. Tripling the spans
  // the offset mapping has to get right is the whole benefit, and it is a
  // modest one; the astral character before every span is what makes the three
  // worth checking.
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
 * `TierConfig.backend`. So a caller can hand `detect` a config saying `webgpu`
 * while the page is running wasm, and nothing inside core would notice. The
 * page is the one place both values are known.
 *
 * This comment used to add that `runArm` "copies its own `ArmSpec.backend` onto
 * every record without plumbing it into the config it forwards". That was true
 * when it was written and Task 12 closed it: `runArm` now builds ONE config
 * object carrying `spec.backend`, uses it for both the `detect` call and the
 * record, and refuses an arm whose `TierConfig` names a different backend.
 * `test/run.spec.ts` covers that end. What remains for THIS test is the case
 * neither of those catches -- a caller reaching `window.__sih.detect` directly
 * with a contradicting config -- which is why the check lives in the page.
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
