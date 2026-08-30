/**
 * The ASSEMBLED tagger against the REAL weights, through core's `detect`.
 *
 * Everything else in this package tests a piece against a fake. This is the one
 * place where a pinned graph, a pinned tokenizer, real text and core's
 * `normalizeFindings` meet -- and `normalizeFindings` THROWS on
 * `text !== message.slice(start, end)`, so reaching an assertion at all is
 * already the strongest statement available about the offset arithmetic.
 *
 * ## Why it skips instead of failing
 *
 * The weights are ~1.5 GB and gitignored, so a fresh clone has none. Every other
 * test in this package runs without them and must keep doing so; making this one
 * a hard requirement would turn `pnpm -r test` red on any machine that has not
 * spent the download. `describe.skipIf` plus the console note below is the whole
 * mechanism -- see `WEIGHTS_PRESENT`.
 *
 * ## Why onnxruntime-NODE stands in here
 *
 * The measured runtime path is onnxruntime-web in a browser, and
 * `apps/eval/test/tier1.spec.ts` is where that runs. This file exists for the
 * half a browser cannot give: a fast, debuggable check that the composition in
 * tagger.ts is right, on a runtime that needs no dev server. The Node binding is
 * injected through `createOrtSession`'s own `loadRuntime` seam, so the REAL
 * `toRuntimeTensors` conversion still builds every feed.
 *
 * The two runtimes were compared directly rather than assumed equivalent. On
 * this machine, with the same 1-class IR and the same message, onnxruntime-node
 * 1.21 CPU and onnxruntime-web 1.22-dev WASM produced IDENTICAL spans on all
 * four loadable rungs, and identical scores to 3 dp on both fp32 rungs; only
 * `gliner-pii-edge-uint8` drifted, by <= 0.03.
 *
 * READ THE TOLERANCE OFF THE ASSERTIONS, not off that 0.03. `EXPECTED` below
 * covers exactly the two fp32 rungs -- `gliner-pii-edge` and `gliner-pii-base`
 * -- and asserts with `toBeCloseTo(score, 2)`, which is +/-0.005, not +/-0.03.
 * The 0.03 belongs to `gliner-pii-edge-uint8`, a rung this file does not assert
 * at all. So the tolerance in force is +/-0.005 over two rungs whose two
 * runtimes agreed to 3 dp, which is a real margin above the observed gap rather
 * than a number borrowed from the worst rung on the ladder. An earlier version
 * of this comment presented the 0.03 as the tolerance's basis; it is not.
 */
import { existsSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { AutoTokenizer, env } from "@huggingface/transformers";
import * as ortNode from "onnxruntime-node";
import { detect, loadPolicyIr, type EntityType, type PolicyIr } from "@sih/core";
import { resolveTier1Config, type Tier1Config } from "../src/config.js";
import { tokenizerFromEncoder } from "../src/encode.js";
import { MODEL_MANIFEST } from "../src/manifest.js";
import { createOrtSession, type OrtRuntime } from "../src/session.js";
import { GlinerSpanTagger } from "../src/tagger.js";

const MODELS_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "models");

/**
 * Present enough to run: the directory exists and holds at least one rung.
 *
 * Checked at MODULE scope so the note below is printed once rather than per
 * test, and so the whole describe block is skipped rather than each test failing
 * separately on the same missing file.
 */
const WEIGHTS_PRESENT =
  existsSync(MODELS_DIR) &&
  readdirSync(MODELS_DIR).some((name) => Object.hasOwn(MODEL_MANIFEST, name));

if (!WEIGHTS_PRESENT) {
  console.warn(
    `[tier1 e2e] skipped: no model weights under ${MODELS_DIR}. ` +
      "Fetch them with `pnpm -C packages/tier1 exec vite-node ../../scripts/fetch-models.ts` " +
      "(scripts/fetch-models.ts verifies every file against MODEL_MANIFEST's pinned hashes).",
  );
} else {
  env.allowRemoteModels = false;
  env.localModelPath = MODELS_DIR;
}

/**
 * Three tier-1 classes, so the class axis is wider than one and a span that
 * decodes to the wrong `classIndex` names the wrong entityType instead of
 * silently landing on the only one available.
 */
const ENTITY_TYPES: EntityType[] = [
  {
    id: "client-name",
    tier: 1,
    nlDefinition: "A client organisation the firm does business with.",
    examples: [],
    counterExamples: [],
    severity: "high",
    surrogateKind: "org-name",
  },
  {
    id: "person-name",
    tier: 1,
    nlDefinition: "A natural person's name.",
    examples: [],
    counterExamples: [],
    severity: "medium",
    surrogateKind: "person-name",
  },
  {
    id: "email-address",
    tier: 1,
    nlDefinition: "An email address.",
    examples: [],
    counterExamples: [],
    severity: "medium",
    surrogateKind: "opaque",
  },
];

const IR_JSON = JSON.stringify({
  irVersion: "1",
  policyHash: "tier1-e2e",
  entityTypes: ENTITY_TYPES,
  rules: [],
  semanticPredicates: [],
  actions: {
    default: Object.fromEntries(ENTITY_TYPES.map((e) => [e.id, "pseudonymize" as const])),
  },
  failMode: "closed",
  latencyBudgetMs: 5000,
  provenance: {},
} satisfies PolicyIr);

/**
 * A fenced code block the orchestrator must keep away from the tagger, an astral
 * character before every span, and prose holding a person, an email and an
 * organisation. `apps/eval/test/tier1.spec.ts` uses this same string, so a
 * browser/Node difference is attributable to the runtime and nothing else.
 */
const MESSAGE =
  "```\nconst apiKey = \"redacted\";\n```\n" +
  "\u{1F642} Contact Priya Sharma at priya@acme.io about the Northwind Traders renewal.";

async function taggerFor(modelId: string): Promise<GlinerSpanTagger> {
  const entry = MODEL_MANIFEST[modelId];
  if (entry === undefined) throw new Error(`unknown model ${modelId}`);
  const hf = await AutoTokenizer.from_pretrained(modelId);
  const tokenizer = tokenizerFromEncoder((text: string, addSpecialTokens: boolean) =>
    Array.from(
      hf(text, { add_special_tokens: addSpecialTokens }).input_ids.data as BigInt64Array,
      Number,
    ),
  );
  // The backend argument is onnxruntime-WEB's vocabulary; the Node binding has
  // its own providers, so this stub drops it. Everything else -- including the
  // tensor conversion under test -- is the real path.
  const runtime = {
    InferenceSession: { create: (path: string) => ortNode.InferenceSession.create(path) },
    Tensor: ortNode.Tensor,
  } as unknown as OrtRuntime;
  const session = await createOrtSession(
    join(MODELS_DIR, modelId, entry.weightsPath),
    "wasm",
    () => Promise.resolve(runtime),
  );
  const config: Tier1Config = resolveTier1Config({ modelId, backend: "wasm", threshold: 0.5 });
  return new GlinerSpanTagger(session, tokenizer, config);
}

const describeSpan = (f: { entityType: string; start: number; end: number; text: string }): string =>
  `${f.entityType}[${String(f.start)},${String(f.end)})=${f.text}`;

/**
 * What each rung reports at threshold 0.5, MEASURED on this machine against
 * these pinned weights.
 *
 * Pinned as spans plus a score TOLERANCE rather than exact floats: the score is
 * a runtime detail (the two ORT builds differ by up to 0.03 on a quantized
 * rung), while the span is the thing the whole package exists to get right. A
 * changed span is a real regression; a changed third decimal is not.
 *
 * These numbers are NOT a quality claim. `gliner-pii-edge` misses the
 * organisation entirely here, which is recorded rather than tuned away -- what
 * the ladder is actually worth is Plan 8's measurement over a real corpus.
 */
const EXPECTED: Readonly<Record<string, readonly { span: string; score: number }[]>> = {
  "gliner-pii-edge": [
    { span: "person-name[46,58)=Priya Sharma", score: 0.722 },
    { span: "email-address[62,75)=priya@acme.io", score: 0.586 },
  ],
  "gliner-pii-base": [
    { span: "person-name[46,58)=Priya Sharma", score: 0.635 },
    { span: "email-address[62,75)=priya@acme.io", score: 0.99 },
    { span: "client-name[86,103)=Northwind Traders", score: 0.643 },
  ],
};

describe.skipIf(!WEIGHTS_PRESENT)("tier-1 end to end against real weights", () => {
  for (const modelId of Object.keys(EXPECTED)) {
    it(
      `${modelId} detects through core over real text`,
      async () => {
        const tagger = await taggerFor(modelId);
        const result = await detect({
          ir: loadPolicyIr(IR_JSON),
          provider: "claude",
          text: MESSAGE,
          config: { tier0: false, tier1: true, tier2: false },
          engines: { tier1: tagger },
        });

        // The contract core enforces itself, restated so a failure names it.
        for (const f of result.findings) {
          expect(MESSAGE.slice(f.start, f.end)).toBe(f.text);
        }
        // Ran, versus never ran: an empty `findings` is a legitimate result for
        // a span tagger, so it cannot be the evidence that the graph executed.
        expect(tagger.stats.inferences).toBeGreaterThan(0);
        expect(tagger.stats.unmappableSpans).toBe(0);

        const expected = EXPECTED[modelId] as readonly { span: string; score: number }[];
        expect(result.findings.map(describeSpan)).toEqual(expected.map((e) => e.span));
        result.findings.forEach((f, i) => {
          expect(f.confidence).toBeCloseTo((expected[i] as { score: number }).score, 2);
        });

        // Nothing from inside the fenced block, which ends at index 34: the
        // orchestrator filters code segments out before the tagger sees them.
        for (const f of result.findings) expect(f.start).toBeGreaterThan(33);
      },
      600_000,
    );
  }

  /**
   * The lockstep drop, reached from a MESSAGE rather than from a fake.
   *
   * U+FEFF is `\S` to the word splitter, so it is a word, and it encodes to zero
   * subwords on the pinned base tokenizer -- so the model's word axis is shorter
   * than the split. encode.ts drops the word from both the word list and the
   * slot numbering together; if it dropped it from only one, every span after
   * the BOM would name its neighbour and still slice cleanly.
   *
   * Run WITH and WITHOUT the BOM so a difference is attributable, and inserted
   * immediately before the organisation so a one-word desync moves a span that
   * the run without it places exactly.
   */
  it(
    "keeps offsets exact across a word that tokenises to nothing",
    async () => {
      const bare = MESSAGE;
      const bommed = MESSAGE.replace("Northwind", "\u{FEFF}Northwind");
      expect(bommed.length).toBe(bare.length + 1);

      const spansFor = async (text: string): Promise<{ spans: string[]; dropped: number }> => {
        const tagger = await taggerFor("gliner-pii-base");
        const result = await detect({
          ir: loadPolicyIr(IR_JSON),
          provider: "claude",
          text,
          config: { tier0: false, tier1: true, tier2: false },
          engines: { tier1: tagger },
        });
        for (const f of result.findings) expect(text.slice(f.start, f.end)).toBe(f.text);
        return { spans: result.findings.map(describeSpan), dropped: tagger.stats.droppedWords };
      };

      const withoutBom = await spansFor(bare);
      const withBom = await spansFor(bommed);
      expect(withoutBom.dropped).toBe(0);
      // The measurement, not an assumption: this is the tokenizer returning zero
      // subwords for the BOM, which is what makes the rest of the test mean
      // anything. If a re-pin changed it, the guard below would pass vacuously.
      expect(withBom.dropped).toBe(1);

      // Every span at or after the BOM shifts by exactly one UTF-16 unit;
      // everything before it is untouched. Derived from the no-BOM run rather
      // than restated, so this cannot drift into agreeing with itself.
      const bomAt = bommed.indexOf("\u{FEFF}");
      const shifted = withoutBom.spans.map((span) =>
        span.replace(/\[(\d+),(\d+)\)/, (_m, s: string, e: string) => {
          const start = Number(s);
          const end = Number(e);
          const shift = (n: number): number => (n >= bomAt ? n + 1 : n);
          return `[${String(shift(start))},${String(shift(end))})`;
        }),
      );
      expect(withBom.spans).toEqual(shifted);
    },
    600_000,
  );
});
