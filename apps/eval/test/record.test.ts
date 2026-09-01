import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  UNCERTAIN_BELOW,
  detect,
  loadPolicyIr,
  type DegradedNotice,
  type DetectionResult,
  type SemanticJudge,
} from "@sih/core";
import { resolveTier1Config } from "@sih/tier1";
import { CorpusItemSchema, loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema, toJsonl, type RunRecord } from "../src/driver/record.js";

const FIXTURES = join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures");
const IR_FIXTURE = join(import.meta.dirname, "..", "fixtures", "minimal-ir.json");

/**
 * The `absent` entries core files for every tier a `TierConfig` switched off,
 * worded as `absentNotice` in packages/core/src/detect/orchestrator.ts words
 * them.
 *
 * Every fixture below is a tier-0 record, so every one of them has two. Written
 * out rather than left as `[]` because `[]` on a tier-0 record is not merely
 * terse, it is FALSE -- `DetectionResult.degraded`'s own doc calls an `absent`
 * entry a statement of coverage and says a tier-0 run legitimately carries two
 * of them. A fixture that models the shape wrongly is how the shape gets built
 * wrongly.
 */
const ABSENT_1_2: DegradedNotice[] = [
  { tier: 1, reason: "absent", detail: "tier 1 was not enabled in this TierConfig, so nothing it detects was looked for" },
  { tier: 2, reason: "absent", detail: "tier 2 was not enabled in this TierConfig, so nothing it detects was looked for" },
];

describe("corpus", () => {
  it("loads the smoke corpus, rejecting any malformed line by number", () => {
    const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));
    expect(items.length).toBeGreaterThanOrEqual(12);
    // Deliberately NOT `items.every(i => CorpusItemSchema.safeParse(i).success)`,
    // which is what stood here and could not fail: loadCorpus returns
    // `safeParse(...).data`, so re-parsing its own output is a tautology that
    // pins a property of the fixture rather than anything the loader does.
    // What the loader actually does with a line is strip what the schema does
    // not declare and keep `meta` whatever it holds -- the behaviour the README
    // warns Plan 7 about, because provenance columns put anywhere but `meta` are
    // lost without a word.
    const [loaded] = loadCorpus(
      '{"id":"a","text":"hello","policy":"p-fin","gold":[],"provenanceColumn":"dropped",' +
        '"meta":{"source":"sharechat","injection":{"carrier":"code-fence","depth":2}}}',
    );
    expect(loaded).toBeDefined();
    expect(loaded).not.toHaveProperty("provenanceColumn");
    // `meta` survives verbatim, nested values included -- it is typed to keep
    // whatever it is given, which is the escape hatch the stripping makes necessary.
    expect(loaded!.meta).toEqual({
      source: "sharechat",
      injection: { carrier: "code-fence", depth: 2 },
    });
  });

  it("names the line number when a record is malformed", () => {
    expect(() => loadCorpus('{"id":"a","text":"x","policy":"p-fin","gold":[]}\n{"id":\n')).toThrow(
      /line 2/,
    );
  });

  it("requires every gold span to quote the text it claims", () => {
    // The corpus is ground truth. A gold span whose offsets do not hold the
    // text it names would silently score every arm against fiction — the same
    // failure normalizeFindings refuses for findings, applied to labels.
    const bad = { id: "a", text: "hello world", policy: "p-fin", gold: [{ start: 0, end: 5, text: "WRONG", entityType: "client-name", action: "block" }] };
    expect(CorpusItemSchema.safeParse(bad).success).toBe(false);
  });
});

describe("RunRecordSchema", () => {
  it("accepts a record carrying everything Plan 8 needs to score without re-running", () => {
    const record = {
      schemaVersion: 1,
      runId: "r1",
      itemId: "a",
      policy: "p-fin",
      irHash: "0".repeat(64),
      policyHash: "1".repeat(64),
      arm: "t0",
      backend: "wasm",
      detector: "core-orchestrator",
      provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
      // `text` added to the specified fixture when RunRecordSchema gained the
      // field: the record now carries the message its offsets index into, so
      // this fixture has to supply one for the spans to validate against.
      text: "hello world",
      findings: [{ start: 0, end: 5, text: "hello", entityType: "client-name", severity: "high", tier: 0, source: "rule", confidence: 0.9, action: "block" }],
      gold: [{ start: 0, end: 5, text: "hello", entityType: "client-name", action: "block" }],
      timings: { tier0Ms: 0.4 },
      error: null,
      degraded: ABSENT_1_2,
    abandonedWorkInFlight: false,
    };
    expect(RunRecordSchema.safeParse(record).success).toBe(true);
  });

  it("carries both hashes, so a record can never be attributed to the wrong IR", () => {
    // Two fields because they answer two different questions, and neither is
    // optional. `irHash` says which IR ARTIFACT ran; `policyHash` is the IR's
    // own field, the compiler's hash of the policy DOCUMENT it was compiled
    // from. Dropping either breaks the document -> IR -> numbers chain at a
    // different link.
    const complete = {
      schemaVersion: 1, runId: "r1", itemId: "a", policy: "p-fin",
      irHash: "0".repeat(64), policyHash: "1".repeat(64),
      arm: "t0", backend: "wasm", detector: "core-orchestrator", provider: "claude",
    config: { tier0: true, tier1: false, tier2: false }, text: "hello world", findings: [], gold: [],
      timings: { tier0Ms: 0 }, error: null, degraded: ABSENT_1_2,
    abandonedWorkInFlight: false,
    };
    expect(RunRecordSchema.safeParse(complete).success).toBe(true);
    const { irHash: _noIrHash, ...withoutIrHash } = complete;
    expect(RunRecordSchema.safeParse(withoutIrHash).success).toBe(false);
    const { policyHash: _noPolicyHash, ...withoutPolicyHash } = complete;
    expect(RunRecordSchema.safeParse(withoutPolicyHash).success).toBe(false);
  });
});

/**
 * The block above is the contract as specified. The block below closes gaps the
 * specified tests demonstrably do not cover: each test here was written against
 * a mutation that left all five of the tests above passing. Cited per test.
 */
describe("guards the specified tests leave open", () => {
  const item = (gold: unknown) => ({ id: "a", text: "hello world", policy: "p-fin", gold });

  it("rejects a gold span whose end runs past the end of the text", () => {
    // MEASURED: deleting `g.end <= item.text.length` from the refine leaves all
    // five specified tests green. String.slice CLAMPS rather than throwing, so
    // slice(0, 9999) on "hello world" returns "hello world" and compares equal
    // to a gold text of "hello world" -- the comparison alone cannot see that
    // `end` was never a real offset. Python's str slicing clamps identically,
    // so Plan 8 would inherit the same blind spot rather than catch it.
    expect(CorpusItemSchema.safeParse(item([{ start: 0, end: 9999, text: "hello world", entityType: "in-pan", action: "block" }])).success).toBe(false);
  });

  it("rejects a zero-width gold span", () => {
    // MEASURED: dropping `.min(1)` from GoldSpanSchema.text leaves all five
    // specified tests green, and {start:5,end:5,text:""} then validates --
    // slice(5,5) is "" and equals the claimed text, so the refine agrees. A
    // zero-width label is not a detectable entity; it is a labelling slip that
    // would count as a missed positive against every arm.
    expect(CorpusItemSchema.safeParse(item([{ start: 5, end: 5, text: "", entityType: "in-pan", action: "block" }])).success).toBe(false);
  });

  it("rejects a negative start offset", () => {
    // MEASURED: relaxing `start` from .nonnegative() to .int() leaves every
    // other test in this file green -- stated without a count, because the last
    // revision of this comment named one and the next commit invalidated it --
    // because a negative index does not fail the refine:
    // slice() rebases it from the end of the string, so {start:-11,end:5} on an
    // 11-char text still yields "hello" and compares equal. A negative offset is
    // not a position in the message under either JS or Python indexing rules.
    expect(CorpusItemSchema.safeParse(item([{ start: -11, end: 5, text: "hello", entityType: "in-pan", action: "block" }])).success).toBe(false);
  });

  it("rejects a gold action outside the policy's action vocabulary", () => {
    // MEASURED: widening the action enum to z.string() leaves all five
    // specified tests green. The vocabulary is core's Action union plus "none",
    // and a plausible-looking typo is exactly what a hand-authored corpus
    // produces; unchecked it becomes a class Plan 8 silently never scores.
    expect(CorpusItemSchema.safeParse(item([{ start: 0, end: 5, text: "hello", entityType: "in-pan", action: "blocked" }])).success).toBe(false);
  });
});

describe("RunRecordSchema guards the specified tests leave open", () => {
  const record = {
    // `as const` rather than a bare 1: this fixture is passed to toJsonl, whose
    // parameter is RunRecord, and z.literal narrows schemaVersion to exactly 1.
    // Written as a literal rather than imported from RECORD_SCHEMA_VERSION so a
    // version bump surfaces here as a compile error to think about, instead of
    // the fixture silently following the constant it is meant to pin.
    schemaVersion: 1 as const,
    runId: "r1",
    itemId: "a",
    policy: "p-fin",
    irHash: "0".repeat(64),
    policyHash: "1".repeat(64),
    arm: "t0",
    backend: "wasm" as const,
    detector: "core-orchestrator" as const,
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    text: "hello world",
    findings: [],
    gold: [],
    timings: { tier0Ms: 0 },
    error: null,
    degraded: ABSENT_1_2,
    abandonedWorkInFlight: false,
  };

  /**
   * A tier-1 record needs `tier1Stats` as well as `tier1Config` -- the schema
   * couples both to `config.tier1` -- so the tier-1 cases below carry this.
   * Zeros are the honest value for a synthetic record: nothing ran, so nothing
   * was lost.
   */
  const STATS = {
    inferences: 1,
    droppedWords: 0,
    truncatedWords: 0,
    overWideSpans: 0,
    unmappableSpans: 0,
    nonFiniteScores: 0,
  };

  it("refuses a schemaVersion other than the one this module writes", () => {
    // MEASURED: relaxing z.literal to z.number() leaves all five specified
    // tests green. RECORD_SCHEMA_VERSION's own doc comment promises Plan 8
    // refuses a version it does not know; without the literal, a future v2
    // record parses here as if it were v1 and the promise is not kept.
    expect(RunRecordSchema.safeParse({ ...record, schemaVersion: 2 }).success).toBe(false);
  });

  it("refuses an irHash that is not a sha256 digest", () => {
    // MEASURED: relaxing the regex to a bare z.string() leaves all five
    // specified tests green, because the only hash they supply is a well-formed
    // one. "test-hash" is not hypothetical: it is the literal policyHash in
    // apps/eval/fixtures/minimal-ir.json today, and forwarding that placeholder
    // as the artifact hash is exactly the mistake this regex exists to refuse.
    expect(RunRecordSchema.safeParse({ ...record, irHash: "test-hash" }).success).toBe(false);
    expect(RunRecordSchema.safeParse({ ...record, irHash: "A".repeat(64) }).success).toBe(false);
  });

  it("accepts a placeholder policyHash, because a hand-written IR legitimately has one", () => {
    // The asymmetry between the two hash fields is deliberate and is half the
    // reason they are two fields. `irHash` is computed by the harness from bytes
    // it is holding, so it can always be a real digest and the strict regex
    // costs nothing. `policyHash` is copied verbatim out of whatever IR the page
    // loaded, and apps/eval/fixtures/minimal-ir.json carries the literal
    // "test-hash" -- constraining this field to 64 hex would force that fixture
    // to state a hash of a policy document that does not exist, i.e. make the
    // harness lie so its own schema would pass. `min(1)` is exactly what core's
    // PolicyIrSchema asks of the field (packages/core/src/policy/schema.ts:126),
    // so this is as strict as the value's own source and no stricter.
    expect(RunRecordSchema.safeParse({ ...record, policyHash: "test-hash" }).success).toBe(true);
    expect(RunRecordSchema.safeParse({ ...record, policyHash: "" }).success).toBe(false);
  });

  it("records the TierConfig that ran, so an arm name cannot stand in for what executed", () => {
    // `arm` is a free string the caller picks. Without the config beside it a
    // run with every tier switched off is byte-for-byte identical to a detector
    // that legitimately found nothing, and both read as "this arm scored zero".
    const { config: _omitted, ...withoutConfig } = record;
    expect(RunRecordSchema.safeParse(withoutConfig).success).toBe(false);
    // The three booleans are required, not optional: "tier1 was off" and "tier1
    // is unstated" are different claims, and only one of them can be scored.
    expect(
      RunRecordSchema.safeParse({ ...record, config: { tier0: true, tier1: false } }).success,
    ).toBe(false);
  });

  it("carries the tier-1 config whole, so no resolved field is dropped before Plan 8", () => {
    // Compared against resolveTier1Config's OWN output rather than a literal
    // written here. zod strips what a schema does not declare, so a schema
    // missing one of the six resolved fields would drop it without a word, and
    // an arm recorded without its threshold is an arm nobody can reproduce.
    const resolved = resolveTier1Config({ backend: "wasm" });
    const parsed = RunRecordSchema.safeParse({
      ...record,
      config: { tier0: true, tier1: true, tier2: false },
      tier1Config: resolved,
      tier1Stats: STATS,
    });
    expect(parsed.success).toBe(true);
    expect(parsed.success && parsed.data.tier1Config).toEqual(resolved);
  });

  it("couples tier1Config to config.tier1 in both directions", () => {
    // Tier 1 on with no config leaves the rung, the threshold and the label form
    // unwritten -- and TierConfig cannot hold them, so nothing else in the
    // record says which of the six rungs produced the numbers.
    expect(
      RunRecordSchema.safeParse({
        ...record,
        config: { tier0: true, tier1: true, tier2: false },
        tier1Stats: STATS,
      }).success,
    ).toBe(false);
    // The other direction is the worse mislabel: a tier-1 config on an arm whose
    // detect ran tier 0. Complete, valid, and describing a run that never
    // happened.
    expect(
      RunRecordSchema.safeParse({ ...record, tier1Config: resolveTier1Config({ backend: "wasm" }) })
        .success,
    ).toBe(false);
  });

  it("couples tier1Stats to config.tier1 AND to error, in every direction", () => {
    // The finding this field closes: `runArm` built every record from
    // `findings` and `timings` alone, so an item whose tail `maxLen` cut off
    // emitted a row byte-identical to one where the model read the whole
    // message and found nothing. Both have no findings and a real `tier1Ms`.
    const t1 = { tier0: false, tier1: true, tier2: false };
    const cfg = resolveTier1Config({ backend: "wasm" });

    // Tier 1 ran and returned: both halves required.
    expect(
      RunRecordSchema.safeParse({ ...record, config: t1, tier1Config: cfg, tier1Stats: STATS })
        .success,
    ).toBe(true);
    expect(
      RunRecordSchema.safeParse({ ...record, config: t1, tier1Config: cfg }).success,
    ).toBe(false);

    // Tier 1 off: counters would be describing a tagger that never existed.
    expect(RunRecordSchema.safeParse({ ...record, tier1Stats: STATS }).success).toBe(false);

    // Threw: `detect` throws whole, so the page's `lastDetect` still holds the
    // PREVIOUS item's delta and there is no per-item answer. Absent says that;
    // a row of zeros would assert nothing was lost on an item that may never
    // have finished. `degraded` goes absent on the same item and for the same
    // reason -- there is no result to read one off -- so it is dropped here
    // too rather than left standing as a clean-run claim on a crashed row.
    expect(
      RunRecordSchema.safeParse({
        ...record,
        config: t1,
        tier1Config: cfg,
        degraded: undefined,
        error: "detector exploded",
      }).success,
    ).toBe(true);
    expect(
      RunRecordSchema.safeParse({
        ...record,
        config: t1,
        tier1Config: cfg,
        tier1Stats: STATS,
        degraded: undefined,
        error: "detector exploded",
      }).success,
    ).toBe(false);
  });

  it("carries every counter Tier1TaggerStats declares, and refuses a partial one", () => {
    // Written as a delete-one-key sweep rather than six named cases, so a
    // counter added to the tagger and forwarded here is covered without this
    // test being edited -- and one dropped from the schema fails immediately.
    const t1 = { tier0: false, tier1: true, tier2: false };
    const cfg = resolveTier1Config({ backend: "wasm" });
    for (const key of Object.keys(STATS)) {
      const { [key]: _dropped, ...partial } = STATS as Record<string, number>;
      expect(
        RunRecordSchema.safeParse({ ...record, config: t1, tier1Config: cfg, tier1Stats: partial })
          .success,
      ).toBe(false);
    }
    // And the values survive the parse rather than being stripped: zod drops
    // what a schema does not declare, so a counter missing from the object
    // above would vanish from the file without a word.
    const parsed = RunRecordSchema.safeParse({
      ...record,
      config: t1,
      tier1Config: cfg,
      tier1Stats: { ...STATS, truncatedWords: 7, unmappableSpans: 2 },
    });
    expect(parsed.success && parsed.data.tier1Stats).toEqual({
      ...STATS,
      truncatedWords: 7,
      unmappableSpans: 2,
    });
  });

  it("requires abandonedWorkInFlight, so a contaminated latency cannot be unstated", () => {
    // A deadline expiry leaves the abandoned detection running in the browser,
    // and every later row is measured under contention with it -- while
    // carrying `error: null`, so a latency aggregate over non-errored rows
    // silently includes them. An ABSENT key would state nothing, which is
    // exactly the reading that makes the aggregate wrong.
    const { abandonedWorkInFlight: _omitted, ...without } = record;
    expect(RunRecordSchema.safeParse(without).success).toBe(false);
    expect(
      RunRecordSchema.safeParse({ ...record, abandonedWorkInFlight: true }).success,
    ).toBe(true);
    expect(
      RunRecordSchema.safeParse({ ...record, abandonedWorkInFlight: "yes" }).success,
    ).toBe(false);
  });

  it("refuses a record whose backend disagrees with itself", () => {
    // Three fields name a backend and they must be one answer. `backend` is the
    // arm label, `config.backend` is what detect was given, `tier1Config.backend`
    // is what the tagger was built with -- a record where they differ cannot be
    // attributed to any runtime at all.
    expect(
      RunRecordSchema.safeParse({ ...record, config: { ...record.config, backend: "webgpu" } })
        .success,
    ).toBe(false);
    expect(
      RunRecordSchema.safeParse({
        ...record,
        config: { tier0: true, tier1: true, tier2: false },
        tier1Config: resolveTier1Config({ backend: "webgpu" }),
        tier1Stats: STATS,
      }).success,
    ).toBe(false);
  });

  it("refuses a t1Model naming a different rung than the tier-1 config does", () => {
    expect(
      RunRecordSchema.safeParse({
        ...record,
        config: { tier0: true, tier1: true, tier2: false, t1Model: "gliner-pii-base" },
        tier1Config: resolveTier1Config({ backend: "wasm", modelId: "gliner-pii-edge" }),
        tier1Stats: STATS,
      }).success,
    ).toBe(false);
  });

  it("requires error to be present, so a crashed item cannot be written as a clean one", () => {
    // MEASURED: adding .optional() alongside .nullable() leaves all five
    // specified tests green. Nullable-but-required is the whole point: null
    // states "this item did not throw", whereas an absent key states nothing,
    // and an arm that crashes on part of the corpus must not be indistinguishable
    // from one that merely scored zero there.
    const { error: _omitted, ...withoutError } = record;
    expect(RunRecordSchema.safeParse(withoutError).success).toBe(false);
  });

  it("frames exactly one record per line and terminates the final line", () => {
    // MEASURED: replacing the join("\n") + "\n" body with join(",") leaves all
    // five specified tests green -- nothing referenced toJsonl at all. Plan 8
    // reads this file line by line, so framing IS the format: a missing final
    // newline drops the last record for a reader that requires terminated
    // lines, and any separator but "\n" loses every record but the first.
    const jsonl = toJsonl([record, { ...record, itemId: "b" }]);
    expect(jsonl.endsWith("\n")).toBe(true);
    const lines = jsonl.split("\n").filter((l) => l !== "");
    expect(lines).toHaveLength(2);
    expect(lines.map((l) => (JSON.parse(l) as { itemId: string }).itemId)).toEqual(["a", "b"]);
    // Round-trips through the schema, so the writer cannot emit a shape its own
    // reader rejects.
    expect(lines.every((l) => RunRecordSchema.safeParse(JSON.parse(l)).success)).toBe(true);
  });
});

/**
 * The offset UNIT, pinned by example. The schemas say offsets are UTF-16 code
 * units; this is what makes that statement fail loudly instead of being prose
 * nobody reruns. It is deliberately asserted against the shipped corpus rather
 * than a synthetic string, because the risk is that someone later "fixes" the
 * corpus to code-point offsets to make a Python reader work, which would break
 * the JS producer and core's own span-fidelity invariant instead.
 */
describe("offset encoding contract", () => {
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));

  it("keeps at least one item where the two offset units genuinely disagree", () => {
    // Guards the guard: if every astral character were dropped from the corpus,
    // the assertions below would still pass while testing nothing at all.
    const disagreeing = items.filter((i) =>
      i.gold.some((g) => [...i.text.slice(0, g.start)].length !== g.start),
    );
    expect(disagreeing.map((i) => i.id)).toContain("pos-emoji-before-pan");
  });

  it("stores UTF-16 code units, so a code-point reading of the same span is wrong", () => {
    const item = items.find((i) => i.id === "pos-emoji-before-pan");
    expect(item).toBeDefined();
    const span = item!.gold[0]!;

    // MEASURED with python3 on this corpus: `text[33:43]` returns "CPT1234H t"
    // while `text.encode("utf-16-le")[66:86].decode("utf-16-le")` returns
    // "ABCPT1234H". len(str) is 60 and len(utf-16-le)//2 is 62 -- the two astral
    // emoji cost one extra code unit each. Across the whole corpus a reader that
    // slices `str` directly gets 6 of 7 spans right and silently corrupts the
    // seventh, which is failure toward wrong numbers rather than a crash.
    expect([...item!.text.slice(0, span.start)].length).not.toBe(span.start);

    // UTF-16 indexing recovers the span ...
    expect(item!.text.slice(span.start, span.end)).toBe(span.text);
    // ... and code-point indexing (what Python's str does) does not. Spreading a
    // string iterates code points, so this is the same read Python performs.
    expect([...item!.text].slice(span.start, span.end).join("")).not.toBe(span.text);
  });
});

/**
 * findings[] negative tests -- the counterpart to the gold-side guards above.
 *
 * These exist because the two arrays are compared against each other and were
 * NOT validated to the same standard: every union here was originally widened
 * to z.string()/z.number(), so gold's action was a checked enum while a
 * finding's was free text. A scorer comparing a validated vocabulary against an
 * unvalidated one reads a typo as a miss rather than as a bug.
 */
describe("findings are validated to core's unions, like gold", () => {
  const base = {
    schemaVersion: 1 as const,
    runId: "r1",
    itemId: "a",
    policy: "p-fin",
    irHash: "0".repeat(64),
    policyHash: "1".repeat(64),
    arm: "t0",
    backend: "wasm" as const,
    detector: "core-orchestrator" as const,
    provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    text: "hello world",
    gold: [],
    timings: { tier0Ms: 0 },
    error: null,
    degraded: ABSENT_1_2,
    abandonedWorkInFlight: false,
  };
  const finding = {
    start: 0, end: 5, text: "hello", entityType: "in-pan",
    severity: "high", tier: 0, source: "pan-rule", confidence: 0.9, action: "block",
  };
  const withFinding = (patch: Record<string, unknown>) =>
    RunRecordSchema.safeParse({ ...base, findings: [{ ...finding, ...patch }] }).success;

  it("accepts a well-formed finding", () => {
    expect(withFinding({})).toBe(true);
  });

  it("rejects the whole bogus finding the widened schema used to accept", () => {
    // MEASURED against the widened version: this exact object validated
    // cleanly -- severity "banana", tier -7, confidence 42, action
    // "obliterate", empty entityType and source, all fine.
    expect(
      RunRecordSchema.safeParse({
        ...base,
        findings: [{ start: 0, end: 5, text: "hello", entityType: "", severity: "banana", tier: -7, source: "", confidence: 42, action: "obliterate" }],
      }).success,
    ).toBe(false);
  });

  it("rejects a severity outside core's Severity union", () => {
    expect(withFinding({ severity: "banana" })).toBe(false);
    expect(withFinding({ severity: "critical" })).toBe(true);
  });

  it("rejects a tier outside core's Tier union", () => {
    expect(withFinding({ tier: -7 })).toBe(false);
    expect(withFinding({ tier: 3 })).toBe(false);
    expect(withFinding({ tier: 2 })).toBe(true);
  });

  it("rejects a confidence outside 0..1", () => {
    // Finding.confidence is documented "0..1"; 42 is not a probability and a
    // scorer thresholding on it would silently admit everything.
    expect(withFinding({ confidence: 42 })).toBe(false);
    expect(withFinding({ confidence: -0.1 })).toBe(false);
    expect(withFinding({ confidence: 0 })).toBe(true);
    expect(withFinding({ confidence: 1 })).toBe(true);
  });

  it("rejects an action outside core's Action union, including gold's extra 'none'", () => {
    expect(withFinding({ action: "obliterate" })).toBe(false);
    // "none" is legal on a GOLD span but not on a resolved finding: core's
    // resolution chain always yields a real action. The two vocabularies are
    // deliberately different sizes, so this asymmetry is asserted rather than
    // assumed.
    expect(withFinding({ action: "none" })).toBe(false);
    expect(CorpusItemSchema.safeParse({ id: "a", text: "hello world", policy: "p-fin", gold: [{ start: 0, end: 5, text: "hello", entityType: "in-pan", action: "none" }] }).success).toBe(true);
  });

  it("rejects an empty entityType or source", () => {
    expect(withFinding({ entityType: "" })).toBe(false);
    expect(withFinding({ source: "" })).toBe(false);
  });
});

describe("the record's own text makes the cross-check executable", () => {
  const base = {
    schemaVersion: 1 as const,
    runId: "r1", itemId: "a", policy: "p-fin", irHash: "0".repeat(64), policyHash: "1".repeat(64),
    arm: "t0", backend: "wasm" as const, detector: "core-orchestrator" as const,
      provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    text: "hello world", findings: [], gold: [],
    timings: { tier0Ms: 0 }, error: null, degraded: ABSENT_1_2,
    abandonedWorkInFlight: false,
  };

  it("requires the message text, so a record can be verified without the corpus", () => {
    // Without this field a reader holds offsets and nothing to index into, and
    // the "exactly text.slice(start,end)" promise on every span is unverifiable
    // from a record alone -- which also made "scores standalone, without a
    // join" false.
    const { text: _omitted, ...withoutText } = base;
    expect(RunRecordSchema.safeParse(withoutText).success).toBe(false);
  });

  it("rejects a finding whose offsets do not hold its text in this record's message", () => {
    expect(RunRecordSchema.safeParse({ ...base, findings: [{ start: 0, end: 5, text: "WRONG", entityType: "in-pan", severity: "high", tier: 0, source: "r", confidence: 0.9, action: "block" }] }).success).toBe(false);
  });

  it("rejects a gold span that disagrees with the record's own text", () => {
    // The corpus already checked gold against the item's text; this catches the
    // copy going wrong -- a record built with one item's gold and another's
    // text, which a join would never notice.
    expect(RunRecordSchema.safeParse({ ...base, gold: [{ start: 0, end: 5, text: "WRONG", entityType: "in-pan", action: "block" }] }).success).toBe(false);
  });
});

describe("loadCorpus rejects duplicate ids", () => {
  it("names the line that repeats an id", () => {
    // Plan 8 joins records to items by itemId. A duplicate does not collide
    // loudly: one item is double-counted and the other silently dropped, which
    // shifts every aggregate by an invisible amount.
    const dup = '{"id":"a","text":"x","policy":"p","gold":[]}\n{"id":"b","text":"y","policy":"p","gold":[]}\n{"id":"a","text":"z","policy":"p","gold":[]}\n';
    expect(() => loadCorpus(dup)).toThrow(/line 3/);
    expect(() => loadCorpus(dup)).toThrow(/repeats id "a"/);
  });
});

describe("toJsonl survives the separators Python treats as newlines", () => {
  it("escapes U+2028 and U+2029 so a line-splitting reader sees one line", () => {
    // MEASURED: JSON.stringify leaves both raw (they are valid unescaped JSON
    // string content), and Python's str.splitlines() then splits on them --
    // a record containing one yields two fragments, both failing json.loads
    // with "Unterminated string".
    const record = {
      schemaVersion: 1 as const,
      runId: "r1", itemId: "a", policy: "p-fin", irHash: "0".repeat(64), policyHash: "1".repeat(64),
      arm: "t0", backend: "wasm" as const, detector: "core-orchestrator" as const,
      provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
      text: "before after end", findings: [], gold: [],
      timings: { tier0Ms: 0 }, error: null, degraded: ABSENT_1_2,
    abandonedWorkInFlight: false,
    };
    const jsonl = toJsonl([record]);
    expect(jsonl).not.toContain(" ");
    expect(jsonl).not.toContain(" ");
    // Still exactly one record, and still round-trips to the original string.
    const lines = jsonl.split("\n").filter((l) => l !== "");
    expect(lines).toHaveLength(1);
    expect((JSON.parse(lines[0]!) as { text: string }).text).toBe("before after end");
  });
});

/**
 * Degenerate findings spans.
 *
 * The hole was real: with neither `.min(1)` on a finding's text nor an explicit
 * `start < end`, all three of {start:999,end:1}, {start:5,end:5} and
 * {start:8,end:3} were ACCEPTED as findings (each with text "") while gold
 * rejected every one -- slice() clamps an inverted or zero-width range to ""
 * rather than throwing, and "" equals a claimed text of "".
 *
 * The two guards are REDUNDANT WITH EACH OTHER, which is worth stating because
 * the mutation results look wrong otherwise. Given the refine's other two
 * clauses: if start >= end then slice is "" so the text must be "", which
 * .min(1) refuses; and if the text is non-empty then start < end follows from
 * the slice comparison. MEASURED: removing either guard alone leaves all of
 * these tests green, and removing BOTH fails them. Both are kept anyway --
 * .min(1) so this schema and GoldSpanSchema read identically rather than
 * relying on a reader to derive the equivalence, and `start < end` so the
 * intent is stated instead of emerging from slice() semantics.
 */
describe("findings cannot carry a degenerate span", () => {
  const base = {
    schemaVersion: 1 as const,
    runId: "r1", itemId: "a", policy: "p-fin", irHash: "0".repeat(64), policyHash: "1".repeat(64),
    arm: "t0", backend: "wasm" as const, detector: "core-orchestrator" as const,
      provider: "claude",
    config: { tier0: true, tier1: false, tier2: false },
    text: "hello world", findings: [], gold: [],
    timings: { tier0Ms: 0 }, error: null, degraded: ABSENT_1_2,
    abandonedWorkInFlight: false,
  };
  const span = (start: number, end: number, text: string) => ({
    start, end, text, entityType: "in-pan",
    severity: "high", tier: 0, source: "pan-rule", confidence: 0.9, action: "block",
  });
  const accepts = (s: object) => RunRecordSchema.safeParse({ ...base, findings: [s] }).success;

  it("rejects an empty finding text, as gold already did", () => {
    expect(accepts(span(5, 5, ""))).toBe(false);
  });

  it("rejects inverted offsets, which slice() would otherwise clamp to nothing", () => {
    // The out-of-range start in the second case is the one worth noticing: it
    // never surfaces anywhere, because the clamped slice agrees before any
    // bounds check on `start` is reached.
    expect(accepts(span(8, 3, ""))).toBe(false);
    expect(accepts(span(999, 1, ""))).toBe(false);
  });

  it("still accepts a well-formed span, so the guards are not simply refusing everything", () => {
    expect(accepts(span(0, 5, "hello"))).toBe(true);
  });

  it("applies the same start < end rule to gold on a record", () => {
    expect(
      RunRecordSchema.safeParse({ ...base, gold: [{ start: 8, end: 3, text: "", entityType: "in-pan", action: "block" }] }).success,
    ).toBe(false);
  });
});

describe("corpus composition the README makes claims about", () => {
  const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));

  it("keeps a negative carrying proper nouns that are not clients", () => {
    // README.md's tier-1 precision paragraph rests entirely on this item: it is
    // the only negative containing capitalized non-client tokens, so without it
    // a tagger that fires on every proper noun scores a clean sweep on the
    // negatives and the README's claim becomes false silently.
    const item = items.find((i) => i.id === "neg-proper-nouns-not-clients");
    expect(item).toBeDefined();
    expect(item!.gold).toEqual([]);
    // Capitalized tokens that are not sentence-initial -- what a span tagger is
    // tempted by. Asserted as a count rather than by name so rewording the item
    // stays cheap while emptying it of proper nouns does not.
    const midSentenceCaps = item!.text.match(/(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+/g) ?? [];
    expect(midSentenceCaps.length).toBeGreaterThanOrEqual(3);
  });

  it("keeps the tier-1 gold values out of the IR's own examples", () => {
    // Contamination guard: minimal-ir.json lists examples per entityType, and a
    // model prompted from the IR would be handed any gold value that appears
    // there. Reads the IR rather than hardcoding "Globex", so adding an example
    // to the fixture that collides with the corpus fails here.
    const ir = JSON.parse(
      readFileSync(join(import.meta.dirname, "..", "fixtures", "minimal-ir.json"), "utf8"),
    ) as { entityTypes: Array<{ examples: string[] }> };
    const examples = new Set(ir.entityTypes.flatMap((e) => e.examples));
    const contaminated = items.flatMap((i) => i.gold.filter((g) => examples.has(g.text)).map((g) => `${i.id}:${g.text}`));
    expect(contaminated).toEqual([]);
  });
});

/**
 * The tier-2 evidence the JSONL was missing.
 *
 * Three fields, each closing a way a complete, schema-valid, perfectly
 * scoreable record describes a tier-2 run nobody can interpret:
 *
 * - `degraded` -- the orchestrator's own account of what this result is short
 *   of. It was being DROPPED: `runArm` projects `DetectionResult` field by
 *   field (`findings`, `timings`), so adding the array to `DetectionResult`
 *   produced no type error and no output change.
 * - `tier2Stats` -- the judge's counters for THIS item, as a delta, the tier-2
 *   twin of `tier1Stats`.
 * - `config.uncertainBelow` -- the escalation threshold. `TierConfig` calls it
 *   an EXPERIMENT variable the bake-off varies per arm, and a row that cannot
 *   say which value produced it cannot be compared with the row beside it.
 */
describe("the tier-2 evidence a record has to carry", () => {
  const base = {
    schemaVersion: 1 as const,
    runId: "r1",
    itemId: "a",
    policy: "p-fin",
    irHash: "0".repeat(64),
    policyHash: "1".repeat(64),
    arm: "t0+t2",
    backend: "webgpu" as const,
    detector: "core-orchestrator" as const,
    provider: "claude",
    text: "hello world",
    findings: [],
    gold: [],
    timings: { tier0Ms: 0.4, tier2Ms: 812 },
    error: null,
    abandonedWorkInFlight: false,
  };
  const T2 = { tier0: false, tier1: false, tier2: true, uncertainBelow: UNCERTAIN_BELOW };
  /** One engine call's row, as `JudgeCallRecord` declares it. */
  const CALL = { finishReason: "stop", promptTokens: 412, completionTokens: 96, ttftMs: 780 };
  /**
   * `JudgeStats` WHOLE, as a per-item delta. Written out rather than reduced to
   * the plan's eight counters: Tasks 6 and 7 added the segment accounting and
   * split the two caller-abort counters, and a record that carries a subset
   * silently decides which of them a scorer is allowed to have.
   */
  const STATS = {
    rung1: 3,
    rung2: 1,
    unresolvedQuotes: 2,
    unknownPredicates: 0,
    duplicatesDropped: 4,
    repairAttempts: 1,
    failedClosed: 0,
    truncatedResponses: 0,
    abortedResponses: 0,
    segmentsJudged: 2,
    segmentsSkipped: 0,
    deadlineExpiries: 0,
    callerAbortsMidGeneration: 0,
    callerAbortsWhileQueued: 0,
    calls: [CALL],
  };
  /**
   * The resolved engine settings a tier-2 record must carry, which the schema
   * couples to `config.tier2` the way `tier1Config` is coupled to
   * `config.tier1`. `DEFAULT_TIER2_CONFIG`'s four fields plus the page's
   * default per-call budget: what an arm loaded with no overrides runs under.
   */
  const T2_CONFIG = {
    modelId: "Qwen3.5-2B-q4f16_1-MLC",
    contextWindowSize: 8192,
    temperature: 0,
    maxTokens: 512,
    callBudgetMs: 60_000,
  };
  const t2 = (patch: Record<string, unknown> = {}) =>
    RunRecordSchema.safeParse({
      ...base,
      config: T2,
      degraded: [],
      tier2Stats: STATS,
      tier2Config: T2_CONFIG,
      ...patch,
    });

  it("accepts a tier-2 record carrying all three, and strips none of it", () => {
    const parsed = t2();
    expect(parsed.success, parsed.success ? "" : JSON.stringify(parsed.error.issues)).toBe(true);
    // Deep-equal rather than a spot check: zod DROPS what a schema does not
    // declare, so a counter missing from the object would vanish from the file
    // without a word and every assertion on the fields that survived would
    // still pass.
    expect(parsed.success && parsed.data.tier2Stats).toEqual(STATS);
    expect(parsed.success && parsed.data.config.uncertainBelow).toBe(UNCERTAIN_BELOW);
  });

  it("couples tier2Stats to config.tier2 AND to error, in every direction", () => {
    // The same coupling `tier1Stats` uses, and for the same reason: the page's
    // `lastDetect` is a DELTA over the judge's cumulative counters, so on a
    // thrown or timed-out item the delta belongs to the PREVIOUS item and
    // absent is the only honest answer. A row of zeros would assert that
    // nothing failed closed on an item whose judge may never have returned.
    expect(t2().success).toBe(true);
    expect(t2({ tier2Stats: undefined }).success).toBe(false);
    expect(t2({ error: "the engine went away", tier2Stats: undefined, degraded: undefined }).success).toBe(true);
    expect(t2({ error: "the engine went away", degraded: undefined }).success).toBe(false);
    // Tier 2 off: counters describing a judge that never existed.
    expect(
      RunRecordSchema.safeParse({
        ...base,
        config: { tier0: true, tier1: false, tier2: false },
        degraded: [],
        tier2Stats: STATS,
      }).success,
    ).toBe(false);
  });

  it("refuses a partial counter set, so a dropped counter cannot go unnoticed", () => {
    // A delete-one-key sweep rather than fifteen named cases, so a counter
    // added to `JudgeStats` and forwarded here is covered without editing this
    // test -- and one dropped from the schema fails immediately.
    for (const key of Object.keys(STATS)) {
      const { [key]: _dropped, ...partial } = STATS as Record<string, unknown>;
      expect(t2({ tier2Stats: partial }).success, `${key} was allowed to be missing`).toBe(false);
    }
  });

  it("rejects a negative counter on every counter, not just the first one", () => {
    for (const key of Object.keys(STATS).filter((k) => k !== "calls")) {
      expect(t2({ tier2Stats: { ...STATS, [key]: -1 } }).success, `${key} accepted -1`).toBe(false);
    }
    // ... and a fractional one. These are event counts; 1.5 segments judged is
    // a corrupted delta, not a measurement.
    expect(t2({ tier2Stats: { ...STATS, segmentsJudged: 1.5 } }).success).toBe(false);
  });

  it("carries ONE ROW PER ENGINE CALL rather than one finishReason per message", () => {
    // The aggregation decision, pinned. A message makes one call per selected
    // segment (plus the one repair retry), and Task 6 kept per-call rows
    // precisely because a single `finishReason` for a message would be a fact
    // about one call presented as a fact about the message. Two rows that
    // DISAGREE is the case a per-message field cannot express at all: this
    // message had one call stop cleanly and one hit the token ceiling, so its
    // judgement is partial, and a scorer that saw only "stop" would score it as
    // complete.
    const parsed = t2({
      tier2Stats: {
        ...STATS,
        calls: [
          CALL,
          { finishReason: "length", promptTokens: 1180, completionTokens: 512, ttftMs: 940 },
        ],
      },
    });
    expect(parsed.success).toBe(true);
    expect(parsed.success && parsed.data.tier2Stats?.calls.map((c) => c.finishReason)).toEqual([
      "stop",
      "length",
    ]);
    // Order is the order the calls were made in, so a row can be attributed to
    // the segment it came from by position; a Set or a count could not.
    expect(parsed.success && parsed.data.tier2Stats?.calls[1]?.promptTokens).toBe(1180);
  });

  it("rejects a finishReason the pinned engine cannot report", () => {
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, finishReason: "banana" }] } }).success).toBe(false);
    // Every word 0.2.84's ChatCompletionFinishReason actually has.
    for (const reason of ["stop", "length", "tool_calls", "abort"]) {
      expect(
        t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, finishReason: reason }] } }).success,
        `${reason} was refused`,
      ).toBe(true);
    }
  });

  it("accepts a call row the engine reported no usage for", () => {
    // `JudgeCallRecord` types all FIVE fields `| undefined`, so a schema that
    // required them would refuse a real call rather than record what it knew.
    // `usage` is optional on the completion, which covers four of them;
    // `finishReason` is undefined-able for a different reason, and NOT because
    // the field can be null -- 0.2.84 declares the non-streaming
    // `ChatCompletion.Choice.finish_reason` required and non-nullable, and the
    // bundle assigns it from `getFinishReason()`, which is declared
    // `| undefined`. See Tier2CallSchema.
    //
    // This is a SCHEMA test and it cannot see the other half: a producer that
    // filled the blanks in before the schema ran would pass it unchanged.
    // test/run.spec.ts pins that half against `runArm`.
    expect(t2({ tier2Stats: { ...STATS, calls: [{}] } }).success).toBe(true);
  });

  it("carries a non-finite time-to-first-token as null, and refuses a raw NaN", () => {
    // MEASURED with zod 4.4.3: `z.number()` rejects both NaN and Infinity, and
    // `JSON.stringify(NaN)` is the string "null". So a NaN copied straight onto
    // a record produces a FILE ITS OWN READER REFUSES -- written as null,
    // rejected on the way back in. `ttftMs` is the field where that is live:
    // `JudgeCallRecord` says the conversion out of
    // `usage.extra.time_to_first_token_s` is deliberately unguarded, because a
    // NaN there is a fact about the call. Null is that fact, spelled so it
    // survives the round trip; `runArm` is what maps it.
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, ttftMs: null }] } }).success).toBe(true);
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, ttftMs: Number.NaN }] } }).success).toBe(false);
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, ttftMs: Infinity }] } }).success).toBe(false);
    // Not nullable on the counts: `usage.prompt_tokens` is either a number the
    // engine reported or absent, and null there would be a third state nothing
    // produces.
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, promptTokens: null }] } }).success).toBe(false);
  });

  it("carries the engine's decode rate, and treats a poisoned one the way ttft is treated", () => {
    // The field the bake-off's `minDecodeTokPerSec` gate is computed from.
    // Nothing else on a record can answer it: `timings.tier2Ms` covers the
    // whole message and no per-call elapsed time is recorded, so a derived rate
    // would charge the judge's prompt assembly, JSON parse and span ladder to
    // the model -- on a FLOOR gate, which kills capable arms.
    //
    // VERIFIED in the installed 0.2.84 bundle: `decode_tokens_per_s` is
    // `completion_tokens / decode_time`, a plain division, so a call
    // interrupted before its first token leaves 0/0. Same round-trip hazard as
    // `ttftMs` and therefore the same three answers.
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, decodeTokPerSec: 31.4 }] } }).success).toBe(true);
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, decodeTokPerSec: null }] } }).success).toBe(true);
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, decodeTokPerSec: Number.NaN }] } }).success).toBe(false);
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, decodeTokPerSec: Infinity }] } }).success).toBe(false);
    // A rate is not a count, so it is not an integer -- and it must not be
    // negative, which no division of two non-negative quantities produces.
    expect(t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, decodeTokPerSec: -1 }] } }).success).toBe(false);
    const parsed = t2({ tier2Stats: { ...STATS, calls: [{ ...CALL, decodeTokPerSec: 31.4 }] } });
    expect(parsed.success && parsed.data.tier2Stats?.calls[0]?.decodeTokPerSec).toBe(31.4);
  });

  it("couples degraded to error, so a thrown item cannot claim a clean run", () => {
    // `DetectionResult.degraded` is REQUIRED there and its doc says why: "[] is
    // a positive claim; undefined would be silence". The record keeps that,
    // except on a thrown item -- `detect` throws whole, so there is no result
    // to read an array off, and [] would be the positive claim that nothing was
    // skipped on an item that never finished.
    expect(t2({ degraded: undefined }).success).toBe(false);
    expect(t2({ error: "boom", tier2Stats: undefined, degraded: undefined }).success).toBe(true);
    expect(t2({ error: "boom", tier2Stats: undefined }).success).toBe(false);
    // A tier-0 arm carries it too: `absent` entries are the only thing that
    // says WHICH tiers a `tier0Ms` of 0 belongs to.
    expect(
      RunRecordSchema.safeParse({
        ...base,
        config: { tier0: true, tier1: false, tier2: false },
        timings: { tier0Ms: 0.4 },
        degraded: [
          { tier: 1, reason: "absent", detail: "tier 1 was not enabled in this TierConfig" },
          { tier: 2, reason: "absent", detail: "tier 2 was not enabled in this TierConfig" },
        ],
      }).success,
    ).toBe(true);
  });

  it("carries every reason word core can emit, including the three no judge counts", () => {
    // `absent`, `scope-unjudged` and the budget-spent-before-start form of
    // `budget-exhausted` are the orchestrator's own facts. No judge counter
    // corresponds to any of them, and the last one makes NO ENGINE CALL AT ALL,
    // so there is nothing for `tier2Stats` to have counted. Without these words
    // on the record they are unreachable from the file.
    for (const reason of [
      "failed-closed",
      "call-budget-exhausted",
      "budget-exhausted",
      "absent",
      "scope-unjudged",
    ]) {
      expect(
        t2({ degraded: [{ tier: 2, reason, detail: "why" }] }).success,
        `${reason} was refused`,
      ).toBe(true);
    }
    expect(t2({ degraded: [{ tier: 2, reason: "vibes", detail: "why" }] }).success).toBe(false);
  });

  it("refuses a notice with no detail and one naming a tier outside core's union", () => {
    // `stampEngineNotice` throws on an empty detail, calling it "the entire
    // human-readable payload of a notice"; the record holds that line rather
    // than accepting what core refuses to produce.
    expect(t2({ degraded: [{ tier: 2, reason: "failed-closed", detail: "" }] }).success).toBe(false);
    expect(t2({ degraded: [{ tier: 3, reason: "failed-closed", detail: "why" }] }).success).toBe(false);
  });

  it("requires the escalation threshold on a tier-2 record, and keeps the value given", () => {
    // Two DIFFERENT values, because a test that only ever exercises the default
    // cannot tell "carries the config" from "hardcodes UNCERTAIN_BELOW" -- the
    // trap this plan has hit twice.
    const { uncertainBelow: _dropped, ...noThreshold } = T2;
    expect(t2({ config: noThreshold }).success).toBe(false);
    for (const value of [0, 0.35, UNCERTAIN_BELOW, 1]) {
      const parsed = t2({ config: { ...T2, uncertainBelow: value } });
      expect(parsed.success, `${String(value)} was refused`).toBe(true);
      expect(parsed.success && parsed.data.config.uncertainBelow).toBe(value);
    }
    // A tier-0 or tier-1 arm never reaches escalation, so it is not asked for
    // one -- the same asymmetry `tier1Config` has.
    expect(
      RunRecordSchema.safeParse({
        ...base,
        config: { tier0: true, tier1: false, tier2: false },
        timings: { tier0Ms: 0.4 },
        degraded: [],
      }).success,
    ).toBe(true);
    // ... and it is not FORBIDDEN one either, which is the direction the refine
    // deliberately leaves open and the direction nothing used to exercise.
    // The review that found this measured a tightening of the refine to a
    // two-way coupling (`(uncertainBelow !== undefined) === config.tier2`)
    // passing the whole suite. It is reachable -- `runArm` builds `{...spec.config, backend}`, so
    // a caller who puts a threshold in a tier-0 arm's TierConfig gets it onto
    // the record -- and under the tightened rule `runBakeoff` would throw on
    // row 1 and discard the arm's whole file AFTER its GPU time was spent. The
    // record's job is to state what `detect` received, not to tidy it away.
    expect(
      RunRecordSchema.safeParse({
        ...base,
        config: { tier0: true, tier1: false, tier2: false, uncertainBelow: 0.35 },
        timings: { tier0Ms: 0.4 },
        degraded: [],
      }).success,
    ).toBe(true);
  });

  it("rejects a threshold escalate.ts would throw on", () => {
    // Restates `uncertainSegmentStarts`' own bound: a finite number in [0, 1].
    // NaN is the value worth naming -- `confidence < NaN` is false for every
    // finding, so an unvalidated NaN switches the uncertainty branch off and
    // every message reports as having nothing uncertain, which is a wrong
    // answer that looks like a right one.
    for (const bad of [-0.1, 1.1, Number.NaN, Infinity]) {
      expect(t2({ config: { ...T2, uncertainBelow: bad } }).success, `${String(bad)} accepted`).toBe(false);
    }
  });

  it("couples tier2Config to config.tier2 in both directions", () => {
    // WHY THE FIELD EXISTS. `TierConfig` carries `t2Model` and nothing else
    // about tier 2, so two arms differing only in their context window, their
    // token ceiling or their per-call budget used to emit records that were
    // byte-identical in every field a scorer can group by -- the same disease
    // `tier1Config` was added to cure one tier down, and the reason Plan 5's
    // own fallback for a model that cannot take 8,192 ("run that arm at 4,096
    // and report the asymmetry") was unexpressible in the output.
    //
    // Both directions, exactly as `tier1Config`: tier 2 on with no config is an
    // arm whose settings went unrecorded, and a config with tier 2 off is a
    // lower-tier run wearing a tier-2 label.
    expect(t2({ tier2Config: undefined }).success).toBe(false);
    expect(
      RunRecordSchema.safeParse({
        ...base,
        config: { tier0: true, tier1: false, tier2: false },
        degraded: [],
        tier2Config: T2_CONFIG,
      }).success,
    ).toBe(false);
    // Unlike `tier2Stats`, it is NOT coupled to `error`: this describes the
    // engine the page HOLDS, which is known before the item runs and stays true
    // whatever the item does.
    expect(t2({ error: "detector exploded", tier2Stats: undefined, degraded: undefined }).success).toBe(true);
    // And it must name the same model `config.t2Model` does, when both are
    // present -- the `t1Model`/`tier1Config` check one tier up.
    expect(
      t2({ config: { ...T2, t2Model: "Phi-4-mini-instruct-q4f16_1-MLC" } }).success,
    ).toBe(false);
    expect(t2({ config: { ...T2, t2Model: T2_CONFIG.modelId } }).success).toBe(true);
  });

  it("refuses tier-2 settings that could not have run", () => {
    // Each bound restates the validator that would have refused the value
    // upstream, so a record cannot state a configuration `resolveTier2Config`
    // or `WebLlmJudge`'s constructor would have thrown on.
    expect(t2({ tier2Config: { ...T2_CONFIG, contextWindowSize: 0 } }).success).toBe(false);
    expect(t2({ tier2Config: { ...T2_CONFIG, contextWindowSize: 8192.5 } }).success).toBe(false);
    expect(t2({ tier2Config: { ...T2_CONFIG, maxTokens: 0 } }).success).toBe(false);
    // A per-call budget of 0 is a deadline that fires immediately, and one
    // above 2^31-1 becomes a ~1 ms one in `setTimeout` rather than a longer
    // one; `cancel.ts` refuses both with the same bound.
    expect(t2({ tier2Config: { ...T2_CONFIG, callBudgetMs: 0 } }).success).toBe(false);
    expect(t2({ tier2Config: { ...T2_CONFIG, callBudgetMs: 2_147_483_648 } }).success).toBe(false);
    expect(t2({ tier2Config: { ...T2_CONFIG, callBudgetMs: 2_147_483_647 } }).success).toBe(true);
    expect(t2({ tier2Config: { ...T2_CONFIG, modelId: "" } }).success).toBe(false);
  });

  it("survives the JSONL round trip with every tier-2 field intact", () => {
    // The file is the deliverable, not the in-memory object. `undefined` inside
    // a call row is dropped by JSON.stringify, so the row comes back with keys
    // MISSING rather than present-and-undefined -- which is why the row fields
    // are `.optional()` and not `.nullable()`.
    const record = {
      ...base,
      config: { ...T2, uncertainBelow: 0.35 },
      degraded: [{ tier: 2 as const, reason: "failed-closed" as const, detail: "unparseable after one repair retry" }],
      tier2Stats: { ...STATS, calls: [CALL, { finishReason: "abort" }] },
      // A NON-DEFAULT window and budget, deliberately: this is the field that
      // makes "run that arm at 4,096 and report the asymmetry" expressible, and
      // a fixture carrying only the defaults could not tell a record that
      // states its settings from one that restates a constant.
      tier2Config: { ...T2_CONFIG, contextWindowSize: 4096, callBudgetMs: 30_000 },
    } as RunRecord;
    const [line] = toJsonl([record]).trim().split("\n");
    const reparsed = RunRecordSchema.safeParse(JSON.parse(line!));
    expect(reparsed.success, reparsed.success ? "" : JSON.stringify(reparsed.error.issues)).toBe(true);
    expect(reparsed.success && reparsed.data.tier2Stats?.calls).toEqual([
      CALL,
      { finishReason: "abort" },
    ]);
    expect(reparsed.success && reparsed.data.degraded).toEqual(record.degraded);
    expect(reparsed.success && reparsed.data.config.uncertainBelow).toBe(0.35);
    expect(reparsed.success && reparsed.data.tier2Config).toEqual({
      modelId: "Qwen3.5-2B-q4f16_1-MLC",
      contextWindowSize: 4096,
      temperature: 0,
      maxTokens: 512,
      callBudgetMs: 30_000,
    });
  });
});

/**
 * The discrimination this whole channel exists for, demonstrated against the
 * REAL orchestrator rather than against notices written by hand.
 *
 * `DetectionResult.degraded`'s own doc states the claim: "a tier-2 arm that
 * failed closed on 40% of its messages is distinguishable from one that found
 * nothing". Until this task the claim stopped at core's return value -- the
 * record dropped the array, so the file Plan 8 scores could not make the
 * distinction the type was added for. What follows is the scorer code, run.
 */
describe("a scorer separates 'failed closed on 40%' from 'found nothing'", () => {
  const ir = loadPolicyIr(readFileSync(IR_FIXTURE, "utf8"));

  /**
   * Five messages, each carrying a tier-0 PAN.
   *
   * minimal-ir.json declares NO semanticPredicates, so the uncertainty half of
   * `selectSegments` is the only thing that can put a judge in the loop, and it
   * needs a prior finding under the threshold. `uncertainBelow: 1` -- "escalate
   * everything short of total certainty", which `uncertainSegmentStarts`
   * documents as legal -- is what makes tier 0's own findings qualify. The
   * `judged` count asserted below is what proves this actually happened rather
   * than the whole comparison passing vacuously with no judge call at all.
   */
  const MESSAGES = [
    "Please file ABCPT1234H against the renewal.",
    "The PAN on record is ABCPT1234H, confirm with finance.",
    "Second reminder: ABCPT1234H is still unverified.",
    "Attach ABCPT1234H to the onboarding packet.",
    "ABCPT1234H was quoted in yesterday's thread.",
  ];
  const FAILED_ON = new Set([MESSAGES[1], MESSAGES[3]]);

  /** A judge that finds nothing, and on some messages reports failing closed. */
  function stubJudge(failsClosed: (text: string) => boolean): {
    judge: SemanticJudge;
    judged: () => number;
  } {
    let calls = 0;
    return {
      judged: () => calls,
      judge: {
        judge: (request) => {
          calls += 1;
          return Promise.resolve({
            findings: [],
            scopesJudged: [],
            degraded: failsClosed(request.text)
              ? [
                  {
                    reason: "failed-closed" as const,
                    detail: "the response was still unparseable after one repair retry",
                  },
                ]
              : [],
          });
        },
      },
    };
  }

  async function runArmOverMessages(
    failsClosed: (text: string) => boolean,
  ): Promise<{ results: DetectionResult[]; judged: number }> {
    const stub = stubJudge(failsClosed);
    const results: DetectionResult[] = [];
    for (const text of MESSAGES) {
      results.push(
        await detect({
          ir,
          provider: "claude",
          text,
          config: { tier0: true, tier1: false, tier2: true, uncertainBelow: 1 },
          engines: { tier2: stub.judge },
        }),
      );
    }
    return { results, judged: stub.judged() };
  }

  /**
   * The same zeroed counters on BOTH arms, and that is the point rather than a
   * convenience: it makes `tier2Stats` incapable of being what separates them,
   * so whatever the scorer below reads has to be coming from `degraded`.
   */
  const ZEROED = {
    rung1: 0, rung2: 0, unresolvedQuotes: 0, unknownPredicates: 0, duplicatesDropped: 0,
    repairAttempts: 0, failedClosed: 0, truncatedResponses: 0, abortedResponses: 0,
    segmentsJudged: 0, segmentsSkipped: 0, deadlineExpiries: 0,
    callerAbortsMidGeneration: 0, callerAbortsWhileQueued: 0, calls: [],
  };

  const asRecords = (results: DetectionResult[]): RunRecord[] =>
    results.map(
      (result, i) =>
        ({
          schemaVersion: 1,
          runId: "demo",
          itemId: `m${String(i)}`,
          policy: "minimal-fixture",
          irHash: "0".repeat(64),
          policyHash: "test-hash",
          arm: "t0+t2",
          backend: "webgpu",
          detector: "core-orchestrator",
          provider: "claude",
          config: { tier0: true, tier1: false, tier2: true, backend: "webgpu", uncertainBelow: 1 },
          text: MESSAGES[i]!,
          findings: result.findings.map((f) => ({
            start: f.start, end: f.end, text: f.text, entityType: f.entityType,
            severity: f.severity, tier: f.tier, source: f.source,
            confidence: f.confidence, action: f.action,
          })),
          gold: [],
          timings: result.timings,
          degraded: result.degraded,
          tier2Stats: ZEROED,
          // Required exactly when `config.tier2` is set. The stub judge here is
          // not a WebLlmJudge and loads no engine, so this states the arm these
          // records would have been written under rather than anything measured
          // -- which is what makes them the shape Plan 8 will actually hold.
          tier2Config: {
            modelId: "Qwen3.5-2B-q4f16_1-MLC",
            contextWindowSize: 8192,
            temperature: 0,
            maxTokens: 512,
            callBudgetMs: 60_000,
          },
          error: null,
          abandonedWorkInFlight: false,
        }) as RunRecord,
    );

  /**
   * THE SCORER. Per MESSAGE, not per segment -- `JudgeStats.failedClosed`
   * counts segments and cannot answer "on what fraction of messages", which is
   * the number the bake-off table reports.
   */
  /** Records as Plan 8 will hold them: written as JSONL and parsed back. */
  const throughTheFile = (results: DetectionResult[]): RunRecord[] =>
    toJsonl(asRecords(results))
      .trim()
      .split("\n")
      .map((line) => RunRecordSchema.parse(JSON.parse(line)));

  const failedClosedMessageRate = (records: readonly RunRecord[]): number => {
    const scoreable = records.filter((r) => r.error === null);
    const closed = scoreable.filter((r) =>
      (r.degraded ?? []).some((d) => d.tier === 2 && d.reason === "failed-closed"),
    );
    return closed.length / scoreable.length;
  };

  it("reads 0.4 and 0 off two arms that are otherwise the same file", async () => {
    const closed = await runArmOverMessages((text) => FAILED_ON.has(text));
    const quiet = await runArmOverMessages(() => false);

    // The judge really was in the loop on every message of both arms. Without
    // this the two arms could agree because neither ever reached tier 2.
    expect(closed.judged).toBe(MESSAGES.length);
    expect(quiet.judged).toBe(MESSAGES.length);

    // Scored off the FILE, not off the objects built above -- which is the only
    // version of this test that can fail. zod strips what a schema does not
    // declare, so a scorer reading `degraded` off a hand-built record reads it
    // whether or not the schema carries it, and this whole comparison passes
    // against the very schema the task exists to fix. MEASURED: before
    // `degraded` was declared, scoring the objects gave 0.4 and 0 exactly as
    // below, while the round trip below gives an empty array on both arms.
    const failing = throughTheFile(closed.results);
    const finding = throughTheFile(quiet.results);

    // Neither arm found anything: `findings` is identical, so recall, precision
    // and every span-derived number are identical too.
    expect(failing.every((r) => r.findings.length === finding[0]!.findings.length)).toBe(true);

    // And EVERY OTHER FIELD is identical as well. `timings` is normalised
    // because tier2Ms is a wall clock and differs by microseconds between two
    // runs of the same code; nothing else is touched. This is the defect
    // restated as an assertion: strip `degraded` -- which is exactly what
    // `runArm` used to do by projecting `findings` and `timings` alone -- and
    // the two arms are the same file.
    const withoutDegraded = (r: RunRecord) => {
      const { degraded: _dropped, timings: _clock, ...rest } = r;
      return rest;
    };
    expect(failing.map(withoutDegraded)).toEqual(finding.map(withoutDegraded));

    // With it, they are two different results.
    expect(failedClosedMessageRate(failing)).toBeCloseTo(0.4, 10);
    expect(failedClosedMessageRate(finding)).toBe(0);
  });

  it("says a message blew its latency budget, which no judge counter reports", async () => {
    // The word with no counter behind it anywhere. The orchestrator records
    // `budget-exhausted` FROM ITS OWN TIMER, not from what the judge returned:
    // a judge that ignores the abort signal and answers in full still ran past
    // the budget. Both arms below run the SAME verdict through the SAME code,
    // so a judge's counters would be identical to the field; only `degraded`
    // separates them.
    const tight = loadPolicyIr(
      JSON.stringify({ ...JSON.parse(readFileSync(IR_FIXTURE, "utf8")), latencyBudgetMs: 100 }),
    );
    const verdict = { findings: [], scopesJudged: [] };
    const answerAfter = (ms: number): SemanticJudge => ({
      judge: async () => {
        await new Promise((resolve) => setTimeout(resolve, ms));
        return verdict;
      },
    });
    const run = (judge: SemanticJudge) =>
      detect({
        ir: tight,
        provider: "claude",
        text: MESSAGES[0]!,
        config: { tier0: true, tier1: false, tier2: true, uncertainBelow: 1 },
        engines: { tier2: judge },
      });

    const slow = await run(answerAfter(400));
    const fast = await run(answerAfter(0));

    expect(slow.findings).toEqual(fast.findings);
    expect(slow.degraded.filter((d) => d.reason === "budget-exhausted")).toHaveLength(1);
    expect(fast.degraded.filter((d) => d.reason === "budget-exhausted")).toHaveLength(0);
    // And the word survives to the FILE, which is where a scorer meets it.
    const [written] = throughTheFile([slow]);
    expect(written!.degraded?.map((d) => d.reason)).toContain("budget-exhausted");
    expect(throughTheFile([fast])[0]!.degraded?.map((d) => d.reason)).not.toContain(
      "budget-exhausted",
    );
  });
});

/**
 * The Approach-B evidence a record has to carry, and the four couplings that
 * decide which arm a row belongs to.
 *
 * Every one of these was a SURVIVING MUTANT before this block existed. The
 * schema had `detector` and `baselineStats` and nothing in vitest ever built a
 * B row, so all four refines could be replaced by `() => true` with the whole
 * suite green -- which is the same shape of hole `tier1Stats` and `tier2Stats`
 * were found in, one arm over.
 *
 * What each coupling prevents, in the order they appear below:
 *
 *   - `detector` REQUIRED with no default. A default of "core-orchestrator"
 *     would let a B row be written as a compiled one by omission.
 *   - `baselineStats` present exactly on a returned B row. Absent, B's counters
 *     have nowhere to go but `tier2Stats`, where `messagesJudged` would be
 *     written under a field named for segments.
 *   - `tier2Stats` present exactly on a returned COMPILED row. A B row also has
 *     `config.tier2` true -- a model read the message -- so without `detector`
 *     in that refine the two arms are indistinguishable to the schema.
 *   - no `uncertainBelow` on a B row. B never escalates, and `gateReport`
 *     compares that number against the threshold the planned distribution was
 *     measured at, which would make the two agree about work no B arm performs.
 */
describe("the Approach-B evidence a record has to carry", () => {
  const base = {
    schemaVersion: 1 as const,
    runId: "r1",
    itemId: "a",
    policy: "p-fin",
    irHash: "0".repeat(64),
    policyHash: "1".repeat(64),
    arm: "baselineB-Qwen3.5-2B-q4f16_1-MLC",
    backend: "webgpu" as const,
    detector: "approach-b" as const,
    provider: "claude",
    // No `uncertainBelow`: this arm does not escalate. Tier 0 is ON, so this is
    // the `baseline-b-tier0` shape -- the arm that separates "compiling helps"
    // from "having patterns helps".
    config: { tier0: true, tier1: false, tier2: true, t2Model: "Qwen3.5-2B-q4f16_1-MLC" },
    tier2Config: {
      modelId: "Qwen3.5-2B-q4f16_1-MLC",
      contextWindowSize: 8192,
      temperature: 0,
      maxTokens: 512,
      callBudgetMs: 60_000,
    },
    text: "hello world",
    findings: [],
    gold: [],
    timings: { tier0Ms: 0.4, tier2Ms: 3612 },
    degraded: [],
    error: null,
    abandonedWorkInFlight: false,
  };
  /**
   * `BaselineStats` WHOLE as a per-item delta, every counter at a DISTINCT
   * value so a field projected into the wrong slot is visible. The three that
   * are not the judge's carry the three largest numbers, for the same reason.
   */
  const B_STATS = {
    rung1: 3,
    rung2: 1,
    unresolvedQuotes: 2,
    unknownEntityTypes: 41,
    duplicatesDropped: 4,
    repairAttempts: 5,
    failedClosed: 6,
    truncatedResponses: 7,
    abortedResponses: 8,
    messagesJudged: 43,
    deadlineExpiries: 9,
    messageBudgetExpiries: 47,
    callerAbortsMidGeneration: 10,
    callerAbortsWhileQueued: 11,
    calls: [{ finishReason: "stop", promptTokens: 1433, completionTokens: 36, ttftMs: 2711 }],
  };
  const b = (patch: Record<string, unknown> = {}) =>
    RunRecordSchema.safeParse({ ...base, baselineStats: B_STATS, ...patch });

  it("accepts an Approach-B row and strips none of its counters", () => {
    const parsed = b();
    expect(parsed.success, parsed.success ? "" : JSON.stringify(parsed.error.issues)).toBe(true);
    // Deep-equal rather than a spot check: zod DROPS what a schema does not
    // declare, so a counter missing from `BaselineStatsSchema` would vanish
    // from the file without a word.
    expect(parsed.success && parsed.data.baselineStats).toEqual(B_STATS);
  });

  it("requires detector, and does not fill it in", () => {
    // The one unconditionally required field this schema has gained, and the
    // mutation worth naming is `.default("core-orchestrator")` rather than
    // deletion: a default would silently turn a row that forgot to say what ran
    // into a compiled row, which is the substitution the field exists to stop.
    //
    // The row below is COMPILED-SHAPED on purpose. Dropping `detector` from a
    // B-shaped row is already refused by the `baselineStats` coupling, so it
    // cannot tell "required" from "defaulted"; this one is valid under a
    // default and invalid without one, which is the only shape that can.
    const { detector: _dropped, ...without } = base;
    const compiledShaped = {
      ...without,
      config: { ...base.config, uncertainBelow: 0.8 },
      tier2Stats: {
        rung1: 0,
        rung2: 0,
        unresolvedQuotes: 0,
        unknownPredicates: 0,
        duplicatesDropped: 0,
        repairAttempts: 0,
        failedClosed: 0,
        truncatedResponses: 0,
        abortedResponses: 0,
        segmentsJudged: 0,
        segmentsSkipped: 0,
        deadlineExpiries: 0,
        callerAbortsMidGeneration: 0,
        callerAbortsWhileQueued: 0,
        calls: [],
      },
    };
    expect(RunRecordSchema.safeParse(compiledShaped).success).toBe(false);
    // The same object WITH the field is accepted, so the refusal above is about
    // the field being absent and not about anything else in the fixture.
    expect(
      RunRecordSchema.safeParse({ ...compiledShaped, detector: "core-orchestrator" }).success,
    ).toBe(true);
    expect(RunRecordSchema.safeParse({ ...without, baselineStats: B_STATS }).success).toBe(false);
    expect(b({ detector: "the-compiler" }).success).toBe(false);
  });

  it("requires baselineStats on a returned B row and forbids them anywhere else", () => {
    expect(b({ baselineStats: undefined }).success).toBe(false);
    // Absent on an ERRORED item, exactly as `tier2Stats` is: the page's
    // `lastDetect` is a delta over cumulative counters, so on a thrown item it
    // belongs to the PREVIOUS item and there is no per-item answer to give.
    expect(b({ error: "boom", degraded: undefined, baselineStats: undefined }).success).toBe(true);
    expect(b({ error: "boom", degraded: undefined }).success).toBe(false);
    // And a COMPILED row must not carry them.
    expect(
      RunRecordSchema.safeParse({
        ...base,
        detector: "core-orchestrator",
        config: { ...base.config, uncertainBelow: 0.8 },
        baselineStats: B_STATS,
        tier2Stats: undefined,
      }).success,
    ).toBe(false);
  });

  it("forbids tier2Stats on a B row, which config.tier2 alone cannot express", () => {
    // The mutation this catches: dropping `detector` from the `tier2Stats`
    // refine. A B row has `config.tier2` true -- a model read the message and
    // every finding it emits carries `tier: 2` -- so on `config.tier2` alone
    // the schema would REQUIRE the judge's counters on an arm that has none,
    // and accept them beside B's own.
    const JUDGE_STATS = {
      rung1: 1,
      rung2: 0,
      unresolvedQuotes: 0,
      unknownPredicates: 0,
      duplicatesDropped: 0,
      repairAttempts: 0,
      failedClosed: 0,
      truncatedResponses: 0,
      abortedResponses: 0,
      segmentsJudged: 1,
      segmentsSkipped: 0,
      deadlineExpiries: 0,
      callerAbortsMidGeneration: 0,
      callerAbortsWhileQueued: 0,
      calls: [],
    };
    expect(base.config.tier2).toBe(true);
    expect(b({ tier2Stats: JUDGE_STATS }).success).toBe(false);
    expect(b({ tier2Stats: JUDGE_STATS, baselineStats: undefined }).success).toBe(false);
  });

  it("refuses an escalation threshold on a B row, and requires one on a compiled row", () => {
    // Both directions, because only together do they say what `uncertainBelow`
    // means. It is the threshold `escalate.ts` compares a prior tier's
    // confidence against in order to decide which SEGMENTS reach the judge;
    // Approach B judges the whole message in one call and never escalates.
    expect(b({ config: { ...base.config, uncertainBelow: 0.8 } }).success).toBe(false);
    expect(
      RunRecordSchema.safeParse({
        ...base,
        detector: "core-orchestrator",
        baselineStats: undefined,
        tier2Stats: {
          rung1: 0,
          rung2: 0,
          unresolvedQuotes: 0,
          unknownPredicates: 0,
          duplicatesDropped: 0,
          repairAttempts: 0,
          failedClosed: 0,
          truncatedResponses: 0,
          abortedResponses: 0,
          segmentsJudged: 0,
          segmentsSkipped: 0,
          deadlineExpiries: 0,
          callerAbortsMidGeneration: 0,
          callerAbortsWhileQueued: 0,
          calls: [],
        },
      }).success,
    ).toBe(false);
  });

  it("survives the JSONL round trip with the three counters no judge reports", () => {
    // The file is the deliverable. `messagesJudged`, `unknownEntityTypes` and
    // `messageBudgetExpiries` are the three fields that make `baselineStats` a
    // second schema rather than a rename, so they are the three worth proving
    // reach a reader intact.
    const [line] = toJsonl([{ ...base, baselineStats: B_STATS } as unknown as RunRecord])
      .trim()
      .split("\n");
    const reparsed = RunRecordSchema.safeParse(JSON.parse(line!));
    expect(reparsed.success, reparsed.success ? "" : JSON.stringify(reparsed.error)).toBe(true);
    expect(reparsed.success && reparsed.data.baselineStats).toEqual(B_STATS);
    expect(reparsed.success && reparsed.data.detector).toBe("approach-b");
  });
});
