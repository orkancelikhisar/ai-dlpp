import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CorpusItemSchema, loadCorpus } from "../src/driver/corpus.js";
import { RunRecordSchema, toJsonl } from "../src/driver/record.js";

const FIXTURES = join(import.meta.dirname, "..", "..", "..", "corpora", "fixtures");

describe("corpus", () => {
  it("loads the smoke corpus, rejecting any malformed line by number", () => {
    const items = loadCorpus(readFileSync(join(FIXTURES, "smoke.jsonl"), "utf8"));
    expect(items.length).toBeGreaterThanOrEqual(12);
    expect(items.every((i) => CorpusItemSchema.safeParse(i).success)).toBe(true);
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
      policyHash: "0".repeat(64),
      arm: "t0",
      backend: "wasm",
      provider: "claude",
      // `text` added to the specified fixture when RunRecordSchema gained the
      // field: the record now carries the message its offsets index into, so
      // this fixture has to supply one for the spans to validate against.
      text: "hello world",
      findings: [{ start: 0, end: 5, text: "hello", entityType: "client-name", severity: "high", tier: 0, source: "rule", confidence: 0.9, action: "block" }],
      gold: [{ start: 0, end: 5, text: "hello", entityType: "client-name", action: "block" }],
      timings: { tier0Ms: 0.4 },
      error: null,
    };
    expect(RunRecordSchema.safeParse(record).success).toBe(true);
  });

  it("carries the policy hash, so a record can never be attributed to the wrong IR", () => {
    const { policyHash: _omitted, ...withoutHash } = {
      schemaVersion: 1, runId: "r1", itemId: "a", policy: "p-fin", policyHash: "0".repeat(64),
      arm: "t0", backend: "wasm", provider: "claude", text: "hello world", findings: [], gold: [],
      timings: { tier0Ms: 0 }, error: null,
    };
    expect(RunRecordSchema.safeParse(withoutHash).success).toBe(false);
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
    policyHash: "0".repeat(64),
    arm: "t0",
    backend: "wasm" as const,
    provider: "claude",
    text: "hello world",
    findings: [],
    gold: [],
    timings: { tier0Ms: 0 },
    error: null,
  };

  it("refuses a schemaVersion other than the one this module writes", () => {
    // MEASURED: relaxing z.literal to z.number() leaves all five specified
    // tests green. RECORD_SCHEMA_VERSION's own doc comment promises Plan 8
    // refuses a version it does not know; without the literal, a future v2
    // record parses here as if it were v1 and the promise is not kept.
    expect(RunRecordSchema.safeParse({ ...record, schemaVersion: 2 }).success).toBe(false);
  });

  it("refuses a policyHash that is not a sha256 digest", () => {
    // MEASURED: relaxing the regex to a bare z.string() leaves all five
    // specified tests green, because the only hash the specified tests supply
    // is a well-formed one and the other test omits the field entirely.
    // "test-hash" is not hypothetical: it is the literal policyHash in
    // apps/eval/fixtures/minimal-ir.json today, so whoever wires Task 3's
    // page API must produce a real digest rather than forward that placeholder.
    expect(RunRecordSchema.safeParse({ ...record, policyHash: "test-hash" }).success).toBe(false);
    expect(RunRecordSchema.safeParse({ ...record, policyHash: "A".repeat(64) }).success).toBe(false);
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
    policyHash: "0".repeat(64),
    arm: "t0",
    backend: "wasm" as const,
    provider: "claude",
    text: "hello world",
    gold: [],
    timings: { tier0Ms: 0 },
    error: null,
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
    runId: "r1", itemId: "a", policy: "p-fin", policyHash: "0".repeat(64),
    arm: "t0", backend: "wasm" as const, provider: "claude",
    text: "hello world", findings: [], gold: [],
    timings: { tier0Ms: 0 }, error: null,
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
      runId: "r1", itemId: "a", policy: "p-fin", policyHash: "0".repeat(64),
      arm: "t0", backend: "wasm" as const, provider: "claude",
      text: "before after end", findings: [], gold: [],
      timings: { tier0Ms: 0 }, error: null,
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
