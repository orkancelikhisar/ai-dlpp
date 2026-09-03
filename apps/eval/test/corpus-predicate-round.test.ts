import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadCorpus } from "../src/driver/corpus.js";
import type { RunRecord } from "../src/driver/record.js";
import {
  MATCH_RULES,
  loadTier2Gold,
  scoreArm,
  scoreTrivialFloors,
} from "../src/driver/score.js";
import {
  BEST_ARM_F1_ON_THIS_GOLD,
  PREDICATE_DISCRIMINATORS,
  PREDICATE_ROUND_ID,
  PREDICATE_ROUND_RELPATHS,
  buildPredicateRoundRecord,
  cohensKappa,
  deriveRowIdMap,
  loadAnnotatorReturns,
  loadPredicateQueue,
  serializePredicateRoundRecord,
  sha256,
} from "../src/corpus/predicate-round.js";

/**
 * The round record, and the three things it exists to make falsifiable.
 *
 * WHAT WENT WRONG WITHOUT IT, all three found by review of the commit that
 * shipped the gold:
 *
 * 1. All 378 `annotators.{a,b}.rationale` strings in the gold were an
 *    adjudicator's rewrite, not what the annotators returned, under a schema
 *    comment promising they were carried verbatim. Nothing could fail, because
 *    the annotators' returns were in a temp directory and the only assertion on
 *    the field was `length > 0`.
 * 2. The round shipped no blindness record at all and no link from the gold to
 *    the queue it was built around, so "two annotators, blind, over an opaque
 *    queue" had no mutant that could fail it.
 * 3. No trivial floor was computed, and when computed it beats every arm.
 *
 * So the assertions here are joins between INDEPENDENT committed artifacts --
 * the annotators' own return files, the queue's own bytes, the corpus, and the
 * gold -- rather than checks of one artifact against itself. The literals were
 * measured in Python over those files before this test existed.
 */

const REPO = fileURLToPath(new URL("../../..", import.meta.url));
const read = (relpath: string): string => readFileSync(`${REPO}${relpath}`, "utf8");

const CORPUS_JSONL = read(PREDICATE_ROUND_RELPATHS.corpus);
const QUEUE_JSONL = read(PREDICATE_ROUND_RELPATHS.queue);
const A_JSON = read(PREDICATE_ROUND_RELPATHS.annotatorA);
const B_JSON = read(PREDICATE_ROUND_RELPATHS.annotatorB);
const GOLD_JSONL = read(PREDICATE_ROUND_RELPATHS.gold);
const RECORD_JSON = read(PREDICATE_ROUND_RELPATHS.record);

const items = loadCorpus(CORPUS_JSONL);
const itemById = new Map(items.map((i) => [i.id, i]));
const gold = loadTier2Gold(GOLD_JSONL);
const queue = loadPredicateQueue(QUEUE_JSONL);
const returns = { a: loadAnnotatorReturns(A_JSON), b: loadAnnotatorReturns(B_JSON) };

const record = buildPredicateRoundRecord({
  corpusJsonl: CORPUS_JSONL,
  queueJsonl: QUEUE_JSONL,
  annotatorAJson: A_JSON,
  annotatorBJson: B_JSON,
  goldJsonl: GOLD_JSONL,
});

describe("the committed round record is pinned to the bytes it describes", () => {
  it("reproduces byte for byte from the five artifacts it names", () => {
    // Every field except the two salt/seed commitments and the prose is
    // computed from the artifacts, so editing any of them without re-emitting
    // this record fails here -- including editing the gold.
    expect(serializePredicateRoundRecord(record)).toBe(RECORD_JSON);
  });

  it("hashes each artifact, and the hashes are of the files it points at", () => {
    expect(record.artifacts.corpus.sha256).toBe(sha256(CORPUS_JSONL));
    expect(record.artifacts.queue.sha256).toBe(sha256(QUEUE_JSONL));
    expect(record.artifacts.annotatorA.sha256).toBe(sha256(A_JSON));
    expect(record.artifacts.annotatorB.sha256).toBe(sha256(B_JSON));
    expect(record.artifacts.gold.sha256).toBe(sha256(GOLD_JSONL));
    // Not vacuous: five different files, five different digests.
    const digests = Object.values(record.artifacts).map((a) => a.sha256);
    expect(new Set(digests).size).toBe(5);
    expect(record.id).toBe(PREDICATE_ROUND_ID);
  });
});

describe("the gold's annotator records are the annotators' own bytes", () => {
  const byRowId = new Map(deriveRowIdMap(queue, items).map((r) => [r.rowId, r.itemId]));
  const goldByItem = new Map(gold.map((g) => [g.itemId, g]));

  it("carries every field of every one of the 378 calls verbatim", () => {
    // The check the round did not have. `satisfies`, `confidence`, `rationale`
    // and the quoted text are compared against the annotator's own return file,
    // joined through the queue row id, on all 189 rows on both sides.
    let checked = 0;
    for (const side of ["a", "b"] as const) {
      for (const returned of returns[side]) {
        const itemId = byRowId.get(returned.rowId);
        expect(itemId, `rowId ${returned.rowId} is not in the queue`).toBeDefined();
        const row = goldByItem.get(itemId!)!;
        const call = row.annotators[side];
        expect(call.satisfies, `${itemId} ${side} satisfies`).toBe(returned.violates);
        expect(call.confidence, `${itemId} ${side} confidence`).toBe(returned.confidence);
        expect(call.rationale, `${itemId} ${side} rationale`).toBe(returned.rationale);
        expect(call.quote?.text, `${itemId} ${side} quote`).toBe(returned.quote);
        checked += 1;
      }
    }
    expect(checked).toBe(378);
  });

  it("REFUSES to build a record for a gold whose annotator block is not verbatim", () => {
    // The check lives in the builder and not only here, so a record cannot be
    // emitted for a falsified gold at all. The assertion above cannot see that:
    // with a correct gold the guard never fires, and MEASURED, disabling it
    // survived the whole suite. So this feeds the builder a gold with one
    // rewritten field and requires the throw.
    const lines = GOLD_JSONL.split("\n").filter((l) => l !== "");
    const bend = (index: number, mutate: (row: Record<string, never>) => void): string => {
      const rows = lines.map((l) => JSON.parse(l) as Record<string, never>);
      mutate(rows[index]!);
      return `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`;
    };
    const inputs = {
      corpusJsonl: CORPUS_JSONL,
      queueJsonl: QUEUE_JSONL,
      annotatorAJson: A_JSON,
      annotatorBJson: B_JSON,
      goldJsonl: GOLD_JSONL,
    };
    // A paraphrased rationale -- the exact defect this round shipped, 378 times.
    const rewritten = bend(0, (row) => {
      const annotators = row["annotators"] as unknown as { a: { rationale: string } };
      annotators.a.rationale = `${annotators.a.rationale} (tidied up by the adjudicator)`;
    });
    expect(() => buildPredicateRoundRecord({ ...inputs, goldJsonl: rewritten })).toThrow(
      /is not what annotator a returned/,
    );
    // And a quoted extent the annotator did not write.
    const positive = lines.findIndex((l) => (JSON.parse(l) as { satisfies: boolean }).satisfies);
    const requoted = bend(positive, (row) => {
      const annotators = row["annotators"] as unknown as { b: { quote: { text: string } } };
      annotators.b.quote.text = annotators.b.quote.text.toUpperCase();
    });
    expect(() => buildPredicateRoundRecord({ ...inputs, goldJsonl: requoted })).toThrow(
      /is not what annotator b returned/,
    );
    // Non-vacuity: the unmodified inputs build without throwing.
    expect(() => buildPredicateRoundRecord(inputs)).not.toThrow();
  });

  it("is not comparing a file with itself: the two annotators wrote different prose", () => {
    // Vacuity control. A join bug that compared A's gold record with A's return
    // twice would pass the test above; these numbers say the two sides are
    // genuinely different text, and that the gold's prose is not a single
    // boilerplate string either.
    const aRationales = returns.a.map((r) => r.rationale);
    const bRationales = returns.b.map((r) => r.rationale);
    expect(aRationales.filter((r, i) => r === bRationales[i])).toEqual([]);
    // MEASURED over the two return files: A reused 31 distinct rationales
    // across its 189 rows and B reused 48. That concentration is itself worth
    // pinning -- the gold this replaces showed 67 and 124, which is how the
    // rewrite was found.
    expect(new Set(aRationales).size).toBe(31);
    expect(new Set(bRationales).size).toBe(48);
    expect(new Set(gold.map((g) => g.annotators.a.rationale)).size).toBe(31);
    expect(new Set(gold.map((g) => g.annotators.b.rationale)).size).toBe(48);
  });

  it("returns a quote and never an offset, which is why the gold's offsets are located", () => {
    // The provenance boundary `AnnotatorCallSchema` documents. If a future
    // round's annotators do return offsets, this fails and the schema comment
    // has to be rewritten with them.
    const keys = new Set<string>();
    for (const json of [A_JSON, B_JSON]) {
      for (const row of JSON.parse(json) as Record<string, unknown>[]) {
        for (const key of Object.keys(row)) keys.add(key);
      }
    }
    expect([...keys].sort()).toEqual(["confidence", "quote", "rationale", "rowId", "violates"]);
    expect(returns.a.filter((r) => r.quote !== undefined)).toHaveLength(19);
    expect(returns.b.filter((r) => r.quote !== undefined)).toHaveLength(19);
  });

  it("agrees on every row, and the record states the agreement it can check", () => {
    expect(record.answers.satisfiesAgreement).toBe(1);
    expect(record.answers.confidenceAgreement).toBe(1);
    expect(record.answers.satisfiesKappa).toBe(1);
    expect(record.answers.confidenceKappa).toBe(1);
    expect(record.answers.positives).toBe(19);
    expect(record.answers.disputed).toBe(10);
  });
});

describe("cohensKappa", () => {
  it("computes a case worked by hand rather than read back from the function", () => {
    // 10 items. A says true on the first five, B on the first three. They agree
    // on 8, so po = 0.8. pe = (5/10)(3/10) + (5/10)(7/10) = 0.5, and
    // (0.8 - 0.5) / (1 - 0.5) = 0.6.
    const a = [true, true, true, true, true, false, false, false, false, false];
    const b = [true, true, true, false, false, false, false, false, false, false];
    expect(cohensKappa(a, b)).toBeCloseTo(0.6, 12);
    // Perfect agreement on a mixed label is 1; chance agreement alone is 0.
    expect(cohensKappa(a, a)).toBe(1);
    expect(cohensKappa([true, false], [false, true])).toBe(-1);
  });

  it("refuses the degenerate case rather than reporting 1", () => {
    // Both raters saying "false" to everything agree perfectly and have learned
    // nothing. Returning 1 there is how a kappa of 1 becomes meaningless.
    expect(() => cohensKappa([false, false], [false, false])).toThrow(/undefined/);
  });
});

describe("the gold is bound to the queue the annotators actually read", () => {
  it("joins every gold row through its rowId to the queue row carrying its message", () => {
    // The link that did not exist. `rowId` -> queue row -> message -> corpus
    // item, checked in that direction, so a gold row claiming a queue id whose
    // message is a different item fails.
    const queueByRowId = new Map(queue.map((r) => [r.rowId, r.text]));
    expect(queue).toHaveLength(189);
    expect(queueByRowId.size).toBe(189);
    for (const row of gold) {
      expect(row.round, row.itemId).toBe(PREDICATE_ROUND_ID);
      const text = queueByRowId.get(row.rowId!);
      expect(text, `${row.itemId} claims rowId ${String(row.rowId)}`).toBeDefined();
      expect(text, row.itemId).toBe(itemById.get(row.itemId)!.text);
    }
    expect(new Set(gold.map((g) => g.rowId)).size).toBe(189);
  });

  it("hands over the message and nothing else, in the committed bytes and not a rebuild", () => {
    // `corpus-predicate-queue.test.ts` asserts these properties of a queue it
    // rebuilds in memory. This asserts them of the file the annotators opened.
    const forbidden = new Set<string>();
    for (const item of items) {
      forbidden.add(item.id);
      const meta = item.meta as Record<string, unknown> | undefined;
      for (const key of ["carrierId", "carrierStratum"]) {
        const value = meta?.[key];
        if (typeof value === "string") forbidden.add(value);
      }
      for (const raw of (meta?.["injections"] ?? []) as Record<string, unknown>[]) {
        for (const key of ["family", "type"]) {
          const value = raw[key];
          if (typeof value === "string") forbidden.add(value);
        }
        const role = ((raw["dimensions"] ?? {}) as Record<string, unknown>)["constructedRole"];
        if (typeof role === "string" && role !== "none") forbidden.add(role);
      }
      for (const span of item.gold) forbidden.add(span.entityType);
    }
    expect(forbidden.size).toBeGreaterThan(100);
    for (const [i, line] of QUEUE_JSONL.split("\n").entries()) {
      if (line === "") continue;
      const raw = JSON.parse(line) as Record<string, unknown>;
      expect(Object.keys(raw).sort(), `line ${i + 1}`).toEqual(["rowId", "text"]);
      const outsideText = JSON.stringify({ rowId: raw["rowId"] });
      for (const token of forbidden) {
        expect(outsideText.includes(token), `line ${i + 1} leaks ${token}`).toBe(false);
      }
    }
  });

  it("records a mapping derived from the artifacts, matching the order the queue was read in", () => {
    expect(record.handover.rows).toHaveLength(189);
    expect(record.handover.rows.map((r) => r.position)).toEqual(queue.map((_, i) => i));
    expect(record.handover.rows.map((r) => r.rowId)).toEqual(queue.map((r) => r.rowId));
    expect(record.handover.saltSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(record.handover.seedSha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("keeps the shuffle's effect in the record, recomputed from the committed order", () => {
    // MEASURED in Python over the committed corpus before this file existed:
    // 162 of 188 adjacent pairs share a carrier in source order. The queue
    // order drives that to 0, which is the one stratum an annotator can see.
    const source = record.handover.adjacency.sourceOrder.find((m) => m.dimension === "carrier")!;
    const queued = record.handover.adjacency.queueOrder.find((m) => m.dimension === "carrier")!;
    expect(source.adjacentPairs).toBe(188);
    expect(source.sameStratumPairs).toBe(162);
    expect(queued.sameStratumPairs).toBe(0);
    // And the dimensions it makes worse are in the record too, not dropped.
    const wave = record.handover.adjacency.queueOrder.find((m) => m.dimension === "wave")!;
    expect(wave.sameStratumPairs).toBeGreaterThan(
      record.handover.adjacency.sourceOrder.find((m) => m.dimension === "wave")!.sameStratumPairs,
    );
  });
});

describe("the disclosures in the record are measurements, not assurances", () => {
  it("shows the read channel is open: twelve fragments decide the corpus and all twelve are committed", () => {
    const families = read("apps/eval/src/corpus/families.v2.ts");
    const all = [...PREDICATE_DISCRIMINATORS.positive, ...PREDICATE_DISCRIMINATORS.negative];
    expect(all).toHaveLength(12);
    for (const fragment of all) {
      expect(families.includes(fragment), `families.v2.ts is missing ${fragment}`).toBe(true);
      expect(items.some((i) => i.text.includes(fragment)), `no message contains ${fragment}`).toBe(true);
    }
    // The file that carries them also states the answer beside them.
    expect(families).toContain("labelBasis");
    expect(families).toContain("constructedRole");
    // And the seven positive fragments decide the gold exactly: a matcher that
    // reads nothing else recovers all 19 positives with no false positive.
    const matched = items.filter((i) => PREDICATE_DISCRIMINATORS.positive.some((f) => i.text.includes(f)));
    expect(matched.map((i) => i.id).sort()).toEqual(gold.filter((g) => g.satisfies).map((g) => g.itemId).sort());
    // Which is what `answers.effectiveDecisions` says: 19 rows, seven decisions.
    expect(Object.values(record.answers.positivesPerDiscriminator).reduce((x, y) => x + y, 0)).toBe(19);
    expect(Object.keys(record.answers.positivesPerDiscriminator)).toHaveLength(7);
    // A second committed file names one of the positives outright.
    const labelling = read("apps/eval/src/corpus/labelling.ts");
    expect(labelling).toContain("inj-o01-1");
    expect(labelling).toContain("the party we signed the confidentiality agreement with is Marrowfield Group");
  });

  it("shows the labels are not independent of the corpus's own construction", () => {
    // The record says this on 189 of 189 and on all 19 spans; here is the
    // measurement it is quoting.
    let sameLabel = 0;
    let sameSpan = 0;
    for (const row of gold) {
      const meta = itemById.get(row.itemId)!.meta as { predicateConstruction?: { constructed?: boolean } };
      if (meta.predicateConstruction!.constructed === row.satisfies) sameLabel += 1;
      const clientName = itemById.get(row.itemId)!.gold.filter((g) => g.entityType === "client-name");
      if (row.spans.length === 1 && clientName.length === 1) {
        const c = clientName[0]!;
        const s = row.spans[0]!;
        if (c.start === s.start && c.end === s.end && c.text === s.text) sameSpan += 1;
      }
    }
    expect(sameLabel).toBe(189);
    expect(sameSpan).toBe(19);
    expect(record.blindness.channels.find((c) => c.channel === "answer-key")!.audited).toBe("audited");
  });

  it("shows the corpus's own scoping fields still describe the previous round", () => {
    // The record discloses this rather than editing the corpus, which is
    // byte-pinned to its builder. The counts are the disclosure.
    let goldIncomplete = 0;
    let predicateUnlabelledClass = 0;
    let questionsUnanswered = 0;
    const scoreableHere: string[] = [];
    for (const item of items) {
      const meta = item.meta as {
        scoringScope?: { goldIsComplete?: boolean; unlabelledClasses?: string[] };
        labelQuestions?: { kind?: string; state?: string }[];
      };
      if (meta.scoringScope?.goldIsComplete === false) goldIncomplete += 1;
      if ((meta.scoringScope?.unlabelledClasses ?? []).includes("pred:client-relationship-disclosure")) {
        predicateUnlabelledClass += 1;
      } else {
        scoreableHere.push(item.id);
      }
      for (const q of meta.labelQuestions ?? []) {
        if (q.kind === "message-predicate" && q.state === "unanswered") questionsUnanswered += 1;
      }
    }
    expect(goldIncomplete).toBe(189);
    expect(predicateUnlabelledClass).toBe(170);
    expect(questionsUnanswered).toBe(189);
    // Maximal disagreement: the 19 items the corpus calls scoreable and this
    // round's 19 positives do not overlap at all.
    const positives = new Set(gold.filter((g) => g.satisfies).map((g) => g.itemId));
    expect(scoreableHere).toHaveLength(19);
    expect(scoreableHere.filter((id) => positives.has(id))).toEqual([]);
    expect(record.corpusScopingDisagreement).toContain("ZERO overlap");
  });

  it("records two UNAUDITED channels and withdraws the commit message's told-channel claim", () => {
    // The honest shape of this round's blindness claim. It is here as an
    // assertion because the temptation a review found is to let an unaudited
    // channel read as an audited one.
    const byChannel = new Map(record.blindness.channels.map((c) => [c.channel, c]));
    expect(byChannel.get("read")!.audited).toBe("unaudited");
    expect(byChannel.get("told")!.audited).toBe("unaudited");
    expect(byChannel.get("structural")!.audited).toBe("audited");
    expect(byChannel.get("order")!.audited).toBe("audited");
    expect(byChannel.get("read")!.breaches).toHaveLength(2);
    expect(byChannel.get("told")!.evidence).toContain("withdrawn");
  });
});

describe("the trivial floor beats every arm measured against this gold", () => {
  const scored = gold.filter((g) => g.status === "scored");
  const floors = scoreTrivialFloors(
    scored.map((g) => ({
      text: itemById.get(g.itemId)!.text,
      goldSpans: g.spans.map((s) => ({ start: s.start, end: s.end })),
    })),
  );
  const byId = new Map(floors.map((f) => [f.reader, f]));

  it("reproduces the counts measured in Python over the same 179 rows", () => {
    expect(scored).toHaveLength(179);
    for (const rule of MATCH_RULES) {
      expect(byId.get("capitalised-multiword")!.byRule[rule], rule).toMatchObject({ tp: 19, fp: 31, fn: 0 });
      expect(byId.get("first-capitalised-multiword")!.byRule[rule], rule).toMatchObject({ tp: 14, fp: 16, fn: 5 });
    }
    expect(byId.get("whole-message")!.byRule.overlap).toMatchObject({ tp: 19, fp: 160, fn: 0 });
    expect(byId.get("capitalised-multiword")!.byRule.overlap.recall).toBe(1);
  });

  it("puts the best arm ever measured against this gold BELOW the floor, under every rule", () => {
    // The round's result, stated as a result. `BEST_ARM_F1_ON_THIS_GOLD` is
    // tier2-Qwen3-4B over runs/slate-rebuild-*, which is not committed; the
    // floor beside it is recomputed here from committed bytes.
    for (const rule of MATCH_RULES) {
      const best = Math.max(...floors.map((f) => f.byRule[rule].f1 ?? 0));
      expect(best, rule).toBeGreaterThan(BEST_ARM_F1_ON_THIS_GOLD);
    }
    expect(BEST_ARM_F1_ON_THIS_GOLD).toBeCloseTo(0.197, 3);
    expect(record.trivialFloor.readers).toEqual(floors);
    expect(record.trivialFloor.verdict).toContain("NO ARM BEATS THE FLOOR UNDER ANY RULE");
    expect(record.limits.join(" ")).toContain("no arm measured against this gold beats the trivial floor");
  });

  it("is not an artifact of the gold spans being short: recall costs the regex nothing", () => {
    // The mechanism, so a reader can tell whether it will hold on another
    // corpus. Every gold span is a capitalised multi-word token, and the whole
    // corpus holds only 50 of them across 179 messages -- 19 gold and 31 not.
    for (const row of scored) {
      for (const span of row.spans) {
        expect(span.text, row.itemId).toMatch(/^[A-Z][a-z]+(?: [A-Z][a-z]+)+$/);
      }
    }
    expect(byId.get("capitalised-multiword")!.findings).toBe(50);
  });
});

describe("this gold actually reaches the scorer", () => {
  // Nothing in the repository joined the 189-row gold to a run, so
  // `assertGoldMatchesRecords`, the slice-back cross-check, the vocabulary
  // filter and the disputed exclusion were all unexercised on it. `runs/` is
  // gitignored, so the run side is built here over the real corpus text.
  const PRED = "pred:client-relationship-disclosure";
  const makeRecord = (
    itemId: string,
    findings: readonly { start: number; end: number; text: string; entityType?: string }[],
    policyHash?: string,
  ): RunRecord =>
    ({
      schemaVersion: 1,
      runId: "predicate-gold-join",
      itemId,
      policy: "p-fin",
      irHash: "b00e5ce67a7e6d1643c81b2e8f29498faad71b5d6b4a6e53b70b644b34f592d0",
      policyHash: policyHash ?? gold[0]!.policyHash,
      arm: "tier2-Test-Model",
      detector: "core-orchestrator",
      backend: "webgpu",
      provider: "claude",
      config: { tier0: true, tier1: false, tier2: true, t2Model: "Test-Model", backend: "webgpu" },
      text: itemById.get(itemId)!.text,
      findings: findings.map((f) => ({
        start: f.start,
        end: f.end,
        text: f.text,
        entityType: f.entityType ?? PRED,
        severity: "high" as const,
        tier: 2 as const,
        source: "Test-Model",
        confidence: 1,
        action: "redact" as const,
      })),
      gold: [],
      timings: { tier0Ms: 0 },
      degraded: [],
      error: null,
      abandonedWorkInFlight: false,
    }) as unknown as RunRecord;

  it("scores all 189 rows, excluding the ten disputed ones and their findings", () => {
    // A perfect arm on the 19 positives, which also fires once on every
    // disputed row. Its precision must be 1: the ten disputed findings are
    // suppressed with the rows. A filter keying on `satisfies` or on
    // `spans.length` instead of on `status` re-admits all ten here.
    const records = gold.map((row) =>
      makeRecord(
        row.itemId,
        row.spans.length > 0
          ? row.spans.map((s) => ({ start: s.start, end: s.end, text: s.text }))
          : row.status === "disputed"
            ? [{ start: 0, end: 3, text: itemById.get(row.itemId)!.text.slice(0, 3) }]
            : [],
      ),
    );
    const armScore = scoreArm({ records, gold });
    expect(armScore.coverage.goldRows).toBe(189);
    expect(armScore.coverage.goldRowsScored).toBe(179);
    expect(armScore.coverage.goldRowsDisputed).toBe(10);
    expect(armScore.coverage.goldRowsWithoutRecord).toBe(0);
    expect(armScore.coverage.recordsWithoutGold).toBe(0);
    expect(armScore.coverage.goldSpansScored).toBe(19);
    expect(armScore.coverage.findingsInVocabulary).toBe(19);
    expect(armScore.coverage.findingsOnDisputedRows).toBe(10);
    for (const rule of MATCH_RULES) {
      expect(armScore.byRule[rule], rule).toMatchObject({ tp: 19, fp: 0, fn: 0, precision: 1, recall: 1, f1: 1 });
    }
  });

  it("ignores a finding at another entityType instead of scoring it a false positive", () => {
    const row = gold.find((g) => g.satisfies)!;
    const armScore = scoreArm({
      records: [makeRecord(row.itemId, [{ start: 0, end: 3, text: itemById.get(row.itemId)!.text.slice(0, 3), entityType: "in-pan" }])],
      gold: [row],
    });
    expect(armScore.coverage.findingsInVocabulary).toBe(0);
    expect(armScore.coverage.findingsOutOfVocabulary).toBe(1);
    for (const rule of MATCH_RULES) expect(armScore.byRule[rule].fp, rule).toBe(0);
  });

  it("refuses a run taken against a different policy document", () => {
    const row = gold.find((g) => g.satisfies)!;
    expect(() =>
      scoreArm({ records: [makeRecord(row.itemId, [], "0".repeat(64))], gold: [row] }),
    ).toThrow(/was labelled against/);
  });

  it("refuses a gold span that does not slice back on the run's own text", () => {
    const row = gold.find((g) => g.satisfies)!;
    const drifted = { ...row, spans: [{ ...row.spans[0]!, start: row.spans[0]!.start + 1 }] };
    expect(() => scoreArm({ records: [makeRecord(row.itemId, [])], gold: [drifted] })).toThrow(
      /the gold and the run disagree about the message/,
    );
  });
});
