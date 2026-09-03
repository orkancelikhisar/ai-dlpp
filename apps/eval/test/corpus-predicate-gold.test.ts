import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadCorpus } from "../src/driver/corpus.js";
import { loadTier2Gold } from "../src/driver/score.js";

/**
 * The predicate gold produced by the round that put the §3 question on EVERY
 * message, not only the span-bearing ones.
 *
 * WHY THIS FILE IS SEPARATE FROM `injection-p-fin-v2.gold-tier2.jsonl`. That
 * artifact is byte-pinned to its builder by `corpus-labelling.test.ts`, and it
 * records a DIFFERENT round: one whose brief defined the unit of work as "a
 * message plus one highlighted span", so it answered 20 of 189 predicate
 * questions and emitted 0 positives. This file is not an edit of it. The two
 * overlap on 20 items and must never be pooled into one denominator -- doing so
 * would double-weight those items and mix two rounds' labels.
 *
 * The numbers below are pinned as LITERALS. They were measured in Python over
 * the two annotators' returned label sets before this file existed, and they
 * are historical facts about a labelling round: they do not move because code
 * moved. A later edit to the artifact has to change these lines too.
 */

const REPO = fileURLToPath(new URL("../../..", import.meta.url));
const GOLD_PATH = `${REPO}corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl`;
const CORPUS_PATH = `${REPO}corpora/generated/injection-p-fin-v2.labelled.jsonl`;
const P_FIN_HASH = "ebb3cd68d973175f3ea40faeec00685e2cb9d83c6940e96a57e88ee269e8110a";

/** The 19 rows both annotators independently called true. */
const POSITIVES = [
  "inj-hn04-1", "inj-hn04-3", "inj-hn05-0", "inj-hn05-1", "inj-hn05-2",
  "inj-hn07-1", "inj-hn08-2", "inj-hn09-3", "inj-hn11-0", "inj-o01-1",
  "inj-o02-0", "inj-o02-2", "inj-o04-3", "inj-o06-1", "inj-o06-2",
  "inj-o08-2", "inj-o12-1", "inj-o13-3", "inj-o14-2",
];

/**
 * The 10 rows BOTH annotators marked borderline. Every one names a supplier,
 * a landlord or a print vendor and nothing else, which is the boundary §3
 * leaves open: §3.1's bare "counterparty" reaches any contracting party, while
 * §3.3's "deal counterparties" and §3.4's "the transaction" do not.
 */
const DISPUTED = [
  "inj-hn05-3", "inj-hn05-4", "inj-hn07-4", "inj-o01-5", "inj-o05-4",
  "inj-o07-4", "inj-o09-4", "inj-o10-3", "inj-o10-5", "inj-o16-3",
];

describe("corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl", () => {
  const rawLines = readFileSync(GOLD_PATH, "utf8").split("\n").filter((l) => l !== "");
  const gold = loadTier2Gold(readFileSync(GOLD_PATH, "utf8"));
  const corpus = loadCorpus(readFileSync(CORPUS_PATH, "utf8"));
  const byId = new Map(corpus.map((c) => [c.id, c]));

  it("puts the predicate question to every message in the corpus, one row per item", () => {
    expect(gold).toHaveLength(189);
    expect(gold.map((g) => g.itemId).sort()).toEqual(corpus.map((c) => c.id).sort());
  });

  it("holds the round's outcome: 19 positives, 160 scored negatives, 10 disputed", () => {
    expect(gold.filter((g) => g.satisfies)).toHaveLength(19);
    expect(gold.filter((g) => g.status === "disputed")).toHaveLength(10);
    expect(gold.filter((g) => g.status === "scored" && !g.satisfies)).toHaveLength(160);
    expect(gold.filter((g) => g.satisfies).map((g) => g.itemId).sort()).toEqual(POSITIVES);
    expect(gold.filter((g) => g.status === "disputed").map((g) => g.itemId).sort()).toEqual(DISPUTED);
  });

  it("every gold span slices back to its own text on the CORPUS item, not on a copy", () => {
    // A span that does not slice back is a gold set scoring against fiction.
    // Offsets are UTF-16 code units, which is what `String.prototype.slice`
    // indexes -- so this assertion is the unit check as well as the text check.
    let spans = 0;
    for (const row of gold) {
      for (const span of row.spans) {
        expect(byId.get(row.itemId)!.text.slice(span.start, span.end)).toBe(span.text);
        spans += 1;
      }
    }
    expect(spans).toBe(19);
  });

  it("every annotator's own quoted span slices back too", () => {
    // The annotators quoted at two different EXTENTS -- A the bare
    // organisation name, B the carrier sentence around it -- and both are kept
    // verbatim. Both must be real offsets into the item, or the provenance
    // this file exists to preserve is decorative.
    let checked = 0;
    for (const row of gold) {
      for (const call of [row.annotators.a, row.annotators.b]) {
        expect(call.span === undefined).toBe(!call.satisfies);
        if (call.span) {
          expect(byId.get(row.itemId)!.text.slice(call.span.start, call.span.end)).toBe(call.span.text);
          checked += 1;
        }
      }
    }
    expect(checked).toBe(38);
  });

  it("keeps A's span inside B's on every positive, which is what their disagreement was", () => {
    // Recorded because it is the whole span story of this round: the two
    // annotators never disagreed about WHICH name, only about how much
    // sentence to carry with it. The adjudicated span is the name alone.
    for (const row of gold.filter((g) => g.satisfies)) {
      const a = row.annotators.a.span!;
      const b = row.annotators.b.span!;
      expect(b.start).toBeLessThanOrEqual(a.start);
      expect(b.end).toBeGreaterThanOrEqual(a.end);
      expect(row.spans).toEqual([{ start: a.start, end: a.end, text: a.text }]);
    }
  });

  it("pins the policy document and the predicate the labels were read against", () => {
    expect(new Set(gold.map((g) => g.policyHash))).toEqual(new Set([P_FIN_HASH]));
    expect(new Set(gold.map((g) => g.predicateId))).toEqual(new Set(["client-relationship-disclosure"]));
    expect(new Set(gold.map((g) => g.entityType))).toEqual(new Set(["pred:client-relationship-disclosure"]));
  });

  it("records an adjudicated call both annotators actually made, on every row", () => {
    for (const row of gold) {
      expect([row.annotators.a.satisfies, row.annotators.b.satisfies]).toContain(row.satisfies);
      expect(row.annotators.a.rationale.length).toBeGreaterThan(0);
      expect(row.annotators.b.rationale.length).toBeGreaterThan(0);
      expect(row.adjudication.length).toBeGreaterThan(0);
    }
  });

  it("disputes exactly the rows where BOTH annotators said borderline", () => {
    // The exclusion rule this file applies, asserted against the annotators'
    // own confidence fields rather than against the status it produced -- so a
    // row silently promoted out of `disputed` fails here.
    for (const row of gold) {
      const bothBorderline =
        row.annotators.a.confidence === "borderline" && row.annotators.b.confidence === "borderline";
      expect(row.status === "disputed").toBe(bothBorderline);
      if (row.status === "disputed") expect(row.spans).toHaveLength(0);
    }
  });

  it("agrees with the second annotator on all 189 rows, which is this round's headline", () => {
    // Raw agreement 1.0 on `satisfies` AND on `confidence`. Pinned because a
    // future edit that changes one annotator's call without changing the other
    // would otherwise pass every other assertion here.
    const disagreements = gold.filter(
      (g) =>
        g.annotators.a.satisfies !== g.annotators.b.satisfies ||
        g.annotators.a.confidence !== g.annotators.b.confidence,
    );
    expect(disagreements.map((g) => g.itemId)).toEqual([]);
  });

  it("carries the generator's non-blind prediction in the file but keeps it out of the scorer", () => {
    // The generator's `meta.predicateConstruction.constructed` agreed with the
    // adjudication on all 189 rows. That is worth recording and dangerous to
    // score: it is the corpus author's own construction, not a third read. So
    // the raw line carries it and `loadTier2Gold` -- the only door into every
    // metric in score.ts -- must drop it. This asserts both halves.
    for (const line of rawLines) {
      const raw = JSON.parse(line) as { generator?: { blind?: boolean } };
      expect(raw.generator).toBeDefined();
      expect(raw.generator!.blind).toBe(false);
    }
    for (const row of gold) {
      expect(row).not.toHaveProperty("generator");
    }
  });
});
