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

  it("every annotator's own quote slices back, at the ONE place it occurs", () => {
    // The annotators quoted at two different EXTENTS -- A the bare
    // organisation name, B the carrier sentence around it -- and both quotes
    // are kept verbatim. Neither annotator returned OFFSETS; the offsets beside
    // each quote were located afterwards by searching the item text, so this
    // asserts the search was unambiguous as well as correct. A quote occurring
    // twice would have had its offsets chosen by a string search rather than by
    // the annotator, and the corpus does carry items with two organisation
    // names in one sentence.
    let checked = 0;
    for (const row of gold) {
      for (const call of [row.annotators.a, row.annotators.b]) {
        expect(call.quote === undefined).toBe(!call.satisfies);
        if (call.quote) {
          const text = byId.get(row.itemId)!.text;
          expect(text.slice(call.quote.start, call.quote.end)).toBe(call.quote.text);
          expect(text.split(call.quote.text).length - 1, `${row.itemId} ${call.quote.text}`).toBe(1);
          expect(text.indexOf(call.quote.text)).toBe(call.quote.start);
          checked += 1;
        }
      }
    }
    expect(checked).toBe(38);
  });

  it("adjudicates the ORGANISATION NAME ALONE, not the carrier sentence either annotator quoted", () => {
    // The span convention, asserted against properties of the MESSAGE rather
    // than against the row's own `annotators` block. The old form of this test
    // compared `row.spans` with `row.annotators.a.quote` and nothing else,
    // which is an expectation derived from the thing under test: widening both
    // together left every assertion passing, and a sentence-extent gold span is
    // an `overlap` match and an `exact` and `iou50` MISS -- two of the three
    // columns would read 0.000 for a reason no reader could see.
    //
    // What makes a span "the name alone" without consulting the gold: it is a
    // run of capitalised words with no lowercase word in it, it is bounded by
    // non-letters on both sides, and it is a PROPER substring of the sentence
    // annotator B quoted around it.
    for (const row of gold.filter((g) => g.satisfies)) {
      const text = byId.get(row.itemId)!.text;
      expect(row.spans).toHaveLength(1);
      const span = row.spans[0]!;
      expect(span.text, row.itemId).toMatch(/^[A-Z][a-z]+(?: [A-Z][a-z]+)+$/);
      expect(/[A-Za-z]/.test(text.slice(Math.max(0, span.start - 1), span.start)), row.itemId).toBe(false);
      expect(/[A-Za-z]/.test(text.slice(span.end, span.end + 1)), row.itemId).toBe(false);
      const b = row.annotators.b.quote!;
      expect(b.start, row.itemId).toBeLessThan(span.start);
      expect(b.end, row.itemId).toBeGreaterThanOrEqual(span.end);
      expect(b.text.includes(span.text), row.itemId).toBe(true);
      // And a sentence's worth of glue really is what B carried around it:
      // MEASURED over the 19, B's quote runs 3 to 12 words longer than the
      // name. 3 is the shortest carrier the corpus has ("we are pitching X").
      expect(b.text.split(" ").length - span.text.split(" ").length, row.itemId).toBeGreaterThanOrEqual(3);
    }
  });

  it("keeps A's quote inside B's on every positive, which is what their disagreement was", () => {
    // Recorded because it is the whole span story of this round: the two
    // annotators never disagreed about WHICH name, only about how much
    // sentence to carry with it.
    for (const row of gold.filter((g) => g.satisfies)) {
      const a = row.annotators.a.quote!;
      const b = row.annotators.b.quote!;
      expect(b.start).toBeLessThanOrEqual(a.start);
      expect(b.end).toBeGreaterThanOrEqual(a.end);
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

  it("names its policy document, its round and its queue row on every row", () => {
    // `policy` was unasserted: score.ts:165-171 argues at length that the two
    // gold files disagree about it ON PURPOSE and that joining two policies
    // into one denominator is the failure to prevent, and the field carrying
    // that meaning had nothing pinning it. `rowId` and `round` are the binding
    // to the handover; `corpus-predicate-round.test.ts` checks where they point.
    for (const row of gold) {
      expect(row.policy, row.itemId).toBe("p-fin");
      expect(row.round, row.itemId).toBe("v2-blind-predicate-all-messages-p-fin");
      expect(row.rowId, row.itemId).toMatch(/^[0-9a-f]{16}$/);
    }
    expect(new Set(gold.map((g) => g.rowId)).size).toBe(189);
  });

  it("takes the WEAKER of the two annotators' confidences, which is what score.ts documents", () => {
    // Asserted against the annotators' own fields rather than assumed. It is
    // vacuous-looking on this artifact and deliberately kept: the two agreed on
    // `confidence` on all 189 rows, so "the weaker" and "either" name the same
    // value here, and a row whose adjudicated confidence drifted away from both
    // annotators would otherwise pass every other assertion in this file.
    for (const row of gold) {
      const weaker =
        row.annotators.a.confidence === "borderline" || row.annotators.b.confidence === "borderline"
          ? "borderline"
          : "clear";
      expect(row.confidence, row.itemId).toBe(weaker);
    }
  });

  it("writes an adjudication that says what the row says", () => {
    // `adjudication` was checked only for `length > 0`, so its prose could
    // contradict the machine-readable fields beside it -- including the SPAN
    // CONVENTION sentence, which states the rule the gold spans follow.
    for (const row of gold) {
      if (row.status === "disputed") {
        expect(row.adjudication.startsWith("DISPUTED."), row.itemId).toBe(true);
      } else {
        expect(row.adjudication.startsWith(`AGREED ${String(row.satisfies)}, `), row.itemId).toBe(true);
      }
      if (row.satisfies) {
        expect(row.adjudication, row.itemId).toContain(
          "SPAN CONVENTION: the organisation name alone",
        );
        // The sentence that was false before this round's fix: the annotators
        // returned quotes, not extents.
        expect(row.adjudication, row.itemId).toContain("NEITHER ANNOTATOR RETURNED OFFSETS");
        expect(row.adjudication, row.itemId).not.toContain("Both extents are recorded verbatim");
        expect(row.adjudication, row.itemId).toContain(row.spans[0]!.text);
      }
    }
  });

  it("records that `confidence` carries no information about the SCORED set", () => {
    // Not a bound, a disclosure. Every borderline row became `disputed`, so
    // `confidence` is "clear" on all 179 scored rows and the field's stated
    // purpose -- letting a reader see how much of a result rests on soft
    // labels -- is vacuous on this artifact. Pinned so a later reader does not
    // take a uniform column as evidence the scored labels were graded soft
    // against hard, and so a round that DOES score a borderline row has to
    // change this line deliberately.
    expect(gold.filter((g) => g.status === "scored" && g.confidence === "borderline")).toEqual([]);
    expect(gold.filter((g) => g.confidence === "borderline")).toHaveLength(10);
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

  it("states the generator's prediction as the corpus states it, and says where it disagrees", () => {
    // The generator block was checked only for existence and `blind === false`,
    // so `constructed` could name the opposite of the corpus's own record and
    // `agreesWithAdjudication` could be true while the two disagreed. Both are
    // checked here against the corpus, which is the block's cited source.
    //
    // The count is the finding, and it is not flattering: the generator's
    // construction record equals the adjudicated label on 189 of 189 rows. The
    // gold is therefore NOT independent of the corpus's construction -- see
    // `corpus-predicate-round.test.ts`, which pins that as a disclosure rather
    // than leaving it to be rediscovered.
    let agree = 0;
    for (const line of rawLines) {
      const raw = JSON.parse(line) as {
        itemId: string;
        satisfies: boolean;
        generator: { constructed: boolean; agreesWithAdjudication: boolean };
      };
      const meta = byId.get(raw.itemId)!.meta as
        | { predicateConstruction?: { constructed?: boolean } }
        | undefined;
      expect(raw.generator.constructed, raw.itemId).toBe(meta!.predicateConstruction!.constructed);
      expect(raw.generator.agreesWithAdjudication, raw.itemId).toBe(
        raw.generator.constructed === raw.satisfies,
      );
      if (raw.generator.agreesWithAdjudication) agree += 1;
    }
    expect(agree).toBe(189);
  });
});
