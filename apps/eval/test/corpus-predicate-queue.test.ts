import { createHash, createHmac } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadCorpus } from "../src/driver/corpus.js";
import { V2_QUEUE_PATH } from "../src/corpus/build-v2.js";
import { BRIEF_PARAPHRASE_FRAGMENT } from "../src/corpus/labelling.js";
import { PREDICATE_ID, PREDICATE_QUESTION } from "../src/corpus/questions.js";
import {
  PREDICATE_QUEUE_SOURCE,
  QUEUE_FILENAME,
  assertQueueCarriesNothingElse,
  buildPredicateQueueArtifacts,
} from "../src/corpus/build-predicate-queue.js";
import {
  ROW_ID_ALGORITHM,
  ROW_ID_HEX_LENGTH,
  SHUFFLE_CONSTRAINT,
  STRATUM_DIMENSIONS,
  adjacencyOf,
  buildPredicateQueue,
  forbiddenTokensOf,
  rowIdFor,
  shuffledPositions,
  strataOf,
  type QueueSourceItem,
} from "../src/corpus/predicate-queue.js";

/**
 * What the annotators are handed, and what it is required not to say.
 *
 * The round this replaces answered 20 of 189 predicate questions and its own
 * blindness audit named three channels that carried answer-bearing content.
 * Two of them were fields on the queue row. So the assertions here are about
 * the ARTIFACT's bytes, not about the builder's return value: every structural
 * check below parses the serialized JSONL line, because the serialized line is
 * what a person opens.
 *
 * Two rules from this repository's standing conventions shape the rest.
 * First, an expectation derived from the thing under test proves only that the
 * code is deterministic -- so the source-order adjacency counts asserted below
 * were measured independently, in Python over the committed JSONL, before this
 * file existed, and are written here as literals. Second, a test exercising
 * only the default configuration cannot tell "reads the config" from
 * "hardcodes the default" -- so the salt and the seed are each exercised with
 * TWO values and the difference is asserted.
 */

const CORPUS_TEXT = readFileSync(PREDICATE_QUEUE_SOURCE, "utf8");
const ITEMS = loadCorpus(CORPUS_TEXT);
const ITEM_BY_ID = new Map(ITEMS.map((i) => [i.id, i]));

/**
 * Fixed for this test and DELIBERATELY not the salt of any emitted queue. The
 * emitted salt lives only in that queue's mapping file: committing it here
 * would put the inverse of a live handover artifact in the repository.
 */
const TEST_SALT = "b1a7f0d9c3e2481a5f6b0c7d8e9a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2";
const TEST_SEED = "corpus-predicate-queue-test-seed";
const OTHER_SALT = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0";
const OTHER_SEED = "corpus-predicate-queue-test-seed-2";

const BUILT = buildPredicateQueueArtifacts({ salt: TEST_SALT, seed: TEST_SEED });
const LINES = BUILT.queueJsonl.split("\n").filter((l) => l !== "");
const ROWS = LINES.map((l) => JSON.parse(l) as Record<string, unknown>);
const MAP = JSON.parse(BUILT.mapJson) as {
  what: string;
  salt: string;
  seed: string;
  rowIdAlgorithm: string;
  shuffleConstraint: string;
  itemCount: number;
  shuffle: { algorithm: string; constraint: string; seed: string };
  source: { path: string; sha256: string; items: number };
  artifact: { path: string; sha256: string; rows: number };
  rows: { position: number; rowId: string; itemId: string }[];
  adjacency: Record<
    string,
    { dimension: string; adjacentPairs: number; sameStratumPairs: number; rate: number; chanceRate: number }[]
  >;
};
const ITEM_ID_BY_ROW_ID = new Map(MAP.rows.map((r) => [r.rowId, r.itemId]));

/** The superseded queue, to keep the negative assertions from being vacuous. */
const OLD_QUEUE_ROWS = readFileSync(V2_QUEUE_PATH, "utf8")
  .split("\n")
  .filter((l) => l !== "")
  .map((l) => JSON.parse(l) as Record<string, unknown>);

/**
 * Every string the corpus uses to name a family, an entity type, a `neg:`
 * label, a constructed role, a carrier, a carrier stratum or an item.
 *
 * Rebuilt here from the raw JSONL with this file's own code rather than
 * imported from `forbiddenTokensOf`, so the scan compares the queue against the
 * corpus and not against the builder's idea of the corpus.
 */
const FORBIDDEN: string[] = (() => {
  const tokens = new Set<string>();
  for (const line of CORPUS_TEXT.split("\n")) {
    if (line.trim() === "") continue;
    const item = JSON.parse(line) as {
      id: string;
      gold?: { entityType?: string }[];
      meta?: {
        carrierId?: string;
        carrierStratum?: string;
        injections?: { family?: string; type?: string; dimensions?: Record<string, string> }[];
      };
    };
    tokens.add(item.id);
    if (item.meta?.carrierId !== undefined) tokens.add(item.meta.carrierId);
    if (item.meta?.carrierStratum !== undefined) tokens.add(item.meta.carrierStratum);
    for (const g of item.gold ?? []) if (g.entityType !== undefined) tokens.add(g.entityType);
    for (const inj of item.meta?.injections ?? []) {
      if (inj.family !== undefined) tokens.add(inj.family);
      if (inj.type !== undefined) tokens.add(inj.type);
      const role = inj.dimensions?.["constructedRole"];
      if (role !== undefined && role !== "none") tokens.add(role);
    }
  }
  return [...tokens].sort();
})();

function carrierOf(rowId: string): string {
  const itemId = ITEM_ID_BY_ROW_ID.get(rowId);
  if (itemId === undefined) throw new Error(`rowId ${rowId} is not in the mapping file`);
  return String(ITEM_BY_ID.get(itemId)!.meta!["carrierId"]);
}

function measurement(order: "sourceOrder" | "queueOrder" | "rowIdSortedOrder", dimension: string) {
  const found = MAP.adjacency[order]!.find((m) => m.dimension === dimension);
  if (found === undefined) throw new Error(`no ${dimension} in ${order}`);
  return found;
}

function longestRun(values: readonly string[]): number {
  let best = 0;
  let run = 0;
  for (let i = 0; i < values.length; i += 1) {
    run = i > 0 && values[i] === values[i - 1] ? run + 1 : 1;
    if (run > best) best = run;
  }
  return best;
}

describe("the predicate queue is one row per corpus item", () => {
  it("covers all 189, exactly once each", () => {
    expect(ITEMS).toHaveLength(189);
    expect(ROWS).toHaveLength(189);
    expect(new Set(MAP.rows.map((r) => r.itemId))).toEqual(new Set(ITEMS.map((i) => i.id)));
    expect(new Set(MAP.rows.map((r) => r.rowId)).size).toBe(189);
  });

  it("carries the message byte-identically", () => {
    for (const row of ROWS) {
      const itemId = ITEM_ID_BY_ROW_ID.get(String(row["rowId"]))!;
      expect(row["text"]).toBe(ITEM_BY_ID.get(itemId)!.text);
    }
  });

  it("puts every question the corpus asks into the round's scope", () => {
    // 189 message-predicate questions were built and 169 were never put. The
    // fix is arithmetic: one row per item, no row selected for anything.
    const withPredicateQuestion = ITEMS.filter((item) =>
      ((item.meta?.["labelQuestions"] ?? []) as { kind?: string }[]).some(
        (q) => q.kind === "message-predicate",
      ),
    );
    expect(withPredicateQuestion).toHaveLength(189);
    expect(ROWS).toHaveLength(withPredicateQuestion.length);
  });
});

describe("a row carries exactly two keys", () => {
  it("rowId and text, on every serialized line", () => {
    for (const [i, row] of ROWS.entries()) {
      expect(Object.keys(row).sort(), `line ${i + 1}`).toEqual(["rowId", "text"]);
    }
  });

  it("drops every field the superseded queue carried", () => {
    const dropped = ["questionId", "itemId", "kind", "question", "options", "span"];
    for (const key of dropped) {
      expect(ROWS.some((r) => key in r), `new queue must not carry ${key}`).toBe(false);
    }
    // Not vacuous: the queue this replaces carried five of the six on every row
    // and the sixth on its 20 contested-span rows.
    for (const key of ["questionId", "itemId", "kind", "question", "options"]) {
      expect(OLD_QUEUE_ROWS.every((r) => key in r), `old queue carried ${key}`).toBe(true);
    }
    expect(OLD_QUEUE_ROWS.filter((r) => "span" in r)).toHaveLength(20);
  });

  it("carries no predicate wording at all", () => {
    // TOLD-1 was a brief that paraphrased the predicate in the compiled IR's
    // own words. The brief is out of this artifact's reach; re-supplying any
    // predicate wording from the artifact is not.
    expect(BUILT.queueJsonl).not.toContain(BRIEF_PARAPHRASE_FRAGMENT);
    expect(BUILT.queueJsonl).not.toContain(PREDICATE_QUESTION);
    expect(BUILT.queueJsonl).not.toContain(PREDICATE_ID);
    // Vacuity control for the first two: the superseded queue contains one of
    // them on every row, and this file is reading the right string.
    expect(BRIEF_PARAPHRASE_FRAGMENT.length).toBeGreaterThan(40);
    const oldText = OLD_QUEUE_ROWS.map((r) => JSON.stringify(r)).join("\n");
    expect(oldText).toContain(PREDICATE_QUESTION);
  });
});

describe("no value in the queue names a family, a type or a stratum", () => {
  it("has a non-empty token list that the superseded queue does hit", () => {
    expect(FORBIDDEN.length).toBeGreaterThan(100);
    expect(FORBIDDEN).toContain("client-name");
    expect(FORBIDDEN).toContain("neg:sftp-endpoint");
    expect(FORBIDDEN).toContain("hard-negative");
    expect(FORBIDDEN).toContain("aadhaar-masked");
    // The superseded queue leaks item ids outside the message on every row.
    const oldOutsideText = OLD_QUEUE_ROWS.map((r) => {
      const { text: _text, ...rest } = r;
      return JSON.stringify(rest);
    }).join("\n");
    expect(FORBIDDEN.filter((t) => oldOutsideText.includes(t)).length).toBeGreaterThan(180);
  });

  it("leaks none of them outside the message", () => {
    // Restricted to the non-`text` part on purpose. MEASURED on the committed
    // corpus, the message text itself contains "client" (7 items),
    // "competitor" (8), "counterparty" (7) and the carrier ids "o14" and "o07"
    // as substrings -- all of them ordinary English or accidental, all of them
    // in bytes the annotator is meant to read. A whole-line substring scan
    // would fail on the message and prove nothing about the metadata.
    for (const [i, row] of ROWS.entries()) {
      const outsideText = JSON.stringify({ rowId: row["rowId"] });
      for (const token of FORBIDDEN) {
        expect(outsideText.includes(token), `line ${i + 1} leaks ${token}`).toBe(false);
      }
    }
  });

  it("has no value that IS one of them", () => {
    const forbidden = new Set(FORBIDDEN);
    for (const [i, row] of ROWS.entries()) {
      for (const value of Object.values(row)) {
        expect(forbidden.has(String(value)), `line ${i + 1} value ${String(value)}`).toBe(false);
      }
    }
  });

  it("makes the rowId structurally incapable of carrying one", () => {
    for (const row of ROWS) expect(String(row["rowId"])).toMatch(/^[0-9a-f]{16}$/);
  });

  it("finds no family name in the item ids either, which predicate-queue.ts cites this file for", () => {
    // `predicate-queue.ts:25` says of the corpus ids "all 189 match
    // `^inj-(o|hn)\d\d-\d$` or `^neg-(o|hn)\d\d$` and no family id appears in
    // any of them. Checked, in `corpus-predicate-queue.test.ts`." It was not:
    // there was no such assertion anywhere in this file. The claim is true, and
    // the citation is what was false.
    const shape = /^inj-(?:o|hn)\d\d-\d$|^neg-(?:o|hn)\d\d$/;
    for (const item of ITEMS) expect(item.id, item.id).toMatch(shape);
    // Non-vacuous, and the load-bearing half: the family ids the corpus DOES
    // carry are real strings, and none of them is a substring of any id.
    const families = new Set<string>();
    for (const item of ITEMS) {
      for (const inj of (item.meta?.["injections"] ?? []) as { family?: string }[]) {
        if (inj.family !== undefined) families.add(inj.family);
      }
    }
    expect(families.size).toBeGreaterThan(10);
    for (const family of families) {
      expect(ITEMS.filter((i) => i.id.includes(family)), family).toEqual([]);
    }
  });
});

describe("the emitter's own refusals, on inputs the committed corpus cannot produce", () => {
  // Both guards live inside the emitter and were exercised by nothing: the test
  // above builds from the one committed corpus at the one committed shape, so
  // `if (false && ...)` on either survived the suite.
  const TOKENS = ["inj-o01-0", "client-name", "hard-negative"];

  it("refuses a third key on a row", () => {
    const line = JSON.stringify({ rowId: "0123456789abcdef", text: "hello", itemId: "inj-o01-0" });
    expect(() => assertQueueCarriesNothingElse(`${line}\n`, TOKENS)).toThrow(/carries keys/);
  });

  it("refuses a forbidden token outside the message, and allows the same token inside it", () => {
    const leak = JSON.stringify({ rowId: "client-name-0000", text: "hello" });
    expect(() => assertQueueCarriesNothingElse(`${leak}\n`, TOKENS)).toThrow(/leaks "client-name"/);
    const inside = JSON.stringify({ rowId: "0123456789abcdef", text: "the client-name is fine here" });
    expect(() => assertQueueCarriesNothingElse(`${inside}\n`, TOKENS)).not.toThrow();
  });

  it("refuses a row whose whole message IS a token", () => {
    const bare = JSON.stringify({ rowId: "0123456789abcdef", text: "client-name" });
    expect(() => assertQueueCarriesNothingElse(`${bare}\n`, TOKENS)).toThrow(/text is exactly/);
  });

  it("names the offending line, so a 189-row emission says which row", () => {
    const good = JSON.stringify({ rowId: "0123456789abcdef", text: "fine" });
    const bad = JSON.stringify({ rowId: "0123456789abcdef", text: "fine", extra: 1 });
    expect(() => assertQueueCarriesNothingElse(`${good}\n${good}\n${bad}\n`, TOKENS)).toThrow(/line 3/);
  });

  it("builds its token list from the corpus, including every id, family, type, role and entity type", () => {
    // `forbiddenTokensOf` had no test of its own. The test above deliberately
    // rebuilds the list rather than importing it, which is right for
    // independence and leaves the emitter's copy uncovered -- so this compares
    // the two lists, which is the only comparison that can catch an empty one.
    const emitted = forbiddenTokensOf(ITEMS as unknown as QueueSourceItem[]);
    expect(emitted).toEqual(FORBIDDEN);
    expect(emitted.length).toBeGreaterThan(100);
    // And on a hand-built item, so the mapping from field to token is visible.
    expect(
      forbiddenTokensOf([
        {
          id: "inj-x01-0",
          text: "t",
          meta: {
            carrierId: "x01",
            carrierStratum: "hard-negative",
            injections: [{ family: "fam", type: "typ", dimensions: { constructedRole: "client" } }],
          },
        },
      ]),
    ).toEqual(["client", "fam", "hard-negative", "inj-x01-0", "typ", "x01"]);
    // `constructedRole: "none"` is not a leak and must not be listed.
    expect(
      forbiddenTokensOf([
        { id: "a", text: "t", meta: { injections: [{ dimensions: { constructedRole: "none" } }] } },
      ]),
    ).toEqual(["a"]);
  });
});

describe("the rowId is opaque and reversible only through the mapping file", () => {
  it("reverses through the map, in file order", () => {
    expect(MAP.rows.map((r) => r.rowId)).toEqual(ROWS.map((r) => String(r["rowId"])));
    for (const entry of MAP.rows) {
      expect(rowIdFor(entry.itemId, TEST_SALT)).toBe(entry.rowId);
    }
  });

  it("changes completely with the salt and not at all without it", () => {
    const again = buildPredicateQueueArtifacts({ salt: TEST_SALT, seed: TEST_SEED });
    expect(again.queueJsonl).toBe(BUILT.queueJsonl);

    const other = buildPredicateQueueArtifacts({ salt: OTHER_SALT, seed: TEST_SEED });
    const mine = new Set(ROWS.map((r) => String(r["rowId"])));
    const theirs = other.queueJsonl
      .split("\n")
      .filter((l) => l !== "")
      .map((l) => String((JSON.parse(l) as Record<string, unknown>)["rowId"]));
    expect(theirs).toHaveLength(189);
    expect(theirs.filter((id) => mine.has(id))).toEqual([]);
    // Same messages in the same order -- only the ids moved.
    const myTexts = ROWS.map((r) => String(r["text"]));
    const theirTexts = other.queueJsonl
      .split("\n")
      .filter((l) => l !== "")
      .map((l) => String((JSON.parse(l) as Record<string, unknown>)["text"]));
    expect(theirTexts).toEqual(myTexts);
  });

  it("refuses a salt too short to be worth having, AT the boundary", () => {
    expect(() => rowIdFor("inj-o01-0", "short")).toThrow(/at least 32/);
    // The 5-character case alone leaves the constant free to be anything above
    // 5 -- including 6, which is brute-forceable by hand against 189 known ids.
    // The error message cannot pin it either: it interpolates `salt.length`,
    // not the threshold.
    expect(() => rowIdFor("inj-o01-0", "x".repeat(31))).toThrow(/at least 32/);
    expect(rowIdFor("inj-o01-0", "x".repeat(32))).toMatch(/^[0-9a-f]{16}$/);
  });

  it("is the HMAC the mapping file says it is, computed independently here", () => {
    // The only previous check was `rowIdFor(itemId, salt) === entry.rowId`,
    // which calls the function that produced the value. MEASURED: replacing
    // HMAC-SHA256(salt, itemId) with SHA-256 over salt||itemId left the map
    // still claiming HMAC and passed the whole suite. So the expectation is
    // recomputed here from node:crypto, and the map's own prose is compared
    // with the implementation rather than with itself.
    for (const entry of MAP.rows) {
      const expected = createHmac("sha256", TEST_SALT).update(entry.itemId, "utf8").digest("hex").slice(0, 16);
      expect(entry.rowId, entry.itemId).toBe(expected);
    }
    expect(MAP.rowIdAlgorithm).toBe(ROW_ID_ALGORITHM);
    expect(MAP.rowIdAlgorithm).toContain("HMAC-SHA256");
    expect(MAP.rowIdAlgorithm).toContain(String(ROW_ID_HEX_LENGTH));
    // Non-vacuity: the salt-prefixed digest is a DIFFERENT id, so the two
    // constructions are distinguishable and this assertion is doing work.
    const notHmac = createHash("sha256").update(`${TEST_SALT}${MAP.rows[0]!.itemId}`, "utf8").digest("hex").slice(0, 16);
    expect(notHmac).not.toBe(MAP.rows[0]!.rowId);
  });
});

describe("the mapping file says what the emission actually did", () => {
  // The map is the round's only provenance artifact and nothing but
  // `rows[].rowId` and `rows[].itemId` was asserted. MEASURED: `position` could
  // carry the source index, `artifact.sha256` the SOURCE's digest, and
  // `rowIdAlgorithm`, `itemCount`, `source.path`, `source.sha256` and
  // `artifact.rows` could all be falsified at once, with a green suite.

  it("records the QUEUE position, not the source index", () => {
    expect(MAP.rows.map((r) => r.position)).toEqual(ROWS.map((_, i) => i));
    // Non-vacuous: the queue order is not the source order, so the two indices
    // genuinely differ. If they agreed, this assertion would prove nothing.
    const sourceIndexOf = new Map(ITEMS.map((item, i) => [item.id, i]));
    const disagreements = MAP.rows.filter((r) => sourceIndexOf.get(r.itemId) !== r.position);
    expect(disagreements.length).toBeGreaterThan(150);
  });

  it("hashes the artifact it emitted and the source it read, and they are different files", () => {
    const digest = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
    expect(MAP.artifact.sha256).toBe(digest(BUILT.queueJsonl));
    expect(MAP.source.sha256).toBe(digest(CORPUS_TEXT));
    expect(MAP.artifact.sha256).not.toBe(MAP.source.sha256);
    expect(MAP.artifact.path).toBe(QUEUE_FILENAME);
    expect(MAP.artifact.rows).toBe(189);
    expect(MAP.source.path).toBe("corpora/generated/injection-p-fin-v2.labelled.jsonl");
    expect(MAP.source.items).toBe(189);
    expect(MAP.itemCount).toBe(189);
    expect(MAP.what).toContain(QUEUE_FILENAME);
    expect(MAP.what).toContain("NEVER hand an annotator this file");
  });

  it("records the salt and the seed it was actually given, not the ones it likes", () => {
    expect(MAP.salt).toBe(TEST_SALT);
    expect(MAP.seed).toBe(TEST_SEED);
    expect(MAP.shuffle.seed).toBe(TEST_SEED);
    expect(MAP.shuffleConstraint).toBe(SHUFFLE_CONSTRAINT);
    expect(MAP.shuffle.constraint).toBe(SHUFFLE_CONSTRAINT);
    // A build with the other salt and the other seed records those instead, so
    // this is reading the option rather than a hardcoded default.
    const other = JSON.parse(
      buildPredicateQueueArtifacts({ salt: OTHER_SALT, seed: OTHER_SEED }).mapJson,
    ) as { salt: string; seed: string; shuffle: { seed: string } };
    expect(other.salt).toBe(OTHER_SALT);
    expect(other.seed).toBe(OTHER_SEED);
    expect(other.shuffle.seed).toBe(OTHER_SEED);
  });
});

describe("the shuffle breaks stratum adjacency", () => {
  it("reproduces the source-order grouping this round is fixing", () => {
    // Measured in Python over the committed JSONL, independently of this code.
    const expected: Record<string, number> = {
      carrier: 162,
      carrierStratum: 187,
      wave: 0,
      density: 24,
      register: 166,
      constructedPrediction: 156,
    };
    for (const dimension of STRATUM_DIMENSIONS) {
      const m = measurement("sourceOrder", dimension);
      expect(m.adjacentPairs, dimension).toBe(188);
      expect(m.sameStratumPairs, dimension).toBe(expected[dimension]);
    }
    // The run, not just the rate: seven consecutive items per carrier, six of
    // them spliced and the seventh the pristine original.
    expect(longestRun(ITEMS.map((i) => String(i.meta!["carrierId"])))).toBe(7);
  });

  it("leaves no two adjacent rows on the same carrier, measured on the emitted file", () => {
    const carriers = ROWS.map((r) => carrierOf(String(r["rowId"])));
    expect(longestRun(carriers)).toBe(1);
    expect(measurement("queueOrder", SHUFFLE_CONSTRAINT).sameStratumPairs).toBe(0);
  });

  it("drops the three dimensions the source order groups, to chance", () => {
    for (const dimension of ["carrier", "carrierStratum", "register"] as const) {
      const before = measurement("sourceOrder", dimension);
      const after = measurement("queueOrder", dimension);
      expect(after.rate, dimension).toBeLessThan(before.rate);
      // Chance is the floor, so "fell" is only meaningful against it. The
      // tolerance is set from a 60-seed sweep of this same builder, whose
      // widest per-dimension sd was 0.035; 0.12 is over three of those.
      expect(Math.abs(after.rate - after.chanceRate), dimension).toBeLessThanOrEqual(0.12);
    }
  });

  it("raises the two the source order interleaves, and says so", () => {
    // Source order walks waves 0..5 then the pristine item, so wave adjacency
    // is 0 by construction and density adjacency is below chance. A shuffle
    // cannot break carrier blocks and keep those; the honest report is that
    // both move UP toward chance. Neither is visible to an annotator as a
    // stratum: `wave` exists only in the item id, and `density` is how many
    // clauses were spliced, which the annotator is reading the message to
    // judge anyway.
    for (const dimension of ["wave", "density"] as const) {
      const before = measurement("sourceOrder", dimension);
      const after = measurement("queueOrder", dimension);
      expect(after.rate, dimension).toBeGreaterThan(before.rate);
      expect(Math.abs(after.rate - after.chanceRate), dimension).toBeLessThanOrEqual(0.12);
    }
  });

  it("does not group strata when the queue is sorted by rowId either", () => {
    // An annotator who sorts the file gets a different order; it must be no
    // more informative than the one they were given.
    const byRowId = measurement("rowIdSortedOrder", "carrier");
    expect(byRowId.rate).toBeLessThanOrEqual(0.1);
    for (const dimension of STRATUM_DIMENSIONS) {
      const m = measurement("rowIdSortedOrder", dimension);
      expect(Math.abs(m.rate - m.chanceRate), dimension).toBeLessThanOrEqual(0.12);
    }
  });

  it("is a function of the seed, and of nothing else", () => {
    const again = shuffledPositions(ITEMS as unknown as QueueSourceItem[], TEST_SEED);
    expect(again).toEqual([...BUILT.built.order]);
    const other = shuffledPositions(ITEMS as unknown as QueueSourceItem[], OTHER_SEED);
    expect(other).not.toEqual(again);
    expect([...other].sort((a, b) => a - b)).toEqual([...again].sort((a, b) => a - b));
    // And it is not the source order, nor its reverse.
    const source = ITEMS.map((_, i) => i);
    expect(again).not.toEqual(source);
    expect(again).not.toEqual([...source].reverse());
  });
});

describe("the builder fails loudly rather than emitting something weaker", () => {
  const item = (id: string, carrier: string): QueueSourceItem => ({
    id,
    text: `message ${id}`,
    meta: {
      carrierId: carrier,
      carrierRegister: "casual",
      density: 1,
      predicateConstruction: { constructed: false },
    },
  });

  it("refuses when zero same-carrier adjacency is impossible", () => {
    expect(() => shuffledPositions([item("a", "c1"), item("b", "c1")], "seed")).toThrow(
      /same-carrier adjacency/,
    );
  });

  it("refuses an item whose stratum it cannot read", () => {
    expect(() => strataOf({ id: "x", text: "t", meta: { carrierId: "c1" } })).toThrow(
      /predicateConstruction/,
    );
    expect(() =>
      strataOf({ id: "x", text: "t", meta: { predicateConstruction: { constructed: false } } }),
    ).toThrow(/meta\.carrierId/);
  });

  it("refuses a rowId collision rather than dropping a row", () => {
    const twins = [item("a", "c1"), item("a", "c2"), item("b", "c3")];
    expect(() => buildPredicateQueue(twins, { salt: TEST_SALT, seed: "s" })).toThrow(/collides/);
  });

  it("computes the chance rate a reader can check by hand", () => {
    // Four items, two carriers, two each. A uniformly random permutation puts
    // the same carrier in 2*1 + 2*1 = 4 of the 4*3 = 12 ordered pairs, so the
    // chance rate is 1/3. Worked out here rather than read back from the
    // function -- without this, `c * (c - 1)` and `c * c` are indistinguishable
    // to every other assertion in this file (MEASURED: the `c * c` mutant
    // survived the whole suite before this test existed).
    const four = [item("a", "c1"), item("b", "c1"), item("c", "c2"), item("d", "c2")];
    expect(adjacencyOf(four).find((m) => m.dimension === "carrier")!.chanceRate).toBeCloseTo(
      1 / 3,
      12,
    );
    // And on the corpus: 27 carriers of 7 items each is 27 * 7 * 6 = 1134 same-
    // carrier ordered pairs out of 189 * 188 = 35532.
    expect(measurement("sourceOrder", "carrier").chanceRate).toBeCloseTo(1134 / 35532, 12);
  });

  it("divides the same-stratum count by the number of ADJACENT PAIRS it reports", () => {
    // `rate` is `same / (n - 1)` and sits beside `adjacentPairs: n - 1`, and
    // nothing tied the two together: the other assertions on it are
    // inequalities and a 0.12 tolerance band, which absorb the 0.53% error at
    // n = 189. MEASURED: `same / n` survived the whole suite.
    for (const order of ["sourceOrder", "queueOrder", "rowIdSortedOrder"] as const) {
      for (const dimension of STRATUM_DIMENSIONS) {
        const m = measurement(order, dimension);
        expect(m.rate * m.adjacentPairs, `${order} ${dimension}`).toBeCloseTo(m.sameStratumPairs, 9);
      }
    }
    // And on a case small enough that n and n - 1 are obviously different: two
    // adjacent same-carrier pairs among three, so 2/3 and not 2/4.
    const three = adjacencyOf([
      { id: "a", text: "t", meta: { carrierId: "c", carrierRegister: "casual", density: 1, predicateConstruction: { constructed: false } } },
      { id: "b", text: "t", meta: { carrierId: "c", carrierRegister: "casual", density: 1, predicateConstruction: { constructed: false } } },
      { id: "c", text: "t", meta: { carrierId: "c", carrierRegister: "casual", density: 1, predicateConstruction: { constructed: false } } },
      { id: "d", text: "t", meta: { carrierId: "z", carrierRegister: "casual", density: 1, predicateConstruction: { constructed: false } } },
    ]).find((m) => m.dimension === "carrier")!;
    expect(three.adjacentPairs).toBe(3);
    expect(three.sameStratumPairs).toBe(2);
    expect(three.rate).toBeCloseTo(2 / 3, 12);
  });

  it("measures adjacency over the order it is given, not the order it likes", () => {
    const three = [item("a", "c1"), item("b", "c1"), item("c", "c2")];
    expect(adjacencyOf(three).find((m) => m.dimension === "carrier")!.sameStratumPairs).toBe(1);
    const spread = [three[0]!, three[2]!, three[1]!];
    expect(adjacencyOf(spread).find((m) => m.dimension === "carrier")!.sameStratumPairs).toBe(0);
  });
});
