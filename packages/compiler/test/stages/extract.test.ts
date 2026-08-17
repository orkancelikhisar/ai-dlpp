import { describe, expect, it } from "vitest";
import {
  MIN_QUOTE_CHARS,
  extract,
  groundQuotes,
  normalizeForQuoteMatch,
} from "../../src/stages/extract.js";
import { FixtureLlmClient } from "../../src/llm/fixture.js";
import { loadTestFixtures } from "../fixtures/index.js";

const POLICY = [
  "# Test Policy",
  "",
  "§1 PAN card numbers must never be shared with any external service.",
  "§2 Client organisation names must be pseudonymized before transmission.",
].join("\n");

describe("normalizeForQuoteMatch", () => {
  it("collapses whitespace and newlines so re-wrapped quotes still match", () => {
    expect(normalizeForQuoteMatch("a  b\n c")).toBe(normalizeForQuoteMatch("a b c"));
  });

  it("preserves character identity (does not lowercase)", () => {
    expect(normalizeForQuoteMatch("PAN")).not.toBe(normalizeForQuoteMatch("pan"));
  });
});

describe("groundQuotes", () => {
  const candidates = [
    { id: "in-pan", sourceQuote: "PAN card numbers must never be shared" },
    { id: "invented", sourceQuote: "Blood type must never be shared" },
  ];

  it("keeps candidates whose quote appears verbatim in the document", () => {
    const { grounded } = groundQuotes(POLICY, candidates);
    expect(grounded.map((c) => c.id)).toEqual(["in-pan"]);
  });

  it("rejects candidates whose quote does not appear (anti-hallucination gate)", () => {
    const { rejected } = groundQuotes(POLICY, candidates);
    expect(rejected).toHaveLength(1);
    expect(rejected[0]!.id).toBe("invented");
    expect(rejected[0]!.reason).toMatch(/not found/i);
  });

  it("tolerates a quote re-wrapped across lines", () => {
    const { grounded } = groundQuotes(POLICY, [
      { id: "client-name", sourceQuote: "Client organisation names must be\n  pseudonymized" },
    ]);
    expect(grounded).toHaveLength(1);
  });

  it("rejects an empty or whitespace-only quote", () => {
    const { rejected } = groundQuotes(POLICY, [{ id: "blank", sourceQuote: "   " }]);
    expect(rejected[0]!.reason).toMatch(/empty/i);
  });

  it("rejects a fragment that would ground against any document", () => {
    // "PAN" is present in POLICY, so bare substring matching grounds it — while
    // demonstrating nothing. Taken to its limit the model could quote "the" and
    // clear the gate against every policy ever written.
    const { grounded, rejected } = groundQuotes(POLICY, [{ id: "fragment", sourceQuote: "PAN" }]);
    expect(grounded).toEqual([]);
    expect(rejected[0]!.reason).toMatch(/too short/i);
    expect(rejected[0]!.reason).toContain(String(MIN_QUOTE_CHARS));
  });

  it("reports a fragment distinctly from an invention, because the fix differs", () => {
    // Too-short means "quote the whole sentence"; not-found means "you made this
    // up". A short quote that is also absent reports too-short: quoting the
    // sentence is the first thing to try either way.
    const [short, absent] = groundQuotes(POLICY, [
      { id: "short", sourceQuote: "pan" },
      { id: "absent", sourceQuote: "Blood type must never be shared with anyone" },
    ]).rejected;
    expect(short!.reason).toMatch(/too short/i);
    expect(absent!.reason).toMatch(/not found/i);
  });

  it("accepts a quote exactly at the floor and rejects the character below it", () => {
    const doc = `A clause reading ${"z".repeat(40)} and no more.`;
    expect(
      groundQuotes(doc, [{ id: "at-floor", sourceQuote: "z".repeat(MIN_QUOTE_CHARS) }]).grounded,
    ).toHaveLength(1);
    expect(
      groundQuotes(doc, [{ id: "under", sourceQuote: "z".repeat(MIN_QUOTE_CHARS - 1) }]).rejected,
    ).toHaveLength(1);
  });
});

/**
 * A second document, because the fixture key is a hash of the request and the
 * request carries the document: a fixture in which the model hallucinates has to
 * answer a different document than the honest one.
 */
const SPARSE_POLICY = [
  "# Sparse Policy",
  "",
  "§1 Client organisation names must be pseudonymized before transmission.",
].join("\n");

describe("extract", () => {
  it("calls the model once and grounds every candidate it returns", async () => {
    // The request shape — {system: SYSTEM, user: document, schemaName: "Extraction",
    // maxTokens: 16000} — needs no assertion here because it is already pinned
    // harder than an assertion could pin it: requestHash covers all four fields,
    // so any change to SYSTEM, the user text, the schema name or the token budget
    // misses the committed fixture and fails this test by name.
    const client = new FixtureLlmClient(loadTestFixtures());
    const result = await extract(client, POLICY);
    expect(result.entityTypes.map((e) => e.id)).toContain("in-pan");
    expect(result.rejected).toEqual([]);
  });

  it("drops a candidate the model invented rather than compiling it", async () => {
    // The gate has to be wired into extract, not merely exist beside it: with a
    // fixture whose every quote grounds, an extract that returned the model's
    // raw output would pass every other test in this file (verified by mutation).
    const result = await extract(new FixtureLlmClient(loadTestFixtures()), SPARSE_POLICY);
    expect(result.entityTypes.map((e) => e.id)).toEqual(["client-name"]);
    expect(result.rejected.map((r) => `${r.kind} ${r.id}`)).toContain("entityType blood-type");
    expect(result.rejected.every((r) => /not found/i.test(r.reason))).toBe(true);
  });

  it("names a rejected action by a synthesized id an auditor can read", async () => {
    // Actions carry no id of their own, and this string is what a human sees in
    // the compilation report's rejected-candidates section — an array index there
    // would name nothing.
    const result = await extract(new FixtureLlmClient(loadTestFixtures()), SPARSE_POLICY);
    expect(result.actions).toHaveLength(1);
    expect(result.rejected.map((r) => `${r.kind} ${r.id}`)).toContain("action blood-type:default");
  });
});
