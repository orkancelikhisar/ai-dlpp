import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import type { ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import type { DetectInput, PolicyIr, Segment } from "@sih/core";
import {
  BASELINE_B_SCHEMA,
  JUDGE_SCHEMA,
  MINIMUM_CANDIDATE_WORDS,
  WebLlmJudge,
  createBaselineB,
  resolveTier2Config,
} from "../src/index.js";
import { fakeEngine, predicateIr } from "./helpers.js";

/**
 * THE EXPERIMENT'S INDEPENDENT VARIABLE, pinned.
 *
 * The bake-off's whole claim is "same corpus, same model, same budget, DIFFERENT
 * METHOD", and the method is these two system turns: the compiled judge is shown
 * compiled predicates and one passage, Approach B is shown the policy document
 * and the whole message. Every other difference between the arms is already
 * asserted somewhere -- the schemas in `schema.test.ts`, the call counts in
 * `judge.test.ts` and `baselineB.test.ts`, the escalation in `escalate.test.ts`.
 * The instructions themselves were asserted almost nowhere: a control mutation
 * that rewrote `judge.ts`'s `SYSTEM_PROMPT` outright survived the entire suite.
 *
 * What DID exist, and what it covers, since it is easy to over- or under-state.
 * `baselineB.test.ts`'s "asks for a quote under the same rules the compiled arm
 * asks under" reads both system turns off real calls and requires every RULE
 * BULLET of each arm's to appear in the other's with two nouns substituted.
 * That is a real assertion and it is RELATIVE: it fires when one arm's bullets
 * drift and the other's do not. It fires in BOTH DIRECTIONS only as of this
 * round -- it used to normalise the JUDGE's side alone, which made a judge
 * bullet that had already drifted toward B's noun match B's unchanged: MEASURED,
 * renaming "passage" to "message" in all four of the judge's remaining rule
 * bullets left the whole 281-test suite green, and so did doing it to one. It
 * still cannot fire on the two arms drifting TOGETHER, and its filter -- lines
 * starting `"- "` or two spaces -- excludes each arm's two task sentences, the
 * JSON-only instruction, the wire-shape line and the `Rules:` header outright.
 * COUNTED off both real prompts: 12 non-blank lines each, 7 of them filtered in
 * and 5 excluded, and those 5 a side were pinned by nothing at all. (An earlier
 * version of this paragraph said seven and six; neither number was either
 * prompt's.)
 *
 * ## What is asserted here, and why not a snapshot
 *
 * A byte-for-byte snapshot fails on a typo fix, and a test that fails on a typo
 * fix gets updated without being read -- which is worse than no test, because it
 * launders the next real change through the same reflex. So each prompt is
 * pinned against a CONTRACT: a list of named properties, each a predicate over
 * one line, keyed on the words that carry the meaning rather than on the
 * sentence around them. The assertion runs BOTH ways:
 *
 *   - every clause must match some line, so DELETING or rewording an
 *     instruction past recognition fails, naming the property that went;
 *   - every non-blank line must match some clause, so ADDING an instruction
 *     fails until whoever added it says in the contract what it is for.
 *
 * The second half is what makes this a pin rather than a spot-check: the prompt
 * is accounted for line by line, and the account is in English next to the test.
 *
 * ## Keyed on POLARITY, not on nouns
 *
 * Every `carriedBy` below keys on the words that carry the instruction's SENSE
 * -- "quote must be copied", "never invent", "every span", "satisfies one of",
 * "is discarded", "from the passage" -- and not on the nouns those words sit
 * around. Keyed on the nouns alone, which is how this contract read when it
 * landed, a rule could be REVERSED while keeping every word the clause tested
 * for; and a reversal is a dropped instruction that leaves its keywords behind,
 * which is exactly the failure the first half above advertises catching.
 * MEASURED against that earlier version, each of these survived all 281 tests:
 * "Never invent one." -> "Invent one where none fits."; "quote must be copied"
 * -> "quote need not be copied" (both of those applied to BOTH arms, as a real
 * edit would be, so the cross-arm parity check could not fire either); and
 * "Report every span of the passage that satisfies one of the predicates." ->
 * "Report only spans of the passage that satisfies none of the predicates."
 * Each of the three fails here now.
 *
 * ## What it does NOT catch, stated so nobody reads it as more than it is
 *
 *   - **An ADDITION inside a matched line.** "Report every span ... that
 *     satisfies one of the predicates. Be generous." still satisfies that
 *     clause, because every phrase the clause keys on is still standing. Only
 *     whole added LINES are caught. What a rewording may NOT do is drop or
 *     reverse one of those phrases.
 *   - **Order.** The clauses are unordered; a prompt with the same instructions
 *     rearranged passes.
 *   - **Effect.** This is a claim about text, not about how a model answers it.
 *     Nothing here was measured on a GPU, and nothing here can be: two prompts
 *     satisfying the same contract can still produce different findings.
 *   - **The two arms against each other.** Deliberately: B is shown a document
 *     and asked about every entityType, the judge is shown predicates and a
 *     passage, and the contracts differ where the methods do. What each
 *     contract DOES pin is its own arm's noun -- the judge's clauses require
 *     "the passage" and B's "the message" wherever the rule names the thing
 *     quoted from -- so a one-sided rename fails here too, and not only in the
 *     cross-arm rule-bullet parity check, which lives in `baselineB.test.ts`
 *     and stays there.
 *
 * Both prompts are read off REAL calls rather than off the exported constants,
 * so a change that reaches the model has to reach these tests too.
 */

const MSG = "Please review the Northwind Traders renewal before Friday.";
const CONFIG = resolveTier2Config({});
const POLICY_ROOT = join(import.meta.dirname, "..", "..", "..", "policies");
const P_CORP = readFileSync(join(POLICY_ROOT, "p-corp.md"), "utf8");
const P_FIN = readFileSync(join(POLICY_ROOT, "p-fin.md"), "utf8");

/** The system message's content: message 0, which is where both arms keep their rules. */
function systemTurn(messages: readonly ChatCompletionMessageParam[]): string {
  const system = messages[0]!;
  expect(system.role).toBe("system");
  return typeof system.content === "string" ? system.content : JSON.stringify(system.content);
}

/** One judge call over one prose segment, and the system turn it sent. */
async function judgeSystemTurn(): Promise<string> {
  const engine = fakeEngine({ findings: [] });
  await new WebLlmJudge(engine, { budgetMs: 30_000 }).judge({
    text: MSG,
    segments: [{ kind: "prose", start: 0, end: MSG.length, text: MSG }],
    ir: predicateIr(),
    priorFindings: [],
    budgetMs: 30_000,
  });
  return systemTurn(engine.calls[0]!.messages);
}

/** One Approach-B call over one message, and the system turn it sent. */
async function baselineSystemTurn(policyText = P_CORP): Promise<string> {
  const engine = fakeEngine({
    requestedModelId: CONFIG.modelId,
    raw: JSON.stringify({ findings: [] }),
  });
  const input: DetectInput = {
    ir: predicateIr(),
    provider: "claude",
    text: MSG,
    config: { tier0: false, tier1: false, tier2: true },
  };
  await createBaselineB({ engine, config: CONFIG, policyText, budgetMs: 30_000 })(input);
  return systemTurn(engine.calls[0]!.messages);
}

/**
 * One property a prompt must state, and the line that carries it.
 *
 * `states` is the sentence a failure prints, so it has to name the PROPERTY and
 * not the wording: "the quote is copied character for character" survives a
 * rewrite of the sentence around it, and is what a reader has to re-establish
 * before changing that line.
 */
interface PromptClause {
  readonly states: string;
  readonly carriedBy: (line: string) => boolean;
}

/**
 * Every clause is stated, and every line states a clause.
 *
 * Blank lines are exempt from the second half and from nothing else: they are
 * layout, and a contract entry for "the prompt has a blank line here" would be
 * the byte-snapshot this file exists to avoid.
 */
function assertPinned(label: string, prompt: string, clauses: readonly PromptClause[]): void {
  const lines = prompt.split("\n").filter((line) => line.trim() !== "");
  expect(lines.length, `${label}: the system turn is empty`).toBeGreaterThan(0);
  for (const clause of clauses) {
    expect(
      lines.some((line) => clause.carriedBy(line)),
      `${label} no longer states: ${clause.states}`,
    ).toBe(true);
  }
  for (const line of lines) {
    expect(
      clauses.some((clause) => clause.carriedBy(line)),
      `${label} carries a line no clause in this contract accounts for, so an instruction was ` +
        `added without anyone saying what it is for: ${JSON.stringify(line)}`,
    ).toBe(true);
  }
}

/** Named things a prompt must NEVER say. */
interface PromptTaboo {
  readonly why: string;
  readonly pattern: RegExp;
}

function assertForbids(label: string, prompt: string, taboos: readonly PromptTaboo[]): void {
  for (const taboo of taboos) {
    expect(taboo.pattern.test(prompt), `${label}: ${taboo.why}`).toBe(false);
  }
}

/**
 * The compiled judge's contract.
 *
 * Its method in one sentence: shown COMPILED PREDICATES and ONE PASSAGE, and no
 * policy text of any kind. Every clause below is a thing a rewording must carry
 * across or the arm is measuring something else.
 */
const JUDGE_CONTRACT: readonly PromptClause[] = [
  {
    states: "the task is an AUDIT of ONE PASSAGE AGAINST declared POLICY PREDICATES",
    carriedBy: (l) =>
      /\baudit\b/i.test(l) &&
      /\bone passage\b/i.test(l) &&
      /\bagainst\b/i.test(l) &&
      /\bpredicates?\b/i.test(l),
  },
  {
    states: "the answer is EVERY span of that passage that SATISFIES ONE OF them",
    carriedBy: (l) =>
      /\bevery span\b/i.test(l) && /\bof the passage\b/i.test(l) && /satisfies one of/i.test(l),
  },
  {
    states: "the answer is JSON and NOTHING ELSE",
    carriedBy: (l) => /answer with JSON/i.test(l) && /nothing else/i.test(l),
  },
  {
    states: "the wire shape, spelled out (its keys are checked against JUDGE_SCHEMA below)",
    carriedBy: (l) => l.startsWith("{") && /"findings"/.test(l),
  },
  {
    states: "the lines that follow are RULES the answer must obey",
    carriedBy: (l) => /^rules:$/i.test(l.trim()),
  },
  {
    states: "the label MUST BE one of the ids listed, and is NEVER INVENTED",
    carriedBy: (l) =>
      /must be one of the ids/i.test(l) && /listed/i.test(l) && /never invent/i.test(l),
  },
  {
    states: "the quote MUST BE COPIED FROM THE PASSAGE, character for character",
    carriedBy: (l) =>
      /\bquote must be copied\b/i.test(l) &&
      /from the passage/i.test(l) &&
      /character for character/i.test(l),
  },
  {
    states: "...INCLUDING ITS punctuation AND capitalisation",
    carriedBy: (l) => /\bits punctuation and capitali[sz]ation\b/i.test(l),
  },
  {
    states: "QUOTE the WHOLE CLAUSE, and AT LEAST the ladder's word floor",
    carriedBy: (l) =>
      /^-\s*quote the whole clause\b/i.test(l.trim()) && /at least \d+ words/i.test(l),
  },
  {
    states: "a quote occurring MORE THAN ONCE in the passage IS DISCARDED, NOT GUESSED AT",
    carriedBy: (l) =>
      /more than once in the passage/i.test(l) &&
      /\bis discarded\b/i.test(l) &&
      /not guessed at/i.test(l),
  },
  {
    states: "NEVER quote anything that is NOT IN THE PASSAGE",
    carriedBy: (l) => /never quote/i.test(l) && /not in the passage/i.test(l),
  },
  {
    states: "IF NOTHING in the passage SATISFIES ANY predicate, in the parser's spelling",
    carriedBy: (l) =>
      /if nothing in the passage/i.test(l) &&
      /satisfies any predicate/i.test(l) &&
      /\{"findings":\[\]\}/.test(l),
  },
];

/**
 * Approach B's contract.
 *
 * Its method in one sentence: shown the POLICY DOCUMENT and the WHOLE MESSAGE,
 * and no compiled artifact of any kind. Nine of the twelve clauses are the
 * judge's with a noun changed, which is deliberate -- the two arms must ask for
 * a quote under the same rules or the comparison measures the prompt engineering
 * -- and the three that differ are the three the method differs on: what is
 * audited, what restricts it, and what the empty answer means.
 */
const BASELINE_CONTRACT: readonly PromptClause[] = [
  {
    states: "the task is an AUDIT of ONE MESSAGE against a POLICY DOCUMENT",
    carriedBy: (l) =>
      /\baudit\b/i.test(l) && /\bone message\b/i.test(l) && /policy document/i.test(l),
  },
  {
    states: "the answer is EVERY span of that message THE POLICY RESTRICTS",
    carriedBy: (l) =>
      /\bevery span\b/i.test(l) && /\bof the message\b/i.test(l) && /the policy restricts/i.test(l),
  },
  {
    states: "the answer is JSON and NOTHING ELSE",
    carriedBy: (l) => /answer with JSON/i.test(l) && /nothing else/i.test(l),
  },
  {
    states: "the wire shape, spelled out (its keys are checked against BASELINE_B_SCHEMA below)",
    carriedBy: (l) => l.startsWith("{") && /"findings"/.test(l),
  },
  {
    states: "the lines that follow are RULES the answer must obey",
    carriedBy: (l) => /^rules:$/i.test(l.trim()),
  },
  {
    states: "the label MUST BE one of the ids listed, and is NEVER INVENTED",
    carriedBy: (l) =>
      /must be one of the ids/i.test(l) && /listed/i.test(l) && /never invent/i.test(l),
  },
  {
    states: "the quote MUST BE COPIED FROM THE MESSAGE, character for character",
    carriedBy: (l) =>
      /\bquote must be copied\b/i.test(l) &&
      /from the message/i.test(l) &&
      /character for character/i.test(l),
  },
  {
    states: "...INCLUDING ITS punctuation AND capitalisation",
    carriedBy: (l) => /\bits punctuation and capitali[sz]ation\b/i.test(l),
  },
  {
    states: "QUOTE the WHOLE CLAUSE, and AT LEAST the ladder's word floor",
    carriedBy: (l) =>
      /^-\s*quote the whole clause\b/i.test(l.trim()) && /at least \d+ words/i.test(l),
  },
  {
    states: "a quote occurring MORE THAN ONCE in the message IS DISCARDED, NOT GUESSED AT",
    carriedBy: (l) =>
      /more than once in the message/i.test(l) &&
      /\bis discarded\b/i.test(l) &&
      /not guessed at/i.test(l),
  },
  {
    states: "NEVER quote anything that is NOT IN THE MESSAGE",
    carriedBy: (l) => /never quote/i.test(l) && /not in the message/i.test(l),
  },
  {
    states: "IF NOTHING in the message is RESTRICTED BY THE POLICY, in the parser's spelling",
    carriedBy: (l) =>
      /if nothing in the message/i.test(l) &&
      /restricted by the polic/i.test(l) &&
      /\{"findings":\[\]\}/.test(l),
  },
];

describe("the compiled judge's system prompt", () => {
  it("states every clause its method requires, and states nothing else", async () => {
    assertPinned("the judge's system turn", await judgeSystemTurn(), JUDGE_CONTRACT);
  });

  it("never names the other arm's vocabulary or the policy document", async () => {
    // The two directions the independent variable can collapse. A judge prompt
    // naming `entityType` is asking for a wire field its own grammar does not
    // mask for -- the instruction and the logit mask would disagree on every
    // call, which constrained decoding hides rather than surfaces. A judge
    // prompt mentioning the policy DOCUMENT is the arm being told the thing
    // only B is shown, which is the whole difference the bake-off measures.
    assertForbids("the judge's system turn", await judgeSystemTurn(), [
      {
        why: "it names B's wire field `entityType`, which JUDGE_SCHEMA does not declare",
        pattern: /entityType/,
      },
      {
        why: "it mentions the policy document, which only Approach B is shown",
        pattern: /policy document/i,
      },
    ]);
  });

  it("asks for the wire fields its own grammar masks for", async () => {
    // The judge's half of the check `baselineB.test.ts` already makes for B, and
    // it was missing: renaming `predicateId` to `entityType` in this prompt
    // alone left the whole suite green, because constrained decoding still
    // yields the schema's key and nothing compares the two.
    //
    // The three names are written out HERE rather than read off the schema
    // alone, so this file is the independent oracle: renaming the field in the
    // schema and the prompt at once has to fail here rather than agree with
    // itself.
    const shape = (await judgeSystemTurn()).split("\n").find((line) => line.startsWith("{"));
    expect(shape).toBeDefined();
    expect(JUDGE_SCHEMA.required).toEqual(["findings"]);
    expect(JUDGE_SCHEMA.properties.findings.items.required).toEqual([
      "predicateId",
      "quote",
      "confidence",
    ]);
    expect(shape).toContain('"findings"');
    for (const field of ["predicateId", "quote", "confidence"]) {
      expect(shape).toContain(`"${field}"`);
    }
  });

  it("sends the SAME instructions on a message-scope call, a segment call and a repair", async () => {
    // The property `buildMessages` claims in as many words -- "IDENTICAL in
    // shape for a segment and for the whole message" -- and the one that makes
    // the two scopes' resolvable and duplicate rates comparable within one arm.
    // A system turn that varied by scope would make a bake-off row's ladder
    // counters a mixture of two conditions, with nothing on the row saying so.
    //
    // The repair turn is in the same test because it is the other way the
    // instructions could drift mid-run: `repairMessage` appends a USER message,
    // so message 0 must still be this exact string on the retry.
    const text = "First line about Northwind Traders.\nSecond line about the renewal.";
    const segments: Segment[] = [
      { kind: "prose", start: 0, end: 35, text: text.slice(0, 35) },
      { kind: "prose", start: 35, end: text.length, text: text.slice(35) },
    ];
    const ir: PolicyIr = predicateIr({
      predicates: [
        { id: "m1", nlPredicate: "the message discusses an unannounced decision", scope: "message" },
        { id: "s1", nlPredicate: "the passage names a client organisation", scope: "segment" },
      ],
    });
    // Unparseable on every call, so the first call of each passage is followed
    // by its repair: 2 calls per passage across 1 message call and 2 segment
    // calls.
    const engine = fakeEngine({ raw: "not json at all" });
    await new WebLlmJudge(engine, { budgetMs: 30_000 }).judge({
      text,
      segments,
      ir,
      priorFindings: [],
      budgetMs: 30_000,
    });
    expect(engine.calls.length).toBe(6);
    // The USER turns differ -- that is the point of the calls -- so this is not
    // a vacuous "all six prompts are equal".
    const users = engine.calls.map((c) => JSON.stringify(c.messages.slice(1)));
    expect(new Set(users).size).toBeGreaterThan(1);
    const systems = engine.calls.map((c) => systemTurn(c.messages));
    expect(new Set(systems).size).toBe(1);
    expect(systems[0]).toBe(await judgeSystemTurn());
  });
});

describe("Approach B's system prompt", () => {
  it("states every clause its method requires, and states nothing else", async () => {
    assertPinned("B's system turn", await baselineSystemTurn(), BASELINE_CONTRACT);
  });

  it("never names the compiled arm's vocabulary", async () => {
    // B has no compiler and therefore no predicates: a prompt naming one is
    // either asking for a wire field `BASELINE_B_SCHEMA` does not declare, or
    // telling the control arm about the artifact whose absence is what makes it
    // the control.
    assertForbids("B's system turn", await baselineSystemTurn(), [
      {
        why: "it names the judge's wire field `predicateId`, which BASELINE_B_SCHEMA does not declare",
        pattern: /predicateId/,
      },
      {
        why: "it names predicates, which B has no compiler to produce",
        pattern: /\bpredicates?\b/i,
      },
    ]);
    expect(BASELINE_B_SCHEMA.properties.findings.items.required).toEqual([
      "entityType",
      "quote",
      "confidence",
    ]);
  });

  it("sends the same instructions whatever policy document the arm carries", async () => {
    // B's prompt is instructions + document + vocabulary + message, and only the
    // instructions are supposed to be fixed across arms. Two DIFFERENT documents
    // here -- `p-corp.md` and `p-fin.md`, 5,272 characters of the latter -- so
    // this cannot pass by the policy being identical, and the user turns are
    // asserted to differ for the same reason.
    expect(P_CORP).not.toBe(P_FIN);
    const corp = await baselineSystemTurn(P_CORP);
    const fin = await baselineSystemTurn(P_FIN);
    expect(corp).toBe(fin);
    // The document really did reach the model, on the turn where it belongs:
    // the system turn is the fixed half and the user turn is the varying one.
    const engine = fakeEngine({
      requestedModelId: CONFIG.modelId,
      raw: JSON.stringify({ findings: [] }),
    });
    await createBaselineB({ engine, config: CONFIG, policyText: P_FIN, budgetMs: 30_000 })({
      ir: predicateIr(),
      provider: "claude",
      text: MSG,
      config: { tier0: false, tier1: false, tier2: true },
    });
    const user = engine.calls[0]!.messages.slice(1).map((m) => String(m.content)).join("\n");
    expect(user).toContain(P_FIN.slice(0, 80));
    expect(systemTurn(engine.calls[0]!.messages)).not.toContain(P_FIN.slice(0, 80));
  });

  it("follows the ladder's word floor when it MOVES, rather than restating today's value", async () => {
    // The standing rule this project has broken twice: a test that only ever
    // exercises the DEFAULT cannot tell "reads the constant" from "hardcodes the
    // literal". `judge.test.ts` makes this assertion for the judge; B's copy of
    // the floor had no such test, and the cross-arm bullet comparison cannot
    // supply one -- both arms hardcoding 3 satisfies it.
    //
    // The floor matters because it is the ladder's, not the prompt's: `peel` in
    // `spans.ts` stops descending below `MINIMUM_CANDIDATE_WORDS`, so a prompt
    // asking for fewer words than the ladder will accept asks B for quotes the
    // harness then throws away, and counts them against B as unresolved.
    vi.resetModules();
    vi.doMock("../src/spans.js", async () => {
      const actual = await vi.importActual<typeof import("../src/spans.js")>("../src/spans.js");
      return { ...actual, MINIMUM_CANDIDATE_WORDS: 7 };
    });
    try {
      const { createBaselineB: rebuilt } = await import("../src/baselineB.js");
      const engine = fakeEngine({
        requestedModelId: CONFIG.modelId,
        raw: JSON.stringify({ findings: [] }),
      });
      await rebuilt({ engine, config: CONFIG, policyText: P_CORP, budgetMs: 30_000 })({
        ir: predicateIr(),
        provider: "claude",
        text: MSG,
        config: { tier0: false, tier1: false, tier2: true },
      });
      expect(systemTurn(engine.calls[0]!.messages)).toContain("at least 7 words");
      // Positive control: 7 is the mocked floor and not the real one, so this
      // cannot be passing because the prompt happens to say 7 anyway.
      expect(MINIMUM_CANDIDATE_WORDS).not.toBe(7);
    } finally {
      vi.doUnmock("../src/spans.js");
      vi.resetModules();
    }
  });

  it("asks for the floor the ladder actually enforces, read back out of the prompt", async () => {
    // The number is read OUT of the prompt rather than interpolated into the
    // expectation, so a prompt spelling it "three" fails here too. Paired with
    // the mocked-floor test above, the two separate "reads the constant" from
    // "hardcodes today's value".
    const asked = /at least (\S+) words/.exec(await baselineSystemTurn());
    expect(asked).not.toBeNull();
    expect(Number(asked![1])).toBe(MINIMUM_CANDIDATE_WORDS);
  });
});
