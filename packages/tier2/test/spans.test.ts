import { describe, expect, it } from "vitest";
import { MINIMUM_CANDIDATE_WORDS, buildFoldMap, resolveQuote } from "../src/spans.js";

const MSG = "Hi team,\n\nAcme  Corp is our biggest client. Please don't tell Acme Corp's rival.";

/**
 * The Plan 5 probe message, verbatim. Every model quote asserted anywhere below
 * was produced against exactly this string, so it is defined once: two copies
 * of a 336-character fixture are two things that can drift apart.
 */
const PROBE = "Hey team - quick one before the board review with Halcyon. Priya Sharma "
  + "(priya.sharma@northwind-traders.io) is asking whether we can hold the Halcyon renewal "
  + "until Q3. Her base is 142,000 GBP and Halcyon's own counsel flagged clause 7. Can someone "
  + "pull the AWS key AKIAIOSFODNN7EXAMPLE out of the staging config before we send anything?";

describe("buildFoldMap", () => {
  it("collapses whitespace runs and casefolds", () => {
    // "Acme  Corp" (two spaces) and a newline run both fold to single spaces,
    // so a model that reflows whitespace still matches.
    expect(buildFoldMap(MSG).folded).toContain("acme corp is our biggest client");
  });

  it("maps a whitespace run to the run's FIRST index, as the doc comment says", () => {
    // Nothing else here would notice: no candidate ever starts or ends on a
    // folded space, so this entry is never read for an offset today. It is
    // pinned because the `end` arithmetic in resolveQuote is derived from it --
    // "one code unit past map[last]" is only sound while a space's map entry
    // means the START of its run.
    const { folded, map } = buildFoldMap("a\n\n\n\nb");
    expect(folded).toBe("a b");
    expect(map).toEqual([0, 1, 5]);
  });
});

describe("resolveQuote", () => {
  it("rung 1: resolves an exact quote and slices the ORIGINAL text", () => {
    const r = resolveQuote(MSG, "Acme  Corp is our biggest client");
    expect(r?.rung).toBe(1);
    expect(MSG.slice(r!.start, r!.end)).toBe(r!.text);
  });

  it("rung 1: resolves a quote whose whitespace and case differ from the message", () => {
    // The model reflowed the double space and lowercased. The span must still
    // land on the ORIGINAL characters, double space included.
    const r = resolveQuote(MSG, "acme corp is our biggest client");
    expect(r?.rung).toBe(1);
    expect(MSG.slice(r!.start, r!.end)).toBe("Acme  Corp is our biggest client");
  });

  it("refuses an ambiguous quote rather than picking an occurrence", () => {
    // "Acme Corp" appears twice. Guessing one is a coin flip that produces a
    // finding pointing at text the model was not talking about -- and it would
    // slice cleanly, so nothing downstream could object.
    const r = resolveQuote(MSG, "Acme Corp");
    expect(r).toBeUndefined();
  });

  it("rung 2: falls back to the longest unique head of the quote", () => {
    // Models append or alter trailing punctuation. Measured: one returned a
    // quote with an added "?" and five words dropped.
    const r = resolveQuote(MSG, "Acme  Corp is our biggest client, obviously!!");
    expect(r?.rung).toBe(2);
    expect(MSG.slice(r!.start, r!.end)).toBe("Acme  Corp is our biggest client");
  });

  it("returns undefined rather than a guess when nothing matches", () => {
    expect(resolveQuote(MSG, "a sentence that is nowhere in this message")).toBeUndefined();
  });

  it("resolves correctly when an emoji precedes the quote", () => {
    // UTF-16 again. Offsets are JS string indices throughout this system.
    const msg = "\u{1F389} Acme Corp is our client";
    const r = resolveQuote(msg, "Acme Corp is our client");
    expect(r).toBeDefined();
    expect(msg.slice(r!.start, r!.end)).toBe("Acme Corp is our client");
  });
});

// ---------------------------------------------------------------------------
// Mis-location, not incoherence.
//
// A test that checks `MSG.slice(start, end) === text` is satisfied by
// construction: the implementation DEFINES text as that slice, and so does
// core (packages/core/src/detect/orchestrator.ts:98 throws on the same
// comparison). A span that points at the wrong place is therefore invisible to
// it -- it slices cleanly and core accepts it. The tests below assert exact
// offsets against independently computed ground truth (indexOf on the
// untouched message), which is the only thing that catches drift.
// ---------------------------------------------------------------------------

/** Ground truth for a quote that occurs verbatim exactly once. */
const truthSpan = (msg: string, sub: string): { start: number; end: number } => {
  const start = msg.indexOf(sub);
  if (start === -1 || msg.indexOf(sub, start + 1) !== -1) {
    throw new Error(`fixture bug: ${JSON.stringify(sub)} is not uniquely present`);
  }
  return { start, end: start + sub.length };
};

describe("buildFoldMap: the fold must not change string length per code unit", () => {
  it("keeps map and folded in lockstep across a casefold that EXPANDS", () => {
    // Measured on this machine (Node 26): of all 65,536 BMP code units, exactly
    // one has a toLowerCase() longer than itself -- U+0130 LATIN CAPITAL LETTER
    // I WITH DOT ABOVE, which lowercases to "i" + U+0307 COMBINING DOT ABOVE,
    // two code units. Pushing that into `folded` while pushing one entry into
    // `map` desynchronizes the two for the whole rest of the string, so every
    // later index maps one place short.
    const msg = "İstanbul office: the merger codename is Bluebird.";
    const { folded, map } = buildFoldMap(msg);
    expect(map).toHaveLength(folded.length);
  });

  it("maps each folded index onto an original character that folds to it", () => {
    // A stronger statement than "in range": map[i] must be the index the folded
    // character actually CAME FROM. Checked over text carrying every fold rule.
    const msg = "İzmir — “Acme’s”  team\n\n🎉 pay: 142,000";
    const { folded, map } = buildFoldMap(msg);
    for (let i = 0; i < folded.length; i += 1) {
      const original = msg[map[i]!]!;
      if (folded[i] === " ") expect(/\s/.test(original)).toBe(true);
      else expect(buildFoldMap(original).folded).toBe(folded[i]);
    }
  });

  it("folds smart punctuation the model normalised away", () => {
    // The distinguishing case, and the one the fold exists for. Putting the
    // SAME curly characters in message and quote proves nothing: both fold
    // identically, so the rule is a no-op and deleting PUNCTUATION_FOLD leaves
    // the suite green. Here only the MESSAGE is curly.
    const msg = "Priya said “the merger codename is Bluebird” — that stays internal.";
    const r = resolveQuote(msg, '"the merger codename is Bluebird"');
    expect(r?.rung).toBe(1);
    expect({ start: r!.start, end: r!.end })
      .toEqual(truthSpan(msg, "“the merger codename is Bluebird”"));
    // Sliced from the original, so the user's own curly quotes come back.
    expect(r!.text).toBe("“the merger codename is Bluebird”");
  });
});

describe("resolveQuote: offsets must land on the ground truth, not merely slice", () => {
  it("does not drift when a casefold-expanding character precedes the quote", () => {
    const msg = "İstanbul office: the merger codename is Bluebird and stays internal.";
    const quote = "the merger codename is Bluebird";
    const r = resolveQuote(msg, quote);
    expect(r).toBeDefined();
    // Ground truth from the untouched message, computed without the fold map.
    expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(msg, quote));
    expect(r!.text).toBe(quote);
  });

  it("puts rung 1 exactly on the ground truth for every fold rule at once", () => {
    const msg = "İzmir team — the codename is  Bluebird’s Ledger, keep it internal.";
    const quote = "the codename is  Bluebird’s Ledger";
    const r = resolveQuote(msg, quote);
    expect(r?.rung).toBe(1);
    expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(msg, quote));
  });

  it("ends the span before the whitespace run, not one character into it", () => {
    // buildFoldMap maps a whitespace RUN to the run's FIRST index, so an `end`
    // derived from a folded space would be runStart + 1 -- one character into
    // the run, whatever the run's length -- when the correct end is runStart.
    // Reachable only if a candidate is allowed to end on whitespace; this pins
    // that it never is. VERIFIED by double mutation: drop foldQuote's trim AND
    // the folded-space guard in `at` together and this quote resolves to
    // {0, 25} -- "The codename is Bluebird\n" -- which slices cleanly and which
    // core would accept. Either one alone still lands on {0, 24}.
    //
    // The quote is padded at BOTH ends: the leading padding is what makes the
    // trim itself load-bearing, since a needle keeping it matches nothing and
    // the whole quote is refused.
    const msg = "The codename is Bluebird\n\n\n\n\nand that is confidential.";
    const quote = "   the codename is bluebird   ";
    const r = resolveQuote(msg, quote);
    expect(r).toBeDefined();
    expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(msg, "The codename is Bluebird"));
    expect(/\s$/.test(r!.text)).toBe(false);
  });

  it("ends a PEELED candidate before the whitespace run too", () => {
    // The same hazard one rung down, and the only shape that reaches it: three
    // words, then a whitespace run, then a divergence. The peel walks the
    // needle back through the run's folded space, so a peel that kept that
    // space would hand `at` a candidate ending on one. VERIFIED by double
    // mutation: drop the peel's space trim AND the folded-space guard and this
    // resolves to {0, 23} -- one character into the run.
    const msg = "The merger codename is\n\n\n\n\nBluebird, obviously.";
    const r = resolveQuote(msg, "The merger codename is elsewhere");
    expect(r?.rung).toBe(2);
    expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(msg, "The merger codename is"));
  });
});

describe("resolveQuote: rung 2 must not shed the secret's tail", () => {
  // Secrets end in characters that are neither letters nor digits. A rung 2
  // that sheds a whole WORD to drop an appended full stop deletes the last word
  // of the quote -- which, for a quote shaped "<label> <secret>", IS the
  // secret. A rung 2 that greedily strips trailing non-word characters instead
  // keeps the word but drops the secret's own final character, and applyActions
  // rewrites by offset, so that character ships in the message. MEASURED by
  // restoring a greedy strip: the first four rows below then lose the secret's
  // final character. The last row is the control the strip cannot damage -- a
  // digit-terminated PIN -- and it is here so the table is not selected.
  const TAILS: ReadonlyArray<readonly [string, string, string]> = [
    ["base64 padding",
     "Rotate creds: the secret is wJalrXUtnFEMI/K7MDENG/bPxRfiCY= before Friday.",
     "the secret is wJalrXUtnFEMI/K7MDENG/bPxRfiCY=."],
    ["a password ending in '!'",
     "The vault password is Tr0ub4dor&3! and it expires soon.",
     "The vault password is Tr0ub4dor&3!."],
    ["a signing key ending in '_'",
     "The signing key is abc-DEF_123_ and we rotate it monthly.",
     "The signing key is abc-DEF_123_."],
    ["an IBAN inside parentheses",
     "Her account is (GB29NWBK60161331926819) and we pay on Fridays.",
     "Her account is (GB29NWBK60161331926819)."],
    ["a digit-terminated PIN",
     "Rotate it: the deploy pin is 8421 before the release goes out.",
     "the deploy pin is 8421."],
  ];

  it.each(TAILS)("keeps the whole secret when the model appends a full stop: %s", (_l, msg, q) => {
    const r = resolveQuote(msg, q);
    expect(r?.rung).toBe(2);
    // The quote minus the appended stop, which is the entire secret.
    expect(r!.text).toBe(q.slice(0, -1));
    expect(msg.slice(r!.start, r!.end)).toBe(r!.text);
  });

  it("resolves where a greedy trailing strip would have to REFUSE", () => {
    // Shortening past the secret's own last character does not merely lose a
    // character, it can lose the resolution: "the deploy token is s3cr3t"
    // occurs twice, so a strip that removes the "=" makes a unique quote
    // ambiguous and the finding is dropped entirely.
    const msg = "Prod: the deploy token is s3cr3t= and staging: the deploy token is s3cr3t. "
      + "Rotate both.";
    const r = resolveQuote(msg, "the deploy token is s3cr3t=.");
    expect(r?.rung).toBe(2);
    expect(r!.text).toBe("the deploy token is s3cr3t=");
    expect(msg.indexOf("the deploy token is s3cr3t")).not
      .toBe(msg.lastIndexOf("the deploy token is s3cr3t"));
  });

  it("keeps the secret inside the span when the model appends a full stop", () => {
    // The measured perturbation. Phi-4-mini returned a quote with an appended
    // "?" in this very corpus (webllm-probe/out-e5.json).
    const r = resolveQuote(PROBE, "the AWS key AKIAIOSFODNN7EXAMPLE.");
    expect(r).toBeDefined();
    expect(r!.text).toContain("AKIAIOSFODNN7EXAMPLE");
    expect(PROBE.slice(r!.start, r!.end)).toBe(r!.text);
  });

  it("keeps the trailing unit when the model appends a full stop to a salary", () => {
    const r = resolveQuote(PROBE, "Her base is 142,000 GBP.");
    expect(r).toBeDefined();
    expect(r!.text).toBe("Her base is 142,000 GBP");
    expect(r!.rung).toBe(2);
  });

  it("recovers all but one character when the model MANGLES the secret's last one", () => {
    // The residual, pinned honestly rather than hidden. Peeling one code point
    // at a time does not cure a perturbation INSIDE the last word: it stops at
    // the longest prefix that still matches, which is one character short of
    // the message's version of the credential. 19 of 20 characters is a large
    // improvement on the 0 of 20 a word-level descent recovers, and it is still
    // one character of a live AWS key left in the message.
    const r = resolveQuote(PROBE, "the AWS key AKIAIOSFODNN7EXAMPLF");
    expect(r?.rung).toBe(2);
    expect(r!.text).toBe("the AWS key AKIAIOSFODNN7EXAMPL");
    expect(PROBE.slice(r!.end)).toMatch(/^E out of the staging config/);
  });

  it("still refuses when a shortened candidate is ambiguous", () => {
    // Peeling must not become a licence to guess.
    const msg = "The client roster is secret. The client roster is public.";
    expect(resolveQuote(msg, "The client roster is!!")).toBeUndefined();
  });
});

describe("resolveQuote: the floors that stop a model scoring by accident", () => {
  it("refuses a two-word prefix even when it is unique", () => {
    // Measured ambiguity: one-word CAPITALIZED quotes are non-unique 22-51% of
    // the time, two-word 7-38%. A two-word prefix that happens to be unique
    // here will not be in the next message, so accepting it buys a number that
    // does not hold.
    const msg = "The Zephyr protocol alpha is our internal codename for the migration.";
    expect(msg.indexOf("Zephyr protocol")).toBeGreaterThan(-1);
    expect(resolveQuote(msg, "Zephyr protocol beta gamma delta")).toBeUndefined();
  });

  it("accepts a candidate at exactly the floor, so raising the floor is visible", () => {
    // The floor was pinned only from below: every test above still passed with
    // MINIMUM_CANDIDATE_WORDS raised to 4, because they all refuse anyway. This
    // resolves at exactly three words and fails if the floor moves up.
    expect(MINIMUM_CANDIDATE_WORDS).toBe(3);
    const msg = "Halcyon's own counsel flagged clause 7 in the redline yesterday.";
    const r = resolveQuote(msg, "Halcyon's own counsel!!");
    expect(r?.rung).toBe(2);
    expect(r!.text).toBe("Halcyon's own counsel");
    expect(r!.text.split(" ")).toHaveLength(MINIMUM_CANDIDATE_WORDS);
  });

  it("refuses when a >=3-word prefix is ambiguous rather than shortening further", () => {
    const msg = "The client roster is secret. The client roster is public.";
    expect(resolveQuote(msg, "The client roster is confidential forever")).toBeUndefined();
  });

  it("refuses an empty or whitespace-only quote", () => {
    expect(resolveQuote(MSG, "")).toBeUndefined();
    expect(resolveQuote(MSG, "   \n\t  ")).toBeUndefined();
  });

  it("never searches for the quote's individual words", () => {
    // There is deliberately no rung that reassembles a quote from words found
    // anywhere in the message. Every word below is present; the phrase is not.
    const msg = "Acme is our client. The budget is secret. Nothing else matters here.";
    expect(resolveQuote(msg, "secret client budget Acme")).toBeUndefined();
  });
});

describe("resolveQuote: UTF-16 boundaries", () => {
  const isHigh = (c: number): boolean => c >= 0xd800 && c <= 0xdbff;
  const isLow = (c: number): boolean => c >= 0xdc00 && c <= 0xdfff;

  it("never splits a surrogate pair", () => {
    // Offsets are UTF-16 code unit indices, so a span boundary CAN fall between
    // the halves of an astral character. Half a pair is a lone surrogate: it
    // still satisfies text === slice(start, end), so core would accept it and
    // the vault would store a broken string.
    const msg = "Report 🎉 the codename is Bluebird 🎉 internally.";
    for (const q of ["the codename is Bluebird", "\uDF89 the codename is Bluebird",
                     "the codename is Bluebird \uD83C", "report 🎉 the codename"]) {
      const r = resolveQuote(msg, q);
      if (r === undefined) continue;
      expect(isLow(msg.charCodeAt(r.start)) && isHigh(msg.charCodeAt(r.start - 1))).toBe(false);
      expect(isHigh(msg.charCodeAt(r.end - 1)) && isLow(msg.charCodeAt(r.end))).toBe(false);
    }
  });

  it("keeps descending when an exact match would split a pair, instead of giving up", () => {
    // Found by differential fuzzing, not by review. The quote ends in a lone
    // HIGH surrogate, which matches the high half of the message's second 🎉 --
    // a unique hit that `at` then correctly refuses. Returning that refusal
    // ends the ladder and loses a quote whose head resolves perfectly one code
    // point further down.
    const msg = "Report 🎉 the codename is Bluebird 🎉 internally.";
    const r = resolveQuote(msg, "the codename is Bluebird \uD83C");
    expect(r?.rung).toBe(2);
    expect({ start: r!.start, end: r!.end })
      .toEqual(truthSpan(msg, "the codename is Bluebird"));
  });

  it("keeps descending when a rung-2 candidate would split a pair", () => {
    // The same fall-through one rung down. Two consecutive lone high surrogates
    // are what it takes to reach: the first peel leaves a candidate that still
    // ends on one, matching the high half of a real pair in the message.
    const msg = "Report 🎉 the codename is Bluebird 🎉🎉 internally.";
    const r = resolveQuote(msg, "the codename is Bluebird \uD83C\uD83C");
    expect(r?.rung).toBe(2);
    expect(r!.text).toBe("the codename is Bluebird");
  });

  it("peels a whole astral character, not one surrogate half", () => {
    // The peel steps by code POINT. With a step of one code UNIT, the candidate
    // after the first step still ends on the high half of the model's astral
    // character, and that half matches the LONE high surrogate in this message
    // -- a unique hit `at` cannot refuse, because there is no pair to split.
    // The span then ends on a character the model never quoted.
    const msg = "The codename is Bluebird \uD801 and internal.";
    const r = resolveQuote(msg, "The codename is Bluebird \u{10400}");
    expect(r?.rung).toBe(2);
    expect({ start: r!.start, end: r!.end })
      .toEqual(truthSpan(msg, "The codename is Bluebird"));
    expect(r!.text).toBe("The codename is Bluebird");
  });

  it("resolves a quote containing an astral character at the exact offsets", () => {
    const msg = "Ship it 🚀 today: the codename is Bluebird 🚀 and confidential.";
    const quote = "the codename is Bluebird 🚀 and confidential";
    const r = resolveQuote(msg, quote);
    expect(r?.rung).toBe(1);
    expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(msg, quote));
  });
});

// ---------------------------------------------------------------------------
// Real model output.
//
// Every quote below was produced by a real model in the Plan 5 feasibility
// probe (scratchpad/webllm-probe/out-e1-q35.json, out-e2-q35.json, out-e5.json,
// out-e6.json) against exactly the PROBE message. They are the whole distinct
// set the corpus contains -- nothing was selected for making the ladder look
// good, and the one non-verbatim quote is included.
// ---------------------------------------------------------------------------

describe("resolveQuote: quotes real models actually returned", () => {
  it("resolves each uniquely-occurring verbatim quote at rung 1, on the ground truth", () => {
    for (const q of ["AKIAIOSFODNN7EXAMPLE", "Her base is 142,000 GBP", "Halcyon renewal",
                     "Halcyon's own counsel flagged clause 7"]) {
      const r = resolveQuote(PROBE, q);
      expect(r?.rung, q).toBe(1);
      expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(PROBE, q));
    }
  });

  it("refuses the one-word quote Phi-4-mini returned seventeen times in one call", () => {
    // Counted from out-e5.json: on each of its three tier2-shaped calls
    // Phi-4-mini emitted 19 quotes, 17 of them the bare "Halcyon" and 2 the
    // two-word "Halcyon renewal". "Halcyon" occurs three times in the message,
    // so picking one would be a coin flip -- and duplicate-heavy output like
    // that is exactly the case where a guessed span would inflate an arm.
    expect(PROBE.split("Halcyon").length - 1).toBe(3);
    expect(resolveQuote(PROBE, "Halcyon")).toBeUndefined();
  });

  it("recovers the non-verbatim quote at rung 2 without losing the credential", () => {
    // Phi-4-mini returned exactly this string in out-e5.json, on all three of
    // its Approach-B-shaped calls: four trailing words dropped and the
    // sentence's question mark pulled up onto "config".
    const q = "Can someone pull the AWS key AKIAIOSFODNN7EXAMPLE out of the staging config?";
    expect(PROBE.includes(q)).toBe(false);
    const r = resolveQuote(PROBE, q);
    expect(r?.rung).toBe(2);
    expect(r!.text).toContain("AKIAIOSFODNN7EXAMPLE");
    // The span must start where the model's quote starts, not somewhere that
    // merely slices cleanly.
    expect(r!.start).toBe(PROBE.indexOf("Can someone pull"));
    expect(PROBE.slice(r!.start, r!.end)).toBe(r!.text);
  });
});
