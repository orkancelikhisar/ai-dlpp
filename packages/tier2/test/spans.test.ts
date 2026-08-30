import { describe, expect, it } from "vitest";
import { buildFoldMap, resolveQuote } from "../src/spans.js";

const MSG = "Hi team,\n\nAcme  Corp is our biggest client. Please don't tell Acme Corp's rival.";

describe("buildFoldMap", () => {
  it("maps every folded index back to an original index", () => {
    const { folded, map } = buildFoldMap(MSG);
    expect(map).toHaveLength(folded.length);
    for (let i = 0; i < folded.length; i += 1) {
      expect(map[i]).toBeGreaterThanOrEqual(0);
      expect(map[i]).toBeLessThan(MSG.length);
    }
  });

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

  it("rung 2: falls back to the longest unique prefix of the quote", () => {
    // Models append or alter trailing punctuation. Measured: one returned a
    // quote with an added "?" and five words dropped.
    const r = resolveQuote(MSG, "Acme  Corp is our biggest client, obviously!!");
    expect(r?.rung).toBe(2);
    expect(MSG.slice(r!.start, r!.end)).toBe(r!.text);
  });

  it("returns undefined rather than a guess when nothing matches", () => {
    expect(resolveQuote(MSG, "a sentence that is nowhere in this message")).toBeUndefined();
  });

  it("never returns a span whose text disagrees with its offsets", () => {
    // The invariant core enforces by throwing. Checked here over every rung so
    // a rung added later cannot violate it quietly.
    for (const q of ["Acme  Corp is our biggest client", "acme corp is our biggest client",
                     "Acme  Corp is our biggest client!!", "nope"]) {
      const r = resolveQuote(MSG, q);
      if (r !== undefined) expect(MSG.slice(r.start, r.end)).toBe(r.text);
    }
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
// Every test above that checks `MSG.slice(start, end) === text` is satisfied by
// construction: the implementation DEFINES text as that slice, and so does
// core (packages/core/src/detect/orchestrator.ts:98 throws on the same
// comparison). A span that points at the wrong place is therefore invisible to
// all of them -- it slices cleanly and core accepts it. The tests below assert
// exact offsets against independently computed ground truth (indexOf on the
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
    // Measured on this machine: of all 65,536 BMP code units, exactly one has a
    // toLowerCase() longer than itself -- U+0130 LATIN CAPITAL LETTER I WITH DOT
    // ABOVE, which lowercases to "i" + U+0307 COMBINING DOT ABOVE, two code
    // units. Pushing that into `folded` while pushing one entry into `map`
    // desynchronizes the two for the whole rest of the string, so every later
    // index maps one place short.
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

  it("ends the span at the last matched character, not inside the whitespace run after it", () => {
    // buildFoldMap maps a whitespace RUN to the run's FIRST index, so an `end`
    // derived from a folded space would land one character into the run rather
    // than after it. Reachable only if a candidate is allowed to end on
    // whitespace; this pins that it never is.
    const msg = "The codename is Bluebird\n\n\n\n\nand that is confidential.";
    const quote = "the codename is bluebird   ";
    const r = resolveQuote(msg, quote);
    expect(r).toBeDefined();
    expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(msg, "The codename is Bluebird"));
    expect(/\s$/.test(r!.text)).toBe(false);
  });
});

describe("resolveQuote: rung 2 must not shed the word carrying the secret", () => {
  const AWS = "Hey team - quick one before the board review with Halcyon. Priya Sharma "
    + "(priya.sharma@northwind-traders.io) is asking whether we can hold the Halcyon renewal "
    + "until Q3. Her base is 142,000 GBP and Halcyon's own counsel flagged clause 7. Can someone "
    + "pull the AWS key AKIAIOSFODNN7EXAMPLE out of the staging config before we send anything?";

  it("keeps the secret inside the span when the model appends a full stop", () => {
    // The measured perturbation. Phi-4-mini returned a quote with an appended
    // "?" in this very corpus (webllm-probe/out-e5.json). Dropping a whole WORD
    // to shed one punctuation character deletes the last word of the quote --
    // which, for a quote shaped "<label> <secret>", IS the secret. The span
    // would then name the secret without covering it, and applyActions would
    // vault the label and leave the credential in the message.
    const r = resolveQuote(AWS, "the AWS key AKIAIOSFODNN7EXAMPLE.");
    expect(r).toBeDefined();
    expect(r!.text).toContain("AKIAIOSFODNN7EXAMPLE");
    expect(AWS.slice(r!.start, r!.end)).toBe(r!.text);
  });

  it("keeps the trailing unit when the model appends a full stop to a salary", () => {
    const r = resolveQuote(AWS, "Her base is 142,000 GBP.");
    expect(r).toBeDefined();
    expect(r!.text).toBe("Her base is 142,000 GBP");
    expect(r!.rung).toBe(2);
  });

  it("still refuses when the punctuation-stripped quote is ambiguous", () => {
    // Stripping must not become a licence to guess.
    const msg = "The client roster is secret. The client roster is public.";
    expect(resolveQuote(msg, "The client roster is!!")).toBeUndefined();
  });
});

describe("resolveQuote: the floors that stop a model scoring by accident", () => {
  it("refuses a two-word prefix even when it is unique", () => {
    // Measured ambiguity: one-word quotes are non-unique 22-51% of the time,
    // two-word 7-38%. A two-word prefix that happens to be unique here will not
    // be in the next message, so accepting it buys a number that does not hold.
    const msg = "The Zephyr protocol alpha is our internal codename for the migration.";
    expect(msg.indexOf("Zephyr protocol")).toBeGreaterThan(-1);
    expect(resolveQuote(msg, "Zephyr protocol beta gamma delta")).toBeUndefined();
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
  it("never splits a surrogate pair", () => {
    // Offsets are UTF-16 code unit indices, so a span boundary CAN fall between
    // the halves of an astral character. Half a pair is a lone surrogate: it
    // still satisfies text === slice(start, end), so core would accept it and
    // the vault would store a broken string.
    const msg = "Report 🎉 the codename is Bluebird 🎉 internally.";
    const isHigh = (c: number): boolean => c >= 0xd800 && c <= 0xdbff;
    const isLow = (c: number): boolean => c >= 0xdc00 && c <= 0xdfff;
    for (const q of ["the codename is Bluebird", "\uDF89 the codename is Bluebird",
                     "the codename is Bluebird \uD83C", "report 🎉 the codename"]) {
      const r = resolveQuote(msg, q);
      if (r === undefined) continue;
      expect(isLow(msg.charCodeAt(r.start)) && isHigh(msg.charCodeAt(r.start - 1))).toBe(false);
      expect(isHigh(msg.charCodeAt(r.end - 1)) && isLow(msg.charCodeAt(r.end))).toBe(false);
    }
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
// out-e6.json) against exactly this message, under grammar-constrained
// decoding at temperature 0. They are the whole distinct set the corpus
// contains -- nothing was selected for making the ladder look good, and the
// one non-verbatim quote is included.
// ---------------------------------------------------------------------------

describe("resolveQuote: quotes real models actually returned", () => {
  const PROBE = "Hey team - quick one before the board review with Halcyon. Priya Sharma "
    + "(priya.sharma@northwind-traders.io) is asking whether we can hold the Halcyon renewal "
    + "until Q3. Her base is 142,000 GBP and Halcyon's own counsel flagged clause 7. Can someone "
    + "pull the AWS key AKIAIOSFODNN7EXAMPLE out of the staging config before we send anything?";

  it("resolves each uniquely-occurring verbatim quote at rung 1, on the ground truth", () => {
    for (const q of ["AKIAIOSFODNN7EXAMPLE", "Her base is 142,000 GBP", "Halcyon renewal",
                     "Halcyon's own counsel flagged clause 7"]) {
      const r = resolveQuote(PROBE, q);
      expect(r?.rung, q).toBe(1);
      expect({ start: r!.start, end: r!.end }).toEqual(truthSpan(PROBE, q));
    }
  });

  it("refuses the one-word quote Phi-4-mini repeated nineteen times", () => {
    // "Halcyon" occurs three times. Picking one would be a coin flip, and the
    // duplicate-heavy output that produced it is exactly the case where a
    // guessed span would inflate an arm's score.
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
