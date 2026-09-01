import { describe, expect, it } from "vitest";
import {
  BASELINE_B_SCHEMA,
  BaselineResponseSchema,
  JUDGE_SCHEMA,
  JudgeResponseSchema,
  parseBaselineResponse,
  parseJudgeResponse,
} from "../src/schema.js";

describe("JUDGE_SCHEMA", () => {
  it("is a plain JSON schema, serializable for the grammar constraint", () => {
    // WebLLM takes response_format.schema as a STRING. A schema carrying a
    // function, a RegExp or a cycle stringifies to something xgrammar cannot
    // compile, and the failure surfaces as a hang rather than an error.
    expect(() => JSON.stringify(JUDGE_SCHEMA)).not.toThrow();
    expect(JSON.parse(JSON.stringify(JUDGE_SCHEMA))).toEqual(JUDGE_SCHEMA);
  });

  it("is deeply frozen, because every engine arm shares this one object", () => {
    // The doc comment claims this; nothing else here would notice if a caller
    // mutated the schema between bake-off arms and silently changed what the
    // later arms were constrained to.
    const thawed: string[] = [];
    const walk = (node: unknown, path: string): void => {
      if (node === null || typeof node !== "object") return;
      if (!Object.isFrozen(node)) thawed.push(path);
      for (const [key, value] of Object.entries(node)) walk(value, `${path}.${key}`);
    };
    walk(JUDGE_SCHEMA, "JUDGE_SCHEMA");
    expect(thawed).toEqual([]);
  });

  it("asks for a verbatim quote, never for offsets", () => {
    // A model asked for character offsets returns wrong ones. The quote is
    // what the span ladder resolves; see spans.ts.
    const props = JUDGE_SCHEMA.properties.findings.items.properties;
    expect(Object.keys(props)).toContain("quote");
    expect(Object.keys(props)).not.toContain("start");
    expect(Object.keys(props)).not.toContain("end");
  });

  it("asks for a MENTION as well, and asks for it AFTER the quote", () => {
    // Two spans, because locating a finding and acting on one want opposite
    // lengths -- see spans.ts. The ORDER is asserted because it is not
    // cosmetic: MEASURED on the shipped compiler, xgrammar emits the
    // properties in declaration order as a fixed key sequence, so the logit
    // mask makes the model write the clause before it commits to the mention.
    // Both orders are the same object to every parser here, so nothing else in
    // this repository would notice a swap.
    const items = JUDGE_SCHEMA.properties.findings.items;
    expect(Object.keys(items.properties)).toEqual([
      "predicateId",
      "quote",
      "mention",
      "confidence",
    ]);
    expect(items.required).toEqual(["predicateId", "quote", "mention", "confidence"]);
    expect(items.properties.mention).toEqual({ type: "string" });
  });

  it("bounds confidence in the GRAMMAR, so an out-of-range value is unemittable", () => {
    // MEASURED on the shipped grammar compiler -- web-llm inlines xgrammar's
    // wasm and @mlc-ai/web-xgrammar@0.1.27 carries the byte-identical binary
    // (727,906 bytes, sha256 80eb86a9e61e8148...), so this was settled under
    // Node without a GPU. xgrammar folds these two keywords into the grammar as
    //   ( "0" | "1" | "0" "." [0-9]{1,6} | "1" "." [0-9]{1,6} )
    // and the logit mask then forecloses 95, -0.5, 1e999 and 0.5000001 before
    // sampling. Without them, `type: "number"` accepts every one of those.
    //
    // The zod check in JudgeResponseSchema is NOT made redundant by this and
    // must stay: the grammar is a sound over-approximation, not an exact one --
    // 1.5 still passes it, because the rule pins only the leading digit.
    const confidence = JUDGE_SCHEMA.properties.findings.items.properties.confidence;
    expect(confidence.minimum).toBe(0);
    expect(confidence.maximum).toBe(1);
  });
});

describe("parseJudgeResponse", () => {
  it("accepts a well-formed response", () => {
    const r = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"Acme Corp is our client","mention":"Acme Corp","confidence":0.9}]}');
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.findings[0]!.quote).toBe("Acme Corp is our client");
  });

  it("reports truncation distinctly from malformation", () => {
    // The two need different responses: truncation says raise max_tokens or
    // shorten the prompt; malformation says the constraint is not working.
    // Measured: every real parse failure was truncation.
    const truncated = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"Acme');
    expect(truncated.ok).toBe(false);
    if (!truncated.ok) expect(truncated.reason).toBe("truncated");
  });

  it("reports a schema mismatch as its own reason", () => {
    const wrong = parseJudgeResponse('{"findings":[{"predicateId":"p1"}]}');
    expect(wrong.ok).toBe(false);
    if (!wrong.ok) expect(wrong.reason).toBe("schema");
  });

  it("rejects an empty quote rather than passing it to the span ladder", () => {
    // An empty quote matches at offset 0 of every message. That is a finding
    // pointing at the wrong text, which is worse than no finding.
    const r = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"","mention":"Acme","confidence":0.9}]}');
    expect(r.ok).toBe(false);
  });

  it("accepts an empty findings array as a real answer", () => {
    // "Nothing here" is a legitimate judgement, not a failure. Conflating it
    // with a parse failure would make a silent model look like a broken one.
    const r = parseJudgeResponse('{"findings":[]}');
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.findings).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Everything below was added after measuring the plan's own heuristic against
// the truncation boundaries a constrained response actually lands on.
// ---------------------------------------------------------------------------

const WELL_FORMED =
  '{"findings":[{"predicateId":"p1","quote":"Acme Corp is our client","mention":"Acme Corp","confidence":0.9},' +
  '{"predicateId":"p2","quote":"Project Halcyon ships in Q3","mention":"Halcyon","confidence":0.42}]}';

describe("truncation is classified structurally, not by sniffing the V8 error text", () => {
  it("classifies EVERY truncation boundary of a real response as truncated", () => {
    // The oracle here is independent of the classifier: truncation IS "a proper
    // prefix of a valid document". Cutting a known-good response at every index
    // enumerates every boundary a token budget can land on -- mid-string,
    // mid-number, mid-key, mid-array, and after a complete object but before
    // its closing brace -- without any expectation derived from the code under
    // test.
    //
    // MEASURED on this machine (Node 26): the error-message heuristic the plan
    // shipped, /Unexpected end of (JSON )?input|Unterminated/i, calls 26 of the
    // 152 cut points below "malformed" -- among them mid-array, mid-number, and
    // after a complete object but before its closing brace. V8 reports
    // "Expected ',' or '}' after property value" for a response cut after a
    // complete confidence number, and that matches neither alternative. Every
    // one of the plan's own six tests stays green against that heuristic; only
    // sweeping the boundaries exposes it.
    expect(parseJudgeResponse(WELL_FORMED).ok).toBe(true);

    const misclassified: string[] = [];
    for (let i = 0; i < WELL_FORMED.length; i++) {
      const prefix = WELL_FORMED.slice(0, i);
      // No proper prefix of this document is itself valid JSON: the outer
      // object closes only at the final character. Assert that rather than
      // assume it, so the fixture is grounded in V8 and not in our classifier.
      expect(() => JSON.parse(prefix)).toThrow();
      const r = parseJudgeResponse(prefix);
      if (r.ok || r.reason !== "truncated") {
        misclassified.push(`at ${i} -> ${r.ok ? "ok" : r.reason} | ...${JSON.stringify(prefix.slice(-18))}`);
      }
    }
    expect(misclassified).toEqual([]);
  });

  it("classifies the real Phi-4-mini truncation from the probe corpus", () => {
    // Verbatim prefix of out-e5.json phi["tier2-shaped"][0].full: 1959 chars,
    // finish="length", "Unterminated string in JSON at position 1959". The
    // model looped a duplicate finding until the budget ran out mid-string.
    const corpus =
      '{"findings": [{"quote": "Halcyon renewal", "before": "Hey te", "label": "Confidential Information", "clause": ""}, {"quote": "Halcyon", "before": "Hey te - quick", "label": "Confidential Information", "clause": ""}, {"quote": "Halcyon renew';
    const r = parseJudgeResponse(corpus);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("truncated");
  });

  it("treats an empty or whitespace-only response as truncated, not malformed", () => {
    // An interrupted stream yields no bytes at all. That is the budget/cancel
    // story, not a broken grammar constraint.
    for (const raw of ["", "   ", "\n"]) {
      const r = parseJudgeResponse(raw);
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.reason).toBe("truncated");
    }
  });

  it("classifies an incomplete literal as truncated but a diverging one as malformed", () => {
    // "nul" can still become "null"; "nope" cannot become anything valid. A
    // classifier that only looks at the error string cannot tell these apart.
    const cut = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"a","confidence":nul');
    expect(cut.ok).toBe(false);
    if (!cut.ok) expect(cut.reason).toBe("truncated");

    const diverged = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"a","confidence":nope}]}');
    expect(diverged.ok).toBe(false);
    if (!diverged.ok) expect(diverged.reason).toBe("malformed");
  });

  it("uses the engine's finish reason when it has one, rather than guessing", () => {
    // MEASURED: out-e5.json records finish="length" on all three failures. That
    // is the engine stating it cut the response off -- a fact, where the shape
    // of the fragment is only evidence. Task 6 should pass it through.
    const r = parseJudgeResponse('{"findings":[{"predicateId":"p1"} bogus', "length");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("truncated");
  });

  it("files a cancelled call as aborted, never as truncation", () => {
    // An interrupted engine returns "", and "" is a perfect truncated prefix --
    // classifyJsonPrefix calls it truncated and is right to. So without the
    // finish_reason branch these two are indistinguishable, and the empty body
    // of a poisoned engine is reported as "the model ran out of tokens". That
    // points at the wrong fix (raise max_tokens does nothing for a call nobody
    // let finish) and inflates the per-arm truncation count the bake-off reads
    // as "this model is too verbose for its budget".
    //
    // The pairing is the assertion: the SAME raw body classifies differently
    // depending only on what the engine said about why it stopped.
    const aborted = parseJudgeResponse("", "abort");
    expect(aborted.ok).toBe(false);
    if (!aborted.ok) {
      expect(aborted.reason).toBe("aborted");
      expect(aborted.detail).toContain("finish_reason=abort");
    }
    expect(parseJudgeResponse("").ok).toBe(false);
    const noReason = parseJudgeResponse("");
    if (!noReason.ok) expect(noReason.reason).toBe("truncated");

    // A partial body under an abort is still an abort: the cut is why it is
    // partial, so the reason has to name the cut and not its shape.
    const partial = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"Acme', "abort");
    expect(partial.ok).toBe(false);
    if (!partial.ok) expect(partial.reason).toBe("aborted");
  });

  it("still returns a schema-valid body as ok even when the call was aborted", () => {
    // Deliberate, and stated so it is not mistaken for an oversight: an abort
    // almost never leaves a complete document behind, but when it does the
    // findings in it are real and the caller holds finishReason too. Filing a
    // parseable answer as a failure would discard genuine findings.
    const r = parseJudgeResponse('{"findings":[]}', "abort");
    expect(r.ok).toBe(true);
  });
});

describe("malformation is still reported as malformation", () => {
  // Without these, a classifier that answered "truncated" unconditionally would
  // pass every test above. Each case is a syntax error that NO suffix can
  // repair, so none of them is a truncated prefix.
  const cases: ReadonlyArray<readonly [string, string]> = [
    ["trailing brace after a complete document", '{"findings":[]}}'],
    ["comma where a value must begin", '{"findings":[,]}'],
    ["trailing comma before a closing brace", '{"findings":[{"quote":"a",}]}'],
    ["single-quoted key", "{'findings':[]}"],
    ["leading zero in a number", '{"findings":[01]}'],
    ["missing comma between members", '{"findings":[] "extra":1}'],
    ["a plus-signed number", '{"findings":[+1]}'],
    ["prose instead of JSON", "no findings were located"],
    ["raw control character inside a string", '{"findings":["a\nb"]}'],
    ["a key not followed by a colon", '{"findings" ["a"]}'],
    // Cut short AND holding a raw newline. It is unterminated, but no suffix
    // repairs it either: a JSON string can never contain a raw control
    // character, so this is malformation, not the benign "raise the budget".
    ["an unterminated string holding a control character", '{"findings":[{"quote":"a\nb'],
  ];

  for (const [name, raw] of cases) {
    it(`reports ${name} as malformed`, () => {
      // Ground the fixture in V8 first: if this does not throw, the case is not
      // a malformed document and the test below would be meaningless.
      expect(() => JSON.parse(raw)).toThrow();
      const r = parseJudgeResponse(raw);
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.reason).toBe("malformed");
    });
  }
});

describe("confidence is validated here, because core says it must be", () => {
  // packages/core/src/detect/merge.ts: "tier adapters must validate confidence
  // at the boundary". packages/core/src/detect/types.ts documents the field as
  // "0..1". This module is that boundary -- Task 6 receives an already-trusted
  // value and should not have to re-defend it.
  const at = (c: string) => parseJudgeResponse(`{"findings":[{"predicateId":"p1","quote":"a","mention":"a","confidence":${c}}]}`);

  it("rejects a confidence that overflowed to Infinity", () => {
    // JSON has no Infinity literal, but 1e999 parses to one. A non-finite
    // confidence makes every merge comparison false and destroys the ordering
    // determinism the rest of the pipeline depends on.
    expect(JSON.parse('{"c":1e999}').c).toBe(Infinity);
    const r = at("1e999");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("schema");
  });

  it("rejects a confidence outside 0..1 rather than clamping it silently", () => {
    // A model answering on a percentage scale is a real failure to honour the
    // contract. Clamping 95 to 1.0 would convert that misunderstanding into a
    // maximally confident finding and hide it from the bake-off's per-arm
    // failure counts; rejecting surfaces and counts it.
    for (const c of ["95", "1.0001", "-0.5", "-1e999"]) {
      const r = at(c);
      expect(r.ok, `confidence ${c} must be rejected`).toBe(false);
      if (!r.ok) expect(r.reason).toBe("schema");
    }
  });

  it("accepts both ends of the documented range", () => {
    for (const c of ["0", "1", "0.5"]) expect(at(c).ok, `confidence ${c}`).toBe(true);
  });

  it("exposes the same rule through JudgeResponseSchema directly", () => {
    expect(JudgeResponseSchema.safeParse({ findings: [{ predicateId: "p", quote: "q", mention: "q", confidence: 0.5 }] }).success).toBe(true);
    expect(JudgeResponseSchema.safeParse({ findings: [{ predicateId: "p", quote: "q", mention: "q", confidence: 2 }] }).success).toBe(false);
  });
});

describe("quote and predicateId are usable by the span ladder", () => {
  it("rejects a whitespace-only quote for the same reason as an empty one", () => {
    // min(1) lets " " through, and a single space matches in almost every
    // message -- the same "finding pointing at the wrong text" defect the empty
    // quote check exists to prevent, one character further along.
    for (const q of [" ", "\\t", "\\n  "]) {
      const r = parseJudgeResponse(`{"findings":[{"predicateId":"p1","quote":"${q}","mention":"a","confidence":0.9}]}`);
      expect(r.ok, `quote ${JSON.stringify(q)} must be rejected`).toBe(false);
    }
  });

  it("rejects an empty predicateId", () => {
    const r = parseJudgeResponse('{"findings":[{"predicateId":"","quote":"a","mention":"a","confidence":0.9}]}');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("schema");
  });

  it("REQUIRES a mention: a judge-shaped answer without one is a schema failure", () => {
    // The field is required rather than optional on purpose. An optional one is
    // a field a grammar lets the model skip, and a model that skips it every
    // time silently restores the whole-clause action span this split exists to
    // end -- with no counter able to see the difference, because there would be
    // no finding to count.
    const r = parseJudgeResponse('{"findings":[{"predicateId":"p1","quote":"a b c","confidence":0.9}]}');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("schema");
  });

  it("rejects an empty or whitespace-only mention, as it does a quote", () => {
    // `mention` decides Finding.start and Finding.end. An empty one matches at
    // offset 0 of its clause, which is a zero-width action span at the clause's
    // first character -- so it is held to the SAME rule as `quote` and not a
    // looser one. `locateFinding` refuses it again at the ladder; two layers,
    // because a body replayed from a record never met the grammar.
    for (const m of ["", " ", "\\t", "\\n  "]) {
      const r = parseJudgeResponse(
        `{"findings":[{"predicateId":"p1","quote":"a b c","mention":"${m}","confidence":0.9}]}`,
      );
      expect(r.ok, `mention ${JSON.stringify(m)} must be rejected`).toBe(false);
    }
  });

  it("preserves duplicate findings instead of quietly deduplicating them", () => {
    // Duplicates are the measured behaviour, not a hypothetical: out-e5.json's
    // truncated Phi-4-mini responses are one finding repeated until the budget
    // ran out, and Plan 5 records Qwen3.5-2B returning the same AWS key three
    // times. Deduplicating here would hide exactly what the "no duplicate-only
    // output" kill rule has to observe.
    const dup = '{"predicateId":"p1","quote":"Halcyon renewal","mention":"Halcyon","confidence":0.9}';
    const r = parseJudgeResponse(`{"findings":[${dup},${dup},${dup}]}`);
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.findings).toHaveLength(3);
  });
});

describe("JUDGE_SCHEMA stays inside the keyword set measured to compile", () => {
  it("carries no value JSON.stringify would silently drop or alter", () => {
    // Round-trip equality via toEqual does NOT catch an undefined-valued key:
    // toEqual ignores undefined properties, so `required: undefined` would pass
    // the plan's check while vanishing from the stringified schema and quietly
    // un-requiring every field. Walk the structure instead.
    const bad: string[] = [];
    const walk = (node: unknown, path: string): void => {
      if (node === undefined) return void bad.push(`${path} is undefined`);
      if (typeof node === "function") return void bad.push(`${path} is a function`);
      if (typeof node === "bigint") return void bad.push(`${path} is a bigint`);
      if (typeof node === "number" && !Number.isFinite(node)) return void bad.push(`${path} is non-finite`);
      if (node === null || typeof node !== "object") return;
      if (node instanceof RegExp || node instanceof Date || node instanceof Map || node instanceof Set) {
        return void bad.push(`${path} is a ${node.constructor.name}`);
      }
      if ("toJSON" in node) bad.push(`${path} has a toJSON hook`);
      for (const key of Reflect.ownKeys(node)) {
        if (typeof key === "symbol") bad.push(`${path} has a symbol key`);
        else walk((node as Record<string, unknown>)[key], `${path}.${key}`);
      }
    };
    walk(JUDGE_SCHEMA, "JUDGE_SCHEMA");
    expect(bad).toEqual([]);
  });

  it("uses only keywords measured to compile on xgrammar 0.1.27", () => {
    // A serializable schema is not necessarily a COMPILABLE one, and an
    // uncompilable schema hangs rather than erroring -- so round-trip equality
    // alone is not enough. Anything outside this set -- $ref, allOf, pattern,
    // patternProperties, maxItems -- is unmeasured and must not ship until it
    // has been put through the compiler.
    //
    // The first five come from the probe's measured schema (webllm-probe
    // page/main2.ts, 77 calls, no hang).
    //
    // `minimum` and `maximum` were MEASURED SEPARATELY, and under Node rather
    // than in a browser, which is possible because web-llm inlines xgrammar's
    // wasm as base64 instead of fetching it and standalone
    // @mlc-ai/web-xgrammar@0.1.27 carries the same binary -- verified by
    // hashing both inlined blobs: 727,906 bytes, sha256 80eb86a9e61e8148...,
    // byte-identical. This exact JUDGE_SCHEMA object was put through
    // `compileJSONSchema`, the call llm_chat itself makes, under a watchdog
    // OUTSIDE the process because a hang inside wasm blocks Node's event loop.
    // It compiled. The compiled rule is
    //   ( "0" | "1" | "0" "." [0-9]{1,6} | "1" "." [0-9]{1,6} )
    // and on the accept/reject oracle it rejects 95, -0.5, 1e999 and 0.5000001
    // while the unbounded `type: "number"` accepts all four.
    const ALLOWED = new Set([
      "type",
      "properties",
      "items",
      "required",
      "additionalProperties",
      "minimum",
      "maximum",
    ]);
    const seen = new Set<string>();
    const walk = (node: unknown): void => {
      if (Array.isArray(node)) return void node.forEach(walk);
      if (node === null || typeof node !== "object") return;
      for (const [key, value] of Object.entries(node)) {
        seen.add(key);
        // Keys under `properties` are OUR field names, not schema keywords.
        if (key === "properties") Object.values(value as object).forEach(walk);
        else walk(value);
      }
    };
    walk(JUDGE_SCHEMA);
    expect([...seen].filter((k) => !ALLOWED.has(k))).toEqual([]);
  });

  it("forbids additional properties, so the model cannot invent offsets", () => {
    // Measured in the probe schema and compiled fine. Without it the grammar
    // permits invented keys: budget spent on fields we ignore, on the one
    // failure mode -- running out of tokens -- that is actually measured here.
    expect(JUDGE_SCHEMA.additionalProperties).toBe(false);
    expect(JUDGE_SCHEMA.properties.findings.items.additionalProperties).toBe(false);
  });

  it("requires every field it declares", () => {
    const items = JUDGE_SCHEMA.properties.findings.items;
    expect([...items.required].sort()).toEqual(Object.keys(items.properties).sort());
    expect(JUDGE_SCHEMA.required).toEqual(["findings"]);
  });
});


describe("BASELINE_B_SCHEMA", () => {
  it("is the judge's schema with exactly one property renamed", () => {
    // Written out separately from JUDGE_SCHEMA rather than derived from it, so
    // this comparison is between two independent objects and a change to
    // either one alone fails here. Anything MORE than the rename -- a looser
    // confidence bound, a dropped additionalProperties -- is one arm decoding
    // under a different grammar, which is the head-to-head measuring the
    // harness instead of the method.
    const renamed = JSON.parse(
      JSON.stringify(JUDGE_SCHEMA).replaceAll("predicateId", "entityType"),
    ) as unknown;
    expect(BASELINE_B_SCHEMA).toEqual(renamed);
  });

  it("names the field Approach B actually asks the model for", () => {
    // The independent half: the shape above is only the right shape if this is
    // the shape. Asserted against literals, not against JUDGE_SCHEMA.
    const items = BASELINE_B_SCHEMA.properties.findings.items;
    expect(items.required).toEqual(["entityType", "quote", "mention", "confidence"]);
    expect(items.additionalProperties).toBe(false);
    expect(items.properties.confidence).toEqual({ type: "number", minimum: 0, maximum: 1 });
  });

  it("is deeply frozen, for the same reason the judge's is", () => {
    const thawed: string[] = [];
    const walk = (node: unknown, path: string): void => {
      if (node === null || typeof node !== "object") return;
      if (!Object.isFrozen(node)) thawed.push(path);
      for (const [key, value] of Object.entries(node)) walk(value, `${path}.${key}`);
    };
    walk(BASELINE_B_SCHEMA, "BASELINE_B_SCHEMA");
    expect(thawed).toEqual([]);
  });

  it("is serializable, since the grammar constraint takes a string", () => {
    // The round trip is asserted rather than the absence of a throw:
    // JSON.stringify(undefined) throws nothing either, so a missing export
    // would pass a not-toThrow check.
    expect(JSON.parse(JSON.stringify(BASELINE_B_SCHEMA))).toEqual(BASELINE_B_SCHEMA);
  });
});

describe("the two arms are parsed under identical rules", () => {
  // The bake-off's headline number is "compiled pipeline versus Approach B".
  // A parser that is stricter for one of them turns a difference in harness
  // into a difference attributed to method, so the two entry points share a
  // private core and these tests are what say so from outside.

  const BODIES: Array<[string, string, string | undefined]> = [
    ["an empty body from a latched engine", "", "abort"],
    ["a body cut off at max_tokens", '{"findings":[{"quote":"Northwind Tr', "length"],
    ["a body cut mid-number with no finish reason", '{"findings":[{"confidence":0.', undefined],
    ["prose instead of JSON", "Here are the findings I found:", undefined],
    ["content after a closed document", '{"findings":[]} and also', undefined],
  ];

  for (const [label, raw, finishReason] of BODIES) {
    it(`classifies ${label} the same way for both arms`, () => {
      const judged = parseJudgeResponse(raw, finishReason);
      const baseline = parseBaselineResponse(raw, finishReason);
      expect(judged.ok).toBe(false);
      expect(baseline.ok).toBe(false);
      if (judged.ok || baseline.ok) return;
      expect(baseline.reason).toBe(judged.reason);
    });
  }

  it("holds each arm to its OWN field name, so a schema swap is loud", () => {
    const asJudge =
      '{"findings":[{"predicateId":"p","quote":"three whole words","mention":"words","confidence":0.5}]}';
    const asBaseline =
      '{"findings":[{"entityType":"in-pan","quote":"three whole words","mention":"words","confidence":0.5}]}';
    expect(parseJudgeResponse(asJudge).ok).toBe(true);
    expect(parseBaselineResponse(asBaseline).ok).toBe(true);
    // Cross-wired, both fail as a SCHEMA mismatch rather than silently
    // producing a finding with an undefined label.
    const crossed = parseJudgeResponse(asBaseline);
    expect(crossed.ok).toBe(false);
    if (!crossed.ok) expect(crossed.reason).toBe("schema");
    const crossedBack = parseBaselineResponse(asJudge);
    expect(crossedBack.ok).toBe(false);
    if (!crossedBack.ok) expect(crossedBack.reason).toBe("schema");
  });

  it("applies the same quote and confidence rules to both arms", () => {
    // The two field validators are shared consts in schema.ts precisely so
    // this holds. A whitespace-only quote matches almost every message at some
    // offset; 1.5 is what the grammar's leading-digit rule still lets through.
    const cases = [
      ['"   "', "0.5"],
      ['"three whole words"', "1.5"],
      ['"three whole words"', "-0.5"],
      ['"three whole words"', "1e999"],
    ];
    for (const [quote, confidence] of cases) {
      const judge = parseJudgeResponse(
        `{"findings":[{"predicateId":"p","quote":${quote},"mention":${quote},"confidence":${confidence}}]}`,
      );
      const baseline = parseBaselineResponse(
        `{"findings":[{"entityType":"e","quote":${quote},"mention":${quote},"confidence":${confidence}}]}`,
      );
      expect(judge.ok, `${quote} / ${confidence}`).toBe(false);
      expect(baseline.ok, `${quote} / ${confidence}`).toBe(false);
    }
  });

  it("accepts the same well-formed answer on both arms", () => {
    // The other direction, so the test above cannot pass by rejecting
    // everything.
    expect(
      parseJudgeResponse(
        '{"findings":[{"predicateId":"p","quote":"three whole words","mention":"words","confidence":1}]}',
      ).ok,
    ).toBe(true);
    expect(
      parseBaselineResponse(
        '{"findings":[{"entityType":"e","quote":"three whole words","mention":"words","confidence":1}]}',
      ).ok,
    ).toBe(true);
  });
});
