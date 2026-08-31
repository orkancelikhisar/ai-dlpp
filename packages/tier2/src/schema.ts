import { z } from "zod";

/**
 * The schema the model is grammar-constrained to. Deliberately small: every
 * field is one the model can actually produce.
 *
 * `quote` and not `start`/`end`. A small model asked for character offsets
 * returns wrong ones -- and a wrong offset is worse than no finding, because
 * core re-derives `text` from the span, so a mis-located finding is
 * schema-valid and passes the fidelity check. Offsets are recovered locally by
 * `spans.ts` instead, and which rung recovered them is reported.
 *
 * The keyword set is held to what has already been shown to compile. Read off
 * webllm-probe `page/main2.ts`: its schema uses `type`, `properties`, `items`,
 * `required` and `additionalProperties` and nothing else, and Plan 5's recorded
 * feasibility run put 77 calls through it on xgrammar 0.1.27 without a hang.
 * `additionalProperties: false` earns its place twice over -- it is inside that
 * set, and without it the grammar permits invented keys, which spend budget on
 * fields we ignore and so feed the one failure mode actually observed here,
 * running out of tokens mid-response.
 *
 * `type: "number"`, and its `minimum`/`maximum`, are MEASURED HERE rather than
 * assumed. The grammar compiler is reachable from Node without a GPU: web-llm
 * inlines xgrammar's wasm as a base64 data URI instead of fetching it, and
 * standalone `@mlc-ai/web-xgrammar@0.1.27` carries the same binary. VERIFIED
 * HERE by hashing both bundles' inlined blobs -- 727,906 bytes, sha256
 * `80eb86a9e61e8148a45d60973ec29ffb85077a3e42cee8f042e1452fffd63774`, present
 * in each -- so a schema compiled under Node exercises the exact binary the
 * browser runs. This object was then put through `compileJSONSchema`, the call
 * `llm_chat` itself makes, under a watchdog OUTSIDE the process because a hang
 * inside wasm blocks Node's event loop. It compiled; nothing hung. That matters
 * because an uncompilable schema HANGS rather than erroring.
 *
 * The bounds are not decoration and not merely a zod convenience. xgrammar
 * folds them into the grammar itself --
 * `("0" | "1" | "0" "." [0-9]{1,6} | "1" "." [0-9]{1,6})` -- so the logit mask
 * forecloses an out-of-range confidence before sampling rather than after.
 * MEASURED on the accept/reject oracle: with bounds, `95` (a model answering on
 * a percentage scale), `-0.5`, `1e999` and `0.5000001` are all REJECTED, where
 * the unbounded `type: "number"` accepts every one of them.
 *
 * Two consequences worth stating rather than discovering:
 *
 * - **The grammar is a sound over-approximation, not an exact one.** `1.5`
 *   still passes, because the rule pins only the leading digit. The zod check
 *   in `JudgeResponseSchema` is therefore NOT redundant and must stay.
 * - **`{1,6}` caps the fraction at six digits.** MEASURED: `0.123456` is
 *   accepted and `0.1234567` is rejected. A model wanting a seventh decimal
 *   place is masked into stopping at six. That is a precision limit on a field
 *   we only ever compare and threshold, not a failure.
 *
 * Frozen for the same reason as the manifest: `readonly` is erased at runtime,
 * and this object is process-wide shared state handed to every engine arm.
 */
export const JUDGE_SCHEMA = deepFreeze({
  type: "object",
  properties: {
    findings: {
      type: "array",
      items: {
        type: "object",
        properties: {
          predicateId: { type: "string" },
          quote: { type: "string" },
          confidence: { type: "number", minimum: 0, maximum: 1 },
        },
        required: ["predicateId", "quote", "confidence"],
        additionalProperties: false,
      },
    },
  },
  required: ["findings"],
  additionalProperties: false,
} as const);

/**
 * The schema the Approach-B arm is constrained to: `JUDGE_SCHEMA` with one
 * property renamed, and nothing else different.
 *
 * Written out rather than derived from `JUDGE_SCHEMA` so the two are
 * independent objects that a test can compare -- a derived one agrees with its
 * source by construction and could only ever prove that the derivation ran.
 * `schema.test.ts` asserts the two are equal once `predicateId` is renamed, and
 * asserts this object's own shape against literals, so a change to either alone
 * fails there.
 *
 * Why the rename at all, rather than reusing the judge's schema unchanged: B
 * has no compiler and therefore no predicates to name. What it names is an
 * entity CLASS from the IR's vocabulary, and a wire field called `predicateId`
 * carrying an entityType id is the field-says-one-thing-holds-another defect
 * this project has shipped twice, one layer down.
 *
 * What the two arms share is what matters for the comparison and is unchanged
 * here: the same keyword set (`type`, `properties`, `items`, `required`,
 * `additionalProperties`, `minimum`, `maximum`), the same nesting, the same
 * `additionalProperties: false`, and the same `[0, 1]` bounds that xgrammar
 * folds into the grammar so an out-of-range confidence is unemittable. Only a
 * property NAME differs.
 *
 * NOT re-measured on the grammar compiler. Task 2 put `JUDGE_SCHEMA` through
 * `compileJSONSchema` -- the call `llm_chat` makes -- under an external
 * watchdog, and it compiled. This object uses that same keyword set and differs
 * from it only in the text of one property name, which is a claim about this
 * object's SHAPE (and is the thing the test above checks); it is not a claim
 * that anyone ran the compiler on this object. If a future arm needs a keyword
 * `JUDGE_SCHEMA` does not already carry, run the Node harness again -- an
 * uncompilable schema hangs rather than erroring.
 */
export const BASELINE_B_SCHEMA = deepFreeze({
  type: "object",
  properties: {
    findings: {
      type: "array",
      items: {
        type: "object",
        properties: {
          entityType: { type: "string" },
          quote: { type: "string" },
          confidence: { type: "number", minimum: 0, maximum: 1 },
        },
        required: ["entityType", "quote", "confidence"],
        additionalProperties: false,
      },
    },
  },
  required: ["findings"],
  additionalProperties: false,
} as const);

// The two field validators both response schemas use, defined once. Shared
// rather than repeated because the arms must be parsed under IDENTICAL
// strictness: a `quote` rule that is tighter for one arm, or a `confidence`
// bound that is looser for the other, would show up in the bake-off as a
// difference in findings and be read as a difference in the models.

/**
 * An empty quote matches at offset 0 of every message, which is a finding
 * pointing at the wrong text. A whitespace-only quote is the same defect one
 * character further along -- a single space matches in almost every message --
 * so the check is "has a non-whitespace character", not `min(1)`. It must NOT
 * be written as `z.string().trim().min(1)`: zod's .trim() rewrites the value,
 * and the span ladder needs the quote verbatim to find it in the message.
 */
const QUOTE_FIELD = z.string().refine((s) => s.trim().length > 0, {
  message: "quote must contain a non-whitespace character",
});

/**
 * core/src/detect/merge.ts requires tier adapters to validate confidence at the
 * boundary, and core/src/detect/types.ts documents the field as 0..1; this
 * module is that boundary. The bounds are not decoration: JSON has no Infinity
 * literal but `1e999` parses to one, and a non-finite confidence makes every
 * merge comparison false, which makes Array#sort's ordering inconsistent and
 * destroys determinism downstream. MEASURED on zod 4.4.3: `z.number()` already
 * rejects NaN and both infinities, and `.min(0).max(1)` excludes them
 * independently of that.
 *
 * This is the SECOND of two layers, and it is not redundant. Both JSON schemas
 * now carry the same bounds, and MEASURED on the shipped grammar compiler they
 * are real: 95, -0.5 and 1e999 become unemittable. But the compiled rule pins
 * only the leading digit -- `("0" | "1" | "0" "." [0-9]{1,6} | "1" "."
 * [0-9]{1,6})` -- so it is a sound over-approximation, not an exact one, and
 * `1.5` still reaches here. Deleting this check because "the grammar already
 * handles it" would let exactly that value through. The grammar also only binds
 * a grammar-constrained call; anything replayed from a record, or read back
 * from a run made before the bounds were added, arrives unfiltered.
 *
 * Out-of-range values are REJECTED, not clamped. A model answering on a
 * percentage scale has failed to honour the contract; clamping 95 to 1.0 would
 * turn that misunderstanding into a maximally confident finding and hide it
 * from the bake-off's per-arm failure counts.
 */
const CONFIDENCE_FIELD = z.number().min(0).max(1);

export const JudgeResponseSchema = z.object({
  findings: z.array(
    z.object({
      predicateId: z.string().min(1),
      quote: QUOTE_FIELD,
      confidence: CONFIDENCE_FIELD,
    }),
  ),
});

/**
 * What the Approach-B arm is allowed to have said, under the same two field
 * validators as the judge's.
 *
 * `entityType` is checked for being a non-empty string and NOT for being an id
 * the IR declares -- deliberately, and symmetrically with `predicateId` above.
 * Rejecting an invented label here would reject the whole RESPONSE, losing the
 * findings alongside it that were fine; both arms instead drop the one finding
 * and count it (`JudgeStats.unknownPredicates`, `BaselineStats.unknownEntityTypes`),
 * because a model inventing labels is a measurement, not a parse error.
 */
export const BaselineResponseSchema = z.object({
  findings: z.array(
    z.object({
      entityType: z.string().min(1),
      quote: QUOTE_FIELD,
      confidence: CONFIDENCE_FIELD,
    }),
  ),
});

export type JudgeResponse = z.infer<typeof JudgeResponseSchema>;
export type BaselineResponse = z.infer<typeof BaselineResponseSchema>;

export type ParseResult<T = JudgeResponse> =
  | { ok: true; value: T }
  | { ok: false; reason: "aborted" | "truncated" | "malformed" | "schema"; detail: string };

/**
 * Parse a model response, distinguishing the four failure modes because they
 * call for different responses: an abort means nobody let the call finish,
 * truncation means raise max_tokens or shorten the prompt, malformation means
 * the grammar constraint is not working, and a schema mismatch means the prompt
 * and the schema disagree.
 *
 * Plan 5's feasibility run recorded 0 malformed responses in 77 constrained
 * calls. VERIFIED HERE against that run's out-e5.json: 3 of Phi-4-mini's 6
 * calls still failed to parse, all three at `finish: "length"` with 599 of 600
 * permitted completion tokens spent, after the model looped one finding until
 * the budget ran out mid-string. So truncation is the common failure and
 * malformation the rare one, and the distinction has to be right rather than
 * approximately right.
 *
 * `finishReason` is the engine's own `choices[0].finish_reason`. Pass it when
 * you have it: "length" is the engine stating that IT cut the response off,
 * where the shape of the fragment is only evidence for the same conclusion.
 * The full union on 0.2.84 is `"stop" | "length" | "tool_calls" | "abort"` plus
 * `undefined`; only "length" and "abort" change the answer here.
 *
 * A body that parses AND validates is returned as `ok` whatever the finish
 * reason says, including "abort". That is deliberate rather than overlooked: an
 * aborted call almost never leaves a schema-valid document behind, and when it
 * does the findings in it are real. The caller holds `finishReason` too and can
 * decide whether to trust a judgement from a call it cancelled.
 */
export function parseJudgeResponse(raw: string, finishReason?: string): ParseResult<JudgeResponse> {
  return parseConstrainedResponse(raw, finishReason, JudgeResponseSchema);
}

/**
 * The same parse for the Approach-B arm, differing only in which zod schema
 * validates the object.
 *
 * A separate entry point rather than a `schema` parameter on the one above so
 * neither arm can be handed the other's validator by accident; a shared PRIVATE
 * core rather than a second implementation because the thing that must not
 * differ between the arms is the failure CLASSIFICATION. B is the arm most
 * exposed to it -- one call for a whole message against the judge's one per
 * segment, so B is likelier to run out of completion tokens -- and a B that
 * called truncation "malformed" would be reported as decoding under a broken
 * grammar constraint while the compiled arm was reported as merely verbose.
 */
export function parseBaselineResponse(
  raw: string,
  finishReason?: string,
): ParseResult<BaselineResponse> {
  return parseConstrainedResponse(raw, finishReason, BaselineResponseSchema);
}

function parseConstrainedResponse<T>(
  raw: string,
  finishReason: string | undefined,
  schema: z.ZodType<T>,
): ParseResult<T> {
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (cause) {
    // Checked BEFORE "length" and before the shape scanner, because it is the
    // one finish_reason that says why there is no body rather than what the
    // body looks like. An interrupted engine returns "", and "" is a perfect
    // truncated prefix -- `classifyJsonPrefix` calls it truncated and is right
    // to -- so without this branch a cancelled call is indistinguishable from a
    // model that ran out of tokens. The two call for opposite responses:
    // raising max_tokens does nothing for a call nobody let finish, and filing
    // it as truncation inflates the per-arm truncation count that the bake-off
    // reads as "this model is too verbose for its budget". Task 3 measured the
    // engine state that produces this: after an interrupt-and-drain the flag
    // stays set, and EVERY later call returns instantly with an empty body and
    // finish_reason "abort" until it is cleared.
    if (finishReason === "abort") {
      return { ok: false, reason: "aborted", detail: `${String(cause)} (finish_reason=abort)` };
    }
    if (finishReason === "length") {
      return { ok: false, reason: "truncated", detail: `${String(cause)} (finish_reason=length)` };
    }
    const shape = classifyJsonPrefix(raw);
    // `complete` means the scanner and V8 disagree about validity, so we do not
    // understand this output. Report the reason that raises an alarm, not the
    // benign one that says "just raise the budget".
    return {
      ok: false,
      reason: shape === "truncated" ? "truncated" : "malformed",
      detail: String(cause),
    };
  }
  const parsed = schema.safeParse(json);
  if (!parsed.success) return { ok: false, reason: "schema", detail: z.prettifyError(parsed.error) };
  return { ok: true, value: parsed.data };
}

// ---------------------------------------------------------------------------

/**
 * Is `text` a valid JSON document, a proper prefix of one, or neither?
 *
 * This exists instead of matching on the thrown SyntaxError's message. That
 * text is a V8 detail we do not control -- ECMA-262 requires a SyntaxError, not
 * any particular wording -- and MEASURED on this machine (Node 26), matching
 * /Unexpected end of (JSON )?input|Unterminated/i puts 6 of a 16-case boundary
 * table, and 26 of the 152 cut points of one real response, in the wrong
 * bucket. A response cut after
 * a complete `confidence` number gets "Expected ',' or '}' after property
 * value", which contains neither alternative and would be reported as a broken
 * grammar constraint. That misdirects the fix, and it also corrupts the
 * per-arm truncation count the bake-off needs in order to tell a budget-killed
 * arm from an incapable one.
 *
 * Truncation has an exact structural definition -- some suffix would complete
 * the document -- so it is decided by scanning rather than guessed. A cut can
 * land inside a string, inside an escape, inside a number's fraction or
 * exponent, inside a `true`/`false`/`null` literal, or between any two tokens
 * with containers still open; each is a distinct arm below.
 */
function classifyJsonPrefix(text: string): "complete" | "truncated" | "malformed" {
  const n = text.length;
  const stack: Array<"object" | "array"> = [];
  let i = 0;
  let mode: "value" | "valueOrArrayEnd" | "keyOrObjectEnd" | "key" | "colon" | "afterValue" = "value";

  // Scanners return the index just past the token, or CUT / BAD.
  const CUT = -1;
  const BAD = -2;
  const isDigit = (c: string | undefined) => c !== undefined && c >= "0" && c <= "9";

  const scanString = (): number => {
    let j = i + 1;
    for (;;) {
      if (j >= n) return CUT;
      const c = text[j]!;
      if (c === '"') return j + 1;
      if (c === "\\") {
        const e = text[j + 1];
        if (e === undefined) return CUT;
        if (e === "u") {
          for (let k = 0; k < 4; k++) {
            const h = text[j + 2 + k];
            if (h === undefined) return CUT;
            if (!((h >= "0" && h <= "9") || (h >= "a" && h <= "f") || (h >= "A" && h <= "F"))) return BAD;
          }
          j += 6;
          continue;
        }
        if (!'"\\/bfnrt'.includes(e)) return BAD;
        j += 2;
        continue;
      }
      // JSON forbids unescaped control characters in strings, and V8 agrees.
      if (c < " ") return BAD;
      j++;
    }
  };

  const scanNumber = (): number => {
    let j = i;
    if (text[j] === "-") j++;
    if (j >= n) return CUT;
    if (text[j] === "0") j++;
    else if (isDigit(text[j])) while (isDigit(text[j])) j++;
    else return BAD;
    // A leading zero followed by a digit is left for the caller: `01` scans as
    // `0` and the stranded `1` fails as a missing delimiter, which is how
    // JSON.parse reports it too.
    if (text[j] === ".") {
      j++;
      if (j >= n) return CUT;
      if (!isDigit(text[j])) return BAD;
      while (isDigit(text[j])) j++;
    }
    if (text[j] === "e" || text[j] === "E") {
      j++;
      if (text[j] === "+" || text[j] === "-") j++;
      if (j >= n) return CUT;
      if (!isDigit(text[j])) return BAD;
      while (isDigit(text[j])) j++;
    }
    return j;
  };

  const scanLiteral = (): number => {
    for (const lit of ["true", "false", "null"]) {
      if (text.startsWith(lit, i)) return i + lit.length;
      // Ran out of input part-way through: `nul` can still become `null`.
      // `nope` cannot become anything, and reaches BAD below.
      if (i + lit.length > n && lit.startsWith(text.slice(i))) return CUT;
    }
    return BAD;
  };

  for (;;) {
    while (i < n && (text[i] === " " || text[i] === "\t" || text[i] === "\n" || text[i] === "\r")) i++;

    if (i >= n) {
      // Nothing left to read. Only a closed top-level value is a whole
      // document; anything else -- including empty input, which is what an
      // interrupted stream leaves behind -- was cut short.
      return mode === "afterValue" && stack.length === 0 ? "complete" : "truncated";
    }

    const c = text[i]!;

    switch (mode) {
      case "valueOrArrayEnd":
        if (c === "]") {
          stack.pop();
          i++;
          mode = "afterValue";
          break;
        }
        mode = "value";
        continue; // re-dispatch this same character as a value

      case "keyOrObjectEnd":
        if (c === "}") {
          stack.pop();
          i++;
          mode = "afterValue";
          break;
        }
        mode = "key";
        continue;

      case "key": {
        if (c !== '"') return "malformed";
        const j = scanString();
        if (j === CUT) return "truncated";
        if (j === BAD) return "malformed";
        i = j;
        mode = "colon";
        break;
      }

      case "colon":
        if (c !== ":") return "malformed";
        i++;
        mode = "value";
        break;

      case "value": {
        if (c === "{") {
          stack.push("object");
          i++;
          mode = "keyOrObjectEnd";
          break;
        }
        if (c === "[") {
          stack.push("array");
          i++;
          mode = "valueOrArrayEnd";
          break;
        }
        const j = c === '"' ? scanString() : c === "-" || isDigit(c) ? scanNumber() : scanLiteral();
        if (j === CUT) return "truncated";
        if (j === BAD) return "malformed";
        i = j;
        mode = "afterValue";
        break;
      }

      case "afterValue": {
        const top = stack[stack.length - 1];
        // Content after a closed top-level value: no suffix repairs that.
        if (top === undefined) return "malformed";
        if (c === ",") {
          i++;
          mode = top === "object" ? "key" : "value";
          break;
        }
        if (c === (top === "object" ? "}" : "]")) {
          stack.pop();
          i++;
          mode = "afterValue";
          break;
        }
        return "malformed";
      }
    }
  }
}

/** Recursive Object.freeze, preserving the literal's inferred type. */
function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object") {
    for (const key of Object.getOwnPropertyNames(value)) {
      deepFreeze((value as Record<string, unknown>)[key]);
    }
    Object.freeze(value);
  }
  return value;
}
