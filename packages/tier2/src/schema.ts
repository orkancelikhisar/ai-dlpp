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
 * UNMEASURED, and the only thing here that is: `type: "number"`. Every field in
 * the probe's schema was a string, so no probe call has exercised a numeric
 * type under this xgrammar build. An uncompilable schema HANGS rather than
 * erroring, so the first browser task to issue a constrained call should
 * confirm this one compiles before a bake-off depends on it.
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
          confidence: { type: "number" },
        },
        required: ["predicateId", "quote", "confidence"],
        additionalProperties: false,
      },
    },
  },
  required: ["findings"],
  additionalProperties: false,
} as const);

export const JudgeResponseSchema = z.object({
  findings: z.array(
    z.object({
      predicateId: z.string().min(1),
      // An empty quote matches at offset 0 of every message, which is a finding
      // pointing at the wrong text. A whitespace-only quote is the same defect
      // one character further along -- a single space matches in almost every
      // message -- so the check is "has a non-whitespace character", not
      // `min(1)`. It must NOT be written as `z.string().trim().min(1)`: zod's
      // .trim() rewrites the value, and the span ladder needs the quote
      // verbatim to find it in the message.
      quote: z.string().refine((s) => s.trim().length > 0, {
        message: "quote must contain a non-whitespace character",
      }),
      // core/src/detect/merge.ts requires tier adapters to validate confidence
      // at the boundary, and core/src/detect/types.ts documents the field as
      // 0..1; this module is that boundary. The bounds are not decoration:
      // JSON has no Infinity literal but `1e999` parses to one, and a
      // non-finite confidence makes every merge comparison false, which makes
      // Array#sort's ordering inconsistent and destroys determinism downstream.
      // MEASURED on zod 4.4.3: `z.number()` already rejects NaN and both
      // infinities, and `.min(0).max(1)` excludes them independently of that.
      //
      // Out-of-range values are REJECTED, not clamped. A model answering on a
      // percentage scale has failed to honour the contract; clamping 95 to 1.0
      // would turn that misunderstanding into a maximally confident finding and
      // hide it from the bake-off's per-arm failure counts.
      confidence: z.number().min(0).max(1),
    }),
  ),
});

export type JudgeResponse = z.infer<typeof JudgeResponseSchema>;

export type ParseResult =
  | { ok: true; value: JudgeResponse }
  | { ok: false; reason: "truncated" | "malformed" | "schema"; detail: string };

/**
 * Parse a model response, distinguishing the three failure modes because they
 * call for different responses: truncation means raise max_tokens or shorten
 * the prompt, malformation means the grammar constraint is not working, and a
 * schema mismatch means the prompt and the schema disagree.
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
 */
export function parseJudgeResponse(raw: string, finishReason?: string): ParseResult {
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (cause) {
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
  const parsed = JudgeResponseSchema.safeParse(json);
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
