import { describe, expect, it } from "vitest";
import {
  encodeWords,
  enumerateSpans,
  tokenizerFromEncoder,
  type SubwordTokenizer,
} from "../src/encode.js";
import { splitWords } from "../src/words.js";

const SOH = "\x01";
const BOM = "\u{FEFF}";

const CLS = 900;
const SEP = 901;

/**
 * A tokenizer stub with the one behaviour that matters here: some inputs
 * encode to nothing.
 *
 * MEASURED on the pinned gliner-pii-base tokenizer through
 * @huggingface/transformers 3.8.1: U+0001, U+0002, U+0007, U+000B, U+001A,
 * U+007F and U+FEFF all come back as zero subwords, while every ordinary word
 * comes back with at least one. Edge's tokenizer returned zero for none of
 * them. This stub reproduces that shape without needing the 1.4 GB of weights
 * on disk.
 */
const ZERO_SUBWORD = new Set([SOH, BOM, "\x07", "\x7F"]);

function fakeEncode(text: string, addSpecialTokens: boolean): readonly number[] {
  const body: number[] = [];
  if (!ZERO_SUBWORD.has(text)) {
    // One id per character, so a word's subword count is its length: enough to
    // tell "first subword" from "the rest" apart.
    for (const ch of text) body.push(ch.codePointAt(0) ?? 0);
  }
  return addSpecialTokens ? [CLS, ...body, SEP] : body;
}

const tokenizer: SubwordTokenizer = tokenizerFromEncoder(fakeEncode);

const PROMPT = { labels: ["person", "email"], entToken: "<<ENT>>", sepToken: "<<SEP>>" };

describe("tokenizerFromEncoder", () => {
  it("discovers the special-token ids instead of taking them as arguments", () => {
    expect(tokenizer.clsId).toBe(CLS);
    expect(tokenizer.sepId).toBe(SEP);
  });

  it("refuses a wrapper that is not exactly two ids", () => {
    expect(() => tokenizerFromEncoder(() => [1, 2, 3])).toThrow(/wrapper/i);
    expect(() => tokenizerFromEncoder(() => [])).toThrow(/wrapper/i);
  });

  it("strips the wrapper from every word encode", () => {
    expect(tokenizer.encodeWord("ab")).toEqual([97, 98]);
  });
});

describe("encodeWords", () => {
  it("drops zero-subword words from the word list and the offsets together", () => {
    // The reference bug: a word tokenizing to nothing desyncs the offset arrays
    // and every later span names the wrong word.
    const encoded = encodeWords(tokenizer, splitWords("call " + SOH + " Acme Corp"));
    expect(encoded.words).toHaveLength(Math.max(...encoded.wordsMask));
    expect(encoded.words.map((w) => w.text)).toEqual(["call", "Acme", "Corp"]);
    expect(encoded.droppedWords).toBe(1);
  });

  it("keeps the surviving words' character offsets pointing at the original text", () => {
    // The whole point of the lockstep drop: word slot k must still be sliceable
    // out of the message the user typed.
    const text = "call " + SOH + " Acme Corp";
    const encoded = encodeWords(tokenizer, splitWords(text));
    expect(encoded.words.map((w) => text.slice(w.start, w.end))).toEqual([
      "call",
      "Acme",
      "Corp",
    ]);
    expect(encoded.words[1]).toEqual({ text: "Acme", start: 7, end: 11 });
  });

  it("sizes text_lengths by the surviving words, which is what the graph allocates", () => {
    // Task 7, measured on the pinned graphs: the word axis is sized by
    // text_lengths, and a words_mask value with no slot fails inside ScatterND
    // rather than being dropped. So these two have to agree exactly.
    const encoded = encodeWords(tokenizer, splitWords("call " + SOH + " Acme " + BOM + " Corp"));
    expect(encoded.textLengths).toBe(encoded.words.length);
    expect(encoded.textLengths).toBe(Math.max(...encoded.wordsMask));
    expect(encoded.droppedWords).toBe(2);
  });

  it("marks the FIRST subword of each word and nothing else", () => {
    // gliner_config.json says subtoken_pooling "first" on all six pinned
    // models, and Task 7 measured that marking the last subword instead runs
    // cleanly while collapsing accuracy -- there is no error to catch it.
    const encoded = encodeWords(tokenizer, splitWords("ab cd"));
    // [CLS] a b c d [SEP] with the stub's one-id-per-character encoding.
    expect(encoded.inputIds).toEqual([CLS, 97, 98, 99, 100, SEP]);
    expect(encoded.wordsMask).toEqual([0, 1, 0, 2, 0, 0]);
  });

  it("gives every prompt token slot 0 and numbers the text words from 1", () => {
    const encoded = encodeWords(tokenizer, splitWords("ab"), { prompt: PROMPT });
    const promptLength = fakeEncode("<<ENT>>person<<ENT>>email<<SEP>>", false).length;
    expect(encoded.wordsMask.slice(0, 1 + promptLength)).toEqual(
      new Array(1 + promptLength).fill(0),
    );
    expect(encoded.wordsMask.filter((s) => s !== 0)).toEqual([1]);
    expect(encoded.textLengths).toBe(1);
  });

  it("wraps the sequence in the discovered specials, never hardcoded ids", () => {
    const encoded = encodeWords(tokenizer, splitWords("ab"), { prompt: PROMPT });
    expect(encoded.inputIds[0]).toBe(CLS);
    expect(encoded.inputIds[encoded.inputIds.length - 1]).toBe(SEP);
    expect(encoded.wordsMask[0]).toBe(0);
    expect(encoded.wordsMask[encoded.wordsMask.length - 1]).toBe(0);
  });

  it("keeps all three parallel arrays the same length, with attention all on", () => {
    const encoded = encodeWords(tokenizer, splitWords("call Acme Corp today"), {
      prompt: PROMPT,
    });
    expect(encoded.attentionMask).toHaveLength(encoded.inputIds.length);
    expect(encoded.wordsMask).toHaveLength(encoded.inputIds.length);
    expect(new Set(encoded.attentionMask)).toEqual(new Set([1]));
  });

  it("numbers the word slots contiguously from 1, with no gap at a dropped word", () => {
    const encoded = encodeWords(tokenizer, splitWords("a " + SOH + " b " + SOH + " c"), {
      prompt: PROMPT,
    });
    expect(encoded.wordsMask.filter((s) => s !== 0)).toEqual([1, 2, 3]);
    expect(encoded.words.map((w) => w.text)).toEqual(["a", "b", "c"]);
  });

  it("truncates trailing words and the word list in lockstep", () => {
    // max_len is 2048 on all six pinned models. Truncating the ids without
    // truncating the words is the same desync as the zero-subword one, just
    // reached from the other end.
    const words = splitWords("aaa bbb ccc ddd");
    const untruncated = encodeWords(tokenizer, words);
    const encoded = encodeWords(tokenizer, words, { maxLen: untruncated.inputIds.length - 4 });
    expect(encoded.inputIds.length).toBeLessThanOrEqual(untruncated.inputIds.length - 4);
    expect(encoded.words.map((w) => w.text)).toEqual(["aaa", "bbb"]);
    expect(encoded.textLengths).toBe(2);
    expect(Math.max(...encoded.wordsMask)).toBe(2);
    expect(encoded.truncatedWords).toBe(2);
  });

  it("never splits a word across the truncation boundary", () => {
    const words = splitWords("aaa bbb ccc");
    for (let maxLen = 2; maxLen <= 14; maxLen += 1) {
      const encoded = encodeWords(tokenizer, words, { maxLen });
      expect(encoded.inputIds.length).toBeLessThanOrEqual(maxLen);
      // Every surviving word contributed all of its subwords.
      const marked = encoded.wordsMask.filter((s) => s !== 0).length;
      expect(marked).toBe(encoded.words.length);
      const subwords = encoded.words.reduce((n, w) => n + tokenizer.encodeWord(w.text).length, 0);
      expect(encoded.inputIds.length).toBe(2 + subwords);
    }
  });

  it("stops at the first word that does not fit, rather than skipping it", () => {
    // A skip instead of a break would feed the model a sentence the user never
    // wrote -- word 3 sitting where word 2 belongs -- and the offsets would
    // still be right for each surviving word, so nothing downstream could tell.
    const words = splitWords("aa bbbbbb c");
    const encoded = encodeWords(tokenizer, words, { maxLen: 6 });
    expect(encoded.words.map((w) => w.text)).toEqual(["aa"]);
    expect(encoded.textLengths).toBe(1);
    expect(encoded.truncatedWords).toBe(2);
  });

  it("keeps the surviving words a contiguous prefix of the message", () => {
    for (const maxLen of [4, 6, 8, 10, 12, 40]) {
      const words = splitWords("aa bbbbbb c dddd e");
      const encoded = encodeWords(tokenizer, words, { maxLen });
      const wanted = words.slice(0, encoded.words.length + encoded.droppedWords);
      expect(encoded.words.map((w) => w.start)).toEqual(
        wanted.filter((w) => tokenizer.encodeWord(w.text).length > 0).map((w) => w.start),
      );
    }
  });

  it("reports counts, never the text it dropped", () => {
    // Error and diagnostic surfaces never echo a confidential-shaped value.
    const encoded = encodeWords(tokenizer, splitWords("secret@acme.io " + SOH), { maxLen: 6 });
    expect(JSON.stringify({ d: encoded.droppedWords, t: encoded.truncatedWords })).not.toContain(
      "acme",
    );
  });

  it("refuses a maxLen the prompt alone cannot fit inside", () => {
    // config.ts already reasons about this collision: 100 classes in the
    // nlDefinition label form cost 2,600-3,700 tokens against a maxLen of
    // 2048. Returning a sequence longer than maxLen would be a silently
    // invalid feed; there is no shorter one to return either.
    expect(() => encodeWords(tokenizer, splitWords("ab"), { prompt: PROMPT, maxLen: 4 })).toThrow(
      /maxLen 4 is too small/,
    );
  });

  it("handles a text with no words at all", () => {
    const encoded = encodeWords(tokenizer, splitWords("   "), { prompt: PROMPT });
    expect(encoded.words).toEqual([]);
    expect(encoded.textLengths).toBe(0);
    expect(encoded.wordsMask.filter((s) => s !== 0)).toEqual([]);
  });
});

describe("enumerateSpans", () => {
  it("enumerates every start against every width, with an inclusive end", () => {
    // Task 7, measured: base's peaks put the person class highest at start word
    // 1 width 1 on a sentence whose name is words 1-2, so width 0 is a one-word
    // span and the stored end is start + width, inclusive.
    const { spanIdx, spanMask } = enumerateSpans(3, 2);
    expect(spanIdx).toEqual([
      [0, 0],
      [0, 1],
      [1, 1],
      [1, 2],
      [2, 2],
      [2, 3],
    ]);
    expect(spanMask).toEqual([true, true, true, true, true, false]);
  });

  it("stays rectangular so it can be a tensor, masking the overruns off", () => {
    // GLiNER.js clamps the end with Math.min(i + j, textLength - 1) and then
    // tests `endIdx < textLength`, which is true for every clamped span -- so
    // its span_mask is all true and every overrunning span is a duplicate of a
    // real one. Neither is true here.
    const { spanIdx, spanMask } = enumerateSpans(4, 12);
    expect(spanIdx).toHaveLength(48);
    expect(spanMask).toHaveLength(48);
    expect(spanMask.filter((m) => m)).toHaveLength(4 + 3 + 2 + 1);
    for (let i = 0; i < spanIdx.length; i += 1) {
      const [start, end] = spanIdx[i] as [number, number];
      expect(spanMask[i]).toBe(end < 4);
      expect(end).toBe(start + (i % 12));
    }
  });

  it("sizes the enumeration from text_lengths, not from a padded word count", () => {
    // Measured by Task 7 on the pinned base graph: text_lengths 9 with span_idx
    // still enumerated over 6 words fails inside the span_rep_layer Reshape,
    // while enumerating over all 9 runs. The two counts are the same number.
    expect(enumerateSpans(9, 12).spanIdx).toHaveLength(108);
  });

  it("returns nothing when there are no words", () => {
    expect(enumerateSpans(0, 12)).toEqual({ spanIdx: [], spanMask: [] });
  });
});
