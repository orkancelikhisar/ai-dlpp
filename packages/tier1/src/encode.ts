import type { WordSpan } from "./words.js";

/**
 * The subword tokenizer, reduced to the two things the encoder needs of it.
 *
 * Structural rather than an import of `@huggingface/transformers`, for two
 * reasons. The encoder's whole job is array bookkeeping, and a stub that
 * returns zero subwords for one input exercises the case that matters without
 * the 1.4 GB of weights on disk. And nothing here should be able to reach for
 * an offsets API: that library has never had one, and the old design for this
 * seam assumed it did.
 */
export interface SubwordTokenizer {
  /** Subword ids for one word, with no special-token wrapper around them. */
  encodeWord(word: string): readonly number[];
  /** The id the pinned post-processor puts at the FRONT of every sequence. */
  readonly clsId: number;
  /** ...and at the END. */
  readonly sepId: number;
}

/**
 * Wraps a raw encode function, reading the special-token ids off an encode of
 * the EMPTY STRING rather than taking them as arguments.
 *
 * That is the whole point: the ids differ per model and hardcoding them would
 * survive a re-pin silently. Measured on the two pinned tokenizers through
 * @huggingface/transformers 3.8.1 -- encoding "" with specials gives
 * [50281, 50282] on gliner-pii-edge and [1, 2] on gliner-pii-base, so a
 * constant lifted from either one would be wrong for the other.
 */
export function tokenizerFromEncoder(
  encode: (text: string, addSpecialTokens: boolean) => readonly number[],
): SubwordTokenizer {
  const wrapper = encode("", true);
  if (wrapper.length !== 2) {
    throw new Error(
      `expected a two-id special-token wrapper from an empty encode, got ${wrapper.length} ids`,
    );
  }
  return {
    clsId: wrapper[0] as number,
    sepId: wrapper[1] as number,
    encodeWord: (word) => encode(word, false),
  };
}

/** The class prompt that precedes the message words in the same sequence. */
export interface EncodePrompt {
  /** One label per tier-1 class, in the order the logits' class axis uses. */
  readonly labels: readonly string[];
  /** gliner_config.json `ent_token`, measured as "<<ENT>>" on both models. */
  readonly entToken: string;
  /** gliner_config.json `sep_token`, measured as "<<SEP>>" on both models. */
  readonly sepToken: string;
}

export interface EncodeOptions {
  readonly prompt?: EncodePrompt;
  /**
   * The manifest's `maxLen` (2048 on all six pinned models). Bounds the whole
   * sequence, specials and prompt included. Omitted means no truncation.
   */
  readonly maxLen?: number;
}

export interface EncodedWords {
  readonly inputIds: readonly number[];
  readonly attentionMask: readonly number[];
  /** 1-based word slot on each word's FIRST subword, 0 everywhere else. */
  readonly wordsMask: readonly number[];
  /** The SURVIVING words, still carrying their offsets into the message. */
  readonly words: readonly WordSpan[];
  /** What `text_lengths` must be fed. Always `words.length`; see below. */
  readonly textLengths: number;
  /** How many words tokenised to nothing. A count, never the text. */
  readonly droppedWords: number;
  /**
   * How many trailing words `maxLen` cut off, counting from the first that did
   * not fit. A count, never the text.
   */
  readonly truncatedWords: number;
}

/**
 * Turns a word list into the four feeds every pinned graph takes.
 *
 * ## The lockstep invariant, and the bug it exists to avoid
 *
 * BOTH reference implementations desync here. GLiNER.js's `encodeInputs`
 * advances its word counter `c` inside `wordTokens.forEach`, so a word that
 * tokenises to zero subwords never advances it -- while `text_lengths` comes
 * from `tokens.length`, which counted that word, and the parallel
 * `wordsStartIdx`/`wordsEndIdx` arrays still hold its offsets. Word-axis index
 * i then carries the (i+1)-th SURVIVING word while offset array index i names
 * the (i+1)-th word overall, and every span from the dropped word onward names
 * its neighbour.
 *
 * It is reachable with one invisible character. Measured on the pinned
 * gliner-pii-base tokenizer: U+0001, U+0002, U+0007, U+000B, U+001A, U+007F
 * and U+FEFF all encode to zero subwords, and all but U+000B are `\S` to the
 * Python splitter, so they ARE words. (Edge's tokenizer returned zero subwords
 * for none of them -- the hazard is real on one rung of the ladder, which is
 * exactly the kind of difference a shared encoder must not depend on.) This is
 * the "neighbouring word to the vault, real value left in the message" failure
 * the whole system exists to prevent.
 *
 * The fix is to drop such a word from `words` and from the slot numbering
 * together, which is why both come out of the same loop below.
 *
 * ## Why `textLengths` is `words.length` and not `max(wordsMask)`
 *
 * Task 7 measured this on the real graphs, and the two are only the same
 * number because of the drop above. `text_lengths` ALLOCATES the word-axis
 * slots: fed 9 with a highest words_mask of 6, edge returns a 9-word axis and
 * pads; fed 4 with a highest words_mask of 6, it fails inside ScatterND with
 * `invalid indice found, indice = 4`. So a slot number with no seat is a hard
 * error rather than a silently dropped word, and the invariant this encoder
 * has to hold is against `text_lengths`.
 *
 * ## Why the FIRST subword
 *
 * `subtoken_pooling` is `"first"` in all six pinned gliner_config.json files,
 * and Task 7 measured that as load-bearing rather than decorative: marking the
 * LAST subword instead runs without error and returns identical shapes while
 * dropping the baseline person score from 0.75 to 0.40 and the email score
 * from 0.82 to 0.05. There is no exception to catch.
 *
 * ## Why the prompt needs no equivalent guard
 *
 * A LABEL that tokenises to nothing would be the same hazard one axis over --
 * a class index meaning a different entity type than the caller thinks. It is
 * not: measured on the pinned base graph, a prompt of three labels one of
 * which is U+0001 (zero subwords on that tokenizer) still returns a logits
 * class axis of extent 3. The axis is sized by the `<<ENT>>` markers, which
 * survive an empty label, so an empty label costs its class its description
 * and shifts nothing.
 *
 * ## The U+FEFF accuracy gap, measured and left open
 *
 * On the pinned base tokenizer through @huggingface/transformers 3.8.1, U+FEFF
 * encodes to zero subwords while the visually identical U+200B, U+200C and
 * U+200D all encode to [507]. So a BOM inside a message is dropped here and
 * the model never sees it. With the lockstep drop that is an ACCURACY gap and
 * never an offset gap -- every surviving word still slices out of the message
 * -- but it does mean a name split by a BOM reads to the model as two words
 * with nothing between them. Not repaired by substituting an id, because no
 * measurement here says which id the training-time Rust tokenizer used.
 */
export function encodeWords(
  tokenizer: SubwordTokenizer,
  words: readonly WordSpan[],
  options: EncodeOptions = {},
): EncodedWords {
  const inputIds: number[] = [tokenizer.clsId];
  const wordsMask: number[] = [0];

  // The prompt. Slot 0 throughout: these positions are read by the class head,
  // not pooled into a word, and a non-zero slot here would claim a word seat
  // that no message word could then use.
  const prompt = options.prompt;
  if (prompt !== undefined) {
    const promptWords: string[] = [];
    for (const label of prompt.labels) promptWords.push(prompt.entToken, label);
    promptWords.push(prompt.sepToken);
    for (const promptWord of promptWords) {
      for (const id of tokenizer.encodeWord(promptWord)) {
        inputIds.push(id);
        wordsMask.push(0);
      }
    }
  }

  // Room for the [SEP] that closes the sequence: `maxLen` bounds the whole
  // tensor, so the last word has to fit with the terminator still to come.
  const budget = options.maxLen === undefined ? Infinity : options.maxLen - 1;
  if (inputIds.length > budget) {
    // No encoding of ANY message exists under these two, so there is nothing
    // to return that a caller could sensibly feed. Counts only, never text:
    // the labels come from the policy and the message never reaches here.
    throw new Error(
      `maxLen ${String(options.maxLen)} is too small for a ${String(inputIds.length)}-token ` +
        `prompt plus its terminator`,
    );
  }

  const surviving: WordSpan[] = [];
  let droppedWords = 0;
  let truncatedWords = 0;
  for (let w = 0; w < words.length; w += 1) {
    const word = words[w] as WordSpan;
    const ids = tokenizer.encodeWord(word.text);
    if (ids.length === 0) {
      // LOCKSTEP: not counted, not seated, and not kept in `surviving` either.
      droppedWords += 1;
      continue;
    }
    if (inputIds.length + ids.length > budget) {
      // LOCKSTEP again, from the other end, and a BREAK rather than a skip: the
      // model reads a contiguous prefix of the message, so keeping a later
      // short word after dropping this one would feed it a sentence the user
      // never wrote. A word is also taken whole or not at all -- half a word's
      // subwords would pool a partial vector into a seat whose offsets still
      // describe the whole word.
      truncatedWords = words.length - w;
      break;
    }
    surviving.push(word);
    const slot = surviving.length; // 1-based, per the reference's convention.
    for (let k = 0; k < ids.length; k += 1) {
      inputIds.push(ids[k] as number);
      wordsMask.push(k === 0 ? slot : 0);
    }
  }
  inputIds.push(tokenizer.sepId);
  wordsMask.push(0);

  return {
    inputIds,
    // All ones. Nothing here pads: a batch of one has nothing to pad against,
    // and the runtime path runs one message at a time.
    attentionMask: inputIds.map(() => 1),
    wordsMask,
    words: surviving,
    textLengths: surviving.length,
    droppedWords,
    truncatedWords,
  };
}

export interface SpanEnumeration {
  /** `[start, end]` per span, with an INCLUSIVE end. */
  readonly spanIdx: readonly (readonly [number, number])[];
  /** False where a span runs off the end of the text. */
  readonly spanMask: readonly boolean[];
}

/**
 * The rectangular span enumeration the markerV0 graph takes.
 *
 * Three things about it are measured rather than chosen. Task 7 fed the pinned
 * base graph `text_lengths` 9 with span_idx still enumerated over 6 words and
 * it failed inside `/core/span_rep_layer/.../Reshape_2`, so the enumeration is
 * sized from `text_lengths` and nothing else. It fed `max_width` 4 instead of
 * 12 and got the same Reshape failure, so the width is baked into the export
 * and is not a runtime knob. And base's peaks put the person class highest at
 * start word 1, width index 1, on a sentence whose name is words 1-2 -- so the
 * stored end is `start + width`, INCLUSIVE, and width index 0 is a one-word
 * span.
 *
 * The enumeration has to stay rectangular to be a tensor, so spans running off
 * the end are enumerated and switched off by the mask. GLiNER.js instead clamps
 * the end with `Math.min(i + j, textLength - 1)` and then tests
 * `endIdx < textLength`, which is true for every clamped span -- its span_mask
 * is all true and each overrunning span duplicates a real one.
 */
export function enumerateSpans(textLengths: number, maxWidth: number): SpanEnumeration {
  const spanIdx: [number, number][] = [];
  const spanMask: boolean[] = [];
  for (let start = 0; start < textLengths; start += 1) {
    for (let width = 0; width < maxWidth; width += 1) {
      const end = start + width;
      spanIdx.push([start, end]);
      spanMask.push(end < textLengths);
    }
  }
  return { spanIdx, spanMask };
}
