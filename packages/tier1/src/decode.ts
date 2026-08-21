/**
 * Model output tensor -> candidate spans, for both tier-1 span modes.
 *
 * Deliberately free of onnxruntime: these take a `Float32Array` and the
 * extents, so they run in Node under vitest against tensors written by hand,
 * and the browser path (Task 10) is the only place a real `ort.Tensor` is
 * unwrapped. The batch axis is not a parameter -- a caller slices one row out
 * and passes the rest.
 *
 * ## Why there are two of these
 *
 * The two model families express a span differently, and NEITHER graph's
 * declared axis names describe the tensor it returns. `torch.onnx.export`
 * wrote whatever `dynamic_axes` said, so edge declares `[position, batch_size,
 * sequence_length, num_classes]` and base declares `[batch_size,
 * sequence_length, num_spans, num_classes]`, and both are wrong. What is
 * decoded here is the layout `scripts/probe-model.ts` measured by running the
 * pinned weights and varying one feed dimension at a time; it is committed at
 * `test/fixtures/model-signature.json`, and `test/decode.test.ts` decodes
 * through that fixture's axis roles so a re-pin that moves an axis fails
 * rather than silently mis-reads.
 *
 *   token_level (edge): [batch, words, classes, 3]  -- start/end/inside scores
 *   markerV0    (base): [batch, words, 12, classes] -- inclusive width offsets
 *
 * ## What these do NOT do
 *
 * Overlaps are not resolved. Core's merge weighs candidates from every tier
 * together, severity-first and then by confidence, so a tier that dropped its
 * own overlapping candidates first would have made a choice that is not its to
 * make. Both decoders therefore return every candidate whose own evidence
 * clears the threshold, including candidates that overlap each other.
 *
 * Padding is not detected either. MEASURED (see the fixture): the word axis is
 * sized by the `text_lengths` input, not by `max(words_mask)`, and edge
 * accepts `text_lengths` 9 against six real words and returns a nine-long word
 * axis. Nothing in the tensor marks the three padded slots, so keeping
 * `text_lengths` equal to the real word count is the encoder's invariant.
 */

/** One candidate, in WORD indices, inclusive at both ends. */
export interface DecodedSpan {
  readonly firstWord: number;
  readonly lastWord: number;
  readonly classIndex: number;
  readonly score: number;
}

/** Extents of one batch row of a `token_level` logits tensor. */
export interface EdgeLogitsDims {
  readonly words: number;
  readonly classes: number;
  readonly slots: number;
}

/** Extents of one batch row of a `markerV0` logits tensor. */
export interface BaseLogitsDims {
  readonly words: number;
  /** Inclusive width offsets per word. Baked into the export at 12. */
  readonly widths: number;
  readonly classes: number;
}

/**
 * The trailing axis of a `token_level` tensor, and what its three slots are.
 *
 * MEASURED on `gliner-pii-edge`, with the labels person/email and these
 * sentences, reading sigmoid of the raw logits:
 *
 * - "Contact Priya Anjali Sharma at priya@acme.io today" -- person start peaks
 *   on "Priya" (0.822) and person end on "Sharma" (0.846), while the middle
 *   word "Anjali" scores only 0.107/0.196 on those two and 0.883 on slot 2,
 *   the highest of any word. Slot 2 fires on a word that is neither a start
 *   nor an end but is part of the entity.
 * - "Email Priya Sharma and Rahul Mehta before Friday" -- slot 2 is 0.859,
 *   0.797, 0.794 and 0.769 on the four name words and 0.091 on "and" between
 *   them. That dip is what distinguishes an inside score from a per-word
 *   entity score.
 * - "Ask Priya about it" -- the one-word person scores 0.723 / 0.728 / 0.690.
 *   All three slots fire on a width-1 entity, so slot 2 is not interior-only.
 *
 * The uint8 rung reproduces the pattern with degraded magnitudes (0.316 on
 * "and" against 0.626-0.787 on the names). Every number above is pinned per
 * rung under `slotSemantics` in the fixture, whole rows rather than the peaks
 * alone, because what slot 2 is shows in which words it is high ON and not in
 * which one word is highest. The two fp16 rungs are absent from that
 * block because they do not load -- Task 7's measurement, recorded in the same
 * fixture as `loadError`, not something re-run here.
 */
export const EDGE_SLOTS = 3;
export const EDGE_SLOT_START = 0;
export const EDGE_SLOT_END = 1;
export const EDGE_SLOT_INSIDE = 2;

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

/**
 * Fails a tensor whose length disagrees with the extents it was described by.
 *
 * Nothing downstream would. A too-LONG array is read at the right strides and
 * quietly truncated. A too-SHORT one is worse: MEASURED, an out-of-range index
 * on a `Float32Array` is `undefined`, `sigmoid(undefined)` is `NaN`, and `NaN <
 * threshold` is `false` -- so every cell past the end passes the filter and is
 * emitted as a span carrying a `NaN` score. This is the only cheap place to
 * catch either, before the first read.
 */
function requireLength(logits: Float32Array, expected: number, layout: string): void {
  if (logits.length !== expected) {
    throw new Error(
      `tier-1 logits length ${logits.length} disagrees with ${layout}, which needs ${expected}`,
    );
  }
}

/**
 * Descending score, then ascending position, then ascending class.
 *
 * A stable sort alone would not give one order, because the order it preserves
 * is each decoder's emission order and those differ: `decodeBaseSpans` walks
 * word-major and already emits in position order, while `decodeEdgeSpans` walks
 * class-major and emits 0:0, 2:0, 0:1, 2:1 where this comparator wants 0:0,
 * 0:1, 2:0, 2:1. Two runs of the eval harness diffed against each other need
 * the same order out of both, so it is stated rather than inherited.
 */
function byScoreThenPosition(a: DecodedSpan, b: DecodedSpan): number {
  return (
    b.score - a.score ||
    a.firstWord - b.firstWord ||
    a.lastWord - b.lastWord ||
    a.classIndex - b.classIndex
  );
}

/**
 * `markerV0` logits -> spans. Axis 2 is an INCLUSIVE width offset: index `w` at
 * word `i` is the span `i..i+w`, so index 0 is a one-word span.
 *
 * That reading is the model's, not a convention: MEASURED on the probe
 * sentence, the highest person score sits at word 1, offset 1, and words 1..2
 * are "Priya Sharma"; the email at word 4 sits at offset 0. A length reading
 * would put the name at offset 2.
 */
export function decodeBaseSpans(
  logits: Float32Array,
  dims: BaseLogitsDims,
  threshold: number,
): DecodedSpan[] {
  const { words, widths, classes } = dims;
  requireLength(
    logits,
    words * widths * classes,
    `[words ${words}, widths ${widths}, classes ${classes}]`,
  );

  const out: DecodedSpan[] = [];
  for (let firstWord = 0; firstWord < words; firstWord += 1) {
    for (let width = 0; width < widths; width += 1) {
      const lastWord = firstWord + width;
      // The [word x width] grid is rectangular because it is a tensor, so its
      // far corner always describes spans running past the end of the text.
      // MEASURED on gliner-pii-base: those cells are not zeroed, they carry a
      // per-class constant (0.032 for person and 0.042 for email on an
      // eight-word feed), which a low enough threshold would admit.
      if (lastWord >= words) break;
      const base = (firstWord * widths + width) * classes;
      for (let classIndex = 0; classIndex < classes; classIndex += 1) {
        // Per-class sigmoid rather than a softmax across classes. The classes
        // are a policy's entity types and they are not mutually exclusive --
        // one span can be both a client name and a project codename -- so
        // making them compete would suppress the weaker of two real findings.
        const score = sigmoid(logits[base + classIndex]!);
        if (score < threshold) continue;
        out.push({ firstWord, lastWord, classIndex, score });
      }
    }
  }
  return out.sort(byScoreThenPosition);
}

/**
 * `token_level` logits -> spans, by pairing start and end slots of one class.
 *
 * A span `i..j` is proposed when, for one class, all of these clear the
 * threshold: slot 0 at `i`, slot 1 at `j`, and slot 2 at EVERY word from `i`
 * to `j`. Its score is the smallest of them.
 *
 * ## Why every pair, and not the nearest end
 *
 * Real output has many peaks, and start/end alone do not say which start goes
 * with which end. Taking the nearest qualifying end scores better on the seven
 * sentences this task measured -- 3 false positives against 10 -- and is still
 * the wrong rule here, because all the pairs sharing a start word overlap each
 * other, so choosing between them is overlap resolution, which belongs to
 * core's merge and not to one tier.
 *
 * The inside slot is what makes that affordable. MEASURED on "Email Priya
 * Sharma and Rahul Mehta before Friday": start peaks at "Priya" and "Rahul",
 * end peaks at "Sharma" and "Mehta", and start/end alone admit a third pair,
 * 1..5, that swallows both names and the word between them at score 0.782.
 * Slot 2 is 0.091 on "and" and rejects it, on the model's own evidence about
 * that word rather than on a comparison between candidates. Aggregating slot 2
 * by mean instead of min does NOT reject it (the mean over words 1..5 is
 * 0.662): one dip is exactly what a mean absorbs.
 *
 * ## Why the score is a minimum
 *
 * `Tier1Config.threshold` is one number shared by both rungs, so an edge score
 * has to be on the same 0..1 scale as base's single sigmoid or the same
 * configured value silently means something stricter here. MEASURED at
 * threshold 0.5 over those seven sentences: multiplying the three components
 * instead lost 6 of the 9 true spans that the minimum kept, because three
 * components at 0.8 multiply to 0.512. A minimum also states what it means --
 * a span is as confident as the weakest thing the model said about it -- and
 * makes the score and the filter the same rule rather than two.
 */
export function decodeEdgeSpans(
  logits: Float32Array,
  dims: EdgeLogitsDims,
  threshold: number,
): DecodedSpan[] {
  const { words, classes, slots } = dims;
  requireLength(
    logits,
    words * classes * slots,
    `[words ${words}, classes ${classes}, slots ${slots}]`,
  );
  if (slots !== EDGE_SLOTS) {
    throw new Error(
      `tier-1 token_level logits must carry ${EDGE_SLOTS} slots per word and class, got ${slots}`,
    );
  }

  const at = (word: number, classIndex: number, slot: number): number =>
    sigmoid(logits[(word * classes + classIndex) * slots + slot]!);

  const out: DecodedSpan[] = [];
  for (let classIndex = 0; classIndex < classes; classIndex += 1) {
    for (let firstWord = 0; firstWord < words; firstWord += 1) {
      const start = at(firstWord, classIndex, EDGE_SLOT_START);
      if (start < threshold) continue;
      // Running minimum of the inside slot over firstWord..lastWord. It only
      // ever falls, so the first word below the threshold ends every span that
      // could start here, and the walk stops there rather than running to the
      // end of the text. Not benchmarked; what suggests the cut bites is that
      // slot 2 measured 0.018-0.091 on the non-entity words of the probe
      // sentences, well under any workable threshold.
      let insideRun = Infinity;
      for (let lastWord = firstWord; lastWord < words; lastWord += 1) {
        insideRun = Math.min(insideRun, at(lastWord, classIndex, EDGE_SLOT_INSIDE));
        if (insideRun < threshold) break;
        const end = at(lastWord, classIndex, EDGE_SLOT_END);
        if (end < threshold) continue;
        out.push({
          firstWord,
          lastWord,
          classIndex,
          score: Math.min(start, end, insideRun),
        });
      }
    }
  }
  return out.sort(byScoreThenPosition);
}
