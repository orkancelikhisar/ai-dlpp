/**
 * Rehydration: surrogate → real in provider responses (spec §5.4). Pure string
 * layer here; the streaming TransformStream builds on it. SSE/JSON framing is
 * adapter territory (Plan 6) — these functions operate on decoded text.
 */

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * One alternation over all surrogates, longest-first so a surrogate containing
 * another as a prefix is matched whole. Undefined when the map is empty.
 *
 * The `(?<![0-9])` / `(?![0-9])` pair is a digit boundary on the whole
 * alternation, and it is load-bearing rather than defensive. Two distinct
 * hazards, one on each side:
 *
 * - **Right.** The vault's pool-exhaustion ladder mints digit-suffixed
 *   surrogates as a matter of routine ("Vantor", then "Vantor 2"), so a
 *   conversation holding both meets ordinary text like "Vantor 2024 report".
 *   Unboundaried, longest-first matches "Vantor 2" and yields the SECOND
 *   entity's real value spliced onto "024" — a rehydration that misattributes
 *   the mention and mangles the year. With the lookahead "Vantor 2" fails at
 *   the "2|024" seam, the alternation retries (longest-first still holds among
 *   the candidates that pass), "Vantor" matches, and the following space clears.
 * - **Left.** `id-number` surrogates outside the PAN shape scramble to all-digit
 *   strings, so a surrogate is routinely a bare number that can sit inside any
 *   longer figure the model writes. With only a right boundary, a surrogate of
 *   "8842" matched the tail of "1998842" and produced "invoice 1991234 total":
 *   the real id's digits spliced into an unrelated number the user then reads as
 *   authoritative. Both edges are needed; neither alone is a fix.
 *
 * Digits only, deliberately NOT `\w` on either side. A letter boundary would
 * refuse to match "Vantor" inside "Vantors", and models pluralize surrogates
 * freely, so that mention would come back to the user still wearing its fake
 * name — the real value lost entirely. Left-letter glue is likewise left
 * unboundaried on purpose: "ClientVantor" → "ClientGlobex" is a rehydration
 * worth having, not a tolerated accident.
 *
 * Known trades, all pinned in rehydrate.test.ts so they read as decisions:
 *
 * 1. Base-name preference at digit seams: "Vantors" → "Globexs" (one wrong
 *    trailing letter, meaning intact) and the genuinely ambiguous "Vantor 22"
 *    resolves to "Vantor" + " 22" rather than "Vantor 2" + "2".
 * 2. A digit-glued mention rehydrates to NOTHING: "Vantor2024" matches no key,
 *    so the fake name reaches the user. That is the side to fail on — the only
 *    alternative is matching at a digit seam, which is hazard one above.
 * 3. Matching is case-sensitive, so a model that writes "the vantor deal" loses
 *    the rehydration. Case-insensitive matching is not the fix: it would
 *    collapse distinct surrogates onto one key and rehydrate ordinary words
 *    that happen to collide with one.
 *
 * Trades 2 and 3 both fail toward leaving a surrogate visible rather than
 * emitting a wrong real value, which is the right direction for this layer:
 * a stale fake name is confusing, a misattributed real one is a privacy and
 * correctness incident.
 */
export function surrogatePattern(map: Map<string, string>): RegExp | undefined {
  if (map.size === 0) return undefined;
  const alts = [...map.keys()].sort((a, b) => b.length - a.length).map(escapeRegex);
  return new RegExp(`(?<![0-9])(?:${alts.join("|")})(?![0-9])`, "g");
}

export function maxSurrogateLength(map: Map<string, string>): number {
  let max = 0;
  for (const s of map.keys()) max = Math.max(max, s.length);
  return max;
}

export function rehydrateText(text: string, map: Map<string, string>): string {
  const pattern = surrogatePattern(map);
  if (!pattern) return text;
  return text.replace(pattern, (m) => map.get(m)!);
}

/**
 * Streaming rehydration (spec §5.4): the same replacement `rehydrateText` makes
 * over a whole string, made incrementally as provider chunks arrive. The
 * contract is exactly that equivalence — for any chunking of any input, the
 * concatenated output equals `rehydrateText(wholeInput, map)` — and the tests
 * assert it directly at every split point rather than spot-checking cases.
 * Operates on decoded text; SSE/JSON framing is the adapter's job (Plan 6).
 *
 * The whole design follows from one observation: a match found in a partial
 * buffer is not yet a decision. Two things can still overturn it, so a match is
 * only **settled** — replaced and emitted — when both are ruled out:
 *
 * 1. **Its right boundary has not arrived.** `(?![0-9])` passes vacuously at the
 *    end of a string, so a match ending at the buffer edge may be vetoed by the
 *    next chunk's first character. Requiring `end < buffer.length` means the
 *    veto character was real text and the decision is final. Without it,
 *    `["total 8842", "1 units"]` rehydrates the id inside "88421" — the exact
 *    splice the boundaries exist to prevent, reintroduced at a chunk seam.
 * 2. **A longer alternative is still arriving.** The pool-exhaustion ladder
 *    mints "Vantor" and then "Vantor 2", so a surrogate that is another's prefix
 *    is routine. Requiring `start <= buffer.length - maxLen` means every
 *    alternative starting there was short enough to be fully visible, so
 *    longest-first had its chance. Without it, `["met Vantor", " 2 today"]`
 *    commits to "Vantor" and hands the reader the FIRST entity's real value for
 *    the SECOND entity's mention — unrecoverable once emitted.
 *
 * Everything not settled stays in `pending` and is reconsidered verbatim next
 * chunk. `pending` is always RAW input, never replaced output: replacements are
 * appended straight to the emitted string and never re-scanned, which is what
 * `String.replace` does over a whole string, and is why the equivalence holds.
 * (It also retires an accepted limitation of the earlier design, which held back
 * a slice of the REPLACED buffer and could re-match a real value's suffix.)
 *
 * **Carry.** `(?<![0-9])` mirrors hazard 1 at the other edge: it also passes
 * vacuously at position 0, so once the digit that should veto a match has been
 * emitted, `pending` alone can no longer see it. The buffer therefore keeps one
 * already-emitted raw character at its head (`carry`), used only as regex left
 * context and never re-emitted — emission starts at `base`, not 0. One character
 * is enough because the lookbehind inspects exactly one.
 *
 * **Scan anchoring.** The scan starts at `base`, not 0, because a key CAN begin
 * at the carry — after "AB" is replaced, its raw "B" survives as the carry, and
 * a key "BC" happily matches across the replacement boundary. `String.replace`
 * resumes AFTER a replaced match (and a carry emitted as plain text was only
 * emitted because nothing settleable started there), so such a match must be
 * excluded — but excluded by ANCHORING, never by skip-and-continue: skipping a
 * yielded match advances the iterator past its whole span, silently swallowing
 * a shorter key inside it ("C" in the {AB, BC, C} case — a lost rehydration the
 * tests pin). Anchoring makes carry-region matches unrepresentable while keys
 * starting at `base` are still found, and the lookbehind still sees the carry:
 * lookarounds examine the subject string, not the scan start.
 *
 * Cost is O(buffer) per chunk, and `pending` never exceeds `maxLen` (emission
 * stops at `buffer.length - maxLen` unless a settled replacement carried the
 * cursor past it, which only shortens the remainder). A long non-matching stream
 * therefore costs one pass over each chunk plus a bounded window, with no
 * unbounded growth. Output chunks are UTF-16 code-unit slices: an emission
 * boundary can split a surrogate pair, so only the CONCATENATION of chunks is
 * guaranteed well-formed — a consumer doing strict-USV per-chunk work must
 * buffer accordingly (TextEncoderStream and JSON stringification are fine).
 */
export function createRehydrateTransform(map: Map<string, string>): TransformStream<string, string> {
  // Snapshot: the transform is a pure function of the map at construction.
  // The pattern and maxLen are frozen below anyway, so a live reference buys
  // nothing — while a caller deleting an entry mid-stream would turn a held-back
  // match's lazy get() into literal "undefined" in user-visible text.
  map = new Map(map);
  const pattern = surrogatePattern(map);
  // Nothing to match: hand chunks straight through, preserving chunk identity.
  if (!pattern) {
    return new TransformStream<string, string>({
      transform(chunk, controller) {
        controller.enqueue(chunk);
      },
    });
  }
  const maxLen = maxSurrogateLength(map);
  /** One already-emitted raw character; left context for the lookbehind only. */
  let carry = "";
  /** Raw, never-replaced input that is not yet safe to emit. */
  let pending = "";

  /**
   * One pass over carry + pending + chunk. Returns the text to emit and leaves
   * `carry`/`pending` positioned for the next call. At `atEnd` there is no next
   * character, so both hazards above are moot and every match is settled.
   */
  const consume = (chunk: string, atEnd: boolean): string => {
    const buffer = carry + pending + chunk;
    const base = carry.length; // 0 or 1: where not-yet-emitted text starts
    const visibleFrom = buffer.length - maxLen; // last start index fully in view

    let out = "";
    let cursor = base; // buffer consumed into `out` so far
    let deferFrom = atEnd ? buffer.length : visibleFrom;

    // Anchored at `base` (see "Scan anchoring" above): matchAll copies the
    // clone's lastIndex as its starting position and never mutates the shared
    // `pattern`, so carry-region matches are unrepresentable and no cross-call
    // lastIndex state exists. Never exclude by skip-and-continue here — the
    // iterator would advance past the skipped span and swallow shorter keys.
    const anchored = new RegExp(pattern.source, pattern.flags);
    anchored.lastIndex = base;
    for (const m of buffer.matchAll(anchored)) {
      const surrogate = m[0]!;
      const start = m.index;
      const end = start + surrogate.length;
      if (!(atEnd || (end < buffer.length && start <= visibleFrom))) {
        // Unsettled. Provably start >= visibleFrom, so this min never actually
        // moves deferFrom; it is written out so the holdback is locally evident
        // and survives a future change to the settled rule.
        deferFrom = Math.min(deferFrom, start);
        break; // every later match is unsettled too
      }
      out += buffer.slice(cursor, start) + map.get(surrogate)!;
      cursor = end;
    }

    const emitEnd = Math.max(cursor, deferFrom);
    out += buffer.slice(cursor, emitEnd);
    if (emitEnd > base) carry = buffer.charAt(emitEnd - 1);
    pending = buffer.slice(emitEnd);
    return out;
  };

  return new TransformStream<string, string>({
    transform(chunk, controller) {
      const out = consume(chunk, false);
      if (out) controller.enqueue(out);
    },
    flush(controller) {
      const out = consume("", true);
      if (out) controller.enqueue(out);
    },
  });
}
