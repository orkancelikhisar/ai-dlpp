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
