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
 * The trailing `(?![0-9])` is a digit boundary on the whole alternation, and it
 * is load-bearing rather than defensive: the vault's pool-exhaustion ladder
 * mints digit-suffixed surrogates as a matter of routine ("Vantor", then
 * "Vantor 2"), so a conversation holding both meets ordinary text like
 * "Vantor 2024 report". Unboundaried, longest-first matches "Vantor 2" there
 * and yields the SECOND entity's real value spliced onto "024" — a rehydration
 * that misattributes the mention and mangles the year. With the lookahead
 * "Vantor 2" fails at the "2|024" seam, the alternation retries (longest-first
 * still holds among the candidates that pass), "Vantor" matches, and the space
 * that follows clears the boundary.
 *
 * Digits only, deliberately NOT `(?!\w)`: a letter boundary would refuse to
 * match "Vantor" inside "Vantors", and models pluralize surrogates freely, so
 * that mention would come back to the user still wearing its fake name — the
 * real value lost entirely. The digit rule errs toward base-name replacement
 * ("Globexs" for "Vantors", "Vantor" + " 22" for "Vantor 22"), which costs a
 * trailing character on the plural and resolves one genuinely ambiguous case
 * toward the shorter key. Both are cheaper than a dropped rehydration; both are
 * pinned in rehydrate.test.ts so the trade is a decision, not a drift.
 */
export function surrogatePattern(map: Map<string, string>): RegExp | undefined {
  if (map.size === 0) return undefined;
  const alts = [...map.keys()].sort((a, b) => b.length - a.length).map(escapeRegex);
  return new RegExp(`(?:${alts.join("|")})(?![0-9])`, "g");
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
