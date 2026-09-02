import type { LabelSpan } from "./labels.js";

/**
 * Controlled injection -- spec 6.2's load-bearing invariant, quoted verbatim
 * because everything here exists to make it literally true:
 *
 *   "injection happens only into certified-clear carriers => on positives, gold
 *    spans are exactly the injected ones; any finding outside them is a true
 *    false positive."
 *
 * The first clause is `certify.ts`'s job (and see its header for how far it
 * currently reaches). The second is this module's, and the way to get it wrong
 * is to search the finished text for the value you just wrote. A search is a
 * SECOND, independent claim about where the value is, and it disagrees with the
 * truth exactly when it matters -- when the value occurs twice, when it lands
 * inside a longer token, when a normalization step moved it. So nothing here
 * ever calls `indexOf` to locate a gold span. The offset is arithmetic over the
 * splice that wrote it, and the search that does happen (`occurrences` below)
 * is a CHECK on that arithmetic, not the source of it.
 *
 * ## Span versus glue
 *
 * An injection is `prefix + value + suffix` and the span covers `value` only.
 * The split is not cosmetic: the glue is what makes an injected identifier read
 * like a sentence ("the PAN on file is ", "."), and a span that swallowed it
 * would be a gold span no correct detector could ever match, which would show
 * up as a permanent recall floor that looks like a model defect.
 */

export interface Injection {
  /**
   * UTF-16 offset into the CARRIER text (before any injection) at which this
   * injection's `prefix` begins. Offsets are carrier-relative on the way in and
   * result-relative on the way out; `applyInjections` is the only thing that
   * knows the mapping between them.
   */
  readonly at: number;
  /** Glue written before `value`. NOT part of the span. */
  readonly prefix: string;
  /** The injected value. This, and only this, is the span. */
  readonly value: string;
  /** Glue written after `value`. NOT part of the span. */
  readonly suffix: string;
  /** The corpus label type for the value: an IR entityType id or a `neg:` id. */
  readonly type: string;
  /** Which generator family produced the value; a diagnosis key, not a label. */
  readonly family: string;
  /** Spec 6.2's controlled dimensions, recorded as written. */
  readonly dimensions: Readonly<Record<string, string>>;
}

export interface WrittenInjection extends Injection {
  /** Where the value ended up in the RESULT text. Derived, never searched for. */
  readonly span: LabelSpan;
}

export interface InjectionResult {
  readonly text: string;
  readonly injections: readonly WrittenInjection[];
}

function occurrences(haystack: string, needle: string): number {
  let n = 0;
  for (let i = haystack.indexOf(needle); i !== -1; i = haystack.indexOf(needle, i + 1)) n += 1;
  return n;
}

/**
 * The invariant, as three checks. Exported and called separately by the
 * generator over the finished corpus, so a corpus that was assembled by some
 * path other than `applyInjections` still cannot be emitted unchecked.
 *
 * 1. `text.slice(start, end) === value` for every injection. The direct reading
 *    of the brief, and the one a downstream reader repeats.
 * 2. Every distinct value occurs in `text` exactly as many times as it was
 *    injected. This is the check that earns the invariant its second clause. If
 *    a value the generator wrote once also appears somewhere the generator did
 *    not write it, then a detector firing there is CORRECT and would be scored
 *    a false positive -- the corpus would be manufacturing false positives and
 *    calling them the model's. Counting rather than forbidding repeats leaves
 *    room for spec 6.2's multi-turn positives, where a value is deliberately
 *    injected twice and carries two gold spans.
 * 3. Spans are pairwise non-overlapping. Two overlapping gold spans make
 *    "exactly the injected ones" ambiguous -- a finding covering both is one
 *    match or two depending on which gold span a scorer walks first.
 */
export function assertInjectionInvariant(text: string, injections: readonly WrittenInjection[]): void {
  for (const inj of injections) {
    const got = text.slice(inj.span.start, inj.span.end);
    if (got !== inj.value) {
      throw new Error(
        `injection invariant violated: text.slice(${inj.span.start}, ${inj.span.end}) is ` +
          `${JSON.stringify(got)} but the injected value was ${JSON.stringify(inj.value)}`,
      );
    }
    if (inj.span.text !== inj.value) {
      throw new Error(
        `injection invariant violated: recorded span text ${JSON.stringify(inj.span.text)} ` +
          `is not the injected value ${JSON.stringify(inj.value)}`,
      );
    }
  }
  const injectedCounts = new Map<string, number>();
  for (const inj of injections) injectedCounts.set(inj.value, (injectedCounts.get(inj.value) ?? 0) + 1);
  for (const [value, expected] of injectedCounts) {
    const actual = occurrences(text, value);
    if (actual !== expected) {
      throw new Error(
        `injection invariant violated: ${JSON.stringify(value)} was injected ${expected} time(s) ` +
          `but occurs ${actual} time(s) in the text, so a finding on it is not provably a match ` +
          `or provably a false positive`,
      );
    }
  }
  const sorted = [...injections].sort((a, b) => a.span.start - b.span.start);
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = sorted[i - 1]!;
    const cur = sorted[i]!;
    if (cur.span.start < prev.span.end) {
      throw new Error(
        `injection invariant violated: gold spans [${prev.span.start},${prev.span.end}) and ` +
          `[${cur.span.start},${cur.span.end}) overlap`,
      );
    }
  }
}

/**
 * Writes every injection into `carrier` and returns the result together with the
 * span each one occupies.
 *
 * Injections are applied in ascending `at`, ties broken by input order, and the
 * running offset delta is what turns a carrier offset into a result offset.
 * Applying them in input order instead would be the classic version of this bug:
 * every earlier splice shifts every later one, and the spans would be silently
 * wrong by the accumulated length of everything written before them.
 */
export function applyInjections(carrier: string, injections: readonly Injection[]): InjectionResult {
  for (const inj of injections) {
    if (!Number.isInteger(inj.at) || inj.at < 0 || inj.at > carrier.length) {
      throw new Error(`injection point ${inj.at} is outside the carrier [0, ${carrier.length}]`);
    }
    if (inj.value.length === 0) throw new Error("an injection value must be non-empty");
    // Checked against the CARRIER, before anything is written, so the message
    // names the real problem (this carrier already contains this value) rather
    // than the symptom the count check would report later.
    if (carrier.includes(inj.value)) {
      throw new Error(
        `carrier already contains the value ${JSON.stringify(inj.value)}; injecting it would ` +
          `create an occurrence with no gold span`,
      );
    }
  }

  const order = injections
    .map((inj, index) => ({ inj, index }))
    .sort((a, b) => (a.inj.at === b.inj.at ? a.index - b.index : a.inj.at - b.inj.at));

  let text = "";
  let cursor = 0;
  const written: (WrittenInjection & { index: number })[] = [];
  for (const { inj, index } of order) {
    text += carrier.slice(cursor, inj.at);
    text += inj.prefix;
    const start = text.length;
    text += inj.value;
    const end = text.length;
    text += inj.suffix;
    cursor = inj.at;
    written.push({ ...inj, span: { start, end, text: inj.value }, index });
  }
  text += carrier.slice(cursor);

  // Returned in INPUT order, not application order: the caller built the
  // injection list and its labels alongside it, and silently reordering the
  // result would misalign a parallel array without any error.
  const result = written.sort((a, b) => a.index - b.index).map(({ index: _i, ...rest }) => rest);
  assertInjectionInvariant(text, result);
  return { text, injections: result };
}
