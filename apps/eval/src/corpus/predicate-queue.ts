import { createHash, createHmac } from "node:crypto";

/**
 * The blind predicate queue: one row per corpus item, a message and an opaque
 * id, and nothing else.
 *
 * ## Why this file exists rather than a second call to `questions.ts`
 *
 * `questions.ts`'s `queueRows` already builds "a blind queue". The round it
 * fed -- `v2-blind-double-labelling-p-fin` -- came back with 20 of 189
 * predicate answers, and its own blindness audit (see `labelling.ts`,
 * `blindness.channels`) named three ways answer-bearing content still reached
 * the annotators. Two of the three are properties of the ROW, and this module
 * is the row rebuilt around them:
 *
 * - The row carried `itemId`. The corpus ids are `inj-<carrier>-<n>` and
 *   `neg-<carrier>`, and `<carrier>` is `hn02`..`hn12` for the 77 items built
 *   on the hard-negative carriers and `o01`..`o16` for the other 112 --
 *   MEASURED over the committed `injection-p-fin-v2.labelled.jsonl`, whose
 *   `meta.carrierStratum` is `"hard-negative"` on exactly the 77 `hn` ids. So
 *   the id named the stratum, the polarity (`neg-` is the density-0 pristine
 *   item) and the position in the carrier's wave. (The commissioning note for
 *   this round also said the ids encode family names. They do not, in v2: all
 *   189 match `^inj-(o|hn)\d\d-\d$` or `^neg-(o|hn)\d\d$` and no family id
 *   appears in any of them. Checked, in `corpus-predicate-queue.test.ts`.)
 * - The row carried `question`, and for the 20 contested-span rows that
 *   question named the span's proposed type and set out both readings of the
 *   policy before the annotator had read it.
 *
 * The third -- TOLD-1, a brief that paraphrased the predicate in the compiled
 * IR's own words -- is a property of the brief, not of the row, and cannot be
 * fixed here. What CAN be fixed here is that the artifact carries no predicate
 * wording of any kind, so nothing in the handover re-supplies it; the test
 * asserts the queue bytes do not contain `BRIEF_PARAPHRASE_FRAGMENT`.
 *
 * ## What a row is
 *
 * `rowId` and `text`. The question is asked once, in the brief, not 189 times
 * in the file: a per-row question field is a place for framing to hide, and the
 * previous round is the evidence that it does.
 */

/** A queue row. Exactly two fields, and the test asserts exactly two. */
export interface PredicateQueueRow {
  readonly rowId: string;
  readonly text: string;
}

/**
 * The subset of a corpus item this module reads.
 *
 * Structural rather than `CorpusItem` so that nothing here imports the zod
 * loader: this module is imported by a test that runs under vitest AND by an
 * emitter run under `vite-node`, and keeping it dependency-free means the
 * builder cannot pick up a transitive dependency on the corpus schema and
 * quietly start reading a field the queue must not carry.
 */
export interface QueueSourceItem {
  readonly id: string;
  readonly text: string;
  readonly meta?: Readonly<Record<string, unknown>> | undefined;
}

export const ROW_ID_HEX_LENGTH = 16;

export const ROW_ID_ALGORITHM =
  `HMAC-SHA256(key = salt, message = itemId), lowercase hex, first ${ROW_ID_HEX_LENGTH} characters`;

/**
 * The opaque row id.
 *
 * What the salt buys and what it does not: the id space is 189 strings this
 * module can enumerate, so anyone holding the salt recovers the item id by
 * hashing all 189 and matching. The salt is therefore not a one-way function in
 * any interesting sense -- it is the single secret that separates the queue
 * from the strata, which is why it is written only to the mapping file and why
 * the mapping file is not a path an annotator is given.
 *
 * 16 hex characters is 64 bits. Uniqueness over 189 ids is asserted, not
 * assumed, by `buildPredicateQueue` on every build.
 */
export function rowIdFor(itemId: string, salt: string): string {
  if (salt.length < 32) {
    throw new Error(`salt must be at least 32 characters; got ${salt.length}`);
  }
  return createHmac("sha256", salt).update(itemId, "utf8").digest("hex").slice(0, ROW_ID_HEX_LENGTH);
}

/**
 * The stratum dimensions, measured on every build.
 *
 * These are the ways the SOURCE order groups items. Only `carrier` is
 * constrained by the shuffle; the rest are reported, including the two the
 * shuffle makes worse.
 */
export const STRATUM_DIMENSIONS = [
  "carrier",
  "carrierStratum",
  "wave",
  "density",
  "register",
  "constructedPrediction",
] as const;

export type StratumDimension = (typeof STRATUM_DIMENSIONS)[number];

/**
 * The dimension the shuffle is required to drive to zero adjacency.
 *
 * `carrier`, because the carrier is the only stratum an annotator can see from
 * the queue alone: the seven items built on one carrier share its sentences
 * verbatim, six of them with spliced clauses and the seventh -- the `neg-` item
 * -- with none. In source order those seven are consecutive, so the run reads
 * as six variations followed by the original, and the seventh is identifiable
 * as the pristine one without reading it. Zeroing carrier adjacency is
 * feasible here and asserted: the largest carrier holds 7 of 189 items, well
 * under the ceil(189/2) = 95 that would make it impossible.
 *
 * `carrierStratum` is NOT constrained because it cannot be: 112 ordinary
 * against 77 hard-negative means at least 112 - 95 = 17 same-stratum
 * adjacencies exist in every possible order.
 */
export const SHUFFLE_CONSTRAINT: StratumDimension = "carrier";

function metaString(item: QueueSourceItem, key: string): string {
  const value = item.meta?.[key];
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  throw new Error(`item ${item.id} has no string or number meta.${key}`);
}

/**
 * The stratum values of one item, for MEASUREMENT and for the shuffle
 * constraint. None of this reaches a row.
 *
 * `wave` is read off the item id because that is where the generator put it;
 * every other dimension is read from `meta`.
 */
export function strataOf(item: QueueSourceItem): Readonly<Record<StratumDimension, string>> {
  const construction = item.meta?.["predicateConstruction"];
  if (typeof construction !== "object" || construction === null) {
    throw new Error(`item ${item.id} has no meta.predicateConstruction`);
  }
  const constructed = (construction as Record<string, unknown>)["constructed"];
  if (typeof constructed !== "boolean") {
    throw new Error(`item ${item.id} has no boolean meta.predicateConstruction.constructed`);
  }
  const stratum = item.meta?.["carrierStratum"];
  if (stratum !== undefined && typeof stratum !== "string") {
    throw new Error(`item ${item.id} has a non-string meta.carrierStratum`);
  }
  const negative = item.id.startsWith("neg-");
  const suffix = item.id.slice(item.id.lastIndexOf("-") + 1);
  return {
    carrier: metaString(item, "carrierId"),
    // Absent means an ordinary carrier -- see `CarrierStratum` in carriers.ts,
    // where the field is optional precisely so adding it changed no bytes.
    carrierStratum: stratum ?? "ordinary",
    wave: negative ? "pristine" : suffix,
    density: metaString(item, "density"),
    register: metaString(item, "carrierRegister"),
    constructedPrediction: String(constructed),
  };
}

export interface AdjacencyMeasurement {
  readonly dimension: StratumDimension;
  readonly adjacentPairs: number;
  readonly sameStratumPairs: number;
  readonly rate: number;
  /**
   * The same-neighbour rate expected of a UNIFORMLY RANDOM permutation of these
   * items: sum over stratum values of n(n-1), over N(N-1). It is the floor a
   * shuffle can be judged against -- "adjacency fell" is only interesting
   * against a number that says how far it could fall.
   */
  readonly chanceRate: number;
}

export function adjacencyOf(ordered: readonly QueueSourceItem[]): AdjacencyMeasurement[] {
  const strata = ordered.map((item) => strataOf(item));
  const n = strata.length;
  return STRATUM_DIMENSIONS.map((dimension) => {
    let same = 0;
    for (let i = 0; i + 1 < n; i += 1) {
      if (strata[i]![dimension] === strata[i + 1]![dimension]) same += 1;
    }
    const counts = new Map<string, number>();
    for (const s of strata) counts.set(s[dimension], (counts.get(s[dimension]) ?? 0) + 1);
    let pairs = 0;
    for (const c of counts.values()) pairs += c * (c - 1);
    return {
      dimension,
      adjacentPairs: n - 1,
      sameStratumPairs: same,
      rate: n > 1 ? same / (n - 1) : 0,
      chanceRate: n > 1 ? pairs / (n * (n - 1)) : 0,
    };
  });
}

/**
 * A deterministic uniform stream in [0, 1), seeded by a string.
 *
 * SHA-256 over `seed:block` rather than a 32-bit LCG so the seed is a string
 * (the mapping file records it as one) and so the whole state is the seed --
 * regenerating the queue needs the seed and this file and nothing else.
 */
function makeRng(seed: string): () => number {
  let block = 0;
  let digest = createHash("sha256").update(`${seed}:${block}`, "utf8").digest();
  let offset = 0;
  return () => {
    if (offset + 4 > digest.length) {
      block += 1;
      digest = createHash("sha256").update(`${seed}:${block}`, "utf8").digest();
      offset = 0;
    }
    const value = digest.readUInt32BE(offset);
    offset += 4;
    return value / 0x1_0000_0000;
  };
}

function violationCount(order: readonly number[], key: readonly string[]): number {
  let v = 0;
  for (let i = 0; i + 1 < order.length; i += 1) {
    if (key[order[i]!] === key[order[i + 1]!]) v += 1;
  }
  return v;
}

/**
 * A seeded permutation of the item indices in which no two adjacent items share
 * a carrier.
 *
 * Fisher-Yates first, then a repair loop that only ever accepts a swap which
 * STRICTLY reduces the number of same-carrier adjacencies. Strict decrease is
 * what makes the loop terminate: the count is a non-negative integer that falls
 * on every iteration, so the guard below can only fire if a corpus arrives on
 * which no single swap helps -- which is a real finding about the corpus and
 * should stop the build rather than silently emit a grouped queue.
 *
 * The repair is seeded too (it picks among the swaps that help), so the whole
 * order is a function of the seed and the source order alone.
 */
export function shuffledPositions(items: readonly QueueSourceItem[], seed: string): number[] {
  const key = items.map((item) => strataOf(item)[SHUFFLE_CONSTRAINT]);
  const rng = makeRng(seed);
  const order = items.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = order[i]!;
    order[i] = order[j]!;
    order[j] = tmp;
  }
  const guard = order.length * order.length;
  for (let step = 0; ; step += 1) {
    let current = violationCount(order, key);
    if (current === 0) break;
    if (step >= guard) {
      throw new Error(`could not remove same-${SHUFFLE_CONSTRAINT} adjacency in ${guard} swaps`);
    }
    let at = -1;
    for (let i = 0; i + 1 < order.length; i += 1) {
      if (key[order[i]!] === key[order[i + 1]!]) {
        at = i + 1;
        break;
      }
    }
    const candidates: number[] = [];
    for (let j = 0; j < order.length; j += 1) {
      if (j === at || j === at - 1) continue;
      const a = order[at]!;
      order[at] = order[j]!;
      order[j] = a;
      if (violationCount(order, key) < current) candidates.push(j);
      const b = order[at]!;
      order[at] = order[j]!;
      order[j] = b;
    }
    if (candidates.length === 0) {
      throw new Error(
        `no single swap reduces same-${SHUFFLE_CONSTRAINT} adjacency below ${current}; ` +
          "the constraint may be infeasible for this corpus",
      );
    }
    const pick = candidates[Math.floor(rng() * candidates.length)]!;
    const a = order[at]!;
    order[at] = order[pick]!;
    order[pick] = a;
    current = violationCount(order, key);
  }
  return order;
}

export interface RowIdMapEntry {
  readonly position: number;
  readonly rowId: string;
  readonly itemId: string;
}

export interface RowIdMap {
  readonly salt: string;
  readonly seed: string;
  readonly rowIdAlgorithm: string;
  readonly shuffleConstraint: StratumDimension;
  readonly itemCount: number;
  readonly rows: readonly RowIdMapEntry[];
  readonly adjacency: {
    readonly sourceOrder: readonly AdjacencyMeasurement[];
    readonly queueOrder: readonly AdjacencyMeasurement[];
    readonly rowIdSortedOrder: readonly AdjacencyMeasurement[];
  };
}

export interface BuiltPredicateQueue {
  readonly rows: readonly PredicateQueueRow[];
  readonly order: readonly number[];
  readonly map: RowIdMap;
}

export interface PredicateQueueOptions {
  readonly salt: string;
  readonly seed: string;
}

export function buildPredicateQueue(
  items: readonly QueueSourceItem[],
  options: PredicateQueueOptions,
): BuiltPredicateQueue {
  const order = shuffledPositions(items, options.seed);
  const rows: PredicateQueueRow[] = [];
  const entries: RowIdMapEntry[] = [];
  const seen = new Map<string, string>();
  for (const [position, index] of order.entries()) {
    const item = items[index]!;
    const rowId = rowIdFor(item.id, options.salt);
    const clash = seen.get(rowId);
    if (clash !== undefined) {
      throw new Error(`rowId ${rowId} collides: ${clash} and ${item.id}`);
    }
    seen.set(rowId, item.id);
    rows.push({ rowId, text: item.text });
    entries.push({ position, rowId, itemId: item.id });
  }
  const ordered = order.map((i) => items[i]!);
  const byRowId = [...items]
    .map((item) => ({ item, rowId: rowIdFor(item.id, options.salt) }))
    .sort((a, b) => (a.rowId < b.rowId ? -1 : a.rowId > b.rowId ? 1 : 0))
    .map((e) => e.item);
  return {
    rows,
    order,
    map: {
      salt: options.salt,
      seed: options.seed,
      rowIdAlgorithm: ROW_ID_ALGORITHM,
      shuffleConstraint: SHUFFLE_CONSTRAINT,
      itemCount: items.length,
      rows: entries,
      adjacency: {
        sourceOrder: adjacencyOf(items),
        queueOrder: adjacencyOf(ordered),
        rowIdSortedOrder: adjacencyOf(byRowId),
      },
    },
  };
}

/** One JSON object per line, `rowId` first, trailing newline. */
export function serializePredicateQueue(rows: readonly PredicateQueueRow[]): string {
  return rows.map((r) => JSON.stringify({ rowId: r.rowId, text: r.text })).join("\n") + "\n";
}

/**
 * Every string in the corpus that names a stratum, a family, a type or an item.
 *
 * Built from the corpus, not from the queue, so a scan of the queue against it
 * is a comparison between two independent things. `corpus-predicate-queue.test.ts`
 * builds its own copy of this list rather than importing it, for the same
 * reason.
 */
export function forbiddenTokensOf(items: readonly QueueSourceItem[]): string[] {
  const tokens = new Set<string>();
  for (const item of items) {
    tokens.add(item.id);
    const meta = item.meta ?? {};
    const carrier = meta["carrierId"];
    if (typeof carrier === "string") tokens.add(carrier);
    const stratum = meta["carrierStratum"];
    if (typeof stratum === "string") tokens.add(stratum);
    for (const raw of (meta["injections"] ?? []) as readonly Record<string, unknown>[]) {
      if (typeof raw["family"] === "string") tokens.add(raw["family"]);
      if (typeof raw["type"] === "string") tokens.add(raw["type"]);
      const dims = (raw["dimensions"] ?? {}) as Record<string, unknown>;
      const role = dims["constructedRole"];
      if (typeof role === "string" && role !== "none") tokens.add(role);
    }
    for (const span of (item as { gold?: readonly Record<string, unknown>[] }).gold ?? []) {
      if (typeof span["entityType"] === "string") tokens.add(span["entityType"]);
    }
  }
  return [...tokens].sort();
}
