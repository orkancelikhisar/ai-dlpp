import type { PolicyIr } from "@sih/core";
import type { CorpusItem } from "../driver/corpus.js";
import { NEG_PREFIX } from "./labels.js";

/**
 * The four leakage measurements, and the reading of the IR that one of them
 * rests on.
 *
 * Every number this module produces is written into the emitted manifest, so
 * the claim "this corpus does not hand the compiled arm the answer" is a
 * regenerable measurement rather than a sentence. The functions are shared by
 * the manifest and by `corpus-v2.test.ts` deliberately: a test that computed
 * its own version of a rate would be checking a second implementation, not the
 * artifact.
 *
 * The `before` figures quoted throughout are measured on
 * `corpora/generated/injection-p-fin-adjudicated-v1.jsonl`, the corpus this
 * round exists to replace, using these same functions.
 */

// -- contextBoost -----------------------------------------------------------

/**
 * How far from a span a contextBoost term counts as "near it".
 *
 * 70 characters either side, and the span's own text is EXCLUDED from the
 * window. The exclusion matters: `CIF 30045512` contains "cif", `KYC 4471200`
 * contains "kyc", and counting those would score a family as boosted because of
 * a substring of the value the family is asking about, which is not a fact
 * about the surrounding text at all.
 *
 * The number is not tuned. `runTier0` does not implement contextBoost, so there
 * is no window in the shipping code to copy; 70 characters is about a clause
 * either side, which is the distance over which a prompt-reading arm can
 * plausibly associate a keyword with a value. It is stated here so a reader can
 * recompute at another width.
 */
export const BOOST_WINDOW = 70;

export interface BoostTerms {
  /** Every contextBoost term in the IR, lowercased, deduplicated, sorted. */
  readonly all: readonly string[];
  /** entityType id -> the terms its own rules declare. */
  readonly byEntityType: Readonly<Record<string, readonly string[]>>;
}

export function boostTerms(ir: PolicyIr): BoostTerms {
  const all = new Set<string>();
  const by: Record<string, Set<string>> = {};
  for (const rule of ir.rules) {
    for (const term of rule.contextBoost ?? []) {
      const t = term.toLowerCase();
      all.add(t);
      (by[rule.entityType] ??= new Set()).add(t);
    }
  }
  const byEntityType: Record<string, readonly string[]> = {};
  for (const k of Object.keys(by).sort()) byEntityType[k] = [...by[k]!].sort();
  return { all: [...all].sort(), byEntityType };
}

/**
 * Word-boundary matching, not substring. "account" must not fire on
 * "storage_account_key" and "connect" must not fire on "connection"; both would
 * turn an ordinary word into a boost hit and make the rate meaningless. The
 * pattern is built per call rather than cached because a shared `/g` regex is
 * stateful across calls, which this repository has been bitten by before (see
 * `orthographicOrgSweep` in `certify.ts`).
 */
function containsTerm(haystack: string, term: string): boolean {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, "i").test(haystack);
}

export function boostNear(
  text: string,
  span: { readonly start: number; readonly end: number },
  terms: readonly string[],
  window = BOOST_WINDOW,
): boolean {
  const before = text.slice(Math.max(0, span.start - window), span.start);
  const after = text.slice(span.end, Math.min(text.length, span.end + window));
  return terms.some((t) => containsTerm(before, t) || containsTerm(after, t));
}

export interface BoostReport {
  readonly window: number;
  readonly terms: number;
  /** Any contextBoost term of any rule, near a gold span. */
  readonly goldSpans: number;
  readonly goldSpansWithBoost: number;
  readonly goldRate: number;
  /** The same, near a confusable span. */
  readonly confusableSpans: number;
  readonly confusableSpansWithBoost: number;
  readonly confusableRate: number;
  readonly delta: number;
  /**
   * The sharper form: a term declared by a rule of the span's OWN entityType,
   * near a gold span; and near a confusable, a term of the entityType that
   * confusable is a near miss for.
   */
  readonly goldOwnTypeRate: number;
  readonly confusablePairedTypeRate: number;
  readonly ownTypeDelta: number;
}

/**
 * `pairedType` maps a `neg:` label to the entityType it is a near miss for, so
 * the own-type rate has a counterpart on the negative side. Supplied by the
 * caller rather than derived: only the family catalogue knows which pair a
 * confusable belongs to, and guessing it from the id would be a second,
 * fallible claim.
 */
export function measureBoost(
  items: readonly CorpusItem[],
  ir: PolicyIr,
  pairedType: Readonly<Record<string, string>>,
): BoostReport {
  const terms = boostTerms(ir);
  let gold = 0;
  let goldBoost = 0;
  let goldOwn = 0;
  let neg = 0;
  let negBoost = 0;
  let negPaired = 0;
  for (const item of items) {
    for (const label of labelSpans(item)) {
      const near = boostNear(item.text, label, terms.all);
      if (label.type.startsWith(NEG_PREFIX)) {
        neg += 1;
        if (near) negBoost += 1;
        const paired = pairedType[label.type];
        if (paired !== undefined && boostNear(item.text, label, terms.byEntityType[paired] ?? [])) negPaired += 1;
      } else {
        gold += 1;
        if (near) goldBoost += 1;
        if (boostNear(item.text, label, terms.byEntityType[label.type] ?? [])) goldOwn += 1;
      }
    }
  }
  const goldRate = gold === 0 ? 0 : goldBoost / gold;
  const confusableRate = neg === 0 ? 0 : negBoost / neg;
  const goldOwnTypeRate = gold === 0 ? 0 : goldOwn / gold;
  const confusablePairedTypeRate = neg === 0 ? 0 : negPaired / neg;
  return {
    window: BOOST_WINDOW,
    terms: terms.all.length,
    goldSpans: gold,
    goldSpansWithBoost: goldBoost,
    goldRate,
    confusableSpans: neg,
    confusableSpansWithBoost: negBoost,
    confusableRate,
    delta: goldRate - confusableRate,
    goldOwnTypeRate,
    confusablePairedTypeRate,
    ownTypeDelta: goldOwnTypeRate - confusablePairedTypeRate,
  };
}

/** Every labelled span on an item -- gold and `neg:` alike -- read out of `meta.injections`. */
export function labelSpans(
  item: CorpusItem,
): readonly { readonly start: number; readonly end: number; readonly text: string; readonly type: string; readonly family: string }[] {
  const raw = (item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[];
  return raw.map((r) => ({
    start: r["start"] as number,
    end: r["end"] as number,
    text: r["text"] as string,
    type: String(r["type"]),
    family: String(r["family"]),
  }));
}

// -- orthography ------------------------------------------------------------

/**
 * The "flag the odd string" detector, written down so the corpus can be scored
 * against it.
 *
 * This is not a proposed system. It is the null hypothesis: a reader that
 * understands nothing, returns every region of a message that LOOKS unlike
 * running prose, and would be indistinguishable from a real detector on a
 * corpus whose injected spans are its only odd-looking tokens. MEASURED on
 * `injection-p-fin-adjudicated-v1`: it returns 95 of that corpus's 108 gold
 * spans, and on 58 of them it returns nothing else in the message -- so on 54%
 * of the gold spans "return the odd string" is a perfect detector, and its
 * precision over the whole corpus is 0.49.
 *
 * The rules are deliberately crude and are listed rather than tuned: a
 * PEM-style dashed block; a token mixing letters and digits; a run of nine or
 * more digits; something with an at-sign between alphanumerics; anything
 * containing "://"; an all-uppercase run of four or more; sixteen or more
 * base64 characters; and a Title-Case bigram, which is what an organisation
 * name looks like in a lowercase chat message.
 */
const ORACLE_TOKEN = /[A-Za-z0-9@._/:+=~-]{3,}/g;
const ORACLE_TITLE_BIGRAM = /\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b/g;
const ORACLE_PEM = /-----BEGIN[^\n]*-----[\s\S]*?-----END[^\n]*-----/g;

function tokenIsOdd(t: string): boolean {
  if (/[0-9]/.test(t) && /[A-Za-z]/.test(t)) return true;
  if (/^[0-9]{9,}$/.test(t)) return true;
  if (/[A-Za-z0-9]@[A-Za-z0-9]/.test(t)) return true;
  if (t.includes("://")) return true;
  if (/^[A-Z]{4,}$/.test(t)) return true;
  if (t.length >= 16 && /^[A-Za-z0-9+/=]+$/.test(t)) return true;
  return false;
}

export interface OracleSpan {
  readonly start: number;
  readonly end: number;
}

export function orthographicOracle(text: string): readonly OracleSpan[] {
  const spans: OracleSpan[] = [];
  for (const m of text.matchAll(ORACLE_PEM)) spans.push({ start: m.index, end: m.index + m[0].length });
  for (const m of text.matchAll(ORACLE_TOKEN)) {
    const start = m.index;
    const end = start + m[0].length;
    if (spans.some((s) => s.start <= start && end <= s.end)) continue;
    if (tokenIsOdd(m[0])) spans.push({ start, end });
  }
  for (const m of text.matchAll(ORACLE_TITLE_BIGRAM)) spans.push({ start: m.index, end: m.index + m[0].length });
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  // Drop anything strictly contained in another hit, so a PEM block counts once
  // rather than once per line.
  return sorted.filter(
    (s) => !sorted.some((o) => o !== s && o.start <= s.start && s.end <= o.end && !(o.start === s.start && o.end === s.end)),
  );
}

function overlaps(a: { start: number; end: number }, b: { start: number; end: number }): boolean {
  return a.start < b.end && b.start < a.end;
}

export interface OrthographyReport {
  readonly goldSpans: number;
  /** Gold spans the oracle returns at all. Its recall numerator. */
  readonly goldSpansFound: number;
  /**
   * The headline: gold spans the oracle FINDS and that are the only region it
   * returns anywhere in that message -- i.e. spans where "return the odd string"
   * is a perfect detector.
   *
   * "Finds AND is alone" rather than "is alone", and the difference is not
   * pedantry: an eight-digit account number is not odd enough for this oracle to
   * return, so a message carrying one and nothing else has no oracle spans at
   * all, and counting that as "the gold span is the unique outlier" would credit
   * the corpus for a span the oracle cannot see. MEASURED during this rebuild:
   * three of four apparent survivors were exactly that.
   */
  readonly goldSpansSolvedByOracle: number;
  readonly solvedRate: number;
  readonly oracleSpans: number;
  readonly oracleSpansOnGold: number;
  readonly oraclePrecision: number;
  readonly oracleRecall: number;
}

export function measureOrthography(items: readonly CorpusItem[]): OrthographyReport {
  let gold = 0;
  let found = 0;
  let solved = 0;
  let oracleSpans = 0;
  let oracleOnGold = 0;
  for (const item of items) {
    const hits = orthographicOracle(item.text);
    oracleSpans += hits.length;
    for (const h of hits) if (item.gold.some((g) => overlaps(h, g))) oracleOnGold += 1;
    for (const g of item.gold) {
      gold += 1;
      const hit = hits.some((h) => overlaps(h, g));
      if (!hit) continue;
      found += 1;
      if (hits.every((h) => overlaps(h, g))) solved += 1;
    }
  }
  return {
    goldSpans: gold,
    goldSpansFound: found,
    goldSpansSolvedByOracle: solved,
    solvedRate: gold === 0 ? 0 : solved / gold,
    oracleSpans,
    oracleSpansOnGold: oracleOnGold,
    oraclePrecision: oracleSpans === 0 ? 0 : oracleOnGold / oracleSpans,
    oracleRecall: gold === 0 ? 0 : found / gold,
  };
}

// -- role readability -------------------------------------------------------

export interface RoleReport {
  readonly orgSpans: number;
  /** Spans whose organisation name appears in only ONE role class corpus-wide. */
  readonly roleLockedSpans: number;
  readonly roleLockedRate: number;
  readonly namesInBothClasses: readonly string[];
  readonly namesInOneClass: readonly { readonly name: string; readonly roleClass: string }[];
}

/**
 * Whether an arm could answer "client or not" from the NAME.
 *
 * A span is role-locked when every occurrence of its name in the whole corpus
 * sits on the same side of the client/non-client line. MEASURED on
 * `injection-p-fin-adjudicated-v1`: 19 of the 39 spans this function counts
 * there, because its client pool, its supplier pool, its competitor pool and
 * its employer name are four disjoint arrays. That corpus also records
 * `constructedRole: "none"` on three families whose clauses do assign a role,
 * which keeps 13 further role-locked spans out of the denominator; counted by
 * hand the figure is 32 of 52.
 *
 * `roleClassOf` is supplied by the caller for the same reason `pairedType` is:
 * the mapping from a written clause to a role class lives in the family
 * catalogue, and re-deriving it here from an id would be a second claim.
 */
export function measureRoles(
  items: readonly CorpusItem[],
  roleClassOf: (dimensions: Readonly<Record<string, string>>, type: string) => string | undefined,
): RoleReport {
  const spans: { name: string; roleClass: string }[] = [];
  for (const item of items) {
    for (const raw of (item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[]) {
      const dimensions = (raw["dimensions"] ?? {}) as Record<string, string>;
      const rc = roleClassOf(dimensions, String(raw["type"]));
      if (rc === undefined) continue;
      spans.push({ name: String(raw["text"]), roleClass: rc });
    }
  }
  const byName = new Map<string, Set<string>>();
  for (const s of spans) {
    const set = byName.get(s.name) ?? new Set<string>();
    set.add(s.roleClass);
    byName.set(s.name, set);
  }
  const locked = spans.filter((s) => byName.get(s.name)!.size === 1);
  return {
    orgSpans: spans.length,
    roleLockedSpans: locked.length,
    roleLockedRate: spans.length === 0 ? 0 : locked.length / spans.length,
    namesInBothClasses: [...byName.entries()].filter(([, v]) => v.size > 1).map(([k]) => k).sort(),
    namesInOneClass: [...byName.entries()]
      .filter(([, v]) => v.size === 1)
      .map(([name, v]) => ({ name, roleClass: [...v][0]! }))
      .sort((a, b) => a.name.localeCompare(b.name)),
  };
}

// -- the IR's own counterExample list ---------------------------------------

/**
 * What each of the IR's 23 `counterExamples` entries DENOTES, written down.
 *
 * This is a reading of `policies/compiled/p-fin.ir.json`, not a fact derived
 * from it, and it is here rather than inferred because the inference cannot be
 * made mechanically. "MICR 400240012" denotes the surface "MICR line"; a token
 * test cannot see that, and a corpus family called `micr-code` that mints
 * `MICR <9 digits>` is the same surface with different digits. That identity is
 * the finding this round exists to close -- 13 of 24 confusable families in
 * `injection-p-fin-adjudicated-v1` are 1:1 with an entry below, carrying 59 of
 * its 108 confusable spans -- and closing it requires naming the surfaces.
 *
 * `corpus-v2.test.ts` asserts this table enumerates exactly the counterExamples
 * the IR actually holds, so a recompiled policy that adds one fails here rather
 * than silently escaping the check.
 */
export const IR_COUNTEREXAMPLE_SURFACES: readonly { readonly counterExample: string; readonly surfaceName: string }[] = [
  { counterExample: "ABCDE1234F", surfaceName: "PAN-shaped string with an invalid holder-type character" },
  { counterExample: "AAAPZ1234", surfaceName: "truncated PAN" },
  { counterExample: "27AABCU9603R1ZM", surfaceName: "GSTIN" },
  { counterExample: "PAN application form 49A", surfaceName: "name of a tax form" },
  { counterExample: "1234 5678 9012", surfaceName: "twelve-digit number with a broken Verhoeff digit" },
  { counterExample: "100234567890", surfaceName: "twelve-digit number with a forbidden leading digit" },
  { counterExample: "9876543210", surfaceName: "ten-digit number" },
  { counterExample: "priya.sharma@meridiancap.example.com", surfaceName: "email address" },
  { counterExample: "MICR 400240012", surfaceName: "MICR line" },
  { counterExample: "SWIFT HDFCINBB", surfaceName: "SWIFT/BIC code" },
  { counterExample: "ticket INC0042318", surfaceName: "incident ticket id" },
  { counterExample: "branch code 0247", surfaceName: "bare branch code" },
  { counterExample: "employee id 88213", surfaceName: "employee id" },
  { counterExample: "the client", surfaceName: "unnamed reference to a client" },
  { counterExample: "our counterparty", surfaceName: "unnamed reference to a counterparty" },
  { counterExample: "Mumbai", surfaceName: "place name" },
  { counterExample: "sk-...", surfaceName: "redaction placeholder in credential shape" },
  { counterExample: "AKIA followed by sixteen uppercase characters", surfaceName: "prose description of a key format" },
  { counterExample: "the token has been rotated", surfaceName: "prose statement that a credential was rotated" },
  { counterExample: "https://intranet.example.com/runbook", surfaceName: "internal https URL" },
  { counterExample: "the production database is down", surfaceName: "prose statement about a database" },
  { counterExample: "-----BEGIN CERTIFICATE REQUEST-----", surfaceName: "certificate signing request block" },
  { counterExample: "the public key fingerprint", surfaceName: "public key fingerprint" },
];

export function irCounterExamples(ir: PolicyIr): readonly string[] {
  return ir.entityTypes.flatMap((e) => e.counterExamples ?? []);
}
