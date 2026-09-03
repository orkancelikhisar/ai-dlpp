import { seededRng, type PolicyIr } from "@sih/core";
import { CorpusItemSchema, type CorpusItem } from "../driver/corpus.js";
import { ALL_CARRIERS, carrierSlots, carrierText, slotPosition, type Carrier } from "./carriers.js";
import {
  certificationSummary,
  certifyCarrier,
  type CarrierCertification,
  type CertificationSummary,
  type CertifyOptions,
} from "./certify.js";
import { checkContamination, type ContaminationReport, type SelfTestExample } from "./contamination.js";
import { CONFUSABLE_FAMILIES, POSITIVE_FAMILIES, type Family } from "./families.js";
import { applyInjections, assertInjectionInvariant, type Injection, type WrittenInjection } from "./inject.js";
import {
  buildViolatesUnder,
  isIrBacked,
  POLICY_IDS,
  toPFinGold,
  unpopulatedLabel,
  type LabelledSpan,
} from "./labels.js";

/**
 * The generator. Seeded, versioned, and reproducible byte for byte.
 *
 * ## Determinism contract
 *
 * The same `seed` and `GENERATOR_VERSION` must produce a byte-identical corpus,
 * because a corpus nobody can regenerate is not an artifact -- it is a file. So:
 *
 * - No clock, no `Math.random`, no environment, no filesystem order. The only
 *   entropy source is `seededRng` from `@sih/core`, which is the same PRNG the
 *   pseudonymization vault seeds surrogates with.
 * - Every item MINTS its values from its OWN stream, keyed
 *   `<seed>|<version>|<carrierId>|<recipeIndex>`. A single shared stream would
 *   work and would be worse: adding one carrier would shift every later item's
 *   values, so a corpus diff after any edit would be total and unreviewable.
 * - WHICH family an item uses is dealt from a single global deck instead, and
 *   that IS order-dependent. The trade is deliberate: an independent per-carrier
 *   draw gave a measured 19-to-3 spread between the most and least represented
 *   entity type over 88 positives, which is a per-type recall denominator of 3
 *   for private keys -- the quantisation this corpus was built to remove. The
 *   deck deals whole shuffled passes over the family list, so every family is
 *   used within one of every other. Adding a carrier therefore re-deals
 *   families downstream of it, which is what `GENERATOR_VERSION` is for.
 * - Objects are built with literal key order and serialized with
 *   `JSON.stringify`, which preserves insertion order for NON-INTEGER-LIKE
 *   string keys -- integer-like keys are emitted first, in numeric order,
 *   whatever the insertion order was. Every accumulated key set that reaches
 *   the output is an entityType id or a policy id, never a number, and every
 *   one is sorted before it is written: `countByType` here, `stagesRun` /
 *   `supplementarySweepsRun` in `certificationSummary`, `sources` /
 *   `taggedSources` in `checkContamination`. Nothing serializes a `Map` or a
 *   `Set` directly.
 *
 * `corpus-artifact.test.ts` holds the other end of this: it regenerates in
 * memory and compares against the committed file, byte for byte.
 */

export const GENERATOR_NAME = "sih-injection-corpus";
/**
 * Bumped whenever the CORPUS bytes change for a fixed seed -- new carriers, new
 * families, changed glue, changed item ids. It is recorded in the manifest and
 * on every item, so a record produced against one corpus can never be silently
 * pooled with a record produced against another.
 *
 * Not the manifest's bytes, and the distinction is load-bearing rather than
 * pedantic: this string is an input to every item's RNG seed below, so bumping
 * it re-mints every value in the corpus. A manifest that reports one more
 * measurement about an unchanged corpus must therefore NOT bump it -- doing so
 * would replace the corpus in order to describe it.
 */
export const GENERATOR_VERSION = "1";
export const DEFAULT_SEED = "sih-p7-injection-v1";

/** Positive families drawn per carrier, one per item. */
const POSITIVES_PER_CARRIER = 4;
/** Confusable-only items per carrier. */
const CONFUSABLE_ITEMS_PER_CARRIER = 2;
/**
 * Recipe indices that additionally carry a confusable at a second slot, so
 * density is a varied dimension and an item can require the arm to separate a
 * real value from a look-alike WITHIN one message.
 */
const DUAL_INJECTION_RECIPES = new Set([1, 3]);
/**
 * Fraction of each stratum assigned to dev. Spec 6.2 sizes dev at ~200 of
 * ~3,000 (about 7%); at this corpus's size that would be a dev slice too small
 * to tune a threshold on, so it is 20% here. Stated rather than silently
 * inherited.
 */
export const DEV_FRACTION = 0.2;

/**
 * `count` families dealt from repeated shuffled passes over `families`. Every
 * family appears floor(count/n) or ceil(count/n) times -- balance by
 * construction rather than by luck.
 */
function deal(rng: () => number, families: readonly Family[], count: number): Family[] {
  const out: Family[] = [];
  while (out.length < count) out.push(...shuffle(rng, families));
  return out.slice(0, count);
}

function shuffle<T>(rng: () => number, xs: readonly T[]): T[] {
  const out = [...xs];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

export interface GenerateOptions {
  readonly seed?: string;
  readonly ir: PolicyIr;
  /** Path-ish label for the IR, recorded in every label's `source`. */
  readonly irSource: string;
  /** sha256 of the IR file bytes. Pins the exact artifact the labels came from. */
  readonly irHash: string;
  readonly carriers?: readonly Carrier[];
  /**
   * Family catalogues, defaulting to the wired ones in `families.ts`.
   *
   * Injectable for the reason `certifyOptions` is: a test that only ever runs
   * the default catalogue cannot tell "reads the catalogue" from "imports the
   * constant", and -- more usefully here -- a candidate wave of families can be
   * DRY-RUN through the whole generator, invariants and all, without emitting
   * an artifact. Spec 6.2 puts certification before labelling, so a wave that
   * has not been certified must be runnable without being committed.
   */
  readonly positiveFamilies?: readonly Family[];
  readonly confusableFamilies?: readonly Family[];
  readonly selfTestExamples: readonly SelfTestExample[];
  readonly certifyOptions?: Omit<CertifyOptions, "ir">;
  readonly devFraction?: number;
  /**
   * A same-shape confusable to inject alongside `family`, or `undefined` to
   * inject none.
   *
   * The defect this closes, MEASURED on
   * `corpora/generated/injection-p-fin-adjudicated-v1.jsonl` by `measureOrthography`:
   * on 58 of its 108 gold spans the crude "return the odd-looking string" oracle
   * returns that span and nothing else, so a reader that understands nothing is a
   * perfect detector on 54% of the gold, and that score would not transfer to
   * real text.
   *
   * The distractor is INJECTED rather than written into the carrier, and the
   * reason is not stylistic. The carriers are the pool certification cleaned:
   * `certify.ts`'s sweeps quarantine any carrier holding a second organisation
   * name (the orthographic sweep), a high-entropy token in a code or kv segment,
   * or anything the widened tier-0 rules match. A carrier that carried a
   * shape-matched distractor would therefore not be a certified carrier, and the
   * injection invariant rests on it being one. An injected distractor is a
   * LABELLED span, which is the other half of the reason: an unlabelled
   * odd-looking token in the text is a false positive the corpus manufactured.
   *
   * Absent on both committed corpora, which is why adding it left them
   * byte-identical.
   */
  readonly distractorFor?: (family: Family, rng: () => number) => Family | undefined;
  /**
   * Extra `meta` keys, merged after the generator's own. Lets a build attach
   * provenance the generator has no opinion about -- the label questions the
   * next adjudication round answers, the scoring scope those questions imply --
   * without this module learning what any of it means.
   */
  readonly itemMetaExtra?: (context: {
    readonly carrier: Carrier;
    readonly recipeIndex: number | undefined;
    readonly written: readonly WrittenInjection[];
    readonly labels: readonly LabelledSpan[];
  }) => Record<string, unknown>;
}

export interface NamedGap {
  readonly id: string;
  readonly what: string;
  readonly blockedOn: string;
}

export interface CorpusManifest {
  readonly generator: { readonly name: string; readonly version: string; readonly seed: string };
  readonly policy: { readonly id: string; readonly irSource: string; readonly irHash: string; readonly policyHash: string };
  readonly counts: {
    readonly items: number;
    readonly positives: number;
    readonly negatives: number;
    readonly goldSpans: number;
    readonly confusableSpans: number;
    readonly goldSpansByType: Readonly<Record<string, number>>;
    readonly confusableSpansByType: Readonly<Record<string, number>>;
    /**
     * The resolution of any per-type rate computed on this corpus, stated
     * because the round that built it was framed as "de-quantising" per-type
     * recall and the framing outran the counts. A type with `n` gold spans has
     * a recall that can only take the values `0, 1/n, ..., 1`; `recallStepAtMin`
     * is `1/n` for the scarcest type, which is the coarsest step any per-type
     * number here can move in.
     */
    readonly perTypeResolution: {
      readonly minGoldSpansPerType: number;
      readonly scarcestTypes: readonly string[];
      readonly recallStepAtMin: number;
      readonly note: string;
    };
  };
  readonly certification: CertificationSummary;
  readonly injection: {
    readonly invariant: string;
    readonly itemsChecked: number;
    readonly spansChecked: number;
    readonly carrierSource: string;
  };
  readonly labels: {
    readonly policies: readonly string[];
    readonly populated: readonly string[];
    readonly unpopulated: readonly { readonly policy: string; readonly blockedOn: string; readonly why: string }[];
  };
  readonly contamination: ContaminationReport;
  readonly splits: {
    readonly devFraction: number;
    readonly dev: readonly string[];
    readonly test: readonly string[];
    readonly disjoint: boolean;
    readonly devSha256Note: string;
  };
  readonly gaps: readonly NamedGap[];
}

export interface GeneratedCorpus {
  readonly items: readonly CorpusItem[];
  readonly manifest: CorpusManifest;
  readonly certifications: readonly CarrierCertification[];
}

interface PlannedItem {
  readonly item: CorpusItem;
  readonly labels: readonly LabelledSpan[];
  readonly written: readonly WrittenInjection[];
  readonly stratum: "positive" | "negative";
}

function buildItem(
  carrier: Carrier,
  recipeIndex: number,
  families: readonly Family[],
  options: GenerateOptions,
  seed: string,
): PlannedItem {
  const rng = seededRng(`${seed}|${GENERATOR_VERSION}|${carrier.id}|${recipeIndex}`);
  const base = carrierText(carrier);
  const slots = carrierSlots(carrier);

  const injections: Injection[] = [];
  const dimensionsFor = (family: Family, slotIndex: number, role?: string) => ({
    surface: family.surface,
    difficulty: family.difficulty,
    position: slotPosition(slotIndex, slots.length),
    glueRegister: family.register,
    carrierRegister: carrier.register,
    registerMatch: family.register === carrier.register ? "true" : "false",
    constructedRole: role ?? family.constructedRole,
  });
  // A value already written in this item under a DIFFERENT type. The fourth
  // injection invariant refuses that contradiction, and re-minting is the only
  // way forward, since the generator gets one pass. Bounded rather than
  // `while`: a family whose mint returns a CONSTANT can never escape, and
  // spinning forever on that would be worse than failing with the fourth
  // check's message, which names both types.
  const mintDistinct = (family: Family): string => {
    let value = family.mint(rng);
    for (let attempt = 0; attempt < 8; attempt += 1) {
      if (!injections.some((i) => i.value === value && i.type !== family.type)) break;
      value = family.mint(rng);
    }
    return value;
  };
  const placed: { family: Family; slotIndex: number }[] = [];
  for (let k = 0; k < families.length; k += 1) {
    const family = families[k]!;
    const slotIndex = (recipeIndex + k) % slots.length;
    // A family pool shared between a client family and a vendor family (which
    // is what `DUAL_ROLE_ORGS` is FOR -- it removes the confound where the
    // organisation's name gives its role away) lets both draw the same name
    // into one item. That item would label one string `client-name` and
    // `neg:...-vendor` at once, and `assertInjectionInvariant`'s fourth check
    // refuses it -- correctly, and with no way forward, since the generator
    // gets one pass. Re-minting is the way forward: `mint` is a pure function
    // of the rng, so a second call draws again and the corpus stays a pure
    // function of the seed.
    //
    // MEASURED over every (positive family x confusable family x carrier)
    // pairing available to wave 1 and wave 2 together: 70 of 40,730 collide,
    // and every one is two organisation families drawing one name. Also
    // MEASURED, by deleting this loop and re-running: at the DEFAULT seed the
    // dealt corpus happens to contain no collision at all, so nothing on the
    // default path exercises this and a test that only generates the default
    // corpus cannot tell the loop from its absence. `corpus-candidate.test.ts`
    // forces one with a two-name pool instead. Bounded
    // rather than `while`: a family whose mint returns a CONSTANT (the tutorial
    // key, the redaction placeholder) can never escape a collision, and
    // spinning forever on that would be worse than failing with the fourth
    // check's message, which names both types.
    const value = mintDistinct(family);
    const { prefix, suffix } = family.glue(value);
    injections.push({
      at: slots[slotIndex]!,
      prefix,
      value,
      suffix,
      type: family.type,
      family: family.id,
      dimensions: dimensionsFor(family, slotIndex),
    });
    placed.push({ family, slotIndex });
    // The companion, if any, goes in at the SAME slot and immediately after, so
    // the two spans land in one clause. `applyInjections` sorts by `at` and
    // breaks ties by input order, so "immediately after" is a property of this
    // push order and not of the offsets.
    const companion = family.companion?.(value);
    if (companion !== undefined) {
      injections.push({
        at: slots[slotIndex]!,
        prefix: companion.prefix,
        value: companion.value,
        suffix: companion.suffix,
        type: companion.type,
        family: companion.family,
        dimensions: dimensionsFor(family, slotIndex, companion.constructedRole),
      });
    }
  }

  // Distractors last, so a distractor can never displace a family's own value
  // out of the slot the recipe chose for it. One slot along from the span it
  // shadows: far enough to be a separate clause, near enough to be in the same
  // message, which is the only place an orthographic reader would look.
  if (options.distractorFor !== undefined) {
    for (const { family, slotIndex } of placed) {
      const distractor = options.distractorFor(family, rng);
      if (distractor === undefined) continue;
      const at = (slotIndex + 1) % slots.length;
      const value = mintDistinct(distractor);
      const { prefix, suffix } = distractor.glue(value);
      injections.push({
        at: slots[at]!,
        prefix,
        value,
        suffix,
        type: distractor.type,
        family: distractor.id,
        dimensions: { ...dimensionsFor(distractor, at), distractorFor: family.id },
      });
    }
  }

  const { text, injections: written } = applyInjections(base, injections);
  const labels: LabelledSpan[] = written.map((w) => ({
    span: w.span,
    type: w.type,
    violatesUnder: buildViolatesUnder(options.ir, w.type, options.irSource),
  }));
  const gold = toPFinGold(labels);

  const item = CorpusItemSchema.parse({
    id: `inj-${carrier.id}-${recipeIndex}`,
    text,
    policy: "p-fin",
    gold,
    meta: {
      ...itemMeta(carrier, seed, options, written, labels),
      ...(options.itemMetaExtra?.({ carrier, recipeIndex, written, labels }) ?? {}),
    },
  });
  return { item, labels, written, stratum: gold.length > 0 ? "positive" : "negative" };
}

function itemMeta(
  carrier: Carrier,
  seed: string,
  options: GenerateOptions,
  written: readonly WrittenInjection[],
  labels: readonly LabelledSpan[],
): Record<string, unknown> {
  const hasOrg = written.some((w) => w.dimensions["constructedRole"] !== "none");
  return {
    generator: GENERATOR_NAME,
    generatorVersion: GENERATOR_VERSION,
    // On every item, not only in the manifest: a JSONL file has no header, so
    // an item lifted out of the file would otherwise carry no way back to the
    // run that made it.
    seed,
    carrierId: carrier.id,
    carrierSource: "hand-authored",
    carrierRegister: carrier.register,
    // Emitted only when the carrier declares one, so that adding the field to
    // `Carrier` left every existing item's bytes untouched -- the committed
    // artifact still reproduces, which is the check that this was additive
    // rather than a claim that it was.
    ...(carrier.stratum === undefined ? {} : { carrierStratum: carrier.stratum }),
    // The gap `../driver/corpus.ts` names in its own comment -- "Nothing in this
    // file records which IR an item's labels were written against, so a policy
    // edited without relabelling its corpus fails silently". This closes it for
    // items this generator emits.
    irHash: options.irHash,
    density: written.length,
    injections: written.map((w) => ({
      start: w.span.start,
      end: w.span.end,
      text: w.span.text,
      type: w.type,
      family: w.family,
      dimensions: w.dimensions,
    })),
    // The full three-policy label, including the confusables that are
    // deliberately absent from `gold`. See `toPFinGold`.
    labels,
    ...(hasOrg
      ? {
          predicateLabelling: {
            state: "unlabelled",
            blockedOn:
              "a blind two-annotator labelling round with adjudication, per the protocol in " +
              "corpora/fixtures/smoke.gold-tier2.jsonl",
            why:
              "the generator knows which role clause it wrote (see injections[].dimensions.constructedRole) " +
              "but that is a fact about generation, not an adjudicated judgement about the finished text. " +
              "No pred: gold is emitted here.",
          },
        }
      : {}),
  };
}

function buildPristineNegative(carrier: Carrier, options: GenerateOptions, seed: string): PlannedItem {
  const text = carrierText(carrier);
  const item = CorpusItemSchema.parse({
    id: `neg-${carrier.id}`,
    text,
    policy: "p-fin",
    gold: [],
    meta: {
      generator: GENERATOR_NAME,
      generatorVersion: GENERATOR_VERSION,
      seed,
      carrierId: carrier.id,
      carrierSource: "hand-authored",
      carrierRegister: carrier.register,
      // See `itemMeta`. The pristine negatives are where this matters most: a
      // hard-negative carrier's uninjected item is the over-blocking measurement.
      ...(carrier.stratum === undefined ? {} : { carrierStratum: carrier.stratum }),
      irHash: options.irHash,
      density: 0,
      injections: [],
      labels: [],
      ...(options.itemMetaExtra?.({ carrier, recipeIndex: undefined, written: [], labels: [] }) ?? {}),
    },
  });
  return { item, labels: [], written: [], stratum: "negative" };
}

/**
 * Reads `meta.injections` back into the shape `assertInjectionInvariant` takes.
 * Everything except the span is filled with placeholders: the invariant only
 * inspects `span` and `value`, and inventing a `type` here would let a wrong
 * one round-trip unnoticed.
 */
function readMetaInjections(item: CorpusItem): WrittenInjection[] {
  const raw = (item.meta?.["injections"] ?? []) as readonly Record<string, unknown>[];
  return raw.map((r) => {
    const start = r["start"];
    const end = r["end"];
    const text = r["text"];
    if (typeof start !== "number" || typeof end !== "number" || typeof text !== "string") {
      throw new Error(`item ${item.id} has a malformed meta.injections entry`);
    }
    return {
      at: start,
      prefix: "",
      value: text,
      suffix: "",
      type: String(r["type"]),
      family: String(r["family"]),
      dimensions: {},
      span: { start, end, text },
    };
  });
}

function countByType(labels: readonly LabelledSpan[], irBacked: boolean): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const l of labels) {
    if (isIrBacked(l.type) !== irBacked) continue;
    counts[l.type] = (counts[l.type] ?? 0) + 1;
  }
  // Sorted: the manifest is compared byte for byte, and object key order from
  // an accumulation loop follows encounter order, which depends on carrier
  // order in a way nobody should have to reason about.
  const sorted: Record<string, number> = {};
  for (const k of Object.keys(counts).sort()) sorted[k] = counts[k]!;
  return sorted;
}

/**
 * Derived from `goldSpansByType`, never from a target. Ties are all listed:
 * naming one scarcest type when two are tied would make the field a fact about
 * sort order.
 */
function perTypeResolution(goldSpansByType: Readonly<Record<string, number>>): {
  minGoldSpansPerType: number;
  scarcestTypes: string[];
  recallStepAtMin: number;
  note: string;
} {
  const entries = Object.entries(goldSpansByType);
  const min = entries.length === 0 ? 0 : Math.min(...entries.map(([, n]) => n));
  return {
    minGoldSpansPerType: min,
    scarcestTypes: entries.filter(([, n]) => n === min).map(([t]) => t).sort(),
    recallStepAtMin: min === 0 ? 0 : 1 / min,
    note:
      entries.length === 0
        ? "no type carries gold, so no per-type rate is defined at all"
        : `per-type recall on this corpus is quantised: the scarcest type has ${min} gold span(s), ` +
          `so its recall moves in steps of ${(100 / min).toFixed(1)} percentage points and no ` +
          `per-type difference smaller than that is resolvable. Reported because it bounds every ` +
          `per-type number computed here.`,
  };
}

export const NAMED_GAPS: readonly NamedGap[] = [
  {
    id: "carriers-not-sharechat",
    what:
      "Carriers are hand-authored, not ShareChat/WildChat conversations. Spec 6.2's negatives (~1,500) " +
      "and its 'real carriers' claim for positives are both unmet.",
    blockedOn: "a network fetch and a licensing review, neither available in this session",
  },
  {
    id: "certification-stage-2",
    what: "Certification stage 2 (high-recall Python model sweep) did not run.",
    blockedOn: "a Python corpora/ toolchain with GLiNER2-PII and gliner-pii-large",
  },
  {
    id: "certification-stage-3",
    what:
      "Certification stage 3 (frontier adjudication against the union of all three policies) did not run, " +
      "so no carrier is certified-clear and no corpus emitted here is certified.",
    blockedOn: "a live frontier call (deferred since 2026-08-18) and compiled p-med / p-corp IRs",
  },
  {
    id: "labels-p-med-p-corp",
    what:
      "violatesUnder carries all three policies; only p-fin is populated. p-med and p-corp are marked " +
      "unpopulated and reading an action out of one throws.",
    blockedOn: "compiling policies/p-med.md and policies/p-corp.md",
  },
  {
    id: "difficulty-implicit",
    what:
      "Spec 6.2's third difficulty tier, 'implicit' ('our biggest client, the Cupertino fruit company'), " +
      "is not generated: it injects no literal value, so no gold span can be derived from the write.",
    blockedOn: "a labelling round that can annotate a span nothing wrote",
  },
  {
    id: "tier2-predicate-gold",
    what:
      "No pred: gold is emitted. Items carrying an organisation are stratified by constructedRole and " +
      "marked meta.predicateLabelling.state = 'unlabelled'.",
    blockedOn: "a blind two-annotator labelling round with adjudication",
  },
  {
    id: "realism-gates",
    what:
      "None of spec 6.2's three realism gates ran: no frontier naturalness score, no adversarial style " +
      "probe, no human spot check. The style probe is the one most likely to fail on a hand-authored pool.",
    blockedOn: "a frontier adjudicator, a trained probe classifier, and human annotators",
  },
  {
    id: "residual-contamination-audit",
    what:
      "Spec 6.2's honesty mechanism (a) -- human review of ~200 certified-clear carriers producing a " +
      "'contamination <= X% at 95% confidence' bound -- did not run.",
    blockedOn: "human annotators",
  },
];

/**
 * Carrier ids must be unique across the whole offered pool, and nothing checked.
 *
 * The pool is assembled by concatenation -- `OFFERED_CARRIERS` is
 * `[...ALL_CARRIERS, ...CANDIDATE_CARRIERS]`, from two hand-authored modules
 * that do not import each other, and `build-v2.ts` adds a third. A repeated id
 * across two of them is a plausible editing mistake and every consequence of it
 * is silent:
 *
 * - `new Map(certifications.map(c => [c.carrierId, c]))` keeps the LAST
 *   certification for a repeated id, so the first carrier's quarantine verdict
 *   is discarded and its clone's verdict answers for both. A quarantined
 *   carrier can be admitted this way.
 * - item ids are built as `inj-<carrierId>-<recipeIndex>`, so two carriers
 *   sharing an id produce colliding item ids, and a scorer joining records to
 *   gold by item id would join to the wrong text.
 * - `adjudication.ts` resolves verdicts by carrier id too, so one carrier's
 *   two-certifier clearance would cover a different carrier's text.
 *
 * Thrown from, not reported: none of the three has a sensible partial answer.
 */
export function assertUniqueCarrierIds(carriers: readonly Carrier[]): void {
  const seen = new Set<string>();
  const duplicates: string[] = [];
  for (const c of carriers) {
    if (seen.has(c.id) && !duplicates.includes(c.id)) duplicates.push(c.id);
    seen.add(c.id);
  }
  if (duplicates.length > 0) {
    throw new Error(
      `carrier ids must be unique across the offered pool; repeated: ${duplicates.join(", ")}. ` +
        `A repeated id silently takes one carrier's certification for another and collides their ` +
        `item ids.`,
    );
  }
}

export function generateCorpus(options: GenerateOptions): GeneratedCorpus {
  const seed = options.seed ?? DEFAULT_SEED;
  const carriers = options.carriers ?? ALL_CARRIERS;
  const devFraction = options.devFraction ?? DEV_FRACTION;

  assertUniqueCarrierIds(carriers);

  const certifications = carriers.map((c) =>
    certifyCarrier(c.id, carrierText(c), { ir: options.ir, ...options.certifyOptions }),
  );
  const byId = new Map(certifications.map((c) => [c.carrierId, c]));

  // Spec 6.2: injection happens only into carriers that came through
  // certification without a hit. `quarantined` carriers are excluded from the
  // corpus entirely; they still appear in the manifest's certification summary,
  // which is what makes the quarantine rate a reported number.
  const usable = carriers.filter((c) => byId.get(c.id)!.status !== "quarantined");

  const positiveFamilies = options.positiveFamilies ?? POSITIVE_FAMILIES;
  const confusableFamilies = options.confusableFamilies ?? CONFUSABLE_FAMILIES;
  const dealRng = seededRng(`${seed}|${GENERATOR_VERSION}|deal`);
  const positiveDeck = deal(dealRng, positiveFamilies, usable.length * POSITIVES_PER_CARRIER);
  const confusableDeck = deal(
    dealRng,
    confusableFamilies,
    usable.length * (CONFUSABLE_ITEMS_PER_CARRIER + DUAL_INJECTION_RECIPES.size),
  );
  let positiveCursor = 0;
  let confusableCursor = 0;

  const planned: PlannedItem[] = [];
  for (const carrier of usable) {
    for (let ri = 0; ri < POSITIVES_PER_CARRIER; ri += 1) {
      const families: Family[] = [positiveDeck[positiveCursor++]!];
      if (DUAL_INJECTION_RECIPES.has(ri)) families.push(confusableDeck[confusableCursor++]!);
      planned.push(buildItem(carrier, ri, families, options, seed));
    }
    for (let k = 0; k < CONFUSABLE_ITEMS_PER_CARRIER; k += 1) {
      planned.push(
        buildItem(carrier, POSITIVES_PER_CARRIER + k, [confusableDeck[confusableCursor++]!], options, seed),
      );
    }
    planned.push(buildPristineNegative(carrier, options, seed));
  }

  const contamination = checkContamination(
    planned.map((p) => ({ id: p.item.id, text: p.item.text })),
    options.selfTestExamples,
  );
  const droppedIds = new Set(contamination.dropped.map((d) => d.itemId));
  const kept = planned.filter((p) => !droppedIds.has(p.item.id));

  // Stratified split: dev must contain both strata or a threshold tuned on it
  // is tuned on one side of the problem. Shuffle is seeded, so the split is a
  // pure function of the seed and is regenerated identically.
  const dev = new Set<string>();
  for (const stratum of ["positive", "negative"] as const) {
    const ids = kept.filter((p) => p.stratum === stratum).map((p) => p.item.id).sort();
    const shuffled = shuffle(seededRng(`${seed}|${GENERATOR_VERSION}|split|${stratum}`), ids);
    for (const id of shuffled.slice(0, Math.ceil(devFraction * ids.length))) dev.add(id);
  }
  const devIds = kept.filter((p) => dev.has(p.item.id)).map((p) => p.item.id);
  const testIds = kept.filter((p) => !dev.has(p.item.id)).map((p) => p.item.id);

  const allLabels = kept.flatMap((p) => p.labels);
  const items = kept.map((p) => p.item);

  // The second check, and deliberately NOT a repeat of the first. The one
  // inside `applyInjections` runs over the spans that function just computed,
  // which is the easy place to be right. This one re-reads the offsets back out
  // of `meta.injections` -- the serialized form, the only thing a downstream
  // consumer can see -- so a bug in `itemMeta`'s mapping from `span` to
  // `{start, end, text}` is caught at generation time rather than by whoever
  // scores against the file. Verified by mutation: swapping `start` and `end`
  // in `itemMeta` is killed here.
  let spansChecked = 0;
  for (const p of kept) {
    const fromFile = readMetaInjections(p.item);
    if (fromFile.length !== p.written.length) {
      throw new Error(`item ${p.item.id} records ${fromFile.length} injections but wrote ${p.written.length}`);
    }
    assertInjectionInvariant(p.item.text, fromFile);
    spansChecked += fromFile.length;
  }

  const summary = certificationSummary(certifications);
  const manifest: CorpusManifest = {
    generator: { name: GENERATOR_NAME, version: GENERATOR_VERSION, seed },
    policy: {
      id: "p-fin",
      irSource: options.irSource,
      irHash: options.irHash,
      policyHash: options.ir.policyHash,
    },
    counts: {
      items: items.length,
      positives: kept.filter((p) => p.stratum === "positive").length,
      negatives: kept.filter((p) => p.stratum === "negative").length,
      goldSpans: allLabels.filter((l) => isIrBacked(l.type)).length,
      confusableSpans: allLabels.filter((l) => !isIrBacked(l.type)).length,
      goldSpansByType: countByType(allLabels, true),
      confusableSpansByType: countByType(allLabels, false),
      perTypeResolution: perTypeResolution(countByType(allLabels, true)),
    },
    certification: summary,
    injection: {
      invariant:
        "every gold span is the span the generator wrote, derived from the splice offset and never " +
        "re-found by search; text.slice(start, end) === the injected value, and each injected value " +
        "occurs in the text exactly as many times as it was injected",
      itemsChecked: items.length,
      spansChecked,
      carrierSource: "hand-authored",
    },
    labels: {
      policies: [...POLICY_IDS],
      populated: ["p-fin"],
      unpopulated: (["p-med", "p-corp"] as const).map((policy) => {
        const l = unpopulatedLabel(policy);
        if (l.state !== "unpopulated") throw new Error("unpopulatedLabel returned a populated label");
        return { policy, blockedOn: l.blockedOn, why: l.why };
      }),
    },
    contamination,
    splits: {
      devFraction,
      dev: devIds,
      test: testIds,
      disjoint: devIds.every((id) => !testIds.includes(id)),
      devSha256Note:
        "the split is a pure function of the seed and the generator version; regenerating with both " +
        "unchanged reproduces these lists exactly. That is DETERMINISM and it is NOT what spec 6.2 " +
        "means by 'test frozen before any tuning' -- an earlier version of this note claimed it was. " +
        "Spec 6.2 asks that nothing be selected using the test half, and this corpus does not meet " +
        "that: measureLeakage runs over all 189 items, dev and test pooled (build-v2.ts passes the " +
        "whole loadCorpus result), and families were removed and rewritten on what those pooled " +
        "measurements said. The selection was for leakage rather than for accuracy, which makes the " +
        "contamination milder than tuning a threshold would be; it does not make the test slice " +
        "held out. Every arm run so far has also SCORED all 189 with no split on any row, so a " +
        "pooled number cannot be reduced to a test-only one after the fact.",
    },
    gaps: NAMED_GAPS,
  };

  return { items, manifest, certifications };
}

/** JSONL, one item per line, trailing newline. The emitted artifact's bytes. */
export function serializeCorpus(items: readonly CorpusItem[]): string {
  return items.map((i) => JSON.stringify(i)).join("\n") + "\n";
}

export function serializeManifest(manifest: CorpusManifest): string {
  return JSON.stringify(manifest, null, 2) + "\n";
}
