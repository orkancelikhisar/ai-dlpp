import { loadPolicyIr, runTier0, segmentText, type PolicyIr } from "@sih/core";

/**
 * Spec 6.2 carrier certification: "no prompt gets a label until certified."
 *
 * The spec names three stages. One of them is implementable in this repository
 * today; two are not. The whole design of this module is about making that
 * arithmetic visible instead of letting a two-of-three pass call itself
 * certified -- the exact intent-as-fact defect the standing conventions record
 * (a `backend` field naming what was requested, a `config` capturing three of
 * six dimensions). So:
 *
 * - `CarrierStatus` has THREE values, not two. `"provisional-clear"` is what a
 *   carrier gets when every stage that RAN was clear and some stage did not
 *   run. It is not `"certified-clear"`, it does not read like it, and it does
 *   not compare equal to it.
 * - `"certified-clear"` is reachable only when all three stage results say
 *   `ran: true`. With the runners this repository can supply, that never
 *   happens; `corpus-certify.test.ts` asserts both halves -- unreachable with
 *   the real pipeline, reachable when three stubs are injected -- because a test
 *   that only ever exercises the default configuration cannot tell "derives the
 *   status" from "hardcodes it".
 *
 * ## Which stage is real
 *
 * Stage 1 (tier-0 sweep at max recall) runs, here, in Node, against the same
 * `runTier0` the extension ships. See `maxRecallIr` for what "max recall" was
 * made to mean and what it was MEASURED to change.
 *
 * Stage 1 is also the CIRCULARITY this module named for stage 2 and not for
 * itself. `runTier0` is the tier-0 arm under test, and `maxRecallIr` is a
 * widened copy of the very IR the corpus is scored under, so the carriers that
 * survive certification are the carriers that arm is silent on, chosen by that
 * arm. Spec 6.2 prescribes stage 1 in those words ("tier-0 sweep, thresholds at
 * max recall"), so replacing it would be running a different stage than the one
 * the spec names; what wave 3 adds instead is `formatSpecSweep` in
 * `format-sweep.ts`, an IR-free, detector-free supplementary sweep that runs
 * beside stage 1 and can also quarantine, plus `CertificationSummary.circularity`,
 * which counts the carriers stage 1 quarantined that the independent sweep did
 * not. That count is the size of the circularity, reported rather than argued
 * about. It does not remove the bias -- a pool cleaned by any detector is easy
 * for that detector, which spec 6.2 says outright -- and it does remove "the arm
 * alone decided".
 *
 * Stage 2 (high-recall model sweep) does not run. Spec 6.2 specifies it as
 * Python, using GLiNER2-PII and gliner-pii-large through their Python runtimes.
 * This repository has no Python package, no Python dependency manifest, and no
 * checkout of either model; `analysis/` from spec 2.2 does not exist yet either.
 * Standing in `@sih/tier1` instead would break the stage's own requirement that
 * the sweep be "configured differently from the arms under test" -- tier 1 IS an
 * arm under test -- so the honest result is that the stage is unrun.
 *
 * Stage 3 (frontier adjudication against the union of all three policies) does
 * not run: it needs a live frontier call, deferred by the user since
 * 2026-08-18, and it needs compiled p-med and p-corp policies, which do not
 * exist (see `labels.ts`). Two independent blockers, both named.
 *
 * ## What the missing stages cost, concretely
 *
 * Stage 1 sees only what tier-0 rules see: regex and entropy over identifier
 * shapes. It cannot see a client organisation name (`client-name` is a tier-1
 * entity) and it cannot see a relationship disclosure (a tier-2 predicate). So
 * spec 6.2's invariant -- "injection happens only into certified-clear carriers
 * => on positives, gold spans are exactly the injected ones" -- is established
 * by stage 1 for tier-0 entity classes ONLY. For `client-name` and for
 * `pred:client-relationship-disclosure`, a carrier could carry an uninjected
 * instance and stage 1 would not know. `orthographicOrgSweep` below is a cheap
 * partial guard against the first of those, and it is deliberately not counted
 * as a certification stage.
 */

export const CERTIFICATION_STAGES = [
  "tier0-sweep",
  "high-recall-model-sweep",
  "frontier-adjudication",
] as const;
export type StageId = (typeof CERTIFICATION_STAGES)[number];

/**
 * A supplementary sweep is one this repository invented, not one spec 6.2
 * names. It can quarantine a carrier; it can never move a carrier toward
 * `certified-clear`, because certification is defined by the three stages
 * above. Kept in a separate list so a reader counting "stages run" against the
 * spec gets the same number the spec would.
 */
export const SUPPLEMENTARY_SWEEPS = [
  "orthographic-org-sweep",
  "blind-double-adjudication-p-fin",
  // Wave 3. See `format-sweep.ts`, and the stage-1 paragraph above for why a
  // supplementary sweep rather than a replacement for stage 1.
  "format-spec-sweep",
] as const;
export type SupplementarySweepId = (typeof SUPPLEMENTARY_SWEEPS)[number];

export interface StageHit {
  readonly sweep: string;
  readonly start: number;
  readonly end: number;
  readonly text: string;
  /** What the sweep called it: an entityType id, or a sweep-specific label. */
  readonly label: string;
  /**
   * Why, in prose, when the label alone does not carry it. Optional so that
   * adding it left every hit already recorded in a committed manifest
   * byte-identical -- `JSON.stringify` omits an absent key, and
   * `corpus-artifact.test.ts` is the check that this was additive rather than a
   * claim that it was.
   */
  readonly note?: string;
}

export type SweepResult =
  | { readonly sweep: string; readonly ran: true; readonly detector: string; readonly hits: readonly StageHit[] }
  | { readonly sweep: string; readonly ran: false; readonly blockedOn: string; readonly why: string };

/**
 * Given a carrier's text, everything the sweep considers sensitive-looking.
 *
 * `carrierId` is passed as a second argument for the sweep whose input is a
 * RECORDED JUDGEMENT rather than the text -- an adjudication round returns a
 * verdict per carrier, and re-deriving which carrier this is by matching text
 * would be a second, fallible claim about identity. Every text-only sweep
 * ignores it; a one-argument function is still a valid `Sweep`.
 */
export type Sweep = (text: string, carrierId: string) => readonly StageHit[];

export const UNRUN_STAGES: Readonly<Record<StageId, { blockedOn: string; why: string }>> = {
  "tier0-sweep": {
    blockedOn: "nothing -- this stage runs",
    why: "unreachable: the tier-0 sweep is always supplied",
  },
  "high-recall-model-sweep": {
    blockedOn: "a Python corpora/ toolchain with GLiNER2-PII and gliner-pii-large checked out",
    why:
      "spec 6.2 specifies this stage as offline Python over the strongest available detectors, " +
      "configured differently from the arms under test. This repository has no Python package and " +
      "no checkout of either model, and substituting @sih/tier1 would violate the stage's own " +
      "requirement, since tier 1 is an arm under test.",
  },
  "frontier-adjudication": {
    blockedOn: "a live frontier-model call (deferred by the user since 2026-08-18) and compiled p-med / p-corp IRs",
    why:
      "spec 6.2 requires adjudication against the union of all three policies. Two of the three " +
      "have never been compiled, so the union does not exist as an artifact, and the adjudicator " +
      "itself is the deferred live call.",
  },
};

// -- stage 1: tier-0 at max recall ------------------------------------------

/**
 * Entropy floor for the max-recall sweep. Not a tuned number: it is low enough
 * that ordinary configuration values clear it, which is what "max recall" is
 * supposed to mean for a certification pass.
 *
 * MEASURED against `policies/compiled/p-fin.ir.json` on the kv block
 * `settings:\nLOG_LEVEL=debug\nMODE=production_readonly\nTIMEOUT_SECONDS=30\n`:
 * the stock rule (threshold 4.0, minLength 24) returns 1 finding; at 2.5 / 12 it
 * returns 3, adding `LOG_LEVEL=debug` and `TIMEOUT_SECONDS=30`. So the lever
 * demonstrably moves, and it moves in the direction a quarantine wants.
 */
export const MAX_RECALL_ENTROPY_THRESHOLD = 2.5;
export const MAX_RECALL_MIN_LENGTH = 12;

/**
 * The IR a max-recall sweep runs. Two transforms, both of which were measured
 * to change what tier 0 returns:
 *
 * 1. Every rule's `validator` is dropped. A validator's job is to REJECT
 *    structurally invalid matches, so removing it is the only way to widen a
 *    regex rule at all -- a regex has no threshold to lower. MEASURED on
 *    "The reference on the form reads ABCDE1234F next to the signature.":
 *    the stock IR returns no findings (`pan-structure` rejects `ABCDE1234F`,
 *    whose 4th character `D` is not a holder type) and the max-recall IR
 *    returns `in-pan` at [32,42). This transform is what lets a certification
 *    sweep quarantine a carrier over a value that merely LOOKS like an
 *    identifier, which is precisely what a quarantine is for.
 *
 * 2. Entropy rules take `min(existing, MAX_RECALL_ENTROPY_THRESHOLD)` and
 *    `min(existing ?? default, MAX_RECALL_MIN_LENGTH)`. `min`, not assignment:
 *    a policy that already scans lower must not be RAISED by a sweep whose name
 *    is "max recall".
 *
 * Note what this does not reach. MEASURED: the git-SHA-shaped token in
 * "The build failed at commit 9f3c1a7de84b25c0af61d9e0b73c25a8f0d41b62 in the
 * log." is not returned at 2.5 / 12 either, because `runTier0` scopes entropy
 * rules to code and kv segments and that sentence segments as prose. Lowering
 * the threshold widens entropy coverage only inside the segments entropy
 * already ran in.
 *
 * The result is re-validated through `loadPolicyIr`, so a transform that
 * produced an IR the runtime would refuse fails here rather than in a sweep.
 */
export function maxRecallIr(ir: PolicyIr): PolicyIr {
  const rules = ir.rules.map((rule) => {
    const { validator: _dropped, ...rest } = rule;
    if (rest.entropyThreshold === undefined) return rest;
    return {
      ...rest,
      entropyThreshold: Math.min(rest.entropyThreshold, MAX_RECALL_ENTROPY_THRESHOLD),
      minLength: Math.min(rest.minLength ?? MAX_RECALL_MIN_LENGTH, MAX_RECALL_MIN_LENGTH),
    };
  });
  return loadPolicyIr(JSON.stringify({ ...ir, rules }));
}

export function tier0Sweep(ir: PolicyIr): Sweep {
  const wide = maxRecallIr(ir);
  return (text) =>
    runTier0(wide, text, segmentText(text)).map((f) => ({
      sweep: "tier0-sweep",
      start: f.start,
      end: f.end,
      text: f.text,
      label: f.entityType,
    }));
}

// -- supplementary: orthographic org sweep ----------------------------------

/**
 * Title-Case bigram or longer, e.g. "Meridian Capital". A carrier containing
 * one is quarantined.
 *
 * This is a PROXY and not a detector. It is here because stage 2 is unrun and
 * stage 1 is blind to organisation names, which leaves the `client-name`
 * positives with no guard at all against a carrier that already names an
 * organisation. An orthographic rule is the strongest thing available without a
 * model, and it is strong in the direction that matters: it over-quarantines.
 * MEASURED consequence of that bias, stated so nobody reads it as a detector --
 * it quarantines "Monday Morning" and "New Delhi" as readily as "Meridian
 * Capital", and it is blind to a lowercase organisation name.
 *
 * It never contributes to `certified-clear`; see `SUPPLEMENTARY_SWEEPS`.
 */
const TITLE_CASE_BIGRAM = /\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b/g;

export const orthographicOrgSweep: Sweep = (text) => {
  const hits: StageHit[] = [];
  // The module-level /g literal is shared across calls, which is safe HERE and
  // would not be with a different API. MEASURED on this exact pattern and
  // string: two successive `[...text.matchAll(re)]` return the same single
  // match and leave `re.lastIndex` at 0, because `matchAll` clones the regex
  // rather than advancing it. `re.exec(text)` twice returns the match and then
  // `null`, leaving lastIndex at 40 in between; `re.test(text)` twice returns
  // true then false. So a later switch to `exec` or `test` over this constant
  // would silently skip the head of every second carrier, and the
  // "is not stateful across calls" test in corpus-certify.test.ts is the guard
  // against exactly that.
  for (const m of text.matchAll(TITLE_CASE_BIGRAM)) {
    hits.push({
      sweep: "orthographic-org-sweep",
      start: m.index,
      end: m.index + m[0].length,
      text: m[0],
      label: "title-case-bigram",
    });
  }
  return hits;
};

// -- certification ----------------------------------------------------------

export type CarrierStatus = "certified-clear" | "provisional-clear" | "quarantined";

export interface CarrierCertification {
  readonly carrierId: string;
  readonly status: CarrierStatus;
  /** The three spec 6.2 stages, always in `CERTIFICATION_STAGES` order. */
  readonly stages: readonly SweepResult[];
  /** Sweeps this repository added. Never counted toward certification. */
  readonly supplementary: readonly SweepResult[];
  /** Every hit from every sweep that ran, spec and supplementary alike. */
  readonly hits: readonly StageHit[];
  readonly unrun: readonly StageId[];
}

export interface CertifyOptions {
  readonly ir: PolicyIr;
  /** Stage 2. Absent in this repository; injectable so the status logic is testable. */
  readonly modelSweep?: Sweep;
  /** Stage 3. Absent in this repository; injectable for the same reason. */
  readonly frontierAdjudication?: Sweep;
  /** Defaults to the orthographic sweep; pass `[]` to run none. */
  readonly supplementarySweeps?: readonly { readonly id: SupplementarySweepId; readonly sweep: Sweep }[];
}

export function certifyCarrier(
  carrierId: string,
  text: string,
  options: CertifyOptions,
): CarrierCertification {
  const stage1 = tier0Sweep(options.ir);
  const supplied: Readonly<Record<StageId, Sweep | undefined>> = {
    "tier0-sweep": stage1,
    "high-recall-model-sweep": options.modelSweep,
    "frontier-adjudication": options.frontierAdjudication,
  };

  const stages: SweepResult[] = [];
  const hits: StageHit[] = [];
  const unrun: StageId[] = [];
  for (const stage of CERTIFICATION_STAGES) {
    const sweep = supplied[stage];
    if (sweep === undefined) {
      stages.push({ sweep: stage, ran: false, ...UNRUN_STAGES[stage] });
      unrun.push(stage);
      continue;
    }
    const found = sweep(text, carrierId).map((h) => ({ ...h, sweep: stage }));
    stages.push({
      sweep: stage,
      ran: true,
      detector: stage === "tier0-sweep" ? "@sih/core runTier0 over maxRecallIr(ir)" : "caller-supplied",
      hits: found,
    });
    hits.push(...found);
  }

  const supplementary: SweepResult[] = [];
  const sweeps = options.supplementarySweeps ?? [{ id: "orthographic-org-sweep" as const, sweep: orthographicOrgSweep }];
  for (const { id, sweep } of sweeps) {
    const found = sweep(text, carrierId).map((h) => ({ ...h, sweep: id }));
    supplementary.push({ sweep: id, ran: true, detector: id, hits: found });
    hits.push(...found);
  }

  // Order matters and is spec 6.2's: "any hit -> quarantine" is unconditional,
  // so a hit outranks a missing stage. A carrier with a hit is quarantined
  // whether or not the other stages ran, and saying "provisional" about it
  // would understate what is actually known.
  const status: CarrierStatus =
    hits.length > 0 ? "quarantined" : unrun.length === 0 ? "certified-clear" : "provisional-clear";
  return { carrierId, status, stages, supplementary, hits, unrun };
}

// -- corpus-level claim -----------------------------------------------------

export interface CertificationSummary {
  /**
   * The word a reader will quote. Derived from the stage results by
   * `certificationSummary`; there is no setter and no argument that can make it
   * say "CERTIFIED" while a stage is unrun -- nor, since the vacuous case was
   * found, while there is no carrier to certify.
   */
  readonly claim: "CERTIFIED" | "NOT CERTIFIED";
  readonly stagesRun: readonly StageId[];
  readonly stagesUnrun: readonly { readonly stage: StageId; readonly blockedOn: string; readonly why: string }[];
  readonly supplementarySweepsRun: readonly string[];
  readonly carriers: {
    readonly total: number;
    readonly certifiedClear: number;
    readonly provisionalClear: number;
    readonly quarantined: number;
  };
  readonly quarantined: readonly { readonly carrierId: string; readonly hits: readonly StageHit[] }[];
  /**
   * What the invariant in spec 6.2 is actually established for, given the
   * stages that ran. A sentence rather than a flag, because it is a caveat a
   * human has to read.
   */
  readonly invariantScope: string;
  /**
   * Present only when an IR-free sweep ran beside stage 1, so that adding this
   * field left both already-committed manifests byte-identical -- which
   * `corpus-artifact.test.ts` and `corpus-adjudicated.test.ts` check.
   *
   * The numbers say how much of the quarantine decision stage 1 made ALONE.
   * `stage1Only` is the set of carriers the tier-0 arm removed and the
   * independent sweep would have kept; it is the exact size of the circularity
   * this pipeline was carrying unstated.
   */
  readonly circularity?: {
    readonly independentSweep: string;
    readonly note: string;
    readonly stage1Only: readonly string[];
    readonly independentOnly: readonly string[];
    readonly both: readonly string[];
  };
}

/**
 * What every stage is blocked on when NOTHING was submitted. Not
 * `UNRUN_STAGES`: that table explains why a stage did not run for a carrier,
 * and "unreachable: the tier-0 sweep is always supplied" is a false sentence
 * about a pool with no carriers in it.
 */
const VACUOUS_BLOCKER = {
  blockedOn: "at least one carrier",
  why:
    "certificationSummary was called with an empty certification list, so no stage ran on anything " +
    "and no carrier was cleared. Every stage is unrun here for that reason and not for the reason " +
    "UNRUN_STAGES gives.",
} as const;

/**
 * The vacuous summary. Split out and returned before any counting, because the
 * general path derives `allStagesRan` from `stagesUnrun.size === 0` and an empty
 * input makes that set empty for the wrong reason: with no carrier there is no
 * stage result saying `ran: false`, so the general path concluded all three
 * stages ran, found `certifiedClear === total` at 0 === 0, and returned
 * `claim: "CERTIFIED"` with `stagesRun: []`. A gate whose job is to refuse
 * cannot have "nothing was submitted" as its most permissive input.
 */
function vacuousSummary(): CertificationSummary {
  return {
    claim: "NOT CERTIFIED",
    stagesRun: [],
    stagesUnrun: CERTIFICATION_STAGES.map((stage) => ({ stage, ...VACUOUS_BLOCKER })),
    supplementarySweepsRun: [],
    carriers: { total: 0, certifiedClear: 0, provisionalClear: 0, quarantined: 0 },
    quarantined: [],
    invariantScope:
      "no carrier was submitted for certification, so the spec 6.2 invariant is established for " +
      "nothing. This is not a clean pool; it is an empty one.",
  };
}

export function certificationSummary(certs: readonly CarrierCertification[]): CertificationSummary {
  if (certs.length === 0) return vacuousSummary();
  const stagesUnrun = new Map<StageId, { stage: StageId; blockedOn: string; why: string }>();
  const stagesRun = new Set<StageId>();
  const supplementarySweepsRun = new Set<string>();
  for (const cert of certs) {
    for (const s of cert.stages) {
      if (s.ran) stagesRun.add(s.sweep as StageId);
      else stagesUnrun.set(s.sweep as StageId, { stage: s.sweep as StageId, blockedOn: s.blockedOn, why: s.why });
    }
    for (const s of cert.supplementary) supplementarySweepsRun.add(s.sweep);
  }
  const counts = { total: certs.length, certifiedClear: 0, provisionalClear: 0, quarantined: 0 };
  for (const cert of certs) {
    if (cert.status === "certified-clear") counts.certifiedClear += 1;
    else if (cert.status === "provisional-clear") counts.provisionalClear += 1;
    else counts.quarantined += 1;
  }
  const allStagesRan = stagesUnrun.size === 0;
  const independentId = "format-spec-sweep";
  const independentRan = supplementarySweepsRun.has(independentId);
  const hitBy = (cert: CarrierCertification, sweep: string) => cert.hits.some((h) => h.sweep === sweep);
  const circularity = independentRan
    ? {
        independentSweep: independentId,
        note:
          "stage 1 is runTier0 over maxRecallIr(ir) -- the tier-0 arm under test, widened. " +
          "stage1Only names the carriers it quarantined that the IR-free format-spec sweep did not, " +
          "which is the part of the quarantine decision the arm made on its own. A pool cleaned by " +
          "any detector is easy for that detector; this counts how much of the cleaning was the arm's.",
        stage1Only: certs
          .filter((c) => hitBy(c, "tier0-sweep") && !hitBy(c, independentId))
          .map((c) => c.carrierId),
        independentOnly: certs
          .filter((c) => !hitBy(c, "tier0-sweep") && hitBy(c, independentId))
          .map((c) => c.carrierId),
        both: certs.filter((c) => hitBy(c, "tier0-sweep") && hitBy(c, independentId)).map((c) => c.carrierId),
      }
    : undefined;
  return {
    claim: allStagesRan && counts.certifiedClear === counts.total ? "CERTIFIED" : "NOT CERTIFIED",
    stagesRun: CERTIFICATION_STAGES.filter((s) => stagesRun.has(s)),
    stagesUnrun: CERTIFICATION_STAGES.filter((s) => stagesUnrun.has(s)).map((s) => stagesUnrun.get(s)!),
    supplementarySweepsRun: [...supplementarySweepsRun].sort(),
    carriers: counts,
    quarantined: certs
      .filter((c) => c.status === "quarantined")
      .map((c) => ({ carrierId: c.carrierId, hits: c.hits })),
    invariantScope:
      (allStagesRan
        ? "all three spec 6.2 stages ran; gold spans on positives are the injected spans. "
        : "stage 1 only: the injection invariant is established for tier-0 entity classes. " +
          "A carrier may still carry an uninjected tier-1 name or a tier-2 relationship that no " +
          "stage that ran can see, so a finding of those classes outside a gold span is not " +
          "provably a false positive. ") +
      // The circularity, stated here rather than only in `circularity`, because
      // this is the sentence a reader quotes. It is the same objection this
      // module raises against standing @sih/tier1 in for stage 2 -- tier 1 is an
      // arm under test -- and stage 1 runs runTier0 over a widened copy of the
      // scoring IR, so it is an arm under test too. Spec 6.2 prescribes stage 1
      // in those words, so it is not replaced; it is disclosed and measured.
      "CIRCULARITY: stage 1 is @sih/core runTier0 over maxRecallIr(ir) -- the tier-0 arm under " +
      "test, widened, run against a widened copy of the very IR the corpus is scored under. The " +
      "carriers that survive it are the carriers that arm is silent on, chosen by that arm, so " +
      "the negatives are easy for it by construction. " +
      (circularity === undefined
        ? "No IR-free sweep ran beside it here, so the size of that bias is UNMEASURED in this " +
          "corpus; see format-sweep.ts for the sweep that measures it."
        : `Measured against the IR-free ${circularity.independentSweep}: ` +
          `${circularity.stage1Only.length} carrier(s) quarantined by stage 1 alone, ` +
          `${circularity.both.length} by both, ${circularity.independentOnly.length} by the ` +
          `independent sweep alone.`),
    ...(circularity === undefined ? {} : { circularity }),
  };
}
