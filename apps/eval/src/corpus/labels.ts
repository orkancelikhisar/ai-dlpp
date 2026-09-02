import type { Action, PolicyIr } from "@sih/core";

/**
 * Spec 6.2's label schema: `{span, type, violatesUnder: {P-FIN, P-MED, P-CORP}}`
 * -- one corpus, three ground truths.
 *
 * ## Why `violatesUnder` is not `Record<PolicyId, Action>`
 *
 * Only p-fin has a compiled IR. `policies/compiled/` holds `p-fin.ir.json` and
 * nothing else; `policies/p-med.md` and `policies/p-corp.md` are prose that has
 * never been through the compiler, and compiling them needs the live frontier
 * call this project has deferred since 2026-08-18. So for two of the three
 * policies there is no artifact from which an action could be read.
 *
 * The flat shape leaves nowhere to say that. Every spelling of "unknown" inside
 * an `Action` is a lie a scorer would silently believe: `"allow"` says the
 * policy permits the span, `"block"` says it forbids it, and an omitted key
 * says the span is unlabelled rather than the policy uncompiled. This project
 * has shipped that defect before under a different name -- a `backend` field
 * naming what was requested rather than what ran -- and it produces confident
 * wrong numbers with a green suite. Hence the discriminated union: reading an
 * action out of a label is a `switch`, and the unpopulated arm has no action to
 * read.
 */

export const POLICY_IDS = ["p-fin", "p-med", "p-corp"] as const;
export type PolicyId = (typeof POLICY_IDS)[number];

/**
 * `Action` plus `"none"`, matching `GoldSpanSchema.action` in
 * `../driver/corpus.ts` rather than core's four-way `Action`. `"none"` means
 * the policy demands nothing of this span -- a span worth labelling because a
 * detector will plausibly fire on it, which the policy deliberately permits.
 */
export type LabelAction = "none" | Action;

export type PolicyLabel =
  | {
      readonly state: "populated";
      readonly action: LabelAction;
      /** Where the action was read from. A provenance string, not a claim. */
      readonly source: string;
    }
  | {
      readonly state: "unpopulated";
      /** The concrete thing that has to happen before this can be populated. */
      readonly blockedOn: string;
      readonly why: string;
    };

/**
 * Prefix for a corpus label type that is NOT an entityType of any compiled IR.
 *
 * Confusables need a type -- "an Aadhaar-shaped number with a broken check
 * digit" is a fact about the span, and it is the fact the whole corpus exists
 * to measure a classifier against. But giving one an IR id would assert that it
 * IS that entity, and giving it a bare invented id would put it in the same
 * namespace as `Finding.entityType`, where a scorer joins gold to findings by
 * string equality. The prefix keeps the two namespaces apart on sight, the way
 * core's `SHADOW_PREFIX` ("pred:") does for predicate shadow ids.
 */
export const NEG_PREFIX = "neg:";

export function isIrBacked(type: string): boolean {
  return !type.startsWith(NEG_PREFIX);
}

export interface LabelSpan {
  readonly start: number;
  readonly end: number;
  /** Exactly `text.slice(start, end)` on the item's text. */
  readonly text: string;
}

export interface LabelledSpan {
  readonly span: LabelSpan;
  readonly type: string;
  readonly violatesUnder: Readonly<Record<PolicyId, PolicyLabel>>;
}

/**
 * The single reason p-med and p-corp are unpopulated, written once so every
 * label in every item carries the same sentence and a reader can grep for it.
 */
export function unpopulatedLabel(policy: PolicyId): PolicyLabel {
  return {
    state: "unpopulated",
    blockedOn: "a live policy compile (frontier call), deferred by the user since 2026-08-18",
    why:
      `policies/${policy}.md has never been compiled: policies/compiled/ contains p-fin.ir.json ` +
      `and no ${policy}.ir.json, so there is no actions table to read an action from. ` +
      `Nothing here guesses one.`,
  };
}

/**
 * The p-fin label for a span of `type`, read out of the compiled IR.
 *
 * Two populated cases, and the second is the one worth reading twice:
 *
 * - `type` is one of the IR's entityTypes -> the action is
 *   `ir.actions.default[type]`, verbatim.
 * - `type` is a `neg:` confusable -> the action is `"none"`, DERIVED from the
 *   IR rather than assumed: p-fin's obligations are enumerated by
 *   `actions.default`, a type absent from that table has no obligation under
 *   p-fin, and a confusable is by construction not an instance of any type in
 *   it. That derivation is the reason a confusable can be labelled at all
 *   without a second adjudication round.
 *
 * `providerOverrides` is deliberately NOT consulted. The override table keys on
 * a provider id (chatgpt / gemini / deepseek), which is a property of the RUN,
 * not of the corpus -- the same span is `pseudonymize` by default and `block`
 * to ChatGPT. Baking one provider's answer into the corpus would freeze a run
 * dimension into the ground truth. A scorer resolves the provider itself, with
 * core's `resolveAction`, from the action recorded here.
 */
export function pFinLabel(ir: PolicyIr, type: string, irSource: string): PolicyLabel {
  if (!isIrBacked(type)) {
    return {
      state: "populated",
      action: "none",
      source: `${irSource}: "${type}" is not an entityType of this IR, so p-fin demands nothing of it`,
    };
  }
  const action = ir.actions.default[type];
  if (action === undefined) {
    throw new Error(
      `corpus label type "${type}" has no action in ${irSource} actions.default; ` +
        `either it is a typo or it needs the ${NEG_PREFIX} prefix`,
    );
  }
  return { state: "populated", action, source: `${irSource}#actions.default.${type}` };
}

/**
 * Reads an action out of a label. THROWS on the unpopulated arm rather than
 * returning a default, because every default is wrong in a way that scores:
 * "allow" inflates precision, "block" inflates recall, and skipping the span
 * silently shrinks a denominator. A scorer that reaches an unpopulated label is
 * scoring a policy this corpus has no ground truth for, and it should stop.
 */
export function actionUnder(label: LabelledSpan, policy: PolicyId): LabelAction {
  const l = label.violatesUnder[policy];
  if (l.state === "unpopulated") {
    throw new Error(
      `no ground truth for policy "${policy}" on span "${label.span.text}" (${label.type}): ${l.why}`,
    );
  }
  return l.action;
}

export function buildViolatesUnder(
  ir: PolicyIr,
  type: string,
  irSource: string,
): Record<PolicyId, PolicyLabel> {
  return {
    "p-fin": pFinLabel(ir, type, irSource),
    "p-med": unpopulatedLabel("p-med"),
    "p-corp": unpopulatedLabel("p-corp"),
  };
}

/**
 * The p-fin projection into `GoldSpanSchema`'s shape (see `../driver/corpus.ts`).
 *
 * IR-backed labels only. A `neg:` confusable is deliberately absent from `gold`
 * even though it is a labelled span, for two independent reasons: its type is
 * not in the `Finding.entityType` namespace a scorer joins on, and its p-fin
 * action is `"none"`, so it contributes to no numerator and no denominator.
 * Dropping it is what makes spec 6.2's invariant literally true of the emitted
 * file -- "any finding outside [the gold spans] is a true false positive" -- and
 * a finding landing on a confusable is exactly the false positive the
 * confusable was injected to provoke. The full label survives in `meta.labels`,
 * so the diagnosis ("which confusable family fooled it") is not lost.
 *
 * `action: "none"` therefore never appears in an emitted gold span today. It is
 * expected to once p-med compiles: `policies/p-med.md` §4.1 says client
 * organisation names "may be shared freely", qualified by §4.5, which withdraws
 * that permission when the name is combined with a patient's clinical detail in
 * the same prompt. So the same `client-name` span looks like `pseudonymize`
 * under p-fin and `none` under p-med -- READ OFF THE PROSE, which is precisely
 * why the p-med field is unpopulated rather than filled in from this paragraph.
 * The field is kept because that projection is the point of carrying three
 * ground truths, not because anything currently writes it.
 */
export function toPFinGold(
  labels: readonly LabelledSpan[],
): { start: number; end: number; text: string; entityType: string; action: LabelAction }[] {
  const gold = [];
  for (const label of labels) {
    if (!isIrBacked(label.type)) continue;
    gold.push({
      start: label.span.start,
      end: label.span.end,
      text: label.span.text,
      entityType: label.type,
      action: actionUnder(label, "p-fin"),
    });
  }
  return gold;
}
