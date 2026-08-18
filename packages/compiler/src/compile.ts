import { loadPolicyIr, type Action, type PolicyIr } from "@sih/core";
import type { LlmClient } from "./llm/client.js";
import { renderReport, type ItemCounts } from "./report.js";
import { emitIr } from "./stages/emit.js";
import { extract, type RejectedCandidate } from "./stages/extract.js";
import { groundProviders, loadManifest } from "./stages/ground.js";
import { mintShadowEntityTypes } from "./stages/predicates.js";
import { runSelfTest, type SelfTestCase } from "./stages/selftest.js";
import { validateIdHygiene, validateRules } from "./stages/validate.js";

/**
 * The end-to-end compile: a policy document in, an auditable IR out.
 *
 * extract -> ground -> predicates -> validate -> emit -> self-test -> report.
 *
 * ORDER NOTE, deviating from the plan's prose, which lists self-test before
 * emit: the self-test stage measures coverage by EXECUTING the compiled policy
 * through core's real detection pipeline, so it needs a loaded `PolicyIr` to
 * run against. The IR must therefore exist before the measurement, and the
 * report — which quotes the measurement — is rendered last. Nothing else moves:
 * the self-test's numbers still describe the exact artifact this function
 * returns, which is the property that made the plan put it late.
 *
 * Two failure classes, kept deliberately distinct:
 *
 * - THROW: a document that yields no entityTypes, an entityType no clause
 *   resolves an action for, an unsafe regex, an invented validator, an IR the
 *   runtime loader rejects. Each of these is unshippable; there is no useful
 *   "compiled with warnings" outcome to hand back.
 * - WARN: a weak entity, an unresolvable provider mention, a conflicting clause,
 *   an id that looks derived from a confidential noun. These are judgements a
 *   human makes from the report, and refusing to emit an IR that is 90% right
 *   helps nobody.
 */

export interface CompileInput {
  /** The compiler's only route to a frontier model. Tests inject fixtures. */
  readonly client: LlmClient;
  /** The policy document, verbatim. Hashed into the IR. */
  readonly document: string;
  /** Parsed `providers.json`; validated by `loadManifest`. */
  readonly manifest: unknown;
  /** Names the report. Not part of the IR — the hash identifies the policy. */
  readonly policyName: string;
  readonly latencyBudgetMs?: number;
}

export interface CompileResult {
  readonly ir: PolicyIr;
  /** Markdown, for a human to audit before shipping. */
  readonly report: string;
  readonly warnings: string[];
  /**
   * True when this exact IR was accepted by `loadPolicyIr` in this process.
   *
   * Always true on a successful return, by construction: every structural
   * failure throws rather than returning `ok: false`, because an IR the runtime
   * rejects is not a result a caller can do anything with. It is computed from
   * the round-trip rather than hardcoded so that it cannot drift from the claim
   * it makes, and it is the flag Task 9's CLI branches on.
   */
  readonly ok: boolean;
  /**
   * The generated corpus with its per-case outcomes. Returned so the evidence
   * behind the coverage numbers ships with the numbers and can be re-executed
   * by anyone auditing them.
   */
  readonly selfTestCases: SelfTestCase[];
}

/** Own-key discipline: every key here is a model-authored id. */
function emptyMap<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

function countByKind(rejected: readonly RejectedCandidate[], kind: string): number {
  return rejected.filter((r) => r.kind === kind).length;
}

/**
 * Refuses to return an IR the runtime could not enforce, with a message that
 * says which policy item is missing rather than which array index failed.
 *
 * Both conditions are `loadPolicyIr` rejections too (`entityTypes` has a `min(1)`
 * and every entityType needs an action mapping). They are checked here first so
 * the compiler explains them in the vocabulary of the document — a candidate was
 * rejected, no clause resolved an action — instead of handing back a zod path.
 */
function assertShippable(
  ir: PolicyIr,
  policyName: string,
  rejected: readonly RejectedCandidate[],
): void {
  if (ir.entityTypes.length === 0) {
    const detail =
      rejected.length === 0
        ? "the model proposed none"
        : `all ${rejected.length} proposed candidates were rejected: ` +
          rejected.map((r) => `${r.kind} "${r.id}" (${r.reason})`).join("; ");
    throw new Error(
      `policy "${policyName}" compiled to no entityTypes, so the IR would enforce nothing — ${detail}`,
    );
  }
  for (const entity of ir.entityTypes) {
    if (!Object.hasOwn(ir.actions.default, entity.id)) {
      throw new Error(
        `policy "${policyName}": entityType "${entity.id}" is actionless — no clause resolved an ` +
          `action for it, so the runtime would detect the value and then forward it unchanged`,
      );
    }
  }
}

export async function compilePolicy(input: CompileInput): Promise<CompileResult> {
  const { client, document, policyName } = input;
  const warnings: string[] = [];

  // 1. Extract, with the anti-hallucination gate already applied: everything
  //    below sees only candidates that quote the document verbatim.
  const extraction = await extract(client, document);
  const rejected: RejectedCandidate[] = [...extraction.rejected];

  // 2. Ground provider mentions to adapter ids.
  const manifest = loadManifest(input.manifest);
  const grounded = groundProviders(manifest, extraction.actions);
  warnings.push(...grounded.warnings);

  // 3. Mint a shadow entityType per semantic predicate, so a tier-2 finding can
  //    reach an action at all.
  const authoredIds = new Set(extraction.entityTypes.map((e) => e.id));
  const shadow = mintShadowEntityTypes(extraction.semanticPredicates, authoredIds);

  // 4. Prove the model's artifacts safe. `validateRules` throws on an unsafe or
  //    unusable rule; id hygiene only ever warns, since the compiler cannot know
  //    which nouns a firm treats as confidential.
  await validateRules(extraction.rules);
  warnings.push(...validateIdHygiene([...extraction.entityTypes, ...shadow.entityTypes]));

  // Shadow defaults are merged after grounding, since they are minted rather
  // than read from a clause. The collision branch is a can't-happen guard: an
  // authored id cannot contain ":" under the extraction prompt, and
  // mintShadowEntityTypes already threw on a collision with a real entityType.
  const defaults = emptyMap<Action>();
  for (const entityId of Object.keys(grounded.defaults)) {
    defaults[entityId] = grounded.defaults[entityId]!;
  }
  for (const shadowId of Object.keys(shadow.defaultActions)) {
    if (Object.hasOwn(defaults, shadowId)) {
      warnings.push(
        `minted default action for shadow entityType "${shadowId}" displaced an action the model ` +
          `emitted for that id; shadows are always "redact" and never pseudonymized`,
      );
    }
    defaults[shadowId] = shadow.defaultActions[shadowId]!;
  }

  // 5. Assemble.
  const emitted = emitIr({
    document,
    entityTypes: extraction.entityTypes,
    rules: extraction.rules,
    semanticPredicates: extraction.semanticPredicates,
    shadowEntityTypes: shadow.entityTypes,
    actions: { default: defaults, providerOverrides: grounded.providerOverrides },
    failMode: extraction.failMode,
    ...(input.latencyBudgetMs === undefined ? {} : { latencyBudgetMs: input.latencyBudgetMs }),
  });
  rejected.push(...emitted.dropped);
  warnings.push(...emitted.warnings);

  assertShippable(emitted.ir, policyName, rejected);

  // 6. Validate our own output through the runtime's loader. The compiler does
  //    not get to decide that what it produced is loadable: the loader does, and
  //    it enforces bounds (the entropy ceiling, nullable regexes, __proto__ keys)
  //    that no stage above duplicates.
  let loaded: PolicyIr;
  try {
    loaded = loadPolicyIr(JSON.stringify(emitted.ir));
  } catch (e) {
    throw new Error(
      `policy "${policyName}": the compiler produced an IR the runtime loader rejects, which is a ` +
        `compiler bug rather than a policy problem — ${(e as Error).message}`,
    );
  }
  const ok = true;

  // 7. Measure the artifact by executing it. Runs against the LOADED IR, not the
  //    assembled one, so the numbers describe exactly what a runtime would run.
  const selfTest = await runSelfTest(client, loaded);
  warnings.push(...selfTest.warnings);

  // What survived the gate, and what the model proposed before it. The
  // "proposed" side is reconstructed from the rejections rather than counted at
  // the source: `extract` returns the survivors, and survivors + rejections IS
  // the model's output, by construction of the gate.
  const groundedCounts: ItemCounts = {
    entityTypes: extraction.entityTypes.length,
    rules: extraction.rules.length,
    semanticPredicates: extraction.semanticPredicates.length,
    actions: extraction.actions.length,
  };
  const proposed: ItemCounts = {
    entityTypes: groundedCounts.entityTypes + countByKind(extraction.rejected, "entityType"),
    rules: groundedCounts.rules + countByKind(extraction.rejected, "rule"),
    semanticPredicates:
      groundedCounts.semanticPredicates + countByKind(extraction.rejected, "semanticPredicate"),
    actions: groundedCounts.actions + countByKind(extraction.rejected, "action"),
  };

  const report = renderReport({
    policyName,
    ir: emitted.ir,
    proposed,
    grounded: groundedCounts,
    rejected,
    selfTest,
    warnings,
  });

  return { ir: emitted.ir, report, warnings, ok, selfTestCases: selfTest.cases };
}
