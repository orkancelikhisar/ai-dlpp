/**
 * The shadow-entityType naming contract, shared by the two ends that must agree
 * on it: the compiler that MINTS a shadow id per semantic predicate
 * (`packages/compiler/src/stages/predicates.ts`) and the tier-2 judge that must
 * NAME one on every finding it emits (`packages/tier2/src/judge.ts`).
 *
 * It lives in core rather than in the compiler for one mechanical reason:
 * `@sih/compiler` is Node-only (it reads files, shells out, and talks to an
 * SDK), while the tier-2 judge runs in a browser page. Importing the compiler
 * from the page to reach two string operations would drag Node's module graph
 * into the bundle. Core is DOM- and Node-free by design -- `test/firewall.test.ts`
 * enforces that on every file under `src/` -- and both packages already depend
 * on it.
 *
 * A second copy of the prefix in the tier-2 package was the alternative, and it
 * is the wrong fix: the two copies are free to drift, and the failure when they
 * do is `normalizeFindings` throwing on EVERY tier-2 finding, which loses the
 * whole message rather than one label. One definition, imported twice.
 *
 * The reasoning for shadow entityTypes existing at all -- and for the rejected
 * alternative, a `Finding.predicateId` field consulted by action resolution --
 * stays with the minting code in the compiler, which is where a reader asking
 * "why is there an entityType I never authored?" arrives.
 */

/**
 * Namespace for minted ids. A prefix rather than a suffix so a shadow is
 * recognizable at a glance in a redaction marker and in the IR, and `:` because
 * the extraction prompt constrains authored ids to lowercase kebab-case -- an
 * authored id cannot contain a colon, so the two spaces cannot overlap even
 * before the compiler's explicit collision check.
 */
export const SHADOW_PREFIX = "pred:";

export function shadowIdFor(predicateId: string): string {
  return `${SHADOW_PREFIX}${predicateId}`;
}
