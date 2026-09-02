/**
 * One pool of organisation names, and the roles a clause can put one in.
 *
 * ## The defect this file exists to remove
 *
 * MEASURED on `corpora/generated/injection-p-fin-adjudicated-v1.jsonl` by
 * `measureRoles`: 19 of its 39 organisation spans carry a name that appears in
 * only ONE role class across the whole corpus. The denominator is 39 rather than
 * 52 because that corpus recorded `constructedRole: "none"` on its competitor,
 * listed-company and employer families even though their clauses assign a role;
 * counting those too the figure is 32 of 52. Both numbers were measured; the
 * first is the one the manifest's own function produces. "Kestrel Ironworks" is only ever a client;
 * "Cobblestone Print Works" is only ever a supplier; "Fenwold Advisory" is only
 * ever a competitor; the Firm's own name is only ever the employer. So the
 * client-vs-non-client question -- the one question that requires reading a
 * clause rather than a string, and the one a 2B model failed in run 03 by
 * putting a byte-perfect span on an organisation name and typing it `in-pan` --
 * was answerable from the name alone on 62% of the spans.
 *
 * `ORG_POOL` is the fix and it is the whole fix: ONE pool, drawn by every
 * organisation family whatever role it writes. Nothing in "Marrowfield Group"
 * says whether the Firm sells to it, buys from it, competes with it or leases
 * a floor from it, and `corpus-orgs.test.ts` asserts that every family that
 * mints an organisation mints from this array and no other.
 *
 * ## The Firm is unnamed in wave 3, and that is a deliberate loss
 *
 * Wave 1 and wave 2 carry `FIRM = "Harborline Capital Advisors"` and a
 * `neg:own-employer-org` family built on it. That family cannot draw from a
 * shared pool: an employer is the one role that is fixed for the whole corpus,
 * so its name is role-locked by construction and no clause can unlock it. It is
 * removed rather than kept with a caveat, and the cost is stated: wave 3 has no
 * item that asks "is the Firm's own name a client name", which is a question
 * worth asking and which this corpus can no longer ask. The reserved-TLD hosts
 * in `surfaces.ts` moved off the Firm's domain for the same reason.
 *
 * ## Fiction, and exactly how far it was checked
 *
 * Every name below was carried over from `universe.candidate.ts`, where each
 * was web-searched as a quoted string on 2026-09-02 and returned no exact
 * match. NOTHING in this file was re-searched this session, because this
 * session has no network. That is a fact about this round, not a claim about
 * the names: they are as checked as they were, and no more.
 */

/**
 * The organisation pool. Six names, deliberately flavour-neutral: no "Print
 * Works", no "Conference Centre", no "Advisory", nothing that answers the role
 * question before the clause does.
 *
 * Five rather than fifteen, and the arithmetic is the reason. The requirement
 * is that every name appear in BOTH role classes in the emitted file, not
 * merely that the code permits it, and the scarce side is the client side: the
 * corpus draws about twenty client-side organisation spans against about forty
 * non-client ones. At six names a six-name pool left one organisation
 * ("Marrowfield Group") drawn as a non-client nine times and as a client never,
 * MEASURED on a full build; at five the expected count per name per side is
 * about four. The emitted per-name coverage is in the manifest's
 * `leakage.roles` either way, so the claim is a measurement rather than a
 * property of the pool size.
 */
export const ORG_POOL = [
  "Marrowfield Group",
  "Ashcombe Holdings",
  "Tarnwick Industries",
  "Lensfield Trading",
  "Bexmoor Associates",
] as const;

/**
 * The roles a clause can assign. The first four are the ones p-fin §3.1 and
 * §3.3 reach -- "client organisation names and counterparty organisation
 * names", extended by §3.3 to "prospective clients, deal counterparties, and
 * parties under a non-disclosure agreement". The rest are roles §3 does not
 * reach at all.
 */
export const CLIENT_SIDE_ROLES = ["client", "prospect", "counterparty", "nda-party"] as const;
export const NON_CLIENT_ROLES = ["vendor", "landlord", "competitor", "listed-company"] as const;
export type ClientSideRole = (typeof CLIENT_SIDE_ROLES)[number];
export type NonClientRole = (typeof NON_CLIENT_ROLES)[number];
export type OrgRole = ClientSideRole | NonClientRole;

export function roleClass(role: OrgRole): "client-side" | "non-client" {
  return (CLIENT_SIDE_ROLES as readonly string[]).includes(role) ? "client-side" : "non-client";
}

export function pickOrg(rng: () => number): string {
  return ORG_POOL[Math.floor(rng() * ORG_POOL.length)]!;
}

/**
 * The name after `value` in the pool, wrapping.
 *
 * Used by the cross-segment families, which name TWO organisations in one
 * clause and assign the roles by ordinal ("the first of those two ..."). It is
 * derived from the value rather than drawn from the rng because the second name
 * has to be guaranteed different from the first: a redraw can repeat, and a
 * clause naming the same organisation twice as "those two" is nonsense that no
 * invariant in this pipeline would catch.
 */
export function nextOrg(value: string): string {
  const i = (ORG_POOL as readonly string[]).indexOf(value);
  if (i < 0) throw new Error(`${JSON.stringify(value)} is not in ORG_POOL`);
  return ORG_POOL[(i + 1) % ORG_POOL.length]!;
}

/**
 * Colleagues, carried over from `universe.candidate.ts`.
 *
 * They are used only in a colleague role, never as a customer, and that is what
 * keeps the `neg:` label defensible: p-fin §1.2 scopes customer data to a
 * person or entity HOLDING AN ACCOUNT with the Firm, and the compiled IR has no
 * personal-name entityType to read an obligation from either. Both readings
 * agree. The names are fictional in the sense that no entry is intended to
 * denote anyone, not in the sense that nobody is called this.
 */
export const PERSON_NAMES = ["Meera Talwatkar", "Arjun Beheria", "Sunita Marlekar"] as const;

/** Software the Firm runs. An ORG tagger tags these; p-fin demands nothing of them. */
export const PRODUCT_NAMES = ["Quillrun", "Wickerpost"] as const;

export function pickFrom(rng: () => number, xs: readonly string[]): string {
  return xs[Math.floor(rng() * xs.length)]!;
}
