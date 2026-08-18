import type { PolicyIr } from "@sih/core";
import type { RejectedCandidate } from "./stages/extract.js";
import { SHADOW_PREFIX } from "./stages/predicates.js";
import type { SelfTestEntityReport, SelfTestReport } from "./stages/selftest.js";

/**
 * The markdown a human reads before shipping a compiled policy.
 *
 * This is the audit surface of the whole compiler, so it is written to be read
 * by someone looking for what went WRONG, not to look good:
 *
 * - Rejected candidates are printed with their reasons. A compile that silently
 *   dropped half the policy must look different on paper from one that dropped
 *   nothing, or the anti-hallucination gate is a mechanism nobody can check.
 * - An unmeasured number is printed as "not measured", never as 0%. The
 *   self-test stage leaves `recall` / `labelRecall` / `fpRate` undefined when
 *   there is no denominator (a skipped tier-1/2 entity, or no cases), and
 *   rendering that as 0% would put a false alarm next to every semantic entity
 *   on every compile — the exact failure the skip rule exists to prevent.
 * - `labelRecall` is printed beside `recall`. They differ exactly when a
 *   stricter overlapping entityType shadowed this one: the value is still
 *   caught, but this entity's label, surrogate and cited clause never appear.
 * - entityType ids get their own section, because they are the one part of the
 *   IR that travels OUTBOUND to the provider.
 *
 * DETERMINISM: no clock, no randomness, no locale-sensitive formatting. Task 9
 * commits a report next to each IR, and a report that changes when nothing
 * changed makes every recompile look like a policy edit.
 */

export interface ItemCounts {
  readonly entityTypes: number;
  readonly rules: number;
  readonly semanticPredicates: number;
  readonly actions: number;
}

export interface ReportInput {
  readonly policyName: string;
  /** The artifact being audited. Every derived count is read from it. */
  readonly ir: PolicyIr;
  /** What the model proposed, BEFORE the quote-grounding gate. */
  readonly proposed: ItemCounts;
  /**
   * What survived the gate. Reported alongside `proposed` rather than derived
   * from the IR: the IR's action maps are DEDUPED (one entry per entityType,
   * one per provider/entityType pair), so counting them would report clauses
   * that merged as clauses that were rejected — a rejection the gate never made.
   */
  readonly grounded: ItemCounts;
  /** Gate rejections plus anything `emit` dropped for a dangling reference. */
  readonly rejected: readonly RejectedCandidate[];
  readonly selfTest: SelfTestReport;
  /** Every warning the pipeline collected, in stage order. */
  readonly warnings: readonly string[];
}

/** `|` would split a table cell; nothing else in markdown breaks a one-line cell. */
function cell(text: string): string {
  return text.replace(/\|/g, "\\|");
}

function table(headers: readonly string[], rows: readonly (readonly string[])[]): string[] {
  return [
    `| ${headers.join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ];
}

/**
 * A fraction as a percentage with its raw counts, or the honest refusal.
 *
 * `undefined` means UNMEASURED. Printing 0% for it would be a claim the compiler
 * never made and cannot support.
 */
function rate(value: number | undefined, numerator: number, denominator: number): string {
  if (value === undefined) return "not measured";
  return `${(value * 100).toFixed(0)}% (${numerator}/${denominator})`;
}

function isShadow(id: string): boolean {
  return id.startsWith(SHADOW_PREFIX);
}

function clauseOf(ir: PolicyIr, id: string): string {
  return Object.hasOwn(ir.provenance, id) ? ir.provenance[id]!.clause : "—";
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

// -- sections ---------------------------------------------------------------

function stageSection(input: ReportInput): string[] {
  const { ir, proposed, grounded, selfTest } = input;
  const shadowIds = ir.entityTypes.filter((e) => isShadow(e.id));
  const authored = ir.entityTypes.length - shadowIds.length;
  const overridePairs = Object.values(ir.actions.providerOverrides ?? {}).reduce(
    (total, byEntity) => total + Object.keys(byEntity).length,
    0,
  );
  const providers = Object.keys(ir.actions.providerOverrides ?? {}).length;
  const defaults = Object.keys(ir.actions.default).length;
  const regexRules = ir.rules.filter((r) => r.regex !== undefined).length;
  const measured = selfTest.entities.filter((e) => !e.skipped).length;
  const skipped = selfTest.entities.length - measured;

  return [
    "## Stages",
    "",
    ...table(
      ["Stage", "Result"],
      [
        [
          "extract",
          `${grounded.entityTypes}/${proposed.entityTypes} entityTypes, ${grounded.rules}/${proposed.rules} rules, ` +
            `${grounded.semanticPredicates}/${proposed.semanticPredicates} semantic predicates and ` +
            `${grounded.actions}/${proposed.actions} action clauses quote the document verbatim`,
        ],
        [
          "ground",
          providers === 0
            ? `${grounded.actions} action clauses resolved to ${pluralize(defaults - shadowIds.length, "policy default")} ` +
              `and no provider override — no mention resolved to an adapter`
            : `${grounded.actions} action clauses resolved to ${pluralize(defaults - shadowIds.length, "policy default")} ` +
              `and ${pluralize(overridePairs, "override")} across ${pluralize(providers, "provider")}`,
        ],
        [
          "predicates",
          `${pluralize(shadowIds.length, "shadow entityType")} minted (tier 2, never pseudonymized)`,
        ],
        [
          "validate",
          `${pluralize(regexRules, "regex rule")} proven non-catastrophic in bounded time`,
        ],
        [
          "self-test",
          `${measured} of ${selfTest.entities.length} entityTypes measured against the real runtime, ` +
            `${skipped} not measurable today`,
        ],
        [
          "emit",
          `${pluralize(ir.entityTypes.length, "entityType")}, ${pluralize(ir.rules.length, "rule")}, ` +
            `${pluralize(defaults, "default action")}, ` +
            `${pluralize(Object.keys(ir.provenance).length, "provenance entry", "provenance entries")}`,
        ],
      ],
    ),
  ];
}

function entitySection(input: ReportInput): string[] {
  const { ir } = input;
  return [
    "## Entity types",
    "",
    ...table(
      ["id", "tier", "severity", "default action", "rules", "clause"],
      ir.entityTypes.map((entity) => [
        `\`${cell(entity.id)}\``,
        String(entity.tier),
        entity.severity,
        Object.hasOwn(ir.actions.default, entity.id) ? ir.actions.default[entity.id]! : "—",
        String(ir.rules.filter((r) => r.entityType === entity.id).length),
        cell(clauseOf(ir, entity.id)),
      ]),
    ),
  ];
}

/**
 * The section this report exists for as much as any other.
 *
 * An entityType id is not an internal name: it is transmitted to the provider on
 * every message that redacts one, inside `[REDACTED:<id>]`. An id derived from
 * the confidential noun it protects ("project-titan") therefore discloses the
 * very thing the redaction removed — the redaction succeeds and the leak
 * happens anyway. The compiler cannot know which nouns a firm treats as
 * confidential, so it prints the whole list for a human to read as text a third
 * party will see.
 */
function outboundSection(input: ReportInput): string[] {
  const { ir } = input;
  return [
    "## Outbound-visible identifiers",
    "",
    "Every id below is transmitted to the LLM provider inside a redaction marker —",
    "`[REDACTED:<id>]` — on every message where that entity is found. Read them as text a",
    "third party will see: an id derived from the confidential value it protects discloses",
    "that value even though the redaction worked.",
    "",
    ...table(
      ["id", "origin", "clause"],
      ir.entityTypes.map((entity) => [
        `\`[REDACTED:${cell(entity.id)}]\``,
        isShadow(entity.id)
          ? `minted for semantic predicate \`${cell(entity.id.slice(SHADOW_PREFIX.length))}\``
          : "authored in the policy",
        cell(clauseOf(ir, entity.id)),
      ]),
    ),
  ];
}

function rejectedSection(input: ReportInput): string[] {
  const { rejected } = input;
  const head = [
    "## Rejected candidates",
    "",
    "Candidates the model proposed that did not reach the IR. Every rule and entity in a",
    "compiled policy must quote the document verbatim; anything that cannot is discarded",
    "here rather than shipped. A compile that dropped half the policy must not read like one",
    "that dropped nothing.",
    "",
  ];
  if (rejected.length === 0) {
    return [
      ...head,
      "None — every candidate grounded in the document, and every rule and action reached an",
      "entityType that survived.",
    ];
  }
  return [
    ...head,
    ...table(
      ["kind", "id", "reason"],
      rejected.map((r) => [cell(r.kind), `\`${cell(r.id)}\``, cell(r.reason)]),
    ),
  ];
}

function coverageRow(entity: SelfTestEntityReport): string[] {
  return [
    `\`${cell(entity.entityType)}\``,
    String(entity.tier),
    entity.skipped ? "not measured" : String(entity.positives),
    rate(entity.recall, entity.caught + entity.shadowed, entity.positives),
    rate(entity.labelRecall, entity.caught, entity.positives),
    entity.skipped ? "not measured" : String(entity.negatives),
    rate(entity.fpRate, entity.falsePositives, entity.negatives),
  ];
}

function selfTestSection(input: ReportInput): string[] {
  const { selfTest } = input;
  const notes = selfTest.entities.flatMap((entity) => {
    if (entity.skipReason !== undefined) {
      return [`- \`${entity.entityType}\`: not measured — ${entity.skipReason}.`];
    }
    if (entity.recall === undefined && entity.fpRate === undefined) {
      return [`- \`${entity.entityType}\`: not measured — no cases were generated for it.`];
    }
    if (entity.shadowed > 0) {
      return [
        `- \`${entity.entityType}\`: ${entity.shadowed} of ${entity.positives} positives were caught ` +
          `under another entityType's label, which is why recall exceeds label recall. The value ` +
          `does not leak; this entity's surrogate and cited clause do not appear.`,
      ];
    }
    return [];
  });

  return [
    "## Self-test coverage",
    "",
    `Model-generated cases executed by the real tier-0 runtime (corpus tag \`${selfTest.corpusTag}\`).`,
    "**recall** counts a positive as caught when any entityType caught it — the leak-prevention",
    "number. **label recall** counts only the positives this entity's own rules labelled; it is",
    "lower exactly when a stricter overlapping entity won the span.",
    "",
    "**\"not measured\" is not zero.** It means no case was executed against that entity, so the",
    "compile makes no claim about it either way. The notes below the table say why.",
    "",
    ...table(
      [
        "entityType",
        "tier",
        "positives",
        "recall",
        "label recall",
        "hard negatives",
        "false-positive rate",
      ],
      selfTest.entities.map(coverageRow),
    ),
    ...(notes.length === 0 ? [] : ["", ...notes]),
  ];
}

function warningSection(input: ReportInput): string[] {
  const { warnings } = input;
  if (warnings.length === 0) {
    return ["## Warnings", "", "None."];
  }
  return [
    "## Warnings",
    "",
    `${pluralize(warnings.length, "warning")}. None of these failed the compile; each is a`,
    "judgement a human has to make.",
    "",
    ...warnings.map((w) => `- ${w}`),
  ];
}

function provenanceSection(input: ReportInput): string[] {
  const { ir } = input;
  const ids = Object.keys(ir.provenance);
  return [
    "## Provenance",
    "",
    "Every emitted item and the sentence of the policy that justifies it. A rule that cannot",
    "be traced to a clause is a rule the compiler invented.",
    "",
    ...ids.map((id) => `- \`${id}\` — ${ir.provenance[id]!.clause}: "${ir.provenance[id]!.quote}"`),
  ];
}

// -- entry point ------------------------------------------------------------

export function renderReport(input: ReportInput): string {
  const { ir, policyName } = input;
  const sections: string[][] = [
    [
      `# Compilation report: ${policyName}`,
      "",
      `- Source policy hash (sha256): \`${ir.policyHash}\``,
      `- IR version: \`${ir.irVersion}\``,
      `- Fail mode: \`${ir.failMode}\``,
      `- Latency budget: ${ir.latencyBudgetMs} ms`,
    ],
    stageSection(input),
    entitySection(input),
    outboundSection(input),
    rejectedSection(input),
    selfTestSection(input),
    warningSection(input),
    provenanceSection(input),
  ];
  return `${sections.map((lines) => lines.join("\n")).join("\n\n")}\n`;
}
