import { describe, expect, it } from "vitest";
import type { EntityType, PolicyIr } from "@sih/core";
import { buildLabels } from "../src/labels.js";
import {
  DEFAULT_TIER1_CONFIG,
  TIER1_LABEL_FORMS,
  type Tier1Config,
  type Tier1LabelForm,
} from "../src/config.js";

/**
 * Verbatim from packages/compiler/test/fixtures/llm: what the compiler actually
 * writes into an nlDefinition is a full sentence aimed at a frontier model, not
 * a noun phrase. Shortening it here would make the label-form tests compare two
 * things that are closer than the real ones.
 */
const CLIENT_NAME_DEFINITION =
  "The name of a client organisation — a company, fund or institution the firm does business with — as it appears in running prose.";

const entity = (over: Partial<EntityType>): EntityType => ({
  id: "client-name",
  tier: 1,
  nlDefinition: CLIENT_NAME_DEFINITION,
  examples: [],
  counterExamples: [],
  severity: "high",
  surrogateKind: "org-name",
  ...over,
});

const ir = (entityTypes: EntityType[]): PolicyIr => ({ entityTypes }) as PolicyIr;

const cfg = (labelForm: Tier1LabelForm = DEFAULT_TIER1_CONFIG.labelForm): Tier1Config => ({
  ...DEFAULT_TIER1_CONFIG,
  labelForm,
});

describe("buildLabels", () => {
  it("includes only tier-1 entityTypes", () => {
    // Tier 0 already caught these deterministically. Handing them to the model
    // duplicates every finding and spends latency to do it; tier 2's engine
    // does not exist here at all.
    const labels = buildLabels(
      ir([
        entity({ id: "client-name", tier: 1 }),
        entity({ id: "in-pan", tier: 0 }),
        entity({ id: "pred:x", tier: 2 }),
      ]),
      cfg(),
    );
    expect(labels.map((l) => l.entityType)).toEqual(["client-name"]);
  });

  it("maps class index to entityType id positionally", () => {
    const labels = buildLabels(
      ir([entity({ id: "a", tier: 1 }), entity({ id: "b", tier: 1 })]),
      cfg(),
    );
    expect(labels[0]!.classIndex).toBe(0);
    expect(labels[1]!.classIndex).toBe(1);
    expect(labels.map((l) => l.entityType)).toEqual(["a", "b"]);
  });

  it("numbers classes over the tier-1 subset, not over ir.entityTypes", () => {
    // The class index is the model's output axis. Numbering it over the
    // unfiltered array leaves a gap wherever a tier-0 entity sat, so every
    // class above the gap decodes to the wrong entityType.
    const labels = buildLabels(
      ir([
        entity({ id: "in-pan", tier: 0 }),
        entity({ id: "a", tier: 1 }),
        entity({ id: "b", tier: 1 }),
      ]),
      cfg(),
    );
    expect(labels.map((l) => l.classIndex)).toEqual([0, 1]);
  });

  it("prompts the model with a spaced phrase, never the kebab-case id", () => {
    // The id is an identifier written for the IR's own key space. The default
    // label form spaces it so the model is prompted with words; the id itself
    // stays as the KEY the finding is emitted under, so only the prompt moves.
    const labels = buildLabels(ir([entity({ id: "client-name" })]), cfg());
    expect(labels[0]!.prompt).toBe("client name");
    expect(labels[0]!.prompt).not.toContain("-");
  });

  it("returns an empty label set when the policy has no tier-1 entities", () => {
    // Not an error: a policy may legitimately declare none, and the tagger must
    // then skip inference entirely rather than call a model with zero classes.
    expect(buildLabels(ir([entity({ tier: 0 })]), cfg())).toEqual([]);
  });
});

describe("buildLabels label form", () => {
  it("passes the nlDefinition through verbatim in the definition form", () => {
    // spec 4.1 names `entityTypes[].{id, nlDefinition}` as the injected labels.
    // The default deviates from that, so the spec's own form has to stay
    // reachable as a config value rather than as a future rewrite of this file.
    const labels = buildLabels(ir([entity({})]), cfg("definition"));
    expect(labels[0]!.prompt).toBe(CLIENT_NAME_DEFINITION);
  });

  it("carries both halves in the id-and-definition form", () => {
    const labels = buildLabels(ir([entity({})]), cfg("id-and-definition"));
    expect(labels[0]!.prompt).toContain("client name");
    expect(labels[0]!.prompt).toContain(CLIENT_NAME_DEFINITION);
  });

  it("keys the finding on the id under every form, so the emit key never moves", () => {
    // The prompt is an experiment variable; the entityType is the join key into
    // ir.actions and into ground truth. If the form leaked into entityType, two
    // arms of the same experiment would emit findings that cannot be compared.
    for (const labelForm of TIER1_LABEL_FORMS) {
      const labels = buildLabels(ir([entity({ id: "client-name" })]), cfg(labelForm));
      expect(labels.map((l) => l.entityType), labelForm).toEqual(["client-name"]);
    }
  });

  it("renders a distinct prompt for every form, so a form is observable in what the model sees", () => {
    // Two forms that rendered the same text would make an arm comparing them
    // report a difference that cannot exist. Distinctness is what makes this
    // ladder rung measurable at all.
    //
    // There is deliberately no test here that buildLabels' default form matches
    // DEFAULT_TIER1_CONFIG: `config` is a required parameter, so there is no
    // default to drift. That guarantee is now a compile error, not an assertion.
    const policy = ir([entity({})]);
    const prompts = TIER1_LABEL_FORMS.map(
      (labelForm) => buildLabels(policy, cfg(labelForm))[0]!.prompt,
    );
    expect(new Set(prompts).size).toBe(TIER1_LABEL_FORMS.length);
  });
});
