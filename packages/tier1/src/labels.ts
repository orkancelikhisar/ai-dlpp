import type { EntityType, PolicyIr } from "@sih/core";
import { DEFAULT_TIER1_CONFIG, type Tier1Config, type Tier1LabelForm } from "./config.js";

export interface Tier1Label {
  /** Position in the label array handed to the model; the model's class index. */
  readonly classIndex: number;
  /** The IR id this class maps back to. Emitted on the Finding; must exist in ir.entityTypes. */
  readonly entityType: string;
  /** What the model is actually prompted with. Shape chosen by `Tier1Config.labelForm`. */
  readonly prompt: string;
}

function promptFor(entity: EntityType, labelForm: Tier1LabelForm): string {
  // Hyphens only: the ids this repo's compiler emits are kebab-case
  // (`client-name`, `in-pan`), and a broader normaliser would silently rewrite
  // an id shape nobody has produced yet.
  const spacedId = entity.id.replace(/-/g, " ");
  switch (labelForm) {
    case "id":
      return spacedId;
    case "definition":
      return entity.nlDefinition;
    case "id-and-definition":
      return `${spacedId}: ${entity.nlDefinition}`;
  }
}

/**
 * Derive the model's label set from the compiled policy.
 *
 * This function IS the policy-adaptivity mechanism named in spec §4.1: a new
 * policy changes what the model looks for with no retraining and no code
 * change, because the labels are read out of `ir.entityTypes` at inference
 * time.
 *
 * Only tier-1 entityTypes get a class. Tier 0 is already resolved by regex and
 * validators before this runs, and tier 2's engine is a different model
 * entirely -- including either would spend a class on a decision that is not
 * this model's to make.
 *
 * `classIndex` is assigned over the FILTERED list, because it addresses the
 * model's output axis rather than a position in the IR. It is the only thing
 * carrying a model output back to an `entityType`, so nothing downstream may
 * re-derive it from `ir.entityTypes` directly.
 *
 * The prompt text is `config.labelForm`, not a constant: which form the model
 * does better with is UNVERIFIED here and is meant to be a rung of the ladder.
 * `Tier1LabelForm` carries the decision, the measurement behind it, and the
 * experiment that would settle it.
 */
export function buildLabels(
  ir: PolicyIr,
  config: Tier1Config = DEFAULT_TIER1_CONFIG,
): Tier1Label[] {
  return ir.entityTypes
    .filter((entity) => entity.tier === 1)
    .map((entity, classIndex) => ({
      classIndex,
      entityType: entity.id,
      prompt: promptFor(entity, config.labelForm),
    }));
}
