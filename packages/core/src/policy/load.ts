import { PolicyIrSchema } from "./schema.js";
import type { PolicyIr } from "./types.js";
import { hasValidator } from "../detect/validators.js";

export class PolicyLoadError extends Error {}
export class PolicyVersionError extends PolicyLoadError {}

export const SUPPORTED_IR_VERSION = "1";

export function loadPolicyIr(jsonText: string): PolicyIr {
  let raw: unknown;
  try {
    raw = JSON.parse(jsonText);
  } catch (e) {
    throw new PolicyLoadError(`IR is not valid JSON: ${(e as Error).message}`);
  }

  const version = (raw as { irVersion?: unknown })?.irVersion;
  if (version !== SUPPORTED_IR_VERSION) {
    throw new PolicyVersionError(
      `unsupported irVersion ${JSON.stringify(version)}; this runtime supports "${SUPPORTED_IR_VERSION}"`,
    );
  }

  const parsed = PolicyIrSchema.safeParse(raw);
  if (!parsed.success) {
    throw new PolicyLoadError(`IR failed validation: ${parsed.error.message}`);
  }
  const ir = parsed.data as PolicyIr;

  for (const rule of ir.rules) {
    if (rule.regex !== undefined) {
      try {
        // "u" alone breaks common IR escapes like \- ; plain compile matches tier-0 usage.
        new RegExp(rule.regex, "g");
      } catch (e) {
        throw new PolicyLoadError(`rule "${rule.id}" has invalid regex: ${(e as Error).message}`);
      }
    }
    if (rule.validator !== undefined && !hasValidator(rule.validator)) {
      throw new PolicyLoadError(`rule "${rule.id}" names unknown validator "${rule.validator}"`);
    }
  }
  return ir;
}
