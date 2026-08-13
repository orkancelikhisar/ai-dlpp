import { z } from "zod";
import { PolicyIrSchema } from "./schema.js";
import type { PolicyIr } from "./types.js";
import { hasValidator } from "../detect/validators.js";

// `instanceof` does not survive structuredClone or cross-context messaging; the name
// string is what shows up in logs, so both classes carry an explicit discriminant.
export class PolicyLoadError extends Error {
  override readonly name: string = "PolicyLoadError";
}
export class PolicyVersionError extends PolicyLoadError {
  override readonly name = "PolicyVersionError";
}

export const SUPPORTED_IR_VERSION = "1";

export function loadPolicyIr(jsonText: string): PolicyIr {
  let raw: unknown;
  try {
    raw = JSON.parse(jsonText);
  } catch (e) {
    throw new PolicyLoadError(`IR is not valid JSON: ${(e as Error).message}`);
  }

  // Input that carries no version is not an IR at all. Classifying it as a version
  // error would tell the caller to recompile their policy, which is bad advice.
  if (typeof raw !== "object" || raw === null || Array.isArray(raw) || !Object.hasOwn(raw, "irVersion")) {
    throw new PolicyLoadError("not a policy IR (missing irVersion)");
  }

  const version = (raw as { irVersion: unknown }).irVersion;
  if (version !== SUPPORTED_IR_VERSION) {
    throw new PolicyVersionError(
      `unsupported irVersion ${JSON.stringify(version)}; this runtime supports "${SUPPORTED_IR_VERSION}"`,
    );
  }

  const parsed = PolicyIrSchema.safeParse(raw);
  if (!parsed.success) {
    throw new PolicyLoadError(`IR failed validation: ${z.prettifyError(parsed.error)}`);
  }
  // Sound only because the version gate above already proved irVersion === "1", which
  // is the single field the schema types more loosely than PolicyIr. Do not reorder.
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
