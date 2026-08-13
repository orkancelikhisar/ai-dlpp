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

/**
 * Rejects a literal own "__proto__" key anywhere in the parsed IR.
 *
 * This cannot be a schema refinement: JSON.parse creates "__proto__" as an own key, but
 * zod's record parse copies keys onto a fresh object, and that assignment fires the
 * Object.prototype `__proto__` setter instead of creating a key — so the key is already
 * gone by the time superRefine runs. For entityTypes the loss fails closed (the entity
 * then has no action mapping and validation rejects the IR), but a dropped
 * providerOverrides["__proto__"] is silent: that provider's overrides simply disappear
 * and resolution falls back to the weaker defaults — a policy downgrade in a
 * hash-stamped security artifact. Refuse the input instead.
 *
 * JSON.parse output is a tree, never cyclic, so plain recursion terminates.
 */
function assertNoProtoKeys(value: unknown, path: string): void {
  if (typeof value !== "object" || value === null) return;
  if (Array.isArray(value)) {
    value.forEach((item, i) => assertNoProtoKeys(item, `${path}[${i}]`));
    return;
  }
  // getOwnPropertyNames sees the own "__proto__" that JSON.parse created; `in` and
  // prototype-chain reads would not distinguish it from the inherited accessor.
  for (const key of Object.getOwnPropertyNames(value)) {
    if (key === "__proto__") {
      throw new PolicyLoadError(`forbidden key "__proto__" at ${path}`);
    }
    assertNoProtoKeys((value as Record<string, unknown>)[key], `${path}.${key}`);
  }
}

export function loadPolicyIr(jsonText: string): PolicyIr {
  let raw: unknown;
  try {
    raw = JSON.parse(jsonText);
  } catch (e) {
    throw new PolicyLoadError(`IR is not valid JSON: ${(e as Error).message}`);
  }

  // Before any structural interpretation: zod would drop such keys, so this is the last
  // point at which they are still observable.
  assertNoProtoKeys(raw, "$");

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
      // A nullable regex (one that can match "") cannot drive a scan: /g/ exec returns an
      // empty match without advancing lastIndex, so detection must skip it, and a rule
      // whose pattern is satisfied by nothing at all is malformed rather than merely
      // noisy. Rejecting here means a policy fails to load instead of loading and then
      // quietly detecting nothing — the fail-open this exists to prevent.
      //
      // Necessary but NOT sufficient, deliberately: test("") probes offset 0 only, where
      // a lookbehind like "(?<=:)\w*" cannot succeed, so that rule loads and still matches
      // empty mid-string. runTier0 keeps its own zero-width guard for exactly that gap.
      // Non-global regex on purpose — test() on a /g/ regex mutates lastIndex.
      if (new RegExp(rule.regex).test("")) {
        throw new PolicyLoadError(`rule "${rule.id}" regex can match the empty string`);
      }
    }
    if (rule.validator !== undefined && !hasValidator(rule.validator)) {
      throw new PolicyLoadError(`rule "${rule.id}" names unknown validator "${rule.validator}"`);
    }
  }
  return ir;
}
