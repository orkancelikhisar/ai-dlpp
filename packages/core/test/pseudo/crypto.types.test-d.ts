/**
 * Type-level pins for the `CryptoKey` alias in src/pseudo/crypto.ts.
 *
 * The alias is `Parameters<typeof crypto.subtle.encrypt>[1]` because the name
 * `CryptoKey` is global only under the DOM lib and @types/node namespaces it
 * (see the Deviations log). That derivation is load-bearing but fragile in one
 * specific way: if a future lib version gives `subtle.encrypt` a second
 * overload, `Parameters<>` silently switches to the last one and `[1]` could
 * become some other parameter — or `any` — retargeting every signature in the
 * module without a single runtime test going red. These assertions fail the
 * build if that happens.
 *
 * Checked by `tsc --noEmit` (tsconfig includes `test`); the `.test-d.ts` suffix
 * keeps vitest from collecting it as a runtime suite.
 */
import type { CryptoKey } from "../../src/pseudo/crypto.js";

/** True only for `any`, which absorbs both branches of the conditional. */
type IsAny<T> = 0 extends 1 & T ? true : false;

// 1. Not `any` — an `any` alias would make every signature vacuous.
export const notAny: IsAny<CryptoKey> = false;

// 2. Structurally the real WebCrypto key, not a widened stand-in.
declare const key: CryptoKey;
export const algorithm: { name: string } = key.algorithm;
export const extractable: boolean = key.extractable;
export const keyType: string = key.type;
export const usages: readonly string[] = key.usages;

// 3. Interchangeable with what the subtle API itself produces, both directions.
declare const fromSubtle: Awaited<ReturnType<typeof crypto.subtle.importKey>>;
export const acceptsSubtleKey: CryptoKey = fromSubtle;
export const subtleAcceptsAlias: typeof fromSubtle = key;

// 4. Still nominal enough to reject a plain object. If the alias ever widens,
//    this @ts-expect-error goes unused and tsc fails on it.
// @ts-expect-error - an arbitrary object is not a CryptoKey
export const rejectsPlainObject: CryptoKey = { algorithm: { name: "AES-GCM" } };
