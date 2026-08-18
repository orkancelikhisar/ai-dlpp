/**
 * AES-GCM helpers for vault-at-rest encryption (spec §5.4). WebCrypto only —
 * `crypto.subtle` and `crypto.getRandomValues` are WHATWG globals in browsers
 * and Node 20+; no node: imports (firewall). Plan 6's IndexedDB store encrypts
 * VaultRecord values with these; the session key lives in chrome.storage.session
 * there. Encryption-at-rest is hygiene, not the core security property.
 */

/**
 * `CryptoKey` is a global *type* only under the DOM lib; @types/node keeps it
 * inside the `webcrypto` namespace, and importing that would mean `node:crypto`
 * (firewall). Deriving it from the global `crypto` value works in both worlds
 * and tracks whichever lib is in scope, so the signatures below stay honest in
 * the browser and in Node without either one being hardcoded.
 */
export type CryptoKey = Parameters<typeof crypto.subtle.encrypt>[1];

const IV_BYTES = 12;
/** AES-GCM's default authentication tag, appended to the ciphertext by WebCrypto. */
const TAG_BYTES = 16;

function bytesToBase64(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}

/**
 * Returns `Uint8Array<ArrayBuffer>`, not bare `Uint8Array`. Since TS 5.7 the bare
 * name means `Uint8Array<ArrayBufferLike>`, which admits `SharedArrayBuffer` and
 * is therefore NOT assignable to the DOM lib's `BufferSource` — so importVaultKey
 * below fails to compile in any program that loads lib.dom, i.e. every program
 * that actually runs this file. Core's own tsconfig is `lib: ["ES2022"]`, where
 * @types/node's looser WebCrypto signatures accept it and the error never
 * appears; the eval harness (apps/eval) is the first DOM-lib consumer to see it.
 * The body already only ever builds a plain ArrayBuffer, so this narrows the
 * declaration to the truth rather than changing behaviour.
 */
function base64ToBytes(b64: string): Uint8Array<ArrayBuffer> {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function generateVaultKey(): Promise<CryptoKey> {
  return crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt", "decrypt"]);
}

export async function exportVaultKey(key: CryptoKey): Promise<string> {
  return bytesToBase64(new Uint8Array(await crypto.subtle.exportKey("raw", key)));
}

/**
 * Imported non-extractable: nothing re-exports an imported key (the base64 the
 * caller already holds *is* the persistable form), so the handle Plan 6 passes
 * around cannot be turned back into key bytes. Narrows the surface; it is not a
 * containment boundary — whoever holds the base64 can import an extractable one.
 */
export async function importVaultKey(b64: string): Promise<CryptoKey> {
  return crypto.subtle.importKey("raw", base64ToBytes(b64), { name: "AES-GCM" }, false, ["encrypt", "decrypt"]);
}

/** Returns base64(iv ‖ ciphertext); IV is random per call — never reuse under GCM. */
export async function encryptString(key: CryptoKey, plaintext: string): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const ct = new Uint8Array(
    await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, new TextEncoder().encode(plaintext)),
  );
  const packed = new Uint8Array(iv.length + ct.length);
  packed.set(iv, 0);
  packed.set(ct, iv.length);
  return bytesToBase64(packed);
}

export async function decryptString(key: CryptoKey, payload: string): Promise<string> {
  const packed = base64ToBytes(payload);
  // Below this length the payload cannot even hold an IV and a tag, so slicing
  // would hand subtle.decrypt a nonsense IV and surface as a generic
  // OperationError indistinguishable from a wrong key or real tampering.
  if (packed.length < IV_BYTES + TAG_BYTES) throw new Error("payload too short");
  const iv = packed.slice(0, IV_BYTES);
  const ct = packed.slice(IV_BYTES);
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
  return new TextDecoder().decode(pt);
}
