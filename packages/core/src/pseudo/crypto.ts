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

function bytesToBase64(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}

function base64ToBytes(b64: string): Uint8Array {
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

export async function importVaultKey(b64: string): Promise<CryptoKey> {
  return crypto.subtle.importKey("raw", base64ToBytes(b64), { name: "AES-GCM" }, true, ["encrypt", "decrypt"]);
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
  const iv = packed.slice(0, IV_BYTES);
  const ct = packed.slice(IV_BYTES);
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
  return new TextDecoder().decode(pt);
}
