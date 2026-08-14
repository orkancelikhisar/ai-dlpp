import { describe, expect, it } from "vitest";
import {
  decryptString, encryptString, exportVaultKey, generateVaultKey, importVaultKey,
} from "../../src/pseudo/crypto.js";

const IV_BYTES = 12;
const TAG_BYTES = 16;

const toBytes = (b64: string): Uint8Array => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

function toB64(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}

/** Flips the low bit of one packed byte, re-encoding the payload around it. */
function flipByte(payload: string, index: number): string {
  const bytes = toBytes(payload);
  bytes[index] = (bytes[index] ?? 0) ^ 0x01;
  return toB64(bytes);
}

describe("vault crypto (AES-GCM)", () => {
  it("round-trips a string", async () => {
    const key = await generateVaultKey();
    const payload = await encryptString(key, "Globex ⇄ Vantor");
    expect(await decryptString(key, payload)).toBe("Globex ⇄ Vantor");
  });

  it("uses a fresh IV per encryption (same plaintext, different ciphertext)", async () => {
    const key = await generateVaultKey();
    expect(await encryptString(key, "same")).not.toBe(await encryptString(key, "same"));
  });

  it("fails to decrypt with the wrong key", async () => {
    const k1 = await generateVaultKey();
    const k2 = await generateVaultKey();
    const payload = await encryptString(k1, "secret");
    await expect(decryptString(k2, payload)).rejects.toThrow();
  });

  it("export/import round-trips the key", async () => {
    const key = await generateVaultKey();
    const imported = await importVaultKey(await exportVaultKey(key));
    const payload = await encryptString(key, "hello");
    expect(await decryptString(imported, payload)).toBe("hello");
  });

  it("imports keys non-extractable (the base64 is the persistable form, not the handle)", async () => {
    const imported = await importVaultKey(await exportVaultKey(await generateVaultKey()));
    expect(imported.extractable).toBe(false);
    await expect(crypto.subtle.exportKey("raw", imported)).rejects.toThrow();
  });
});

/**
 * GCM's authentication is the whole reason this is AEAD and not raw AES: a
 * silent downgrade (unauthenticated mode, a stripped tag, a decrypt that
 * returns garbage instead of throwing) would leave at-rest records forgeable
 * without any test going red. These pin the rejection paths and the framing
 * they depend on.
 */
describe("vault crypto tamper-evidence and framing", () => {
  it("rejects a flipped ciphertext byte", async () => {
    const key = await generateVaultKey();
    const payload = await encryptString(key, "authenticate me");
    await expect(decryptString(key, flipByte(payload, IV_BYTES))).rejects.toThrow();
  });

  it("rejects a flipped IV byte", async () => {
    const key = await generateVaultKey();
    const payload = await encryptString(key, "authenticate me");
    await expect(decryptString(key, flipByte(payload, 0))).rejects.toThrow();
  });

  it("rejects a payload too short to hold an IV and a tag, with a legible error", async () => {
    const key = await generateVaultKey();
    const runt = toB64(new Uint8Array(IV_BYTES + TAG_BYTES - 1));
    await expect(decryptString(key, runt)).rejects.toThrow(/payload too short/);
  });

  it("frames the payload as iv ‖ ciphertext ‖ tag", async () => {
    const key = await generateVaultKey();
    const plaintext = "framing check ⇄ multi-byte";
    const utf8Len = new TextEncoder().encode(plaintext).length;
    const packed = toBytes(await encryptString(key, plaintext));
    expect(packed.length).toBe(IV_BYTES + utf8Len + TAG_BYTES);
  });
});

describe("vault crypto payload shapes", () => {
  it("round-trips the empty string", async () => {
    const key = await generateVaultKey();
    expect(await decryptString(key, await encryptString(key, ""))).toBe("");
  });

  it("round-trips multi-byte text (emoji, CJK, combining marks)", async () => {
    const key = await generateVaultKey();
    const tricky = "👨‍👩‍👧‍👦 family, 中文, ñ, ⇄, 𝔘𝔫𝔦𝔠𝔬𝔡𝔢, é";
    expect(await decryptString(key, await encryptString(key, tricky))).toBe(tricky);
  });

  it("round-trips a large payload without overflowing the stack", async () => {
    // The base64 helpers build their string one char at a time precisely so a
    // big payload cannot blow the call stack the way spread/apply would.
    const key = await generateVaultKey();
    const big = "ü".repeat(200_000); // ~400 KB of UTF-8
    expect(await decryptString(key, await encryptString(key, big))).toBe(big);
  });
});
