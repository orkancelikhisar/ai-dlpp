import { describe, expect, it } from "vitest";
import {
  decryptString, encryptString, exportVaultKey, generateVaultKey, importVaultKey,
} from "../../src/pseudo/crypto.js";

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
});
