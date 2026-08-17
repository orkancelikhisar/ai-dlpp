import { describe, expect, it } from "vitest";
import {
  applyActions,
  createRehydrateTransform,
  detect,
  loadPolicyIr,
  MemoryVaultStore,
  rehydrateText,
  Vault,
  type Finding,
  type SpanTagger,
} from "../../src/index.js";
import * as publicApi from "../../src/index.js";
import { minimalIr } from "../fixtures/minimal-ir.js";

const ir = loadPolicyIr(JSON.stringify(minimalIr()));

/** Any fixed string; the vault only requires it to be non-empty (Task 4, D1). */
const SALT = "e2e-install-salt";
const newVault = () => new Vault(new MemoryVaultStore(), SALT);

/** Stub tier-1 tagger that flags every "Globex" occurrence as client-name. */
const globexTagger: SpanTagger = {
  async tag(segments, taggedIr): Promise<Finding[]> {
    const findings: Finding[] = [];
    for (const seg of segments) {
      for (let idx = seg.text.indexOf("Globex"); idx !== -1; idx = seg.text.indexOf("Globex", idx + 1)) {
        findings.push({
          start: seg.start + idx,
          end: seg.start + idx + 6,
          text: "Globex",
          entityType: "client-name",
          severity: taggedIr.entityTypes.find((e) => e.id === "client-name")!.severity,
          tier: 1,
          source: "stub-tagger",
          confidence: 0.9,
        });
      }
    }
    return findings;
  },
};

const t0t1 = { tier0: true, tier1: true, tier2: false };

describe("end-to-end: detect → apply → rehydrate", () => {
  it("round-trips a pseudonymized conversation through a streamed response", async () => {
    const vault = newVault();
    const text = "Draft an email to Globex about the renewal. Globex prefers Q3.";

    const { findings } = await detect({
      ir,
      provider: "chatgpt",
      text,
      config: t0t1,
      engines: { tier1: globexTagger },
    });
    expect(findings).toHaveLength(2);
    expect(findings.every((f) => f.action === "pseudonymize")).toBe(true);

    const outbound = await applyActions(text, findings, vault, "conv-e2e", ir);
    expect(outbound.blocked).toBe(false);
    expect(outbound.text).not.toContain("Globex");
    const surrogate = outbound.applied[0]!.replacement;
    expect(outbound.applied[1]!.replacement).toBe(surrogate); // referential integrity

    // Simulate the provider echoing the surrogate in a streamed response,
    // split mid-surrogate to exercise the holdback.
    const response = `Sure — here's the email to ${surrogate} about their renewal.`;
    const cut = response.indexOf(surrogate) + Math.ceil(surrogate.length / 2);
    const transform = createRehydrateTransform(await vault.rehydrationMap("conv-e2e"));
    const writer = transform.writable.getWriter();
    const reader = transform.readable.getReader();
    const parts: string[] = [];
    const readAll = (async () => {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
      }
    })();
    await writer.write(response.slice(0, cut));
    await writer.write(response.slice(cut));
    await writer.close();
    await readAll;

    expect(parts.join("")).toBe(`Sure — here's the email to Globex about their renewal.`);
  });

  it("provider override changes the outbound treatment end-to-end", async () => {
    const vault = newVault();
    const text = "Summarize the Globex contract.";
    const { findings } = await detect({
      ir,
      provider: "deepseek",
      text,
      config: t0t1,
      engines: { tier1: globexTagger },
    });
    expect(findings[0]!.action).toBe("redact"); // deepseek override in the fixture
    const outbound = await applyActions(text, findings, vault, "conv-ds", ir);
    expect(outbound.text).toBe("Summarize the [REDACTED:client-name] contract.");
    // Redaction is one-way by design: nothing was minted, so the conversation
    // has no rehydration map and the response comes back as the model wrote it.
    // Only pseudonymization buys the round trip.
    expect(await vault.rehydrationMap("conv-ds")).toEqual(new Map());
  });

  // The non-streaming path is what a non-SSE adapter (a JSON response, a
  // clipboard rehydrate) uses, and the plan's round-trip only exercises the
  // TransformStream.
  it("rehydrates a whole response in one pass", async () => {
    const vault = newVault();
    const text = "Globex renewed.";
    const { findings } = await detect({
      ir, provider: "chatgpt", text, config: t0t1, engines: { tier1: globexTagger },
    });
    const outbound = await applyActions(text, findings, vault, "conv-batch", ir);
    const surrogate = outbound.applied[0]!.replacement;
    const map = await vault.rehydrationMap("conv-batch");
    expect(rehydrateText(`I see ${surrogate} renewed; ${surrogate} is up to date.`, map)).toBe(
      "I see Globex renewed; Globex is up to date.",
    );
  });

  // What Plan 6's review sheet consumes: WHAT is blocking the send, and WHERE
  // it sits in the text the user is being shown (which is the rewritten text,
  // not the one they typed).
  it("locates a blocked span in the rewritten text", async () => {
    const vault = newVault();
    const text = "Globex sent PAN ABCPD1234E for the tax form.";
    const { findings } = await detect({
      ir, provider: "chatgpt", text, config: t0t1, engines: { tier1: globexTagger },
    });
    expect(findings.map((f) => [f.entityType, f.action])).toEqual([
      ["client-name", "pseudonymize"],
      ["in-pan", "block"],
    ]);

    const outbound = await applyActions(text, findings, vault, "conv-block", ir);
    expect(outbound.blocked).toBe(true);
    expect(outbound.skipped).toHaveLength(1);
    const pan = outbound.skipped[0]!;
    expect(pan.entityType).toBe("in-pan");
    expect(pan.action).toBe("block");
    // Blocking does not rewrite, and the offsets locate it after the
    // pseudonymization ahead of it moved everything to its right.
    expect(outbound.text.slice(pan.newStart, pan.newEnd)).toBe("ABCPD1234E");
    const shift = outbound.applied[0]!.replacement.length - "Globex".length;
    expect(pan.newStart).toBe(pan.start + shift);
  });
});

describe("public API surface", () => {
  it("exports the pseudo layer", () => {
    // Types are erased, so the type exports this file imports are pinned by
    // typecheck rather than here; these are the runtime values.
    for (const name of [
      "fnv1a64", "mulberry32", "seededRng",
      "generateSurrogate",
      "Vault", "MemoryVaultStore",
      "generateVaultKey", "exportVaultKey", "importVaultKey", "encryptString", "decryptString",
      "applyActions",
      "rehydrateText", "createRehydrateTransform", "maxSurrogateLength",
    ]) {
      expect(publicApi).toHaveProperty(name);
    }
  });

  // Deliberate non-export, pinned so it is not "helpfully" re-added: the
  // compiled alternation is an implementation detail of this layer's matching
  // rules (longest-first ordering, the digit boundary), and every one of those
  // is a decision rehydrateText/createRehydrateTransform own. A caller holding
  // the RegExp can run its own replacement pass against a stateful `g`-flagged
  // instance and quietly diverge from both.
  it("does not export surrogatePattern", () => {
    expect(publicApi).not.toHaveProperty("surrogatePattern");
  });
});
