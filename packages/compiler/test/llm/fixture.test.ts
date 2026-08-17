import { describe, expect, it } from "vitest";
import { z } from "zod";
import { FixtureLlmClient, requestHash } from "../../src/llm/fixture.js";
import type { LlmRequest } from "../../src/llm/client.js";

const Shape = z.object({ answer: z.string() });
const req: LlmRequest = { system: "sys", user: "usr", schemaName: "Shape", maxTokens: 100 };

describe("requestHash", () => {
  it("is stable for identical requests", () => {
    expect(requestHash(req)).toBe(requestHash({ ...req }));
  });

  it("changes when any field changes", () => {
    const base = requestHash(req);
    // system is pinned explicitly: without this assertion, dropping `system`
    // from the hash entirely passes every other test in this file (the
    // boundary test varies `user`, and the FixtureLlmClient tests key their
    // maps with requestHash itself, so they stay self-consistent under any
    // hash). Two prompts differing only in system would then share a fixture.
    expect(requestHash({ ...req, system: "other" })).not.toBe(base);
    expect(requestHash({ ...req, user: "other" })).not.toBe(base);
    expect(requestHash({ ...req, schemaName: "Other" })).not.toBe(base);
    expect(requestHash({ ...req, maxTokens: 101 })).not.toBe(base);
  });

  it("does not collide across a field boundary", () => {
    // "ab"+"c" and "a"+"bc" must not hash alike — the separator is load-bearing.
    expect(requestHash({ ...req, system: "ab", user: "c" })).not.toBe(
      requestHash({ ...req, system: "a", user: "bc" }),
    );
  });
});

describe("FixtureLlmClient", () => {
  it("replays a recorded response and validates it against the schema", async () => {
    const client = new FixtureLlmClient(new Map([[requestHash(req), { answer: "42" }]]));
    expect(await client.complete(req, Shape)).toEqual({ answer: "42" });
  });

  it("throws a recordable error on a miss, naming the hash", async () => {
    const client = new FixtureLlmClient(new Map());
    await expect(client.complete(req, Shape)).rejects.toThrow(
      new RegExp(`no fixture.*${requestHash(req)}`, "i"),
    );
  });

  it("rejects a fixture that does not match the schema", async () => {
    const client = new FixtureLlmClient(new Map([[requestHash(req), { answer: 42 }]]));
    await expect(client.complete(req, Shape)).rejects.toThrow(/fixture.*schema/i);
  });

  it("names the hash in the schema-mismatch error too, not just on a miss", async () => {
    // Both error contracts are symmetric: whichever you hit, the hash tells you
    // which fixture file to fix.
    const client = new FixtureLlmClient(new Map([[requestHash(req), { answer: 42 }]]));
    await expect(client.complete(req, Shape)).rejects.toThrow(new RegExp(requestHash(req)));
  });
});
