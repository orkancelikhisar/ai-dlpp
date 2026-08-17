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
});
