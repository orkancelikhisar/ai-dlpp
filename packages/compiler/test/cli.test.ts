import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicyIr } from "@sih/core";
import { runCli, type CliDeps } from "../src/cli.js";
import { FixtureLlmClient } from "../src/llm/fixture.js";
import { loadTestFixtures } from "./fixtures/index.js";

const POLICIES = join(import.meta.dirname, "..", "..", "..", "policies");

describe("runCli", () => {
  it("writes an IR and a report, exiting 0", async () => {
    const out = mkdtempSync(join(tmpdir(), "sih-cli-"));
    const code = await runCli(
      [
        "--policy",
        join(POLICIES, "p-fin.md"),
        "--providers",
        join(POLICIES, "providers.json"),
        "--out",
        out,
        "--name",
        "p-fin",
      ],
      { fixtures: loadTestFixtures() },
    );
    expect(code).toBe(0);
    const ir = readFileSync(join(out, "p-fin.ir.json"), "utf8");
    expect(() => loadPolicyIr(ir)).not.toThrow();
    expect(readFileSync(join(out, "p-fin.report.md"), "utf8")).toMatch(/# Compilation report/i);
  });

  it("exits 1 with a legible message on a missing policy file", async () => {
    const code = await runCli(
      [
        "--policy",
        "/nope.md",
        "--providers",
        join(POLICIES, "providers.json"),
        "--out",
        "/tmp",
        "--name",
        "x",
      ],
      {
        fixtures: loadTestFixtures(),
      },
    );
    expect(code).toBe(1);
  });

  it("refuses to run live without an explicit --live flag", async () => {
    // Guards against a test or CI run silently spending money.
    const code = await runCli(
      [
        "--policy",
        join(POLICIES, "p-fin.md"),
        "--providers",
        join(POLICIES, "providers.json"),
        "--out",
        "/tmp",
        "--name",
        "x",
      ],
      {},
    );
    expect(code).toBe(1);
  });
});

/**
 * Appended beyond the plan's three, each closing a hole a mutation proved real.
 * The plan's three pass unchanged against: an implementation that never writes
 * the self-test corpus, one that hardcodes the output basename, one that never
 * prints a warning, and — most seriously — one that goes LIVE by default.
 */
describe("runCli (holes the plan's three tests leave open)", () => {
  const args = (out: string, name = "p-fin") => [
    "--policy",
    join(POLICIES, "p-fin.md"),
    "--providers",
    join(POLICIES, "providers.json"),
    "--out",
    out,
    "--name",
    name,
  ];
  const tmp = () => mkdtempSync(join(tmpdir(), "sih-cli-"));

  /** Captures both streams so printing can be asserted rather than assumed. */
  function recorder(): CliDeps & { out: string[]; err: string[] } {
    const out: string[] = [];
    const err: string[] = [];
    return { out, err, write: (l) => out.push(l), writeError: (l) => err.push(l) };
  }

  it("never builds the live client when --live is absent", async () => {
    // The plan's third test asserts only the exit code, and an implementation
    // that falls through to the live client STILL exits 1 here — because
    // constructing it throws "could not resolve authentication method" when no
    // credential is present. That makes the guard a property of the machine
    // rather than of the code: on a developer box with ANTHROPIC_API_KEY set,
    // the same bug bills a real request. Pinning the factory instead makes the
    // guard hold in every environment.
    let built = 0;
    const code = await runCli(args("/tmp"), {
      liveClient: () => {
        built += 1;
        throw new Error("SPENT MONEY");
      },
    });
    expect(code).toBe(1);
    expect(built).toBe(0);
  });

  it("refuses --live alongside fixtures rather than silently preferring one", async () => {
    let built = 0;
    const code = await runCli([...args("/tmp"), "--live"], {
      fixtures: loadTestFixtures(),
      liveClient: () => {
        built += 1;
        throw new Error("SPENT MONEY");
      },
    });
    expect(code).toBe(1);
    expect(built).toBe(0);
  });

  it("routes --live to the live client seam", async () => {
    // The counterpart to the two refusals: without this, a --live flag that was
    // wired to nothing would look exactly like a correct implementation. The
    // stub is a FixtureLlmClient, so this exercises the routing and not a socket.
    const out = tmp();
    let built = 0;
    const code = await runCli([...args(out), "--live"], {
      liveClient: () => {
        built += 1;
        return new FixtureLlmClient(loadTestFixtures());
      },
    });
    expect(code).toBe(0);
    expect(built).toBe(1);
  });

  it("writes the self-test corpus beside the IR and the report", async () => {
    const out = tmp();
    expect(await runCli(args(out), { fixtures: loadTestFixtures() })).toBe(0);
    const cases: unknown = JSON.parse(readFileSync(join(out, "p-fin.selftest.json"), "utf8"));
    expect(Array.isArray(cases)).toBe(true);
    // The corpus is the evidence behind the report's coverage numbers; shipping
    // the numbers without it makes them unfalsifiable.
    expect((cases as unknown[]).length).toBeGreaterThan(0);
    for (const c of cases as { entityType: string; corpusTag: string }[]) {
      expect(c.entityType).toBeTruthy();
      expect(c.corpusTag).toBeTruthy();
    }
  });

  it("names all three artifacts from --name, not from the policy", async () => {
    const out = tmp();
    expect(await runCli(args(out, "renamed"), { fixtures: loadTestFixtures() })).toBe(0);
    expect(() => loadPolicyIr(readFileSync(join(out, "renamed.ir.json"), "utf8"))).not.toThrow();
    expect(readFileSync(join(out, "renamed.report.md"), "utf8")).toContain("renamed");
    expect(readFileSync(join(out, "renamed.selftest.json"), "utf8").length).toBeGreaterThan(0);
    expect(existsSync(join(out, "p-fin.ir.json"))).toBe(false);
  });

  it("prints every warning the compile produced", async () => {
    const rec = recorder();
    const out = tmp();
    expect(await runCli(args(out), { ...rec, fixtures: loadTestFixtures() })).toBe(0);
    // p-fin compiles with real warnings (a weak rule, an over-firing rule, a
    // shadowed positive). Exiting 0 without surfacing them is the failure this
    // pins: the exit code says "fine" and nothing says otherwise.
    const warnings = rec.err.filter((l) => l.startsWith("warning: "));
    expect(warnings.length).toBeGreaterThan(0);
    expect(rec.out.join("\n")).toContain("compiled p-fin");
  });

  it("rejects a --name that would escape the output directory", async () => {
    const rec = recorder();
    // The output directory is NESTED inside a fresh temp dir so that "..", the
    // very place an escape lands, is unique to this run. Pointing the assertion
    // at the shared tmpdir instead makes the test permanently red the first time
    // any run does escape — which is exactly what happened while mutation-testing
    // this check, and it cost a confusing debugging detour.
    const base = mkdtempSync(join(tmpdir(), "sih-cli-"));
    const out = join(base, "out");
    const code = await runCli(args(out, "../escaped"), { ...rec, fixtures: loadTestFixtures() });
    expect(code).toBe(1);
    expect(existsSync(join(base, "escaped.ir.json"))).toBe(false);
  });

  it("rejects an unrecognised flag instead of ignoring it", async () => {
    // A typo'd "--liv" must not run to completion as if nothing were wrong.
    const rec = recorder();
    const code = await runCli([...args("/tmp"), "--liv"], {
      ...rec,
      fixtures: loadTestFixtures(),
    });
    expect(code).toBe(1);
    expect(rec.err.join("\n")).toContain("--liv");
  });

  it("says which file it could not read", async () => {
    const rec = recorder();
    const code = await runCli(
      [
        "--policy",
        "/nope.md",
        "--providers",
        join(POLICIES, "providers.json"),
        "--out",
        "/tmp",
        "--name",
        "x",
      ],
      { ...rec, fixtures: loadTestFixtures() },
    );
    expect(code).toBe(1);
    expect(rec.err.join("\n")).toContain("/nope.md");
  });
});
