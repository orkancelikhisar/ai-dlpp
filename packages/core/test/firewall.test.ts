import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = join(import.meta.dirname, "..", "src");

/** Node-only APIs that would break @sih/core in the browser. */
const FORBIDDEN: Array<{ name: string; pattern: RegExp }> = [
  { name: "node: import", pattern: /from\s+["']node:/ },
  { name: "process global", pattern: /(?<![.\w])process\./ },
  { name: "Buffer global", pattern: /(?<![.\w])Buffer\b/ },
];

function tsFilesUnder(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) return tsFilesUnder(p);
    return p.endsWith(".ts") ? [p] : [];
  });
}

describe("runtime-portability firewall", () => {
  it("src/ contains no Node-only APIs (must run in the browser too)", () => {
    const violations: string[] = [];
    for (const file of tsFilesUnder(SRC)) {
      const content = readFileSync(file, "utf8");
      for (const { name, pattern } of FORBIDDEN) {
        if (pattern.test(content)) violations.push(`${file}: ${name}`);
      }
    }
    expect(violations).toEqual([]);
  });
});
