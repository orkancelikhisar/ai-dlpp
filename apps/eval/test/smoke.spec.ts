import { expect, test } from "@playwright/test";

test("core detects a tier-0 entity inside real Chrome", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__sih !== undefined);

  const result = await page.evaluate(async () => {
    return window.__sih!.detect({
      text: "My PAN is AFTPD1298Q, please help.",
      provider: "claude",
      config: { tier0: true, tier1: false, tier2: false },
    });
  });

  expect(result.findings).toHaveLength(1);
  expect(result.findings[0]!.entityType).toBe("in-pan");
  // Offsets are absolute into the message and must survive the structured-clone
  // boundary between the page and the driver intact. Asserted as literals rather
  // than recomputed from the input string: deriving them here would re-implement
  // in the driver the one thing the page is here to prove core still does.
  //
  // Both offsets AND text, deliberately. Core's own normalizeFindings enforces
  // text === message.slice(start, end) inside the page, so in-page they cannot
  // drift apart -- but nothing enforces it across the evaluate boundary, and a
  // driver that reads only `text` cannot tell a faithful clone from one whose
  // numbers were dropped or defaulted on the way out. Everything downstream of
  // this harness is offsets-first.
  expect(result.findings[0]!.start).toBe(10);
  expect(result.findings[0]!.end).toBe(20);
  expect(result.findings[0]!.text).toBe("AFTPD1298Q");
  expect(typeof result.timings.tier0Ms).toBe("number");
});
