import { test as base, chromium, type BrowserContext, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * The dev server the whole harness runs against.
 *
 * Exported and imported by `playwright.config.ts` rather than written twice,
 * because the tier-2 specs cannot take it from the project's `use`: they run in
 * a browser context this module launches itself (see `test` below), and
 * `baseURL` is one of the options a self-launched context does not receive.
 * Two copies would drift into a suite where half the specs talk to a server the
 * other half is not using.
 */
export const BASE_URL = "http://localhost:5178";

/**
 * Where the tier-2 browser profile lives, and it is deliberately OUTSIDE the
 * repository.
 *
 * A model cache is gigabytes: the four pinned arms together are more than 7 GB
 * of weights at one origin. This plan's own commit step is `git add -A`, and
 * this project has already swept a scratch artifact into a commit that way, so
 * a profile under `apps/eval/` would be one forgotten .gitignore line away from
 * a multi-gigabyte commit. Keeping it under the user's cache directory makes
 * that impossible rather than unlikely.
 *
 * `SIH_EVAL_PROFILE_DIR` overrides it, which is what a CI image with a
 * pre-warmed cache (or a machine with a small home volume) needs.
 */
export function tier2ProfileDir(): string {
  return process.env["SIH_EVAL_PROFILE_DIR"] ?? join(homedir(), ".cache", "sih-eval", "chrome-profile");
}

/**
 * Navigate and wait for the page to publish its API, reporting a pageerror by
 * message. A copy of `tier1.spec.ts`'s, because the failure it exists to
 * translate is the same one: the IRs are parsed at module scope, so a malformed
 * fixture leaves `__sih` unpublished and the wait fails as a bare timeout
 * naming nothing.
 */
export async function openHarness(page: Page): Promise<void> {
  const pageErrors: Error[] = [];
  page.on("pageerror", (error) => pageErrors.push(error));
  await page.goto("/");
  try {
    await page.waitForFunction(() => window.__sih !== undefined, undefined, { timeout: 30_000 });
  } catch (cause) {
    if (pageErrors.length > 0) {
      throw new Error(
        `harness never became ready; the page threw: ${pageErrors.map((e) => e.message).join("; ")}`,
        { cause },
      );
    }
    throw cause;
  }
}

/**
 * The tier-2 specs' `test`, whose `page` lives in a PERSISTENT browser profile.
 *
 * This is a correctness matter, not a convenience, and the numbers are the
 * argument. web-llm caches model weights in the Cache API under the PAGE's
 * origin, and MEASURED here by loading all four pinned arms at this origin,
 * they come to 7.49 GB together -- 1.08 GB for Qwen3.5-2B and roughly 2 to
 * 2.3 GB for each of the other three.
 *
 * `navigator.storage.estimate().quota`, MEASURED on this machine against this
 * same dev server, four times:
 *
 * - Playwright's ordinary context (`browser.newContext()`, which is what the
 *   built-in `page` fixture builds on): 3,221 MB standalone, and 4,295 MB from
 *   inside this suite with the profile mutated to use one. Chrome derives the
 *   number from free disk, so it moves; both are far under what four arms need.
 * - This persistent profile: 10,737 MB while empty, 18,230 MB once it held the
 *   7.49 GB of weights.
 *
 * So the ordinary context runs out of quota partway through a bake-off, and a
 * `QuotaExceededError` mid-download surfaces as a failed load -- indistinguishable
 * from a model that cannot load at all.
 *
 * The profile is also what makes the SECOND run cheap: the weights stay in it,
 * so a load that first cost a download costs a cache read afterwards.
 *
 * Playwright has no config option for this -- `launchPersistentContext` is an
 * API call -- so the context is launched here, which costs the options the
 * built-in fixtures would have applied. MEASURED rather than assumed, because
 * the first version of this module guessed wrong in both directions:
 *
 * - `screenshot: "only-on-failure"` and `trace: "retain-on-failure"` DO reach
 *   this context. Playwright's artifacts recorder attaches to every context the
 *   client creates, including this one -- a failing test here wrote both a
 *   screenshot and a trace.zip with nothing in this module doing it, and a
 *   hand-rolled `tracing.start()` failed with "Tracing has been already
 *   started". So there is no re-implementation here, and there must not be.
 * - `baseURL` does not arrive on its own, so it is passed explicitly below.
 *   That is what keeps `page.goto("/")` working in these specs exactly as it
 *   does in the others.
 */
// The empty first type argument is not an oversight: this suite declares no
// test-scoped fixture of its own. It OVERRIDES the built-in `page`, which
// `extend` accepts without declaring, and the second argument is where the
// worker-scoped context has to be named.
export const test = base.extend<{}, { tier2Context: BrowserContext }>({
  tier2Context: [
    async ({ channel, headless }, use, workerInfo) => {
      // `parallelIndex`, NEVER `workerIndex`, and the difference is not
      // pedantic: the first version of this guard used `workerIndex` and turned
      // a green suite red. MEASURED on a `workers: 1` run of the whole suite,
      // the first tier-2 test already had `workerIndex` 1 and the next spec file
      // 2 -- Playwright numbers worker PROCESSES and had started fresh ones,
      // for reasons of its own that this comment does not need to know.
      // `parallelIndex` is the SLOT, between 0 and workers - 1, and it is the
      // number that says whether two browsers could be alive at once.
      //
      // Chromium takes an exclusive lock on a profile directory, so a genuinely
      // second worker fails inside the launcher with a message about
      // SingletonLock. It is also the wrong thing to want: Plan 4 established
      // that parallel workers race over model loading, which is why
      // `playwright.config.ts` pins `workers: 1`.
      if (workerInfo.parallelIndex > 0) {
        throw new Error(
          `the tier-2 specs share one browser profile (${tier2ProfileDir()}) and cannot run in ` +
            `more than one worker at a time; playwright.config.ts pins workers: 1 and this run ` +
            `has parallel slot ${workerInfo.parallelIndex}`,
        );
      }
      const dir = tier2ProfileDir();
      mkdirSync(dir, { recursive: true });
      // `channel` and `headless` are read from the project rather than
      // hardcoded: both are worker-scoped options, and `channel: "chromium"` is
      // load-bearing (playwright.config.ts records that the headless shell's
      // requestAdapter() resolves null, and web-llm THROWS on a null adapter).
      const context = await chromium.launchPersistentContext(dir, { channel, headless, baseURL: BASE_URL });
      await use(context);
      await context.close();
    },
    { scope: "worker" },
  ],
  // A fresh page per test in the SHARED context, so each test starts on a clean
  // page while the profile -- and with it the model cache -- persists across the
  // whole file. Closed after each test: a page holding a loaded engine keeps its
  // weights on the GPU.
  page: async ({ tier2Context }, use) => {
    const page = await tier2Context.newPage();
    await use(page);
    await page.close();
  },
});

export { expect } from "@playwright/test";
