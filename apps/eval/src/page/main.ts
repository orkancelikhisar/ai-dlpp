import { detect, loadPolicyIr, type DetectionResult, type TierConfig } from "@sih/core";
// Copied from packages/core/test/fixtures/minimal-ir.ts and owned by this app:
// core's test fixture is a TypeScript module, and importing it would both make
// a test-only artifact a runtime dependency of the harness and bypass the thing
// worth exercising here -- the extension receives a compiled IR as JSON text and
// parses it with loadPolicyIr, so the page must too. Later tasks replace this
// with a real compiled policy; until then it is a placeholder, not a baseline.
import irJson from "../../fixtures/minimal-ir.json?raw";

export interface DetectRequest {
  text: string;
  provider: string;
  config: TierConfig;
}

/**
 * The page's whole API surface: every spec reaches detection through this, and
 * everything below `detect` is core, imported unmodified from the package root
 * exactly as the extension will import it.
 *
 * A CONVENTION, not a sandbox. The property is frozen and non-writable below,
 * so a spec cannot swap the implementation out from under the page -- but
 * nothing stops one importing @sih/core (or anything else) into the browser
 * context and measuring that instead. Keeping the harness honest about what it
 * measures is a review obligation; this interface is only what makes the
 * intended path the easy one.
 */
export interface SihPageApi {
  detect(request: DetectRequest): Promise<DetectionResult>;
}

declare global {
  interface Window {
    /**
     * Optional because it genuinely is: the driver navigates, then polls for
     * this property to appear. Typing it as always-present would make the wait
     * that every spec opens with look like dead code.
     */
    __sih?: SihPageApi;
  }
}

// Parsed at module scope so a malformed IR fails BEFORE `__sih` is published,
// rather than a spec receiving an API that throws on first use. The cost is
// that the failure reaches the driver only as a pageerror -- `openHarness` in
// smoke.spec.ts listens for exactly that and re-throws it with the message.
const ir = loadPolicyIr(irJson);

const api: SihPageApi = {
  // Destructured and forwarded field by field rather than spread, so the set of
  // things a spec can influence is exactly the three fields of DetectRequest and
  // is visible in one line.
  //
  // `engines` is absent because tier 0 needs none -- NOT as a permanent rule.
  // Task 10 wires a real tier-1 engine in here, and when it does the engine must
  // be constructed in the page (module-level, loaded once, named in the result
  // so the record says which engine produced the numbers) rather than passed in
  // through DetectRequest: an engine crossing the evaluate boundary is a stub by
  // construction, and the harness would report its latency as core's.
  detect: ({ text, provider, config }) => detect({ ir, provider, text, config }),
};

// Non-writable and non-configurable, not just assigned. Reassigning
// `window.__sih` from a spec would redirect every later measurement in that
// worker to something that is not core, and silently.
Object.defineProperty(window, "__sih", {
  value: Object.freeze(api),
  writable: false,
  configurable: false,
});

// Not what the driver waits on -- `__sih` is -- but it makes a headed run and
// the only-on-failure screenshot say whether the module reached its end.
document.getElementById("status")!.textContent = "ready";
