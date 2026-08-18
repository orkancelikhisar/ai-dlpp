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
 * The page's whole API surface. Playwright reaches detection ONLY through this,
 * so the harness cannot accidentally measure a re-implementation: everything
 * below `detect` is core, imported unmodified.
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

// Parse at module scope, so a malformed IR fails before `__sih` is published
// and the driver's wait times out instead of a spec seeing an API that throws
// on first use.
const ir = loadPolicyIr(irJson);

window.__sih = {
  // The request is destructured and forwarded field by field rather than spread:
  // `engines` stays absent here, so a spec cannot smuggle a stub detector across
  // the evaluate boundary and have the harness report it as core's numbers.
  detect: ({ text, provider, config }) => detect({ ir, provider, text, config }),
};

// Not what the driver waits on -- `__sih` is -- but it makes a headed run and a
// screenshot on failure say whether the module reached its end.
document.getElementById("status")!.textContent = "ready";
