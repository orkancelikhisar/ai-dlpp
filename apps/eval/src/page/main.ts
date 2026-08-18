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
  /**
   * sha256 of the IR artifact this page loaded, lowercase hex. Async because it
   * is a WebCrypto digest; see `irHash` below for why this is not the IR's own
   * `policyHash` field.
   */
  irHash(): Promise<string>;
  /**
   * The loaded IR's `policyHash` field verbatim -- the compiler's hash of the
   * policy DOCUMENT, which answers a different question from `irHash` and is
   * carried alongside it rather than instead of it. Synchronous: a field read,
   * not a digest.
   */
  policyHash(): string;
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

/**
 * sha256 of `irJson` -- the exact bytes of the IR file this page loaded. This is
 * what a record's `irHash` carries; `ir.policyHash` is carried separately and
 * verbatim, because the two answer different questions.
 *
 * The immediate reason they cannot be one field is that this fixture's
 * `policyHash` is the literal string "test-hash" while a record's `irHash`
 * requires /^[0-9a-f]{64}$/. The lasting reason survives that fixture being
 * replaced by a compiled policy: `ir.policyHash` is the compiler's hash of the
 * policy DOCUMENT, while a record has to answer "which IR produced these
 * numbers". Compilation is model-driven, so the same document compiled twice by
 * the same compiler can yield two different IRs carrying that identical
 * `policyHash` -- it identifies the input, never the artifact.
 *
 * Hashing the raw TEXT rather than a re-serialization of the parsed IR is what
 * makes the answer checkable from outside the browser. MEASURED: for the current
 * fixture this returns
 * cf82e7e925ef6b80036f96225d056516ad1bd885efcb0b6264f903e9fae8271a, which is
 * exactly what `shasum -a 256 apps/eval/fixtures/minimal-ir.json` prints -- so
 * someone holding only a JSONL file can confirm the artifact instead of taking
 * the record's word for it. test/run.spec.ts asserts that equality on every run.
 *
 * The cost is that a whitespace-only reformat of the file changes the hash while
 * the IR is semantically identical. That is the direction to err in: it can call
 * two identical IRs different, never two different IRs the same.
 *
 * Computed once at module scope and handed out as a promise rather than awaited
 * here. A top-level await would turn this file into an async module, and I have
 * not measured whether a rejected async module evaluation still reaches
 * Playwright's `pageerror` listener -- which `openHarness` in smoke.spec.ts
 * relies on to report a malformed IR by name. Keeping the module synchronous
 * leaves that path exactly as Task 1 left it.
 */
const irHash = crypto.subtle
  .digest("SHA-256", new TextEncoder().encode(irJson))
  .then((digest) =>
    Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join(""),
  );

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
  irHash: () => irHash,
  policyHash: () => ir.policyHash,
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
