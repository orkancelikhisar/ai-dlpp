/**
 * The capability-ceiling arm's entry point: `pnpm -C apps/eval ceiling`.
 *
 * A shim on purpose. Everything this run decides lives in `ceiling-run.ts`,
 * which exports it, because a top-level `main()` that exports nothing is
 * unreachable from a test: MEASURED on 2026-09-08 against `887317f`, reverting
 * the ledger call site here to `` `ceiling-${runId}.spend.json` `` -- the defect
 * that destroyed a 768-call spend ledger -- left `vitest run` at
 * `Tests 791 passed (791)`, exit 0. Nothing that stays in THIS file can be
 * covered, so nothing that decides anything stays in it.
 *
 * See `ceiling-run.ts` for the environment variables and the order of
 * operations.
 */
import { runCeiling } from "./ceiling-run.js";

await runCeiling();
