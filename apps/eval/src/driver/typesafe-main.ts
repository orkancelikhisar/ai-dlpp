/** CLI shim: `pnpm -C apps/eval typesafe`. Everything testable lives in typesafe-run.ts. */
import { runTypeSafe } from "./typesafe-run.js";

runTypeSafe().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});
