import type { ResolvedFinding } from "../detect/types.js";
import type { PolicyIr } from "../policy/types.js";
import type { Vault } from "./vault.js";

export interface AppliedReplacement {
  /** Original-text span (== the finding's span). */
  start: number;
  end: number;
  /** Span of `replacement` in the rewritten text. */
  newStart: number;
  newEnd: number;
  replacement: string;
  entityType: string;
  action: "pseudonymize" | "redact";
}

export interface ApplyResult {
  text: string;
  /**
   * True when any finding's action is "block". Block does NOT rewrite its span —
   * the caller decides whether the message may be sent at all (spec §5.2: block-
   * severity findings disable "send unmodified"); other findings are still
   * rewritten so the review sheet can show the would-be result.
   */
  blocked: boolean;
  applied: AppliedReplacement[];
}

/**
 * Applies resolved actions to the message text. Findings are guaranteed
 * pairwise disjoint and sorted by start (DetectionResult contract), so a single
 * forward pass assembles the output; new offsets fall out of the assembly.
 *
 * **Errors here are always blocking, whatever `ir.failMode` says.** A rejection
 * from this function (a vault refusal, an exhausted generator pool) means the
 * text was NOT fully rewritten, and there is no partial result to forward: the
 * only string that exists is the original, real values and all. `failMode`
 * open/closed is a choice about DETECTION failures — whether a message nobody
 * managed to scan may go out anyway — and reusing it here inverts its meaning,
 * turning "we could not check this" into "we checked it, found things the policy
 * said to rewrite, failed to rewrite them, and sent them regardless". Callers
 * must surface an apply-stage error to the user and send nothing.
 *
 * **Scope: exactly the resolved spans, never more.** Merge resolution lets a
 * LOSER be wider than the winner that displaced it (merge.ts: winners keep their
 * own span), so a cluster can leave remainder characters that some rule did once
 * call sensitive uncovered by any surviving finding. That residual is knowingly
 * left alone — the motivating case is "SECRET_TOKEN=<secret>" scored as one
 * entropy run against a regex matching just the secret, where the remainder is
 * the key NAME and rewriting it costs utility for nothing. Widening to cluster
 * unions (for critical clusters, say) is the obvious alternative and is a
 * deliberate non-feature for now; revisit if Plan 8's eval shows leaks through
 * residuals. `clusterOverlapping` already exposes the losers such a policy would
 * need, so nothing here forecloses it.
 */
export async function applyActions(
  text: string,
  findings: ResolvedFinding[],
  vault: Vault,
  conversationId: string,
  ir: PolicyIr,
): Promise<ApplyResult> {
  const blocked = findings.some((f) => f.action === "block");
  const sorted = [...findings].sort((a, b) => a.start - b.start);

  let out = "";
  let cursor = 0;
  const applied: AppliedReplacement[] = [];

  for (const f of sorted) {
    if (f.action !== "pseudonymize" && f.action !== "redact") continue;
    // Awaited in the loop on purpose, not gathered with Promise.all: the vault
    // serializes mints per conversation anyway (they are a read-modify-write
    // over one record), so parallelism buys nothing here and only makes the
    // order in which candidates claim surrogates depend on scheduling — which
    // would make the uniqueness ladder's suffix assignment nondeterministic.
    const replacement =
      f.action === "pseudonymize"
        ? await vault.mint(conversationId, f.text, f.entityType, ir)
        : `[REDACTED:${f.entityType}]`;
    out += text.slice(cursor, f.start);
    const newStart = out.length;
    out += replacement;
    applied.push({
      start: f.start,
      end: f.end,
      newStart,
      newEnd: out.length,
      replacement,
      entityType: f.entityType,
      action: f.action,
    });
    cursor = f.end;
  }
  out += text.slice(cursor);

  return { text: out, blocked, applied };
}
