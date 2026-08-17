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

/**
 * A span the policy did NOT rewrite, located in the rewritten text.
 *
 * The text is identical on both sides — `rewritten.slice(newStart, newEnd)` ===
 * `original.slice(start, end)` — so the only information here is the offset
 * shift, and that is exactly the part a caller cannot recompute cheaply.
 * Everything to the right of a length-changing replacement moves, so a UI
 * holding original offsets (Plan 6's review sheet, highlighting what is
 * blocking the send) would otherwise have to replay every delta in `applied`
 * to find where a blocked span landed. The forward pass knows it for free.
 */
export interface SkippedSpan {
  /** Original-text span (== the finding's span). */
  start: number;
  end: number;
  /** The same characters' span in the rewritten text. */
  newStart: number;
  newEnd: number;
  entityType: string;
  action: "block" | "allow";
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
  /**
   * Spans left verbatim because their action was `block` or `allow`, in span
   * order, carrying their position in `text`. Disjoint from `applied`: every
   * finding lands in exactly one of the two lists.
   */
  skipped: SkippedSpan[];
}

/**
 * The DetectionResult contract (spans in bounds, non-inverted, pairwise
 * disjoint), enforced instead of assumed. Every violation this rejects used to
 * produce a plausible-looking result containing REAL text, which is the one
 * outcome this module exists to prevent and the one a caller cannot detect:
 * `applied` stays internally consistent either way, so nothing downstream can
 * tell the difference.
 *
 * - **Overlap.** [0,10) then [5,8) regresses the cursor; `slice(10, 5)` is "",
 *   the cursor then jumps to 8, and characters 8-9 of a span the policy said to
 *   rewrite are re-emitted verbatim in the tail copy.
 * - **Negative start.** JS reads a negative slice index from the END of the
 *   string, so `slice(0, -4)` silently succeeds and the replacement lands in an
 *   unrelated position with the whole original surviving around it.
 * - **Out-of-range end.** Swallows the tail: the gap copy after the last span
 *   is empty, so text nobody looked at disappears from the message.
 * - **Zero-width.** Splices a redaction marker into the message at a point
 *   where nothing was detected, or hands the vault an empty value to mint.
 * - **Non-numeric.** A NaN, a JSON-borne `null`, an omitted field: `slice`
 *   coerces rather than complains. `start: NaN` swallows everything between the
 *   cursor and the span; `end: NaN` emits the marker and then the WHOLE
 *   original after it -- a leak wearing the shape of a successful redaction.
 * - **Fractional.** `slice` truncates, so [2.5, 6.5) rewrites characters 2-6
 *   while `applied` reports 2.5 and 6.5: offsets no consumer can trust.
 *
 * Compared against `lastEnd` -- the previous finding's end, whatever its action
 * -- rather than the rewrite `cursor`. The two are equal for rewritten spans,
 * but `block`/`allow` leave the cursor where it was, so a cursor-based check is
 * blind to a rewrite landing inside a blocked span: that mangles the blocked
 * text and silently falsifies the `skipped` offsets reported below.
 *
 * `normalizeFindings` rejects most of this at the pipeline entrance, and this
 * guard is NOT a delegation to it. `applyActions` is an exported entrance in
 * its own right: findings can be hand-built, replayed from storage, or
 * deserialized from a worker message, none of which passed through `detect`.
 */
function spanViolation(text: string, f: ResolvedFinding, lastEnd: number): string | undefined {
  // Positive conditions, each negated once, and the integer test FIRST. The
  // original form was four negated comparisons chained with `||`, which admits
  // exactly the values that have no ordering: every comparison against NaN is
  // false, so a non-numeric offset satisfied all four and reached `slice`.
  // Ordering is only total once both offsets are known to be real integers.
  if (!(Number.isInteger(f.start) && Number.isInteger(f.end))) return "non-integer offsets";
  if (!(f.start >= 0)) return "negative start";
  if (f.start === f.end) return "zero-width span";
  if (!(f.start < f.end)) return "end before start";
  if (!(f.end <= text.length)) return `end past text length ${text.length}`;
  if (!(f.start >= lastEnd)) {
    return `starts before the previous span ended (${lastEnd}); findings must be pairwise disjoint`;
  }
  return undefined;
}

function assertSpanSane(text: string, f: ResolvedFinding, lastEnd: number): void {
  const violation = spanViolation(text, f, lastEnd);
  if (violation === undefined) return;
  throw new Error(
    `applyActions: invalid span [${f.start}, ${f.end}) for entityType "${f.entityType}" ` +
      `from source "${f.source}": ${violation}`,
  );
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
  let lastEnd = 0;
  const applied: AppliedReplacement[] = [];
  const skipped: SkippedSpan[] = [];

  for (const f of sorted) {
    assertSpanSane(text, f, lastEnd);
    lastEnd = f.end;
    if (f.action !== "pseudonymize" && f.action !== "redact") {
      // Everything from `cursor` to `f.start` is copied verbatim, so the shift
      // at this point is exactly `out.length - cursor` -- no delta replay, and
      // no dependence on where the copy actually happens (the gap is appended
      // later, by the next rewrite or by the final tail copy).
      const newStart = out.length + (f.start - cursor);
      skipped.push({
        start: f.start,
        end: f.end,
        newStart,
        newEnd: newStart + (f.end - f.start),
        entityType: f.entityType,
        action: f.action,
      });
      continue;
    }
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

  return { text: out, blocked, applied, skipped };
}
