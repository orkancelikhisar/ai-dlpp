import type { MLCEngineInterface } from "@mlc-ai/web-llm";

/**
 * Thrown when a call was stopped before it produced an answer.
 *
 * `reason` and `interrupted` exist because ONE message cannot describe all
 * three ways that happens without stating a falsehood in two of them:
 *
 * - `budget`: the deadline really did expire and the generation was interrupted.
 * - `aborted` with `interrupted: true`: the caller withdrew mid-generation. The
 *   budget was not exceeded; it was abandoned with time still on it.
 * - `aborted` with `interrupted: false`: the caller withdrew while the call was
 *   still QUEUED. Nothing ran and nothing was interrupted -- deliberately, since
 *   interrupting for a call that never started would poison the engine for
 *   whoever runs next.
 *
 * The single message this class used to carry -- "exceeded its Nms budget and
 * was interrupted" -- was false in both `aborted` cases, and a record built
 * from it could not say which of the three had happened. That is the
 * intent-recorded-as-fact defect this project has shipped before.
 */
export class DeadlineExpired extends Error {
  constructor(
    readonly reason: "budget" | "aborted",
    readonly budgetMs: number,
    readonly interrupted: boolean,
  ) {
    super(
      reason === "budget"
        ? `tier-2 call exceeded its ${budgetMs}ms budget and was interrupted`
        : interrupted
          ? `tier-2 call was aborted by its caller mid-generation and was interrupted; ` +
            `its ${budgetMs}ms budget had not expired`
          : `tier-2 call was aborted by its caller while still queued; nothing ran, ` +
            `nothing was interrupted, and its ${budgetMs}ms budget never started`,
    );
    this.name = "DeadlineExpired";
  }
}

/**
 * The part of a WebLLM engine this wrapper needs.
 *
 * `interruptGenerate` returns `void | Promise<void>` because the two shipped
 * engines disagree: `MLCEngine` returns a promise, while `MLCEngineInterface`
 * declares `() => void` and `WebWorkerMLCEngine` posts a message and drops the
 * promise it creates. Typing it as `Promise<void>` excludes the worker engine.
 *
 * `clearInterrupt` has no counterpart in the WebLLM API and exists because of
 * a measurement -- see `mlcInterruptible`, which is the only place that knows
 * how to perform it.
 */
export interface Interruptible {
  interruptGenerate(): void | Promise<void>;
  clearInterrupt(): void;
}

/**
 * Adapt a real WebLLM engine to the seam above.
 *
 * MEASURED against `@mlc-ai/web-llm` 0.2.84 in real Chrome on
 * `Qwen3.5-2B-q4f16_1-MLC`: `interruptGenerate()` sets an engine-wide
 * `interruptSignal` boolean, and NOTHING on the non-streaming path clears it
 * afterwards. `_generate` clears it on entry, but `chatCompletion` tests it
 * BEFORE deciding to call `_generate` and the poisoned branch returns `""`
 * without ever reaching the clear. Measured: after a single
 * interrupt-and-drain, the next two calls each returned in 0 ms with
 * `finish_reason: "abort"` and empty content, and writing the flag back to
 * false restored normal service in the same page (168 ms, `finish_reason:
 * "stop"`, real text). Two calls are what was observed; the source is what
 * makes it permanent, since the poisoned branch is the one path that never
 * reaches the clear.
 *
 * `resetChat()` does not clear it -- it never touches the field. Only a
 * streaming call self-heals, because `asyncGenerate` clears the flag
 * unconditionally on entry; the pinned tier-2 recipe is non-streaming, so that
 * escape hatch does not apply. Writing the field is therefore the only recovery
 * short of unloading the model, which costs a multi-second reload.
 *
 * The cast is deliberate: `interruptSignal` is declared `private` in
 * `engine.d.ts`, which is a compile-time marker only -- at runtime it is an
 * ordinary property. Reaching past the type is the narrowest available fix, and
 * it is confined to this function so there is exactly one place to revisit if
 * upstream ever clears the flag itself.
 */
export function mlcInterruptible(engine: MLCEngineInterface): Interruptible {
  return {
    interruptGenerate: () => engine.interruptGenerate(),
    clearInterrupt: () => {
      (engine as unknown as { interruptSignal: boolean }).interruptSignal = false;
    },
  };
}

/**
 * One in-flight call per engine.
 *
 * Not an optimisation -- a correctness requirement, and the reason is a
 * measurement. `interruptGenerate()` is engine-wide: a single boolean with no
 * per-request handle, so an interrupt raised for one call lands on whichever
 * call the engine is running. MEASURED with two overlapping calls on one engine
 * where only the first had a short budget: the first rejected `DeadlineExpired`
 * as intended, and the second RESOLVED with `finish_reason: "abort"` and a
 * zero-length body. A silently empty answer that a judge reads as "this segment
 * is clean" is a false negative in a system whose whole job is not missing
 * things.
 *
 * Serialising here rather than relying on the engine's own lock is what makes
 * the clear safe to order. The engine releases its internal lock BEFORE the
 * awaiting call's promise settles, so a queued second call can read
 * `interruptSignal` while it is still set -- our clear runs a microtask too
 * late. Holding the turn until after the clear removes that race. Under the
 * pinned one-model-per-engine rule it costs no throughput, because the engine
 * already serialises non-streaming calls on that model's own lock; it WOULD
 * serialise two models sharing one engine, which the engine's per-model locks
 * would not.
 */
const inFlight = new WeakMap<Interruptible, Promise<void>>();

/**
 * The largest delay `setTimeout` stores without overflowing its 32-bit field.
 * Anything above it, and several values below it, are silently REINTERPRETED
 * rather than rejected -- see `runWithDeadline`.
 *
 * Exported from this MODULE, and deliberately not from the package index, so
 * `judge.ts` can refuse a bad budget at CONSTRUCTION rather than many segments
 * later on its first engine call. A second copy of the literal there would be
 * free to drift from the bound actually enforced below.
 */
export const MAX_BUDGET_MS = 2_147_483_647;

/**
 * Run one engine call under a deadline, interrupting AND DRAINING on expiry.
 *
 * MEASURED, and the reason this function exists: a `Promise.race` between the
 * call and a timer abandons the promise but not the generation, which keeps
 * running and keeps the engine's per-model lock. On the pinned NON-streaming
 * recipe that is a latency wedge rather than a permanent one -- a 1500 ms race
 * against a 512-token generation delayed the next call by 10,235 ms, after
 * which the engine recovered on its own. On the STREAMING path, abandoning a
 * `for await` never releases the lock at all and the next call does not return;
 * that is where the permanent wedge was measured. Both are unacceptable at a
 * budget of a few seconds per segment.
 *
 * So on expiry this asks the engine to stop, then AWAITS the original promise
 * before rejecting. The await is the load-bearing part: resolving earlier hands
 * control back to a caller that will immediately start the next segment against
 * a still-busy engine, which is the same wedge one level up. Measured, a
 * drained call settled 9-18 ms past its 1500 ms budget across two runs.
 *
 * Draining is necessary and NOT sufficient, which is the second half of this
 * function. A drained engine is left with its interrupt flag set and answers
 * every later call instantly and emptily; `clearInterrupt` after the drain is
 * what actually returns the engine to service. See `mlcInterruptible`.
 *
 * `budgetMs` is measured from the moment this call reaches the engine, not from
 * the moment it was requested, so time spent queued behind another call on the
 * same engine does not count against it. A caller that needs a bound on total
 * elapsed time should pass `outer`, which is honoured while queued.
 *
 * @throws a plain Error, synchronously in effect, when `budgetMs` is not a
 *   finite duration in `(0, 2147483647]`. Infinity is REJECTED rather than read
 *   as "no deadline" -- see the guard, which carries the measurement.
 * @throws {DeadlineExpired} when the budget expired or `outer` aborted. Read
 *   its `reason` to tell those apart; they are not the same event.
 */
export async function runWithDeadline<T>(
  engine: Interruptible,
  call: (signal: AbortSignal) => Promise<T>,
  budgetMs: number,
  outer?: AbortSignal,
): Promise<T> {
  // `setTimeout` does not reject a nonsense delay, it REINTERPRETS one, and
  // every reinterpretation lands on the same value: 1 ms. MEASURED HERE on
  // Node 26 -- Infinity, NaN, 0, -1 and 2147483648 each fired in 1-4 ms, while
  // 2147483647 and 1e9 did not fire at all within 120 ms. So the natural
  // spelling of "no budget", `Number.POSITIVE_INFINITY`, does the exact
  // opposite: it trips instantly, interrupts an engine that has not answered
  // yet, and produces a DeadlineExpired reading "exceeded its Infinityms
  // budget".
  //
  // REJECTED rather than reinterpreted as "no deadline", and that is the
  // deliberate half of this decision. `budgetMs` is a required parameter
  // precisely because both known ways this engine stops responding present as a
  // call that never returns; accepting a value meaning "wait forever" would
  // reopen the hole the required parameter closes. A caller who genuinely wants
  // no practical bound passes a large finite number and can see it in the
  // record. `manifest.ts` validates its four numbers with the same shape of
  // explicit guard.
  //
  // Checked before the queue turn is taken, so a bad budget cannot occupy a
  // slot on an engine other calls are waiting for.
  if (!(Number.isFinite(budgetMs) && budgetMs > 0 && budgetMs <= MAX_BUDGET_MS)) {
    throw new Error(
      `tier-2 budgetMs must be a finite number of milliseconds in (0, ${MAX_BUDGET_MS}], ` +
        `got ${budgetMs}; setTimeout silently turns anything else into a 1ms deadline`,
    );
  }

  // Take a turn on this engine before doing anything observable.
  const prior = inFlight.get(engine) ?? Promise.resolve();
  let releaseTurn!: () => void;
  inFlight.set(engine, new Promise<void>((resolve) => { releaseTurn = resolve; }));
  await prior;

  try {
    // Checked after the wait as well as before it: a caller that gave up while
    // queued must not start a generation only to interrupt it, and interrupting
    // for a call that never ran would poison the engine for the next one.
    if (outer?.aborted === true) throw new DeadlineExpired("aborted", budgetMs, false);

    const ac = new AbortController();
    // Which of the two stops fired, or undefined if neither did. A boolean
    // could only say THAT something stopped the call, and the error has to say
    // which: an outer abort is not a budget overrun and reporting it as one
    // would put a fabricated timeout into every cancelled row of the bake-off.
    let tripped: "budget" | "aborted" | undefined;

    const trip = (why: "budget" | "aborted") => {
      if (tripped !== undefined) return;
      tripped = why;
      ac.abort();
      // Fire-and-forget by necessity: the drain below waits on the generation,
      // not on this acknowledgement, and the worker engine does not return a
      // promise here at all. The catch is not optional -- an unhandled
      // rejection from a worker round-trip would surface as a process-level
      // error unrelated to the call that caused it.
      //
      // If the interrupt never lands the drain still terminates, because the
      // generation stops at `max_tokens` on its own; the call then overruns its
      // budget rather than hanging. Measured on the control run where nothing
      // was interrupted: a 512-token generation finished and freed the engine
      // after 10,235 ms.
      try {
        void Promise.resolve(engine.interruptGenerate()).catch(() => {});
      } catch {
        /* a synchronous throw here must not pre-empt the drain */
      }
    };

    // Two distinct closures, not one shared handler: `removeEventListener`
    // needs the same reference it was given, and each has to name its own
    // cause.
    const onBudget = () => trip("budget");
    const onAbort = () => trip("aborted");

    const timer = setTimeout(onBudget, budgetMs);
    outer?.addEventListener("abort", onAbort, { once: true });

    try {
      const value = await call(ac.signal);
      // Deliberately checked AFTER the await: the interrupt makes the engine
      // return early rather than throw -- measured, the drained call resolved
      // rather than rejecting, 9-18 ms past a 1500 ms budget -- so a drained
      // call resolves normally with a partial or empty body. Returning that as a
      // real answer would report a truncated judgement as a complete one.
      if (tripped !== undefined) throw new DeadlineExpired(tripped, budgetMs, true);
      return value;
    } finally {
      clearTimeout(timer);
      outer?.removeEventListener("abort", onAbort);
      // Only after the drain, and only if we were the one who interrupted.
      // Clearing a flag we did not set would release someone else's interrupt
      // and let a generation they had already given up on run to completion.
      if (tripped !== undefined) engine.clearInterrupt();
    }
  } finally {
    releaseTurn();
  }
}
