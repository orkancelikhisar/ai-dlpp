import { describe, expect, it, vi } from "vitest";
import { runWithDeadline, DeadlineExpired } from "../src/cancel.js";

/**
 * The DeadlineExpired a call rejected with.
 *
 * `await p.catch((e) => e)` types as `resolved | DeadlineExpired`, which makes
 * every field access a cast; and it silently returns the RESOLVED value when
 * the call did not reject at all, so an assertion written that way passes
 * vacuously the moment the rejection stops happening. This throws instead.
 */
async function rejection(p: Promise<unknown>): Promise<DeadlineExpired> {
  try {
    await p;
  } catch (e) {
    return e as DeadlineExpired;
  }
  throw new Error("expected this call to reject, but it resolved");
}

/**
 * A fake engine that models what web-llm 0.2.84 was MEASURED to do, not what a
 * cancellable API would ideally do. Three behaviours are copied from the real
 * engine because each one breaks a plausible-looking implementation:
 *
 * 1. `create` ignores the AbortSignal entirely. `chat.completions.create` takes
 *    no signal -- only `interruptGenerate()` stops a generation. A fake that
 *    resolved on abort would let an implementation that never interrupts pass.
 * 2. The interrupt flag is STICKY. `_generate` clears it on entry, but
 *    `chatCompletion` checks it BEFORE calling `_generate` and the poisoned
 *    branch never clears it -- so once set, every later non-streaming call
 *    returns "" instantly, forever. Measured in real Chrome: after one expiry
 *    the next two calls returned in 0 ms with finish_reason "abort" and empty
 *    content, and only writing the flag back to false recovered the engine.
 * 3. Calls serialize on a per-engine lock that is acquired BEFORE the flag is
 *    checked, so a second call already waiting cannot skip the poisoned branch.
 */
function fakeEngine(opts: { hangs?: boolean } = {}) {
  const state = {
    interrupted: 0,
    drained: 0,
    calls: 0,
    cleared: 0,
    /** Calls short-circuited by the sticky flag: silently empty answers. */
    poisoned: 0,
    /** Models MLCEngine's own `interruptSignal` field. */
    interruptSignal: false,
    /** Whether the next generation is a long one. Mutable: a judge issues both. */
    hangs: opts.hangs === true,
  };
  let lock: Promise<void> = Promise.resolve();

  return {
    state,
    interruptGenerate() {
      state.interrupted += 1;
      state.interruptSignal = true;
    },
    clearInterrupt() {
      state.cleared += 1;
      state.interruptSignal = false;
    },
    async create(_signal?: AbortSignal): Promise<string> {
      const prior = lock;
      let release!: () => void;
      lock = new Promise<void>((r) => { release = r; });
      await prior;
      try {
        state.calls += 1;
        if (state.interruptSignal) {
          state.poisoned += 1;
          return "";
        }
        if (state.hangs) {
          await new Promise<void>((resolve) => {
            const t = setInterval(() => {
              if (state.interruptSignal) { clearInterval(t); resolve(); }
            }, 1);
          });
          state.drained += 1;
          return "";
        }
        return '{"findings":[]}';
      } finally {
        release();
      }
    },
  };
}

describe("runWithDeadline", () => {
  it("returns the value when the call finishes in time", async () => {
    const e = fakeEngine();
    await expect(runWithDeadline(e, (s) => e.create(s), 1000)).resolves.toBe('{"findings":[]}');
    expect(e.state.interrupted).toBe(0);
    // Nothing was interrupted, so nothing may be cleared: clearing here would
    // stomp on a concurrent caller's in-flight interrupt.
    expect(e.state.cleared).toBe(0);
  });

  it("interrupts AND drains on expiry, so the next call still works", async () => {
    // The whole point. A Promise.race would reject here and leave the engine
    // generating; measured on the real pipeline, the next call then never
    // returns. Interrupting and awaiting the drain is what keeps the engine
    // usable.
    const e = fakeEngine({ hangs: true });
    await expect(runWithDeadline(e, (s) => e.create(s), 20)).rejects.toBeInstanceOf(DeadlineExpired);
    expect(e.state.interrupted).toBe(1);
    expect(e.state.drained).toBe(1);
    // The follow-up is a short generation, as a next segment would be.
    e.state.hangs = false;
    // NOT `toBeDefined()`: the measured failure resolves with "" rather than
    // hanging, and "" is defined. The assertion has to name the real answer or
    // it passes against the exact engine state this function exists to prevent.
    await expect(runWithDeadline(e, (s) => e.create(s), 1000)).resolves.toBe('{"findings":[]}');
    expect(e.state.poisoned).toBe(0);
  });

  it("clears the sticky interrupt flag, so the next call is not silently empty", async () => {
    // MEASURED in real Chrome on Qwen3.5-2B-q4f16_1-MLC: interrupt-and-drain
    // alone leaves `interruptSignal` true. The next two calls each returned in
    // 0 ms, finish_reason "abort", content "". That is not a wedge that hangs
    // -- it is worse, a fast empty answer a judge would read as "no findings"
    // on every message from then on.
    const e = fakeEngine({ hangs: true });
    await expect(runWithDeadline(e, (s) => e.create(s), 20)).rejects.toBeInstanceOf(DeadlineExpired);
    expect(e.state.interruptSignal).toBe(false);
    expect(e.state.cleared).toBe(1);
  });

  it("does not resolve before the drain completes", async () => {
    // If the deadline path resolves while the engine is still generating, the
    // caller starts the next segment against a busy engine -- the wedge, one
    // level up.
    const e = fakeEngine({ hangs: true });
    const p = runWithDeadline(e, (s) => e.create(s), 10).catch(() => "rejected");
    await p;
    expect(e.state.drained).toBe(1);
  });

  it("interrupts and drains on an outer abort, but does not call it a budget overrun", async () => {
    // An outer abort takes the SAME recovery path as a deadline -- interrupt,
    // drain, clear -- and that half is asserted below. What it must not do is
    // report itself as the same EVENT: 0 ms of a 5000 ms budget had been spent.
    // A record that cannot tell "we ran out of time" from "the caller changed
    // its mind" attributes a cancelled run to a model being too slow.
    const e = fakeEngine({ hangs: true });
    const ac = new AbortController();
    const p = runWithDeadline(e, (s) => e.create(s), 5000, ac.signal);
    // Abort only once the request has actually reached the engine. Aborting
    // sooner exercises the queue-entry path instead (covered below), and there
    // the correct behaviour is the opposite: no interrupt at all.
    await vi.waitFor(() => { expect(e.state.calls).toBe(1); });
    ac.abort();
    const err = await rejection(p);
    expect(err).toBeInstanceOf(DeadlineExpired);
    expect(err.reason).toBe("aborted");
    // Interrupted -- true here, and the reason `interrupted` is a field rather
    // than a constant baked into the message.
    expect(err.interrupted).toBe(true);
    expect(err.message).not.toMatch(/exceeded/);
    expect(e.state.interrupted).toBe(1);
    expect(e.state.drained).toBe(1);
    expect(e.state.interruptSignal).toBe(false);
  });

  it("rejects immediately when the outer signal is already aborted, claiming nothing", async () => {
    const e = fakeEngine({ hangs: true });
    const p = runWithDeadline(e, (s) => e.create(s), 5000, AbortSignal.abort());
    const err = await rejection(p);
    expect(err).toBeInstanceOf(DeadlineExpired);
    // Never handed to the engine at all, so there is nothing to drain or clear.
    expect(e.state.calls).toBe(0);
    expect(e.state.cleared).toBe(0);
    // ...and therefore the error must claim neither. The old single message
    // said "exceeded its 5000ms budget and was interrupted" here, which was
    // false on both counts, and the two assertions above are what prove it.
    expect(err.reason).toBe("aborted");
    expect(err.interrupted).toBe(false);
    expect(err.message).not.toMatch(/exceeded/);
    expect(err.message).toMatch(/nothing ran/);
    expect(err.message).toMatch(/budget never started/);
  });

  it("refuses a budgetMs setTimeout would silently reinterpret", async () => {
    // MEASURED on Node 26: setTimeout coerces Infinity, NaN, 0, -1 and
    // 2147483648 to a 1 ms delay, while 2147483647 and 1e9 do not fire within
    // 120 ms. So POSITIVE_INFINITY, the natural spelling of "no budget", is the
    // fastest deadline available and interrupts the engine before it answers.
    //
    // REJECTED rather than reinterpreted as "no deadline": a budget is required
    // precisely because both known ways this engine stops responding present as
    // a call that never returns, so "wait forever" is the one value this seam
    // must not accept.
    const e = fakeEngine({ hangs: true });
    for (const bad of [Number.POSITIVE_INFINITY, Number.NaN, 0, -1, 2_147_483_648, -0.5]) {
      await expect(runWithDeadline(e, (s) => e.create(s), bad)).rejects.toThrow(/budgetMs/);
    }
    // Refused before the engine was touched, before a queue turn was taken, and
    // without the interrupt that a 1 ms deadline would have fired.
    expect(e.state.calls).toBe(0);
    expect(e.state.interrupted).toBe(0);
    // Not a DeadlineExpired: this is a caller bug, not a call that ran out of
    // time, and reporting it as the latter would put a phantom timeout in the
    // per-arm counts.
    await expect(runWithDeadline(e, (s) => e.create(s), 0)).rejects.not.toBeInstanceOf(
      DeadlineExpired,
    );
    // The upper boundary is usable, not merely the safe middle.
    const ok = fakeEngine();
    await expect(runWithDeadline(ok, (s) => ok.create(s), 2_147_483_647)).resolves.toBeDefined();
  });

  it("clears its timer on the success path", async () => {
    // A leaked timer keeps the process alive after the suite finishes, which
    // shows up as vitest hanging rather than as a failure.
    vi.useFakeTimers();
    const e = fakeEngine();
    await runWithDeadline(e, (s) => e.create(s), 1000);
    expect(vi.getTimerCount()).toBe(0);
    vi.useRealTimers();
  });

  it("clears its timer when the call throws synchronously", async () => {
    vi.useFakeTimers();
    const e = fakeEngine();
    const boom = () => { throw new Error("engine exploded before returning a promise"); };
    await expect(runWithDeadline(e, boom, 1000)).rejects.toThrow("engine exploded");
    expect(vi.getTimerCount()).toBe(0);
    vi.useRealTimers();
  });

  it("does not let one call's expiry kill an overlapping call", async () => {
    // The judge calls this once per segment. `interruptGenerate()` is
    // engine-wide -- a single boolean, with no per-request handle -- so an
    // interrupt raised for segment A lands on whatever segment B is running.
    // MEASURED in real Chrome: with A and B issued together and only A given a
    // short budget, A rejected DeadlineExpired and B RESOLVED with
    // finish_reason "abort" and a zero-length body. B looked successful and was
    // empty, which a judge reads as "segment B is clean".
    const e = fakeEngine({ hangs: true });
    const a = runWithDeadline(e, (s) => e.create(s), 20).catch((err: unknown) => err);
    const b = runWithDeadline(e, (s) => { e.state.hangs = false; return e.create(s); }, 5000);

    expect(await a).toBeInstanceOf(DeadlineExpired);
    await expect(b).resolves.toBe('{"findings":[]}');
    expect(e.state.poisoned).toBe(0);
  });

  it("interrupts once when the deadline and an outer abort both fire", async () => {
    // Both triggers stay armed until the call settles, so a slow drain can see
    // the outer abort and then the deadline. Interrupting twice is not
    // corrupting on this engine, but it doubles the worker round-trip and makes
    // "who interrupted this generation" unanswerable; one interrupt per call is
    // the property worth holding.
    const state = { interrupted: 0, cleared: 0, started: false };
    const slow = {
      interruptGenerate() { state.interrupted += 1; },
      clearInterrupt() { state.cleared += 1; },
      // A drain that takes 150 ms: long enough for the 50 ms deadline to fire
      // after the outer abort has already tripped.
      create: () => {
        state.started = true;
        return new Promise<string>((resolve) => setTimeout(() => resolve(""), 150));
      },
    };
    const ac = new AbortController();
    const p = runWithDeadline(slow, () => slow.create(), 50, ac.signal);
    await vi.waitFor(() => { expect(state.started).toBe(true); });
    ac.abort();
    await expect(p).rejects.toBeInstanceOf(DeadlineExpired);
    expect(state.interrupted).toBe(1);
    expect(state.cleared).toBe(1);
  });

  it("still drains and clears when interruptGenerate rejects", async () => {
    // `MLCEngineInterface` declares `interruptGenerate: () => void`, and the
    // web-worker engine's version drops the promise it creates rather than
    // returning it. A seam that assumed a well-behaved Promise<void> would turn
    // a worker-side failure into an unhandled rejection and skip the drain.
    const e = fakeEngine({ hangs: true });
    const rejecting = {
      state: e.state,
      interruptGenerate(): Promise<void> {
        e.interruptGenerate();
        return Promise.reject(new Error("worker never answered"));
      },
      clearInterrupt: () => e.clearInterrupt(),
      create: (s?: AbortSignal) => e.create(s),
    };
    await expect(runWithDeadline(rejecting, (s) => rejecting.create(s), 20)).rejects.toBeInstanceOf(
      DeadlineExpired,
    );
    expect(e.state.drained).toBe(1);
    expect(e.state.interruptSignal).toBe(false);
  });

  it("clears the flag even when the drained call rejects", async () => {
    // The engine was measured to RETURN early on interrupt rather than throw,
    // so this is the defensive path: if a future version rejects instead, the
    // engine must still be left usable, and the caller must see the real error.
    const e = fakeEngine({ hangs: true });
    const failing = {
      state: e.state,
      interruptGenerate: () => e.interruptGenerate(),
      clearInterrupt: () => e.clearInterrupt(),
      async create(): Promise<string> {
        await new Promise<void>((resolve) => {
          const t = setInterval(() => {
            if (e.state.interruptSignal) { clearInterval(t); resolve(); }
          }, 1);
        });
        throw new Error("aborted by interrupt");
      },
    };
    await expect(runWithDeadline(failing, () => failing.create(), 20)).rejects.toThrow(
      "aborted by interrupt",
    );
    expect(e.state.interruptSignal).toBe(false);
    expect(e.state.cleared).toBe(1);
  });
});
