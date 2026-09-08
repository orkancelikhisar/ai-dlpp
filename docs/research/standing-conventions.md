# Standing conventions — every AI-DLPP task

Every item here came from a defect actually shipped in this repository. None is boilerplate.
This file lives in the repo because its predecessor lived in a session scratchpad and was
deleted underneath four running agents; the paper's methodology section cites it.

## 1. Comment accuracy is this project's #1 recurring defect

Found in essentially every review round across Plans 3–7. Real examples:

- A config comment said Vite's `server.fs.allow` *adds* to the defaults. It **replaces** them.
- A test comment claimed Playwright's protocol and `structuredClone` differ on `undefined`.
  Measured: they do not.
- A justification appealed to GLiNER's "label encoder". Both pinned checkpoints carry
  `labels_encoder: null` — the component does not exist.
- A PROMPT SENTENCE told the model that short quotes "cannot be located and are discarded".
  Rung 1 accepts a one-word quote fine.
- An error message said `setTimeout` collapses a numeric string to 1 ms. `"300"` fires at 302 ms.
- A module header said "no byte-level BPE can emit more tokens than the string has characters".
  Byte-level BPE bottoms out at one token per BYTE; 20 emoji = 40 UTF-16 units = 60 tokens.

Therefore:

- **Do not write a comment asserting a mechanism, guarantee, or upstream library behaviour you
  have not personally run.** Measure it, then state what you measured. If you cannot measure
  it, say so in the comment.
- Never present someone else's measurement as your own observation. Attribute it.
- Comments explain WHY, not what. House style: `packages/core/src/detect/orchestrator.ts`.
- A comment describing a guarantee the code does not provide is a defect even when the code is
  right.

## 2. Tests must assert what only the real system can satisfy

Plan-supplied tests have been found defective more than a dozen times, always because the
expectation was derived from the thing under test. Variants seen here:

- a hash test keyed by the hash function; a namespacing test building its regex from the
  constant; a corpus test whose id matched the implementation's identical wrong id;
- **a test that exercises only the DEFAULT value, so hardcoding the default survives** — found
  twice in one file (`context_window_size` and `temperature`);
- **a fixture too small for two rules to give different answers** — a percentile test at n=25
  let p95→p96 survive; it needed n=100;
- a `toEqual` of the artifact's own field against itself.

Rules:

- Before declaring done, **deliberately break your implementation and confirm the intended test
  fails.** Report what you mutated, what caught it, and **what survived** — survivors are the
  useful finding.
- **Include a CONTROL mutation you expect to SURVIVE** (reword a string no test reads). If it is
  reported killed, your harness is lying. A harness once returned `tail`'s exit status, reported
  18 surviving mutants as all-killed, and was believed twice. Another restored files in edit
  order and silently corrupted its own copy; the control caught it.
- **Diff-verify that a mutation applied by CONTENT HASH (md5), not `git diff`** — blind to
  untracked files; `git checkout --` silently failed to restore an untracked file once.
- **Never mutate the shared working tree while other agents run.** Extract a pristine copy
  (`git archive HEAD | tar -x -C <scratch>`), confirm it matches HEAD by md5, mutate THAT.
- A test that only proves "something happened" is not coverage.

## 3. Records must state fact, not intent

A `backend` field naming what was *requested*; a `policyHash` naming a document while holding an
artifact hash; a `config` capturing three of six ladder dimensions; a `loadedModelId` seeded from
the requested id; a gold file whose "verbatim" annotator records had been rewritten. Each would
have produced confident, wrong numbers with a fully green suite.

- If a field describes what ran, populate it from what ran.
- A pin is a request; the response is the fact. Record the provider/model/window the response
  carries, beside what was asked for.
- If a value cannot be recovered, an empty field with a stated reason is honest; a reconstructed
  one presented as verbatim is not.

## 4. Do not tune inputs so results look better

A task once edited its corpus to dodge a bad-looking result. Reverted; the measurement was fixed
instead. A corpus was rebuilt from the compiled IR's own `examples`/`counterExamples` with the
compiled arm's `contextBoost` keywords planted near positives only; its F1 of ~0.60 was discarded
unread. **If something produces an ugly number, first establish whether it is TRUE. If it is,
keep it and report it.** Never let a gold value leak into the model's own prompt. Never adjust a
label, threshold, budget or gate to make a number fit. Never drop the comparison (trivial floor,
oracle baseline) that makes a number look bad.

## 5. Push back with evidence

Briefs handed to implementers have been WRONG in every round. Implementers who disproved them by
measurement did the right thing: `context_window_size` in the wrong `CreateMLCEngine` argument;
message-scope judging predicted to cost one extra call and measured as one fewer; a review's
"span was constructed" claim where the span was genuine and the OFFSETS were invented. If a brief
contradicts what the code or the library actually does, measure it and say so.

## 6. Mechanics that have each cost a round

- **Capture exit codes DIRECTLY, on their own line.** Never through a pipe, never as the last of
  a chain. A command ending in `tail` reported exit 0 over a failed Playwright run.
- `pnpm -r test` is the root command. **`npx vitest run` sweeps up Playwright specs it cannot
  run** and reports failures that are not real. `tsc` is not at the root: `npm run -s typecheck`.
- Keep scratch files namespaced and **outside the package directories** — the runner collects
  them, and one was swept into a commit by `git add -A`. **Stage only your own paths.**
- Model weights live in gitignored `models/`; run outputs in gitignored `runs/`. Never commit,
  never delete.
- Before Playwright: `lsof -ti :5178` must be empty. A dev server from another checkout serves a
  DIFFERENT page; the harness guard names the offending directory.
- A skipped tier-2 Playwright suite still exits 0. **"The specs passed" never means "tier 2 ran".**
- Escape trap: the Write/Edit tools decode `\uXXXX`, `\n`, `\t` into RAW characters. Prefer
  `\x1C`/`\u{1C}` or a quoted heredoc; scan for raw control characters before committing.

## 7. Blind annotation

Two annotators, isolated directories holding only the policy and the data. Forbidden: run
outputs, the compiled IR (it encodes one reading of the policy), manifests, family definitions,
any test asserting findings. **Audit `wasToldAbout` as well as `filesRead`** — a brief once
paraphrased the predicate in the compiled IR's own `nlPredicate` wording while forbidding the IR.
Point annotators at the policy's OWN section heading and require them to quote the clause they
applied. Opaque row ids, deterministic shuffle. Report agreement BEFORE adjudicating; exclude
disputed rows rather than forcing them; commit both annotators' returns verbatim and make the
builder throw on divergence.

## 8. Tier-2 specifics

- `@mlc-ai/web-llm` pinned to exactly `0.2.84`. `context_window_size` goes in the THIRD
  `CreateMLCEngine` argument; the second silently drops it.
- `enable_thinking`: **`false` is the dangerous value**, not `true`. The pipeline tests
  `=== false` and only that branch emits the think block. The key lives under `extra_body`.
- Never `structural_tag` (hangs). `response_format.schema` must be a STRINGIFIED JSON schema. An
  uncompilable xgrammar grammar HANGS rather than errors — compile every schema under an external
  watchdog with a known-good control first.
- Never `Promise.race` an engine call: interrupt → drain → **clearInterrupt()**, or every later
  call returns instantly empty and a judge reads that as "no findings" forever.
- Spans: `Finding.start/end` is the ACTION span (the mention); the enclosing clause locates it.
  Mis-located spans pass core's fidelity check, so `rung` is a first-class reported metric.
- A trivial orthographic reader is competitive with every local arm. **No arm figure is quoted
  without the floor beside it.** The overlap figure is definition-sensitive (78–89% pooled
  depending on the rule; the judge-only arms range 14–80%); state the definition.

## 9. Score at the granularity the gold actually records

Found on 2026-09-08, while re-deriving a floor to check it. It changed the ceiling arm's
central conclusion, and nothing in the test suite could have surfaced it.

The tier-2 predicate gold carries **179 message-level booleans and 19 spans**: the blind round
asked annotators one question per message, and the spans are supplementary. The scorer's table,
labelled "predicate level", paired a finding's **span** against a gold **span** — so an arm that
correctly identified a disclosing message but pointed at the wrong phrase took a false positive
*and* a false negative. Both metrics are legitimate. Only one of them is the judgment the
annotators were actually asked for, and the document presented the other as though it were.

The two disagree about the experiment's headline question. Span-wise: local 0.197 < floor 0.571 ≈
ceiling 0.565, so scale does not get you past a regex. Message-wise, same rows and same gold:
local 0.262 < floor 0.776 < **ceiling 0.905**, so scale clearly does. One of those says the
in-browser constraint is not what costs accuracy; the other says it substantially is.

The rules that follow:

- **Identify the gold's native unit before choosing a metric.** Count the annotations. If there are
  179 of one kind and 19 of another, the first one is the primary label and the second is
  supplementary.
- **A stricter metric is not automatically the honest one.** Requiring a span match on a
  message-level label is a real and defensible requirement — pseudonymisation needs the span — but
  it is a *different question*, and reporting it alone silently answers the easier question with
  the harder one's number.
- **When both are meaningful, report both, side by side, and say which one each conclusion rests
  on.** If they disagree, that disagreement is a finding, not a problem to resolve by picking one.
- **Re-derive a floor independently before building a conclusion on it.** The 0.571 floor
  reproduced exactly (tp 14, fp 16, fn 5) — the check that confirmed it was sound is the same check
  that exposed the metric mismatch, because the message-level reimplementation scored 0.776 on
  identical inputs and the gap demanded an explanation.
- **Be suspicious when several matching rules return identical numbers.** `exact`, `overlap` and
  `iou50` all gave 0.571 here. That is not robustness; it means the rules carry no information on
  this data (gold predicate spans *are* capitalised organisation names, so a reader either hits one
  exactly or misses entirely). Identical columns are a signal to ask what the columns are doing.

## 10. Assert the value that gets USED, not the value a function returns

The same defect escaped three fixes in this project, on 2026-09-06/08, and each fix was a
reasonable-looking response to the previous failure:

1. A ledger filename was built inline from the base run id. It overwrote a completed run's ledger —
   768 calls and $0.19269 replaced by a 403-call partial, eight seconds into a relaunch.
2. Fix 1 pinned the id-arithmetic helper (`passRunIdFor`) with a test. **The call site survived a
   mutant.**
3. Fix 2 extracted `spendLedgerFileFor` so "the wiring is what gets asserted". **The call site
   survived again** — the wiring lived in a script file that ends in `await main()` and exports
   nothing, so no test could import it. All 29 mutants planted in the three such files survived.
4. The obvious fix — extract a pure `resolveRunPlan` from `main()` — **would have failed the same
   way a third time.** It kills a mutant that changes the derivation, but not one that *ignores the
   returned value* and rebuilds the wrong string at the point of use. `resolveRunPlan` keeps
   returning the right answer and nothing consumes it.

What finally worked: make the **consumer** an injected dependency, so the test observes the value
that actually arrived at the write.

The rule: **when the defect is "nobody uses the return value", no test of the returning function can
catch it.** Ask what the wrong behaviour would look like from the outside — a file written under the
wrong name — and assert *that*, not the string a helper hands back.

Two structural corollaries:

- **A module that exports nothing is a module with no tests.** A script ending in a top-level
  `main()` is untestable by construction, however clean the functions inside it are. Split it into
  an importable module plus a thin shim before writing the first test.
- **Extraction moves the boundary; it does not close it.** After extracting, plant the mutant at the
  new call site and confirm it dies. If it survives, the extraction bought nothing.
