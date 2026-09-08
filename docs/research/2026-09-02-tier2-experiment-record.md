# The tier-2 experiment: a research record

**Branch** `feat/tier2-judge` · **HEAD** `0edb3db` · **Written** 2026-09-07 ·
**Machine** one macOS host (Darwin 24.6.0), one GPU, Node v26.0.0

## How to read this document

Every quantitative claim below names the artifact it was read from and, where the
number was computed rather than stored, the command that computed it. Numbers were
re-derived from the artifacts for this document; none was copied from a commit
message, a plan document or a summary. Where a source I was given disagrees with an
artifact, both are reported and the artifact wins — those cases are collected in
[§11](#11-could-not-verify).

Two conventions are used throughout:

- **"MEASURED HERE"** means I ran something for this document and report its output.
- **"stored"** means the number is a field in a committed or on-disk artifact and I
  read it out.

Derivations marked *(derivation script)* were run by loading the repository's own
committed modules under `vitest` from a scratch directory outside the repository, so
that no source, test, corpus or run artifact was modified:

```
# config: {test:{root:"apps/eval", include:["<scratch>/*.derive.ts"]}}
cd apps/eval && ./node_modules/.bin/vitest run --config <scratch>/derive.config.mjs <scratch>/<name>.derive.ts
```

The scripts import `apps/eval/src/driver/score.js`, `apps/eval/src/driver/corpus.js`,
`apps/eval/src/corpus/leakage.js`, `apps/eval/src/corpus/predicate-round.js` and
`apps/eval/src/corpus/families.v2.js` directly, so every metric below is computed by
the repository's own definitions rather than by a re-implementation.

**Terms expanded on first use.** *Tier 0* = compiled regular expressions plus named
validators. *Tier 1* = a zero-shot span tagger (GLiNER-class ONNX model). *Tier 2* =
a local instruct LLM run in the browser through WebGPU. *IR* = the Policy
Intermediate Representation, the JSON artifact a policy document compiles into.
*Approach B* = the control arm: one LLM given the whole policy document plus the whole
message, no compiler and no tiers. *Arm* = one (method family × model) configuration.
*Gold span* = a labelled correct answer, expressed as character offsets into a
message.

---

## 1. Research question

The specification states the question in one sentence
(`docs/superpowers/specs/2026-08-13-ai-dlpp-design.md`, §1, line 13):

> **How well can policy-conditioned, fully-local models prevent confidential-data
> leakage in LLM prompts, and at what latency/hardware cost?**

### What fraction of it this record answers

The harness itself states its own scope, per arm, in the `experimentScope` field of
every gates row (read from `runs/slate-p-fin-02.gates.jsonl`, first record). Its
verdict, quoted from the artifact:

> the models are policy-conditioned (at one policy) and fully local (every arm runs in
> the page and no prompt leaves the browser); the latency cost is per-call TTFT and
> decode rate on one machine, at the message budget in `run.latencyBudgetMs` […] and
> the hardware cost is `engineLoadMs`, `engineWarmupMs` and `originStorageBytes` on
> that one GPU. It answers NOTHING about 'how well … prevent leakage': no
> leak-prevention rate, over-blocking rate, span P/R/F1, policy-adaptivity delta or
> utility number is computed anywhere in this repository.

That was true of the harness when the field was written and is still true of the
harness. It is **no longer the whole truth of the repository**: `apps/eval/src/driver/
score.ts` (added by `975b5c2`, 714 lines at that commit) computes span-level
precision, recall and F1 for tier-2 predicate findings against a blind-labelled gold
set. It is not a gate and feeds no verdict (`score.ts:99–105`).

So, against the five clauses of the question:

| Clause | Answered here? |
|---|---|
| "policy-conditioned" | Partially. One policy document of the three has ever been compiled (`policies/compiled/` holds `p-fin` only). The adaptivity comparison the spec §6.4.3 defines needs three and cannot run. |
| "fully-local" | Yes, by construction. Every arm executes in the page; the cloud is touched only at compile time, and even that ran from recorded fixtures. |
| "models" | Four tier-2 models × four method families = 16 arms, all executed end to end. |
| "prevent confidential-data leakage" | **Weakly, and negatively.** Span-level and predicate-level accuracy now exist for one policy on one 189-item corpus. No leak-prevention rate or over-blocking rate as the spec defines them (§6.4.1) is computed. The accuracy numbers that do exist do not beat trivial baselines — see §5. |
| "at what latency/hardware cost" | Yes, on one machine and one GPU: per-call time-to-first-token, decode rate, budget exhaustion, warm engine-load time, and total origin storage. |

---

## 2. Apparatus

### 2.1 The four-tier pipeline

The design (spec §4.1) is: tier 0 (compiled regex + named validators + context boost +
entropy scanning, sub-millisecond, synchronous) → tier 1 (zero-shot span tagger,
~20–80 ms) → tier 2 (local instruct LLM via WebLLM, WebGPU only, fires only where
tiers 0/1 left uncertainty or where the IR declares semantic predicates).

**Tier 1 did not run in any experiment reported here.** Verified by histogramming the
`tier` field of every finding across the 16 arms of the `slate-rebuild-01*` runs
(MEASURED HERE, `jq` over `runs/slate-rebuild-01*.jsonl`):

```
783 tier 0  api-credential          543 tier 2  pred:client-relationship-disclosure
288 tier 0  bank-account-identifier  40 tier 2  in-aadhaar
184 tier 0  in-aadhaar               15 tier 2  bank-account-identifier
106 tier 0  internal-customer-id     12 tier 2  api-credential
 96 tier 0  private-key-material      7 tier 2  internal-customer-id
 68 tier 0  db-connection-string      7 tier 2  in-pan
 40 tier 0  in-pan                    4 tier 2  db-connection-string
                                      1 tier 2  client-name
```

No finding at tier 1 exists. This matters for §5b: `client-name` is a **tier-1**
entity type in the compiled IR (`policies/compiled/p-fin.ir.json`,
`entityTypes[4].tier === 1`), so across 16 arms × 189 items there is exactly **one**
`client-name` finding in the whole experiment, and it came from tier 2, not tier 1
(`runs/slate-rebuild-01.baselineB-Qwen3.5-2B-q4f16_1-MLC.jsonl`, item `inj-o10-5`).

### 2.2 The four method families

Each family is a `TierConfig` plus a detector choice; the ablation is a configuration
parameter, not a code branch (spec §4.1). Read from the `family` and `config` fields
of the run records.

| Family | tier 0 | tier 1 | tier 2 | Detector | What it isolates |
|---|---|---|---|---|---|
| `compiled` | on | off | on | core orchestrator + `WebLlmJudge` | The full compiled pipeline |
| `compiled-tier2-only` | off | off | on | core orchestrator + `WebLlmJudge` | The judge alone, without deterministic priors |
| `baseline-b` | off | off | on | `createBaselineB` | Approach B: policy-in-context, no compiler |
| `baseline-b-tier0` | on | off | on | `createBaselineB` + tier-0 priors | Approach B given the compiler's deterministic half |

Held fixed across all four (from the plan's own asymmetry table,
`docs/superpowers/plans/2026-08-30-05-tier2-judge-baseline.md`, and confirmed against
the `run.tier2Config` block on every gates row): the same engine object, model,
context window (8192), temperature (0), `max_tokens` (512), grammar-constrained
decoding path, per-call budget (60,000 ms), per-item ceiling (250,000 ms), corpus,
provider, IR and source policy.

**Approach B is the control and its cost is not to be equalised.** The measured
asymmetries that remain, with their sizes taken from the run artifacts:

- B's prompt carries the whole policy document. `policies/p-fin.md` is 5,272 UTF-16
  characters and 5,320 UTF-8 bytes (MEASURED HERE: `wc -c` gives 5320;
  `readFileSync(...,"utf8").length` gives 5272). Measured prompt sizes on the smoke
  corpus, `runs/slate-p-fin-02.gates.jsonl`: B p50 **1,410** tokens against the
  compiled judge's **252–280**, a ratio of 5.0–5.6×.
- B is asked about the IR's whole entity-type vocabulary plus the predicate in one
  call; the judge's call carries one predicate.
- B spends one `max_tokens` budget per message; the judge spends one per judged unit.
  On `p-fin` these coincide, because the policy's only predicate is message-scoped
  (§2.5).
- The `p95-ttft` gate's ceiling was derived at the judge's ~1.1 kB prompt and is
  applied to B unchanged. The plan calls this **"UNFIXED, and the largest one."**

### 2.3 The four tier-2 models and their VRAM

`packages/tier2/src/manifest.ts` (introduced by `4180ac1`, VRAM field corrected by
`6b169be`) declares four models. **VERIFIED HERE against the installed library** by
requiring `@mlc-ai/web-llm` from `packages/tier2/node_modules` and reading
`prebuiltAppConfig.model_list` (163 models, library version 0.2.84):

| Model id | `vramRequiredMb` in manifest | `vram_required_MB` in `prebuiltAppConfig` | manifest `decodeTokPerSec` | thinking mode |
|---|---|---|---|---|
| `Qwen3.5-2B-q4f16_1-MLC` | 2245 | 2245.44 | 40 | yes |
| `Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC` | 2864 | 2863.69 | 30 | no |
| `Qwen3-4B-q4f16_1-MLC` | 3432 | 3431.59 | 25 | yes |
| `Phi-4-mini-instruct-q4f16_1-MLC` | 3438 | 3437.58 | 24 | no |

All four match the library's value rounded. Two further checks:

- The library's own `overrides.context_window_size` is **4096** for all four; the
  experiment runs every arm at **8192** (`run.tier2Config.contextWindowSize` on every
  gates row). The lift is justified by Approach B's prompt, not by tier 2 — see the
  plan's Task 9 correction.
- `gemma-3-4b-it-q4f16_1-MLC` is **absent** from `prebuiltAppConfig` under any
  capitalisation, confirming spec §4.2's "Gemma-3-4b-it is dropped". The only Gemma-3
  build present is `gemma3-1b-it-q4f16_1-MLC` (711.07 MB), a different weight class.

**VRAM is not download size and not disk.** The manifest says so and gives the
measurement: a full cold load of `Qwen3.5-2B` leaves 1,079 MB of origin storage against
the 2,245 recorded. The four VRAM figures sum to 11,978.30 MB (11,979 summing the
rounded values, which is the figure spec §4.2 quotes); the measured origin storage on a
profile holding all four arms is **7,490,154,752 bytes = 7.49 GB**
(`originStorageBytes`, identical on all 16 rows of `runs/slate-p-fin-02.gates.jsonl`).

**Warm engine-load times, not cold ones.** The harness uses
`chromium.launchPersistentContext` (`apps/eval/test/tier2-profile.ts:169`), so weights
are already cached. `engineLoadMs` on `runs/slate-p-fin-02.gates.jsonl` runs 1,465 ms
(`tier2only-Qwen3.5-2B`) to 3,060 ms (`baselineB+tier0-Phi-4-mini`); the maximum across
all nine gates files is 3,597.69 ms (`baselineB+tier0-Phi-4-mini`, `slate-rebuild-01c`).
`engineWarmupMs` is 411–607 ms on every arm except the Ministral arms, which take
2,245–2,326 ms. The manifest's note that Phi-4-mini's *cold* load is 73.6 s cites a
file (`out-e5.json`) that does not exist in this repository; see §11.

### 2.4 The pinned call recipe

`packages/tier2/src/engine.ts` (`449bc3b`; four defects found and recorded by
`07e0622`). `buildCallParams` sends exactly four keys: `messages`, `temperature`,
`max_tokens`, and `response_format: { type: "json_object", schema: <stringified> }`.
Each clause has a measured justification, quoted from the module docblock at
`engine.ts:193–225`:

| Clause | Measured justification |
|---|---|
| **No `enable_thinking` at any depth** | On 0.2.84 the key lives under `extra_body`; verified by reading the installed bundle — the pipeline tests it with `=== false`, and that branch encodes an empty think block onto `outputIds`, so the literal tag lands in `message.content` and `JSON.parse` throws. `undefined` and `true` take the other branch, so omitting and setting-false are *not* the same. |
| **No `structural_tag`** | Measured upstream to hang forever; `json_object` + `schema` constrains the same grammar. |
| **No `stream`** | The permanent engine wedge is on the streaming path: abandoning a `for await` never releases the per-model lock. |
| **Schema stringified once** | `response_format.schema` is declared `string` on 0.2.84, so an object there is a value the grammar compiler cannot read. A non-string or empty value surfaces as a call that never returns, so `buildCallParams` throws on it rather than letting it become somebody else's hang. |
| **`temperature: 0`** | Enforced by `resolveTier2Config`, which throws on any other value "for a reproducible bake-off". |
| **`launchPersistentContext`** | Spec §4.2: without it 3 of 4 cold loads fail with `QuotaExceededError`. |
| **Cancellation interrupts and drains, never `Promise.race`** | Spec §4.2 and `e59ca54`: abandoning a stream leaves the engine wedged and the *next* call does not return. `44e4273` records that interrupt-and-drain alone was insufficient — the engine latches — and the fix adds an interrupt-clear. |

### 2.5 The span-recovery ladder and the span-convention split

**The ladder** (`packages/tier2/src/spans.ts`, `2e9b268`; three plan defects recorded by
`9ea3994`; a further refusal added by `1540aba`). The model returns text; core demands
offsets. Core's own check (`orchestrator.ts:98`) only catches an *incoherent* finding,
never a *mis-located* one — a wrong span still slices cleanly and would send the
neighbouring word to the vault while the real secret stays in the message. The ladder:

- **Rung 1** — the folded quote occurs in the passage exactly once. The strong case.
- **Rung 2** — it does not occur, but a prefix of it does, exactly once, after peeling
  one code point at a time off the tail, stopping at `MINIMUM_CANDIDATE_WORDS = 3`.
- Anything else is refused, and counted (`unresolvedQuotes`).

`MINIMUM_CANDIDATE_WORDS` is 3 because Plan 5's feasibility research measured one-word
capitalized quotes non-unique 22–51 % of the time at a 1,000-character window, two-word
7–38 %, three-word about 0 % (`spans.ts:19–22`).

**The span-convention split** (`47ba8f6`, reviewed by `1540aba`). Locating and acting
want opposite things: locating wants a long quote (uniqueness), acting wants a short
span (`applyActions` rewrites by offset). The conflict was counted, not argued
(`spans.ts:23–33`): over every `pred:` finding in `runs/slate-p-fin-02.*.jsonl` (208
records, 16 arms) there were 13 such findings, and **five spanned the entire message** —
`pos-client-name-prose` [0,74) of 74 on `tier2-Phi-4-mini` and its `tier2only-` twin,
`pos-pan-prose` [0,48) of 48 on three more. Since the action on
`pred:client-relationship-disclosure` is `redact`, a [0,74) span leaves the whole
message as `[REDACTED:pred:client-relationship-disclosure]`.

So the model now returns **two** spans: an enclosing **evidence clause** which the
ladder places in the passage, and a **mention** inside it which the action covers.
`Finding.start/end` carries the mention. `resolveMention` searches only inside the
already-placed clause.

**What the split did and did not buy, from the same module's own measurement**
(`score.ts:46–74`), summed over all 16 arms and 208 records of each slate:

```
slate-p-fin-02 (one-span ask)   exact tp 0 fp 13 | overlap tp 3 fp 10 | iou50 tp 0 fp 13
slate-p-fin-03 (two-span ask)   exact tp 0 fp 11 | overlap tp 0 fp 11 | iou50 tp 0 fp 11
```

Overlap true positives fell 3 → 0. Whole-message `pred:` spans fell 5 of 13 → 1 of 11 —
that is the defect it was made for. **It should not be written up as improving
detection coverage; on this evidence it did not.**

`wholeClauseMentions` counts the escape hatch where the model returns the clause as its
own mention. MEASURED HERE by summing the counter across run records:

- `runs/slate-p-fin-03.*` (13-item smoke corpus): 15 of (rung1 34 + rung2 0) = **44.1 %**
- `runs/slate-rebuild-01*` (189-item corpus): 236 of (rung1 667 + rung2 4) = **35.2 %**

`unresolvedMentions` is 0 across `slate-p-fin-03` and 53 across `slate-rebuild-01*`.

### 2.6 Message-scope judging and the partition rule

`a82b22b`, reviewed by `82a5fce`. A `SemanticPredicate` declares `scope: "segment"` or
`scope: "message"`. `policies/compiled/p-fin.ir.json` declares exactly one predicate,
`client-relationship-disclosure`, with `scope: "message"`.

**The two scopes PARTITION the predicate list.** The message call is not an addition to
unchanged per-segment calls; a message-scoped predicate is removed from the segment
loop entirely. The plan's Task 13 entry gives the reasoning and the consequence: on
`p-fin` the compiled arm makes **one call per message, not `selectedSegments + 1`**.
The message call goes first, because it is exactly one call known before the run.

The effect, from the run artifacts (`runs/slate-p-fin-02.gates.jsonl`,
`tier2-Qwen3.5-2B` row): `judgedUnit: "message"`, `judgedUnitsPerItem` p50 = p95 = max
= 1, `ladder.messageScopeCalls` 13 over 13 items, `ladder.unitsJudged` 13,
`degradedNotices["scope-unjudged"]` **0**. The plan records the before/after:
`scope-unjudged` went from 3-of-3 on both compiled families to 0 on all four.

A both-scopes policy would pay `selected + 1` calls. **No such policy exists in this
repository, so that cost is unmeasured.**

---

## 3. Measurement infrastructure

This section exists so a reviewer can see that the harness cannot report a dead
browser, a skipped tier or an unanswered model as a result.

### 3.1 The five run gates

`apps/eval/src/driver/bakeoff.ts`, `GATES` at line 101 (pinned by `364b939`, amended by
`d851131` and `1770229`).

| Gate | Threshold | Derived? |
|---|---|---|
| `p95-ttft` | ≤ 1500 ms | Derived from a measured prompt size: driving the real `WebLlmJudge` over this corpus's median and largest selected segments with a capturing engine gives a whole prompt of **1,031 characters at the median segment and 1,122 at the largest**, of which 776 is the fixed system turn. |
| `decode-rate` | ≥ 25 tok/s | Derived from the hardware: `manifest.ts`'s four measured rates are 40 / 30 / 25 / 24, and 25 is the third of the four. The docblock notes this "kills Phi-4-mini on the manifest's own number before the arm runs." |
| `resolvable-rate` | ≥ 0.8 | **A CHOSEN floor, not a derived one**, and the docblock says so: spec §4.2 asks for "spans resolvable at rung ≤ 2" and gives no number, and no run had produced a distribution to set one from. |
| `duplicate-rate` | ≤ 0.9 | Bracketed by two measured duplicate behaviours: above Qwen3.5-2B's observed 0.667 ("found only the AWS key, three times over") and below Phi-4-mini's ~0.95 quote loop. The plan's own 0.5 had no derivation and would have killed the arm it recommends as primary. |
| `non-empty-after-stop` | predicate | Fires only if something already stopped an item. |

### 3.2 Derived minimum sample sizes

`minSampleForOneContrary` (`bakeoff.ts:1346`) computes, for each rate gate, the
smallest sample at which **one** contrary observation does not by itself decide the
verdict. Quoting the derivation (`bakeoff.ts:1327–1334`):

> COMPUTED from the gate's own threshold, not chosen: it searches for the first `n` at
> which the gate's own `passes` accepts the rate that a sample of `n` with exactly one
> contrary observation produces. Solving the two inequalities by hand gives
> `n >= 1/(1 - floor)` and `n >= 1/ceiling`, so 0.8 gives 5 and 0.9-as-a-ceiling gives
> 2; the search is used instead of the closed form because `1 / (1 - 0.8)` in doubles
> is 5.000000000000001, and because the search uses the gate's ACTUAL comparator.

It is explicitly **not** a confidence interval and makes no distributional claim.
Observed values in the artifacts: `p95-ttft` and `decode-rate` carry `minSample: 1`;
`resolvable-rate` carries 5; `duplicate-rate` carries 2; `non-empty-after-stop` carries
1 (read from `runs/slate-p-fin-02.gates.jsonl`).

**This mechanism has a visible before/after in the artifacts.** In
`runs/slate-p-fin-01.gates.jsonl`, `minSample` is `null` on every gate and the two
compiled Ministral arms are killed by `resolvable-rate` on a sample of **two** quotes:
*"1 of 2 quote(s) resolved to a span … a rate of 0.500 against a floor of 0.8"*. In
`slate-p-fin-02` and `-03` the same observation is reported as
`verdict: "not-measured"` with `sample: 2, minSample: 5`, and `killedOnRunGates` is
`false`. The gate is the same; what changed is that a verdict is now withheld where it
would be a statement about one observation.

### 3.3 `killedOnRunGates` is not an accuracy verdict

Renamed from `killed` deliberately. `scoring.verdictMeans` ships on every gates row and
states it (quoted from `runs/slate-p-fin-02.gates.jsonl`):

> `killedOnRunGates` is a THROUGHPUT AND HYGIENE verdict and is not a selection between
> models. It is the disjunction of the gates on this report, and every one of them is a
> property of the run […] None of them reads a gold label, so none of them can say
> whether this arm was RIGHT. […] An arm can pass every one of them and be the worst
> model on the slate.

`bakeoff.test.ts` pins this as a behaviour: two arms with byte-identical throughput,
one whose findings match its gold and one whose identical findings match nothing, get
the same value.

**The artifacts show the sharpest form of this.** In `runs/slate-p-fin-02.gates.jsonl`,
six Approach-B arms answered **zero** calls (`answeredCalls: 0`,
`degradedItems["budget-exhausted"]: 13` of 13 items, `ladder.messageBudgetExpiries: 13`)
and their `killedOnRunGates` is **`false`** — every gate reports `not-measured`, so the
disjunction is empty. An arm that produced nothing at all is not "killed". Reading
`killedOnRunGates` as a ranking would put those six above the two B arms that did
answer.

### 3.4 The `scoring` boundary

`ArmScoringBoundary` (`6e39bdd`; note the commit's own subject overclaims — see §8.5)
sits on every gates row. `accuracyGated` is `false` and stays false. It names, from the
arm's own rows: which tiers ran, how many gold spans exist per tier, which tiers carry
gold that this arm did not run, and which gold entity types the IR does not declare.

On the smoke corpus this produces three `cannotScore` sentences, quoted from
`runs/slate-p-fin-02.gates.jsonl`:

1. *"this arm RAN tier 2 and the 13 row(s) here carry no gold span at that tier, so a
   scorer joining findings to gold has nothing for a tier-2 finding to match: every one
   of them counts as a false positive INCLUDING every correct one, and an arm that found
   nothing at tier 2 scores as the most precise."*
2. *"the 13 row(s) here carry 2 gold span(s) at tier 1 (client-name) and this arm did
   NOT run tier 1 … a recall taken over `record.gold` is bounded above by 3/5 here
   whatever the model does."*
3. *"gold entityType(s) aws-key, generic-secret are not declared by the IR this arm
   ran."*

The stored `goldSpansByTier` for the smoke corpus is `{0: 3, 1: 2, 2: 0}` — five of the
fixture's seven gold spans are tier-assignable; the other two (`aws-key`,
`generic-secret`) belong to no tier because the IR does not declare them.

### 3.5 The `degraded` channel and its five reason words

`packages/core/src/detect/types.ts:86–133` (`58f86e7`, pinned by `c15a1cd`). The union
is deliberately split in two so an engine cannot assert a fact only the orchestrator
knows.

**Engine words** — what happened inside one engine's own run:

- `failed-closed` — the tier ran and refused to answer for part of the message (spec
  §7: a tier-2 body still invalid after one repair retry is flagged for user review and
  never passed through).
- `call-budget-exhausted` — one of the engine's *own* calls did not answer inside the
  per-call budget the engine holds. **Not** `ir.latencyBudgetMs`; a run can carry this
  word with almost all the message budget unspent.

**Orchestrator words** — facts about the calls `detect` itself made:

- `budget-exhausted` — `ir.latencyBudgetMs` for this **message** ran out, whether the
  tier was cut short or never started.
- `absent` — the tier was not enabled in this `TierConfig`. Not a failure, and
  deliberately not spelled as one.
- `scope-unjudged` — the policy declares predicates in a scope the judge did not
  evaluate.

The two budget words are separate because they count different events against different
denominators: *"One word for both would report a model with a tight per-call budget as
violating the spec 5.3 MESSAGE budget it never touched."*

Every gates row carries both `degradedNotices` (notice count) and `degradedItems` (item
count) keyed by all five words, so an empty `findings` array can always be attributed.

### 3.6 What the run records do NOT carry

**No run record names a code revision, and none carries a prompt hash.** The record
schema (`RunRecordSchema`, `schemaVersion: 1`) has `arm`, `detector`, `config`,
`backend`, `provider`, `policy`, `policyHash`, `irHash`, `degraded`, `timings`,
`findings`, `gold`, `text`, `tier2Config` and `tier2Stats`/`baselineStats` — and no
commit, no build id, no prompt digest.

This is not hypothetical. Two consequences are visible in the artifacts:

1. `runs/slate-p-fin-02.gates.jsonl` carries `minSample` on every gate — a field
   introduced by commit `1770229` (2026-09-01 17:46) — while the file's mtime is
   2026-09-01 17:01. The runs were taken from a **working tree**, not from a commit.
2. `slate-p-fin-02` and `slate-p-fin-03` have **byte-identical `run` blocks** (same
   `runId` aside, same `irHash`, `policyHash`, `latencyBudgetMs`, `itemTimeoutMs`,
   `uncertainBelow` and `tier2Config`), yet the compiled arms' p50 prompt tokens are
   280 in `-02` and 361 in `-03`, and B's are 1,410 and 1,494 — a constant +81/+84
   shift. Neither of the two commits between the two runs (`1770229`, `975b5c2`)
   touches a prompt file. **The cause of that shift cannot be recovered from the
   artifacts.**

A reviewer should treat run-to-run comparisons across these slates as comparisons
between two unrecorded code states.

---

## 4. Corpus

### 4.1 The smoke fixture

`corpora/fixtures/smoke.jsonl` — **13 items** (MEASURED HERE: `wc -l`). Six positives,
seven negatives, **7 gold spans**: 3 `in-pan`, 2 `client-name`, 1 `aws-key`, 1
`generic-secret`. Labelled under policy `minimal-fixture`, which is *not* `p-fin`, so
no accuracy number over `record.gold` is derivable from it in either direction.

`corpora/fixtures/smoke.gold-tier2.jsonl` — 13 rows, one per item, all `status:
"scored"`, **2 positives** (`pos-client-name-prose`, `pos-multiline-pan-and-client`)
and 11 negatives, labelled against `policies/p-fin.md` §3 with the policy hash
`ebb3cd68…`. Added by `975b5c2`.

### 4.2 The injection corpus v2

`corpora/generated/injection-p-fin-v2.jsonl` — **189 items** (MEASURED HERE), 108
positives, 81 negatives, **108 gold spans** and 219 confusable (`neg:`) spans. Stored
counts in `injection-p-fin-v2.manifest.json` `.counts` agree exactly with the file.
sha256 `25e3a3e0c439fe0f8c6a1df787d64bfbd9dc0745f85687f455d4581eade6cf12`, 878,475
bytes. Generator `sih-injection-corpus` v1, seed `sih-p7-unleaked-v2`.

Gold spans by type (stored, `.counts.goldSpansByType`): `client-name` 19,
`private-key-material` 18, `bank-account-identifier` 17, `api-credential` 16,
`in-aadhaar` 11, `internal-customer-id` 11, `db-connection-string` 9, `in-pan` 7.

Confusable spans by type: 27 `neg:` families, largest `neg:dh-parameters-block` 19,
`neg:tan` 15, `neg:package-spec` 13, `neg:sftp-endpoint` 13.

**Per-type resolution is quantised** (stored, `.counts.perTypeResolution`): the scarcest
type (`in-pan`) has 7 gold spans, so its recall moves in steps of 14.3 percentage
points, and **no per-type difference smaller than that is resolvable on this corpus.**

#### The injection invariant and how it is enforced

Stored, `.injection.invariant`:

> every gold span is the span the generator wrote, derived from the splice offset and
> never re-found by search; `text.slice(start, end) === the injected value`, and each
> injected value occurs in the text exactly as many times as it was injected

Checked over 189 items and 327 spans. This is what breaks the two-positive ceiling of
hand-labelling — but it works only for **identifiers**. A message-scope predicate cannot
be spliced in, so predicate gold still needs blind judgement per item (§4.4).

`carrierSource: "hand-authored"` — the carriers are **not** the ShareChat/WildChat
conversations spec §6.2 specifies.

#### Certification: one stage real, two named gaps, `NOT CERTIFIED`

Stored, `.certification`:

- `claim: "NOT CERTIFIED"`
- `stagesRun: ["tier0-sweep"]`
- `stagesUnrun`: **`high-recall-model-sweep`** (blocked on a Python `corpora/`
  toolchain with GLiNER2-PII and `gliner-pii-large` checked out — this repository has
  no Python package and no checkout of either model, and substituting `@sih/tier1`
  would violate the stage's own requirement since tier 1 is an arm under test) and
  **`frontier-adjudication`** (blocked on a live frontier-model call, deferred by the
  user since 2026-08-18, and on compiled `p-med`/`p-corp` IRs — two of the three
  policies have never been compiled, so the "union of all three policies" the spec
  requires does not exist as an artifact).
- Carriers: 57 total, **0 certified-clear**, 27 provisional-clear, 30 quarantined.
- Supplementary sweeps that did run: `blind-double-adjudication-p-fin`,
  `format-spec-sweep`, `orthographic-org-sweep`.

**The circularity is measured rather than merely disclosed.** Stage 1 is `runTier0` over
a widened copy of the scoring IR — i.e. the tier-0 arm under test. `.certification.
circularity` counts the split against the IR-free `format-spec-sweep`: `stage1Only: []`,
`independentOnly: []`, `both: [d01-email, d03-kv, d04-pan-shaped, d05-aadhaar-digits,
d06-ifsc-shaped, d07-entropy-fence]`. So on this corpus **no carrier was quarantined by
the arm under test alone**; all six quarantine decisions the two sweeps could both see
agreed.

#### The five leakage measurements, before and after

`apps/eval/src/corpus/leakage.ts` defines all five. **RE-COMPUTED HERE** by calling
`measureOrthography`, `measureBoost` and `measureRoles` over both corpora *(derivation
script)*, using the wave-2 paired-type map that `corpus-v2.test.ts` commits for the
before corpus:

| Measurement | Before (`injection-p-fin-adjudicated-v1.jsonl`, 189 items, 108 gold) | After (`injection-p-fin-v2.jsonl`, 189 items, 108 gold) |
|---|---|---|
| **1. Gold spans the oracle solves uniquely** | 58 of 108 | **0 of 108** |
| **2. Orthographic oracle, scored as an arm is** (overlap, max one-to-one matching) | P 0.492 R 0.880 F1 0.631 (tp 95, fp 98, fn 13) | P **0.307** R **0.870** F1 0.454 (tp 94, fp 212, fn 14) |
| **2b. Oracle budget-matched** (told each item's gold-span count, first N hits in document order) | P 0.931 R 0.880 F1 0.905 | P **0.705** R **0.685** F1 0.695 |
| **2c. First oracle hit is the gold span** | 95 of 108 | **74 of 108** (68.5 %) |
| **3. contextBoost, as `runTier0` reads it** (window 40, substring, span included) | gold 51.9 % vs confusable 6.5 %, delta **+0.4537**; own-type delta **+0.3981** | gold 50.0 % vs confusable 42.5 %, delta **+0.0753**; own-type delta **+0.0979** |
| **3b. contextBoost, wide word-boundary reading** (window 70, span excluded) | delta +0.3981, own-type +0.3333 | delta **−0.0075**, own-type +0.0242 |
| **4. Organisation names locked to one role class** | 19 of 39 org spans | **0 of 59** |
| **4b. Name-only classifier accuracy / majority baseline / lift** | 0.795 / 0.641 / +0.154 | 0.712 / 0.678 / **+0.034** |
| **5. Confusable families 1:1 with an IR counterExample surface** | 13 of 24 (this one figure is not in a manifest: it is stated in the module docblocks at `apps/eval/src/corpus/families.v2.ts:65` and `apps/eval/src/corpus/leakage.ts:776`, and I did not re-derive it — the check is a surface-name comparison against a hand-written table, not a function over the corpus) | **0 of 28** (stored, `.leakage.irOverlap.confusableFamiliesSharingAnIrCounterExampleSurface` is `[]` against `confusableFamilies: 28`) |

The corpus carries 23 IR counterExamples and 38 contextBoost terms (MEASURED HERE via
`irCounterExamples(ir).length` and `boostTerms(ir).all.length`).

**The honest note on contextBoost symmetry.** The manifest's `.leakage.boost.verdict`
reads `NOT SYMMETRIC` under the tier-0 reading, and it is right to. `runTier0` really
does implement contextBoost at window 40, substring match, with the span's own text
inside the window (`packages/core/src/detect/tier0.ts` `hasNearbyKeyword`). Measured that
way the residual is **+0.0753 any-term and +0.0979 own-type, in the compiled arm's
favour** — small, but not zero. The earlier published −0.0075 was taken at window 70 with
word-boundary matching and the span excluded, which is not the mechanism the detector
implements. **Any per-entityType accuracy number off this corpus carries a keyword
advantage of that size for an arm that reads contextBoost.**

**An open leak the rebuild did not close.** `.leakage.position`: the shape-matched
distractor is placed after the span it shadows on **81 of 105 pairs (77.1 %)**, and on
**84 of 108** items carrying both a gold and a confusable span the *earlier* span is the
gold one (77.8 % against a 50 % coin flip). The manifest calls this "AN OPEN LEAK" —
position substitutes for the orthographic discriminator the distractor removed. It is
not fixed because fixing it moves every offset and the labels a blind round wrote are
about these offsets.

#### The certification that was gamed by construction

Stored, `.leakage.orthography` docblock and `.unvalidated[0]`: `goldSpansSolvedByOracle
= 0` was the statistic that certified "the gold span is not the unique orthographic
outlier", and **it cannot carry that claim**. Injecting one same-shape distractor per
positive drives it to zero *by construction* — a second oracle hit anywhere in the
message makes the gold span not-alone regardless of whether the distractor is as odd,
as plausible, or even in the same clause. This was demonstrated in `corpus-v2.test.ts`
by padding the *old* corpus and watching its solved rate hit 0 with recall unchanged.
The statistic is kept as a measurement and re-documented as certifying nothing. The
replacement is the budget-matched oracle and `firstHitIsGold`, both of which a
distractor cannot flatter.

#### Effective N and the 12-fragment leak

Both are properties of the *predicate* side and are covered in §4.4 and §4.5.

#### Contamination against the compiler self-test

Stored, `.contamination`: 8-gram containment against 723 self-test examples from 24
sources, threshold >0.7, `itemsChecked: 189, itemsKept: 189, dropped: []`,
`maxScoreKept: 0`. The manifest immediately qualifies its own zero: 115 examples are
unscoreable, 148 were scored at their own length, 300 are matchable only through their
own identifier, and a separate phrase-overlap measurement reports the longest contiguous
shared word run per item — histogram 1:1, 2:77, 3:70, 4:36, **5:5**. The five worst
share a five-token stem with `policies/compiled/p-fin.selftest.json` ("the permanent
account number on", "the servicing console shows cif") at an 8-gram containment of
0.000. Reported, not dropped on.

### 4.3 Splits

Stored, `.splits`: `devFraction 0.2`, **39 dev items and 150 test**, `disjoint: true`.
The manifest's own note is the important part and is quoted rather than paraphrased:

> That is DETERMINISM and it is NOT what spec 6.2 means by 'test frozen before any
> tuning' … `measureLeakage` runs over all 189 items, dev and test pooled … and
> families were removed and rewritten on what those pooled measurements said. … Every
> arm run so far has also SCORED all 189 with no split on any row, so a pooled number
> cannot be reduced to a test-only one after the fact.

VERIFIED HERE: no field named `split` appears on any record in `runs/slate-rebuild-01*`.

### 4.4 The two blind annotation rounds

Both rounds are recorded as artifacts, not as prose. Read
`injection-p-fin-v2.labelled.manifest.json` (round 1) and
`injection-p-fin-v2.predicate-round.json` (round 2).

#### Round 1 — `v2-blind-double-labelling-p-fin` (`5ce1e34`)

Two annotators, A and B, each given `policies/p-fin.md` and the queue, forbidden the
families file, the manifests, the compiled IR, the gold fixtures, `runs/` and the git
log.

**Coverage.** The queue holds **209 rows**: 189 message-predicate questions plus 20
contested-span questions naming items that also carry a message-predicate row. Both
annotators answered all 20 contested-span rows; **neither answered any span-less row**,
because the brief defined the unit of work as "a message plus one highlighted span" and
169 of the 189 items have no span. So `itemsAnswered: 20`, `itemsUnanswered: 169`.

**Agreement (stored, `.agreement`).**

| Field | n | agreements | raw | Cohen's κ | note the artifact attaches |
|---|---|---|---|---|---|
| `spanLabelCorrect` | 20 | 13 | 0.65 | **0** | DEGENERATE — A used a single category, forcing chance agreement to equal observed agreement and κ to exactly 0 whatever B answered. "0 here does not mean 'no better than chance', it means kappa cannot see this table." Marginals: A 20 true / 0 false; B 13 true / 7 false. |
| `satisfiesPredicate` | 20 | 20 | 1.00 | **undefined** | Both used a single category, so chance agreement is 1. "The raw agreement of 1.0 is agreement on one class only and says nothing about how reliably the other class can be labelled — this round contains no instance of it." |
| `predicateConfidence` | 20 | 19 | 0.95 | 0.643 | Skewed (smallest cell 1 of 20), so κ is unstable; one reclassified item moves it materially. |

The artifact adds a caveat over all three: the two annotators read the same queue rows,
which carry the generator's own two-sided argument and its scoping instruction, so
*"their agreement is agreement within a shared framing"*, and both were given the same
directional prior about the answer key, which inflates agreement where both yielded.

**Adjudication outcome.** `neg:sftp-endpoint` (13 spans) affirmed and moved into the
false-positive claim; `neg:retrieval-reference` (7 spans) left **disputed** and excluded
from both the match set and the false-positive set. Predicate side: 19 scored, 1
disputed, **0 positives**.

**Recorded blindness breaches (stored, `.round.blindness.channels`).** Four channels
audited; `read` and `told` marked `"partial"`.

- **Read.** Both annotators opened `injection-p-fin-v2.jsonl`, which carries gold and
  `meta` — *forced by the task*, because the queue deliberately omits the proposed span
  type and the corpus file is the only place to recover it. A additionally probed both
  superseded corpora programmatically; B's orienting `head -c 2000` of the queue
  rendered two items in full, one of which is an uncovered NDA-party positive.
- **Told, TOLD-1 (the serious one, and machine-checked).** The brief handed both
  annotators a paraphrase of the predicate containing the exact phrase held in
  `BRIEF_PARAPHRASE_FRAGMENT` (`apps/eval/src/corpus/labelling.ts:77`), **which is
  `policies/compiled/p-fin.ir.json`'s own `semanticPredicates[0].nlPredicate`** — the
  artifact the arm under test detects against, and on the forbidden list. The phrase
  occurs in the IR and in neither `policies/p-fin.md` nor `questions.ts`'s
  `PREDICATE_QUESTION`; `corpus-labelling.test.ts` asserts all three. Both annotators
  independently noticed the paraphrase drops §3.3's "parties under a non-disclosure
  agreement" and restored it; neither knew where the wording came from.
- **Told, TOLD-2.** The brief supplied a *directional* prior on the answer key — that
  several proposed types had been found contradicted by the policy, with a worked
  example — and that prior is **stale**: the family it describes (`csr-pem-block`) is in
  `REMOVED_FAMILIES` and is not in v2. Every one of the 20 proposals the annotators saw
  was of the shape the prior named. B states outright that the seven flips cannot be
  claimed uncontaminated by it.
- **Told, TOLD-3.** Distributional hints in both directions. Both annotators flagged
  them and returned a uniformly false predicate column anyway, "holding an answer the
  hint argued against".
- **Structural-1.** Item ids encode the stratum: `inj-hn*` vs `inj-o*` reads as
  hard-negative vs ordinary. 8 of the 20 labelled items carry the `hn` prefix.
- **Structural-2.** The queue's own `question` field names the span's semantic type in
  prose and pre-argues both readings with clause numbers attached, and bounds the
  predicate question to "sections 3.1, 3.3 and 3.4 only". `questions.ts`'s docblock
  claimed the queue "carries no family id, no constructedRole and no gold, so an
  annotator cannot read the generator's intent off the row". **These two disclosures
  falsify that claim, and the claim was in a committed manifest.**
- **Scope-1.** The scoping choice removed 89 % of the questions, and the artifact
  records it as a breach rather than a detail: *"a scoping choice that removes 89% of
  the questions is not a detail of the brief, it is the result of the round."*

#### Round 2 — `v2-blind-predicate-all-messages-p-fin` (`52cc75f`, `f3d2495`, `0edb3db`)

The queue was rebuilt to carry **only** `{rowId, text}` — opaque HMAC-SHA256 row ids
(first 16 hex characters, keyed by an uncommitted salt), a deterministic shuffle
constrained on carrier, and no family id, entity type, constructed role, carrier id,
carrier stratum or item id anywhere outside the message.

**Results (stored, `.answers`).** `rowsPut` 189; both annotators answered **189/189**;
`satisfiesAgreement` **1.0**, `satisfiesKappa` **1.0**; `confidenceAgreement` 1.0,
`confidenceKappa` 1.0; **19 positives**; **10 disputed**; 19 quotes returned by each
annotator, all verbatim from the policy.

**What the round record says κ = 1.000 does and does not mean**, quoted from
`.answers.effectiveDecisions`:

> the 19 positives are seven fixed carrier templates drawn over five organisation
> names … An arm that memorised seven strings scores here as if it had read §3, and a
> confidence interval computed at n=19 is computed on a unit that is correlated in
> groups of two and three. Raw agreement and kappa of 1.0 are facts about two readers
> of a formulaic corpus, not evidence that §3 is well defined.

**Effective N is 7.** VERIFIED HERE *(derivation script)*: all 19 positives match one of
the seven `PREDICATE_DISCRIMINATORS.positive` fragments, with counts
{"the party we signed the confidentiality agreement with is": 3, "the onboarding pack
for": 3, "the fee note we send to": 3, "the first of those two is the one we act for":
3, "we are on the other side of that trade from": 3, "the mandate we hold for": 2, "we
are pitching": 2} = 19, over exactly the five names of `ORG_POOL` (Marrowfield Group,
Ashcombe Holdings, Tarnwick Industries, Lensfield Trading, Bexmoor Associates).

At framing level, 7 of 7 correct supports a one-sided 95 % lower bound of
0.05^(1/7) = **65.2 %** (COMPUTED HERE, rule of three).

**Blindness (stored, `.blindness.channels`).** `structural` and `order` are `audited`
and clean — every property of the queue is recomputed from its bytes, and same-carrier
adjacency is 0 of 188 adjacent pairs in queue order against 162 in source order. But:

- **`read` is UNAUDITED, and that is a regression.** Neither annotator was asked for a
  `filesRead` list and neither returned one. Round 1 collected both.
- **`told` is UNAUDITED.** No brief retained, no `wasToldAbout` disclosure. The record
  goes further and withdraws a claim made in the commit that shipped the gold: *"The
  commit that shipped this gold (f3d2495) states that two told-channel leaks remain and
  that both annotators flagged them unprompted … NOTHING in this repository or in the
  round's retained artifacts records those flags … so that claim is unsupported and is
  withdrawn here rather than repeated."* And: *"If the brief did state a positive count,
  both annotators returning exactly 19 is what that would produce, and no evidence
  separates the two explanations."*
- **`answer-key` is `audited`, and the finding is total.**
  `meta.predicateConstruction.constructed` equals the adjudicated `satisfies` on
  **189 of 189** rows, and each of the 19 gold spans is byte- and offset-identical to
  that item's own `client-name` gold entry. VERIFIED HERE: the corpus holds exactly 19
  items carrying a `client-name` gold span, and those are the 19 positives. So a `pred:`
  score over this gold measures the same target as a `client-name` span score and
  **must not be reported as an independent check on the semantic judge.**

### 4.5 The 12-fragment leak disclosed in `families.v2.ts`

`apps/eval/src/corpus/predicate-round.ts:116` defines `PREDICATE_DISCRIMINATORS`: **7
positive fragments and 5 negative fragments = 12**. All twelve appear verbatim in
`apps/eval/src/corpus/families.v2.ts`, beside a `constructedRole` and a
`labelBasis.reading` naming the clause and the intended answer. The round record states
the consequence flatly:

> An annotator with ordinary repository read access can label all 189 rows without
> opening the policy.

A second, sharper instance: `apps/eval/src/corpus/labelling.ts`, committed *before* the
round, quotes `inj-o01-1`'s "the party we signed the confidentiality agreement with is
Marrowfield Group" and calls it a clean §3.3 NDA-party hit. `inj-o01-1` is one of the
round's 19 positives, named by id and by organisation.

Closing this needs a regenerated corpus and a re-run round.

### 4.6 The corpus's own list of what it cannot support

`.unvalidated` in the manifest is fifteen entries long. The load-bearing ones not
already covered above:

- **Every minted value is gated by the shipping validators at mint time.** The corpus
  can contain no real PAN that `pan-structure` rejects and no non-PAN that it accepts,
  so per-type tier-0 recall on `in-pan`/`in-aadhaar` and tier-0's false-positive rate on
  the PAN confusable are **upper bounds the corpus manufactured**, not measurements of
  the world. A prompting arm reasoning from the policy text gets no such floor.
- **The contested set was chosen by the author of the labels.** `CONTESTED_TYPES` is
  exactly the families whose own author wrote a `contestedBy` string — 2 of 28. The
  blind round therefore covered 20 of 219 confusable spans. In particular
  `neg:org-vendor` (11), `neg:org-landlord` (6), `neg:org-cross-segment-supplier` (3)
  and `neg:org-cross-segment-landlord` (6) sit on exactly the client/non-client axis
  `client-name` is scored on and **were never put to an annotator**. `client-name`
  precision rests on 26 author-only labels.
- **`neg:batch-sequence` (9 spans) is the structural twin of a type the round left
  disputed.** Both it and `neg:retrieval-reference` mint twelve digits with a lead digit
  2–9 and a valid Verhoeff check, so all 16 satisfy the IR's written `in-aadhaar`
  definition and the compiled tier-0 arm fires on every one **by construction**. The 7
  retrieval references are excluded from the false-positive claim; the 9 batch sequences
  are scored as true false positives on the strength of one authored clause.
- **Several confusable families carry their exculpation as a first-person assertion
  inside the message** — "nothing sensitive was in the paste", "which is the firm's own".
  `p-fin` forbids exactly that inference elsewhere (§4.1, §2.5), so on roughly a fifth of
  the confusable surface **an arm that believes the sender scores as precise**, and the
  over-blocking rate is flattered by that much.

### 4.7 What the corpus can and cannot rank

`injection-p-fin-v2.labelled.manifest.json` `.canSupport.verdict`, quoted:

> THIS CORPUS CAN SUPPORT A COARSE MODEL RANKING ON ENTITY-TYPE SPAN RECALL AND ON
> OVER-BLOCKING, AND CANNOT SUPPORT ONE ON THE PREDICATE.

Two structural ceilings recorded there, both of which bound *every* compiled arm
regardless of what it detects:

- **`private-key-material`: 0 of 18 under `exact` and `iou50`.** Gold is the whole PEM
  block; `p-fin`'s one rule for the class matches only the BEGIN header. Under
  `overlap`, reach is 12 of 18. *"a table column using it reports a span convention and
  not a detector."*
- **`bank-account-identifier`: 11 of 17 reachable under `exact`** (14 under `iou50` and
  `overlap`).

And `.canSupport.predicate`: that round contributed 0 positives and 19 scored negatives,
so predicate recall is **undefined on that file alone**; pooling with the smoke fixture's
2 positives still leaves 2, quantising recall to steps of 50 percentage points.

---

## 5. Results

**The floors are in the same tables as the arm figures throughout. This is not a
stylistic choice.** `scoreArm` emits `floors` beside `byRule` and adds a caveat naming
every rule an arm fails to beat, precisely so the comparison cannot be dropped
downstream (`score.ts:738–748`, `:946–966`).

### 5a. Latency and throughput on the smoke corpus

Source: `runs/slate-p-fin-02.gates.jsonl` (13-item smoke corpus, real compiled `p-fin`
at the compiler's own 5,000 ms message budget — `run.latencyBudgetTimesCompilerDefault`
is 1 on every row). Extracted with `jq` over the gates file.

| Arm | Family | Items | Answered calls | Prompt tok p50 | p95 TTFT (ms) | Sustained decode (tok/s) | Items budget-exhausted | `p95-ttft` | `killedOnRunGates` |
|---|---|---|---|---|---|---|---|---|---|
| tier2-Qwen3.5-2B | compiled | 13 | 13 | 280 | 566.11 | 47.09 | 0 | **pass** | false |
| tier2only-Qwen3.5-2B | compiled-t2-only | 13 | 13 | 277 | 556.24 | 47.73 | 0 | **pass** | false |
| baselineB-Qwen3.5-2B | baseline-b | 13 | 12 | **1410** | **2735.88** | 37.81 | 1 | **fail** | true |
| baselineB+tier0-Qwen3.5-2B | baseline-b-tier0 | 13 | 12 | **1424** | **2866.09** | 37.60 | 1 | **fail** | true |
| tier2-Ministral-3-3B | compiled | 13 | 12 | 267 | 1010.01 | 33.86 | 1 | **pass** | false |
| tier2only-Ministral-3-3B | compiled-t2-only | 13 | 13 | 267 | 927.03 | 34.96 | 0 | **pass** | false |
| baselineB-Ministral-3-3B | baseline-b | 13 | **0** | — | — | — | **13** | not-measured | false |
| baselineB+tier0-Ministral-3-3B | baseline-b-tier0 | 13 | **0** | — | — | — | **13** | not-measured | false |
| tier2-Qwen3-4B | compiled | 13 | 13 | 266 | 1244.00 | 27.86 | 0 | **pass** | false |
| tier2only-Qwen3-4B | compiled-t2-only | 13 | 13 | 261 | 1133.86 | 28.23 | 0 | **pass** | false |
| baselineB-Qwen3-4B | baseline-b | 13 | **0** | — | — | — | **13** | not-measured | false |
| baselineB+tier0-Qwen3-4B | baseline-b-tier0 | 13 | **0** | — | — | — | **13** | not-measured | false |
| tier2-Phi-4-mini | compiled | 13 | 13 | 257 | 928.82 | 30.71 | 0 | **pass** | false |
| tier2only-Phi-4-mini | compiled-t2-only | 13 | 13 | 252 | 920.65 | 31.02 | 0 | **pass** | false |
| baselineB-Phi-4-mini | baseline-b | 13 | **0** | — | — | — | **13** | not-measured | false |
| baselineB+tier0-Phi-4-mini | baseline-b-tier0 | 13 | **0** | — | — | — | **13** | not-measured | false |

**The compiled-4/4 vs B-0/4 finding, stated precisely.** The compiled families pass the
`p95-ttft` gate on **4 of 4 models** (p95 TTFT 556–1244 ms against a 1,500 ms ceiling).
Approach B passes on **0 of 4**. But the two halves of that sentence are not symmetric,
and the asymmetry matters:

- On the **fastest** model (Qwen3.5-2B) B answered **12 of 13** items and **failed** the
  gate at 2,735.88 ms (its `+tier0` twin at 2,866.09 ms).
- On the **other three** models B answered **zero** calls: `budget-exhausted` on 13 of
  13 items, `messageBudgetExpiries` 13, every gate `not-measured`, and — see §3.3 —
  `killedOnRunGates: false`. **B did not fail those gates; it produced nothing for them
  to rule on.**

**Mechanism.** B's prompt is 1,410 p50 tokens (it carries the whole 5,320-byte policy)
against the compiled judge's 252–280. At `p-fin`'s real 5,000 ms message budget, that
prompt does not reach a first token on three of four models.

**Reproducibility across the three smoke slates.** `slate-p-fin-01` (same code state as
`-02`) gives compiled p95 TTFT 560.42–1260.93 ms and the same 4/4 vs 0/4 pattern.
`slate-p-fin-03` (a later, unrecorded code state — see §3.6) shifts every prompt up by
~81 tokens and gives compiled 687.65–1467.79 ms, still 8 of 8 compiled rows passing, and
B at 2,889.41 / 3,050.26 ms on Qwen3.5-2B, zero-answering on the other three.

**On the 189-item corpus the picture changes, and this must not be omitted.** From
`runs/slate-rebuild-01*.gates.jsonl` (same 5,000 ms budget, longer messages), only the
two **Qwen3.5-2B compiled** arms still pass `p95-ttft`:

| Arm | Answered calls / items | Prompt tok p50 | p95 TTFT (ms) | Items budget-exhausted | `p95-ttft` |
|---|---|---|---|---|---|
| tier2-Qwen3.5-2B | 190 / 189 | 447 | 1184 | 0 | **pass** |
| tier2only-Qwen3.5-2B | 190 / 189 | 441 | 1200 | 0 | **pass** |
| tier2-Ministral-3-3B | 148 / 189 | 428 | 2144 | 45 | fail |
| tier2only-Ministral-3-3B | 172 / 189 | 426 | 1999 | 22 | fail |
| tier2-Qwen3-4B | 119 / 189 | 408 | 2010 | 70 | fail |
| tier2only-Qwen3-4B | 132 / 189 | 410 | 2463 | 57 | fail |
| tier2-Phi-4-mini | 186 / 189 | 415 | 2031 | 3 | fail |
| tier2only-Phi-4-mini | 187 / 189 | 410 | 1920 | 2 | fail |
| baselineB-Qwen3.5-2B | 116 / 189 | 1570 | 3227 | 73 | fail |
| baselineB+tier0-Qwen3.5-2B | 125 / 189 | 1582 | 3177 | 64 | fail |
| baselineB / baselineB+tier0, other three models | **0** / 189 | — | — | **189** | not-measured |

**So "compiled passes 4/4" is a fact about a 13-item corpus of short messages, and does
not transfer to the 189-item corpus, where it is 1 of 4 models.**

### 5b. Span-level accuracy on the injection corpus v2

Source: the 16 arms of `runs/slate-rebuild-01*.jsonl` (3,024 records = 16 × 189) scored
against `injection-p-fin-v2.jsonl`'s **108 entity gold spans** over all **189 items**.

**Definition, stated because it is load-bearing.** Scored **type-blind** (any finding
against any gold span), under the `overlap` rule with **maximum-cardinality one-to-one
matching**, using `scoreEveryRule` from `apps/eval/src/driver/score.ts` — the same
matcher `leakage.ts` uses for the oracle, which its own field labels *"overlap, maximum
one-to-one matching — the shape `apps/eval/src/driver/score.ts` uses"*. This makes the
arms and the oracle commensurable. MEASURED HERE *(derivation script)*.

| Reader / arm | Findings | tp | fp | fn | **P** | **R** | **F1** (overlap) | exact F1 | iou50 F1 |
|---|---|---|---|---|---|---|---|---|---|
| **FLOOR — orthographic oracle, unbudgeted** | 306 | 94 | 212 | 14 | **0.307** | **0.870** | **0.454** | — | — |
| **FLOOR — orthographic oracle, budget-matched** | 105 | 74 | 31 | 34 | **0.705** | **0.685** | **0.695** | — | — |
| baselineB+tier0-Qwen3.5-2B | 218 | 79 | 139 | 29 | 0.362 | 0.731 | **0.485** | 0.301 | 0.368 |
| baselineB+tier0-Ministral-3-3B | 200 | 72 | 128 | 36 | 0.360 | 0.667 | 0.468 | 0.351 | 0.370 |
| baselineB+tier0-Phi-4-mini | 200 | 72 | 128 | 36 | 0.360 | 0.667 | 0.468 | 0.351 | 0.370 |
| baselineB+tier0-Qwen3-4B | 200 | 72 | 128 | 36 | 0.360 | 0.667 | 0.468 | 0.351 | 0.370 |
| tier2-Qwen3-4B | 248 | 82 | 166 | 26 | 0.331 | 0.759 | 0.461 | 0.360 | 0.376 |
| tier2-Ministral-3-3B | 218 | 72 | 146 | 36 | 0.330 | 0.667 | 0.442 | 0.331 | 0.350 |
| tier2-Qwen3.5-2B | 255 | 79 | 176 | 29 | 0.310 | 0.731 | 0.435 | 0.298 | 0.314 |
| tier2-Phi-4-mini | 336 | 91 | 245 | 17 | 0.271 | 0.843 | 0.410 | 0.297 | 0.311 |
| baselineB-Qwen3.5-2B | 44 | 25 | 19 | 83 | 0.568 | 0.231 | 0.329 | 0.105 | 0.211 |
| tier2only-Phi-4-mini | 154 | 34 | 120 | 74 | 0.221 | 0.315 | 0.260 | 0.145 | 0.160 |
| tier2only-Qwen3-4B | 60 | 17 | 43 | 91 | 0.283 | 0.157 | 0.202 | 0.179 | 0.190 |
| tier2only-Qwen3.5-2B | 40 | 7 | 33 | 101 | 0.175 | 0.065 | 0.095 | 0.014 | 0.014 |
| tier2only-Ministral-3-3B | 21 | 1 | 20 | 107 | 0.048 | 0.009 | 0.016 | 0.000 | 0.000 |
| baselineB-Ministral-3-3B | **0** | 0 | 0 | 108 | undef | 0.000 | undef | undef | undef |
| baselineB-Phi-4-mini | **0** | 0 | 0 | 108 | undef | 0.000 | undef | undef | undef |
| baselineB-Qwen3-4B | **0** | 0 | 0 | 108 | undef | 0.000 | undef | undef | undef |

**Reading the table.**

- The **best arm on F1 is `baselineB+tier0-Qwen3.5-2B` at P 0.362 / R 0.731 / F1 0.485.**
  That arm's tier-0 half is doing most of the work: the three `baselineB+tier0` arms whose
  LLM answered zero calls (Ministral, Phi-4-mini, Qwen3-4B) score P 0.360 / R 0.667 —
  i.e. **tier 0 alone, with the LLM contributing nothing, reaches 0.468 F1.**
- Against the budget-matched oracle (P 0.705 / R 0.685 / F1 0.695), **no arm is close.**
  Against the unbudgeted oracle (P 0.307 / R 0.870), the best arm is +0.055 on precision
  and −0.139 on recall.
- The four `tier2only-` arms — the judge with no deterministic priors — are the weakest
  block, from F1 0.016 to 0.260.

**The "findings land on oracle spans" measurement.** I was told this reads 100 %. **It
does not, under any of the four natural readings.** MEASURED HERE *(derivation script)*,
counting a finding as "on an oracle span" when it overlaps one:

| Population | Pooled over all 16 arms | Per-arm range |
|---|---|---|
| All findings | 1,718 / 2,194 = **78.30 %** | 4.8 % – 91.0 % |
| Non-`pred:` findings only | 1,483 / 1,651 = **89.82 %** | 60.5 % – 91.8 % |
| `pred:` findings only | 235 / 543 = **43.28 %** | 0.0 % – 65.0 % |
| Findings that overlap a gold span | 805 / 919 = **87.60 %** | 57.1 % – 100.0 % |

(The second and third rows partition the first: 1,483 + 235 = 1,718, which is the
independently counted all-findings numerator, so the three are consistent.)

The strongest true statement the artifacts support is the contrast between rows two and
three: **the tier-0-driven findings are 89.8 % explicable as orthographic oddness, and
the tier-2 predicate findings are not** (43.3 %) — because the predicate targets
organisation names in prose, which the oracle's Title-Case bigram rule catches only
sometimes. The claimed 100 % is in §11.

**The near-miss pair table.** A "pair" here is an observed co-occurrence in one message
of a positive-family span and a confusable-family span. Scored **type-matched**: a
positive-side hit requires a finding whose `entityType` equals the gold span's type; a
negative-side hit requires a finding whose `entityType` equals `PAIRED_TYPE[negType]` —
the entity type that confusable family is a near miss *for*
(`apps/eval/src/corpus/families.v2.ts:1427`). MEASURED HERE *(derivation script)*;
denominators are (spans × 16 arms).

Every pair in which **both** sides are an organisation family:

| Positive family | Confusable family | Positive-side hits | Negative-side hits |
|---|---|---|---|
| client-org-cross-segment | client-org-cross-segment-landlord | 0 / 48 | 0 / 48 |
| client-org-fee-note | org-landlord | 0 / 16 | 0 / 16 |
| client-org-fee-note | org-listed-company | 0 / 16 | 0 / 16 |
| client-org-fee-note | org-vendor | 0 / 16 | 0 / 16 |
| client-org-mandate | org-competitor | 0 / 32 | 0 / 32 |
| client-org-nda | org-vendor | 0 / 48 | 0 / 48 |
| client-org-onboarding | org-competitor | 0 / 16 | 0 / 16 |
| client-org-onboarding | org-landlord | 0 / 16 | 0 / 16 |
| client-org-onboarding | org-vendor | 0 / 16 | 0 / 16 |
| client-org-prospect | org-competitor | 0 / 16 | 0 / 16 |
| client-org-prospect | org-landlord | 0 / 16 | 0 / 16 |
| client-org-trade-counterparty | org-listed-company | 0 / 16 | 0 / 16 |
| client-org-trade-counterparty | org-vendor | 0 / 32 | 0 / 32 |

**Thirteen pairs, zero on both halves, for every one of the 16 arms** — the
client-versus-non-client discrimination the predicate exists for. Two further pairs
whose confusable is `client-name`-paired but not an organisation (`client-org-fee-note |
product-name`, `client-org-trade-counterparty | person-name`) are also 0/16 on both
halves.

**The strongest caveat this result carries, and it is decisive.** The positive half is
**zero by construction, not by failure.** `client-name` is a tier-1 entity type, tier 1
ran in no arm, and `p-fin` has no tier-0 rule for it — the labelled manifest records
`compiledArmReach: null` for that class. Across the whole experiment exactly one
`client-name` finding exists. So the table shows that **no arm can emit the right label
on an organisation name at all**; it does not show that the models cannot tell a client
from a landlord. What it does show, on the negative half, is that no arm produced a
`client-name` false positive either.

**The pairs that do discriminate are mechanical.** Same definition, same denominators:

| Positive family | Confusable family | Positive-side | Negative-side |
|---|---|---|---|
| jdbc-url-kv | sftp-endpoint | **24 / 48 (50.0 %)** | 0 / 48 |
| jdbc-url-kv | deployment-name | 8 / 16 (50.0 %) | 0 / 16 |
| pan-kv | tan-plain | **24 / 48 (50.0 %)** | 0 / 48 |
| pan-prose-boosted | tan-plain | 9 / 16 (56.3 %) | 0 / 16 |
| pan-prose-boosted | tan-boosted | 8 / 16 (50.0 %) | 0 / 32 |
| kyc-case-id | vendor-invoice-number | 28 / 48 (58.3 %) | 0 / 48 |
| labelled-account-credited | settlement-batch-number | 27 / 48 (56.3 %) | 0 / 48 |

The ~50 % ceiling is structural: 8 of the 16 arms run tier 0, and tier 0 is what fires.

### 5c. Predicate accuracy on the injection corpus v2

Source: `runs/slate-rebuild-01*.jsonl` scored against
`corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl` by `scoreArms` from
`apps/eval/src/driver/score.ts`. MEASURED HERE *(derivation script)*.

Gold: 189 rows, **179 scored**, 10 disputed, **19 positives**. Vocabulary:
`pred:client-relationship-disclosure`.

**The floors first, because that is the order they must be read in.** Every floor reader
is a function of the message alone — none is given the policy, the predicate, the
organisation pool, or the gold (`score.ts:566–570`). Scored over exactly the rows the
arms were scored on, by the same matcher.

| Reader | What it does | Findings | P | R | **F1 (exact = overlap = iou50)** |
|---|---|---|---|---|---|
| **first-capitalised-multiword** | the FIRST run of two or more capitalised words, and nothing else | 30 | 0.467 | 0.737 | **0.571** |
| **capitalised-multiword** | every run of two or more capitalised words | 50 | 0.380 | 1.000 | **0.551** |
| **whole-message** | one span covering the entire message | 179 | 0.106 (overlap) | 1.000 (overlap) | **0.192** (overlap); 0.000 exact and iou50 |

All sixteen arms, sorted by overlap F1:

| Arm | tp | fp | fn | P | R | **F1 (overlap)** | F1 exact | F1 iou50 | Judge answered / 179 | In-vocab findings | Hidden by the disputed exclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tier2-Qwen3-4B | 6 | 36 | 13 | 0.143 | 0.316 | **0.197** | 0.197 | 0.197 | 109 | 42 | 8 |
| tier2only-Qwen3-4B | 6 | 46 | 13 | 0.115 | 0.316 | 0.169 | 0.169 | 0.169 | 122 | 52 | 8 |
| tier2-Phi-4-mini | 10 | 119 | 9 | 0.078 | 0.526 | 0.135 | 0.122 | 0.122 | 176 | 129 | 13 |
| tier2only-Phi-4-mini | 10 | 131 | 9 | 0.071 | 0.526 | 0.125 | 0.100 | 0.100 | 177 | 141 | 13 |
| tier2-Qwen3.5-2B | 3 | 51 | 16 | 0.056 | 0.158 | 0.082 | 0.027 | 0.027 | 179 | 54 | 2 |
| tier2only-Qwen3.5-2B | 1 | 37 | 18 | 0.026 | 0.053 | 0.035 | 0.000 | 0.000 | 179 | 38 | 2 |
| tier2-Ministral-3-3B | 0 | 16 | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 140 | 16 | 2 |
| tier2only-Ministral-3-3B | 0 | 20 | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 162 | 20 | 1 |
| baselineB-Qwen3.5-2B | 0 | 1 | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 108 | 1 | 0 |
| baselineB+tier0-Qwen3.5-2B | 0 | 1 | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 117 | 1 | 0 |
| baselineB-Ministral-3-3B | 0 | 0 | 19 | **undef** | 0.000 | **undef** | undef | undef | **0** | 0 | 0 |
| baselineB-Phi-4-mini | 0 | 0 | 19 | **undef** | 0.000 | **undef** | undef | undef | **0** | 0 | 0 |
| baselineB-Qwen3-4B | 0 | 0 | 19 | **undef** | 0.000 | **undef** | undef | undef | **0** | 0 | 0 |
| baselineB+tier0-Ministral-3-3B | 0 | 0 | 19 | **undef** | 0.000 | **undef** | undef | undef | **0** | 0 | 0 |
| baselineB+tier0-Phi-4-mini | 0 | 0 | 19 | **undef** | 0.000 | **undef** | undef | undef | **0** | 0 | 0 |
| baselineB+tier0-Qwen3-4B | 0 | 0 | 19 | **undef** | 0.000 | **undef** | undef | undef | **0** | 0 | 0 |

`winnersByRule` reports `disagree: false` — all three rules name the same winner,
`tier2-Qwen3-4B`, at F1 0.197 — with **6 of 16 arms unrankable** (no defined F1).

**The 2.9× statement, computed.** Best arm F1 = 2·6/(2·6+36+13) = 12/61 = 0.19672. Best
floor F1 = 2·14/(28+16+5) = 28/49 = 0.57143. Ratio = **2.905**. So: *the best of sixteen
arms scores a factor of 2.9 below a regular expression that reads neither the policy nor
the predicate.* **No arm beats any floor under any rule**, except that `tier2-Qwen3-4B`
exceeds `whole-message`'s overlap F1 of 0.192 by **0.005**.

The round record's own verdict names the reason: *"the corpus is lowercase informal
prose in which the organisation names are almost the only capitalised multi-word tokens,
so recall 1.000 costs a regular expression nothing."* An oracle handed the answer's
vocabulary — every occurrence of the five `ORG_POOL` names — scores F1 **0.585**, higher
still, and is explicitly *not* a floor because it has been told what to look for.

**The disputed exclusion is one-sided, and its size per arm is in the table.** All ten
disputed rows are negatives carrying no gold span, so the exclusion removes false
positives and nothing else. It hides 8 of `tier2-Qwen3-4B`'s 50 in-vocabulary findings,
moving its overlap precision from 0.120 to 0.143 — **a 19 % relative improvement produced
by the exclusion rather than by the arm.**

**Six arms read `P = undefined, R = 0.000`.** They produced no completed call at all.
`score.ts` deliberately makes precision undefined (not 0, not 1) in that case and F1
undefined with it, and puts those arms in `unrankable` rather than ranking them last —
*"an arm that made no claim has not lost a comparison, it was not in one."*

#### The discriminating-negatives analysis

160 of the 179 scored rows are negatives. **152 of those 160 carry no organisation name
at all** (MEASURED HERE against `ORG_POOL` *(derivation script)*), so a pooled
false-positive rate over them is mostly a claim about inert text. The **8** negatives
that do carry an organisation name are the discriminating ones — and all 8 match one of
the five `PREDICATE_DISCRIMINATORS.negative` fragments, so the two definitions coincide.

Per-arm firing on those 8 (a `pred:` finding anywhere on the row):

| Arm | Fires on discriminating negatives | Fires on all 160 negatives | Fires on the 19 positives |
|---|---|---|---|
| tier2-Phi-4-mini | **7 / 8** | 108 / 160 | 17 / 19 |
| tier2only-Phi-4-mini | **7 / 8** | 120 / 160 | 17 / 19 |
| tier2-Qwen3-4B | 2 / 8 | 34 / 160 | 8 / 19 |
| tier2only-Qwen3-4B | 2 / 8 | 44 / 160 | 8 / 19 |
| tier2-Qwen3.5-2B | 1 / 8 | 48 / 160 | 6 / 19 |
| tier2only-Ministral-3-3B | 1 / 8 | 16 / 160 | 4 / 19 |
| tier2-Ministral-3-3B | 0 / 8 | 12 / 160 | 4 / 19 |
| tier2only-Qwen3.5-2B | 0 / 8 | 33 / 160 | 5 / 19 |
| all eight Approach-B arms | 0 / 8 | 0–1 / 160 | 0 / 19 |

**Phi-4-mini fires on 7 of the 8 discriminating negatives while firing on 17 of the 19
positives: it is name-triggered, not relationship-reasoning.** (Including the ten
disputed rows, which are all organisation-bearing, both Phi-4-mini arms fire on 17 of
the 18 non-positive organisation-bearing rows.)

An arm firing on none of the 8 supports a one-sided 95 % upper bound of
1 − 0.05^(1/8) = **31.2 %** on the discriminating false-positive rate (COMPUTED HERE,
rule of three). Under the alternative definition "any row containing a capitalised
multiword" the denominator is 11 and the bound is 23.8 %; Phi-4-mini fires on 10 of
those 11.

### 5d. The `[43,59) "Tamarind Grocers"` row

The single most informative row in the experiment, quoted verbatim from the run record
it is in — `runs/slate-p-fin-03.baselineB-Qwen3.5-2B-q4f16_1-MLC.jsonl`, item
`pos-client-name-prose`:

```json
{"start":43,"end":59,"text":"Tamarind Grocers","entityType":"in-pan",
 "severity":"critical","tier":2,"source":"Qwen3.5-2B-q4f16_1-MLC",
 "confidence":1,"action":"block"}
```

The message is `Can you draft a contract renewal email for Tamarind Grocers before
Friday?` (74 characters). The corpus gold span (`corpora/fixtures/smoke.jsonl`) is
`{"start":43,"end":59,"text":"Tamarind Grocers","entityType":"client-name","action":
"pseudonymize"}`, and the blind-labelled tier-2 gold
(`corpora/fixtures/smoke.gold-tier2.jsonl`) carries the same span
`{"start":43,"end":59,"text":"Tamarind Grocers"}` as the evidence for
`satisfies: true`.

**A 2-billion-parameter local model pointed at exactly the right sixteen characters —
byte-identical offsets to two independently written gold sets — and called them an
Indian Permanent Account Number.** The span is right; the label is not merely wrong but
in a different data class, and it carries `severity: "critical"`, `action: "block"` and
`confidence: 1`.

MEASURED HERE: this is the only finding with `text === "Tamarind Grocers"` anywhere in
`runs/slate-p-fin-0*`.

---

## 6. Findings

Stated as claims a paper could make, each with its evidence pointer and the strongest
caveat that applies. **Negative results first, because they are the result.**

### N1. On this corpus, no arm beats a trivial capitalisation rule at the semantic task

*Evidence:* §5c. Best of sixteen arms F1 0.197; best floor F1 0.571; ratio 2.9×. Only
`whole-message` (F1 0.192) is beaten, by 0.005.
*Caveat, and it is severe:* the gold's positive set is exactly the set of items carrying
a `client-name` gold span (189/189 agreement with the generator's own construction
record), and the 19 positives come from 7 fixed sentence templates over 5 names. The
floor wins because the corpus is lowercase prose where those names are nearly the only
capitalised bigrams. **This is as much a finding about the corpus as about the models**,
and the round record says so.

### N2. On this corpus, a two-rule orthographic reader is competitive with every arm at span finding

*Evidence:* §5b. Oracle unbudgeted P 0.307 / R 0.870; budget-matched P 0.705 / R 0.685 /
F1 0.695; best arm P 0.362 / R 0.731 / F1 0.485. Definitions in
`apps/eval/src/corpus/leakage.ts:317–375`.
*Caveat:* the budget-matched oracle is handed each item's gold-span count, which no arm
gets; the unbudgeted oracle is the like-for-like comparison and the best arm does beat it
on F1 (0.485 vs 0.454). The honest statement is that **the best arm's margin over "return
every odd-looking string" is 0.031 F1, and most of that arm's score is tier-0 regex.**

### N3. Nothing in the experiment discriminates a client organisation from a non-client one

*Evidence:* §5b's near-miss table — 13 organisation/organisation pairs, 0 on both halves,
all 16 arms; and §5c's discriminating-negatives table — the arm with the best positive
recall (Phi-4-mini, 17/19) also fires on 7 of the 8 discriminating negatives.
*Caveat:* the positive half of the near-miss table is zero **by construction** (tier 1
never ran; `client-name` has no tier-0 rule), so it is evidence of a pipeline
configuration, not of model capability. The discriminating-negatives result is the one
that bears on capability, and its denominator is **8**.

### N4. Approach B does not reach a first token within a shipped policy's message budget on three of four models

*Evidence:* §5a. `answeredCalls: 0`, `budget-exhausted` 13/13 (smoke) and 189/189
(injection corpus) for Ministral-3-3B, Qwen3-4B and Phi-4-mini, on both B families, at
`latencyBudgetMs: 5000`. On Qwen3.5-2B, B answers and fails the p95-TTFT gate at
2,735.88 ms.
*Caveat:* the 1,500 ms ceiling was derived at the compiled judge's ~1.1 kB prompt and is
applied to B unchanged — the plan calls this the largest unfixed asymmetry. **B's failure
is not "B is bad at the task"; it is "B does not fit the budget the compiler emits."**
And B's zero-answer arms are recorded as `killedOnRunGates: false`, which is the harness
refusing to convert silence into a verdict.

### N5. The span-convention split fixed the action-span defect and did not improve detection

*Evidence:* §2.5. Whole-message `pred:` spans 5/13 → 1/11; overlap true positives 3 → 0;
exact and iou50 stayed at 0 on both slates. `unresolvedMentions` is 0 on
`slate-p-fin-03`, so the refusal path's cost is **unobserved rather than bounded**.
*Caveat:* the two slates are different code states and, on `exact`/`iou50`, different
questions asked of the model; 3 and 0 are not two measurements of the same quantity.

### P1. The compiled pipeline is 5× cheaper per call than policy-in-context, and that gap is a prompt-size gap

*Evidence:* §5a. Compiled p50 prompt 252–280 tokens vs B's 1,410; p95 TTFT 556–1,244 ms
vs 2,736–2,866 ms on the one model where B answers. The policy document is 5,320 bytes.
*Caveat:* B is handed the compiler's entity-type ids as a concession without which no B
finding would pass `normalizeFindings`, so B is not a pure "no compiler" arm; and this is
one machine, one GPU, one policy.

### P2. A message-scoped predicate costs FEWER calls, not more

*Evidence:* §2.6. `judgedUnitsPerItem` p50=p95=max=1; `scope-unjudged` 3-of-3 → 0 on both
compiled families; the two scopes partition the predicate list rather than the message
call being an addition.
*Caveat:* `p-fin` declares only a message-scoped predicate. A both-scopes policy would
pay `selected + 1` calls and **no such policy exists here, so that is arithmetic, not a
measurement.**

### P3. Tier 0 carries the compiled pipeline's span accuracy; the local LLM adds little and costs recall precision

*Evidence:* §5b. The three `baselineB+tier0` arms whose LLM answered **zero** calls score
P 0.360 / R 0.667 / F1 0.468 — within 0.017 F1 of the best arm overall. Adding a judge
raises recall (Phi-4-mini 0.843) and lowers precision (0.271).
*Caveat:* the corpus mints every value through the shipping validators, so tier-0 recall
on `in-pan`/`in-aadhaar` is a manufactured upper bound (§4.6).

### P4. The binding constraint is classification, not span extraction

*Evidence:* §5d — the byte-exact `[43,59)` span labelled `in-pan` at `confidence: 1`;
§5b's finding that 89.8 % of non-predicate findings land on orthographically odd regions;
§5c's floors.
*Caveat:* single-row evidence for the headline instance; the supporting aggregates come
from one corpus whose organisation names are drawn from a five-name pool.

---

## 7. Threats to validity

Exhaustive as far as I could make it. Each is stated with its size where a size exists.

1. **One machine, one GPU, one browser.** No field of any run record names either. Every
   latency, decode rate, load time and storage figure is a single-machine observation
   with n = 1 machine. `experimentScope` on every gates row says the hardware axis is
   held at one point.
2. **One policy compiled.** `policies/compiled/` holds `p-fin` only. `p-med` and `p-corp`
   have never been compiled — `scripts/compile-policies.ts` exits 1 on a fixture miss for
   both, because `requestHash` keys a fixture by the prompt. So spec §6.1's "three
   deliberately disagreeing policies" and §6.4.3's policy-adaptivity delta cannot be
   computed, and the corpus manifest states `p-med`/`p-corp` labels are **unpopulated**.
   Any per-policy number off this corpus is a `p-fin` number.
3. **The compile is fixture-replayed, not a live frontier compile.** Both model-driven
   compiler stages were answered from committed hand-authored fixtures.
   `src/llm/anthropic.ts` has never executed. The open question — *does a frontier model
   given only a policy document produce an IR that survives the anti-hallucination
   gate?* — is unanswered. The entity vocabulary behind the IR is not a frontier model's.
4. **Carriers are hand-authored and no realism gate ran.** Stored,
   `.unvalidated`: the carriers are not the ShareChat/WildChat conversations spec §6.2
   specifies, and **none of its three realism gates ran** — no frontier naturalness score,
   no adversarial style probe, no human spot check. The manifest names the style probe as
   the one this pool would most likely fail, because carriers and injections share an
   author.
5. **The corpus is NOT CERTIFIED.** Two of three certification stages are unrun and 0 of
   57 carriers are certified-clear (§4.2). The invariant spec §6.2 buys — "on positives,
   gold spans are exactly the injected ones; any finding outside them is a true false
   positive" — **is not in force.**
6. **Certification circularity.** Stage 1 is `runTier0` over a widened copy of the
   scoring IR, i.e. the tier-0 arm under test. The circularity is measured (0 carriers
   quarantined by stage 1 alone against the IR-free sweep) rather than removed, because
   spec §6.2 prescribes stage 1 in those words.
7. **Dev and test are pooled.** 39 dev / 150 test are declared and disjoint, but
   `measureLeakage` ran over all 189 with dev and test pooled, families were removed and
   rewritten on those pooled measurements, and **every arm run scored all 189 with no
   split recorded on any row.** A pooled number cannot be reduced to a test-only one
   after the fact.
8. **Effective N on the predicate is 7, not 189.** The 19 positives come from 7 sentence
   templates over 5 organisation names (§4.4). Errors correlate within a template.
   Framing-level, perfect recall supports only ≥ 65.2 % at 95 %.
9. **The discriminating-negative denominator is 8.** 152 of 160 scored negatives carry no
   organisation name, so a pooled false-positive rate is mostly a claim about inert text.
   An arm firing on none of the 8 supports only ≤ 31.2 % at 95 %.
10. **The predicate gold is not independent of the corpus's construction record.**
    `meta.predicateConstruction.constructed` equals the adjudicated label on **189 of
    189** rows, and the 19 positives are exactly the items carrying a `client-name` gold
    span. A `pred:` score over this gold measures the same target as a `client-name` span
    score.
11. **The 12-fragment leak.** Seven message fragments decide every positive and five more
    decide the organisation-bearing negatives; all twelve are committed verbatim in
    `families.v2.ts` beside the intended answer. `labelling.ts`, committed before the
    round, names one positive by item id and organisation. Round 2's read channel is
    **unaudited**, so nothing rules out use.
12. **Round 2's `told` channel is unaudited and lands on the round's most quotable
    numbers.** No brief retained, no disclosure collected. If the brief stated a positive
    count, both annotators returning exactly 19 is what that would produce, and no
    evidence separates the two explanations.
13. **Round 1 carried three told-channel breaches, one machine-verified.** TOLD-1 handed
    annotators the compiled IR's own `nlPredicate` while forbidding them the IR; TOLD-2
    supplied a stale directional prior about a family already removed; TOLD-3 supplied
    distributional hints. Plus two structural leaks: stratum-encoding item ids, and a
    `question` field that names the span type and pre-argues both readings.
14. **contextBoost is not symmetric.** +0.0753 any-term and +0.0979 own-type in the
    compiled arm's favour, measured the way `runTier0` actually reads it. Any
    per-entityType number carries a keyword advantage of that size.
15. **Position substitutes for the orthographic discriminator.** The distractor follows
    the span it shadows on 81 of 105 pairs; on 84 of 108 items the earlier of the two
    spans is the gold one. Open, disclosed, unfixed.
16. **Two IR `examples` strings occur verbatim in the corpus text** — the two PEM headers
    `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----BEGIN RSA PRIVATE KEY-----`, 3 items
    each (stored, `.leakage.irOverlap.irSurfacesFoundInCorpusText`). The value-level check
    cannot see them. `carriersCheckedAgainstIrSurfaces` is `false`.
17. **Every minted value is validator-gated at mint time**, which systematically excludes
    the compiled arm's own tier-0 failure mode: no real PAN that `pan-structure` rejects,
    no non-PAN that it accepts (§4.6).
18. **A fifth of the confusable surface carries first-person exculpation** the policy
    elsewhere forbids relying on, so an arm that believes the sender scores as precise.
19. **26 confusable labels on the client/non-client axis were never adjudicated.**
    `CONTESTED_TYPES` was chosen by the labels' own author (2 of 28 families), so the
    blind round covered 20 of 219 confusable spans, and `neg:org-vendor`,
    `neg:org-landlord` and the two cross-segment org families were never put to an
    annotator.
20. **`neg:batch-sequence` (9 spans) is scored as a true false positive on one authored
    clause** while its structural twin `neg:retrieval-reference` (7 spans) was left
    disputed and excluded. Both satisfy the IR's written `in-aadhaar` definition, so tier
    0 fires on all 16 by construction.
21. **Structural score ceilings.** `private-key-material` reads 0 of 18 for every
    compiled arm under `exact` and `iou50` (gold is the whole PEM block; the rule matches
    the header). `bank-account-identifier` reaches 11 of 17 under `exact`. Those columns
    report a span convention, not a detector.
22. **Per-type recall is quantised** at 5.3–14.3 percentage points; `in-pan` (7 spans) and
    `db-connection-string` (9 spans) cannot resolve small differences at all.
23. **The disputed exclusion is one-sided.** All 10 disputed rows are negatives with no
    gold span, so it removes false positives only. Sized per arm in §5c (up to a 19 %
    relative precision improvement).
24. **Tier 1 never ran.** Every span number here is a tier-0-and-tier-2 number, and the
    entity class the semantic predicate shadows (`client-name`) is a tier-1 class no arm
    could emit.
25. **No run record names a code revision, and two runs with byte-identical `run` blocks
    differ by ~81 prompt tokens for reasons no commit explains** (§3.6). Cross-slate
    comparisons are comparisons between unrecorded code states.
26. **`p95` is a maximum at these sample sizes.** `percentile` is nearest-rank, and
    `ceil(0.95·n) = n` for every n < 20, so on the 13-item smoke corpus the p95-TTFT gate
    is applied to each arm's slowest call. The gates say so in prose on every affected
    row.
27. **Two of four numeric gates are chosen, not derived.** `minResolvableRate` 0.8 is a
    chosen floor with no distribution behind it; `maxDuplicateRate` 0.9 is bracketed by
    two single-message observations, and the constant's own comment says two points are
    not a distribution.
28. **The `resolvable-rate` gate's own history.** In `slate-p-fin-01` it killed two arms
    on a sample of two quotes, before `minSample` existed.
29. **The engine-load figures are warm, not cold.** The harness uses a persistent Chrome
    profile, so `engineLoadMs` of 1.5–3.6 s is a re-load from cache, not a first download.
30. **`@mlc-ai/web-llm` is not resolvable from `apps/eval`.** `pnpm -r test` fails there
    (§10), so the assertion that the harness's WebGPU adapter floor matches the installed
    bundle's numbers is currently unexecuted.
31. **The Plan 3 live compile is deferred and the capability-ceiling arm (spec §4.2b) has
    never been built**, so "is the browser the bottleneck, or is small-model capability
    the bottleneck?" remains unanswerable from this repository (§9).
32. **No statistical machinery from spec §6.4 is implemented.** No bootstrap CIs, no
    McNemar's test, no `analysis/` directory, no Python anywhere. Every interval in this
    document is a rule-of-three bound I computed by hand and labelled as such.

---

## 8. Process findings

The review/verify/fix discipline is methodologically relevant, because in five of the
cases below the defect would have produced a complete, schema-valid, internally
consistent, **wrong** result.

### 8.1 The null-contrast fixture (`rules: []`) — `4df1022`

`apps/eval/fixtures/semantic-ir.json` shipped `rules: []`, while `planBakeoff`
hard-requires an IR with `semanticPredicates` — which made it the only IR any tier-2 arm
could run. So `runTier0` found nothing on any item, no prior made a segment uncertain,
and **the two contrasts Plan 5 calls thesis-separating were null by construction**:
`compiled` vs `compiled-tier2-only`, and `baseline-b` vs `baseline-b-tier0`. Both arms
in each pair would have produced matching numbers, and *"a reader would have concluded
that deterministic patterns contribute nothing to recall without a single deterministic
pattern having been evaluated."* Measured through the shipped page API: tier-0 findings
per item were [0,0,0] under `semantic-ir.json` against [1,1,1] under `minimal-ir.json`.

Two tests had **encoded the defect as expected behaviour** — one asserting the two arms
escalate identically "and that is a fact about the IR". The commit message records how it
was found: *"Found by the one review lens that died mid-response and was re-run rather
than written off."*

### 8.2 Approach B could not run while the plan's Done criteria said it had — AUDIT-4

Three independent blockers, verified by trying to run the arm rather than by reading
code: no page door (`grep -c createBaselineB apps/eval/src` returned 0), no
`BaselineStats` field on the record schema, and every IR in the repository carrying
`policyHash: "test-hash"` so `planBakeoff`'s `sha256(policyText) === ir.policyHash`
pairing check threw before `assertPageCanRun` was reached. All three were closed by
compiling `p-fin` from recorded fixtures; `packages/compiler/test/compiled.test.ts`
recompiles the committed IR on every suite run and requires byte equality, so a hand-edit
is a red test.

### 8.3 Corpus v1 was contaminated toward the compiled arm and its numbers were discarded

The manifest's `.supersedes.why` names the reasons and the rebuild measured all of them
(§4.2). Sizes, re-derived here: contextBoost delta **+0.4537** under the detector's own
reading; **58 of 108** gold spans were the unique orthographic outlier; **19 of 39**
organisation names were locked to one role class; and the budget-matched oracle scored
**P 0.931 / R 0.880 / F1 0.905** on it — i.e. the old corpus was very nearly solved by
"return the odd string". The fifth defect, "13 of 24 confusable families were 1:1 with an
IR counterExample", is quoted from a module docblock and is the one figure here I could
not re-derive (see §11).

Runs against that corpus exist on disk (`runs/slate-corpus-01*`, three gates files,
`run.corpus: injection-p-fin-adjudicated-v1.jsonl`) and **no accuracy number from them
appears in this document.** The superseded corpora are kept rather than deleted, because
*"deleting a measurement that was made is not correcting it."*

### 8.4 The gamed uniqueness certification — `3300b93`

`goldSpansSolvedByOracle = 0` was designed as the certification that "the gold span is not
the unique orthographic outlier", and it certifies nothing: one same-shape distractor per
positive drives it to zero whatever the message looks like. Demonstrated in the test by
padding the **old** corpus and watching its solved rate go to zero with recall unchanged.
Kept as a measurement, re-documented as certifying nothing, and replaced by figures a
distractor cannot flatter. **The general lesson the artifact draws: judge a corpus by a
budget-matched baseline, never by a uniqueness count.**

### 8.5 Falsified provenance, and its enforcement — `0edb3db`

The first emit of the predicate gold rewrote **all 378** `annotators.{a,b}.rationale`
strings — an adjudicator's rewrite — under a schema comment saying they were carried
verbatim *"so a later reader can re-adjudicate without the labelling round"*. They are
now restored byte for byte from the annotators' own return files, and both return files
are committed (`injection-p-fin-v2.predicate-annotator-{a,b}.json`, sha256 recorded in
the round record). `buildPredicateRoundRecord` throws on any divergence, so a record
cannot be emitted for a falsified gold; `corpus-predicate-round.test.ts` fails on any
divergence.

**The fixer also refused half of the review.** The commit records that
`annotators.a.span.text` **was** A's own returned quote on all 19 positives; what was
invented were the **offsets** — neither annotator returned one, both returned a quoted
string — and the prose describing them. The field is now `quote`, and the schema states
which halves are the annotator's and which are the adjudicator's.

The same commit **withdraws a claim from its own predecessor's commit message**: that
both annotators flagged two told-channel leaks unprompted. No artifact supports it, so it
is withdrawn rather than repeated, and a measured open leak (the 12 fragments) is
disclosed in its place.

### 8.6 A commit subject that cannot be fixed in place

`6e39bdd`'s subject reads *"fix(eval): refuse to score a tier the corpus cannot score"*
and the commit deliberately implements **no refusal** — its own body says "the run is NOT
refused". Only the subject survives into `git log --oneline`. The plan records the
decision not to rewrite three commits' history and states what the subject should have
read: *"name the tiers a corpus cannot score"*.

### 8.7 The recurring defect pattern

Recorded in the plan and visible in 8.1: **plan-supplied tests derive their expectation
from the code under test, so they pass under the mutation they exist to catch.** The
countermeasure used is mutation testing. One round's result is recorded with its
controls: **27 mutants, 27 killed**, over a `git ls-files -co` copy of the working tree,
with the source tree verified untouched by md5 afterwards, and **four CONTROL mutants
(two comment rewords and two unread error messages) all survived** as they should. Ten
mutants survived a first Node pass and each named a real hole — nothing in vitest built an
Approach-B record, so all four of `RunRecordSchema`'s new refines could be replaced by
`() => true` with a green suite. One control was initially reported killed; that was the
harness lying (the copy's dev server had exited mid-batch) and the batch was re-run under
a harness that checks the server is alive before and after every mutant.

### 8.8 Counts of findings raised / confirmed / refuted

The only per-round counts I can verify from a committed artifact or commit message:

| Round | Recorded count | Source |
|---|---|---|
| Final review round (2026-09-01) | *"Three lenses reviewed the round's last three commits and found **29 items**; every one was checked against the code before anything was changed."* | plan, "Deviations — the final review round" |
| Corpus review | *"The corpus machinery had **eight confirmed defects and three open decisions**."* | `ea0fa45` commit body |
| Corpus-rebuild review | *"**Three findings** made a published number wrong or a stated property untrue."* | `3300b93` commit body |
| Predicate-gold review | *"**Three defects** in f3d2495, in the order they matter."* — one of which the fixer partly refuted (§8.5) | `0edb3db` commit body |
| Mutation round | 27 mutants, 27 killed, 4 controls survived, 10 first-pass survivors closed | plan, "Documentation truth pass" |

**No workflow journal recording raised/confirmed/refuted per lens is committed in this
repository**, so no other counts are reported here.

### 8.9 A defect found while writing this document

The manifest's `.leakage.boost.verdict` string, and the commit message of `3300b93` that
introduced it, both read: *"far smaller than the corpus this replaces — which read
**+0.4537 and +0.4352** on the same two forms"*, set against the v2 residuals +0.0753
(any-term delta) and +0.0979 (own-type delta).

RE-COMPUTED HERE by running `measureBoost` over
`injection-p-fin-adjudicated-v1.jsonl` with the wave-2 paired-type map committed in
`corpus-v2.test.ts` *(derivation script)*: the before corpus's `asTier0Reads.delta` is
**0.45370** — the first number is right — and its `asTier0Reads.ownTypeDelta` is
**0.39815**, not 0.4352. The value 0.43519 is that corpus's `goldOwnTypeRate`, a **rate**,
not a delta, and it is being compared with a delta. No test pins the literal, so nothing
caught it. **The magnitude and direction of the claim are unaffected — the residual really
did fall by roughly 4× on the own-type form — but the published second number is the wrong
statistic.**

---

## 9. What is not yet done

1. ~~**The capability-ceiling arm (spec §4.2b).**~~ **BUILT AND RUN.** See
   `docs/research/2026-09-07-ceiling-arm.md`. It was executed over OpenRouter against six
   open-weight models (30–120 B) with providers pinned and fallbacks disabled, rather than
   via llama.cpp or Ollama — the models are the ones a Spark-class box or an EU-hosted
   server could run, so the arm answers spec §4.2b's question while remaining a
   *capability* ceiling rather than a *deployment* one. Prompts left the browser, which is
   why it is a measurement arm and not a shippable path; the cloud boundary for the
   product is unchanged.

   **The result depends on which of two metrics is asked, and the document reports the
   stricter one.** Scored span-wise (as §9 there does), the ceiling arms land in the
   neighbourhood of the trivial capitalisation floor (0.571) and do not clearly separate
   from it. Scored on the **message-level boolean the predicate gold actually records**,
   the best compiled judge does separate, and so does the whole ordering. On the same 179
   rows: **best local (in-browser 2–4 B) 0.275 < trivial floor 0.776 < best ceiling
   (hosted 30–120 B) 0.905**, the ceiling arm holding perfect recall with 4 false
   positives. The mechanism is precision — local arms over-fire badly (no local arm
   exceeds precision 0.25; `Phi-4-mini` reaches recall 0.895 with 108 false positives),
   which is §5d's "right region, wrong label" and is what disappears at scale.

   **The two metrics answer spec §4.2b oppositely, and this is the open question.**
   Span-wise the ceiling arms stop at the floor, so the browser constraint looks like it
   is not what costs accuracy; message-wise they clear it while the local arms do not, so
   the gap **is** substantially the price of the in-browser constraint. Both are computed
   from the same rows and the same gold. See that document's §7.5, including the caveat
   that the floor's perfect message-level recall is partly the corpus's known
   12-fragment leak. Pass 1's best thinking-off
   arm scored 0.565 (−0.006); pass 2's same arm scored 0.615 (+0.044) at `temperature: 0`,
   with three items changing hands on a 19-positive gold. Pass-to-pass variance exceeds the
   distance to the floor, so this sample does not separate them. Against the *local* arms
   the improvement is real and large — 2.87× at the predicate level, 2.04× at the span
   level — which sharpens §5d rather than overturning it: **the binding constraint is
   classification, not span extraction**, and it is still binding at 120 B. Span placement
   is essentially solved at this size (2 unresolved quotes and 8 unresolved mentions on
   1,243 findings).

   Cost: **$0.386 over 2,461 calls**, reconciled against the key endpoint. What the arm did
   *not* settle, and what is still untested in its own instrumentation, is
   `2026-09-07-ceiling-arm.md` §11.
2. **A regenerated corpus that closes the 12-fragment leak, followed by a re-run blind
   round.** Also on that list: randomising distractor position (which closes §7.15 but
   moves every offset, so the labels must be re-collected), and handing annotators
   anonymised item ids with the `question` field's argument scaffolding stripped.
3. **The live frontier compile.** `ANTHROPIC_API_KEY` plus
   `pnpm -C packages/compiler exec vite-node ../../scripts/record-fixtures.ts --yes`.
   Recording **overwrites** the hand-authored fixtures. Until it runs, `p-med` and
   `p-corp` have no IR and the policy-adaptivity metric has no denominator.
4. **Certification stages 2 and 3**, which need a Python `corpora/` toolchain and the
   deferred frontier call respectively.
5. **The `analysis/` half of spec §2.2** — bootstrap CIs, McNemar's test, the two-axis
   leak-prevention/over-blocking plot, and the utility-preservation study. None exists.
6. **Plan 6, the WXT MV3 extension.** Nothing in this record has been executed inside an
   extension; the harness is a Playwright page.
7. **A second policy's worth of scope coverage.** The `selected + 1` call cost of a
   both-scopes policy is arithmetic, never measured.

---

## 10. Test suite, as it stands

MEASURED HERE. `pnpm -r test` at HEAD `0edb3db`:

```
packages/core      Test Files  22 passed (22)   Tests  354 passed (354)
packages/tier2     Test Files   9 passed  (9)   Tests  328 passed (328)
packages/compiler  Test Files  13 passed (13)   Tests  139 passed (139)
packages/tier1     Test Files  10 passed (10)   Tests  219 passed (219)
apps/eval          Test Files   1 failed | 22 passed (23)
                        Tests   1 failed | 709 passed (710)
Exit status 1
```

**Totals: 77 test files, 1,750 vitest tests, 1,749 passed, 1 failed.** The command exits
non-zero.

The failure is `apps/eval/test/page-webgpu-floor.test.ts > web-llm's adapter floor > is
the floor the installed bundle enforces, number for number`:
`Error: Cannot find module '@mlc-ai/web-llm'`. Diagnosis: `@mlc-ai/web-llm@0.2.84` is
declared only in `packages/tier2/package.json` and installed only at
`packages/tier2/node_modules/@mlc-ai/web-llm`; the test calls
`require.resolve("@mlc-ai/web-llm")` from `apps/eval`, which does not declare it. The
other four assertions in that file pass.

Because `apps/eval`'s test script is `vitest run && playwright test`, **Playwright never
ran under `pnpm -r test`.** Listed separately: `playwright test --list` reports **82 tests
in 9 files**.

**The README is stale on these counts.** `README.md:173` claims *"1,329 tests — 1,247
vitest (core 354, tier2 281, eval 254, tier1 219, compiler 139) plus 82 Playwright"*.
Core, tier1 and compiler match; `tier2` is 328 not 281 and `eval` is 710 not 254, both
because the last four commits added tests. The README does not record that the suite
currently exits 1.

---

## 11. Could not verify

Numbers I was given or that appear in project notes and **could not re-derive from an
artifact**. None of them is used anywhere above.

1. **"100 % of every arm's findings land on spans the oracle also flags."** Not
   reproducible under any of four readings. MEASURED HERE, pooled over 16 arms: all
   findings 78.30 %; non-`pred:` findings 89.82 %; `pred:` findings 43.28 %; findings that
   overlap a gold span 87.60 %. Per-arm the all-findings rate ranges 4.8 %–91.0 %. §5b
   reports the measured figures instead.
2. **"14 discriminating negatives"** and **"Phi-4-mini fires on 13 of 14."** MEASURED
   HERE: 8 scored negatives carry an `ORG_POOL` organisation name (and the same 8 match a
   `PREDICATE_DISCRIMINATORS.negative` fragment); Phi-4-mini fires on **7 of 8**. Under
   the alternative definition "contains any capitalised multiword" the denominator is 11
   and Phi-4-mini fires on 10. Neither gives 13/14.
3. **"139 of 160 scored negatives carry no org name."** MEASURED HERE: **152** of 160.
4. **"On the 14 discriminating negatives the bound is ≤ 19.3 %."** The arithmetic
   1 − 0.05^(1/14) = 19.3 % is correct for n = 14, but n is 8 here, giving **≤ 31.2 %**.
5. **"First round: 13/13 agreement."** — *Resolved after this document was first
   written.* The figure is real and belongs to a DIFFERENT round from the one this item
   originally examined. `corpora/fixtures/smoke.gold-tier2.jsonl` (the 13-item smoke
   round, two blind annotators, adjudicated against p-fin §3.1) records both annotators
   agreeing on `satisfies` for **13 of 13** rows (recomputed from the artifact's
   `annotators.a/b.satisfies` fields). The 20-row figures quoted here — `spanLabelCorrect`
   13 of 20 (raw 0.65, κ degenerate at 0) and `satisfiesPredicate` 20/20 (κ undefined) —
   belong to the v2 corpus's contested-SPAN round. Both are correct; they are two rounds.
6. **"smoke.jsonl has 7 gold spans, 5 at tier 0, 2 at tier 1."** The file has 7 gold
   spans, but the harness's own `goldSpansByTier` is `{0: 3, 1: 2, 2: 0}` — three at tier
   0, not five; the remaining two (`aws-key`, `generic-secret`) are not declared by the IR
   and belong to no tier.
7. **Phi-4-mini cold load 73.6 s.** `packages/tier2/src/manifest.ts` cites
   `out-e5.json phi.load.ms=73592`. **No file named `out-e5.json` or `out-*.json` exists
   in this repository.** The warm loads I can read are 1.5–3.6 s.
8. **`+0.4352` as the before-corpus own-type contextBoost delta** (published in the
   manifest verdict and in `3300b93`'s message). Re-computed value: **0.39815**; 0.43519
   is that corpus's own-type *rate*. See §8.9.
9. **Which commit produced any given run.** No *local* run record names a code revision,
   and the `slate-p-fin-02` mtime precedes the commit that introduced a field its records
   carry (§3.6). The ~81-token prompt shift between `slate-p-fin-02` and `slate-p-fin-03`
   is unexplained by any commit in the interval. **Still unresolved for every local run,
   and unresolvable — the information was never recorded.**

   **Fixed going forward, and the ceiling arm is the first run that has it.** Every
   `CeilingRecord` carries `gitSha` and `gitDirty`, captured once at launch by
   `gitProvenance()` and stamped on every row, so each of the 5,580 thinking-off rows
   names the revision that produced it (`8c4fc8b` for passes 2–3, `bbdbfb1` for pass 1).
   Two limits, stated because a reader will otherwise over-trust the field: it records the
   state **at launch**, not throughout — a tree edited mid-run still reports the launch
   value — and `gitDirty: false` means only that `git status --porcelain` was empty then.
   The driver has no dynamic imports, so every module is read and cached at startup and
   the sha is genuinely the code that ran. See `2026-09-07-ceiling-arm.md` §11.4.
10. **"13 of 24 confusable families in the v1 corpus were 1:1 with an IR counterExample
    surface."** Stated in the docblocks of `apps/eval/src/corpus/families.v2.ts:65` and
    `apps/eval/src/corpus/leakage.ts:776`. The check is a comparison of hand-written
    surface names against `IR_COUNTEREXAMPLE_SURFACES` (23 entries), not a function over
    the corpus, and the v1 families' surface names are no longer in the tree — so there
    is nothing to run it against. The **after** half (0 of 28) *is* stored in the
    manifest and is used above.
11. **The standing brief at
    `/private/tmp/.../d5f3dd55-.../scratchpad/p5/standing-brief.md`.** The file did not
    exist when this document was written: the session scratchpad it lived in had been
    deleted underneath four running agents. It has since been reconstructed from its
    earlier reads and committed as `docs/research/standing-conventions.md` (commit
    `80a9c5e`), which is where every future brief points. This document was written
    following two of its rules in prose — records state fact rather than intent (§3
    there) and comments must be accurate (§1) — and has been checked against the rest
    after the fact. The disappearance is itself recorded in that file's preamble.

---

## 12. Reference index

Every commit cited above, in the order it first appears, with its subject line. Run
`git show <hash>` on any of them.

| # | Hash | Subject |
|---|---|---|
| 1 | `975b5c2` | feat(eval): a blind-labelled tier-2 gold set and the span scorer that reads it |
| 2 | `4180ac1` | feat(tier2): package scaffold with the measured model slate |
| 3 | `6b169be` | fix(tier2): sizeMb was VRAM, and contextWindowSize had no guard |
| 4 | `449bc3b` | feat(tier2): engine seam with the measured call recipe |
| 5 | `07e0622` | docs(plan-5): record four defects Task 5 found in the plan's engine seam |
| 6 | `e59ca54` | feat(tier2): interrupt-and-drain cancellation, never a timeout race |
| 7 | `44e4273` | docs(plan-5): interrupt-and-drain was not sufficient; the engine latches |
| 8 | `2e9b268` | feat(tier2): span-recovery ladder that refuses ambiguity instead of guessing |
| 9 | `9ea3994` | docs(plan-5): record three defects Task 4 found in the plan's span ladder |
| 10 | `47ba8f6` | feat(tier2): separate the span that locates a finding from the span an action rewrites |
| 11 | `1540aba` | fix(tier2): close what the span-convention review confirmed |
| 12 | `a82b22b` | feat(tier2): judge message-scoped predicates against the whole message |
| 13 | `82a5fce` | fix(tier2): close what the message-scope review confirmed |
| 14 | `364b939` | fix(eval): pin the bake-off gates and the harness seams the reviews found unguarded |
| 15 | `d851131` | fix(eval): close what the Task 10-12 reviews confirmed |
| 16 | `1770229` | fix(eval): refuse a half-specified IR, and stop gates ruling on two observations |
| 17 | `6e39bdd` | fix(eval): refuse to score a tier the corpus cannot score |
| 18 | `58f86e7` | feat(core): a degraded channel, message scope, and a budget that spans segments |
| 19 | `c15a1cd` | fix(core): pin the degraded channel's guarantees and correct its claims |
| 20 | `5ce1e34` | feat(eval): a blind-adjudicated corpus, and the predicate gold the round did not reach |
| 21 | `52cc75f` | feat(eval): a predicate queue that hands over the message and nothing else |
| 22 | `f3d2495` | feat(eval): predicate gold from a blind round over every message |
| 23 | `0edb3db` | fix(eval): restore the predicate gold's provenance and publish its floor |
| 24 | `3300b93` | fix(eval): close what the corpus-rebuild review confirmed |
| 25 | `4df1022` | fix(eval): give the bake-off's only IR the tier-0 half of a policy |
| 26 | `ea0fa45` | fix(eval): close what the corpus review confirmed |

Other commits in the branch's history referenced by the narrative but not cited for a
number: `d3b5fd3` (Plan 5 added), `490b3bb` (spec §4.2b, the capability-ceiling arm),
`bec5f98` (spec amended after measurement), `461bad6` (Approach-B baseline),
`474309e` (escalation policy), `d794c2e` (grammar-constrained JSON contract),
`b0dab17` (WebLlmJudge), `3c36fcf` (engine lifecycle on a persistent profile),
`0279840` (rung distribution and token accounting in the JSONL),
`863cd91` (four-arm bake-off driver), `fdcc416` (report the judged unit, pin both
prompts), `bc3e10f` (integrity review), `d0dadd5` (make the head-to-head runnable, and
run it), `2c34bad` / `d1f933a` / `0b18785` / `db440c3` (the injection corpus pipeline).

### Artifacts cited

| Path | What was read from it |
|---|---|
| `docs/superpowers/specs/2026-08-13-ai-dlpp-design.md` | research question; tier design; model slate; corpus design; metric definitions; §4.2b |
| `docs/superpowers/plans/2026-08-30-05-tier2-judge-baseline.md` | deviations log; gate derivations; asymmetry table; Task 13 |
| `README.md`, `apps/eval/README.md` | claimed test counts; harness description |
| `packages/tier2/src/{manifest,engine,spans}.ts` | model slate; pinned recipe; ladder |
| `packages/core/src/detect/types.ts` | the five degraded reason words |
| `apps/eval/src/driver/{bakeoff,score}.ts` | gates, `minSample`, scoring boundary; match rules, floors, `scoreArm` |
| `apps/eval/src/corpus/{leakage,predicate-round,families.v2,labelling}.ts` | oracle, boost, roles; discriminators; near-miss pairing; `BRIEF_PARAPHRASE_FRAGMENT` |
| `policies/p-fin.md`, `policies/compiled/p-fin.{ir.json,report.md}` | policy size and structure; 9 entity types, 10 rules, 1 predicate, budget 5000, failMode closed |
| `corpora/fixtures/smoke{,.gold-tier2}.jsonl` | 13 items, 7 gold spans; 2 predicate positives |
| `corpora/generated/injection-p-fin-v2.*` | corpus, manifests, labelled corpus, both golds, predicate round record, both annotator return files |
| `corpora/generated/injection-p-fin-adjudicated-v1.jsonl` | the superseded corpus, for the before/after leakage table |
| `runs/slate-p-fin-{01,02,03}.*` | smoke-corpus latency, throughput, gates |
| `runs/slate-rebuild-01{,b,c}.*` | injection-corpus accuracy and gates |
| `runs/slate-corpus-01{,b,c}.*` | the discarded v1-corpus runs (named, not quoted) |
| `packages/tier2/node_modules/@mlc-ai/web-llm` | `prebuiltAppConfig`, for the VRAM verification |
