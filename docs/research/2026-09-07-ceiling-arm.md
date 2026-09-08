# The capability-ceiling arm: six large open-weight models, thinking off

**Date:** 2026-09-07 · **Branch:** `feat/tier2-judge` · **Code:** `apps/eval/src/driver/ceiling.ts`,
`ceiling-main.ts`, `ceiling-score.ts` · **Tests:** `apps/eval/test/ceiling.test.ts`

Every number in this document was read from a run artifact under gitignored `runs/` by
`pnpm -C apps/eval ceiling:score`, or from a live API response captured during the run. None was
recalled. Where a number is an estimate rather than a measurement it says so.

---

## 1. What this arm is, and what it is not

Spec §4.2b asks for a capability-ceiling arm to separate two explanations of the tier-2 result:
*is the task hard, or are 2–4B browser-runnable models too small?* The spec describes that arm as a
**local server** (llama.cpp/Ollama), which keeps intact the cloud boundary this project exists to
defend.

**This arm does not do that.** It runs six larger open-weight models over the OpenRouter API, so
every corpus message left the machine. That is a real deviation and it is stated here rather than
buried:

- it answers the **capability** question §4.2b poses — the models are open-weight and could be
  served locally on adequate hardware;
- it answers **nothing** about the privacy boundary;
- it is **not a shippable configuration** and must never appear in a product column.

What *is* preserved is the method. Both prompts, both response schemas, the span ladder and the
parse-and-one-repair rule are **imported from `@sih/tier2`**, not reimplemented. `judge.ts` and
`baselineB.ts` gained four aliased exports (`buildJudgeMessages`, `judgeRepairMessage`,
`buildBaselineMessages`, `baselineRepairMessage`) so this file could reuse them; `createBaselineB`
now goes through `buildBaselineMessages` too, so there is one composition and not two.
`packages/tier2/test/prompts.test.ts` pins both system turns line by line and stayed green
throughout — the prompts are byte-identical to the ones the browser arms send.

### The two families

| family | arm name | equivalent local arm | what the model is shown |
|---|---|---|---|
| judge | `ceiling-judge-<model>` | `compiled-tier2-only` | the compiled predicate + the whole message |
| B | `ceiling-b-<model>` | `baseline-b` | the whole `p-fin.md` + every entityType id + the message |

p-fin's only predicate is `scope: "message"`, so the judge family makes exactly one call per item,
as the local compiled arm does. Neither family runs tier 0 or tier 1.

### The one mechanism that genuinely differs

The browser arms constrain decoding with **xgrammar**, compiled from the schema inside WebLLM.
This arm sends the same schema as an OpenRouter
`response_format: {type: "json_schema", strict: true}` and the **provider** constrains decoding, by
whatever means it uses. Different mechanisms, different failure modes — so every row carries
`outputMechanism: "provider-json-schema"`, and §7 reports parse failures and repairs per model as a
finding about the mechanism, not only about the model.

### Record shape

`CeilingRecordSchema` is a **sibling** of `RunRecordSchema`, not an extension. To satisfy
`RunRecordSchema` a row from here would have to claim a `backend` of `"wasm"` or `"webgpu"` (an
HTTPS call to a remote GPU is neither), a `detector` of `"core-orchestrator"` or `"approach-b"`
(neither implementation ran), a `tier2Config` naming a context window a hosted endpoint does not
expose, and — through the refines — a stats object whose fields describe events (engine latching,
caller aborts mid-generation) that cannot occur over HTTP. Four lies to reuse one schema.

Instead the row carries the identity and scoring fields verbatim and honest transport telemetry
beside them, and `score.ts` reads both shapes through a new structural `ScoreableRecord` type. One
scorer, one gold, one set of floors; two record shapes, each stating what actually happened.

Every ceiling row also carries **`gitSha` and `gitDirty`** — the first run artifacts in this
repository that name the code revision that produced them. `gitDirty` is `true` for this run and is
recorded honestly: the run was made from a working tree carrying this code before it was committed,
and committing early to make the flag read `false` would be the intent-as-fact defect in its purest
form.

---

## 2. Where the brief was wrong

Briefs on this project have been wrong in every round. This one had five errors and one dead link.

1. **The standing-brief path was dead.** Its scratchpad directory had been deleted. A durable copy
   now lives at `docs/research/standing-conventions.md`.
2. **`z-ai/glm-5.3-flash` cannot be run with thinking off — on any provider.** See §6. The brief
   pinned it as one of six "thinking off" models; the endpoint answers HTTP 400
   *"Reasoning is mandatory for this endpoint and cannot be disabled."*
3. **The cheapest-first ordering was wrong.** The brief's order put GLM-Flash ahead of Qwen-Flash.
   On the unweighted `in + out` price sum GLM (0.65) *is* dearer than Qwen-Flash (0.62) — the brief
   had them the other way round — and on this corpus's actual prompt-heavy mix the gap is wider
   still. Ordering is now by `estimateCostUsd` at a stated representative token mix.
4. **`injection-p-fin-v2.gold-tier2.jsonl` cannot rank anything.** The brief says to score against
   both golds. That gold's 20 rows carry **0 positives and 0 gold spans**, so `tp = fn = 0` for
   every arm and recall and F1 are *undefined* for all of them under every rule. It measures
   precision only. It is scored below, with that stated on the table.
5. **Mistral's quantization is `unknown`, and its pin resolves to three endpoints at two prices.**
   Recorded as unknown; cost taken from each response's own `usage.cost` rather than a price table,
   which is what makes the ledger correct despite the ambiguity. Mistral's is the only **EU-hosted**
   first-party endpoint on the slate; the other five pins are US providers reached through a
   US-headquartered aggregator.
6. **Streaming everything is not safe.** See §6: one endpoint's streaming path died mid-run while
   its non-streaming path kept answering in 2.5s.

Everything else in the brief checked out. All six model ids exist; all six pins were confirmed
against `/api/v1/models/{id}/endpoints` before any spend; and the claim that DigitalOcean is the
only Nemotron endpoint advertising `structured_outputs` (DeepInfra bf16 does not) is correct.

---

## 3. The probe — three items per family per model, before any budget was committed

Read from `runs/ceiling-ceiling-01.spend.json` (`probes[]`).

| model | pin | family | parsed | provider answered | pin honoured | reasoning tokens | TTFT ms | wall ms | verdict |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-v4-flash-0731 | DeepInfra | judge | 3/3 | DeepInfra | yes | 0,0,0 | 869,905,902 | 876,2493,921 | pass |
| deepseek-v4-flash-0731 | DeepInfra | b | 3/3 | DeepInfra | yes | 0,0,0 | 1313,1282,1281 | 2671,2724,6519 | pass |
| qwen3.8-flash | Alibaba | judge | 3/3 | Alibaba | yes | 0,0,0 | 741,769,765 | 813,1564,835 | pass |
| qwen3.8-flash | Alibaba | b | 3/3 | Alibaba | yes | 0,0,0 | 5689,3821,5887 | 6617,5314,8564 | pass |
| **glm-5.3-flash** | BaseTen | judge | **0/3** | none | — | — | — | — | **skip** |
| **glm-5.3-flash** | BaseTen | b | **0/3** | none | — | — | — | — | **skip** |
| mistral-small-2603 | Mistral | judge | 3/3 | Mistral | yes | 0,0,0 | 451,452,562 | 452,452,584 | pass |
| mistral-small-2603 | Mistral | b | 3/3 | Mistral | yes | 0,0,0 | 1078,432,472 | 1670,977,2003 | pass |
| nemotron-3-super-120b-a12b | DigitalOcean | judge | 3/3 | DigitalOcean | yes | 0,0,0 | 930,795,657 | 1442,5146,1238 | pass |
| nemotron-3-super-120b-a12b | DigitalOcean | b | 3/3 | DigitalOcean | yes | 0,0,0 | 707,834,842 | 5840,4685,19483 | pass |
| qwen3.8-27b | Parasail | judge | 3/3 | Parasail | yes | 0,0,0 | 490,599,626 | 663,1577,675 | pass |
| qwen3.8-27b | Parasail | b | 3/3 | Parasail | yes | 0,0,0 | 615,874,880 | 1865,2377,4113 | pass |

**Every pin was honoured on every answered call.** No response named a provider other than the one
`provider.order` asked for, in the probe or in the full run (§7). `allow_fallbacks: false` is what
makes this checkable rather than hopeful: with fallbacks on, a busy pinned provider silently routes
elsewhere and the latency columns become facts about routing wearing the model's name.

**GLM-5.3-flash failed both probes with HTTP 400 and was skipped, not retried into the budget:**

```
{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.",
          "code":400,"metadata":{"provider_name":null}}}
```

---

## 4. How it was run

```
pnpm -C apps/eval ceiling        # SIH_CEILING_RUN_ID, _PASSES, _PROBE, _LIMIT, _MODELS, _THINKING, _PIN
pnpm -C apps/eval ceiling:score  # every table below
```

- **Thinking off** via OpenRouter's unified `reasoning: {enabled: false}`. `{enabled: false}` and not
  `{effort: "none"}`: OpenRouter documents `effort` as the OpenAI/Grok spelling, five of the six
  models advertise `reasoning_effort` and `qwen/qwen3.8-flash` does not, and `enabled` is the field
  all six accept. What was **asked** is on every row as `reasoningRequest`; what **happened** is
  `calls[].reasoningTokens`, read back from `usage.completion_tokens_details.reasoning_tokens`. Two
  columns, so a model that ignores the request is visible rather than assumed away.
- **Structured output** via `response_format: {type: "json_schema", strict: true, schema}` carrying
  `JUDGE_SCHEMA` / `BASELINE_B_SCHEMA` unchanged — `minimum`/`maximum` bounds included, which every
  answering provider accepted.
- **`max_tokens: 600`**. This was *intended* to match the local arms and does **not**:
  `DEFAULT_TIER2_CONFIG` is **512**. The 88-token asymmetry, and which way it cuts, is §7.1. **`temperature: 0`**, so repeat passes measure provider and
  routing variance rather than the sampler.
- **Provider pinned** with `provider: {order: [<name>], allow_fallbacks: false}`. A pin is a
  request; every row records `requestedProvider` **and** the `provider` the response carried.
- **Concurrency 3**, exponential backoff on 429/5xx, every retry recorded on the call row.
- **Cheapest model first**, so a tripped guard leaves the most expensive arm partial.

### The spend guard

Two independent numbers can stop the run, and both are checked, because they can disagree and the
pessimistic one is the one that can exhaust the key:

1. the sum of every response's own `usage.cost` (obtained by sending `usage: {include: true}`),
   added to whatever the key had spent before the run started; and
2. the key endpoint's own `usage`, re-read from `GET /api/v1/auth/key` every 50 calls.

Either reaching **$7.00** stops the run — $3 under the key's $10 cap, which covers in-flight
concurrency at the moment of the trip and the final key read. A price-table estimate is accumulated
alongside as a cross-check but is never the primary number: the `Mistral` pin alone resolves to
three endpoints at two different prices, so only the response's own `cost` is knowable in advance.
`runs/ceiling-<runId>.spend.json` is rewritten after **every** call, so a crash leaves an accurate
ledger.

### Four timing columns, and what each one is

| column | meaning | transferable? |
|---|---|---|
| `reasoningTokens` | `usage.completion_tokens_details.reasoning_tokens` | **yes — a model property** |
| `ttftMs` | ms to the first non-empty **content** delta | no — a provider fact |
| `decodeTokPerSec` | `completionTokens / ((wallMs − ttftMs) / 1000)` | no — a provider fact |
| `wallMs` | request start to final chunk | no — what a user waits through |

TTFT is timed to the first *content* delta, not the first byte: OpenRouter opens every stream with
`: OPENROUTER PROCESSING` comment frames and a role-only delta, and timing to those would measure
the aggregator's connection handling. `decodeTokPerSec` is `null`, never `Infinity`, when the decode
window is zero. Every usage number is `null` rather than `0` when a provider reported none — a `0`
in the reasoning column would be the measured claim this experiment is looking for, so it must be a
measurement and not a default.

---

## 5. GLM-5.3-flash: the model that refuses to stop thinking

`z-ai/glm-5.3-flash` answered **HTTP 400 on every thinking-off request, on all eleven of its
providers** — BaseTen, DeepInfra, Fireworks, Together, Parasail, CoreWeave, Modal, NextBit,
Sail Research, Cloudflare and Z.AI:

> `Reasoning is mandatory for this endpoint and cannot be disabled.`

This is a **model** property, not an endpoint one, and **OpenRouter's own metadata does not
advertise it**: the model's `reasoning_config` is `null` and its `supported_parameters` lists
`reasoning`. Only a live call reveals it. Anyone planning a thinking-off evaluation from the
catalogue alone would have budgeted for six models and got five.

It was therefore run as a **separate, explicitly non-comparable arm** with `reasoning:
{enabled: true}`, pinned to NextBit (its slate pin, BaseTen, was rate-limited upstream when the
thinking-on run started). Rows carry `thinkingRequested: "on"` and the scorer labels them
**thinking ON** so they can never be read as one of the thinking-off arms.

Two results came out of it, and both are about the cost of mandatory reasoning:

- **The judge family works.** 189/189 rows, 179/179 items answered — the only arm on either side of
  this experiment with perfect coverage.
- **The Approach-B family cannot run at all.** Its probe parsed **0 of 3**, and the reason is exact:
  the model spent **596–599 of its 600 completion tokens on reasoning** and emitted no answer.
  `ttftMs` is `null` on all six of those calls because *no content delta ever arrived* — reasoning
  deltas are not content. The arm was skipped rather than retried into the budget.

That second result is the local Approach-B failure repeated at the ceiling **for an entirely
different reason**. Locally B blew the 5,000 ms latency budget because its prompt is 1,410 tokens.
Here latency was never the constraint; the *completion* budget was, and mandatory reasoning consumed
it before the task began.

## 6. What each family actually emits

Counted from `runs/ceiling-01.ceiling-{judge,b}-deepseek-v4-flash-0731.jsonl`:

| | judge family | Approach-B family |
|---|---|---|
| findings emitted | 30, **all** `pred:client-relationship-disclosure` | 204, spread over **all nine** entityTypes |
| of which at the predicate | 30 | **7** |
| unresolved quotes / mentions | 0 / 0 | 0 / 0 |
| whole-clause mentions | 0 | **33** |

This is the mechanism behind the family gap in §7. Approach B is asked about nine classes at once
and spends its findings on identifiers, naming the relationship predicate **7 times in 189
messages**; the compiled judge is asked about one predicate and names it 30 times.

**The span ladder placed every finding on both arms of this pair** — 0 unresolved quotes and 0
unresolved mentions across these 234 findings. It is not clean across the whole slate: pooled over
all ten pass-1 arms the counts are **2 unresolved quotes** (all in `b-mistral`) and **8 unresolved
mentions** (`judge-nemotron` 3, `judge-qwen3.8-27b` 2, `judge-qwen3.8-flash` 2, `b-mistral` 1), on
**1,243 findings**. Approach B also resolves to the whole clause far more often than the judge does
— 157 of the 158 pooled whole-clause mentions are B-family. The local arms do not manage this, and it is worth stating plainly:
at this model size, quoting a clause verbatim and pointing at a shorter span inside it is a solved
problem. Whatever is going wrong is not span extraction.

---

## 7. Five measurement defects found in this arm's own instrumentation

Recorded here rather than quietly fixed, because each one changes how a number below reads.

### 7.1 The completion budget was NOT matched to the local arms (88 tokens, favouring this arm)

A docblock in `ceiling-main.ts` asserted *"The local arms run at 600 (`DEFAULT_TIER2_CONFIG`)"*.
**They do not.** `packages/tier2/src/manifest.ts:116` is `maxTokens: 512`, and its own docblock at
:81 says it is *"UNCHANGED at 512 through the two-span schema change"*. Nobody had read the
constant. So the hosted arms ran with **600** completion tokens against the browser arms' **512** —
an 88-token asymmetry introduced by a comment. Standing-conventions §1, exactly.

**Which way it cuts:** it favours the ceiling arms. That is the *conservative* direction for this
document's headline — the ceiling arms had the larger budget and still did not beat the trivial
floor — and the *flattering* direction for any local-vs-ceiling gap.

**How much it actually bit, counted over the finished slate.** An earlier draft of this section,
written when only the two DeepSeek arms existed, said *"not one thinking-off call reached 512"*.
**That is false on the completed run** and is corrected here rather than quietly amended: three
calls did.

| population | calls | completions ≥ 512 | truncated at 600 |
|---|---|---|---|
| all ten thinking-off arms | 1,854 | **3 (0.162%)** | **0** |
| — `ceiling-b-mistral-small-2603` | 189 | 1 (544 tokens) | 0 |
| — `ceiling-b-qwen3.8-flash` | 176 | 2 (534, 581 tokens) | 0 |
| `ceiling-judge-glm-5.3-flash` **thinking ON** | 216 | **55** | **48 (22.2%)** |

So the asymmetry is **not** perfectly inert: **3 of 1,854 thinking-off calls (0.162%)** produced
completions the browser arms' 512-token cap would have cut short, both in Approach-B arms, all three
in the 512–581 range. None of the ten thinking-off arms truncated at 600, so no thinking-off result
is a truncation artefact *at the cap that was used* — but three answers would have been at 512.
Whether those three change a finding is not knowable without re-running at 512, which was not done.
The effect is bounded and small; it is not zero, and the honest statement is the percentage, not the
word "inert". It remains large only for the thinking-on arm. `ceiling-score.ts` prints the `calls >=512` column so this
stays checkable rather than asserted, and the record now carries `maxTokens` and
`localArmMaxTokens` on every row written from here on. It was **not** changed mid-experiment: the
paid run was in flight, and a slate half-measured at each value is worse than one measured at a
documented 600.

### 7.2 The thinking-on arm cannot be measured at this cap — its F1 is not a clean number

GLM-5.3-flash's finish reasons over 216 calls are **168 `stop` / 48 `length`**: **22.2% of its
answers were cut off**, with reasoning p50 228 and max **exactly 600, the cap**. Its
**F1 0.571 is therefore "GLM under a 600-token cap with 22% of its answers truncated"**, not a
measurement of GLM. It is reported that way everywhere below and must not be quoted bare.

The Approach-B probe's 0/3 is the same defect at 100%: 596–599 reasoning tokens of 600, on both the
initial call *and* the repair turn.

**The general finding: a thinking-on phase cannot be run at the local arms' budget at all.** Any
future thinking-on comparison needs a uniformly larger `max_tokens` — 8,192 is the obvious value —
applied to **every** model, with `finish_reason: "length"` counted per arm. The thinking-off arms
are unaffected: DeepSeek is clean at **0 truncations and 0 reasoning tokens across all 355 calls**.

### 7.3 Two different wall clocks, and which column is which

DeepInfra rate-limited **45–50% of the DeepSeek calls** — 154 429s on the judge arm, 161 on B, some
items needing four retries. A latency column that silently absorbed those backoff sleeps would be a
fact about the aggregator wearing the model's name (standing-conventions §3).

It does not, and that is **proved by a test rather than assumed**: `callChat` re-initialises its
clock *inside* the retry loop, so `calls[].ttftMs`, `calls[].decodeTokPerSec` and `calls[].wallMs`
time **the successful attempt only** — every failed attempt and every backoff sleep excluded. The
record's **top-level `wallMs`** is the end-to-end figure that *does* include them. Both are carried;
the transport table prints both, plus the 429 count that explains the gap:

| arm | call wall p50 (attempt) | item wall p50 (end to end) | 429s |
|---|---|---|---|
| `ceiling-judge-deepseek-v4-flash-0731` | 1,325 ms | 2,127 ms | 154 |
| `ceiling-b-deepseek-v4-flash-0731` | 2,000 ms | 4,089 ms | 161 |
| `ceiling-judge-glm-5.3-flash` (thinking ON) | 5,444 ms | 4,885 ms | 0 |

So DeepSeek's TTFT p95 of 4,828 ms is **not** a backoff artefact — it is that provider's own
first-token latency under load. Still a provider fact, but a different one than throttling.

**Concurrency was deliberately NOT lowered for DeepInfra mid-run.** DeepSeek is the only
DeepInfra-pinned model and both its pass-1 arms were already complete; dropping to concurrency 2 for
passes 2–3 would make the repeat passes non-comparable with pass 1, which is the one thing repeat
passes exist to measure. The 429 counts are reported per arm instead.

---

### 7.4 Arms are scored on items the provider never let them attempt

**This one was found while checking a different claim, and it moves the headline.**

68 of the 3,780 rows across passes 1–2 (**1.8%**) carry `calls: []`, `provider: null` and an
`error` of the form *"OpenRouter returned 429"* — the call exhausted its retries against a rate
limit and no answer was ever produced. A representative row spent **16.6 s** in backoff before
giving up. These are not parse failures and not refusals; the model never saw the item.

`ceiling-score.ts:122` computes `positives` **once over the whole gold**, independent of what any
given arm was able to attempt:

```ts
const positives = gold.filter((g) => g.status === "scored" && g.satisfies).length;
```

So the recall denominator is 19 for every arm, and **an item the provider refused to serve is
scored identically to an item the model read and missed.** The scorer prints an `answered/scored`
column, so the information is on the page — but it is never applied.

Where the losses fell, and how unevenly:

| arm | unanswered | of those, gold-**positive** |
|---|---|---|
| `ceiling-01 b-deepseek` | 19 | 2 |
| `ceiling-01 b-qwen3.8-flash` | 13 | 1 |
| `ceiling-01 judge-deepseek` | 4 | **1** |
| `ceiling-02 b-deepseek` | 4 | 0 |
| `ceiling-02 b-nemotron` | 2 | 0 |
| `ceiling-02 judge-deepseek` | 1 | **0** |
| `ceiling-02 judge-nemotron` | 25 | 1 |
| all other 13 arms | 0 | 0 |

**Why it matters to §10.1.** The two arms in the floor comparison are the last two rows of the
DeepSeek judge pair. Pass 1 was rate-limited off `inj-o02-0`, a gold positive; pass 2 lost none.
Holding precision fixed and removing only that unattempted positive from pass 1's denominator:

| pass | as published | attempted-only |
|---|---|---|
| `ceiling-01` judge-deepseek | 0.565 (−0.006 vs floor) | **0.577** |
| `ceiling-02` judge-deepseek | 0.615 (+0.044 vs floor) | 0.615 |

So a meaningful part of pass 1's "just below the floor" result is **a provider's rate limiter, not
the model**. This compounds the §10.1 correction rather than replacing it.

**The same rows also inflate the latency columns, and only at the tail.** `ceiling-score.ts` builds
its TTFT, decode-rate and per-call wall percentiles from `calls`, so a row with `calls: []`
contributes nothing to any of them — those three columns are clean. The **item wall** column is
different by design: it is the record's top-level `wallMs`, documented in the scorer as *"end to
end, backoff included"*. That is accurate, but a reader will picture backoff around a call that
eventually succeeded, not a row where no call ever succeeded. Measured both ways:

| arm | dead rows | item wall p50 (all / answered) | item wall p95 (all / answered) |
|---|---|---|---|
| `ceiling-02 judge-nemotron` | 25 | 1,309 / 1,221 ms | **17,631 / 7,000 ms** |
| `ceiling-01 b-deepseek` | 19 | 4,089 / 3,465 ms | 17,451 / 17,712 ms |
| `ceiling-01 b-qwen3.8-flash` | 13 | 4,249 / 3,815 ms | 95,539 / 78,287 ms |
| `ceiling-01 judge-deepseek` | 4 | 2,127 / 2,102 ms | 16,484 / 12,423 ms |
| `ceiling-02 b-deepseek` | 4 | 2,896 / 2,793 ms | 16,088 / 13,588 ms |

**p50 moves by 2–16%; p95 moves by up to 2.5×.** The worst case is an arm whose published tail
latency is `17,631 ms` and whose tail latency *for work actually done* is `7,000 ms`. For a project
whose deliverable includes a latency budget, the median is safe to read as published and **the p95
item-wall figures are not** — they are a statement about the provider's rate limiter as much as
about the model. The per-call wall p50/p95 columns beside them are unaffected and are the ones to
quote for model latency.

**What this table is not.** The floor is a deterministic function of the text and would also be
evaluated on a different subset if the arm's subset were used; these attempted-only figures compare
an adjusted arm against an **unadjusted** 0.571 floor and are therefore **not yet like-for-like**.
The correct fix is to score each arm, and every floor, over the intersection of items that arm
actually answered — reported *beside* the whole-gold numbers, never instead of them, since the
unanswered rows are a real cost of running against a rate-limited hosted provider and should not be
defined away. That change is not in this document's numbers; it is queued with the other scorer
work in §11.2, and every figure above is the as-published scoring unless the row says otherwise.

---

### 7.5 The predicate gold is a message-level label, and it is scored span-wise

**This is the largest single effect found in this review, and it changes the answer to §10.1.**

The predicate gold carries **179 scored message-level booleans and 19 spans**. The blind annotation
round asked annotators one question per message — *does this message satisfy the predicate* — and
the `spans` field is supplementary to that judgment. The table in §9.1, labelled "predicate level",
does **not** score that judgment: it goes through `scoreArms`, which pairs a finding's **span**
against a gold **span**. An arm that correctly identifies a disclosing message but points at the
wrong phrase scores a false positive *and* a false negative.

That is a defensible metric — pseudonymisation needs the span, and flagging the right message for
the wrong reason is a real failure — but it is the **stricter of two**, and the document presents it
as though it were the natural one.

**The floor's 0.571 reproduces exactly** under an independent reimplementation (tp 14, fp 16, fn 5;
identical under exact, overlap and iou50 — the three rules carry no information here, because gold
predicate spans *are* capitalised organisation names like `Marrowfield Group`, so the reader either
hits one exactly or misses it entirely; there is no partial-overlap regime). The published span-wise
numbers are correct. The point is what the other metric says.

**Scored on the judgment the gold actually records** — an arm predicts *satisfies* iff it emits any
`pred:` finding on that message:

| | P | R | F1 | tp / fp / fn |
|---|---|---|---|---|
| **FLOOR** capitalised-multiword | 0.633 | **1.000** | 0.776 | 19 / 11 / 0 |
| `ceiling-02 judge-deepseek` | 0.826 | **1.000** | **0.905** | 19 / 4 / 0 |
| `ceiling-01 judge-deepseek` | 0.833 | 0.789 | **0.811** | 15 / 3 / 4 |
| `ceiling-02 judge-qwen3.8-27b` | 0.633 | 1.000 | 0.776 | 19 / 11 / 0 |
| `ceiling-01 judge-qwen3.8-27b` | 0.613 | 1.000 | 0.760 | 19 / 12 / 0 |
| `glmon-01 judge-glm` (thinking on) | **1.000** | 0.579 | 0.733 | 11 / 0 / 8 |
| `ceiling-02 b-nemotron` | 0.000 | 0.000 | 0.000 | 0 / 0 / 19 |

**Both passes of the compiled judge on DeepSeek beat the floor, by 0.129 and 0.035** — margins far
outside the ±0.05-per-item noise that makes the span-wise comparison unreadable. The best arm holds
**perfect recall with 4 false positives across 179 messages**, against the floor's 11. The B family
collapses here (recall 0.158 or worse, two arms at zero), exactly as §6 predicts: it is asked about
nine entity classes and rarely names the predicate at all.

**What this does and does not license.**

- It does **not** retract the span-wise result. Both are real: *these models find the right message
  and are much less reliable about where in it to point* — the same shape as the local finding in
  §5d, now at 30–120 B.
- The floor's **perfect message-level recall is a corpus artifact** as much as a result: every
  positive in this corpus contains a capitalised organisation name, which is the known open leak
  (12 fragments in `families.v2.ts`). The floor cannot miss. That inflates the floor and makes the
  best arm's margin *harder* to achieve, not easier — but it also means neither number transfers to
  a corpus without that property.
**The local arms, recomputed at message level on the same rows.** The 32 in-browser arms that ran
against the v2 corpus (`slate-corpus-01*`, `slate-rebuild-01*`) all cover the full 179 scored items,
so they are directly comparable. Best eight:

| local arm | P | R | F1 | tp / fp / fn |
|---|---|---|---|---|
| `tier2-Qwen3-4B` (corpus-01c) | 0.169 | 0.737 | **0.275** | 14 / 69 / 5 |
| `tier2only-Qwen3-4B` (corpus-01c) | 0.161 | 0.737 | 0.264 | 14 / 73 / 5 |
| `tier2-Qwen3-4B` (rebuild-01c) | 0.190 | 0.421 | 0.262 | 8 / 34 / 11 |
| `tier2-Phi-4-mini` (rebuild-01c) | 0.136 | 0.895 | 0.236 | 17 / 108 / 2 |
| `tier2-Ministral-3-3B` (rebuild-01) | 0.250 | 0.211 | 0.229 | 4 / 12 / 15 |
| `tier2only-Qwen3-4B` (rebuild-01c) | 0.154 | 0.421 | 0.225 | 8 / 44 / 11 |
| `tier2only-Phi-4-mini` (rebuild-01c) | 0.124 | 0.895 | 0.218 | 17 / 120 / 2 |
| `tier2-Qwen3.5-2B` (corpus-01) | 0.140 | 0.421 | 0.211 | 8 / 49 / 11 |

**The three-way comparison, all on the same 179 items:**

| | message-level F1 |
|---|---|
| best **local**, in-browser 2–4 B | **0.275** |
| trivial **floor**, `capitalised-multiword` | **0.776** |
| best **ceiling**, hosted 30–120 B | **0.905** |

**This inverts §10.1's answer.** Span-wise, the local arms are far below the floor and the ceiling
arms sit level with it, which is what produced *"the browser constraint is not what is costing
accuracy."* Message-wise, the local arms are still far below the floor — **and the ceiling arms are
clearly above it.** On the question spec §4.2b was written to settle, the message-level reading says
the gap **is** substantially the price of the in-browser constraint.

The mechanism is precision, not recall. The local arms **over-fire**: `Phi-4-mini` reaches recall
0.895 with **108 false positives** on 179 messages, and no local arm exceeds precision 0.25. The
floor gets 0.633. The best ceiling arm gets **0.826 with perfect recall**. That is the same "right
region, wrong label" failure §5d found locally, and it is what disappears at 30–120 B.

**The entity path, same treatment.** The other half of what the product decides is *"does this
message contain a sensitive entity at all"* — the flag-for-action decision, as opposed to which span
to pseudonymise. 108 of the 189 items carry at least one entity gold span (57.1%), so the trivial
floors here are strong and must be stated:

| | P | R | F1 |
|---|---|---|---|
| FLOOR always-fire | 0.571 | 1.000 | 0.727 |
| FLOOR any 6+-digit run | 0.740 | 0.343 | 0.468 |
| FLOOR capitalised-multiword | 0.725 | 0.269 | 0.392 |
| **best arm** — `ceiling-02 b-deepseek` | **0.832** | **0.870** | **0.851** |

**+0.124 over the strongest floor**, and unlike the predicate table this is the *Approach-B* family
winning, not the compiled judge. So the two families are **complementary rather than ranked**: B is
a usable entity detector and a poor predicate detector (§7.5 above); the compiled judge is the
reverse. That is an argument for the tiered design running both, and it is not visible in either
span-level table.

**Caveats that bound this, and they are real.** The floor's perfect message-level recall is partly
the corpus's known 12-fragment leak — every positive contains a capitalised organisation name, so
the floor cannot miss, and none of these numbers transfers to a corpus without that property. The
local arms ran in a different harness (Playwright + WebLLM) from the hosted arms, though against the
same corpus, the same gold and the same `pred:`-emitted decision rule. And 0.905 is one arm on one
pass of a 19-positive gold; §10.1's variance warning applies to it exactly as it applies to the
span-wise figures.

**What follows.** §9.1 should report both metrics side by side, and §10.1's answer differs by which
one is asked: span-wise, the ceiling arms sit in the floor's neighbourhood; message-wise, the best
compiled judge separates from it clearly. Neither of those was the document's original claim, which
was that they stop at the floor. This is queued with the scorer work in §11.2 and is **not** in the
§9 tables, which remain span-wise throughout.

---

## 8. How the run actually went, including the interruption

Pass 1 was launched with `SIH_CEILING_PASSES=3`. It was **stopped externally after seven of its ten
arms**, with the spend guard nowhere near tripping — **$0.138 spent against a $7.00 hard stop and a
$10 key limit**. The constraint on this experiment was wall-clock time, not money, and saying so
matters: none of the guard's stop paths fired, so nothing below is a partial-arm artefact of the
budget.

The three unfinished arms (`ceiling-b-nemotron`, `ceiling-judge-qwen3.8-27b`,
`ceiling-b-qwen3.8-27b`) were then re-run to completion under the same `runId`, same code, same
pins. `ceiling-judge-nemotron` was re-run alongside them and its file overwritten, so all four of
those arms come from one contiguous window rather than straddling the interruption. The first
window's ledger is preserved as `runs/ceiling-ceiling-01.part1.spend.json`; the final
`runs/ceiling-ceiling-01.spend.json` covers the second.

**Passes 2 and 3 were launched separately, after pass 1 was analysed.** The `$2.50` gate that guards
them was never reached — pass 1 cost $0.39 in total — so they were run as
`SIH_CEILING_PASS_START=2 SIH_CEILING_PASSES=2`, writing `ceiling-02` and `ceiling-03`.

That flag exists because launching them the obvious way **would have destroyed pass 1**. The inline
run-id arithmetic stripped a trailing `-\d+` and re-appended the *pass* number, so
`runId=ceiling-02, passes=1` resolved to `ceiling-01` and would have silently overwritten all ten
pass-1 arm files. It was caught by printing the mapping before launching rather than trusting it,
extracted into a tested `passRunIdFor`, and the driver now announces
`will write passes: ceiling-02, ceiling-03` before it starts. A mutant that restores the old
arithmetic is killed by the suite.

What repeats can and cannot show here: every call is made at `temperature: 0`, so a repeat measures
**provider and routing variance, not sampler variance**. The accuracy columns should be near-stable;
the latency columns will not be, and the 429 counts in §7.3 show why. With **19 positives**, one
item changing hands moves F1 by roughly 0.05 — which is larger than the 0.006 gap between the best
thinking-off arm and the floor, so the repeat passes in §9 are what say whether that gap is real
or noise. (This sentence previously pointed at a "§12"; this document ends at §11.)

### 8.1 What stopped the first window: unknown, and not a deliberate kill

**I did not kill it.** The one `pkill -f ceiling-main.ts` issued in this session was minutes
earlier and targeted a *different* process — the standalone probe run (`runId=probe`), killed
deliberately while isolating the Alibaba streaming hang, before the `ceiling-01` driver was
launched at all.

What was observed of the `ceiling-01` process, and nothing more: the harness reported the
background task as stopped; `run.log` ends after the Nemotron-judge arm line with **no `[stop]`,
no error and no `[ceiling] done`**; the ledger's `tripped` is `false` and its last write is
08:52:24Z; no ceiling process was alive afterwards. **The cause is unknown.** It was not the spend
guard, which had $6.86 of headroom, and it was not an unhandled error, which would have been
written. Recording it as unknown rather than guessing, per standing-conventions §3 — an
interruption with no stated cause is a gap a reader fills with the worst assumption.

### 8.2 Whether the code differed between the two windows: it did, and none of it reaches a call

Both windows stamp `gitSha bbdbfb1` with `gitDirty: true`, so **the stamps cannot distinguish the
two** — the dirty flag is honest but not discriminating. What changed between the launches, in full:

| file | change | reaches a call? |
|---|---|---|
| `ceiling.ts` | added `REASONING_ON`, `ThinkingRequest`, `reasoningRequestFor`, `ChatCall.thinking`; `reasoning: REASONING_OFF` → `reasoning: reasoningRequestFor(call.thinking ?? "off")` | **no** — see below |
| `ceiling.ts` | `thinkingRequested` widened from `literal("off")` to `enum(["off","on"])` | no — record field |
| `ceiling.ts` | added `applyPinOverrides` | **no** — window 2 passed no override |
| `ceiling.ts` | added optional `maxTokens` / `localArmMaxTokens` record fields | no — written, never sent |
| `ceiling.ts` | docblock edits (`wallMs` clocks) | no |
| `ceiling-main.ts` | `SIH_CEILING_THINKING`, `SIH_CEILING_PIN` wiring; `LOCAL_ARM_MAX_TOKENS`; corrected `MAX_TOKENS` comment; one log line | no — `MAX_TOKENS` stayed **600** |
| `ceiling-score.ts` | reporting only | no — not loaded by the driver |

**Untouched between the launches:** `callChat`, `consumeStream`, `consumeWhole`, `collectJudge`,
`collectBaseline`, the `parseJudgeResponse`/`parseBaselineResponse` calls, the span-ladder calls and
every timing statement.

Three tests now **pin** the equivalence rather than leaving it argued:

- `buildRequestBody(CALL)` deep-equals a literal of exactly what window 1 sent, `max_tokens: 600`
  and `reasoning: {enabled: false}` included;
- `reasoningRequestFor("off") === REASONING_OFF` — identity on the same frozen object, so the one
  changed line in the request path is a no-op for every thinking-off call;
- `applyPinOverrides(models, undefined)` returns the slate unchanged.

And the **records agree**. Across all 1,512 thinking-off rows from both windows, `reasoningRequest`
is `{"enabled":false}`, `thinkingRequested` is `off` and `outputMechanism` is
`provider-json-schema`, without exception. The **only** difference on disk is that window-2 rows
carry `maxTokens: 600` and window-1 rows omit the field — which is why it is optional in the schema:
back-filling it would be inventing provenance.

**Conclusion: behaviourally one run; a record-shape difference in the arms produced after the
edits.** The doc presents them as one run on that basis, and this table is the evidence for it.

### 8.3 Both ledgers are on disk, and joined

The completion run reused `runId=ceiling-01` and therefore overwrote
`runs/ceiling-ceiling-01.spend.json`. Window 1 was copied to
`runs/ceiling-ceiling-01.part1.spend.json` **before** the relaunch, so nothing was lost.
`pnpm -C apps/eval ceiling:ledger` joins all four segments into
`runs/ceiling-combined.spend.json` and reconciles the total against `GET /auth/key`.

The residual between the ledgers and the key is reported **with its sign and without a story**. It
has two known contributors pulling opposite ways — diagnostic curls made outside the driver (which
push the key *above* the ledgers) and the key endpoint's accounting trailing the per-response
`usage.cost` (which pushes it *below*) — and the script cannot attribute it.

**Correction.** An earlier draft of this paragraph said the diagnostic-spend claim was withdrawn
because *"the measured sign was negative, which contradicts that"*. The measured residual is
**+$0.002626 — positive** (§6). The retraction was reasoned from a figure that appears in no
artifact. The honest statement is the joiner's own: a positive residual is **consistent with**
diagnostic spend outside the driver and does **not** establish it, because the lagging-accounting
contributor pushes the other way and neither is separately measured. The claim stays out, but for
that reason, not the one previously given.

**The same `runId` collision later destroyed the window-2 ledger, and it has been restored.** The
pass-2 launch wrote its ledger to `runs/ceiling-ceiling-01.spend.json` again, replacing window 2's
768 calls / $0.19269 with a 489-call partial. Running the `regenerateWith` command above in that
state would have silently rewritten the join from $0.386071 to $0.217950 — the reconciliation
destroying its own input. Window 2 was rebuilt from the segment preserved inside
`runs/ceiling-combined.spend.json` (a ledger segment carries every field of the ledger file it came
from, plus three the joiner adds), and the killed partial was moved to
`runs/orphaned/killed-pass2-launch.ceiling-ceiling-01.spend.json` rather than deleted. The four
files on disk now re-derive **2,461 calls / $0.386071** — the committed total — so the documented
command reproduces the documented number again. The code defect behind the collision is fixed in
`fd2087a`; §11 records what is still untested about that fix.

---

## 9. The table — every arm, both levels, with the floors in it

Generated by `pnpm -C apps/eval ceiling:score`. **All 11 ceiling arms honoured their provider pin on
every answered call.**

### 9.1 Predicate level — `pred:client-relationship-disclosure`

189 gold rows, 179 scored, 10 disputed, **19 positives**. Identical under all three match rules for
every ceiling arm (each emits one span per finding, and the gold span is that span), so one column
suffices; the local arms' three-rule spread is in the generated output.

| arm | kind | P | R | **F1** | findings |
|---|---|---|---|---|---|
| `ceiling-judge-glm-5.3-flash` **thinking ON, 22% truncated** | ceiling | 0.625 | 0.526 | **0.571** | 16 |
| **FLOOR — `first-capitalised-multiword`** | floor | 0.467 | 0.737 | **0.571** | 30 |
| `ceiling-judge-deepseek-v4-flash-0731` | ceiling | 0.481 | 0.684 | 0.565 | 27 |
| `ceiling-judge-qwen3.8-27b` | ceiling | 0.421 | 0.842 | 0.561 | 38 |
| **FLOOR — `capitalised-multiword`** | floor | 0.380 | 1.000 | 0.551 | 50 |
| `ceiling-judge-nemotron-3-super-120b-a12b` | ceiling | 0.545 | 0.316 | 0.400 | 11 |
| `ceiling-judge-qwen3.8-flash` | ceiling | 0.213 | 0.842 | 0.340 | 75 |
| `ceiling-b-mistral-small-2603` | ceiling | 0.200 | 0.263 | 0.227 | 25 |
| **`tier2-Qwen3-4B` — best LOCAL arm** | local | 0.143 | 0.316 | 0.197 | 42 |
| **FLOOR — `whole-message`** (overlap only) | floor | 0.106 | 1.000 | 0.192 | 179 |
| `ceiling-b-qwen3.8-flash` | ceiling | 0.500 | 0.105 | 0.174 | 4 |
| `tier2only-Qwen3-4B` | local | 0.115 | 0.316 | 0.169 | 52 |
| `tier2-Phi-4-mini` | local | 0.070 | 0.474 | 0.122 | 129 |
| `ceiling-b-deepseek`, `ceiling-b-qwen3.8-27b`, `ceiling-judge-mistral` | ceiling | 0.000 | 0.000 | 0.000 | 4/3/1 |
| `ceiling-b-nemotron` | ceiling | — | 0.000 | — | **0** |
| all 6 local `baselineB*` arms on 3 of 4 models | local | — | 0.000 | — | 0 |

**Under every rule — exact, overlap, iou50 — the scorer's own verdict is `arms beating it: NONE`.**

### 9.2 Span level — entity gold (108 spans over 189 items), MODEL-ONLY arms

Only Approach-B-shaped arms emit entity spans; the judge family emits the shadow predicate alone by
construction and has no span score. **The eight local `tier2-*` and `baselineB+tier0-*` arms are
excluded from this table entirely** — they run tier 0, so their spans are the compiled regex layer's
and not their model's, which is why they cluster at P≈0.36 whatever model they name. `ceiling-score.ts`
prints them in a separate "NOT model-only" block below the table for completeness.

| arm | kind | findings | tp | fp | fn | P | R | **F1** |
|---|---|---|---|---|---|---|---|---|
| **FLOOR — orthographic oracle, budget-matched** | floor | — | — | — | — | 0.705 | 0.685 | **0.695** |
| `ceiling-b-nemotron-3-super-120b-a12b` | ceiling | 123 | 78 | 45 | 30 | 0.634 | 0.722 | **0.675** |
| `ceiling-b-qwen3.8-27b` | ceiling | 241 | 102 | 139 | 6 | 0.423 | 0.944 | 0.585 |
| `ceiling-b-deepseek-v4-flash-0731` | ceiling | 197 | 87 | 110 | 21 | 0.442 | 0.806 | 0.570 |
| `ceiling-b-mistral-small-2603` | ceiling | 196 | 80 | 116 | 28 | 0.408 | 0.741 | 0.526 |
| `ceiling-b-qwen3.8-flash` | ceiling | 261 | 96 | 165 | 12 | 0.368 | 0.889 | 0.520 |
| **FLOOR — orthographic oracle, unbudgeted** | floor | — | — | — | — | 0.307 | 0.870 | 0.454 |
| `baselineB-Qwen3.5-2B` — best model-only LOCAL arm | local | 43 | 25 | 18 | 83 | 0.581 | 0.231 | **0.331** |

Best model-only span arm **0.675** against a budget-matched oracle floor of **0.695**: still below,
by 0.020. Every ceiling B arm beats the *unbudgeted* oracle (0.454) and the best local model-only arm
(0.331); none beats the budget-matched one.

---

## 10. The six questions

### 1. Does ANY model — local or ceiling — beat the trivial floors? By how much?

**No. Not one, at either level, under any match rule.** The scorer's own verdict line reads
`arms beating it: NONE` for exact, overlap and iou50.

| level | best arm | best floor | gap |
|---|---|---|---|
| predicate | `ceiling-judge-glm` **0.571** (thinking on, 22% truncated) | `first-capitalised-multiword` **0.571** | **0.000 — a tie, not a win** |
| predicate, thinking-off only | `ceiling-judge-deepseek` **0.565** | **0.571** | **−0.006** |
| span | `ceiling-b-nemotron` **0.675** | oracle budget-matched **0.695** | **−0.020** |

The ceiling arms are a large improvement **over the local arms** — 0.565 against 0.197 at the
predicate level is **2.87×**, and 0.675 against 0.331 at the span level is **2.04×** — and that
improvement takes them to *level with a regular expression* and no further. Two independent gold
sets and two independent floors agree.

This is the answer §4.2b was built to get, and it is the unwelcome one: **the browser constraint is
not what is costing accuracy.** Making the model 30–60× larger and giving it a datacentre GPU moves
the numbers up to roughly the floor. On this corpus and this policy, the task itself is not being
solved decisively by any of these methods.

> **Correction (pass 2). An earlier draft of this paragraph ended "moves the numbers up to the floor
> and stops." The repeat pass this document commissioned to settle that has settled it the other
> way, and the word "stops" is withdrawn.**
>
> Re-scoring with pass 2 present, `ceiling-judge-deepseek-v4-flash-0731 [ceiling-02]` scores
> **P 0.485 / R 0.842 / F1 0.615** at the predicate level, and `ceiling:score` names it under all
> three matching rules: *"best floor F1 0.571; arms beating it: ceiling-judge-deepseek-v4-flash-0731
> [ceiling-02]"*. That is **+0.044 over the floor**, where pass 1 was −0.006.
>
> | pass | predicted | tp | fp | fn | F1 | vs floor 0.571 |
> |---|---|---|---|---|---|---|
> | `ceiling-01` | 27 | 13 | 14 | 6 | 0.565 | −0.006 |
> | `ceiling-02` | 33 | 16 | 17 | 3 | **0.615** | **+0.044** |
>
> Both passes are `temperature: 0`; the difference is provider and routing variance, not the
> sampler. **Three items changed hands** on a 19-positive gold — and §8 computes that one item is
> worth ≈0.05 F1 here, so the whole pass-to-pass swing is three items and the original "gap" it was
> compared against was one-eighth of a single item. A tie and a −0.006 gap were never enough to
> support "stops"; the defensible claim on this sample is that **the ceiling arms land in the
> neighbourhood of the floor, with pass-to-pass variance larger than their distance from it.**
>
> Pass 3 was in flight when this correction was written and is not in these figures. §9 carries the
> per-pass table once all three land; this paragraph is rewritten against the full variance then,
> not before.
>
> **Two further corrections bear on this paragraph and pull the same way.** §7.4: pass 1 was
> rate-limited off a gold positive it never got to attempt, and the scorer charges that as a miss —
> removing it moves pass 1 to 0.577. §7.5, larger: the predicate gold is a **message-level** label
> scored span-wise here, and on the judgment the gold actually records, **both** DeepSeek judge
> passes beat the floor — 0.905 and 0.811 against 0.776. The span-wise tie is real and so is the
> message-level separation; they say different things about the same rows, and "the task itself is
> not being solved by any of these methods" is not supportable as written. §7.5 goes further: with
> the local arms recomputed on the same 179 rows, the message-level ordering is **local 0.275 <
> floor 0.776 < ceiling 0.905**, which *inverts* this section's answer to spec §4.2b — on that
> reading the gap **is** substantially the price of the in-browser constraint.

### 2. Compiled (judge) or prompting (B) at the ceiling — and does it differ from the local result?

**At the predicate level the compiled family wins, decisively — but the margin is model-dependent,
not structural.**

| model | judge F1 | B F1 |
|---|---|---|
| deepseek-v4-flash | **0.565** | 0.000 |
| qwen3.8-27b | **0.561** | 0.000 |
| nemotron-3-super-120b | **0.400** | — (0 findings) |
| qwen3.8-flash | **0.340** | 0.174 |
| mistral-small-2603 | 0.000 | **0.227** |

Four of five models are better compiled; **Mistral inverts it completely**, and the mechanism is
visible in the emissions: asked about one predicate, Mistral says almost nothing (1 finding in 189
messages); asked about nine entity classes, it names the predicate 29 times. DeepSeek does the
opposite — 30 predicate findings as a judge, 7 as B. So "compiled beats prompting" is a claim about
four of these five models, not about the two methods.

**Yes, the answer differs from the local result, and the reason differs more than the answer does.**
Locally Approach B *could not finish inside the budget at all* — on three of four models it answered
**zero** calls, budget-exhausted 13 of 13, because its 1,410-token prompt blew the 5,000 ms message
budget. Here B finishes everywhere: **five of five models, 179/179 items answered on three of them,
zero parse failures.** It is not a latency failure any more. It is a *task* failure — B spreads 204
findings across nine entity classes and names the relationship predicate 7 times. Removing the
budget constraint revealed that B's problem was never only the budget.

And at the **span** level, B is the only family with numbers at all, and its best is the best
ceiling result in this whole experiment (0.675). The two families are good at different things.

### 3. Reasoning tokens with thinking off: zero everywhere, or not?

**Zero everywhere it could be asked — and one model would not be asked.**

- **Ten of ten thinking-off arms reported `reasoning_tokens` p50 0 and max 0**, across **1,854
  calls**, on five models and five providers. **No model leaked a single reasoning token.**
  `reasoning: {enabled: false}` was honoured exactly.
- **`z-ai/glm-5.3-flash` cannot be asked.** It returns HTTP 400 *"Reasoning is mandatory for this
  endpoint and cannot be disabled"* on **all eleven of its providers** (§5). Not a leak — a refusal.
  OpenRouter's metadata does not advertise it: `reasoning_config` is `null` and
  `supported_parameters` lists `reasoning`. **Only a live call reveals it**, which is the
  transferable lesson for anyone planning a thinking-off slate from the catalogue.

Run thinking-**on**, GLM emitted reasoning p50 **228** and max **600** — the cap — on 216 calls.

### 4. Latency, and what it implies for a DGX Spark

**State plainly: TTFT and decode rate are PROVIDER facts.** They describe whose GPU answered, under
what load, behind a US aggregator. They do not transfer. `reasoningTokens` and `completionTokens`
do — they are properties of the model and the task.

| arm | TTFT p50/p95 ms | decode tok/s p50 | **call** wall p50 ms | **item** wall p50 ms | 429s |
|---|---|---|---|---|---|
| `judge-mistral-small-2603` | 345 / 639 | 170.6 | **385** | 386 | 0 |
| `judge-qwen3.8-27b` | 545 / 908 | 112.9 | 650 | 651 | 0 |
| `b-mistral-small-2603` | 331 / 603 | 208.2 | 772 | 774 | 0 |
| `judge-qwen3.8-flash` | 718 / 5697 | 82.6 | 1044 | 1190 | 5 |
| `judge-nemotron-3-super-120b` | 755 / 1180 | 14.3 | 1262 | 1263 | 0 |
| `judge-deepseek-v4-flash` | 1198 / 4828 | 726.7 | 1325 | 2127 | 154 |
| `b-qwen3.8-27b` | 604 / 894 | 109.9 | 1404 | 1405 | 0 |
| `b-deepseek-v4-flash` | 890 / 2297 | 89.3 | 2000 | 4089 | 161 |
| `b-qwen3.8-flash` | 880 / 8591 | 113.3 | 2600 | 4249 | 25 |
| `b-nemotron-3-super-120b` | 745 / 1208 | 13.7 | 5056 | 18724 | 0 |
| `judge-glm-5.3-flash` **thinking ON** | 4476 / 10891 | 1017.4 | 5444 | 4885 | 0 |

**The transferable number.** `reasoningTokens ÷ 60 tok/s` — a DGX Spark-class decode budget for a
model this size — is **0.00 s for every one of the ten thinking-off arms**, because every one
reported zero reasoning tokens. Reasoning is simply not a cost in the thinking-off condition.

The cost that *is* real is total decode. At 60 tok/s the **median completion** implies:

- **judge family: 0.08–0.12 s** (5–7 tokens — these models answer this predicate in one short JSON
  object);
- **Approach-B family: 0.95–1.80 s** (57–108 tokens — nine classes to report on);
- **GLM thinking-on: 3.92 s** (235 tokens, of which **228 are reasoning**) — and its *max* is 600
  tokens, **10.0 s of pure decode**, 22% of the time producing nothing usable because the cap cut it
  off.

So on a DGX Spark the compiled judge is roughly a **tenth of a second of decode per message** and
mandatory reasoning is **~33× that**, for a score that ties rather than beats it.

### 5. Structured-output compliance

**Perfect, on every thinking-off arm.**

| population | calls | parse failures | repairs | truncated |
|---|---|---|---|---|
| all ten thinking-off arms | **1,854** | **0** | **0** | **0** |
| `judge-glm-5.3-flash` thinking ON | 216 | **48 (22.2%)** | 27 | 48 |

`response_format: {type: "json_schema", strict: true}` — carrying `JUDGE_SCHEMA` / `BASELINE_B_SCHEMA`
unchanged, `minimum`/`maximum` bounds included — was accepted by **all five** answering providers and
produced **zero** malformed or schema-invalid bodies in 1,854 calls. The one repair turn was never
needed. This is a genuinely different result from the local arms, where Phi-4-mini failed to parse 3
of 6 calls in a probe.

**Every one of GLM's 48 failures is `finish_reason: "length"` — truncation, not malformation.** That
is a finding about the *token budget* meeting mandatory reasoning, not about the mechanism. The
provider-side json_schema mechanism itself did not fail once in this experiment.

The span ladder was nearly clean: pooled over all ten pass-1 arms, **2 unresolved quotes and 8
unresolved mentions on 1,243 findings** — 0.8 per 100. (An earlier draft of this line said 0 and 0
"across every ceiling arm"; that was the DeepSeek pair's figure generalised to the slate.) At this
model size, quoting a clause verbatim and pointing at a shorter span inside it
is a solved problem. Whatever is failing, it is not span extraction — which restates the local
finding that *the binding constraint is classification, not span extraction*, now at 30–120B.

### 6. Spend

**$0.386 total against a $10 key limit and a $7.00 hard stop. The guard never tripped.**

| segment | calls | cost |
|---|---|---|
| probe | 11 | $0.00121 |
| `ceiling-01` window 1 (arms 1–7) | 1,457 | $0.13645 |
| `glmon-01` (GLM thinking on) | 225 | $0.05572 |
| `ceiling-01` window 2 (arms 8–10) | 768 | $0.19269 |
| **ledger total** | **2,461** | **$0.38607** |

Per model (thinking-off arms, both families): qwen3.8-27b $0.116, nemotron $0.072, qwen3.8-flash
$0.032, mistral $0.025, deepseek **$0.013**. Per family: judge $0.074, B $0.184 — **B costs 2.5×
the judge**, which is its 1,410-token prompt on every call.

**Final `GET /api/v1/auth/key`: `usage` $0.388698, `limit` $10.** Residual against the ledger is
**+$0.002626** (the key exceeds the ledgers). Reported with its sign and no story attached:
diagnostic curls outside the driver push the key *above* the ledgers, the key endpoint's accounting
trailing per-response `usage.cost` pushes it *below*, and this cannot attribute between them. Both
figures are `keyUsageFinalUsd` and `residualUsd` in `runs/ceiling-combined.spend.json`.

An earlier draft of this line read *"`usage` $0.37858 … residual −$0.00749 (ledgers exceed the
key)"*. **That reading appears in no artifact**, and the sign was the wrong one; §8.3's retraction
was reasoned from a negative residual that is in fact positive. Corrected against the join.

**This reconciliation is point-in-time.** `ledgerTotalUsd` re-derives from four files on disk and is
stable; `keyUsageFinalUsd` is read live from the key endpoint, so re-running `ceiling:ledger` after
any later pass spends on the same key returns a larger key figure and a correspondingly larger
positive residual. The $0.388698 above is the reading at the moment pass 1 ended.

**The experiment was bounded by wall-clock time, not by budget** — it used 3.9% of the key.

---

## 11. What this arm did NOT fix, and one green test that is luck

Recorded so a reader does not credit this work with more than it did.

**`apps/eval/test/page-webgpu-floor.test.ts` passes, and that is install-state luck, not a fix.**
It calls `require.resolve("@mlc-ai/web-llm")` at line 51, and **`@mlc-ai/web-llm` is not declared in
`apps/eval/package.json`** — neither a dependency nor a devDependency. It resolves only because the
package happens to be reachable through the `@sih/tier2` workspace link in this particular pnpm
install. The only manifest change this work made was adding `vite-node`; the undeclared dependency
is untouched and remains a latent break under a clean or differently-hoisted install. It is being
declared separately in a cleanup commit and is **not** part of this change.

**Also not addressed here:**

- The ledger filename doubles its prefix (`runs/ceiling-ceiling-01.spend.json`) because `runId`
  already begins with `ceiling`. Left alone deliberately: renaming it mid-experiment would make the
  artifacts disagree with the code that produced them.
- `p-med` and `p-corp` still have no compiled IR, so this arm — like every other — is p-fin only.
- The corpus's known open leak (12 fragments in `families.v2.ts` decide every label) is unchanged
  and bears on these numbers exactly as it bears on the local arms'.

### 11.1 The ledger fix is still not tested at the site that broke

`fd2087a` extracted `spendLedgerFileFor` so the ledger filename would be pinned by a test. **The
function is pinned; the call site is not.** Reverting `ceiling-main.ts:308` from
`spendLedgerFileFor(runId, passStart)` back to `` `ceiling-${runId}.spend.json` `` — byte-for-byte
the defect that destroyed a 768-call ledger — **survives all 791 tests.** Verified directly:
mutation applied by exact-string replace, md5 confirmed changed, suite green, restored by absolute
path with md5 confirmed returned.

**The commit message for `fd2087a` says "Both mutants — the real defect and ignoring `passStart` —
are now killed." That is wrong.** Only the second is. The extraction moved the boundary up one
layer and did not close it.

The cause is structural: `ceiling-main.ts`, `ceiling-score.ts` and `ceiling-ledger.ts` each end in a
top-level `main()` and **export nothing**, and `apps/eval/test/ceiling.test.ts` imports only
`../src/driver/ceiling.js`. Nothing in the workspace imports the three scripts; they are reachable
only through their `pnpm` aliases. **All 29 mutants planted across those three files survived**,
including: the spend guard never fed (`onCall` hook deleted), the mid-arm and between-arms stop
paths disabled, `gitDirty` hardcoded, arm files named by the base `runId` instead of the pass id,
the probe verdict forced to "pass", the final ledger write deleted, and — in `ceiling-ledger.ts` —
the residual **sign flipped** and `guardEverTripped` hardcoded false. That last pair matters to this
document directly: §6 and §8.3 above both had sign errors, and no test could have caught either.

### 11.2 Mutation coverage, measured rather than asserted

An adversarial review ran **169 mutants: 58 killed, 110 survived**, against a harness proved honest
in both directions (three positive controls killed, two negative controls survived, checked at the
start and the end of the run). Two survivors are intended controls and three are provably
equivalent, leaving **105 real coverage gaps**. The previously reported "35/35 mutants killed" was
not false — the mutants in that set are genuinely killed, including the retry-clock mutant this
document relies on in §7.3 — but it measured a narrow band. What it missed is everything **one
layer out from the unit under test**:

- The three script files entirely (above).
- **21 mutants of `runCeilingItem`'s record write survive**, because every fixture uses a case where
  the *requested* value and the *actual* value are the same string. `provider: lastProvider` can be
  replaced by `provider: model.provider` — **the pin recorded as the fact** — and the suite stays
  green, which makes §9's "all 11 arms honoured their provider pin" unfalsifiable by the tests. The
  same holds for `modelId`, and for `reasoningTokens ?? null → ?? 0`, the `null`-is-not-zero
  distinction this arm's whole thinking-off claim rests on.
- **No fetch mock inspects `init.body`**, so sending the wrong family's schema, the wrong family's
  messages, or dropping `reasoning: {enabled: false}` from the wire while the row still records
  `thinkingRequested: "on"` all survive. The "same method, bigger model" claim is unverified at the
  caller.

None of this is evidence that the reported numbers are wrong — the artifacts were separately
re-derived and the accuracy, span, transport, token-budget and ledger tables all reproduce. It is
evidence that **the tests would not have caught it if they were**, which is a different and weaker
guarantee than the 35/35 figure implied.

### 11.3 A docblock that justifies the cost model is arithmetically false

`ceiling.ts:138-141` justifies ordering the slate by a prompt-weighted cost with: *"the unweighted
`in + out` sum gives a DIFFERENT order: under it GLM-Flash (0.15+0.50) sorts ahead of Qwen-Flash
(0.15+0.47) … Ordering by the sum would have put the more expensive of the two first."*

**It does not.** 0.62 < 0.65, so an ascending sort by the unweighted sum puts Qwen-Flash first —
the same as the shipped order. More strongly, the two models have **equal input rates** (0.15
each), so under any positive weighting the comparison reduces to the output rate, 0.47 < 0.50. The
named pair provably cannot distinguish the two orderings, and the test that cites this reasoning
cannot make the distinction it claims. Mutation confirms it: changing the representative token mix
from 1700/200 to 200/1700 survives.

The weighting is still the right choice — nemotron (0.30/0.65) and qwen-27b (0.24/2.20) genuinely
flip at a prompt/completion ratio near 25.8, and *that* pair is what the test is actually sensitive
to — but the stated justification names the one pair where it cannot hold. Standing-conventions §1.

### 11.4 Provenance of passes 2 and 3

`gitProvenance()` is called **once**, at `ceiling-main.ts:260`, and its values are stamped on every
row. Rows from passes 2–3 therefore record `gitDirty: false` at `8c4fc8b` as the state **at launch**,
not throughout. Documentation edits — including this section — were made to the working tree while
those passes were running. The code that produced the rows is exactly `8c4fc8b`: the driver has no
dynamic imports, so every module was read and cached at startup, and `ceiling.ts` / `ceiling-main.ts`
were deliberately left untouched until all three passes finished, so that a relaunch could not
silently mix code versions across passes.

