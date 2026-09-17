# The TypeSafe arm: structure, and what it can and cannot be compared with

**Status:** structure built and exercised end to end with a keyless dry run; no real call has
been made. Written 2026-09-17, before the API key exists, so the thresholds and the headline
metric are chosen here rather than after seeing the numbers.

## 1. Why this arm is not shaped like the others

Every tier-2 arm in this repository so far asks a model to WRITE a JSON object carrying a label
and a quoted clause, then recovers a span from the text it wrote. TypeSafe's System One model
(Jev) cannot do that. `POST /v1/systemone` takes state plus named questions and returns a typed
answer per question: a `noul` is a probability of yes, a `choice` is one option from a map the
caller supplies plus the whole distribution over it. There is no generated text, no stream, no
reasoning trace.

So the decomposition inverts: **code proposes, the model judges, code places the span.**

| Question | Primitive | Where the span comes from |
|---|---|---|
| Does the message disclose a client relationship (`client-relationship-disclosure`)? | `noul`, instructions = the IR's own `nlPredicate` | nowhere: a noul returns a probability and no clause |
| Is this candidate span an entity the policy governs, and which one? | `choice` per candidate, criteria = the IR's 8 entity types with their `nlDefinition`, examples and counter-examples, plus `not-confidential` | the regex or oracle that proposed the candidate |

Candidates come from `runTier0` and `orthographicOracle`, both already in this repository and
both recall-tuned on purpose. That choice is deliberate: the orthographic oracle is the paper's
own no-model floor, so "what does a judgment add on top of the floor's candidates" becomes a
directly countable question rather than an inference.

One request per message carries the predicate and every candidate question. The docs state that
questions in a request are answered independently and that the document dominates the bill, so
batching costs nothing in accuracy and saves both money and round trips.

## 2. What this arm does not have, and will not fake

`runs/<id>.ts-judgment.gates.jsonl` carries a `notMeasured` block naming each one:

- **`ttftMs`, `decodeTokPerSec`** — one JSON body, no stream. There is no first token and no
  decode window. The hosted-LLM arms' latency columns therefore have no counterpart here; the
  comparable number is per-message wall time.
- **`reasoningTokens`** — System One returns a judgment, not reasoning.
- **span-wise predicate F1** — a noul returns no clause, so the predicate can be scored at
  MESSAGE level only. The paper's span-wise predicate column is not available for this arm and
  must be printed as a dash, never as a zero.
- **provider, quantization** — a single vendor endpoint with no provider routing to pin.

This is why the arm has its own record schema (`TypeSafeRecordSchema`) rather than reusing
`CeilingRecordSchema`. Borrowing that schema would mean writing nulls into fields that later
read as measurements, which is the defect §10 of the ceiling record exists to prevent.

## 3. Thresholds: decided before the data exists

A judgment arm returns a probability, so a single F1 is a choice, not a result. Records store
the RAW probability and `projectFindings` is the only place a threshold turns it into a finding.
Three numbers get reported, always together:

1. **At 0.5** — the default a caller gets with no tuning.
2. **Best threshold** — the maximum over the sweep, labelled FITTED, because the threshold was
   chosen on the rows it is then scored on.
3. **Split-half** — the threshold is chosen on one deterministic half of the items and spent on
   the other, both ways round, and averaged. **This is the headline number** and the only one
   comparable with the generative arms, which never got to tune anything.

Beside them: ROC-AUC and average precision, which measure the ranking independently of any
threshold, and a ten-bin calibration table, which asks whether a 0.7 means seven in ten. No
other arm in this study could be asked that question.

## 4. The filter effect

The entity job here is pure keep-or-reject over candidates the code found, so the model's
contribution is countable in four numbers: candidates on gold, candidates off gold, correct
rejections (precision the regex tier could not have), wrongful rejections (recall this arm
destroyed). The tier-0-only baseline is computed from the same candidate list with no model at
all, and on the dry run it reproduces the paper's in-browser pipeline exactly: prevention 0.667,
over-blocking 0.247. That agreement is the check that the scorer is measuring the same thing the
paper measured.

## 5. Cost model

`docs.typesafe.ai/models.md`, read 2026-09-17: input **$42 per billion tokens**, output free;
`jev-latest` resolves to `jev-1.13.0`, which is what the runner pins. Cost is computed from the
response's own `usage.input_tokens`, never estimated. The run carries a hard cap
(`SIH_TS_SPEND_CAP_USD`, default $2) and writes a ledger beside the rows.

The dry run's stub priced the full corpus at ~$0.016 for 189 messages, about **$0.08 per 1,000
messages**, on ~1,800 input tokens per request. Note where those tokens go: the 8 entity
definitions are repeated inside every candidate question, so input size grows with the candidate
count rather than with the message. A real run will report the true figure; this is the
expectation it will be checked against.

## 6. Comparability, stated up front

TypeSafe is a hosted API. Prompts leave the machine, so this arm answers the same question the
hosted slate answered: **what a larger, non-shippable judge could do**, not what the extension
can ship. It cannot appear in the in-browser column of any table.

What it can be placed beside, on the same 189 messages and the same gold:

| Paper number | Comparable? |
|---|---|
| Predicate F1, message level, vs the 0.776 no-model floor | yes, using the split-half number |
| Predicate F1, span-wise | no: no clause is returned |
| Entity span F1, prevention, over-blocking | yes, at a stated threshold |
| Per-message wall time | yes |
| Cost per 1,000 messages | yes |
| TTFT, decode rate, reasoning tokens | no: this transport has none |

## 7. Running it

```bash
# structure check, no key and no network: writes runs/ts-dryrun.*
SIH_TS_RUN_ID=ts-dryrun SIH_TS_DRY_RUN=1 pnpm -C apps/eval typesafe

# the real run, key from TYPESAFE_API_KEY or from .env at the repo root (gitignored)
SIH_TS_RUN_ID=ts-01 pnpm -C apps/eval typesafe
SIH_TS_RUN_ID=ts-01 pnpm -C apps/eval typesafe:score
```

Knobs: `SIH_TS_LIMIT` (probe a few items first), `SIH_TS_CONCURRENCY` (default 4, against a
documented 1,200 requests/minute), `SIH_TS_MAX_CANDIDATES` (default 24), `SIH_TS_SPEND_CAP_USD`,
`SIH_TS_TIMEOUT_MS`, `SIH_TS_MAX_ATTEMPTS`. The runner refuses to overwrite an existing arm file,
so each run needs a fresh id.

**Three passes, not one.** §9.3 of the ceiling record measured message-level F1 moving by up to
0.344 between identical passes at temperature 0. Nothing yet says a judgment model is steadier,
so the plan is `ts-01`, `ts-02`, `ts-03` and the spread reported beside the mean.

## 8. Known limits of the design

- **The candidate list is the recall ceiling.** The model cannot label a span nobody proposed, so
  this arm's entity recall is bounded by tier 0 plus the orthographic oracle. That is a property
  of the design, and the tier-0 baseline row is what makes it visible.
- **One policy, one predicate, 19 positives.** Same bound as every other arm here: one item moves
  recall by 0.053.
- **The corpus is synthetic**, and its generator gives every positive a capitalised organisation
  name, which is why the no-model floor is as strong as it is.
- **A dry run's numbers are noise by construction** (the stub hashes ids into probabilities). The
  dry run proves the pipeline, never the model. Its ROC-AUC came out at 0.538, near chance, which
  is the expected reading and a check that no label leaks into scoring.

---

# Results: three passes, 189 messages, 2026-09-17

Runs `ts-01`, `ts-02`, `ts-03`, model `jev-1.13.0`, concurrency 4. **189 of 189 answered on every
pass, zero errors, zero 429s, zero retries.** $0.021803 per pass, $0.0654 for the three.

## 9.1 The semantic predicate, message level

| | pass 1 | pass 2 | pass 3 | mean |
|---|---|---|---|---|
| F1 at the default 0.5 | 0.944 | 0.973 | 0.973 | **0.963** |
| precision / recall at 0.5 | 1.000 / 0.895 | 1.000 / 0.947 | 1.000 / 0.947 | 1.000 / 0.930 |
| F1, split-half (the honest number) | 0.971 | 0.971 | 0.971 | **0.971** |
| F1, best threshold (FITTED) | 1.000 | 1.000 | 1.000 | 1.000 |
| best threshold | 0.41 | 0.48 | 0.42 | 0.44 |
| ROC-AUC | 1.000 | 1.000 | 1.000 | **1.000** |

Beside the study's own numbers: the model-free floor is **0.776**, the best in-browser arm is
**0.262**, and the best hosted LLM judge, DeepSeek V4 Flash, is **0.881** mean over three passes
(0.927 at its best). This arm clears the floor on every pass at the untuned default, and its
ranking is perfect: in pass 1 the highest-scoring negative sits at 0.29 and the lowest-scoring
positive at 0.41, so ANY threshold in that gap separates the corpus completely. The two
"misses" at 0.5 are positives at 0.49 and 0.41, both still above every negative.

Calibration is one-sided rather than well spread: 160 of 179 messages land under 0.3 and every
one of them is a true negative; every message over 0.4 is a true positive. The model is not
producing a graded belief here, it is producing a near-separating score.

**Stability, which is the result the LLM arms could not deliver.** Message-level F1 moved by
0.029 across the three passes, against 0.109 mean and 0.344 max for the hosted LLM arms at
temperature 0 (§9.3 of the ceiling record). The predicate probability was identical on 88 of 189
messages and never moved by more than 0.070; candidate labels were identical on 407 of 417
(97.6%).

## 9.2 Entity spans, and what the judgment adds

| | tier 0 alone | TypeSafe at 0.5 (3-pass mean) |
|---|---|---|
| span F1, overlap rule | 0.459 | **0.551** |
| leak prevention | 0.667 | **0.870** |
| over-blocking | 0.247 | 0.251 |

Same candidates, same corpus: the judgment lifts prevention from two leak-bearing messages in
three to seven in eight **at the same false-alarm rate**. At a stricter 0.85 it runs the other
way, 0.778 prevention at 0.123 over-blocking, which is half the false alarms of the regex tier
alone. The knob is real and it is cheap to move, because the run stored probabilities.

The filter effect says it directly: of 101 tier-0 candidates that touch no gold span, the model
rejected **80** (79.2%); of 105 that do touch gold, it wrongly rejected **7** (6.7%).

For comparison from the paper, the hosted policy-in-context arms: DeepSeek 0.837 prevention at
0.210 over-blocking, Qwen3.8 27B 0.938 at 0.444.

## 9.3 Cost and time

$0.1154 per 1,000 messages, 2,490 input tokens p50 per message, 3 questions per message at the
median. **277-282 ms per message p50, 328-398 ms p95** — one request carrying every question.
The compiled LLM judge costs $0.0856 per 1,000 and runs 357-1,209 ms per call, and the best
in-browser arm takes 1,018 ms per message. So this arm is about 1.3x the judge's price and about
3x faster than the browser, for a score that clears a floor neither the browser nor four of the
five hosted models could.

## 9.4 What these numbers do not say

- **It is hosted.** Prompts leave the machine, so this is a Q2-style comparator and not a
  shippable tier, exactly like the OpenRouter slate. The design it argues for is the same one:
  a local server running the judgment, with the in-browser regex tier in front.
- **Perfect separation flatters everyone.** The corpus is synthetic and its positives are
  generated from templates; the same property is why the no-model floor reaches 0.776. A clean
  0.12 gap between the highest negative and the lowest positive is a statement about this corpus
  as much as about the model, and it will not transfer unexamined to real traffic.
- **19 positives.** One item moves recall by 0.053, which is most of the pass-to-pass movement.
- **Entity recall is capped by the candidate generator**, not by the model: tier 0 and the
  orthographic oracle propose, and nothing else can be found. The tier-0 baseline row is there
  so that ceiling stays visible.
- **The span-wise predicate column stays empty.** A noul returns no clause.
