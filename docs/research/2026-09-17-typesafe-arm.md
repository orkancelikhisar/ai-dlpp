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
