# AI-DLPP · AI Data Leak Prevention Plugin

A browser extension that checks what you paste into ChatGPT, Claude or Gemini **before it is sent**, and strips out confidential data. The check runs **on your own machine**.

**Research prototype.** Full study: [paper, 7 pages](docs/paper/paper.pdf) · [summary, 1 page](docs/paper/executive-summary.pdf)

```mermaid
flowchart LR
  A["Your prompt"] --> B["Pattern rules<br/>keys, IDs, account numbers"]
  B --> C["Small AI model<br/>does this reveal a client?"]
  C --> D{"Leak?"}
  D -- no --> E["Send"]
  D -- yes --> F["Redact or block"]
```

## The answers

We tested 4 small models inside Chrome, 5 bigger hosted models, and one model of a different kind, on 189 realistic work messages under a bank-style data policy.

**1. Can a small AI model in the browser catch leaks in real time?**
**Partly.** It is fast and free, and its pattern rules catch keys and ID numbers. But it cannot make the policy's judgment calls: it scores worse than a one-line rule with no AI.

**2. Would bigger models do better?**
**Yes, on accuracy.** The best one beats that rule and stops more leaks. But only 1 of the 5 did, and none of them run inside a browser.

**3. Would a different kind of model do better still?**
**Yes, and by the widest margin.** A judgment model returns a probability instead of text, so code finds the candidate spans and the model only keeps or rejects them. It never flagged a single clean message, in any of three runs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme-understanding-dark.svg">
  <img alt="Policy-understanding score, 0 to 1. TypeSafe Jev 0.96, DeepSeek V4 Flash 0.88, Qwen3.8 27B 0.77, Qwen3.8 Flash 0.47, Nemotron 3 Super 120B 0.37, best small model in browser 0.26, Mistral Small 0.21. A no-AI rule scores 0.78." src="docs/assets/readme-understanding-light.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme-prevention-dark.svg">
  <img alt="Out of 100 leaks: in browser stops 66, hosted DeepSeek 84, judgment model 87. Out of 100 safe prompts: in browser flags 25, hosted 21, judgment model 25, or 12 with the threshold turned up." src="docs/assets/readme-prevention-light.svg" width="760">
</picture>

## At a glance

| | Small model, in the browser | Best bigger model, hosted (DeepSeek) | Judgment model, hosted (TypeSafe Jev) |
|---|---|---|---|
| Understands the policy | ❌ Worse than a no-AI rule | ✅ Beats it, the only 1 of 5 that does | ✅ Beats it by the widest margin |
| False alarms on the policy question | ❌ 34 of 160 clean messages | ✅ 3 to 4 of 160 | ✅ **0 of 160, in every run** |
| Leaks stopped | ⚠️ 66 in 100 | ✅ 84 in 100 | ✅ 87 in 100, or 78 with fewer false alarms |
| Safe prompts flagged by mistake | ⚠️ 25 in 100 | ⚠️ 21 in 100 | ⚠️ 25 in 100, or 12 with the dial turned up |
| Time per message | ✅ About 1 second | ✅ 1 to 2 seconds | ✅ 0.28 seconds |
| Cost | ✅ Free | ✅ About $0.04 per 1,000 messages | ✅ About $0.12 per 1,000 messages |
| Prompt stays on your machine | ✅ Yes | ❌ No, unless you host it yourself | ❌ No, unless you host it yourself |

## Bottom line

Keep the in-browser checker for fast, free catching of keys and ID numbers. For judgment calls like "does this message reveal a client?", send the candidates to a judgment model on a server you control. The shape matters more than the vendor: code proposes the spans, a small model keeps or rejects them, and you choose where the threshold sits.

---

[Paper](docs/paper/paper.pdf) · [One-page summary](docs/paper/executive-summary.pdf) · [Research records](docs/research/) · [Technical notes and how to run it](docs/technical-notes.md)

All test data here is synthetic. Every key, ID number, name, email and policy document was written or generated for the study.

Released under the [MIT License](LICENSE).
