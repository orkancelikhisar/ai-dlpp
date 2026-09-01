# Compilation report: p-fin

- Source policy hash (sha256): `ebb3cd68d973175f3ea40faeec00685e2cb9d83c6940e96a57e88ee269e8110a`
- IR version: `1`
- Fail mode: `closed`
- Latency budget: 5000 ms

## Stages

| Stage | Result |
| --- | --- |
| extract | 8/8 entityTypes, 10/10 rules, 1/1 semantic predicates and 31/31 action clauses quote the document verbatim |
| ground | 31 action clauses resolved to 8 policy defaults and 21 overrides across 3 providers |
| predicates | 1 shadow entityType minted (tier 2, never pseudonymized) |
| validate | 9 regex rules proven non-catastrophic in bounded time |
| self-test | 7 of 9 entityTypes measured against the real runtime, 2 not measurable today |
| emit | 9 entityTypes, 10 rules, 9 default actions, 20 provenance entries |

## Entity types

| id | tier | severity | default action | rules | clause |
| --- | --- | --- | --- | --- | --- |
| `in-pan` | 0 | critical | block | 1 | §2.1 |
| `in-aadhaar` | 0 | critical | block | 1 | §2.2 |
| `bank-account-identifier` | 0 | critical | block | 3 | §2.3 |
| `internal-customer-id` | 0 | high | block | 1 | §2.4 |
| `client-name` | 1 | high | pseudonymize | 0 | §3.1 |
| `api-credential` | 0 | critical | block | 2 | §4.1 |
| `db-connection-string` | 0 | critical | block | 1 | §4.2 |
| `private-key-material` | 0 | critical | block | 1 | §4.3 |
| `pred:client-relationship-disclosure` | 2 | high | redact | 0 | §3.4 |

## Outbound-visible identifiers

Every id below is transmitted to the LLM provider inside a redaction marker —
`[REDACTED:<id>]` — on every message where that entity is found. Read them as text a
third party will see: an id derived from the confidential value it protects discloses
that value even though the redaction worked.

| id | origin | clause |
| --- | --- | --- |
| `[REDACTED:in-pan]` | authored in the policy | §2.1 |
| `[REDACTED:in-aadhaar]` | authored in the policy | §2.2 |
| `[REDACTED:bank-account-identifier]` | authored in the policy | §2.3 |
| `[REDACTED:internal-customer-id]` | authored in the policy | §2.4 |
| `[REDACTED:client-name]` | authored in the policy | §3.1 |
| `[REDACTED:api-credential]` | authored in the policy | §4.1 |
| `[REDACTED:db-connection-string]` | authored in the policy | §4.2 |
| `[REDACTED:private-key-material]` | authored in the policy | §4.3 |
| `[REDACTED:pred:client-relationship-disclosure]` | minted for semantic predicate `client-relationship-disclosure` | §3.4 |

## Rejected candidates

Candidates the model proposed that did not reach the IR. Every rule and entity in a
compiled policy must quote the document verbatim; anything that cannot is discarded
here rather than shipped. A compile that dropped half the policy must not read like one
that dropped nothing.

None — every candidate grounded in the document, and every rule and action reached an
entityType that survived.

## Self-test coverage

Model-generated cases executed by the real tier-0 runtime (corpus tag `selftest-v1`).
**recall** counts a positive as caught when any entityType caught it — the leak-prevention
number. **label recall** counts only the positives this entity's own rules labelled; it is
lower exactly when a stricter overlapping entity won the span.

**"not measured" is not zero.** It means no case was executed against that entity, so the
compile makes no claim about it either way. The notes below the table say why.

| entityType | tier | positives | recall | label recall | hard negatives | false-positive rate |
| --- | --- | --- | --- | --- | --- | --- |
| `in-pan` | 0 | 20 | 100% (20/20) | 100% (20/20) | 20 | 5% (1/20) |
| `in-aadhaar` | 0 | 20 | 100% (20/20) | 100% (20/20) | 20 | 5% (1/20) |
| `bank-account-identifier` | 0 | 20 | 95% (19/20) | 95% (19/20) | 20 | 10% (2/20) |
| `internal-customer-id` | 0 | 20 | 50% (10/20) | 50% (10/20) | 20 | 0% (0/20) |
| `client-name` | 1 | not measured | not measured | not measured | not measured | not measured |
| `api-credential` | 0 | 20 | 90% (18/20) | 90% (18/20) | 20 | 25% (5/20) |
| `db-connection-string` | 0 | 20 | 85% (17/20) | 80% (16/20) | 20 | 0% (0/20) |
| `private-key-material` | 0 | 20 | 75% (15/20) | 75% (15/20) | 20 | 0% (0/20) |
| `pred:client-relationship-disclosure` | 2 | not measured | not measured | not measured | not measured | not measured |

- `client-name`: not measured — entityType is tier 1; only tier 0 executes today, so this compile cannot measure it — the tier 1 engine arrives in Plan 4.
- `db-connection-string`: 1 of 20 positives were caught under another entityType's label, which is why recall exceeds label recall. The value does not leak; this entity's surrogate and cited clause do not appear.
- `pred:client-relationship-disclosure`: not measured — entityType is tier 2; only tier 0 executes today, so this compile cannot measure it — the tier 2 engine arrives in Plan 5.

## Warnings

9 warnings. None of these failed the compile; each is a
judgement a human has to make.

- entityType id "private-key-material" shares token "private" with its own examples; ids ship outbound inside [REDACTED:<id>], so an id derived from a confidential value defeats the redaction — rename it to a generic class name
- entityType id "pred:client-relationship-disclosure" shares token "client" with its own predicate definition; ids ship outbound inside [REDACTED:<id>], so an id derived from a confidential value defeats the redaction — rename it to a generic class name
- provider override "chatgpt" → "pred:client-relationship-disclosure" was inherited from the strictest clause the policy states for "chatgpt" (block): provider clauses are written against named entity classes, and a semantic predicate is not one, so the clause would otherwise not reach it — review that this is what the policy intends
- provider override "gemini" → "pred:client-relationship-disclosure" was inherited from the strictest clause the policy states for "gemini" (block): provider clauses are written against named entity classes, and a semantic predicate is not one, so the clause would otherwise not reach it — review that this is what the policy intends
- provider override "deepseek" → "pred:client-relationship-disclosure" was inherited from the strictest clause the policy states for "deepseek" (block): provider clauses are written against named entity classes, and a semantic predicate is not one, so the clause would otherwise not reach it — review that this is what the policy intends
- self-test: entityType "internal-customer-id" recall 0.50 (10/20 generated positives caught by any entityType) is below threshold 0.8 — these values reach the provider, so no rule in the policy matches what this entity's own definition describes
- self-test: entityType "api-credential" false-positive rate 0.25 (5/20 hard negatives detected) exceeds the maximum 0.1 — its rules fire on text the policy does not cover
- self-test: entityType "db-connection-string" was shadowed on 1/20 generated positives — another entityType won the span under overlap resolution, so the value is still caught but this entity's label, surrogate, and cited clause do not appear
- self-test: entityType "private-key-material" recall 0.75 (15/20 generated positives caught by any entityType) is below threshold 0.8 — these values reach the provider, so no rule in the policy matches what this entity's own definition describes

## Provenance

Every emitted item and the sentence of the policy that justifies it. A rule that cannot
be traced to a clause is a rule the compiler invented.

- `in-pan` — §2.1: "Permanent Account Number (PAN) values must never be sent to an external AI assistant."
- `in-aadhaar` — §2.2: "Aadhaar numbers must never be sent to an external AI assistant."
- `bank-account-identifier` — §2.3: "Bank account numbers, including IFSC-qualified account strings and UPI virtual payment addresses, must never be sent to an external AI assistant."
- `internal-customer-id` — §2.4: "Internal customer identifiers, including customer reference numbers, CIF numbers, and KYC case ids, must never be sent to an external AI assistant."
- `client-name` — §3.1: "Client organisation names and counterparty organisation names must be replaced with a consistent pseudonym before a prompt is sent to an external AI assistant."
- `api-credential` — §4.1: "API keys, access tokens, and client secrets must never be included in a prompt to an external AI assistant, including when the person sending the prompt believes the credential has already been revoked."
- `db-connection-string` — §4.2: "Database connection strings must never be included in a prompt to an external AI assistant, even when the password field has been removed, because the host and schema names disclose the Firm's internal topology."
- `private-key-material` — §4.3: "Private keys, certificate material, and HSM key labels must never be included in a prompt to an external AI assistant."
- `in-pan-format` — §2.1: "Permanent Account Number (PAN) values must never be sent to an external AI assistant."
- `in-aadhaar-format` — §2.2: "Aadhaar numbers must never be sent to an external AI assistant."
- `ifsc-qualified-account` — §2.3: "Bank account numbers, including IFSC-qualified account strings and UPI virtual payment addresses, must never be sent to an external AI assistant."
- `upi-vpa-format` — §2.3: "Bank account numbers, including IFSC-qualified account strings and UPI virtual payment addresses, must never be sent to an external AI assistant."
- `labelled-account-number` — §2.3: "Bank account numbers, including IFSC-qualified account strings and UPI virtual payment addresses, must never be sent to an external AI assistant."
- `internal-customer-id-format` — §2.4: "Internal customer identifiers, including customer reference numbers, CIF numbers, and KYC case ids, must never be sent to an external AI assistant."
- `api-credential-prefix` — §4.1: "API keys, access tokens, and client secrets must never be included in a prompt to an external AI assistant, including when the person sending the prompt believes the credential has already been revoked."
- `api-credential-entropy` — §4.1: "API keys, access tokens, and client secrets must never be included in a prompt to an external AI assistant, including when the person sending the prompt believes the credential has already been revoked."
- `db-connection-string-format` — §4.2: "Database connection strings must never be included in a prompt to an external AI assistant, even when the password field has been removed, because the host and schema names disclose the Firm's internal topology."
- `pem-private-key-header` — §4.3: "Private keys, certificate material, and HSM key labels must never be included in a prompt to an external AI assistant."
- `client-relationship-disclosure` — §3.4: "This section exists because the pairing of a client name with a routine operational question is itself a disclosure of the relationship, and the relationship is frequently the confidential part of the transaction."
- `pred:client-relationship-disclosure` — §3.4: "This section exists because the pairing of a client name with a routine operational question is itself a disclosure of the relationship, and the relationship is frequently the confidential part of the transaction."
