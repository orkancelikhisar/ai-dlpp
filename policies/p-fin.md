# Use of External AI Assistants — Customer Data Handling Standard

**Document:** COMP-2026-11 (P-FIN) · **Owner:** Compliance Desk · **Classification:** Internal
**Effective:** 1 April 2026 · **Supersedes:** COMP-2025-04 · **Review:** annually, or on any change to an assistant vendor agreement

This standard was issued after the quarterly control review found that staff were pasting extracts from the servicing console into consumer chat assistants to draft customer correspondence. It is written to be read once and applied every day. Where a clause is absolute, it says so, and no local practice or team convention relieves it.

## §1 Scope and definitions

§1.1 This standard governs every interaction between a person acting for the Firm and any external artificial-intelligence assistant, including chat interfaces, coding assistants, meeting summarisers, and browser extensions that transmit text to a third party.

§1.2 "Customer data" means any information that identifies, or that can be combined with other information to identify, a person or entity holding an account with the Firm, whether that information originates in a core banking record, a support ticket, a spreadsheet, a screenshot, or a colleague's message.

§1.3 This standard binds every employee, contractor, secondee, and vendor with access to Firm systems, and applies regardless of whether the assistant is reached from a Firm-issued device or a personal one.

§1.4 A prompt is treated as disclosure at the moment it is transmitted, and deleting the conversation afterwards does not undo the disclosure.

## §2 Customer identifiers — forbidden outright

§2.1 Permanent Account Number (PAN) values must never be sent to an external AI assistant.

§2.2 Aadhaar numbers must never be sent to an external AI assistant.

§2.3 Bank account numbers, including IFSC-qualified account strings and UPI virtual payment addresses, must never be sent to an external AI assistant.

§2.4 Internal customer identifiers, including customer reference numbers, CIF numbers, and KYC case ids, must never be sent to an external AI assistant.

§2.5 The prohibition in this section is absolute, and it is not relieved by masking part of the value, by truncating it, by reversing its digits, or by the claim that the record came from a test environment.

## §3 Client and counterparty names — pseudonymize

§3.1 Client organisation names and counterparty organisation names must be replaced with a consistent pseudonym before a prompt is sent to an external AI assistant.

§3.2 The pseudonym substituted for a client name must stay stable for the whole conversation, so that an analyst can still follow which counterparty is which without the real name ever leaving the Firm.

§3.3 The names of prospective clients, deal counterparties, and parties under a non-disclosure agreement are treated exactly as existing client names are treated.

§3.4 This section exists because the pairing of a client name with a routine operational question is itself a disclosure of the relationship, and the relationship is frequently the confidential part of the transaction.

## §4 Credentials and secrets — blocked

§4.1 API keys, access tokens, and client secrets must never be included in a prompt to an external AI assistant, including when the person sending the prompt believes the credential has already been revoked.

§4.2 Database connection strings must never be included in a prompt to an external AI assistant, even when the password field has been removed, because the host and schema names disclose the Firm's internal topology.

§4.3 Private keys, certificate material, and HSM key labels must never be included in a prompt to an external AI assistant.

§4.4 A credential that has been pasted into an external assistant is treated as compromised, and it must be rotated within four hours and reported to the Security Operations Centre.

## §5 Approved and prohibited services

§5.1 No customer data of any classification may be sent to non-enterprise or foreign-hosted services.

§5.2 The Firm holds an enterprise agreement with Anthropic Claude, and Claude is the only external assistant approved for work that involves customer data.

§5.3 The Firm has not contracted with DeepSeek, which is foreign-hosted, and no Firm information of any kind may be sent to DeepSeek.

§5.4 The Firm has not contracted with Google Gemini, and Gemini may be used only for public reference material that contains no customer data.

§5.5 The Firm has not contracted with OpenAI ChatGPT, and ChatGPT may be used only for public reference material that contains no customer data.

§5.6 A personal or trial subscription to an approved vendor is a non-enterprise service for the purposes of §5.1, and the enterprise agreement does not extend to it.

## §6 Enforcement

§6.1 The Firm's data-loss prevention layer applies this standard at the point of transmission, and a blocked prompt is logged with the clause that blocked it and never with the value that triggered it.

§6.2 A breach of §2 or §4 is a reportable security incident and must be raised with the Compliance Desk on the same working day.

§6.3 Exceptions are granted only in writing by the Head of Compliance, are time-bound to ninety days, and are never available for §2.
