# Standard IS-014: Company Information in External AI Assistants

| | |
|---|---|
| **Standard id** | IS-014 |
| **Revision** | 3 (supersedes revision 2, withdrawn) |
| **Owner** | Office of the Chief Information Security Officer |
| **Classification** | Internal |
| **Effective** | 2026-02-01 |
| **Next review** | 2027-02-01 |

**Conformance language.** The words MUST, MUST NOT, and MAY carry the meanings assigned to them in the Company's policy drafting guide. A control stated with MUST NOT admits no local exception unless §7 is followed.

## §1 Scope and applicability

§1.1 This standard defines the controls that apply when Company information is entered into an external artificial-intelligence assistant that the Company does not operate.

§1.2 This standard applies to all workers, including employees, contingent workers, and third parties acting on the Company's behalf, on Company-managed and personal devices alike.

§1.3 This standard governs the content of a prompt rather than the choice of assistant, and the list of assistants approved for business use is maintained separately by the Office of the CISO.

§1.4 Where an information type is not named in this standard, the Company's information classification scheme determines its handling, and information classified Confidential or above MUST NOT be entered into an external AI assistant.

## §2 Credentials and secrets

§2.1 API keys, access tokens, and passwords MUST NOT be included in a prompt to an external AI assistant.

§2.2 Database connection strings, private keys, and code-signing certificates MUST NOT be included in a prompt to an external AI assistant.

§2.3 A credential that appears inside a configuration file, log excerpt, or stack trace is still a credential, and such a file MUST NOT be pasted into an external AI assistant with the credential left in place.

§2.4 A credential exposed to an external AI assistant MUST be rotated within twenty-four hours and reported through the security incident process.

## §3 Unreleased financial and transaction information

§3.1 Financial results that have not yet been publicly released MUST NOT be discussed with an external AI assistant, in figures or in summary.

§3.2 Internal revenue forecasts, sales pipeline projections, and budget models MUST NOT be shared with an external AI assistant, whether expressed as exact figures, as ranges, or as narrative description.

§3.3 Merger, acquisition, divestiture, and fundraising activity that the Company has not announced MUST NOT be described to an external AI assistant, including in hypothetical or anonymised form.

§3.4 The controls in this section apply to the substance of the information and not to any particular wording, and a paraphrase from which a reader could infer an unannounced result or transaction is itself a disclosure.

§3.5 Financial information the Company has already published in a filing or press release is public and MAY be discussed without restriction.

## §4 Compensation information

§4.1 The salary of any individual MUST NOT be entered into an external AI assistant.

§4.2 Bonus amounts, equity grant sizes, and severance terms MUST NOT be entered into an external AI assistant.

§4.3 Internal salary bands and compensation ranges that the Company has not published MUST NOT be shared with an external AI assistant, even when no individual is named.

§4.4 A pay range stated in a job advertisement the Company has published is public information and MAY be discussed freely.

## §5 Internal project codenames

§5.1 Internal project codenames MUST be replaced with a consistent pseudonym before a prompt is sent to an external AI assistant.

§5.2 A codename stays confidential after the associated product ships under a public name, because the mapping between the two discloses how the Company sequences its roadmap.

§5.3 The pseudonym substituted for a codename MUST stay stable for the duration of a conversation so that the assistant's answers remain coherent.

§5.4 Internal repository and system names that follow the Company's codename convention are treated as codenames under this section.

## §6 Information that does not require redaction

§6.1 The name of a Company employee, standing alone and absent other confidential context, does not require redaction and MAY appear in a prompt.

§6.2 An employee's job title, team name, and office location are directory information and do not require redaction.

§6.3 The permission in §6.1 is withdrawn as soon as the name appears together with information restricted by §3, §4, or §5, and the prompt is then handled at the higher restriction.

§6.4 Published marketing material, published product documentation, and open-source code the Company has already released MAY be shared without restriction.

## §7 Compliance and exceptions

§7.1 Where automated controls stop a prompt, the event is logged with the clause that stopped it and never with the content that triggered it.

§7.2 An exception to this standard requires written approval from the Office of the CISO, is time-bound to ninety days, and is not available for §2.

§7.3 Non-conformance is handled under the Company's disciplinary policy, and a suspected disclosure MUST be reported through the security incident process within one business day.

### Revision history

| Rev | Date | Change |
|---|---|---|
| 1 | 2024-09-01 | Initial issue covering credentials only. |
| 2 | 2025-05-15 | Added unreleased financial information; withdrawn after the classification scheme was reissued. |
| 3 | 2026-02-01 | Realigned to the current classification scheme; added compensation and codename controls; added §6 to stop over-redaction of routine internal correspondence. |
