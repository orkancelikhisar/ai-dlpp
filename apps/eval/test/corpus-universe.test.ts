import { describe, expect, it } from "vitest";
import { getValidator, seededRng } from "@sih/core";
import {
  CLIENT_ORGS,
  NON_CLIENT_ORGS,
  mintAadhaar,
  mintBadAadhaar,
  mintBic,
  mintCustomerId,
  mintGitSha,
  mintIfsc,
  mintInvalidPan,
  mintPan,
  mintPemBlock,
  mintTicketId,
  mintUpiVpa,
  spaceAadhaar,
} from "../src/corpus/universe.js";

/**
 * Property tests over many seeds. Each mint's contract is checked against the
 * REAL validator or the REAL rule regex from `policies/compiled/p-fin.ir.json`,
 * never against a restatement of the generator's own pattern -- a test that
 * built its expectation from the generator would pass for any generator.
 */
const SEEDS = Array.from({ length: 200 }, (_, i) => seededRng(`universe-property-${i}`));

const panStructure = getValidator("pan-structure");
const verhoeff = getValidator("verhoeff");

/** Rule sources copied out of the compiled IR, which is what tier 0 runs. */
const PAN_RE = /^[A-Z]{5}[0-9]{4}[A-Z]$/;
const AADHAAR_RE = /^[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}$/;
const IFSC_RE = /^[A-Z]{4}0[A-Z0-9]{6}$/;
const UPI_RE = /^[a-zA-Z0-9][a-zA-Z0-9._-]{2,}@[a-zA-Z][a-zA-Z0-9]{2,}$/;
const CUSTOMER_ID_RE = /^(?:CIF|CRN|KYC)[ /:#-]?[0-9]{6,10}$/;

describe("minted values satisfy the contract their family claims", () => {
  it("mintPan is accepted by the shipping pan-structure validator", () => {
    for (const rng of SEEDS) {
      const pan = mintPan(rng);
      expect([pan, PAN_RE.test(pan), panStructure(pan)]).toEqual([pan, true, true]);
    }
  });

  it("mintInvalidPan matches the regex and is REJECTED by the validator", () => {
    // Both halves matter: a value that failed the regex too would be a
    // confusable nothing ever confuses.
    for (const rng of SEEDS) {
      const pan = mintInvalidPan(rng);
      expect([pan, PAN_RE.test(pan), panStructure(pan)]).toEqual([pan, true, false]);
    }
  });

  it("mintAadhaar passes Verhoeff and mintBadAadhaar fails it", () => {
    for (const rng of SEEDS) {
      const good = mintAadhaar(rng);
      expect([good, AADHAAR_RE.test(good), verhoeff(good)]).toEqual([good, true, true]);
      expect([good, AADHAAR_RE.test(spaceAadhaar(good))]).toEqual([good, true]);
      const bad = mintBadAadhaar(rng);
      expect([bad, AADHAAR_RE.test(bad), verhoeff(bad)]).toEqual([bad, true, false]);
    }
  });

  it("mintIfsc matches the IFSC rule and mintBic does not", () => {
    for (const rng of SEEDS) {
      expect(IFSC_RE.test(mintIfsc(rng))).toBe(true);
      const bic = mintBic(rng);
      expect([bic, IFSC_RE.test(bic), /^[A-Z]{8}$/.test(bic)]).toEqual([bic, false, true]);
    }
  });

  it("mintUpiVpa matches the UPI rule", () => {
    for (const rng of SEEDS) expect(UPI_RE.test(mintUpiVpa(rng, "okaxis"))).toBe(true);
  });

  it("mintCustomerId matches the internal-customer-id rule and mintTicketId does not", () => {
    for (const rng of SEEDS) {
      expect(CUSTOMER_ID_RE.test(mintCustomerId(rng, "CIF", " "))).toBe(true);
      expect(CUSTOMER_ID_RE.test(mintCustomerId(rng, "CRN", "-"))).toBe(true);
      const ticket = mintTicketId(rng);
      expect([ticket, CUSTOMER_ID_RE.test(ticket)]).toEqual([ticket, false]);
    }
  });

  it("mintGitSha is 40 lowercase hex", () => {
    for (const rng of SEEDS) expect(/^[0-9a-f]{40}$/.test(mintGitSha(rng))).toBe(true);
  });

  it("mintPemBlock is a complete block, not a bare header", () => {
    for (const rng of SEEDS.slice(0, 20)) {
      const pem = mintPemBlock(rng);
      expect(pem.startsWith("-----BEGIN RSA PRIVATE KEY-----\n")).toBe(true);
      expect(pem.endsWith("\n-----END RSA PRIVATE KEY-----")).toBe(true);
    }
  });
});

describe("the universe is internally distinct", () => {
  it("no client organisation is also a non-client organisation", () => {
    const overlap = CLIENT_ORGS.filter((c) => (NON_CLIENT_ORGS as readonly string[]).includes(c));
    expect(overlap).toEqual([]);
  });

  it("no organisation name is a substring of another", () => {
    // A substring pair would make the "occurs exactly as many times as injected"
    // invariant ambiguous the first time both landed in one message.
    const all = [...CLIENT_ORGS, ...NON_CLIENT_ORGS];
    for (const a of all) for (const b of all) if (a !== b) expect([a, b, b.includes(a)]).toEqual([a, b, false]);
  });
});

describe("determinism of the mints", () => {
  it("the same key produces the same value", () => {
    expect(mintPan(seededRng("k"))).toBe(mintPan(seededRng("k")));
    expect(mintAadhaar(seededRng("k"))).toBe(mintAadhaar(seededRng("k")));
  });

  it("different keys produce different values", () => {
    const pans = new Set(Array.from({ length: 50 }, (_, i) => mintPan(seededRng(`k${i}`))));
    expect(pans.size).toBeGreaterThan(40);
  });
});
