import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadPolicyIr, type PolicyIr } from "@sih/core";
import {
  NEG_PREFIX,
  POLICY_IDS,
  actionUnder,
  buildViolatesUnder,
  isIrBacked,
  pFinLabel,
  toPFinGold,
  unpopulatedLabel,
  type LabelledSpan,
} from "../src/corpus/labels.js";

const REPO = fileURLToPath(new URL("../../..", import.meta.url));
const IR_SOURCE = "policies/compiled/p-fin.ir.json";
const IR: PolicyIr = loadPolicyIr(readFileSync(`${REPO}${IR_SOURCE}`, "utf8"));

const span = (text: string) => ({ start: 0, end: text.length, text });

describe("the three-policy label", () => {
  it("carries all three policies", () => {
    expect(Object.keys(buildViolatesUnder(IR, "in-pan", IR_SOURCE)).sort()).toEqual([...POLICY_IDS].sort());
  });

  it("reads p-fin's action out of the IR rather than restating it", () => {
    // Two entityTypes with DIFFERENT actions in the same table, so a hardcoded
    // "block" fails on the second.
    expect(pFinLabel(IR, "in-pan", IR_SOURCE)).toEqual({
      state: "populated",
      action: "block",
      source: `${IR_SOURCE}#actions.default.in-pan`,
    });
    expect(pFinLabel(IR, "client-name", IR_SOURCE)).toMatchObject({ state: "populated", action: "pseudonymize" });
  });

  it("follows a DIFFERENT ir, so it cannot be reading a memorised table", () => {
    // The standing conventions' rule: a test that exercises only the shipped
    // configuration cannot tell "reads the config" from "hardcodes it".
    const rewritten: PolicyIr = {
      ...IR,
      actions: { ...IR.actions, default: { ...IR.actions.default, "in-pan": "redact" } },
    };
    expect(pFinLabel(rewritten, "in-pan", "other.json")).toEqual({
      state: "populated",
      action: "redact",
      source: "other.json#actions.default.in-pan",
    });
  });

  it("leaves p-med and p-corp unpopulated, with a blocker named", () => {
    const v = buildViolatesUnder(IR, "in-pan", IR_SOURCE);
    for (const policy of ["p-med", "p-corp"] as const) {
      const label = v[policy];
      expect(label.state).toBe("unpopulated");
      if (label.state !== "unpopulated") throw new Error("unreachable");
      expect(label.blockedOn).toContain("2026-08-18");
      expect(label.why).toContain(`policies/${policy}.md`);
      // The whole point: there is no action to read, not an action that says
      // "unknown". Nothing on this object is assignable to an Action.
      expect(Object.hasOwn(label, "action")).toBe(false);
    }
  });

  it("throws rather than defaulting when a scorer asks for an unpopulated policy", () => {
    const label: LabelledSpan = {
      span: span("AAAPZ1234C"),
      type: "in-pan",
      violatesUnder: buildViolatesUnder(IR, "in-pan", IR_SOURCE),
    };
    expect(actionUnder(label, "p-fin")).toBe("block");
    expect(() => actionUnder(label, "p-med")).toThrow(/no ground truth for policy "p-med"/);
    expect(() => actionUnder(label, "p-corp")).toThrow(/no ground truth for policy "p-corp"/);
  });

  it("refuses a type that is neither an IR entityType nor prefixed neg:", () => {
    expect(() => pFinLabel(IR, "in-pans", IR_SOURCE)).toThrow(/has no action in/);
  });

  it("labels a neg: confusable none under p-fin, and says where that came from", () => {
    const label = pFinLabel(IR, `${NEG_PREFIX}swift-bic`, IR_SOURCE);
    expect(label).toMatchObject({ state: "populated", action: "none" });
    if (label.state !== "populated") throw new Error("unreachable");
    expect(label.source).toContain("is not an entityType of this IR");
  });

  it("unpopulatedLabel names the policy it is about", () => {
    const med = unpopulatedLabel("p-med");
    const corp = unpopulatedLabel("p-corp");
    if (med.state !== "unpopulated" || corp.state !== "unpopulated") throw new Error("unreachable");
    expect(med.why).not.toBe(corp.why);
  });
});

describe("toPFinGold", () => {
  const labels: LabelledSpan[] = [
    { span: span("AAAPZ1234C"), type: "in-pan", violatesUnder: buildViolatesUnder(IR, "in-pan", IR_SOURCE) },
    {
      span: { start: 20, end: 28, text: "KESTINBB" },
      type: `${NEG_PREFIX}swift-bic`,
      violatesUnder: buildViolatesUnder(IR, `${NEG_PREFIX}swift-bic`, IR_SOURCE),
    },
  ];

  it("keeps IR-backed labels and drops confusables", () => {
    expect(toPFinGold(labels)).toEqual([
      { start: 0, end: 10, text: "AAAPZ1234C", entityType: "in-pan", action: "block" },
    ]);
  });

  it("never emits a neg: id into the entityType namespace a scorer joins on", () => {
    for (const g of toPFinGold(labels)) expect(isIrBacked(g.entityType)).toBe(true);
  });
});
