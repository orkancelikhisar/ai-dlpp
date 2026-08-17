import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { groundProviders, loadManifest, resolveProviderMention } from "../../src/stages/ground.js";

const MANIFEST = loadManifest({
  manifestVersion: "1",
  providers: [
    { id: "claude", displayName: "Anthropic Claude", vendor: "Anthropic", hostingRegion: "us", enterpriseAgreement: true, aliases: ["claude", "anthropic"] },
    { id: "deepseek", displayName: "DeepSeek", vendor: "DeepSeek", hostingRegion: "cn", enterpriseAgreement: false, aliases: ["deepseek", "foreign-hosted", "chinese-hosted"] },
  ],
});

describe("groundProviders", () => {
  const actions = [
    { entityType: "client-name", action: "pseudonymize" as const, sourceQuote: "q" },
    { entityType: "client-name", action: "redact" as const, providerMention: "Chinese-hosted services", sourceQuote: "q" },
    { entityType: "client-name", action: "allow" as const, providerMention: "our enterprise Claude agreement", sourceQuote: "q" },
  ];

  it("routes an unqualified action to the default map", () => {
    const { defaults } = groundProviders(MANIFEST, actions);
    expect(defaults["client-name"]).toBe("pseudonymize");
  });

  it("resolves an alias mention to its adapter id", () => {
    const { providerOverrides } = groundProviders(MANIFEST, actions);
    expect(providerOverrides["deepseek"]!["client-name"]).toBe("redact");
  });

  it("resolves a mention containing a provider name among other words", () => {
    const { providerOverrides } = groundProviders(MANIFEST, actions);
    expect(providerOverrides["claude"]!["client-name"]).toBe("allow");
  });

  it("warns rather than dropping an unresolvable mention", () => {
    const { warnings, providerOverrides } = groundProviders(MANIFEST, [
      { entityType: "client-name", action: "block" as const, providerMention: "Foocorp AI", sourceQuote: "q" },
    ]);
    expect(warnings[0]).toMatch(/Foocorp AI/);
    expect(Object.keys(providerOverrides)).toHaveLength(0);
  });

  it("prefers the longest matching alias when several match", () => {
    // "chinese-hosted" and "deepseek" both belong to deepseek; a mention naming
    // both must resolve once, not twice.
    const { providerOverrides } = groundProviders(MANIFEST, [
      { entityType: "x", action: "block" as const, providerMention: "DeepSeek and other Chinese-hosted services", sourceQuote: "q" },
    ]);
    expect(Object.keys(providerOverrides)).toEqual(["deepseek"]);
  });

  it("rejects a manifest with duplicate ids or aliases", () => {
    expect(() =>
      loadManifest({
        manifestVersion: "1",
        providers: [
          { id: "a", displayName: "A", vendor: "A", hostingRegion: "us", enterpriseAgreement: false, aliases: ["x"] },
          { id: "b", displayName: "B", vendor: "B", hostingRegion: "us", enterpriseAgreement: false, aliases: ["x"] },
        ],
      }),
    ).toThrow(/alias "x"/i);
  });
});

// ---------------------------------------------------------------------------
// Additions beyond the plan's six.
// ---------------------------------------------------------------------------

describe("resolveProviderMention", () => {
  it("reports the longest alias as the reason a provider matched", () => {
    // Not just dedupe: the alias that won is what a human reads in the report
    // when auditing why a clause landed on this adapter. Manifest-order matching
    // would answer "deepseek" here.
    const { matches } = resolveProviderMention(MANIFEST, "DeepSeek and other Chinese-hosted services");
    expect(matches).toHaveLength(1);
    expect(matches[0]!.providerId).toBe("deepseek");
    expect(matches[0]!.alias).toBe("chinese-hosted");
  });

  it("matches case-insensitively and ignores surrounding words", () => {
    const { matches } = resolveProviderMention(MANIFEST, "anything routed through ANTHROPIC today");
    expect(matches.map((m) => m.providerId)).toEqual(["claude"]);
    expect(matches[0]!.via).toBe("provider");
  });
});

describe("loadManifest", () => {
  it("rejects duplicate provider ids", () => {
    expect(() =>
      loadManifest({
        manifestVersion: "1",
        providers: [
          { id: "a", displayName: "A", vendor: "A", hostingRegion: "us", enterpriseAgreement: false, aliases: ["x"] },
          { id: "a", displayName: "A2", vendor: "A", hostingRegion: "us", enterpriseAgreement: false, aliases: ["y"] },
        ],
      }),
    ).toThrow(/duplicate provider id "a"/i);
  });

  it("rejects an alias claimed by both a provider and a category", () => {
    // Categories and providers share one alias namespace: if "chinese-hosted"
    // could mean either, resolution silently picks whichever the loader indexed.
    expect(() =>
      loadManifest({
        manifestVersion: "1",
        providers: [
          { id: "a", displayName: "A", vendor: "A", hostingRegion: "cn", enterpriseAgreement: false, aliases: ["chinese-hosted"] },
        ],
        categories: [
          { id: "cn", description: "hosted in china", aliases: ["chinese-hosted"], match: { hostingRegionIn: ["cn"] } },
        ],
      }),
    ).toThrow(/alias "chinese-hosted"/i);
  });

  it("rejects a category whose match constrains nothing", () => {
    // An empty match selects every provider, which would turn one stray word in
    // a mention into a firm-wide block.
    expect(() =>
      loadManifest({
        manifestVersion: "1",
        providers: [
          { id: "a", displayName: "A", vendor: "A", hostingRegion: "us", enterpriseAgreement: false, aliases: ["a"] },
        ],
        categories: [{ id: "everything", description: "oops", aliases: ["services"], match: {} }],
      }),
    ).toThrow(/constrain at least one attribute/i);
  });
});

describe("groundProviders conflict and key handling", () => {
  it("keeps the more restrictive action when two mentions collide, whatever the input order", () => {
    const permissive = { entityType: "client-name", action: "allow" as const, providerMention: "Anthropic", sourceQuote: "q" };
    const strict = { entityType: "client-name", action: "block" as const, providerMention: "claude", sourceQuote: "q" };

    const forward = groundProviders(MANIFEST, [permissive, strict]);
    const reverse = groundProviders(MANIFEST, [strict, permissive]);

    expect(forward.providerOverrides["claude"]!["client-name"]).toBe("block");
    expect(reverse.providerOverrides["claude"]!["client-name"]).toBe("block");
    expect(forward.warnings.join("\n")).toMatch(/conflicting actions/i);
  });

  it("warns rather than globalising an action whose mention is blank", () => {
    // A blank mention swallowed into `defaults` would promote a provider-scoped
    // "allow" into a firm-wide one.
    const { defaults, warnings } = groundProviders(MANIFEST, [
      { entityType: "client-name", action: "allow" as const, providerMention: "   ", sourceQuote: "q" },
    ]);
    expect(Object.keys(defaults)).toHaveLength(0);
    expect(warnings[0]).toMatch(/blank|empty/i);
  });

  it("treats prototype-named ids as ordinary keys", () => {
    const { defaults, providerOverrides } = groundProviders(MANIFEST, [
      { entityType: "toString", action: "block" as const, sourceQuote: "q" },
      { entityType: "constructor", action: "redact" as const, providerMention: "deepseek", sourceQuote: "q" },
    ]);
    expect(defaults["toString"]).toBe("block");
    expect(Object.keys(defaults)).toEqual(["toString"]);
    expect(providerOverrides["deepseek"]!["constructor"]).toBe("redact");
    // A `{}` accumulator would answer with Function.prototype.toString here.
    expect(providerOverrides["toString"]).toBeUndefined();
    expect(defaults["constructor"]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// The real manifest against the real policy. The inline MANIFEST above cannot
// catch the failure this section exists for: P-FIN §5.1 is a blanket clause
// whose wording contains a term that used to be an *alias of one provider*, so
// naive substring matching narrowed the firm's broadest prohibition to DeepSeek
// alone and left ChatGPT and Gemini on the permissive default.
// ---------------------------------------------------------------------------

const ROOT = join(import.meta.dirname, "..", "..", "..", "..", "policies");
const REAL = loadManifest(JSON.parse(readFileSync(join(ROOT, "providers.json"), "utf8")));
const P_FIN = readFileSync(join(ROOT, "p-fin.md"), "utf8");

const CLAUSE_5_1 =
  "No customer data of any classification may be sent to non-enterprise or foreign-hosted services.";

describe("P-FIN §5 against the real provider manifest", () => {
  it("still has the §5.1 wording this suite is written against", () => {
    expect(P_FIN).toContain(CLAUSE_5_1);
  });

  it.each([CLAUSE_5_1, "non-enterprise or foreign-hosted services", "non-enterprise services"])(
    "grounds the blanket clause onto every provider it covers: %s",
    (mention) => {
      const { providerOverrides, warnings } = groundProviders(REAL, [
        { entityType: "customer-id", action: "block" as const, providerMention: mention, sourceQuote: CLAUSE_5_1 },
      ]);
      // Naive per-provider alias matching answers ["deepseek"] and silently
      // leaves chatgpt and gemini permissive; dropping the category terms with
      // no replacement answers [] plus a warning.
      expect(Object.keys(providerOverrides).sort()).toEqual(["chatgpt", "deepseek", "gemini"]);
      // §5.2 approves Claude for customer-data work, so the blanket clause must
      // NOT reach it — a jurisdiction-relative reading of "foreign-hosted"
      // (every provider is foreign to an Indian firm) would block it.
      expect(providerOverrides["claude"]).toBeUndefined();
      expect(warnings).toEqual([]);
    },
  );

  it.each([
    ["our enterprise agreement with Anthropic Claude", "claude"],
    ["DeepSeek", "deepseek"],
    ["Google Gemini", "gemini"],
    ["OpenAI ChatGPT", "chatgpt"],
  ])("resolves the named-vendor clause %s to %s", (mention, id) => {
    const { providerOverrides, warnings } = groundProviders(REAL, [
      { entityType: "customer-id", action: "block" as const, providerMention: mention, sourceQuote: CLAUSE_5_1 },
    ]);
    expect(Object.keys(providerOverrides)).toEqual([id]);
    expect(warnings).toEqual([]);
  });

  it("warns on a bare 'foreign-hosted' rather than guessing a jurisdiction", () => {
    // Deliberately undefined in the manifest: "foreign" is decidable only
    // against a home jurisdiction the manifest does not record. The old alias
    // made it a silent synonym for DeepSeek.
    const { providerOverrides, warnings } = groundProviders(REAL, [
      { entityType: "customer-id", action: "block" as const, providerMention: "foreign-hosted services", sourceQuote: CLAUSE_5_1 },
    ]);
    expect(Object.keys(providerOverrides)).toHaveLength(0);
    expect(warnings[0]).toMatch(/foreign-hosted services/);
  });

  it("resolves 'Chinese-hosted' from the hostingRegion attribute, not from one vendor's alias list", () => {
    const { matches } = resolveProviderMention(REAL, "Chinese-hosted services");
    expect(matches.map((m) => m.providerId)).toEqual(["deepseek"]);
    expect(matches[0]!.via).toBe("category");
  });

  it("carries no category term in any provider's alias list", () => {
    // The structural version of the bug: a category masquerading as a provider
    // name resolves a firm-wide clause to whichever vendor happens to own it.
    const categoryAliases = new Set(REAL.categories.flatMap((c) => c.aliases));
    for (const provider of REAL.providers) {
      for (const alias of provider.aliases) {
        expect(categoryAliases.has(alias), `${provider.id} alias ${alias}`).toBe(false);
      }
    }
  });
});
