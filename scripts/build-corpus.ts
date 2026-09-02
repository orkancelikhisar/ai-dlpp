/**
 * build-corpus.ts -- regenerates the injection corpus and its manifest.
 *
 *   pnpm -C packages/compiler exec vite-node ../../scripts/build-corpus.ts
 *
 * Run from `packages/compiler` for the same reason `compile-policies.ts` and
 * `fetch-models.ts` are: `vite-node` is a devDependency of that package and of
 * `packages/tier1`, and adding a fourth copy to `apps/eval` to save a `-C` flag
 * would be a lockfile change in exchange for nothing.
 *
 * The generator is seeded and versioned, so a run either reproduces the
 * committed bytes exactly or the seed, the generator version, the carrier pool,
 * the family catalogue or the compiled IR changed. `corpus-artifact.test.ts`
 * asserts that reproduction on every suite run, which is what makes running
 * this script optional rather than a step nobody remembers.
 */
import { writeArtifacts, CORPUS_PATH, MANIFEST_PATH } from "../apps/eval/src/corpus/build.js";

const built = writeArtifacts();
const m = built.manifest;
console.log(`wrote ${CORPUS_PATH}`);
console.log(`wrote ${MANIFEST_PATH}`);
console.log(
  `${m.counts.items} items (${m.counts.positives} positive / ${m.counts.negatives} negative), ` +
    `${m.counts.goldSpans} gold spans, ${m.counts.confusableSpans} confusable spans`,
);
console.log(`certification: ${m.certification.claim} -- stages run: ${m.certification.stagesRun.join(", ")}`);
console.log(`contamination: ${m.contamination.dropped.length} dropped of ${m.contamination.itemsChecked} checked`);
console.log(`sha256 ${m.artifact.corpusSha256}`);
