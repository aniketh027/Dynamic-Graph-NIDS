import { projectFacts as facts } from './project-facts'

// A compact presentation snapshot, not an evaluator. Focused tests bind these
// values to the compact reports at the frozen scientific release below.
function freeze<T>(value: T): T {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child)
    Object.freeze(value)
  }
  return value
}

const study = 'e10-official-20261001-v1'
const comparisonRoot = `results/stage_e12b/${study}/e12b-20261006-v1`
const policyRoot = `results/stage_e12/${study}/e12a-20261006-v1`

export const evidenceSources = freeze({
  finalResults: { label: 'Final scientific interpretation', path: 'docs/final_experimental_results.md' },
  baseline: { label: 'E.11 · recovered unpruned baseline', path: `results/stage_e11/${study}/recovery/e11-recovery-20261006-v1/stage_e11_report.json` },
  selection: { label: 'E.12A · validation-only policy selection', path: `${policyRoot}/stage_e12a_report.json` },
  policy: { label: 'E.12A · fixed attention policy', path: `${policyRoot}/proposed_policy.json` },
  freeze: { label: 'E.12A · separate policy freeze attestation', path: `${policyRoot}-freeze/freeze_attestation.json` },
  comparison: { label: 'E.12B · frozen test comparison', path: `${comparisonRoot}/stage_e12b_report.json` },
  benchmarks: { label: 'E.12B · GPU validation benchmarks', path: `${comparisonRoot}/validation_benchmarks.json` },
  provenance: { label: 'Scientific release provenance', path: 'docs/release_provenance.md' },
} as const)

export const modelPerformanceEvidence = freeze({
  release: { tag: 'v1.0-experimental-complete', commit: '2854d153d65c93689875884f3ad45f0b3b468ec2' },
  e2: {
    seeds: [2025, 2026, 2027],
    eligibleClassCodes: [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14],
    unpruned: { mean: 0.8759879709008183, sampleSD: 0.02264023047925946 },
    pruned: { mean: 0.8896826983866664, sampleSD: 0.009045955441241697 },
    perSeed: [
      { seed: 2025, unpruned: 0.8513260520437465, pruned: 0.8972391626354593, change: 0.04591311059171288 },
      { seed: 2026, unpruned: 0.8808074487931062, pruned: 0.8796591323330499, change: -0.001148316460056309 },
      { seed: 2027, unpruned: 0.8958304118656021, pruned: 0.8921498001914897, change: -0.0036806116741123773 },
    ],
  },
  multiclassParameters: 7183,
  benchmarks: {
    gpu: 'NVIDIA GeForce RTX 4050 Laptop GPU',
    // Range of nine paired medians: E1/E2/E3, each at three seeds.
    pairedSpeedRatio: { min: 1.0345396125839361, max: 1.1172861113456347 },
    // E1/E2 allocator peaks; E3 has different parameter and memory counts.
    peakAllocatedBytes: { unpruned: 51959296, pruned: 48886272 },
    peakReservedBytes: { unpruned: 58720256, pruned: 58720256 },
    batchSize: 256,
    batchStarts: [0, 256, 512, 768],
    warmupsPerPathPerBatch: 5,
    measuredPairsPerBatch: 20,
  },
  diagnostic: {
    e1: { unpruned: 0.9307991287327075, pruned: 0.9307918723561247 },
    e3: { unpruned: 0.08903136152054146, pruned: 0.08904805808166393 },
  },
} as const)

// Reuse the existing graph presentation authority; no second topology or
// feature mapping is introduced. The sample count follows that same identity.
export const graphComparison = freeze([
  { label: 'Feature nodes', unpruned: facts.inputFeatures, pruned: facts.retainedFeatures },
  { label: 'Category nodes', unpruned: facts.categoryNodes, pruned: facts.categoryNodes },
  { label: 'Sample nodes', unpruned: facts.unprunedNodes - facts.inputFeatures - facts.categoryNodes, pruned: facts.prunedNodes - facts.retainedFeatures - facts.categoryNodes },
  { label: 'Total nodes', unpruned: facts.unprunedNodes, pruned: facts.prunedNodes },
  { label: 'Directed edges', unpruned: facts.unprunedEdges, pruned: facts.prunedEdges },
  { label: 'E2 HTGNN parameters', unpruned: modelPerformanceEvidence.multiclassParameters, pruned: modelPerformanceEvidence.multiclassParameters },
] as const)
