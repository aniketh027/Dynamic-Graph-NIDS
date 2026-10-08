import './register-tsx.mjs'
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync, existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

const { MemoryRouter } = await import('react-router')
const { LazyMotion, domAnimation } = await import('motion/react')
const { ModelPerformance } = await import('../src/pages/workspace-pages.tsx')
const { evidenceSources: sources, graphComparison, modelPerformanceEvidence: evidence } = await import('../src/lib/model-performance-evidence.ts')
const { projectFacts: facts } = await import('../src/lib/project-facts.ts')
const { ReplayContext } = await import('../src/lib/replay-context.ts')
const { DemoContext } = await import('../src/lib/demo-context.ts')
const { BackendContext } = await import('../src/lib/api/context.ts')

const root = new URL('../../', import.meta.url)
const read = path => JSON.parse(readFileSync(new URL(path, root), 'utf8'))
const baseline = read(sources.baseline.path)
const comparison = read(sources.comparison.path)
const benchmarks = read(sources.benchmarks.path).measurements
const summary = task => comparison.comparison.seed_summary.find(row => row.task === task)
const noop = () => {}
function render(replay = { state: { phase: 'idle', result: null } }, demo = { enabled: false }, backend = { state: 'offline', status: null }) {
  return renderToStaticMarkup(React.createElement(MemoryRouter, null,
    React.createElement(LazyMotion, { features: domAnimation },
      React.createElement(BackendContext.Provider, { value: backend },
        React.createElement(DemoContext.Provider, { value: demo },
          React.createElement(ReplayContext.Provider, { value: replay }, React.createElement(ModelPerformance)))))))
}
const html = render()

test('E2 means, sample SD and seed comparisons match both frozen compact reports', () => {
  const e2 = summary('E2')
  const original = baseline.seed_summary.find(row => row.task === 'E2' && row.model_family === 'htgnn')
  for (const path of ['unpruned', 'pruned']) {
    assert.equal(evidence.e2[path].mean, e2.metrics['primary.macro_f1'][path].mean)
    assert.equal(evidence.e2[path].sampleSD, e2.metrics['primary.macro_f1'][path].sample_std_ddof1)
    const values = evidence.e2.perSeed.map(row => row[path])
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length
    const sampleSD = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1))
    assert.ok(Math.abs(mean - evidence.e2[path].mean) < 1e-15)
    assert.ok(Math.abs(sampleSD - evidence.e2[path].sampleSD) < 1e-15)
    assert.match(html, new RegExp(evidence.e2[path].mean.toFixed(6)))
    assert.ok(html.includes(evidence.e2[path].sampleSD.toFixed(6)))
  }
  assert.equal(evidence.e2.unpruned.mean, original.primary_mean)
  assert.equal(evidence.e2.unpruned.sampleSD, original.primary_sample_std_ddof1)
  assert.deepEqual(evidence.e2.seeds, e2.seeds)
  for (const row of evidence.e2.perSeed) {
    const actual = comparison.comparison.paired.find(pair => pair.task === 'E2' && pair.seed === row.seed).changes['primary.macro_f1']
    assert.equal(row.unpruned, actual.unpruned)
    assert.equal(row.pruned, actual.pruned)
    assert.equal(row.change, actual.absolute_change)
  }
  assert.match(html, /sample SD \(ddof=1\)/)
  assert.match(html, /2025, 2026, 2027/)
  assert.match(html, /One seed improved; two declined slightly/)
})

test('E2 eligible true-row test definition is explicit and separate from all-test descriptive metrics', () => {
  const records = baseline.results.filter(row => row.task === 'E2' && row.model_family === 'htgnn')
  for (const row of records) {
    assert.equal(row.metrics.primary_name, 'eligible_class_macro_f1')
    assert.equal(row.metrics.primary_true_row_filter, 'eligible_classes')
    assert.deepEqual(row.metrics.primary.class_codes, evidence.e2.eligibleClassCodes)
    assert.notEqual(row.metrics.primary.macro_f1, row.metrics.all_test_descriptive.macro_f1)
  }
  for (const label of ['CSE-CIC-IDS2018', 'E2 primary supervised multiclass', 'Frozen test evaluation', 'train ≥100, validation ≥30 and test ≥30', 'True test rows are restricted to eligible classes', 'Scale 0–1']) assert.ok(html.includes(label), label)
  assert.doesNotMatch(html, /0\.731777|0\.733112/)
})

test('structural counts reuse project facts and match the fixed frozen policy and report', () => {
  const policy = read(sources.policy.path)
  const attestation = read(sources.freeze.path)
  assert.equal(attestation.review_status, 'PASS')
  assert.equal(attestation.owner_approved_freeze, true)
  assert.equal(attestation.policy_canonical_sha256, comparison.policy_sha256)
  assert.equal(policy.policy_sha256, comparison.policy_sha256)
  assert.equal(read(sources.selection.path).policy_sha256, comparison.policy_sha256)
  const keys = ['feature_nodes', 'category_nodes', 'sample_nodes', 'nodes', 'edges']
  keys.forEach((key, i) => {
    assert.equal(graphComparison[i].unpruned, comparison.graph.original_graph[key])
    assert.equal(graphComparison[i].pruned, comparison.graph.pruned_graph[key])
    assert.match(html, new RegExp(`<th scope="row">${graphComparison[i].label}</th><td>${graphComparison[i].unpruned}</td><td>${graphComparison[i].pruned}</td>`))
  })
  assert.equal(facts.retainedFeatures, attestation.retained_feature_count)
  assert.equal(facts.prunedNodes, attestation.total_nodes)
  assert.equal(facts.prunedEdges, attestation.total_edges)
  assert.deepEqual(facts.removedFeatureIds, attestation.removed_feature_ids)
  assert.match(html, /full canonical 78-feature CSV/)
  assert.match(html, /input adapter builds the full graph/)
})

test('E2 parameter counts and trained weights remain unchanged', () => {
  const records = baseline.results.filter(row => row.task === 'E2' && row.model_family === 'htgnn')
  for (const row of records) assert.equal(evidence.multiclassParameters, row.checkpoint_provenance.parameter_count)
  assert.equal(comparison.model_states_unchanged, true)
  assert.equal(comparison.training, false)
  assert.equal(comparison.optimizer, false)
  assert.equal(comparison.unpruned_test_evaluation_repeated, false)
  assert.equal(comparison.comparison.test_based_selection, false)
  assert.match(html, /E2 HTGNN parameters<\/th><td>7,183<\/td><td>7,183/)
  assert.match(html, /no model weight pruning or retraining/)
})

test('GPU validation speed range and protocol match all nine persisted measurements', () => {
  const ratios = benchmarks.map(row => row.paired_speed_ratio_median)
  assert.equal(benchmarks.length, 9)
  assert.equal(evidence.benchmarks.pairedSpeedRatio.min, Math.min(...ratios))
  assert.equal(evidence.benchmarks.pairedSpeedRatio.max, Math.max(...ratios))
  for (const row of benchmarks) {
    assert.equal(row.plan.split, 'validation')
    assert.equal(row.plan.batch_size, evidence.benchmarks.batchSize)
    assert.deepEqual(row.plan.sequential_batch_starts, evidence.benchmarks.batchStarts)
    assert.equal(row.observed_warmups, evidence.benchmarks.warmupsPerPathPerBatch)
    assert.equal(row.observed_pairs_per_batch, evidence.benchmarks.measuredPairsPerBatch)
    assert.equal(row.plan.deterministic_algorithms, true)
  }
  for (const label of ['1.034540×–1.117286×', 'RTX 4050 Laptop GPU', 'Validation only', 'Range across E1 / E2 / E3', '80 pairs', 'host-to-device transfer are excluded', 'do not measure current CPU CSV upload/inference latency']) assert.ok(html.includes(label), label)
})

test('allocator measurements retain E1/E2 scope and do not confuse allocation with reservation', () => {
  for (const row of benchmarks.filter(row => row.task !== 'E3')) {
    for (const path of ['unpruned', 'pruned']) {
      assert.equal(row.gpu_memory[path].reproducible, true)
      assert.equal(evidence.benchmarks.peakAllocatedBytes[path], row.gpu_memory[path].bytes.peak_allocated)
    }
  }
  for (const row of benchmarks) for (const path of ['unpruned', 'pruned']) assert.equal(evidence.benchmarks.peakReservedBytes[path], row.gpu_memory[path].bytes.peak_reserved)
  for (const label of ['49.552', '46.622', '2.931 MiB / 5.91%', 'Peak PyTorch allocated · E1 / E2', 'Peak PyTorch reserved · All tasks', 'Reservation unchanged', 'resident full adapter input']) assert.ok(html.includes(label), label)
  assert.match(html, /56 <span>→<\/span> 56/)
})

test('E1 and E3 limitations remain visible with their distinct metrics and scopes', () => {
  for (const [task, key] of [['E1', 'primary.macro_f1'], ['E3', 'pr_auc_average_precision']]) {
    for (const path of ['unpruned', 'pruned']) assert.equal(evidence.diagnostic[task.toLowerCase()][path], summary(task).metrics[key][path].mean)
  }
  const section = html.slice(html.indexOf('id="scientific-limitations"'), html.indexOf('id="evidence-provenance"'))
  for (const label of ['only Benign', '0.930799 → 0.930792', 'Poor chronological generalization', 'average-precision PR-AUC 0.089031 → 0.089048', 'approximately 0.089', 'CPU inference and offline file replay', 'not evidence of statistical significance', 'formal noninferiority guarantee', 'validation GPU measurements']) assert.ok(section.includes(label), label)
  assert.doesNotMatch(section, /<details|role="tooltip"|<(?:section|div|p|h[23])[^>]*\shidden(?:=|\s|>)/)
})

test('uploads, demo, mock presentation and backend status cannot change frozen evidence markup', () => {
  const wire = JSON.parse(readFileSync(new URL('./synthetic-wire-response.json', import.meta.url), 'utf8').replace(/^\uFEFF/, ''))
  const result = structuredClone(wire.analysis)
  result.source_filename = 'UNTRUSTED-UPLOAD.csv'
  result.intrusion_prediction_count = 9999
  const variants = [
    [{ state: { phase: 'complete', result }, session: { inspect: noop } }, { enabled: false }, { state: 'connected' }],
    [{ state: { phase: 'complete', result }, presentation: 'mock' }, { enabled: true, flowCount: 999999 }, { state: 'offline' }],
    [{ state: { phase: 'analyzing', result: null } }, { enabled: true }, { state: 'connecting' }],
    [{ state: { phase: 'analysis_error', result: null } }, { enabled: false }, { state: 'connected', status: { model_loaded: false } }],
  ]
  for (const args of variants) assert.equal(render(...args), html)
  assert.doesNotMatch(html, /UNTRUSTED-UPLOAD|9999|REAL ANALYSIS|NO MODEL OUTPUT|DEMO EVENT/)
  assert.match(html, /DEMO \/ SIMULATED activity and MOCK QA fixtures do not change this evidence/)
  assert.match(html, /Unlabeled uploaded CSVs cannot determine accuracy, precision, recall, F1, a confusion matrix or PR-AUC/)
})

test('snapshot is deeply immutable against accidental session mutation', () => {
  for (const mutation of [
    () => { evidence.e2.pruned.mean = 1 },
    () => { evidence.e2.seeds.push(2028) },
    () => { evidence.e2.perSeed[0].pruned = 1 },
    () => { evidence.benchmarks.peakReservedBytes.pruned = 0 },
    () => { graphComparison[0].pruned = 1 },
    () => { sources.comparison.path = 'invented.json' },
  ]) assert.throws(mutation, TypeError)
})

test('source references exist and match the exact frozen release Git blobs', () => {
  assert.match(html, /v1\.0-experimental-complete/)
  assert.ok(html.includes(evidence.release.commit))
  for (const source of Object.values(sources)) {
    const file = new URL(source.path, root)
    assert.ok(existsSync(file), source.path)
    assert.ok(html.includes(source.label), source.label)
    assert.ok(html.includes(source.path), source.path)
    const frozen = execFileSync('git', ['show', `${evidence.release.commit}:${source.path}`], { cwd: fileURLToPath(root), maxBuffer: 2 * 1024 * 1024 })
    assert.deepEqual(readFileSync(file), frozen, source.path)
  }
})

test('no blank precision, recall or latency card and no unsupported positive claim', () => {
  assert.doesNotMatch(html, /Evidence integration pending|No evidence connected|<h2>Precision<|<h2>Recall<|<h2>Inference latency<|<h2>Accuracy</)
  assert.doesNotMatch(html, /statistically significant improvement|guaranteed noninferiority|universal deployment speedup|training acceleration|smaller checkpoint|50% overhead reduction/i)
  assert.match(html, /These descriptive results do not establish statistical significance or statistical noninferiority/)
  assert.match(html, /neither end-to-end application speedup nor reduced total GPU or CPU memory/)
  assert.match(html, /Available without a backend connection/)
})
