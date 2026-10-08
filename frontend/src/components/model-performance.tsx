import { ArrowRight, Database, FileCheck2, Info, TriangleAlert } from 'lucide-react'
import { PageHeading } from '@/components/app-shell'
import { Panel } from '@/components/panel'
import { StatCard } from '@/components/stat-card'
import { StatusBadge } from '@/components/status-badge'
import { evidenceSources, graphComparison, modelPerformanceEvidence as evidence } from '@/lib/model-performance-evidence'
import { projectFacts as facts } from '@/lib/project-facts'

const metric = (value: number) => value.toFixed(6)
const count = (value: number) => value.toLocaleString('en-US')
const mib = (bytes: number) => bytes / 1024 ** 2

export function ModelPerformance() {
  const { e2, benchmarks, diagnostic } = evidence
  const savedBytes = benchmarks.peakAllocatedBytes.unpruned - benchmarks.peakAllocatedBytes.pruned

  return <div className="page-content performance-page">
    <PageHeading eyebrow="FROZEN SCIENTIFIC EVIDENCE" title="Model Performance"
      description="Completed E.11 / E.12B research results, preserved with their evaluation scope."
      action={<StatusBadge tone="good" dot>Research complete</StatusBadge>} />

    <section className="performance-overview" aria-label="Scientific evidence overview">
      <div className="performance-identity"><Database size={17} aria-hidden="true" /><span><strong>{facts.dataset}</strong><span>E2 primary supervised multiclass · Three frozen HTGNN seeds · Fixed attention-pruned graph</span></span></div>
      <p><Info size={14} aria-hidden="true" /><span>Historical research measurements. Uploaded CSV predictions, DEMO / SIMULATED activity and MOCK QA fixtures do not change this evidence. Available without a backend connection.</span></p>
    </section>

    <nav className="performance-nav" aria-label="Model Performance sections">
      <a href="#e2-comparison">E2 comparison</a><a href="#graph-efficiency">Graph structure</a><a href="#validation-efficiency">GPU validation</a><a href="#scientific-limitations">Limitations</a><a href="#evidence-provenance">Sources</a>
    </nav>

    <div className="stat-grid performance-stats">
      <StatCard title="E2 unpruned mean macro-F1" value={metric(e2.unpruned.mean)} description="Frozen test · Eligible classes" />
      <StatCard title="E2 attention-pruned mean macro-F1" value={metric(e2.pruned.mean)} description="Frozen test · Same reporting definition" />
      <StatCard title="Frozen HTGNN seeds" value={e2.seeds.length} description={e2.seeds.join(' / ')} />
      <StatCard title="Retained feature nodes" value={{ numerator: facts.retainedFeatures, denominator: facts.inputFeatures }} description="Full 78-feature CSV still required" />
    </div>

    <div className="performance-comparison-grid">
      <section id="e2-comparison" aria-label="Primary E2 performance comparison">
        <Panel title="Unpruned vs attention-pruned HTGNN" subtitle="E2 primary supervised multiclass · Frozen test evaluation" action={<StatusBadge tone="info">E.11 → E.12B</StatusBadge>}>
          <figure className="performance-chart" aria-labelledby="e2-chart-caption">
            <figcaption id="e2-chart-caption">Mean eligible-class macro-F1 <span>Scale 0–1 · Higher is better</span></figcaption>
            {([['Unpruned HTGNN', e2.unpruned, 'unpruned'], ['Attention-pruned HTGNN', e2.pruned, 'pruned']] as const).map(([label, result, key]) => <div className={`performance-bar-row ${key}`} key={key}>
              <div className="performance-bar-label"><span>{label}</span><strong>{metric(result.mean)}</strong></div>
              <div className="performance-bar-track" aria-hidden="true"><span style={{ width: `${result.mean * 100}%` }} /></div>
              <p>Across-seed sample SD: <strong>{metric(result.sampleSD)}</strong></p>
            </div>)}
            <div className="performance-axis" aria-hidden="true"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1</span></div>
          </figure>
          <div className="performance-body">
            <p className="performance-definition">Equal-weight mean of per-class F1 for classes with train ≥100, validation ≥30 and test ≥30 flow-window samples. True test rows are restricted to eligible classes; all 15 output classes remain available for prediction.</p>
            <table className="performance-table performance-seeds">
              <caption>Three frozen seeds · Eligible-class test macro-F1</caption>
              <thead><tr><th scope="col">Seed</th><th scope="col">Unpruned</th><th scope="col">Attention-pruned</th><th scope="col">Absolute change</th></tr></thead>
              <tbody>{e2.perSeed.map(row => <tr key={row.seed}><th scope="row">{row.seed}</th><td>{metric(row.unpruned)}</td><td>{metric(row.pruned)}</td><td className={row.change > 0 ? 'performance-positive' : 'performance-caution'}>{row.change > 0 ? '+' : '−'}{metric(Math.abs(row.change))}</td></tr>)}</tbody>
            </table>
            <p className="performance-note">Mean and sample SD (ddof=1) across seeds {e2.seeds.join(', ')}. One seed improved; two declined slightly. These descriptive results do not establish statistical significance or statistical noninferiority.</p>
          </div>
          <div className="performance-panel-footer"><FileCheck2 size={13} aria-hidden="true" />One E.12A validation-selected, frozen policy; trained HTGNN weights unchanged.</div>
        </Panel>
      </section>

      <section id="graph-efficiency" aria-label="Graph optimization efficiency">
        <Panel title="Less inference graph work" subtitle="Per-graph structure · Same trained E2 HTGNN">
          <div className="performance-graph-summary"><span>{facts.unprunedNodes}<small>unpruned nodes</small></span><ArrowRight size={20} aria-hidden="true" /><span>{facts.prunedNodes}<small>pruned nodes</small></span></div>
          <div className="performance-body">
            <table className="performance-table performance-structure"><caption className="sr-only">Unpruned and fixed attention-pruned graph counts</caption>
              <thead><tr><th scope="col">Structural measure</th><th scope="col">Unpruned</th><th scope="col">Pruned</th></tr></thead>
              <tbody>{graphComparison.map(row => <tr key={row.label}><th scope="row">{row.label}</th><td>{count(row.unpruned)}</td><td>{count(row.pruned)}</td></tr>)}</tbody>
            </table>
            <p className="performance-note">Attention-guided inference-time graph pruning; no model weight pruning or retraining.</p>
            <p className="performance-input-note"><Info size={14} aria-hidden="true" /><span>The application still requires the full canonical {facts.inputFeatures}-feature CSV. The input adapter builds the full graph before feature compaction.</span></p>
          </div>
        </Panel>
      </section>
    </div>

    <section id="validation-efficiency" aria-label="Validation-only efficiency measurements">
      <Panel title="Frozen GPU validation measurements" subtitle={`${benchmarks.gpu} · Resident-forward benchmark`} action={<StatusBadge tone="info">Validation only</StatusBadge>}>
        <div className="performance-benchmark-grid">
          <div className="performance-benchmark"><h3>Paired median speed ratios</h3><strong>{metric(benchmarks.pairedSpeedRatio.min)}×–{metric(benchmarks.pairedSpeedRatio.max)}×</strong><p>Range across E1 / E2 / E3 and all three seeds; median of paired unpruned / pruned forward times.</p></div>
          <div className="performance-benchmark"><h3>Peak PyTorch allocated · E1 / E2</h3><strong>{mib(benchmarks.peakAllocatedBytes.unpruned).toFixed(3)} <span>→</span> {mib(benchmarks.peakAllocatedBytes.pruned).toFixed(3)} <small>MiB</small></strong><p>Approx. {mib(savedBytes).toFixed(3)} MiB / {(savedBytes / benchmarks.peakAllocatedBytes.unpruned * 100).toFixed(2)}% less transient allocated peak.</p></div>
          <div className="performance-benchmark"><h3>Peak PyTorch reserved · All tasks</h3><strong>{mib(benchmarks.peakReservedBytes.unpruned)} <span>→</span> {mib(benchmarks.peakReservedBytes.pruned)} <small>MiB</small></strong><p>Reservation unchanged. Caching allocator measurements include the model and resident full adapter input.</p></div>
        </div>
        <div className="performance-body performance-protocol"><p>Four sequential validation batches of {benchmarks.batchSize} windows per task and seed; {benchmarks.warmupsPerPathPerBatch} warmups per path per batch, {benchmarks.measuredPairsPerBatch} measured pairs per batch (80 pairs), alternating path order. Float32 with strict deterministic algorithms.</p><p>Forward timing includes input checks, compaction, encoders and head. Loading, conversion and host-to-device transfer are excluded. These GPU observations do not measure current CPU CSV upload/inference latency or establish deployment-wide speed or memory gains.</p></div>
      </Panel>
    </section>

    <section id="scientific-limitations" aria-label="Scientific limitations">
      <Panel className="performance-limitations" title="Scientific limitations" subtitle="Read alongside the positive E2 finding" action={<TriangleAlert size={17} aria-hidden="true" />}>
        <div className="performance-limitation-grid">
          <div><h3>E1 · Diagnostic only</h3><p>Mean support-restricted macro-F1 {metric(diagnostic.e1.unpruned)} → {metric(diagnostic.e1.pruned)}: essentially unchanged. This reporting subset contains only Benign, so it does not establish strong chronological multiclass attack detection.</p></div>
          <div><h3>E3 · Poor chronological generalization</h3><p>Mean average-precision PR-AUC {metric(diagnostic.e3.unpruned)} → {metric(diagnostic.e3.pruned)} (approximately 0.089), essentially unchanged. Attention pruning did not resolve the chronological binary generalization failure.</p></div>
          <div><h3>E2 · Observed comparison</h3><p>The mean increase is not evidence of statistical significance or a formal noninferiority guarantee. E2 preserves within-class chronology rather than a global timeline. Repeated network states can occur across partitions.</p></div>
          <div><h3>Application · CPU offline replay</h3><p>The application currently uses CPU inference and offline file replay. The validation GPU measurements establish neither end-to-end application speedup nor reduced total GPU or CPU memory.</p></div>
        </div>
        <p className="performance-note performance-arithmetic-note">Historical E.11 CUDA arithmetic is not certified bitwise equivalent to the strict E.12 deterministic context. Attention is a ranking heuristic, not causal feature importance.</p>
      </Panel>
    </section>

    <section id="evidence-provenance" aria-label="Provenance and metric interpretation">
      <Panel title="Evidence provenance & interpretation" subtitle="Repository-relative references · Compact frozen results">
        <div className="performance-release"><StatusBadge tone="good">Frozen scientific release</StatusBadge><code>{evidence.release.tag}</code><code>{evidence.release.commit}</code></div>
        <div className="performance-body performance-interpretation"><p>{facts.dataset} · T={facts.windowSize} consecutive flow records; the final flow supplies the graph target. E.11 unpruned test results were reused in E.12B. E.12A selected the fixed mask using training attention and validation only.</p><p>Unlabeled uploaded CSVs cannot determine accuracy, precision, recall, F1, a confusion matrix or PR-AUC. Session predictions and their uncalibrated Model score are separate from every historical metric on this page.</p></div>
        <ul className="performance-sources">{Object.values(evidenceSources).map(source => <li key={source.path}><span>{source.label}</span><code>{source.path}</code></li>)}</ul>
      </Panel>
    </section>
  </div>
}
