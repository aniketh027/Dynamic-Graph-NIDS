import { useState } from 'react'
import type { AnalysisResponse, Prediction } from '@/lib/api/replay'
import { intrusionPredictions, predictionPage } from '@/lib/replay-session'
import { useReplay, useReplayPresentation } from '@/lib/replay-context'
import { Panel } from '@/components/panel'
import { StatCard } from '@/components/stat-card'
import { StatusBadge } from '@/components/status-badge'
import { Button } from '@/components/ui/button'

export function PredictionTable({ predictions }: { predictions: Prediction[] }) {
  const [page, setPage] = useState(0)
  const { session } = useReplay()
  const pages = Math.max(1, Math.ceil(predictions.length / 50))
  const current = Math.min(page, pages - 1)
  const visible = predictionPage(predictions, current)
  return <div className="replay-predictions"><p className="replay-helper">Window and target flow indices are zero-based. Model score is an uncalibrated normalized score for the predicted class.</p>
    <div className="replay-table-scroll" tabIndex={0} role="region" aria-label="Prediction table"><table className="replay-table"><thead><tr><th>Window</th><th>Target flow</th><th>Prediction</th><th>Model score</th><th>Status</th><th><span className="sr-only">Inspect</span></th></tr></thead><tbody>{visible.map(p => <tr key={p.window_index}><td>{p.window_index}</td><td>{p.target_flow_index}</td><td>{p.prediction_label}</td><td>{p.model_score.toFixed(4)}</td><td><StatusBadge tone={p.is_attack ? 'bad' : 'good'}>{p.is_attack ? 'INTRUSION' : 'BENIGN'}</StatusBadge></td><td>{p.is_attack && <Button variant="ghost" size="sm" aria-label={`Inspect window ${p.window_index}`} onClick={() => session.inspect(p)}>Inspect</Button>}</td></tr>)}</tbody></table></div>
    {!predictions.length && <p className="replay-notice">No intrusion predictions in this analysis.</p>}
    <div className="replay-mobile-cards">{visible.map(p => <article className="replay-prediction-card" key={p.window_index} aria-label={`Prediction window ${p.window_index}`}>
      <dl><div><dt>Prediction</dt><dd>{p.prediction_label}</dd></div><div><dt>Status</dt><dd><StatusBadge tone={p.is_attack ? 'bad' : 'good'}>{p.is_attack ? 'INTRUSION' : 'BENIGN'}</StatusBadge></dd></div><div><dt>Window</dt><dd>{p.window_index}</dd></div><div><dt>Target flow</dt><dd>{p.target_flow_index}</dd></div><div><dt>Model score</dt><dd>{p.model_score.toFixed(4)}</dd></div></dl>
      {p.is_attack && <Button variant="outline" size="sm" aria-label={`Inspect window ${p.window_index}`} onClick={() => session.inspect(p)}>Inspect</Button>}
    </article>)}</div>
    {predictions.length > 0 && <div className="replay-pagination"><Button variant="outline" size="sm" disabled={current === 0} onClick={() => setPage(current - 1)}>Previous</Button><span role="status">Page {current + 1} of {pages} · {predictions.length} predictions · 50 per page</span><Button variant="outline" size="sm" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Next</Button></div>}
  </div>
}
export function ClassDistribution({ result }: { result: AnalysisResponse }) {
  return <Panel title="Class distribution" subtitle="Returned prediction counts · All model classes"><div className="replay-distribution">{Object.entries(result.prediction_class_counts).map(([label, count]) => <div key={label}><span>{label}</span><meter aria-label={`${label} predictions`} min={0} max={result.complete_window_count} value={count} /><strong>{count}</strong></div>)}</div></Panel>
}
export function ReplayResults({ result, dashboard = false }: { result: AnalysisResponse; dashboard?: boolean }) {
  const presentation = useReplayPresentation()
  return <section className="replay-results" aria-label={`${presentation.kind} analysis results`}><div className="replay-result-heading"><StatusBadge tone={presentation.mock ? 'warning' : 'good'}>{presentation.badge}</StatusBadge><StatusBadge>{presentation.qualifier}</StatusBadge><span>{result.source_filename}</span></div>
    <p className="replay-helper">Analysis complete · Current frontend session only; refreshing clears results. These are inference counts, not research performance measurements.</p>
    <div className="stat-grid"><StatCard title="Total predictions" value={result.complete_window_count} description={`${result.input_flow_count} input flow rows`} /><StatCard title="Benign" value={result.benign_prediction_count} description="Returned benign predictions" /><StatCard title="Intrusions" value={result.intrusion_prediction_count} description="Returned attack predictions" /><StatCard title="Trailing flows" value={result.trailing_flow_count} description="Incomplete window · not inferred" /></div>
    <div className="replay-model-facts">{result.model.task} {result.model.family} · seed {result.model.deployment_seed} · {result.model.inference_mode}<br />{result.graph.retained_feature_nodes}/{result.graph.input_feature_nodes} features retained · {result.graph.category_nodes} categories · {result.graph.total_nodes_per_graph} nodes · {result.graph.edges_per_graph} edges · {result.graph.parameter_count.toLocaleString()} parameters unchanged</div>
    <ClassDistribution result={result} />
    <Panel title={dashboard ? `Recent ${presentation.predictions.toLowerCase()}` : `${presentation.mock ? 'Mock ' : ''}Window predictions`} subtitle={dashboard ? 'Most recent 50 by CSV replay order' : 'One prediction per non-overlapping five-flow window'}><PredictionTable key={result.analysis_id} predictions={dashboard ? intrusionPredictions(result).slice(-50).reverse() : result.predictions} /></Panel>
  </section>
}
