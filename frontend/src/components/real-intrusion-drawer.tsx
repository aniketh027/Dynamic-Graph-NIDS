import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { StatusBadge } from '@/components/status-badge'
import { useReplay, useReplayPresentation } from '@/lib/replay-context'
import type { AnalysisResponse, Prediction } from '@/lib/api/replay'

export function RealIntrusionDrawer() {
  const presentation = useReplayPresentation()
  const { state: { selected: event, result }, session } = useReplay()
  return <Sheet open={Boolean(event && result)} onOpenChange={open => { if (!open) session.inspect(null) }}>
    <SheetContent side="right" overlayClassName="inspector-backdrop" className="intrusion-drawer inspector-intrusion replay-inspector">
      <SheetHeader className="inspector-header"><p className="eyebrow">INTRUSION INSPECTOR</p><div className="inspector-badges"><StatusBadge tone="bad">INTRUSION</StatusBadge><StatusBadge>{presentation.badge}</StatusBadge><StatusBadge>{presentation.qualifier}</StatusBadge></div><SheetTitle>{event?.prediction_label ?? 'Prediction details'}</SheetTitle><SheetDescription>{presentation.description}</SheetDescription></SheetHeader>
      {event && result && <RealIntrusionDetails event={event} result={result} />}
      <div className="inspector-footer"><p>{presentation.mock ? 'Mock fixture · No model output' : 'File replay result · Current frontend session only'}</p></div>
    </SheetContent>
  </Sheet>
}

export function RealIntrusionDetails({ event, result }: { event: Prediction; result: AnalysisResponse }) {
  const presentation = useReplayPresentation()
  return <div className="inspector-scroll" role="region" aria-label={`${presentation.kind} intrusion details`} tabIndex={0}>
        <dl className="inspector-identity"><div><dt>Replay target</dt><dd>Flow {event.target_flow_index}</dd></div><div><dt>Source filename</dt><dd>{result.source_filename}</dd></div><div><dt>Analysis ID</dt><dd className="mono">{result.analysis_id}</dd></div></dl>
        <section className="inspector-section"><h3>DETECTION</h3><dl className="inspector-facts"><div><dt>Model score</dt><dd>{event.model_score.toFixed(4)}</dd></div><div><dt>Window index</dt><dd>{event.window_index}</dd></div><div><dt>Target flow index</dt><dd>{event.target_flow_index}</dd></div><div><dt>Model</dt><dd>{result.model.task} {result.model.family} · seed {result.model.deployment_seed}</dd></div></dl><p className="inspector-note">Uncalibrated normalized score for the predicted class.</p></section>
        <section className="inspector-section"><h3>FLOW WINDOW</h3><p>{result.preprocessing.temporal_window} consecutive flow records in CSV row order. Non-overlapping windows; the fifth/final flow is the target.</p></section>
        <section className="inspector-section"><h3>GRAPH</h3><p>{result.graph.retained_feature_nodes}/{result.graph.input_feature_nodes} feature nodes retained · {result.graph.category_nodes} categories · {result.graph.total_nodes_per_graph} nodes · {result.graph.edges_per_graph} edges</p><p className="inspector-note">{result.graph.parameter_count.toLocaleString()} parameters unchanged · {result.model.inference_mode}</p></section>
        <section className="inspector-section"><h3>MODEL IDENTITY</h3><p className="mono">{result.model.checkpoint_id}</p><p className="inspector-note">Checkpoint SHA-256: {result.model.checkpoint_sha256}</p></section>
        <section className="inspector-section"><h3>APPLICATION TRACE</h3><ol className="event-trace">{['File validated', 'Analysis submitted', 'Prediction returned', 'Intrusion class predicted'].map((label, index) => <li key={label}><span>0{index + 1}</span><div><strong>{label}</strong></div></li>)}</ol></section>
      </div>
}
