import { Activity, ArrowLeft, ArrowUpRight, ShieldAlert } from 'lucide-react'
import { Link } from 'react-router'
import { Button } from '@/components/ui/button'
import { PageHeading } from '@/components/app-shell'
import { EmptyState } from '@/components/empty-state'
import { GraphSummary } from '@/components/graph-summary'
import { Panel } from '@/components/panel'
import { PipelinePanel } from '@/components/pipeline-panel'
import { StatCard } from '@/components/stat-card'
import { StatusBadge } from '@/components/status-badge'
import { projectFacts as facts } from '@/lib/project-facts'
import { TrafficFileInput } from '@/components/traffic-file-input'
import { ProcessingScanner } from '@/components/processing-scanner'
import { DemoControls } from '@/components/demo-controls'
import { DemoAlert } from '@/components/alert-card'
import { EventsPanel } from '@/components/events-panel'
import { GraphStructure } from '@/components/graph-structure'
import { useDemo } from '@/lib/demo-context'
import { demoPipelineStage } from '@/lib/demo-data'
import { useBackend } from '@/lib/api/context'
import { adapterLabel, connectionLabel } from '@/lib/api/client'
import { useReplay, useReplayPresentation } from '@/lib/replay-context'
import { ReplayResults, PredictionTable } from '@/components/replay-results'
import { intrusionPredictions } from '@/lib/replay-session'

const analyzeLink = <Button variant="outline" asChild><Link to="/analyze-traffic">Open Analyze Traffic<ArrowUpRight size={14} /></Link></Button>

export function AnalyzeTraffic() {
  const demo = useDemo()
  const { state } = useReplay()
  const backend = useBackend()
  const stage = demoPipelineStage(demo.step)
  const preparing = stage >= 1 && stage <= 3
  const scannerLabel = ['SCANNING TRAFFIC FLOWS', 'VALIDATING TRAFFIC FILE', 'PREPARING 5-FLOW WINDOWS', 'BUILDING GRAPH', 'ANALYSIS PREVIEW', 'PREDICTION PREVIEW'][stage]
  return <div className="page-content"><PageHeading eyebrow="OFFLINE ANALYSIS" title="Analyze Traffic" description="Validate a strict application CSV, then run the frozen model on its flow windows." action={<StatusBadge>{state.phase.replaceAll('_', ' ')}</StatusBadge>} />
    {state.result && <ReplayResults result={state.result} />}<Panel title="Traffic input" subtitle="CSV / file replay workspace">
      <TrafficFileInput />
      <section className="input-contract"><h3>INPUT CONTRACT</h3><dl className="input-contract-grid"><div><dt>Required schema</dt><dd>{facts.inputFeatures} model-input features</dd></div><div><dt>Temporal window</dt><dd>{facts.windowSize} consecutive flow records in CSV row order</dd></div><div><dt>Prediction target</dt><dd>5th / final flow</dd></div><div><dt>Inference graph</dt><dd>{facts.retainedFeatures} retained feature nodes</dd></div><div><dt>Backend</dt><dd>{connectionLabel(backend)}</dd></div><div><dt>Model adapter</dt><dd>{adapterLabel(backend)}</dd></div><div><dt>Traffic analysis</dt><dd>Bounded CSV replay</dd></div></dl></section>
    </Panel><DemoControls />{demo.mode === 'running' && <Panel className="demo-scanner-panel" title={preparing ? 'Demo preparation state' : 'Demo analysis state'} subtitle="DEMO / SIMULATED · Frontend only"><ProcessingScanner variant={preparing ? 'crosshair' : 'radar'} active compact label={`DEMO · ${scannerLabel}`} description="Synthetic interface animation only. No file is processed and no backend progress is measured." /></Panel>}{demo.enabled && <PipelinePanel />}</div>
}

export function NetworkMonitor() {
  const backend = useBackend()
  return <div className="page-content"><PageHeading eyebrow="NETWORK WORKSPACE" title="Network Monitor" description="Traffic visibility for future connected analysis sessions." action={<StatusBadge dot tone="warning">Capture not connected</StatusBadge>} />
    <div className="capture-status-strip"><span><strong>LIVE CAPTURE</strong>Not connected</span><span><strong>APPLICATION MODE</strong>Offline / file replay</span><span><strong>NETWORK INTERFACE</strong><Button variant="outline" size="sm" disabled>Selection coming later</Button></span></div><Panel title="Network activity" subtitle="No live capture backend is connected"><div className="monitor-empty"><div className="chart-grid" aria-hidden="true" /><EmptyState icon={Activity} title="Live capture not connected" description="Use traffic replay / analysis mode. Live packet capture is reserved for a future phase." action={analyzeLink} /></div></Panel>
    <Panel title="Current capabilities" subtitle="Application UI and frozen input contract"><dl className="capabilities-grid"><div><dt>CSV / file replay</dt><dd>Strict 78-feature CSV · Bounded replay available</dd></div><div><dt>Validated input schema</dt><dd>{facts.inputFeatures} model-input features</dd></div><div><dt>Temporal window</dt><dd>{facts.windowSize} consecutive flow records in CSV row order</dd></div><div><dt>Model</dt><dd>{facts.model}</dd></div><div><dt>Inference mode</dt><dd>{facts.inferenceMode}</dd></div><div><dt>Live interface capture</dt><dd>Not connected · Future phase</dd></div></dl></Panel>
    <div className="placeholder-note"><span className="tiny-dot" />{connectionLabel(backend)} · {adapterLabel(backend)} · Live capture unsupported · File replay available in Analyze Traffic; no live traffic is displayed.</div></div>
}

export function IntrusionAlerts() {
  const presentation = useReplayPresentation()
  const demo = useDemo()
  const { state } = useReplay()
  if (state.result) return <div className="page-content"><PageHeading eyebrow={`${presentation.badge} · ${presentation.qualifier}`} title="Intrusion Alerts" description="Attack predictions from the latest analysis in this frontend session." action={<StatusBadge tone="bad">{state.result.intrusion_prediction_count} intrusions</StatusBadge>} /><Panel title={presentation.predictions} subtitle={state.result.source_filename}><PredictionTable key={state.result.analysis_id} predictions={intrusionPredictions(state.result)} /></Panel><DemoControls /></div>
  return <div className="page-content"><PageHeading eyebrow="EVENT INVESTIGATION" title="Intrusion Alerts" description="Inspect intrusion events and trace them through the detection pipeline." action={<StatusBadge tone={demo.enabled ? 'info' : 'neutral'}>{demo.enabled ? `${demo.events.length} demo events` : '0 events'}</StatusBadge>} />
    {demo.enabled && <div className="severity-key" aria-label="Event severity legend"><span>SEVERITY</span><StatusBadge>INFO</StatusBadge><StatusBadge tone="good">SAFE</StatusBadge><StatusBadge tone="warning">WARNING</StatusBadge><StatusBadge tone="bad">HIGH</StatusBadge><StatusBadge tone="bad">CRITICAL</StatusBadge><small>Legend only</small></div>}
    <DemoControls />{demo.enabled ? <div className="demo-alert-grid">{demo.events.filter(event => ['high', 'critical'].includes(event.severity)).slice(0, 2).map(event => <DemoAlert key={event.id} event={event} dismissible />)}</div> : <Panel title="Intrusion event stream" subtitle="Model scores and event details will appear after analysis"><EmptyState icon={ShieldAlert} title="No intrusion events to inspect" description="No completed real analysis in this session. Validate and analyze a CSV to inspect returned predictions." action={analyzeLink} /></Panel>}{demo.enabled && <EventsPanel full />}</div>
}

export function GraphOptimization() {
  return <div className="page-content"><PageHeading eyebrow="INFERENCE STRUCTURE" title="Graph Optimization" description="Attention-guided inference-time graph pruning." action={<StatusBadge tone="info">Frozen research policy</StatusBadge>} />
    <div className="stat-grid"><StatCard title="Input features" value={facts.inputFeatures} description="Validated model-input schema" /><StatCard title="Retained feature nodes" value={facts.retainedFeatures} description={`${facts.removedFeatures} feature nodes removed`} /><StatCard title="Total graph nodes" value={facts.prunedNodes} description={`${facts.unprunedNodes} → ${facts.prunedNodes}`} /><StatCard title="Graph edges" value={facts.prunedEdges} description={`${facts.unprunedEdges} → ${facts.prunedEdges} directed edges`} /></div>
    <GraphStructure />
    <div className="graph-page-grid"><GraphSummary expanded /><Panel title="What the policy changes" subtitle="Graph structure at inference"><div className="contract-explanation"><h3>One frozen attention-derived mask</h3><p>The inference graph retains {facts.retainedFeatures} of {facts.inputFeatures} feature nodes, all {facts.categoryNodes} category nodes, and the sample node.</p><h3>Model parameters stay unchanged</h3><p>The policy reduces graph computation during inference. It does not change learned model weights.</p><h3>The input schema remains intact</h3><p>Supply the validated {facts.inputFeatures}-feature input schema. A file containing only {facts.retainedFeatures} arbitrary features is not a supported input.</p><StatusBadge>Frozen policy · Used by application inference</StatusBadge></div></Panel></div></div>
}

export { ModelPerformance } from '@/components/model-performance'

export function NotFound() {
  return <div className="page-content"><PageHeading eyebrow="404 / UNKNOWN ROUTE" title="Page not found" description="This route is not part of the analysis console." /><Button variant="outline" asChild><Link to="/"><ArrowLeft size={14} />Return to dashboard</Link></Button></div>
}
