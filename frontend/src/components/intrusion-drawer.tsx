import { Download, Layers3, Network, Search, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { StatusBadge } from '@/components/status-badge'
import { useDemo } from '@/lib/demo-context'
import { eventTime, severityTone } from '@/lib/demo-data'
import { projectFacts as facts } from '@/lib/project-facts'
import { useReplay } from '@/lib/replay-context'
import { RealIntrusionDrawer } from '@/components/real-intrusion-drawer'

export function IntrusionDrawer() {
  const { selectedEvent: event, inspect } = useDemo()
  const { state } = useReplay()
  if (state.result) return <RealIntrusionDrawer />
  return <Sheet open={Boolean(event)} onOpenChange={open => { if (!open) inspect(null) }}>
    <SheetContent side="right" overlayClassName="inspector-backdrop" className={`intrusion-drawer inspector-${event?.severity ?? 'info'}`}>
      <SheetHeader className="inspector-header"><p className="eyebrow">INTRUSION INSPECTOR</p>
        {event && <div className="inspector-badges"><StatusBadge tone={severityTone(event.severity)}>{event.severity.toUpperCase()}</StatusBadge><StatusBadge tone="info">DEMO EVENT</StatusBadge></div>}
        <SheetTitle>{event?.prediction ?? event?.title ?? 'Event details'}</SheetTitle>
        <SheetDescription>Frontend investigation preview. All event fields are synthetic.</SheetDescription>
      </SheetHeader>
      {event && <div className="inspector-scroll" role="region" aria-label="Intrusion details" tabIndex={0}>
        <dl className="inspector-identity"><div><dt>Timestamp</dt><dd>{eventTime(event.timestamp)} IST</dd></div><div><dt>Window identifier</dt><dd className="mono">{event.id}</dd></div></dl>
        <section className="inspector-section"><h3>DETECTION</h3><dl className="inspector-facts"><div><dt>Model score</dt><dd>{event.modelScore?.toFixed(3) ?? '—'}</dd></div><div><dt>Prediction</dt><dd>{event.prediction ?? 'No prediction for this event'}</dd></div><div><dt>Model / mode</dt><dd>{facts.model} / {facts.inferenceMode}</dd></div><div><dt>Latency</dt><dd>{event.latency ? `${event.latency.toFixed(1)} ms (synthetic)` : '—'}</dd></div></dl><p className="inspector-note">Scores and timings are UI fixtures, not model measurements.</p></section>
        <section className="inspector-section"><h3>FLOW WINDOW</h3><div className="flow-window-records" aria-label="Five consecutive clean flow records; fifth record is the target">{Array.from({ length: facts.windowSize }, (_, index) => <div key={index}><Layers3 size={14} aria-hidden="true" /><span>FLOW 0{index + 1}</span>{index === 4 && <strong>TARGET</strong>}</div>)}</div><p className="inspector-note">Five consecutive clean flow records. Prediction targets the fifth / final flow. Raw flow payloads are unavailable in this demo.</p></section>
        <section className="inspector-section"><h3>GRAPH</h3><div className="inspector-graph"><Network size={24} aria-hidden="true" /><span><strong>{facts.retainedFeatures} / {facts.inputFeatures}</strong> feature nodes retained</span><ArrowRight size={14} aria-hidden="true" /><span><strong>{facts.prunedNodes}</strong> nodes · <strong>{facts.prunedEdges}</strong> edges</span></div><p className="inspector-note">All 8 category nodes remain. Model parameters are unchanged.</p></section>
        <section className="inspector-section"><h3>EVENT TRACE <span>DEMO</span></h3><ol className="event-trace">{['Traffic validation', '5-flow window assembly', 'Graph construction', 'HTGNN inference', 'Prediction / alert result'].map((label, index) => <li key={label}><span>0{index + 1}</span><div><strong>{label}</strong><small>Simulated interface step</small></div></li>)}</ol></section>
      </div>}
      <div className="inspector-footer"><div><Button variant="outline" disabled aria-describedby="inspector-actions-note"><Search />Inspect Window</Button><Button variant="outline" disabled aria-describedby="inspector-actions-note"><Download />Export</Button></div><p id="inspector-actions-note">Window payloads and exports require backend data.</p></div>
    </SheetContent>
  </Sheet>
}
