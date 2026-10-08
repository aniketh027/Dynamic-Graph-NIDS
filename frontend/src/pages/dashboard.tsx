import { m, useReducedMotion } from 'motion/react'
import { Box, FileUp, GitBranch, Layers3, Network } from 'lucide-react'
import { Link } from 'react-router'
import { Button } from '@/components/ui/button'
import { BackendNotice, PageHeading } from '@/components/app-shell'
import { StatusBadge } from '@/components/status-badge'
import { StatCard } from '@/components/stat-card'
import { TrafficActivityPanel, DetectionActivityPanel } from '@/components/activity-panels'
import { PipelinePanel } from '@/components/pipeline-panel'
import { EventsPanel } from '@/components/events-panel'
import { GraphSummary } from '@/components/graph-summary'
import { projectFacts as facts } from '@/lib/project-facts'
import { DemoControls } from '@/components/demo-controls'
import { useDemo } from '@/lib/demo-context'
import { useReplay } from '@/lib/replay-context'
import { ReplayResults } from '@/components/replay-results'

export function Dashboard() {
  const demo = useDemo()
  const { state } = useReplay()
  const reducedMotion = useReducedMotion()
  return (
    <m.div className="page-content" initial={reducedMotion ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2 }}>
      <PageHeading eyebrow="NETWORK INTRUSION DETECTION CONSOLE" title="Dynamic Graph NIDS"
        description="A clear view of your traffic, detections, and inference graph."
        action={<Button asChild className="analyze-button"><Link to="/analyze-traffic"><FileUp size={15} />Analyze traffic</Link></Button>} />
      <div className="overview-meta"><StatusBadge dot tone="good">Console ready</StatusBadge><span className="meta-divider" /><span>{facts.model}</span><StatusBadge tone="info">{facts.inferenceMode}</StatusBadge><span className="overview-context">Offline / file replay</span></div>
      <BackendNotice />
      <DemoControls />
      {state.result ? <ReplayResults result={state.result} dashboard /> : state.phase === 'analyzing' || state.phase === 'validating' ? <p role="status" className="replay-notice">{state.phase === 'analyzing' ? 'Analysis in progress. Waiting for the backend result.' : 'Validating traffic file. Model inference has not run yet.'}</p> : <><section className="stat-grid" aria-label="Graph and session overview">
        <StatCard title="Features retained" value={{ numerator: facts.retainedFeatures, denominator: facts.inputFeatures }} icon={Layers3} description={`${facts.removedFeatures} feature nodes removed · Frozen policy`} />
        <StatCard title="Graph nodes" value={facts.prunedNodes} icon={Network} description={`${facts.unprunedNodes} → ${facts.prunedNodes} · Attention-pruned graph`} />
        <StatCard title="Graph edges" value={facts.prunedEdges} icon={GitBranch} description={`${facts.unprunedEdges} → ${facts.prunedEdges} · Directed structural edges`} />
        <StatCard title={demo.enabled ? 'Demo flow records' : 'Flows analyzed'} value={demo.enabled ? demo.flowCount : 0} icon={Box} description={demo.enabled ? 'DEMO · Synthetic counter' : 'Waiting for analysis'} />
      </section>
      <div className="analytics-grid"><TrafficActivityPanel /><DetectionActivityPanel /></div>
      <PipelinePanel />
      <div className="detail-grid"><EventsPanel /><GraphSummary /></div></>}
    </m.div>
  )
}
