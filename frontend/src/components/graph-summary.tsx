import { ArrowRight, Network } from 'lucide-react'
import { Link } from 'react-router'
import { Panel } from '@/components/panel'
import { StatusBadge } from '@/components/status-badge'
import { projectFacts as facts } from '@/lib/project-facts'

export function GraphSummary({ expanded = false }: { expanded?: boolean }) {
  return (
    <Panel title="Graph optimization" subtitle="Attention-guided inference-time pruning"
      action={<StatusBadge tone="info">Frozen policy</StatusBadge>} className="graph-summary">
      <div className="retention-summary"><Network size={19} aria-hidden="true" /><span>Feature nodes retained</span><strong>{facts.retainedFeatures}<span> / {facts.inputFeatures}</span></strong></div>
      <div className="retention-bar" role="img" aria-label={`${facts.retainedFeatures} retained and ${facts.removedFeatures} removed feature nodes`}>
        <span style={{ width: `${facts.retainedFeatures / facts.inputFeatures * 100}%` }} /><span className="removed-segment" />
      </div>
      <div className="retention-legend"><span><i className="legend-square retained" />{facts.retainedFeatures} retained</span><span><i className="legend-square removed" />{facts.removedFeatures} removed</span></div>
      <dl className="graph-facts">
        <div><dt>Total graph nodes</dt><dd><span>{facts.unprunedNodes}</span><ArrowRight size={12} aria-hidden="true" /><strong>{facts.prunedNodes}</strong></dd></div>
        <div><dt>Directed graph edges</dt><dd><span>{facts.unprunedEdges}</span><ArrowRight size={12} aria-hidden="true" /><strong>{facts.prunedEdges}</strong></dd></div>
        {expanded && <><div><dt>Input schema features</dt><dd>{facts.inputFeatures}</dd></div><div><dt>Category nodes retained</dt><dd>{facts.categoryNodes} / {facts.categoryNodes}</dd></div></>}
      </dl>
      <p className="graph-note">{expanded ? 'All 8 category nodes and the sample node remain. The validated input schema still requires 78 model-input features. Model parameters are unchanged.' : 'Inference graph reduced. Model parameters unchanged.'}</p>
      {!expanded && <Link to="/graph-optimization" className="text-link">Inspect graph structure<ArrowRight size={13} aria-hidden="true" /></Link>}
    </Panel>
  )
}
