import { useState } from 'react'
import { ArrowRight, Box, Layers3, Network } from 'lucide-react'
import { Panel } from '@/components/panel'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { projectFacts as facts } from '@/lib/project-facts'

const removedIds = new Set<number>(facts.removedFeatureIds)
const featureIds = Array.from({ length: facts.inputFeatures }, (_, id) => id)

export function GraphStructure() {
  const [mode, setMode] = useState('pruned')
  const [inspectedFeature, setInspectedFeature] = useState<number | null>(null)
  const pruned = mode === 'pruned'
  const featureLabel = (id: number) => `Feature ${id} · ${removedIds.has(id) ? (pruned ? 'Removed by frozen pruning policy' : 'Included in unpruned graph; removed by frozen pruning policy') : 'Retained'}`
  return <Panel title="Inference graph structure" subtitle="Frozen research topology · Node-group illustration">
    <Tabs value={mode} onValueChange={setMode} className="graph-tabs"><TabsList><TabsTrigger value="unpruned">Unpruned</TabsTrigger><TabsTrigger value="pruned">Attention-pruned</TabsTrigger></TabsList></Tabs>
    <section className="feature-map" aria-label="Frozen feature-node policy">
      <div className="feature-map-heading"><h3>{facts.inputFeatures} INPUT FEATURE NODES</h3><div className="feature-map-legend"><span><i className="feature-key retained-key" />{pruned ? facts.retainedFeatures : facts.inputFeatures} retained</span>{pruned && <span><i className="feature-key removed-key" />{facts.removedFeatures} removed</span>}</div></div>
      <ol className="feature-node-grid" aria-label="Original feature IDs 0 through 77">{featureIds.map(id => <li key={id}><button type="button" className={`feature-indicator${pruned && removedIds.has(id) ? ' feature-removed' : ' feature-retained'}`} aria-label={featureLabel(id)} title={featureLabel(id)} onMouseEnter={() => setInspectedFeature(id)} onFocus={() => setInspectedFeature(id)} onClick={() => setInspectedFeature(id)}>{String(id).padStart(2, '0')}</button></li>)}</ol>
      <p className="feature-inspection" aria-live="polite">{inspectedFeature === null ? 'Hover or focus an original feature ID to inspect its frozen policy state.' : featureLabel(inspectedFeature)}</p>
      <div className="feature-map-flow"><span>{facts.inputFeatures}-feature input schema stays intact</span><ArrowRight size={13} aria-hidden="true" /><strong>{pruned ? facts.retainedFeatures : facts.inputFeatures} feature nodes at inference</strong></div>
    </section>
    <div className="graph-structure"><div><Layers3 size={26} strokeWidth={1.2} aria-hidden="true" /><strong>{pruned ? facts.retainedFeatures : facts.inputFeatures}</strong><span>Feature nodes</span></div><ArrowRight size={17} aria-hidden="true" /><div><Network size={26} strokeWidth={1.2} aria-hidden="true" /><strong>{facts.categoryNodes}</strong><span>Category nodes</span></div><ArrowRight size={17} aria-hidden="true" /><div><Box size={26} strokeWidth={1.2} aria-hidden="true" /><strong>1</strong><span>Sample node</span></div></div>
    <div className="graph-structure-footer"><span>{facts.unprunedNodes} → {facts.prunedNodes} total nodes</span><span>{facts.unprunedEdges} → {facts.prunedEdges} directed edges</span><span>Parameters unchanged</span></div>
  </Panel>
}
