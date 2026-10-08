import { ArrowRight, Binary, Cpu, FileInput, Layers3, Network, SlidersHorizontal } from 'lucide-react'
import { Panel } from '@/components/panel'
import { StatusBadge } from '@/components/status-badge'
import { projectFacts as facts } from '@/lib/project-facts'
import { useDemo } from '@/lib/demo-context'
import { demoPipelineStage } from '@/lib/demo-data'

const steps = [
  { title: 'Traffic Input', detail: 'Offline / file replay', icon: FileInput },
  { title: 'Flow Preprocessing', detail: `${facts.inputFeatures} input features`, icon: SlidersHorizontal },
  { title: `${facts.windowSize}-Flow Window`, detail: 'Consecutive clean records', icon: Layers3 },
  { title: 'Graph Construction', detail: 'Feature → category → sample', icon: Network },
  { title: facts.model, detail: 'Attention-pruned inference', icon: Cpu },
  { title: 'Prediction', detail: 'Fifth / final flow target', icon: Binary },
] as const

export function PipelinePanel() {
  const demo = useDemo()
  const active = demo.mode === 'running'
  const stage = demoPipelineStage(demo.step)
  const result = demo.events.find(event => event.prediction)
  const predictionTone = result?.severity === 'safe' ? ' pipeline-benign'
    : result && ['high', 'critical'].includes(result.severity) ? ' pipeline-intrusion' : ''
  return (
    <Panel title="Detection pipeline" subtitle="From traffic records to a final-flow prediction"
      action={<StatusBadge dot tone={demo.enabled ? 'info' : 'neutral'}>{demo.enabled ? `DEMO · ${demo.mode.toUpperCase()}` : 'Idle'}</StatusBadge>} className={`pipeline-panel${active ? ' pipeline-running' : ''}`}>
      <ol className="pipeline-steps" aria-label="Intrusion detection pipeline">
        {steps.map(({ title, detail, icon: Icon }, index) => (
          <li key={title} aria-current={active && index === stage ? 'step' : undefined} className={`pipeline-step${active && index === stage ? ` pipeline-current${[1, 3].includes(index) ? ' pipeline-amber' : index === 5 ? predictionTone : ''}` : active && index < stage ? ' pipeline-complete' : ''}`}>
            <div className="pipeline-node"><span className="pipeline-number">0{index + 1}</span><Icon size={21} strokeWidth={1.5} aria-hidden="true" /></div>
            <h3>{title}</h3><p>{active && index === 5 && stage === 5 && result?.prediction ? `DEMO · ${result.prediction}` : detail}</p>
            {index < steps.length - 1 && <span className="pipeline-connection" aria-hidden="true"><span className="connection-line" /><ArrowRight size={12} />{active && <span className="pipeline-particle" style={{ animationDelay: `${index * 0.2}s` }} />}</span>}
          </li>
        ))}
      </ol>
      <div className="panel-footnote"><span className="tiny-dot" /><span>T=5 · Five consecutive clean flow records · Prediction target: fifth / final flow{demo.enabled && ' · Animation is a frontend demo'}</span></div>
    </Panel>
  )
}
