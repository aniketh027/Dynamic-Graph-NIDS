import { FlaskConical, Pause, Play, RotateCcw, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/status-badge'
import { useDemo } from '@/lib/demo-context'
import { useReplay } from '@/lib/replay-context'

export function DemoControls({ compact = false }: { compact?: boolean }) {
  const demo = useDemo()
  const { state, session } = useReplay()
  const start = () => { session.select(null); demo.start() }
  if (state.phase === 'analyzing' || state.phase === 'validating') return null
  return <div className={`demo-controls${compact ? ' demo-compact' : ''}`}>
    {!demo.enabled ? <><span className="demo-intro"><FlaskConical size={14} aria-hidden="true" />Explore the interface with synthetic traffic.</span>
      <Button variant="outline" size="sm" onClick={start}><Play size={12} />{state.result ? 'Switch to demo (clear result)' : 'Start demo'}</Button></>
      : <><StatusBadge tone="info" dot>DEMO · {demo.mode === 'running' ? 'SIMULATED' : demo.mode.toUpperCase()}</StatusBadge>
        <span className="demo-explanation">Synthetic UI values · Frontend demo</span>
        <div className="demo-actions">{demo.mode === 'running' ? <Button variant="ghost" size="sm" onClick={demo.pause}><Pause />Pause</Button>
          : demo.mode === 'paused' ? <Button variant="ghost" size="sm" onClick={demo.resume}><Play />Resume</Button>
          : <Button variant="ghost" size="sm" onClick={start}><Play />Restart</Button>}
          <Button variant="ghost" size="sm" onClick={demo.stop} disabled={demo.mode === 'stopped'}><Square />Stop</Button>
          <Button variant="ghost" size="sm" onClick={demo.reset}><RotateCcw />Reset</Button></div></>}
  </div>
}
