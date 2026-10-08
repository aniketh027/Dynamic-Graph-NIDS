import { lazy, Suspense, useState } from 'react'
import { Activity, ArrowUpRight, ShieldCheck, Unplug } from 'lucide-react'
import { Link } from 'react-router'
import { Panel } from '@/components/panel'
import { EmptyState } from '@/components/empty-state'
import { StatusBadge } from '@/components/status-badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useDemo } from '@/lib/demo-context'

const StreamingChart = lazy(() => import('@/components/traffic-chart'))

export function TrafficActivityPanel() {
  const demo = useDemo()
  const [seconds, setSeconds] = useState('30')
  return <Panel title="Traffic activity" subtitle={demo.enabled ? 'Synthetic flow records / second · Not measured traffic' : 'Flow volume across an analysis session'}
    action={demo.enabled ? <Select value={seconds} onValueChange={setSeconds}><SelectTrigger aria-label="Traffic time window" className="compact-select"><SelectValue /></SelectTrigger><SelectContent>{['30', '60', '120'].map(value => <SelectItem value={value} key={value}>{value}s window</SelectItem>)}</SelectContent></Select> : <StatusBadge dot>No active session</StatusBadge>} className="traffic-panel">
    {demo.enabled ? <Suspense fallback={<EmptyState compact icon={Activity} title="Preparing demo chart" description="Synthetic frontend visualization." />}><StreamingChart seconds={Number(seconds)} /></Suspense> : <div className="traffic-chart-shell"><div className="chart-grid" aria-hidden="true" /><span className="axis-label axis-y" aria-hidden="true">FLOW RECORDS</span>
      <EmptyState icon={Activity} title="No active analysis session" description="Your traffic timeline will appear here once analysis begins." action={<Link to="/analyze-traffic" className="text-link">Go to Analyze Traffic<ArrowUpRight size={13} aria-hidden="true" /></Link>} /><div className="chart-axis" aria-hidden="true"><span>SESSION START</span><span>TIME →</span></div></div>}
    <div className="activity-footer"><span>{demo.enabled ? <><span className="legend-dot traffic-blue" />DEMO · {demo.mode === 'running' ? 'Synthetic stream' : 'Frozen for inspection'}</> : <><Unplug size={12} aria-hidden="true" />Waiting for traffic data</>}</span><span>{demo.enabled ? 'Hover to inspect · IST' : <>Throughput <strong>—</strong><span className="activity-divider" />Latency <strong>—</strong></>}</span></div>
  </Panel>
}

export function DetectionActivityPanel() {
  const demo = useDemo()
  const benign = demo.events.filter(event => event.severity === 'safe').length
  const attack = demo.events.filter(event => ['high', 'critical'].includes(event.severity)).length
  const total = benign + attack
  return <Panel title="Detection activity" subtitle={demo.enabled ? 'Synthetic event fixtures · Not model outputs' : 'Benign and intrusion events'} action={demo.enabled ? <StatusBadge tone="info">DEMO</StatusBadge> : <ShieldCheck size={17} className="muted-icon" aria-hidden="true" />} className="detection-panel">
    <div className="detection-empty"><div className="empty-donut" role="img" aria-label={demo.enabled ? `${benign} benign and ${attack} intrusion demo events.` : 'Detection distribution unavailable. No predictions have been generated.'}>
      <svg viewBox="0 0 140 140" aria-hidden="true"><circle cx="70" cy="70" r="55" fill="none" stroke="currentColor" strokeWidth="10" />{demo.enabled ? <><circle cx="70" cy="70" r="55" fill="none" stroke="var(--green)" strokeWidth="10" strokeDasharray={`${benign / total * 345.6} 345.6`} /><circle cx="70" cy="70" r="55" fill="none" stroke="var(--red)" strokeWidth="10" strokeDasharray={`${attack / total * 345.6} 345.6`} strokeDashoffset={-benign / total * 345.6} /></> : <circle cx="70" cy="70" r="55" fill="none" stroke="var(--border)" strokeWidth="10" strokeDasharray="2 17" />}</svg>
      <div><strong>{demo.enabled ? total : '—'}</strong><span>{demo.enabled ? 'DEMO EVENTS' : 'NO PREDICTIONS'}</span></div></div><div className="detection-legend"><div><span><i className="legend-dot benign" />Benign</span><strong>{demo.enabled ? benign : '—'}</strong></div><div><span><i className="legend-dot attack" />Intrusion</span><strong>{demo.enabled ? attack : '—'}</strong></div></div></div>
    <div className="activity-footer"><span>{demo.enabled ? 'UI fixtures only' : 'Waiting for analysis'}</span><span>{demo.enabled ? `${total} demo events` : '0 detections'}</span></div>
  </Panel>
}
