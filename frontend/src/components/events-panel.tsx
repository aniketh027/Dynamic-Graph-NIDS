import { useMemo, useState } from 'react'
import { ArrowUpRight, Logs, Search } from 'lucide-react'
import { Link } from 'react-router'
import { Panel } from '@/components/panel'
import { EmptyState } from '@/components/empty-state'
import { StatusBadge } from '@/components/status-badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useDemo } from '@/lib/demo-context'
import { eventTime, severityTone } from '@/lib/demo-data'

export function EventsPanel({ full = false }: { full?: boolean }) {
  const demo = useDemo()
  const [search, setSearch] = useState('')
  const [severity, setSeverity] = useState('all')
  const [kind, setKind] = useState('all')
  const filtered = useMemo(() => demo.events.filter(event => (severity === 'all' || severity === event.severity)
    && (kind === 'all' || (kind === 'benign' ? event.severity === 'safe' : ['high', 'critical'].includes(event.severity)))
    && `${event.title} ${event.component} ${event.id} ${event.description}`.toLowerCase().includes(search.toLowerCase())), [demo.events, severity, kind, search])
  const rows = full ? filtered : filtered.slice(0, 5)
  return <Panel title={full ? 'Event investigation' : 'Recent events'} subtitle={demo.enabled ? 'DEMO · Deterministic synthetic events' : 'Pipeline events and intrusion activity'}
    action={full ? <StatusBadge tone={demo.enabled ? 'info' : 'neutral'}>{demo.enabled ? 'DEMO DATA' : 'No session'}</StatusBadge> : <Link className="text-link" to="/intrusion-alerts">View all<ArrowUpRight size={13} aria-hidden="true" /></Link>} className="events-panel">
    <div className="event-controls"><label className="event-search"><Search size={13} aria-hidden="true" /><input value={search} onChange={event => setSearch(event.target.value)} placeholder="Search events…" aria-label="Search events" /></label>
      <Select value={severity} onValueChange={setSeverity}><SelectTrigger className="compact-select severity-select" aria-label="Filter event severity"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All severity</SelectItem>{['info', 'safe', 'warning', 'high', 'critical', 'error'].map(value => <SelectItem key={value} value={value}>{value.toUpperCase()}</SelectItem>)}</SelectContent></Select>
      {full && <Tabs value={kind} onValueChange={setKind}><TabsList><TabsTrigger value="all">All</TabsTrigger><TabsTrigger value="benign">Benign</TabsTrigger><TabsTrigger value="attack">Attack</TabsTrigger></TabsList></Tabs>}
    </div>
    <div className="events-table-wrap"><table className="events-table"><caption className="sr-only">{demo.enabled ? 'Demo event log. All values are synthetic.' : 'Event log. No analysis is connected.'}</caption><thead><tr>{['Severity', 'Time', 'Component', 'Event', 'Model score', 'Latency'].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead><tbody>
      {rows.map(event => <tr key={event.id} className="event-row" onClick={() => demo.inspect(event)}><td><StatusBadge tone={severityTone(event.severity)}>{event.severity.toUpperCase()}</StatusBadge></td><td><time dateTime={new Date(event.timestamp * 1000).toISOString()}>{eventTime(event.timestamp)}</time><small>IST · DEMO</small></td><td>{event.component}</td><td><Button variant="ghost" className="event-open" onClick={() => demo.inspect(event)} aria-label={`Inspect ${event.id}`}>{event.title}<ArrowUpRight size={12} className="row-inspect-icon" aria-hidden="true" /></Button><small>{event.id} · DEMO</small></td><td>{event.modelScore?.toFixed(3) ?? '—'}</td><td>{event.latency ? `${event.latency.toFixed(1)} ms` : '—'}</td></tr>)}
    </tbody></table></div>
    {rows.length === 0 && <EmptyState compact icon={Logs} title={demo.enabled ? 'No matching demo events' : 'Your event stream starts here'} description={demo.enabled ? 'Adjust your search or filters to inspect the synthetic event log.' : 'No events yet. Analysis events will appear when traffic data is processed.'} />}
    <div className="events-footer"><span>{filtered.length} {demo.enabled ? 'demo events' : 'events'}{!full && filtered.length > 5 && ' · Latest 5 shown'}</span><span className="mono">{demo.enabled ? 'SYNTHETIC SCORES / TIMINGS' : 'SESSION —'}</span></div>
  </Panel>
}
