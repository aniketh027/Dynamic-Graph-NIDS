import { useState } from 'react'
import { CircleCheck, Info, ShieldAlert, TriangleAlert, X, CircleX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/status-badge'
import { useDemo } from '@/lib/demo-context'
import { eventTime, severityTone, type Severity, type TrafficEvent } from '@/lib/demo-data'

const icons = { info: Info, safe: CircleCheck, warning: TriangleAlert, high: ShieldAlert, critical: ShieldAlert, error: CircleX }
export function AlertCard({ severity, title, description, demo = false, timestamp, onDismiss, onInspect }: {
  severity: Severity; title: string; description: string; demo?: boolean; timestamp?: number; onDismiss?: () => void; onInspect?: () => void
}) {
  const Icon = icons[severity]
  return <div className={`alert-card alert-${severity}`}>
    <Icon className="alert-symbol" size={18} aria-hidden="true" /><div className="alert-body">
      <div className="alert-meta"><StatusBadge tone={severityTone(severity)}>{severity.toUpperCase()}</StatusBadge>{demo && <span className="demo-text">DEMO EVENT</span>}{timestamp && <time dateTime={new Date(timestamp * 1000).toISOString()}>{eventTime(timestamp)} IST</time>}</div>
      <h3>{title}</h3><p>{description}</p>{onInspect && <Button variant="link" size="sm" onClick={onInspect}>Inspect event</Button>}
    </div>{onDismiss && <Button variant="ghost" size="icon-sm" aria-label="Dismiss notification" onClick={onDismiss}><X /></Button>}
  </div>
}

export function DemoAlert({ event, dismissible = false }: { event: TrafficEvent; dismissible?: boolean }) {
  const { inspect } = useDemo()
  const [visible, setVisible] = useState(true)
  return visible ? <AlertCard {...event} demo timestamp={event.timestamp} onInspect={() => inspect(event)} onDismiss={dismissible ? () => setVisible(false) : undefined} /> : null
}

// Toasts announce a user-triggered UI change once, never every stream update.
export function ToastViewport() {
  const { notice, dismissNotice } = useDemo()
  return <div className="toast-viewport" role="region" aria-label="Notifications"><div aria-live="polite" aria-atomic="true">
    {notice && <AlertCard {...notice} onDismiss={dismissNotice} />}
  </div></div>
}
