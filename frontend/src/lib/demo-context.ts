import { createContext, useContext } from 'react'
import type { Severity, TrafficEvent, TrafficPoint } from '@/lib/demo-data'

export type Notice = { id: number; severity: Severity; title: string; description: string; demo: boolean }
export type DemoSession = {
  mode: 'idle' | 'running' | 'paused' | 'stopped'
  enabled: boolean
  step: number
  points: TrafficPoint[]
  events: TrafficEvent[]
  flowCount: number
  start: () => void
  pause: () => void
  resume: () => void
  stop: () => void
  reset: () => void
  selectedEvent: TrafficEvent | null
  inspect: (event: TrafficEvent | null) => void
  notice: Notice | null
  dismissNotice: () => void
  notify: (severity: Severity, title: string, description: string, demo?: boolean) => void
}

export const DemoContext = createContext<DemoSession | null>(null)
export function useDemo() {
  const value = useContext(DemoContext)
  if (!value) throw new Error('Demo controls must be inside DemoProvider')
  return value
}
