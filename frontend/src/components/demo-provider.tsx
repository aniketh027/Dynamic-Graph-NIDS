import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { DemoContext, type DemoSession, type Notice } from '@/lib/demo-context'
import { makeDemoEvents, syntheticFlowRate, type Severity, type TrafficEvent, type TrafficPoint } from '@/lib/demo-data'

type StreamState = { mode: DemoSession['mode']; epoch: number; step: number; points: TrafficPoint[]; flowCount: number }
const emptyStream: StreamState = { mode: 'idle', epoch: 0, step: 0, points: [], flowCount: 0 }

export function DemoProvider({ children }: { children: ReactNode }) {
  const [stream, setStream] = useState<StreamState>(emptyStream)
  const [selectedEvent, inspect] = useState<TrafficEvent | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const notify = useCallback((severity: Severity, title: string, description: string, demo = false) => {
    setNotice({ id: Date.now(), severity, title, description, demo })
  }, [])
  const dismissNotice = useCallback(() => setNotice(null), [])

  useEffect(() => {
    if (stream.mode !== 'running') return
    const timer = window.setInterval(() => {
      setStream(previous => {
        if (previous.mode !== 'running') return previous
        const step = previous.step + 1
        const value = syntheticFlowRate(step + 30)
        return { ...previous, step, flowCount: previous.flowCount + value,
          points: [...previous.points.slice(-179), { time: Date.now() / 1000, value }] }
      })
    }, 1000)
    return () => window.clearInterval(timer)
  }, [stream.mode])

  // A paused session retains its dataset. Stop also retains it for inspection;
  // reset clears points, events, the drawer and any demo notification together.
  const start = useCallback(() => {
    const epoch = Math.floor(Date.now() / 1000)
    const points = Array.from({ length: 31 }, (_, step) => ({ time: epoch - 30 + step, value: syntheticFlowRate(step) }))
    setStream({ mode: 'running', epoch, step: 0, points, flowCount: 0 })
    inspect(null)
    notify('info', 'Synthetic stream started', 'Frontend demo only. No file replay, model inference, or live capture is running.', true)
  }, [notify])
  const pause = useCallback(() => setStream(previous => ({ ...previous, mode: previous.mode === 'running' ? 'paused' : previous.mode })), [])
  const resume = useCallback(() => setStream(previous => ({ ...previous, mode: previous.mode === 'paused' ? 'running' : previous.mode })), [])
  const stop = useCallback(() => setStream(previous => ({ ...previous, mode: previous.mode === 'idle' ? 'idle' : 'stopped' })), [])
  const reset = useCallback(() => { setStream(emptyStream); inspect(null); setNotice(null) }, [])
  const events = useMemo(() => stream.mode === 'idle' ? [] : makeDemoEvents(stream.epoch, stream.step), [stream.mode, stream.epoch, stream.step])
  const value = useMemo<DemoSession>(() => ({ ...stream, enabled: stream.mode !== 'idle', events, start, pause, resume, stop, reset,
    selectedEvent, inspect, notice, notify, dismissNotice }), [stream, events, start, pause, resume, stop, reset, selectedEvent, notice, notify, dismissNotice])
  return <DemoContext.Provider value={value}>{children}</DemoContext.Provider>
}
