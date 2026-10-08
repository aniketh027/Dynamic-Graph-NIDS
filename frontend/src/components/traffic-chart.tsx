import { useReducedMotion } from 'motion/react'
import { LiveLineChart } from '@/components/ui/charts/live-line-chart'
import { LiveLine } from '@/components/ui/charts/live-line'
import { LiveXAxis } from '@/components/ui/charts/live-x-axis'
import { LiveYAxis } from '@/components/ui/charts/live-y-axis'
import { ChartTooltip } from '@/components/ui/charts/tooltip/chart-tooltip'
import { useDemo } from '@/lib/demo-context'
import { eventTime } from '@/lib/demo-data'

function TrafficTooltip({ point }: { point: Record<string, unknown> }) {
  // The live registry accepts Unix seconds, then exposes tooltip points as Date.
  const seconds = (point.date instanceof Date ? point.date.getTime() : Number(point.date)) / 1000
  const value = Number(point.value)
  return <div className="traffic-tooltip"><span>DEMO · SYNTHETIC</span>
    <strong>{Number.isFinite(value) ? Math.round(value) : '—'} flow records / s</strong>
    <small>{Number.isFinite(seconds) ? `${eventTime(seconds)} IST` : 'Time unavailable'}</small></div>
}

export default function StreamingTrafficChart({ seconds }: { seconds: number }) {
  const demo = useDemo()
  const reducedMotion = useReducedMotion()
  const value = demo.points.at(-1)?.value ?? 0
  return <div className="stream-chart" role="img" aria-label={`Synthetic traffic chart. Latest value ${value} flow records per second. ${seconds}-second window. ${demo.mode}.`}>
    <LiveLineChart data={demo.points} value={value} window={seconds} paused={demo.mode !== 'running'} nowOffsetUnits={0.5}
      lerpSpeed={reducedMotion ? 1 : 0.08} margin={{ top: 26, left: 43, right: 54, bottom: 32 }} className="bklit-traffic-chart" style={{ height: '100%', touchAction: 'pan-y' }}>
      <LiveLine dataKey="value" stroke="var(--blue)" strokeWidth={2} pulse={!reducedMotion && demo.mode === 'running'} dotSize={3} badge formatValue={number => `${Math.round(number)}`} />
      <ChartTooltip showDatePill={false} damping={reducedMotion ? 0 : 20} content={({ point }) => <TrafficTooltip point={point} />} />
      <LiveXAxis numTicks={3} formatTime={milliseconds => eventTime(milliseconds / 1000)} />
      <LiveYAxis position="left" allowDecimals={false} formatValue={number => `${Math.round(number)}`} />
    </LiveLineChart>
  </div>
}
