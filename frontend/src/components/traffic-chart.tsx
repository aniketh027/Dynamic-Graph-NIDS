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
  const time = point.date instanceof Date && Number.isFinite(point.date.getTime())
    ? `${eventTime(point.date.getTime() / 1000)} IST`
    : 'Time unavailable'
  const value = typeof point.value === 'number' && Number.isFinite(point.value) ? Math.round(point.value) : null
  return <div className="traffic-tooltip"><span>DEMO · SYNTHETIC</span>
    <strong>{value === null ? 'Flow rate unavailable' : `${value} flow records / s`}</strong>
    <small>{time}</small></div>
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
