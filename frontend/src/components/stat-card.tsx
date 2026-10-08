import { m, useReducedMotion } from 'motion/react'
import { ArrowDownRight, ArrowUpRight, Minus, type LucideIcon } from 'lucide-react'
import { Card } from '@/components/ui/card'
import NumberFlow from '@number-flow/react'

export type StatCardProps = {
  title: string
  value: number | string | { numerator: number; denominator: number }
  format?: 'count' | 'decimal' | 'percentage'
  decimalPlaces?: number
  suffix?: string
  state?: 'good' | 'bad' | 'neutral'
  icon?: LucideIcon
  description?: string
  trend?: { direction: 'up' | 'down' | 'flat'; label: string }
}

export function StatCard({ title, value, format = 'count', decimalPlaces,
  suffix, state = 'neutral', icon: Icon, description, trend }: StatCardProps) {
  const reducedMotion = useReducedMotion()
  const TrendIcon = trend?.direction === 'up' ? ArrowUpRight : trend?.direction === 'down' ? ArrowDownRight : Minus
  const places = decimalPlaces ?? (format === 'count' ? 0 : 1)
  const numericValue = typeof value === 'object' ? value.numerator : value
  const displayed = typeof numericValue === 'number'
    ? <NumberFlow value={numericValue} locales="en-US" format={{ minimumFractionDigits: places, maximumFractionDigits: places }} animated={!reducedMotion} trend={0} />
    : numericValue
  // Percentage values are supplied in percentage units (e.g. 78.2, not .782).
  // Trend direction never chooses its own semantic color; the caller sets state.
  const unit = suffix ?? (format === 'percentage' ? '%' : undefined)

  return (
    <Card className={`stat-card stat-${state}`}>
      <div className="stat-heading"><h2>{title}</h2>{Icon && <Icon size={17} aria-hidden="true" />}</div>
      <m.div className="stat-value" initial={reducedMotion ? false : { opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
        {displayed}
        {typeof value === 'object' && <span className="stat-denominator"> / {value.denominator}</span>}
        {unit && <span className="stat-unit">{unit}</span>}
      </m.div>
      <div className="stat-description">
        {trend ? <><TrendIcon size={13} aria-hidden="true" /><span>{trend.label}</span></> : <span>{description}</span>}
      </div>
    </Card>
  )
}
