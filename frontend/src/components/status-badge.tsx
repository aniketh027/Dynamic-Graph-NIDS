import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

type StatusBadgeProps = {
  children: ReactNode
  tone?: 'neutral' | 'info' | 'good' | 'warning' | 'bad'
  dot?: boolean
  className?: string
}

export function StatusBadge({ children, tone = 'neutral', dot = false, className }: StatusBadgeProps) {
  return (
    <span className={cn('status-badge', `status-${tone}`, className)}>
      {dot && <span className="status-dot" aria-hidden="true" />}
      {children}
    </span>
  )
}
