import { Crosshair, Radar } from 'lucide-react'
import { cn } from '@/lib/utils'

export function ProcessingScanner({ variant = 'radar', active = false, label, description, compact = false }: {
  variant?: 'radar' | 'crosshair'; active?: boolean; label: string; description: string; compact?: boolean
}) {
  const Icon = variant === 'radar' ? Radar : Crosshair
  return <div className={cn('processing-scanner', `scanner-${variant}`, active && 'scanner-active', compact && 'scanner-compact')}>
    <div className="scanner-disc" aria-hidden="true"><span className="scanner-grid" /><span className="scanner-sweep" /><span className="scanner-orbit" /><Icon strokeWidth={1} /><span className="scanner-center" /></div>
    <div className="scanner-copy"><h3>{label}</h3><p>{description}</p></div>
  </div>
}
