export type Severity = 'info' | 'safe' | 'warning' | 'high' | 'critical' | 'error'
export type TrafficEvent = {
  id: string
  severity: Severity
  timestamp: number
  component: string
  title: string
  description: string
  prediction?: string
  modelScore?: number
  latency?: number
  source: 'demo'
}
export type TrafficPoint = { time: number; value: number }

// Two seconds per presentation stage; the existing demo clock remains authoritative.
export function demoPipelineStage(step: number) {
  return Math.floor(step / 2) % 6
}

// Deterministic presentation fixtures. These never read a file or model output.
export function syntheticFlowRate(step: number) {
  return Math.round(38 + 14 * Math.sin(step * 0.31) + 7 * Math.cos(step * 0.73) + (step % 17 === 0 ? 18 : 0))
}

const fixtures: Omit<TrafficEvent, 'id' | 'timestamp' | 'source'>[] = [
  { severity: 'info', component: 'preprocessor', title: 'Traffic schema check', description: 'Simulated file-validation step. No file contents were read.' },
  { severity: 'info', component: 'flow-engine', title: '5-flow window assembled', description: 'Five synthetic clean flow records; the fifth record is the prediction target.' },
  { severity: 'warning', component: 'graph-builder', title: 'Graph construction preview', description: 'A visual preview of the 70-node / 69-edge inference graph. No graph was constructed.' },
  { severity: 'safe', component: 'HTGNN', title: 'Benign prediction preview', description: 'Synthetic benign event for interface inspection.', prediction: 'Benign (simulated)', modelScore: 0.18, latency: 2.4 },
  { severity: 'high', component: 'HTGNN', title: 'Intrusion prediction preview', description: 'Synthetic intrusion event for interface inspection.', prediction: 'Intrusion (simulated)', modelScore: 0.86, latency: 2.8 },
  { severity: 'critical', component: 'alert-engine', title: 'DDoS event preview', description: 'Simulated DDoS alert. This is not a recorded intrusion or research result.', prediction: 'DDoS (simulated)', modelScore: 0.93, latency: 3.1 },
]

export function makeDemoEvents(epoch: number, step: number): TrafficEvent[] {
  const count = 6 + Math.floor(step / 5)
  return Array.from({ length: Math.min(count, 24) }, (_, index) => {
    const ordinal = count - index - 1
    return { ...fixtures[ordinal % fixtures.length], id: `DEMO-W${String(ordinal + 1).padStart(4, '0')}`,
      timestamp: epoch - 25 + ordinal * 5, source: 'demo' }
  })
}

export function eventTime(seconds: number) {
  return new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Kolkata' }).format(new Date(seconds * 1000))
}

export function severityTone(severity: Severity) {
  return severity === 'safe' ? 'good' : severity === 'warning' ? 'warning'
    : ['high', 'critical', 'error'].includes(severity) ? 'bad' : 'info'
}
