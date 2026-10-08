import { createContext, useContext } from 'react'
import type { ReplaySession, ReplayState } from './replay-session'
export const ReplayContext = createContext<{ state: ReplayState; session: ReplaySession; presentation?: 'mock' } | null>(null)
export function useReplay() {
  const value = useContext(ReplayContext)
  if (!value) throw new Error('ReplayProvider is required')
  return value
}

// Only the development QA provider opts in. Never infer authenticity from a filename or payload.
export function useReplayPresentation() {
  const { presentation } = useReplay()
  const mock = presentation === 'mock'
  return {
    mock,
    badge: mock ? 'MOCKED UI VALIDATION' : 'REAL ANALYSIS',
    qualifier: mock ? 'NO MODEL OUTPUT' : 'FROZEN MODEL',
    kind: mock ? 'Mock' : 'Real',
    predictions: mock ? 'Mock intrusion predictions' : 'Real intrusion predictions',
    description: mock ? 'Mocked UI validation fixture. No model output. Indices are zero-based.' : 'Frozen model prediction from uploaded CSV replay. Indices are zero-based.',
  }
}
