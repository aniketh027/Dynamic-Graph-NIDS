import { useEffect, useState, useSyncExternalStore, type ReactNode } from 'react'
import { ReplaySession } from '@/lib/replay-session'
import { ReplayContext } from '@/lib/replay-context'
export function ReplayProvider({ children }: { children: ReactNode }) {
  const [session] = useState(() => new ReplaySession(import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'))
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot)
  useEffect(() => () => session.dispose(), [session])
  return <ReplayContext.Provider value={{ state, session }}>{children}</ReplayContext.Provider>
}
