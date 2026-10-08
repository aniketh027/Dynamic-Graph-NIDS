import { useEffect, useState, type ReactNode } from 'react'
import { apiBaseUrl, checkBackend, retryDelay, type Connection } from '@/lib/api/client'
import { BackendContext } from '@/lib/api/context'

export function BackendProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>({ state: 'checking', status: null })
  useEffect(() => {
    let active = true
    let timer: ReturnType<typeof setTimeout> | undefined
    let controller: AbortController | undefined
    let consecutiveFailures = 0
    async function poll() {
      const request = new AbortController()
      controller = request
      const timeout = setTimeout(() => request.abort(new DOMException('Backend status deadline exceeded', 'TimeoutError')), 3000)
      let result: Connection
      try {
        result = await checkBackend(apiBaseUrl(import.meta.env.VITE_API_BASE_URL), request.signal)
      } catch {
        // URL configuration is validated before fetch. Network/body errors are typed by the client.
        result = { state: 'offline', status: null, reason: 'configuration' }
      } finally {
        clearTimeout(timeout)
      }
      if (!active) return
      setConnection(result)
      consecutiveFailures = result.state === 'offline' ? Math.min(consecutiveFailures + 1, 4) : 0
      timer = setTimeout(poll, retryDelay(result, consecutiveFailures))
    }
    void poll()
    return () => { active = false; clearTimeout(timer); controller?.abort() }
  }, [])
  return <BackendContext.Provider value={connection}>{children}</BackendContext.Provider>
}
