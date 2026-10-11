import { decodeAnalysis, decodeValidation, ReplayError } from './replay.ts'
import { matchesCsvDescriptor } from '../csv-compatibility.ts'
/** Central application API client; status polling remains independent of uploads. */
export type AdapterState = 'uninitialized' | 'loading' | 'ready' | 'error'
export type OfflineReason = 'network' | 'timeout' | 'http' | 'malformed' | 'cancelled' | 'configuration'
export interface BackendStatus {
  backend_connected: true
  model_adapter_state: AdapterState
  model_loaded: boolean
  inference_ready: boolean
  live_capture_supported: false
  file_replay_supported: boolean
  mode: 'offline_application_preview'
  error: { code: string; message: string } | null
}
export type Connection =
  | { state: 'checking'; status: null }
  | { state: 'offline'; status: null; reason: OfflineReason }
  | { state: 'connected'; status: BackendStatus }

const adapterStates = new Set(['uninitialized', 'loading', 'ready', 'error'])

function validStatus(value: unknown): value is BackendStatus {
  if (!value || typeof value !== 'object') return false
  const data = value as Record<string, unknown>
  const error = data.error
  return data.backend_connected === true && typeof data.model_adapter_state === 'string'
    && adapterStates.has(data.model_adapter_state) && typeof data.model_loaded === 'boolean'
    && typeof data.inference_ready === 'boolean' && data.live_capture_supported === false
    && typeof data.file_replay_supported === 'boolean' && data.mode === 'offline_application_preview'
    && (data.model_adapter_state === 'ready') === data.model_loaded
    && (error === null || (typeof error === 'object' && error !== null
      && typeof (error as Record<string, unknown>).code === 'string'
      && typeof (error as Record<string, unknown>).message === 'string'))
}

export function apiBaseUrl(configured?: string): string {
  const value = configured?.trim() || 'http://127.0.0.1:8000'
  const parsed = new URL(value)
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password
    || parsed.search || parsed.hash) throw new Error('Invalid application API URL')
  return value.replace(/\/+$/, '')
}

export async function checkBackend(baseUrl: string, signal?: AbortSignal,
  fetcher: typeof fetch = fetch): Promise<Connection> {
  const offline = (reason: OfflineReason): Connection => ({ state: 'offline', status: null, reason })
  const failureReason = (error: unknown, fallback: OfflineReason): OfflineReason => {
    const cause = signal?.aborted ? signal.reason : error
    if (cause instanceof Error && cause.name === 'TimeoutError') return 'timeout'
    if (signal?.aborted || (cause instanceof Error && cause.name === 'AbortError')) return 'cancelled'
    return fallback
  }
  if (signal?.aborted) return offline(failureReason(signal.reason, 'cancelled'))
  // A failed Promise.all does not cancel its sibling. Own the request lifetime
  // so neither an outstanding fetch nor a response body survives this poll.
  const request = new AbortController()
  const relayAbort = () => request.abort(signal?.reason)
  signal?.addEventListener('abort', relayAbort, { once: true })
  let parsing = false
  try {
    const options = { signal: request.signal, cache: 'no-store' as const, headers: { Accept: 'application/json' } }
    const [health, status] = await Promise.all([
      fetcher(`${baseUrl}/api/v1/health`, options), fetcher(`${baseUrl}/api/v1/status`, options),
    ])
    if (!health.ok || !status.ok) return offline('http')
    parsing = true
    const [healthData, statusData] = await Promise.all([health.json(), status.json()])
    if (healthData?.status !== 'ok' || healthData?.service !== 'graphnids-backend' || !validStatus(statusData)) {
      return offline('malformed')
    }
    if (signal?.aborted) return offline(failureReason(signal.reason, 'cancelled'))
    return { state: 'connected', status: statusData }
  } catch (error) {
    return offline(failureReason(error, parsing && error instanceof SyntaxError ? 'malformed' : 'network'))
  } finally {
    signal?.removeEventListener('abort', relayAbort)
    request.abort()
  }
}

export function retryDelay(connection: Connection, consecutiveFailures: number): number {
  if (connection.state !== 'offline') return 10000
  return Math.min(10000 * 2 ** Math.max(0, Math.min(consecutiveFailures - 1, 3)), 60000)
}

export function connectionLabel(connection: Connection): string {
  return connection.state === 'connected' ? 'Backend connected'
    : connection.state === 'checking' ? 'Checking backend' : 'Backend disconnected'
}

export function adapterLabel(connection: Connection): string {
  if (connection.state !== 'connected') return 'Model adapter unavailable'
  switch (connection.status.model_adapter_state) {
    case 'ready': return 'Model adapter ready'
    case 'loading': return 'Model adapter loading'
    case 'error': return 'Model adapter error'
    default: return 'Model adapter not loaded'
  }
}

export const VALIDATION_TIMEOUT_MS = 30000
export const ANALYSIS_TIMEOUT_MS = 10 * 60 * 1000
type ReplayOptions = { signal?: AbortSignal; fetcher?: typeof fetch; timeoutMs?: number }
async function upload<T>(baseUrl: string, file: File, operation: 'validate' | 'analyze',
  decode: (value: unknown) => T, options: ReplayOptions): Promise<T> {
  const request = new AbortController()
  const relay = () => request.abort(options.signal?.reason)
  options.signal?.addEventListener('abort', relay, { once: true })
  if (options.signal?.aborted) relay()
  const timer = setTimeout(() => request.abort(new DOMException('Request deadline exceeded', 'TimeoutError')),
    options.timeoutMs ?? (operation === 'analyze' ? ANALYSIS_TIMEOUT_MS : VALIDATION_TIMEOUT_MS))
  try {
    if (request.signal.aborted) throw request.signal.reason
    let base: string
    try { base = apiBaseUrl(baseUrl) } catch { throw new ReplayError('CONFIGURATION') }
    const response = await (options.fetcher ?? fetch)(`${base}/api/v1/replay/${operation}`, {
      method: 'POST', body: file, signal: request.signal, cache: 'no-store',
      headers: { Accept: 'application/json', 'Content-Type': 'text/csv', 'X-Filename': file.name },
    })
    const data = await response.json()
    if (request.signal.aborted) throw request.signal.reason
    if (!response.ok) throw new ReplayError(typeof data?.error?.code === 'string' ? data.error.code : 'INTERNAL_ERROR')
    try { return decode(data) } catch { throw new ReplayError('MALFORMED') }
  } catch (error) {
    if (request.signal.aborted) throw new ReplayError(request.signal.reason?.name === 'TimeoutError' ? 'TIMEOUT' : 'CANCELLED')
    if (error instanceof ReplayError) throw error
    throw new ReplayError(error instanceof SyntaxError ? 'MALFORMED' : 'NETWORK')
  } finally {
    clearTimeout(timer)
    options.signal?.removeEventListener('abort', relay)
    request.abort()
  }
}
export const validateReplay = (base: string, file: File, options: ReplayOptions = {}) => upload(base, file, 'validate', decodeValidation, options)
export const analyzeReplay = (base: string, file: File, options: ReplayOptions = {}) => upload(base, file, 'analyze', decodeAnalysis, options)

/** Read-only schema check; no filename, content or provenance is transmitted. */
export async function checkCsvDescriptor(baseUrl: string, options: ReplayOptions = {}): Promise<void> {
  const request = new AbortController()
  const relay = () => request.abort(options.signal?.reason)
  options.signal?.addEventListener('abort', relay, { once: true })
  if (options.signal?.aborted) relay()
  const timer = setTimeout(() => request.abort(new DOMException('Schema deadline exceeded', 'TimeoutError')),
    options.timeoutMs ?? VALIDATION_TIMEOUT_MS)
  try {
    if (request.signal.aborted) throw request.signal.reason
    let base: string
    try { base = apiBaseUrl(baseUrl) } catch { throw new ReplayError('CONFIGURATION') }
    const response = await (options.fetcher ?? fetch)(`${base}/api/v1/schema`, {
      signal: request.signal, cache: 'no-store', headers: { Accept: 'application/json' },
    })
    const value = await response.json()
    if (request.signal.aborted) throw request.signal.reason
    if (!response.ok || !matchesCsvDescriptor(value?.csv_compatibility)) throw new ReplayError('CSV_DESCRIPTOR_MISMATCH')
  } catch (cause) {
    if (request.signal.aborted) throw new ReplayError(request.signal.reason?.name === 'TimeoutError' ? 'TIMEOUT' : 'CANCELLED')
    if (cause instanceof ReplayError) throw cause
    throw new ReplayError(cause instanceof SyntaxError ? 'CSV_DESCRIPTOR_MISMATCH' : 'NETWORK')
  } finally {
    clearTimeout(timer); options.signal?.removeEventListener('abort', relay); request.abort()
  }
}
