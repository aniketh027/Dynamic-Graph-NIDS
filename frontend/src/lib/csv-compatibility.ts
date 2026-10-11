import { CSV_SCHEMA } from './csv-schema.generated.ts'

export type SourceProfile = 'canonical_78' | 'standard_80' | 'extended_84'
export type PreparationProfile = Exclude<SourceProfile, 'canonical_78'>
export interface PreparationRecord {
  source_profile: PreparationProfile; source_sha256: string; prepared_sha256: string
  source_size_bytes: number; prepared_size_bytes: number; feature_schema_identity: string
  contract_version: string; removed_metadata_fields: readonly string[]
  record_count: number; window_count: number; trailing_record_count: number
  rows_removed: 0; units_changed: false; values_imputed: false; transforms_applied: false
}
export interface WorkerResult { profile: SourceProfile; prepared?: ArrayBuffer; metadata?: PreparationRecord }
export type CsvOperation = 'inspect' | 'prepare'
export class CsvPreparationError extends Error {
  readonly code: string
  constructor(code: string) { super('Local CSV processing failed.'); this.code = code }
}

// Compare every structural field, type and key as well as the identity. A copied
// hash cannot authenticate altered content; unknown fields/versions fail closed.
function equalStructure(actual: unknown, expected: unknown): boolean {
  if (actual === expected) return true
  if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length
    && expected.every((value, index) => equalStructure(actual[index], value))
  if (expected && typeof expected === 'object') {
    if (!actual || typeof actual !== 'object' || Array.isArray(actual)) return false
    const a = actual as Record<string, unknown>, e = expected as Record<string, unknown>
    return Object.keys(a).length === Object.keys(e).length
      && Object.keys(e).every(key => Object.hasOwn(a, key) && equalStructure(a[key], e[key]))
  }
  return false
}
export function matchesCsvDescriptor(value: unknown): boolean { return equalStructure(value, CSV_SCHEMA) }

export interface CsvWorkerPort {
  postMessage(message: { operation: CsvOperation; file: File }): void
  terminate(): void
  onmessage: ((event: MessageEvent) => void) | null
  onerror: ((event: ErrorEvent) => void) | null
}
export type WorkerFactory = () => CsvWorkerPort
const browserWorker: WorkerFactory = () => new Worker(new URL('./csv-preparation.worker.ts', import.meta.url), { type: 'module' })

/** One ephemeral worker per operation. No file content enters the network. */
export function runCsvWorker(file: File, operation: CsvOperation, signal: AbortSignal,
  factory: WorkerFactory = browserWorker): Promise<WorkerResult> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new CsvPreparationError('CANCELLED')); return }
    let worker: CsvWorkerPort
    try { worker = factory() } catch { reject(new CsvPreparationError('WORKER_UNAVAILABLE')); return }
    let settled = false
    const finish = (value?: WorkerResult, code?: string) => {
      if (settled) return
      settled = true
      signal.removeEventListener('abort', cancel)
      worker.onmessage = worker.onerror = null
      worker.terminate()
      if (code) reject(new CsvPreparationError(code)); else resolve(value!)
    }
    const cancel = () => finish(undefined, 'CANCELLED')
    signal.addEventListener('abort', cancel, { once: true })
    worker.onerror = event => { event.preventDefault(); finish(undefined, 'LOCAL_CSV_ERROR') }
    worker.onmessage = event => {
      const data = event.data
      if (data?.ok !== true) { finish(undefined, typeof data?.code === 'string' ? data.code : 'LOCAL_CSV_ERROR'); return }
      const value = data.result as WorkerResult
      if (!value || !['canonical_78', 'standard_80', 'extended_84'].includes(value.profile)
        || (operation === 'prepare' && (!(value.prepared instanceof ArrayBuffer) || !value.metadata))) {
        finish(undefined, 'LOCAL_CSV_ERROR'); return
      }
      finish(value)
    }
    try { worker.postMessage({ operation, file }) } catch { finish(undefined, 'LOCAL_CSV_ERROR') }
  })
}
