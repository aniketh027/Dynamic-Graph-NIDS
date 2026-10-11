import { analyzeReplay, validateReplay, checkCsvDescriptor } from './api/client.ts'
import { ReplayError, type AnalysisResponse, type Prediction, type ValidationResponse } from './api/replay.ts'
import { CsvPreparationError, runCsvWorker, type SourceProfile, type PreparationRecord } from './csv-compatibility.ts'
import { CSV_SCHEMA } from './csv-schema.generated.ts'

export type ReplayPhase = 'idle' | 'inspecting' | 'file_selected' | 'preparation_required' | 'preparing' | 'prepared' | 'preparation_error' | 'unsupported' | 'validating' | 'validated' | 'validation_error' | 'analyzing' | 'complete' | 'analysis_error' | 'backend_offline'
export interface ReplayState {
  phase: ReplayPhase; file: File | null; validation: ValidationResponse | null
  result: AnalysisResponse | null; error: ReplayError | null; selected: Prediction | null
  sourceProfile: SourceProfile | null; preparedFile: File | null; preparation: PreparationRecord | null
}
const initial = (): ReplayState => ({ phase: 'idle', file: null, validation: null, result: null, error: null, selected: null, sourceProfile: null, preparedFile: null, preparation: null })
export const intrusionPredictions = (result: AnalysisResponse) => result.predictions.filter(p => p.is_attack)
export const predictionPage = (predictions: Prediction[], page: number) => predictions.slice(page * 50, (page + 1) * 50)

/** One session owns one worker or request. Navigation retains settled state.
 * A generation prevents late responses (including abort-ignoring transports) restoring old results. */
export class ReplaySession {
  private state = initial()
  private listeners = new Set<() => void>()
  private request: AbortController | null = null
  private generation = 0
  private base: string
  private client: { validateReplay: typeof validateReplay; analyzeReplay: typeof analyzeReplay; checkCsvDescriptor?: typeof checkCsvDescriptor }
  private local: typeof runCsvWorker
  constructor(base: string, client = { validateReplay, analyzeReplay, checkCsvDescriptor }, local = runCsvWorker) {
    this.base = base; this.client = client; this.local = local
  }
  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<ReplayState>) { this.state = { ...this.state, ...patch }; this.listeners.forEach(l => l()) }
  private invalidate() { this.generation++; this.request?.abort(); this.request = null }
  dispose = () => { this.cancel() }
  select = async (file: File | null, selectionError?: string) => {
    this.invalidate()
    const code = selectionError || (file && (!/\.csv$/i.test(file.name) ? 'INVALID_FILE_TYPE' : file.size > 10 * 1024 * 1024 ? 'FILE_TOO_LARGE' : null))
    this.update({ ...initial(), file, phase: code ? 'validation_error' : file ? 'file_selected' : 'idle', error: code ? new ReplayError(code) : null })
    if (!file || code) return
    const request = new AbortController(), generation = ++this.generation
    this.request = request
    this.update({ phase: 'inspecting' })
    try {
      const result = await this.local(file, 'inspect', request.signal)
      if (generation !== this.generation) return
      this.update({ sourceProfile: result.profile, phase: result.profile === 'canonical_78' ? 'file_selected' : 'preparation_required' })
    } catch (cause) {
      if (generation !== this.generation) return
      const error = new ReplayError(cause instanceof CsvPreparationError ? cause.code : 'LOCAL_CSV_ERROR')
      this.update({ phase: error.code === 'UNSUPPORTED_CSV_FORMAT' ? 'unsupported' : 'preparation_error', error })
    } finally { if (generation === this.generation) this.request = null }
  }
  prepare = async () => {
    const { file, sourceProfile, preparedFile } = this.state
    if (!file || this.request || preparedFile || !sourceProfile || sourceProfile === 'canonical_78') return
    const request = new AbortController(), generation = ++this.generation
    this.request = request
    this.update({ phase: 'preparing', error: null, validation: null, result: null, selected: null, preparation: null })
    try {
      const result = await this.local(file, 'prepare', request.signal)
      if (generation !== this.generation) return
      const m = result.metadata
      if (result.profile !== sourceProfile || !result.prepared || !m
        || m.source_profile !== sourceProfile || m.source_size_bytes !== file.size
        || m.prepared_size_bytes !== result.prepared.byteLength || m.prepared_size_bytes > CSV_SCHEMA.limits.output_bytes
        || m.feature_schema_identity !== CSV_SCHEMA.descriptor_sha256 || m.contract_version !== CSV_SCHEMA.contract_version
        || !/^[a-f0-9]{64}$/.test(m.source_sha256) || !/^[a-f0-9]{64}$/.test(m.prepared_sha256)
        || !Number.isInteger(m.record_count) || m.record_count < 1 || m.record_count > CSV_SCHEMA.limits.flow_records
        || m.window_count !== Math.floor(m.record_count / 5) || m.trailing_record_count !== m.record_count % 5
        || m.rows_removed !== 0 || m.units_changed !== false || m.values_imputed !== false || m.transforms_applied !== false
        || JSON.stringify(m.removed_metadata_fields) !== JSON.stringify(CSV_SCHEMA.profiles[sourceProfile].removed_metadata_fields)) throw new CsvPreparationError('LOCAL_CSV_ERROR')
      const prepared = new File([result.prepared], 'graphnids-prepared.csv', { type: 'text/csv' })
      this.update({ phase: 'prepared', preparedFile: prepared, preparation: m })
    } catch (cause) {
      if (generation !== this.generation) return
      this.update({ phase: 'preparation_error', error: new ReplayError(cause instanceof CsvPreparationError ? cause.code : 'LOCAL_CSV_ERROR') })
    } finally { if (generation === this.generation) this.request = null }
  }
  inspect = (selected: Prediction | null) => { this.update({ selected }) }
  cancel = () => {
    const phase = this.state.phase
    this.invalidate()
    if (phase === 'validating' || phase === 'analyzing') this.update({ phase: phase === 'validating' ? 'validation_error' : 'analysis_error', error: new ReplayError('CANCELLED') })
    if (phase === 'inspecting' || phase === 'preparing') this.update({ phase: 'preparation_error', error: new ReplayError('LOCAL_CANCELLED'), preparedFile: null, preparation: null })
  }
  validate = () => this.run(false)
  analyze = () => this.run(true)
  private async run(analyze: boolean) {
    const { file: source, preparedFile, preparation, sourceProfile, validation } = this.state
    const file = sourceProfile === 'canonical_78' ? source : preparedFile
    if (!file || this.request || (preparedFile && !preparation) || (analyze && (!validation || validation.complete_window_count === 0))) return
    if (!/\.csv$/i.test(file.name) || file.size > 10 * 1024 * 1024) return
    const request = new AbortController()
    this.request = request
    const generation = ++this.generation
    this.update({ phase: analyze ? 'analyzing' : 'validating', error: null, result: null, selected: null, ...(analyze ? {} : { validation: null }) })
    try {
      if (preparation) {
        await (this.client.checkCsvDescriptor ?? checkCsvDescriptor)(this.base, { signal: request.signal })
        if (generation !== this.generation || request.signal.aborted) return
      }
      if (analyze) {
        const result = await this.client.analyzeReplay(this.base, file, { signal: request.signal })
        if (generation !== this.generation) return
        if (result.source_sha256 !== validation!.sha256 || result.source_filename !== file.name
          || result.source_size_bytes !== file.size || result.input_flow_count !== validation!.flow_row_count) throw new ReplayError('MALFORMED')
        this.update({ phase: 'complete', result })
      } else {
        const result = await this.client.validateReplay(this.base, file, { signal: request.signal })
        if (generation !== this.generation) return
        if (result.filename !== file.name || result.size_bytes !== file.size
          || (preparation && (result.sha256 !== preparation.prepared_sha256 || result.flow_row_count !== preparation.record_count))) throw new ReplayError('MALFORMED')
        this.update({ phase: 'validated', validation: result })
      }
    } catch (cause) {
      if (generation !== this.generation) return
      const error = cause instanceof ReplayError ? cause : new ReplayError('NETWORK')
      this.update({ error, phase: error.code === 'NETWORK' ? 'backend_offline' : analyze ? 'analysis_error' : 'validation_error' })
    } finally { if (generation === this.generation) this.request = null }
  }
}
