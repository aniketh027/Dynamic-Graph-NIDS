import { analyzeReplay, validateReplay } from './api/client.ts'
import { ReplayError, type AnalysisResponse, type Prediction, type ValidationResponse } from './api/replay.ts'

export type ReplayPhase = 'idle' | 'file_selected' | 'validating' | 'validated' | 'validation_error' | 'analyzing' | 'complete' | 'analysis_error' | 'backend_offline'
export interface ReplayState {
  phase: ReplayPhase; file: File | null; validation: ValidationResponse | null
  result: AnalysisResponse | null; error: ReplayError | null; selected: Prediction | null
}
const initial = (): ReplayState => ({ phase: 'idle', file: null, validation: null, result: null, error: null, selected: null })
export const intrusionPredictions = (result: AnalysisResponse) => result.predictions.filter(p => p.is_attack)
export const predictionPage = (predictions: Prediction[], page: number) => predictions.slice(page * 50, (page + 1) * 50)

/** One session owns one request. Navigation retains it; replacement/unmount cancels it.
 * A generation prevents late responses (including abort-ignoring transports) restoring old results. */
export class ReplaySession {
  private state = initial()
  private listeners = new Set<() => void>()
  private request: AbortController | null = null
  private generation = 0
  private base: string
  private client: { validateReplay: typeof validateReplay; analyzeReplay: typeof analyzeReplay }
  constructor(base: string, client = { validateReplay, analyzeReplay }) { this.base = base; this.client = client }
  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<ReplayState>) { this.state = { ...this.state, ...patch }; this.listeners.forEach(l => l()) }
  dispose = () => { this.generation++; this.request?.abort(); this.request = null }
  select = (file: File | null, selectionError?: string) => {
    this.dispose()
    const code = selectionError || (file && (!/\.csv$/i.test(file.name) ? 'INVALID_FILE_TYPE' : file.size > 10 * 1024 * 1024 ? 'FILE_TOO_LARGE' : null))
    this.update({ ...initial(), file, phase: code ? 'validation_error' : file ? 'file_selected' : 'idle', error: code ? new ReplayError(code) : null })
  }
  inspect = (selected: Prediction | null) => { this.update({ selected }) }
  cancel = () => {
    const phase = this.state.phase
    this.dispose()
    if (phase === 'validating' || phase === 'analyzing') this.update({ phase: phase === 'validating' ? 'validation_error' : 'analysis_error', error: new ReplayError('CANCELLED') })
  }
  validate = () => this.run(false)
  analyze = () => this.run(true)
  private async run(analyze: boolean) {
    const { file, validation } = this.state
    if (!file || this.request || (analyze && (!validation || validation.complete_window_count === 0))) return
    if (!/\.csv$/i.test(file.name) || file.size > 10 * 1024 * 1024) return
    const request = new AbortController()
    this.request = request
    const generation = ++this.generation
    this.update({ phase: analyze ? 'analyzing' : 'validating', error: null, result: null, selected: null, ...(analyze ? {} : { validation: null }) })
    try {
      if (analyze) {
        const result = await this.client.analyzeReplay(this.base, file, { signal: request.signal })
        if (generation !== this.generation) return
        if (result.source_sha256 !== validation!.sha256 || result.source_filename !== file.name
          || result.source_size_bytes !== file.size || result.input_flow_count !== validation!.flow_row_count) throw new ReplayError('MALFORMED')
        this.update({ phase: 'complete', result })
      } else {
        const result = await this.client.validateReplay(this.base, file, { signal: request.signal })
        if (generation !== this.generation) return
        if (result.filename !== file.name || result.size_bytes !== file.size) throw new ReplayError('MALFORMED')
        this.update({ phase: 'validated', validation: result })
      }
    } catch (cause) {
      if (generation !== this.generation) return
      const error = cause instanceof ReplayError ? cause : new ReplayError('NETWORK')
      this.update({ error, phase: error.code === 'NETWORK' ? 'backend_offline' : analyze ? 'analysis_error' : 'validation_error' })
    } finally { if (generation === this.generation) this.request = null }
  }
}
