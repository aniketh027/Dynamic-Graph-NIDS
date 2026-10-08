/** C2A wire contract. Indices are zero-based; scores are uncalibrated. */
export interface ValidationResponse {
  validation_id: string; filename: string; size_bytes: number; sha256: string
  flow_row_count: number; canonical_feature_count: 78; columns_reordered: boolean
  schema_valid: true; complete_window_count: number; trailing_flow_count: number
  temporal_window: 5; replay_order: 'csv-row-order'; transform_state_sha256: string
  model_inference_performed: false
}
export interface Prediction {
  window_index: number; target_flow_index: number; prediction_class_id: number
  prediction_label: string; model_score: number; is_attack: boolean
}
export interface ReplayModel {
  task: 'E2'; family: 'HTGNN'; deployment_seed: 2026; checkpoint_id: string
  checkpoint_sha256: string; inference_mode: 'attention-pruned'
  model_score_semantics: 'uncalibrated normalized model score'
}
export interface Preprocessing {
  feature_count: 78; transform_state_id: string; transform_state_sha256: string; temporal_window: 5
}
export interface ReplayGraph {
  input_feature_nodes: number; retained_feature_nodes: number; category_nodes: number
  total_nodes_per_graph: number; edges_per_graph: number; parameter_count: number
}
export interface ReplayProvenance {
  source_kind: 'external_application'; analysis_id: string; uploaded_file_sha256: string
  application_adapter_version: string; feature_schema_sha256: string; feature_order_sha256: string
  transform_state_id: string; transform_state_sha256: string; temporal_window: 5
  replay_order_semantics: 'csv-row-order'; graph_schema_version: string; graph_schema_sha256: string
  graph_taxonomy_version: string; graph_taxonomy_sha256: string; pruning_policy_sha256: string
  pruning_policy_record_sha256: string; checkpoint_id: string; checkpoint_sha256: string
  deployment_seed: 2026; model_task: 'E2'; class_mapping_sha256: string; api_version: 'v1'
}
export interface AnalysisResponse {
  analysis_id: string; source_filename: string; source_sha256: string; source_size_bytes: number
  input_flow_count: number; complete_window_count: number; trailing_flow_count: number
  columns_reordered: boolean; replay_order_semantics: 'csv-row-order'; flow_index_base: 0
  benign_prediction_count: number; intrusion_prediction_count: number
  prediction_class_counts: Record<string, number>; predictions: Prediction[]
  model: ReplayModel; preprocessing: Preprocessing; graph: ReplayGraph
  provenance: ReplayProvenance; provenance_sha256: string
}

type RecordValue = Record<string, unknown>
const record = (v: unknown): v is RecordValue => !!v && typeof v === 'object' && !Array.isArray(v)
const integer = (v: unknown): v is number => Number.isSafeInteger(v) && (v as number) >= 0
const string = (v: unknown) => typeof v === 'string' && v.length > 0
const hash = (v: unknown) => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v)
function fields(v: RecordValue, names: string, test: (v: unknown) => boolean) {
  return names.split(' ').every(key => test(v[key]))
}
const counts = (rows: number, windows: number, trailing: number) =>
  rows > 0 && rows <= 25000 && windows === Math.floor(rows / 5) && trailing === rows % 5

export function decodeValidation(value: unknown): ValidationResponse {
  if (!record(value) || !fields(value, 'validation_id filename', string)
    || !fields(value, 'sha256 transform_state_sha256', hash)
    || !fields(value, 'size_bytes flow_row_count complete_window_count trailing_flow_count', integer)
    || value.canonical_feature_count !== 78 || value.temporal_window !== 5
    || value.schema_valid !== true || value.model_inference_performed !== false
    || typeof value.columns_reordered !== 'boolean' || value.replay_order !== 'csv-row-order'
    || !counts(value.flow_row_count as number, value.complete_window_count as number, value.trailing_flow_count as number)) {
    throw new Error('Malformed validation response')
  }
  return value as unknown as ValidationResponse
}

export function decodeAnalysis(value: unknown): AnalysisResponse {
  const bad = () => { throw new Error('Malformed analysis response') }
  if (!record(value)) return bad()
  const { model, preprocessing, graph, provenance, predictions, prediction_class_counts: histogram } = value
  if (!fields(value, 'analysis_id source_filename', string)
    || !fields(value, 'source_sha256 provenance_sha256', hash)
    || !fields(value, 'source_size_bytes input_flow_count complete_window_count trailing_flow_count benign_prediction_count intrusion_prediction_count', integer)
    || value.flow_index_base !== 0 || value.replay_order_semantics !== 'csv-row-order'
    || typeof value.columns_reordered !== 'boolean'
    || !counts(value.input_flow_count as number, value.complete_window_count as number, value.trailing_flow_count as number)
    || !record(model) || model.task !== 'E2' || model.family !== 'HTGNN' || model.deployment_seed !== 2026
    || model.inference_mode !== 'attention-pruned' || model.model_score_semantics !== 'uncalibrated normalized model score'
    || !string(model.checkpoint_id) || !hash(model.checkpoint_sha256)
    || !record(preprocessing) || preprocessing.feature_count !== 78 || preprocessing.temporal_window !== 5
    || !string(preprocessing.transform_state_id) || !hash(preprocessing.transform_state_sha256)
    || !record(graph) || !fields(graph, 'input_feature_nodes retained_feature_nodes category_nodes total_nodes_per_graph edges_per_graph parameter_count', integer)
    || !record(provenance) || provenance.source_kind !== 'external_application'
    || provenance.api_version !== 'v1' || provenance.model_task !== 'E2' || provenance.deployment_seed !== 2026
    || provenance.temporal_window !== 5 || provenance.replay_order_semantics !== 'csv-row-order'
    || !fields(provenance, 'analysis_id application_adapter_version transform_state_id graph_schema_version graph_taxonomy_version checkpoint_id', string)
    || !fields(provenance, 'uploaded_file_sha256 feature_schema_sha256 feature_order_sha256 transform_state_sha256 graph_schema_sha256 graph_taxonomy_sha256 pruning_policy_sha256 pruning_policy_record_sha256 checkpoint_sha256 class_mapping_sha256', hash)
    || provenance.analysis_id !== value.analysis_id || provenance.uploaded_file_sha256 !== value.source_sha256
    || provenance.checkpoint_id !== model.checkpoint_id || provenance.checkpoint_sha256 !== model.checkpoint_sha256
    || provenance.transform_state_id !== preprocessing.transform_state_id || provenance.transform_state_sha256 !== preprocessing.transform_state_sha256
    || !Array.isArray(predictions) || predictions.length < 1 || predictions.length > 5000
    || predictions.length !== value.complete_window_count || !record(histogram)
    || Object.keys(histogram).length !== 15 || !Object.values(histogram).every(integer)) return bad()
  const actual: Record<string, number> = Object.create(null)
  let attacks = 0
  for (const [index, p] of predictions.entries()) {
    if (!record(p) || p.window_index !== index || p.target_flow_index !== index * 5 + 4
      || !integer(p.prediction_class_id) || p.prediction_class_id >= 15 || !string(p.prediction_label)
      || typeof p.model_score !== 'number' || !Number.isFinite(p.model_score) || p.model_score < 0 || p.model_score > 1
      || typeof p.is_attack !== 'boolean' || p.is_attack !== (p.prediction_label !== 'Benign')
      || !Object.hasOwn(histogram, p.prediction_label as string)) return bad()
    const label = p.prediction_label as string
    actual[label] = (actual[label] ?? 0) + 1
    if (p.is_attack) attacks++
  }
  if (value.intrusion_prediction_count !== attacks || value.benign_prediction_count !== predictions.length - attacks
    || Object.entries(histogram).some(([label, count]) => count !== (actual[label] ?? 0))) return bad()
  return value as unknown as AnalysisResponse
}

export const replayMessages: Record<string, string> = {
  INVALID_FILE_TYPE: 'Choose an uncompressed .csv file with the 78-feature application schema.',
  FILE_TOO_LARGE: 'The maximum file size is 10 MiB.', EMPTY_FILE: 'The CSV contains no flow rows.',
  CSV_PARSE_ERROR: 'The CSV could not be read. Check its UTF-8 encoding and formatting.',
  DUPLICATE_COLUMN: 'The CSV contains duplicate feature columns.',
  UNEXPECTED_COLUMN: 'Remove extra columns, including Label, Timestamp and metadata.',
  MISSING_COLUMN: 'The CSV is missing required canonical feature columns.',
  SCHEMA_MISMATCH: 'Supply exactly the canonical 78 model-input feature columns.',
  NON_NUMERIC_FEATURE: 'A feature contains unsupported nonnumeric text.',
  TOO_MANY_ROWS: 'The maximum upload is 25,000 flow rows.',
  INSUFFICIENT_FLOWS: 'At least five flow rows are required for one prediction.',
  TRANSFORM_STATE_INVALID: 'The frozen preprocessing state is unavailable or invalid.',
  TRANSFORM_FAILED: 'The backend could not transform this file.',
  MODEL_NOT_READY: 'The frozen model is unavailable. Try again when it is ready.',
  MODEL_NOT_LOADED: 'The frozen model is not loaded.', BACKEND_NOT_READY: 'The backend is not ready.',
  ANALYSIS_BUSY: 'Another analysis is already running. Try again when it finishes.',
  INFERENCE_FAILED: 'Model inference failed. No partial results were accepted.',
  INVALID_REQUEST: 'The backend could not accept this upload.',
  CAPABILITY_NOT_IMPLEMENTED: 'This backend does not support the requested operation.',
  INTERNAL_ERROR: 'The backend could not complete the request.',
  NETWORK: 'Backend unavailable. Check the connection and retry.',
  TIMEOUT: 'The request timed out. Server processing may still finish; wait before retrying.',
  CANCELLED: 'Request cancelled. Server processing may still finish.',
  MALFORMED: 'The backend returned an incompatible response. No results were accepted.',
  CONFIGURATION: 'The application API URL is invalid.',
}
export class ReplayError extends Error {
  readonly code: string
  constructor(code: string) { super(replayMessages[code] ?? 'The request failed. Please try again.'); this.code = code }
}
