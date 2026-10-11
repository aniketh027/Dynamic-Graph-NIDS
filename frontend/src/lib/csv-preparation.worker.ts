import { CSV_SCHEMA as schema } from './csv-schema.generated.ts'
import { CsvPreparationError, type CsvOperation, type PreparationProfile, type SourceProfile, type WorkerResult } from './csv-compatibility.ts'

const fail = (code = 'LOCAL_CSV_ERROR'): never => { throw new CsvPreparationError(code) }
const stripPoints = new Set<number>(schema.numeric.strip_code_points)
const numeric = new RegExp(`^(?:${schema.numeric.pattern})$`, 'i')
const encoder = new TextEncoder()

/** Python strip semantics, deliberately excluding JS-only BOM whitespace. */
export function pythonStrip(value: string): string {
  let start = 0, end = value.length
  while (start < end && stripPoints.has(value.charCodeAt(start))) start++
  while (end > start && stripPoints.has(value.charCodeAt(end - 1))) end--
  return value.slice(start, end)
}
export function validateFeature(cell: string): void {
  // Valid numeric tokens and strip whitespace are all BMP. Rejecting an
  // oversized UTF-16 cell containing astral text cannot exclude a valid token.
  if (cell.length > schema.limits.retained_field_characters) fail('FEATURE_FIELD_TOO_LARGE')
  const value = pythonStrip(cell)
  const match = numeric.exec(value)
  // JS $ permits a final LF; require full matched length as Python fullmatch.
  if (!match || match[0].length !== value.length) fail('NON_NUMERIC_FEATURE')
  if (/^[+-]?(?:nan|inf(?:inity)?)$/i.test(value)) return
  const number = Number(value)
  if (!Number.isFinite(number) || Math.abs(number) > Number(schema.numeric.float32_max_finite)) fail('NON_NUMERIC_FEATURE')
}

/** One logical record at a time. Slices avoid per-character string arrays even
 * for large discarded metadata. Quoted CR/LF and escaped quotes stay exact. */
export function* csvRecords(text: string): Generator<string[]> {
  let i = 0
  while (i < text.length) {
    const row: string[] = []
    if (text[i] === '\r' || text[i] === '\n') {
      const cr = text[i++] === '\r'
      if (cr && text[i] === '\n') i++
      yield row
      continue
    }
    while (true) {
      let cell: string
      if (text[i] === '"') {
        const start = ++i
        let closed = false
        while (i < text.length) {
          if (text[i] === '"') {
            if (text[i + 1] === '"') { i += 2; continue }
            closed = true; break
          }
          i++
        }
        if (!closed) fail()
        cell = text.slice(start, i).replaceAll('""', '"')
        i++
        if (i < text.length && !',\r\n'.includes(text[i])) fail()
      } else {
        const start = i
        while (i < text.length && !',\r\n'.includes(text[i])) {
          if (text[i] === '"') fail()
          i++
        }
        cell = text.slice(start, i)
      }
      row.push(cell)
      if (row.length > 84) fail()
      if (text[i] === ',') { i++; continue }
      if (text[i] === '\r') { i++; if (text[i] === '\n') i++ }
      else if (text[i] === '\n') i++
      break
    }
    yield row
  }
}
const same = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x, i) => x === b[i])
export function recognizeHeader(header: string[]): SourceProfile {
  if (new Set(header).size !== header.length) fail('UNSUPPORTED_CSV_FORMAT')
  for (const profile of ['standard_80', 'extended_84'] as const) {
    if (same(header, schema.profiles[profile].columns)) return profile
  }
  const canonical = new Set<string>(schema.canonical_columns)
  if (header.length === canonical.size && header.every(name => canonical.has(name))) return 'canonical_78'
  return fail('UNSUPPORTED_CSV_FORMAT')
}
const serialize = (cells: readonly string[]) => encoder.encode(cells.map(cell => `"${cell.replaceAll('"', '""')}"`).join(',') + '\n')
const sha256 = async (bytes: ArrayBuffer) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte => byte.toString(16).padStart(2, '0')).join('')

export async function processCsv(file: Blob, operation: CsvOperation): Promise<WorkerResult> {
  if (!['inspect', 'prepare'].includes(operation)) fail()
  if (file.size > schema.limits.source_bytes) fail('FILE_TOO_LARGE')
  const source = await file.arrayBuffer()
  if (source.byteLength !== file.size || source.byteLength > schema.limits.source_bytes) fail('FILE_TOO_LARGE')
  let text: string
  try { text = new TextDecoder('utf-8', { fatal: true }).decode(source) } catch { return fail() }
  const records = csvRecords(text)
  const header = records.next().value ?? []
  const profile = recognizeHeader(header)
  if (operation === 'inspect') return { profile }
  if (profile === 'canonical_78') fail('PREPARATION_NOT_REQUIRED')
  const sourceProfile = profile as PreparationProfile
  const layout = schema.profiles[sourceProfile]
  // Fixed bounded output allocation; no complete table or all-row string list.
  const output = new Uint8Array(schema.limits.output_bytes)
  let offset = 0, count = 0
  const append = (cells: readonly string[]) => {
    const bytes = serialize(cells)
    if (offset + bytes.byteLength > output.byteLength) fail('PREPARED_FILE_TOO_LARGE')
    output.set(bytes, offset); offset += bytes.byteLength
  }
  append(schema.canonical_columns)
  for (const cells of records) {
    if (count === schema.limits.flow_records) fail('TOO_MANY_ROWS')
    if (cells.length !== header.length) fail()
    if (same(cells, schema.profiles.standard_80.columns) || same(cells, schema.profiles.extended_84.columns)) fail('EMBEDDED_HEADER')
    const retained = layout.projection_indices.map(index => cells[index])
    retained.forEach(validateFeature)
    append(retained)
    count++
  }
  if (!count) fail('EMPTY_FILE')
  const prepared = output.slice(0, offset).buffer
  return { profile, prepared, metadata: {
    source_profile: sourceProfile, source_sha256: await sha256(source), prepared_sha256: await sha256(prepared),
    source_size_bytes: source.byteLength, prepared_size_bytes: offset,
    feature_schema_identity: schema.descriptor_sha256, contract_version: schema.contract_version,
    removed_metadata_fields: layout.removed_metadata_fields, record_count: count,
    window_count: Math.floor(count / schema.limits.window), trailing_record_count: count % schema.limits.window,
    rows_removed: 0, units_changed: false, values_imputed: false, transforms_applied: false,
  } }
}

// Node tests import the pure functions. Only a dedicated browser worker owns
// this handler; the main window cannot perform preparation through this module.
if (typeof self !== 'undefined' && typeof document === 'undefined') {
  const scope = self as unknown as { onmessage: (event: MessageEvent) => void; postMessage: (value: unknown, transfer?: Transferable[]) => void }
  scope.onmessage = async event => {
    try {
      const result = await processCsv(event.data.file, event.data.operation)
      scope.postMessage({ ok: true, result }, result.prepared ? [result.prepared] : [])
    } catch (cause) {
      scope.postMessage({ ok: false, code: cause instanceof CsvPreparationError ? cause.code : 'LOCAL_CSV_ERROR' })
    }
  }
}
