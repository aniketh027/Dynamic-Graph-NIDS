import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { FileSpreadsheet, FileUp, FolderOpen, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useDemo } from '@/lib/demo-context'
import { useReplay } from '@/lib/replay-context'
import { useBackend } from '@/lib/api/context'
import { ProcessingScanner } from '@/components/processing-scanner'
import { CSV_SCHEMA } from '@/lib/csv-schema.generated'

export function TrafficFileInput() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const demo = useDemo()
  const { state, session } = useReplay()
  const backend = useBackend()
  const { file, validation, phase, error, sourceProfile, preparedFile, preparation } = state
  const needsPreparation = sourceProfile === 'standard_80' || sourceProfile === 'extended_84'
  const removed = needsPreparation ? CSV_SCHEMA.profiles[sourceProfile].removed_metadata_fields : []
  const fileTone = phase === 'validating' ? 'validating' : validation ? 'validated' : error ? 'error' : 'selected'
  const fileStatus = phase === 'inspecting' ? 'Inspecting locally' : phase === 'preparing' ? 'Preparing locally'
    : phase === 'validating' ? 'Validation in progress' : validation ? 'Backend validated'
      : preparedFile ? 'Prepared locally — backend validation required' : needsPreparation ? 'Source recognized; local preparation required'
        : error ? 'Validation not successful' : 'Selected; schema not validated'
  const uploadBusy = phase === 'validating' || phase === 'analyzing'
  const localBusy = phase === 'inspecting' || phase === 'preparing'
  const busy = uploadBusy || localBusy
  const online = backend.state === 'connected'
  const choose = (files: FileList | null) => {
    if (!files?.length) return
    demo.reset()
    session.select(files.length === 1 ? files[0] : null, files.length !== 1 ? 'INVALID_REQUEST' : undefined)
  }
  const onChange = (event: ChangeEvent<HTMLInputElement>) => { choose(event.target.files); event.target.value = '' }
  const onDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files) }
  return <div className="file-input-area">
    <input ref={inputRef} type="file" accept=".csv,text/csv" className="sr-only" tabIndex={-1} aria-label="Choose traffic CSV" onChange={onChange} />
    <div className={`traffic-dropzone${dragging ? ' dropzone-dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
      <span className="dropzone-icon"><FileUp size={27} strokeWidth={1.3} aria-hidden="true" /></span><h3>{dragging ? 'Release to select your CSV' : 'Drop a traffic CSV here'}</h3>
      <p>Canonical 78 features or exact documented 80/84 columns · Maximum 10 MiB</p>
      <Button variant="outline" onClick={() => inputRef.current?.click()}><FolderOpen size={14} />{file ? 'Replace file' : 'Browse files'}</Button>
      <span className="dropzone-note">80/84-column sources require explicit local preparation · No PCAP or arbitrary CICFlowMeter CSV</span>
    </div>
    {file && <div className={`selected-file-row file-${fileTone}`}><FileSpreadsheet size={20} aria-hidden="true" /><div><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(1)} KiB · {fileStatus}</span></div><Button variant="ghost" size="icon-sm" aria-label="Remove selected file" onClick={() => session.select(null)}><X /></Button></div>}
    {!online && <p role="status" className="replay-notice">{backend.state === 'checking' ? 'Checking backend connection.' : 'Backend offline.'} Backend connection is required to validate and analyze. Local selection, preparation and demo remain available.</p>}
    {needsPreparation && !preparedFile && <section className="replay-preparation" aria-label="Local CSV preparation"><h3>Source format recognized · {sourceProfile === 'standard_80' ? 'Standard 80 columns' : 'Extended 84 columns'}</h3>
      <p>Preparation occurs locally. Remove only: <strong>{removed.join(', ')}</strong>.</p><p>Only the resulting 78-feature CSV will be uploaded. Removed metadata stays in your browser. Retained model features remain in the upload; this is not complete anonymization.</p>
      <Button disabled={busy} onClick={() => { demo.reset(); void session.prepare() }}>Prepare locally</Button>
    </section>}
    {preparedFile && preparation && <section className="replay-preparation" aria-label="Preparation provenance"><h3>{validation ? 'Local preparation complete' : 'Prepared locally — backend validation required'}</h3>
      <dl className="input-contract-grid"><div><dt>Upload filename</dt><dd>{preparedFile.name}</dd></div><div><dt>Prepared size</dt><dd>{preparedFile.size} bytes</dd></div>
        <div><dt>Flow records / windows / trailing</dt><dd>{preparation.record_count} / {preparation.window_count} / {preparation.trailing_record_count}</dd></div>
        <div><dt>Removed field names</dt><dd>{preparation.removed_metadata_fields.join(', ')}</dd></div>
        <div><dt>Source SHA-256 · local only</dt><dd>{preparation.source_sha256}</dd></div><div><dt>Prepared SHA-256</dt><dd>{preparation.prepared_sha256}</dd></div></dl>
      <p>Record order and decoded feature text preserved. No rows removed, units changed, values imputed or transforms applied. Source linkage is calculated locally; the backend authenticates only the prepared upload.</p>
    </section>}
    {error && <p role="alert" className="replay-error"><strong>{error.code}</strong> · {error.message}</p>}
    {validation && <section className="replay-validation" aria-label="Validation metadata"><h3>File validated</h3><dl className="input-contract-grid">
      <div><dt>Flow rows</dt><dd>{validation.flow_row_count}</dd></div><div><dt>Features</dt><dd>{validation.canonical_feature_count}</dd></div>
      <div><dt>Complete windows</dt><dd>{validation.complete_window_count}</dd></div><div><dt>Trailing flows</dt><dd>{validation.trailing_flow_count} · not inferred</dd></div>
      <div><dt>Column order</dt><dd>{validation.columns_reordered ? 'Reordered by backend' : 'Canonical'}</dd></div><div><dt>File SHA-256</dt><dd title={validation.sha256}>{validation.sha256.slice(0, 16)}…</dd></div>
    </dl><p>CSV row order preserved · Non-overlapping groups of five · Fifth/final flow target.</p>
      {phase === 'validated' && <strong>Model inference has not run yet.</strong>}
      {validation.complete_window_count === 0 && <p>At least five flow rows are needed to analyze. Replace this file with a complete window.</p>}
    </section>}
    <div className="replay-actions">
      {!validation && <Button disabled={!file || !online || busy || !(sourceProfile === 'canonical_78' || preparedFile) || error?.code === 'INVALID_FILE_TYPE' || error?.code === 'FILE_TOO_LARGE'} onClick={() => { demo.reset(); void session.validate() }}>{preparedFile ? 'Validate prepared file' : 'Validate file'}</Button>}
      {validation && phase !== 'complete' && <Button disabled={!online || busy || validation.complete_window_count === 0} onClick={() => { demo.reset(); void session.analyze() }}>Analyze traffic</Button>}
      {phase === 'preparation_error' && !sourceProfile && file && <Button variant="outline" onClick={() => { void session.select(file) }}>Retry local inspection</Button>}
      {busy && <Button variant="outline" onClick={session.cancel}>{localBusy ? 'Cancel local processing' : 'Cancel request'}</Button>}
      <span className="replay-helper">Selecting a file clears the previous session result. Validation uploads the canonical CSV to the configured backend. Leaving Analyze Traffic cancels pending work; settled results stay in this session.</span>
    </div>
    <div role="status" aria-live="polite">{busy && <ProcessingScanner active compact variant={phase === 'analyzing' ? 'radar' : 'crosshair'} label={localBusy ? phase === 'inspecting' ? 'INSPECTING CSV LOCALLY' : 'PREPARING CSV LOCALLY' : phase === 'validating' ? 'VALIDATING TRAFFIC FILE' : 'RUNNING FROZEN HTGNN INFERENCE'} description={localBusy ? 'Browser worker only. No source upload or model computation.' : phase === 'validating' ? 'Checking schema and replay contract. No model inference in validation.' : 'Analysis in progress. CPU processing may take several minutes. Server phase and percentage are not available.'} />}</div>
  </div>
}
