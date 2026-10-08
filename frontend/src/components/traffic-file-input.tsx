import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { FileSpreadsheet, FileUp, FolderOpen, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useDemo } from '@/lib/demo-context'
import { useReplay } from '@/lib/replay-context'
import { useBackend } from '@/lib/api/context'
import { ProcessingScanner } from '@/components/processing-scanner'

export function TrafficFileInput() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const demo = useDemo()
  const { state, session } = useReplay()
  const backend = useBackend()
  const { file, validation, phase, error } = state
  const fileTone = phase === 'validating' ? 'validating' : validation ? 'validated' : error ? 'error' : 'selected'
  const fileStatus = phase === 'validating' ? 'Validation in progress' : validation ? 'Backend validated' : error ? 'Validation not successful' : 'Selected; schema not validated'
  const busy = phase === 'validating' || phase === 'analyzing'
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
      <p>78-feature GraphNIDS application CSV · Maximum 10 MiB</p>
      <Button variant="outline" onClick={() => inputRef.current?.click()}><FolderOpen size={14} />{file ? 'Replace file' : 'Browse files'}</Button>
      <span className="dropzone-note">No Label, Timestamp or extra metadata · No PCAP or arbitrary CICFlowMeter CSV</span>
    </div>
    {file && <div className={`selected-file-row file-${fileTone}`}><FileSpreadsheet size={20} aria-hidden="true" /><div><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(1)} KiB · {fileStatus}</span></div><Button variant="ghost" size="icon-sm" aria-label="Remove selected file" onClick={() => session.select(null)}><X /></Button></div>}
    {!online && <p role="status" className="replay-notice">{backend.state === 'checking' ? 'Checking backend connection.' : 'Backend offline.'} Backend connection is required to validate and analyze. Local file selection and demo remain available.</p>}
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
      {!validation && <Button disabled={!file || !online || busy || error?.code === 'INVALID_FILE_TYPE' || error?.code === 'FILE_TOO_LARGE'} onClick={() => { demo.reset(); void session.validate() }}>Validate file</Button>}
      {validation && phase !== 'complete' && <Button disabled={!online || busy || validation.complete_window_count === 0} onClick={() => { demo.reset(); void session.analyze() }}>Analyze traffic</Button>}
      {busy && <Button variant="outline" onClick={session.cancel}>Cancel request</Button>}
      <span className="replay-helper">Selecting a file clears the previous session result. Validation sends the CSV to the configured backend.</span>
    </div>
    <div role="status" aria-live="polite">{busy && <ProcessingScanner active compact variant={phase === 'validating' ? 'crosshair' : 'radar'} label={phase === 'validating' ? 'VALIDATING TRAFFIC FILE' : 'RUNNING FROZEN HTGNN INFERENCE'} description={phase === 'validating' ? 'Checking schema and replay contract. No model inference in validation.' : 'Analysis in progress. CPU processing may take several minutes. Server phase and percentage are not available.'} />}</div>
  </div>
}
