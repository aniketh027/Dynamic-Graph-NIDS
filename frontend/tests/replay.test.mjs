import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import { analyzeReplay, validateReplay, ANALYSIS_TIMEOUT_MS, VALIDATION_TIMEOUT_MS } from '../src/lib/api/client.ts'
import { decodeAnalysis, decodeValidation, ReplayError, replayMessages } from '../src/lib/api/replay.ts'
import { ReplaySession, intrusionPredictions, predictionPage } from '../src/lib/replay-session.ts'
import { processCsv } from '../src/lib/csv-preparation.worker.ts'
import { CSV_SCHEMA } from '../src/lib/csv-schema.generated.ts'
import { createHash } from 'node:crypto'

const wire = JSON.parse((await readFile(new URL('./synthetic-wire-response.json', import.meta.url), 'utf8')).replace(/^\uFEFF/, ''))
const file = new File([' '.repeat(wire.validation.size_bytes)], wire.validation.filename)
const reply = (value, status = 200) => new Response(JSON.stringify(value), { status })
const fixture = () => structuredClone(wire)
const canonicalInspection = async () => ({ profile: 'canonical_78' })
const client = () => ({ validateReplay: async () => fixture().validation, analyzeReplay: async () => fixture().analysis })

test('actual C2A synthetic wire responses decode with full provenance and all classes', () => {
  assert.deepEqual(decodeValidation(wire.validation), wire.validation)
  assert.deepEqual(decodeAnalysis(wire.analysis), wire.analysis)
  assert.equal(Object.keys(wire.analysis.prediction_class_counts).length, 15)
})
test('validation and analysis send raw file bytes, basename and content type', async () => {
  for (const [operation, fn] of [['validate', validateReplay], ['analyze', analyzeReplay]]) {
    let signal
    await fn('http://localhost:8000/', file, { fetcher: async (url, options) => {
      assert.equal(url, `http://localhost:8000/api/v1/replay/${operation}`)
      assert.equal(options.body, file); assert.equal(options.method, 'POST')
      assert.equal(options.headers['X-Filename'], file.name)
      assert.equal(options.headers['Content-Type'], 'text/csv'); signal = options.signal
      return reply(operation === 'validate' ? wire.validation : wire.analysis)
    } })
    assert.equal(signal.aborted, true, 'request resources released after success')
  }
})
test('typed schema, busy and model failures use concise messages, never stack text', async () => {
  for (const code of Object.keys(replayMessages)) {
    await assert.rejects(analyzeReplay('http://localhost', file, { fetcher: async () => reply({error:{code,message:'private stack trace'}},409) }), error => error.code === code && !error.message.includes('private'))
  }
})
test('invalid validation, corrupt histogram, indices, score, provenance and class fields rejected', () => {
  for (const change of [v => v.model_inference_performed = true, v => v.schema_valid = false, v => v.flow_row_count++, v => v.sha256 = 'x']) {
    const v = fixture().validation; change(v); assert.throws(() => decodeValidation(v))
  }
  for (const change of [v => v.predictions[0].model_score = NaN, v => v.predictions[0].model_score = 1.1,
    v => v.predictions[0].is_attack = true, v => v.predictions[0].target_flow_index = 5,
    v => v.prediction_class_counts.Benign++, v => v.provenance.checkpoint_id = 'wrong',
    v => delete v.provenance.feature_order_sha256, v => v.predictions[0].prediction_class_id = 15,
    v => v.predictions.pop(), v => v.model.deployment_seed = 2027]) {
    const v = fixture().analysis; change(v); assert.throws(() => decodeAnalysis(v))
  }
})
test('network failure and malformed responses are typed', async () => {
  await assert.rejects(validateReplay('http://localhost', file, {fetcher: async () => { throw new TypeError('secret') }}), {code:'NETWORK'})
  await assert.rejects(validateReplay('http://localhost', file, {fetcher: async () => reply({})}), {code:'MALFORMED'})
  await assert.rejects(validateReplay('file:///bad', file), {code:'CONFIGURATION'})
})
test('analysis has an independent long deadline, aborts network and body parsing, cleans listener', async () => {
  assert.ok(ANALYSIS_TIMEOUT_MS >= 600000); assert.ok(ANALYSIS_TIMEOUT_MS > VALIDATION_TIMEOUT_MS)
  for (const parsing of [false, true]) {
    let signal
    const hang = options => new Promise((_, reject) => { signal = options.signal; signal.addEventListener('abort', () => reject(signal.reason), {once:true}) })
    await assert.rejects(analyzeReplay('http://localhost', file, {timeoutMs:10,fetcher:async (_, options) => parsing ? {ok:true,json:()=>hang(options)} : hang(options)}), {code:'TIMEOUT'})
    assert.equal(signal.aborted,true)
  }
  const controller = new AbortController(); controller.abort()
  let called = false
  await assert.rejects(analyzeReplay('http://localhost',file,{signal:controller.signal,fetcher:async()=>{called=true}}),{code:'CANCELLED'})
  assert.equal(called,false)
})
test('successful validation gates analysis; state propagates through one session subscription', async () => {
  const session = new ReplaySession('http://localhost',client(),canonicalInspection); let notifications = 0
  const unsubscribe = session.subscribe(()=>notifications++)
  await session.analyze(); assert.equal(session.getSnapshot().phase,'idle')
  await session.select(file); await session.analyze(); assert.equal(session.getSnapshot().result,null)
  await session.validate(); assert.equal(session.getSnapshot().phase,'validated')
  await session.analyze(); assert.equal(session.getSnapshot().phase,'complete')
  assert.equal(session.getSnapshot().result.complete_window_count,2); assert.ok(notifications >= 5)
  unsubscribe(); session.dispose()
})
test('replacement resets validation, result and drawer; late response cannot restore old state', async () => {
  const api=client(); let resolve
  api.validateReplay=()=>new Promise(r=>resolve=r)
  const session=new ReplaySession('http://localhost',api,canonicalInspection);await session.select(file)
  const pending=session.validate();await session.select(new File(['x'],'replacement.csv'))
  resolve(wire.validation);await pending
  assert.equal(session.getSnapshot().phase,'file_selected');assert.equal(session.getSnapshot().validation,null)
  const done=new ReplaySession('http://localhost',client(),canonicalInspection);await done.select(file);await done.validate();await done.analyze()
  done.inspect(wire.analysis.predictions[0]);await done.select(file)
  assert.equal(done.getSnapshot().result,null);assert.equal(done.getSnapshot().selected,null)
})
test('duplicates are ignored, cancellation invalidates stale analysis, busy is retryable', async () => {
  const api=client();let resolve,calls=0
  api.analyzeReplay=()=>{calls++;return new Promise(r=>resolve=r)}
  const s=new ReplaySession('http://localhost',api,canonicalInspection);await s.select(file);await s.validate()
  const pending=s.analyze();await s.analyze();assert.equal(calls,1)
  s.cancel();resolve(wire.analysis);await pending;assert.equal(s.getSnapshot().result,null)
  assert.equal(s.getSnapshot().error.code,'CANCELLED')
  api.analyzeReplay=async()=>{throw new ReplayError('ANALYSIS_BUSY')};await s.analyze()
  assert.equal(s.getSnapshot().error.code,'ANALYSIS_BUSY');assert.ok(s.getSnapshot().validation)
  api.analyzeReplay=client().analyzeReplay;await s.analyze();assert.ok(s.getSnapshot().result)
})
test('offline and invalid validation never enable inference; mismatched file identity rejected', async () => {
  for(const code of ['NETWORK','MISSING_COLUMN']) {
    const api=client();api.validateReplay=async()=>{throw new ReplayError(code)}
    const s=new ReplaySession('http://localhost',api,canonicalInspection);await s.select(file);await s.validate();await s.analyze()
    assert.equal(s.getSnapshot().validation,null);assert.equal(s.getSnapshot().result,null)
  }
  const api=client();api.analyzeReplay=async()=>({...wire.analysis,source_sha256:'a'.repeat(64)})
  const s=new ReplaySession('http://localhost',api,canonicalInspection);await s.select(file);await s.validate();await s.analyze()
  assert.equal(s.getSnapshot().error.code,'MALFORMED')
})
test('benign/intrusion mapping and pagination are bounded for 5000 predictions', () => {
  const mock=fixture().analysis
  mock.predictions=[{...mock.predictions[0],prediction_label:'Bot',is_attack:true},mock.predictions[1]]
  assert.equal(intrusionPredictions(mock).length,1)
  const rows=Array.from({length:5000},(_,i)=>({...mock.predictions[0],window_index:i}))
  assert.equal(predictionPage(rows,0).length,50);assert.equal(predictionPage(rows,99)[49].window_index,4999)
})
test('switching to demo clears all real state through the same reset action', async()=>{
  const s=new ReplaySession('http://localhost',client(),canonicalInspection);await s.select(file);await s.validate();await s.analyze()
  s.select(null);assert.equal(s.getSnapshot().phase,'idle');assert.equal(s.getSnapshot().result,null)
})

const hashBytes = bytes => createHash('sha256').update(bytes).digest('hex')
const sourceFile = (profile = 'extended_84') => {
  const h = CSV_SCHEMA.profiles[profile].columns
  const quote = v => '"' + v.replaceAll('"', '""') + '"'
  return new File([[h, ...Array.from({ length: 12 }, (_, row) => h.map(name => CSV_SCHEMA.canonical_columns.includes(name) ? String(row) : 'PRIVATE-IP-TIMESTAMP-LABEL'))]
    .map(row => row.map(quote).join(',') + '\n').join('')], 'sensitive-original.csv')
}
const localWorker = async (f, operation) => processCsv(f, operation)
function preparedClient(calls = []) {
  const validation = async f => ({ ...wire.validation, filename: f.name, size_bytes: f.size, sha256: hashBytes(Buffer.from(await f.arrayBuffer())) })
  return {
    checkCsvDescriptor: async () => { calls.push('schema') },
    validateReplay: async (_, f) => { calls.push(f); return validation(f) },
    analyzeReplay: async (_, f) => {
      calls.push(f); const v = await validation(f)
      return { ...fixture().analysis, source_filename: f.name, source_size_bytes: f.size, source_sha256: v.sha256,
        provenance: { ...wire.analysis.provenance, uploaded_file_sha256: v.sha256 } }
    },
  }
}
async function preparedSession(api = preparedClient(), local = localWorker, source = sourceFile()) {
  const s = new ReplaySession('http://localhost', api, local)
  await s.select(source); await s.prepare(); return s
}

test('explicit preparation is required; only the safe prepared file reaches validate/analyze', async () => {
  for (const profile of ['standard_80', 'extended_84']) {
    const source = sourceFile(profile), calls = [], s = new ReplaySession('http://localhost', preparedClient(calls), localWorker)
    await s.select(source)
    assert.equal(s.getSnapshot().phase, 'preparation_required')
    await s.validate(); await s.analyze(); assert.deepEqual(calls, [])
    await s.prepare(); assert.equal(s.getSnapshot().phase, 'prepared'); assert.equal(s.getSnapshot().validation, null)
    const prepared = s.getSnapshot().preparedFile
    assert.equal(s.getSnapshot().file, source); assert.notEqual(prepared, source)
    assert.equal(prepared.name, 'graphnids-prepared.csv'); assert.doesNotMatch(await prepared.text(), /PRIVATE-IP|TIMESTAMP|LABEL/)
    await s.validate(); assert.equal(s.getSnapshot().phase, 'validated')
    await s.analyze(); assert.equal(s.getSnapshot().phase, 'complete')
    assert.deepEqual(calls, ['schema', prepared, 'schema', prepared])
    assert.equal(s.getSnapshot().preparation.record_count, s.getSnapshot().result.input_flow_count)
    assert.equal(s.getSnapshot().preparation.prepared_sha256, s.getSnapshot().validation.sha256)
  }
})

test('prepared response linkage rejects filename, size, digest and flow count mismatches', async () => {
  for (const change of [v => v.filename = 'sensitive-original.csv', v => v.size_bytes++, v => v.sha256 = 'a'.repeat(64), v => v.flow_row_count++]) {
    const api = preparedClient(), original = api.validateReplay
    api.validateReplay = async (...args) => { const v = await original(...args); change(v); return v }
    const s = await preparedSession(api); await s.validate()
    assert.equal(s.getSnapshot().phase, 'validation_error'); assert.equal(s.getSnapshot().error.code, 'MALFORMED')
    assert.equal(s.getSnapshot().validation, null); await s.analyze(); assert.equal(s.getSnapshot().result, null)
  }
})

test('schema mismatch blocks both prepared operations; canonical regression skips schema GET', async () => {
  const calls = [], api = preparedClient(calls), s = await preparedSession(api)
  api.checkCsvDescriptor = async () => { throw new ReplayError('CSV_DESCRIPTOR_MISMATCH') }
  await s.validate(); assert.deepEqual(calls, []); assert.equal(s.getSnapshot().error.code, 'CSV_DESCRIPTOR_MISMATCH')
  api.checkCsvDescriptor = async () => {}; await s.validate(); calls.length = 0
  api.checkCsvDescriptor = async () => { throw new ReplayError('CSV_DESCRIPTOR_MISMATCH') }
  await s.analyze(); assert.deepEqual(calls, []); assert.equal(s.getSnapshot().result, null)
  const canonical = new File([CSV_SCHEMA.canonical_columns.join(',') + '\n' + Array(78).fill('0').join(',') + '\n'], 'canonical.csv')
  let upload
  const c = new ReplaySession('http://localhost', { validateReplay: async (_, f) => { upload = f; return { ...wire.validation, filename: f.name, size_bytes: f.size } },
    analyzeReplay: api.analyzeReplay, checkCsvDescriptor: api.checkCsvDescriptor }, localWorker)
  await c.select(canonical); await c.validate(); assert.equal(c.getSnapshot().phase, 'validated'); assert.equal(upload, canonical)
})

test('replacement during inspection invalidates an abort-ignoring worker result', async () => {
  let resolve, signal
  const source = sourceFile(), replacement = new File(['synthetic'], 'replacement.csv')
  const local = (f, _, abort) => f === source ? new Promise(r => { resolve = r; signal = abort }) : Promise.resolve({ profile: 'canonical_78' })
  const s = new ReplaySession('http://localhost', preparedClient(), local)
  const pending = s.select(source); assert.equal(s.getSnapshot().phase, 'inspecting')
  await s.select(replacement); assert.equal(signal.aborted, true)
  resolve({ profile: 'extended_84' }); await pending
  assert.equal(s.getSnapshot().file, replacement); assert.equal(s.getSnapshot().sourceProfile, 'canonical_78')
  assert.equal(s.getSnapshot().preparation, null)
})

for (const action of ['cancel', 'replace', 'remove', 'dispose']) test(`${action} during preparation rejects late worker publication and duplicate clicks`, async () => {
  const source = sourceFile(), result = await processCsv(source, 'prepare')
  let resolve, signal, count = 0
  const local = async (f, op, abort) => op === 'inspect' ? { profile: f === source ? 'extended_84' : 'canonical_78' }
    : new Promise(r => { resolve = r; signal = abort; count++ })
  const s = new ReplaySession('http://localhost', preparedClient(), local)
  await s.select(source); const pending = s.prepare(); await s.prepare(); assert.equal(count, 1)
  if (action === 'replace') await s.select(new File(['x'], 'replacement.csv'))
  else if (action === 'remove') await s.select(null)
  else s[action]()
  assert.equal(signal.aborted, true); resolve(result); await pending
  assert.equal(s.getSnapshot().preparedFile, null); assert.equal(s.getSnapshot().preparation, null)
  assert.notEqual(s.getSnapshot().phase, 'prepared')
})

for (const stage of ['schema', 'validate', 'analyze']) test(`replacement during ${stage} cancels and suppresses abort-ignoring network responses`, async () => {
  const source = sourceFile(), api = preparedClient(), s = await preparedSession(api, localWorker, source)
  let resolve, signal, calls = 0
  if (stage === 'analyze') await s.validate()
  const key = stage === 'schema' ? 'checkCsvDescriptor' : stage === 'validate' ? 'validateReplay' : 'analyzeReplay'
  const original = api[key]
  api[key] = async (...args) => { calls++; signal = (stage === 'schema' ? args[1] : args[2]).signal; const v = await original(...args); return new Promise(r => { resolve = () => r(v) }) }
  const pending = stage === 'analyze' ? s.analyze() : s.validate()
  for (let i = 0; i < 20 && !resolve; i++) await new Promise(r => setTimeout(r, 1))
  assert.ok(resolve); await (stage === 'analyze' ? s.analyze() : s.validate()); assert.equal(calls, 1)
  await s.select(null); assert.equal(signal.aborted, true); resolve(); await pending
  assert.equal(s.getSnapshot().phase, 'idle'); assert.equal(s.getSnapshot().result, null); assert.equal(s.getSnapshot().validation, null)
})

test('navigation cancellation retains settled preparation, validation and completed results', async () => {
  const s = await preparedSession()
  const prepared = s.getSnapshot().preparedFile; s.cancel(); assert.equal(s.getSnapshot().preparedFile, prepared)
  await s.validate(); const validation = s.getSnapshot().validation; s.cancel(); assert.equal(s.getSnapshot().validation, validation)
  await s.analyze(); const result = s.getSnapshot().result; s.cancel(); assert.equal(s.getSnapshot().result, result)
  s.dispose(); assert.equal(s.getSnapshot().result, result)
  await s.select(null); assert.equal(s.getSnapshot().preparedFile, null); assert.equal(s.getSnapshot().result, null)
})

test('offline schema check never uploads; preparation and retry remain local and available', async () => {
  const calls = [], api = preparedClient(calls)
  api.checkCsvDescriptor = async () => { throw new ReplayError('NETWORK') }
  const s = await preparedSession(api); assert.equal(s.getSnapshot().phase, 'prepared')
  await s.validate(); assert.equal(s.getSnapshot().phase, 'backend_offline'); assert.deepEqual(calls, [])
  assert.ok(s.getSnapshot().preparedFile); api.checkCsvDescriptor = async () => {}; await s.validate()
  assert.equal(s.getSnapshot().phase, 'validated')
})
