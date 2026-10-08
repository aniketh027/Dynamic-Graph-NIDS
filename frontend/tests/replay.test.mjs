import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import { analyzeReplay, validateReplay, ANALYSIS_TIMEOUT_MS, VALIDATION_TIMEOUT_MS } from '../src/lib/api/client.ts'
import { decodeAnalysis, decodeValidation, ReplayError, replayMessages } from '../src/lib/api/replay.ts'
import { ReplaySession, intrusionPredictions, predictionPage } from '../src/lib/replay-session.ts'

const wire = JSON.parse((await readFile(new URL('./synthetic-wire-response.json', import.meta.url), 'utf8')).replace(/^\uFEFF/, ''))
const file = new File([' '.repeat(wire.validation.size_bytes)], wire.validation.filename)
const reply = (value, status = 200) => new Response(JSON.stringify(value), { status })
const fixture = () => structuredClone(wire)
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
  const session = new ReplaySession('http://localhost',client()); let notifications = 0
  const unsubscribe = session.subscribe(()=>notifications++)
  await session.analyze(); assert.equal(session.getSnapshot().phase,'idle')
  session.select(file); await session.analyze(); assert.equal(session.getSnapshot().result,null)
  await session.validate(); assert.equal(session.getSnapshot().phase,'validated')
  await session.analyze(); assert.equal(session.getSnapshot().phase,'complete')
  assert.equal(session.getSnapshot().result.complete_window_count,2); assert.ok(notifications >= 5)
  unsubscribe(); session.dispose()
})
test('replacement resets validation, result and drawer; late response cannot restore old state', async () => {
  const api=client(); let resolve
  api.validateReplay=()=>new Promise(r=>resolve=r)
  const session=new ReplaySession('http://localhost',api);session.select(file)
  const pending=session.validate();session.select(new File(['x'],'replacement.csv'))
  resolve(wire.validation);await pending
  assert.equal(session.getSnapshot().phase,'file_selected');assert.equal(session.getSnapshot().validation,null)
  const done=new ReplaySession('http://localhost',client());done.select(file);await done.validate();await done.analyze()
  done.inspect(wire.analysis.predictions[0]);done.select(file)
  assert.equal(done.getSnapshot().result,null);assert.equal(done.getSnapshot().selected,null)
})
test('duplicates are ignored, cancellation invalidates stale analysis, busy is retryable', async () => {
  const api=client();let resolve,calls=0
  api.analyzeReplay=()=>{calls++;return new Promise(r=>resolve=r)}
  const s=new ReplaySession('http://localhost',api);s.select(file);await s.validate()
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
    const s=new ReplaySession('http://localhost',api);s.select(file);await s.validate();await s.analyze()
    assert.equal(s.getSnapshot().validation,null);assert.equal(s.getSnapshot().result,null)
  }
  const api=client();api.analyzeReplay=async()=>({...wire.analysis,source_sha256:'a'.repeat(64)})
  const s=new ReplaySession('http://localhost',api);s.select(file);await s.validate();await s.analyze()
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
  const s=new ReplaySession('http://localhost',client());s.select(file);await s.validate();await s.analyze()
  s.select(null);assert.equal(s.getSnapshot().phase,'idle');assert.equal(s.getSnapshot().result,null)
})
