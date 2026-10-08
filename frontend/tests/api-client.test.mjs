import assert from 'node:assert/strict'
import test from 'node:test'
import { adapterLabel, apiBaseUrl, checkBackend, connectionLabel, retryDelay } from '../src/lib/api/client.ts'

const healthy = { status: 'ok', service: 'graphnids-backend' }
const status = { backend_connected: true, model_adapter_state: 'ready', model_loaded: true,
  inference_ready: false, live_capture_supported: false, file_replay_supported: false,
  mode: 'offline_application_preview', error: null }
const reply = (value, code = 200) => new Response(JSON.stringify(value), { status: code })

test('local default and configured URL normalization', () => {
  assert.equal(apiBaseUrl(), 'http://127.0.0.1:8000')
  assert.equal(apiBaseUrl(' http://localhost:8000/ '), 'http://localhost:8000')
  for (const url of ['file:///tmp', 'http://user:secret@localhost', 'http://localhost?secret=1']) {
    assert.throws(() => apiBaseUrl(url))
  }
})
test('checks only health and status, preserves honest readiness', async () => {
  const paths = []
  const result = await checkBackend(apiBaseUrl(), undefined, async (url, options) => {
    paths.push(url); assert.equal(options.cache, 'no-store')
    assert.equal(options.method, undefined)
    return reply(url.endsWith('/health') ? healthy : status)
  })
  assert.equal(result.state, 'connected')
  assert.equal(connectionLabel(result), 'Backend connected')
  assert.equal(adapterLabel(result), 'Model adapter ready')
  assert.equal(result.status.inference_ready, false)
  assert.deepEqual(paths.sort(), ['http://127.0.0.1:8000/api/v1/health','http://127.0.0.1:8000/api/v1/status'])
})
test('unloaded and errored adapters still mean a connected backend', async () => {
  for (const state of ['uninitialized', 'loading', 'error']) {
    const result = await checkBackend(apiBaseUrl(), undefined, async url => reply(url.endsWith('/health')
      ? healthy : { ...status, model_adapter_state: state, model_loaded: false }))
    assert.equal(result.state, 'connected')
    assert.notEqual(adapterLabel(result), 'Model adapter ready')
  }
})
test('offline failures retain a typed reason without exception details', async () => {
  for (const [fetcher, reason] of [
    [async () => { throw new TypeError('private network details') }, 'network'],
    [async () => { throw new DOMException('private timeout details', 'TimeoutError') }, 'timeout'],
    [async () => { throw new DOMException('private cancellation details', 'AbortError') }, 'cancelled'],
    [async () => reply({}, 503), 'http'],
  ]) {
    assert.deepEqual(await checkBackend(apiBaseUrl(), undefined, fetcher), { state: 'offline', status: null, reason })
  }
})
test('accepts C2A replay readiness without issuing any upload requests', async () => {
  for (const ready of [false, true]) {
    const paths = []
    const result = await checkBackend(apiBaseUrl(), undefined, async url => {
      paths.push(url)
      return reply(url.endsWith('/health') ? healthy
        : { ...status, file_replay_supported: true, inference_ready: ready })
    })
    assert.equal(result.state, 'connected')
    assert.equal(result.status.inference_ready, ready)
    assert.equal(result.status.file_replay_supported, true)
    assert.ok(paths.every(url => url.endsWith('/health') || url.endsWith('/status')))
  }
})

test('rejects malformed JSON, foreign health and inconsistent or invented capabilities', async () => {
  const invalid = [{}, { ...status, model_loaded: false }, { ...status, file_replay_supported: 'true' },
    { ...status, inference_ready: 'true' }, { ...status, live_capture_supported: true },
    { ...status, model_adapter_state: 'unknown' }, { ...status, error: 'private' }]
  for (const data of invalid) {
    assert.deepEqual(await checkBackend(apiBaseUrl(), undefined, async url => reply(url.endsWith('/health') ? healthy : data)),
      { state: 'offline', status: null, reason: 'malformed' })
  }
  assert.equal((await checkBackend(apiBaseUrl(), undefined, async () => reply({ status: 'ok', service: 'another-app' }))).state, 'offline')
  assert.deepEqual(await checkBackend(apiBaseUrl(), undefined, async () => new Response('{broken')),
    { state: 'offline', status: null, reason: 'malformed' })
})
test('forwards cancellation and can recover from offline on a later poll', async () => {
  const controller = new AbortController()
  assert.equal((await checkBackend(apiBaseUrl(), controller.signal, async () => { throw new Error('offline') })).state, 'offline')
  const result = await checkBackend(apiBaseUrl(), controller.signal, async (url, options) => {
    assert.ok(options.signal instanceof AbortSignal)
    return reply(url.endsWith('/health') ? healthy : status)
  })
  assert.equal(result.state, 'connected')
})

test('an early failure cancels the parallel request or response body', async () => {
  for (const parsing of [false, true]) {
    let released = false
    const fetcher = async (url, options) => {
      if (url.endsWith('/health')) {
        if (!parsing) throw new TypeError('unreachable')
        return new Response('{broken')
      }
      const hanging = () => new Promise(resolve => {
        options.signal?.addEventListener('abort', () => { released = true; resolve(reply(status)) }, { once: true })
      })
      return parsing ? { ok: true, json: hanging } : hanging()
    }
    const result = await checkBackend(apiBaseUrl(), undefined, fetcher)
    assert.equal(result.reason, parsing ? 'malformed' : 'network')
    assert.equal(released, true, 'The sibling operation must not outlive the failed poll')
  }
})

test('a connection loss while reading JSON is classified as network failure', async () => {
  const result = await checkBackend(apiBaseUrl(), undefined, async () => ({ ok: true,
    json: async () => { throw new TypeError('connection lost') } }))
  assert.deepEqual(result, { state: 'offline', status: null, reason: 'network' })
})

test('success, disconnect and recovery never reuse stale model readiness', async () => {
  let online = true
  const fetcher = async url => {
    if (!online) throw new TypeError('unreachable')
    return reply(url.endsWith('/health') ? healthy : status)
  }
  assert.equal((await checkBackend(apiBaseUrl(), undefined, fetcher)).status.model_loaded, true)
  online = false
  const offline = await checkBackend(apiBaseUrl(), undefined, fetcher)
  assert.deepEqual(offline, { state: 'offline', status: null, reason: 'network' })
  assert.equal(connectionLabel(offline), 'Backend disconnected')
  assert.equal(adapterLabel(offline), 'Model adapter unavailable')
  online = true
  assert.equal((await checkBackend(apiBaseUrl(), undefined, fetcher)).state, 'connected')
})

test('timeout abort is distinct from cleanup cancellation, including body parsing', async () => {
  for (const parsing of [false, true]) {
    const controller = new AbortController()
    const fetcher = async () => {
      const fail = () => {
        controller.abort(new DOMException('Status deadline exceeded', 'TimeoutError'))
        throw controller.signal.reason
      }
      return parsing ? { ok: true, json: async () => fail() } : fail()
    }
    assert.deepEqual(await checkBackend(apiBaseUrl(), controller.signal, fetcher),
      { state: 'offline', status: null, reason: 'timeout' })
  }
  const cancelled = new AbortController()
  cancelled.abort()
  assert.deepEqual(await checkBackend(apiBaseUrl(), cancelled.signal, async () => { throw cancelled.signal.reason }),
    { state: 'offline', status: null, reason: 'cancelled' })
})

test('bounded offline backoff resets immediately after successful recovery', () => {
  const offline = { state: 'offline', status: null, reason: 'network' }
  assert.deepEqual([1, 2, 3, 4, 50].map(count => retryDelay(offline, count)), [10000, 20000, 40000, 60000, 60000])
  assert.equal(retryDelay({ state: 'connected', status }, 50), 10000)
})
