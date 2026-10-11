import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'
import { CSV_SCHEMA as schema } from '../src/lib/csv-schema.generated.ts'
import { processCsv, csvRecords, validateFeature, pythonStrip } from '../src/lib/csv-preparation.worker.ts'
import { matchesCsvDescriptor, runCsvWorker } from '../src/lib/csv-compatibility.ts'

const cases = JSON.parse(readFileSync(new URL('./fixtures/csv-compatibility-cases.json', import.meta.url), 'utf8'))
const root = fileURLToPath(new URL('../../', import.meta.url))
const quote = value => '"' + value.replaceAll('"', '""') + '"'
const csv = (header, rows = [], { newline = '\n', bom = false, quoted = true } = {}) =>
  (bom ? '\ufeff' : '') + [header, ...rows].map(row => row.map(value => quoted || /[,"\r\n]/.test(value) ? quote(value) : value).join(',') + newline).join('')
const sourceRows = (columns, count = 17) => Array.from({ length: count }, (_, row) => columns.map(name => schema.canonical_columns.includes(name)
  ? String(row * 1000 + schema.canonical_columns.indexOf(name)) : `LOCAL-PRIVATE-${name},"secret"\r\nline`))
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
const file = text => new File([text], 'private-source-identity.csv')
const output = result => Buffer.from(result.prepared)

for (const profile of ['standard_80', 'extended_84']) {
  for (const bom of [false, true]) for (const newline of ['\n', '\r\n']) for (const quoted of [false, true]) {
    test(`${profile} ${bom ? 'BOM' : 'no BOM'} ${JSON.stringify(newline)} quoted=${quoted}: exact records, cells, tail, hashes and privacy`, async () => {
      const columns = schema.profiles[profile].columns, rows = sourceRows(columns)
      rows[3] = [...rows[2]]
      const source = file(csv(columns, rows, { bom, newline, quoted }))
      assert.equal((await processCsv(source, 'inspect')).profile, profile)
      const result = await processCsv(source, 'prepare'), retained = rows.map(row => schema.profiles[profile].projection_indices.map(i => row[i]))
      assert.deepEqual([...csvRecords(output(result).toString())], [schema.canonical_columns, ...retained])
      assert.deepEqual(output(result), Buffer.from(csv(schema.canonical_columns, retained)))
      assert.equal(result.metadata.record_count, 17); assert.equal(result.metadata.window_count, 3); assert.equal(result.metadata.trailing_record_count, 2)
      for (const target of [4, 9, 14]) assert.equal(retained[target][0], String(target * 1000))
      assert.equal(result.metadata.source_sha256, digest(Buffer.from(await source.arrayBuffer())))
      assert.equal(result.metadata.prepared_sha256, digest(output(result)))
      assert.doesNotMatch(output(result).toString() + JSON.stringify(result.metadata), /LOCAL-PRIVATE|secret|private-source-identity/)
      for (const flag of ['units_changed', 'values_imputed', 'transforms_applied']) assert.equal(result.metadata[flag], false)
      assert.equal(result.metadata.rows_removed, 0)
    })
  }
}

test('canonical and shuffled canonical headers retain original upload bytes and need no preparation', async () => {
  for (const header of [schema.canonical_columns, [...schema.canonical_columns].reverse()]) {
    const source = file(csv(header, [header.map(() => '-0')]))
    assert.deepEqual(await processCsv(source, 'inspect'), { profile: 'canonical_78' })
    await assert.rejects(processCsv(source, 'prepare'), { code: 'PREPARATION_NOT_REQUIRED' })
    assert.ok((await source.text()).includes('"-0"'))
  }
})

test('Python whitespace, complete numeric grammar, signed special values and float32 edges', () => {
  for (const value of cases.accepted_numeric) assert.doesNotThrow(() => validateFeature(value), value)
  for (const value of cases.rejected_numeric) assert.throws(() => validateFeature(value), { code: 'NON_NUMERIC_FEATURE' }, value)
  for (const point of schema.numeric.strip_code_points) assert.equal(pythonStrip(String.fromCodePoint(point) + '1' + String.fromCodePoint(point)), '1')
  assert.equal(pythonStrip('\ufeff1'), '\ufeff1')
})

test('unsupported schemas, aliases, reordered sources, duplicated headers and pruned inputs fail locally', async () => {
  const h = [...schema.profiles.standard_80.columns]
  const headers = [[...h].reverse(), [...schema.profiles.extended_84.columns].reverse(), h.slice(0, -1), [...h, 'extra'],
    [h[0], h[0], ...h.slice(2)], [' Dst Port', ...h.slice(1)], ['Dst Port ', ...h.slice(1)], ['dst Port', ...h.slice(1)],
    h.map(v => v === 'CWE Flag Count' ? 'CWR Flag Count' : v), schema.canonical_columns.slice(0, 61), ['Zeek', 'NetFlow', 'Suricata']]
  for (const header of headers) await assert.rejects(processCsv(file(csv(header)), 'inspect'), { code: 'UNSUPPORTED_CSV_FORMAT' })
  for (const text of ['PCAP\x00', 'PK\x03\x04', '', '\n']) await assert.rejects(processCsv(file(text), 'inspect'))
})

test('strict UTF-8 and malformed quotes, including removed metadata, widths and embedded headers', async () => {
  const h = schema.profiles.standard_80.columns, base = csv(h), row = sourceRows(h, 1)[0]
  for (const text of [base, base + '\n', base + '"unterminated', base + '"1"x,2\n', base + 'a"b,2\n',
    csv(h, [row.slice(0, -1)]), csv(h, [[...row, 'extra']]), csv(h, [h]),
    base + row.map((v, i) => i === 2 ? 'metadata"bad' : quote(v)).join(',') + '\n',
    base + row.map((v, i) => i === 2 ? '"metadata"tail' : quote(v)).join(',') + '\n']) {
    await assert.rejects(processCsv(file(text), 'prepare'), error => !/LOCAL-PRIVATE|secret|metadata/.test(error.message))
  }
  for (const bytes of [Buffer.from([0xff]), Buffer.concat([Buffer.from(base), Buffer.from([0xc0, 0xaf])]),
    Buffer.concat([Buffer.from(base), Buffer.from([0xed, 0xa0, 0x80])])]) await assert.rejects(processCsv(new Blob([bytes]), 'prepare'))
})

test('source exact byte cap and discarded metadata larger than retained-field cap', async () => {
  const h = schema.profiles.standard_80.columns, row = h.map(() => '0')
  const initial = Buffer.byteLength(csv(h, [row]))
  row[2] = 'x'.repeat(schema.limits.source_bytes - initial + 1)
  const source = file(csv(h, [row]))
  assert.equal(source.size, schema.limits.source_bytes)
  assert.equal((await processCsv(source, 'prepare')).metadata.record_count, 1)
  await assert.rejects(processCsv(new Blob([source, 'x']), 'prepare'), { code: 'FILE_TOO_LARGE' })
  row[2] = 'private'; row[0] = ' '.repeat(schema.limits.retained_field_characters - 1) + '1'
  assert.equal((await processCsv(file(csv(h, [row])), 'prepare')).metadata.record_count, 1)
  row[0] += ' '
  await assert.rejects(processCsv(file(csv(h, [row])), 'prepare'), { code: 'FEATURE_FIELD_TOO_LARGE' })
})

test('independent output cap below, at and above 10 MiB with source below bound', async () => {
  const h = schema.profiles.standard_80.columns
  // Unquoted input saves 156 bytes per row. Padding is numeric strip whitespace.
  const rows = [h.map(() => '0'), h.map(() => '0')]
  for (const delta of [-1, 0, 1]) {
    const target = schema.limits.output_bytes + delta
    const retained = rows.map(row => schema.profiles.standard_80.projection_indices.map(i => row[i]))
    let remaining = target - Buffer.byteLength(csv(schema.canonical_columns, retained))
    const padded = rows.map(row => [...row])
    for (const row of padded) for (const index of schema.profiles.standard_80.projection_indices) {
      const length = Math.min(remaining, schema.limits.retained_field_characters - 1)
      row[index] = ' '.repeat(length) + '0'; remaining -= length
    }
    assert.equal(remaining, 0)
    const source = file(csv(h, padded, { quoted: false })); assert.ok(source.size < schema.limits.source_bytes)
    if (delta > 0) await assert.rejects(processCsv(source, 'prepare'), { code: 'PREPARED_FILE_TOO_LARGE' })
    else assert.equal((await processCsv(source, 'prepare')).prepared.byteLength, target)
  }
})

test('record cap preserves incomplete tails and caps complete windows at 5000', async () => {
  const h = schema.profiles.standard_80.columns, row = h.map(() => '0'), header = csv(h), body = csv([], [row]).slice(1)
  for (const count of [24999, 25000, 25001]) {
    const source = file(header + body.repeat(count))
    if (count > 25000) await assert.rejects(processCsv(source, 'prepare'), { code: 'TOO_MANY_ROWS' })
    else {
      const result = await processCsv(source, 'prepare')
      assert.equal(result.metadata.record_count, count); assert.equal(result.metadata.window_count, Math.floor(count / 5))
      assert.equal(result.metadata.trailing_record_count, count % 5)
    }
  }
})

test('structural descriptor comparison rejects missing, malformed and copied-hash drift', () => {
  assert.equal(matchesCsvDescriptor(structuredClone(schema)), true)
  for (const value of [null, {}, [], { ...schema, contract_version: 'unknown' }, { ...schema, extra: true }]) assert.equal(matchesCsvDescriptor(value), false)
  for (const mutate of [v => v.canonical_columns.reverse(), v => v.profiles.standard_80.projection_indices[0] = 2,
    v => v.limits.retained_field_characters++, v => v.numeric.pattern = '.*', v => v.descriptor_sha256 = '0'.repeat(64),
    v => delete v.profiles.extended_84, v => v.numeric.strip_code_points.pop()]) {
    const value = structuredClone(schema); mutate(value); assert.equal(matchesCsvDescriptor(value), false)
  }
})

test('worker transport terminates on success, failure, abort, crash and late messages', async () => {
  const source = file('synthetic')
  for (const mode of ['success', 'error', 'abort', 'crash', 'postError', 'invalid']) {
    const controller = new AbortController(); let terminations = 0, handler
    const port = { onmessage: null, onerror: null, terminate: () => terminations++, postMessage(message) {
      assert.equal(message.file, source); assert.equal(message.operation, 'inspect'); handler = this.onmessage
      if (mode === 'postError') throw Error('private')
    } }
    const promise = runCsvWorker(source, 'inspect', controller.signal, () => port)
    if (mode === 'success') port.onmessage({ data: { ok: true, result: { profile: 'canonical_78' } } })
    if (mode === 'error') port.onmessage({ data: { ok: false, code: 'LOCAL_CSV_ERROR' } })
    if (mode === 'invalid') port.onmessage({ data: { ok: true, result: {} } })
    if (mode === 'abort') controller.abort()
    if (mode === 'crash') port.onerror({ preventDefault() {} })
    if (mode === 'success') assert.equal((await promise).profile, 'canonical_78')
    else await assert.rejects(promise)
    handler?.({ data: { ok: true, result: { profile: 'canonical_78' } } })
    assert.equal(terminations, 1); assert.equal(port.onmessage, null); assert.equal(port.onerror, null)
  }
  const c = new AbortController(); c.abort()
  await assert.rejects(runCsvWorker(source, 'inspect', c.signal, () => { throw Error('must not create') }), { code: 'CANCELLED' })
})

test('audited C3A Python differential oracle: exact bytes, numeric acceptance and rejection', async () => {
  const inputs = []
  for (const profile of ['standard_80', 'extended_84']) {
    const h = schema.profiles[profile].columns
    for (const bom of [false, true]) for (const newline of ['\n', '\r\n']) for (const quoted of [false, true]) inputs.push(file(csv(h, sourceRows(h), { bom, newline, quoted })))
    for (const value of [...cases.accepted_numeric, ...cases.rejected_numeric]) {
      const row = sourceRows(h, 1)[0]; row[h.indexOf(schema.canonical_columns[0])] = value
      inputs.push(file(csv(h, [row])))
    }
    inputs.push(file(csv(h) + '"unterminated'), file(csv(h, [h])), file(csv([...h].reverse(), sourceRows([...h].reverse(), 1))))
  }
  const request = await Promise.all(inputs.map(async source => ({ source_base64: Buffer.from(await source.arrayBuffer()).toString('base64') })))
  const oracle = spawnSync(`${root}.venv/Scripts/python.exe`, ['-B', '-m', 'backend.tests.run_csv_ui_safe', '--differential'], {
    cwd: root, input: JSON.stringify(request), encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
  })
  assert.equal(oracle.status, 0, oracle.stderr)
  assert.match(oracle.stderr, /C3B unexpected guard violations: 0/)
  const results = JSON.parse(oracle.stdout); assert.equal(results.length, inputs.length)
  let accepted = 0, rejected = 0
  for (const [index, source] of inputs.entries()) {
    const reference = results[index]
    if (!reference.accepted) { await assert.rejects(processCsv(source, 'prepare')); rejected++; continue }
    const result = await processCsv(source, 'prepare'); accepted++
    assert.deepEqual(output(result), Buffer.from(reference.prepared_base64, 'base64'), `case ${index}`)
    assert.equal(result.metadata.prepared_sha256, reference.prepared_sha256)
    assert.equal(result.metadata.source_sha256, reference.source_sha256)
    assert.equal(result.metadata.record_count, reference.record_count)
  }
  assert.equal(accepted, 62); assert.equal(rejected, 48)
})
