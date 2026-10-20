import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { spawnSync } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'

const webRoot = fileURLToPath(new URL('..', import.meta.url))
const fixtures: string[] = []

function fixture(): { root: string; schema: string; temp: string } {
  const root = mkdtempSync(join(webRoot, '.api-types-test-'))
  fixtures.push(root)
  const schema = join(root, 'src', 'api', 'schema.d.ts')
  const temp = join(root, 'tmp')
  mkdirSync(dirname(schema), { recursive: true })
  mkdirSync(join(root, 'scripts'))
  mkdirSync(join(temp, 'node-compile-cache'), { recursive: true })
  writeFileSync(
    join(root, 'scripts', 'api-types.mjs'),
    readFileSync(join(webRoot, 'scripts', 'api-types.mjs')),
  )
  writeFileSync(join(root, 'openapi.json'), readFileSync(join(webRoot, 'openapi.json')))
  writeFileSync(schema, readFileSync(join(webRoot, 'src', 'api', 'schema.d.ts')))
  return { root, schema, temp }
}

afterEach(() => {
  for (const path of fixtures.splice(0)) rmSync(path, { recursive: true, force: true })
})

describe('API type drift check', () => {
  it('removes its temporary directory after reporting drift', () => {
    const { root, schema, temp } = fixture()
    const original = readFileSync(schema, 'utf8')
    writeFileSync(schema, `${original}\ninterface DriftMarker {}\n`)

    const result = spawnSync(process.execPath, [join(root, 'scripts', 'api-types.mjs'), '--check'], {
      cwd: root,
      encoding: 'utf8',
      env: { ...process.env, TMPDIR: temp },
    })

    expect(result.status).toBe(1)
    expect(result.stderr).toContain(
      'src/api/schema.d.ts is out of date with openapi.json. Run: npm run gen:api',
    )
    expect(readFileSync(schema, 'utf8')).toBe(`${original}\ninterface DriftMarker {}\n`)
    expect(readdirSync(temp).filter((entry) => entry.startsWith('gen-api-'))).toEqual([])
    expect(readdirSync(temp)).toEqual(['node-compile-cache'])
  })

  it('exits successfully without generated temp directories or committed changes when clean', () => {
    const { root, schema, temp } = fixture()
    const original = readFileSync(schema, 'utf8')

    const result = spawnSync(process.execPath, [join(root, 'scripts', 'api-types.mjs'), '--check'], {
      cwd: root,
      encoding: 'utf8',
      env: { ...process.env, TMPDIR: temp },
    })

    expect(result.status).toBe(0)
    expect(readFileSync(schema, 'utf8')).toBe(original)
    expect(readdirSync(temp).filter((entry) => entry.startsWith('gen-api-'))).toEqual([])
    expect(readdirSync(temp)).toEqual(['node-compile-cache'])
  })
})
