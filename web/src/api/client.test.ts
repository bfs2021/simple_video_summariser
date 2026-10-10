import { QueryClient } from '@tanstack/react-query'
import { afterEach, describe, expect, expectTypeOf, it, vi } from 'vitest'
import { shouldRetry } from '../queryClient'
import { ApiError, api, apiRequest, rethrowHttp } from './index'
import type { components } from './index'

interface Call {
  url: string
  init: RequestInit
}

function stubFetch(respond: (call: Call) => Response | Promise<Response>): Call[] {
  const calls: Call[] = []
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const call = { url: String(input), init: init ?? {} }
    calls.push(call)
    return respond(call)
  })
  return calls
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

const HEALTH = { status: 'ok', queue: {} }

afterEach(() => {
  vi.unstubAllGlobals()
})

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise
  } catch (error) {
    return error
  }
  throw new Error('expected the request to reject')
}

async function apiError(promise: Promise<unknown>): Promise<ApiError> {
  const error = await rejection(promise)
  expect(error).toBeInstanceOf(ApiError)
  return error as ApiError
}

describe('URL building', () => {
  it('builds every URL through apiUrl', async () => {
    const calls = stubFetch(() => json(HEALTH))
    await api.get('/healthz')
    expect(calls[0]?.url).toBe('/api/healthz')
  })

  it('substitutes a path param, keeping a leading dash', async () => {
    const calls = stubFetch(() => json({}))
    await api.get('/videos/{video_id}', { params: { path: { video_id: '-wNyEUrxzFU' } } })
    expect(calls[0]?.url).toBe('/api/videos/-wNyEUrxzFU')
  })

  it.each([
    ['a/b', 'a%2Fb'],
    ['a?x=1', 'a%3Fx%3D1'],
    ['a#frag', 'a%23frag'],
    ['a b', 'a%20b'],
    ['../x', '..%2Fx'],
    ['//evil.example', '%2F%2Fevil.example'],
    ['{video_id}', '%7Bvideo_id%7D'],
    ['100%', '100%25'],
  ])('encodes reserved characters in %j', async (value, encoded) => {
    const calls = stubFetch(() => json({}))
    await api.get('/videos/{video_id}', { params: { path: { video_id: value } } })
    expect(calls[0]?.url).toBe(`/api/videos/${encoded}`)
  })

  it('stringifies a numeric path param', async () => {
    const calls = stubFetch(() => json({}))
    await api.post('/ops/jobs/{id}/retry', { params: { path: { id: 12 } } })
    expect(calls[0]?.url).toBe('/api/ops/jobs/12/retry')
  })

  it('throws before fetch when a placeholder is left (a JS caller bypassing the types)', async () => {
    const calls = stubFetch(() => json({}))
    const loose = api.get as unknown as (path: string, options?: unknown) => Promise<unknown>
    for (const options of [undefined, { params: { path: {} } }, { params: { path: { video_id: null } } }, { params: { path: { other: 'x' } } }]) {
      await expect(loose('/videos/{video_id}', options)).rejects.toThrow()
    }
    expect(calls).toEqual([])
  })

  it('does not look a path param up on the prototype', async () => {
    const calls = stubFetch(() => json({}))
    const loose = api.get as unknown as (path: string, options?: unknown) => Promise<unknown>
    await expect(loose('/videos/{toString}', { params: { path: {} } })).rejects.toThrow()
    expect(calls).toEqual([])
  })
})

describe('query serialisation', () => {
  it('stringifies numbers and booleans and keeps the key order', async () => {
    const calls = stubFetch(() => json({}))
    await api.get('/videos', { params: { query: { limit: 50, offset: 0, channel: 'UCx' } } })
    expect(calls[0]?.url).toBe('/api/videos?limit=50&offset=0&channel=UCx')
  })

  it('leaves out undefined and null values', async () => {
    const calls = stubFetch(() => json({}))
    await api.get('/videos', { params: { query: { channel: null, status: undefined, limit: 5 } } })
    expect(calls[0]?.url).toBe('/api/videos?limit=5')
  })

  it.each([undefined, {}, { channel: null, status: undefined }])(
    'adds no trailing ? for empty query %j',
    async (query) => {
      const calls = stubFetch(() => json({}))
      await api.get('/videos', query === undefined ? undefined : { params: { query } })
      expect(calls[0]?.url).toBe('/api/videos')
    },
  )

  it('repeats the key for an array value and skips empty items', async () => {
    const calls = stubFetch(() => json({}))
    const loose = api.get as unknown as (path: string, options?: unknown) => Promise<unknown>
    await loose('/videos', { params: { query: { kind: ['a', 'b', null, 3] } } })
    expect(calls[0]?.url).toBe('/api/videos?kind=a&kind=b&kind=3')
  })

  it('encodes a hostile value so it cannot add parameters', async () => {
    const calls = stubFetch(() => json({}))
    await api.get('/search', { params: { query: { q: 'a&limit=50#x' } } })
    expect(calls[0]?.url).toBe('/api/search?q=a%26limit%3D50%23x')
  })
})

describe('request options', () => {
  it('sends Accept, same-origin credentials and the method, and no body or Content-Type for a GET', async () => {
    const calls = stubFetch(() => json(HEALTH))
    await api.get('/healthz')
    const init = calls[0]?.init as RequestInit
    expect(init.method).toBe('GET')
    expect(init.credentials).toBe('same-origin')
    expect(init.headers).toEqual({ Accept: 'application/json' })
    expect(init.body).toBeUndefined()
    expect(init).not.toHaveProperty('mode', 'cors')
  })

  it('sends a JSON body with a Content-Type', async () => {
    const calls = stubFetch(() => json({ ok: 1 }, 201))
    await api.post('/channels', { body: { url: 'https://example.test/c?x=1' } })
    const init = calls[0]?.init as RequestInit
    expect(init.method).toBe('POST')
    expect(init.body).toBe(JSON.stringify({ url: 'https://example.test/c?x=1' }))
    expect(init.headers).toEqual({ Accept: 'application/json', 'Content-Type': 'application/json' })
    expect(init.credentials).toBe('same-origin')
  })

  it('a request without a body sends no Content-Type', async () => {
    const calls = stubFetch(() => json({}))
    await api.post('/ops/jobs/{id}/retry', { params: { path: { id: 1 } } })
    const init = calls[0]?.init as RequestInit
    expect(init.body).toBeUndefined()
    expect(init.headers).toEqual({ Accept: 'application/json' })
  })

  it('apiRequest and api.<method> make the same call', async () => {
    const calls = stubFetch(() => json(HEALTH))
    await apiRequest('get', '/healthz')
    await api.get('/healthz')
    expect(calls[0]).toEqual(calls[1])
  })

  it('passes the signal to fetch', async () => {
    const calls = stubFetch(() => json(HEALTH))
    const controller = new AbortController()
    await api.get('/healthz', { signal: controller.signal })
    expect((calls[0]?.init as RequestInit).signal).toBe(controller.signal)
  })
})

describe('responses', () => {
  it('resolves the parsed JSON of a 2xx', async () => {
    stubFetch(() => json(HEALTH))
    expect(await api.get('/healthz')).toEqual(HEALTH)
  })

  it('resolves a 201 with its body', async () => {
    stubFetch(() => json({ id: 1 }, 201))
    expect(await api.post('/channels', { body: { url: 'x' } })).toEqual({ id: 1 })
  })

  it('resolves a 204 to undefined without reading a body', async () => {
    stubFetch(() => new Response(null, { status: 204 }))
    expect(await api.get('/healthz')).toBeUndefined()
  })

  it('resolves a 2xx with an empty body to undefined', async () => {
    stubFetch(() => new Response('', { status: 200 }))
    expect(await api.get('/healthz')).toBeUndefined()
  })

  it('a 2xx whose body is not JSON rejects with kind parse and no status', async () => {
    stubFetch(() => new Response('<html>SECRET</html>', { status: 200 }))
    const error = await apiError(api.get('/healthz'))
    expect(error.kind).toBe('parse')
    expect(error.status).toBeUndefined()
    expect(error.message).not.toContain('SECRET')
  })
})

describe('errors', () => {
  it('a 404 carries the status, the detail and the body', async () => {
    stubFetch(() => json({ detail: 'video not found' }, 404))
    const error = await apiError(api.get('/videos/{video_id}', { params: { path: { video_id: 'abc' } } }))
    expect(error).toMatchObject({
      kind: 'http',
      status: 404,
      detail: 'video not found',
      body: { detail: 'video not found' },
    })
  })

  it('a 409 keeps its extra fields in body', async () => {
    const body = { detail: 'job is not dead', state: 'running', job_id: 9 }
    stubFetch(() => json(body, 409))
    const error = await apiError(api.post('/ops/jobs/{id}/retry', { params: { path: { id: 9 } } }))
    expect(error.status).toBe(409)
    expect(error.detail).toBe('job is not dead')
    expect(error.body).toEqual(body)
  })

  it('a 422 with an array detail gives detail null and keeps the array in body', async () => {
    const body = { detail: [{ loc: ['query', 'q'], msg: 'field required', type: 'missing' }] }
    stubFetch(() => json(body, 422))
    const error = await apiError(api.get('/search', { params: { query: { q: '' } } }))
    expect(error.status).toBe(422)
    expect(error.detail).toBeNull()
    expect(error.body).toEqual(body)
  })

  it('a 503 /healthz body without a detail gives detail null', async () => {
    stubFetch(() => json({ status: 'unavailable' }, 503))
    const error = await apiError(api.get('/healthz'))
    expect(error).toMatchObject({ status: 503, detail: null, body: { status: 'unavailable' } })
  })

  it.each([
    ['an HTML body', '<html><body>Bad Gateway</body></html>'],
    ['an empty body', ''],
    ['a JSON null', 'null'],
  ])('a 502 with %s gives status 502 and detail null without a parse error', async (_name, text) => {
    stubFetch(() => new Response(text, { status: 502 }))
    const error = await apiError(api.get('/healthz'))
    expect(error).toMatchObject({ kind: 'http', status: 502, detail: null, body: null })
  })

  it('a JSON string body is kept as the body, with detail null', async () => {
    stubFetch(() => new Response('"boom"', { status: 502 }))
    const error = await apiError(api.get('/healthz'))
    expect(error).toMatchObject({ status: 502, detail: null, body: 'boom' })
  })

  it('a network failure rejects with kind network, no status and the cause', async () => {
    const cause = new TypeError('Failed to fetch')
    stubFetch(() => Promise.reject(cause))
    const error = await apiError(api.get('/healthz'))
    expect(error.kind).toBe('network')
    expect(error.status).toBeUndefined()
    expect(error.cause).toBe(cause)
  })

  it('an abort rejects with the AbortError as it is', async () => {
    const abort = new DOMException('aborted', 'AbortError')
    stubFetch(() => Promise.reject(abort))
    expect(await rejection(api.get('/healthz'))).toBe(abort)
  })

  it('an abort during the body read is not wrapped either', async () => {
    const controller = new AbortController()
    stubFetch(() => {
      const response = new Response('{}')
      vi.spyOn(response, 'text').mockImplementation(async () => {
        controller.abort()
        throw new DOMException('aborted', 'AbortError')
      })
      return response
    })
    const error = await rejection(api.get('/healthz', { signal: controller.signal }))
    expect(error).not.toBeInstanceOf(ApiError)
    expect((error as Error).name).toBe('AbortError')
  })

  it('a custom abort reason is not wrapped as an API error', async () => {
    const controller = new AbortController()
    controller.abort('stop')
    stubFetch(() => Promise.reject(new TypeError('x')))
    expect(await rejection(api.get('/healthz', { signal: controller.signal }))).not.toBeInstanceOf(ApiError)
  })

  it('does not wrap a rejection that is not a TypeError', async () => {
    const odd = new RangeError('weird')
    stubFetch(() => Promise.reject(odd))
    expect(await rejection(api.get('/healthz'))).toBe(odd)
  })

  it('the message is the method, the path template and the status, and nothing else', async () => {
    stubFetch(() => json({ detail: 'LEAK-ME' }, 404))
    const error = await apiError(
      api.get('/videos/{video_id}', { params: { path: { video_id: 'secret-id' } } }),
    )
    expect(error.message).toBe('GET /videos/{video_id} failed: 404')
    expect(error.message).not.toMatch(/LEAK-ME|secret-id/)
    const withQuery = await apiError(api.get('/search', { params: { query: { q: 'secret-query' } } }))
    expect(withQuery.message).toBe('GET /search failed: 404')
  })

  it('the network and parse messages are fixed too', async () => {
    stubFetch(() => Promise.reject(new TypeError('boom http://internal.host/x')))
    expect((await apiError(api.get('/healthz'))).message).toBe('GET /healthz failed: network error')
    stubFetch(() => new Response('{{', { status: 200 }))
    expect((await apiError(api.get('/healthz'))).message).toBe('GET /healthz failed: invalid response')
  })
})

describe('the retry policy (#47) on these errors', () => {
  it('does not retry a 404 and retries a 503 and a network error at most twice', async () => {
    stubFetch(() => json({ detail: 'x' }, 404))
    const notFound = await apiError(api.get('/healthz'))
    expect(shouldRetry(0, notFound)).toBe(false)

    stubFetch(() => json({ status: 'unavailable' }, 503))
    const unavailable = await apiError(api.get('/healthz'))
    expect([0, 1, 2].map((n) => shouldRetry(n, unavailable))).toEqual([true, true, false])

    stubFetch(() => Promise.reject(new TypeError('x')))
    const network = await apiError(api.get('/healthz'))
    expect([0, 1, 2].map((n) => shouldRetry(n, network))).toEqual([true, true, false])
  })

  it('retries a parse error', async () => {
    stubFetch(() => new Response('nope', { status: 200 }))
    const parse = await apiError(api.get('/healthz'))
    expect([0, 1, 2].map((n) => shouldRetry(n, parse))).toEqual([true, true, false])
  })
})

describe('rethrowHttp', () => {
  it('maps an HTTP error and passes everything else through', () => {
    const http = new ApiError('http', 'm', { status: 404 })
    const mapped = new Error('mapped')
    expect(() => rethrowHttp(http, () => mapped)).toThrow(mapped)
    const network = new ApiError('network', 'm')
    expect(() => rethrowHttp(network, () => mapped)).toThrow(network)
    const abort = new DOMException('a', 'AbortError')
    expect(() => rethrowHttp(abort, () => mapped)).toThrow(abort)
  })
})

describe('as a TanStack Query queryFn', () => {
  it('is typed from the schema and resolves the stubbed value', async () => {
    stubFetch(() => json(HEALTH))
    const client = new QueryClient()
    const data = await client.fetchQuery({
      queryKey: ['health'],
      queryFn: ({ signal }) => api.get('/healthz', { signal }),
    })
    expectTypeOf(data).toEqualTypeOf<components['schemas']['HealthOut']>()
    expect(data).toEqual(HEALTH)
  })

  it('hands the query signal to fetch, so cancelling the query aborts the request', async () => {
    const seen: (AbortSignal | null | undefined)[] = []
    stubFetch(({ init }) => {
      seen.push(init.signal)
      return json(HEALTH)
    })
    const client = new QueryClient()
    await client.fetchQuery({ queryKey: ['h'], queryFn: ({ signal }) => api.get('/healthz', { signal }) })
    expect(seen[0]).toBeInstanceOf(AbortSignal)
  })
})
