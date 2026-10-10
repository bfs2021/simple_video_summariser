// Compile-time tests for the request helper (#118). They run under
// `npm run typecheck` (and so `npm run build`): each `@ts-expect-error` line is
// a rejected call, and deleting one makes tsc fail with "unused directive".
// Nothing here executes the helper; `typeCases` is never called.
import { describe, expectTypeOf, it } from 'vitest'
import { api, apiRequest } from './client'
import type {
  MethodsOf,
  PathsFor,
  RequestArgs,
  RequestOptions,
  ResponseOf,
} from './client'
import type { components } from './index'

declare const signal: AbortSignal

async function typeCases(): Promise<void> {
  // -- path and method keys ---------------------------------------------------
  await api.get('/healthz')
  await apiRequest('get', '/healthz')
  // @ts-expect-error a path that is not in the schema
  await api.get('/nope')
  // @ts-expect-error a path that is not in the schema (apiRequest)
  await apiRequest('get', '/nope')
  // @ts-expect-error post on a GET-only path
  await api.post('/healthz')
  // @ts-expect-error post on a GET-only path (apiRequest)
  await apiRequest('post', '/healthz')
  // @ts-expect-error a method that does not exist
  await apiRequest('trace', '/healthz')

  // -- path params --------------------------------------------------------------
  await api.get('/videos/{video_id}', { params: { path: { video_id: 'abc' } } })
  // @ts-expect-error path params are required for a templated path
  await api.get('/videos/{video_id}')
  // @ts-expect-error params without its path member
  await api.get('/videos/{video_id}', { params: {} })
  // @ts-expect-error a misspelt param name
  await api.get('/videos/{video_id}', { params: { path: { videoId: 'abc' } } })
  // @ts-expect-error a param of the wrong type
  await api.get('/videos/{video_id}', { params: { path: { video_id: 3 } } })
  // @ts-expect-error an extra path param
  await api.get('/videos/{video_id}', { params: { path: { video_id: 'a', extra: 'b' } } })
  // @ts-expect-error path params on a path without template segments
  await api.get('/healthz', { params: { path: { id: 1 } } })
  await api.post('/ops/jobs/{id}/retry', { params: { path: { id: 7 } } })

  // -- query ----------------------------------------------------------------------
  await api.get('/videos')
  await api.get('/videos', { params: { query: { limit: 5, status: 'done' } } })
  await api.get('/videos', { params: {} })
  // @ts-expect-error an unknown query key
  await api.get('/videos', { params: { query: { nope: 1 } } })
  // @ts-expect-error a query value of the wrong type
  await api.get('/videos', { params: { query: { limit: 'five' } } })
  // @ts-expect-error a query value outside the enum
  await api.get('/videos', { params: { query: { status: 'bogus' } } })
  await api.get('/search', { params: { query: { q: 'x' } } })
  // @ts-expect-error a required query param (q) cannot be left out
  await api.get('/search', { params: { query: { limit: 5 } } })
  // @ts-expect-error a required query param means params is required
  await api.get('/search')
  // @ts-expect-error query on an operation with no query parameters
  await api.get('/healthz', { params: { query: { a: 1 } } })

  // -- body -----------------------------------------------------------------------
  await api.post('/channels', { body: { url: 'UCabc' } })
  // @ts-expect-error a required body cannot be left out
  await api.post('/channels')
  // @ts-expect-error a required body cannot be left out (options without it)
  await api.post('/channels', {})
  // @ts-expect-error a body of the wrong shape
  await api.post('/channels', { body: { nope: 1 } })
  // @ts-expect-error a body on an operation with no request body
  await api.post('/ops/jobs/{id}/retry', { params: { path: { id: 7 } }, body: {} })
  // @ts-expect-error a body on a GET
  await api.get('/healthz', { body: {} })

  // -- success response ---------------------------------------------------------------
  const health = await api.get('/healthz', { signal })
  expectTypeOf(health).toEqualTypeOf<components['schemas']['HealthOut']>()
  expectTypeOf(health.status).toEqualTypeOf<string>()
  const viaRequest = await apiRequest('get', '/healthz')
  expectTypeOf(viaRequest).toEqualTypeOf<components['schemas']['HealthOut']>()
  const channel = await api.post('/channels', { body: { url: 'UCabc' } })
  expectTypeOf(channel).toEqualTypeOf<components['schemas']['ChannelRegistration']>()
  // @ts-expect-error a field the schema does not have
  health.nope

  // -- a text/html success response is not fetched ----------------------------------------
  // @ts-expect-error GET /videos/{video_id}/render is text/html
  await api.get('/videos/{video_id}/render', { params: { path: { video_id: 'abc' } } })
  // @ts-expect-error GET /videos/{video_id}/render is text/html (apiRequest)
  await apiRequest('get', '/videos/{video_id}/render', {
    params: { path: { video_id: 'abc' } },
  })
}
void typeCases

// ---- the generic types, on a hand-written fixture in the shape openapi-typescript emits ----

interface Fixture {
  '/things': {
    get: {
      parameters: { query?: { tag?: string[] }; header?: never; path?: never; cookie?: never }
      requestBody?: never
      responses: { 200: { headers: object; content: { 'application/json': { n: number } } } }
    }
    put: {
      parameters: { query?: never; header?: never; path?: never; cookie?: never }
      // an optional body
      requestBody?: { content: { 'application/json': { name: string } } }
      responses: { 201: { headers: object; content: { 'application/json': { id: number } } } }
    }
    post: {
      parameters: { query?: never; header?: never; path?: never; cookie?: never }
      requestBody: { content: { 'application/json': { name: string } } }
      responses: { 201: { headers: object; content: { 'application/json': { id: number } } } }
    }
    delete?: never
    patch?: never
  }
  '/things/{id}': {
    get?: never
    delete: {
      parameters: { query?: never; header?: never; path: { id: number }; cookie?: never }
      requestBody?: never
      responses: { 204: { headers: object; content?: never } }
    }
    post: {
      parameters: { query?: never; header?: never; path: { id: number }; cookie?: never }
      requestBody?: never
      // a JSON 200 next to a text 202: not callable
      responses: {
        200: { headers: object; content: { 'application/json': { ok: true } } }
        202: { headers: object; content: { 'text/plain': string } }
      }
    }
    put?: never
    patch?: never
  }
  '/only-errors': {
    get: {
      parameters: { query?: never; header?: never; path?: never; cookie?: never }
      requestBody?: never
      responses: { 500: { headers: object; content: { 'application/json': { e: string } } } }
    }
    post?: never
    put?: never
    delete?: never
    patch?: never
  }
}

declare function call<P extends keyof Fixture, M extends MethodsOf<Fixture, P>>(
  method: M,
  path: P,
  ...args: RequestArgs<Fixture, P, M>
): Promise<ResponseOf<Fixture, P, M>>

async function fixtureCases(): Promise<void> {
  // optional body: may be left out or given
  await call('put', '/things')
  await call('put', '/things', { body: { name: 'a' } })
  // @ts-expect-error the optional body still has to match the schema
  await call('put', '/things', { body: { name: 1 } })
  // a required body
  await call('post', '/things', { body: { name: 'a' } })
  // @ts-expect-error a required body cannot be left out
  await call('post', '/things')
  // a 204 resolves to undefined
  const gone = await call('delete', '/things/{id}', { params: { path: { id: 1 } } })
  expectTypeOf(gone).toEqualTypeOf<undefined>()
  // an array-valued query parameter
  await call('get', '/things', { params: { query: { tag: ['a', 'b'] } } })
  // @ts-expect-error an array query parameter holds the item type
  await call('get', '/things', { params: { query: { tag: [1] } } })
  // @ts-expect-error a success status that is not JSON next to a JSON one is rejected
  await call('post', '/things/{id}', { params: { path: { id: 1 } } })
  // @ts-expect-error an operation with no 2xx response is rejected
  await call('get', '/only-errors')
  // @ts-expect-error a method the path does not declare
  await call('get', '/things/{id}', { params: { path: { id: 1 } } })
}
void fixtureCases

describe('the helper types', () => {
  it('derives the callable methods and paths from the schema shape', () => {
    expectTypeOf<MethodsOf<Fixture, '/things'>>().toEqualTypeOf<'get' | 'put' | 'post'>()
    expectTypeOf<MethodsOf<Fixture, '/things/{id}'>>().toEqualTypeOf<'delete'>()
    expectTypeOf<MethodsOf<Fixture, '/only-errors'>>().toEqualTypeOf<never>()
    expectTypeOf<PathsFor<Fixture, 'delete'>>().toEqualTypeOf<'/things/{id}'>()
  })

  it('makes the options required exactly when something in them is required', () => {
    expectTypeOf<RequestArgs<Fixture, '/things', 'get'>>().toEqualTypeOf<
      [options?: RequestOptions<Fixture, '/things', 'get'>]
    >()
    expectTypeOf<RequestArgs<Fixture, '/things', 'post'>>().toEqualTypeOf<
      [options: RequestOptions<Fixture, '/things', 'post'>]
    >()
    expectTypeOf<RequestArgs<Fixture, '/things/{id}', 'delete'>>().toEqualTypeOf<
      [options: RequestOptions<Fixture, '/things/{id}', 'delete'>]
    >()
  })

  it('types a JSON response from the schema', () => {
    expectTypeOf<ResponseOf<Fixture, '/things', 'get'>>().toEqualTypeOf<{ n: number }>()
    expectTypeOf<ResponseOf<Fixture, '/things', 'put'>>().toEqualTypeOf<{ id: number }>()
  })
})
