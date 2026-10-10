// The typed request helper (#118): the only module that calls `fetch`. Callers
// give it a path and a method from the generated `paths` type; params, query,
// body and the JSON response are inferred from that type. The types are
// compile-time only: nothing here validates a response.
import { apiUrl } from './base'
import { ApiError } from './errors'
import type { paths } from './index'

// ---- types, written against the shape openapi-typescript emits -------------

type HttpMethod = 'get' | 'put' | 'post' | 'delete' | 'patch'
type SuccessStatus = 200 | 201 | 202 | 203 | 204 | 205 | 206 | 207 | 208 | 226

declare const notJson: unique symbol
/** Stands in for a success response that has content, none of it `application/json`. */
type NotJson = typeof notJson

type OperationOf<Paths, P extends keyof Paths, M extends HttpMethod> = M extends keyof Paths[P]
  ? NonNullable<Paths[P][M]>
  : never

type ResponsesOf<Op> = Op extends { responses: infer R } ? R : never

type SuccessBodies<Op> = {
  [S in keyof ResponsesOf<Op> & SuccessStatus]: ResponsesOf<Op>[S] extends { content: infer C }
    ? 'application/json' extends keyof C
      ? C['application/json']
      : NotJson
    : undefined
}[keyof ResponsesOf<Op> & SuccessStatus]

/** An operation the helper can call: it has a 2xx response and every 2xx is JSON or empty. */
type Callable<Op> = [Op] extends [never]
  ? false
  : [SuccessBodies<Op>] extends [never]
    ? false
    : NotJson extends SuccessBodies<Op>
      ? false
      : true

export type MethodsOf<Paths, P extends keyof Paths> = {
  [M in HttpMethod]: Callable<OperationOf<Paths, P, M>> extends true ? M : never
}[HttpMethod]

export type PathsFor<Paths, M extends HttpMethod> = {
  [P in keyof Paths]: M extends MethodsOf<Paths, P> ? P : never
}[keyof Paths]

type ParametersOf<Op> = Op extends { parameters: infer P } ? P : never
type PathParamsOf<Op> = NonNullable<ParametersOf<Op> extends { path?: infer X } ? X : never>
type QueryParamsOf<Op> = NonNullable<ParametersOf<Op> extends { query?: infer X } ? X : never>

type PathPart<Op> = [PathParamsOf<Op>] extends [never] ? { path?: never } : { path: PathParamsOf<Op> }
type QueryPart<Op> = [QueryParamsOf<Op>] extends [never]
  ? { query?: never }
  : ParametersOf<Op> extends { query: unknown }
    ? { query: QueryParamsOf<Op> }
    : { query?: QueryParamsOf<Op> }
type ParamsFor<Op> = PathPart<Op> & QueryPart<Op>

type JsonBodyOf<Op> = NonNullable<Op extends { requestBody?: infer B } ? B : never> extends infer B
  ? B extends { content: { 'application/json': infer J } }
    ? J
    : never
  : never

type BodyPart<Op> = [JsonBodyOf<Op>] extends [never]
  ? { body?: never }
  : Op extends { requestBody: unknown }
    ? { body: JsonBodyOf<Op> }
    : { body?: JsonBodyOf<Op> }

type ParamsField<Op> = {} extends ParamsFor<Op> ? { params?: ParamsFor<Op> } : { params: ParamsFor<Op> }

export type RequestOptions<Paths, P extends keyof Paths, M extends HttpMethod> = ParamsField<
  OperationOf<Paths, P, M>
> &
  BodyPart<OperationOf<Paths, P, M>> & { signal?: AbortSignal }

export type RequestArgs<Paths, P extends keyof Paths, M extends HttpMethod> =
  {} extends RequestOptions<Paths, P, M>
    ? [options?: RequestOptions<Paths, P, M>]
    : [options: RequestOptions<Paths, P, M>]

export type ResponseOf<Paths, P extends keyof Paths, M extends HttpMethod> = Exclude<
  SuccessBodies<OperationOf<Paths, P, M>>,
  NotJson
>

// ---- runtime ----------------------------------------------------------------

interface LooseOptions {
  params?: { path?: Record<string, unknown>; query?: Record<string, unknown> }
  body?: unknown
  signal?: AbortSignal
}

/** The single seam for request options. A future auth header goes here (D13); v1 sends none. */
function requestInit(method: string, body: string | undefined, signal: AbortSignal | undefined): RequestInit {
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  return { method, headers, credentials: 'same-origin', body, signal }
}

function fillPath(template: string, values: Record<string, unknown> | undefined): string {
  const filled = template.replace(/\{([^{}]+)\}/g, (placeholder, name: string) => {
    const value = values !== undefined && Object.hasOwn(values, name) ? values[name] : undefined
    return value === undefined || value === null ? placeholder : encodeURIComponent(String(value))
  })
  if (/[{}]/.test(filled)) throw new Error(`Missing path parameter for ${template}`)
  return filled
}

function queryString(query: Record<string, unknown> | undefined): string {
  const out = new URLSearchParams()
  for (const [key, value] of Object.entries(query ?? {})) {
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item !== undefined && item !== null) out.append(key, String(item))
    }
  }
  return out.toString()
}

function isAbort(error: unknown, signal: AbortSignal | undefined): boolean {
  if (signal?.aborted === true) return true
  return typeof error === 'object' && error !== null && (error as { name?: unknown }).name === 'AbortError'
}

function parseJson(text: string): { ok: true; value: unknown } | { ok: false } {
  try {
    return { ok: true, value: JSON.parse(text) as unknown }
  } catch {
    return { ok: false }
  }
}

async function send(method: string, template: string, options: LooseOptions): Promise<unknown> {
  const label = `${method.toUpperCase()} ${template}`
  const path = fillPath(template, options.params?.path)
  const qs = queryString(options.params?.query)
  const url = apiUrl(qs === '' ? path : `${path}?${qs}`)
  const body = options.body === undefined ? undefined : JSON.stringify(options.body)

  let res: Response
  let text: string
  try {
    res = await fetch(url, requestInit(method.toUpperCase(), body, options.signal))
    text = await res.text() // a 204 has an empty body, which resolves to undefined below
  } catch (error) {
    if (isAbort(error, options.signal) || !(error instanceof TypeError)) throw error
    throw new ApiError('network', `${label} failed: network error`, { cause: error })
  }

  const parsed = text.trim() === '' ? ({ ok: true, value: null } as const) : parseJson(text)
  if (!res.ok) {
    const value = parsed.ok ? parsed.value : null
    const detail =
      typeof value === 'object' && value !== null && typeof (value as { detail?: unknown }).detail === 'string'
        ? (value as { detail: string }).detail
        : null
    throw new ApiError('http', `${label} failed: ${res.status}`, { status: res.status, detail, body: value })
  }
  if (text.trim() === '') return undefined
  if (!parsed.ok) throw new ApiError('parse', `${label} failed: invalid response`)
  return parsed.value
}

/** `apiRequest('get', '/videos/{video_id}', { params: { path: { video_id } }, signal })`. */
export function apiRequest<P extends keyof paths, M extends MethodsOf<paths, P>>(
  method: M,
  path: P,
  ...args: RequestArgs<paths, P, M>
): Promise<ResponseOf<paths, P, M>> {
  return send(method, path as string, (args[0] ?? {}) as LooseOptions) as Promise<ResponseOf<paths, P, M>>
}

function verb<M extends HttpMethod>(method: M) {
  return <P extends PathsFor<paths, M>>(
    path: P,
    ...args: RequestArgs<paths, P, M>
  ): Promise<ResponseOf<paths, P, M>> =>
    send(method, path as string, (args[0] ?? {}) as LooseOptions) as Promise<ResponseOf<paths, P, M>>
}

/** `api.get(path, options)` and friends: the same call with the method fixed. */
export const api = {
  get: verb('get'),
  post: verb('post'),
  put: verb('put'),
  patch: verb('patch'),
  delete: verb('delete'),
}
