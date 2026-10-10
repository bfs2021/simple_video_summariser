/** What went wrong: an HTTP error status, no response at all, or a 2xx body that is not JSON. */
export type ApiErrorKind = 'http' | 'network' | 'parse'

/**
 * The one error the request helper rejects with (except for an abort, which
 * passes through as it is). `message` is fixed and short: the method, the path
 * template and the status. It never holds the response body, a path value or a
 * query string, so a view that shows it by mistake leaks nothing.
 *
 * `status` is the HTTP status of an `http` error and `undefined` otherwise, so
 * the retry predicate (queryClient.ts) never mistakes a network or parse error
 * for a 4xx.
 */
export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number | undefined
  /** The body's `detail` when it is a string, else `null`. */
  readonly detail: string | null
  /** The parsed JSON body, or `null` when the body is empty or not JSON. */
  readonly body: unknown

  constructor(
    kind: ApiErrorKind,
    message: string,
    init: { status?: number; detail?: string | null; body?: unknown; cause?: unknown } = {},
  ) {
    super(message, init.cause === undefined ? undefined : { cause: init.cause })
    this.name = 'ApiError'
    this.kind = kind
    this.status = init.status
    this.detail = init.detail ?? null
    this.body = init.body ?? null
  }
}

/**
 * For a hook that keeps its own error class: an HTTP error becomes `make(error)`,
 * anything else (a network, parse or abort rejection) is rethrown untouched.
 */
export function rethrowHttp(error: unknown, make: (error: ApiError, status: number) => Error): never {
  if (error instanceof ApiError && error.kind === 'http') throw make(error, error.status ?? 0)
  throw error
}
