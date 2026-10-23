// Pure helpers for the Ops view (#54): the shared kind/state/error-class sets,
// URL-parameter parsing (the URL is untrusted input), request building, the
// error-class hints, the retry-result messages and response shape guards.
// No React. The server (#39, #44) stays authoritative; nothing here re-derives
// queue state, it only avoids requests that are certain to be rejected.
import type { components, operations } from '../api'

type Query = NonNullable<operations['get_ops_jobs_ops_jobs_get']['parameters']['query']>
export type JobOut = components['schemas']['JobOut']
export type JobListOut = components['schemas']['JobListOut']
export type HealthOut = components['schemas']['HealthOut']
export type OpsChannelOut = components['schemas']['OpsChannelOut']
export type OpsChannelListOut = components['schemas']['OpsChannelListOut']

/** The one place the sets and their order live: the depth table and the filters both use it. */
export const JOB_KINDS = ['ingest', 'transcribe', 'analyze'] as const satisfies readonly NonNullable<Query['kind']>[]
export const JOB_STATES = ['pending', 'running', 'done', 'dead'] as const satisfies readonly NonNullable<Query['state']>[]
/** The eight `common.errors.ErrorClass` values (architecture §8.4). */
export const ERROR_CLASSES = [
  'PERMANENT_SOURCE',
  'TRANSIENT_NETWORK',
  'RATE_LIMITED',
  'TOOL_FAILURE',
  'LLM_INVALID_OUTPUT',
  'LLM_UNAVAILABLE',
  'RESOURCE',
  'BUG',
] as const satisfies readonly NonNullable<Query['error_class']>[]
/** Sent as `error_class=none`: jobs with a NULL class (for example a reaped job). */
export const NO_CLASS = 'none' as const satisfies NonNullable<Query['error_class']>

export type JobKind = (typeof JOB_KINDS)[number]
export type JobState = (typeof JOB_STATES)[number]
export type ErrorClassFilter = (typeof ERROR_CLASSES)[number] | typeof NO_CLASS

export const DEFAULT_STATE: JobState = 'dead'
/** Fixed page size (#44's default is 50 too). */
export const OPS_PAGE_SIZE = 50
/** Longest collapsed `last_error`, in code points, ellipsis included. */
export const COLLAPSED_ERROR_CHARS = 200

function oneOf<T extends string>(set: readonly T[], v: string): v is T {
  return (set as readonly string[]).includes(v)
}

export function isPositiveSafeInt(v: unknown): v is number {
  return typeof v === 'number' && Number.isSafeInteger(v) && v >= 1
}

// ---- URL state ------------------------------------------------------------

export interface OpsFilters {
  state: JobState
  kind: JobKind | null
  errorClass: ErrorClassFilter | null
}

export interface OpsUrlState {
  filters: OpsFilters
  /** The keyset cursor (`before`), or null for the first page. */
  before: number | null
  /** The URL with invalid ops parameters dropped (others untouched), or null if nothing changed. */
  corrected: URLSearchParams | null
}

const BEFORE = /^[1-9][0-9]*$/

/**
 * A parameter is used only if it is present exactly once with an allowed value.
 * Anything else (wrong case, empty, repeated, out of range) is dropped.
 */
export function parseOpsParams(params: URLSearchParams): OpsUrlState {
  const next = new URLSearchParams(params)
  let changed = false

  function pick<T>(name: string, accept: (v: string) => T | null): T | null {
    if (!params.has(name)) return null
    const all = params.getAll(name)
    const first = all[0] as string
    const value = all.length === 1 ? accept(first) : null
    if (value === null) {
      next.delete(name)
      changed = true
    }
    return value
  }

  const state = pick('state', (v) => (oneOf(JOB_STATES, v) ? v : null)) ?? DEFAULT_STATE
  const kind = pick('kind', (v) => (oneOf(JOB_KINDS, v) ? v : null))
  const errorClass = pick('error_class', (v) =>
    oneOf(ERROR_CLASSES, v) || v === NO_CLASS ? (v as ErrorClassFilter) : null,
  )
  const before = pick('before', (v) => {
    if (!BEFORE.test(v)) return null
    const n = Number(v)
    return Number.isSafeInteger(n) ? n : null
  })

  return { filters: { state, kind, errorClass }, before, corrected: changed ? next : null }
}

export function isDefaultFilters(f: OpsFilters): boolean {
  return f.state === DEFAULT_STATE && f.kind === null && f.errorClass === null
}

/** The view's own URL query string (no leading `?`); defaults are left out. */
export function opsSearch(f: OpsFilters, before: number | null): string {
  const p = new URLSearchParams()
  if (f.state !== DEFAULT_STATE) p.set('state', f.state)
  if (f.kind !== null) p.set('kind', f.kind)
  if (f.errorClass !== null) p.set('error_class', f.errorClass)
  if (before !== null) p.set('before', String(before))
  return p.toString()
}

/** The GET /ops/jobs query, typed from the schema. `state` is always sent: the API's default is every state. */
export function jobsQuery(f: OpsFilters, before: number | null): Query {
  const q: Query = { state: f.state }
  if (f.kind !== null) q.kind = f.kind
  if (f.errorClass !== null) q.error_class = f.errorClass
  if (before !== null) q.before_id = before
  q.limit = OPS_PAGE_SIZE
  return q
}

/** The same query as a query string. The request helper takes `jobsQuery` directly. */
export function buildJobsRequest(f: OpsFilters, before: number | null): URLSearchParams {
  return new URLSearchParams(Object.entries(jobsQuery(f, before)).map(([k, v]) => [k, String(v)]))
}

// ---- display --------------------------------------------------------------

const HINTS = new Map<string, string>([
  ['PERMANENT_SOURCE', 'Video removed, private or blocked. Retry only if it is back'],
  ['TRANSIENT_NETWORK', 'Network error'],
  ['RATE_LIMITED', 'Rate-limited or bot-checked by YouTube'],
  ['TOOL_FAILURE', 'yt-dlp is probably broken. Update yt-dlp, then retry'],
  ['LLM_INVALID_OUTPUT', 'The model returned invalid output'],
  ['LLM_UNAVAILABLE', 'The LLM provider was unreachable'],
  ['RESOURCE', 'Disk full or out of memory. Fix that before retrying'],
  ['BUG', 'Unhandled exception. See the traceback'],
])

/** The code (or "No class") plus a one-line hint. An unknown string is shown as-is, with no hint. */
export function errorClassInfo(value: string | null): { label: string; hint: string | null } {
  if (value === null) return { label: 'No class', hint: 'Often a lost worker (stale heartbeat)' }
  return { label: value, hint: HINTS.get(value) ?? null }
}

/** `YYYY-MM-DD HH:MM UTC`; null is a dash; a value that is not a date stays as text. */
export function formatUtc(iso: string | null): string {
  if (iso === null) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return `${d.toISOString().slice(0, 16).replace('T', ' ')} UTC`
}

/**
 * The collapsed `last_error`: its first non-blank line, at most 200 code points
 * with the ellipsis counted. Null when there is nothing to show.
 */
export function collapseError(text: string | null): string | null {
  if (text === null || text.trim() === '') return null
  const line = text.split(/\r\n|\r|\n/).find((l) => l.trim() !== '') as string
  const chars = [...line]
  if (chars.length <= COLLAPSED_ERROR_CHARS) return line
  return chars.slice(0, COLLAPSED_ERROR_CHARS - 1).join('') + '…'
}

// ---- retry ----------------------------------------------------------------

/** The path (relative to the API base) for retrying a job. Throws for an id that is not a positive safe integer. */
export function retryPath(id: number): string {
  if (!isPositiveSafeInt(id)) throw new Error('Invalid job id')
  return `ops/jobs/${id}/retry`
}

export type RetryOutcome =
  | { type: 'network' }
  | { type: 'http'; status: number; body: unknown }

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

/** The fixed message for a retry result. The raw response body is never shown. */
export function retryMessage(jobId: number, outcome: RetryOutcome): string {
  if (outcome.type === 'network') return 'Retry failed'
  switch (outcome.status) {
    case 200:
      return `Queued again: job #${jobId} is pending`
    case 404:
      return 'This job no longer exists'
    case 429:
      return 'Too many requests. Try again shortly.'
    case 503:
      return 'The database is unavailable. Try again shortly.'
    case 409: {
      const body = outcome.body
      if (isObject(body) && body.detail === 'job is not dead') {
        const s = body.state
        return typeof s === 'string' && oneOf(JOB_STATES, s) ? `Job is already ${s}` : 'Job is no longer dead'
      }
      if (isObject(body) && body.detail === 'superseded by a newer job') {
        return isPositiveSafeInt(body.job_id)
          ? `A newer job #${body.job_id} exists for this video`
          : 'A newer job exists for this video'
      }
      return 'The job could not be retried'
    }
    default:
      return 'Retry failed'
  }
}

// ---- queue depth ----------------------------------------------------------

export interface DepthTable {
  kinds: string[]
  states: string[]
  count: (kind: string, state: string) => number
  deadTotal: number
}

function own(o: object, key: string): boolean {
  return Object.hasOwn(o, key)
}

/** Known kinds and states first, in order; unknown ones after, in response order. A missing cell is 0. */
export function depthTable(queue: HealthOut['queue']): DepthTable {
  const kinds: string[] = [...JOB_KINDS]
  const states: string[] = [...JOB_STATES]
  for (const [kind, row] of Object.entries(queue)) {
    if (!kinds.includes(kind)) kinds.push(kind)
    for (const state of Object.keys(row)) if (!states.includes(state)) states.push(state)
  }
  const count = (kind: string, state: string): number => {
    if (!own(queue, kind)) return 0
    const row = queue[kind] as Record<string, number>
    return own(row, state) ? (row[state] as number) : 0
  }
  const deadTotal = kinds.reduce((sum, k) => sum + count(k, 'dead'), 0)
  return { kinds, states, count, deadTotal }
}

// ---- response guards ------------------------------------------------------

function isCount(v: unknown): boolean {
  return typeof v === 'number' && Number.isSafeInteger(v) && v >= 0
}

export function isOpsChannelList(v: unknown): v is OpsChannelListOut {
  return (
    isObject(v) &&
    Array.isArray(v.items) &&
    v.items.every(
      (item) =>
        isObject(item) &&
        typeof item.channel_id === 'string' &&
        isStringOrNull(item.title) &&
        isStringOrNull(item.poll_failing_since) &&
        isCount(item.poll_failures) &&
        isStringOrNull(item.last_poll_err) &&
        isStringOrNull(item.feed_gap_detected_at) &&
        (item.feed_gap_reason === null ||
          item.feed_gap_reason === 'suspected' ||
          item.feed_gap_reason === 'no_anchor' ||
          item.feed_gap_reason === 'listing_window_exhausted'),
    )
  )
}

export function isHealth(v: unknown): v is HealthOut {
  if (!isObject(v) || typeof v.status !== 'string' || !isObject(v.queue)) return false
  return Object.values(v.queue).every((row) => isObject(row) && Object.values(row).every(isCount))
}

function isStringOrNull(v: unknown): boolean {
  return v === null || typeof v === 'string'
}

function isJob(v: unknown): v is JobOut {
  return (
    isObject(v) &&
    isPositiveSafeInt(v.id) &&
    typeof v.video_id === 'string' &&
    isStringOrNull(v.video_title) &&
    typeof v.kind === 'string' &&
    typeof v.state === 'string' &&
    isCount(v.attempts) &&
    isStringOrNull(v.error_class) &&
    isStringOrNull(v.last_error) &&
    isStringOrNull(v.finished_at) &&
    typeof v.created_at === 'string' &&
    (v.progress === undefined || v.progress === null || isObject(v.progress))
  )
}

export function isJobList(v: unknown): v is JobListOut {
  return (
    isObject(v) &&
    Array.isArray(v.items) &&
    v.items.every(isJob) &&
    (v.next_before_id === null || isPositiveSafeInt(v.next_before_id))
  )
}
