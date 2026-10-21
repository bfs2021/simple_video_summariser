// Pure helpers for the Library view (#49): the status label, URL-parameter
// parsing (the URL is untrusted input) and the GET /videos query. No React.
import type { components, operations } from '../api'

type VideoItem = components['schemas']['VideoItem']
type VideoQuery = NonNullable<operations['get_videos_videos_get']['parameters']['query']>
export type VideoStatus = VideoItem['status']
export type VideoSort = NonNullable<VideoQuery['sort']>[number]
export type SortOrder = NonNullable<VideoQuery['order']>[number]
export interface SortKey {
  sort: VideoSort
  order: SortOrder
}

export const PAGE_SIZE = 50
// #42 rejects an offset above this.
export const MAX_OFFSET = 1_000_000

export const STATUSES: readonly VideoStatus[] = ['done', 'processing', 'failed', 'unavailable', 'idle']
export const SORTS: readonly VideoSort[] = ['title', 'published', 'status', 'channel']
export const ORDERS: readonly SortOrder[] = ['asc', 'desc']

// ---- status label -------------------------------------------------------

export interface StatusLabel {
  label: string
  /** Secondary text, e.g. the failure's error class. */
  detail: string | null
}

const STAGE_NOUN = new Map([
  ['ingest', 'download'],
  ['transcribe', 'transcription'],
  ['analyze', 'analysis'],
])

function stageNoun(kind: string | undefined): string | null {
  return kind === undefined ? null : (STAGE_NOUN.get(kind) ?? null)
}

function capitalise(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

/** The label for a row. Consumes the server's `status`; never re-derives it. */
export function statusLabel(v: Pick<VideoItem, 'status' | 'active_job' | 'last_failure'>): StatusLabel {
  const job = v.active_job
  switch (v.status) {
    case 'done':
      return { label: job ? 'Analysed · re-analysing' : 'Analysed', detail: null }
    case 'processing': {
      const stage = stageNoun(job?.kind)
      if (stage === null || job === null) return { label: 'Processing', detail: null }
      if (job.state === 'pending') return { label: `Queued: ${stage}`, detail: null }
      if (job.state === 'running') return { label: `${capitalise(stage)} in progress`, detail: null }
      return { label: 'Processing', detail: null }
    }
    case 'failed': {
      const stage = stageNoun(v.last_failure?.kind)
      return {
        label: stage === null ? 'Failed' : `Failed at ${stage}`,
        detail: v.last_failure?.error_class ?? null,
      }
    }
    case 'unavailable':
      return { label: 'Unavailable on YouTube', detail: null }
    case 'idle':
      return { label: 'Not processed', detail: null }
    default:
      return { label: 'Unknown status', detail: null }
  }
}

/** Label for a status filter option. */
export function statusFilterLabel(status: VideoStatus): string {
  switch (status) {
    case 'done':
      return 'Analysed'
    case 'processing':
      return 'Processing'
    case 'failed':
      return 'Failed'
    case 'unavailable':
      return 'Unavailable on YouTube'
    case 'idle':
      return 'Not processed'
  }
}

// ---- dates (UTC calendar days, no date library) -----------------------------

const DATE = /^\d{4}-\d{2}-\d{2}$/

/** A real `YYYY-MM-DD` calendar day (year 1000-9999). */
export function isValidDate(s: string): boolean {
  if (!DATE.test(s) || s < '1000-01-01') return false
  const t = Date.parse(`${s}T00:00:00Z`)
  return !Number.isNaN(t) && new Date(t).toISOString().slice(0, 10) === s
}

/** The next UTC day, or null when it would leave four-digit years. */
export function addOneDay(s: string): string | null {
  const next = new Date(Date.parse(`${s}T00:00:00Z`) + 86_400_000).toISOString()
  return DATE.test(next.slice(0, 10)) ? next.slice(0, 10) : null
}

/** `published_at` as a UTC `YYYY-MM-DD`, or null if absent or unparseable. */
export function publishedDate(publishedAt: string | null): string | null {
  if (publishedAt === null) return null
  const t = Date.parse(publishedAt)
  if (Number.isNaN(t)) return null
  const day = new Date(t).toISOString().slice(0, 10)
  return DATE.test(day) ? day : null
}

// ---- URL state ------------------------------------------------------------

export interface LibraryState {
  channel: string | null
  from: string | null
  to: string | null
  status: VideoStatus | null
  sorts: SortKey[]
  page: number
}

export const EMPTY_STATE: LibraryState = {
  channel: null,
  from: null,
  to: null,
  status: null,
  sorts: [{ sort: 'published', order: 'desc' }],
  page: 1,
}

const CHANNEL_ID = /^UC[A-Za-z0-9_-]{22}$/
const WHOLE_NUMBER = /^[1-9][0-9]*$/

export type LibraryKey = 'channel' | 'from' | 'to' | 'status' | 'sort' | 'order' | 'page'

export interface ParsedParams {
  state: LibraryState
  /** Known keys that were present but rejected; the caller drops them from the URL. */
  invalid: LibraryKey[]
}

function isStatus(s: string): s is VideoStatus {
  return (STATUSES as readonly string[]).includes(s)
}

function isSort(s: string): s is VideoSort {
  return (SORTS as readonly string[]).includes(s)
}

function isOrder(s: string): s is SortOrder {
  return (ORDERS as readonly string[]).includes(s)
}

function isPage(s: string): boolean {
  if (!WHOLE_NUMBER.test(s)) return false
  const n = Number(s)
  return Number.isSafeInteger(n) && (n - 1) * PAGE_SIZE <= MAX_OFFSET
}

/** Validate every known parameter; unknown ones are ignored, never forwarded. */
export function parseLibraryParams(params: URLSearchParams): ParsedParams {
  const state: LibraryState = { ...EMPTY_STATE }
  const invalid: LibraryKey[] = []

  const read = (key: LibraryKey, ok: (v: string) => boolean): string | null => {
    const all = params.getAll(key)
    if (all.length === 0) return null
    const v = all[0] as string
    if (all.length > 1 || !ok(v)) {
      invalid.push(key)
      return null
    }
    return v
  }

  state.channel = read('channel', (v) => CHANNEL_ID.test(v))
  state.from = read('from', isValidDate)
  state.to = read('to', isValidDate)
  const status = read('status', isStatus)
  state.status = status === null ? null : (status as VideoStatus)
  const sorts = params.getAll('sort')
  const orders = params.getAll('order')
  const sortInvalid =
    sorts.length > 4 ||
    orders.length > 4 ||
    sorts.some((value) => !isSort(value)) ||
    orders.some((value) => !isOrder(value)) ||
    new Set(sorts).size !== sorts.length ||
    ((sorts.length > 1 || orders.length > 1) && sorts.length !== orders.length)
  if (sortInvalid) {
    invalid.push('sort', 'order')
  } else if (sorts.length <= 1 && orders.length <= 1) {
    state.sorts = [
      {
        sort: (sorts[0] ?? 'published') as VideoSort,
        order: (orders[0] ?? 'desc') as SortOrder,
      },
    ]
  } else {
    state.sorts = sorts.map((sort, index) => ({ sort: sort as VideoSort, order: orders[index] as SortOrder }))
  }
  const page = read('page', isPage)
  state.page = page === null ? 1 : Number(page)
  return { state, invalid }
}

/** The URL form of a state: unset values and page 1 are omitted. */
export function toSearchParams(state: LibraryState): URLSearchParams {
  const p = new URLSearchParams()
  if (state.channel !== null) p.set('channel', state.channel)
  if (state.from !== null) p.set('from', state.from)
  if (state.to !== null) p.set('to', state.to)
  if (state.status !== null) p.set('status', state.status)
  if (!isDefaultSort(state.sorts)) {
    for (const pair of state.sorts) {
      p.append('sort', pair.sort)
      p.append('order', pair.order)
    }
  }
  if (state.page > 1) p.set('page', String(state.page))
  return p
}

function isDefaultSort(sorts: readonly SortKey[]): boolean {
  return sorts.length === 1 && sorts[0]?.sort === 'published' && sorts[0].order === 'desc'
}

export function hasFilters(state: LibraryState): boolean {
  return state.channel !== null || state.from !== null || state.to !== null || state.status !== null
}

/** True when both bounds are set and From is after To. */
export function isReversedRange(state: LibraryState): boolean {
  return state.from !== null && state.to !== null && state.from > state.to
}

/** The `GET /videos` query for a state, typed from the schema. Unset parameters are omitted. */
export function videosQuery(
  state: LibraryState,
): NonNullable<operations['get_videos_videos_get']['parameters']['query']> {
  const q: NonNullable<operations['get_videos_videos_get']['parameters']['query']> = {}
  if (state.channel !== null) q.channel = state.channel
  if (state.status !== null) q.status = state.status
  if (state.from !== null) q.published_after = state.from
  if (state.to !== null) {
    // The UI's "To" is inclusive; #42's published_before is exclusive.
    const before = addOneDay(state.to)
    if (before !== null) q.published_before = before
  }
  if (!isDefaultSort(state.sorts)) {
    q.sort = state.sorts.map((pair) => pair.sort)
    q.order = state.sorts.map((pair) => pair.order)
  }
  q.limit = PAGE_SIZE
  q.offset = (state.page - 1) * PAGE_SIZE
  return q
}

/** The same query as a query string. The request helper takes `videosQuery` directly. */
export function buildVideosQuery(state: LibraryState): URLSearchParams {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(videosQuery(state))) {
    for (const item of Array.isArray(value) ? value : [value]) params.append(key, String(item))
  }
  return params
}
