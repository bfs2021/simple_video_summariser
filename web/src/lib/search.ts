// Pure helpers for the Search view (#52): query validation, URL-parameter
// parsing (the URL is untrusted input), the GET /search query, paging maths and
// the excerpt render model. No React. The server (#43) stays authoritative;
// these checks only avoid sending requests that are certain to be rejected.
import type { components, operations } from '../api'

type SearchPage = components['schemas']['SearchPage']
type Span = components['schemas']['SpanOut']
type Excerpt = components['schemas']['ExcerptOut']

/** Fixed page size: #43's default `limit`. */
export const SEARCH_PAGE_SIZE = 20
/** #43 rejects an `offset` above this. */
export const MAX_SEARCH_OFFSET = 1000
/** #43 rejects a trimmed `q` longer than this many characters (Python `str` length). */
export const MAX_QUERY_CHARS = 200
/** The last page whose offset is within the cap: (51 - 1) * 20 = 1000. */
export const MAX_PAGE = MAX_SEARCH_OFFSET / SEARCH_PAGE_SIZE + 1

export type QueryCheck =
  | { ok: true; q: string }
  | { ok: false; reason: 'empty' | 'too_long' | 'nul' }

/** Trim, then reject empty, NUL and over-long. Length is in code points, not UTF-16 units. */
export function validateQuery(raw: string): QueryCheck {
  const q = raw.trim()
  if (q === '') return { ok: false, reason: 'empty' }
  if (q.includes('\u0000')) return { ok: false, reason: 'nul' }
  if ([...q].length > MAX_QUERY_CHARS) return { ok: false, reason: 'too_long' }
  return { ok: true, q }
}

export function offsetForPage(page: number): number {
  return (page - 1) * SEARCH_PAGE_SIZE
}

/** Next needs more results AND a next offset inside #43's cap (page 51 is the last). */
export function nextEnabled(page: number, hasMore: boolean): boolean {
  return hasMore && offsetForPage(page + 1) <= MAX_SEARCH_OFFSET
}

// ---- URL state ------------------------------------------------------------

export interface SearchUrlState {
  /** A validated, trimmed query, or null for the idle state. */
  q: string | null
  page: number
  /** The URL with invalid `q`/`page` fixed (unknown keys untouched), or null if nothing changed. */
  corrected: URLSearchParams | null
}

const PAGE = /^[1-9]\d{0,2}$/

export function parseSearchParams(params: URLSearchParams): SearchUrlState {
  const next = new URLSearchParams(params)
  let changed = false

  let q: string | null = null
  const rawQ = params.get('q')
  if (rawQ !== null) {
    const check = validateQuery(rawQ)
    if (check.ok) {
      q = check.q
      if (check.q !== rawQ || params.getAll('q').length > 1) {
        next.set('q', check.q)
        changed = true
      }
    } else {
      next.delete('q')
      changed = true
    }
  }

  let page = 1
  const rawPage = params.get('page')
  if (rawPage !== null) {
    const n = PAGE.test(rawPage) ? Number(rawPage) : 0
    if (n >= 1 && n <= MAX_PAGE) {
      page = n
    } else {
      next.delete('page')
      changed = true
    }
  }

  return { q, page, corrected: changed ? next : null }
}

/** The view's own URL query string (no leading `?`); page 1 is left out. */
export function searchHref(q: string, page: number): string {
  const p = new URLSearchParams({ q })
  if (page > 1) p.set('page', String(page))
  return p.toString()
}

/** The GET /search query, typed from the schema. `q` is encoded by the request helper only. */
export function searchQuery(q: string, page: number): NonNullable<operations['get_search_search_get']['parameters']['query']> {
  return { q, limit: SEARCH_PAGE_SIZE, offset: offsetForPage(page) }
}

/** The same query as a query string. The request helper takes `searchQuery` directly. */
export function buildSearchRequest(q: string, page: number): URLSearchParams {
  return new URLSearchParams(Object.entries(searchQuery(q, page)).map(([k, v]) => [k, String(v)]))
}

// ---- response -------------------------------------------------------------

/** The most seconds a hit link carries; #43 stores segment starts up to 1e9. */
export const MAX_LINK_SEC = 1_000_000_000

/** `/videos/{id}?t=<whole seconds>` for a usable segment start; null otherwise. */
export function timestampHref(videoId: string, startSec: number | null): string | null {
  if (typeof startSec !== 'number' || !Number.isFinite(startSec)) return null
  if (startSec < 0 || startSec > MAX_LINK_SEC) return null
  return `/videos/${encodeURIComponent(videoId)}?t=${Math.floor(startSec)}`
}

export interface ExcerptFragment {
  spans: Span[]
  /** The excerpt's start in seconds, or null when it has no usable one. */
  startSec: number | null
}

/**
 * Drop empty spans, then excerpts with nothing left. Order and whitespace are
 * kept. A negative or non-finite `start_sec` is treated as absent.
 */
export function excerptFragments(excerpts: Excerpt[]): ExcerptFragment[] {
  return excerpts
    .map((e) => ({
      spans: e.spans.filter((s) => s.text !== ''),
      startSec:
        typeof e.start_sec === 'number' && Number.isFinite(e.start_sec) && e.start_sec >= 0
          ? e.start_sec
          : null,
    }))
    .filter((e) => e.spans.length > 0)
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function isStringOrNull(v: unknown): boolean {
  return v === null || typeof v === 'string'
}

function isSpan(v: unknown): boolean {
  return isObject(v) && typeof v.text === 'string' && typeof v.match === 'boolean'
}

function isExcerpt(v: unknown): boolean {
  return (
    isObject(v) &&
    Array.isArray(v.spans) &&
    v.spans.every(isSpan) &&
    (v.start_sec === null || typeof v.start_sec === 'number')
  )
}

function isResult(v: unknown): boolean {
  return (
    isObject(v) &&
    typeof v.video_id === 'string' &&
    isStringOrNull(v.title) &&
    isStringOrNull(v.channel_id) &&
    isStringOrNull(v.channel_title) &&
    isStringOrNull(v.published_at) &&
    isStringOrNull(v.unavailable) &&
    Array.isArray(v.excerpts) &&
    v.excerpts.every(isExcerpt)
  )
}

/** Runtime check of the fields the view reads; a failure is shown as a generic error. */
export function isSearchPage(v: unknown): v is SearchPage {
  return (
    isObject(v) &&
    typeof v.has_more === 'boolean' &&
    typeof v.offset === 'number' &&
    typeof v.limit === 'number' &&
    Array.isArray(v.results) &&
    v.results.every(isResult)
  )
}
