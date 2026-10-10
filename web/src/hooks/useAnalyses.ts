import { useQuery } from '@tanstack/react-query'
import type { components } from '../api'
import { api, rethrowHttp } from '../api'

export type AnalysisRun = components['schemas']['AnalysisOut']
type AnalysisList = components['schemas']['AnalysisList']

/** #42 caps `limit` at 50; four pages bound the walk at 200 runs. */
export const PAGE_SIZE = 50
export const MAX_PAGES = 4

/** A non-2xx response. Carries only the status: the body is never surfaced. */
export class AnalysesApiError extends Error {
  readonly status: number
  constructor(status: number) {
    super(`HTTP ${status}`)
    this.name = 'AnalysesApiError'
    this.status = status
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

const isNumberOrNull = (v: unknown) => v === null || typeof v === 'number'
const isStringOrNull = (v: unknown) => v === null || typeof v === 'string'

/** The fields the view cannot render without; the rest is typed by the schema. */
function isRun(v: unknown): v is AnalysisRun {
  return (
    isRecord(v) &&
    typeof v.id === 'number' &&
    typeof v.transcript_id === 'number' &&
    typeof v.model === 'string' &&
    typeof v.prompt_version === 'string' &&
    typeof v.chunk_strategy === 'string' &&
    typeof v.transcript_source === 'string' &&
    typeof v.created_at === 'string' &&
    typeof v.tldr === 'string' &&
    typeof v.input_tokens === 'number' &&
    typeof v.output_tokens === 'number' &&
    isNumberOrNull(v.cost_usd) &&
    isNumberOrNull(v.duration_ms) &&
    Array.isArray(v.topics) &&
    v.topics.every(
      (t) => isRecord(t) && typeof t.title === 'string' && isStringOrNull(t.summary) && isNumberOrNull(t.start_sec) && typeof t.seq === 'number',
    ) &&
    Array.isArray(v.claims) &&
    v.claims.every(
      (c) =>
        isRecord(c) &&
        typeof c.text === 'string' &&
        typeof c.speaker === 'string' &&
        isStringOrNull(c.confidence) &&
        isNumberOrNull(c.start_sec),
    ) &&
    Array.isArray(v.quotes) &&
    v.quotes.every(
      (q) => isRecord(q) && typeof q.text === 'string' && typeof q.speaker === 'string' && isNumberOrNull(q.start_sec),
    )
  )
}

function isList(v: unknown): v is AnalysisList {
  return isRecord(v) && typeof v.total === 'number' && Array.isArray(v.items) && v.items.every(isRun)
}

async function fetchPage(videoId: string, offset: number, limit: number, signal: AbortSignal): Promise<AnalysisList> {
  const body: unknown = await api
    .get('/videos/{video_id}/analyses', {
      params: { path: { video_id: videoId }, query: { limit, offset } },
      signal,
    })
    .catch((error: unknown) => rethrowHttp(error, (_e, status) => new AnalysesApiError(status)))
  if (!isList(body)) throw new Error('Unexpected response shape')
  return body
}

export interface Analyses {
  /** Newest first, as the API returns them; each id once. */
  runs: AnalysisRun[]
  total: number
  /** Runs exist beyond the 200 that were loaded. */
  capped: boolean
}

async function walkPages(videoId: string, signal: AbortSignal): Promise<Analyses> {
  const runs: AnalysisRun[] = []
  const seen = new Set<number>()
  let total = 0
  let capped = false
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const offset = page * PAGE_SIZE
    const body = await fetchPage(videoId, offset, PAGE_SIZE, signal)
    total = body.total
    for (const run of body.items) {
      if (seen.has(run.id)) continue
      seen.add(run.id)
      runs.push(run)
    }
    // An empty page before `total` means the data shifted: stop, don't loop.
    if (body.items.length === 0 || offset + body.items.length >= total) break
    capped = page === MAX_PAGES - 1
  }
  return { runs, total, capped }
}

/**
 * Every analysis run of one video (at most 200), newest first. `videoId` must
 * already have passed `parseVideoId`; `null` makes no request. The key
 * carries the id, so one video's runs never answer another's. No polling:
 * runs are immutable, a new one appears on reload or Retry.
 */
export function useAnalyses(videoId: string | null) {
  return useQuery({
    queryKey: ['analyses', videoId],
    queryFn: ({ signal }) => walkPages(videoId as string, signal),
    enabled: videoId !== null,
  })
}

/**
 * The number of runs, from one `limit=1` request. Kept under its own key,
 * apart from the full walk, for the "Compare analyses" link on the Video
 * detail view.
 */
export function useAnalysisTotal(videoId: string | null) {
  return useQuery({
    queryKey: ['analyses-total', videoId],
    queryFn: async ({ signal }) => {
      const { total } = await fetchPage(videoId as string, 0, 1, signal)
      if (!Number.isInteger(total) || total < 0) throw new Error('Unexpected response shape')
      return total
    },
    enabled: videoId !== null,
  })
}
