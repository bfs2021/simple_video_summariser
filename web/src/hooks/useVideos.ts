import { keepPreviousData, useQuery } from '@tanstack/react-query'
import type { components } from '../api'
import { api, rethrowHttp } from '../api'
import { videosQuery, type LibraryState } from '../lib/library'

export type VideoList = components['schemas']['VideoList']

/** Rows with an active job change under the user; refetch at this interval. */
export const POLL_INTERVAL_MS = 15_000

/** A non-2xx response. Carries only the status: the body is never surfaced. */
export class ApiError extends Error {
  readonly status: number
  constructor(status: number) {
    super(`HTTP ${status}`)
    this.name = 'ApiError'
    this.status = status
  }
}

function isVideoList(v: unknown): v is VideoList {
  if (typeof v !== 'object' || v === null) return false
  const o = v as Record<string, unknown>
  return Array.isArray(o.items) && typeof o.total === 'number'
}

async function fetchVideos(query: ReturnType<typeof videosQuery>, signal: AbortSignal): Promise<VideoList> {
  const body: unknown = await api
    .get('/videos', { params: { query }, signal })
    .catch((error: unknown) => rethrowHttp(error, (_e, status) => new ApiError(status)))
  if (!isVideoList(body)) throw new Error('Unexpected response shape')
  return body
}

/**
 * One page of GET /api/videos. The key carries every filter and the page, so
 * each combination is cached separately. While `enabled` is false (an invalid
 * date range) nothing is requested and the previous results stay in place.
 */
export function useVideos(state: LibraryState, enabled: boolean) {
  const query = videosQuery(state)
  return useQuery({
    queryKey: ['videos', query],
    queryFn: ({ signal }) => fetchVideos(query, signal),
    enabled,
    placeholderData: keepPreviousData,
    // Poll only while a visible row is still moving through the pipeline.
    refetchInterval: (q) =>
      q.state.data?.items.some((item) => item.active_job !== null) ? POLL_INTERVAL_MS : false,
  })
}
