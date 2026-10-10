import { useQuery } from '@tanstack/react-query'
import type { components } from '../api'
import { api, rethrowHttp } from '../api'

export type VideoDetailData = components['schemas']['VideoDetail']

/** Refetch interval while a job is active for the video. */
export const POLL_INTERVAL_MS = 15_000

/** A non-2xx response. Carries only the status: the body is never surfaced. */
export class VideoApiError extends Error {
  readonly status: number
  constructor(status: number) {
    super(`HTTP ${status}`)
    this.name = 'VideoApiError'
    this.status = status
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

/** The few fields the view cannot render without; the rest is typed by the schema. */
function isVideoDetail(v: unknown): v is VideoDetailData {
  if (!isRecord(v) || typeof v.status !== 'string' || !('analysis' in v)) return false
  const a = v.analysis
  if (a === null) return true
  return (
    isRecord(a) &&
    typeof a.tldr === 'string' &&
    Array.isArray(a.topics) &&
    Array.isArray(a.claims) &&
    Array.isArray(a.quotes)
  )
}

async function fetchVideo(videoId: string, signal: AbortSignal): Promise<VideoDetailData> {
  const body: unknown = await api
    .get('/videos/{video_id}', { params: { path: { video_id: videoId } }, signal })
    .catch((error: unknown) => rethrowHttp(error, (_e, status) => new VideoApiError(status)))
  if (!isVideoDetail(body)) throw new Error('Unexpected response shape')
  return body
}

/**
 * GET /api/videos/{id}. Pass the id only after `parseVideoId`; `null` makes no
 * request. The key carries the id, so one video's data never answers another.
 */
export function useVideo(videoId: string | null) {
  return useQuery({
    queryKey: ['video', videoId],
    queryFn: ({ signal }) => fetchVideo(videoId as string, signal),
    enabled: videoId !== null,
    refetchInterval: (q) => (q.state.data?.active_job != null ? POLL_INTERVAL_MS : false),
  })
}
