import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, rethrowHttp } from '../api'
import { apiUrl } from '../api/base'
import {
  jobsQuery,
  isHealth,
  isJobList,
  retryMessage,
  retryPath,
  type HealthOut,
  type JobListOut,
  type OpsFilters,
  type RetryOutcome,
} from '../lib/ops'

export type { HealthOut, JobListOut, JobOut } from '../lib/ops'

/** The queue depth is re-read this often while the Ops view is mounted. */
export const QUEUE_DEPTH_POLL_MS = 15_000

/** A non-2xx response. Carries only the status: the body is never surfaced. */
export class OpsError extends Error {
  readonly status: number
  constructor(status: number) {
    super(`HTTP ${status}`)
    this.name = 'OpsError'
    this.status = status
  }
}

/** Queue depth by kind and state (GET /api/healthz), refetched every 15 s while mounted. */
export function useQueueDepth() {
  return useQuery({
    queryKey: ['ops', 'queue-depth'],
    queryFn: async ({ signal }): Promise<HealthOut> => {
      const body: unknown = await api
        .get('/healthz', { signal })
        .catch((error: unknown) => rethrowHttp(error, (_e, status) => new OpsError(status)))
      if (!isHealth(body)) throw new Error('Unexpected response shape')
      return body
    },
    refetchInterval: QUEUE_DEPTH_POLL_MS,
  })
}

/** One page of GET /api/ops/jobs. The key carries every filter and the cursor. */
export function useOpsJobs(filters: OpsFilters, before: number | null) {
  return useQuery({
    queryKey: ['ops', 'jobs', filters.state, filters.kind, filters.errorClass, before],
    queryFn: async ({ signal }): Promise<JobListOut> => {
      const body: unknown = await api
        .get('/ops/jobs', { params: { query: jobsQuery(filters, before) }, signal })
        .catch((error: unknown) => rethrowHttp(error, (_e, status) => new OpsError(status)))
      if (!isJobList(body)) throw new Error('Unexpected response shape')
      return body
    },
    placeholderData: keepPreviousData,
  })
}

/**
 * POST /api/ops/jobs/{id}/retry. Never retried automatically. Every HTTP status
 * (and a network failure) is turned into a fixed message and handed to
 * `onResult`, which also runs if the row that started it has since vanished.
 * After any HTTP response the job list and the depth are refetched; after a
 * network failure nothing is, since nothing is known to have changed.
 */
export function useRetryJob(onResult: (jobId: number, message: string) => void) {
  const client = useQueryClient()
  return useMutation({
    retry: 0,
    mutationFn: async (id: number): Promise<RetryOutcome> => {
      const path = retryPath(id) // throws for an id that is not a positive safe integer
      // A network failure rejects: the mutation errors, with no refetch.
      const res = await fetch(apiUrl(path), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      })
      let body: unknown
      try {
        body = await res.json()
      } catch {
        body = undefined
      }
      return { type: 'http', status: res.status, body }
    },
    onSuccess: async (outcome, id) => {
      onResult(id, retryMessage(id, outcome))
      await client.invalidateQueries({ queryKey: ['ops'] })
    },
    onError: (_error, id) => onResult(id, retryMessage(id, { type: 'network' })),
  })
}
