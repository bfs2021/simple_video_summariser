import { keepPreviousData, useQuery } from '@tanstack/react-query'
import type { components } from '../api'
import { api, rethrowHttp } from '../api'
import { isSearchPage, searchQuery } from '../lib/search'

export type SearchPage = components['schemas']['SearchPage']

/** A non-2xx response. Carries only the status: the body is never surfaced. */
export class SearchError extends Error {
  readonly status: number
  constructor(status: number) {
    super(`HTTP ${status}`)
    this.name = 'SearchError'
    this.status = status
  }
}

async function fetchSearch(q: string, page: number, signal: AbortSignal): Promise<SearchPage> {
  const body: unknown = await api
    .get('/search', { params: { query: searchQuery(q, page) }, signal })
    .catch((error: unknown) => rethrowHttp(error, (_e, status) => new SearchError(status)))
  if (!isSearchPage(body)) throw new Error('Unexpected response shape')
  return body
}

/**
 * One page of GET /api/search. `q` is a validated query or null; null means the
 * idle state and nothing is requested. The key carries both q and the page, so
 * a late response for an older query or page can only fill its own cache entry.
 */
export function useSearch(q: string | null, page: number) {
  return useQuery({
    queryKey: ['search', q, page],
    queryFn: ({ signal }) => {
      if (q === null) throw new Error('Search ran without a query')
      return fetchSearch(q, page, signal)
    },
    enabled: q !== null,
    placeholderData: keepPreviousData,
  })
}
