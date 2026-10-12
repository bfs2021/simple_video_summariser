import { useEffect } from 'react'
import { Link, useSearchParams } from 'react-router'
import { ApiError, useVideos } from '../hooks/useVideos'
import {
  hasFilters,
  isReversedRange,
  isValidDate,
  MAX_OFFSET,
  PAGE_SIZE,
  parseLibraryParams,
  publishedDate,
  statusFilterLabel,
  statusLabel,
  STATUSES,
  toSearchParams,
  type LibraryState,
  type VideoStatus,
} from '../lib/library'

// Semantic HTML first; styling is class hooks into web/index.css (design-system
// §11, #120). Filter and page state lives in the URL only; TanStack Query owns
// the server state (architecture §9).
export function Library() {
  const [params, setParams] = useSearchParams()
  const { state, invalid } = parseLibraryParams(params)

  // An invalid parameter is dropped from the URL (replace, not push). This
  // render already ignores it, so nothing invalid is ever requested.
  const invalidKeys = invalid.join(',')
  useEffect(() => {
    if (invalidKeys === '') return
    const next = new URLSearchParams(params)
    for (const key of invalidKeys.split(',')) next.delete(key)
    setParams(next, { replace: true })
  }, [invalidKeys, params, setParams])

  const reversed = isReversedRange(state)
  const q = useVideos(state, !reversed)

  const go = (patch: Partial<LibraryState>) => {
    // Any filter change returns to page 1; a page change passes `page` itself.
    setParams(toSearchParams({ ...state, page: 1, ...patch }))
  }

  const data = q.data
  const offset = (state.page - 1) * PAGE_SIZE
  const switching = q.isPlaceholderData
  const pastEnd = data !== undefined && data.total > 0 && data.items.length === 0
  const lastPage = data === undefined ? 1 : Math.min(Math.ceil(data.total / PAGE_SIZE), MAX_OFFSET / PAGE_SIZE + 1)
  const hasNext =
    data !== undefined && offset + data.items.length < data.total && state.page * PAGE_SIZE <= MAX_OFFSET

  const channelTitle =
    state.channel === null
      ? null
      : (data?.items.find((v) => v.channel_id === state.channel && v.channel_title !== null)
          ?.channel_title ?? null)

  return (
    <section className="page">
      <h1>Library</h1>

      <form className="card filters" onSubmit={(e) => e.preventDefault()}>
        <label className="field">
          From{' '}
          <input
            className="input"
            type="date"
            value={state.from ?? ''}
            onChange={(e) => onDate('from', e.target.value)}
          />
        </label>
        <label className="field">
          To{' '}
          <input
            className="input"
            type="date"
            value={state.to ?? ''}
            onChange={(e) => onDate('to', e.target.value)}
          />
        </label>
        <label className="field">
          Status{' '}
          <select
            className="input"
            value={state.status ?? ''}
            onChange={(e) => go({ status: (e.target.value || null) as VideoStatus | null })}
          >
            <option value="">Any</option>
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {statusFilterLabel(s)}
              </option>
            ))}
          </select>
        </label>
        {hasFilters(state) && (
          <button className="btn" type="button" onClick={() => setParams(new URLSearchParams())}>
            Clear all filters
          </button>
        )}
      </form>

      {reversed && (
        <p className="alert alert--danger" role="alert">
          'From' is after 'To'
        </p>
      )}

      {state.channel !== null && (
        <div className="row" role="group" aria-label="Active filters">
          Channel: <strong>{channelTitle ?? state.channel}</strong>{' '}
          <button className="btn" type="button" onClick={() => go({ channel: null })}>
            Clear
          </button>
        </div>
      )}

      {q.isError ? (
        <div className="alert alert--danger" role="alert">
          <p>
            {q.error instanceof ApiError && q.error.status === 503
              ? 'The database is unavailable. Try again shortly.'
              : 'Could not load videos'}
          </p>
          <button className="btn btn--primary" type="button" onClick={() => void q.refetch()}>
            Retry
          </button>
        </div>
      ) : (
        <>
          {(q.isFetching && data === undefined) || switching ? (
            <p className="muted" role="status">
              Loading…
            </p>
          ) : null}
          <p className="muted" role="status" aria-label="Results" aria-live="polite">
            {data !== undefined && data.items.length > 0
              ? `${offset + 1}–${offset + data.items.length} of ${data.total}`
              : ''}
          </p>

          {data !== undefined && data.total === 0 && !hasFilters(state) && (
            <p>Nothing has been processed yet</p>
          )}
          {data !== undefined && data.total === 0 && hasFilters(state) && (
            <p>No videos match these filters</p>
          )}
          {pastEnd && (
            <p>
              <span>No videos on this page</span>{' '}
              <Link to={{ search: toSearchParams({ ...state, page: lastPage }).toString() }}>
                Go to the last page
              </Link>
            </p>
          )}

          {data !== undefined && data.items.length > 0 && (
            <div className="card card--flush table-scroll" role="region" aria-label="Videos" tabIndex={0}>
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Title</th>
                    <th scope="col">Channel</th>
                    <th scope="col">Published</th>
                    <th scope="col">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((v) => {
                    const status = statusLabel(v)
                    const id = v.channel_id
                    return (
                      <tr key={v.video_id}>
                        <td>
                          <Link to={`/videos/${v.video_id}`}>{v.title ?? v.video_id}</Link>
                        </td>
                        <td>
                          {id !== null ? (
                            <button className="btn" type="button" onClick={() => go({ channel: id })}>
                              {v.channel_title ?? id}
                            </button>
                          ) : (
                            (v.channel_title ?? 'Unknown channel')
                          )}
                        </td>
                        <td>{publishedDate(v.published_at) ?? 'Date unknown'}</td>
                        <td>
                          <span className={`badge badge--${statusTone(v.status)}`}>
                            <span>{status.label}</span>
                            {status.detail !== null && (
                              <>
                                {' '}
                                (<span>{status.detail}</span>)
                              </>
                            )}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          <nav className="pagination" aria-label="Pagination">
            <button
              className="btn"
              type="button"
              disabled={state.page <= 1 || switching || reversed}
              onClick={() => go({ page: state.page - 1 })}
            >
              Previous
            </button>
            <button
              className="btn btn--primary"
              type="button"
              disabled={!hasNext || switching || reversed}
              onClick={() => go({ page: state.page + 1 })}
            >
              Next
            </button>
          </nav>
        </>
      )}
    </section>
  )

  function onDate(key: 'from' | 'to', value: string) {
    if (value === '') go({ [key]: null })
    else if (isValidDate(value)) go({ [key]: value })
  }
}

type StatusTone = 'success' | 'warning' | 'danger' | 'neutral'

// Badge colour per video status; unknown and idle states stay neutral.
function statusTone(status: VideoStatus): StatusTone {
  switch (status) {
    case 'done':
      return 'success'
    case 'processing':
      return 'warning'
    case 'failed':
      return 'danger'
    default:
      return 'neutral'
  }
}
