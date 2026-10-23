import { useEffect } from 'react'
import { Link, useSearchParams } from 'react-router'
import { AddVideoForm } from '../components/AddVideoForm'
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
  type SortKey,
  type SortOrder,
  type VideoSort,
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
    setParams(toSearchParams({ ...state, page: 1, ...patch }))
  }

  const sortBy = (sort: VideoSort) => {
    const index = state.sorts.findIndex((pair) => pair.sort === sort)
    if (index < 0) go({ sorts: [...state.sorts, { sort, order: defaultOrder(sort) }] })
    else go({ sorts: state.sorts.map((pair, i) => (i === index ? { ...pair, order: opposite(pair.order) } : pair)) })
  }

  const moveSort = (index: number, delta: -1 | 1) => {
    const sorts = [...state.sorts]
    const other = index + delta
    ;[sorts[index], sorts[other]] = [sorts[other] as SortKey, sorts[index] as SortKey]
    go({ sorts })
  }

  const removeSort = (index: number) => {
    const sorts = state.sorts.filter((_pair, i) => i !== index)
    go({ sorts: sorts.length === 0 ? [{ sort: 'published', order: 'desc' }] : sorts })
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

      <AddVideoForm />

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
          <button
            className="btn"
            type="button"
            onClick={() => setParams(toSearchParams({ ...state, channel: null, from: null, to: null, status: null, page: 1 }))}
          >
            Clear all filters
          </button>
        )}
      </form>

      <div className="card" role="group" aria-label="Sort order">
        <strong>Sort order</strong>
        <ol>
          {state.sorts.map((pair, index) => (
            <li key={pair.sort}>
              {index + 1} {sortLabel(pair.sort)} {direction(pair.order)}{' '}
              <button
                className="btn"
                type="button"
                aria-label={`Move ${sortLabel(pair.sort)} earlier`}
                disabled={index === 0}
                onClick={() => moveSort(index, -1)}
              >
                Earlier
              </button>{' '}
              <button
                className="btn"
                type="button"
                aria-label={`Move ${sortLabel(pair.sort)} later`}
                disabled={index === state.sorts.length - 1}
                onClick={() => moveSort(index, 1)}
              >
                Later
              </button>{' '}
              <button
                className="btn"
                type="button"
                aria-label={`Remove ${sortLabel(pair.sort)} from sort order`}
                onClick={() => removeSort(index)}
              >
                Remove
              </button>
            </li>
          ))}
        </ol>
        {!isDefaultSort(state.sorts) && (
          <button
            className="btn"
            type="button"
            aria-label="Reset sort to Published descending"
            onClick={() => go({ sorts: [{ sort: 'published', order: 'desc' }] })}
          >
            Reset sort
          </button>
        )}
      </div>

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
                    <SortHeader label="Title" sort="title" state={state} onSort={sortBy} />
                    <SortHeader label="Channel" sort="channel" state={state} onSort={sortBy} />
                    <SortHeader label="Published" sort="published" state={state} onSort={sortBy} />
                    <SortHeader label="Status" sort="status" state={state} onSort={sortBy} />
                    <th scope="col">Transcript</th>
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
                        <td>
                          {v.has_transcript ? (
                            <Link to={`/videos/${v.video_id}/transcript`}>Transcript</Link>
                          ) : (
                            <span className="muted">Not yet</span>
                          )}
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

interface SortHeaderProps {
  label: string
  sort: VideoSort
  state: Pick<LibraryState, 'sorts'>
  onSort: (sort: VideoSort) => void
}

function SortHeader({ label, sort, state, onSort }: SortHeaderProps) {
  const index = state.sorts.findIndex((pair) => pair.sort === sort)
  const pair = index < 0 ? null : state.sorts[index] as SortKey
  const action = pair === null ? defaultOrder(sort) : opposite(pair.order)
  return (
    <th scope="col" aria-sort={index === 0 && pair !== null ? direction(pair.order) : undefined}>
      <button
        className="btn"
        type="button"
        onClick={() => onSort(sort)}
        aria-label={pair === null ? `Append ${label} ${direction(action)}` : `Toggle ${label} to ${direction(action)}`}
      >
        {label}
        {pair !== null ? ` ${index + 1} ${direction(pair.order)}` : ''}
      </button>
    </th>
  )
}

function sortLabel(sort: VideoSort): string {
  return sort.charAt(0).toUpperCase() + sort.slice(1)
}

function isDefaultSort(sorts: readonly SortKey[]): boolean {
  return sorts.length === 1 && sorts[0]?.sort === 'published' && sorts[0].order === 'desc'
}

function defaultOrder(sort: VideoSort): SortOrder {
  return sort === 'published' ? 'desc' : 'asc'
}

function opposite(order: SortOrder): SortOrder {
  return order === 'asc' ? 'desc' : 'asc'
}

function direction(order: SortOrder): 'ascending' | 'descending' {
  return order === 'asc' ? 'ascending' : 'descending'
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
