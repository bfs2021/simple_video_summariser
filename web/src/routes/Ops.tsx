import { useEffect, useId, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { JobProgress } from '../components/JobProgress'
import { OpsError, useOpsChannels, useOpsJobs, useQueueDepth, useRetryJob, type JobOut, type OpsChannelOut } from '../hooks/useOps'
import {
  collapseError,
  depthTable,
  ERROR_CLASSES,
  errorClassInfo,
  formatUtc,
  isDefaultFilters,
  JOB_KINDS,
  JOB_STATES,
  NO_CLASS,
  opsSearch,
  parseOpsParams,
  type OpsFilters,
} from '../lib/ops'

// Plain semantic HTML on purpose: styling is #132. Filters and the cursor live
// in the URL only; TanStack Query owns the server state (architecture §9).
// Everything shown is a React text node: stored text is never parsed as HTML.
export function Ops() {
  const [params, setParams] = useSearchParams()
  const { filters, before, corrected } = parseOpsParams(params)

  // Invalid parameters are dropped from the URL (replace, not push). This render
  // already ignores them, so nothing invalid is ever requested or shown.
  const fix = corrected === null ? null : corrected.toString()
  useEffect(() => {
    if (fix !== null) setParams(new URLSearchParams(fix), { replace: true })
  }, [fix, setParams])

  const depth = useQueueDepth()
  const channels = useOpsChannels()
  const jobs = useOpsJobs(filters, before)
  const [result, setResult] = useState<{ jobId: number; message: string } | null>(null)
  const depthId = useId()
  const channelsId = useId()
  const jobsId = useId()

  const go = (f: OpsFilters, cursor: number | null) => {
    setResult(null)
    const next = opsSearch(f, cursor)
    if (next !== params.toString()) setParams(new URLSearchParams(next))
  }

  const data = jobs.data
  const placeholder = jobs.isPlaceholderData
  const loadingJobs = jobs.isPending || (jobs.isFetching && placeholder)
  const rowVanished = result !== null && !(data?.items ?? []).some((j) => j.id === result.jobId)
  const table = depth.data === undefined ? null : depthTable(depth.data.queue)

  return (
    <section>
      <h1>Ops</h1>
      <p>
        <button
          type="button"
          onClick={() => {
            setResult(null)
            void depth.refetch()
            void channels.refetch()
            void jobs.refetch()
          }}
        >
          Refresh
        </button>
      </p>

      <section aria-labelledby={depthId}>
        <h2 id={depthId}>Queue depth</h2>
        {depth.isPending && <p>Loading queue depth…</p>}
        {depth.isError && (
          <p role="alert">
            {depth.error instanceof OpsError && depth.error.status === 503
              ? 'Queue depth unavailable: the database is down'
              : 'Could not load queue depth'}{' '}
            <button type="button" onClick={() => void depth.refetch()}>
              Retry
            </button>
          </p>
        )}
        {table !== null && (
          <>
            <p>{table.deadTotal === 0 ? 'No dead jobs' : `${table.deadTotal} dead ${table.deadTotal === 1 ? 'job' : 'jobs'}`}</p>
            <table>
              <thead>
                <tr>
                  <th scope="col">Kind</th>
                  {table.states.map((s) => (
                    <th key={s} scope="col">
                      {s}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.kinds.map((k) => (
                  <tr key={k}>
                    <th scope="row">{k}</th>
                    {table.states.map((s) => (
                      <td key={s}>{table.count(k, s)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>

      <section aria-labelledby={channelsId}>
        <h2 id={channelsId}>Channels needing attention</h2>
        {channels.isPending && <p role="status">Loading channels…</p>}
        {channels.isError && (
          <p role="alert">
            {channels.error instanceof OpsError && channels.error.status === 503
              ? 'The database is unavailable. Try again shortly.'
              : 'Could not load channels'}{' '}
            <button type="button" onClick={() => void channels.refetch()}>
              Retry
            </button>
          </p>
        )}
        {channels.data !== undefined && channels.data.items.length === 0 && <p>No channels need attention</p>}
        {channels.data !== undefined && channels.data.items.length > 0 && (
          <div role="region" aria-label="Channels needing attention table" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th scope="col">Channel</th>
                  <th scope="col">Conditions</th>
                  <th scope="col">Detected</th>
                  <th scope="col">Guidance</th>
                </tr>
              </thead>
              <tbody>{channels.data.items.map((channel) => <ChannelRow key={channel.channel_id} channel={channel} />)}</tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby={jobsId}>
        <h2 id={jobsId}>Jobs</h2>
        <Filters filters={filters} onChange={(f) => go(f, null)} />
        <p aria-live="polite">{rowVanished ? result.message : ''}</p>
        {loadingJobs && <p>Loading jobs…</p>}
        {jobs.isError && (
          <p role="alert">
            {jobs.error instanceof OpsError && jobs.error.status === 503
              ? 'The database is unavailable. Try again shortly.'
              : 'Could not load jobs'}{' '}
            <button type="button" onClick={() => void jobs.refetch()}>
              Retry
            </button>
          </p>
        )}
        {data !== undefined && (
          <>
            {data.items.length === 0 && !placeholder && (
              <p>
                {isDefaultFilters(filters) ? 'No dead jobs' : 'No jobs match these filters'}{' '}
                {!isDefaultFilters(filters) && (
                  <button
                    type="button"
                    onClick={() => go({ state: 'dead', kind: null, errorClass: null }, null)}
                  >
                    Reset filters
                  </button>
                )}
              </p>
            )}
            {data.items.length > 0 && (
              <table>
                <thead>
                  <tr>
                    <th scope="col">Job</th>
                    <th scope="col">Video</th>
                    <th scope="col">Kind</th>
                    <th scope="col">State</th>
                    <th scope="col">Progress</th>
                    <th scope="col">Attempts</th>
                    <th scope="col">Error class</th>
                    <th scope="col">Time</th>
                    <th scope="col">Last error</th>
                    <th scope="col">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((j) => (
                    <JobRow
                      key={j.id}
                      job={j}
                      message={result?.jobId === j.id ? result.message : ''}
                      onResult={(jobId, message) => setResult({ jobId, message })}
                      onAction={() => setResult(null)}
                    />
                  ))}
                </tbody>
              </table>
            )}
            <nav aria-label="Pagination">
              <button
                type="button"
                disabled={data.next_before_id === null || placeholder}
                onClick={() => data.next_before_id !== null && go(filters, data.next_before_id)}
              >
                Older
              </button>{' '}
              <button type="button" disabled={before === null || placeholder} onClick={() => go(filters, null)}>
                Newest
              </button>
            </nav>
          </>
        )}
      </section>
    </section>
  )
}

const GAP_COPY = {
  suspected: 'Possible feed gap awaiting reconciliation',
  no_anchor: 'Feed gap could not be recovered: no safe anchor',
  listing_window_exhausted: 'Feed gap may continue beyond the reconciliation window',
} as const

function ChannelRow({ channel }: { channel: OpsChannelOut }) {
  const gap = channel.feed_gap_reason
  return (
    <tr>
      <th scope="row">{channel.title ?? channel.channel_id}</th>
      <td>
        {channel.poll_failing_since !== null && <p>RSS polling failure ({channel.poll_failures})</p>}
        {gap !== null && <p>{GAP_COPY[gap]}</p>}
      </td>
      <td>
        {channel.poll_failing_since !== null && <time dateTime={channel.poll_failing_since}>{formatUtc(channel.poll_failing_since)}</time>}
        {channel.poll_failing_since !== null && channel.feed_gap_detected_at !== null && ' · '}
        {channel.feed_gap_detected_at !== null && <time dateTime={channel.feed_gap_detected_at}>{formatUtc(channel.feed_gap_detected_at)}</time>}
      </td>
      <td>
        {gap !== null && (
          <>
            <p>Run a channel backfill to recover older uploads</p>
            <code>POST /api/channels/{channel.channel_id}/backfill</code>
          </>
        )}
      </td>
    </tr>
  )
}

function Filters({ filters, onChange }: { filters: OpsFilters; onChange: (f: OpsFilters) => void }) {
  const stateId = useId()
  const kindId = useId()
  const classId = useId()
  return (
    <form onSubmit={(e) => e.preventDefault()}>
      <label htmlFor={stateId}>State</label>{' '}
      <select
        id={stateId}
        value={filters.state}
        onChange={(e) => {
          const state = JOB_STATES.find((s) => s === e.target.value)
          if (state !== undefined) onChange({ ...filters, state })
        }}
      >
        {JOB_STATES.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>{' '}
      <label htmlFor={kindId}>Kind</label>{' '}
      <select
        id={kindId}
        value={filters.kind ?? ''}
        onChange={(e) => onChange({ ...filters, kind: JOB_KINDS.find((k) => k === e.target.value) ?? null })}
      >
        <option value="">Any</option>
        {JOB_KINDS.map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>{' '}
      <label htmlFor={classId}>Error class</label>{' '}
      <select
        id={classId}
        value={filters.errorClass ?? ''}
        onChange={(e) => {
          const v = e.target.value
          onChange({
            ...filters,
            errorClass: v === NO_CLASS ? NO_CLASS : (ERROR_CLASSES.find((c) => c === v) ?? null),
          })
        }}
      >
        <option value="">Any</option>
        {ERROR_CLASSES.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
        <option value={NO_CLASS}>No class</option>
      </select>
    </form>
  )
}

function JobRow({
  job: j,
  message,
  onResult,
  onAction,
}: {
  job: JobOut
  message: string
  onResult: (jobId: number, message: string) => void
  onAction: () => void
}) {
  const [confirming, setConfirming] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const retry = useRetryJob(onResult)
  const retryRef = useRef<HTMLButtonElement>(null)
  const confirmRef = useRef<HTMLButtonElement>(null)
  const restoreFocus = useRef(false)
  // Set synchronously on click: isPending only turns true on a later tick.
  const sent = useRef(false)

  // Focus follows the confirmation: into it when it opens, back out on Cancel.
  useEffect(() => {
    if (confirming) confirmRef.current?.focus()
    else if (restoreFocus.current) {
      restoreFocus.current = false
      retryRef.current?.focus()
    }
  }, [confirming])

  const cls = errorClassInfo(j.error_class)
  const collapsed = collapseError(j.last_error)
  const finished = j.finished_at !== null
  const videoName = j.video_title ?? j.video_id
  const busy = retry.isPending

  const confirm = () => {
    if (sent.current) return
    sent.current = true
    onAction()
    retry.mutate(j.id, {
      onSettled: () => {
        sent.current = false
        setConfirming(false)
      },
    })
  }

  return (
    <tr>
      <th scope="row">{j.id}</th>
      <td>
        <Link to={`/videos/${encodeURIComponent(j.video_id)}`}>{videoName}</Link>
      </td>
      <td>{j.kind}</td>
      <td>{j.state}</td>
      <td>
        <JobProgress kind={j.kind} state={j.state} progress={j.progress ?? null} />
      </td>
      <td>{j.attempts}</td>
      <td>
        <span>{cls.label}</span>
        {cls.hint !== null && (
          <>
            {' '}
            <small>{cls.hint}</small>
          </>
        )}
      </td>
      <td>
        <span>{formatUtc(finished ? j.finished_at : j.created_at)}</span> <small>{finished ? 'finished' : 'created'}</small>
      </td>
      <td>
        {collapsed === null ? (
          '—'
        ) : (
          <>
            {expanded ? (
              <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxWidth: '100%', overflow: 'auto' }}>
                {j.last_error}
              </pre>
            ) : (
              <span>{collapsed}</span>
            )}{' '}
            <button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
              {expanded ? 'Hide' : 'Show full error'}
            </button>
          </>
        )}
      </td>
      <td>
        {j.state === 'dead' &&
          (confirming ? (
            <div>
              <p>
                Retry job #{j.id} ({j.kind}) for {videoName}?
                {j.kind === 'transcribe' && ' A transcription can take hours of CPU.'}
              </p>
              <button type="button" ref={confirmRef} disabled={busy} onClick={confirm}>
                Confirm retry
              </button>{' '}
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  restoreFocus.current = true
                  setConfirming(false)
                }}
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              type="button"
              ref={retryRef}
              onClick={() => {
                onAction()
                setConfirming(true)
              }}
            >
              Retry
            </button>
          ))}
        <span aria-live="polite">{message}</span>
      </td>
    </tr>
  )
}
