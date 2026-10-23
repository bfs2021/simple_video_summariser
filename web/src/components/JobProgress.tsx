import { useEffect, useRef, useState } from 'react'
import type { components } from '../api'
import { ANNOUNCE_EVERY_MS, formatDuration, formatRemaining, stuckMinutes } from '../lib/progress'

export type Progress = components['schemas']['ProgressOut']

/** How often a running job re-checks whether its figures have stopped advancing. */
const CLOCK_TICK_MS = 10_000

// Transcription progress for the Ops job list and the video page (#245).
// Only a transcribe job shows any: a running one its percent, a native
// <progress>, the elapsed time and the ETA; a pending one the estimated length.
// Everything is a React text child; plain HTML on purpose, styling is #124.
export function JobProgress({
  kind,
  state,
  progress,
}: {
  kind: string
  state: string
  progress: Progress | null
}) {
  if (kind !== 'transcribe' || progress === null) return null
  if (state === 'pending') {
    if (progress.estimated_sec == null) return null
    return <span>{`estimated ${formatRemaining(progress.estimated_sec)}${rough(progress)}`}</span>
  }
  if (state !== 'running') return null
  return <Running progress={progress} />
}

const rough = (p: Progress): string => (p.estimate_source === 'default' ? ' (rough)' : '')

function useNow(everyMs: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), everyMs)
    return () => clearInterval(id)
  }, [everyMs])
  return now
}

function Running({ progress: p }: { progress: Progress }) {
  const now = useNow(CLOCK_TICK_MS)
  const stuck = stuckMinutes(p.updated_at ?? null, now)
  const left = stuck === null && p.eta_sec != null ? `${formatRemaining(p.eta_sec)} left` : null
  const spoken = [p.percent == null ? null : `${p.percent} percent done`, left].filter((x) => x !== null).join(', ')

  return (
    <div>
      {p.percent != null && <span>{p.percent}%</span>}{' '}
      <progress max={100} value={p.percent ?? undefined} aria-label="Transcription progress" />
      {p.elapsed_sec != null && (
        <>
          <br />
          <small>{`Elapsed ${formatDuration(p.elapsed_sec)}`}</small>
        </>
      )}
      {left !== null && (
        <>
          <br />
          <small>{`${left}${rough(p)}`}</small>
        </>
      )}
      {stuck !== null && (
        <>
          <br />
          <strong role="status">{`no update for ${stuck} min`}</strong>
        </>
      )}
      <Announcement text={spoken} />
    </div>
  )
}

/**
 * A polite live region that repeats `text` to assistive technology at most once per
 * `ANNOUNCE_EVERY_MS`, however often `text` changes. The visible figures update on
 * every poll; this is the only part a screen reader is told about, so it is
 * visually hidden (the standard clip pattern, not decoration).
 */
function Announcement({ text }: { text: string }) {
  const [spoken, setSpoken] = useState(text)
  const lastAt = useRef<number | null>(null)

  useEffect(() => {
    lastAt.current ??= Date.now()
    if (text === spoken) return
    const wait = Math.max(0, ANNOUNCE_EVERY_MS - (Date.now() - lastAt.current))
    const id = setTimeout(() => {
      lastAt.current = Date.now()
      setSpoken(text)
    }, wait)
    return () => clearTimeout(id)
  }, [text, spoken])

  return (
    <span
      aria-live="polite"
      style={{
        position: 'absolute',
        width: '1px',
        height: '1px',
        overflow: 'hidden',
        clip: 'rect(0 0 0 0)',
        whiteSpace: 'nowrap',
      }}
    >
      {spoken}
    </span>
  )
}
