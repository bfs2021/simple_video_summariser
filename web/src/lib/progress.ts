// Pure helpers for the transcription progress shown on the Ops job list and the
// video page (#245). No React. The server owns the figures; this only formats
// them and decides whether they have stopped advancing.

/** The worker stores fresh figures this often; the Ops list refetches this often while a job runs. */
export const PROGRESS_POLL_MS = 30_000
/** A running job whose figures are this old (3 ticks) is shown as stalled. */
export const STUCK_AFTER_MS = 3 * PROGRESS_POLL_MS
/** The percent and ETA are announced to assistive technology at most this often. */
export const ANNOUNCE_EVERY_MS = 60_000

/** `1 h 2 min 3 s`; zero parts are left out and a lone zero is `0 s`. Never negative. */
export function formatDuration(sec: number): string {
  const total = Number.isFinite(sec) && sec > 0 ? Math.round(sec) : 0
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const parts = [h > 0 ? `${h} h` : '', m > 0 ? `${m} min` : '', s > 0 ? `${s} s` : ''].filter((p) => p !== '')
  return parts.length === 0 ? '0 s' : parts.join(' ')
}

/** A time left or an estimated length, rounded to the minute: `about 12 min`, `about 1 h 5 min`. */
export function formatRemaining(sec: number): string {
  const minutes = Number.isFinite(sec) && sec > 0 ? Math.round(sec / 60) : 0
  if (minutes < 1) return 'less than a minute'
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return `about ${[h > 0 ? `${h} h` : '', m > 0 ? `${m} min` : ''].filter((p) => p !== '').join(' ')}`
}

/**
 * Whole minutes (at least 1) since `updatedAt` when that is at least 90 s before
 * `nowMs`, else null: the figures are still advancing, or there is nothing to judge.
 */
export function stuckMinutes(updatedAt: string | null, nowMs: number): number | null {
  if (updatedAt === null) return null
  const at = Date.parse(updatedAt)
  if (Number.isNaN(at)) return null
  const age = nowMs - at
  if (age < STUCK_AFTER_MS) return null
  return Math.max(1, Math.floor(age / 60_000))
}
