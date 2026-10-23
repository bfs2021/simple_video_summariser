import { describe, expect, it } from 'vitest'
import {
  ANNOUNCE_EVERY_MS,
  formatDuration,
  formatRemaining,
  PROGRESS_POLL_MS,
  STUCK_AFTER_MS,
  stuckMinutes,
} from './progress'

describe('constants', () => {
  it('poll every 30 s, call a job stuck after 3 ticks, announce once a minute', () => {
    expect(PROGRESS_POLL_MS).toBe(30_000)
    expect(STUCK_AFTER_MS).toBe(3 * PROGRESS_POLL_MS)
    expect(ANNOUNCE_EVERY_MS).toBe(60_000)
  })
})

describe('formatDuration', () => {
  it.each([
    [0, '0 s'],
    [45, '45 s'],
    [60, '1 min'],
    [125, '2 min 5 s'],
    [3600, '1 h'],
    [3723, '1 h 2 min 3 s'],
    [44.6, '45 s'],
  ])('%d s is %s', (sec, text) => {
    expect(formatDuration(sec)).toBe(text)
  })

  it('treats a negative or non-finite value as zero', () => {
    expect(formatDuration(-5)).toBe('0 s')
    expect(formatDuration(Number.NaN)).toBe('0 s')
  })
})

describe('formatRemaining', () => {
  it.each([
    [0, 'less than a minute'],
    [29, 'less than a minute'],
    [30, 'about 1 min'],
    [720, 'about 12 min'],
    [3600, 'about 1 h'],
    [3900, 'about 1 h 5 min'],
  ])('%d s is %s', (sec, text) => {
    expect(formatRemaining(sec)).toBe(text)
  })
})

describe('stuckMinutes', () => {
  const now = Date.parse('2026-10-23T12:00:00Z')
  const ago = (ms: number) => new Date(now - ms).toISOString()

  it('is null just under 90 seconds and for a fresh update', () => {
    expect(stuckMinutes(ago(89_999), now)).toBeNull()
    expect(stuckMinutes(ago(0), now)).toBeNull()
  })

  it('is 1 at exactly 90 seconds', () => {
    expect(stuckMinutes(ago(90_000), now)).toBe(1)
  })

  it('counts whole minutes after that', () => {
    expect(stuckMinutes(ago(119_000), now)).toBe(1)
    expect(stuckMinutes(ago(5 * 60_000 + 1), now)).toBe(5)
  })

  it('is null without a usable timestamp, and for a timestamp in the future', () => {
    expect(stuckMinutes(null, now)).toBeNull()
    expect(stuckMinutes('not a date', now)).toBeNull()
    expect(stuckMinutes(ago(-60_000), now)).toBeNull()
  })
})
