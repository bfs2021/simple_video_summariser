import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { JobProgress, type Progress } from './JobProgress'

const NOW = Date.parse('2026-10-23T12:00:00Z')

function figures(over: Partial<Progress> = {}): Progress {
  return {
    percent: 25,
    done_sec: 300,
    duration_sec: 1200,
    elapsed_sec: 600,
    eta_sec: 1800,
    estimated_sec: 1500,
    estimate_source: 'history',
    updated_at: new Date(NOW - 10_000).toISOString(),
    ...over,
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW)
})
afterEach(() => {
  vi.useRealTimers()
})

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('a running transcribe job', () => {
  it('shows the percent, a native progress element, the elapsed time and the ETA', () => {
    const { container } = render(<JobProgress kind="transcribe" state="running" progress={figures()} />)
    expect(screen.getByText('25%')).toBeTruthy()
    const bar = container.querySelector('progress') as HTMLProgressElement
    expect(bar.value).toBe(25)
    expect(bar.max).toBe(100)
    expect(screen.getByText('Elapsed 10 min')).toBeTruthy()
    expect(screen.getByText('about 30 min left')).toBeTruthy()
  })

  it('labels an ETA that rests on the default RTF as rough, and a measured one not', () => {
    const { rerender } = render(
      <JobProgress kind="transcribe" state="running" progress={figures({ estimate_source: 'default' })} />,
    )
    expect(screen.getByText('about 30 min left (rough)')).toBeTruthy()
    rerender(<JobProgress kind="transcribe" state="running" progress={figures({ estimate_source: 'history' })} />)
    expect(screen.queryByText(/rough/)).toBeNull()
  })

  it('shows an unknown percent as an indeterminate bar and no ETA', () => {
    const { container } = render(
      <JobProgress kind="transcribe" state="running" progress={figures({ percent: null, eta_sec: null })} />,
    )
    expect((container.querySelector('progress') as HTMLProgressElement).hasAttribute('value')).toBe(false)
    expect(screen.queryByText(/left/)).toBeNull()
  })

  it('is empty without progress', () => {
    const { container } = render(<JobProgress kind="transcribe" state="running" progress={null} />)
    expect(container.innerHTML).toBe('')
  })
})

describe('a pending transcribe job', () => {
  it('shows the estimated duration, rough when it is the default', () => {
    const p = figures({ percent: null, done_sec: null, elapsed_sec: null, updated_at: null, estimate_source: 'default' })
    const { container } = render(<JobProgress kind="transcribe" state="pending" progress={p} />)
    expect(screen.getByText('estimated about 25 min (rough)')).toBeTruthy()
    expect(container.querySelector('progress')).toBeNull()
  })

  it('shows nothing when there is no estimate', () => {
    const p = figures({ estimated_sec: null, estimate_source: null })
    const { container } = render(<JobProgress kind="transcribe" state="pending" progress={p} />)
    expect(container.innerHTML).toBe('')
  })
})

describe('every other job', () => {
  it.each([
    ['ingest', 'running'],
    ['analyze', 'pending'],
    ['transcribe', 'done'],
    ['transcribe', 'dead'],
  ])('a %s job in state %s shows nothing', (kind, state) => {
    const { container } = render(<JobProgress kind={kind} state={state} progress={figures()} />)
    expect(container.innerHTML).toBe('')
  })
})

describe('stuck detection', () => {
  const at = (ageMs: number) => figures({ updated_at: new Date(NOW - ageMs).toISOString() })

  it('shows no warning one millisecond under 90 seconds', () => {
    render(<JobProgress kind="transcribe" state="running" progress={at(89_999)} />)
    expect(screen.queryByText(/no update/)).toBeNull()
    expect(screen.getByText('about 30 min left')).toBeTruthy()
  })

  it('replaces the ETA with "no update for N min" at exactly 90 seconds', () => {
    render(<JobProgress kind="transcribe" state="running" progress={at(90_000)} />)
    const warning = screen.getByRole('status')
    expect(warning.textContent).toBe('no update for 1 min')
    expect(screen.queryByText(/left/)).toBeNull()
  })

  it('notices on its own, without new data, once the figures are old enough', async () => {
    render(<JobProgress kind="transcribe" state="running" progress={at(60_000)} />)
    expect(screen.queryByText(/no update/)).toBeNull()
    await advance(40_000)
    expect(screen.getByText('no update for 1 min')).toBeTruthy()
    await advance(120_000)
    expect(screen.getByText('no update for 3 min')).toBeTruthy()
  })

  it('goes away when fresh figures arrive', () => {
    const { rerender } = render(<JobProgress kind="transcribe" state="running" progress={at(200_000)} />)
    expect(screen.getByText('no update for 3 min')).toBeTruthy()
    rerender(<JobProgress kind="transcribe" state="running" progress={at(5_000)} />)
    expect(screen.queryByText(/no update/)).toBeNull()
  })

  it('never judges a pending job', () => {
    const p = figures({ updated_at: null, percent: null })
    render(<JobProgress kind="transcribe" state="pending" progress={p} />)
    expect(screen.queryByText(/no update/)).toBeNull()
  })
})

describe('announcements', () => {
  const live = (c: HTMLElement) => c.querySelector('[aria-live="polite"]') as HTMLElement

  it('keeps one polite region that says the percent and the time left', () => {
    const { container } = render(<JobProgress kind="transcribe" state="running" progress={figures()} />)
    expect(live(container).textContent).toBe('25 percent done, about 30 min left')
  })

  it('speaks a new figure at most once a minute, however often figures arrive', async () => {
    const { container, rerender } = render(
      <JobProgress kind="transcribe" state="running" progress={figures({ percent: 25 })} />,
    )
    const spoken = () => live(container).textContent
    await advance(30_000)
    rerender(<JobProgress kind="transcribe" state="running" progress={figures({ percent: 30, updated_at: new Date(Date.now()).toISOString() })} />)
    await advance(10_000)
    expect(spoken()).toBe('25 percent done, about 30 min left')
    await advance(20_000) // a minute since the first one
    expect(spoken()).toBe('30 percent done, about 30 min left')

    rerender(<JobProgress kind="transcribe" state="running" progress={figures({ percent: 35, updated_at: new Date(Date.now()).toISOString() })} />)
    await advance(30_000)
    expect(spoken()).toBe('30 percent done, about 30 min left')
    await advance(30_000)
    expect(spoken()).toBe('35 percent done, about 30 min left')
  })
})
