import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { components } from '../api'
import { Ops } from './Ops'

type JobOut = components['schemas']['JobOut']
type JobListOut = components['schemas']['JobListOut']
type HealthOut = components['schemas']['HealthOut']
type OpsChannelListOut = components['schemas']['OpsChannelListOut']

function job(id: number, over: Partial<JobOut> = {}): JobOut {
  return {
    id,
    video_id: `v${String(id).padStart(10, '0')}`,
    video_title: `Title ${id}`,
    kind: 'ingest',
    dedupe_key: 'k',
    state: 'dead',
    priority: 0,
    attempts: 3,
    error_class: 'TOOL_FAILURE',
    last_error: `error ${id}`,
    run_after: '2026-10-01T12:00:00Z',
    locked_by: null,
    heartbeat_at: null,
    finished_at: '2026-10-01T23:30:00-07:00',
    created_at: '2026-09-30T08:05:00Z',
    ...over,
  }
}

function list(items: JobOut[], next: number | null = null): JobListOut {
  return { items, next_before_id: next }
}

function health(queue: HealthOut['queue'] = {}): HealthOut {
  return { status: 'ok', queue }
}

function channels(items: OpsChannelListOut['items'] = []): OpsChannelListOut {
  return { items }
}

interface Call {
  url: URL
  method: string
  headers: Headers
  body: string | null
}
let calls: Call[]
type Reply = unknown | Response | Promise<Response>
type Handler = (c: Call) => Reply

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

function stubFetch(h: Handler) {
  calls = []
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const c: Call = {
        url: new URL(String(input), 'http://localhost'),
        method: init?.method ?? 'GET',
        headers: new Headers(init?.headers),
        body: typeof init?.body === 'string' ? init.body : null,
      }
      calls.push(c)
      const r = h(c)
      if (r instanceof Promise) return r
      return Promise.resolve(r instanceof Response ? r : json(r))
    }),
  )
}

/** Routes by path: healthz, ops/jobs (GET) and retry (POST). */
function api(opts: { health?: Handler; channels?: Handler; jobs?: Handler; retry?: Handler } = {}): Handler {
  return (c) => {
    if (c.url.pathname === '/api/healthz') return (opts.health ?? (() => health()))(c)
    if (c.url.pathname === '/api/ops/channels') return (opts.channels ?? (() => channels()))(c)
    if (c.url.pathname === '/api/ops/jobs') return (opts.jobs ?? (() => list([])))(c)
    if (/^\/api\/ops\/jobs\/\d+\/retry$/.test(c.url.pathname)) {
      return (opts.retry ?? (() => json({ detail: 'job not found' }, 404)))(c)
    }
    throw new Error(`unexpected request ${c.method} ${c.url.pathname}`)
  }
}

const jobGets = () => calls.filter((c) => c.url.pathname === '/api/ops/jobs')
const healthGets = () => calls.filter((c) => c.url.pathname === '/api/healthz')
const posts = () => calls.filter((c) => c.method === 'POST')
const lastJobGet = () => jobGets()[jobGets().length - 1] as Call

function deferred() {
  let resolve!: (r: Response) => void
  const promise = new Promise<Response>((res) => (resolve = res))
  return { promise, resolve }
}

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

async function mount(entry = '/ops', defaults: { mutations?: { retry: number } } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, ...defaults } })
  const router = createMemoryRouter(
    [
      { path: '/ops', element: <Ops /> },
      { path: '/videos/:videoId', element: <p>detail page</p> },
    ],
    { initialEntries: [entry] },
  )
  const view = render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  await settle()
  return { router, view }
}

const depth = () => screen.getByRole('region', { name: 'Queue depth' })
const jobsRegion = () => screen.getByRole('region', { name: 'Jobs' })
const jobsTable = () => within(jobsRegion()).getByRole('table')
function rowFor(id: number): HTMLTableRowElement {
  const rows = within(jobsTable()).getAllByRole('row') as HTMLTableRowElement[]
  const row = rows.find((r) => r.cells[0]?.textContent === String(id))
  if (!row) throw new Error(`no row for job ${id}`)
  return row
}
const bodyRows = () => within(jobsTable()).getAllByRole('row').slice(1)
const click = async (el: HTMLElement) => {
  fireEvent.click(el)
  await settle()
}
const select = (name: RegExp | string) => screen.getByRole('combobox', { name }) as HTMLSelectElement
const livePolite = () => Array.from(document.querySelectorAll('[aria-live="polite"]'))
const liveText = () => livePolite().map((e) => e.textContent).join('|')

beforeEach(() => {
  vi.useFakeTimers()
})

describe('channels needing attention', () => {
  it('requests attention and renders both conditions with exact gap guidance', async () => {
    stubFetch(
      api({
        channels: () =>
          channels([
            {
              channel_id: 'UCabc',
              title: 'Channel title',
              poll_failing_since: '2026-10-02T00:00:00Z',
              poll_failures: 3,
              last_poll_err: 'failed',
              feed_gap_detected_at: '2026-10-01T00:00:00Z',
              feed_gap_reason: 'no_anchor',
            },
          ]),
      }),
    )
    await mount()
    const request = calls.find((c) => c.url.pathname === '/api/ops/channels') as Call
    expect(request.url.searchParams.get('attention')).toBe('true')
    const region = screen.getByRole('region', { name: 'Channels needing attention' })
    expect(within(region).getByText('RSS polling failure (3)')).toBeTruthy()
    expect(within(region).getByText('Feed gap could not be recovered: no safe anchor')).toBeTruthy()
    expect(within(region).getByText('Run a channel backfill to recover older uploads')).toBeTruthy()
    expect(within(region).getByText('POST /api/channels/UCabc/backfill')).toBeTruthy()
    expect(within(region).getByText('2026-10-01 00:00 UTC')).toBeTruthy()
  })

  it.each([
    ['suspected', 'Possible feed gap awaiting reconciliation'],
    ['listing_window_exhausted', 'Feed gap may continue beyond the reconciliation window'],
  ] as const)('renders %s copy', async (reason, copy) => {
    stubFetch(api({ channels: () => channels([{ channel_id: 'UCx', title: null, poll_failing_since: null, poll_failures: 0, last_poll_err: null, feed_gap_detected_at: '2026-10-01T00:00:00Z', feed_gap_reason: reason }]) }))
    await mount()
    expect(screen.getByText(copy)).toBeTruthy()
  })
})

describe('route and requests', () => {
  it('renders the Ops heading and requests dead jobs, limit 50, plus the queue depth', async () => {
    stubFetch(api())
    await mount()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Ops')
    expect(jobGets()).toHaveLength(1)
    expect(Object.fromEntries(lastJobGet().url.searchParams)).toEqual({ state: 'dead', limit: '50' })
    expect(healthGets()).toHaveLength(1)
    expect(calls.every((c) => c.method === 'GET')).toBe(true)
  })

  it('a deep link restores the filters and the cursor, and sends them as query params', async () => {
    stubFetch(api())
    await mount('/ops?state=pending&kind=transcribe&error_class=none&before=77')
    expect(select('State').value).toBe('pending')
    expect(select('Kind').value).toBe('transcribe')
    expect(select('Error class').value).toBe('none')
    expect(Object.fromEntries(lastJobGet().url.searchParams)).toEqual({
      state: 'pending',
      kind: 'transcribe',
      error_class: 'none',
      before_id: '77',
      limit: '50',
    })
  })

  it('omits unset parameters instead of sending them empty', async () => {
    stubFetch(api())
    await mount('/ops?kind=analyze')
    const q = lastJobGet().url.searchParams
    expect(q.has('error_class')).toBe(false)
    expect(q.has('before_id')).toBe(false)
    expect(q.get('kind')).toBe('analyze')
  })
})

describe('filters', () => {
  it('has labelled selects with the documented options', async () => {
    stubFetch(api())
    await mount()
    const opts = (s: HTMLSelectElement) => Array.from(s.options).map((o) => [o.value, o.textContent])
    expect(opts(select('State')).map((o) => o[0])).toEqual(['pending', 'running', 'done', 'dead'])
    expect(select('State').value).toBe('dead')
    expect(opts(select('Kind'))).toEqual([
      ['', 'Any'],
      ['ingest', 'ingest'],
      ['transcribe', 'transcribe'],
      ['analyze', 'analyze'],
    ])
    expect(opts(select('Error class')).map((o) => o[0])).toEqual([
      '', 'PERMANENT_SOURCE', 'TRANSIENT_NETWORK', 'RATE_LIMITED', 'TOOL_FAILURE',
      'LLM_INVALID_OUTPUT', 'LLM_UNAVAILABLE', 'RESOURCE', 'BUG', 'none',
    ])
    expect(opts(select('Error class'))[0]).toEqual(['', 'Any'])
    expect(opts(select('Error class')).at(-1)).toEqual(['none', 'No class'])
  })

  it('each change pushes a history entry, requests with the new filter and AND-combines', async () => {
    stubFetch(api())
    const { router } = await mount()
    fireEvent.change(select('Kind'), { target: { value: 'transcribe' } })
    await settle()
    expect(router.state.location.search).toBe('?kind=transcribe')
    expect(router.state.historyAction).toBe('PUSH')
    fireEvent.change(select('Error class'), { target: { value: 'BUG' } })
    await settle()
    fireEvent.change(select('State'), { target: { value: 'done' } })
    await settle()
    expect(router.state.location.search).toBe('?state=done&kind=transcribe&error_class=BUG')
    expect(Object.fromEntries(lastJobGet().url.searchParams)).toEqual({
      state: 'done', kind: 'transcribe', error_class: 'BUG', limit: '50',
    })
    fireEvent.change(select('Kind'), { target: { value: '' } })
    await settle()
    expect(lastJobGet().url.searchParams.has('kind')).toBe(false)
    fireEvent.change(select('State'), { target: { value: 'dead' } })
    await settle()
    expect(router.state.location.search).toBe('?error_class=BUG')
  })

  it.each([
    ['State', 'pending', { state: 'pending' }],
    ['Kind', 'analyze', { state: 'dead', kind: 'analyze' }],
    ['Error class', 'BUG', { state: 'dead', error_class: 'BUG' }],
    ['Error class', 'none', { state: 'dead', error_class: 'none' }],
  ])('changing only %s to %s requests a fresh list (every filter is in the key)', async (label, value, expected) => {
    stubFetch(api())
    await mount()
    fireEvent.change(select(label), { target: { value } })
    await settle()
    expect(jobGets()).toHaveLength(2)
    expect(Object.fromEntries(lastJobGet().url.searchParams)).toEqual({ ...expected, limit: '50' })
  })

  it('changing a filter resets to the first page', async () => {
    stubFetch(api())
    const { router } = await mount('/ops?before=500&kind=ingest')
    fireEvent.change(select('Kind'), { target: { value: 'analyze' } })
    await settle()
    expect(router.state.location.search).toBe('?kind=analyze')
    expect(lastJobGet().url.searchParams.has('before_id')).toBe(false)
  })

  it('Back restores the previous filter', async () => {
    stubFetch(api())
    const { router } = await mount()
    fireEvent.change(select('Kind'), { target: { value: 'analyze' } })
    await settle()
    await act(async () => {
      await router.navigate(-1)
    })
    await settle()
    expect(select('Kind').value).toBe('')
    expect(lastJobGet().url.searchParams.has('kind')).toBe(false)
  })
})

describe('queue depth', () => {
  const rowText = (name: string) =>
    within(within(depth()).getByRole('rowheader', { name })
      .closest('tr') as HTMLElement).getAllByRole('cell').map((c) => c.textContent)

  it('is a table with column and row headers, known order, and zeros for missing cells', async () => {
    stubFetch(api({ health: () => health({ transcribe: { running: 2, dead: 1 }, ingest: { pending: 5 } }) }))
    await mount()
    const t = within(depth()).getByRole('table')
    expect(within(t).getAllByRole('columnheader').map((c) => c.textContent)).toEqual([
      'Kind', 'pending', 'running', 'done', 'dead',
    ])
    expect(within(t).getAllByRole('rowheader').map((c) => c.textContent)).toEqual(['ingest', 'transcribe', 'analyze'])
    expect(rowText('ingest')).toEqual(['5', '0', '0', '0'])
    expect(rowText('transcribe')).toEqual(['0', '2', '0', '1'])
    expect(rowText('analyze')).toEqual(['0', '0', '0', '0'])
  })

  it('an empty queue object is a table of zeros', async () => {
    stubFetch(api())
    await mount()
    const cells = within(within(depth()).getByRole('table')).getAllByRole('cell')
    expect(cells).toHaveLength(12)
    expect(cells.every((c) => c.textContent === '0')).toBe(true)
  })

  it('shows an unknown kind and an unknown state after the known ones without crashing', async () => {
    stubFetch(api({ health: () => health({ notify: { pending: 4, snoozed: 9 }, ingest: { dead: 2 } }) }))
    await mount()
    const t = within(depth()).getByRole('table')
    expect(within(t).getAllByRole('columnheader').map((c) => c.textContent)).toEqual([
      'Kind', 'pending', 'running', 'done', 'dead', 'snoozed',
    ])
    expect(within(t).getAllByRole('rowheader').map((c) => c.textContent)).toEqual([
      'ingest', 'transcribe', 'analyze', 'notify',
    ])
    expect(rowText('notify')).toEqual(['4', '0', '0', '0', '9'])
    expect(rowText('ingest')).toEqual(['0', '0', '0', '2', '0'])
  })

  it.each([
    [{}, 'No dead jobs'],
    [{ ingest: { dead: 1 } }, '1 dead job'],
    [{ ingest: { dead: 1 }, analyze: { dead: 2 }, notify: { dead: 4 } }, '7 dead jobs'],
  ])('states the dead total in text for %j', async (queue, text) => {
    stubFetch(api({ health: () => health(queue) }))
    await mount()
    expect(within(depth()).getByText(text)).toBeTruthy()
  })

  it('renders hostile kind and state names as text', async () => {
    stubFetch(api({ health: () => health({ '<img src=x onerror=alert(1)>': { '<b>x</b>': 1 } }) }))
    await mount()
    expect(within(depth()).getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
    expect(document.querySelector('img, b')).toBeNull()
  })

  it('polls every 15 s while mounted and stops after unmount', async () => {
    stubFetch(api())
    const { view } = await mount()
    expect(healthGets()).toHaveLength(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(14_999)
    })
    expect(healthGets()).toHaveLength(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(healthGets()).toHaveLength(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(healthGets()).toHaveLength(3)
    view.unmount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000)
    })
    expect(healthGets()).toHaveLength(3)
  })

  it('Refresh refetches both the depth and the job list at once', async () => {
    stubFetch(api())
    await mount()
    await click(screen.getByRole('button', { name: 'Refresh' }))
    expect(healthGets()).toHaveLength(2)
    expect(jobGets()).toHaveLength(2)
  })
})

describe('job rows', () => {
  it('lists rows in API order, untouched, with every documented field', async () => {
    const items = [
      job(30, { video_title: 'Zebra', kind: 'transcribe', attempts: 5, error_class: 'RESOURCE' }),
      job(10, { video_title: 'Apple' }),
      job(20, { video_title: null, video_id: 'abcdefghijk', finished_at: null, state: 'pending', error_class: null }),
    ]
    stubFetch(api({ jobs: () => list(items) }))
    await mount()
    expect(bodyRows().map((r) => (r as HTMLTableRowElement).cells[0]?.textContent)).toEqual(['30', '10', '20'])
    const r30 = rowFor(30)
    expect(within(r30).getByRole('link', { name: 'Zebra' }).getAttribute('href')).toBe('/videos/v0000000030')
    expect(r30.textContent).toContain('transcribe')
    expect(r30.textContent).toContain('dead')
    expect(r30.textContent).toContain('5')
    expect(r30.textContent).toContain('RESOURCE')
    expect(r30.textContent).toContain('Disk full or out of memory. Fix that before retrying')
    // finished job: finished_at, converted to UTC (the test TZ is America/Los_Angeles)
    expect(within(r30).getByText('2026-10-02 06:30 UTC')).toBeTruthy()
    // unfinished job: created_at; title falls back to the video id
    const r20 = rowFor(20)
    expect(within(r20).getByRole('link', { name: 'abcdefghijk' }).getAttribute('href')).toBe('/videos/abcdefghijk')
    expect(within(r20).getByText('2026-09-30 08:05 UTC')).toBeTruthy()
    expect(within(r20).getByText('No class')).toBeTruthy()
    expect(within(r20).getByText('Often a lost worker (stale heartbeat)')).toBeTruthy()
  })

  it('has column headers and never shows the payload', async () => {
    stubFetch(api({ jobs: () => list([{ ...job(1), payload: { secret: 'p' } } as JobOut]) }))
    await mount()
    expect(within(jobsTable()).getAllByRole('columnheader').length).toBeGreaterThanOrEqual(8)
    expect(document.body.textContent).not.toContain('secret')
  })

  it('encodes a hostile video id in the link', async () => {
    stubFetch(api({ jobs: () => list([job(1, { video_id: '../x?y#z', video_title: null })]) }))
    await mount()
    expect(within(rowFor(1)).getByRole('link').getAttribute('href')).toBe('/videos/..%2Fx%3Fy%23z')
  })

  it('shows an unknown error class as-is with no hint, and a prototype-ish one too', async () => {
    stubFetch(api({ jobs: () => list([job(1, { error_class: 'WEIRD_NEW' }), job(2, { error_class: 'constructor' })]) }))
    await mount()
    expect(within(rowFor(1)).getByText('WEIRD_NEW')).toBeTruthy()
    expect(within(rowFor(2)).getByText('constructor')).toBeTruthy()
    expect(rowFor(2).textContent).not.toContain('function')
  })

  it('falls back to created_at when finished_at is null', async () => {
    stubFetch(api({ jobs: () => list([job(1, { finished_at: null })]) }))
    await mount()
    expect(within(rowFor(1)).getByText('2026-09-30 08:05 UTC')).toBeTruthy()
  })

  it('renders hostile title, last_error and error_class as text and creates no element', async () => {
    const evil = '<img src=x onerror=alert(1)>'
    stubFetch(
      api({ jobs: () => list([job(1, { video_title: evil, last_error: evil, error_class: evil, locked_by: evil })]) }),
    )
    await mount()
    const row = rowFor(1)
    expect(within(row).getAllByText(evil).length).toBeGreaterThanOrEqual(3)
    await click(within(row).getByRole('button', { name: 'Show full error' }))
    expect(document.querySelector('img')).toBeNull()
    expect(row.querySelectorAll('pre')).toHaveLength(1)
    expect(row.querySelector('pre')?.textContent).toBe(evil)
  })
})

describe('last_error', () => {
  const trace = ['Traceback (most recent call last):', '  File "a.py", line 1, in <module>', '    boom()', 'ValueError: nöpe ✓'].join('\n')

  it('collapses to the first line and expands into a <pre> with the full text, then hides', async () => {
    stubFetch(api({ jobs: () => list([job(1, { last_error: trace })]) }))
    await mount()
    const row = rowFor(1)
    expect(within(row).getByText('Traceback (most recent call last):')).toBeTruthy()
    expect(row.querySelector('pre')).toBeNull()
    const show = within(row).getByRole('button', { name: 'Show full error' })
    expect(show.getAttribute('aria-expanded')).toBe('false')
    await click(show)
    const pre = row.querySelector('pre') as HTMLPreElement
    expect(pre.textContent).toBe(trace)
    expect(within(row).getByRole('button', { name: 'Hide' }).getAttribute('aria-expanded')).toBe('true')
    await click(within(row).getByRole('button', { name: 'Hide' }))
    expect(row.querySelector('pre')).toBeNull()
  })

  it('expanding one row leaves the others collapsed', async () => {
    stubFetch(api({ jobs: () => list([job(1, { last_error: trace }), job(2, { last_error: trace })]) }))
    await mount()
    await click(within(rowFor(1)).getByRole('button', { name: 'Show full error' }))
    expect(rowFor(1).querySelector('pre')).not.toBeNull()
    expect(rowFor(2).querySelector('pre')).toBeNull()
  })

  it('keeps an 8 KiB single line out of the collapsed view and whole in the <pre>', async () => {
    const big = 'x'.repeat(8192)
    stubFetch(api({ jobs: () => list([job(1, { last_error: big })]) }))
    await mount()
    const row = rowFor(1)
    expect(row.textContent).not.toContain('x'.repeat(201))
    expect(row.textContent).toContain('x'.repeat(199) + '…')
    await click(within(row).getByRole('button', { name: 'Show full error' }))
    expect(row.querySelector('pre')?.textContent).toBe(big)
  })

  it('handles a 100-line traceback', async () => {
    const tb = Array.from({ length: 100 }, (_, i) => `  line ${i}`).join('\n')
    stubFetch(api({ jobs: () => list([job(1, { last_error: `Traceback:\n${tb}` })]) }))
    await mount()
    const row = rowFor(1)
    expect(row.textContent).not.toContain('line 5')
    await click(within(row).getByRole('button', { name: 'Show full error' }))
    expect(row.querySelector('pre')?.textContent).toBe(`Traceback:\n${tb}`)
  })

  it.each([null, '', '   '])('shows a dash and no control for %j', async (v) => {
    stubFetch(api({ jobs: () => list([job(1, { last_error: v })]) }))
    await mount()
    const row = rowFor(1)
    expect(within(row).queryByRole('button', { name: /full error|Hide/ })).toBeNull()
    expect(within(row).getAllByText('—').length).toBeGreaterThanOrEqual(1)
  })
})

describe('paging', () => {
  const rows = (from: number, n: number) => Array.from({ length: n }, (_, i) => job(from - i))
  const older = () => screen.getByRole('button', { name: 'Older' }) as HTMLButtonElement
  const newest = () => screen.getByRole('button', { name: 'Newest' }) as HTMLButtonElement

  it('an empty list: both controls disabled', async () => {
    stubFetch(api())
    await mount()
    expect(older().disabled).toBe(true)
    expect(newest().disabled).toBe(true)
  })

  it('exactly 50 rows with next_before_id=null: Older is disabled', async () => {
    stubFetch(api({ jobs: () => list(rows(100, 50), null) }))
    await mount()
    expect(bodyRows()).toHaveLength(50)
    expect(older().disabled).toBe(true)
  })

  it('51 matching rows: page 1 full with a cursor, page 2 with one row, and back to newest', async () => {
    stubFetch(
      api({
        jobs: (c) => (c.url.searchParams.get('before_id') === '51' ? list([job(1)], null) : list(rows(101, 50), 51)),
      }),
    )
    const { router } = await mount()
    expect(bodyRows()).toHaveLength(50)
    expect(older().disabled).toBe(false)
    expect(newest().disabled).toBe(true)
    await click(older())
    expect(router.state.location.search).toBe('?before=51')
    expect(router.state.historyAction).toBe('PUSH')
    expect(lastJobGet().url.searchParams.get('before_id')).toBe('51')
    expect(bodyRows()).toHaveLength(1)
    expect(older().disabled).toBe(true)
    expect(newest().disabled).toBe(false)
    await click(newest())
    expect(router.state.location.search).toBe('')
    expect(lastJobGet().url.searchParams.has('before_id')).toBe(false)
    expect(bodyRows()).toHaveLength(50)
  })

  it('keeps the current rows with a loading indicator and disabled controls while the next page loads', async () => {
    const d = deferred()
    stubFetch(api({ jobs: (c) => (c.url.searchParams.has('before_id') ? d.promise : list(rows(100, 3), 98)) }))
    await mount()
    await click(older())
    expect(bodyRows()).toHaveLength(3)
    expect(within(jobsRegion()).getByText('Loading jobs…')).toBeTruthy()
    expect(older().disabled).toBe(true)
    expect(newest().disabled).toBe(true)
    await act(async () => {
      d.resolve(json(list([job(97)], null)))
    })
    await settle()
    expect(bodyRows()).toHaveLength(1)
    expect(within(jobsRegion()).queryByText('Loading jobs…')).toBeNull()
  })
})

describe('empty states', () => {
  it('default filter: "No dead jobs" and no reset control', async () => {
    stubFetch(api())
    await mount()
    expect(within(jobsRegion()).getByText('No dead jobs')).toBeTruthy()
    expect(within(jobsRegion()).queryByText('No jobs match these filters')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Reset filters' })).toBeNull()
  })

  it('other filters: "No jobs match these filters" with Reset filters, which clears the URL', async () => {
    stubFetch(api())
    const { router } = await mount('/ops?state=pending&kind=ingest')
    expect(within(jobsRegion()).getByText('No jobs match these filters')).toBeTruthy()
    expect(within(jobsRegion()).queryByText('No dead jobs')).toBeNull()
    await click(screen.getByRole('button', { name: 'Reset filters' }))
    expect(router.state.location.search).toBe('')
    expect(select('State').value).toBe('dead')
    expect(select('Kind').value).toBe('')
    expect(lastJobGet().url.searchParams.get('state')).toBe('dead')
  })

  it.each(['/ops?error_class=none', '/ops?state=pending', '/ops?kind=ingest', '/ops?state=dead&kind=analyze'])(
    'a single non-default filter (%s) triggers the second message',
    async (entry) => {
      stubFetch(api())
      await mount(entry)
      expect(within(jobsRegion()).getByText('No jobs match these filters')).toBeTruthy()
      expect(within(jobsRegion()).queryByText('No dead jobs')).toBeNull()
    },
  )

  it('?state=dead alone is still the default filter', async () => {
    stubFetch(api())
    await mount('/ops?state=dead')
    expect(within(jobsRegion()).getByText('No dead jobs')).toBeTruthy()
  })

  it('shows neither empty message from stale empty data while another filter loads', async () => {
    const d = deferred()
    stubFetch(api({ jobs: (c) => (c.url.searchParams.has('kind') ? d.promise : list([])) }))
    await mount()
    expect(within(jobsRegion()).getByText('No dead jobs')).toBeTruthy()
    fireEvent.change(select('Kind'), { target: { value: 'ingest' } })
    await settle()
    expect(within(jobsRegion()).getByText('Loading jobs…')).toBeTruthy()
    expect(within(jobsRegion()).queryByText('No dead jobs')).toBeNull()
    expect(within(jobsRegion()).queryByText('No jobs match these filters')).toBeNull()
    await act(async () => {
      d.resolve(json(list([])))
    })
    await settle()
    expect(within(jobsRegion()).getByText('No jobs match these filters')).toBeTruthy()
  })
})

describe('untrusted URL parameters', () => {
  it('never sends or shows a marker in any invalid parameter, and drops them with replace', async () => {
    stubFetch(api())
    const { router } = await mount('/ops?state=MARKER1&kind=MARKER2&error_class=MARKER3&before=MARKER4')
    expect(calls.length).toBeGreaterThan(0)
    for (const c of calls) expect(c.url.toString()).not.toContain('MARKER')
    expect(document.body.innerHTML).not.toContain('MARKER')
    expect(Object.fromEntries(lastJobGet().url.searchParams)).toEqual({ state: 'dead', limit: '50' })
    expect(router.state.location.search).toBe('')
    expect(router.state.historyAction).toBe('REPLACE')
    expect(select('State').value).toBe('dead')
    expect(within(jobsRegion()).getByText('No dead jobs')).toBeTruthy()
  })

  it.each([
    ['state=DEAD', 'state'],
    ['kind=Ingest', 'kind'],
    ['state=', 'state'],
    ['kind=', 'kind'],
    ['error_class=', 'error_class'],
    ['error_class=bug', 'error_class'],
    ['before=0', 'before_id'],
    ['before=-1', 'before_id'],
    ['before=1.5', 'before_id'],
    ['before=abc', 'before_id'],
    ['before=1e3', 'before_id'],
    ['before=9007199254740992', 'before_id'],
    ['before=007', 'before_id'],
  ])('%s is rejected and not sent', async (qs, param) => {
    stubFetch(api())
    await mount(`/ops?${qs}`)
    const sent = lastJobGet().url.searchParams
    expect(sent.has(param)).toBe(param === 'state' ? true : false)
    if (param === 'state') expect(sent.get('state')).toBe('dead')
  })

  it('keeps the largest safe cursor', async () => {
    stubFetch(api())
    await mount('/ops?before=9007199254740991')
    expect(lastJobGet().url.searchParams.get('before_id')).toBe('9007199254740991')
  })
})

describe('load errors', () => {
  it('shows a loading indicator per part first, not an empty table or the empty message', async () => {
    const h = deferred()
    const j = deferred()
    stubFetch(api({ health: () => h.promise, jobs: () => j.promise }))
    await mount()
    expect(within(depth()).getByText('Loading queue depth…')).toBeTruthy()
    expect(within(jobsRegion()).getByText('Loading jobs…')).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.queryByText('No dead jobs')).toBeNull()
    await act(async () => {
      h.resolve(json(health()))
      j.resolve(json(list([])))
    })
    await settle()
    expect(screen.queryByText('Loading jobs…')).toBeNull()
  })

  it('the depth failing does not stop the job list, and vice versa', async () => {
    stubFetch(api({ health: () => json({}, 500) , jobs: () => list([job(1)]) }))
    await mount()
    expect(within(depth()).getByText('Could not load queue depth')).toBeTruthy()
    expect(rowFor(1)).toBeTruthy()
  })

  it('the job list failing does not stop the depth', async () => {
    stubFetch(api({ health: () => health({ ingest: { dead: 3 } }), jobs: () => json({}, 500) }))
    await mount()
    expect(within(jobsRegion()).getByText('Could not load jobs')).toBeTruthy()
    expect(within(depth()).getByText('3 dead jobs')).toBeTruthy()
  })

  it('healthz 503 says the database is down, with Retry that refetches', async () => {
    let n = 0
    stubFetch(api({ health: () => (n++ === 0 ? json({ detail: 'SECRET' }, 503) : health({ ingest: { dead: 2 } })) }))
    await mount()
    expect(within(depth()).getByText('Queue depth unavailable: the database is down')).toBeTruthy()
    await click(within(depth()).getByRole('button', { name: 'Retry' }))
    expect(within(depth()).getByText('2 dead jobs')).toBeTruthy()
    expect(within(depth()).queryByText(/unavailable/)).toBeNull()
  })

  it('jobs 503 says the database is unavailable, with Retry that refetches', async () => {
    let n = 0
    stubFetch(api({ jobs: () => (n++ === 0 ? json({ detail: 'SECRET' }, 503) : list([job(4)])) }))
    await mount()
    expect(within(jobsRegion()).getByText('The database is unavailable. Try again shortly.')).toBeTruthy()
    await click(within(jobsRegion()).getByRole('button', { name: 'Retry' }))
    expect(rowFor(4)).toBeTruthy()
  })

  it.each([
    ['500', () => json({ detail: 'SECRET /api/healthz Traceback' }, 500)],
    ['422', () => json({ detail: [{ msg: 'SECRET', input: 'SECRET' }] }, 422)],
    ['404', () => json({ detail: 'SECRET' }, 404)],
    ['a network failure', () => Promise.reject(new TypeError('SECRET fetch failed /api/healthz'))],
    ['a malformed body', () => json({ unexpected: 'SECRET' })],
    ['a non-JSON body', () => new Response('SECRET <html>', { status: 200 })],
  ])('%s shows the generic message, a Retry button, and no body, URL or exception text', async (_n, reply) => {
    stubFetch(api({ health: reply as Handler, jobs: reply as Handler }))
    await mount()
    expect(within(depth()).getByText('Could not load queue depth')).toBeTruthy()
    expect(within(jobsRegion()).getByText('Could not load jobs')).toBeTruthy()
    expect(within(depth()).getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(within(jobsRegion()).getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(document.body.textContent).not.toMatch(/SECRET|\/api\/|TypeError|fetch failed/)
    expect(screen.queryByText('No dead jobs')).toBeNull()
  })
})

describe('retry', () => {
  const retryBtn = (id: number) => within(rowFor(id)).queryByRole('button', { name: 'Retry' })

  it('offers Retry only on dead rows', async () => {
    stubFetch(
      api({
        jobs: () =>
          list([job(4), job(3, { state: 'pending' }), job(2, { state: 'running' }), job(1, { state: 'done' })]),
      }),
    )
    await mount('/ops?state=done')
    expect(retryBtn(4)).not.toBeNull()
    for (const id of [1, 2, 3]) expect(retryBtn(id)).toBeNull()
    for (const id of [1, 2, 3]) expect(within(rowFor(id)).queryByRole('button', { name: /confirm|cancel/i })).toBeNull()
  })

  it('Retry asks first, names job, kind and video, sends nothing, and moves focus to Confirm', async () => {
    stubFetch(api({ jobs: () => list([job(12, { kind: 'analyze', video_title: 'My video' })]) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    const row = rowFor(12)
    expect(posts()).toHaveLength(0)
    expect(row.textContent).toContain('job #12')
    expect(row.textContent).toContain('analyze')
    expect(row.textContent).toContain('My video')
    expect(row.textContent).not.toMatch(/hours of CPU/)
    expect(document.activeElement).toBe(within(row).getByRole('button', { name: 'Confirm retry' }))
  })

  it('a transcribe job warns that transcription can take hours of CPU', async () => {
    stubFetch(api({ jobs: () => list([job(12, { kind: 'transcribe' })]) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    expect(rowFor(12).textContent).toMatch(/hours of CPU/)
  })

  it('Cancel sends nothing and returns focus to Retry', async () => {
    stubFetch(api({ jobs: () => list([job(12)]) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Cancel' }))
    expect(posts()).toHaveLength(0)
    expect(within(rowFor(12)).queryByRole('button', { name: 'Confirm retry' })).toBeNull()
    expect(document.activeElement).toBe(retryBtn(12))
  })

  it('does not use window.confirm', async () => {
    const confirm = vi.fn(() => true)
    vi.stubGlobal('confirm', confirm)
    stubFetch(api({ jobs: () => list([job(12)]), retry: () => job(12, { state: 'pending' }) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(confirm).not.toHaveBeenCalled()
  })

  it('Confirm sends exactly one POST with a JSON content type and body {}', async () => {
    stubFetch(api({ jobs: () => list([job(12)]), retry: () => job(12, { state: 'pending' }) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(posts()).toHaveLength(1)
    const p = posts()[0] as Call
    expect(p.url.pathname).toBe('/api/ops/jobs/12/retry')
    expect(p.url.search).toBe('')
    expect(p.headers.get('content-type')).toBe('application/json')
    expect(p.body).toBe('{}')
  })

  it('a double click sends one request, disables that row, and leaves other rows usable', async () => {
    const d = deferred()
    stubFetch(api({ jobs: () => list([job(12), job(11)]), retry: () => d.promise }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    const confirm = within(rowFor(12)).getByRole('button', { name: 'Confirm retry' })
    fireEvent.click(confirm)
    fireEvent.click(confirm)
    await settle()
    expect(posts()).toHaveLength(1)
    expect((within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }) as HTMLButtonElement).disabled).toBe(true)
    expect((within(rowFor(12)).getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)
    expect((retryBtn(11) as HTMLButtonElement).disabled).toBe(false)
    await act(async () => {
      d.resolve(json(job(12, { state: 'pending' })))
    })
    await settle()
  })

  it('does not retry the request automatically on a server error', async () => {
    stubFetch(api({ jobs: () => list([job(12)]), retry: () => json({}, 500) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(posts()).toHaveLength(1)
  })

  it.each<[string, () => Reply, string]>([
    ['200', () => job(12, { state: 'pending' }), 'Queued again: job #12 is pending'],
    ['404', () => json({ detail: 'job not found' }, 404), 'This job no longer exists'],
    ['409 not dead (running)', () => json({ detail: 'job is not dead', state: 'running' }, 409), 'Job is already running'],
    ['409 not dead (unknown state)', () => json({ detail: 'job is not dead', state: '<b>x</b>' }, 409), 'Job is no longer dead'],
    ['409 superseded', () => json({ detail: 'superseded by a newer job', job_id: 99 }, 409), 'A newer job #99 exists for this video'],
    ['409 superseded (bad id)', () => json({ detail: 'superseded by a newer job', job_id: '99' }, 409), 'A newer job exists for this video'],
    ['409 other', () => json({ detail: 'SECRET' }, 409), 'The job could not be retried'],
    ['429', () => json({ detail: 'SECRET' }, 429), 'Too many requests. Try again shortly.'],
    ['503', () => json({ detail: 'SECRET' }, 503), 'The database is unavailable. Try again shortly.'],
    ['500', () => json({ detail: 'SECRET' }, 500), 'Retry failed'],
    ['415', () => json({ detail: 'SECRET' }, 415), 'Retry failed'],
    ['a network failure', () => Promise.reject(new TypeError('SECRET boom')), 'Retry failed'],
  ])('shows the fixed message for %s and never the body', async (_n, reply, message) => {
    // The list keeps returning the job, so the row stays and shows the message.
    stubFetch(api({ jobs: () => list([job(12)]), retry: reply }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(within(rowFor(12)).getByText(message)).toBeTruthy()
    expect(liveText()).toContain(message)
    expect(document.body.textContent).not.toContain('SECRET')
    // the row is back to idle: Retry is offered again, no confirmation left open
    expect(retryBtn(12)).not.toBeNull()
    expect(within(rowFor(12)).queryByRole('button', { name: 'Confirm retry' })).toBeNull()
  })

  it('refetches the list and the depth after any response, so a retried job leaves the dead list', async () => {
    let retried = false
    stubFetch(
      api({
        jobs: () => list(retried ? [job(11)] : [job(12), job(11)]),
        retry: () => {
          retried = true
          return job(12, { state: 'pending' })
        },
        health: () => health({ ingest: { dead: retried ? 1 : 2, pending: retried ? 1 : 0 } }),
      }),
    )
    await mount()
    expect(within(depth()).getByText('2 dead jobs')).toBeTruthy()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(jobGets()).toHaveLength(2)
    expect(healthGets()).toHaveLength(2)
    expect(() => rowFor(12)).toThrow()
    expect(within(depth()).getByText('1 dead job')).toBeTruthy()
    // the message outlives the vanished row, in a polite live region
    expect(liveText()).toContain('Queued again: job #12 is pending')
    expect(screen.getByText('Queued again: job #12 is pending')).toBeTruthy()
  })

  it.each([404, 409, 429, 500, 503])('refetches after a %i too', async (status) => {
    stubFetch(api({ jobs: () => list([job(12)]), retry: () => json({ detail: 'x' }, status) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(jobGets()).toHaveLength(2)
    expect(healthGets()).toHaveLength(2)
  })

  it('does not refetch after a network failure', async () => {
    stubFetch(api({ jobs: () => list([job(12)]), retry: () => Promise.reject(new TypeError('down')) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(jobGets()).toHaveLength(1)
    expect(healthGets()).toHaveLength(1)
  })

  it('does not retry a network failure automatically, even when the client would', async () => {
    stubFetch(api({ jobs: () => list([job(12)]), retry: () => Promise.reject(new TypeError('down')) }))
    await mount('/ops', { mutations: { retry: 3 } })
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(posts()).toHaveLength(1)
    expect(screen.getByText('Retry failed')).toBeTruthy()
  })

  it('Refresh clears the result message', async () => {
    stubFetch(api({ jobs: () => list([job(12)]), retry: () => json({}, 429) }))
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(screen.getByText('Too many requests. Try again shortly.')).toBeTruthy()
    await click(screen.getByRole('button', { name: 'Refresh' }))
    expect(screen.queryByText('Too many requests. Try again shortly.')).toBeNull()
  })

  it('confirming another job clears an earlier result that arrived meanwhile', async () => {
    const a = deferred()
    const b = deferred()
    stubFetch(
      api({
        jobs: () => list([job(12), job(11)]),
        retry: (c) => (c.url.pathname.includes('/12/') ? a.promise : b.promise),
      }),
    )
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    await click(retryBtn(11) as HTMLElement)
    await act(async () => {
      a.resolve(json({}, 429))
    })
    await settle()
    expect(screen.getByText('Too many requests. Try again shortly.')).toBeTruthy()
    await click(within(rowFor(11)).getByRole('button', { name: 'Confirm retry' }))
    expect(screen.queryByText('Too many requests. Try again shortly.')).toBeNull()
    await act(async () => {
      b.resolve(json({}, 404))
    })
    await settle()
  })

  it('keeps the result message until the next action, then replaces it', async () => {
    let status = 429
    stubFetch(
      api({
        jobs: () => list([job(12)]),
        retry: () => (status === 429 ? json({}, 429) : json({ detail: 'job not found' }, 404)),
      }),
    )
    await mount()
    await click(retryBtn(12) as HTMLElement)
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(screen.getByText('Too many requests. Try again shortly.')).toBeTruthy()
    status = 404
    await click(retryBtn(12) as HTMLElement)
    expect(screen.queryByText('Too many requests. Try again shortly.')).toBeNull()
    await click(within(rowFor(12)).getByRole('button', { name: 'Confirm retry' }))
    expect(screen.getByText('This job no longer exists')).toBeTruthy()
    expect(screen.queryByText('Too many requests. Try again shortly.')).toBeNull()
  })
})

describe('accessibility and source', () => {
  it('every interactive control is a button, link or select with an accessible name', async () => {
    stubFetch(api({ jobs: () => list([job(1)]) }))
    await mount()
    for (const el of screen.getAllByRole('button')) expect(el.tagName).toBe('BUTTON')
    for (const s of screen.getAllByRole('combobox')) expect(s.tagName).toBe('SELECT')
    expect(select('State').labels?.length).toBe(1)
    expect(select('Kind').labels?.length).toBe(1)
    expect(select('Error class').labels?.length).toBe(1)
    expect(document.querySelector('[onclick], [tabindex]')).toBeNull()
  })
})
