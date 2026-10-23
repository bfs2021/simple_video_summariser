import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { components } from '../api'
import { FakePlayerProvider } from '../components/FakePlayerProvider'
import { Compare } from './Compare'
import { routes } from './index'

type AnalysisOut = components['schemas']['AnalysisOut']
type VideoDetailOut = components['schemas']['VideoDetail']

const ID = 'dQw4w9WgXcQ'
const DASH_ID = '-wNyEUrxzFU'
const OTHER_ID = 'abcdefghijk'

let nextId: number
function run(over: Partial<AnalysisOut> = {}): AnalysisOut {
  const id = over.id ?? nextId++
  return {
    id,
    transcript_id: 1,
    transcript_source: 'captions',
    chunk_strategy: 'fixed',
    model: 'gpt',
    prompt_version: 'v1',
    created_at: '2026-10-01T08:09:10Z',
    input_tokens: 1000,
    output_tokens: 200,
    cost_usd: null,
    duration_ms: null,
    tldr: `TLDR of ${id}`,
    speaker_roster: null,
    topics: [],
    claims: [],
    quotes: [],
    ...over,
  }
}

function claim(over: Partial<AnalysisOut['claims'][number]> = {}): AnalysisOut['claims'][number] {
  return { text: 'A claim', speaker: 'Ada', confidence: 'high', start_sec: 10, source_chunk_seq: 0, ...over }
}

function quote(over: Partial<AnalysisOut['quotes'][number]> = {}): AnalysisOut['quotes'][number] {
  return { text: 'A quote', speaker: 'Ada', start_sec: 10, source_chunk_seq: 0, ...over }
}

function video(over: Partial<VideoDetailOut> = {}): VideoDetailOut {
  return {
    video_id: ID,
    title: 'My Video',
    channel_id: 'UC' + 'a'.repeat(22),
    channel_title: 'My Channel',
    published_at: '2026-09-30T23:30:00-07:00',
    duration_sec: 3723,
    latest_analysis_at: null,
    has_transcript: false,
    origin: 'manual',
    unavailable: null,
    status: 'done',
    active_job: null,
    last_failure: null,
    transcript: null,
    analysis: null,
    ...over,
  }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

let requests: URL[]
let seeks: number[]

/** Fake API: serves `runs` (newest first) page by page, like #42, limit capped at 50. */
interface Api {
  runs: AnalysisOut[]
  total?: number
  video?: () => Response | Promise<Response>
  analyses?: (url: URL, n: number) => Response | Promise<Response> | undefined
}

function stubApi(api: Api) {
  requests = []
  const fn = vi.fn((input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    requests.push(url)
    if (url.pathname.endsWith('/analyses')) {
      const custom = api.analyses?.(url, requests.filter((u) => u.pathname.endsWith('/analyses')).length)
      if (custom !== undefined) return Promise.resolve(custom)
      const limit = Math.min(Number(url.searchParams.get('limit') ?? 50), 50)
      const offset = Number(url.searchParams.get('offset') ?? 0)
      return Promise.resolve(
        json({ items: api.runs.slice(offset, offset + limit), limit, offset, total: api.total ?? api.runs.length }),
      )
    }
    return Promise.resolve(api.video ? api.video() : json(video()))
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

const analysesRequests = () => requests.filter((u) => u.pathname.endsWith('/analyses'))

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

async function mount(entry = `/videos/${ID}/compare`) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [
      { path: '/', element: <p>Library page</p> },
      {
        path: '/videos/:videoId/compare',
        element: (
          <FakePlayerProvider seeks={seeks}>
            <Compare />
          </FakePlayerProvider>
        ),
      },
    ],
    { initialEntries: [entry] },
  )
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  await settle()
  return router
}

beforeEach(() => {
  vi.useFakeTimers()
  seeks = []
  nextId = 1
})

const body = () => document.body.textContent ?? ''
const region = (name: string) => screen.getByRole('region', { name })
const selA = () => screen.getByLabelText('Analysis A') as HTMLSelectElement
const selB = () => screen.getByLabelText('Analysis B') as HTMLSelectElement
const selected = (el: HTMLSelectElement) => Number(el.value)
const cells = (row: HTMLElement) => within(row).getAllByRole('cell').map((c) => c.textContent)

/** Newest first, ids descending, like #42. */
function history(n: number, over: (i: number) => Partial<AnalysisOut> = () => ({})): AnalysisOut[] {
  return Array.from({ length: n }, (_, i) => run({ id: n - i, ...over(i) }))
}

/** Four runs: 4,3 are gpt/v2; 2 is local/v2; 1 is gpt/v1. */
const MIXED = [
  run({ id: 4, model: 'gpt', prompt_version: 'v2', created_at: '2026-10-04T00:00:00Z' }),
  run({ id: 3, model: 'gpt', prompt_version: 'v2', created_at: '2026-10-03T00:00:00Z' }),
  run({ id: 2, model: 'local', prompt_version: 'v2', created_at: '2026-10-02T00:00:00Z' }),
  run({ id: 1, model: 'gpt', prompt_version: 'v1', created_at: '2026-10-01T00:00:00Z' }),
]

describe('route and input validation', () => {
  it.each([
    ['10 characters', 'abcdefghij'],
    ['12 characters', 'abcdefghijkl'],
    ['an encoded slash (10 decoded)', 'abcde%2Ffghij'],
    ['an encoded slash (11 decoded)', 'abcde%2Ffghijk'],
    ['dot dot', '..'],
    ['a dot', 'abcdefghi.j'],
    ['an encoded space', 'abcdefghij%20'],
    ['a non-ASCII letter', 'abcdefghijé'],
    ['an encoded newline', 'abcdefghij%0A'],
  ])('shows Video not found and requests nothing for %s', async (_n, raw) => {
    const fetchFn = stubApi({ runs: MIXED })
    await mount(`/videos/${raw}/compare`)
    expect(screen.getByText('Video not found')).toBeTruthy()
    expect(fetchFn).not.toHaveBeenCalled()
    expect(document.querySelector('iframe')).toBeNull()
    expect(screen.getByRole('link', { name: /library/i }).getAttribute('href')).toBe('/')
  })

  it('accepts a leading dash and requests exactly that id', async () => {
    stubApi({ runs: MIXED })
    await mount(`/videos/${DASH_ID}/compare`)
    expect(requests.every((u) => u.pathname.startsWith(`/api/videos/${DASH_ID}`))).toBe(true)
    expect(analysesRequests()).toHaveLength(1)
  })

  it('is case-sensitive: the id is sent as written', async () => {
    stubApi({ runs: MIXED })
    await mount('/videos/dqw4w9wgxcq/compare')
    expect(analysesRequests()[0]?.pathname).toBe('/api/videos/dqw4w9wgxcq/analyses')
  })

  it.each([
    ['zero', '0'],
    ['negative', '-1'],
    ['fraction', '1.5'],
    ['exponent', '1e3'],
    ['letters', 'abc'],
    ['leading zeros', '007'],
    ['empty', ''],
    ['above 2^53 - 1', '9007199254740992'],
    ['plus sign', '%2B1'],
    ['space', '%201'],
  ])('drops a %s param: default selection, no notice, rewritten URL', async (_n, raw) => {
    stubApi({ runs: MIXED })
    const router = await mount(`/videos/${ID}/compare?a=${raw}&b=${raw}`)
    expect([selected(selA()), selected(selB())]).toEqual([4, 2])
    expect(body()).not.toContain('was not found')
    expect(router.state.location.search).toBe('?a=4&b=2')
  })

  it('drops a repeated param and rewrites the URL to hold the actual selection', async () => {
    stubApi({ runs: MIXED })
    const router = await mount(`/videos/${ID}/compare?a=1&a=2&b=3&b=3`)
    expect([selected(selA()), selected(selB())]).toEqual([4, 2])
    expect(router.state.location.search).toBe('?a=4&b=2')
  })

  it('never requests or shows a param value that carries a marker', async () => {
    stubApi({ runs: MIXED })
    const marker = 'MARKER-<img src=x onerror=alert(1)>'
    const router = await mount(`/videos/${ID}/compare?a=${encodeURIComponent(marker)}&b=${encodeURIComponent(marker)}&x=1`)
    expect(requests.map((u) => u.href).join('\n')).not.toContain('MARKER')
    expect(body()).not.toContain('MARKER')
    expect(document.body.innerHTML).not.toContain('MARKER')
    expect(document.querySelector('img')).toBeNull()
    expect(router.state.location.search).not.toContain('MARKER')
    expect(router.state.location.search).toContain('x=1')
  })

  it('a well-formed id that is not among the runs shows one fixed notice and no echo', async () => {
    stubApi({ runs: MIXED })
    const router = await mount(`/videos/${ID}/compare?a=987654&b=3`)
    expect(screen.getByText('A requested analysis was not found for this video. Showing the default comparison.')).toBeTruthy()
    expect(body()).not.toContain('987654')
    // A is dropped (default: newest), b = 3 is kept.
    expect([selected(selA()), selected(selB())]).toEqual([4, 3])
    expect(router.state.location.search).toBe('?a=4&b=3')
  })

  it('shows the notice once even when both ids are unknown', async () => {
    stubApi({ runs: MIXED })
    await mount(`/videos/${ID}/compare?a=987654&b=987655`)
    expect(screen.getAllByText(/was not found for this video/)).toHaveLength(1)
    expect([selected(selA()), selected(selB())]).toEqual([4, 2])
  })

  it('shows no unknown-id notice for a valid selection', async () => {
    stubApi({ runs: MIXED })
    await mount(`/videos/${ID}/compare?a=1&b=2`)
    expect(body()).not.toContain('was not found')
  })

  it('a equal to b: b is dropped and defaulted, without the notice', async () => {
    stubApi({ runs: MIXED })
    const router = await mount(`/videos/${ID}/compare?a=2&b=2`)
    expect([selected(selA()), selected(selB())]).toEqual([2, 4])
    expect(body()).not.toContain('was not found')
    expect(router.state.location.search).toBe('?a=2&b=4')
  })

  it('rewrites a bare URL with replace, so Back does not return to it, and keeps a good URL untouched', async () => {
    stubApi({ runs: MIXED })
    const router = await mount()
    expect(router.state.location.search).toBe('?a=4&b=2')
    expect(router.state.historyAction).toBe('REPLACE')
    expect(router.state.historyAction).not.toBe('PUSH')
  })

  it('does not rewrite a URL that already holds the resolved selection', async () => {
    stubApi({ runs: MIXED })
    const router = await mount(`/videos/${ID}/compare?b=2&a=4`)
    expect(router.state.historyAction).toBe('POP')
    expect(router.state.location.search).toBe('?b=2&a=4')
  })

  it('does not rewrite the URL before the runs have loaded', async () => {
    stubApi({ runs: MIXED, analyses: () => new Promise<Response>(() => {}) })
    const router = await mount(`/videos/${ID}/compare?a=abc`)
    expect(router.state.location.search).toBe('?a=abc')
  })
})

describe('data', () => {
  it.each([
    [0, 1],
    [1, 1],
    [50, 1],
    [51, 2],
    [100, 2],
    [101, 3],
    [200, 4],
    [250, 4],
  ])('total=%i takes %i request(s) with limit=50 and offsets 0, 50, 100, 150', async (total, expected) => {
    stubApi({ runs: history(Math.min(total, 200)), total })
    await mount()
    expect(analysesRequests()).toHaveLength(expected)
    expect(analysesRequests().map((u) => [u.searchParams.get('limit'), u.searchParams.get('offset')])).toEqual(
      [0, 50, 100, 150].slice(0, expected).map((o) => ['50', String(o)]),
    )
  })

  it('walks the pages one after another, never in parallel', async () => {
    let inFlight = 0
    let maxInFlight = 0
    stubApi({
      runs: history(120),
      analyses: (url) => {
        inFlight += 1
        maxInFlight = Math.max(maxInFlight, inFlight)
        const offset = Number(url.searchParams.get('offset'))
        const all = history(120)
        return new Promise<Response>((resolve) => {
          queueMicrotask(() => {
            inFlight -= 1
            resolve(json({ items: all.slice(offset, offset + 50), limit: 50, offset, total: 120 }))
          })
        })
      },
    })
    await mount()
    expect(maxInFlight).toBe(1)
    expect(analysesRequests()).toHaveLength(3)
  })

  it('shows all 200 runs and no cap notice at total=200', async () => {
    stubApi({ runs: history(200), total: 200 })
    await mount()
    expect(selA().options).toHaveLength(200)
    expect(body()).not.toContain('Showing the newest')
  })

  it('shows 200 runs and the cap notice at total=250', async () => {
    stubApi({ runs: history(200), total: 250 })
    await mount()
    expect(selA().options).toHaveLength(200)
    expect(screen.getByText('Showing the newest 200 of 250 analyses')).toBeTruthy()
  })

  it('stops at a page with no items before total is reached', async () => {
    stubApi({
      runs: history(60),
      total: 500,
      analyses: (url) =>
        Number(url.searchParams.get('offset')) >= 50
          ? json({ items: [], limit: 50, offset: 50, total: 500 })
          : undefined,
    })
    await mount()
    expect(analysesRequests()).toHaveLength(2)
    expect(selA().options).toHaveLength(50)
  })

  it('an empty first page before total shows the zero-run state after one request', async () => {
    stubApi({ runs: [], total: 10 })
    await mount()
    expect(analysesRequests()).toHaveLength(1)
    expect(screen.getByText('This video has no analyses yet.')).toBeTruthy()
  })

  it('keeps a run id that appears on two pages once, at its first position', async () => {
    const all = history(60)
    stubApi({
      runs: all,
      analyses: (url) => {
        const offset = Number(url.searchParams.get('offset'))
        // A new analysis shifted the offsets: id 11 (last of page 1) repeats on page 2.
        const items = offset === 0 ? all.slice(0, 50) : [all[49] as AnalysisOut, ...all.slice(50)]
        return json({ items, limit: 50, offset, total: 60 })
      },
    })
    await mount()
    const ids = Array.from(selA().options).map((o) => Number(o.value))
    expect(ids).toHaveLength(60)
    expect(new Set(ids).size).toBe(60)
    expect(ids).toEqual(all.map((r) => r.id))
  })

  it('keeps the API order and never re-sorts by date or id', async () => {
    const odd = [
      run({ id: 3, created_at: '2026-01-01T00:00:00Z' }),
      run({ id: 9, created_at: '2026-12-01T00:00:00Z', model: 'other' }),
      run({ id: 5, created_at: '2026-06-01T00:00:00Z' }),
    ]
    stubApi({ runs: odd })
    await mount()
    expect(Array.from(selA().options).map((o) => Number(o.value))).toEqual([3, 9, 5])
    expect(selected(selA())).toBe(3)
  })

  it('uses the video title from the video hook as the only h1', async () => {
    stubApi({ runs: MIXED, video: () => json(video({ title: 'Local vs cloud' })) })
    await mount()
    expect(screen.getAllByRole('heading', { level: 1 }).map((h) => h.textContent)).toEqual(['Local vs cloud'])
    expect(requests.filter((u) => u.pathname === `/api/videos/${ID}`)).toHaveLength(1)
  })

  it('falls back to the video_id when the video request fails, and still compares', async () => {
    stubApi({ runs: MIXED, video: () => json({ detail: 'SECRET' }, 500) })
    await mount()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(ID)
    expect(selected(selA())).toBe(4)
    expect(body()).not.toContain('SECRET')
  })

  it('falls back to the video_id when the title is empty', async () => {
    stubApi({ runs: MIXED, video: () => json(video({ title: '' })) })
    await mount()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(ID)
  })

  it('never shows one video runs under another video URL', async () => {
    stubApi({
      runs: [run({ id: 1, model: 'FIRST-MODEL' }), run({ id: 2, model: 'x' })],
      analyses: (url) => (url.pathname.includes(OTHER_ID) ? new Promise<Response>(() => {}) : undefined),
    })
    const router = await mount()
    expect(body()).toContain('FIRST-MODEL')
    await act(async () => {
      await router.navigate(`/videos/${OTHER_ID}/compare`)
    })
    await settle()
    expect(body()).not.toContain('FIRST-MODEL')
    expect(screen.getByRole('status').textContent).toMatch(/loading/i)
    expect(analysesRequests().at(-1)?.pathname).toBe(`/api/videos/${OTHER_ID}/analyses`)
  })

  it('does not poll the analyses', async () => {
    stubApi({ runs: MIXED, video: () => json(video({ active_job: { id: 1, kind: 'analyze', state: 'running' } as VideoDetailOut['active_job'] })) })
    await mount()
    const before = analysesRequests().length
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10 * 60_000)
    })
    expect(analysesRequests()).toHaveLength(before)
  })
})

describe('page states', () => {
  it('shows loading text with no pickers, tables or player while pending', async () => {
    stubApi({ runs: MIXED, analyses: () => new Promise<Response>(() => {}) })
    await mount()
    expect(screen.getByRole('status').textContent).toMatch(/loading/i)
    expect(screen.queryAllByRole('table')).toHaveLength(0)
    expect(screen.queryAllByRole('combobox')).toHaveLength(0)
    expect(screen.queryAllByRole('heading', { level: 2 })).toHaveLength(0)
  })

  it('shows Video not found with a Library link and no player on a 404', async () => {
    stubApi({ runs: [], analyses: () => json({ detail: 'video not found' }, 404) })
    await mount()
    expect(screen.getByText('Video not found')).toBeTruthy()
    expect(screen.getByRole('link', { name: /library/i }).getAttribute('href')).toBe('/')
    expect(document.querySelector('iframe')).toBeNull()
    expect(screen.queryAllByRole('combobox')).toHaveLength(0)
  })

  it.each([
    ['503', () => json({ detail: 'SECRET-STACK' }, 503)],
    ['500', () => json({ detail: 'Traceback SECRET-STACK' }, 500)],
    ['a network error', () => Promise.reject(new TypeError('SECRET-STACK failed to fetch'))],
    ['a non-JSON 200', () => new Response('<html>SECRET-STACK</html>', { status: 200 })],
    ['a JSON 200 of the wrong shape', () => json({ items: 'nope', total: 1 })],
    ['an item of the wrong shape', () => json({ items: [{ id: 1 }], limit: 50, offset: 0, total: 1 })],
  ])('shows the error state with Retry for %s, leaking nothing', async (_n, respond) => {
    stubApi({ runs: MIXED, analyses: respond })
    await mount()
    expect(screen.getByText('Could not load the analyses')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(body()).not.toContain('SECRET-STACK')
    expect(body()).not.toContain('/api/')
    expect(document.querySelector('iframe')).toBeNull()
    expect(screen.queryAllByRole('table')).toHaveLength(0)
  })

  it('a failure on page 2 shows the error and no partial comparison', async () => {
    stubApi({
      runs: history(120, () => ({ model: 'PARTIAL-MODEL' })),
      analyses: (url) => (url.searchParams.get('offset') === '50' ? json({ detail: 'x' }, 503) : undefined),
    })
    await mount()
    expect(analysesRequests()).toHaveLength(2)
    expect(screen.getByText('Could not load the analyses')).toBeTruthy()
    expect(body()).not.toContain('PARTIAL-MODEL')
    expect(screen.queryAllByRole('combobox')).toHaveLength(0)
  })

  it('Retry refetches from the first page and recovers', async () => {
    let failing = true
    const all = history(120)
    stubApi({
      runs: all,
      analyses: (url) =>
        failing && url.searchParams.get('offset') === '50' ? json({ detail: 'x' }, 503) : undefined,
    })
    await mount()
    expect(analysesRequests().map((u) => u.searchParams.get('offset'))).toEqual(['0', '50'])
    failing = false
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await settle()
    expect(analysesRequests().map((u) => u.searchParams.get('offset'))).toEqual(['0', '50', '0', '50', '100'])
    expect(selA().options).toHaveLength(120)
    expect(screen.queryByText('Could not load the analyses')).toBeNull()
  })

  it('shows the zero-run state with a link to the video detail view, no pickers and no player', async () => {
    stubApi({ runs: [] })
    await mount()
    expect(screen.getByText('This video has no analyses yet.')).toBeTruthy()
    expect(screen.getByRole('link', { name: /video/i }).getAttribute('href')).toBe(`/videos/${ID}`)
    expect(screen.queryAllByRole('combobox')).toHaveLength(0)
    expect(document.querySelector('iframe')).toBeNull()
  })

  it('shows a lone run alone, with the message, no second picker and no empty column', async () => {
    stubApi({ runs: [run({ id: 7, tldr: 'Only TLDR' })] })
    const router = await mount(`/videos/${ID}/compare?b=7`)
    expect(screen.getByText('Only one analysis exists for this video, so there is nothing to compare yet.')).toBeTruthy()
    expect(screen.queryByLabelText('Analysis B')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Swap A and B' })).toBeNull()
    expect(body()).toContain('Only TLDR')
    expect(screen.queryAllByRole('columnheader', { name: /Analysis B/ })).toHaveLength(0)
    expect(document.querySelector('iframe')).not.toBeNull()
    expect(router.state.location.search).toBe('?a=7')
  })
})

describe('pickers', () => {
  it('has two labelled native selects listing every run with the exact option text', async () => {
    stubApi({ runs: MIXED })
    await mount()
    expect(selA().tagName).toBe('SELECT')
    expect(selB().tagName).toBe('SELECT')
    expect(Array.from(selA().options).map((o) => o.textContent)).toEqual([
      'gpt · v2 · 2026-10-04 00:00 UTC · captions · #4',
      'gpt · v2 · 2026-10-03 00:00 UTC · captions · #3',
      'local · v2 · 2026-10-02 00:00 UTC · captions · #2',
      'gpt · v1 · 2026-10-01 00:00 UTC · captions · #1',
    ])
    expect(Array.from(selB().options).map((o) => o.textContent)).toEqual(
      Array.from(selA().options).map((o) => o.textContent),
    )
  })

  it('disables the run chosen on the other side in each picker', async () => {
    stubApi({ runs: MIXED })
    await mount()
    const disabled = (el: HTMLSelectElement) => Array.from(el.options).filter((o) => o.disabled).map((o) => Number(o.value))
    expect(disabled(selA())).toEqual([2])
    expect(disabled(selB())).toEqual([4])
  })

  it('defaults A to the newest and B to the newest with a different model/prompt pair', async () => {
    stubApi({ runs: MIXED })
    await mount()
    expect([selected(selA()), selected(selB())]).toEqual([4, 2])
  })

  it('defaults B to the second newest when every run has the same pair', async () => {
    stubApi({ runs: history(3) })
    await mount()
    expect([selected(selA()), selected(selB())]).toEqual([3, 2])
  })

  it('with only a, B follows the rule relative to A', async () => {
    stubApi({ runs: MIXED })
    await mount(`/videos/${ID}/compare?a=2`)
    expect([selected(selA()), selected(selB())]).toEqual([2, 4])
  })

  it('with only b, A is the newest run other than B', async () => {
    stubApi({ runs: MIXED })
    await mount(`/videos/${ID}/compare?b=4`)
    expect([selected(selA()), selected(selB())]).toEqual([3, 4])
  })

  it('swap exchanges the sides and the URL, with a push so Back walks', async () => {
    stubApi({ runs: MIXED })
    const router = await mount()
    fireEvent.click(screen.getByRole('button', { name: 'Swap A and B' }))
    await settle()
    expect([selected(selA()), selected(selB())]).toEqual([2, 4])
    expect(router.state.location.search).toBe('?a=2&b=4')
    expect(router.state.historyAction).toBe('PUSH')
    await act(async () => {
      await router.navigate(-1)
    })
    await settle()
    expect(router.state.location.search).toBe('?a=4&b=2')
    expect([selected(selA()), selected(selB())]).toEqual([4, 2])
    await act(async () => {
      await router.navigate(1)
    })
    await settle()
    expect([selected(selA()), selected(selB())]).toEqual([2, 4])
  })

  it('changing a picker pushes the new selection and Back restores the old one', async () => {
    stubApi({ runs: MIXED })
    const router = await mount()
    fireEvent.change(selA(), { target: { value: '3' } })
    await settle()
    expect(router.state.location.search).toBe('?a=3&b=2')
    expect(router.state.historyAction).toBe('PUSH')
    fireEvent.change(selB(), { target: { value: '1' } })
    await settle()
    expect(router.state.location.search).toBe('?a=3&b=1')
    expect(router.state.historyAction).toBe('PUSH')
    await act(async () => {
      await router.navigate(-1)
    })
    await settle()
    expect([selected(selA()), selected(selB())]).toEqual([3, 2])
  })

  it('ignores a picker value that is not one of the runs: no URL change, no history entry', async () => {
    stubApi({ runs: MIXED })
    const router = await mount()
    const before = router.state.location.search
    fireEvent.change(selA(), { target: { value: '999' } })
    await settle()
    expect(router.state.location.search).toBe(before)
    expect(router.state.historyAction).toBe('REPLACE')
    expect([selected(selA()), selected(selB())]).toEqual([4, 2])
  })

  it('a spoofed picker value that parses but is not a run reaches the handler and is dropped without a PUSH', async () => {
    stubApi({ runs: MIXED })
    const router = await mount()
    const before = router.state.location.search
    // jsdom reads a <select> set to a missing option back as '', which the
    // parser already rejects. Adding a real option outside React lets the
    // genuine '999' reach the change handler, so the runs.some check is hit.
    for (const sel of [selA(), selB()]) {
      const bogus = document.createElement('option')
      bogus.value = '999'
      sel.appendChild(bogus)
    }
    const actions: string[] = []
    const unsubscribe = router.subscribe((state) => actions.push(`${state.historyAction} ${state.location.search}`))
    fireEvent.change(selA(), { target: { value: '999' } })
    fireEvent.change(selB(), { target: { value: '999' } })
    await settle()
    unsubscribe()
    expect(actions.filter((a) => a.startsWith('PUSH'))).toEqual([])
    expect(actions).toEqual([])
    expect(router.state.location.search).toBe(before)
    expect([selected(selA()), selected(selB())]).toEqual([4, 2])
  })

  it('does not refetch the runs when the selection changes', async () => {
    stubApi({ runs: MIXED })
    await mount()
    const before = analysesRequests().length
    fireEvent.click(screen.getByRole('button', { name: 'Swap A and B' }))
    await settle()
    expect(analysesRequests()).toHaveLength(before)
  })

  it('keeps unrelated query params when the selection changes', async () => {
    stubApi({ runs: MIXED })
    const router = await mount(`/videos/${ID}/compare?x=1`)
    fireEvent.click(screen.getByRole('button', { name: 'Swap A and B' }))
    await settle()
    expect(new URLSearchParams(router.state.location.search).get('x')).toBe('1')
  })
})

describe('run details', () => {
  const rowOf = (name: string) => screen.getByRole('row', { name: new RegExp(`^${name}\\b`) })

  function mountPair(a: Partial<AnalysisOut>, b: Partial<AnalysisOut>) {
    stubApi({ runs: [run({ id: 2, ...a }), run({ id: 1, ...b })] })
    return mount(`/videos/${ID}/compare?a=2&b=1`)
  }

  it('has one row per field with A before B', async () => {
    await mountPair({ model: 'gpt', prompt_version: 'v2', cost_usd: 0.01234, duration_ms: 3_723_000 }, { model: 'local' })
    const details = within(region('Run details'))
    const labels = details.getAllByRole('rowheader').map((h) => h.textContent?.replace(/\s*\(differs\)$/, ''))
    expect(labels).toEqual([
      'model',
      'prompt_version',
      'chunk_strategy',
      'transcript_source',
      'created_at',
      'input_tokens',
      'output_tokens',
      'cost_usd',
      'duration_ms',
      'claim count',
      'quote count',
      'topic count',
      'claims with speaker unknown',
      'claims, confidence high',
      'claims, confidence medium',
      'claims, confidence low',
      'claims, confidence not given',
    ])
    expect(cells(rowOf('model'))).toEqual(['gpt', 'local'])
    expect(cells(rowOf('cost_usd'))).toEqual(['$0.0123', 'not recorded'])
    expect(cells(rowOf('duration_ms'))).toEqual(['1 h 2 min 3 s', 'not recorded'])
    expect(cells(rowOf('created_at'))).toEqual(['2026-10-01 08:09:10 UTC', '2026-10-01 08:09:10 UTC'])
    expect(cells(rowOf('input_tokens'))).toEqual(['1000', '1000'])
  })

  it('shows $0.0000 for a zero cost and 0 s for a zero duration', async () => {
    await mountPair({ cost_usd: 0, duration_ms: 0 }, { cost_usd: 0, duration_ms: 999 })
    expect(cells(rowOf('cost_usd'))).toEqual(['$0.0000', '$0.0000'])
    expect(cells(rowOf('duration_ms'))).toEqual(['0 s', '0 s'])
  })

  it('marks differing rows with the text "differs" and equal rows without it', async () => {
    await mountPair({ model: 'gpt', input_tokens: 5 }, { model: 'local', input_tokens: 5 })
    expect(rowOf('model').textContent).toContain('differs')
    expect(rowOf('input_tokens').textContent).not.toContain('differs')
    expect(rowOf('prompt_version').textContent).not.toContain('differs')
  })

  it('marks cost as differing when only one side recorded it', async () => {
    await mountPair({ cost_usd: 0 }, { cost_usd: null })
    expect(rowOf('cost_usd').textContent).toContain('differs')
  })

  it('counts claims, quotes, topics, unknown speakers and confidence per side', async () => {
    await mountPair(
      {
        claims: [
          claim({ confidence: 'high' }),
          claim({ confidence: 'high', speaker: 'unknown' }),
          claim({ confidence: 'medium' }),
          claim({ confidence: null }),
          claim({ confidence: 'weird' }),
        ],
        quotes: [quote(), quote()],
        topics: [{ seq: 0, title: 't', summary: null, start_sec: 1 }],
      },
      {},
    )
    expect(cells(rowOf('claim count'))).toEqual(['5', '0'])
    expect(cells(rowOf('quote count'))).toEqual(['2', '0'])
    expect(cells(rowOf('topic count'))).toEqual(['1', '0'])
    expect(cells(rowOf('claims with speaker unknown'))).toEqual(['1', '0'])
    expect(cells(rowOf('claims, confidence high'))).toEqual(['2', '0'])
    expect(cells(rowOf('claims, confidence medium'))).toEqual(['1', '0'])
    expect(cells(rowOf('claims, confidence low'))).toEqual(['0', '0'])
    expect(cells(rowOf('claims, confidence not given'))).toEqual(['2', '0'])
  })

  it('has Analysis A and Analysis B column headers that also name the run', async () => {
    await mountPair({ model: 'gpt', prompt_version: 'v2' }, { model: 'local', prompt_version: 'v1' })
    const heads = within(region('Run details')).getAllByRole('columnheader').map((h) => h.textContent)
    expect(heads[1]).toContain('Analysis A')
    expect(heads[1]).toContain('gpt · v2')
    expect(heads[2]).toContain('Analysis B')
    expect(heads[2]).toContain('local · v1')
  })
})

describe('notices', () => {
  const same = 'Same model and prompt version: differences show run-to-run variation.'

  it('warns about different transcripts with both sources, and not otherwise', async () => {
    stubApi({
      runs: [
        run({ id: 2, transcript_id: 2, transcript_source: 'whisper', model: 'a' }),
        run({ id: 1, transcript_id: 1, transcript_source: 'captions', model: 'b' }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(
      screen.getByText(
        'These runs used different transcripts (whisper vs captions), so differences may come from the transcript rather than the model or prompt.',
      ),
    ).toBeTruthy()
  })

  it('warns when only the transcript id differs, naming equal sources', async () => {
    stubApi({ runs: [run({ id: 2, transcript_id: 2, model: 'a' }), run({ id: 1, transcript_id: 1, model: 'b' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(screen.getByText(/different transcripts \(captions vs captions\)/)).toBeTruthy()
  })

  it('does not warn about transcripts when the transcript id matches even if sources differ in text', async () => {
    stubApi({
      runs: [run({ id: 2, model: 'a', transcript_source: 'whisper' }), run({ id: 1, model: 'b' })],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(body()).not.toContain('different transcripts')
  })

  it('warns when the chunk strategy differs, separately', async () => {
    stubApi({ runs: [run({ id: 2, chunk_strategy: 'fixed', model: 'a' }), run({ id: 1, chunk_strategy: 'topic', model: 'b' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(screen.getAllByText(/chunk/i, { selector: 'p' }).length).toBeGreaterThan(0)
    expect(body()).not.toContain('different transcripts')
    expect(body()).not.toContain(same)
  })

  it('shows the same-model notice only when model and prompt version are both equal', async () => {
    stubApi({ runs: history(2) })
    await mount()
    expect(screen.getByText(same)).toBeTruthy()
  })

  it.each([
    ['model', { model: 'other' }],
    ['prompt_version', { prompt_version: 'v9' }],
  ])('omits the same-model notice when only %s differs', async (_n, over) => {
    stubApi({ runs: [run({ id: 2, ...over }), run({ id: 1 })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(body()).not.toContain(same)
  })

  it('shows no notice for runs that differ only in model', async () => {
    stubApi({ runs: [run({ id: 2, model: 'x' }), run({ id: 1 })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(body()).not.toContain('different transcripts')
    expect(body()).not.toContain('chunk strategy')
  })
})

describe('what each side shows', () => {
  it('shows both TL;DRs side by side in one row, keeping line breaks', async () => {
    stubApi({ runs: [run({ id: 2, tldr: 'one\ntwo' }), run({ id: 1, tldr: 'three', model: 'x' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const row = within(region('TL;DR')).getAllByRole('row').at(-1) as HTMLElement
    const [a, b] = within(row).getAllByRole('cell')
    expect(a?.textContent).toBe('one\ntwo')
    expect(b?.textContent).toBe('three')
    expect(getComputedStyle((a as HTMLElement).firstElementChild as HTMLElement).whiteSpace).toBe('pre-line')
  })

  it('lists speakers per side with roles, following the roster rules', async () => {
    stubApi({
      runs: [
        run({
          id: 2,
          speaker_roster: {
            speakers: [
              { name: 'Ada', role: 'host' },
              { name: 'Bob', role: 42 },
              { name: '   ', role: 'ghost' },
              { role: 'nameless' },
              'string entry',
              null,
              { name: 'Cy', role: '' },
            ],
          },
        }),
        run({ id: 1, model: 'x', speaker_roster: null }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const row = within(region('Speakers')).getAllByRole('row').at(-1) as HTMLElement
    const [a, b] = within(row).getAllByRole('cell')
    expect(within(a as HTMLElement).getAllByRole('listitem').map((l) => l.textContent)).toEqual(['Ada (host)', 'Bob', 'Cy'])
    expect(b?.textContent).toBe('No roster')
  })

  it.each([
    ['null', null],
    ['an array', ['x']],
    ['a string', 'x'],
    ['no speakers array', { speakers: 'x' }],
    ['an empty speakers array', { speakers: [] }],
    ['only invalid entries', { speakers: [{ name: 5 }, {}] }],
  ])('shows "No roster" for %s', async (_n, roster) => {
    stubApi({ runs: [run({ id: 2, speaker_roster: roster as AnalysisOut['speaker_roster'] }), run({ id: 1, model: 'x' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const row = within(region('Speakers')).getAllByRole('row').at(-1) as HTMLElement
    expect(within(row).getAllByRole('cell')[0]?.textContent).toBe('No roster')
  })

  it('lists topics per side in seq order with title, summary and timestamp, unaligned', async () => {
    stubApi({
      runs: [
        run({
          id: 2,
          topics: [
            { seq: 1, title: 'Second', summary: 'sum two', start_sec: 400 },
            { seq: 0, title: 'First', summary: null, start_sec: 5 },
            { seq: 2, title: 'Untimed', summary: null, start_sec: null },
          ],
        }),
        run({ id: 1, model: 'x', topics: [{ seq: 0, title: 'Other', summary: null, start_sec: 61 }] }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const row = within(region('Topics')).getAllByRole('row').at(-1) as HTMLElement
    const [a, b] = within(row).getAllByRole('cell')
    const text = (a as HTMLElement).textContent ?? ''
    expect(text.indexOf('First')).toBeLessThan(text.indexOf('Second'))
    expect(text.indexOf('Second')).toBeLessThan(text.indexOf('Untimed'))
    expect(text).toContain('sum two')
    expect(within(a as HTMLElement).getByRole('button', { name: 'Seek to 6:40' })).toBeTruthy()
    expect(within(b as HTMLElement).getByRole('button', { name: 'Seek to 1:01' })).toBeTruthy()
    expect(within(a as HTMLElement).getAllByRole('button')).toHaveLength(2)
  })

  it('shows a placeholder when a side has no topics', async () => {
    stubApi({ runs: [run({ id: 2 }), run({ id: 1, model: 'x' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const row = within(region('Topics')).getAllByRole('row').at(-1) as HTMLElement
    expect(cells(row)).toEqual(['No topics extracted.', 'No topics extracted.'])
  })
})

describe('claims and quotes aligned by time window', () => {
  const claimsTable = () => within(region('Claims'))

  function rows() {
    return claimsTable()
      .getAllByRole('row')
      .slice(1)
      .map((r) => [within(r).getByRole('rowheader').textContent, ...cells(r)])
  }

  it('aligns claims by window and keeps a one-sided window visible', async () => {
    stubApi({
      runs: [
        run({
          id: 2,
          claims: [claim({ text: 'A early', start_sec: 10 }), claim({ text: 'A boundary', start_sec: 300 })],
        }),
        run({
          id: 1,
          model: 'x',
          claims: [claim({ text: 'B early', start_sec: 299.999 }), claim({ text: 'B late', start_sec: 4000 })],
        }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const r = rows()
    expect(r.map((x) => x[0])).toEqual(['0:00–5:00', '5:00–10:00', '1:05:00–1:10:00'])
    expect(within(claimsTable().getAllByRole('row')[0] as HTMLElement).getAllByRole('columnheader')[0]?.textContent).toBe('Time window')
    expect(r[0]?.[1]).toContain('A early')
    expect(r[0]?.[2]).toContain('B early')
    expect(r[1]?.[1]).toContain('A boundary')
    expect(r[1]?.[2]).toBe('No claims in this window')
    expect(r[2]?.[1]).toBe('No claims in this window')
    expect(r[2]?.[2]).toContain('B late')
  })

  it('puts items without a usable timestamp in a final No timestamp row', async () => {
    stubApi({
      runs: [
        run({ id: 2, claims: [claim({ text: 'untimed', start_sec: null }), claim({ text: 'timed', start_sec: 1 })] }),
        run({ id: 1, model: 'x', claims: [claim({ text: 'neg', start_sec: -5 })] }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const r = rows()
    expect(r.map((x) => x[0])).toEqual(['0:00–5:00', 'No timestamp'])
    expect(r[1]?.[1]).toContain('untimed')
    expect(r[1]?.[2]).toContain('neg')
    expect(within(claimsTable().getAllByRole('row').at(-1) as HTMLElement).queryAllByRole('button')).toHaveLength(0)
  })

  it('shows the empty message and no table when neither side has claims', async () => {
    stubApi({ runs: [run({ id: 2 }), run({ id: 1, model: 'x' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(claimsTable().getByText('No claims extracted in either analysis.')).toBeTruthy()
    expect(claimsTable().queryAllByRole('table')).toHaveLength(0)
  })

  it('shows Unattributed, the confidence label, and never drops unknown-speaker claims', async () => {
    stubApi({
      runs: [
        run({
          id: 2,
          claims: [
            claim({ text: 'c1', speaker: 'unknown', confidence: 'low', start_sec: 1 }),
            claim({ text: 'c2', speaker: 'Ada', confidence: null, start_sec: 2 }),
            claim({ text: 'c3', speaker: 'Bob', confidence: 'medium', start_sec: 3 }),
          ],
        }),
        run({ id: 1, model: 'x' }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const items = claimsTable().getAllByRole('listitem')
    expect(items).toHaveLength(3)
    expect(items[0]?.textContent).toContain('Unattributed')
    expect(items[0]?.textContent).not.toContain('unknown')
    expect(items[0]?.textContent).toContain('Confidence: low')
    expect(items[1]?.textContent).not.toContain('Confidence')
    expect(items[1]?.textContent).toContain('Ada')
    expect(items[2]?.textContent).toContain('Confidence: medium')
  })

  it('aligns quotes with the same rule, speaker rule and empty message', async () => {
    stubApi({
      runs: [
        run({ id: 2, quotes: [quote({ text: 'qa', speaker: 'unknown', start_sec: 310 })] }),
        run({ id: 1, model: 'x', quotes: [quote({ text: 'qb', start_sec: 20 })] }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const q = within(region('Quotes'))
    const r = q.getAllByRole('row').slice(1).map((x) => [within(x).getByRole('rowheader').textContent, ...cells(x)])
    expect(r.map((x) => x[0])).toEqual(['0:00–5:00', '5:00–10:00'])
    expect(r[0]?.[1]).toBe('No quotes in this window')
    expect(r[0]?.[2]).toContain('qb')
    expect(r[1]?.[1]).toContain('Unattributed')
    expect(r[1]?.[2]).toBe('No quotes in this window')
  })

  it('shows the empty quotes message when neither side has quotes', async () => {
    stubApi({ runs: [run({ id: 2 }), run({ id: 1, model: 'x' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(within(region('Quotes')).getByText('No quotes extracted in either analysis.')).toBeTruthy()
  })

  it('a click on a B-side claim timestamp seeks the shared player to that start_sec', async () => {
    stubApi({
      runs: [
        run({ id: 2, claims: [claim({ text: 'A', start_sec: 10 })] }),
        run({ id: 1, model: 'x', claims: [claim({ text: 'B', start_sec: 4321.5 })] }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const bCell = within(claimsTable().getAllByRole('row').at(-1) as HTMLElement).getAllByRole('cell')[1] as HTMLElement
    fireEvent.click(within(bCell).getByRole('button', { name: 'Seek to 1:12:01' }))
    expect(seeks).toEqual([4321.5])
  })

  it('renders every non-null topic, claim and quote start_sec through the Timestamp button', async () => {
    stubApi({
      runs: [
        run({
          id: 2,
          topics: [{ seq: 0, title: 't', summary: null, start_sec: 61 }],
          claims: [claim({ start_sec: 62 })],
          quotes: [quote({ start_sec: 63 })],
        }),
        run({ id: 1, model: 'x', claims: [claim({ start_sec: 64 })], quotes: [quote({ start_sec: null })] }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const buttons = screen.getAllByRole('button', { name: /^Seek to / }).map((b) => b.getAttribute('aria-label'))
    expect(buttons.sort()).toEqual(['Seek to 1:01', 'Seek to 1:02', 'Seek to 1:03', 'Seek to 1:04'])
  })
})

describe('player', () => {
  it('embeds exactly one player for the validated video id', async () => {
    stubApi({ runs: MIXED })
    await mount(`/videos/${DASH_ID}/compare`)
    const frames = document.querySelectorAll('iframe')
    expect(frames).toHaveLength(1)
    expect(frames[0]?.getAttribute('src')).toContain(`/embed/${DASH_ID}`)
  })
})

describe('untrusted text', () => {
  const XSS1 = '<script>alert(1)</script>'
  const XSS2 = '"><img src=x onerror=alert(1)>'

  it('renders hostile stored text and titles as literal text', async () => {
    const hostile = (tag: string): Partial<AnalysisOut> => ({
      model: tag,
      prompt_version: tag,
      transcript_source: tag,
      chunk_strategy: tag,
      tldr: tag,
      speaker_roster: { speakers: [{ name: tag, role: tag }] },
      topics: [{ seq: 0, title: tag, summary: tag, start_sec: 1 }],
      claims: [claim({ text: tag, speaker: tag, confidence: tag })],
      quotes: [quote({ text: tag, speaker: tag })],
    })
    stubApi({
      runs: [run({ id: 2, ...hostile(XSS1) }), run({ id: 1, transcript_id: 9, ...hostile(XSS2) })],
      video: () => json(video({ title: XSS2 })),
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(document.querySelector('script')).toBeNull()
    expect(document.querySelector('img')).toBeNull()
    expect(document.body.querySelectorAll('[onerror]')).toHaveLength(0)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(XSS2)
    expect(body()).toContain(XSS1)
    expect(body()).toContain(XSS2)
    // Hostile text inside an option and in a notice stays text too.
    expect(selA().options[0]?.textContent).toContain(XSS1)
  })

  it('renders non-ASCII text unchanged', async () => {
    const text = 'Café 日本語 😀 עברית مرحبا'
    stubApi({
      runs: [
        run({ id: 2, tldr: text, claims: [claim({ text, speaker: text })] }),
        run({ id: 1, model: 'x' }),
      ],
      video: () => json(video({ title: text })),
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(text)
    expect(region('TL;DR').textContent).toContain(text)
    expect(region('Claims').textContent).toContain(text)
  })

  it('builds no href or src from stored text', async () => {
    stubApi({
      runs: [
        run({ id: 2, model: 'javascript:alert(1)', tldr: 'http://evil.example/', claims: [claim({ text: 'http://evil.example/x' })] }),
        run({ id: 1, model: 'x' }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const hrefs = Array.from(document.querySelectorAll('[href]')).map((e) => e.getAttribute('href'))
    expect(hrefs.every((h) => h === '/' || h === `/videos/${ID}`)).toBe(true)
    const srcs = Array.from(document.querySelectorAll('[src]')).map((e) => e.tagName)
    expect(srcs).toEqual(['IFRAME'])
    expect(document.body.innerHTML).not.toContain('evil.example"')
  })
})

describe('accessibility', () => {
  it('has one h1 and one h2 per section, in order', async () => {
    stubApi({ runs: MIXED })
    await mount()
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
    expect(screen.getAllByRole('heading', { level: 2 }).map((h) => h.textContent)).toEqual([
      'Run details',
      'TL;DR',
      'Speakers',
      'Topics',
      'Claims',
      'Quotes',
    ])
  })

  it('puts the A cell before the B cell and headers on every table', async () => {
    stubApi({
      runs: [
        run({ id: 2, claims: [claim({ text: 'AAA', start_sec: 1 })] }),
        run({ id: 1, model: 'x', claims: [claim({ text: 'BBB', start_sec: 1 })] }),
      ],
    })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    const tables = screen.getAllByRole('table')
    expect(tables.length).toBeGreaterThanOrEqual(2)
    for (const t of tables) {
      const heads = within(t).getAllByRole('columnheader').map((h) => h.textContent ?? '')
      const a = heads.findIndex((h) => h.includes('Analysis A'))
      const b = heads.findIndex((h) => h.includes('Analysis B'))
      expect(a).toBeGreaterThanOrEqual(0)
      expect(b).toBeGreaterThan(a)
    }
    const row = within(region('Claims')).getAllByRole('row')[1] as HTMLElement
    const [aCell, bCell] = within(row).getAllByRole('cell')
    expect(aCell?.textContent).toContain('AAA')
    expect(bCell?.textContent).toContain('BBB')
    expect(within(row).getByRole('rowheader').textContent).toBe('0:00–5:00')
  })

  it('has labelled pickers and keyboard-operable (native) controls', async () => {
    stubApi({ runs: [run({ id: 2, claims: [claim()] }), run({ id: 1, model: 'x' })] })
    await mount(`/videos/${ID}/compare?a=2&b=1`)
    expect(screen.getByLabelText('Analysis A')).toBeTruthy()
    expect(screen.getByLabelText('Analysis B')).toBeTruthy()
    const swap = screen.getByRole('button', { name: 'Swap A and B' })
    expect(swap.tagName).toBe('BUTTON')
    for (const b of screen.getAllByRole('button')) expect(b.tagName).toBe('BUTTON')
    expect(screen.getAllByRole('button').every((b) => b.getAttribute('tabindex') !== '-1')).toBe(true)
  })
})

describe('real route table', () => {
  it('mounts a player provider so timestamps are seek buttons in the app', async () => {
    stubApi({ runs: [run({ id: 2, claims: [claim({ start_sec: 75 })] }), run({ id: 1, model: 'x' })] })
    const router = createMemoryRouter(routes, { initialEntries: [`/videos/${ID}/compare?a=2&b=1`] })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    )
    await settle()
    expect(screen.getByRole('button', { name: 'Seek to 1:15' })).toBeTruthy()
  })
})
