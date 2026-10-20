import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen, within } from '@testing-library/react'
import { setupUser } from '../test/user'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { components } from '../api'
import { Search } from './Search'

const user = setupUser

type SearchResult = components['schemas']['SearchResult']
type SearchPage = components['schemas']['SearchPage']

function hit(n: number, over: Partial<SearchResult> = {}): SearchResult {
  return {
    video_id: `v${String(n).padStart(10, '0')}`,
    title: `Title ${n}`,
    channel_id: 'UC' + 'a'.repeat(22),
    channel_title: 'Chan A',
    published_at: '2026-09-01T12:00:00Z',
    duration_sec: 60,
    unavailable: null,
    transcript_source: 'whisper',
    matched_in: ['transcript'],
    excerpts: [{ spans: [{ text: `excerpt ${n}`, match: false }], start_sec: null }],
    ...over,
  }
}

function page(results: SearchResult[], hasMore = false, offset = 0): SearchPage {
  return { results, limit: 20, offset, has_more: hasMore }
}

function hits(from: number, count: number): SearchResult[] {
  return Array.from({ length: count }, (_, i) => hit(from + i))
}

let requests: URL[]
type Reply = SearchPage | Response | Promise<Response>

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

function stubFetch(respond: (u: URL) => Reply) {
  requests = []
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost')
      requests.push(url)
      const r = respond(url)
      return r instanceof Response || r instanceof Promise ? Promise.resolve(r) : Promise.resolve(json(r))
    }),
  )
}

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

async function mount(entry = '/search') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [
      { path: '/search', element: <Search /> },
      { path: '/videos/:videoId', element: <p>detail page</p> },
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

const input = () => screen.getByLabelText(/search/i) as HTMLInputElement
const live = () => document.querySelector('[aria-live="polite"]') as HTMLElement
const last = () => requests[requests.length - 1] as URL

async function submit(value: string) {
  await user().clear(input())
  await user().type(input(), value)
  await user().keyboard('{Enter}')
  await settle()
}

beforeEach(() => {
  vi.useFakeTimers()
})

describe('idle state and the form', () => {
  it('shows the hint, an empty labelled search box and sends no request', async () => {
    stubFetch(() => page([]))
    await mount()
    expect(requests).toHaveLength(0)
    expect(screen.getByText('Search what was said in processed videos')).toBeTruthy()
    expect(screen.getByRole('search')).toBeTruthy()
    expect(input().type).toBe('search')
    expect(input().value).toBe('')
    expect(screen.getByRole('button', { name: 'Search' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Search')
  })

  it.each(['/search?q=', '/search?q=%20%20%20', '/search?page=3'])('%s is idle with no request', async (entry) => {
    stubFetch(() => page([]))
    await mount(entry)
    expect(requests).toHaveLength(0)
    expect(screen.getByText('Search what was said in processed videos')).toBeTruthy()
  })

  it('does not search on each keystroke', async () => {
    stubFetch(() => page([]))
    await mount()
    await user().type(input(), 'ca')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000)
    })
    expect(requests).toHaveLength(0)
  })
})

describe('submitting', () => {
  it('trims, pushes ?q=, keeps the trimmed text in the box and requests page 1', async () => {
    stubFetch(() => page(hits(1, 2)))
    const router = await mount()
    await submit('  hello world \n')
    expect(router.state.location.search).toBe('?q=hello+world')
    expect(router.state.historyAction).toBe('PUSH')
    expect(input().value).toBe('hello world')
    expect(requests).toHaveLength(1)
    expect(last().pathname).toBe('/api/search')
    expect(Object.fromEntries(last().searchParams)).toEqual({ q: 'hello world', limit: '20', offset: '0' })
  })

  it('submits with the button as well as with Enter', async () => {
    stubFetch(() => page([]))
    await mount()
    await user().type(input(), 'cats')
    await user().click(screen.getByRole('button', { name: 'Search' }))
    await settle()
    expect(last().searchParams.get('q')).toBe('cats')
  })

  it('sends a hostile q only through URLSearchParams', async () => {
    stubFetch(() => page([]))
    await mount()
    await submit('a&limit=50#x')
    expect(last().searchParams.get('q')).toBe('a&limit=50#x')
    expect(last().searchParams.get('limit')).toBe('20')
    expect(last().searchParams.getAll('limit')).toEqual(['20'])
    expect(last().hash).toBe('')
  })

  it.each(['the', '-dogs'])('sends %j unchanged', async (q) => {
    stubFetch(() => page([]))
    await mount()
    await submit(q)
    expect(last().searchParams.get('q')).toBe(q)
  })

  it('submitting the same query again adds no history entry and resets to page 1 otherwise', async () => {
    stubFetch(() => page(hits(1, 20), true))
    const router = await mount('/search')
    await submit('cats')
    await submit('cats')
    expect(requests).toHaveLength(1)
    await router.navigate(-1)
    await settle()
    expect(router.state.location.search).toBe('')
    expect(screen.getByText('Search what was said in processed videos')).toBeTruthy()
  })

  it('returns to page 1 when the query is resubmitted from page 3', async () => {
    stubFetch(() => page(hits(1, 3)))
    const router = await mount('/search?q=cats&page=3')
    await submit('cats')
    expect(router.state.location.search).toBe('?q=cats')
  })

  it('a whitespace-only submit sends nothing and goes back to the idle state', async () => {
    stubFetch(() => page(hits(1, 1)))
    const router = await mount('/search?q=cats')
    expect(requests).toHaveLength(1)
    await submit('   ')
    expect(requests).toHaveLength(1)
    expect(router.state.location.search).toBe('')
    expect(screen.getByText('Search what was said in processed videos')).toBeTruthy()
  })

  it('restores the query and results on Back and Forward', async () => {
    stubFetch((u) => page([hit(u.searchParams.get('q') === 'a' ? 1 : 2)]))
    const router = await mount()
    await submit('a')
    await submit('b')
    expect(screen.getByText('Title 2')).toBeTruthy()
    await router.navigate(-1)
    await settle()
    expect(input().value).toBe('a')
    expect(screen.getByText('Title 1')).toBeTruthy()
    await router.navigate(1)
    await settle()
    expect(input().value).toBe('b')
    expect(screen.getByText('Title 2')).toBeTruthy()
  })

  it('restores query and page from a deep link', async () => {
    stubFetch(() => page(hits(21, 20), true, 20))
    await mount('/search?q=deep%20link&page=2')
    expect(input().value).toBe('deep link')
    expect(last().searchParams.get('offset')).toBe('20')
    expect(screen.getByText('Results 21–40')).toBeTruthy()
  })
})

describe('client-side query validation', () => {
  it.each([
    ['200 ASCII', 'a'.repeat(200)],
    ['200 emoji', '\u{1F600}'.repeat(200)],
  ])('sends %s', async (_n, q) => {
    stubFetch(() => page([]))
    await mount()
    await submit(q)
    expect(requests).toHaveLength(1)
    expect(last().searchParams.get('q')).toBe(q)
    expect(screen.queryByText('Search terms are limited to 200 characters')).toBeNull()
  })

  it.each([
    ['201 ASCII', 'a'.repeat(201)],
    ['201 emoji', '\u{1F600}'.repeat(201)],
  ])('does not send %s, shows the message and keeps the text', async (_n, q) => {
    stubFetch(() => page([]))
    const router = await mount()
    await submit(q)
    expect(requests).toHaveLength(0)
    expect(screen.getByText('Search terms are limited to 200 characters')).toBeTruthy()
    expect(input().value).toBe(q)
    expect(router.state.location.search).toBe('')
  })

  it('clears the message once a valid query is submitted', async () => {
    stubFetch(() => page([]))
    await mount()
    await submit('a'.repeat(201))
    await submit('ok')
    expect(screen.queryByText('Search terms are limited to 200 characters')).toBeNull()
    expect(requests).toHaveLength(1)
  })

  it('does not send a query containing NUL', async () => {
    stubFetch(() => page([]))
    await mount()
    await submit('a\u0000b')
    expect(requests).toHaveLength(0)
    expect(input().value).toBe('a\u0000b')
    expect(document.body.textContent).not.toContain('\u0000')
  })

  it('leaves other control characters to the server', async () => {
    stubFetch(() => page([]))
    await mount()
    await submit('a\u0001b')
    expect(last().searchParams.get('q')).toBe('a\u0001b')
  })
})

describe('URL parameters are untrusted', () => {
  it.each([
    ['over-long', `/search?q=${'a'.repeat(201)}MARKER`],
    ['NUL', '/search?q=a%00MARKER'],
  ])('drops a %s q, corrects the URL with replace and shows nothing of it', async (_n, entry) => {
    stubFetch(() => page([]))
    const router = await mount(entry)
    expect(requests).toHaveLength(0)
    expect(router.state.location.search).toBe('')
    expect(router.state.historyAction).toBe('REPLACE')
    expect(input().value).toBe('')
    expect(document.body.textContent).not.toContain('MARKER')
    expect(screen.getByText('Search what was said in processed videos')).toBeTruthy()
  })

  it('does not throw or blank the page on a malformed percent-encoding', async () => {
    stubFetch(() => page([]))
    await mount('/search?q=%E0%A4%A')
    expect(screen.getByRole('search')).toBeTruthy()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Search')
  })

  it('accepts page 51 and falls back from page 52', async () => {
    stubFetch(() => page([]))
    await mount('/search?q=a&page=51')
    expect(last().searchParams.get('offset')).toBe('1000')
    vi.unstubAllGlobals()
    stubFetch(() => page([]))
    const router = await mount('/search?q=a&page=52')
    expect(last().searchParams.get('offset')).toBe('0')
    expect(router.state.location.search).toBe('?q=a')
  })

  it.each(['0', '-1', '1.5', 'abc', '1e3', '', '999999999999999999999'])(
    'falls back to page 1 for page=%j and corrects the URL',
    async (p) => {
      stubFetch(() => page(hits(1, 1)))
      const router = await mount(`/search?q=a&page=${p}`)
      expect(last().searchParams.get('offset')).toBe('0')
      expect(router.state.location.search).toBe('?q=a')
      expect(router.state.historyAction).toBe('REPLACE')
    },
  )

  it('never forwards or shows unknown parameters or a bad page value', async () => {
    stubFetch(() => page(hits(1, 1)))
    await mount('/search?q=a&page=MARKER_PAGE&extra=MARKER_EXTRA')
    expect(requests.map((r) => r.href).join()).not.toContain('MARKER')
    expect(document.body.textContent).not.toContain('MARKER')
    expect([...last().searchParams.keys()].sort()).toEqual(['limit', 'offset', 'q'])
  })
})

describe('a result', () => {
  it('shows title link, channel, UTC date and the excerpt marks in order', async () => {
    stubFetch(() =>
      page([
        hit(1, {
          video_id: 'abc-_Z09xyz',
          title: 'The Zebra',
          channel_title: 'Wildlife',
          published_at: '2026-09-01T23:30:00Z',
          excerpts: [
            {
              spans: [
                { text: 'the ', match: false },
                { text: 'zebra', match: true },
                { text: ' ran fast', match: false },
              ],
              start_sec: null,
            },
            {
              spans: [
                { text: 'another ', match: false },
                { text: 'zebra', match: true },
              ],
              start_sec: null,
            },
          ],
        }),
      ]),
    )
    await mount('/search?q=zebra')
    const item = screen.getAllByRole('listitem')[0] as HTMLElement
    const link = within(item).getByRole('link', { name: 'The Zebra' })
    expect(link.getAttribute('href')).toBe('/videos/abc-_Z09xyz')
    expect(item.textContent).toContain('Wildlife')
    expect(item.textContent).toContain('2026-09-01')
    expect(item.textContent).not.toContain('2026-09-02')
    const marks = [...item.querySelectorAll('mark')]
    expect(marks.map((m) => m.textContent)).toEqual(['zebra', 'zebra'])
    expect(item.querySelector('p')?.textContent).toBe('the zebra ran fast … another zebra')
  })

  it('renders as an ordered list in the server order', async () => {
    stubFetch(() => page([hit(3), hit(1), hit(2)]))
    await mount('/search?q=a')
    expect(document.querySelector('ol')).not.toBeNull()
    expect(screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)).toEqual([
      'Title 3',
      'Title 1',
      'Title 2',
    ])
  })

  it('encodes the video id in the link', async () => {
    stubFetch(() => page([hit(1, { video_id: 'a/b?c#d%e' })]))
    await mount('/search?q=a')
    expect(screen.getByRole('link', { name: 'Title 1' }).getAttribute('href')).toBe(
      `/videos/${encodeURIComponent('a/b?c#d%e')}`,
    )
  })

  it('falls back to the video id, channel id and "Date unknown"', async () => {
    stubFetch(() =>
      page([
        hit(1, { title: null, channel_title: null, channel_id: 'UCchan', published_at: null }),
        hit(2, { title: null, channel_title: null, channel_id: null, published_at: null }),
      ]),
    )
    await mount('/search?q=a')
    const [one, two] = screen.getAllByRole('listitem') as [HTMLElement, HTMLElement]
    expect(within(one).getByRole('link', { name: 'v0000000001' })).toBeTruthy()
    expect(one.textContent).toContain('UCchan')
    expect(one.textContent).toContain('Date unknown')
    expect(two.textContent).toContain('Unknown channel')
  })

  it('shows dates in UTC whatever the local zone', async () => {
    expect(new Date('2026-09-01T00:30:00Z').getDate()).toBe(31) // local zone is Los Angeles
    stubFetch(() =>
      page([
        hit(1, { published_at: '2026-09-01T00:30:00Z' }),
        hit(2, { published_at: '2026-09-01T23:30:00Z' }),
      ]),
    )
    await mount('/search?q=a')
    const [one, two] = screen.getAllByRole('listitem') as [HTMLElement, HTMLElement]
    expect(one.textContent).toContain('2026-09-01')
    expect(one.textContent).not.toContain('2026-08-31')
    expect(two.textContent).toContain('2026-09-01')
    expect(two.textContent).not.toContain('2026-09-02')
  })

  it('marks an unavailable video but keeps it listed and linked', async () => {
    stubFetch(() => page([hit(1, { unavailable: 'removed' }), hit(2)]))
    await mount('/search?q=a')
    const [one, two] = screen.getAllByRole('listitem') as [HTMLElement, HTMLElement]
    expect(one.textContent).toContain('Unavailable on YouTube')
    expect(within(one).getByRole('link', { name: 'Title 1' })).toBeTruthy()
    expect(two.textContent).not.toContain('Unavailable on YouTube')
  })

  it('renders no excerpt block for excerpts: []', async () => {
    stubFetch(() => page([hit(1, { excerpts: [] })]))
    await mount('/search?q=a')
    const item = screen.getByRole('listitem')
    expect(within(item).getByRole('link', { name: 'Title 1' })).toBeTruthy()
    expect(item.querySelector('p')).toBeNull()
    expect(item.querySelector('mark')).toBeNull()
  })

  it('renders nothing for empty spans and fragments and no stray separator or mark', async () => {
    stubFetch(() =>
      page([
        hit(1, {
          excerpts: [
            { spans: [{ text: '', match: true }], start_sec: null },
            { spans: [], start_sec: null },
            {
              spans: [
                { text: '', match: false },
                { text: 'solo', match: false },
              ],
              start_sec: null,
            },
          ],
        }),
      ]),
    )
    await mount('/search?q=a')
    const item = screen.getByRole('listitem')
    expect(item.querySelector('mark')).toBeNull()
    expect(item.querySelector('p')?.textContent).toBe('solo')
  })

  it('renders a fragment with no matched span as plain text', async () => {
    stubFetch(() => page([hit(1, { excerpts: [{ spans: [{ text: 'plain words', match: false }], start_sec: null }] })]))
    await mount('/search?q=a')
    expect(screen.getByText('plain words')).toBeTruthy()
    expect(document.querySelector('mark')).toBeNull()
  })

  it('shows HTML-looking stored text literally and creates no elements', async () => {
    const evil = '<img src=x onerror=alert(1)>'
    stubFetch(() =>
      page([
        hit(1, {
          title: evil,
          channel_title: '<b>fake</b>',
          excerpts: [
            {
              spans: [
                { text: evil, match: false },
                { text: '<b>fake</b>', match: true },
              ],
              start_sec: null,
            },
          ],
        }),
      ]),
    )
    await mount('/search?q=a')
    expect(document.querySelector('img')).toBeNull()
    expect(document.querySelector('b')).toBeNull()
    expect(document.querySelectorAll('mark')).toHaveLength(1)
    expect(document.querySelector('mark')?.textContent).toBe('<b>fake</b>')
    expect(screen.getByRole('link', { name: evil })).toBeTruthy()
    expect(screen.getByText('<b>fake</b>', { selector: 'span, small, p' })).toBeTruthy()
  })

  it('never echoes the query as markup', async () => {
    stubFetch(() => page([]))
    await mount(`/search?q=${encodeURIComponent('<img src=x onerror=alert(1)>')}`)
    expect(document.querySelector('img')).toBeNull()
    expect(document.body.textContent).toContain('No transcripts match this search')
    expect(document.body.textContent).not.toContain('onerror')
  })

  it('renders no timestamp link for an excerpt with start_sec: null', async () => {
    stubFetch(() => page([hit(1)]))
    await mount('/search?q=a')
    expect(screen.getAllByRole('link')).toHaveLength(1)
    expect(document.querySelector('time')).toBeNull()
  })

  const withStarts = (starts: (number | null)[]) =>
    hit(1, {
      video_id: 'abc-_Z09xyz',
      excerpts: starts.map((start_sec, i) => ({
        spans: [{ text: `part ${i}`, match: i === 0 }],
        start_sec,
      })),
    })

  it('links each excerpt with a start to ?t= whole seconds, labelled by its time', async () => {
    stubFetch(() => page([withStarts([12.5, 600, 3723])]))
    await mount('/search?q=a')
    const item = screen.getByRole('listitem')
    const links = within(item).getAllByRole('link', { name: /^Open at/ })
    expect(links.map((l) => l.getAttribute('href'))).toEqual([
      '/videos/abc-_Z09xyz?t=12',
      '/videos/abc-_Z09xyz?t=600',
      '/videos/abc-_Z09xyz?t=3723',
    ])
    expect(links.map((l) => l.textContent)).toEqual(['0:12', '10:00', '1:02:03'])
    expect(links.map((l) => l.getAttribute('aria-label'))).toEqual([
      'Open at 0:12',
      'Open at 10:00',
      'Open at 1:02:03',
    ])
    // The title link is still the plain detail route.
    expect(within(item).getByRole('link', { name: 'Title 1' }).getAttribute('href')).toBe(
      '/videos/abc-_Z09xyz',
    )
  })

  it('start_sec: 0 is a link with t=0; a null one next to it has none', async () => {
    stubFetch(() => page([withStarts([0, null])]))
    await mount('/search?q=a')
    const links = screen.getAllByRole('link', { name: /^Open at/ })
    expect(links.map((l) => l.getAttribute('href'))).toEqual(['/videos/abc-_Z09xyz?t=0'])
    expect(document.querySelector('li p')?.textContent).toContain('part 1')
  })

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY, 1e12])(
    'start_sec %s renders the excerpt without a link',
    async (bad) => {
      stubFetch(() => page([withStarts([bad])]))
      await mount('/search?q=a')
      expect(screen.queryAllByRole('link', { name: /^Open at/ })).toHaveLength(0)
      expect(screen.getByText('part 0')).toBeTruthy()
    },
  )

  it('clicking a timestamp link navigates to the detail route with t intact', async () => {
    stubFetch(() => page([withStarts([3723.9])]))
    const router = await mount('/search?q=zebra')
    await user().click(screen.getByRole('link', { name: 'Open at 1:02:03' }))
    await settle()
    expect(router.state.location.pathname).toBe('/videos/abc-_Z09xyz')
    expect(router.state.location.search).toBe('?t=3723')
  })

  it('encodes a video id in the link path', async () => {
    stubFetch(() => page([hit(1, { video_id: 'a/b?c', excerpts: [{ spans: [{ text: 'x', match: true }], start_sec: 5 }] })]))
    await mount('/search?q=a')
    expect(screen.getByRole('link', { name: 'Open at 0:05' }).getAttribute('href')).toBe(
      '/videos/a%2Fb%3Fc?t=5',
    )
  })
})

describe('paging', () => {
  it('shows the range, and Next/Previous behave on the first page', async () => {
    stubFetch(() => page(hits(1, 20), true))
    await mount('/search?q=a')
    expect(live().textContent).toContain('Results 1–20')
    expect(live().textContent).not.toMatch(/ of \d/)
    expect((screen.getByRole('button', { name: 'Previous' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('exactly 20 results with has_more false: one full page, Next disabled', async () => {
    stubFetch(() => page(hits(1, 20), false))
    await mount('/search?q=a')
    expect(screen.getAllByRole('listitem')).toHaveLength(20)
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('a partial last page shows its range and disables Next', async () => {
    stubFetch(() => page(hits(41, 7), false, 40))
    await mount('/search?q=a&page=3')
    expect(live().textContent).toContain('Results 41–47')
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Previous' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('page 51 disables Next even when has_more is true', async () => {
    stubFetch(() => page(hits(1, 20), true, 1000))
    await mount('/search?q=a&page=51')
    expect(live().textContent).toContain('Results 1001–1020')
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('page 50 with has_more enables Next', async () => {
    stubFetch(() => page(hits(1, 20), true, 980))
    await mount('/search?q=a&page=50')
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('Next and Previous push history entries and move focus to the results heading', async () => {
    stubFetch((u) => page(hits(1, 20), true, Number(u.searchParams.get('offset'))))
    const router = await mount('/search?q=a')
    await user().click(screen.getByRole('button', { name: 'Next' }))
    await settle()
    expect(router.state.location.search).toBe('?q=a&page=2')
    expect(router.state.historyAction).toBe('PUSH')
    expect(last().searchParams.get('offset')).toBe('20')
    expect(document.activeElement).toBe(screen.getByRole('heading', { name: 'Results' }))
    await user().click(screen.getByRole('button', { name: 'Previous' }))
    await settle()
    expect(router.state.location.search).toBe('?q=a')
    await router.navigate(-1)
    await settle()
    expect(router.state.location.search).toBe('?q=a&page=2')
  })

  it('keeps the previous results with a loading indicator and disables the controls while loading', async () => {
    const second = deferred()
    stubFetch((u) =>
      u.searchParams.get('offset') === '0' ? page(hits(1, 20), true) : second.promise,
    )
    await mount('/search?q=a')
    await user().click(screen.getByRole('button', { name: 'Next' }))
    await settle()
    expect(screen.getByText('Title 1')).toBeTruthy()
    expect(screen.getByText('Searching…')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Previous' }) as HTMLButtonElement).disabled).toBe(true)
    second.resolve(json(page(hits(21, 20), false, 20)))
    await settle()
    expect(screen.queryByText('Title 1')).toBeNull()
    expect(screen.getByText('Title 21')).toBeTruthy()
    expect(screen.queryByText('Searching…')).toBeNull()
  })

  it('a deep link past the end says so and links to page 1', async () => {
    stubFetch(() => page([], false, 120))
    await mount('/search?q=cats&page=7')
    expect(screen.getByText('No results on this page')).toBeTruthy()
    expect(document.body.textContent).not.toContain('No transcripts match this search')
    expect(screen.getByRole('link', { name: 'Go to page 1' }).getAttribute('href')).toBe('/search?q=cats')
  })
})

describe('states', () => {
  it('shows a loading indicator, not the empty message, on the first search', async () => {
    const d = deferred()
    stubFetch(() => d.promise)
    await mount('/search?q=a')
    expect(screen.getByText('Searching…')).toBeTruthy()
    expect(document.body.textContent).not.toContain('No transcripts match this search')
    d.resolve(json(page([])))
    await settle()
    expect(screen.queryByText('Searching…')).toBeNull()
  })

  it('page 1 with no results shows the hint and does not repeat the query', async () => {
    stubFetch(() => page([]))
    await mount('/search?q=zxqv')
    expect(screen.getByText('No transcripts match this search')).toBeTruthy()
    expect(document.body.textContent).toMatch(/common words/i)
    expect(document.body.textContent).toMatch(/at least one word that is not excluded with -/)
    expect(live().textContent).toContain('No transcripts match this search')
    // Only the input holds the query; no text node echoes it.
    expect(document.body.textContent).not.toContain('zxqv')
    expect(screen.queryByRole('list')).toBeNull()
  })

  it('announces the range in a polite live region', async () => {
    stubFetch(() => page(hits(1, 3)))
    await mount('/search?q=a')
    expect(live().textContent).toContain('Results 1–3')
  })

  it('503: unavailable message with Retry that refetches', async () => {
    let calls = 0
    stubFetch(() => {
      calls += 1
      return calls === 1 ? json({ detail: 'SECRET-SQL-TEXT' }, 503) : page(hits(1, 1))
    })
    await mount('/search?q=a')
    expect(live().textContent).toContain('Search is unavailable right now. Try again shortly.')
    expect(document.body.textContent).not.toContain('SECRET')
    expect(document.body.textContent).not.toContain('/api/search')
    await user().click(screen.getByRole('button', { name: 'Retry' }))
    await settle()
    expect(requests).toHaveLength(2)
    expect(screen.getByText('Title 1')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })

  it('422: could-not-run message with no Retry', async () => {
    stubFetch(() => json({ detail: [{ msg: 'SECRET-DETAIL' }] }, 422))
    await mount('/search?q=a')
    expect(live().textContent).toContain('This search could not be run. Try shorter or different words.')
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect(document.body.textContent).not.toContain('SECRET')
  })

  it('another non-2xx: generic message with Retry', async () => {
    stubFetch(() => json({ detail: 'SECRET' }, 500))
    await mount('/search?q=a')
    expect(live().textContent).toContain('Could not search')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(document.body.textContent).not.toContain('SECRET')
  })

  it('network failure: generic message with Retry, no exception text', async () => {
    requests = []
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('SECRET connect ECONNREFUSED'))))
    await mount('/search?q=a')
    expect(live().textContent).toContain('Could not search')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(document.body.textContent).not.toContain('SECRET')
  })

  it.each([
    ['not json', new Response('<html>SECRET</html>', { status: 200 })],
    ['wrong shape', json({ results: 'nope' })],
    ['bad span', json({ results: [{ ...hit(1), excerpts: [{ spans: [{ text: 1 }], start_sec: null }] }], limit: 20, offset: 0, has_more: false })],
  ])('malformed 200 body (%s): generic error, nothing rendered', async (_n, res) => {
    stubFetch(() => res)
    await mount('/search?q=a')
    expect(live().textContent).toContain('Could not search')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(document.body.textContent).not.toContain('SECRET')
  })

  it('never shows an older response that arrives after a newer submission', async () => {
    const first = deferred()
    const second = deferred()
    stubFetch((u) => (u.searchParams.get('q') === 'old' ? first.promise : second.promise))
    await mount()
    await submit('old')
    await submit('new')
    expect(requests.map((r) => r.searchParams.get('q'))).toEqual(['old', 'new'])
    second.resolve(json(page([hit(2, { title: 'NEW RESULT' })])))
    await settle()
    first.resolve(json(page([hit(1, { title: 'OLD RESULT' })])))
    await settle()
    expect(screen.getByText('NEW RESULT')).toBeTruthy()
    expect(screen.queryByText('OLD RESULT')).toBeNull()
  })

  it('never shows an older page that arrives after a newer one was requested', async () => {
    const p2 = deferred()
    const p3 = deferred()
    stubFetch((u) => {
      const off = u.searchParams.get('offset')
      return off === '0' ? page(hits(1, 20), true) : off === '20' ? p2.promise : p3.promise
    })
    const router = await mount('/search?q=a')
    await user().click(screen.getByRole('button', { name: 'Next' }))
    await settle()
    // Controls are disabled while page 2 loads; a deep link (Forward, a click on
    // another link) can still move to page 3.
    await act(async () => {
      await router.navigate('/search?q=a&page=3')
    })
    await settle()
    p3.resolve(json(page([hit(99, { title: 'PAGE THREE' })], false, 40)))
    await settle()
    p2.resolve(json(page([hit(98, { title: 'PAGE TWO' })], false, 20)))
    await settle()
    expect(screen.queryByText('PAGE TWO')).toBeNull()
    expect(screen.getByText('PAGE THREE')).toBeTruthy()
  })
})

describe('accessibility', () => {
  it('has a labelled input, a polite live region and mark elements for matches', async () => {
    stubFetch(() => page([hit(1, { excerpts: [{ spans: [{ text: 'hit', match: true }], start_sec: null }] })]))
    await mount('/search?q=hit')
    expect(screen.getByLabelText(/search/i)).toBe(input())
    expect(live()).not.toBeNull()
    expect(document.querySelector('mark')?.textContent).toBe('hit')
  })
})
