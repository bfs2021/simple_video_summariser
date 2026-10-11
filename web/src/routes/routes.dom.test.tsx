import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider, type RouteObject } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { routes } from './index'


function mount(entry: string, table: RouteObject[] = routes) {
  const router = createMemoryRouter(table, { initialEntries: [entry] })
  render(
    <QueryClientProvider client={new QueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  return router
}

describe('shell and navigation', () => {
  it('navigates with the nav links without a document request', () => {
    const router = mount('/')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Library')
    fireEvent.click(screen.getByRole('link', { name: 'Search' }))
    expect(router.state.location.pathname).toBe('/search')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Search')
    fireEvent.click(screen.getByRole('link', { name: 'Ops' }))
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Ops')
  })

  it('keeps an API link as a plain anchor the router does not intercept', () => {
    mount('/')
    const a = screen.getByRole('link', { name: /api health/i })
    expect(a.getAttribute('href')).toBe('/api/healthz')
    // The router handles clicks on the React root, below `document`, so a
    // router-handled link is already preventDefault()ed by the time it bubbles
    // here. Record that, then cancel so jsdom does not try to navigate.
    let interceptedByRouter: boolean | undefined
    document.addEventListener('click', (e) => {
      interceptedByRouter = e.defaultPrevented
      e.preventDefault()
    })
    a.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, button: 0 }))
    expect(interceptedByRouter).toBe(false)
  })
})

describe('not found', () => {
  it('renders the path as literal text, sets the title and links home', () => {
    mount('/<img src=x onerror=alert(1)>')
    expect(screen.getByRole('heading', { name: 'Page not found' })).toBeTruthy()
    expect(document.querySelector('img')).toBeNull()
    expect(screen.getByText('/<img src=x onerror=alert(1)>')).toBeTruthy()
    expect(document.title).toBe('Page not found')
    const home = screen.getAllByRole('link').filter((l) => l.textContent === 'Back to Library')
    expect(home[0]?.getAttribute('href')).toBe('/')
  })
})

describe('error boundary', () => {
  it('shows a generic message without the error text', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    function Boom(): never {
      throw new Error('secret-internal-detail')
    }
    const root = routes[0] as { children: RouteObject[] } & RouteObject
    const table = [
      { ...root, children: [...root.children, { path: 'boom', element: <Boom /> }] },
    ] as RouteObject[]
    mount('/boom', table)
    expect(screen.getByText('Something went wrong')).toBeTruthy()
    expect(document.body.textContent).not.toContain('secret-internal-detail')
    expect(screen.getByRole('link', { name: 'Back to Library' }).getAttribute('href')).toBe('/')
  })
})

// Query strings must survive routing (#122). This block mounts the SHARED
// `routes` table, the only thing that proves the registered route lets the
// query reach the view; the view tests (Library.dom, VideoDetail.dom,
// Transcript.dom) build their own local router tables and only prove each view
// reads its param. Issues #49, #50 and #51 each deleted one query-string row
// from the shared tests, believing the view tests covered it, and the coverage
// silently vanished (a useEffect stripping the query left every test green).
//
// Do NOT delete a row here when a view replaces its placeholder. REWRITE the
// row: keep the same URL and assert the search string is unchanged after the
// view has rendered and its effects have run. #52 (Search), #53 (Compare) and
// #54 (Ops) must do the same for their rows.
describe('query strings survive the shared route table', () => {
  let requests: URL[]

  beforeEach(() => {
    vi.useFakeTimers()
    requests = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        requests.push(new URL(String(input), 'http://localhost'))
        return new Response('{}', { status: 500, headers: { 'content-type': 'application/json' } })
      }),
    )
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  // Let mount effects, the first fetch and any redirect they cause run. Plain
  // act + advanceTimersByTimeAsync: waitFor is not needed to wait for "nothing
  // changed", and it does not detect Vitest's fake timers anyway.
  async function settle() {
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50)
      })
    }
  }

  it.each([
    ['/?page=2&status=failed', '?page=2&status=failed'],
    ['/videos/-wNyEUrxzFU?t=3723', '?t=3723'],
    ['/videos/-wNyEUrxzFU/transcript?page=3', '?page=3'],
    ['/ops?state=dead&before=100', '?state=dead&before=100'],
    ['/ops?state=pending&kind=ingest&error_class=none', '?state=pending&kind=ingest&error_class=none'],
  ])('%s keeps its search string after effects have run', async (entry, search) => {
    const router = mount(entry)
    await settle()
    expect(router.state.location.search).toBe(search)
    expect(router.state.location.pathname).toBe(entry.split('?')[0])
  })

  it('the transcript route delivers page 3 to the request as offset=400', async () => {
    mount('/videos/-wNyEUrxzFU/transcript?page=3')
    await settle()
    const transcript = requests.filter((u) => u.pathname.endsWith('/transcript'))
    expect(transcript.length).toBeGreaterThan(0)
    expect(transcript[0]?.searchParams.get('offset')).toBe('400')
  })
})

describe('trailing slash on a video id', () => {
  // The SSR trailing-slash table in routes.test.tsx compares heading(), which is
  // '' for VideoDetail on both sides (it renders no <h1> while loading), so it
  // only proves "not Page not found". Assert the real marker on both forms.
  it.each(['/videos/-wNyEUrxzFU', '/videos/-wNyEUrxzFU/'])('%s renders VideoDetail', (entry) => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})))
    try {
      mount(entry)
      expect(screen.getByText(/Loading video/)).toBeTruthy()
      expect(screen.queryByText('Page not found')).toBeNull()
    } finally {
      vi.unstubAllGlobals()
    }
  })
})

// #110: a search hit links to the moment it was said. This goes through the
// SHARED route table (not a local router), so it proves that the registered
// routes deliver the link's `?t=` to the detail route and that nothing strips it
// once the destination view has rendered and run its effects. The seek itself
// (player ready, once) belongs to the `?t=` deep-link work (#129).
describe('search hit timestamp links through the shared route table', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = new URL(String(input), 'http://localhost')
        if (url.pathname.endsWith('/search')) {
          return new Response(
            JSON.stringify({
              results: [
                {
                  video_id: '-wNyEUrxzFU',
                  title: 'The Zebra',
                  channel_id: null,
                  channel_title: null,
                  published_at: null,
                  duration_sec: null,
                  unavailable: null,
                  transcript_source: 'whisper',
                  matched_in: ['transcript'],
                  excerpts: [
                    { spans: [{ text: 'a zebra', match: true }], start_sec: 3723.9 },
                    { spans: [{ text: 'no time', match: false }], start_sec: null },
                  ],
                },
              ],
              limit: 20,
              offset: 0,
              has_more: false,
            }),
            { status: 200, headers: { 'content-type': 'application/json' } },
          )
        }
        return new Response('{}', { status: 500, headers: { 'content-type': 'application/json' } })
      }),
    )
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  async function settle() {
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50)
      })
    }
  }

  it('clicking a hit time opens the detail route with ?t= intact after effects ran', async () => {
    const router = mount('/search?q=zebra')
    await settle()
    const link = screen.getByRole('link', { name: 'Open at 1:02:03' })
    expect(link.getAttribute('href')).toBe('/videos/-wNyEUrxzFU?t=3723')
    expect(screen.getAllByRole('link', { name: /^Open at/ })).toHaveLength(1)

    fireEvent.click(link)
    await settle()

    expect(router.state.location.pathname).toBe('/videos/-wNyEUrxzFU')
    expect(router.state.location.search).toBe('?t=3723')
    // The destination is the real VideoDetail (not Search, not Page not found).
    expect(screen.queryByRole('heading', { name: 'Search' })).toBeNull()
    expect(screen.queryByText('Page not found')).toBeNull()
  })
})
