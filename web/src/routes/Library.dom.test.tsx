import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { components } from '../api'
import { Library } from './Library'

type VideoItem = components['schemas']['VideoItem']
type VideoList = components['schemas']['VideoList']

const UC = 'UC' + 'a'.repeat(22)
const UC2 = 'UC' + 'b'.repeat(22)

function vid(n: number, over: Partial<VideoItem> = {}): VideoItem {
  return {
    video_id: `v${String(n).padStart(10, '0')}`,
    title: `Title ${n}`,
    channel_id: UC,
    channel_title: 'Chan A',
    published_at: '2026-09-01T12:00:00Z',
    duration_sec: 60,
    latest_analysis_at: null,
    origin: 'manual',
    unavailable: null,
    status: 'done',
    active_job: null,
    last_failure: null,
    ...over,
  }
}

function list(total: number, items: VideoItem[], offset = 0): VideoList {
  return { items, total, offset, limit: 50 }
}

function range(from: number, count: number): VideoItem[] {
  return Array.from({ length: count }, (_, i) => vid(from + i))
}

type Responder = (q: URLSearchParams) => Response | Promise<Response> | VideoList
let requests: URL[]

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

function stubFetch(respond: Responder) {
  requests = []
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost')
      requests.push(url)
      const r = respond(url.searchParams)
      return Promise.resolve(r instanceof Response || r instanceof Promise ? r : json(r))
    }),
  )
}

/** Answers like #42 would for `total` rows, honouring offset/limit. */
function serverWith(total: number): Responder {
  return (q) => {
    const offset = Number(q.get('offset'))
    const count = Math.max(0, Math.min(50, total - offset))
    return list(total, range(offset + 1, count), offset)
  }
}

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

async function mount(entry = '/') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [
      { path: '/', element: <Library /> },
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

beforeEach(() => {
  vi.useFakeTimers()
})

const search = (r: { state: { location: { search: string } } }) => r.state.location.search
const last = () => requests[requests.length - 1] as URL

describe('request construction', () => {
  it('fetches same-origin /api/videos with limit and offset only when nothing is set', async () => {
    stubFetch(serverWith(3))
    await mount()
    expect(requests).toHaveLength(1)
    expect(last().pathname).toBe('/api/videos')
    expect(last().search).toBe('?limit=50&offset=0')
  })

  it('restores every filter from the URL and sends the translated query', async () => {
    stubFetch(serverWith(500))
    await mount(`/?channel=${UC}&from=2026-09-01&to=2026-09-01&status=failed&page=3`)
    expect(Object.fromEntries(last().searchParams)).toEqual({
      channel: UC,
      status: 'failed',
      published_after: '2026-09-01',
      published_before: '2026-09-02',
      limit: '50',
      offset: '100',
    })
    expect((screen.getByLabelText('From') as HTMLInputElement).value).toBe('2026-09-01')
    expect((screen.getByLabelText('To') as HTMLInputElement).value).toBe('2026-09-01')
    expect((screen.getByLabelText('Status') as HTMLSelectElement).value).toBe('failed')
  })

  it('To on the year boundary sends the first day of the next year', async () => {
    stubFetch(serverWith(1))
    await mount('/?to=2026-12-31')
    expect(last().searchParams.get('published_before')).toBe('2027-01-01')
  })

  it('To on 29 Feb of a leap year sends 1 March', async () => {
    stubFetch(serverWith(1))
    await mount('/?to=2024-02-29')
    expect(last().searchParams.get('published_before')).toBe('2024-03-01')
  })

  it('caches each filter combination separately', async () => {
    stubFetch((q) =>
      list(1, [vid(1, { title: q.get('status') === 'idle' ? 'Idle one' : 'Any one' })]),
    )
    const router = await mount()
    await act(async () => {
      fireEvent.change(screen.getByLabelText('Status'), { target: { value: 'idle' } })
    })
    await settle()
    expect(screen.getByRole('link', { name: 'Idle one' })).toBeTruthy()
    await act(async () => {
      await router.navigate(-1)
    })
    // Rendered from the cache with no time advanced; a refetch follows in the background.
    expect(screen.getByRole('link', { name: 'Any one' })).toBeTruthy()
    await settle()
    expect(requests.map((r) => r.searchParams.get('status'))).toEqual([null, 'idle', null])
  })
})

describe('sorting', () => {
  it('marks only Published descending and gives every header an action name', async () => {
    stubFetch(serverWith(2))
    await mount()
    const headers = screen.getAllByRole('columnheader')
    expect(headers.map((header) => [header.textContent, header.getAttribute('aria-sort')])).toEqual([
      ['Title', null],
      ['Channel', null],
      ['Published descending', 'descending'],
      ['Status', null],
    ])
    expect(screen.getByRole('button', { name: 'Sort by Title ascending' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Sort by Channel ascending' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Sort by Published ascending' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Sort by Status ascending' })).toBeTruthy()
  })

  it.each([
    ['Title', 'title', 'asc'],
    ['Channel', 'channel', 'asc'],
    ['Status', 'status', 'asc'],
  ])('selects %s with its default direction, preserving filters and resetting page', async (label, sort, order) => {
    stubFetch(serverWith(120))
    const router = await mount('/?status=idle&page=2')
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: `Sort by ${label} ascending` }))
    })
    await settle()
    expect(search(router)).toBe(`?status=idle&sort=${sort}&order=${order}`)
    expect(last().searchParams.get('sort')).toBe(sort)
    expect(last().searchParams.get('order')).toBe(order)
    expect(last().searchParams.get('offset')).toBe('0')
  })

  it('toggles an active header and Back restores the prior sort', async () => {
    stubFetch(serverWith(2))
    const router = await mount('/?sort=title&order=asc')
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Sort by Title descending' }))
    })
    await settle()
    expect(search(router)).toBe('?sort=title&order=desc')
    expect(screen.getByRole('columnheader', { name: 'Title descending' }).getAttribute('aria-sort')).toBe(
      'descending',
    )
    await act(async () => {
      await router.navigate(-1)
    })
    await settle()
    expect(search(router)).toBe('?sort=title&order=asc')
    expect(screen.getByRole('columnheader', { name: 'Title ascending' }).getAttribute('aria-sort')).toBe(
      'ascending',
    )
  })

  it('clear all filters preserves sorting and sorting alone does not show it', async () => {
    stubFetch(serverWith(2))
    const router = await mount('/?status=done&sort=channel&order=desc')
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Clear all filters' }))
    })
    await settle()
    expect(search(router)).toBe('?sort=channel&order=desc')
    expect(screen.queryByRole('button', { name: 'Clear all filters' })).toBeNull()
  })
})

describe('rows', () => {
  it('shows title link, channel, UTC date and status', async () => {
    stubFetch(() => list(1, [vid(1, { published_at: '2026-09-01T02:00:00Z' })]))
    await mount()
    const row = screen.getByRole('row', { name: /Title 1/ })
    const link = within(row).getByRole('link', { name: 'Title 1' })
    expect(link.getAttribute('href')).toBe('/videos/v0000000001')
    expect(within(row).getByRole('button', { name: 'Chan A' })).toBeTruthy()
    // 02:00Z is still 2026-08-31 in America/Los_Angeles; UTC is expected.
    expect(within(row).getByText('2026-09-01')).toBeTruthy()
    expect(within(row).getByText('Analysed')).toBeTruthy()
    expect(screen.getAllByRole('columnheader').map((h) => h.textContent)).toEqual([
      'Title',
      'Channel',
      'Published descending',
      'Status',
    ])
  })

  it('falls back for a stub row: video_id, channel id, unknown date', async () => {
    stubFetch(() =>
      list(1, [vid(1, { title: null, channel_title: null, published_at: null, status: 'idle' })]),
    )
    await mount()
    const row = screen.getAllByRole('row')[1] as HTMLElement
    expect(within(row).getByRole('link', { name: 'v0000000001' }).getAttribute('href')).toBe(
      '/videos/v0000000001',
    )
    expect(within(row).getByRole('button', { name: UC })).toBeTruthy()
    expect(within(row).getByText('Date unknown')).toBeTruthy()
    expect(within(row).getByText('Not processed')).toBeTruthy()
  })

  it('says "Unknown channel" without a control when both channel fields are null', async () => {
    stubFetch(() => list(1, [vid(1, { channel_id: null, channel_title: null })]))
    await mount()
    expect(screen.getByText('Unknown channel')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Unknown channel' })).toBeNull()
  })

  it('shows a title without channel id as plain text', async () => {
    stubFetch(() => list(1, [vid(1, { channel_id: null, channel_title: 'Orphan' })]))
    await mount()
    expect(screen.getByText('Orphan')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Orphan' })).toBeNull()
  })

  it('renders stored strings as literal text and creates no element', async () => {
    const evil = '<img src=x onerror=alert(1)>'
    stubFetch(() =>
      list(1, [
        vid(1, {
          title: evil,
          channel_title: evil + 'c',
          status: 'failed',
          last_failure: { job_id: 1, kind: 'ingest', error_class: evil + 'e', finished_at: null },
        }),
      ]),
    )
    await mount()
    expect(document.querySelector('img')).toBeNull()
    expect(screen.getByRole('link', { name: evil })).toBeTruthy()
    expect(screen.getByText(evil + 'c')).toBeTruthy()
    expect(screen.getByText(evil + 'e')).toBeTruthy()
  })

  it('shows the failure class as secondary text and never last_error', async () => {
    stubFetch(() =>
      list(1, [
        {
          ...vid(1, {
            status: 'failed',
            last_failure: { job_id: 1, kind: 'ingest', error_class: 'TOOL_FAILURE', finished_at: null },
          }),
          last_error: 'SECRET-TRACE',
        } as VideoItem,
      ]),
    )
    await mount()
    expect(screen.getByText('Failed at download')).toBeTruthy()
    expect(screen.getByText('TOOL_FAILURE')).toBeTruthy()
    expect(document.body.textContent).not.toContain('SECRET-TRACE')
  })

  it('renders a processing row, an unknown status and an unknown job kind without crashing', async () => {
    stubFetch(() =>
      list(3, [
        vid(1, { status: 'processing', active_job: { id: 1, kind: 'ingest', state: 'running' } }),
        vid(2, { status: 'weird' as VideoItem['status'] }),
        vid(3, { status: 'processing', active_job: { id: 2, kind: 'novel', state: 'pending' } }),
      ]),
    )
    await mount()
    expect(screen.getByText('Download in progress')).toBeTruthy()
    expect(screen.getByText('Unknown status')).toBeTruthy()
    expect(within(screen.getByRole('table')).getByText('Processing')).toBeTruthy()
    expect(screen.getAllByRole('row')).toHaveLength(4)
  })

  it('keeps the order the server returned', async () => {
    stubFetch(() => list(3, [vid(3), vid(1), vid(2)]))
    await mount()
    const titles = screen.getAllByRole('link', { name: /^Title/ }).map((a) => a.textContent)
    expect(titles).toEqual(['Title 3', 'Title 1', 'Title 2'])
  })

  it('navigates to the detail route from the title link', async () => {
    stubFetch(() => list(1, [vid(1)]))
    const router = await mount()
    await act(async () => {
      fireEvent.click(screen.getByRole('link', { name: 'Title 1' }))
    })
    expect(router.state.location.pathname).toBe('/videos/v0000000001')
  })
})

describe('filters', () => {
  it('clicking a channel sets ?channel and resets to page 1', async () => {
    stubFetch(serverWith(120))
    const router = await mount('/?page=2')
    await act(async () => {
      fireEvent.click(screen.getAllByRole('button', { name: 'Chan A' })[0] as HTMLElement)
    })
    await settle()
    expect(search(router)).toBe(`?channel=${UC}`)
    expect(last().searchParams.get('channel')).toBe(UC)
    expect(last().searchParams.get('offset')).toBe('0')
  })

  it('shows the active channel by title when a row supplies it, else by id; Clear removes it', async () => {
    stubFetch(() => list(1, [vid(1)]))
    const router = await mount(`/?channel=${UC}`)
    const filter = screen.getByRole('group', { name: 'Active filters' })
    expect(within(filter).getByText('Chan A')).toBeTruthy()
    await act(async () => {
      fireEvent.click(within(filter).getByRole('button', { name: 'Clear' }))
    })
    await settle()
    expect(search(router)).toBe('')
    expect(last().searchParams.has('channel')).toBe(false)
  })

  it('falls back to the raw channel id when no loaded row has a title', async () => {
    stubFetch(() => list(1, [vid(1, { channel_id: UC2, channel_title: 'Other' })]))
    await mount(`/?channel=${UC}`)
    const filter = screen.getByRole('group', { name: 'Active filters' })
    expect(within(filter).getByText(UC)).toBeTruthy()
  })

  it('the status select offers Any plus the five human-labelled values', async () => {
    stubFetch(serverWith(1))
    await mount()
    const select = screen.getByLabelText('Status') as HTMLSelectElement
    expect([...select.options].map((o) => [o.value, o.textContent])).toEqual([
      ['', 'Any'],
      ['done', 'Analysed'],
      ['processing', 'Processing'],
      ['failed', 'Failed'],
      ['unavailable', 'Unavailable on YouTube'],
      ['idle', 'Not processed'],
    ])
  })

  it('changing a filter resets to page 1 and pushes history so Back works', async () => {
    stubFetch(serverWith(500))
    const router = await mount('/?page=4')
    await act(async () => {
      fireEvent.change(screen.getByLabelText('Status'), { target: { value: 'done' } })
    })
    await settle()
    expect(search(router)).toBe('?status=done')
    expect(last().searchParams.get('offset')).toBe('0')
    await act(async () => {
      await router.navigate(-1)
    })
    await settle()
    expect(search(router)).toBe('?page=4')
    expect((screen.getByLabelText('Status') as HTMLSelectElement).value).toBe('')
  })

  it('sets From and To from the date inputs and combines them with AND', async () => {
    stubFetch(serverWith(1))
    const router = await mount(`/?status=idle`)
    await act(async () => {
      fireEvent.change(screen.getByLabelText('From'), { target: { value: '2026-09-01' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByLabelText('To'), { target: { value: '2026-09-01' } })
    })
    await settle()
    expect(search(router)).toBe('?from=2026-09-01&to=2026-09-01&status=idle')
    expect(last().search).toBe(
      '?status=idle&published_after=2026-09-01&published_before=2026-09-02&limit=50&offset=0',
    )
  })

  it('clearing a date input removes the parameter', async () => {
    stubFetch(serverWith(1))
    const router = await mount('/?from=2026-09-01')
    await act(async () => {
      fireEvent.change(screen.getByLabelText('From'), { target: { value: '' } })
    })
    await settle()
    expect(search(router)).toBe('')
  })

  it('shows "Clear all filters" only when a filter is set, and it clears them all', async () => {
    stubFetch(serverWith(5))
    const router = await mount()
    expect(screen.queryByRole('button', { name: 'Clear all filters' })).toBeNull()
    cleanup()
    const r2 = await mount(`/?channel=${UC}&from=2026-01-01&to=2026-02-02&status=done&page=2`)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Clear all filters' }))
    })
    await settle()
    expect(search(r2)).toBe('')
    expect(search(router)).toBe('')
  })

  it('does not show "Clear all filters" for the page number alone', async () => {
    stubFetch(serverWith(500))
    await mount('/?page=2')
    expect(screen.queryByRole('button', { name: 'Clear all filters' })).toBeNull()
  })

  it('From after To: inline message, no new request, last valid results stay visible', async () => {
    stubFetch(serverWith(3))
    const router = await mount('/?to=2026-09-01')
    expect(requests).toHaveLength(1)
    await act(async () => {
      fireEvent.change(screen.getByLabelText('From'), { target: { value: '2026-09-02' } })
    })
    await settle()
    expect(screen.getByText("'From' is after 'To'")).toBeTruthy()
    expect(requests).toHaveLength(1)
    expect(screen.getByRole('link', { name: 'Title 1' })).toBeTruthy()
    expect(search(router)).toBe('?from=2026-09-02&to=2026-09-01')
  })

  it('From after To on a deep link sends no request at all', async () => {
    stubFetch(serverWith(3))
    await mount('/?from=2026-09-02&to=2026-09-01')
    expect(requests).toHaveLength(0)
    expect(screen.getByText("'From' is after 'To'")).toBeTruthy()
    expect(screen.queryByText('Nothing has been processed yet')).toBeNull()
  })
})

describe('untrusted URL parameters', () => {
  it('drops each invalid parameter: never sent, never shown, URL rewritten with replace', async () => {
    stubFetch(serverWith(2))
    const m = 'MARKER'
    const router = await mount(
      `/?channel=${m}&from=${m}&to=${m}&status=${m}&sort=${m}&order=${m}&page=${m}&q=${m}&before=${m}&offset=${m}&limit=${m}`,
    )
    expect(requests).toHaveLength(1)
    expect(last().search).toBe('?limit=50&offset=0')
    for (const r of requests) expect(r.href).not.toContain(m)
    expect(document.body.innerHTML).not.toContain(m)
    expect(document.body.textContent).not.toContain(m)
    expect((screen.getByLabelText('From') as HTMLInputElement).value).toBe('')
    // Invalid known params are gone from the URL, replaced rather than pushed.
    expect(search(router)).not.toMatch(/channel|from|to=|status|sort|order|page/)
    expect(router.state.historyAction).toBe('REPLACE')
  })

  it.each(['0', '-1', '1.5', 'abc', '1e3', '99999999999999999999', '20002'])(
    'page=%s falls back to page 1',
    async (p) => {
      stubFetch(serverWith(500))
      const router = await mount(`/?page=${p}`)
      expect(last().searchParams.get('offset')).toBe('0')
      expect(search(router)).toBe('')
    },
  )

  it('accepts the largest page within the offset cap', async () => {
    stubFetch(() => list(1_000_100, [vid(1)], 1_000_000))
    await mount('/?page=20001')
    expect(last().searchParams.get('offset')).toBe('1000000')
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(true)
  })

  it('keeps valid parameters while dropping an invalid one', async () => {
    stubFetch(serverWith(500))
    const router = await mount(`/?status=done&channel=nope&page=2`)
    expect(search(router)).toBe('?status=done&page=2')
    expect(last().searchParams.get('channel')).toBeNull()
  })

  it('a screen reader sees no marker even in the filter summary', async () => {
    stubFetch(serverWith(1))
    await mount('/?channel=%3Cimg%20src%3Dx%3E')
    expect(document.querySelector('img')).toBeNull()
    expect(screen.queryByRole('group', { name: 'Active filters' })).toBeNull()
  })
})

describe('pagination', () => {
  const status = () => screen.getByRole('status', { name: 'Results' })

  it('page 1 of 234: range, Previous disabled, Next enabled', async () => {
    stubFetch(serverWith(234))
    await mount()
    expect(status().textContent).toBe('1–50 of 234')
    expect(screen.getByRole('button', { name: 'Previous' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(false)
    expect(status().getAttribute('aria-live')).toBe('polite')
  })

  it('page 2 shows 51–100 of 234 and Next/Previous push the page into the URL', async () => {
    stubFetch(serverWith(234))
    const router = await mount()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    })
    await settle()
    expect(search(router)).toBe('?page=2')
    expect(status().textContent).toBe('51–100 of 234')
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Previous' }))
    })
    await settle()
    expect(search(router)).toBe('')
    expect(status().textContent).toBe('1–50 of 234')
  })

  it('last partial page (234, page 5): 201–234 and Next disabled', async () => {
    stubFetch(serverWith(234))
    await mount('/?page=5')
    expect(status().textContent).toBe('201–234 of 234')
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Previous' }).hasAttribute('disabled')).toBe(false)
  })

  it('exactly one full page (total=50): Next disabled', async () => {
    stubFetch(serverWith(50))
    await mount()
    expect(status().textContent).toBe('1–50 of 50')
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(true)
  })

  it('total=51: page 1 has Next, page 2 has one row', async () => {
    stubFetch(serverWith(51))
    await mount()
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(false)
    cleanup()
    await mount('/?page=2')
    expect(screen.getAllByRole('row')).toHaveLength(2)
    expect(status().textContent).toBe('51–51 of 51')
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(true)
  })

  it('a deep link past the end explains itself and links to the last page', async () => {
    stubFetch(serverWith(234))
    const router = await mount('/?status=done&page=99')
    expect(screen.getByText('No videos on this page')).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
    const link = screen.getByRole('link', { name: /last page/i })
    await act(async () => {
      fireEvent.click(link)
    })
    await settle()
    expect(search(router)).toBe('?status=done&page=5')
    expect(screen.getByRole('status', { name: 'Results' }).textContent).toBe('201–234 of 234')
  })

  it('keeps the current rows with a loading indicator and disabled controls while the next page loads', async () => {
    let release: (r: Response) => void = () => {}
    stubFetch((q) => {
      if (q.get('offset') === '0') return list(120, range(1, 50))
      return new Promise<Response>((res) => {
        release = res
      })
    })
    await mount()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    })
    await settle()
    expect(screen.getByRole('link', { name: 'Title 1' })).toBeTruthy()
    expect(screen.getByText('Loading…')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Previous' }).hasAttribute('disabled')).toBe(true)
    await act(async () => {
      release(json(list(120, range(51, 50), 50)))
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByRole('link', { name: 'Title 51' })).toBeTruthy()
    expect(screen.queryByText('Loading…')).toBeNull()
    expect(screen.getByRole('button', { name: 'Next' }).hasAttribute('disabled')).toBe(false)
  })
})

describe('states', () => {
  it('first visit shows a loading indicator, not the empty message', async () => {
    stubFetch(() => new Promise<Response>(() => {}))
    await mount()
    expect(screen.getByText('Loading…')).toBeTruthy()
    expect(screen.queryByText('Nothing has been processed yet')).toBeNull()
    expect(screen.queryByText('No videos match these filters')).toBeNull()
    expect(screen.queryByRole('table')).toBeNull()
  })

  it('empty database', async () => {
    stubFetch(() => list(0, []))
    await mount()
    expect(screen.getByText('Nothing has been processed yet')).toBeTruthy()
    expect(screen.queryByText('No videos match these filters')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Clear all filters' })).toBeNull()
  })

  it('no matches with filters set offers Clear all filters', async () => {
    stubFetch(() => list(0, []))
    const router = await mount('/?status=idle')
    expect(screen.getByText('No videos match these filters')).toBeTruthy()
    expect(screen.queryByText('Nothing has been processed yet')).toBeNull()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Clear all filters' }))
    })
    await settle()
    expect(search(router)).toBe('')
  })

  it('503 shows the database message and Retry refetches', async () => {
    let healthy = false
    stubFetch(() => (healthy ? list(1, [vid(1)]) : json({ detail: 'SECRET db down' }, 503)))
    await mount()
    expect(screen.getByText('The database is unavailable. Try again shortly.')).toBeTruthy()
    expect(document.body.textContent).not.toContain('SECRET')
    expect(document.body.textContent).not.toContain('/api/videos')
    healthy = true
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    })
    await settle()
    expect(screen.getByRole('link', { name: 'Title 1' })).toBeTruthy()
    expect(screen.queryByText('The database is unavailable. Try again shortly.')).toBeNull()
  })

  it.each([500, 404, 422])('HTTP %i shows the generic message without the body', async (code) => {
    stubFetch(() => json({ detail: 'SECRET detail' }, code))
    await mount()
    expect(screen.getByText('Could not load videos')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(document.body.textContent).not.toContain('SECRET')
  })

  it('a network failure shows the generic message without exception text', async () => {
    stubFetch(() => Promise.reject(new TypeError('SECRET failed to fetch http://x/api/videos')))
    await mount()
    expect(screen.getByText('Could not load videos')).toBeTruthy()
    expect(document.body.textContent).not.toContain('SECRET')
    expect(document.body.textContent).not.toContain('http://x')
  })

  it('a malformed 200 body is the generic error, not a crash', async () => {
    stubFetch(() => json({ nope: true }))
    await mount()
    expect(screen.getByText('Could not load videos')).toBeTruthy()
  })
})

describe('polling', () => {
  const active = { id: 1, kind: 'ingest', state: 'running' }

  it('refetches every 15 s while a row has an active job and stops when none does', async () => {
    let calls = 0
    stubFetch(() => {
      calls += 1
      // First two responses carry an active job, the third does not.
      return list(1, [vid(1, calls < 3 ? { status: 'processing', active_job: active } : {})])
    })
    await mount()
    expect(calls).toBe(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(14_999)
    })
    expect(calls).toBe(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(calls).toBe(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(calls).toBe(3)
    expect(screen.getByText('Analysed')).toBeTruthy()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000)
    })
    expect(calls).toBe(3)
  })

  it('does not poll when no row has an active job', async () => {
    stubFetch(() => list(1, [vid(1)]))
    await mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(requests).toHaveLength(1)
  })

  it('a re-analysing done row keeps the poll alive', async () => {
    stubFetch(() => list(1, [vid(1, { active_job: { id: 3, kind: 'analyze', state: 'pending' } })]))
    await mount()
    expect(screen.getByText('Analysed · re-analysing')).toBeTruthy()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(requests).toHaveLength(2)
  })

  it('stops polling on unmount', async () => {
    stubFetch(() => list(1, [vid(1, { status: 'processing', active_job: active })]))
    await mount()
    expect(requests).toHaveLength(1)
    cleanup()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(requests).toHaveLength(1)
  })
})

describe('accessibility', () => {
  it('labels every filter control and uses a table with headers', async () => {
    stubFetch(serverWith(2))
    await mount()
    expect(screen.getByLabelText('From').tagName).toBe('INPUT')
    expect(screen.getByLabelText('To').tagName).toBe('INPUT')
    expect(screen.getByLabelText('Status').tagName).toBe('SELECT')
    expect(screen.getByRole('table')).toBeTruthy()
    expect(screen.getAllByRole('columnheader')).toHaveLength(4)
  })

  it('uses no dangerouslySetInnerHTML in the view sources', async () => {
    const { readFileSync } = await import('node:fs')
    for (const f of ['Library.tsx', '../hooks/useVideos.ts', '../lib/library.ts']) {
      expect(readFileSync(new URL(f, import.meta.url), 'utf8')).not.toContain('dangerouslySetInnerHTML')
    }
  })
})
