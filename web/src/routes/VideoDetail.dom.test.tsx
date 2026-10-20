import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen, within } from '@testing-library/react'
import { setupUser } from '../test/user'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { components } from '../api'
import { FakePlayerProvider } from '../components/FakePlayerProvider'
import { VideoDetail } from './VideoDetail'

const user = setupUser

type VideoDetailOut = components['schemas']['VideoDetail']
type AnalysisOut = components['schemas']['AnalysisOut']

const ID = 'dQw4w9WgXcQ'
const DASH_ID = '-wNyEUrxzFU'

function analysis(over: Partial<AnalysisOut> = {}): AnalysisOut {
  return {
    id: 1,
    transcript_id: 1,
    transcript_source: 'captions',
    chunk_strategy: 'fixed',
    model: 'test-model-1',
    prompt_version: 'v7',
    created_at: '2026-10-01T08:09:10Z',
    input_tokens: 111111,
    output_tokens: 222222,
    cost_usd: 3.5,
    duration_ms: 987654,
    tldr: 'A short summary.',
    speaker_roster: { speakers: [{ name: 'Ada', role: 'host' }] },
    topics: [{ seq: 0, title: 'Intro', summary: 'Opening remarks', start_sec: 5 }],
    claims: [{ text: 'Water is wet', speaker: 'Ada', confidence: 'high', start_sec: 65, source_chunk_seq: 0 }],
    quotes: [{ text: 'Hello there', speaker: 'Ada', start_sec: 10, source_chunk_seq: 0 }],
    ...over,
  }
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
    origin: 'manual',
    unavailable: null,
    status: 'done',
    active_job: null,
    last_failure: null,
    transcript: null,
    analysis: analysis(),
    ...over,
  }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

let requests: URL[]
let seeks: number[]
type Responder = (url: URL, n: number) => Response | Promise<Response>

function stubFetch(respond: Responder) {
  requests = []
  const fn = vi.fn((input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    // The "Compare analyses" count request (#53) is covered in
    // VideoDetail.compareLink.dom.test.tsx; here it only has to fail quietly.
    if (url.pathname.endsWith('/analyses')) return Promise.resolve(json({ detail: 'not under test' }, 404))
    requests.push(url)
    return Promise.resolve(respond(url, requests.length))
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

async function mount(entry = `/videos/${ID}`) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [
      { path: '/', element: <p>Library page</p> },
      {
        path: '/videos/:videoId',
        element: (
          <FakePlayerProvider seeks={seeks}>
            <VideoDetail />
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

async function mountWith(v: VideoDetailOut) {
  stubFetch(() => json(v))
  return mount(`/videos/${v.video_id}`)
}

beforeEach(() => {
  vi.useFakeTimers()
  seeks = []
})

const section = (name: string) => screen.getByRole('region', { name })
const body = () => document.body.textContent ?? ''

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
    const fetchFn = stubFetch(() => json(video()))
    await mount(`/videos/${raw}`)
    expect(screen.getByText('Video not found')).toBeTruthy()
    expect(fetchFn).not.toHaveBeenCalled()
    expect(document.querySelector('iframe')).toBeNull()
    expect(screen.getByRole('link', { name: /library/i }).getAttribute('href')).toBe('/')
  })

  it('accepts an id with a leading dash and requests exactly that id', async () => {
    stubFetch(() => json(video({ video_id: DASH_ID })))
    await mount(`/videos/${DASH_ID}`)
    expect(requests.map((u) => u.pathname)).toEqual([`/api/videos/${DASH_ID}`])
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('My Video')
  })

  it('is case-sensitive: the id is sent as written', async () => {
    stubFetch(() => json(video()))
    await mount('/videos/dqw4w9wgxcq')
    expect(requests[0]?.pathname).toBe('/api/videos/dqw4w9wgxcq')
  })

  it('never shows the first video under the second video URL', async () => {
    const other = 'abcdefghijk'
    stubFetch((url) =>
      url.pathname.endsWith(other)
        ? new Promise<Response>(() => {}) // never settles
        : json(video({ title: 'First Video' })),
    )
    const router = await mount(`/videos/${ID}`)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('First Video')
    await act(async () => {
      await router.navigate(`/videos/${other}`)
    })
    await settle()
    expect(body()).not.toContain('First Video')
    expect(screen.getByText(/loading/i)).toBeTruthy()
    expect(requests.at(-1)?.pathname).toBe(`/api/videos/${other}`)
  })
})

describe('page states', () => {
  it('shows loading text, no sections and no player while pending', async () => {
    stubFetch(() => new Promise<Response>(() => {}))
    await mount()
    expect(screen.getByRole('status').textContent).toMatch(/loading/i)
    expect(screen.queryAllByRole('heading', { level: 2 })).toHaveLength(0)
    expect(document.querySelector('iframe')).toBeNull()
  })

  it('shows Video not found with a Library link and no player on a 404', async () => {
    stubFetch(() => json({ detail: 'nope' }, 404))
    await mount()
    expect(screen.getByText('Video not found')).toBeTruthy()
    expect(screen.getByRole('link', { name: /library/i }).getAttribute('href')).toBe('/')
    expect(document.querySelector('iframe')).toBeNull()
  })

  it.each([
    ['500', () => json({ detail: 'Traceback SECRET-STACK' }, 500)],
    ['503', () => json({ detail: 'SECRET-STACK' }, 503)],
    ['a network error', () => Promise.reject(new TypeError('SECRET-STACK failed to fetch'))],
    ['a non-JSON 200', () => new Response('<html>SECRET-STACK</html>', { status: 200 })],
    ['a JSON 200 of the wrong shape', () => json({ items: [] })],
  ])('shows a fixed message and Retry for %s, never the raw error', async (_n, respond) => {
    stubFetch(respond as Responder)
    await mount()
    expect(screen.getByText('Could not load this video')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(body()).not.toContain('SECRET-STACK')
    expect(document.querySelector('iframe')).toBeNull()
  })

  it('Retry refetches and then shows the video', async () => {
    stubFetch((_u, n) => (n === 1 ? json({}, 503) : json(video())))
    await mount()
    await user().click(screen.getByRole('button', { name: 'Retry' }))
    await settle()
    expect(requests).toHaveLength(2)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('My Video')
  })

  it.each([
    ['ingest', 'processing', 'running', 'Fetching video details'],
    ['transcribe', 'processing', 'running', 'Transcribing'],
    ['analyze', 'processing', 'running', 'Analysing'],
    ['embed', 'processing', 'running', 'Processing'],
    ['ingest', 'processing', 'pending', 'Queued: Fetching video details'],
    ['analyze', 'processing', 'pending', 'Queued: Analysing'],
    ['weird', 'processing', 'pending', 'Queued: Processing'],
  ])('processing %s/%s shows "%s"', async (kind, status, state, text) => {
    await mountWith(
      video({
        analysis: null,
        status: status as 'processing',
        active_job: { id: 1, kind, state },
      }),
    )
    expect(screen.getByRole('status').textContent).toBe(text)
    expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
    expect(document.querySelector('iframe')).not.toBeNull()
    expect(screen.queryByRole('region', { name: 'TL;DR' })).toBeNull()
  })

  it('failed shows kind and error_class as plain text, no last_error', async () => {
    await mountWith(
      video({
        analysis: null,
        status: 'failed',
        last_failure: { job_id: 3, kind: 'transcribe', error_class: 'WhisperError', finished_at: null },
      }),
    )
    const text = screen.getByRole('status').textContent ?? ''
    expect(text).toContain('Processing failed')
    expect(text).toContain('transcribe')
    expect(text).toContain('WhisperError')
  })

  it('failed tolerates a null error_class and a missing last_failure', async () => {
    await mountWith(
      video({
        analysis: null,
        status: 'failed',
        last_failure: { job_id: 3, kind: 'analyze', error_class: null, finished_at: null },
      }),
    )
    expect(screen.getByRole('status').textContent).toContain('Processing failed')
    expect(body()).not.toContain('null')
  })

  it('failed with no last_failure still says Processing failed', async () => {
    await mountWith(video({ analysis: null, status: 'failed', last_failure: null }))
    expect(screen.getByRole('status').textContent).toBe('Processing failed')
  })

  it('unavailable with no analysis', async () => {
    await mountWith(video({ analysis: null, status: 'unavailable', unavailable: 'private' }))
    expect(screen.getByRole('status').textContent).toBe('This video is no longer available on YouTube')
  })

  it('idle with no analysis', async () => {
    await mountWith(video({ analysis: null, status: 'idle' }))
    expect(screen.getByRole('status').textContent).toBe('Not analysed yet')
  })

  it('an existing analysis plus an active job renders in full with a re-analysis note', async () => {
    await mountWith(video({ status: 'processing', active_job: { id: 2, kind: 'analyze', state: 'pending' } }))
    expect(screen.getByText('Re-analysis in progress')).toBeTruthy()
    expect(screen.getByText('A short summary.')).toBeTruthy()
    expect(screen.queryByText(/Queued:/)).toBeNull()
  })

  it('shows no re-analysis note when there is no active job', async () => {
    await mountWith(video())
    expect(screen.queryByText('Re-analysis in progress')).toBeNull()
  })

  it('an unavailable video with an analysis renders it plus the note', async () => {
    await mountWith(video({ unavailable: 'removed', status: 'unavailable' }))
    expect(screen.getByText('This video is no longer available on YouTube')).toBeTruthy()
    expect(screen.getByText('A short summary.')).toBeTruthy()
  })
})

describe('polling', () => {
  it('refetches once after 15 s while processing and stops once done', async () => {
    stubFetch((_u, n) =>
      n === 1
        ? json(video({ analysis: null, status: 'processing', active_job: { id: 1, kind: 'analyze', state: 'running' } }))
        : json(video()),
    )
    await mount()
    expect(screen.getByRole('status').textContent).toBe('Analysing')
    await advance(14_999)
    expect(requests).toHaveLength(1)
    await advance(1)
    expect(requests).toHaveLength(2)
    // The analysis appears without a reload.
    await advance(1_000)
    expect(screen.getByText('A short summary.')).toBeTruthy()
    await advance(120_000)
    expect(requests).toHaveLength(2)
  })

  it('does not poll a video with no active job', async () => {
    stubFetch(() => json(video()))
    await mount()
    await advance(120_000)
    expect(requests).toHaveLength(1)
  })
})

describe('header and player', () => {
  it('holds the title in the only h1 and sets document.title', async () => {
    await mountWith(video())
    const h1 = screen.getAllByRole('heading', { level: 1 })
    expect(h1).toHaveLength(1)
    expect(h1[0]?.textContent).toBe('My Video')
    expect(document.title).toBe('My Video')
  })

  it.each([null, ''])('falls back to the video_id for title %j', async (title) => {
    await mountWith(video({ title }))
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(ID)
    expect(document.title).toBe(ID)
  })

  it('shows the channel title, falling back to channel_id, else nothing', async () => {
    await mountWith(video())
    expect(screen.getByText('My Channel')).toBeTruthy()
  })

  it('falls back to the channel_id when the channel title is null', async () => {
    await mountWith(video({ channel_title: null, channel_id: 'UCfallbackfallbackfallba' }))
    expect(screen.getByText('UCfallbackfallbackfallba')).toBeTruthy()
  })

  it('omits the channel when both are null', async () => {
    await mountWith(video({ channel_title: null, channel_id: null }))
    expect(screen.queryByText('My Channel')).toBeNull()
    expect(screen.queryByTestId('channel')).toBeNull()
  })

  it('shows the publish date in UTC regardless of the local zone', async () => {
    // 23:30 at UTC-7 on 30 Sep is 06:30 UTC on 1 Oct; the pinned TZ is LA.
    await mountWith(video())
    expect(screen.getByText('2026-10-01')).toBeTruthy()
  })

  it('omits date and duration when null, and ignores an unparsable date', async () => {
    await mountWith(video({ published_at: null, duration_sec: null }))
    expect(body()).not.toMatch(/\d{4}-\d{2}-\d{2}(?!T)/)
    expect(screen.queryByText('1:02:03')).toBeNull()
  })

  it('omits an unparsable published_at', async () => {
    await mountWith(video({ published_at: 'not a date' }))
    expect(body()).not.toContain('Invalid')
    expect(body()).not.toContain('not a date')
  })

  it('shows the duration through the shared time formatter', async () => {
    await mountWith(video())
    expect(screen.getByText('1:02:03')).toBeTruthy()
  })

  it('links to YouTube in a new tab with noopener noreferrer', async () => {
    await mountWith(video({ video_id: DASH_ID }))
    const a = screen.getByRole('link', { name: 'Open on YouTube' })
    expect(a.getAttribute('href')).toBe(`https://www.youtube.com/watch?v=${DASH_ID}`)
    expect(a.getAttribute('target')).toBe('_blank')
    expect(a.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('shows a plain Printable version link only when there is an analysis', async () => {
    await mountWith(video())
    const a = screen.getByRole('link', { name: 'Printable version' })
    expect(a.getAttribute('href')).toBe(`/api/videos/${ID}/render`)
    expect(a.getAttribute('target')).toBe('_blank')
    expect(a.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('omits Printable version when analysis is null', async () => {
    await mountWith(video({ analysis: null, status: 'idle' }))
    expect(screen.queryByRole('link', { name: 'Printable version' })).toBeNull()
    expect(screen.getByRole('link', { name: 'Open on YouTube' })).toBeTruthy()
  })

  it('embeds the player for the validated id, above the analysis', async () => {
    await mountWith(video())
    const iframe = document.querySelector('iframe') as HTMLIFrameElement
    expect(new URL(iframe.src).pathname).toBe(`/embed/${ID}`)
    expect(document.querySelectorAll('iframe')).toHaveLength(1)
    const tldr = section('TL;DR')
    expect(iframe.compareDocumentPosition(tldr) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('embeds the player when there is no analysis', async () => {
    await mountWith(video({ analysis: null, status: 'idle' }))
    expect(document.querySelector('iframe')).not.toBeNull()
  })
})

describe('analysis content', () => {
  it('orders the sections TL;DR, Speakers, Topics, Claims, Quotes, with one h2 each', async () => {
    await mountWith(video())
    const h2 = screen.getAllByRole('heading', { level: 2 }).map((h) => h.textContent)
    expect(h2).toEqual(['TL;DR', 'Speakers', 'Topics', 'Claims', 'Quotes'])
    const footer = (document.querySelector('footer') as HTMLElement)
    const quotes = section('Quotes')
    expect(quotes.compareDocumentPosition(footer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('keeps line breaks in the tl;dr', async () => {
    await mountWith(video({ analysis: analysis({ tldr: 'line one\nline two' }) }))
    const el = within(section('TL;DR')).getByText(/line one/)
    expect(el.textContent).toBe('line one\nline two')
    expect(getComputedStyle(el).whiteSpace).toBe('pre-line')
  })

  it('lists roster speakers with name and role', async () => {
    await mountWith(
      video({ analysis: analysis({ speaker_roster: { speakers: [{ name: 'Ada', role: 'host' }, { name: 'Bob' }] } }) }),
    )
    const items = within(section('Speakers')).getAllByRole('listitem')
    expect(items.map((i) => i.textContent)).toEqual(['Ada (host)', 'Bob'])
  })

  it.each([
    ['null', null],
    ['a string', 'x'],
    ['an array', []],
    ['no speakers key', {}],
    ['speakers not an array', { speakers: 4 }],
    ['an empty array', { speakers: [] }],
    ['only invalid entries', { speakers: [null, 1, {}, { name: '' }, { name: 3 }] }],
  ])('leaves the Speakers section out for %s', async (_n, roster) => {
    await mountWith(video({ analysis: analysis({ speaker_roster: roster }) }))
    expect(screen.queryByRole('heading', { name: 'Speakers' })).toBeNull()
    expect(screen.getByRole('heading', { name: 'Topics' })).toBeTruthy()
  })

  it('skips malformed roster entries but keeps the valid ones, with a non-string role dropped', async () => {
    await mountWith(
      video({ analysis: analysis({ speaker_roster: { speakers: [7, { name: 'Ok', role: 9 }, { name: '' }] } }) }),
    )
    const items = within(section('Speakers')).getAllByRole('listitem')
    expect(items.map((i) => i.textContent)).toEqual(['Ok'])
  })

  it('shows topics in order with title, summary and a seeking timestamp', async () => {
    await mountWith(
      video({
        analysis: analysis({
          topics: [
            { seq: 0, title: 'First', summary: 'About first', start_sec: 5 },
            { seq: 1, title: 'Second', summary: null, start_sec: null },
          ],
        }),
      }),
    )
    const items = within(section('Topics')).getAllByRole('listitem')
    expect(items).toHaveLength(2)
    expect(items[0]?.textContent).toContain('First')
    expect(items[0]?.textContent).toContain('About first')
    expect(items[1]?.textContent).toContain('Second')
    expect(within(items[1] as HTMLElement).queryByRole('button')).toBeNull()
    await user().click(within(items[0] as HTMLElement).getByRole('button', { name: 'Seek to 0:05' }))
    expect(seeks).toEqual([5])
  })

  it('shows claims with speaker, confidence and a timestamp that seeks the player', async () => {
    await mountWith(video())
    const item = within(section('Claims')).getByRole('listitem')
    expect(item.textContent).toContain('Water is wet')
    expect(item.textContent).toContain('Ada')
    expect(item.textContent).toContain('Confidence: high')
    await user().click(within(item).getByRole('button', { name: 'Seek to 1:05' }))
    expect(seeks).toEqual([65])
  })

  it('shows a claim timestamp that is reachable from the keyboard (a real button)', async () => {
    await mountWith(video())
    const btn = within(section('Claims')).getByRole('button', { name: 'Seek to 1:05' })
    expect(btn.tagName).toBe('BUTTON')
    expect(btn.getAttribute('tabindex')).not.toBe('-1')
  })

  it('labels speaker "unknown" as Unattributed in claims and quotes, never dropping them', async () => {
    await mountWith(
      video({
        analysis: analysis({
          claims: [
            { text: 'C1', speaker: 'unknown', confidence: null, start_sec: null, source_chunk_seq: null },
            { text: 'C2', speaker: 'Zed (not in roster)', confidence: 'low', start_sec: 1, source_chunk_seq: null },
          ],
          quotes: [{ text: 'Q1', speaker: 'unknown', start_sec: null, source_chunk_seq: null }],
        }),
      }),
    )
    const claims = within(section('Claims')).getAllByRole('listitem')
    expect(claims).toHaveLength(2)
    expect(claims[0]?.textContent).toContain('Unattributed')
    expect(claims[0]?.textContent).not.toContain('unknown')
    expect(claims[1]?.textContent).toContain('Zed (not in roster)')
    const quote = within(section('Quotes')).getByRole('listitem')
    expect(quote.textContent).toContain('Unattributed')
    expect(quote.textContent).not.toContain('unknown')
  })

  it('does not treat other casings of unknown as unattributed', async () => {
    await mountWith(
      video({
        analysis: analysis({
          claims: [{ text: 'C', speaker: 'Unknown', confidence: null, start_sec: null, source_chunk_seq: null }],
        }),
      }),
    )
    expect(within(section('Claims')).getByRole('listitem').textContent).toContain('Unknown')
  })

  it('shows no confidence label for null and the stored text for any other value', async () => {
    await mountWith(
      video({
        analysis: analysis({
          claims: [
            { text: 'A', speaker: 'Ada', confidence: null, start_sec: null, source_chunk_seq: null },
            { text: 'B', speaker: 'Ada', confidence: 'medium', start_sec: null, source_chunk_seq: null },
            { text: 'C', speaker: 'Ada', confidence: 'very sure', start_sec: null, source_chunk_seq: null },
          ],
        }),
      }),
    )
    const [a, b, c] = within(section('Claims')).getAllByRole('listitem')
    expect(a?.textContent).not.toContain('Confidence')
    expect(b?.textContent).toContain('Confidence: medium')
    expect(c?.textContent).toContain('Confidence: very sure')
  })

  it('renders no timestamp and no placeholder for a null start_sec', async () => {
    await mountWith(
      video({
        analysis: analysis({
          claims: [{ text: 'A', speaker: 'Ada', confidence: null, start_sec: null, source_chunk_seq: null }],
          quotes: [{ text: 'Q', speaker: 'Ada', start_sec: null, source_chunk_seq: null }],
          topics: [{ seq: 0, title: 'T', summary: null, start_sec: null }],
        }),
      }),
    )
    expect(within(section('Claims')).queryByRole('button')).toBeNull()
    expect(within(section('Quotes')).queryByRole('button')).toBeNull()
    expect(within(section('Topics')).queryByRole('button')).toBeNull()
    expect(document.querySelectorAll('li time')).toHaveLength(0)
    expect(section('Claims').textContent).not.toMatch(/--|n\/a|null/i)
  })

  it('leaves a negative start_sec to Timestamp: no timestamp', async () => {
    await mountWith(
      video({
        analysis: analysis({
          quotes: [{ text: 'Q', speaker: 'Ada', start_sec: -3, source_chunk_seq: null }],
        }),
      }),
    )
    expect(within(section('Quotes')).queryByRole('button')).toBeNull()
  })

  it('a click on a quote timestamp seeks with its start_sec', async () => {
    await mountWith(video())
    await user().click(within(section('Quotes')).getByRole('button', { name: 'Seek to 0:10' }))
    expect(seeks).toEqual([10])
  })

  it.each([
    ['Topics', 'No topics extracted.', { topics: [] }],
    ['Claims', 'No claims extracted.', { claims: [] }],
    ['Quotes', 'No quotes extracted.', { quotes: [] }],
  ])('keeps the %s heading with a fixed line when empty', async (name, line, over) => {
    await mountWith(video({ analysis: analysis(over) }))
    expect(within(section(name)).getByText(line)).toBeTruthy()
    expect(within(section(name)).queryByRole('listitem')).toBeNull()
  })

  it('the footer shows model, prompt version and UTC ISO created_at only', async () => {
    await mountWith(video({ analysis: analysis({ created_at: '2026-10-01T01:09:10-07:00' }) }))
    const footer = (document.querySelector('footer') as HTMLElement).textContent ?? ''
    expect(footer).toContain('test-model-1')
    expect(footer).toContain('v7')
    expect(footer).toContain('2026-10-01T08:09:10.000Z')
    for (const hidden of ['111111', '222222', '3.5', '987654']) expect(body()).not.toContain(hidden)
  })
})

describe('untrusted text', () => {
  const SCRIPT = '<script>alert(1)</script>'
  const IMG = '"><img src=x onerror=alert(1)>'

  it('renders markup in every text field as literal text', async () => {
    await mountWith(
      video({
        title: SCRIPT,
        channel_title: IMG,
        analysis: analysis({
          tldr: SCRIPT,
          speaker_roster: { speakers: [{ name: IMG, role: SCRIPT }] },
          topics: [{ seq: 0, title: SCRIPT, summary: IMG, start_sec: null }],
          claims: [{ text: IMG, speaker: SCRIPT, confidence: IMG, start_sec: null, source_chunk_seq: null }],
          quotes: [{ text: SCRIPT, speaker: IMG, start_sec: null, source_chunk_seq: null }],
          model: SCRIPT,
          prompt_version: IMG,
        }),
      }),
    )
    expect(document.querySelector('script')).toBeNull()
    expect(document.querySelector('img')).toBeNull()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(SCRIPT)
    expect(within(section('TL;DR')).getByText(SCRIPT)).toBeTruthy()
    expect(within(section('Quotes')).getAllByRole('listitem')[0]?.textContent).toContain(SCRIPT)
    expect(document.title).toBe(SCRIPT)
  })

  it('builds no URL from data: only the three known links exist', async () => {
    await mountWith(
      video({
        title: 'https://evil.example/x',
        analysis: analysis({ tldr: 'see https://evil.example/a and www.evil.example' }),
      }),
    )
    const hrefs = [...document.querySelectorAll('a')].map((a) => a.getAttribute('href'))
    expect(hrefs.sort()).toEqual(
      [`/api/videos/${ID}/render`, `https://www.youtube.com/watch?v=${ID}`].sort(),
    )
    expect(document.querySelectorAll('[src]:not(iframe)')).toHaveLength(0)
  })

  it('renders non-ASCII text unchanged', async () => {
    const text = 'Café 日本語 😀 مرحبا'
    await mountWith(
      video({ title: text, analysis: analysis({ tldr: text, claims: [{ text, speaker: text, confidence: null, start_sec: null, source_chunk_seq: null }] }) }),
    )
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(text)
    expect(within(section('TL;DR')).getByText(text)).toBeTruthy()
    expect(within(section('Claims')).getByRole('listitem').textContent).toContain(text)
  })
})

describe('accessibility', () => {
  it('uses lists for topics, claims and quotes', async () => {
    await mountWith(video())
    for (const name of ['Topics', 'Claims', 'Quotes']) {
      expect(within(section(name)).getByRole('list')).toBeTruthy()
    }
  })
})

describe('link to the transcript (#51)', () => {
  const summary = { id: 3, source: 'whisper', language: 'en', segment_count: 10, speaker_source: 'none' }

  it('links to the transcript route when the video has a transcript', async () => {
    await mountWith(video({ transcript: summary }))
    expect(screen.getByRole('link', { name: 'Transcript' }).getAttribute('href')).toBe(
      `/videos/${ID}/transcript`,
    )
  })

  it.each(['processing', 'failed', 'idle'] as const)(
    'has no link when the transcript is null and the video is %s',
    async (status) => {
      await mountWith(video({ transcript: null, status, analysis: null }))
      expect(screen.queryByRole('link', { name: 'Transcript' })).toBeNull()
    },
  )

  it('has no link for a done video whose transcript is null', async () => {
    await mountWith(video({ transcript: null }))
    expect(screen.queryByRole('link', { name: 'Transcript' })).toBeNull()
  })
})
