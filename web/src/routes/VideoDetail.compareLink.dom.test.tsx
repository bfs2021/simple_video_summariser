import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { components } from '../api'
import { FakePlayerProvider } from '../components/FakePlayerProvider'
import { Compare } from './Compare'
import { VideoDetail } from './VideoDetail'

// The "Compare analyses" entry point of #53 on the Video detail view (#50).
type VideoDetailOut = components['schemas']['VideoDetail']

const ID = 'dQw4w9WgXcQ'
const DASH_ID = '-wNyEUrxzFU'

function video(id = ID): VideoDetailOut {
  return {
    video_id: id,
    title: 'My Video',
    channel_id: 'UC' + 'a'.repeat(22),
    channel_title: 'My Channel',
    published_at: null,
    duration_sec: null,
    latest_analysis_at: null,
    origin: 'manual',
    unavailable: null,
    status: 'idle',
    active_job: null,
    last_failure: null,
    transcript: null,
    analysis: null,
    progress: null,
  }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

let requests: URL[]

function stubApi(analyses: (url: URL) => Response | Promise<Response>) {
  requests = []
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost')
      requests.push(url)
      return Promise.resolve(url.pathname.endsWith('/analyses') ? analyses(url) : json(video(url.pathname.split('/')[3])))
    }),
  )
}

const totalResponse = (total: number) => () => json({ items: [], limit: 1, offset: 0, total })

async function mount(id = ID) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [
      {
        path: '/videos/:videoId',
        element: (
          <FakePlayerProvider>
            <VideoDetail />
          </FakePlayerProvider>
        ),
      },
    ],
    { initialEntries: [`/videos/${id}`] },
  )
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
})

const link = () => screen.queryByRole('link', { name: 'Compare analyses' })

describe('Compare analyses link on the Video detail view', () => {
  it.each([
    [0, false],
    [1, false],
    [2, true],
    [3, true],
    [250, true],
  ])('with total=%i the link is present: %s', async (total, present) => {
    stubApi(totalResponse(total))
    await mount()
    expect(link() !== null).toBe(present)
    if (present) expect(link()?.getAttribute('href')).toBe(`/videos/${ID}/compare`)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('My Video')
  })

  it('asks for one analysis only: limit=1, offset=0, once', async () => {
    stubApi(totalResponse(2))
    await mount()
    const asked = requests.filter((u) => u.pathname.endsWith('/analyses'))
    expect(asked).toHaveLength(1)
    expect(asked[0]?.pathname).toBe(`/api/videos/${ID}/analyses`)
    expect(asked[0]?.searchParams.get('limit')).toBe('1')
    expect(asked[0]?.searchParams.get('offset')).toBe('0')
  })

  it('builds the href from the validated id, leading dash included', async () => {
    stubApi(totalResponse(2))
    await mount(DASH_ID)
    expect(link()?.getAttribute('href')).toBe(`/videos/${DASH_ID}/compare`)
  })

  it.each([
    ['a 500', () => json({ detail: 'SECRET' }, 500)],
    ['a 404', () => json({ detail: 'SECRET' }, 404)],
    ['a network error', () => Promise.reject(new TypeError('SECRET'))],
    ['a response of the wrong shape', () => json({ items: [] })],
    ['a fractional total', () => json({ items: [], total: 2.5 })],
    ['a negative total', () => json({ items: [], total: -5 })],
    ['a non-numeric total', () => json({ items: [], total: '5' })],
  ])('omits the link and leaves the view intact after %s', async (_n, respond) => {
    stubApi(respond)
    await mount()
    expect(link()).toBeNull()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('My Video')
    expect(document.body.textContent).not.toContain('SECRET')
  })

  it('makes no analyses request for an invalid id', async () => {
    stubApi(totalResponse(5))
    await mount('abcdefghij')
    expect(requests).toHaveLength(0)
  })
})

describe('query cache', () => {
  it('keeps the count and the full walk under separate keys in one client', async () => {
    const run = (id: number, model: string) => ({
      id, transcript_id: 1, transcript_source: 'captions', chunk_strategy: 'fixed', model, prompt_version: 'v1',
      created_at: '2026-10-01T08:09:10Z', input_tokens: 1, output_tokens: 1, cost_usd: null, duration_ms: null,
      tldr: 't', speaker_roster: null, topics: [], claims: [], quotes: [],
    })
    const items = [run(2, 'a'), run(1, 'b')]
    stubApi((url) =>
      json({ items: url.searchParams.get('limit') === '1' ? items.slice(0, 1) : items, limit: 50, offset: 0, total: 2 }),
    )
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const router = createMemoryRouter(
      [
        { path: '/videos/:videoId', element: <VideoDetail /> },
        { path: '/videos/:videoId/compare', element: <FakePlayerProvider><Compare /></FakePlayerProvider> },
      ],
      { initialEntries: [`/videos/${ID}`] },
    )
    render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(link()).not.toBeNull()
    await act(async () => {
      await router.navigate(`/videos/${ID}/compare`)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByLabelText('Analysis A')).toBeTruthy()
    expect(screen.getByLabelText('Analysis B')).toBeTruthy()
  })
})
