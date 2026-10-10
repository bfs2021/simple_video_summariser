// The request helper through the REAL app wiring (#118): the shared `routes`
// table and `createQueryClient()` (the production retry policy), not a router
// or client built by the test. Per-view tests build their own harness, and that
// is how a missing provider and query-string erosion once stayed green.
import { QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createQueryClient } from '../queryClient'
import { routes } from '../routes'

interface Call {
  url: URL
  init: RequestInit
}

let calls: Call[]

function stub(respond: (url: URL) => Response) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost')
      calls.push({ url, init: init ?? {} })
      return respond(url)
    }),
  )
}

function mount(entry: string) {
  const router = createMemoryRouter(routes, { initialEntries: [entry] })
  render(
    <QueryClientProvider client={createQueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  return router
}

async function settle() {
  for (let i = 0; i < 3; i++) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(50)
    })
  }
}

const json = (body: unknown, status: number) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

beforeEach(() => {
  vi.useFakeTimers()
  calls = []
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('the helper through the production routes and query client', () => {
  it('requests a video with a leading dash through the helper, with its fixed options', async () => {
    stub(() => json({ detail: 'video not found' }, 404))
    mount('/videos/-wNyEUrxzFU')
    await settle()

    const video = calls.filter((c) => c.url.pathname === '/api/videos/-wNyEUrxzFU')
    expect(video).toHaveLength(1) // a 404 is not retried by the production policy
    const init = video[0]?.init as RequestInit
    expect(init.method).toBe('GET')
    expect(init.credentials).toBe('same-origin')
    expect(init.headers).toEqual({ Accept: 'application/json' })
    expect(init.signal).toBeInstanceOf(AbortSignal)
  })

  it('the 404 of the helper reaches the view as its own not-found state', async () => {
    stub(() => json({ detail: 'video not found' }, 404))
    mount('/videos/-wNyEUrxzFU')
    await settle()

    expect(screen.getByRole('heading', { name: 'Video not found' })).toBeTruthy()
  })

  it('a 503 is retried by the production policy at most twice, then shown', async () => {
    stub(() => json({ detail: 'database unavailable' }, 503))
    mount('/videos/-wNyEUrxzFU')
    // The default retry delay doubles from 1 s; give the three attempts time.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000)
    })

    expect(calls.filter((c) => c.url.pathname === '/api/videos/-wNyEUrxzFU')).toHaveLength(3)
  })

  it('the library route reaches the helper as a typed query string', async () => {
    stub(() => json({ items: [], total: 0 }, 200))
    const router = mount('/?status=failed&page=2')
    await settle()

    const list = calls.find((c) => c.url.pathname === '/api/videos')
    expect(list?.url.search).toBe('?status=failed&limit=50&offset=50')
    expect(router.state.location.search).toBe('?status=failed&page=2')
  })

  it('the ops route reads queue depth and jobs through the helper', async () => {
    stub((url) =>
      url.pathname === '/api/healthz'
        ? json({ status: 'ok', queue: {} }, 200)
        : json({ items: [], next_before: null }, 200),
    )
    mount('/ops?state=pending&kind=ingest&error_class=none')
    await settle()

    const jobs = calls.find((c) => c.url.pathname === '/api/ops/jobs')
    expect(jobs?.url.search).toBe('?state=pending&kind=ingest&error_class=none&limit=50')
    expect(calls.some((c) => c.url.pathname === '/api/healthz')).toBe(true)
    expect(calls.every((c) => (c.init.headers as Record<string, string>).Accept === 'application/json')).toBe(true)
  })
})
