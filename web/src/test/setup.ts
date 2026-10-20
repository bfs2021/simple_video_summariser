// The single setup file of the jsdom Vitest project (web/vite.config.ts, #127).
// Do not duplicate any of this per test file.
import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, vi } from 'vitest'

// Vitest `globals` is off (as in #47), so Testing Library cannot register its
// own automatic cleanup: it only does so when a global `afterEach` exists.
// This explicit afterEach(cleanup) is required. Do not "simplify" it away.
//
// Convention for jsdom-unimplemented APIs (matchMedia, ResizeObserver,
// IntersectionObserver, scrollIntoView, Element.animate): none is stubbed here
// because no component needs one yet. The issue that first needs one adds it
// to this file, with a comment naming the component that needs it.

// jsdom does not implement window.scrollTo; react-router's <ScrollRestoration />
// (routes/Shell.tsx) calls it on every navigation.
beforeEach(() => {
  vi.stubGlobal('scrollTo', () => {})
})

class UnexpectedNetworkAccessError extends Error {
  override name = 'UnexpectedNetworkAccessError'
}

const initialTitle = document.title
const initialHead = document.head.innerHTML

function blockedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const method = (init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase()
  const url = input instanceof Request ? input.url : String(input)
  const file = expect.getState().testPath ?? 'unknown test file'
  throw new UnexpectedNetworkAccessError(
    `unexpected network access in tests: ${method} ${url} (in ${file}). Stub fetch in the test.`,
  )
}

beforeEach(() => {
  vi.stubGlobal('fetch', blockedFetch)
})

afterEach(() => {
  cleanup()
  document.body.replaceChildren()
  document.head.innerHTML = initialHead
  document.title = initialTitle
  localStorage.clear()
  sessionStorage.clear()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})
