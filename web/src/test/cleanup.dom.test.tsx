import { render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { expect, it, vi } from 'vitest'
import { Greeting } from './fixtures/Greeting'

const cleanups: string[] = []
const originalTitle = document.title

function Tracked({ name }: { name: string }) {
  useEffect(() => () => void cleanups.push(name), [name])
  return null
}

function expectClean(previous: string) {
  expect(document.body.childElementCount).toBe(0)
  expect(document.getElementById(`stray-${previous}`)).toBeNull()
  expect(document.getElementById(`stray-link-${previous}`)).toBeNull()
  expect(cleanups).toContain(previous)
  expect(document.title).toBe(originalTitle)
  expect(localStorage.length).toBe(0)
  expect(sessionStorage.length).toBe(0)
  expect(`leaky${previous}` in globalThis).toBe(false)
  expect(vi.isFakeTimers()).toBe(false)
}

function dirty(name: string) {
  render(<Greeting name={name} />)
  render(<Tracked name={name} />)
  expect(screen.getByRole('heading', { name: `Hello, ${name}` })).toBeTruthy()
  const stray = document.createElement('div')
  stray.id = `stray-${name}`
  document.body.append(stray)
  const link = document.createElement('link')
  link.id = `stray-link-${name}`
  document.head.append(link)
  document.title = `dirty ${name}`
  localStorage.setItem('k', name)
  sessionStorage.setItem('k', name)
  vi.stubGlobal(`leaky${name}`, 1)
  vi.useFakeTimers()
}

let previous: string | undefined

function verifyPreviousAndDirty(name: string) {
  if (previous !== undefined) expectClean(previous)
  dirty(name)
  previous = name
}

// Both cases run the same verify-then-dirty protocol, so whichever runs second
// proves the explicit setup afterEach reset. Do not simplify this back into a
// first-writes/second-reads pair whose result depends on shuffled test order.
it('resets state between the Ada and Grace cases', () => {
  verifyPreviousAndDirty('Ada')
})

it('resets state between the Grace and Ada cases', () => {
  verifyPreviousAndDirty('Grace')
})
