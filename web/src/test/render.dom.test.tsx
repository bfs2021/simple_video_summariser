import { act, render, screen, waitFor } from '@testing-library/react'
import { setupUser } from './user'
import { describe, expect, it, vi } from 'vitest'
import { DelayedStatus, Greeting, MessageListener } from './fixtures/Greeting'

describe('component rendering', () => {
  it('renders a component and finds it through a query', () => {
    render(<Greeting name="Ada" />)
    expect(screen.getByRole('heading', { name: 'Hello, Ada' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Wave' })).toHaveAccessibleName('Wave')
  })

  it('tabs to and activates a button through a realistic keyboard interaction', async () => {
    const onClick = vi.fn()
    render(<button onClick={onClick}>go</button>)
    const user = setupUser()

    await user.tab()
    expect(screen.getByRole('button', { name: 'go' })).toHaveFocus()
    await user.keyboard('{Enter}')

    expect(onClick).toHaveBeenCalledOnce()
  })
})

describe('fake timers with waitFor', () => {
  it('updates after a timeout once the timers are advanced explicitly', async () => {
    vi.useFakeTimers()
    render(<DelayedStatus delayMs={15_000} />)
    expect(screen.getByRole('status').textContent).toBe('loading')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    // Testing Library only detects Jest's fake timers, so waitFor's trailing
    // setTimeout(0) is faked here and must be advanced by hand.
    const ready = waitFor(() => expect(screen.getByRole('status').textContent).toBe('ready'))
    await vi.advanceTimersByTimeAsync(0)
    await ready
  })
})

describe('window message events', () => {
  it('re-renders a component that handles a message', () => {
    render(<MessageListener />)
    expect(screen.getByRole('status').textContent).toBe('none')
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: 'seek:42' }))
    })
    expect(screen.getByRole('status').textContent).toBe('seek:42')
  })
})
