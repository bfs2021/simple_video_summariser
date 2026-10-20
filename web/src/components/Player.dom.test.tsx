import { act, fireEvent, render, screen } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Player } from './Player'
import { PlayerProvider, usePlayer } from './PlayerContext'
import { setupUser } from '../test/user'
import { Timestamp } from './Timestamp'

const click = (element: HTMLElement) => setupUser().click(element)

const ORIGIN = 'https://www.youtube-nocookie.com'
const ID = 'dQw4w9WgXcQ'

let posts: Array<{ message: Record<string, unknown>; targetOrigin: string }>

function StatusProbe() {
  const player = usePlayer()
  return <output data-testid="status">{player?.status ?? 'none'}</output>
}

function setup(ui: React.ReactNode = <Player videoId={ID} />, strict = false) {
  const tree = (
    <PlayerProvider>
      {ui}
      <StatusProbe />
      <Timestamp seconds={30.5} />
    </PlayerProvider>
  )
  const result = render(strict ? <StrictMode>{tree}</StrictMode> : tree)
  return { ...result, tree, strict }
}

function iframeOf(container: HTMLElement): HTMLIFrameElement {
  const el = container.querySelector('iframe')
  if (!el) throw new Error('no iframe rendered')
  return el
}

/** Records what the page posts to the iframe's (real jsdom) contentWindow. */
function spyOnEmbed(iframe: HTMLIFrameElement): Window {
  const target = iframe.contentWindow
  if (!target) throw new Error('jsdom iframe has no contentWindow')
  vi.spyOn(target, 'postMessage').mockImplementation(((message: string, targetOrigin: string) => {
    posts.push({ message: JSON.parse(message), targetOrigin })
  }) as Window['postMessage'])
  return target
}

function fromEmbed(source: Window, data: unknown, origin: string = ORIGIN) {
  act(() => {
    window.dispatchEvent(new MessageEvent('message', { data, origin, source }))
  })
}
const json = (event: string, extra: object = {}) => JSON.stringify({ event, ...extra })
const status = () => screen.getByTestId('status').textContent
const commands = () => posts.filter((p) => p.message['event'] === 'command')
const handshakes = () => posts.filter((p) => p.message['event'] === 'listening')

beforeEach(() => {
  vi.useFakeTimers()
  posts = []
})
afterEach(() => {
  vi.useRealTimers()
})

describe('Player with an invalid videoId', () => {
  it.each([
    ['10 characters', 'dQw4w9WgXc'],
    ['12 characters', 'dQw4w9WgXcQQ'],
    ['empty', ''],
    ['path traversal', '../../evil/x'],
    ['encoded slash', 'abc%2Fdefghi'],
    ['whitespace', 'dQw4w9 gXcQ'],
    ['trailing newline', 'dQw4w9WgXc\n'],
    ['query char', 'dQw4w9WgX?Q'],
    ['hash char', 'dQw4w9WgX#Q'],
    ['non-ASCII', 'dQw4w9WgXcé'],
    ['url', 'https://evil.example/'],
  ])('%s renders no iframe, says unavailable and reports error', (_name, id) => {
    const { container } = setup(<Player videoId={id} />)
    expect(container.querySelector('iframe')).toBeNull()
    expect(container.querySelector('.player--error')?.textContent).toContain('Video unavailable')
    expect(container.innerHTML).not.toContain('evil')
    expect(status()).toBe('error')
  })

  it('seekTo in that state does nothing and opens no tab', async () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null)
    setup(<Player videoId="short" />)
    await click(screen.getByRole('button', { name: /Seek to/ }))
    expect(open).not.toHaveBeenCalled()
    expect(posts).toEqual([])
  })

  it('the case of the id matters and a leading dash is valid', () => {
    const { container } = setup(<Player videoId="-wNyEUrxzFU" />)
    expect(iframeOf(container).src).toContain('/embed/-wNyEUrxzFU')
  })
})

describe('Player iframe attributes', () => {
  it('builds the src with the nocookie host and exactly the required params', () => {
    const { container } = setup()
    const url = new URL(iframeOf(container).getAttribute('src') ?? '')
    expect(url.origin).toBe(ORIGIN)
    expect(url.pathname).toBe(`/embed/${ID}`)
    expect(Object.fromEntries(url.searchParams)).toEqual({
      enablejsapi: '1',
      origin: window.location.origin,
      playsinline: '1',
      rel: '0',
    })
    expect(url.searchParams.has('autoplay')).toBe(false)
  })

  it('sets title, allow, allowfullscreen, referrerpolicy and responsive layout', () => {
    const { container } = setup()
    const iframe = iframeOf(container)
    expect(iframe.getAttribute('title')).toBe('YouTube video player')
    expect(iframe.getAttribute('allow')).toBe('autoplay; encrypted-media; picture-in-picture; fullscreen')
    expect(iframe.hasAttribute('allowfullscreen')).toBe(true)
    expect(iframe.getAttribute('referrerpolicy')).toBe('strict-origin-when-cross-origin')
    expect(iframe.style.width).toBe('100%')
    expect(iframe.style.aspectRatio).toBe('16 / 9')
    expect(iframe.getAttribute('width')).toBeNull()
    expect(iframe.getAttribute('height')).toBeNull()
    expect(container.querySelector('.player')).not.toBeNull()
  })

  it('uses "Video: <title>" and never treats the title as HTML', () => {
    const { container } = setup(<Player videoId={ID} title={'<img src=x onerror=alert(1)>'} />)
    const iframe = iframeOf(container)
    expect(iframe.getAttribute('title')).toBe('Video: <img src=x onerror=alert(1)>')
    expect(container.querySelector('img')).toBeNull()
  })

  it('falls back to the default title for an empty title', () => {
    const { container } = setup(<Player videoId={ID} title="" />)
    expect(iframeOf(container).getAttribute('title')).toBe('YouTube video player')
  })

  it('loads nothing: no fetch and no script element', () => {
    const { container } = setup()
    expect(document.querySelector('script')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })

  it('starts in loading status', () => {
    setup()
    expect(status()).toBe('loading')
  })
})

describe('Player handshake, ready and seek queue', () => {
  it('sends the handshake only after the iframe load event', () => {
    const { container } = setup()
    const iframe = iframeOf(container)
    spyOnEmbed(iframe)
    vi.advanceTimersByTime(2_000)
    expect(posts).toEqual([])
    fireEvent.load(iframe)
    expect(handshakes().length).toBeGreaterThanOrEqual(1)
    expect(handshakes()[0]?.targetOrigin).toBe(ORIGIN)
  })

  it('queues seeks while loading and sends only the last one once ready', async () => {
    const { container } = setup()
    const iframe = iframeOf(container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    const button = screen.getByRole('button', { name: /Seek to/ })
    await click(button)
    await click(button)
    await click(button)
    expect(commands()).toEqual([])
    fromEmbed(embed, json('onReady'))
    expect(status()).toBe('ready')
    expect(commands().map((c) => [c.message['func'], c.message['args']])).toEqual([
      ['seekTo', [30.5, true]],
      ['playVideo', []],
    ])
    for (const c of commands()) expect(c.targetOrigin).toBe(ORIGIN)
  })

  it('once ready each click sends one seekTo and one playVideo', async () => {
    const { container } = setup()
    const iframe = iframeOf(container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    fromEmbed(embed, json('initialDelivery'))
    expect(status()).toBe('ready')
    await click(screen.getByRole('button', { name: /Seek to/ }))
    await click(screen.getByRole('button', { name: /Seek to/ }))
    expect(commands().map((c) => c.message['func'])).toEqual(['seekTo', 'playVideo', 'seekTo', 'playVideo'])
  })

  it('stops the handshake once ready', () => {
    const { container } = setup()
    const iframe = iframeOf(container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    fromEmbed(embed, json('onReady'))
    const count = handshakes().length
    vi.advanceTimersByTime(30_000)
    expect(handshakes()).toHaveLength(count)
  })
})

describe('Player ignores hostile messages', () => {
  const otherFrame = () => {
    const other = document.createElement('iframe')
    document.body.append(other)
    const w = other.contentWindow
    if (!w) throw new Error('no window')
    return w
  }

  it.each([
    ['other origin', (_e: Window) => ({ data: json('onReady'), origin: 'https://www.youtube.com', source: _e })],
    ['evil origin', (_e: Window) => ({ data: json('onReady'), origin: 'https://evil.example', source: _e })],
    ['window itself as source', (_e: Window) => ({ data: json('onReady'), origin: ORIGIN, source: window })],
    ['another iframe as source', (_e: Window) => ({ data: json('onReady'), origin: ORIGIN, source: otherFrame() })],
    ['null source', (_e: Window) => ({ data: json('onReady'), origin: ORIGIN, source: null })],
    ['non-string data', (_e: Window) => ({ data: { event: 'onReady' }, origin: ORIGIN, source: _e })],
    ['invalid JSON', (_e: Window) => ({ data: '{oops', origin: ORIGIN, source: _e })],
    ['JSON null', (_e: Window) => ({ data: 'null', origin: ORIGIN, source: _e })],
    ['array', (_e: Window) => ({ data: '[]', origin: ORIGIN, source: _e })],
    ['number', (_e: Window) => ({ data: '1', origin: ORIGIN, source: _e })],
    ['no event', (_e: Window) => ({ data: '{}', origin: ORIGIN, source: _e })],
    ['unknown event', (_e: Window) => ({ data: json('whatever'), origin: ORIGIN, source: _e })],
    ['forged error from wrong origin', (_e: Window) => ({ data: json('onError', { info: 150 }), origin: 'https://evil.example', source: _e })],
  ])('%s changes nothing', async (_name, build) => {
    const log = vi.spyOn(console, 'error').mockImplementation(() => {})
    const { container } = setup()
    const iframe = iframeOf(container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    const init: MessageEventInit<unknown> = build(embed)
    expect(() =>
      act(() => {
        window.dispatchEvent(new MessageEvent('message', init))
      }),
    ).not.toThrow()
    expect(status()).toBe('loading')
    await click(screen.getByRole('button', { name: /Seek to/ }))
    expect(commands()).toEqual([])
    expect(container.querySelector('iframe')).not.toBeNull()
    expect(log).not.toHaveBeenCalled()
  })
})

describe('Player failure paths', () => {
  it.each([2, 5, 100, 101, 150, 999])('onError %s shows the fallback and reports error', (code) => {
    const { container } = setup()
    const iframe = iframeOf(container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    fromEmbed(embed, json('onError', { info: code }))
    expect(status()).toBe('error')
    expect(container.querySelector('iframe')).toBeNull()
    const player = container.querySelector('.player--error')
    expect(player?.textContent).toMatch(/unavailable|cannot|can't/i)
    const link = player?.querySelector('a')
    expect(link?.getAttribute('href')).toBe(`https://www.youtube.com/watch?v=${ID}`)
    expect(link?.getAttribute('target')).toBe('_blank')
    expect(link?.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('a Timestamp click after an embed error opens the watch URL at the floored second in a new tab', async () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null)
    const { container } = setup()
    const iframe = iframeOf(container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    fromEmbed(embed, json('onError', { info: 150 }))
    await click(screen.getByRole('button', { name: /Seek to/ }))
    expect(open).toHaveBeenCalledTimes(1)
    expect(open).toHaveBeenCalledWith(`https://www.youtube.com/watch?v=${ID}&t=30s`, '_blank', 'noopener')
    expect(commands()).toEqual([])
  })

  it('a handshake that never gets an answer ends in error after the bounded attempts', async () => {
    const { container } = setup()
    const iframe = iframeOf(container)
    spyOnEmbed(iframe)
    fireEvent.load(iframe)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(9_500)
    })
    expect(handshakes()).toHaveLength(20)
    expect(status()).toBe('loading')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(status()).toBe('error')
    expect(container.querySelector('.player--error a')).not.toBeNull()
    expect(handshakes()).toHaveLength(20)
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe('Player videoId change', () => {
  it('replaces the iframe, resets to loading and discards the queued seek', async () => {
    const view = setup(<Player videoId={ID} />)
    const first = iframeOf(view.container)
    const firstEmbed = spyOnEmbed(first)
    fireEvent.load(first)
    await click(screen.getByRole('button', { name: /Seek to/ }))
    fromEmbed(firstEmbed, json('onReady'))
    expect(status()).toBe('ready')
    posts = []

    // Queue a seek on a not-yet-ready player, then switch video.
    view.rerender(
      <PlayerProvider>
        <Player videoId="aaaaaaaaaaa" />
        <StatusProbe />
        <Timestamp seconds={30.5} />
      </PlayerProvider>,
    )
    const second = iframeOf(view.container)
    expect(second).not.toBe(first)
    expect(second.src).toContain('/embed/aaaaaaaaaaa')
    expect(status()).toBe('loading')
    const secondEmbed = spyOnEmbed(second)
    await click(screen.getByRole('button', { name: /Seek to/ }))
    view.rerender(
      <PlayerProvider>
        <Player videoId="bbbbbbbbbbb" />
        <StatusProbe />
        <Timestamp seconds={30.5} />
      </PlayerProvider>,
    )
    const third = iframeOf(view.container)
    const thirdEmbed = spyOnEmbed(third)
    fireEvent.load(third)
    fromEmbed(secondEmbed, json('onReady'))
    expect(status()).toBe('loading')
    expect(commands()).toEqual([])
    fromEmbed(thirdEmbed, json('onReady'))
    expect(status()).toBe('ready')
    expect(commands()).toEqual([])
  })

  it('a message from the old iframe does not make the new one ready', () => {
    const view = setup(<Player videoId={ID} />)
    const oldEmbed = spyOnEmbed(iframeOf(view.container))
    view.rerender(
      <PlayerProvider>
        <Player videoId="aaaaaaaaaaa" />
        <StatusProbe />
        <Timestamp seconds={30.5} />
      </PlayerProvider>,
    )
    fromEmbed(oldEmbed, json('onReady'))
    expect(status()).toBe('loading')
  })

  it('leaving the error state when the id changes to a valid one', () => {
    const view = setup(<Player videoId="bad" />)
    expect(status()).toBe('error')
    view.rerender(
      <PlayerProvider>
        <Player videoId={ID} />
        <StatusProbe />
        <Timestamp seconds={30.5} />
      </PlayerProvider>,
    )
    expect(status()).toBe('loading')
    expect(view.container.querySelector('iframe')).not.toBeNull()
  })
})

describe('Player unmount', () => {
  it('removes the message listener and the timer, and seekTo posts nothing afterwards', () => {
    const add = vi.spyOn(window, 'addEventListener')
    const remove = vi.spyOn(window, 'removeEventListener')
    let captured: ((s: number) => void) | undefined
    function Capture() {
      captured = usePlayer()?.seekTo
      return null
    }
    const view = render(
      <PlayerProvider>
        <Player videoId={ID} />
        <Capture />
      </PlayerProvider>,
    )
    const iframe = iframeOf(view.container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    const messageAdds = () => add.mock.calls.filter((c) => c[0] === 'message').length
    const messageRemoves = () => remove.mock.calls.filter((c) => c[0] === 'message').length
    expect(messageAdds() - messageRemoves()).toBe(1)

    view.unmount()
    expect(messageAdds() - messageRemoves()).toBe(0)
    expect(vi.getTimerCount()).toBe(0)
    posts = []
    expect(() => captured?.(10)).not.toThrow()
    fromEmbed(embed, json('onReady'))
    vi.advanceTimersByTime(60_000)
    expect(posts).toEqual([])
  })

  it('drops a queued seek on unmount', async () => {
    const view = setup()
    const iframe = iframeOf(view.container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    await click(screen.getByRole('button', { name: /Seek to/ }))
    view.unmount()
    fromEmbed(embed, json('onReady'))
    expect(commands()).toEqual([])
  })
})

describe('Player under StrictMode', () => {
  it('has one active message listener and sends one seekTo per click', async () => {
    const add = vi.spyOn(window, 'addEventListener')
    const remove = vi.spyOn(window, 'removeEventListener')
    const { container } = setup(<Player videoId={ID} />, true)
    const net = () =>
      add.mock.calls.filter((c) => c[0] === 'message').length -
      remove.mock.calls.filter((c) => c[0] === 'message').length
    expect(net()).toBe(1)
    const iframe = iframeOf(container)
    const embed = spyOnEmbed(iframe)
    fireEvent.load(iframe)
    fromEmbed(embed, json('onReady'))
    expect(status()).toBe('ready')
    await click(screen.getByRole('button', { name: /Seek to/ }))
    expect(commands().filter((c) => c.message['func'] === 'seekTo')).toHaveLength(1)
    expect(commands().filter((c) => c.message['func'] === 'playVideo')).toHaveLength(1)
  })
})

describe('Player without a provider', () => {
  it('renders without throwing', () => {
    const { container } = render(<Player videoId={ID} />)
    expect(container.querySelector('iframe')).not.toBeNull()
  })
})
