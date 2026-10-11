import { describe, expect, it } from 'vitest'
import {
  buildSearchRequest,
  excerptFragments,
  isSearchPage,
  MAX_PAGE,
  nextEnabled,
  offsetForPage,
  parseSearchParams,
  searchHref,
  timestampHref,
  validateQuery,
} from './search'

describe('validateQuery', () => {
  it('trims and accepts an ordinary query', () => {
    expect(validateQuery('  cats and dogs \n')).toEqual({ ok: true, q: 'cats and dogs' })
  })

  it.each(['', '   ', '\t\n '])('treats %j as empty', (raw) => {
    expect(validateQuery(raw)).toEqual({ ok: false, reason: 'empty' })
  })

  it('counts 200 ASCII characters as valid and 201 as too long', () => {
    expect(validateQuery('a'.repeat(200)).ok).toBe(true)
    expect(validateQuery('a'.repeat(201))).toEqual({ ok: false, reason: 'too_long' })
  })

  it('counts code points, not UTF-16 units: 200 emoji pass, 201 do not', () => {
    const emoji = '\u{1F600}'
    expect(emoji.length).toBe(2)
    expect(validateQuery(emoji.repeat(200)).ok).toBe(true)
    expect(validateQuery(emoji.repeat(201))).toEqual({ ok: false, reason: 'too_long' })
  })

  it('applies the length limit after trimming', () => {
    expect(validateQuery(` ${'a'.repeat(200)} `).ok).toBe(true)
  })

  it('rejects a NUL anywhere but leaves other control characters to the server', () => {
    expect(validateQuery('a\u0000b')).toEqual({ ok: false, reason: 'nul' })
    expect(validateQuery('\u0000')).toEqual({ ok: false, reason: 'nul' })
    expect(validateQuery('a\u0001b')).toEqual({ ok: true, q: 'a\u0001b' })
  })

  it.each(['the', '-dogs', 'and of'])('sends %j unchanged (server decides it is empty)', (raw) => {
    expect(validateQuery(raw)).toEqual({ ok: true, q: raw })
  })
})

describe('page maths', () => {
  it('computes the offset from the page', () => {
    expect(offsetForPage(1)).toBe(0)
    expect(offsetForPage(2)).toBe(20)
    expect(offsetForPage(51)).toBe(1000)
    expect(MAX_PAGE).toBe(51)
  })

  it('enables Next only when the server has more and the next offset is within the cap', () => {
    expect(nextEnabled(1, true)).toBe(true)
    expect(nextEnabled(1, false)).toBe(false)
    expect(nextEnabled(50, true)).toBe(true)
    expect(nextEnabled(51, true)).toBe(false)
  })
})

describe('parseSearchParams', () => {
  const parse = (s: string) => parseSearchParams(new URLSearchParams(s))

  it('is idle without q, and needs no correction', () => {
    expect(parse('')).toEqual({ q: null, page: 1, corrected: null })
  })

  it.each(['q=', 'q=%20%20', 'q=%09'])('drops an empty or blank q in %s', (s) => {
    const r = parse(s)
    expect(r.q).toBeNull()
    expect(r.corrected?.has('q')).toBe(false)
  })

  it('reads a valid q and page', () => {
    expect(parse('q=cats&page=3')).toEqual({ q: 'cats', page: 3, corrected: null })
  })

  it('trims q and asks for the URL to be corrected', () => {
    const r = parse('q=%20cats%20')
    expect(r.q).toBe('cats')
    expect(r.corrected?.get('q')).toBe('cats')
  })

  it('drops an over-long q, a NUL q, and corrects the URL', () => {
    for (const s of [`q=${'a'.repeat(201)}`, 'q=a%00b']) {
      const r = parse(s)
      expect(r.q).toBeNull()
      expect(r.corrected?.has('q')).toBe(false)
    }
  })

  it('accepts page 51 and rejects page 52', () => {
    expect(parse('q=a&page=51')).toMatchObject({ page: 51, corrected: null })
    const r = parse('q=a&page=52')
    expect(r.page).toBe(1)
    expect(r.corrected?.has('page')).toBe(false)
  })

  it.each(['0', '-1', '1.5', 'abc', '1e3', '', '99999999999999999999', '+2', ' 2', '02', '0x10'])(
    'falls back to page 1 for page=%j',
    (p) => {
      const r = parse(`q=a&page=${encodeURIComponent(p)}`)
      expect(r.page).toBe(1)
      expect(r.corrected?.has('page')).toBe(false)
      expect(r.corrected?.get('q')).toBe('a')
    },
  )

  it('keeps unknown parameters out of the result and leaves them in the corrected URL untouched', () => {
    const r = parse('q=a&page=zzz&evil=MARKER')
    expect(r).not.toHaveProperty('evil')
    expect(r.corrected?.get('evil')).toBe('MARKER')
  })

  it('does not throw on a malformed percent-encoding', () => {
    const r = parse('q=%E0%A4%A')
    expect(() => r).not.toThrow()
    expect(typeof r.q === 'string' || r.q === null).toBe(true)
  })

  it('uses the first q when several are given', () => {
    expect(parse('q=one&q=two').q).toBe('one')
  })
})

describe('request and URL construction', () => {
  it('builds the API query only through URLSearchParams, fixed limit', () => {
    const p = buildSearchRequest('a&limit=50#x', 3)
    expect(p.get('q')).toBe('a&limit=50#x')
    expect(p.get('limit')).toBe('20')
    expect(p.get('offset')).toBe('40')
    expect([...p.keys()].sort()).toEqual(['limit', 'offset', 'q'])
    expect(p.toString()).not.toContain('#')
  })

  it('builds the view URL, omitting page 1', () => {
    expect(searchHref('a b&c', 1)).toBe('q=a+b%26c')
    expect(searchHref('a', 2)).toBe('q=a&page=2')
  })
})

describe('excerptFragments', () => {
  const spans = [
    { text: 'the ', match: false },
    { text: 'zebra', match: true },
    { text: ' ran', match: false },
  ]

  it('keeps spans in order with whitespace intact, and the start', () => {
    expect(excerptFragments([{ spans, start_sec: 12.5 }])).toEqual([{ spans, startSec: 12.5 }])
  })

  it('drops empty spans and excerpts with no text left', () => {
    expect(
      excerptFragments([
        { spans: [{ text: '', match: true }], start_sec: 1 },
        { spans: [], start_sec: 2 },
        {
          spans: [
            { text: '', match: false },
            { text: 'x', match: true },
          ],
          start_sec: null,
        },
      ]),
    ).toEqual([{ spans: [{ text: 'x', match: true }], startSec: null }])
  })

  it('keeps 0 and treats a negative or non-finite start as absent', () => {
    const starts = [0, -1, Number.NaN, Number.POSITIVE_INFINITY, null]
    expect(
      excerptFragments(starts.map((start_sec) => ({ spans, start_sec }))).map((e) => e.startSec),
    ).toEqual([0, null, null, null, null])
  })

  it('returns nothing for no excerpts', () => {
    expect(excerptFragments([])).toEqual([])
  })
})

describe('timestampHref', () => {
  it.each([
    [12.5, '/videos/abc?t=12'],
    [0, '/videos/abc?t=0'],
    [3723, '/videos/abc?t=3723'],
    [0.999, '/videos/abc?t=0'],
    [1_000_000_000, '/videos/abc?t=1000000000'],
  ])('%s -> %s', (sec, href) => {
    expect(timestampHref('abc', sec)).toBe(href)
  })

  it.each([null, -0.5, Number.NaN, Number.POSITIVE_INFINITY, 1_000_000_001, 1e21])(
    '%s -> no link',
    (sec) => {
      expect(timestampHref('abc', sec)).toBeNull()
    },
  )

  it('encodes the video id', () => {
    expect(timestampHref('a/b c', 1)).toBe('/videos/a%2Fb%20c?t=1')
  })
})

describe('isSearchPage', () => {
  const ok = {
    results: [
      {
        video_id: 'abcdefghijk',
        title: null,
        channel_id: null,
        channel_title: null,
        published_at: null,
        duration_sec: null,
        unavailable: null,
        transcript_source: 'whisper',
        excerpts: [{ spans: [{ text: 'a', match: true }], start_sec: 3 }],
      },
    ],
    limit: 20,
    offset: 0,
    has_more: false,
  }

  it('accepts a well-formed page', () => {
    expect(isSearchPage(ok)).toBe(true)
    expect(isSearchPage({ ...ok, results: [] })).toBe(true)
  })

  it.each([
    ['null', null],
    ['string', 'x'],
    ['array', []],
    ['no results', { ...ok, results: undefined }],
    ['results not array', { ...ok, results: {} }],
    ['has_more string', { ...ok, has_more: 'no' }],
    ['offset missing', { ...ok, offset: undefined }],
    ['result null', { ...ok, results: [null] }],
    ['video_id number', { ...ok, results: [{ ...ok.results[0], video_id: 5 }] }],
    ['title number', { ...ok, results: [{ ...ok.results[0], title: 5 }] }],
    ['excerpts missing', { ...ok, results: [{ ...ok.results[0], excerpts: undefined }] }],
    ['excerpt not an object', { ...ok, results: [{ ...ok.results[0], excerpts: ['x'] }] }],
    ['start_sec a string', { ...ok, results: [{ ...ok.results[0], excerpts: [{ spans: [], start_sec: '1' }] }] }],
    ['start_sec missing', { ...ok, results: [{ ...ok.results[0], excerpts: [{ spans: [] }] }] }],
    ['span text number', { ...ok, results: [{ ...ok.results[0], excerpts: [{ spans: [{ text: 1, match: true }], start_sec: null }] }] }],
    ['span match string', { ...ok, results: [{ ...ok.results[0], excerpts: [{ spans: [{ text: 'a', match: 'y' }], start_sec: null }] }] }],
  ])('rejects %s', (_name, body) => {
    expect(isSearchPage(body)).toBe(false)
  })
})
