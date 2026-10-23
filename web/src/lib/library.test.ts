import { describe, expect, it } from 'vitest'
import type { components } from '../api'
import {
  addOneDay,
  buildVideosQuery,
  isValidDate,
  PAGE_SIZE,
  parseLibraryParams,
  publishedDate,
  statusLabel,
  toSearchParams,
  type LibraryState,
} from './library'

type VideoItem = components['schemas']['VideoItem']

function item(over: Partial<VideoItem>): VideoItem {
  return {
    video_id: 'aaaaaaaaaaa',
    title: 't',
    channel_id: null,
    channel_title: null,
    published_at: null,
    duration_sec: null,
    latest_analysis_at: null,
    has_transcript: false,
    origin: 'manual',
    unavailable: null,
    status: 'idle',
    active_job: null,
    last_failure: null,
    ...over,
  }
}
const job = (kind: string, state: string) => ({ id: 1, kind, state })
const failure = (kind: string, error_class: string | null) => ({
  job_id: 1,
  kind,
  error_class,
  finished_at: null,
})

describe('statusLabel', () => {
  it.each<[string, Partial<VideoItem>, string, string | null]>([
    ['done', { status: 'done' }, 'Analysed', null],
    ['done + job', { status: 'done', active_job: job('analyze', 'pending') }, 'Analysed · re-analysing', null],
    ['queued ingest', { status: 'processing', active_job: job('ingest', 'pending') }, 'Queued: download', null],
    ['queued transcribe', { status: 'processing', active_job: job('transcribe', 'pending') }, 'Queued: transcription', null],
    ['queued analyze', { status: 'processing', active_job: job('analyze', 'pending') }, 'Queued: analysis', null],
    ['running ingest', { status: 'processing', active_job: job('ingest', 'running') }, 'Download in progress', null],
    ['running transcribe', { status: 'processing', active_job: job('transcribe', 'running') }, 'Transcription in progress', null],
    ['running analyze', { status: 'processing', active_job: job('analyze', 'running') }, 'Analysis in progress', null],
    ['failed ingest', { status: 'failed', last_failure: failure('ingest', 'TOOL_FAILURE') }, 'Failed at download', 'TOOL_FAILURE'],
    ['failed transcribe', { status: 'failed', last_failure: failure('transcribe', 'X') }, 'Failed at transcription', 'X'],
    ['failed analyze', { status: 'failed', last_failure: failure('analyze', 'Y') }, 'Failed at analysis', 'Y'],
    ['unavailable', { status: 'unavailable' }, 'Unavailable on YouTube', null],
    ['idle', { status: 'idle' }, 'Not processed', null],
  ])('%s', (_n, over, label, detail) => {
    expect(statusLabel(item(over))).toEqual({ label, detail })
  })

  it.each<[string, Partial<VideoItem>, string, string | null]>([
    ['unknown job kind (pending)', { status: 'processing', active_job: job('mystery', 'pending') }, 'Processing', null],
    ['unknown job kind (running)', { status: 'processing', active_job: job('mystery', 'running') }, 'Processing', null],
    ['unknown job state', { status: 'processing', active_job: job('ingest', 'weird') }, 'Processing', null],
    ['processing without job', { status: 'processing' }, 'Processing', null],
    ['failed without last_failure', { status: 'failed' }, 'Failed', null],
    ['failed with null error_class', { status: 'failed', last_failure: failure('ingest', null) }, 'Failed at download', null],
    ['failed with unknown kind and class', { status: 'failed', last_failure: failure('zzz', 'E') }, 'Failed', 'E'],
    ['unknown status', { status: 'bogus' as VideoItem['status'] }, 'Unknown status', null],
    ['prototype-ish status', { status: '__proto__' as VideoItem['status'] }, 'Unknown status', null],
  ])('fallback: %s', (_n, over, label, detail) => {
    expect(statusLabel(item(over))).toEqual({ label, detail })
  })
})

describe('dates', () => {
  it.each(['2026-09-01', '2024-02-29', '1999-12-31'])('accepts %s', (d) => {
    expect(isValidDate(d)).toBe(true)
  })
  it.each([
    '2026-02-30', '2026-13-01', '2025-02-29', '2026-00-10', '2026-01-00', '2026-1-1',
    '', ' 2026-09-01', '2026-09-01 ', '2026-09-01\n', '20260-01-01', 'abc', '0000-01-01', '２０２６-09-01',
  ])('rejects %j', (d) => {
    expect(isValidDate(d)).toBe(false)
  })

  it.each([
    ['2026-09-01', '2026-09-02'],
    ['2026-12-31', '2027-01-01'],
    ['2026-01-31', '2026-02-01'],
    ['2026-02-28', '2026-03-01'],
    ['2024-02-28', '2024-02-29'],
    ['2024-02-29', '2024-03-01'],
    ['2026-04-30', '2026-05-01'],
  ])('addOneDay(%s) = %s', (d, next) => {
    expect(addOneDay(d)).toBe(next)
  })
  it('addOneDay has no successor past year 9999', () => {
    expect(addOneDay('9999-12-31')).toBeNull()
  })

  it('shows publication dates as UTC days regardless of the local zone (TZ is LA)', () => {
    expect(publishedDate('2026-09-01T02:00:00Z')).toBe('2026-09-01')
    expect(publishedDate('2026-09-01T23:59:59Z')).toBe('2026-09-01')
    expect(publishedDate('2026-09-01T00:00:00+05:00')).toBe('2026-08-31')
    expect(publishedDate(null)).toBeNull()
    expect(publishedDate('not a date')).toBeNull()
  })
})

const UC = 'UC' + 'a'.repeat(22)
const parse = (qs: string) => parseLibraryParams(new URLSearchParams(qs))

describe('parseLibraryParams', () => {
  it('accepts two-, three-, and four-key sort sequences', () => {
    expect(parse('sort=title&order=asc&sort=status&order=desc').state.sorts).toEqual([
      { sort: 'title', order: 'asc' },
      { sort: 'status', order: 'desc' },
    ])
    expect(
      parse('sort=channel&order=asc&sort=published&order=desc&sort=status&order=asc').state.sorts,
    ).toEqual([
      { sort: 'channel', order: 'asc' },
      { sort: 'published', order: 'desc' },
      { sort: 'status', order: 'asc' },
    ])
    expect(
      parse('sort=channel&order=asc&sort=published&order=desc&sort=status&order=asc&sort=title&order=desc')
        .state.sorts,
    ).toHaveLength(4)
  })

  it('accepts a full valid set with ordered multi-sort pairs', () => {
    const r = parse(`channel=${UC}&from=2026-01-02&to=2026-03-04&status=failed&sort=title&order=asc&sort=status&order=desc&page=3`)
    expect(r.state).toEqual({
      channel: UC,
      from: '2026-01-02',
      to: '2026-03-04',
      status: 'failed',
      sorts: [
        { sort: 'title', order: 'asc' },
        { sort: 'status', order: 'desc' },
      ],
      page: 3,
    })
    expect(r.invalid).toEqual([])
  })
  it('defaults when empty', () => {
    expect(parse('').state).toEqual({
      channel: null,
      from: null,
      to: null,
      status: null,
      sorts: [{ sort: 'published', order: 'desc' }],
      page: 1,
    })
  })
  it.each(['done', 'processing', 'failed', 'unavailable', 'idle'])('accepts status %s', (s) => {
    expect(parse(`status=${s}`).state.status).toBe(s)
  })
  it.each([
    ['sort=title', [{ sort: 'title', order: 'desc' }]],
    ['order=asc', [{ sort: 'published', order: 'asc' }]],
    ['sort=status&order=asc', [{ sort: 'status', order: 'asc' }]],
  ])('applies scalar compatibility for %s', (query, sorts) => {
    expect(parse(query).state.sorts).toEqual(sorts)
  })
  it.each([
    'sort=',
    'sort=TITLE',
    'sort=nope',
    'order=',
    'order=ASC',
    'order=nope',
    'sort=title&sort=channel',
    'order=asc&order=desc',
    'sort=title&sort=title&order=asc&order=desc',
    'sort=title&sort=status&order=asc',
    'sort=title&sort=status&sort=channel&sort=published&sort=title&order=asc&order=asc&order=asc&order=asc&order=asc',
  ])('drops the whole invalid sort sequence in %s', (query) => {
    const r = parse(query)
    expect(r.state.sorts).toEqual([{ sort: 'published', order: 'desc' }])
    expect(r.invalid).toEqual(['sort', 'order'])
  })

  it.each([
    'UC' + 'a'.repeat(21), 'UC' + 'a'.repeat(23), 'XX' + 'a'.repeat(22), 'uc' + 'a'.repeat(22),
    'UC' + 'a'.repeat(21) + '!', 'UC' + 'a'.repeat(22) + '\n', '', 'UC' + 'é'.repeat(22),
    'X' + UC, '<img src=x>' + UC, ' ' + UC, '\n' + UC,
  ])('drops channel %j', (c) => {
    const r = parse(`channel=${encodeURIComponent(c)}`)
    expect(r.state.channel).toBeNull()
    expect(r.invalid).toEqual(['channel'])
  })

  it.each(['2026-02-30', '2026-13-01', 'MARKER', '2026-9-1', '<script>'])('drops from/to %j', (d) => {
    const r = parse(`from=${encodeURIComponent(d)}&to=${encodeURIComponent(d)}`)
    expect(r.state.from).toBeNull()
    expect(r.state.to).toBeNull()
    expect(r.invalid).toEqual(['from', 'to'])
  })

  it.each(['Done', 'DONE', 'all', 'MARKER', '', 'done,failed', '__proto__', 'constructor', 'toString'])(
    'drops status %j',
    (s) => {
      const r = parse(`status=${encodeURIComponent(s)}`)
      expect(r.state.status).toBeNull()
      expect(r.invalid).toEqual(['status'])
    },
  )

  it.each(['0', '-1', '1.5', 'abc', '1e3', '0x10', ' 2', '2 ', '+2', '01', '', '20002', '99999999999999999999', 'Infinity', 'NaN'])(
    'falls back to page 1 for page=%j',
    (p) => {
      const r = parse(`page=${encodeURIComponent(p)}`)
      expect(r.state.page).toBe(1)
      expect(r.invalid).toEqual(['page'])
    },
  )

  it('applies the offset cap exactly: page 20001 (offset 1_000_000) ok, 20002 not', () => {
    expect(parse('page=20001').state.page).toBe(20001)
    expect(parse('page=20002').state.page).toBe(1)
  })

  it('treats a repeated parameter as invalid', () => {
    const r = parse(`page=2&page=3`)
    expect(r.state.page).toBe(1)
    expect(r.invalid).toEqual(['page'])
  })

  it('ignores unknown parameters without reporting them', () => {
    const r = parse('q=MARKER&before=MARKER&limit=1000&offset=5')
    expect(r.state).toEqual({
      channel: null,
      from: null,
      to: null,
      status: null,
      sorts: [{ sort: 'published', order: 'desc' }],
      page: 1,
    })
    expect(r.invalid).toEqual([])
  })
})

describe('buildVideosQuery', () => {
  const base: LibraryState = {
    channel: null,
    from: null,
    to: null,
    status: null,
    sorts: [{ sort: 'published', order: 'desc' }],
    page: 1,
  }
  const q = (s: Partial<LibraryState>) => buildVideosQuery({ ...base, ...s })

  it('sends only limit and offset when nothing is set', () => {
    expect(q({}).toString()).toBe('limit=50&offset=0')
    expect(PAGE_SIZE).toBe(50)
  })
  it('omits the default sort and preserves repeated pair order', () => {
    expect(q({}).toString()).toBe('limit=50&offset=0')
    expect(
      q({ sorts: [{ sort: 'channel', order: 'asc' }, { sort: 'published', order: 'desc' }] }).toString(),
    ).toBe('sort=channel&sort=published&order=asc&order=desc&limit=50&offset=0')
  })
  it('computes offset from the page', () => {
    expect(q({ page: 2 }).get('offset')).toBe('50')
    expect(q({ page: 5 }).get('offset')).toBe('200')
    expect(q({ page: 20001 }).get('offset')).toBe('1000000')
  })
  it('From=To sends an exclusive next-day upper bound', () => {
    const p = q({ from: '2026-09-01', to: '2026-09-01' })
    expect(p.get('published_after')).toBe('2026-09-01')
    expect(p.get('published_before')).toBe('2026-09-02')
  })
  it.each([
    ['2026-12-31', '2027-01-01'],
    ['2026-01-31', '2026-02-01'],
    ['2024-02-29', '2024-03-01'],
  ])('To=%s sends published_before=%s', (to, before) => {
    const p = q({ to })
    expect(p.get('published_before')).toBe(before)
    expect(p.has('published_after')).toBe(false)
  })
  it('omits published_before for the last representable day', () => {
    expect(q({ to: '9999-12-31' }).has('published_before')).toBe(false)
  })
  it('sends channel and status, and combines all filters', () => {
    const p = q({ channel: UC, status: 'idle', from: '2026-01-01', to: '2026-01-02', page: 2 })
    expect(Object.fromEntries(p)).toEqual({
      channel: UC,
      status: 'idle',
      published_after: '2026-01-01',
      published_before: '2026-01-03',
      limit: '50',
      offset: '50',
    })
  })
})

describe('toSearchParams', () => {
  it('omits unset values, page 1, and the default sort pair', () => {
    expect(
      toSearchParams({
        channel: null,
        from: null,
        to: null,
        status: null,
        sorts: [{ sort: 'published', order: 'desc' }],
        page: 1,
      }).toString(),
    ).toBe('')
    expect(
      toSearchParams({
        channel: UC,
        from: '2026-01-01',
        to: '2026-02-02',
        status: 'done',
        sorts: [
          { sort: 'channel', order: 'asc' },
          { sort: 'status', order: 'desc' },
        ],
        page: 4,
      }).toString(),
    ).toBe(`channel=${UC}&from=2026-01-01&to=2026-02-02&status=done&sort=channel&order=asc&sort=status&order=desc&page=4`)
  })
})
