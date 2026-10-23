# Design System

**Status:** the visual direction is still open. The SPA ships deliberately
unstyled, and the owner has not chosen a look yet (#124: options A editorial,
B dense tool, C hybrid, D media-first). This document records what the code
does today (audited 2026-10-11), the conventions every view already follows,
and the constraints that whoever styles the app must respect. It does **not**
define colour, type or spacing tokens for the SPA, because none exist yet and
inventing them is #124's job.

Read this before touching anything in `web/src/`. If you are styling a view
(#120 Library, #130 Transcript, #131 Search, #132 Ops) and #124 is not done
yet, stop: those issues are blocked on it.

---

## 1. Ground rules for UI work right now

1. **Do not add visual styling to a view.** No stylesheet, no `className`
   with rules, no `style={{…}}` beyond the layout-safety cases listed in §3.
   Each view's header comment says this ("Plain semantic HTML on purpose:
   styling is #1xx").
2. **Do not invent tokens.** No colour, font, spacing or radius value enters
   `web/src/` until #124 fills in §11 of this document.
3. **Do not add a CSS framework, component library, icon set or font
   package** without asking the owner (AGENTS.md -> Rules). Tailwind,
   shadcn, MUI and so on are all new npm dependencies. Vite can import plain
   `.css` files without any new dependency.
4. **Keep the semantics.** Tests find elements by role, label and text
   (`getByRole('button', { name: 'Retry' })`, `getByRole('navigation', { name: 'Pagination' })`).
   Styling must not replace a `<table>`, `<label>`, `<nav>`, `role` or
   `aria-live` region. Every pattern in §6 and §7 must survive styling.
5. **Everything from the API is a React text child.** No
   `dangerouslySetInnerHTML`, `innerHTML` or `insertAdjacentHTML`. Source-scan
   tests enforce this (`routes/*.source.test.ts`).

---

## 2. Inventory: what styling exists today

| Where | What | Status |
|---|---|---|
| `web/` | Tailwind config, PostCSS config, `*.css`, CSS modules, CSS-in-JS | **None.** No `tailwind.config.*`, no `index.css`, no styling dependency in `package.json` |
| `web/index.html` | `<style>`, font links, `color-scheme` meta, theme-color | **None.** Only `charset`, `viewport` (`width=device-width, initial-scale=1.0`) and `<title>ytdigest</title>` |
| `web/src/**/*.tsx` | `className` | 4 class names, **no CSS behind any of them** (§3) |
| `web/src/**/*.tsx` | inline `style` | 4 occurrences, all layout safety (§3) |
| `services/api/templates/analysis.html` | One inline `<style>` | **The only real visual styling in the repo**: the server-rendered printable page (§4) |
| ESLint / Stylelint | design-related lint rules | **None.** No ESLint or Stylelint config exists; `tsc --noEmit` is the only static check |
| Design tokens / CSS custom properties | `--*` variables | **None** |
| Dark mode | `prefers-color-scheme`, theme toggle | **None** (§5) |

So the SPA renders with the browser's user-agent stylesheet: default serif
font (Times New Roman in most browsers) at 16px, black text on white, blue and
purple underlined links, native form controls, UA margins on headings,
paragraphs and lists, and no table borders.

---

## 3. Inline styles and class hooks in the SPA

### Inline styles (the only allowed kind, until #124)

Each one stops content from breaking the layout, not decoration. Keep them
when styling arrives, or move them into the stylesheet unchanged.

| File | Element | Style | Why |
|---|---|---|---|
| `components/AnalysisView.tsx` | TL;DR `<p>` | `whiteSpace: 'pre-line'` | TL;DR text carries line breaks from the LLM |
| `routes/Compare.tsx` | TL;DR `<p>` in each column | `whiteSpace: 'pre-line'` | Same, in the side-by-side view |
| `components/Player.tsx` | YouTube `<iframe>` | `width: '100%', aspectRatio: '16 / 9'` | Fluid embed at the video's aspect ratio |
| `routes/Ops.tsx` | expanded `last_error` `<pre>` | `whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxWidth: '100%', overflow: 'auto'` | 8 KiB tracebacks with long tokens must not widen the table |
| `components/JobProgress.tsx` | the visually hidden `<span aria-live="polite">` | `position: 'absolute', width: '1px', height: '1px', overflow: 'hidden', clip: 'rect(0 0 0 0)', whiteSpace: 'nowrap'` | The standard screen-reader-only clip: the visible percent and ETA change every poll, only this copy is announced, at most once a minute |

Inline styles are allowed by the production CSP (`style-src 'self' 'unsafe-inline'`).

### Class names (hooks with no CSS yet)

| Class | Element | File |
|---|---|---|
| `player` | wrapper `<div>` of the embed | `components/Player.tsx` |
| `player player--error` | wrapper when the video cannot play ("Video unavailable", or "This video cannot be played here." + "Watch on YouTube") | `components/Player.tsx` |
| `timestamp` | the seek `<button>` | `components/Timestamp.tsx` |
| `speaker` | speaker label `<span>` in a transcript segment | `routes/Transcript.tsx` |

The `player--error` modifier shows a BEM-style naming habit
(`block--modifier`). It is the only precedent, not a decided convention: #124
picks the CSS approach.

---

## 4. The printable page: the only palette and type in use

`GET /api/videos/{id}/render` (`services/api/render.py`, template
`services/api/templates/analysis.html`) is a self-contained HTML page with one
inline `<style>`, no script and no external resources. The SPA links to it as
"Printable version" from the video detail view. It is the only place in the
repo with chosen colours and type. It is **not** the SPA design system, but
it is the one existing precedent #124 can adopt or reject.

### Colours

| Use | Value | Contrast on `#fff` |
|---|---|---|
| Body text | `#1a1a1a` | 17.4:1 |
| Background | `#fff` | — |
| Secondary text (`.meta`, `.empty`, `.who`, `.confidence`, footer) | `#555` | 7.46:1 (AA and AAA) |
| Links | `#0b57d0` | 6.39:1 (AA) |
| Rules (`h2` underline, footer top border) | `#ccc` | 1.61:1 (decorative only, carries no meaning) |

In print (`@media print`), links take the text colour and lose their underline.

### Type

| Element | Size | Other |
|---|---|---|
| `body` | browser default (16px) | `font-family: system-ui, sans-serif; line-height: 1.5` |
| `h1` | `1.6rem` | margin `0 0 0.25rem` |
| `h2` | `1.2rem` | margin `1.5rem 0 0.5rem`, `border-bottom: 1px solid #ccc` |
| `.who`, `.confidence` | `0.9rem` | `#555` |
| `footer` | `0.85rem` | `#555`, `margin-top: 2rem`, `padding-top: 0.5rem`, top border |
| `.empty` | inherited | `#555`, italic |

### Layout

- Single column: `max-width: 48rem; margin: 0 auto; padding: 1rem`
- Lists: `padding-left: 1.25rem; margin: 0`; items `margin: 0 0 0.5rem`
- `.sub` (secondary line under an item): `margin: 0.1rem 0 0`
- Units are `rem` throughout, no `px` font sizes

---

## 5. Colour, typography, spacing and dark mode in the SPA

- **Colour:** none set. UA defaults only. No meaning is carried by colour
  anywhere: statuses, errors, matches and states are all text (§8).
- **Typography:** none set. The UA serif font, default sizes. The heading
  hierarchy is the one thing that is fixed (§7.1).
- **Spacing:** none set. Inline items are separated by literal spaces
  (`{' '}`) and the ` · ` separator (§7.10), not by margin or gap. There is no
  grid, flex or gap anywhere in the SPA.
- **Layout:** normal document flow. The only sized element is the player
  (100% wide, 16:9). There is no max width, so lines run the full window width.
- **Responsive:** no media queries. Wide tables (Ops jobs: 10 columns; Compare:
  3 columns) can overflow on a phone. The styling issues own this.
- **Dark mode:** none. There is no `color-scheme` meta or CSS property, so the
  page renders light even when the OS is dark. The printable page hard-codes
  `background: #fff`.

---

## 6. Components

### Shared components (`web/src/components/`)

| Component | What it renders | Rules |
|---|---|---|
| `Timestamp` | `<button type="button" class="timestamp" aria-label="Seek to M:SS"><time datetime="PT…">M:SS</time></button>` inside a `PlayerProvider`; a bare `<time>` outside one; nothing for a null, negative or non-finite time | **The only seek path in the app.** Only this file may call `seekTo(` (enforced by `components/sourceScan.test.ts`). Format is `M:SS` under an hour and `H:MM:SS` from one hour, matching the server's `format_timestamp` |
| `Player` | `<div class="player"><iframe …></div>`; error states in `player--error` | The only YouTube integration. Only `components/Player.tsx` and `components/player/` may mention `youtube-nocookie.com` or `postMessage` (source scan). iframe `title` is `Video: {title}` or `YouTube video player`. If the embed fails, a seek opens `youtube.com/watch` in a new tab |
| `PlayerProvider` / `usePlayer()` | context only | One provider per page that has a player. `usePlayer()` returns `null` outside one, and callers fall back instead of throwing |
| `AnalysisView` | TL;DR, Speakers, Topics, Claims, Quotes sections, then a `<footer>` with model, prompt version and `<time>` | Used by VideoDetail and by Compare's one-run fallback. Empty sections say "No topics extracted." (and similar), they are never omitted, except Speakers, which is omitted when the roster is empty |
| `ClaimItem`, `QuoteItem`, `TopicItem` | one `<li>` each | The one copy of item markup, shared by AnalysisView and Compare table cells. Speaker `unknown` displays as **"Unattributed"** (same word as the server renderer). Quotes use `<q>`; topic titles use `<strong>`; every time goes through `<Timestamp>` |

### Route-local components (not exported; reuse by moving them, not copying)

| Component | File | Notes |
|---|---|---|
| `Shell` | `routes/Shell.tsx` | `<header><nav aria-label="Main">` + `<main><Outlet/></main>` + `ScrollRestoration` |
| `Section` | `AnalysisView.tsx` **and** `Compare.tsx` | Two copies: `<section aria-labelledby><h2 id>`. AnalysisView derives the id from the title; Compare uses `useId()` |
| `NotFound` / `VideoNotFound` | `NotFound.tsx`, `VideoDetail.tsx`, `Compare.tsx`, `Transcript.tsx` | Four variants (§10) |
| `SearchForm`, `ResultItem`, `TimestampLink` | `Search.tsx` | `TimestampLink` is a `<Link aria-label="Open at M:SS"><time>`, the link form of a timestamp for pages with no player (#129 will take it over) |
| `Picker`, `Notices`, `SideBySide`, `Aligned`, `Details` | `Compare.tsx` | Two-column comparison tables |
| `Filters`, `JobRow` | `Ops.tsx` | Filter selects; row with inline retry confirmation |
| `JobProgress` | `components/JobProgress.tsx` | Ops "Progress" column and the video page (#245). Transcribe jobs only: running shows `NN%`, a native `<progress>`, "Elapsed …" and "about N min left" (plus "(rough)" when the estimate rests on the default RTF); pending shows "estimated about N min"; every other kind and state renders nothing. If `updated_at` is 90 s old or more while running, the ETA is replaced by `<strong role="status">no update for N min</strong>` (bold text: no warning colour exists until #124) |
| `Segments` | `Transcript.tsx` | Numbered `<ol start>` of segments with pagination |

### Native elements, and how they are used

| Element | Convention |
|---|---|
| `<button>` | Always an explicit `type`: `"button"` everywhere except the search form's `type="submit"`. **No variants** (primary, secondary, danger, icon) exist. Disabled state is the native `disabled` attribute only |
| `<Link>` (react-router) | Every in-app navigation |
| `<a>` | Only for URLs the router must not own: external YouTube links and `/api/...`. External and printable links use `target="_blank" rel="noopener noreferrer"`. The shell's "API health" link opens in the same tab |
| `<input>` | `type="date"` (Library From/To) and `type="search"` (Search). No text-input styling or placeholders: every input has a visible `<label>` |
| `<select>` | Library status, Ops filters, Compare pickers. "Any" is the empty-value first option. Compare disables (does not hide) the run already picked on the other side |
| `<table>` | Every column header is `<th scope="col">`, and row headers are `<th scope="row">` (Ops job id, depth kind, Compare field and time window) |
| `<ol>` / `<ul>` | `<ol>` where order matters (topics, search results, transcript segments with `start`); `<ul>` otherwise |
| `<time>` | Every timestamp and date-time, with a machine `dateTime` where one is available |
| `<mark>` | Matched search terms. Browser default yellow highlight |
| `<q>`, `<strong>`, `<small>`, `<code>`, `<pre>` | Quotes; topic titles and the active channel name; Ops hints and "finished"/"created"; the unknown path on the 404 page; the expanded traceback |

**What does not exist:** modals or dialogs, cards, badges or pills, chips,
icons (no icon set, no SVG, no emoji), spinners or skeletons, toasts,
tooltips, tabs, accordions (beyond one `aria-expanded` toggle), thumbnails or
any `<img>`, and an app footer.

---

## 7. Patterns

### 7.1 Page structure and headings

- Every route renders inside `Shell`'s `<main>`. One `<h1>` per page: the view
  name ("Library", "Search", "Ops") or the video title.
- Major blocks are `<section aria-labelledby={h2 id}>` with an `<h2>`. Search
  result titles are `<h3>`.
- Video pages (VideoDetail, Compare) use `<article>`; the others use `<section>`.
- `document.title` is set per page: the video title, `{title} – Transcript`,
  `Compare: {title}`, `Page not found`, `Video not found`. Library, Search and
  Ops leave it as `ytdigest`.

### 7.2 Navigation

- Main nav: Library, Search, Ops, then "API health" (a plain `<a>` to
  `/api/healthz`), separated by spaces. Plain `Link`, not `NavLink`, so no
  link is marked as the current page (§9).
- Video detail links row: "Open on YouTube · Transcript · Compare analyses ·
  Printable version". Each one after the first appears only when it applies:
  a transcript exists; 2 or more analyses exist; an analysis exists.
- Sub-pages link back with text links: "Back to video" (Transcript), "Video
  details" (Compare), "Back to Library" (404 pages).

### 7.3 URL is the state

Filters, page numbers, the search query and the Compare selection live **only**
in the URL search params. Invalid params are dropped with
`setParams(…, { replace: true })`, and the render ignores them, so nothing
invalid is ever requested. Changing any filter resets to page 1. TanStack
Query owns server state; there is no global client-state store.

### 7.4 Forms

- Labels: either the `<label>` wraps the control (Library), or
  `<label htmlFor={useId()}>` (Search, Ops, Compare). Never placeholder-only.
- **Filters apply on change, with no submit button**: the `<form>` has
  `onSubmit={e => e.preventDefault()}` (Library, Ops). Search is the one form
  that submits.
- Validation errors (Search): the input gets `aria-invalid` and
  `aria-describedby` pointing at a `<p role="alert">` below the form, e.g.
  "Search terms are limited to 200 characters". Library's reversed date range
  shows `<p role="alert">'From' is after 'To'</p>` and disables paging.
- "Clear all filters" / "Reset filters" appear only when a filter is active.
  The active channel filter is a `role="group" aria-label="Active filters"`
  with the channel name in `<strong>` and a "Clear" button.

### 7.5 Loading, empty and error states

| State | Pattern | Copy examples |
|---|---|---|
| First load | `<p role="status">` text ending in `…` (U+2026) | "Loading…", "Loading video…", "Loading transcript…", "Loading analyses…" |
| Page switch | the previous page stays visible (TanStack `placeholderData`); a "Loading…" line appears; Previous/Next are `disabled` | — |
| Empty | plain `<p>`, distinct wording for "nothing exists" and "nothing matches" | "Nothing has been processed yet" / "No videos match these filters"; "No dead jobs" |
| Past the last page | text plus a link back | "No videos on this page" + "Go to the last page"; Search: "Go to page 1" |
| Error | `role="alert"` container with a short message and a **Retry** button | "Could not load videos", "Could not load the transcript" |
| Database down (HTTP 503) | its own message | "The database is unavailable. Try again shortly." |
| 404 | a "not found" page with a link back | "Video not found" + "Back to Library" |
| Route crash | `RouteError`: "Something went wrong" + "Back to Library" | The thrown error is never read or shown |

Rules: raw server error text is never shown to the user (Ops shows
`last_error` because it is a diagnostic screen). No retry is offered for a
4xx (`queryClient.ts` also does not auto-retry a 4xx; other errors retry up to
2 times). Search hides Retry for a 422.

### 7.6 Pagination

- `<nav aria-label="Pagination">` containing "Previous" and "Next" buttons
  (Ops uses "Older" and "Newest" for its cursor).
- Range text in a live region: `51–100 of 234` (en dash, U+2013); Transcript
  uses `Segments 1–100 of 1 234` with spaces as thousands separators
  (`formatCount`) and also shows "Page N of M" between the buttons.
- Buttons are `disabled` at the ends and while the next page loads.
- After a page change, focus moves to the results `<h2 tabIndex={-1}>`
  (Search "Results", Transcript "Segments").

### 7.7 Confirmation of costly actions (Ops retry)

Inline, inside the table row, never a modal: "Retry" becomes a sentence ("Retry
job #12 (transcribe) for {video}? A transcription can take hours of CPU.") with
"Confirm retry" and "Cancel". Focus moves to "Confirm retry" when it opens and
back to "Retry" on Cancel. Both are `disabled` while the request runs. The
outcome is announced in an `aria-live="polite"` span in the row, or in a
section-level live region if the row has disappeared.

### 7.8 Disclosure

Ops `last_error`: the first line, cut to 200 characters with `…`, plus a
`<button aria-expanded>` that switches between "Show full error" and "Hide".
The expanded text is a `<pre>` (§3).

### 7.9 Player and seeking

A page that seeks (Compare; the route wraps it in `PlayerProvider`) renders one
`<Player>` near the top, and every time inside it is a `<Timestamp>` button. On
a page without a provider, the same component renders plain `<time>` text.
Search uses `TimestampLink` instead (opens the detail view). The embed host is
`youtube-nocookie.com`, and no YouTube script is loaded (architecture §9.1).

### 7.10 Text conventions

- **Separator:** ` · ` (middle dot, U+00B7, with spaces) between metadata
  items: "Channel · 2026-09-01 · 12:34", "Model: x · Prompt version: y".
- **Ellipsis:** `…` (U+2026), never three dots.
- **Ranges:** en dash `–` (U+2013). **Missing value in a table cell:** em dash `—`.
- **Dates:** always UTC, never the viewer's locale: `YYYY-MM-DD` for publish
  dates, `YYYY-MM-DD HH:MM UTC` (Ops, Compare pickers),
  `YYYY-MM-DD HH:MM:SS UTC` (Compare details). No `Intl`, no relative times.
- **Durations:** `M:SS` / `H:MM:SS` for media times; `1 h 2 min 3 s` for run
  durations; cost `$0.0123` (4 decimals); absent values say "not recorded".
- **Fallbacks:** "Unknown channel", "Date unknown", "Language unknown",
  "Unattributed", the `video_id` when there is no title.
- **Spelling and case:** British ("Analysed", "Analysing"); sentence case for
  headings and buttons ("Clear all filters", "Swap A and B"); short status
  lines have no full stop, sentences do.

---

## 8. Accessibility patterns in use

| Pattern | Where |
|---|---|
| Landmarks: `header`, `nav aria-label="Main"`, `main`, `nav aria-label="Pagination"`, `form role="search"` | Shell, list views, Search |
| `section aria-labelledby` → `h2` | AnalysisView, Compare, Ops |
| `role="status"` / `aria-live="polite"` for loading text, result counts and retry outcomes | all views |
| `role="alert"` for errors and validation | all views (exceptions in §10) |
| `aria-invalid` + `aria-describedby` on an invalid input | Search |
| `aria-expanded` on a disclosure button | Ops |
| `aria-label` naming icon-less controls that show only a time: "Seek to 1:02", "Open at 1:02" (visible text is part of the name) | Timestamp, Search |
| `th scope="col"` / `scope="row"` on every table | Library, Compare, Ops |
| Programmatic focus: results heading after paging; confirm button in and out | Search, Transcript, Ops |
| Native `disabled` rather than `aria-disabled` | all buttons |
| Descriptive iframe `title` | Player |
| `<html lang="en">` | `index.html` |
| Nothing relies on colour; status, matches and errors are all text | everywhere (trivially, since there is no colour) |

Contrast, focus indicators and target sizes are the browser defaults today, so
they cannot be judged until styling exists. The bar the styling issues already
set (from #120's acceptance criteria): body and status text 4.5:1 (WCAG AA),
meaningful borders and fills 3:1, visible focus at 3:1, touch targets at least
44×44px, no horizontal page scroll at 360px wide or at 200% zoom on 1280px,
any animation stopped under `prefers-reduced-motion: reduce`, and no state
told apart by colour alone. Check in both themes if #124 defines two.

---

## 9. Hard constraints for whoever styles the app

- **CSP** (`ops/nginx.conf`, architecture §11.3): `default-src 'self'`,
  `style-src 'self' 'unsafe-inline'`, `img-src 'self' https://i.ytimg.com data:`,
  `frame-src https://www.youtube-nocookie.com`. So: no Google Fonts or other
  font or CSS CDN (fonts fall under `default-src 'self'` and must be
  self-hosted and bundled); images only from our origin, YouTube thumbnails or
  `data:` URIs; inline `style` attributes are fine.
- **Thumbnails**, if #124 wants them: built only from a `video_id` matching
  `^[A-Za-z0-9_-]{11}$`, as `https://i.ytimg.com/vi/<id>/mqdefault.jpg`, with
  `alt=""`, `loading="lazy"` and a fixed size (detail in #120).
- **No new npm dependency without asking** (AGENTS.md -> Rules).
- **Tests stay green unchanged:** styling must not edit or remove a test
  assertion (#120). Tests are role- and text-based, so keep accessible names
  and copy stable.
- **React Compiler is on:** do not hand-memoise with `useMemo`/`useCallback`.
  `Compare.source.test.ts` enforces this for the Compare files; the rule applies
  everywhere.

---

## 10. Known gaps and inconsistencies in the current code

Not design decisions, just drift to fix or settle when styling lands:

- **No current-page indicator** in the main nav (`Link`, not `NavLink`; no
  `aria-current`). There is no skip-to-content link.
- **Error markup varies.** Library and Transcript use
  `<div role="alert"><p/><button>Retry</button></div>`. Ops puts the button
  inside `<p role="alert">`. Compare's alert `<p>` leaves the button outside
  it. VideoDetail's load error is an `<h1>` + `<p>` + button with **no**
  `role="alert"`.
- **Loading markup varies.** Ops ("Loading queue depth…", "Loading jobs…") and
  Search ("Searching…") are plain `<p>`, not `role="status"` (Search's live
  region carries the result text instead).
- **Four not-found variants.** `NotFound.tsx`, VideoDetail and Compare say
  "Back to Library" and set `document.title`. Transcript's `VideoNotFound`
  links "Library" and sets no title.
- **Duplicates:** the `Section` component (§6) and the speaker-roster list
  (AnalysisView and Compare) each exist twice.
- **Pagination text differs:** only Transcript shows "Page N of M".
- **Code style:** `routes/Transcript.tsx` uses double quotes and semicolons;
  every other file uses single quotes and no semicolons. There is no
  formatter config to enforce either.
- **Leftovers:** `src/App.tsx` (an `<h1>ytdigest</h1>`) is used only by its
  own test. architecture.md §9 names `ClaimList.tsx`, but the file is
  `components/ClaimItem.tsx`.
- **Wide tables** (Ops jobs, Compare) have no narrow-screen layout.

---

## 11. Decisions owed by #124

These are open. Until the owner decides, a view must not pick its own answer.

- Visual direction: A editorial, B dense tool, C hybrid, D media-first
- Colour tokens (text, muted, background, surface, border, link, focus,
  success, warning, danger, highlight), and whether the printable page's
  `#1a1a1a` / `#555` / `#0b57d0` / `#ccc` seed them
- Light only, or light and dark (and whether to set `color-scheme`)
- Font stack (system stack like the printable page, or a self-hosted face) and
  the type scale
- Spacing scale, radius, content max width (the printable page uses `48rem`)
- CSS approach: global stylesheet, CSS modules, or a framework (a framework
  needs owner approval as a new dependency)
- Button variants, status badge or label treatment, row density, thumbnails
- Phone layout for wide tables; where the player sits on the Transcript page
- Focus ring style and minimum touch-target size

When #124 is done, replace this section with the tokens, and update §1 so
styling is allowed.
