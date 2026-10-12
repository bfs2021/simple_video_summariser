# Recorded yt-dlp channel resolutions (#81)

Captured on 2026-10-12 with yt-dlp 2026.08.19 (`requirements.backend.txt`), by:

```
yt-dlp --flat-playlist --dump-single-json --playlist-items 0 --no-warnings <url>
```

with `<url>` one of `https://www.youtube.com/@YouTube`, `https://www.youtube.com/c/YouTube`,
`https://www.youtube.com/user/YouTube` and `https://www.youtube.com/@Ünal`. Each file keeps
only `channel_id`, `id`, `channel` and `title`, the fields the adapter reads. The three
`YouTube` forms all resolve to `UCBR8-60-B28hp2BmDPdntcQ`. `@Ünal` is a non-ASCII handle and
resolves to `UC6FTzBckPgMYMNMwmwAhjqg`.
