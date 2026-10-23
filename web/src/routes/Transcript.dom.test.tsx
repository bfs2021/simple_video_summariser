import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "../api";
import { FakePlayerProvider } from "../components/FakePlayerProvider";
import { Transcript } from "./Transcript";

type TranscriptOut = components["schemas"]["TranscriptOut"];
type SegmentOut = components["schemas"]["SegmentOut"];

const ID = "-wNyEUrxzFU";
const OTHER = "AbC-_xYz012";

function seg(index: number, over: Partial<SegmentOut> = {}): SegmentOut {
  return {
    index,
    start: index * 5,
    end: index * 5 + 4,
    text: `Segment ${index}`,
    speaker: null,
    ...over,
  };
}

/** What #42 returns for `total` segments, honouring offset/limit. */
function transcriptPage(
  total: number,
  offset = 0,
  limit = 200,
  over: Partial<TranscriptOut> = {},
): TranscriptOut {
  const count = Math.max(0, Math.min(limit, total - offset));
  return {
    video_id: ID,
    transcript_id: 7,
    source: "youtube_manual",
    language: "en",
    speaker_source: "none",
    total_segments: total,
    offset,
    limit,
    segments: Array.from({ length: count }, (_, i) => seg(offset + i)),
    ...over,
  };
}

function videoBody(
  title: string | null,
  id = ID,
): components["schemas"]["VideoDetail"] {
  return {
    video_id: id,
    title,
    channel_id: null,
    channel_title: null,
    published_at: null,
    duration_sec: null,
    origin: "manual",
    unavailable: null,
    status: "done",
    latest_analysis_at: null,
    active_job: null,
    last_failure: null,
    transcript: null,
    analysis: null,
    progress: null,
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

type Handler = (url: URL) => Response | Promise<Response> | TranscriptOut;

let requests: URL[];
const transcriptRequests = () =>
  requests.filter((u) => u.pathname.endsWith("/transcript"));

function stubApi(handler: Handler, title: string | null | "fail" = "My Video") {
  requests = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://localhost");
      requests.push(url);
      if (!url.pathname.endsWith("/transcript")) {
        const id = url.pathname.split("/").pop() ?? "";
        if (title === "fail")
          return Promise.resolve(json({ detail: "boom" }, 500));
        return Promise.resolve(json(videoBody(title, id)));
      }
      const r = handler(url);
      return Promise.resolve(
        r instanceof Response || r instanceof Promise ? r : json(r),
      );
    }),
  );
}

/** A server with `total` segments for any video. */
function serverWith(total: number, over: Partial<TranscriptOut> = {}): Handler {
  return (url) =>
    transcriptPage(
      total,
      Number(url.searchParams.get("offset")),
      Number(url.searchParams.get("limit")),
      over,
    );
}

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

let seeks: number[];

async function mount(entry = `/videos/${ID}/transcript`) {
  seeks = [];
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const router = createMemoryRouter(
    [
      { path: "/", element: <p>library page</p> },
      { path: "/videos/:videoId", element: <p>detail page</p> },
      { path: "/videos/:videoId/transcript", element: <Transcript /> },
    ],
    { initialEntries: [entry] },
  );
  const view = render(
    <QueryClientProvider client={client}>
      <FakePlayerProvider seeks={seeks}>
        <RouterProvider router={router} />
      </FakePlayerProvider>
    </QueryClientProvider>,
  );
  await settle();
  return { router, client, ...view };
}

async function click(el: HTMLElement) {
  fireEvent.click(el);
  await settle();
}

beforeEach(() => {
  vi.useFakeTimers();
});

const list = () => screen.getByRole("list");
const items = () => within(list()).getAllByRole("listitem");
const next = () =>
  screen.getByRole("button", { name: "Next" }) as HTMLButtonElement;
const prev = () =>
  screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement;
const rangeText = () =>
  document.querySelector('[aria-live="polite"]')?.textContent ?? "";

describe("route input validation", () => {
  it.each([
    ["a 10-character id", "abcdefghij"],
    ["a 12-character id", "abcdefghijkl"],
    ["a percent-encoded slash", "abc%2Fdefghi"],
    ["dot dots", ".."],
    ["a space", "abc defghij"],
    ["a dot", "abcdefghij."],
    ["a non-ASCII letter", "abcdefghijé"],
  ])('%s shows "Video not found" and makes no request', async (_name, id) => {
    stubApi(serverWith(5));
    await mount(`/videos/${id}/transcript`);
    expect(screen.getByText("Video not found")).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Library" }).getAttribute("href"),
    ).toBe("/");
    expect(requests).toHaveLength(0);
    expect(document.querySelector("iframe")).toBeNull();
  });

  it("accepts a leading dash and keeps the case", async () => {
    stubApi(serverWith(3));
    await mount(`/videos/${ID}/transcript`);
    expect(transcriptRequests()[0]?.pathname).toBe(
      `/api/videos/${ID}/transcript`,
    );
    expect(items()).toHaveLength(3);
  });
});

describe("page parameter", () => {
  it("starts on page 1 with offset 0 and limit 200, built with query parameters", async () => {
    stubApi(serverWith(5));
    await mount();
    expect(transcriptRequests()).toHaveLength(1);
    expect(transcriptRequests()[0]?.search).toBe("?offset=0&limit=200");
  });

  it("restores page 3 from the URL on reload", async () => {
    stubApi(serverWith(2500));
    await mount(`/videos/${ID}/transcript?page=3`);
    expect(transcriptRequests()[0]?.search).toBe("?offset=400&limit=200");
    expect(rangeText()).toContain("Segments 401–600 of 2 500");
    expect(screen.getByText("Page 3 of 13")).toBeTruthy();
  });

  it.each(["0", "-1", "1.5", "abc", "1e3", "", "99999999999999999999", "5002"])(
    "page=%j falls back to page 1, is replaced out of the URL and never sent",
    async (value) => {
      stubApi(serverWith(5));
      const { router } = await mount(`/videos/${ID}/transcript?page=${value}`);
      expect(transcriptRequests()).toHaveLength(1);
      expect(transcriptRequests()[0]?.search).toBe("?offset=0&limit=200");
      expect(router.state.location.search).toBe("");
      expect(router.state.historyAction).toBe("REPLACE");
      expect(router.state.location.key).toBeTruthy();
    },
  );

  it("drops a repeated page parameter", async () => {
    stubApi(serverWith(2500));
    const { router } = await mount(`/videos/${ID}/transcript?page=2&page=3`);
    expect(transcriptRequests()[0]?.search).toBe("?offset=0&limit=200");
    expect(router.state.location.search).toBe("");
  });

  it("accepts the last page the offset cap allows", async () => {
    stubApi(serverWith(2500));
    await mount(`/videos/${ID}/transcript?page=5001`);
    expect(transcriptRequests()[0]?.search).toBe("?offset=1000000&limit=200");
  });

  it("keeps other query parameters when it drops an invalid page", async () => {
    stubApi(serverWith(5));
    const { router } = await mount(`/videos/${ID}/transcript?page=0&x=1`);
    expect(router.state.location.search).toBe("?x=1");
  });

  it("never sends or shows a hostile page value", async () => {
    const marker = "MARKER<b>zz</b>";
    stubApi(serverWith(5));
    await mount(`/videos/${ID}/transcript?page=${encodeURIComponent(marker)}`);
    for (const u of requests)
      expect(decodeURIComponent(u.href)).not.toContain("MARKER");
    expect(document.body.innerHTML).not.toContain("MARKER");
  });

  it("pushes a page change so Back and Forward walk the pages", async () => {
    stubApi(serverWith(500));
    const { router } = await mount();
    await click(next());
    expect(router.state.location.search).toBe("?page=2");
    expect(router.state.historyAction).toBe("PUSH");
    expect(transcriptRequests()[1]?.search).toBe("?offset=200&limit=200");
    await click(next());
    expect(router.state.location.search).toBe("?page=3");
    await act(async () => {
      await router.navigate(-1);
    });
    await settle();
    expect(rangeText()).toContain("Segments 201–400 of 500");
    await act(async () => {
      await router.navigate(-1);
    });
    await settle();
    expect(rangeText()).toContain("Segments 1–200 of 500");
    await act(async () => {
      await router.navigate(1);
    });
    await settle();
    expect(rangeText()).toContain("Segments 201–400 of 500");
  });
});

describe("data", () => {
  it("uses only the first page request per render and shows at most 200 segments", async () => {
    stubApi(serverWith(2500));
    await mount();
    expect(transcriptRequests()).toHaveLength(1);
    expect(items()).toHaveLength(200);
  });

  it("caches pages separately, so going back does not refetch", async () => {
    stubApi(serverWith(500));
    await mount();
    await click(next());
    expect(transcriptRequests()).toHaveLength(2);
    await click(prev());
    expect(items()[0]?.textContent).toContain("Segment 0");
  });

  it("does not show the first video segments under the second video", async () => {
    stubApi((url) =>
      transcriptPage(3, 0, 200, {
        video_id: url.pathname.includes(OTHER) ? OTHER : ID,
        segments: [
          seg(0, {
            text: url.pathname.includes(OTHER) ? "second video" : "first video",
          }),
        ],
      }),
    );
    const { router } = await mount();
    expect(screen.getByText("first video")).toBeTruthy();
    await act(async () => {
      await router.navigate(`/videos/${OTHER}/transcript`);
    });
    await settle();
    expect(screen.queryByText("first video")).toBeNull();
    expect(screen.getByText("second video")).toBeTruthy();
  });

  it("does not poll", async () => {
    stubApi(serverWith(5));
    await mount();
    const before = requests.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10 * 60_000);
    });
    expect(requests).toHaveLength(before);
  });
});

describe("heading and title", () => {
  it("uses the video title for the h1 and document.title", async () => {
    stubApi(serverWith(5), "My Video");
    await mount();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
      "My Video",
    );
    expect(document.title).toBe("My Video – Transcript");
  });

  it.each([
    ["null", null],
    ["empty", ""],
    ["failed", "fail" as const],
  ])(
    "falls back to the video id when the title is %s, and the transcript still renders",
    async (_n, title) => {
      stubApi(serverWith(5), title);
      await mount();
      expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(ID);
      expect(document.title).toBe(`${ID} – Transcript`);
      expect(items()).toHaveLength(5);
    },
  );

  it("shows the video id while the video request is pending", async () => {
    stubApi(serverWith(5), "My Video");
    // Render synchronously, before any microtask runs.
    seeks = [];
    render(
      <QueryClientProvider client={new QueryClient()}>
        <FakePlayerProvider>
          <RouterProvider
            router={createMemoryRouter(
              [
                {
                  path: "/videos/:videoId/transcript",
                  element: <Transcript />,
                },
              ],
              {
                initialEntries: [`/videos/${ID}/transcript`],
              },
            )}
          />
        </FakePlayerProvider>
      </QueryClientProvider>,
    );
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(ID);
  });

  it("renders a hostile title as text", async () => {
    stubApi(serverWith(1), "<img src=x onerror=alert(1)>");
    await mount();
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
      "<img src=x onerror=alert(1)>",
    );
    expect(document.querySelector("h1 img")).toBeNull();
    expect(document.title).toBe("<img src=x onerror=alert(1)> – Transcript");
  });
});

describe("what the page shows", () => {
  it("links back to the video and embeds the player for the validated id", async () => {
    stubApi(serverWith(5));
    await mount();
    expect(
      screen.getByRole("link", { name: "Back to video" }).getAttribute("href"),
    ).toBe(`/videos/${ID}`);
    const frames = document.querySelectorAll("iframe");
    expect(frames).toHaveLength(1);
    expect(frames[0]?.getAttribute("src")).toContain(`/embed/${ID}`);
  });

  it.each([
    ["youtube_manual", "en", "YouTube captions (manual)", "en"],
    ["youtube_auto", "de", "YouTube auto-generated captions", "de"],
    ["whisper", null, "Machine transcription", "Language unknown"],
    ["brand_new", "fr", "brand_new", "fr"],
  ])(
    "describes source %s / language %s",
    async (source, language, srcText, langText) => {
      stubApi(serverWith(5, { source, language }));
      await mount();
      const line = screen.getByText(srcText, { exact: false });
      expect(line.textContent).toContain(srcText);
      expect(line.textContent).toContain(langText);
    },
  );

  it("renders an ordered list in API order, numbered from offset + 1, keyed by absolute index", async () => {
    stubApi(serverWith(500));
    await mount(`/videos/${ID}/transcript?page=2`);
    expect(list().tagName).toBe("OL");
    expect(list().getAttribute("start")).toBe("201");
    const texts = items().map((li) => li.textContent ?? "");
    expect(texts[0]).toContain("Segment 200");
    expect(texts[199]).toContain("Segment 399");
  });

  it("does not re-sort or filter what the API returned", async () => {
    stubApi(() =>
      transcriptPage(3, 0, 200, {
        segments: [
          seg(2, { start: 50 }),
          seg(0, { start: 90 }),
          seg(1, { start: 1 }),
        ],
      }),
    );
    await mount();
    expect(
      items().map((li) => /Segment \d/.exec(li.textContent ?? "")?.[0]),
    ).toEqual(["Segment 2", "Segment 0", "Segment 1"]);
  });

  it("shows start through Timestamp and seeks the player with the segment start on click", async () => {
    stubApi(() =>
      transcriptPage(2, 0, 200, {
        segments: [seg(0, { start: 0 }), seg(1, { start: 3723.5 })],
      }),
    );
    await mount();
    await click(screen.getByRole("button", { name: "Seek to 1:02:03" }));
    expect(seeks).toEqual([3723.5]);
    await click(screen.getByRole("button", { name: "Seek to 0:00" }));
    expect(seeks).toEqual([3723.5, 0]);
  });

  it("does not show the end time", async () => {
    stubApi(() =>
      transcriptPage(1, 0, 200, {
        segments: [seg(0, { start: 1, end: 98765 })],
      }),
    );
    await mount();
    expect(document.body.textContent).not.toContain("98765");
    expect(document.body.textContent).not.toContain("27:26");
  });

  it("shows long text in full and keeps empty and blank segments in the numbering", async () => {
    const long = "word ".repeat(5000).trim();
    stubApi(() =>
      transcriptPage(3, 0, 200, {
        segments: [
          seg(0, { text: long }),
          seg(1, { text: "" }),
          seg(2, { text: "   " }),
        ],
      }),
    );
    await mount();
    expect(items()).toHaveLength(3);
    expect(items()[0]?.textContent).toContain(long);
    expect(within(items()[1] as HTMLElement).getByRole("button")).toBeTruthy();
    expect(within(items()[2] as HTMLElement).getByRole("button")).toBeTruthy();
  });

  it("renders non-ASCII text unchanged", async () => {
    const texts = [
      "café naïve",
      "日本語のテキスト",
      "😀 emoji 👍",
      "مرحبا بالعالم",
    ];
    stubApi(() =>
      transcriptPage(4, 0, 200, {
        segments: texts.map((text, i) => seg(i, { text })),
      }),
    );
    await mount();
    texts.forEach((t, i) => expect(items()[i]?.textContent).toContain(t));
  });
});

describe("speaker labels", () => {
  const withSpeakers = (...speakers: Array<string | null>) =>
    stubApi(() =>
      transcriptPage(speakers.length, 0, 200, {
        speaker_source: "subtitle_labels",
        segments: speakers.map((speaker, i) => seg(i, { speaker })),
      }),
    );

  it("labels the first segment and each change, with null counting as a value", async () => {
    withSpeakers("A", "A", null, "A");
    await mount();
    const has = (i: number) => (items()[i]?.textContent ?? "").includes("A");
    // Segment text is "Segment N" which has no capital A.
    expect([0, 1, 2, 3].map(has)).toEqual([true, false, false, true]);
  });

  it("shows no label, placeholder or empty slot when every speaker is null", async () => {
    withSpeakers(null, null, null);
    await mount();
    expect(document.body.textContent).not.toMatch(/unknown speaker|speaker/i);
    expect(document.querySelectorAll(".speaker")).toHaveLength(0);
    expect(document.querySelectorAll("li strong, li cite, li b")).toHaveLength(
      0,
    );
  });

  it("shows no placeholder for an empty speaker", async () => {
    withSpeakers("", "Bob", "");
    await mount();
    expect(items()[0]?.textContent).not.toMatch(/unknown/i);
    expect(items()[1]?.textContent).toContain("Bob");
    expect(items()[2]?.textContent).not.toContain("Bob");
  });

  it("restarts the labelling at the top of every page", async () => {
    stubApi((url) => {
      const offset = Number(url.searchParams.get("offset"));
      return transcriptPage(400, offset, 200, {
        segments: Array.from({ length: 200 }, (_, i) =>
          seg(offset + i, { speaker: "Zed" }),
        ),
      });
    });
    await mount();
    expect(items()[0]?.textContent).toContain("Zed");
    await click(next());
    expect(items()[0]?.textContent).toContain("Zed");
    expect(items()[1]?.textContent).not.toContain("Zed");
  });
});

describe("untrusted text", () => {
  it("renders hostile text, speakers and URL-looking text as literal text", async () => {
    const script = "<script>alert(1)</script>";
    const speaker = '"><img src=x onerror=alert(1)>';
    stubApi(() =>
      transcriptPage(2, 0, 200, {
        speaker_source: "subtitle_labels",
        segments: [
          seg(0, { text: script, speaker }),
          seg(1, {
            text: "see https://evil.example/x and javascript:alert(1)",
            speaker,
          }),
        ],
      }),
    );
    await mount();
    expect(items()[0]?.textContent).toContain(script);
    expect(items()[0]?.textContent).toContain(speaker);
    expect(document.querySelector("script")).toBeNull();
    expect(document.querySelector("img")).toBeNull();
    expect(items()[1]?.textContent).toContain("https://evil.example/x");
    for (const a of Array.from(document.querySelectorAll("li a")))
      throw new Error(`link from data: ${a.outerHTML}`);
    expect(document.querySelector("[onerror]")).toBeNull();
    expect(list().querySelectorAll("[href], [src]")).toHaveLength(0);
  });
});

describe("pagination", () => {
  it.each([
    [1, 1, "Segments 1–1 of 1", "Page 1 of 1"],
    [200, 200, "Segments 1–200 of 200", "Page 1 of 1"],
    [201, 200, "Segments 1–200 of 201", "Page 1 of 2"],
  ])(
    "total %i: page 1 shows %i segments, range %s",
    async (total, shown, range, pageText) => {
      stubApi(serverWith(total));
      await mount();
      expect(items()).toHaveLength(shown);
      expect(rangeText()).toBe(range);
      expect(screen.getByText(pageText)).toBeTruthy();
      expect(prev().disabled).toBe(true);
      expect(next().disabled).toBe(total <= 200);
    },
  );

  it("exactly one full page has no page 2", async () => {
    stubApi(serverWith(200));
    const { router } = await mount();
    expect(next().disabled).toBe(true);
    fireEvent.click(next());
    expect(router.state.location.search).toBe("");
  });

  it("total 201: page 2 has one segment and Next is disabled", async () => {
    stubApi(serverWith(201));
    await mount();
    await click(next());
    expect(items()).toHaveLength(1);
    expect(rangeText()).toBe("Segments 201–201 of 201");
    expect(next().disabled).toBe(true);
    expect(prev().disabled).toBe(false);
  });

  it("total 2 500: 13 pages, the last partial with 100 segments", async () => {
    stubApi(serverWith(2500));
    await mount(`/videos/${ID}/transcript?page=13`);
    expect(items()).toHaveLength(100);
    expect(rangeText()).toBe("Segments 2 401–2 500 of 2 500");
    expect(screen.getByText("Page 13 of 13")).toBeTruthy();
    expect(next().disabled).toBe(true);
    expect(prev().disabled).toBe(false);
  });

  it('shows the example range "Segments 201–400 of 2 500"', async () => {
    stubApi(serverWith(2500));
    await mount(`/videos/${ID}/transcript?page=2`);
    expect(rangeText()).toBe("Segments 201–400 of 2 500");
    expect(items()).toHaveLength(200);
  });

  it("a deep link past the end says so and links to the last page", async () => {
    stubApi(serverWith(2500));
    const { router } = await mount(`/videos/${ID}/transcript?page=99`);
    expect(screen.getByText("No segments on this page")).toBeTruthy();
    expect(screen.queryByRole("list")).toBeNull();
    const link = screen.getByRole("link", { name: /last page/i });
    expect(link.getAttribute("href")).toBe(`/videos/${ID}/transcript?page=13`);
    await click(link);
    expect(router.state.location.search).toBe("?page=13");
    expect(items()).toHaveLength(100);
  });

  it("an empty transcript says so and has no pagination controls", async () => {
    stubApi(serverWith(0));
    await mount();
    expect(screen.getByText("This transcript is empty")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Next" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Previous" })).toBeNull();
    expect(screen.queryByRole("list")).toBeNull();
    expect(screen.queryByText(/Page \d/)).toBeNull();
  });

  it('a non-empty total with an empty segment list on page 1 is not "empty transcript"', async () => {
    stubApi(() => transcriptPage(5, 0, 200, { segments: [] }));
    await mount();
    expect(screen.queryByText("This transcript is empty")).toBeNull();
    expect(screen.getByText("No segments on this page")).toBeTruthy();
  });

  it("keeps the current segments, shows a loading indicator and disables controls while the next page loads", async () => {
    let release: (r: Response) => void = () => {};
    stubApi((url) => {
      const offset = Number(url.searchParams.get("offset"));
      if (offset === 0) return transcriptPage(500, 0);
      return new Promise<Response>((resolve) => {
        release = resolve;
      });
    });
    await mount();
    fireEvent.click(next());
    await settle();
    expect(items()[0]?.textContent).toContain("Segment 0");
    expect(screen.getByText(/loading/i)).toBeTruthy();
    expect(next().disabled).toBe(true);
    expect(prev().disabled).toBe(true);
    await act(async () => {
      release(json(transcriptPage(500, 200)));
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(items()[0]?.textContent).toContain("Segment 200");
    expect(screen.queryByText(/loading/i)).toBeNull();
    expect(prev().disabled).toBe(false);
  });

  it("does not remount the player across a page change", async () => {
    stubApi(serverWith(500));
    await mount();
    const frame = document.querySelector("iframe");
    expect(frame).not.toBeNull();
    await click(next());
    await click(next());
    expect(document.querySelector("iframe")).toBe(frame);
    expect(document.querySelectorAll("iframe")).toHaveLength(1);
  });

  it("moves focus to the top of the segment list after a page change, not on first render", async () => {
    stubApi(serverWith(500));
    await mount();
    expect(document.activeElement).toBe(document.body);
    await click(next());
    const active = document.activeElement as HTMLElement;
    expect(active).not.toBe(document.body);
    expect(active.getAttribute("tabindex")).toBe("-1");
    expect(active.tagName).toMatch(/^H[2-6]$/);
    // It sits above the list in document order.
    expect(
      active.compareDocumentPosition(list()) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("announces the range in a polite live region", async () => {
    stubApi(serverWith(500));
    await mount();
    const region = document.querySelector('[aria-live="polite"]');
    expect(region?.textContent).toBe("Segments 1–200 of 500");
    await click(next());
    expect(document.querySelector('[aria-live="polite"]')?.textContent).toBe(
      "Segments 201–400 of 500",
    );
  });
});

describe("transcript replaced mid-read", () => {
  it("announces the update, follows the new source and never mixes transcripts", async () => {
    stubApi((url) => {
      const offset = Number(url.searchParams.get("offset"));
      if (offset === 0)
        return transcriptPage(500, 0, 200, {
          transcript_id: 1,
          source: "youtube_auto",
        });
      return transcriptPage(500, offset, 200, {
        transcript_id: 2,
        source: "whisper",
      });
    });
    await mount();
    expect(
      screen.queryByText("This transcript was updated while you were reading"),
    ).toBeNull();
    expect(document.body.textContent).toContain(
      "YouTube auto-generated captions",
    );
    await click(next());
    expect(
      screen.getByText("This transcript was updated while you were reading"),
    ).toBeTruthy();
    expect(document.body.textContent).toContain("Machine transcription");
    expect(document.body.textContent).not.toContain(
      "YouTube auto-generated captions",
    );
    expect(items()).toHaveLength(200);
    expect(items()[0]?.textContent).toContain("Segment 200");
  });

  it("shows no notice when the transcript id is unchanged", async () => {
    stubApi(serverWith(500));
    await mount();
    await click(next());
    expect(screen.queryByText(/updated while you were reading/)).toBeNull();
  });
});

describe("failure states", () => {
  it("shows a loading indicator with text on first visit, not the empty message", async () => {
    stubApi(() => new Promise<Response>(() => {}));
    await mount();
    expect(screen.getByText(/loading transcript/i)).toBeTruthy();
    expect(screen.queryByText("This transcript is empty")).toBeNull();
    expect(screen.queryByRole("list")).toBeNull();
  });

  it('404 "video not found": states it, links to the Library and embeds no player', async () => {
    stubApi(() => json({ detail: "video not found" }, 404));
    await mount();
    expect(screen.getByText("Video not found")).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Library" }).getAttribute("href"),
    ).toBe("/");
    expect(document.querySelector("iframe")).toBeNull();
  });

  it('404 "transcript not found": says no transcript yet, links back to the video and keeps the player', async () => {
    stubApi(() => json({ detail: "transcript not found" }, 404));
    await mount();
    expect(screen.getByText("No transcript yet for this video")).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Back to video" }).getAttribute("href"),
    ).toBe(`/videos/${ID}`);
    expect(document.querySelector("iframe")).not.toBeNull();
    expect(screen.queryByText("Video not found")).toBeNull();
  });

  it.each([
    ["an unknown detail", json({ detail: "something else" }, 404)],
    ["no JSON body", new Response("<html>nope</html>", { status: 404 })],
    ["a non-string detail", json({ detail: { x: 1 } }, 404)],
    ["an array body", json([], 404)],
  ])("any other 404 (%s) is treated as video not found", async (_n, res) => {
    stubApi(() => res);
    await mount();
    expect(screen.getByText("Video not found")).toBeTruthy();
    expect(document.querySelector("iframe")).toBeNull();
  });

  it("503 explains the database is unavailable, and Retry refetches", async () => {
    let calls = 0;
    stubApi(() =>
      ++calls === 1
        ? json({ detail: "database unavailable" }, 503)
        : transcriptPage(3),
    );
    await mount();
    expect(
      screen.getByText("The database is unavailable. Try again shortly."),
    ).toBeTruthy();
    expect(document.body.textContent).not.toContain('database unavailable"');
    await click(screen.getByRole("button", { name: "Retry" }));
    expect(calls).toBe(2);
    expect(items()).toHaveLength(3);
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it.each([
    ["a 500 with a body", () => json({ detail: "secret internal trace" }, 500)],
    ["a 422", () => json({ detail: [{ msg: "secret validation text" }] }, 422)],
    ["a 502", () => new Response("<h1>Bad gateway</h1>", { status: 502 })],
    ["a malformed 200", () => new Response("not json", { status: 200 })],
    ["a 200 of the wrong shape", () => json({ nope: true })],
    [
      "a network failure",
      () =>
        Promise.reject(
          new TypeError("Failed to fetch https://internal.example/x"),
        ),
    ],
  ])(
    "%s shows a generic message with Retry and leaks nothing",
    async (_n, make) => {
      stubApi(make as Handler);
      await mount();
      expect(screen.getByText("Could not load the transcript")).toBeTruthy();
      expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
      const text = document.body.textContent ?? "";
      for (const leak of [
        "secret",
        "Bad gateway",
        "internal.example",
        "Failed to fetch",
        "/api/videos",
        "HTTP ",
      ]) {
        expect(text).not.toContain(leak);
      }
    },
  );

  it("still shows the player after a generic error", async () => {
    stubApi(() => json({}, 500));
    await mount();
    expect(document.querySelector("iframe")).not.toBeNull();
  });
});
