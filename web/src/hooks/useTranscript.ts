import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { components } from "../api";
import { api, rethrowHttp, type ApiError } from "../api";
import { PAGE_SIZE } from "../lib/transcript";

export type TranscriptPage = components["schemas"]["TranscriptOut"];

/** `video` and `transcript` are the two 404 bodies of #42; anything else is `other`. */
export type NotFoundKind = "video" | "transcript";

/**
 * A non-2xx response. Carries the status and, for a 404, which resource is
 * missing. The body text is never kept, so it cannot reach the page.
 */
export class TranscriptApiError extends Error {
  readonly status: number;
  readonly notFound: NotFoundKind | null;
  constructor(status: number, notFound: NotFoundKind | null = null) {
    super(`HTTP ${status}`);
    this.name = "TranscriptApiError";
    this.status = status;
    this.notFound = notFound;
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

function isSegment(v: unknown): boolean {
  return (
    isRecord(v) &&
    typeof v.index === "number" &&
    typeof v.start === "number" &&
    typeof v.text === "string" &&
    (v.speaker === null || typeof v.speaker === "string")
  );
}

function isTranscriptPage(v: unknown): v is TranscriptPage {
  return (
    isRecord(v) &&
    typeof v.transcript_id === "number" &&
    typeof v.source === "string" &&
    (v.language === null || typeof v.language === "string") &&
    typeof v.total_segments === "number" &&
    Array.isArray(v.segments) &&
    v.segments.every(isSegment)
  );
}

/** Any 404 that is not exactly "transcript not found" counts as a missing video. */
function notFoundKind(error: ApiError): NotFoundKind {
  return error.detail === "transcript not found" ? "transcript" : "video";
}

export function transcriptQuery(page: number) {
  return { offset: (page - 1) * PAGE_SIZE, limit: PAGE_SIZE };
}

async function fetchTranscript(
  videoId: string,
  page: number,
  signal: AbortSignal,
): Promise<TranscriptPage> {
  const body: unknown = await api
    .get("/videos/{video_id}/transcript", {
      params: { path: { video_id: videoId }, query: transcriptQuery(page) },
      signal,
    })
    .catch((error: unknown) =>
      rethrowHttp(error, (e, status) =>
        status === 404
          ? new TranscriptApiError(404, notFoundKind(e))
          : new TranscriptApiError(status),
      ),
    );
  if (!isTranscriptPage(body)) throw new Error("Unexpected response shape");
  return body;
}

/**
 * One server-side page of the video's best transcript. The key carries the id
 * and the page, so pages are cached separately and one video's segments never
 * answer another's. No polling: it refetches on page change, Retry or remount.
 * `videoId` must already have passed `parseVideoId`.
 */
export function useTranscript(videoId: string, page: number) {
  return useQuery({
    queryKey: ["transcript", videoId, page],
    queryFn: ({ signal }) => fetchTranscript(videoId, page, signal),
    placeholderData: keepPreviousData,
  });
}
