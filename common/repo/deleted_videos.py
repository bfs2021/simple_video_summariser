from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg


class VideoNotFound(Exception):
    pass


class VideoRunning(Exception):
    pass


@dataclass(frozen=True, slots=True)
class DeletedVideo:
    media_paths: tuple[str, ...]


def delete_video(
    conn: psycopg.Connection[Any],
    video_id: str,
    *,
    before_commit: Callable[[], None] | None = None,
) -> DeletedVideo:
    with conn.transaction():
        states = conn.execute(
            "SELECT state FROM jobs WHERE video_id = %s ORDER BY id FOR UPDATE", (video_id,)
        ).fetchall()
        video = conn.execute(
            "SELECT 1 FROM videos WHERE video_id = %s FOR UPDATE", (video_id,)
        ).fetchone()
        if video is None:
            raise VideoNotFound(video_id)
        if any(row[0] == "running" for row in states):
            raise VideoRunning(video_id)
        paths = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT path FROM media WHERE video_id = %s ORDER BY id", (video_id,)
            ).fetchall()
        )
        conn.execute(
            "INSERT INTO deleted_videos (video_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (video_id,),
        )
        conn.execute("DELETE FROM jobs WHERE video_id = %s", (video_id,))
        conn.execute("DELETE FROM videos WHERE video_id = %s", (video_id,))
        if before_commit is not None:
            before_commit()
    return DeletedVideo(paths)
