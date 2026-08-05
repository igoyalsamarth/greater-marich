"""Download videos from YouTube using yt-dlp with FFmpeg for post-processing."""

from __future__ import annotations

from pathlib import Path

import yt_dlp

from lib.constants import VIDEOS_DIR
from functions.naming import next_numbered_filename


def download_youtube_video(
    url: str,
    output_dir: str | Path = VIDEOS_DIR,
    *,
    format: str = "bv*+ba/b",
    merge_format: str = "mp4",
) -> Path:
    """Download a YouTube video and return the path to the saved file.

    Uses yt-dlp to fetch streams and FFmpeg (must be installed on PATH) to merge
    separate video/audio tracks into a single file.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded file.
        format: yt-dlp format selector. Defaults to best video + best audio.
        merge_format: Container format when merging separate streams.

    Returns:
        Path to the downloaded video file.

    Raises:
        yt_dlp.utils.DownloadError: If the download fails.
        RuntimeError: If the output file path cannot be determined.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if numbered_stem := next_numbered_filename(output_dir):
        outtmpl = str(output_dir / f"{numbered_stem}.%(ext)s")
    else:
        outtmpl = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts: dict = {
        "outtmpl": outtmpl,
        "format": format,
        "merge_output_format": merge_format,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        if filepath := info.get("filepath"):
            return Path(filepath)

        downloaded_path = Path(ydl.prepare_filename(info))

    merged_path = downloaded_path.with_suffix(f".{merge_format}")
    if merged_path.exists():
        return merged_path
    if downloaded_path.exists():
        return downloaded_path

    raise RuntimeError(f"Download finished but file not found: {downloaded_path}")
