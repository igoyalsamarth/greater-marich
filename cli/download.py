from __future__ import annotations

from pathlib import Path

import typer

from lib.constants import VIDEOS_DIR
from functions.download import download_youtube_video

app = typer.Typer(help="Download videos from YouTube.")


@app.command()
def video(
    url: str,
    output: Path = typer.Option(
        VIDEOS_DIR,
        "-o",
        "--output",
        help="Output directory (default: outputs/videos).",
        dir_okay=True,
        file_okay=False,
    ),
    video_format: str = typer.Option(
        "bv*+ba/b",
        "--format",
        help="yt-dlp format selector (best video + audio by default).",
    ),
) -> None:
    """Download a YouTube video."""
    path = download_youtube_video(url, output, format=video_format)
    typer.echo(f"Saved to {path}")
