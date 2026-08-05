from __future__ import annotations

from pathlib import Path

import typer

from functions.convert import convert_video_to_audio

app = typer.Typer(help="Convert media files.")


@app.command()
def audio(
    video: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the source video file.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory (default: audio/).",
        file_okay=True,
        dir_okay=True,
    ),
) -> None:
    """Extract audio from a video as lossless WAV for speech-to-text."""
    path = convert_video_to_audio(video, output)
    typer.echo(f"Saved to {path}")
