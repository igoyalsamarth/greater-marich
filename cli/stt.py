from __future__ import annotations

from pathlib import Path

import typer

from functions.stt import (
    DEFAULT_LANGUAGE_CODE,
    transcribe_audio,
)

app = typer.Typer(help="Speech-to-text transcription.")


@app.command()
def transcribe(
    audio: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the converted audio file (uses separated vocals).",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output JSON file or directory (default: stt/).",
        file_okay=True,
        dir_okay=True,
    ),
    language_code: str = typer.Option(
        DEFAULT_LANGUAGE_CODE,
        "--language-code",
        help="Language code of the input audio.",
    ),
    with_timestamps: bool = typer.Option(
        True,
        "--with-timestamps",
        help="Whether to include timestamps in the transcription.",
    ),
) -> None:
    """Transcribe separated vocals using Sarvam's batch speech-to-text API."""
    path = transcribe_audio(
        audio,
        output,
        language_code=language_code,
        with_timestamps=with_timestamps,
    )
    typer.echo(f"Saved to {path}")
