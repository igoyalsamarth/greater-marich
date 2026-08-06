from __future__ import annotations

from pathlib import Path

import typer

from functions.stt import (
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_MODE,
    DEFAULT_MODEL,
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
        help="Output directory (default: outputs/stt/<name>/).",
        file_okay=True,
        dir_okay=True,
    ),
    language_code: str = typer.Option(
        DEFAULT_LANGUAGE_CODE,
        "--language-code",
        help="Language code of the input audio.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Sarvam STT model.",
    ),
    mode: str = typer.Option(
        DEFAULT_MODE,
        "--mode",
        help="Transcription mode for saaras models.",
    ),
) -> None:
    """Transcribe separated vocals with Sarvam diarization and emotion enrichment."""
    dialogues_path = transcribe_audio(
        audio,
        output,
        model=model,
        mode=mode,
        language_code=language_code,
    )
    characters_path = dialogues_path.parent / "characters.json"
    typer.echo(f"Saved characters to {characters_path}")
    typer.echo(f"Saved dialogues to {dialogues_path}")
