from __future__ import annotations

from pathlib import Path

import typer

from functions.speech import DEFAULT_TTS_MODEL, generate_speech_snippets

app = typer.Typer(help="Generate speech snippets from diarized transcripts.")


@app.command()
def generate(
    transcript_json: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to translated JSON with diarized entries.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory (default: outputs/speech/<name>/).",
        file_okay=False,
        dir_okay=True,
    ),
    model: str = typer.Option(
        DEFAULT_TTS_MODEL,
        "--model",
        help="Sarvam TTS model.",
    ),
) -> None:
    """Generate Sarvam TTS snippets from translated dialogues."""
    path = generate_speech_snippets(transcript_json, output, model=model)
    typer.echo(f"Saved to {path}")
