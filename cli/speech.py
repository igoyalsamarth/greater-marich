from __future__ import annotations

from pathlib import Path

import typer

from functions.speech import generate_speech_snippets

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
        help="Output directory (default: speech/<name>/).",
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Generate TTS audio snippets matched to separated vocals volume."""
    path = generate_speech_snippets(transcript_json, output)
    typer.echo(f"Saved to {path}")
