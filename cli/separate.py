from __future__ import annotations

from pathlib import Path

import typer

from functions.separate import separate_audio

app = typer.Typer(help="Separate vocals and instruments from audio.")


@app.command()
def run(
    audio: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the converted audio file.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory (default: separation/<name>/).",
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Separate vocals, instruments, and other stems from audio."""
    path = separate_audio(audio, output)
    typer.echo(f"Saved to {path}")
