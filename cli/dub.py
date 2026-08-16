from __future__ import annotations

from pathlib import Path

import typer

from functions.dub import dub_video

app = typer.Typer(help="Dub videos with separated instrumental and generated speech.")


@app.command()
def create(
    mapping_json: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to outputs/speech/<name>/<name>.json mapping file.",
    ),
    video: Path | None = typer.Option(
        None,
        "--video",
        help="Source video (default: outputs/videos/<name>.mp4).",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output video file (default: outputs/dub/<name>.mp4).",
        file_okay=True,
        dir_okay=False,
    ),
    speech_gain_db: float = typer.Option(
        0.0,
        "--speech-gain-db",
        help="Extra gain (dB) applied to the speech bus after source matching.",
    ),
) -> None:
    """Mux downloaded video with separated instrumental and dubbed speech."""
    path = dub_video(mapping_json, video, output, speech_gain_db=speech_gain_db)
    typer.echo(f"Saved to {path}")
