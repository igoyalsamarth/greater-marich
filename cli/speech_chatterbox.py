from __future__ import annotations

from pathlib import Path

import typer

from functions.speech_chatterbox import generate_chatterbox_speech
from lib.chatterbox_loader import (
    DEFAULT_MODEL,
    HINDI_LANGUAGE_ID,
    download_chatterbox_model,
    normalize_chatterbox_language_id,
)

app = typer.Typer(help="Generate speech with Chatterbox Multilingual Hindi cloning.")


@app.command()
def download(
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Hugging Face model id for the Hindi Chatterbox language pack.",
    ),
) -> None:
    """Download Chatterbox Hindi model weights to the local cache."""
    path = download_chatterbox_model(model)
    typer.echo(f"Model cached at {path}")


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
        help="Output directory (default: outputs/speech-chatterbox/<name>/).",
        file_okay=False,
        dir_okay=True,
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Hugging Face model id for the Hindi Chatterbox language pack.",
    ),
    language_id: str = typer.Option(
        HINDI_LANGUAGE_ID,
        "--language-id",
        help="Chatterbox language id. Must be ISO 639-1 'hi', not 'hi-IN'.",
    ),
) -> None:
    """Generate Hindi speech with zero-shot voice cloning from separated vocals."""
    path = generate_chatterbox_speech(
        transcript_json,
        output,
        model_id=model,
        language_id=normalize_chatterbox_language_id(language_id),
    )
    typer.echo(f"Saved to {path}")
