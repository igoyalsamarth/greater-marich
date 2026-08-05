from __future__ import annotations

from pathlib import Path

import typer

from functions.translate import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TARGET_LANGUAGE,
    translate_diarized_transcript,
)

app = typer.Typer(help="Translate STT output.")


@app.command()
def diarized(
    stt_json: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the STT output JSON file.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output JSON file or directory (default: translate/).",
        file_okay=True,
        dir_okay=True,
    ),
    source_language_code: str | None = typer.Option(
        None,
        "--source-language-code",
        help="Source language (default: taken from the STT JSON).",
    ),
    target_language_code: str = typer.Option(
        DEFAULT_TARGET_LANGUAGE,
        "--target-language-code",
        help="Target language for translation.",
    ),
    model: str = typer.Option(
        DEFAULT_LLM_MODEL,
        "--model",
        help="Sarvam chat model to use for translation.",
    ),
) -> None:
    """Translate each diarized transcript entry using Sarvam's chat LLM."""
    path = translate_diarized_transcript(
        stt_json,
        output,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        model=model,
    )
    typer.echo(f"Saved to {path}")
