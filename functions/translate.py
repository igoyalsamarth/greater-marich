"""Translate diarized STT output using Pydantic AI and OpenAI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from functions.dialogues import dialogue_entries, load_dialogues
from functions.translate_agent import summarize_plot, translate_line
from functions.translate_models import (
    CharacterPersona,
    LinePosition,
    TranslationRequest,
    build_plot_summary_request,
    enriched_dialogue_line,
    load_character_personas,
    neighbor_lines,
    scene_metadata_from_stt,
    translated_entry_from_source,
)
from lib.constants import TRANSLATE_DIR

load_dotenv()

DEFAULT_TARGET_LANGUAGE = "hi-IN"
DEFAULT_LLM_MODEL = "gpt-5.4-mini"


def _ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")


def _source_name(path: Path) -> str:
    if path.stem == "dialogues":
        return path.parent.name
    return path.stem


def _resolve_output_path(stt_path: Path, output: str | Path | None) -> Path:
    name = _source_name(stt_path)
    if output is None:
        return TRANSLATE_DIR / name / "dialogues.json"

    output_path = Path(output).expanduser().resolve()
    if output_path.suffix:
        return output_path
    return output_path / "dialogues.json"


def _characters_path_for_stt(stt_path: Path) -> Path:
    return stt_path.parent / "characters.json"


def _build_translation_request(
    *,
    data: dict[str, Any],
    entries: list[dict[str, Any]],
    index: int,
    personas: dict[str, CharacterPersona],
    source_language_code: str,
    target_language_code: str,
    plot_summary: str,
    translated_texts: list[str | None],
) -> TranslationRequest:
    total = len(entries)
    return TranslationRequest(
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        plot_summary=plot_summary,
        line_position=LinePosition(
            index=index,
            total=total,
            is_first=index == 0,
            is_last=index == total - 1,
        ),
        scene=scene_metadata_from_stt(data),
        characters=sorted(personas.values(), key=lambda persona: persona.id),
        previous_lines=neighbor_lines(
            entries=entries,
            index=index,
            personas=personas,
            direction="previous",
            translated_texts=translated_texts,
        ),
        current_line=enriched_dialogue_line(
            index=index,
            entry=entries[index],
            personas=personas,
            entries=entries,
            include_speakability_hint=True,
        ),
        next_lines=neighbor_lines(
            entries=entries,
            index=index,
            personas=personas,
            direction="next",
            translated_texts=translated_texts,
        ),
    )


def _build_translated_output(
    *,
    data: dict[str, Any],
    source_language_code: str,
    target_language_code: str,
    model: str,
    source_dialogues_path: Path,
    plot_summary: str,
    translated_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": data.get("name"),
        "language_code": target_language_code,
        "source_language_code": source_language_code,
        "source_audio": data.get("source_audio"),
        "source_dialogues": str(source_dialogues_path),
        "num_speakers": data.get("num_speakers"),
        "translation_model": model,
        "plot_summary": plot_summary,
        "transcript": " ".join(
            entry["transcript"] for entry in translated_entries if entry.get("transcript")
        ),
        "diarized_transcript": {"entries": translated_entries},
    }


def translate_diarized_transcript(
    stt_path: str | Path,
    output: str | Path | None = None,
    *,
    source_language_code: str | None = None,
    target_language_code: str = DEFAULT_TARGET_LANGUAGE,
    model: str = DEFAULT_LLM_MODEL,
) -> Path:
    """Translate each dialogue entry using a two-step Pydantic AI flow with OpenAI.

    Step 1 summarizes the full scene for plot context. Step 2 translates each line
    with that summary, character personas, and surrounding dialogue context.

    Args:
        stt_path: Path to ``outputs/stt/<name>/dialogues.json``.
        output: Output JSON file or directory. Defaults to ``outputs/translate/<name>/dialogues.json``.
        source_language_code: Source language. Defaults to the STT file's ``language_code``.
        target_language_code: Target language for translation.
        model: OpenAI chat model to use for translation.

    Returns:
        Path to the saved translated JSON file.
    """
    _ensure_openai_api_key()

    stt_path = Path(stt_path).expanduser().resolve()
    if not stt_path.is_file():
        raise FileNotFoundError(f"STT file not found: {stt_path}")

    data = load_dialogues(stt_path)
    entries = dialogue_entries(data)
    source_language = source_language_code or data.get("language_code") or "auto"
    personas = load_character_personas(_characters_path_for_stt(stt_path))

    plot_summary = summarize_plot(
        build_plot_summary_request(
            data=data,
            entries=entries,
            personas=personas,
            source_language_code=source_language,
        ),
        model=model,
    )

    translated_entries: list[dict[str, Any]] = []
    translated_texts: list[str | None] = [None] * len(entries)

    for index, entry in enumerate(entries):
        current_text = str(entry.get("transcript") or entry.get("text") or "").strip()
        if not current_text:
            translated_entries.append(translated_entry_from_source(entry, ""))
            continue

        request = _build_translation_request(
            data=data,
            entries=entries,
            index=index,
            personas=personas,
            source_language_code=source_language,
            target_language_code=target_language_code,
            plot_summary=plot_summary,
            translated_texts=translated_texts,
        )
        translated_text = translate_line(request, model=model)
        translated_texts[index] = translated_text
        translated_entries.append(translated_entry_from_source(entry, translated_text))

    output_data = _build_translated_output(
        data=data,
        source_language_code=source_language,
        target_language_code=target_language_code,
        model=model,
        source_dialogues_path=stt_path,
        plot_summary=plot_summary,
        translated_entries=translated_entries,
    )

    output_path = _resolve_output_path(stt_path, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
