"""Shared helpers for speech generation pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from functions.dialogues import dialogue_entries, load_dialogues
from lib.constants import SEPARATION_DIR


def source_name(path: Path) -> str:
    if path.stem == "dialogues":
        return path.parent.name
    return path.stem


def segment_id(index: int) -> str:
    return f"{index:04d}"


def entry_text(entry: dict[str, Any]) -> str:
    return str(entry.get("transcript") or entry.get("text") or "")


def entry_start(entry: dict[str, Any]) -> float:
    if "start" in entry:
        return float(entry["start"])
    return float(entry["start_time_seconds"])


def entry_end(entry: dict[str, Any]) -> float:
    if "end" in entry:
        return float(entry["end"])
    return float(entry["end_time_seconds"])


def entry_duration(entry: dict[str, Any]) -> float:
    return max(entry_end(entry) - entry_start(entry), 0.0)


def entry_speaker_id(entry: dict[str, Any]) -> str:
    return str(entry.get("speaker_id") or entry.get("character_id") or "")


def vocals_path_for_name(name: str) -> Path:
    vocals_path = SEPARATION_DIR / name / f"{name}_vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError(
            f"Separated vocals not found: {vocals_path}. "
            "Run `separate run` on the converted audio first."
        )
    return vocals_path


def instrumental_path_for_name(name: str) -> Path | None:
    instrumental_path = SEPARATION_DIR / name / f"{name}_instrumental.wav"
    return instrumental_path if instrumental_path.is_file() else None


def resolve_source_dialogues(
    data: dict[str, Any],
    transcript_path: Path,
) -> list[dict[str, Any]]:
    """Load the original STT dialogues aligned with translated entries."""
    source_path = data.get("source_dialogues")
    if source_path:
        return dialogue_entries(load_dialogues(Path(source_path)))

    from lib.constants import STT_DIR

    name = data.get("name") or source_name(transcript_path)
    stt_path = STT_DIR / name / "dialogues.json"
    if not stt_path.is_file():
        raise FileNotFoundError(
            f"Source STT dialogues not found: {stt_path}. "
            "Re-run translate so source_dialogues is recorded, or run STT first."
        )
    return dialogue_entries(load_dialogues(stt_path))


def resolve_characters_path(
    data: dict[str, Any],
    transcript_path: Path,
) -> Path:
    """Resolve STT characters.json for the scene behind translated dialogues."""
    source_path = data.get("source_dialogues")
    if source_path:
        characters_path = Path(source_path).parent / "characters.json"
        if characters_path.is_file():
            return characters_path

    from lib.constants import STT_DIR

    name = data.get("name") or source_name(transcript_path)
    characters_path = STT_DIR / name / "characters.json"
    if not characters_path.is_file():
        raise FileNotFoundError(
            f"Characters file not found: {characters_path}. "
            "Run STT first to generate character profiles."
        )
    return characters_path


def write_speech_mapping(
    *,
    mapping_path: Path,
    mapping: dict[str, Any],
) -> Path:
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mapping_path
