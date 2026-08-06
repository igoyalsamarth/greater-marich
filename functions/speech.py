"""Generate speech audio snippets from diarized transcript entries."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from client.sarvam_client import get_sarvam_client
from functions.audio import match_volume_to_reference
from functions.dialogues import dialogue_entries, load_dialogues
from lib.constants import SEPARATION_DIR, SPEECH_DIR

DEFAULT_TTS_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "kabir"
SPEAKER_VOICE_MAP = {
    "char_0": "aditya",
    "char_1": "roopa",
}


def _speech_dir_for_name(name: str) -> Path:
    return SPEECH_DIR / name


def _speaker_for_id(speaker_id: str) -> str:
    return SPEAKER_VOICE_MAP.get(speaker_id, DEFAULT_SPEAKER)


def _entry_text(entry: dict[str, Any]) -> str:
    return str(entry.get("transcript") or entry.get("text") or "")


def _entry_start(entry: dict[str, Any]) -> float:
    if "start" in entry:
        return float(entry["start"])
    return float(entry["start_time_seconds"])


def _entry_end(entry: dict[str, Any]) -> float:
    if "end" in entry:
        return float(entry["end"])
    return float(entry["end_time_seconds"])


def _entry_duration(entry: dict[str, Any]) -> float:
    return max(_entry_end(entry) - _entry_start(entry), 0.0)


def _entry_speaker_id(entry: dict[str, Any]) -> str:
    return str(entry.get("speaker_id") or entry.get("character_id") or "")


def _source_name(path: Path) -> str:
    if path.stem == "dialogues":
        return path.parent.name
    return path.stem


def _segment_id(index: int) -> str:
    return f"{index:04d}"


def _vocals_path_for_name(name: str) -> Path:
    vocals_path = SEPARATION_DIR / name / f"{name}_vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError(
            f"Separated vocals not found: {vocals_path}. "
            "Run `separate run` on the converted audio first."
        )
    return vocals_path


def generate_speech_snippets(
    transcript_json: str | Path,
    output_dir: str | Path | None = None,
    *,
    model: str = DEFAULT_TTS_MODEL,
) -> Path:
    """Generate TTS audio for each diarized entry and write a segment mapping.

    Reads ``outputs/translate/<name>/dialogues.json`` (or STT dialogues) and
    writes per-line WAV clips under ``outputs/speech/<name>/``.

    Each snippet is volume-matched to the corresponding slice of the separated
    vocals track. Sarvam Bulbul v3 is called with text, speaker, and language only.

    Args:
        transcript_json: Path to translated (or STT) JSON with diarized entries.
        output_dir: Output directory. Defaults to ``outputs/speech/<name>/``.
        model: Sarvam TTS model.

    Returns:
        Path to the mapping JSON file (``<output_dir>/<name>.json``).
    """
    transcript_path = Path(transcript_json).expanduser().resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_path}")

    data = load_dialogues(transcript_path)
    entries = dialogue_entries(data)

    name = data.get("name") or _source_name(transcript_path)
    speech_dir = Path(output_dir).expanduser().resolve() if output_dir else _speech_dir_for_name(name)
    speech_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = _vocals_path_for_name(name)

    client = get_sarvam_client()
    language_code = data.get("language_code") or "hi-IN"

    segments: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        segment_id = _segment_id(index)
        audio_file = f"{segment_id}.wav"
        audio_path = speech_dir / audio_file

        text = _entry_text(entry).strip()
        speaker_id = _entry_speaker_id(entry)
        emotion = str(entry.get("emotion") or "neutral")
        start_seconds = _entry_start(entry)
        end_seconds = _entry_end(entry)
        slot_seconds = _entry_duration(entry)

        volume_match: dict[str, float] | None = None
        if text:
            response = client.text_to_speech.convert(
                text=text,
                language_code=language_code,
                speaker=_speaker_for_id(speaker_id),
                model=model,
                output_audio_codec="wav",
            )
            audio_path.write_bytes(base64.b64decode(response.audios[0]))
            volume_match = match_volume_to_reference(
                audio_path,
                vocals_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )

        segment: dict[str, Any] = {
            "id": segment_id,
            "audio_file": audio_file,
            "start_time_seconds": round(start_seconds, 3),
            "end_time_seconds": round(end_seconds, 3),
            "slot_duration_seconds": round(slot_seconds, 3),
            "speaker_id": speaker_id,
            "character_id": speaker_id,
            "emotion": emotion,
            "transcript": text,
            "tts_speaker": _speaker_for_id(speaker_id),
        }
        if volume_match:
            segment["volume_match"] = volume_match
        segments.append(segment)

    mapping = {
        "name": name,
        "language_code": language_code,
        "source_language_code": data.get("source_language_code"),
        "source_json": str(transcript_path),
        "source_dialogues": data.get("source_dialogues"),
        "translation_model": data.get("translation_model"),
        "tts_model": model,
        "reference_vocals": str(vocals_path),
        "segments": segments,
    }

    mapping_path = speech_dir / f"{name}.json"
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mapping_path
