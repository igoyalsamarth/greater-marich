"""Generate speech audio snippets from diarized transcript entries."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from client.sarvam_client import get_sarvam_client
from functions.audio import match_volume_to_reference
from functions.dialogues import dialogue_entries, load_dialogues
from functions.speech_shared import (
    entry_duration,
    entry_end,
    entry_speaker_id,
    entry_start,
    entry_text,
    segment_id,
    source_name,
    vocals_path_for_name,
    write_speech_mapping,
)
from lib.constants import SPEECH_DIR
from lib.emotion_profile import dominant_emotion, emotion_profile_from_entry

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

    name = data.get("name") or source_name(transcript_path)
    speech_dir = Path(output_dir).expanduser().resolve() if output_dir else _speech_dir_for_name(name)
    speech_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = vocals_path_for_name(name)

    client = get_sarvam_client()
    language_code = data.get("language_code") or "hi-IN"

    segments: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        seg_id = segment_id(index)
        audio_file = f"{seg_id}.wav"
        audio_path = speech_dir / audio_file

        text = entry_text(entry).strip()
        speaker_id = entry_speaker_id(entry)
        emotion_profile = emotion_profile_from_entry(entry)
        start_seconds = entry_start(entry)
        end_seconds = entry_end(entry)
        slot_seconds = entry_duration(entry)

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
            "id": seg_id,
            "audio_file": audio_file,
            "start_time_seconds": round(start_seconds, 3),
            "end_time_seconds": round(end_seconds, 3),
            "slot_duration_seconds": round(slot_seconds, 3),
            "speaker_id": speaker_id,
            "character_id": speaker_id,
            "emotion_profile": emotion_profile,
            "emotion": dominant_emotion(emotion_profile),
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
        "tts_engine": "sarvam",
        "reference_vocals": str(vocals_path),
        "segments": segments,
    }

    return write_speech_mapping(
        mapping_path=speech_dir / f"{name}.json",
        mapping=mapping,
    )
