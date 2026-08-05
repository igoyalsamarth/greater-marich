"""Generate speech audio snippets from diarized transcript entries."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from client.sarvam_client import get_sarvam_client
from functions.audio import match_volume_to_reference
from lib.constants import SEPARATION_DIR, SPEECH_DIR

DEFAULT_TTS_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "kabir"
SPEAKER_VOICE_MAP = {
    "0": "kabir",
    "1": "kabir",
    "2": "kabir",
    "3": "suhani",
}


def _speech_dir_for_name(name: str) -> Path:
    return SPEECH_DIR / name


def _speaker_for_id(speaker_id: str) -> str:
    return SPEAKER_VOICE_MAP.get(speaker_id, DEFAULT_SPEAKER)


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

    Each snippet is volume-matched to the corresponding slice of the separated
    vocals track from ``separation/<name>/<name>_vocals.wav``.

    Args:
        transcript_json: Path to translated (or STT) JSON with diarized entries.
        output_dir: Output directory. Defaults to ``speech/<name>/``.
        model: Sarvam TTS model.

    Returns:
        Path to the mapping JSON file (``<output_dir>/<name>.json``).
    """
    transcript_path = Path(transcript_json).expanduser().resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_path}")

    data: dict[str, Any] = json.loads(transcript_path.read_text(encoding="utf-8"))
    entries = (data.get("diarized_transcript") or {}).get("entries") or []
    if not entries:
        raise RuntimeError("No diarized_transcript entries found in JSON")

    name = transcript_path.stem
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

        text = entry.get("transcript", "").strip()
        volume_match: dict[str, float] | None = None
        if text:
            response = client.text_to_speech.convert(
                text=text,
                language_code=language_code,
                speaker=_speaker_for_id(str(entry.get("speaker_id", ""))),
                model=model,
                output_audio_codec="wav",
            )
            audio_path.write_bytes(base64.b64decode(response.audios[0]))
            volume_match = match_volume_to_reference(
                audio_path,
                vocals_path,
                start_seconds=float(entry["start_time_seconds"]),
                end_seconds=float(entry["end_time_seconds"]),
            )

        segment: dict[str, Any] = {
            "id": segment_id,
            "audio_file": audio_file,
            "start_time_seconds": entry["start_time_seconds"],
            "end_time_seconds": entry["end_time_seconds"],
            "speaker_id": entry.get("speaker_id"),
            "transcript": entry.get("transcript", ""),
        }
        if volume_match:
            segment["volume_match"] = volume_match
        segments.append(segment)

    mapping = {
        "name": name,
        "language_code": language_code,
        "source_json": str(transcript_path),
        "reference_vocals": str(vocals_path),
        "segments": segments,
    }

    mapping_path = speech_dir / f"{name}.json"
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mapping_path
