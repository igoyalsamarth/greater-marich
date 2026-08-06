"""Speech-to-text with Sarvam diarization, embeddings, and emotion."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from functions.dialogues import dialogue_entries
from functions.stt_embeddings import average_embeddings, extract_voice_embedding
from functions.stt_emotion import detect_emotion
from functions.stt_features import (
    dominant_emotions,
    estimate_gender,
    extract_segment_file,
    measure_pitch_mean,
    measure_speech_rate,
)
from functions.stt_transcribe import (
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_NUM_SPEAKERS,
    transcribe_audio as transcribe_with_sarvam,
)
from lib.constants import SEPARATION_DIR, STT_DIR

DEFAULT_LANGUAGE_CODE = "en-IN"


def _base_name(audio_path: Path) -> str:
    stem = audio_path.stem
    if stem.endswith("_vocals"):
        return stem[: -len("_vocals")]
    return stem


def _resolve_vocals_audio(audio_path: Path) -> Path:
    name = _base_name(audio_path)
    vocals_path = SEPARATION_DIR / name / f"{name}_vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError(
            f"Separated vocals not found: {vocals_path}. "
            "Run `separate run` on the converted audio first."
        )
    return vocals_path


def _resolve_output_dir(audio_path: Path, output: str | Path | None) -> Path:
    name = _base_name(audio_path)
    if output is None:
        return STT_DIR / name

    output_path = Path(output).expanduser().resolve()
    if output_path.suffix:
        return output_path.parent
    return output_path


def _speaker_to_character_id(speaker: str, mapping: dict[str, str]) -> str:
    if speaker not in mapping:
        mapping[speaker] = f"char_{len(mapping)}"
    return mapping[speaker]


def _sarvam_diarized_entries(transcription: dict[str, Any]) -> list[dict[str, Any]]:
    entries = (transcription.get("diarized_transcript") or {}).get("entries") or []
    if not entries:
        raise RuntimeError("No diarized_transcript entries found in Sarvam response")
    return entries


def _build_characters(
    speaker_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    characters: list[dict[str, Any]] = []
    for character_id in sorted(
        speaker_data,
        key=lambda item: int(item.split("_", maxsplit=1)[-1]),
    ):
        data = speaker_data[character_id]
        pitch_values = data["pitch_values"]
        pitch_mean = float(sum(pitch_values) / len(pitch_values)) if pitch_values else None
        speech_rates = data["speech_rates"]
        speech_rate = (
            float(sum(speech_rates) / len(speech_rates)) if speech_rates else 0.0
        )
        characters.append(
            {
                "id": character_id,
                "name": None,
                "voice_embedding": average_embeddings(data["embeddings"]),
                "gender": estimate_gender(pitch_mean),
                "pitch_mean": pitch_mean,
                "speech_rate": round(speech_rate, 3),
                "dominant_emotions": dominant_emotions(data["emotions"]),
            }
        )
    return characters


def transcribe_audio(
    audio_path: str | Path,
    output: str | Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    mode: str = DEFAULT_MODE,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    upload_timeout: float = 300.0,
    poll_timeout: int = 1800,
) -> Path:
    """Transcribe separated vocals and produce character and dialogue outputs.

    Uses Sarvam for diarization and transcription, then enriches each turn with
    voice embeddings, pitch, and emotion.

    Args:
        audio_path: Path to the converted audio file (for example ``outputs/audio/video1.wav``).
            The separated vocals at ``outputs/separation/<name>/<name>_vocals.wav`` are used.
        output: Output directory. Defaults to ``outputs/stt/<name>/``.
        model: Sarvam STT model.
        mode: Transcription mode for saaras models.
        language_code: Language of the input audio.
        upload_timeout: Seconds to wait for file upload.
        poll_timeout: Seconds to wait for the batch job to finish.

    Returns:
        Path to ``dialogues.json``.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    name = _base_name(audio_path)
    vocals_path = _resolve_vocals_audio(audio_path)
    output_dir = _resolve_output_dir(audio_path, output)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcription = transcribe_with_sarvam(
        vocals_path,
        model=model,
        mode=mode,
        language_code=language_code,
        upload_timeout=upload_timeout,
        poll_timeout=poll_timeout,
    )
    entries = _sarvam_diarized_entries(transcription)

    speaker_to_character: dict[str, str] = {}
    speaker_data: dict[str, dict[str, Any]] = {}
    diarized_entries: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for index, entry in enumerate(entries):
            speaker = str(entry.get("speaker_id", ""))
            character_id = _speaker_to_character_id(speaker, speaker_to_character)
            if character_id not in speaker_data:
                speaker_data[character_id] = {
                    "embeddings": [],
                    "pitch_values": [],
                    "speech_rates": [],
                    "emotions": [],
                }

            start = float(entry["start_time_seconds"])
            end = float(entry["end_time_seconds"])
            text = str(entry.get("transcript", "")).strip()

            segment_path = extract_segment_file(
                vocals_path, start, end, temp_path, index
            )
            emotion = detect_emotion(segment_path)
            embedding = extract_voice_embedding(segment_path)
            pitch_mean = measure_pitch_mean(segment_path)
            duration = max(end - start, 0.0)
            speech_rate = measure_speech_rate(text, duration)

            speaker_data[character_id]["embeddings"].append(embedding)
            if pitch_mean is not None:
                speaker_data[character_id]["pitch_values"].append(pitch_mean)
            speaker_data[character_id]["speech_rates"].append(speech_rate)
            speaker_data[character_id]["emotions"].append(emotion)

            diarized_entries.append(
                {
                    "transcript": text,
                    "start_time_seconds": round(start, 3),
                    "end_time_seconds": round(end, 3),
                    "speaker_id": character_id,
                    "emotion": emotion,
                }
            )

    characters = _build_characters(speaker_data)
    characters_path = output_dir / "characters.json"
    dialogues_path = output_dir / "dialogues.json"

    characters_path.write_text(
        json.dumps({"name": name, "characters": characters}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dialogues_path.write_text(
        json.dumps(
            {
                "name": name,
                "language_code": language_code,
                "source_audio": str(vocals_path),
                "diarization_model": model,
                "num_speakers": DEFAULT_NUM_SPEAKERS,
                "transcription_model": model,
                "request_id": transcription.get("request_id"),
                "transcript": transcription.get("transcript", ""),
                "diarized_transcript": {"entries": diarized_entries},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dialogues_path
