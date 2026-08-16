"""Generate speech with Chatterbox Multilingual Hindi voice cloning."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from functions.audio import match_volume_to_reference
from functions.dialogues import dialogue_entries, load_dialogues
from functions.speech_shared import (
    entry_duration,
    entry_end,
    entry_speaker_id,
    entry_start,
    entry_text,
    resolve_characters_path,
    resolve_source_dialogues,
    segment_id,
    source_name,
    vocals_path_for_name,
    write_speech_mapping,
)
from functions.stt_features import (
    concat_audio_segments,
    extract_segment_file,
)
from functions.translate_models import load_character_personas
from lib.chatterbox_loader import (
    DEFAULT_MODEL,
    HINDI_LANGUAGE_ID,
    load_chatterbox_model,
    prepare_voice_conditionals,
    save_speech,
    synthesize,
)
from lib.constants import SPEECH_CHATTERBOX_DIR
from lib.emotion_profile import (
    dominant_emotion,
    emotion_delivery_hint,
    emotion_profile_from_entry,
)

MIN_PROMPT_DURATION_SECONDS = 1.0


def _speech_dir_for_name(name: str) -> Path:
    return SPEECH_CHATTERBOX_DIR / name


def _exaggeration_from_profile(emotion_profile: dict[str, float]) -> float:
    """Map emotion intensity to Chatterbox exaggeration (0.25–0.75)."""
    if not emotion_profile:
        return 0.5
    top_score = max(emotion_profile.values())
    return round(0.25 + min(float(top_score), 1.0) * 0.5, 3)


def _temperature_from_profile(emotion_profile: dict[str, float]) -> float:
    if not emotion_profile:
        return 0.8
    variability = (
        emotion_profile.get("surprise", 0.0)
        + emotion_profile.get("happiness", 0.0) * 0.6
        + emotion_profile.get("anger", 0.0) * 0.4
    )
    return round(0.75 + min(variability, 1.0) * 0.15, 3)


def _build_character_reference_audio(
    *,
    vocals_path: Path,
    source_entries: list[dict[str, Any]],
    speaker_id: str,
    output_path: Path,
    temp_dir: Path,
) -> dict[str, Any]:
    """Concatenate every vocal segment for a speaker into one cloning reference."""
    speaker_entries = [
        entry
        for entry in source_entries
        if entry_speaker_id(entry) == speaker_id and entry_duration(entry) > 0
    ]
    speaker_entries.sort(key=entry_start)

    if not speaker_entries:
        raise ValueError(f"No source dialogue segments found for {speaker_id!r}.")

    segment_paths: list[Path] = []
    source_transcripts: list[str] = []
    total_source_seconds = 0.0

    for index, entry in enumerate(speaker_entries):
        segment_paths.append(
            extract_segment_file(
                vocals_path,
                entry_start(entry),
                entry_end(entry),
                temp_dir,
                index,
            )
        )
        source_transcripts.append(entry_text(entry).strip())
        total_source_seconds += entry_duration(entry)

    duration_seconds = concat_audio_segments(segment_paths, output_path)

    return {
        "speaker_id": speaker_id,
        "reference_audio": str(output_path),
        "reference_vocals": str(vocals_path),
        "segment_count": len(segment_paths),
        "total_source_seconds": round(total_source_seconds, 3),
        "reference_duration_seconds": round(duration_seconds, 3),
        "source_transcripts": source_transcripts,
    }


def generate_chatterbox_speech(
    transcript_json: str | Path,
    output_dir: str | Path | None = None,
    *,
    model_id: str = DEFAULT_MODEL,
    language_id: str = HINDI_LANGUAGE_ID,
) -> Path:
    """Generate Hindi speech clips with Chatterbox voice cloning.

    Builds one zero-shot cloning reference per speaker by concatenating all of
    their source vocal segments from the separated vocals track, then synthesizes
    each translated line with per-line ``emotion_profile`` mapped to exaggeration.
    """
    transcript_path = Path(transcript_json).expanduser().resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_path}")

    data = load_dialogues(transcript_path)
    entries = dialogue_entries(data)
    source_entries = resolve_source_dialogues(data, transcript_path)
    if len(source_entries) != len(entries):
        raise RuntimeError(
            "Translated dialogues and source STT dialogues have different lengths. "
            "Re-run translate on the current STT output."
        )

    name = data.get("name") or source_name(transcript_path)
    speech_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else _speech_dir_for_name(name)
    )
    speech_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = vocals_path_for_name(name)
    language_code = data.get("language_code") or "hi-IN"
    characters_path = resolve_characters_path(data, transcript_path)
    personas = load_character_personas(characters_path)

    model = load_chatterbox_model(model_id)
    segments: list[dict[str, Any]] = []
    character_references: dict[str, dict[str, Any]] = {}
    prepared_speaker: str | None = None

    speaker_ids = sorted(
        {entry_speaker_id(entry) for entry in source_entries if entry_speaker_id(entry)}
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for speaker_id in speaker_ids:
            reference_path = speech_dir / f"_ref_{speaker_id}.wav"
            character_references[speaker_id] = _build_character_reference_audio(
                vocals_path=vocals_path,
                source_entries=source_entries,
                speaker_id=speaker_id,
                output_path=reference_path,
                temp_dir=temp_path / speaker_id,
            )
            persona = personas.get(speaker_id)
            if persona:
                character_references[speaker_id]["attributes"] = (
                    persona.attributes.model_dump()
                )
                character_references[speaker_id]["character_emotion_profile"] = (
                    persona.emotion_profile
                )

        for index, (entry, source_entry) in enumerate(
            zip(entries, source_entries, strict=True),
            start=1,
        ):
            seg_id = segment_id(index)
            audio_file = f"{seg_id}.wav"
            audio_path = speech_dir / audio_file

            hindi_text = entry_text(entry).strip()
            source_text = entry_text(source_entry).strip()
            speaker_id = entry_speaker_id(entry)
            emotion_profile = emotion_profile_from_entry(entry)
            delivery_hint = emotion_delivery_hint(emotion_profile)
            exaggeration = _exaggeration_from_profile(emotion_profile)
            temperature = _temperature_from_profile(emotion_profile)
            start_seconds = entry_start(entry)
            end_seconds = entry_end(entry)
            slot_seconds = entry_duration(entry)
            persona = personas.get(speaker_id)
            character_attributes = (
                persona.attributes.model_dump() if persona else None
            )

            reference_meta = character_references.get(speaker_id)
            reference_path = (
                Path(reference_meta["reference_audio"])
                if reference_meta
                else None
            )
            reference_duration = (
                float(reference_meta["reference_duration_seconds"])
                if reference_meta
                else 0.0
            )

            volume_match: dict[str, float] | None = None

            if (
                hindi_text
                and reference_path
                and reference_path.is_file()
                and reference_duration >= MIN_PROMPT_DURATION_SECONDS
            ):
                if prepared_speaker != speaker_id:
                    prepare_voice_conditionals(
                        model,
                        prompt_wav=reference_path,
                        exaggeration=exaggeration,
                    )
                    prepared_speaker = speaker_id

                speech, sample_rate = synthesize(
                    model,
                    text=hindi_text,
                    language_id=language_id,
                    exaggeration=exaggeration,
                    temperature=temperature,
                )
                save_speech(audio_path, speech, sample_rate)
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
                "character_attributes": character_attributes,
                "emotion_profile": emotion_profile,
                "emotion": dominant_emotion(emotion_profile),
                "delivery_hint": delivery_hint,
                "exaggeration": exaggeration,
                "temperature": temperature,
                "transcript": hindi_text,
                "source_transcript": source_text,
                "reference_audio": str(reference_path) if reference_path else None,
                "synthesis_mode": "chatterbox_clone",
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
        "characters_json": str(characters_path),
        "translation_model": data.get("translation_model"),
        "plot_summary": data.get("plot_summary"),
        "tts_model": model_id,
        "tts_engine": "chatterbox-hi",
        "reference_vocals": str(vocals_path),
        "character_references": character_references,
        "segments": segments,
    }

    return write_speech_mapping(
        mapping_path=speech_dir / f"{name}.json",
        mapping=mapping,
    )
