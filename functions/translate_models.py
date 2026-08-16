from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from lib.emotion_profile import (
    compact_emotion_profile,
    dominant_emotion,
    emotion_delivery_hint,
    emotion_profile_from_entry,
    legacy_emotion_profile,
)

CONTEXT_WINDOW = 2


class CharacterAttributes(BaseModel):
  """Signal and WavLM-derived speaker traits."""

  model_config = ConfigDict(extra="ignore")

  gender: str = "unknown"
  gender_confidence: float | None = None
  age_estimate: float | None = None
  pitch_mean: float | None = None
  pitch_range: float | None = None
  speech_rate: float = 0.0
  energy: float | None = None


class CharacterPersona(BaseModel):
    """Speaker profile used for dubbing context (no voice embeddings)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str | None = None
    attributes: CharacterAttributes = Field(default_factory=CharacterAttributes)
    emotion_profile: dict[str, float] = Field(default_factory=dict)

    @property
    def gender(self) -> str:
        return self.attributes.gender

    @property
    def pitch_mean(self) -> float | None:
        return self.attributes.pitch_mean

    @property
    def speech_rate(self) -> float:
        return self.attributes.speech_rate

    @property
    def dominant_emotions(self) -> list[str]:
        if not self.emotion_profile:
            return ["neutral"]
        ranked = sorted(
            self.emotion_profile.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [emotion for emotion, _score in ranked[:3]]

    @computed_field
    @property
    def dominant_emotion(self) -> str:
        return dominant_emotion(self.emotion_profile)

    @computed_field
    @property
    def significant_emotions(self) -> dict[str, float]:
        return compact_emotion_profile(self.emotion_profile)

    @computed_field
    @property
    def delivery_hint(self) -> str:
        return emotion_delivery_hint(self.emotion_profile)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        attributes = data.get("attributes")
        if attributes is None and any(
            key in data
            for key in ("gender", "pitch_mean", "speech_rate", "energy", "pitch_range", "age_estimate")
        ):
            data = {
                **data,
                "attributes": {
                    "gender": data.get("gender", "unknown"),
                    "gender_confidence": data.get("gender_confidence"),
                    "age_estimate": data.get("age_estimate"),
                    "pitch_mean": data.get("pitch_mean"),
                    "pitch_range": data.get("pitch_range"),
                    "speech_rate": data.get("speech_rate", 0.0),
                    "energy": data.get("energy"),
                },
            }

        if not data.get("emotion_profile") and data.get("dominant_emotions"):
            emotions = data["dominant_emotions"]
            if isinstance(emotions, list) and emotions:
                profile = {str(emotion): 0.0 for emotion in emotions}
                profile[str(emotions[0])] = 1.0
                data["emotion_profile"] = profile

        return data


class DialogueLine(BaseModel):
    """A single diarized line with timing and delivery metadata."""

    index: int
    transcript: str
    speaker_id: str
    emotion_profile: dict[str, float] = Field(default_factory=dict)
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float

    @computed_field
    @property
    def dominant_emotion(self) -> str:
        return dominant_emotion(self.emotion_profile)

    @computed_field
    @property
    def significant_emotions(self) -> dict[str, float]:
        return compact_emotion_profile(self.emotion_profile)

    @computed_field
    @property
    def delivery_hint(self) -> str:
        return emotion_delivery_hint(self.emotion_profile)

    @property
    def emotion(self) -> str:
        return self.dominant_emotion

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_emotion(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if not data.get("emotion_profile") and data.get("emotion"):
            data["emotion_profile"] = legacy_emotion_profile(str(data["emotion"]))
        return data


class EnrichedDialogueLine(BaseModel):
    """Dialogue line with speaker persona and optional target-language text."""

    dialogue: DialogueLine
    speaker_persona: CharacterPersona
    translated_transcript: str | None = Field(
        default=None,
        description="Target-language transcript when this line is already translated.",
    )
    gap_before_seconds: float | None = Field(
        default=None,
        description="Silence gap before this line starts (from the prior line ending).",
    )
    speakability_hint: str | None = Field(
        default=None,
        description="Guidance for fitting speech into the allotted duration.",
    )
    delivery_hint: str | None = Field(
        default=None,
        description=(
            "How this specific line should sound, combining line emotion_profile "
            "with the speaker's baseline delivery."
        ),
    )


class LinePosition(BaseModel):
    """Where the current line sits in the full scene."""

    index: int
    total: int
    is_first: bool
    is_last: bool


class SceneMetadata(BaseModel):
    """Scene-level metadata from the STT output."""

    model_config = ConfigDict(extra="ignore")

    name: str
    language_code: str
    source_audio: str | None = None
    diarization_model: str | None = None
    transcription_model: str | None = None
    num_speakers: int | None = None
    request_id: str | None = None


class PlotSummaryRequest(BaseModel):
    """Full-scene context passed to the plot summarization agent."""

    source_language_code: str
    scene: SceneMetadata
    characters: list[CharacterPersona]
    dialogues: list[DialogueLine]


class PlotSummaryResult(BaseModel):
    """Structured plot summary used to guide line-by-line translation."""

    plot_summary: str = Field(
        ...,
        description=(
            "A concise plot summary covering setting, character roles and relationships, "
            "emotional arc, and overall tone. Note likely STT errors and inferred intent."
        ),
    )


class TranslationRequest(BaseModel):
    """Structured input passed to the translation agent."""

    source_language_code: str
    target_language_code: str
    plot_summary: str
    line_position: LinePosition
    scene: SceneMetadata
    characters: list[CharacterPersona]
    previous_lines: list[EnrichedDialogueLine] = Field(
        default_factory=list,
        description="Up to two source lines before current_line (chronological order).",
    )
    current_line: EnrichedDialogueLine
    next_lines: list[EnrichedDialogueLine] = Field(
        default_factory=list,
        description="Up to two source lines after current_line (chronological order).",
    )


class TranslationResult(BaseModel):
    """Structured output from the translation agent."""

    translated_line: str = Field(
        ...,
        description=(
            "The dubbed translation of current_line only, in the target language, "
            "matching tone, emotion, and speakable duration."
        ),
    )


def translated_entry_from_source(
    entry: dict[str, Any],
    translated_text: str,
) -> dict[str, Any]:
    """Build a diarized entry in the same shape as STT dialogues.json."""
    return {
        "transcript": translated_text,
        "start_time_seconds": round(
            float(entry.get("start_time_seconds", entry.get("start", 0))), 3
        ),
        "end_time_seconds": round(
            float(entry.get("end_time_seconds", entry.get("end", 0))), 3
        ),
        "speaker_id": str(entry.get("speaker_id") or entry.get("character_id") or ""),
        "emotion_profile": emotion_profile_from_entry(entry),
    }


def dialogue_line_from_entry(index: int, entry: dict[str, Any]) -> DialogueLine:
    start = float(entry.get("start_time_seconds", entry.get("start", 0)))
    end = float(entry.get("end_time_seconds", entry.get("end", 0)))
    return DialogueLine(
        index=index,
        transcript=str(entry.get("transcript") or entry.get("text") or ""),
        speaker_id=str(entry.get("speaker_id") or entry.get("character_id") or ""),
        emotion_profile=emotion_profile_from_entry(entry),
        start_time_seconds=round(start, 3),
        end_time_seconds=round(end, 3),
        duration_seconds=round(max(end - start, 0.0), 3),
    )


def _gap_before_seconds(
    entries: list[dict[str, Any]],
    index: int,
) -> float | None:
    if index <= 0:
        return None
    previous_end = float(
        entries[index - 1].get("end_time_seconds", entries[index - 1].get("end", 0))
    )
    current_start = float(
        entries[index].get("start_time_seconds", entries[index].get("start", 0))
    )
    return round(max(current_start - previous_end, 0.0), 3)


def _delivery_hint(dialogue: DialogueLine, persona: CharacterPersona) -> str:
    line_hint = dialogue.delivery_hint
    persona_hint = persona.delivery_hint
    if line_hint == persona_hint:
        return line_hint
    return (
        f"Line delivery: {line_hint} "
        f"Speaker baseline across the scene: {persona_hint}"
    )


def _speakability_hint(dialogue: DialogueLine, persona: CharacterPersona) -> str:
    if dialogue.duration_seconds <= 0:
        return "Keep the line as short as possible."
    if persona.speech_rate > 0:
        target_words = max(int(dialogue.duration_seconds * persona.speech_rate), 1)
        return (
            f"Fit within {dialogue.duration_seconds:.2f}s "
            f"(~{target_words} words at this speaker's pace)."
        )
    return f"Fit within {dialogue.duration_seconds:.2f}s when spoken aloud."


def enriched_dialogue_line(
    *,
    index: int,
    entry: dict[str, Any],
    personas: dict[str, CharacterPersona],
    entries: list[dict[str, Any]],
    translated_transcript: str | None = None,
    include_speakability_hint: bool = False,
) -> EnrichedDialogueLine:
    dialogue = dialogue_line_from_entry(index, entry)
    persona = persona_for_speaker(personas, dialogue.speaker_id)
    return EnrichedDialogueLine(
        dialogue=dialogue,
        speaker_persona=persona,
        translated_transcript=translated_transcript,
        gap_before_seconds=_gap_before_seconds(entries, index),
        speakability_hint=(
            _speakability_hint(dialogue, persona) if include_speakability_hint else None
        ),
        delivery_hint=_delivery_hint(dialogue, persona),
    )


def scene_metadata_from_stt(data: dict[str, Any]) -> SceneMetadata:
    return SceneMetadata(
        name=str(data.get("name") or ""),
        language_code=str(data.get("language_code") or "auto"),
        source_audio=data.get("source_audio"),
        diarization_model=data.get("diarization_model"),
        transcription_model=data.get("transcription_model"),
        num_speakers=data.get("num_speakers"),
        request_id=data.get("request_id"),
    )


def load_character_personas(characters_path: Path) -> dict[str, CharacterPersona]:
    if not characters_path.is_file():
        raise FileNotFoundError(
            f"Characters file not found: {characters_path}. "
            "Run STT first to generate characters.json."
        )

    data = json.loads(characters_path.read_text(encoding="utf-8"))
    personas: dict[str, CharacterPersona] = {}
    for character in data.get("characters") or []:
        persona = CharacterPersona.model_validate(character)
        personas[persona.id] = persona
    return personas


def build_plot_summary_request(
    *,
    data: dict[str, Any],
    entries: list[dict[str, Any]],
    personas: dict[str, CharacterPersona],
    source_language_code: str,
) -> PlotSummaryRequest:
    return PlotSummaryRequest(
        source_language_code=source_language_code,
        scene=scene_metadata_from_stt(data),
        characters=sorted(personas.values(), key=lambda persona: persona.id),
        dialogues=[
            dialogue_line_from_entry(index, entry)
            for index, entry in enumerate(entries)
        ],
    )


def persona_for_speaker(
    personas: dict[str, CharacterPersona],
    speaker_id: str,
) -> CharacterPersona:
    if speaker_id in personas:
        return personas[speaker_id]
    return CharacterPersona(
        id=speaker_id or "unknown",
        attributes=CharacterAttributes(),
        emotion_profile={"neutral": 1.0},
    )


def neighbor_lines(
    *,
    entries: list[dict[str, Any]],
    index: int,
    personas: dict[str, CharacterPersona],
    direction: Literal["previous", "next"],
    translated_texts: list[str | None],
    window: int = CONTEXT_WINDOW,
) -> list[EnrichedDialogueLine]:
    """Return up to ``window`` neighboring lines in chronological order."""
    lines: list[EnrichedDialogueLine] = []
    if direction == "previous":
        indices = [
            neighbor_index
            for offset in range(1, window + 1)
            if (neighbor_index := index - offset) >= 0
        ]
        indices.reverse()
    else:
        indices = [
            neighbor_index
            for offset in range(1, window + 1)
            if (neighbor_index := index + offset) < len(entries)
        ]

    for neighbor_index in indices:
        translated = translated_texts[neighbor_index] if direction == "previous" else None
        lines.append(
            enriched_dialogue_line(
                index=neighbor_index,
                entry=entries[neighbor_index],
                personas=personas,
                entries=entries,
                translated_transcript=translated,
            )
        )
    return lines
