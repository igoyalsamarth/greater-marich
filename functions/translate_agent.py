from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from functions.translate_models import (
    PlotSummaryRequest,
    PlotSummaryResult,
    TranslationRequest,
    TranslationResult,
)

SUMMARY_SYSTEM_PROMPT = """You are a script analyst preparing context for a dubbing translation team.

You receive a JSON object with the full diarized transcript of a scene, speaker personas,
and scene metadata. The transcript may contain STT errors (missing punctuation, homophones,
merged words, filler words).

Your job:
- Infer the intended story, setting, and situation.
- Describe each speaker's role and relationship to others.
- Capture the emotional arc and tone across the scene.
- Flag likely transcription mistakes and what the speaker probably meant.
- Note where pauses or gaps between lines matter for pacing.
- Write a single cohesive plot_summary that translators can use for every line.

Return structured output with plot_summary only."""


TRANSLATION_SYSTEM_PROMPT = """You are a professional dubbing translator.

You receive a JSON object for one line in a dubbed scene. Each contextual line includes:
- dialogue: source transcript, speaker_id, emotion, and exact timing (start/end/duration)
- speaker_persona: gender, pitch, speech rate, and dominant emotions for THAT line's speaker
- translated_transcript: target-language text when that line is already translated
- gap_before_seconds: pause since the previous line ended
- speakability_hint: duration budget for the current line (on current_line only)

The request also includes:
- plot_summary: full-scene context and likely STT corrections
- line_position: where this line falls in the scene
- characters: all speaker personas in the scene
- previous_lines: up to two source lines before current_line (chronological), with
  translated_transcript filled in when already translated
- current_line: the line to translate
- next_lines: up to two source lines after current_line (chronological)

Rules:
1. Translate ONLY current_line.dialogue.transcript into the target language.
2. Use plot_summary to fix STT errors and infer intended meaning.
3. Use previous_lines and next_lines to keep dialogue connected and natural.
4. Match current_line.dialogue.emotion and current_line.speaker_persona delivery.
5. Honor current_line.speakability_hint — shorten or rephrase to fit the time slot.
6. Keep terminology and register consistent with translated_transcript on previous_lines.
7. Return structured output with translated_line containing only the dubbed line."""


def _agent_model_name(model: str) -> str:
    if ":" in model:
        return model
    return f"openai:{model}"


@lru_cache
def _get_summary_agent(model: str) -> Agent[None, PlotSummaryResult]:
    return Agent(
        _agent_model_name(model),
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        output_type=PlotSummaryResult,
    )


@lru_cache
def _get_translation_agent(model: str) -> Agent[None, TranslationResult]:
    return Agent(
        _agent_model_name(model),
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        output_type=TranslationResult,
    )


def summarize_plot(request: PlotSummaryRequest, *, model: str) -> str:
    """Summarize the full scene for use as translation context."""
    agent = _get_summary_agent(model)
    result = agent.run_sync(request.model_dump_json(indent=2))
    plot_summary = result.output.plot_summary.strip()
    if not plot_summary:
        raise RuntimeError("No plot summary returned for the scene")
    return plot_summary


def translate_line(request: TranslationRequest, *, model: str) -> str:
    """Translate one line using a Pydantic AI agent with structured I/O."""
    agent = _get_translation_agent(model)
    result = agent.run_sync(request.model_dump_json(indent=2))
    translated = result.output.translated_line.strip()
    if not translated:
        raise RuntimeError(
            f"No translation returned for dialogue index {request.current_line.dialogue.index}"
        )
    return translated
