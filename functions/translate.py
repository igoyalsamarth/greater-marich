"""Translate diarized STT output using Sarvam's chat LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from client.sarvam_client import get_sarvam_client
from lib.constants import TRANSLATE_DIR

DEFAULT_TARGET_LANGUAGE = "hi-IN"
DEFAULT_LLM_MODEL = "sarvam-105b"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a professional dubbing translator. Translate spoken dialog for voice-over dubbing.

Rules:
- Translate only the "Current line" into the target language.
- Preserve meaning, tone, and speaker intent.
- The dubbed line must be speakable within the given time limit.
- If needed, rephrase to be shorter while keeping the meaning natural.
- Output ONLY the translated line, with no quotes, labels, or explanation."""


def _resolve_output_path(stt_path: Path, output: str | Path | None) -> Path:
    if output is None:
        return TRANSLATE_DIR / f"{stt_path.stem}.json"

    output_path = Path(output).expanduser().resolve()
    if output_path.suffix:
        return output_path
    return output_path / f"{stt_path.stem}.json"


def _dialog_duration(entry: dict[str, Any]) -> float:
    return float(entry["end_time_seconds"]) - float(entry["start_time_seconds"])


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def _context_line(entries: list[dict[str, Any]], index: int) -> str:
    if index < 0 or index >= len(entries):
        return "(none)"
    text = entries[index].get("transcript", "").strip()
    return text or "(none)"


def _build_translation_prompt(
    *,
    previous_line: str,
    current_line: str,
    next_line: str,
    time_limit: str,
    source_language_code: str,
    target_language_code: str,
) -> str:
    return f"""Source language: {source_language_code}
Target language: {target_language_code}

Previous line: {previous_line}
Current line: {current_line}
Next line: {next_line}

Time limit: {time_limit} (the translation must fit when spoken aloud)

Translate the current line."""


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def _extract_reasoning_translation(reasoning_content: str) -> str | None:
    patterns = [
        r"Final output:\s*[`\"']([^`\"']+)[`\"']",
        r"Final output:\s*([^\n]+)",
        r"translated line:\s*[`\"']([^`\"']+)[`\"']",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, reasoning_content, flags=re.IGNORECASE)
        if matches:
            candidate = _strip_wrapping_quotes(matches[-1])
            if candidate:
                return candidate

    backtick_matches = re.findall(r"`([^`]+)`", reasoning_content)
    for candidate in reversed(backtick_matches):
        candidate = candidate.strip()
        if candidate and not candidate.startswith("http"):
            return candidate
    return None


def _extract_assistant_text(message: Any) -> str | None:
    content = getattr(message, "content", None)
    if content and content.strip():
        return content.strip()

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return _extract_reasoning_translation(reasoning_content)
    return None


def _fallback_translate(
    text: str,
    *,
    source_language_code: str,
    target_language_code: str,
) -> str:
    client = get_sarvam_client()
    response = client.text.translate(
        input=text,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )
    return response.translated_text


def _translate_entry(
    entries: list[dict[str, Any]],
    index: int,
    *,
    source_language_code: str,
    target_language_code: str,
    model: str,
) -> str:
    entry = entries[index]
    text = entry.get("transcript", "")
    if not text.strip():
        return text

    prompt = _build_translation_prompt(
        previous_line=_context_line(entries, index - 1),
        current_line=text.strip(),
        next_line=_context_line(entries, index + 1),
        time_limit=_format_duration(_dialog_duration(entry)),
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )

    client = get_sarvam_client()
    response = client.chat.completions(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    choice = response.choices[0]
    translated = _extract_assistant_text(choice.message)
    if not translated:
        if choice.finish_reason == "length":
            translated = _fallback_translate(
                text.strip(),
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
        else:
            raise RuntimeError(
                f"No translation returned for entry {index + 1} "
                f"(finish_reason={choice.finish_reason})"
            )
    return _strip_wrapping_quotes(translated)


def translate_diarized_transcript(
    stt_path: str | Path,
    output: str | Path | None = None,
    *,
    source_language_code: str | None = None,
    target_language_code: str = DEFAULT_TARGET_LANGUAGE,
    model: str = DEFAULT_LLM_MODEL,
) -> Path:
    """Translate each entry in an STT JSON's diarized transcript.

    Args:
        stt_path: Path to the STT output JSON file.
        output: Output JSON file or directory. Defaults to ``translate/<name>.json``.
        source_language_code: Source language. Defaults to the STT file's ``language_code``.
        target_language_code: Target language for translation.
        model: Sarvam chat model to use for translation.

    Returns:
        Path to the saved translated JSON file.
    """
    stt_path = Path(stt_path).expanduser().resolve()
    if not stt_path.is_file():
        raise FileNotFoundError(f"STT file not found: {stt_path}")

    data: dict[str, Any] = json.loads(stt_path.read_text(encoding="utf-8"))
    diarized = data.get("diarized_transcript") or {}
    entries = diarized.get("entries") or []
    if not entries:
        raise RuntimeError("No diarized_transcript entries found in STT JSON")

    source_language = source_language_code or data.get("language_code") or "auto"

    translated_entries = []
    for index, entry in enumerate(entries):
        translated_entries.append(
            {
                **entry,
                "transcript": _translate_entry(
                    entries,
                    index,
                    source_language_code=source_language,
                    target_language_code=target_language_code,
                    model=model,
                ),
            }
        )

    output_data = {**data}
    output_data["diarized_transcript"] = {"entries": translated_entries}
    output_data["language_code"] = target_language_code
    output_data["transcript"] = " ".join(
        entry["transcript"] for entry in translated_entries if entry["transcript"]
    )

    timestamps = output_data.get("timestamps")
    if isinstance(timestamps, dict) and "words" in timestamps:
        timestamps["words"] = [entry["transcript"] for entry in translated_entries]

    output_path = _resolve_output_path(stt_path, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
