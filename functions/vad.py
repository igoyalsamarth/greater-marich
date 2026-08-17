"""Silero VAD helpers for speech-tail detection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_VAD_SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def _load_silero_vad():
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except ImportError as exc:
        raise ImportError(
            "Silero VAD is required for Chatterbox tail trimming. "
            "Install extras with `uv sync --extra chatterbox`."
        ) from exc
    return load_silero_vad(), get_speech_timestamps, read_audio


def speech_timestamps(
    audio_path: Path,
    *,
    min_speech_ms: int = 80,
    min_silence_ms: int = 220,
    speech_pad_ms: int = 120,
) -> list[tuple[float, float]]:
    """Return (start, end) seconds for voiced regions in ``audio_path``."""
    model, get_speech_timestamps, read_audio = _load_silero_vad()
    wav = read_audio(str(audio_path), sampling_rate=_VAD_SAMPLE_RATE)
    stamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=_VAD_SAMPLE_RATE,
        return_seconds=True,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    regions: list[tuple[float, float]] = []
    for stamp in stamps:
        start = float(stamp["start"])
        end = float(stamp["end"])
        if end > start:
            regions.append((start, end))
    return regions


def merge_speech_regions(
    regions: list[tuple[float, float]],
    *,
    max_gap_seconds: float,
) -> list[tuple[float, float]]:
    """Join VAD chunks separated by a short gap so one phrase stays intact."""
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
