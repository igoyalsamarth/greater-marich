from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from pyannote.audio import Pipeline

from client.hf_client import get_hf_token
from lib.torch_device import get_torch_device

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
MIN_TURN_DURATION_SECONDS = 0.2
MERGE_GAP_SECONDS = 0.35


@dataclass(frozen=True)
class DiarizedTurn:
    start: float
    end: float
    speaker: str


@lru_cache
def _load_diarization_pipeline() -> Pipeline:
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=get_hf_token())
    pipeline.to(get_torch_device())
    return pipeline


def _iter_diarization_turns(diarization) -> list[DiarizedTurn]:
    turns: list[DiarizedTurn] = []

    if hasattr(diarization, "itertracks"):
        for turn, _track, speaker in diarization.itertracks(yield_label=True):
            turns.append(DiarizedTurn(start=turn.start, end=turn.end, speaker=str(speaker)))
    else:
        for turn, speaker in diarization:
            turns.append(DiarizedTurn(start=turn.start, end=turn.end, speaker=str(speaker)))

    filtered: list[DiarizedTurn] = []
    for turn in turns:
        if turn.end - turn.start >= MIN_TURN_DURATION_SECONDS:
            filtered.append(turn)
    return filtered


def _merge_adjacent_turns(turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
    if not turns:
        return []

    merged: list[DiarizedTurn] = [turns[0]]
    for turn in turns[1:]:
        previous = merged[-1]
        if turn.speaker == previous.speaker and turn.start - previous.end <= MERGE_GAP_SECONDS:
            merged[-1] = DiarizedTurn(
                start=previous.start,
                end=max(previous.end, turn.end),
                speaker=previous.speaker,
            )
        else:
            merged.append(turn)
    return merged


def diarize_audio(audio_path: Path) -> list[DiarizedTurn]:
    """Run pyannote speaker diarization on an audio file."""
    pipeline = _load_diarization_pipeline()
    with torch.inference_mode():
        output = pipeline(str(audio_path))

    if hasattr(output, "exclusive_speaker_diarization"):
        diarization = output.exclusive_speaker_diarization
    elif hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    else:
        diarization = output

    turns = _iter_diarization_turns(diarization)
    turns.sort(key=lambda turn: turn.start)
    return _merge_adjacent_turns(turns)
