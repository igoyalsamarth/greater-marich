from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Audio, Inference, Model

from client.hf_client import get_hf_token
from lib.torch_device import get_torch_device

EMBEDDING_MODEL = "pyannote/embedding"
EMBEDDING_SAMPLE_RATE = 16000
MIN_EMBEDDING_DURATION_SECONDS = 1.0


@lru_cache
def _load_embedding_inference() -> Inference:
    model = Model.from_pretrained(EMBEDDING_MODEL, token=get_hf_token())
    inference = Inference(model, window="whole")
    inference.to(get_torch_device())
    return inference


@lru_cache
def _embedding_audio_loader() -> Audio:
    return Audio(sample_rate=EMBEDDING_SAMPLE_RATE, mono="downmix")


def _waveform_input(audio_path: Path) -> dict[str, torch.Tensor | int]:
    waveform, sample_rate = _embedding_audio_loader()(str(audio_path))
    min_samples = int(MIN_EMBEDDING_DURATION_SECONDS * sample_rate)
    if waveform.shape[-1] < min_samples:
        waveform = torch.nn.functional.pad(waveform, (0, min_samples - waveform.shape[-1]))
    return {"waveform": waveform, "sample_rate": sample_rate}


def extract_voice_embedding(audio_path: Path) -> list[float]:
    """Extract a speaker embedding from an audio clip."""
    inference = _load_embedding_inference()
    with torch.inference_mode():
        embedding = inference(_waveform_input(audio_path))

    vector = np.asarray(embedding).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(float).tolist()


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []

    matrix = np.asarray(embeddings, dtype=float)
    mean = matrix.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.astype(float).tolist()
