"""WavLM-large character analysis via Vox-Profile fine-tuned models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from functions.stt_features import _read_mono_wav
from lib.emotion_profile import EMOTION_LABELS
from lib.torch_device import get_torch_device
from lib.wavlm_loader import (
    load_demographics_model,
    load_emotion_model,
    release_wavlm_models,
    unload_demographics_model,
    unload_emotion_model,
)

SAMPLE_RATE = 16_000
MIN_MODEL_DURATION_SECONDS = 3.0
MAX_MODEL_DURATION_SECONDS = 15.0

SEX_LABELS = ("female", "male")
_EMOTION_MODEL_LABELS = (
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
    "Other",
)


def _prepare_waveform(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}")

    waveform = audio.astype(np.float32, copy=False)
    min_samples = int(MIN_MODEL_DURATION_SECONDS * SAMPLE_RATE)
    max_samples = int(MAX_MODEL_DURATION_SECONDS * SAMPLE_RATE)

    if waveform.size < min_samples:
        waveform = np.pad(waveform, (0, min_samples - waveform.size))
    if waveform.size > max_samples:
        waveform = waveform[:max_samples]

    return torch.from_numpy(waveform).unsqueeze(0).to(get_torch_device())


def _emotion_profile_from_logits(logits: torch.Tensor) -> dict[str, float]:
    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    profile = {
        label: float(probabilities[index])
        for index, label in enumerate(_EMOTION_MODEL_LABELS)
    }
    return {
        EMOTION_LABELS[index]: round(profile[label], 4)
        for index, label in enumerate(_EMOTION_MODEL_LABELS)
    }


def _sex_probabilities(logits: torch.Tensor) -> dict[str, float]:
    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return {
        SEX_LABELS[index]: float(probabilities[index])
        for index in range(len(SEX_LABELS))
    }


def _age_estimate_from_output(age_output: torch.Tensor) -> float:
    """Convert demographics age head output to years (0-100)."""
    if age_output.shape[-1] == 1:
        return round(float(age_output.reshape(-1)[0].item()) * 100, 1)

    probabilities = F.softmax(age_output, dim=-1).detach().cpu().numpy().reshape(-1)
    bin_centers = np.linspace(0.0, 1.0, num=probabilities.size)
    return round(float(np.dot(probabilities, bin_centers)) * 100, 1)


def _demographics_for_waveform(
    model: torch.nn.Module,
    waveform: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    with torch.inference_mode():
        age_output, sex_output = model(waveform)

    sex_probs = _sex_probabilities(sex_output)
    age_estimate = _age_estimate_from_output(age_output)
    return age_estimate, sex_probs


def _emotion_for_waveform(
    model: torch.nn.Module,
    waveform: torch.Tensor,
) -> tuple[dict[str, float], list[float]]:
    with torch.inference_mode():
        emotion_logits, embedding, *_ = model(waveform, return_feature=True)

    emotion_profile = _emotion_profile_from_logits(emotion_logits)
    vector = embedding.detach().cpu().numpy().reshape(-1)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return emotion_profile, vector.astype(float).tolist()


def _merge_analysis(
    *,
    age_estimate: float,
    sex_probs: dict[str, float],
    emotion_profile: dict[str, float],
    voice_embedding: list[float],
) -> dict[str, Any]:
    gender = max(sex_probs, key=sex_probs.get)
    return {
        "gender": gender,
        "gender_confidence": round(sex_probs[gender], 4),
        "sex_probabilities": sex_probs,
        "age_estimate": age_estimate,
        "emotion_profile": emotion_profile,
        "voice_embedding": voice_embedding,
    }


def analyze_segments(audio_paths: list[Path]) -> list[dict[str, Any]]:
    """Run demographics then emotion models, one head loaded at a time."""
    if not audio_paths:
        return []

    waveforms = []
    for audio_path in audio_paths:
        audio, sample_rate = _read_mono_wav(audio_path)
        waveforms.append(_prepare_waveform(audio, sample_rate))

    demographics_results: list[tuple[float, dict[str, float]]] = []
    demographics_model = load_demographics_model()
    for waveform in waveforms:
        demographics_results.append(_demographics_for_waveform(demographics_model, waveform))

    del demographics_model
    unload_demographics_model()

    emotion_results: list[tuple[dict[str, float], list[float]]] = []
    emotion_model = load_emotion_model()
    for waveform in waveforms:
        emotion_results.append(_emotion_for_waveform(emotion_model, waveform))

    del emotion_model
    unload_emotion_model()
    release_wavlm_models()

    return [
        _merge_analysis(
            age_estimate=age_estimate,
            sex_probs=sex_probs,
            emotion_profile=emotion_profile,
            voice_embedding=voice_embedding,
        )
        for (age_estimate, sex_probs), (emotion_profile, voice_embedding) in zip(
            demographics_results,
            emotion_results,
            strict=True,
        )
    ]


def analyze_segment(audio_path: Path) -> dict[str, Any]:
    """Run WavLM demographics and emotion models on a segment clip."""
    return analyze_segments([audio_path])[0]


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []

    matrix = np.asarray(embeddings, dtype=float)
    mean = matrix.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.astype(float).tolist()


def average_emotion_profiles(
    profiles: list[dict[str, float]],
) -> dict[str, float]:
    if not profiles:
        return {label: 0.0 for label in EMOTION_LABELS}

    averaged = {
        label: float(np.mean([profile.get(label, 0.0) for profile in profiles]))
        for label in EMOTION_LABELS
    }
    total = sum(averaged.values())
    if total > 0:
        averaged = {label: value / total for label, value in averaged.items()}
    return {label: round(value, 4) for label, value in averaged.items()}


def aggregate_gender(
    sex_probabilities: list[dict[str, float]],
) -> tuple[str, float]:
    if not sex_probabilities:
        return "unknown", 0.0

    female = float(np.mean([item.get("female", 0.0) for item in sex_probabilities]))
    male = float(np.mean([item.get("male", 0.0) for item in sex_probabilities]))
    if female >= male:
        return "female", round(female, 4)
    return "male", round(male, 4)
