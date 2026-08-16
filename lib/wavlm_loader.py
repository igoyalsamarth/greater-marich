"""Load Vox-Profile WavLM heads without redundant base-model downloads."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

from lib.torch_device import get_torch_device
from lib.vox_profile_source import load_demographics_wrapper, load_emotion_wrapper

WAVLM_MODEL_ID = "microsoft/wavlm-large"
DEMOGRAPHICS_MODEL = "tiantiaf/wavlm-large-age-sex"
EMOTION_MODEL = "tiantiaf/wavlm-large-categorical-emotion"

_ORIGINAL_WAVLM_LOADER = WavLMModel.from_pretrained
_ORIGINAL_PROCESSOR_LOADER = Wav2Vec2FeatureExtractor.from_pretrained
_shared_feature_extractor: Wav2Vec2FeatureExtractor | None = None


def shared_feature_extractor() -> Wav2Vec2FeatureExtractor:
    global _shared_feature_extractor
    if _shared_feature_extractor is None:
        _shared_feature_extractor = _ORIGINAL_PROCESSOR_LOADER(WAVLM_MODEL_ID)
    return _shared_feature_extractor


@contextmanager
def _skip_pretrained_wavlm_init() -> Iterator[None]:
    """Build wrapper shells from config; weights come from task checkpoints."""

    @classmethod
    def wavlm_from_config_only(cls, pretrained_model_name_or_path, *args, **kwargs):
        del args, kwargs
        config = WavLMConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config)

    @classmethod
    def shared_processor_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
        del cls, pretrained_model_name_or_path, args, kwargs
        return shared_feature_extractor()

    WavLMModel.from_pretrained = wavlm_from_config_only
    Wav2Vec2FeatureExtractor.from_pretrained = shared_processor_loader
    try:
        yield
    finally:
        WavLMModel.from_pretrained = _ORIGINAL_WAVLM_LOADER
        Wav2Vec2FeatureExtractor.from_pretrained = _ORIGINAL_PROCESSOR_LOADER


def _attach_shared_processor(model: torch.nn.Module) -> torch.nn.Module:
    model.processor = shared_feature_extractor()
    return model


@lru_cache
def load_demographics_model() -> torch.nn.Module:
    wrapper = load_demographics_wrapper()
    with _skip_pretrained_wavlm_init():
        model = wrapper.from_pretrained(DEMOGRAPHICS_MODEL)
    device = get_torch_device()
    model = _attach_shared_processor(model).to(device)
    model.eval()
    return model


@lru_cache
def load_emotion_model() -> torch.nn.Module:
    wrapper = load_emotion_wrapper()
    with _skip_pretrained_wavlm_init():
        model = wrapper.from_pretrained(EMOTION_MODEL)
    device = get_torch_device()
    model = _attach_shared_processor(model).to(device)
    model.eval()
    return model


def unload_demographics_model() -> None:
    load_demographics_model.cache_clear()


def unload_emotion_model() -> None:
    load_emotion_model.cache_clear()


def release_wavlm_models() -> None:
    unload_demographics_model()
    unload_emotion_model()
    gc.collect()
    device = get_torch_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
