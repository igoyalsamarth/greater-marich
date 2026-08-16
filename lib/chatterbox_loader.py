"""Load ResembleAI Chatterbox Multilingual Hindi for voice cloning."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

DEFAULT_MODEL = "ResembleAI/Chatterbox-Multilingual-hi"
BASE_REPO = "ResembleAI/chatterbox"
HINDI_LANGUAGE_ID = "hi"

_CACHE_ROOT = Path.home() / ".cache" / "greater-marich" / "chatterbox-hi"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_hindi_checkpoint(model_id: str = DEFAULT_MODEL) -> Path:
    if model_id != DEFAULT_MODEL:
        raise ValueError(
            f"Only {DEFAULT_MODEL!r} is supported; got {model_id!r}."
        )

    if _checkpoint_ready(_CACHE_ROOT):
        return _CACHE_ROOT

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    base_dir = Path(
        snapshot_download(
            repo_id=BASE_REPO,
            repo_type="model",
            allow_patterns=["ve.pt"],
            token=os.getenv("HF_TOKEN"),
        )
    )
    hi_dir = Path(
        snapshot_download(
            repo_id=DEFAULT_MODEL,
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
        )
    )

    shutil.copy2(base_dir / "ve.pt", _CACHE_ROOT / "ve.pt")
    for name in (
        "t3_hi.safetensors",
        "s3gen_v3.safetensors",
        "grapheme_mtl_merged_expanded_v1.json",
    ):
        shutil.copy2(hi_dir / name, _CACHE_ROOT / name)

    return _CACHE_ROOT


def _checkpoint_ready(ckpt_dir: Path) -> bool:
    return all(
        (ckpt_dir / name).is_file()
        for name in (
            "ve.pt",
            "t3_hi.safetensors",
            "s3gen_v3.safetensors",
            "grapheme_mtl_merged_expanded_v1.json",
        )
    )


def _load_hindi_model(ckpt_dir: Path, device: str) -> Any:
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    map_location = torch.device("cpu") if device in {"cpu", "mps"} else None

    ve = VoiceEncoder()
    ve.load_state_dict(
        torch.load(ckpt_dir / "ve.pt", map_location=map_location, weights_only=True)
    )
    ve.to(device).eval()

    t3 = T3(T3Config.multilingual())
    t3_state = load_safetensors(ckpt_dir / "t3_hi.safetensors")
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(
        load_safetensors(ckpt_dir / "s3gen_v3.safetensors"),
        strict=False,
    )
    s3gen.to(device).eval()

    tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device)


def download_chatterbox_model(model_id: str = DEFAULT_MODEL) -> Path:
    """Download and cache Chatterbox Hindi weights from Hugging Face."""
    return _ensure_hindi_checkpoint(model_id)


@lru_cache
def load_chatterbox_model(model_id: str = DEFAULT_MODEL) -> Any:
    device = _pick_device()
    ckpt_dir = _ensure_hindi_checkpoint(model_id)
    return _load_hindi_model(ckpt_dir, device)


def prepare_voice_conditionals(
    model: Any,
    *,
    prompt_wav: Path,
    exaggeration: float = 0.5,
) -> None:
    model.prepare_conditionals(str(prompt_wav), exaggeration=exaggeration)


def synthesize(
    model: Any,
    *,
    text: str,
    language_id: str = HINDI_LANGUAGE_ID,
    exaggeration: float = 0.5,
    prompt_wav: Path | None = None,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    repetition_penalty: float = 2.0,
    min_p: float = 0.05,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, int]:
    wav = model.generate(
        text,
        language_id=language_id,
        audio_prompt_path=str(prompt_wav) if prompt_wav else None,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
    )
    return wav, int(model.sr)


def save_speech(
    audio_path: Path,
    speech: torch.Tensor,
    sample_rate: int,
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(audio_path), speech.cpu(), sample_rate)
