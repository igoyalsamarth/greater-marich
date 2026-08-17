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
# Chatterbox language ids are ISO 639-1 ("hi"). BCP-47 tags like "hi-IN"
# are rejected by the tokenizer and produce unstable Hindi speech.
HINDI_LANGUAGE_ID = "hi"
# Stable cloning defaults. Higher temperature/CFG and emotion-scaled
# exaggeration cause random noises and end-of-line glitches, especially
# when the reference is English and the target is Hindi.
STABLE_EXAGGERATION = 0.5
# 0.4 sounded robotic; 0.8 added glitches. 0.6 keeps some natural variation.
STABLE_TEMPERATURE = 0.6
# CFG is Chatterbox's pace control. 0.0 is the cross-language tip but makes
# Hindi slow and word-by-word. 0.4 restores rhythm while limiting English pull.
CROSS_LANGUAGE_CFG_WEIGHT = 0.4

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


class _IdentityWatermarker:
    """Pass-through stand-in for Resemble Perth (avoids end-of-clip artifacts)."""

    def apply_watermark(self, wav, sample_rate=None, **kwargs):
        return wav


def _patch_perth_pkg_resources() -> None:
    """Load Perth without importing deprecated ``pkg_resources``.

    ``resemble-perth`` resolves ``pretrained/`` via ``pkg_resources.resource_filename``,
    which warns on Setuptools 81+ and will break when that API is removed.
    """
    import importlib.util
    import sys
    from types import ModuleType

    if "perth.perth_net" in sys.modules:
        return

    spec = importlib.util.find_spec("perth")
    if spec is None or not spec.origin:
        return

    perth_net_dir = Path(spec.origin).resolve().parent / "perth_net"
    if not perth_net_dir.is_dir():
        return

    module = ModuleType("perth.perth_net")
    module.__file__ = str(perth_net_dir / "__init__.py")
    module.__path__ = [str(perth_net_dir)]
    module.__package__ = "perth.perth_net"
    module.PREPACKAGED_MODELS_DIR = str(perth_net_dir / "pretrained")
    sys.modules["perth.perth_net"] = module
    parent = sys.modules.get("perth")
    if parent is not None:
        parent.perth_net = module


def _patch_diffusers_lora_linear() -> None:
    """Use ``nn.Linear`` instead of deprecated ``LoRACompatibleLinear``.

    Chatterbox's SnakeBeta treats that class as a plain linear layer (no LoRA
    weights). Instantiating it still emits a PEFT FutureWarning from diffusers 0.29.
    """
    import sys

    import torch.nn as nn
    import diffusers.models.lora as diffusers_lora

    diffusers_lora.LoRACompatibleLinear = nn.Linear
    transformer = sys.modules.get("chatterbox.models.s3gen.matcha.transformer")
    if transformer is not None:
        transformer.LoRACompatibleLinear = nn.Linear


def _load_hindi_model(ckpt_dir: Path, device: str) -> Any:
    _patch_perth_pkg_resources()
    _patch_diffusers_lora_linear()

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
    tts = ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device)
    # Perth's neural watermark can leave residual vowels/noise after the sentence.
    tts.watermarker = _IdentityWatermarker()
    return tts


def download_chatterbox_model(model_id: str = DEFAULT_MODEL) -> Path:
    """Download and cache Chatterbox Hindi weights from Hugging Face."""
    return _ensure_hindi_checkpoint(model_id)


@lru_cache
def load_chatterbox_model(model_id: str = DEFAULT_MODEL) -> Any:
    device = _pick_device()
    ckpt_dir = _ensure_hindi_checkpoint(model_id)
    return _load_hindi_model(ckpt_dir, device)


def normalize_chatterbox_language_id(language_id: str | None) -> str:
    """Return a Chatterbox language id such as ``hi``, never ``hi-IN``."""
    raw = (language_id or HINDI_LANGUAGE_ID).strip().replace("_", "-")
    code = raw.split("-", 1)[0].lower()
    return code or HINDI_LANGUAGE_ID


def prepare_voice_conditionals(
    model: Any,
    *,
    prompt_wav: Path,
    exaggeration: float = STABLE_EXAGGERATION,
) -> None:
    """Cache speaker conditionals. ``exaggeration`` is baked into this cache.

    Synthesize with the same exaggeration; a different value would only patch
    ``emotion_adv`` without rebuilding the voice prompt, which is unstable.
    """
    model.prepare_conditionals(str(prompt_wav), exaggeration=exaggeration)


def synthesize(
    model: Any,
    *,
    text: str,
    language_id: str = HINDI_LANGUAGE_ID,
    exaggeration: float = STABLE_EXAGGERATION,
    prompt_wav: Path | None = None,
    cfg_weight: float = CROSS_LANGUAGE_CFG_WEIGHT,
    temperature: float = STABLE_TEMPERATURE,
    repetition_penalty: float = 2.0,
    min_p: float = 0.1,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, int]:
    wav = model.generate(
        text,
        language_id=normalize_chatterbox_language_id(language_id),
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
