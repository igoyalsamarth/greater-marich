from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from funasr import AutoModel
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

EMOTION_MODEL_HF_REPO = "emotion2vec/emotion2vec_plus_large"
EMOTION_MODEL = "iic/emotion2vec_plus_large"
EMOTION_CACHE_DIRNAME = "models--emotion2vec--emotion2vec_plus_large"
_UNKNOWN_EMOTIONS = frozenset({"<unk>", "unk", "unknown"})
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def normalize_emotion(label: str) -> str:
    """Normalize Emotion2Vec labels to plain English emotion names."""
    cleaned = label.strip()
    if not cleaned or cleaned.lower() in _UNKNOWN_EMOTIONS:
        return "neutral"

    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", maxsplit=1)[-1].strip()

    cleaned = _CJK_PATTERN.sub("", cleaned).strip()
    if not cleaned or cleaned.lower() in _UNKNOWN_EMOTIONS:
        return "neutral"

    return cleaned.lower()


def _emotion_cache_hint() -> str:
    return (
        f"Delete the incomplete cache folder '{EMOTION_CACHE_DIRNAME}' "
        "under ~/.cache/huggingface/hub/ and retry."
    )


def _ensure_emotion_model_downloaded() -> str:
    try:
        return snapshot_download(
            repo_id=EMOTION_MODEL_HF_REPO,
            resume_download=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download Emotion2Vec+ large. "
            f"{_emotion_cache_hint()} Original error: {exc}"
        ) from exc


@lru_cache
def _load_emotion_model() -> AutoModel:
    _ensure_emotion_model_downloaded()
    try:
        return AutoModel(
            model=EMOTION_MODEL,
            hub="hf",
            disable_update=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load Emotion2Vec+ large. "
            f"{_emotion_cache_hint()} Original error: {exc}"
        ) from exc


def detect_emotion(audio_path: Path) -> str:
    """Detect the dominant emotion for an audio clip."""
    model = _load_emotion_model()
    result = model.generate(str(audio_path), granularity="utterance")
    if not result:
        return normalize_emotion("unknown")

    item = result[0]
    labels = item.get("labels") or []
    scores = item.get("scores") or []
    if not labels or not scores:
        return normalize_emotion("unknown")

    return normalize_emotion(
        str(labels[int(max(range(len(scores)), key=scores.__getitem__))])
    )
