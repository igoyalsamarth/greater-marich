"""Bootstrap the Vox-Profile model wrappers from GitHub."""

from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

VOX_PROFILE_REPO = "https://github.com/tiantiaf0627/vox-profile-release.git"
VOX_PROFILE_CACHE = Path.home() / ".cache" / "greater-marich" / "vox-profile-release"


def ensure_vox_profile_source() -> Path:
    src = VOX_PROFILE_CACHE / "src"
    marker = src / "model" / "age_sex" / "wavlm_demographics.py"
    if marker.is_file():
        return src

    VOX_PROFILE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if VOX_PROFILE_CACHE.exists():
        shutil.rmtree(VOX_PROFILE_CACHE)

    subprocess.run(
        ["git", "clone", "--depth", "1", VOX_PROFILE_REPO, str(VOX_PROFILE_CACHE)],
        check=True,
    )
    if not marker.is_file():
        raise RuntimeError(
            f"Vox-Profile source is missing expected file: {marker}"
        )
    return src


def _import_from_subpackage(subpackage: str, module_name: str) -> Any:
    src = ensure_vox_profile_source()
    package_path = str(src / "model" / subpackage)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    return __import__(module_name)


@lru_cache
def load_demographics_wrapper() -> type[Any]:
    module = _import_from_subpackage("age_sex", "wavlm_demographics")
    return module.WavLMWrapper


@lru_cache
def load_emotion_wrapper() -> type[Any]:
    module = _import_from_subpackage("emotion", "wavlm_emotion")
    return module.WavLMWrapper
