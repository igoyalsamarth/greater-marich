"""Separate vocals and instruments using a hybrid of two BS-RoFormer models."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from bs_roformer import BSRoformerSession

from lib.constants import SEPARATION_DIR

DEFAULT_BACKEND = "mlx"
VOCALS_MODEL = "roformer-model-bs-roformer-sw-by-jarredou"
INSTRUMENTAL_MODEL = "roformer-model-bs-roformer-vocals-resurrection-by-unwa"

CheckpointCallback = Callable[[str], None]


def _default_checkpoint(message: str) -> None:
    print(message, flush=True)


def _find_output_path(manifest, output_id: str) -> Path:
    for output in manifest.outputs:
        if output.output_id == output_id:
            return Path(output.output_path)
    raise RuntimeError(
        f"Model did not produce '{output_id}' stem. "
        f"Available: {[output.output_id for output in manifest.outputs]}"
    )


def _write_manifest(manifest_path: Path, data: dict[str, Any]) -> None:
    manifest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _infer(
    *,
    model: str,
    input_dir: Path,
    store_dir: Path,
    backend: str,
):
    store_dir.mkdir(parents=True, exist_ok=True)
    with BSRoformerSession(model_name=model, backend=backend) as session:
        return session.infer(input_dir, store_dir=store_dir)


def separate_audio(
    audio: str | Path,
    output_dir: str | Path | None = None,
    *,
    backend: str = DEFAULT_BACKEND,
    on_checkpoint: CheckpointCallback | None = None,
) -> Path:
    """Separate an audio file into vocals and instrumental using two BS-RoFormer models.

    Vocals come from the jarredou SW model; instrumental comes from unwa's vocals
    resurrection model (derived as mix minus vocals). Outputs
    ``<name>_vocals.wav`` and ``<name>_instrumental.wav`` under
    ``outputs/separation/<name>/``.

    After each model finishes, the corresponding stem and manifest checkpoint are
    written before the next model starts.

    Args:
        audio: Path to the source audio file (for example ``outputs/audio/video1.wav``).
        output_dir: Output directory. Defaults to ``outputs/separation/<name>/``.
        backend: Compute backend. Use ``mlx`` on Apple Silicon.
        on_checkpoint: Optional callback for progress messages between model runs.

    Returns:
        Path to the separation output directory.
    """
    checkpoint = on_checkpoint or _default_checkpoint

    audio_path = Path(audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    name = audio_path.stem
    separation_dir = (
        Path(output_dir).expanduser().resolve() if output_dir else SEPARATION_DIR / name
    )
    separation_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = separation_dir / f"{name}.json"
    vocals_dest = separation_dir / f"{name}_vocals.wav"
    instrumental_dest = separation_dir / f"{name}_instrumental.wav"

    manifest: dict[str, Any] = {
        "name": name,
        "source_audio": str(audio_path),
        "vocals_model": VOCALS_MODEL,
        "instrumental_model": INSTRUMENTAL_MODEL,
        "backend": backend,
        "status": "running",
        "outputs": [],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_tmp = Path(tmp_dir)
        input_dir = base_tmp / "input"
        input_dir.mkdir()
        shutil.copy2(audio_path, input_dir / audio_path.name)

        checkpoint(f"Running vocals separation ({VOCALS_MODEL})...")
        vocals_manifest = _infer(
            model=VOCALS_MODEL,
            input_dir=input_dir,
            store_dir=base_tmp / "vocals",
            backend=backend,
        )
        vocals_src = _find_output_path(vocals_manifest, "vocals")
        shutil.copy2(vocals_src, vocals_dest)

        manifest["status"] = "vocals_complete"
        manifest["outputs"] = [
            {
                "output_id": "vocals",
                "output_path": str(vocals_dest),
                "source_model": VOCALS_MODEL,
            }
        ]
        _write_manifest(manifest_path, manifest)
        checkpoint(f"Vocals complete: {vocals_dest}")

        checkpoint(f"Running instrumental separation ({INSTRUMENTAL_MODEL})...")
        instrumental_manifest = _infer(
            model=INSTRUMENTAL_MODEL,
            input_dir=input_dir,
            store_dir=base_tmp / "instrumental",
            backend=backend,
        )
        instrumental_src = _find_output_path(instrumental_manifest, "instrumental")
        shutil.copy2(instrumental_src, instrumental_dest)

        manifest["status"] = "complete"
        manifest["outputs"] = [
            *manifest["outputs"],
            {
                "output_id": "instrumental",
                "output_path": str(instrumental_dest),
                "source_model": INSTRUMENTAL_MODEL,
            },
        ]
        _write_manifest(manifest_path, manifest)
        checkpoint(f"Instrumental complete: {instrumental_dest}")

    checkpoint(f"Separation complete: {separation_dir}")
    return separation_dir
