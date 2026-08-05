"""Separate vocals and instruments from audio using BS-RoFormer (MLX on Mac)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from bs_roformer import BSRoformerSession

from lib.constants import SEPARATION_DIR

DEFAULT_BACKEND = "mlx"
# Vocals-only model: outputs vocals + derived instrumental (everything else).
DEFAULT_MODEL = "roformer-model-bs-roformer-vocals-resurrection-by-unwa"
KEEP_STEMS = frozenset({"vocals", "instrumental"})


def _keep_only_vocals_and_instrumental(
    separation_dir: Path,
    name: str,
    manifest,
) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []

    for output in manifest.outputs:
        if output.output_id in KEEP_STEMS:
            kept.append(output.as_dict())
        else:
            Path(output.output_path).unlink(missing_ok=True)

    for path in separation_dir.glob(f"{name}_*.wav"):
        if path.stem not in {f"{name}_vocals", f"{name}_instrumental"}:
            path.unlink(missing_ok=True)

    return kept


def separate_audio(
    audio: str | Path,
    output_dir: str | Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
) -> Path:
    """Separate an audio file into vocals and instrumental using BS-RoFormer.

    Outputs ``<name>_vocals.wav`` and ``<name>_instrumental.wav`` under
    ``separation/<name>/``.

    Args:
        audio: Path to the source audio file (for example ``audio/video1.wav``).
        output_dir: Output directory. Defaults to ``separation/<name>/``.
        model: BS-RoFormer registry model slug.
        backend: Compute backend. Use ``mlx`` on Apple Silicon.

    Returns:
        Path to the separation output directory.
    """
    audio_path = Path(audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    name = audio_path.stem
    separation_dir = (
        Path(output_dir).expanduser().resolve() if output_dir else SEPARATION_DIR / name
    )
    separation_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir)
        input_wav = input_dir / audio_path.name
        shutil.copy2(audio_path, input_wav)

        with BSRoformerSession(model_name=model, backend=backend) as session:
            manifest = session.infer(input_dir, store_dir=separation_dir)

    outputs = _keep_only_vocals_and_instrumental(separation_dir, name, manifest)

    manifest_path = separation_dir / f"{name}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": name,
                "source_audio": str(audio_path),
                "model": model,
                "backend": backend,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return separation_dir
