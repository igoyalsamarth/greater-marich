"""Audio volume helpers using FFmpeg."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

_VOLUME_DB_PATTERN = re.compile(r"mean_volume:\s*([-\d.]+)\s*dB")
_MIN_MEASURABLE_VOLUME_DB = -55.0
_MIN_GAIN_DB = -18.0
_MAX_GAIN_DB = 18.0


def measure_mean_volume_db(
    audio_path: Path,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float | None:
    """Measure mean volume in dB for an audio file or a time slice."""
    command = ["ffmpeg", "-hide_banner", "-nostats"]
    if start_seconds is not None:
        command.extend(["-ss", f"{start_seconds:.3f}"])
    if end_seconds is not None:
        command.extend(["-to", f"{end_seconds:.3f}"])
    command.extend(
        [
            "-i",
            str(audio_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
    )

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg volumedetect failed")

    volumes = [float(match) for match in _VOLUME_DB_PATTERN.findall(result.stderr)]
    if not volumes:
        return None

    volume = volumes[-1]
    if volume <= _MIN_MEASURABLE_VOLUME_DB:
        return None
    return volume


def apply_volume_gain(audio_path: Path, gain_db: float) -> None:
    """Apply a gain adjustment in dB to a WAV file in place."""
    if abs(gain_db) < 0.05:
        return

    gain_db = max(_MIN_GAIN_DB, min(_MAX_GAIN_DB, gain_db))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_path.parent) as tmp:
        tmp_path = Path(tmp.name)

    try:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-af",
            f"volume={gain_db:.2f}dB",
            str(tmp_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg volume adjust failed")
        tmp_path.replace(audio_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def match_volume_to_reference(
    audio_path: Path,
    reference_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, float] | None:
    """Adjust ``audio_path`` so its mean volume matches the reference slice."""
    reference_db = measure_mean_volume_db(
        reference_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    if reference_db is None:
        return None

    source_db = measure_mean_volume_db(audio_path)
    if source_db is None:
        return None

    gain_db = reference_db - source_db
    apply_volume_gain(audio_path, gain_db)
    return {
        "reference_volume_db": round(reference_db, 2),
        "source_volume_db": round(source_db, 2),
        "applied_gain_db": round(max(_MIN_GAIN_DB, min(_MAX_GAIN_DB, gain_db)), 2),
    }
