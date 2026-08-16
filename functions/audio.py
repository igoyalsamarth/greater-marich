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
_MIN_ATEMPO = 0.5
_MAX_ATEMPO = 2.0
_DURATION_FIT_TOLERANCE = 0.08
_MAX_SPEECH_BUS_GAIN_DB = 12.0


def clamp_volume_gain_db(gain_db: float) -> float:
    return max(_MIN_GAIN_DB, min(_MAX_GAIN_DB, gain_db))


def compute_volume_gain_db(
    reference_db: float | None,
    source_db: float | None,
) -> float | None:
    """Return gain in dB to apply to ``source`` so it matches ``reference``."""
    if reference_db is None or source_db is None:
        return None
    return clamp_volume_gain_db(reference_db - source_db)


def speech_bus_gain_db(
    reference_vocals: Path,
    instrumental: Path,
) -> float:
    """Lift speech when the instrumental stem is hotter than the vocals stem."""
    vocals_db = measure_mean_volume_db(reference_vocals)
    instrumental_db = measure_mean_volume_db(instrumental)
    if vocals_db is None or instrumental_db is None:
        return 0.0
    return max(0.0, min(_MAX_SPEECH_BUS_GAIN_DB, instrumental_db - vocals_db))


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

    gain_db = compute_volume_gain_db(reference_db, source_db)
    if gain_db is None:
        return None

    apply_volume_gain(audio_path, gain_db)
    return {
        "reference_volume_db": round(reference_db, 2),
        "source_volume_db": round(source_db, 2),
        "applied_gain_db": round(gain_db, 2),
    }


def measure_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed to probe audio duration")
    return float(result.stdout.strip())


def fit_audio_to_duration(
    audio_path: Path,
    target_seconds: float,
) -> dict[str, float | bool] | None:
    """Time-stretch audio toward a target slot duration using FFmpeg atempo."""
    if target_seconds <= 0:
        return None

    source_seconds = measure_audio_duration(audio_path)
    if source_seconds <= 0:
        return None

    ratio = source_seconds / target_seconds
    if abs(ratio - 1.0) <= _DURATION_FIT_TOLERANCE:
        return {
            "source_duration_seconds": round(source_seconds, 3),
            "target_duration_seconds": round(target_seconds, 3),
            "fit_applied": False,
        }

    tempo = min(max(ratio, _MIN_ATEMPO), _MAX_ATEMPO)
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
            f"atempo={tempo:.4f}",
            str(tmp_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg atempo failed")
        tmp_path.replace(audio_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    fitted_seconds = measure_audio_duration(audio_path)
    return {
        "source_duration_seconds": round(source_seconds, 3),
        "target_duration_seconds": round(target_seconds, 3),
        "applied_tempo": round(tempo, 4),
        "fitted_duration_seconds": round(fitted_seconds, 3),
        "fit_applied": True,
    }


def cleanup_speech_tail(
    audio_path: Path,
    *,
    fade_seconds: float = 0.04,
    silence_threshold_db: float = -42.0,
    min_trailing_silence: float = 0.08,
) -> dict[str, float | bool]:
    """Trim trailing silence/noise and fade out the end of a speech clip."""
    source_seconds = measure_audio_duration(audio_path)
    if source_seconds <= fade_seconds:
        return {"tail_cleanup_applied": False}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_path.parent) as tmp:
        trimmed_path = Path(tmp.name)

    try:
        trim_filter = (
            "areverse,"
            f"silenceremove=stop_periods=1:stop_duration={min_trailing_silence:.3f}:"
            f"stop_threshold={silence_threshold_db}dB,"
            "areverse"
        )
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-af",
                trim_filter,
                str(trimmed_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg tail trim failed")

        trimmed_seconds = measure_audio_duration(trimmed_path)
        if trimmed_seconds <= fade_seconds:
            return {"tail_cleanup_applied": False}

        fade_start = max(trimmed_seconds - fade_seconds, 0.0)
        fade_filter = f"afade=t=out:st={fade_start:.4f}:d={fade_seconds:.4f}"
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(trimmed_path),
                "-af",
                fade_filter,
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg tail fade failed")
    finally:
        trimmed_path.unlink(missing_ok=True)

    final_seconds = measure_audio_duration(audio_path)
    return {
        "tail_cleanup_applied": True,
        "source_duration_seconds": round(source_seconds, 3),
        "trimmed_duration_seconds": round(trimmed_seconds, 3),
        "final_duration_seconds": round(final_seconds, 3),
    }


def trim_speech_to_max_duration(
    audio_path: Path,
    max_seconds: float,
) -> dict[str, float | bool]:
    """Hard-cap speech length to reduce long-tail generation artifacts."""
    if max_seconds <= 0:
        return {"duration_cap_applied": False}

    source_seconds = measure_audio_duration(audio_path)
    if source_seconds <= max_seconds:
        return {"duration_cap_applied": False}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_path.parent) as tmp:
        capped_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-af",
                f"atrim=0:{max_seconds:.3f},asetpts=PTS-STARTPTS",
                str(capped_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg duration cap failed")
        capped_path.replace(audio_path)
    finally:
        if capped_path.exists():
            capped_path.unlink(missing_ok=True)

    return {
        "duration_cap_applied": True,
        "source_duration_seconds": round(source_seconds, 3),
        "max_duration_seconds": round(max_seconds, 3),
        "final_duration_seconds": round(measure_audio_duration(audio_path), 3),
    }


def trim_trailing_silence(
    audio_path: Path,
    *,
    silence_threshold_db: float = -38.0,
    min_trailing_silence: float = 0.15,
) -> dict[str, float | bool]:
    """Remove only trailing silence after synthesis; does not cap speech length."""
    source_seconds = measure_audio_duration(audio_path)
    if source_seconds <= min_trailing_silence:
        return {"trailing_silence_trimmed": False}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_path.parent) as tmp:
        trimmed_path = Path(tmp.name)

    try:
        trim_filter = (
            "areverse,"
            f"silenceremove=stop_periods=1:stop_duration={min_trailing_silence:.3f}:"
            f"stop_threshold={silence_threshold_db}dB,"
            "areverse"
        )
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-af",
                trim_filter,
                str(trimmed_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "FFmpeg trailing silence trim failed"
            )
        trimmed_seconds = measure_audio_duration(trimmed_path)
        if trimmed_seconds >= source_seconds - 0.02:
            return {"trailing_silence_trimmed": False}
        trimmed_path.replace(audio_path)
    finally:
        if trimmed_path.exists():
            trimmed_path.unlink(missing_ok=True)

    return {
        "trailing_silence_trimmed": True,
        "source_duration_seconds": round(source_seconds, 3),
        "final_duration_seconds": round(measure_audio_duration(audio_path), 3),
    }
