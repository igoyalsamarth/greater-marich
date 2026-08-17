"""Audio volume helpers using FFmpeg."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

_VOLUME_DB_PATTERN = re.compile(r"mean_volume:\s*([-\d.]+)\s*dB")
_SILENCE_START_PATTERN = re.compile(r"silence_start:\s*([-\d.]+)")
_SILENCE_DURATION_PATTERN = re.compile(
    r"silence_end:\s*([-\d.]+)\s*\|\s*silence_duration:\s*([-\d.]+)"
)
_MIN_MEASURABLE_VOLUME_DB = -55.0
_MIN_GAIN_DB = -18.0
_MAX_GAIN_DB = 18.0
_MIN_ATEMPO = 0.5
_MAX_ATEMPO = 2.0
_MAX_SPEECH_STRETCH = 1.35
_DURATION_FIT_TOLERANCE = 0.08
_MAX_SPEECH_BUS_GAIN_DB = 12.0
# atempo aims just under the original slot plus this; leftover after that
# can still exist until the hard cap.
ATEMPO_TARGET_OVERSHOOT_SECONDS = 1.0
# Absolute ceiling after the original dialogue slot.
MAX_DURATION_OVERSHOOT_SECONDS = 2.0
APPLY_DURATION_CAP = True
# VAD leftover search starts this far before the original source end-time.
_VAD_TAIL_LOOKBACK_SECONDS = 0.5


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
    raw = result.stdout.strip()
    if not raw or raw.upper() == "N/A":
        return 0.0
    return float(raw)


def fit_audio_to_duration(
    audio_path: Path,
    target_seconds: float,
    *,
    speed_up_only: bool = False,
    max_tempo: float | None = None,
    min_overshoot_seconds: float = 0.0,
) -> dict[str, float | bool] | None:
    """Time-stretch audio toward a target duration using FFmpeg atempo."""
    if target_seconds <= 0:
        return None

    source_seconds = measure_audio_duration(audio_path)
    if source_seconds <= 0:
        return None

    overshoot = source_seconds - target_seconds
    ratio = source_seconds / target_seconds
    if speed_up_only:
        if overshoot <= min_overshoot_seconds:
            return {
                "source_duration_seconds": round(source_seconds, 3),
                "target_duration_seconds": round(target_seconds, 3),
                "fit_applied": False,
            }
    elif abs(ratio - 1.0) <= _DURATION_FIT_TOLERANCE:
        return {
            "source_duration_seconds": round(source_seconds, 3),
            "target_duration_seconds": round(target_seconds, 3),
            "fit_applied": False,
        }

    ceiling = max_tempo if max_tempo is not None else _MAX_ATEMPO
    tempo = min(max(ratio, _MIN_ATEMPO), ceiling)
    if speed_up_only:
        tempo = max(tempo, 1.0)
    if abs(tempo - 1.0) < 0.01:
        return {
            "source_duration_seconds": round(source_seconds, 3),
            "target_duration_seconds": round(target_seconds, 3),
            "fit_applied": False,
        }

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_path.parent) as tmp:
        tmp_path = Path(tmp.name)
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
                f"atempo={tempo:.4f}",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
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


def _trailing_silence_filter(*, duration: float, threshold_db: float) -> str:
    """Trim silence from the end only. ``start_periods`` on reversed audio.

    ``stop_periods`` after ``areverse`` keeps only the last phrase of a
    multi-sentence line, because it stops at the first pause from the end.
    """
    return (
        "areverse,"
        f"silenceremove=start_periods=1:start_duration={duration:.3f}:"
        f"start_threshold={threshold_db}dB,"
        "areverse"
    )


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
        trim_filter = _trailing_silence_filter(
            duration=min_trailing_silence,
            threshold_db=silence_threshold_db,
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
        if trimmed_seconds <= fade_seconds or trimmed_seconds < source_seconds * 0.8:
            return {"tail_cleanup_applied": False}
        if source_seconds - trimmed_seconds < 0.12:
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


def _apply_audio_filter(audio_path: Path, audio_filter: str, *, error: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_path.parent) as tmp:
        tmp_path = Path(tmp.name)
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
                audio_filter,
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or error)
        tmp_path.replace(audio_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def trim_speech_to_max_duration(
    audio_path: Path,
    max_seconds: float,
    *,
    slot_seconds: float | None = None,
    fade_seconds: float = 0.05,
) -> dict[str, float | bool]:
    """Hard-cap speech length to reduce long-tail generation artifacts.

    Prefer cutting inside a pause that starts after the original slot, so the
    last real word is kept. Otherwise cut at ``max_seconds``.
    """
    if max_seconds <= 0:
        return {"duration_cap_applied": False}

    source_seconds = measure_audio_duration(audio_path)
    if source_seconds <= max_seconds:
        return {"duration_cap_applied": False}

    cut_at = max_seconds

    fade = min(fade_seconds, max(cut_at - 0.02, 0.0))
    fade_start = max(cut_at - fade, 0.0)
    audio_filter = (
        f"atrim=0:{cut_at:.4f},asetpts=PTS-STARTPTS"
    )
    if fade > 0.01:
        audio_filter += f",afade=t=out:st={fade_start:.4f}:d={fade:.4f}"
    _apply_audio_filter(
        audio_path,
        audio_filter,
        error="FFmpeg duration cap failed",
    )

    return {
        "duration_cap_applied": True,
        "source_duration_seconds": round(source_seconds, 3),
        "max_duration_seconds": round(max_seconds, 3),
        "cut_at_seconds": round(cut_at, 3),
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
        trim_filter = _trailing_silence_filter(
            duration=min_trailing_silence,
            threshold_db=silence_threshold_db,
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
        if (
            trimmed_seconds >= source_seconds - 0.02
            or trimmed_seconds < source_seconds * 0.8
        ):
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


def _silence_regions(audio_path: Path, *, noise_db: float, min_duration: float) -> list[tuple[float, float]]:
    """Return (silence_start, silence_duration) pairs from FFmpeg silencedetect."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={noise_db:.1f}dB:d={min_duration:.3f}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg silencedetect failed")

    starts = [float(match) for match in _SILENCE_START_PATTERN.findall(result.stderr)]
    durations = [float(match[1]) for match in _SILENCE_DURATION_PATTERN.findall(result.stderr)]
    regions: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        duration = durations[index] if index < len(durations) else min_duration
        regions.append((start, duration))
    return regions


def trim_hallucinated_tail(
    audio_path: Path,
    *,
    slot_seconds: float | None = None,
    keep_decay_seconds: float = 0.32,
    fade_seconds: float = 0.05,
    min_pause_seconds: float = 0.28,
    max_leftover_seconds: float = 1.0,
) -> dict[str, float | bool]:
    """Drop a short leftover burst near the end of the take, using Silero VAD.

    Only look at pauses from ``source_end - 0.5s`` onward so mid-sentence
    gaps are never cut. A later VAD region is leftover only if it is short
    and separated by a tight pause. Keep extra tail padding so the last
    phoneme is not eaten.
    """
    from functions.vad import merge_speech_regions, speech_timestamps

    source_seconds = measure_audio_duration(audio_path)
    if slot_seconds is None or slot_seconds <= 0:
        return {"hallucinated_tail_trimmed": False}
    if source_seconds <= slot_seconds + 0.08:
        return {"hallucinated_tail_trimmed": False}

    earliest = max(0.0, slot_seconds - _VAD_TAIL_LOOKBACK_SECONDS)
    regions = merge_speech_regions(
        speech_timestamps(audio_path),
        max_gap_seconds=0.25,
    )
    if not regions:
        return {"hallucinated_tail_trimmed": False}

    cut_at: float | None = None
    last_end = regions[-1][1]
    for _start, end in regions[:-1]:
        if end < earliest:
            continue
        next_regions = [item for item in regions if item[0] >= end + min_pause_seconds]
        if not next_regions:
            continue
        leftover = last_end - next_regions[0][0]
        if leftover > max_leftover_seconds:
            continue
        cut_at = end + keep_decay_seconds
        break

    if cut_at is None:
        return {"hallucinated_tail_trimmed": False}

    cut_at = max(cut_at, earliest)
    if APPLY_DURATION_CAP:
        cut_at = min(cut_at, slot_seconds + MAX_DURATION_OVERSHOOT_SECONDS)
    cut_at = min(cut_at, source_seconds)

    if cut_at >= source_seconds - 0.08:
        return {"hallucinated_tail_trimmed": False}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_path.parent) as tmp:
        trimmed_path = Path(tmp.name)

    try:
        fade_start = max(cut_at - fade_seconds, 0.0)
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
                f"atrim=0:{cut_at:.4f},asetpts=PTS-STARTPTS,"
                f"afade=t=out:st={fade_start:.4f}:d={fade_seconds:.4f}",
                str(trimmed_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg VAD tail trim failed")
        trimmed_path.replace(audio_path)
    finally:
        trimmed_path.unlink(missing_ok=True)

    return {
        "hallucinated_tail_trimmed": True,
        "source_duration_seconds": round(source_seconds, 3),
        "cut_at_seconds": round(cut_at, 3),
        "final_duration_seconds": round(measure_audio_duration(audio_path), 3),
    }


def finalize_generated_speech(
    audio_path: Path,
    *,
    slot_seconds: float | None = None,
) -> dict[str, float | bool]:
    """Keep the full generated line, VAD-trim leftover, then fit under the cap.

    1. Soft cutoff: Silero VAD leftover burst from ``slot - 0.5s`` onward,
       with tail padding so the last word can finish.
    2. ``atempo`` speed-up to just under ``slot + 1s`` (max 1.35x).
    3. Hard cutoff: ``slot + 2s`` if anything is still over.
    """
    result: dict[str, float | bool] = {}
    result.update(trim_hallucinated_tail(audio_path, slot_seconds=slot_seconds))
    if APPLY_DURATION_CAP and slot_seconds and slot_seconds > 0:
        atempo_target = slot_seconds + ATEMPO_TARGET_OVERSHOOT_SECONDS
        hard_cutoff = slot_seconds + MAX_DURATION_OVERSHOOT_SECONDS
        fit = fit_audio_to_duration(
            audio_path,
            atempo_target - 0.06,
            speed_up_only=True,
            max_tempo=_MAX_SPEECH_STRETCH,
            min_overshoot_seconds=0.04,
        )
        if fit:
            result["duration_fit"] = fit
        result.update(
            trim_speech_to_max_duration(
                audio_path,
                hard_cutoff,
                slot_seconds=slot_seconds,
            )
        )
    return result
