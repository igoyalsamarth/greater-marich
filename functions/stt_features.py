from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

def extract_audio_segment(
    source_audio: Path,
    start: float,
    end: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed to extract audio segment")


def _read_mono_wav(audio_path: Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode().strip() or "FFmpeg failed to decode audio")

    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float64) / 32768.0
    return audio, sample_rate


def _estimate_pitch_hz(
    chunk: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = 75.0,
    fmax: float = 400.0,
) -> float | None:
    if chunk.size < sample_rate / fmax:
        return None

    chunk = chunk - np.mean(chunk)
    if np.max(np.abs(chunk)) < 1e-4:
        return None

    autocorr = np.correlate(chunk, chunk, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(len(autocorr) - 1, int(sample_rate / fmin))
    if min_lag >= max_lag:
        return None

    lag = int(np.argmax(autocorr[min_lag : max_lag + 1])) + min_lag
    if autocorr[lag] < 0.3 * autocorr[0]:
        return None
    return sample_rate / lag


def _collect_pitch_values(audio_path: Path) -> list[float]:
    audio, sample_rate = _read_mono_wav(audio_path)
    if audio.size == 0:
        return []

    frame_length = sample_rate // 10
    hop_length = frame_length // 2
    pitch_values: list[float] = []
    for start in range(0, len(audio) - frame_length, hop_length):
        pitch = _estimate_pitch_hz(audio[start : start + frame_length], sample_rate)
        if pitch is not None:
            pitch_values.append(pitch)
    return pitch_values


def measure_pitch_mean(audio_path: Path) -> float | None:
    pitch_values = _collect_pitch_values(audio_path)
    if not pitch_values:
        return None
    return float(np.median(pitch_values))


def measure_pitch_range(audio_path: Path) -> float | None:
    pitch_values = _collect_pitch_values(audio_path)
    if len(pitch_values) < 2:
        return None
    return float(np.percentile(pitch_values, 75) - np.percentile(pitch_values, 25))


def measure_energy(audio_path: Path) -> float:
    audio, _sample_rate = _read_mono_wav(audio_path)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def measure_speech_rate(text: str, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    words = [word for word in text.split() if word.strip()]
    if not words:
        return 0.0
    return len(words) / duration_seconds


def extract_segment_file(
    source_audio: Path,
    start: float,
    end: float,
    temp_dir: Path,
    index: int,
) -> Path:
    segment_path = temp_dir / f"segment_{index:04d}.wav"
    extract_audio_segment(source_audio, start, end, segment_path)
    return segment_path


def concat_audio_segments(
    segment_paths: list[Path],
    output_path: Path,
) -> float:
    """Concatenate mono WAV clips into one file. Returns total duration in seconds."""
    if not segment_paths:
        raise ValueError("segment_paths must not be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(segment_paths) == 1:
        shutil.copy2(segment_paths[0], output_path)
    else:
        list_path = output_path.with_suffix(".concat.txt")
        list_path.write_text(
            "\n".join(f"file '{path.resolve()}'" for path in segment_paths),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        list_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "FFmpeg failed to concatenate audio"
            )

    audio, sample_rate = _read_mono_wav(output_path)
    return audio.size / sample_rate
