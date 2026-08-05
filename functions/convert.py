"""Extract audio from video files using FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from lib.constants import AUDIO_DIR
from functions.naming import next_numbered_filename

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp4", ".flac"}
DEFAULT_AUDIO_EXTENSION = ".wav"


def _resolve_output_path(
    input_path: Path,
    output: str | Path | None,
) -> Path:
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        if output_path.suffix:
            if output_path.suffix.lower() in {".mp4", ".m4a"}:
                output_path = output_path.with_suffix(DEFAULT_AUDIO_EXTENSION)
            return output_path
        output_dir = output_path
    else:
        output_dir = AUDIO_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    if numbered_stem := next_numbered_filename(output_dir, extensions=AUDIO_EXTENSIONS):
        return output_dir / f"{numbered_stem}{DEFAULT_AUDIO_EXTENSION}"
    return output_dir / f"{input_path.stem}{DEFAULT_AUDIO_EXTENSION}"


def convert_video_to_audio(
    input_path: str | Path,
    output: str | Path | None = None,
) -> Path:
    """Extract audio from a video file as lossless WAV for speech-to-text.

    Uses 16-bit PCM with the source sample rate and channel layout preserved.
    FFmpeg must be installed and available on PATH.

    Args:
        input_path: Path to the source video file.
        output: Output file or directory. Defaults to ``audio/<name>.wav``.

    Returns:
        Path to the created audio file.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If FFmpeg fails.
    """
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")

    output_path = _resolve_output_path(input_path, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed to extract audio")

    return output_path
