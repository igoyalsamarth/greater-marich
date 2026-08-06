"""Dub video with separated instrumental and generated speech snippets."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from lib.constants import DUB_DIR, SEPARATION_DIR, VIDEOS_DIR


def _instrumental_path_for_name(name: str) -> Path:
    instrumental_path = SEPARATION_DIR / name / f"{name}_instrumental.wav"
    if not instrumental_path.is_file():
        raise FileNotFoundError(
            f"Separated instrumental not found: {instrumental_path}. "
            "Run `separate run` on the converted audio first."
        )
    return instrumental_path


def _segment_start_seconds(segment: dict[str, Any]) -> float:
    return float(segment["start_time_seconds"])


def _segment_slot_seconds(segment: dict[str, Any]) -> float | None:
    if segment.get("slot_duration_seconds") is not None:
        return max(float(segment["slot_duration_seconds"]), 0.0)
    if "end_time_seconds" in segment:
        return max(
            float(segment["end_time_seconds"]) - _segment_start_seconds(segment),
            0.0,
        )
    return None


def _segments_with_audio(
    speech_dir: Path,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    for segment in segments:
        audio_path = speech_dir / segment["audio_file"]
        if audio_path.is_file():
            ready.append(segment)
    return sorted(ready, key=_segment_start_seconds)


def _build_audio_filter(
    segments: list[dict[str, Any]],
    *,
    instrumental_input_index: int = 1,
    speech_start_index: int = 2,
) -> str:
    filter_parts: list[str] = []
    speech_labels: list[str] = []

    for offset, segment in enumerate(segments):
        input_index = speech_start_index + offset
        delay_ms = int(round(_segment_start_seconds(segment) * 1000))
        label = f"v{offset}"
        slot_seconds = _segment_slot_seconds(segment)
        trim_filter = (
            f"atrim=0:{slot_seconds:.3f},asetpts=PTS-STARTPTS,"
            if slot_seconds and slot_seconds > 0
            else ""
        )
        filter_parts.append(
            f"[{input_index}:a]aresample=48000,"
            f"pan=stereo|c0=c0|c1=c0,"
            f"{trim_filter}"
            f"adelay={delay_ms}|{delay_ms}[{label}]"
        )
        speech_labels.append(f"[{label}]")

    if len(speech_labels) == 1:
        vocals_label = speech_labels[0]
    else:
        speech_mix = "".join(speech_labels)
        filter_parts.append(
            f"{speech_mix}amix=inputs={len(speech_labels)}:"
            f"duration=longest:dropout_transition=0:normalize=0[vocals]"
        )
        vocals_label = "[vocals]"

    filter_parts.append(
        f"[{instrumental_input_index}:a]aresample=48000[inst];"
        f"[inst]{vocals_label}"
        f"amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
    )
    return ";".join(filter_parts)


def dub_video(
    mapping_json: str | Path,
    video: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    """Mux downloaded video with separated instrumental and dubbed speech.

    Args:
        mapping_json: Path to ``outputs/speech/<name>/<name>.json`` segment mapping.
        video: Source video file. Defaults to ``outputs/videos/<name>.mp4``.
        output: Output video file. Defaults to ``outputs/dub/<name>.mp4``.

    Returns:
        Path to the dubbed video file.
    """
    mapping_path = Path(mapping_json).expanduser().resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping JSON not found: {mapping_path}")

    data: dict[str, Any] = json.loads(mapping_path.read_text(encoding="utf-8"))
    name = data.get("name") or mapping_path.stem
    segments = _segments_with_audio(mapping_path.parent, data.get("segments") or [])
    if not segments:
        raise RuntimeError("No speech audio files found for mapping JSON")

    speech_dir = mapping_path.parent
    video_path = Path(video).expanduser().resolve() if video else VIDEOS_DIR / f"{name}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    instrumental_path = _instrumental_path_for_name(name)

    output_path = Path(output).expanduser().resolve() if output else DUB_DIR / f"{name}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = _build_audio_filter(segments)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(instrumental_path),
    ]
    for segment in segments:
        audio_path = speech_dir / segment["audio_file"]
        command.extend(["-i", str(audio_path)])

    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    )

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed to dub video")

    return output_path
