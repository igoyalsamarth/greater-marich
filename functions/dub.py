"""Dub video with separated instrumental and generated speech snippets."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from functions.audio import (
    compute_volume_gain_db,
    measure_mean_volume_db,
    speech_bus_gain_db,
)
from lib.constants import DUB_DIR, SEPARATION_DIR, VIDEOS_DIR


def _name_for_mapping(mapping_path: Path, data: dict[str, Any]) -> str:
    return str(data.get("name") or mapping_path.stem)


def _instrumental_path_for_name(name: str) -> Path:
    instrumental_path = SEPARATION_DIR / name / f"{name}_instrumental.wav"
    if not instrumental_path.is_file():
        raise FileNotFoundError(
            f"Separated instrumental not found: {instrumental_path}. "
            "Run `separate run` on the converted audio first."
        )
    return instrumental_path


def _reference_vocals_path(name: str, data: dict[str, Any]) -> Path:
    vocals_path = SEPARATION_DIR / name / f"{name}_vocals.wav"
    if vocals_path.is_file():
        return vocals_path

    reference = data.get("reference_vocals")
    if reference:
        vocals_path = Path(reference).expanduser().resolve()
        if vocals_path.is_file():
            return vocals_path

    raise FileNotFoundError(
        f"Separated vocals not found for {name!r}. "
        "Run `separate run` on the converted audio first."
    )


def _segment_start_seconds(segment: dict[str, Any]) -> float:
    return float(segment["start_time_seconds"])


def _segment_end_seconds(segment: dict[str, Any]) -> float | None:
    if segment.get("end_time_seconds") is not None:
        return float(segment["end_time_seconds"])
    if segment.get("slot_duration_seconds") is not None:
        return _segment_start_seconds(segment) + float(segment["slot_duration_seconds"])
    return None


def _segment_volume_gain_db(
    segment: dict[str, Any],
    speech_path: Path,
    reference_vocals: Path,
) -> float:
    end_seconds = _segment_end_seconds(segment)
    if end_seconds is None:
        return 0.0

    reference_db = measure_mean_volume_db(
        reference_vocals,
        start_seconds=_segment_start_seconds(segment),
        end_seconds=end_seconds,
    )
    speech_db = measure_mean_volume_db(speech_path)
    gain_db = compute_volume_gain_db(reference_db, speech_db)
    return gain_db if gain_db is not None else 0.0


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
    speech_dir: Path,
    reference_vocals: Path,
    instrumental_path: Path,
    speech_gain_db: float = 0.0,
    instrumental_input_index: int = 1,
    speech_start_index: int = 2,
) -> str:
    filter_parts: list[str] = []
    speech_labels: list[str] = []

    for offset, segment in enumerate(segments):
        input_index = speech_start_index + offset
        delay_ms = int(round(_segment_start_seconds(segment) * 1000))
        label = f"v{offset}"
        speech_path = speech_dir / segment["audio_file"]
        gain_db = _segment_volume_gain_db(segment, speech_path, reference_vocals)
        volume_filter = (
            f"volume={gain_db:.2f}dB,"
            if abs(gain_db) >= 0.05
            else ""
        )
        filter_parts.append(
            f"[{input_index}:a]aresample=48000,"
            f"pan=stereo|c0=c0|c1=c0,"
            f"{volume_filter}"
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

    bus_gain_db = speech_bus_gain_db(reference_vocals, instrumental_path) + speech_gain_db
    if abs(bus_gain_db) >= 0.05:
        filter_parts.append(f"{vocals_label}volume={bus_gain_db:.2f}dB[vocals_boosted]")
        vocals_label = "[vocals_boosted]"

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
    *,
    speech_gain_db: float = 0.0,
) -> Path:
    """Mux downloaded video with separated instrumental and dubbed speech.

    Each speech clip is gain-matched to its source vocal slice before mixing.
    When the instrumental stem is louder than the vocals stem, the speech bus is
    boosted so dialogue is not buried under the music.

    Args:
        mapping_json: Path to ``outputs/speech/<name>/<name>.json`` segment mapping.
        video: Source video file. Defaults to ``outputs/videos/<name>.mp4``.
        output: Output video file. Defaults to ``outputs/dub/<name>.mp4``.
        speech_gain_db: Extra gain applied to the speech bus after matching.

    Returns:
        Path to the dubbed video file.
    """
    mapping_path = Path(mapping_json).expanduser().resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping JSON not found: {mapping_path}")

    data: dict[str, Any] = json.loads(mapping_path.read_text(encoding="utf-8"))
    name = _name_for_mapping(mapping_path, data)
    segments = _segments_with_audio(mapping_path.parent, data.get("segments") or [])
    if not segments:
        raise RuntimeError("No speech audio files found for mapping JSON")

    speech_dir = mapping_path.parent
    video_path = Path(video).expanduser().resolve() if video else VIDEOS_DIR / f"{name}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    instrumental_path = _instrumental_path_for_name(name)
    reference_vocals = _reference_vocals_path(name, data)

    output_path = Path(output).expanduser().resolve() if output else DUB_DIR / f"{name}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = _build_audio_filter(
        segments,
        speech_dir=speech_dir,
        reference_vocals=reference_vocals,
        instrumental_path=instrumental_path,
        speech_gain_db=speech_gain_db,
    )

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
