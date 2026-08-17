"""Generate speech with Chatterbox Multilingual Hindi voice cloning."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from functions.audio import finalize_generated_speech, match_volume_to_reference
from functions.dialogues import dialogue_entries, load_dialogues
from functions.speech_shared import (
    entry_duration,
    entry_end,
    entry_speaker_id,
    entry_start,
    entry_text,
    instrumental_path_for_name,
    resolve_characters_path,
    resolve_source_dialogues,
    segment_id,
    source_name,
    vocals_path_for_name,
    write_speech_mapping,
)
from functions.stt_features import (
    _estimate_pitch_hz,
    _read_mono_wav,
    concat_audio_segments,
    extract_segment_file,
    measure_energy,
)
from functions.translate_models import load_character_personas
from lib.chatterbox_loader import (
    CROSS_LANGUAGE_CFG_WEIGHT,
    DEFAULT_MODEL,
    HINDI_LANGUAGE_ID,
    STABLE_EXAGGERATION,
    STABLE_TEMPERATURE,
    load_chatterbox_model,
    normalize_chatterbox_language_id,
    prepare_voice_conditionals,
    save_speech,
    synthesize,
)
from lib.constants import SPEECH_CHATTERBOX_DIR
from lib.emotion_profile import (
    dominant_emotion,
    emotion_delivery_hint,
    emotion_profile_from_entry,
)

# Chatterbox only consumes ~6s (T3) / ~10s (S3Gen) of the prompt, so a short
# clean reference clones more reliably than concatenating every line.
MIN_PROMPT_DURATION_SECONDS = 1.0
MIN_REFERENCE_SECONDS = 6.0
MAX_REFERENCE_SECONDS = 15.0
MIN_CANDIDATE_SECONDS = 0.85
OVERLAP_SECONDS = 0.04
ADJACENT_SECONDS = 0.12
START_CROP_SECONDS = 0.06
# Keep the last phoneme's decay. Cutting STT end-times flush with the next
# turn makes the cloning prompt end mid-word and the model copies that.
END_PAD_SECONDS = 0.14
REFERENCE_GAP_SECONDS = 0.05
SILENCE_RMS = 0.01
MAX_SILENCE_RATIO = 0.4
MAX_CLIPPING_RATIO = 0.002
MIN_VOICED_RATIO = 0.2
MIN_SNR_DB = 8.0
MUSIC_ENERGY_RATIO = 1.25
MIN_PRIMARY_WORDS = 2

# Volume matching can clip peaks and lift residual noise in generated tails.
# Keep the raw WAV as the pipeline output until comparison shows it is safe.
APPLY_VOLUME_MATCH = False
_CLIP_PEAK = 0.98
_NOISE_GAIN_DB = 0.5
_NOISE_AMPLIFY_RATIO = 1.4
_AUDIBLE_NOISE_RMS = 0.005

# Narrow delivery bands around the stable defaults. Wider swings caused
# glitches; these only nudge emotion and pace a little.
_EXAGGERATION_RANGE = (0.50, 0.58)
_TEMPERATURE_RANGE = (0.58, 0.66)
_CFG_RANGE = (0.36, 0.48)
# Short prompts otherwise come out word-by-word. CFG is Chatterbox pace;
# a higher value connects the words instead of parking a gap between them.
SHORT_LINE_WORD_LIMIT = 3
_SHORT_LINE_CFG_RANGE = (0.50, 0.58)


def _speech_dir_for_name(name: str) -> Path:
    return SPEECH_CHATTERBOX_DIR / name


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return round(min(max(value, bounds[0]), bounds[1]), 2)


def _profile_score(emotion_profile: dict[str, float], labels: tuple[str, ...]) -> float:
    return min(sum(float(emotion_profile.get(label, 0.0)) for label in labels), 1.0)


def _delivery_from_profile(emotion_profile: dict[str, float]) -> tuple[float, float, float]:
    """Map line emotion to a small exaggeration / temperature / CFG nudge.

    Happiness, surprise, and anger add a bit of energy and pace. Sad and fear
    pull slightly slower and flatter. Neutral stays on the stable defaults.
    """
    intensity = min(max(emotion_profile.values()) if emotion_profile else 0.0, 1.0)
    energy = _profile_score(emotion_profile, ("happiness", "surprise", "anger"))
    weight = _profile_score(emotion_profile, ("sad", "fear"))
    non_neutral = min(1.0 - float(emotion_profile.get("neutral", 0.0)), 1.0)

    exaggeration = _clamp(
        STABLE_EXAGGERATION + non_neutral * (0.04 + intensity * 0.06),
        _EXAGGERATION_RANGE,
    )
    temperature = _clamp(
        STABLE_TEMPERATURE + (energy - weight) * 0.08,
        _TEMPERATURE_RANGE,
    )
    cfg_weight = _clamp(
        CROSS_LANGUAGE_CFG_WEIGHT + (energy * 0.10 - weight * 0.08),
        _CFG_RANGE,
    )
    return exaggeration, temperature, cfg_weight


def _word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def _intervals_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return min(end_a, end_b) - max(start_a, start_b) > OVERLAP_SECONDS


def _clean_reference_window(
    entry: dict[str, Any],
    all_entries: list[dict[str, Any]],
    speaker_id: str,
) -> tuple[float, float] | None:
    """Return a non-overlapping window that keeps complete words.

    Only the start is cropped when another speaker just finished. The end is
    padded so the last syllable decays naturally instead of being chopped.
    """
    start = entry_start(entry)
    end = entry_end(entry)
    if end <= start:
        return None

    previous_end = float("-inf")
    next_start = float("inf")
    for other in all_entries:
        if other is entry or entry_speaker_id(other) == speaker_id:
            continue
        other_start = entry_start(other)
        other_end = entry_end(other)
        if _intervals_overlap(start, end, other_start, other_end):
            return None
        if other_end <= start:
            previous_end = max(previous_end, other_end)
        if other_start >= end:
            next_start = min(next_start, other_start)

    if 0.0 <= start - previous_end <= ADJACENT_SECONDS:
        start += START_CROP_SECONDS
    end = min(end + END_PAD_SECONDS, next_start - 0.02)

    if end - start < MIN_CANDIDATE_SECONDS:
        return None
    return start, end


def _frame_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.size == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
            "voiced_ratio": 0.0,
            "snr_db": 0.0,
            "noise_rms": 0.0,
        }

    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(abs_audio))
    clipping_ratio = float(np.mean(abs_audio >= _CLIP_PEAK))

    frame_length = max(1, sample_rate // 50)
    hop_length = max(1, frame_length // 2)
    pitch_frame = max(1, sample_rate // 10)
    rms_frames: list[float] = []
    voiced_frames = 0
    pitch_frames = 0
    for index in range(0, len(audio) - frame_length + 1, hop_length):
        chunk = audio[index : index + frame_length]
        rms_frames.append(float(np.sqrt(np.mean(np.square(chunk)))))
    for index in range(0, len(audio) - pitch_frame + 1, pitch_frame // 2):
        pitch_frames += 1
        if _estimate_pitch_hz(audio[index : index + pitch_frame], sample_rate) is not None:
            voiced_frames += 1

    silence_ratio = (
        float(np.mean([frame < SILENCE_RMS for frame in rms_frames]))
        if rms_frames
        else 1.0
    )
    voiced_ratio = voiced_frames / pitch_frames if pitch_frames else 0.0
    loud = [frame for frame in rms_frames if frame >= SILENCE_RMS]
    quiet = [frame for frame in rms_frames if frame < SILENCE_RMS]
    noise_rms = float(np.median(quiet)) if quiet else float(np.percentile(rms_frames, 10))
    signal_rms = float(np.median(loud)) if loud else rms
    snr_db = 20.0 * math.log10(max(signal_rms, 1e-8) / max(noise_rms, 1e-8))

    return {
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "voiced_ratio": voiced_ratio,
        "snr_db": snr_db,
        "noise_rms": noise_rms,
    }


def _trim_leading_silence(source_path: Path, dest_path: Path) -> float | None:
    """Drop leading silence only. Trailing trim would cut the last phoneme."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-af",
            "silenceremove=start_periods=1:start_duration=0.04:start_threshold=-40dB",
            str(dest_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not dest_path.is_file():
        return None
    audio, sample_rate = _read_mono_wav(dest_path)
    if audio.size == 0:
        return None
    return audio.size / sample_rate


def _write_silence_clip(path: Path, duration_seconds: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            f"{duration_seconds:.3f}",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed to create silence")


def _quality_score(
    *,
    metrics: dict[str, float],
    duration_seconds: float,
    word_count: int,
    music_ratio: float,
) -> tuple[float, str | None]:
    """Score a clip for cloning. Hard rejects return a reason instead of a score."""
    if metrics["rms"] < SILENCE_RMS * 0.6 or metrics["silence_ratio"] > 0.7:
        return 0.0, "silence"
    if metrics["clipping_ratio"] > MAX_CLIPPING_RATIO:
        return 0.0, "clipped"
    if music_ratio >= MUSIC_ENERGY_RATIO:
        return 0.0, "music"
    if metrics["voiced_ratio"] < 0.08 and metrics["rms"] > SILENCE_RMS:
        return 0.0, "noisy"
    if duration_seconds < 0.6:
        return 0.0, "clipped_words"

    soft_reject = None
    long_enough_for_pauses = duration_seconds >= 2.0
    too_silent = metrics["silence_ratio"] > 0.75 or (
        metrics["silence_ratio"] > MAX_SILENCE_RATIO and not long_enough_for_pauses
    )
    too_unvoiced = metrics["voiced_ratio"] < MIN_VOICED_RATIO and not (
        long_enough_for_pauses and metrics["voiced_ratio"] >= 0.12
    )
    if (
        too_silent
        or too_unvoiced
        or metrics["snr_db"] < MIN_SNR_DB
        or word_count < MIN_PRIMARY_WORDS
        or duration_seconds < MIN_CANDIDATE_SECONDS
    ):
        soft_reject = "low_quality"

    duration_bonus = min(duration_seconds / 3.0, 2.5)
    score = (
        metrics["voiced_ratio"] * 2.0
        + min(metrics["snr_db"] / 20.0, 1.0) * 1.5
        + (1.0 - metrics["silence_ratio"])
        + duration_bonus
        - min(music_ratio, 2.0) * 0.4
    )
    if soft_reject:
        score *= 0.35
    return score, soft_reject


def _select_reference_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick a 6–15s set of the cleanest clips, best-first for the prompt window."""
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    primary = [item for item in ranked if item["reject_reason"] is None]
    fallback = [item for item in ranked if item["reject_reason"] == "low_quality"]
    selected: list[dict[str, Any]] = []
    total = 0.0

    for pool in (primary, fallback):
        for item in pool:
            duration = float(item["duration_seconds"])
            if total + duration > MAX_REFERENCE_SECONDS:
                if selected:
                    continue
            selected.append(item)
            total += duration
            if total >= MAX_REFERENCE_SECONDS:
                break
        if total >= MIN_REFERENCE_SECONDS:
            break

    # Longest clean clips first: Chatterbox only reads ~6s (T3) / ~10s (S3Gen).
    selected.sort(key=lambda item: (-float(item["duration_seconds"]), -float(item["score"])))
    return selected


def _build_character_reference_audio(
    *,
    vocals_path: Path,
    source_entries: list[dict[str, Any]],
    speaker_id: str,
    output_path: Path,
    temp_dir: Path,
    instrumental_path: Path | None = None,
) -> dict[str, Any]:
    """Build one clean 6–15s cloning prompt per speaker.

    Concatenating every line mixes silence, music bleed, overlap, and clipped
    words into the speaker embedding and is a common source of glitches.
    """
    speaker_entries = [
        entry
        for entry in source_entries
        if entry_speaker_id(entry) == speaker_id and entry_duration(entry) > 0
    ]
    speaker_entries.sort(key=entry_start)

    if not speaker_entries:
        raise ValueError(f"No source dialogue segments found for {speaker_id!r}.")

    temp_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    rejected_count = 0
    total_source_seconds = 0.0

    for index, entry in enumerate(speaker_entries):
        total_source_seconds += entry_duration(entry)
        window = _clean_reference_window(entry, source_entries, speaker_id)
        if window is None:
            rejected_count += 1
            continue

        start, end = window
        raw_path = extract_segment_file(vocals_path, start, end, temp_dir, index)
        trimmed_path = temp_dir / f"trimmed_{index:04d}.wav"
        trimmed_duration = _trim_leading_silence(raw_path, trimmed_path)
        if trimmed_duration is not None and trimmed_duration >= 0.5:
            clip_path = trimmed_path
            duration_seconds = trimmed_duration
            audio, sample_rate = _read_mono_wav(trimmed_path)
        else:
            clip_path = raw_path
            audio, sample_rate = _read_mono_wav(raw_path)
            duration_seconds = audio.size / sample_rate if sample_rate else 0.0

        metrics = _frame_metrics(audio, sample_rate)
        music_ratio = 0.0
        if instrumental_path is not None:
            inst_path = extract_segment_file(
                instrumental_path,
                start,
                end,
                temp_dir / "instrumental",
                index,
            )
            vocal_energy = max(metrics["rms"], 1e-8)
            music_ratio = measure_energy(inst_path) / vocal_energy

        transcript = entry_text(entry).strip()
        score, reject_reason = _quality_score(
            metrics=metrics,
            duration_seconds=duration_seconds,
            word_count=_word_count(transcript),
            music_ratio=music_ratio,
        )
        if reject_reason in {"silence", "clipped", "music", "noisy", "clipped_words"}:
            rejected_count += 1
            continue

        candidates.append(
            {
                "path": clip_path,
                "start": start,
                "duration_seconds": duration_seconds,
                "transcript": transcript,
                "score": score,
                "reject_reason": reject_reason,
            }
        )

    selected = _select_reference_candidates(candidates)
    if not selected:
        # Last resort: keep the longest remaining speaker clip so cloning can run.
        fallback_entry = max(speaker_entries, key=entry_duration)
        fallback_path = extract_segment_file(
            vocals_path,
            entry_start(fallback_entry),
            entry_end(fallback_entry),
            temp_dir,
            9999,
        )
        selected = [
            {
                "path": fallback_path,
                "start": entry_start(fallback_entry),
                "duration_seconds": entry_duration(fallback_entry),
                "transcript": entry_text(fallback_entry).strip(),
                "score": 0.0,
                "reject_reason": "fallback",
            }
        ]
        rejected_count = max(rejected_count - 1, 0)

    concat_paths: list[Path] = []
    silence_path = temp_dir / "reference_gap.wav"
    if len(selected) > 1:
        _write_silence_clip(silence_path, REFERENCE_GAP_SECONDS)
    for index, item in enumerate(selected):
        concat_paths.append(item["path"])
        if index < len(selected) - 1:
            concat_paths.append(silence_path)

    duration_seconds = concat_audio_segments(concat_paths, output_path)

    return {
        "speaker_id": speaker_id,
        "reference_audio": str(output_path),
        "reference_vocals": str(vocals_path),
        "segment_count": len(selected),
        "total_source_seconds": round(total_source_seconds, 3),
        "reference_duration_seconds": round(duration_seconds, 3),
        "source_transcripts": [item["transcript"] for item in selected],
        "rejected_segment_count": rejected_count,
    }


def _wav_peak_and_noise(audio_path: Path) -> tuple[float, float]:
    audio, sample_rate = _read_mono_wav(audio_path)
    if audio.size == 0:
        return 0.0, 0.0
    metrics = _frame_metrics(audio, sample_rate)
    return metrics["peak"], metrics["noise_rms"]


def _compare_volume_match(
    audio_path: Path,
    vocals_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    """Compare raw vs volume-matched audio without replacing the raw WAV."""
    raw_peak, raw_noise = _wav_peak_and_noise(audio_path)
    comparison: dict[str, Any] = {
        "applied": False,
        "raw_peak": round(raw_peak, 4),
        "raw_noise_rms": round(raw_noise, 6),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        post_path = Path(temp_dir) / audio_path.name
        shutil.copy2(audio_path, post_path)
        volume_match = match_volume_to_reference(
            post_path,
            vocals_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        if not volume_match:
            comparison["reason"] = "no measurable reference or source volume"
            comparison["safe_to_apply"] = False
            return comparison

        post_peak, post_noise = _wav_peak_and_noise(post_path)
        gain_db = float(volume_match["applied_gain_db"])
        would_clip = post_peak >= _CLIP_PEAK
        would_amplify_noise = (
            gain_db > _NOISE_GAIN_DB
            and raw_noise >= _AUDIBLE_NOISE_RMS
            and post_noise > raw_noise * _NOISE_AMPLIFY_RATIO
        )
        safe_to_apply = not would_clip and not would_amplify_noise
        comparison.update(
            {
                "reference_volume_db": volume_match["reference_volume_db"],
                "source_volume_db": volume_match["source_volume_db"],
                "proposed_gain_db": gain_db,
                "post_peak": round(post_peak, 4),
                "post_noise_rms": round(post_noise, 6),
                "would_clip": would_clip,
                "would_amplify_noise": would_amplify_noise,
                "safe_to_apply": safe_to_apply,
            }
        )

        # Re-enable only when comparison shows no clipping or noise lift.
        if APPLY_VOLUME_MATCH and safe_to_apply:
            shutil.copy2(post_path, audio_path)
            comparison["applied"] = True
            comparison["reason"] = "applied; comparison showed no clipping or noise lift"
        elif APPLY_VOLUME_MATCH:
            comparison["reason"] = "skipped; volume match would clip or amplify noise"
        else:
            comparison["reason"] = (
                "temporarily disabled; kept raw WAV. Re-enable APPLY_VOLUME_MATCH "
                "only when safe_to_apply is true."
            )

    return comparison


def generate_chatterbox_speech(
    transcript_json: str | Path,
    output_dir: str | Path | None = None,
    *,
    model_id: str = DEFAULT_MODEL,
    language_id: str = HINDI_LANGUAGE_ID,
) -> Path:
    """Generate Hindi speech clips with Chatterbox voice cloning.

    Builds one clean 6–15s zero-shot cloning reference per speaker, then
    synthesizes each translated line with a light emotion/pace nudge around
    stable exaggeration, temperature, ``cfg_weight``, and language id ``hi``.
    """
    transcript_path = Path(transcript_json).expanduser().resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_path}")

    data = load_dialogues(transcript_path)
    entries = dialogue_entries(data)
    source_entries = resolve_source_dialogues(data, transcript_path)
    if len(source_entries) != len(entries):
        raise RuntimeError(
            "Translated dialogues and source STT dialogues have different lengths. "
            "Re-run translate on the current STT output."
        )

    name = data.get("name") or source_name(transcript_path)
    speech_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else _speech_dir_for_name(name)
    )
    speech_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = vocals_path_for_name(name)
    instrumental_path = instrumental_path_for_name(name)
    language_code = data.get("language_code") or "hi-IN"
    # Chatterbox must receive "hi". Mapping language_code stays BCP-47 for metadata.
    tts_language_id = normalize_chatterbox_language_id(language_id)
    characters_path = resolve_characters_path(data, transcript_path)
    personas = load_character_personas(characters_path)

    model = load_chatterbox_model(model_id)
    segments: list[dict[str, Any]] = []
    character_references: dict[str, dict[str, Any]] = {}
    prepared_speaker: str | None = None
    prepared_exaggeration: float | None = None

    speaker_ids = sorted(
        {entry_speaker_id(entry) for entry in source_entries if entry_speaker_id(entry)}
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for speaker_id in speaker_ids:
            reference_path = speech_dir / f"_ref_{speaker_id}.wav"
            character_references[speaker_id] = _build_character_reference_audio(
                vocals_path=vocals_path,
                source_entries=source_entries,
                speaker_id=speaker_id,
                output_path=reference_path,
                temp_dir=temp_path / speaker_id,
                instrumental_path=instrumental_path,
            )
            persona = personas.get(speaker_id)
            if persona:
                character_references[speaker_id]["attributes"] = (
                    persona.attributes.model_dump()
                )
                character_references[speaker_id]["character_emotion_profile"] = (
                    persona.emotion_profile
                )

        for index, (entry, source_entry) in enumerate(
            zip(entries, source_entries, strict=True),
            start=1,
        ):
            seg_id = segment_id(index)
            audio_file = f"{seg_id}.wav"
            audio_path = speech_dir / audio_file

            hindi_text = entry_text(entry).strip()
            source_text = entry_text(source_entry).strip()
            word_count = _word_count(hindi_text)
            speaker_id = entry_speaker_id(entry)
            emotion_profile = emotion_profile_from_entry(entry)
            delivery_hint = emotion_delivery_hint(emotion_profile)
            exaggeration, temperature, cfg_weight = _delivery_from_profile(
                emotion_profile
            )
            start_seconds = entry_start(entry)
            end_seconds = entry_end(entry)
            slot_seconds = entry_duration(entry)
            # Short prompts come out word-by-word; raise CFG so they speak
            # connected instead of collapsing pauses after the fact.
            if word_count <= SHORT_LINE_WORD_LIMIT:
                cfg_weight = _clamp(cfg_weight + 0.12, _SHORT_LINE_CFG_RANGE)
            persona = personas.get(speaker_id)
            character_attributes = (
                persona.attributes.model_dump() if persona else None
            )

            reference_meta = character_references.get(speaker_id)
            reference_path = (
                Path(reference_meta["reference_audio"])
                if reference_meta
                else None
            )
            reference_duration = (
                float(reference_meta["reference_duration_seconds"])
                if reference_meta
                else 0.0
            )

            volume_match: dict[str, Any] | None = None
            volume_comparison: dict[str, Any] | None = None

            if (
                hindi_text
                and reference_path
                and reference_path.is_file()
                and reference_duration >= MIN_PROMPT_DURATION_SECONDS
            ):
                if (
                    prepared_speaker != speaker_id
                    or prepared_exaggeration != exaggeration
                ):
                    prepare_voice_conditionals(
                        model,
                        prompt_wav=reference_path,
                        exaggeration=exaggeration,
                    )
                    prepared_speaker = speaker_id
                    prepared_exaggeration = exaggeration

                speech, sample_rate = synthesize(
                    model,
                    text=hindi_text,
                    language_id=tts_language_id,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                save_speech(audio_path, speech, sample_rate)
                finalize_generated_speech(
                    audio_path,
                    slot_seconds=slot_seconds,
                )
                volume_comparison = _compare_volume_match(
                    audio_path,
                    vocals_path,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
                if volume_comparison.get("applied"):
                    volume_match = {
                        "reference_volume_db": volume_comparison["reference_volume_db"],
                        "source_volume_db": volume_comparison["source_volume_db"],
                        "applied_gain_db": volume_comparison["proposed_gain_db"],
                    }

            segment: dict[str, Any] = {
                "id": seg_id,
                "audio_file": audio_file,
                "start_time_seconds": round(start_seconds, 3),
                "end_time_seconds": round(end_seconds, 3),
                "slot_duration_seconds": round(slot_seconds, 3),
                "speaker_id": speaker_id,
                "character_id": speaker_id,
                "character_attributes": character_attributes,
                "emotion_profile": emotion_profile,
                "emotion": dominant_emotion(emotion_profile),
                "delivery_hint": delivery_hint,
                "exaggeration": exaggeration,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
                "transcript": hindi_text,
                "source_transcript": source_text,
                "reference_audio": str(reference_path) if reference_path else None,
                "synthesis_mode": "chatterbox_clone",
            }
            if volume_match:
                segment["volume_match"] = volume_match
            if volume_comparison:
                segment["volume_comparison"] = volume_comparison
            segments.append(segment)

    mapping = {
        "name": name,
        "language_code": language_code,
        "tts_language_id": tts_language_id,
        "source_language_code": data.get("source_language_code"),
        "source_json": str(transcript_path),
        "source_dialogues": data.get("source_dialogues"),
        "characters_json": str(characters_path),
        "translation_model": data.get("translation_model"),
        "plot_summary": data.get("plot_summary"),
        "tts_model": model_id,
        "tts_engine": "chatterbox-hi",
        "cfg_weight": CROSS_LANGUAGE_CFG_WEIGHT,
        "reference_vocals": str(vocals_path),
        "character_references": character_references,
        "segments": segments,
    }

    return write_speech_mapping(
        mapping_path=speech_dir / f"{name}.json",
        mapping=mapping,
    )
