"""Speech-to-text transcription using Sarvam's batch API."""

from __future__ import annotations

from pathlib import Path

import httpx

from client.sarvam_client import get_sarvam_client
from lib.constants import SEPARATION_DIR, STT_DIR

DEFAULT_MODEL = "saaras:v4"
DEFAULT_MODE = "verbatim"
DEFAULT_LANGUAGE_CODE = "en-IN"


def _base_name(audio_path: Path) -> str:
    stem = audio_path.stem
    if stem.endswith("_vocals"):
        return stem[: -len("_vocals")]
    return stem


def _resolve_vocals_audio(audio_path: Path) -> Path:
    name = _base_name(audio_path)
    vocals_path = SEPARATION_DIR / name / f"{name}_vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError(
            f"Separated vocals not found: {vocals_path}. "
            "Run `separate run` on the converted audio first."
        )
    return vocals_path


def _resolve_output_path(audio_path: Path, output: str | Path | None) -> Path:
    name = _base_name(audio_path)
    if output is None:
        return STT_DIR / f"{name}.json"

    output_path = Path(output).expanduser().resolve()
    if output_path.suffix:
        return output_path
    return output_path / f"{name}.json"


def _download_transcription(job, output_path: Path) -> None:
    mappings = job.get_output_mappings()
    if not mappings:
        results = job.get_file_results()
        failed = results.get("failed", [])
        if failed:
            message = failed[0].get("error_message") or "Speech-to-text job failed"
            raise RuntimeError(message)
        raise RuntimeError("No transcription output available")

    mapping = mappings[0]
    client = get_sarvam_client()
    download_links = client.speech_to_text_job.get_download_links(
        job_id=job.job_id,
        files=[mapping["output_file"]],
    )
    url = download_links.download_urls[mapping["output_file"]].file_url

    response = httpx.get(url)
    if response.is_error:
        raise RuntimeError(
            f"Failed to download transcription for {mapping['output_file']}: "
            f"{response.status_code}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def transcribe_audio(
    audio_path: str | Path,
    output: str | Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    mode: str = DEFAULT_MODE,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    with_diarization: bool = True,
    with_timestamps: bool = True,
    upload_timeout: float = 300.0,
    poll_timeout: int = 1800,
) -> Path:
    """Transcribe separated vocals with Sarvam's batch speech-to-text API.

    Args:
        audio_path: Path to the converted audio file (for example ``audio/video1.wav``).
            The separated vocals at ``separation/<name>/<name>_vocals.wav`` are used.
        output: Output JSON file or directory. Defaults to ``stt/<name>.json``.
        model: Sarvam STT model.
        mode: Transcription mode for saaras models.
        language_code: Language of the input audio.
        with_diarization: Whether to distinguish speakers.
        with_timestamps: Whether to include timestamps in the transcription.
        upload_timeout: Seconds to wait for file upload.
        poll_timeout: Seconds to wait for the batch job to finish.

    Returns:
        Path to the saved transcription JSON file.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    vocals_path = _resolve_vocals_audio(audio_path)
    output_path = _resolve_output_path(audio_path, output)
    client = get_sarvam_client()

    job = client.speech_to_text_job.create_job(
        model=model,
        mode=mode,
        language_code=language_code,
        with_diarization=with_diarization,
        with_timestamps=with_timestamps
    )

    job.upload_files([str(vocals_path)], timeout=upload_timeout)
    job.start()
    status = job.wait_until_complete(timeout=poll_timeout)

    if status.job_state.lower() == "failed":
        results = job.get_file_results()
        failed = results.get("failed", [])
        message = failed[0].get("error_message") if failed else "Speech-to-text job failed"
        raise RuntimeError(message)

    _download_transcription(job, output_path)
    return output_path
