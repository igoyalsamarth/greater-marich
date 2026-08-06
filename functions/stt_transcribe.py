from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from client.sarvam_client import get_sarvam_client

DEFAULT_MODEL = "saaras:v4"
DEFAULT_MODE = "verbatim"
DEFAULT_NUM_SPEAKERS = 2


def _download_transcription(job) -> dict[str, Any]:
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
    return response.json()


def transcribe_audio(
    audio_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    mode: str = DEFAULT_MODE,
    language_code: str,
    upload_timeout: float = 300.0,
    poll_timeout: int = 1800,
) -> dict[str, Any]:
    """Transcribe an audio file with Sarvam's batch speech-to-text API."""
    client = get_sarvam_client()
    job = client.speech_to_text_job.create_job(
        model=model,
        mode=mode,
        language_code=language_code,
        with_diarization=True,
        with_timestamps=True,
        num_speakers=DEFAULT_NUM_SPEAKERS,
    )
    job.upload_files([str(audio_path)], timeout=upload_timeout)
    job.start()
    status = job.wait_until_complete(timeout=poll_timeout)

    if status.job_state.lower() == "failed":
        results = job.get_file_results()
        failed = results.get("failed", [])
        message = failed[0].get("error_message") if failed else "Speech-to-text job failed"
        raise RuntimeError(message)

    return _download_transcription(job)
