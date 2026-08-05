"""Infer download filenames from existing files in the videos directory."""

from __future__ import annotations

import re
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
_NUMBERED_STEM = re.compile(r"^(?P<prefix>.+?)(?P<num>\d+)$")


def next_numbered_filename(
    output_dir: Path,
    *,
    extensions: set[str] | None = None,
) -> str | None:
    """Return the next filename stem if the folder follows a numbered pattern.

    For example, if the folder contains ``video1.mp4`` and ``video2.mp4``, this
    returns ``video3``. Returns ``None`` when no numbered pattern is detected.
    """
    patterns: dict[tuple[str, str], list[str]] = {}
    allowed_extensions = extensions or VIDEO_EXTENSIONS

    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            continue
        match = _NUMBERED_STEM.match(path.stem)
        if not match:
            continue
        key = (match.group("prefix"), ext)
        patterns.setdefault(key, []).append(match.group("num"))

    if not patterns:
        return None

    prefix, ext = max(patterns, key=lambda key: (len(patterns[key]), max(map(int, patterns[key]))))
    numbers = patterns[(prefix, ext)]
    num_width = max(len(num) for num in numbers)
    next_num = max(int(num) for num in numbers) + 1

    stem = f"{prefix}{str(next_num).zfill(num_width)}"
    while (output_dir / f"{stem}{ext}").exists():
        next_num += 1
        stem = f"{prefix}{str(next_num).zfill(num_width)}"

    return stem
