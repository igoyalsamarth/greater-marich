from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dialogues(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    entries = (data.get("diarized_transcript") or {}).get("entries") or []
    dialogues = data.get("dialogues") or []
    if entries or dialogues:
        return data

    raise RuntimeError(f"No diarized transcript found in {path}")


def dialogue_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    entries = (data.get("diarized_transcript") or {}).get("entries") or []
    if entries:
        return entries
    return data.get("dialogues") or []
