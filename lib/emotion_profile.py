from __future__ import annotations

EMOTION_LABELS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sad",
    "surprise",
    "other",
)


def empty_emotion_profile() -> dict[str, float]:
    return {label: 0.0 for label in EMOTION_LABELS}


def dominant_emotion(profile: dict[str, float]) -> str:
    if not profile:
        return "neutral"
    return max(profile, key=profile.get)


def legacy_emotion_profile(emotion: str) -> dict[str, float]:
    """Coerce a single legacy emotion label into a one-hot profile."""
    label = emotion.strip().lower() or "neutral"
    if label == "sadness":
        label = "sad"
    profile = empty_emotion_profile()
    if label in profile:
        profile[label] = 1.0
    else:
        profile["neutral"] = 1.0
    return profile


def emotion_profile_from_entry(entry: dict) -> dict[str, float]:
    profile = entry.get("emotion_profile")
    if isinstance(profile, dict) and profile:
        return {
            str(label): round(float(value), 4)
            for label, value in profile.items()
            if label in EMOTION_LABELS
        }

    legacy = entry.get("emotion")
    if legacy:
        return legacy_emotion_profile(str(legacy))
    return {"neutral": 1.0}


def compact_emotion_profile(
    profile: dict[str, float],
    *,
    threshold: float = 0.05,
) -> dict[str, float]:
    """Keep only emotions that meaningfully contribute to delivery."""
    if not profile:
        return {"neutral": 1.0}
    compact = {
        label: round(float(score), 4)
        for label, score in profile.items()
        if label in EMOTION_LABELS and float(score) >= threshold
    }
    if not compact:
        dominant = dominant_emotion(profile)
        return {dominant: round(float(profile.get(dominant, 1.0)), 4)}
    return compact


def blend_emotion_profiles(
    line_profile: dict[str, float],
    character_profile: dict[str, float] | None,
    *,
    line_weight: float = 0.7,
) -> dict[str, float]:
    """Blend line-level delivery with a character's baseline emotion profile."""
    if not character_profile:
        return line_profile

    weight = min(max(line_weight, 0.0), 1.0)
    character_weight = 1.0 - weight
    return {
        label: round(
            weight * float(line_profile.get(label, 0.0))
            + character_weight * float(character_profile.get(label, 0.0)),
            4,
        )
        for label in EMOTION_LABELS
    }


def emotion_delivery_hint(
    profile: dict[str, float],
    *,
    threshold: float = 0.08,
) -> str:
    """Short natural-language delivery guidance for translation agents."""
    if not profile:
        return "Neutral delivery."

    ranked = sorted(profile.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = ranked[0]
    hint = f"Primarily {primary} ({primary_score:.0%})"

    secondary = [
        f"{label} ({score:.0%})"
        for label, score in ranked[1:3]
        if score >= threshold
    ]
    if secondary:
        hint += f", with undertones of {', '.join(secondary)}"
    return hint + "."
