"""Two small safety pieces.

1. Safe mode (on by default, you can switch it off): quietly adds "nudity" words to the
   negative prompt so the AI steers away from adult content.
2. A hard rule that can't be switched off: prompts that mix children with sexual
   content are refused.
"""

from __future__ import annotations

import re

SAFE_NEGATIVE = "nsfw, nude, naked, nudity, explicit, sexual, lingerie, gore"

_MINOR = [
    "child", "children", "kid", "kids", "kiddo", "toddler", "baby", "infant", "minor", "minors",
    "underage", "under age", "preteen", "pre-teen", "tween", "young girl", "young boy",
    "little girl", "little boy", "schoolgirl", "schoolboy", "school girl", "school boy",
    "loli", "lolita", "shota", "teen", "teens", "teenager", "teenage", "juvenile",
    "elementary school", "middle school", "high school", "grade schooler",
    "1 year old", "2 year old", "3 year old", "4 year old", "5 year old", "6 year old",
    "7 year old", "8 year old", "9 year old", "10 year old", "11 year old", "12 year old",
    "13 year old", "14 year old", "15 year old", "16 year old", "17 year old",
]
_SEXUAL = [
    "nsfw", "nude", "nudes", "naked", "nudity", "topless", "bottomless", "sex", "sexy",
    "sexual", "sexualized", "erotic", "porn", "porno", "pornographic", "hentai", "lewd",
    "explicit", "seductive", "lingerie", "undressed", "undressing", "strip", "stripping",
    "orgasm", "genital", "genitals", "breasts", "nipples", "fetish", "bdsm", "xxx",
    "panties", "bikini", "provocative", "sensual", "aroused", "intercourse",
]


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"(\d+)\s*[-_]?\s*(?:years?|yrs?|yo)\s*[-_]?\s*old", r"\1 year old", text)
    text = re.sub(r"(\d+)\s*yo\b", r"\1 year old", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f" {text.strip()} "


def _has(text: str, words) -> bool:
    return any(f" {w} " in text for w in words)


def check_prompt(prompt: str) -> str | None:
    """Return a reason string if the prompt is not allowed, else None."""
    t = _norm(prompt)
    if _has(t, _MINOR) and _has(t, _SEXUAL):
        return ("This prompt mixes children (or teens) with sexual content, "
                "which the app will never make. Please change your prompt.")
    return None


def build_negative(user_negative: str, safe_mode: bool) -> str:
    parts = [p for p in [(user_negative or "").strip()] if p]
    if safe_mode:
        parts.append(SAFE_NEGATIVE)
    return ", ".join(parts)
