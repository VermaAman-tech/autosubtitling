from __future__ import annotations

from pathlib import Path

import pysrt

from project.metrics import SubtitleCue


def split_subtitle_lines(text: str, max_chars: int = 42) -> str:
    words = text.split()
    lines = []
    current = []
    current_len = 0

    for word in words:
        extra = len(word) if not current else len(word) + 1
        if current and current_len + extra > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))

    return "\n".join(lines[:2])


def write_srt(cues: list[SubtitleCue], path: Path) -> None:
    subs = pysrt.SubRipFile()
    for idx, cue in enumerate(cues, start=1):
        subs.append(
            pysrt.SubRipItem(
                index=idx,
                start=pysrt.SubRipTime(seconds=cue.start),
                end=pysrt.SubRipTime(seconds=cue.end),
                text=split_subtitle_lines(cue.text),
            )
        )
    subs.save(path, encoding="utf-8")
