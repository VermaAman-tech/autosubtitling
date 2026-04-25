from __future__ import annotations

from dataclasses import dataclass

from jiwer import cer, wer


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def compute_wer(reference: str, hypothesis: str) -> float:
    return float(wer(normalize_text(reference), normalize_text(hypothesis)))


def compute_cer(reference: str, hypothesis: str) -> float:
    return float(cer(normalize_text(reference), normalize_text(hypothesis)))


@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str


def timestamp_mae(reference: list[SubtitleCue], predicted: list[SubtitleCue]) -> float:
    count = min(len(reference), len(predicted))
    if count == 0:
        return float("nan")
    total = 0.0
    for idx in range(count):
        total += abs(reference[idx].start - predicted[idx].start)
        total += abs(reference[idx].end - predicted[idx].end)
    return total / (2 * count)


def cps_violations(cues: list[SubtitleCue], limit: float = 20.0) -> int:
    violations = 0
    for cue in cues:
        duration = max(0.1, cue.end - cue.start)
        cps = len(cue.text) / duration
        if cps > limit:
            violations += 1
    return violations
