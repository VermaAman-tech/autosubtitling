from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel


@dataclass
class ASRPrediction:
    text: str
    start: float
    end: float
    avg_logprob: float
    # Latency fields
    audio_duration_sec: float = 0.0
    inference_time_ms: float = 0.0
    rtf: float = 0.0            # Real-Time Factor = inference_time / audio_duration


@dataclass
class BenchmarkStats:
    total_audio_sec: float = 0.0
    total_inference_ms: float = 0.0
    segment_count: int = 0
    rtf_per_segment: list[float] = field(default_factory=list)

    @property
    def mean_rtf(self) -> float:
        if not self.rtf_per_segment:
            return float("nan")
        return float(sum(self.rtf_per_segment) / len(self.rtf_per_segment))

    @property
    def overall_rtf(self) -> float:
        if self.total_audio_sec == 0:
            return float("nan")
        return (self.total_inference_ms / 1000.0) / self.total_audio_sec

    @property
    def p50_latency_ms(self) -> float:
        if not self.rtf_per_segment:
            return float("nan")
        sorted_rtf = sorted(self.rtf_per_segment)
        return sorted_rtf[len(sorted_rtf) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self.rtf_per_segment:
            return float("nan")
        sorted_rtf = sorted(self.rtf_per_segment)
        idx = int(0.95 * len(sorted_rtf))
        return sorted_rtf[min(idx, len(sorted_rtf) - 1)]


class WhisperASR:
    def __init__(self, model_size: str = "tiny.en") -> None:
        self.model_size = model_size
        t0 = time.perf_counter()
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0
        self.stats = BenchmarkStats()

    def transcribe_file(self, path: Path, audio_duration_sec: float = 0.0) -> ASRPrediction:
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            str(path),
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text_parts = []
        start = None
        end = None
        logprobs = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            start = segment.start if start is None else min(start, segment.start)
            end = segment.end if end is None else max(end, segment.end)
            if segment.avg_logprob is not None:
                logprobs.append(segment.avg_logprob)

        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        dur = audio_duration_sec if audio_duration_sec > 0 else max((end or 0.0), 0.01)
        rtf = (inference_time_ms / 1000.0) / max(dur, 1e-6)

        self.stats.total_audio_sec += dur
        self.stats.total_inference_ms += inference_time_ms
        self.stats.segment_count += 1
        self.stats.rtf_per_segment.append(rtf)

        return ASRPrediction(
            text=" ".join(part for part in text_parts if part).strip(),
            start=0.0 if start is None else float(start),
            end=0.0 if end is None else float(end),
            avg_logprob=float(sum(logprobs) / len(logprobs)) if logprobs else float("nan"),
            audio_duration_sec=dur,
            inference_time_ms=inference_time_ms,
            rtf=rtf,
        )

    def transcribe_audio_array(self, path: Path, audio, sr: int) -> ASRPrediction:
        sf.write(path, audio, sr)
        audio_duration_sec = len(audio) / sr
        return self.transcribe_file(path, audio_duration_sec)

    def reset_stats(self) -> None:
        self.stats = BenchmarkStats()
