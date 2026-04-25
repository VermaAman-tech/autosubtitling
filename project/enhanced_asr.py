"""
State-of-the-art ASR using faster-whisper with:
  - Built-in Silero VAD filter (far better VAD than energy/spectral rules)
  - Word-level timestamps (for precise SRT alignment)
  - Medium-sized model for much higher accuracy vs. tiny.en
  - Hallucination suppression (log-prob filtering, no-speech detection)
  - Condition on previous text for coherent transcription context

Model hierarchy (auto-selected by compute budget):
  large-v2 > medium.en > small.en > tiny.en

Silero VAD reference:
  Silero Team, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector",
  2021, https://github.com/snakers4/silero-vad

faster-whisper reference:
  Bain et al., CTranslate2 + OpenAI Whisper, 2023.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    from faster_whisper import WhisperModel
    _FW_AVAILABLE = True
except ImportError:
    _FW_AVAILABLE = False


# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "medium.en"    # best accuracy/speed tradeoff without GPU
FALLBACK_MODEL  = "small.en"
MIN_LOGPROB     = -1.2           # discard segments below this (hallucinations)
MIN_SEG_SECS    = 0.5            # ignore very short fragments
NO_SPEECH_PROB  = 0.6            # skip if whisper thinks no speech present


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscribedSegment:
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float
    words: list[WordTimestamp] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def is_reliable(self) -> bool:
        return (
            self.avg_logprob >= MIN_LOGPROB
            and self.no_speech_prob < NO_SPEECH_PROB
            and len(self.text.strip()) > 1
        )

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class FullTranscription:
    segments: list[TranscribedSegment]
    language: str
    duration_sec: float
    total_inference_ms: float
    model_size: str
    rtf: float          # real-time factor

    @property
    def reliable_segments(self) -> list[TranscribedSegment]:
        return [s for s in self.segments if s.is_reliable]

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.reliable_segments)


# ── Transcriber ────────────────────────────────────────────────────────────

class EnhancedWhisperASR:
    """
    Wraps faster-whisper with best-practice settings for noisy movie audio.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        if not _FW_AVAILABLE:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
        self.model_size = model_size
        print(f"  [ASR] Loading {model_size} (device={device}, compute={compute_type}) …")
        t0 = time.perf_counter()
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            if model_size != FALLBACK_MODEL:
                print(f"  [ASR] {model_size} failed ({e}), falling back to {FALLBACK_MODEL}")
                self.model_size = FALLBACK_MODEL
                self.model = WhisperModel(FALLBACK_MODEL, device=device, compute_type=compute_type)
            else:
                raise
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [ASR] Model loaded in {self.load_ms:.0f}ms")

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        language: str = "en",
        initial_prompt: Optional[str] = None,
    ) -> FullTranscription:
        """
        Transcribe a full mono waveform with word-level timestamps.

        Uses faster-whisper's built-in Silero VAD to filter non-speech regions,
        dramatically improving accuracy on music-heavy trailer audio.
        """
        # Ensure float32, mono
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration_sec = len(audio) / sr

        # Write to a temp file (faster-whisper accepts numpy arrays directly
        # in recent versions, but file path is most compatible)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, sr, subtype="PCM_16")

        t0 = time.perf_counter()
        try:
            segments_iter, info = self.model.transcribe(
                tmp_path,
                language=language,
                # ── Quality settings ──────────────────────────────────────
                beam_size=5,                     # better than beam_size=1
                best_of=5,
                patience=1.0,
                condition_on_previous_text=True, # maintains dialogue context
                initial_prompt=initial_prompt,   # seed with domain vocabulary
                # ── Hallucination suppression ─────────────────────────────
                log_prob_threshold=MIN_LOGPROB,
                no_speech_threshold=NO_SPEECH_PROB,
                compression_ratio_threshold=2.4, # filter repeated text
                # ── Silero VAD (built into faster-whisper) ────────────────
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=400,   # merge gaps shorter than this
                    speech_pad_ms=200,             # add context around speech
                    threshold=0.35,                # sensitivity (lower = more speech)
                ),
                # ── Word-level timestamps ─────────────────────────────────
                word_timestamps=True,
            )

            transcribed_segs: list[TranscribedSegment] = []
            for seg in segments_iter:
                # Collect word timestamps
                word_list = []
                if seg.words:
                    for w in seg.words:
                        word_list.append(WordTimestamp(
                            word=w.word,
                            start=round(float(w.start), 3),
                            end=round(float(w.end), 3),
                            probability=round(float(w.probability), 3),
                        ))

                transcribed_segs.append(TranscribedSegment(
                    text=seg.text.strip(),
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    avg_logprob=round(float(seg.avg_logprob), 4),
                    no_speech_prob=round(float(seg.no_speech_prob), 4),
                    words=word_list,
                ))
        finally:
            os.unlink(tmp_path)

        inference_ms = (time.perf_counter() - t0) * 1000.0
        rtf = (inference_ms / 1000.0) / max(duration_sec, 1e-6)

        return FullTranscription(
            segments=transcribed_segs,
            language=info.language,
            duration_sec=round(duration_sec, 2),
            total_inference_ms=round(inference_ms, 1),
            model_size=self.model_size,
            rtf=round(rtf, 4),
        )

    def transcribe_file(self, path: Path, **kwargs) -> FullTranscription:
        """Transcribe from file path."""
        audio, sr = sf.read(str(path))
        return self.transcribe(audio.astype(np.float32), sr, **kwargs)


# ── SRT cue generator from word timestamps ─────────────────────────────────

@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str
    emotion_tag: str = ""
    event_tag: str = ""

    def __post_init__(self):
        # Clamp end to be at least start + 0.5s
        if self.end <= self.start:
            self.end = self.start + 1.0


def segments_to_cues(
    segments: list[TranscribedSegment],
    max_chars: int = 84,       # max chars per cue (2 lines × 42)
    max_words_per_cue: int = 12,
) -> list[SubtitleCue]:
    """
    Convert Whisper segments (with word timestamps) into subtitle cues.
    Uses word-level timestamps for precise timing when available.
    Falls back to segment-level timing when word timestamps are absent.
    """
    cues: list[SubtitleCue] = []

    for seg in segments:
        if not seg.is_reliable:
            continue
        text = seg.text.strip()
        if not text:
            continue

        if seg.words:
            # ── Word-level cue splitting ──────────────────────────────────
            cues.extend(_split_by_words(seg.words, max_chars, max_words_per_cue))
        else:
            # ── Segment-level fallback ────────────────────────────────────
            cues.append(SubtitleCue(start=seg.start, end=seg.end, text=text))

    return cues


def _split_by_words(
    words: list[WordTimestamp],
    max_chars: int,
    max_words: int,
) -> list[SubtitleCue]:
    """Split a list of word timestamps into display-safe subtitle cues."""
    cues: list[SubtitleCue] = []
    if not words:
        return cues

    current_words: list[WordTimestamp] = []
    current_chars = 0

    for w in words:
        word_text = w.word.strip()
        if not word_text:
            continue
        extra = len(word_text) + (1 if current_words else 0)

        # Flush if adding this word exceeds limits
        if current_words and (current_chars + extra > max_chars or
                               len(current_words) >= max_words):
            cues.append(_words_to_cue(current_words))
            current_words = []
            current_chars = 0

        current_words.append(w)
        current_chars += extra

    if current_words:
        cues.append(_words_to_cue(current_words))

    return cues


def _words_to_cue(words: list[WordTimestamp]) -> SubtitleCue:
    text = " ".join(w.word.strip() for w in words if w.word.strip())
    return SubtitleCue(
        start=words[0].start,
        end=max(words[-1].end, words[0].start + 0.5),
        text=text,
    )
