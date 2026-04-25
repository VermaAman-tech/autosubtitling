"""
On-the-go / live streaming subtitling pipeline.

Simulates real-time subtitle generation by processing audio in overlapping
chunks with a configurable latency budget. Can be used either with a live
microphone stream or a pre-recorded file.

Design
------
  ┌─────────────┐   raw audio    ┌──────────────────────┐
  │  Audio src  │ ─────────────► │  Chunker (ring buf)  │
  │ (mic / file)│                └──────────┬───────────┘
  └─────────────┘                           │ chunk (N sec)
                                            ▼
                              ┌─────────────────────────┐
                              │  Silero VAD filter      │  ← built-in
                              └──────────┬──────────────┘
                                         │ speech regions
                                         ▼
                              ┌─────────────────────────┐
                              │  faster-whisper ASR     │  medium model
                              └──────────┬──────────────┘
                                         │ segments + words
                                         ▼
                              ┌─────────────────────────┐
                              │  Emotion + Event layer  │  opensmile + librosa
                              └──────────┬──────────────┘
                                         │ annotated cues
                                         ▼
                              ┌─────────────────────────┐
                              │  SRT / console output   │
                              └─────────────────────────┘

Overlap strategy
----------------
Each chunk overlaps with the previous by `overlap_sec` seconds.  This
ensures words at chunk boundaries aren't lost or mis-timed.  Deduplication
is done by tracking the last emitted end-time and discarding any cue whose
start is earlier.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import soundfile as sf

from project.enhanced_asr import (
    EnhancedWhisperASR,
    SubtitleCue,
    segments_to_cues,
)
from project.sound_events import (
    classify_segment,
    dominant_event_at_time,
    AudioTimeline,
    EventLabel,
)
from project.emotion_detector import classify_emotion


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class LiveConfig:
    chunk_sec: float   = 5.0        # seconds per processing chunk
    overlap_sec: float = 1.5        # overlap with previous chunk
    model_size: str    = "medium.en"
    language: str      = "en"
    print_live: bool   = True       # print cues to console as they arrive
    detect_emotions: bool = True
    detect_events: bool   = True
    # SRT output (optional — if set, writes incrementally)
    srt_output_path: Optional[Path] = None


# ── Live subtitle cue ──────────────────────────────────────────────────────

@dataclass
class LiveCue:
    index: int
    start: float
    end: float
    text: str
    emotion: str = "neutral"
    event: str   = ""
    latency_ms: float = 0.0    # wall-clock time from chunk start to cue emit

    def to_srt_block(self, include_annotations: bool = True) -> str:
        def _fmt(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int(round((t - int(t)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines = [
            str(self.index),
            f"{_fmt(self.start)} --> {_fmt(self.end)}",
        ]
        body = self.text
        if include_annotations:
            tags = []
            if self.event and self.event not in (EventLabel.SPEECH, EventLabel.AMBIENT):
                tags.append(f"[{self.event}]")
            if self.emotion and self.emotion != "neutral":
                tags.append(f"[{self.emotion.upper()}]")
            if tags:
                body = " ".join(tags) + " " + body
        lines.append(body)
        return "\n".join(lines)

    def to_console_line(self) -> str:
        emo_icon = {"excited": "🤩", "tense": "😬", "angry": "😠",
                    "sad": "😢", "fearful": "😨", "neutral": ""}.get(self.emotion, "")
        ev_icon  = {"LAUGHTER": "😂", "MUSIC": "🎵", "IMPACT": "💥",
                    "APPLAUSE": "👏", "SILENCE": "🔇"}.get(self.event, "")
        t_start  = f"{self.start:6.2f}s"
        return f"  {t_start}  {ev_icon}{emo_icon}  {self.text}"


# ── Chunk iterator ─────────────────────────────────────────────────────────

def _file_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_sec: float,
    overlap_sec: float,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Yield (start_offset_sec, chunk_audio) pairs with overlap.
    The caller is responsible for deduplication.
    """
    chunk_len   = int(chunk_sec * sr)
    overlap_len = int(overlap_sec * sr)
    hop_len     = chunk_len - overlap_len

    pos = 0
    while pos < len(audio):
        chunk = audio[pos: pos + chunk_len]
        yield (pos / sr, chunk)
        pos += hop_len
        if pos + overlap_len >= len(audio):
            break


# ── Live pipeline ──────────────────────────────────────────────────────────

class LiveSubtitlePipeline:
    """
    Processes audio in real-time chunks and emits live subtitle cues.
    """

    def __init__(self, config: Optional[LiveConfig] = None) -> None:
        self.cfg = config or LiveConfig()
        self.asr = EnhancedWhisperASR(model_size=self.cfg.model_size)
        self._cue_index   = 0
        self._last_end    = 0.0   # dedup: don't re-emit cues before this
        self._srt_handle  = None
        if self.cfg.srt_output_path:
            self._srt_handle = open(self.cfg.srt_output_path, "w", encoding="utf-8")

    def close(self) -> None:
        if self._srt_handle:
            self._srt_handle.close()
            self._srt_handle = None

    # ── Core processing ────────────────────────────────────────────────────

    def process_chunk(
        self,
        chunk_audio: np.ndarray,
        sr: int,
        chunk_start_sec: float,
        on_cue: Optional[Callable[[LiveCue], None]] = None,
    ) -> list[LiveCue]:
        """
        Process one audio chunk and return new (non-duplicate) LiveCues.
        """
        chunk_t0 = time.perf_counter()

        # 1. Transcribe with Silero VAD + medium Whisper
        transcription = self.asr.transcribe(
            chunk_audio, sr,
            language=self.cfg.language,
        )

        # 2. Convert to subtitle cues
        raw_cues = segments_to_cues(transcription.reliable_segments)

        # 3. Offset timestamps by chunk position and deduplicate
        new_cues: list[LiveCue] = []
        for cue in raw_cues:
            abs_start = cue.start + chunk_start_sec
            abs_end   = cue.end   + chunk_start_sec

            # Skip if already emitted (overlap region)
            if abs_start < self._last_end - 0.05:
                continue

            self._cue_index += 1

            # 4. Emotion detection on the relevant chunk slice
            emotion = "neutral"
            if self.cfg.detect_emotions:
                seg_audio = chunk_audio[
                    int(cue.start * sr): int(cue.end * sr)
                ]
                if len(seg_audio) > int(0.3 * sr):
                    emo = classify_emotion(seg_audio, sr,
                                          word_count=len(cue.text.split()),
                                          duration_sec=cue.end - cue.start)
                    emotion = emo.label

            # 5. Sound event detection for this window
            event = ""
            if self.cfg.detect_events:
                seg_audio = chunk_audio[
                    max(0, int((cue.start - 0.5) * sr)):
                    min(len(chunk_audio), int((cue.end + 0.5) * sr))
                ]
                if len(seg_audio) > int(0.2 * sr):
                    ev = classify_segment(seg_audio, sr)
                    if ev.label not in (EventLabel.SPEECH, EventLabel.AMBIENT):
                        event = ev.label

            latency_ms = (time.perf_counter() - chunk_t0) * 1000.0
            live_cue = LiveCue(
                index=self._cue_index,
                start=round(abs_start, 3),
                end=round(abs_end, 3),
                text=cue.text,
                emotion=emotion,
                event=event,
                latency_ms=round(latency_ms, 1),
            )
            new_cues.append(live_cue)
            self._last_end = abs_end

            # Callbacks / output
            if self.cfg.print_live:
                print(live_cue.to_console_line())
            if self._srt_handle:
                self._srt_handle.write(live_cue.to_srt_block() + "\n\n")
                self._srt_handle.flush()
            if on_cue:
                on_cue(live_cue)

        return new_cues

    # ── File-based streaming simulation ───────────────────────────────────

    def process_file(
        self,
        audio_path: Path,
        on_cue: Optional[Callable[[LiveCue], None]] = None,
        simulate_realtime: bool = False,
    ) -> list[LiveCue]:
        """
        Simulate live processing of a pre-recorded audio file.

        Parameters
        ----------
        audio_path       : path to WAV file
        on_cue           : callback fired for each new cue
        simulate_realtime: if True, sleep between chunks to match real time
        """
        audio, sr = sf.read(str(audio_path))
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        all_cues: list[LiveCue] = []

        print(f"\n  [live] Streaming {len(audio)/sr:.1f}s of audio  "
              f"(chunk={self.cfg.chunk_sec}s, overlap={self.cfg.overlap_sec}s)")
        print(f"  [live] ── Starting live subtitle output ──\n")

        for chunk_start, chunk_audio in _file_chunks(
            audio, sr, self.cfg.chunk_sec, self.cfg.overlap_sec
        ):
            t_wall0 = time.perf_counter()
            new = self.process_chunk(chunk_audio, sr, chunk_start, on_cue=on_cue)
            all_cues.extend(new)

            if simulate_realtime:
                elapsed = time.perf_counter() - t_wall0
                sleep = self.cfg.chunk_sec - self.cfg.overlap_sec - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        self.close()
        print(f"\n  [live] Done. {len(all_cues)} cues emitted.")
        return all_cues

    # ── Microphone (real-time) support ─────────────────────────────────────

    def start_microphone(self, sample_rate: int = 16000) -> None:
        """
        Start real-time transcription from the default microphone.
        Requires: pip install sounddevice
        Press Ctrl+C to stop.
        """
        try:
            import sounddevice as sd
        except ImportError:
            print("  [live] sounddevice not installed. Run: pip install sounddevice")
            return

        chunk_len   = int(self.cfg.chunk_sec * sample_rate)
        overlap_len = int(self.cfg.overlap_sec * sample_rate)
        ring_buf    = np.zeros(chunk_len * 2, dtype=np.float32)
        pos         = [0]
        wall_start  = [time.perf_counter()]

        print("  [live] Microphone active. Speak now. (Ctrl+C to stop)")

        def _callback(indata, frames, time_info, status):
            chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            # Append to ring buffer
            end = pos[0] + len(chunk)
            if end > len(ring_buf):
                ring_buf[:] = 0
                pos[0] = 0
                end = len(chunk)
            ring_buf[pos[0]: end] = chunk
            pos[0] = end

            if pos[0] >= chunk_len:
                buf_copy = ring_buf[: pos[0]].copy()
                offset_sec = time.perf_counter() - wall_start[0] - self.cfg.chunk_sec
                self.process_chunk(buf_copy, sample_rate, max(0.0, offset_sec))
                # Shift ring buffer (keep overlap)
                ring_buf[:overlap_len] = ring_buf[pos[0] - overlap_len: pos[0]]
                ring_buf[overlap_len:] = 0
                pos[0] = overlap_len

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(sample_rate * 0.1),
            callback=_callback,
        ):
            try:
                while True:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("\n  [live] Stopped.")
        self.close()


# ── Quick test helper ──────────────────────────────────────────────────────

def run_live_on_file(
    audio_path: Path,
    srt_out: Optional[Path] = None,
    chunk_sec: float = 6.0,
    overlap_sec: float = 1.5,
    model_size: str = "medium.en",
) -> list[LiveCue]:
    """Convenience wrapper for file-based live simulation."""
    cfg = LiveConfig(
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
        model_size=model_size,
        print_live=True,
        detect_emotions=True,
        detect_events=True,
        srt_output_path=srt_out,
    )
    pipeline = LiveSubtitlePipeline(cfg)
    return pipeline.process_file(audio_path)
