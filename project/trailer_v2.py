"""
Enhanced end-to-end trailer analysis pipeline (v2).

Improvements over v1
────────────────────
  ASR      : faster-whisper medium.en (vs tiny.en)
             + built-in Silero VAD (vs custom energy/spectral VAD)
             + word-level timestamps (vs segment-level)
             + hallucination suppression via log-prob + no-speech thresholds

  Emotion  : opensmile eGeMAPS v02 acoustic features per dialogue segment
             → 6-class label + valence/arousal dimensions

  Events   : librosa-based sound event classifier
             → LAUGHTER, MUSIC, IMPACT, APPLAUSE, SILENCE labels
             → scene boundary detection (Foote novelty score)

  SRT out  : three output flavours
             1. clean.srt          — plain clean subtitles
             2. annotated.srt      — emotion + event tags inline
             3. timeline.json      — full analysis record

  Plot     : waveform + spectrogram + event timeline + scene boundaries
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import librosa
    import librosa.display
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

from project.enhanced_asr import (
    EnhancedWhisperASR,
    SubtitleCue,
    segments_to_cues,
    DEFAULT_MODEL,
)
from project.emotion_detector import classify_emotion, EMOTION_EMOJI
from project.sound_events import (
    build_audio_timeline,
    dominant_event_at_time,
    EventLabel,
    EVENT_EMOJI,
)

# Domain prompt seeds Whisper with MCU vocabulary for better accuracy
MARVEL_PROMPT = (
    "Avengers, SHIELD, Loki, Fury, Stark, Banner, Thor, Natasha, "
    "Hawkeye, Hulk, Iron Man, Captain America, tesseract, arc reactor, "
    "scepter, Chitauri, helicarrier, HYDRA, genius billionaire playboy philanthropist"
)

SR = 16_000


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, wav_path: Path) -> None:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"

    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(SR), "-vn", "-acodec", "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# SRT writers
# ---------------------------------------------------------------------------

def _fmt_time(t: float) -> str:
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_clean_srt(cues: list[SubtitleCue], path: Path) -> None:
    """Plain SRT — no annotations, Netflix-style two-line wrapping."""
    lines = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_time(cue.start)} --> {_fmt_time(cue.end)}")
        lines.append(_wrap_text(cue.text, 42))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_annotated_srt(cues: list[SubtitleCue], path: Path) -> None:
    """Annotated SRT — adds emotion + event inline tags."""
    lines = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_time(cue.start)} --> {_fmt_time(cue.end)}")
        body = cue.text
        prefixes = []
        if cue.event_tag:
            prefixes.append(cue.event_tag)
        if cue.emotion_tag and cue.emotion_tag not in ("[NEUTRAL]",):
            prefixes.append(cue.emotion_tag)
        if prefixes:
            body = " ".join(prefixes) + "  " + body
        lines.append(_wrap_text(body, 56))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _wrap_text(text: str, max_chars: int = 42) -> str:
    words = text.split()
    if not words:
        return ""
    lines, current, cur_len = [], [], 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and cur_len + extra > max_chars:
            lines.append(" ".join(current))
            current, cur_len = [word], len(word)
        else:
            current.append(word)
            cur_len += extra
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines[:2])   # max 2 display lines


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_enhanced_pipeline(
    video_path: Path,
    output_dir: Path,
    model_size: str = DEFAULT_MODEL,
    run_live: bool = False,
) -> dict:
    """
    Full enhanced pipeline: ASR + emotion + events + SRT + plot.

    Returns a summary dict with all metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    print(f"\n{'═'*60}")
    print(f"  EE 679 — Enhanced Trailer Analysis Pipeline v2")
    print(f"{'═'*60}")

    # ── 1. Audio extraction ────────────────────────────────────────────────
    print("\n[1/5] Extracting audio …")
    wav_path = output_dir / "trailer_audio.wav"
    extract_audio(video_path, wav_path)
    audio = load_audio(wav_path)
    duration_sec = len(audio) / SR
    print(f"      Audio: {duration_sec:.1f}s  @ {SR}Hz")

    # ── 2. Enhanced ASR (Silero VAD + medium model) ────────────────────────
    print(f"\n[2/5] Transcribing with faster-whisper [{model_size}] + Silero VAD …")
    asr = EnhancedWhisperASR(model_size=model_size)
    transcription = asr.transcribe(
        audio, SR,
        language="en",
        initial_prompt=MARVEL_PROMPT,
    )
    reliable = transcription.reliable_segments
    print(f"      {len(transcription.segments)} raw segments → "
          f"{len(reliable)} reliable  |  RTF={transcription.rtf:.3f}  |  "
          f"lang={transcription.language}")

    # ── 3. Sound event timeline ────────────────────────────────────────────
    print("\n[3/5] Building sound event timeline …")
    timeline = build_audio_timeline(audio, SR, chunk_sec=1.5, hop_sec=0.5)
    print(f"      {len(timeline.events)} event spans  |  "
          f"{len(timeline.boundaries)} scene boundaries  |  "
          f"compute={timeline.compute_ms:.0f}ms")
    for ev in timeline.events[:6]:
        print(f"        {ev.emoji} {ev.label:<10}  {ev.start:.1f}s–{ev.end:.1f}s  "
              f"conf={ev.confidence:.2f}"
              + (f"  ({ev.sub_label})" if ev.sub_label else ""))

    # ── 4. Emotion + event annotation per subtitle cue ────────────────────
    print("\n[4/5] Annotating cues with emotion + events …")
    raw_cues = segments_to_cues(reliable)
    annotated_cues: list[SubtitleCue] = []

    for cue in raw_cues:
        seg_audio = audio[int(cue.start * SR): int(cue.end * SR)]

        # Emotion
        emotion = "neutral"
        emotion_tag = ""
        if len(seg_audio) > int(0.3 * SR):
            emo = classify_emotion(
                seg_audio, SR,
                word_count=len(cue.text.split()),
                duration_sec=cue.end - cue.start,
            )
            emotion = emo.label
            if emotion != "neutral":
                emoji = EMOTION_EMOJI.get(emotion, "")
                emotion_tag = f"[{emotion.upper()}]{emoji}"

        # Sound event at the midpoint of this cue
        mid = (cue.start + cue.end) / 2.0
        dom_ev = dominant_event_at_time(timeline, mid)
        event_tag = ""
        if dom_ev and dom_ev.label not in (EventLabel.SPEECH, EventLabel.AMBIENT):
            ev_emoji = EVENT_EMOJI.get(dom_ev.label, "")
            sub = f"/{dom_ev.sub_label}" if dom_ev.sub_label else ""
            event_tag = f"[{dom_ev.label}{sub}]{ev_emoji}"

        cue.emotion_tag = emotion_tag
        cue.event_tag   = event_tag
        annotated_cues.append(cue)

        if emotion != "neutral" or event_tag:
            print(f"        {cue.start:.1f}s  {event_tag} {emotion_tag}  \"{cue.text[:50]}\"")

    print(f"      {len(annotated_cues)} subtitle cues generated")

    # ── 5. Write outputs ───────────────────────────────────────────────────
    print("\n[5/5] Writing outputs …")

    clean_srt      = output_dir / "trailer_clean.srt"
    annotated_srt  = output_dir / "trailer_annotated.srt"
    timeline_json  = output_dir / "trailer_timeline.json"
    plot_path      = output_dir / "trailer_analysis.png"

    write_clean_srt(annotated_cues, clean_srt)
    write_annotated_srt(annotated_cues, annotated_srt)

    # Timeline JSON
    timeline_data = {
        "video": video_path.name,
        "duration_sec": round(duration_sec, 2),
        "asr_model": model_size,
        "rtf": transcription.rtf,
        "segments": [
            {
                "start": s.start, "end": s.end, "text": s.text,
                "avg_logprob": s.avg_logprob,
            }
            for s in reliable
        ],
        "cues": [
            {
                "index": i + 1,
                "start": c.start, "end": c.end, "text": c.text,
                "emotion": c.emotion_tag, "event": c.event_tag,
            }
            for i, c in enumerate(annotated_cues)
        ],
        "sound_events": [
            {
                "start": e.start, "end": e.end,
                "label": e.label, "confidence": e.confidence,
                "sub_label": e.sub_label,
            }
            for e in timeline.events
        ],
        "scene_boundaries": [
            {"time": b.time, "confidence": b.confidence, "description": b.description}
            for b in timeline.boundaries
        ],
    }
    with open(timeline_json, "w") as fh:
        json.dump(timeline_data, fh, indent=2)

    # Plot
    if _PLOT_AVAILABLE:
        _plot_analysis(audio, SR, timeline, annotated_cues,
                       timeline.boundaries, plot_path)

    total_ms = (time.perf_counter() - t_total) * 1000.0

    print(f"\n{'═'*60}")
    print(f"  ✓ Pipeline complete in {total_ms/1000:.1f}s")
    print(f"  • clean SRT      → {clean_srt.name}")
    print(f"  • annotated SRT  → {annotated_srt.name}")
    print(f"  • timeline JSON  → {timeline_json.name}")
    if _PLOT_AVAILABLE:
        print(f"  • analysis plot  → {plot_path.name}")
    print(f"{'═'*60}\n")

    return {
        "clean_srt":     str(clean_srt),
        "annotated_srt": str(annotated_srt),
        "timeline_json": str(timeline_json),
        "plot":          str(plot_path) if _PLOT_AVAILABLE else None,
        "cue_count":     len(annotated_cues),
        "rtf":           transcription.rtf,
        "total_sec":     round(total_ms / 1000, 1),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_analysis(
    audio: np.ndarray,
    sr: int,
    timeline,
    cues: list[SubtitleCue],
    boundaries,
    out_path: Path,
) -> None:
    """5-panel analysis figure."""
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(5, 1, hspace=0.55, figure=fig)

    times = np.arange(len(audio)) / sr
    xlim  = (0, len(audio) / sr)

    # ── Panel 1: Waveform + scene boundaries ──────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(times, audio, lw=0.35, color="steelblue", alpha=0.8)
    for b in boundaries:
        ax0.axvline(b.time, color="red", lw=1.0, alpha=0.7, linestyle="--")
    ax0.set_xlim(xlim)
    ax0.set_ylabel("Amp")
    ax0.set_title("Waveform  (— scene boundaries)", fontsize=10, fontweight="bold")
    ax0.grid(alpha=0.15)

    # ── Panel 2: Mel spectrogram ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = librosa.display.specshow(mel_db, x_axis="time", y_axis="mel",
                                       sr=sr, hop_length=512, ax=ax1, cmap="magma")
    ax1.set_title("Mel Spectrogram (80 bands)", fontsize=10)
    fig.colorbar(img, ax=ax1, format="%+2.0f dB", pad=0.01)
    ax1.set_xlim(xlim)

    # ── Panel 3: Sound event timeline ─────────────────────────────────────
    EVENT_COLORS = {
        EventLabel.SPEECH:   "#4CAF50",
        EventLabel.MUSIC:    "#2196F3",
        EventLabel.LAUGHTER: "#FF9800",
        EventLabel.IMPACT:   "#F44336",
        EventLabel.APPLAUSE: "#9C27B0",
        EventLabel.SILENCE:  "#9E9E9E",
        EventLabel.AMBIENT:  "#607D8B",
    }
    ax2 = fig.add_subplot(gs[2])
    for ev in timeline.events:
        color = EVENT_COLORS.get(ev.label, "#607D8B")
        ax2.barh(0.5, ev.end - ev.start, left=ev.start, height=0.7,
                 color=color, alpha=0.75 * ev.confidence + 0.25)
        if ev.end - ev.start > 1.5:
            lbl = f"{ev.emoji}{ev.label[:4]}"
            ax2.text((ev.start + ev.end) / 2, 0.5, lbl,
                     ha="center", va="center", fontsize=7, color="white",
                     fontweight="bold")
    # legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=l) for l, c in EVENT_COLORS.items()]
    ax2.legend(handles=legend_els, loc="upper right", fontsize=6, ncol=4)
    ax2.set_xlim(xlim)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_title("Sound Event Timeline", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.grid(alpha=0.15, axis="x")

    # ── Panel 4: Emotion timeline (subtitle cues coloured by emotion) ──────
    EMO_COLORS = {
        "neutral":  "#78909C",
        "excited":  "#FDD835",
        "tense":    "#EF5350",
        "sad":      "#42A5F5",
        "angry":    "#FF5722",
        "fearful":  "#AB47BC",
    }
    ax3 = fig.add_subplot(gs[3])
    for cue in cues:
        emo   = cue.emotion_tag.split("]")[0].lstrip("[").lower() if cue.emotion_tag else "neutral"
        color = EMO_COLORS.get(emo, "#78909C")
        ax3.barh(0.5, cue.end - cue.start, left=cue.start, height=0.6,
                 color=color, alpha=0.85)
    from matplotlib.patches import Patch as MPatch
    emo_legend = [MPatch(facecolor=c, label=l) for l, c in EMO_COLORS.items()]
    ax3.legend(handles=emo_legend, loc="upper right", fontsize=6, ncol=6)
    ax3.set_xlim(xlim)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_title("Emotion Timeline (per subtitle cue)", fontsize=10, fontweight="bold")
    ax3.grid(alpha=0.15, axis="x")

    # ── Panel 5: Subtitle text overlay ────────────────────────────────────
    ax4 = fig.add_subplot(gs[4])
    rms_env = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
    rms_t   = librosa.frames_to_time(np.arange(len(rms_env)), sr=sr, hop_length=512)
    ax4.fill_between(rms_t, rms_env, alpha=0.4, color="grey", label="RMS")
    for cue in cues:
        ax4.axvspan(cue.start, cue.end, alpha=0.12, color="blue")
        snippet = cue.text[:28] + ("…" if len(cue.text) > 28 else "")
        ax4.text(cue.start + 0.1, ax4.get_ylim()[1] * 0.6, snippet,
                 fontsize=6, color="navy", rotation=0,
                 verticalalignment="center", clip_on=True)
    ax4.set_xlim(xlim)
    ax4.set_title("Subtitle Cue Regions + RMS Envelope", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("RMS")
    ax4.grid(alpha=0.15)

    fig.suptitle(
        "EE 679 — Enhanced Trailer Analysis (v2)\n"
        f"ASR: faster-whisper medium.en + Silero VAD | "
        f"Emotion: opensmile eGeMAPSv02 | Events: librosa",
        fontsize=11, fontweight="bold", y=0.99,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Plot saved → {out_path.name}")
