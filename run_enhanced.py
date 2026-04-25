#!/usr/bin/env python3
"""
run_enhanced.py — EE 679 Enhanced Trailer Analysis Pipeline (v2)

Usage
-----
  # Full pipeline on the Avengers trailer (batch mode)
  python run_enhanced.py

  # Choose model size (tiny.en | small.en | medium.en | large-v2)
  python run_enhanced.py --model small.en

  # Live streaming simulation mode
  python run_enhanced.py --live

  # Live streaming only (skip full batch pipeline)
  python run_enhanced.py --live --live-only

  # Live on a different audio file
  python run_enhanced.py --live --audio path/to/file.wav

Pipeline overview
-----------------
  1. ffmpeg: extract mono 16 kHz WAV from MP4
  2. faster-whisper medium.en + built-in Silero VAD
     → word-level timestamps, hallucination-suppressed segments
  3. opensmile eGeMAPSv02 → emotion label per cue (neutral/excited/tense/sad/angry/fearful)
  4. librosa sound event classifier → LAUGHTER/MUSIC/IMPACT/APPLAUSE labels
  5. Foote novelty score → scene boundary times
  6. Outputs:
       outputs_v2/trailer_clean.srt      ← plain subtitles
       outputs_v2/trailer_annotated.srt  ← with [EMOTION] [EVENT] tags
       outputs_v2/trailer_timeline.json  ← full analysis record
       outputs_v2/trailer_analysis.png   ← 5-panel visualisation
  7. Live streaming simulation (--live flag):
       outputs_v2/trailer_live.srt       ← incrementally written SRT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Locate the trailer MP4 ─────────────────────────────────────────────────
_HERE = Path(__file__).parent
# Prefer Avengers, then any MP4 with "trailer" in the name, then any MP4
_TRAILER = (
    next((_HERE.glob("*[Aa]vengers*.mp4")), None)
    or next((p for p in _HERE.glob("*.mp4") if "trailer" in p.name.lower()), None)
    or next(_HERE.glob("*.mp4"), None)
)
_OUTPUT_DIR = _HERE / "outputs_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EE 679 Enhanced Pipeline v2")
    p.add_argument("--video",  default=str(_TRAILER),
                   help="Path to input video (MP4)")
    p.add_argument("--audio",  default=None,
                   help="Path to input audio (WAV) — skips ffmpeg extraction")
    p.add_argument("--out",    default=str(_OUTPUT_DIR),
                   help="Output directory")
    p.add_argument("--model",  default="medium.en",
                   choices=["tiny.en", "small.en", "medium.en",
                            "medium", "large-v2", "large-v3"],
                   help="Whisper model size (default: medium.en)")
    p.add_argument("--live",   action="store_true",
                   help="Also run live streaming simulation")
    p.add_argument("--live-only", action="store_true",
                   help="Run only live streaming, skip full pipeline")
    p.add_argument("--chunk",  type=float, default=6.0,
                   help="Live pipeline chunk size in seconds (default: 6)")
    p.add_argument("--overlap", type=float, default=1.5,
                   help="Live pipeline overlap in seconds (default: 1.5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Batch pipeline ─────────────────────────────────────────────────────
    if not args.live_only:
        if not args.video or not Path(args.video).exists():
            print(f"[ERROR] Video file not found: {args.video}")
            print("        Place the Avengers trailer MP4 in the project folder,")
            print("        or pass --video path/to/trailer.mp4")
            sys.exit(1)

        from project.trailer_v2 import run_enhanced_pipeline
        results = run_enhanced_pipeline(
            video_path=Path(args.video),
            output_dir=out_dir,
            model_size=args.model,
        )

        print("Outputs written to:", out_dir)
        print("  clean SRT     :", results["clean_srt"])
        print("  annotated SRT :", results["annotated_srt"])
        print("  timeline JSON :", results["timeline_json"])
        if results.get("plot"):
            print("  analysis plot :", results["plot"])
        print(f"\n  {results['cue_count']} subtitle cues  |  RTF={results['rtf']:.3f}  |  "
              f"total={results['total_sec']:.0f}s")

    # ── Live streaming simulation ──────────────────────────────────────────
    if args.live or args.live_only:
        print("\n" + "─" * 60)
        print("  LIVE STREAMING SIMULATION")
        print("─" * 60)

        # Determine audio source
        audio_src = None
        if args.audio:
            audio_src = Path(args.audio)
        else:
            # Use extracted WAV if batch ran, otherwise extract now
            extracted = out_dir / "trailer_audio.wav"
            if extracted.exists():
                audio_src = extracted
            elif args.video and Path(args.video).exists():
                from project.trailer_v2 import extract_audio
                extracted.parent.mkdir(parents=True, exist_ok=True)
                print("  Extracting audio for live pipeline …")
                extract_audio(Path(args.video), extracted)
                audio_src = extracted

        if audio_src is None or not audio_src.exists():
            print("[ERROR] No audio source available for live pipeline.")
            sys.exit(1)

        from project.live_pipeline import run_live_on_file
        live_srt = out_dir / "trailer_live.srt"
        cues = run_live_on_file(
            audio_path=audio_src,
            srt_out=live_srt,
            chunk_sec=args.chunk,
            overlap_sec=args.overlap,
            model_size=args.model,
        )
        print(f"\n  Live SRT written → {live_srt}")
        print(f"  Total cues: {len(cues)}")


if __name__ == "__main__":
    main()
