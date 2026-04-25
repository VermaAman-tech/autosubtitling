# EE 679 — Adaptive Auto-Subtitling for Movies

> **Course Project**: Noise-Aware Movie Auto-Subtitling with Classical Speech Processing and Whisper ASR

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Repository Structure](#repository-structure)
4. [Setup & Installation](#setup--installation)
5. [Quick Start](#quick-start)
6. [Running on Your Own Videos](#running-on-your-own-videos)
7. [Batch Evaluation on Many Trailers](#batch-evaluation-on-many-trailers)
8. [Experiment Suite](#experiment-suite)
9. [Metrics](#metrics)
10. [Results](#results)
11. [Module Reference](#module-reference)

---

## Project Overview

This project builds an **end-to-end auto-subtitling pipeline** for movie trailers that goes beyond plain transcription. It produces **annotated SRT** files with:

- **Emotion tags**: `[SAD]😢`, `[TENSE]😬`, `[ANGRY]😠`, `[EXCITED]🤩`, `[FEARFUL]😨`
- **Sound event tags**: `[MUSIC/heroic]🎵`, `[LAUGHTER]😂`, `[IMPACT]💥`, `[APPLAUSE]👏`
- **Scene boundary markers** detected from audio character changes

The pipeline combines:
- **faster-whisper** (CTranslate2 Whisper) for ASR with **Silero VAD** for speech filtering
- **opensmile eGeMAPSv02** (88-dim acoustic features) for emotion classification
- **librosa** hand-crafted features for sound event detection and scene segmentation

---

## Pipeline Architecture

```
INPUT: Movie/Trailer (MP4)
         │
         ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Audio Extraction (ffmpeg)         │
│  MP4 → mono 16kHz PCM WAV                  │
│  Uses imageio-ffmpeg (bundled binary)       │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 2: ASR — faster-whisper + Silero VAD │
│                                             │
│  • Silero VAD filters non-speech regions    │
│    (music, silence, effects) BEFORE ASR     │
│  • Whisper medium.en (769M params, int8)    │
│  • beam_size=5, best_of=5                   │
│  • Word-level timestamps for SRT alignment  │
│  • Hallucination suppression:               │
│    - log_prob threshold > -1.2              │
│    - no_speech_prob < 0.6                   │
│    - compression_ratio < 2.4               │
│  • Domain prompt seeding (character names)  │
│                                             │
│  Output: list of TranscribedSegment with    │
│          word-level timestamps              │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 3: Sound Event Detection (librosa)   │
│                                             │
│  Classifies 1.5s chunks (0.5s hop) into:   │
│    SPEECH   — voiced, formant structure     │
│    MUSIC    — harmonic, wide bandwidth      │
│    LAUGHTER — 4-8 Hz AM rate, voiced        │
│    IMPACT   — sharp onset, low-freq burst   │
│    APPLAUSE — flat spectrum, high ZCR       │
│    SILENCE  — RMS < -45 dB                  │
│                                             │
│  Scene boundaries via Foote novelty score   │
│  on MFCC+chroma+spectral-contrast features  │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 4: Emotion Detection (opensmile)     │
│                                             │
│  Per subtitle cue:                          │
│    Extract 88-dim eGeMAPSv02 functionals    │
│    → F0 mean/std, loudness, HNR, jitter,   │
│      shimmer, alpha ratio, speech rate      │
│    → Compute arousal + valence dimensions   │
│    → Rule-based label via Russell's         │
│      circumplex model                       │
│                                             │
│  Labels: neutral, excited, tense, sad,      │
│          angry, fearful                     │
│  Fallback: librosa RMS if opensmile absent  │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 5: Output Generation                 │
│                                             │
│  • trailer_clean.srt     — plain subtitles  │
│  • trailer_annotated.srt — with tags inline │
│  • trailer_timeline.json — full analysis    │
│  • trailer_analysis.png  — 5-panel plot     │
│    (waveform, spectrogram, events,          │
│     emotions, subtitle regions + RMS)       │
└─────────────────────────────────────────────┘
```

---

## Repository Structure

```
EE 679 Project/
├── project/                      # Core Python package
│   ├── __init__.py
│   ├── enhanced_asr.py           # ★ faster-whisper + Silero VAD wrapper
│   │                             #   EnhancedWhisperASR, SubtitleCue, segments_to_cues
│   ├── emotion_detector.py       # ★ opensmile eGeMAPSv02 emotion classifier
│   │                             #   classify_emotion → EmotionResult
│   ├── sound_events.py           # ★ librosa sound event classifier + scene boundaries
│   │                             #   classify_segment, build_audio_timeline, detect_scene_boundaries
│   ├── trailer_v2.py             # ★ Main pipeline orchestrator (v2)
│   │                             #   run_enhanced_pipeline() — calls stages 1-5
│   ├── live_pipeline.py          #   Live/streaming subtitle pipeline
│   │                             #   LiveSubtitlePipeline — chunk-based streaming
│   ├── asr.py                    #   Basic Whisper ASR (used by old experiments)
│   ├── audio_utils.py            #   Audio enhancement: Wiener, spectral subtraction,
│   │                             #   adaptive routing, VAD methods, noise synthesis
│   ├── dataset.py                #   LibriSpeech test-clean downloader/preparer
│   ├── metrics.py                #   WER, CER, timestamp MAE, CPS violations
│   ├── subtitles.py              #   Basic SRT writer
│   ├── marvel_pipeline.py        #   Old v1 pipeline (for exp1 comparison)
│   └── pipeline.py               #   Old pipeline utilities
│
├── run_enhanced.py               # ★ CLI entry point — run pipeline on a single video
├── run_trailer_experiments.py    # ★ Experiment runner — evaluates against reference SRTs
├── run_full_experiments.py       #   Full 6-experiment suite (LibriSpeech + trailers)
├── run_experiments.py            #   Legacy experiment stub
│
├── requirements.txt              # All Python dependencies
├── Dockerfile                    # Docker build (optional)
│
├── *.mp4                         # Input trailer videos
├── *.srt                         # Reference (ground-truth) SRT files
│                                 #   (must share same basename as .mp4)
│
├── outputs_v2/                   # Avengers medium.en outputs
│   ├── trailer_clean.srt
│   ├── trailer_annotated.srt
│   ├── trailer_timeline.json
│   ├── trailer_analysis.png
│   └── trailer_audio.wav
│
├── outputs_spiderman_v2/         # Spider-Man medium.en outputs
│   └── (same structure)
│
├── experiment_results/           # Experiment outputs
│   ├── EXPERIMENT_REPORT.md      # Auto-generated markdown report
│   ├── fig_model_comparison.png  # WER/CER/RTF comparison charts
│   ├── fig_emotion_events.png    # Emotion + event distribution charts
│   ├── raw_results.json          # Machine-readable results
│   ├── avengers_medium_en/       # Per-trailer per-model outputs
│   └── spiderman_medium_en/
│
└── outputs/                      # Old experiment outputs (LibriSpeech)
```

---

## Setup & Installation

```bash
# 1. Clone/navigate to the project
cd "/Users/amanverma/Downloads/EE 679 Project"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. (First run only) The medium.en model (~300MB) auto-downloads from HuggingFace
#    and caches in ~/.cache/huggingface/
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `faster-whisper` | CTranslate2 Whisper ASR + bundled Silero VAD |
| `opensmile` | eGeMAPSv02 acoustic features for emotion detection |
| `librosa` | Sound event features (HPSS, onset, spectral) |
| `soundfile` | WAV I/O |
| `imageio-ffmpeg` | Bundled ffmpeg for audio extraction |
| `jiwer` | WER / CER computation |
| `matplotlib` | Visualisation |
| `scipy` | Peak-picking for scene boundaries |
| `pandas`, `numpy` | Data handling |
| `pysrt` | SRT parsing |

---

## Quick Start

### Run on a single video (default: Avengers trailer)

```bash
# Uses medium.en model, outputs to outputs_v2/
python run_enhanced.py

# Specify a different video
python run_enhanced.py --video path/to/movie.mp4 --out my_outputs/

# Choose model size
python run_enhanced.py --model tiny.en       # fastest, lower quality
python run_enhanced.py --model medium.en     # default, best tradeoff
python run_enhanced.py --model large-v2      # highest quality, slowest

# Also run live streaming simulation
python run_enhanced.py --live
```

### Run experiments with WER evaluation

```bash
# Evaluates all *.mp4 files that have matching *.srt references
python run_trailer_experiments.py --models medium.en

# Compare multiple models
python run_trailer_experiments.py --models tiny.en medium.en
```

---

## Running on Your Own Videos

### Single video (no reference SRT needed)

```bash
python run_enhanced.py --video /path/to/movie.mp4 --out /path/to/output/
```

This produces:
- `trailer_clean.srt` — plain subtitles
- `trailer_annotated.srt` — with `[EMOTION]` `[EVENT]` tags
- `trailer_timeline.json` — full analysis data
- `trailer_analysis.png` — 5-panel plot

### Full movie

The pipeline works on any length audio. For a full movie:
```bash
python run_enhanced.py --video "movie.mp4" --model medium.en --out movie_output/
```

> **Note**: A 2-hour movie takes ~15-20 min on CPU with medium.en (RTF ≈ 0.1-0.5).

---

## Batch Evaluation on Many Trailers

To test on 50-100 trailers with ground-truth SRT files:

### Step 1: Organize your data

Place all trailers and their reference SRTs in the project directory (or a subdirectory). 
Each video **must** have a matching `.srt` file with the **same basename**:

```
trailers/
├── avengers.mp4
├── avengers.srt          ← ground-truth reference
├── spider_man.mp4
├── spider_man.srt
├── batman_begins.mp4
├── batman_begins.srt
└── ...
```

### Step 2: Run batch evaluation

The `run_trailer_experiments.py` script auto-discovers all `*.mp4` files with matching `*.srt`:

```bash
# Evaluate all trailers in current directory
python run_trailer_experiments.py --models medium.en --output-dir batch_results/

# Compare models
python run_trailer_experiments.py --models tiny.en medium.en --output-dir batch_results/
```

### Step 3: Check outputs

```
batch_results/
├── EXPERIMENT_REPORT.md          # Full markdown report with tables
├── fig_model_comparison.png      # WER/CER/RTF bar charts
├── fig_emotion_events.png        # Emotion + event distributions
├── raw_results.json              # Machine-readable for your own analysis
├── trailer1_medium_en/           # Per-trailer outputs
│   ├── trailer_clean.srt
│   ├── trailer_annotated.srt
│   ├── trailer_timeline.json
│   └── trailer_analysis.png
└── ...
```

### Where to get trailers + ground-truth SRTs

1. **YouTube trailers** — download with `yt-dlp`:
   ```bash
   # Install yt-dlp
   pip install yt-dlp
   
   # Download trailer + auto-generated subtitles
   yt-dlp -f "best[height<=1080]" --write-auto-sub --sub-lang en \
          --convert-subs srt -o "trailers/%(title)s.%(ext)s" \
          "https://youtube.com/watch?v=VIDEO_ID"
   ```

2. **OpenSLR / LibriSpeech** — for controlled speech experiments:
   ```bash
   python run_full_experiments.py --data-dir data --output-dir librispeech_results
   ```

---

## Experiment Suite

### Experiment 1: Trailer Auto-Subtitling
- **What**: Run pipeline on real movie trailers, compare WER vs reference SRT
- **Script**: `run_trailer_experiments.py`
- **Key metrics**: WER, CER, timestamp MAE, CPS violations

### Experiment 2: Noise Robustness (LibriSpeech)
- **What**: Add synthetic noise (white, pink, babble, soundtrack) at various SNR levels
- **Script**: `run_full_experiments.py` → `exp2_noise_robustness()`
- **Variables**: 4 noise types × 6 SNR levels (−5 to 20 dB) × 4 enhancement systems
- **Key finding**: Raw ASR is robust at high SNR; adaptive routing helps at low SNR

### Experiment 3: VAD Ablation Study
- **What**: Compare 3 VAD methods (energy, MFCC, spectral) on speech coverage
- **Script**: `run_full_experiments.py` → `exp3_vad_ablation()`
- **Key finding**: Spectral VAD has fewer false positives under music

### Experiment 4: Enhancement Method Ablation
- **What**: Wiener vs spectral subtraction vs adaptive routing under soundtrack noise
- **Script**: `run_full_experiments.py` → `exp4_enhancement_ablation()`
- **Key finding**: Wiener filter is gentler; spectral subtraction over-suppresses tonal noise

### Experiment 5: Model Size Comparison
- **What**: tiny.en vs base.en (and now medium.en) — accuracy vs latency
- **Script**: `run_full_experiments.py` → `exp5_model_comparison()` or `run_trailer_experiments.py`
- **Key finding**: medium.en reduces WER by 2× at 8× more compute

### Experiment 6: Subtitle Quality & Readability
- **What**: Characters-per-second (CPS) violations, timestamp accuracy
- **Script**: `run_full_experiments.py` → `exp6_subtitle_quality()`
- **Key finding**: CPS violations come from long segments, not ASR errors

---

## Metrics

| Metric | Formula | What it measures | Target |
|--------|---------|-----------------|--------|
| **WER** | (S+D+I)/N | Word-level transcription accuracy | < 15% |
| **CER** | char-level edit distance / ref chars | Character-level accuracy | < 10% |
| **RTF** | inference_time / audio_duration | Speed (< 1 = faster than real-time) | < 1.0 |
| **CPS** | characters / cue_duration | Readability (Netflix limit: 20 CPS) | ≤ 20 |
| **TS-MAE** | mean |ref_timestamp − pred_timestamp| | Timing accuracy | < 0.5s |

---

## Results

### Model Comparison on Real Trailers

| Trailer | Model | WER ↓ | CER ↓ | RTF ↓ |
|---------|-------|-------|-------|-------|
| **Avengers** | medium.en | **4.88%** | **3.10%** | 0.107 |
| Avengers | tiny.en | 10.57% | 7.01% | 0.013 |
| **Spider-Man** | medium.en | **14.36%** | **8.41%** | 0.412 |
| Spider-Man | tiny.en | 21.29% | 17.61% | 0.013 |

### Emotion Detection (medium.en)

| Trailer | SAD | TENSE | ANGRY | NEUTRAL | Total |
|---------|-----|-------|-------|---------|-------|
| Avengers | 8 (62%) | 0 | 0 | 5 (38%) | 13 |
| Spider-Man | 2 (11%) | 2 (11%) | 2 (11%) | 12 (67%) | 18 |

### Sound Event Detection

| Trailer | SPEECH | MUSIC | LAUGHTER | IMPACT | Boundaries |
|---------|--------|-------|----------|--------|-----------|
| Avengers | 32 | 27 | 9 | 0 | 33 |
| Spider-Man | 50 | 39 | 10 | 1 | 41 |

---

## Module Reference

### `project/enhanced_asr.py`
**Core ASR module.** Wraps faster-whisper with Silero VAD, word timestamps, and hallucination filtering.

- `EnhancedWhisperASR(model_size="medium.en")` — load model
- `.transcribe(audio, sr)` → `FullTranscription` with `reliable_segments`
- `segments_to_cues(segments)` → `list[SubtitleCue]` — splits by word timestamps

### `project/emotion_detector.py`
**Acoustic emotion classifier.** Uses opensmile eGeMAPSv02 features with rule-based classification.

- `classify_emotion(audio, sr, word_count, duration_sec)` → `EmotionResult`
  - `.label` — one of: neutral, excited, tense, sad, angry, fearful
  - `.valence` — negative to positive (−1 to +1)
  - `.arousal` — calm to excited (0 to 1)

### `project/sound_events.py`
**Sound event classifier and scene boundary detector.**

- `classify_segment(audio, sr)` → `SoundEvent` (SPEECH/MUSIC/LAUGHTER/IMPACT/APPLAUSE/SILENCE)
- `build_audio_timeline(audio, sr)` → `AudioTimeline` with events + boundaries
- `detect_scene_boundaries(audio, sr)` → `list[SceneBoundary]` via Foote novelty

### `project/trailer_v2.py`
**Main pipeline orchestrator.** Calls all stages and writes outputs.

- `run_enhanced_pipeline(video_path, output_dir, model_size)` → dict with file paths + metrics

### `project/audio_utils.py`
**Audio enhancement toolkit.** Wiener filter, spectral subtraction, adaptive routing, VAD methods, noise synthesis.

- `enhance_audio(audio, method)` — apply enhancement
- `should_enhance(features)` — adaptive routing decision
- `VAD_METHODS` — dict of VAD functions (energy, mfcc, spectral)

### `project/live_pipeline.py`
**Streaming subtitle pipeline.** Processes audio in overlapping chunks with deduplication.

- `LiveSubtitlePipeline(config)` — create pipeline
- `.process_file(audio_path)` — simulate live on a file
- `.start_microphone()` — real-time from mic (requires sounddevice)
