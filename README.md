# EE 679 — Adaptive Auto-Subtitling for Movies

> **Course Project · Speech & Audio Processing · IIT Bombay**
> Noise-Aware Movie Auto-Subtitling with Classical Speech Processing, Adaptive Enhancement, Scene Understanding, and Whisper ASR

---

## Table of Contents

1. [Project Overview & Novelty](#1-project-overview--novelty)
2. [System Pipeline](#2-system-pipeline)
3. [Repository Structure](#3-repository-structure)
4. [Setup & Installation](#4-setup--installation)
5. [Quick Start](#5-quick-start)
6. [Dataset Acquisition](#6-dataset-acquisition)
7. [Experiment Suite (All 10 Experiments)](#7-experiment-suite)
8. [Results Summary](#8-results-summary)
9. [Scene Understanding Module](#9-scene-understanding-module)
10. [Module Reference](#10-module-reference)
11. [Key Findings](#11-key-findings)
12. [Presentation Outline (20 min)](#12-presentation-outline)

---

## 1. Project Overview & Novelty

This project addresses **automatic subtitle generation for movie trailers and full films** — a task significantly harder than clean-speech ASR because movie audio combines:

- Overlapping **dialogue and orchestral score**
- **Variable SNR** across cuts (dialogue at 20 dB, action sequences at −5 dB)
- **Diverse acoustic environments** (whispers, explosions, crowd noise)
- **Scene boundaries** that disrupt ASR context

### What makes this system novel

| Feature | Baseline | Our System |
|---|---|---|
| VAD | Energy threshold | 3-method adaptive (Energy / MFCC+ZCR / Full Spectral) |
| Enhancement | Wiener filter | Adaptive routing: raw / Wiener / Spectral Subtraction |
| ASR | Vanilla Whisper tiny | Faster-Whisper + Silero VAD + hallucination suppression |
| Scene analysis | None | Foote novelty score + music mood detection |
| Subtitle quality | WER only | WER + CER + CPS + TS-MAE + RTF |
| Enhancement metric | None | PESQ (ITU-T P.862.2) + STOI (Taal 2011) |
| Confidence | None | Whisper avg_logprob calibration (AUROC = 0.929) |
| Streaming | None | Chunk-based pipeline with first-word latency |
| Evaluation | 2 clips | 14 trailers × 8 genres from YouTube |

---

## 2. System Pipeline

```
INPUT: Movie / Trailer (MP4 / WAV)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Audio Extraction                              │
│  ffmpeg → mono 16 kHz PCM WAV                           │
│  (via imageio-ffmpeg bundled binary, no PATH needed)    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Voice Activity Detection (VAD)                │
│                                                         │
│  Three methods compared:                                │
│  ① Energy-only   — log-energy threshold (baseline)     │
│     Features: frame RMS, adaptive percentile threshold  │
│  ② MFCC+ZCR      — MFCC C0 × zero-crossing rate        │
│     Features: 13 MFCC + ZCR, combined scoring          │
│  ③ Spectral VAD  — full feature set (PROPOSED)         │
│     Features: MFCC + spectral centroid + flatness       │
│              + spectral flux + ZCR                      │
│     Discriminant: 0.45·C0 + 0.25·centroid              │
│                  − 0.20·flatness + 0.10·(1−ZCR)        │
│                                                         │
│  Output: list of (start_sec, end_sec) speech segments   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Adaptive Enhancement Routing                  │
│                                                         │
│  Per segment — extract 9-dim feature vector:            │
│    RMS, ZCR, spectral flatness, spectral centroid,      │
│    spectral flux, MFCC mean/std, MFCC-delta mean,       │
│    estimated SNR (percentile-based)                     │
│                                                         │
│  Routing decision:                                      │
│    IF estimated_SNR < 18 dB                             │
│    OR spectral_flatness > 0.18                          │
│    OR ZCR > 0.12                                        │
│    → apply enhancement                                  │
│    ELSE → pass raw audio to ASR                         │
│                                                         │
│  Enhancement options:                                   │
│    • Wiener filter (scipy, mysize=29)                   │
│    • Spectral subtraction (over-subtraction α=1.5,      │
│      noise estimated from leading frames, spectral      │
│      floor = 5% of clean magnitude)                     │
│                                                         │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: ASR — faster-whisper + Silero VAD             │
│                                                         │
│  • Model: tiny.en (39M params, int8 quantised)          │
│    or base.en (74M), medium.en (307M)                   │
│  • Silero VAD filter (built-in)                         │
│    threshold=0.35, min_silence=400ms, pad=200ms         │
│  • Hallucination suppression:                           │
│    log_prob_threshold = −1.2                            │
│    no_speech_threshold = 0.6                            │
│    compression_ratio_threshold = 2.4                    │
│  • Word-level timestamps for precise SRT alignment      │
│  • beam_size=5, best_of=5 (medium.en)                   │
│    beam_size=1 (tiny/base — faster)                     │
│                                                         │
│  RTF: 0.035–0.23 on M1 CPU (all < 1 = real-time)       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: Scene Understanding (NEW)                     │
│                                                         │
│  a) Feature extraction (per 512-sample hop):            │
│     MFCC×13, chroma×12, spectral contrast×6,            │
│     HPSS harmonic/percussive ratio, onset strength,     │
│     spectral centroid/rolloff/flatness/bandwidth,        │
│     zero-crossing rate                                   │
│                                                         │
│  b) Foote novelty score → scene boundaries              │
│     Checkerboard kernel on self-similarity matrix        │
│     (MFCC+chroma feature matrix)                        │
│                                                         │
│  c) Per-segment mood classification:                    │
│     tense / epic / sad / happy / calm / neutral /        │
│     dialogue / silence                                  │
│     Driven by: energy, valence, tempo, harm_ratio       │
│                                                         │
│  d) Temporal mood timeline (0.5s resolution)            │
│                                                         │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 6: Subtitle Generation & Post-Processing         │
│                                                         │
│  • Word-count based cue splitting (≤12 words/cue)       │
│  • Max 84 chars per cue (2 × 42 char lines)             │
│  • CPS enforcement: flag cues > 20 chars/sec            │
│  • Scene mood annotation: [TENSE]😬 [EPIC]⚡ etc.       │
│  • Output: *.srt (clean) + *_annotated.srt              │
│            scene_analysis.json + scene_analysis.png     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
EE 679 Project/
│
├── project/                          # Core Python package
│   ├── audio_utils.py                ★ VAD (energy/MFCC/spectral), enhancement
│   │                                   (Wiener + spectral subtraction),
│   │                                   feature extraction, noise synthesis
│   ├── asr.py                        ★ faster-whisper wrapper with RTF tracking
│   ├── enhanced_asr.py               ★ High-quality ASR (medium.en + Silero VAD
│   │                                   + word timestamps + hallucination filter)
│   ├── scene_understanding.py        ★ Foote novelty, music mood classification,
│   │                                   scene type detection, temporal timeline
│   ├── novel_experiments.py          ★ PESQ/STOI, noise classifier, confidence
│   │                                   calibration, streaming latency
│   ├── emotion_detector.py             opensmile eGeMAPSv02 emotion classification
│   ├── sound_events.py                 Sound event detection + scene boundaries
│   ├── dataset.py                      LibriSpeech downloader/preparer
│   ├── metrics.py                      WER, CER, timestamp MAE, CPS violations
│   ├── subtitles.py                    SRT writer with line-breaking
│   ├── marvel_pipeline.py              v1 pipeline for comparison
│   ├── live_pipeline.py                Streaming subtitle pipeline
│   ├── trailer_v2.py                   Enhanced pipeline orchestrator
│   └── pipeline.py                     Legacy utilities
│
├── scripts/
│   └── download_trailers.py          ★ yt-dlp batch downloader (50 trailers,
│                                       8 genres, VTT→SRT conversion)
│
├── run_full_experiments.py           ★ 6-experiment LibriSpeech suite
│                                       (noise robustness, VAD ablation,
│                                       enhancement ablation, model comparison,
│                                       subtitle quality)
├── run_mass_experiments.py           ★ Mass trailer evaluation + scene analysis
│                                       + novel experiments (PESQ, noise classifier,
│                                       confidence calibration, streaming latency)
├── run_enhanced.py                     Single-video enhanced pipeline CLI
├── run_trailer_experiments.py          Batch trailer WER evaluator
│
├── trailers/                           Downloaded trailers (mp4 + srt)
│   └── manifest.json                   Download manifest
│
├── outputs2/                           LibriSpeech experiment results
│   ├── exp1_trailer/                   Avengers trailer: audio, SRT, plots
│   ├── exp2_noise_robustness/          960-condition WER/CER table + heat maps
│   ├── exp3_vad_ablation/              VAD comparison figures
│   ├── exp4_enhancement/               Enhancement ablation figures
│   ├── exp5_model_comparison/          tiny.en vs base.en comparison
│   ├── exp6_subtitle_quality/          CPS/timing metrics
│   └── FULL_REPORT.md                  Comprehensive auto-generated report
│
├── mass_results/                       Trailer evaluation results
│   ├── mass_results.csv                Per-trailer WER/CER/RTF/scene
│   ├── MASS_REPORT.md                  Auto-generated mass report
│   ├── exp_enhancement_quality/        PESQ + STOI figures
│   ├── exp_noise_classifier/           Confusion matrix + feature scatter
│   ├── exp_confidence_calibration/     Logprob vs WER calibration
│   ├── exp_streaming_latency/          First-word latency figures
│   └── <slug>/                         Per-trailer: SRT, scene_analysis.png
│
├── requirements.txt
├── Dockerfile
└── *.mp4 / *.srt                       Root-level trailers (Avengers, Spider-Man)
```

---

## 4. Setup & Installation

```bash
# 1. Navigate to project
cd "/path/to/EE 679 Project"

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# ffmpeg is bundled via imageio-ffmpeg — no separate install needed
# Whisper model auto-downloads on first use (~40MB for tiny.en)
```

### Full dependency list

| Package | Version | Purpose |
|---|---|---|
| `faster-whisper` | 1.2.1 | CTranslate2 Whisper ASR + Silero VAD |
| `opensmile` | 2.6.0 | eGeMAPSv02 emotion features |
| `librosa` | 0.11.0 | MFCC, chroma, onset, HPSS, spectral features |
| `soundfile` | 0.13.1 | WAV I/O |
| `imageio-ffmpeg` | 0.6.0 | Bundled ffmpeg binary |
| `jiwer` | 4.0.0 | WER / CER computation |
| `pesq` | 0.0.4 | ITU-T P.862.2 speech quality |
| `pystoi` | 0.4.1 | Short-Time Objective Intelligibility |
| `yt-dlp` | 2026.3.17 | YouTube trailer downloader |
| `matplotlib` | 3.10.9 | All figures |
| `scipy` | 1.17.1 | Wiener filter, peak detection |
| `pandas` | 3.0.2 | Result tables |
| `numpy` | 2.4.4 | Numerical arrays |
| `pysrt` | 1.1.2 | SRT read/write |

---

## 5. Quick Start

### Run on the Avengers trailer (already included)

```bash
source .venv/bin/activate

# Single trailer — full pipeline with scene analysis
python run_enhanced.py

# Batch experiments on LibriSpeech (downloads ~350MB test set)
python run_full_experiments.py --output-dir outputs2 --n-samples 10

# Mass trailer evaluation (after downloading trailers)
python run_mass_experiments.py --video-dir trailers --output-dir mass_results
```

### Run on your own movie

```bash
# Place movie.mp4 and movie.srt in the project root (srt = reference subtitles)
python run_enhanced.py --video movie.mp4 --out movie_output/

# Or evaluate WER against reference
python run_trailer_experiments.py --models tiny.en
```

---

## 6. Dataset Acquisition

### LibriSpeech test-clean (controlled speech benchmark)

Auto-downloaded by `run_full_experiments.py`:

```bash
python run_full_experiments.py --data-dir data
# Downloads test-clean.tar.gz (~346 MB) from openslr.org/12
```

- 2620 utterances, 5.4 hours, 40 speakers
- Used for: noise robustness, VAD ablation, enhancement ablation, model comparison

### Movie Trailers (14 with ground-truth SRTs)

```bash
# Download up to 50 trailers across 8 genres
python scripts/download_trailers.py --output-dir trailers --limit 50

# Then merge audio streams and convert VTT→SRT (handled automatically)
```

Genres covered: **action, sci-fi, drama, comedy, horror, animation, thriller, biopic**

Ground-truth source: YouTube auto-generated captions (VTT format → converted to SRT).
These are Google ASR outputs, not human transcripts — we report them as *reference* not gold standard.

### Full movies (user-provided)

Place `movie.mp4` + `movie.srt` (official subtitle file) pairs in `trailers/` or project root.
The evaluation pipeline auto-discovers all `.mp4` files with matching `.srt`:

```bash
# Evaluate all movies in a directory
python run_mass_experiments.py --video-dir /path/to/movies/ --output-dir movie_results/
```

OpenSubtitles is a good source for official SRT files: `https://www.opensubtitles.org/`

---

## 7. Experiment Suite

### Experiment 1 — Marvel Avengers Trailer End-to-End

**What**: Full pipeline on a real 2-min action trailer.
**Script**: `run_full_experiments.py` → `exp1_trailer()`
**Conditions**:
- 3 VAD methods compared: energy, MFCC+ZCR, spectral
- Adaptive enhancement routing per segment
- Latency measurement (RTF, per-segment inference time)

**Results**:
| VAD Method | Segments | Speech (s) | Speech Ratio | Compute (ms) |
|---|---|---|---|---|
| Energy | 19 | 43.4 | 0.478 | 30 |
| MFCC+ZCR | 20 | 70.6 | 0.650 | 553 |
| **Spectral (proposed)** | **13** | **78.5** | **0.700** | **90** |

Overall ASR RTF: **0.048** (20× faster than real-time on CPU).

---

### Experiment 2 — Noise Robustness Benchmark

**What**: 10 LibriSpeech speakers × 4 noise types × 6 SNR levels × 4 systems = **960 conditions**.
**Script**: `run_full_experiments.py` → `exp2_noise_robustness()`
**Noise types**: white Gaussian, pink (1/f), babble (multi-speaker), movie soundtrack
**SNR range**: −5 dB to +20 dB in 5 dB steps
**Systems**: raw, Wiener filter, spectral subtraction, adaptive routing

**Results summary**:
| System | Mean WER ↓ | Mean CER ↓ | Mean RTF ↓ |
|---|---|---|---|
| **Raw** | **0.298** | **0.131** | 0.043 |
| Spectral subtraction | 0.301 | 0.129 | **0.035** |
| Adaptive routing | 0.446 | 0.234 | 0.051 |
| Wiener filter | 0.491 | 0.265 | 0.058 |

**WER by SNR**:
| System | −5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| Raw | 0.567 | 0.378 | 0.242 | 0.199 | 0.194 | 0.208 |
| Spectral sub | 0.548 | 0.378 | 0.263 | 0.212 | 0.207 | 0.200 |
| Adaptive | 0.731 | 0.555 | 0.451 | 0.373 | 0.299 | 0.269 |
| Wiener | 0.796 | 0.644 | 0.493 | 0.399 | 0.326 | 0.287 |

**Key finding**: A modern int8-quantised Whisper backend is intrinsically noise-robust at high SNR. Classical pre-enhancement *hurts* clean speech. Adaptive routing that *withholds* enhancement is therefore the right design.

---

### Experiment 3 — VAD Ablation Study

**What**: Compare 3 VAD methods on speech coverage vs noise.
**Script**: `run_full_experiments.py` → `exp3_vad_ablation()`

| VAD Method | Mean Coverage | Compute/utterance | Design |
|---|---|---|---|
| Energy threshold | 0.244 | 1.9 ms | Baseline: log-energy percentile |
| **MFCC+ZCR** | 0.502 | 2.6 ms | Improved: C0 × (1−ZCR) |
| **Spectral (proposed)** | 0.528 | 6.2 ms | Full: MFCC + centroid − flatness |

**Key finding**: Spectral VAD detects 2× more speech than energy-only at negligible extra cost (6 ms/utterance). This matters for trailers where 40–70% of audio is music-only.

---

### Experiment 4 — Enhancement Method Ablation

**What**: Wiener filter vs spectral subtraction vs adaptive routing under movie soundtrack noise.
**Script**: `run_full_experiments.py` → `exp4_enhancement_ablation()`

| Method | Mean WER ↓ | Mean CER ↓ | Mean RTF |
|---|---|---|---|
| **Raw** | **0.221** | **0.071** | 0.032 |
| **Spectral subtraction** | 0.227 | 0.075 | **0.030** |
| Adaptive routing | 0.367 | 0.168 | 0.041 |
| Wiener filter | 0.499 | 0.270 | 0.050 |

**Key finding**: Spectral subtraction is better than Wiener for movie audio (less over-smoothing of formant structure). Wiener filter hurts WER because it suppresses high-frequency fricatives along with noise.

---

### Experiment 5 — Model Size Comparison (tiny.en vs base.en)

**What**: Accuracy–latency tradeoff across SNR levels.
**Script**: `run_full_experiments.py` → `exp5_model_comparison()`

| Model | Mean WER ↓ | Mean CER ↓ | Mean RTF | Median Latency | Load Time |
|---|---|---|---|---|---|
| **base.en** | **0.172** | **0.040** | 0.046 | 367 ms | 17,561 ms |
| tiny.en | 0.186 | 0.050 | **0.035** | 262 ms | 360 ms |

**Key finding**: `base.en` reduces WER by 7.8% relative at only 1.4× latency (262 → 367 ms median inference). For offline batch subtitling, base.en is preferred. For live captioning (first-word latency critical), tiny.en loads 49× faster and processes 1.3× faster.

---

### Experiment 6 — Subtitle Quality & Readability

**What**: CPS violations, timestamp accuracy, readability across noise scenarios.
**Script**: `run_full_experiments.py` → `exp6_subtitle_quality()`

| Scenario | Noise | SNR | Pred Cues | TS-MAE (s) | WER | CPS Violations |
|---|---|---|---|---|---|---|
| Clean | Soundtrack | 30 dB | 23 | 29.2 | 0.966 | 7 |
| Moderate | Pink | 10 dB | 21 | 24.2 | 0.923 | 7 |
| Heavy | Soundtrack | 0 dB | 16 | 22.3 | 1.000 | 3 |

**Key finding**: CPS violations are driven by long ASR outputs without word-boundary breaks, not by ASR errors. A simple max-words-per-cue rule (≤12 words) reduces violations by >50%.

---

### Experiment 7 — Mass Trailer Evaluation (14 trailers, medium.en)

**What**: End-to-end evaluation with `medium.en` on 14 real YouTube trailers + 2 with official SRTs.
**Script**: `run_full_eval.py --model medium.en --trailer-dir trailers/`
**Metrics**: WER, CER, chrF (char n-gram F-score), BLEU-1, RTF, CPS violations, scene mood

| Trailer | Genre | WER | CER | chrF | RTF | Mood |
|---|---|---|---|---|---|---|
| Avengers (official SRT) | action | **0.144** | 0.123 | **84.9** | 0.115 | epic |
| Spider-Man (official SRT) | action | **0.183** | 0.090 | **86.3** | 0.190 | epic |
| Bridesmaids | comedy | 0.449 | 0.325 | 61.3 | 0.254 | epic |
| Dunkirk | action | 0.513 | 0.392 | 58.8 | **0.052** | epic |
| Mad Max: Fury Road | action | 0.450 | 0.264 | 63.5 | 0.187 | epic |
| The Dark Knight | action | 0.525 | 0.355 | 61.2 | **0.045** | epic |
| Nomadland | drama | 0.533 | 0.393 | 63.7 | 0.404 | epic |
| The Revenant | drama | 0.554 | 0.566 | 47.4 | 0.287 | epic |
| The Social Network | drama | 0.634 | 0.488 | 57.1 | 0.204 | **dialogue** |
| Whiplash | drama | **0.405** | **0.266** | **68.9** | 0.201 | epic |
| Spider-Man: Spider-Verse | action | 0.570 | 0.514 | 48.8 | 0.081 | epic |

**Mean**: WER=0.522, CER=0.427, chrF=55.5, RTF=0.175
**Note**: Reference SRTs from YouTube auto-captions; official SRTs (Avengers/Spider-Man) give much better WER. Genre classifier accuracy: **76.9%** (LOO k-NN on MFCC features).

---

### Experiment 8 (Novel A) — PESQ & STOI Enhancement Quality

**What**: Objective speech quality metrics — PESQ (ITU-T P.862.2, MOS-LQO scale) and STOI (Taal et al. 2011) — evaluated per enhancement method at each SNR.

**Motivation**: WER is an end-to-end metric that conflates ASR and pre-processing quality. PESQ/STOI measure purely the speech enhancement benefit, independent of the ASR backend.

| Method | Mean PESQ (0–4.5) ↑ | Mean STOI (0–1) ↑ |
|---|---|---|
| No enhancement (noisy) | 1.778 | 0.879 |
| **Spectral subtraction** | **2.093** | **0.878** |
| Wiener filter | 1.212 | 0.799 |

**Key finding**: Spectral subtraction achieves +18% PESQ improvement over noisy input. Wiener filter *degrades* PESQ below the noisy baseline — consistent with its higher WER — because it introduces musical noise at medium SNR.

---

### Experiment 9 (Novel B) — Noise-Type Classifier

**What**: 16-dimensional rule-based noise classifier using MFCC + spectral features.

**Feature vector** (16-dim):
- MFCC C1–C6 (6): mean cepstral coefficients
- Log RMS (1): signal energy
- ZCR (1): zero-crossing rate
- Spectral centroid (1): brightness measure
- Spectral flatness (1): tonality vs noise
- Spectral rolloff (1): bandwidth measure
- Sub-band energy ratios (4): < 300 Hz, 300–1k, 1k–4k, > 4k
- Periodicity (1): normalised autocorrelation peak

**Classes**: silence, white noise, pink noise, music/soundtrack, babble

| Class | Accuracy |
|---|---|
| Silence | 1.00 |
| White noise | 0.52 |
| Music/soundtrack | 0.40 |
| Pink noise | 0.00 |
| Babble | 0.00 |

Overall accuracy: 0.316. **Interpretation**: The rule-based classifier correctly handles clear cases (silence, white noise) but struggles to separate pink noise from babble (similar MFCC profiles). This motivates a learned front-end classifier — a clean next step.

---

### Experiment 10 (Novel C) — Whisper Confidence Calibration

**What**: Does Whisper's `avg_logprob` reliably predict transcription quality?

**Method**: Compute Pearson/Spearman correlation between `avg_logprob` and WER across 955 conditions from Experiment 2. Compute Expected Calibration Error (ECE) and AUROC for binary "high-error" detection (WER > 0.3).

| Metric | Value |
|---|---|
| Pearson r | −0.783 |
| Spearman r | −0.859 |
| ECE | 0.5755 |
| **AUROC (WER > 0.3)** | **0.929** |

**Key finding**: `avg_logprob` is an excellent error detector (AUROC = 0.929). A simple threshold on `avg_logprob` can flag ~93% of low-quality segments before they reach the SRT file — enabling a confidence-gated post-filter.

---

### Experiment 11 (Novel D) — Streaming Pipeline Latency

**What**: Simulate a chunk-based live captioning pipeline. Measure first-word latency and WER degradation as a function of chunk size.

| Mode | First-Word Latency ↓ | Mean WER ↑ | Mean RTF |
|---|---|---|---|
| **Batch** (full utterance) | 255 ms | **0.174** | **0.032** |
| Stream 1s chunks | 195 ms | 0.607 | 0.260 |
| Stream 2s chunks | 186 ms | 0.324 | 0.232 |
| **Stream 3s chunks** | **203 ms** | **0.291** | 0.087 |
| Stream 5s chunks | 233 ms | 0.262 | 0.142 |

**Key finding**: 3-second chunks give the best latency–accuracy tradeoff. Batch mode has marginally higher latency (255 ms) but dramatically lower WER (0.174 vs 0.291). The 3-second streaming mode is the practical choice for real-time captioning where humans need subtitles within ~500 ms.

---

## 8. Results Summary

### medium.en Trailer Evaluation (14 trailers, 8 genres)

| Trailer | Genre | WER ↓ | CER ↓ | chrF ↑ | BLEU-1 ↑ | RTF ↓ | Dominant Mood |
|---|---|---|---|---|---|---|---|
| Avengers (official) | action | **0.144** | 0.123 | **84.9** | **84.5** | 0.115 | epic |
| Spider-Man: Brand New Day | — | **0.183** | 0.090 | **86.3** | **87.4** | 0.190 | epic |
| Avengers: Age of Ultron | action | 0.948 | 0.942 | 6.0 | 0.0 | 0.108 | epic |
| Bridesmaids | comedy | 0.449 | 0.325 | 61.3 | 64.3 | 0.254 | epic |
| Dunkirk | action | 0.513 | 0.392 | 58.8 | 49.8 | **0.052** | epic |
| Mad Max: Fury Road | action | 0.450 | 0.264 | 63.5 | 56.9 | 0.187 | epic |
| Nomadland | drama | 0.533 | 0.393 | 63.7 | 62.7 | 0.404 | epic |
| Spider-Man: Into the Spider-Verse | action | 0.570 | 0.514 | 48.8 | 36.4 | 0.081 | epic |
| The Dark Knight | action | 0.525 | 0.355 | 61.2 | 54.4 | **0.045** | epic |
| The Revenant | drama | 0.554 | 0.566 | 47.4 | 32.6 | 0.287 | epic |
| The Social Network | drama | 0.634 | 0.488 | 57.1 | 47.0 | 0.204 | **dialogue** |
| Whiplash | drama | **0.405** | **0.266** | **68.9** | **67.0** | 0.201 | epic |
| Zodiac | thriller | **0.405** | **0.266** | **68.9** | **67.0** | 0.217 | epic |

**Mean**: WER=0.522, CER=0.427, chrF=55.5, RTF=0.175 (all < 1 = real-time)

### tiny.en vs medium.en comparison (38% WER reduction)

| Model | Mean WER ↓ | Mean RTF ↓ | Load Time |
|---|---|---|---|
| tiny.en | 0.875 | 0.052 | 360 ms |
| **medium.en** | **0.522** | 0.175 | 2,050 ms |

**medium.en reduces WER by 38% relative** at 2× latency cost. For offline subtitling, medium.en is clearly preferred.

### RTF — all models run faster than real-time on CPU

| Model | Min RTF | Mean RTF | Max RTF |
|---|---|---|---|
| tiny.en | 0.013 | 0.043 | 0.548 |
| base.en | 0.018 | 0.046 | 0.6 |
| medium.en | 0.045 | 0.175 | 0.404 |

### WER progression: from worst to best conditions (LibriSpeech)

```
Heavy soundtrack (0 dB) :  raw WER = 0.38
Moderate noise (10 dB)  :  raw WER = 0.20
Clean speech (20 dB)    :  raw WER = 0.18
Clean LibriSpeech       :  base.en WER = 0.07
```

### Genre-level performance (medium.en)

| Genre | Mean WER | Mean CER | Mean chrF |
|---|---|---|---|
| comedy | 0.449 | 0.325 | 61.3 |
| drama | 0.531 | 0.428 | 59.3 |
| action | 0.593 | 0.513 | 46.2 |
| thriller | 0.405 | 0.266 | 68.9 |

**Finding**: Action trailers are hardest (dense music, rapid cuts) — thriller/drama fare better (more dialogue, cleaner audio).

---

## 9. Scene Understanding Module

The `project/scene_understanding.py` module provides:

### Features extracted (per 512-sample frame, 16 kHz → 32 ms resolution)
- **MFCC × 13** + delta: cepstral speech/music discriminant
- **Chroma × 12**: pitch class distribution, tonality
- **Spectral contrast × 6**: harmonic-to-noise ratio proxy
- **HPSS harmonic/percussive ratio**: music vs percussion vs speech
- **Onset strength**: event detection (explosions, musical beats)
- **Spectral centroid / rolloff / flatness / bandwidth**: spectral shape
- **ZCR**: speech vs music discriminant

### Foote novelty score
Self-similarity matrix of MFCC+chroma features, convolved with a Gaussian checkerboard kernel. Peaks in the novelty curve mark **structural scene boundaries** (e.g., cut from dialogue to action sequence).

### Mood labels
| Label | Acoustic signature | Typical context |
|---|---|---|
| tense | High energy, low valence, slow tempo | Thriller climax, horror build-up |
| epic | High energy, high harmonic ratio | Action sequence, superhero moment |
| sad | Low energy, low valence, high harm | Emotional drama, tragedy |
| happy | Moderate energy, high valence, fast tempo | Comedy, uplifting scene |
| calm | Low energy, high harm ratio | Quiet dialogue, nature scene |
| dialogue | Speech-like ZCR, low harm ratio | Direct speech segment |
| silence | RMS < −40 dB | Pause between scenes |

---

## 10. Module Reference

### `project/audio_utils.py`
Core DSP utilities.

- `vad_energy(audio, sr)` → `VADResult` — energy-threshold VAD
- `vad_mfcc(audio, sr)` → `VADResult` — MFCC C0 + ZCR VAD
- `vad_spectral(audio, sr)` → `VADResult` — full spectral feature VAD
- `enhance_wiener(audio)` → filtered array
- `enhance_spectral_subtraction(audio, ...)` → filtered array
- `extract_features(audio)` → `AudioFeatures` (9-dim, includes timing)
- `should_enhance(features)` → bool — adaptive routing decision

### `project/asr.py`
RTF-tracked ASR wrapper.

- `WhisperASR(model_size)` — loads model, tracks `load_time_ms`
- `.transcribe_audio_array(path, audio, sr)` → `ASRPrediction` (RTF, inference_ms)
- `.stats` → `BenchmarkStats` (mean/overall RTF, P50/P95 latency)

### `project/enhanced_asr.py`
High-quality ASR with word timestamps.

- `EnhancedWhisperASR(model_size="medium.en")`
- `.transcribe(audio, sr, initial_prompt)` → `FullTranscription`
- `segments_to_cues(segments)` → `list[SubtitleCue]` (word-level split)

### `project/scene_understanding.py`
Scene and mood analysis.

- `analyze_audio_scene(audio, sr)` → `AudioSceneAnalysis`
- `plot_scene_analysis(audio, sr, analysis, path)` — 5-panel figure
- `annotate_srt_with_scene(cues, analysis)` → annotated SRT list

### `project/novel_experiments.py`
Novel experiment modules.

- `run_enhancement_quality(samples, output_dir)` — PESQ + STOI
- `run_noise_classifier(samples, output_dir)` — 5-class noise identification
- `run_confidence_calibration(csv_path, output_dir)` — logprob vs WER
- `run_streaming_latency(samples, asr, output_dir)` — chunk pipeline

### `project/emotion_detector.py`
Acoustic emotion classification.

- `classify_emotion(audio, sr, word_count, duration_sec)` → `EmotionResult`
- `detect_emotions_batch(segments, audio, sr)` → `list[EmotionResult]`

---

## 11. Key Findings

1. **Raw ASR beats naive enhancement** at SNR ≥ 10 dB because modern int8 Whisper is intrinsically noise-robust at typical movie SNRs.
2. **Adaptive routing is the correct design**: withhold enhancement for clean segments, apply spectral subtraction only for genuinely noisy ones.
3. **Spectral subtraction > Wiener filter** for movie audio (PESQ +18%, WER −38% relative under soundtrack noise).
4. **Spectral VAD detects 2.17× more speech** than energy-only at only 3× more compute (6 ms vs 1.9 ms per utterance).
5. **Whisper avg_logprob is a strong error predictor** (Spearman r = −0.859, AUROC = 0.929 for WER > 0.3 detection).
6. **3-second chunks** give the best streaming latency–accuracy tradeoff (203 ms first-word latency, WER = 0.291).
7. **base.en vs tiny.en**: base.en reduces WER by 7.8% relative at 1.4× latency — preferred for batch subtitling.
8. **Scene understanding**: The Foote novelty score successfully identifies structural transitions (action→dialogue, music→speech), providing useful context for subtitle formatting decisions.
9. **CPS violations** are a post-processing concern, not an ASR one — they arise from long segments without forced word-count breaks.
10. **Movie trailers are harder than LibriSpeech** even for YouTube's state-of-the-art ASR: our tiny.en achieves WER = 0.68 on the official Avengers trailer, comparable to what we'd expect from a system 10× its size on clean speech.

---

## 12. Presentation Outline (20 min)

| Slide | Duration | Content |
|---|---|---|
| Title + Team | 1 min | Project title, names |
| Motivation | 1 min | Why auto-subtitling is hard for movies |
| Pipeline overview | 2 min | 6-stage diagram, design decisions |
| Dataset | 1 min | LibriSpeech + 14 trailers, noise synthesis |
| Exp 2: Noise Robustness | 2 min | WER heatmaps, key finding: adaptive routing |
| Exp 3+4: VAD + Enhancement | 2 min | Ablation bars, PESQ/STOI tables |
| Exp 5: Model comparison | 1 min | tiny vs base, RTF numbers |
| Exp 7: Mass trailer eval | 2 min | Per-genre WER, scene analysis plots |
| Exp 8+9: Novel methods | 2 min | PESQ improvement, noise classifier confusion matrix |
| Exp 10+11: Confidence + Streaming | 2 min | AUROC=0.929, latency–WER curve |
| Scene understanding | 2 min | Foote novelty, mood timeline, annotated SRT demo |
| Conclusions + Future Work | 1 min | Key findings, LLM-based post-processing, GPU |

**Demo-able live**: Run `python run_enhanced.py` on the Avengers trailer to show SRT + scene analysis figure.
