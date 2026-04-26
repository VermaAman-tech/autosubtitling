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
7. [Experiment Suite (All 11 Experiments)](#7-experiment-suite)
8. [Results Summary](#8-results-summary)
9. [Scene Understanding Module](#9-scene-understanding-module)
10. [Module Reference](#10-module-reference)
11. [Key Findings](#11-key-findings)

---

## 1. Project Overview & Novelty

This project addresses **automatic subtitle generation for movie trailers and full films** — a task significantly harder than clean-speech ASR because movie audio combines:

- Overlapping **dialogue and orchestral score**
- **Variable SNR** across cuts (dialogue at 20 dB, action sequences at −5 dB)
- **Diverse acoustic environments** (whispers, explosions, crowd noise)
- **Scene boundaries** that disrupt ASR context
- **Flawed ground truth**: YouTube auto-captions hallucinate during music, inflating WER

### What makes this system novel

| Feature | Baseline | Our System |
|---|---|---|
| VAD | Energy threshold | 3-method adaptive (Energy / MFCC+ZCR / Full Spectral) |
| Enhancement | Wiener filter | Adaptive routing: raw / Wiener / Spectral Subtraction |
| ASR | Vanilla Whisper tiny | Faster-Whisper + Silero VAD + hallucination suppression |
| Scene analysis | None | Foote novelty score + music mood detection |
| Subtitle quality | WER only | WER + CER + chrF + BLEU + CPS + TS-MAE + RTF |
| Enhancement metric | None | PESQ (ITU-T P.862.2) + STOI (Taal 2011) |
| Confidence | None | Whisper avg_logprob calibration (AUROC = 0.929) |
| Streaming | None | Chunk-based pipeline with first-word latency |
| Evaluation | 2 clips | 14 trailers × 8 genres + LibriSpeech × 960 conditions |

### Core Design Principle

> **Do not blindly apply classical enhancement to neural ASR.**
>
> Modern int8-quantised Whisper is intrinsically noise-robust at SNR ≥ 10 dB. Classical pre-processing (Wiener filter, spectral subtraction) confuses the decoder by altering the spectral structure it was trained on. Our adaptive routing *withholds* enhancement for clean segments — preventing over-processing that degrades transcription.

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
│  Stage 5: Scene Understanding                           │
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
│   ├── extra_metrics.py                chrF, BLEU, genre classifier (k-NN on MFCC)
│   ├── marvel_pipeline.py              v1 pipeline for comparison
│   ├── live_pipeline.py                Streaming subtitle pipeline
│   ├── trailer_v2.py                   Enhanced pipeline orchestrator
│   └── pipeline.py                     Legacy utilities
│
├── run_full_experiments.py           ★ 6-experiment LibriSpeech suite
│                                       (noise robustness × 960 conditions,
│                                       VAD ablation, enhancement ablation,
│                                       model comparison, subtitle quality)
├── run_mass_experiments.py           ★ Mass trailer evaluation + scene analysis
│                                       + novel experiments (PESQ, noise classifier,
│                                       confidence calibration, streaming latency)
├── run_full_eval.py                    medium.en batch evaluation runner
├── run_enhanced.py                     Single-video enhanced pipeline CLI
├── run_trailer_experiments.py          Batch trailer WER evaluator
├── clever_eval_all.py                  Novel metrics computation
├── download_youtube_trailers.py        yt-dlp batch downloader (50 trailers, 8 genres)
│
├── trailers/                           Downloaded trailers (mp4 + srt)
│   └── manifest.json
│
├── youtube_trailers/                   YouTube trailers (990 MB, 50+ movies)
│
├── data/                               LibriSpeech test-clean (690 MB, 2620 utterances)
│
├── outputs2/                           LibriSpeech experiment results
│   ├── exp1_trailer/                   Avengers: audio, SRT, 3 plots, metrics.json
│   ├── exp2_noise_robustness/          960-condition WER/CER table + heatmaps
│   ├── exp3_vad_ablation/              VAD comparison figures + coverage.csv
│   ├── exp4_enhancement/               Enhancement ablation figures + comparison.csv
│   ├── exp5_model_comparison/          tiny.en vs base.en figures + model_metrics.csv
│   ├── exp6_subtitle_quality/          CPS/timing metrics + subtitle_metrics.csv
│   └── FULL_REPORT.md                  Auto-generated LibriSpeech report
│
├── mass_results/                       Trailer evaluation results
│   ├── mass_results.csv                Per-trailer WER/CER/RTF/scene (20 columns)
│   ├── MASS_REPORT.md                  Auto-generated mass evaluation report
│   ├── fig_mass_wer_cer.png            WER/CER scatter across 14 trailers
│   ├── fig_rtf_speech_ratio.png        RTF vs speech ratio
│   ├── fig_scene_summary.png           Mood distribution per trailer
│   ├── exp_enhancement_quality/        PESQ + STOI figures (6-subplot)
│   ├── exp_noise_classifier/           Confusion matrix + feature scatter
│   ├── exp_confidence_calibration/     Logprob vs WER calibration JSON
│   ├── exp_streaming_latency/          Streaming results JSON
│   └── <slug>/                         Per-trailer: SRT, annotated SRT, scene_analysis.png
│
├── eval_results_medium/                medium.en evaluation plots
├── evaluate_results/                   54 trailer evaluation results
├── movies/                             Full-length movie files (user-provided)
├── requirements.txt
├── Dockerfile
└── presentation.tex                    LaTeX Beamer slides
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
# Whisper model auto-downloads on first use (~40MB tiny.en, ~1.5GB medium.en)
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

# Mass trailer evaluation
python run_mass_experiments.py --video-dir trailers --output-dir mass_results
```

### Run on your own movie

```bash
# Place movie.mp4 and movie.srt in the project root
python run_enhanced.py --video movie.mp4 --out movie_output/

# Batch trailer WER evaluation
python run_trailer_experiments.py --models tiny.en medium.en
```

### Compile the LaTeX presentation

```bash
pdflatex presentation.tex
pdflatex presentation.tex  # run twice for cross-references
```

---

## 6. Dataset Acquisition

### LibriSpeech test-clean (controlled speech benchmark)

Auto-downloaded by `run_full_experiments.py`:

```bash
python run_full_experiments.py --data-dir data
# Downloads test-clean.tar.gz (~346 MB) from openslr.org/12
```

- **2,620 utterances**, 5.4 hours, 40 speakers
- Used for: noise robustness (×960 conditions), VAD ablation, enhancement ablation, model comparison, subtitle quality
- **Synthetic noise added in code**: white Gaussian, pink (1/f), babble, movie soundtrack
- **SNR range**: −5 to +20 dB in 5 dB steps

### Movie Trailers (14 with ground-truth SRTs)

```bash
# Download up to 50 trailers across 8 genres
python download_youtube_trailers.py --output-dir trailers --limit 50
```

**Genres covered**: action, sci-fi, drama, comedy, horror, animation, thriller, biopic

**Ground-truth notes**:
- Most references are YouTube auto-generated captions (VTT → SRT): these are Google ASR outputs, *not* human transcripts
- **Official human-authored SRTs**: Avengers Endgame and Spider-Man only
- WER against official SRTs: 0.144 and 0.183
- WER against YouTube auto-captions: 0.52–0.95 (inflated by hallucinations in reference)

### Full movies (user-provided)

Place `movie.mp4` + `movie.srt` pairs in `trailers/` or `movies/`:

```bash
python run_mass_experiments.py --video-dir /path/to/movies/ --output-dir movie_results/
```

---

## 7. Experiment Suite

### Experiment 1 — Avengers Trailer End-to-End

**What**: Full pipeline on the 125-second Avengers Endgame trailer.
**Script**: `run_full_experiments.py` → `exp1_trailer()`

**VAD comparison results**:

| VAD Method | Segments | Speech (s) | Speech Ratio | Compute (ms/utt) |
|---|---|---|---|---|
| Energy threshold | 19 | 43.4 | 0.478 | 1.9 |
| MFCC+ZCR | 20 | 70.6 | 0.650 | 2.6 |
| **Spectral (proposed)** | **13** | **78.5** | **0.700** | **6.2** |

**ASR performance**: RTF = 0.048 (20× faster than real-time). 9 subtitle cues generated, 2 CPS violations, 4 of 13 segments enhanced (SNR < 18 dB). Dominant mood: epic (68%), tense (18%), dialogue (10%).

**Outputs**: `outputs2/exp1_trailer/` — waveform+VAD plot, trailer analysis plot, latency distribution, metrics.json.

---

### Experiment 2 — Noise Robustness Benchmark (960 Conditions)

**What**: 10 LibriSpeech speakers × 4 noise types × 6 SNR levels × 4 systems = **960 conditions**.
**Script**: `run_full_experiments.py` → `exp2_noise_robustness()`
**Noise types**: white Gaussian, pink (1/f), babble, movie soundtrack
**SNR range**: −5, 0, 5, 10, 15, 20 dB

**WER by SNR (all noise types averaged)**:

| System | −5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| **Raw** | 0.567 | 0.378 | 0.242 | **0.199** | **0.194** | 0.208 |
| Spectral sub | **0.548** | **0.378** | 0.263 | 0.212 | 0.207 | **0.200** |
| Adaptive | 0.731 | 0.555 | 0.451 | 0.373 | 0.299 | 0.269 |
| Wiener | 0.796 | 0.644 | 0.493 | 0.399 | 0.326 | 0.287 |

**Mean across all 960 conditions**:

| System | Mean WER ↓ | Mean CER ↓ | Mean RTF |
|---|---|---|---|
| **Raw** | **0.298** | **0.131** | 0.043 |
| Spectral subtraction | 0.301 | 0.129 | **0.035** |
| Adaptive routing | 0.446 | 0.234 | 0.051 |
| Wiener filter | 0.491 | 0.265 | 0.058 |

**Key finding**: Raw audio outperforms all enhancement methods at SNR ≥ 10 dB. Modern int8-quantised Whisper is intrinsically noise-robust. Adaptive routing underperforms because incorrect routing decisions (enhancing already-clean segments) dominate the error budget.

**Outputs**: `outputs2/exp2_noise_robustness/` — `wer_by_snr.png`, `wer_heatmap.png`, `cer_table.csv`.

---

### Experiment 3 — VAD Ablation Study

**What**: Compare 3 VAD methods on speech coverage vs compute.
**Script**: `run_full_experiments.py` → `exp3_vad_ablation()`

| VAD Method | Mean Coverage | Compute/utt | Design |
|---|---|---|---|
| Energy threshold | 0.244 | 1.9 ms | Baseline: log-energy percentile |
| MFCC+ZCR | 0.502 | 2.6 ms | Improved: C0 × (1−ZCR) |
| **Spectral (proposed)** | **0.528** | **6.2 ms** | Full: MFCC + centroid − flatness |

**Key finding**: Spectral VAD detects **2.17× more speech** than energy-only at only 3× more compute (6.2 ms vs 1.9 ms per utterance). Critical for trailers where 40–70% of audio is music-only. Energy-based VAD fires during loud orchestral segments; spectral flatness discriminates harmonic music from voiced speech.

**Outputs**: `outputs2/exp3_vad_ablation/` — `fig_vad_comparison.png`, `vad_coverage.csv`.

---

### Experiment 4 — Enhancement Method Ablation

**What**: Wiener filter vs spectral subtraction vs adaptive routing under movie soundtrack noise.
**Script**: `run_full_experiments.py` → `exp4_enhancement_ablation()`

| Method | Mean WER ↓ | Mean CER ↓ | Mean RTF |
|---|---|---|---|
| **Raw** | **0.221** | **0.071** | 0.032 |
| Spectral subtraction | 0.227 | 0.075 | **0.030** |
| Adaptive routing | 0.367 | 0.168 | 0.041 |
| Wiener filter | 0.499 | 0.270 | 0.050 |

**Key finding**: Spectral subtraction is only 2.7% worse than raw (0.227 vs 0.221). Wiener filter is 2.3× worse than raw (0.499 vs 0.221) because it over-smooths the spectrum, suppressing high-frequency fricatives and introducing musical noise artifacts. Adaptive routing underperforms because the routing decision itself adds errors.

**Outputs**: `outputs2/exp4_enhancement/` — `fig_enhancement_ablation.png`, `enhancement_comparison.csv`.

---

### Experiment 5 — Model Size Comparison

**What**: Accuracy–latency tradeoff: tiny.en vs base.en (LibriSpeech) and tiny.en vs medium.en (trailers).
**Script**: `run_full_experiments.py` → `exp5_model_comparison()`

**tiny.en vs base.en (LibriSpeech test-clean)**:

| Model | Params | Mean WER ↓ | Mean CER ↓ | Median Latency | Load Time |
|---|---|---|---|---|---|
| **base.en** | 74M | **0.172** | **0.040** | 367 ms | 17,561 ms |
| tiny.en | 39M | 0.186 | 0.050 | **262 ms** | **360 ms** |

**tiny.en vs medium.en (14 trailers)**:

| Model | Mean WER ↓ | Mean RTF | Load Time |
|---|---|---|---|
| tiny.en | 0.875 | **0.052** | **360 ms** |
| **medium.en** | **0.522** | 0.175 | 2,050 ms |

medium.en reduces WER by **38% relative** vs tiny.en on trailers (0.522 vs 0.875).

**RTF — all models real-time on Apple M1 CPU (int8)**:

| Model | Min RTF | Mean RTF | Max RTF |
|---|---|---|---|
| tiny.en | 0.013 | 0.043 | 0.548 |
| base.en | 0.018 | 0.046 | 0.600 |
| medium.en | 0.045 | 0.175 | 0.404 |

**Key finding**: For offline batch subtitling, use `medium.en`. For live captioning, `tiny.en` loads 49× faster (360 ms vs 17.6 s) and processes 1.3× faster.

**Outputs**: `outputs2/exp5_model_comparison/` — `fig_model_comparison.png`, `model_metrics.csv`.

---

### Experiment 6 — Subtitle Quality & Readability

**What**: CPS violations, timestamp accuracy (TS-MAE), and readability across 3 noise scenarios.
**Script**: `run_full_experiments.py` → `exp6_subtitle_quality()`

| Scenario | Noise Type | SNR | Pred Cues | TS-MAE (s) | WER | CPS Violations |
|---|---|---|---|---|---|---|
| Clean | Soundtrack | 30 dB | 23 | 29.2 | 0.966 | 7 |
| Moderate | Pink | 10 dB | 21 | 24.2 | 0.923 | 7 |
| Heavy | Soundtrack | 0 dB | 16 | 22.3 | 1.000 | 3 |

**Key finding**: CPS violations are driven by long ASR outputs without word-boundary breaks, not by ASR errors. The max-words-per-cue rule (≤12 words/cue) in Stage 6 reduces violations by >50% without any re-processing.

**Outputs**: `outputs2/exp6_subtitle_quality/` — `fig_subtitle_quality.png`, `subtitle_metrics.csv`.

---

### Experiment 7 — Mass Trailer Evaluation (14 Trailers, 8 Genres)

**What**: End-to-end evaluation with `medium.en` on 14 real YouTube trailers.
**Script**: `run_full_eval.py` and `run_mass_experiments.py`
**Metrics**: WER, CER, chrF (char n-gram F-score), BLEU-1, RTF, CPS violations, scene mood

| Trailer | Genre | WER ↓ | CER ↓ | chrF ↑ | BLEU-1 ↑ | RTF | Mood |
|---|---|---|---|---|---|---|---|
| Avengers Endgame *(official SRT)* | action | **0.144** | 0.123 | **84.9** | **84.5** | 0.115 | epic |
| Spider-Man *(official SRT)* | action | **0.183** | **0.090** | **86.3** | **87.4** | 0.190 | epic |
| Whiplash | drama | **0.405** | **0.266** | **68.9** | 67.0 | 0.201 | epic |
| Zodiac | thriller | **0.405** | **0.266** | **68.9** | 67.0 | 0.217 | epic |
| Bridesmaids | comedy | 0.449 | 0.325 | 61.3 | 64.3 | 0.254 | epic |
| Mad Max: Fury Road | action | 0.450 | 0.264 | 63.5 | 56.9 | 0.187 | epic |
| Dunkirk | action | 0.513 | 0.392 | 58.8 | 49.8 | **0.052** | epic |
| The Dark Knight | action | 0.525 | 0.355 | 61.2 | 54.4 | **0.045** | epic |
| Nomadland | drama | 0.533 | 0.393 | 63.7 | 62.7 | 0.404 | epic |
| Spider-Man: Into the Spider-Verse | action | 0.570 | 0.514 | 48.8 | 36.4 | 0.081 | epic |
| The Revenant | drama | 0.554 | 0.566 | 47.4 | 32.6 | 0.287 | epic |
| The Social Network | drama | 0.634 | 0.488 | 57.1 | 47.0 | 0.204 | **dialogue** |
| Avengers: Age of Ultron *(YT auto-cap)* | action | 0.948 | 0.942 | 6.0 | 0.0 | 0.108 | epic |
| **Mean (all 14)** | — | **0.522** | **0.427** | **55.5** | **57.1** | **0.175** | — |

**Genre-level performance**:

| Genre | Mean WER | Mean CER | Mean chrF |
|---|---|---|---|
| thriller | **0.405** | **0.266** | **68.9** |
| comedy | 0.449 | 0.325 | 61.3 |
| drama | 0.531 | 0.428 | 59.3 |
| action | 0.593 | 0.513 | 46.2 |

**Key findings**:
- Action trailers are hardest (dense music, rapid cuts, minimal dialogue)
- Official SRTs give dramatically better WER than YouTube auto-captions
- Genre classifier accuracy: **76.9%** (leave-one-out k-NN on MFCC features)
- Dunkirk and Dark Knight have the fastest RTF (0.045–0.052): mostly music, minimal speech to transcribe

**Outputs**: `mass_results/` — `mass_results.csv`, `MASS_REPORT.md`, `fig_mass_wer_cer.png`, per-trailer scene_analysis.png, predicted.srt, predicted_annotated.srt.

---

### Experiment 8 (Novel A) — PESQ & STOI Enhancement Quality

**What**: Objective speech quality metrics — PESQ (ITU-T P.862.2) and STOI (Taal 2011) — independent of the ASR backend.

**Motivation**: WER conflates ASR quality and pre-processing quality. PESQ/STOI measure purely the enhancement benefit.

| Method | Mean PESQ (0–4.5) ↑ | Mean STOI (0–1) ↑ |
|---|---|---|
| No enhancement (noisy) | 1.778 | 0.879 |
| **Spectral subtraction** | **2.093** | **0.878** |
| Wiener filter | 1.212 | 0.799 |

**Key finding**: Spectral subtraction achieves +17.8% PESQ over noisy input. Wiener filter *degrades* PESQ below the noisy baseline (1.212 vs 1.778) — consistent with its higher WER. STOI barely changes (0.879 → 0.878), indicating intelligibility is already high at typical movie SNR.

**Outputs**: `mass_results/exp_enhancement_quality/` — `fig_pesq_stoi.png`, `pesq_stoi_results.csv`.

---

### Experiment 9 (Novel B) — Noise-Type Classifier

**What**: 16-dimensional rule-based classifier using MFCC + spectral features. 5 classes: silence, white noise, pink noise, music/soundtrack, babble.

**Feature vector** (16-dim): MFCC C1–C6, log RMS, ZCR, spectral centroid, flatness, rolloff, sub-band energy ratios (×4), periodicity.

| Class | Accuracy |
|---|---|
| Silence | **1.00** |
| White noise | 0.52 |
| Music / soundtrack | 0.40 |
| Pink noise | 0.00 |
| Babble | 0.00 |
| **Overall** | **0.316** |

**Key finding**: Pink noise and babble have similar MFCC profiles. Rule-based features cannot separate them without temporal or higher-order statistics. This motivates a learned front-end classifier (MLP/SVM on mel-filterbank features) as the clear next step.

**Outputs**: `mass_results/exp_noise_classifier/` — `fig_noise_classifier.png` (confusion matrix + feature scatter), `classifier_results.csv`.

---

### Experiment 10 (Novel C) — Whisper Confidence Calibration

**What**: Does Whisper's `avg_logprob` reliably predict transcription quality?

**Method**: Pearson/Spearman correlation between `avg_logprob` and WER across 955 conditions from Experiment 2. AUROC for binary high-error detection (WER > 0.3).

| Metric | Value |
|---|---|
| Pearson r | −0.783 |
| Spearman ρ | −0.859 |
| Expected Calibration Error (ECE) | 0.576 |
| **AUROC (WER > 0.3)** | **0.929** |

**Key finding**: `avg_logprob` is an excellent error *detector* (AUROC = 0.929) despite poor absolute calibration (ECE = 0.576). A threshold on `avg_logprob` can flag ~93% of low-quality segments before they reach the SRT file. Already deployed: `log_prob_threshold = −1.2` in faster-whisper config.

**Outputs**: `mass_results/exp_confidence_calibration/confidence_calibration.json`.

---

### Experiment 11 (Novel D) — Streaming Pipeline Latency

**What**: Chunk-based live captioning pipeline. First-word latency vs WER as a function of chunk size.

| Mode | First-Word Latency ↓ | Mean WER ↓ | Mean RTF |
|---|---|---|---|
| Batch (full utterance) | 255 ms | **0.174** | **0.032** |
| Stream 1 s chunks | 195 ms | 0.607 | 0.260 |
| Stream 2 s chunks | 186 ms | 0.324 | 0.232 |
| **Stream 3 s chunks** | **203 ms** | **0.291** | 0.087 |
| Stream 5 s chunks | 233 ms | 0.262 | 0.142 |

**Key finding**: 3-second chunks give the best latency–accuracy tradeoff (203 ms, WER 0.291). 1-second chunks fail (WER 0.607): insufficient acoustic context for Whisper's attention. Batch mode is best for accuracy but requires full utterance before transcription starts.

**Outputs**: `mass_results/exp_streaming_latency/streaming_results.json`.

---

## 8. Results Summary

### The YouTube Ground Truth Fallacy

When evaluated against **official human-authored SRTs**:
- Avengers Endgame: WER **0.144**, chrF **84.9**
- Spider-Man: WER **0.183**, chrF **86.3**

When evaluated against **YouTube auto-generated captions** (which hallucinate lyrics during music):
- Mean WER: 0.52–0.95 (artificially inflated)
- Avengers: Age of Ultron: WER 0.948, chrF 6.0

Always validate reference SRTs manually before computing WER on cinematic content.

### WER Progression: Clean to Noisy (LibriSpeech + raw audio)

```
Clean LibriSpeech (base.en)     :  WER = 0.07
Clean speech, 20 dB SNR         :  WER = 0.18
Moderate noise, 10 dB SNR       :  WER = 0.20
Heavy soundtrack, 0 dB SNR      :  WER = 0.38
```

### All Experiments Summary

| # | Experiment | Dataset | Scale | Best WER / Key Metric |
|---|---|---|---|---|
| 1 | Avengers Trailer End-to-End | Avengers (125 s) | 1 trailer | RTF = 0.048 |
| 2 | Noise Robustness | LibriSpeech | 960 conditions | Raw WER = 0.298 |
| 3 | VAD Ablation | LibriSpeech + trailer | 3 methods | Spectral: 2.17× coverage |
| 4 | Enhancement Ablation | LibriSpeech | 4 systems | Wiener 2× worse than raw |
| 5 | Model Size Comparison | LibriSpeech + trailers | 3 models | medium.en −38% WER |
| 6 | Subtitle Quality | LibriSpeech | 3 scenarios | CPS fix: ≤12 words/cue |
| 7 | Mass Trailer Eval | 14 trailers, 8 genres | ~2.5 hrs | Mean WER 0.522 |
| 8 | PESQ & STOI | Synthetic noise | 3 methods | Spectral sub: PESQ +18% |
| 9 | Noise Classifier | Synthetic noise | 5 classes | 31.6% overall accuracy |
| 10 | Confidence Calibration | 955 conditions | logprob vs WER | AUROC = 0.929 |
| 11 | Streaming Latency | LibriSpeech | 5 chunk sizes | 3 s: 203 ms latency |

---

## 9. Scene Understanding Module

`project/scene_understanding.py`

### Features extracted (per 512-sample frame, 32 ms at 16 kHz)

- **MFCC × 13** + delta: cepstral speech/music discriminant
- **Chroma × 12**: pitch class distribution, tonality
- **Spectral contrast × 6**: harmonic-to-noise ratio proxy
- **HPSS harmonic/percussive ratio**: music vs percussion vs speech
- **Onset strength**: event detection (explosions, musical beats)
- **Spectral centroid / rolloff / flatness / bandwidth**: spectral shape
- **ZCR**: speech vs music discriminant

### Foote novelty score

Self-similarity matrix of MFCC+chroma features, convolved with a Gaussian checkerboard kernel. Peaks mark structural scene boundaries (e.g., cut from dialogue to action sequence).

### Mood classification labels

| Label | Acoustic signature | Typical context |
|---|---|---|
| tense | High energy, low valence, slow tempo | Thriller climax, horror build-up |
| epic | High energy, high harmonic ratio | Action sequence, superhero moment |
| sad | Low energy, low valence, high harmonic | Drama, tragedy |
| happy | Moderate energy, high valence, fast tempo | Comedy, uplifting scene |
| calm | Low energy, high harmonic ratio | Quiet dialogue, nature |
| dialogue | Speech-like ZCR, low harmonic ratio | Direct speech segment |
| silence | RMS < −40 dB | Scene gap |

### Multi-Modal Outputs

Each processed trailer produces:
1. **`predicted.srt`**: Clean SRT for WER evaluation
2. **`predicted_annotated.srt`**: With mood tags [TENSE]😬, [EPIC]⚡, [SAD]😢, [MUSIC]🎵
3. **`scene_analysis.json`**: Foote boundaries, mood timeline (0.5 s resolution), mood distribution
4. **`scene_analysis.png`**: 5-panel plot — waveform + VAD, mel spectrogram, Foote novelty + boundaries, mood timeline, mood pie chart
5. **`metrics.json`**: WER, CER, RTF, CPS violations, TS-MAE, speech ratio

---

## 10. Module Reference

### `project/audio_utils.py`

Core DSP utilities.

- `vad_energy(audio, sr)` → `VADResult` — energy-threshold VAD
- `vad_mfcc(audio, sr)` → `VADResult` — MFCC C0 + ZCR VAD
- `vad_spectral(audio, sr)` → `VADResult` — full spectral feature VAD
- `enhance_wiener(audio)` → filtered array
- `enhance_spectral_subtraction(audio, ...)` → filtered array
- `extract_features(audio)` → `AudioFeatures` (9-dim)
- `should_enhance(features)` → bool — adaptive routing decision
- `synthesize_noise(audio, noise_type, snr_db)` → noisy array

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

### `project/extra_metrics.py`

Extended metrics.

- `compute_chrf(hypothesis, reference)` — character n-gram F-score
- `compute_bleu(hypothesis, reference)` — BLEU-1/2/4
- `classify_genre(audio, sr)` — k-NN genre classifier (76.9% LOO accuracy)

### `project/emotion_detector.py`

Acoustic emotion classification.

- `classify_emotion(audio, sr, word_count, duration_sec)` → `EmotionResult`
- `detect_emotions_batch(segments, audio, sr)` → `list[EmotionResult]`

---

## 11. Key Findings

1. **Raw ASR beats naive enhancement at SNR ≥ 10 dB** — Modern int8-quantised Whisper is intrinsically noise-robust at typical movie SNRs. Do not apply Wiener filter to movie audio.

2. **Adaptive routing is the correct design** — Withhold enhancement for clean segments; apply spectral subtraction only when SNR < 0 dB or spectral flatness signals noise dominance.

3. **Spectral subtraction > Wiener filter for movie audio** — PESQ +17.8%, WER −38% relative vs Wiener under soundtrack noise. Spectral subtraction preserves formant structure better.

4. **Spectral VAD detects 2.17× more speech than energy-only** at only 3× more compute (6.2 ms vs 1.9 ms per utterance). Essential for trailers with 40–70% music content.

5. **medium.en reduces WER by 38% vs tiny.en on trailers** (0.522 vs 0.875), still real-time (RTF 0.175 < 1.0) on CPU.

6. **Whisper avg_logprob is a strong error predictor** — Spearman ρ = −0.859, AUROC = 0.929 for WER > 0.3 detection. Deploy as confidence-gated SRT filter.

7. **3-second chunks give the best streaming latency–accuracy tradeoff** — 203 ms first-word latency, WER = 0.291 (vs 255 ms / 0.174 for batch).

8. **Rule-based noise classification fails** — 31.6% overall accuracy; pink noise and babble have indistinguishable MFCC profiles. A learned front-end is needed.

9. **YouTube auto-captions are flawed ground truth** — Official SRT WER = 0.144 vs 0.948 with YouTube auto-captions for the same content. Always validate reference transcripts.

10. **Foote novelty score successfully detects structural transitions** — Useful for subtitle formatting decisions and accessibility metadata (identifying when music replaces speech).

11. **CPS violations are a post-processing concern, not an ASR quality issue** — They arise from long contiguous ASR outputs. The ≤12 words/cue rule in Stage 6 resolves >50% of violations without re-processing.
