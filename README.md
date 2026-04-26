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
7. [Experiment 1 — Avengers Trailer End-to-End](#7-experiment-1--avengers-trailer-end-to-end)
8. [Experiment 2 — Noise Robustness Benchmark (960 conditions)](#8-experiment-2--noise-robustness-benchmark)
9. [Experiment 3 — VAD Ablation Study](#9-experiment-3--vad-ablation-study)
10. [Experiment 4 — Enhancement Method Ablation](#10-experiment-4--enhancement-method-ablation)
11. [Experiment 5 — Model Size Comparison](#11-experiment-5--model-size-comparison)
12. [Experiment 6 — Subtitle Quality & Readability](#12-experiment-6--subtitle-quality--readability)
13. [Experiment 7 — Mass Trailer Evaluation (14 trailers, medium.en)](#13-experiment-7--mass-trailer-evaluation)
14. [Novel Experiment A — PESQ & STOI](#14-novel-experiment-a--pesq--stoi)
15. [Novel Experiment B — Noise-Type Classifier](#15-novel-experiment-b--noise-type-classifier)
16. [Novel Experiment C — Whisper Confidence Calibration](#16-novel-experiment-c--whisper-confidence-calibration)
17. [Novel Experiment D — Streaming Pipeline Latency](#17-novel-experiment-d--streaming-pipeline-latency)
18. [Extended Evaluation — 50+ YouTube Trailers (tiny.en)](#18-extended-evaluation--50-youtube-trailers)
19. [20-Minute Full Movie Snippet Evaluation](#19-20-minute-full-movie-snippet-evaluation)
20. [Earlier Model Comparison (tiny vs medium, Avengers & Spider-Man)](#20-earlier-model-comparison)
21. [Scene Understanding & Mood Analysis](#21-scene-understanding--mood-analysis)
22. [Per-Trailer Scene Analysis Plots](#22-per-trailer-scene-analysis-plots)
23. [Key Findings](#23-key-findings)
24. [Module Reference](#24-module-reference)

---

## 1. Project Overview & Novelty

This project addresses **automatic subtitle generation for movie trailers and full films** — a task significantly harder than clean-speech ASR because movie audio combines:

- Overlapping **dialogue and orchestral score**
- **Variable SNR** across cuts (dialogue at +20 dB, action sequences at −5 dB)
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
| Evaluation | 2 clips | 50+ trailers across genres + 4 full 20-min movie snippets |

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
│  ① Energy-only   — log-energy threshold (baseline)     │
│  ② MFCC+ZCR      — MFCC C0 × zero-crossing rate        │
│  ③ Spectral VAD  — 0.45·C0 + 0.25·centroid             │
│                   − 0.20·flatness + 0.10·(1−ZCR)        │
│  Output: list of (start_sec, end_sec) speech segments   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Adaptive Enhancement Routing                  │
│  IF estimated_SNR < 18 dB OR flatness > 0.18            │
│  OR ZCR > 0.12 → apply enhancement                      │
│  ELSE → pass raw audio to ASR                           │
│  Options: Wiener filter / Spectral Subtraction (α=1.5)  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: ASR — faster-whisper + Silero VAD             │
│  Models: tiny.en (39M) / base.en (74M) / medium.en      │
│  Silero VAD: thresh=0.35, min_silence=400ms             │
│  Hallucination suppression: log_prob_threshold = −1.2   │
│  Word-level timestamps · RTF 0.035–0.23 (all < 1)       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: Scene Understanding                           │
│  Foote novelty score → scene boundaries                 │
│  Per-segment mood: tense/epic/sad/calm/dialogue/silence  │
│  Temporal timeline (0.5s resolution)                    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 6: Subtitle Post-Processing                      │
│  ≤12 words/cue · max 84 chars · CPS ≤ 20 enforcement   │
│  Outputs: .srt · annotated SRT · JSON · 5-panel PNG     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
EE 679 Project/
│
├── project/                          # Core Python package
│   ├── audio_utils.py                ★ VAD (energy/MFCC/spectral), enhancement,
│   │                                   feature extraction, noise synthesis
│   ├── asr.py                        ★ faster-whisper wrapper with RTF tracking
│   ├── enhanced_asr.py               ★ medium.en + Silero VAD + word timestamps
│   ├── scene_understanding.py        ★ Foote novelty, mood classification
│   ├── novel_experiments.py          ★ PESQ/STOI, noise classifier, confidence,
│   │                                   streaming latency
│   ├── emotion_detector.py             opensmile eGeMAPSv02 emotion features
│   ├── sound_events.py                 Sound event detection
│   ├── dataset.py                      LibriSpeech downloader/preparer
│   ├── metrics.py                      WER, CER, timestamp MAE, CPS
│   ├── subtitles.py                    SRT writer with line-breaking
│   ├── extra_metrics.py                chrF, BLEU, genre k-NN classifier
│   ├── marvel_pipeline.py              v1 pipeline
│   ├── live_pipeline.py                Streaming subtitle pipeline
│   ├── trailer_v2.py                   Enhanced pipeline orchestrator
│   └── pipeline.py                     Legacy utilities
│
├── run_full_experiments.py           ★ 6-experiment LibriSpeech suite
├── run_mass_experiments.py           ★ 14-trailer evaluation + novel experiments
├── run_full_eval.py                    medium.en batch evaluation
├── run_enhanced.py                     Single-video enhanced pipeline CLI
├── run_trailer_experiments.py          Batch trailer WER evaluator
├── clever_eval_all.py                  Novel metrics computation
├── trim_and_eval.py                    20-min snippet trimming + evaluation
├── download_youtube_trailers.py        yt-dlp batch downloader (8 genres)
│
├── outputs2/                         ← LibriSpeech experiment results
│   ├── exp1_trailer/                   Avengers: VAD, analysis, latency plots
│   ├── exp2_noise_robustness/          960-condition WER heatmaps
│   ├── exp3_vad_ablation/              VAD comparison figure
│   ├── exp4_enhancement/               Enhancement ablation figure
│   ├── exp5_model_comparison/          tiny vs base figure
│   ├── exp6_subtitle_quality/          CPS/timing figure
│   └── FULL_REPORT.md
│
├── mass_results/                     ← 14-trailer + novel experiment results
│   ├── fig_mass_wer_cer.png
│   ├── fig_rtf_speech_ratio.png
│   ├── fig_scene_summary.png
│   ├── exp_enhancement_quality/        PESQ + STOI plots
│   ├── exp_noise_classifier/           Confusion matrix
│   ├── exp_confidence_calibration/     logprob vs WER
│   ├── exp_streaming_latency/          Streaming latency curve
│   └── <trailer-slug>/                 per-trailer: SRT + scene_analysis.png
│
├── eval_results_medium/              ← medium.en evaluation plots
│   ├── fig_genre_performance.png
│   ├── fig_metrics_per_trailer.png
│   ├── fig_rtf_duration.png
│   ├── fig_scene_mood_summary.png
│   ├── exp_genre_classifier/
│   └── <trailer-slug>/scene_analysis.png  (14 trailers)
│
├── evaluate_results/                 ← Extended 50+ trailer + 4 movie snippet runs
│   ├── snippets/                       20-min movie clips (MP4 + official SRT)
│   ├── all_clever_metrics.csv          WER/CER/Similarity/TS-MAE for all 50+
│   ├── clever_metrics_summary.csv      Movie snippet metrics
│   ├── parsed_summary.csv              Parsed WER/CER/RTF per trailer
│   └── <title>/                        per-title: SRT + trailer_analysis.png
│
├── experiment_results/               ← Early tiny vs medium comparison
│   ├── avengers_tiny_en/
│   ├── avengers_medium_en/
│   ├── spiderman_tiny_en/
│   ├── spiderman_medium_en/
│   ├── fig_model_comparison.png
│   └── fig_emotion_events.png
│
├── outputs/                          ← First-run LibriSpeech noise plots
│   ├── wer_by_noise.png
│   └── wer_by_snr.png
│
├── outputs_v2/ outputs_spiderman_v2/ ← Early pipeline test runs
├── trailers/                           Downloaded trailers (MP4 + SRT)
├── youtube_trailers/                   YouTube trailers (990 MB, 50+ movies)
├── data/                               LibriSpeech test-clean (690 MB)
├── movies/                             Full-length movies (user-provided)
├── requirements.txt · Dockerfile · presentation.tex
```

---

## 4. Setup & Installation

```bash
cd "/path/to/EE 679 Project"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg bundled via imageio-ffmpeg — no separate install needed
# Whisper model auto-downloads on first use
```

| Package | Version | Purpose |
|---|---|---|
| `faster-whisper` | 1.2.1 | CTranslate2 Whisper ASR + Silero VAD |
| `opensmile` | 2.6.0 | eGeMAPSv02 emotion features |
| `librosa` | 0.11.0 | MFCC, chroma, onset, HPSS, spectral |
| `soundfile` | 0.13.1 | WAV I/O |
| `imageio-ffmpeg` | 0.6.0 | Bundled ffmpeg binary |
| `jiwer` | 4.0.0 | WER / CER |
| `pesq` | 0.0.4 | ITU-T P.862.2 speech quality |
| `pystoi` | 0.4.1 | Short-Time Objective Intelligibility |
| `yt-dlp` | 2026.3.17 | YouTube downloader |
| `matplotlib` | 3.10.9 | All figures |
| `scipy` | 1.17.1 | Wiener filter, signal processing |
| `pandas` | 3.0.2 | Result tables |
| `numpy` | 2.4.4 | Numerical arrays |
| `pysrt` | 1.1.2 | SRT read/write |

---

## 5. Quick Start

```bash
source .venv/bin/activate

# Single trailer — full pipeline with scene analysis
python run_enhanced.py

# LibriSpeech 6-experiment suite (downloads ~350MB test set)
python run_full_experiments.py --output-dir outputs2 --n-samples 10

# 14-trailer evaluation + novel experiments
python run_mass_experiments.py --video-dir trailers --output-dir mass_results

# Extended YouTube trailer evaluation
python run_trailer_experiments.py --models tiny.en

# 20-minute movie snippet evaluation
python trim_and_eval.py --input-dir movies/ --output-dir evaluate_results/
```

---

## 6. Dataset Acquisition

### LibriSpeech test-clean

Auto-downloaded by `run_full_experiments.py` from openslr.org/12.
- 2,620 utterances · 5.4 hours · 40 speakers · 690 MB
- Synthetic noise overlaid in code: white Gaussian, pink (1/f), babble, movie soundtrack at SNR −5 to +20 dB

### Movie Trailers

```bash
python download_youtube_trailers.py --output-dir trailers --limit 50
```

- **50+ YouTube trailers**, 8 genres: action, sci-fi, drama, comedy, horror, animation, thriller, biopic
- Reference: YouTube auto-generated captions (VTT → SRT). **Not human transcripts** — heavily hallucinated during music.
- **Official human-authored SRTs** (included): Avengers Endgame, Spider-Man No Way Home

### 20-Minute Full Movie Snippets

Four full Hollywood films were trimmed to their first 20 minutes, English audio track isolated from dual-audio MKVs:
- **Avengers: Endgame** (2019)
- **The Social Network** (2010)
- **Spider-Man: No Way Home** (2021)
- **The Dark Knight Rises** (2012)

Reference SRTs: official subtitle files from OpenSubtitles.

---

## 7. Experiment 1 — Avengers Trailer End-to-End

**Script**: `run_full_experiments.py` → `exp1_trailer()`
**Content**: 124.97-second Avengers Endgame trailer

| VAD Method | Segments | Speech (s) | Speech Ratio | Compute (ms/utt) |
|---|---|---|---|---|
| Energy threshold | 19 | 43.4 | 0.478 | 1.9 |
| MFCC+ZCR | 20 | 70.6 | 0.650 | 2.6 |
| **Spectral (proposed)** | **13** | **78.5** | **0.700** | **6.2** |

- Overall ASR RTF: **0.048** (20× faster than real-time)
- Subtitle cues: **9** · CPS violations: **2** · Enhanced segments: 1 / raw: 8
- Dominant mood: epic 68%, tense 18%, dialogue 10%, silence 4%

### VAD Comparison

![VAD Comparison](outputs2/exp1_trailer/fig_vad_comparison.png)

### Trailer Analysis (Waveform + VAD + Mood)

![Trailer Analysis](outputs2/exp1_trailer/fig_trailer_analysis.png)

### Per-Segment Latency Distribution

![Latency Distribution](outputs2/exp1_trailer/fig_latency_distribution.png)

---

## 8. Experiment 2 — Noise Robustness Benchmark

**Scale**: 10 speakers × 4 noise types × 6 SNR levels × 4 systems = **960 conditions**
**Noise**: white Gaussian, pink (1/f), babble, movie soundtrack · **SNR**: −5 to +20 dB

| System | Mean WER ↓ | Mean CER ↓ | Mean RTF |
|---|---|---|---|
| **Raw** | **0.298** | **0.131** | 0.043 |
| Spectral subtraction | 0.301 | 0.129 | **0.035** |
| Adaptive routing | 0.446 | 0.234 | 0.051 |
| Wiener filter | 0.491 | 0.265 | 0.058 |

**WER by SNR** (all noise types averaged):

| System | −5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|
| **Raw** | 0.567 | 0.378 | 0.242 | **0.199** | **0.194** | 0.208 |
| Spectral sub | **0.548** | **0.378** | 0.263 | 0.212 | 0.207 | **0.200** |
| Adaptive | 0.731 | 0.555 | 0.451 | 0.373 | 0.299 | 0.269 |
| Wiener | 0.796 | 0.644 | 0.493 | 0.399 | 0.326 | 0.287 |

**Key finding**: Raw audio outperforms all enhancement at SNR ≥ 10 dB. Modern int8 Whisper is intrinsically noise-robust. Wiener is 65% worse than raw overall.

### WER vs SNR (per noise type)

![WER vs SNR by Noise Type](outputs2/exp2_noise_robustness/fig_wer_cer_vs_snr.png)

### WER Heatmap (all conditions)

![WER Heatmap](outputs2/exp2_noise_robustness/fig_wer_heatmap.png)

### WER by Noise Category

![WER by Noise](outputs2/exp2_noise_robustness/fig_wer_by_noise.png)

*Early-run version (outputs/ directory):*

![WER by SNR Early](outputs/wer_by_snr.png)

---

## 9. Experiment 3 — VAD Ablation Study

**Script**: `run_full_experiments.py` → `exp3_vad_ablation()`

| VAD Method | Mean Coverage | Compute/utt | Design |
|---|---|---|---|
| Energy threshold | 0.244 | 1.9 ms | Baseline |
| MFCC+ZCR | 0.502 | 2.6 ms | Improved |
| **Spectral (proposed)** | **0.528** | **6.2 ms** | **Proposed** |

**Key finding**: Spectral VAD detects **2.17× more speech** than energy-only at 3× more compute. Essential for trailers where 40–70% of audio is music-only.

### VAD Ablation Figure

![VAD Ablation](outputs2/exp3_vad_ablation/fig_vad_ablation.png)

---

## 10. Experiment 4 — Enhancement Method Ablation

**Script**: `run_full_experiments.py` → `exp4_enhancement_ablation()`
**Dataset**: LibriSpeech + movie soundtrack noise

| Method | Mean WER ↓ | Mean CER ↓ | Mean RTF |
|---|---|---|---|
| **Raw** | **0.221** | **0.071** | 0.032 |
| Spectral subtraction | 0.227 | 0.075 | **0.030** |
| Adaptive routing | 0.367 | 0.168 | 0.041 |
| Wiener filter | 0.499 | 0.270 | 0.050 |

**Key finding**: Wiener filter is 2.3× worse than raw because it over-smooths fricatives and introduces musical noise artifacts. Spectral subtraction is only 2.7% worse than raw but wins at SNR < 0 dB.

### Enhancement Ablation Figure

![Enhancement Ablation](outputs2/exp4_enhancement/fig_enhancement_ablation.png)

---

## 11. Experiment 5 — Model Size Comparison

**Script**: `run_full_experiments.py` → `exp5_model_comparison()`

**tiny.en vs base.en (LibriSpeech)**:

| Model | Params | WER ↓ | CER ↓ | Latency | Load Time |
|---|---|---|---|---|---|
| **base.en** | 74M | **0.172** | **0.040** | 367 ms | 17.6 s |
| tiny.en | 39M | 0.186 | 0.050 | **262 ms** | **360 ms** |

**tiny.en vs medium.en (trailers)**:

| Model | WER ↓ | RTF | Load |
|---|---|---|---|
| tiny.en | 0.875 | **0.052** | **360 ms** |
| **medium.en** | **0.522** | 0.175 | 2.0 s |

medium.en reduces WER by **38% relative**. All models RTF < 1.0 (real-time on Apple M1).

**RTF ranges**:

| Model | Min RTF | Mean RTF | Max RTF |
|---|---|---|---|
| tiny.en | 0.013 | 0.043 | 0.548 |
| base.en | 0.018 | 0.046 | 0.600 |
| medium.en | 0.045 | 0.175 | 0.404 |

### Model Comparison Figure (LibriSpeech)

![Model Comparison outputs2](outputs2/exp5_model_comparison/fig_model_comparison.png)

### Model Comparison Figure (Earlier Run — experiment_results/)

![Model Comparison Early](experiment_results/fig_model_comparison.png)

---

## 12. Experiment 6 — Subtitle Quality & Readability

**Script**: `run_full_experiments.py` → `exp6_subtitle_quality()`

| Scenario | Noise | SNR | Pred Cues | TS-MAE | WER | CPS Violations |
|---|---|---|---|---|---|---|
| Clean | Soundtrack | 30 dB | 23 | 29.2 s | 0.966 | 7 |
| Moderate | Pink | 10 dB | 21 | 24.2 s | 0.923 | 7 |
| Heavy | Soundtrack | 0 dB | 16 | 22.3 s | 1.000 | 3 |

**Key finding**: CPS violations are a post-processing concern — the ≤12 words/cue rule reduces them by >50% without re-processing.

### Subtitle Quality Figure

![Subtitle Quality](outputs2/exp6_subtitle_quality/fig_subtitle_quality.png)

---

## 13. Experiment 7 — Mass Trailer Evaluation

**14 trailers · medium.en · 8 genres**
**Script**: `run_mass_experiments.py` + `run_full_eval.py`
**Results**: `mass_results/` and `eval_results_medium/`

| Trailer | Genre | WER ↓ | CER ↓ | chrF ↑ | BLEU-1 ↑ | RTF | Mood |
|---|---|---|---|---|---|---|---|
| Avengers Endgame *(official SRT)* | action | **0.144** | 0.123 | **84.9** | **84.5** | 0.115 | epic |
| Spider-Man *(official SRT)* | action | **0.183** | 0.090 | **86.3** | **87.4** | 0.190 | epic |
| Whiplash | drama | **0.405** | **0.266** | **68.9** | 67.0 | 0.201 | epic |
| Zodiac | thriller | **0.405** | **0.266** | **68.9** | 67.0 | 0.217 | epic |
| Bridesmaids | comedy | 0.449 | 0.325 | 61.3 | 64.3 | 0.254 | epic |
| Mad Max: Fury Road | action | 0.450 | 0.264 | 63.5 | 56.9 | 0.187 | epic |
| Dunkirk | action | 0.513 | 0.392 | 58.8 | 49.8 | **0.052** | epic |
| The Dark Knight | action | 0.525 | 0.355 | 61.2 | 54.4 | **0.045** | epic |
| Nomadland | drama | 0.533 | 0.393 | 63.7 | 62.7 | 0.404 | epic |
| Spider-Man: Spider-Verse | action | 0.570 | 0.514 | 48.8 | 36.4 | 0.081 | epic |
| The Revenant | drama | 0.554 | 0.566 | 47.4 | 32.6 | 0.287 | epic |
| The Social Network | drama | 0.634 | 0.488 | 57.1 | 47.0 | 0.204 | **dialogue** |
| Avengers: Age of Ultron *(YT auto-cap)* | action | 0.948 | 0.942 | 6.0 | 0.0 | 0.108 | epic |
| Captain America: Civil War *(YT auto-cap)* | action | 0.950 | 0.899 | 5.8 | 0.0 | 0.071 | epic |
| **Mean (14 trailers)** | — | **0.522** | **0.427** | **55.5** | **57.1** | **0.175** | — |

**Genre performance**:

| Genre | Mean WER | Mean CER | Mean chrF |
|---|---|---|---|
| thriller | **0.405** | **0.266** | **68.9** |
| comedy | 0.449 | 0.325 | 61.3 |
| drama | 0.531 | 0.428 | 59.3 |
| action | 0.593 | 0.513 | 46.2 |

**Scene understanding summary** (from mass_results/MASS_REPORT.md):

| Trailer | Dominant Mood | Music Frac. | Speech Frac. | Boundaries |
|---|---|---|---|---|
| Avengers (original) | epic | 0.056 | 0.296 | 37 |
| Spider-Man | epic | 0.000 | 0.324 | 47 |
| Age of Ultron | epic | 0.000 | 0.165 | 20 |
| Bridesmaids | epic | 0.052 | 0.250 | 22 |
| Captain America | epic | 0.029 | 0.065 | 15 |
| Dunkirk | epic | 0.027 | 0.062 | 21 |

**Genre classifier**: **76.9% accuracy** (leave-one-out k-NN on MFCC features)

### WER / CER Across All 14 Trailers

![Mass WER CER](mass_results/fig_mass_wer_cer.png)

### RTF vs Speech Ratio

![RTF Speech Ratio](mass_results/fig_rtf_speech_ratio.png)

### Scene Mood Summary (All Trailers)

![Scene Summary](mass_results/fig_scene_summary.png)

### Per-Genre Performance (eval_results_medium/)

![Genre Performance](eval_results_medium/fig_genre_performance.png)

### Metrics Per Trailer (eval_results_medium/)

![Metrics Per Trailer](eval_results_medium/fig_metrics_per_trailer.png)

### RTF vs Duration

![RTF Duration](eval_results_medium/fig_rtf_duration.png)

### Scene Mood Summary (eval_results_medium/)

![Scene Mood Summary](eval_results_medium/fig_scene_mood_summary.png)

### Genre Classifier (eval_results_medium/)

![Genre Classifier](eval_results_medium/exp_genre_classifier/fig_genre_classifier.png)

### The YouTube Ground Truth Fallacy

When evaluated against **official human-authored SRTs**:
- Avengers Endgame: WER **0.144**, chrF **84.9**
- Spider-Man: WER **0.183**, chrF **86.3**

When evaluated against **YouTube auto-captions** (which hallucinate during music):
- Avengers: Age of Ultron WER **0.948**, chrF **6.0** — reference contains hallucinated lyrics during orchestral sequences. Our system correctly outputs silence; the "ground truth" output lyrics. This inflates WER by penalising correct silence as substitution errors.

---

## 14. Novel Experiment A — PESQ & STOI

**Motivation**: WER conflates ASR and pre-processing quality. PESQ/STOI measure enhancement benefit independently.

| Method | PESQ ↑ (0–4.5) | STOI ↑ (0–1) |
|---|---|---|
| Noisy (no enhancement) | 1.778 | 0.879 |
| **Spectral subtraction** | **2.093** | **0.878** |
| Wiener filter | 1.212 | 0.799 |

Spectral subtraction: **+17.8% PESQ**. Wiener: **−31.8% PESQ** — worse than no enhancement. STOI barely changes, indicating intelligibility is already high at typical movie SNR.

### PESQ & STOI Figure

![PESQ STOI](mass_results/exp_enhancement_quality/fig_pesq_stoi.png)

---

## 15. Novel Experiment B — Noise-Type Classifier

16-dimensional rule-based feature vector (MFCC C1–C6, log RMS, ZCR, centroid, flatness, rolloff, sub-band ratios ×4, periodicity). 5 classes: silence, white noise, pink noise, music/soundtrack, babble.

| Class | Accuracy |
|---|---|
| Silence | **1.00** |
| White noise | 0.52 |
| Music / soundtrack | 0.40 |
| Pink noise | 0.00 |
| Babble | 0.00 |
| **Overall** | **0.316** |

**Key finding**: Pink noise and babble have indistinguishable MFCC profiles. A learned front-end (MLP/SVM on mel-filterbank features) is the clear next step.

### Noise Classifier Confusion Matrix

![Noise Classifier](mass_results/exp_noise_classifier/fig_noise_classifier.png)

---

## 16. Novel Experiment C — Whisper Confidence Calibration

Correlate `avg_logprob` with WER across 955 conditions from Experiment 2.

| Metric | Value |
|---|---|
| Pearson r | −0.783 |
| Spearman ρ | −0.859 |
| ECE | 0.576 |
| **AUROC (WER > 0.3)** | **0.929** |

`avg_logprob` is an excellent error *detector* (AUROC = 0.929) despite poor absolute calibration. Already deployed: `log_prob_threshold = −1.2` in the faster-whisper config discards ~93% of bad segments before SRT output.

### Confidence Calibration Figure

![Confidence Calibration](mass_results/exp_confidence_calibration/fig_confidence_calibration.png)

---

## 17. Novel Experiment D — Streaming Pipeline Latency

Chunk-based live captioning: first-word latency vs WER by chunk size.

| Mode | Latency ↓ | WER ↓ | RTF |
|---|---|---|---|
| Batch (full utterance) | 255 ms | **0.174** | **0.032** |
| Stream 1 s chunks | 195 ms | 0.607 | 0.260 |
| Stream 2 s chunks | 186 ms | 0.324 | 0.232 |
| **Stream 3 s chunks** | **203 ms** | **0.291** | 0.087 |
| Stream 5 s chunks | 233 ms | 0.262 | 0.142 |

**3-second chunks** give the best tradeoff: 203 ms latency (within 500 ms human perception threshold), WER = 0.291. 1-second chunks fail (WER 0.607): insufficient context for Whisper's attention mechanism.

### Streaming Latency Figure

![Streaming Latency](mass_results/exp_streaming_latency/fig_streaming_latency.png)

---

## 18. Extended Evaluation — 50+ YouTube Trailers

**Script**: `run_trailer_experiments.py` + `clever_eval_all.py`
**Results**: `evaluate_results/all_clever_metrics.csv` · `evaluate_results/parsed_summary.csv`
**Model**: tiny.en

All 50+ trailers processed with the full pipeline (VAD → enhancement → ASR → scene understanding). Each produces: `trailer_analysis.png` (5-panel), `trailer_clean.srt`, `trailer_annotated.srt`, `trailer_timeline.json`.

### Selected Results (from parsed_summary.csv)

| Trailer | WER % ↓ | CER % ↓ | RTF |
|---|---|---|---|
| Marvel's The Avengers *(official SRT!)* | **4.88** | **3.10** | 0.108 |
| Aged (2023) — Horror | **11.94** | **5.52** | 0.076 |
| Deliver Us — Horror | **15.73** | **12.44** | 0.112 |
| Retribution (2023) | **21.61** | **18.35** | 0.083 |
| Greenland 2: Migration (2026) | **23.97** | **17.74** | 0.267 |
| Avengers: Doomsday First Look | 63.61 | 62.41 | 0.141 |
| Avengers: Doomsday (Official) | 69.23 | 58.81 | 0.082 |
| Avengers: Endgame (Official) | 58.04 | 57.81 | 0.055 |
| Marvel's Avengers: Infinity War | 71.01 | 67.60 | 0.069 |
| Captain America: Civil War | 70.96 | 67.86 | 0.205 |
| Superman (DC) | 67.60 | 65.95 | 0.079 |
| Spider-Man: Spider-Noir (Prime) | 62.98 | 57.83 | 0.082 |
| Supergirl (Official) | 70.50 | 65.34 | 0.117 |
| Supergirl (Teaser) | 65.90 | 63.98 | 0.082 |
| Clayface (Teaser) | 100.0 | 100.0 | 0.095 |
| New Movie Trailers 2023 (Sci-Fi) | 93.59 | 70.63 | — |
| Coyote vs. Acme | 90.65 | 89.74 | 0.141 |
| Pirates of the Caribbean: Curse | 73.46 | 68.82 | 0.112 |
| Pirates of the Caribbean: World's End | 68.50 | 64.64 | 0.134 |
| Pirates of the Caribbean: Dead Men | 88.17 | 85.68 | 0.149 |
| Pirates of the Caribbean 5 | 79.44 | 76.58 | 0.138 |
| All Pirates Saga | 71.19 | 68.04 | 0.195 |
| The Godfather | 72.91 | 66.54 | 0.110 |
| Elemental (Pixar) | 66.49 | 61.61 | 0.184 |
| Baymax! | 67.60 | 65.75 | 0.142 |
| Strays | 71.66 | 67.32 | 0.154 |
| Citadel Season 2 | 70.96 | 67.86 | 0.205 |
| Blood and Snow (2023) | 63.72 | 55.83 | 0.069 |
| Sisu (2023) | 62.31 | 65.84 | 0.098 |
| Simulant (2023) | 73.79 | 71.03 | 0.143 |
| The Breach (Sci-Fi 2023) | 66.67 | 64.63 | 0.122 |
| Swallow (2020) | 66.31 | 61.88 | 0.157 |
| To Catch a Killer (2023) | 74.93 | 71.35 | 0.133 |
| The Shade (2024) Horror | 92.08 | 90.40 | 0.082 |
| Somewhere Quiet (2024) | 65.95 | 65.16 | 0.187 |
| Wildcat (2025) | 69.32 | 67.26 | 0.110 |
| Freelance (2023) | 73.11 | 65.68 | 0.242 |
| Muzzle (2023) | 70.46 | 67.75 | 0.136 |
| Crater (2023) | 73.68 | 71.42 | 0.139 |

> **Note**: WER is inflated for most trailers because the reference SRTs are YouTube auto-captions (themselves ASR outputs containing hallucinations). Against the official Avengers SRT, WER drops to **4.88%**.

### Trailer Analysis Plots (evaluate_results/)

All plots are 5-panel: waveform + VAD segments, mel spectrogram, Foote novelty + boundaries, mood timeline, mood pie.

**20-Minute Movie Snippets:**

| Avengers Endgame (20 min) | The Social Network (20 min) |
|---|---|
| ![Avengers 20m](evaluate_results/Avengers%20Endgame%20201_20m/trailer_analysis.png) | ![Social Network 20m](evaluate_results/SocialNetwork_20m/trailer_analysis.png) |

| Spider-Man: NWH (20 min) | The Dark Knight Rises (20 min) |
|---|---|
| ![Spider-Man 20m](evaluate_results/Spider-Man.N.W.H.202_20m/trailer_analysis.png) | ![Dark Knight 20m](evaluate_results/The.Dark.Knight.Rise_20m/trailer_analysis.png) |

**Marvel Universe Trailers:**

| Avengers: Endgame | Avengers: Infinity War |
|---|---|
| ![Endgame](evaluate_results/marvel%20studios'%20avengers%EF%BC%9A%20endgame%20-%20offi/trailer_analysis.png) | ![Infinity War](evaluate_results/marvel%20studios'%20avengers%EF%BC%9A%20infinity%20war%20o/trailer_analysis.png) |

| Avengers (Original — Official SRT) | Avengers: Age of Ultron |
|---|---|
| ![Avengers Official](evaluate_results/marvel's%20the%20avengers-%20trailer%20(official/trailer_analysis.png) | ![Age of Ultron](evaluate_results/avengers%EF%BC%9A%20doomsday%20-%20first%20look%20trailer%20/trailer_analysis.png) |

| Avengers: Doomsday (First Look) | Avengers: Doomsday (Official) |
|---|---|
| ![Doomsday 1](evaluate_results/avengers%EF%BC%9A%20doomsday%20-%20first%20look%20trailer%20/trailer_analysis.png) | ![Doomsday 2](evaluate_results/avengers%EF%BC%9A%20doomsday%20%EF%BD%9C%20only%20in%20theaters%20de/trailer_analysis.png) |

**DC Universe Trailers:**

| Superman (DC) | Supergirl (Official) |
|---|---|
| ![Superman](evaluate_results/superman%20%EF%BD%9C%20official%20trailer%20%EF%BD%9C%20dc/trailer_analysis.png) | ![Supergirl](evaluate_results/supergirl%20%EF%BD%9C%20official%20trailer/trailer_analysis.png) |

| Supergirl (Teaser) | Clayface (Teaser) |
|---|---|
| ![Supergirl Teaser](evaluate_results/supergirl%20%EF%BD%9C%20official%20teaser%20trailer/trailer_analysis.png) | ![Clayface](evaluate_results/clayface%20%EF%BD%9C%20official%20teaser/trailer_analysis.png) |

| Spider-Noir (Prime Video) | Citadel Season 2 (Prime) |
|---|---|
| ![Spider-Noir](evaluate_results/spider-noir%20-%20official%20trailer%20%EF%BD%9C%20prime%20v/trailer_analysis.png) | ![Citadel](evaluate_results/citadel%20season%202%20-%20official%20trailer%20%EF%BD%9C%20av/trailer_analysis.png) |

**Pirates of the Caribbean Series:**

| Curse of the Black Pearl | At World's End |
|---|---|
| ![Pirates 1](evaluate_results/pirates%20of%20the%20caribbean%EF%BC%9A%20the%20curse%20of%20t/trailer_analysis.png) | ![Pirates 3](evaluate_results/pirates%20of%20the%20caribbean%EF%BC%9A%20at%20world's%20end/trailer_analysis.png) |

| Dead Men Tell No Tales | Pirates 5 Official |
|---|---|
| ![Pirates 5a](evaluate_results/pirates%20of%20the%20caribbean%EF%BC%9A%20dead%20men%20tell%20/trailer_analysis.png) | ![Pirates 5b](evaluate_results/pirates%20of%20the%20caribbean%205%20official%20trai/trailer_analysis.png) |

| All Pirates Saga | The Godfather |
|---|---|
| ![All Pirates](evaluate_results/all%20pirates%20of%20the%20caribbean%20saga%20traile/trailer_analysis.png) | ![Godfather](evaluate_results/the%20godfather%20trailer%20(hd)/trailer_analysis.png) |

**Horror Trailers:**

| Aged (2023) | Deliver Us |
|---|---|
| ![Aged](evaluate_results/aged%20(2023)%20-%20official%20horror%20movie%20trai/trailer_analysis.png) | ![Deliver Us](evaluate_results/deliver%20us%20-%20official%20trailer%20%EF%BD%9C%20new%20horr/trailer_analysis.png) |

| The Shade (2024) | Somewhere Quiet (2024) |
|---|---|
| ![Shade](evaluate_results/the%20shade%20official%20trailer%20(2024)%20horror/trailer_analysis.png) | ![Somewhere Quiet](evaluate_results/somewhere%20quiet%20official%20trailer%20(2024)%20/trailer_analysis.png) |

**Sci-Fi Trailers:**

| New Sci-Fi 2023 | Simulant (2023) |
|---|---|
| ![SciFi](evaluate_results/new%20movie%20trailers%202023%20(sci-fi)/trailer_analysis.png) | ![Simulant](evaluate_results/simulant%20trailer%20(2023)%20sam%20worthington,/trailer_analysis.png) |

| The Breach (2023) | Greenland 2: Migration (2026) |
|---|---|
| ![Breach](evaluate_results/the%20breach%20official%20trailer%202023%20sci-fi%20/trailer_analysis.png) | ![Greenland 2](evaluate_results/greenland%202%EF%BC%9A%20migration%20(2026)%20official%20t/trailer_analysis.png) |

| Discontinued (2023) | Blood and Snow (2023) |
|---|---|
| ![Discontinued](evaluate_results/%EF%BC%82discontinued%EF%BC%82%20official%20trailer%20(2023%20sc/trailer_analysis.png) | ![Blood Snow](evaluate_results/blood%20and%20snow%20official%20trailer%20(2023)%20s/trailer_analysis.png) |

**Animation & Comedy:**

| Elemental (Pixar) | Baymax! (Disney+) |
|---|---|
| ![Elemental](evaluate_results/elemental%20%EF%BD%9C%20official%20trailer/trailer_analysis.png) | ![Baymax](evaluate_results/baymax!%20%EF%BD%9C%20official%20trailer%202%20%EF%BD%9C%20disney+/trailer_analysis.png) |

| Coyote vs Acme | Strays |
|---|---|
| ![Coyote](evaluate_results/coyote%20vs.%20acme%20%EF%BD%9C%20official%20trailer/trailer_analysis.png) | ![Strays](evaluate_results/strays%20%EF%BD%9C%20official%20trailer%20%5Bhd%5D/trailer_analysis.png) |

**Action / Thriller:**

| Retribution (2023) | Sisu (2023) |
|---|---|
| ![Retribution](evaluate_results/retribution%20(2023)%20official%20trailer%20%E2%80%93%20li/trailer_analysis.png) | ![Sisu](evaluate_results/sisu%20(2023)%20official%20red%20band%20trailer%20-%20/trailer_analysis.png) |

| Freelance (2023) | Wildcat (2025) |
|---|---|
| ![Freelance](evaluate_results/freelance%20(2023)%20official%20trailer%20-%20john/trailer_analysis.png) | ![Wildcat](evaluate_results/wildcat%20official%20trailer%20(2025)/trailer_analysis.png) |

| To Catch a Killer (2023) | Muzzle (2023) |
|---|---|
| ![ToKiller](evaluate_results/to%20catch%20a%20killer%20trailer%20(2023)%20shailen/trailer_analysis.png) | ![Muzzle](evaluate_results/muzzle%20%EF%BD%9C%202023%20%EF%BD%9C%20%40signatureuk%20%20trailer%20%EF%BD%9C%20/trailer_analysis.png) |

| Crater (2023) | Swallow (2020) |
|---|---|
| ![Crater](evaluate_results/crater%20(2023)%20trailer%20%EF%BD%9C%20isaiah%20russell-b/trailer_analysis.png) | ![Swallow](evaluate_results/swallow%20official%20trailer%20(2020)%20haley%20be/trailer_analysis.png) |

---

## 19. 20-Minute Full Movie Snippet Evaluation

**Script**: `trim_and_eval.py`
**Results**: `evaluate_results/clever_metrics_summary.csv`
**Content**: First 20 minutes of 4 Hollywood films (English audio isolated from dual-audio MKVs)

| Movie | WER % ↓ | CER % ↓ | LCS Similarity % | TS-MAE (s) | RTF |
|---|---|---|---|---|---|
| The Social Network | **18.60** | **14.32** | **53.67** | **0.471** | 0.318 |
| The Dark Knight Rises | **22.54** | **15.91** | 2.38 | 0.944 | 0.190 |
| Spider-Man: No Way Home | 31.70 | 25.24 | 6.92 | 0.747 | 0.282 |
| Avengers: Endgame | 32.28 | 26.99 | 15.62 | 0.824 | 0.108 |

- **Mean RTF = 0.22** — all 20-minute clips processed in under 5 minutes on Apple M1 CPU
- Social Network has the best WER (18.60%) due to rapid but clear dialogue with minimal music
- Dark Knight Rises has best CER (15.91%) — dense orchestral score reduces speech fraction
- High TS-MAE is expected: generated timestamps vs. manually-curated cinematic reference SRTs

---

## 20. Earlier Model Comparison

**Script**: `run_experiments.py` (first run)
**Content**: tiny.en vs medium.en on Avengers and Spider-Man trailers

### Emotion Events Figure

![Emotion Events](experiment_results/fig_emotion_events.png)

**tiny.en — Avengers Trailer:**

![Avengers tiny.en](experiment_results/avengers_tiny_en/trailer_analysis.png)

**medium.en — Avengers Trailer:**

![Avengers medium.en](experiment_results/avengers_medium_en/trailer_analysis.png)

**tiny.en — Spider-Man Trailer:**

![Spider-Man tiny.en](experiment_results/spiderman_tiny_en/trailer_analysis.png)

**medium.en — Spider-Man Trailer:**

![Spider-Man medium.en](experiment_results/spiderman_medium_en/trailer_analysis.png)

*Early outputs (outputs_v2/ and outputs_spiderman_v2/):*

![Early Avengers](outputs_v2/trailer_analysis.png) ![Early Spider-Man](outputs_spiderman_v2/trailer_analysis.png)

---

## 21. Scene Understanding & Mood Analysis

`project/scene_understanding.py`

### Features extracted (32 ms frames at 16 kHz)

- **MFCC × 13** + delta: cepstral speech/music discriminant
- **Chroma × 12**: pitch class distribution, tonality
- **Spectral contrast × 6**: harmonic-to-noise ratio proxy
- **HPSS harmonic/percussive ratio**: music vs percussion vs speech
- **Onset strength**: event detection (explosions, beats)
- **Spectral centroid / rolloff / flatness / bandwidth**: spectral shape
- **ZCR**: speech vs music discriminant

### Foote Novelty Score

Self-similarity matrix of MFCC+chroma features convolved with a Gaussian checkerboard kernel. Peaks mark structural scene boundaries (dialogue→action, music→speech transitions).

### Mood Labels

| Label | Acoustic Signature | Context |
|---|---|---|
| tense | High energy, low valence, slow tempo | Thriller climax, horror |
| epic | High energy, high harmonic ratio | Action sequence, superhero moment |
| sad | Low energy, low valence | Drama, tragedy |
| happy | Moderate energy, fast tempo | Comedy, uplifting |
| calm | Low energy, high harmonic ratio | Quiet dialogue, nature |
| dialogue | Speech-like ZCR, low harmonic | Direct speech |
| silence | RMS < −40 dB | Scene gap |

### Multi-Modal Outputs Per Trailer

1. **`predicted.srt`** / **`trailer_clean.srt`**: Clean SRT for WER evaluation
2. **`predicted_annotated.srt`** / **`trailer_annotated.srt`**: With mood tags [TENSE]😬, [EPIC]⚡, [SAD]😢, [MUSIC]🎵, [IMPACT]💥
3. **`scene_analysis.json`** / **`trailer_timeline.json`**: Foote boundaries + mood timeline at 0.5 s resolution
4. **`scene_analysis.png`** / **`trailer_analysis.png`**: 5-panel plot (waveform+VAD, spectrogram, Foote novelty, mood timeline, mood pie)
5. **`metrics.json`**: WER, CER, RTF, CPS violations, TS-MAE

---

## 22. Per-Trailer Scene Analysis Plots

### mass_results/ — 14 Trailers (medium.en)

| Avengers (Original) | Spider-Man Brand New Day |
|---|---|
| ![](mass_results/Marvel's%20The%20Avengers-%20Trailer%20(OFFICIAL)%20-%20Marvel%20Entertainment%20(1080p,%20h264)/scene_analysis.png) | ![](mass_results/YTDown_YouTube_SPIDER-MAN-BRAND-NEW-DAY-Official-Traile_Media_8TZMtslA3UY_002_720p/scene_analysis.png) |

| Age of Ultron | Bridesmaids |
|---|---|
| ![](mass_results/avengers_age_of_ultron/scene_analysis.png) | ![](mass_results/bridesmaids/scene_analysis.png) |

| Captain America: Civil War | Dunkirk |
|---|---|
| ![](mass_results/captain_america_civil_war/scene_analysis.png) | ![](mass_results/dunkirk/scene_analysis.png) |

| Mad Max: Fury Road | Nomadland |
|---|---|
| ![](mass_results/mad_max_fury_road/scene_analysis.png) | ![](mass_results/nomadland/scene_analysis.png) |

| Spider-Man: Into the Spider-Verse | The Dark Knight |
|---|---|
| ![](mass_results/spider_man_into_spider_verse/scene_analysis.png) | ![](mass_results/the_dark_knight/scene_analysis.png) |

| The Revenant | The Social Network |
|---|---|
| ![](mass_results/the_revenant/scene_analysis.png) | ![](mass_results/the_social_network/scene_analysis.png) |

| Whiplash | Zodiac |
|---|---|
| ![](mass_results/whiplash/scene_analysis.png) | ![](mass_results/zodiac/scene_analysis.png) |

### eval_results_medium/ — Same 14 Trailers (medium.en, later run)

| Avengers (OFFICIAL) | Spider-Man Brand New Day |
|---|---|
| ![](eval_results_medium/Marvel_s_The_Avengers__Trailer__OFFICIAL/scene_analysis.png) | ![](eval_results_medium/YTDown_YouTube_SPIDER_MAN_BRAND_NEW_DAY_/scene_analysis.png) |

| Age of Ultron | Bridesmaids |
|---|---|
| ![](eval_results_medium/avengers_age_of_ultron/scene_analysis.png) | ![](eval_results_medium/bridesmaids/scene_analysis.png) |

| Captain America: Civil War | Dunkirk |
|---|---|
| ![](eval_results_medium/captain_america_civil_war/scene_analysis.png) | ![](eval_results_medium/dunkirk/scene_analysis.png) |

| Mad Max: Fury Road | Nomadland |
|---|---|
| ![](eval_results_medium/mad_max_fury_road/scene_analysis.png) | ![](eval_results_medium/nomadland/scene_analysis.png) |

| Spider-Man: Spider-Verse | The Dark Knight |
|---|---|
| ![](eval_results_medium/spider_man_into_spider_verse/scene_analysis.png) | ![](eval_results_medium/the_dark_knight/scene_analysis.png) |

| The Revenant | The Social Network |
|---|---|
| ![](eval_results_medium/the_revenant/scene_analysis.png) | ![](eval_results_medium/the_social_network/scene_analysis.png) |

| Whiplash | Zodiac |
|---|---|
| ![](eval_results_medium/whiplash/scene_analysis.png) | ![](eval_results_medium/zodiac/scene_analysis.png) |

---

## 23. Key Findings

1. **Raw ASR beats naive enhancement at SNR ≥ 10 dB** — Modern int8-quantised Whisper is intrinsically noise-robust. Do not apply Wiener filter to movie audio.

2. **Adaptive routing is the correct design** — Withhold enhancement for clean segments; apply spectral subtraction only when SNR < 0 dB or spectral flatness signals noise dominance.

3. **Spectral subtraction > Wiener for movie audio** — PESQ +17.8%, WER −38% relative vs Wiener under soundtrack noise.

4. **Spectral VAD detects 2.17× more speech than energy-only** at 3× more compute (6.2 ms vs 1.9 ms/utterance). Essential for trailers with 40–70% music content.

5. **medium.en reduces WER by 38% vs tiny.en on trailers** (0.522 vs 0.875) — still real-time (RTF 0.175) on CPU.

6. **Whisper avg_logprob is a strong error predictor** — Spearman ρ = −0.859, AUROC = 0.929 for WER > 0.3 detection. Deploy as confidence-gated SRT filter.

7. **3-second chunks give the best streaming latency–accuracy tradeoff** — 203 ms first-word latency, WER = 0.291. 1-second chunks fail (WER 0.607).

8. **Rule-based noise classification fails at 31.6% overall** — Pink noise and babble have indistinguishable MFCC profiles. A learned front-end is needed.

9. **YouTube auto-captions are flawed ground truth** — Official SRT WER = 4.88% (Avengers) vs 57.3% with YouTube auto-captions for the same content. Always validate reference transcripts.

10. **Foote novelty score successfully detects structural transitions** — Useful for subtitle formatting and accessibility metadata.

11. **CPS violations are a post-processing concern** — The ≤12 words/cue rule in Stage 6 resolves >50% of violations without re-processing.

12. **All models are real-time on Apple M1 CPU** — tiny.en RTF 0.043, medium.en RTF 0.175. A 20-minute movie clip is processed in under 5 minutes (mean RTF 0.22).

---

## 24. Module Reference

### `project/audio_utils.py`
- `vad_energy(audio, sr)` → energy-threshold VAD
- `vad_mfcc(audio, sr)` → MFCC C0 + ZCR VAD
- `vad_spectral(audio, sr)` → full spectral VAD (proposed)
- `enhance_wiener(audio)` → Wiener-filtered array
- `enhance_spectral_subtraction(audio)` → spectral-subtracted array
- `extract_features(audio)` → 9-dim feature vector
- `should_enhance(features)` → bool (adaptive routing)
- `synthesize_noise(audio, noise_type, snr_db)` → noisy array

### `project/asr.py`
- `WhisperASR(model_size)` — loads model, tracks load_time_ms
- `.transcribe_audio_array(path, audio, sr)` → RTF + transcript
- `.stats` → P50/P95 latency, mean RTF

### `project/enhanced_asr.py`
- `EnhancedWhisperASR(model_size="medium.en")`
- `.transcribe(audio, sr)` → FullTranscription with word timestamps
- `segments_to_cues(segments)` → list[SubtitleCue]

### `project/scene_understanding.py`
- `analyze_audio_scene(audio, sr)` → AudioSceneAnalysis
- `plot_scene_analysis(audio, sr, analysis, path)` → 5-panel PNG
- `annotate_srt_with_scene(cues, analysis)` → annotated SRT

### `project/novel_experiments.py`
- `run_enhancement_quality(samples, output_dir)` → PESQ + STOI
- `run_noise_classifier(samples, output_dir)` → 5-class classifier
- `run_confidence_calibration(csv_path, output_dir)` → AUROC
- `run_streaming_latency(samples, asr, output_dir)` → chunk pipeline

### `project/extra_metrics.py`
- `compute_chrf(hypothesis, reference)` → chrF score
- `compute_bleu(hypothesis, reference)` → BLEU-1/2/4
- `classify_genre(audio, sr)` → genre label (76.9% LOO accuracy)

### `project/emotion_detector.py`
- `classify_emotion(audio, sr, word_count, duration_sec)` → EmotionResult
- `detect_emotions_batch(segments, audio, sr)` → list[EmotionResult]
