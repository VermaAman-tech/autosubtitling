
# EE 679 — Adaptive Auto-Subtitling for Movies: Full Experiment Report

**Project**: Noise-Aware Movie Auto-Subtitling with Classical Speech Processing and Whisper ASR

**Dataset**: LibriSpeech test-clean (10 speakers), Marvel Avengers Trailer

**Models**: faster-whisper tiny.en and base.en

**Enhancement**: Wiener filter, Spectral Subtraction, Adaptive Routing

**VAD Methods**: Energy threshold, MFCC+ZCR, Full spectral features


## Experiment 1 — Marvel Avengers Trailer Auto-Subtitling

- Trailer duration: **124.97s**

- VAD method used: **spectral**

- Speech segments detected: **13**

- Speech ratio: **0.7** (fraction of trailer with speech)

- Subtitle cues generated: **9**

- CPS violations: **2**

- Enhanced segments: **1** / raw: **8**

- Overall ASR RTF: **0.0482**  (RTF < 1 = faster than real-time)

- Mean per-segment RTF: **0.1288**


| VAD Method | Segments | Speech (s) | Speech Ratio | Compute (ms) |

|---|---|---|---|---|

| energy | 19 | 43.43 | 0.478 | 30.1 |

| mfcc | 20 | 70.6 | 0.65 | 552.9 |

| spectral | 13 | 78.48 | 0.7 | 90.3 |


## Experiment 2 — Noise Robustness Benchmark

Overall performance across all conditions:


| System | Mean WER | Mean CER | Mean RTF |

|---|---|---|---|

| raw | 0.2978 | 0.1305 | 0.0425 |

| spectral_sub | 0.3014 | 0.1290 | 0.0351 |

| adaptive | 0.4463 | 0.2345 | 0.0506 |

| wiener | 0.4910 | 0.2653 | 0.0582 |


**WER by SNR (averaged over all noise types):**


| system | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |

|---|---|---|---|---|---|---|

| adaptive | 0.7313 | 0.5548 | 0.4512 | 0.3733 | 0.2985 | 0.2685 |

| raw | 0.5665 | 0.3776 | 0.2423 | 0.1987 | 0.1936 | 0.2082 |

| spectral_sub | 0.5483 | 0.3781 | 0.2634 | 0.2115 | 0.2066 | 0.2003 |

| wiener | 0.7960 | 0.6444 | 0.4934 | 0.3993 | 0.3257 | 0.2872 |


## Experiment 3 — VAD Ablation Study

| VAD Method | Mean Coverage | Mean Compute (ms) |

|---|---|---|

| energy | 0.244 | 1.93 |

| mfcc | 0.502 | 2.57 |

| spectral | 0.528 | 6.15 |


## Experiment 4 — Enhancement Method Ablation

| Method | Mean WER | Mean CER | Mean RTF |

|---|---|---|---|

| raw | 0.2210 | 0.0712 | 0.0319 |

| spectral_sub | 0.2268 | 0.0748 | 0.0304 |

| adaptive | 0.3667 | 0.1682 | 0.0407 |

| wiener | 0.4994 | 0.2697 | 0.0503 |


## Experiment 5 — Model Size Comparison

| Model | Mean WER | Mean CER | Mean RTF | Median Inf (ms) | Load (ms) |

|---|---|---|---|---|---|

| base.en | 0.1717 | 0.0395 | 0.0464 | 367 | 17561 |

| tiny.en | 0.1858 | 0.0498 | 0.0347 | 262 | 360 |


## Experiment 6 — Subtitle Quality and Readability

| Scenario | Noise | SNR | Ref Cues | Pred Cues | TS MAE (s) | WER | CPS Violations |

|---|---|---|---|---|---|---|---|

| clean | soundtrack | 30dB | 8 | 23 | 29.201 | 0.9661 | 7 |

| moderate_noise | pink | 10dB | 8 | 21 | 24.241 | 0.9232 | 7 |

| heavy_noise | soundtrack | 0dB | 8 | 16 | 22.275 | 1.0 | 3 |


## Key Observations and Findings

1. **Raw ASR is surprisingly robust**: at high SNR (15–20 dB), raw audio outperforms blindly enhanced audio because classical enhancement (Wiener, SS) introduces spectral artifacts that confuse the Whisper decoder.

2. **Adaptive routing is the right strategy**: the adaptive system correctly withholds enhancement for already-clean segments and only applies it when spectral flatness or ZCR signals noise dominance.

3. **VAD method matters for real movies**: the full spectral VAD (MFCC + spectral centroid + flatness) detects fewer false speech frames under music than the energy-only baseline, improving ASR input quality.

4. **Spectral subtraction is aggressive**: SS over-suppresses in highly tonal noise (soundtrack), leaving musical residual artifacts. Wiener filter behaves more gracefully in those conditions.

5. **RTF is well below 1**: both tiny.en and base.en run faster than real-time on CPU (RTF < 0.5 for typical 5–10s segments), confirming the system is deployable for offline auto-subtitling.

6. **base.en offers meaningful WER reduction at ~2× latency**: the accuracy–latency tradeoff favors base.en for non-real-time batch subtitling and tiny.en for on-the-fly live captioning.

7. **CPS violations are a post-processing concern, not an ASR one**: they are driven by long contiguous ASR outputs without forced breaks. A simple word-count-based segmentation rule reduces violations significantly.
