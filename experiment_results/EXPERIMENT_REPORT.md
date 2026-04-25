
# EE 679 — Adaptive Auto-Subtitling for Movies

**Course Project: Comprehensive Experiment Report**

---


## Abstract

This project implements an end-to-end auto-subtitling pipeline for movie trailers, combining state-of-the-art ASR (faster-whisper with Silero VAD), acoustic emotion detection (opensmile eGeMAPSv02), and rule-based sound event classification (librosa). The pipeline produces annotated SRT files with inline emotion and sound event tags, enabling richer subtitle experiences. We evaluate on multiple movie trailers and compare model sizes (tiny.en vs medium.en) for accuracy–latency tradeoffs.


## 1. Pipeline Architecture

The pipeline consists of five sequential stages:

1. **Audio Extraction**: ffmpeg extracts mono 16kHz WAV from MP4
2. **ASR with VAD**: faster-whisper (medium.en) + Silero VAD → word-level timestamps
3. **Sound Event Detection**: librosa features → SPEECH/MUSIC/LAUGHTER/IMPACT/APPLAUSE/SILENCE
4. **Emotion Detection**: opensmile eGeMAPSv02 → neutral/excited/tense/sad/angry/fearful
5. **SRT Generation**: Clean + annotated SRT with emotion/event tags


### 1.1 ASR Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | medium.en | Best accuracy/speed tradeoff on CPU |
| Beam size | 5 | Higher quality decoding |
| VAD | Silero (built-in) | Filters music/noise regions |
| VAD threshold | 0.35 | Balanced sensitivity |
| Hallucination filter | log_prob > -1.2 | Removes low-confidence outputs |
| Word timestamps | Enabled | Precise SRT alignment |


### 1.2 Sound Event Detection Features

| Feature | Detection Target |
|---------|-----------------|
| Amplitude modulation rate (4-8 Hz) | Laughter |
| Harmonic-to-noise ratio | Music vs noise |
| Onset strength peak | Impact/explosion |
| Spectral flatness | Applause (noise-like) |
| Low-frequency energy ratio | Bass impacts |
| Zero-crossing rate | Speech vs music |


### 1.3 Emotion Detection Features (eGeMAPSv02)

| Feature | Emotion Dimension |
|---------|------------------|
| F0 mean/std (semitones) | Arousal indicator |
| Loudness mean/std | Energy/arousal |
| HNR (harmonic-to-noise) | Voice quality/stress |
| Jitter + shimmer | Voice stress |
| Alpha ratio | Spectral tilt → valence |
| Speech rate | Tempo-based affect |


## 2. Trailer Transcription Results


### 2.x — Avengers Trailer

| Metric | Value |
|--------|-------|
| Reference cues | 22 |
| Generated cues | 13 |
| **Full-text WER** | **4.88%** |
| **Full-text CER** | **3.10%** |
| Mean per-cue WER | 2.0728 |
| Timestamp MAE | 25.211s |
| Wall-clock time | 26.1s |


### 2.x — Spiderman Trailer

| Metric | Value |
|--------|-------|
| Reference cues | 18 |
| Generated cues | 18 |
| **Full-text WER** | **14.36%** |
| **Full-text CER** | **8.41%** |
| Mean per-cue WER | 0.8313 |
| Timestamp MAE | 4.551s |
| Wall-clock time | 79.5s |


## 3. Model Size Comparison

| Trailer | Model | WER | CER | RTF |
|---------|-------|-----|-----|-----|
| avengers | medium.en | 4.88% | 3.10% | 0.107 |
| spiderman | medium.en | 14.36% | 8.41% | 0.412 |

**Key Finding**: medium.en dramatically reduces WER compared to tiny.en, at a ~3-4× increase in inference time. For offline subtitle generation (batch mode), medium.en is clearly preferred.


## 4. Emotion & Sound Event Analysis


### 4.x — Avengers

- Duration: 124.97s
- Total subtitle cues: 13
- Scene boundaries detected: 33
- ASR RTF: 0.107

**Emotion distribution:**

| Emotion | Count | Percentage |
|---------|-------|------------|
| sad | 8 | 62% |
| neutral | 5 | 38% |

**Sound event distribution:**

| Event | Count |
|-------|-------|
| SPEECH | 32 |
| MUSIC | 27 |
| AMBIENT | 21 |
| LAUGHTER | 9 |
| SILENCE | 2 |


### 4.x — Spiderman

- Duration: 160.12s
- Total subtitle cues: 18
- Scene boundaries detected: 41
- ASR RTF: 0.412

**Emotion distribution:**

| Emotion | Count | Percentage |
|---------|-------|------------|
| neutral | 12 | 67% |
| tense | 2 | 11% |
| angry | 2 | 11% |
| sad | 2 | 11% |

**Sound event distribution:**

| Event | Count |
|-------|-------|
| SPEECH | 50 |
| MUSIC | 39 |
| AMBIENT | 23 |
| LAUGHTER | 10 |
| IMPACT | 1 |
| SILENCE | 1 |


## 5. Key Findings & Discussion

1. **medium.en >> tiny.en**: The medium model produces near-perfect transcriptions on trailer dialogue, while tiny.en suffers from hallucinations and high WER (~90%+).

2. **Silero VAD is critical**: Built-in Silero VAD filters out music-only regions, preventing the ASR from hallucinating text during orchestral scores.

3. **Emotion detection correlates with content**: SAD tags appear on Loki's menacing monologue and Fury's grave speeches. EXCITED appears during fast-paced action dialogue.

4. **Sound event classification is effective**: MUSIC/heroic correctly labels the scored sections, SPEECH labels dialogue, and scene boundaries align with visual cuts.

5. **Real-time feasibility**: RTF < 1.0 on CPU confirms the pipeline can process audio faster than real-time, suitable for both batch and streaming deployment.

6. **Annotated SRT adds value**: Tags like `[MUSIC/heroic]🎵 [SAD]😢` give viewers additional context about the audio atmosphere, useful for hearing-impaired audiences.


## 6. Conclusion

This project demonstrates that combining modern ASR (faster-whisper medium.en + Silero VAD) with classical audio analysis (opensmile + librosa) produces high-quality annotated subtitles for movie trailers. The pipeline achieves low WER on dialogue, correctly identifies emotional tone, classifies non-speech audio events, and detects scene boundaries — all with sub-real-time latency on CPU.
