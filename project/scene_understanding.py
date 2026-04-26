"""
Scene Understanding from Audio
================================
Analyzes the acoustic context of movie/trailer audio beyond speech:
  - Music mood detection (tense, epic, sad, happy, calm, neutral)
  - Scene type classification (action, dialogue, transition, silence)
  - Acoustic scene novelty (Foote self-similarity novelty score)
  - Temporal mood timeline for annotated SRT export

All features derived from classical DSP taught in EE 679:
  MFCC, chroma, spectral contrast, RMS, tempo, onset strength,
  harmonic/percussive separation (HPSS), tonnetz.

References:
  - Foote, J. (1999). Visualizing music and audio using self-similarity.
    ACM Multimedia.
  - Kim et al. (2010). Music emotion recognition: A state-of-the-art review.
    ISMIR.
  - Müller, M. (2015). Fundamentals of Music Processing. Springer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _PLT = True
except ImportError:
    _PLT = False


TARGET_SR = 16_000

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class MusicMood:
    """Per-frame mood estimate from musical audio features."""
    label: str          # tense | epic | sad | happy | calm | neutral | dialogue
    valence: float      # -1 (negative) … +1 (positive)
    energy: float       # 0 (low energy) … 1 (high energy)
    tempo_bpm: float    # estimated tempo, 0 if unreliable
    confidence: float   # 0..1

    EMOJI = {
        "tense":    "😬",
        "epic":     "⚡",
        "sad":      "😢",
        "happy":    "😊",
        "calm":     "😌",
        "neutral":  "😐",
        "dialogue": "🗣",
        "silence":  "🤫",
    }

    @property
    def emoji(self) -> str:
        return self.EMOJI.get(self.label, "")

    @property
    def srt_tag(self) -> str:
        return f"[{self.label.upper()}]{self.emoji}"


@dataclass
class SceneBoundary:
    """A detected scene/section boundary in the audio."""
    time_sec: float
    novelty_score: float    # Foote novelty score at this boundary
    context_before: str     # scene type before this boundary
    context_after: str      # scene type after


@dataclass
class SceneSegment:
    """A contiguous scene with uniform acoustic character."""
    start_sec: float
    end_sec: float
    scene_type: str         # action | dialogue | transition | music_only | silence
    mood: MusicMood
    has_speech: bool
    rms_db: float
    spectral_centroid_hz: float
    tempo_bpm: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class AudioSceneAnalysis:
    """Full scene analysis result for one audio track."""
    duration_sec: float
    segments: list[SceneSegment]
    boundaries: list[SceneBoundary]
    scene_mood_timeline: list[dict]   # frame-level timeline for plotting
    dominant_mood: str
    music_fraction: float             # fraction of audio with music
    speech_fraction: float
    silence_fraction: float
    compute_ms: float

    @property
    def mood_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for seg in self.segments:
            counts[seg.mood.label] = counts.get(seg.mood.label, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "duration_sec": self.duration_sec,
            "dominant_mood": self.dominant_mood,
            "music_fraction": round(self.music_fraction, 3),
            "speech_fraction": round(self.speech_fraction, 3),
            "silence_fraction": round(self.silence_fraction, 3),
            "mood_counts": self.mood_counts,
            "n_segments": len(self.segments),
            "n_boundaries": len(self.boundaries),
            "compute_ms": round(self.compute_ms, 1),
        }


# ─── Feature extraction ────────────────────────────────────────────────────────

def _extract_scene_features(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> dict:
    """
    Extract a rich feature set for scene understanding.
    Returns dict of frame-wise feature arrays (all same length).
    """
    audio = audio.astype(np.float32)

    # Basic energy
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)

    # Spectral shape
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    flatness  = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    contrast  = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=6)

    # MFCC (13) + delta
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    d_mfcc = librosa.feature.delta(mfcc)

    # Harmonic/percussive separation
    harmonic, percussive = librosa.effects.hpss(audio)
    harm_rms = librosa.feature.rms(y=harmonic, frame_length=n_fft, hop_length=hop_length)[0]
    perc_rms = librosa.feature.rms(y=percussive, frame_length=n_fft, hop_length=hop_length)[0]
    harm_ratio = harm_rms / (harm_rms + perc_rms + 1e-8)   # high = more tonal/musical

    # Chroma (12 pitch classes) — harmonic content
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop_length, n_chroma=12)
    chroma_std = np.std(chroma, axis=0)  # spread across pitch classes

    # Onset strength (percussive events)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)

    # Zero crossing rate (speech vs music discriminator)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft//4, hop_length=hop_length)[0]

    # Align all to min length
    n = min(len(rms), len(centroid), len(flatness), len(onset_env), len(harm_ratio), len(zcr))

    return {
        "rms_db":       rms_db[:n],
        "centroid":     centroid[:n] / (sr / 2),      # normalized
        "rolloff":      rolloff[:n] / (sr / 2),       # normalized
        "flatness":     flatness[:n],
        "bandwidth":    bandwidth[:n] / (sr / 2),     # normalized
        "contrast_mean":np.mean(contrast[:, :n], axis=0),
        "mfcc":         mfcc[:, :n],
        "d_mfcc_norm":  np.linalg.norm(d_mfcc[:, :n], axis=0),
        "harm_ratio":   harm_ratio[:n],
        "perc_rms":     perc_rms[:n],
        "chroma_std":   chroma_std[:n],
        "onset_env":    onset_env[:n],
        "zcr":          zcr[:n],
        "n_frames":     n,
    }


def _classify_frame_mood(
    rms_db: float,
    centroid: float,  # spectral centroid (normalised, kept for potential future use)
    flatness: float,
    harm_ratio: float,
    onset_env: float,
    zcr: float,
    chroma_std: float,
    contrast_mean: float,
    d_mfcc_norm: float,
    tempo_bpm: float = 0.0,
) -> MusicMood:
    """
    Rule-based per-frame mood classification based on Russell's circumplex model.

    Features map to the 2D affective space:
      Valence  (negative ↔ positive): chroma variety, spectral contrast, harm ratio
      Arousal  (calm ↔ excited):      RMS energy, onset strength, tempo

    Labels and their acoustic signatures:
      silence  : RMS < −42 dB
      dialogue : speech-like ZCR + formant pattern + low harmonic ratio
      tense    : high arousal + negative valence + slow/building (suspense music)
      epic     : high arousal + high harmonic ratio + strong percussive onset (orchestral)
      sad      : low arousal + negative valence + high harmonic sustain
      happy    : moderate arousal + positive valence + fast tempo (upbeat)
      calm     : low arousal + positive/neutral valence + sustained harmonics
      neutral  : mid-range everything (background underscore, no strong mood)
    """
    # ── Silence ──────────────────────────────────────────────────────────────
    if rms_db < -50:
        return MusicMood("silence", 0.0, 0.0, tempo_bpm, 0.92)

    # ── Energy (arousal proxy) ────────────────────────────────────────────────
    # rms_db typically −50..0 dB for normalised audio content
    energy = float(np.clip((rms_db + 50) / 50, 0, 1))

    # ── Speech detection ─────────────────────────────────────────────────────
    # Speech: ZCR moderate, low harmonic ratio, changing MFCC
    is_speech = (zcr > 0.035) and (harm_ratio < 0.52) and (d_mfcc_norm > 0.3) and (flatness < 0.35)
    if is_speech:
        speech_energy = energy
        valence_s = 0.05  # slight positive (most dialogue is neutral-positive)
        return MusicMood("dialogue", valence_s, speech_energy, tempo_bpm, 0.72)

    # ── Valence computation ───────────────────────────────────────────────────
    # Major-like tonality: chroma variety (major chords spread across pitch classes)
    chroma_valence = float(np.clip((chroma_std - 0.08) / 0.20, -1.0, 1.0))

    # High spectral contrast → greater separation between harmonic peaks and noise → often minor/tense
    contrast_valence = float(np.clip(0.3 - contrast_mean * 0.6, -1.0, 1.0))

    # Harmonic ratio: pure tonal content alone is neutral (could be major or minor)
    # Combine with chroma: high chroma_std + high harm_ratio → positive (major)
    valence = float(np.clip(
        0.50 * chroma_valence + 0.30 * contrast_valence + 0.20 * (harm_ratio - 0.5),
        -1.0, 1.0
    ))

    # ── Onset/percussive strength (action indicator) ─────────────────────────
    # Normalize onset to 0..1 (typical range 0..3)
    onset_norm = float(np.clip(onset_env / 2.0, 0.0, 1.0))

    # ── Tempo contribution ────────────────────────────────────────────────────
    fast_tempo  = (tempo_bpm > 105) if tempo_bpm > 0 else False
    build_tempo = (70 <= tempo_bpm <= 105) if tempo_bpm > 0 else True  # suspense range

    # ── Combined arousal ─────────────────────────────────────────────────────
    arousal = float(np.clip(
        0.50 * energy + 0.30 * onset_norm + 0.20 * (float(fast_tempo) * 0.3),
        0.0, 1.0
    ))

    # ── Label assignment ──────────────────────────────────────────────────────
    # Priority order: most-specific checks first

    # EPIC: high energy + high harmonic content + strong onsets (orchestral action)
    if energy > 0.50 and harm_ratio > 0.45 and onset_norm > 0.20:
        conf = 0.60 + energy * 0.25 + harm_ratio * 0.15
        return MusicMood("epic", valence, arousal, tempo_bpm, float(np.clip(conf, 0, 1)))

    # TENSE: moderate+ energy + negative/neutral valence + building tempo (suspense)
    if energy > 0.40 and valence < 0.05 and build_tempo and harm_ratio < 0.55:
        conf = 0.58 + energy * 0.20 - valence * 0.15
        return MusicMood("tense", valence, arousal, tempo_bpm, float(np.clip(conf, 0, 1)))

    # SAD: low-moderate energy + negative valence + sustained harmonics
    if energy < 0.50 and valence < -0.05 and harm_ratio > 0.35 and onset_norm < 0.35:
        conf = 0.55 + (0.50 - energy) * 0.40 + abs(valence) * 0.20
        return MusicMood("sad", valence, arousal, tempo_bpm, float(np.clip(conf, 0, 1)))

    # HAPPY: moderate+ energy + positive valence + fast tempo
    if energy > 0.35 and valence > 0.10 and fast_tempo:
        conf = 0.55 + valence * 0.25 + (energy - 0.35) * 0.20
        return MusicMood("happy", valence, arousal, tempo_bpm, float(np.clip(conf, 0, 1)))

    # CALM: low energy + positive/neutral valence + harmonic (gentle underscore)
    if energy < 0.45 and valence >= -0.05 and harm_ratio > 0.40:
        conf = 0.52 + (0.45 - energy) * 0.30 + harm_ratio * 0.15
        return MusicMood("calm", valence, arousal, tempo_bpm, float(np.clip(conf, 0, 1)))

    # NEUTRAL: catch-all
    return MusicMood("neutral", valence, arousal, tempo_bpm, 0.45)


# ─── Foote novelty score ────────────────────────────────────────────────────────

def _foote_novelty(
    feature_matrix: np.ndarray,
    kernel_size: int = 16,
) -> np.ndarray:
    """
    Compute the Foote self-similarity novelty score.

    The novelty at time t is the "checkerboard" correlation of the
    self-similarity matrix, peaking at structural boundaries.

    feature_matrix: (n_features, n_frames)
    Returns novelty curve of length n_frames.
    """
    # Normalise features per frame
    feat = feature_matrix.copy()
    norms = np.linalg.norm(feat, axis=0, keepdims=True) + 1e-8
    feat = feat / norms

    # Self-similarity matrix
    sim = feat.T @ feat   # (n_frames, n_frames)
    sim = np.clip(sim, -1, 1)

    # Gaussian checkerboard kernel
    k = kernel_size
    kernel = np.zeros((2*k, 2*k))
    kernel[:k, :k] = +1
    kernel[k:, k:] = +1
    kernel[:k, k:] = -1
    kernel[k:, :k] = -1
    # Gaussian taper
    g = np.outer(
        np.exp(-np.arange(-k, k)**2 / (2*(k/3)**2)),
        np.exp(-np.arange(-k, k)**2 / (2*(k/3)**2)),
    )
    kernel = kernel * g

    from scipy.signal import fftconvolve
    novelty = np.zeros(sim.shape[0])
    for t in range(k, sim.shape[0] - k):
        patch = sim[t-k:t+k, t-k:t+k]
        if patch.shape == kernel.shape:
            novelty[t] = float(np.sum(patch * kernel))

    # Normalise
    mx = np.max(np.abs(novelty)) + 1e-8
    novelty = novelty / mx
    return novelty


def _pick_boundaries(novelty: np.ndarray, min_gap_frames: int = 8, threshold: float = 0.3) -> list[int]:
    from scipy.signal import find_peaks
    peaks, props = find_peaks(novelty, distance=min_gap_frames, height=threshold)
    return peaks.tolist()


# ─── Main analysis entry point ─────────────────────────────────────────────────

def analyze_audio_scene(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    hop_length: int = 512,
    n_fft: int = 2048,
    min_segment_secs: float = 0.5,
) -> AudioSceneAnalysis:
    """
    Full scene analysis pipeline:
      1. Extract frame-level features
      2. Compute Foote novelty → scene boundaries
      3. Classify mood per segment
      4. Aggregate statistics

    Parameters
    ----------
    audio         : mono float32 waveform
    sr            : sample rate
    hop_length    : STFT hop length (affects time resolution)
    segment_secs  : target duration for each scene segment
    """
    t0 = time.perf_counter()
    audio = audio.astype(np.float32)
    duration_sec = len(audio) / sr

    # ── 1. Feature extraction ───────────────────────────────────────────────
    feats = _extract_scene_features(audio, sr, hop_length, n_fft)
    n_frames = feats["n_frames"]
    frames_per_sec = sr / hop_length

    # Global tempo estimate
    tempo_arr, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    global_tempo = float(tempo_arr[0]) if hasattr(tempo_arr, "__len__") and len(tempo_arr) > 0 else float(tempo_arr)

    # ── 2. Foote novelty + boundaries ───────────────────────────────────────
    # Use MFCC + chroma as main feature matrix for novelty
    n = feats["n_frames"]
    chroma_short = np.repeat(
        feats.get("chroma_std", np.zeros(n)).reshape(1, -1),
        3, axis=0
    )
    feat_matrix = np.vstack([feats["mfcc"][:, :n], chroma_short[:, :n]])

    min_gap = max(4, int(min_segment_secs * frames_per_sec))
    novelty = _foote_novelty(feat_matrix, kernel_size=min(16, n_frames//4))
    boundary_frames = _pick_boundaries(novelty, min_gap_frames=min_gap, threshold=0.25)

    # Convert to times
    boundary_times = [0.0] + [f / frames_per_sec for f in boundary_frames] + [duration_sec]
    boundary_times = sorted(set(round(t, 3) for t in boundary_times))

    # ── 3. Per-segment classification ───────────────────────────────────────
    segments: list[SceneSegment] = []
    scene_mood_timeline: list[dict] = []

    # Frame-level mood for timeline
    for frame_idx in range(0, n_frames, max(1, int(0.5 * frames_per_sec))):
        t_sec = frame_idx / frames_per_sec
        mood = _classify_frame_mood(
            rms_db=float(feats["rms_db"][frame_idx]),
            centroid=float(feats["centroid"][frame_idx]),
            flatness=float(feats["flatness"][frame_idx]),
            harm_ratio=float(feats["harm_ratio"][frame_idx]),
            onset_env=float(feats["onset_env"][frame_idx]),
            zcr=float(feats["zcr"][frame_idx]),
            chroma_std=float(feats["chroma_std"][frame_idx]),
            contrast_mean=float(feats["contrast_mean"][frame_idx]),
            d_mfcc_norm=float(feats["d_mfcc_norm"][frame_idx]),
            tempo_bpm=global_tempo,
        )
        scene_mood_timeline.append({
            "time_sec": round(t_sec, 2),
            "mood": mood.label,
            "energy": round(mood.energy, 3),
            "valence": round(mood.valence, 3),
        })

    for i in range(len(boundary_times) - 1):
        seg_start = boundary_times[i]
        seg_end   = boundary_times[i + 1]
        if seg_end - seg_start < min_segment_secs:
            continue

        seg_audio = audio[int(seg_start * sr): int(seg_end * sr)]
        if len(seg_audio) < int(min_segment_secs * sr):
            continue

        # Frame indices for this segment
        f_start = int(seg_start * frames_per_sec)
        f_end   = min(int(seg_end * frames_per_sec), n_frames)
        if f_end <= f_start:
            continue

        seg_rms_db        = float(np.mean(feats["rms_db"][f_start:f_end]))
        seg_centroid_mean = float(np.mean(feats["centroid"][f_start:f_end])) * (sr / 2)
        seg_flatness      = float(np.mean(feats["flatness"][f_start:f_end]))
        seg_harm_ratio    = float(np.mean(feats["harm_ratio"][f_start:f_end]))
        seg_onset         = float(np.mean(feats["onset_env"][f_start:f_end]))
        seg_zcr           = float(np.mean(feats["zcr"][f_start:f_end]))
        seg_chroma_std    = float(np.mean(feats["chroma_std"][f_start:f_end]))
        seg_contrast      = float(np.mean(feats["contrast_mean"][f_start:f_end]))
        seg_d_mfcc        = float(np.mean(feats["d_mfcc_norm"][f_start:f_end]))

        # Local tempo
        try:
            local_tempo_arr, _ = librosa.beat.beat_track(y=seg_audio, sr=sr)
            local_tempo = float(local_tempo_arr[0]) if hasattr(local_tempo_arr, "__len__") and len(local_tempo_arr) > 0 else float(local_tempo_arr)
        except Exception:
            local_tempo = global_tempo

        mood = _classify_frame_mood(
            rms_db=seg_rms_db,
            centroid=seg_centroid_mean / (sr / 2),
            flatness=seg_flatness,
            harm_ratio=seg_harm_ratio,
            onset_env=seg_onset,
            zcr=seg_zcr,
            chroma_std=seg_chroma_std,
            contrast_mean=seg_contrast,
            d_mfcc_norm=seg_d_mfcc,
            tempo_bpm=local_tempo,
        )
        mood.tempo_bpm = local_tempo

        # Scene type
        has_speech  = seg_zcr > 0.04 and seg_harm_ratio < 0.6
        is_silence  = seg_rms_db < -40
        is_music    = seg_harm_ratio > 0.45 and not has_speech
        is_action   = seg_onset > 0.5 and seg_rms_db > -20

        if is_silence:
            scene_type = "silence"
        elif has_speech and is_music:
            scene_type = "dialogue"
        elif has_speech:
            scene_type = "dialogue"
        elif is_action:
            scene_type = "action"
        elif is_music:
            scene_type = "music_only"
        else:
            scene_type = "transition"

        segments.append(SceneSegment(
            start_sec=round(seg_start, 3),
            end_sec=round(seg_end, 3),
            scene_type=scene_type,
            mood=mood,
            has_speech=has_speech,
            rms_db=round(seg_rms_db, 2),
            spectral_centroid_hz=round(seg_centroid_mean, 1),
            tempo_bpm=round(local_tempo, 1),
        ))

    # ── 4. Scene boundaries ──────────────────────────────────────────────────
    boundaries: list[SceneBoundary] = []
    for idx, bf in enumerate(boundary_frames):
        t = bf / frames_per_sec
        before_seg = next((s for s in reversed(segments) if s.start_sec <= t), None)
        after_seg  = next((s for s in segments if s.start_sec >= t), None)
        boundaries.append(SceneBoundary(
            time_sec=round(t, 3),
            novelty_score=round(float(novelty[bf]), 4),
            context_before=before_seg.scene_type if before_seg else "unknown",
            context_after=after_seg.scene_type   if after_seg  else "unknown",
        ))

    # ── 5. Global statistics ─────────────────────────────────────────────────
    all_labels  = [s.mood.label for s in segments]
    total_dur   = sum(s.duration for s in segments)
    music_dur   = sum(s.duration for s in segments if s.scene_type == "music_only")
    speech_dur  = sum(s.duration for s in segments if s.has_speech)
    silence_dur = sum(s.duration for s in segments if s.scene_type == "silence")

    if all_labels:
        from collections import Counter
        dominant_mood = Counter(all_labels).most_common(1)[0][0]
    else:
        dominant_mood = "neutral"

    compute_ms = (time.perf_counter() - t0) * 1000.0

    return AudioSceneAnalysis(
        duration_sec=round(duration_sec, 2),
        segments=segments,
        boundaries=boundaries,
        scene_mood_timeline=scene_mood_timeline,
        dominant_mood=dominant_mood,
        music_fraction=round(music_dur / max(total_dur, 1e-3), 3),
        speech_fraction=round(speech_dur / max(total_dur, 1e-3), 3),
        silence_fraction=round(silence_dur / max(total_dur, 1e-3), 3),
        compute_ms=round(compute_ms, 1),
    )


# ─── Visualization ─────────────────────────────────────────────────────────────

MOOD_COLORS = {
    "tense":    "#E53935",
    "epic":     "#9C27B0",
    "sad":      "#1E88E5",
    "happy":    "#FDD835",
    "calm":     "#43A047",
    "neutral":  "#78909C",
    "dialogue": "#00ACC1",
    "silence":  "#ECEFF1",
}

SCENE_COLORS = {
    "action":     "#FF5722",
    "dialogue":   "#2196F3",
    "music_only": "#9C27B0",
    "transition": "#FF9800",
    "silence":    "#B0BEC5",
}


def plot_scene_analysis(
    audio: np.ndarray,
    sr: int,
    analysis: AudioSceneAnalysis,
    output_path: Path,
    title: str = "Scene Analysis",
) -> None:
    """Generate a 5-panel scene analysis figure."""
    if not _PLT:
        return

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(5, 1, hspace=0.55)

    t_audio = np.arange(len(audio)) / sr

    # ── Panel 1: Waveform ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_audio, audio, linewidth=0.3, color="steelblue", alpha=0.75)
    ax0.set_xlim([0, len(audio)/sr])
    ax0.set_ylabel("Amplitude")
    ax0.set_title(f"{title} — Waveform", fontsize=10, fontweight="bold")
    ax0.grid(alpha=0.2)
    # Mark scene boundaries
    for bnd in analysis.boundaries:
        ax0.axvline(bnd.time_sec, color="red", linewidth=0.8, alpha=0.5)

    # ── Panel 2: Mel spectrogram ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    try:
        import librosa.display
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr,
                                 hop_length=512, ax=ax1, cmap="magma")
        ax1.set_title("Mel Spectrogram (80 bands, STFT hop=512)", fontsize=9)
        ax1.set_xlim([0, len(audio)/sr])
    except Exception:
        ax1.text(0.5, 0.5, "Mel spectrogram unavailable", ha="center", va="center")

    # ── Panel 3: Scene type timeline ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    for seg in analysis.segments:
        color = SCENE_COLORS.get(seg.scene_type, "grey")
        ax2.barh(0, seg.duration, left=seg.start_sec, height=0.8,
                color=color, alpha=0.8, edgecolor="none")
        if seg.duration > 2.0:
            ax2.text(seg.start_sec + seg.duration/2, 0, seg.scene_type,
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
    patches = [plt.Rectangle((0,0),1,1, color=SCENE_COLORS[sc]) for sc in SCENE_COLORS]
    ax2.legend(patches, list(SCENE_COLORS.keys()), loc="upper right", fontsize=7,
               ncol=len(SCENE_COLORS)//2 + 1)
    ax2.set_title("Scene Type Timeline", fontsize=9)
    ax2.set_xlim([0, len(audio)/sr])
    ax2.set_yticks([])
    ax2.set_xlabel("Time (s)")

    # ── Panel 4: Mood timeline ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    for seg in analysis.segments:
        color = MOOD_COLORS.get(seg.mood.label, "grey")
        ax3.barh(0, seg.duration, left=seg.start_sec, height=0.8,
                color=color, alpha=0.85, edgecolor="none")
        if seg.duration > 1.5:
            ax3.text(seg.start_sec + seg.duration/2, 0,
                    seg.mood.label + seg.mood.emoji,
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
    patches_m = [plt.Rectangle((0,0),1,1, color=MOOD_COLORS[m]) for m in MOOD_COLORS if m != "silence"]
    ax3.legend(patches_m, [m for m in MOOD_COLORS if m != "silence"],
               loc="upper right", fontsize=7, ncol=4)
    ax3.set_title("Scene Mood Timeline (Valence × Energy Analysis)", fontsize=9)
    ax3.set_xlim([0, len(audio)/sr])
    ax3.set_yticks([])
    ax3.set_xlabel("Time (s)")

    # ── Panel 5: Energy + valence curves ──────────────────────────────────
    ax4 = fig.add_subplot(gs[4])
    timeline = analysis.scene_mood_timeline
    if timeline:
        t_vals    = [x["time_sec"] for x in timeline]
        energies  = [x["energy"]   for x in timeline]
        valences  = [x["valence"]  for x in timeline]
        smooth = lambda x, w=5: uniform_filter1d(x, size=w)
        ax4.plot(t_vals, smooth(energies),  color="tomato",   linewidth=1.5, label="Energy")
        ax4.plot(t_vals, smooth(valences),  color="steelblue", linewidth=1.5, label="Valence")
        ax4.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax4.set_xlim([0, len(audio)/sr])
        ax4.set_ylim([-1.05, 1.05])
        ax4.set_ylabel("Score")
        ax4.set_xlabel("Time (s)")
        ax4.set_title("Energy & Valence Curves (Smoothed)", fontsize=9)
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.2)
        # Shade tense/epic regions
        for seg in analysis.segments:
            if seg.mood.label in ("tense", "epic"):
                ax4.axvspan(seg.start_sec, seg.end_sec,
                           color=MOOD_COLORS[seg.mood.label], alpha=0.12)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def annotate_srt_with_scene(
    subtitle_cues,    # list of SubtitleCue-like objects (start, end, text)
    analysis: AudioSceneAnalysis,
) -> list[dict]:
    """
    Annotate subtitle cues with scene context:
    returns list of dicts with text + mood_tag + scene_type.
    """
    annotated = []
    for cue in subtitle_cues:
        # Find the segment containing this cue's midpoint
        mid = (cue.start + cue.end) / 2
        seg = next(
            (s for s in analysis.segments if s.start_sec <= mid < s.end_sec),
            None
        )
        if seg:
            mood_tag   = seg.mood.srt_tag
            scene_type = seg.scene_type
        else:
            mood_tag   = ""
            scene_type = "unknown"
        annotated.append({
            "start":      cue.start,
            "end":        cue.end,
            "text":       cue.text,
            "mood_tag":   mood_tag,
            "scene_type": scene_type,
            "annotated_text": f"{mood_tag} {cue.text}".strip() if mood_tag else cue.text,
        })
    return annotated
