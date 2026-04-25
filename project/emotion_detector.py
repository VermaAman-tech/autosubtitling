"""
Acoustic emotion detection using opensmile eGeMAPS v02 features.

Strategy (no training required):
- Extract 88-dim eGeMAPSv02 functional features per speech segment
- Apply interpretable decision rules derived from speech emotion research:
    * F0 statistics  →  pitch/excitement indicator
    * Loudness       →  energy/arousal indicator
    * HNR (jitter)   →  voice quality / stress indicator
    * Speaking rate  →  tempo-based affect
    * Spectral flux  →  articulatory precision
- Output one of: neutral, excited, tense, sad, angry, fearful
- Also output valence (positive/negative) and arousal (high/low)

Reference: Schuller et al., "openSMILE — the Munich versatile and fast
open-source audio feature extractor", ACM Multimedia 2010.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    import opensmile
    _SMILE_AVAILABLE = True
except ImportError:
    _SMILE_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


EMOTIONS = ["neutral", "excited", "tense", "sad", "angry", "fearful"]

EMOTION_EMOJI = {
    "neutral":  "😐",
    "excited":  "🤩",
    "tense":    "😬",
    "sad":      "😢",
    "angry":    "😠",
    "fearful":  "😨",
}


@dataclass
class EmotionResult:
    label: str                      # primary emotion label
    valence: float                  # -1 (negative) .. +1 (positive)
    arousal: float                  # 0 (calm) .. 1 (excited)
    confidence: float               # 0..1 heuristic confidence
    f0_mean_st: float               # F0 in semitones
    f0_std_st: float
    loudness_mean: float
    hnr_proxy: float                # harmonic quality proxy
    speech_rate_proxy: float        # segments-per-second proxy
    compute_ms: float

    @property
    def emoji(self) -> str:
        return EMOTION_EMOJI.get(self.label, "")

    @property
    def srt_tag(self) -> str:
        """Short tag suitable for SRT annotation."""
        return f"[{self.label.upper()}]"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_egemaps(audio: np.ndarray, sr: int) -> Optional[dict]:
    """Extract eGeMAPS v02 functionals. Returns feature dict or None."""
    if not _SMILE_AVAILABLE:
        return None
    if len(audio) < sr * 0.1:   # too short
        return None
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, sr, subtype="PCM_16")
        feats_df = smile.process_file(tmp_path)
        os.unlink(tmp_path)
        return feats_df.iloc[0].to_dict()
    except Exception:
        return None


def _safe(feat_dict: dict, key: str, default: float = 0.0) -> float:
    v = feat_dict.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return float(v)


# ---------------------------------------------------------------------------
# Heuristic emotion classifier
# ---------------------------------------------------------------------------

def classify_emotion(audio: np.ndarray, sr: int,
                     word_count: int = 0,
                     duration_sec: float = 0.0) -> EmotionResult:
    """
    Classify emotion from raw audio using eGeMAPS features + heuristics.

    Parameters
    ----------
    audio        : mono float32 waveform
    sr           : sample rate
    word_count   : number of words transcribed (for speech rate proxy)
    duration_sec : segment duration in seconds

    Returns
    -------
    EmotionResult
    """
    t0 = time.perf_counter()
    feats = _extract_egemaps(audio, sr)

    if feats is None:
        # Fallback: librosa-based minimal features
        return _librosa_fallback(audio, sr, word_count, duration_sec, t0)

    # ── Core features ──────────────────────────────────────────────────────
    f0_mean  = _safe(feats, "F0semitoneFrom27.5Hz_sma3nz_amean")
    f0_std   = _safe(feats, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm")
    f0_rise  = _safe(feats, "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope")

    loud_mean = _safe(feats, "loudness_sma3_amean")
    loud_std  = _safe(feats, "loudness_sma3_stddevNorm")

    # HNR proxy: harmonicDifference measures harmonic structure
    hnr_proxy = _safe(feats, "HNRdBACF_sma3nz_amean", default=10.0)

    # Spectral features
    alpha_ratio = _safe(feats, "alphaRatioV_sma3nz_amean")   # high freq energy ratio
    hammar_idx  = _safe(feats, "hammarbergIndexV_sma3nz_amean")

    # Jitter/shimmer proxy (voice quality)
    jitter  = _safe(feats, "jitterLocal_sma3nz_amean")
    shimmer = _safe(feats, "shimmerLocaldB_sma3nz_amean")
    voice_stress = jitter + shimmer * 0.1

    # Speech rate proxy
    if duration_sec > 0 and word_count > 0:
        speech_rate = word_count / duration_sec   # words per second
    else:
        speech_rate = 2.5  # neutral default

    # ── Derived dimensions ─────────────────────────────────────────────────
    # Arousal: driven by loudness, F0, speech rate
    # Normalise loudness to ~0..1 range (typical 0..3)
    loud_norm  = min(loud_mean / 2.5, 1.0)
    f0_norm    = min(max((f0_mean - 18) / 25, 0.0), 1.0)  # typical 15..45 semitones
    rate_norm  = min(speech_rate / 5.0, 1.0)
    arousal    = 0.4 * loud_norm + 0.35 * f0_norm + 0.25 * rate_norm

    # Valence: driven by F0 contour (rising→positive), harmonic quality, alpha
    f0_rise_norm  = min(max(f0_rise / 3.0, 0.0), 1.0)
    hnr_norm      = min(max((hnr_proxy - 5) / 20, 0.0), 1.0)
    alpha_neg     = min(max(-alpha_ratio / 10, 0.0), 1.0)  # lower alpha → more positive
    valence       = 0.35 * f0_rise_norm + 0.35 * hnr_norm - 0.30 * alpha_neg
    valence       = max(-1.0, min(1.0, valence * 2 - 0.3))  # re-centre

    # ── Rule-based label assignment ────────────────────────────────────────
    # Based on Russell's circumplex model + speech-emotion literature
    if arousal > 0.65 and valence > 0.0:
        label = "excited"
        confidence = min(arousal * 0.9 + valence * 0.3, 1.0)

    elif arousal > 0.55 and valence < -0.1 and voice_stress > 0.08:
        label = "angry"
        confidence = min(arousal * 0.8 - valence * 0.3, 1.0)

    elif arousal > 0.45 and valence < -0.2 and f0_std > 0.5:
        label = "fearful"
        confidence = 0.6 + abs(valence) * 0.3

    elif arousal > 0.50 and valence < 0.0 and voice_stress > 0.05:
        label = "tense"
        confidence = 0.55 + arousal * 0.3

    elif arousal < 0.30 and valence < -0.05:
        label = "sad"
        confidence = 0.5 + (0.30 - arousal) * 0.8

    else:
        label = "neutral"
        confidence = 0.5 + max(0.0, 0.35 - abs(valence)) + max(0.0, 0.35 - arousal) * 0.5

    confidence = float(np.clip(confidence, 0.0, 1.0))
    elapsed = (time.perf_counter() - t0) * 1000.0

    return EmotionResult(
        label=label,
        valence=round(valence, 3),
        arousal=round(arousal, 3),
        confidence=round(confidence, 3),
        f0_mean_st=round(f0_mean, 2),
        f0_std_st=round(f0_std, 3),
        loudness_mean=round(loud_mean, 3),
        hnr_proxy=round(hnr_proxy, 2),
        speech_rate_proxy=round(speech_rate, 2),
        compute_ms=round(elapsed, 1),
    )


def _librosa_fallback(audio: np.ndarray, sr: int,
                      word_count: int, duration_sec: float,
                      t0: float) -> EmotionResult:
    """Minimal librosa-based fallback when opensmile is unavailable."""
    if not _LIBROSA_AVAILABLE:
        return EmotionResult(
            label="neutral", valence=0.0, arousal=0.3, confidence=0.3,
            f0_mean_st=0.0, f0_std_st=0.0, loudness_mean=0.0,
            hnr_proxy=0.0, speech_rate_proxy=2.5,
            compute_ms=(time.perf_counter() - t0) * 1000.0,
        )
    import librosa
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-12))
    arousal = min(rms * 6.0, 1.0)
    valence = 0.0
    label = "excited" if arousal > 0.6 else ("tense" if arousal > 0.4 else "neutral")
    return EmotionResult(
        label=label, valence=valence, arousal=round(arousal, 3),
        confidence=0.35, f0_mean_st=0.0, f0_std_st=0.0,
        loudness_mean=round(rms, 4), hnr_proxy=0.0,
        speech_rate_proxy=word_count / max(duration_sec, 0.01),
        compute_ms=round((time.perf_counter() - t0) * 1000.0, 1),
    )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def detect_emotions_batch(
    segments: list[tuple[float, float]],
    audio: np.ndarray,
    sr: int,
    word_counts: Optional[list[int]] = None,
) -> list[EmotionResult]:
    """
    Detect emotions for a list of (start, end) audio segments.

    Parameters
    ----------
    segments    : list of (start_sec, end_sec)
    audio       : full mono waveform
    sr          : sample rate
    word_counts : optional list of word counts per segment
    """
    results = []
    for i, (start, end) in enumerate(segments):
        seg = audio[int(start * sr): int(end * sr)]
        wc = word_counts[i] if word_counts else 0
        dur = end - start
        results.append(classify_emotion(seg, sr, word_count=wc, duration_sec=dur))
    return results
