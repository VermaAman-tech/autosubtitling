from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.signal import lfilter, wiener


TARGET_SR = 16_000
_FRAME_MS = 25
_HOP_MS = 10
_FRAME_LEN = int(TARGET_SR * _FRAME_MS / 1000)
_HOP_LEN = int(TARGET_SR * _HOP_MS / 1000)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    rms: float
    zcr: float
    spectral_flatness: float
    spectral_centroid: float
    spectral_flux: float
    mfcc_mean: float
    mfcc_std: float
    mfcc_delta_mean: float
    estimated_snr_db: float
    extract_time_ms: float = 0.0


@dataclass
class VADResult:
    segments: list[tuple[float, float]]   # (start_sec, end_sec)
    method: str
    compute_time_ms: float
    speech_ratio: float                    # fraction of audio classified as speech
    frame_labels: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak == 0:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def pad_or_trim(noise: np.ndarray, target_len: int) -> np.ndarray:
    if len(noise) >= target_len:
        return noise[:target_len]
    reps = int(math.ceil(target_len / max(1, len(noise))))
    return np.tile(noise, reps)[:target_len]


def synthesize_noise(kind: str, length: int, sr: int = TARGET_SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length) / sr

    if kind == "white":
        signal = rng.normal(0, 1, length)
    elif kind == "pink":
        white = rng.normal(0, 1, length)
        signal = lfilter(
            [0.049922, 0.095993, 0.050612, -0.004408],
            [1, -2.494956, 2.017265, -0.522189],
            white,
        )
    elif kind == "hum":
        signal = 0.8 * np.sin(2 * np.pi * 60 * t) + 0.4 * np.sin(2 * np.pi * 120 * t)
        signal += 0.15 * rng.normal(0, 1, length)
    elif kind == "babble":
        # Simulated multi-speaker babble: sum of shifted + pitched copies
        signal = np.zeros(length, dtype=np.float32)
        for shift in [0, 800, 1600, 3200]:
            rolled = np.roll(rng.normal(0, 1, length), shift)
            signal += 0.25 * rolled.astype(np.float32)
    elif kind == "soundtrack":
        signal = np.zeros(length, dtype=np.float32)
        chord_sets = [
            (220.0, 277.18, 329.63),
            (196.0, 246.94, 293.66),
            (174.61, 220.0, 261.63),
        ]
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 0.3 * t))
        percussive = rng.normal(0, 0.15, length) * (np.sin(2 * np.pi * 3.0 * t) > 0.95)
        for idx, freqs in enumerate(chord_sets):
            block = len(t) // len(chord_sets)
            start = idx * block
            end = length if idx == len(chord_sets) - 1 else (idx + 1) * block
            local_t = t[: end - start]
            chord = sum(np.sin(2 * np.pi * freq * local_t) for freq in freqs) / len(freqs)
            signal[start:end] = chord
        signal = signal * envelope + percussive + 0.03 * rng.normal(0, 1, length)
    else:
        raise ValueError(f"Unsupported noise kind: {kind}")

    return normalize_audio(signal.astype(np.float32))


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean = clean.astype(np.float32)
    noise = pad_or_trim(noise.astype(np.float32), len(clean))
    clean_power = float(np.mean(clean ** 2) + 1e-12)
    noise_power = float(np.mean(noise ** 2) + 1e-12)
    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = math.sqrt(desired_noise_power / noise_power)
    mixed = clean + scale * noise
    return normalize_audio(mixed)


# ---------------------------------------------------------------------------
# Enhancement methods
# ---------------------------------------------------------------------------

def enhance_wiener(audio: np.ndarray) -> np.ndarray:
    """Wiener filter (scipy implementation)."""
    filtered = wiener(audio.astype(np.float32), mysize=29)
    return normalize_audio(filtered)


def enhance_spectral_subtraction(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_fft: int = 512,
    hop_length: int = 160,
    noise_frames: int = 6,
    over_subtraction: float = 1.5,
) -> np.ndarray:
    """
    Classical spectral subtraction using a noise estimate from the first
    few frames (assumed to be silence/noise-only).
    """
    audio = audio.astype(np.float32)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # Noise estimate from leading frames
    noise_est = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

    # Subtract and half-wave rectify
    mag_clean = mag - over_subtraction * noise_est
    mag_clean = np.maximum(mag_clean, 0.05 * mag)  # spectral floor

    cleaned = librosa.istft(mag_clean * np.exp(1j * phase), hop_length=hop_length, length=len(audio))
    return normalize_audio(cleaned.astype(np.float32))


def enhance_audio(audio: np.ndarray, method: str = "wiener") -> np.ndarray:
    if method == "wiener":
        return enhance_wiener(audio)
    if method == "spectral_subtraction":
        return enhance_spectral_subtraction(audio)
    raise ValueError(f"Unknown enhancement method: {method}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(audio: np.ndarray, sr: int = TARGET_SR) -> AudioFeatures:
    t0 = time.perf_counter()
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=400, hop_length=160)))
    spectral_flatness = float(
        np.mean(librosa.feature.spectral_flatness(y=audio, n_fft=512, hop_length=160))
    )
    spectral_centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=512, hop_length=160))
    )

    # Spectral flux: mean frame-to-frame magnitude change
    mag = np.abs(librosa.stft(audio, n_fft=512, hop_length=160))
    flux = float(np.mean(np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0) + 1e-12)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_mean = float(np.mean(mfcc))
    mfcc_std = float(np.std(mfcc))
    mfcc_delta_mean = float(np.mean(np.abs(mfcc_delta)))

    signal_floor = np.percentile(np.abs(audio), 20)
    signal_peak = np.percentile(np.abs(audio), 95) + 1e-6
    estimated_snr_db = float(20 * np.log10((signal_peak + 1e-6) / (signal_floor + 1e-6)))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AudioFeatures(
        rms=rms,
        zcr=zcr,
        spectral_flatness=spectral_flatness,
        spectral_centroid=spectral_centroid,
        spectral_flux=flux,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        mfcc_delta_mean=mfcc_delta_mean,
        estimated_snr_db=estimated_snr_db,
        extract_time_ms=elapsed_ms,
    )


def should_enhance(features: AudioFeatures) -> bool:
    """Adaptive routing rule: combine multiple indicators."""
    return (
        features.estimated_snr_db < 18.0
        or features.spectral_flatness > 0.18
        or features.zcr > 0.12
    )


# ---------------------------------------------------------------------------
# VAD methods
# ---------------------------------------------------------------------------

def _frames_to_segments(
    labels: list[int],
    hop_len: int,
    sr: int,
    min_dur: float = 0.25,
    merge_gap: float = 0.40,
) -> list[tuple[float, float]]:
    """Convert per-frame speech labels to (start, end) segments."""
    segments: list[tuple[float, float]] = []
    start_idx = None
    for idx, label in enumerate(labels):
        if label == 1 and start_idx is None:
            start_idx = idx
        elif label == 0 and start_idx is not None:
            segments.append((start_idx * hop_len / sr, idx * hop_len / sr))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx * hop_len / sr, len(labels) * hop_len / sr))

    # merge close gaps
    merged: list[tuple[float, float]] = []
    for seg in segments:
        if seg[1] - seg[0] < min_dur:
            continue
        if merged and seg[0] - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return merged


def vad_energy(audio: np.ndarray, sr: int = TARGET_SR) -> VADResult:
    """
    Baseline VAD: log-energy threshold only.
    This is the classical, textbook approach.
    """
    t0 = time.perf_counter()
    frame_len = _FRAME_LEN
    hop_len = _HOP_LEN
    energies = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len].astype(np.float32)
        energies.append(float(np.mean(frame ** 2)))
    if not energies:
        return VADResult([], "energy", 0.0, 0.0)

    # Adaptive threshold: percentile-based
    threshold = max(np.percentile(energies, 40) * 2.0, 1e-7)
    labels = [1 if e > threshold else 0 for e in energies]
    segments = _frames_to_segments(labels, hop_len, sr)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    speech_ratio = sum(1 for lb in labels if lb == 1) / max(len(labels), 1)
    return VADResult(
        segments=segments,
        method="energy",
        compute_time_ms=elapsed_ms,
        speech_ratio=speech_ratio,
        frame_labels=labels,
    )


def vad_mfcc(audio: np.ndarray, sr: int = TARGET_SR) -> VADResult:
    """
    Improved VAD: MFCC C0 (log-energy) + ZCR combined.
    Closer to the GMM-based VAD discussed in class.
    """
    t0 = time.perf_counter()
    frame_len = _FRAME_LEN
    hop_len = _HOP_LEN

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        hop_length=hop_len,
    )
    zcr_frames = librosa.feature.zero_crossing_rate(
        y=audio.astype(np.float32), frame_length=frame_len, hop_length=hop_len
    )[0]

    # C0 (energy) + ZCR combined score
    c0 = mfcc[0]  # log-energy MFCC coefficient
    c0_norm = (c0 - np.mean(c0)) / (np.std(c0) + 1e-8)
    zcr_norm = (zcr_frames - np.mean(zcr_frames)) / (np.std(zcr_frames) + 1e-8)

    # Speech: high energy AND moderate ZCR (not silence, not pure noise)
    combined = 0.7 * c0_norm + 0.3 * (1.0 - zcr_norm)
    threshold = np.percentile(combined, 35)
    labels = [1 if v > threshold else 0 for v in combined]

    segments = _frames_to_segments(labels, hop_len, sr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    speech_ratio = sum(1 for lb in labels if lb == 1) / max(len(labels), 1)
    return VADResult(
        segments=segments,
        method="mfcc",
        compute_time_ms=elapsed_ms,
        speech_ratio=speech_ratio,
        frame_labels=labels,
    )


def vad_spectral(audio: np.ndarray, sr: int = TARGET_SR) -> VADResult:
    """
    Full-feature VAD: MFCC + spectral flatness + spectral centroid.
    The proposed adaptive method — uses the richest feature set.
    """
    t0 = time.perf_counter()
    hop_len = _HOP_LEN
    frame_len = _FRAME_LEN

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32), sr=sr, n_mfcc=13, n_fft=512, hop_length=hop_len
    )
    flatness = librosa.feature.spectral_flatness(
        y=audio.astype(np.float32), n_fft=512, hop_length=hop_len
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio.astype(np.float32), sr=sr, n_fft=512, hop_length=hop_len
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=audio.astype(np.float32), frame_length=frame_len, hop_length=hop_len
    )[0]

    n = min(len(flatness), len(centroid), len(zcr), mfcc.shape[1])
    c0 = mfcc[0, :n]
    flat = flatness[:n]
    cent = centroid[:n]
    zc = zcr[:n]

    def _norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

    # Speech: high log-energy, high centroid, low flatness, moderate ZCR
    score = (
        0.45 * _norm(c0)
        + 0.25 * _norm(cent)
        - 0.20 * _norm(flat)
        + 0.10 * (1.0 - _norm(zc))
    )
    threshold = np.percentile(score, 30)
    labels = [1 if v > threshold else 0 for v in score]

    segments = _frames_to_segments(labels, hop_len, sr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    speech_ratio = sum(1 for lb in labels if lb == 1) / max(len(labels), 1)
    return VADResult(
        segments=segments,
        method="spectral",
        compute_time_ms=elapsed_ms,
        speech_ratio=speech_ratio,
        frame_labels=labels,
    )


VAD_METHODS = {
    "energy": vad_energy,
    "mfcc": vad_mfcc,
    "spectral": vad_spectral,
}


# ---------------------------------------------------------------------------
# Legacy detect_speech_regions (kept for pipeline.py compat)
# ---------------------------------------------------------------------------

def detect_speech_regions(audio: np.ndarray, sr: int = TARGET_SR, frame_ms: int = 30) -> list[tuple[float, float]]:
    return vad_spectral(audio, sr).segments
