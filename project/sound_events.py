"""
Sound event detection for movie trailers.

Detects the following events WITHOUT any neural network training:
  - LAUGHTER   : characteristic ~4–8 Hz amplitude modulation + harmonic voice quality
  - MUSIC      : strong harmonic structure, wide spectral bandwidth, low ZCR variance
  - SILENCE    : RMS below noise floor threshold
  - IMPACT     : rapid onset + low-frequency burst (action / explosion / hit)
  - APPLAUSE   : dense, noise-like texture at mid-to-high frequencies
  - SPEECH     : voiced, formant-structured audio (catch-all for labelled speech)

Also provides:
  - Scene boundary detection — major changes in audio character signal a cut
  - Timeline annotation — builds a list of (start, end, label) event spans

All methods use librosa-based hand-crafted features (research-validated).

References:
  - Cai et al., "Highlight Sound Effects Detection in Audio Stream", ICME 2003.
  - Xu et al., "Audio classification using a large scale ontology of everyday sound",
    DCASE 2017.
  - Knox & Mirza, "Laughter detection in broadcast audio", Interspeech 2019.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ── Event label constants ──────────────────────────────────────────────────
class EventLabel:
    SPEECH    = "SPEECH"
    LAUGHTER  = "LAUGHTER"
    MUSIC     = "MUSIC"
    SILENCE   = "SILENCE"
    IMPACT    = "IMPACT"
    APPLAUSE  = "APPLAUSE"
    AMBIENT   = "AMBIENT"


EVENT_EMOJI = {
    EventLabel.SPEECH:   "🗣️",
    EventLabel.LAUGHTER: "😂",
    EventLabel.MUSIC:    "🎵",
    EventLabel.SILENCE:  "🔇",
    EventLabel.IMPACT:   "💥",
    EventLabel.APPLAUSE: "👏",
    EventLabel.AMBIENT:  "🌊",
}


@dataclass
class SoundEvent:
    start: float
    end: float
    label: str
    confidence: float
    sub_label: Optional[str] = None   # e.g. "heroic" for MUSIC

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def emoji(self) -> str:
        return EVENT_EMOJI.get(self.label, "")

    def __repr__(self) -> str:
        return (f"SoundEvent({self.label} {self.emoji} "
                f"{self.start:.2f}s–{self.end:.2f}s  conf={self.confidence:.2f})")


@dataclass
class SceneBoundary:
    time: float            # seconds
    confidence: float
    description: str = ""  # what changed at this boundary


@dataclass
class AudioTimeline:
    events: list[SoundEvent] = field(default_factory=list)
    boundaries: list[SceneBoundary] = field(default_factory=list)
    compute_ms: float = 0.0


# ---------------------------------------------------------------------------
# Low-level feature helpers
# ---------------------------------------------------------------------------

def _amp_modulation_rate(audio: np.ndarray, sr: int,
                          frame_len: int = 512, hop: int = 128) -> tuple[float, float]:
    """
    Estimate the dominant amplitude-modulation rate in Hz.
    Laughter typically has AM rate in 4–8 Hz.
    Returns (dominant_rate_hz, modulation_depth).
    """
    # 1. Compute short-time RMS envelope
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop)[0]
    if len(rms) < 8:
        return 0.0, 0.0

    # 2. Normalise and remove DC
    rms = rms.astype(np.float64)
    rms = rms - rms.mean()
    if rms.std() < 1e-8:
        return 0.0, 0.0
    rms /= rms.std()

    # 3. FFT of envelope to find dominant modulation rate
    envelope_sr = sr / hop   # frames per second
    n = len(rms)
    spectrum = np.abs(np.fft.rfft(rms, n=max(n, 256)))
    freqs    = np.fft.rfftfreq(max(n, 256), d=1.0 / envelope_sr)

    # Focus on 2–12 Hz range
    mask = (freqs >= 2.0) & (freqs <= 12.0)
    if not np.any(mask):
        return 0.0, 0.0

    sub_spectrum = spectrum[mask]
    sub_freqs    = freqs[mask]
    peak_idx     = int(np.argmax(sub_spectrum))
    dom_rate     = float(sub_freqs[peak_idx])
    mod_depth    = float(sub_spectrum[peak_idx]) / (np.mean(spectrum) + 1e-8)

    return dom_rate, mod_depth


def _harmonic_ratio(audio: np.ndarray, sr: int, n_fft: int = 1024) -> float:
    """Ratio of harmonic to total energy (harmonic mean across frames)."""
    harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)
    h_energy = float(np.mean(harmonic ** 2) + 1e-12)
    p_energy = float(np.mean(percussive ** 2) + 1e-12)
    return h_energy / (h_energy + p_energy)


def _spectral_bandwidth_std(audio: np.ndarray, sr: int) -> float:
    """Std of spectral bandwidth over time — high for music, low for speech."""
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=256)[0]
    return float(np.std(bw))


def _onset_strength_peak(audio: np.ndarray, sr: int) -> float:
    """Max onset strength — peaks sharply for impacts."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=256)
    return float(np.max(onset_env))


def _low_freq_energy_ratio(audio: np.ndarray, sr: int, cutoff: int = 300) -> float:
    """Fraction of energy below cutoff Hz — high for bass-heavy impacts."""
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs < cutoff
    low_e  = float(np.mean(stft[low_mask, :] ** 2))
    tot_e  = float(np.mean(stft ** 2) + 1e-12)
    return low_e / tot_e


def _spectral_flatness_mean(audio: np.ndarray) -> float:
    flat = librosa.feature.spectral_flatness(y=audio, hop_length=256)[0]
    return float(np.mean(flat))


def _zcr_mean(audio: np.ndarray) -> float:
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=256)[0]
    return float(np.mean(zcr))


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    return float(20 * np.log10(rms + 1e-12))


# ---------------------------------------------------------------------------
# Per-segment classifier
# ---------------------------------------------------------------------------

def classify_segment(audio: np.ndarray, sr: int) -> SoundEvent:
    """
    Classify a short audio segment into a sound event label.
    Uses a cascade of rule-based checks.
    """
    if not _LIBROSA_AVAILABLE:
        return SoundEvent(0.0, len(audio) / sr, EventLabel.AMBIENT, 0.5)

    audio = audio.astype(np.float32)
    dur = len(audio) / sr

    # ── 1. Silence check ──────────────────────────────────────────────────
    rms_db = _rms_db(audio)
    if rms_db < -45.0:
        return SoundEvent(0.0, dur, EventLabel.SILENCE, 0.95)

    # ── 2. Extract features ───────────────────────────────────────────────
    flat       = _spectral_flatness_mean(audio)
    zcr        = _zcr_mean(audio)
    harm_ratio = _harmonic_ratio(audio, sr)
    onset_peak = _onset_strength_peak(audio, sr)
    low_e_rat  = _low_freq_energy_ratio(audio, sr)
    bw_std     = _spectral_bandwidth_std(audio, sr)
    am_rate, am_depth = _amp_modulation_rate(audio, sr)

    # ── 3. Impact detection ───────────────────────────────────────────────
    # Sharp onset + low-frequency burst
    if onset_peak > 18.0 and low_e_rat > 0.30 and flat > 0.05:
        conf = min(0.5 + (onset_peak - 18) * 0.02 + low_e_rat * 0.5, 0.95)
        return SoundEvent(0.0, dur, EventLabel.IMPACT, round(conf, 3))

    # ── 4. Music detection (BEFORE laughter — prevents action score false-positives)
    # Strong harmonic content + wide spectral bandwidth variation + low flatness.
    # Also catches "intense/action" music that might otherwise fool laughter detector:
    #   - lower bw_std threshold (150) to catch rhythmic action cues
    #   - OR onset_peak > 8 + low zcr (heavily percussive scored music)
    is_intense_music = (onset_peak > 8.0 and zcr < 0.15 and harm_ratio > 0.45
                        and low_e_rat > 0.15)
    if (harm_ratio > 0.60 and bw_std > 150 and flat < 0.12 and zcr < 0.15) or is_intense_music:
        sub = "intense" if onset_peak > 10 else ("heroic" if low_e_rat > 0.20 else "ambient")
        conf = min(0.5 + harm_ratio * 0.4 + (1.0 - flat) * 0.2, 0.92)
        return SoundEvent(0.0, dur, EventLabel.MUSIC, round(conf, 3), sub_label=sub)

    # ── 5. Laughter detection ─────────────────────────────────────────────
    # Voiced (high harm_ratio) + AM rate 4–8 Hz + strong modulation depth.
    # Music is already handled above; these guards remain for safety:
    #   - am_depth > 4.5 — requires substantial, not just incidental modulation
    #   - zcr < 0.10    — laughter is voiced, low ZCR
    is_likely_music = bw_std > 300 or onset_peak > 10.0
    if (not is_likely_music
            and harm_ratio > 0.45
            and 3.5 <= am_rate <= 9.0
            and am_depth > 4.5
            and zcr < 0.10):
        conf = min(0.5 + (am_depth - 2.5) * 0.05 + harm_ratio * 0.3, 0.92)
        return SoundEvent(0.0, dur, EventLabel.LAUGHTER, round(conf, 3))

    # ── 5. Applause detection ─────────────────────────────────────────────
    # Dense noise-like, flat spectrum, high ZCR, many rapid onsets
    if flat > 0.15 and zcr > 0.20 and onset_peak > 8.0 and harm_ratio < 0.35:
        conf = min(0.5 + flat * 1.5 + zcr * 0.8, 0.90)
        return SoundEvent(0.0, dur, EventLabel.APPLAUSE, round(conf, 3))

    # ── 7. Speech detection ───────────────────────────────────────────────
    # Moderate ZCR, moderate harmonic ratio, higher centroid
    if 0.04 < zcr < 0.25 and harm_ratio > 0.30 and flat < 0.20:
        conf = min(0.5 + harm_ratio * 0.3 + (0.20 - flat), 0.88)
        return SoundEvent(0.0, dur, EventLabel.SPEECH, round(conf, 3))

    return SoundEvent(0.0, dur, EventLabel.AMBIENT, 0.40)


# ---------------------------------------------------------------------------
# Scene boundary detection
# ---------------------------------------------------------------------------

def detect_scene_boundaries(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 1.0,
    hop_sec: float = 0.25,
    threshold: float = 0.55,
) -> list[SceneBoundary]:
    """
    Detect audio scene boundaries using self-similarity matrix novelty score.

    Based on: Foote, "Automatic audio segmentation using a measure of audio
    novelty", ICME 2000. Extended with multi-feature contrast.

    Parameters
    ----------
    audio        : full mono waveform
    sr           : sample rate
    window_sec   : analysis window size
    hop_sec      : hop between windows
    threshold    : normalised novelty score threshold
    """
    if not _LIBROSA_AVAILABLE:
        return []

    audio = audio.astype(np.float32)
    win_frames = int(window_sec * sr)
    hop_frames = int(hop_sec * sr)

    # ── Build feature matrix: Chroma + MFCCs + spectral contrast ──────────
    n_fft = 1024
    hop   = 256

    mfcc     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20,
                                     n_fft=n_fft, hop_length=hop)         # (20, T)
    chroma   = librosa.feature.chroma_stft(y=audio, sr=sr,
                                            n_fft=n_fft, hop_length=hop)  # (12, T)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                                                  n_fft=n_fft,
                                                  hop_length=hop)          # (7, T)

    # Stack and normalise each row
    feat = np.vstack([mfcc, chroma, contrast])                             # (39, T)
    feat = (feat - feat.mean(axis=1, keepdims=True)) / (
        feat.std(axis=1, keepdims=True) + 1e-8)

    T = feat.shape[1]
    frame_hop = max(1, int(hop_sec * sr / hop))
    frame_win = max(2, int(window_sec * sr / hop))

    # ── Compute cosine similarity between consecutive windows ──────────────
    novelty_times = []
    novelty_scores = []
    for t in range(frame_win, T - frame_win, frame_hop):
        before = feat[:, max(0, t - frame_win): t]
        after  = feat[:, t: min(T, t + frame_win)]
        if before.shape[1] == 0 or after.shape[1] == 0:
            continue
        mu_b = before.mean(axis=1)
        mu_a = after.mean(axis=1)
        cos_sim = float(
            np.dot(mu_b, mu_a) /
            (np.linalg.norm(mu_b) * np.linalg.norm(mu_a) + 1e-8)
        )
        novelty_scores.append(1.0 - cos_sim)   # dissimilarity = novelty
        novelty_times.append(t * hop / sr)

    if not novelty_scores:
        return []

    scores = np.array(novelty_scores)
    # Normalise to 0–1
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Peak picking
    from scipy.signal import argrelextrema
    peaks = argrelextrema(scores, np.greater, order=3)[0]
    boundaries = []
    for pk in peaks:
        if scores[pk] >= threshold:
            t_sec = novelty_times[pk]
            conf  = round(float(scores[pk]), 3)
            desc  = _describe_boundary(audio, sr, t_sec, window_sec)
            boundaries.append(SceneBoundary(time=round(t_sec, 2),
                                             confidence=conf,
                                             description=desc))
    return boundaries


def _describe_boundary(audio: np.ndarray, sr: int,
                        t_sec: float, window: float = 1.0) -> str:
    """Produce a short text description of what changed at a boundary."""
    half = int(window * sr / 2)
    center = int(t_sec * sr)
    before = audio[max(0, center - half): center]
    after  = audio[center: min(len(audio), center + half)]
    if len(before) < 100 or len(after) < 100:
        return "scene change"

    rms_b  = float(np.sqrt(np.mean(before ** 2) + 1e-12))
    rms_a  = float(np.sqrt(np.mean(after ** 2) + 1e-12))
    flat_b = float(np.mean(librosa.feature.spectral_flatness(y=before, hop_length=128)))
    flat_a = float(np.mean(librosa.feature.spectral_flatness(y=after,  hop_length=128)))

    energy_change = rms_a / (rms_b + 1e-8)
    if energy_change > 1.8:
        return "quiet → action"
    if energy_change < 0.55:
        return "action → quiet"
    if flat_b < 0.08 and flat_a > 0.15:
        return "music → noise"
    if flat_b > 0.12 and flat_a < 0.08:
        return "noise → music/speech"
    return "audio character change"


# ---------------------------------------------------------------------------
# Full timeline builder
# ---------------------------------------------------------------------------

def build_audio_timeline(
    audio: np.ndarray,
    sr: int,
    chunk_sec: float = 1.5,
    hop_sec: float = 0.75,
    min_event_sec: float = 0.4,
) -> AudioTimeline:
    """
    Build a complete audio event timeline for the full waveform.

    Returns AudioTimeline with events and scene boundaries.
    """
    t0 = time.perf_counter()
    if not _LIBROSA_AVAILABLE:
        return AudioTimeline(compute_ms=0.0)

    total_dur = len(audio) / sr
    chunk_len = int(chunk_sec * sr)
    hop_len   = int(hop_sec * sr)

    raw_events: list[SoundEvent] = []

    # Classify each chunk
    pos = 0
    while pos < len(audio):
        chunk = audio[pos: pos + chunk_len]
        if len(chunk) < int(0.1 * sr):
            break
        ev = classify_segment(chunk, sr)
        start_sec = pos / sr
        end_sec   = min(start_sec + len(chunk) / sr, total_dur)
        raw_events.append(SoundEvent(
            start=round(start_sec, 3),
            end=round(end_sec, 3),
            label=ev.label,
            confidence=ev.confidence,
            sub_label=ev.sub_label,
        ))
        pos += hop_len

    # Merge consecutive same-label events
    merged = _merge_events(raw_events, min_event_sec)

    # Scene boundaries
    boundaries = detect_scene_boundaries(audio, sr)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AudioTimeline(events=merged, boundaries=boundaries,
                         compute_ms=round(elapsed_ms, 1))


def _merge_events(
    events: list[SoundEvent],
    min_dur: float,
) -> list[SoundEvent]:
    """Merge consecutive same-label events and filter short ones."""
    if not events:
        return []
    merged: list[SoundEvent] = []
    cur = SoundEvent(events[0].start, events[0].end,
                     events[0].label, events[0].confidence, events[0].sub_label)
    for ev in events[1:]:
        if ev.label == cur.label:
            cur = SoundEvent(cur.start, ev.end, cur.label,
                             max(cur.confidence, ev.confidence), cur.sub_label)
        else:
            if cur.duration >= min_dur:
                merged.append(cur)
            cur = SoundEvent(ev.start, ev.end, ev.label, ev.confidence, ev.sub_label)
    if cur.duration >= min_dur:
        merged.append(cur)
    return merged


def events_at_time(timeline: AudioTimeline, t: float) -> list[SoundEvent]:
    """Return all events active at time t."""
    return [ev for ev in timeline.events if ev.start <= t <= ev.end]


def dominant_event_at_time(timeline: AudioTimeline, t: float) -> Optional[SoundEvent]:
    """Return the highest-confidence event active at time t."""
    active = events_at_time(timeline, t)
    if not active:
        return None
    return max(active, key=lambda e: e.confidence)
