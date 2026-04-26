"""
Extra evaluation metrics and novel experiments:
  - chrF score (character n-gram F-score) — more robust than WER for ASR
  - BLEU-1/2/4 (n-gram precision)
  - Genre classifier from audio features
  - Speaking rate timeline
"""
from __future__ import annotations

import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import sacrebleu
    _SACRE = True
except ImportError:
    _SACRE = False

TARGET_SR = 16_000


# ─── chrF score ───────────────────────────────────────────────────────────────

def compute_chrf(reference: str, hypothesis: str, char_order: int = 6) -> float:
    """
    Character n-gram F-score (Popović 2015).
    More robust than WER: handles morphological variation, punctuation.
    Range: 0 (no overlap) → 100 (perfect).
    """
    if _SACRE:
        try:
            metric = sacrebleu.CHRF(char_order=char_order)
            score = metric.corpus_score([hypothesis], [[reference]])
            return float(score.score)
        except Exception:
            pass
    # Fallback: manual character bigram F-score
    return _chrf_manual(reference, hypothesis, char_order)


def _chrf_manual(ref: str, hyp: str, n: int = 2) -> float:
    def ngrams(text: str, n: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in range(len(text) - n + 1):
            g = text[i:i+n]
            counts[g] = counts.get(g, 0) + 1
        return counts

    ref_n = ngrams(ref.lower(), n)
    hyp_n = ngrams(hyp.lower(), n)
    if not ref_n or not hyp_n:
        return 0.0

    overlap = sum(min(ref_n.get(g, 0), hyp_n.get(g, 0)) for g in hyp_n)
    precision = overlap / sum(hyp_n.values())
    recall    = overlap / sum(ref_n.values())
    if precision + recall == 0:
        return 0.0
    f = 2 * precision * recall / (precision + recall)
    return round(f * 100, 2)


def compute_bleu(reference: str, hypothesis: str) -> dict[str, float]:
    """BLEU-1, BLEU-2, BLEU-4 scores."""
    if _SACRE:
        try:
            results = {}
            for n in [1, 2, 4]:
                metric = sacrebleu.BLEU(max_ngram_order=n, effective_order=True)
                score  = metric.corpus_score([hypothesis], [[reference]])
                results[f"bleu{n}"] = round(float(score.score), 2)
            return results
        except Exception:
            pass
    return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}


# ─── Genre classifier ─────────────────────────────────────────────────────────

GENRE_LABELS = ["action", "drama", "comedy", "horror", "scifi", "animation", "thriller", "biopic"]

_GENRE_COLOR = {
    "action":    "#E53935",
    "drama":     "#1E88E5",
    "comedy":    "#FDD835",
    "horror":    "#212121",
    "scifi":     "#9C27B0",
    "animation": "#43A047",
    "thriller":  "#FF6F00",
    "biopic":    "#795548",
}


def extract_genre_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    30-dimensional genre feature vector:
      - 13 mean MFCCs (timbral character)
      - 13 MFCC variances (dynamic variation)
      - Mean spectral centroid (brightness)
      - Mean spectral rolloff (bandwidth)
      - Mean ZCR (speech vs music)
      - Mean RMS (loudness)
    """
    audio = audio.astype(np.float32)
    hop   = 512
    n_fft = 2048

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop, n_fft=n_fft)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var  = np.var(mfcc, axis=1)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)))
    rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)))
    zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft//4, hop_length=hop)))
    rms      = float(np.sqrt(np.mean(audio**2) + 1e-12))

    return np.concatenate([mfcc_mean, mfcc_var, [centroid/8000, rolloff/8000, zcr*10, rms*5]])


def run_genre_classifier(
    trailer_pairs: list[tuple[Path, str]],   # [(wav_path, genre_label), ...]
    output_dir: Path,
) -> dict:
    """
    Train and evaluate a k-NN genre classifier (leave-one-out cross-validation).

    The goal is NOT state-of-the-art genre recognition, but to demonstrate
    that classical MFCC features contain discriminative genre information —
    a key course concept.
    """
    exp_dir = output_dir / "exp_genre_classifier"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if len(trailer_pairs) < 4:
        print("  [GC] Not enough trailers for genre classification (need ≥4)")
        return {}

    print(f"  [GC] Extracting features from {len(trailer_pairs)} trailers…")
    features = []
    labels   = []

    for wav_path, genre in trailer_pairs:
        try:
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            # Use first 60s only (trailers) to speed up
            audio = audio[:TARGET_SR * 60]
            feats = extract_genre_features(audio, TARGET_SR)
            features.append(feats)
            labels.append(genre)
        except Exception as exc:
            print(f"  [GC] Skip {wav_path.name}: {exc}")

    if len(features) < 4:
        return {}

    X = np.array(features)
    y = np.array(labels)

    # Normalize features
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X   = (X - mu) / std

    # Leave-one-out k-NN (k=3)
    correct = 0
    preds   = []
    for i in range(len(X)):
        train_X = np.delete(X, i, axis=0)
        train_y = np.delete(y, i, axis=0)
        test_x  = X[i]

        # Euclidean distance to all training samples
        dists = np.linalg.norm(train_X - test_x, axis=1)
        k     = min(3, len(dists))
        nn_idx = np.argsort(dists)[:k]
        nn_labels = train_y[nn_idx]

        # Majority vote
        from collections import Counter
        pred = Counter(nn_labels).most_common(1)[0][0]
        preds.append(pred)
        if pred == y[i]:
            correct += 1

    accuracy = correct / len(y)
    print(f"  [GC] LOO accuracy: {accuracy:.3f}  ({correct}/{len(y)})")

    # Per-class accuracy
    per_class: dict[str, float] = {}
    for genre in set(y):
        idx = np.where(y == genre)[0]
        cls_correct = sum(1 for i in idx if preds[i] == genre)
        per_class[genre] = round(cls_correct / len(idx), 3)

    # Save results
    results = {
        "n_trailers": len(y),
        "loo_accuracy": round(accuracy, 4),
        "per_class_accuracy": per_class,
        "labels": y.tolist(),
        "predictions": preds,
    }

    _plot_genre_results(X, y, preds, results, exp_dir)
    return results


def _plot_genre_results(X, y, preds, results, exp_dir: Path) -> None:
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── PCA scatter of feature space ─────────────────────────────────────────
    n_comp = min(2, X.shape[1], X.shape[0] - 1)
    if n_comp >= 2:
        pca = PCA(n_components=2)
        X2  = pca.fit_transform(X)
        genres_uniq = sorted(set(y))
        for g in genres_uniq:
            idx = np.where(y == g)[0]
            axes[0].scatter(X2[idx, 0], X2[idx, 1],
                           c=_GENRE_COLOR.get(g, "grey"), label=g, s=80, alpha=0.85, edgecolors="white")
        for i, (xi, yi) in enumerate(zip(X2[:, 0], X2[:, 1])):
            correct = (preds[i] == y[i])
            marker  = "o" if correct else "x"
            axes[0].scatter(xi, yi, marker=marker, c="none",
                           edgecolors="black" if correct else "red", s=100, linewidths=1.5)
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var.)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var.)")
        axes[0].set_title("Genre Feature Space (PCA, k-NN LOO)")
        axes[0].legend(fontsize=8, loc="best")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "Not enough data for PCA", ha="center")

    # ── Per-class accuracy bar chart ──────────────────────────────────────────
    pc = results.get("per_class_accuracy", {})
    if pc:
        genres = list(pc.keys())
        accs   = [pc[g] for g in genres]
        colors = [_GENRE_COLOR.get(g, "grey") for g in genres]
        axes[1].barh(genres, [a*100 for a in accs], color=colors, alpha=0.85)
        axes[1].axvline(results["loo_accuracy"]*100, color="red", linestyle="--",
                       label=f"Overall: {results['loo_accuracy']*100:.0f}%")
        axes[1].set_xlabel("Accuracy (%)")
        axes[1].set_title("Genre Classifier — Per-Class Accuracy (k-NN LOO)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, axis="x")
        for i, (acc, genre) in enumerate(zip(accs, genres)):
            axes[1].text(acc*100 + 1, i, f"{acc*100:.0f}%", va="center", fontsize=9)

    fig.suptitle(f"Genre Classification from MFCC Features (LOO Accuracy = {results['loo_accuracy']*100:.0f}%)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "fig_genre_classifier.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [GC] Saved → {exp_dir}/fig_genre_classifier.png")


# ─── Speaking rate timeline ────────────────────────────────────────────────────

def speaking_rate_timeline(
    subtitle_cues: list[dict],
    audio_duration: float,
    output_dir: Path,
    slug: str = "trailer",
) -> dict:
    """
    Compute speaking rate (words per minute) across the subtitle timeline.
    Reveals fast-paced action vs slow dramatic dialogue.
    """
    if not subtitle_cues:
        return {}

    wpm_vals    = []
    midpoints   = []
    segment_dur = []

    for cue in subtitle_cues:
        start = cue.get("start", 0)
        end   = cue.get("end", start + 1)
        text  = cue.get("text", "")
        dur   = max(end - start, 0.1)
        words = len(text.split())
        wpm   = words / dur * 60
        wpm_vals.append(wpm)
        midpoints.append((start + end) / 2)
        segment_dur.append(dur)

    stats = {
        "mean_wpm":   round(float(np.mean(wpm_vals)), 1),
        "median_wpm": round(float(np.median(wpm_vals)), 1),
        "max_wpm":    round(float(np.max(wpm_vals)), 1),
        "min_wpm":    round(float(np.min(wpm_vals)), 1),
        "std_wpm":    round(float(np.std(wpm_vals)), 1),
    }
    return stats
