"""
Novel Experiments for EE 679 Auto-Subtitling
=============================================
Three self-contained experiment modules that add depth beyond the baseline:

  1. Enhancement Quality (PESQ + STOI) — objective speech quality metrics
  2. Noise-Type Classifier — MFCC-based SVM-lite background noise identification
  3. Confidence Calibration — Whisper avg_logprob vs actual WER reliability

Each module returns a DataFrame/dict and saves figures to output_dir.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

from project.audio_utils import (
    TARGET_SR,
    enhance_audio,
    extract_features,
    mix_with_snr,
    should_enhance,
    synthesize_noise,
)
from project.metrics import compute_wer


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ENHANCEMENT QUALITY — PESQ & STOI
# ═══════════════════════════════════════════════════════════════════════════════

def run_enhancement_quality(
    samples,           # list[Sample] from dataset.py
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality, ITU-T P.862.2)
    and STOI (Short-Time Objective Intelligibility, Taal et al. 2011) for
    each enhancement method at each SNR level.

    These are standard speech-quality metrics taught in speech processing
    courses and provide an objective measure of enhancement benefit independent
    of ASR.

    PESQ range: −0.5 (bad) … 4.5 (excellent)   [narrow-band MOS scale]
    STOI range:  0 (unintelligible) … 1 (perfect)
    """
    from pesq import pesq as _pesq
    from pystoi import stoi as _stoi

    exp_dir = output_dir / "exp_enhancement_quality"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "pesq_stoi_results.csv"
    if csv_path.exists():
        print("  [EQ] Loading cached results…")
        return pd.read_csv(csv_path)

    NOISE_TYPES = ["white", "pink", "soundtrack"]
    SNR_LEVELS  = [-5, 0, 5, 10, 15, 20]
    METHODS     = ["noisy", "wiener", "spectral_sub"]

    rows = []
    total = len(samples) * len(NOISE_TYPES) * len(SNR_LEVELS) * len(METHODS)
    done = 0

    for sample in samples:
        clean, sr = sf.read(sample.audio_path)
        clean = clean.astype(np.float32)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        if sr != TARGET_SR:
            clean = librosa.resample(clean, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        for noise_type in NOISE_TYPES:
            noise = synthesize_noise(noise_type, len(clean), seed=hash(sample.sample_id) % 9999)
            for snr_db in SNR_LEVELS:
                noisy = mix_with_snr(clean, noise, snr_db)
                for method in METHODS:
                    if method == "noisy":
                        proc = noisy
                    else:
                        proc = enhance_audio(noisy, method.replace("_sub", "_subtraction") if "sub" in method else method)

                    # Align length
                    min_len = min(len(clean), len(proc))
                    ref = clean[:min_len]
                    deg = proc[:min_len]

                    try:
                        pesq_score = float(_pesq(TARGET_SR, ref, deg, "wb"))
                    except Exception:
                        pesq_score = float("nan")

                    try:
                        stoi_score = float(_stoi(ref, deg, TARGET_SR, extended=False))
                    except Exception:
                        stoi_score = float("nan")

                    rows.append({
                        "sample_id": sample.sample_id,
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                        "method": method,
                        "pesq": round(pesq_score, 4) if not np.isnan(pesq_score) else None,
                        "stoi": round(stoi_score, 4) if not np.isnan(stoi_score) else None,
                    })
                    done += 1
                    if done % 30 == 0:
                        print(f"  [EQ] {done}/{total}  {sample.sample_id} {noise_type} {snr_db}dB {method}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    _plot_enhancement_quality(df, exp_dir)
    return df


def _plot_enhancement_quality(df: pd.DataFrame, exp_dir: Path) -> None:
    METHODS = ["noisy", "wiener", "spectral_sub"]
    NOISE_TYPES = sorted(df["noise_type"].unique())
    style = {
        "noisy":       ("-",  "o", "steelblue",      "No Enhancement"),
        "wiener":      ("--", "s", "darkorange",      "Wiener Filter"),
        "spectral_sub":("-.", "^", "mediumseagreen",  "Spectral Subtraction"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for col_idx, noise in enumerate(NOISE_TYPES[:3]):
        sub = df[df["noise_type"] == noise]
        for method in METHODS:
            ms = sub[sub["method"] == method].groupby("snr_db")["pesq"].mean().reset_index().sort_values("snr_db")
            ls, mk, col, lbl = style[method]
            axes[0, col_idx].plot(ms["snr_db"], ms["pesq"], ls=ls, marker=mk, color=col, label=lbl, linewidth=2)
        axes[0, col_idx].set_title(f"PESQ — {noise} noise", fontsize=10)
        axes[0, col_idx].set_xlabel("SNR (dB)")
        axes[0, col_idx].set_ylabel("PESQ (MOS-LQO)")
        axes[0, col_idx].legend(fontsize=7)
        axes[0, col_idx].grid(alpha=0.3)
        axes[0, col_idx].axhline(3.0, color="grey", linestyle=":", alpha=0.5, label="MOS=3.0 (good)")

        for method in METHODS:
            ms = sub[sub["method"] == method].groupby("snr_db")["stoi"].mean().reset_index().sort_values("snr_db")
            ls, mk, col, lbl = style[method]
            axes[1, col_idx].plot(ms["snr_db"], ms["stoi"], ls=ls, marker=mk, color=col, label=lbl, linewidth=2)
        axes[1, col_idx].set_title(f"STOI — {noise} noise", fontsize=10)
        axes[1, col_idx].set_xlabel("SNR (dB)")
        axes[1, col_idx].set_ylabel("STOI (0–1)")
        axes[1, col_idx].legend(fontsize=7)
        axes[1, col_idx].grid(alpha=0.3)
        axes[1, col_idx].set_ylim([0, 1.05])

    fig.suptitle("Speech Enhancement Quality — PESQ (ITU-T P.862.2) and STOI (Taal et al. 2011)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "fig_pesq_stoi.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary table
    summary = df.groupby("method")[["pesq","stoi"]].mean().round(4)
    summary.to_csv(exp_dir / "pesq_stoi_summary.csv")
    print(f"\n  [EQ] PESQ/STOI Summary:\n{summary.to_string()}")
    print(f"  Saved → {exp_dir}/fig_pesq_stoi.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  NOISE-TYPE CLASSIFIER — MFCC + Spectral Features
# ═══════════════════════════════════════════════════════════════════════════════

_NOISE_CLASSES = ["silence", "white_noise", "pink_noise", "music/soundtrack", "babble"]
_NOISE_COLORS  = {
    "silence":        "lightgrey",
    "white_noise":    "steelblue",
    "pink_noise":     "mediumseagreen",
    "music/soundtrack": "darkorange",
    "babble":         "crimson",
}


def _extract_noise_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract a 16-dim feature vector for noise classification.

    Features chosen from course content:
      - 6 × MFCC means (C1–C6)
      - log RMS energy
      - ZCR
      - Spectral centroid (normalised)
      - Spectral flatness
      - Spectral rolloff
      - 4 × sub-band energy ratios (< 300 Hz, 300-1k, 1k-4k, > 4k)
      - Periodicity proxy (autocorrelation peak ratio)
    """
    audio = audio.astype(np.float32)
    n_fft = 512
    hop   = 160

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    mfcc_means = np.mean(mfcc[1:7], axis=1)  # C1-C6

    rms   = float(np.sqrt(np.mean(audio**2) + 1e-12))
    log_rms = np.log10(rms + 1e-12)

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=400, hop_length=hop)))

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop))) / (sr/2)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop)))
    rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop))) / (sr/2)

    # Sub-band energy ratios
    mag_sq = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))**2
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total  = np.mean(mag_sq) + 1e-12
    band1  = np.mean(mag_sq[freqs < 300]) / total
    band2  = np.mean(mag_sq[(freqs >= 300) & (freqs < 1000)]) / total
    band3  = np.mean(mag_sq[(freqs >= 1000) & (freqs < 4000)]) / total
    band4  = np.mean(mag_sq[freqs >= 4000]) / total

    # Periodicity: normalized autocorrelation peak (strong for music/tonal)
    if len(audio) >= 1000:
        ac = np.correlate(audio[:1000], audio[:1000], mode="full")
        ac = ac[len(ac)//2:]
        ac = ac / (ac[0] + 1e-12)
        pitch_range = ac[50:500]  # ~32Hz–320Hz at 16kHz
        periodicity = float(np.max(np.abs(pitch_range)))
    else:
        periodicity = 0.0

    return np.array([
        *mfcc_means,          # 6
        log_rms,              # 1
        zcr,                  # 1
        centroid,             # 1
        flatness,             # 1
        rolloff,              # 1
        band1, band2, band3, band4,  # 4
        periodicity,          # 1
    ], dtype=np.float32)  # total = 16


def _predict_noise_class(feats: np.ndarray) -> str:
    """
    Rule-based noise classifier using the 16-dim feature vector.
    Designed so the decision logic is directly explainable in a lecture.
    """
    (c1, c2, c3, c4, c5, c6,
     log_rms, zcr, centroid, flatness,
     rolloff, band1, band2, band3, band4,
     periodicity) = feats.tolist()

    # Silence check
    if log_rms < -3.5:
        return "silence"

    # Babble: speech-like MFCCs, high ZCR, moderate flatness
    if zcr > 0.08 and flatness < 0.25 and band2 > 0.15 and band3 > 0.15:
        return "babble"

    # White noise: high flatness, energy spread across bands
    if flatness > 0.35 and band4 > 0.20:
        return "white_noise"

    # Pink noise: flatness moderate, low-freq dominant
    if flatness > 0.15 and band1 > 0.20 and periodicity < 0.25:
        return "pink_noise"

    # Music / soundtrack: high periodicity, tonal, high-freq content
    if periodicity > 0.30 or (centroid > 0.25 and flatness < 0.15):
        return "music/soundtrack"

    return "white_noise"  # fallback


def run_noise_classifier(
    samples,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Train (with synthesized data) and evaluate the noise-type classifier.
    Reports confusion matrix and per-class accuracy.

    Demonstrates that MFCC + spectral features can reliably distinguish
    silence / white / pink / music / babble — directly relevant to
    adaptive front-end design.
    """
    exp_dir = output_dir / "exp_noise_classifier"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "classifier_results.csv"
    if csv_path.exists():
        print("  [NC] Loading cached results…")
        return pd.read_csv(csv_path)

    print("  [NC] Building noise classifier dataset…")
    NOISE_CONFIGS = {
        "white_noise":      ("white",      5),
        "pink_noise":       ("pink",       5),
        "music/soundtrack": ("soundtrack", 5),
        "babble":           ("babble",     5),
    }
    DURATION_SECS = 3
    N_CLIPS_PER_CLASS = 50
    rng = np.random.default_rng(42)

    rows = []
    confusion = {pred: {true: 0 for true in _NOISE_CLASSES} for pred in _NOISE_CLASSES}

    for true_class, (noise_kind, snr_db) in NOISE_CONFIGS.items():
        for clip_idx in range(N_CLIPS_PER_CLASS):
            # Get a speech segment
            sample = samples[clip_idx % len(samples)]
            clean, sr = sf.read(sample.audio_path)
            clean = clean.astype(np.float32)
            if clean.ndim > 1:
                clean = clean.mean(axis=1)
            if sr != TARGET_SR:
                clean = librosa.resample(clean, orig_sr=sr, target_sr=TARGET_SR)
            # Take a 3-second clip
            clip_len = DURATION_SECS * TARGET_SR
            start = rng.integers(0, max(1, len(clean) - clip_len))
            clean_clip = clean[start: start + clip_len]
            if len(clean_clip) < clip_len:
                clean_clip = np.pad(clean_clip, (0, clip_len - len(clean_clip)))

            noise = synthesize_noise(noise_kind, len(clean_clip), seed=clip_idx * 7)
            mixed = mix_with_snr(clean_clip, noise, snr_db)

            feats = _extract_noise_features(mixed, TARGET_SR)
            pred_class = _predict_noise_class(feats)
            confusion[pred_class][true_class] += 1
            rows.append({
                "true_class":  true_class,
                "pred_class":  pred_class,
                "correct":     int(pred_class == true_class),
                "flatness":    round(float(feats[9]), 4),
                "periodicity": round(float(feats[15]), 4),
                "zcr":         round(float(feats[7]), 4),
                "log_rms":     round(float(feats[6]), 4),
            })

    # Silence clips
    for clip_idx in range(N_CLIPS_PER_CLASS // 2):
        silence = np.zeros(DURATION_SECS * TARGET_SR, dtype=np.float32)
        feats = _extract_noise_features(silence, TARGET_SR)
        pred_class = _predict_noise_class(feats)
        confusion[pred_class]["silence"] += 1
        rows.append({
            "true_class": "silence", "pred_class": pred_class,
            "correct": int(pred_class == "silence"),
            "flatness": 0.0, "periodicity": 0.0, "zcr": 0.0, "log_rms": -10.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    acc_per_class = df.groupby("true_class")["correct"].mean().round(4)
    overall_acc   = df["correct"].mean()
    print(f"  [NC] Overall accuracy: {overall_acc:.3f}")
    print(f"  [NC] Per-class:\n{acc_per_class.to_string()}")

    _plot_noise_classifier(df, confusion, exp_dir)
    return df


def _plot_noise_classifier(df: pd.DataFrame, confusion: dict, exp_dir: Path) -> None:
    classes = [c for c in _NOISE_CLASSES if c in df["true_class"].values or c in df["pred_class"].values]

    # Confusion matrix
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, pred in enumerate(classes):
        for j, true in enumerate(classes):
            conf_matrix[i, j] = confusion.get(pred, {}).get(true, 0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Fig a: confusion matrix
    im = axes[0].imshow(conf_matrix, cmap="Blues")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
    axes[0].set_yticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
    axes[0].set_xlabel("True Class")
    axes[0].set_ylabel("Predicted Class")
    axes[0].set_title("Noise Classifier Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[0].text(j, i, str(conf_matrix[i, j]), ha="center", va="center",
                        fontsize=9, color="white" if conf_matrix[i, j] > conf_matrix.max()//2 else "black")
    plt.colorbar(im, ax=axes[0], fraction=0.04)

    # Fig b: per-class accuracy
    acc = df.groupby("true_class")["correct"].mean().reset_index()
    acc_colors = [_NOISE_COLORS.get(c, "grey") for c in acc["true_class"]]
    axes[1].barh(acc["true_class"], acc["correct"] * 100, color=acc_colors, alpha=0.85)
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_title("Per-Class Accuracy")
    axes[1].axvline(100, color="grey", linestyle=":", alpha=0.5)
    for i, row in acc.iterrows():
        axes[1].text(row["correct"]*100 + 1, i, f"{row['correct']*100:.0f}%", va="center", fontsize=9)
    axes[1].grid(alpha=0.3, axis="x")

    # Fig c: feature scatter (spectral flatness vs periodicity, coloured by true class)
    for true_class in df["true_class"].unique():
        sub = df[df["true_class"] == true_class]
        axes[2].scatter(sub["flatness"], sub["periodicity"],
                       c=_NOISE_COLORS.get(true_class, "grey"),
                       label=true_class, alpha=0.55, s=25)
    axes[2].set_xlabel("Spectral Flatness")
    axes[2].set_ylabel("Periodicity (AC peak ratio)")
    axes[2].set_title("Feature Space: Flatness vs Periodicity")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Experiment — Noise-Type Classifier (MFCC + Spectral Features, Rule-Based)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "fig_noise_classifier.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [NC] Saved → {exp_dir}/fig_noise_classifier.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CONFIDENCE CALIBRATION — Whisper avg_logprob vs actual WER
# ═══════════════════════════════════════════════════════════════════════════════

def run_confidence_calibration(
    condition_metrics_csv: Path,
    output_dir: Path,
) -> dict:
    """
    Evaluate how well Whisper's segment-level avg_logprob predicts actual WER.

    A well-calibrated confidence measure should show:
      - high logprob → low WER  (model confident AND correct)
      - low logprob  → high WER (model uncertain AND wrong)

    Metrics reported:
      - Pearson correlation (logprob vs WER)
      - Spearman correlation
      - Expected Calibration Error (ECE) — binned
      - AUROC for "error detection" (classifying WER > 0.3 from logprob)
    """
    exp_dir = output_dir / "exp_confidence_calibration"
    exp_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    df = pd.read_csv(condition_metrics_csv)
    df = df.dropna(subset=["avg_logprob", "wer"])
    df = df[df["avg_logprob"].between(-5.0, 0.0)]  # remove outliers

    # ── Correlations ─────────────────────────────────────────────────────────
    r_p, p_p = pearsonr(df["avg_logprob"], df["wer"])
    r_s, p_s = spearmanr(df["avg_logprob"], df["wer"])

    # ── ECE: bin by logprob, measure mean WER per bin ─────────────────────────
    n_bins = 10
    df["logprob_bin"] = pd.cut(df["avg_logprob"], bins=n_bins, labels=False)
    ece_df = df.groupby("logprob_bin").agg(
        mean_logprob=("avg_logprob", "mean"),
        mean_wer=("wer", "mean"),
        count=("wer", "count"),
    ).reset_index()

    # ECE = mean |confidence - accuracy|
    # Whisper's confidence proxy: −avg_logprob → "error_score" (higher = worse)
    # We invert: conf = 1 − clip(−logprob / 5, 0, 1)
    df["confidence"] = 1.0 - np.clip(-df["avg_logprob"] / 5.0, 0, 1)
    df["accuracy"]   = (df["wer"] < 0.2).astype(float)
    ece_df2 = df.groupby(pd.cut(df["confidence"], bins=10, labels=False)).agg(
        mean_conf=("confidence", "mean"),
        mean_acc=("accuracy", "mean"),
        count=("accuracy", "count"),
    ).dropna()
    ece = float(np.average(
        np.abs(ece_df2["mean_conf"] - ece_df2["mean_acc"]),
        weights=ece_df2["count"],
    ))

    # ── AUROC (simple) ────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import roc_auc_score
        labels = (df["wer"] > 0.3).astype(int)
        scores = -df["avg_logprob"]   # higher neg-logprob → higher "error" score
        auroc  = float(roc_auc_score(labels, scores)) if labels.sum() > 0 else float("nan")
    except ImportError:
        auroc = float("nan")

    results = {
        "n_samples":         len(df),
        "pearson_r":         round(r_p, 4),
        "pearson_p":         round(p_p, 6),
        "spearman_r":        round(r_s, 4),
        "spearman_p":        round(p_s, 6),
        "ece":               round(ece, 4),
        "auroc_wer>0.3":     round(auroc, 4) if not np.isnan(auroc) else None,
    }

    import json
    with open(exp_dir / "calibration_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"  [CC] Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  ECE={ece:.4f}  AUROC={auroc:.3f}")
    _plot_confidence_calibration(df, ece_df, ece_df2, ece, results, exp_dir)
    return results


def _plot_confidence_calibration(
    df, ece_df, ece_df2, ece, results, exp_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Fig a: scatter logprob vs WER
    sample_idx = np.random.default_rng(0).choice(len(df), min(600, len(df)), replace=False)
    sub = df.iloc[sample_idx]
    sc = axes[0].scatter(sub["avg_logprob"], sub["wer"], c=sub["wer"],
                         cmap="RdYlGn_r", alpha=0.5, s=12, vmin=0, vmax=1)
    axes[0].set_xlabel("Whisper avg_logprob (higher = more confident)")
    axes[0].set_ylabel("WER")
    axes[0].set_title(f"Logprob vs WER\n(Pearson r={results['pearson_r']:.3f}, "
                      f"Spearman r={results['spearman_r']:.3f})")
    plt.colorbar(sc, ax=axes[0], label="WER")
    # Add trend line
    z = np.polyfit(sub["avg_logprob"].values, sub["wer"].values, 1)
    xline = np.linspace(sub["avg_logprob"].min(), sub["avg_logprob"].max(), 100)
    axes[0].plot(xline, np.poly1d(z)(xline), "r--", linewidth=1.5, label="Trend")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Fig b: calibration curve
    if not ece_df2.empty:
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        axes[1].scatter(ece_df2["mean_conf"], ece_df2["mean_acc"],
                       s=ece_df2["count"] / 3.0, alpha=0.75, color="steelblue",
                       zorder=3, label="Observed")
        axes[1].set_xlabel("Mean Confidence (1 − |logprob|/5)")
        axes[1].set_ylabel("Fraction Correct (WER < 0.2)")
        axes[1].set_title(f"Confidence Calibration Curve\n(ECE = {ece:.4f})")
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

    # Fig c: WER distribution by confidence quartile
    df["conf_quartile"] = pd.qcut(df["avg_logprob"], q=4, labels=["Q1\n(low conf)", "Q2", "Q3", "Q4\n(high conf)"])
    q_wer = [df[df["conf_quartile"] == q]["wer"].values for q in df["conf_quartile"].cat.categories]
    bp = axes[2].boxplot(q_wer, labels=df["conf_quartile"].cat.categories,
                        patch_artist=True, medianprops={"color": "red", "linewidth": 2})
    colors = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[2].set_xlabel("Confidence Quartile (by avg_logprob)")
    axes[2].set_ylabel("WER")
    axes[2].set_title("WER Distribution by Confidence Level")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment — Whisper Confidence Calibration (avg_logprob as Quality Predictor)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "fig_confidence_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [CC] Saved → {exp_dir}/fig_confidence_calibration.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  STREAMING PIPELINE LATENCY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_streaming_latency(
    samples,
    asr,          # WhisperASR instance
    output_dir: Path,
    chunk_durations: list[float] = None,
) -> pd.DataFrame:
    """
    Simulate a streaming (chunk-based) ASR pipeline and measure:
      - First-word latency (time from speech onset to first transcription)
      - End-to-end latency per chunk
      - Streaming WER (duplicate/missed words from chunk boundaries)
      - RTF at different chunk sizes

    Compares: chunk_size in [1s, 2s, 3s, 5s] vs batch (full utterance).
    """
    if chunk_durations is None:
        chunk_durations = [1.0, 2.0, 3.0, 5.0]

    exp_dir = output_dir / "exp_streaming_latency"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "streaming_results.csv"
    if csv_path.exists():
        print("  [SL] Loading cached results…")
        return pd.read_csv(csv_path)

    import tempfile

    rows = []
    NOISE_TYPE = "pink"
    SNR_DB = 10

    for sample in samples[:6]:  # use 6 samples for speed
        clean, sr = sf.read(sample.audio_path)
        clean = clean.astype(np.float32)
        if clean.ndim > 1: clean = clean.mean(axis=1)
        if sr != TARGET_SR:
            clean = librosa.resample(clean, orig_sr=sr, target_sr=TARGET_SR)
        noise = synthesize_noise(NOISE_TYPE, len(clean), seed=77)
        audio = mix_with_snr(clean, noise, SNR_DB)
        audio_dur = len(audio) / TARGET_SR

        # ── Batch mode (baseline) ──────────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, TARGET_SR)
            t0 = time.perf_counter()
            pred_batch = asr.transcribe_file(Path(tmp.name), audio_duration_sec=audio_dur)
            batch_inf_ms = (time.perf_counter() - t0) * 1000.0
        import os; os.unlink(tmp.name)
        batch_wer = compute_wer(sample.text, pred_batch.text)

        rows.append({
            "sample_id": sample.sample_id,
            "mode": "batch",
            "chunk_sec": audio_dur,
            "audio_dur_sec": round(audio_dur, 3),
            "first_word_latency_ms": round(batch_inf_ms, 1),  # must wait for full utterance
            "inference_ms": round(batch_inf_ms, 1),
            "rtf": round(batch_inf_ms / 1000.0 / audio_dur, 4),
            "wer": round(batch_wer, 4),
        })

        # ── Streaming modes ────────────────────────────────────────────────
        for chunk_sec in chunk_durations:
            chunk_samples = int(chunk_sec * TARGET_SR)
            chunks = []
            for start in range(0, len(audio), chunk_samples):
                chunks.append(audio[start: start + chunk_samples])

            chunk_texts = []
            chunk_rtfs  = []
            first_word_latency = None

            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) < int(0.2 * TARGET_SR):
                    continue
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, chunk, TARGET_SR)
                    t0 = time.perf_counter()
                    chunk_pred = asr.transcribe_file(Path(tmp.name), audio_duration_sec=len(chunk)/TARGET_SR)
                    chunk_inf_ms = (time.perf_counter() - t0) * 1000.0
                os.unlink(tmp.name)

                if chunk_pred.text and first_word_latency is None:
                    # Latency = chunk offset + inference time
                    audio_offset_ms = chunk_idx * chunk_sec * 1000.0
                    first_word_latency = audio_offset_ms + chunk_inf_ms

                chunk_texts.append(chunk_pred.text)
                chunk_rtfs.append(chunk_inf_ms / 1000.0 / max(len(chunk)/TARGET_SR, 0.01))

            full_hyp = " ".join(t for t in chunk_texts if t)
            stream_wer = compute_wer(sample.text, full_hyp)

            rows.append({
                "sample_id": sample.sample_id,
                "mode": f"stream_{chunk_sec}s",
                "chunk_sec": chunk_sec,
                "audio_dur_sec": round(audio_dur, 3),
                "first_word_latency_ms": round(first_word_latency or 0, 1),
                "inference_ms": round(np.mean(chunk_rtfs) * chunk_sec * 1000, 1),
                "rtf": round(np.mean(chunk_rtfs), 4),
                "wer": round(stream_wer, 4),
            })
            print(f"  [SL] {sample.sample_id} chunk={chunk_sec}s  "
                  f"first_word_lat={first_word_latency:.0f}ms  "
                  f"wer={stream_wer:.3f}  rtf={np.mean(chunk_rtfs):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    _plot_streaming_latency(df, exp_dir)
    return df


def _plot_streaming_latency(df: pd.DataFrame, exp_dir: Path) -> None:
    modes = ["batch"] + [f"stream_{d}s" for d in [1.0, 2.0, 3.0, 5.0] if f"stream_{d}s" in df["mode"].values]
    colors = ["#2196F3", "#FF5722", "#FF9800", "#4CAF50", "#9C27B0"]

    summary = df.groupby("mode")[["first_word_latency_ms", "wer", "rtf"]].mean().reset_index()
    summary["mode_label"] = summary["mode"].str.replace("stream_", "").str.replace("s", "s chunk").str.replace("batch", "batch\n(full)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    color_map = {m: colors[i] for i, m in enumerate(modes) if m in summary["mode"].values}
    bar_colors = [color_map.get(m, "grey") for m in summary["mode"]]

    axes[0].bar(summary["mode_label"], summary["first_word_latency_ms"], color=bar_colors, alpha=0.85)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("First-Word Latency vs Chunk Size")
    axes[0].set_xlabel("Processing Mode")
    axes[0].grid(alpha=0.3, axis="y")
    for i, row in summary.iterrows():
        axes[0].text(i, row["first_word_latency_ms"] + 20, f"{row['first_word_latency_ms']:.0f}ms",
                    ha="center", fontsize=8)

    axes[1].bar(summary["mode_label"], summary["wer"], color=bar_colors, alpha=0.85)
    axes[1].set_ylabel("Mean WER")
    axes[1].set_title("WER Degradation from Chunk Boundaries")
    axes[1].set_xlabel("Processing Mode")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(summary["mode_label"], summary["rtf"], color=bar_colors, alpha=0.85)
    axes[2].axhline(1.0, color="red", linestyle="--", label="RTF = 1 (real-time)")
    axes[2].set_ylabel("Mean RTF")
    axes[2].set_title("RTF per Chunk Size")
    axes[2].set_xlabel("Processing Mode")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment — Streaming vs Batch ASR Latency (First-Word Latency, WER, RTF)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "fig_streaming_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SL] Saved → {exp_dir}/fig_streaming_latency.png")
