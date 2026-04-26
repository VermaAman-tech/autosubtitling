#!/usr/bin/env python3
"""
EE 679 — Mass Trailer Evaluation + Scene Understanding
=======================================================
Processes every trailer (and movie) in --video-dir that has a matching .srt,
runs the full pipeline (ASR + emotion + scene understanding), computes
WER/CER against ground-truth SRTs, and generates a comprehensive report.

Also runs the three novel experiments:
  • PESQ / STOI speech quality metrics
  • Noise-type classifier evaluation
  • Confidence calibration
  • Streaming pipeline latency

Usage:
    # 1. Download trailers first
    python scripts/download_trailers.py --output-dir trailers --limit 50

    # 2. Run mass evaluation
    python run_mass_experiments.py --video-dir trailers --output-dir mass_results

    # Optionally skip slow novel experiments
    python run_mass_experiments.py --video-dir trailers --no-pesq --no-streaming
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

from project.audio_utils import TARGET_SR
from project.metrics import compute_wer, compute_cer, SubtitleCue, cps_violations, timestamp_mae
from project.scene_understanding import (
    analyze_audio_scene,
    plot_scene_analysis,
    annotate_srt_with_scene,
    MOOD_COLORS,
    SCENE_COLORS,
)

import librosa


# ─── SRT utilities ────────────────────────────────────────────────────────────

def parse_srt(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks  = re.split(r"\n\s*\n", content.strip())
    cues    = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1].strip()
        )
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end   = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        text  = " ".join(lines[2:]).strip()
        text  = re.sub(r"\[[\w/]+\][^\s]*\s*", "", text)  # strip annotation tags
        text  = re.sub(r"<[^>]+>", "", text)               # strip HTML
        text  = re.sub(r"\(.*?\)", "", text)                # strip parentheticals
        text  = " ".join(text.split())
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def srt_to_text(cues: list[dict]) -> str:
    return " ".join(c["text"] for c in cues)


def get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: Path, wav_path: Path) -> None:
    import subprocess
    ffmpeg = get_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", str(video_path),
           "-ac", "1", "-ar", str(TARGET_SR), "-vn",
           "-acodec", "pcm_s16le", str(wav_path)]
    subprocess.run(cmd, capture_output=True, check=True)


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


# ─── Per-trailer processing ────────────────────────────────────────────────────

def process_one_trailer(
    video_path: Path,
    ref_srt_path: Path,
    output_dir: Path,
    asr,
    run_scene: bool = True,
) -> dict:
    """
    Process one trailer end-to-end and return metric dict.
    """
    slug    = video_path.stem
    out_dir = output_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / "audio.wav"
    if not wav_path.exists():
        extract_audio(video_path, wav_path)
    audio = load_wav(wav_path)
    dur   = len(audio) / TARGET_SR

    # ── ASR ──────────────────────────────────────────────────────────────────
    asr.reset_stats()
    from project.audio_utils import VAD_METHODS, enhance_audio, extract_features, should_enhance
    vad_result = VAD_METHODS["spectral"](audio)

    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    pred_cues: list[SubtitleCue] = []
    seg_records = []

    for idx, (seg_start, seg_end) in enumerate(vad_result.segments):
        seg = audio[int(seg_start * TARGET_SR): int(seg_end * TARGET_SR)]
        if len(seg) < int(0.2 * TARGET_SR):
            continue
        feats    = extract_features(seg)
        proc_seg = enhance_audio(seg) if should_enhance(feats) else seg
        wav_seg  = tmp_dir / f"seg_{idx:04d}.wav"
        pred     = asr.transcribe_audio_array(wav_seg, proc_seg, TARGET_SR)
        if pred.text:
            pred_cues.append(SubtitleCue(start=seg_start, end=seg_end, text=pred.text))
            seg_records.append({
                "seg_idx": idx, "start": seg_start, "end": seg_end,
                "text": pred.text, "rtf": pred.rtf,
                "inference_ms": pred.inference_time_ms,
                "enhanced": should_enhance(feats),
            })

    # Write predicted SRT
    from project.subtitles import write_srt
    write_srt(pred_cues, out_dir / "predicted.srt")

    # ── WER / CER vs reference ───────────────────────────────────────────────
    ref_cues = parse_srt(ref_srt_path)
    ref_text = srt_to_text(ref_cues)
    pred_text = " ".join(c.text for c in pred_cues)

    overall_wer = compute_wer(ref_text, pred_text) if ref_text and pred_text else 1.0
    overall_cer = compute_cer(ref_text, pred_text) if ref_text and pred_text else 1.0

    # Cue-level metrics (paired on nearest timestamp)
    ref_sub_cues = [SubtitleCue(start=c["start"], end=c["end"], text=c["text"]) for c in ref_cues]
    ts_mae_val = timestamp_mae(ref_sub_cues, pred_cues)
    cps_viol   = cps_violations(pred_cues)

    stats = asr.stats

    result = {
        "slug":              slug,
        "duration_sec":      round(dur, 2),
        "vad_segments":      len(vad_result.segments),
        "speech_ratio":      round(vad_result.speech_ratio, 3),
        "pred_cues":         len(pred_cues),
        "ref_cues":          len(ref_cues),
        "wer":               round(overall_wer, 4),
        "cer":               round(overall_cer, 4),
        "ts_mae_sec":        round(ts_mae_val, 3) if ts_mae_val == ts_mae_val else None,
        "cps_violations":    cps_viol,
        "asr_rtf_mean":      round(stats.mean_rtf, 4),
        "asr_rtf_overall":   round(stats.overall_rtf, 4),
        "asr_total_inf_ms":  round(stats.total_inference_ms, 1),
        "enhanced_segs":     sum(1 for r in seg_records if r["enhanced"]),
    }

    # ── Scene understanding ──────────────────────────────────────────────────
    if run_scene and dur < 600:  # skip very long files for speed
        print(f"  [scene] Running scene analysis on {slug}…")
        t_scene = time.perf_counter()
        scene   = analyze_audio_scene(audio, TARGET_SR)
        scene_ms = (time.perf_counter() - t_scene) * 1000.0

        # Save scene analysis JSON
        scene_dict = scene.to_dict()
        with open(out_dir / "scene_analysis.json", "w") as fh:
            json.dump(scene_dict, fh, indent=2)

        # Annotate subtitles with scene context
        annotated = annotate_srt_with_scene(pred_cues, scene)
        with open(out_dir / "annotated_srt.json", "w") as fh:
            json.dump(annotated, fh, indent=2)

        # Write annotated SRT
        _write_annotated_srt(annotated, out_dir / "predicted_annotated.srt")

        # Plot scene analysis figure
        plot_scene_analysis(audio, TARGET_SR, scene,
                           out_dir / "scene_analysis.png",
                           title=slug.replace("_", " ").title())

        result.update({
            "scene_dominant_mood":   scene.dominant_mood,
            "scene_music_fraction":  scene.music_fraction,
            "scene_speech_fraction": scene.speech_fraction,
            "scene_n_segments":      len(scene.segments),
            "scene_n_boundaries":    len(scene.boundaries),
            "scene_compute_ms":      round(scene_ms, 1),
        })
        print(f"  [scene] {slug}: dominant_mood={scene.dominant_mood}  "
              f"music={scene.music_fraction:.2f}  boundaries={len(scene.boundaries)}")

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"  [{slug}] WER={overall_wer:.3f}  CER={overall_cer:.3f}  "
          f"RTF={stats.overall_rtf:.4f}  cues={len(pred_cues)}")
    return result


def _write_annotated_srt(annotated: list[dict], path: Path) -> None:
    lines = []
    for idx, cue in enumerate(annotated, start=1):
        start_ts = _sec_to_srt_time(cue["start"])
        end_ts   = _sec_to_srt_time(cue["end"])
        text     = cue["annotated_text"]
        lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def _sec_to_srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# ─── Aggregate analysis ────────────────────────────────────────────────────────

def aggregate_results(results: list[dict], output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "mass_results.csv", index=False)

    print("\n" + "═"*60)
    print("AGGREGATE RESULTS")
    print("═"*60)
    print(df[["slug","wer","cer","asr_rtf_overall","pred_cues","cps_violations"]].to_string(index=False))

    # ── Figures ─────────────────────────────────────────────────────────────

    # Fig 1: WER / CER by video
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    videos = df["slug"].str.replace("_", " ").str[:20]
    x = np.arange(len(df))
    axes[0].bar(x, df["wer"], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(videos, rotation=40, ha="right", fontsize=7)
    axes[0].set_ylabel("WER")
    axes[0].set_title("WER per Trailer")
    axes[0].axhline(df["wer"].mean(), color="red", linestyle="--",
                    label=f"Mean={df['wer'].mean():.3f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, df["cer"], color="darkorange", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(videos, rotation=40, ha="right", fontsize=7)
    axes[1].set_ylabel("CER")
    axes[1].set_title("CER per Trailer")
    axes[1].axhline(df["cer"].mean(), color="red", linestyle="--",
                    label=f"Mean={df['cer'].mean():.3f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Mass Trailer Evaluation — WER and CER per Video", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_mass_wer_cer.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: RTF distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, df["asr_rtf_overall"], color="mediumseagreen", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(videos, rotation=40, ha="right", fontsize=7)
    axes[0].axhline(1.0, color="red", linestyle="--", label="RTF=1 (real-time)")
    axes[0].set_ylabel("Overall RTF")
    axes[0].set_title("ASR RTF per Trailer (< 1 = faster than real-time)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].scatter(df["speech_ratio"], df["wer"], c=df["asr_rtf_overall"],
                   cmap="RdYlGn_r", s=60, alpha=0.8)
    for _, row in df.iterrows():
        axes[1].annotate(row["slug"][:12], (row["speech_ratio"], row["wer"]),
                        fontsize=6, alpha=0.6)
    axes[1].set_xlabel("Speech Ratio (VAD)")
    axes[1].set_ylabel("WER")
    axes[1].set_title("WER vs Speech Ratio")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_rtf_speech_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: Scene understanding summary (if available)
    if "scene_dominant_mood" in df.columns:
        _plot_scene_summary(df, output_dir)

    return df


def _plot_scene_summary(df: pd.DataFrame, output_dir: Path) -> None:
    from collections import Counter

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mood distribution
    moods = [m for m in df["scene_dominant_mood"].dropna()]
    mood_counts = Counter(moods)
    labels = list(mood_counts.keys())
    vals   = [mood_counts[l] for l in labels]
    colors = [MOOD_COLORS.get(l, "grey") for l in labels]
    axes[0].bar(labels, vals, color=colors, alpha=0.85)
    axes[0].set_title("Dominant Mood Distribution Across Trailers")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3, axis="y")

    # Music fraction vs WER
    df_s = df.dropna(subset=["scene_music_fraction", "wer"])
    axes[1].scatter(df_s["scene_music_fraction"], df_s["wer"],
                   color="darkorange", alpha=0.75, s=50)
    for _, row in df_s.iterrows():
        axes[1].annotate(row["slug"][:10], (row["scene_music_fraction"], row["wer"]),
                        fontsize=6, alpha=0.5)
    if len(df_s) > 1:
        z = np.polyfit(df_s["scene_music_fraction"], df_s["wer"], 1)
        xline = np.linspace(df_s["scene_music_fraction"].min(), df_s["scene_music_fraction"].max(), 50)
        axes[1].plot(xline, np.poly1d(z)(xline), "r--", linewidth=1.5)
    axes[1].set_xlabel("Music Fraction")
    axes[1].set_ylabel("WER")
    axes[1].set_title("WER vs Music Fraction\n(higher music → harder ASR?)")
    axes[1].grid(alpha=0.3)

    # Scene composition pie
    if "scene_speech_fraction" in df.columns:
        avg_speech  = df["scene_speech_fraction"].mean()
        avg_music   = df["scene_music_fraction"].mean()
        avg_silence = 1 - avg_speech - avg_music
        avg_silence = max(0, avg_silence)
        wedge_sizes = [avg_speech, avg_music, avg_silence]
        wedge_labels = [
            f"Speech\n{avg_speech*100:.0f}%",
            f"Music Only\n{avg_music*100:.0f}%",
            f"Other/Silence\n{avg_silence*100:.0f}%",
        ]
        wedge_colors = ["#2196F3", "#9C27B0", "#B0BEC5"]
        axes[2].pie(wedge_sizes, labels=wedge_labels, colors=wedge_colors,
                   autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
        axes[2].set_title("Average Scene Composition\n(All Trailers)")

    fig.suptitle("Scene Understanding Summary — Mood, Music, Speech Distribution",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_scene_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scene summary figure saved")


# ─── Novel experiments wrapper ─────────────────────────────────────────────────

def run_novel_experiments(
    samples,
    asr,
    output_dir: Path,
    run_pesq:      bool = True,
    run_noise_cls: bool = True,
    run_conf_cal:  bool = True,
    run_streaming: bool = True,
) -> dict:
    from project.novel_experiments import (
        run_enhancement_quality,
        run_noise_classifier,
        run_confidence_calibration,
        run_streaming_latency,
    )

    novel_results = {}

    if run_pesq:
        print("\n── Novel Exp A: PESQ/STOI Enhancement Quality ──")
        df_pesq = run_enhancement_quality(samples, output_dir)
        if not df_pesq.empty:
            summary = df_pesq.groupby("method")[["pesq","stoi"]].mean().round(4)
            print(f"\n  PESQ/STOI Summary:\n{summary.to_string()}")
            novel_results["pesq_stoi"] = summary.to_dict()

    if run_noise_cls:
        print("\n── Novel Exp B: Noise-Type Classifier ──")
        df_nc = run_noise_classifier(samples, output_dir)
        if not df_nc.empty:
            acc = df_nc.groupby("true_class")["correct"].mean()
            novel_results["noise_classifier"] = {
                "overall_accuracy": round(float(df_nc["correct"].mean()), 4),
                "per_class": acc.round(4).to_dict(),
            }

    if run_conf_cal:
        print("\n── Novel Exp C: Confidence Calibration ──")
        csv_path = output_dir / "exp2_noise_robustness" / "full_results.csv"
        if csv_path.exists():
            cal_results = run_confidence_calibration(csv_path, output_dir)
            novel_results["confidence_calibration"] = cal_results
        else:
            print("  [CC] Skipped: no condition_metrics.csv found")

    if run_streaming:
        print("\n── Novel Exp D: Streaming Pipeline Latency ──")
        df_stream = run_streaming_latency(samples[:6], asr, output_dir)
        if not df_stream.empty:
            stream_summary = df_stream.groupby("mode")[["first_word_latency_ms","wer","rtf"]].mean().round(4)
            print(f"\n  Streaming Summary:\n{stream_summary.to_string()}")
            novel_results["streaming"] = stream_summary.to_dict()

    return novel_results


# ─── Master report ─────────────────────────────────────────────────────────────

def write_mass_report(
    df_trailers: pd.DataFrame,
    novel_results: dict,
    output_dir: Path,
) -> None:
    lines = []
    def p(t): lines.append(t + "\n")
    def h1(t): lines.append(f"\n# {t}\n")
    def h2(t): lines.append(f"\n## {t}\n")

    h1("EE 679 — Mass Trailer Evaluation + Scene Understanding Report")
    p(f"Videos processed: **{len(df_trailers)}**")
    if not df_trailers.empty:
        p(f"Mean WER: **{df_trailers['wer'].mean():.4f}**  "
          f"Mean CER: **{df_trailers['cer'].mean():.4f}**  "
          f"Mean RTF: **{df_trailers['asr_rtf_overall'].mean():.4f}**")

    h2("Per-Trailer Results")
    cols = ["slug","duration_sec","pred_cues","ref_cues","wer","cer","asr_rtf_overall","cps_violations"]
    available = [c for c in cols if c in df_trailers.columns]
    if not df_trailers.empty and available:
        p("| " + " | ".join(available) + " |")
        p("|" + "---|"*len(available))
        for _, row in df_trailers.iterrows():
            p("| " + " | ".join(str(row.get(c, "")) for c in available) + " |")

    if "scene_dominant_mood" in df_trailers.columns:
        h2("Scene Understanding Summary")
        scene_cols = ["slug","scene_dominant_mood","scene_music_fraction","scene_speech_fraction","scene_n_segments","scene_n_boundaries"]
        sc_avail = [c for c in scene_cols if c in df_trailers.columns]
        p("| " + " | ".join(sc_avail) + " |")
        p("|" + "---|"*len(sc_avail))
        for _, row in df_trailers.dropna(subset=["scene_dominant_mood"]).iterrows():
            p("| " + " | ".join(str(row.get(c,"")) for c in sc_avail) + " |")

    if "pesq_stoi" in novel_results:
        h2("Novel Experiment A — PESQ/STOI Speech Enhancement Quality")
        pq = novel_results["pesq_stoi"]
        for method, vals in pq.items():
            p(f"- **{method}**: PESQ={vals.get('pesq','?')}  STOI={vals.get('stoi','?')}")

    if "noise_classifier" in novel_results:
        h2("Novel Experiment B — Noise-Type Classifier")
        nc = novel_results["noise_classifier"]
        p(f"Overall accuracy: **{nc['overall_accuracy']:.4f}**")
        p("Per-class accuracy:")
        for cls, acc in nc.get("per_class", {}).items():
            p(f"  - {cls}: {acc:.4f}")

    if "confidence_calibration" in novel_results:
        h2("Novel Experiment C — Whisper Confidence Calibration")
        cc = novel_results["confidence_calibration"]
        p(f"- Pearson r: **{cc.get('pearson_r','?')}**")
        p(f"- Spearman r: **{cc.get('spearman_r','?')}**")
        p(f"- ECE: **{cc.get('ece','?')}**")
        p(f"- AUROC (WER>0.3): **{cc.get('auroc_wer>0.3','?')}**")

    if "streaming" in novel_results:
        h2("Novel Experiment D — Streaming Pipeline Latency")
        st = novel_results["streaming"]
        if "first_word_latency_ms" in st:
            p("| Mode | First-Word Latency (ms) | WER | RTF |")
            p("|---|---|---|---|")
            lats = st["first_word_latency_ms"]
            wers = st.get("wer", {})
            rtfs = st.get("rtf", {})
            for mode in lats:
                p(f"| {mode} | {lats[mode]:.0f} | {wers.get(mode,0):.4f} | {rtfs.get(mode,0):.4f} |")

    report_path = output_dir / "MASS_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written → {report_path}")

    with open(output_dir / "novel_results.json", "w") as fh:
        import json
        json.dump(novel_results, fh, indent=2)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir",   default="trailers",      help="Directory with .mp4 + matching .srt files")
    parser.add_argument("--output-dir",  default="mass_results",  help="Where to write outputs")
    parser.add_argument("--model",       default="tiny.en",       help="Whisper model size")
    parser.add_argument("--no-scene",    action="store_true",     help="Skip scene understanding")
    parser.add_argument("--no-pesq",     action="store_true",     help="Skip PESQ/STOI experiment")
    parser.add_argument("--no-noise-cls",action="store_true",     help="Skip noise classifier")
    parser.add_argument("--no-conf-cal", action="store_true",     help="Skip confidence calibration")
    parser.add_argument("--no-streaming",action="store_true",     help="Skip streaming latency")
    parser.add_argument("--data-dir",    default="data",          help="LibriSpeech data dir (for novel exps)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir  = Path(args.video_dir)

    t_start = time.perf_counter()
    print(f"\n{'═'*60}")
    print("EE 679 — Mass Evaluation + Scene Understanding")
    print(f"{'═'*60}")
    print(f"Video dir  : {video_dir.resolve()}")
    print(f"Output dir : {output_dir.resolve()}")
    print(f"Model      : {args.model}")

    # ── Collect trailers with matching SRTs ───────────────────────────────────
    pairs: list[tuple[Path, Path]] = []

    # Also include already-downloaded trailers in root dir
    for mp4 in sorted(list(video_dir.glob("*.mp4")) + list(Path(".").glob("*.mp4"))):
        srt_candidates = [
            mp4.with_suffix(".srt"),
            mp4.parent / (mp4.stem + ".en.srt"),
        ]
        for srt in srt_candidates:
            if srt.exists():
                pairs.append((mp4, srt))
                break

    if not pairs:
        print(f"\n[WARN] No mp4+srt pairs found in {video_dir}")
        print("  Run: python scripts/download_trailers.py --output-dir trailers")
        print("  Then re-run this script.")
        # Still run novel experiments on LibriSpeech
    else:
        print(f"\nFound {len(pairs)} trailer(s) with reference SRTs")

    # ── Load ASR ─────────────────────────────────────────────────────────────
    from project.asr import WhisperASR
    print(f"\nLoading Whisper {args.model}…")
    asr = WhisperASR(model_size=args.model)
    print(f"  Load time: {asr.load_time_ms:.0f}ms")

    # ── Process each trailer ──────────────────────────────────────────────────
    all_results = []
    for mp4, srt in pairs:
        print(f"\n{'─'*50}")
        print(f"Processing: {mp4.name}")
        try:
            result = process_one_trailer(
                mp4, srt, output_dir, asr,
                run_scene=not args.no_scene,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"  [ERROR] {mp4.name}: {exc}")
            import traceback; traceback.print_exc()

    df_trailers = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    if not df_trailers.empty:
        df_trailers = aggregate_results(all_results, output_dir)

    # ── Novel experiments (use LibriSpeech) ───────────────────────────────────
    print(f"\n{'─'*50}")
    print("Running novel experiments on LibriSpeech…")
    try:
        from project.dataset import prepare_librispeech_subset
        samples = prepare_librispeech_subset(Path(args.data_dir), max_items=8)

        # Re-use existing noise robustness CSV if available
        existing_csv = Path("outputs2/exp2_noise_robustness/full_results.csv")
        if not existing_csv.exists():
            existing_csv = output_dir / "exp2_noise_robustness" / "full_results.csv"

        novel = run_novel_experiments(
            samples, asr, output_dir,
            run_pesq=      not args.no_pesq,
            run_noise_cls= not args.no_noise_cls,
            run_conf_cal=  not args.no_conf_cal and existing_csv.exists(),
            run_streaming= not args.no_streaming,
        )
    except Exception as exc:
        print(f"  [WARN] Novel experiments partial failure: {exc}")
        import traceback; traceback.print_exc()
        novel = {}

    write_mass_report(df_trailers, novel, output_dir)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'═'*60}")
    print(f"Done in {elapsed/60:.1f} min  →  {output_dir.resolve()}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
