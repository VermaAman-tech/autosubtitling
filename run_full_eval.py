#!/usr/bin/env python3
"""
EE 679 — Full Evaluation Pipeline (medium.en)
==============================================
The definitive evaluation script. Runs medium.en on every trailer that
has a matching reference SRT, computes WER/CER/chrF/BLEU, performs scene
understanding with the improved classifier, runs the genre classifier,
and produces a comprehensive report with all figures.

Usage:
    python run_full_eval.py                              # defaults
    python run_full_eval.py --model medium.en            # explicitly
    python run_full_eval.py --trailer-dir trailers/      # custom dir
    python run_full_eval.py --no-scene --no-genre        # skip slow parts
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
from project.extra_metrics import compute_chrf, compute_bleu, run_genre_classifier, speaking_rate_timeline
from project.scene_understanding import analyze_audio_scene, plot_scene_analysis, annotate_srt_with_scene
import librosa


# ─── SRT / text utilities ─────────────────────────────────────────────────────

def parse_srt(path: Path) -> list[dict]:
    text   = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n{2,}", text.strip())
    cues   = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]?\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]?\d{3})",
            lines[1].strip() if len(lines) > 1 else "",
        )
        if not m:
            continue
        def _ts(s):
            s = s.replace(",", ".")
            p = s.split(":")
            return int(p[0])*3600 + int(p[1])*60 + float(p[2])
        start = _ts(m.group(1))
        end   = _ts(m.group(2))
        raw   = " ".join(lines[2:])
        raw   = re.sub(r"<[^>]+>", "", raw)
        raw   = re.sub(r"\[.*?\]", "", raw)
        raw   = re.sub(r"&amp;", "&", raw)
        text_ = " ".join(raw.split()).strip()
        if text_:
            cues.append({"start": start, "end": end, "text": text_})
    return cues


def _get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(mp4: Path, wav: Path) -> None:
    import subprocess
    ff  = _get_ffmpeg()
    cmd = [ff, "-y", "-i", str(mp4), "-ac", "1", "-ar", str(TARGET_SR),
           "-vn", "-acodec", "pcm_s16le", str(wav)]
    subprocess.run(cmd, capture_output=True, check=True)


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def _sec_to_srt(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_srt(cues: list[dict], path: Path) -> None:
    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(f"{i}\n{_sec_to_srt(c['start'])} --> {_sec_to_srt(c['end'])}\n{c['text']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


# ─── Per-trailer processing ───────────────────────────────────────────────────

def process_trailer(
    mp4: Path,
    ref_srt: Path,
    out_dir: Path,
    asr,              # EnhancedWhisperASR
    run_scene: bool,
    genre: str = "unknown",
) -> dict:
    slug = re.sub(r"[^\w]", "_", mp4.stem)[:40]
    tdir = out_dir / slug
    tdir.mkdir(parents=True, exist_ok=True)

    # ── Audio extraction ──────────────────────────────────────────────────────
    wav_path = tdir / "audio.wav"
    if not wav_path.exists():
        print(f"  Extracting audio…")
        extract_audio(mp4, wav_path)
    audio = load_wav(wav_path)
    dur   = len(audio) / TARGET_SR

    # ── Transcription with medium.en ──────────────────────────────────────────
    print(f"  Transcribing with {asr.model_size}…")
    t0      = time.perf_counter()
    result  = asr.transcribe(audio, TARGET_SR)
    inf_ms  = (time.perf_counter() - t0) * 1000.0

    from project.enhanced_asr import segments_to_cues
    pred_cue_objs = segments_to_cues(result.reliable_segments)
    pred_cues = [{"start": c.start, "end": c.end, "text": c.text} for c in pred_cue_objs]

    write_srt(pred_cues, tdir / "predicted_clean.srt")
    pred_text = " ".join(c["text"] for c in pred_cues)
    hyp_wc    = len(pred_text.split())

    # ── Reference SRT ────────────────────────────────────────────────────────
    ref_cues = parse_srt(ref_srt)
    ref_text = " ".join(c["text"] for c in ref_cues)
    ref_wc   = len(ref_text.split())

    # ── Metrics ───────────────────────────────────────────────────────────────
    wer  = compute_wer(ref_text, pred_text)   if ref_text and pred_text else 1.0
    cer  = compute_cer(ref_text, pred_text)   if ref_text and pred_text else 1.0
    chrf = compute_chrf(ref_text, pred_text)  if ref_text and pred_text else 0.0
    bleu = compute_bleu(ref_text, pred_text)  if ref_text and pred_text else {}

    # Timestamp MAE & CPS
    ref_sub  = [SubtitleCue(c["start"], c["end"], c["text"]) for c in ref_cues]
    pred_sub = [SubtitleCue(c["start"], c["end"], c["text"]) for c in pred_cues]
    ts_mae   = timestamp_mae(ref_sub, pred_sub)
    cps_viol = cps_violations(pred_sub)

    # Speaking rate
    wpm_stats = speaking_rate_timeline(pred_cues, dur, tdir, slug)

    metrics = {
        "slug":          slug,
        "genre":         genre,
        "duration_sec":  round(dur, 1),
        "model":         asr.model_size,
        "rtf":           round(result.rtf, 4),
        "inference_ms":  round(inf_ms, 0),
        "pred_cues":     len(pred_cues),
        "ref_cues":      len(ref_cues),
        "pred_words":    hyp_wc,
        "ref_words":     ref_wc,
        "wer":           round(wer,  4),
        "cer":           round(cer,  4),
        "chrf":          round(chrf, 2),
        "bleu1":         bleu.get("bleu1", 0.0),
        "bleu2":         bleu.get("bleu2", 0.0),
        "bleu4":         bleu.get("bleu4", 0.0),
        "ts_mae_sec":    round(ts_mae, 3) if ts_mae == ts_mae else None,
        "cps_violations":cps_viol,
        "mean_wpm":      wpm_stats.get("mean_wpm", 0),
        "max_wpm":       wpm_stats.get("max_wpm", 0),
    }
    print(f"  WER={wer:.3f}  CER={cer:.3f}  chrF={chrf:.1f}  "
          f"RTF={result.rtf:.3f}  cues={len(pred_cues)}")

    # ── Scene understanding ───────────────────────────────────────────────────
    if run_scene:
        print(f"  Running scene analysis…")
        scene = analyze_audio_scene(audio, TARGET_SR)
        with open(tdir / "scene_analysis.json", "w") as fh:
            json.dump(scene.to_dict(), fh, indent=2)

        annotated = annotate_srt_with_scene(pred_sub, scene)
        ann_cues  = [{"start":  a["start"], "end": a["end"],
                      "text": a["annotated_text"]} for a in annotated]
        write_srt(ann_cues, tdir / "predicted_annotated.srt")

        plot_scene_analysis(
            audio, TARGET_SR, scene,
            tdir / "scene_analysis.png",
            title=slug.replace("_", " ").title(),
        )

        metrics.update({
            "scene_dominant_mood":   scene.dominant_mood,
            "scene_music_fraction":  scene.music_fraction,
            "scene_speech_fraction": scene.speech_fraction,
            "scene_n_segments":      len(scene.segments),
            "scene_n_boundaries":    len(scene.boundaries),
        })

        # Mood breakdown for this trailer
        mood_counts = scene.mood_counts
        metrics["mood_counts_json"] = json.dumps(mood_counts)
        print(f"  Scene: dominant={scene.dominant_mood}  music={scene.music_fraction:.2f}  "
              f"boundaries={len(scene.boundaries)}")

    with open(tdir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


# ─── Aggregate figures ────────────────────────────────────────────────────────

def make_aggregate_figures(df: pd.DataFrame, out_dir: Path) -> None:

    # ── Fig 1: WER/CER/chrF per trailer ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    slugs  = df["slug"].str.replace("_", " ").str[:22]
    x      = np.arange(len(df))
    genres = df.get("genre", pd.Series(["unknown"]*len(df)))
    GCOL   = {"action": "#E53935", "drama": "#1E88E5", "comedy": "#FDD835",
               "horror": "#212121", "scifi": "#9C27B0", "animation": "#43A047",
               "thriller": "#FF6F00", "biopic": "#795548", "unknown": "#78909C"}
    colors = [GCOL.get(g, "#78909C") for g in genres]

    for ax, col, label, fmt in [
        (axes[0], "wer",  "WER ↓",  "{:.3f}"),
        (axes[1], "cer",  "CER ↓",  "{:.3f}"),
        (axes[2], "chrf", "chrF ↑", "{:.1f}"),
    ]:
        if col not in df.columns:
            continue
        vals = df[col].fillna(0).values
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(float(np.mean(vals)), color="red", linestyle="--",
                   label=f"Mean={np.mean(vals):.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, rotation=42, ha="right", fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(f"{label} per Trailer")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    # Legend for genres
    handles = [plt.Rectangle((0,0),1,1, color=GCOL.get(g,"grey"), alpha=0.85, label=g)
               for g in sorted(set(genres)) if g in GCOL]
    fig.legend(handles=handles, loc="upper right", fontsize=7, title="Genre",
               bbox_to_anchor=(0.99, 0.98))
    fig.suptitle(f"medium.en Auto-Subtitling Metrics — {len(df)} Trailers", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_metrics_per_trailer.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: Genre-level summary ────────────────────────────────────────────
    if "genre" in df.columns and df["genre"].nunique() > 1:
        genre_sum = df.groupby("genre")[["wer","cer","chrf","rtf"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        gx = np.arange(len(genre_sum))
        w  = 0.35
        gc = [GCOL.get(g,"grey") for g in genre_sum["genre"]]
        axes[0].bar(gx - w/2, genre_sum["wer"],  w, color=gc, alpha=0.80, label="WER")
        axes[0].bar(gx + w/2, genre_sum["cer"],  w, color=gc, alpha=0.50, label="CER", hatch="//")
        axes[0].set_xticks(gx)
        axes[0].set_xticklabels(genre_sum["genre"], rotation=30, ha="right")
        axes[0].set_ylabel("Score")
        axes[0].set_title("WER and CER by Genre")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(gx, genre_sum["chrf"], color=gc, alpha=0.85)
        axes[1].set_xticks(gx)
        axes[1].set_xticklabels(genre_sum["genre"], rotation=30, ha="right")
        axes[1].set_ylabel("chrF Score")
        axes[1].set_title("chrF (Character n-gram F-score) by Genre")
        axes[1].grid(alpha=0.3, axis="y")
        for i, row in genre_sum.iterrows():
            axes[1].text(i, row["chrf"] + 0.5, f"{row['chrf']:.1f}", ha="center", fontsize=8)

        fig.suptitle("Genre-Level Performance Analysis", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_genre_performance.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Fig 3: RTF distribution ────────────────────────────────────────────────
    if "rtf" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(x, df["rtf"], color="mediumseagreen", alpha=0.85)
        axes[0].axhline(1.0, color="red", linestyle="--", label="RTF = 1 (real-time)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(slugs, rotation=42, ha="right", fontsize=7)
        axes[0].set_ylabel("RTF")
        axes[0].set_title(f"{df['model'].iloc[0]} RTF per Trailer (all < 1 = real-time)")
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis="y")

        if "duration_sec" in df.columns:
            axes[1].scatter(df["duration_sec"], df["wer"], c=colors, s=60, alpha=0.8)
            for _, row in df.iterrows():
                axes[1].annotate(row["slug"][:10], (row["duration_sec"], row["wer"]),
                                fontsize=6, alpha=0.5)
            axes[1].set_xlabel("Duration (s)")
            axes[1].set_ylabel("WER")
            axes[1].set_title("WER vs Trailer Duration")
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "fig_rtf_duration.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Fig 4: Scene mood distribution (if available) ─────────────────────────
    if "scene_dominant_mood" in df.columns:
        _plot_scene_summary_v2(df, out_dir)

    print(f"  Figures saved to {out_dir}")


def _plot_scene_summary_v2(df: pd.DataFrame, out_dir: Path) -> None:
    from collections import Counter
    MCOL = {"tense":"#E53935","epic":"#9C27B0","sad":"#1E88E5","happy":"#FDD835",
             "calm":"#43A047","neutral":"#78909C","dialogue":"#00ACC1","silence":"#ECEFF1"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mood counts across all trailers (from mood_counts_json)
    all_mood_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        mc_json = row.get("mood_counts_json", "{}")
        try:
            mc = json.loads(mc_json)
            for mood, cnt in mc.items():
                all_mood_counts[mood] = all_mood_counts.get(mood, 0) + cnt
        except Exception:
            pass

    if all_mood_counts:
        moods  = sorted(all_mood_counts, key=all_mood_counts.get, reverse=True)
        counts = [all_mood_counts[m] for m in moods]
        cols   = [MCOL.get(m, "grey") for m in moods]
        axes[0].barh(moods, counts, color=cols, alpha=0.85)
        axes[0].set_xlabel("Segment Count")
        axes[0].set_title("Mood Distribution (All Trailers)")
        axes[0].grid(alpha=0.3, axis="x")

    # Dominant mood per trailer
    mood_per = df[["slug","scene_dominant_mood"]].dropna()
    mood_cts = Counter(mood_per["scene_dominant_mood"])
    labs = list(mood_cts.keys())
    vals = [mood_cts[l] for l in labs]
    axes[1].bar(labs, vals, color=[MCOL.get(l,"grey") for l in labs], alpha=0.85)
    axes[1].set_ylabel("# Trailers")
    axes[1].set_title("Dominant Mood per Trailer")
    axes[1].grid(alpha=0.3, axis="y")

    # Music fraction vs WER
    df_s = df.dropna(subset=["scene_music_fraction","wer"])
    if len(df_s) > 1:
        axes[2].scatter(df_s["scene_music_fraction"]*100, df_s["wer"],
                       c=[MCOL.get(m,"grey") for m in df_s.get("scene_dominant_mood",["neutral"]*len(df_s))],
                       s=70, alpha=0.8)
        for _, row in df_s.iterrows():
            axes[2].annotate(row["slug"][:10], (row["scene_music_fraction"]*100, row["wer"]),
                            fontsize=6, alpha=0.6)
        z = np.polyfit(df_s["scene_music_fraction"], df_s["wer"], 1)
        xline = np.linspace(df_s["scene_music_fraction"].min(), df_s["scene_music_fraction"].max(), 50)
        axes[2].plot(xline*100, np.poly1d(z)(xline), "r--", lw=1.5, label="Trend")
        axes[2].set_xlabel("Music Fraction (%)")
        axes[2].set_ylabel("WER")
        axes[2].set_title("Higher Music → Higher WER?")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

    fig.suptitle("Scene Understanding Summary — Mood Distribution and Music Impact on ASR",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_scene_mood_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── Summary report ───────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, genre_results: dict, out_dir: Path) -> None:
    lines = []
    def p(t): lines.append(t + "\n")
    def h1(t): lines.append(f"\n# {t}\n")
    def h2(t): lines.append(f"\n## {t}\n")

    h1("EE 679 — Full Evaluation Report (medium.en)")
    p(f"**Trailers evaluated**: {len(df)}")
    p(f"**Model**: {df['model'].iloc[0] if not df.empty else 'medium.en'}")
    p(f"**Mean WER**: {df['wer'].mean():.4f}   **Mean CER**: {df['cer'].mean():.4f}   "
      f"**Mean chrF**: {df['chrf'].mean():.1f}   **Mean RTF**: {df['rtf'].mean():.4f}")

    h2("Per-Trailer Results")
    cols = ["slug","genre","duration_sec","pred_cues","ref_cues",
            "wer","cer","chrf","bleu1","rtf","cps_violations",
            "scene_dominant_mood","scene_music_fraction"]
    avail = [c for c in cols if c in df.columns]
    p("| " + " | ".join(avail) + " |")
    p("|" + "---|" * len(avail))
    for _, row in df.iterrows():
        vals = []
        for c in avail:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v) if v == v else "—")
        p("| " + " | ".join(vals) + " |")

    h2("Genre-Level Summary")
    if "genre" in df.columns:
        gs = df.groupby("genre")[["wer","cer","chrf","rtf"]].mean().round(4)
        p("| genre | WER | CER | chrF | RTF |")
        p("|---|---|---|---|---|")
        for g, row in gs.iterrows():
            p(f"| {g} | {row.wer:.4f} | {row.cer:.4f} | {row.chrf:.1f} | {row.rtf:.4f} |")

    if genre_results:
        h2("Genre Classifier (MFCC k-NN LOO)")
        p(f"- Overall LOO accuracy: **{genre_results.get('loo_accuracy', 0)*100:.0f}%**  "
          f"({genre_results.get('n_trailers','?')} trailers)")
        pc = genre_results.get("per_class_accuracy", {})
        if pc:
            p("| Genre | Accuracy |")
            p("|---|---|")
            for g, a in sorted(pc.items()):
                p(f"| {g} | {a*100:.0f}% |")

    h2("Key Findings")
    best = df.nsmallest(3, "wer")[["slug","wer","chrf","genre"]]
    worst = df.nlargest(3, "wer")[["slug","wer","chrf","genre"]]
    p("**Best transcribed trailers** (lowest WER):")
    for _, row in best.iterrows():
        p(f"  - {row['slug']}: WER={row.wer:.3f}, chrF={row.chrf:.1f} ({row.genre})")
    p("\n**Hardest to transcribe** (highest WER):")
    for _, row in worst.iterrows():
        p(f"  - {row['slug']}: WER={row.wer:.3f}, chrF={row.chrf:.1f} ({row.genre})")

    if "scene_music_fraction" in df.columns:
        r = np.corrcoef(df["scene_music_fraction"].fillna(0), df["wer"])[0, 1]
        p(f"\n**Music fraction vs WER correlation**: r = {r:.3f}  "
          f"({'positive — more music → harder ASR' if r > 0 else 'negative — unexpected'})")

    out_dir.joinpath("FULL_EVAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report → {out_dir}/FULL_EVAL_REPORT.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

GENRE_MAP = {
    "avengers": "action", "spider_man": "action", "avengers_age": "action",
    "captain_america": "action", "dark_knight": "action", "mad_max": "action",
    "dunkirk": "action", "iron_man": "action",
    "inception": "scifi", "interstellar": "scifi", "dune": "scifi",
    "martian": "scifi", "gravity": "scifi", "arrival": "scifi",
    "blade_runner": "scifi", "annihilation": "scifi",
    "revenant": "drama", "social_network": "drama", "nomadland": "drama",
    "whiplash": "drama", "1917": "drama", "manchester": "drama",
    "bridesmaids": "comedy", "budapest": "comedy", "knives": "comedy",
    "hereditary": "horror", "quiet_place": "horror", "get_out": "horror",
    "midsommar": "horror",
    "spider_verse": "animation", "soul": "animation", "moana": "animation",
    "frozen": "animation", "onward": "animation",
    "zodiac": "thriller", "gone_girl": "thriller", "prisoners": "thriller",
    "bohemian": "biopic", "judy": "biopic", "theory": "biopic",
}


def infer_genre(slug: str) -> str:
    slug_lower = slug.lower()
    for key, genre in GENRE_MAP.items():
        if key in slug_lower:
            return genre
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trailer-dir", default=".", help="Directory with MP4+SRT pairs")
    parser.add_argument("--output-dir",  default="eval_results_medium")
    parser.add_argument("--model",       default="medium.en")
    parser.add_argument("--no-scene",    action="store_true")
    parser.add_argument("--no-genre",    action="store_true")
    args = parser.parse_args()

    out_dir     = Path(args.output_dir)
    trailer_dir = Path(args.trailer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    print(f"\n{'═'*60}")
    print(f"EE 679 — Full Evaluation with {args.model}")
    print(f"{'═'*60}")

    # ── Collect MP4+SRT pairs ─────────────────────────────────────────────────
    pairs: list[tuple[Path, Path]] = []
    search_dirs = [Path("."), trailer_dir]
    for d in dict.fromkeys(search_dirs):        # deduplicate while preserving order
        for mp4 in sorted(d.glob("*.mp4")):
            srt = mp4.with_suffix(".srt")
            if srt.exists() and mp4.stat().st_size > 100_000:  # skip tiny/broken files
                pairs.append((mp4, srt))

    if not pairs:
        print(f"[WARN] No MP4+SRT pairs found in {trailer_dir} or .")
        return

    print(f"Found {len(pairs)} trailer(s) with reference SRTs\n")

    # ── Load model ────────────────────────────────────────────────────────────
    from project.enhanced_asr import EnhancedWhisperASR
    print(f"Loading {args.model}…")
    asr = EnhancedWhisperASR(model_size=args.model)
    print()

    # ── Process each trailer ──────────────────────────────────────────────────
    all_results = []
    for mp4, srt in pairs:
        print(f"{'─'*50}")
        print(f"{mp4.name}")
        genre = infer_genre(mp4.stem)
        try:
            r = process_trailer(mp4, srt, out_dir, asr,
                                run_scene=not args.no_scene,
                                genre=genre)
            all_results.append(r)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("No results to aggregate.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "results.csv", index=False)

    # ── Aggregate figures ─────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Generating aggregate figures…")
    make_aggregate_figures(df, out_dir)

    # ── Genre classifier ──────────────────────────────────────────────────────
    genre_results = {}
    if not args.no_genre and len(all_results) >= 4:
        print(f"\n{'─'*50}")
        print("Running genre classifier…")
        wav_pairs = []
        for r in all_results:
            wav = out_dir / r["slug"] / "audio.wav"
            g   = r.get("genre", "unknown")
            if wav.exists() and g != "unknown":
                wav_pairs.append((wav, g))
        if wav_pairs:
            genre_results = run_genre_classifier(wav_pairs, out_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(df, genre_results, out_dir)

    # Print summary
    print(f"\n{'═'*60}")
    print(f"Results: {len(df)} trailers  model={args.model}")
    print(f"  Mean WER  : {df['wer'].mean():.4f}")
    print(f"  Mean CER  : {df['cer'].mean():.4f}")
    print(f"  Mean chrF : {df['chrf'].mean():.1f}")
    print(f"  Mean RTF  : {df['rtf'].mean():.4f}")
    print(f"  Total time: {(time.perf_counter()-t_start)/60:.1f} min")
    print(f"  Output    : {out_dir.resolve()}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
