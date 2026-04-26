import subprocess
import pysrt
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

from project.trailer_v2 import run_enhanced_pipeline
from project.metrics import compute_wer, compute_cer
from run_trailer_experiments import discover_trailers, normalize

def extract_20m(mkv_in: Path, srt_in: Path, mp4_out: Path, srt_out: Path, audio_idx: int):
    print(f"\n--- Extracting {mp4_out.name} ---")
    if not mp4_out.exists():
        subprocess.run([
            'ffmpeg', '-y', '-i', str(mkv_in), 
            '-ss', '00:00:00', '-t', '00:20:00',
            '-map', '0:v', '-map', f'0:a:{audio_idx}',
            '-c:v', 'copy', '-c:a', 'copy',
            str(mp4_out)
        ], check=True, capture_output=True)
        print(f"Video extracted to {mp4_out}")
    else:
        print(f"{mp4_out} already exists.")
        
    if not srt_out.exists() and srt_in.exists():
        subs = pysrt.open(str(srt_in))
        trimmed_subs = [s for s in subs if s.start.ordinal < 20 * 60 * 1000]
        pysrt.SubRipFile(items=trimmed_subs).save(str(srt_out), encoding='utf-8')
        print(f"SRT extracted to {srt_out}")
    elif srt_out.exists():
        print(f"{srt_out} already exists.")

def compute_similarity(text1: str, text2: str) -> float:
    # Character-level matching ratio (similar to chrF)
    return SequenceMatcher(None, text1, text2).ratio()

def compute_timestamp_mae(ref_srt_path: Path, pred_srt_path: Path) -> float:
    try:
        ref_subs = pysrt.open(str(ref_srt_path))
        pred_subs = pysrt.open(str(pred_srt_path))
    except Exception:
        return float('nan')
        
    if not ref_subs or not pred_subs:
        return float('nan')
        
    # Simple nearest-neighbor time matching
    total_error = 0.0
    matches = 0
    for p_sub in pred_subs:
        p_start = p_sub.start.ordinal / 1000.0
        # Find nearest ref start time
        nearest = min(ref_subs, key=lambda r: abs((r.start.ordinal/1000.0) - p_start))
        r_start = nearest.start.ordinal / 1000.0
        # Only count if it's within a reasonable window (e.g., 5 seconds)
        if abs(r_start - p_start) < 5.0:
            total_error += abs(r_start - p_start)
            matches += 1
            
    return total_error / matches if matches > 0 else float('nan')

def evaluate_snippet(name: str, vid: Path, srt: Path, out_dir: Path):
    print(f"\nEvaluating: {name}")
    media_out_dir = out_dir / name
    
    results = run_enhanced_pipeline(
        video_path=vid,
        output_dir=media_out_dir,
        model_size="medium.en"
    )
    
    # Text Extraction using parsing logic from evaluate.py
    from evaluate import read_srt_text
    ref_text = read_srt_text(srt)
    pred_clean_srt = Path(results["clean_srt"])
    pred_text = read_srt_text(pred_clean_srt)
    
    wer = compute_wer(ref_text, pred_text) if ref_text else 1.0
    cer = compute_cer(ref_text, pred_text) if ref_text else 1.0
    sim = compute_similarity(ref_text, pred_text) if ref_text else 0.0
    
    ts_mae = compute_timestamp_mae(srt, pred_clean_srt)
    
    return {
        "Trailer": name,
        "WER (%)": wer * 100,
        "CER (%)": cer * 100,
        "Similarity (%)": sim * 100,
        "TS-MAE (s)": ts_mae,
        "RTF": results["rtf"],
    }

def main():
    snippets_dir = Path("evaluate_results/snippets")
    snippets_dir.mkdir(parents=True, exist_ok=True)
    
    movie_dir = Path("movies")
    
    movies_to_trim = [
        ("Avengers Endgame 2019 Dual Audio Hindi 480p BluRay", 1), # eng is index 1
        ("Spider-Man.N.W.H.2021.480p.EXT.WEB-DL.Hindi.ORG-English.ESub.x264-", 1), # eng is index 1
        ("The.Dark.Knight.Rises.2012.480p.Dual.Audio.Hin-Eng", 0), # eng is index 0
    ]
    
    # 1. Trim remaining 3 movies
    for name, a_idx in movies_to_trim:
        mkv = movie_dir / f"{name}.mkv"
        srt = movie_dir / f"{name}.srt"
        out_mp4 = snippets_dir / f"{name[:20]}_20m.mp4"
        out_srt = snippets_dir / f"{name[:20]}_20m.srt"
        extract_20m(mkv, srt, out_mp4, out_srt, a_idx)
        
    # Copy social network there too just for completeness
    sn_mp4 = movie_dir / "SocialNetwork_20m.mp4"
    sn_srt = movie_dir / "SocialNetwork_20m.srt"
    if sn_mp4.exists():
        import shutil
        sn_mp4_out = snippets_dir / "SocialNetwork_20m.mp4"
        sn_srt_out = snippets_dir / "SocialNetwork_20m.srt"
        if not sn_mp4_out.exists(): shutil.copy(sn_mp4, sn_mp4_out)
        if not sn_srt_out.exists(): shutil.copy(sn_srt, sn_srt_out)
        
    # 2. Evaluate
    results = []
    out_eval = Path("evaluate_results")
    for mp4 in snippets_dir.glob("*.mp4"):
        srt = mp4.with_suffix(".srt")
        if srt.exists():
            res = evaluate_snippet(mp4.stem, mp4, srt, out_eval)
            results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(out_eval / "clever_metrics_summary.csv", index=False)
    
    print("\n" + "="*80)
    print(" ADVANCED EVALUATION RESULTS (20-min Snippets)")
    print("="*80)
    print(df.to_string(index=False, float_format="%.2f"))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
