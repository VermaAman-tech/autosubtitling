import argparse
import json
import re
from pathlib import Path
import pandas as pd

from project.trailer_v2 import run_enhanced_pipeline
from project.metrics import compute_wer, compute_cer

# For parsing SRT cues
from run_trailer_experiments import discover_trailers, normalize

def read_srt_text(srt_path: Path) -> str:
    """Read an SRT file and return normalized space-separated text, ignoring tags."""
    if not srt_path.exists():
        return ""
    try:
        text = srt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = srt_path.read_text(encoding="latin-1")
        
    lines = text.split("\n")
    cues_text = []
    current_cue = []
    
    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            if current_cue:
                cues_text.append(" ".join(current_cue))
                current_cue = []
            continue
        
        # Strip annotation tags like [SAD]😢 [MUSIC/heroic]🎵
        stripped = re.sub(r"\[[\w/]+\][^\s]*\s*", "", line)
        # Strip TurboScribe watermark or typical youtube auto-captions tags like [Music]
        stripped = re.sub(r"\[Music\]", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\(Transcribed by.*?\)\s*", "", stripped)
        
        if stripped.strip():
            current_cue.append(stripped.strip())
            
    if current_cue:
        cues_text.append(" ".join(current_cue))
        
    full_text = " ".join(cues_text)
    return normalize(full_text)


def evaluate_pair(name: str, video_path: Path, ref_srt_path: Path, output_dir: Path, model_size: str):
    print(f"\n{'='*80}")
    print(f"Evaluating: {name}")
    print(f"Video: {video_path}")
    print(f"Ref SRT: {ref_srt_path}")
    print(f"{'='*80}")
    
    # Create specific output folder for this media
    media_out_dir = output_dir / name
    media_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run the full pipeline (generates clean and annotated SRTs)
    results = run_enhanced_pipeline(
        video_path=video_path,
        output_dir=media_out_dir,
        model_size=model_size
    )
    
    # 2. Extract normalized text from the GT SRT and the Predicted Clean SRT
    ref_text = read_srt_text(ref_srt_path)
    
    pred_clean_srt_path = Path(results["clean_srt"])
    pred_text = read_srt_text(pred_clean_srt_path)
    
    # 3. Compute metrics
    if not ref_text:
        wer = 0.0 if not pred_text else 1.0
        cer = 0.0 if not pred_text else 1.0
    else:
        # compute_wer and cer from jiwer via our metrics.py
        wer = compute_wer(ref_text, pred_text)
        cer = compute_cer(ref_text, pred_text)
        
    results["wer"] = wer
    results["cer"] = cer
    results["ref_text_len"] = len(ref_text)
    results["pred_text_len"] = len(pred_text)
    
    print(f"  --> WER: {wer:.2%} | CER: {cer:.2%} | RTF: {results['rtf']:.3f}")
    
    return results


def discover_media(project_dir: Path) -> list[dict]:
    media_files = []
    seen_names = set()
    
    # Check both mp4 and mkv
    for ext in ("*.mp4", "*.mkv"):
        for video_path in sorted(project_dir.glob(ext)):
            srt_path = video_path.with_suffix(".srt")
            if srt_path.exists():
                raw_name = video_path.stem[:40].lower()
                name = raw_name
                counter = 2
                while name in seen_names:
                    name = f"{raw_name}_{counter}"
                    counter += 1
                seen_names.add(name)
                
                media_files.append({
                    "name": name,
                    "video": video_path,
                    "ref_srt": srt_path
                })
    return media_files

def main():
    parser = argparse.ArgumentParser(description="Full Movie & Trailer Pipeline Evaluation")
    parser.add_argument("--model", default="medium.en", help="Model size to use")
    parser.add_argument("--out", default="evaluate_results", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    project_dir = Path(__file__).parent
    
    trailers = discover_media(project_dir / "youtube_trailers")
    movies = discover_media(project_dir / "movies")
    
    all_media = trailers + movies
    print(f"Found {len(trailers)} trailers and {len(movies)} movies with reference SRTs.")
    
    if not all_media:
        print("No media found!")
        return
        
    results_list = []
    
    for media in all_media:
        if not media["ref_srt"]:
            print(f"Skipping {media['name']} (no ref SRT)")
            continue
            
        res = evaluate_pair(
            name=media["name"],
            video_path=media["video"],
            ref_srt_path=media["ref_srt"],
            output_dir=out_dir,
            model_size=args.model
        )
        
        # Store for CSV
        results_list.append({
            "name": media["name"],
            "type": "trailer" if media in trailers else "movie",
            "model": args.model,
            "wer": res["wer"],
            "cer": res["cer"],
            "rtf": res["rtf"],
            "total_sec": res["total_sec"],
            "clean_srt": res["clean_srt"],
            "annotated_srt": res["annotated_srt"]
        })
        
    # Save combined report
    df = pd.DataFrame(results_list)
    df.to_csv(out_dir / "full_evaluation_summary.csv", index=False)
    
    print(f"\n{'='*80}")
    print("ALL EVALUATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"Summary saved to {out_dir / 'full_evaluation_summary.csv'}")
    
    # Print aggregate stats
    print("\nAggregate Statistics:")
    for mtype in df["type"].unique():
        sub = df[df["type"] == mtype]
        print(f"\n[{mtype.upper()}s - {len(sub)} items]")
        print(f"  Mean WER: {sub['wer'].mean():.2%}")
        print(f"  Mean CER: {sub['cer'].mean():.2%}")
        print(f"  Mean RTF: {sub['rtf'].mean():.3f}")

if __name__ == "__main__":
    main()
