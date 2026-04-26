import pandas as pd
from pathlib import Path

from trim_and_eval import compute_timestamp_mae, compute_similarity
from evaluate import read_srt_text
from project.metrics import compute_wer, compute_cer

def evaluate_folder(gt_srt_dir: Path, eval_dir: Path, results_list: list):
    """Scan eval_dir for generated SRTs, find matching GTs, and compute metrics."""
    for folder in eval_dir.iterdir():
        if not folder.is_dir(): continue
        
        pred_srt = folder / "trailer_clean.srt"
        if not pred_srt.exists(): continue
            
        # Try to find corresponding GT SRT
        # GT SRTs are in youtube_trailers/ or movies/ (or snippets)
        name = folder.name
        # Match ignoring case and extension
        possible_gts = list(gt_srt_dir.glob("*.srt"))
        gt_match = None
        for gt in possible_gts:
            if gt.stem.lower()[:40] == name.lower()[:40]:
                gt_match = gt
                break
                
        if not gt_match:
            continue
            
        ref_text = read_srt_text(gt_match)
        pred_text = read_srt_text(pred_srt)
        
        wer = compute_wer(ref_text, pred_text) if ref_text else 1.0
        cer = compute_cer(ref_text, pred_text) if ref_text else 1.0
        sim = compute_similarity(ref_text, pred_text) if ref_text else 0.0
        ts_mae = compute_timestamp_mae(gt_match, pred_srt)
        
        results_list.append({
            "Media": name,
            "Type": "Trailer" if "youtube_trailers" in str(gt_srt_dir) else "Movie Snippet",
            "WER (%)": wer * 100,
            "CER (%)": cer * 100,
            "Similarity (%)": sim * 100,
            "TS-MAE (s)": ts_mae,
        })

def main():
    results = []
    eval_dir = Path("evaluate_results")
    
    # 1. Evaluate Trailers
    evaluate_folder(Path("youtube_trailers"), eval_dir, results)
    
    # 2. Evaluate Movie Snippets
    evaluate_folder(Path("evaluate_results/snippets"), eval_dir, results)
    
    df = pd.DataFrame(results)
    df.to_csv("evaluate_results/all_clever_metrics.csv", index=False)
    
    print("\n" + "="*90)
    print(f" COMPREHENSIVE ADVANCED EVALUATION RESULTS (N={len(df)})")
    print("="*90)
    print(df.to_string(index=False, float_format="%.2f"))
    print("-" * 90)
    print(f"Mean WER:        {df['WER (%)'].mean():.2f}%")
    print(f"Mean CER:        {df['CER (%)'].mean():.2f}%")
    print(f"Mean Similarity: {df['Similarity (%)'].mean():.2f}%")
    print(f"Mean TS-MAE:     {df['TS-MAE (s)'].mean():.2f} s")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
