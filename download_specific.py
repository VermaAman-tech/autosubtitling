import os
import subprocess
from pathlib import Path
import random

def download_specific_trailers(output_dir: str = "youtube_trailers"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    queries = [
        "spiderman movie trailer official",
        "avengers movie trailer official",
        "dune movie trailer official",
        "pirates of the caribbean movie trailer official",
        "godfather movie trailer official",
        "dc comics movie trailer official"
    ]
    
    for query in queries:
        print(f"Searching and downloading for: {query}")
        
        # Request 5 results per query to ensure we get some with subtitles
        search_query = f"ytsearch5:{query}"
        
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--convert-subs", "srt",
            "--merge-output-format", "mp4",
            "-o", f"{output_dir}/%(title)s.%(ext)s",
            search_query
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading for '{query}': {e}")
            
    print("\nDownload complete! Cleaning up MP4s without SRTs...")
    
    # Clean up MP4s without SRTs
    valid_pairs = []
    for mp4 in out_path.glob("*.mp4"):
        srt_en = mp4.with_suffix(".en.srt")
        srt = mp4.with_suffix(".srt")
        if srt_en.exists() or srt.exists():
            valid_pairs.append(mp4)
        else:
            mp4.unlink()
            
    print(f"Total valid MP4+SRT pairs: {len(valid_pairs)}")
    
    # Trim down to 40 pairs if we have more
    target_count = 40
    if len(valid_pairs) > target_count:
        print(f"Trimming to {target_count} pairs...")
        to_delete = random.sample(valid_pairs, len(valid_pairs) - target_count)
        for mp4 in to_delete:
            srt_en = mp4.with_suffix(".en.srt")
            srt = mp4.with_suffix(".srt")
            mp4.unlink()
            if srt_en.exists(): srt_en.unlink()
            if srt.exists(): srt.unlink()
            
    # Final count
    final_count = len(list(out_path.glob("*.mp4")))
    print(f"Final valid pair count: {final_count}")

if __name__ == "__main__":
    download_specific_trailers()
