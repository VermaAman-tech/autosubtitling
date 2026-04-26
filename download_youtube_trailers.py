import os
import subprocess
from pathlib import Path

def download_trailers(output_dir: str = "youtube_trailers", total_count: int = 50):
    """
    Downloads movie trailers and their auto-generated subtitles from YouTube.
    Uses yt-dlp to search and download.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # We will search across a few genres to get a diverse set
    genres = ["action", "sci-fi", "drama", "comedy", "horror", "animation", "thriller"]
    
    # Calculate how many to download per genre
    per_genre = max(1, total_count // len(genres))
    
    print(f"Downloading ~{total_count} trailers to {out_path.resolve()}...")
    
    for genre in genres:
        search_query = f"ytsearch{per_genre}:official movie trailer 2023 {genre}"
        print(f"Searching and downloading for genre: {genre}")
        
        # Build the yt-dlp command
        # -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" -> High quality, max 1080p
        # --write-auto-sub -> Download auto-generated subtitles (ground truth proxy)
        # --sub-lang en -> English subtitles
        # --convert-subs srt -> Convert to SRT format for our pipeline
        # --merge-output-format mp4 -> Ensure the output is an MP4 file
        # -o -> Output template
        
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
            print(f"Error downloading {genre} trailers: {e}")
            
    print("\nDownload complete! Check the folder:", output_dir)
    
if __name__ == "__main__":
    download_trailers(output_dir="youtube_trailers", total_count=50)
