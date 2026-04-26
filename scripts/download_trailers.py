#!/usr/bin/env python3
"""
Trailer Dataset Builder
=======================
Downloads movie trailers + auto-generated English subtitles from YouTube
using yt-dlp. Auto-subs are used as approximate ground-truth for WER evaluation.

Usage:
    python scripts/download_trailers.py [--output-dir trailers] [--limit 20]

The curated list spans 6 genres so experiments cover diverse acoustic conditions:
  ACTION    — heavy music, explosions, rapid cuts → hardest for ASR
  DRAMA     — clean dialogue, minimal music       → easiest
  SCI-FI    — orchestral score, VFX audio         → moderate
  COMEDY    — dialogue + laughter                 → moderate
  HORROR    — whisper/quiet + sudden loud events  → challenging
  ANIMATION — very clean studio dialogue          → easy/baseline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# ─── Curated trailer list ─────────────────────────────────────────────────────
# Format: (youtube_id, title_slug, genre)
# Selected from official studio channels with reliable English auto-captions.

TRAILERS: list[tuple[str, str, str]] = [
    # ── ACTION ─────────────────────────────────────────────────────────────
    ("hA6hldpSTF8", "avengers_age_of_ultron",          "action"),
    ("QwievZ1Tx-8", "captain_america_civil_war",        "action"),
    ("0wBe0B3RhXY", "black_panther",                    "action"),
    ("6ZfuNTqbHE8", "dunkirk",                          "action"),
    ("EXeTwQWrcwY", "mad_max_fury_road",                 "action"),
    ("7ANSMZsrNpM", "thor_ragnarok",                    "action"),
    ("d9MyW72ELq0", "iron_man",                         "action"),
    ("tmeOjFno6Do", "the_dark_knight",                  "action"),
    ("hWOvRjJEJC4", "guardians_of_the_galaxy",          "action"),
    ("FGnKMQNcNMk", "mission_impossible_fallout",        "action"),

    # ── SCI-FI ─────────────────────────────────────────────────────────────
    ("2LqzF5WauAw", "inception",                        "scifi"),
    ("zSWdZVtXT7E", "interstellar",                     "scifi"),
    ("rrr_8kMZyoA", "arrival",                          "scifi"),
    ("9GkVhgIUFpo", "blade_runner_2049",                 "scifi"),
    ("ByXuk9QqQkk", "dune",                              "scifi"),
    ("sGbxmsDFVnE", "the_martian",                      "scifi"),
    ("7nMSM1dRIhk", "gravity",                          "scifi"),
    ("j9n-k4p_GVc", "annihilation",                     "scifi"),

    # ── DRAMA ──────────────────────────────────────────────────────────────
    ("q0qD2K2RWkc", "the_revenant",                     "drama"),
    ("uYPbbksJxIg", "whiplash",                         "drama"),
    ("5WpRHSGCl4o", "1917",                              "drama"),
    ("sj9J2ecsSpo", "the_social_network",               "drama"),
    ("gCcOzvAMRVU", "manchester_by_the_sea",             "drama"),
    ("s78cvVyjO9w", "parasite",                         "drama"),
    ("cNi_HC839Wo", "nomadland",                        "drama"),

    # ── COMEDY ─────────────────────────────────────────────────────────────
    ("mVjYG9TSN88", "the_grand_budapest_hotel",         "comedy"),
    ("FTajt2vS7vw", "knives_out",                       "comedy"),
    ("AtNHHRUiPpo", "game_night",                       "comedy"),
    ("RDeubpnMFuk", "superbad",                         "comedy"),
    ("6hB3S9bIaco", "bridesmaids",                      "comedy"),

    # ── HORROR ─────────────────────────────────────────────────────────────
    ("V6wWKNij93o", "hereditary",                       "horror"),
    ("cGNf6J7xlO0", "a_quiet_place",                    "horror"),
    ("5Uc6SFgOS5o", "get_out",                          "horror"),
    ("G98GFgQGRmE", "midsommar",                        "horror"),
    ("jUhiAEAZOas", "the_conjuring",                    "horror"),

    # ── ANIMATION ──────────────────────────────────────────────────────────
    ("YoHD9XEInc0", "spider_man_into_spider_verse",     "animation"),
    ("KQaOq-DhPLk", "soul",                             "animation"),
    ("CanCiAGDtS4", "moana",                            "animation"),
    ("CimadB3o5BE", "frozen",                           "animation"),
    ("JQ3qgPy9CuY", "onward",                           "animation"),

    # ── THRILLER ───────────────────────────────────────────────────────────
    ("qMH3sWolBCo", "gone_girl",                        "thriller"),
    ("ImMFOE2YiQQ", "nocturnal_animals",                "thriller"),
    ("MrTprcneHBk", "prisoners",                        "thriller"),
    ("uYPbbksJxIg", "zodiac",                           "thriller"),

    # ── DOCUMENTARY / BIOGRAPHICAL ─────────────────────────────────────────
    ("Y0v9cK2N1U8", "bohemian_rhapsody",                "biopic"),
    ("l6M9NVhN2wM", "judy",                             "biopic"),
    ("aUwmJqr2Yjs", "the_theory_of_everything",         "biopic"),
]


@dataclass
class DownloadResult:
    youtube_id: str
    title_slug: str
    genre: str
    mp4_path: str
    srt_path: str
    duration_sec: float
    success: bool
    error: str = ""


def get_ytdlp_path() -> str:
    """Find yt-dlp in the current environment."""
    import shutil
    ytdlp = shutil.which("yt-dlp")
    if ytdlp:
        return ytdlp
    # Try the venv's bin directly
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")


def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil
        ff = shutil.which("ffmpeg")
        if ff:
            return ff
        raise RuntimeError("ffmpeg not found")


def download_trailer(
    youtube_id: str,
    title_slug: str,
    genre: str,
    output_dir: Path,
    ytdlp_path: str,
    ffmpeg_path: str,
    max_height: int = 480,
) -> DownloadResult:
    """Download one trailer + auto-generated English subtitles."""
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    mp4_path = output_dir / f"{title_slug}.mp4"
    srt_path = output_dir / f"{title_slug}.srt"

    if mp4_path.exists() and srt_path.exists():
        # Already downloaded; just measure duration
        try:
            import soundfile as sf
            import subprocess as _sp
            # Quick duration via ffprobe-style probe
            probe = _sp.run(
                [ffmpeg_path, "-i", str(mp4_path), "-f", "null", "-"],
                capture_output=True, text=True, timeout=30
            )
            dur_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", probe.stderr)
            dur = 0.0
            if dur_match:
                h, m, s = dur_match.groups()
                dur = int(h)*3600 + int(m)*60 + float(s)
        except Exception:
            dur = 0.0
        print(f"  [skip] {title_slug} already downloaded")
        return DownloadResult(
            youtube_id=youtube_id, title_slug=title_slug, genre=genre,
            mp4_path=str(mp4_path), srt_path=str(srt_path),
            duration_sec=dur, success=True,
        )

    # yt-dlp command: download best audio+video ≤ max_height, + auto subtitles
    tmp_template = str(output_dir / f"{title_slug}.%(ext)s")
    cmd = [
        ytdlp_path,
        url,
        "--ffmpeg-location", str(Path(ffmpeg_path).parent),
        "-f", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--convert-subs", "srt",
        "--no-playlist",
        "--no-keep-video",
        "--merge-output-format", "mp4",
        "-o", tmp_template,
        "--quiet",
        "--no-warnings",
    ]

    print(f"  Downloading: {title_slug} ({genre})  [{youtube_id}]")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            return DownloadResult(
                youtube_id=youtube_id, title_slug=title_slug, genre=genre,
                mp4_path="", srt_path="", duration_sec=0.0, success=False,
                error=result.stderr[:300],
            )

        # Find the auto-generated srt (yt-dlp names it *.en.srt or *.en-auto.srt)
        srt_candidates = list(output_dir.glob(f"{title_slug}*.srt"))
        if srt_candidates:
            srt_cand = srt_candidates[0]
            if srt_cand != srt_path:
                srt_cand.rename(srt_path)

        # Probe duration
        probe = subprocess.run(
            [ffmpeg_path, "-i", str(mp4_path), "-f", "null", "-"],
            capture_output=True, text=True, timeout=30
        )
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", probe.stderr)
        dur = 0.0
        if dur_match:
            h, m, s = dur_match.groups()
            dur = int(h)*3600 + int(m)*60 + float(s)

        print(f"    ✓ {title_slug}  dur={dur:.0f}s  ({elapsed:.0f}s download)")
        return DownloadResult(
            youtube_id=youtube_id, title_slug=title_slug, genre=genre,
            mp4_path=str(mp4_path), srt_path=str(srt_path) if srt_path.exists() else "",
            duration_sec=dur, success=True,
        )

    except subprocess.TimeoutExpired:
        return DownloadResult(
            youtube_id=youtube_id, title_slug=title_slug, genre=genre,
            mp4_path="", srt_path="", duration_sec=0.0, success=False,
            error="Timeout (>120s)",
        )
    except Exception as e:
        return DownloadResult(
            youtube_id=youtube_id, title_slug=title_slug, genre=genre,
            mp4_path="", srt_path="", duration_sec=0.0, success=False,
            error=str(e),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="trailers")
    parser.add_argument("--limit", type=int, default=20, help="Max trailers to download")
    parser.add_argument("--genre", default=None, help="Filter by genre")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ytdlp   = get_ytdlp_path()
    ffmpeg  = get_ffmpeg_path()

    print(f"yt-dlp:  {ytdlp}")
    print(f"ffmpeg:  {ffmpeg}")
    print(f"Output:  {output_dir.resolve()}")

    trailer_list = TRAILERS
    if args.genre:
        trailer_list = [(yt_id, slug, g) for yt_id, slug, g in trailer_list if g == args.genre]
    trailer_list = trailer_list[: args.limit]

    print(f"\nDownloading {len(trailer_list)} trailers …\n")

    results = []
    for yt_id, slug, genre in trailer_list:
        res = download_trailer(yt_id, slug, genre, output_dir, ytdlp, ffmpeg)
        results.append(asdict(res))
        time.sleep(1)  # be polite to YouTube

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    ok = sum(1 for r in results if r["success"])
    with_srt = sum(1 for r in results if r["success"] and r["srt_path"])
    print(f"\nDone: {ok}/{len(results)} downloaded  {with_srt} have SRT  → {manifest_path}")


if __name__ == "__main__":
    main()
