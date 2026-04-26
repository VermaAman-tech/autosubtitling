#!/usr/bin/env python3
"""
YouTube VTT → clean SRT converter (v2)
========================================
Extracts word-level timestamps from YouTube's rolling auto-caption VTT format.
The key insight: every word added to the screen has a <HH:MM:SS.mmm><c> tag.
We collect all these words with their exact timestamps, deduplicate, and
group into subtitle cues.
"""
from __future__ import annotations
import re, sys
from pathlib import Path


def _ts(t: str) -> float:
    t = t.replace(",", ".")
    p = t.split(":")
    return int(p[0])*3600 + int(p[1])*60 + float(p[2])


def _fmt(sec: float) -> str:
    h = int(sec//3600); m = int((sec%3600)//60); s = sec%60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _strip_junk(text: str) -> str:
    """Remove HTML tags, VTT positioning, music tags."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\b(align|position|line|size):[^\s]+", "", text)
    text = re.sub(r"\[Music\]|\[Applause\]|\[Laughter\]|\[Noise\]", "",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"&amp;", "&", text)
    return " ".join(text.split())


def parse_vtt(vtt_text: str) -> list[tuple[float, float, str]]:
    """
    Parse YouTube VTT and return list of (start, end, word) tuples.

    YouTube VTT block structure:
      <block_start> --> <block_end> [positioning]
      [old rolling text still on screen]
      word_A<WORD_START_TS><c> word_B</c><WORD_START_TS><c> word_C</c>...

    We only extract the word-level <c> tagged text from each block,
    ignoring the first line (which is the already-shown rolling text).
    """
    entries: list[tuple[float, float, str]] = []

    # Remove WEBVTT header
    vtt_text = re.sub(r"^WEBVTT.*?\n", "", vtt_text).strip()

    # Split into blocks (separated by blank lines)
    raw_blocks = re.split(r"\n{2,}", vtt_text)

    for block in raw_blocks:
        lines = [l.rstrip() for l in block.splitlines()]
        lines = [l for l in lines if l]  # remove blank lines within block

        if not lines:
            continue

        # Find timing line
        timing_idx = -1
        t_start = t_end = 0.0
        for i, ln in enumerate(lines):
            m = re.match(
                r"(\d{2}:\d{2}:\d{2}[\.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{3})",
                ln)
            if m:
                t_start = _ts(m.group(1))
                t_end   = _ts(m.group(2))
                timing_idx = i
                break

        if timing_idx < 0 or t_end - t_start < 0.05:
            continue  # skip header-only or ultra-short blocks

        content_lines = lines[timing_idx+1:]
        if not content_lines:
            continue

        # ── Extract word-level <c> entries from content lines ──────────────
        # We look for patterns: sometext<HH:MM:SS.mmm><c> nextword</c>
        # The last non-empty content line is the new text; the first is old text.
        # BUT to be safe, scan ALL content lines for <c> patterns.

        combined = " ".join(content_lines)

        # Pattern: <TIMESTAMP><c>word(s)</c>
        word_ts_re = re.compile(
            r"<(\d{2}:\d{2}:\d{2}[\.,]\d{3})><c>(.*?)</c>",
            re.DOTALL,
        )

        # Also get text BEFORE the first timestamp (leading word at block_start)
        first_ts_match = re.search(r"<\d{2}:\d{2}:\d{2}[\.,]\d{3}>", combined)
        if first_ts_match:
            leading = combined[:first_ts_match.start()]
            leading = _strip_junk(leading)
            # Only use last-line leading text (first line is old rolling text)
            leading_parts = leading.split()
            # Heuristic: if the leading text is more than 8 words, take last 4
            # (earlier words are from rolling-off old line)
            if len(leading_parts) > 8:
                leading_parts = leading_parts[-4:]
            if leading_parts:
                entries.append((t_start, t_start + 0.5, " ".join(leading_parts)))

        for m in word_ts_re.finditer(combined):
            word_ts   = _ts(m.group(1))
            word_text = _strip_junk(m.group(2))
            if word_text:
                entries.append((word_ts, word_ts + 0.8, word_text))

    return entries


def entries_to_srt(
    entries: list[tuple[float, float, str]],
    max_words: int = 10,
    gap_threshold: float = 1.0,
) -> str:
    """Group word entries into subtitle cues."""
    if not entries:
        return ""

    # Sort and deduplicate
    entries.sort(key=lambda x: x[0])
    deduped: list[tuple[float, float, str]] = []
    seen_words: list[str] = []
    for start, end, text in entries:
        words = text.split()
        new_words = []
        for w in words:
            w_clean = w.lower().strip(".,!?;:")
            if w_clean and w_clean not in seen_words[-4:]:  # rolling 4-word window
                new_words.append(w)
                seen_words.append(w_clean)
        if new_words:
            deduped.append((start, end, " ".join(new_words)))

    # Group into cues
    cues: list[tuple[float, float, list[str]]] = []
    cue_words: list[str] = []
    cue_start = cue_end = 0.0

    for start, end, text in deduped:
        if not cue_words:
            cue_start = start
        elif (start - cue_end > gap_threshold) or len(cue_words) >= max_words:
            cues.append((cue_start, cue_end, list(cue_words)))
            cue_words = []
            cue_start = start

        cue_words.extend(text.split())
        cue_end = max(end, start + 0.8)

    if cue_words:
        cues.append((cue_start, cue_end, cue_words))

    # Format
    srt_lines = []
    for i, (s, e, words) in enumerate(cues, 1):
        if e <= s:
            e = s + 1.0
        text = " ".join(words).strip()
        if not text:
            continue
        srt_lines += [str(i), f"{_fmt(s)} --> {_fmt(e)}", text, ""]

    return "\n".join(srt_lines)


def convert_all(trailers_dir: Path) -> None:
    count = 0
    for vtt in sorted(trailers_dir.glob("*.vtt")):
        slug = re.sub(r"\.en$", "", vtt.stem)
        srt_out = trailers_dir / f"{slug}.srt"
        vtt_text = vtt.read_text(encoding="utf-8", errors="replace")
        entries  = parse_vtt(vtt_text)
        srt      = entries_to_srt(entries)
        if srt:
            srt_out.write_text(srt, encoding="utf-8")
            n = srt.count("-->")
            sample = " ".join(w for _, _, t in entries[:8] for w in t.split())[:80]
            print(f"  ✓ {slug}: {n} cues  [{sample}]")
            count += 1
        else:
            print(f"  ✗ {slug}: 0 words extracted")
    print(f"\nConverted {count} files")


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trailers")
    convert_all(d)
