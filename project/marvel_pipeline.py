"""
End-to-end auto-subtitling pipeline for the Marvel Avengers trailer.

Extracts audio via ffmpeg, runs all three VAD methods, applies adaptive
enhancement, and transcribes with latency tracking.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

from project.audio_utils import (
    TARGET_SR,
    VAD_METHODS,
    enhance_audio,
    extract_features,
    should_enhance,
)
from project.asr import WhisperASR
from project.metrics import SubtitleCue, cps_violations
from project.subtitles import write_srt


def get_ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio_from_video(video_path: Path, output_wav: Path) -> None:
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-vn",
        "-acodec", "pcm_s16le",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)


def run_trailer_pipeline(
    video_path: Path,
    output_dir: Path,
    asr: WhisperASR,
    vad_method: str = "spectral",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp_audio"
    tmp_dir.mkdir(exist_ok=True)

    # ── 1. Extract audio ──────────────────────────────────────────────────
    t_extract_start = time.perf_counter()
    wav_path = output_dir / "trailer_audio.wav"
    extract_audio_from_video(video_path, wav_path)
    audio = load_audio(wav_path)
    audio_duration = len(audio) / TARGET_SR
    extract_time_ms = (time.perf_counter() - t_extract_start) * 1000.0

    print(f"  [trailer] Audio extracted: {audio_duration:.1f}s  ({extract_time_ms:.0f}ms)")

    # ── 2. VAD ────────────────────────────────────────────────────────────
    vad_fn = VAD_METHODS[vad_method]
    vad_result = vad_fn(audio)
    print(
        f"  [trailer] VAD ({vad_method}): {len(vad_result.segments)} segments  "
        f"speech_ratio={vad_result.speech_ratio:.2f}  "
        f"compute={vad_result.compute_time_ms:.0f}ms"
    )

    # ── 3. ASR with adaptive enhancement ─────────────────────────────────
    asr.reset_stats()
    cues: list[SubtitleCue] = []
    segment_records = []

    for idx, (start, end) in enumerate(vad_result.segments):
        seg = audio[int(start * TARGET_SR) : int(end * TARGET_SR)]
        if len(seg) < int(0.2 * TARGET_SR):
            continue

        # Adaptive routing
        feats = extract_features(seg)
        use_enh = should_enhance(feats)
        proc_seg = enhance_audio(seg) if use_enh else seg
        enhancement_label = "enhanced" if use_enh else "raw"

        seg_path = tmp_dir / f"trailer_seg_{idx:04d}_{enhancement_label}.wav"
        pred = asr.transcribe_audio_array(seg_path, proc_seg, TARGET_SR)

        if pred.text:
            cues.append(SubtitleCue(start=start, end=end, text=pred.text))
            segment_records.append(
                {
                    "seg_idx": idx,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration_sec": round(end - start, 3),
                    "enhanced": use_enh,
                    "text": pred.text,
                    "avg_logprob": round(pred.avg_logprob, 4) if not np.isnan(pred.avg_logprob) else None,
                    "inference_ms": round(pred.inference_time_ms, 1),
                    "rtf": round(pred.rtf, 4),
                    "estimated_snr_db": round(feats.estimated_snr_db, 2),
                    "spectral_flatness": round(feats.spectral_flatness, 4),
                }
            )

    # Write SRT
    srt_path = output_dir / "trailer_predicted.srt"
    write_srt(cues, srt_path)

    # ── 4. Collect metrics ────────────────────────────────────────────────
    stats = asr.stats
    results = {
        "video": video_path.name,
        "audio_duration_sec": round(audio_duration, 2),
        "vad_method": vad_method,
        "vad_segments_detected": len(vad_result.segments),
        "vad_speech_ratio": round(vad_result.speech_ratio, 3),
        "vad_compute_ms": round(vad_result.compute_time_ms, 1),
        "subtitle_cues": len(cues),
        "cps_violations": cps_violations(cues),
        "enhanced_segments": sum(1 for r in segment_records if r["enhanced"]),
        "raw_segments": sum(1 for r in segment_records if not r["enhanced"]),
        "asr_total_audio_sec": round(stats.total_audio_sec, 2),
        "asr_total_inference_ms": round(stats.total_inference_ms, 1),
        "asr_mean_rtf": round(stats.mean_rtf, 4),
        "asr_overall_rtf": round(stats.overall_rtf, 4),
        "audio_extract_ms": round(extract_time_ms, 1),
        "segment_records": segment_records,
    }

    with open(output_dir / "trailer_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  [trailer] {len(cues)} subtitle cues  overall_RTF={results['asr_overall_rtf']:.4f}")
    return results


def plot_trailer_analysis(audio: np.ndarray, vad_results: dict[str, object], output_path: Path) -> None:
    """
    Plot: waveform + spectrogram + VAD comparison for three methods side by side.
    """
    sr = TARGET_SR
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 1, hspace=0.55)

    # 1. Waveform
    ax0 = fig.add_subplot(gs[0])
    times = np.arange(len(audio)) / sr
    ax0.plot(times, audio, linewidth=0.4, color="steelblue", alpha=0.8)
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Marvel Avengers Trailer — Waveform", fontsize=11, fontweight="bold")
    ax0.set_xlim([0, len(audio) / sr])
    ax0.grid(alpha=0.2)

    # 2. Mel spectrogram
    ax1 = fig.add_subplot(gs[1])
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr, hop_length=512, ax=ax1, cmap="magma")
    ax1.set_title("Mel Spectrogram (80 bands)", fontsize=10)
    fig.colorbar(img, ax=ax1, format="%+2.0f dB", pad=0.01)
    ax1.set_xlim([0, len(audio) / sr])

    # 3. VAD comparison bars
    ax2 = fig.add_subplot(gs[2])
    colors = {"energy": "tomato", "mfcc": "mediumseagreen", "spectral": "dodgerblue"}
    for row_idx, (method, vad_res) in enumerate(vad_results.items()):
        y_base = row_idx
        for start, end in vad_res.segments:
            ax2.barh(y_base, end - start, left=start, height=0.6, color=colors[method], alpha=0.75)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Energy VAD", "MFCC VAD", "Spectral VAD"])
    ax2.set_xlim([0, len(audio) / sr])
    ax2.set_title("VAD Method Comparison — Detected Speech Segments", fontsize=10)
    ax2.set_xlabel("Time (s)")
    ax2.grid(alpha=0.2, axis="x")

    # 4. Enhancement routing decisions
    ax3 = fig.add_subplot(gs[3])
    # Plot energy envelope with routing
    hop = 512
    rms_env = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms_env)), sr=sr, hop_length=hop)
    ax3.fill_between(rms_times, rms_env, alpha=0.5, color="grey", label="RMS energy")
    # Mark which VAD segments would be enhanced
    spec_vad = vad_results.get("spectral")
    if spec_vad:
        for start, end in spec_vad.segments:
            seg = audio[int(start * sr): int(end * sr)]
            if len(seg) > 100:
                from project.audio_utils import extract_features, should_enhance
                feats = extract_features(seg)
                color = "orangered" if should_enhance(feats) else "limegreen"
                ax3.axvspan(start, end, alpha=0.25, color=color)
    ax3.set_xlim([0, len(audio) / sr])
    ax3.set_title("Adaptive Enhancement Routing (red=enhance, green=raw)", fontsize=10)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("RMS")
    ax3.grid(alpha=0.2)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved {output_path.name}")
