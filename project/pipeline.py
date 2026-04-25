from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from project.asr import WhisperASR
from project.audio_utils import (
    TARGET_SR,
    detect_speech_regions,
    enhance_audio,
    extract_features,
    mix_with_snr,
    should_enhance,
    synthesize_noise,
)
from project.dataset import Sample, prepare_librispeech_subset
from project.metrics import SubtitleCue, compute_cer, compute_wer, cps_violations, timestamp_mae
from project.subtitles import write_srt


SNR_LEVELS = [20, 10, 0]
NOISE_TYPES = ["white", "pink", "soundtrack"]
SYSTEMS = ["raw", "enhanced", "adaptive"]


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    if sr != TARGET_SR:
        raise RuntimeError(f"Expected sample rate {TARGET_SR}, got {sr}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)


def choose_audio(system: str, mixed_audio: np.ndarray) -> tuple[np.ndarray, bool]:
    if system == "raw":
        return mixed_audio, False
    if system == "enhanced":
        return enhance_audio(mixed_audio), True
    features = extract_features(mixed_audio)
    use_enhancement = should_enhance(features)
    return (enhance_audio(mixed_audio) if use_enhancement else mixed_audio), use_enhancement


def build_movie_reel(samples: list[Sample], output_audio_path: Path) -> tuple[np.ndarray, list[SubtitleCue]]:
    pieces = []
    cues = []
    current_time = 0.0

    for idx, sample in enumerate(samples[:6]):
        silence_before = 0.35 + 0.08 * idx
        silence_after = 0.5 + 0.05 * idx
        pre = np.zeros(int(TARGET_SR * silence_before), dtype=np.float32)
        post = np.zeros(int(TARGET_SR * silence_after), dtype=np.float32)
        speech = load_audio(sample.audio_path)
        background = synthesize_noise("soundtrack", len(speech), seed=100 + idx)
        mixed_speech = mix_with_snr(speech, background, 8)

        pieces.extend([pre, mixed_speech, post])
        start = current_time + silence_before
        end = start + len(mixed_speech) / TARGET_SR
        cues.append(SubtitleCue(start=start, end=end, text=sample.text))
        current_time = end + silence_after

    full_audio = np.concatenate(pieces)
    sf.write(output_audio_path, full_audio, TARGET_SR)
    return full_audio, cues


def run_benchmark(samples: list[Sample], asr: WhisperASR, output_dir: Path) -> pd.DataFrame:
    rows = []
    temp_dir = output_dir / "tmp_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, sample in enumerate(samples):
        clean_audio = load_audio(sample.audio_path)
        for noise_type in NOISE_TYPES:
            noise = synthesize_noise(noise_type, len(clean_audio), seed=sample_idx + len(noise_type))
            for snr_db in SNR_LEVELS:
                mixed = mix_with_snr(clean_audio, noise, snr_db)
                for system in SYSTEMS:
                    chosen_audio, used_enhancement = choose_audio(system, mixed)
                    temp_path = temp_dir / f"{sample.sample_id}_{noise_type}_{snr_db}_{system}.wav"
                    prediction = asr.transcribe_audio_array(temp_path, chosen_audio, TARGET_SR)
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "noise_type": noise_type,
                            "snr_db": snr_db,
                            "system": system,
                            "used_enhancement": used_enhancement,
                            "reference": sample.text,
                            "hypothesis": prediction.text,
                            "wer": compute_wer(sample.text, prediction.text),
                            "cer": compute_cer(sample.text, prediction.text),
                            "duration_sec": sample.duration,
                            "avg_logprob": prediction.avg_logprob,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "condition_metrics.csv", index=False)
    return df


def run_subtitle_experiment(samples: list[Sample], asr: WhisperASR, output_dir: Path) -> dict:
    reel_path = output_dir / "synthetic_movie_reel.wav"
    reel_audio, reference_cues = build_movie_reel(samples, reel_path)
    predicted_regions = detect_speech_regions(reel_audio)

    predicted_cues = []
    for idx, (start, end) in enumerate(predicted_regions):
        seg_audio = reel_audio[int(start * TARGET_SR) : int(end * TARGET_SR)]
        if len(seg_audio) < int(0.25 * TARGET_SR):
            continue
        seg_path = output_dir / "tmp_audio" / f"subtitle_segment_{idx}.wav"
        prediction = asr.transcribe_audio_array(seg_path, seg_audio, TARGET_SR)
        if prediction.text:
            predicted_cues.append(SubtitleCue(start=start, end=end, text=prediction.text))

    write_srt(reference_cues, output_dir / "reference_subtitles.srt")
    write_srt(predicted_cues, output_dir / "predicted_subtitles.srt")

    results = {
        "reference_cues": len(reference_cues),
        "predicted_cues": len(predicted_cues),
        "timestamp_mae_sec": timestamp_mae(reference_cues, predicted_cues),
        "reference_cps_violations": cps_violations(reference_cues),
        "predicted_cps_violations": cps_violations(predicted_cues),
    }

    with open(output_dir / "subtitle_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def make_figures(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.groupby(["system", "snr_db"], as_index=False)
        .agg(mean_wer=("wer", "mean"), mean_cer=("cer", "mean"))
        .sort_values(["system", "snr_db"])
    )
    summary.to_csv(output_dir / "system_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    for system in SYSTEMS:
        subset = summary[summary["system"] == system]
        plt.plot(subset["snr_db"], subset["mean_wer"], marker="o", label=system)
    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean WER")
    plt.title("ASR Robustness on Synthetic Movie Noise")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wer_by_snr.png", dpi=160)
    plt.close()

    noise_summary = (
        df.groupby(["system", "noise_type"], as_index=False)
        .agg(mean_wer=("wer", "mean"))
        .sort_values(["noise_type", "system"])
    )
    plt.figure(figsize=(8, 5))
    for offset, system in enumerate(SYSTEMS):
        subset = noise_summary[noise_summary["system"] == system]
        x = np.arange(len(NOISE_TYPES)) + (offset - 1) * 0.22
        plt.bar(x, subset["mean_wer"], width=0.22, label=system)
    plt.xticks(np.arange(len(NOISE_TYPES)), NOISE_TYPES)
    plt.ylabel("Mean WER")
    plt.title("ASR Performance by Noise Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wer_by_noise.png", dpi=160)
    plt.close()


def write_report(df: pd.DataFrame, subtitle_results: dict, output_dir: Path) -> None:
    summary = pd.read_csv(output_dir / "system_summary.csv")
    best_overall = (
        df.groupby("system", as_index=False).agg(mean_wer=("wer", "mean"), mean_cer=("cer", "mean"))
        .sort_values("mean_wer")
        .iloc[0]
    )

    report = f"""# EE 679 Auto-Subtitling Experiment Report

## Setup
- Speech dataset: LibriSpeech `test-clean` subset (8 samples, 1 sample per speaker)
- Noise conditions: white, pink, soundtrack
- SNR levels: 20 dB, 10 dB, 0 dB
- ASR model: faster-whisper `tiny.en`
- Front-end systems: raw, enhanced (Wiener), adaptive (MFCC + spectral rules)

## Main Findings
- Best overall system: **{best_overall['system']}**
- Mean WER of best system: **{best_overall['mean_wer']:.3f}**
- Mean CER of best system: **{best_overall['mean_cer']:.3f}**
- Subtitle timestamp MAE: **{subtitle_results['timestamp_mae_sec']:.3f} sec**
- Predicted subtitle CPS violations: **{subtitle_results['predicted_cps_violations']}**

## System Summary
```text
{summary.to_string(index=False)}
```

## Subtitle Metrics
```json
{json.dumps(subtitle_results, indent=2)}
```
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tmp_audio").mkdir(parents=True, exist_ok=True)

    samples = prepare_librispeech_subset(data_dir, max_items=8)
    asr = WhisperASR(model_size="tiny.en")

    df = run_benchmark(samples, asr, output_dir)
    subtitle_results = run_subtitle_experiment(samples, asr, output_dir)
    make_figures(df, output_dir)
    write_report(df, subtitle_results, output_dir)

    print(f"Wrote results to {output_dir}")
