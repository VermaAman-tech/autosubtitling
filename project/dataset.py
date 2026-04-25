from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

from project.audio_utils import TARGET_SR, normalize_audio


LIBRISPEECH_TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


@dataclass
class Sample:
    sample_id: str
    speaker_id: int
    chapter_id: int
    utterance_id: int
    text: str
    audio_path: Path
    duration: float


def _save_resampled_audio(audio, original_sr: int, out_path: Path) -> float:
    audio_np = np.asarray(audio, dtype=np.float32)
    if original_sr != TARGET_SR:
        audio_np = librosa.resample(audio_np, orig_sr=original_sr, target_sr=TARGET_SR)
    audio_np = normalize_audio(audio_np.astype(np.float32))
    sf.write(out_path, audio_np, TARGET_SR)
    return len(audio_np) / TARGET_SR


def _download_if_needed(archive_path: Path) -> None:
    if archive_path.exists():
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(LIBRISPEECH_TEST_CLEAN_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(archive_path, "wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading LibriSpeech") as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))


def _extract_if_needed(archive_path: Path, raw_root: Path) -> Path:
    extracted_root = raw_root / "LibriSpeech" / "test-clean"
    if extracted_root.exists():
        return extracted_root
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=raw_root)
    return extracted_root


def prepare_librispeech_subset(root: Path, max_items: int = 12) -> list[Sample]:
    dataset_root = root / "raw"
    processed_root = root / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "downloads" / "test-clean.tar.gz"
    _download_if_needed(archive_path)
    extracted_root = _extract_if_needed(archive_path, dataset_root)

    chosen: list[Sample] = []
    transcript_files = sorted(extracted_root.glob("*/*/*.trans.txt"))

    for transcript_path in tqdm(transcript_files, desc="Selecting LibriSpeech samples"):
        utterances = transcript_path.read_text(encoding="utf-8").strip().splitlines()
        for line in utterances:
            utt_id, transcript = line.split(" ", 1)
            speaker_id, chapter_id, utterance_id = [int(part) for part in utt_id.split("-")]
            if any(item.speaker_id == speaker_id for item in chosen):
                continue

            flac_path = transcript_path.parent / f"{utt_id}.flac"
            if not flac_path.exists():
                continue
            waveform, sr = sf.read(flac_path)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            duration = len(waveform) / sr
            if duration < 3.0 or duration > 10.5:
                continue

            out_path = processed_root / f"{utt_id}.wav"
            duration_sec = _save_resampled_audio(waveform, sr, out_path)
            chosen.append(
                Sample(
                    sample_id=utt_id,
                    speaker_id=speaker_id,
                    chapter_id=chapter_id,
                    utterance_id=utterance_id,
                    text=transcript.strip().lower(),
                    audio_path=out_path,
                    duration=duration_sec,
                )
            )
            if len(chosen) >= max_items:
                break
        if len(chosen) >= max_items:
            break

    if len(chosen) < max_items:
        raise RuntimeError(f"Expected {max_items} samples but only found {len(chosen)}")
    return chosen
