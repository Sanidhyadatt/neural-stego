"""Dataset module for Neural Audio Steganography.

Provides:
- StegoDataset: PyTorch Dataset scanning one or multiple audio directories, holding out
  2% of files as a validation split, skipping silent chunks (RMS < 0.005), applying
  peak normalization and random training gain, and pairing with 128-bit random payloads.
- generate_synthetic_data: Generates mixed sine waves + noise WAV files for smoke tests
  and offline verification.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import List, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ml.audio_utils import load_audio
from ml.config import (
    CHUNK_SAMPLES,
    PAYLOAD_BITS,
    SAMPLE_RATE,
    SUPPORTED_INPUT_EXTENSIONS,
)

MIN_AUDIO_DURATION_SEC = 0.5
MIN_AUDIO_SAMPLES = int(MIN_AUDIO_DURATION_SEC * SAMPLE_RATE)  # 8000 samples
MIN_CHUNK_RMS = 0.005
VAL_HOLDOUT_RATIO = 0.02


def generate_synthetic_data(
    output_dir: Union[str, Path],
    num_files: int = 20,
    sample_rate: int = SAMPLE_RATE,
) -> List[Path]:
    """Generate synthetic WAV files consisting of harmonic sine waves and noise.

    Args:
        output_dir: Directory where synthetic audio files will be saved.
        num_files: Number of audio files to generate (default 20).
        sample_rate: Audio sampling frequency in Hz (default 16000).

    Returns:
        List[Path]: Paths of generated WAV files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    rng = np.random.RandomState(42)

    for i in range(num_files):
        duration = rng.uniform(3.0, 5.0)
        num_samples = int(duration * sample_rate)
        t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float32)

        f1 = rng.uniform(150.0, 400.0)
        f2 = rng.uniform(400.0, 1000.0)
        f3 = rng.uniform(1000.0, 2500.0)

        phase1 = rng.uniform(0.0, 2 * np.pi)
        phase2 = rng.uniform(0.0, 2 * np.pi)
        phase3 = rng.uniform(0.0, 2 * np.pi)

        signal = (
            0.4 * np.sin(2 * np.pi * f1 * t + phase1)
            + 0.3 * np.sin(2 * np.pi * f2 * t + phase2)
            + 0.2 * np.sin(2 * np.pi * f3 * t + phase3)
        )

        noise = rng.normal(0.0, 0.05, num_samples).astype(np.float32)
        combined = signal + noise

        peak = np.max(np.abs(combined))
        if peak > 1e-6:
            combined = (combined / peak) * 0.90

        file_path = out_path / f"synthetic_{i:03d}.wav"
        sf.write(
            str(file_path),
            combined,
            samplerate=sample_rate,
            subtype="PCM_16",
            format="WAV",
        )
        generated_files.append(file_path)

    return generated_files


class StegoDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Audio Steganography Dataset.

    Scans one or multiple root directories for audio files, holds out 2% of files for
    validation, skips silent chunks (RMS < 0.005), peak-normalizes chunks, applies random
    gain on training chunks, and pairs each chunk with a random 128-bit payload.

    Args:
        root_dir: Single directory path or sequence of directory paths containing audio files.
        split: Dataset split ('train', 'val', or 'all').
        auto_generate_synthetic: If True and root_dir has no files, generate synthetic data.
    """

    def __init__(
        self,
        root_dir: Union[str, Path, Sequence[Union[str, Path]]],
        split: str = "train",
        auto_generate_synthetic: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(root_dir, (str, Path)):
            self.root_dirs: List[Path] = [Path(root_dir)]
        else:
            self.root_dirs = [Path(p) for p in root_dir]
        if not self.root_dirs:
            raise ValueError("At least one root_dir must be provided.")

        self.root_dir = self.root_dirs[0]
        self.split = split.lower()
        self.file_list: List[Path] = []

        all_found = self._scan_audio_dirs(self.root_dirs)

        if len(all_found) == 0 and auto_generate_synthetic:
            synthetic_dir = self.root_dirs[0] / "synthetic"
            generate_synthetic_data(synthetic_dir, num_files=20)
            all_found = self._scan_audio_dirs(self.root_dirs)

        if len(all_found) == 0:
            raise FileNotFoundError(
                f"No supported audio files {SUPPORTED_INPUT_EXTENSIONS} found in {self.root_dirs}."
            )

        self.file_list = self._partition_split(all_found, self.split)

    def _scan_audio_dirs(self, roots: Sequence[Path]) -> List[Path]:
        """Recursively scan multiple directories for supported non-empty audio files."""
        seen: set[Path] = set()
        candidates: List[Path] = []
        extensions = set(SUPPORTED_INPUT_EXTENSIONS)

        for root in roots:
            if not root.exists():
                continue
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    ext = Path(fname).suffix.lower()
                    if ext in extensions:
                        candidate = Path(dirpath) / fname
                        resolved = candidate.resolve()
                        if resolved not in seen and candidate.stat().st_size > 0:
                            seen.add(resolved)
                            candidates.append(candidate)

        return sorted(candidates, key=lambda p: str(p))

    def _partition_split(self, files: List[Path], split: str) -> List[Path]:
        """Partition sorted file list into first 98% train and last 2% validation splits."""
        if split == "all" or len(files) <= 1:
            return files

        n_total = len(files)
        n_val = max(1, int(round(VAL_HOLDOUT_RATIO * n_total)))
        n_val = min(n_val, n_total - 1)

        if split in ("val", "valid", "validation"):
            return files[-n_val:]
        elif split == "train":
            return files[:-n_val]
        else:
            return files

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def _extract_non_silent_chunk(wave: torch.Tensor, max_tries: int = 8) -> torch.Tensor | None:
        """Sample a 2-second chunk from wave whose RMS is at least MIN_CHUNK_RMS (0.005)."""
        audio_len = wave.shape[0]
        if audio_len <= CHUNK_SAMPLES:
            pad_len = CHUNK_SAMPLES - audio_len
            chunk = (
                torch.nn.functional.pad(wave, (0, pad_len), mode="constant", value=0.0)
                if pad_len > 0
                else wave
            )
            rms = float(torch.sqrt(torch.mean(chunk ** 2)).item())
            return chunk if rms >= MIN_CHUNK_RMS else None

        max_offset = audio_len - CHUNK_SAMPLES
        for _ in range(max_tries):
            start_offset = random.randint(0, max_offset)
            chunk = wave[start_offset : start_offset + CHUNK_SAMPLES]
            rms = float(torch.sqrt(torch.mean(chunk ** 2)).item())
            if rms >= MIN_CHUNK_RMS:
                return chunk

        return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load non-silent audio chunk, peak-normalize, apply random train gain, and pair bits."""
        is_train = self.split == "train"

        for _ in range(10):
            file_path = self.file_list[idx]
            try:
                wave = load_audio(file_path)
            except Exception:
                idx = random.randint(0, len(self.file_list) - 1)
                continue

            if wave.shape[0] < MIN_AUDIO_SAMPLES:
                idx = random.randint(0, len(self.file_list) - 1)
                continue

            chunk = self._extract_non_silent_chunk(wave)
            if chunk is None:
                idx = random.randint(0, len(self.file_list) - 1)
                continue

            # Peak normalization to unit peak
            peak = float(chunk.abs().max().item())
            if peak <= 1e-6:
                idx = random.randint(0, len(self.file_list) - 1)
                continue
            chunk = chunk / peak

            if is_train:
                # Apply random gain on each training chunk while ensuring RMS stays >= 0.005
                norm_rms = float(torch.sqrt(torch.mean(chunk ** 2)).item())
                min_gain = min(0.95, max(0.25, (MIN_CHUNK_RMS * 1.1) / max(norm_rms, 1e-6)))
                gain = random.uniform(min_gain, 0.98)
                chunk = chunk * gain
            else:
                chunk = chunk * 0.95

            chunk = chunk.clamp(-1.0, 1.0)
            message_bits = torch.randint(0, 2, (PAYLOAD_BITS,)).float()
            return chunk, message_bits

        # Fallback non-silent synthetic harmonic chunk if every sampled file was silent/invalid
        t = torch.linspace(0.0, 2.0, CHUNK_SAMPLES, dtype=torch.float32)
        chunk = 0.5 * torch.sin(2.0 * np.pi * 440.0 * t)
        if is_train:
            chunk = chunk * random.uniform(0.5, 0.98)
        message_bits = torch.randint(0, 2, (PAYLOAD_BITS,)).float()
        return chunk, message_bits
