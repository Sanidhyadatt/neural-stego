"""Dataset module for Neural Audio Steganography.

Provides:
- StegoDataset: PyTorch Dataset extracting random 2-second peak-normalized audio chunks
  and paired 128-bit random secret messages.
- generate_synthetic_data: Generates mixed sine waves + noise WAV files for smoke tests
  and offline training/verification.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import List, Optional, Tuple, Union

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

    # Deterministic generation for reproducibility
    rng = np.random.RandomState(42)

    for i in range(num_files):
        # Durations between 3.0 and 5.0 seconds
        duration = rng.uniform(3.0, 5.0)
        num_samples = int(duration * sample_rate)
        t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float32)

        # 3 random harmonically rich frequencies
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

        # Subtle white noise
        noise = rng.normal(0.0, 0.05, num_samples).astype(np.float32)
        combined = signal + noise

        # Peak normalize to 0.90
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

    Recursively scans audio files, verifies minimum duration of 0.5s, and extracts
    random 2-second chunks (32,000 samples) paired with 128-bit random message tensors.

    Args:
        root_dir: Root directory containing audio files.
        split: Dataset split ('train', 'val', 'test', or 'all').
        auto_generate_synthetic: If True and root_dir has no files, generate synthetic data.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        auto_generate_synthetic: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.file_list: List[Path] = []

        # Collect files
        all_found = self._scan_audio_files(self.root_dir)

        if len(all_found) == 0 and auto_generate_synthetic:
            # Generate 20 synthetic files into root_dir / "synthetic"
            synthetic_dir = self.root_dir / "synthetic"
            generate_synthetic_data(synthetic_dir, num_files=20)
            all_found = self._scan_audio_files(self.root_dir)

        if len(all_found) == 0:
            raise FileNotFoundError(
                f"No supported audio files {SUPPORTED_INPUT_EXTENSIONS} found in {self.root_dir}."
            )

        # Partition files based on split
        self.file_list = self._partition_split(all_found, self.split)

    def _scan_audio_files(self, root: Path) -> List[Path]:
        """Recursively search for supported audio files and cache valid candidates."""
        if not root.exists():
            return []

        candidates: List[Path] = []
        extensions = set(SUPPORTED_INPUT_EXTENSIONS)

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in extensions:
                    candidate = Path(dirpath) / fname
                    # Fast check: skip zero-byte or unreadable files
                    if candidate.stat().st_size > 0:
                        candidates.append(candidate)

        return sorted(candidates)

    def _partition_split(self, files: List[Path], split: str) -> List[Path]:
        """Split file list into train/val/test partitions."""
        # If a dedicated subfolder for the split exists, use those files
        split_dir = self.root_dir / split
        if split_dir.is_dir():
            split_files = [f for f in files if split_dir in f.parents or f.parent == split_dir]
            if len(split_files) > 0:
                return split_files

        if split == "all" or len(files) <= 2:
            return files

        n_total = len(files)
        n_train = max(1, int(0.80 * n_total))
        n_val = max(1, int(0.10 * n_total))

        if split == "train":
            return files[:n_train]
        elif split in ("val", "valid", "validation"):
            return files[n_train : n_train + n_val]
        elif split == "test":
            return files[n_train + n_val :]
        else:
            return files

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio chunk and paired message bits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - audio_chunk: [32000] float32 tensor peak-normalized.
                - message_bits: [128] float32 binary tensor {0.0, 1.0}.
        """
        for _ in range(5):  # Max 5 fallback retries if an audio file is too short
            file_path = self.file_list[idx]
            try:
                wave = load_audio(file_path)
            except Exception:
                # If corrupted, pick another index
                idx = random.randint(0, len(self.file_list) - 1)
                continue

            # Skip files shorter than 0.5 s (8000 samples)
            if wave.shape[0] < MIN_AUDIO_SAMPLES:
                idx = random.randint(0, len(self.file_list) - 1)
                continue

            # Extract random 2-second chunk (32,000 samples)
            audio_len = wave.shape[0]
            if audio_len > CHUNK_SAMPLES:
                max_offset = audio_len - CHUNK_SAMPLES
                start_offset = random.randint(0, max_offset)
                chunk = wave[start_offset : start_offset + CHUNK_SAMPLES]
            else:
                # Zero-pad if shorter than 2 seconds
                pad_len = CHUNK_SAMPLES - audio_len
                chunk = torch.nn.functional.pad(wave, (0, pad_len), mode="constant", value=0.0)

            # Peak-normalize to 0.95
            peak = chunk.abs().max()
            if peak > 1e-6:
                chunk = (chunk / peak) * 0.95

            # Random 128-bit secret message
            message_bits = torch.randint(0, 2, (PAYLOAD_BITS,)).float()

            return chunk, message_bits

        # Fallback if multiple files failed: return synthetic sine wave
        t = torch.linspace(0, 2.0, CHUNK_SAMPLES)
        chunk = 0.5 * torch.sin(2 * np.pi * 440.0 * t)
        message_bits = torch.randint(0, 2, (PAYLOAD_BITS,)).float()
        return chunk, message_bits
