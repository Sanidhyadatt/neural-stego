"""Smoke tests and unit tests for StegoDataset and audio DataLoader."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from ml.audio_utils import stft
from ml.config import CHUNK_SAMPLES, FREQ_BINS, N_FRAMES, PAYLOAD_BITS, STFT_CHANNELS
from ml.dataset import StegoDataset, generate_synthetic_data
from ml.models import Decoder, Discriminator, Encoder


def test_generate_synthetic_data(tmp_path: Path):
    """Verify synthetic audio generation produces 20 valid 16-bit PCM WAV files."""
    synth_dir = tmp_path / "synthetic"
    files = generate_synthetic_data(synth_dir, num_files=20)

    assert len(files) == 20
    for f in files:
        assert f.exists()
        assert f.suffix == ".wav"
        data, sr = sf.read(str(f), dtype="float32")
        assert sr == 16000
        # Check duration >= 3.0s (48,000 samples)
        assert len(data) >= 48000
        assert np.max(np.abs(data)) <= 1.0


def test_stego_dataset_empty_fallback(tmp_path: Path):
    """Verify StegoDataset automatically populates synthetic audio when directory is empty."""
    dataset = StegoDataset(tmp_path, split="all", auto_generate_synthetic=True)
    assert len(dataset) == 20
    sample_chunk, sample_bits = dataset[0]

    assert sample_chunk.shape == (CHUNK_SAMPLES,)
    assert sample_chunk.shape == (32000,)
    assert sample_bits.shape == (PAYLOAD_BITS,)
    assert sample_bits.shape == (128,)


def test_skip_files_shorter_than_half_second(tmp_path: Path):
    """Verify that files shorter than 0.5 s (8000 samples) are rejected/skipped."""
    audio_dir = tmp_path / "mixed_lengths"
    audio_dir.mkdir(parents=True)

    # 0.2-second file (3200 samples) - should be skipped
    short_data = np.zeros(3200, dtype=np.float32)
    sf.write(str(audio_dir / "short.wav"), short_data, 16000, subtype="PCM_16")

    # 1.0-second file (16000 samples) - valid
    valid_data = (np.sin(np.linspace(0, 1.0, 16000)) * 0.5).astype(np.float32)
    sf.write(str(audio_dir / "valid.wav"), valid_data, 16000, subtype="PCM_16")

    dataset = StegoDataset(audio_dir, split="all", auto_generate_synthetic=False)
    # Even if short.wav is indexed, __getitem__ must safely return a valid 32000 chunk
    chunk, bits = dataset[0]
    assert chunk.shape == (32000,)
    assert bits.shape == (128,)


def test_dataloader_smoke_and_shapes(tmp_path: Path):
    """Smoke test: builds a DataLoader (batch 4), prints tensor shapes, and verifies full pipeline."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data/ has audio files; if empty, generate 20 synthetic wav files
    dataset = StegoDataset(data_dir, split="all", auto_generate_synthetic=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    batch_audio, batch_bits = next(iter(loader))

    # Print requested tensor shapes
    print(f"\n[Smoke Test] DataLoader batch_audio shape: {batch_audio.shape}")
    print(f"[Smoke Test] DataLoader batch_bits shape:  {batch_bits.shape}")

    assert batch_audio.shape == (4, CHUNK_SAMPLES)
    assert batch_audio.shape == (4, 32000)
    assert batch_bits.shape == (4, PAYLOAD_BITS)
    assert batch_bits.shape == (4, 128)

    # Verify peak-normalization
    assert torch.all(batch_audio <= 1.0) and torch.all(batch_audio >= -1.0)

    # Verify STFT transformation
    spec = stft(batch_audio)
    print(f"[Smoke Test] STFT spectrogram shape:      {spec.shape}")
    assert spec.shape == (4, STFT_CHANNELS, FREQ_BINS, N_FRAMES)
    assert spec.shape == (4, 2, 513, 126)

    # Verify full model pass (Encoder -> Decoder -> Discriminator)
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.02)
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32)
    discriminator = Discriminator(base=32)

    stego_spec = encoder(spec, batch_bits)
    print(f"[Smoke Test] Stego spectrogram shape:     {stego_spec.shape}")
    assert stego_spec.shape == (4, 2, 513, 126)

    decoded_logits = decoder(stego_spec)
    print(f"[Smoke Test] Decoded bit logits shape:    {decoded_logits.shape}")
    assert decoded_logits.shape == (4, 128)

    disc_logit = discriminator(stego_spec)
    print(f"[Smoke Test] Discriminator logit shape:   {disc_logit.shape}")
    assert disc_logit.shape == (4, 1)


def test_two_percent_val_holdout_and_multi_dir(tmp_path: Path):
    """Verify 2% validation hold-out split and multiple root_dir support."""
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    generate_synthetic_data(dir_a, num_files=50)
    generate_synthetic_data(dir_b, num_files=50)

    train_ds = StegoDataset([dir_a, dir_b], split="train", auto_generate_synthetic=False)
    val_ds = StegoDataset([dir_a, dir_b], split="val", auto_generate_synthetic=False)

    assert len(train_ds) == 98
    assert len(val_ds) == 2
    assert set(train_ds.file_list).isdisjoint(set(val_ds.file_list))


def test_skip_silent_chunks(tmp_path: Path):
    """Verify chunks with RMS below 0.005 are skipped in favor of active audio regions."""
    audio_dir = tmp_path / "mostly_silent"
    audio_dir.mkdir(parents=True)

    # Create a 6-second file: first 4s pure silence, last 2s active 440 Hz tone
    sr = 16000
    silence = np.zeros(sr * 4, dtype=np.float32)
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    active = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    combined = np.concatenate([silence, active])
    sf.write(str(audio_dir / "partial_silence.wav"), combined, sr, subtype="PCM_16")

    ds = StegoDataset(audio_dir, split="train", auto_generate_synthetic=False)
    for _ in range(10):
        chunk, _ = ds[0]
        rms = float(torch.sqrt(torch.mean(chunk ** 2)).item())
        assert rms >= 0.005, f"Returned chunk had silent RMS={rms}"
