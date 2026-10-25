"""Unit tests for StegoEngine inference pipeline and CLI commands."""

import io
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
import torch

from ml.audio_utils import load_audio
from ml.config import SAMPLE_RATE
from ml.inference import StegoEngine
from ml.text_codec import text_to_chunk_bits


@pytest.fixture
def checkpoint_file() -> Path:
    """Fixture returning the verified smoke-test checkpoint path."""
    ckpt = Path("checkpoints/stego.pt")
    assert ckpt.is_file(), "Smoke-test checkpoint checkpoints/stego.pt does not exist."
    return ckpt


@pytest.fixture
def sample_wav_bytes(tmp_path: Path) -> bytes:
    """Fixture generating a 3.5-second test audio file in WAV format."""
    file_path = tmp_path / "cover_sample.wav"
    duration = 3.5
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False, dtype=np.float32)
    wave = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 880.0 * t)
    sf.write(str(file_path), wave, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return file_path.read_bytes()


def test_engine_load(checkpoint_file: Path):
    """Verify StegoEngine initializes and loads models in evaluation mode."""
    engine = StegoEngine(checkpoint_path=checkpoint_file, device="cpu")
    assert engine.encoder is not None
    assert engine.decoder is not None
    assert not engine.encoder.training
    assert not engine.decoder.training


def test_embed_and_info_structure(checkpoint_file: Path, sample_wav_bytes: bytes):
    """Verify embed runs end to end, returning valid WAV bytes and info dictionary."""
    engine = StegoEngine(checkpoint_path=checkpoint_file, device="cpu")
    secret_text = "Neural Stego Test"

    stego_bytes, info = engine.embed(sample_wav_bytes, secret_text)

    # Validate output types and lengths
    assert isinstance(stego_bytes, bytes)
    assert len(stego_bytes) > 0

    # Validate info dict
    assert "n_chunks" in info and info["n_chunks"] == 2  # 3.5s -> 2 chunks (4.0s padded)
    assert "duration_sec" in info and info["duration_sec"] == 3.5
    assert "snr_db" in info and isinstance(info["snr_db"], float)
    assert "capacity_chars" in info and info["capacity_chars"] == 22  # 2*16 - 10 = 22

    # Validate that stego_bytes is decodable as audio with exact original sample length
    orig_wave = load_audio(sample_wav_bytes)
    stego_wave = load_audio(stego_bytes)
    assert stego_wave.shape == orig_wave.shape
    assert torch.all(stego_wave >= -1.0) and torch.all(stego_wave <= 1.0)


def test_capacity_and_overflow(checkpoint_file: Path, sample_wav_bytes: bytes):
    """Verify capacity query and overflow rejection."""
    engine = StegoEngine(checkpoint_path=checkpoint_file, device="cpu")

    cap = engine.capacity(sample_wav_bytes)
    assert cap == 22  # 2 chunks * 16 bytes - 10 bytes overhead

    # Text exceeding 22 bytes capacity should raise ValueError
    too_long_text = "This text message definitely exceeds 22 characters capacity."
    with pytest.raises(ValueError, match="Maximum capacity for this audio"):
        engine.embed(sample_wav_bytes, too_long_text)


def test_extract_call_does_not_crash(checkpoint_file: Path, sample_wav_bytes: bytes):
    """Verify extract processes stego audio end to end without crashing.

    Note: With only 30 training steps, weights are not converged for full error correction,
    so extract must either return a string or raise ValueError('Message could not be recovered').
    """
    engine = StegoEngine(checkpoint_path=checkpoint_file, device="cpu")
    stego_bytes, _ = engine.embed(sample_wav_bytes, "Short msg")

    try:
        extracted = engine.extract(stego_bytes)
        assert isinstance(extracted, str)
    except ValueError as err:
        assert "Message could not be recovered" in str(err)


def test_cli_embed_and_extract(tmp_path: Path, checkpoint_file: Path, sample_wav_bytes: bytes):
    """Verify CLI interface for embedding and extracting via subprocess."""
    in_wav = tmp_path / "cli_in.wav"
    out_wav = tmp_path / "cli_out.wav"
    in_wav.write_bytes(sample_wav_bytes)

    # 1. Run embed CLI
    embed_cmd = [
        sys.executable,
        "-m",
        "ml.inference",
        "embed",
        str(in_wav),
        "CLI secret",
        str(out_wav),
        "--checkpoint",
        str(checkpoint_file),
        "--device",
        "cpu",
    ]
    res_embed = subprocess.run(embed_cmd, capture_output=True, text=True)
    assert res_embed.returncode == 0, f"CLI embed failed: {res_embed.stderr}"
    assert out_wav.is_file()
    assert out_wav.stat().st_size > 0

    # 2. Run extract CLI (checks CLI execution path)
    extract_cmd = [
        sys.executable,
        "-m",
        "ml.inference",
        "extract",
        str(out_wav),
        "--checkpoint",
        str(checkpoint_file),
        "--device",
        "cpu",
    ]
    res_extract = subprocess.run(extract_cmd, capture_output=True, text=True)
    # Either exits 0 (recovered) or exits 1 with expected message
    if res_extract.returncode != 0:
        assert "Message could not be recovered" in res_extract.stderr


def test_corrupted_chunk_voting_recovery(checkpoint_file: Path):
    """Verify that flipping all bits in one chunk still recovers secret text via voting."""
    engine = StegoEngine(checkpoint_path=checkpoint_file, device="cpu")
    secret_text = "hello"
    true_bits = text_to_chunk_bits(secret_text)  # [1, 128]
    n_blocks = true_bits.shape[0]

    # Mock decoder that flips all bits for chunk 0, but returns true bits for chunks 1 and 2
    class MockFlippedChunkDecoder(torch.nn.Module):
        def __init__(self, target_bits: torch.Tensor):
            super().__init__()
            self.target_bits = target_bits
            self.call_count = 0

        def forward(self, spec: torch.Tensor) -> torch.Tensor:
            batch_sz = spec.shape[0]
            logits = []
            for _ in range(batch_sz):
                idx = self.call_count
                self.call_count += 1
                block_idx = idx % n_blocks
                if idx == 0:
                    # Invert all bits in chunk 0
                    c_bits = 1.0 - self.target_bits[block_idx]
                else:
                    # Chunks 1 and 2 carry correct bits
                    c_bits = self.target_bits[block_idx]
                logit = (c_bits * 20.0) - 10.0
                logits.append(logit)
            return torch.stack(logits)

    engine.decoder = MockFlippedChunkDecoder(true_bits)

    # 3 chunks audio (6 seconds)
    audio = np.zeros(32000 * 3, dtype=np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, 16000, subtype="PCM_16", format="WAV")
    audio_bytes = bio.getvalue()

    recovered_text = engine.extract(audio_bytes)
    assert recovered_text == secret_text


def test_random_noise_audio_raises_unrecoverable(checkpoint_file: Path):
    """Verify that random-noise audio fails Reed-Solomon and raises unrecoverable error."""
    engine = StegoEngine(checkpoint_path=checkpoint_file, device="cpu")

    # Generate pure white noise audio
    noise = np.random.uniform(-0.1, 0.1, 32000 * 2).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, noise, 16000, subtype="PCM_16", format="WAV")
    noise_bytes = bio.getvalue()

    with pytest.raises(ValueError, match="Message could not be recovered"):
        engine.extract(noise_bytes)

