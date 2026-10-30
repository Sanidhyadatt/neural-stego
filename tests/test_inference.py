"""Unit tests for StegoEngine inference pipeline and CLI commands."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import pytest
import soundfile as sf
import torch

from ml.audio_utils import istft, load_audio, save_wav_bytes
from ml.config import CHUNK_SAMPLES, PAYLOAD_BITS, SAMPLE_RATE
from ml.inference import StegoEngine
from ml.models import Decoder, Encoder
from ml.text_codec import text_to_chunk_bits


def make_synthetic_wav_bytes(duration_sec: float, seed: int = 42) -> bytes:
    """Generate synthetic non-periodic cover audio (chirp + harmonics + subtle noise)."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    signal = (
        0.35 * np.sin(2.0 * np.pi * (180.0 + 35.0 * t) * t)
        + 0.20 * np.sin(2.0 * np.pi * 440.0 * t)
        + rng.normal(0.0, 0.12, size=n_samples).astype(np.float32)
    )
    signal = np.clip(signal, -0.95, 0.95)
    bio = io.BytesIO()
    sf.write(bio, signal, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return bio.getvalue()


@pytest.fixture
def random_checkpoint(tmp_path: Path) -> Path:
    """Create a randomly-initialized model checkpoint on disk for offline testing."""
    torch.manual_seed(101)
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.02)
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32)
    ckpt_path = tmp_path / "random_stego.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "epoch": 0,
            "config": {"payload_bits": PAYLOAD_BITS, "base": 32, "alpha": 0.02},
        },
        ckpt_path,
    )
    return ckpt_path


class AlignedChunkMockDecoder(torch.nn.Module):
    """Mock neural decoder that recovers embedded chunk bits when chunk-aligned.

    Records (reference_stego_waveform, target_bits) produced by the randomly-initialized
    Encoder during embed(), and predicts matching logits during extract() only when a
    candidate chunk aligns within +/- 1200 samples of an embedded chunk.
    """

    def __init__(self, tolerance_samples: int = 1200) -> None:
        super().__init__()
        self.tolerance_samples = tolerance_samples
        self.recorded: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def attach_to_encoder(self, encoder: torch.nn.Module) -> None:
        def _hook(_mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            bits = inputs[1].detach().cpu()
            waves = istft(output.detach()).cpu()
            for w, b in zip(waves, bits):
                self.recorded.append((w, b))

        encoder.register_forward_hook(_hook)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        cand_waves = istft(spec).detach().cpu()  # [B, 32000]
        batch_sz = cand_waves.shape[0]
        if not self.recorded:
            return torch.zeros((batch_sz, PAYLOAD_BITS), dtype=torch.float32, device=spec.device)

        ref_waves = torch.stack([r[0] for r in self.recorded], dim=0)  # [M, 32000]
        ref_bits = torch.stack([r[1] for r in self.recorded], dim=0)  # [M, 128]

        n_fft = 65536
        c_fft = torch.fft.rfft(cand_waves, n=n_fft)  # [B, F]
        r_fft = torch.fft.rfft(ref_waves, n=n_fft)  # [M, F]
        xcorr = torch.fft.irfft(c_fft.unsqueeze(1) * r_fft.conj().unsqueeze(0), n=n_fft)

        tol = self.tolerance_samples
        window_corr = torch.cat([xcorr[:, :, : tol + 1], xcorr[:, :, -tol:]], dim=-1)
        peak_corr = window_corr.max(dim=-1).values  # [B, M]

        cand_norm = torch.norm(cand_waves, dim=-1, keepdim=True)
        ref_norm = torch.norm(ref_waves, dim=-1).unsqueeze(0)
        norm_corr = peak_corr / (cand_norm * ref_norm + 1e-8)

        best_score, best_idx = norm_corr.max(dim=-1)  # [B]
        logits = torch.zeros((batch_sz, PAYLOAD_BITS), dtype=torch.float32)

        for i in range(batch_sz):
            if best_score[i].item() > 0.60:
                logits[i] = ref_bits[best_idx[i]] * 20.0 - 10.0

        return logits.to(spec.device)


def test_engine_load(random_checkpoint: Path):
    """Verify StegoEngine initializes and loads randomly-initialized models in eval mode."""
    engine = StegoEngine(checkpoint_path=random_checkpoint, device="cpu")
    assert engine.encoder is not None
    assert engine.decoder is not None
    assert not engine.encoder.training
    assert not engine.decoder.training


def test_capacity_and_audio_too_short_error(random_checkpoint: Path):
    """Verify capacity uses max_text_bytes and embed raises exact 'Audio too short' error."""
    engine = StegoEngine(checkpoint_path=random_checkpoint, device="cpu")
    sample_wav = make_synthetic_wav_bytes(3.5)  # 2 chunks

    cap = engine.capacity(sample_wav)
    assert cap == 20  # 2 * 16 - 12 = 20

    # 50 ASCII bytes -> 50 + 12 = 62 -> 4 blocks -> needs 8 seconds
    too_long_text = "A" * 50
    with pytest.raises(
        ValueError,
        match=r"Audio too short for this message\. Needs at least 8 seconds\.",
    ):
        engine.embed(sample_wav, too_long_text)


@pytest.mark.parametrize(
    ("secret_text", "duration_sec"),
    [
        ("hi", 6.0),
        ("0123456789" * 10, 20.0),
        ("Hidden audio emojis! 🎵🎧🔒🚀✨", 10.0),
    ],
)
def test_roundtrip_messages_on_audio(
    random_checkpoint: Path,
    secret_text: str,
    duration_sec: float,
):
    """Verify end-to-end embed + extract roundtrip for 'hi', 100-char string, and emoji string."""
    engine = StegoEngine(checkpoint_path=random_checkpoint, device="cpu")
    assert engine.encoder is not None
    mock_dec = AlignedChunkMockDecoder()
    mock_dec.attach_to_encoder(engine.encoder)
    engine.decoder = mock_dec

    cover_bytes = make_synthetic_wav_bytes(duration_sec)
    stego_bytes, info = engine.embed(cover_bytes, secret_text)

    assert len(stego_bytes) > 0
    assert info["duration_sec"] == round(duration_sec, 2)

    recovered = engine.extract(stego_bytes)
    assert recovered == secret_text


def test_corrupt_25_percent_bits_in_one_block_engine_recovery(random_checkpoint: Path):
    """Verify StegoEngine recovers message when 25% of bits (32 bits) in one block are corrupted."""
    engine = StegoEngine(checkpoint_path=random_checkpoint, device="cpu")
    secret_text = "Block FEC test!"
    true_bits = text_to_chunk_bits(secret_text)  # [2, 128]
    assert true_bits.shape[0] == 2

    # Corrupt 25% (first 32 bits = 4 bytes) of block 0
    corrupted_bits = true_bits.clone()
    corrupted_bits[0, :32] = 1.0 - corrupted_bits[0, :32]

    class SinglePassCorruptedBlockDecoder(torch.nn.Module):
        def forward(self, spec: torch.Tensor) -> torch.Tensor:
            b_sz = spec.shape[0]
            out = torch.stack([corrupted_bits[i % 2] for i in range(b_sz)], dim=0)
            return (out * 20.0 - 10.0).to(spec.device)

    engine.decoder = SinglePassCorruptedBlockDecoder()
    cover_bytes = make_synthetic_wav_bytes(4.0)  # 2 chunks (no extra voting copies)
    assert engine.extract(cover_bytes) == secret_text


def test_flip_all_bits_in_one_chunk_voting_recovery(random_checkpoint: Path):
    """Verify flipping all 128 bits in one chunk still recovers secret text via voting."""
    engine = StegoEngine(checkpoint_path=random_checkpoint, device="cpu")
    secret_text = "Voting works! 🚀"
    true_bits = text_to_chunk_bits(secret_text)  # [2, 128]
    n_blocks = true_bits.shape[0]

    class MockFlippedChunkDecoder(torch.nn.Module):
        def forward(self, spec: torch.Tensor) -> torch.Tensor:
            batch_sz = spec.shape[0]
            logits = []
            for idx in range(batch_sz):
                block_idx = idx % n_blocks
                if idx == 0:
                    # Flip ALL 128 bits in chunk 0
                    c_bits = 1.0 - true_bits[block_idx]
                else:
                    c_bits = true_bits[block_idx]
                logits.append(c_bits * 20.0 - 10.0)
            return torch.stack(logits, dim=0).to(spec.device)

    engine.decoder = MockFlippedChunkDecoder()
    # 6 chunks (12s) -> block 0 appears in chunks 0, 2, 4 (2 valid vs 1 flipped)
    cover_bytes = make_synthetic_wav_bytes(12.0)
    recovered = engine.extract(cover_bytes)
    assert recovered == secret_text


def test_random_noise_audio_raises_value_error(random_checkpoint: Path):
    """Verify random noise audio fails magic/CRC/RS and raises ValueError."""
    engine = StegoEngine(checkpoint_path=random_checkpoint, device="cpu")

    rng = np.random.RandomState(2026)
    noise = rng.uniform(-0.2, 0.2, CHUNK_SAMPLES * 2).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, noise, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    noise_bytes = bio.getvalue()

    with pytest.raises(ValueError, match="Message could not be recovered"):
        engine.extract(noise_bytes)


def test_trimmed_5000_samples_from_start_still_decodes(random_checkpoint: Path):
    """Verify a stego clip trimmed by 5000 samples from the start still decodes via shift search."""
    engine = StegoEngine(checkpoint_path=random_checkpoint, device="cpu")
    assert engine.encoder is not None
    mock_dec = AlignedChunkMockDecoder()
    mock_dec.attach_to_encoder(engine.encoder)
    engine.decoder = mock_dec

    secret_text = "Trimmed clip recovery!"  # 22 bytes -> 3 blocks (tiled across 6 chunks)
    cover_bytes = make_synthetic_wav_bytes(12.0, seed=99)
    stego_bytes, _ = engine.embed(cover_bytes, secret_text)

    # Trim 5000 samples from the start of the stego audio
    stego_wave = load_audio(stego_bytes)
    trimmed_wave = stego_wave[5000:]
    trimmed_bytes = save_wav_bytes(trimmed_wave)

    # Unshifted chunk decoding must fail due to 5000-sample misalignment
    assert engine._decode_waveform_chunks(load_audio(trimmed_bytes)) is None

    # Full extract() shifts by steps of 2000 samples and recovers the exact message
    recovered = engine.extract(trimmed_bytes)
    assert recovered == secret_text


def test_cli_embed_and_extract(tmp_path: Path, random_checkpoint: Path):
    """Verify CLI interface for embedding and extracting via subprocess."""
    in_wav = tmp_path / "cli_in.wav"
    out_wav = tmp_path / "cli_out.wav"
    in_wav.write_bytes(make_synthetic_wav_bytes(4.0))

    embed_cmd = [
        sys.executable,
        "-m",
        "ml.inference",
        "embed",
        str(in_wav),
        "CLI secret",
        str(out_wav),
        "--checkpoint",
        str(random_checkpoint),
        "--device",
        "cpu",
    ]
    res_embed = subprocess.run(embed_cmd, capture_output=True, text=True)
    assert res_embed.returncode == 0, f"CLI embed failed: {res_embed.stderr}"
    assert out_wav.is_file()
    assert out_wav.stat().st_size > 0

    extract_cmd = [
        sys.executable,
        "-m",
        "ml.inference",
        "extract",
        str(out_wav),
        "--checkpoint",
        str(random_checkpoint),
        "--device",
        "cpu",
    ]
    res_extract = subprocess.run(extract_cmd, capture_output=True, text=True)
    if res_extract.returncode != 0:
        assert "Message could not be recovered" in res_extract.stderr
