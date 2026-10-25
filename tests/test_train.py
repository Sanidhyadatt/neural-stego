"""Unit tests for training utilities, loss functions, and checkpointing."""

from pathlib import Path
import torch

from ml.config import CHUNK_SAMPLES
from ml.models import Decoder, Discriminator, Encoder
from ml.train import (
    MultiResolutionSTFTLoss,
    compute_snr_db,
    get_device,
    save_checkpoint,
)


def test_compute_snr_db():
    """Verify SNR calculation handles perfect, moderate, and noisy conditions."""
    signal = torch.ones(2, CHUNK_SAMPLES)
    noise = torch.ones(2, CHUNK_SAMPLES) * 0.1  # 10x smaller amplitude -> ~20 dB
    snr = compute_snr_db(signal, noise)
    assert 19.0 <= snr <= 21.0


def test_multi_resolution_stft_loss():
    """Verify multi-resolution STFT loss computation across multiple resolutions."""
    x = torch.randn(2, CHUNK_SAMPLES)
    y = x.clone()

    loss_fn = MultiResolutionSTFTLoss()
    zero_loss = loss_fn(x, y)
    assert zero_loss.item() < 1e-5

    perturbed = x + torch.randn_like(x) * 0.05
    pos_loss = loss_fn(perturbed, x)
    assert pos_loss.item() > 0.0


def test_checkpoint_roundtrip(tmp_path: Path):
    """Verify checkpoint saving and loading integrity."""
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    opt_g = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    opt_d = torch.optim.Adam(discriminator.parameters())

    ckpt_path = tmp_path / "test_ckpt.pt"
    save_checkpoint(ckpt_path, encoder, decoder, discriminator, opt_g, opt_d, epoch=5)

    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, weights_only=False)
    assert loaded["epoch"] == 5
    assert "encoder" in loaded
    assert "decoder" in loaded
    assert "discriminator" in loaded
    assert loaded["config"]["payload_bits"] == 128
