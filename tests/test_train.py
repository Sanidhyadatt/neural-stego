"""Unit tests for production training pipeline: CLI, curriculum, differentiable augs, EMA, and checkpoints."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ml.config import CHUNK_SAMPLES, PAYLOAD_BITS
from ml.dataset import StegoDataset
from ml.inference import StegoEngine
from ml.models import Decoder, Discriminator, Encoder
from ml.train import (
    ModelEMA,
    MultiResolutionSTFTLoss,
    StegoAugmentation,
    compute_snr_db,
    evaluate_validation,
    get_curriculum_params,
    parse_args,
    save_checkpoint,
)


def test_compute_snr_db():
    """Verify SNR calculation handles known signal/noise ratios."""
    signal = torch.ones(2, CHUNK_SAMPLES)
    noise = torch.ones(2, CHUNK_SAMPLES) * 0.1  # ~20 dB
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


def test_parse_args_defaults_and_flags():
    """Verify new CLI flags including multiple --data-dir, --steps, --amp, --resume, --num-workers."""
    defaults = parse_args([])
    assert defaults.steps is None
    assert defaults.epochs is None
    assert defaults.batch_size == 16
    assert defaults.data_dir is None  # resolves to ["data"] in train()
    assert defaults.amp is False
    assert defaults.resume is None
    assert defaults.ckpt_dir == "checkpoints"
    assert defaults.device == "auto"
    assert defaults.num_workers == 2

    custom = parse_args(
        [
            "--steps",
            "5000",
            "--batch-size",
            "8",
            "--data-dir",
            "dir_a",
            "dir_b",
            "--amp",
            "--resume",
            "--ckpt-dir",
            "my_ckpts",
            "--device",
            "cpu",
            "--num-workers",
            "4",
        ]
    )
    assert custom.steps == 5000
    assert custom.batch_size == 8
    assert custom.data_dir == ["dir_a", "dir_b"]
    assert custom.amp is True
    assert custom.resume == ""
    assert custom.ckpt_dir == "my_ckpts"
    assert custom.device == "cpu"
    assert custom.num_workers == 4


def test_curriculum_schedule():
    """Verify all 4 curriculum stages across 100 total steps."""
    # Stage 1: 0-10% (steps 1..10)
    c1 = get_curriculum_params(1, 100)
    assert c1["msg_weight"] == 5.0
    assert c1["l1_weight"] == 1.0
    assert c1["adv_weight"] == 0.0
    assert c1["mrstft_weight"] == 0.0
    assert c1["alpha"] == 0.05
    assert c1["aug_prob"] == 0.0
    assert c1["force_pcm16"] is False

    # Stage 2: 10-30% (steps 11..30) -> alpha anneals 0.05 -> 0.02
    c2_start = get_curriculum_params(11, 100)
    c2_mid = get_curriculum_params(21, 100)
    assert c2_start["adv_weight"] == 0.1
    assert c2_start["l1_weight"] == 10.0
    assert c2_start["mrstft_weight"] == 1.0
    assert c2_start["msg_weight"] == 1.0
    assert abs(c2_start["alpha"] - 0.05) < 1e-6
    assert abs(c2_mid["alpha"] - 0.035) < 1e-6
    assert c2_mid["aug_prob"] == 0.0
    assert c2_mid["force_pcm16"] is False

    # Stage 3: 30-80% -> aug_prob ramps from 0.0 toward 0.5, force_pcm16 still False
    c3_start = get_curriculum_params(31, 100)
    c3_mid = get_curriculum_params(65, 100)
    assert abs(c3_start["alpha"] - 0.02) < 1e-6
    assert abs(c3_start["aug_prob"] - 0.0) < 1e-6
    assert 0.20 < c3_mid["aug_prob"] < 0.35
    assert c3_mid["force_pcm16"] is False

    # Stage 4: Last 20% (steps 81..100) -> force_pcm16 is True, aug_prob reaches 0.5 at step 100
    c4_start = get_curriculum_params(81, 100)
    c4_end = get_curriculum_params(100, 100)
    assert c4_start["force_pcm16"] is True
    assert c4_end["force_pcm16"] is True
    assert abs(c4_end["aug_prob"] - 0.5) < 1e-6


def test_differentiable_augmentations_preserve_gradients():
    """Verify every augmentation preserves shape [B, 32000] and valid non-zero gradients."""
    aug = StegoAugmentation()

    ops = [
        aug.apply_gaussian_noise,
        aug.apply_random_gain,
        aug.apply_pcm16_quantization,
        aug.apply_lowpass_filter,
        aug.apply_random_resample,
        aug.apply_random_shift,
        lambda w: aug(w, aug_prob=1.0, force_pcm16=True),
    ]

    for op in ops:
        x = (torch.randn(2, CHUNK_SAMPLES) * 0.3).requires_grad_(True)
        y = op(x)
        assert y.shape == (2, CHUNK_SAMPLES)
        loss = (y ** 2).mean()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum().item() > 0.0


def test_checkpoint_ema_and_inference_loads_ema(tmp_path: Path):
    """Verify checkpoint saves EMA/schedulers/step and StegoEngine loads EMA weights."""
    torch.manual_seed(0)
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.02)
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32)
    discriminator = Discriminator(base=32)

    ema = ModelEMA(encoder, decoder, decay=0.999)

    # Perturb live encoder/decoder weights strongly so EMA differs visibly from live weights
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(5.0)
        for p in decoder.parameters():
            p.add_(5.0)
    ema.update(encoder, decoder)

    opt_g = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=100, eta_min=1e-5)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=100, eta_min=1e-5)

    ckpt_path = tmp_path / "stego.pt"
    save_checkpoint(
        ckpt_path,
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        opt_g=opt_g,
        opt_d=opt_d,
        step=42,
        ema=ema,
        sched_g=sched_g,
        sched_d=sched_d,
    )

    loaded = torch.load(ckpt_path, weights_only=False)
    assert loaded["step"] == 42
    assert "ema" in loaded and "encoder" in loaded["ema"] and "decoder" in loaded["ema"]
    assert "scheduler" in loaded
    assert loaded["config"]["payload_bits"] == 128

    # StegoEngine must load EMA weights, NOT raw live encoder/decoder weights
    engine = StegoEngine(checkpoint_path=ckpt_path, device="cpu")
    assert engine.encoder is not None and engine.decoder is not None
    for p_eng, p_ema, p_live in zip(
        engine.encoder.parameters(), ema.encoder.parameters(), encoder.parameters()
    ):
        assert torch.allclose(p_eng, p_ema)
        assert not torch.allclose(p_eng, p_live)


def test_evaluate_validation(tmp_path: Path):
    """Verify evaluate_validation computes bit_acc, snr_db, and d_acc."""
    ds = StegoDataset(tmp_path / "val_synth", split="val", auto_generate_synthetic=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    enc = Encoder()
    dec = Decoder()
    disc = Discriminator()

    metrics = evaluate_validation(enc, dec, disc, loader, torch.device("cpu"), max_batches=2)
    assert 0.0 <= metrics["bit_acc"] <= 100.0
    assert isinstance(metrics["snr_db"], float)
    assert 0.0 <= metrics["d_acc"] <= 1.0
