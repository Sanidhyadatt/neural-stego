"""Production GAN training pipeline for Neural Audio Steganography.

Features:
- Step-based training (--steps, with --epochs fallback) across one or multiple --data-dir roots
- Mixed precision (--amp), gradient clipping (1.0), Adam (2e-4) + cosine LR decay to 1e-5
- Exponential Moving Average (EMA, decay=0.999) of Encoder and Decoder weights
- 4-stage curriculum by % of total steps (decoder warmup -> GAN+MRSTFT -> progressive augs -> mandatory PCM16)
- Differentiable channel augmentation layer (Gaussian noise, gain, STE 16-bit quantization,
  6-8 kHz FIR low-pass filter, 12-16 kHz resample, non-wrapping time shift)
- Periodic validation on 2% held-out split (bit accuracy, SNR, Discriminator accuracy)
- Full checkpointing every 1000 steps and resume support (--resume)
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

from ml.audio_utils import istft, stft
from ml.config import (
    CHUNK_SAMPLES,
    PAYLOAD_BITS,
    SAMPLE_RATE,
)
from ml.dataset import StegoDataset
from ml.models import Decoder, Discriminator, Encoder

DEFAULT_TOTAL_STEPS = 20000
BASE_LR = 2e-4
MIN_LR = 1e-5
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.999
LOG_EVERY_STEPS = 100
CKPT_EVERY_STEPS = 1000
VAL_EVERY_STEPS = 2000


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss combining spectral convergence and log-magnitude loss."""

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_lengths: Tuple[int, ...] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute average multi-resolution STFT loss in float32."""
        x_f = x.float()
        y_f = y.float()
        total_loss = torch.tensor(0.0, device=x_f.device, dtype=torch.float32)
        eps = 1e-5

        for n_fft, hop, win in self.resolutions:
            window = torch.hann_window(win, device=x_f.device, dtype=torch.float32)
            x_stft = torch.stft(
                x_f,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                center=True,
                return_complex=True,
            )
            y_stft = torch.stft(
                y_f,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                center=True,
                return_complex=True,
            )

            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)

            sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + eps)
            log_loss = F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))
            total_loss = total_loss + sc_loss + log_loss

        return total_loss / len(self.resolutions)


class ModelEMA:
    """Exponential Moving Average tracker for Encoder and Decoder networks."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        decay: float = EMA_DECAY,
    ) -> None:
        self.decay = decay
        self.encoder: Encoder = copy.deepcopy(encoder).eval()
        self.decoder: Decoder = copy.deepcopy(decoder).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _update_module(ema_mod: nn.Module, live_mod: nn.Module, decay: float) -> None:
        for p_ema, p_live in zip(ema_mod.parameters(), live_mod.parameters()):
            p_ema.mul_(decay).add_(p_live.detach(), alpha=1.0 - decay)
        for b_ema, b_live in zip(ema_mod.buffers(), live_mod.buffers()):
            if b_ema.is_floating_point():
                b_ema.mul_(decay).add_(b_live.detach(), alpha=1.0 - decay)
            else:
                b_ema.copy_(b_live.detach())

    @torch.no_grad()
    def update(self, encoder: Encoder, decoder: Decoder) -> None:
        """Update EMA shadow parameters and buffers from live models."""
        self.encoder.alpha = encoder.alpha
        self._update_module(self.encoder, encoder, self.decay)
        self._update_module(self.decoder, decoder, self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decay": self.decay,
            "alpha": getattr(self.encoder, "alpha", 0.02),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
        if "decay" in state:
            self.decay = float(state["decay"])
        if "alpha" in state:
            self.encoder.alpha = float(state["alpha"])


class StegoAugmentation(nn.Module):
    """Differentiable waveform augmentation layer applied to stego_wave before Decoder."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.resample_rates = list(range(12000, sample_rate + 1, 200))

    @staticmethod
    def apply_gaussian_noise(wave: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with random std in [0.0, 0.003]."""
        std = random.uniform(0.0, 0.003)
        if std == 0.0:
            return wave
        return wave + torch.randn_like(wave) * std

    @staticmethod
    def apply_random_gain(wave: torch.Tensor) -> torch.Tensor:
        """Scale waveform by random per-example gain in [0.5, 1.5]."""
        gains = torch.empty((wave.shape[0], 1), device=wave.device, dtype=wave.dtype).uniform_(
            0.5, 1.5
        )
        return wave * gains

    @staticmethod
    def apply_pcm16_quantization(wave: torch.Tensor) -> torch.Tensor:
        """16-bit PCM quantization (step 1/32768) with straight-through gradient estimator."""
        clamped = wave.clamp(-1.0, 1.0)
        quantized = torch.round(clamped * 32768.0) / 32768.0
        return wave + (quantized - wave).detach()

    def apply_lowpass_filter(self, wave: torch.Tensor) -> torch.Tensor:
        """Differentiable windowed-sinc FIR low-pass filter with random cutoff in [6, 8] kHz."""
        cutoff_hz = random.uniform(6000.0, 8000.0)
        num_taps = 63
        half = num_taps // 2
        n = torch.arange(-half, half + 1, device=wave.device, dtype=wave.dtype)
        fc = cutoff_hz / float(self.sample_rate)
        window = torch.hann_window(num_taps, device=wave.device, dtype=wave.dtype)
        kernel = torch.sinc(2.0 * fc * n) * window
        kernel = kernel / (kernel.sum() + 1e-12)
        return F.conv1d(wave.unsqueeze(1), kernel.view(1, 1, -1), padding=half).squeeze(1)

    def apply_random_resample(self, wave: torch.Tensor) -> torch.Tensor:
        """Differentiable down/up resample via intermediate rate in [12, 16] kHz."""
        mid_sr = random.choice(self.resample_rates)
        if mid_sr == self.sample_rate:
            return wave
        orig_len = wave.shape[-1]
        down = torchaudio.functional.resample(wave, orig_freq=self.sample_rate, new_freq=mid_sr)
        up = torchaudio.functional.resample(down, orig_freq=mid_sr, new_freq=self.sample_rate)
        if up.shape[-1] < orig_len:
            up = F.pad(up, (0, orig_len - up.shape[-1]))
        elif up.shape[-1] > orig_len:
            up = up[..., :orig_len]
        return up

    @staticmethod
    def apply_random_shift(wave: torch.Tensor) -> torch.Tensor:
        """Random time shift of 0..2000 samples by cropping and zero-padding (no wraparound)."""
        shift = random.randint(0, 2000)
        if shift == 0:
            return wave
        total_len = wave.shape[-1]
        if random.random() < 0.5:
            return F.pad(wave[:, shift:], (0, shift), mode="constant", value=0.0)
        return F.pad(wave, (shift, 0), mode="constant", value=0.0)[:, :total_len]

    def forward(
        self,
        stego_wave: torch.Tensor,
        aug_prob: float = 0.0,
        force_pcm16: bool = False,
    ) -> torch.Tensor:
        """Apply independent random channel augmentations to stego_wave in float32."""
        x = stego_wave.float()

        if aug_prob > 0.0 and random.random() < aug_prob:
            x = self.apply_gaussian_noise(x)

        if aug_prob > 0.0 and random.random() < aug_prob:
            x = self.apply_random_gain(x)

        if force_pcm16 or (aug_prob > 0.0 and random.random() < aug_prob):
            x = self.apply_pcm16_quantization(x)

        if aug_prob > 0.0 and random.random() < aug_prob:
            x = self.apply_lowpass_filter(x)

        if aug_prob > 0.0 and random.random() < aug_prob:
            x = self.apply_random_resample(x)

        if aug_prob > 0.0 and random.random() < aug_prob:
            x = self.apply_random_shift(x)

        return x


def get_curriculum_params(step: int, total_steps: int) -> Dict[str, Any]:
    """Compute loss weights, Encoder alpha, and augmentation schedule by % of total steps.

    Curriculum stages (step in 1..total_steps):
    - 0 to 10%:  msg=5.0, l1=1.0, mrstft=0.0, adv=0.0, alpha=0.05, aug_prob=0.0
    - 10 to 30%: msg=1.0, l1=10.0, mrstft=1.0, adv=0.1, alpha anneals 0.05 -> 0.02, aug_prob=0.0
    - 30 to 100%: msg=1.0, l1=10.0, mrstft=1.0, adv=0.1, alpha=0.02, aug_prob ramps 0.0 -> 0.5
    - Last 20% (>= 80%): force_pcm16=True (always apply 16-bit quantization)
    """
    denom = float(max(1, total_steps))
    frac = max(0.0, min(1.0, float(step - 1) / denom))

    if frac < 0.10:
        msg_weight = 5.0
        l1_weight = 1.0
        mrstft_weight = 0.0
        adv_weight = 0.0
        alpha = 0.05
        aug_prob = 0.0
    elif frac < 0.30:
        msg_weight = 1.0
        l1_weight = 10.0
        mrstft_weight = 1.0
        adv_weight = 0.1
        t_alpha = (frac - 0.10) / 0.20
        alpha = 0.05 + (0.02 - 0.05) * t_alpha
        aug_prob = 0.0
    else:
        msg_weight = 1.0
        l1_weight = 10.0
        mrstft_weight = 1.0
        adv_weight = 0.1
        alpha = 0.02
        max_frac = float(max(1, total_steps - 1)) / denom if total_steps > 1 else 1.0
        span = max(1e-8, max_frac - 0.30)
        t_aug = max(0.0, min(1.0, (frac - 0.30) / span))
        aug_prob = 0.5 * t_aug

    force_pcm16 = frac >= 0.80

    return {
        "msg_weight": msg_weight,
        "l1_weight": l1_weight,
        "mrstft_weight": mrstft_weight,
        "adv_weight": adv_weight,
        "alpha": alpha,
        "aug_prob": aug_prob,
        "force_pcm16": force_pcm16,
    }


def compute_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Compute mean Signal-to-Noise Ratio (SNR) in decibels (dB) across batch."""
    sig_power = torch.mean(signal.float() ** 2, dim=-1)
    noise_power = torch.mean(noise.float() ** 2, dim=-1)
    snr = 10.0 * torch.log10(sig_power / (noise_power + 1e-10))
    return float(snr.mean().item())


def get_device(device_arg: str) -> torch.device:
    """Determine compute device based on preference and hardware availability."""
    if device_arg.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _format_eta(seconds: float) -> str:
    """Format remaining seconds as HH:MM:SS."""
    sec = max(0, int(round(seconds)))
    hrs, rem = divmod(sec, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _infinite_batches(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches endlessly across passes over loader."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_validation(
    encoder: Encoder,
    decoder: Decoder,
    discriminator: Discriminator,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 32,
) -> Dict[str, float]:
    """Evaluate bit accuracy (%), SNR (dB), and Discriminator accuracy ([0, 1]) on val split."""
    was_enc_train = encoder.training
    was_dec_train = decoder.training
    was_disc_train = discriminator.training

    encoder.eval()
    decoder.eval()
    discriminator.eval()

    bit_accs: List[float] = []
    snrs: List[float] = []
    d_accs: List[float] = []

    for batch_idx, (cover_wave, bits) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        cover_wave = cover_wave.to(device)
        bits = bits.to(device)

        cover_spec = stft(cover_wave)
        stego_spec = encoder(cover_spec, bits)
        stego_wave = istft(stego_spec)

        # Quantize to 16-bit PCM as in real deployment
        stego_wave_eval = torch.round(stego_wave.clamp(-1.0, 1.0) * 32768.0) / 32768.0
        pred_logits = decoder(stft(stego_wave_eval))

        pred_bits = (pred_logits >= 0.0).float()
        bit_acc = float((pred_bits == bits).float().mean().item()) * 100.0
        snr_val = compute_snr_db(cover_wave, stego_wave - cover_wave)

        # LSGAN real target = 1.0, fake target = 0.0 -> decision threshold = 0.5
        d_real = discriminator(cover_spec)
        d_fake = discriminator(stego_spec)
        acc_real = float((d_real >= 0.5).float().mean().item())
        acc_fake = float((d_fake < 0.5).float().mean().item())
        d_acc = 0.5 * (acc_real + acc_fake)

        bit_accs.append(bit_acc)
        snrs.append(snr_val)
        d_accs.append(d_acc)

    if was_enc_train:
        encoder.train()
    if was_dec_train:
        decoder.train()
    if was_disc_train:
        discriminator.train()

    return {
        "bit_acc": float(np.mean(bit_accs)) if bit_accs else 0.0,
        "snr_db": float(np.mean(snrs)) if snrs else 0.0,
        "d_acc": float(np.mean(d_accs)) if d_accs else 0.5,
    }


def save_checkpoint(
    path: Path,
    encoder: nn.Module,
    decoder: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    step: int = 0,
    ema: Optional[ModelEMA] = None,
    sched_g: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    sched_d: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler_g: Optional[torch.amp.GradScaler] = None,
    scaler_d: Optional[torch.amp.GradScaler] = None,
    best_bit_acc: float = -1.0,
    extra_config: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
) -> None:
    """Save full training state (models, EMA, optimizers, schedulers, step, config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {
        "sample_rate": SAMPLE_RATE,
        "chunk_samples": CHUNK_SAMPLES,
        "payload_bits": PAYLOAD_BITS,
        "alpha": getattr(encoder, "alpha", 0.02),
        "base": getattr(encoder, "base", 32),
    }
    if extra_config:
        cfg.update(extra_config)

    ema_dict = (
        ema.state_dict()
        if ema is not None
        else {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "decay": EMA_DECAY,
            "alpha": getattr(encoder, "alpha", 0.02),
        }
    )

    payload: Dict[str, Any] = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema": ema_dict,
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "scheduler": {
            "scheduler_g": sched_g.state_dict() if sched_g is not None else None,
            "scheduler_d": sched_d.state_dict() if sched_d is not None else None,
        },
        "step": step,
        "best_bit_acc": best_bit_acc,
        "config": cfg,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if scaler_g is not None and scaler_g.is_enabled():
        payload["scaler_g"] = scaler_g.state_dict()
    if scaler_d is not None and scaler_d.is_enabled():
        payload["scaler_d"] = scaler_d.state_dict()

    torch.save(payload, str(path))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for production training."""
    parser = argparse.ArgumentParser(description="Neural Audio Steganography GAN Training")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Total training steps (default {DEFAULT_TOTAL_STEPS} unless --epochs is set)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Fallback training epochs if --steps is omitted",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        action="extend",
        default=None,
        help="One or more audio data root directories (default: data)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Resume training from checkpoint path (defaults to {ckpt-dir}/stego.pt if path omitted)",
    )
    parser.add_argument(
        "--ckpt-dir",
        "--checkpoint-dir",
        dest="ckpt_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes (default: 2)",
    )
    return parser.parse_args(argv)


def train(args: Optional[argparse.Namespace] = None) -> None:
    """Execute the production step-based GAN training pipeline."""
    if args is None:
        args = parse_args()

    device = get_device(args.device)
    data_dirs: List[str] = list(args.data_dir) if args.data_dir else ["data"]
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = ckpt_dir / "stego.pt"
    best_checkpoint_file = ckpt_dir / "stego_best.pt"

    use_amp = bool(args.amp)
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    print(
        f"[Train] Device: {device} | AMP: {use_amp} | Batch size: {args.batch_size} | "
        f"Workers: {args.num_workers} | Data dirs: {data_dirs}"
    )

    # Datasets & DataLoaders (98% train / 2% held-out validation)
    train_dataset = StegoDataset(data_dirs, split="train", auto_generate_synthetic=True)
    val_dataset = StegoDataset(data_dirs, split="val", auto_generate_synthetic=False)

    drop_last = len(train_dataset) >= args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Resolve total steps (--steps takes priority, --epochs works as fallback)
    if args.steps is not None:
        total_steps = args.steps
    elif args.epochs is not None:
        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
    else:
        total_steps = DEFAULT_TOTAL_STEPS

    print(
        f"[Train] Train files: {len(train_dataset)} | Val files: {len(val_dataset)} | "
        f"Total steps: {total_steps}"
    )

    # Models & EMA
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.05).to(device)
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32).to(device)
    discriminator = Discriminator(base=32).to(device)
    ema = ModelEMA(encoder, decoder, decay=EMA_DECAY)
    augmenter = StegoAugmentation(sample_rate=SAMPLE_RATE).to(device)

    # Losses
    mrstft_loss_fn = MultiResolutionSTFTLoss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Optimizers (Adam lr=2e-4) & Cosine LR Schedulers (-> 1e-5)
    g_params = list(encoder.parameters()) + list(decoder.parameters())
    opt_g = torch.optim.Adam(g_params, lr=BASE_LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=BASE_LR, betas=(0.5, 0.999))

    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=max(1, total_steps), eta_min=MIN_LR
    )
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=max(1, total_steps), eta_min=MIN_LR
    )

    scaler_enabled = use_amp and device.type == "cuda"
    scaler_g = torch.amp.GradScaler(amp_device_type, enabled=scaler_enabled)
    scaler_d = torch.amp.GradScaler(amp_device_type, enabled=scaler_enabled)

    start_step = 1
    best_bit_acc = -1.0

    # Resume support
    if args.resume is not None:
        resume_path = Path(args.resume) if args.resume != "" else checkpoint_file
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found at: {resume_path}")

        print(f"[Resume] Restoring training state from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        discriminator.load_state_dict(ckpt["discriminator"])

        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            ema.load_state_dict(ckpt["ema"])
        else:
            ema = ModelEMA(encoder, decoder, decay=EMA_DECAY)

        if "optimizer_g" in ckpt:
            opt_g.load_state_dict(ckpt["optimizer_g"])
        if "optimizer_d" in ckpt:
            opt_d.load_state_dict(ckpt["optimizer_d"])

        sched_state = ckpt.get("scheduler", {})
        if isinstance(sched_state, dict):
            if sched_state.get("scheduler_g") is not None:
                sched_g.load_state_dict(sched_state["scheduler_g"])
                sched_g.T_max = max(sched_g.T_max, total_steps)
            if sched_state.get("scheduler_d") is not None:
                sched_d.load_state_dict(sched_state["scheduler_d"])
                sched_d.T_max = max(sched_d.T_max, total_steps)

        if "scaler_g" in ckpt and scaler_g.is_enabled():
            scaler_g.load_state_dict(ckpt["scaler_g"])
        if "scaler_d" in ckpt and scaler_d.is_enabled():
            scaler_d.load_state_dict(ckpt["scaler_d"])

        saved_step = int(ckpt.get("step", 0))
        best_bit_acc = float(ckpt.get("best_bit_acc", -1.0))
        start_step = saved_step + 1
        print(f"[Resume] Restored checkpoint at step {saved_step}; continuing from step {start_step}.")

        if start_step > total_steps:
            print(
                f"[Resume] Checkpoint step ({saved_step}) already reached target --steps ({total_steps}). "
                f"Pass a larger --steps value to continue training."
            )
            return

    encoder.train()
    decoder.train()
    discriminator.train()

    train_iter = _infinite_batches(train_loader)
    loop_start_time = time.perf_counter()

    win_bit_accs: List[float] = []
    win_snrs: List[float] = []
    win_l1s: List[float] = []
    win_d_losses: List[float] = []

    try:
        for step in range(start_step, total_steps + 1):
            cover_wave, bits = next(train_iter)
            cover_wave = cover_wave.to(device, non_blocking=True)
            bits = bits.to(device, non_blocking=True)

            cur = get_curriculum_params(step, total_steps)
            encoder.alpha = cur["alpha"]
            msg_weight = cur["msg_weight"]
            l1_weight = cur["l1_weight"]
            mrstft_weight = cur["mrstft_weight"]
            adv_weight = cur["adv_weight"]
            aug_prob = cur["aug_prob"]
            force_pcm16 = cur["force_pcm16"]

            # 1. STFT -> Encoder -> iSTFT
            with torch.amp.autocast(device_type=amp_device_type, enabled=False):
                cover_spec = stft(cover_wave.float())

            with torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=use_amp):
                stego_spec = encoder(cover_spec, bits)

            with torch.amp.autocast(device_type=amp_device_type, enabled=False):
                stego_wave = istft(stego_spec.float())

            # 2. Discriminator update (when adversarial loss active)
            if adv_weight > 0.0:
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=amp_device_type, dtype=amp_dtype, enabled=use_amp
                ):
                    d_real = discriminator(cover_spec)
                    d_fake = discriminator(stego_spec.detach())
                    loss_d = 0.5 * (
                        F.mse_loss(d_real, torch.ones_like(d_real))
                        + F.mse_loss(d_fake, torch.zeros_like(d_fake))
                    )

                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler_d.step(opt_d)
                scaler_d.update()
                sched_d.step()
                d_loss_val = float(loss_d.item())
            else:
                d_loss_val = 0.0

            # 3. Differentiable channel augmentation on stego_wave -> STFT -> Decoder + Generator losses
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_type, enabled=False):
                aug_wave = augmenter(stego_wave, aug_prob=aug_prob, force_pcm16=force_pcm16)
                aug_spec = stft(aug_wave)
                loss_l1 = F.l1_loss(stego_wave, cover_wave.float())
                loss_mrstft = (
                    mrstft_loss_fn(stego_wave, cover_wave.float())
                    if mrstft_weight > 0.0
                    else torch.tensor(0.0, device=device)
                )

            with torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=use_amp):
                pred_logits = decoder(aug_spec)
                loss_msg = bce_loss_fn(pred_logits, bits)

                if adv_weight > 0.0:
                    d_fake_for_g = discriminator(stego_spec)
                    loss_adv = F.mse_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
                else:
                    loss_adv = torch.tensor(0.0, device=device)

                loss_g = (
                    l1_weight * loss_l1
                    + mrstft_weight * loss_mrstft
                    + adv_weight * loss_adv
                    + msg_weight * loss_msg
                )

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(g_params, max_norm=GRAD_CLIP_NORM)
            scaler_g.step(opt_g)
            scaler_g.update()
            sched_g.step()

            # 4. Update EMA of Encoder and Decoder
            ema.update(encoder, decoder)

            # 5. Track metrics
            with torch.no_grad():
                pred_bits = (pred_logits >= 0.0).float()
                bit_acc = float((pred_bits == bits).float().mean().item()) * 100.0
                snr_val = compute_snr_db(cover_wave, stego_wave - cover_wave)
                l1_val = float(loss_l1.item())

            win_bit_accs.append(bit_acc)
            win_snrs.append(snr_val)
            win_l1s.append(l1_val)
            win_d_losses.append(d_loss_val)

            # 6. One-line summary every 100 steps
            if step % LOG_EVERY_STEPS == 0 or step == total_steps:
                elapsed = time.perf_counter() - loop_start_time
                steps_done = step - start_step + 1
                eta_sec = (elapsed / max(1, steps_done)) * (total_steps - step)
                print(
                    f"step={step}/{total_steps} | "
                    f"bit_acc={float(np.mean(win_bit_accs)):.2f}% | "
                    f"snr={float(np.mean(win_snrs)):.2f} dB | "
                    f"l1={float(np.mean(win_l1s)):.5f} | "
                    f"d_loss={float(np.mean(win_d_losses)):.4f} | "
                    f"eta={_format_eta(eta_sec)}",
                    flush=True,
                )
                win_bit_accs.clear()
                win_snrs.clear()
                win_l1s.clear()
                win_d_losses.clear()

            # 7. Validation every 2000 steps on held-out split & best checkpoint saving
            if step % VAL_EVERY_STEPS == 0:
                val_metrics = evaluate_validation(
                    ema.encoder,
                    ema.decoder,
                    discriminator,
                    val_loader,
                    device,
                )
                v_acc = val_metrics["bit_acc"]
                v_snr = val_metrics["snr_db"]
                v_d_acc = val_metrics["d_acc"]
                print(
                    f"[Val] step={step} | bit_acc={v_acc:.2f}% | "
                    f"snr={v_snr:.2f} dB | d_acc={v_d_acc:.3f}"
                )
                if v_acc > best_bit_acc:
                    best_bit_acc = v_acc
                    save_checkpoint(
                        best_checkpoint_file,
                        encoder=encoder,
                        decoder=decoder,
                        discriminator=discriminator,
                        opt_g=opt_g,
                        opt_d=opt_d,
                        step=step,
                        ema=ema,
                        sched_g=sched_g,
                        sched_d=sched_d,
                        scaler_g=scaler_g,
                        scaler_d=scaler_d,
                        best_bit_acc=best_bit_acc,
                        extra_config={"total_steps": total_steps},
                    )
                    print(f"Saved best (bit_acc={v_acc:.2f}%, snr={v_snr:.2f} dB)")

            # 8. Periodic checkpoint every 1000 steps (and at final step)
            if step % CKPT_EVERY_STEPS == 0 or step == total_steps:
                save_checkpoint(
                    checkpoint_file,
                    encoder=encoder,
                    decoder=decoder,
                    discriminator=discriminator,
                    opt_g=opt_g,
                    opt_d=opt_d,
                    step=step,
                    ema=ema,
                    sched_g=sched_g,
                    sched_d=sched_d,
                    scaler_g=scaler_g,
                    scaler_d=scaler_d,
                    best_bit_acc=best_bit_acc,
                    extra_config={"total_steps": total_steps},
                )
                print(f"[Checkpoint] Saved {checkpoint_file} at step {step}")
    finally:
        # Ensure multi-worker DataLoader processes shut down cleanly on Windows/POSIX
        del train_iter


if __name__ == "__main__":
    train()
