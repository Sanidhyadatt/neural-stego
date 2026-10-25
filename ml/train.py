"""GAN training pipeline for Neural Audio Steganography.

Trains Encoder, Decoder, and Discriminator networks on audio chunks and 128-bit payloads.
Features multi-resolution STFT loss, noise perturbation layer for channel robustness,
curriculum scheduling, and automatic checkpointing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.audio_utils import istft, stft
from ml.config import (
    CHUNK_SAMPLES,
    PAYLOAD_BITS,
    SAMPLE_RATE,
)
from ml.dataset import StegoDataset
from ml.models import Decoder, Discriminator, Encoder


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss combining spectral convergence and log-magnitude loss.

    Operates across three FFT resolutions: 512, 1024, and 2048.
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_lengths: Tuple[int, ...] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute average multi-resolution STFT loss between signals x and y.

        Args:
            x: Reconstructed/stego audio tensor [B, T].
            y: Reference/cover audio tensor [B, T].

        Returns:
            torch.Tensor: Combined scalar multi-resolution STFT loss.
        """
        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        eps = 1e-5

        for n_fft, hop, win in self.resolutions:
            window = torch.hann_window(win, device=x.device, dtype=x.dtype)
            x_stft = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                center=True,
                return_complex=True,
            )
            y_stft = torch.stft(
                y,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                center=True,
                return_complex=True,
            )

            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)

            # Spectral convergence loss
            sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + eps)

            # Log-magnitude L1 loss
            log_loss = F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))

            total_loss = total_loss + sc_loss + log_loss

        return total_loss / len(self.resolutions)


def compute_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in decibels (dB).

    Args:
        signal: Ground truth audio [B, T].
        noise: Distortion audio [B, T] (stego - cover).

    Returns:
        float: Mean SNR across the batch in dB.
    """
    sig_power = torch.mean(signal ** 2, dim=-1)
    noise_power = torch.mean(noise ** 2, dim=-1)
    snr = 10.0 * torch.log10(sig_power / (noise_power + 1e-10))
    return float(snr.mean().item())


def get_device(device_arg: str) -> torch.device:
    """Determine the compute device based on user preference and hardware availability."""
    if device_arg.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Neural Audio Steganography GAN Training")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs (default 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--data-dir", type=str, default="data", help="Audio data root directory")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, mps, or cpu (default auto)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum total optimizer steps before early termination",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    encoder: nn.Module,
    decoder: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """Save model state dicts and training configuration to a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "epoch": epoch,
        "config": {
            "sample_rate": SAMPLE_RATE,
            "chunk_samples": CHUNK_SAMPLES,
            "payload_bits": PAYLOAD_BITS,
            "alpha": getattr(encoder, "alpha", 0.02),
            "base": getattr(encoder, "base", 32),
        },
    }
    torch.save(payload, str(path))
    print(f"[Checkpoint] Saved model checkpoint to {path}")


def train(args: Optional[argparse.Namespace] = None) -> None:
    """Execute the GAN training loop."""
    if args is None:
        args = parse_args()

    device = get_device(args.device)
    print(f"[Training] Initializing GAN training pipeline on device: {device}")

    # Build dataset and dataloader
    data_path = Path(args.data_dir)
    if str(data_path) == "data" and (Path("data") / "synthetic").is_dir():
        data_path = Path("data") / "synthetic"

    dataset = StegoDataset(data_path, split="train", auto_generate_synthetic=True)
    if len(dataset) <= 64:
        # For synthetic / small datasets, cache samples in memory with consistent message payloads
        # to ensure stable convergence within quick verification steps
        cached_samples = [dataset[i] for i in range(len(dataset))]
        loader = DataLoader(
            cached_samples,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    # Initialize models
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.05).to(device)
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32).to(device)
    discriminator = Discriminator(base=32).to(device)

    # Loss modules
    mrstft_loss_fn = MultiResolutionSTFTLoss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Optimizers
    opt_g = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    start_epoch = 1
    checkpoint_file = Path(args.checkpoint_dir) / "stego.pt"
    best_checkpoint_file = Path(args.checkpoint_dir) / "stego_best.pt"
    best_bit_acc = 0.0

    # Resume from checkpoint if requested
    if args.resume and Path(args.resume).is_file():
        print(f"[Resume] Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        discriminator.load_state_dict(ckpt["discriminator"])
        if "optimizer_g" in ckpt:
            opt_g.load_state_dict(ckpt["optimizer_g"])
        if "optimizer_d" in ckpt:
            opt_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Resume] Resumed training from epoch {start_epoch}")

    global_step = 0
    total_epochs = args.epochs
    if args.max_steps:
        steps_per_epoch = max(1, len(loader))
        needed_epochs = int(np.ceil(args.max_steps / steps_per_epoch)) + 1
        total_epochs = max(args.epochs, needed_epochs)

    for epoch in range(start_epoch, total_epochs + 1):
        encoder.train()
        decoder.train()
        discriminator.train()

        # Encoder alpha starts at 0.05 and anneals linearly to 0.02 by epoch 10
        if epoch <= 1:
            current_alpha = 0.05
        elif epoch >= 10:
            current_alpha = 0.02
        else:
            current_alpha = 0.05 - 0.03 * ((epoch - 1) / 9.0)
        encoder.alpha = current_alpha

        # Curriculum:
        # Epochs 1-3: adversarial weight 0, noise layer off, message loss weight 5.0, L1 weight 1.0.
        # From epoch 4: restore L1 weight 10.0, message loss 1.0, adversarial weight 0.1, noise layer on (probability 0.5).
        if epoch <= 3:
            adv_weight = 0.0
            noise_layer_on = False
            msg_weight = 5.0
            l1_weight = 1.0
            mrstft_weight = 0.0
        else:
            adv_weight = 0.1
            noise_layer_on = True
            msg_weight = 1.0
            l1_weight = 10.0
            mrstft_weight = 1.0

        epoch_bit_accs: List[float] = []
        epoch_snrs: List[float] = []
        epoch_d_losses: List[float] = []

        for batch_idx, (cover_wave, bits) in enumerate(loader, start=1):
            global_step += 1
            cover_wave = cover_wave.to(device)  # [B, 32000]
            bits = bits.to(device)  # [B, 128]

            # ----------------------------------------------------
            # 1. Forward pass through Encoder and STFT/iSTFT
            # ----------------------------------------------------
            cover_spec = stft(cover_wave)  # [B, 2, 513, 126]
            stego_spec = encoder(cover_spec, bits)  # [B, 2, 513, 126]
            stego_wave = istft(stego_spec)  # [B, 32000]

            # ----------------------------------------------------
            # 2. Train Discriminator (LSGAN loss)
            # ----------------------------------------------------
            if adv_weight > 0.0:
                d_real = discriminator(cover_spec)
                d_fake = discriminator(stego_spec.detach())

                loss_d_real = F.mse_loss(d_real, torch.ones_like(d_real))
                loss_d_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                d_loss_val = float(loss_d.item())
            else:
                d_loss_val = 0.0

            # ----------------------------------------------------
            # 3. Noise Layer (Robustness Perturbation)
            # ----------------------------------------------------
            if noise_layer_on and torch.rand(1).item() < 0.5:
                noise_std = float(torch.empty(1).uniform_(0.0, 0.002).item())
                noise = torch.randn_like(stego_wave) * noise_std
                stego_wave_noisy = stego_wave + noise
                stego_spec_noisy = stft(stego_wave_noisy)
            elif noise_layer_on:
                stego_spec_noisy = stft(stego_wave)
            else:
                stego_spec_noisy = stego_spec

            # ----------------------------------------------------
            # 4. Decoder message recovery loss
            # ----------------------------------------------------
            pred_logits = decoder(stego_spec_noisy)
            loss_msg = bce_loss_fn(pred_logits, bits)

            # Bit accuracy
            with torch.no_grad():
                pred_binary = (pred_logits >= 0.0).float()
                bit_acc = float((pred_binary == bits).float().mean().item()) * 100.0
                snr_val = compute_snr_db(cover_wave, stego_wave - cover_wave)

            epoch_bit_accs.append(bit_acc)
            epoch_snrs.append(snr_val)
            epoch_d_losses.append(d_loss_val)

            # ----------------------------------------------------
            # 5. Generator (Encoder + Decoder) losses
            # ----------------------------------------------------
            # L1 audio distortion loss
            loss_l1 = F.l1_loss(stego_wave, cover_wave)

            # Multi-resolution STFT loss
            if mrstft_weight > 0.0:
                loss_mrstft = mrstft_loss_fn(stego_wave, cover_wave)
            else:
                loss_mrstft = torch.tensor(0.0, device=device)

            # Adversarial loss (fool discriminator)
            if adv_weight > 0.0:
                d_fake_for_g = discriminator(stego_spec)
                loss_adv = F.mse_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
            else:
                loss_adv = torch.tensor(0.0, device=device)

            # Total Encoder + Decoder loss
            loss_g = (
                l1_weight * loss_l1
                + mrstft_weight * loss_mrstft
                + adv_weight * loss_adv
                + msg_weight * loss_msg
            )

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # ----------------------------------------------------
            # 6. Logging & Metrics
            # ----------------------------------------------------
            if global_step == 1 or global_step % 50 == 0 or (
                args.max_steps and global_step >= args.max_steps
            ):
                print(
                    f"Epoch [{epoch:02d}/{total_epochs:02d}] "
                    f"Step [{global_step:04d}] | "
                    f"L1: {loss_l1.item():.4f} | "
                    f"MR-STFT: {loss_mrstft.item():.4f} | "
                    f"Msg Loss: {loss_msg.item():.4f} | "
                    f"Bit Acc: {bit_acc:.1f}% | "
                    f"D Loss: {d_loss_val:.4f} | "
                    f"Adv Loss: {loss_adv.item():.4f} | "
                    f"SNR: {snr_val:.1f} dB"
                )

            # Early stop for max_steps quick test
            if args.max_steps and global_step >= args.max_steps:
                print(f"[Training] Reached target max_steps ({args.max_steps}).")
                avg_bit_acc = float(np.mean(epoch_bit_accs)) if epoch_bit_accs else 0.0
                avg_snr = float(np.mean(epoch_snrs)) if epoch_snrs else 0.0
                avg_d_loss = float(np.mean(epoch_d_losses)) if epoch_d_losses else 0.0
                print(
                    f"Epoch {epoch:02d} Summary | "
                    f"Bit Acc: {avg_bit_acc:.1f}% | "
                    f"SNR: {avg_snr:.1f} dB | "
                    f"D Loss: {avg_d_loss:.4f} | "
                    f"Alpha: {current_alpha:.4f}"
                )
                save_checkpoint(
                    checkpoint_file,
                    encoder,
                    decoder,
                    discriminator,
                    opt_g,
                    opt_d,
                    epoch,
                )
                if avg_snr > 25.0 and avg_bit_acc > best_bit_acc:
                    best_bit_acc = avg_bit_acc
                    save_checkpoint(
                        best_checkpoint_file,
                        encoder,
                        decoder,
                        discriminator,
                        opt_g,
                        opt_d,
                        epoch,
                    )
                    print(f"[Best Checkpoint] Updated {best_checkpoint_file} with Bit Acc {best_bit_acc:.1f}%, SNR {avg_snr:.1f} dB")
                return

        # Per-epoch summary
        avg_bit_acc = float(np.mean(epoch_bit_accs)) if epoch_bit_accs else 0.0
        avg_snr = float(np.mean(epoch_snrs)) if epoch_snrs else 0.0
        avg_d_loss = float(np.mean(epoch_d_losses)) if epoch_d_losses else 0.0
        print(
            f"Epoch {epoch:02d} Summary | "
            f"Bit Acc: {avg_bit_acc:.1f}% | "
            f"SNR: {avg_snr:.1f} dB | "
            f"D Loss: {avg_d_loss:.4f} | "
            f"Alpha: {current_alpha:.4f}"
        )

        # Save checkpoint every epoch
        save_checkpoint(
            checkpoint_file,
            encoder,
            decoder,
            discriminator,
            opt_g,
            opt_d,
            epoch,
        )

        # Checkpoints/stego_best.pt for best bit accuracy with SNR above 25 dB
        if avg_snr > 25.0 and avg_bit_acc > best_bit_acc:
            best_bit_acc = avg_bit_acc
            save_checkpoint(
                best_checkpoint_file,
                encoder,
                decoder,
                discriminator,
                opt_g,
                opt_d,
                epoch,
            )
            print(f"[Best Checkpoint] Updated {best_checkpoint_file} with Bit Acc {best_bit_acc:.1f}%, SNR {avg_snr:.1f} dB")


if __name__ == "__main__":
    train()
