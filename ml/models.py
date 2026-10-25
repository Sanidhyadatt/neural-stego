"""Neural network models for Audio Steganography.

Implements GAN-based steganography architecture:
1. Encoder: Injects 128-bit message into cover spectrogram via small U-Net residual generator.
2. Decoder: Recovers 128 raw bit logits from stego spectrograms.
3. Discriminator: Distinguishes cover vs. stego spectrograms with spectral normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import spectral_norm
except ImportError:
    from torch.nn.utils import spectral_norm  # Fallback for older torch versions

from ml.config import PAYLOAD_BITS, STFT_CHANNELS


class ConvBlock(nn.Module):
    """Standard double convolution block with BatchNorm2d and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """U-Net residual generator embedding secret messages into audio spectrograms.

    Args:
        payload_bits: Number of secret payload bits per chunk (default 128).
        base: Base channel multiplier for convolutions (default 32).
        alpha: Residual scaling factor for perturbation control (default 0.02).
    """

    def __init__(
        self,
        payload_bits: int = PAYLOAD_BITS,
        base: int = 32,
        alpha: float = 0.02,
    ) -> None:
        super().__init__()
        self.payload_bits = payload_bits
        self.base = base
        self.alpha = alpha

        # Linear projection for message embedding
        self.msg_linear = nn.Linear(payload_bits, 64)
        self.msg_act = nn.LeakyReLU(0.2, inplace=True)

        # Total input channels: 2 (cover spectrogram) + 64 (message feature map) = 66
        in_channels = STFT_CHANNELS + 64

        # Encoder (Downsampling)
        self.enc1 = ConvBlock(in_channels, base)
        self.down1 = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc2 = ConvBlock(base, base * 2)
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc3 = ConvBlock(base * 2, base * 4)
        self.down3 = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck
        self.bottleneck = ConvBlock(base * 4, base * 8)

        # Decoder (Upsampling with skip connections)
        self.up3_conv = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec3 = ConvBlock(base * 8, base * 4)  # base*4 (up) + base*4 (skip)

        self.up2_conv = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = ConvBlock(base * 4, base * 2)  # base*2 (up) + base*2 (skip)

        self.up1_conv = nn.Sequential(
            nn.Conv2d(base * 2, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = ConvBlock(base * 2, base)  # base (up) + base (skip)

        # Final residual projection
        self.final_conv = nn.Conv2d(base, STFT_CHANNELS, kernel_size=3, padding=1)

    def forward(
        self,
        cover_spec: torch.Tensor,
        message_bits: torch.Tensor,
    ) -> torch.Tensor:
        """Embed message bits into the cover spectrogram.

        Args:
            cover_spec: Spectrogram tensor [B, 2, 513, 126].
            message_bits: Message bits tensor [B, 128].

        Returns:
            torch.Tensor: Stego spectrogram tensor [B, 2, 513, 126].
        """
        b, _, h, w = cover_spec.shape

        # Embed message: Linear(128 -> 64) and broadcast to [B, 64, H, W]
        msg_emb = self.msg_act(self.msg_linear(message_bits))
        msg_map = msg_emb.view(b, 64, 1, 1).expand(b, 64, h, w)

        # Concatenate cover spectrogram and broadcasted message map -> [B, 66, H, W]
        x = torch.cat([cover_spec, msg_map], dim=1)

        # Encoder path
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        # Bottleneck
        b_feat = self.bottleneck(d3)

        # Decoder path with safe interpolation to handle odd spatial dimensions
        u3 = self.up3_conv(b_feat)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3_out = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2_conv(d3_out)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2_out = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1_conv(d2_out)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1_out = self.dec1(torch.cat([u1, e1], dim=1))

        # Output residual
        residual = self.final_conv(d1_out)
        if residual.shape[-2:] != (h, w):
            residual = F.interpolate(residual, size=(h, w), mode="bilinear", align_corners=False)

        # Apply tanh scaling and add residual to cover spectrogram
        stego_spec = cover_spec + self.alpha * torch.tanh(residual)
        return stego_spec


class Decoder(nn.Module):
    """Convolutional decoder recovering raw bit logits from stego spectrograms.

    Args:
        payload_bits: Number of payload bits to predict (default 128).
        base: Base channel multiplier for convolutions (default 32).
    """

    def __init__(
        self,
        payload_bits: int = PAYLOAD_BITS,
        base: int = 32,
    ) -> None:
        super().__init__()
        self.payload_bits = payload_bits
        self.base = base

        self.features = nn.Sequential(
            nn.Conv2d(STFT_CHANNELS, base, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 4, base * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 8, base * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(base * 8 * 4 * 4, payload_bits)

    def forward(self, stego_spec: torch.Tensor) -> torch.Tensor:
        """Extract raw bit logits from the stego spectrogram.

        Args:
            stego_spec: Stego spectrogram tensor [B, 2, 513, 126].

        Returns:
            torch.Tensor: Raw bit logits tensor [B, 128] (without sigmoid).
        """
        feats = self.features(stego_spec)
        pooled = self.pool(feats)
        flat = torch.flatten(pooled, start_dim=1)
        logits = self.fc(flat)
        return logits


class Discriminator(nn.Module):
    """Spectral-normalized convolutional discriminator for audio spectrograms.

    Args:
        base: Base channel multiplier for convolutions (default 32).
    """

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.base = base

        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(STFT_CHANNELS, base, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base * 4, base * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base * 8, base * 8, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = spectral_norm(nn.Linear(base * 8, 1))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Classify spectrogram as real cover vs. generated stego.

        Args:
            spec: Spectrogram tensor [B, 2, 513, 126].

        Returns:
            torch.Tensor: Single raw scalar logit per sample [B, 1].
        """
        feats = self.features(spec)
        pooled = self.global_pool(feats)
        flat = torch.flatten(pooled, start_dim=1)
        logit = self.classifier(flat)
        return logit
