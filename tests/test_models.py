"""Unit tests for GAN models: Encoder, Decoder, and Discriminator."""

import torch

from ml.config import FREQ_BINS, N_FRAMES, PAYLOAD_BITS, STFT_CHANNELS
from ml.models import Decoder, Discriminator, Encoder

MAX_PARAM_LIMIT = 10_000_000
BATCH_SIZE = 2


def test_encoder_output_shape_and_params():
    """Verify Encoder output shape [B, 2, 513, 126] and parameter count < 10M."""
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.02)
    encoder.eval()

    param_count = sum(p.numel() for p in encoder.parameters())
    assert param_count < MAX_PARAM_LIMIT, (
        f"Encoder parameters {param_count:,} exceeded limit of {MAX_PARAM_LIMIT:,}"
    )

    cover_spec = torch.randn(BATCH_SIZE, STFT_CHANNELS, FREQ_BINS, N_FRAMES)
    message_bits = torch.randint(0, 2, (BATCH_SIZE, PAYLOAD_BITS)).float()

    with torch.no_grad():
        stego_spec = encoder(cover_spec, message_bits)

    assert stego_spec.shape == (BATCH_SIZE, STFT_CHANNELS, FREQ_BINS, N_FRAMES)
    assert stego_spec.shape == (2, 2, 513, 126)


def test_decoder_output_shape_and_params():
    """Verify Decoder output shape [B, 128] raw logits and parameter count < 10M."""
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32)
    decoder.eval()

    param_count = sum(p.numel() for p in decoder.parameters())
    assert param_count < MAX_PARAM_LIMIT, (
        f"Decoder parameters {param_count:,} exceeded limit of {MAX_PARAM_LIMIT:,}"
    )

    stego_spec = torch.randn(BATCH_SIZE, STFT_CHANNELS, FREQ_BINS, N_FRAMES)

    with torch.no_grad():
        logits = decoder(stego_spec)

    assert logits.shape == (BATCH_SIZE, PAYLOAD_BITS)
    assert logits.shape == (2, 128)


def test_discriminator_output_shape_and_params():
    """Verify Discriminator output shape [B, 1] raw logit and parameter count < 10M."""
    discriminator = Discriminator(base=32)
    discriminator.eval()

    param_count = sum(p.numel() for p in discriminator.parameters())
    assert param_count < MAX_PARAM_LIMIT, (
        f"Discriminator parameters {param_count:,} exceeded limit of {MAX_PARAM_LIMIT:,}"
    )

    spec = torch.randn(BATCH_SIZE, STFT_CHANNELS, FREQ_BINS, N_FRAMES)

    with torch.no_grad():
        out = discriminator(spec)

    assert out.shape == (BATCH_SIZE, 1)
    assert out.shape == (2, 1)


def test_gradient_flow_backward():
    """Verify forward and backward passes operate without shape or gradient errors."""
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.02)
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32)
    discriminator = Discriminator(base=32)

    encoder.train()
    decoder.train()
    discriminator.train()

    cover_spec = torch.randn(BATCH_SIZE, STFT_CHANNELS, FREQ_BINS, N_FRAMES, requires_grad=True)
    message_bits = torch.randint(0, 2, (BATCH_SIZE, PAYLOAD_BITS)).float()

    # Forward through encoder
    stego_spec = encoder(cover_spec, message_bits)
    assert stego_spec.shape == cover_spec.shape

    # Forward through decoder
    decoded_logits = decoder(stego_spec)
    assert decoded_logits.shape == (BATCH_SIZE, PAYLOAD_BITS)

    # Forward through discriminator
    disc_real = discriminator(cover_spec)
    disc_fake = discriminator(stego_spec)
    assert disc_real.shape == (BATCH_SIZE, 1)
    assert disc_fake.shape == (BATCH_SIZE, 1)

    # Backward pass
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(decoded_logits, message_bits)
    bce_loss.backward()

    # Verify gradients computed
    assert encoder.final_conv.weight.grad is not None
    assert decoder.fc.weight.grad is not None
