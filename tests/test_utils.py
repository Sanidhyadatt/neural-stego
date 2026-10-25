"""Unit tests for audio processing and text codec utilities."""

import random
import numpy as np
import pytest
import torch

from ml.audio_utils import (
    istft,
    load_audio,
    merge_chunks,
    save_wav_bytes,
    split_chunks,
    stft,
)
from ml.config import CHUNK_SAMPLES, FREQ_BINS, N_FRAMES, STFT_CHANNELS
from ml.text_codec import (
    chunk_bits_to_text,
    max_text_length,
    text_to_chunk_bits,
)


def test_stft_istft_roundtrip():
    """Verify STFT to iSTFT reconstruction roundtrip error is < 1e-4."""
    batch_size = 4
    torch.manual_seed(42)
    # Generate random audio batch in [-0.5, 0.5]
    wave_batch = (torch.rand(batch_size, CHUNK_SAMPLES, dtype=torch.float32) - 0.5)

    spec = stft(wave_batch)
    assert spec.shape == (batch_size, STFT_CHANNELS, FREQ_BINS, N_FRAMES)
    assert spec.shape == (batch_size, 2, 513, 126)

    reconstructed = istft(spec)
    assert reconstructed.shape == (batch_size, CHUNK_SAMPLES)

    max_error = (reconstructed - wave_batch).abs().max().item()
    assert max_error < 1e-4, f"STFT/iSTFT roundtrip error {max_error} exceeded 1e-4"


def test_split_merge_chunks():
    """Verify split_chunks followed by merge_chunks perfectly preserves signal and length."""
    test_lengths = [16000, 32000, 50000, 64000, 75432]
    torch.manual_seed(123)

    for length in test_lengths:
        wave = torch.randn(length, dtype=torch.float32)
        chunks, orig_len = split_chunks(wave)

        assert orig_len == length
        expected_chunks = (length + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
        assert chunks.shape == (expected_chunks, CHUNK_SAMPLES)

        # Check zero-padding in last chunk if length wasn't an exact multiple
        remainder = length % CHUNK_SAMPLES
        if remainder != 0:
            padded_tail = chunks[-1, remainder:]
            assert torch.all(padded_tail == 0.0)

        # Merge back and verify signal
        merged = merge_chunks(chunks, orig_len)
        assert merged.shape == wave.shape
        assert torch.equal(merged, wave)


def test_text_roundtrip():
    """Verify text encoding and decoding for 'hello', 100-char string, and emoji string."""
    test_strings = [
        "hello",
        "The quick brown fox jumps over the lazy dog. A neural steganography system hides data imperceptibly.",
        "Audio steganography with emojis! 🎵🎧🔒🚀✨",
    ]

    for text in test_strings:
        bits = text_to_chunk_bits(text)
        assert bits.ndim == 2
        assert bits.shape[1] == 128
        assert torch.all((bits == 0.0) | (bits == 1.0))

        recovered = chunk_bits_to_text(bits)
        assert recovered == text, f"Failed roundtrip for text: {text!r}"


def test_random_bit_flips_recovery():
    """Verify that flipping 3 random bits in encoded payload still decodes correctly."""
    messages = [
        "Secret message 123!",
        "Robust error correction test with Reed-Solomon parity.",
    ]

    for msg in messages:
        bits = text_to_chunk_bits(msg)
        total_bits = bits.numel()

        # Run multiple random flip trials
        for seed in [10, 20, 30, 40, 50]:
            rng = random.Random(seed)
            flip_indices = rng.sample(range(total_bits), 3)

            corrupted_bits = bits.clone().view(-1)
            for idx in flip_indices:
                corrupted_bits[idx] = 1.0 - corrupted_bits[idx]

            corrupted_bits = corrupted_bits.view(bits.shape)
            recovered = chunk_bits_to_text(corrupted_bits)
            assert recovered == msg, (
                f"Failed to recover {msg!r} after flipping bits {flip_indices}"
            )


def test_max_text_length():
    """Verify max_text_length boundary calculation and enforcement."""
    # 1 chunk = 16 bytes. Overhead = 2 (header) + 8 (RS) = 10 bytes. Capacity = 6 bytes.
    assert max_text_length(1) == 6
    assert max_text_length(2) == 22
    assert max_text_length(0) == 0

    valid_text_1_chunk = "123456"  # exactly 6 bytes
    bits = text_to_chunk_bits(valid_text_1_chunk, n_chunks=1)
    assert bits.shape == (1, 128)
    assert chunk_bits_to_text(bits) == valid_text_1_chunk

    invalid_text_1_chunk = "1234567"  # 7 bytes exceeds 6
    with pytest.raises(ValueError):
        text_to_chunk_bits(invalid_text_1_chunk, n_chunks=1)


def test_save_and_load_audio():
    """Verify audio serialization to 16-bit PCM WAV and deserialization."""
    torch.manual_seed(999)
    original_wave = (torch.rand(32000, dtype=torch.float32) * 1.8 - 0.9)  # in [-0.9, 0.9]

    wav_bytes = save_wav_bytes(original_wave)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0

    loaded_wave = load_audio(wav_bytes)
    assert loaded_wave.shape == original_wave.shape
    # 16-bit PCM quantization error is ~ 1 / 32768 ~= 3e-5
    assert torch.allclose(loaded_wave, original_wave, atol=1e-4)


def test_unrecoverable_message_raises():
    """Verify that completely corrupted or random bit payloads raise ValueError."""
    # 128 random bits
    random_bits = torch.randint(0, 2, (1, 128)).float()
    with pytest.raises(ValueError, match="Message could not be recovered"):
        chunk_bits_to_text(random_bits)
