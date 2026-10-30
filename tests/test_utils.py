"""Unit tests for audio processing and message framing text codec."""

import reedsolo
import pytest
import torch
import numpy as np

from ml.audio_utils import (
    istft,
    load_audio,
    merge_chunks,
    save_wav_bytes,
    split_chunks,
    stft,
)
from ml.config import CHUNK_SAMPLES, FREQ_BINS, N_FRAMES, PAYLOAD_BITS, STFT_CHANNELS
from ml.text_codec import (
    MAGIC_BYTE,
    RS_PARITY_BYTES,
    chunk_bits_to_text,
    crc16,
    max_text_bytes,
    text_to_chunk_bits,
)


def test_stft_istft_roundtrip():
    """Verify STFT to iSTFT reconstruction roundtrip error is < 1e-4."""
    batch_size = 4
    torch.manual_seed(42)
    wave_batch = torch.rand(batch_size, CHUNK_SAMPLES, dtype=torch.float32) - 0.5

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

        remainder = length % CHUNK_SAMPLES
        if remainder != 0:
            padded_tail = chunks[-1, remainder:]
            assert torch.all(padded_tail == 0.0)

        merged = merge_chunks(chunks, orig_len)
        assert merged.shape == wave.shape
        assert torch.equal(merged, wave)


def test_text_codec_roundtrip():
    """Verify codec roundtrip for 'hi', a 100-character string, and an emoji string."""
    hundred_char_str = "0123456789" * 10
    assert len(hundred_char_str) == 100

    emoji_str = "Audio steganography with emojis! 🎵🎧🔒🚀✨"

    for text in ["hi", hundred_char_str, emoji_str]:
        bits = text_to_chunk_bits(text)
        assert bits.ndim == 2
        assert bits.shape[1] == PAYLOAD_BITS
        assert torch.all((bits == 0.0) | (bits == 1.0))

        # Verify raw frame prefix starts with magic 0x53 and UTF-8 byte length
        raw_bytes = bytes(np.packbits(bits.numpy().astype(np.uint8).reshape(-1)).tobytes())
        payload_utf8 = text.encode("utf-8")
        assert raw_bytes[0] == MAGIC_BYTE
        assert raw_bytes[1] == len(payload_utf8)

        recovered = chunk_bits_to_text(bits)
        assert recovered == text, f"Failed roundtrip for text: {text!r}"


def test_corrupt_25_percent_of_bits_in_one_block_recovers():
    """Verify corrupting 25% of the bits (32 bits / 4 bytes) in one block still recovers."""
    messages = [
        "hi",
        "0123456789" * 10,
        "Emoji recovery! 🎵🎧🔒",
    ]

    for msg in messages:
        bits = text_to_chunk_bits(msg)
        # Corrupt 25% of block 0 (first 32 bits = 4 bytes, including magic & length!)
        corrupted_header_block = bits.clone()
        corrupted_header_block[0, :32] = 1.0 - corrupted_header_block[0, :32]
        assert chunk_bits_to_text(corrupted_header_block) == msg

        # Also corrupt 25% of the last block (middle 32 bits = bytes 4..7)
        corrupted_last_block = bits.clone()
        corrupted_last_block[-1, 32:64] = 1.0 - corrupted_last_block[-1, 32:64]
        assert chunk_bits_to_text(corrupted_last_block) == msg


def test_magic_and_crc_required_for_decode():
    """Verify decoding fails unless RS decodes AND magic matches AND CRC passes."""
    rsc = reedsolo.RSCodec(RS_PARITY_BYTES)
    payload = b"hello"
    valid_crc = crc16(payload).to_bytes(2, "big")

    # 1. Wrong magic byte (0x54 instead of 0x53), valid RS and valid CRC
    bad_magic_frame = bytes([0x54, len(payload)]) + payload + valid_crc
    encoded_bad_magic = bytes(rsc.encode(bad_magic_frame)).ljust(32, b"\x00")
    bits_bad_magic = torch.from_numpy(
        np.unpackbits(np.frombuffer(encoded_bad_magic, dtype=np.uint8)).copy()
    ).float().view(2, 128)
    with pytest.raises(ValueError, match="Message could not be recovered"):
        chunk_bits_to_text(bits_bad_magic)

    # 2. Valid magic byte (0x53) and valid RS, but wrong CRC16
    bad_crc = ((crc16(payload) ^ 0xFFFF) & 0xFFFF).to_bytes(2, "big")
    bad_crc_frame = bytes([MAGIC_BYTE, len(payload)]) + payload + bad_crc
    encoded_bad_crc = bytes(rsc.encode(bad_crc_frame)).ljust(32, b"\x00")
    bits_bad_crc = torch.from_numpy(
        np.unpackbits(np.frombuffer(encoded_bad_crc, dtype=np.uint8)).copy()
    ).float().view(2, 128)
    with pytest.raises(ValueError, match="Message could not be recovered"):
        chunk_bits_to_text(bits_bad_crc)


def test_max_text_bytes():
    """Verify max_text_bytes(n_chunks) = n_chunks*16 - 12 and boundary enforcement."""
    assert max_text_bytes(0) == 0
    assert max_text_bytes(1) == 4
    assert max_text_bytes(2) == 20
    assert max_text_bytes(3) == 36
    assert max_text_bytes(7) == 100
    assert max_text_bytes(30) == 255  # 1-byte length cap

    valid_1_chunk = "abcd"  # 4 bytes
    bits = text_to_chunk_bits(valid_1_chunk, n_chunks=1)
    assert bits.shape == (1, 128)
    assert chunk_bits_to_text(bits) == valid_1_chunk

    with pytest.raises(ValueError):
        text_to_chunk_bits("abcde", n_chunks=1)  # 5 bytes > 4

    with pytest.raises(ValueError):
        text_to_chunk_bits("x" * 256)  # exceeds 255-byte frame limit


def test_save_and_load_audio():
    """Verify audio serialization to 16-bit PCM WAV and deserialization."""
    torch.manual_seed(999)
    original_wave = torch.rand(32000, dtype=torch.float32) * 1.8 - 0.9

    wav_bytes = save_wav_bytes(original_wave)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0

    loaded_wave = load_audio(wav_bytes)
    assert loaded_wave.shape == original_wave.shape
    assert torch.allclose(loaded_wave, original_wave, atol=1e-4)


def test_unrecoverable_message_raises():
    """Verify that random bit payloads raise ValueError('Message could not be recovered')."""
    torch.manual_seed(7)
    random_bits = torch.randint(0, 2, (2, 128)).float()
    with pytest.raises(ValueError, match="Message could not be recovered"):
        chunk_bits_to_text(random_bits)
