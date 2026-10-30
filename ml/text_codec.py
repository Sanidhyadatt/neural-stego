"""Text codec for Neural Audio Steganography.

Message frame specification:
- 1-byte magic: 0x53
- 1-byte length: UTF-8 payload byte count (0..255)
- Payload: UTF-8 encoded message bytes
- 2-byte CRC16: Big-endian CRC-16-CCITT (0xFFFF init) of the payload
- Reed-Solomon FEC: 8 parity bytes over the whole frame
- Padding: Zero-padded to a multiple of 16 bytes, split into 16-byte blocks,
  and unpacked to bits of shape [n_blocks, 128].
"""

from __future__ import annotations

import binascii
from typing import Final, Optional

import numpy as np
import reedsolo
import torch

from ml.config import PAYLOAD_BITS, PAYLOAD_BYTES

# Frame protocol constants
MAGIC_BYTE: Final[int] = 0x53
MAGIC_BYTES: Final[int] = 1
LENGTH_BYTES: Final[int] = 1
CRC_BYTES: Final[int] = 2
MAX_PAYLOAD_BYTES: Final[int] = 255
RS_PARITY_BYTES: Final[int] = 8
RS_MAX_DATA_PER_BLOCK: Final[int] = 255 - RS_PARITY_BYTES  # 247 bytes in GF(2^8)
FRAME_HEADER_OVERHEAD: Final[int] = MAGIC_BYTES + LENGTH_BYTES + CRC_BYTES  # 4 bytes
FRAME_OVERHEAD: Final[int] = FRAME_HEADER_OVERHEAD + RS_PARITY_BYTES  # 12 bytes
BLOCK_BYTES: Final[int] = PAYLOAD_BYTES  # 16 bytes = 128 bits
BITS_PER_BYTE: Final[int] = 8

_RSC: Final[reedsolo.RSCodec] = reedsolo.RSCodec(RS_PARITY_BYTES)


def crc16(data: bytes) -> int:
    """Compute 16-bit CRC-CCITT (polynomial 0x1021, initial value 0xFFFF)."""
    return binascii.crc_hqx(data, 0xFFFF)


def _rs_encoded_length(payload_len: int) -> int:
    """Compute exact byte length after Reed-Solomon encoding for a given payload length."""
    frame_len = payload_len + FRAME_HEADER_OVERHEAD
    n_rs_blocks = (frame_len + RS_MAX_DATA_PER_BLOCK - 1) // RS_MAX_DATA_PER_BLOCK
    return frame_len + RS_PARITY_BYTES * n_rs_blocks


def max_text_bytes(n_chunks: int) -> int:
    """Compute the maximum UTF-8 payload byte length that fits in n_chunks.

    Uses n_chunks * 16 - 12 (the 12-byte frame overhead), clamped to [0, 255].

    Args:
        n_chunks: Number of 2-second audio chunks available.

    Returns:
        int: Maximum UTF-8 payload byte length.
    """
    if n_chunks <= 0:
        return 0
    return min(MAX_PAYLOAD_BYTES, max(0, n_chunks * BLOCK_BYTES - FRAME_OVERHEAD))


# Backward-compatible alias
max_text_length = max_text_bytes


def text_to_chunk_bits(text: str, n_chunks: Optional[int] = None) -> torch.Tensor:
    """Encode a UTF-8 string into a [n_blocks, 128] binary float32 tensor.

    Framing steps:
        1. Encode text as UTF-8 bytes (max 255 bytes).
        2. Build frame: [0x53, length] + payload + CRC16(payload).
        3. Apply Reed-Solomon encoding with 8 parity bytes over the whole frame.
        4. Zero-pad to a multiple of 16 bytes and split into 16-byte blocks.
        5. Convert blocks to binary float32 tensor of shape [n_blocks, 128].

    Args:
        text: Input string to encode.
        n_chunks: Optional target block count to validate against and zero-pad out to.

    Returns:
        torch.Tensor: Float32 tensor of shape [n_blocks, 128] with values in {0.0, 1.0}.

    Raises:
        ValueError: If UTF-8 payload exceeds 255 bytes or exceeds target n_chunks.
    """
    payload_bytes = text.encode("utf-8")
    payload_len = len(payload_bytes)

    if payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Text UTF-8 byte length ({payload_len}) exceeds 1-byte maximum of {MAX_PAYLOAD_BYTES} bytes."
        )

    payload_crc = crc16(payload_bytes).to_bytes(CRC_BYTES, byteorder="big")
    frame = bytes([MAGIC_BYTE, payload_len]) + payload_bytes + payload_crc

    encoded_rs = bytes(_RSC.encode(frame))

    remainder = len(encoded_rs) % BLOCK_BYTES
    if remainder != 0:
        pad_len = BLOCK_BYTES - remainder
        padded_bytes = encoded_rs + (b"\x00" * pad_len)
    else:
        padded_bytes = encoded_rs

    n_blocks = len(padded_bytes) // BLOCK_BYTES

    if n_chunks is not None:
        if n_chunks < n_blocks:
            max_allowed = max_text_bytes(n_chunks)
            raise ValueError(
                f"Payload requires {n_blocks} blocks ({payload_len} bytes), "
                f"but only {n_chunks} chunks are available (max {max_allowed} bytes)."
            )
        if n_chunks > n_blocks:
            extra_padding = (n_chunks - n_blocks) * BLOCK_BYTES
            padded_bytes = padded_bytes + (b"\x00" * extra_padding)
            n_blocks = n_chunks

    byte_arr = np.frombuffer(padded_bytes, dtype=np.uint8)
    bit_arr = np.unpackbits(byte_arr)
    chunk_bits = torch.from_numpy(bit_arr.copy()).float().view(n_blocks, PAYLOAD_BITS)
    return chunk_bits


def _try_decode_candidate(raw_bytes: bytes, cand_payload_len: int) -> Optional[str]:
    """Attempt RS decode, magic check, length check, CRC16 verification, and UTF-8 decode."""
    if cand_payload_len < 0 or cand_payload_len > MAX_PAYLOAD_BYTES:
        return None

    cand_encoded_len = _rs_encoded_length(cand_payload_len)
    if cand_encoded_len > len(raw_bytes):
        return None

    candidate = raw_bytes[:cand_encoded_len]
    try:
        res = _RSC.decode(candidate)
        decoded_frame = bytes(res[0] if isinstance(res, tuple) else res)
    except Exception:
        return None

    if len(decoded_frame) != cand_payload_len + FRAME_HEADER_OVERHEAD:
        return None

    # Verify magic byte and length field
    if decoded_frame[0] != MAGIC_BYTE or decoded_frame[1] != cand_payload_len:
        return None

    payload = decoded_frame[2 : 2 + cand_payload_len]
    expected_crc = int.from_bytes(
        decoded_frame[2 + cand_payload_len : 4 + cand_payload_len],
        byteorder="big",
    )
    if crc16(payload) != expected_crc:
        return None

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def chunk_bits_to_text(bits: torch.Tensor) -> str:
    """Decode a [n_blocks, 128] bit tensor back into the original UTF-8 text.

    Decoding succeeds only if Reed-Solomon decoding succeeds AND magic byte (0x53)
    matches AND CRC16 of the payload passes. Otherwise raises ValueError.

    Args:
        bits: Float32 tensor of shape [n_blocks, 128] or compatible flattened tensor.

    Returns:
        str: Recovered UTF-8 text string.

    Raises:
        ValueError: Always with "Message could not be recovered" on any decoding failure.
    """
    if bits.numel() == 0:
        raise ValueError("Message could not be recovered")

    bit_bool = (bits >= 0.5).to(torch.uint8)
    flat_bits = bit_bool.detach().cpu().numpy().reshape(-1)

    if flat_bits.size % BITS_PER_BYTE != 0:
        raise ValueError("Message could not be recovered")

    raw_bytes = bytes(np.packbits(flat_bits).tobytes())
    total_bytes = len(raw_bytes)

    if total_bytes < FRAME_OVERHEAD:
        raise ValueError("Message could not be recovered")

    tried_lengths: set[int] = set()

    # 1. Fast path: try length byte directly from raw frame header if uncorrupted
    hint_len = raw_bytes[1]
    tried_lengths.add(hint_len)
    decoded = _try_decode_candidate(raw_bytes, hint_len)
    if decoded is not None:
        return decoded

    # 2. Try all payload lengths whose padded RS frame lands in the final 16-byte block
    min_last_block_len = max(FRAME_OVERHEAD, total_bytes - (BLOCK_BYTES - 1))
    for cand_len in range(MAX_PAYLOAD_BYTES + 1):
        if cand_len in tried_lengths:
            continue
        enc_len = _rs_encoded_length(cand_len)
        if min_last_block_len <= enc_len <= total_bytes:
            tried_lengths.add(cand_len)
            decoded = _try_decode_candidate(raw_bytes, cand_len)
            if decoded is not None:
                return decoded

    # 3. Fallback for explicitly zero-padded multi-chunk tensors (e.g. n_chunks > n_blocks)
    for cand_len in range(MAX_PAYLOAD_BYTES + 1):
        if cand_len in tried_lengths:
            continue
        enc_len = _rs_encoded_length(cand_len)
        if enc_len < min_last_block_len:
            tail = raw_bytes[enc_len:]
            # Zero-padded tail can have at most RS_PARITY_BYTES // 2 corrupted bytes if recoverable
            non_zero_tail = len(tail) - tail.count(0)
            if non_zero_tail <= RS_PARITY_BYTES:
                decoded = _try_decode_candidate(raw_bytes, cand_len)
                if decoded is not None:
                    return decoded

    raise ValueError("Message could not be recovered")
