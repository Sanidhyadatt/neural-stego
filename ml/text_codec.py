"""Text codec for Neural Audio Steganography.

Encodes UTF-8 strings into binary chunk bit tensors using a 2-byte big-endian
length header, Reed-Solomon (reedsolo) forward error correction with 8 parity bytes,
zero-padding to 16-byte boundaries (128 bits per chunk), and decodes with 0.5
thresholding and error correction.
"""

from __future__ import annotations

from typing import Final, Optional

import numpy as np
import reedsolo
import torch

from ml.config import PAYLOAD_BITS, PAYLOAD_BYTES

# Protocol constants
RS_PARITY_BYTES: Final[int] = 8
HEADER_BYTES: Final[int] = 2
CHUNK_BYTES: Final[int] = PAYLOAD_BYTES  # 16 bytes = 128 bits
BITS_PER_BYTE: Final[int] = 8


def max_text_length(n_chunks: int) -> int:
    """Compute the maximum UTF-8 payload length in bytes that can fit into n_chunks.

    Args:
        n_chunks: Number of 2-second audio chunks available.

    Returns:
        int: Maximum text byte length (equivalent to ASCII character count).
    """
    if n_chunks <= 0:
        return 0
    total_capacity = n_chunks * CHUNK_BYTES
    overhead = HEADER_BYTES + RS_PARITY_BYTES  # 10 bytes overhead
    return max(0, total_capacity - overhead)


def text_to_chunk_bits(text: str, n_chunks: Optional[int] = None) -> torch.Tensor:
    """Encode text into a [N_chunks, 128] binary float32 tensor.

    Protocol framing:
        1. Convert text to UTF-8 bytes.
        2. Prepend 2-byte big-endian length header.
        3. Apply Reed-Solomon encoding with 8 parity bytes over full framed message.
        4. Zero-pad to a multiple of 16 bytes (or pad to n_chunks * 16 if specified).
        5. Convert to bit representations of shape [N_chunks, 128].

    Args:
        text: Input string to encode.
        n_chunks: Optional target chunk count to pad out to.

    Returns:
        torch.Tensor: Float32 tensor of shape [N_chunks, 128] with binary values {0.0, 1.0}.

    Raises:
        ValueError: If text length exceeds maximum allowable capacity or target n_chunks.
    """
    data_bytes = text.encode("utf-8")
    payload_len = len(data_bytes)

    if payload_len > 65535:
        raise ValueError(
            f"Text byte length ({payload_len}) exceeds 2-byte header maximum of 65535 bytes."
        )

    # 2-byte big-endian length header + text data
    header = payload_len.to_bytes(HEADER_BYTES, byteorder="big")
    framed_message = header + data_bytes

    # Reed-Solomon encoding with 8 parity bytes
    rsc = reedsolo.RSCodec(RS_PARITY_BYTES)
    encoded_rs = bytes(rsc.encode(framed_message))

    # Pad with zeros to multiple of 16 bytes
    remainder = len(encoded_rs) % CHUNK_BYTES
    if remainder != 0:
        pad_len = CHUNK_BYTES - remainder
        padded_bytes = encoded_rs + (b"\x00" * pad_len)
    else:
        padded_bytes = encoded_rs

    required_chunks = len(padded_bytes) // CHUNK_BYTES

    if n_chunks is not None:
        if n_chunks < required_chunks:
            max_allowed = max_text_length(n_chunks)
            raise ValueError(
                f"Payload requires {required_chunks} chunks ({payload_len} bytes), "
                f"but only {n_chunks} chunks are available (max {max_allowed} bytes)."
            )
        if n_chunks > required_chunks:
            extra_padding = (n_chunks - required_chunks) * CHUNK_BYTES
            padded_bytes = padded_bytes + (b"\x00" * extra_padding)
            required_chunks = n_chunks

    # Unpack bytes into bits (MSB first)
    byte_arr = np.frombuffer(padded_bytes, dtype=np.uint8)
    bit_arr = np.unpackbits(byte_arr)  # uint8 array of {0, 1}

    # Reshape into [N_chunks, 128] float32 tensor
    chunk_bits = torch.from_numpy(bit_arr.copy()).float().view(required_chunks, PAYLOAD_BITS)
    return chunk_bits


def chunk_bits_to_text(bits: torch.Tensor) -> str:
    """Decode a [N_chunks, 128] bit tensor back into the secret text.

    Decoding process:
        1. Threshold bits at 0.5 to form binary values.
        2. Pack bits into a raw byte stream.
        3. Identify Reed-Solomon codeword boundaries and correct transmission errors.
        4. Validate 2-byte length header and decode UTF-8 string.

    Args:
        bits: Float32 tensor of shape [N_chunks, 128] or compatible flattened tensor.

    Returns:
        str: Decoded secret text.

    Raises:
        ValueError: If Reed-Solomon decoding fails or message cannot be recovered.
    """
    if bits.numel() == 0:
        return ""

    # Threshold at 0.5
    bit_bool = (bits >= 0.5).to(torch.uint8)
    flat_bits = bit_bool.detach().cpu().numpy().reshape(-1)

    if flat_bits.size % BITS_PER_BYTE != 0:
        raise ValueError("Bit count must be a multiple of 8 to decode into bytes.")

    raw_bytes = bytes(np.packbits(flat_bits).tobytes())
    total_bytes = len(raw_bytes)
    overhead = HEADER_BYTES + RS_PARITY_BYTES

    if total_bytes < overhead:
        raise ValueError("Message could not be recovered")

    rsc = reedsolo.RSCodec(RS_PARITY_BYTES)

    def _try_decode_candidate(cand_codeword_len: int) -> Optional[str]:
        if cand_codeword_len < overhead or cand_codeword_len > total_bytes:
            return None
        candidate_bytes = raw_bytes[:cand_codeword_len]
        try:
            res = rsc.decode(candidate_bytes)
            # reedsolo returns (decoded_message, decoded_msgecc, errata_pos)
            decoded_frame = bytes(res[0] if isinstance(res, tuple) else res)
            if len(decoded_frame) < HEADER_BYTES:
                return None
            msg_len = int.from_bytes(decoded_frame[:HEADER_BYTES], byteorder="big")
            if msg_len == len(decoded_frame) - HEADER_BYTES:
                msg_bytes = decoded_frame[HEADER_BYTES : HEADER_BYTES + msg_len]
                return msg_bytes.decode("utf-8")
        except Exception:
            return None
        return None

    # First attempt: Try extracting length from first 2 bytes if uncorrupted
    initial_length = int.from_bytes(raw_bytes[:HEADER_BYTES], byteorder="big")
    initial_codeword_len = initial_length + overhead
    decoded_text = _try_decode_candidate(initial_codeword_len)
    if decoded_text is not None:
        return decoded_text

    # Second attempt: Exhaustive search over all candidate codeword lengths
    # bounded by total chunk payload capacity
    for cand_len in range(overhead, total_bytes + 1):
        if cand_len == initial_codeword_len:
            continue
        decoded_text = _try_decode_candidate(cand_len)
        if decoded_text is not None:
            return decoded_text

    raise ValueError("Message could not be recovered")
