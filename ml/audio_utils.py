"""Audio utility functions for Neural Audio Steganography.

Provides robust audio loading, resampling to 16 kHz mono, splitting and merging
2-second chunks with padding/trimming, STFT/iSTFT transformations with Hann windowing,
and 16-bit PCM WAV encoding.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ml.config import (
    AUDIO_MAX_VAL,
    AUDIO_MIN_VAL,
    CENTER,
    CHUNK_SAMPLES,
    HOP_LENGTH,
    N_FFT,
    OUTPUT_FORMAT,
    OUTPUT_SUBTYPE,
    SAMPLE_RATE,
    WIN_LENGTH,
)


def load_audio(path_or_bytes: Union[str, Path, bytes, BinaryIO]) -> torch.Tensor:
    """Load an audio file or byte buffer, convert to mono, and resample to 16 kHz.

    Args:
        path_or_bytes: File path, raw audio bytes, or file-like binary stream.

    Returns:
        torch.Tensor: 1D float32 tensor of shape [T] normalized in [-1.0, 1.0].
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        buffer: Union[io.BytesIO, str, BinaryIO] = io.BytesIO(path_or_bytes)
    elif isinstance(path_or_bytes, Path):
        buffer = str(path_or_bytes)
    else:
        buffer = path_or_bytes

    waveform: torch.Tensor
    sr: int

    try:
        # First attempt using soundfile for broad cross-platform audio support
        data, sr = sf.read(buffer, dtype="float32")
        # soundfile returns [frames] for mono or [frames, channels] for multi-channel
        tensor_data = torch.from_numpy(data).float()
        if tensor_data.ndim == 1:
            waveform = tensor_data.unsqueeze(0)  # [1, T]
        else:
            # Multi-channel: shape [T, C] -> transpose to [C, T]
            waveform = tensor_data.transpose(0, 1)
    except Exception:
        # Fallback to torchaudio if soundfile cannot parse (e.g., certain container formats)
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        waveform, sr = torchaudio.load(buffer)
        waveform = waveform.float()

    # Convert to mono by averaging across all channels: [C, T] -> [1, T]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16000 Hz if necessary
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sr,
            new_freq=SAMPLE_RATE,
        )

    # Squeeze to 1D shape [T] and clamp within [-1.0, 1.0]
    mono_audio = waveform.squeeze(0).clamp(AUDIO_MIN_VAL, AUDIO_MAX_VAL)
    return mono_audio


def split_chunks(wave: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Split 1D audio waveform into 2-second chunks (32,000 samples) with zero-padding.

    Args:
        wave: Audio tensor of shape [T] or [1, T].

    Returns:
        Tuple[torch.Tensor, int]:
            - chunks: Float32 tensor of shape [N, 32000].
            - original_length: Original sample count before padding.
    """
    flat_wave = wave.squeeze()
    original_length = flat_wave.shape[0]

    if original_length == 0:
        return torch.zeros((0, CHUNK_SAMPLES), dtype=wave.dtype, device=wave.device), 0

    remainder = original_length % CHUNK_SAMPLES
    if remainder != 0:
        pad_needed = CHUNK_SAMPLES - remainder
        padded_wave = torch.nn.functional.pad(
            flat_wave,
            (0, pad_needed),
            mode="constant",
            value=0.0,
        )
    else:
        padded_wave = flat_wave

    num_chunks = padded_wave.shape[0] // CHUNK_SAMPLES
    chunks = padded_wave.view(num_chunks, CHUNK_SAMPLES)
    return chunks, original_length


def merge_chunks(chunks: torch.Tensor, original_length: int) -> torch.Tensor:
    """Merge [N, 32000] chunks back into a continuous waveform trimmed to original length.

    Args:
        chunks: Tensor of shape [N, 32000] or flattened equivalent.
        original_length: Number of samples in the original audio before zero-padding.

    Returns:
        torch.Tensor: Continuous 1D waveform of shape [original_length].
    """
    if chunks.numel() == 0 or original_length == 0:
        return torch.zeros((0,), dtype=chunks.dtype, device=chunks.device)

    flat_wave = chunks.reshape(-1)
    return flat_wave[:original_length]


def save_wav_bytes(wave: torch.Tensor) -> bytes:
    """Encode audio tensor into 16-bit PCM WAV byte stream.

    Args:
        wave: Audio tensor of shape [T] or [1, T] normalized in [-1.0, 1.0].

    Returns:
        bytes: 16-bit PCM WAV encoded binary data.
    """
    clamped_wave = wave.squeeze().detach().cpu().clamp(AUDIO_MIN_VAL, AUDIO_MAX_VAL)
    wave_np = clamped_wave.numpy().astype(np.float32)

    bio = io.BytesIO()
    sf.write(
        bio,
        wave_np,
        samplerate=SAMPLE_RATE,
        subtype=OUTPUT_SUBTYPE,
        format=OUTPUT_FORMAT,
    )
    return bio.getvalue()


def stft(wave_batch: torch.Tensor) -> torch.Tensor:
    """Compute Short-Time Fourier Transform for a batch of 2-second audio chunks.

    Args:
        wave_batch: Audio batch tensor of shape [B, 32000].

    Returns:
        torch.Tensor: Spectrogram tensor with 2 channels (real, imag) of shape [B, 2, 513, 126].
    """
    if wave_batch.ndim == 1:
        wave_batch = wave_batch.unsqueeze(0)

    device = wave_batch.device
    dtype = wave_batch.dtype
    window = torch.hann_window(WIN_LENGTH, device=device, dtype=dtype)

    complex_spec = torch.stft(
        wave_batch,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=CENTER,
        return_complex=True,
    )  # [B, 513, 126]

    # Stack real and imaginary components into channel dimension: [B, 2, 513, 126]
    return torch.stack([complex_spec.real, complex_spec.imag], dim=1)


def istft(spec: torch.Tensor) -> torch.Tensor:
    """Compute Inverse Short-Time Fourier Transform reconstructing audio chunks.

    Args:
        spec: Spectrogram tensor of shape [B, 2, 513, 126].

    Returns:
        torch.Tensor: Reconstructed audio waveform batch of shape [B, 32000].
    """
    real = spec[:, 0, :, :]
    imag = spec[:, 1, :, :]
    complex_spec = torch.complex(real, imag)

    device = spec.device
    dtype = real.dtype
    window = torch.hann_window(WIN_LENGTH, device=device, dtype=dtype)

    reconstructed_wave = torch.istft(
        complex_spec,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=CENTER,
        length=CHUNK_SAMPLES,
    )
    return reconstructed_wave
