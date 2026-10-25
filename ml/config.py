"""Configuration constants for Neural Audio Steganography.

This module provides all signal processing parameters, STFT dimensions,
audio constraints, and payload specifications for the GAN steganography system.
Derived values are computed dynamically from fundamental audio parameters.
"""

from typing import Final, Tuple

# Audio representation parameters
SAMPLE_RATE: Final[int] = 16000
AUDIO_CHANNELS: Final[int] = 1  # Mono audio
CHUNK_DURATION_SEC: Final[float] = 2.0
CHUNK_SAMPLES: Final[int] = int(SAMPLE_RATE * CHUNK_DURATION_SEC)  # 32000 samples

# Audio amplitude bounds for normalized float32 format
AUDIO_MIN_VAL: Final[float] = -1.0
AUDIO_MAX_VAL: Final[float] = 1.0

# Short-Time Fourier Transform (STFT) configuration
N_FFT: Final[int] = 1024
HOP_LENGTH: Final[int] = 256
HOP: Final[int] = HOP_LENGTH
WIN_LENGTH: Final[int] = 1024
CENTER: Final[bool] = True
WINDOW_TYPE: Final[str] = "hann"

# Spectrogram dimensions (computed from signal parameters)
# For real FFT (one-sided), frequency bins = N_FFT // 2 + 1
FREQ_BINS: Final[int] = N_FFT // 2 + 1  # 513

# When center=True, signal is padded symmetrically by N_FFT // 2 on both ends.
# Padded samples = CHUNK_SAMPLES + 2 * (N_FFT // 2) = CHUNK_SAMPLES + N_FFT.
# Frame count formula: (padded_samples - N_FFT) // HOP_LENGTH + 1 = CHUNK_SAMPLES // HOP_LENGTH + 1.
N_FRAMES: Final[int] = (
    (CHUNK_SAMPLES // HOP_LENGTH + 1)
    if CENTER
    else ((CHUNK_SAMPLES - N_FFT) // HOP_LENGTH + 1)
)  # 126

# Spectrogram channels: 2 representing (real, imag) components
STFT_CHANNELS: Final[int] = 2
SPECTROGRAM_SHAPE: Final[Tuple[int, int, int]] = (
    STFT_CHANNELS,
    FREQ_BINS,
    N_FRAMES,
)  # (2, 513, 126)

# Steganography payload specifications
PAYLOAD_BITS: Final[int] = 128
PAYLOAD_BYTES: Final[int] = PAYLOAD_BITS // 8  # 16 bytes

# Output audio specifications
OUTPUT_FORMAT: Final[str] = "WAV"
OUTPUT_SUBTYPE: Final[str] = "PCM_16"
OUTPUT_SAMPLE_RATE: Final[int] = SAMPLE_RATE
OUTPUT_BITS_PER_SAMPLE: Final[int] = 16

# Supported input audio formats
SUPPORTED_INPUT_EXTENSIONS: Final[Tuple[str, ...]] = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
)
