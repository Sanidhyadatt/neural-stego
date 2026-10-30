"""Inference module and CLI for Neural Audio Steganography.

Provides:
- StegoEngine: High-level inference engine for embedding and extracting messages.
- Command-line interface:
    python -m ml.inference embed in.wav "secret" out.wav
    python -m ml.inference extract out.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ml.audio_utils import (
    istft,
    load_audio,
    merge_chunks,
    save_wav_bytes,
    split_chunks,
    stft,
)
from ml.config import (
    CHUNK_DURATION_SEC,
    CHUNK_SAMPLES,
    PAYLOAD_BITS,
    SAMPLE_RATE,
)
from ml.models import Decoder, Encoder
from ml.text_codec import chunk_bits_to_text, max_text_bytes, text_to_chunk_bits
from ml.train import get_device


class StegoEngine:
    """Production inference engine for audio steganography embedding and extraction.

    Args:
        checkpoint_path: Path to PyTorch model weights (.pt). Defaults to 'checkpoints/stego.pt'.
        device: Target compute device ('auto', 'cuda', 'mps', or 'cpu').
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = "checkpoints/stego.pt",
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = get_device(device)
        self.encoder: Optional[Encoder] = None
        self.decoder: Optional[Decoder] = None

        if self.checkpoint_path and self.checkpoint_path.is_file():
            self.load()

    def load(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """Load encoder and decoder weights from the checkpoint into eval mode.

        Args:
            checkpoint_path: Optional path to override default checkpoint.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path or not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        ema_state = ckpt.get("ema") if isinstance(ckpt.get("ema"), dict) else {}

        payload_bits = cfg.get("payload_bits", PAYLOAD_BITS)
        base = cfg.get("base", 32)
        alpha = ema_state.get("alpha", cfg.get("alpha", 0.02))

        self.encoder = Encoder(payload_bits=payload_bits, base=base, alpha=alpha).to(self.device)
        self.decoder = Decoder(payload_bits=payload_bits, base=base).to(self.device)

        encoder_weights = ema_state.get("encoder") or ckpt.get("ema_encoder") or ckpt["encoder"]
        decoder_weights = ema_state.get("decoder") or ckpt.get("ema_decoder") or ckpt["decoder"]

        self.encoder.load_state_dict(encoder_weights)
        self.decoder.load_state_dict(decoder_weights)

        self.encoder.eval()
        self.decoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(
        self,
        audio_bytes: bytes,
        text: str,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Embed a secret text string into an input audio stream.

        Args:
            audio_bytes: Input audio data in any supported format (WAV, MP3, FLAC, OGG).
            text: Secret text string to conceal.

        Returns:
            Tuple[bytes, dict]:
                - wav_bytes: 16-bit PCM WAV bytes containing the hidden message.
                - info: Execution metadata with n_chunks, duration_sec, snr_db, capacity_chars.

        Raises:
            ValueError: If message requires more blocks than available chunks in the audio,
                or exceeds codec maximum payload length.
        """
        if self.encoder is None:
            self.load()
            assert self.encoder is not None

        # 1. Load audio, convert to mono 16 kHz float32 [-1, 1]
        wave = load_audio(audio_bytes)
        total_samples = wave.shape[0]
        duration_sec = total_samples / SAMPLE_RATE

        # 2. Split into 2-second chunks (32,000 samples)
        chunks, original_length = split_chunks(wave)
        n_chunks = chunks.shape[0]
        capacity_chars = max_text_bytes(n_chunks)

        # 3. Convert text to 16-byte message blocks [n_blocks, 128]
        text_bits = text_to_chunk_bits(text)
        n_blocks = text_bits.shape[0]

        # 4. Check block count against available audio chunks
        if n_blocks > n_chunks:
            needed_sec = int(n_blocks * CHUNK_DURATION_SEC)
            raise ValueError(
                f"Audio too short for this message. Needs at least {needed_sec} seconds."
            )

        # 5. Tile message blocks cyclically across ALL chunks (chunk i carries block i % n_blocks)
        full_bits = torch.stack([text_bits[i % n_blocks] for i in range(n_chunks)], dim=0)

        # 6. Encode all chunks in batches of 8
        batch_size = 8
        stego_chunks_list: List[torch.Tensor] = []

        for i in range(0, n_chunks, batch_size):
            sub_chunks = chunks[i : i + batch_size].to(self.device)
            sub_bits = full_bits[i : i + batch_size].to(self.device)

            spec = stft(sub_chunks)
            stego_spec = self.encoder(spec, sub_bits)
            stego_sub_chunks = istft(stego_spec)
            stego_chunks_list.append(stego_sub_chunks.cpu())

        stego_chunks = torch.cat(stego_chunks_list, dim=0)  # [n_chunks, 32000]

        # 7. Compute Signal-to-Noise Ratio (SNR) in dB
        orig_flat = chunks.reshape(-1)[:original_length]
        stego_flat = stego_chunks.reshape(-1)[:original_length]
        noise_flat = stego_flat - orig_flat

        sig_power = torch.mean(orig_flat ** 2).item()
        noise_power = torch.mean(noise_flat ** 2).item()
        snr_db = 10.0 * float(np.log10(sig_power / (noise_power + 1e-10)))

        # 8. Merge chunks, clip to [-1, 1], and serialize to 16-bit PCM WAV
        merged = merge_chunks(stego_chunks, original_length)
        clipped = torch.clamp(merged, -1.0, 1.0)
        wav_bytes = save_wav_bytes(clipped)

        info = {
            "n_chunks": n_chunks,
            "duration_sec": round(duration_sec, 2),
            "snr_db": round(snr_db, 2),
            "capacity_chars": capacity_chars,
        }
        return wav_bytes, info

    def _decode_waveform_chunks(self, wave: torch.Tensor) -> Optional[str]:
        """Compute chunk sigmoid probabilities, vote modulo candidate n_blocks, and try decoding."""
        assert self.decoder is not None
        chunks, _ = split_chunks(wave)
        n_chunks = chunks.shape[0]
        if n_chunks == 0:
            return None

        batch_size = 16
        logits_list: List[torch.Tensor] = []

        for i in range(0, n_chunks, batch_size):
            sub_chunks = chunks[i : i + batch_size].to(self.device)
            spec = stft(sub_chunks)
            logits = self.decoder(spec)  # [B, 128]
            logits_list.append(logits.cpu())

        all_logits = torch.cat(logits_list, dim=0)  # [n_chunks, 128]

        # (a) Compute sigmoid probabilities for every chunk
        probs = torch.sigmoid(all_logits)

        # (b) For each candidate n_blocks from 1 to n_chunks, average probabilities of chunks
        # sharing the same index modulo n_blocks, threshold at 0.5, and try to decode.
        for cand_n_blocks in range(1, n_chunks + 1):
            avg_probs = torch.zeros((cand_n_blocks, PAYLOAD_BITS), dtype=torch.float32)
            for b in range(cand_n_blocks):
                avg_probs[b] = probs[b::cand_n_blocks].mean(dim=0)
            voted_bits = (avg_probs >= 0.5).float()
            try:
                return chunk_bits_to_text(voted_bits)
            except ValueError:
                continue

        return None

    @torch.no_grad()
    def extract(self, audio_bytes: bytes) -> str:
        """Extract secret message text from stego audio bytes.

        Args:
            audio_bytes: Stego audio data in any supported format.

        Returns:
            str: Decoded secret message string.

        Raises:
            ValueError: With "Message could not be recovered" if extraction fails.
        """
        if self.decoder is None:
            self.load()
            assert self.decoder is not None

        wave = load_audio(audio_bytes)
        if wave.numel() == 0:
            raise ValueError("Message could not be recovered")

        # Initial attempt (shift = 0): steps (a) and (b)
        decoded = self._decode_waveform_chunks(wave)
        if decoded is not None:
            return decoded

        # (c) Retry with audio shifted by 0 to 31999 samples (step 2000, crop and zero-pad,
        # do not wrap around) and repeat (b). Shift 0 was already attempted above.
        orig_chunks, _ = split_chunks(wave)
        padded_len = orig_chunks.shape[0] * CHUNK_SAMPLES

        for shift in range(2000, CHUNK_SAMPLES, 2000):
            # Zero-pad start and crop to chunk-aligned length (re-aligns start-trimmed clips)
            shifted_pad_start = F.pad(wave, (shift, 0), mode="constant", value=0.0)[:padded_len]
            decoded = self._decode_waveform_chunks(shifted_pad_start)
            if decoded is not None:
                return decoded

            # Crop start and zero-pad tail via split_chunks (re-aligns start-padded clips)
            if wave.shape[0] > shift:
                shifted_crop_start = wave[shift:]
                decoded = self._decode_waveform_chunks(shifted_crop_start)
                if decoded is not None:
                    return decoded

        # (d) Nothing passed magic, RS, and CRC
        raise ValueError("Message could not be recovered")

    def capacity(self, audio_bytes: bytes) -> int:
        """Calculate maximum UTF-8 byte capacity for an audio file.

        Args:
            audio_bytes: Audio file binary content.

        Returns:
            int: Maximum allowable UTF-8 byte payload length.
        """
        wave = load_audio(audio_bytes)
        chunks, _ = split_chunks(wave)
        return max_text_bytes(chunks.shape[0])


def main() -> None:
    """Command-line interface for Neural Audio Steganography inference."""
    parser = argparse.ArgumentParser(description="Neural Audio Steganography CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Embed command: embed in.wav "secret" out.wav
    embed_p = subparsers.add_parser("embed", help="Embed secret text into audio file")
    embed_p.add_argument("input_audio", type=str, help="Path to input audio file")
    embed_p.add_argument("secret_text", type=str, help="Secret text message to conceal")
    embed_p.add_argument("output_audio", type=str, help="Destination path for output stego WAV")
    embed_p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stego.pt",
        help="Path to model weights checkpoint",
    )
    embed_p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cpu)",
    )

    # Extract command: extract out.wav
    extract_p = subparsers.add_parser("extract", help="Extract secret text from stego audio")
    extract_p.add_argument("stego_audio", type=str, help="Path to stego audio file")
    extract_p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stego.pt",
        help="Path to model weights checkpoint",
    )
    extract_p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cpu)",
    )

    # Capacity command: capacity in.wav
    cap_p = subparsers.add_parser("capacity", help="Check maximum text capacity of an audio file")
    cap_p.add_argument("input_audio", type=str, help="Path to audio file")
    cap_p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cpu)",
    )

    args = parser.parse_args()

    if args.command == "embed":
        engine = StegoEngine(checkpoint_path=args.checkpoint, device=args.device)
        input_data = Path(args.input_audio).read_bytes()
        print(f"[Embed] Processing {args.input_audio}...")
        stego_bytes, info = engine.embed(input_data, args.secret_text)
        Path(args.output_audio).write_bytes(stego_bytes)
        print(f"[Embed] Successfully wrote stego audio to {args.output_audio}")
        print(
            f"[Embed] Chunks: {info['n_chunks']} | "
            f"Duration: {info['duration_sec']}s | "
            f"SNR: {info['snr_db']} dB | "
            f"Capacity: {info['capacity_chars']} bytes"
        )

    elif args.command == "extract":
        engine = StegoEngine(checkpoint_path=args.checkpoint, device=args.device)
        stego_data = Path(args.stego_audio).read_bytes()
        print(f"[Extract] Extracting message from {args.stego_audio}...")
        try:
            extracted_text = engine.extract(stego_data)
            print(f"[Extract] Secret Message: {extracted_text}")
        except ValueError as err:
            print(f"[Extract Error] {err}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "capacity":
        engine = StegoEngine(checkpoint_path=None, device=args.device)
        input_data = Path(args.input_audio).read_bytes()
        max_chars = engine.capacity(input_data)
        print(f"[Capacity] Max text length: {max_chars} bytes.")


if __name__ == "__main__":
    main()
