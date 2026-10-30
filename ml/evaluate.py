"""End-to-end evaluation benchmark for Neural Audio Steganography.

Evaluates StegoEngine on held-out audio files (the last 2% of the sorted file list,
matching training's validation split) using N random 6-to-10-second clips and random
messages of 5 to 30 characters across five channel conditions:
1. clean (16-bit PCM WAV roundtrip)
2. noise std 0.002
3. gain 0.7x
4. 8 kHz resampling roundtrip (16 kHz -> 8 kHz -> 16 kHz)
5. MP3 128 kbps roundtrip via ffmpeg (skipped with a note if ffmpeg is missing)

Prints a formatted results table and finishes with PASS if clean exact-match rate >= 99.5%
and clean average SNR >= 30 dB, otherwise FAIL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import string
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torchaudio

from ml.audio_utils import load_audio, save_wav_bytes
from ml.config import SAMPLE_RATE
from ml.dataset import StegoDataset
from ml.inference import StegoEngine

MESSAGE_ALPHABET = string.ascii_letters + string.digits + " .,!?-_"


def compute_waveform_snr_db(reference: torch.Tensor, degraded: torch.Tensor) -> float:
    """Compute SNR in dB between 1D reference cover waveform and degraded waveform."""
    min_len = min(reference.shape[0], degraded.shape[0])
    if min_len == 0:
        return 0.0
    ref = reference[:min_len].float()
    deg = degraded[:min_len].float()
    noise = deg - ref
    sig_power = float(torch.mean(ref ** 2).item())
    noise_power = float(torch.mean(noise ** 2).item())
    return 10.0 * float(np.log10(sig_power / (noise_power + 1e-10)))


def _sample_clip_from_heldout(
    heldout_waves: Sequence[torch.Tensor],
    rng: random.Random,
    min_sec: float = 6.0,
    max_sec: float = 10.0,
) -> torch.Tensor:
    """Sample a random 6-to-10-second clip strictly from held-out validation waveforms."""
    duration_sec = rng.uniform(min_sec, max_sec)
    target_samples = int(round(duration_sec * SAMPLE_RATE))

    primary = heldout_waves[rng.randrange(len(heldout_waves))]
    if primary.shape[0] >= target_samples:
        max_start = primary.shape[0] - target_samples
        start = rng.randint(0, max_start)
        return primary[start : start + target_samples].clone()

    # Concatenate held-out segments when individual held-out files are shorter than 6-10s
    segments: List[torch.Tensor] = [primary]
    collected = primary.shape[0]
    while collected < target_samples:
        nxt = heldout_waves[rng.randrange(len(heldout_waves))]
        segments.append(nxt)
        collected += nxt.shape[0]

    combined = torch.cat(segments, dim=0)
    return combined[:target_samples].clone()


def _random_message(rng: random.Random, min_chars: int = 5, max_chars: int = 30) -> str:
    """Generate a random non-whitespace-only ASCII message of 5 to 30 characters."""
    length = rng.randint(min_chars, max_chars)
    chars = [rng.choice(MESSAGE_ALPHABET) for _ in range(length)]
    msg = "".join(chars)
    if not msg.strip():
        msg = "A" + msg[1:]
    return msg


def _mp3_roundtrip_via_ffmpeg(
    ffmpeg_bin: str,
    wav_bytes: bytes,
    tmp_dir: Path,
) -> bytes:
    """Encode 16-bit WAV bytes to 128 kbps MP3 and decode back to 16 kHz mono WAV via ffmpeg."""
    in_wav = tmp_dir / "in_stego.wav"
    mid_mp3 = tmp_dir / "mid_128k.mp3"
    out_wav = tmp_dir / "out_mp3_rt.wav"

    in_wav.write_bytes(wav_bytes)

    cmd_enc = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(in_wav),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(mid_mp3),
    ]
    res_enc = subprocess.run(cmd_enc, capture_output=True)
    if res_enc.returncode != 0:
        # Fallback without explicit libmp3lame flag if another MP3 encoder is compiled in
        cmd_enc_alt = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(in_wav),
            "-b:a",
            "128k",
            str(mid_mp3),
        ]
        subprocess.run(cmd_enc_alt, check=True, capture_output=True)

    cmd_dec = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mid_mp3),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd_dec, check=True, capture_output=True)
    return out_wav.read_bytes()


def evaluate(
    data_dirs: Sequence[str],
    checkpoint: str,
    n_samples: int = 200,
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run N-sample evaluation across all channel conditions and print report table."""
    rng = random.Random(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    # Load held-out validation split (exact last 2% of sorted file list)
    val_dataset = StegoDataset(data_dirs, split="val", auto_generate_synthetic=True)
    heldout_files = val_dataset.file_list
    if not heldout_files:
        raise RuntimeError("No held-out validation audio files available for evaluation.")

    heldout_waves: List[torch.Tensor] = []
    for fpath in heldout_files:
        w = load_audio(fpath)
        if w.numel() > 0:
            peak = float(w.abs().max().item())
            if peak > 1e-6:
                w = (w / peak) * 0.90
            heldout_waves.append(w)

    if not heldout_waves:
        raise RuntimeError("Held-out validation audio files contained no readable samples.")

    engine = StegoEngine(checkpoint_path=checkpoint, device=device)
    if engine.encoder is None or engine.decoder is None:
        engine.load()

    ffmpeg_bin = shutil.which("ffmpeg")
    has_ffmpeg = ffmpeg_bin is not None

    print(
        f"[Evaluate] Held-out files: {len(heldout_files)} | Trials (N): {n_samples} | "
        f"Checkpoint: {checkpoint} | Device: {engine.device}",
        flush=True,
    )
    if not has_ffmpeg:
        print(
            "[Evaluate] Note: ffmpeg is not installed or not in PATH; skipping MP3 128 kbps condition.",
            flush=True,
        )

    condition_names = [
        "clean",
        "noise std 0.002",
        "gain 0.7x",
        "8 kHz resample",
    ]
    if has_ffmpeg:
        condition_names.append("MP3 128 kbps")

    matches: Dict[str, int] = {c: 0 for c in condition_names}
    snrs: Dict[str, List[float]] = {c: [] for c in condition_names}
    embed_times_sec: List[float] = []

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        for _ in range(n_samples):
            cover_wave = _sample_clip_from_heldout(heldout_waves, rng, min_sec=6.0, max_sec=10.0)
            cover_wav_bytes = save_wav_bytes(cover_wave)
            cover_pcm_wave = load_audio(cover_wav_bytes)
            secret_msg = _random_message(rng, min_chars=5, max_chars=30)

            # Real pipeline: StegoEngine.embed -> 16-bit WAV bytes -> reload
            t_start = time.perf_counter()
            stego_wav_bytes, _ = engine.embed(cover_wav_bytes, secret_msg)
            embed_times_sec.append(time.perf_counter() - t_start)

            stego_reloaded = load_audio(stego_wav_bytes)
            orig_len = stego_reloaded.shape[0]

            # Build per-condition 16-bit WAV byte streams -> reload -> extract
            cond_wav_bytes: Dict[str, bytes] = {}

            # 1. Clean 16-bit WAV roundtrip
            cond_wav_bytes["clean"] = save_wav_bytes(stego_reloaded)

            # 2. Additive Gaussian noise (std 0.002)
            noise = torch.randn(stego_reloaded.shape, generator=torch_gen, dtype=torch.float32) * 0.002
            noisy_wave = (stego_reloaded + noise).clamp(-1.0, 1.0)
            cond_wav_bytes["noise std 0.002"] = save_wav_bytes(noisy_wave)

            # 3. Gain 0.7x
            gain_wave = (stego_reloaded * 0.7).clamp(-1.0, 1.0)
            cond_wav_bytes["gain 0.7x"] = save_wav_bytes(gain_wave)

            # 4. 8 kHz resampling roundtrip
            down_8k = torchaudio.functional.resample(stego_reloaded, orig_freq=SAMPLE_RATE, new_freq=8000)
            up_16k = torchaudio.functional.resample(down_8k, orig_freq=8000, new_freq=SAMPLE_RATE)
            if up_16k.shape[0] < orig_len:
                up_16k = torch.nn.functional.pad(up_16k, (0, orig_len - up_16k.shape[0]))
            elif up_16k.shape[0] > orig_len:
                up_16k = up_16k[:orig_len]
            cond_wav_bytes["8 kHz resample"] = save_wav_bytes(up_16k.clamp(-1.0, 1.0))

            # 5. MP3 128 kbps roundtrip via ffmpeg (if available)
            if has_ffmpeg and ffmpeg_bin is not None:
                cond_wav_bytes["MP3 128 kbps"] = _mp3_roundtrip_via_ffmpeg(
                    ffmpeg_bin, stego_wav_bytes, tmp_dir
                )

            # Reload each 16-bit WAV, measure SNR, and run StegoEngine.extract
            for cond in condition_names:
                c_bytes = cond_wav_bytes[cond]
                c_wave = load_audio(c_bytes)
                snrs[cond].append(compute_waveform_snr_db(cover_pcm_wave, c_wave))
                try:
                    extracted = engine.extract(c_bytes)
                    if extracted == secret_msg:
                        matches[cond] += 1
                except ValueError:
                    pass

    avg_embed_ms = (float(np.mean(embed_times_sec)) * 1000.0) if embed_times_sec else 0.0

    # Print formatted results table
    header = (
        f"{'Condition':<20} | {'Exact-Match Rate':>16} | "
        f"{'Avg SNR (dB)':>14} | {'Avg Embed Time':>16}"
    )
    sep = "-" * len(header)
    print("\n" + sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    summary_rows: Dict[str, Dict[str, Any]] = {}
    for cond in condition_names:
        rate_pct = (matches[cond] / max(1, n_samples)) * 100.0
        mean_snr = float(np.mean(snrs[cond])) if snrs[cond] else 0.0
        summary_rows[cond] = {
            "exact_match_pct": rate_pct,
            "avg_snr_db": mean_snr,
            "avg_embed_ms": avg_embed_ms,
        }
        print(
            f"{cond:<20} | {rate_pct:>15.2f}% | {mean_snr:>11.2f} dB | {avg_embed_ms:>13.1f} ms",
            flush=True,
        )

    if not has_ffmpeg:
        print(
            f"{'MP3 128 kbps':<20} | {'SKIPPED (no ffmpeg)':>16} | {'N/A':>14} | {'N/A':>16}",
            flush=True,
        )

    print(sep, flush=True)

    clean_rate = summary_rows["clean"]["exact_match_pct"]
    clean_snr = summary_rows["clean"]["avg_snr_db"]
    passed = (clean_rate >= 99.5) and (clean_snr >= 30.0)
    verdict = "PASS" if passed else "FAIL"
    print(verdict, flush=True)

    return {
        "conditions": summary_rows,
        "ffmpeg_available": has_ffmpeg,
        "passed": passed,
        "verdict": verdict,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for ml.evaluate."""
    parser = argparse.ArgumentParser(description="Evaluate Neural Audio Steganography pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        action="extend",
        default=None,
        help="Audio dataset directory or directories (default: data)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stego.pt",
        help="Path to model checkpoint (default: checkpoints/stego.pt)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of random evaluation clips (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    data_dirs = list(args.data_dir) if args.data_dir else ["data"]
    evaluate(
        data_dirs=data_dirs,
        checkpoint=args.checkpoint,
        n_samples=args.n,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

