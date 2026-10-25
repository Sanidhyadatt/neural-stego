"""Download script for LibriSpeech dev-clean dataset.

Downloads torchaudio LIBRISPEECH 'dev-clean' subset (~330 MB) into the data/ directory.
If download fails, prints instructions for manual audio file placement or synthetic generation.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import torchaudio


def download_librispeech(target_dir: Optional[Path] = None) -> bool:
    """Download the LibriSpeech 'dev-clean' subset into target_dir.

    Args:
        target_dir: Destination path (defaults to repo root / data).

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent / "data"

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LibriSpeech 'dev-clean' dataset to {target_dir}...")

    try:
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(target_dir),
            url="dev-clean",
            download=True,
        )
        print(f"Successfully downloaded and loaded LibriSpeech dev-clean ({len(dataset)} items).")
        return True
    except Exception as exc:
        print("\n" + "=" * 76, file=sys.stderr)
        print(f"[!] LibriSpeech download could not be completed: {exc}", file=sys.stderr)
        print(
            "Please put any .wav, .flac, or .mp3 files into the 'data/' directory manually.",
            file=sys.stderr,
        )
        print(
            "You can also use the built-in synthetic generator to create test files instantly:\n"
            "    python -c \"from ml.dataset import generate_synthetic_data; "
            "generate_synthetic_data('data/synthetic', num_files=20)\"",
            file=sys.stderr,
        )
        print("=" * 76 + "\n", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = download_librispeech()
    if not success:
        sys.exit(1)
