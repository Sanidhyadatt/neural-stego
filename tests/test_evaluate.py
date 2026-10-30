"""Unit tests for ml/evaluate.py benchmark script."""

from pathlib import Path
import torch

from ml.config import PAYLOAD_BITS
from ml.dataset import StegoDataset, generate_synthetic_data
from ml.evaluate import evaluate, parse_args
from ml.models import Decoder, Encoder


def test_evaluate_cli_defaults():
    """Verify argparse defaults: --n 200, --checkpoint checkpoints/stego.pt."""
    args = parse_args([])
    assert args.n == 200
    assert args.checkpoint == "checkpoints/stego.pt"
    assert args.data_dir is None

    custom = parse_args(["--data-dir", "data/synthetic", "--checkpoint", "ckpt.pt", "--n", "5"])
    assert custom.data_dir == ["data/synthetic"]
    assert custom.checkpoint == "ckpt.pt"
    assert custom.n == 5


def test_last_two_percent_split_rule(tmp_path: Path):
    """Verify val split holds out the exact last 2% of the sorted file list."""
    synth_dir = tmp_path / "split_check"
    generate_synthetic_data(synth_dir, num_files=50)

    all_ds = StegoDataset(synth_dir, split="all", auto_generate_synthetic=False)
    train_ds = StegoDataset(synth_dir, split="train", auto_generate_synthetic=False)
    val_ds = StegoDataset(synth_dir, split="val", auto_generate_synthetic=False)

    assert len(all_ds.file_list) == 50
    assert val_ds.file_list == all_ds.file_list[-1:]
    assert train_ds.file_list == all_ds.file_list[:-1]


def test_evaluate_smoke_on_synthetic_data(tmp_path: Path):
    """Verify evaluate() runs end to end on synthetic data without crashing."""
    synth_dir = tmp_path / "synth_eval"
    generate_synthetic_data(synth_dir, num_files=20)

    ckpt_path = tmp_path / "smoke_stego.pt"
    torch.manual_seed(0)
    enc = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.02)
    dec = Decoder(payload_bits=PAYLOAD_BITS, base=32)
    torch.save(
        {
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "ema": {"encoder": enc.state_dict(), "decoder": dec.state_dict(), "decay": 0.999},
            "step": 1,
            "config": {"payload_bits": PAYLOAD_BITS, "base": 32, "alpha": 0.02},
        },
        ckpt_path,
    )

    res = evaluate(
        data_dirs=[str(synth_dir)],
        checkpoint=str(ckpt_path),
        n_samples=2,
        device="cpu",
        seed=123,
    )
    assert "clean" in res["conditions"]
    assert "noise std 0.002" in res["conditions"]
    assert "gain 0.7x" in res["conditions"]
    assert "8 kHz resample" in res["conditions"]
    assert res["verdict"] in ("PASS", "FAIL")

