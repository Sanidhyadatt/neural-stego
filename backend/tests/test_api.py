"""Integration and API unit tests for FastAPI backend."""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from backend.main import app, create_app, resolve_checkpoint_path
from ml.config import PAYLOAD_BITS, SAMPLE_RATE
from ml.models import Decoder, Encoder


@pytest.fixture
def random_checkpoint_path(tmp_path: Path) -> Path:
    """Create a randomly-initialized model checkpoint in tmp_path."""
    torch.manual_seed(1234)
    encoder = Encoder(payload_bits=PAYLOAD_BITS, base=32, alpha=0.02)
    decoder = Decoder(payload_bits=PAYLOAD_BITS, base=32)
    ckpt_path = tmp_path / "api_random_stego.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "epoch": 0,
            "config": {"payload_bits": PAYLOAD_BITS, "base": 32, "alpha": 0.02},
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, random_checkpoint_path: Path):
    """Create a TestClient with lifespan events using a randomly-initialized checkpoint."""
    monkeypatch.setenv("STEGO_CHECKPOINT", str(random_checkpoint_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """Generate a 3.0-second synthetic WAV audio byte string (2 chunks)."""
    duration = 3.0
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False, dtype=np.float32)
    wave = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    bio = io.BytesIO()
    sf.write(bio, wave, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return bio.getvalue()


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200 and model_loaded status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_capacity_endpoint(client: TestClient, sample_wav_bytes: bytes):
    """Verify /api/capacity calculates max_chars from max_text_bytes and duration."""
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    res = client.post("/api/capacity", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["max_chars"] == 20  # 3.0s = 2 chunks -> 2 * 16 - 12 = 20 bytes
    assert data["duration_sec"] == 3.0


def test_embed_endpoint(client: TestClient, sample_wav_bytes: bytes):
    """Verify /api/embed embeds text and returns 16-bit WAV with metadata headers."""
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    data = {"message": "Hello World"}
    res = client.post("/api/embed", files=files, data=data)

    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert 'attachment; filename="stego.wav"' in res.headers["content-disposition"]
    assert "X-Snr-Db" in res.headers
    assert "X-Chunks" in res.headers
    assert "X-Duration" in res.headers

    stego_bytes = res.content
    assert len(stego_bytes) > 0
    _, sr = sf.read(io.BytesIO(stego_bytes))
    assert sr == 16000


def test_extract_endpoint_mocked_roundtrip(client: TestClient, sample_wav_bytes: bytes):
    """Verify /api/extract returns 200 and exact message when decoder recovers payload."""
    import backend.main as b_main
    from ml.text_codec import text_to_chunk_bits

    secret = "API secret"
    target_bits = text_to_chunk_bits(secret)  # [2, 128]

    class PerfectMockDecoder(torch.nn.Module):
        def forward(self, spec: torch.Tensor) -> torch.Tensor:
            b_sz = spec.shape[0]
            bits = torch.stack([target_bits[i % target_bits.shape[0]] for i in range(b_sz)], dim=0)
            return (bits * 20.0 - 10.0).to(spec.device)

    assert b_main.engine is not None
    b_main.engine.decoder = PerfectMockDecoder()

    files = {"file": ("stego.wav", sample_wav_bytes, "audio/wav")}
    res = client.post("/api/extract", files=files)
    assert res.status_code == 200
    assert res.json() == {"message": secret}


def test_extract_unrecoverable_audio_returns_422(client: TestClient):
    """Verify 422 with exact detail returned when audio has no recoverable message."""
    rng = np.random.RandomState(99)
    noise = rng.uniform(-0.1, 0.1, 32000 * 2).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, noise, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    noise_bytes = bio.getvalue()

    files = {"file": ("noise.wav", noise_bytes, "audio/wav")}
    res = client.post("/api/extract", files=files)
    assert res.status_code == 422
    assert res.json() == {
        "detail": "No readable message found. The file may not contain a hidden message, or it was altered after embedding."
    }


def test_validation_unsupported_format(client: TestClient):
    """Verify 400 returned for unsupported extensions."""
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    res = client.post("/api/capacity", files=files)
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]


def test_validation_empty_message(client: TestClient, sample_wav_bytes: bytes):
    """Verify 422 returned when message is empty."""
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    data = {"message": "   "}
    res = client.post("/api/embed", files=files, data=data)
    assert res.status_code == 422
    assert "Message must not be empty" in res.json()["detail"]


def test_validation_message_too_long_returns_audio_too_short(
    client: TestClient, sample_wav_bytes: bytes
):
    """Verify 422 with 'Audio too short' detail when message needs more blocks than chunks."""
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    # 50 bytes + 12 overhead = 62 bytes -> 4 blocks -> needs 8 seconds
    too_long = "A" * 50
    data = {"message": too_long}
    res = client.post("/api/embed", files=files, data=data)
    assert res.status_code == 422
    assert res.json() == {
        "detail": "Audio too short for this message. Needs at least 8 seconds."
    }


def test_validation_audio_exceeds_duration_limit(client: TestClient):
    """Verify 400 returned when audio exceeds 5 minutes (300 seconds)."""
    duration = 305.0
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False, dtype=np.float32)
    wave = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    bio = io.BytesIO()
    sf.write(bio, wave, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    audio_bytes = bio.getvalue()

    files = {"file": ("long.wav", audio_bytes, "audio/wav")}
    res = client.post("/api/capacity", files=files)
    assert res.status_code == 400
    assert "exceeds the 5-minute limit" in res.json()["detail"]


def test_missing_checkpoint_returns_503(monkeypatch: pytest.MonkeyPatch):
    """Verify endpoints return 503 if model checkpoint is missing."""
    import backend.main as b_main

    monkeypatch.setenv("STEGO_CHECKPOINT", "nonexistent_dir/missing.pt")

    with TestClient(b_main.app) as uninit_client:
        h_res = uninit_client.get("/health")
        assert h_res.status_code == 200
        assert h_res.json()["model_loaded"] is False

        files = {"file": ("test.wav", b"fake", "audio/wav")}
        cap_res = uninit_client.post("/api/capacity", files=files)
        assert cap_res.status_code == 503
        assert "Model checkpoint not found" in cap_res.json()["detail"]


def test_cors_default_origin(client: TestClient):
    """Verify default allowed origin http://localhost:3000 is accepted."""
    res = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_trycloudflare_regex_origin(client: TestClient):
    """Verify trycloudflare.com subdomains match allow_origin_regex."""
    origin = "https://quick-tunnel-demo.trycloudflare.com"
    res = client.get("/health", headers={"Origin": origin})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == origin


def test_cors_unauthorized_origin_rejected(client: TestClient):
    """Verify unlisted origins do not receive access-control-allow-origin header."""
    res = client.get("/health", headers={"Origin": "https://malicious-domain.com"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers


def test_cors_expose_headers(client: TestClient, sample_wav_bytes: bytes):
    """Verify X- headers are exposed via CORS."""
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    data = {"message": "Hello"}
    res = client.post(
        "/api/embed",
        files=files,
        data=data,
        headers={"Origin": "http://localhost:3000"},
    )
    assert res.status_code == 200
    expose_headers = res.headers.get("access-control-expose-headers", "")
    assert "X-Snr-Db" in expose_headers
    assert "X-Chunks" in expose_headers
    assert "X-Duration" in expose_headers


def test_cors_env_var_comma_separated_origins(
    monkeypatch: pytest.MonkeyPatch, random_checkpoint_path: Path
):
    """Verify STEGO_CORS_ORIGINS accepts comma-separated origins plus trycloudflare regex."""
    monkeypatch.setenv("STEGO_CHECKPOINT", str(random_checkpoint_path))
    monkeypatch.setenv(
        "STEGO_CORS_ORIGINS",
        "https://alpha.example.com, https://beta.example.com ",
    )
    custom_app = create_app()
    with TestClient(custom_app) as custom_client:
        res_alpha = custom_client.get("/health", headers={"Origin": "https://alpha.example.com"})
        assert res_alpha.headers.get("access-control-allow-origin") == "https://alpha.example.com"

        res_beta = custom_client.get("/health", headers={"Origin": "https://beta.example.com"})
        assert res_beta.headers.get("access-control-allow-origin") == "https://beta.example.com"

        res_cf = custom_client.get(
            "/health", headers={"Origin": "https://preview-123.trycloudflare.com"}
        )
        assert (
            res_cf.headers.get("access-control-allow-origin")
            == "https://preview-123.trycloudflare.com"
        )

        res_default = custom_client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" not in res_default.headers


def test_default_checkpoint_prefers_stego_best_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, random_checkpoint_path: Path
):
    """Verify default checkpoint resolves to checkpoints/stego_best.pt if it exists, else stego.pt."""
    import shutil
    import backend.main as b_main

    monkeypatch.delenv("STEGO_CHECKPOINT", raising=False)
    monkeypatch.chdir(tmp_path)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Neither exists -> defaults to checkpoints/stego.pt
    assert resolve_checkpoint_path() == Path("checkpoints/stego.pt")

    # 2. Only checkpoints/stego.pt exists -> loads checkpoints/stego.pt
    fallback_ckpt = ckpt_dir / "stego.pt"
    shutil.copy2(random_checkpoint_path, fallback_ckpt)
    assert resolve_checkpoint_path() == Path("checkpoints/stego.pt")
    with TestClient(b_main.app) as fallback_client:
        assert fallback_client.get("/health").json()["model_loaded"] is True

    # 3. Both exist -> prefers checkpoints/stego_best.pt
    best_ckpt = ckpt_dir / "stego_best.pt"
    shutil.copy2(random_checkpoint_path, best_ckpt)
    assert resolve_checkpoint_path() == Path("checkpoints/stego_best.pt")

    loaded_paths: list[Path] = []
    orig_engine_cls = b_main.StegoEngine

    def tracking_engine(*args, **kwargs):
        loaded_paths.append(Path(kwargs["checkpoint_path"]))
        return orig_engine_cls(*args, **kwargs)

    monkeypatch.setattr(b_main, "StegoEngine", tracking_engine)
    with TestClient(b_main.app) as best_client:
        assert best_client.get("/health").json()["model_loaded"] is True
    assert loaded_paths == [Path("checkpoints/stego_best.pt")]
