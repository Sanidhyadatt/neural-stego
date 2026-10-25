"""Integration and API unit tests for FastAPI backend."""

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import MAX_UPLOAD_SIZE, app
from ml.config import SAMPLE_RATE


@pytest.fixture
def client():
    """Create a TestClient with lifespan events executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """Generate a 3.0-second valid WAV audio byte string."""
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
    assert isinstance(data["model_loaded"], bool)
    assert data["model_loaded"] is True


def test_capacity_endpoint(client: TestClient, sample_wav_bytes: bytes):
    """Verify /api/capacity calculates max characters and duration."""
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    res = client.post("/api/capacity", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "max_chars" in data
    assert "duration_sec" in data
    assert data["max_chars"] == 22  # 3.0s = 2 chunks -> 22 bytes capacity
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

    # Validate returned WAV data
    stego_bytes = res.content
    assert len(stego_bytes) > 0
    wave, sr = sf.read(io.BytesIO(stego_bytes))
    assert sr == 16000


def test_extract_endpoint(client: TestClient, sample_wav_bytes: bytes):
    """Verify /api/extract processes audio without server crash."""
    # First embed a message
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    data = {"message": "Secret123"}
    embed_res = client.post("/api/embed", files=files, data=data)
    assert embed_res.status_code == 200

    # Next call extract
    extract_files = {"file": ("stego.wav", embed_res.content, "audio/wav")}
    res = client.post("/api/extract", files=extract_files)
    # Response is either 200 with extracted message or 422 unrecoverable
    assert res.status_code in (200, 422)
    if res.status_code == 200:
        assert "message" in res.json()
    else:
        assert "No readable message found" in res.json()["detail"]


def test_extract_unrecoverable_audio_returns_422(client: TestClient):
    """Verify 422 returned when audio does not contain readable steganography."""
    # Pure noise audio
    noise = np.random.uniform(-0.1, 0.1, 32000 * 2).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, noise, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    noise_bytes = bio.getvalue()

    files = {"file": ("noise.wav", noise_bytes, "audio/wav")}
    res = client.post("/api/extract", files=files)
    assert res.status_code == 422
    assert res.json()["detail"] == (
        "No readable message found. The file may not contain a hidden message, or it was altered after embedding."
    )


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


def test_validation_message_too_long(client: TestClient, sample_wav_bytes: bytes):
    """Verify 422 returned when message exceeds audio capacity."""
    files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    # Capacity is 22 chars for 3.0s audio
    too_long = "A" * 50
    data = {"message": too_long}
    res = client.post("/api/embed", files=files, data=data)
    assert res.status_code == 422
    assert "exceeds maximum capacity" in res.json()["detail"]


def test_validation_audio_exceeds_duration_limit(client: TestClient):
    """Verify 400 returned when audio exceeds 5 minutes (300 seconds)."""
    # 305 seconds of audio
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


def test_missing_checkpoint_returns_503(monkeypatch):
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

