"""FastAPI backend service for Neural Audio Steganography.

Provides REST API endpoints for:
- Health check and model readiness (/health)
- Capacity estimation (/api/capacity)
- Audio steganography embedding (/api/embed)
- Secret message extraction (/api/extract)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure repository root is on sys.path for clean, robust imports across environments
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.audio_utils import load_audio
from ml.config import SAMPLE_RATE, SUPPORTED_INPUT_EXTENSIONS
from ml.inference import StegoEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-stego-api")

# Constraints
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB limit
MAX_AUDIO_DURATION_SEC = 300.0  # 5 minutes

# Global StegoEngine instance managed via lifespan
engine: Optional[StegoEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle, preloading the StegoEngine."""
    global engine
    ckpt_env = os.environ.get("STEGO_CHECKPOINT", "checkpoints/stego.pt")
    ckpt_path = Path(ckpt_env)

    if ckpt_path.is_file():
        try:
            logger.info(f"Loading StegoEngine checkpoint from: {ckpt_path}")
            engine = StegoEngine(checkpoint_path=ckpt_path)
            logger.info("StegoEngine successfully loaded and ready for inference.")
        except Exception as exc:
            logger.error(f"Failed to load checkpoint at {ckpt_path}: {exc}")
            engine = None
    else:
        logger.warning(
            f"Checkpoint file not found at '{ckpt_path}'. API starting in uninitialized mode."
        )
        engine = None

    yield
    # Cleanup on shutdown if required
    engine = None


app = FastAPI(
    title="Neural Audio Steganography API",
    description="High-performance GAN-based audio steganography service",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
cors_origins_env = os.environ.get("STEGO_CORS_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.trycloudflare\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Snr-Db", "X-Chunks", "X-Duration"],
)


# Response Models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class CapacityResponse(BaseModel):
    max_chars: int
    duration_sec: float


class ExtractResponse(BaseModel):
    message: str


def _validate_upload(file: UploadFile, content: bytes) -> None:
    """Validate upload payload size, file extension, and basic integrity."""
    # 1. Size validation
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the 25 MB limit (received {len(content) / (1024 * 1024):.2f} MB).",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 2. File extension validation
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Supported formats: {', '.join(SUPPORTED_INPUT_EXTENSIONS)}."
            ),
        )


def _check_engine_available() -> StegoEngine:
    """Ensure StegoEngine is loaded; otherwise raise 503."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoint not found. Train a model first.",
        )
    return engine


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API operational status and whether model weights are loaded."""
    is_loaded = engine is not None and engine.encoder is not None
    return HealthResponse(status="ok", model_loaded=is_loaded)


@app.post("/api/capacity", response_model=CapacityResponse)
async def check_capacity(file: UploadFile = File(...)) -> CapacityResponse:
    """Compute maximum character capacity and duration for an audio file."""
    active_engine = _check_engine_available()
    content = await file.read()
    _validate_upload(file, content)

    try:
        wave = await asyncio.to_thread(load_audio, content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse audio data: {exc}",
        )

    duration_sec = wave.shape[0] / SAMPLE_RATE
    if duration_sec > MAX_AUDIO_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration_sec:.1f}s) exceeds the 5-minute limit (300s).",
        )

    max_chars = await asyncio.to_thread(active_engine.capacity, content)
    return CapacityResponse(
        max_chars=max_chars,
        duration_sec=round(duration_sec, 2),
    )


@app.post("/api/embed")
async def embed_message(
    file: UploadFile = File(...),
    message: str = Form(...),
) -> Response:
    """Embed secret text into cover audio and stream back 16-bit PCM WAV."""
    active_engine = _check_engine_available()
    content = await file.read()
    _validate_upload(file, content)

    # Validate message length
    if len(message.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message must not be empty.",
        )

    try:
        wave = await asyncio.to_thread(load_audio, content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse audio data: {exc}",
        )

    duration_sec = wave.shape[0] / SAMPLE_RATE
    if duration_sec > MAX_AUDIO_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration_sec:.1f}s) exceeds the 5-minute limit (300s).",
        )

    max_chars = await asyncio.to_thread(active_engine.capacity, content)
    # Validate against capacity (character count / UTF-8 byte length)
    if len(message.encode("utf-8")) > max_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Message length ({len(message)} chars) exceeds maximum capacity "
                f"of {max_chars} characters for this audio file."
            ),
        )

    try:
        # Run inference in threadpool to keep FastAPI async event loop unblocked
        wav_bytes, info = await asyncio.to_thread(active_engine.embed, content, message)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed message into audio: {exc}",
        )

    headers = {
        "Content-Disposition": 'attachment; filename="stego.wav"',
        "X-Snr-Db": str(info["snr_db"]),
        "X-Chunks": str(info["n_chunks"]),
        "X-Duration": str(info["duration_sec"]),
    }
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers=headers,
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_message(file: UploadFile = File(...)) -> ExtractResponse:
    """Extract concealed secret text from stego audio file."""
    active_engine = _check_engine_available()
    content = await file.read()
    _validate_upload(file, content)

    try:
        wave = await asyncio.to_thread(load_audio, content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse audio data: {exc}",
        )

    duration_sec = wave.shape[0] / SAMPLE_RATE
    if duration_sec > MAX_AUDIO_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration_sec:.1f}s) exceeds the 5-minute limit (300s).",
        )

    try:
        # Run extraction in threadpool
        extracted_text = await asyncio.to_thread(active_engine.extract, content)
        return ExtractResponse(message=extracted_text)
    except ValueError:
        # Reed-Solomon unrecoverable error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable message found. The file may not contain a hidden message, or it was altered after embedding.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract message: {exc}",
        )
