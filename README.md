# Neural Audio Steganography System

A production-grade full-stack framework for concealing secret text payloads imperceptibly inside audio signals using a PyTorch GAN architecture (Encoder, Decoder, and Discriminator) operating in the complex STFT domain, served through a high-performance FastAPI backend, and visualized with a Next.js 14+ TypeScript dashboard.

---

## Architecture Diagram

```
[ Secret Text Message ]
          │
          ▼
┌────────────────────────────────────────┐
│  Text Codec & Framing (ml/text_codec)  │
│  - UTF-8 serialization                 │
│  - 2-byte Big-Endian Length Header     │
│  - Reed-Solomon FEC (8 Parity Bytes)   │
│  - Pad to 16-Byte Block (128 bits/2s)  │
└──────────────────┬─────────────────────┘
                   │
                   ▼ Binary Message Bits [B, 128]
                   │
[ Cover Audio (16 kHz Mono) ]
          │
          ▼
┌────────────────────────────────────────┐
│     STFT Transform (ml/audio_utils)    │
│  - n_fft=1024, hop=256, win=1024       │
│  - Complex Spectrogram [B, 2, 513, 126]│
└──────────────────┬─────────────────────┘
                   │
                   ├───────────────────────────────┐
                   │ Cover Spec [B, 2, 513, 126]   │
                   ▼                               │
┌────────────────────────────────────────┐         │
│      U-Net Encoder (ml/models)         │         │
│  - Linear Projection of 128 Bits       │         │
│  - Residual Convolutional Generator    │         │
│  - Residual Mask Addition (alpha=0.02) │         │
└──────────────────┬─────────────────────┘         │
                   │                               │
                   ▼ Stego Spec [B, 2, 513, 126]   │
                   │                               │
                   ├───────────────┐               │
                   │               ▼               ▼
                   │   ┌────────────────────────────────────────┐
                   │   │   Discriminator Network (ml/models)    │
                   │   │   - Conv2d Spectral Patch Classifier   │
                   │   │   - Evaluates Cover vs Stego Realism   │
                   │   │   - Minimax GAN Adversarial Loss       │
                   │   └────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│      iSTFT Inverse Synthesis           │
│  - Chunk Reassembly & Overlap-Add      │
│  - 16-bit PCM WAV Encoding (PCM_16)    │
└──────────────────┬─────────────────────┘
                   │
                   ▼ [ Stego Audio WAV ]
                   │
                   ▼
┌────────────────────────────────────────┐
│      Neural Decoder (ml/models)        │
│  - STFT of Stego Audio Chunks          │
│  - Conv2d Deep Feature Extractor       │
│  - 128-bit Sigmoid Logits [B, 128]     │
│  - 0.5 Thresholding + RS FEC Recovery  │
└──────────────────┬─────────────────────┘
                   │
                   ▼
       [ Recovered Secret Text ]
```

---

## Monorepo Layout

```
neural-stego/
├── ml/           # PyTorch GAN models, audio processing, datasets & inference
├── backend/      # FastAPI REST service (/health, /api/capacity, /api/embed, /api/extract)
├── frontend/     # Next.js 14+ dashboard, TypeScript, Tailwind CSS, Web Audio waveform
├── checkpoints/  # PyTorch model weights (stego.pt)
├── data/         # Raw and processed audio datasets (LibriSpeech / synthetic)
├── test_files/   # Sample audio and test files for end-to-end verification
├── Makefile      # Build, train, test, and serve automation targets
└── pyproject.toml# Python workspace configuration
```

---

## System Specifications

| Parameter | Specification |
| :--- | :--- |
| **Audio Format** | Mono, 16,000 Hz, float32 normalized in `[-1.0, 1.0]` |
| **Chunk Length** | 2.0 seconds (32,000 samples) per chunk with zero-padding & auto-trimming |
| **STFT Dimensions** | `n_fft=1024`, `hop_length=256`, `win_length=1024`, Hann, `center=True` |
| **Spectrogram Tensor** | 2 channels `(real, imag)`, shape `[B, 2, 513, 126]` |
| **Payload Capacity** | 128 bits (16 bytes) per 2-second audio chunk |
| **Protocol Overhead**| 2 bytes length header + 8 bytes Reed-Solomon parity = 10 bytes |
| **Usable Payload**   | `(n_chunks * 16) - 10` bytes (e.g. 6 bytes for 2s, 22 bytes for 4s) |
| **Input Formats**    | WAV, MP3, FLAC, OGG (converted automatically to 16 kHz mono) |
| **Output Format**   | 16-bit PCM WAV (`PCM_16`) |

---

## Setup Guide

### Prerequisites

- **Python 3.10+** (Python 3.11 recommended)
- **Node.js 18.17+** & `npm`
- **Git**

### 1. Python Environment Setup

Create and activate a virtual environment at the repository root:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# Activate on Linux / macOS:
source .venv/bin/activate
```

### 2. Install ML & Backend Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install ML dependencies (PyTorch, torchaudio, reedsolo, soundfile, librosa)
pip install -r ml/requirements.txt

# Install FastAPI backend dependencies (FastAPI, uvicorn, pydantic)
pip install -r backend/requirements.txt
```

Verify your computed STFT constants:
```bash
python -c "from ml.config import FREQ_BINS, N_FRAMES; print(f'STFT dimensions: {FREQ_BINS} x {N_FRAMES}')"
# Expected: STFT dimensions: 513 x 126
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Data Download & Preparation

The training pipeline uses clean speech audio signals (e.g., LibriSpeech).

### Option A: Automated LibriSpeech Download (~330 MB)
Download the standard LibriSpeech `dev-clean` dataset into the `data/` directory:

```bash
python -m ml.download_data
```

### Option B: Built-in Synthetic Audio Generation
If internet access is restricted or you need instant test audio:

```bash
python -c "from ml.dataset import generate_synthetic_data; generate_synthetic_data('data/synthetic', num_files=20)"
```

### Option C: Custom Audio Placement
Place any `.wav`, `.flac`, or `.mp3` files directly into `data/`. The dataset loader will automatically discover and convert them to 16 kHz mono on the fly.

---

## Model Training

The GAN framework trains the Encoder (Generator), Decoder, and Discriminator using multi-resolution STFT loss, L1 time-domain loss, message binary cross-entropy, and adversarial minimax loss.

### Full Training Run
```bash
python -m ml.train --epochs 20 --batch-size 16 --data-dir data --checkpoint-dir checkpoints --device auto
```

### Quick Smoke Test / Verification Run
To verify the training loop with a small number of optimizer steps:
```bash
python -m ml.train --epochs 1 --max-steps 30 --batch-size 4 --data-dir data --checkpoint-dir checkpoints --device cpu
```

### Training CLI Arguments

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `16` | Audio chunk batch size |
| `--data-dir` | `data` | Directory containing audio files |
| `--checkpoint-dir` | `checkpoints`| Directory where `stego.pt` is saved |
| `--device` | `auto` | Target device (`auto`, `cuda`, `mps`, `cpu`) |
| `--max-steps` | `None` | Max optimizer steps before early termination |
| `--resume` | `None` | Path to existing checkpoint to resume training |
| `--lr` | `2e-4` | Optimizer learning rate |

---

## Command-Line Inference

You can embed and extract secret text directly from the command line:

```bash
# 1. Embed secret message into audio
python -m ml.inference embed in.wav "Confidential Payload" out.wav --checkpoint checkpoints/stego.pt

# 2. Extract concealed message from stego audio
python -m ml.inference extract out.wav --checkpoint checkpoints/stego.pt

# 3. Check character capacity of an audio file
python -m ml.inference capacity in.wav
```

---

## Running the API & Web Dashboard

### 1. Start the FastAPI Backend (Port 8000)

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health at:
- `http://localhost:8000/health`
- Interactive Swagger docs: `http://localhost:8000/docs`

### 2. Start the Next.js Dashboard (Port 3000)

```bash
cd frontend
npm run dev -- -p 3000
```

Open your browser at `http://localhost:3000` to access the steganography studio.

---

## Makefile Automation

A convenient `Makefile` is included at the repository root:

```bash
make setup    # Install all Python and Node.js dependencies
make train    # Run GAN model training pipeline
make api      # Start the FastAPI backend on port 8000
make web      # Start the Next.js frontend dev server on port 3000
make test     # Run all unit tests (pytest) and verify Next.js build
```

---

## Limitations

Please note the following technical and operational constraints:

1. **Output Must Remain Uncompressed WAV**:
   - The neural steganography system operates by introducing imperceptible, micro-level residual perturbations in the STFT spectral coefficients.
   - **Lossy re-encoding (such as MP3, AAC, or OGG compression) discards high-frequency and low-energy spectral components and will destroy the embedded secret payload.** The stego audio file must remain in lossless PCM WAV format (`.wav`).
2. **Steganalysis Resilience**:
   - The embedding network is trained against a spectral convolutional Discriminator to produce realistic, artifact-free audio with Signal-to-Noise Ratio (SNR) exceeding 70-85 dB.
   - While the perturbations are mathematically imperceptible to human hearing and fool the co-trained discriminator, **resistance against specialized forensic steganalysis tools or advanced statistical audio steganalysis cannot be guaranteed.**
3. **Payload Capacity Constraints**:
   - The physical capacity is **128 bits (16 bytes)** per 2.0-second audio chunk.
   - To guarantee transmission integrity, framing overhead uses **2 bytes** for the payload length header and **8 bytes** for Reed-Solomon forward error correction parity.
   - Consequently, the net usable text capacity is:
     $$\text{Capacity (bytes)} = (N_{\text{chunks}} \times 16) - 10$$
     - 1 chunk (2.0s): **6 bytes** (6 characters)
     - 2 chunks (4.0s): **22 bytes** (22 characters)
     - 3 chunks (6.0s): **38 bytes** (38 characters)
     - Audio shorter than 2 seconds cannot store a message with full Reed-Solomon protection.
