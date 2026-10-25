.PHONY: help setup train api web test

help:
	@echo "Available commands:"
	@echo "  make setup - Install Python (ML & Backend) and Node.js dependencies"
	@echo "  make train - Run the GAN training pipeline"
	@echo "  make api   - Start the FastAPI backend server on port 8000"
	@echo "  make web   - Start the Next.js frontend dev server on port 3000"
	@echo "  make test  - Run Python unit tests (pytest) and Next.js production build"

setup:
	python -m pip install --upgrade pip
	pip install -r ml/requirements.txt
	pip install -r backend/requirements.txt
	cd frontend && npm install

train:
	python -m ml.train --epochs 20 --batch-size 16 --data-dir data --checkpoint-dir checkpoints

api:
	python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd frontend && npm run dev -- -p 3000

test:
	pytest
	cd frontend && npm run build
