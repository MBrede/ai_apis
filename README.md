# AI APIs Collection

FastAPI microservices for AI models: Stable Diffusion, Whisper, Text Classification, and Telegram Bot.

## Quick Start (Docker - Recommended)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys and tokens

# 2. Start all services
docker-compose up -d

# 3. Initialize MongoDB (optional, if USE_MONGODB=true)
docker-compose exec mongodb python scripts/init_mongodb.py
```

Services will be available at:
- Stable Diffusion: http://localhost:1234
- Whisper: http://localhost:8080
- Text Classification: http://localhost:8000
- MongoDB: localhost:27017

## Installation Options

### Docker (Recommended)
```bash
docker-compose up -d
```

### Manual Installation
```bash
pip install -e ".[stable-diffusion,whisper,text-analysis,bot]"
```

Install only what you need:
- Bot only: `pip install -e ".[bot]"`
- SD only: `pip install -e ".[stable-diffusion]"`
- Whisper only: `pip install -e ".[whisper]"`
- Text analysis only: `pip install -e ".[text-analysis]"`

## Configuration

Required environment variables in `.env`:

```bash
# Authentication (supports comma-separated multiple keys)
API_KEY=your-key-here
ADMIN_API_KEY=your-admin-key

# External Services
HF_TOKEN=your-huggingface-token
TELEGRAM_TOKEN=your-telegram-token  # For bot

# MongoDB (optional)
USE_MONGODB=true
MONGO_ROOT_USER=admin
MONGO_ROOT_PASSWORD=changeme

# API Endpoints (for Docker, use service names)
OLLAMA_HOST=localhost
SD_HOST=localhost
WHISPER_HOST=localhost
```

## API Usage

All APIs require `X-API-Key` header:

```python
import requests

headers = {"X-API-Key": "your-api-key-here"}

# Stable Diffusion
response = requests.post(
    "http://localhost:1234/post_config",
    headers=headers,
    params={"prompt": "A beautiful landscape", "model_id": "stabilityai/stable-diffusion-2-1"}
)

# Whisper
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8080/transcribe",
        headers=headers,
        files={"file": f}
    )

# Text Classification
response = requests.post(
    "http://localhost:8000/predict_proba/",
    headers=headers,
    json={"texts": ["This is great!"], "model_name": "cardiffnlp/twitter-roberta-base-sentiment"}
)
```

## Features

- **GPU Memory Management**: Automatic model unloading after configurable timeout
- **Multiple API Keys**: Support comma-separated keys in environment variables or MongoDB
- **MongoDB Integration**: Optional persistent storage for API keys and bot settings
- **Docker Support**: Full GPU support with NVIDIA runtime
- **Modular Dependencies**: Install only what you need
- **Comprehensive Testing**: Unit tests for all core modules and APIs
- **CI/CD Pipeline**: Automated linting, testing, and Docker builds

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 13.0+ (for ML APIs)
- Docker with NVIDIA runtime (for Docker deployment)

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/core/test_config.py -v

# Run tests in parallel (faster)
pytest -n auto
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### CI/CD

The project uses GitHub Actions for continuous integration:

- **Lint & Type Check** (`lint.yml`): Runs on every push and PR
  - Black formatting check
  - isort import sorting
  - Ruff linting
  - MyPy type checking

- **Tests** (`test.yml`): Runs on every push and PR
  - Unit tests with pytest
  - Coverage reporting
  - Import validation

- **Docker Build** (`docker.yml`): Builds and scans Docker images
  - Multi-service builds
  - Security scanning with Trivy
  - Automatic push to GHCR on main branch

All workflows run on Python 3.12 and include caching for faster builds.

## Documentation

- DOCKER.md - Complete Docker deployment guide
- CRITIQUE.md - Code quality analysis
- See CLAUDE.md for coding standards

## Project Structure

```
src/
├── core/              # Shared utilities (auth, config, buffer, bot)
├── image_generation/  # Stable Diffusion API
├── audio/             # Whisper transcription + diarization
├── text_analysis/     # Text classification
└── examples/          # Usage examples
```
