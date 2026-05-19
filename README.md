# AI APIs Collection

FastAPI microservices for AI models: Stable Diffusion, Whisper (with WhisperX diarization), Text Classification, Telegram Bot, and Nextcloud transcription sync.

## Quick Start (Docker)

```bash
cp .env.example .env
# Edit .env with your API keys and tokens
docker-compose up -d
```

Services:
- Stable Diffusion: http://localhost:1234
- Whisper: http://localhost:8080
- Text Classification: http://localhost:8000

## Kubernetes Deployment

```bash
# Build images, push to Docker Hub, and generate Helm values
./scripts/k8s_deploy.sh

# Deploy or upgrade
helm upgrade --install ai-apis helm/ai-apis -n ai-apis -f my-values.yaml
```

`scripts/k8s_deploy.sh` reads `.env`, builds all images, pushes them, and writes `my-values.yaml`. To regenerate values without rebuilding:

```bash
./scripts/k8s_deploy.sh --values-only
```

## Authentication

Two modes — configure one in `.env`:

| Mode | When to use | Config |
|---|---|---|
| Static API key | Local / simple deployments | `API_KEY=...` |
| Keycloak OIDC | Kubernetes / multi-user | `KEYCLOAK_URL=...` + `KEYCLOAK_CLIENT_*` |

With Keycloak, pass a Bearer token. For service-to-service calls (bot, Nextcloud sync) the Client Credentials flow is used automatically.

All endpoints require either `X-API-Key: <key>` or `Authorization: Bearer <token>`.

## Whisper API

### Basic transcription

```python
import requests

with open("audio.wav", "rb") as f:
    r = requests.post(
        "http://localhost:8080/transcribe/",
        headers={"X-API-Key": "your-key"},
        files={"file": f},
        params={"model_to_use": "turbo"},
    )
print(r.json()["answer"])
```

### Transcription with speaker diarization

```python
with open("meeting.mp3", "rb") as f:
    r = requests.post(
        "http://localhost:8080/transcribe_and_diarize/",
        headers={"X-API-Key": "your-key"},
        files={"file": f},
        params={
            "num_speakers": 2,        # or min_speakers + max_speakers
            "model_to_use": "turbo",
            "backend": "whisperx",    # "whisperx" (default) or "whisper" (legacy)
            "align": True,            # phoneme alignment (disable if quality degrades)
            "include_fillers": False, # retain ähm/uhm/erm fillers
            "max_words_per_second": 6.0,  # hallucination filter (0 to disable)
            "top_n_languages": 2,     # keep only N most common languages (0 to disable)
        },
    )
for seg in r.json()["answer"]:
    print(f"[{seg['START']:.1f}s] {seg['SPEAKER']}: {seg['TRANSCRIPTION']}")
```

Response segments:

```json
[
  {"SPEAKER": "SPEAKER_00", "START": 0.5, "DURATION": 4.2, "TRANSCRIPTION": "Hello, how are you?", "LANGUAGE": "en"},
  {"SPEAKER": "SPEAKER_01", "START": 5.1, "DURATION": 3.8, "TRANSCRIPTION": "I'm doing well, thanks.", "LANGUAGE": "en"}
]
```

#### Backend comparison

| | `whisperx` (default) | `whisper` (legacy) |
|---|---|---|
| Approach | Full-audio batched transcription → align → diarize | Diarize first → transcribe each segment separately |
| Hallucination risk | Low (full context) | Higher (isolated short segments) |
| Speed | Faster | Slower |
| Alignment | wav2vec2 phoneme alignment | None |

#### Filename-encoded parameters (Nextcloud sync)

Instead of passing parameters to the API, encode them in the filename — the sync job picks them up automatically:

| Filename pattern | Effect |
|---|---|
| `interview_2.mp3` | 2 speakers |
| `session_3spk.wav` | 3 speakers |
| `meeting_2speakers.m4a` | 2 speakers |
| `interview_fillers.mp3` | retain filler words |
| `meeting_fillers_2.mp3` | fillers + 2 speakers |

## Nextcloud Sync

A Kubernetes CronJob that scans a Nextcloud folder for audio/video files, transcribes them, and uploads `.txt` and `.srt` outputs to a `transcriptions/` subfolder next to each source file.

Configure in `.env`:

```bash
NEXTCLOUD_URL=https://cloud.example.com
NEXTCLOUD_USER=user@example.com
NEXTCLOUD_DAV_USER=internal_username   # if WebDAV path differs from login
NEXTCLOUD_PASSWORD=...
NEXTCLOUD_FOLDER=/Shared/transcription
NUM_SPEAKERS=2                          # or MIN_SPEAKERS + MAX_SPEAKERS
NEXTCLOUD_SCHEDULE=0 2 * * *           # daily at 02:00
```

Files that already have a transcript in `transcriptions/` are skipped. Speaker count and filler-word retention can be overridden per-file via filename encoding (see above).

## Installation (local / development)

```bash
# Install all extras
uv pip install -e ".[stable-diffusion,whisper,text-analysis,bot]"

# Whisper only (includes WhisperX)
uv pip install -e ".[whisper]"
```

## Development

```bash
uv pip install -e ".[dev]"
pytest
black src/ tests/ && isort src/ tests/
ruff check src/ tests/
```

## Project Structure

```
src/
├── core/              # Auth, config, buffer base class
├── audio/             # Whisper + WhisperX transcription & diarization
├── image_generation/  # Stable Diffusion API
├── text_analysis/     # Text classification
└── nextcloud/         # Nextcloud sync job
scripts/
├── k8s_deploy.sh      # Build, push, generate Helm values
└── build_and_push_base.sh
helm/ai-apis/          # Helm chart for all services
docker/                # Dockerfiles (*.hub = fast builds via pre-built base)
```

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA (for ML APIs)
- Docker with NVIDIA runtime (for local Docker deployment)
- Kubernetes + Helm 3 + Longhorn storage (for k8s deployment)
