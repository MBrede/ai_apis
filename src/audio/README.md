# Whisper Audio API

Audio transcription with speaker diarization.

## Endpoints

- `POST /transcribe` - Transcribe audio
- `POST /transcribe_and_diarize` - Transcribe with speaker labels
- `GET /buffer_status` - Check model status

## Example

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8080/transcribe",
        headers={"X-API-Key": "your-key"},
        files={"file": f},
        params={"model": "turbo"}
    )
```

## Features

- Automatic GPU memory management (5-minute timeout)
- Speaker diarization with pyannote.audio
- Multiple Whisper model sizes
- Supports WAV, MP3, M4A formats
