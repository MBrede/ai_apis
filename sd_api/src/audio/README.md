# Audio Processing APIs

Speech-to-text transcription with speaker diarization using OpenAI Whisper and Pyannote.

## Files

### `whisper_api.py`
**Port:** 8080
**Models:** Whisper (tiny, base, small, medium, large, turbo)
**GPU:** cuda (with CPU fallback)

High-quality audio transcription with optional speaker identification.

#### Features:
- Automatic speech recognition (ASR)
- Multi-language support (98+ languages)
- Speaker diarization (who spoke when)
- Multiple Whisper model sizes
- Audio format conversion via FFmpeg
- Long-form audio support

#### Endpoints:
- `POST /transcribe/` - Transcribe audio file
  - Form data: `file` (audio file), `model_to_use` (whisper model)
  - Returns: `{"answer": "transcription text"}`

- `POST /transcribe_and_diarize/` - Transcribe with speaker labels
  - Form data: `file`, `model_to_use`, `num_speakers` OR `min_speakers` + `max_speakers`
  - Returns: `{"answer": [{speaker, start, duration, transcription, language}]}`

#### Start Command:
```bash
gunicorn whisper_api:app -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 -t 30000
```

#### Environment Variables:
- `hf_token` - HuggingFace token (for Pyannote speaker diarization model)

#### Model Selection:

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| tiny | Very Fast | Low | 1GB | Real-time, low quality OK |
| base | Fast | Medium | 1GB | Quick transcription |
| small | Medium | Good | 2GB | Balanced |
| medium | Slow | Better | 5GB | High quality |
| large | Very Slow | Best | 10GB | Maximum accuracy |
| turbo | Fast | Good | 6GB | **Recommended** - best speed/quality |

#### Usage Examples:

**Simple Transcription:**
```python
import requests

# Transcribe an audio file
files = {'file': ('recording.mp3', open('recording.mp3', 'rb'), 'audio/mpeg')}
response = requests.post(
    "http://localhost:8080/transcribe",
    params={"model_to_use": "turbo"},
    files=files
)

result = response.json()
print(f"Transcription: {result['answer']}")
```

**With Speaker Diarization:**
```python
# Known number of speakers
files = {'file': ('meeting.wav', open('meeting.wav', 'rb'), 'audio/wav')}
response = requests.post(
    "http://localhost:8080/transcribe_and_diarize",
    params={
        "model_to_use": "turbo",
        "num_speakers": 3  # Exactly 3 speakers
    },
    files=files
)

segments = response.json()['answer']
for seg in segments:
    print(f"[{seg['START']}s - {seg['DURATION']}s] {seg['SPEAKER']}: {seg['TRANSCRIPTION']}")
    print(f"Language: {seg['LANGUAGE']}\n")
```

**Unknown Number of Speakers:**
```python
# Let the model detect speakers
files = {'file': ('podcast.mp3', open('podcast.mp3', 'rb'), 'audio/mpeg')}
response = requests.post(
    "http://localhost:8080/transcribe_and_diarize",
    params={
        "model_to_use": "turbo",
        "min_speakers": 2,
        "max_speakers": 5
    },
    files=files
)
```

#### Python Client Example:
```python
def transcribe_audio(audio_path: str, model: str = "turbo") -> str:
    """Transcribe audio file using Whisper API."""
    with open(audio_path, 'rb') as f:
        files = {'file': (audio_path, f, 'audio/mpeg')}
        response = requests.post(
            "http://localhost:8080/transcribe",
            params={"model_to_use": model},
            files=files
        )
        response.raise_for_status()
        return response.json()['answer']

def transcribe_with_speakers(
    audio_path: str,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
    model: str = "turbo"
) -> list[dict]:
    """Transcribe with speaker identification."""
    with open(audio_path, 'rb') as f:
        files = {'file': (audio_path, f, 'audio/mpeg')}
        params = {"model_to_use": model}

        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        response = requests.post(
            "http://localhost:8080/transcribe_and_diarize",
            params=params,
            files=files
        )
        response.raise_for_status()
        return response.json()['answer']
```

## Supported Audio Formats

Via FFmpeg, supports:
- MP3
- WAV
- M4A
- FLAC
- OGG
- OPUS
- WebM
- And many more...

Audio is automatically converted to mono WAV for processing.

## Language Support

Whisper supports 98+ languages, including:
- English, Spanish, French, German, Italian, Portuguese
- Chinese, Japanese, Korean
- Arabic, Hebrew, Russian
- Hindi, Bengali, Urdu
- And many more...

Language is automatically detected during transcription.

## Performance

**Transcription Speed (on RTX 4090):**
- tiny: ~100x realtime
- base: ~50x realtime
- small: ~25x realtime
- medium: ~10x realtime
- large: ~5x realtime
- **turbo: ~20x realtime** ⭐ Recommended

*Note: "10x realtime" means a 10-minute audio file takes 1 minute to transcribe.*

**Diarization adds ~2-3x overhead** due to:
1. Audio segmentation
2. Speaker embedding extraction
3. Clustering
4. Per-segment transcription

## Hardware Requirements

**Minimum:**
- GPU: 4GB VRAM (for turbo/small models)
- RAM: 8GB
- CPU: 4 cores (for FFmpeg conversion)

**Recommended:**
- GPU: 8GB+ VRAM
- RAM: 16GB
- CPU: 8+ cores
- SSD storage (for fast audio I/O)

**CPU-only mode:**
- Works but ~10x slower
- Good for lightweight deployments
- Use `tiny` or `base` models

## Configuration

### Change Default Model:
```python
# In whisper_api.py, line 18:
loaded_model = "turbo"  # Change this
model = whisper.load_model(loaded_model)
```

### Disable Speaker Diarization:
Remove or comment out lines 25-26 (Pyannote pipeline loading) if you don't need diarization. This will reduce memory usage.

## Known Issues

1. **Temporary files** not always cleaned up properly
2. **Long audio files** may timeout (increase gunicorn timeout: `-t 60000`)
3. **FFmpeg dependency** required but not checked on startup
4. **No streaming support** - full file must be uploaded
5. **Memory usage** grows with audio length
6. **No audio validation** - malformed files can crash

## Optimization Tips

1. **Pre-process audio:**
   ```bash
   # Convert to optimal format
   ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
   ```

2. **Use appropriate model:**
   - Podcasts/clear audio: `turbo` or `small`
   - Noisy audio: `medium` or `large`
   - Real-time: `tiny` or `base`

3. **Chunk long audio:**
   Split files longer than 1 hour to avoid timeout

4. **Enable batching:**
   Process multiple files in parallel (requires code changes)

## Troubleshooting

**Error: "ffmpeg not found"**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Out of memory with diarization:**
- Use smaller audio chunks
- Reduce `max_speakers` parameter
- Disable diarization if not needed
- Use CPU for Pyannote: change `.to(torch.device('cuda'))` to `.to('cpu')`

**Poor transcription quality:**
- Use larger Whisper model
- Pre-process audio (denoise, normalize)
- Ensure audio is clear and loud enough
- Check language detection is correct

**Timeout errors:**
- Increase gunicorn timeout: `-t 60000`
- Use smaller Whisper model
- Split long audio files

## Future Improvements

- [ ] Add streaming transcription (real-time)
- [ ] Implement audio preprocessing (denoising, normalization)
- [ ] Add subtitle generation (SRT, VTT formats)
- [ ] Support for direct URL input
- [ ] Implement result caching
- [ ] Add word-level timestamps
- [ ] Support for custom vocabularies
- [ ] Add translation endpoint (transcribe + translate to English)
- [ ] Batch processing endpoint
- [ ] WebSocket support for real-time transcription
- [ ] Add confidence scores in output
- [ ] Support for punctuation and capitalization control
- [ ] Add profanity filtering option

## Dependencies

- **openai-whisper** >= 20250625 - Speech recognition
- **pyannote-audio** >= 4.0.1 - Speaker diarization
- **ffmpeg-python** >= 0.2.0 - Audio format conversion
- **torch** >= 2.6.0 - Deep learning backend

## License

- **Whisper**: MIT License
- **Pyannote**: MIT License
- Requires HuggingFace token for Pyannote models (requires accepting model terms)

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [Whisper Model Card](https://github.com/openai/whisper/blob/main/model-card.md)
- [Pyannote Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization)
