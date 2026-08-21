"""
Legacy per-segment backend: plain Whisper transcription + a standalone
pyannote diarization pipeline, combined by the caller via ffmpeg-cut
segments (see `diarize_audio` in whisper_api.py). Predates WhisperX/the
`transcribe_and_diarize`-per-buffer backends below; kept as the
`backend="whisper"` fallback.
"""

from datetime import datetime

import torch
import whisper
from pyannote.audio import Pipeline

from src.core.buffer_class import Model_Buffer
from src.core.config import config


class WhisperBuffer(Model_Buffer):
    """Buffer for Whisper transcription model with automatic unloading."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None

    def load_model(self, model_name: str, timeout: int = 300, **kwargs):
        """Load Whisper model with automatic unloading after timeout."""
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        super().load_model(timeout=timeout)

        self.model = whisper.load_model(model_name, **kwargs)
        self.model_name = model_name
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        """Transcribe audio file."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.reset_timer()
        return self.model.transcribe(audio_path, **kwargs)


class DiarizationBuffer(Model_Buffer):
    """Buffer for pyannote speaker diarization pipeline."""

    def __init__(self):
        super().__init__()

    def load_model(self, timeout: int = 300, **kwargs):
        """Load diarization pipeline with automatic unloading after timeout."""
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        super().load_model(timeout=timeout)

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=config.HF_TOKEN
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def diarize(
        self,
        audio_path: str,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
    ):
        """Perform speaker diarization on audio file."""
        if not self.is_loaded():
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")

        self.reset_timer()

        if num_speakers is not None:
            return self.pipeline(audio_path, num_speakers=num_speakers)
        elif min_speakers is not None:
            return self.pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        else:
            raise ValueError(
                "Either num_speakers or min_speakers and max_speakers must be provided."
            )
