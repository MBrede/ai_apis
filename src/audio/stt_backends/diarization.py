"""
Standalone pyannote speaker-diarization pipeline (`backend="whisper"`'s
diarization half — see `diarize_audio` in whisper_api.py, which pairs this
with `WhisperBuffer` via ffmpeg-cut segments).

Distinct from the `whisperx.diarize.DiarizationPipeline` wrapper every
other backend uses via `DiarizingASRBuffer._diarize_pipeline_instance()` in
`base.py` — this is the older raw `pyannote.audio.Pipeline`, kept only for
the legacy per-segment fallback.
"""

from datetime import datetime

import torch
from pyannote.audio import Pipeline

from src.core.buffer_class import Model_Buffer
from src.core.config import config


class DiarizationBuffer(Model_Buffer):
    """Buffer for pyannote speaker diarization pipeline."""

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
