"""
Plain Whisper transcription buffer (`backend="whisper"`'s ASR half — see
`diarize_audio` in whisper_api.py, which pairs this with `DiarizationBuffer`
via ffmpeg-cut segments). No diarization of its own.
"""

from datetime import datetime

from src.core.buffer_class import Model_Buffer


class WhisperBuffer(Model_Buffer):
    """Buffer for Whisper transcription model with automatic unloading."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None

    def load_model(self, model_name: str, timeout: int = 60, **kwargs):
        """Load Whisper model with automatic unloading after timeout."""
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        import whisper

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
