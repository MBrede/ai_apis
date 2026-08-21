"""
nvidia/nemotron-3.5-asr-streaming-0.6b backend (`backend="nemotron-asr"`).
https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b

Uses `transformers.pipeline("automatic-speech-recognition", ...)` — the
model card documents this as a supported loading path (alongside NeMo, not
used here to avoid a second heavy framework dependency) and it's the
simplest correct integration: no custom chat-template/generate() plumbing
needed, unlike ArkASRBuffer/GraniteSpeechBuffer.

No word-level timestamps documented for this model — `DiarizeFirstASRBuffer`,
same as ArkASRBuffer/HojoASRBuffer. Cache-aware streaming architecture
(FastConformer-RNNT); the pipeline API here always runs it in plain batch
mode per call, streaming state is not used — fine for diarize-turn-sized
chunks.
"""

from datetime import datetime

from .base import DiarizeFirstASRBuffer


class NemotronASRBuffer(DiarizeFirstASRBuffer):
    MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    DEFAULT_LANGUAGE = "English"

    def load_model(self, timeout: int = 300):
        """Load Nemotron-3.5-ASR via the transformers ASR pipeline, with automatic unloading."""
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        from transformers import pipeline

        super().load_model(timeout=timeout)

        def _load():
            self.model = pipeline(
                "automatic-speech-recognition",
                model=self.MODEL_ID,
                device=self.device,
            )

        self._load_or_cleanup(_load)
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def _transcribe_chunk(self, chunk) -> str:
        """Run one pipeline() call on a diarize-turn audio slice (numpy float32, 16kHz)."""
        result = self.model({"array": chunk, "sampling_rate": 16000})
        return result["text"]
