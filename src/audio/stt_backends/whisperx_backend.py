"""
WhisperX backend (`backend="whisperx"`, the default): batched transcription
+ forced phoneme alignment + diarization. Doesn't fit
`AlignThenDiarizeASRBuffer`'s manual-chunk-loop template — WhisperX batches
the whole file in one call and has its own per-language align-model cache —
so it only inherits `DiarizingASRBuffer` for the shared diarization-pipeline/
device/`ensure_loaded` plumbing.
"""

import logging
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from datetime import datetime

from .base import DiarizingASRBuffer, merge_speaker_segments

logger = logging.getLogger(__name__)

FILLER_PROMPT = (
    "Äh, ähm, also, ja, genau, hmm, nein, okay, ne, eben, halt, eigentlich, "
    "um, uh, yeah, so, like, I mean, you know, erm, ah, right, well, actually"
)


@contextmanager
def _patched_initial_prompt(pipeline, prompt: str | None):
    """Temporarily inject initial_prompt into a FasterWhisperPipeline via its options dataclass."""
    if not prompt or not hasattr(pipeline, "options"):
        yield
        return
    original = pipeline.options
    try:
        pipeline.options = dataclass_replace(original, initial_prompt=prompt)
        yield
    finally:
        pipeline.options = original


class WhisperXBuffer(DiarizingASRBuffer):
    """Buffer for WhisperX pipeline: batched transcription + forced alignment + diarization."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None
        self.compute_type: str = "float16" if self.device == "cuda" else "int8"
        self._align_cache: dict = {}  # language_code -> (align_model, metadata)

    def load_model(self, model_name: str, timeout: int = 60):
        """Load WhisperX transcription model with automatic unloading after timeout."""
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        import whisperx

        super().load_model(timeout=timeout)

        self.model = whisperx.load_model(model_name, self.device, compute_type=self.compute_type)
        self.model_name = model_name
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def ensure_loaded(self, model_to_use: str | None = None) -> None:
        """Honors a requested Whisper size, unlike the base class's ignore-and-load-once default."""
        model_to_use = model_to_use or "turbo"
        if not self.is_loaded() or self.model_name != model_to_use:
            logger.info("Loading WhisperX model on request: %s", model_to_use)
            self.load_model(model_to_use)

    def _align_model(self, language_code: str):
        import whisperx

        if language_code not in self._align_cache:
            self._align_cache[language_code] = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
        return self._align_cache[language_code]

    def transcribe_and_diarize(
        self,
        audio_path: str,
        batch_size: int = 16,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        align: bool = True,
        initial_prompt: str | None = None,
    ) -> list[dict]:
        """Transcribe, align, and diarize in one pass. Speaker constraints are optional.

        Args:
            align: Run forced phoneme-level alignment before speaker assignment.
                Improves speaker boundary precision but can degrade quality when
                the wav2vec2 model has weak support for the detected language.
                Set to False to use segment-level timestamps instead.
            initial_prompt: Seed text injected via FasterWhisperPipeline.options.
                Used to nudge Whisper towards retaining filler words.
        """
        import whisperx

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.reset_timer()

        audio = whisperx.load_audio(audio_path)

        # 1. Batched transcription of full audio (preserves context, far less hallucination)
        with _patched_initial_prompt(self.model, initial_prompt):
            result = self.model.transcribe(audio, batch_size=batch_size)
        language = result["language"]

        # 2. Forced phoneme-level alignment for word timestamps (optional)
        if align:
            try:
                model_a, metadata = self._align_model(language)
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, self.device,
                    return_char_alignments=False,
                )
                logger.info("Alignment complete for language '%s'", language)
            except Exception as exc:
                logger.warning(
                    "Alignment failed for language '%s' (%s) — falling back to segment timestamps",
                    language, exc,
                )

        # 3. Diarize (speaker constraints optional — pyannote can auto-detect)
        diarize_kwargs: dict = {}
        if num_speakers is not None:
            diarize_kwargs["min_speakers"] = num_speakers
            diarize_kwargs["max_speakers"] = num_speakers
        elif min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = self._diarize_pipeline_instance()(audio, **diarize_kwargs)

        # 4. Assign speakers to transcript segments
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Merge consecutive segments from the same speaker into one turn
        return merge_speaker_segments(result["segments"], language)
