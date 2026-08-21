"""
Qwen3-ASR-1.7B backend (`backend="qwen3-asr"`).

Uses the `qwen-asr` PyPI package's `Qwen3ASRModel`, NOT raw
`transformers.AutoModel*` — Qwen3-ASR ships its own inference package
(`pip install qwen-asr`), confirmed against the model card at
https://huggingface.co/Qwen/Qwen3-ASR-1.7B (2026-08-19). The forced aligner
(`Qwen/Qwen3-ForcedAligner-0.6B`) is passed in at
`Qwen3ASRModel.from_pretrained(..., forced_aligner=...)` and handled
internally by `model.transcribe(..., return_time_stamps=True)` — no
separate aligner model/call to wire up by hand.

Chunked into <=5min windows (the aligner's documented per-call limit).

Remaining implementation-time risk: `results[0]`'s exact attribute names
for per-word timestamps (`.time_stamps`, or the result object being itself
indexable — the model card's own example is ambiguous between the two) are
not 100% confirmed; `_extract_word_timestamps` below tries both and falls
back to a single coarse chunk-level segment (no `words` key) if neither
shape matches, so a mismatch degrades gracefully to "less precise speaker
boundaries" instead of crashing.
"""

import logging
from datetime import datetime

import torch

from .base import AlignThenDiarizeASRBuffer

logger = logging.getLogger(__name__)


class Qwen3ASRBuffer(AlignThenDiarizeASRBuffer):
    QWEN_ASR_REPO = "Qwen/Qwen3-ASR-1.7B"
    QWEN_ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
    MODEL_ID = QWEN_ASR_REPO
    MAX_CHUNK_SECONDS = 300  # Qwen3-ForcedAligner's documented per-call limit
    DEFAULT_LANGUAGE = "German"  # this deployment's primary transcript language — no auto-detect wired up yet

    def __init__(self):
        super().__init__()
        self.model_name: str = None

    def load_model(self, model_name: str = QWEN_ASR_REPO, timeout: int = 300):
        """Load Qwen3-ASR (+ bundled forced aligner) with automatic unloading after timeout."""
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        from qwen_asr import Qwen3ASRModel

        super().load_model(timeout=timeout)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        def _load():
            self.model = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=self.device,
                forced_aligner=self.QWEN_ALIGNER_REPO,
                forced_aligner_kwargs=dict(dtype=dtype, device_map=self.device),
            )

        self._load_or_cleanup(_load)
        self.model_name = model_name
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def _transcribe_chunk_with_words(self, chunk, offset: float, align: bool) -> dict:
        results = self.model.transcribe(
            audio=[(chunk, 16000)],
            language=[self.DEFAULT_LANGUAGE],
            return_time_stamps=align,
        )
        r = results[0]
        words = self._extract_word_timestamps(r, offset) if align else []
        if words:
            return {"start": words[0]["start"], "end": words[-1]["end"], "text": r.text, "words": words}
        return {"start": offset, "end": offset + len(chunk) / 16000, "text": r.text}

    @staticmethod
    def _extract_word_timestamps(result, offset_seconds: float) -> list[dict]:
        """Reshape Qwen3-ASR's per-word timestamps into whisperx's expected dict shape.

        See module docstring's "Remaining implementation-time risk" note —
        tries `result.time_stamps` first, then falls back to treating
        `result` itself as the iterable of per-word entries (both forms
        appear in the model card's own examples). Returns `[]` (not a
        crash) if neither shape matches, degrading gracefully to a single
        coarse chunk-level segment in the caller.

        Args:
            result: One element of `model.transcribe(...)`'s return list.
            offset_seconds: Added to every timestamp — chunk N's audio starts
                at `N * MAX_CHUNK_SECONDS` in the original file, but the
                aligner only sees that one chunk and reports timestamps
                relative to it.
        """
        entries = getattr(result, "time_stamps", None)
        if entries is None:
            try:
                entries = list(result)
            except TypeError:
                logger.warning("Qwen3-ASR result has no recognisable per-word timestamp shape.")
                return []

        words = []
        for w in entries:
            try:
                words.append(
                    {
                        "word": w.text,
                        "start": w.start_time + offset_seconds,
                        "end": w.end_time + offset_seconds,
                    }
                )
            except AttributeError as exc:
                logger.warning("Unexpected Qwen3-ASR timestamp entry shape (%s) — skipping word.", exc)
        return words
